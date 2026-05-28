package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/sozercan/vekil/logger"
)

func (h *ProxyHandler) newInferenceUpstreamContext(streaming bool) (context.Context, context.CancelFunc) {
	// Use background context with timeout to avoid cancellation from client
	// disconnects while still preventing goroutine leaks on upstream hangs.
	timeout := upstreamTimeout
	if streaming {
		timeout = h.effectiveStreamingUpstreamTimeout()
	}
	return context.WithTimeout(context.Background(), timeout)
}

func upstreamStatusCode(err error, fallback int) int {
	var upstreamErr *upstreamError
	if errors.As(err, &upstreamErr) {
		return upstreamErr.statusCode
	}
	var providerErr *providerRequestError
	if errors.As(err, &providerErr) {
		return providerErr.statusCode
	}
	return fallback
}

func extractRequestModel(body []byte) string {
	dec := json.NewDecoder(bytes.NewReader(body))
	tok, err := dec.Token()
	if err != nil {
		return ""
	}

	delim, ok := tok.(json.Delim)
	if !ok || delim != '{' {
		return ""
	}

	for dec.More() {
		keyToken, err := dec.Token()
		if err != nil {
			return ""
		}

		key, ok := keyToken.(string)
		if !ok {
			return ""
		}

		if key == "model" {
			var model string
			if err := dec.Decode(&model); err != nil {
				return ""
			}
			return strings.TrimSpace(model)
		}

		if err := skipJSONValue(dec); err != nil {
			return ""
		}
	}

	return ""
}

func skipJSONValue(dec *json.Decoder) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}

	delim, ok := tok.(json.Delim)
	if !ok {
		return nil
	}

	switch delim {
	case '{':
		for dec.More() {
			if _, err := dec.Token(); err != nil {
				return err
			}
			if err := skipJSONValue(dec); err != nil {
				return err
			}
		}
		_, err = dec.Token()
		return err
	case '[':
		for dec.More() {
			if err := skipJSONValue(dec); err != nil {
				return err
			}
		}
		_, err = dec.Token()
		return err
	default:
		return nil
	}
}

func mergeHeaderValues(dst, src http.Header) {
	for key, values := range src {
		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func (h *ProxyHandler) resolveProviderRequest(body []byte, endpoint string) (*providerRuntime, []byte, error) {
	model := extractRequestModel(body)
	if route, ok := h.providerSetup().lookupModelRoute(model); ok && route != nil {
		return nil, nil, &providerRequestError{statusCode: http.StatusBadRequest, err: fmt.Errorf("model route %q must be resolved per retry attempt", model)}
	}
	provider, owner, known := h.resolveProviderModel(model, endpoint)
	if provider == nil {
		return nil, nil, &providerRequestError{statusCode: http.StatusInternalServerError, err: fmt.Errorf("no provider available for endpoint %s", endpoint)}
	}
	if !providerSupportsEndpoint(provider, endpoint) {
		return nil, nil, &providerRequestError{
			statusCode: http.StatusBadRequest,
			err:        fmt.Errorf("provider %q does not support %s", provider.id, endpoint),
		}
	}
	if known && !providerModelSupportsEndpoint(owner, endpoint) {
		return nil, nil, &providerRequestError{
			statusCode: http.StatusBadRequest,
			err:        fmt.Errorf("model %q does not support %s", model, endpoint),
		}
	}

	rewrittenBody, _, err := rewriteRequestModelForProvider(body, owner.upstreamModel)
	if err != nil {
		return nil, nil, &providerRequestError{statusCode: http.StatusBadRequest, err: err}
	}
	return provider, rewrittenBody, nil
}

func providerSupportsEndpoint(provider *providerRuntime, endpoint string) bool {
	if provider == nil {
		return false
	}
	if provider.kind == providerTypeOpenAICodex {
		return supportsEndpoint(openAICodexProviderEndpoints, endpoint)
	}
	return true
}

func (h *ProxyHandler) postJSONEndpoint(ctx context.Context, path string, body []byte) (*http.Response, error) {
	return h.postJSONEndpointWithHeaders(ctx, path, body, nil)
}

func (h *ProxyHandler) postJSONEndpointWithHeaders(ctx context.Context, path string, body []byte, extraHeaders http.Header) (*http.Response, error) {
	if route, ok := h.providerSetup().lookupModelRoute(extractRequestModel(body)); ok {
		return h.postModelRouteJSONEndpointWithHeaders(ctx, path, body, extraHeaders, route)
	}

	provider, rewrittenBody, err := h.resolveProviderRequest(body, path)
	if err != nil {
		return nil, err
	}

	excluded := make(map[string]struct{})
	return h.doWithRetryEndpoint(func(attempt int) (*http.Request, *providerEndpointRuntime, error) {
		endpoint, degraded, err := provider.selectEndpoint(excluded)
		if err != nil {
			return nil, nil, err
		}
		if degraded && h != nil && h.log != nil && endpoint != nil {
			h.log.Info("provider endpoint pool degraded; probing least-recently-failed endpoint", logger.F("provider", provider.id), logger.F("endpoint_id", endpoint.id), logger.F("upstream_endpoint", path), logger.F("attempt", attempt))
		}
		req, err := h.newProviderEndpointJSONRequest(ctx, provider, endpoint, http.MethodPost, path, rewrittenBody, extraHeaders, "")
		if err != nil {
			return nil, nil, err
		}
		if endpoint != nil {
			excluded[endpoint.id] = struct{}{}
		}
		return req, endpoint, nil
	})
}

func (h *ProxyHandler) postModelRouteJSONEndpointWithHeaders(ctx context.Context, path string, body []byte, extraHeaders http.Header, route *modelRouteRuntime) (*http.Response, error) {
	if route == nil {
		return nil, &providerRequestError{statusCode: http.StatusInternalServerError, err: fmt.Errorf("model route is required")}
	}
	if !route.supportsEndpoint(path) {
		return nil, &providerRequestError{statusCode: http.StatusBadRequest, err: fmt.Errorf("model %q does not support %s", route.publicID, path)}
	}

	excludedCandidates := make(map[string]struct{})
	return h.doWithRetryModelRoute(func(attempt int) (*http.Request, *providerEndpointRuntime, *modelRouteCandidateRuntime, error) {
		candidate, degraded, err := route.selectCandidate(excludedCandidates)
		if err != nil {
			return nil, nil, nil, err
		}
		if degraded && h != nil && h.log != nil && candidate != nil {
			h.log.Info("model route pool degraded; probing least-recently-failed candidate", logger.F("route", route.publicID), logger.F("provider", candidate.providerID), logger.F("model", candidate.modelID), logger.F("upstream_endpoint", path), logger.F("attempt", attempt))
		}
		setup := h.providerSetup()
		provider := setup.providerByID(candidate.providerID)
		model, ok := setup.lookupProviderModel(candidate.providerID, candidate.modelID)
		if provider == nil || !ok {
			return nil, nil, nil, &providerRequestError{statusCode: http.StatusInternalServerError, err: fmt.Errorf("model route %q candidate %s is unavailable", route.publicID, candidate.key())}
		}
		if !providerSupportsEndpoint(provider, path) || !providerModelSupportsEndpoint(model, path) {
			return nil, nil, nil, &providerRequestError{statusCode: http.StatusBadRequest, err: fmt.Errorf("model route %q candidate %s does not support %s", route.publicID, candidate.key(), path)}
		}
		rewrittenBody, _, err := rewriteRequestModelForProvider(body, model.upstreamModel)
		if err != nil {
			return nil, nil, nil, &providerRequestError{statusCode: http.StatusBadRequest, err: err}
		}
		upstreamEndpoint, endpointDegraded, err := provider.selectEndpoint(nil)
		if err != nil {
			return nil, nil, nil, err
		}
		if endpointDegraded && h != nil && h.log != nil && upstreamEndpoint != nil {
			h.log.Info("provider endpoint pool degraded; probing least-recently-failed endpoint", logger.F("provider", provider.id), logger.F("endpoint_id", upstreamEndpoint.id), logger.F("upstream_endpoint", path), logger.F("attempt", attempt))
		}
		req, err := h.newProviderEndpointJSONRequest(ctx, provider, upstreamEndpoint, http.MethodPost, path, rewrittenBody, extraHeaders, "")
		if err != nil {
			return nil, nil, nil, err
		}
		excludedCandidates[candidate.key()] = struct{}{}
		return req, upstreamEndpoint, candidate, nil
	})
}

func (h *ProxyHandler) postChatCompletions(ctx context.Context, body []byte) (*http.Response, error) {
	return h.postJSONEndpoint(ctx, "/chat/completions", body)
}

func (h *ProxyHandler) postResponsesWithHeaders(ctx context.Context, body []byte, extraHeaders http.Header) (*http.Response, error) {
	return h.postJSONEndpointWithHeaders(ctx, "/responses", body, extraHeaders)
}

func writeUpstreamResponse(w http.ResponseWriter, resp *http.Response) {
	defer func() { _ = resp.Body.Close() }()
	copyPassthroughHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}
