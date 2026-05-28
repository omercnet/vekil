package proxy

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

type modelRouteRuntime struct {
	publicID      string
	endpoints     []string
	selector      providerEndpointSelectorKind
	candidates    []*modelRouteCandidateRuntime
	selectorState providerEndpointSelectorState
}

type modelRouteCandidateRuntime struct {
	providerID string
	modelID    string
	weight     int
	health     *providerEndpointHealthRuntime
}

type modelRouteAttempt struct {
	route     *modelRouteRuntime
	candidate *modelRouteCandidateRuntime
	provider  *providerRuntime
	model     providerModel
	body      []byte
}

var modelRouteMu sync.Mutex

func (ps *providerSetup) buildModelRoutes(configs []ModelRouteConfig) error {
	if len(configs) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(configs))
	for i, cfg := range configs {
		fieldPath := fmt.Sprintf("model_routes[%d]", i)
		route, err := ps.buildModelRoute(fieldPath, cfg)
		if err != nil {
			return err
		}
		if _, exists := seen[route.publicID]; exists {
			return fmt.Errorf("duplicate model route public_id %q", route.publicID)
		}
		seen[route.publicID] = struct{}{}
		if ps.providerOwnedModelIDExists(route.publicID) {
			return fmt.Errorf("model route %q collides with provider-owned model id", route.publicID)
		}
		if ps.modelRoutes == nil {
			ps.modelRoutes = make(map[string]*modelRouteRuntime)
		}
		ps.modelRoutes[route.publicID] = route
		ps.modelRouteOrder = append(ps.modelRouteOrder, route.publicID)
		ps.models[route.publicID] = route.syntheticModel()
	}
	return nil
}

func (ps *providerSetup) buildModelRoute(fieldPath string, cfg ModelRouteConfig) (*modelRouteRuntime, error) {
	publicID := strings.TrimSpace(cfg.PublicID)
	if publicID == "" {
		return nil, fmt.Errorf("%s.public_id is required", fieldPath)
	}
	if len(cfg.Candidates) == 0 {
		return nil, fmt.Errorf("%s.candidates must include at least one candidate", fieldPath)
	}
	selector, err := parseProviderEndpointSelector(cfg.Selector, len(cfg.Candidates))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", fieldPath, err)
	}

	candidates := make([]*modelRouteCandidateRuntime, 0, len(cfg.Candidates))
	seen := make(map[string]struct{}, len(cfg.Candidates))
	for i, candidateCfg := range cfg.Candidates {
		candidate, err := ps.buildModelRouteCandidate(fmt.Sprintf("%s.candidates[%d]", fieldPath, i), candidateCfg)
		if err != nil {
			return nil, err
		}
		key := candidate.key()
		if _, exists := seen[key]; exists {
			return nil, fmt.Errorf("%s.candidates[%d] duplicates provider/model %q", fieldPath, i, key)
		}
		seen[key] = struct{}{}
		candidates = append(candidates, candidate)
	}

	endpoints, err := ps.modelRouteEndpoints(fieldPath, cfg.Endpoints, candidates)
	if err != nil {
		return nil, err
	}
	return &modelRouteRuntime{publicID: publicID, endpoints: endpoints, selector: selector, candidates: candidates}, nil
}

func (ps *providerSetup) buildModelRouteCandidate(fieldPath string, cfg ModelRouteCandidateConfig) (*modelRouteCandidateRuntime, error) {
	providerID := strings.TrimSpace(cfg.Provider)
	modelID := strings.TrimSpace(cfg.Model)
	if providerID == "" {
		return nil, fmt.Errorf("%s.provider is required", fieldPath)
	}
	if modelID == "" {
		return nil, fmt.Errorf("%s.model is required", fieldPath)
	}
	if ps.providerByID(providerID) == nil {
		return nil, fmt.Errorf("%s.provider %q is not configured", fieldPath, providerID)
	}
	if _, ok := ps.lookupProviderModel(providerID, modelID); !ok {
		return nil, fmt.Errorf("%s.model %q is not exposed by provider %q", fieldPath, modelID, providerID)
	}
	weight := 1
	if cfg.Weight != nil {
		weight = *cfg.Weight
	}
	if weight <= 0 {
		return nil, fmt.Errorf("%s.weight must be greater than 0", fieldPath)
	}
	health, err := newProviderEndpointHealthRuntime(cfg.Health)
	if err != nil {
		return nil, fmt.Errorf("%s.health: %w", fieldPath, err)
	}
	return &modelRouteCandidateRuntime{providerID: providerID, modelID: modelID, weight: weight, health: health}, nil
}

func (ps *providerSetup) modelRouteEndpoints(fieldPath string, explicit []string, candidates []*modelRouteCandidateRuntime) ([]string, error) {
	if len(explicit) > 0 {
		endpoints := normalizeProviderEndpoints(explicit)
		for _, endpoint := range endpoints {
			for _, candidate := range candidates {
				provider := ps.providerByID(candidate.providerID)
				model, _ := ps.lookupProviderModel(candidate.providerID, candidate.modelID)
				if !providerSupportsEndpoint(provider, endpoint) || !providerModelSupportsEndpoint(model, endpoint) {
					return nil, fmt.Errorf("%s.endpoints contains %s but candidate %s/%s does not support it", fieldPath, endpoint, candidate.providerID, candidate.modelID)
				}
			}
		}
		return endpoints, nil
	}

	var intersection []string
	for i, candidate := range candidates {
		provider := ps.providerByID(candidate.providerID)
		model, _ := ps.lookupProviderModel(candidate.providerID, candidate.modelID)
		candidateEndpoints := effectiveCandidateEndpoints(provider, model)
		if i == 0 {
			intersection = append([]string(nil), candidateEndpoints...)
			continue
		}
		intersection = intersectEndpoints(intersection, candidateEndpoints)
	}
	if len(intersection) == 0 {
		return nil, fmt.Errorf("%s has no common endpoint supported by all candidates", fieldPath)
	}
	return intersection, nil
}

func effectiveCandidateEndpoints(provider *providerRuntime, model providerModel) []string {
	endpoints := model.supportedEndpoints
	if len(endpoints) == 0 {
		endpoints = defaultStaticProviderEndpoints
	}
	if provider != nil && provider.kind == providerTypeOpenAICodex {
		endpoints = intersectEndpoints(endpoints, openAICodexProviderEndpoints)
	}
	return endpoints
}

func intersectEndpoints(left, right []string) []string {
	rightSet := make(map[string]struct{}, len(right))
	for _, endpoint := range right {
		rightSet[endpoint] = struct{}{}
	}
	result := make([]string, 0, len(left))
	for _, endpoint := range left {
		if _, ok := rightSet[endpoint]; ok {
			result = append(result, endpoint)
		}
	}
	return result
}

func (r *modelRouteRuntime) syntheticModel() providerModel {
	raw, _ := synthesizeRouteModelRaw(r.publicID, r.endpoints)
	return providerModel{publicID: r.publicID, upstreamModel: r.publicID, providerID: "route:" + r.publicID, supportedEndpoints: append([]string(nil), r.endpoints...), raw: raw}
}

func synthesizeRouteModelRaw(publicID string, endpoints []string) (json.RawMessage, error) {
	return json.Marshal(map[string]interface{}{
		"id":                  publicID,
		"object":              "model",
		"created":             0,
		"owned_by":            "route:" + publicID,
		"name":                publicID,
		"supported_endpoints": append([]string(nil), endpoints...),
	})
}

func (r *modelRouteRuntime) supportsEndpoint(endpoint string) bool {
	return supportsEndpoint(r.endpoints, endpoint)
}

func (r *modelRouteRuntime) selectCandidate(exclude map[string]struct{}) (*modelRouteCandidateRuntime, bool, error) {
	if r == nil {
		return nil, false, fmt.Errorf("model route is required")
	}
	now := time.Now()
	candidates := r.candidateOptions(now, exclude)
	if len(candidates) == 0 && len(exclude) > 0 {
		candidates = r.candidateOptions(now, nil)
	}
	if len(candidates) == 0 {
		return r.leastRecentlyFailedCandidate(), true, nil
	}
	switch r.selector {
	case providerEndpointSelectorWeighted:
		return r.selectWeightedCandidate(candidates), false, nil
	case providerEndpointSelectorLeastLatency:
		return r.selectLeastLatencyCandidate(candidates), false, nil
	default:
		return r.selectRoundRobinCandidate(candidates), false, nil
	}
}

func (r *modelRouteRuntime) candidateOptions(now time.Time, exclude map[string]struct{}) []*modelRouteCandidateRuntime {
	options := make([]*modelRouteCandidateRuntime, 0, len(r.candidates))
	for _, candidate := range r.candidates {
		if exclude != nil {
			if _, ok := exclude[candidate.key()]; ok {
				continue
			}
		}
		if candidate.isAvailable(now) {
			options = append(options, candidate)
		}
	}
	return options
}

func (r *modelRouteRuntime) selectRoundRobinCandidate(candidates []*modelRouteCandidateRuntime) *modelRouteCandidateRuntime {
	modelRouteMu.Lock()
	defer modelRouteMu.Unlock()
	idx := r.selectorState.roundRobinIndex % len(candidates)
	r.selectorState.roundRobinIndex++
	return candidates[idx]
}

func (r *modelRouteRuntime) selectWeightedCandidate(candidates []*modelRouteCandidateRuntime) *modelRouteCandidateRuntime {
	modelRouteMu.Lock()
	defer modelRouteMu.Unlock()
	for range len(candidates) * 2 {
		candidate := candidates[r.selectorState.weightedIndex%len(candidates)]
		if r.selectorState.weightedOffset < candidate.weight {
			r.selectorState.weightedOffset++
			if r.selectorState.weightedOffset >= candidate.weight {
				r.selectorState.weightedOffset = 0
				r.selectorState.weightedIndex++
			}
			return candidate
		}
		r.selectorState.weightedOffset = 0
		r.selectorState.weightedIndex++
	}
	return candidates[0]
}

func (r *modelRouteRuntime) selectLeastLatencyCandidate(candidates []*modelRouteCandidateRuntime) *modelRouteCandidateRuntime {
	best := candidates[0]
	bestLatency, bestHasLatency := best.latency()
	for _, candidate := range candidates[1:] {
		latency, hasLatency := candidate.latency()
		switch {
		case hasLatency && !bestHasLatency:
			best, bestLatency, bestHasLatency = candidate, latency, hasLatency
		case hasLatency == bestHasLatency && latency < bestLatency:
			best, bestLatency, bestHasLatency = candidate, latency, hasLatency
		}
	}
	return best
}

func (r *modelRouteRuntime) leastRecentlyFailedCandidate() *modelRouteCandidateRuntime {
	best := r.candidates[0]
	bestFailure := best.lastFailure()
	for _, candidate := range r.candidates[1:] {
		failure := candidate.lastFailure()
		if bestFailure.IsZero() || (!failure.IsZero() && failure.Before(bestFailure)) {
			best = candidate
			bestFailure = failure
		}
	}
	return best
}

func (c *modelRouteCandidateRuntime) key() string { return c.providerID + "/" + c.modelID }

func (c *modelRouteCandidateRuntime) isAvailable(now time.Time) bool {
	if c == nil {
		return false
	}
	return (&providerEndpointRuntime{health: c.health}).isAvailable(now)
}

func (c *modelRouteCandidateRuntime) recordFailure(now time.Time) {
	if c == nil {
		return
	}
	(&providerEndpointRuntime{health: c.health}).recordFailure(now)
}

func (c *modelRouteCandidateRuntime) recordSuccess(latency time.Duration) {
	if c == nil {
		return
	}
	(&providerEndpointRuntime{health: c.health}).recordSuccess(latency)
}

func (c *modelRouteCandidateRuntime) latency() (time.Duration, bool) {
	if c == nil {
		return 0, false
	}
	return (&providerEndpointRuntime{health: c.health}).latency()
}

func (c *modelRouteCandidateRuntime) lastFailure() time.Time {
	if c == nil {
		return time.Time{}
	}
	return (&providerEndpointRuntime{health: c.health}).lastFailure()
}
