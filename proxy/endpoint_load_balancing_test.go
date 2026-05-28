package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sozercan/vekil/auth"
	"github.com/sozercan/vekil/logger"
	"gopkg.in/yaml.v3"
)

func endpointIntPtr(v int) *int { return &v }

func testEndpoint(id string, weight int) *providerEndpointRuntime {
	health, err := newProviderEndpointHealthRuntime(ProviderEndpointHealthConfig{ErrorBudget: "2/s", Cooldown: "20ms"})
	if err != nil {
		panic(err)
	}
	return &providerEndpointRuntime{id: id, baseURL: "https://" + id + ".example/openai/v1", apiKey: id + "-key", weight: weight, health: health}
}

func TestErrorBudgetParseAndConfigRoundTrip(t *testing.T) {
	for _, raw := range []string{"1/ms", "2/s", "3/m", "4/h"} {
		budget, err := parseEndpointErrorBudget(raw)
		if err != nil {
			t.Fatalf("parseEndpointErrorBudget(%q) error = %v", raw, err)
		}
		if budget.limit <= 0 || budget.window <= 0 {
			t.Fatalf("parseEndpointErrorBudget(%q) = %+v", raw, budget)
		}
	}

	for _, raw := range []string{"", "0/s", "1/day", "bogus"} {
		if _, err := parseEndpointErrorBudget(raw); err == nil && raw != "" {
			t.Fatalf("parseEndpointErrorBudget(%q) error = nil, want invalid form", raw)
		}
	}

	cfg := ProvidersConfig{Providers: []ProviderConfig{{
		ID:       "azure",
		Type:     "azure-openai",
		Selector: "weighted",
		Endpoints: []ProviderEndpointConfig{{
			ID: "east", BaseURL: "https://east.openai.azure.com/openai/v1", APIKey: "east-key", Weight: endpointIntPtr(2),
			Health: ProviderEndpointHealthConfig{ErrorBudget: "7/m", Cooldown: "45s"},
		}},
		Models: []ProviderModelConfig{{PublicID: "gpt-test", Endpoints: []string{"/responses"}}},
	}}}
	jsonBody, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var jsonRoundTrip ProvidersConfig
	if err := json.Unmarshal(jsonBody, &jsonRoundTrip); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	yamlBody, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}
	var yamlRoundTrip ProvidersConfig
	if err := yaml.Unmarshal(yamlBody, &yamlRoundTrip); err != nil {
		t.Fatalf("yaml.Unmarshal() error = %v", err)
	}
	if jsonRoundTrip.Providers[0].Endpoints[0].Health.ErrorBudget != "7/m" || yamlRoundTrip.Providers[0].Endpoints[0].Health.Cooldown != "45s" {
		t.Fatalf("health config did not round-trip: json=%+v yaml=%+v", jsonRoundTrip.Providers[0].Endpoints[0].Health, yamlRoundTrip.Providers[0].Endpoints[0].Health)
	}
}

func TestLoadProvidersConfigAzureEndpointPool(t *testing.T) {
	providerWeight := 2
	cfg := ProvidersConfig{Providers: []ProviderConfig{{
		ID:        "azure",
		Type:      "azure-openai",
		Default:   true,
		Selector:  "weighted",
		APIKeyEnv: "SHARED_AZURE_KEY",
		Endpoints: []ProviderEndpointConfig{
			{ID: "east", BaseURL: "https://east.openai.azure.com/openai/v1", Weight: &providerWeight, Health: ProviderEndpointHealthConfig{ErrorBudget: "3/m", Cooldown: "10s"}},
			{ID: "west", BaseURL: "https://west.openai.azure.com/openai/v1", APIKey: "west-key"},
		},
		Models: []ProviderModelConfig{{PublicID: "gpt-test", Deployment: "gpt-test-prod", Endpoints: []string{"/responses"}}},
	}}}
	t.Setenv("SHARED_AZURE_KEY", "shared-key")
	handler := &ProxyHandler{copilotURL: "https://copilot.example.com"}
	providers, _, defaultProviderID, err := handler.buildProviders(cfg)
	if err != nil {
		t.Fatalf("buildProviders() error = %v", err)
	}
	provider := providers["azure"]
	if defaultProviderID != "azure" || provider == nil {
		t.Fatalf("default=%q provider=%v, want azure provider", defaultProviderID, provider)
	}
	if provider.baseURL != "https://east.openai.azure.com/openai/v1" || provider.apiKey != "shared-key" {
		t.Fatalf("legacy provider fields = (%q,%q), want first endpoint", provider.baseURL, provider.apiKey)
	}
	if len(provider.endpoints) != 2 || provider.endpoints[0].id != "east" || provider.endpoints[1].apiKey != "west-key" {
		t.Fatalf("provider endpoints = %+v, want east/west with endpoint credentials", provider.endpoints)
	}
	model := provider.staticModels["gpt-test"]
	if !reflect.DeepEqual(model.supportedEndpoints, []string{"/responses"}) {
		t.Fatalf("model route allowlist = %v, want [/responses]", model.supportedEndpoints)
	}
}

func TestLoadProvidersConfigRejectsInvalidEndpointForms(t *testing.T) {
	tests := []struct {
		name string
		cfg  ProvidersConfig
		want string
	}{
		{name: "zero weight", cfg: ProvidersConfig{Providers: []ProviderConfig{{ID: "azure", Type: "azure-openai", Endpoints: []ProviderEndpointConfig{{ID: "east", BaseURL: "https://east.openai.azure.com/openai/v1", APIKey: "key", Weight: endpointIntPtr(0)}}, Models: []ProviderModelConfig{{PublicID: "gpt"}}}}}, want: "weight must be greater than 0"},
		{name: "bad health", cfg: ProvidersConfig{Providers: []ProviderConfig{{ID: "azure", Type: "azure-openai", Endpoints: []ProviderEndpointConfig{{ID: "east", BaseURL: "https://east.openai.azure.com/openai/v1", APIKey: "key", Health: ProviderEndpointHealthConfig{ErrorBudget: "1/day"}}}, Models: []ProviderModelConfig{{PublicID: "gpt"}}}}}, want: "error_budget"},
		{name: "copilot endpoints deferred", cfg: ProvidersConfig{Providers: []ProviderConfig{{ID: "copilot", Type: "copilot", Endpoints: []ProviderEndpointConfig{{ID: "a", BaseURL: "https://example.com", APIKey: "key"}}}}}, want: "Copilot multi-account rotation is deferred"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := &ProxyHandler{copilotURL: "https://copilot.example.com"}
			_, _, _, err := handler.buildProviders(tt.cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("buildProviders() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestSelectorStrategies(t *testing.T) {
	tests := []struct {
		name     string
		selector providerEndpointSelectorKind
		prepare  func([]*providerEndpointRuntime)
		want     []string
	}{
		{name: "round robin", selector: providerEndpointSelectorRoundRobin, want: []string{"east", "west", "east", "west"}},
		{name: "weighted", selector: providerEndpointSelectorWeighted, want: []string{"east", "east", "west", "east", "east", "west"}},
		{name: "least latency", selector: providerEndpointSelectorLeastLatency, prepare: func(endpoints []*providerEndpointRuntime) {
			endpoints[0].recordSuccess(50 * time.Millisecond)
			endpoints[1].recordSuccess(10 * time.Millisecond)
		}, want: []string{"west", "west", "west"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &providerRuntime{id: "azure", selector: tt.selector, endpoints: []*providerEndpointRuntime{testEndpoint("east", 2), testEndpoint("west", 1)}}
			if tt.prepare != nil {
				tt.prepare(provider.endpoints)
			}
			got := make([]string, 0, len(tt.want))
			for range tt.want {
				endpoint, degraded, err := provider.selectEndpoint(nil)
				if err != nil {
					t.Fatalf("selectEndpoint() error = %v", err)
				}
				if degraded {
					t.Fatal("selectEndpoint() degraded = true, want false")
				}
				got = append(got, endpoint.id)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("selection = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEndpointHealthQuarantineCooldownAndDegradedProbe(t *testing.T) {
	provider := &providerRuntime{id: "azure", selector: providerEndpointSelectorRoundRobin, endpoints: []*providerEndpointRuntime{testEndpoint("east", 1), testEndpoint("west", 1)}}
	now := time.Now()
	provider.endpoints[0].recordFailure(now)
	provider.endpoints[0].recordFailure(now.Add(time.Millisecond))
	endpoint, degraded, err := provider.selectEndpoint(nil)
	if err != nil {
		t.Fatalf("selectEndpoint() error = %v", err)
	}
	if degraded || endpoint.id != "west" {
		t.Fatalf("after east quarantine got endpoint=%s degraded=%v, want west false", endpoint.id, degraded)
	}
	provider.endpoints[1].recordFailure(now)
	provider.endpoints[1].recordFailure(now.Add(time.Millisecond))
	endpoint, degraded, err = provider.selectEndpoint(nil)
	if err != nil {
		t.Fatalf("selectEndpoint() all unhealthy error = %v", err)
	}
	if !degraded || endpoint == nil {
		t.Fatalf("all unhealthy degraded=%v endpoint=%v, want degraded probe", degraded, endpoint)
	}
	time.Sleep(25 * time.Millisecond)
	endpoint, degraded, err = provider.selectEndpoint(nil)
	if err != nil {
		t.Fatalf("selectEndpoint() after cooldown error = %v", err)
	}
	if degraded {
		t.Fatal("selectEndpoint() after cooldown degraded = true, want false")
	}
}

func TestProviderRequestRetriesAlternateHealthyEndpoint(t *testing.T) {
	var eastCalls atomic.Int32
	east := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		eastCalls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer east.Close()

	var westCalls atomic.Int32
	west := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		westCalls.Add(1)
		if got := r.Header.Get("api-key"); got != "west-key" {
			t.Fatalf("api-key = %q, want west-key", got)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer west.Close()

	h, err := NewProxyHandler(auth.NewTestAuthenticator("tok"), logger.New(logger.LevelError), WithProvidersConfig(ProvidersConfig{Providers: []ProviderConfig{{
		ID:       "azure",
		Type:     "azure-openai",
		Default:  true,
		Selector: "round_robin",
		Endpoints: []ProviderEndpointConfig{
			{ID: "east", BaseURL: east.URL + "/openai/v1", APIKey: "east-key", Health: ProviderEndpointHealthConfig{ErrorBudget: "1/s", Cooldown: "1s"}},
			{ID: "west", BaseURL: west.URL + "/openai/v1", APIKey: "west-key", Health: ProviderEndpointHealthConfig{ErrorBudget: "1/s", Cooldown: "1s"}},
		},
		Models: []ProviderModelConfig{{PublicID: "gpt-test", Endpoints: []string{"/responses"}}},
	}}}))
	if err != nil {
		t.Fatalf("NewProxyHandler() error = %v", err)
	}
	h.maxRetries = 2
	h.retryBaseDelay = time.Millisecond

	resp, err := h.postResponsesWithHeaders(context.Background(), []byte(`{"model":"gpt-test","input":"hello"}`), nil)
	if err != nil {
		t.Fatalf("postResponsesWithHeaders() error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK || eastCalls.Load() != 1 || westCalls.Load() != 1 {
		t.Fatalf("status=%d eastCalls=%d westCalls=%d, want 200/1/1", resp.StatusCode, eastCalls.Load(), westCalls.Load())
	}
}

func TestEndpointAwareLogsDoNotLeakSecrets(t *testing.T) {
	readEnd, writeEnd, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe() error = %v", err)
	}
	originalStderr := os.Stderr
	os.Stderr = writeEnd
	t.Cleanup(func() { os.Stderr = originalStderr })

	provider := &providerRuntime{id: "azure", kind: providerTypeAzureOpenAI, endpoints: []*providerEndpointRuntime{{id: "east", baseURL: "https://example.openai.azure.com/openai/v1", apiKey: "secret-key", weight: 1, health: mustDefaultProviderEndpointHealth()}}}
	h := &ProxyHandler{log: logger.New(logger.LevelDebug)}
	_, err = h.newProviderEndpointJSONRequest(context.Background(), provider, provider.endpoints[0], http.MethodPost, "/responses", []byte(`{}`), nil, "")
	_ = writeEnd.Close()
	if err != nil {
		t.Fatalf("newProviderEndpointJSONRequest() error = %v", err)
	}
	buf := make([]byte, 4096)
	n, _ := readEnd.Read(buf)
	logOutput := string(buf[:n])
	if !strings.Contains(logOutput, "east") || strings.Contains(logOutput, "secret-key") || strings.Contains(logOutput, "api-key") {
		t.Fatalf("endpoint log leaked secret or omitted endpoint id: %s", logOutput)
	}
}
