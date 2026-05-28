package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"
	"time"
)

func TestModelRouteConfigValidationAndCatalog(t *testing.T) {
	t.Parallel()

	h := &ProxyHandler{copilotURL: "https://copilot.example.com", client: http.DefaultClient}
	cfg := ProvidersConfig{
		Providers: []ProviderConfig{
			{
				ID: "east", Type: "azure-openai", Default: true,
				BaseURL: "https://east.example.com/openai/v1", APIKey: "east-key",
				Models: []ProviderModelConfig{{PublicID: "east-gpt", Deployment: "east-dep", Endpoints: []string{"/responses"}}},
			},
			{
				ID: "west", Type: "azure-openai",
				BaseURL: "https://west.example.com/openai/v1", APIKey: "west-key",
				Models: []ProviderModelConfig{{PublicID: "west-gpt", Deployment: "west-dep", Endpoints: []string{"/responses", "/chat/completions"}}},
			},
		},
		ModelRoutes: []ModelRouteConfig{{
			PublicID: "balanced-gpt",
			Selector: "weighted",
			Candidates: []ModelRouteCandidateConfig{
				{Provider: "east", Model: "east-gpt"},
				{Provider: "west", Model: "west-gpt"},
			},
		}},
	}

	setup, err := h.buildConfiguredProviderSetup(context.Background(), cfg)
	if err != nil {
		t.Fatalf("buildConfiguredProviderSetup() error = %v", err)
	}
	route, ok := setup.lookupModelRoute("balanced-gpt")
	if !ok {
		t.Fatal("expected balanced-gpt route")
	}
	if !reflect.DeepEqual(route.endpoints, []string{"/responses"}) {
		t.Fatalf("route endpoints = %v, want [/responses]", route.endpoints)
	}
	model, ok := setup.lookupModel("balanced-gpt")
	if !ok || model.providerID != "route:balanced-gpt" {
		t.Fatalf("catalog route model = %#v, %v; want route owner", model, ok)
	}

	entry, _, err := (&ProxyHandler{client: http.DefaultClient, providersState: setup}).buildMergedModelsEntry(context.Background(), "", "")
	if err != nil {
		t.Fatalf("buildMergedModelsEntry() error = %v", err)
	}
	var payload struct {
		Data []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(entry.body, &payload); err != nil {
		t.Fatalf("Unmarshal catalog: %v", err)
	}
	found := false
	for _, model := range payload.Data {
		if model.ID == "balanced-gpt" && model.OwnedBy == "route:balanced-gpt" {
			found = true
		}
	}
	if !found {
		t.Fatalf("catalog missing synthetic route model: %s", string(entry.body))
	}
}

func TestModelRouteRejectsCollisionsAndIncompatibleEndpoints(t *testing.T) {
	t.Parallel()

	base := ProvidersConfig{Providers: []ProviderConfig{{
		ID: "azure", Type: "azure-openai", Default: true,
		BaseURL: "https://azure.example.com/openai/v1", APIKey: "key",
		Models: []ProviderModelConfig{{PublicID: "gpt", Deployment: "dep", Endpoints: []string{"/responses"}}},
	}}}
	h := &ProxyHandler{copilotURL: "https://copilot.example.com", client: http.DefaultClient}

	collision := base
	collision.ModelRoutes = []ModelRouteConfig{{PublicID: "gpt", Candidates: []ModelRouteCandidateConfig{{Provider: "azure", Model: "gpt"}}}}
	if _, err := h.buildConfiguredProviderSetup(context.Background(), collision); err == nil {
		t.Fatal("buildConfiguredProviderSetup() error = nil, want route/provider public_id collision")
	}

	incompatible := base
	incompatible.ModelRoutes = []ModelRouteConfig{{PublicID: "balanced", Endpoints: []string{"/chat/completions"}, Candidates: []ModelRouteCandidateConfig{{Provider: "azure", Model: "gpt"}}}}
	if _, err := h.buildConfiguredProviderSetup(context.Background(), incompatible); err == nil {
		t.Fatal("buildConfiguredProviderSetup() error = nil, want endpoint incompatibility")
	}
}

func TestModelRouteAllowsExplicitDuplicateCandidateModelIDs(t *testing.T) {
	t.Parallel()

	h := &ProxyHandler{copilotURL: "https://copilot.example.com", client: http.DefaultClient}
	setup, err := h.buildConfiguredProviderSetup(context.Background(), ProvidersConfig{
		Providers: []ProviderConfig{
			{ID: "a", Type: "azure-openai", Default: true, BaseURL: "https://a.example.com/openai/v1", APIKey: "a-key", Models: []ProviderModelConfig{{PublicID: "gpt", Deployment: "dep-a", Endpoints: []string{"/responses"}}}},
			{ID: "b", Type: "azure-openai", BaseURL: "https://b.example.com/openai/v1", APIKey: "b-key", Models: []ProviderModelConfig{{PublicID: "gpt", Deployment: "dep-b", Endpoints: []string{"/responses"}}}},
		},
		ModelRoutes: []ModelRouteConfig{{PublicID: "balanced", Candidates: []ModelRouteCandidateConfig{{Provider: "a", Model: "gpt"}, {Provider: "b", Model: "gpt"}}}},
	})
	if err != nil {
		t.Fatalf("buildConfiguredProviderSetup() error = %v", err)
	}
	if _, ok := setup.lookupModel("gpt"); ok {
		t.Fatal("duplicate candidate model gpt was globally exposed; want route-only exposure")
	}
	if _, ok := setup.lookupModelRoute("balanced"); !ok {
		t.Fatal("expected balanced route")
	}
}

func TestModelRouteRoundRobinRewritesPerRequest(t *testing.T) {
	t.Parallel()

	var bodies []string
	handler, closeServers := newModelRouteTestHandler(t, []func(http.ResponseWriter, *http.Request){
		func(w http.ResponseWriter, r *http.Request) {
			bodies = append(bodies, readRequestModelForTest(t, r))
			_, _ = w.Write([]byte(`{"ok":true}`))
		},
		func(w http.ResponseWriter, r *http.Request) {
			bodies = append(bodies, readRequestModelForTest(t, r))
			_, _ = w.Write([]byte(`{"ok":true}`))
		},
	})
	defer closeServers()

	for range 2 {
		resp, err := handler.postResponsesWithHeaders(context.Background(), []byte(`{"model":"balanced","input":"hi"}`), nil)
		if err != nil {
			t.Fatalf("postResponsesWithHeaders() error = %v", err)
		}
		drainAndClose(resp.Body)
	}
	if !reflect.DeepEqual(bodies, []string{"dep-a", "dep-b"}) {
		t.Fatalf("upstream models = %v, want [dep-a dep-b]", bodies)
	}
}

func TestModelRouteRetryFailoverAcrossCandidates(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	var models []string
	handler, closeServers := newModelRouteTestHandler(t, []func(http.ResponseWriter, *http.Request){
		func(w http.ResponseWriter, r *http.Request) {
			attempts.Add(1)
			models = append(models, readRequestModelForTest(t, r))
			w.WriteHeader(http.StatusTooManyRequests)
		},
		func(w http.ResponseWriter, r *http.Request) {
			attempts.Add(1)
			models = append(models, readRequestModelForTest(t, r))
			_, _ = w.Write([]byte(`{"ok":true}`))
		},
	})
	defer closeServers()

	resp, err := handler.postResponsesWithHeaders(context.Background(), []byte(`{"model":"balanced","input":"hi"}`), nil)
	if err != nil {
		t.Fatalf("postResponsesWithHeaders() error = %v", err)
	}
	drainAndClose(resp.Body)
	if attempts.Load() != 2 {
		t.Fatalf("attempts = %d, want 2", attempts.Load())
	}
	if !reflect.DeepEqual(models, []string{"dep-a", "dep-b"}) {
		t.Fatalf("retry models = %v, want [dep-a dep-b]", models)
	}
}

func newModelRouteTestHandler(t *testing.T, handlers []func(http.ResponseWriter, *http.Request)) (*ProxyHandler, func()) {
	t.Helper()
	servers := make([]*httptest.Server, 0, len(handlers))
	providers := make([]ProviderConfig, 0, len(handlers))
	candidates := make([]ModelRouteCandidateConfig, 0, len(handlers))
	for i, handle := range handlers {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/openai/v1/responses" {
				t.Fatalf("path = %q, want /openai/v1/responses", r.URL.Path)
			}
			handle(w, r)
		}))
		servers = append(servers, server)
		id := string(rune('a' + i))
		providers = append(providers, ProviderConfig{
			ID: "provider-" + id, Type: "azure-openai", Default: i == 0,
			BaseURL: server.URL + "/openai/v1", APIKey: "key-" + id,
			Models: []ProviderModelConfig{{PublicID: "model-" + id, Deployment: "dep-" + id, Endpoints: []string{"/responses"}}},
		})
		candidates = append(candidates, ModelRouteCandidateConfig{Provider: "provider-" + id, Model: "model-" + id})
	}

	h := &ProxyHandler{client: servers[0].Client(), copilotURL: "https://copilot.example.com", maxRetries: 2, retryBaseDelay: time.Millisecond}
	setup, err := h.buildConfiguredProviderSetup(context.Background(), ProvidersConfig{Providers: providers, ModelRoutes: []ModelRouteConfig{{PublicID: "balanced", Selector: "round_robin", Candidates: candidates}}})
	if err != nil {
		t.Fatalf("buildConfiguredProviderSetup() error = %v", err)
	}
	h.providersState = setup
	return h, func() {
		for _, server := range servers {
			server.Close()
		}
	}
}

func readRequestModelForTest(t *testing.T, r *http.Request) string {
	t.Helper()
	defer r.Body.Close()
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		t.Fatalf("Decode request: %v", err)
	}
	return payload.Model
}
