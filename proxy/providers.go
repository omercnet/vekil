package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/sozercan/vekil/logger"
	"gopkg.in/yaml.v3"
)

type providerType string

const (
	providerTypeCopilot     providerType = "copilot"
	providerTypeAzureOpenAI providerType = "azure-openai"
	providerTypeOpenAICodex providerType = "openai-codex"
)

var defaultStaticProviderEndpoints = []string{"/chat/completions", "/responses"}
var openAICodexProviderEndpoints = []string{"/responses"}

// ProvidersConfig configures optional non-Copilot upstream providers.
// When empty, the proxy keeps its legacy zero-config Copilot behavior.
type ProvidersConfig struct {
	Providers      []ProviderConfig     `json:"providers" yaml:"providers"`
	ModelRoutes    []ModelRouteConfig   `json:"model_routes,omitempty" yaml:"model_routes,omitempty"`
	ToolOptimizers ToolOptimizersConfig `json:"tool_optimizers,omitempty" yaml:"tool_optimizers,omitempty"`
}

// ModelRouteConfig exposes one synthetic public model backed by provider/model candidates.
type ModelRouteConfig struct {
	PublicID   string                      `json:"public_id" yaml:"public_id"`
	Endpoints  []string                    `json:"endpoints,omitempty" yaml:"endpoints,omitempty"`
	Selector   string                      `json:"selector,omitempty" yaml:"selector,omitempty"`
	Candidates []ModelRouteCandidateConfig `json:"candidates" yaml:"candidates"`
}

// ModelRouteCandidateConfig references a model owned by one configured provider.
type ModelRouteCandidateConfig struct {
	Provider string                       `json:"provider" yaml:"provider"`
	Model    string                       `json:"model" yaml:"model"`
	Weight   *int                         `json:"weight,omitempty" yaml:"weight,omitempty"`
	Health   ProviderEndpointHealthConfig `json:"health,omitempty" yaml:"health,omitempty"`
}

// ProviderConfig configures one upstream provider instance.
type ProviderConfig struct {
	ID            string                      `json:"id" yaml:"id"`
	Type          string                      `json:"type" yaml:"type"`
	Default       bool                        `json:"default,omitempty" yaml:"default,omitempty"`
	Selector      string                      `json:"selector,omitempty" yaml:"selector,omitempty"`
	IncludeModels []string                    `json:"include_models,omitempty" yaml:"include_models,omitempty"`
	ExcludeModels []string                    `json:"exclude_models,omitempty" yaml:"exclude_models,omitempty"`
	BaseURL       string                      `json:"base_url,omitempty" yaml:"base_url,omitempty"`
	APIKey        string                      `json:"api_key,omitempty" yaml:"api_key,omitempty"`
	APIKeyEnv     string                      `json:"api_key_env,omitempty" yaml:"api_key_env,omitempty"`
	APIVersion    string                      `json:"api_version,omitempty" yaml:"api_version,omitempty"`
	Endpoints     []ProviderEndpointConfig    `json:"endpoints,omitempty" yaml:"endpoints,omitempty"`
	Headers       CopilotHeaderProfilesConfig `json:"headers,omitempty" yaml:"headers,omitempty"`
	Models        []ProviderModelConfig       `json:"models,omitempty" yaml:"models,omitempty"`
}

// ProviderEndpointConfig configures one upstream endpoint inside a provider-local pool.
type ProviderEndpointConfig struct {
	ID         string                       `json:"id" yaml:"id"`
	BaseURL    string                       `json:"base_url" yaml:"base_url"`
	APIKey     string                       `json:"api_key,omitempty" yaml:"api_key,omitempty"`
	APIKeyEnv  string                       `json:"api_key_env,omitempty" yaml:"api_key_env,omitempty"`
	APIVersion string                       `json:"api_version,omitempty" yaml:"api_version,omitempty"`
	Weight     *int                         `json:"weight,omitempty" yaml:"weight,omitempty"`
	Health     ProviderEndpointHealthConfig `json:"health,omitempty" yaml:"health,omitempty"`
}

// ProviderEndpointHealthConfig controls endpoint quarantine behavior.
type ProviderEndpointHealthConfig struct {
	ErrorBudget string `json:"error_budget,omitempty" yaml:"error_budget,omitempty"`
	Cooldown    string `json:"cooldown,omitempty" yaml:"cooldown,omitempty"`
}

// ProviderModelConfig maps a public model ID exposed by this proxy to the
// upstream model or deployment name used by the provider.
type ProviderModelConfig struct {
	PublicID            string   `json:"public_id" yaml:"public_id"`
	Deployment          string   `json:"deployment,omitempty" yaml:"deployment,omitempty"`
	Name                string   `json:"name,omitempty" yaml:"name,omitempty"`
	Endpoints           []string `json:"endpoints,omitempty" yaml:"endpoints,omitempty"`
	ModelPickerEnabled  *bool    `json:"model_picker_enabled,omitempty" yaml:"model_picker_enabled,omitempty"`
	ModelPickerCategory string   `json:"model_picker_category,omitempty" yaml:"model_picker_category,omitempty"`
	ReasoningEffort     []string `json:"reasoning_effort,omitempty" yaml:"reasoning_effort,omitempty"`
	Vision              *bool    `json:"vision,omitempty" yaml:"vision,omitempty"`
	ParallelToolCalls   *bool    `json:"parallel_tool_calls,omitempty" yaml:"parallel_tool_calls,omitempty"`
	ContextWindow       *int64   `json:"context_window,omitempty" yaml:"context_window,omitempty"`
}

type providerRuntime struct {
	id             string
	kind           providerType
	isDefault      bool
	baseURL        string
	apiKey         string
	apiVersion     string
	selector       providerEndpointSelectorKind
	endpoints      []*providerEndpointRuntime
	selectorState  providerEndpointSelectorState
	includeModels  map[string]struct{}
	excludeModels  map[string]struct{}
	staticModels   map[string]providerModel
	staticConfigs  map[string]ProviderModelConfig
	staticOrder    []string
	codexAuth      *openAICodexAuth
	headerProfiles CopilotHeaderProfilesConfig
}

type providerModel struct {
	publicID           string
	upstreamModel      string
	providerID         string
	supportedEndpoints []string
	disabled           bool
	raw                json.RawMessage
}

type providerSetup struct {
	providers          map[string]*providerRuntime
	providerOrder      []string
	defaultProviderID  string
	modelsMu           sync.RWMutex
	models             map[string]providerModel
	providerModels     map[string]map[string]providerModel
	modelRoutes        map[string]*modelRouteRuntime
	modelRouteOrder    []string
	routeCandidateRefs map[string]struct{}
	ambiguousModelIDs  map[string]struct{}
	hasConfiguredState bool
}

type providerModelsFetchResult struct {
	models      []providerModel
	etag        string
	notModified bool
}

type openAICodexReasoningPreset struct {
	Effort string `json:"effort"`
}

type openAICodexModelPayload struct {
	Slug                        string                       `json:"slug"`
	DisplayName                 string                       `json:"display_name"`
	Description                 string                       `json:"description"`
	Visibility                  string                       `json:"visibility"`
	SupportedInAPI              bool                         `json:"supported_in_api"`
	Priority                    int                          `json:"priority"`
	SupportedReasoningLevels    []openAICodexReasoningPreset `json:"supported_reasoning_levels"`
	SupportsParallelToolCalls   bool                         `json:"supports_parallel_tool_calls"`
	SupportsImageDetailOriginal bool                         `json:"supports_image_detail_original"`
	SupportsReasoningSummaries  bool                         `json:"supports_reasoning_summaries"`
	SupportVerbosity            bool                         `json:"support_verbosity"`
	ContextWindow               *int64                       `json:"context_window,omitempty"`
	MaxContextWindow            *int64                       `json:"max_context_window,omitempty"`
	AutoCompactTokenLimit       *int64                       `json:"auto_compact_token_limit,omitempty"`
	EffectiveContextWindowPct   int64                        `json:"effective_context_window_percent,omitempty"`
	InputModalities             []string                     `json:"input_modalities"`
	ExperimentalSupportedTools  []string                     `json:"experimental_supported_tools"`
	BaseInstructions            string                       `json:"base_instructions"`
	ShellType                   string                       `json:"shell_type"`
	DefaultReasoningLevel       string                       `json:"default_reasoning_level"`
}

type providerRequestError struct {
	statusCode int
	err        error
}

type azureBaseURLKind int

const (
	azureBaseURLKindInvalid azureBaseURLKind = iota
	azureBaseURLKindLegacyOpenAI
	azureBaseURLKindOpenAIV1
	azureBaseURLKindModels
)

func (e *providerRequestError) Error() string {
	if e == nil || e.err == nil {
		return ""
	}
	return e.err.Error()
}

func (e *providerRequestError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.err
}

func LoadProvidersConfigFile(path string) (ProvidersConfig, error) {
	var cfg ProvidersConfig
	path = strings.TrimSpace(path)
	if path == "" {
		return cfg, nil
	}

	body, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read providers config %q: %w", path, err)
	}
	if err := decodeProvidersConfigFile(path, body, &cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func decodeProvidersConfigFile(path string, body []byte, cfg *ProvidersConfig) error {
	if len(bytes.TrimSpace(body)) == 0 {
		return fmt.Errorf("providers config %q is empty", path)
	}

	switch strings.ToLower(filepath.Ext(path)) {
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(body, cfg); err != nil {
			return fmt.Errorf("decode providers config %q as YAML: %w", path, err)
		}
	default:
		if err := json.Unmarshal(body, cfg); err != nil {
			return fmt.Errorf("decode providers config %q as JSON: %w", path, err)
		}
	}
	return nil
}

func (c ProvidersConfig) UsesCopilot() bool {
	if len(c.Providers) == 0 {
		return true
	}
	for _, provider := range c.Providers {
		if providerType(strings.TrimSpace(provider.Type)) == providerTypeCopilot {
			return true
		}
	}
	return false
}

func defaultProviderSetup(h *ProxyHandler) *providerSetup {
	return &providerSetup{
		providers: map[string]*providerRuntime{
			"copilot": {
				id:        "copilot",
				kind:      providerTypeCopilot,
				isDefault: true,
				baseURL:   strings.TrimRight(h.copilotURL, "/"),
				selector:  providerEndpointSelectorRoundRobin,
				endpoints: []*providerEndpointRuntime{{
					id:      "default",
					baseURL: strings.TrimRight(h.copilotURL, "/"),
					weight:  1,
					health:  mustDefaultProviderEndpointHealth(),
				}},
				includeModels: map[string]struct{}{},
				excludeModels: map[string]struct{}{},
				staticModels:  map[string]providerModel{},
			},
		},
		providerOrder:     []string{"copilot"},
		defaultProviderID: "copilot",
		models:            map[string]providerModel{},
		providerModels:    map[string]map[string]providerModel{},
		modelRoutes:       map[string]*modelRouteRuntime{},
	}
}

func (h *ProxyHandler) providerSetup() *providerSetup {
	if h != nil && h.providersState != nil {
		return h.providersState
	}
	return defaultProviderSetup(h)
}

func (ps *providerSetup) defaultProvider() *providerRuntime {
	if ps == nil {
		return nil
	}
	return ps.providers[ps.defaultProviderID]
}

func (ps *providerSetup) providerByID(id string) *providerRuntime {
	if ps == nil {
		return nil
	}
	return ps.providers[id]
}

func (ps *providerSetup) lookupModel(model string) (providerModel, bool) {
	if ps == nil {
		return providerModel{}, false
	}
	ps.modelsMu.RLock()
	defer ps.modelsMu.RUnlock()
	pm, ok := ps.models[strings.TrimSpace(model)]
	return pm, ok
}

func (ps *providerSetup) lookupProviderModel(providerID, model string) (providerModel, bool) {
	if ps == nil {
		return providerModel{}, false
	}
	ps.modelsMu.RLock()
	defer ps.modelsMu.RUnlock()
	models := ps.providerModels[strings.TrimSpace(providerID)]
	if models == nil {
		return providerModel{}, false
	}
	pm, ok := models[strings.TrimSpace(model)]
	return pm, ok
}

func (ps *providerSetup) lookupModelRoute(model string) (*modelRouteRuntime, bool) {
	if ps == nil {
		return nil, false
	}
	route := ps.modelRoutes[strings.TrimSpace(model)]
	return route, route != nil
}

func (ps *providerSetup) setProviderModels(providerID string, models []providerModel) {
	if ps.providerModels == nil {
		ps.providerModels = make(map[string]map[string]providerModel)
	}
	byID := make(map[string]providerModel, len(models))
	for _, model := range models {
		byID[model.publicID] = model
	}
	ps.providerModels[providerID] = byID
}

func (ps *providerSetup) replaceProviderModels(providerID string, models []providerModel) error {
	if ps == nil {
		return nil
	}

	provider := ps.providerByID(providerID)
	models = filterProviderModels(provider, models)

	ps.modelsMu.Lock()
	defer ps.modelsMu.Unlock()

	next := make(map[string]providerModel, len(ps.models)+len(models)+len(ps.modelRoutes))
	for publicID, model := range ps.models {
		if model.providerID == providerID {
			continue
		}
		next[publicID] = model
	}

	ps.setProviderModels(providerID, models)
	oldModels := ps.models
	ps.models = next
	for _, model := range models {
		if err := ps.addCatalogProviderModel(model); err != nil {
			ps.models = oldModels
			return err
		}
	}
	next = ps.models
	for _, routeID := range ps.modelRouteOrder {
		route := ps.modelRoutes[routeID]
		if route != nil {
			next[route.publicID] = route.syntheticModel()
		}
	}
	ps.models = next
	return nil
}

func (ps *providerSetup) modelsForProvider(providerID string) []providerModel {
	if ps == nil {
		return nil
	}
	ps.modelsMu.RLock()
	defer ps.modelsMu.RUnlock()

	models := make([]providerModel, 0)
	for _, model := range ps.models {
		if model.providerID == providerID {
			models = append(models, model)
		}
	}
	return models
}

func (h *ProxyHandler) initializeProviders() error {
	if len(h.providersConfig.Providers) == 0 {
		return nil
	}

	setup, err := h.buildConfiguredProviderSetup(context.Background(), h.providersConfig)
	if err != nil {
		return err
	}
	h.providersState = setup
	return nil
}

func (h *ProxyHandler) buildConfiguredProviderSetup(ctx context.Context, cfg ProvidersConfig) (*providerSetup, error) {
	providers, providerOrder, defaultProviderID, err := h.buildProviders(cfg)
	if err != nil {
		return nil, err
	}

	setup := &providerSetup{
		providers:          providers,
		providerOrder:      providerOrder,
		defaultProviderID:  defaultProviderID,
		models:             make(map[string]providerModel),
		providerModels:     make(map[string]map[string]providerModel),
		modelRoutes:        make(map[string]*modelRouteRuntime),
		routeCandidateRefs: collectModelRouteCandidateRefs(cfg.ModelRoutes),
		ambiguousModelIDs:  make(map[string]struct{}),
		hasConfiguredState: true,
	}

	needsDynamicModelValidation := len(providers) > 1 && hasDynamicProvider(providers)

	if !needsDynamicModelValidation {
		for _, provider := range providers {
			models := filterProviderModels(provider, orderedStaticProviderModels(provider))
			setup.setProviderModels(provider.id, models)
			for _, model := range models {
				if err := setup.addCatalogProviderModel(model); err != nil {
					return nil, err
				}
			}
		}
		if err := setup.buildModelRoutes(cfg.ModelRoutes); err != nil {
			return nil, err
		}
		return setup, nil
	}

	if len(providers) == 0 {
		return setup, nil
	}

	ctx, cancel := context.WithTimeout(ctx, modelsUpstreamTimeout)
	defer cancel()

	for _, providerID := range providerOrder {
		provider := providers[providerID]
		if !providerUsesDynamicModels(provider) {
			models := filterProviderModels(provider, orderedStaticProviderModels(provider))
			setup.setProviderModels(provider.id, models)
			for _, model := range models {
				if err := setup.addCatalogProviderModel(model); err != nil {
					return nil, err
				}
			}
			continue
		}

		result, err := h.fetchProviderModels(ctx, provider, "", "")
		if err != nil {
			return nil, fmt.Errorf("load models for provider %q: %w", provider.id, err)
		}
		models := filterProviderModels(provider, result.models)
		setup.setProviderModels(provider.id, models)
		for _, model := range models {
			if err := setup.addCatalogProviderModel(model); err != nil {
				return nil, err
			}
		}
	}

	if err := setup.buildModelRoutes(cfg.ModelRoutes); err != nil {
		return nil, err
	}

	return setup, nil
}

func (ps *providerSetup) addCatalogProviderModel(model providerModel) error {
	if _, ambiguous := ps.ambiguousModelIDs[model.publicID]; ambiguous {
		if ps.isModelRouteCandidate(model.providerID, model.publicID) {
			return nil
		}
		return providerModelCollisionError(model.publicID, "explicit model route candidates", model.providerID)
	}
	if existing, exists := ps.models[model.publicID]; exists {
		if existing.providerID == model.providerID {
			return nil
		}
		if ps.isModelRouteCandidate(existing.providerID, existing.publicID) && ps.isModelRouteCandidate(model.providerID, model.publicID) {
			delete(ps.models, model.publicID)
			if ps.ambiguousModelIDs == nil {
				ps.ambiguousModelIDs = make(map[string]struct{})
			}
			ps.ambiguousModelIDs[model.publicID] = struct{}{}
			return nil
		}
		return providerModelCollisionError(model.publicID, existing.providerID, model.providerID)
	}
	ps.models[model.publicID] = model
	return nil
}

func collectModelRouteCandidateRefs(routes []ModelRouteConfig) map[string]struct{} {
	refs := make(map[string]struct{})
	for _, route := range routes {
		for _, candidate := range route.Candidates {
			providerID := strings.TrimSpace(candidate.Provider)
			modelID := strings.TrimSpace(candidate.Model)
			if providerID != "" && modelID != "" {
				refs[providerID+"/"+modelID] = struct{}{}
			}
		}
	}
	return refs
}

func (ps *providerSetup) isModelRouteCandidate(providerID, modelID string) bool {
	if ps == nil || ps.routeCandidateRefs == nil {
		return false
	}
	_, ok := ps.routeCandidateRefs[strings.TrimSpace(providerID)+"/"+strings.TrimSpace(modelID)]
	return ok
}

func (ps *providerSetup) providerOwnedModelIDExists(publicID string) bool {
	publicID = strings.TrimSpace(publicID)
	for _, models := range ps.providerModels {
		if _, ok := models[publicID]; ok {
			return true
		}
	}
	return false
}

func (h *ProxyHandler) buildProviders(cfg ProvidersConfig) (map[string]*providerRuntime, []string, string, error) {
	providers := make(map[string]*providerRuntime, len(cfg.Providers))
	providerOrder := make([]string, 0, len(cfg.Providers))
	defaultProviderID := ""
	copilotProviders := 0

	for _, raw := range cfg.Providers {
		provider, err := buildProviderRuntime(raw, h.copilotURL)
		if err != nil {
			return nil, nil, "", err
		}
		if _, exists := providers[provider.id]; exists {
			return nil, nil, "", fmt.Errorf("duplicate provider id %q", provider.id)
		}
		providers[provider.id] = provider
		providerOrder = append(providerOrder, provider.id)
		if provider.kind == providerTypeCopilot {
			copilotProviders++
			if copilotProviders > 1 {
				return nil, nil, "", fmt.Errorf("multiple copilot providers configured; only one copilot provider is supported")
			}
		}
		if provider.isDefault {
			if defaultProviderID != "" {
				return nil, nil, "", fmt.Errorf("multiple default providers configured: %q and %q", defaultProviderID, provider.id)
			}
			defaultProviderID = provider.id
		}
	}

	if len(providers) == 0 {
		return nil, nil, "", fmt.Errorf("providers config must include at least one provider when provided explicitly")
	}

	if defaultProviderID == "" {
		switch {
		case len(providers) == 1:
			for id := range providers {
				defaultProviderID = id
			}
		case copilotProviders == 1:
			for _, provider := range providers {
				if provider.kind == providerTypeCopilot {
					defaultProviderID = provider.id
					break
				}
			}
		default:
			return nil, nil, "", fmt.Errorf("multiple providers configured but no default provider selected")
		}
	}

	if defaultProvider := providers[defaultProviderID]; defaultProvider != nil {
		defaultProvider.isDefault = true
	}

	return providers, providerOrder, defaultProviderID, nil
}

func buildProviderRuntime(cfg ProviderConfig, defaultCopilotURL string) (*providerRuntime, error) {
	id := strings.TrimSpace(cfg.ID)
	if id == "" {
		return nil, fmt.Errorf("provider id is required")
	}

	kind := providerType(strings.TrimSpace(cfg.Type))
	switch kind {
	case providerTypeCopilot, providerTypeAzureOpenAI, providerTypeOpenAICodex:
	default:
		return nil, fmt.Errorf("provider %q has unsupported type %q", id, cfg.Type)
	}

	runtime := &providerRuntime{
		id:            id,
		kind:          kind,
		isDefault:     cfg.Default,
		includeModels: make(map[string]struct{}, len(cfg.IncludeModels)),
		excludeModels: make(map[string]struct{}, len(cfg.ExcludeModels)),
		staticModels:  make(map[string]providerModel, len(cfg.Models)),
		staticConfigs: make(map[string]ProviderModelConfig, len(cfg.Models)),
	}

	for _, included := range cfg.IncludeModels {
		included = strings.TrimSpace(included)
		if included != "" {
			runtime.includeModels[included] = struct{}{}
		}
	}

	for _, excluded := range cfg.ExcludeModels {
		excluded = strings.TrimSpace(excluded)
		if excluded != "" {
			runtime.excludeModels[excluded] = struct{}{}
		}
	}

	switch kind {
	case providerTypeCopilot:
		if len(cfg.Endpoints) > 0 {
			return nil, fmt.Errorf("provider %q endpoints are supported only for azure-openai providers in v1; Copilot multi-account rotation is deferred", id)
		}
		runtime.baseURL = strings.TrimRight(defaultCopilotURL, "/")
		runtime.selector = providerEndpointSelectorRoundRobin
		runtime.endpoints = []*providerEndpointRuntime{{
			id:      "default",
			baseURL: runtime.baseURL,
			weight:  1,
			health:  mustDefaultProviderEndpointHealth(),
		}}
		runtime.headerProfiles = cfg.Headers
	case providerTypeAzureOpenAI:
		endpoints, err := buildAzureProviderEndpoints(id, cfg)
		if err != nil {
			return nil, err
		}
		runtime.endpoints = endpoints
		runtime.baseURL = endpoints[0].baseURL
		runtime.apiVersion = endpoints[0].apiVersion
		runtime.apiKey = endpoints[0].apiKey
		selector, err := parseProviderEndpointSelector(cfg.Selector, len(endpoints))
		if err != nil {
			return nil, fmt.Errorf("provider %q: %w", id, err)
		}
		runtime.selector = selector
		if len(cfg.Models) == 0 {
			return nil, fmt.Errorf("provider %q must configure at least one model", id)
		}
		for _, modelCfg := range cfg.Models {
			model, err := buildStaticProviderModel(id, modelCfg)
			if err != nil {
				return nil, err
			}
			if !runtime.allowsModel(model.publicID) {
				continue
			}
			if _, exists := runtime.staticModels[model.publicID]; exists {
				return nil, fmt.Errorf("provider %q configures model %q more than once", id, model.publicID)
			}
			runtime.staticModels[model.publicID] = model
			runtime.staticConfigs[model.publicID] = normalizeProviderModelConfig(modelCfg)
			runtime.staticOrder = append(runtime.staticOrder, model.publicID)
		}
	case providerTypeOpenAICodex:
		if len(cfg.Endpoints) > 0 {
			return nil, fmt.Errorf("provider %q endpoints are supported only for azure-openai providers in v1", id)
		}
		baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
		if baseURL == "" {
			baseURL = defaultOpenAICodexBaseURL
		}
		if err := validateGenericProviderBaseURL(id, "OpenAI Codex", baseURL); err != nil {
			return nil, err
		}
		codexAuth, err := newOpenAICodexAuth()
		if err != nil {
			return nil, fmt.Errorf("provider %q: %w", id, err)
		}
		runtime.baseURL = baseURL
		runtime.selector = providerEndpointSelectorRoundRobin
		runtime.endpoints = []*providerEndpointRuntime{{
			id:      "default",
			baseURL: baseURL,
			weight:  1,
			health:  mustDefaultProviderEndpointHealth(),
		}}
		runtime.codexAuth = codexAuth
	}

	return runtime, nil
}

func buildAzureProviderEndpoints(providerID string, cfg ProviderConfig) ([]*providerEndpointRuntime, error) {
	if len(cfg.Endpoints) == 0 {
		baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
		if baseURL == "" {
			return nil, fmt.Errorf("provider %q must set base_url", providerID)
		}
		if err := validateAzureProviderBaseURL(providerID, baseURL); err != nil {
			return nil, err
		}
		apiKey, err := resolveProviderAPIKey(providerID, "api_key", cfg.APIKey, cfg.APIKeyEnv)
		if err != nil {
			return nil, err
		}
		health, err := newProviderEndpointHealthRuntime(ProviderEndpointHealthConfig{})
		if err != nil {
			return nil, fmt.Errorf("provider %q endpoint default health: %w", providerID, err)
		}
		return []*providerEndpointRuntime{{
			id:         "default",
			baseURL:    baseURL,
			apiKey:     apiKey,
			apiVersion: strings.TrimSpace(cfg.APIVersion),
			weight:     1,
			health:     health,
		}}, nil
	}

	endpoints := make([]*providerEndpointRuntime, 0, len(cfg.Endpoints))
	seen := make(map[string]struct{}, len(cfg.Endpoints))
	for i, endpointCfg := range cfg.Endpoints {
		fieldPath := fmt.Sprintf("providers[%q].endpoints[%d]", providerID, i)
		endpointID := strings.TrimSpace(endpointCfg.ID)
		if endpointID == "" {
			return nil, fmt.Errorf("%s.id is required", fieldPath)
		}
		if _, exists := seen[endpointID]; exists {
			return nil, fmt.Errorf("provider %q has duplicate endpoint id %q", providerID, endpointID)
		}
		seen[endpointID] = struct{}{}

		baseURL := strings.TrimRight(strings.TrimSpace(endpointCfg.BaseURL), "/")
		if baseURL == "" {
			return nil, fmt.Errorf("%s.base_url is required", fieldPath)
		}
		if err := validateAzureProviderBaseURL(providerID, baseURL); err != nil {
			return nil, fmt.Errorf("%s.base_url: %w", fieldPath, err)
		}

		apiKeyRaw := endpointCfg.APIKey
		apiKeyEnv := endpointCfg.APIKeyEnv
		if strings.TrimSpace(apiKeyRaw) == "" && strings.TrimSpace(apiKeyEnv) == "" {
			apiKeyRaw = cfg.APIKey
			apiKeyEnv = cfg.APIKeyEnv
		}
		apiKey, err := resolveProviderAPIKey(providerID, fieldPath, apiKeyRaw, apiKeyEnv)
		if err != nil {
			return nil, err
		}

		weight := 1
		if endpointCfg.Weight != nil {
			weight = *endpointCfg.Weight
		}
		if weight <= 0 {
			return nil, fmt.Errorf("%s.weight must be greater than 0", fieldPath)
		}

		health, err := newProviderEndpointHealthRuntime(endpointCfg.Health)
		if err != nil {
			return nil, fmt.Errorf("%s.health: %w", fieldPath, err)
		}
		apiVersion := strings.TrimSpace(endpointCfg.APIVersion)
		if apiVersion == "" {
			apiVersion = strings.TrimSpace(cfg.APIVersion)
		}
		endpoints = append(endpoints, &providerEndpointRuntime{
			id:         endpointID,
			baseURL:    baseURL,
			apiKey:     apiKey,
			apiVersion: apiVersion,
			weight:     weight,
			health:     health,
		})
	}
	return endpoints, nil
}

func validateAzureProviderBaseURL(providerID, baseURL string) error {
	switch classifyAzureBaseURL(baseURL) {
	case azureBaseURLKindOpenAIV1, azureBaseURLKindLegacyOpenAI:
		return nil
	case azureBaseURLKindModels:
		return fmt.Errorf("provider %q has unsupported Azure base_url %q: Azure AI Foundry /models inference endpoints are not supported; use the OpenAI-compatible endpoint ending in /openai/v1 instead", providerID, baseURL)
	default:
		return fmt.Errorf("provider %q has unsupported Azure base_url %q: expected an absolute URL whose path ends in /openai/v1 or /openai, with no query string or fragment", providerID, baseURL)
	}
}

func resolveProviderAPIKey(providerID, fieldPath, apiKeyRaw, apiKeyEnv string) (string, error) {
	apiKey := strings.TrimSpace(apiKeyRaw)
	if apiKey == "" && strings.TrimSpace(apiKeyEnv) != "" {
		apiKey = strings.TrimSpace(os.Getenv(strings.TrimSpace(apiKeyEnv)))
	}
	if apiKey == "" {
		return "", fmt.Errorf("provider %q %s must set api_key or api_key_env", providerID, fieldPath)
	}
	return apiKey, nil
}

func filterProviderModels(provider *providerRuntime, models []providerModel) []providerModel {
	if provider == nil || len(models) == 0 {
		return models
	}
	if len(provider.includeModels) == 0 && len(provider.excludeModels) == 0 {
		return models
	}

	firstFiltered := -1
	for i, model := range models {
		if !provider.allowsModel(model.publicID) {
			firstFiltered = i
			break
		}
	}
	if firstFiltered == -1 {
		return models
	}

	filtered := make([]providerModel, 0, len(models)-1)
	filtered = append(filtered, models[:firstFiltered]...)
	for _, model := range models[firstFiltered+1:] {
		if provider.allowsModel(model.publicID) {
			filtered = append(filtered, model)
		}
	}
	return filtered
}

func hasDynamicProvider(providers map[string]*providerRuntime) bool {
	for _, provider := range providers {
		if providerUsesDynamicModels(provider) {
			return true
		}
	}
	return false
}

func providerUsesDynamicModels(provider *providerRuntime) bool {
	if provider == nil {
		return false
	}
	return provider.kind == providerTypeCopilot || provider.kind == providerTypeOpenAICodex
}

func (p *providerRuntime) allowsModel(model string) bool {
	if p == nil {
		return true
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return false
	}
	if len(p.includeModels) > 0 {
		if _, included := p.includeModels[model]; !included {
			return false
		}
	}
	if _, excluded := p.excludeModels[model]; excluded {
		return false
	}
	return true
}

func providerModelCollisionError(publicID, existingProviderID, incomingProviderID string) error {
	return fmt.Errorf(
		"model %q is exposed by both provider %q and provider %q; resolve by adding include_models to the dynamic provider or exclude_models to one provider",
		publicID,
		existingProviderID,
		incomingProviderID,
	)
}

func buildStaticProviderModel(providerID string, cfg ProviderModelConfig) (providerModel, error) {
	publicID := strings.TrimSpace(cfg.PublicID)
	if publicID == "" {
		return providerModel{}, fmt.Errorf("provider %q contains a model without public_id", providerID)
	}

	upstreamModel := strings.TrimSpace(cfg.Deployment)
	if upstreamModel == "" {
		upstreamModel = publicID
	}

	name := strings.TrimSpace(cfg.Name)
	if name == "" {
		name = publicID
	}

	endpoints := normalizeProviderEndpoints(cfg.Endpoints)
	raw, err := synthesizeProviderModelRaw(providerID, publicID, name, endpoints, cfg)
	if err != nil {
		return providerModel{}, err
	}

	return providerModel{
		publicID:           publicID,
		upstreamModel:      upstreamModel,
		providerID:         providerID,
		supportedEndpoints: endpoints,
		raw:                raw,
	}, nil
}

func normalizeProviderModelConfig(cfg ProviderModelConfig) ProviderModelConfig {
	cfg.PublicID = strings.TrimSpace(cfg.PublicID)
	cfg.Deployment = strings.TrimSpace(cfg.Deployment)
	cfg.Name = strings.TrimSpace(cfg.Name)
	cfg.ModelPickerCategory = strings.TrimSpace(cfg.ModelPickerCategory)
	if cfg.Endpoints != nil {
		cfg.Endpoints = append([]string(nil), cfg.Endpoints...)
	}
	if cfg.ReasoningEffort != nil {
		cfg.ReasoningEffort = append([]string(nil), cfg.ReasoningEffort...)
	}
	return cfg
}

func normalizeProviderEndpoints(endpoints []string) []string {
	if len(endpoints) == 0 {
		return append([]string(nil), defaultStaticProviderEndpoints...)
	}

	normalized := make([]string, 0, len(endpoints))
	seen := make(map[string]struct{}, len(endpoints))
	for _, endpoint := range endpoints {
		endpoint = strings.TrimSpace(endpoint)
		if endpoint == "" {
			continue
		}
		if _, ok := seen[endpoint]; ok {
			continue
		}
		seen[endpoint] = struct{}{}
		normalized = append(normalized, endpoint)
	}
	if len(normalized) == 0 {
		return append([]string(nil), defaultStaticProviderEndpoints...)
	}
	return normalized
}

func synthesizeProviderModelRaw(providerID, publicID, name string, endpoints []string, cfg ProviderModelConfig) (json.RawMessage, error) {
	type limits struct {
		MaxContextWindowTokens int64 `json:"max_context_window_tokens,omitempty"`
	}
	type supports struct {
		ParallelToolCalls bool     `json:"parallel_tool_calls"`
		ReasoningEffort   []string `json:"reasoning_effort,omitempty"`
		Vision            bool     `json:"vision"`
	}
	type capabilities struct {
		Limits   limits   `json:"limits,omitempty"`
		Supports supports `json:"supports,omitempty"`
	}

	modelPickerEnabled := true
	if cfg.ModelPickerEnabled != nil {
		modelPickerEnabled = *cfg.ModelPickerEnabled
	}

	parallelToolCalls := false
	if cfg.ParallelToolCalls != nil {
		parallelToolCalls = *cfg.ParallelToolCalls
	}

	vision := false
	if cfg.Vision != nil {
		vision = *cfg.Vision
	}

	contextWindow := int64(0)
	if cfg.ContextWindow != nil {
		contextWindow = *cfg.ContextWindow
	}

	category := strings.TrimSpace(cfg.ModelPickerCategory)
	if category == "" {
		category = "versatile"
	}

	payload := map[string]interface{}{
		"id":                  publicID,
		"object":              "model",
		"created":             0,
		"owned_by":            providerID,
		"name":                name,
		"supported_endpoints": endpoints,
		"capabilities": capabilities{
			Limits: limits{
				MaxContextWindowTokens: contextWindow,
			},
			Supports: supports{
				ParallelToolCalls: parallelToolCalls,
				ReasoningEffort:   append([]string(nil), cfg.ReasoningEffort...),
				Vision:            vision,
			},
		},
		"model_picker_enabled":  modelPickerEnabled,
		"model_picker_category": category,
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal model %q for provider %q: %w", publicID, providerID, err)
	}
	return raw, nil
}

func (h *ProxyHandler) resolveProviderModel(model, endpoint string) (*providerRuntime, providerModel, bool) {
	setup := h.providerSetup()
	model = strings.TrimSpace(model)
	if model != "" {
		if providerModel, ok := setup.lookupModel(model); ok {
			provider := setup.providerByID(providerModel.providerID)
			if provider != nil {
				return provider, providerModel, true
			}
		}
	}

	defaultProvider := setup.defaultProvider()
	if defaultProvider == nil {
		return nil, providerModel{}, false
	}
	return defaultProvider, providerModel{
		publicID:           model,
		upstreamModel:      model,
		providerID:         defaultProvider.id,
		supportedEndpoints: nil,
	}, false
}

func providerModelSupportsEndpoint(model providerModel, endpoint string) bool {
	if len(model.supportedEndpoints) == 0 {
		return true
	}
	return supportsEndpoint(model.supportedEndpoints, endpoint)
}

func rewriteRequestModelForProvider(body []byte, upstreamModel string) ([]byte, bool, error) {
	upstreamModel = strings.TrimSpace(upstreamModel)
	if upstreamModel == "" {
		return body, false, nil
	}

	current := extractResponsesRequestModel(body)
	if current == "" {
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			return body, false, nil
		}
		current = strings.TrimSpace(payload.Model)
	}

	if current == "" || current == upstreamModel {
		return body, false, nil
	}
	return rewriteResponsesRequestModel(body, upstreamModel)
}

func (h *ProxyHandler) providerRequestURL(provider *providerRuntime, path string, extraQuery string) (string, error) {
	return h.providerEndpointRequestURL(provider, nil, path, extraQuery)
}

func (h *ProxyHandler) providerEndpointRequestURL(provider *providerRuntime, endpoint *providerEndpointRuntime, path string, extraQuery string) (string, error) {
	if provider == nil {
		return "", fmt.Errorf("provider is required")
	}
	if endpoint == nil && len(provider.endpoints) > 0 {
		endpoint = provider.endpoints[0]
	}

	baseURL := strings.TrimRight(provider.baseURL, "/")
	apiVersion := provider.apiVersion
	if endpoint != nil {
		baseURL = strings.TrimRight(endpoint.baseURL, "/")
		apiVersion = strings.TrimSpace(endpoint.apiVersion)
	}
	fullURL := baseURL + path
	if provider.kind != providerTypeAzureOpenAI || apiVersion == "" || classifyAzureBaseURL(baseURL) == azureBaseURLKindOpenAIV1 {
		return appendRawQuery(fullURL, extraQuery), nil
	}
	return appendRawQuery(fullURL, appendQuery("api-version="+url.QueryEscape(apiVersion), extraQuery)), nil
}

func classifyAzureBaseURL(baseURL string) azureBaseURLKind {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if trimmed == "" {
		return azureBaseURLKindInvalid
	}

	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return azureBaseURLKindInvalid
	}
	if parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" || strings.Contains(trimmed, "#") {
		return azureBaseURLKindInvalid
	}

	path := strings.TrimRight(parsed.Path, "/")
	switch {
	case strings.HasSuffix(path, "/openai/v1"):
		return azureBaseURLKindOpenAIV1
	case strings.HasSuffix(path, "/openai"):
		return azureBaseURLKindLegacyOpenAI
	case strings.HasSuffix(path, "/models"):
		return azureBaseURLKindModels
	default:
		return azureBaseURLKindInvalid
	}
}

func validateGenericProviderBaseURL(providerID, label, baseURL string) error {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return fmt.Errorf("provider %q has unsupported %s base_url %q: expected an absolute URL with no query string or fragment", providerID, label, baseURL)
	}
	if parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" || strings.Contains(trimmed, "#") {
		return fmt.Errorf("provider %q has unsupported %s base_url %q: expected an absolute URL with no query string or fragment", providerID, label, baseURL)
	}
	return nil
}

func appendQuery(parts ...string) string {
	combined := ""
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if combined == "" {
			combined = part
			continue
		}
		combined += "&" + part
	}
	return combined
}

func appendRawQuery(rawURL, rawQuery string) string {
	rawQuery = strings.TrimSpace(strings.TrimPrefix(rawQuery, "?"))
	if rawQuery == "" {
		return rawURL
	}
	separator := "?"
	if strings.Contains(rawURL, "?") {
		separator = "&"
	}
	return rawURL + separator + rawQuery
}

func (h *ProxyHandler) applyProviderHeaders(req *http.Request, provider *providerRuntime, endpoint string) error {
	return h.applyProviderEndpointHeaders(req, provider, nil, endpoint)
}

func (h *ProxyHandler) applyProviderEndpointHeaders(req *http.Request, provider *providerRuntime, upstreamEndpoint *providerEndpointRuntime, endpoint string) error {
	if provider == nil {
		return &providerRequestError{statusCode: http.StatusInternalServerError, err: fmt.Errorf("provider is required")}
	}

	switch provider.kind {
	case providerTypeCopilot:
		token, err := h.auth.GetToken(req.Context())
		if err != nil {
			return &providerRequestError{statusCode: http.StatusInternalServerError, err: err}
		}
		h.setCopilotHeadersForProvider(req, token, provider, endpoint)
	case providerTypeAzureOpenAI:
		clearCopilotHeaders(req.Header)
		apiKey := provider.apiKey
		if upstreamEndpoint != nil {
			apiKey = upstreamEndpoint.apiKey
		}
		req.Header.Set("api-key", apiKey)
		req.Header.Set("Content-Type", "application/json")
	case providerTypeOpenAICodex:
		clearCopilotHeaders(req.Header)
		if provider.codexAuth == nil {
			return &providerRequestError{statusCode: http.StatusInternalServerError, err: fmt.Errorf("provider %q has no OpenAI Codex auth configured", provider.id)}
		}
		credentials, err := provider.codexAuth.credentials(req.Context(), h.client)
		if err != nil {
			return &providerRequestError{statusCode: http.StatusInternalServerError, err: err}
		}
		req.Header.Set("Authorization", "Bearer "+credentials.accessToken)
		if credentials.accountID != "" {
			req.Header.Set("ChatGPT-Account-ID", credentials.accountID)
		}
		if credentials.fedRAMP {
			req.Header.Set("X-OpenAI-Fedramp", "true")
		}
		req.Header.Set("Content-Type", "application/json")
	default:
		return &providerRequestError{statusCode: http.StatusInternalServerError, err: fmt.Errorf("unsupported provider type %q", provider.kind)}
	}
	return nil
}

func (h *ProxyHandler) newProviderJSONRequest(ctx context.Context, provider *providerRuntime, method, path string, body []byte, extraHeaders http.Header, extraQuery string) (*http.Request, error) {
	return h.newProviderEndpointJSONRequest(ctx, provider, nil, method, path, body, extraHeaders, extraQuery)
}

func (h *ProxyHandler) newProviderEndpointJSONRequest(ctx context.Context, provider *providerRuntime, upstreamEndpoint *providerEndpointRuntime, method, path string, body []byte, extraHeaders http.Header, extraQuery string) (*http.Request, error) {
	fullURL, err := h.providerEndpointRequestURL(provider, upstreamEndpoint, path, extraQuery)
	if err != nil {
		return nil, err
	}

	var bodyReader io.Reader
	if len(body) > 0 {
		bodyReader = bytes.NewReader(body)
	}

	req, err := http.NewRequestWithContext(ctx, method, fullURL, bodyReader)
	if err != nil {
		return nil, err
	}
	if len(extraHeaders) > 0 {
		mergeHeaderValues(req.Header, extraHeaders)
	}
	if err := h.applyProviderEndpointHeaders(req, provider, upstreamEndpoint, path); err != nil {
		return nil, err
	}
	if h != nil && h.log != nil && upstreamEndpoint != nil {
		h.log.Debug("selected upstream endpoint", logger.F("provider", provider.id), logger.F("endpoint_id", upstreamEndpoint.id), logger.F("upstream_endpoint", path))
	}
	return req, nil
}

func (h *ProxyHandler) fetchProviderModels(ctx context.Context, provider *providerRuntime, rawQuery, ifNoneMatch string) (providerModelsFetchResult, error) {
	if provider == nil {
		return providerModelsFetchResult{}, fmt.Errorf("provider is required")
	}

	switch provider.kind {
	case providerTypeAzureOpenAI:
		models := orderedStaticProviderModels(provider)

		// Azure /models is only a best-effort metadata overlay for the configured
		// static catalog. Routing still comes from provider.models[], and sparse
		// or failed Azure metadata probes should leave the configured model list untouched.
		resp, err := h.doWithRetry(func() (*http.Request, error) {
			return h.newProviderJSONRequest(ctx, provider, http.MethodGet, "/models", nil, nil, "")
		})
		if err != nil {
			return providerModelsFetchResult{models: models}, nil
		}
		defer drainAndClose(resp.Body)
		if resp.StatusCode != http.StatusOK {
			return providerModelsFetchResult{models: models}, nil
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return providerModelsFetchResult{models: models}, nil
		}

		overlayModels, err := decodeProviderModelsFromBody(provider, body)
		if err != nil {
			return providerModelsFetchResult{models: models}, nil
		}

		overlayByID := make(map[string]providerModel, len(overlayModels))
		for _, overlay := range overlayModels {
			overlayByID[overlay.publicID] = overlay
		}
		for i, staticModel := range models {
			cfg, ok := provider.staticConfigs[staticModel.publicID]
			if !ok {
				continue
			}
			overlay, ok := findProviderModelMetadataOverlay(cfg, overlayByID)
			if !ok {
				continue
			}
			models[i] = mergeStaticProviderMetadata(staticModel, cfg, overlay)
		}

		return providerModelsFetchResult{models: models}, nil
	case providerTypeCopilot:
		resp, err := h.doWithRetry(func() (*http.Request, error) {
			req, err := h.newProviderJSONRequest(ctx, provider, http.MethodGet, "/models", nil, nil, rawQuery)
			if err != nil {
				return nil, err
			}
			if strings.TrimSpace(ifNoneMatch) != "" {
				req.Header.Set("If-None-Match", ifNoneMatch)
			}
			return req, nil
		})
		if err != nil {
			return providerModelsFetchResult{}, err
		}
		defer func() { _ = resp.Body.Close() }()

		result := providerModelsFetchResult{etag: resp.Header.Get("ETag")}
		if resp.StatusCode == http.StatusNotModified {
			result.notModified = true
			return result, nil
		}
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			return providerModelsFetchResult{}, &providerRequestError{
				statusCode: resp.StatusCode,
				err:        fmt.Errorf("unexpected /models status %d: %s", resp.StatusCode, string(body)),
			}
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return providerModelsFetchResult{}, err
		}

		models, err := decodeProviderModelsFromBody(provider, body)
		if err != nil {
			return providerModelsFetchResult{}, err
		}
		result.models = models
		return result, nil
	case providerTypeOpenAICodex:
		modelsQuery := openAICodexModelsRawQuery(rawQuery)
		resp, err := h.doWithRetry(func() (*http.Request, error) {
			req, err := h.newProviderJSONRequest(ctx, provider, http.MethodGet, "/models", nil, nil, modelsQuery)
			if err != nil {
				return nil, err
			}
			if strings.TrimSpace(ifNoneMatch) != "" {
				req.Header.Set("If-None-Match", ifNoneMatch)
			}
			return req, nil
		})
		if err != nil {
			return providerModelsFetchResult{}, err
		}
		defer func() { _ = resp.Body.Close() }()

		result := providerModelsFetchResult{etag: resp.Header.Get("ETag")}
		if resp.StatusCode == http.StatusNotModified {
			result.notModified = true
			return result, nil
		}
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			return providerModelsFetchResult{}, &providerRequestError{
				statusCode: resp.StatusCode,
				err:        fmt.Errorf("unexpected /models status %d: %s", resp.StatusCode, string(body)),
			}
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return providerModelsFetchResult{}, err
		}

		models, err := decodeOpenAICodexModelsFromBody(provider, body)
		if err != nil {
			return providerModelsFetchResult{}, err
		}
		result.models = models
		return result, nil
	default:
		return providerModelsFetchResult{}, fmt.Errorf("unsupported provider type %q", provider.kind)
	}
}

func orderedStaticProviderModels(provider *providerRuntime) []providerModel {
	if provider == nil {
		return nil
	}
	models := make([]providerModel, 0, len(provider.staticModels))
	for _, publicID := range provider.staticOrder {
		model, ok := provider.staticModels[publicID]
		if ok {
			models = append(models, model)
		}
	}
	return models
}

func findProviderModelMetadataOverlay(cfg ProviderModelConfig, overlayByID map[string]providerModel) (providerModel, bool) {
	publicID := strings.TrimSpace(cfg.PublicID)
	if publicID != "" {
		if overlay, ok := overlayByID[publicID]; ok {
			return overlay, true
		}
	}

	deployment := strings.TrimSpace(cfg.Deployment)
	if deployment != "" && deployment != publicID {
		if overlay, ok := overlayByID[deployment]; ok {
			return overlay, true
		}
	}

	return providerModel{}, false
}

func mergeStaticProviderMetadata(static providerModel, cfg ProviderModelConfig, overlay providerModel) providerModel {
	mergedRaw, err := mergeProviderModelMetadataOverlayRaw(static.raw, overlay.raw, cfg)
	if err != nil {
		return static
	}
	static.raw = mergedRaw
	return static
}

// mergeProviderModelMetadataOverlayRaw opportunistically copies provider metadata
// that already exists in the Azure /models overlay payload. It does not rewrite
// configured public IDs or endpoint allowlists, and it does not synthesize
// Codex-facing fields that an upstream provider omitted.
func mergeProviderModelMetadataOverlayRaw(baseRaw, overlayRaw json.RawMessage, cfg ProviderModelConfig) (json.RawMessage, error) {
	if len(baseRaw) == 0 || len(overlayRaw) == 0 {
		return append(json.RawMessage(nil), baseRaw...), nil
	}

	base, err := decodeRawJSONObject(baseRaw)
	if err != nil {
		return nil, err
	}
	overlay, err := decodeRawJSONObject(overlayRaw)
	if err != nil {
		return nil, err
	}

	for key, value := range overlay {
		if _, exists := base[key]; !exists {
			base[key] = append(json.RawMessage(nil), value...)
		}
	}

	if strings.TrimSpace(cfg.Name) == "" {
		copyRawField(base, overlay, "name")
	}
	if cfg.ModelPickerEnabled == nil {
		copyRawField(base, overlay, "model_picker_enabled")
	}
	if strings.TrimSpace(cfg.ModelPickerCategory) == "" {
		copyRawField(base, overlay, "model_picker_category")
	}

	if err := mergeProviderModelCapabilitiesOverlay(base, overlay, cfg); err != nil {
		return nil, err
	}

	merged, err := json.Marshal(base)
	if err != nil {
		return nil, err
	}
	return merged, nil
}

func mergeProviderModelCapabilitiesOverlay(base, overlay map[string]json.RawMessage, cfg ProviderModelConfig) error {
	baseCaps, err := decodeOptionalRawJSONObject(base["capabilities"])
	if err != nil {
		return err
	}
	overlayCaps, err := decodeOptionalRawJSONObject(overlay["capabilities"])
	if err != nil {
		return err
	}

	baseSupports, err := decodeOptionalRawJSONObject(baseCaps["supports"])
	if err != nil {
		return err
	}
	overlaySupports, err := decodeOptionalRawJSONObject(overlayCaps["supports"])
	if err != nil {
		return err
	}

	if cfg.ReasoningEffort == nil {
		copyRawField(baseSupports, overlaySupports, "reasoning_effort")
	}
	if cfg.ParallelToolCalls == nil {
		copyRawField(baseSupports, overlaySupports, "parallel_tool_calls")
	}
	if cfg.Vision == nil {
		copyRawField(baseSupports, overlaySupports, "vision")
	}

	if len(baseSupports) > 0 {
		encoded, err := json.Marshal(baseSupports)
		if err != nil {
			return err
		}
		baseCaps["supports"] = encoded
	}

	baseLimits, err := decodeOptionalRawJSONObject(baseCaps["limits"])
	if err != nil {
		return err
	}
	overlayLimits, err := decodeOptionalRawJSONObject(overlayCaps["limits"])
	if err != nil {
		return err
	}

	if cfg.ContextWindow == nil {
		copyRawField(baseLimits, overlayLimits, "max_context_window_tokens")
	}

	if len(baseLimits) > 0 {
		encoded, err := json.Marshal(baseLimits)
		if err != nil {
			return err
		}
		baseCaps["limits"] = encoded
	}

	if len(baseCaps) > 0 {
		encoded, err := json.Marshal(baseCaps)
		if err != nil {
			return err
		}
		base["capabilities"] = encoded
	}

	return nil
}

func decodeRawJSONObject(raw json.RawMessage) (map[string]json.RawMessage, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if payload == nil {
		payload = map[string]json.RawMessage{}
	}
	return payload, nil
}

func decodeOptionalRawJSONObject(raw json.RawMessage) (map[string]json.RawMessage, error) {
	if len(raw) == 0 {
		return map[string]json.RawMessage{}, nil
	}
	return decodeRawJSONObject(raw)
}

func copyRawField(dst, src map[string]json.RawMessage, field string) {
	if dst == nil || src == nil {
		return
	}
	value, ok := src[field]
	if !ok {
		return
	}
	dst[field] = append(json.RawMessage(nil), value...)
}

func decodeProviderModelsFromBody(provider *providerRuntime, body []byte) ([]providerModel, error) {
	if provider == nil {
		return nil, fmt.Errorf("provider is required")
	}

	var upstream struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(body, &upstream); err != nil {
		return nil, err
	}

	models := make([]providerModel, 0, len(upstream.Data))
	indexByID := make(map[string]int, len(upstream.Data))
	for _, raw := range upstream.Data {
		var parsed struct {
			ID                 string   `json:"id"`
			SupportedEndpoints []string `json:"supported_endpoints"`
			Policy             struct {
				State string `json:"state"`
			} `json:"policy"`
		}
		if err := json.Unmarshal(raw, &parsed); err != nil {
			continue
		}
		publicID := strings.TrimSpace(parsed.ID)
		if publicID == "" {
			continue
		}
		if !provider.allowsModel(publicID) {
			continue
		}

		supportedEndpoints := normalizeDynamicProviderEndpoints(parsed.SupportedEndpoints)
		disabled := strings.EqualFold(parsed.Policy.State, "disabled")
		if index, duplicate := indexByID[publicID]; duplicate {
			merged := models[index]
			merged.supportedEndpoints = mergeDynamicProviderEndpoints(merged.supportedEndpoints, supportedEndpoints)
			merged.disabled = merged.disabled && disabled
			baseRaw := merged.raw
			if merged.disabled != models[index].disabled && !merged.disabled {
				baseRaw = raw
			}
			merged.raw = mergeProviderModelRaw(baseRaw, merged.supportedEndpoints)
			models[index] = merged
			continue
		}

		indexByID[publicID] = len(models)
		models = append(models, providerModel{
			publicID:           publicID,
			upstreamModel:      publicID,
			providerID:         provider.id,
			supportedEndpoints: supportedEndpoints,
			disabled:           disabled,
			raw:                mergeProviderModelRaw(raw, supportedEndpoints),
		})
	}

	return models, nil
}

func decodeOpenAICodexModelsFromBody(provider *providerRuntime, body []byte) ([]providerModel, error) {
	if provider == nil {
		return nil, fmt.Errorf("provider is required")
	}

	var upstream struct {
		Models []json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(body, &upstream); err != nil {
		return nil, err
	}

	models := make([]providerModel, 0, len(upstream.Models))
	seen := make(map[string]struct{}, len(upstream.Models))
	for _, raw := range upstream.Models {
		var parsed openAICodexModelPayload
		if err := json.Unmarshal(raw, &parsed); err != nil {
			continue
		}

		publicID := strings.TrimSpace(parsed.Slug)
		if publicID == "" {
			continue
		}
		if _, duplicate := seen[publicID]; duplicate {
			continue
		}
		visibilityList := strings.EqualFold(strings.TrimSpace(parsed.Visibility), "list")
		if !parsed.SupportedInAPI || !visibilityList {
			continue
		}
		if !provider.allowsModel(publicID) {
			continue
		}

		modelRaw, err := synthesizeOpenAICodexModelRaw(provider.id, parsed)
		if err != nil {
			return nil, err
		}

		seen[publicID] = struct{}{}
		models = append(models, providerModel{
			publicID:           publicID,
			upstreamModel:      publicID,
			providerID:         provider.id,
			supportedEndpoints: append([]string(nil), openAICodexProviderEndpoints...),
			raw:                modelRaw,
		})
	}

	return models, nil
}

func synthesizeOpenAICodexModelRaw(providerID string, parsed openAICodexModelPayload) (json.RawMessage, error) {
	reasoningEffort := make([]string, 0, len(parsed.SupportedReasoningLevels))
	for _, level := range parsed.SupportedReasoningLevels {
		effort := strings.TrimSpace(level.Effort)
		if effort != "" {
			reasoningEffort = append(reasoningEffort, effort)
		}
	}

	name := strings.TrimSpace(parsed.DisplayName)
	if name == "" {
		name = strings.TrimSpace(parsed.Slug)
	}

	vision := parsed.SupportsImageDetailOriginal
	for _, modality := range parsed.InputModalities {
		if strings.EqualFold(strings.TrimSpace(modality), "image") {
			vision = true
			break
		}
	}

	modelPickerCategory := "versatile"
	switch {
	case parsed.Priority <= 0:
		modelPickerCategory = "powerful"
	case parsed.Priority >= 8:
		modelPickerCategory = "lightweight"
	}

	var maxContextWindowTokens int64
	switch {
	case parsed.MaxContextWindow != nil && *parsed.MaxContextWindow > 0:
		maxContextWindowTokens = *parsed.MaxContextWindow
	case parsed.ContextWindow != nil && *parsed.ContextWindow > 0:
		maxContextWindowTokens = *parsed.ContextWindow
	}

	payload := map[string]interface{}{
		"id":                    parsed.Slug,
		"object":                "model",
		"created":               0,
		"owned_by":              providerID,
		"name":                  name,
		"supported_endpoints":   openAICodexProviderEndpoints,
		"model_picker_enabled":  parsed.SupportedInAPI && strings.EqualFold(strings.TrimSpace(parsed.Visibility), "list"),
		"model_picker_category": modelPickerCategory,
		"capabilities": map[string]interface{}{
			"limits": map[string]interface{}{
				"max_context_window_tokens": maxContextWindowTokens,
			},
			"supports": map[string]interface{}{
				"parallel_tool_calls": parsed.SupportsParallelToolCalls,
				"reasoning_effort":    reasoningEffort,
				"vision":              vision,
			},
		},
		"slug":                           parsed.Slug,
		"display_name":                   name,
		"description":                    parsed.Description,
		"visibility":                     parsed.Visibility,
		"supported_in_api":               parsed.SupportedInAPI,
		"priority":                       parsed.Priority,
		"supports_reasoning_summaries":   parsed.SupportsReasoningSummaries,
		"support_verbosity":              parsed.SupportVerbosity,
		"supports_parallel_tool_calls":   parsed.SupportsParallelToolCalls,
		"supports_image_detail_original": parsed.SupportsImageDetailOriginal,
		"input_modalities":               parsed.InputModalities,
		"experimental_supported_tools":   parsed.ExperimentalSupportedTools,
		"base_instructions":              parsed.BaseInstructions,
		"shell_type":                     parsed.ShellType,
		"default_reasoning_level":        strings.TrimSpace(parsed.DefaultReasoningLevel),
	}
	if parsed.ContextWindow != nil {
		payload["context_window"] = *parsed.ContextWindow
	}
	if parsed.MaxContextWindow != nil {
		payload["max_context_window"] = *parsed.MaxContextWindow
	}
	if parsed.AutoCompactTokenLimit != nil {
		payload["auto_compact_token_limit"] = *parsed.AutoCompactTokenLimit
	}
	if parsed.EffectiveContextWindowPct > 0 {
		payload["effective_context_window_percent"] = parsed.EffectiveContextWindowPct
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal OpenAI Codex model %q for provider %q: %w", parsed.Slug, providerID, err)
	}
	return raw, nil
}

func openAICodexModelsRawQuery(rawQuery string) string {
	rawQuery = strings.TrimSpace(strings.TrimPrefix(rawQuery, "?"))
	if rawQueryHasParam(rawQuery, "client_version") {
		return rawQuery
	}
	return appendQuery(rawQuery, "client_version="+url.QueryEscape(defaultOpenAICodexClientVersion))
}

func rawQueryHasParam(rawQuery, name string) bool {
	rawQuery = strings.TrimSpace(strings.TrimPrefix(rawQuery, "?"))
	if rawQuery == "" {
		return false
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return strings.Contains(rawQuery, url.QueryEscape(name)+"=") || strings.Contains(rawQuery, name+"=")
	}
	_, ok := values[name]
	return ok
}

func normalizeDynamicProviderEndpoints(endpoints []string) []string {
	if len(endpoints) == 0 {
		return nil
	}

	normalized := make([]string, 0, len(endpoints))
	seen := make(map[string]struct{}, len(endpoints))
	for _, endpoint := range endpoints {
		endpoint = strings.TrimSpace(endpoint)
		if endpoint == "" {
			continue
		}
		if _, exists := seen[endpoint]; exists {
			continue
		}
		seen[endpoint] = struct{}{}
		normalized = append(normalized, endpoint)
	}
	if len(normalized) == 0 {
		return nil
	}
	return normalized
}

func mergeDynamicProviderEndpoints(existing, incoming []string) []string {
	if len(existing) == 0 || len(incoming) == 0 {
		return nil
	}

	merged := append([]string(nil), existing...)
	seen := make(map[string]struct{}, len(existing)+len(incoming))
	for _, endpoint := range existing {
		seen[endpoint] = struct{}{}
	}
	for _, endpoint := range incoming {
		if _, exists := seen[endpoint]; exists {
			continue
		}
		seen[endpoint] = struct{}{}
		merged = append(merged, endpoint)
	}
	return merged
}

func mergeProviderModelRaw(raw json.RawMessage, supportedEndpoints []string) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(raw, &payload); err != nil {
		return append(json.RawMessage(nil), raw...)
	}

	if len(supportedEndpoints) == 0 {
		delete(payload, "supported_endpoints")
	} else if encoded, err := json.Marshal(supportedEndpoints); err == nil {
		payload["supported_endpoints"] = encoded
	}

	merged, err := json.Marshal(payload)
	if err != nil {
		return append(json.RawMessage(nil), raw...)
	}
	return merged
}
