package proxy

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultEndpointErrorBudget = "10/m"
	defaultEndpointCooldown    = 30 * time.Second
	latencyEWMANumerator       = 2
	latencyEWMADenominator     = 10
)

type providerEndpointSelectorKind string

const (
	providerEndpointSelectorRoundRobin   providerEndpointSelectorKind = "round_robin"
	providerEndpointSelectorWeighted     providerEndpointSelectorKind = "weighted"
	providerEndpointSelectorLeastLatency providerEndpointSelectorKind = "least_latency"
)

type providerEndpointSelectorState struct {
	mu              sync.Mutex
	roundRobinIndex int
	weightedIndex   int
	weightedOffset  int
}

type providerEndpointRuntime struct {
	id         string
	baseURL    string
	apiKey     string
	apiVersion string
	weight     int
	health     *providerEndpointHealthRuntime
}

type endpointErrorBudget struct {
	limit  int
	window time.Duration
}

type providerEndpointHealthRuntime struct {
	mu               sync.Mutex
	errorBudget      endpointErrorBudget
	cooldown         time.Duration
	failures         []time.Time
	quarantinedUntil time.Time
	lastFailure      time.Time
	latencyEWMA      time.Duration
	hasLatency       bool
}

func parseProviderEndpointSelector(raw string, endpointCount int) (providerEndpointSelectorKind, error) {
	selector := providerEndpointSelectorKind(strings.TrimSpace(raw))
	if selector == "" {
		if endpointCount > 1 {
			return providerEndpointSelectorRoundRobin, nil
		}
		return providerEndpointSelectorRoundRobin, nil
	}
	switch selector {
	case providerEndpointSelectorRoundRobin, providerEndpointSelectorWeighted, providerEndpointSelectorLeastLatency:
		return selector, nil
	default:
		return "", fmt.Errorf("unsupported selector %q: expected round_robin, weighted, or least_latency", raw)
	}
}

func parseEndpointErrorBudget(raw string) (endpointErrorBudget, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		raw = defaultEndpointErrorBudget
	}
	parts := strings.Split(raw, "/")
	if len(parts) != 2 {
		return endpointErrorBudget{}, fmt.Errorf("invalid error_budget %q: expected N/{ms,s,m,h}", raw)
	}
	limit, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil || limit <= 0 {
		return endpointErrorBudget{}, fmt.Errorf("invalid error_budget %q: count must be positive", raw)
	}
	window, err := parseEndpointBudgetWindow(strings.TrimSpace(parts[1]))
	if err != nil {
		return endpointErrorBudget{}, fmt.Errorf("invalid error_budget %q: %w", raw, err)
	}
	return endpointErrorBudget{limit: limit, window: window}, nil
}

func parseEndpointBudgetWindow(raw string) (time.Duration, error) {
	switch raw {
	case "ms":
		return time.Millisecond, nil
	case "s":
		return time.Second, nil
	case "m":
		return time.Minute, nil
	case "h":
		return time.Hour, nil
	default:
		d, err := time.ParseDuration(raw)
		if err != nil || d <= 0 {
			return 0, fmt.Errorf("window must be one of ms, s, m, h, or a positive duration")
		}
		return d, nil
	}
}

func parseEndpointCooldown(raw string) (time.Duration, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return defaultEndpointCooldown, nil
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		return 0, fmt.Errorf("invalid cooldown %q: expected positive duration", raw)
	}
	return d, nil
}

func newProviderEndpointHealthRuntime(cfg ProviderEndpointHealthConfig) (*providerEndpointHealthRuntime, error) {
	budget, err := parseEndpointErrorBudget(cfg.ErrorBudget)
	if err != nil {
		return nil, err
	}
	cooldown, err := parseEndpointCooldown(cfg.Cooldown)
	if err != nil {
		return nil, err
	}
	return &providerEndpointHealthRuntime{errorBudget: budget, cooldown: cooldown}, nil
}

func mustDefaultProviderEndpointHealth() *providerEndpointHealthRuntime {
	health, err := newProviderEndpointHealthRuntime(ProviderEndpointHealthConfig{})
	if err != nil {
		panic(err)
	}
	return health
}

func (p *providerRuntime) selectEndpoint(exclude map[string]struct{}) (*providerEndpointRuntime, bool, error) {
	if p == nil {
		return nil, false, fmt.Errorf("provider is required")
	}
	if len(p.endpoints) == 0 {
		return nil, false, fmt.Errorf("provider %q has no endpoints", p.id)
	}
	now := time.Now()
	candidates := p.endpointCandidates(now, exclude)
	if len(candidates) == 0 && len(exclude) > 0 {
		candidates = p.endpointCandidates(now, nil)
	}
	if len(candidates) == 0 {
		return p.leastRecentlyFailedEndpoint(), true, nil
	}
	switch p.selector {
	case providerEndpointSelectorWeighted:
		return p.selectWeightedEndpoint(candidates), false, nil
	case providerEndpointSelectorLeastLatency:
		return p.selectLeastLatencyEndpoint(candidates), false, nil
	default:
		return p.selectRoundRobinEndpoint(candidates), false, nil
	}
}

func (p *providerRuntime) endpointCandidates(now time.Time, exclude map[string]struct{}) []*providerEndpointRuntime {
	candidates := make([]*providerEndpointRuntime, 0, len(p.endpoints))
	for _, endpoint := range p.endpoints {
		if endpoint == nil {
			continue
		}
		if exclude != nil {
			if _, skipped := exclude[endpoint.id]; skipped {
				continue
			}
		}
		if endpoint.isAvailable(now) {
			candidates = append(candidates, endpoint)
		}
	}
	return candidates
}

func (p *providerRuntime) selectRoundRobinEndpoint(candidates []*providerEndpointRuntime) *providerEndpointRuntime {
	p.selectorState.mu.Lock()
	defer p.selectorState.mu.Unlock()
	idx := p.selectorState.roundRobinIndex % len(candidates)
	p.selectorState.roundRobinIndex++
	return candidates[idx]
}

func (p *providerRuntime) selectWeightedEndpoint(candidates []*providerEndpointRuntime) *providerEndpointRuntime {
	p.selectorState.mu.Lock()
	defer p.selectorState.mu.Unlock()
	for range len(candidates) * 2 {
		candidate := candidates[p.selectorState.weightedIndex%len(candidates)]
		if p.selectorState.weightedOffset < candidate.weight {
			p.selectorState.weightedOffset++
			if p.selectorState.weightedOffset >= candidate.weight {
				p.selectorState.weightedOffset = 0
				p.selectorState.weightedIndex++
			}
			return candidate
		}
		p.selectorState.weightedOffset = 0
		p.selectorState.weightedIndex++
	}
	return candidates[0]
}

func (p *providerRuntime) selectLeastLatencyEndpoint(candidates []*providerEndpointRuntime) *providerEndpointRuntime {
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

func (p *providerRuntime) leastRecentlyFailedEndpoint() *providerEndpointRuntime {
	best := p.endpoints[0]
	bestFailure := best.lastFailure()
	for _, endpoint := range p.endpoints[1:] {
		failure := endpoint.lastFailure()
		if bestFailure.IsZero() || (!failure.IsZero() && failure.Before(bestFailure)) {
			best = endpoint
			bestFailure = failure
		}
	}
	return best
}

func (e *providerEndpointRuntime) isAvailable(now time.Time) bool {
	if e == nil || e.health == nil {
		return true
	}
	e.health.mu.Lock()
	defer e.health.mu.Unlock()
	return !now.Before(e.health.quarantinedUntil)
}

func (e *providerEndpointRuntime) recordFailure(now time.Time) {
	if e == nil || e.health == nil {
		return
	}
	e.health.mu.Lock()
	defer e.health.mu.Unlock()
	e.health.lastFailure = now
	cutoff := now.Add(-e.health.errorBudget.window)
	kept := e.health.failures[:0]
	for _, failure := range e.health.failures {
		if !failure.Before(cutoff) {
			kept = append(kept, failure)
		}
	}
	e.health.failures = append(kept, now)
	if len(e.health.failures) >= e.health.errorBudget.limit {
		e.health.quarantinedUntil = now.Add(e.health.cooldown)
	}
}

func (e *providerEndpointRuntime) recordSuccess(latency time.Duration) {
	if e == nil || e.health == nil || latency <= 0 {
		return
	}
	e.health.mu.Lock()
	defer e.health.mu.Unlock()
	if e.health.hasLatency {
		e.health.latencyEWMA = (e.health.latencyEWMA*time.Duration(latencyEWMADenominator-latencyEWMANumerator) + latency*time.Duration(latencyEWMANumerator)) / latencyEWMADenominator
	} else {
		e.health.latencyEWMA = latency
		e.health.hasLatency = true
	}
	if !e.health.quarantinedUntil.IsZero() && time.Now().After(e.health.quarantinedUntil) {
		e.health.failures = nil
		e.health.quarantinedUntil = time.Time{}
	}
}

func (e *providerEndpointRuntime) latency() (time.Duration, bool) {
	if e == nil || e.health == nil {
		return 0, false
	}
	e.health.mu.Lock()
	defer e.health.mu.Unlock()
	return e.health.latencyEWMA, e.health.hasLatency
}

func (e *providerEndpointRuntime) lastFailure() time.Time {
	if e == nil || e.health == nil {
		return time.Time{}
	}
	e.health.mu.Lock()
	defer e.health.mu.Unlock()
	return e.health.lastFailure
}
