import { useEffect, useMemo, useState } from 'react';
import { DEFAULT_FILTERS, fetchModels, fetchSystemInfo } from './api';

const THEME_KEY = 'llmfit-theme';

const FIT_OPTIONS = [
  { value: 'marginal', label: 'Runnable (Marginal+)' },
  { value: 'good', label: 'Good or better' },
  { value: 'perfect', label: 'Perfect only' },
  { value: 'too_tight', label: 'Too-tight only' },
  { value: 'all', label: 'All levels' }
];

const RUNTIME_OPTIONS = [
  { value: 'any', label: 'Any runtime' },
  { value: 'mlx', label: 'MLX' },
  { value: 'llamacpp', label: 'llama.cpp' },
  { value: 'vllm', label: 'vLLM' }
];

const USE_CASE_OPTIONS = [
  { value: 'all', label: 'All use cases' },
  { value: 'general', label: 'General' },
  { value: 'coding', label: 'Coding' },
  { value: 'reasoning', label: 'Reasoning' },
  { value: 'chat', label: 'Chat' },
  { value: 'multimodal', label: 'Multimodal' },
  { value: 'embedding', label: 'Embedding' }
];

const LIMIT_OPTIONS = [
  { value: '10', label: '10' },
  { value: '20', label: '20' },
  { value: '50', label: '50' },
  { value: '100', label: '100' },
  { value: '200', label: '200' },
  { value: '', label: 'All' }
];

const SORT_OPTIONS = [
  { value: 'score', label: 'Sort: Score' },
  { value: 'tps', label: 'Sort: TPS' },
  { value: 'params', label: 'Sort: Params' },
  { value: 'mem', label: 'Sort: Memory' },
  { value: 'ctx', label: 'Sort: Context' },
  { value: 'date', label: 'Sort: Release date' },
  { value: 'use_case', label: 'Sort: Use case' }
];

function initialTheme() {
  if (typeof window === 'undefined') {
    return 'light';
  }

  const stored = window.localStorage.getItem(THEME_KEY);
  if (stored === 'light' || stored === 'dark') {
    return stored;
  }

  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function round(value, digits = 1) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(digits);
}

function fitClass(code) {
  return `fit fit-${code || 'unknown'}`;
}

function modeClass(code) {
  return `mode mode-${code || 'unknown'}`;
}

function SystemCard({ label, value, detail }) {
  return (
    <article className="system-card">
      <p className="system-label">{label}</p>
      <p className="system-value">{value}</p>
      {detail ? <p className="system-detail">{detail}</p> : null}
    </article>
  );
}

function MetricBar({ label, value }) {
  const safe = Number.isFinite(value) ? Math.max(0, Math.min(value, 100)) : 0;
  return (
    <div className="metric-row">
      <div className="metric-text">
        <span>{label}</span>
        <span>{round(value, 1)}</span>
      </div>
      <div className="metric-track">
        <div className="metric-fill" style={{ width: `${safe}%` }} />
      </div>
    </div>
  );
}

function fitRank(level) {
  switch (level) {
    case 'perfect':
      return 3;
    case 'good':
      return 2;
    case 'marginal':
      return 1;
    case 'too_tight':
      return 0;
    default:
      return -1;
  }
}

function applyClientFitFilter(models, minFit) {
  const list = Array.isArray(models) ? models : [];
  if (minFit === 'all') {
    return list;
  }
  if (minFit === 'too_tight') {
    return list.filter((model) => model.fit_level === 'too_tight');
  }

  const threshold = fitRank(minFit);
  return list.filter((model) => {
    const rank = fitRank(model.fit_level);
    return rank >= threshold;
  });
}

export default function App() {
  const [theme, setTheme] = useState(initialTheme);
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [systemState, setSystemState] = useState({
    loading: true,
    error: '',
    payload: null
  });
  const [modelsState, setModelsState] = useState({
    loading: true,
    error: '',
    models: [],
    total: 0,
    returned: 0
  });
  const [selectedModelName, setSelectedModelName] = useState(null);
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    const controller = new AbortController();

    async function loadSystem() {
      setSystemState((prev) => ({ ...prev, loading: true, error: '' }));
      try {
        const payload = await fetchSystemInfo(controller.signal);
        setSystemState({ loading: false, error: '', payload });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setSystemState({
          loading: false,
          error: error instanceof Error ? error.message : 'Unable to load system details.',
          payload: null
        });
      }
    }

    loadSystem();
    return () => controller.abort();
  }, [refreshTick]);

  useEffect(() => {
    const controller = new AbortController();

    async function loadModels() {
      setModelsState((prev) => ({ ...prev, loading: true, error: '' }));
      try {
        const payload = await fetchModels(filters, controller.signal);
        const fetchedModels = Array.isArray(payload.models) ? payload.models : [];
        const fitFiltered = applyClientFitFilter(fetchedModels, filters.minFit);
        const limit = Number.parseInt(filters.limit, 10);
        const models = Number.isFinite(limit) && limit > 0 ? fitFiltered.slice(0, limit) : fitFiltered;
        const serverTotal =
          typeof payload.total_models === 'number' && Number.isFinite(payload.total_models)
            ? payload.total_models
            : fitFiltered.length;
        const total = filters.minFit === 'too_tight' ? fitFiltered.length : serverTotal;
        setModelsState({
          loading: false,
          error: '',
          models,
          total,
          returned: models.length
        });

        setSelectedModelName((current) => {
          if (!current) {
            return models[0]?.name ?? null;
          }
          const stillVisible = models.some((model) => model.name === current);
          return stillVisible ? current : models[0]?.name ?? null;
        });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setModelsState({
          loading: false,
          error: error instanceof Error ? error.message : 'Unable to load model fits.',
          models: [],
          total: 0,
          returned: 0
        });
        setSelectedModelName(null);
      }
    }

    loadModels();
    return () => controller.abort();
  }, [filters, refreshTick]);

  const selectedModel = useMemo(
    () => modelsState.models.find((model) => model.name === selectedModelName) ?? null,
    [modelsState.models, selectedModelName]
  );

  const handleFieldChange = (field) => (event) => {
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value;
    setFilters((current) => ({
      ...current,
      [field]: value
    }));
  };

  const gpus = systemState.payload?.system?.gpus ?? [];
  const gpuSummary =
    gpus.length === 0
      ? 'No GPU detected'
      : gpus
          .map((gpu) => `${gpu.name}${gpu.vram_gb ? ` (${round(gpu.vram_gb, 1)} GB)` : ''}`)
          .join(', ');

  return (
    <div className="page-shell">
      <div className="orb orb-one" aria-hidden="true" />
      <div className="orb orb-two" aria-hidden="true" />

      <header className="hero-shell">
        <div>
          <p className="hero-eyebrow">Local LLM Planning</p>
          <h1>llmfit Dashboard</h1>
          <p className="hero-copy">Hundreds of models &amp; providers. One command to find what runs on your hardware.</p>
        </div>

        <div className="hero-actions">
          <button type="button" onClick={() => setFilters(DEFAULT_FILTERS)} className="btn btn-ghost">
            Reset filters
          </button>
          <button type="button" onClick={() => setRefreshTick((tick) => tick + 1)} className="btn btn-accent">
            Refresh
          </button>
          <button
            type="button"
            onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
            className="btn btn-theme"
          >
            {theme === 'dark' ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
      </header>

      <section className="panel system-panel">
        <div className="panel-heading">
          <h2>System Summary</h2>
          {systemState.payload?.node ? (
            <span className="chip">
              {systemState.payload.node.name} · {systemState.payload.node.os}
            </span>
          ) : null}
        </div>

        {systemState.error ? (
          <div role="alert" className="alert error">
            Could not load system information: {systemState.error}. Make sure `llmfit serve` is running.
          </div>
        ) : null}

        <div className="system-grid" aria-busy={systemState.loading}>
          <SystemCard
            label="CPU"
            value={systemState.payload?.system?.cpu_name ?? 'Loading…'}
            detail={
              systemState.payload?.system?.cpu_cores
                ? `${systemState.payload.system.cpu_cores} cores`
                : undefined
            }
          />
          <SystemCard
            label="Total RAM"
            value={
              systemState.payload?.system?.total_ram_gb
                ? `${round(systemState.payload.system.total_ram_gb, 1)} GB`
                : '—'
            }
          />
          <SystemCard
            label="Available RAM"
            value={
              systemState.payload?.system?.available_ram_gb
                ? `${round(systemState.payload.system.available_ram_gb, 1)} GB`
                : '—'
            }
          />
          <SystemCard
            label="GPU"
            value={gpuSummary}
            detail={
              systemState.payload?.system?.unified_memory
                ? 'Unified memory (CPU + GPU shared)'
                : undefined
            }
          />
        </div>
      </section>

      <section className="panel models-panel">
        <div className="panel-heading">
          <h2>Model Fit Explorer</h2>
          <span className="chip">
            {modelsState.returned} shown / {modelsState.total} matched
          </span>
        </div>

        <div className="filters-shell">
          <label>
            <span>Search</span>
            <input
              type="text"
              value={filters.search}
              onChange={handleFieldChange('search')}
              placeholder="model, provider, use case"
            />
          </label>

          <label>
            <span>Fit filter</span>
            <select value={filters.minFit} onChange={handleFieldChange('minFit')}>
              {FIT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Runtime</span>
            <select value={filters.runtime} onChange={handleFieldChange('runtime')}>
              {RUNTIME_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Use case</span>
            <select value={filters.useCase} onChange={handleFieldChange('useCase')}>
              {USE_CASE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Provider</span>
            <input
              type="text"
              value={filters.provider}
              onChange={handleFieldChange('provider')}
              placeholder="Meta, Qwen, Mistral"
            />
          </label>

          <label>
            <span>Sort</span>
            <select value={filters.sort} onChange={handleFieldChange('sort')}>
              {SORT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Limit</span>
            <select value={String(filters.limit)} onChange={handleFieldChange('limit')}>
              {LIMIT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        {modelsState.error ? (
          <div role="alert" className="alert error">
            Could not load models: {modelsState.error}. Confirm this page is opened from `llmfit serve`.
          </div>
        ) : null}

        <div className="models-layout" aria-busy={modelsState.loading}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Provider</th>
                  <th>Params</th>
                  <th>Fit</th>
                  <th>Mode</th>
                  <th>Runtime</th>
                  <th>Score</th>
                  <th>TPS</th>
                  <th>Mem%</th>
                  <th>Context</th>
                  <th>Release</th>
                </tr>
              </thead>
              <tbody>
                {modelsState.loading ? (
                  <tr>
                    <td colSpan="11" className="table-status">
                      Loading model fit data…
                    </td>
                  </tr>
                ) : null}

                {!modelsState.loading && modelsState.models.length === 0 ? (
                  <tr>
                    <td colSpan="11" className="table-status">
                      No models match the current filters.
                    </td>
                  </tr>
                ) : null}

                {!modelsState.loading
                  ? modelsState.models.map((model) => (
                      <tr
                        key={model.name}
                        className={model.name === selectedModelName ? 'selected' : ''}
                        onClick={() => setSelectedModelName(model.name)}
                      >
                        <td className="model-name">{model.name}</td>
                        <td>{model.provider}</td>
                        <td>{round(model.params_b, 1)}B</td>
                        <td>
                          <span className={fitClass(model.fit_level)}>{model.fit_label}</span>
                        </td>
                        <td>
                          <span className={modeClass(model.run_mode)}>{model.run_mode_label}</span>
                        </td>
                        <td>{model.runtime_label}</td>
                        <td>{round(model.score, 1)}</td>
                        <td>{round(model.estimated_tps, 1)}</td>
                        <td>{round(model.utilization_pct, 1)}</td>
                        <td>{model.context_length?.toLocaleString?.() ?? model.context_length ?? '—'}</td>
                        <td>{model.release_date ?? '—'}</td>
                      </tr>
                    ))
                  : null}
              </tbody>
            </table>
          </div>

          <aside className="details-panel">
            {selectedModel ? (
              <>
                <div className="details-header">
                  <h3>{selectedModel.name}</h3>
                  <span className={fitClass(selectedModel.fit_level)}>{selectedModel.fit_label}</span>
                </div>

                <dl className="details-grid">
                  <div>
                    <dt>Provider</dt>
                    <dd>{selectedModel.provider}</dd>
                  </div>
                  <div>
                    <dt>Run mode</dt>
                    <dd>{selectedModel.run_mode_label}</dd>
                  </div>
                  <div>
                    <dt>Runtime</dt>
                    <dd>{selectedModel.runtime_label}</dd>
                  </div>
                  <div>
                    <dt>Best quant</dt>
                    <dd>{selectedModel.best_quant}</dd>
                  </div>
                  <div>
                    <dt>Memory required</dt>
                    <dd>{round(selectedModel.memory_required_gb, 2)} GB</dd>
                  </div>
                  <div>
                    <dt>Memory available</dt>
                    <dd>{round(selectedModel.memory_available_gb, 2)} GB</dd>
                  </div>
                </dl>

                <div className="metrics-card">
                  <h4>Score Breakdown</h4>
                  <MetricBar label="Quality" value={selectedModel.score_components?.quality} />
                  <MetricBar label="Speed" value={selectedModel.score_components?.speed} />
                  <MetricBar label="Fit" value={selectedModel.score_components?.fit} />
                  <MetricBar label="Context" value={selectedModel.score_components?.context} />
                </div>

                <div className="metrics-card">
                  <h4>Performance</h4>
                  <MetricBar label="Memory Utilization %" value={selectedModel.utilization_pct} />
                  <div className="kpi-grid">
                    <div>
                      <span>Composite score</span>
                      <strong>{round(selectedModel.score, 1)}</strong>
                    </div>
                    <div>
                      <span>Estimated TPS</span>
                      <strong>{round(selectedModel.estimated_tps, 1)}</strong>
                    </div>
                  </div>
                </div>

                {Array.isArray(selectedModel.notes) && selectedModel.notes.length > 0 ? (
                  <div className="metrics-card">
                    <h4>Notes</h4>
                    <ul>
                      {selectedModel.notes.map((note) => (
                        <li key={note}>{note}</li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <p className="muted-copy">No additional notes for this model fit.</p>
                )}
              </>
            ) : (
              <p className="muted-copy">Select a model row to inspect detailed fit diagnostics.</p>
            )}
          </aside>
        </div>
      </section>
    </div>
  );
}
