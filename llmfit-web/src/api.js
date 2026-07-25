export const DEFAULT_FILTERS = {
  search: '',
  minFit: 'marginal',
  runtime: 'any',
  useCase: 'all',
  provider: '',
  sort: 'score',
  limit: '50'
};

// -- Tauri detection and invoke helper ---------------------------------------
const isTauri = typeof window !== 'undefined' &&
  (window.__TAURI_INTERNALS__ || window.__TAURI__);

async function tauriInvoke(cmd, args = {}) {
  if (window.__TAURI_INTERNALS__) {
    return window.__TAURI_INTERNALS__.invoke(cmd, args);
  }
  if (window.__TAURI__ && window.__TAURI__.core) {
    return window.__TAURI__.core.invoke(cmd, args);
  }
  throw new Error('Tauri IPC not available');
}

// -- Simulation params helper ------------------------------------------------
function buildSimArgs(simulation) {
  const args = {};
  const ramGb = parseOptionalNumber(simulation?.ramGb);
  if (ramGb !== null && ramGb > 0) args.ram_gb = ramGb;
  const vramGb = parseOptionalNumber(simulation?.vramGb);
  if (vramGb !== null && vramGb >= 0) args.vram_gb = vramGb;
  const cpuCores = parseOptionalNumber(simulation?.cpuCores);
  if (cpuCores !== null && cpuCores > 0) args.cpu_cores = Math.trunc(cpuCores);
  return Object.keys(args).length > 0 ? { sim: args } : {};
}

// -- Shared helpers ----------------------------------------------------------
function trimOrEmpty(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function parseOptionalNumber(value) {
  const raw = trimOrEmpty(String(value ?? ''));
  if (!raw) return null;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return null;
  return parsed;
}

async function parseJsonOrThrow(response) {
  let payload;
  try { payload = await response.json(); }
  catch (err) { throw new Error('Server returned an invalid JSON response.'); }
  if (!response.ok) {
    const message = payload?.error || `Request failed with status ${response.status}.`;
    throw new Error(message);
  }
  return payload;
}

export function appendSimulationParams(params, simulation = {}) {
  const ramGb = parseOptionalNumber(simulation.ramGb);
  if (ramGb !== null && ramGb > 0) params.set('ram_gb', String(ramGb));
  const vramGb = parseOptionalNumber(simulation.vramGb);
  if (vramGb !== null && vramGb >= 0) params.set('vram_gb', String(vramGb));
  const cpuCores = parseOptionalNumber(simulation.cpuCores);
  if (cpuCores !== null && cpuCores > 0) params.set('cpu_cores', String(Math.trunc(cpuCores)));
  return params;
}

export function buildModelsQuery(filters, simulation = {}) {
  const params = new URLSearchParams();
  const search = trimOrEmpty(filters.search);
  if (search) params.set('search', search);
  const provider = trimOrEmpty(filters.provider);
  if (provider) params.set('provider', provider);
  const minFit = filters.minFit || 'marginal';
  if (minFit === 'all' || minFit === 'too_tight') {
    params.set('min_fit', 'too_tight');
    params.set('include_too_tight', 'true');
  } else {
    params.set('min_fit', minFit);
    params.set('include_too_tight', 'false');
  }
  if (filters.runtime && filters.runtime !== 'any') params.set('runtime', filters.runtime);
  if (filters.useCase && filters.useCase !== 'all') params.set('use_case', filters.useCase);
  if (filters.sort) params.set('sort', filters.sort);
  const license = trimOrEmpty(filters.license);
  if (license) params.set('license', license);
  const maxContext = trimOrEmpty(String(filters.maxContext || ''));
  if (maxContext) {
    const parsed = Number.parseInt(maxContext, 10);
    if (Number.isFinite(parsed) && parsed > 0) params.set('max_context', String(parsed));
  }
  appendSimulationParams(params, simulation);
  return params.toString();
}

// -- Public API ---------------------------------------------------------------

export async function fetchSystemInfo(simulation = {}, signal) {
  if (isTauri) {
    const simArgs = buildSimArgs(simulation);
    return tauriInvoke('get_system_specs', simArgs);
  }
  const query = appendSimulationParams(new URLSearchParams(), simulation).toString();
  const path = query ? `/api/v1/system?${query}` : '/api/v1/system';
  const response = await fetch(path, { signal });
  return parseJsonOrThrow(response);
}

export async function fetchModels(filters, simulation = {}, signal) {
  if (isTauri) {
    const simArgs = buildSimArgs(simulation);
    const result = await tauriInvoke('get_models', simArgs);
    // If there's a search query or runtime filter, we can use the search_models
    // command or just filter client-side — we get all models, so client filter is fine.
    return result;
  }
  const query = buildModelsQuery(filters, simulation);
  const path = query ? `/api/v1/models?${query}` : '/api/v1/models';
  const response = await fetch(path, { signal });
  return parseJsonOrThrow(response);
}

export async function fetchRuntimes(signal) {
  if (isTauri) {
    return tauriInvoke('get_runtimes');
  }
  const response = await fetch('/api/v1/runtimes', { signal });
  return parseJsonOrThrow(response);
}

export async function fetchInstalled(signal) {
  if (isTauri) {
    return tauriInvoke('get_installed');
  }
  const response = await fetch('/api/v1/installed', { signal });
  return parseJsonOrThrow(response);
}

export async function startDownload(model, runtime, signal) {
  if (isTauri) {
    return tauriInvoke('start_pull', { modelTag: model });
  }
  const response = await fetch('/api/v1/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, runtime }),
    signal
  });
  return parseJsonOrThrow(response);
}

export async function fetchDownloadStatus(id, signal) {
  if (isTauri) {
    return tauriInvoke('poll_pull');
  }
  const response = await fetch(`/api/v1/download/${encodeURIComponent(id)}/status`, { signal });
  return parseJsonOrThrow(response);
}

export async function openExternal(url) {
  if (isTauri) {
    return tauriInvoke('open_url', { url });
  }
  window.open(url, '_blank');
}

export async function isOllamaAvailable(signal) {
  if (isTauri) {
    return tauriInvoke('is_ollama_available');
  }
  try {
    const response = await fetch('/api/v1/runtimes', { signal });
    const data = await parseJsonOrThrow(response);
    const ollama = (data.runtimes ?? []).find((r) => r.name === 'ollama');
    return ollama?.installed === true;
  } catch {
    return false;
  }
}

export async function fetchPlanEstimate(
  { model, context, quant, kv_quant, target_tps },
  simulation = {},
  signal
) {
  if (isTauri) {
    const simArgs = buildSimArgs(simulation);
    return tauriInvoke('estimate_plan', {
      modelName: model,
      context,
      quant: quant || null,
      kvQuantStr: kv_quant || null,
      targetTps: target_tps || null,
      ...simArgs,
    });
  }
  const body = { model, context, quant, kv_quant, target_tps };
  const ramGb = parseOptionalNumber(simulation.ramGb);
  if (ramGb !== null && ramGb > 0) body.ram_gb = ramGb;
  const vramGb = parseOptionalNumber(simulation.vramGb);
  if (vramGb !== null && vramGb >= 0) body.vram_gb = vramGb;
  const cpuCores = parseOptionalNumber(simulation.cpuCores);
  if (cpuCores !== null && cpuCores > 0) body.cpu_cores = Math.trunc(cpuCores);
  const response = await fetch('/api/v1/plan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal
  });
  return parseJsonOrThrow(response);
}
