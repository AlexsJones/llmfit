import { invoke } from '@tauri-apps/api/core';

export const DEFAULT_FILTERS = {
  search: '',
  minFit: 'marginal',
  runtime: 'any',
  useCase: 'all',
  provider: '',
  sort: 'score',
  limit: '50'
};

export async function fetchSystemInfo(signal) {
  const specs = await invoke('get_system_specs');
  return {
    node: { name: 'desktop', os: getOs() },
    system: specs
  };
}

export async function fetchModels(filters, signal) {
  const allFits = await invoke('get_model_fits');
  const fetchedModels = Array.isArray(allFits) ? allFits : [];
  const fitFiltered = applyClientFitFilter(fetchedModels, filters.minFit);
  const limit = Number.parseInt(filters.limit, 10);
  const models = Number.isFinite(limit) && limit > 0 ? fitFiltered.slice(0, limit) : fitFiltered;
  const filtered = applyFilters(models, filters);

  return {
    node: { name: 'desktop', os: getOs() },
    system: await invoke('get_system_specs'),
    total_models: fitFiltered.length,
    returned_models: filtered.length,
    filters: filters,
    models: filtered
  };
}

function getOs() {
  if (typeof navigator !== 'undefined') {
    const ua = navigator.userAgent;
    if (ua.includes('Mac')) return 'macOS';
    if (ua.includes('Win')) return 'Windows';
    if (ua.includes('Linux')) return 'Linux';
    return ua;
  }
  return 'unknown';
}

function fitRank(level) {
  switch (level) {
    case 'perfect': return 3;
    case 'good': return 2;
    case 'marginal': return 1;
    case 'too_tight': return 0;
    default: return -1;
  }
}

export function applyClientFitFilter(models, minFit) {
  const list = Array.isArray(models) ? models : [];
  if (minFit === 'all') return list;
  if (minFit === 'too_tight') return list.filter(m => m.fit_level === 'too_tight');
  const threshold = fitRank(minFit);
  return list.filter(model => fitRank(model.fit_level) >= threshold);
}

export function applyFilters(fits, filters) {
  let result = fits;

  if (filters.search) {
    const s = filters.search.toLowerCase();
    result = result.filter(f =>
      (f.name || '').toLowerCase().includes(s) ||
      (f.provider || '').toLowerCase().includes(s) ||
      (f.parameter_count || '').toLowerCase().includes(s) ||
      (f.use_case || '').toLowerCase().includes(s) ||
      (f.category || '').toLowerCase().includes(s)
    );
  }

  if (filters.runtime && filters.runtime !== 'any') {
    result = result.filter(f => f.runtime === filters.runtime);
  }

  if (filters.useCase && filters.useCase !== 'all') {
    const uc = filters.useCase.toLowerCase();
    result = result.filter(f => (f.use_case || '').toLowerCase() === uc);
  }

  if (filters.provider) {
    const p = filters.provider.toLowerCase();
    result = result.filter(f => (f.provider || '').toLowerCase().includes(p));
  }

  return result;
}
