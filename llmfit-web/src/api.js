export const DEFAULT_FILTERS = {
  search: '',
  minFit: 'marginal',
  runtime: 'any',
  useCase: 'all',
  provider: '',
  sort: 'score',
  limit: '50'
};

function trimOrEmpty(value) {
  return typeof value === 'string' ? value.trim() : '';
}

export function buildModelsQuery(filters) {
  const params = new URLSearchParams();

  const search = trimOrEmpty(filters.search);
  if (search) {
    params.set('search', search);
  }

  const provider = trimOrEmpty(filters.provider);
  if (provider) {
    params.set('provider', provider);
  }

  const minFit = filters.minFit || 'marginal';
  const needsClientFitProcessing = minFit === 'too_tight';

  if (minFit === 'all' || minFit === 'too_tight') {
    // too_tight is the lowest level, so this returns all fits.
    // We post-filter client-side for the too-tight-only mode.
    params.set('min_fit', 'too_tight');
    params.set('include_too_tight', 'true');
  } else {
    params.set('min_fit', minFit);
    params.set('include_too_tight', 'false');
  }

  if (filters.runtime && filters.runtime !== 'any') {
    params.set('runtime', filters.runtime);
  }

  if (filters.useCase && filters.useCase !== 'all') {
    params.set('use_case', filters.useCase);
  }

  if (filters.sort) {
    params.set('sort', filters.sort);
  }


  // New server-side filter params
  const license = trimOrEmpty(filters.license);
  if (license) {
    params.set('license', license);
  }

  const maxContext = trimOrEmpty(String(filters.maxContext || ''));
  if (maxContext) {
    const parsed = Number.parseInt(maxContext, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      params.set('max_context', String(parsed));
    }
  }

  return params.toString();
}

async function parseJsonOrThrow(response) {
  let payload;
  try {
    payload = await response.json();
  } catch (err) {
    throw new Error('Server returned an invalid JSON response.');
  }

  if (!response.ok) {
    const message = payload?.error || `Request failed with status ${response.status}.`;
    throw new Error(message);
  }

  return payload;
}

export async function fetchSystemInfo(signal) {
  const response = await fetch('/api/v1/system', { signal });
  return parseJsonOrThrow(response);
}

export async function fetchModels(filters, signal) {
  const query = buildModelsQuery(filters);
  const path = query ? `/api/v1/models?${query}` : '/api/v1/models';
  const response = await fetch(path, { signal });
  return parseJsonOrThrow(response);
}

export async function fetchRuntimes(signal) {
  const response = await fetch('/api/v1/runtimes', { signal });
  return parseJsonOrThrow(response);
}

export async function fetchInstalled(signal) {
  const response = await fetch('/api/v1/installed', { signal });
  return parseJsonOrThrow(response);
}

export async function startDownload(model, runtime, signal) {
  const response = await fetch('/api/v1/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, runtime }),
    signal
  });
  return parseJsonOrThrow(response);
}

export async function fetchDownloadStatus(id, signal) {
  const response = await fetch(`/api/v1/download/${encodeURIComponent(id)}/status`, { signal });
  return parseJsonOrThrow(response);
}

export async function fetchPlanEstimate({ model, context, quant, target_tps }, signal) {
  const response = await fetch('/api/v1/plan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, context, quant, target_tps }),
    signal
  });
  return parseJsonOrThrow(response);
}
