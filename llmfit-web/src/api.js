const API_BASE = import.meta.env.VITE_API_BASE || '';

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

  const limit = Number.parseInt(filters.limit, 10);
  if (!needsClientFitProcessing && Number.isFinite(limit) && limit > 0) {
    params.set('limit', String(limit));
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
  try {
    const response = await fetch(`${API_BASE}/api/v1/system`, { signal });
    return await parseJsonOrThrow(response);
  } catch (err) {
    console.warn("Using mock system data due to fetch error:", err);
    return {
      node: { name: "MacBook Pro", os: "macOS 14.2" },
      system: {
        cpu_name: "Apple M3 Max",
        cpu_cores: 16,
        total_ram_gb: 128.0,
        available_ram_gb: 98.4,
        unified_memory: true,
        gpus: [{ name: "Apple M3 Max GPU", vram_gb: 128.0 }]
      }
    };
  }
}

export async function fetchModels(filters, signal) {
  try {
    const query = buildModelsQuery(filters);
    const path = query ? `${API_BASE}/api/v1/models?${query}` : `${API_BASE}/api/v1/models`;
    const response = await fetch(path, { signal });
    return await parseJsonOrThrow(response);
  } catch (err) {
    console.warn("Using mock model data due to fetch error:", err);
    const mockModels = [
      {
        name: "Llama-3-70B-Instruct",
        provider: "Meta",
        params_b: 70.0,
        fit_level: "perfect",
        fit_label: "Perfect",
        run_mode: "gpu",
        run_mode_label: "GPU",
        runtime_label: "llama.cpp",
        score: 98.5,
        estimated_tps: 34.2,
        utilization_pct: 42.1,
        context_length: 8192,
        release_date: "2024-04-18",
        best_quant: "Q4_K_M",
        memory_required_gb: 41.2,
        memory_available_gb: 98.4,
        score_components: { quality: 99, speed: 94, fit: 100, context: 85 },
        notes: ["Runs entirely in VRAM. Outstanding performance for complex reasoning.", "Quantization recommended over FP16."]
      },
      {
        name: "Mistral-Large-2407",
        provider: "Mistral AI",
        params_b: 123.0,
        fit_level: "good",
        fit_label: "Good Fit",
        run_mode: "cpu_offload",
        run_mode_label: "GPU + CPU",
        runtime_label: "llama.cpp",
        score: 87.2,
        estimated_tps: 18.5,
        utilization_pct: 85.0,
        context_length: 32768,
        release_date: "2024-07-24",
        best_quant: "Q4_K_M",
        memory_required_gb: 82.5,
        memory_available_gb: 98.4,
        score_components: { quality: 96, speed: 72, fit: 85, context: 99 },
        notes: ["Partial offload to CPU required. Speed will be good but not instant.", "Massive context capabilities available."]
      },
      {
        name: "Qwen2.5-14B-Coder",
        provider: "Alibaba",
        params_b: 14.0,
        fit_level: "perfect",
        fit_label: "Perfect",
        run_mode: "gpu",
        run_mode_label: "GPU",
        runtime_label: "vLLM",
        score: 94.8,
        estimated_tps: 89.1,
        utilization_pct: 12.5,
        context_length: 32768,
        release_date: "2024-09-18",
        best_quant: "AWQ",
        memory_required_gb: 8.5,
        memory_available_gb: 98.4,
        score_components: { quality: 91, speed: 99, fit: 100, context: 95 },
        notes: ["Unreal coding speed. Can serve multiple concurrent users comfortably.", "Highly recommended for coding workflows."]
      },
      {
        name: "Gemma-2-27B-it",
        provider: "Google",
        params_b: 27.0,
        fit_level: "good",
        fit_label: "Good Fit",
        run_mode: "gpu",
        run_mode_label: "GPU",
        runtime_label: "vLLM",
        score: 91.0,
        estimated_tps: 54.3,
        utilization_pct: 22.0,
        context_length: 8192,
        release_date: "2024-06-27",
        best_quant: "GPTQ",
        memory_required_gb: 16.2,
        memory_available_gb: 98.4,
        score_components: { quality: 93, speed: 88, fit: 100, context: 82 },
        notes: ["State of the art in its weight class. Very fast inference mode available."]
      },
      {
        name: "DeepSeek-V3-Base",
        provider: "DeepSeek",
        params_b: 671.0,
        fit_level: "too_tight",
        fit_label: "Too Tight",
        run_mode: "cpu_only",
        run_mode_label: "CPU Only",
        runtime_label: "llama.cpp",
        score: 45.1,
        estimated_tps: 1.2,
        utilization_pct: 185.0,
        context_length: 128000,
        release_date: "2024-12-26",
        best_quant: "Q4_K_M",
        memory_required_gb: 385.0,
        memory_available_gb: 98.4,
        score_components: { quality: 99, speed: 10, fit: 25, context: 100 },
        notes: ["Model exceeds available system memory by nearly 4x. Will swap heavily and be unusable."]
      }
    ];
    return {
      models: mockModels,
      total_models: mockModels.length
    };
  }
}
