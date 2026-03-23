import { describe, it, expect, vi } from 'vitest';
import { fetchModels, fetchSystemInfo, DEFAULT_FILTERS, applyClientFitFilter, applyFilters } from './api';

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn()
}));

const { invoke } = await import('@tauri-apps/api/core');

const sampleModels = [
  { name: 'llama-3-8b', provider: 'Meta', parameter_count: '8B', fit_level: 'perfect', runtime: 'llamacpp', use_case: 'chat', category: 'llm' },
  { name: 'qwen-2-7b', provider: 'Qwen', parameter_count: '7B', fit_level: 'good', runtime: 'llamacpp', use_case: 'coding', category: 'llm' },
  { name: 'mistral-7b', provider: 'Mistral', parameter_count: '7B', fit_level: 'marginal', runtime: 'llamacpp', use_case: 'chat', category: 'llm' },
  { name: 'llama-3-70b', provider: 'Meta', parameter_count: '70B', fit_level: 'too_tight', runtime: 'llamacpp', use_case: 'chat', category: 'llm' },
  { name: 'codellama-13b', provider: 'Meta', parameter_count: '13B', fit_level: 'good', runtime: 'ollama', use_case: 'coding', category: 'code' },
];

describe('applyClientFitFilter', () => {
  it('returns all models when minFit is "all"', () => {
    const result = applyClientFitFilter(sampleModels, 'all');
    expect(result).toHaveLength(5);
  });

  it('filters to only perfect models when minFit is "perfect"', () => {
    const result = applyClientFitFilter(sampleModels, 'perfect');
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('llama-3-8b');
  });

  it('filters to perfect and good when minFit is "good"', () => {
    const result = applyClientFitFilter(sampleModels, 'good');
    expect(result).toHaveLength(3);
    expect(result.map(m => m.name)).toContain('llama-3-8b');
    expect(result.map(m => m.name)).toContain('qwen-2-7b');
    expect(result.map(m => m.name)).toContain('codellama-13b');
  });

  it('filters to perfect, good, and marginal when minFit is "marginal"', () => {
    const result = applyClientFitFilter(sampleModels, 'marginal');
    expect(result).toHaveLength(4);
    expect(result.every(m => m.fit_level !== 'too_tight')).toBe(true);
  });

  it('returns only too_tight models when minFit is "too_tight"', () => {
    const result = applyClientFitFilter(sampleModels, 'too_tight');
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('llama-3-70b');
  });

  it('handles non-array input gracefully', () => {
    expect(applyClientFitFilter(null, 'all')).toEqual([]);
    expect(applyClientFitFilter(undefined, 'good')).toEqual([]);
  });
});

describe('applyFilters', () => {
  it('returns all models with default filters', () => {
    const result = applyFilters(sampleModels, DEFAULT_FILTERS);
    expect(result).toHaveLength(5);
  });

  it('filters by search term matching name', () => {
    const result = applyFilters(sampleModels, { ...DEFAULT_FILTERS, search: 'llama' });
    expect(result).toHaveLength(3);
    expect(result.map(m => m.name)).toContain('llama-3-8b');
    expect(result.map(m => m.name)).toContain('llama-3-70b');
    expect(result.map(m => m.name)).toContain('codellama-13b');
  });

  it('filters by search term matching provider', () => {
    const result = applyFilters(sampleModels, { ...DEFAULT_FILTERS, search: 'qwen' });
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('qwen-2-7b');
  });

  it('filters by search term matching parameter count', () => {
    const result = applyFilters(sampleModels, { ...DEFAULT_FILTERS, search: '70B' });
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('llama-3-70b');
  });

  it('filters by search term matching use case', () => {
    const result = applyFilters(sampleModels, { ...DEFAULT_FILTERS, search: 'coding' });
    expect(result).toHaveLength(2);
    expect(result.map(m => m.name)).toContain('qwen-2-7b');
    expect(result.map(m => m.name)).toContain('codellama-13b');
  });

  it('filters by runtime', () => {
    const result = applyFilters(sampleModels, { ...DEFAULT_FILTERS, runtime: 'ollama' });
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('codellama-13b');
  });

  it('does not filter when runtime is "any"', () => {
    const result = applyFilters(sampleModels, { ...DEFAULT_FILTERS, runtime: 'any' });
    expect(result).toHaveLength(5);
  });

  it('filters by use case', () => {
    const result = applyFilters(sampleModels, { ...DEFAULT_FILTERS, useCase: 'coding' });
    expect(result).toHaveLength(2);
  });

  it('does not filter when useCase is "all"', () => {
    const result = applyFilters(sampleModels, { ...DEFAULT_FILTERS, useCase: 'all' });
    expect(result).toHaveLength(5);
  });

  it('filters by provider', () => {
    const result = applyFilters(sampleModels, { ...DEFAULT_FILTERS, provider: 'Meta' });
    expect(result).toHaveLength(3);
    expect(result.every(m => m.provider === 'Meta')).toBe(true);
  });

  it('applies multiple filters together', () => {
    const result = applyFilters(sampleModels, {
      ...DEFAULT_FILTERS,
      search: 'llama',
      runtime: 'llamacpp',
      useCase: 'chat'
    });
    expect(result).toHaveLength(2);
    expect(result.map(m => m.name)).toContain('llama-3-8b');
    expect(result.map(m => m.name)).toContain('llama-3-70b');
  });
});

describe('fetchModels', () => {
  it('fetches, filters by fit level, applies limit, and filters', async () => {
    invoke.mockImplementation((cmd) => {
      if (cmd === 'get_model_fits') return Promise.resolve(sampleModels);
      if (cmd === 'get_system_specs') return Promise.resolve({ ram_gb: 32, vram_gb: 16 });
      return Promise.resolve(null);
    });

    const result = await fetchModels({
      ...DEFAULT_FILTERS,
      minFit: 'good',
      limit: '2',
      runtime: 'llamacpp'
    });

    expect(result.total_models).toBe(3);
    expect(result.returned_models).toBe(2);
    expect(result.models).toHaveLength(2);
    expect(result.models.every(m => ['perfect', 'good'].includes(m.fit_level))).toBe(true);
    expect(result.models.every(m => m.runtime === 'llamacpp')).toBe(true);
  });

  it('handles empty model list from invoke', async () => {
    invoke.mockImplementation((cmd) => {
      if (cmd === 'get_model_fits') return Promise.resolve([]);
      if (cmd === 'get_system_specs') return Promise.resolve({});
      return Promise.resolve(null);
    });

    const result = await fetchModels(DEFAULT_FILTERS);
    expect(result.models).toEqual([]);
    expect(result.total_models).toBe(0);
  });
});

describe('fetchSystemInfo', () => {
  it('returns system specs with node info', async () => {
    invoke.mockResolvedValue({ ram_gb: 64, vram_gb: 24, cpu_cores: 16 });
    const result = await fetchSystemInfo();
    expect(result.node.name).toBe('desktop');
    expect(result.system.ram_gb).toBe(64);
  });
});

describe('DEFAULT_FILTERS', () => {
  it('has expected default values', () => {
    expect(DEFAULT_FILTERS.minFit).toBe('marginal');
    expect(DEFAULT_FILTERS.runtime).toBe('any');
    expect(DEFAULT_FILTERS.useCase).toBe('all');
    expect(DEFAULT_FILTERS.sort).toBe('score');
    expect(DEFAULT_FILTERS.limit).toBe('50');
  });
});
