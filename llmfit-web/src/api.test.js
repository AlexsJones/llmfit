import { appendSimulationParams, buildModelsQuery } from './api';

describe('buildModelsQuery', () => {
  it('maps filter state to API query parameters', () => {
    const query = buildModelsQuery({
      search: 'qwen',
      minFit: 'good',
      runtime: 'llamacpp',
      useCase: 'coding',
      provider: 'Qwen',
      sort: 'tps',
      limit: 25
    });

    const params = new URLSearchParams(query);
    expect(params.get('search')).toBe('qwen');
    expect(params.get('min_fit')).toBe('good');
    expect(params.get('runtime')).toBe('llamacpp');
    expect(params.get('use_case')).toBe('coding');
    expect(params.get('provider')).toBe('Qwen');
    expect(params.get('sort')).toBe('tps');
    expect(params.get('limit')).toBeNull();
    expect(params.get('include_too_tight')).toBe('false');
  });

  it('requests broad fit set for too-tight-only mode', () => {
    const query = buildModelsQuery({
      search: '',
      minFit: 'too_tight',
      runtime: 'any',
      useCase: 'all',
      provider: '',
      sort: 'score',
      limit: 25
    });

    const params = new URLSearchParams(query);
    expect(params.get('min_fit')).toBe('too_tight');
    expect(params.get('include_too_tight')).toBe('true');
    expect(params.get('limit')).toBeNull();
  });

  it('uses broad query mode for all-level filter', () => {
    const query = buildModelsQuery({
      search: '',
      minFit: 'all',
      runtime: 'any',
      useCase: 'all',
      provider: ' ',
      sort: 'score',
      limit: ''
    });

    const params = new URLSearchParams(query);
    expect(params.get('search')).toBeNull();
    expect(params.get('min_fit')).toBe('too_tight');
    expect(params.get('runtime')).toBeNull();
    expect(params.get('use_case')).toBeNull();
    expect(params.get('provider')).toBeNull();
    expect(params.get('sort')).toBe('score');
    expect(params.get('limit')).toBeNull();
    expect(params.get('include_too_tight')).toBe('true');
  });

  it('includes hardware simulation params when present', () => {
    const query = buildModelsQuery(
      {
        search: 'qwen',
        minFit: 'good',
        runtime: 'llamacpp',
        useCase: 'coding',
        provider: 'Qwen',
        sort: 'tps',
        limit: 25
      },
      {
        ramGb: '64',
        vramGb: '24',
        cpuCores: '16'
      }
    );

    const params = new URLSearchParams(query);
    expect(params.get('ram_gb')).toBe('64');
    expect(params.get('vram_gb')).toBe('24');
    expect(params.get('cpu_cores')).toBe('16');
  });
});

describe('appendSimulationParams', () => {
  it('ignores invalid values and preserves zero vram', () => {
    const params = appendSimulationParams(new URLSearchParams(), {
      ramGb: 'abc',
      vramGb: '0',
      cpuCores: '-2'
    });

    expect(params.get('ram_gb')).toBeNull();
    expect(params.get('vram_gb')).toBe('0');
    expect(params.get('cpu_cores')).toBeNull();
  });
});
