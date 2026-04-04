import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';

function jsonResponse(payload, { ok = true, status = 200 } = {}) {
  return {
    ok,
    status,
    json: async () => payload
  };
}

const systemPayload = {
  node: { name: 'local-node', os: 'darwin' },
  system: {
    cpu_name: 'Apple M3 Max',
    cpu_cores: 14,
    total_ram_gb: 64,
    available_ram_gb: 51.4,
    gpus: [{ name: 'Apple GPU', vram_gb: 64 }],
    unified_memory: true
  }
};

const modelsPayload = {
  total_models: 2,
  returned_models: 2,
  models: [
    {
      name: 'Qwen/Qwen2.5-7B-Instruct',
      provider: 'Qwen',
      params_b: 7,
      fit_level: 'good',
      fit_label: 'Good',
      run_mode: 'gpu',
      run_mode_label: 'GPU',
      runtime: 'llamacpp',
      runtime_label: 'llama.cpp',
      score: 86,
      estimated_tps: 34.5,
      utilization_pct: 58.9,
      memory_required_gb: 7.4,
      memory_available_gb: 12.5,
      context_length: 32768,
      best_quant: 'Q5_K_M',
      release_date: '2025-02-01',
      score_components: {
        quality: 87,
        speed: 80,
        fit: 90,
        context: 85
      },
      notes: ['Runs smoothly on most laptops']
    },
    {
      name: 'meta-llama/Llama-3.1-8B-Instruct',
      provider: 'Meta',
      params_b: 8,
      fit_level: 'marginal',
      fit_label: 'Marginal',
      run_mode: 'cpu_offload',
      run_mode_label: 'CPU Offload',
      runtime: 'llamacpp',
      runtime_label: 'llama.cpp',
      score: 74,
      estimated_tps: 19.2,
      utilization_pct: 87.5,
      memory_required_gb: 10.1,
      memory_available_gb: 11.5,
      context_length: 8192,
      best_quant: 'Q4_K_M',
      release_date: '2024-11-10',
      score_components: {
        quality: 78,
        speed: 66,
        fit: 72,
        context: 74
      },
      notes: []
    },
    {
      name: 'LargeModel/220B-Preview',
      provider: 'Example',
      params_b: 220,
      fit_level: 'too_tight',
      fit_label: 'Too Tight',
      run_mode: 'cpu_only',
      run_mode_label: 'CPU Only',
      runtime: 'llamacpp',
      runtime_label: 'llama.cpp',
      score: 44,
      estimated_tps: 1.9,
      utilization_pct: 165.2,
      memory_required_gb: 92.4,
      memory_available_gb: 56.0,
      context_length: 32768,
      best_quant: 'Q2_K',
      release_date: '2025-01-02',
      score_components: {
        quality: 95,
        speed: 8,
        fit: 10,
        context: 62
      },
      notes: ['Requires substantially more memory than this system']
    }
  ]
};

describe('App', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it('renders models and refetches when sort changes', async () => {
    const fetchMock = vi.fn((url) => {
      const target = String(url);
      if (target.includes('/api/v1/system')) {
        return Promise.resolve(jsonResponse(systemPayload));
      }
      if (target.includes('/api/v1/models')) {
        return Promise.resolve(jsonResponse(modelsPayload));
      }
      return Promise.reject(new Error(`Unexpected URL: ${target}`));
    });

    vi.stubGlobal('fetch', fetchMock);

    render(<App />);

    await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct');
    expect(screen.getByText('meta-llama/Llama-3.1-8B-Instruct')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Sort'), { target: { value: 'tps' } });

    await waitFor(() => {
      const queriedWithTps = fetchMock.mock.calls.some(([url]) => String(url).includes('sort=tps'));
      expect(queriedWithTps).toBe(true);
    });
  });

  it('opens detail diagnostics when a model row is selected', async () => {
    vi.stubGlobal('fetch', vi.fn((url) => {
      const target = String(url);
      if (target.includes('/api/v1/system')) {
        return Promise.resolve(jsonResponse(systemPayload));
      }
      return Promise.resolve(jsonResponse(modelsPayload));
    }));

    render(<App />);

    const modelCell = (await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct'))[0];
    fireEvent.click(modelCell);

    expect(screen.getByText('Score Breakdown')).toBeInTheDocument();
    expect(screen.getByText('Runs smoothly on most laptops')).toBeInTheDocument();
  });

  it('shows actionable error message when model fetch fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url) => {
        const target = String(url);
        if (target.includes('/api/v1/system')) {
          return Promise.resolve(jsonResponse(systemPayload));
        }
        return Promise.resolve(jsonResponse({ error: 'backend unavailable' }, { ok: false, status: 500 }));
      })
    );

    render(<App />);

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Could not load models: backend unavailable');
    expect(alert).toHaveTextContent('llmfit serve');
  });

  it('switches theme via theme picker', async () => {
    vi.stubGlobal('fetch', vi.fn((url) => {
      const target = String(url);
      if (target.includes('/api/v1/system')) {
        return Promise.resolve(jsonResponse(systemPayload));
      }
      return Promise.resolve(jsonResponse(modelsPayload));
    }));

    render(<App />);

    const picker = await screen.findByLabelText('Theme');
    fireEvent.change(picker, { target: { value: 'catppuccin-mocha' } });

    expect(document.documentElement.dataset.theme).toBe('catppuccin-mocha');
  });

  it('can filter to too-tight only', async () => {
    vi.stubGlobal('fetch', vi.fn((url) => {
      const target = String(url);
      if (target.includes('/api/v1/system')) {
        return Promise.resolve(jsonResponse(systemPayload));
      }
      return Promise.resolve(jsonResponse(modelsPayload));
    }));

    render(<App />);

    await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct');
    fireEvent.change(screen.getByLabelText('Fit filter'), { target: { value: 'too_tight' } });

    expect(await screen.findAllByText('LargeModel/220B-Preview')).not.toHaveLength(0);
    expect(screen.queryAllByText('Qwen/Qwen2.5-7B-Instruct')).toHaveLength(0);
  });
});
