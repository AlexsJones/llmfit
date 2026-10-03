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
      usable_context: 16384,
      effective_context_length: 8192,
      best_quant: 'Q5_K_M',
      release_date: '2025-02-01',
      disk_size_gb: 5.4,
      estimate_confidence: 'measured_community',
      estimate_confidence_label: 'measured (community)',
      measured_tps: {
        tok_s: 41.2,
        sample_count: 3,
        hardware_label: 'Apple M3 Max',
        source: 'community'
      },
      estimate_basis: {
        method: 'gpu_bandwidth_roofline',
        gpu_bandwidth_gbps: 400,
        ddr_bandwidth_gbps: null,
        efficiency: 0.55,
        assumed_context: 8192,
        local_calibration: null
      },
      verify_command: 'llama-bench -m qwen2.5-7b-q5_k_m.gguf',
      prefill_tps: null,
      ttft_ms: null,
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
      installed: true,
      gguf_sources: [{ repo: 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', provider: 'bartowski' }],
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

const concurrencyPayload = {
  model: 'Qwen/Qwen2.5-7B-Instruct',
  run_mode: 'gpu',
  fit_level: 'good',
  target_users: null,
  max_context_for_target: null,
  estimate: {
    pool_gb: 12.5,
    weights_resident_gb: 5.6,
    kv_budget_gb: 6.9,
    quant: 'Q5_K_M',
    kv_quant: 'fp16',
    native_context: 32768,
    ladder: [
      { requested_context: 4096, effective_context: 4096, clamped: false, per_session_kv_gb: 0.21, max_sessions: 32 },
      { requested_context: 32768, effective_context: 32768, clamped: false, per_session_kv_gb: 1.75, max_sessions: 3 },
      { requested_context: 65536, effective_context: 32768, clamped: true, per_session_kv_gb: 1.75, max_sessions: 3 }
    ]
  }
};

const storagePayload = {
  system: systemPayload.system,
  storage: {
    estimate_notice: 'Estimates only.',
    keep_requested: 3,
    selected_count: 1,
    eligible_count: 1,
    models: [
      {
        name: 'Qwen/Qwen2.5-7B-Instruct',
        best_quant: 'Q5_K_M',
        fit_level: 'Good',
        runtime: 'LlamaCpp',
        score: 86,
        disk_size_gb: 5.4,
        effective_context_length: 8192
      }
    ],
    library_gb: 5.4,
    download_scratch_gb: 5.4,
    need_gb: 110.8,
    target_capacity_gb: 130.4,
    minimum_ssd_gb: 256,
    suggested_ssd_gb: 512,
    warnings: []
  }
};

function installFetchMock() {
  const fetchMock = vi.fn((url) => {
    const target = String(url);
    if (target.includes('/api/v1/concurrency')) {
      const users = new URL(target, 'http://localhost').searchParams.get('users');
      return Promise.resolve(
        jsonResponse(
          users
            ? { ...concurrencyPayload, target_users: Number(users), max_context_for_target: 16384 }
            : concurrencyPayload
        )
      );
    }
    if (target.includes('/api/v1/storage')) {
      return Promise.resolve(jsonResponse(storagePayload));
    }
    if (target.includes('/api/v1/system')) {
      return Promise.resolve(jsonResponse(systemPayload));
    }
    if (target.includes('/api/v1/models')) {
      return Promise.resolve(jsonResponse(modelsPayload));
    }
    return Promise.reject(new Error(`Unexpected URL: ${target}`));
  });

  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

const originalNavigatorLanguage = Object.getOwnPropertyDescriptor(window.navigator, 'language');

function setNavigatorLanguage(language) {
  Object.defineProperty(window.navigator, 'language', {
    configurable: true,
    value: language
  });
}

describe('App', () => {
  beforeEach(() => {
    setNavigatorLanguage('en-US');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    window.localStorage.clear();
    if (originalNavigatorLanguage) {
      Object.defineProperty(window.navigator, 'language', originalNavigatorLanguage);
    }
  });

  it('renders models and refetches when sort changes', async () => {
    const fetchMock = installFetchMock();

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
    installFetchMock();

    render(<App />);

    const modelCell = (await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct'))[0];
    fireEvent.click(modelCell);

    expect(screen.getByText('Score Breakdown')).toBeInTheDocument();
    expect(screen.getByText('Runs smoothly on most laptops')).toBeInTheDocument();
  });

  it('surfaces measured throughput, usable context and the estimate basis', async () => {
    installFetchMock();

    render(<App />);

    const modelCell = (await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct'))[0];
    fireEvent.click(modelCell);

    // Table: memory-capped context and the measured figure outrank the estimate.
    expect(screen.getByText('32,768 \u2192 16,384')).toBeInTheDocument();
    expect(screen.getAllByText('41.2').length).toBeGreaterThan(0);

    // Detail panel.
    expect(screen.getByText('Measured (community)')).toBeInTheDocument();
    expect(screen.getByText('Community leaderboard: 3 run(s) on Apple M3 Max')).toBeInTheDocument();
    expect(screen.getByText('GPU bandwidth roofline')).toBeInTheDocument();
    expect(screen.getByText('llama-bench -m qwen2.5-7b-q5_k_m.gguf')).toBeInTheDocument();
    expect(screen.getByText('5.4 GB')).toBeInTheDocument();
  });

  it('loads the concurrent-session ladder for the selected model', async () => {
    const fetchMock = installFetchMock();

    render(<App />);

    const modelCell = (await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct'))[0];
    fireEvent.click(modelCell);

    expect(await screen.findByText('Concurrent Sessions')).toBeInTheDocument();
    expect(await screen.findByText('32')).toBeInTheDocument();
    // The clamped 64k rung repeats the 32k row and is collapsed.
    expect(screen.queryByText('clamped', { exact: false })).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes('/api/v1/concurrency?model=Qwen%2FQwen2.5-7B-Instruct&kv_quant=fp16')
      )
    ).toBe(true);
  });

  it('sends the target session count on submit', async () => {
    const fetchMock = installFetchMock();

    render(<App />);

    fireEvent.click((await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct'))[0]);
    fireEvent.change(await screen.findByLabelText('Target sessions'), {
      target: { value: '8' }
    });
    fireEvent.click(screen.getByRole('button', { name: 'Estimate capacity' }));

    expect(
      await screen.findByText('8 concurrent sessions fit up to a 16k context.')
    ).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).includes('/api/v1/concurrency') && String(url).includes('users=8'))
    ).toBe(true);
  });

  it('estimates storage for a model library', async () => {
    const fetchMock = installFetchMock();

    render(<App />);

    fireEvent.click(await screen.findByRole('button', { name: 'Open planner' }));
    fireEvent.change(screen.getByLabelText('Models to keep'), { target: { value: '5' } });
    fireEvent.click(screen.getByRole('button', { name: 'Estimate storage' }));

    expect(await screen.findByText('512 GB')).toBeInTheDocument();
    expect(screen.getByText('110.8 GB')).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).includes('/api/v1/storage?keep=5'))
    ).toBe(true);
  });

  it('filters to installed models and by parameter range', async () => {
    installFetchMock();

    render(<App />);

    await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct');
    fireEvent.click(screen.getByRole('button', { name: /More filters/ }));

    fireEvent.change(screen.getByLabelText('Availability'), { target: { value: 'installed' } });
    await waitFor(() => {
      expect(screen.queryByText('Qwen/Qwen2.5-7B-Instruct')).not.toBeInTheDocument();
    });
    expect(screen.getAllByText('meta-llama/Llama-3.1-8B-Instruct').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Installed').length).toBeGreaterThan(0);

    fireEvent.change(screen.getByLabelText('Availability'), { target: { value: 'all' } });
    fireEvent.change(screen.getByLabelText('Maximum parameters (billions)'), {
      target: { value: '7.5' }
    });
    await waitFor(() => {
      expect(screen.queryByText('meta-llama/Llama-3.1-8B-Instruct')).not.toBeInTheDocument();
    });
    expect(screen.getAllByText('Qwen/Qwen2.5-7B-Instruct').length).toBeGreaterThan(0);
  });

  it('flags a server running under a hardware profile', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url) => {
        const target = String(url);
        if (target.includes('/api/v1/system')) {
          return Promise.resolve(jsonResponse({ ...systemPayload, profile: 'dgx-spark' }));
        }
        return Promise.resolve(jsonResponse(modelsPayload));
      })
    );

    render(<App />);

    expect(await screen.findByText('Profile: dgx-spark')).toBeInTheDocument();
  });

  it('drops the storage estimate when the simulated hardware changes', async () => {
    installFetchMock();

    render(<App />);

    fireEvent.click(await screen.findByRole('button', { name: 'Open planner' }));
    fireEvent.click(screen.getByRole('button', { name: 'Estimate storage' }));
    expect(await screen.findByText('512 GB')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('RAM (GB)'), { target: { value: '32' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply simulation' }));

    await waitFor(() => {
      expect(screen.queryByText('512 GB')).not.toBeInTheDocument();
    });
  });

  it('ignores a concurrency response for superseded inputs', async () => {
    const baseFetch = installFetchMock();
    let releaseFirst;
    let concurrencyCalls = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn((url, init) => {
        if (String(url).includes('/api/v1/concurrency')) {
          concurrencyCalls += 1;
          if (concurrencyCalls === 1) {
            const stale = {
              ...concurrencyPayload,
              estimate: {
                ...concurrencyPayload.estimate,
                ladder: [
                  { requested_context: 4096, effective_context: 4096, clamped: false, per_session_kv_gb: 0.21, max_sessions: 999 }
                ]
              }
            };
            return new Promise((resolve) => {
              releaseFirst = () => resolve(jsonResponse(stale));
            });
          }
        }
        return baseFetch(url, init);
      })
    );

    render(<App />);

    fireEvent.click((await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct'))[0]);
    await waitFor(() => expect(releaseFirst).toBeTypeOf('function'));
    fireEvent.change(screen.getByLabelText('KV cache'), { target: { value: 'q8_0' } });
    expect(await screen.findByText('32')).toBeInTheDocument();

    releaseFirst();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryByText('999')).not.toBeInTheDocument();
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
    installFetchMock();

    render(<App />);

    const picker = await screen.findByLabelText('Theme');
    fireEvent.change(picker, { target: { value: 'catppuccin-mocha' } });

    expect(document.documentElement.dataset.theme).toBe('catppuccin-mocha');
  });

  it('can filter to too-tight only', async () => {
    installFetchMock();

    render(<App />);

    await screen.findAllByText('Qwen/Qwen2.5-7B-Instruct');
    fireEvent.change(screen.getByLabelText('Fit filter'), { target: { value: 'too_tight' } });

    expect(await screen.findAllByText('LargeModel/220B-Preview')).not.toHaveLength(0);
    expect(screen.queryAllByText('Qwen/Qwen2.5-7B-Instruct')).toHaveLength(0);
  });

  it('defaults to Chinese when navigator language is Chinese', async () => {
    setNavigatorLanguage('zh-CN');
    installFetchMock();

    render(<App />);

    expect(await screen.findByRole('heading', { name: 'llmfit 控制台' })).toBeInTheDocument();
    expect(screen.getByText('模型适配分析')).toBeInTheDocument();
    expect(window.localStorage.getItem('llmfit.locale')).toBe('zh-CN');
  });

  it('persists manual locale switching across remounts', async () => {
    installFetchMock();

    const { unmount } = render(<App />);

    const localePicker = await screen.findByLabelText('Language');
    fireEvent.change(localePicker, { target: { value: 'zh-CN' } });

    expect(await screen.findByRole('heading', { name: 'llmfit 控制台' })).toBeInTheDocument();
    expect(window.localStorage.getItem('llmfit.locale')).toBe('zh-CN');

    unmount();
    installFetchMock();
    render(<App />);

    expect(await screen.findByLabelText('语言')).toHaveValue('zh-CN');
    expect(screen.getByText('系统信息')).toBeInTheDocument();
  });
});
