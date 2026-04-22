const en = {
  language: {
    label: 'Language',
    english: 'English',
    chinese: '中文'
  },
  header: {
    eyebrow: 'Local LLM Planning',
    title: 'llmfit Dashboard',
    copy: 'Hundreds of models & providers. One command to find what runs on your hardware.',
    resetFilters: 'Reset filters',
    refresh: 'Refresh',
    themeLabel: 'Theme',
    localeLabel: 'Language'
  },
  themes: {
    default: 'Default',
    dracula: 'Dracula',
    solarized: 'Solarized',
    nord: 'Nord',
    monokai: 'Monokai',
    gruvbox: 'Gruvbox',
    'catppuccin-latte': 'Catppuccin Latte',
    'catppuccin-frappe': 'Catppuccin Frappé',
    'catppuccin-macchiato': 'Catppuccin Macchiato',
    'catppuccin-mocha': 'Catppuccin Mocha'
  },
  system: {
    title: 'System Summary',
    noGpu: 'No GPU detected',
    loading: 'Loading…',
    error: ({ error }) => `Could not load system information: ${error}. Make sure \`llmfit serve\` is running.`,
    unifiedMemory: 'Unified memory (CPU + GPU shared)',
    cores: ({ count }) => `${count} cores`,
    labels: {
      cpu: 'CPU',
      totalRam: 'Total RAM',
      availableRam: 'Available RAM',
      gpu: 'GPU'
    }
  },
  models: {
    title: 'Model Fit Explorer',
    compareAction: ({ count }) => `Compare (${count})`,
    compareDisabledTooltip: 'Select at least 2 models to compare',
    summary: ({ returned, total }) => `${returned} shown / ${total} matched`
  },
  filters: {
    searchLabel: 'Search',
    searchPlaceholder: 'model, provider, use case',
    fitLabel: 'Fit filter',
    runtimeLabel: 'Runtime',
    useCaseLabel: 'Use case',
    providerLabel: 'Provider',
    providerPlaceholder: 'Meta, Qwen, Mistral',
    sortLabel: 'Sort',
    limitLabel: 'Limit',
    limitAll: 'All',
    capabilityLabel: 'Capability',
    licenseLabel: 'License',
    licensePlaceholder: 'apache-2.0, mit, ...',
    quantizationLabel: 'Quantization',
    runModeLabel: 'Run mode',
    paramsBucketLabel: 'Params bucket',
    tensorParallelLabel: 'Tensor Parallel',
    maxContextLabel: 'Max context',
    maxContextPlaceholder: 'e.g. 32768',
    advancedMore: 'More filters',
    advancedLess: 'Fewer filters',
    advancedActive: ({ count }) => `(${count} active)`,
    multiSelect: {
      any: 'Any',
      selectedCount: ({ count }) => `${count} selected`,
      noOptions: 'No options available'
    },
    fitOptions: {
      marginal: 'Runnable (Marginal+)',
      good: 'Good or better',
      perfect: 'Perfect only',
      too_tight: 'Too-tight only',
      all: 'All levels'
    },
    runtimeOptions: {
      any: 'Any runtime',
      mlx: 'MLX',
      llamacpp: 'llama.cpp',
      vllm: 'vLLM'
    },
    useCaseOptions: {
      all: 'All use cases',
      general: 'General',
      coding: 'Coding',
      reasoning: 'Reasoning',
      chat: 'Chat',
      multimodal: 'Multimodal',
      embedding: 'Embedding'
    },
    sortOptions: {
      score: 'Sort: Score',
      tps: 'Sort: TPS',
      params: 'Sort: Params',
      mem: 'Sort: Memory',
      ctx: 'Sort: Context',
      date: 'Sort: Release date',
      use_case: 'Sort: Use case'
    },
    paramsBucketOptions: {
      all: 'All sizes',
      tiny: 'Tiny (<3B)',
      small: 'Small (3-8B)',
      medium: 'Medium (8-30B)',
      large: 'Large (30-70B)',
      xl: 'XL (70B+)'
    },
    tpOptions: {
      all: 'Any TP',
      1: 'TP=1',
      2: 'TP=2',
      4: 'TP=4',
      8: 'TP=8'
    }
  },
  table: {
    error: ({ error }) => `Could not load models: ${error}. Confirm this page is opened from \`llmfit serve\`.`,
    loading: 'Loading model fit data…',
    empty: 'No models match the current filters.',
    copyModelName: 'Copy model name',
    addToComparison: 'Add to comparison',
    maxCompare: ({ count }) => `Max ${count} models for comparison`,
    installed: 'Installed',
    columns: {
      compare: 'Cmp',
      model: 'Model',
      provider: 'Provider',
      params: 'Params',
      fit: 'Fit',
      mode: 'Mode',
      runtime: 'Runtime',
      score: 'Score',
      tps: 'TPS',
      mem: 'Mem%',
      context: 'Context',
      release: 'Release'
    }
  },
  detail: {
    selectPrompt: 'Select a model row to inspect detailed fit diagnostics.',
    sections: {
      capabilities: 'Capabilities',
      ggufSources: 'GGUF Sources',
      scoreBreakdown: 'Score Breakdown',
      performance: 'Performance',
      notes: 'Notes'
    },
    fields: {
      provider: 'Provider',
      runMode: 'Run mode',
      runtime: 'Runtime',
      bestQuant: 'Best quant',
      memoryRequired: 'Memory required',
      memoryAvailable: 'Memory available',
      license: 'License',
      moeOffloaded: 'MoE offloaded'
    },
    metrics: {
      quality: 'Quality',
      speed: 'Speed',
      fit: 'Fit',
      context: 'Context',
      memoryUtilization: 'Memory Utilization %',
      compositeScore: 'Composite score',
      estimatedTps: 'Estimated TPS'
    },
    noMoeValue: 'Yes (MoE)',
    noNotes: 'No additional notes for this model fit.'
  },
  compare: {
    titleEmpty: 'Model Comparison',
    instructions: 'Select models using the checkboxes in the table to compare them side by side.',
    close: 'Close',
    headerCount: ({ count }) => `Comparing ${count} model${count !== 1 ? 's' : ''}`,
    fields: {
      fitLevel: 'Fit level',
      score: 'Score',
      tps: 'TPS',
      memoryRequired: 'Memory required (GB)',
      memoryAvailable: 'Memory available (GB)',
      bestQuant: 'Best quant',
      context: 'Context',
      runtime: 'Runtime',
      runMode: 'Run mode'
    }
  },
  labels: {
    fit: {
      perfect: 'Perfect',
      good: 'Good',
      marginal: 'Marginal',
      too_tight: 'Too Tight'
    },
    runMode: {
      gpu: 'GPU',
      moe_offload: 'MoE Offload',
      cpu_offload: 'CPU Offload',
      cpu_only: 'CPU Only'
    },
    useCase: {
      general: 'General',
      coding: 'Coding',
      reasoning: 'Reasoning',
      chat: 'Chat',
      multimodal: 'Multimodal',
      embedding: 'Embedding'
    }
  },
  test: {
    fallbackOnly: 'Fallback only'
  }
};

export default en;
