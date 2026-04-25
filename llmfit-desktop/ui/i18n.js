(function initI18n(global) {
  const LOCALE_KEY = 'llmfit.locale';
  const FALLBACK_LOCALE = 'en';

  const MESSAGES = {
    en: {
      language: {
        label: 'Language',
        english: 'English',
        chinese: '中文'
      },
      system: {
        title: 'System',
        cpu: 'CPU',
        totalRam: 'Total RAM',
        availableRam: 'Available RAM',
        memory: 'Memory',
        gpu: 'GPU',
        detecting: 'Detecting…',
        noGpu: 'No GPU detected',
        sharedMemory: 'Shared memory',
        unifiedMemory: 'Unified (CPU + GPU shared)',
        errorLoading: 'Error loading specs',
        cores: ({ count }) => `${count} cores`,
        gpuIndexed: ({ index }) => `GPU ${index}`
      },
      desktop: {
        pageTitle: 'llmfit',
        modelsTitle: 'Model Compatibility',
        searchPlaceholder: 'Filter models...',
        allFitLevels: 'All Fit Levels',
        loadingModels: 'Loading models...',
        noModels: 'No models found',
        errorLoadingModels: 'Error loading models',
        notes: 'Notes',
        fitAnalysis: 'Fit Analysis',
        installed: 'Installed',
        notInstalled: 'Not Installed',
        downloadViaOllama: '⬇ Download via Ollama',
        close: 'Close',
        parameters: 'Parameters',
        quantization: 'Quantization',
        runtime: 'Runtime',
        score: 'Score',
        estSpeed: 'Est. Speed',
        useCase: 'Use Case',
        memorySummary: ({ required, available }) => `Memory: ${required} / ${available} GB`,
        startingDownload: 'Starting download...',
        downloadComplete: 'Download complete!',
        errorPrefix: 'Error: '
      },
      table: {
        model: 'Model',
        params: 'Params',
        quant: 'Quant',
        fit: 'Fit',
        mode: 'Mode',
        score: 'Score',
        ramReq: 'RAM Req',
        estTps: 'Est. TPS',
        useCase: 'Use Case'
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
      }
    },
    'zh-CN': {
      language: {
        label: '语言',
        english: 'English',
        chinese: '中文'
      },
      system: {
        title: '系统信息',
        cpu: 'CPU',
        totalRam: '总内存',
        availableRam: '可用内存',
        memory: '内存',
        gpu: 'GPU',
        detecting: '检测中…',
        noGpu: '未检测到 GPU',
        sharedMemory: '共享内存',
        unifiedMemory: '统一内存（CPU 与 GPU 共享）',
        errorLoading: '加载硬件信息失败',
        cores: ({ count }) => `${count} 核`,
        gpuIndexed: ({ index }) => `GPU ${index}`
      },
      desktop: {
        pageTitle: 'llmfit',
        modelsTitle: '模型适配分析',
        searchPlaceholder: '筛选模型...',
        allFitLevels: '全部适配等级',
        loadingModels: '正在加载模型...',
        noModels: '未找到匹配模型',
        errorLoadingModels: '加载模型失败',
        notes: '说明',
        fitAnalysis: '适配分析',
        installed: '已安装',
        notInstalled: '未安装',
        downloadViaOllama: '⬇ 通过 Ollama 下载',
        close: '关闭',
        parameters: '参数量',
        quantization: '量化',
        runtime: '运行时',
        score: '得分',
        estSpeed: '预估速度',
        useCase: '用途',
        memorySummary: ({ required, available }) => `内存：${required} / ${available} GB`,
        startingDownload: '开始下载...',
        downloadComplete: '下载完成！',
        errorPrefix: '错误：'
      },
      table: {
        model: '模型',
        params: '参数量',
        quant: '量化',
        fit: '适配度',
        mode: '模式',
        score: '得分',
        ramReq: '内存需求',
        estTps: '预估 TPS',
        useCase: '用途'
      },
      labels: {
        fit: {
          perfect: '完美适配',
          good: '良好适配',
          marginal: '勉强可用',
          too_tight: '过紧无法稳定运行'
        },
        runMode: {
          gpu: 'GPU',
          moe_offload: 'MoE 卸载',
          cpu_offload: 'CPU 卸载',
          cpu_only: '仅 CPU'
        },
        useCase: {
          general: '通用',
          coding: '编程',
          reasoning: '推理',
          chat: '对话',
          multimodal: '多模态',
          embedding: '向量嵌入'
        }
      }
    }
  };

  function getNestedValue(obj, key) {
    return key.split('.').reduce((acc, part) => (acc ? acc[part] : undefined), obj);
  }

  function formatMessage(message, params) {
    if (typeof message === 'function') {
      return message(params || {});
    }
    if (typeof message !== 'string') {
      return message;
    }
    return message.replace(/\{(\w+)\}/g, function replaceToken(_, token) {
      return params && params[token] != null ? String(params[token]) : `{${token}}`;
    });
  }

  function normalizeLocale(locale) {
    if (!locale || typeof locale !== 'string') {
      return FALLBACK_LOCALE;
    }
    return locale.toLowerCase().startsWith('zh') ? 'zh-CN' : FALLBACK_LOCALE;
  }

  function getStoredLocale() {
    try {
      const stored = global.localStorage.getItem(LOCALE_KEY);
      return stored ? normalizeLocale(stored) : null;
    } catch (_) {
      return null;
    }
  }

  function detectLocale() {
    return getStoredLocale() || normalizeLocale(global.navigator && global.navigator.language);
  }

  let currentLocale = detectLocale();
  const listeners = new Set();

  function t(key, params) {
    const value =
      getNestedValue(MESSAGES[currentLocale], key) ??
      getNestedValue(MESSAGES[FALLBACK_LOCALE], key) ??
      key;
    return formatMessage(value, params);
  }

  function setLocale(locale) {
    const nextLocale = normalizeLocale(locale);
    if (nextLocale === currentLocale) {
      return;
    }
    currentLocale = nextLocale;
    try {
      global.localStorage.setItem(LOCALE_KEY, currentLocale);
    } catch (_) {
      // ignore storage failures
    }
    document.documentElement.lang = currentLocale;
    listeners.forEach(function notify(listener) {
      listener(currentLocale);
    });
  }

  function getLocale() {
    return currentLocale;
  }

  function subscribe(listener) {
    listeners.add(listener);
    return function unsubscribe() {
      listeners.delete(listener);
    };
  }

  function applyStaticTranslations() {
    document.title = t('desktop.pageTitle');
    document.documentElement.lang = currentLocale;

    document.querySelectorAll('[data-i18n]').forEach(function updateText(node) {
      node.textContent = t(node.getAttribute('data-i18n'));
    });

    document.querySelectorAll('[data-i18n-placeholder]').forEach(function updatePlaceholder(node) {
      node.setAttribute('placeholder', t(node.getAttribute('data-i18n-placeholder')));
    });

    document.querySelectorAll('[data-i18n-aria-label]').forEach(function updateAria(node) {
      node.setAttribute('aria-label', t(node.getAttribute('data-i18n-aria-label')));
    });
  }

  function normalizeFitCode(value) {
    if (!value) return null;
    const normalized = String(value).trim().toLowerCase().replace(/[\s-]+/g, '_');
    const aliases = {
      perfect: 'perfect',
      good: 'good',
      marginal: 'marginal',
      too_tight: 'too_tight',
      tootight: 'too_tight'
    };
    return aliases[normalized] || null;
  }

  function normalizeRunModeCode(value) {
    if (!value) return null;
    const normalized = String(value).trim().toLowerCase().replace(/[\s-]+/g, '_');
    const aliases = {
      gpu: 'gpu',
      moe_offload: 'moe_offload',
      cpu_offload: 'cpu_offload',
      cpu_only: 'cpu_only'
    };
    return aliases[normalized] || null;
  }

  function normalizeUseCaseCode(value) {
    if (!value) return null;
    const normalized = String(value).trim().toLowerCase();
    return ['general', 'coding', 'reasoning', 'chat', 'multimodal', 'embedding'].includes(normalized)
      ? normalized
      : null;
  }

  function translateFitLevel(value) {
    const code = normalizeFitCode(value);
    return code ? t(`labels.fit.${code}`) : (value || '—');
  }

  function translateRunMode(value) {
    const code = normalizeRunModeCode(value);
    return code ? t(`labels.runMode.${code}`) : (value || '—');
  }

  function translateUseCase(value) {
    const code = normalizeUseCaseCode(value);
    return code ? t(`labels.useCase.${code}`) : (value || '—');
  }

  global.llmfitI18n = {
    LOCALE_KEY,
    getLocale,
    setLocale,
    subscribe,
    t,
    applyStaticTranslations,
    normalizeFitCode,
    normalizeRunModeCode,
    normalizeUseCaseCode,
    translateFitLevel,
    translateRunMode,
    translateUseCase
  };
})(window);
