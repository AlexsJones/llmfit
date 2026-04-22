const zhCN = {
  language: {
    label: '语言',
    english: 'English',
    chinese: '中文'
  },
  header: {
    eyebrow: '本地 LLM 规划',
    title: 'llmfit 控制台',
    copy: '聚合海量模型与提供方，用一条命令找出哪些模型能在你的硬件上真正跑起来。',
    resetFilters: '重置筛选',
    refresh: '刷新',
    themeLabel: '主题',
    localeLabel: '语言'
  },
  themes: {
    default: '默认',
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
    title: '系统信息',
    noGpu: '未检测到 GPU',
    loading: '加载中…',
    error: ({ error }) => `无法加载系统信息：${error}。请确认 \`llmfit serve\` 正在运行。`,
    unifiedMemory: '统一内存（CPU 与 GPU 共享）',
    cores: ({ count }) => `${count} 核`,
    labels: {
      cpu: 'CPU',
      totalRam: '总内存',
      availableRam: '可用内存',
      gpu: 'GPU'
    }
  },
  simulation: {
    title: '硬件模拟',
    active: '模拟已启用',
    idleHint: '填写目标机器的 RAM、VRAM 或 CPU 核心数后，可按该硬件重新计算全部模型适配度。',
    activeHint: '当前模型适配结果和规划结果都基于你设置的模拟硬件。',
    fields: {
      ram: '内存（GB）',
      vram: '显存（GB）',
      cpuCores: 'CPU 核心数'
    },
    placeholders: {
      ram: '例如 64',
      vram: '例如 24',
      cpuCores: '例如 16'
    },
    actions: {
      apply: '应用模拟',
      update: '更新模拟',
      reset: '重置'
    }
  },
  models: {
    title: '模型适配分析',
    compareAction: ({ count }) => `对比（${count}）`,
    compareDisabledTooltip: '至少选择 2 个模型后才能对比',
    summary: ({ returned, total }) => `当前显示 ${returned} / 匹配 ${total}`
  },
  filters: {
    searchLabel: '搜索',
    searchPlaceholder: '模型、提供方、用途',
    fitLabel: '适配筛选',
    runtimeLabel: '运行时',
    useCaseLabel: '用途',
    providerLabel: '提供方',
    providerPlaceholder: '如 Meta、Qwen、Mistral',
    sortLabel: '排序',
    limitLabel: '数量限制',
    limitAll: '全部',
    capabilityLabel: '能力',
    licenseLabel: '许可证',
    licensePlaceholder: '如 apache-2.0、mit',
    quantizationLabel: '量化',
    runModeLabel: '运行模式',
    paramsBucketLabel: '参数区间',
    tensorParallelLabel: '张量并行',
    maxContextLabel: '最大上下文',
    maxContextPlaceholder: '例如 32768',
    advancedMore: '更多筛选',
    advancedLess: '收起筛选',
    advancedActive: ({ count }) => `（已启用 ${count} 项）`,
    multiSelect: {
      any: '不限',
      selectedCount: ({ count }) => `已选 ${count} 项`,
      noOptions: '暂无可选项'
    },
    fitOptions: {
      marginal: '可运行（勉强可用及以上）',
      good: '良好适配及以上',
      perfect: '仅完美适配',
      too_tight: '仅显示过紧模型',
      all: '全部等级'
    },
    runtimeOptions: {
      any: '任意运行时',
      mlx: 'MLX',
      llamacpp: 'llama.cpp',
      vllm: 'vLLM'
    },
    useCaseOptions: {
      all: '全部用途',
      general: '通用',
      coding: '编程',
      reasoning: '推理',
      chat: '对话',
      multimodal: '多模态',
      embedding: '向量嵌入'
    },
    sortOptions: {
      score: '排序：得分',
      tps: '排序：TPS',
      params: '排序：参数量',
      mem: '排序：内存',
      ctx: '排序：上下文',
      date: '排序：发布日期',
      use_case: '排序：用途'
    },
    paramsBucketOptions: {
      all: '全部尺寸',
      tiny: '超小（<3B）',
      small: '小型（3-8B）',
      medium: '中型（8-30B）',
      large: '大型（30-70B）',
      xl: '超大（70B+）'
    },
    tpOptions: {
      all: '任意 TP',
      1: 'TP=1',
      2: 'TP=2',
      4: 'TP=4',
      8: 'TP=8'
    }
  },
  table: {
    error: ({ error }) => `无法加载模型数据：${error}。请确认当前页面来自 \`llmfit serve\`。`,
    loading: '正在加载模型适配数据…',
    empty: '当前筛选条件下没有匹配的模型。',
    copyModelName: '复制模型名称',
    addToComparison: '加入对比',
    maxCompare: ({ count }) => `最多只能对比 ${count} 个模型`,
    installed: '已安装',
    columns: {
      compare: '对比',
      model: '模型',
      provider: '提供方',
      params: '参数量',
      fit: '适配度',
      mode: '模式',
      runtime: '运行时',
      score: '得分',
      tps: 'TPS',
      mem: '内存%',
      context: '上下文',
      release: '发布'
    }
  },
  detail: {
    selectPrompt: '点击模型行以查看更详细的适配诊断。',
    sections: {
      capabilities: '能力',
      ggufSources: 'GGUF 来源',
      scoreBreakdown: '评分拆解',
      performance: '性能',
      notes: '说明'
    },
    fields: {
      provider: '提供方',
      runMode: '运行模式',
      runtime: '运行时',
      bestQuant: '最佳量化',
      memoryRequired: '所需内存',
      memoryAvailable: '可用内存',
      license: '许可证',
      moeOffloaded: 'MoE 卸载'
    },
    metrics: {
      quality: '质量',
      speed: '速度',
      fit: '适配度',
      context: '上下文',
      memoryUtilization: '内存利用率 %',
      compositeScore: '综合得分',
      estimatedTps: '预估 TPS'
    },
    noMoeValue: '是（MoE）',
    noNotes: '该模型适配结果暂无额外说明。'
  },
  plan: {
    title: '硬件规划',
    defaultHint: '估算当前模型的最低硬件要求和推荐硬件要求。',
    simulatedHint: '当前估算结果基于你设置的模拟硬件环境。',
    simulatedBadge: '模拟中',
    error: ({ error }) => `规划请求失败：${error}`,
    errorFallback: '无法估算硬件规划。',
    validation: {
      context: '上下文长度必须是正整数。',
      targetTps: '目标 TPS 必须是正数。'
    },
    fields: {
      context: '上下文',
      quant: '量化',
      kvQuant: 'KV 量化',
      targetTps: '目标 TPS'
    },
    placeholders: {
      context: '例如 8192',
      quant: '例如 Q4_K_M',
      kvQuant: 'fp16、fp8、q8_0、q4_0、tq',
      targetTps: '可选'
    },
    actions: {
      estimate: '生成规划',
      loading: '估算中…'
    },
    sections: {
      paths: '运行路径',
      upgrades: '升级建议',
      kvAlternatives: 'KV Cache 备选方案'
    },
    summary: {
      current: '当前状态',
      minimum: '最低硬件',
      recommended: '推荐硬件',
      fitLevel: '适配等级',
      runMode: '运行模式',
      estimatedTps: '预估 TPS',
      kvQuant: 'KV 量化',
      vram: '显存',
      ram: '内存',
      cpuCores: 'CPU 核心',
      notRequired: '不要求'
    },
    paths: {
      gpu: 'GPU',
      cpu_offload: 'CPU 卸载',
      cpu_only: '纯 CPU'
    },
    pathsFeasible: {
      yes: '可行',
      no: '不可行'
    },
    noUpgrades: '当前目标已经满足该规划要求。',
    kvTable: {
      quant: 'KV 量化',
      memory: '总内存',
      kvCache: 'KV Cache',
      savings: '节省',
      supported: '支持'
    }
  },
  compare: {
    titleEmpty: '模型对比',
    instructions: '勾选表格中的模型后，可以在这里并排对比。',
    close: '关闭',
    headerCount: ({ count }) => `正在对比 ${count} 个模型`,
    fields: {
      fitLevel: '适配等级',
      score: '得分',
      tps: 'TPS',
      memoryRequired: '所需内存（GB）',
      memoryAvailable: '可用内存（GB）',
      bestQuant: '最佳量化',
      context: '上下文',
      runtime: '运行时',
      runMode: '运行模式'
    }
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
};

export default zhCN;
