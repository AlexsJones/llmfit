const de = {
  language: {
    label: 'Sprache',
    english: 'English',
    chinese: '中文',
    german: 'Deutsch'
  },
  header: {
    eyebrow: 'Lokale LLMs planen',
    title: 'llmfit-Dashboard',
    copy: 'Hunderte Modelle und Anbieter. Ein Befehl zeigt, was auf deiner Hardware läuft.',
    resetFilters: 'Filter zurücksetzen',
    refresh: 'Aktualisieren',
    themeLabel: 'Farbschema',
    localeLabel: 'Sprache'
  },
  themes: {
    default: 'Standard',
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
    title: 'Systemübersicht',
    noGpu: 'Keine GPU erkannt',
    loading: 'Wird geladen…',
    error: ({ error }) => `Systeminformationen konnten nicht geladen werden: ${error}. Stelle sicher, dass \`llmfit serve\` läuft.`,
    unifiedMemory: 'Gemeinsamer Speicher (CPU + GPU)',
    cores: ({ count }) => `${count} ${count === 1 ? 'Kern' : 'Kerne'}`,
    labels: {
      cpu: 'CPU',
      totalRam: 'Gesamter RAM',
      availableRam: 'Verfügbarer RAM',
      gpu: 'GPU'
    }
  },
  simulation: {
    title: 'Hardware-Simulation',
    active: 'Simulation aktiv',
    idleHint: 'Passe RAM, VRAM oder CPU-Kerne an, um alle Modelle für ein Zielsystem neu zu bewerten.',
    activeHint: 'Die aktuellen Modellbewertungen und Planungsschätzungen basieren auf deiner simulierten Hardware.',
    fields: {
      ram: 'RAM (GB)',
      vram: 'VRAM (GB)',
      cpuCores: 'CPU-Kerne'
    },
    placeholders: {
      ram: 'z. B. 64',
      vram: 'z. B. 24',
      cpuCores: 'z. B. 16'
    },
    actions: {
      apply: 'Simulation anwenden',
      update: 'Simulation aktualisieren',
      reset: 'Zurücksetzen'
    }
  },
  models: {
    title: 'Modellkompatibilität',
    compareAction: ({ count }) => `Vergleichen (${count})`,
    compareDisabledTooltip: 'Wähle mindestens 2 Modelle zum Vergleichen aus',
    summary: ({ returned, total }) => `${returned} angezeigt / ${total} passend`
  },
  filters: {
    searchLabel: 'Suche',
    searchPlaceholder: 'Modell, Anbieter, Anwendungsfall',
    fitLabel: 'Kompatibilität',
    runtimeLabel: 'Laufzeitumgebung',
    useCaseLabel: 'Anwendungsfall',
    providerLabel: 'Anbieter',
    providerPlaceholder: 'Meta, Qwen, Mistral',
    sortLabel: 'Sortierung',
    limitLabel: 'Anzahl',
    limitAll: 'Alle',
    capabilityLabel: 'Fähigkeit',
    licenseLabel: 'Lizenz',
    licensePlaceholder: 'apache-2.0, mit, ...',
    quantizationLabel: 'Quantisierung',
    runModeLabel: 'Ausführungsmodus',
    paramsBucketLabel: 'Modellgröße',
    tensorParallelLabel: 'Tensorparallelität',
    maxContextLabel: 'Maximaler Kontext',
    maxContextPlaceholder: 'z. B. 32768',
    advancedMore: 'Weitere Filter',
    advancedLess: 'Weniger Filter',
    advancedActive: ({ count }) => `(${count} aktiv)`,
    multiSelect: {
      any: 'Beliebig',
      selectedCount: ({ count }) => `${count} ausgewählt`,
      noOptions: 'Keine Optionen verfügbar'
    },
    fitOptions: {
      marginal: 'Ausführbar (mindestens eingeschränkt)',
      good: 'Gut oder besser',
      perfect: 'Nur perfekt',
      too_tight: 'Nur unzureichend',
      all: 'Alle Stufen'
    },
    runtimeOptions: {
      any: 'Alle Laufzeitumgebungen',
      mlx: 'MLX',
      llamacpp: 'llama.cpp',
      vllm: 'vLLM'
    },
    useCaseOptions: {
      all: 'Alle Anwendungsfälle',
      general: 'Allgemein',
      coding: 'Programmierung',
      reasoning: 'Logisches Schlussfolgern',
      chat: 'Chat',
      multimodal: 'Multimodal',
      embedding: 'Embedding'
    },
    sortOptions: {
      score: 'Sortierung: Bewertung',
      tps: 'Sortierung: TPS',
      params: 'Sortierung: Parameter',
      mem: 'Sortierung: Speicher',
      ctx: 'Sortierung: Kontext',
      date: 'Sortierung: Veröffentlichungsdatum',
      use_case: 'Sortierung: Anwendungsfall'
    },
    paramsBucketOptions: {
      all: 'Alle Größen',
      tiny: 'Sehr klein (<3 Mrd.)',
      small: 'Klein (3–8 Mrd.)',
      medium: 'Mittel (8–30 Mrd.)',
      large: 'Groß (30–70 Mrd.)',
      xl: 'Sehr groß (70+ Mrd.)'
    },
    tpOptions: {
      all: 'Beliebige TP',
      1: 'TP=1',
      2: 'TP=2',
      4: 'TP=4',
      8: 'TP=8'
    }
  },
  table: {
    error: ({ error }) => `Modelle konnten nicht geladen werden: ${error}. Stelle sicher, dass diese Seite über \`llmfit serve\` geöffnet wurde.`,
    loading: 'Modellbewertungen werden geladen…',
    empty: 'Keine Modelle entsprechen den aktuellen Filtern.',
    copyModelName: 'Modellnamen kopieren',
    addToComparison: 'Zum Vergleich hinzufügen',
    maxCompare: ({ count }) => `Maximal ${count} Modelle zum Vergleichen`,
    installed: 'Installiert',
    columns: {
      compare: 'Vgl.',
      model: 'Modell',
      provider: 'Anbieter',
      params: 'Parameter',
      fit: 'Kompatibilität',
      mode: 'Modus',
      runtime: 'Laufzeitumgebung',
      score: 'Bewertung',
      tps: 'TPS',
      mem: 'Speicher %',
      context: 'Kontext',
      release: 'Veröffentlichung'
    }
  },
  detail: {
    selectPrompt: 'Wähle eine Modellzeile aus, um die detaillierte Kompatibilitätsanalyse anzuzeigen.',
    sections: {
      capabilities: 'Fähigkeiten',
      ggufSources: 'GGUF-Quellen',
      scoreBreakdown: 'Bewertung im Detail',
      performance: 'Leistung',
      notes: 'Hinweise'
    },
    fields: {
      provider: 'Anbieter',
      runMode: 'Ausführungsmodus',
      runtime: 'Laufzeitumgebung',
      bestQuant: 'Beste Quantisierung',
      memoryRequired: 'Benötigter Speicher',
      memoryAvailable: 'Verfügbarer Speicher',
      license: 'Lizenz',
      moeOffloaded: 'MoE ausgelagert'
    },
    metrics: {
      quality: 'Qualität',
      speed: 'Geschwindigkeit',
      fit: 'Kompatibilität',
      context: 'Kontext',
      memoryUtilization: 'Speicherauslastung %',
      compositeScore: 'Gesamtbewertung',
      estimatedTps: 'Geschätzte TPS'
    },
    noMoeValue: 'Ja (MoE)',
    noNotes: 'Keine zusätzlichen Hinweise zur Kompatibilität dieses Modells.'
  },
  plan: {
    title: 'Planung',
    defaultHint: 'Schätze die minimale und empfohlene Hardware für dieses Modell.',
    simulatedHint: 'Diese Schätzung verwendet dein simuliertes Hardwareprofil.',
    simulatedBadge: 'Simuliert',
    error: ({ error }) => `Planungsanfrage fehlgeschlagen: ${error}`,
    errorFallback: 'Hardwareanforderungen konnten nicht geschätzt werden.',
    validation: {
      context: 'Der Kontext muss eine positive ganze Zahl sein.',
      targetTps: 'Die Ziel-TPS müssen eine positive Zahl sein.'
    },
    fields: {
      context: 'Kontext',
      quant: 'Quantisierung',
      kvQuant: 'KV-Quantisierung',
      targetTps: 'Ziel-TPS'
    },
    placeholders: {
      context: 'z. B. 8192',
      quant: 'z. B. Q4_K_M',
      kvQuant: 'fp16, fp8, q8_0, q4_0, tq',
      targetTps: 'optional'
    },
    actions: {
      estimate: 'Hardwarebedarf schätzen',
      loading: 'Wird geschätzt…'
    },
    sections: {
      paths: 'Ausführungsmöglichkeiten',
      upgrades: 'Aufrüstungsempfehlungen',
      kvAlternatives: 'KV-Cache-Alternativen'
    },
    summary: {
      current: 'Aktueller Status',
      minimum: 'Minimale Hardware',
      recommended: 'Empfohlene Hardware',
      fitLevel: 'Kompatibilitätsstufe',
      runMode: 'Ausführungsmodus',
      estimatedTps: 'Geschätzte TPS',
      kvQuant: 'KV-Quantisierung',
      vram: 'VRAM',
      ram: 'RAM',
      cpuCores: 'CPU-Kerne',
      notRequired: 'Nicht erforderlich'
    },
    paths: {
      gpu: 'GPU',
      cpu_offload: 'CPU-Auslagerung',
      cpu_only: 'Nur CPU'
    },
    pathsFeasible: {
      yes: 'Möglich',
      no: 'Nicht möglich'
    },
    noUpgrades: 'Das aktuelle Zielsystem erfüllt diese Anforderungen bereits.',
    kvTable: {
      quant: 'KV-Quantisierung',
      memory: 'Gesamtspeicher',
      kvCache: 'KV-Cache',
      savings: 'Einsparung',
      supported: 'Unterstützt'
    }
  },
  compare: {
    titleEmpty: 'Modellvergleich',
    instructions: 'Wähle Modelle über die Kontrollkästchen in der Tabelle aus, um sie nebeneinander zu vergleichen.',
    close: 'Schließen',
    headerCount: ({ count }) => `Vergleich von ${count} ${count === 1 ? 'Modell' : 'Modellen'}`,
    fields: {
      fitLevel: 'Kompatibilitätsstufe',
      score: 'Bewertung',
      tps: 'TPS',
      memoryRequired: 'Benötigter Speicher (GB)',
      memoryAvailable: 'Verfügbarer Speicher (GB)',
      bestQuant: 'Beste Quantisierung',
      context: 'Kontext',
      runtime: 'Laufzeitumgebung',
      runMode: 'Ausführungsmodus'
    }
  },
  labels: {
    fit: {
      perfect: 'Perfekt',
      good: 'Gut',
      marginal: 'Eingeschränkt',
      too_tight: 'Unzureichend'
    },
    runMode: {
      gpu: 'GPU',
      moe_offload: 'MoE-Auslagerung',
      cpu_offload: 'CPU-Auslagerung',
      cpu_only: 'Nur CPU'
    },
    useCase: {
      general: 'Allgemein',
      coding: 'Programmierung',
      reasoning: 'Logisches Schlussfolgern',
      chat: 'Chat',
      multimodal: 'Multimodal',
      embedding: 'Embedding'
    }
  }
};

export default de;
