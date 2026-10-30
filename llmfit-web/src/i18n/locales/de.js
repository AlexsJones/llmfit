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
    profile: ({ name }) => `Profil: ${name}`,
    profileHint: 'Der Server bewertet Modelle anhand dieses Hardwareprofils, nicht anhand des Rechners, auf dem er läuft.',
    title: 'Systemübersicht',
    noGpu: 'Keine GPU erkannt',
    loading: 'Wird geladen…',
    error: ({ error }) => `Systeminformationen konnten nicht geladen werden: ${error}. Stelle sicher, dass \`llmfit serve\` läuft.`,
    unifiedMemory: 'Gemeinsamer Speicher (CPU + GPU)',
    cores: ({ count }) => `${count} ${count === 1 ? 'Kern' : 'Kerne'}`,
    freeVram: ({ value }) => `${value} GB frei`,
    bandwidth: ({ value }) => `${value} GB/s`,
    ofTotal: ({ value }) => `von insgesamt ${value} GB`,
    labels: {
      gpuAvailable: 'Für die GPU nutzbarer Speicher',
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
    availabilityLabel: 'Verfügbarkeit',
    availabilityOptions: {
      all: 'Alle',
      gguf: 'GGUF verfügbar',
      installed: 'Installiert'
    },
    paramsRangeLabel: 'Parameter (Mrd.)',
    memRangeLabel: 'Speicher %',
    rangeMin: 'Min.',
    rangeMax: 'Max.',
    paramsMinAria: 'Minimale Parameteranzahl (Milliarden)',
    paramsMaxAria: 'Maximale Parameteranzahl (Milliarden)',
    memMinAria: 'Minimale Speicherauslastung in %',
    memMaxAria: 'Maximale Speicherauslastung in %',
    installedFirstLabel: 'Installierte zuerst',
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
    measured: 'Auf passender Hardware gemessen',
    usableContext: ({ usable }) => `Nutzbarer Kontext auf dieser Hardware: ${usable} Tokens`,
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
    copyCommand: 'Befehl kopieren',
    verifyHint: 'Reproduziere diese Schätzung auf deiner eigenen Hardware:',
    measuredFrom: ({ count, hardware, source }) =>
      `${source}: ${count} ${count === 1 ? 'Durchlauf' : 'Durchläufe'} auf ${hardware}`,
    pcieMatch: 'PCIe-Übereinstimmung',
    pcieMatchGrades: {
      exact_pcie: 'exakte PCIe-Breiten',
      legacy: 'CPU/GPU-Ersatzzuordnung (PCIe nicht verifiziert)'
    },
    basis: {
      method: 'Methode',
      efficiency: 'Effizienzfaktor',
      gpuBandwidth: 'GPU-Bandbreite',
      ddrBandwidth: 'RAM-Bandbreite des Systems',
      assumedContext: 'Angenommener Kontext',
      localCalibration: 'Lokale Kalibrierung'
    },
    selectPrompt: 'Wähle eine Modellzeile aus, um die detaillierte Kompatibilitätsanalyse anzuzeigen.',
    sections: {
      estimateBasis: 'Grundlage der Schätzung',
      capabilities: 'Fähigkeiten',
      ggufSources: 'GGUF-Quellen',
      scoreBreakdown: 'Bewertung im Detail',
      performance: 'Leistung',
      notes: 'Hinweise'
    },
    fields: {
      usableContext: 'Nutzbarer / nativer Kontext',
      effectiveContext: 'Für die Schätzung verwendeter Kontext',
      diskSize: 'Download-Größe',
      ollamaTag: 'Ollama-Tag',
      confidence: 'Verlässlichkeit des Durchsatzes',
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
      measuredTps: 'Gemessene TPS',
      prefillTps: 'Prefill-TPS',
      ttft: 'Zeit bis zum ersten Token',
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
  concurrency: {
    title: 'Parallele Sitzungen',
    hint: 'Wie viele Sitzungen bei jeder Kontextlänge gleichzeitig im Speicher Platz haben.',
    fields: {
      kvQuant: 'KV-Cache',
      users: 'Ziel-Sitzungen'
    },
    placeholders: {
      users: 'Optional'
    },
    actions: {
      estimate: 'Kapazität schätzen'
    },
    table: {
      context: 'Kontext',
      perSession: 'KV / Sitzung',
      sessions: 'Sitzungen'
    },
    clamped: 'begrenzt',
    summary: ({ pool, weights, kv, quant, native }) =>
      `Pool ${pool} GB · Gewichte ${weights} GB im Speicher · ${kv} GB für KV · ${quant} · nativ ${native}`,
    targetFits: ({ users, context }) =>
      `${users} parallele ${users === 1 ? 'Sitzung passt' : 'Sitzungen passen'} bis zu einem Kontext von ${context}.`,
    targetMisses: ({ users }) =>
      `${users} parallele ${users === 1 ? 'Sitzung passt' : 'Sitzungen passen'} bei keinem der aufgeführten Kontexte in den Speicher.`,
    ceilingNote: 'Obergrenze der Speicherkapazität (Sitzungen im Speicher), keine Durchsatzangabe unter Last.',
    error: ({ error }) => `Parallelität konnte nicht geschätzt werden: ${error}`
  },
  storage: {
    title: 'Speicherplatz-Planer',
    hint: 'Dimensioniere eine SSD für eine Bibliothek lauffähiger Modelle, die nacheinander genutzt werden.',
    fields: {
      keep: 'Zu behaltende Modelle',
      selection: 'Auswahl',
      osReserve: 'Reserve fürs Betriebssystem',
      scratch: 'Download-Zwischenspeicher',
      headroom: 'Freier Puffer %',
      search: 'Suche',
      perfect: 'Nur perfekt passende'
    },
    placeholders: {
      search: 'Name, Anbieter oder Größe'
    },
    selection: {
      score: 'Höchste Bewertung',
      largest: 'Größte Gewichte'
    },
    actions: {
      show: 'Planer öffnen',
      hide: 'Planer ausblenden',
      estimate: 'Speicherplatz schätzen'
    },
    summary: {
      library: 'Modellbibliothek',
      scratch: 'Download-Zwischenspeicher',
      need: 'Gesamtbedarf',
      minimumSsd: 'Minimale SSD',
      suggestedSsd: 'Empfohlene SSD'
    },
    table: {
      disk: 'Datenträger'
    },
    selected: ({ selected, requested, eligible }) =>
      `${selected} von ${requested} angeforderten Modellen ausgewählt (${eligible} geeignet).`,
    error: ({ error }) => `Speicherbedarf konnte nicht geschätzt werden: ${error}`
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
    confidence: {
      measured_local: 'Gemessen (dieser Rechner)',
      measured_community: 'Gemessen (Community)',
      calibrated: 'Kalibriert',
      estimated: 'Geschätzt',
      unsupported: 'Nicht unterstützt'
    },
    measuredSource: {
      community: 'Community-Bestenliste',
      community_llmfit: 'llmfit-Community-Einreichungen',
      local_bench: 'Lokaler llmfit-Benchmark'
    },
    basisMethod: {
      gpu_bandwidth_roofline: 'GPU-Bandbreiten-Roofline',
      backend_constant: 'Backend-Konstante',
      cpu_constant: 'CPU-Konstante',
      unsupported: 'Nicht geschätzt'
    },
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
