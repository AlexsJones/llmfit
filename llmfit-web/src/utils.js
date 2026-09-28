export function round(value, digits = 1) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '\u2014';
  }
  return value.toFixed(digits);
}

export function fitClass(code) {
  return `fit fit-${code || 'unknown'}`;
}

export function modeClass(code) {
  return `mode mode-${code || 'unknown'}`;
}

export function fitRank(level) {
  switch (level) {
    case 'perfect':
      return 3;
    case 'good':
      return 2;
    case 'marginal':
      return 1;
    case 'too_tight':
      return 0;
    default:
      return -1;
  }
}

export function normalizeFitCode(value) {
  if (!value) return null;
  const normalized = String(value).trim().toLowerCase().replace(/[\s-]+/g, '_');
  const aliases = {
    perfect: 'perfect',
    good: 'good',
    marginal: 'marginal',
    too_tight: 'too_tight',
    tootight: 'too_tight'
  };
  return aliases[normalized] ?? null;
}

export function normalizeRunModeCode(value) {
  if (!value) return null;
  const normalized = String(value).trim().toLowerCase().replace(/[\s-]+/g, '_');
  const aliases = {
    gpu: 'gpu',
    moe_offload: 'moe_offload',
    cpu_offload: 'cpu_offload',
    cpu_only: 'cpu_only'
  };
  return aliases[normalized] ?? null;
}

export function normalizeUseCaseCode(value) {
  if (!value) return null;
  const normalized = String(value).trim().toLowerCase();
  const allowed = ['general', 'coding', 'reasoning', 'chat', 'multimodal', 'embedding'];
  return allowed.includes(normalized) ? normalized : null;
}

export function translateFitLevel(t, fitLevel, fitLabel) {
  const code = normalizeFitCode(fitLevel) ?? normalizeFitCode(fitLabel);
  return code ? t(`labels.fit.${code}`) : (fitLabel ?? '\u2014');
}

export function translateRunMode(t, runMode, runModeLabel) {
  const code = normalizeRunModeCode(runMode) ?? normalizeRunModeCode(runModeLabel);
  return code ? t(`labels.runMode.${code}`) : (runModeLabel ?? runMode ?? '\u2014');
}

export function translateUseCase(t, useCase) {
  const code = normalizeUseCaseCode(useCase);
  return code ? t(`labels.useCase.${code}`) : (useCase ?? '\u2014');
}

const CONFIDENCE_CODES = [
  'measured_local',
  'measured_community',
  'calibrated',
  'estimated',
  'unsupported'
];

export function normalizeConfidenceCode(value) {
  if (!value) return null;
  const normalized = String(value).trim().toLowerCase();
  return CONFIDENCE_CODES.includes(normalized) ? normalized : null;
}

export function translateConfidence(t, code, label) {
  const normalized = normalizeConfidenceCode(code);
  return normalized ? t(`labels.confidence.${normalized}`) : (label ?? '\u2014');
}

export function confidenceClass(code) {
  return `confidence confidence-${normalizeConfidenceCode(code) ?? 'unknown'}`;
}

// Measured tok/s outranks the formula estimate wherever both exist, the same
// precedence the TUI and CLI use.
export function throughputOf(model) {
  const measured = model?.measured_tps;
  if (measured && typeof measured.tok_s === 'number') {
    return { tps: measured.tok_s, measured: true };
  }
  return { tps: model?.estimated_tps, measured: false };
}

// "native → usable" when memory caps the context below the model's window.
export function formatContextWindow(model, locale) {
  const native = model?.context_length;
  if (typeof native !== 'number') {
    return native ?? '\u2014';
  }
  const usable = model?.usable_context;
  const nativeText = native.toLocaleString(locale);
  if (typeof usable === 'number' && usable > 0 && usable < native) {
    return `${nativeText} \u2192 ${usable.toLocaleString(locale)}`;
  }
  return nativeText;
}

export function applyClientFitFilter(models, minFit) {
  const list = Array.isArray(models) ? models : [];
  if (minFit === 'all') {
    return list;
  }
  if (minFit === 'too_tight') {
    return list.filter((model) => model.fit_level === 'too_tight');
  }

  const threshold = fitRank(minFit);
  return list.filter((model) => {
    const rank = fitRank(model.fit_level);
    return rank >= threshold;
  });
}

export function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(() => {
      fallbackCopy(text);
    });
  } else {
    fallbackCopy(text);
  }
}

export const copyModelName = copyText;

function fallbackCopy(text) {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.left = '-9999px';
  textarea.style.top = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand('copy');
  } catch (_) {
    // ignore
  }
  document.body.removeChild(textarea);
}

export function collectUniqueValues(models, field) {
  const set = new Set();
  for (const model of models) {
    const val = model[field];
    if (Array.isArray(val)) {
      for (const v of val) {
        if (v) set.add(v);
      }
    } else if (val) {
      set.add(val);
    }
  }
  return Array.from(set).sort();
}

export function applyClientFilters(models, filters) {
  let result = models;

  if (filters.quant && filters.quant.length > 0) {
    result = result.filter(
      (m) => m.best_quant && filters.quant.includes(m.best_quant)
    );
  }

  if (filters.runMode && filters.runMode.length > 0) {
    result = result.filter(
      (m) => m.run_mode && filters.runMode.includes(m.run_mode)
    );
  }

  if (filters.capability && filters.capability.length > 0) {
    result = result.filter((m) => {
      const caps = Array.isArray(m.capabilities) ? m.capabilities : [];
      return filters.capability.every((c) => caps.includes(c));
    });
  }

  if (filters.paramsBucket && filters.paramsBucket !== 'all') {
    const bucket = filters.paramsBucket;
    result = result.filter((m) => {
      const p = m.params_b;
      if (typeof p !== 'number') return false;
      switch (bucket) {
        case 'tiny':
          return p < 3;
        case 'small':
          return p >= 3 && p < 8;
        case 'medium':
          return p >= 8 && p < 30;
        case 'large':
          return p >= 30 && p < 70;
        case 'xl':
          return p >= 70;
        default:
          return true;
      }
    });
  }

  if (filters.availability === 'gguf') {
    result = result.filter(
      (m) => Array.isArray(m.gguf_sources) && m.gguf_sources.length > 0
    );
  } else if (filters.availability === 'installed') {
    result = result.filter((m) => m.installed === true);
  }

  result = applyRange(result, 'params_b', filters.paramsMin, filters.paramsMax);
  result = applyRange(result, 'utilization_pct', filters.memMin, filters.memMax);

  if (filters.tp && filters.tp !== 'all') {
    const tpVal = Number(filters.tp);
    if (Number.isFinite(tpVal)) {
      result = result.filter((m) => {
        const supported = Array.isArray(m.supports_tp) ? m.supports_tp : [];
        return supported.includes(tpVal);
      });
    }
  }

  if (filters.installedFirst) {
    // Array.prototype.sort is stable, so the server's ranking is kept
    // within the installed and not-installed groups.
    result = [...result].sort((a, b) => Number(b.installed === true) - Number(a.installed === true));
  }

  return result;
}

function parseBound(value) {
  if (value === undefined || value === null || String(value).trim() === '') {
    return null;
  }
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

// Inclusive min/max on a numeric field; a blank or invalid bound is ignored.
function applyRange(models, field, min, max) {
  const lo = parseBound(min);
  const hi = parseBound(max);
  if (lo === null && hi === null) return models;
  return models.filter((m) => {
    const value = m[field];
    if (typeof value !== 'number') return false;
    return (lo === null || value >= lo) && (hi === null || value <= hi);
  });
}
