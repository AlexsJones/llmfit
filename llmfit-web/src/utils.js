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

export function copyModelName(name) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(name).catch(() => {
      fallbackCopy(name);
    });
  } else {
    fallbackCopy(name);
  }
}

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

  // quant filter
  if (filters.quant && filters.quant.length > 0) {
    result = result.filter(
      (m) => m.best_quant && filters.quant.includes(m.best_quant)
    );
  }

  // runMode filter
  if (filters.runMode && filters.runMode.length > 0) {
    result = result.filter(
      (m) => m.run_mode && filters.runMode.includes(m.run_mode)
    );
  }

  // capability filter
  if (filters.capability && filters.capability.length > 0) {
    result = result.filter((m) => {
      const caps = Array.isArray(m.capabilities) ? m.capabilities : [];
      return filters.capability.every((c) => caps.includes(c));
    });
  }

  // paramsBucket filter
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

  // tp filter
  if (filters.tp && filters.tp !== 'all') {
    const tpVal = Number(filters.tp);
    if (Number.isFinite(tpVal)) {
      result = result.filter((m) => {
        const supported = Array.isArray(m.supports_tp) ? m.supports_tp : [];
        return supported.includes(tpVal);
      });
    }
  }

  return result;
}
