import { normalizeLocale, resolveMessage } from './contexts/I18nContext';
import en from './i18n/locales/en';
import de from './i18n/locales/de';

describe('i18n helpers', () => {
  it('normalizes Chinese locales to zh-CN', () => {
    expect(normalizeLocale('zh')).toBe('zh-CN');
    expect(normalizeLocale('zh-TW')).toBe('zh-CN');
  });

  it('falls back to English when a key is missing in the active locale', () => {
    expect(resolveMessage('zh-CN', 'test.fallbackOnly')).toBe('Fallback only');
  });
});

describe('locale parity', () => {
  function keyPaths(obj, prefix = '') {
    return Object.entries(obj).flatMap(([key, value]) =>
      value && typeof value === 'object'
        ? keyPaths(value, `${prefix}${key}.`)
        : [`${prefix}${key}`]
    );
  }

  it('translates every English key into zh-CN', async () => {
    const en = (await import('./i18n/locales/en.js')).default;
    const zh = (await import('./i18n/locales/zh-CN.js')).default;
    const zhKeys = new Set(keyPaths(zh));
    const missing = keyPaths(en).filter(
      (key) => !key.startsWith('test.') && !zhKeys.has(key)
    );
    expect(missing).toEqual([]);
  });
});

function messageKeys(messages, prefix = '') {
  return Object.entries(messages).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return typeof value === 'object' ? messageKeys(value, path) : [path];
  });
}

describe('German translations', () => {
  it.each(['de', 'de-DE', 'de-AT', 'de-CH', 'DE-de'])('recognizes %s as German', (locale) => {
    expect(normalizeLocale(locale)).toBe('de');
  });

  it('covers every user-facing English message', () => {
    expect(messageKeys(de).sort()).toEqual(messageKeys(en).filter((key) => !key.startsWith('test.')).sort());
  });

  it('formats dynamic messages and singular/plural forms', () => {
    expect(resolveMessage('de', 'system.cores', { count: 1 })).toBe('1 Kern');
    expect(resolveMessage('de', 'system.cores', { count: 8 })).toBe('8 Kerne');
    expect(resolveMessage('de', 'models.summary', { returned: 5, total: 12 })).toBe('5 angezeigt / 12 passend');
    expect(resolveMessage('de', 'compare.headerCount', { count: 1 })).toBe('Vergleich von 1 Modell');
    expect(resolveMessage('de', 'compare.headerCount', { count: 2 })).toBe('Vergleich von 2 Modellen');
    expect(resolveMessage('de', 'plan.error', { error: 'offline' })).toBe('Planungsanfrage fehlgeschlagen: offline');
  });

  it('retains fallback behavior for missing messages and unsupported locales', () => {
    expect(resolveMessage('de', 'test.fallbackOnly')).toBe('Fallback only');
    expect(resolveMessage('de', 'unknown.message')).toBe('unknown.message');
    expect(normalizeLocale('fr-FR')).toBe('en');
    expect(normalizeLocale('den')).toBe('en');
  });
});
