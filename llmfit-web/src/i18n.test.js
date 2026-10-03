import { normalizeLocale, resolveMessage } from './contexts/I18nContext';

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
