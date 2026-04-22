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
