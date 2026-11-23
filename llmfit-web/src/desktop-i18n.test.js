import desktopI18n from '../../llmfit-desktop/ui/i18n.js?raw';
import desktopHtml from '../../llmfit-desktop/ui/index.html?raw';

function loadDesktopI18n(language, storedLocale) {
  const storage = new Map(storedLocale ? [['llmfit.locale', storedLocale]] : []);
  const global = {
    navigator: { language },
    localStorage: {
      getItem: (key) => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value)
    }
  };
  const doc = document.implementation.createHTMLDocument();
  doc.documentElement.innerHTML = desktopHtml;
  new Function('window', 'document', desktopI18n)(global, doc);
  return { i18n: global.llmfitI18n, storage, doc };
}

describe('desktop German localization', () => {
  it.each(['de', 'de-DE', 'de-AT', 'de-CH'])('detects %s and translates the interface', (language) => {
    const { i18n, doc } = loadDesktopI18n(language);
    i18n.applyStaticTranslations();

    expect(i18n.getLocale()).toBe('de');
    expect(doc.documentElement.lang).toBe('de');
    expect(doc.querySelector('#locale-select').getAttribute('aria-label')).toBe('Sprache');
    expect(doc.querySelector('option[value="de"]').textContent).toBe('Deutsch');
    expect(doc.querySelector('#search').placeholder).toBe('Modelle filtern…');
    expect(i18n.t('system.cores', { count: 1 })).toBe('1 Kern');
    expect(i18n.t('system.cores', { count: 4 })).toBe('4 Kerne');
    expect(i18n.t('desktop.memorySummary', { required: 8, available: 16 })).toBe('Speicher: 8 / 16 GB');
    expect(i18n.translateFitLevel('Too Tight')).toBe('Unzureichend');
    expect(i18n.translateRunMode('CPU Offload')).toBe('CPU-Auslagerung');
    expect(i18n.translateUseCase('coding')).toBe('Programmierung');
  });

  it('persists locale changes and notifies listeners', () => {
    const { i18n, storage, doc } = loadDesktopI18n('en-US');
    const listener = vi.fn();
    const unsubscribe = i18n.subscribe(listener);
    i18n.setLocale('de-CH');

    expect(storage.get('llmfit.locale')).toBe('de');
    expect(doc.documentElement.lang).toBe('de');
    expect(listener).toHaveBeenCalledWith('de');
    expect(loadDesktopI18n('zh-CN', storage.get('llmfit.locale')).i18n.getLocale()).toBe('de');

    unsubscribe();
    i18n.setLocale('en');
    expect(listener).toHaveBeenCalledTimes(1);
    expect(i18n.t('desktop.close')).toBe('Close');
  });

  it('retains Chinese, English, and unknown-key fallback behavior', () => {
    expect(loadDesktopI18n('zh-TW').i18n.getLocale()).toBe('zh-CN');
    expect(loadDesktopI18n('fr-FR').i18n.getLocale()).toBe('en');
    expect(loadDesktopI18n('de').i18n.t('unknown.message')).toBe('unknown.message');
  });
});
