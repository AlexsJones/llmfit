import { useState, useEffect } from 'react';
import { useFilterDispatch } from '../contexts/FilterContext';
import { useI18n } from '../contexts/I18nContext';
import { useModelContext } from '../contexts/ModelContext';

const THEME_KEY = 'llmfit-theme';

const THEMES = [
  { value: 'default', labelKey: 'themes.default' },
  { value: 'dracula', labelKey: 'themes.dracula' },
  { value: 'solarized', labelKey: 'themes.solarized' },
  { value: 'nord', labelKey: 'themes.nord' },
  { value: 'monokai', labelKey: 'themes.monokai' },
  { value: 'gruvbox', labelKey: 'themes.gruvbox' },
  { value: 'catppuccin-latte', labelKey: 'themes.catppuccin-latte' },
  { value: 'catppuccin-frappe', labelKey: 'themes.catppuccin-frappe' },
  { value: 'catppuccin-macchiato', labelKey: 'themes.catppuccin-macchiato' },
  { value: 'catppuccin-mocha', labelKey: 'themes.catppuccin-mocha' },
];

const LANGUAGE_OPTIONS = [
  { value: 'en', labelKey: 'language.english' },
  { value: 'zh-CN', labelKey: 'language.chinese' },
];

function initialTheme() {
  if (typeof window === 'undefined') {
    return 'default';
  }

  const stored = window.localStorage.getItem(THEME_KEY);
  if (stored && THEMES.some((t) => t.value === stored)) {
    return stored;
  }

  if (stored === 'light') return 'catppuccin-latte';
  if (stored === 'dark') return 'default';

  return window.matchMedia?.('(prefers-color-scheme: light)').matches
    ? 'catppuccin-latte'
    : 'default';
}

export default function Header() {
  const { locale, setLocale, t } = useI18n();
  const [theme, setTheme] = useState(initialTheme);
  const dispatch = useFilterDispatch();
  const { triggerRefresh } = useModelContext();

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  return (
    <header className="hero-shell">
      <div>
        <p className="hero-eyebrow">{t('header.eyebrow')}</p>
        <h1>{t('header.title')}</h1>
        <p className="hero-copy">{t('header.copy')}</p>
      </div>

      <div className="hero-actions">
        <button
          type="button"
          onClick={() => dispatch({ type: 'RESET_FILTERS' })}
          className="btn btn-ghost"
        >
          {t('header.resetFilters')}
        </button>
        <button
          type="button"
          onClick={triggerRefresh}
          className="btn btn-accent"
        >
          {t('header.refresh')}
        </button>
        <select
          value={locale}
          onChange={(e) => setLocale(e.target.value)}
          className="btn btn-theme theme-select"
          aria-label={t('header.localeLabel')}
        >
          {LANGUAGE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {t(option.labelKey)}
            </option>
          ))}
        </select>
        <select
          value={theme}
          onChange={(e) => setTheme(e.target.value)}
          className="btn btn-theme theme-select"
          aria-label={t('header.themeLabel')}
        >
          {THEMES.map((themeOption) => (
            <option key={themeOption.value} value={themeOption.value}>
              {t(themeOption.labelKey)}
            </option>
          ))}
        </select>
      </div>
    </header>
  );
}
