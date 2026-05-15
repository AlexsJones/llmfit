import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import en from '../i18n/locales/en';
import zhCN from '../i18n/locales/zh-CN';

export const LOCALE_STORAGE_KEY = 'llmfit.locale';
const FALLBACK_LOCALE = 'en';
const SUPPORTED_LOCALES = {
  en,
  'zh-CN': zhCN
};

const I18nContext = createContext({
  locale: FALLBACK_LOCALE,
  setLocale: () => {},
  t: (key) => key
});

function getNestedValue(obj, key) {
  return key.split('.').reduce((acc, part) => (acc ? acc[part] : undefined), obj);
}

function formatMessage(message, params = {}) {
  if (typeof message === 'function') {
    return message(params);
  }

  if (typeof message !== 'string') {
    return message;
  }

  return message.replace(/\{(\w+)\}/g, (_, token) => {
    return params[token] == null ? `{${token}}` : String(params[token]);
  });
}

export function normalizeLocale(locale) {
  if (!locale || typeof locale !== 'string') {
    return FALLBACK_LOCALE;
  }

  const normalized = locale.toLowerCase();
  if (normalized.startsWith('zh')) {
    return 'zh-CN';
  }

  return FALLBACK_LOCALE;
}

export function resolveLocale() {
  if (typeof window === 'undefined') {
    return FALLBACK_LOCALE;
  }

  try {
    const stored = window.localStorage.getItem(LOCALE_STORAGE_KEY);
    if (stored) {
      return normalizeLocale(stored);
    }
  } catch (_) {
    // ignore storage failures
  }

  return normalizeLocale(window.navigator?.language);
}

export function resolveMessage(locale, key, params = {}) {
  const normalizedLocale = normalizeLocale(locale);
  const message =
    getNestedValue(SUPPORTED_LOCALES[normalizedLocale], key) ??
    getNestedValue(SUPPORTED_LOCALES[FALLBACK_LOCALE], key) ??
    key;

  return formatMessage(message, params);
}

export function I18nProvider({ children }) {
  const [locale, setLocaleState] = useState(resolveLocale);

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.lang = locale;
    }

    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
      } catch (_) {
        // ignore storage failures
      }
    }
  }, [locale]);

  const value = useMemo(() => {
    return {
      locale,
      setLocale: (nextLocale) => setLocaleState(normalizeLocale(nextLocale)),
      t: (key, params) => resolveMessage(locale, key, params)
    };
  }, [locale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  return useContext(I18nContext);
}
