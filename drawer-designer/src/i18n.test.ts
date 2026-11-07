import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { LANGUAGES, baseText, de, detectLang, translate } from './i18n';

const SRC = new URL('.', import.meta.url).pathname;

function sources(dir = SRC): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const path = join(dir, e.name);
    if (e.isDirectory()) return sources(path);
    if (!/\.tsx?$/.test(e.name) || e.name.endsWith('.test.ts') || e.name === 'i18n.ts') return [];
    return [readFileSync(path, 'utf8')];
  });
}

const code = sources().join('\n');
/** Every `t('…')` / `tr('…')` call with a literal key. */
const called = [...code.matchAll(/\b(?:t|tr)\(\s*'((?:[^'\\]|\\.)*)'/g)].map((m) => m[1].replace(/\\'/g, "'"));

describe('interface language', () => {
  it('translates every string the code asks for', () => {
    const missing = [...new Set(called)].filter((k) => !(k in de)).sort();
    expect(missing).toEqual([]);
  });

  it('has no leftover entries for strings the code no longer contains', () => {
    // Module labels and the like reach `t()` through a variable, so match any
    // literal in the source rather than only `t()` call sites.
    const stale = Object.keys(de).filter((k) => !code.includes(`'${k.replace(/'/g, "\\'")}'`));
    expect(stale).toEqual([]);
  });

  it('keeps the placeholders of every translation', () => {
    const holes = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
    const broken = Object.entries(de).filter(([k, v]) => holes(baseText(k)).join() !== holes(v).join());
    expect(broken).toEqual([]);
  });

  it('falls back to English, strips the context and fills variables', () => {
    expect(translate('en', 'Width')).toBe('Width');
    expect(translate('de', 'Width')).toBe('Breite');
    expect(translate('de', 'Not translated yet')).toBe('Not translated yet');
    expect(translate('en', 'Height|bin')).toBe('Height');
    expect(translate('de', '{n} compartments', { n: 3 })).toBe('3 Fächer');
    // An unknown placeholder is left alone rather than printed as "undefined".
    expect(translate('en', '{n} compartments')).toBe('{n} compartments');
  });

  it('picks a language from the browser and ignores unknown ones', () => {
    const nav = (langs: string[]) => {
      // node defines navigator as a getter, so replace the property outright.
      Object.defineProperty(globalThis, 'navigator', {
        value: { languages: langs, language: langs[0] },
        configurable: true,
      });
      return detectLang();
    };
    expect(nav(['de-AT', 'en'])).toBe('de');
    expect(nav(['en-GB'])).toBe('en');
    expect(nav(['fr-FR'])).toBe('en');
    expect(LANGUAGES.map((l) => l.id)).toEqual(['en', 'de']);
  });
});
