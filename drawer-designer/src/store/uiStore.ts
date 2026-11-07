import { create } from 'zustand';
import { type Lang, type Vars, detectLang, translate } from '../i18n';
import type { Compartment } from '../model/types';
import { PALETTES, type ThemeName, type ThemeSetting, resolveTheme } from '../theme';

export type ViewMode = '2d' | '3d' | 'split';

interface UiState {
  /** Every selected compartment; `selectedId` is the primary (last clicked) one. */
  selectedIds: string[];
  selectedId: string | null;
  selectedDividerId: string | null;
  hoverId: string | null;
  view: ViewMode;
  gridSnap: number;
  /** Snap the layout to the Gridfinity 42 mm grid. */
  gridfinity: boolean;
  edgeSnap: boolean;
  showDrawer: boolean;
  exportOpen: boolean;
  setupOpen: boolean;
  projectsOpen: boolean;
  shortcutsOpen: boolean;
  themeSetting: ThemeSetting;
  theme: ThemeName;
  lang: Lang;
  plateSpacing: number;
  plateRotate: boolean;
  explode: number;
  /** Fraction of the workspace given to the 2D editor in split view (0.2..0.8). */
  splitRatio: number;
  /** Copied compartments, held in memory for paste. */
  clipboard: Compartment[];
  mobilePanel: 'tools' | 'properties' | null;
  toast: { id: number; message: string; tone: 'info' | 'warn' } | null;

  select(id: string | null, dividerId?: string | null): void;
  /** Add or remove one compartment from the selection (Shift-click). */
  toggleSelect(id: string): void;
  selectMany(ids: string[]): void;
  setHover(id: string | null): void;
  setView(v: ViewMode): void;
  setGridSnap(v: number): void;
  setGridfinity(v: boolean): void;
  setEdgeSnap(v: boolean): void;
  setShowDrawer(v: boolean): void;
  setExportOpen(v: boolean): void;
  setSetupOpen(v: boolean): void;
  setProjectsOpen(v: boolean): void;
  setShortcutsOpen(v: boolean): void;
  setTheme(v: ThemeSetting): void;
  setLang(v: Lang): void;
  setPlateSpacing(v: number): void;
  setPlateRotate(v: boolean): void;
  setExplode(v: number): void;
  setSplitRatio(v: number): void;
  setClipboard(items: Compartment[]): void;
  setMobilePanel(p: UiState['mobilePanel']): void;
  notify(message: string, tone?: 'info' | 'warn'): void;
}

const THEME_KEY = 'organa-theme';
const LANG_KEY = 'organa-lang';

function readStoredLang(): Lang {
  try {
    const v = window.localStorage.getItem(LANG_KEY);
    if (v === 'en' || v === 'de') return v;
  } catch {
    // Private browsing; fall back to the browser's own setting.
  }
  return detectLang();
}

/** Keep the document language in step so screen readers and hyphenation follow. */
export function applyLang(lang: Lang) {
  if (typeof document !== 'undefined') document.documentElement.lang = lang;
}

function readStoredTheme(): ThemeSetting {
  try {
    const v = window.localStorage.getItem(THEME_KEY);
    if (v === 'light' || v === 'dark' || v === 'system') return v;
  } catch {
    // ignore
  }
  return 'system';
}

export function applyTheme(theme: ThemeName) {
  if (typeof document !== 'undefined') document.documentElement.dataset.theme = theme;
}

let toastId = 0;

export const useUi = create<UiState>()((set) => ({
  selectedIds: [],
  selectedId: null,
  selectedDividerId: null,
  hoverId: null,
  view: 'split',
  gridSnap: 5,
  gridfinity: false,
  edgeSnap: true,
  showDrawer: true,
  exportOpen: false,
  setupOpen: false,
  projectsOpen: false,
  shortcutsOpen: false,
  themeSetting: readStoredTheme(),
  theme: resolveTheme(readStoredTheme()),
  lang: readStoredLang(),
  plateSpacing: 4,
  plateRotate: true,
  explode: 0,
  splitRatio: 0.5,
  clipboard: [],
  mobilePanel: null,
  toast: null,

  select: (id, dividerId = null) => set({ selectedId: id, selectedIds: id ? [id] : [], selectedDividerId: dividerId }),
  toggleSelect: (id) =>
    set((s) => {
      const has = s.selectedIds.includes(id);
      const selectedIds = has ? s.selectedIds.filter((x) => x !== id) : [...s.selectedIds, id];
      return { selectedIds, selectedId: has ? (selectedIds[selectedIds.length - 1] ?? null) : id, selectedDividerId: null };
    }),
  selectMany: (ids) => set({ selectedIds: ids, selectedId: ids[ids.length - 1] ?? null, selectedDividerId: null }),
  setHover: (id) => set({ hoverId: id }),
  setView: (view) => set({ view }),
  setGridSnap: (gridSnap) => set({ gridSnap }),
  setGridfinity: (gridfinity) => set({ gridfinity, gridSnap: gridfinity ? 0 : 5 }),
  setEdgeSnap: (edgeSnap) => set({ edgeSnap }),
  setShowDrawer: (showDrawer) => set({ showDrawer }),
  setExportOpen: (exportOpen) => set({ exportOpen }),
  setSetupOpen: (setupOpen) => set({ setupOpen }),
  setProjectsOpen: (projectsOpen) => set({ projectsOpen }),
  setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
  setTheme: (themeSetting) => {
    const theme = resolveTheme(themeSetting);
    try {
      window.localStorage.setItem(THEME_KEY, themeSetting);
    } catch {
      // Private browsing; the choice simply will not persist.
    }
    applyTheme(theme);
    set({ themeSetting, theme });
  },
  setLang: (lang) => {
    try {
      window.localStorage.setItem(LANG_KEY, lang);
    } catch {
      // Private browsing; the choice simply will not persist.
    }
    applyLang(lang);
    set({ lang });
  },
  setPlateSpacing: (plateSpacing) => set({ plateSpacing }),
  setPlateRotate: (plateRotate) => set({ plateRotate }),
  setExplode: (explode) => set({ explode }),
  setSplitRatio: (splitRatio) => set({ splitRatio: Math.min(0.8, Math.max(0.2, splitRatio)) }),
  setClipboard: (clipboard) => set({ clipboard }),
  setMobilePanel: (mobilePanel) => set({ mobilePanel }),
  notify: (message, tone = 'info') => {
    toastId += 1;
    const id = toastId;
    set({ toast: { id, message, tone } });
    setTimeout(() => set((s) => (s.toast?.id === id ? { toast: null } : s)), 2600);
  },
}));

// Apply the stored choice immediately, and follow the system when set to 'system'.
applyTheme(useUi.getState().theme);
applyLang(useUi.getState().lang);
if (typeof window !== 'undefined' && window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    const { themeSetting, setTheme } = useUi.getState();
    if (themeSetting === 'system') setTheme('system');
  });
}

/** Palette for the current theme, for SVG and 3D code that cannot use CSS tokens. */
export const usePalette = () => PALETTES[useUi((s) => s.theme)];

/**
 * Translator for the current language. Components call `const t = useT()` and
 * then `t('Bin height')`; re-rendering on a language change comes for free
 * because the hook subscribes to the store.
 */
export function useT() {
  const lang = useUi((s) => s.lang);
  return (key: string, vars?: Vars) => translate(lang, key, vars);
}

/** Translate outside React, for code that has no hook available. */
export const tr = (key: string, vars?: Vars) => translate(useUi.getState().lang, key, vars);
