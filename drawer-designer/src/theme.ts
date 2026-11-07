/**
 * Palette shared by CSS, the SVG editor and the 3D scene.
 *
 * Tailwind tokens in `index.css` cover ordinary markup, but the editor draws
 * SVG attributes and the 3D scene builds materials, and both need the values as
 * JavaScript. Keeping one table here stops the two drifting apart.
 */
export type ThemeName = 'light' | 'dark';
export type ThemeSetting = ThemeName | 'system';

export interface Palette {
  /** Editor surfaces */
  canvas: string;
  drawerBody: string;
  drawerFloor: string;
  drawerEdge: string;
  grid: string;
  gridStrong: string;
  /** Compartments */
  binFill: string;
  binStroke: string;
  binStrokeHover: string;
  pocketFill: string;
  pocketFillSelected: string;
  pocketStroke: string;
  pocketStrokeSelected: string;
  divider: string;
  dividerSelected: string;
  /** Type */
  label: string;
  labelSelected: string;
  dim: string;
  dimMuted: string;
  labelPlate: string;
  /** Status */
  accent: string;
  accentInk: string;
  accentLine: string;
  warn: string;
  warnSoft: string;
  danger: string;
  dangerSoft: string;
  gap: string;
  gapInk: string;
  /** Module preview tones */
  previewWall: string;
  previewWallSelected: string;
  previewBlock: string;
  previewBlockSelected: string;
  previewHole: string;
  previewHoleStroke: string;
  previewOutline: string;
  /** 3D scene */
  sceneFog: string;
  sceneBackdrop: string;
  partBin: string;
  partInsert: string;
  partSelected: string;
  partEdge: string;
  partEdgeSelected: string;
  partEdgeWarn: string;
  ghostWall: string;
  ghostEdge: string;
  ghostFloor: string;
  hemiGround: string;
}

export const PALETTES: Record<ThemeName, Palette> = {
  light: {
    canvas: '#f6f6f4',
    drawerBody: '#e9e8e4',
    drawerFloor: '#fbfbfa',
    drawerEdge: '#cfcec9',
    grid: '#18181b',
    gridStrong: '#18181b',
    binFill: '#ffffff',
    binStroke: '#c9c8c3',
    binStrokeHover: '#71717a',
    pocketFill: '#f7f7f5',
    pocketFillSelected: '#f1f9f5',
    pocketStroke: '#e4e3df',
    pocketStrokeSelected: '#b5ddcb',
    divider: '#b9b8b2',
    dividerSelected: '#6fb49a',
    label: '#3f3f46',
    labelSelected: '#0b5c42',
    dim: '#8b8b93',
    dimMuted: '#a1a1aa',
    labelPlate: '#ffffff',
    accent: '#0f7a57',
    accentInk: '#0b5c42',
    accentLine: '#b5ddcb',
    warn: '#c2410c',
    warnSoft: '#fff4ec',
    danger: '#dc2626',
    dangerSoft: '#fef2f2',
    gap: '#f97316',
    gapInk: '#c2410c',
    previewWall: '#b9b8b2',
    previewWallSelected: '#6fb49a',
    previewBlock: '#efeeea',
    previewBlockSelected: '#ddf0e7',
    previewHole: '#ffffff',
    previewHoleStroke: '#cfcec9',
    previewOutline: '#a1a1aa',
    sceneFog: '#eeeeea',
    sceneBackdrop: 'radial-gradient(120% 100% at 50% 0%, #f7f7f5 0%, #eceae6 55%, #e4e2dd 100%)',
    partBin: '#f6f6f3',
    partInsert: '#e6e3da',
    partSelected: '#cfe8da',
    partEdge: '#86867f',
    partEdgeSelected: '#0b5c42',
    partEdgeWarn: '#ea580c',
    ghostWall: '#b9c9c1',
    ghostEdge: '#8d9c95',
    ghostFloor: '#e7e6e2',
    hemiGround: '#cfccc4',
  },
  dark: {
    canvas: '#141416',
    drawerBody: '#1d1d20',
    drawerFloor: '#232327',
    drawerEdge: '#34343a',
    grid: '#ffffff',
    gridStrong: '#ffffff',
    binFill: '#2a2a2f',
    binStroke: '#45454d',
    binStrokeHover: '#8b8b93',
    pocketFill: '#212125',
    pocketFillSelected: '#15291f',
    pocketStroke: '#35353b',
    pocketStrokeSelected: '#2c6a4f',
    divider: '#56565e',
    dividerSelected: '#39a37a',
    label: '#d4d4d8',
    labelSelected: '#6fd3a8',
    dim: '#8b8b93',
    dimMuted: '#6b6b73',
    labelPlate: '#2a2a2f',
    accent: '#2fa97c',
    accentInk: '#6fd3a8',
    accentLine: '#2c6a4f',
    warn: '#f59e5b',
    warnSoft: '#2d1d12',
    danger: '#f87171',
    dangerSoft: '#2b1517',
    gap: '#f59e5b',
    gapInk: '#f59e5b',
    previewWall: '#56565e',
    previewWallSelected: '#39a37a',
    previewBlock: '#2f2f35',
    previewBlockSelected: '#1b332a',
    previewHole: '#1a1a1d',
    previewHoleStroke: '#45454d',
    previewOutline: '#6b6b73',
    sceneFog: '#161619',
    sceneBackdrop: 'radial-gradient(120% 100% at 50% 0%, #212126 0%, #171719 55%, #121214 100%)',
    partBin: '#4a4a52',
    partInsert: '#3d3d44',
    partSelected: '#2d6b51',
    partEdge: '#7a7a84',
    partEdgeSelected: '#6fd3a8',
    partEdgeWarn: '#f59e5b',
    ghostWall: '#5a6b63',
    ghostEdge: '#6b7c74',
    ghostFloor: '#2a2a2e',
    hemiGround: '#2a2a2e',
  },
};

/** Grid line opacity differs per theme: dark lines on light, light lines on dark. */
export const GRID_OPACITY: Record<ThemeName, { minor: number; major: number }> = {
  light: { minor: 0.05, major: 0.09 },
  dark: { minor: 0.06, major: 0.11 },
};

export function resolveTheme(setting: ThemeSetting): ThemeName {
  if (setting !== 'system') return setting;
  if (typeof window === 'undefined' || !window.matchMedia) return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
