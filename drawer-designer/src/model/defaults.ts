import type { BuildVolume, Compartment, Design, DrawerConfig, PrintSettings, UsableArea } from './types';

/** Defaults tuned for a 0.4 mm nozzle FDM printer. */
export const DEFAULT_PRINT: PrintSettings = {
  wallThickness: 1.6, // 4 perimeters at 0.4 mm line width
  bottomThickness: 1.2, // 6 layers at 0.2 mm
  cornerStyle: 'rounded',
  cornerRadius: 4,
  binHeight: 48,
  dividerThickness: 1.2, // 3 perimeters
  dividerHeight: null,
  fitGap: 0.5,
};

export const DEFAULT_DRAWER: DrawerConfig = {
  width: 400,
  depth: 300,
  height: 50,
  clearance: 1,
};

export const DEFAULT_BUILD_VOLUME: BuildVolume = { x: 256, y: 256, z: 256 };

/** Smallest compartment footprint the editor allows (mm). */
export const MIN_COMPARTMENT = 12;

/** Space left above the organiser so the drawer closes freely (mm). */
export const TOP_CLEARANCE = 2;

export const LIMITS = {
  drawer: { min: 30, max: 2000 },
  drawerHeight: { min: 8, max: 500 },
  clearance: { min: 0, max: 20 },
  wall: { min: 0.8, max: 6 },
  bottom: { min: 0.6, max: 6 },
  radius: { min: 0, max: 30 },
  dividerThickness: { min: 0.8, max: 5 },
  fitGap: { min: 0, max: 3 },
  binHeight: { min: 5, max: 500 },
  build: { min: 50, max: 1000 },
};

export function usableArea(drawer: DrawerConfig): UsableArea {
  return {
    width: Math.max(0, drawer.width - 2 * drawer.clearance),
    depth: Math.max(0, drawer.depth - 2 * drawer.clearance),
    height: drawer.height,
  };
}

export function compartmentHeight(c: Compartment, print: PrintSettings): number {
  return c.height ?? print.binHeight;
}

export function createEmptyDesign(drawer: DrawerConfig = DEFAULT_DRAWER, print: Partial<PrintSettings> = {}): Design {
  return {
    version: 1,
    name: 'Untitled organiser',
    drawer,
    print: {
      ...DEFAULT_PRINT,
      binHeight: Math.max(LIMITS.binHeight.min, drawer.height - TOP_CLEARANCE),
      ...print,
    },
    buildVolume: { ...DEFAULT_BUILD_VOLUME },
    compartments: [],
  };
}

let idCounter = 0;
export function newId(prefix: string): string {
  idCounter += 1;
  const rand = Math.random().toString(36).slice(2, 7);
  return `${prefix}_${rand}${idCounter.toString(36)}`;
}

export function round(v: number, step = 0.1): number {
  return Math.round(v / step) * step;
}

export function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}
