import { findPrinter, lineWidthFor } from './printers';
import type { Compartment, Design, DrawerConfig, PrinterConfig, PrintSettings, UsableArea } from './types';

/** Defaults tuned for a 0.4 mm nozzle FDM printer. */
export const DEFAULT_PRINT: PrintSettings = {
  wallThickness: 1.68, // 4 perimeters at a 0.42 mm line width
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

/** Default machine: the most common build volume among current desktop printers. */
export const DEFAULT_PRINTER_ID = 'bambu-x2d';

export function printerFromPreset(id: string): PrinterConfig {
  const p = findPrinter(id);
  if (!p) {
    return {
      presetId: 'custom',
      label: 'Custom printer',
      build: { x: 256, y: 256, z: 256 },
      nozzle: 0.4,
      layerHeight: 0.2,
      speed: 200,
      efficiency: 0.55,
      infill: 0.15,
      startupMinutes: 5,
    };
  }
  return {
    presetId: p.id,
    label: `${p.brand} ${p.model}`,
    build: { ...p.build },
    nozzle: p.nozzle,
    layerHeight: 0.2,
    speed: p.speed,
    efficiency: p.efficiency,
    infill: 0.15,
    startupMinutes: 5,
  };
}

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
  nozzle: { min: 0.1, max: 2 },
  layer: { min: 0.04, max: 1.2 },
  speed: { min: 10, max: 1000 },
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

export function createEmptyDesign(
  drawer: DrawerConfig = DEFAULT_DRAWER,
  print: Partial<PrintSettings> = {},
  printerId: string = DEFAULT_PRINTER_ID,
): Design {
  const printer = printerFromPreset(printerId);
  return {
    version: 1,
    name: 'Untitled organiser',
    drawer,
    print: {
      ...DEFAULT_PRINT,
      // Start from a wall that is a whole number of extrusions on this machine.
      wallThickness: wallsForNozzle(printer.nozzle),
      binHeight: Math.max(LIMITS.binHeight.min, drawer.height - TOP_CLEARANCE),
      ...print,
    },
    printer,
    compartments: [],
  };
}

/** Default wall: four perimeters, which is sturdy without being wasteful. */
export function wallsForNozzle(nozzle: number): number {
  return Math.round(4 * lineWidthFor(nozzle) * 100) / 100;
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
