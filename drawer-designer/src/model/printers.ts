/**
 * Printer presets.
 *
 * Figures are the manufacturers' published build volumes and stock nozzle sizes,
 * rounded to whole millimetres. They are a starting point, not gospel: every value
 * stays editable, because firmware updates, modified hot ends and slicer margins
 * all move these numbers. `custom` carries whatever the user set.
 */
import type { BuildVolume } from './types';

export interface PrinterPreset {
  id: string;
  brand: string;
  model: string;
  build: BuildVolume;
  /** Stock nozzle diameter (mm). */
  nozzle: number;
  /** Typical outer-wall print speed used for the time estimate (mm/s). */
  speed: number;
  /** Fraction of nominal speed actually reached, covering acceleration and travel. */
  efficiency: number;
  /** Shown under the picker when the machine has a caveat worth knowing. */
  note?: string;
}

export const PRINTERS: PrinterPreset[] = [
  // --- Bambu Lab -------------------------------------------------------------
  {
    id: 'bambu-x2d',
    brand: 'Bambu Lab',
    model: 'X2D',
    build: { x: 256, y: 256, z: 260 },
    nozzle: 0.4,
    speed: 300,
    efficiency: 0.6,
    note: 'Single-nozzle figure. In dual-nozzle mode the usable width drops to about 235 mm.',
  },
  {
    id: 'bambu-h2d',
    brand: 'Bambu Lab',
    model: 'H2D',
    build: { x: 325, y: 320, z: 325 },
    nozzle: 0.4,
    speed: 300,
    efficiency: 0.6,
    note: 'Single-nozzle figure. In dual-nozzle mode the usable width drops to about 300 mm.',
  },
  { id: 'bambu-x1c', brand: 'Bambu Lab', model: 'X1 Carbon', build: { x: 256, y: 256, z: 256 }, nozzle: 0.4, speed: 300, efficiency: 0.6 },
  { id: 'bambu-p1s', brand: 'Bambu Lab', model: 'P1S', build: { x: 256, y: 256, z: 256 }, nozzle: 0.4, speed: 250, efficiency: 0.58 },
  { id: 'bambu-p1p', brand: 'Bambu Lab', model: 'P1P', build: { x: 256, y: 256, z: 256 }, nozzle: 0.4, speed: 250, efficiency: 0.58 },
  { id: 'bambu-a1', brand: 'Bambu Lab', model: 'A1', build: { x: 256, y: 256, z: 256 }, nozzle: 0.4, speed: 250, efficiency: 0.55 },
  { id: 'bambu-a1-mini', brand: 'Bambu Lab', model: 'A1 mini', build: { x: 180, y: 180, z: 180 }, nozzle: 0.4, speed: 250, efficiency: 0.55 },

  // --- Prusa -----------------------------------------------------------------
  { id: 'prusa-core-one', brand: 'Prusa', model: 'Core One', build: { x: 250, y: 220, z: 270 }, nozzle: 0.4, speed: 200, efficiency: 0.55 },
  { id: 'prusa-mk4s', brand: 'Prusa', model: 'MK4S', build: { x: 250, y: 210, z: 220 }, nozzle: 0.4, speed: 200, efficiency: 0.52 },
  { id: 'prusa-mini', brand: 'Prusa', model: 'MINI+', build: { x: 180, y: 180, z: 180 }, nozzle: 0.4, speed: 150, efficiency: 0.5 },
  { id: 'prusa-xl', brand: 'Prusa', model: 'XL', build: { x: 360, y: 360, z: 360 }, nozzle: 0.4, speed: 200, efficiency: 0.55 },

  // --- Creality --------------------------------------------------------------
  { id: 'creality-k1', brand: 'Creality', model: 'K1 / K1C', build: { x: 220, y: 220, z: 250 }, nozzle: 0.4, speed: 300, efficiency: 0.55 },
  { id: 'creality-k1-max', brand: 'Creality', model: 'K1 Max', build: { x: 300, y: 300, z: 300 }, nozzle: 0.4, speed: 300, efficiency: 0.55 },
  { id: 'creality-ender3', brand: 'Creality', model: 'Ender 3 (V2/V3)', build: { x: 220, y: 220, z: 250 }, nozzle: 0.4, speed: 100, efficiency: 0.45 },

  // --- Others ----------------------------------------------------------------
  { id: 'elegoo-neptune4', brand: 'Elegoo', model: 'Neptune 4', build: { x: 225, y: 225, z: 265 }, nozzle: 0.4, speed: 250, efficiency: 0.5 },
  { id: 'anycubic-kobra3', brand: 'Anycubic', model: 'Kobra 3', build: { x: 250, y: 250, z: 260 }, nozzle: 0.4, speed: 250, efficiency: 0.5 },
  { id: 'voron-350', brand: 'Voron', model: '2.4 (350)', build: { x: 350, y: 350, z: 330 }, nozzle: 0.4, speed: 300, efficiency: 0.6 },
];

export const CUSTOM_PRINTER = 'custom';

export function findPrinter(id: string): PrinterPreset | null {
  return PRINTERS.find((p) => p.id === id) ?? null;
}

/** Presets grouped by brand, in list order, for an <optgroup> picker. */
export function printersByBrand(): { brand: string; items: PrinterPreset[] }[] {
  const out: { brand: string; items: PrinterPreset[] }[] = [];
  for (const p of PRINTERS) {
    const group = out.find((g) => g.brand === p.brand);
    if (group) group.items.push(p);
    else out.push({ brand: p.brand, items: [p] });
  }
  return out;
}

/** Does a printer id still describe the design's current settings? */
export function matchesPreset(id: string, build: BuildVolume, nozzle: number): boolean {
  const p = findPrinter(id);
  if (!p) return false;
  return p.build.x === build.x && p.build.y === build.y && p.build.z === build.z && p.nozzle === nozzle;
}

// ---- nozzle-aware print advice ----------------------------------------------

/** Extrusion width a nozzle produces at a typical 0.45 × diameter ratio (mm). */
export function lineWidthFor(nozzle: number): number {
  return Math.round(nozzle * 1.05 * 100) / 100;
}

export interface WallAdvice {
  /** Number of whole perimeters the wall thickness corresponds to. */
  perimeters: number;
  /** Nearest thickness that is a whole number of perimeters (mm). */
  suggested: number;
  /** True when the wall is not a whole multiple of the line width. */
  awkward: boolean;
  /** True when the wall is thinner than a single extrusion. */
  tooThin: boolean;
}

/**
 * A wall that is not a whole multiple of the extrusion width leaves the slicer with a
 * gap it has to fill with a thin, weak line. Walls thinner than one line cannot print.
 */
export function wallAdvice(wall: number, nozzle: number): WallAdvice {
  const line = lineWidthFor(nozzle);
  const exact = wall / line;
  const perimeters = Math.max(1, Math.round(exact));
  const suggested = Math.round(perimeters * line * 100) / 100;
  return {
    perimeters,
    suggested,
    awkward: Math.abs(exact - perimeters) > 0.12,
    tooThin: wall < line - 1e-6,
  };
}

/** Layer-height advice: multiples of 0.04 mm print best, and 0.75 × nozzle is a sane cap. */
export function maxLayerHeight(nozzle: number): number {
  return Math.round(nozzle * 0.75 * 100) / 100;
}
