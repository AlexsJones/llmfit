import { Scope, buildBin, roundedRect } from '../geometry/bin';
import {
  ICONS,
  cableStore,
  cardHolder,
  gridfinityBin,
  gridfinityPlate,
  holeGrid,
  outlineInsert,
  penHolder,
  pocket,
  screwOrganiser,
  stackable,
  toolHolder,
} from './library';
import type { BuiltPart, ModuleDefinition, MountingSystem, PreviewShape } from './types';

/** Scoop radius scaled to the bin, capped for very tall or shallow bins (mm). */
export function scoopRadiusFor(innerHeight: number, innerDepth: number): number {
  return Math.max(0, Math.min(15, innerHeight * 0.6, innerDepth * 0.35));
}

/** Label ledge depth (mm). */
export function labelTabFor(innerDepth: number): number {
  return Math.max(0, Math.min(12, innerDepth * 0.25));
}

export const standardBin: ModuleDefinition = {
  kind: 'bin',
  label: 'Storage bin',
  description: 'Open bin with optional fixed dividers.',
  category: 'storage',
  status: 'available',
  icon: ICONS.bin,
  params: [
    { key: 'scoop', label: 'Finger scoop', type: 'boolean', help: 'Rounded ramp along the front wall so small parts slide out.' },
    { key: 'labelTab', label: 'Label ledge', type: 'boolean', help: 'Support-free ledge along the back rim for labels.' },
  ],
  defaults: { scoop: false, labelTab: false },
  supportsDividers: true,
  build: ({ kernel, bin, params }) => {
    const inner = bin.depth - 2 * bin.wall;
    const spec = {
      ...bin,
      scoopRadius: params.scoop ? scoopRadiusFor(bin.height - bin.bottom, inner) : 0,
      labelTab: params.labelTab ? labelTabFor(inner) : 0,
    };
    return [{ key: 'bin', name: 'Bin', role: 'bin', manifold: buildBin(kernel, spec), offset: [0, 0, 0] }];
  },
  preview: ({ bin, params, gap }) => {
    const { iw, id } = pocket(bin);
    const x = gap / 2 + bin.wall;
    const y = gap / 2 + bin.wall;
    const out: PreviewShape[] = [];
    if (params.labelTab) out.push({ kind: 'rect', x, y, w: iw, d: labelTabFor(id), tone: 'block' });
    if (params.scoop) {
      const r = scoopRadiusFor(bin.height - bin.bottom, id);
      out.push({ kind: 'rect', x, y: y + id - r, w: iw, d: r, tone: 'scoop' });
    }
    return out;
  },
};

/** A tray holding a grid of small removable cups – the first "accessory" style module. */
export const cupTray: ModuleDefinition = {
  kind: 'cup-tray',
  label: 'Removable cups',
  description: 'Tray with a grid of lift-out cups for small parts.',
  category: 'inserts',
  status: 'available',
  icon: ICONS.cups,
  params: [
    { key: 'cols', label: 'Columns', type: 'number', min: 1, max: 12, step: 1 },
    { key: 'rows', label: 'Rows', type: 'number', min: 1, max: 12, step: 1 },
    { key: 'tolerance', label: 'Cup clearance', type: 'number', min: 0.1, max: 2, step: 0.05, unit: 'mm', help: 'Gap around each cup so it lifts out easily.' },
    { key: 'cupWall', label: 'Cup wall', type: 'number', min: 0.8, max: 3, step: 0.1, unit: 'mm' },
    { key: 'headroom', label: 'Cup headroom', type: 'number', min: 0, max: 20, step: 0.5, unit: 'mm', help: 'How far the cup rim sits below the tray rim.' },
  ],
  defaults: { cols: 2, rows: 2, tolerance: 0.4, cupWall: 1.2, headroom: 1 },
  supportsDividers: false,
  build: ({ kernel, bin, params }) => {
    const tray = buildBin(kernel, { ...bin, dividers: [] });
    const parts: BuiltPart[] = [{ key: 'tray', name: 'Tray', role: 'bin', manifold: tray, offset: [0, 0, 0] }];
    const cols = Math.max(1, Math.round(Number(params.cols)));
    const rows = Math.max(1, Math.round(Number(params.rows)));
    const tol = Number(params.tolerance);
    const cupWall = Number(params.cupWall);
    const iw = bin.width - 2 * bin.wall;
    const id = bin.depth - 2 * bin.wall;
    // Cups share the pocket evenly; each keeps `tol` clearance to walls and neighbours.
    const cw = (iw - tol * (cols + 1)) / cols;
    const cd = (id - tol * (rows + 1)) / rows;
    const ch = bin.height - bin.bottom - Number(params.headroom);
    const innerR = bin.cornerRadius > 0 ? Math.max(bin.cornerRadius - bin.wall - tol, 1) : 0;
    if (cw < 2 * cupWall + 2 || cd < 2 * cupWall + 2 || ch < bin.bottom + 2) return parts;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cup = buildBin(kernel, {
          width: cw,
          depth: cd,
          height: ch,
          wall: cupWall,
          bottom: bin.bottom,
          cornerRadius: innerR,
          dividers: [],
          dividerThickness: bin.dividerThickness,
          dividerHeight: ch,
        });
        parts.push({
          key: `cup-${r}-${c}`,
          name: `Cup ${r + 1}·${c + 1}`,
          role: 'insert',
          manifold: cup,
          offset: [bin.wall + tol + c * (cw + tol), bin.depth - bin.wall - tol - (r + 1) * cd - r * tol, bin.bottom],
        });
      }
    }
    return parts;
  },
  preview: ({ bin, params, gap }) => {
    const cols = Math.max(1, Math.round(Number(params.cols)));
    const rows = Math.max(1, Math.round(Number(params.rows)));
    const tol = Number(params.tolerance);
    const { iw, id } = pocket(bin);
    const cw = (iw - tol * (cols + 1)) / cols;
    const cd = (id - tol * (rows + 1)) / rows;
    if (cw <= 0 || cd <= 0) return [];
    const out: PreviewShape[] = [];
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        out.push({
          kind: 'rect',
          x: gap / 2 + bin.wall + tol + c * (cw + tol),
          y: gap / 2 + bin.wall + tol + r * (cd + tol),
          w: cw,
          d: cd,
          r: Math.min(2, cw / 4),
          tone: 'hole',
        });
    return out;
  },
};

export const MODULES: ModuleDefinition[] = [
  standardBin,
  stackable,
  cupTray,
  holeGrid,
  screwOrganiser,
  penHolder,
  cableStore,
  toolHolder,
  cardHolder,
  outlineInsert,
  gridfinityBin,
  gridfinityPlate,
];

const byKind = new Map(MODULES.map((m) => [m.kind, m]));

export function getModule(kind: string): ModuleDefinition {
  return byKind.get(kind) ?? standardBin;
}

export const MOUNTING_SYSTEMS: MountingSystem[] = [{ id: 'none', label: 'None (free-standing)' }];

export function getMounting(id: string): MountingSystem {
  return MOUNTING_SYSTEMS.find((m) => m.id === id) ?? MOUNTING_SYSTEMS[0];
}

// Re-exported so future modules can build custom outlines with the same helpers.
export { Scope, roundedRect };
