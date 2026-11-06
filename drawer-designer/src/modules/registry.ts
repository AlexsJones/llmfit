import { Scope, buildBin, roundedRect } from '../geometry/bin';
import type { BuiltPart, ModuleDefinition, MountingSystem } from './types';

const ICONS = {
  bin: 'M4 6h16v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2zM4 6l2-2h12l2 2',
  cups: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
  screws: 'M12 3v18M8 7h8M9 11h6M10 15h4',
  pen: 'M7 21h10M9 21V9l3-6 3 6v12',
  cable: 'M12 4a8 8 0 1 0 8 8M12 8a4 4 0 1 0 4 4',
  tool: 'M14 4l6 6-9 9H5v-6z',
  stack: 'M4 8l8-4 8 4-8 4zM4 12l8 4 8-4M4 16l8 4 8-4',
  insert: 'M4 4h16v16H4zM8 8h8v8H8z',
};

export const standardBin: ModuleDefinition = {
  kind: 'bin',
  label: 'Storage bin',
  description: 'Open bin with optional fixed dividers.',
  category: 'storage',
  status: 'available',
  icon: ICONS.bin,
  params: [],
  defaults: {},
  supportsDividers: true,
  build: ({ kernel, bin }) => [{ key: 'bin', name: 'Bin', role: 'bin', manifold: buildBin(kernel, bin), offset: [0, 0, 0] }],
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
};

/** Roadmap modules – visible in the library to communicate the ecosystem, not placeable yet. */
const planned = (kind: string, label: string, description: string, icon: string, category: ModuleDefinition['category']): ModuleDefinition => ({
  kind,
  label,
  description,
  category,
  status: 'planned',
  icon,
  params: [],
  defaults: {},
  supportsDividers: false,
});

export const MODULES: ModuleDefinition[] = [
  standardBin,
  cupTray,
  planned('screw-organiser', 'Screw organiser', 'Sloped scoops for screws, nuts and electronics.', ICONS.screws, 'organisers'),
  planned('pen-holder', 'Pen holder', 'Angled cradles for pens and markers.', ICONS.pen, 'organisers'),
  planned('cable-store', 'Cable storage', 'Spools and channels for cables.', ICONS.cable, 'organisers'),
  planned('tool-holder', 'Tool holder', 'Custom cut-outs for hand tools.', ICONS.tool, 'organisers'),
  planned('stackable', 'Stackable tray', 'Bins with a stacking lip.', ICONS.stack, 'storage'),
  planned('custom-insert', 'Custom insert', 'Drop-in insert from an outline.', ICONS.insert, 'inserts'),
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
