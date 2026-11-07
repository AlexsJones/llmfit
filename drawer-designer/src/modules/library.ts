/**
 * Accessory modules built on top of the base bin.
 *
 * Coordinate helpers: modules reason in "pocket" coordinates (u from the
 * pocket's left wall, v from its back wall, both in mm). These map to the
 * bin's geometry frame (Y grows towards the front → back) and to the 2D
 * editor's layout frame (y grows towards the front) with `geo*` / `lay*`.
 */
import type { Manifold } from 'manifold-3d';
import { type BinSpec, CIRCLE_SEGMENTS, Scope, buildBin, innerRadius, roundedRect } from '../geometry/bin';
import type { Kernel } from '../geometry/kernel';
import { decodeOutline, fitOutline, growPolygon, polygonArea } from '../geometry/outline';
import type { ParamValue } from '../model/types';
import type { BuiltPart, ModuleDefinition, ModulePreviewContext, PreviewShape } from './types';

export const ICONS = {
  bin: 'M4 6h16v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2zM4 6l2-2h12l2 2',
  cups: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
  screws: 'M4 4h16v16H4zM12 4v16M4 12h16M6 10c2 0 3-1 4-2M14 10c2 0 3-1 4-2',
  pen: 'M3 8h18M3 13h18M3 18h18M6 6v4M6 11v4M6 16v4',
  cable: 'M4 6h16v12H4zM9.5 6v12M14.5 6v12M7 6a1.2 1.2 0 0 0 0 2.5M12 6a1.2 1.2 0 0 0 0 2.5M17 6a1.2 1.2 0 0 0 0 2.5',
  tool: 'M4 4h16v16H4zM8 7v10M12 7v10M16 7v10',
  stack: 'M4 8l8-4 8 4-8 4zM4 12l8 4 8-4M4 16l8 4 8-4',
  cards: 'M4 5h16v14H4zM8 5v14M12 5v14M16 5v14',
  outline: 'M4 4h16v16H4zM9 16c-1-3 1-5 2-7s3-1 3 1-2 3-1 5',
  insert: 'M4 4h16v16H4zM8.5 8.5h.01M12 8.5h.01M15.5 8.5h.01M8.5 12h.01M12 12h.01M15.5 12h.01M8.5 15.5h.01M12 15.5h.01M15.5 15.5h.01',
};

// ---- helpers ------------------------------------------------------------------

const int = (p: Record<string, ParamValue>, k: string, min = 1) => Math.max(min, Math.round(Number(p[k])));
const num = (p: Record<string, ParamValue>, k: string) => Number(p[k]);

export function pocket(bin: BinSpec) {
  return { iw: bin.width - 2 * bin.wall, id: bin.depth - 2 * bin.wall };
}
/** Pocket u → bin geometry X. */
const geoX = (bin: BinSpec, u: number) => bin.wall + u;
/** Pocket v (from back) → bin geometry Y (from front). */
const geoY = (bin: BinSpec, v: number) => bin.depth - bin.wall - v;
/** Pocket u → layout x inside the compartment rectangle. */
const layX = (ctx: ModulePreviewContext, u: number) => ctx.gap / 2 + ctx.bin.wall + u;
/** Pocket v → layout y inside the compartment rectangle. */
const layY = (ctx: ModulePreviewContext, v: number) => ctx.gap / 2 + ctx.bin.wall + v;

/**
 * Material always left under a pocket, so a cut never lands exactly on the bin's floor
 * face (which would leave coincident vertices) and the floor stays printable (mm).
 */
export const MIN_FLOOR_UNDER_CUT = 0.8;

/**
 * Gap kept between the top of a filled insert block and the bin's rim. Ending the block
 * flush with the rim would make its top face coplanar with the walls' top face (mm).
 */
export const RIM_STEP = 0.8;

/** Top of the filled block and the height a pocket of `depth` starts at. */
export function cutRange(bin: BinSpec, depth: number) {
  const floor = bin.bottom + MIN_FLOOR_UNDER_CUT;
  const top = Math.min(bin.height - RIM_STEP, Math.max(floor + 1, bin.bottom + depth + 1));
  const z0 = Math.max(floor, top - depth);
  return { top, z0, ok: top - z0 > 0.2 && top > floor };
}

/** A solid block filling the pocket from the floor up to `top`, slightly overlapping walls and floor so it fuses. */
function pocketBlock(k: Kernel, s: Scope, bin: BinSpec, top: number): Manifold {
  const { iw, id } = pocket(bin);
  const outline = s.keep(roundedRect(k, s, iw, id, innerRadius(bin)).translate([bin.wall, bin.wall]));
  const grown = s.keep(outline.offset(0.2, 'Round', 2, CIRCLE_SEGMENTS));
  return s.keep(s.keep(grown.extrude(top - bin.bottom + 0.5)).translate([0, 0, bin.bottom - 0.5]));
}

function pocketPreview(ctx: ModulePreviewContext): PreviewShape {
  const { iw, id } = pocket(ctx.bin);
  return { kind: 'rect', x: layX(ctx, 0), y: layY(ctx, 0), w: iw, d: id, r: innerRadius(ctx.bin), tone: 'block' };
}

/** Run a module build with its own scope; the base bin is built from `base`, then `shape` adds/cuts features. */
function withBase(k: Kernel, base: BinSpec, shape: (s: Scope, body: Manifold) => Manifold): BuiltPart[] {
  const s = new Scope();
  const bin = s.keep(buildBin(k, base));
  const body = shape(s, bin);
  s.dispose(body);
  return [{ key: 'bin', name: 'Bin', role: 'bin', manifold: body, offset: [0, 0, 0] }];
}

function evenly(n: number, span: number): number[] {
  return Array.from({ length: Math.max(0, n - 1) }, (_, i) => (span * (i + 1)) / n);
}

// ---- small-parts organiser -----------------------------------------------------

function screwScoopRadius(bin: BinSpec, rows: number) {
  const { id } = pocket(bin);
  return Math.max(0, Math.min(12, (bin.height - bin.bottom) * 0.6, (id / rows) * 0.4, bin.dividerHeight - bin.bottom - 0.5));
}

export const screwOrganiser: ModuleDefinition = {
  kind: 'screw-organiser',
  label: 'Small-parts organiser',
  description: 'Grid of cells with scooped fronts for screws and electronics.',
  category: 'organisers',
  status: 'available',
  icon: ICONS.screws,
  params: [
    { key: 'cols', label: 'Columns', type: 'number', min: 1, max: 10, step: 1 },
    { key: 'rows', label: 'Rows', type: 'number', min: 1, max: 8, step: 1 },
  ],
  defaults: { cols: 3, rows: 2 },
  supportsDividers: false,
  build: ({ kernel, bin, params }) => {
    const cols = int(params, 'cols');
    const rows = int(params, 'rows');
    const { iw, id } = pocket(bin);
    const t = bin.dividerThickness;
    const r = screwScoopRadius(bin, rows);
    // Horizontal dividers measured from the front (geometry Y); each cell's scoop starts at the divider's back face.
    const fronts = evenly(rows, id).map((y) => bin.wall + y);
    const spec: BinSpec = {
      ...bin,
      dividers: [
        ...evenly(cols, iw).map((u) => ({ orientation: 'vertical' as const, at: geoX(bin, u) })),
        ...fronts.map((y) => ({ orientation: 'horizontal' as const, at: y })),
      ],
      extraScoops: [{ y0: bin.wall, r }, ...fronts.map((y) => ({ y0: y + t / 2, r }))],
    };
    return [{ key: 'bin', name: 'Bin', role: 'bin', manifold: buildBin(kernel, spec), offset: [0, 0, 0] }];
  },
  preview: (ctx) => {
    const { bin, params } = ctx;
    const cols = int(params, 'cols');
    const rows = int(params, 'rows');
    const { iw, id } = pocket(bin);
    const t = bin.dividerThickness;
    const r = screwScoopRadius(bin, rows);
    const out: PreviewShape[] = [];
    // Scoops sit at the front (larger v) of every row.
    for (let j = 0; j < rows; j++) {
      const front = id - (id * j) / rows - (j > 0 ? t / 2 : 0);
      if (r >= 2) out.push({ kind: 'rect', x: layX(ctx, 0), y: layY(ctx, front - r), w: iw, d: r, tone: 'scoop' });
    }
    for (const u of evenly(cols, iw)) out.push({ kind: 'rect', x: layX(ctx, u) - t / 2, y: layY(ctx, 0), w: t, d: id, tone: 'wall' });
    for (const v of evenly(rows, id)) out.push({ kind: 'rect', x: layX(ctx, 0), y: layY(ctx, v) - t / 2, w: iw, d: t, tone: 'wall' });
    return out;
  },
};

// ---- pen tray -----------------------------------------------------------------------

function penLayout(bin: BinSpec, params: Record<string, ParamValue>) {
  const { id } = pocket(bin);
  const rad = (num(params, 'diameter') + 0.6) / 2; // 0.6 mm clearance
  const n = Math.max(0, Math.min(int(params, 'slots'), Math.floor(id / (2 * rad + 1.2))));
  const pitch = n ? id / n : 0;
  const centreZ = bin.bottom + 1.2 + rad; // 1.2 mm of material under each cradle
  const top = Math.min(bin.height - 0.5, centreZ);
  return { rad, n, pitch, centreZ, top };
}

export const penHolder: ModuleDefinition = {
  kind: 'pen-holder',
  label: 'Pen tray',
  description: 'Rounded cradles that keep pens and markers in rows.',
  category: 'organisers',
  status: 'available',
  icon: ICONS.pen,
  params: [
    { key: 'slots', label: 'Cradles', type: 'number', min: 1, max: 16, step: 1 },
    { key: 'diameter', label: 'Pen diameter', type: 'number', min: 5, max: 30, step: 0.5, unit: 'mm' },
  ],
  defaults: { slots: 4, diameter: 12 },
  supportsDividers: false,
  build: ({ kernel: k, bin, params }) => {
    const { rad, n, pitch, centreZ, top } = penLayout(bin, params);
    const { iw } = pocket(bin);
    return withBase(k, { ...bin, dividers: [] }, (s, body) => {
      if (!n || top - bin.bottom < 1) return body;
      let out = s.keep(body.add(pocketBlock(k, s, bin, top)));
      const cutters: Manifold[] = [];
      for (let i = 0; i < n; i++) {
        const y = geoY(bin, pitch * (i + 0.5));
        const cyl = s.keep(k.Manifold.cylinder(iw + 4, rad, rad, CIRCLE_SEGMENTS));
        cutters.push(s.keep(s.keep(cyl.rotate([0, 90, 0])).translate([bin.wall - 2, y, centreZ])));
      }
      out = s.keep(out.subtract(s.keep(k.Manifold.union(cutters))));
      return out;
    });
  },
  preview: (ctx) => {
    const { rad, n, pitch } = penLayout(ctx.bin, ctx.params);
    const { iw } = pocket(ctx.bin);
    const out: PreviewShape[] = [pocketPreview(ctx)];
    for (let i = 0; i < n; i++) {
      out.push({ kind: 'rect', x: layX(ctx, 0), y: layY(ctx, pitch * (i + 0.5) - rad), w: iw, d: 2 * rad, r: rad, tone: 'hole' });
    }
    return out;
  },
};

// ---- cable organiser ------------------------------------------------------------------

function cableLayout(bin: BinSpec, params: Record<string, ParamValue>) {
  const { iw } = pocket(bin);
  const lanes = int(params, 'lanes');
  const laneW = iw / lanes;
  const r = Math.max(0, Math.min(num(params, 'notch') / 2, (bin.height - bin.bottom) / 2 - 1, laneW / 2 - 1));
  return { lanes, laneW, r };
}

export const cableStore: ModuleDefinition = {
  kind: 'cable-store',
  label: 'Cable organiser',
  description: 'Lanes for coiled cables with exit notches in the front wall.',
  category: 'organisers',
  status: 'available',
  icon: ICONS.cable,
  params: [
    { key: 'lanes', label: 'Lanes', type: 'number', min: 1, max: 10, step: 1 },
    { key: 'notch', label: 'Notch width', type: 'number', min: 4, max: 24, step: 0.5, unit: 'mm', help: 'Width of the cable exit cut into the front wall.' },
  ],
  defaults: { lanes: 3, notch: 8 },
  supportsDividers: false,
  build: ({ kernel: k, bin, params }) => {
    const { lanes, laneW, r } = cableLayout(bin, params);
    const dividers = evenly(lanes, pocket(bin).iw).map((u) => ({ orientation: 'vertical' as const, at: geoX(bin, u) }));
    return withBase(k, { ...bin, dividers }, (s, body) => {
      if (r < 1.5) return body;
      const cutters: Manifold[] = [];
      for (let i = 0; i < lanes; i++) {
        const cyl = s.keep(k.Manifold.cylinder(bin.wall + 2, r, r, CIRCLE_SEGMENTS));
        // Rotate the cylinder axis from Z to +Y, then centre it on the rim of the front wall.
        cutters.push(s.keep(s.keep(cyl.rotate([-90, 0, 0])).translate([geoX(bin, laneW * (i + 0.5)), -1, bin.height])));
      }
      return s.keep(body.subtract(s.keep(k.Manifold.union(cutters))));
    });
  },
  preview: (ctx) => {
    const { lanes, laneW, r } = cableLayout(ctx.bin, ctx.params);
    const { id } = pocket(ctx.bin);
    const t = ctx.bin.dividerThickness;
    const out: PreviewShape[] = [];
    for (const u of evenly(lanes, pocket(ctx.bin).iw)) out.push({ kind: 'rect', x: layX(ctx, u) - t / 2, y: layY(ctx, 0), w: t, d: id, tone: 'wall' });
    if (r >= 1.5) {
      for (let i = 0; i < lanes; i++) {
        out.push({ kind: 'rect', x: layX(ctx, laneW * (i + 0.5)) - r, y: layY(ctx, id), w: 2 * r, d: ctx.bin.wall, tone: 'hole' });
      }
    }
    return out;
  },
};

// ---- tool holder ----------------------------------------------------------------------

const TOOL_MARGIN = 6;

function toolLayout(bin: BinSpec, params: Record<string, ParamValue>) {
  const { iw, id } = pocket(bin);
  const want = num(params, 'slotWidth');
  const n = Math.max(0, Math.min(int(params, 'slots'), Math.floor(iw / (Math.min(want, iw) + 2))));
  const pitch = n ? iw / n : 0;
  const sw = Math.min(want, pitch - 2);
  const len = id - 2 * TOOL_MARGIN;
  const { top, z0, ok: deep } = cutRange(bin, num(params, 'cutDepth'));
  const grip = params.grip ? Math.min(sw + 14, pitch - 1.5, len) : 0;
  return { n, pitch, sw, len, top, z0, grip, ok: n > 0 && sw >= 2 && len >= sw && deep };
}

export const toolHolder: ModuleDefinition = {
  kind: 'tool-holder',
  label: 'Tool holder',
  description: 'Solid insert with rounded cut-outs for screwdrivers and hand tools.',
  category: 'organisers',
  status: 'available',
  icon: ICONS.tool,
  params: [
    { key: 'slots', label: 'Slots', type: 'number', min: 1, max: 12, step: 1 },
    { key: 'slotWidth', label: 'Slot width', type: 'number', min: 4, max: 80, step: 0.5, unit: 'mm' },
    { key: 'cutDepth', label: 'Cut depth', type: 'number', min: 3, max: 120, step: 0.5, unit: 'mm' },
    { key: 'grip', label: 'Finger reliefs', type: 'boolean', help: 'Round pockets in the middle of each slot so tools are easy to grab.' },
  ],
  defaults: { slots: 3, slotWidth: 18, cutDepth: 20, grip: true },
  supportsDividers: false,
  build: ({ kernel: k, bin, params }) => {
    const L = toolLayout(bin, params);
    return withBase(k, { ...bin, dividers: [] }, (s, body) => {
      if (!L.ok) return body;
      let out = s.keep(body.add(pocketBlock(k, s, bin, L.top)));
      const outlines = [];
      for (let i = 0; i < L.n; i++) {
        const cx = geoX(bin, L.pitch * (i + 0.5));
        outlines.push(s.keep(roundedRect(k, s, L.sw, L.len, L.sw / 2).translate([cx - L.sw / 2, bin.wall + TOOL_MARGIN])));
        if (L.grip > L.sw) outlines.push(s.keep(s.keep(k.CrossSection.circle(L.grip / 2, CIRCLE_SEGMENTS)).translate([cx, bin.depth / 2])));
      }
      const cut = s.keep(s.keep(s.keep(k.CrossSection.union(outlines)).extrude(L.top - L.z0 + 1)).translate([0, 0, L.z0]));
      out = s.keep(out.subtract(cut));
      return out;
    });
  },
  preview: (ctx) => {
    const L = toolLayout(ctx.bin, ctx.params);
    const { id } = pocket(ctx.bin);
    const out: PreviewShape[] = [pocketPreview(ctx)];
    if (!L.ok) return out;
    for (let i = 0; i < L.n; i++) {
      const cu = L.pitch * (i + 0.5);
      out.push({ kind: 'rect', x: layX(ctx, cu - L.sw / 2), y: layY(ctx, TOOL_MARGIN), w: L.sw, d: L.len, r: L.sw / 2, tone: 'hole' });
      if (L.grip > L.sw) out.push({ kind: 'circle', cx: layX(ctx, cu), cy: layY(ctx, id / 2), r: L.grip / 2, tone: 'hole' });
    }
    return out;
  },
};

// ---- stackable bin ----------------------------------------------------------------------

export const stackable: ModuleDefinition = {
  kind: 'stackable',
  label: 'Stackable bin',
  description: 'Bin with a stepped foot that nests into the bin below.',
  category: 'storage',
  status: 'available',
  icon: ICONS.stack,
  params: [
    { key: 'footHeight', label: 'Foot height', type: 'number', min: 1.5, max: 8, step: 0.5, unit: 'mm' },
    { key: 'tolerance', label: 'Stack clearance', type: 'number', min: 0.1, max: 1, step: 0.05, unit: 'mm', help: 'Gap between the foot and the pocket of the bin below.' },
  ],
  defaults: { footHeight: 3, tolerance: 0.3 },
  supportsDividers: false,
  build: ({ kernel: k, bin, params }) => {
    const foot = num(params, 'footHeight');
    const tol = num(params, 'tolerance');
    // The floor must reach above the foot step so the walls stay connected to it.
    const base: BinSpec = { ...bin, dividers: [], bottom: Math.max(bin.bottom, foot + 0.8) };
    const { iw, id } = pocket(bin);
    return withBase(k, base, (s, body) => {
      if (iw - 2 * tol < 4 || id - 2 * tol < 4 || foot >= bin.height - base.bottom) return body;
      const r = Math.max(0, bin.cornerRadius);
      const outer = s.keep(roundedRect(k, s, bin.width + 2, bin.depth + 2, r > 0 ? r + 1 : 0).translate([-1, -1]));
      const ri = innerRadius(bin);
      const footprint = s.keep(roundedRect(k, s, iw - 2 * tol, id - 2 * tol, ri > 0 ? Math.max(ri - tol, 0.5) : 0).translate([bin.wall + tol, bin.wall + tol]));
      const ring = s.keep(s.keep(s.keep(outer.subtract(footprint)).extrude(foot + 1)).translate([0, 0, -1]));
      return s.keep(body.subtract(ring));
    });
  },
  preview: (ctx) => {
    const tol = num(ctx.params, 'tolerance');
    const { iw, id } = pocket(ctx.bin);
    return [{ kind: 'rect', x: layX(ctx, tol), y: layY(ctx, tol), w: iw - 2 * tol, d: id - 2 * tol, r: innerRadius(ctx.bin), tone: 'outline' }];
  },
};

// ---- hole grid insert -------------------------------------------------------------------

function holeLayout(bin: BinSpec, params: Record<string, ParamValue>) {
  const { iw, id } = pocket(bin);
  const cols = int(params, 'cols');
  const rows = int(params, 'rows');
  const cw = iw / cols;
  const cd = id / rows;
  const size = Math.min(num(params, 'size'), cw - 1.2, cd - 1.2);
  const { top, z0, ok: deep } = cutRange(bin, num(params, 'depth'));
  return { cols, rows, cw, cd, size, top, z0, round: params.shape !== 'square', ok: size >= 1 && deep };
}

export const holeGrid: ModuleDefinition = {
  kind: 'custom-insert',
  label: 'Hole grid insert',
  description: 'Solid insert with round or square holes for bits, batteries and vials.',
  category: 'inserts',
  status: 'available',
  icon: ICONS.insert,
  params: [
    {
      key: 'shape',
      label: 'Hole shape',
      type: 'select',
      options: [
        { value: 'round', label: 'Round' },
        { value: 'square', label: 'Square' },
      ],
    },
    { key: 'size', label: 'Hole size', type: 'number', min: 2, max: 80, step: 0.1, unit: 'mm', help: 'Diameter (round) or side length (square). Add ~0.3 mm for a slip fit.' },
    { key: 'cols', label: 'Columns', type: 'number', min: 1, max: 30, step: 1 },
    { key: 'rows', label: 'Rows', type: 'number', min: 1, max: 30, step: 1 },
    { key: 'depth', label: 'Hole depth', type: 'number', min: 2, max: 120, step: 0.5, unit: 'mm' },
  ],
  defaults: { shape: 'round', size: 7.3, cols: 6, rows: 3, depth: 12 },
  supportsDividers: false,
  build: ({ kernel: k, bin, params }) => {
    const L = holeLayout(bin, params);
    return withBase(k, { ...bin, dividers: [] }, (s, body) => {
      if (!L.ok) return body;
      let out = s.keep(body.add(pocketBlock(k, s, bin, L.top)));
      const h = L.top - L.z0 + 1;
      const cutters: Manifold[] = [];
      for (let r = 0; r < L.rows; r++) {
        for (let c = 0; c < L.cols; c++) {
          const x = geoX(bin, L.cw * (c + 0.5));
          const y = geoY(bin, L.cd * (r + 0.5));
          const hole = L.round
            ? s.keep(k.Manifold.cylinder(h, L.size / 2, L.size / 2, 48)).translate([x, y, L.z0])
            : s.keep(k.Manifold.cube([L.size, L.size, h])).translate([x - L.size / 2, y - L.size / 2, L.z0]);
          cutters.push(s.keep(hole));
        }
      }
      out = s.keep(out.subtract(s.keep(k.Manifold.union(cutters))));
      return out;
    });
  },
  preview: (ctx) => {
    const L = holeLayout(ctx.bin, ctx.params);
    const out: PreviewShape[] = [pocketPreview(ctx)];
    if (!L.ok) return out;
    for (let r = 0; r < L.rows; r++) {
      for (let c = 0; c < L.cols; c++) {
        const cx = layX(ctx, L.cw * (c + 0.5));
        const cy = layY(ctx, L.cd * (r + 0.5));
        out.push(
          L.round
            ? { kind: 'circle', cx, cy, r: L.size / 2, tone: 'hole' }
            : { kind: 'rect', x: cx - L.size / 2, y: cy - L.size / 2, w: L.size, d: L.size, tone: 'hole' },
        );
      }
    }
    return out;
  },
};


// ---- card and stick holder ----------------------------------------------------------

/** Slot presets in mm: [thickness, length]. Real card sizes plus a printing allowance. */
export const CARD_PRESETS: Record<string, { label: string; t: number; len: number }> = {
  sd: { label: 'SD card', t: 2.6, len: 25.5 },
  microsd: { label: 'microSD', t: 1.4, len: 12.5 },
  'usb-a': { label: 'USB-A stick', t: 9, len: 20 },
  'usb-c': { label: 'USB-C stick', t: 6, len: 13 },
  cf: { label: 'CF card', t: 4.2, len: 44 },
  sim: { label: 'SIM tray / pick', t: 1.6, len: 26 },
};

function cardLayout(bin: BinSpec, params: Record<string, ParamValue>) {
  const preset = CARD_PRESETS[String(params.preset)] ?? CARD_PRESETS.sd;
  const { iw, id } = pocket(bin);
  const t = preset.t + num(params, 'clearance');
  const len = Math.min(preset.len + num(params, 'clearance'), id - 2);
  const wallBetween = Math.max(0.8, num(params, 'rib'));
  const pitch = t + wallBetween;
  const n = Math.max(0, Math.min(int(params, 'slots', 0), Math.floor((iw - wallBetween) / pitch)));
  const used = n * pitch - wallBetween;
  const u0 = (iw - used) / 2;
  const { top, z0, ok: deep } = cutRange(bin, num(params, 'slotDepth'));
  const relief = params.relief ? Math.min(14, id / 3) : 0;
  return { preset, t, len, pitch, n, u0, top, z0, relief, ok: n > 0 && len >= 4 && deep };
}

export const cardHolder: ModuleDefinition = {
  kind: 'card-holder',
  label: 'Card & stick holder',
  description: 'Upright slots for SD cards, USB sticks and similar.',
  category: 'organisers',
  status: 'available',
  icon: ICONS.cards,
  params: [
    {
      key: 'preset',
      label: 'Holds',
      type: 'select',
      options: Object.entries(CARD_PRESETS).map(([value, p]) => ({ value, label: `${p.label} (${p.t} × ${p.len} mm)` })),
    },
    { key: 'slots', label: 'Slots', type: 'number', min: 1, max: 60, step: 1 },
    { key: 'slotDepth', label: 'Slot depth', type: 'number', min: 3, max: 120, step: 0.5, unit: 'mm' },
    { key: 'clearance', label: 'Clearance', type: 'number', min: 0, max: 2, step: 0.1, unit: 'mm', help: 'Added to the nominal size so cards slide in and out.' },
    { key: 'rib', label: 'Rib thickness', type: 'number', min: 0.8, max: 5, step: 0.1, unit: 'mm', help: 'Material left between two slots.' },
    { key: 'relief', label: 'Grab notch', type: 'boolean', help: 'Cross cut-out in the middle so you can pinch a card out.' },
  ],
  defaults: { preset: 'sd', slots: 8, slotDepth: 16, clearance: 0.5, rib: 1.6, relief: true },
  supportsDividers: false,
  build: ({ kernel: k, bin, params }) => {
    const L = cardLayout(bin, params);
    return withBase(k, { ...bin, dividers: [] }, (s, body) => {
      if (!L.ok) return body;
      let out = s.keep(body.add(pocketBlock(k, s, bin, L.top)));
      const { id } = pocket(bin);
      const v0 = (id - L.len) / 2;
      const outlines = [];
      for (let i = 0; i < L.n; i++) {
        const u = L.u0 + i * L.pitch;
        outlines.push(s.keep(k.CrossSection.square([L.t, L.len]).translate([geoX(bin, u), bin.wall + v0])));
      }
      if (L.relief >= 4) {
        // One notch across all slots so a card can be pinched out.
        const span = L.n * L.pitch - (L.pitch - L.t);
        outlines.push(
          s.keep(
            roundedRect(k, s, span, L.relief, L.relief / 2).translate([geoX(bin, L.u0), bin.depth / 2 - L.relief / 2]),
          ),
        );
      }
      const cut = s.keep(s.keep(s.keep(k.CrossSection.union(outlines)).extrude(L.top - L.z0 + 1)).translate([0, 0, L.z0]));
      out = s.keep(out.subtract(cut));
      return out;
    });
  },
  preview: (ctx) => {
    const L = cardLayout(ctx.bin, ctx.params);
    const { id } = pocket(ctx.bin);
    const out: PreviewShape[] = [pocketPreview(ctx)];
    if (!L.ok) return out;
    const v0 = (id - L.len) / 2;
    for (let i = 0; i < L.n; i++) {
      out.push({ kind: 'rect', x: layX(ctx, L.u0 + i * L.pitch), y: layY(ctx, v0), w: L.t, d: L.len, tone: 'hole' });
    }
    if (L.relief >= 4) {
      const span = L.n * L.pitch - (L.pitch - L.t);
      out.push({ kind: 'rect', x: layX(ctx, L.u0), y: layY(ctx, id / 2 - L.relief / 2), w: span, d: L.relief, r: L.relief / 2, tone: 'hole' });
    }
    return out;
  },
};

// ---- custom outline insert -------------------------------------------------------

function outlineLayout(bin: BinSpec, params: Record<string, ParamValue>) {
  const { iw, id } = pocket(bin);
  const pts = decodeOutline(params.outline);
  const count = Math.max(1, Math.min(int(params, 'count'), 12));
  const margin = num(params, 'margin');
  const cellW = iw / count;
  const { top, z0, ok: deep } = cutRange(bin, num(params, 'depth'));
  const tol = num(params, 'tolerance');
  const shape = pts.length >= 3 ? fitOutline(growPolygon(pts, tol / Math.max(cellW, id)), cellW, id, margin) : [];
  return { pts, count, cellW, id, top, z0, shape, ok: shape.length >= 3 && deep };
}

export const outlineInsert: ModuleDefinition = {
  kind: 'outline-insert',
  label: 'Custom shape insert',
  description: 'Cuts your own outline – drawn or traced from a photo – into a solid insert.',
  category: 'inserts',
  status: 'available',
  icon: ICONS.outline,
  params: [
    { key: 'outline', label: 'Outline', type: 'outline' },
    { key: 'count', label: 'Copies', type: 'number', min: 1, max: 12, step: 1, help: 'Side-by-side pockets; each is scaled to fit its share of the width.' },
    { key: 'depth', label: 'Pocket depth', type: 'number', min: 2, max: 120, step: 0.5, unit: 'mm' },
    { key: 'margin', label: 'Edge margin', type: 'number', min: 0, max: 30, step: 0.5, unit: 'mm' },
    { key: 'tolerance', label: 'Clearance', type: 'number', min: 0, max: 3, step: 0.1, unit: 'mm', help: 'Grows the shape so the object drops in easily.' },
  ],
  defaults: { outline: '', count: 1, depth: 15, margin: 5, tolerance: 0.6 },
  supportsDividers: false,
  build: ({ kernel: k, bin, params }) => {
    const L = outlineLayout(bin, params);
    return withBase(k, { ...bin, dividers: [] }, (s, body) => {
      if (!L.ok) return body;
      let out = s.keep(body.add(pocketBlock(k, s, bin, L.top)));
      const sections = [];
      for (let i = 0; i < L.count; i++) {
        // Layout y grows towards the front; geometry y grows towards the back.
        const poly: [number, number][] = L.shape.map((p) => [geoX(bin, i * L.cellW + p.x), geoY(bin, p.y)]);
        if (polygonArea(L.shape) > 0) poly.reverse(); // keep contours counter-clockwise after the y flip
        sections.push(s.keep(k.CrossSection.ofPolygons([poly], 'NonZero')));
      }
      const cut = s.keep(s.keep(s.keep(k.CrossSection.union(sections)).extrude(L.top - L.z0 + 1)).translate([0, 0, L.z0]));
      out = s.keep(out.subtract(cut));
      return out;
    });
  },
  preview: (ctx) => {
    const L = outlineLayout(ctx.bin, ctx.params);
    const out: PreviewShape[] = [pocketPreview(ctx)];
    if (!L.ok) return out;
    for (let i = 0; i < L.count; i++) {
      out.push({
        kind: 'path',
        points: L.shape.map((p) => [layX(ctx, i * L.cellW + p.x), layY(ctx, p.y)] as [number, number]),
        tone: 'hole',
      });
    }
    return out;
  },
};
