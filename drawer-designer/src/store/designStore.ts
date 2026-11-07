/**
 * Design store: owns the serialisable Design plus undo/redo history.
 * All layout mutations go through here so the 2D editor, properties panel,
 * keyboard shortcuts and 3D view stay consistent.
 */
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import {
  boundsOf,
  distributeRun,
  evenDividers,
  expandToFill,
  findFreeSpot,
  fitToBounds,
  gridRects,
  mergeBox,
  moveGroup,
  overlaps,
  inBounds,
  rectOf,
  splitCompartment,
  splitToFit,
} from '../layout/ops';
import { MIN_COMPARTMENT, createEmptyDesign, newId, printerFromPreset, usableArea, wallsForNozzle } from '../model/defaults';
import { matchesPreset } from '../model/printers';
import type {
  Compartment,
  Design,
  Divider,
  DividerOrientation,
  DrawerConfig,
  PrinterConfig,
  PrintSettings,
  Rect,
} from '../model/types';
import { PITCH } from '../geometry/gridfinity';
import { getModule } from '../modules/registry';
import { tr } from './uiStore';

const HISTORY_LIMIT = 100;

export type StartLayout = { kind: 'empty' } | { kind: 'grid'; rows: number; cols: number };

interface DesignState {
  design: Design;
  past: Design[];
  future: Design[];
  hasProject: boolean;
  /** Id of the saved project this design belongs to, if any. */
  projectId: string | null;
  /**
   * Counts every change to the design. A subscriber below bumps it, so edits made
   * by calling `setState` directly are counted too.
   */
  revision: number;
  /** The revision that was last written to a project or a file. */
  savedRevision: number;
  /** Mark the current design as saved, so nothing warns about losing it. */
  markSaved(): void;

  undo(): void;
  redo(): void;
  beginGesture(): void;
  endGesture(): void;

  newDesign(drawer: DrawerConfig, print: Partial<PrintSettings>, layout: StartLayout, name?: string, printerId?: string): void;
  rename(name: string): void;
  updateDrawer(patch: Partial<DrawerConfig>): void;
  updatePrint(patch: Partial<PrintSettings>): void;
  updatePrinter(patch: Partial<PrinterConfig>): void;
  /** Apply a printer preset (or switch to a custom machine). */
  selectPrinter(presetId: string): void;

  addCompartment(kind?: string): string | null;
  /** Add a compartment at an explicit rectangle (validated). */
  addCompartmentRect(rect: Rect, kind?: string): string | null;
  updateCompartment(id: string, patch: Partial<Compartment>, transient?: boolean): void;
  /** Apply a rectangle if it stays in bounds and does not overlap. Returns success. */
  setCompartmentRect(id: string, rect: Rect, transient?: boolean): boolean;
  deleteCompartment(id: string): void;
  deleteMany(ids: string[]): void;
  /** Merge compartments whose union is a rectangle into one. Returns the new id. */
  mergeCompartments(ids: string[]): string | null;
  /** Move several compartments together; the group is clamped and never overlaps others. */
  moveMany(ids: string[], dx: number, dy: number, transient?: boolean): void;
  /** Turn a compartment about its centre (degrees); 0 puts it back on the grid. */
  setRotation(id: string, deg: number, transient?: boolean): void;
  /** Replace the whole design (import, or loading a saved project). */
  loadDesign(design: Design): void;
  duplicateCompartment(id: string): string | null;
  /** Paste compartments, offset into free space. Returns the new ids. */
  pasteCompartments(items: Compartment[]): string[];
  /** Insert a row or column of cells next to the given compartment. */
  insertLane(id: string, axis: 'x' | 'y', before: boolean): string[];
  splitCompartment(id: string, cut: DividerOrientation, n: number): string[];
  splitToFit(id: string): string[];
  expandCompartment(id: string): void;
  fillGaps(): void;
  distribute(id: string, axis: 'x' | 'y'): void;
  setKind(ids: string[], kind: string): void;
  applyGrid(rows: number, cols: number): void;
  /** Fill the usable area with whole Gridfinity cells, each holding a grid bin. */
  applyGridfinity(cols: number, rows: number): void;
  clearLayout(): void;

  addDivider(id: string, orientation: DividerOrientation): string | null;
  updateDivider(id: string, dividerId: string, position: number, transient?: boolean): void;
  removeDivider(id: string, dividerId: string): void;
  setEvenDividers(id: string, orientation: DividerOrientation, count: number): void;
}

function bounds(d: Design) {
  const u = usableArea(d.drawer);
  return { width: u.width, depth: u.depth };
}

/** Default compartment name, in the language the interface is showing. */
function numbered(i: number) {
  return tr('Compartment {n}', { n: i });
}

function nextName(d: Design): string {
  const used = new Set(d.compartments.map((c) => c.name));
  for (let i = 1; ; i++) if (!used.has(numbered(i))) return numbered(i);
}

function makeCompartment(d: Design, rect: Rect, kind = 'bin'): Compartment {
  return {
    id: newId('cp'),
    kind,
    name: nextName(d),
    ...rect,
    height: null,
    dividers: [],
    params: { ...getModule(kind).defaults },
  };
}

function renameSequential(cs: Compartment[]): Compartment[] {
  return cs.map((c, i) => ({ ...c, name: numbered(i + 1) }));
}

const r2 = (v: number) => Math.round(v * 100) / 100;

export const useDesign = create<DesignState>()(
  persist(
    (set, get) => {
      const commit = (next: Design, transient = false) => {
        const s = get();
        if (transient) set({ design: next });
        else set({ design: next, past: [...s.past, s.design].slice(-HISTORY_LIMIT), future: [] });
      };
      const mapCompartments = (fn: (cs: Compartment[]) => Compartment[], transient = false) => {
        const d = get().design;
        commit({ ...d, compartments: fn(d.compartments) }, transient);
      };
      const mapOne = (id: string, fn: (c: Compartment) => Compartment, transient = false) =>
        mapCompartments((cs) => cs.map((c) => (c.id === id ? fn(c) : c)), transient);

      return {
        design: createEmptyDesign(),
        past: [],
        future: [],
        hasProject: false,
        projectId: null,
        revision: 0,
        savedRevision: 0,

        markSaved() {
          const r = get().revision;
          set({ savedRevision: r });
        },

        undo() {
          const { past, design, future } = get();
          if (!past.length) return;
          set({ design: past[past.length - 1], past: past.slice(0, -1), future: [design, ...future].slice(0, HISTORY_LIMIT) });
        },
        redo() {
          const { past, design, future } = get();
          if (!future.length) return;
          set({ design: future[0], past: [...past, design], future: future.slice(1) });
        },
        beginGesture() {
          const s = get();
          set({ past: [...s.past, s.design].slice(-HISTORY_LIMIT), future: [] });
        },
        endGesture() {
          const s = get();
          if (s.past.length && s.past[s.past.length - 1] === s.design) set({ past: s.past.slice(0, -1) });
        },

        newDesign(drawer, print, layout, name, printerId) {
          const d = createEmptyDesign(drawer, print, printerId);
          if (name) d.name = name;
          if (layout.kind === 'grid') {
            const rects = gridRects(layout.rows, layout.cols, bounds(d));
            d.compartments = renameSequential(rects.map((r) => makeCompartment(d, r)));
          }
          const r = get().revision + 1;
          set({ design: d, past: [], future: [], hasProject: true, projectId: null, revision: r, savedRevision: r });
        },
        rename(name) {
          commit({ ...get().design, name });
        },
        updateDrawer(patch) {
          const d = get().design;
          const drawer = { ...d.drawer, ...patch };
          const next = { ...d, drawer };
          next.compartments = fitToBounds(d.compartments, bounds(next));
          commit(next);
        },
        updatePrint(patch) {
          const d = get().design;
          commit({ ...d, print: { ...d.print, ...patch } });
        },
        updatePrinter(patch) {
          const d = get().design;
          const printer = { ...d.printer, ...patch };
          // Any hand edit of a machine value means it is no longer that preset.
          const touchesHardware = patch.build !== undefined || patch.nozzle !== undefined;
          if (touchesHardware && !matchesPreset(printer.presetId, printer.build, printer.nozzle)) {
            printer.presetId = 'custom';
            printer.label = 'Custom printer';
          }
          commit({ ...d, printer });
        },
        selectPrinter(presetId) {
          const d = get().design;
          const next = printerFromPreset(presetId);
          // Keep the user's own tuning; only the machine's hardware facts change.
          const printer = { ...next, layerHeight: d.printer.layerHeight, infill: d.printer.infill, startupMinutes: d.printer.startupMinutes };
          // A different nozzle changes what counts as a clean wall; retune it only when the
          // current value was itself the default for the old nozzle.
          const print =
            printer.nozzle !== d.printer.nozzle && Math.abs(d.print.wallThickness - wallsForNozzle(d.printer.nozzle)) < 1e-6
              ? { ...d.print, wallThickness: wallsForNozzle(printer.nozzle) }
              : d.print;
          commit({ ...d, printer, print });
        },

        addCompartment(kind = 'bin') {
          const d = get().design;
          const b = bounds(d);
          const target = Math.max(MIN_COMPARTMENT, Math.min(80, b.width / 3, b.depth / 3));
          const spot = findFreeSpot(target, target, d.compartments, b);
          if (!spot) return null;
          const c = makeCompartment(d, { x: r2(spot.x), y: r2(spot.y), w: r2(spot.w), d: r2(spot.d) }, kind);
          commit({ ...d, compartments: [...d.compartments, c] });
          return c.id;
        },
        addCompartmentRect(rect, kind = 'bin') {
          const d = get().design;
          const r = { x: r2(rect.x), y: r2(rect.y), w: r2(rect.w), d: r2(rect.d) };
          if (r.w < MIN_COMPARTMENT - 1e-6 || r.d < MIN_COMPARTMENT - 1e-6) return null;
          if (!inBounds(r, bounds(d)) || d.compartments.some((o) => overlaps(r, o))) return null;
          const c = makeCompartment(d, r, kind);
          commit({ ...d, compartments: [...d.compartments, c] });
          return c.id;
        },
        updateCompartment(id, patch, transient) {
          mapOne(id, (c) => ({ ...c, ...patch }), transient);
        },
        setCompartmentRect(id, rect, transient) {
          const d = get().design;
          // A turned compartment keeps its angle through a move, and the checks
          // below have to see it or they would test the wrong footprint.
          const rot = d.compartments.find((c) => c.id === id)?.rot;
          const r = { x: r2(rect.x), y: r2(rect.y), w: r2(rect.w), d: r2(rect.d), ...(rot ? { rot } : {}) };
          if (r.w < MIN_COMPARTMENT - 1e-6 || r.d < MIN_COMPARTMENT - 1e-6) return false;
          if (!inBounds(r, bounds(d))) return false;
          if (d.compartments.some((o) => o.id !== id && overlaps(r, o))) return false;
          mapOne(
            id,
            (c) => ({
              ...c,
              ...r,
              // keep dividers inside the compartment
              dividers: c.dividers.filter((dv) => dv.position < (dv.orientation === 'vertical' ? r.w : r.d)),
            }),
            transient,
          );
          return true;
        },
        deleteCompartment(id) {
          mapCompartments((cs) => cs.filter((c) => c.id !== id));
        },
        deleteMany(ids) {
          const set = new Set(ids);
          mapCompartments((cs) => cs.filter((c) => !set.has(c.id)));
        },
        setKind(ids, kind) {
          const set = new Set(ids);
          mapCompartments((cs) => cs.map((c) => (set.has(c.id) ? { ...c, kind, params: { ...getModule(kind).defaults } } : c)));
        },
        setRotation(id, deg, transient) {
          const d = get().design;
          const c = d.compartments.find((x) => x.id === id);
          if (!c) return;
          // Keep it in 0..359 and drop the field entirely when it is back on the grid.
          const rot = Math.round((((deg % 360) + 360) % 360) * 10) / 10;
          const b = bounds(d);
          const turned = { ...c, rot: rot || undefined } as Compartment;
          // Turning about the centre pushes the corners outwards, so slide the
          // compartment back inside before giving up on the angle.
          const box = boundsOf(rectOf(turned));
          const next = {
            ...turned,
            x: r2(turned.x + Math.min(Math.max(-box.x, 0), Math.max(0, b.width - box.w - box.x))),
            y: r2(turned.y + Math.min(Math.max(-box.y, 0), Math.max(0, b.depth - box.d - box.y))),
          };
          const others = d.compartments.filter((o) => o.id !== id).map(rectOf);
          if (!inBounds(next, b) || others.some((o) => overlaps(rectOf(next), o))) return;
          mapOne(id, () => next, transient);
        },
        mergeCompartments(ids) {
          const d = get().design;
          const picked = d.compartments.filter((c) => ids.includes(c.id));
          if (picked.length < 2) return null;
          // Merging, splitting and the lane operations are grid ideas; a turned
          // compartment has no row or column to belong to.
          if (picked.some((c) => c.rot)) return null;
          const box = mergeBox(picked.map(rectOf), d.compartments.filter((c) => !ids.includes(c.id)).map(rectOf));
          if (!box) return null;
          const first = picked[0];
          const merged: Compartment = {
            ...first,
            id: newId('cp'),
            x: r2(box.x),
            y: r2(box.y),
            w: r2(box.w),
            d: r2(box.d),
            // Dividers are positioned relative to the old, smaller footprint, so they are dropped.
            dividers: [],
            params: { ...first.params },
          };
          const rest = d.compartments.filter((c) => !ids.includes(c.id));
          const at = d.compartments.findIndex((c) => c.id === first.id);
          const keepBefore = rest.filter((c) => d.compartments.indexOf(c) < at);
          const keepAfter = rest.filter((c) => d.compartments.indexOf(c) > at);
          commit({ ...d, compartments: [...keepBefore, merged, ...keepAfter] });
          return merged.id;
        },
        moveMany(ids, dx, dy, transient) {
          const d = get().design;
          const group = d.compartments.filter((c) => ids.includes(c.id));
          if (!group.length) return;
          const others = d.compartments.filter((c) => !ids.includes(c.id)).map(rectOf);
          const mv = moveGroup(group.map(rectOf), dx, dy, others, bounds(d));
          if (!mv.dx && !mv.dy) return;
          const set = new Set(ids);
          mapCompartments(
            (cs) => cs.map((c) => (set.has(c.id) ? { ...c, x: r2(c.x + mv.dx), y: r2(c.y + mv.dy) } : c)),
            transient,
          );
        },
        loadDesign(design) {
          const r = get().revision + 1;
          set({ design, past: [], future: [], hasProject: true, revision: r, savedRevision: r });
        },
        duplicateCompartment(id) {
          const d = get().design;
          const src = d.compartments.find((c) => c.id === id);
          if (!src) return null;
          const spot = findFreeSpot(src.w, src.d, d.compartments, bounds(d));
          if (!spot || spot.w < src.w - 0.01 || spot.d < src.d - 0.01) return null;
          const copy: Compartment = {
            ...src,
            id: newId('cp'),
            name: `${src.name} copy`,
            x: r2(spot.x),
            y: r2(spot.y),
            dividers: src.dividers.map((dv) => ({ ...dv, id: newId('dv') })),
            params: { ...src.params },
          };
          commit({ ...d, compartments: [...d.compartments, copy] });
          return copy.id;
        },
        pasteCompartments(items) {
          const d = get().design;
          const b = bounds(d);
          const placed: Compartment[] = [];
          const taken = [...d.compartments.map(rectOf)];
          // Keep the group's shape: find one offset that works for all of them.
          let shift: { dx: number; dy: number } | null = null;
          for (const step of [0, 1, 2, 3, 4, 5, 6, 7, 8]) {
            const dx = step * 10;
            const dy = step * 10;
            const ok = items.every((c) => {
              const r = { x: c.x + dx, y: c.y + dy, w: c.w, d: c.d };
              return inBounds(r, b) && !taken.some((o) => overlaps(r, o));
            });
            if (ok) {
              shift = { dx, dy };
              break;
            }
          }
          if (!shift) {
            // Fall back to packing them one by one wherever they fit.
            for (const c of items) {
              const spot = findFreeSpot(c.w, c.d, taken, b);
              if (!spot || spot.w < c.w - 0.01 || spot.d < c.d - 0.01) continue;
              const copy = { ...c, id: newId('cp'), x: r2(spot.x), y: r2(spot.y), dividers: c.dividers.map((dv) => ({ ...dv, id: newId('dv') })), params: { ...c.params } };
              placed.push(copy);
              taken.push(rectOf(copy));
            }
          } else {
            for (const c of items) {
              placed.push({
                ...c,
                id: newId('cp'),
                x: r2(c.x + shift.dx),
                y: r2(c.y + shift.dy),
                dividers: c.dividers.map((dv) => ({ ...dv, id: newId('dv') })),
                params: { ...c.params },
              });
            }
          }
          if (!placed.length) return [];
          commit({ ...d, compartments: [...d.compartments, ...placed] });
          return placed.map((c) => c.id);
        },
        insertLane(id, axis, before) {
          const d = get().design;
          const b = bounds(d);
          const src = d.compartments.find((c) => c.id === id);
          if (!src || src.rot) return [];
          const size = axis === 'x' ? src.w : src.d;
          const at = axis === 'x' ? src.x : src.y;
          const cut = before ? at : at + size;
          // Everything from the cut onwards moves over to make room.
          const moved = d.compartments.map((c) => {
            const pos = axis === 'x' ? c.x : c.y;
            if (pos < cut - 0.01) return c;
            return axis === 'x' ? { ...c, x: r2(c.x + size) } : { ...c, y: r2(c.y + size) };
          });
          if (moved.some((c) => !inBounds(rectOf(c), b))) return [];
          // The new lane mirrors the source compartment's cross-section.
          const fresh = makeCompartment(
            { ...d, compartments: moved },
            axis === 'x' ? { x: r2(cut), y: src.y, w: size, d: src.d } : { x: src.x, y: r2(cut), w: src.w, d: size },
            src.kind,
          );
          commit({ ...d, compartments: [...moved, fresh] });
          return [fresh.id];
        },
        splitCompartment(id, cut, n) {
          const d = get().design;
          const src = d.compartments.find((c) => c.id === id);
          if (!src || src.rot) return [];
          const span = cut === 'vertical' ? src.w : src.d;
          if (span / n < MIN_COMPARTMENT) return [];
          const pieces = splitCompartment(src, cut, n);
          commit({ ...d, compartments: d.compartments.flatMap((c) => (c.id === id ? pieces : [c])) });
          return pieces.map((p) => p.id);
        },
        splitToFit(id) {
          const d = get().design;
          const src = d.compartments.find((c) => c.id === id);
          if (!src || src.rot) return [];
          const pieces = splitToFit(src, d.printer.build.x, d.printer.build.y, d.print.fitGap);
          if (pieces.length < 2) return [];
          commit({ ...d, compartments: d.compartments.flatMap((c) => (c.id === id ? pieces : [c])) });
          return pieces.map((p) => p.id);
        },
        expandCompartment(id) {
          const d = get().design;
          const b = bounds(d);
          if (d.compartments.find((c) => c.id === id)?.rot) return;
          mapOne(id, (c) => {
            const others = d.compartments.filter((o) => o.id !== id).map(rectOf);
            const r = expandToFill(rectOf(c), others, b);
            return { ...c, x: r2(r.x), y: r2(r.y), w: r2(r.w), d: r2(r.d) };
          });
        },
        fillGaps() {
          const d = get().design;
          const b = bounds(d);
          let cs = d.compartments;
          // Expand in reading order so the result is deterministic.
          const order = [...cs]
            .filter((c) => !c.rot)
            .sort((a, c) => a.y - c.y || a.x - c.x)
            .map((c) => c.id);
          for (const id of order) {
            cs = cs.map((c) => {
              if (c.id !== id) return c;
              const r = expandToFill(rectOf(c), cs.filter((o) => o.id !== id).map(rectOf), b);
              return { ...c, x: r2(r.x), y: r2(r.y), w: r2(r.w), d: r2(r.d) };
            });
          }
          commit({ ...d, compartments: cs });
        },
        distribute(id, axis) {
          if (get().design.compartments.find((c) => c.id === id)?.rot) return;
          mapCompartments((cs) => distributeRun(cs, id, axis));
        },
        applyGrid(rows, cols) {
          const d = get().design;
          const rects = gridRects(rows, cols, bounds(d));
          const base = { ...d, compartments: [] };
          commit({ ...d, compartments: renameSequential(rects.map((r) => makeCompartment(base, r))) });
        },
        applyGridfinity(cols, rows) {
          const d = get().design;
          const b = bounds(d);
          // Centre the lattice in whatever space is left over.
          const ox = r2((b.width - cols * PITCH) / 2);
          const oy = r2((b.depth - rows * PITCH) / 2);
          const cs: Compartment[] = [];
          for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
              const base = { ...d, compartments: cs };
              cs.push(
                makeCompartment(base, { x: r2(ox + col * PITCH), y: r2(oy + row * PITCH), w: PITCH, d: PITCH }, 'gridfinity-bin'),
              );
            }
          }
          commit({ ...d, compartments: renameSequential(cs) });
        },
        clearLayout() {
          commit({ ...get().design, compartments: [] });
        },

        addDivider(id, orientation) {
          const d = get().design;
          const c = d.compartments.find((x) => x.id === id);
          if (!c || !getModule(c.kind).supportsDividers) return null;
          const span = orientation === 'vertical' ? c.w : c.d;
          // Place the new divider in the middle of the largest free gap.
          const stops = [0, ...c.dividers.filter((dv) => dv.orientation === orientation).map((dv) => dv.position), span].sort((a, b) => a - b);
          let best = 0;
          let at = span / 2;
          for (let i = 0; i < stops.length - 1; i++) {
            if (stops[i + 1] - stops[i] > best) {
              best = stops[i + 1] - stops[i];
              at = (stops[i] + stops[i + 1]) / 2;
            }
          }
          const dv: Divider = { id: newId('dv'), orientation, position: r2(at) };
          mapOne(id, (x) => ({ ...x, dividers: [...x.dividers, dv] }));
          return dv.id;
        },
        updateDivider(id, dividerId, position, transient) {
          mapOne(
            id,
            (c) => ({
              ...c,
              dividers: c.dividers.map((dv) => {
                if (dv.id !== dividerId) return dv;
                const span = dv.orientation === 'vertical' ? c.w : c.d;
                const min = d0(get().design);
                return { ...dv, position: r2(Math.min(Math.max(position, min), span - min)) };
              }),
            }),
            transient,
          );
        },
        removeDivider(id, dividerId) {
          mapOne(id, (c) => ({ ...c, dividers: c.dividers.filter((dv) => dv.id !== dividerId) }));
        },
        setEvenDividers(id, orientation, count) {
          mapOne(id, (c) => ({ ...c, dividers: evenDividers(c, orientation, count) }));
        },
      };
    },
    {
      name: 'organa-design-v1',
      storage: createJSONStorage(() => {
        try {
          return window.localStorage;
        } catch {
          return memoryStorage;
        }
      }),
      partialize: (s) => ({
        design: s.design,
        hasProject: s.hasProject,
        projectId: s.projectId,
        revision: s.revision,
        savedRevision: s.savedRevision,
      }),
      version: 1,
    },
  ),
);

/** Minimum distance between a divider and the compartment's outer edge. */
function d0(d: Design) {
  return d.print.wallThickness + d.print.fitGap / 2 + d.print.dividerThickness / 2 + 1;
}

const mem = new Map<string, string>();
const memoryStorage: Storage = {
  get length() {
    return mem.size;
  },
  clear: () => mem.clear(),
  getItem: (k) => mem.get(k) ?? null,
  key: (i) => [...mem.keys()][i] ?? null,
  removeItem: (k) => void mem.delete(k),
  setItem: (k, v) => void mem.set(k, v),
};

export const useDesignActions = () => useDesign.getState();

/**
 * Count every replacement of the design object. Mutations that deliberately reset
 * the saved mark bump `revision` themselves, and are skipped here.
 */
useDesign.subscribe((s, prev) => {
  if (s.design !== prev.design && s.revision === prev.revision) useDesign.setState({ revision: s.revision + 1 });
});

/** True when the design has changed since it was last saved or exported. */
export const useUnsaved = () => useDesign((s) => s.revision !== s.savedRevision);
