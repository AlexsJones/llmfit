/**
 * Design store: owns the serialisable Design plus undo/redo history.
 * All layout mutations go through here so the 2D editor, properties panel,
 * keyboard shortcuts and 3D view stay consistent.
 */
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import {
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
import { MIN_COMPARTMENT, createEmptyDesign, newId, usableArea } from '../model/defaults';
import type {
  BuildVolume,
  Compartment,
  Design,
  Divider,
  DividerOrientation,
  DrawerConfig,
  PrintSettings,
  Rect,
} from '../model/types';
import { getModule } from '../modules/registry';

const HISTORY_LIMIT = 100;

export type StartLayout = { kind: 'empty' } | { kind: 'grid'; rows: number; cols: number };

interface DesignState {
  design: Design;
  past: Design[];
  future: Design[];
  hasProject: boolean;
  /** Id of the saved project this design belongs to, if any. */
  projectId: string | null;

  undo(): void;
  redo(): void;
  beginGesture(): void;
  endGesture(): void;

  newDesign(drawer: DrawerConfig, print: Partial<PrintSettings>, layout: StartLayout, name?: string): void;
  rename(name: string): void;
  updateDrawer(patch: Partial<DrawerConfig>): void;
  updatePrint(patch: Partial<PrintSettings>): void;
  updateBuildVolume(patch: Partial<BuildVolume>): void;

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
  /** Replace the whole design (import, or loading a saved project). */
  loadDesign(design: Design): void;
  duplicateCompartment(id: string): string | null;
  splitCompartment(id: string, cut: DividerOrientation, n: number): string[];
  splitToFit(id: string): string[];
  expandCompartment(id: string): void;
  fillGaps(): void;
  distribute(id: string, axis: 'x' | 'y'): void;
  setKind(ids: string[], kind: string): void;
  applyGrid(rows: number, cols: number): void;
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

function nextName(d: Design): string {
  const used = new Set(d.compartments.map((c) => c.name));
  for (let i = 1; ; i++) if (!used.has(`Compartment ${i}`)) return `Compartment ${i}`;
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
  return cs.map((c, i) => ({ ...c, name: `Compartment ${i + 1}` }));
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

        newDesign(drawer, print, layout, name) {
          const d = createEmptyDesign(drawer, print);
          if (name) d.name = name;
          if (layout.kind === 'grid') {
            const rects = gridRects(layout.rows, layout.cols, bounds(d));
            d.compartments = renameSequential(rects.map((r) => makeCompartment(d, r)));
          }
          set({ design: d, past: [], future: [], hasProject: true, projectId: null });
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
        updateBuildVolume(patch) {
          const d = get().design;
          commit({ ...d, buildVolume: { ...d.buildVolume, ...patch } });
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
          const r = { x: r2(rect.x), y: r2(rect.y), w: r2(rect.w), d: r2(rect.d) };
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
        mergeCompartments(ids) {
          const d = get().design;
          const picked = d.compartments.filter((c) => ids.includes(c.id));
          if (picked.length < 2) return null;
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
          set({ design, past: [], future: [], hasProject: true });
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
        splitCompartment(id, cut, n) {
          const d = get().design;
          const src = d.compartments.find((c) => c.id === id);
          if (!src) return [];
          const span = cut === 'vertical' ? src.w : src.d;
          if (span / n < MIN_COMPARTMENT) return [];
          const pieces = splitCompartment(src, cut, n);
          commit({ ...d, compartments: d.compartments.flatMap((c) => (c.id === id ? pieces : [c])) });
          return pieces.map((p) => p.id);
        },
        splitToFit(id) {
          const d = get().design;
          const src = d.compartments.find((c) => c.id === id);
          if (!src) return [];
          const pieces = splitToFit(src, d.buildVolume.x, d.buildVolume.y, d.print.fitGap);
          if (pieces.length < 2) return [];
          commit({ ...d, compartments: d.compartments.flatMap((c) => (c.id === id ? pieces : [c])) });
          return pieces.map((p) => p.id);
        },
        expandCompartment(id) {
          const d = get().design;
          const b = bounds(d);
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
          const order = [...cs].sort((a, c) => a.y - c.y || a.x - c.x).map((c) => c.id);
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
          mapCompartments((cs) => distributeRun(cs, id, axis));
        },
        applyGrid(rows, cols) {
          const d = get().design;
          const rects = gridRects(rows, cols, bounds(d));
          const base = { ...d, compartments: [] };
          commit({ ...d, compartments: renameSequential(rects.map((r) => makeCompartment(base, r))) });
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
      partialize: (s) => ({ design: s.design, hasProject: s.hasProject, projectId: s.projectId }),
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
