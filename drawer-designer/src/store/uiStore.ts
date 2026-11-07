import { create } from 'zustand';

export type ViewMode = '2d' | '3d' | 'split';

interface UiState {
  /** Every selected compartment; `selectedId` is the primary (last clicked) one. */
  selectedIds: string[];
  selectedId: string | null;
  selectedDividerId: string | null;
  hoverId: string | null;
  view: ViewMode;
  gridSnap: number;
  edgeSnap: boolean;
  showDrawer: boolean;
  exportOpen: boolean;
  setupOpen: boolean;
  projectsOpen: boolean;
  plateSpacing: number;
  plateRotate: boolean;
  explode: number;
  mobilePanel: 'tools' | 'properties' | null;
  toast: { id: number; message: string; tone: 'info' | 'warn' } | null;

  select(id: string | null, dividerId?: string | null): void;
  /** Add or remove one compartment from the selection (Shift-click). */
  toggleSelect(id: string): void;
  selectMany(ids: string[]): void;
  setHover(id: string | null): void;
  setView(v: ViewMode): void;
  setGridSnap(v: number): void;
  setEdgeSnap(v: boolean): void;
  setShowDrawer(v: boolean): void;
  setExportOpen(v: boolean): void;
  setSetupOpen(v: boolean): void;
  setProjectsOpen(v: boolean): void;
  setPlateSpacing(v: number): void;
  setPlateRotate(v: boolean): void;
  setExplode(v: number): void;
  setMobilePanel(p: UiState['mobilePanel']): void;
  notify(message: string, tone?: 'info' | 'warn'): void;
}

let toastId = 0;

export const useUi = create<UiState>()((set) => ({
  selectedIds: [],
  selectedId: null,
  selectedDividerId: null,
  hoverId: null,
  view: 'split',
  gridSnap: 5,
  edgeSnap: true,
  showDrawer: true,
  exportOpen: false,
  setupOpen: false,
  projectsOpen: false,
  plateSpacing: 4,
  plateRotate: true,
  explode: 0,
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
  setEdgeSnap: (edgeSnap) => set({ edgeSnap }),
  setShowDrawer: (showDrawer) => set({ showDrawer }),
  setExportOpen: (exportOpen) => set({ exportOpen }),
  setSetupOpen: (setupOpen) => set({ setupOpen }),
  setProjectsOpen: (projectsOpen) => set({ projectsOpen }),
  setPlateSpacing: (plateSpacing) => set({ plateSpacing }),
  setPlateRotate: (plateRotate) => set({ plateRotate }),
  setExplode: (explode) => set({ explode }),
  setMobilePanel: (mobilePanel) => set({ mobilePanel }),
  notify: (message, tone = 'info') => {
    toastId += 1;
    const id = toastId;
    set({ toast: { id, message, tone } });
    setTimeout(() => set((s) => (s.toast?.id === id ? { toast: null } : s)), 2600);
  },
}));
