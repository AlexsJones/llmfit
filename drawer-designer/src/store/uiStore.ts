import { create } from 'zustand';

export type ViewMode = '2d' | '3d' | 'split';

interface UiState {
  selectedId: string | null;
  selectedDividerId: string | null;
  hoverId: string | null;
  view: ViewMode;
  gridSnap: number;
  edgeSnap: boolean;
  showDrawer: boolean;
  exportOpen: boolean;
  setupOpen: boolean;
  mobilePanel: 'tools' | 'properties' | null;
  toast: { id: number; message: string; tone: 'info' | 'warn' } | null;

  select(id: string | null, dividerId?: string | null): void;
  setHover(id: string | null): void;
  setView(v: ViewMode): void;
  setGridSnap(v: number): void;
  setEdgeSnap(v: boolean): void;
  setShowDrawer(v: boolean): void;
  setExportOpen(v: boolean): void;
  setSetupOpen(v: boolean): void;
  setMobilePanel(p: UiState['mobilePanel']): void;
  notify(message: string, tone?: 'info' | 'warn'): void;
}

let toastId = 0;

export const useUi = create<UiState>()((set) => ({
  selectedId: null,
  selectedDividerId: null,
  hoverId: null,
  view: 'split',
  gridSnap: 5,
  edgeSnap: true,
  showDrawer: true,
  exportOpen: false,
  setupOpen: false,
  mobilePanel: null,
  toast: null,

  select: (id, dividerId = null) => set({ selectedId: id, selectedDividerId: dividerId }),
  setHover: (id) => set({ hoverId: id }),
  setView: (view) => set({ view }),
  setGridSnap: (gridSnap) => set({ gridSnap }),
  setEdgeSnap: (edgeSnap) => set({ edgeSnap }),
  setShowDrawer: (showDrawer) => set({ showDrawer }),
  setExportOpen: (exportOpen) => set({ exportOpen }),
  setSetupOpen: (setupOpen) => set({ setupOpen }),
  setMobilePanel: (mobilePanel) => set({ mobilePanel }),
  notify: (message, tone = 'info') => {
    toastId += 1;
    const id = toastId;
    set({ toast: { id, message, tone } });
    setTimeout(() => set((s) => (s.toast?.id === id ? { toast: null } : s)), 2600);
  },
}));
