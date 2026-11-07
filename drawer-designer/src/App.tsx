import { AlertTriangle, Info, Loader2, SlidersHorizontal, Wrench } from 'lucide-react';
import { useEffect } from 'react';
import { Editor2D } from './components/Editor2D';
import { ExportDialog } from './components/ExportDialog';
import { LeftPanel } from './components/LeftPanel';
import { RightPanel } from './components/RightPanel';
import { ProjectsDialog } from './components/ProjectsDialog';
import { SetupDialog } from './components/SetupDialog';
import { TopBar, ViewSwitch } from './components/TopBar';
import { Viewport3D } from './components/Viewport3D';
import { cx } from './components/ui/controls';
import { GeometryProvider, useKernel } from './hooks/geometry';
import { rectOf, sweepMove } from './layout/ops';
import { usableArea } from './model/defaults';
import { useDesign } from './store/designStore';
import { useUi } from './store/uiStore';

function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest('input, textarea, select, [contenteditable=true]')) return;
      const mod = e.metaKey || e.ctrlKey;
      const d = useDesign.getState();
      const ui = useUi.getState();
      const sel = ui.selectedId;
      const multi = ui.selectedIds;
      if (mod && e.key.toLowerCase() === 's') {
        e.preventDefault();
        ui.setProjectsOpen(true);
      } else if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) d.redo();
        else d.undo();
      } else if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        d.redo();
      } else if (mod && e.key.toLowerCase() === 'd' && sel) {
        e.preventDefault();
        const id = d.duplicateCompartment(sel);
        if (id) ui.select(id);
      } else if (mod && e.key.toLowerCase() === 'a') {
        e.preventDefault();
        ui.selectMany(d.design.compartments.map((c) => c.id));
      } else if (mod && e.key.toLowerCase() === 'm' && multi.length > 1) {
        e.preventDefault();
        const id = d.mergeCompartments(multi);
        if (id) ui.select(id);
        else ui.notify('These compartments do not form a rectangle', 'warn');
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && sel) {
        e.preventDefault();
        if (ui.selectedDividerId) {
          d.removeDivider(sel, ui.selectedDividerId);
          ui.select(sel);
        } else {
          d.deleteMany(multi.length ? multi : [sel]);
          ui.select(null);
        }
      } else if (e.key === 'Escape') {
        ui.select(null);
      } else if (!mod && e.key.toLowerCase() === 'a') {
        const id = d.addCompartment();
        if (id) ui.select(id);
      } else if (!mod && ['1', '2', '3'].includes(e.key)) {
        ui.setView(e.key === '1' ? '2d' : e.key === '2' ? 'split' : '3d');
      } else if (sel && e.key.startsWith('Arrow')) {
        e.preventDefault();
        const step = e.shiftKey ? 10 : 1;
        const dx = e.key === 'ArrowLeft' ? -step : e.key === 'ArrowRight' ? step : 0;
        const dy = e.key === 'ArrowUp' ? -step : e.key === 'ArrowDown' ? step : 0;
        if (multi.length > 1) {
          d.moveMany(multi, dx, dy);
          return;
        }
        const c = d.design.compartments.find((x) => x.id === sel);
        if (!c) return;
        const u = usableArea(d.design.drawer);
        const others = d.design.compartments.filter((o) => o.id !== sel).map(rectOf);
        d.setCompartmentRect(sel, sweepMove(rectOf(c), c.x + dx, c.y + dy, others, { width: u.width, depth: u.depth }));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

function Workspace() {
  const view = useUi((s) => s.view);
  return (
    <div className="relative flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-line bg-surface shadow-panel">
      <div className={cx('grid min-h-0 flex-1', view === 'split' ? 'grid-rows-2 xl:grid-cols-2 xl:grid-rows-1' : 'grid-cols-1')}>
        {view !== '3d' && (
          <div className={cx('relative min-h-0 min-w-0', view === 'split' && 'border-b border-line xl:border-r xl:border-b-0')}>
            <PaneLabel>Layout · top view</PaneLabel>
            <Editor2D compact={view === 'split'} />
          </div>
        )}
        {view !== '2d' && (
          <div className="relative min-h-0 min-w-0">
            <Viewport3D />
          </div>
        )}
      </div>
    </div>
  );
}

function PaneLabel({ children }: { children: React.ReactNode }) {
  return <div className="pointer-events-none absolute top-3 left-3 z-10 hidden text-[11px] font-medium tracking-wide text-faint uppercase md:block">{children}</div>;
}

function Toast() {
  const toast = useUi((s) => s.toast);
  if (!toast) return null;
  return (
    <div key={toast.id} className="pointer-events-none fixed bottom-20 left-1/2 z-[60] -translate-x-1/2 animate-slide-up lg:bottom-6">
      <div className={cx('flex items-center gap-2 rounded-xl px-3.5 py-2 text-[12.5px] shadow-float', toast.tone === 'warn' ? 'bg-ink text-white' : 'bg-ink text-white')}>
        {toast.tone === 'warn' ? <AlertTriangle size={14} className="text-orange-300" /> : <Info size={14} className="text-blue-300" />}
        {toast.message}
      </div>
    </div>
  );
}

function MobilePanels() {
  const panel = useUi((s) => s.mobilePanel);
  const selectedId = useUi((s) => s.selectedId);
  const view = useUi((s) => s.view);
  const set = useUi.getState().setMobilePanel;
  return (
    <>
      <nav className="flex shrink-0 items-center justify-between gap-1 border-t border-line bg-surface px-3 pt-2 pb-[max(0.5rem,env(safe-area-inset-bottom))] lg:hidden">
        <button type="button" onClick={() => set(panel === 'tools' ? null : 'tools')} className={cx('flex flex-1 flex-col items-center gap-0.5 rounded-lg py-1 text-[11px]', panel === 'tools' ? 'text-accent' : 'text-muted')}>
          <Wrench size={18} /> Tools
        </button>
        <ViewSwitch view={view} onChange={useUi.getState().setView} className="md:hidden" />
        <button type="button" onClick={() => set(panel === 'properties' ? null : 'properties')} className={cx('flex flex-1 flex-col items-center gap-0.5 rounded-lg py-1 text-[11px]', panel === 'properties' ? 'text-accent' : 'text-muted')}>
          <SlidersHorizontal size={18} /> {selectedId ? 'Compartment' : 'Settings'}
        </button>
      </nav>
      {panel && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 animate-fade-in bg-ink/20" onClick={() => set(null)} />
          <div className="scroll-thin absolute inset-x-0 bottom-0 max-h-[72vh] animate-slide-up overflow-y-auto rounded-t-2xl bg-surface pb-[env(safe-area-inset-bottom)] shadow-pop">
            <div className="sticky top-0 z-10 flex justify-center bg-surface py-2">
              <div className="h-1 w-10 rounded-full bg-line-strong" />
            </div>
            {panel === 'tools' ? <LeftPanel /> : <RightPanel />}
          </div>
        </div>
      )}
    </>
  );
}

export default function App() {
  const { kernel, error } = useKernel();
  const hasProject = useDesign((s) => s.hasProject);
  const setupOpen = useUi((s) => s.setupOpen);
  useShortcuts();

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-center">
        <div>
          <div className="text-[15px] font-semibold">The geometry engine failed to load</div>
          <div className="mt-1 text-[13px] text-muted">{error}</div>
        </div>
      </div>
    );
  }
  if (!kernel) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex items-center gap-2 text-[13px] text-muted">
          <Loader2 size={16} className="animate-spin" /> Loading geometry engine…
        </div>
      </div>
    );
  }

  return (
    <GeometryProvider kernel={kernel}>
      <div className="flex h-full flex-col">
        <TopBar />
        <div className="flex min-h-0 flex-1">
          <aside className="scroll-thin hidden w-[272px] shrink-0 overflow-y-auto border-r border-line bg-surface lg:block">
            <LeftPanel />
          </aside>
          <main className="min-w-0 flex-1 p-2 sm:p-3">
            <Workspace />
          </main>
          <aside className="hidden w-[304px] shrink-0 border-l border-line bg-surface lg:block">
            <RightPanel />
          </aside>
        </div>
        <MobilePanels />
      </div>
      {(setupOpen || !hasProject) && <SetupDialog />}
      <ProjectsDialog />
      <ExportDialog />
      <Toast />
    </GeometryProvider>
  );
}
