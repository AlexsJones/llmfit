import { Box, Columns2, Download, FilePlus2, Redo2, Square, Undo2 } from 'lucide-react';
import { useDesign } from '../store/designStore';
import { type ViewMode, useUi } from '../store/uiStore';
import { Button, IconButton, Segmented } from './ui/controls';

export function Logo() {
  return (
    <div className="flex items-center gap-2">
      <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden>
        <rect width="32" height="32" rx="8" fill="#18181b" />
        <rect x="7" y="7" width="8" height="18" rx="2" fill="#fff" />
        <rect x="17" y="7" width="8" height="8" rx="2" fill="#fff" opacity=".7" />
        <rect x="17" y="17" width="8" height="8" rx="2" fill="#fff" opacity=".45" />
      </svg>
      <span className="text-[14px] font-semibold tracking-tight text-ink">Organa</span>
    </div>
  );
}

export function TopBar() {
  const name = useDesign((s) => s.design.name);
  const canUndo = useDesign((s) => s.past.length > 0);
  const canRedo = useDesign((s) => s.future.length > 0);
  const view = useUi((s) => s.view);
  const actions = useDesign.getState();
  const ui = useUi.getState();

  return (
    <header className="flex h-13 shrink-0 items-center gap-3 border-b border-line bg-surface/80 px-3 backdrop-blur sm:px-4">
      <Logo />
      <div className="mx-1 hidden h-5 w-px bg-line sm:block" />
      <input
        value={name}
        onChange={(e) => useDesign.setState((s) => ({ design: { ...s.design, name: e.target.value } }))}
        onFocus={() => actions.beginGesture()}
        onBlur={() => actions.endGesture()}
        className="hidden min-w-0 max-w-64 flex-1 truncate rounded-md bg-transparent px-1.5 py-1 text-[13px] font-medium text-ink-2 outline-none hover:bg-subtle focus:bg-subtle sm:block"
        aria-label="Design name"
      />
      <div className="flex-1 sm:hidden" />
      <div className="flex items-center">
        <IconButton label="Undo (⌘Z)" disabled={!canUndo} onClick={actions.undo}>
          <Undo2 size={16} />
        </IconButton>
        <IconButton label="Redo (⇧⌘Z)" disabled={!canRedo} onClick={actions.redo}>
          <Redo2 size={16} />
        </IconButton>
      </div>
      <div className="hidden flex-1 justify-center md:flex">
        <ViewSwitch view={view} onChange={ui.setView} />
      </div>
      <div className="flex items-center gap-1.5">
        <Button variant="ghost" onClick={() => ui.setSetupOpen(true)} title="Start a new design">
          <FilePlus2 size={14} /> New
        </Button>
        <Button variant="primary" onClick={() => ui.setExportOpen(true)}>
          <Download size={14} /> <span className="hidden sm:inline">Export</span>
        </Button>
      </div>
    </header>
  );
}

export function ViewSwitch({ view, onChange, className }: { view: ViewMode; onChange: (v: ViewMode) => void; className?: string }) {
  return (
    <Segmented
      value={view}
      onChange={onChange}
      className={className}
      options={[
        { value: '2d', label: <><Square size={13} /> 2D</>, title: 'Top-down layout (1)' },
        { value: 'split', label: <><Columns2 size={13} /> Split</>, title: 'Side by side (2)' },
        { value: '3d', label: <><Box size={13} /> 3D</>, title: '3D preview (3)' },
      ]}
    />
  );
}
