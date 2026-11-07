import { Box, CircleHelp, Columns2, Download, FilePlus2, FolderOpen, Moon, Redo2, Square, Sun, SunMoon, Undo2 } from 'lucide-react';
import { useDesign, useUnsaved } from '../store/designStore';
import { type ViewMode, useUi } from '../store/uiStore';
import { Button, IconButton, Segmented, cx } from './ui/controls';

/** Three bins seen from above; the small one carries the accent. */
export function Logo({ onDark }: { onDark?: boolean } = {}) {
  const bar = onDark ? '#ffffff' : '#18181b';
  return (
    <div className="flex items-center gap-2">
      <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden>
        {!onDark && <rect width="32" height="32" rx="8" fill="#18181b" />}
        <rect x={onDark ? 3 : 7} y={onDark ? 4 : 7} width="8" height="18" rx="2" fill={onDark ? bar : '#fff'} />
        <rect x={onDark ? 13 : 17} y={onDark ? 4 : 7} width="8" height="8" rx="2" fill={onDark ? bar : '#fff'} opacity=".55" />
        <rect x={onDark ? 13 : 17} y={onDark ? 14 : 17} width="8" height="8" rx="2" fill="#2fbd8a" />
      </svg>
      <span className={cx('text-[14px] font-semibold tracking-tight', !onDark && 'text-ink')}>Organa</span>
    </div>
  );
}

export function TopBar() {
  const name = useDesign((s) => s.design.name);
  const canUndo = useDesign((s) => s.past.length > 0);
  const canRedo = useDesign((s) => s.future.length > 0);
  const unsaved = useUnsaved();
  const view = useUi((s) => s.view);
  const themeSetting = useUi((s) => s.themeSetting);
  const actions = useDesign.getState();
  const ui = useUi.getState();

  return (
    <header className="topbar flex h-13 shrink-0 items-center gap-3 border-b border-black/60 bg-[#18181b] px-3 text-white sm:px-4">
      <Logo onDark />
      <div className="mx-1 hidden h-5 w-px bg-white/15 sm:block" />
      <input
        value={name}
        onChange={(e) => useDesign.setState((s) => ({ design: { ...s.design, name: e.target.value } }))}
        onFocus={() => actions.beginGesture()}
        onBlur={() => actions.endGesture()}
        className="hidden min-w-0 max-w-64 flex-1 truncate rounded-md bg-transparent px-1.5 py-1 text-[13px] font-medium text-white/75 outline-none transition-colors placeholder:text-white/35 hover:bg-white/10 focus:bg-white/10 focus:text-white sm:block"
        aria-label="Design name"
      />
      <div className="flex-1 sm:hidden" />
      <div className="flex items-center">
        <IconButton onDark label="Undo (⌘Z)" disabled={!canUndo} onClick={actions.undo}>
          <Undo2 size={16} />
        </IconButton>
        <IconButton onDark label="Redo (⇧⌘Z)" disabled={!canRedo} onClick={actions.redo}>
          <Redo2 size={16} />
        </IconButton>
      </div>
      <div className="hidden flex-1 justify-center md:flex">
        <ViewSwitch onDark view={view} onChange={ui.setView} />
      </div>
      <div className="flex items-center gap-1.5">
        <Button variant="onDark" onClick={() => ui.setSetupOpen(true)} title="Start a new design">
          <FilePlus2 size={14} /> <span className="hidden sm:inline">New</span>
        </Button>
        <Button
          variant="onDark"
          onClick={() => ui.setProjectsOpen(true)}
          title={unsaved ? 'Unsaved changes – saved projects, import and export (⌘S)' : 'Saved projects, import and export (⌘S)'}
        >
          <FolderOpen size={14} /> <span className="hidden sm:inline">Projects</span>
          {unsaved && <span className="ml-0.5 size-1.5 rounded-full bg-accent" aria-label="Unsaved changes" />}
        </Button>
        <IconButton
          onDark
          label={themeSetting === 'dark' ? 'Switch to light' : themeSetting === 'light' ? 'Follow the system theme' : 'Switch to dark'}
          onClick={() => ui.setTheme(themeSetting === 'dark' ? 'light' : themeSetting === 'light' ? 'system' : 'dark')}
          className="hidden sm:inline-flex"
        >
          {themeSetting === 'dark' ? <Moon size={16} /> : themeSetting === 'light' ? <Sun size={16} /> : <SunMoon size={16} />}
        </IconButton>
        <IconButton onDark label="Keyboard shortcuts (?)" onClick={() => ui.setShortcutsOpen(true)} className="hidden sm:inline-flex">
          <CircleHelp size={16} />
        </IconButton>
        <Button variant="accent" onClick={() => ui.setExportOpen(true)}>
          <Download size={14} /> <span className="hidden sm:inline">Export</span>
        </Button>
      </div>
    </header>
  );
}

export function ViewSwitch({
  view,
  onChange,
  className,
  onDark,
}: {
  view: ViewMode;
  onChange: (v: ViewMode) => void;
  className?: string;
  onDark?: boolean;
}) {
  return (
    <Segmented
      value={view}
      onChange={onChange}
      onDark={onDark}
      className={className}
      options={[
        { value: '2d', label: <><Square size={13} /> 2D</>, title: 'Top-down layout (1)' },
        { value: 'split', label: <><Columns2 size={13} /> Split</>, title: 'Side by side (2)' },
        { value: '3d', label: <><Box size={13} /> 3D</>, title: '3D preview (3)' },
      ]}
    />
  );
}
