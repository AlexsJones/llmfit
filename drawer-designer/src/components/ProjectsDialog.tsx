import { Clock, Database, Download, FolderOpen, HardDriveDownload, Save, ShieldCheck, Trash2, Upload, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useDesign, useUnsaved } from '../store/designStore';
import {
  type ProjectMeta,
  type StorageReport,
  backupAll,
  deleteProject,
  downloadDesign,
  isBackupBundle,
  listProjects,
  loadProject,
  parseDesign,
  restoreBundle,
  saveProject,
  storageReport,
} from '../store/projects';
import { useUi } from '../store/uiStore';
import { safeFileName } from '../geometry/stl';
import { Button, IconButton, Modal, cx } from './ui/controls';

function kb(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} kB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function ago(t: number) {
  const m = Math.round((Date.now() - t) / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  return new Date(t).toLocaleDateString();
}

export function ProjectsDialog() {
  const open = useUi((s) => s.projectsOpen);
  return (
    <Modal open={open} onClose={() => useUi.getState().setProjectsOpen(false)} width="max-w-xl">
      {open && <ProjectsContent />}
    </Modal>
  );
}

function ProjectsContent() {
  const design = useDesign((s) => s.design);
  const currentId = useDesign((s) => s.projectId);
  const unsaved = useUnsaved();
  const [items, setItems] = useState<ProjectMeta[]>([]);
  const [storage, setStorage] = useState<StorageReport | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const ui = useUi.getState();
  const close = () => ui.setProjectsOpen(false);
  const refresh = () => {
    setItems(listProjects());
    setStorage(storageReport());
  };

  useEffect(refresh, []);

  const save = (asNew: boolean) => {
    const id = saveProject(design, asNew ? undefined : (currentId ?? undefined));
    if (!id) {
      ui.notify('Could not save – browser storage is full', 'warn');
      return;
    }
    useDesign.setState({ projectId: id });
    useDesign.getState().markSaved();
    refresh();
    ui.notify(`Saved “${design.name}”`);
  };

  const backup = () => {
    const count = backupAll();
    if (!count) {
      ui.notify('There is nothing saved to back up yet', 'warn');
      return;
    }
    refresh();
    ui.notify(`Backed up ${count} project${count === 1 ? '' : 's'} to a file`);
  };

  const exportFile = () => {
    downloadDesign(design, safeFileName(design.name));
    useDesign.getState().markSaved();
  };

  const open = (id: string) => {
    const d = loadProject(id);
    if (!d) {
      ui.notify('That project could not be read', 'warn');
      return;
    }
    useDesign.getState().loadDesign(d);
    useDesign.setState({ projectId: id });
    ui.select(null);
    close();
  };

  const importFile = async (file: File) => {
    try {
      const text = await file.text();
      const data: unknown = JSON.parse(text);
      if (isBackupBundle(data)) {
        const result = restoreBundle(data);
        if (!result) {
          ui.notify('That backup could not be restored – browser storage may be full', 'warn');
          return;
        }
        refresh();
        ui.notify(
          `Restored ${result.added} project${result.added === 1 ? '' : 's'}` + (result.skipped ? ` · ${result.skipped} skipped` : ''),
        );
        return;
      }
      const parsed = parseDesign(data);
      if (!parsed) {
        ui.notify('That file is not an Organa design', 'warn');
        return;
      }
      useDesign.getState().loadDesign(parsed);
      useDesign.setState({ projectId: null });
      ui.select(null);
      close();
      ui.notify(`Imported “${parsed.name}” · ${parsed.compartments.length} compartments`);
    } catch {
      ui.notify('That file could not be read', 'warn');
    }
  };

  return (
    <div className="flex max-h-[88vh] flex-col">
      <div className="flex items-start gap-3 border-b border-line px-6 py-5">
        <div className="flex-1">
          <h2 className="text-[17px] font-semibold tracking-tight text-ink">Projects</h2>
          <p className="mt-0.5 text-[12.5px] text-muted">Saved in this browser. Export a file to keep or share a design.</p>
        </div>
        <IconButton label="Close" onClick={close}>
          <X size={16} />
        </IconButton>
      </div>

      <div className="flex flex-wrap gap-2 border-b border-line bg-canvas/60 px-6 py-3">
        <Button variant="primary" onClick={() => save(false)}>
          <Save size={14} /> {currentId ? 'Save' : 'Save project'}
        </Button>
        {currentId && (
          <Button onClick={() => save(true)}>
            <HardDriveDownload size={14} /> Save as copy
          </Button>
        )}
        <div className="flex-1" />
        <Button onClick={exportFile}>
          <Download size={14} /> Export file
        </Button>
        <Button onClick={() => fileRef.current?.click()} title="Open a design file or restore a backup">
          <Upload size={14} /> Import
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = '';
            if (f) void importFile(f);
          }}
        />
      </div>

      {storage && (
        <div
          className={cx(
            'flex flex-wrap items-center gap-x-3 gap-y-2 border-b px-6 py-3 text-[12px]',
            storage.stale ? 'border-warn/25 bg-warn-soft text-warn' : 'border-line bg-canvas/40 text-muted',
          )}
        >
          {storage.stale ? <Database size={14} className="shrink-0" /> : <ShieldCheck size={14} className="shrink-0 text-accent" />}
          <p className="min-w-0 flex-1 leading-snug">
            {storage.projects === 0 ? (
              <>Projects are kept in this browser only. Clearing site data deletes them, so keep a backup file.</>
            ) : storage.lastBackupAt === null ? (
              <>
                {storage.projects} project{storage.projects === 1 ? '' : 's'} · {kb(storage.bytes)} in this browser, with no backup
                yet. Clearing site data or switching browser loses them.
              </>
            ) : (
              <>
                {storage.projects} project{storage.projects === 1 ? '' : 's'} · {kb(storage.bytes)} in this browser. Last backup{' '}
                {ago(storage.lastBackupAt)}
                {storage.unbacked > 0 && <> · {storage.unbacked} changed since.</>}
              </>
            )}
          </p>
          <Button size="sm" variant={storage.stale ? 'primary' : 'ghost'} onClick={backup} disabled={storage.projects === 0}>
            <Database size={13} /> Back up all
          </Button>
        </div>
      )}

      {unsaved && (
        <div className="border-b border-line bg-accent-soft/40 px-6 py-2.5 text-[12px] text-ink-2">
          “{design.name}” has changes that are not in a project or a file yet.
        </div>
      )}

      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto p-3">
        {items.length === 0 ? (
          <div className="py-10 text-center text-[13px] text-muted">
            <FolderOpen size={22} className="mx-auto mb-2 text-faint" />
            No saved projects yet.
          </div>
        ) : (
          <ul className="space-y-1">
            {items.map((p) => (
              <li
                key={p.id}
                className={cx(
                  'flex items-center gap-3 rounded-lg border px-3 py-2 transition-colors',
                  p.id === currentId ? 'border-accent/40 bg-accent-soft/50' : 'border-transparent hover:bg-subtle',
                )}
              >
                <button type="button" onClick={() => open(p.id)} className="min-w-0 flex-1 text-left">
                  <div className="truncate text-[13px] font-medium text-ink">{p.name}</div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted">
                    <Clock size={11} /> {ago(p.savedAt)} · {p.compartments} compartments
                  </div>
                </button>
                {confirmId === p.id ? (
                  <div className="flex items-center gap-1">
                    <Button
                      size="sm"
                      variant="danger"
                      onClick={() => {
                        deleteProject(p.id);
                        if (p.id === currentId) useDesign.setState({ projectId: null });
                        setConfirmId(null);
                        refresh();
                      }}
                    >
                      Delete
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setConfirmId(null)}>
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <>
                    <Button size="sm" onClick={() => open(p.id)}>
                      Open
                    </Button>
                    <IconButton label="Delete project" onClick={() => setConfirmId(p.id)}>
                      <Trash2 size={14} />
                    </IconButton>
                  </>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
