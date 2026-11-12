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
import { useT, useUi, tr } from '../store/uiStore';
import { safeFileName } from '../geometry/stl';
import { Button, IconButton, Modal, cx } from './ui/controls';

function kb(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} kB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function ago(at: number, lang: string) {
  const m = Math.round((Date.now() - at) / 60000);
  if (m < 1) return tr('just now');
  if (m < 60) return tr('{n} min ago', { n: m });
  const h = Math.round(m / 60);
  if (h < 24) return tr('{n} h ago', { n: h });
  return new Date(at).toLocaleDateString(lang);
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
  const t = useT();
  const lang = useUi((s) => s.lang);
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
      ui.notify(t('Could not save – browser storage is full'), 'warn');
      return;
    }
    useDesign.setState({ projectId: id });
    useDesign.getState().markSaved();
    refresh();
    ui.notify(t('Saved “{name}”', { name: design.name }));
  };

  const backup = () => {
    const count = backupAll();
    if (!count) {
      ui.notify(t('There is nothing saved to back up yet'), 'warn');
      return;
    }
    refresh();
    ui.notify(count === 1 ? t('Backed up 1 project to a file') : t('Backed up {n} projects to a file', { n: count }));
  };

  const exportFile = () => {
    downloadDesign(design, safeFileName(design.name));
    useDesign.getState().markSaved();
  };

  const open = (id: string) => {
    const d = loadProject(id);
    if (!d) {
      ui.notify(t('That project could not be read'), 'warn');
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
          ui.notify(t('That backup could not be restored – browser storage may be full'), 'warn');
          return;
        }
        refresh();
        ui.notify(
          (result.added === 1 ? t('Restored 1 project') : t('Restored {n} projects', { n: result.added })) +
            (result.skipped ? ` · ${t('{n} skipped', { n: result.skipped })}` : ''),
        );
        return;
      }
      const parsed = parseDesign(data);
      if (!parsed) {
        ui.notify(t('That file is not an Organa design'), 'warn');
        return;
      }
      useDesign.getState().loadDesign(parsed);
      useDesign.setState({ projectId: null });
      ui.select(null);
      close();
      ui.notify(`${t('Imported “{name}”', { name: parsed.name })} · ${t('{n} compartments', { n: parsed.compartments.length })}`);
    } catch {
      ui.notify(t('That file could not be read'), 'warn');
    }
  };

  return (
    <div className="flex max-h-[88vh] flex-col">
      <div className="flex items-start gap-3 border-b border-line px-6 py-5">
        <div className="flex-1">
          <h2 className="text-[17px] font-semibold tracking-tight text-ink">{t('Projects')}</h2>
          <p className="mt-0.5 text-[12.5px] text-muted">{t('Saved in this browser. Export a file to keep or share a design.')}</p>
        </div>
        <IconButton label={t('Close')} onClick={close}>
          <X size={16} />
        </IconButton>
      </div>

      <div className="flex flex-wrap gap-2 border-b border-line bg-canvas/60 px-6 py-3">
        <Button variant="primary" onClick={() => save(false)}>
          <Save size={14} /> {currentId ? t('Save') : t('Save project')}
        </Button>
        {currentId && (
          <Button onClick={() => save(true)}>
            <HardDriveDownload size={14} /> {t('Save as copy')}
          </Button>
        )}
        <div className="flex-1" />
        <Button onClick={exportFile}>
          <Download size={14} /> {t('Export file')}
        </Button>
        <Button onClick={() => fileRef.current?.click()} title={t('Open a design file or restore a backup')}>
          <Upload size={14} /> {t('Import')}
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
              <>{t('Projects are kept in this browser only. Clearing site data deletes them, so keep a backup file.')}</>
            ) : storage.lastBackupAt === null ? (
              <>
                {t('{n} saved', { n: storage.projects })} ·{' '}
                {t('{size} in this browser, with no backup yet. Clearing site data or switching browser loses them.', {
                  size: kb(storage.bytes),
                })}
              </>
            ) : (
              <>
                {t('{n} saved', { n: storage.projects })} · {t('{size} in this browser', { size: kb(storage.bytes) })} ·{' '}
                {t('last backup {when}', { when: ago(storage.lastBackupAt, lang) })}
                {storage.unbacked > 0 && <> · {t('{n} changed since', { n: storage.unbacked })}</>}
              </>
            )}
          </p>
          <Button size="sm" variant={storage.stale ? 'primary' : 'ghost'} onClick={backup} disabled={storage.projects === 0}>
            <Database size={13} /> {t('Back up all')}
          </Button>
        </div>
      )}

      {unsaved && (
        <div className="border-b border-line bg-accent-soft/40 px-6 py-2.5 text-[12px] text-ink-2">
          {t('“{name}” has changes that are not in a project or a file yet.', { name: design.name })}
        </div>
      )}

      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto p-3">
        {items.length === 0 ? (
          <div className="py-10 text-center text-[13px] text-muted">
            <FolderOpen size={22} className="mx-auto mb-2 text-faint" />
            {t('No saved projects yet.')}
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
                    <Clock size={11} /> {ago(p.savedAt, lang)} · {t('{n} compartments', { n: p.compartments })}
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
                      {t('Delete')}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setConfirmId(null)}>
                      {t('Cancel')}
                    </Button>
                  </div>
                ) : (
                  <>
                    <Button size="sm" onClick={() => open(p.id)}>
                      {t('Open')}
                    </Button>
                    <IconButton label={t('Delete project')} onClick={() => setConfirmId(p.id)}>
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
