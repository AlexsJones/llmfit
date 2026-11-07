import { Keyboard, X } from 'lucide-react';
import { useT } from '../store/uiStore';
import { Kbd, IconButton, Modal } from './ui/controls';

const isMac = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform ?? '');
const MOD = isMac ? '⌘' : 'Ctrl';

/** Built per render so the labels follow the chosen language. */
const groups = (t: (k: string) => string): { title: string; items: [string, string][] }[] => [
  {
    title: t('Layout'),
    items: [
      ['A', t('Add a compartment')],
      [t('Double-click'), t('Fill empty space with a new compartment')],
      [t('Drag'), t('Move; it slides along neighbours')],
      [t('Drag an edge or corner'), t('Resize')],
      [t('Arrows'), t('Nudge 1 mm (Shift: 10 mm)')],
      [t('Delete'), t('Delete the selection')],
    ],
  },
  {
    title: t('Selection'),
    items: [
      [t('Shift-click'), t('Add or remove from the selection')],
      [t('Drag on empty space'), t('Select with a box')],
      [`${MOD}+A`, t('Select all')],
      [`${MOD}+M`, t('Merge the selection into one')],
      [`${MOD}+C / ${MOD}+V`, t('Copy and paste')],
      [`${MOD}+D`, t('Duplicate')],
      [t('Esc'), t('Deselect')],
    ],
  },
  {
    title: t('View'),
    items: [
      ['1 / 2 / 3', t('2D, split, 3D')],
      [t('Scroll'), t('Zoom')],
      [t('Right-drag or Alt-drag'), t('Pan')],
      [t('Drag in 3D'), t('Orbit')],
    ],
  },
  {
    title: t('File'),
    items: [
      [`${MOD}+Z`, t('Undo')],
      [`Shift+${MOD}+Z`, t('Redo')],
      [`${MOD}+S`, t('Projects, import and export')],
      ['?', t('This list')],
    ],
  },
];

export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const t = useT();
  const GROUPS = groups(t);
  return (
    <Modal open={open} onClose={onClose} width="max-w-2xl">
      <div className="flex max-h-[88vh] flex-col">
        <div className="flex items-center gap-3 border-b border-line px-6 py-4">
          <Keyboard size={16} className="text-accent" />
          <h2 className="flex-1 text-[16px] font-semibold tracking-tight text-ink">{t('Keyboard & mouse')}</h2>
          <IconButton label={t('Close')} onClick={onClose}>
            <X size={16} />
          </IconButton>
        </div>
        <div className="scroll-thin grid min-h-0 flex-1 gap-x-8 gap-y-5 overflow-y-auto px-6 py-5 sm:grid-cols-2">
          {GROUPS.map((g) => (
            <section key={g.title}>
              <h3 className="mb-2 text-[10.5px] font-semibold tracking-[0.06em] text-muted uppercase">{g.title}</h3>
              <dl className="space-y-1.5">
                {g.items.map(([keys, what]) => (
                  <div key={keys} className="flex items-baseline justify-between gap-3">
                    <dt className="shrink-0">
                      <Kbd>{keys}</Kbd>
                    </dt>
                    <dd className="flex-1 text-right text-[12px] leading-snug text-ink-2">{what}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </div>
    </Modal>
  );
}
