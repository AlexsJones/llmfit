import { Keyboard, X } from 'lucide-react';
import { Kbd, IconButton, Modal } from './ui/controls';

const isMac = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform ?? '');
const MOD = isMac ? '⌘' : 'Ctrl';

const GROUPS: { title: string; items: [string, string][] }[] = [
  {
    title: 'Layout',
    items: [
      ['A', 'Add a compartment'],
      ['Double-click', 'Fill empty space with a new compartment'],
      ['Drag', 'Move; it slides along neighbours'],
      ['Drag an edge or corner', 'Resize'],
      ['Arrows', 'Nudge 1 mm (Shift: 10 mm)'],
      ['Delete', 'Delete the selection'],
    ],
  },
  {
    title: 'Selection',
    items: [
      ['Shift-click', 'Add or remove from the selection'],
      ['Drag on empty space', 'Select with a box'],
      [`${MOD}+A`, 'Select all'],
      [`${MOD}+M`, 'Merge the selection into one'],
      [`${MOD}+C / ${MOD}+V`, 'Copy and paste'],
      [`${MOD}+D`, 'Duplicate'],
      ['Esc', 'Deselect'],
    ],
  },
  {
    title: 'View',
    items: [
      ['1 / 2 / 3', '2D, split, 3D'],
      ['Scroll', 'Zoom'],
      ['Right-drag or Alt-drag', 'Pan'],
      ['Drag in 3D', 'Orbit'],
    ],
  },
  {
    title: 'File',
    items: [
      [`${MOD}+Z`, 'Undo'],
      [`Shift+${MOD}+Z`, 'Redo'],
      [`${MOD}+S`, 'Projects, import and export'],
      ['?', 'This list'],
    ],
  },
];

export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Modal open={open} onClose={onClose} width="max-w-2xl">
      <div className="flex max-h-[88vh] flex-col">
        <div className="flex items-center gap-3 border-b border-line px-6 py-4">
          <Keyboard size={16} className="text-accent" />
          <h2 className="flex-1 text-[16px] font-semibold tracking-tight text-ink">Keyboard &amp; mouse</h2>
          <IconButton label="Close" onClick={onClose}>
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
