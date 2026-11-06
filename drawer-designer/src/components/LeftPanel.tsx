import { Eraser, Grid2x2, Maximize2, Plus } from 'lucide-react';
import { useState } from 'react';
import { LIMITS, usableArea } from '../model/defaults';
import { MODULES } from '../modules/registry';
import { useDesign } from '../store/designStore';
import { useUi } from '../store/uiStore';
import { Badge, Button, NumberField, Section, Segmented, Toggle, cx } from './ui/controls';

const PRESETS: [number, number][] = [
  [1, 2],
  [2, 2],
  [2, 3],
  [3, 3],
  [3, 4],
  [4, 5],
];

export function LeftPanel() {
  const drawer = useDesign((s) => s.design.drawer);
  const count = useDesign((s) => s.design.compartments.length);
  const actions = useDesign.getState();
  const gridSnap = useUi((s) => s.gridSnap);
  const edgeSnap = useUi((s) => s.edgeSnap);
  const ui = useUi.getState();
  const usable = usableArea(drawer);
  const [rows, setRows] = useState(2);
  const [cols, setCols] = useState(3);

  const add = (kind: string) => {
    const id = actions.addCompartment(kind);
    if (id) ui.select(id);
    else ui.notify('No free space left – resize or delete a compartment first', 'warn');
  };

  const applyGrid = (r: number, c: number) => {
    actions.applyGrid(r, c);
    ui.select(null);
    if (count) ui.notify('Layout replaced · Undo with ⌘Z / Ctrl+Z');
  };

  return (
    <div className="flex h-full flex-col">
      <Section title="Drawer">
        <div className="grid grid-cols-2 gap-2">
          <NumberField label="Width" value={drawer.width} min={LIMITS.drawer.min} max={LIMITS.drawer.max} onChange={(width) => actions.updateDrawer({ width })} />
          <NumberField label="Depth" value={drawer.depth} min={LIMITS.drawer.min} max={LIMITS.drawer.max} onChange={(depth) => actions.updateDrawer({ depth })} />
          <NumberField label="Height" value={drawer.height} min={LIMITS.drawerHeight.min} max={LIMITS.drawerHeight.max} onChange={(height) => actions.updateDrawer({ height })} />
          <NumberField
            label="Wall clearance"
            value={drawer.clearance}
            step={0.5}
            min={LIMITS.clearance.min}
            max={LIMITS.clearance.max}
            onChange={(clearance) => actions.updateDrawer({ clearance })}
            hint="Gap kept between the organiser and each drawer wall"
          />
        </div>
        <div className="mt-3 flex items-center justify-between rounded-lg bg-subtle px-3 py-2">
          <span className="text-[11px] font-medium text-muted">Usable area</span>
          <span className="num font-mono text-[12px] text-ink">
            {usable.width} × {usable.depth} × {usable.height}
          </span>
        </div>
      </Section>

      <Section title="Layout">
        <div className="grid grid-cols-2 gap-2">
          <Button variant="primary" onClick={() => add('bin')} title="Add compartment (A)">
            <Plus size={14} /> Compartment
          </Button>
          <Button onClick={() => actions.fillGaps()} disabled={!count} title="Grow every compartment until it touches its neighbours">
            <Maximize2 size={13} /> Fill gaps
          </Button>
        </div>
        <div className="mt-3 mb-1.5 text-[11px] font-medium text-muted">Quick grid</div>
        <div className="grid grid-cols-3 gap-1.5">
          {PRESETS.map(([r, c]) => (
            <button
              key={`${r}x${c}`}
              type="button"
              onClick={() => applyGrid(r, c)}
              className="group flex flex-col items-center gap-1 rounded-lg border border-line bg-surface p-1.5 transition-all hover:border-accent/50 hover:shadow-panel"
              title={`${r} rows × ${c} columns`}
            >
              <GridGlyph rows={r} cols={c} />
              <span className="num text-[10.5px] text-muted group-hover:text-ink">
                {r}×{c}
              </span>
            </button>
          ))}
        </div>
        <div className="mt-2 flex items-end gap-1.5">
          <div className="flex-1">
            <NumberField compact label="Rows" unit="" value={rows} min={1} max={20} onChange={setRows} />
          </div>
          <div className="flex-1">
            <NumberField compact label="Cols" unit="" value={cols} min={1} max={20} onChange={setCols} />
          </div>
          <Button size="sm" className="h-7" onClick={() => applyGrid(rows, cols)}>
            <Grid2x2 size={12} /> Apply
          </Button>
        </div>
        <Button variant="ghost" size="sm" className="mt-2 w-full text-muted" disabled={!count} onClick={() => { actions.clearLayout(); ui.select(null); }}>
          <Eraser size={12} /> Clear layout
        </Button>
      </Section>

      <Section title="Snapping">
        <div className="mb-1.5 text-[11px] font-medium text-muted">Grid</div>
        <Segmented
          size="sm"
          className="w-full"
          value={String(gridSnap)}
          onChange={(v) => ui.setGridSnap(Number(v))}
          options={[
            { value: '0', label: 'Off' },
            { value: '1', label: '1 mm' },
            { value: '5', label: '5 mm' },
            { value: '10', label: '10 mm' },
          ]}
        />
        <div className="mt-2">
          <Toggle label="Snap to edges" checked={edgeSnap} onChange={ui.setEdgeSnap} />
        </div>
      </Section>

      <Section title="Modules">
        <div className="space-y-1.5">
          {MODULES.map((m) => {
            const available = m.status === 'available';
            return (
              <button
                key={m.kind}
                type="button"
                disabled={!available}
                onClick={() => add(m.kind)}
                className={cx(
                  'flex w-full items-center gap-2.5 rounded-lg border p-2 text-left transition-all',
                  available ? 'border-line bg-surface hover:border-accent/50 hover:shadow-panel' : 'border-dashed border-line bg-transparent',
                )}
              >
                <span className={cx('flex h-8 w-8 shrink-0 items-center justify-center rounded-md', available ? 'bg-accent-soft text-accent-ink' : 'bg-subtle text-faint')}>
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                    <path d={m.icon} />
                  </svg>
                </span>
                <span className="min-w-0 flex-1">
                  <span className={cx('flex items-center gap-1.5 text-[12.5px] font-medium', available ? 'text-ink' : 'text-muted')}>
                    {m.label}
                    {!available && <Badge>Soon</Badge>}
                  </span>
                  <span className="block truncate text-[11px] text-muted">{m.description}</span>
                </span>
                {available && <Plus size={14} className="text-faint" />}
              </button>
            );
          })}
        </div>
      </Section>
    </div>
  );
}

function GridGlyph({ rows, cols }: { rows: number; cols: number }) {
  const w = 34;
  const h = 22;
  const g = 1.5;
  const cw = (w - g * (cols - 1)) / cols;
  const ch = (h - g * (rows - 1)) / rows;
  const cells = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) cells.push(<rect key={`${r}${c}`} x={c * (cw + g)} y={r * (ch + g)} width={cw} height={ch} rx={1.5} />);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} className="fill-subtle stroke-line-strong group-hover:fill-accent-soft group-hover:stroke-accent/60" strokeWidth={0.8}>
      {cells}
    </svg>
  );
}
