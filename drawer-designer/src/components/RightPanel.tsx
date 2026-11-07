import {
  AlertTriangle,
  ArrowLeftRight,
  ArrowUpDown,
  CheckCircle2,
  Columns3,
  Combine,
  Copy,
  Download,
  Maximize2,
  PenLine,
  Rows3,
  Scissors,
  Trash2,
  X,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import { useGeometry } from '../hooks/geometry';
import { decodeOutline } from '../geometry/outline';
import { OutlineEditor } from './OutlineEditor';
import { coverage, mergeBox, rectOf, resizeEdge, sweepMove } from '../layout/ops';
import { LIMITS, MIN_COMPARTMENT, usableArea } from '../model/defaults';
import type { Compartment, DividerOrientation } from '../model/types';
import { PLA_DENSITY, fitsBuildVolume } from '../geometry/parts';
import { exportCompartment } from '../geometry/exporter';
import { getModule, MODULES } from '../modules/registry';
import { useDesign } from '../store/designStore';
import { useUi } from '../store/uiStore';
import { Badge, Button, IconButton, NumberField, Section, Segmented, Stat, Toggle, cx } from './ui/controls';

const fmt = (v: number) => String(Math.round(v * 10) / 10);

export function RightPanel() {
  const selectedId = useUi((s) => s.selectedId);
  const selectedIds = useUi((s) => s.selectedIds);
  const compartment = useDesign((s) => s.design.compartments.find((c) => c.id === selectedId) ?? null);
  if (selectedIds.length > 1) return <MultiSelection ids={selectedIds} />;
  return compartment ? <CompartmentProperties key={compartment.id} c={compartment} /> : <OrganiserProperties />;
}

/** Properties shown when several compartments are selected. */
function MultiSelection({ ids }: { ids: string[] }) {
  const design = useDesign((s) => s.design);
  const actions = useDesign.getState();
  const ui = useUi.getState();
  const picked = design.compartments.filter((c) => ids.includes(c.id));
  const others = design.compartments.filter((c) => !ids.includes(c.id)).map(rectOf);
  const box = mergeBox(picked.map(rectOf), others);
  const area = picked.reduce((s, c) => s + c.w * c.d, 0);
  const kinds = [...new Set(picked.map((c) => c.kind))];

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="text-[14px] font-semibold text-ink">{picked.length} compartments</div>
          <div className="mt-0.5 text-[12px] text-muted">{kinds.length === 1 ? getModule(kinds[0]).label : `${kinds.length} module types`}</div>
        </div>
        <IconButton label="Deselect (Esc)" onClick={() => ui.select(null)}>
          <X size={15} />
        </IconButton>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <Section title="Selection">
          <Stat label="Compartments" value={picked.length} />
          <Stat label="Combined area" value={`${Math.round(area / 100) / 100} cm²`} />
          <Button
            className="mt-2 w-full"
            disabled={!box}
            title={box ? 'Combine into one compartment' : 'Only compartments that together form a rectangle can be merged'}
            onClick={() => {
              const id = actions.mergeCompartments(ids);
              if (id) ui.select(id);
              else ui.notify('These compartments do not form a rectangle', 'warn');
            }}
          >
            <Combine size={13} /> Merge into one
          </Button>
          {!box && <p className="mt-1.5 text-[11px] leading-relaxed text-faint">Merging needs a selection that forms a full rectangle with no gaps.</p>}
        </Section>

        <Section title="Apply to all">
          <div className="mb-1.5 text-[11px] font-medium text-muted">Module</div>
          <select
            value={kinds.length === 1 ? kinds[0] : ''}
            onChange={(e) => e.target.value && actions.setKind(ids, e.target.value)}
            className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-[13px] outline-none focus:border-accent"
          >
            {kinds.length > 1 && <option value="">Mixed</option>}
            {MODULES.filter((m) => m.status === 'available').map((m) => (
              <option key={m.kind} value={m.kind}>
                {m.label}
              </option>
            ))}
          </select>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <Button size="sm" onClick={() => ids.forEach((id) => actions.updateCompartment(id, { height: null }))}>
              Default height
            </Button>
            <Button size="sm" onClick={() => ids.forEach((id) => actions.expandCompartment(id))}>
              <Maximize2 size={12} /> Expand
            </Button>
          </div>
          <Button variant="danger" size="sm" className="mt-2 w-full" onClick={() => { actions.deleteMany(ids); ui.select(null); }}>
            <Trash2 size={12} /> Delete {picked.length} compartments
          </Button>
        </Section>

        <Section title="Tips">
          <p className="text-[11.5px] leading-relaxed text-muted">
            Shift-click to add or remove compartments. Drag on empty space to draw a selection box. Drag any selected compartment to move
            the whole group.
          </p>
        </Section>
      </div>
    </div>
  );
}

function CompartmentProperties({ c }: { c: Compartment }) {
  const design = useDesign((s) => s.design);
  const actions = useDesign.getState();
  const ui = useUi.getState();
  const selectedDividerId = useUi((s) => s.selectedDividerId);
  const { byCompartment } = useGeometry();
  const geo = byCompartment.get(c.id);
  const mod = getModule(c.kind);
  const usable = usableArea(design.drawer);
  const bounds = { width: usable.width, depth: usable.depth };
  const others = design.compartments.filter((o) => o.id !== c.id).map(rectOf);
  const params = { ...mod.defaults, ...c.params };

  const setPos = (axis: 'x' | 'y', v: number) => {
    const r = sweepMove(rectOf(c), axis === 'x' ? v : c.x, axis === 'y' ? v : c.y, others, bounds);
    actions.setCompartmentRect(c.id, r);
    if (Math.abs((axis === 'x' ? r.x : r.y) - v) > 0.05) ui.notify('Position limited by a neighbour or the drawer edge', 'warn');
  };
  const setSize = (axis: 'w' | 'd', v: number) => {
    const r = resizeEdge(rectOf(c), axis === 'w' ? 'right' : 'bottom', (axis === 'w' ? c.x : c.y) + v, others, bounds);
    actions.setCompartmentRect(c.id, r);
    if (Math.abs((axis === 'w' ? r.w : r.d) - v) > 0.05) ui.notify('Size limited by a neighbour or the drawer edge', 'warn');
  };
  const split = (cut: DividerOrientation, n: number) => {
    const ids = actions.splitCompartment(c.id, cut, n);
    if (ids.length) ui.select(ids[0]);
    else ui.notify('Too small to split further', 'warn');
  };

  const exceeds = geo?.parts.some((p) => !fitsBuildVolume(p.size, design.buildVolume)) ?? false;

  const exportThis = () => geo && exportCompartment(design, geo.parts);
  const [outlineKey, setOutlineKey] = useState<string | null>(null);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <input
            value={c.name}
            onChange={(e) => actions.updateCompartment(c.id, { name: e.target.value }, true)}
            onFocus={() => actions.beginGesture()}
            onBlur={() => actions.endGesture()}
            className="w-full rounded-md bg-transparent px-1 -mx-1 text-[14px] font-semibold text-ink outline-none hover:bg-subtle focus:bg-subtle"
          />
          <div className="mt-0.5 flex items-center gap-1.5">
            <Badge tone="accent">{mod.label}</Badge>
            {geo?.error ? <Badge tone="warn">{geo.error}</Badge> : exceeds ? <Badge tone="warn">Too large to print</Badge> : <Badge tone="ok">Printable</Badge>}
          </div>
        </div>
        <IconButton label="Deselect (Esc)" onClick={() => ui.select(null)}>
          <X size={15} />
        </IconButton>
      </div>

      {outlineKey && (
        <OutlineEditor
          value={String(params[outlineKey] ?? '')}
          onCancel={() => setOutlineKey(null)}
          onApply={(encoded) => {
            actions.updateCompartment(c.id, { params: { ...c.params, [outlineKey]: encoded } });
            setOutlineKey(null);
          }}
        />
      )}
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <Section title="Position & size">
          <div className="grid grid-cols-2 gap-2">
            <NumberField label="X" value={c.x} min={0} max={bounds.width - c.w} onChange={(v) => setPos('x', v)} />
            <NumberField label="Y" value={c.y} min={0} max={bounds.depth - c.d} onChange={(v) => setPos('y', v)} />
            <NumberField label="Width" value={c.w} min={MIN_COMPARTMENT} max={bounds.width} onChange={(v) => setSize('w', v)} />
            <NumberField label="Depth" value={c.d} min={MIN_COMPARTMENT} max={bounds.depth} onChange={(v) => setSize('d', v)} />
          </div>
          <div className="mt-3">
            <Toggle
              label="Custom height"
              checked={c.height !== null}
              onChange={(on) => actions.updateCompartment(c.id, { height: on ? design.print.binHeight : null })}
            />
            {c.height !== null && (
              <div className="mt-2">
                <NumberField label="Height" value={c.height} min={LIMITS.binHeight.min} max={LIMITS.binHeight.max} onChange={(height) => actions.updateCompartment(c.id, { height })} />
              </div>
            )}
          </div>
          <div className="mt-3">
            <div className="mb-1.5 text-[11px] font-medium text-muted">Module</div>
            <select
              value={c.kind}
              onChange={(e) => actions.updateCompartment(c.id, { kind: e.target.value, params: { ...getModule(e.target.value).defaults } })}
              className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-[13px] outline-none focus:border-accent"
            >
              {MODULES.filter((m) => m.status === 'available').map((m) => (
                <option key={m.kind} value={m.kind}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </Section>

        {mod.params.length > 0 && (
          <Section title={mod.label}>
            <div className="grid grid-cols-2 gap-2">
              {mod.params.map((p) =>
                p.type === 'number' ? (
                  <NumberField
                    key={p.key}
                    label={p.label}
                    unit={p.unit ?? ''}
                    value={Number(params[p.key])}
                    min={p.min}
                    max={p.max}
                    step={p.step}
                    hint={p.help}
                    onChange={(v) => actions.updateCompartment(c.id, { params: { ...c.params, [p.key]: v } })}
                  />
                ) : p.type === 'boolean' ? (
                  <div key={p.key} className="col-span-2">
                    <Toggle label={p.label} checked={Boolean(params[p.key])} onChange={(v) => actions.updateCompartment(c.id, { params: { ...c.params, [p.key]: v } })} />
                  </div>
                ) : p.type === 'outline' ? (
                  <div key={p.key} className="col-span-2">
                    <span className="mb-1 block text-[11px] font-medium text-muted">{p.label}</span>
                    <div className="flex items-center gap-2">
                      <OutlineThumb value={String(params[p.key] ?? '')} />
                      <div className="flex-1">
                        <Button size="sm" className="w-full" onClick={() => setOutlineKey(p.key)}>
                          <PenLine size={12} /> {decodeOutline(params[p.key]).length >= 3 ? 'Edit shape' : 'Draw or trace a shape'}
                        </Button>
                        {decodeOutline(params[p.key]).length >= 3 && (
                          <Button size="sm" variant="ghost" className="mt-1 w-full text-muted" onClick={() => actions.updateCompartment(c.id, { params: { ...c.params, [p.key]: '' } })}>
                            Remove shape
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ) : p.type === 'select' ? (
                  <label key={p.key} className="col-span-2 block">
                    <span className="mb-1 block text-[11px] font-medium text-muted">{p.label}</span>
                    <select
                      value={String(params[p.key])}
                      onChange={(e) => actions.updateCompartment(c.id, { params: { ...c.params, [p.key]: e.target.value } })}
                      className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-[13px] outline-none focus:border-accent"
                    >
                      {p.options?.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null,
              )}
            </div>
          </Section>
        )}

        {mod.supportsDividers && (
          <Section title="Dividers">
            <div className="grid grid-cols-2 gap-2">
              <Button size="sm" onClick={() => { const id = actions.addDivider(c.id, 'vertical'); if (id) ui.select(c.id, id); }}>
                <Columns3 size={13} /> Vertical
              </Button>
              <Button size="sm" onClick={() => { const id = actions.addDivider(c.id, 'horizontal'); if (id) ui.select(c.id, id); }}>
                <Rows3 size={13} /> Horizontal
              </Button>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <EvenCount label="Evenly ↔" count={c.dividers.filter((d) => d.orientation === 'vertical').length} onChange={(n) => actions.setEvenDividers(c.id, 'vertical', n)} />
              <EvenCount label="Evenly ↕" count={c.dividers.filter((d) => d.orientation === 'horizontal').length} onChange={(n) => actions.setEvenDividers(c.id, 'horizontal', n)} />
            </div>
            {c.dividers.length > 0 && (
              <div className="mt-3 space-y-1.5">
                {c.dividers.map((dv) => (
                  <div
                    key={dv.id}
                    className={cx('flex items-center gap-2 rounded-lg p-1 pl-2 transition-colors', dv.id === selectedDividerId ? 'bg-accent-soft' : 'bg-subtle/70')}
                    onClick={() => ui.select(c.id, dv.id)}
                  >
                    {dv.orientation === 'vertical' ? <Columns3 size={13} className="text-muted" /> : <Rows3 size={13} className="text-muted" />}
                    <div className="flex-1">
                      <NumberField
                        compact
                        value={dv.position}
                        min={0}
                        max={dv.orientation === 'vertical' ? c.w : c.d}
                        onChange={(v) => actions.updateDivider(c.id, dv.id, v)}
                        hint={dv.orientation === 'vertical' ? 'Distance from the left edge' : 'Distance from the back edge'}
                      />
                    </div>
                    <IconButton label="Remove divider" className="h-7 w-7" onClick={(e) => { e.stopPropagation(); actions.removeDivider(c.id, dv.id); }}>
                      <Trash2 size={13} />
                    </IconButton>
                  </div>
                ))}
              </div>
            )}
            <p className="mt-2 text-[11px] leading-relaxed text-faint">Drag dividers in the editor to reposition them; double-click to remove.</p>
          </Section>
        )}

        <Section title="Arrange">
          <div className="grid grid-cols-2 gap-1.5">
            <Button size="sm" onClick={() => split('vertical', 2)} title="Split into two side-by-side compartments">
              <ArrowLeftRight size={12} /> Split ½
            </Button>
            <Button size="sm" onClick={() => split('horizontal', 2)} title="Split into two stacked compartments">
              <ArrowUpDown size={12} /> Split ½
            </Button>
            <Button size="sm" onClick={() => split('vertical', 3)}>
              <ArrowLeftRight size={12} /> Split ⅓
            </Button>
            <Button size="sm" onClick={() => split('horizontal', 3)}>
              <ArrowUpDown size={12} /> Split ⅓
            </Button>
            <Button size="sm" onClick={() => actions.expandCompartment(c.id)} title="Grow until touching neighbours">
              <Maximize2 size={12} /> Expand
            </Button>
            <Button size="sm" onClick={() => { const id = actions.duplicateCompartment(c.id); if (id) ui.select(id); else ui.notify('No free space for a copy', 'warn'); }}>
              <Copy size={12} /> Duplicate
            </Button>
            <Button size="sm" onClick={() => actions.distribute(c.id, 'x')} title="Equalise widths across this row">
              <Columns3 size={12} /> Even row
            </Button>
            <Button size="sm" onClick={() => actions.distribute(c.id, 'y')} title="Equalise depths across this column">
              <Rows3 size={12} /> Even column
            </Button>
          </div>
          <Button variant="danger" size="sm" className="mt-2 w-full" onClick={() => { actions.deleteCompartment(c.id); ui.select(null); }}>
            <Trash2 size={12} /> Delete compartment
          </Button>
        </Section>

        <Section title="Printable parts">
          {geo?.parts.map((p) => {
            const ok = fitsBuildVolume(p.size, design.buildVolume);
            return (
              <div key={p.id} className="mb-2 rounded-lg border border-line p-2.5 last:mb-0">
                <div className="flex items-center justify-between">
                  <span className="truncate text-[12px] font-medium text-ink">{p.name}</span>
                  {ok ? <CheckCircle2 size={14} className="text-ok" /> : <AlertTriangle size={14} className="text-warn" />}
                </div>
                <div className="num mt-1 font-mono text-[11.5px] text-muted">
                  {fmt(p.size[0])} × {fmt(p.size[1])} × {fmt(p.size[2])} mm
                </div>
                <div className="num text-[11px] text-faint">≈ {fmt(p.volume * PLA_DENSITY)} g PLA · {p.triangles.toLocaleString()} tris</div>
                {p.warnings.map((w) => (
                  <div key={w} className="mt-1 text-[11px] text-warn">
                    {w}
                  </div>
                ))}
              </div>
            );
          }).slice(0, 4)}
          {(geo?.parts.length ?? 0) > 4 && <div className="text-[11px] text-muted">+ {(geo?.parts.length ?? 0) - 4} more identical inserts</div>}
          {exceeds && (
            <Button size="sm" className="mt-2 w-full" onClick={() => { const ids = actions.splitToFit(c.id); if (ids.length) { ui.select(ids[0]); ui.notify(`Split into ${ids.length} printable pieces`); } }}>
              <Scissors size={12} /> Split to fit build plate
            </Button>
          )}
          <Button size="sm" className="mt-2 w-full" disabled={!geo?.parts.length} onClick={exportThis}>
            <Download size={12} /> {(geo?.parts.length ?? 0) > 1 ? 'Download STLs (ZIP)' : 'Download STL'}
          </Button>
        </Section>
      </div>
    </div>
  );
}

/** Small preview of a stored outline. */
function OutlineThumb({ value }: { value: string }) {
  const pts = decodeOutline(value);
  return (
    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg border border-line bg-subtle">
      {pts.length >= 3 ? (
        <svg viewBox="-0.08 -0.08 1.16 1.16" className="h-9 w-9">
          <polygon points={pts.map((p) => `${p.x},${p.y}`).join(' ')} fill="#dbe3ff" stroke="#3a63f3" strokeWidth={0.04} strokeLinejoin="round" />
        </svg>
      ) : (
        <PenLine size={15} className="text-faint" />
      )}
    </div>
  );
}

function EvenCount({ label, count, onChange }: { label: string; count: number; onChange: (n: number) => void }) {
  return (
    <div className="flex h-7 items-center justify-between rounded-lg border border-line bg-surface pl-2 text-[11.5px] text-muted">
      <span>{label}</span>
      <div className="flex items-center">
        <button type="button" className="h-7 w-6 text-ink-2 hover:text-ink disabled:opacity-30" disabled={count === 0} onClick={() => onChange(Math.max(0, count - 1))}>
          −
        </button>
        <span className="num w-4 text-center text-ink">{count}</span>
        <button type="button" className="h-7 w-6 text-ink-2 hover:text-ink" onClick={() => onChange(Math.min(20, count + 1))}>
          +
        </button>
      </div>
    </div>
  );
}

function OrganiserProperties() {
  const design = useDesign((s) => s.design);
  const actions = useDesign.getState();
  const { print, buildVolume } = design;
  const { parts, byCompartment } = useGeometry();
  const usable = usableArea(design.drawer);
  const cov = coverage(design.compartments, { width: usable.width, depth: usable.depth });
  const grams = useMemo(() => parts.reduce((s, p) => s + p.volume * PLA_DENSITY, 0), [parts]);
  const oversized = parts.filter((p) => !fitsBuildVolume(p.size, buildVolume));
  const tooTall = print.binHeight > design.drawer.height;
  const errors = [...byCompartment.values()].filter((g) => g.error);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-line px-4 py-3">
        <div className="text-[14px] font-semibold text-ink">Organiser</div>
        <div className="mt-0.5 text-[12px] text-muted">Global settings · select a compartment to edit it</div>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <Section title="Walls & floor">
          <div className="grid grid-cols-2 gap-2">
            <NumberField label="Wall thickness" value={print.wallThickness} step={0.2} min={LIMITS.wall.min} max={LIMITS.wall.max} onChange={(wallThickness) => actions.updatePrint({ wallThickness })} hint="Multiples of your line width print best (0.4 mm nozzle → 1.2 / 1.6 / 2.0)" />
            <NumberField label="Bottom thickness" value={print.bottomThickness} step={0.2} min={LIMITS.bottom.min} max={LIMITS.bottom.max} onChange={(bottomThickness) => actions.updatePrint({ bottomThickness })} />
            <NumberField label="Bin height" value={print.binHeight} min={LIMITS.binHeight.min} max={LIMITS.binHeight.max} onChange={(binHeight) => actions.updatePrint({ binHeight })} />
            <NumberField label="Fit gap" value={print.fitGap} step={0.1} min={LIMITS.fitGap.min} max={LIMITS.fitGap.max} onChange={(fitGap) => actions.updatePrint({ fitGap })} hint="Total gap between neighbouring bins so they drop in without force" />
          </div>
          {tooTall && <div className="mt-2 text-[11px] text-warn">Bins are taller than the drawer ({design.drawer.height} mm).</div>}
        </Section>

        <Section title="Corners">
          <Segmented
            className="w-full"
            value={print.cornerStyle}
            onChange={(cornerStyle) => actions.updatePrint({ cornerStyle })}
            options={[
              { value: 'rounded', label: 'Rounded' },
              { value: 'square', label: 'Square' },
            ]}
          />
          {print.cornerStyle === 'rounded' && (
            <div className="mt-2">
              <NumberField label="Corner radius" value={print.cornerRadius} step={0.5} min={0.5} max={LIMITS.radius.max} onChange={(cornerRadius) => actions.updatePrint({ cornerRadius })} />
            </div>
          )}
        </Section>

        <Section title="Dividers">
          <div className="grid grid-cols-2 gap-2">
            <NumberField label="Thickness" value={print.dividerThickness} step={0.2} min={LIMITS.dividerThickness.min} max={LIMITS.dividerThickness.max} onChange={(dividerThickness) => actions.updatePrint({ dividerThickness })} />
            <NumberField
              label="Height"
              value={print.dividerHeight ?? print.binHeight}
              disabled={print.dividerHeight === null}
              min={print.bottomThickness + 1}
              max={LIMITS.binHeight.max}
              onChange={(dividerHeight) => actions.updatePrint({ dividerHeight })}
            />
          </div>
          <div className="mt-2">
            <Toggle label="Full-height dividers" checked={print.dividerHeight === null} onChange={(on) => actions.updatePrint({ dividerHeight: on ? null : Math.round(print.binHeight * 0.7) })} />
          </div>
        </Section>

        <Section title="Printer">
          <div className="mb-1.5 text-[11px] font-medium text-muted">Build volume</div>
          <div className="grid grid-cols-3 gap-1.5">
            <NumberField compact value={buildVolume.x} min={LIMITS.build.min} max={LIMITS.build.max} onChange={(x) => actions.updateBuildVolume({ x })} unit="X" />
            <NumberField compact value={buildVolume.y} min={LIMITS.build.min} max={LIMITS.build.max} onChange={(y) => actions.updateBuildVolume({ y })} unit="Y" />
            <NumberField compact value={buildVolume.z} min={LIMITS.build.min} max={LIMITS.build.max} onChange={(z) => actions.updateBuildVolume({ z })} unit="Z" />
          </div>
        </Section>

        <Section title="Summary">
          <Stat label="Compartments" value={design.compartments.length} />
          <Stat label="Printable parts" value={parts.length} />
          <Stat label="Area used" value={`${Math.round(cov * 100)}%`} />
          <Stat label="Est. filament (PLA)" value={`${Math.round(grams)} g`} />
          <Stat label="Oversized parts" value={oversized.length} tone={oversized.length ? 'warn' : 'ok'} />
          {(oversized.length > 0 || errors.length > 0) && (
            <div className="mt-2 space-y-1">
              {oversized.slice(0, 5).map((p) => (
                <button key={p.id} type="button" onClick={() => useUi.getState().select(p.compartmentId)} className="flex w-full items-center gap-1.5 rounded-md bg-warn-soft px-2 py-1 text-left text-[11px] text-warn">
                  <AlertTriangle size={12} /> {p.name} exceeds the build volume
                </button>
              ))}
              {errors.map((g) => (
                <button key={g.compartmentId} type="button" onClick={() => useUi.getState().select(g.compartmentId)} className="flex w-full items-center gap-1.5 rounded-md bg-warn-soft px-2 py-1 text-left text-[11px] text-warn">
                  <AlertTriangle size={12} /> {design.compartments.find((c) => c.id === g.compartmentId)?.name}: {g.error}
                </button>
              ))}
            </div>
          )}
        </Section>
      </div>
    </div>
  );
}
