import {
  AlertTriangle,
  ArrowLeftRight,
  ArrowUpDown,
  CheckCircle2,
  Columns3,
  Combine,
  Download,
  Info,
  Maximize2,
  PenLine,
  RotateCw,
  Rows3,
  Scissors,
  Shrink,
  Trash2,
  X,
} from 'lucide-react';
import { type ReactNode, useMemo, useState } from 'react';
import { useGeometry } from '../hooks/geometry';
import { decodeOutline } from '../geometry/outline';
import { OutlineEditor } from './OutlineEditor';
import { coverage, mergeBox, rectOf, resizeEdge, sweepMove } from '../layout/ops';
import { LIMITS, MIN_COMPARTMENT, usableArea } from '../model/defaults';
import { findPrinter, lineWidthFor, maxLayerHeight, printersByBrand, wallAdvice } from '../model/printers';
import type { Compartment, DividerOrientation } from '../model/types';
import { filamentGrams, fitsBuildVolume } from '../geometry/parts';
import { exportCompartment } from '../geometry/exporter';
import { getModule, MODULES } from '../modules/registry';
import { useDesign } from '../store/designStore';
import { useT, useUi, usePalette } from '../store/uiStore';
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
  const t = useT();

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="text-[14px] font-semibold text-ink">{t('{n} compartments', { n: picked.length })}</div>
          <div className="mt-0.5 text-[12px] text-muted">{kinds.length === 1 ? t(getModule(kinds[0]).label) : t('{n} module types', { n: kinds.length })}</div>
        </div>
        <IconButton label={`${t('Deselect')} (Esc)`} onClick={() => ui.select(null)}>
          <X size={15} />
        </IconButton>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <Section title={t('Selection')}>
          <Stat label={t('Compartments')} value={picked.length} />
          <Stat label={t('Combined area')} value={`${Math.round(area / 100) / 100} cm²`} />
          <Button
            className="mt-2 w-full"
            disabled={!box}
            title={box ? t('Combine into one compartment') : t('Only compartments that together form a rectangle can be merged')}
            onClick={() => {
              const id = actions.mergeCompartments(ids);
              if (id) ui.select(id);
              else ui.notify(t('These compartments do not form a rectangle'), 'warn');
            }}
          >
            <Combine size={13} /> {t('Merge into one')}
          </Button>
          {!box && <p className="mt-1.5 text-[11px] leading-relaxed text-faint">{t('Merging needs a selection that forms a full rectangle with no gaps.')}</p>}
        </Section>

        <Section title={t('Apply to all')}>
          <div className="mb-1.5 text-[11px] font-medium text-muted">{t('Module')}</div>
          <select
            value={kinds.length === 1 ? kinds[0] : ''}
            onChange={(e) => e.target.value && actions.setKind(ids, e.target.value)}
            className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-[13px] outline-none focus:border-accent"
          >
            {kinds.length > 1 && <option value="">{t('Mixed')}</option>}
            {MODULES.filter((m) => m.status === 'available').map((m) => (
              <option key={m.kind} value={m.kind}>
                {t(m.label)}
              </option>
            ))}
          </select>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <Button size="sm" onClick={() => ids.forEach((id) => actions.updateCompartment(id, { height: null }))}>
              {t('Default height')}
            </Button>
            <Button size="sm" onClick={() => ids.forEach((id) => actions.expandCompartment(id))}>
              <Maximize2 size={12} /> {t('Expand')}
            </Button>
          </div>
          <Button variant="danger" size="sm" className="mt-2 w-full" onClick={() => { actions.deleteMany(ids); ui.select(null); }}>
            <Trash2 size={12} /> {t('Delete {n} compartments', { n: picked.length })}
          </Button>
        </Section>

        <Section title={t('Tips')}>
          <p className="text-[11.5px] leading-relaxed text-muted">
            {t('Shift-click to add or remove compartments. Drag on empty space to draw a selection box. Drag any selected compartment to move the whole group.')}
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
  const t = useT();

  const setPos = (axis: 'x' | 'y', v: number) => {
    const r = sweepMove(rectOf(c), axis === 'x' ? v : c.x, axis === 'y' ? v : c.y, others, bounds);
    actions.setCompartmentRect(c.id, r);
    if (Math.abs((axis === 'x' ? r.x : r.y) - v) > 0.05) ui.notify(t('Position limited by a neighbour or the drawer edge'), 'warn');
  };
  const setSize = (axis: 'w' | 'd', v: number) => {
    const r = resizeEdge(rectOf(c), axis === 'w' ? 'right' : 'bottom', (axis === 'w' ? c.x : c.y) + v, others, bounds);
    actions.setCompartmentRect(c.id, r);
    if (Math.abs((axis === 'w' ? r.w : r.d) - v) > 0.05) ui.notify(t('Size limited by a neighbour or the drawer edge'), 'warn');
  };
  const split = (cut: DividerOrientation, n: number) => {
    const ids = actions.splitCompartment(c.id, cut, n);
    if (ids.length) ui.select(ids[0]);
    else ui.notify(t('Too small to split further'), 'warn');
  };

  const exceeds = geo?.parts.some((p) => !fitsBuildVolume(p.size, design.printer.build)) ?? false;
  const infill = design.printer.infill;

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
            <Badge tone="accent">{t(mod.label)}</Badge>
            {geo?.error ? (
              <Badge tone="warn">{t('Not buildable')}</Badge>
            ) : exceeds ? (
              <Badge tone="warn">{t('Too large to print')}</Badge>
            ) : (
              <Badge tone="ok">{t('Printable')}</Badge>
            )}
          </div>
        </div>
        <IconButton label={`${t('Deselect')} (Esc)`} onClick={() => ui.select(null)}>
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
      {geo?.error && (
        <div className="flex items-start gap-1.5 border-b border-line bg-warn-soft px-4 py-2 text-[11.5px] leading-relaxed text-warn">
          <AlertTriangle size={13} className="mt-[1px] shrink-0" />
          <span>{geo.error}</span>
        </div>
      )}
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <Section title={t('Position & size')}>
          <div className="grid grid-cols-2 gap-2">
            <NumberField label="X" value={c.x} min={0} max={bounds.width - c.w} onChange={(v) => setPos('x', v)} />
            <NumberField label="Y" value={c.y} min={0} max={bounds.depth - c.d} onChange={(v) => setPos('y', v)} />
            <NumberField label={t('Width')} value={c.w} min={MIN_COMPARTMENT} max={bounds.width} onChange={(v) => setSize('w', v)} />
            <NumberField label={t('Depth')} value={c.d} min={MIN_COMPARTMENT} max={bounds.depth} onChange={(v) => setSize('d', v)} />
          </div>
          <div className="mt-3">
            <div className="flex items-end gap-1.5">
              <div className="flex-1">
                <NumberField
                  compact
                  label={t('Rotation')}
                  unit="°"
                  value={c.rot ?? 0}
                  min={0}
                  max={359.9}
                  step={5}
                  onChange={(deg) => {
                    const before = useDesign.getState().design.compartments.find((x) => x.id === c.id)?.rot ?? 0;
                    actions.setRotation(c.id, deg);
                    const after = useDesign.getState().design.compartments.find((x) => x.id === c.id)?.rot ?? 0;
                    if (after === before && Math.round(deg) !== Math.round(before)) {
                      ui.notify(t('No room to turn it that far'), 'warn');
                    }
                  }}
                />
              </div>
              <Button size="sm" className="h-7" title={t('Turn 90°')} onClick={() => actions.setRotation(c.id, (c.rot ?? 0) + 90)}>
                <RotateCw size={12} /> 90°
              </Button>
              <Button size="sm" className="h-7" disabled={!c.rot} onClick={() => actions.setRotation(c.id, 0)}>
                {t('Reset')}
              </Button>
            </div>
            {Boolean(c.rot) && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-faint">
                {t('A turned compartment keeps its own frame: it can be moved and resized, but splitting, merging and the row and column tools need it back on the grid. The printed part is unchanged.')}
              </p>
            )}
          </div>
          <div className="mt-3">
            <Toggle
              label={t('Custom height')}
              checked={c.height !== null}
              onChange={(on) => actions.updateCompartment(c.id, { height: on ? design.print.binHeight : null })}
            />
            {c.height !== null && (
              <div className="mt-2">
                <NumberField label={t('Height')} value={c.height} min={LIMITS.binHeight.min} max={LIMITS.binHeight.max} onChange={(height) => actions.updateCompartment(c.id, { height })} />
              </div>
            )}
          </div>
          <div className="mt-3">
            <div className="mb-1.5 text-[11px] font-medium text-muted">{t('Module')}</div>
            <select
              value={c.kind}
              onChange={(e) => actions.updateCompartment(c.id, { kind: e.target.value, params: { ...getModule(e.target.value).defaults } })}
              className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-[13px] outline-none focus:border-accent"
            >
              {MODULES.filter((m) => m.status === 'available').map((m) => (
                <option key={m.kind} value={m.kind}>
                  {t(m.label)}
                </option>
              ))}
            </select>
          </div>
        </Section>

        {mod.params.length > 0 && (
          <Section title={t(mod.label)}>
            {mod.naturalSize && (
              <Button
                size="sm"
                className="mb-2.5 w-full"
                title={t('Give back the space this module does not use')}
                onClick={() => {
                  const result = actions.fitToModule(c.id);
                  if (result.kind === 'fitted') ui.notify(t('Compartment shrunk to the module'));
                  else if (result.kind === 'exact') ui.notify(t('The module already uses the whole compartment'));
                  else if (result.kind === 'small')
                    ui.notify(t('Too small for this module: it wants {w} × {d} mm', { w: result.w, d: result.d }), 'warn');
                }}
              >
                <Shrink size={12} /> {t('Shrink to fit the module')}
              </Button>
            )}
            <div className="grid grid-cols-2 gap-2">
              {mod.params
                // Keep the panel honest: only show fields that affect the current setup.
                .filter((p) => p.showIf?.(params) ?? true)
                .map((p) =>
                p.type === 'number' ? (
                  <NumberField
                    key={p.key}
                    label={t(p.label)}
                    unit={p.unit ?? ''}
                    value={Number(params[p.key])}
                    min={p.min}
                    max={p.max}
                    step={p.step}
                    hint={p.help ? t(p.help) : undefined}
                    onChange={(v) => actions.updateCompartment(c.id, { params: { ...c.params, [p.key]: v } })}
                  />
                ) : p.type === 'boolean' ? (
                  <div key={p.key} className="col-span-2">
                    <Toggle label={t(p.label)} checked={Boolean(params[p.key])} onChange={(v) => actions.updateCompartment(c.id, { params: { ...c.params, [p.key]: v } })} />
                  </div>
                ) : p.type === 'outline' ? (
                  <div key={p.key} className="col-span-2">
                    <span className="mb-1 block text-[11px] font-medium text-muted">{t(p.label)}</span>
                    <div className="flex items-center gap-2">
                      <OutlineThumb value={String(params[p.key] ?? '')} />
                      <div className="flex-1">
                        <Button size="sm" className="w-full" onClick={() => setOutlineKey(p.key)}>
                          <PenLine size={12} /> {decodeOutline(params[p.key]).length >= 3 ? t('Edit shape') : t('Draw or trace a shape')}
                        </Button>
                        {decodeOutline(params[p.key]).length >= 3 && (
                          <Button size="sm" variant="ghost" className="mt-1 w-full text-muted" onClick={() => actions.updateCompartment(c.id, { params: { ...c.params, [p.key]: '' } })}>
                            {t('Remove shape')}
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ) : p.type === 'select' ? (
                  <label key={p.key} className="col-span-2 block">
                    <span className="mb-1 block text-[11px] font-medium text-muted">{t(p.label)}</span>
                    <select
                      value={String(params[p.key])}
                      onChange={(e) => actions.updateCompartment(c.id, { params: { ...c.params, [p.key]: e.target.value } })}
                      className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-[13px] outline-none focus:border-accent"
                    >
                      {p.options?.map((o) => (
                        <option key={o.value} value={o.value}>
                          {t(o.label)}
                          {o.detail ? ` (${o.detail})` : ''}
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
          <Section title={t('Dividers')}>
            <div className="grid grid-cols-2 gap-2">
              <Button size="sm" onClick={() => { const id = actions.addDivider(c.id, 'vertical'); if (id) ui.select(c.id, id); }}>
                <Columns3 size={13} /> {t('Vertical')}
              </Button>
              <Button size="sm" onClick={() => { const id = actions.addDivider(c.id, 'horizontal'); if (id) ui.select(c.id, id); }}>
                <Rows3 size={13} /> {t('Horizontal')}
              </Button>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <EvenCount label={`${t('Evenly')} ↔`} count={c.dividers.filter((d) => d.orientation === 'vertical').length} onChange={(n) => actions.setEvenDividers(c.id, 'vertical', n)} />
              <EvenCount label={`${t('Evenly')} ↕`} count={c.dividers.filter((d) => d.orientation === 'horizontal').length} onChange={(n) => actions.setEvenDividers(c.id, 'horizontal', n)} />
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
                        hint={dv.orientation === 'vertical' ? t('Distance from the left edge') : t('Distance from the back edge')}
                      />
                    </div>
                    <IconButton label={t('Remove divider')} className="h-7 w-7" onClick={(e) => { e.stopPropagation(); actions.removeDivider(c.id, dv.id); }}>
                      <Trash2 size={13} />
                    </IconButton>
                  </div>
                ))}
              </div>
            )}
            <p className="mt-2 text-[11px] leading-relaxed text-faint">{t('Drag dividers in the editor to reposition them; double-click to remove.')}</p>
          </Section>
        )}

        <Section title={t('Arrange')} defaultOpen={false}>
          <p className="mb-2 text-[11px] leading-relaxed text-faint">{t('Split, duplicate and delete are also on the toolbar above the selected compartment.')}</p>
          <div className="grid grid-cols-2 gap-1.5">
            <Button size="sm" onClick={() => split('vertical', 3)}>
              <ArrowLeftRight size={12} /> {t('Split')} ⅓
            </Button>
            <Button size="sm" onClick={() => split('horizontal', 3)}>
              <ArrowUpDown size={12} /> {t('Split')} ⅓
            </Button>
            <Button size="sm" onClick={() => actions.distribute(c.id, 'x')} title={t('Equalise widths across this row')}>
              <Columns3 size={12} /> {t('Even row')}
            </Button>
            <Button size="sm" onClick={() => actions.distribute(c.id, 'y')} title={t('Equalise depths across this column')}>
              <Rows3 size={12} /> {t('Even column')}
            </Button>
            <Button size="sm" onClick={() => { const ids = actions.insertLane(c.id, 'x', true); if (!ids.length) ui.notify(t('No room to insert a column'), 'warn'); }}>
              <Columns3 size={12} /> {t('Column before')}
            </Button>
            <Button size="sm" onClick={() => { const ids = actions.insertLane(c.id, 'y', true); if (!ids.length) ui.notify(t('No room to insert a row'), 'warn'); }}>
              <Rows3 size={12} /> {t('Row before')}
            </Button>
          </div>
        </Section>

        <Section title={t('Printable parts')}>
          {geo?.parts.map((p) => {
            const ok = fitsBuildVolume(p.size, design.printer.build);
            return (
              <div key={p.id} className="mb-2 rounded-lg border border-line p-2.5 last:mb-0">
                <div className="flex items-center justify-between">
                  <span className="truncate text-[12px] font-medium text-ink">{p.name}</span>
                  {ok ? <CheckCircle2 size={14} className="text-ok" /> : <AlertTriangle size={14} className="text-warn" />}
                </div>
                <div className="num mt-1 font-mono text-[11.5px] text-muted">
                  {fmt(p.size[0])} × {fmt(p.size[1])} × {fmt(p.size[2])} mm
                </div>
                <div className="num text-[11px] text-faint">≈ {fmt(filamentGrams(p, infill))} g PLA · {t('{n} tris', { n: p.triangles.toLocaleString() })}</div>
                {p.warnings.map((w) => (
                  <div key={w} className="mt-1 text-[11px] text-warn">
                    {w}
                  </div>
                ))}
              </div>
            );
          }).slice(0, 4)}
          {(geo?.parts.length ?? 0) > 4 && <div className="text-[11px] text-muted">{t('+ {n} more identical inserts', { n: (geo?.parts.length ?? 0) - 4 })}</div>}
          {exceeds && (
            <Button size="sm" className="mt-2 w-full" onClick={() => { const ids = actions.splitToFit(c.id); if (ids.length) { ui.select(ids[0]); ui.notify(t('Split into {n} printable pieces', { n: ids.length })); } }}>
              <Scissors size={12} /> {t('Split to fit build plate')}
            </Button>
          )}
          <Button size="sm" className="mt-2 w-full" disabled={!geo?.parts.length} onClick={exportThis}>
            <Download size={12} /> {(geo?.parts.length ?? 0) > 1 ? t('Download STLs (ZIP)') : t('Download STL')}
          </Button>
        </Section>
      </div>
    </div>
  );
}

/** Small preview of a stored outline. */
function OutlineThumb({ value }: { value: string }) {
  const pts = decodeOutline(value);
  const c = usePalette();
  return (
    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg border border-line bg-subtle">
      {pts.length >= 3 ? (
        <svg viewBox="-0.08 -0.08 1.16 1.16" className="h-9 w-9">
          <polygon points={pts.map((p) => `${p.x},${p.y}`).join(' ')} fill={c.partSelected} stroke={c.accent} strokeWidth={0.04} strokeLinejoin="round" />
        </svg>
      ) : (
        <PenLine size={15} className="text-faint" />
      )}
    </div>
  );
}

/** One-line note under a field, optionally with a one-click fix. */
function Hint({ children, tone = 'info', action }: { children: ReactNode; tone?: 'info' | 'warn'; action?: { label: string; onClick: () => void } }) {
  return (
    <div
      className={cx(
        'flex items-start gap-1.5 rounded-md px-2 py-1.5 text-[11px] leading-relaxed',
        tone === 'warn' ? 'bg-warn-soft text-warn' : 'bg-subtle text-muted',
      )}
    >
      <Info size={12} className="mt-[1px] shrink-0 opacity-70" />
      <span className="flex-1">{children}</span>
      {action && (
        <button type="button" onClick={action.onClick} className="shrink-0 font-medium whitespace-nowrap underline underline-offset-2 hover:no-underline">
          {action.label}
        </button>
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
  const { print } = design;
  const bv = design.printer.build;
  const { parts, byCompartment } = useGeometry();
  const usable = usableArea(design.drawer);
  const cov = coverage(design.compartments, { width: usable.width, depth: usable.depth });
  const printer = design.printer;
  const preset = findPrinter(printer.presetId);
  const grams = useMemo(() => parts.reduce((s, p) => s + filamentGrams(p, printer.infill), 0), [parts, printer.infill]);
  const oversized = parts.filter((p) => !fitsBuildVolume(p.size, bv));
  const wall = wallAdvice(print.wallThickness, printer.nozzle);
  const floorLayers = Math.round(print.bottomThickness / printer.layerHeight);
  const floorAwkward = Math.abs(print.bottomThickness / printer.layerHeight - floorLayers) > 0.1;
  const tooTall = print.binHeight > design.drawer.height;
  const errors = [...byCompartment.values()].filter((g) => g.error);
  const t = useT();

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-line px-4 py-3">
        <div className="text-[14px] font-semibold text-ink">{t('Organiser')}</div>
        <div className="mt-0.5 text-[12px] text-muted">{t('Global settings · select a compartment to edit it')}</div>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <Section title={t('Walls & floor')}>
          <div className="grid grid-cols-2 gap-2">
            <NumberField
              label={t('Wall thickness')}
              value={print.wallThickness}
              step={0.2}
              min={LIMITS.wall.min}
              max={LIMITS.wall.max}
              onChange={(wallThickness) => actions.updatePrint({ wallThickness })}
              hint={t('A whole number of {line} mm lines prints best with your {nozzle} mm nozzle', {
                line: lineWidthFor(printer.nozzle),
                nozzle: printer.nozzle,
              })}
            />
            <NumberField label={t('Bottom thickness')} value={print.bottomThickness} step={0.2} min={LIMITS.bottom.min} max={LIMITS.bottom.max} onChange={(bottomThickness) => actions.updatePrint({ bottomThickness })} />
            <NumberField label={t('Bin height')} value={print.binHeight} min={LIMITS.binHeight.min} max={LIMITS.binHeight.max} onChange={(binHeight) => actions.updatePrint({ binHeight })} />
            <NumberField label={t('Fit gap')} value={print.fitGap} step={0.1} min={LIMITS.fitGap.min} max={LIMITS.fitGap.max} onChange={(fitGap) => actions.updatePrint({ fitGap })} hint={t('Total gap between neighbouring bins so they drop in without force')} />
          </div>
          <div className="mt-2 space-y-1">
            {tooTall && <Hint tone="warn">{t('Bins are taller than the drawer ({h} mm).', { h: design.drawer.height })}</Hint>}
            {wall.tooThin && (
              <Hint
                tone="warn"
                action={{ label: t('Use {v} mm', { v: wall.suggested }), onClick: () => actions.updatePrint({ wallThickness: wall.suggested }) }}
              >
                {t('Thinner than one {line} mm line – your slicer cannot print this wall.', { line: lineWidthFor(printer.nozzle) })}
              </Hint>
            )}
            {!wall.tooThin && wall.awkward && (
              <Hint action={{ label: t('Use {v} mm', { v: wall.suggested }), onClick: () => actions.updatePrint({ wallThickness: wall.suggested }) }}>
                {wall.perimeters === 1
                  ? t('Not a whole number of lines; {v} mm is 1 clean perimeter.', { v: wall.suggested })
                  : t('Not a whole number of lines; {v} mm is {n} clean perimeters.', { v: wall.suggested, n: wall.perimeters })}
              </Hint>
            )}
            {floorAwkward && (
              <Hint
                action={{
                  label: t('Use {v} mm', { v: Math.round(floorLayers * printer.layerHeight * 100) / 100 }),
                  onClick: () => actions.updatePrint({ bottomThickness: Math.round(floorLayers * printer.layerHeight * 100) / 100 }),
                }}
              >
                {t('Floor is not a whole number of {h} mm layers.', { h: printer.layerHeight })}
              </Hint>
            )}
          </div>
        </Section>

        <Section title={t('Corners')}>
          <Segmented
            className="w-full"
            value={print.cornerStyle}
            onChange={(cornerStyle) => actions.updatePrint({ cornerStyle })}
            options={[
              { value: 'rounded', label: t('Rounded') },
              { value: 'square', label: t('Square') },
            ]}
          />
          {print.cornerStyle === 'rounded' && (
            <div className="mt-2">
              <NumberField label={t('Corner radius')} value={print.cornerRadius} step={0.5} min={0.5} max={LIMITS.radius.max} onChange={(cornerRadius) => actions.updatePrint({ cornerRadius })} />
            </div>
          )}
        </Section>

        <Section title={t('Dividers')}>
          <div className="grid grid-cols-2 gap-2">
            <NumberField label={t('Thickness')} value={print.dividerThickness} step={0.2} min={LIMITS.dividerThickness.min} max={LIMITS.dividerThickness.max} onChange={(dividerThickness) => actions.updatePrint({ dividerThickness })} />
            <NumberField
              label={t('Height')}
              value={print.dividerHeight ?? print.binHeight}
              disabled={print.dividerHeight === null}
              min={print.bottomThickness + 1}
              max={LIMITS.binHeight.max}
              onChange={(dividerHeight) => actions.updatePrint({ dividerHeight })}
            />
          </div>
          <div className="mt-2">
            <Toggle label={t('Full-height dividers')} checked={print.dividerHeight === null} onChange={(on) => actions.updatePrint({ dividerHeight: on ? null : Math.round(print.binHeight * 0.7) })} />
          </div>
        </Section>

        <Section title={t('Printer')}>
          <label className="block">
            <span className="mb-1 block text-[11px] font-medium text-muted">{t('Machine')}</span>
            <select
              value={printer.presetId}
              onChange={(e) => actions.selectPrinter(e.target.value)}
              className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-[13px] outline-none focus:border-accent"
            >
              {printersByBrand().map((g) => (
                <optgroup key={g.brand} label={g.brand}>
                  {g.items.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.model} · {m.build.x}×{m.build.y}×{m.build.z}
                    </option>
                  ))}
                </optgroup>
              ))}
              <option value="custom">{t('Custom printer')}</option>
            </select>
          </label>
          {preset?.note && <p className="mt-1.5 text-[11px] leading-relaxed text-muted">{t(preset.note)}</p>}

          <div className="mt-3 mb-1.5 text-[11px] font-medium text-muted">{t('Build volume')}</div>
          <div className="grid grid-cols-3 gap-1.5">
            <NumberField compact value={bv.x} min={LIMITS.build.min} max={LIMITS.build.max} onChange={(x) => actions.updatePrinter({ build: { ...bv, x } })} unit="X" />
            <NumberField compact value={bv.y} min={LIMITS.build.min} max={LIMITS.build.max} onChange={(y) => actions.updatePrinter({ build: { ...bv, y } })} unit="Y" />
            <NumberField compact value={bv.z} min={LIMITS.build.min} max={LIMITS.build.max} onChange={(z) => actions.updatePrinter({ build: { ...bv, z } })} unit="Z" />
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <NumberField label={t('Nozzle')} value={printer.nozzle} min={LIMITS.nozzle.min} max={LIMITS.nozzle.max} step={0.1} onChange={(nozzle) => actions.updatePrinter({ nozzle })} />
            <NumberField
              label={t('Layer height')}
              value={printer.layerHeight}
              min={LIMITS.layer.min}
              max={Math.min(LIMITS.layer.max, maxLayerHeight(printer.nozzle))}
              step={0.02}
              onChange={(layerHeight) => actions.updatePrinter({ layerHeight })}
              hint={t('Up to {max} mm with a {nozzle} mm nozzle', { max: maxLayerHeight(printer.nozzle), nozzle: printer.nozzle })}
            />
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-faint">
            {t('Saved with the design, so the file carries the machine it was drawn for.')}
          </p>
        </Section>

        <Section title={t('Summary')}>
          <Stat label={t('Compartments')} value={design.compartments.length} />
          <Stat label={t('Printable parts')} value={parts.length} />
          <Stat label={t('Area used')} value={`${Math.round(cov * 100)}%`} />
          <Stat label={t('Est. filament (PLA)')} value={`${Math.round(grams)} g`} />
          <Stat label={t('Oversized parts')} value={oversized.length} tone={oversized.length ? 'warn' : 'ok'} />
          {(oversized.length > 0 || errors.length > 0) && (
            <div className="mt-2 space-y-1">
              {oversized.slice(0, 5).map((p) => (
                <button key={p.id} type="button" onClick={() => useUi.getState().select(p.compartmentId)} className="flex w-full items-center gap-1.5 rounded-md bg-warn-soft px-2 py-1 text-left text-[11px] text-warn">
                  <AlertTriangle size={12} /> {t('{name} exceeds the build volume', { name: p.name })}
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
