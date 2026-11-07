import { ChevronDown } from 'lucide-react';
import { type ButtonHTMLAttributes, type ReactNode, useEffect, useId, useRef, useState } from 'react';

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(' ');
}

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';

export function Button({
  variant = 'secondary',
  size = 'md',
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' }) {
  return (
    <button
      type="button"
      {...rest}
      className={cx(
        'inline-flex items-center justify-center gap-1.5 rounded-lg font-medium whitespace-nowrap select-none transition-all duration-150',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 disabled:opacity-40 disabled:pointer-events-none active:scale-[0.98]',
        size === 'sm' ? 'h-7 px-2.5 text-[12px]' : 'h-8 px-3 text-[13px]',
        variant === 'primary' && 'bg-ink text-white shadow-sm hover:bg-ink-2',
        variant === 'secondary' && 'bg-surface text-ink border border-line shadow-[0_1px_1px_rgb(0_0_0/0.03)] hover:border-line-strong hover:bg-subtle/60',
        variant === 'ghost' && 'text-ink-2 hover:bg-subtle hover:text-ink',
        variant === 'danger' && 'text-danger hover:bg-red-50',
        className,
      )}
    >
      {children}
    </button>
  );
}

export function IconButton({
  label,
  active,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; active?: boolean }) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      {...rest}
      className={cx(
        'inline-flex h-8 w-8 items-center justify-center rounded-lg text-ink-2 transition-colors duration-150',
        'hover:bg-subtle hover:text-ink disabled:opacity-35 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40',
        active && 'bg-subtle text-ink',
        className,
      )}
    >
      {children}
    </button>
  );
}

/**
 * Numeric input that keeps a local draft while typing and commits on blur / Enter.
 * Arrow keys step (Shift = ×10). Values are clamped to [min, max].
 */
export function NumberField({
  label,
  value,
  onChange,
  min = -Infinity,
  max = Infinity,
  step = 1,
  unit = 'mm',
  disabled,
  hint,
  compact,
}: {
  label?: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  disabled?: boolean;
  hint?: string;
  compact?: boolean;
}) {
  const id = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const ref = useRef<HTMLInputElement>(null);
  const shown = draft ?? format(value);

  const commit = (raw: string) => {
    setDraft(null);
    const v = Number(raw.replace(',', '.'));
    if (!Number.isFinite(v)) return;
    const clamped = Math.min(max, Math.max(min, v));
    if (clamped !== value) onChange(Math.round(clamped * 100) / 100);
  };

  return (
    <label htmlFor={id} className={cx('group block', disabled && 'opacity-50')} title={hint}>
      {label && <span className="mb-1 block text-[11px] font-medium text-muted">{label}</span>}
      <div
        className={cx(
          'flex items-center rounded-lg border border-line bg-surface transition-colors',
          'focus-within:border-accent focus-within:ring-3 focus-within:ring-accent/12 hover:border-line-strong',
          compact ? 'h-7' : 'h-8',
        )}
      >
        <input
          ref={ref}
          id={id}
          type="text"
          inputMode="decimal"
          disabled={disabled}
          value={shown}
          onChange={(e) => setDraft(e.target.value)}
          onFocus={(e) => e.target.select()}
          onBlur={(e) => commit(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              commit((e.target as HTMLInputElement).value);
              ref.current?.blur();
            } else if (e.key === 'Escape') {
              setDraft(null);
              ref.current?.blur();
            } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
              e.preventDefault();
              const base = Number((draft ?? String(value)).replace(',', '.'));
              const delta = (e.key === 'ArrowUp' ? 1 : -1) * step * (e.shiftKey ? 10 : 1);
              const next = Math.min(max, Math.max(min, (Number.isFinite(base) ? base : value) + delta));
              setDraft(null);
              onChange(Math.round(next * 100) / 100);
            }
          }}
          className="num w-full min-w-0 bg-transparent px-2.5 text-[13px] text-ink outline-none"
        />
        {unit && <span className="pr-2.5 text-[11px] text-faint select-none">{unit}</span>}
      </div>
    </label>
  );
}

function format(v: number) {
  return String(Math.round(v * 100) / 100);
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  size = 'md',
  className,
}: {
  value: T;
  options: { value: T; label: ReactNode; title?: string }[];
  onChange: (v: T) => void;
  size?: 'sm' | 'md';
  className?: string;
}) {
  return (
    <div className={cx('inline-flex rounded-lg bg-subtle p-0.5', className)}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          title={o.title}
          onClick={() => onChange(o.value)}
          className={cx(
            'flex flex-1 items-center justify-center gap-1.5 rounded-md font-medium whitespace-nowrap transition-all duration-150',
            size === 'sm' ? 'h-6 px-2 text-[11px]' : 'h-7 px-3 text-[12px]',
            o.value === value ? 'bg-surface text-ink shadow-[0_1px_2px_rgb(0_0_0/0.08)]' : 'text-muted hover:text-ink',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between gap-3 py-1 text-[13px] text-ink-2"
    >
      <span>{label}</span>
      <span className={cx('relative h-[18px] w-8 shrink-0 rounded-full transition-colors duration-200', checked ? 'bg-accent' : 'bg-line-strong')}>
        <span
          className={cx(
            'absolute top-[2px] left-0 h-[14px] w-[14px] rounded-full bg-white shadow-sm transition-transform duration-200',
            checked ? 'translate-x-[16px]' : 'translate-x-[2px]',
          )}
        />
      </span>
    </button>
  );
}

export function Section({
  title,
  children,
  action,
  defaultOpen = true,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border-b border-line px-4 py-3.5 last:border-b-0">
      <div className="flex items-center justify-between">
        <button type="button" onClick={() => setOpen(!open)} className="flex items-center gap-1 text-[11px] font-semibold tracking-wide text-muted uppercase hover:text-ink">
          <ChevronDown size={12} className={cx('transition-transform duration-200', !open && '-rotate-90')} />
          {title}
        </button>
        {action}
      </div>
      {open && <div className="mt-3 animate-fade-in">{children}</div>}
    </section>
  );
}

export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: 'ok' | 'warn' }) {
  return (
    <div className="flex items-center justify-between py-1 text-[12px]">
      <span className="text-muted">{label}</span>
      <span className={cx('num font-medium', tone === 'warn' ? 'text-warn' : tone === 'ok' ? 'text-ok' : 'text-ink')}>{value}</span>
    </div>
  );
}

export function Badge({ tone = 'neutral', children }: { tone?: 'neutral' | 'ok' | 'warn' | 'accent'; children: ReactNode }) {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10.5px] font-medium',
        tone === 'neutral' && 'bg-subtle text-muted',
        tone === 'ok' && 'bg-ok-soft text-ok',
        tone === 'warn' && 'bg-warn-soft text-warn',
        tone === 'accent' && 'bg-accent-soft text-accent-ink',
      )}
    >
      {children}
    </span>
  );
}

export function Modal({ open, onClose, children, width = 'max-w-2xl' }: { open: boolean; onClose?: () => void; children: ReactNode; width?: string }) {
  useEffect(() => {
    if (!open || !onClose) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center p-0 sm:items-center sm:p-6">
      <div className="absolute inset-0 animate-fade-in bg-ink/25 backdrop-blur-[2px]" onClick={onClose} />
      <div className={cx('relative max-h-[92vh] w-full animate-pop-in overflow-hidden rounded-t-2xl bg-surface shadow-pop sm:rounded-2xl', width)}>{children}</div>
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-line bg-subtle px-1 font-mono text-[10px] text-muted">{children}</kbd>;
}
