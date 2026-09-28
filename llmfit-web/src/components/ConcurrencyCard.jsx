import { useCallback, useEffect, useState } from 'react';
import { fetchConcurrency } from '../api';
import { useI18n } from '../contexts/I18nContext';
import { useModelContext } from '../contexts/ModelContext';
import { round } from '../utils';

export const KV_QUANT_OPTIONS = ['fp16', 'fp8', 'q8_0', 'q4_0', 'tq'];

function formatContextShort(tokens) {
  if (typeof tokens !== 'number') return '—';
  return tokens % 1024 === 0 ? `${tokens / 1024}k` : String(tokens);
}

function formatPerSession(gb) {
  if (typeof gb !== 'number') return '—';
  return gb >= 0.01 ? `${gb.toFixed(2)} GB` : `${(gb * 1024).toFixed(1)} MiB`;
}

// Rungs above the native window all clamp to it and repeat the same row;
// show each distinct (context, sessions) pair once, as the CLI does.
function distinctRungs(ladder) {
  const rows = [];
  let last = null;
  for (const slot of ladder ?? []) {
    const key = `${slot.effective_context}:${slot.max_sessions}`;
    if (key === last) continue;
    last = key;
    rows.push(slot);
  }
  return rows;
}

export default function ConcurrencyCard({ model }) {
  const { t } = useI18n();
  const { appliedSimulation } = useModelContext();
  const [kvQuant, setKvQuant] = useState('fp16');
  const [users, setUsers] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const load = useCallback(
    async (signal) => {
      setLoading(true);
      setError('');
      try {
        const payload = await fetchConcurrency(
          { model: model.name, kv_quant: kvQuant, users },
          appliedSimulation,
          signal
        );
        setResult(payload);
      } catch (err) {
        if (err?.name === 'AbortError') return;
        setResult(null);
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    // `users` is applied on submit, not on every keystroke.
    [model.name, kvQuant, appliedSimulation]
  );

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  function handleSubmit(event) {
    event.preventDefault();
    load();
  }

  const estimate = result?.estimate;
  const targetUsers = result?.target_users;

  return (
    <div className="metrics-card">
      <h4>{t('concurrency.title')}</h4>
      <p className="muted-copy">{t('concurrency.hint')}</p>

      <form className="plan-form" onSubmit={handleSubmit}>
        <div className="plan-grid">
          <label>
            <span>{t('concurrency.fields.kvQuant')}</span>
            <select value={kvQuant} onChange={(event) => setKvQuant(event.target.value)}>
              {KV_QUANT_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>{t('concurrency.fields.users')}</span>
            <input
              type="number"
              min="1"
              step="1"
              value={users}
              onChange={(event) => setUsers(event.target.value)}
              placeholder={t('concurrency.placeholders.users')}
            />
          </label>
        </div>
        <div className="plan-actions">
          <button type="submit" className="btn btn-accent btn-sm" disabled={loading}>
            {loading ? t('plan.actions.loading') : t('concurrency.actions.estimate')}
          </button>
        </div>
      </form>

      {error ? (
        <div role="alert" className="alert error">
          {t('concurrency.error', { error })}
        </div>
      ) : null}

      {estimate ? (
        <>
          <p className="muted-copy">
            {t('concurrency.summary', {
              pool: round(estimate.pool_gb, 1),
              weights: round(estimate.weights_resident_gb, 1),
              kv: round(estimate.kv_budget_gb, 1),
              quant: estimate.quant,
              native: formatContextShort(estimate.native_context)
            })}
          </p>
          <div className="table-wrap plan-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{t('concurrency.table.context')}</th>
                  <th>{t('concurrency.table.perSession')}</th>
                  <th>{t('concurrency.table.sessions')}</th>
                </tr>
              </thead>
              <tbody>
                {distinctRungs(estimate.ladder).map((slot) => (
                  <tr key={slot.requested_context}>
                    <td>
                      {formatContextShort(slot.effective_context)}
                      {slot.clamped ? (
                        <span className="muted-copy"> ({t('concurrency.clamped')})</span>
                      ) : null}
                    </td>
                    <td>{formatPerSession(slot.per_session_kv_gb)}</td>
                    <td>{slot.max_sessions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {typeof targetUsers === 'number' ? (
            <p className="plan-notice">
              {typeof result.max_context_for_target === 'number'
                ? t('concurrency.targetFits', {
                    users: targetUsers,
                    context: formatContextShort(result.max_context_for_target)
                  })
                : t('concurrency.targetMisses', { users: targetUsers })}
            </p>
          ) : null}
          <p className="muted-copy">{t('concurrency.ceilingNote')}</p>
        </>
      ) : null}
    </div>
  );
}
