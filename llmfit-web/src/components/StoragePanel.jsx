import { useState } from 'react';
import { fetchStorage } from '../api';
import { useI18n } from '../contexts/I18nContext';
import { useModelContext } from '../contexts/ModelContext';
import { round, fitClass, normalizeFitCode, translateFitLevel } from '../utils';

const DEFAULT_FORM = {
  keep: '3',
  selection: 'score',
  osReserve: '100G',
  scratch: 'auto',
  headroom: '15',
  perfect: false,
  search: ''
};

function SummaryCard({ label, value }) {
  return (
    <div className="plan-summary-card">
      <h5>{label}</h5>
      <p className="storage-value">{value}</p>
    </div>
  );
}

function formatSsd(gb) {
  if (typeof gb !== 'number') return '—';
  return gb >= 1000 ? `${round(gb / 1000, gb % 1000 === 0 ? 0 : 1)} TB` : `${gb} GB`;
}

export default function StoragePanel() {
  const { t } = useI18n();
  const { appliedSimulation } = useModelContext();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  function update(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      const payload = await fetchStorage(
        {
          keep: form.keep,
          selection: form.selection,
          os_reserve: form.osReserve,
          scratch: form.scratch,
          headroom: form.headroom,
          perfect: form.perfect,
          search: form.search
        },
        appliedSimulation
      );
      setResult(payload.storage ?? null);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel storage-panel">
      <div className="panel-heading">
        <h2>{t('storage.title')}</h2>
        <div className="panel-heading-actions">
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            aria-expanded={open}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? t('storage.actions.hide') : t('storage.actions.show')}
          </button>
        </div>
      </div>

      {open ? (
        <>
          <p className="muted-copy">{t('storage.hint')}</p>
          <form className="plan-form" onSubmit={handleSubmit}>
            <div className="plan-grid storage-grid">
              <label>
                <span>{t('storage.fields.keep')}</span>
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={form.keep}
                  onChange={(event) => update('keep', event.target.value)}
                />
              </label>
              <label>
                <span>{t('storage.fields.selection')}</span>
                <select
                  value={form.selection}
                  onChange={(event) => update('selection', event.target.value)}
                >
                  <option value="score">{t('storage.selection.score')}</option>
                  <option value="largest">{t('storage.selection.largest')}</option>
                </select>
              </label>
              <label>
                <span>{t('storage.fields.osReserve')}</span>
                <input
                  type="text"
                  value={form.osReserve}
                  onChange={(event) => update('osReserve', event.target.value)}
                  placeholder="100G"
                />
              </label>
              <label>
                <span>{t('storage.fields.scratch')}</span>
                <input
                  type="text"
                  value={form.scratch}
                  onChange={(event) => update('scratch', event.target.value)}
                  placeholder="auto"
                />
              </label>
              <label>
                <span>{t('storage.fields.headroom')}</span>
                <input
                  type="number"
                  min="0"
                  max="99"
                  step="1"
                  value={form.headroom}
                  onChange={(event) => update('headroom', event.target.value)}
                />
              </label>
              <label>
                <span>{t('storage.fields.search')}</span>
                <input
                  type="text"
                  value={form.search}
                  onChange={(event) => update('search', event.target.value)}
                  placeholder={t('storage.placeholders.search')}
                />
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={form.perfect}
                  onChange={(event) => update('perfect', event.target.checked)}
                />
                <span>{t('storage.fields.perfect')}</span>
              </label>
            </div>
            <div className="plan-actions">
              <button type="submit" className="btn btn-accent btn-sm" disabled={loading}>
                {loading ? t('plan.actions.loading') : t('storage.actions.estimate')}
              </button>
            </div>
          </form>

          {error ? (
            <div role="alert" className="alert error">
              {t('storage.error', { error })}
            </div>
          ) : null}

          {result ? (
            <div className="plan-results">
              <p className="plan-notice">{result.estimate_notice}</p>
              <div className="plan-summary-grid">
                <SummaryCard
                  label={t('storage.summary.library')}
                  value={`${round(result.library_gb, 1)} GB`}
                />
                <SummaryCard
                  label={t('storage.summary.scratch')}
                  value={`${round(result.download_scratch_gb, 1)} GB`}
                />
                <SummaryCard
                  label={t('storage.summary.need')}
                  value={`${round(result.need_gb, 1)} GB`}
                />
                <SummaryCard
                  label={t('storage.summary.minimumSsd')}
                  value={formatSsd(result.minimum_ssd_gb)}
                />
                <SummaryCard
                  label={t('storage.summary.suggestedSsd')}
                  value={formatSsd(result.suggested_ssd_gb)}
                />
              </div>

              <p className="muted-copy">
                {t('storage.selected', {
                  selected: result.selected_count,
                  requested: result.keep_requested,
                  eligible: result.eligible_count
                })}
              </p>

              {Array.isArray(result.models) && result.models.length > 0 ? (
                <div className="table-wrap plan-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>{t('table.columns.model')}</th>
                        <th>{t('detail.fields.bestQuant')}</th>
                        <th>{t('table.columns.fit')}</th>
                        <th>{t('table.columns.score')}</th>
                        <th>{t('storage.table.disk')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.models.map((model) => (
                        <tr key={model.name}>
                          <td className="model-name">{model.name}</td>
                          <td>{model.best_quant}</td>
                          <td>
                            <span className={fitClass(normalizeFitCode(model.fit_level))}>
                              {translateFitLevel(t, model.fit_level, model.fit_level)}
                            </span>
                          </td>
                          <td>{round(model.score, 1)}</td>
                          <td>{round(model.disk_size_gb, 1)} GB</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}

              {Array.isArray(result.warnings) && result.warnings.length > 0 ? (
                <ul className="plan-list">
                  {result.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
