import { useEffect, useMemo, useState } from 'react';
import { fetchPlanEstimate } from '../api';
import { useI18n } from '../contexts/I18nContext';
import { useModelContext } from '../contexts/ModelContext';
import { round, fitClass, translateFitLevel, translateRunMode } from '../utils';

function MetricBar({ label, value }) {
  const safe = Number.isFinite(value) ? Math.max(0, Math.min(value, 100)) : 0;
  return (
    <div className="metric-row">
      <div className="metric-text">
        <span>{label}</span>
        <span>{round(value, 1)}</span>
      </div>
      <div className="metric-track">
        <div className="metric-fill" style={{ width: `${safe}%` }} />
      </div>
    </div>
  );
}

function HardwareEstimateCard({ title, estimate, t }) {
  return (
    <div className="plan-summary-card">
      <h5>{title}</h5>
      <dl className="plan-summary-list">
        <div>
          <dt>{t('plan.summary.vram')}</dt>
          <dd>
            {typeof estimate?.vram_gb === 'number'
              ? `${round(estimate.vram_gb, 1)} GB`
              : t('plan.summary.notRequired')}
          </dd>
        </div>
        <div>
          <dt>{t('plan.summary.ram')}</dt>
          <dd>{typeof estimate?.ram_gb === 'number' ? `${round(estimate.ram_gb, 1)} GB` : '\u2014'}</dd>
        </div>
        <div>
          <dt>{t('plan.summary.cpuCores')}</dt>
          <dd>{typeof estimate?.cpu_cores === 'number' ? estimate.cpu_cores : '\u2014'}</dd>
        </div>
      </dl>
    </div>
  );
}

function translatePlanPath(t, path) {
  return t(`plan.paths.${path}`);
}

export default function DetailPanel() {
  const { t } = useI18n();
  const { models, selectedModelName, appliedSimulation, simulationActive } = useModelContext();
  const [planForm, setPlanForm] = useState({
    context: '',
    quant: '',
    kvQuant: '',
    targetTps: ''
  });
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState('');
  const [planResult, setPlanResult] = useState(null);

  const selectedModel = useMemo(
    () => models.find((m) => m.name === selectedModelName) ?? null,
    [models, selectedModelName]
  );

  useEffect(() => {
    if (!selectedModel) {
      setPlanForm({
        context: '',
        quant: '',
        kvQuant: '',
        targetTps: ''
      });
      setPlanResult(null);
      setPlanError('');
      setPlanLoading(false);
      return;
    }

    setPlanForm({
      context: String(
        Math.max(
          1,
          Math.min(
            Number.isFinite(selectedModel.context_length)
              ? selectedModel.context_length
              : 4096,
            8192
          )
        )
      ),
      quant: selectedModel.best_quant ?? '',
      kvQuant: '',
      targetTps: ''
    });
    setPlanResult(null);
    setPlanError('');
    setPlanLoading(false);
  }, [selectedModel]);

  if (!selectedModel) {
    return (
      <aside className="details-panel">
        <p className="muted-copy">
          {t('detail.selectPrompt')}
        </p>
      </aside>
    );
  }

  const ggufSources = Array.isArray(selectedModel.gguf_sources)
    ? selectedModel.gguf_sources
    : [];
  const capabilities = Array.isArray(selectedModel.capabilities)
    ? selectedModel.capabilities
    : [];
  const license = selectedModel.license || null;
  const isMoe = selectedModel.is_moe === true;
  const moeOffloadedGb = selectedModel.moe_offloaded_gb;

  function updatePlanField(field, value) {
    setPlanForm((current) => ({
      ...current,
      [field]: value
    }));
  }

  async function handlePlanSubmit(event) {
    event.preventDefault();

    const context = Number.parseInt(planForm.context, 10);
    if (!Number.isFinite(context) || context <= 0) {
      setPlanError(t('plan.validation.context'));
      setPlanResult(null);
      return;
    }

    let targetTps;
    if (planForm.targetTps.trim()) {
      targetTps = Number.parseFloat(planForm.targetTps);
      if (!Number.isFinite(targetTps) || targetTps <= 0) {
        setPlanError(t('plan.validation.targetTps'));
        setPlanResult(null);
        return;
      }
    }

    setPlanLoading(true);
    setPlanError('');
    try {
      const payload = await fetchPlanEstimate(
        {
          model: selectedModel.name,
          context,
          quant: planForm.quant.trim() || undefined,
          kv_quant: planForm.kvQuant.trim() || undefined,
          target_tps: targetTps
        },
        appliedSimulation
      );
      setPlanResult(payload);
    } catch (error) {
      setPlanResult(null);
      setPlanError(
        error instanceof Error ? error.message : t('plan.errorFallback')
      );
    } finally {
      setPlanLoading(false);
    }
  }

  return (
    <aside className="details-panel">
      <div className="details-header">
        <h3>{selectedModel.name}</h3>
        <span className={fitClass(selectedModel.fit_level)}>
          {translateFitLevel(t, selectedModel.fit_level, selectedModel.fit_label)}
        </span>
      </div>

      <dl className="details-grid">
        <div>
          <dt>{t('detail.fields.provider')}</dt>
          <dd>{selectedModel.provider}</dd>
        </div>
        <div>
          <dt>{t('detail.fields.runMode')}</dt>
          <dd>{translateRunMode(t, selectedModel.run_mode, selectedModel.run_mode_label)}</dd>
        </div>
        <div>
          <dt>{t('detail.fields.runtime')}</dt>
          <dd>{selectedModel.runtime_label}</dd>
        </div>
        <div>
          <dt>{t('detail.fields.bestQuant')}</dt>
          <dd>{selectedModel.best_quant}</dd>
        </div>
        <div>
          <dt>{t('detail.fields.memoryRequired')}</dt>
          <dd>{round(selectedModel.memory_required_gb, 2)} GB</dd>
        </div>
        <div>
          <dt>{t('detail.fields.memoryAvailable')}</dt>
          <dd>{round(selectedModel.memory_available_gb, 2)} GB</dd>
        </div>
        {license && (
          <div>
            <dt>{t('detail.fields.license')}</dt>
            <dd>{license}</dd>
          </div>
        )}
        {isMoe && (
          <div>
            <dt>{t('detail.fields.moeOffloaded')}</dt>
            <dd>
              {typeof moeOffloadedGb === 'number'
                ? `${round(moeOffloadedGb, 2)} GB`
                : t('detail.noMoeValue')}
            </dd>
          </div>
        )}
      </dl>

      <div className="metrics-card planning-card">
        <div className="planning-header">
          <div>
            <h4>{t('plan.title')}</h4>
            <p className="muted-copy">
              {simulationActive ? t('plan.simulatedHint') : t('plan.defaultHint')}
            </p>
          </div>
          {simulationActive ? (
            <span className="chip chip-accent">{t('plan.simulatedBadge')}</span>
          ) : null}
        </div>

        <form className="plan-form" onSubmit={handlePlanSubmit}>
          <div className="plan-grid">
            <label>
              <span>{t('plan.fields.context')}</span>
              <input
                type="number"
                min="1"
                step="1"
                value={planForm.context}
                onChange={(event) => updatePlanField('context', event.target.value)}
                placeholder={t('plan.placeholders.context')}
              />
            </label>

            <label>
              <span>{t('plan.fields.quant')}</span>
              <input
                type="text"
                value={planForm.quant}
                onChange={(event) => updatePlanField('quant', event.target.value)}
                placeholder={t('plan.placeholders.quant')}
              />
            </label>

            <label>
              <span>{t('plan.fields.kvQuant')}</span>
              <input
                type="text"
                value={planForm.kvQuant}
                onChange={(event) => updatePlanField('kvQuant', event.target.value)}
                placeholder={t('plan.placeholders.kvQuant')}
              />
            </label>

            <label>
              <span>{t('plan.fields.targetTps')}</span>
              <input
                type="number"
                min="0"
                step="0.1"
                value={planForm.targetTps}
                onChange={(event) => updatePlanField('targetTps', event.target.value)}
                placeholder={t('plan.placeholders.targetTps')}
              />
            </label>
          </div>

          <div className="plan-actions">
            <button type="submit" className="btn btn-accent btn-sm" disabled={planLoading}>
              {planLoading ? t('plan.actions.loading') : t('plan.actions.estimate')}
            </button>
          </div>
        </form>

        {planError ? (
          <div role="alert" className="alert error">
            {t('plan.error', { error: planError })}
          </div>
        ) : null}

        {planResult ? (
          <div className="plan-results">
            <p className="plan-notice">{planResult.estimate_notice}</p>

            <div className="plan-summary-grid">
              <div className="plan-summary-card">
                <h5>{t('plan.summary.current')}</h5>
                <dl className="plan-summary-list">
                  <div>
                    <dt>{t('plan.summary.fitLevel')}</dt>
                    <dd>
                      {translateFitLevel(
                        t,
                        planResult.current?.fit_level,
                        planResult.current?.fit_level
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>{t('plan.summary.runMode')}</dt>
                    <dd>
                      {translateRunMode(
                        t,
                        planResult.current?.run_mode,
                        planResult.current?.run_mode
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>{t('plan.summary.estimatedTps')}</dt>
                    <dd>{round(planResult.current?.estimated_tps, 1)}</dd>
                  </div>
                  <div>
                    <dt>{t('plan.summary.kvQuant')}</dt>
                    <dd>{planResult.kv_quant?.label ?? planResult.kv_quant ?? '\u2014'}</dd>
                  </div>
                </dl>
              </div>

              <HardwareEstimateCard
                title={t('plan.summary.minimum')}
                estimate={planResult.minimum}
                t={t}
              />
              <HardwareEstimateCard
                title={t('plan.summary.recommended')}
                estimate={planResult.recommended}
                t={t}
              />
            </div>

            <div className="plan-section">
              <h5>{t('plan.sections.paths')}</h5>
              <div className="plan-runpaths">
                {(planResult.run_paths ?? []).map((path) => (
                  <article key={path.path} className="plan-path-card">
                    <div className="plan-path-header">
                      <strong>{translatePlanPath(t, path.path)}</strong>
                      <span className={`fit ${path.feasible ? 'fit-good' : 'fit-too_tight'}`}>
                        {path.feasible ? t('plan.pathsFeasible.yes') : t('plan.pathsFeasible.no')}
                      </span>
                    </div>
                    <dl className="plan-summary-list">
                      <div>
                        <dt>{t('plan.summary.fitLevel')}</dt>
                        <dd>
                          {path.fit_level
                            ? translateFitLevel(t, path.fit_level, path.fit_level)
                            : '\u2014'}
                        </dd>
                      </div>
                      <div>
                        <dt>{t('plan.summary.estimatedTps')}</dt>
                        <dd>
                          {typeof path.estimated_tps === 'number'
                            ? round(path.estimated_tps, 1)
                            : '\u2014'}
                        </dd>
                      </div>
                    </dl>
                    <div className="plan-path-estimates">
                      <HardwareEstimateCard
                        title={t('plan.summary.minimum')}
                        estimate={path.minimum}
                        t={t}
                      />
                      <HardwareEstimateCard
                        title={t('plan.summary.recommended')}
                        estimate={path.recommended}
                        t={t}
                      />
                    </div>
                    {Array.isArray(path.notes) && path.notes.length > 0 ? (
                      <ul className="plan-list">
                        {path.notes.map((note, index) => (
                          <li key={`${path.path}-${index}`}>{note}</li>
                        ))}
                      </ul>
                    ) : null}
                  </article>
                ))}
              </div>
            </div>

            <div className="plan-section">
              <h5>{t('plan.sections.upgrades')}</h5>
              {Array.isArray(planResult.upgrade_deltas) && planResult.upgrade_deltas.length > 0 ? (
                <ul className="plan-list">
                  {planResult.upgrade_deltas.map((item, index) => (
                    <li key={`${item.resource}-${index}`}>{item.description}</li>
                  ))}
                </ul>
              ) : (
                <p className="muted-copy">{t('plan.noUpgrades')}</p>
              )}
            </div>

            {Array.isArray(planResult.kv_alternatives) && planResult.kv_alternatives.length > 0 ? (
              <div className="plan-section">
                <h5>{t('plan.sections.kvAlternatives')}</h5>
                <div className="table-wrap plan-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>{t('plan.kvTable.quant')}</th>
                        <th>{t('plan.kvTable.memory')}</th>
                        <th>{t('plan.kvTable.kvCache')}</th>
                        <th>{t('plan.kvTable.savings')}</th>
                        <th>{t('plan.kvTable.supported')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {planResult.kv_alternatives.map((alt) => (
                        <tr key={alt.kv_quant?.label ?? alt.kv_quant}>
                          <td>{alt.kv_quant?.label ?? alt.kv_quant}</td>
                          <td>{round(alt.memory_required_gb, 1)} GB</td>
                          <td>{round(alt.kv_cache_gb, 1)} GB</td>
                          <td>{round((alt.savings_fraction ?? 0) * 100, 1)}%</td>
                          <td>{alt.supported ? t('plan.pathsFeasible.yes') : t('plan.pathsFeasible.no')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      {capabilities.length > 0 && (
        <div className="metrics-card">
          <h4>{t('detail.sections.capabilities')}</h4>
          <div className="capability-badges">
            {capabilities.map((cap) => (
              <span key={cap} className="capability-badge">
                {cap}
              </span>
            ))}
          </div>
        </div>
      )}

      {ggufSources.length > 0 && (
        <div className="metrics-card">
          <h4>{t('detail.sections.ggufSources')}</h4>
          <ul className="gguf-list">
            {ggufSources.map((source, idx) => {
              const repo = typeof source === 'string' ? source : source.repo;
              const provider = typeof source === 'string' ? null : source.provider;
              const href = repo
                ? (repo.startsWith('http') ? repo : `https://huggingface.co/${repo}`)
                : '#';
              return (
                <li key={repo || idx}>
                  <a href={href} target="_blank" rel="noopener noreferrer">
                    {repo}
                  </a>
                  {provider && <span className="gguf-quant">{provider}</span>}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <div className="metrics-card">
        <h4>{t('detail.sections.scoreBreakdown')}</h4>
        <MetricBar
          label={t('detail.metrics.quality')}
          value={selectedModel.score_components?.quality}
        />
        <MetricBar
          label={t('detail.metrics.speed')}
          value={selectedModel.score_components?.speed}
        />
        <MetricBar
          label={t('detail.metrics.fit')}
          value={selectedModel.score_components?.fit}
        />
        <MetricBar
          label={t('detail.metrics.context')}
          value={selectedModel.score_components?.context}
        />
      </div>

      <div className="metrics-card">
        <h4>{t('detail.sections.performance')}</h4>
        <MetricBar
          label={t('detail.metrics.memoryUtilization')}
          value={selectedModel.utilization_pct}
        />
        <div className="kpi-grid">
          <div>
            <span>{t('detail.metrics.compositeScore')}</span>
            <strong>{round(selectedModel.score, 1)}</strong>
          </div>
          <div>
            <span>{t('detail.metrics.estimatedTps')}</span>
            <strong>{round(selectedModel.estimated_tps, 1)}</strong>
          </div>
        </div>
      </div>

      {Array.isArray(selectedModel.notes) && selectedModel.notes.length > 0 ? (
        <div className="metrics-card">
          <h4>{t('detail.sections.notes')}</h4>
          <ul>
            {selectedModel.notes.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="muted-copy">
          {t('detail.noNotes')}
        </p>
      )}
    </aside>
  );
}
