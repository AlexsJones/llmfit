import { useMemo } from 'react';
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

export default function DetailPanel() {
  const { t } = useI18n();
  const { models, selectedModelName } = useModelContext();

  const selectedModel = useMemo(
    () => models.find((m) => m.name === selectedModelName) ?? null,
    [models, selectedModelName]
  );

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
