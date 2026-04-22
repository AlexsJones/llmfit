import { useMemo } from 'react';
import { useI18n } from '../contexts/I18nContext';
import { useModelContext } from '../contexts/ModelContext';
import { round, translateFitLevel, translateRunMode } from '../utils';

function buildCompareFields(t) {
  return [
    { key: 'fit_level', label: t('compare.fields.fitLevel'), type: 'fit' },
    { key: 'score', label: t('compare.fields.score'), type: 'number', digits: 1, best: 'max' },
    {
      key: 'estimated_tps',
      label: t('compare.fields.tps'),
      type: 'number',
      digits: 1,
      best: 'max'
    },
    {
      key: 'memory_required_gb',
      label: t('compare.fields.memoryRequired'),
      type: 'number',
      digits: 2,
      best: 'min'
    },
    {
      key: 'memory_available_gb',
      label: t('compare.fields.memoryAvailable'),
      type: 'number',
      digits: 2,
      best: 'max'
    },
    { key: 'best_quant', label: t('compare.fields.bestQuant'), type: 'text' },
    {
      key: 'context_length',
      label: t('compare.fields.context'),
      type: 'number',
      digits: 0,
      best: 'max'
    },
    { key: 'runtime_label', label: t('compare.fields.runtime'), type: 'text' },
    { key: 'run_mode', label: t('compare.fields.runMode'), type: 'run_mode' }
  ];
}

function bestValue(compareModels, field) {
  if (field.type !== 'number' || !field.best) return null;
  const values = compareModels
    .map((m) => m[field.key])
    .filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (values.length === 0) return null;
  return field.best === 'max' ? Math.max(...values) : Math.min(...values);
}

function formatField(t, locale, model, field) {
  const val = model[field.key];
  if (field.type === 'number') {
    if (field.key === 'context_length') {
      return typeof val === 'number' ? val.toLocaleString(locale) : (val ?? '\u2014');
    }
    return round(val, field.digits ?? 1);
  }
  if (field.type === 'fit') {
    return translateFitLevel(t, model.fit_level, model.fit_label);
  }
  if (field.type === 'run_mode') {
    return translateRunMode(t, model.run_mode, model.run_mode_label);
  }
  return val ?? '\u2014';
}

export default function ComparePanel({ onClose }) {
  const { locale, t } = useI18n();
  const { models, compareList } = useModelContext();
  const close = onClose || (() => {});
  const compareFields = useMemo(() => buildCompareFields(t), [t]);

  const compareModels = useMemo(() => {
    return compareList
      .map((name) => models.find((m) => m.name === name))
      .filter(Boolean);
  }, [models, compareList]);

  if (compareModels.length === 0) {
    return (
      <div className="compare-panel">
        <div className="compare-header">
          <h3>{t('compare.titleEmpty')}</h3>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={close}
          >
            {t('compare.close')}
          </button>
        </div>
        <p className="muted-copy">
          {t('compare.instructions')}
        </p>
      </div>
    );
  }

  return (
    <div className="compare-panel">
      <div className="compare-header">
        <h3>{t('compare.headerCount', { count: compareModels.length })}</h3>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={close}
        >
          {t('compare.close')}
        </button>
      </div>

      <div className="table-wrap">
        <table className="compare-table">
          <thead>
            <tr>
              <th>&nbsp;</th>
              {compareModels.map((m) => (
                <th key={m.name}>
                  {m.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {compareFields.map((field) => {
              const best = bestValue(compareModels, field);
              return (
                <tr key={field.key}>
                  <td>{field.label}</td>
                  {compareModels.map((m) => {
                    const raw = m[field.key];
                    const isBest =
                      best !== null &&
                      typeof raw === 'number' &&
                      Number.isFinite(raw) &&
                      raw === best;
                    return (
                      <td
                        key={m.name}
                        className={isBest ? 'compare-best' : ''}
                      >
                        {formatField(t, locale, m, field)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
