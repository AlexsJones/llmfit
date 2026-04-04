import { useMemo } from 'react';
import { useModelContext } from '../contexts/ModelContext';
import { round } from '../utils';

const COMPARE_FIELDS = [
  { key: 'fit_label', label: 'Fit level', type: 'text' },
  { key: 'score', label: 'Score', type: 'number', digits: 1, best: 'max' },
  {
    key: 'estimated_tps',
    label: 'TPS',
    type: 'number',
    digits: 1,
    best: 'max'
  },
  {
    key: 'memory_required_gb',
    label: 'Memory required (GB)',
    type: 'number',
    digits: 2,
    best: 'min'
  },
  {
    key: 'memory_available_gb',
    label: 'Memory available (GB)',
    type: 'number',
    digits: 2,
    best: 'max'
  },
  { key: 'best_quant', label: 'Best quant', type: 'text' },
  {
    key: 'context_length',
    label: 'Context',
    type: 'number',
    digits: 0,
    best: 'max'
  },
  { key: 'runtime_label', label: 'Runtime', type: 'text' },
  { key: 'run_mode_label', label: 'Run mode', type: 'text' }
];

function bestValue(compareModels, field) {
  if (field.type !== 'number' || !field.best) return null;
  const values = compareModels
    .map((m) => m[field.key])
    .filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (values.length === 0) return null;
  return field.best === 'max' ? Math.max(...values) : Math.min(...values);
}

function formatField(model, field) {
  const val = model[field.key];
  if (field.type === 'number') {
    if (field.key === 'context_length') {
      return val?.toLocaleString?.() ?? val ?? '\u2014';
    }
    return round(val, field.digits ?? 1);
  }
  return val ?? '\u2014';
}

export default function ComparePanel({ onClose }) {
  const { models, compareList } = useModelContext();
  const close = onClose || (() => {});

  const compareModels = useMemo(() => {
    return compareList
      .map((name) => models.find((m) => m.name === name))
      .filter(Boolean);
  }, [models, compareList]);

  if (compareModels.length === 0) {
    return (
      <div className="compare-panel">
        <div className="compare-header">
          <h3>Model Comparison</h3>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={close}
          >
            Close
          </button>
        </div>
        <p className="muted-copy">
          Select models using the checkboxes in the table to compare them side by
          side.
        </p>
      </div>
    );
  }

  return (
    <div className="compare-panel">
      <div className="compare-header">
        <h3>
          Comparing {compareModels.length} model
          {compareModels.length !== 1 ? 's' : ''}
        </h3>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={close}
        >
          Close
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
            {COMPARE_FIELDS.map((field) => {
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
                        {formatField(m, field)}
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
