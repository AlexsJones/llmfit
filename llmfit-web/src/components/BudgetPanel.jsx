import { useState, useRef } from 'react';
import { fetchBudget } from '../api';
import { round, fitClass } from '../utils';

function ConfigCard({ config }) {
  const vramLabel = config.vram_gb != null
    ? `${config.vram_gb} GB VRAM`
    : 'CPU-only';

  return (
    <div className="panel" style={{ marginBottom: '1rem' }}>
      <div className="panel-heading" style={{ flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '1rem' }}>{config.name}</h3>
          <p className="system-detail" style={{ margin: '0.2rem 0 0' }}>{config.notes}</p>
        </div>
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <span className="chip chip-accent">${config.price_usd}</span>
          <span className="chip">{vramLabel}</span>
          <span className="chip">{round(config.ram_gb, 0)} GB RAM</span>
          <span className="chip">{config.cpu_cores} cores</span>
          <span className="chip">{config.gpu_backend}</span>
        </div>
      </div>

      {config.models.length === 0 ? (
        <p className="system-detail" style={{ padding: '0.5rem 0' }}>
          No models match the current fit filter.
        </p>
      ) : (
        <div className="table-wrap" style={{ marginTop: '0.5rem' }}>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Size</th>
                <th>Fit</th>
                <th>Score</th>
                <th>tok/s</th>
                <th>Mem</th>
                <th>Quant</th>
                <th>Runtime</th>
              </tr>
            </thead>
            <tbody>
              {config.models.map((m, i) => (
                <tr key={i}>
                  <td>
                    <span className="model-name" title={m.name} style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>
                      {m.name.length > 48 ? m.name.slice(0, 46) + '…' : m.name}
                    </span>
                  </td>
                  <td>{m.parameter_count}</td>
                  <td>
                    <span className={fitClass(m.fit_level)}>
                      {m.fit_label ?? m.fit_level}
                    </span>
                  </td>
                  <td><strong>{Math.round(m.score)}</strong></td>
                  <td>{round(m.estimated_tps, 1)}</td>
                  <td>{round(m.memory_required_gb, 1)} GB</td>
                  <td>{m.best_quant}</td>
                  <td>{m.runtime_label ?? m.runtime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function BudgetForm({ onResult }) {
  const [budget, setBudget] = useState('800');
  const [limit, setLimit] = useState('5');
  const [minFit, setMinFit] = useState('marginal');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setLoading(true);
    setError(null);

    try {
      const data = await fetchBudget(Number(budget), Number(limit), minFit, ctrl.signal);
      onResult(data);
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="simulation-panel" onSubmit={handleSubmit}>
      <div className="simulation-header">
        <div>
          <h3>Configure</h3>
          <p className="muted-copy">Find hardware within your purchase budget.</p>
        </div>
        <div className="simulation-actions">
          <button type="submit" className="btn btn-accent btn-sm" disabled={loading}>
            {loading ? 'Loading…' : 'Find hardware'}
          </button>
        </div>
      </div>

      <div className="simulation-grid">
        <label>
          <span>Max budget (USD)</span>
          <input
            type="number"
            min="0"
            step="50"
            value={budget}
            onChange={e => setBudget(e.target.value)}
            placeholder="e.g. 800"
            required
          />
        </label>
        <label>
          <span>Models per config</span>
          <input
            type="number"
            min="1"
            max="20"
            value={limit}
            onChange={e => setLimit(e.target.value)}
            placeholder="5"
          />
        </label>
        <label>
          <span>Min fit level</span>
          <select value={minFit} onChange={e => setMinFit(e.target.value)}>
            <option value="marginal">Marginal</option>
            <option value="good">Good</option>
            <option value="perfect">Perfect</option>
          </select>
        </label>
      </div>

      {error && (
        <div role="alert" className="alert error" style={{ marginTop: '0.75rem' }}>
          {error}
        </div>
      )}
    </form>
  );
}

export default function BudgetPanel({ onClose }) {
  const [result, setResult] = useState(null);

  return (
    <section className="panel system-panel">
      <div className="panel-heading">
        <h2>Hardware Budget Advisor</h2>
        <div className="panel-heading-actions">
          {result && (
            <span className="chip">
              {result.configurations.length} config{result.configurations.length !== 1 ? 's' : ''} within ${result.budget_usd}
            </span>
          )}
          <button type="button" className="btn btn-ghost btn-sm" onClick={onClose}>
            Close
          </button>
        </div>
      </div>

      <p className="muted-copy" style={{ marginBottom: '1rem' }}>
        Enter a purchase budget and see which hardware configurations fit — and which LLM models each one can run.
      </p>

      <BudgetForm onResult={setResult} />

      {result && (
        <div style={{ marginTop: '1.5rem' }}>
          {result.configurations.length === 0 ? (
            <p className="system-detail">No configurations found within a ${result.budget_usd} budget.</p>
          ) : (
            result.configurations.map((config, i) => (
              <ConfigCard key={i} config={config} />
            ))
          )}
        </div>
      )}
    </section>
  );
}
