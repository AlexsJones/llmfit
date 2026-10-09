import { useI18n } from '../contexts/I18nContext';
import { useModelContext } from '../contexts/ModelContext';
import { round } from '../utils';

function SystemCard({ label, value, detail }) {
  return (
    <article className="system-card">
      <p className="system-label">{label}</p>
      <p className="system-value">{value}</p>
      {detail ? <p className="system-detail">{detail}</p> : null}
    </article>
  );
}

export default function SystemPanel() {
  const { t } = useI18n();
  const {
    systemInfo,
    systemLoading,
    systemError,
    simulationDraft,
    updateSimulationDraft,
    simulationActive,
    applySimulation,
    resetSimulation
  } = useModelContext();

  const system = systemInfo?.system;
  const gpus = system?.gpus ?? [];
  const gpuSummary =
    gpus.length === 0
      ? t('system.noGpu')
      : gpus
          .map(
            (gpu) =>
              `${gpu.count > 1 ? `${gpu.count}\u00d7 ` : ''}${gpu.name}${
                gpu.vram_gb ? ` (${round(gpu.vram_gb, 1)} GB)` : ''
              }`
          )
          .join(', ');

  // One line per GPU with what the estimator actually uses: free memory,
  // memory bandwidth and the backend.
  const gpuDetail = gpus
    .map((gpu) => {
      const parts = [gpu.backend];
      if (typeof gpu.free_vram_gb === 'number') {
        parts.push(t('system.freeVram', { value: round(gpu.free_vram_gb, 1) }));
      }
      if (typeof gpu.memory_bandwidth_gbps === 'number') {
        parts.push(t('system.bandwidth', { value: round(gpu.memory_bandwidth_gbps, 0) }));
      }
      return parts.filter(Boolean).join(' \u00b7 ');
    })
    .filter(Boolean)
    .join('; ');

  function handleSubmit(event) {
    event.preventDefault();
    applySimulation();
  }

  return (
    <section className="panel system-panel">
      <div className="panel-heading">
        <h2>{t('system.title')}</h2>
        <div className="panel-heading-actions">
          {simulationActive ? (
            <span className="chip chip-accent">{t('simulation.active')}</span>
          ) : null}
          {systemInfo?.profile ? (
            <span className="chip chip-accent" title={t('system.profileHint')}>
              {t('system.profile', { name: systemInfo.profile })}
            </span>
          ) : null}
          {systemInfo?.node ? (
            <span className="chip">
              {systemInfo.node.name} &middot; {systemInfo.node.os}
            </span>
          ) : null}
        </div>
      </div>

      {systemError ? (
        <div role="alert" className="alert error">
          {t('system.error', { error: systemError })}
        </div>
      ) : null}

      <div className="system-grid" aria-busy={systemLoading}>
        <SystemCard
          label={t('system.labels.cpu')}
          value={systemInfo?.system?.cpu_name ?? t('system.loading')}
          detail={
            systemInfo?.system?.cpu_cores
              ? t('system.cores', { count: systemInfo.system.cpu_cores })
              : undefined
          }
        />
        <SystemCard
          label={t('system.labels.totalRam')}
          value={
            systemInfo?.system?.total_ram_gb
              ? `${round(systemInfo.system.total_ram_gb, 1)} GB`
              : '\u2014'
          }
        />
        <SystemCard
          label={t('system.labels.availableRam')}
          value={
            systemInfo?.system?.available_ram_gb
              ? `${round(systemInfo.system.available_ram_gb, 1)} GB`
              : '\u2014'
          }
        />
        <SystemCard
          label={t('system.labels.gpu')}
          value={gpuSummary}
          detail={
            [
              system?.unified_memory ? t('system.unifiedMemory') : null,
              gpuDetail || null
            ]
              .filter(Boolean)
              .join(' \u2014 ') || undefined
          }
        />
        {typeof system?.gpu_available_gb === 'number' ? (
          <SystemCard
            label={t('system.labels.gpuAvailable')}
            value={`${round(system.gpu_available_gb, 1)} GB`}
            detail={
              typeof system.gpu_vram_gb === 'number'
                ? t('system.ofTotal', { value: round(system.gpu_vram_gb, 1) })
                : undefined
            }
          />
        ) : null}
      </div>

      <form className="simulation-panel" onSubmit={handleSubmit}>
        <div className="simulation-header">
          <div>
            <h3>{t('simulation.title')}</h3>
            <p className="muted-copy">
              {simulationActive
                ? t('simulation.activeHint')
                : t('simulation.idleHint')}
            </p>
          </div>
          <div className="simulation-actions">
            <button type="submit" className="btn btn-accent btn-sm">
              {simulationActive
                ? t('simulation.actions.update')
                : t('simulation.actions.apply')}
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={resetSimulation}
              disabled={!simulationActive && !Object.values(simulationDraft).some(Boolean)}
            >
              {t('simulation.actions.reset')}
            </button>
          </div>
        </div>

        <div className="simulation-grid">
          <label>
            <span>{t('simulation.fields.ram')}</span>
            <input
              type="number"
              min="0"
              step="0.1"
              value={simulationDraft.ramGb}
              onChange={(event) => updateSimulationDraft('ramGb', event.target.value)}
              placeholder={t('simulation.placeholders.ram')}
            />
          </label>

          <label>
            <span>{t('simulation.fields.vram')}</span>
            <input
              type="number"
              min="0"
              step="0.1"
              value={simulationDraft.vramGb}
              onChange={(event) => updateSimulationDraft('vramGb', event.target.value)}
              placeholder={t('simulation.placeholders.vram')}
            />
          </label>

          <label>
            <span>{t('simulation.fields.cpuCores')}</span>
            <input
              type="number"
              min="1"
              step="1"
              value={simulationDraft.cpuCores}
              onChange={(event) => updateSimulationDraft('cpuCores', event.target.value)}
              placeholder={t('simulation.placeholders.cpuCores')}
            />
          </label>
        </div>
      </form>
    </section>
  );
}
