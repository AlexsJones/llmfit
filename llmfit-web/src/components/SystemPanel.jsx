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
  const { systemInfo, systemLoading, systemError } = useModelContext();

  const gpus = systemInfo?.system?.gpus ?? [];
  const gpuSummary =
    gpus.length === 0
      ? t('system.noGpu')
      : gpus
          .map(
            (gpu) =>
              `${gpu.name}${gpu.vram_gb ? ` (${round(gpu.vram_gb, 1)} GB)` : ''}`
          )
          .join(', ');

  return (
    <section className="panel system-panel">
      <div className="panel-heading">
        <h2>{t('system.title')}</h2>
        {systemInfo?.node ? (
          <span className="chip">
            {systemInfo.node.name} &middot; {systemInfo.node.os}
          </span>
        ) : null}
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
            systemInfo?.system?.unified_memory
              ? t('system.unifiedMemory')
              : undefined
          }
        />
      </div>
    </section>
  );
}
