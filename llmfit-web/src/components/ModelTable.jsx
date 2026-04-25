import { useI18n } from '../contexts/I18nContext';
import { useModelContext, MAX_COMPARE } from '../contexts/ModelContext';
import {
  round,
  fitClass,
  modeClass,
  copyModelName,
  translateFitLevel,
  translateRunMode,
} from '../utils';

export default function ModelTable() {
  const { locale, t } = useI18n();
  const {
    models,
    loading,
    error,
    selectedModelName,
    setSelectedModelName,
    compareList,
    toggleCompare,
    installedModels
  } = useModelContext();

  const installedSet = new Set(
    Array.isArray(installedModels) ? installedModels : []
  );
  const compareFull = compareList.length >= MAX_COMPARE;

  return (
    <div className="table-wrap">
      {error ? (
        <div role="alert" className="alert error" style={{ margin: '0.75rem' }}>
          {t('table.error', { error })}
        </div>
      ) : null}

      <table>
        <thead>
          <tr>
            <th className="col-compare">{t('table.columns.compare')}</th>
            <th>{t('table.columns.model')}</th>
            <th>{t('table.columns.provider')}</th>
            <th>{t('table.columns.params')}</th>
            <th>{t('table.columns.fit')}</th>
            <th>{t('table.columns.mode')}</th>
            <th>{t('table.columns.runtime')}</th>
            <th>{t('table.columns.score')}</th>
            <th>{t('table.columns.tps')}</th>
            <th>{t('table.columns.mem')}</th>
            <th>{t('table.columns.context')}</th>
            <th>{t('table.columns.release')}</th>
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr>
              <td colSpan="12" className="table-status">
                {t('table.loading')}
              </td>
            </tr>
          ) : null}

          {!loading && models.length === 0 && !error ? (
            <tr>
              <td colSpan="12" className="table-status">
                {t('table.empty')}
              </td>
            </tr>
          ) : null}

          {!loading
            ? models.map((model) => {
                const isSelected = model.name === selectedModelName;
                const isCompared = compareList.includes(model.name);
                const isInstalled = installedSet.has(model.name);
                const disableCompare = !isCompared && compareFull;

                return (
                  <tr
                    key={model.name}
                    className={isSelected ? 'selected' : ''}
                    onClick={() => setSelectedModelName(model.name)}
                  >
                    <td
                      className="col-compare"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <input
                        type="checkbox"
                        className="compare-checkbox"
                        checked={isCompared}
                        disabled={disableCompare}
                        onChange={() => toggleCompare(model.name)}
                        title={
                          disableCompare
                            ? t('table.maxCompare', { count: MAX_COMPARE })
                            : t('table.addToComparison')
                        }
                      />
                    </td>
                    <td className="model-name">
                      <span>{model.name}</span>
                      {isInstalled && (
                        <span className="chip chip-installed">{t('table.installed')}</span>
                      )}
                      <button
                        type="button"
                        className="btn-copy"
                        title={t('table.copyModelName')}
                        onClick={(e) => {
                          e.stopPropagation();
                          copyModelName(model.name);
                        }}
                      >
                        &#x2398;
                      </button>
                    </td>
                    <td>{model.provider}</td>
                    <td>{round(model.params_b, 1)}B</td>
                    <td>
                      <span className={fitClass(model.fit_level)}>
                        {translateFitLevel(t, model.fit_level, model.fit_label)}
                      </span>
                    </td>
                    <td>
                      <span className={modeClass(model.run_mode)}>
                        {translateRunMode(t, model.run_mode, model.run_mode_label)}
                      </span>
                    </td>
                    <td>{model.runtime_label}</td>
                    <td>{round(model.score, 1)}</td>
                    <td>{round(model.estimated_tps, 1)}</td>
                    <td>{round(model.utilization_pct, 1)}</td>
                    <td>
                      {typeof model.context_length === 'number'
                        ? model.context_length.toLocaleString(locale)
                        : model.context_length ?? '\u2014'}
                    </td>
                    <td>{model.release_date ?? '\u2014'}</td>
                  </tr>
                );
              })
            : null}
        </tbody>
      </table>
    </div>
  );
}
