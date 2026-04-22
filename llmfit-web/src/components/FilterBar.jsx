import { useMemo, useRef, useState, useEffect } from 'react';
import { useFilters, useFilterDispatch } from '../contexts/FilterContext';
import { useI18n } from '../contexts/I18nContext';
import { useModelContext } from '../contexts/ModelContext';
import { collectUniqueValues } from '../utils';

function MultiSelectDropdown({ label, field, options }) {
  const { t } = useI18n();
  const filters = useFilters();
  const dispatch = useFilterDispatch();
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  const selected = filters[field] || [];
  const count = selected.length;

  useEffect(() => {
    function handleClickOutside(e) {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open]);

  function toggle(value) {
    const next = selected.includes(value)
      ? selected.filter((v) => v !== value)
      : [...selected, value];
    dispatch({ type: 'SET_FILTER', field, value: next });
  }

  return (
    <div className="multi-select-wrap" ref={ref}>
      <span>{label}</span>
      <button
        type="button"
        className="multi-select-btn"
        onClick={() => setOpen((o) => !o)}
      >
        {count > 0
          ? t('filters.multiSelect.selectedCount', { count })
          : t('filters.multiSelect.any')}
        <span className="multi-select-caret">{open ? '\u25B2' : '\u25BC'}</span>
      </button>
      {open && (
        <div className="multi-select-popover">
          {options.length === 0 ? (
            <p className="multi-select-empty">{t('filters.multiSelect.noOptions')}</p>
          ) : (
            options.map((opt) => (
              <label key={opt} className="multi-select-option">
                <input
                  type="checkbox"
                  checked={selected.includes(opt)}
                  onChange={() => toggle(opt)}
                />
                <span>{opt}</span>
              </label>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default function FilterBar() {
  const { t } = useI18n();
  const filters = useFilters();
  const dispatch = useFilterDispatch();
  const { allModels } = useModelContext();

  const FIT_OPTIONS = useMemo(
    () => [
      { value: 'marginal', label: t('filters.fitOptions.marginal') },
      { value: 'good', label: t('filters.fitOptions.good') },
      { value: 'perfect', label: t('filters.fitOptions.perfect') },
      { value: 'too_tight', label: t('filters.fitOptions.too_tight') },
      { value: 'all', label: t('filters.fitOptions.all') },
    ],
    [t]
  );

  const RUNTIME_OPTIONS = useMemo(
    () => [
      { value: 'any', label: t('filters.runtimeOptions.any') },
      { value: 'mlx', label: t('filters.runtimeOptions.mlx') },
      { value: 'llamacpp', label: t('filters.runtimeOptions.llamacpp') },
      { value: 'vllm', label: t('filters.runtimeOptions.vllm') },
    ],
    [t]
  );

  const USE_CASE_OPTIONS = useMemo(
    () => [
      { value: 'all', label: t('filters.useCaseOptions.all') },
      { value: 'general', label: t('filters.useCaseOptions.general') },
      { value: 'coding', label: t('filters.useCaseOptions.coding') },
      { value: 'reasoning', label: t('filters.useCaseOptions.reasoning') },
      { value: 'chat', label: t('filters.useCaseOptions.chat') },
      { value: 'multimodal', label: t('filters.useCaseOptions.multimodal') },
      { value: 'embedding', label: t('filters.useCaseOptions.embedding') },
    ],
    [t]
  );

  const LIMIT_OPTIONS = useMemo(
    () => [
      { value: '10', label: '10' },
      { value: '20', label: '20' },
      { value: '50', label: '50' },
      { value: '100', label: '100' },
      { value: '200', label: '200' },
      { value: '', label: t('filters.limitAll') },
    ],
    [t]
  );

  const SORT_OPTIONS = useMemo(
    () => [
      { value: 'score', label: t('filters.sortOptions.score') },
      { value: 'tps', label: t('filters.sortOptions.tps') },
      { value: 'params', label: t('filters.sortOptions.params') },
      { value: 'mem', label: t('filters.sortOptions.mem') },
      { value: 'ctx', label: t('filters.sortOptions.ctx') },
      { value: 'date', label: t('filters.sortOptions.date') },
      { value: 'use_case', label: t('filters.sortOptions.use_case') },
    ],
    [t]
  );

  const PARAMS_BUCKET_OPTIONS = useMemo(
    () => [
      { value: 'all', label: t('filters.paramsBucketOptions.all') },
      { value: 'tiny', label: t('filters.paramsBucketOptions.tiny') },
      { value: 'small', label: t('filters.paramsBucketOptions.small') },
      { value: 'medium', label: t('filters.paramsBucketOptions.medium') },
      { value: 'large', label: t('filters.paramsBucketOptions.large') },
      { value: 'xl', label: t('filters.paramsBucketOptions.xl') },
    ],
    [t]
  );

  const TP_OPTIONS = useMemo(
    () => [
      { value: 'all', label: t('filters.tpOptions.all') },
      { value: '1', label: t('filters.tpOptions.1') },
      { value: '2', label: t('filters.tpOptions.2') },
      { value: '4', label: t('filters.tpOptions.4') },
      { value: '8', label: t('filters.tpOptions.8') },
    ],
    [t]
  );

  const handleChange = (field) => (e) => {
    const value =
      e.target.type === 'checkbox' ? e.target.checked : e.target.value;
    dispatch({ type: 'SET_FILTER', field, value });
  };

  const availableCapabilities = useMemo(
    () => collectUniqueValues(allModels, 'capabilities'),
    [allModels]
  );

  const availableQuants = useMemo(
    () => collectUniqueValues(allModels, 'best_quant'),
    [allModels]
  );

  const availableRunModes = useMemo(
    () => collectUniqueValues(allModels, 'run_mode'),
    [allModels]
  );

  const advancedCount =
    (filters.capability.length > 0 ? 1 : 0) +
    (filters.license ? 1 : 0) +
    (filters.quant.length > 0 ? 1 : 0) +
    (filters.runMode.length > 0 ? 1 : 0) +
    (filters.paramsBucket !== 'all' ? 1 : 0) +
    (filters.tp !== 'all' ? 1 : 0) +
    (filters.maxContext ? 1 : 0);

  return (
    <div className="filters-outer">
      <div className="filters-shell">
        <label>
          <span>{t('filters.searchLabel')}</span>
          <input
            type="text"
            value={filters.search}
            onChange={handleChange('search')}
            placeholder={t('filters.searchPlaceholder')}
          />
        </label>

        <label>
          <span>{t('filters.fitLabel')}</span>
          <select value={filters.minFit} onChange={handleChange('minFit')}>
            {FIT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>{t('filters.runtimeLabel')}</span>
          <select value={filters.runtime} onChange={handleChange('runtime')}>
            {RUNTIME_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>{t('filters.useCaseLabel')}</span>
          <select value={filters.useCase} onChange={handleChange('useCase')}>
            {USE_CASE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>{t('filters.providerLabel')}</span>
          <input
            type="text"
            value={filters.provider}
            onChange={handleChange('provider')}
            placeholder={t('filters.providerPlaceholder')}
          />
        </label>

        <label>
          <span>{t('filters.sortLabel')}</span>
          <select value={filters.sort} onChange={handleChange('sort')}>
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>{t('filters.limitLabel')}</span>
          <select
            value={String(filters.limit)}
            onChange={handleChange('limit')}
          >
            {LIMIT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="filters-toggle-row">
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={() =>
            dispatch({
              type: 'SET_FILTER',
              field: 'showAdvanced',
              value: !filters.showAdvanced
            })
          }
        >
          {filters.showAdvanced
            ? t('filters.advancedLess')
            : t('filters.advancedMore')}
          {advancedCount > 0 ? ` ${t('filters.advancedActive', { count: advancedCount })}` : ''}
        </button>
      </div>

      {filters.showAdvanced && (
        <div className="filters-shell filters-advanced">
          <MultiSelectDropdown
            label={t('filters.capabilityLabel')}
            field="capability"
            options={availableCapabilities}
          />

          <label>
            <span>{t('filters.licenseLabel')}</span>
            <input
              type="text"
              value={filters.license}
              onChange={handleChange('license')}
              placeholder={t('filters.licensePlaceholder')}
            />
          </label>

          <MultiSelectDropdown
            label={t('filters.quantizationLabel')}
            field="quant"
            options={availableQuants}
          />

          <MultiSelectDropdown
            label={t('filters.runModeLabel')}
            field="runMode"
            options={availableRunModes}
          />

          <label>
            <span>{t('filters.paramsBucketLabel')}</span>
            <select
              value={filters.paramsBucket}
              onChange={handleChange('paramsBucket')}
            >
              {PARAMS_BUCKET_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>{t('filters.tensorParallelLabel')}</span>
            <select value={filters.tp} onChange={handleChange('tp')}>
              {TP_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>{t('filters.maxContextLabel')}</span>
            <input
              type="number"
              value={filters.maxContext}
              onChange={handleChange('maxContext')}
              placeholder={t('filters.maxContextPlaceholder')}
              min="0"
            />
          </label>
        </div>
      )}
    </div>
  );
}
