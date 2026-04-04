import { useEffect } from 'react';
import { fetchModels } from '../api';
import { useFilters } from '../contexts/FilterContext';
import { useModelContext } from '../contexts/ModelContext';
import { applyClientFitFilter, applyClientFilters } from '../utils';

export function useModels() {
  const filters = useFilters();
  const {
    setModels,
    setAllModels,
    setTotal,
    setReturned,
    setLoading,
    setError,
    setSelectedModelName,
    refreshTick
  } = useModelContext();

  useEffect(() => {
    const controller = new AbortController();

    async function loadModels() {
      setLoading(true);
      setError('');
      try {
        const payload = await fetchModels(filters, controller.signal);
        const fetchedModels = Array.isArray(payload.models)
          ? payload.models
          : [];

        const fitFiltered = applyClientFitFilter(fetchedModels, filters.minFit);
        setAllModels(fitFiltered);
        const clientFiltered = applyClientFilters(fitFiltered, filters);

        const limit = Number.parseInt(filters.limit, 10);
        const models =
          Number.isFinite(limit) && limit > 0
            ? clientFiltered.slice(0, limit)
            : clientFiltered;

        const total = clientFiltered.length;

        setModels(models);
        setTotal(total);
        setReturned(models.length);
        setLoading(false);

        setSelectedModelName((current) => {
          if (!current) {
            return models[0]?.name ?? null;
          }
          const stillVisible = models.some((m) => m.name === current);
          return stillVisible ? current : models[0]?.name ?? null;
        });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setModels([]);
        setTotal(0);
        setReturned(0);
        setLoading(false);
        setError(
          error instanceof Error ? error.message : 'Unable to load model fits.'
        );
        setSelectedModelName(null);
      }
    }

    loadModels();
    return () => controller.abort();
  }, [
    filters,
    refreshTick,
    setModels,
    setAllModels,
    setTotal,
    setReturned,
    setLoading,
    setError,
    setSelectedModelName
  ]);
}
