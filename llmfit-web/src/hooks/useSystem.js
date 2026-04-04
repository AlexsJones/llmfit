import { useEffect } from 'react';
import { fetchSystemInfo } from '../api';
import { useModelContext } from '../contexts/ModelContext';

export function useSystem() {
  const {
    setSystemInfo,
    setSystemLoading,
    setSystemError,
    refreshTick
  } = useModelContext();

  useEffect(() => {
    const controller = new AbortController();

    async function loadSystem() {
      setSystemLoading(true);
      setSystemError('');
      try {
        const payload = await fetchSystemInfo(controller.signal);
        setSystemInfo(payload);
        setSystemLoading(false);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setSystemLoading(false);
        setSystemError(
          error instanceof Error
            ? error.message
            : 'Unable to load system details.'
        );
        setSystemInfo(null);
      }
    }

    loadSystem();
    return () => controller.abort();
  }, [refreshTick, setSystemInfo, setSystemLoading, setSystemError]);
}
