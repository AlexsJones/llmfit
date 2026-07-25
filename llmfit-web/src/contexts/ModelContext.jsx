import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { fetchDownloadStatus, isOllamaAvailable, startDownload } from '../api';

const ModelContext = createContext(null);

const MAX_COMPARE = 5;
const EMPTY_SIMULATION = {
  ramGb: '',
  vramGb: '',
  cpuCores: ''
};

function sanitizeSimulation(simulation) {
  return {
    ramGb: String(simulation?.ramGb ?? '').trim(),
    vramGb: String(simulation?.vramGb ?? '').trim(),
    cpuCores: String(simulation?.cpuCores ?? '').trim()
  };
}

export function ModelProvider({ children }) {
  const [models, setModels] = useState([]);
  const [allModels, setAllModels] = useState([]); // pre-client-filter, for dropdown options
  const [total, setTotal] = useState(0);
  const [returned, setReturned] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [systemInfo, setSystemInfo] = useState(null);
  const [systemLoading, setSystemLoading] = useState(true);
  const [systemError, setSystemError] = useState('');
  const [selectedModelName, setSelectedModelName] = useState(null);
  const [compareList, setCompareList] = useState([]);
  const [installedModels, setInstalledModels] = useState([]);
  const [downloadStates, setDownloadStates] = useState({});
  const [ollamaAvailable, setOllamaAvailable] = useState(false);
  const [ollamaChecking, setOllamaChecking] = useState(true);
  const pollRefs = useRef({});
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const available = await isOllamaAvailable();
        if (!cancelled) setOllamaAvailable(available);
      } catch {
        if (!cancelled) setOllamaAvailable(false);
      } finally {
        if (!cancelled) setOllamaChecking(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const startModelDownload = useCallback(async (modelName) => {
    if (downloadStates[modelName]) return;
    setDownloadStates((prev) => ({ ...prev, [modelName]: { progress: 0, status: 'starting', done: false, error: null } }));
    try {
      const result = await startDownload(modelName, 'ollama');
      const id = result.id;
      pollRefs.current[modelName] = setInterval(async () => {
        try {
          const status = await fetchDownloadStatus(id);
          setDownloadStates((prev) => ({
            ...prev,
            [modelName]: {
              progress: status.progress_pct,
              status: status.status,
              done: status.done,
              error: status.error,
            },
          }));
          if (status.done) {
            clearInterval(pollRefs.current[modelName]);
            delete pollRefs.current[modelName];
            if (!status.error) {
              setInstalledModels((prev) =>
                prev.includes(modelName) ? prev : [...prev, modelName]
              );
            }
            setTimeout(() => {
              setDownloadStates((prev) => {
                const next = { ...prev };
                delete next[modelName];
                return next;
              });
            }, 3000);
          }
        } catch {
          clearInterval(pollRefs.current[modelName]);
          delete pollRefs.current[modelName];
          setDownloadStates((prev) => ({
            ...prev,
            [modelName]: { progress: null, status: 'error', done: true, error: 'Poll failed' },
          }));
        }
      }, 500);
    } catch (err) {
      setDownloadStates((prev) => ({
        ...prev,
        [modelName]: { progress: null, status: 'error', done: true, error: err.message },
      }));
    }
  }, [downloadStates]);

  const [simulationDraft, setSimulationDraft] = useState(EMPTY_SIMULATION);
  const [appliedSimulation, setAppliedSimulation] = useState(EMPTY_SIMULATION);

  const triggerRefresh = useCallback(() => {
    setRefreshTick((t) => t + 1);
  }, []);

  const updateSimulationDraft = useCallback((field, value) => {
    setSimulationDraft((current) => ({
      ...current,
      [field]: value
    }));
  }, []);

  const applySimulation = useCallback(() => {
    setAppliedSimulation(sanitizeSimulation(simulationDraft));
  }, [simulationDraft]);

  const resetSimulation = useCallback(() => {
    setSimulationDraft(EMPTY_SIMULATION);
    setAppliedSimulation(EMPTY_SIMULATION);
  }, []);

  const simulationActive = Object.values(appliedSimulation).some((value) => value !== '');

  const toggleCompare = useCallback((modelName) => {
    setCompareList((prev) => {
      if (prev.includes(modelName)) {
        return prev.filter((n) => n !== modelName);
      }
      if (prev.length >= MAX_COMPARE) {
        return prev;
      }
      return [...prev, modelName];
    });
  }, []);

  const clearCompare = useCallback(() => {
    setCompareList([]);
  }, []);

  const value = {
    models,
    setModels,
    allModels,
    setAllModels,
    total,
    setTotal,
    returned,
    setReturned,
    loading,
    setLoading,
    error,
    setError,
    systemInfo,
    setSystemInfo,
    systemLoading,
    setSystemLoading,
    systemError,
    setSystemError,
    selectedModelName,
    setSelectedModelName,
    compareList,
    toggleCompare,
    clearCompare,
    installedModels,
    setInstalledModels,
    downloadStates,
    ollamaAvailable,
    ollamaChecking,
    startModelDownload,
    refreshTick,
    triggerRefresh,
    simulationDraft,
    updateSimulationDraft,
    appliedSimulation,
    simulationActive,
    applySimulation,
    resetSimulation
  };

  return (
    <ModelContext.Provider value={value}>{children}</ModelContext.Provider>
  );
}

export function useModelContext() {
  const ctx = useContext(ModelContext);
  if (ctx === null) {
    throw new Error('useModelContext must be used within a ModelProvider');
  }
  return ctx;
}

export { MAX_COMPARE };
