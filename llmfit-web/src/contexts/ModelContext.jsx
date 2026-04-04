import { createContext, useCallback, useContext, useState } from 'react';

const ModelContext = createContext(null);

const MAX_COMPARE = 5;

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
  const [refreshTick, setRefreshTick] = useState(0);

  const triggerRefresh = useCallback(() => {
    setRefreshTick((t) => t + 1);
  }, []);

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
    refreshTick,
    triggerRefresh
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
