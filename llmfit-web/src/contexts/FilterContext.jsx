import { createContext, useContext, useReducer } from 'react';

export const DEFAULT_FILTER_STATE = {
  search: '',
  minFit: 'marginal',
  runtime: 'any',
  useCase: 'all',
  provider: '',
  sort: 'score',
  limit: '50',
  // Advanced filters
  license: '',
  maxContext: '',
  capability: [],
  quant: [],
  runMode: [],
  paramsBucket: 'all',
  tp: 'all',
  showAdvanced: false
};

function filterReducer(state, action) {
  switch (action.type) {
    case 'SET_FILTER':
      return { ...state, [action.field]: action.value };
    case 'RESET_FILTERS':
      return { ...DEFAULT_FILTER_STATE };
    default:
      return state;
  }
}

const FilterContext = createContext(null);
const FilterDispatchContext = createContext(null);

export function FilterProvider({ children }) {
  const [filters, dispatch] = useReducer(filterReducer, DEFAULT_FILTER_STATE);

  return (
    <FilterContext.Provider value={filters}>
      <FilterDispatchContext.Provider value={dispatch}>
        {children}
      </FilterDispatchContext.Provider>
    </FilterContext.Provider>
  );
}

export function useFilters() {
  const ctx = useContext(FilterContext);
  if (ctx === null) {
    throw new Error('useFilters must be used within a FilterProvider');
  }
  return ctx;
}

export function useFilterDispatch() {
  const ctx = useContext(FilterDispatchContext);
  if (ctx === null) {
    throw new Error('useFilterDispatch must be used within a FilterProvider');
  }
  return ctx;
}
