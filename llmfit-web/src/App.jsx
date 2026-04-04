import { useState } from 'react';
import { FilterProvider } from './contexts/FilterContext';
import { ModelProvider, useModelContext } from './contexts/ModelContext';
import { useModels } from './hooks/useModels';
import { useSystem } from './hooks/useSystem';
import Header from './components/Header';
import SystemPanel from './components/SystemPanel';
import FilterBar from './components/FilterBar';
import ModelTable from './components/ModelTable';
import DetailPanel from './components/DetailPanel';
import ComparePanel from './components/ComparePanel';

function DataLoader() {
  useModels();
  useSystem();
  return null;
}

function ModelsSection() {
  const { compareList, clearCompare, returned, total } = useModelContext();
  const [showCompare, setShowCompare] = useState(false);

  const compareCount = compareList.length;

  function openCompare() {
    if (compareCount >= 2) setShowCompare(true);
  }

  function closeCompare() {
    setShowCompare(false);
    clearCompare();
  }

  return (
    <section className="panel models-panel">
      <div className="panel-heading">
        <h2>Model Fit Explorer</h2>
        <div className="panel-heading-actions">
          {compareCount > 0 && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={openCompare}
              disabled={compareCount < 2}
              title={compareCount < 2 ? 'Select at least 2 models to compare' : ''}
            >
              Compare ({compareCount})
            </button>
          )}
          <span className="chip">
            {returned} shown / {total} matched
          </span>
        </div>
      </div>

      <FilterBar />

      {showCompare && compareCount >= 2 ? (
        <ComparePanel onClose={closeCompare} />
      ) : (
        <div className="models-layout">
          <ModelTable />
          <DetailPanel />
        </div>
      )}
    </section>
  );
}

export default function App() {
  return (
    <FilterProvider>
      <ModelProvider>
        <DataLoader />
        <div className="page-shell">
          <div className="orb orb-one" aria-hidden="true" />
          <div className="orb orb-two" aria-hidden="true" />

          <Header />
          <SystemPanel />
          <ModelsSection />
        </div>
      </ModelProvider>
    </FilterProvider>
  );
}
