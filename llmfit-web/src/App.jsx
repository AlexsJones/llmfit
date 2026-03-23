import React, { useEffect, useMemo, useState } from 'react';
import { DEFAULT_FILTERS, fetchModels, fetchSystemInfo } from './api';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { 
  Server, Cpu, HardDrive, RefreshCw, Sun, Moon, 
  Activity, Monitor, Database, Settings, BarChart 
} from "lucide-react";

const THEME_KEY = 'llmfit-theme';

const FIT_OPTIONS = [
  { value: 'marginal', label: 'Runnable (Marginal+)' },
  { value: 'good', label: 'Good or better' },
  { value: 'perfect', label: 'Perfect only' },
  { value: 'too_tight', label: 'Too-tight only' },
  { value: 'all', label: 'All levels' }
];

const RUNTIME_OPTIONS = [
  { value: 'any', label: 'Any runtime' },
  { value: 'mlx', label: 'MLX' },
  { value: 'llamacpp', label: 'llama.cpp' },
  { value: 'vllm', label: 'vLLM' }
];

const USE_CASE_OPTIONS = [
  { value: 'all', label: 'All use cases' },
  { value: 'general', label: 'General' },
  { value: 'coding', label: 'Coding' },
  { value: 'reasoning', label: 'Reasoning' },
  { value: 'chat', label: 'Chat' },
  { value: 'multimodal', label: 'Multimodal' },
  { value: 'embedding', label: 'Embedding' }
];

const LIMIT_OPTIONS = [
  { value: '10', label: '10' },
  { value: '20', label: '20' },
  { value: '50', label: '50' },
  { value: '100', label: '100' },
  { value: '200', label: '200' },
  { value: '', label: 'All' }
];

const SORT_OPTIONS = [
  { value: 'score', label: 'Sort: Score' },
  { value: 'tps', label: 'Sort: TPS' },
  { value: 'params', label: 'Sort: Params' },
  { value: 'mem', label: 'Sort: Memory' },
  { value: 'ctx', label: 'Sort: Context' },
  { value: 'date', label: 'Sort: Release date' },
  { value: 'use_case', label: 'Sort: Use case' }
];

function initialTheme() {
  if (typeof window === 'undefined') return 'dark';
  const stored = window.localStorage.getItem(THEME_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function round(value, digits = 1) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}

function getFitBadgeVariant(level) {
  switch (level) {
    case 'perfect': return 'default'; // primary
    case 'good': return 'secondary';
    case 'marginal': return 'outline';
    case 'too_tight': return 'destructive';
    default: return 'outline';
  }
}

function fitRank(level) {
  switch (level) {
    case 'perfect': return 3;
    case 'good': return 2;
    case 'marginal': return 1;
    case 'too_tight': return 0;
    default: return -1;
  }
}

function applyClientFitFilter(models, minFit) {
  const list = Array.isArray(models) ? models : [];
  if (minFit === 'all') return list;
  if (minFit === 'too_tight') return list.filter((m) => m.fit_level === 'too_tight');
  const threshold = fitRank(minFit);
  return list.filter((m) => fitRank(m.fit_level) >= threshold);
}

function MetricBar({ label, value }) {
  const safe = Number.isFinite(value) ? Math.max(0, Math.min(value, 100)) : 0;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center text-xs font-medium text-muted-foreground">
        <span>{label}</span>
        <span>{round(value, 1)}</span>
      </div>
      <Progress value={safe} className="h-1.5" />
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState(initialTheme);
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [systemState, setSystemState] = useState({ loading: true, error: '', payload: null });
  const [modelsState, setModelsState] = useState({ loading: true, error: '', models: [], total: 0, returned: 0 });
  const [selectedModelName, setSelectedModelName] = useState(null);
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    document.documentElement.classList.remove('light', 'dark');
    document.documentElement.classList.add(theme);
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    const controller = new AbortController();
    async function loadSystem() {
      setSystemState((prev) => ({ ...prev, loading: true, error: '' }));
      try {
        const payload = await fetchSystemInfo(controller.signal);
        setSystemState({ loading: false, error: '', payload });
      } catch (error) {
        if (!controller.signal.aborted) {
          setSystemState({ loading: false, error: error.message || 'Unable to load system details.', payload: null });
        }
      }
    }
    loadSystem();
    return () => controller.abort();
  }, [refreshTick]);

  useEffect(() => {
    const controller = new AbortController();
    async function loadModels() {
      setModelsState((prev) => ({ ...prev, loading: true, error: '' }));
      try {
        const payload = await fetchModels(filters, controller.signal);
        const fetchedModels = Array.isArray(payload.models) ? payload.models : [];
        const fitFiltered = applyClientFitFilter(fetchedModels, filters.minFit);
        const limit = Number.parseInt(filters.limit, 10);
        const models = Number.isFinite(limit) && limit > 0 ? fitFiltered.slice(0, limit) : fitFiltered;
        const total = filters.minFit === 'too_tight' ? fitFiltered.length : (payload.total_models || fitFiltered.length);
        
        setModelsState({ loading: false, error: '', models, total, returned: models.length });
        setSelectedModelName((current) => {
          if (!current || !models.some((m) => m.name === current)) return models[0]?.name ?? null;
          return current;
        });
      } catch (error) {
        if (!controller.signal.aborted) {
          setModelsState({ loading: false, error: error.message || 'Unable to load model fits.', models: [], total: 0, returned: 0 });
          setSelectedModelName(null);
        }
      }
    }
    loadModels();
    return () => controller.abort();
  }, [filters, refreshTick]);

  const selectedModel = useMemo(
    () => modelsState.models.find((m) => m.name === selectedModelName) ?? null,
    [modelsState.models, selectedModelName]
  );

  const handleFieldChange = (field, value) => {
    setFilters((current) => ({ ...current, [field]: value }));
  };

  const gpus = systemState.payload?.system?.gpus ?? [];
  const gpuSummary = gpus.length === 0 ? 'No GPU detected' : gpus.map((gpu) => `${gpu.name}${gpu.vram_gb ? ` (${round(gpu.vram_gb, 1)}VRAM)` : ''}`).join(', ');

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans">
      <header className="sticky top-0 z-10 bg-background/80 backdrop-blur-md border-b">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Activity className="h-6 w-6 text-primary" />
            <div>
              <h1 className="text-lg font-semibold tracking-tight leading-none">LLMFit Explorer</h1>
              <p className="text-xs text-muted-foreground mt-1">Local hardware compatibility</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm" onClick={() => setFilters(DEFAULT_FILTERS)}>
              Reset
            </Button>
            <Button variant="outline" size="sm" onClick={() => setRefreshTick((t) => t + 1)}>
              <RefreshCw className="mr-2 h-4 w-4" /> Refresh
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}>
              {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-6 space-y-6">
        
        {/* System Summary */}
        <section className="space-y-3">
          <div className="flex items-center space-x-2">
            <Monitor className="h-5 w-5 text-muted-foreground" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">System Overview</h2>
            {systemState.payload?.node && (
              <Badge variant="secondary" className="ml-auto font-mono text-xs">
                {systemState.payload.node.name} • {systemState.payload.node.os}
              </Badge>
            )}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">CPU</CardTitle>
                <Cpu className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-lg font-bold truncate" title={systemState.payload?.system?.cpu_name}>
                  {systemState.payload?.system?.cpu_name ?? '—'}
                </div>
                <p className="text-xs text-muted-foreground">
                  {systemState.payload?.system?.cpu_cores ? `${systemState.payload.system.cpu_cores} cores` : '—'}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total RAM</CardTitle>
                <MemoryStick className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemState.payload?.system?.total_ram_gb ? `${round(systemState.payload.system.total_ram_gb, 1)} GB` : '—'}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avail. RAM</CardTitle>
                <HardDrive className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-primary">
                  {systemState.payload?.system?.available_ram_gb ? `${round(systemState.payload.system.available_ram_gb, 1)} GB` : '—'}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">GPU</CardTitle>
                <Server className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-sm font-bold truncate" title={gpuSummary}>{gpuSummary}</div>
                <p className="text-xs text-muted-foreground">
                  {systemState.payload?.system?.unified_memory ? 'Unified Memory' : 'Discrete'}
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Filters */}
        <Card className="border-border/50">
          <CardContent className="p-4 grid grid-cols-2 lg:grid-cols-6 gap-4 items-end">
            <div className="space-y-1.5 col-span-2 lg:col-span-1">
              <Label className="text-xs">Search</Label>
              <Input 
                placeholder="Model, provider..." 
                value={filters.search} 
                onChange={(e) => handleFieldChange('search', e.target.value)}
                className="h-9"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Fit Filter</Label>
              <Select value={filters.minFit} onValueChange={(v) => handleFieldChange('minFit', v)}>
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>{FIT_OPTIONS.map(o => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Runtime</Label>
              <Select value={filters.runtime} onValueChange={(v) => handleFieldChange('runtime', v)}>
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>{RUNTIME_OPTIONS.map(o => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Use Case</Label>
              <Select value={filters.useCase} onValueChange={(v) => handleFieldChange('useCase', v)}>
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>{USE_CASE_OPTIONS.map(o => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Sort By</Label>
              <Select value={filters.sort} onValueChange={(v) => handleFieldChange('sort', v)}>
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>{SORT_OPTIONS.map(o => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}</SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Main Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="col-span-1 lg:col-span-2 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Models ({modelsState.returned} / {modelsState.total})</h3>
            </div>
            <div className="rounded-md border bg-card overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow className="bg-muted/50">
                    <TableHead>Model</TableHead>
                    <TableHead>Fit</TableHead>
                    <TableHead>Mode</TableHead>
                    <TableHead className="text-right">Score</TableHead>
                    <TableHead className="text-right">TPS</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {modelsState.loading && (
                    <TableRow><TableCell colSpan={5} className="text-center py-8 text-muted-foreground">Loading fits...</TableCell></TableRow>
                  )}
                  {!modelsState.loading && modelsState.models.length === 0 && (
                    <TableRow><TableCell colSpan={5} className="text-center py-8 text-muted-foreground">No matches found.</TableCell></TableRow>
                  )}
                  {!modelsState.loading && modelsState.models.map((model) => (
                    <TableRow 
                      key={model.name} 
                      className={`cursor-pointer transition-colors ${model.name === selectedModelName ? 'bg-primary/5' : ''}`}
                      onClick={() => setSelectedModelName(model.name)}
                    >
                      <TableCell className="font-medium text-sm">
                        <div className="flex flex-col">
                          <span>{model.name}</span>
                          <span className="text-xs text-muted-foreground">{round(model.params_b, 1)}B • {model.provider}</span>
                        </div>
                      </TableCell>
                      <TableCell><Badge variant={getFitBadgeVariant(model.fit_level)}>{model.fit_label}</Badge></TableCell>
                      <TableCell><Badge variant="outline" className="font-mono text-[10px] uppercase">{model.run_mode}</Badge></TableCell>
                      <TableCell className="text-right font-mono">{round(model.score, 1)}</TableCell>
                      <TableCell className="text-right font-mono text-muted-foreground">{round(model.estimated_tps, 1)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>

          <div className="col-span-1">
            <div className="sticky top-20">
              <Card className="h-full border-border/50 shadow-sm">
                {selectedModel ? (
                  <>
                    <CardHeader className="pb-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <CardTitle className="text-lg leading-tight">{selectedModel.name}</CardTitle>
                          <CardDescription className="mt-1">{selectedModel.provider}</CardDescription>
                        </div>
                        <Badge variant={getFitBadgeVariant(selectedModel.fit_level)}>{selectedModel.fit_label}</Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      <div className="grid grid-cols-2 gap-y-4 gap-x-2 text-sm">
                        <div>
                          <span className="text-muted-foreground text-xs block">Runtime</span>
                          <span className="font-medium">{selectedModel.runtime_label}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground text-xs block">Quant</span>
                          <span className="font-mono">{selectedModel.best_quant}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground text-xs block">Required Mem</span>
                          <span className="font-mono">{round(selectedModel.memory_required_gb, 2)}G</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground text-xs block">Available Mem</span>
                          <span className="font-mono">{round(selectedModel.memory_available_gb, 2)}G</span>
                        </div>
                      </div>

                      <div className="space-y-4 pt-4 border-t border-border/50">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center">
                          <BarChart className="w-3 h-3 mr-1" /> Score Analysis
                        </h4>
                        <div className="space-y-3">
                          <MetricBar label="Quality" value={selectedModel.score_components?.quality} />
                          <MetricBar label="Speed" value={selectedModel.score_components?.speed} />
                          <MetricBar label="Fit" value={selectedModel.score_components?.fit} />
                          <MetricBar label="Context" value={selectedModel.score_components?.context} />
                        </div>
                      </div>

                      <div className="rounded-md bg-muted/30 p-3 space-y-1">
                        <h4 className="text-xs font-semibold text-muted-foreground mb-2">Metrics</h4>
                        <div className="flex justify-between text-sm">
                          <span>Utilization</span>
                          <span className="font-mono">{round(selectedModel.utilization_pct, 1)}%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Context Size</span>
                          <span className="font-mono">{selectedModel.context_length?.toLocaleString() || '—'}</span>
                        </div>
                        <div className="flex justify-between text-sm font-medium pt-2 border-t border-muted-foreground/10">
                          <span>Final TPS</span>
                          <span className="text-primary">{round(selectedModel.estimated_tps, 1)}</span>
                        </div>
                      </div>

                      {selectedModel.notes && selectedModel.notes.length > 0 && (
                        <div className="pt-2">
                          <h4 className="text-xs font-semibold text-muted-foreground mb-2">Notes</h4>
                          <ul className="text-xs space-y-1 text-muted-foreground list-disc pl-4">
                            {selectedModel.notes.map((n, i) => <li key={i}>{n}</li>)}
                          </ul>
                        </div>
                      )}
                    </CardContent>
                  </>
                ) : (
                  <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">
                    Select a model to view details
                  </div>
                )}
              </Card>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
