const invoke = window.__TAURI_INTERNALS__
  ? window.__TAURI_INTERNALS__.invoke
  : async (cmd) => { console.warn('Tauri not available, cmd:', cmd); return null; };

let allFits = [];

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

async function loadSpecs() {
  try {
    const specs = await invoke('get_system_specs');
    if (!specs) return;
    document.getElementById('cpu-name').textContent = specs.cpu_name;
    document.getElementById('cpu-cores').textContent = specs.cpu_cores + ' cores';
    document.getElementById('ram').textContent = specs.total_ram_gb.toFixed(1) + ' GB';
    document.getElementById('gpu-name').textContent = specs.gpu_name || 'No GPU detected';
    document.getElementById('gpu-vram').textContent = specs.gpu_vram_gb > 0
      ? specs.gpu_vram_gb.toFixed(1) + ' GB VRAM (' + specs.gpu_backend + ')'
      : '';
  } catch (e) {
    console.error('Failed to load specs:', e);
    document.getElementById('cpu-name').textContent = 'Error loading specs';
  }
}

function fitClass(level) {
  switch (level) {
    case 'Perfect': return 'fit-perfect';
    case 'Good': return 'fit-good';
    case 'Marginal': return 'fit-marginal';
    default: return 'fit-tight';
  }
}

function modeClass(mode) {
  switch (mode) {
    case 'GPU': return 'mode-gpu';
    case 'MoE Offload': return 'mode-moe';
    case 'CPU Offload': return 'mode-cpuoffload';
    default: return 'mode-cpuonly';
  }
}

function renderModels(fits) {
  const tbody = document.getElementById('models-body');
  if (!fits || fits.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="loading">No models found</td></tr>';
    return;
  }
  tbody.innerHTML = fits.map(f => `
    <tr>
      <td><strong>${esc(f.name)}</strong></td>
      <td>${esc(f.params_b.toFixed(1))}B</td>
      <td>${esc(f.quant)}</td>
      <td class="${fitClass(f.fit_level)}">${esc(f.fit_level)}</td>
      <td class="${modeClass(f.run_mode)}">${esc(f.run_mode)}</td>
      <td>${esc(f.score.toFixed(0))}</td>
      <td>${esc(f.memory_required_gb.toFixed(1))} GB</td>
      <td>${esc(f.estimated_tps.toFixed(1))}</td>
      <td>${esc(f.use_case)}</td>
    </tr>
  `).join('');
}

function applyFilters() {
  const search = document.getElementById('search').value.toLowerCase();
  const fitFilter = document.getElementById('fit-filter').value;

  let filtered = allFits;
  if (search) {
    filtered = filtered.filter(f => f.name.toLowerCase().includes(search));
  }
  if (fitFilter !== 'all') {
    filtered = filtered.filter(f => f.fit_level === fitFilter);
  }
  renderModels(filtered);
}

async function loadModels() {
  try {
    allFits = await invoke('get_model_fits') || [];
    applyFilters();
  } catch (e) {
    console.error('Failed to load models:', e);
    document.getElementById('models-body').innerHTML =
      '<tr><td colspan="9" class="loading">Error loading models</td></tr>';
  }
}

document.getElementById('search').addEventListener('input', applyFilters);
document.getElementById('fit-filter').addEventListener('change', applyFilters);

loadSpecs();
loadModels();
