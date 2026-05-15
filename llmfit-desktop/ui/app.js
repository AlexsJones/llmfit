const invoke = window.__TAURI_INTERNALS__
  ? window.__TAURI_INTERNALS__.invoke
  : async (cmd) => { console.warn('Tauri not available, cmd:', cmd); return null; };

const {
  t,
  getLocale,
  setLocale,
  subscribe,
  applyStaticTranslations,
  translateFitLevel,
  translateRunMode,
  translateUseCase,
} = window.llmfitI18n;

let allFits = [];
let ollamaAvailable = false;
let pullInterval = null;
let lastSpecs = null;
let currentModalFit = null;

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderSpecs(specs) {
  if (!specs) return;

  document.getElementById('cpu-name').textContent = specs.cpu_name;
  document.getElementById('cpu-cores').textContent = t('system.cores', { count: specs.cpu_cores });
  document.getElementById('ram-total').textContent = specs.total_ram_gb.toFixed(1) + ' GB';
  document.getElementById('ram-available').textContent = specs.available_ram_gb.toFixed(1) + ' GB';

  const container = document.getElementById('gpus-container');
  container.innerHTML = '';

  if (specs.gpus.length === 0) {
    const card = document.createElement('div');
    card.className = 'spec-card';
    card.innerHTML = '<span class="spec-label">' + esc(t('system.gpu')) + '</span>' +
      '<span class="spec-value">' + esc(t('system.noGpu')) + '</span>';
    container.appendChild(card);
  } else {
    specs.gpus.forEach((gpu, i) => {
      const card = document.createElement('div');
      card.className = 'spec-card';
      const label = specs.gpus.length > 1 ? t('system.gpuIndexed', { index: i + 1 }) : t('system.gpu');
      const countStr = gpu.count > 1 ? ' ×' + gpu.count : '';
      const vramStr = gpu.vram_gb != null ? gpu.vram_gb.toFixed(1) + ' GB VRAM' : t('system.sharedMemory');
      const backendStr = gpu.backend !== 'None' ? gpu.backend : '';
      const details = [vramStr, backendStr].filter(Boolean).join(' · ');
      card.innerHTML = '<span class="spec-label">' + esc(label) + '</span>' +
        '<span class="spec-value">' + esc(gpu.name + countStr) + '</span>' +
        '<span class="spec-detail">' + esc(details) + '</span>';
      container.appendChild(card);
    });
  }

  if (specs.unified_memory) {
    const archCard = document.getElementById('memory-arch-card');
    archCard.style.display = '';
    document.getElementById('memory-arch').textContent = t('system.unifiedMemory');
  }
}

async function loadSpecs() {
  try {
    const specs = await invoke('get_system_specs');
    if (!specs) return;
    lastSpecs = specs;
    renderSpecs(specs);
  } catch (e) {
    console.error('Failed to load specs:', e);
    document.getElementById('cpu-name').textContent = t('system.errorLoading');
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

function showModal(fit) {
  currentModalFit = fit;
  const modal = document.getElementById('model-modal');
  const body = document.getElementById('modal-body');

  const memBar = Math.min(fit.utilization_pct, 100);
  const memBarClass = fit.utilization_pct > 95 ? 'bar-red' : fit.utilization_pct > 80 ? 'bar-yellow' : 'bar-green';

  let notesHtml = '';
  if (fit.notes && fit.notes.length > 0) {
    notesHtml = '<div class="modal-section"><h4>' + esc(t('desktop.notes')) + '</h4><ul>' +
      fit.notes.map(n => '<li>' + esc(n) + '</li>').join('') +
      '</ul></div>';
  }

  const installedBadge = fit.installed
    ? '<span class="badge badge-installed">' + esc(t('desktop.installed')) + '</span>'
    : '<span class="badge badge-not-installed">' + esc(t('desktop.notInstalled')) + '</span>';

  const downloadBtn = (!fit.installed && ollamaAvailable)
    ? '<button class="btn-download">' + esc(t('desktop.downloadViaOllama')) + '</button>'
    : '';

  body.innerHTML = `
    <div class="modal-header-row">
      <h3>${esc(fit.name)}</h3>
      ${installedBadge}
    </div>

    <div class="modal-grid">
      <div class="modal-stat">
        <span class="stat-label">${esc(t('desktop.parameters'))}</span>
        <span class="stat-value">${esc(fit.params_b.toFixed(1))}B</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">${esc(t('desktop.quantization'))}</span>
        <span class="stat-value">${esc(fit.quant)}</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">${esc(t('desktop.runtime'))}</span>
        <span class="stat-value">${esc(fit.runtime)}</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">${esc(t('desktop.score'))}</span>
        <span class="stat-value">${esc(fit.score.toFixed(0))}/100</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">${esc(t('desktop.estSpeed'))}</span>
        <span class="stat-value">${esc(fit.estimated_tps.toFixed(1))} tok/s</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">${esc(t('desktop.useCase'))}</span>
        <span class="stat-value">${esc(translateUseCase(fit.use_case))}</span>
      </div>
    </div>

    <div class="modal-section">
      <h4>${esc(t('desktop.fitAnalysis'))}</h4>
      <div class="fit-row">
        <span class="${fitClass(fit.fit_level)}">${esc(translateFitLevel(fit.fit_level))}</span>
        <span class="fit-detail">${esc(translateRunMode(fit.run_mode))}</span>
      </div>
      <div class="mem-bar-container">
        <div class="mem-bar-label">
          <span>${esc(t('desktop.memorySummary', { required: fit.memory_required_gb.toFixed(1), available: fit.memory_available_gb.toFixed(1) }))}</span>
          <span>${esc(fit.utilization_pct.toFixed(0))}%</span>
        </div>
        <div class="mem-bar-track">
          <div class="mem-bar-fill ${memBarClass}" style="width: ${memBar}%"></div>
        </div>
      </div>
    </div>

    ${notesHtml}

    <div id="pull-status" class="pull-status" style="display:none">
      <div class="pull-status-text"></div>
      <div class="mem-bar-track">
        <div class="pull-bar-fill" style="width: 0%"></div>
      </div>
    </div>

    <div class="modal-actions">
      ${downloadBtn}
      <button class="btn-close" onclick="closeModal()">${esc(t('desktop.close'))}</button>
    </div>
  `;

  const dlBtn = body.querySelector('.btn-download');
  if (dlBtn) dlBtn.addEventListener('click', () => pullModel(fit.name));

  modal.classList.add('visible');
}

function closeModal() {
  currentModalFit = null;
  document.getElementById('model-modal').classList.remove('visible');
  if (pullInterval) {
    clearInterval(pullInterval);
    pullInterval = null;
  }
}
window.closeModal = closeModal;

async function pullModel(name) {
  const statusEl = document.getElementById('pull-status');
  const textEl = statusEl.querySelector('.pull-status-text');
  const barEl = statusEl.querySelector('.pull-bar-fill');
  const btn = document.querySelector('.btn-download');

  statusEl.style.display = '';
  if (btn) btn.disabled = true;
  textEl.textContent = t('desktop.startingDownload');

  try {
    await invoke('start_pull', { modelTag: name });

    pullInterval = setInterval(async () => {
      try {
        const s = await invoke('poll_pull');
        if (!s) return;
        textEl.textContent = s.status;
        if (s.percent != null) barEl.style.width = s.percent + '%';
        if (s.done) {
          clearInterval(pullInterval);
          pullInterval = null;
          if (s.error) {
            textEl.textContent = t('desktop.errorPrefix') + s.error;
            if (btn) btn.disabled = false;
          } else {
            textEl.textContent = t('desktop.downloadComplete');
            barEl.style.width = '100%';
            await loadModels();
          }
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
    }, 500);
  } catch (e) {
    textEl.textContent = t('desktop.errorPrefix') + e;
    if (btn) btn.disabled = false;
  }
}

function renderModels(fits) {
  const tbody = document.getElementById('models-body');
  if (!fits || fits.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="loading">' + esc(t('desktop.noModels')) + '</td></tr>';
    return;
  }
  tbody.innerHTML = fits.map((f, i) => `
    <tr class="model-row" data-index="${i}">
      <td><strong>${esc(f.name)}</strong>${f.installed ? ' <span class="installed-dot" title="' + esc(t('desktop.installed')) + '">●</span>' : ''}</td>
      <td>${esc(f.params_b.toFixed(1))}B</td>
      <td>${esc(f.quant)}</td>
      <td class="${fitClass(f.fit_level)}">${esc(translateFitLevel(f.fit_level))}</td>
      <td class="${modeClass(f.run_mode)}">${esc(translateRunMode(f.run_mode))}</td>
      <td>${esc(f.score.toFixed(0))}</td>
      <td>${esc(f.memory_required_gb.toFixed(1))} GB</td>
      <td>${esc(f.estimated_tps.toFixed(1))}</td>
      <td>${esc(translateUseCase(f.use_case))}</td>
    </tr>
  `).join('');

  const currentFits = fits;
  tbody.querySelectorAll('.model-row').forEach(row => {
    row.addEventListener('click', () => {
      const idx = parseInt(row.dataset.index, 10);
      showModal(currentFits[idx]);
    });
  });
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
      '<tr><td colspan="9" class="loading">' + esc(t('desktop.errorLoadingModels')) + '</td></tr>';
  }
}

function rerenderForLocale() {
  applyStaticTranslations();
  document.getElementById('locale-select').value = getLocale();
  if (lastSpecs) {
    renderSpecs(lastSpecs);
  }
  applyFilters();
  if (currentModalFit) {
    showModal(currentModalFit);
  }
}

document.getElementById('model-modal').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeModal();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

document.getElementById('search').addEventListener('input', applyFilters);
document.getElementById('fit-filter').addEventListener('change', applyFilters);
document.getElementById('locale-select').addEventListener('change', (e) => {
  setLocale(e.target.value);
});

subscribe(rerenderForLocale);

async function init() {
  applyStaticTranslations();
  document.getElementById('locale-select').value = getLocale();
  ollamaAvailable = await invoke('is_ollama_available') || false;
  loadSpecs();
  loadModels();
}

init();
