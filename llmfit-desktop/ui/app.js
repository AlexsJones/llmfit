const invoke = window.__TAURI_INTERNALS__
  ? window.__TAURI_INTERNALS__.invoke
  : async (cmd) => { console.warn('Tauri not available, cmd:', cmd); return null; };

let allFits = [];
let ollamaAvailable = false;
let pullInterval = null;

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
    document.getElementById('ram-total').textContent = specs.total_ram_gb.toFixed(1) + ' GB';
    document.getElementById('ram-available').textContent = specs.available_ram_gb.toFixed(1) + ' GB';

    const container = document.getElementById('gpus-container');
    container.innerHTML = '';

    if (specs.gpus.length === 0) {
      const card = document.createElement('div');
      card.className = 'spec-card';
      card.innerHTML = '<span class="spec-label">GPU</span>' +
        '<span class="spec-value">No GPU detected</span>';
      container.appendChild(card);
    } else {
      specs.gpus.forEach((gpu, i) => {
        const card = document.createElement('div');
        card.className = 'spec-card';
        const label = specs.gpus.length > 1 ? 'GPU ' + (i + 1) : 'GPU';
        const countStr = gpu.count > 1 ? ' ×' + gpu.count : '';
        const vramStr = gpu.vram_gb != null ? gpu.vram_gb.toFixed(1) + ' GB VRAM' : 'Shared memory';
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
      document.getElementById('memory-arch').textContent = 'Unified (CPU + GPU shared)';
    }
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

function showModal(fit) {
  const modal = document.getElementById('model-modal');
  const body = document.getElementById('modal-body');

  const memBar = Math.min(fit.utilization_pct, 100);
  const memBarClass = fit.utilization_pct > 95 ? 'bar-red' : fit.utilization_pct > 80 ? 'bar-yellow' : 'bar-green';

  let notesHtml = '';
  if (fit.notes && fit.notes.length > 0) {
    notesHtml = '<div class="modal-section"><h4>Notes</h4><ul>' +
      fit.notes.map(n => '<li>' + esc(n) + '</li>').join('') +
      '</ul></div>';
  }

  const installedBadge = fit.installed
    ? '<span class="badge badge-installed">Installed</span>'
    : '<span class="badge badge-not-installed">Not Installed</span>';

  const downloadBtn = (!fit.installed && ollamaAvailable)
    ? '<button class="btn-download">⬇ Download via Ollama</button>'
    : '';

  body.innerHTML = `
    <div class="modal-header-row">
      <h3>${esc(fit.name)}</h3>
      ${installedBadge}
    </div>

    <div class="modal-grid">
      <div class="modal-stat">
        <span class="stat-label">Parameters</span>
        <span class="stat-value">${esc(fit.params_b.toFixed(1))}B</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">Quantization</span>
        <span class="stat-value">${esc(fit.quant)}</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">Runtime</span>
        <span class="stat-value">${esc(fit.runtime)}</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">Score</span>
        <span class="stat-value">${esc(fit.score.toFixed(0))}/100</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">Est. Speed</span>
        <span class="stat-value">${esc(fit.estimated_tps.toFixed(1))} tok/s</span>
      </div>
      <div class="modal-stat">
        <span class="stat-label">Use Case</span>
        <span class="stat-value">${esc(fit.use_case)}</span>
      </div>
    </div>

    <div class="modal-section">
      <h4>Fit Analysis</h4>
      <div class="fit-row">
        <span class="${fitClass(fit.fit_level)}">${esc(fit.fit_level)}</span>
        <span class="fit-detail">${esc(fit.run_mode)}</span>
      </div>
      <div class="mem-bar-container">
        <div class="mem-bar-label">
          <span>Memory: ${esc(fit.memory_required_gb.toFixed(1))} / ${esc(fit.memory_available_gb.toFixed(1))} GB</span>
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
      <button class="btn-close" onclick="closeModal()">Close</button>
    </div>
  `;

  const dlBtn = body.querySelector('.btn-download');
  if (dlBtn) dlBtn.addEventListener('click', () => pullModel(fit.name));

  modal.classList.add('visible');
}

function closeModal() {
  document.getElementById('model-modal').classList.remove('visible');
  if (pullInterval) {
    clearInterval(pullInterval);
    pullInterval = null;
  }
}

async function pullModel(name) {
  const statusEl = document.getElementById('pull-status');
  const textEl = statusEl.querySelector('.pull-status-text');
  const barEl = statusEl.querySelector('.pull-bar-fill');
  const btn = document.querySelector('.btn-download');

  statusEl.style.display = '';
  if (btn) btn.disabled = true;
  textEl.textContent = 'Starting download...';

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
            textEl.textContent = 'Error: ' + s.error;
            if (btn) btn.disabled = false;
          } else {
            textEl.textContent = 'Download complete!';
            barEl.style.width = '100%';
            // Refresh model list
            await loadModels();
          }
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
    }, 500);
  } catch (e) {
    textEl.textContent = 'Error: ' + e;
    if (btn) btn.disabled = false;
  }
}

function renderModels(fits) {
  const tbody = document.getElementById('models-body');
  if (!fits || fits.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="loading">No models found</td></tr>';
    return;
  }
  tbody.innerHTML = fits.map((f, i) => `
    <tr class="model-row" data-index="${i}">
      <td><strong>${esc(f.name)}</strong>${f.installed ? ' <span class="installed-dot" title="Installed">●</span>' : ''}</td>
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

  // Attach click handlers
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
      '<tr><td colspan="9" class="loading">Error loading models</td></tr>';
  }
}

// Close modal on backdrop click
document.getElementById('model-modal').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeModal();
});

// Close modal on Escape
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

document.getElementById('search').addEventListener('input', applyFilters);
document.getElementById('fit-filter').addEventListener('change', applyFilters);

async function init() {
  ollamaAvailable = await invoke('is_ollama_available') || false;
  loadSpecs();
  loadModels();
}

init();
