// ── Tauri IPC ─────────────────────────────────────────────────────────────
const { invoke } = window.__TAURI__.core;

// ── State ─────────────────────────────────────────────────────────────────
let allModels = [];
let sortKey = 'score';
let sortAsc = false;
let selectedModel = null;

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadSystemInfo();
  await loadModels();
  setupEventListeners();
});

async function loadSystemInfo() {
  try {
    const info = await invoke('get_system_info');
    document.getElementById('sys-cpu').textContent = `${info.cpu} (${info.cores} cores)`;
    document.getElementById('sys-ram').textContent = `${info.ram_gb.toFixed(1)} GB`;
    document.getElementById('sys-gpu').textContent = info.gpu;
    document.getElementById('sys-vram').textContent = info.vram_gb
      ? `${info.vram_gb.toFixed(1)} GB${info.unified_memory ? ' (unified)' : ''}`
      : 'N/A';

    const ollamaEl = document.getElementById('sys-ollama');
    if (info.ollama_available) {
      ollamaEl.textContent = `✓ (${info.ollama_installed_count} installed)`;
      ollamaEl.className = 'sys-value available';
    } else {
      ollamaEl.textContent = '✗ Not running';
      ollamaEl.className = 'sys-value unavailable';
    }
  } catch (e) {
    console.error('Failed to load system info:', e);
  }
}

async function loadModels() {
  try {
    allModels = await invoke('get_model_fits');
    renderTable();
  } catch (e) {
    console.error('Failed to load models:', e);
  }
}

// ── Filtering ─────────────────────────────────────────────────────────────
function getFilteredModels() {
  const search = document.getElementById('search').value.toLowerCase();
  const fitFilter = document.getElementById('filter-fit').value;
  const catFilter = document.getElementById('filter-category').value;
  const installedOnly = document.getElementById('filter-installed').checked;

  return allModels.filter(m => {
    if (search && !m.name.toLowerCase().includes(search) &&
        !m.provider.toLowerCase().includes(search)) return false;
    if (fitFilter !== 'all' && m.fit_level !== fitFilter) return false;
    if (catFilter !== 'all' && m.category !== catFilter) return false;
    if (installedOnly && !m.installed) return false;
    return true;
  });
}

// ── Sorting ───────────────────────────────────────────────────────────────
function sortModels(models) {
  return [...models].sort((a, b) => {
    let va = a[sortKey], vb = b[sortKey];
    if (typeof va === 'string') {
      va = va.toLowerCase(); vb = vb.toLowerCase();
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    if (typeof va === 'boolean') { va = va ? 1 : 0; vb = vb ? 1 : 0; }
    return sortAsc ? va - vb : vb - va;
  });
}

// ── Rendering ─────────────────────────────────────────────────────────────
function renderTable() {
  const filtered = sortModels(getFilteredModels());
  document.getElementById('model-count').textContent = `${filtered.length} of ${allModels.length} models`;

  const tbody = document.getElementById('model-tbody');
  tbody.innerHTML = filtered.map(m => `
    <tr class="model-row ${selectedModel === m.name ? 'selected' : ''}" data-name="${m.name}">
      <td>
        <div class="model-name">${escapeHtml(m.name.split('/').pop())}</div>
        <div class="model-provider">${escapeHtml(m.provider)}</div>
      </td>
      <td>${m.params}</td>
      <td><strong>${m.score.toFixed(0)}</strong></td>
      <td><span class="fit-badge fit-${m.fit_level}">${m.fit_emoji} ${m.fit_level}</span></td>
      <td>${m.estimated_tps.toFixed(1)}</td>
      <td>${m.best_quant}</td>
      <td>${m.run_mode}</td>
      <td>${m.utilization_pct.toFixed(0)}%</td>
      <td>${(m.context_length / 1000).toFixed(0)}k</td>
      <td class="${m.installed ? 'installed-yes' : 'installed-no'}">${m.installed ? '✓' : '—'}</td>
    </tr>
  `).join('');

  // Click handlers
  tbody.querySelectorAll('.model-row').forEach(row => {
    row.addEventListener('click', () => showDetail(row.dataset.name));
  });
}

function showDetail(name) {
  selectedModel = name;
  const model = allModels.find(m => m.name === name);
  if (!model) return;

  const panel = document.getElementById('detail-panel');
  const content = document.getElementById('detail-content');

  content.innerHTML = `
    <div class="detail-header">
      <h2>${escapeHtml(model.name.split('/').pop())}</h2>
      <div class="provider">${escapeHtml(model.provider)} · ${escapeHtml(model.name)}</div>
    </div>

    <div class="detail-section">
      <h3>Fit Assessment</h3>
      <div class="detail-grid">
        <div class="detail-item">
          <span class="label">Fit Level</span>
          <span class="value"><span class="fit-badge fit-${model.fit_level}">${model.fit_emoji} ${model.fit_level}</span></span>
        </div>
        <div class="detail-item">
          <span class="label">Overall Score</span>
          <span class="value">${model.score.toFixed(1)}</span>
        </div>
        <div class="detail-item">
          <span class="label">Est. Speed</span>
          <span class="value">${model.estimated_tps.toFixed(1)} tok/s</span>
        </div>
        <div class="detail-item">
          <span class="label">Run Mode</span>
          <span class="value">${model.run_mode}</span>
        </div>
        <div class="detail-item">
          <span class="label">Quantization</span>
          <span class="value">${model.best_quant}</span>
        </div>
        <div class="detail-item">
          <span class="label">Context</span>
          <span class="value">${(model.context_length / 1000).toFixed(0)}k</span>
        </div>
      </div>
    </div>

    <div class="detail-section">
      <h3>Memory</h3>
      <div class="detail-grid">
        <div class="detail-item">
          <span class="label">Required</span>
          <span class="value">${model.memory_required_gb.toFixed(1)} GB</span>
        </div>
        <div class="detail-item">
          <span class="label">Available</span>
          <span class="value">${model.memory_available_gb.toFixed(1)} GB</span>
        </div>
      </div>
      <div style="margin-top: 8px">
        <div class="score-bar">
          <div class="score-bar-fill memory" style="width: ${Math.min(model.utilization_pct, 100)}%"></div>
        </div>
        <div class="score-bar-label">
          <span>Memory utilization</span>
          <span>${model.utilization_pct.toFixed(1)}%</span>
        </div>
      </div>
    </div>

    <div class="detail-section">
      <h3>Score Breakdown</h3>
      ${scoreBar('Memory', model.score_memory, 'memory')}
      ${scoreBar('Speed', model.score_speed, 'speed')}
      ${scoreBar('Quality', model.score_quality, 'quality')}
      ${scoreBar('Context', model.score_context, 'context')}
    </div>

    <div class="detail-section">
      <h3>Category</h3>
      <div class="detail-item">
        <span class="value">${escapeHtml(model.category)} — ${escapeHtml(model.use_case)}</span>
      </div>
    </div>

    ${model.notes.length > 0 ? `
      <div class="detail-section">
        <h3>Notes</h3>
        ${model.notes.map(n => `<div class="note">${escapeHtml(n)}</div>`).join('')}
      </div>
    ` : ''}

    <div class="detail-section">
      <h3>Status</h3>
      <div class="detail-item">
        <span class="value ${model.installed ? 'installed-yes' : 'installed-no'}">
          ${model.installed ? '✓ Installed' : '✗ Not installed'}
        </span>
      </div>
    </div>
  `;

  panel.classList.remove('hidden');
  panel.classList.add('visible');
  renderTable(); // update selected highlight
}

function scoreBar(label, value, cls) {
  const pct = Math.max(0, Math.min(100, (value / 30) * 100));
  return `
    <div class="score-bar-container">
      <div class="score-bar-label">
        <span>${label}</span>
        <span>${value.toFixed(1)}</span>
      </div>
      <div class="score-bar">
        <div class="score-bar-fill ${cls}" style="width: ${pct}%"></div>
      </div>
    </div>
  `;
}

// ── Event Listeners ───────────────────────────────────────────────────────
function setupEventListeners() {
  document.getElementById('search').addEventListener('input', renderTable);
  document.getElementById('filter-fit').addEventListener('change', renderTable);
  document.getElementById('filter-category').addEventListener('change', renderTable);
  document.getElementById('filter-installed').addEventListener('change', renderTable);

  document.getElementById('btn-refresh').addEventListener('click', async () => {
    await loadSystemInfo();
    await loadModels();
  });

  document.getElementById('detail-close').addEventListener('click', () => {
    selectedModel = null;
    document.getElementById('detail-panel').classList.remove('visible');
    document.getElementById('detail-panel').classList.add('hidden');
    renderTable();
  });

  // Column sorting
  document.querySelectorAll('th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (sortKey === key) {
        sortAsc = !sortAsc;
      } else {
        sortKey = key;
        sortAsc = false;
      }
      document.querySelectorAll('th').forEach(t => t.classList.remove('sorted', 'asc', 'desc'));
      th.classList.add('sorted', sortAsc ? 'asc' : 'desc');
      renderTable();
    });
  });

  // Keyboard: Escape closes detail
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && selectedModel) {
      selectedModel = null;
      document.getElementById('detail-panel').classList.remove('visible');
      renderTable();
    }
  });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
