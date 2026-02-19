use llmfit_core::fit::{self, ModelFit};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::ModelDatabase;
use llmfit_core::providers::{self, ModelProvider, OllamaProvider};
use serde::Serialize;
use std::collections::HashSet;
use std::sync::Mutex;
use tauri::State;

// ── Shared app state ──────────────────────────────────────────────────────

struct AppState {
    specs: SystemSpecs,
    fits: Vec<ModelFit>,
    ollama: OllamaProvider,
    ollama_installed: HashSet<String>,
    ollama_available: bool,
}

impl AppState {
    fn new() -> Self {
        let specs = SystemSpecs::detect();
        let db = ModelDatabase::new();
        let ollama = OllamaProvider::new();
        let ollama_available = ollama.is_available();
        let ollama_installed = if ollama_available {
            ollama.installed_models()
        } else {
            HashSet::new()
        };

        let mut fits: Vec<ModelFit> = db
            .get_all_models()
            .iter()
            .map(|m| {
                let mut f = ModelFit::analyze(m, &specs);
                f.installed = providers::is_model_installed(&m.name, &ollama_installed);
                f
            })
            .collect();
        fits = fit::rank_models_by_fit(fits);

        Self {
            specs,
            fits,
            ollama,
            ollama_installed,
            ollama_available,
        }
    }
}

// ── JSON response types ───────────────────────────────────────────────────

#[derive(Serialize)]
struct SystemInfo {
    cpu: String,
    cores: usize,
    ram_gb: f64,
    gpu: String,
    gpu_backend: String,
    vram_gb: Option<f64>,
    unified_memory: bool,
    ollama_available: bool,
    ollama_installed_count: usize,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    provider: String,
    params: String,
    score: f64,
    fit_level: String,
    fit_emoji: String,
    estimated_tps: f64,
    best_quant: String,
    run_mode: String,
    use_case: String,
    category: String,
    memory_required_gb: f64,
    memory_available_gb: f64,
    utilization_pct: f64,
    context_length: u32,
    installed: bool,
    notes: Vec<String>,
    score_fit: f64,
    score_speed: f64,
    score_quality: f64,
    score_context: f64,
}

impl From<&ModelFit> for ModelInfo {
    fn from(f: &ModelFit) -> Self {
        Self {
            name: f.model.name.clone(),
            provider: f.model.provider.clone(),
            params: f.model.parameter_count.clone(),
            score: f.score,
            fit_level: format!("{:?}", f.fit_level),
            fit_emoji: f.fit_emoji().to_string(),
            estimated_tps: f.estimated_tps,
            best_quant: f.best_quant.clone(),
            run_mode: f.run_mode_text().to_string(),
            use_case: f.model.use_case.clone(),
            category: f.use_case.label().to_string(),
            memory_required_gb: f.memory_required_gb,
            memory_available_gb: f.memory_available_gb,
            utilization_pct: f.utilization_pct,
            context_length: f.model.context_length,
            installed: f.installed,
            notes: f.notes.clone(),
            score_fit: f.score_components.fit,
            score_speed: f.score_components.speed,
            score_quality: f.score_components.quality,
            score_context: f.score_components.context,
        }
    }
}

fn build_system_info(s: &AppState) -> SystemInfo {
    SystemInfo {
        cpu: s.specs.cpu_name.clone(),
        cores: s.specs.total_cpu_cores,
        ram_gb: s.specs.total_ram_gb,
        gpu: s.specs.gpu_name.clone().unwrap_or_else(|| "None".into()),
        gpu_backend: format!("{:?}", s.specs.backend),
        vram_gb: s.specs.gpu_vram_gb,
        unified_memory: s.specs.unified_memory,
        ollama_available: s.ollama_available,
        ollama_installed_count: s.ollama_installed.len(),
    }
}

// ── Tauri commands ────────────────────────────────────────────────────────

#[tauri::command]
fn get_system_info(state: State<Mutex<AppState>>) -> Result<SystemInfo, String> {
    let s = state.lock().map_err(|e| e.to_string())?;
    Ok(build_system_info(&s))
}

#[tauri::command]
fn get_model_fits(state: State<Mutex<AppState>>) -> Result<Vec<ModelInfo>, String> {
    let s = state.lock().map_err(|e| e.to_string())?;
    Ok(s.fits.iter().map(ModelInfo::from).collect())
}

#[tauri::command]
fn get_model_detail(state: State<Mutex<AppState>>, name: String) -> Result<Option<ModelInfo>, String> {
    let s = state.lock().map_err(|e| e.to_string())?;
    Ok(s.fits
        .iter()
        .find(|f| f.model.name == name)
        .map(ModelInfo::from))
}

#[tauri::command]
fn refresh_installed(state: State<Mutex<AppState>>) -> Result<SystemInfo, String> {
    let mut s = state.lock().map_err(|e| e.to_string())?;
    s.ollama_installed = s.ollama.installed_models();
    s.ollama_available = s.ollama.is_available();
    let installed = s.ollama_installed.clone();
    for f in &mut s.fits {
        f.installed = providers::is_model_installed(&f.model.name, &installed);
    }
    Ok(build_system_info(&s))
}

// ── Entry point ───────────────────────────────────────────────────────────

fn main() {
    tauri::Builder::default()
        .manage(Mutex::new(AppState::new()))
        .invoke_handler(tauri::generate_handler![
            get_system_info,
            get_model_fits,
            get_model_detail,
            refresh_installed,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
