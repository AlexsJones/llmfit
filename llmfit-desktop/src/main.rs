#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use llmfit_core::fit::{FitLevel, InferenceRuntime, ModelFit, RunMode};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::ModelDatabase;
use llmfit_core::providers::{ModelProvider, OllamaProvider, PullEvent};
use serde::Serialize;
use std::sync::Mutex;
use tauri::State;

#[derive(Serialize)]
struct GpuInfoJs {
    name: String,
    vram_gb: Option<f64>,
    backend: String,
    count: u32,
    unified_memory: bool,
}

#[derive(Serialize)]
struct SystemInfo {
    total_ram_gb: f64,
    available_ram_gb: f64,
    cpu_name: String,
    cpu_cores: usize,
    gpus: Vec<GpuInfoJs>,
    unified_memory: bool,
}

#[derive(Serialize, Clone)]
struct ModelFitInfo {
    name: String,
    params_b: f64,
    quant: String,
    fit_level: String,
    run_mode: String,
    score: f64,
    memory_required_gb: f64,
    memory_available_gb: f64,
    utilization_pct: f64,
    estimated_tps: f64,
    use_case: String,
    runtime: String,
    installed: bool,
    notes: Vec<String>,
    release_date: Option<String>,
}

#[derive(Serialize)]
struct PullStatus {
    status: String,
    percent: Option<f64>,
    done: bool,
    error: Option<String>,
}

struct AppState {
    ollama: OllamaProvider,
    pull_handle: Mutex<Option<llmfit_core::providers::PullHandle>>,
}

#[tauri::command]
fn get_system_specs() -> Result<SystemInfo, String> {
    let specs = SystemSpecs::detect();
    let gpus = specs
        .gpus
        .iter()
        .map(|g| GpuInfoJs {
            name: g.name.clone(),
            vram_gb: g.vram_gb,
            backend: format!("{:?}", g.backend),
            count: g.count,
            unified_memory: g.unified_memory,
        })
        .collect();
    Ok(SystemInfo {
        total_ram_gb: specs.total_ram_gb,
        available_ram_gb: specs.available_ram_gb,
        cpu_name: specs.cpu_name.clone(),
        cpu_cores: specs.total_cpu_cores,
        gpus,
        unified_memory: specs.unified_memory,
    })
}

#[tauri::command]
fn get_model_fits() -> Result<Vec<ModelFitInfo>, String> {
    let specs = SystemSpecs::detect();
    let db = ModelDatabase::new();

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .map(|m| ModelFit::analyze(m, &specs))
        .collect();

    fits = llmfit_core::fit::rank_models_by_fit(fits);

    Ok(fits
        .into_iter()
        .map(|f| ModelFitInfo {
            name: f.model.name.clone(),
            params_b: f.model.parameters_raw.unwrap_or(0) as f64 / 1e9,
            quant: f.best_quant.clone(),
            fit_level: match f.fit_level {
                FitLevel::Perfect => "Perfect".to_string(),
                FitLevel::Good => "Good".to_string(),
                FitLevel::Marginal => "Marginal".to_string(),
                FitLevel::TooTight => "Too Tight".to_string(),
            },
            run_mode: match f.run_mode {
                RunMode::Gpu => "GPU".to_string(),
                RunMode::CpuOffload => "CPU Offload".to_string(),
                RunMode::CpuOnly => "CPU Only".to_string(),
                RunMode::MoeOffload => "MoE Offload".to_string(),
            },
            score: f.score,
            memory_required_gb: f.memory_required_gb,
            memory_available_gb: f.memory_available_gb,
            utilization_pct: f.utilization_pct,
            estimated_tps: f.estimated_tps,
            use_case: format!("{:?}", f.use_case),
            runtime: match f.runtime {
                InferenceRuntime::LlamaCpp => "llama.cpp".to_string(),
                InferenceRuntime::Mlx => "MLX".to_string(),
            },
            installed: f.installed,
            notes: f.notes.clone(),
            release_date: f.model.release_date.clone(),
        })
        .collect())
}

#[tauri::command]
fn start_pull(model_tag: String, state: State<'_, AppState>) -> Result<String, String> {
    let handle = state.ollama.start_pull(&model_tag)?;
    let mut pull = state.pull_handle.lock().map_err(|e| e.to_string())?;
    *pull = Some(handle);
    Ok("started".to_string())
}

#[tauri::command]
fn poll_pull(state: State<'_, AppState>) -> Result<PullStatus, String> {
    let pull = state.pull_handle.lock().map_err(|e| e.to_string())?;
    if let Some(ref handle) = *pull {
        match handle.receiver.try_recv() {
            Ok(PullEvent::Progress { status, percent }) => Ok(PullStatus {
                status,
                percent,
                done: false,
                error: None,
            }),
            Ok(PullEvent::Done) => Ok(PullStatus {
                status: "Complete".to_string(),
                percent: Some(100.0),
                done: true,
                error: None,
            }),
            Ok(PullEvent::Error(e)) => Ok(PullStatus {
                status: "Error".to_string(),
                percent: None,
                done: true,
                error: Some(e),
            }),
            Err(std::sync::mpsc::TryRecvError::Empty) => Ok(PullStatus {
                status: "Waiting...".to_string(),
                percent: None,
                done: false,
                error: None,
            }),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Ok(PullStatus {
                status: "Complete".to_string(),
                percent: Some(100.0),
                done: true,
                error: None,
            }),
        }
    } else {
        Err("No pull in progress".to_string())
    }
}

#[tauri::command]
fn is_ollama_available(state: State<'_, AppState>) -> bool {
    state.ollama.is_available()
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            ollama: OllamaProvider::new(),
            pull_handle: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            get_system_specs,
            get_model_fits,
            start_pull,
            poll_pull,
            is_ollama_available,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
