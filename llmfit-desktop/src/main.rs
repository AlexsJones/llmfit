#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use llmfit_core::fit::{FitLevel, ModelFit, RunMode};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::ModelDatabase;
use serde::Serialize;

#[derive(Serialize)]
struct SystemInfo {
    total_ram_gb: f64,
    cpu_name: String,
    cpu_cores: usize,
    gpu_name: String,
    gpu_vram_gb: f64,
    gpu_backend: String,
}

#[derive(Serialize)]
struct ModelFitInfo {
    name: String,
    params_b: f64,
    quant: String,
    fit_level: String,
    run_mode: String,
    score: f64,
    memory_required_gb: f64,
    memory_available_gb: f64,
    estimated_tps: f64,
    use_case: String,
    notes: Vec<String>,
}

#[tauri::command]
fn get_system_specs() -> Result<SystemInfo, String> {
    let specs = SystemSpecs::detect();
    Ok(SystemInfo {
        total_ram_gb: specs.total_ram_gb,
        cpu_name: specs.cpu_name.clone(),
        cpu_cores: specs.total_cpu_cores,
        gpu_name: specs
            .gpus
            .first()
            .map(|g| g.name.clone())
            .unwrap_or_default(),
        gpu_vram_gb: specs.gpus.first().and_then(|g| g.vram_gb).unwrap_or(0.0),
        gpu_backend: specs
            .gpus
            .first()
            .map(|g| format!("{:?}", g.backend))
            .unwrap_or_else(|| "None".to_string()),
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
            estimated_tps: f.estimated_tps,
            use_case: format!("{:?}", f.use_case),
            notes: f.notes.clone(),
        })
        .collect())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![get_system_specs, get_model_fits])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
