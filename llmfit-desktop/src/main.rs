#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use llmfit_core::fit::{
    FitLevel, InferenceRuntime, ModelFit, RunMode, backend_compatible, rank_models_by_fit,
};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::ModelDatabase;
use llmfit_core::plan::{PlanRequest, estimate_model_plan};
use llmfit_core::providers::{
    DockerModelRunnerProvider, LlamaCppProvider, LmStudioProvider, MlxProvider, ModelProvider,
    OllamaProvider, PullEvent, VllmProvider,
};
use tauri::State;

// ── Shared app state ──────────────────────────────────────────────────────

struct AppState {
    specs: SystemSpecs,
    models_db: ModelDatabase,
    ollama: OllamaProvider,
    pull_handle: Mutex<Option<PullHandleHolder>>,
}

struct PullHandleHolder {
    handle: llmfit_core::providers::PullHandle,
}

// ── "What if" hardware simulation ─────────────────────────────────────────

#[derive(Clone, serde::Deserialize)]
struct SimulationParams {
    ram_gb: Option<f64>,
    vram_gb: Option<f64>,
    cpu_cores: Option<usize>,
}

fn apply_simulation(mut specs: SystemSpecs, sim: SimulationParams) -> SystemSpecs {
    if let Some(ram) = sim.ram_gb {
        specs = specs.with_ram_override(ram);
    }
    if let Some(vram) = sim.vram_gb {
        specs = specs.with_gpu_memory_override(vram);
    }
    if let Some(cores) = sim.cpu_cores {
        if cores > 0 {
            specs = specs.with_cpu_core_override(cores);
        }
    }
    specs
}

// ── Build response models matching the web API shapes ─────────────────────

fn system_json(specs: &SystemSpecs) -> serde_json::Value {
    let gpus_json: Vec<serde_json::Value> = specs
        .gpus
        .iter()
        .map(|g| {
            serde_json::json!({
                "name": g.name,
                "vram_gb": g.vram_gb.map(round2),
                "backend": g.backend.label(),
                "count": g.count,
                "unified_memory": g.unified_memory,
                "memory_bandwidth_gbps": llmfit_core::hardware::gpu_memory_bandwidth_gbps(&g.name),
            })
        })
        .collect();

    serde_json::json!({
        "total_ram_gb": round2(specs.total_ram_gb),
        "available_ram_gb": round2(specs.available_ram_gb),
        "cpu_cores": specs.total_cpu_cores,
        "cpu_name": specs.cpu_name,
        "has_gpu": specs.has_gpu,
        "gpu_vram_gb": specs.gpu_vram_gb.map(round2),
        "gpu_available_gb": specs.gpu_available_gb.map(round2),
        "gpu_name": specs.gpu_name,
        "gpu_count": specs.gpu_count,
        "unified_memory": specs.unified_memory,
        "backend": specs.backend.label(),
        "gpus": gpus_json,
    })
}

fn fit_code(fit_level: FitLevel) -> &'static str {
    match fit_level {
        FitLevel::Perfect => "perfect",
        FitLevel::Good => "good",
        FitLevel::Marginal => "marginal",
        FitLevel::TooTight => "too_tight",
    }
}

fn mode_code(run_mode: RunMode) -> &'static str {
    match run_mode {
        RunMode::Gpu => "gpu",
        RunMode::TensorParallel => "tensor_parallel",
        RunMode::MoeOffload => "moe_offload",
        RunMode::CpuOffload => "cpu_offload",
        RunMode::CpuOnly => "cpu_only",
    }
}

fn runtime_code(runtime: InferenceRuntime) -> &'static str {
    match runtime {
        InferenceRuntime::Mlx => "mlx",
        InferenceRuntime::LlamaCpp => "llamacpp",
        InferenceRuntime::Vllm => "vllm",
        InferenceRuntime::Unsupported => "unsupported",
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn fit_to_json(fit: &ModelFit) -> serde_json::Value {
    serde_json::json!({
        "name": fit.model.name,
        "provider": fit.model.provider,
        "parameter_count": fit.model.parameter_count,
        "params_b": round2(fit.model.params_b()),
        "context_length": fit.model.context_length,
        "usable_context": fit.usable_context,
        "effective_context_length": fit.effective_context_length,
        "use_case": fit.model.use_case,
        "category": fit.use_case.label(),
        "release_date": fit.model.release_date,
        "is_moe": fit.model.is_moe,
        "fit_level": fit_code(fit.fit_level),
        "fit_label": fit.fit_text(),
        "run_mode": mode_code(fit.run_mode),
        "run_mode_label": fit.run_mode_text(),
        "score": round1(fit.score),
        "score_components": {
            "quality": round1(fit.score_components.quality),
            "speed": round1(fit.score_components.speed),
            "fit": round1(fit.score_components.fit),
            "context": round1(fit.score_components.context),
        },
        "estimated_tps": round1(fit.estimated_tps),
        "runtime": runtime_code(fit.runtime),
        "runtime_label": fit.runtime_text(),
        "best_quant": fit.best_quant,
        "memory_required_gb": round2(fit.memory_required_gb),
        "memory_available_gb": round2(fit.memory_available_gb),
        "moe_offloaded_gb": fit.moe_offloaded_gb.map(round2),
        "total_memory_gb": round2(fit.memory_required_gb + fit.moe_offloaded_gb.unwrap_or(0.0)),
        "utilization_pct": round1(fit.utilization_pct),
        "notes": fit.notes,
        "gguf_sources": fit.model.gguf_sources,
        "capabilities": fit.model.capabilities,
        "capability_ids": fit.model.capabilities,
        "license": fit.model.license,
        "supports_tp": fit.model.valid_tp_sizes(),
        "installed": fit.installed,
        "disk_size_gb": round2(fit.model.estimate_disk_gb(&fit.best_quant)),
        "ollama_name": llmfit_core::providers::ollama_pull_tag(&fit.model.name),
        "estimate_basis": fit.estimate_basis,
        "verify_command": None::<String>,
        "measured_tps": fit.measured_tps,
    })
}

// ── Tauri commands ────────────────────────────────────────────────────────

#[tauri::command]
fn get_system_specs(state: State<'_, AppState>, sim: Option<SimulationParams>) -> serde_json::Value {
    let specs = match sim {
        Some(p) => apply_simulation(state.specs.clone(), p),
        None => state.specs.clone(),
    };

    serde_json::json!({
        "node": {
            "name": whoami(),
            "os": std::env::consts::OS,
        },
        "system": system_json(&specs),
    })
}

#[tauri::command]
fn get_models(state: State<'_, AppState>, sim: Option<SimulationParams>) -> serde_json::Value {
    let specs = match sim {
        Some(p) => apply_simulation(state.specs.clone(), p),
        None => state.specs.clone(),
    };

    let all_models = state.models_db.get_all_models();
    let fits: Vec<ModelFit> = all_models
        .iter()
        .filter(|m| backend_compatible(m, &specs))
        .map(|m| ModelFit::analyze(m, &specs))
        .collect();

    let ranked = rank_models_by_fit(fits);
    let total = ranked.len();

    let models: Vec<serde_json::Value> = ranked.iter().map(fit_to_json).collect();

    serde_json::json!({
        "node": {
            "name": whoami(),
            "os": std::env::consts::OS,
        },
        "system": system_json(&specs),
        "total_models": total,
        "returned_models": total,
        "filters": {
            "limit": null,
            "min_fit": null,
            "sort": "score",
        },
        "models": models,
    })
}

#[tauri::command]
fn search_models(state: State<'_, AppState>, query: String, sim: Option<SimulationParams>) -> serde_json::Value {
    let specs = match sim {
        Some(p) => apply_simulation(state.specs.clone(), p),
        None => state.specs.clone(),
    };

    let q = query.to_lowercase();
    let all_models = state.models_db.get_all_models();
    let fits: Vec<ModelFit> = all_models
        .iter()
        .filter(|m| backend_compatible(m, &specs))
        .filter(|m| {
            m.name.to_lowercase().contains(&q)
                || m.provider.to_lowercase().contains(&q)
                || m.parameter_count.to_lowercase().contains(&q)
        })
        .map(|m| ModelFit::analyze(m, &specs))
        .collect();

    let ranked = rank_models_by_fit(fits);
    let total = ranked.len();

    let models: Vec<serde_json::Value> = ranked.iter().map(fit_to_json).collect();

    serde_json::json!({
        "node": {
            "name": whoami(),
            "os": std::env::consts::OS,
        },
        "system": system_json(&specs),
        "total_models": total,
        "returned_models": total,
        "filters": {
            "search": query,
        },
        "models": models,
    })
}

#[tauri::command]
fn get_runtimes() -> serde_json::Value {
    let mut runtimes = Vec::new();

    for (name, provider) in [
        ("ollama", Box::new(OllamaProvider::new()) as Box<dyn ModelProvider>),
        ("mlx", Box::new(MlxProvider::new()) as Box<dyn ModelProvider>),
        ("llamacpp", Box::new(LlamaCppProvider::new()) as Box<dyn ModelProvider>),
        ("docker_model_runner", Box::new(DockerModelRunnerProvider::new()) as Box<dyn ModelProvider>),
        ("lmstudio", Box::new(LmStudioProvider::new()) as Box<dyn ModelProvider>),
        ("vllm", Box::new(VllmProvider::new()) as Box<dyn ModelProvider>),
    ] {
        let available = provider.is_available();
        runtimes.push(serde_json::json!({
            "name": name,
            "installed": available,
        }));
    }

    serde_json::json!({ "runtimes": runtimes, "warnings": [] })
}

#[tauri::command]
fn get_installed(state: State<'_, AppState>) -> serde_json::Value {
    let mut models = Vec::new();

    if state.ollama.is_available() {
        for name in state.ollama.installed_models() {
            models.push(serde_json::json!({ "name": name, "runtime": "ollama" }));
        }
    }

    let providers: [(&str, Box<dyn ModelProvider>); 5] = [
        ("mlx", Box::new(MlxProvider::new())),
        ("llamacpp", Box::new(LlamaCppProvider::new())),
        ("docker_model_runner", Box::new(DockerModelRunnerProvider::new())),
        ("lmstudio", Box::new(LmStudioProvider::new())),
        ("vllm", Box::new(VllmProvider::new())),
    ];

    for (name, provider) in providers {
        if provider.is_available() {
            for model_name in provider.installed_models() {
                models.push(serde_json::json!({ "name": model_name, "runtime": name }));
            }
        }
    }

    serde_json::json!({ "models": models, "warnings": [] })
}

#[tauri::command]
fn start_pull(model_tag: String, state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let handle = state.ollama.start_pull(&model_tag)?;
    let mut pull = state.pull_handle.lock().map_err(|e| e.to_string())?;
    *pull = Some(PullHandleHolder { handle });

    let id = format!("dl-{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0));

    Ok(serde_json::json!({
        "id": id,
        "model": model_tag,
        "runtime": "ollama",
        "status": "pulling",
    }))
}

#[tauri::command]
fn poll_pull(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let pull = state.pull_handle.lock().map_err(|e| e.to_string())?;
    let result = if let Some(ref holder) = *pull {
        match holder.handle.receiver.try_recv() {
            Ok(PullEvent::Progress { status, percent }) => serde_json::json!({
                "status": status,
                "progress_pct": percent,
                "done": false,
                "error": null,
            }),
            Ok(PullEvent::Done) => serde_json::json!({
                "status": "done",
                "progress_pct": 100.0,
                "done": true,
                "error": null,
            }),
            Ok(PullEvent::Error(e)) => serde_json::json!({
                "status": "error",
                "progress_pct": null,
                "done": true,
                "error": e,
            }),
            Err(std::sync::mpsc::TryRecvError::Empty) => serde_json::json!({
                "status": "downloading",
                "progress_pct": null,
                "done": false,
                "error": null,
            }),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => serde_json::json!({
                "status": "done",
                "progress_pct": 100.0,
                "done": true,
                "error": null,
            }),
        }
    } else {
        serde_json::json!({
            "status": "idle",
            "progress_pct": null,
            "done": true,
            "error": null,
        })
    };
    Ok(result)
}

#[tauri::command]
fn is_ollama_available(state: State<'_, AppState>) -> bool {
    state.ollama.is_available()
}

#[tauri::command]
fn estimate_plan(
    state: State<'_, AppState>,
    model_name: String,
    context: u32,
    quant: Option<String>,
    kv_quant_str: Option<String>,
    target_tps: Option<f64>,
    sim: Option<SimulationParams>,
) -> Result<serde_json::Value, String> {
    let specs = match sim {
        Some(p) => apply_simulation(state.specs.clone(), p),
        None => state.specs.clone(),
    };

    let model = state
        .models_db
        .get_all_models()
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case(&model_name))
        .ok_or_else(|| format!("Model '{}' not found", model_name))?
        .clone();

    let kv_quant = match kv_quant_str.as_deref() {
        Some(s) => Some(
            llmfit_core::models::KvQuant::parse(s)
                .ok_or_else(|| format!("Unsupported kv_quant '{}'. Valid: fp16, fp8, q8_0, q4_0, tq", s))?,
        ),
        None => None,
    };

    let request = PlanRequest {
        context,
        quant,
        target_tps,
        kv_quant,
    };

    let estimate = estimate_model_plan(&model, &request, &specs)
        .map_err(|e| format!("Plan estimation failed: {}", e))?;

    Ok(serde_json::to_value(&estimate).map_err(|e| format!("Serialization error: {}", e))?)
}

#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    let result = if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(["/c", "start", "", &url])
            .spawn()
            .map(|_| ())
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open")
            .arg(&url)
            .spawn()
            .map(|_| ())
    } else {
        std::process::Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .map(|_| ())
    };
    result.map_err(|e| format!("Failed to open URL: {}", e))
}

fn whoami() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "local".to_string())
}

// ── App entrypoint ────────────────────────────────────────────────────────

fn main() {
    let specs = SystemSpecs::detect();
    let db = ModelDatabase::new();

    tauri::Builder::default()
        .manage(AppState {
            specs,
            models_db: db,
            ollama: OllamaProvider::new(),
            pull_handle: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            get_system_specs,
            get_models,
            search_models,
            get_runtimes,
            get_installed,
            start_pull,
            poll_pull,
            is_ollama_available,
            estimate_plan,
            open_url,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
