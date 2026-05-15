use llmfit_core::fit::{
    FitLevel, InferenceRuntime, ModelFit, SortColumn, backend_compatible,
    rank_models_by_fit_opts_col,
};
use llmfit_core::hardware::{GpuBackend, SystemSpecs};
use llmfit_core::models::{LlmModel, ModelDatabase, UseCase};
use llmfit_core::plan::{PlanRequest, estimate_model_plan};
use llmfit_core::providers::{
    DockerModelRunnerProvider, LlamaCppProvider, LmStudioProvider, MlxProvider, ModelProvider,
    OllamaProvider, VllmProvider,
};
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::serve_shared;

// --- Tool input schemas ---

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct RecommendModelsParams {
    /// Maximum number of models to return (default: 10)
    pub limit: Option<usize>,
    /// Filter by use case: general, coding, reasoning, chat, multimodal, embedding
    pub use_case: Option<String>,
    /// Minimum fit level: perfect, good, marginal
    pub min_fit: Option<String>,
    /// Filter by runtime: mlx, llamacpp, vllm
    pub runtime: Option<String>,
    /// Filter by license string
    pub license: Option<String>,
    /// Sort by: score, tps, params, mem, ctx, date
    pub sort: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchModelsParams {
    /// Search query (matches model name, provider, parameters, use case)
    pub query: String,
    /// Maximum number of results to return (default: 10)
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct PlanHardwareParams {
    /// Model name to plan for
    pub model: String,
    /// Context window size (default: 8192)
    pub context: Option<u32>,
    /// Quantization level (e.g. Q4_K_M, Q8_0)
    pub quant: Option<String>,
    /// Target tokens per second
    pub target_tps: Option<f64>,
}

// --- Tool output types ---

#[derive(Serialize)]
struct RuntimeInfo {
    name: &'static str,
    installed: bool,
}

#[derive(Serialize)]
struct InstalledModel {
    name: String,
    runtime: String,
}

// --- MCP Server ---

#[derive(Clone)]
pub struct LlmfitMcpServer {
    specs: SystemSpecs,
    models: Vec<LlmModel>,
    context_limit: Option<u32>,
    node_name: String,
    tool_router: ToolRouter<Self>,
}

#[tool_handler]
impl ServerHandler for LlmfitMcpServer {}

#[tool_router]
impl LlmfitMcpServer {
    pub fn new(
        specs: SystemSpecs,
        models: Vec<LlmModel>,
        context_limit: Option<u32>,
        node_name: String,
    ) -> Self {
        Self {
            specs,
            models,
            context_limit,
            node_name,
            tool_router: Self::tool_router(),
        }
    }

    /// Get this node's hardware specifications including RAM, GPU, CPU details
    #[tool(
        name = "get_system_specs",
        description = "Get hardware specs for this node (RAM, GPU, CPU)"
    )]
    async fn get_system_specs(&self) -> String {
        let result = serde_json::json!({
            "node": self.node_name,
            "system": serve_shared::system_json(&self.specs),
        });
        serde_json::to_string_pretty(&result).unwrap_or_default()
    }

    /// Recommend LLM models that fit this system's hardware
    #[tool(
        name = "recommend_models",
        description = "Recommend LLM models that fit this system's hardware, with optional filtering by use case, fit level, runtime, and license"
    )]
    async fn recommend_models(&self, params: Parameters<RecommendModelsParams>) -> String {
        let params = params.0;
        let limit = params.limit.unwrap_or(10);
        let sort_column = parse_sort(params.sort.as_deref());
        let min_fit = parse_min_fit(params.min_fit.as_deref());
        let runtime_filter = parse_runtime(params.runtime.as_deref());
        let use_case_filter = parse_use_case(params.use_case.as_deref());

        let mut fits = self.analyze_all();

        if let Some(min) = min_fit {
            fits.retain(|f| fit_at_least(f.fit_level, min));
        } else {
            fits.retain(|f| f.fit_level != FitLevel::TooTight);
        }

        if let Some(rt) = runtime_filter {
            fits.retain(|f| f.runtime == rt);
        }

        if let Some(uc) = use_case_filter {
            fits.retain(|f| f.use_case == uc);
        }

        if let Some(ref lic) = params.license {
            fits.retain(|f| llmfit_core::models::matches_license_filter(&f.model.license, lic));
        }

        let total = fits.len();
        let mut ranked = rank_models_by_fit_opts_col(fits, false, sort_column);
        ranked.truncate(limit);

        let result = serde_json::json!({
            "total_models": total,
            "returned_models": ranked.len(),
            "models": ranked.iter().map(serve_shared::fit_to_json).collect::<Vec<_>>(),
        });
        serde_json::to_string_pretty(&result).unwrap_or_default()
    }

    /// Search for models by name, provider, or use case
    #[tool(
        name = "search_models",
        description = "Search for LLM models by name, provider, or use case"
    )]
    async fn search_models(&self, params: Parameters<SearchModelsParams>) -> String {
        let params = params.0;
        let limit = params.limit.unwrap_or(10);
        let search_lower = params.query.to_lowercase();

        let mut fits = self.analyze_all();
        fits.retain(|f| {
            f.model.name.to_lowercase().contains(&search_lower)
                || f.model.provider.to_lowercase().contains(&search_lower)
                || f.model
                    .parameter_count
                    .to_lowercase()
                    .contains(&search_lower)
                || f.model.use_case.to_lowercase().contains(&search_lower)
                || f.use_case.label().to_lowercase().contains(&search_lower)
        });
        fits.retain(|f| f.fit_level != FitLevel::TooTight);

        let total = fits.len();
        let mut ranked = rank_models_by_fit_opts_col(fits, false, SortColumn::Score);
        ranked.truncate(limit);

        let result = serde_json::json!({
            "query": params.query,
            "total_matches": total,
            "returned_models": ranked.len(),
            "models": ranked.iter().map(serve_shared::fit_to_json).collect::<Vec<_>>(),
        });
        serde_json::to_string_pretty(&result).unwrap_or_default()
    }

    /// Plan hardware requirements for running a specific model
    #[tool(
        name = "plan_hardware",
        description = "Plan hardware requirements for running a specific model at a given context window size"
    )]
    async fn plan_hardware(&self, params: Parameters<PlanHardwareParams>) -> String {
        let params = params.0;
        let model = self
            .models
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(&params.model));

        let Some(model) = model else {
            return serde_json::json!({
                "error": format!("model '{}' not found", params.model),
            })
            .to_string();
        };

        let request = PlanRequest {
            context: params.context.unwrap_or(8192),
            quant: params.quant,
            target_tps: params.target_tps,
            kv_quant: None,
        };

        match estimate_model_plan(model, &request, &self.specs) {
            Ok(estimate) => {
                serde_json::to_string_pretty(&serde_json::json!(estimate)).unwrap_or_default()
            }
            Err(e) => serde_json::json!({ "error": e }).to_string(),
        }
    }

    /// Check which inference runtimes are installed and available
    #[tool(
        name = "get_runtimes",
        description = "Check which local inference runtimes (Ollama, MLX, llama.cpp, vLLM, etc.) are installed"
    )]
    async fn get_runtimes(&self) -> String {
        let mut set = tokio::task::JoinSet::new();
        set.spawn_blocking(|| RuntimeInfo {
            name: "ollama",
            installed: OllamaProvider::new().is_available(),
        });
        set.spawn_blocking(|| RuntimeInfo {
            name: "mlx",
            installed: MlxProvider::new().is_available(),
        });
        set.spawn_blocking(|| RuntimeInfo {
            name: "llamacpp",
            installed: LlamaCppProvider::new().is_available(),
        });
        set.spawn_blocking(|| RuntimeInfo {
            name: "docker_model_runner",
            installed: DockerModelRunnerProvider::new().is_available(),
        });
        set.spawn_blocking(|| RuntimeInfo {
            name: "lmstudio",
            installed: LmStudioProvider::new().is_available(),
        });
        set.spawn_blocking(|| RuntimeInfo {
            name: "vllm",
            installed: VllmProvider::new().is_available(),
        });

        let mut runtimes = Vec::new();
        while let Some(result) = set.join_next().await {
            if let Ok(info) = result {
                runtimes.push(info);
            }
        }

        let result = serde_json::json!({ "runtimes": runtimes });
        serde_json::to_string_pretty(&result).unwrap_or_default()
    }

    /// List models currently installed in local inference runtimes
    #[tool(
        name = "get_installed_models",
        description = "List models currently installed in local inference runtimes (Ollama, MLX, llama.cpp, etc.)"
    )]
    async fn get_installed_models(&self) -> String {
        let mut set = tokio::task::JoinSet::new();
        set.spawn_blocking(|| {
            let p = OllamaProvider::new();
            ("ollama", p.is_available(), p.installed_models())
        });
        set.spawn_blocking(|| {
            let p = MlxProvider::new();
            ("mlx", p.is_available(), p.installed_models())
        });
        set.spawn_blocking(|| {
            let p = LlamaCppProvider::new();
            ("llamacpp", p.is_available(), p.installed_models())
        });
        set.spawn_blocking(|| {
            let p = DockerModelRunnerProvider::new();
            (
                "docker_model_runner",
                p.is_available(),
                p.installed_models(),
            )
        });
        set.spawn_blocking(|| {
            let p = LmStudioProvider::new();
            ("lmstudio", p.is_available(), p.installed_models())
        });
        set.spawn_blocking(|| {
            let p = VllmProvider::new();
            ("vllm", p.is_available(), p.installed_models())
        });

        let mut models = Vec::new();
        while let Some(result) = set.join_next().await {
            if let Ok((name, available, installed)) = result {
                if available {
                    for model_name in installed {
                        models.push(InstalledModel {
                            name: model_name,
                            runtime: name.to_string(),
                        });
                    }
                }
            }
        }

        let result = serde_json::json!({ "models": models });
        serde_json::to_string_pretty(&result).unwrap_or_default()
    }
}

impl LlmfitMcpServer {
    fn analyze_all(&self) -> Vec<ModelFit> {
        let is_apple_silicon = self.specs.backend == GpuBackend::Metal && self.specs.unified_memory;
        let mut fits: Vec<ModelFit> = self
            .models
            .iter()
            .filter(|m| backend_compatible(m, &self.specs))
            .map(|m| {
                ModelFit::analyze_with_forced_runtime(m, &self.specs, self.context_limit, None)
            })
            .collect();

        if !is_apple_silicon {
            fits.retain(|f| !f.model.is_mlx_only());
        }

        fits
    }
}

// --- Entry point ---

pub fn run_mcp_server(
    overrides: &super::HardwareOverrides,
    context_limit: Option<u32>,
) -> Result<(), String> {
    let specs = super::detect_specs(overrides);
    let db = ModelDatabase::new();
    let models = db.get_all_models().clone();
    let node_name = std::env::var("HOSTNAME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "unknown-node".to_string());

    let server = LlmfitMcpServer::new(specs, models, context_limit, node_name);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("failed to start tokio runtime: {e}"))?;

    runtime.block_on(async move {
        let transport = rmcp::transport::io::stdio();
        let service = server
            .serve(transport)
            .await
            .map_err(|e| format!("MCP server error: {e}"))?;
        service
            .waiting()
            .await
            .map_err(|e| format!("MCP server terminated: {e}"))?;
        Ok(())
    })
}

// --- Helper parsers ---

fn parse_sort(raw: Option<&str>) -> SortColumn {
    match raw.unwrap_or("score").trim().to_lowercase().as_str() {
        "tps" | "tokens" | "throughput" => SortColumn::Tps,
        "params" | "parameters" => SortColumn::Params,
        "mem" | "memory" | "mem_pct" => SortColumn::MemPct,
        "ctx" | "context" => SortColumn::Ctx,
        "date" | "release" => SortColumn::ReleaseDate,
        "use" | "use_case" => SortColumn::UseCase,
        _ => SortColumn::Score,
    }
}

fn parse_min_fit(raw: Option<&str>) -> Option<FitLevel> {
    raw.map(|s| match s.trim().to_lowercase().as_str() {
        "perfect" => FitLevel::Perfect,
        "good" => FitLevel::Good,
        "marginal" => FitLevel::Marginal,
        _ => FitLevel::Marginal,
    })
}

fn parse_runtime(raw: Option<&str>) -> Option<InferenceRuntime> {
    raw.and_then(|s| match s.trim().to_lowercase().as_str() {
        "mlx" => Some(InferenceRuntime::Mlx),
        "llamacpp" | "llama.cpp" | "llama_cpp" => Some(InferenceRuntime::LlamaCpp),
        "vllm" => Some(InferenceRuntime::Vllm),
        _ => None,
    })
}

fn parse_use_case(raw: Option<&str>) -> Option<UseCase> {
    raw.and_then(|s| match s.trim().to_lowercase().as_str() {
        "coding" | "code" => Some(UseCase::Coding),
        "reasoning" | "reason" => Some(UseCase::Reasoning),
        "chat" => Some(UseCase::Chat),
        "multimodal" | "vision" => Some(UseCase::Multimodal),
        "embedding" | "embed" => Some(UseCase::Embedding),
        "general" => Some(UseCase::General),
        _ => None,
    })
}

fn fit_at_least(actual: FitLevel, minimum: FitLevel) -> bool {
    let rank = |fit: FitLevel| match fit {
        FitLevel::Perfect => 3,
        FitLevel::Good => 2,
        FitLevel::Marginal => 1,
        FitLevel::TooTight => 0,
    };
    rank(actual) >= rank(minimum)
}
