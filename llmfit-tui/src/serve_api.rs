use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, LazyLock};

use axum::extract::{Path, Query, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use llmfit_core::fit::{
    FitLevel, InferenceRuntime, ModelFit, SortColumn, backend_compatible,
    rank_models_by_fit_opts_col,
};
use llmfit_core::hardware::{GpuBackend, SystemSpecs};
use llmfit_core::models::{LlmModel, ModelDatabase, UseCase};
use serde::{Deserialize, Serialize};

include!(concat!(env!("OUT_DIR"), "/web_assets.rs"));

static ASSET_MAP: LazyLock<HashMap<&'static str, &'static EmbeddedAsset>> =
    LazyLock::new(|| EMBEDDED_WEB_ASSETS.iter().map(|a| (a.path, a)).collect());

#[derive(Clone)]
struct AppState {
    node_name: String,
    os: String,
    specs: SystemSpecs,
    models: Vec<LlmModel>,
    context_limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ModelsQuery {
    limit: Option<usize>,
    #[serde(alias = "n")]
    top: Option<usize>,
    perfect: Option<bool>,
    min_fit: Option<String>,
    runtime: Option<String>,
    use_case: Option<String>,
    provider: Option<String>,
    search: Option<String>,
    sort: Option<String>,
    include_too_tight: Option<bool>,
    max_context: Option<u32>,
    force_runtime: Option<String>,
}

#[derive(Debug, Serialize)]
struct NodeInfo {
    name: String,
    os: String,
}

#[derive(Debug, Serialize)]
struct ApiEnvelope {
    node: NodeInfo,
    system: serde_json::Value,
    total_models: usize,
    returned_models: usize,
    filters: serde_json::Value,
    models: Vec<serde_json::Value>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": self.message,
            })),
        )
            .into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

pub fn run_serve(
    host: &str,
    port: u16,
    memory_override: &Option<String>,
    context_limit: Option<u32>,
) -> Result<(), String> {
    let ip: IpAddr = host
        .parse()
        .map_err(|_| format!("invalid --host value: '{host}'"))?;
    let addr = SocketAddr::new(ip, port);

    let specs = detect_specs(memory_override);
    let db = ModelDatabase::new();
    let all_models = db.get_all_models().clone();

    let node_name = std::env::var("HOSTNAME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "unknown-node".to_string());

    let state = Arc::new(AppState {
        node_name,
        os: std::env::consts::OS.to_string(),
        specs,
        models: all_models,
        context_limit,
    });

    let app = build_router(state);

    println!("llmfit dashboard listening on http://{}/", addr);
    println!("  API models: http://{}/api/v1/models", addr);
    println!("  GET /health");
    println!("  GET /api/v1/system");
    println!("  GET /api/v1/models?limit=20&min_fit=marginal&sort=score");
    println!("  GET /api/v1/models/top?limit=5&use_case=coding&min_fit=good");
    println!("  GET /api/v1/models/<name>");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("failed to start tokio runtime: {e}"))?;

    runtime
        .block_on(async move {
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| ApiError::internal(format!("bind failed on {addr}: {e}")))?;

            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await
                .map_err(|e| ApiError::internal(format!("server error: {e}")))
        })
        .map_err(|e| e.message)
}

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(web_index))
        .route("/assets/{*path}", get(web_asset))
        .route("/health", get(health))
        .route("/api/v1/system", get(system))
        .route("/api/v1/models", get(models))
        .route("/api/v1/models/top", get(top_models))
        .route("/api/v1/models/{name}", get(model_by_name))
        .route("/{*path}", get(spa_fallback))
        .with_state(state)
}

async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "node": {
            "name": state.node_name,
            "os": state.os,
        }
    }))
}

async fn system(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "node": {
            "name": state.node_name,
            "os": state.os,
        },
        "system": system_json(&state.specs),
    }))
}

async fn web_index() -> Response {
    serve_web_path("/index.html")
}

async fn web_asset(Path(path): Path<String>) -> Response {
    let asset_path = format!("/assets/{}", path.trim_start_matches('/'));
    serve_web_path(&asset_path)
}

async fn spa_fallback(Path(path): Path<String>) -> Response {
    if path.starts_with("api/") || path == "health" || path.starts_with("assets/") {
        return StatusCode::NOT_FOUND.into_response();
    }
    serve_web_path("/index.html")
}

fn serve_web_path(path: &str) -> Response {
    let Some(asset) = find_web_asset(path) else {
        return StatusCode::NOT_FOUND.into_response();
    };

    let mut response = asset.bytes.to_vec().into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(asset.content_type));
    let cache_value = if path.starts_with("/assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static(cache_value));
    response
}

fn find_web_asset(path: &str) -> Option<&'static EmbeddedAsset> {
    ASSET_MAP.get(path).copied()
}

async fn models(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ModelsQuery>,
) -> ApiResult<Json<ApiEnvelope>> {
    let mut fits = filtered_fits(&state, &query, false)?;
    let total_models = fits.len();

    let limit = query.limit.or(query.top).unwrap_or(usize::MAX);
    if limit < fits.len() {
        fits.truncate(limit);
    }

    let envelope = ApiEnvelope {
        node: NodeInfo {
            name: state.node_name.clone(),
            os: state.os.clone(),
        },
        system: system_json(&state.specs),
        total_models,
        returned_models: fits.len(),
        filters: active_filters_json(&query, false),
        models: fits.iter().map(fit_to_json).collect(),
    };

    Ok(Json(envelope))
}

async fn top_models(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ModelsQuery>,
) -> ApiResult<Json<ApiEnvelope>> {
    let mut fits = filtered_fits(&state, &query, true)?;
    let total_models = fits.len();

    let limit = query.limit.or(query.top).unwrap_or(5);
    if limit < fits.len() {
        fits.truncate(limit);
    }

    let envelope = ApiEnvelope {
        node: NodeInfo {
            name: state.node_name.clone(),
            os: state.os.clone(),
        },
        system: system_json(&state.specs),
        total_models,
        returned_models: fits.len(),
        filters: active_filters_json(&query, true),
        models: fits.iter().map(fit_to_json).collect(),
    };

    Ok(Json(envelope))
}

async fn model_by_name(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(query): Query<ModelsQuery>,
) -> ApiResult<Json<ApiEnvelope>> {
    let mut scoped = query;
    scoped.search = Some(name);

    let mut fits = filtered_fits(&state, &scoped, false)?;
    let total_models = fits.len();

    let limit = scoped.limit.or(scoped.top).unwrap_or(20);
    if limit < fits.len() {
        fits.truncate(limit);
    }

    let envelope = ApiEnvelope {
        node: NodeInfo {
            name: state.node_name.clone(),
            os: state.os.clone(),
        },
        system: system_json(&state.specs),
        total_models,
        returned_models: fits.len(),
        filters: active_filters_json(&scoped, false),
        models: fits.iter().map(fit_to_json).collect(),
    };

    Ok(Json(envelope))
}

fn filtered_fits(
    state: &AppState,
    query: &ModelsQuery,
    top_only: bool,
) -> Result<Vec<ModelFit>, ApiError> {
    let sort_column = parse_sort(query.sort.as_deref())?;
    let min_fit = parse_min_fit(query.min_fit.as_deref())?;
    let runtime_filter = parse_runtime(query.runtime.as_deref())?;
    let use_case_filter = parse_use_case(query.use_case.as_deref())?;

    let context_limit = query.max_context.or(state.context_limit);
    let forced_rt = parse_force_runtime(query.force_runtime.as_deref())?;
    let mut fits: Vec<ModelFit> = state
        .models
        .iter()
        .filter(|m| backend_compatible(m, &state.specs))
        .map(|m| ModelFit::analyze_with_forced_runtime(m, &state.specs, context_limit, forced_rt))
        .collect();

    let is_apple_silicon = state.specs.backend == GpuBackend::Metal && state.specs.unified_memory;
    if !is_apple_silicon {
        fits.retain(|f| !f.model.is_mlx_only());
    }

    if let Some(provider) = query.provider.as_ref() {
        let provider_lower = provider.to_lowercase();
        fits.retain(|f| f.model.provider.to_lowercase().contains(&provider_lower));
    }

    if let Some(search) = query.search.as_ref() {
        let search_lower = search.to_lowercase();
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
    }

    if query.perfect.unwrap_or(false) {
        fits.retain(|f| f.fit_level == FitLevel::Perfect);
    } else {
        fits.retain(|f| fit_at_least(f.fit_level, min_fit));
    }

    match runtime_filter {
        RuntimeFilter::Any => {}
        RuntimeFilter::Mlx => fits.retain(|f| f.runtime == InferenceRuntime::Mlx),
        RuntimeFilter::Vllm => fits.retain(|f| f.runtime == InferenceRuntime::Vllm),
        RuntimeFilter::LlamaCpp => {
            fits.retain(|f| f.runtime == InferenceRuntime::LlamaCpp);
        }
    }

    if let Some(use_case) = use_case_filter {
        fits.retain(|f| f.use_case == use_case);
    }

    let include_too_tight = query.include_too_tight.unwrap_or(!top_only);
    if top_only || !include_too_tight {
        fits.retain(|f| f.fit_level != FitLevel::TooTight);
    }

    Ok(rank_models_by_fit_opts_col(fits, false, sort_column))
}

#[derive(Debug, Clone, Copy)]
enum RuntimeFilter {
    Any,
    Mlx,
    LlamaCpp,
    Vllm,
}

fn parse_sort(raw: Option<&str>) -> Result<SortColumn, ApiError> {
    let value = raw.unwrap_or("score").trim().to_lowercase();
    let sort = match value.as_str() {
        "score" => SortColumn::Score,
        "tps" | "tokens" | "throughput" => SortColumn::Tps,
        "params" | "parameters" => SortColumn::Params,
        "mem" | "memory" | "mem_pct" | "utilization" => SortColumn::MemPct,
        "ctx" | "context" => SortColumn::Ctx,
        "date" | "release" | "released" => SortColumn::ReleaseDate,
        "use" | "use_case" | "usecase" => SortColumn::UseCase,
        _ => {
            return Err(ApiError::bad_request(
                "invalid sort value: use score|tps|params|mem|ctx|date|use_case",
            ));
        }
    };
    Ok(sort)
}

fn parse_min_fit(raw: Option<&str>) -> Result<FitLevel, ApiError> {
    let value = raw.unwrap_or("marginal").trim().to_lowercase();
    let min_fit = match value.as_str() {
        "perfect" => FitLevel::Perfect,
        "good" => FitLevel::Good,
        "marginal" => FitLevel::Marginal,
        "too_tight" | "tootight" | "tight" => FitLevel::TooTight,
        _ => {
            return Err(ApiError::bad_request(
                "invalid min_fit value: use perfect|good|marginal|too_tight",
            ));
        }
    };
    Ok(min_fit)
}

fn parse_runtime(raw: Option<&str>) -> Result<RuntimeFilter, ApiError> {
    let Some(value) = raw else {
        return Ok(RuntimeFilter::Any);
    };

    let runtime = match value.trim().to_lowercase().as_str() {
        "any" => RuntimeFilter::Any,
        "mlx" => RuntimeFilter::Mlx,
        "llamacpp" | "llama.cpp" | "llama_cpp" => RuntimeFilter::LlamaCpp,
        "vllm" => RuntimeFilter::Vllm,
        _ => {
            return Err(ApiError::bad_request(
                "invalid runtime value: use any|mlx|llamacpp|vllm",
            ));
        }
    };
    Ok(runtime)
}

fn parse_force_runtime(
    raw: Option<&str>,
) -> Result<Option<llmfit_core::fit::InferenceRuntime>, ApiError> {
    let Some(value) = raw else {
        return Ok(None);
    };
    match value.trim().to_lowercase().as_str() {
        "mlx" => Ok(Some(llmfit_core::fit::InferenceRuntime::Mlx)),
        "llamacpp" | "llama.cpp" | "llama_cpp" => {
            Ok(Some(llmfit_core::fit::InferenceRuntime::LlamaCpp))
        }
        "vllm" => Ok(Some(llmfit_core::fit::InferenceRuntime::Vllm)),
        _ => Err(ApiError::bad_request(
            "invalid force_runtime value: use mlx|llamacpp|vllm",
        )),
    }
}

fn parse_use_case(raw: Option<&str>) -> Result<Option<UseCase>, ApiError> {
    let Some(value) = raw else {
        return Ok(None);
    };

    let use_case = match value.trim().to_lowercase().as_str() {
        "coding" | "code" => UseCase::Coding,
        "reasoning" | "reason" => UseCase::Reasoning,
        "chat" => UseCase::Chat,
        "multimodal" | "vision" => UseCase::Multimodal,
        "embedding" | "embed" => UseCase::Embedding,
        "general" => UseCase::General,
        _ => {
            return Err(ApiError::bad_request(
                "invalid use_case value: use general|coding|reasoning|chat|multimodal|embedding",
            ));
        }
    };
    Ok(Some(use_case))
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

fn active_filters_json(query: &ModelsQuery, top_only: bool) -> serde_json::Value {
    serde_json::json!({
        "limit": query.limit.or(query.top),
        "perfect": query.perfect,
        "min_fit": query.min_fit,
        "runtime": query.runtime,
        "use_case": query.use_case,
        "provider": query.provider,
        "search": query.search,
        "sort": query.sort,
        "max_context": query.max_context,
        "include_too_tight": query.include_too_tight,
        "top_only": top_only,
    })
}

fn fit_level_code(fit_level: FitLevel) -> &'static str {
    match fit_level {
        FitLevel::Perfect => "perfect",
        FitLevel::Good => "good",
        FitLevel::Marginal => "marginal",
        FitLevel::TooTight => "too_tight",
    }
}

fn run_mode_code(run_mode: llmfit_core::fit::RunMode) -> &'static str {
    match run_mode {
        llmfit_core::fit::RunMode::Gpu => "gpu",
        llmfit_core::fit::RunMode::MoeOffload => "moe_offload",
        llmfit_core::fit::RunMode::CpuOffload => "cpu_offload",
        llmfit_core::fit::RunMode::CpuOnly => "cpu_only",
    }
}

fn runtime_code(runtime: InferenceRuntime) -> &'static str {
    match runtime {
        InferenceRuntime::Mlx => "mlx",
        InferenceRuntime::LlamaCpp => "llamacpp",
        InferenceRuntime::Vllm => "vllm",
    }
}

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
        "gpu_name": specs.gpu_name,
        "gpu_count": specs.gpu_count,
        "unified_memory": specs.unified_memory,
        "backend": specs.backend.label(),
        "gpus": gpus_json,
    })
}

fn fit_to_json(fit: &ModelFit) -> serde_json::Value {
    serde_json::json!({
        "name": fit.model.name,
        "provider": fit.model.provider,
        "parameter_count": fit.model.parameter_count,
        "params_b": round2(fit.model.params_b()),
        "context_length": fit.model.context_length,
        "use_case": fit.model.use_case,
        "category": fit.use_case.label(),
        "release_date": fit.model.release_date,
        "is_moe": fit.model.is_moe,
        "fit_level": fit_level_code(fit.fit_level),
        "fit_label": fit.fit_text(),
        "run_mode": run_mode_code(fit.run_mode),
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
    })
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Detect system specs with optional GPU memory override.
fn detect_specs(memory_override: &Option<String>) -> SystemSpecs {
    let specs = SystemSpecs::detect();
    if let Some(mem_str) = memory_override {
        match llmfit_core::hardware::parse_memory_size(mem_str) {
            Some(gb) => specs.with_gpu_memory_override(gb),
            None => specs,
        }
    } else {
        specs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt as _;
    use std::future::Future;
    use tower::ServiceExt;

    fn test_state() -> Arc<AppState> {
        let db = ModelDatabase::new();
        Arc::new(AppState {
            node_name: "test-node".to_string(),
            os: "test-os".to_string(),
            specs: SystemSpecs::detect(),
            models: db.get_all_models().clone(),
            context_limit: None,
        })
    }

    fn test_router() -> Router {
        build_router(test_state())
    }

    fn find_asset_path_with_ext(ext: &str) -> Option<&'static EmbeddedAsset> {
        EMBEDDED_WEB_ASSETS
            .iter()
            .find(|asset| asset.path.starts_with("/assets/") && asset.path.ends_with(ext))
    }

    fn run_async<T>(future: impl Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(future)
    }

    #[test]
    fn root_serves_index_html() {
        run_async(async {
            let response = test_router()
                .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.headers().get(CONTENT_TYPE).unwrap(),
                "text/html; charset=utf-8"
            );
        });
    }

    #[test]
    fn assets_route_serves_embedded_file_with_content_type() {
        let Some(asset) = find_asset_path_with_ext(".js")
            .or_else(|| find_asset_path_with_ext(".css"))
            .or_else(|| find_asset_path_with_ext(".svg"))
        else {
            panic!("no embedded assets available under /assets/");
        };

        run_async(async {
            let response = test_router()
                .oneshot(
                    Request::builder()
                        .uri(asset.path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.headers().get(CONTENT_TYPE).unwrap(),
                asset.content_type
            );
        });
    }

    #[test]
    fn unknown_non_api_routes_fallback_to_index() {
        run_async(async {
            let response = test_router()
                .oneshot(
                    Request::builder()
                        .uri("/dashboard/models")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.headers().get(CONTENT_TYPE).unwrap(),
                "text/html; charset=utf-8"
            );
        });
    }

    #[test]
    fn existing_api_route_response_shape_is_preserved() {
        run_async(async {
            let response = test_router()
                .oneshot(
                    Request::builder()
                        .uri("/api/v1/system")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert!(value.get("node").is_some());
            assert!(value.get("system").is_some());
        });
    }

    #[test]
    fn unknown_api_paths_do_not_fallback_to_html() {
        run_async(async {
            let response = test_router()
                .oneshot(
                    Request::builder()
                        .uri("/api/v1/not-found")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::NOT_FOUND);
        });
    }
}
