//! LLM inference benchmarking against Ollama, vLLM, and MLX endpoints.
//!
//! Measures time-to-first-token (TTFT), tokens per second (TPS),
//! and total latency using real inference requests.

use std::time::{Duration, Instant};

/// Results from a single benchmark run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchRun {
    /// Time to first token in milliseconds, if measurable.
    /// - Ollama: measured from `eval_duration` (accurate).
    /// - vLLM/MLX: `None` — would require streaming to measure; only wall-clock
    ///   total is available.
    pub ttft_ms: Option<f64>,
    /// Output tokens per second.
    pub tps: f64,
    /// Total request latency in milliseconds.
    pub total_ms: f64,
    /// Number of prompt tokens processed.
    pub prompt_tokens: u32,
    /// Number of output tokens generated.
    pub output_tokens: u32,
}

/// Aggregated benchmark results across multiple runs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchResult {
    pub model: String,
    pub provider: String,
    pub runs: Vec<BenchRun>,
    pub summary: BenchSummary,
}

/// Statistical summary of benchmark runs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchSummary {
    pub num_runs: usize,
    pub avg_ttft_ms: Option<f64>,
    pub avg_tps: f64,
    pub min_tps: f64,
    pub max_tps: f64,
    pub avg_total_ms: f64,
    pub avg_output_tokens: f64,
}

impl BenchSummary {
    fn from_runs(runs: &[BenchRun]) -> Self {
        let n = runs.len() as f64;
        if runs.is_empty() {
            return BenchSummary {
                num_runs: 0,
                avg_ttft_ms: None,
                avg_tps: 0.0,
                min_tps: 0.0,
                max_tps: 0.0,
                avg_total_ms: 0.0,
                avg_output_tokens: 0.0,
            };
        }
        // Only compute avg TTFT if any run has a measured value
        let ttft_values: Vec<f64> = runs.iter().filter_map(|r| r.ttft_ms).collect();
        let avg_ttft_ms = if ttft_values.is_empty() {
            None
        } else {
            Some(ttft_values.iter().sum::<f64>() / ttft_values.len() as f64)
        };
        BenchSummary {
            num_runs: runs.len(),
            avg_ttft_ms,
            avg_tps: runs.iter().map(|r| r.tps).sum::<f64>() / n,
            min_tps: runs.iter().map(|r| r.tps).fold(f64::INFINITY, f64::min),
            max_tps: runs.iter().map(|r| r.tps).fold(0.0_f64, f64::max),
            avg_total_ms: runs.iter().map(|r| r.total_ms).sum::<f64>() / n,
            avg_output_tokens: runs.iter().map(|r| r.output_tokens as f64).sum::<f64>() / n,
        }
    }
}

/// Test prompts of varying lengths for benchmarking.
const BENCH_PROMPTS: &[&str] = &[
    "Explain what a hash table is in 2 sentences.",
    "Write a Python function that checks if a string is a palindrome. Include a docstring.",
    "Compare and contrast TCP and UDP protocols. Cover reliability, ordering, speed, and common use cases. Be concise.",
    "You are a senior software engineer. Review this code and suggest improvements:\n\n```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```",
];

// ── Ollama benchmarking ────────────────────────────────────────────

/// Ollama /api/generate response fields we care about.
#[derive(serde::Deserialize, Default)]
#[allow(dead_code)]
struct OllamaGenResponse {
    #[serde(default)]
    response: String,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>, // nanoseconds
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>, // nanoseconds
    #[serde(default)]
    total_duration: Option<u64>, // nanoseconds
}

/// Benchmark a model via Ollama's /api/generate endpoint.
pub fn bench_ollama(
    base_url: &str,
    model: &str,
    num_runs: usize,
    on_progress: &dyn Fn(usize, usize),
) -> Result<BenchResult, String> {
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
    let mut runs = Vec::with_capacity(num_runs);

    // Warmup request (don't count it)
    on_progress(0, num_runs);
    if let Err(e) = ollama_generate(&url, model, "Say hello.", 300) {
        return Err(format!(
            "Warmup request failed (is the model loaded?): {}",
            e
        ));
    }

    for i in 0..num_runs {
        on_progress(i + 1, num_runs);
        let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()];
        let run = ollama_generate(&url, model, prompt, 300)?;
        runs.push(run);
    }

    let summary = BenchSummary::from_runs(&runs);
    Ok(BenchResult {
        model: model.to_string(),
        provider: "ollama".to_string(),
        runs,
        summary,
    })
}

fn ollama_generate(
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<BenchRun, String> {
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "num_predict": max_tokens,
        }
    });

    let start = Instant::now();
    let resp = ureq::post(url)
        .config()
        .timeout_global(Some(Duration::from_secs(300)))
        .build()
        .send_json(&body)
        .map_err(|e| format!("Ollama request failed: {}", e))?;

    let total_wall = start.elapsed();

    let resp_body: OllamaGenResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("Ollama JSON parse error: {}", e))?;

    // Ollama provides native timing in nanoseconds
    let prompt_tokens = resp_body.prompt_eval_count.unwrap_or(0) as u32;
    let output_tokens = resp_body.eval_count.unwrap_or(0) as u32;

    let ttft_ms = resp_body
        .prompt_eval_duration
        .map(|ns| ns as f64 / 1_000_000.0);

    let tps = if let (Some(eval_count), Some(eval_dur)) =
        (resp_body.eval_count, resp_body.eval_duration)
    {
        if eval_dur > 0 {
            eval_count as f64 / (eval_dur as f64 / 1_000_000_000.0)
        } else {
            0.0
        }
    } else if output_tokens > 0 {
        // Fallback to wall-clock
        output_tokens as f64 / total_wall.as_secs_f64()
    } else {
        0.0
    };

    let total_ms = resp_body
        .total_duration
        .map(|ns| ns as f64 / 1_000_000.0)
        .unwrap_or(total_wall.as_secs_f64() * 1000.0);

    Ok(BenchRun {
        ttft_ms,
        tps,
        total_ms,
        prompt_tokens,
        output_tokens,
    })
}

// ── OpenAI-compatible benchmarking (vLLM, MLX) ────────────────────

/// OpenAI chat completion response fields we care about.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ChatChoice {
    #[serde(default)]
    message: Option<ChatMessage>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(serde::Deserialize)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Benchmark a model via OpenAI-compatible /v1/chat/completions (vLLM, MLX).
pub fn bench_openai_compat(
    base_url: &str,
    model: &str,
    provider_name: &str,
    num_runs: usize,
    on_progress: &dyn Fn(usize, usize),
) -> Result<BenchResult, String> {
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let mut runs = Vec::with_capacity(num_runs);

    // Warmup
    on_progress(0, num_runs);
    if let Err(e) = openai_chat(&url, model, "Say hello.", 100) {
        return Err(format!(
            "Warmup request failed (is the endpoint reachable?): {}",
            e
        ));
    }

    for i in 0..num_runs {
        on_progress(i + 1, num_runs);
        let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()];
        let run = openai_chat(&url, model, prompt, 300)?;
        runs.push(run);
    }

    let summary = BenchSummary::from_runs(&runs);
    Ok(BenchResult {
        model: model.to_string(),
        provider: provider_name.to_string(),
        runs,
        summary,
    })
}

fn openai_chat(url: &str, model: &str, prompt: &str, max_tokens: u32) -> Result<BenchRun, String> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": false,
    });

    let start = Instant::now();

    // TTFT is estimated from wall clock: prompt_tokens / total_tokens * total_time.
    // This is a rough heuristic — actual TTFT requires streaming (not implemented).
    let resp = ureq::post(url)
        .config()
        .timeout_global(Some(Duration::from_secs(300)))
        .build()
        .send_json(&body)
        .map_err(|e| format!("{} request failed: {}", url, e))?;

    let total_wall = start.elapsed();

    let completion: ChatCompletionResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;

    let usage = completion.usage.unwrap_or(ChatUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
    });

    let output_tokens = usage.completion_tokens;
    let prompt_tokens = usage.prompt_tokens;

    // TTFT cannot be measured without streaming — set to None.
    let total_ms = total_wall.as_secs_f64() * 1000.0;

    let tps = if output_tokens > 0 && total_wall.as_secs_f64() > 0.0 {
        output_tokens as f64 / total_wall.as_secs_f64()
    } else {
        0.0
    };

    Ok(BenchRun {
        ttft_ms: None,
        tps,
        total_ms,
        prompt_tokens,
        output_tokens,
    })
}

// ── Auto-detect and benchmark ──────────────────────────────────────

/// Which provider to benchmark against.
#[derive(Debug, Clone)]
pub enum BenchTarget {
    Ollama { url: String, model: String },
    VLlm { url: String, model: String },
    Mlx { url: String, model: String },
}

/// Auto-detect available providers and pick the best one to benchmark.
pub fn auto_detect_target(model_hint: Option<&str>) -> Result<BenchTarget, String> {
    // Check vLLM via VLLM_PORT env var (defaults to 8000)
    let vllm_port = std::env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string());
    let vllm_url = format!("http://localhost:{}", vllm_port);
    if let Ok(model_name) = detect_vllm_model(&vllm_url, model_hint) {
        return Ok(BenchTarget::VLlm {
            url: vllm_url,
            model: model_name,
        });
    }

    // Check Ollama
    let ollama_url =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    if ureq::get(&format!("{}/api/tags", ollama_url))
        .config()
        .timeout_global(Some(Duration::from_secs(2)))
        .build()
        .call()
        .is_ok()
    {
        if let Ok(model_name) = detect_ollama_model(&ollama_url, model_hint) {
            return Ok(BenchTarget::Ollama {
                url: ollama_url,
                model: model_name,
            });
        }
    }

    // Check MLX
    let mlx_url =
        std::env::var("MLX_LM_HOST").unwrap_or_else(|_| "http://localhost:8080".to_string());
    if ureq::get(&format!("{}/v1/models", mlx_url))
        .config()
        .timeout_global(Some(Duration::from_secs(2)))
        .build()
        .call()
        .is_ok()
    {
        if let Ok(model_name) = detect_openai_model(&mlx_url, model_hint) {
            return Ok(BenchTarget::Mlx {
                url: mlx_url,
                model: model_name,
            });
        }
    }

    Err("No inference provider found. Start Ollama, vLLM, or MLX first.".to_string())
}

/// Discover all available models across all providers.
pub fn discover_all_targets() -> Vec<BenchTarget> {
    let mut targets = Vec::new();

    // Check vLLM via VLLM_PORT env var (defaults to 8000)
    let vllm_port = std::env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string());
    let vllm_url = format!("http://localhost:{}", vllm_port);
    if let Ok(models) = list_openai_models(&vllm_url) {
        for model in models {
            targets.push(BenchTarget::VLlm {
                url: vllm_url.clone(),
                model,
            });
        }
    }

    // Check Ollama
    let ollama_url =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    if let Ok(models) = list_ollama_models(&ollama_url) {
        for model in models {
            targets.push(BenchTarget::Ollama {
                url: ollama_url.clone(),
                model,
            });
        }
    }

    // Check MLX
    let mlx_url =
        std::env::var("MLX_LM_HOST").unwrap_or_else(|_| "http://localhost:8080".to_string());
    if let Ok(models) = list_openai_models(&mlx_url) {
        for model in models {
            targets.push(BenchTarget::Mlx {
                url: mlx_url.clone(),
                model,
            });
        }
    }

    targets
}

fn list_openai_models(base_url: &str) -> Result<Vec<String>, String> {
    let url = format!("{}/v1/models", base_url);
    let resp = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(3)))
        .build()
        .call()
        .map_err(|e| format!("{}", e))?;

    let body: serde_json::Value = resp.into_body().read_json().map_err(|e| format!("{}", e))?;
    let models = body
        .get("data")
        .and_then(|d: &serde_json::Value| d.as_array())
        .ok_or("no data")?;

    Ok(models
        .iter()
        .filter_map(|m| {
            m.get("id")
                .and_then(|i: &serde_json::Value| i.as_str())
                .map(|s| s.to_string())
        })
        .collect())
}

fn list_ollama_models(base_url: &str) -> Result<Vec<String>, String> {
    let url = format!("{}/api/tags", base_url);
    let resp = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(3)))
        .build()
        .call()
        .map_err(|e| format!("{}", e))?;

    #[derive(serde::Deserialize)]
    struct Tags {
        models: Vec<M>,
    }
    #[derive(serde::Deserialize)]
    struct M {
        name: String,
    }

    let tags: Tags = resp.into_body().read_json().map_err(|e| format!("{}", e))?;
    Ok(tags.models.into_iter().map(|m| m.name).collect())
}

/// Detect model from a given base URL (OpenAI-compatible /v1/models).
pub fn detect_model_from_url(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    detect_openai_model(base_url, hint)
}

fn detect_vllm_model(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    detect_openai_model(base_url, hint)
}

fn detect_openai_model(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    let url = format!("{}/v1/models", base_url);
    let resp = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .call()
        .map_err(|e| format!("Cannot reach {}: {}", url, e))?;

    let body: serde_json::Value = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON error: {}", e))?;

    let models = body
        .get("data")
        .and_then(|d: &serde_json::Value| d.as_array())
        .ok_or("No models found")?;

    if models.is_empty() {
        return Err("No models loaded".to_string());
    }

    // If hint provided, find matching model
    if let Some(hint) = hint {
        let hint_lower = hint.to_lowercase();
        for m in models {
            if let Some(id) = m.get("id").and_then(|i: &serde_json::Value| i.as_str()) {
                if id.to_lowercase().contains(&hint_lower) {
                    return Ok(id.to_string());
                }
            }
        }
    }

    // Return first model
    models[0]
        .get("id")
        .and_then(|i: &serde_json::Value| i.as_str())
        .map(|s| s.to_string())
        .ok_or("Model has no id".to_string())
}

fn detect_ollama_model(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    let url = format!("{}/api/tags", base_url);
    let resp = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .call()
        .map_err(|e| format!("Cannot reach Ollama: {}", e))?;

    #[derive(serde::Deserialize)]
    struct Tags {
        models: Vec<Model>,
    }
    #[derive(serde::Deserialize)]
    struct Model {
        name: String,
    }

    let tags: Tags = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON error: {}", e))?;

    if tags.models.is_empty() {
        return Err("No models installed in Ollama".to_string());
    }

    if let Some(hint) = hint {
        let hint_lower = hint.to_lowercase();
        for m in &tags.models {
            if m.name.to_lowercase().contains(&hint_lower) {
                return Ok(m.name.clone());
            }
        }
    }

    Ok(tags.models[0].name.clone())
}

// ── Display helpers ────────────────────────────────────────────────

impl BenchResult {
    pub fn display(&self) {
        println!();
        println!("  === Benchmark Results ===");
        println!("  Model:    {}", self.model);
        println!("  Provider: {}", self.provider);
        println!("  Runs:     {}", self.summary.num_runs);
        println!();
        println!(
            "  TPS:      {:.1} avg  ({:.1} min / {:.1} max)",
            self.summary.avg_tps, self.summary.min_tps, self.summary.max_tps
        );
        if let Some(ttft) = self.summary.avg_ttft_ms {
            println!("  TTFT:     {:.0} ms avg", ttft);
        } else {
            println!("  TTFT:     n/a (streaming required)");
        }
        println!("  Latency:  {:.0} ms avg", self.summary.avg_total_ms);
        println!(
            "  Output:   {:.0} tokens avg",
            self.summary.avg_output_tokens
        );
        println!();

        // Per-run breakdown
        println!("  Run  TPS      TTFT     Latency  Tokens");
        println!("  ───  ───────  ───────  ───────  ──────");
        for (i, run) in self.runs.iter().enumerate() {
            println!(
                "  {:>3}  {:>6.1}   {:>5}ms  {:>5.0}ms  {:>5}",
                i + 1,
                run.tps,
                run.ttft_ms
                    .map(|t| format!("{:.0}", t))
                    .unwrap_or_else(|| "n/a".to_string()),
                run.total_ms,
                run.output_tokens
            );
        }
        println!();
    }

    pub fn display_json(&self) {
        let json = serde_json::json!({
            "benchmark": {
                "model": self.model,
                "provider": self.provider,
                "summary": self.summary,
                "runs": self.runs,
            }
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).expect("JSON serialization failed")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_run(ttft_ms: f64, tps: f64, total_ms: f64, output_tokens: u32) -> BenchRun {
        BenchRun {
            ttft_ms: Some(ttft_ms),
            tps,
            total_ms,
            prompt_tokens: 10,
            output_tokens,
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // BenchSummary::from_runs
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_summary_multiple_runs() {
        let runs = vec![
            make_run(100.0, 20.0, 500.0, 50),
            make_run(150.0, 30.0, 600.0, 60),
            make_run(200.0, 10.0, 700.0, 70),
        ];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 3);
        assert!((s.avg_ttft_ms.unwrap() - 150.0).abs() < 0.01);
        assert!((s.avg_tps - 20.0).abs() < 0.01);
        assert!((s.min_tps - 10.0).abs() < 0.01);
        assert!((s.max_tps - 30.0).abs() < 0.01);
        assert!((s.avg_total_ms - 600.0).abs() < 0.01);
        assert!((s.avg_output_tokens - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_summary_single_run() {
        let runs = vec![make_run(100.0, 25.0, 500.0, 50)];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 1);
        assert!((s.avg_ttft_ms.unwrap() - 100.0).abs() < 0.01);
        assert!((s.avg_tps - 25.0).abs() < 0.01);
        // min == max == avg for a single run
        assert!((s.min_tps - 25.0).abs() < 0.01);
        assert!((s.max_tps - 25.0).abs() < 0.01);
        assert!((s.avg_total_ms - 500.0).abs() < 0.01);
        assert!((s.avg_output_tokens - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_summary_empty_runs() {
        let runs: Vec<BenchRun> = vec![];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 0);
        assert_eq!(s.avg_tps, 0.0);
        assert_eq!(s.min_tps, 0.0);
        assert_eq!(s.max_tps, 0.0);
        assert_eq!(s.avg_ttft_ms, None);
        assert_eq!(s.avg_total_ms, 0.0);
        assert_eq!(s.avg_output_tokens, 0.0);
    }

    #[test]
    fn test_summary_min_max_correctness() {
        let runs = vec![
            make_run(50.0, 5.0, 200.0, 20),
            make_run(60.0, 50.0, 300.0, 30),
            make_run(70.0, 25.0, 400.0, 40),
            make_run(80.0, 100.0, 500.0, 50),
            make_run(90.0, 1.0, 600.0, 60),
        ];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 5);
        assert!((s.min_tps - 1.0).abs() < 0.01);
        assert!((s.max_tps - 100.0).abs() < 0.01);
        // avg_tps = (5+50+25+100+1)/5 = 36.2
        assert!((s.avg_tps - 36.2).abs() < 0.01);
    }

    #[test]
    fn test_summary_identical_runs() {
        let runs = vec![
            make_run(100.0, 20.0, 500.0, 50),
            make_run(100.0, 20.0, 500.0, 50),
            make_run(100.0, 20.0, 500.0, 50),
        ];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 3);
        assert!((s.avg_tps - 20.0).abs() < 0.01);
        assert!((s.min_tps - 20.0).abs() < 0.01);
        assert!((s.max_tps - 20.0).abs() < 0.01);
        assert!((s.avg_ttft_ms.unwrap() - 100.0).abs() < 0.01);
    }
}
