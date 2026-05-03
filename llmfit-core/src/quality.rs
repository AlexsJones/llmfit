//! Role-based quality benchmarking for model routing.
//!
//! Tests models against 13 Amplifier roles with scored rubrics,
//! producing a routing matrix that maps roles to optimal models.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use regex::Regex;
use serde::{Deserialize, Serialize};

// ── Types ──────────────────────────────────────────────────────────

/// A single scoring rule applied to a model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringRule {
    pub pattern: String,
    pub weight: i32,
    #[serde(default)]
    pub negate: bool,
    #[serde(default)]
    pub case_insensitive: bool,
}

/// Definition of a single quality test loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTestDef {
    pub name: String,
    pub prompt: String,
    #[serde(default)]
    pub rules: Vec<ScoringRule>,
    /// Weight given to speed in the composite score (default 1.0).
    #[serde(default = "default_speed_weight")]
    pub speed_weight: Option<f64>,
    /// Maximum tokens to generate (default 1024).
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature (default 0.3).
    #[serde(default)]
    pub temperature: Option<f64>,
}

fn default_speed_weight() -> Option<f64> {
    Some(1.0)
}

/// A role definition containing its description and test suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleDef {
    pub description: String,
    pub tests: Vec<QualityTestDef>,
}

/// Top-level quality benchmark configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    pub roles: BTreeMap<String, RoleDef>,
}

/// Result of a single quality test against one model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityResult {
    pub test_name: String,
    pub role: String,
    /// Quality score from rubric evaluation (0-10).
    pub quality: f64,
    /// Output tokens per second.
    pub tok_per_sec: f64,
    /// Composite score blending quality and speed.
    pub composite: f64,
    /// First 150 chars of the model response.
    pub response_preview: String,
    /// Time to first token in milliseconds (if available).
    pub ttft_ms: Option<f64>,
    /// Total wall-clock time in seconds.
    pub wall_time_sec: f64,
    /// Number of output tokens.
    pub eval_tokens: u64,
    /// Error message if the test failed.
    pub error: Option<String>,
}

/// Aggregated score for a single role across all its tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleScore {
    pub role: String,
    pub quality: f64,
    pub speed: f64,
    pub composite: f64,
    pub test_count: usize,
}

/// Complete quality results for one model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQualityResult {
    pub model: String,
    pub provider: String,
    pub roles: Vec<RoleScore>,
    pub test_results: Vec<QualityResult>,
    pub overall_quality: f64,
    pub overall_speed: f64,
    pub overall_composite: f64,
}

/// Routing recommendation: which model is best for a given role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRecommendation {
    pub role: String,
    pub model: String,
    pub quality: f64,
    pub speed: f64,
    pub composite: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

// ── Ollama response types (reuse pattern from bench.rs) ────────────

#[derive(Deserialize, Default)]
#[allow(dead_code)]
struct OllamaGenResponse {
    #[serde(default)]
    response: String,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    total_duration: Option<u64>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ChatChoice {
    #[serde(default)]
    message: Option<ChatMessage>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

// ── Scoring ────────────────────────────────────────────────────────

/// Evaluate a model response against a set of scoring rules.
///
/// Returns a quality score clamped to 0-10.
pub fn evaluate_response(text: &str, rules: &[ScoringRule]) -> f64 {
    let mut score: i32 = 0;
    for rule in rules {
        let flags = if rule.case_insensitive { "(?i)" } else { "" };
        let pattern = format!("{}{}", flags, rule.pattern);
        let matched = Regex::new(&pattern)
            .map(|re| re.is_match(text))
            .unwrap_or(false);

        if rule.negate {
            if !matched {
                score += rule.weight;
            }
        } else if matched {
            score += rule.weight;
        }
    }
    (score.max(0).min(10)) as f64
}

/// Extract a code block from a markdown-fenced response.
///
/// Returns the content inside the first ``` block, or the full text
/// if no fences are found.
pub fn extract_code_block(text: &str) -> String {
    if let Some(start) = text.find("```") {
        // Skip past the opening fence + optional language tag
        let after_fence = &text[start + 3..];
        let content_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let content = &after_fence[content_start..];
        if let Some(end) = content.find("```") {
            return content[..end].trim().to_string();
        }
        return content.trim().to_string();
    }
    text.trim().to_string()
}

// ── HTTP helpers (following bench.rs ureq patterns) ────────────────

/// Raw response from an inference call, carrying text + timing info.
pub struct InferenceResponse {
    pub text: String,
    pub eval_count: u64,
    pub tok_per_sec: f64,
    pub ttft_ms: Option<f64>,
    pub wall_time_sec: f64,
}

pub fn quality_ollama_generate(
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f64,
) -> Result<InferenceResponse, String> {
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
    });

    let start = Instant::now();
    let resp = ureq::post(url)
        .config()
        .timeout_global(Some(Duration::from_secs(600)))
        .build()
        .send_json(&body)
        .map_err(|e| format!("Ollama request failed: {}", e))?;

    let wall_time = start.elapsed();

    let resp_body: OllamaGenResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("Ollama JSON parse error: {}", e))?;

    let eval_count = resp_body.eval_count.unwrap_or(0);

    let ttft_ms = resp_body
        .prompt_eval_duration
        .map(|ns| ns as f64 / 1_000_000.0);

    let tok_per_sec = if let (Some(ec), Some(ed)) = (resp_body.eval_count, resp_body.eval_duration)
    {
        if ed > 0 {
            ec as f64 / (ed as f64 / 1_000_000_000.0)
        } else {
            0.0
        }
    } else if eval_count > 0 {
        eval_count as f64 / wall_time.as_secs_f64()
    } else {
        0.0
    };

    Ok(InferenceResponse {
        text: resp_body.response,
        eval_count,
        tok_per_sec,
        ttft_ms,
        wall_time_sec: wall_time.as_secs_f64(),
    })
}

fn quality_openai_chat(
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f64,
) -> Result<InferenceResponse, String> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": false,
    });

    let start = Instant::now();
    let resp = ureq::post(url)
        .config()
        .timeout_global(Some(Duration::from_secs(600)))
        .build()
        .send_json(&body)
        .map_err(|e| format!("OpenAI-compat request failed: {}", e))?;

    let wall_time = start.elapsed();

    let completion: ChatCompletionResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;

    let usage = completion.usage.unwrap_or(ChatUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
    });

    let text = completion
        .choices
        .first()
        .and_then(|c| c.message.as_ref())
        .and_then(|m| m.content.clone())
        .unwrap_or_default();

    let eval_count = usage.completion_tokens as u64;
    let tok_per_sec = if eval_count > 0 && wall_time.as_secs_f64() > 0.0 {
        eval_count as f64 / wall_time.as_secs_f64()
    } else {
        0.0
    };

    Ok(InferenceResponse {
        text,
        eval_count,
        tok_per_sec,
        ttft_ms: None,
        wall_time_sec: wall_time.as_secs_f64(),
    })
}

// ── Public benchmark entry points ──────────────────────────────────

/// Run quality benchmarks against an Ollama endpoint.
///
/// Iterates through all roles/tests in `config`, sends prompts to the
/// model, scores responses, and returns aggregated results.
pub fn bench_quality_ollama(
    base_url: &str,
    model: &str,
    config: &QualityConfig,
    role_filter: Option<&[String]>,
) -> Result<ModelQualityResult, String> {
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));

    // Warmup
    let _ = quality_ollama_generate(&url, model, "Say hello.", 64, 0.3);

    run_all_tests(
        model,
        "ollama",
        config,
        role_filter,
        |prompt, max_tok, temp| quality_ollama_generate(&url, model, prompt, max_tok, temp),
    )
}

/// Run quality benchmarks against an OpenAI-compatible endpoint (vLLM, MLX).
pub fn bench_quality_openai_compat(
    base_url: &str,
    model: &str,
    provider: &str,
    config: &QualityConfig,
    role_filter: Option<&[String]>,
) -> Result<ModelQualityResult, String> {
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    // Warmup
    let _ = quality_openai_chat(&url, model, "Say hello.", 64, 0.3);

    run_all_tests(
        model,
        provider,
        config,
        role_filter,
        |prompt, max_tok, temp| quality_openai_chat(&url, model, prompt, max_tok, temp),
    )
}

/// Shared test runner used by both Ollama and OpenAI-compat paths.
fn run_all_tests<F>(
    model: &str,
    provider: &str,
    config: &QualityConfig,
    role_filter: Option<&[String]>,
    generate: F,
) -> Result<ModelQualityResult, String>
where
    F: Fn(&str, u32, f64) -> Result<InferenceResponse, String>,
{
    let mut all_results = Vec::new();
    let mut role_scores = Vec::new();

    for (role_name, role_def) in &config.roles {
        // Skip roles not in the filter (if a filter is set)
        if let Some(filter) = role_filter {
            if !filter.iter().any(|f| f == role_name) {
                continue;
            }
        }

        let mut role_results = Vec::new();

        for test_def in &role_def.tests {
            let max_tokens = test_def.max_tokens.unwrap_or(1024);
            let temperature = test_def.temperature.unwrap_or(0.3);

            let result = match generate(&test_def.prompt, max_tokens, temperature) {
                Ok(resp) => {
                    let quality = evaluate_response(&resp.text, &test_def.rules);
                    let speed_weight = test_def.speed_weight.unwrap_or(1.0);
                    let speed_norm = (resp.tok_per_sec / 3.0).min(10.0);
                    let composite =
                        (quality * 2.0 + speed_norm * speed_weight) / (2.0 + speed_weight);

                    QualityResult {
                        test_name: test_def.name.clone(),
                        role: role_name.clone(),
                        quality,
                        tok_per_sec: resp.tok_per_sec,
                        composite,
                        response_preview: resp.text.chars().take(150).collect(),
                        ttft_ms: resp.ttft_ms,
                        wall_time_sec: resp.wall_time_sec,
                        eval_tokens: resp.eval_count,
                        error: None,
                    }
                }
                Err(e) => QualityResult {
                    test_name: test_def.name.clone(),
                    role: role_name.clone(),
                    quality: 0.0,
                    tok_per_sec: 0.0,
                    composite: 0.0,
                    response_preview: String::new(),
                    ttft_ms: None,
                    wall_time_sec: 0.0,
                    eval_tokens: 0,
                    error: Some(e),
                },
            };

            role_results.push(result.clone());
            all_results.push(result);
        }

        // Compute role averages
        let n = role_results.len() as f64;
        if n > 0.0 {
            let avg_q = role_results.iter().map(|r| r.quality).sum::<f64>() / n;
            let avg_s = role_results.iter().map(|r| r.tok_per_sec).sum::<f64>() / n;
            let avg_c = role_results.iter().map(|r| r.composite).sum::<f64>() / n;

            role_scores.push(RoleScore {
                role: role_name.clone(),
                quality: (avg_q * 10.0).round() / 10.0,
                speed: (avg_s * 10.0).round() / 10.0,
                composite: (avg_c * 10.0).round() / 10.0,
                test_count: role_results.len(),
            });
        }
    }

    // Compute overall averages
    let n_roles = role_scores.len() as f64;
    let overall_quality = if n_roles > 0.0 {
        role_scores.iter().map(|r| r.quality).sum::<f64>() / n_roles
    } else {
        0.0
    };
    let overall_speed = if n_roles > 0.0 {
        role_scores.iter().map(|r| r.speed).sum::<f64>() / n_roles
    } else {
        0.0
    };
    let overall_composite = if n_roles > 0.0 {
        role_scores.iter().map(|r| r.composite).sum::<f64>() / n_roles
    } else {
        0.0
    };

    Ok(ModelQualityResult {
        model: model.to_string(),
        provider: provider.to_string(),
        roles: role_scores,
        test_results: all_results,
        overall_quality,
        overall_speed,
        overall_composite,
    })
}

// ── Routing matrix ─────────────────────────────────────────────────

/// Compute the best model for each role based on composite scores.
pub fn compute_routing(results: &[ModelQualityResult]) -> Vec<RoutingRecommendation> {
    let mut role_map: HashMap<String, Vec<(String, f64, f64, f64)>> = HashMap::new();

    for mr in results {
        for rs in &mr.roles {
            role_map.entry(rs.role.clone()).or_default().push((
                mr.model.clone(),
                rs.quality,
                rs.speed,
                rs.composite,
            ));
        }
    }

    let mut routing = Vec::new();
    for (role, mut scores) in role_map {
        scores.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((model, q, s, c)) = scores.first() {
            routing.push(RoutingRecommendation {
                role,
                model: model.clone(),
                quality: *q,
                speed: *s,
                composite: *c,
                note: None,
            });
        }
    }

    routing.sort_by(|a, b| a.role.cmp(&b.role));
    routing
}

/// Compute runner-up models for each role with contextual notes.
pub fn compute_runner_ups(results: &[ModelQualityResult]) -> Vec<RoutingRecommendation> {
    let mut role_map: HashMap<String, Vec<(String, f64, f64, f64)>> = HashMap::new();

    for mr in results {
        for rs in &mr.roles {
            role_map.entry(rs.role.clone()).or_default().push((
                mr.model.clone(),
                rs.quality,
                rs.speed,
                rs.composite,
            ));
        }
    }

    let mut runner_ups = Vec::new();
    for (role, mut scores) in role_map {
        scores.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        if scores.len() >= 2 {
            let (ref _best_model, best_q, best_s, best_c) = scores[0];
            let (ref ru_model, ru_q, ru_s, ru_c) = scores[1];

            let note = if ru_s > best_s * 1.5 {
                "consider for speed".to_string()
            } else if (ru_q - best_q).abs() < 1.0 {
                "close quality".to_string()
            } else if ru_c > best_c * 0.9 {
                "competitive composite".to_string()
            } else {
                "alternative".to_string()
            };

            runner_ups.push(RoutingRecommendation {
                role,
                model: ru_model.clone(),
                quality: ru_q,
                speed: ru_s,
                composite: ru_c,
                note: Some(note),
            });
        }
    }

    runner_ups.sort_by(|a, b| a.role.cmp(&b.role));
    runner_ups
}

// ── YAML config loading ────────────────────────────────────────────

/// Parse a YAML string into a `QualityConfig`.
pub fn load_quality_config(yaml: &str) -> Result<QualityConfig, String> {
    serde_yml::from_str(yaml).map_err(|e| format!("Failed to parse quality config: {}", e))
}

/// Return the built-in default quality config (embedded from `data/benchmarks.yaml`).
pub fn default_quality_config() -> QualityConfig {
    let yaml = include_str!("../data/benchmarks.yaml");
    load_quality_config(yaml).expect("embedded benchmarks.yaml is invalid")
}

// ── Display helpers ────────────────────────────────────────────────

impl ModelQualityResult {
    /// Print a human-readable summary of quality results.
    pub fn display(&self) {
        println!();
        println!("  === Quality Benchmark Results ===");
        println!("  Model:    {}", self.model);
        println!("  Provider: {}", self.provider);
        println!();
        println!(
            "  Overall:  quality={:.1}  speed={:.1} tok/s  composite={:.1}",
            self.overall_quality, self.overall_speed, self.overall_composite
        );
        println!();
        println!("  Role             Quality  Speed    Composite  Tests");
        println!("  ───────────────  ───────  ───────  ─────────  ─────");
        for rs in &self.roles {
            println!(
                "  {:<15}  {:>6.1}   {:>6.1}   {:>8.1}   {:>4}",
                rs.role, rs.quality, rs.speed, rs.composite, rs.test_count
            );
        }
        println!();
    }

    /// Print results as JSON.
    pub fn display_json(&self) {
        let json = serde_json::json!({
            "quality_benchmark": {
                "model": self.model,
                "provider": self.provider,
                "overall": {
                    "quality": self.overall_quality,
                    "speed": self.overall_speed,
                    "composite": self.overall_composite,
                },
                "role_scores": self.roles,
                "test_results": self.test_results,
            }
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).expect("JSON serialization failed")
        );
    }
}

impl RoutingRecommendation {
    /// Print a routing matrix row.
    pub fn display_row(&self) {
        let note_str = self
            .note
            .as_deref()
            .map(|n| format!("  ({})", n))
            .unwrap_or_default();
        println!(
            "  {:<17} -> {:<30}  q={:.1}  s={:.1}  c={:.1}{}",
            self.role, self.model, self.quality, self.speed, self.composite, note_str
        );
    }
}

/// Print a full routing matrix.
pub fn display_routing_matrix(
    routing: &[RoutingRecommendation],
    runner_ups: &[RoutingRecommendation],
) {
    println!();
    println!("  === Routing Matrix ===");
    println!("  Best model per role:");
    println!();
    for r in routing {
        r.display_row();
    }

    if !runner_ups.is_empty() {
        println!();
        println!("  Runner-ups:");
        println!();
        for r in runner_ups {
            r.display_row();
        }
    }
    println!();
}

// ── Frontier model baselines ──────────────────────────────────────

/// A baseline score for a frontier model on a specific role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineScore {
    pub quality: f64,
    pub speed: f64,
    pub composite: f64,
}

/// Baseline results for a frontier model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineModel {
    pub model: String,
    pub provider: String,
    pub roles: HashMap<String, BaselineScore>,
    pub overall: BaselineScore,
}

#[derive(Debug, Clone, Deserialize)]
struct BaselinesFile {
    baselines: Vec<BaselineModel>,
}

/// Load embedded frontier model baselines.
pub fn load_baselines() -> Vec<BaselineModel> {
    let json = include_str!("../data/baselines.json");
    serde_json::from_str::<BaselinesFile>(json)
        .map(|f| f.baselines)
        .unwrap_or_default()
}

/// Compare a local model's role scores against frontier baselines.
/// Returns (role, local_composite, baseline_name, baseline_composite, pct_of_frontier).
pub fn compare_to_baselines(
    result: &ModelQualityResult,
    baselines: &[BaselineModel],
) -> Vec<(String, f64, String, f64, f64)> {
    let mut comparisons = Vec::new();
    for rs in &result.roles {
        // Find best baseline for this role
        let best_baseline = baselines
            .iter()
            .filter_map(|b| {
                b.roles
                    .get(&rs.role)
                    .map(|bs| (b.model.as_str(), bs.composite))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((baseline_model, baseline_composite)) = best_baseline {
            let pct = if baseline_composite > 0.0 {
                (rs.composite / baseline_composite) * 100.0
            } else {
                0.0
            };
            comparisons.push((
                rs.role.clone(),
                rs.composite,
                baseline_model.to_string(),
                baseline_composite,
                pct,
            ));
        }
    }
    comparisons
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_response_basic_match() {
        let rules = vec![
            ScoringRule {
                pattern: "hello".to_string(),
                weight: 3,
                negate: false,
                case_insensitive: true,
            },
            ScoringRule {
                pattern: "world".to_string(),
                weight: 2,
                negate: false,
                case_insensitive: false,
            },
        ];
        assert!((evaluate_response("Hello world", &rules) - 5.0).abs() < 0.01);
        assert!((evaluate_response("Hello World", &rules) - 3.0).abs() < 0.01);
        assert!((evaluate_response("goodbye", &rules) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_response_negate() {
        let rules = vec![ScoringRule {
            pattern: "error".to_string(),
            weight: 5,
            negate: true,
            case_insensitive: false,
        }];
        // negate: score if pattern NOT found
        assert!((evaluate_response("all good", &rules) - 5.0).abs() < 0.01);
        assert!((evaluate_response("found error", &rules) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_response_clamp() {
        let rules = vec![
            ScoringRule {
                pattern: "a".to_string(),
                weight: 8,
                negate: false,
                case_insensitive: false,
            },
            ScoringRule {
                pattern: "b".to_string(),
                weight: 8,
                negate: false,
                case_insensitive: false,
            },
        ];
        // Both match = 16, but clamped to 10
        assert!((evaluate_response("a b", &rules) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_response_negative_clamp() {
        let rules = vec![ScoringRule {
            pattern: "bad".to_string(),
            weight: -5,
            negate: false,
            case_insensitive: false,
        }];
        // Negative score clamped to 0
        assert!((evaluate_response("bad", &rules) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_code_block() {
        let md = "Here is the code:\n```python\ndef hello():\n    print('hi')\n```\nDone.";
        assert_eq!(extract_code_block(md), "def hello():\n    print('hi')");
    }

    #[test]
    fn test_extract_code_block_no_fence() {
        let plain = "def hello():\n    print('hi')";
        assert_eq!(extract_code_block(plain), plain.trim());
    }

    #[test]
    fn test_load_quality_config() {
        let yaml = r#"
roles:
  general:
    description: General tasks
    tests:
      - name: test1
        prompt: "Say hello"
        rules:
          - { pattern: "hello", weight: 5, case_insensitive: true }
"#;
        let config = load_quality_config(yaml).unwrap();
        assert!(config.roles.contains_key("general"));
        assert_eq!(config.roles["general"].tests.len(), 1);
        assert_eq!(config.roles["general"].tests[0].rules.len(), 1);
    }

    #[test]
    fn test_default_quality_config_loads() {
        let config = default_quality_config();
        assert!(
            !config.roles.is_empty(),
            "default config should have at least one role"
        );
        // Check a known role exists
        assert!(
            config.roles.contains_key("general"),
            "default config should have 'general' role"
        );
    }

    #[test]
    fn test_compute_routing_single_model() {
        let results = vec![ModelQualityResult {
            model: "test-model".to_string(),
            provider: "ollama".to_string(),
            roles: vec![
                RoleScore {
                    role: "general".to_string(),
                    quality: 7.0,
                    speed: 30.0,
                    composite: 6.5,
                    test_count: 3,
                },
                RoleScore {
                    role: "coding".to_string(),
                    quality: 8.0,
                    speed: 25.0,
                    composite: 7.0,
                    test_count: 5,
                },
            ],
            test_results: vec![],
            overall_quality: 7.5,
            overall_speed: 27.5,
            overall_composite: 6.75,
        }];

        let routing = compute_routing(&results);
        assert_eq!(routing.len(), 2);
        assert!(routing.iter().all(|r| r.model == "test-model"));
    }

    #[test]
    fn test_compute_routing_picks_best() {
        let results = vec![
            ModelQualityResult {
                model: "fast-model".to_string(),
                provider: "ollama".to_string(),
                roles: vec![RoleScore {
                    role: "fast".to_string(),
                    quality: 5.0,
                    speed: 100.0,
                    composite: 8.0,
                    test_count: 2,
                }],
                test_results: vec![],
                overall_quality: 5.0,
                overall_speed: 100.0,
                overall_composite: 8.0,
            },
            ModelQualityResult {
                model: "smart-model".to_string(),
                provider: "ollama".to_string(),
                roles: vec![RoleScore {
                    role: "fast".to_string(),
                    quality: 9.0,
                    speed: 10.0,
                    composite: 7.0,
                    test_count: 2,
                }],
                test_results: vec![],
                overall_quality: 9.0,
                overall_speed: 10.0,
                overall_composite: 7.0,
            },
        ];

        let routing = compute_routing(&results);
        assert_eq!(routing.len(), 1);
        assert_eq!(routing[0].model, "fast-model"); // higher composite
    }

    #[test]
    fn test_compute_runner_ups() {
        let results = vec![
            ModelQualityResult {
                model: "best".to_string(),
                provider: "ollama".to_string(),
                roles: vec![RoleScore {
                    role: "coding".to_string(),
                    quality: 9.0,
                    speed: 20.0,
                    composite: 8.0,
                    test_count: 3,
                }],
                test_results: vec![],
                overall_quality: 9.0,
                overall_speed: 20.0,
                overall_composite: 8.0,
            },
            ModelQualityResult {
                model: "second".to_string(),
                provider: "ollama".to_string(),
                roles: vec![RoleScore {
                    role: "coding".to_string(),
                    quality: 7.0,
                    speed: 50.0,
                    composite: 6.0,
                    test_count: 3,
                }],
                test_results: vec![],
                overall_quality: 7.0,
                overall_speed: 50.0,
                overall_composite: 6.0,
            },
        ];

        let runner_ups = compute_runner_ups(&results);
        assert_eq!(runner_ups.len(), 1);
        assert_eq!(runner_ups[0].model, "second");
        assert!(runner_ups[0].note.is_some());
    }
}
