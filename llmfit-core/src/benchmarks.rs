//! Client for the localmaxxing.com benchmark API.
//!
//! Fetches real-world benchmark results (tok/s, TTFT, VRAM usage) for
//! hardware configurations that match the user's detected system specs.
//! Includes an embedded cache as fallback when the API is unreachable.

use crate::hardware::{GpuBackend, SystemSpecs};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

const BASE_URL: &str = "https://localmaxxing.com/api";

// Embedded benchmark cache — scraped by scripts/scrape_benchmarks.py
const BENCHMARK_CACHE_JSON: &str = include_str!("../data/benchmark_cache.json");

// ── Response types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkEntry {
    pub id: String,
    #[serde(default)]
    pub hf_id: String,
    #[serde(default)]
    pub engine_name: String,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub tok_s_out: Option<f64>,
    #[serde(default)]
    pub tok_s_total: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub peak_vram_gb: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub hardware: Option<HardwareInfo>,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub verified: Option<bool>,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    #[serde(default)]
    pub hw_class: Option<String>,
    #[serde(default)]
    pub gpu_name: Option<String>,
    #[serde(default)]
    pub vram_gb: Option<f64>,
    #[serde(default)]
    pub gpu_count: Option<u32>,
    #[serde(default)]
    pub chip_vendor: Option<String>,
    #[serde(default)]
    pub chip_family: Option<String>,
    #[serde(default)]
    pub chip_variant: Option<String>,
    #[serde(default)]
    pub unified_memory_gb: Option<f64>,
    #[serde(default)]
    pub cpu: Option<String>,
    #[serde(default)]
    pub ram_gb: Option<f64>,
    #[serde(default)]
    pub os: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEntry {
    pub id: String,
    #[serde(default)]
    pub tok_s_out: Option<f64>,
    #[serde(default)]
    pub tok_s_total: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub peak_vram_gb: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub model: Option<LeaderboardModel>,
    #[serde(default)]
    pub hardware: Option<HardwareInfo>,
    #[serde(default)]
    pub engine: Option<LeaderboardEngine>,
    #[serde(default)]
    pub user: Option<LeaderboardUser>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardModel {
    #[serde(default)]
    pub hf_id: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub params: Option<f64>,
    #[serde(default)]
    pub is_mo_e: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEngine {
    #[serde(default)]
    pub engine_name: String,
    #[serde(default)]
    pub engine_version: Option<String>,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub backend: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardUser {
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub verified: Option<bool>,
}

impl LeaderboardEntry {
    /// Helper to get the model HF ID.
    pub fn hf_id(&self) -> &str {
        self.model.as_ref().map(|m| m.hf_id.as_str()).unwrap_or("")
    }

    /// Helper to get the engine name.
    pub fn engine_name(&self) -> &str {
        self.engine
            .as_ref()
            .map(|e| e.engine_name.as_str())
            .unwrap_or("")
    }

    /// Helper to get the quantization.
    pub fn quantization(&self) -> &str {
        self.engine
            .as_ref()
            .map(|e| e.quantization.as_str())
            .unwrap_or("")
    }

    /// Helper to get the username.
    pub fn username(&self) -> &str {
        self.user
            .as_ref()
            .and_then(|u| u.username.as_deref())
            .unwrap_or("anon")
    }

    /// Helper to check verified status.
    pub fn verified(&self) -> bool {
        self.user.as_ref().and_then(|u| u.verified).unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResponse {
    pub benchmarks: Vec<BenchmarkEntry>,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub limit: u64,
    #[serde(default)]
    pub offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardResponse {
    pub rows: Vec<LeaderboardEntry>,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub limit: u64,
    #[serde(default)]
    pub offset: u64,
}

// ── Query builder ────────────────────────────────────────────────────

/// Map detected hardware to API query parameters for matching benchmarks.
pub fn hw_query_params(specs: &SystemSpecs) -> Vec<(&'static str, String)> {
    let mut params: Vec<(&str, String)> = Vec::new();

    if specs.unified_memory {
        params.push(("hwClass", "UNIFIED".to_string()));

        // Apple Silicon
        if specs.backend == GpuBackend::Metal {
            params.push(("chipVendor", "apple".to_string()));
            if let Some(ref gpu) = specs.gpu_name {
                // e.g. "Apple M2 Max" → chipFamily "m2", chipVariant "max"
                let lower = gpu.to_lowercase();
                if let Some(rest) = lower.strip_prefix("apple ") {
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    if !parts.is_empty() {
                        params.push(("chipFamily", parts[0].to_string()));
                    }
                    if parts.len() > 1 {
                        params.push(("chipVariant", parts[1].to_string()));
                    }
                }
            }
        }
    } else if specs.has_gpu {
        params.push(("hwClass", "DISCRETE_GPU".to_string()));

        if let Some(ref name) = specs.gpu_name {
            params.push(("gpuName", name.clone()));
        }
    } else {
        params.push(("hwClass", "CPU_ONLY".to_string()));
    }

    params
}

/// Map detected hardware to leaderboard query parameters.
pub fn hw_leaderboard_params(specs: &SystemSpecs) -> Vec<(&'static str, String)> {
    let mut params: Vec<(&str, String)> = Vec::new();

    if specs.unified_memory {
        params.push(("hwClass", "UNIFIED".to_string()));
    } else if specs.has_gpu {
        params.push(("hwClass", "DISCRETE_GPU".to_string()));
    } else {
        params.push(("hwClass", "CPU_ONLY".to_string()));
    }

    // Use hardware name for fuzzy match
    if let Some(ref name) = specs.gpu_name {
        params.push(("hardwareName", name.clone()));
    }

    // VRAM tier
    if let Some(vram) = specs.total_gpu_vram_gb {
        let tier = nearest_mem_tier(vram);
        if tier > 0 {
            params.push(("memTier", tier.to_string()));
        }
    } else if specs.unified_memory {
        let tier = nearest_mem_tier(specs.total_ram_gb);
        if tier > 0 {
            params.push(("memTier", tier.to_string()));
        }
    }

    // OS
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    };
    params.push(("os", os.to_string()));

    params
}

fn nearest_mem_tier(gb: f64) -> u32 {
    const TIERS: [u32; 9] = [8, 12, 16, 24, 32, 48, 80, 96, 128];
    let mut best = 0u32;
    let mut best_dist = f64::MAX;
    for &t in &TIERS {
        let d = (gb - t as f64).abs();
        if d < best_dist {
            best_dist = d;
            best = t;
        }
    }
    best
}

// ── Embedded cache ───────────────────────────────────────────────────

/// Cache structure matching the scraper output.
#[derive(Debug, Clone, Deserialize)]
struct BenchmarkCache {
    #[serde(default)]
    scraped_at: Option<String>,
    #[serde(default)]
    presets: std::collections::HashMap<String, CachedPreset>,
}

#[derive(Debug, Clone, Deserialize)]
struct CachedPreset {
    rows: Vec<LeaderboardEntry>,
    #[serde(default)]
    total: u64,
}

/// Lazily parsed embedded benchmark cache.
fn embedded_cache() -> &'static BenchmarkCache {
    static CACHE: OnceLock<BenchmarkCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        serde_json::from_str(BENCHMARK_CACHE_JSON).unwrap_or_else(|_| BenchmarkCache {
            scraped_at: None,
            presets: std::collections::HashMap::new(),
        })
    })
}

/// Look up cached leaderboard data for a hardware preset label.
pub fn cached_leaderboard_for_preset(label: &str) -> Option<LeaderboardResponse> {
    let cache = embedded_cache();
    cache.presets.get(label).map(|p| LeaderboardResponse {
        rows: p.rows.clone(),
        total: p.total,
        limit: p.rows.len() as u64,
        offset: 0,
    })
}

/// Returns the scrape timestamp of the embedded cache, if available.
pub fn cache_timestamp() -> Option<&'static str> {
    embedded_cache().scraped_at.as_deref()
}

// ── Fetch functions ──────────────────────────────────────────────────

/// Fetch benchmarks matching the user's hardware.
pub fn fetch_benchmarks(
    specs: &SystemSpecs,
    api_key: Option<&str>,
    limit: u32,
) -> Result<BenchmarkResponse, String> {
    let mut params = hw_query_params(specs);
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/benchmarks?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: BenchmarkResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

/// Fetch benchmarks for a specific model on matching hardware.
pub fn fetch_benchmarks_for_model(
    specs: &SystemSpecs,
    hf_id: &str,
    api_key: Option<&str>,
    limit: u32,
) -> Result<BenchmarkResponse, String> {
    let mut params = hw_query_params(specs);
    params.push(("hfId", hf_id.to_string()));
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/benchmarks?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: BenchmarkResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

/// Fetch the leaderboard filtered to matching hardware.
pub fn fetch_leaderboard(
    specs: &SystemSpecs,
    api_key: Option<&str>,
    limit: u32,
) -> Result<LeaderboardResponse, String> {
    let mut params = hw_leaderboard_params(specs);
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/leaderboard?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: LeaderboardResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

// ── Hardware presets ─────────────────────────────────────────────────

/// A selectable hardware configuration for browsing benchmarks.
#[derive(Debug, Clone)]
pub struct HardwarePreset {
    pub label: &'static str,
    pub hw_class: &'static str,
    pub hardware_name: Option<&'static str>,
    pub mem_tier: Option<u32>,
}

impl HardwarePreset {
    /// Returns the built-in list of popular hardware presets.
    pub fn all() -> &'static [HardwarePreset] {
        &HARDWARE_PRESETS
    }
}

static HARDWARE_PRESETS: [HardwarePreset; 27] = [
    // NVIDIA consumer
    HardwarePreset {
        label: "RTX 5090 (32 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 5090"),
        mem_tier: Some(32),
    },
    HardwarePreset {
        label: "RTX 5080 (16 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 5080"),
        mem_tier: Some(16),
    },
    HardwarePreset {
        label: "RTX 4090 (24 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 4090"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "RTX 4080 (16 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 4080"),
        mem_tier: Some(16),
    },
    HardwarePreset {
        label: "RTX 4070 Ti (12 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 4070"),
        mem_tier: Some(12),
    },
    HardwarePreset {
        label: "RTX 3090 (24 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 3090"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "RTX 3080 (10 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 3080"),
        mem_tier: Some(12),
    },
    HardwarePreset {
        label: "RTX 3060 (12 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 3060"),
        mem_tier: Some(12),
    },
    // NVIDIA datacenter
    HardwarePreset {
        label: "A100 (80 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("A100"),
        mem_tier: Some(80),
    },
    HardwarePreset {
        label: "A100 (40 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("A100"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "H100 (80 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("H100"),
        mem_tier: Some(80),
    },
    HardwarePreset {
        label: "L40S (48 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("L40S"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "T4 (16 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("T4"),
        mem_tier: Some(16),
    },
    // AMD
    HardwarePreset {
        label: "RX 7900 XTX (24 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("7900 XTX"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "RX 7900 XT (20 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("7900 XT"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "MI300X (192 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("MI300X"),
        mem_tier: Some(128),
    },
    // Apple Silicon
    HardwarePreset {
        label: "Apple M4 Max (128 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Max"),
        mem_tier: Some(128),
    },
    HardwarePreset {
        label: "Apple M4 Max (64 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Max"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "Apple M4 Pro (48 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Pro"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "Apple M4 Pro (24 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Pro"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "Apple M3 Max (128 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M3 Max"),
        mem_tier: Some(128),
    },
    HardwarePreset {
        label: "Apple M3 Max (96 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M3 Max"),
        mem_tier: Some(96),
    },
    HardwarePreset {
        label: "Apple M2 Ultra (192 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M2 Ultra"),
        mem_tier: Some(128),
    },
    HardwarePreset {
        label: "Apple M2 Max (96 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M2 Max"),
        mem_tier: Some(96),
    },
    HardwarePreset {
        label: "Apple M2 Pro (32 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M2 Pro"),
        mem_tier: Some(32),
    },
    HardwarePreset {
        label: "Apple M1 Max (64 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M1 Max"),
        mem_tier: Some(48),
    },
    // CPU only
    HardwarePreset {
        label: "CPU Only",
        hw_class: "CPU_ONLY",
        hardware_name: None,
        mem_tier: None,
    },
];

/// Fetch leaderboard for a specific hardware preset.
pub fn fetch_leaderboard_for_preset(
    preset: &HardwarePreset,
    api_key: Option<&str>,
    limit: u32,
) -> Result<LeaderboardResponse, String> {
    let mut params: Vec<(&str, String)> = Vec::new();
    params.push(("hwClass", preset.hw_class.to_string()));
    if let Some(name) = preset.hardware_name {
        params.push(("hardwareName", name.to_string()));
    }
    if let Some(tier) = preset.mem_tier {
        params.push(("memTier", tier.to_string()));
    }
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/leaderboard?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: LeaderboardResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

/// Minimal percent-encoding for query values.
fn urlencoded(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push(char::from(b"0123456789ABCDEF"[(b >> 4) as usize]));
                out.push(char::from(b"0123456789ABCDEF"[(b & 0xf) as usize]));
            }
        }
    }
    out
}
