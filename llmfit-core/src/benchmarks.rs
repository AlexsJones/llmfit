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

// Community submissions contributed via `llmfit bench --share`, aggregated
// from data/community/ by build.rs. Merged submission → next release ships it.
const COMMUNITY_BENCH_JSON: &str =
    include_str!(concat!(env!("OUT_DIR"), "/community_benchmarks.json"));

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
    pub engine_flags: Option<LeaderboardEngineFlags>,
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

/// Per-run engine acceleration flags. Speculative decoding and MTP runs
/// measure draft-accelerated throughput, which can exceed the memory
/// bandwidth roofline that plain autoregressive estimates model — consumers
/// comparing against `estimate_tps` must filter these out.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEngineFlags {
    #[serde(default)]
    pub spec_decoding: Option<bool>,
    #[serde(default)]
    pub mtp_enabled: Option<bool>,
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
        let tier = lookup_mem_tier(vram);
        if tier > 0 {
            params.push(("memTier", tier.to_string()));
        }
    } else if specs.unified_memory {
        let tier = lookup_mem_tier(specs.total_ram_gb);
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

/// Bucket a memory size into the tier used as a *lookup key* against the
/// cached community results.
///
/// Nearest-match is the right rule here and the wrong one in `share.rs`: a
/// lookup wants the closest bucket that has data even when the fit is not
/// exact, because returning nothing is worse than returning an approximation.
/// A declared capacity wants the opposite.
///
/// That is why this is not the same function as `declared_mem_tier` in
/// `share.rs` despite looking almost identical, and why the two ladders are
/// allowed to differ. Please do not unify them.
fn lookup_mem_tier(gb: f64) -> u32 {
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

/// Number of benchmarks in the embedded cache for a hardware preset label.
/// Uses the server-reported total from scrape time when present (the cached
/// rows themselves are capped per preset).
pub fn cached_preset_benchmark_count(label: &str) -> Option<u64> {
    embedded_cache()
        .presets
        .get(label)
        .map(|p| p.total.max(p.rows.len() as u64))
}

/// Returns the scrape timestamp of the embedded cache, if available.
pub fn cache_timestamp() -> Option<&'static str> {
    embedded_cache().scraped_at.as_deref()
}

/// All preset labels present in the embedded benchmark cache. Used by the
/// estimate-calibration test to replay every cached measurement.
pub fn cached_preset_labels() -> Vec<&'static str> {
    let mut labels: Vec<&'static str> = embedded_cache()
        .presets
        .keys()
        .map(|s| s.as_str())
        .collect();
    labels.sort_unstable();
    labels
}

// ── Measured throughput lookup (provenance-weighted estimates) ──────

/// Where a measured throughput came from. Priority when annotating fits:
/// your own runs, then llmfit community submissions with matching CPU and GPU
/// names, then localmaxxing medians on matching presets —
/// all of which outrank the formula estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeasuredSource {
    /// localmaxxing.com leaderboard runs on hardware matching the detected GPU.
    #[default]
    Community,
    /// `llmfit bench --share` submissions merged into the llmfit repo,
    /// recorded with matching CPU and GPU names.
    CommunityLlmfit,
    /// `llmfit bench` runs recorded in this machine's local store.
    LocalBench,
}

/// A real measured throughput from benchmark runs — either the community
/// leaderboard on matching hardware, or the user's own `llmfit bench` runs on
/// this very machine. When present, this is ground truth and should be
/// displayed with priority over the formula estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredTps {
    /// Measured generation throughput (tok/s): community median, or the most
    /// recent local run.
    pub tok_s: f64,
    /// Number of runs behind the value.
    pub sample_count: u32,
    /// Hardware context for the runs. Match quality stays in `match_grade`.
    pub hardware_label: String,
    #[serde(default)]
    pub source: MeasuredSource,
    /// PCIe match quality for local and llmfit community runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub match_grade: Option<HardwareMatch>,
}

/// Number of machines `specs` describes: the cluster's node count in cluster
/// mode, otherwise 1.
pub fn spec_node_count(specs: &SystemSpecs) -> u32 {
    if specs.cluster_mode {
        specs.cluster_node_count.max(1)
    } else {
        1
    }
}

/// The deployment layout a measured-throughput lookup must match (#1227).
///
/// A measured result only stands in for an estimate made for the same
/// layout. Results that don't record a node count are single-node, so they
/// never match a cluster lookup, and multi-node results never match a
/// single-node one. When the lookup knows its tensor- or pipeline-parallel
/// size, the result must have the same size (an unrecorded size counts as 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutQuery {
    /// Machines the estimate is for. Always at least 1.
    pub node_count: u32,
    /// Tensor-parallel size of the estimate, when it has one.
    pub tp_size: Option<u32>,
    /// Pipeline-parallel size of the estimate, when it has one.
    pub pp_size: Option<u32>,
}

impl LayoutQuery {
    /// One machine, with no parallelism constraint.
    pub const SINGLE_NODE: Self = Self {
        node_count: 1,
        tp_size: None,
        pp_size: None,
    };

    /// Match on node count only, for lookups made without a fit.
    pub fn for_specs(specs: &SystemSpecs) -> Self {
        Self {
            node_count: spec_node_count(specs),
            ..Self::SINGLE_NODE
        }
    }

    /// Match on node count and the fit's parallel layout.
    pub fn for_fit(specs: &SystemSpecs, fit: &crate::fit::ModelFit) -> Self {
        Self {
            tp_size: fit.tp_size,
            // TODO(#1226): key on the fit's pipeline-parallel size once
            // `ModelFit` records one. Cluster fits are tensor-parallel only
            // until then, and `tp_size` already separates the layouts.
            pp_size: None,
            ..Self::for_specs(specs)
        }
    }

    /// Whether a result recorded with `layout` may stand in for this lookup.
    pub fn accepts(&self, layout: &crate::bench::BenchLayout) -> bool {
        layout.nodes() == self.node_count
            && self.tp_size.is_none_or(|tp| layout.tp() == tp)
            && self.pp_size.is_none_or(|pp| layout.pp() == pp)
    }
}

/// Find the embedded-cache hardware preset matching the detected GPU.
pub fn cached_preset_for_specs(specs: &SystemSpecs) -> Option<&'static HardwarePreset> {
    let gpu = specs.gpu_name.as_deref()?.to_lowercase();
    HardwarePreset::all().iter().find(|p| {
        p.hardware_name
            .is_some_and(|hw| gpu.contains(&hw.to_lowercase()))
    })
}

/// Prebuilt index of comparable measured runs for the detected hardware.
/// Build once per fit pass, then O(1) lookups per model.
pub struct MeasuredTpsIndex {
    hardware_label: &'static str,
    /// (model slug, quantization lowercase) -> sorted tok/s samples.
    samples: std::collections::HashMap<(String, String), Vec<f64>>,
}

impl MeasuredTpsIndex {
    /// Build the index from the embedded cache for the preset matching the
    /// detected GPU. Only rows comparable to `estimated_tps` are indexed:
    /// single-request generation (batch <= 1), no speculative decoding /
    /// MTP. Returns None when the GPU has no preset or no rows qualify.
    ///
    /// localmaxxing rows carry no node count, so they are single-node results
    /// and never stand in for a multi-node cluster (#1227). Without this, a
    /// cluster's GPU label such as `NVIDIA GB10 (×3)` would match the
    /// single-GPU preset.
    pub fn for_specs(specs: &SystemSpecs) -> Option<Self> {
        if spec_node_count(specs) > 1 {
            return None;
        }
        let preset = cached_preset_for_specs(specs)?;
        let resp = cached_leaderboard_for_preset(preset.label)?;
        Self::from_rows(&resp.rows, preset.label)
    }

    fn from_rows(rows: &[LeaderboardEntry], hardware_label: &'static str) -> Option<Self> {
        let mut samples: std::collections::HashMap<(String, String), Vec<f64>> =
            std::collections::HashMap::new();
        for row in rows {
            if row.batch_size.unwrap_or(1) > 1 {
                continue;
            }
            if row
                .engine_flags
                .as_ref()
                .is_some_and(|f| f.spec_decoding.unwrap_or(false) || f.mtp_enabled.unwrap_or(false))
            {
                continue;
            }
            let Some(tok_s) = row.tok_s_out.filter(|t| *t > 0.5) else {
                continue;
            };
            let hf_id = row.hf_id();
            let quant = row.quantization();
            if hf_id.is_empty() || quant.is_empty() {
                continue;
            }
            samples
                .entry((crate::models::canonical_slug(hf_id), quant.to_lowercase()))
                .or_default()
                .push(tok_s);
        }
        if samples.is_empty() {
            return None;
        }
        for s in samples.values_mut() {
            s.sort_by(|a, b| a.partial_cmp(b).expect("tok/s samples are finite"));
        }
        Some(Self {
            hardware_label,
            samples,
        })
    }

    /// Median measured tok/s for this model+quant, if any comparable runs
    /// exist on the matched hardware.
    pub fn lookup(&self, model_hf_id: &str, quant: &str) -> Option<MeasuredTps> {
        let key = (
            crate::models::canonical_slug(model_hf_id),
            quant.to_lowercase(),
        );
        let s = self.samples.get(&key)?;
        let n = s.len();
        let median = if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        };
        Some(MeasuredTps {
            tok_s: median,
            sample_count: n as u32,
            hardware_label: self.hardware_label.to_string(),
            source: MeasuredSource::Community,
            match_grade: None,
        })
    }
}

/// One-shot lookup: median measured tok/s for `model_hf_id` at `quant` on
/// hardware matching the detected specs. Prefer [`MeasuredTpsIndex`] when
/// annotating many fits.
pub fn measured_tps_for(
    specs: &SystemSpecs,
    model_hf_id: &str,
    quant: &str,
) -> Option<MeasuredTps> {
    MeasuredTpsIndex::for_specs(specs)?.lookup(model_hf_id, quant)
}

// ── Embedded llmfit community submissions ───────────────────────────

/// All community benchmark submissions embedded at build time (see
/// build.rs). Each element is a full submission payload conforming to
/// `data/community/schema.json`.
pub fn community_submissions() -> &'static [serde_json::Value] {
    static CACHE: OnceLock<Vec<serde_json::Value>> = OnceLock::new();
    CACHE.get_or_init(|| serde_json::from_str(COMMUNITY_BENCH_JSON).unwrap_or_default())
}

/// Links for the selected GPU group. Mixed GPU models have unknown topology.
pub(crate) fn benchmark_pcie_links(specs: &SystemSpecs) -> Vec<crate::hardware::PcieLink> {
    let Some(name) = specs.gpu_name.as_deref() else {
        return Vec::new();
    };
    let Some(gpu) = specs
        .gpus
        .iter()
        .find(|gpu| gpu.name.eq_ignore_ascii_case(name))
    else {
        return Vec::new();
    };
    if gpu.count != specs.gpu_count
        || gpu.pcie_links.len() != gpu.count as usize
        || gpu
            .pcie_links
            .iter()
            .any(|link| link.width.is_none_or(|width| width == 0))
    {
        return Vec::new();
    }
    gpu.pcie_links.clone()
}

fn recorded_pcie_widths(hw: &serde_json::Value) -> Option<Vec<u8>> {
    let links = hw.get("pcieLinks")?.as_array()?;
    if links.is_empty() {
        return None;
    }
    let mut widths: Vec<u8> = links
        .iter()
        .map(|link| {
            let width = link["width"].as_f64()?;
            (width.fract() == 0.0 && (1.0..=255.0).contains(&width)).then_some(width as u8)
        })
        .collect::<Option<_>>()?;
    widths.sort_unstable();
    Some(widths)
}

/// PCIe evidence behind a local or llmfit community benchmark match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareMatch {
    /// Both sides report the same complete set of PCIe widths.
    ExactPcie,
    /// CPU and GPU names match, but at least one side lacks complete widths.
    Legacy,
}

impl HardwareMatch {
    pub fn label(self) -> &'static str {
        match self {
            Self::ExactPcie => "exact PCIe widths",
            Self::Legacy => "CPU/GPU fallback (PCIe unverified)",
        }
    }
}

/// Match CPU, GPU, and known PCIe widths across local and community results.
/// Records without a GPU count remain eligible.
/// Missing links on either side use the CPU and GPU match.
/// Reject a recorded GPU count when it is invalid or differs from the current count.
/// Reject recorded links when their widths are invalid.
/// Ignore generation because it can fall while the GPU is idle.
pub fn hardware_payload_match_grade(
    hw: &serde_json::Value,
    specs: &SystemSpecs,
) -> Option<HardwareMatch> {
    let cpu_ok = hw["cpu"]
        .as_str()
        .is_some_and(|c| c.eq_ignore_ascii_case(&specs.cpu_name));
    let gpu_ok = match (&specs.gpu_name, hw["hardwareName"].as_str()) {
        (Some(now), Some(then)) => now.eq_ignore_ascii_case(then),
        (None, None) => true,
        _ => false,
    };
    if !cpu_ok || !gpu_ok {
        return None;
    }
    if let Some(recorded) = hw.get("gpuCount") {
        let count = recorded
            .as_u64()
            .and_then(|count| u32::try_from(count).ok())?;
        if count != specs.gpu_count || (specs.gpu_name.is_some() && count == 0) {
            return None;
        }
    }
    let links = benchmark_pcie_links(specs);
    let current_widths = links
        .iter()
        .map(|link| link.width.filter(|width| *width > 0))
        .collect::<Option<Vec<_>>>();
    match (current_widths, recorded_pcie_widths(hw)) {
        (Some(mut current), Some(recorded)) if !current.is_empty() => {
            current.sort_unstable();
            (current == recorded).then_some(HardwareMatch::ExactPcie)
        }
        (Some(current), Some(_)) if current.is_empty() => Some(HardwareMatch::Legacy),
        (_, None) if hw.get("pcieLinks").is_none() => Some(HardwareMatch::Legacy),
        _ => None,
    }
}

/// Keep the boolean matcher for callers that only need eligibility.
pub fn hardware_payload_matches(hw: &serde_json::Value, specs: &SystemSpecs) -> bool {
    hardware_payload_match_grade(hw, specs).is_some()
}

/// One benchmark result from a community submission, for leaderboard display.
#[derive(Debug, Clone)]
pub struct CommunityResult {
    pub model: String,
    pub provider: String,
    pub avg_tps: f64,
    pub ttft_ms: Option<f64>,
    pub match_grade: HardwareMatch,
    /// Deployment layout. Empty for single-node results.
    pub layout: crate::bench::BenchLayout,
}

/// Community results recorded on hardware matching `specs`, newest
/// submission first. Only results with the same node count as `specs` are
/// returned (#1227); results with an invalid layout are skipped.
pub fn community_results_for_specs(specs: &SystemSpecs) -> Vec<CommunityResult> {
    community_results_in(community_submissions(), specs)
}

fn community_results_in(
    submissions: &[serde_json::Value],
    specs: &SystemSpecs,
) -> Vec<CommunityResult> {
    let nodes = spec_node_count(specs);
    let mut subs: Vec<(&serde_json::Value, HardwareMatch)> = submissions
        .iter()
        .filter_map(|s| hardware_payload_match_grade(&s["hardware"], specs).map(|grade| (s, grade)))
        .collect();
    subs.sort_by_key(|(s, _)| std::cmp::Reverse(s["submittedAtUnix"].as_u64().unwrap_or(0)));

    let mut out = Vec::new();
    for (s, match_grade) in subs {
        for r in s["results"]
            .as_array()
            .map(|v| v.as_slice())
            .unwrap_or_default()
        {
            let Some(tps) = r["avgTps"]
                .as_f64()
                .filter(|t| crate::bench::is_plausible_tps(*t))
            else {
                continue;
            };
            let Some(layout) = crate::bench::BenchLayout::from_payload_result(r)
                .filter(|layout| layout.nodes() == nodes)
            else {
                continue;
            };
            out.push(CommunityResult {
                model: r["model"].as_str().unwrap_or("?").to_string(),
                provider: r["provider"].as_str().unwrap_or("").to_string(),
                avg_tps: tps,
                ttft_ms: r["avgTtftMs"].as_f64(),
                match_grade,
                layout,
            });
        }
    }
    out
}

/// Index of embedded community submissions with matching CPU and GPU names.
/// It annotates fit rows. It sits between local runs and preset medians.
/// A fresh install can use these runs for measured values and calibration.
pub struct CommunityBenchIndex {
    /// Newest submission first.
    entries: Vec<CommunityResult>,
}

impl CommunityBenchIndex {
    pub fn for_specs(specs: &SystemSpecs) -> Option<Self> {
        let entries = community_results_for_specs(specs);
        (!entries.is_empty()).then_some(Self { entries })
    }

    /// Median community tok/s for a catalog model, from results whose layout
    /// `layout` accepts. Use exact PCIe runs when they exist for that model.
    pub fn lookup(&self, model_hf_name: &str, layout: &LayoutQuery) -> Option<MeasuredTps> {
        let matching: Vec<&CommunityResult> = self
            .entries
            .iter()
            .filter(|r| layout.accepts(&r.layout))
            .filter(|r| crate::providers::tag_matches_model(&r.model, model_hf_name))
            .collect();
        let match_grade = matching
            .iter()
            .any(|r| r.match_grade == HardwareMatch::ExactPcie)
            .then_some(HardwareMatch::ExactPcie)
            .or_else(|| matching.first().map(|_| HardwareMatch::Legacy))?;
        let mut matches: Vec<f64> = matching
            .into_iter()
            .filter(|r| r.match_grade == match_grade)
            .map(|r| r.avg_tps)
            .collect();
        matches.sort_by(|a, b| a.partial_cmp(b).expect("tok/s values are finite"));
        let n = matches.len();
        let median = if n % 2 == 1 {
            matches[n / 2]
        } else {
            (matches[n / 2 - 1] + matches[n / 2]) / 2.0
        };
        Some(MeasuredTps {
            tok_s: median,
            sample_count: n as u32,
            hardware_label: "matching CPU/GPU".to_string(),
            source: MeasuredSource::CommunityLlmfit,
            match_grade: Some(match_grade),
        })
    }
}

// ── Fetch functions ──────────────────────────────────────────────────

/// GET a benchmark-API URL and parse the JSON response. All API calls go
/// through here so they share one timeout — without it a black-holed
/// connection blocks the caller indefinitely.
fn get_json<T: serde::de::DeserializeOwned>(url: &str, api_key: Option<&str>) -> Result<T, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .build()
        .into();
    let mut req = agent.get(url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    resp.into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))
}

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
    get_json(&url, api_key)
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
    get_json(&url, api_key)
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
    get_json(&url, api_key)
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
    get_json(&url, api_key)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn row(json: &str) -> LeaderboardEntry {
        serde_json::from_str(json).expect("valid test row")
    }

    fn specs(cpu: &str, gpu: Option<&str>) -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: cpu.to_string(),
            has_gpu: gpu.is_some(),
            gpu_vram_gb: None,
            total_gpu_vram_gb: None,
            gpu_available_gb: None,
            gpu_name: gpu.map(str::to_string),
            gpu_count: u32::from(gpu.is_some()),
            unified_memory: false,
            backend: GpuBackend::CpuX86,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    #[test]
    fn community_embed_parses_and_is_seeded() {
        let subs = community_submissions();
        // data/community/ ships with at least the first genuine submission;
        // this breaks if build.rs stops aggregating it into the binary.
        assert!(!subs.is_empty(), "embedded community aggregate is empty");
        for s in subs {
            assert_eq!(s["schemaVersion"], 1, "unexpected submission shape");
            assert!(s["results"].as_array().is_some_and(|r| !r.is_empty()));
            assert!(s["hardware"]["cpu"].is_string());
        }
    }

    #[test]
    fn community_results_filter_by_hardware() {
        use crate::hardware::{GpuInfo, PcieLink};
        // The seeded submissions have no PCIe widths.
        let seeded = specs(
            "Intel(R) Core(TM) Ultra 7 258V",
            Some("Intel Arc Graphics 130V/140V (integrated)"),
        );
        let rows = community_results_for_specs(&seeded);
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|r| r.avg_tps > 0.0));
        assert!(rows.iter().all(|r| r.match_grade == HardwareMatch::Legacy));

        let mut upgraded = seeded.clone();
        upgraded.gpus.push(GpuInfo {
            name: upgraded.gpu_name.clone().expect("seeded GPU name"),
            vram_gb: None,
            backend: GpuBackend::CpuX86,
            count: 1,
            unified_memory: false,
            free_vram_gb: None,
            free_vram_per_card_gb: Vec::new(),
            pcie_links: vec![PcieLink {
                index: Some(0),
                generation: Some(4),
                width: Some(16),
                max_generation: Some(4),
                max_width: Some(16),
            }],
        });
        assert_eq!(community_results_for_specs(&upgraded).len(), rows.len());

        // A different machine gets nothing.
        let other = specs("AMD Ryzen 9 7950X", Some("NVIDIA GeForce RTX 4090"));
        assert!(community_results_for_specs(&other).is_empty());
    }

    #[test]
    fn hardware_payload_matching_rules() {
        use serde_json::json;
        let hw = json!({"cpu": "Test CPU", "hardwareName": "Test GPU"});
        assert!(hardware_payload_matches(
            &hw,
            &specs("Test CPU", Some("Test GPU"))
        ));
        assert!(hardware_payload_matches(
            &hw,
            &specs("test cpu", Some("test gpu"))
        ));
        assert!(!hardware_payload_matches(&hw, &specs("Test CPU", None)));
        assert!(!hardware_payload_matches(
            &hw,
            &specs("Other CPU", Some("Test GPU"))
        ));

        let cpu_only = json!({"cpu": "Test CPU", "hardwareName": null});
        assert!(hardware_payload_matches(
            &cpu_only,
            &specs("Test CPU", None)
        ));
        assert!(!hardware_payload_matches(
            &cpu_only,
            &specs("Test CPU", Some("Test GPU"))
        ));
    }

    #[test]
    fn gpu_count_must_match_when_recorded() {
        use serde_json::json;

        let one = specs("Test CPU", Some("RTX 3090"));
        let mut two = one.clone();
        two.gpu_count = 2;
        let recorded_one = json!({
            "cpu": "Test CPU", "hardwareName": "RTX 3090", "gpuCount": 1
        });
        let recorded_two = json!({
            "cpu": "Test CPU", "hardwareName": "RTX 3090", "gpuCount": 2
        });

        assert_eq!(
            hardware_payload_match_grade(&recorded_one, &one),
            Some(HardwareMatch::Legacy)
        );
        assert_eq!(
            hardware_payload_match_grade(&recorded_two, &two),
            Some(HardwareMatch::Legacy)
        );
        assert_eq!(hardware_payload_match_grade(&recorded_one, &two), None);
        assert_eq!(hardware_payload_match_grade(&recorded_two, &one), None);

        let legacy = json!({"cpu": "Test CPU", "hardwareName": "RTX 3090"});
        assert_eq!(
            hardware_payload_match_grade(&legacy, &two),
            Some(HardwareMatch::Legacy)
        );
    }

    #[test]
    fn invalid_recorded_gpu_counts_do_not_match() {
        use serde_json::json;

        let machine = specs("Test CPU", Some("RTX 3090"));
        for count in [
            json!(null),
            json!(0),
            json!(-1),
            json!(1.5),
            json!("1"),
            json!(u64::from(u32::MAX) + 1),
        ] {
            let hw = json!({
                "cpu": "Test CPU", "hardwareName": "RTX 3090", "gpuCount": count
            });
            assert_eq!(hardware_payload_match_grade(&hw, &machine), None);
        }

        let cpu_only = specs("Test CPU", None);
        let recorded_cpu_only = json!({
            "cpu": "Test CPU", "hardwareName": null, "gpuCount": 0
        });
        assert_eq!(
            hardware_payload_match_grade(&recorded_cpu_only, &cpu_only),
            Some(HardwareMatch::Legacy)
        );
    }

    #[test]
    fn pcie_hardware_fixture_matches_width_and_legacy_policy() {
        use crate::hardware::{GpuInfo, PcieLink};
        let mut machine = specs("Test CPU", Some("Tesla V100"));
        machine.backend = GpuBackend::Cuda;
        machine.gpus = vec![GpuInfo {
            name: "Tesla V100".to_string(),
            vram_gb: Some(16.0),
            backend: GpuBackend::Cuda,
            count: 1,
            unified_memory: false,
            free_vram_gb: None,
            free_vram_per_card_gb: Vec::new(),
            pcie_links: vec![PcieLink {
                index: Some(0),
                generation: Some(1),
                width: Some(4),
                max_generation: Some(3),
                max_width: Some(16),
            }],
        }];
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/pcie-benchmark-hardware.json"
        ))
        .expect("valid PCIe fixture");
        for case in fixtures.as_array().expect("fixture array") {
            assert_eq!(
                hardware_payload_matches(&case["hardware"], &machine),
                case["matchesX4"].as_bool().expect("expected match result"),
                "{}",
                case["name"]
            );
        }
        assert_eq!(
            hardware_payload_match_grade(&fixtures[2]["hardware"], &machine),
            Some(HardwareMatch::ExactPcie)
        );
        assert_eq!(
            hardware_payload_match_grade(&fixtures[4]["hardware"], &machine),
            Some(HardwareMatch::Legacy)
        );
        assert_eq!(
            hardware_payload_match_grade(&fixtures[5]["hardware"], &machine),
            None
        );
        machine.gpus[0].pcie_links.clear();
        assert!(hardware_payload_matches(&fixtures[4]["hardware"], &machine));
        assert!(hardware_payload_matches(&fixtures[2]["hardware"], &machine));
        assert_eq!(
            hardware_payload_match_grade(&fixtures[2]["hardware"], &machine),
            Some(HardwareMatch::Legacy)
        );
        assert!(!hardware_payload_matches(
            &fixtures[5]["hardware"],
            &machine
        ));
    }

    #[test]
    fn community_index_prefers_exact_pcie_for_the_same_model() {
        let entry = |avg_tps, match_grade| CommunityResult {
            model: "llama3.1:8b".to_string(),
            provider: "ollama".to_string(),
            avg_tps,
            ttft_ms: None,
            match_grade,
            layout: crate::bench::BenchLayout::default(),
        };
        let index = CommunityBenchIndex {
            entries: vec![
                entry(100.0, HardwareMatch::Legacy),
                entry(40.0, HardwareMatch::ExactPcie),
                entry(60.0, HardwareMatch::ExactPcie),
            ],
        };
        let exact = index
            .lookup("test/llama-3.1-8b", &LayoutQuery::SINGLE_NODE)
            .expect("exact runs");
        assert_eq!(exact.tok_s, 50.0);
        assert_eq!(exact.sample_count, 2);
        assert_eq!(exact.match_grade, Some(HardwareMatch::ExactPcie));
        assert_eq!(exact.hardware_label, "matching CPU/GPU");

        let legacy = CommunityBenchIndex {
            entries: vec![entry(100.0, HardwareMatch::Legacy)],
        }
        .lookup("test/llama-3.1-8b", &LayoutQuery::SINGLE_NODE)
        .expect("legacy run");
        assert_eq!(legacy.match_grade, Some(HardwareMatch::Legacy));
        assert_eq!(legacy.hardware_label, "matching CPU/GPU");
    }

    fn layout(nodes: Option<u32>, tp: Option<u32>, pp: Option<u32>) -> crate::bench::BenchLayout {
        crate::bench::BenchLayout {
            node_count: nodes,
            tp_size: tp,
            pp_size: pp,
            interconnect: None,
        }
    }

    #[test]
    fn layout_query_separates_single_and_multi_node() {
        let single = LayoutQuery::SINGLE_NODE;
        let three = LayoutQuery {
            node_count: 3,
            ..LayoutQuery::SINGLE_NODE
        };

        // Old results (no layout) and explicit one-node results are single-node.
        assert!(single.accepts(&layout(None, None, None)));
        assert!(single.accepts(&layout(Some(1), Some(2), None)));
        assert!(!three.accepts(&layout(None, None, None)));
        assert!(!three.accepts(&layout(Some(1), Some(2), None)));

        // Multi-node results only match a lookup for the same node count.
        assert!(!single.accepts(&layout(Some(3), Some(3), None)));
        assert!(three.accepts(&layout(Some(3), Some(3), None)));
        assert!(!three.accepts(&layout(Some(2), Some(2), None)));
    }

    #[test]
    fn layout_query_keys_on_known_parallel_sizes() {
        let tp3 = LayoutQuery {
            node_count: 3,
            tp_size: Some(3),
            pp_size: None,
        };
        assert!(tp3.accepts(&layout(Some(3), Some(3), None)));
        assert!(tp3.accepts(&layout(Some(3), Some(3), Some(1))));
        // Same nodes, different split.
        assert!(!tp3.accepts(&layout(Some(3), Some(1), Some(3))));
        // An unrecorded size counts as 1.
        assert!(!tp3.accepts(&layout(Some(3), None, Some(3))));

        let pp3 = LayoutQuery {
            node_count: 3,
            tp_size: None,
            pp_size: Some(3),
        };
        assert!(pp3.accepts(&layout(Some(3), Some(1), Some(3))));
        assert!(!pp3.accepts(&layout(Some(3), Some(3), None)));
    }

    #[test]
    fn layout_query_for_specs_uses_cluster_node_count() {
        let mut machine = specs("Test CPU", Some("NVIDIA GB10"));
        assert_eq!(LayoutQuery::for_specs(&machine), LayoutQuery::SINGLE_NODE);
        machine.cluster_mode = true;
        machine.cluster_node_count = 3;
        assert_eq!(LayoutQuery::for_specs(&machine).node_count, 3);
        // A cluster config with one node is a single machine.
        machine.cluster_node_count = 1;
        assert_eq!(LayoutQuery::for_specs(&machine), LayoutQuery::SINGLE_NODE);
    }

    #[test]
    fn community_index_keeps_multi_node_runs_out_of_single_node_lookups() {
        let entry = |avg_tps, layout| CommunityResult {
            model: "llama3.1:8b".to_string(),
            provider: "vllm".to_string(),
            avg_tps,
            ttft_ms: None,
            match_grade: HardwareMatch::Legacy,
            layout,
        };
        let index = CommunityBenchIndex {
            entries: vec![
                entry(30.0, layout(Some(2), Some(2), None)),
                entry(10.0, layout(None, None, None)),
            ],
        };
        let name = "test/llama-3.1-8b";
        let single = index
            .lookup(name, &LayoutQuery::SINGLE_NODE)
            .expect("single-node run");
        assert_eq!((single.tok_s, single.sample_count), (10.0, 1));
        let two = LayoutQuery {
            node_count: 2,
            tp_size: Some(2),
            pp_size: None,
        };
        let multi = index.lookup(name, &two).expect("two-node run");
        assert_eq!((multi.tok_s, multi.sample_count), (30.0, 1));
        // No three-node run recorded.
        let three = LayoutQuery {
            node_count: 3,
            ..two
        };
        assert!(index.lookup(name, &three).is_none());
    }

    #[test]
    fn community_results_skip_other_node_counts_and_bad_layouts() {
        use serde_json::json;
        let machine = specs("Test CPU", Some("NVIDIA GB10"));
        let hardware = json!({
            "hwClass": "UNIFIED", "hardwareName": "NVIDIA GB10",
            "gpuCount": 1, "cpu": "Test CPU"
        });
        let submission = json!({
            "submittedAtUnix": 1,
            "hardware": hardware,
            "results": [
                {"model": "a", "provider": "vllm", "avgTps": 10.0},
                {"model": "b", "provider": "vllm", "avgTps": 20.0,
                 "nodeCount": 2, "tpSize": 2, "interconnect": "200GbE RoCE"},
                {"model": "c", "provider": "vllm", "avgTps": 30.0, "nodeCount": 0},
                // Well-typed but invalid layouts are skipped too.
                {"model": "d", "provider": "vllm", "avgTps": 40.0, "nodeCount": 2},
                {"model": "e", "provider": "vllm", "avgTps": 50.0,
                 "nodeCount": 2, "tpSize": 2, "interconnect": " "},
                {"model": "f", "provider": "vllm", "avgTps": 60.0,
                 "nodeCount": 2, "tpSize": 2, "interconnect": "x".repeat(65)},
                {"model": "g", "provider": "vllm", "avgTps": 70.0, "interconnect": ""}
            ]
        });
        let models = |specs: &SystemSpecs| -> Vec<String> {
            community_results_in(std::slice::from_ref(&submission), specs)
                .into_iter()
                .map(|r| r.model)
                .collect()
        };
        assert_eq!(models(&machine), vec!["a"]);
        // Same hardware identity, read as a two-node cluster: only the
        // two-node result qualifies.
        let mut cluster = machine.clone();
        cluster.cluster_mode = true;
        cluster.cluster_node_count = 2;
        let rows = community_results_in(std::slice::from_ref(&submission), &cluster);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, "b");
        assert_eq!(rows[0].layout.tp_size, Some(2));
        assert_eq!(rows[0].layout.interconnect.as_deref(), Some("200GbE RoCE"));
    }

    #[test]
    fn measured_preset_index_never_serves_a_cluster() {
        // The single-GPU preset matches by substring, so a cluster's
        // `RTX 4090 (×2)` label would hit it without the node-count guard.
        let single = specs("Test CPU", Some("NVIDIA GeForce RTX 4090"));
        assert!(MeasuredTpsIndex::for_specs(&single).is_some());
        let mut cluster = specs("Cluster (×2)", Some("NVIDIA GeForce RTX 4090 (×2)"));
        assert!(cached_preset_for_specs(&cluster).is_some());
        cluster.cluster_mode = true;
        cluster.cluster_node_count = 2;
        assert!(MeasuredTpsIndex::for_specs(&cluster).is_none());
    }

    #[test]
    fn pcie_multigpu_widths_match_without_card_order() {
        use crate::hardware::{GpuInfo, PcieLink};
        use serde_json::json;
        let mut machine = specs("Test CPU", Some("Tesla V100"));
        machine.gpu_count = 2;
        let link = |index, width| PcieLink {
            index: Some(index),
            generation: Some(1),
            width: Some(width),
            max_generation: Some(3),
            max_width: Some(16),
        };
        machine.gpus = vec![GpuInfo {
            name: "Tesla V100".to_string(),
            vram_gb: Some(32.0),
            backend: GpuBackend::Cuda,
            count: 2,
            unified_memory: false,
            free_vram_gb: None,
            free_vram_per_card_gb: Vec::new(),
            pcie_links: vec![link(0, 4), link(1, 16)],
        }];
        let hw = json!({
            "cpu": "Test CPU", "hardwareName": "Tesla V100", "gpuCount": 2,
            "pcieLinks": [{"width": 16}, {"width": 4}]
        });
        assert_eq!(
            hardware_payload_match_grade(&hw, &machine),
            Some(HardwareMatch::ExactPcie)
        );
        let mut wrong_count = hw.clone();
        wrong_count["gpuCount"] = json!(1);
        assert_eq!(hardware_payload_match_grade(&wrong_count, &machine), None);
        let narrower = json!({
            "cpu": "Test CPU", "hardwareName": "Tesla V100",
            "pcieLinks": [{"width": 8}, {"width": 4}]
        });
        assert!(!hardware_payload_matches(&narrower, &machine));
        let legacy = json!({"cpu": "Test CPU", "hardwareName": "Tesla V100"});
        assert!(hardware_payload_matches(&legacy, &machine));
        machine.gpus[0].pcie_links.pop();
        assert!(hardware_payload_matches(&hw, &machine));
        assert!(!hardware_payload_matches(
            &narrower,
            &specs("Other CPU", Some("Tesla V100"))
        ));
    }

    #[test]
    fn mixed_gpu_models_use_cpu_gpu_fallback_when_widths_are_unknown() {
        use crate::hardware::{GpuInfo, PcieLink};
        use serde_json::json;
        let mut machine = specs("Test CPU", Some("RTX 4090"));
        machine.gpu_count = 2;
        let gpu = |name: &str, index| GpuInfo {
            name: name.to_string(),
            vram_gb: Some(24.0),
            backend: GpuBackend::Cuda,
            count: 1,
            unified_memory: false,
            free_vram_gb: None,
            free_vram_per_card_gb: Vec::new(),
            pcie_links: vec![PcieLink {
                index: Some(index),
                generation: Some(1),
                width: Some(16),
                max_generation: Some(4),
                max_width: Some(16),
            }],
        };
        machine.gpus = vec![gpu("RTX 4090", 0), gpu("RTX 3060", 1)];
        assert!(benchmark_pcie_links(&machine).is_empty());
        let legacy = json!({"cpu": "Test CPU", "hardwareName": "RTX 4090"});
        assert!(hardware_payload_matches(&legacy, &machine));
        let with_links = json!({
            "cpu": "Test CPU", "hardwareName": "RTX 4090",
            "pcieLinks": [{"width": 16}, {"width": 16}]
        });
        assert!(hardware_payload_matches(&with_links, &machine));
    }

    #[test]
    fn test_measured_index_median_and_comparability_filters() {
        let rows = vec![
            row(
                r#"{"id":"1","tokSOut":100.0,"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"llama.cpp","quantization":"Q4_K_M"}}"#,
            ),
            row(
                r#"{"id":"2","tokSOut":120.0,"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"llama.cpp","quantization":"Q4_K_M"}}"#,
            ),
            // Batched serving measures a different quantity — excluded.
            row(
                r#"{"id":"3","tokSOut":900.0,"batchSize":8,"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"vllm","quantization":"Q4_K_M"}}"#,
            ),
            // Draft-accelerated runs beat the roofline — excluded.
            row(
                r#"{"id":"4","tokSOut":500.0,"engineFlags":{"specDecoding":true},"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"llama.cpp","quantization":"Q4_K_M"}}"#,
            ),
        ];
        let idx = MeasuredTpsIndex::from_rows(&rows, "RTX 3090 (24 GB)")
            .expect("comparable rows should build an index");

        let m = idx
            .lookup("unsloth/Qwen3-4B-GGUF", "q4_k_m")
            .expect("quant match is case-insensitive");
        assert_eq!(
            m.sample_count, 2,
            "batched/spec-decoding rows must not count"
        );
        assert!((m.tok_s - 110.0).abs() < 1e-9, "median of 100 and 120");
        assert_eq!(m.hardware_label, "RTX 3090 (24 GB)");

        assert!(
            idx.lookup("unsloth/Qwen3-4B-GGUF", "Q8_0").is_none(),
            "different quant must not match"
        );
    }

    #[test]
    fn test_measured_index_none_when_no_comparable_rows() {
        let rows = vec![row(
            r#"{"id":"1","tokSOut":900.0,"batchSize":16,"model":{"hfId":"a/b"},"engine":{"engineName":"vllm","quantization":"FP8"}}"#,
        )];
        assert!(MeasuredTpsIndex::from_rows(&rows, "A100 (80 GB)").is_none());
    }
}
