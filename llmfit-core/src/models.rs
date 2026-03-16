use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Quantization levels ordered from best quality to most compressed.
/// Used for dynamic quantization selection: try the best that fits.
pub const QUANT_HIERARCHY: &[&str] = &["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"];

/// MLX-native quantization hierarchy (best quality to most compressed).
pub const MLX_QUANT_HIERARCHY: &[&str] = &["mlx-8bit", "mlx-4bit"];

/// Bytes per parameter for each quantization level.
pub fn quant_bpp(quant: &str) -> f64 {
    match quant {
        "F32" => 4.0,
        "F16" | "BF16" => 2.0,
        "Q8_0" => 1.05,
        "Q6_K" => 0.80,
        "Q5_K_M" => 0.68,
        "Q4_K_M" | "Q4_0" => 0.58,
        "Q3_K_M" => 0.48,
        "Q2_K" => 0.37,
        "mlx-4bit" => 0.55,
        "mlx-8bit" => 1.0,
        "AWQ-4bit" => 0.5,
        "AWQ-8bit" => 1.0,
        "GPTQ-Int4" => 0.5,
        "GPTQ-Int8" => 1.0,
        _ => 0.58,
    }
}

/// Speed multiplier for quantization (lower quant = faster inference).
pub fn quant_speed_multiplier(quant: &str) -> f64 {
    match quant {
        "F16" | "BF16" => 0.6,
        "Q8_0" => 0.8,
        "Q6_K" => 0.95,
        "Q5_K_M" => 1.0,
        "Q4_K_M" | "Q4_0" => 1.15,
        "Q3_K_M" => 1.25,
        "Q2_K" => 1.35,
        "mlx-4bit" => 1.15,
        "mlx-8bit" => 0.85,
        "AWQ-4bit" | "GPTQ-Int4" => 1.2,
        "AWQ-8bit" | "GPTQ-Int8" => 0.85,
        _ => 1.0,
    }
}

/// Bytes per parameter for a given quantization format.
/// Used by the bandwidth-based tok/s estimator to compute model size in GB.
pub fn quant_bytes_per_param(quant: &str) -> f64 {
    match quant {
        "F16" | "BF16" => 2.0,
        "Q8_0" => 1.0,
        "Q6_K" => 0.75,
        "Q5_K_M" => 0.625,
        "Q4_K_M" | "Q4_0" => 0.5,
        "Q3_K_M" => 0.375,
        "Q2_K" => 0.25,
        "mlx-4bit" => 0.5,
        "mlx-8bit" => 1.0,
        "AWQ-4bit" | "GPTQ-Int4" => 0.5,
        "AWQ-8bit" | "GPTQ-Int8" => 1.0,
        _ => 0.5, // default to ~4-bit
    }
}

/// Quality penalty for quantization (lower quant = lower quality).
pub fn quant_quality_penalty(quant: &str) -> f64 {
    match quant {
        "F16" | "BF16" => 0.0,
        "Q8_0" => 0.0,
        "Q6_K" => -1.0,
        "Q5_K_M" => -2.0,
        "Q4_K_M" | "Q4_0" => -5.0,
        "Q3_K_M" => -8.0,
        "Q2_K" => -12.0,
        "mlx-4bit" => -4.0,
        "mlx-8bit" => 0.0,
        "AWQ-4bit" => -3.0,
        "AWQ-8bit" => 0.0,
        "GPTQ-Int4" => -3.0,
        "GPTQ-Int8" => 0.0,
        _ => -5.0,
    }
}

/// Model capability flags (orthogonal to UseCase).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Vision,
    ToolUse,
}

impl Capability {
    pub fn label(&self) -> &'static str {
        match self {
            Capability::Vision => "Vision",
            Capability::ToolUse => "Tool Use",
        }
    }

    pub fn all() -> &'static [Capability] {
        &[Capability::Vision, Capability::ToolUse]
    }

    /// Infer capabilities from model metadata when not explicitly set in JSON.
    pub fn infer(model: &LlmModel) -> Vec<Capability> {
        let mut caps = model.capabilities.clone();
        let name = model.name.to_lowercase();
        let use_case = model.use_case.to_lowercase();

        // Vision detection
        if !caps.contains(&Capability::Vision)
            && (name.contains("vision")
                || name.contains("-vl-")
                || name.ends_with("-vl")
                || name.contains("llava")
                || name.contains("onevision")
                || name.contains("pixtral")
                || use_case.contains("vision")
                || use_case.contains("multimodal"))
        {
            caps.push(Capability::Vision);
        }

        // Tool use detection (known model families)
        if !caps.contains(&Capability::ToolUse)
            && (use_case.contains("tool")
                || use_case.contains("function call")
                || name.contains("qwen3")
                || name.contains("qwen2.5")
                || name.contains("command-r")
                || (name.contains("llama-3") && name.contains("instruct"))
                || (name.contains("mistral") && name.contains("instruct"))
                || name.contains("hermes"))
        {
            caps.push(Capability::ToolUse);
        }

        caps
    }
}

/// Model weight format — determines which inference runtime to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    Awq,
    Gptq,
    Mlx,
    Safetensors,
}

impl Default for ModelFormat {
    fn default() -> Self {
        ModelFormat::Gguf
    }
}

impl ModelFormat {
    /// Returns true for formats that are pre-quantized at a fixed bit width
    /// and cannot be dynamically re-quantized (AWQ, GPTQ).
    pub fn is_prequantized(&self) -> bool {
        matches!(self, ModelFormat::Awq | ModelFormat::Gptq)
    }
}

/// Use-case category for scoring weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum UseCase {
    General,
    Coding,
    Reasoning,
    Chat,
    Multimodal,
    Embedding,
}

impl UseCase {
    pub fn label(&self) -> &'static str {
        match self {
            UseCase::General => "General",
            UseCase::Coding => "Coding",
            UseCase::Reasoning => "Reasoning",
            UseCase::Chat => "Chat",
            UseCase::Multimodal => "Multimodal",
            UseCase::Embedding => "Embedding",
        }
    }

    /// Infer use-case from the model's use_case field and name.
    pub fn from_model(model: &LlmModel) -> Self {
        let name = model.name.to_lowercase();
        let use_case = model.use_case.to_lowercase();

        if use_case.contains("embedding") || name.contains("embed") || name.contains("bge") {
            UseCase::Embedding
        } else if name.contains("code") || use_case.contains("code") {
            UseCase::Coding
        } else if use_case.contains("vision") || use_case.contains("multimodal") {
            UseCase::Multimodal
        } else if use_case.contains("reason")
            || use_case.contains("chain-of-thought")
            || name.contains("deepseek-r1")
        {
            UseCase::Reasoning
        } else if use_case.contains("chat") || use_case.contains("instruction") {
            UseCase::Chat
        } else {
            UseCase::General
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmModel {
    pub name: String,
    pub provider: String,
    pub parameter_count: String,
    #[serde(default)]
    pub parameters_raw: Option<u64>,
    pub min_ram_gb: f64,
    pub recommended_ram_gb: f64,
    pub min_vram_gb: Option<f64>,
    pub quantization: String,
    pub context_length: u32,
    pub use_case: String,
    #[serde(default)]
    pub is_moe: bool,
    #[serde(default)]
    pub num_experts: Option<u32>,
    #[serde(default)]
    pub active_experts: Option<u32>,
    #[serde(default)]
    pub active_parameters: Option<u64>,
    #[serde(default)]
    pub release_date: Option<String>,
    /// Known GGUF download sources (e.g. unsloth, bartowski repos on HuggingFace)
    #[serde(default)]
    pub gguf_sources: Vec<GgufSource>,
    /// Model capabilities (vision, tool use, etc.)
    #[serde(default)]
    pub capabilities: Vec<Capability>,
    /// Model weight format (gguf, awq, gptq, mlx, safetensors)
    #[serde(default)]
    pub format: ModelFormat,
}

/// A known GGUF download source for a model on HuggingFace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufSource {
    /// HuggingFace repo ID (e.g. "unsloth/Llama-3.1-8B-Instruct-GGUF")
    pub repo: String,
    /// Provider who published the GGUF (e.g. "unsloth", "bartowski")
    pub provider: String,
}

impl LlmModel {
    /// MLX models are Apple-only — they won't run on NVIDIA/AMD/Intel hardware.
    /// We detect them by the `-MLX-` suffix that's standard on HuggingFace
    /// (e.g. `Qwen3-8B-MLX-4bit`, `LFM2-1.2B-MLX-8bit`).
    pub fn is_mlx_model(&self) -> bool {
        let name_lower = self.name.to_lowercase();
        name_lower.contains("-mlx-") || name_lower.ends_with("-mlx")
    }

    /// Returns true if this model uses a pre-quantized format (AWQ/GPTQ)
    /// that cannot be dynamically re-quantized.
    pub fn is_prequantized(&self) -> bool {
        self.format.is_prequantized()
    }

    /// Bytes-per-parameter for the model's quantization level.
    fn quant_bpp(&self) -> f64 {
        quant_bpp(&self.quantization)
    }

    /// Parameter count in billions, extracted from parameters_raw or parameter_count.
    pub fn params_b(&self) -> f64 {
        if let Some(raw) = self.parameters_raw {
            raw as f64 / 1_000_000_000.0
        } else {
            // Parse from string like "7B", "1.1B", "137M"
            let s = self.parameter_count.trim().to_uppercase();
            if let Some(num_str) = s.strip_suffix('B') {
                num_str.parse::<f64>().unwrap_or(7.0)
            } else if let Some(num_str) = s.strip_suffix('M') {
                num_str.parse::<f64>().unwrap_or(0.0) / 1000.0
            } else {
                7.0
            }
        }
    }

    /// Estimate memory required (GB) at a given quantization and context length.
    /// Formula: model_weights + KV_cache + runtime_overhead
    pub fn estimate_memory_gb(&self, quant: &str, ctx: u32) -> f64 {
        let bpp = quant_bpp(quant);
        let params = self.params_b();
        let model_mem = params * bpp;
        // KV cache: ~0.000008 GB per billion params per context token
        let kv_cache = 0.000008 * params * ctx as f64;
        // Runtime overhead (CUDA/Metal context, buffers)
        let overhead = 0.5;
        model_mem + kv_cache + overhead
    }

    /// Select the best quantization level that fits within a memory budget.
    /// Returns the quant name and estimated memory in GB, or None if nothing fits.
    pub fn best_quant_for_budget(&self, budget_gb: f64, ctx: u32) -> Option<(&'static str, f64)> {
        self.best_quant_for_budget_with(budget_gb, ctx, QUANT_HIERARCHY)
    }

    /// Select the best quantization from a custom hierarchy that fits within a memory budget.
    pub fn best_quant_for_budget_with(
        &self,
        budget_gb: f64,
        ctx: u32,
        hierarchy: &[&'static str],
    ) -> Option<(&'static str, f64)> {
        // Try best quality first
        for &q in hierarchy {
            let mem = self.estimate_memory_gb(q, ctx);
            if mem <= budget_gb {
                return Some((q, mem));
            }
        }
        // Try halving context once
        let half_ctx = ctx / 2;
        if half_ctx >= 1024 {
            for &q in hierarchy {
                let mem = self.estimate_memory_gb(q, half_ctx);
                if mem <= budget_gb {
                    return Some((q, mem));
                }
            }
        }
        None
    }

    /// For MoE models, compute estimated VRAM for active experts only.
    /// Returns None for dense models.
    pub fn moe_active_vram_gb(&self) -> Option<f64> {
        if !self.is_moe {
            return None;
        }
        let active_params = self.active_parameters? as f64;
        let bpp = self.quant_bpp();
        let size_gb = (active_params * bpp) / (1024.0 * 1024.0 * 1024.0);
        Some((size_gb * 1.1).max(0.5))
    }

    /// Returns true if this model is MLX-specific (Apple Silicon only).
    /// MLX models are identified by having "-MLX" in their name.
    pub fn is_mlx_only(&self) -> bool {
        self.name.to_uppercase().contains("-MLX")
    }

    /// For MoE models, compute RAM needed for offloaded (inactive) experts.
    /// Returns None for dense models.
    pub fn moe_offloaded_ram_gb(&self) -> Option<f64> {
        if !self.is_moe {
            return None;
        }
        let active = self.active_parameters? as f64;
        let total = self.parameters_raw? as f64;
        let inactive = total - active;
        if inactive <= 0.0 {
            return Some(0.0);
        }
        let bpp = self.quant_bpp();
        Some((inactive * bpp) / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Intermediate struct matching the JSON schema from the scraper.
/// Extra fields are ignored when mapping to LlmModel.
#[derive(Debug, Clone, Deserialize)]
struct HfModelEntry {
    name: String,
    provider: String,
    parameter_count: String,
    #[serde(default)]
    parameters_raw: Option<u64>,
    min_ram_gb: f64,
    recommended_ram_gb: f64,
    min_vram_gb: Option<f64>,
    quantization: String,
    context_length: u32,
    use_case: String,
    #[serde(default)]
    is_moe: bool,
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    active_experts: Option<u32>,
    #[serde(default)]
    active_parameters: Option<u64>,
    #[serde(default)]
    release_date: Option<String>,
    #[serde(default)]
    gguf_sources: Vec<GgufSource>,
    #[serde(default)]
    capabilities: Vec<Capability>,
    #[serde(default)]
    format: ModelFormat,
    #[serde(default)]
    hf_downloads: u64,
    #[serde(default)]
    hf_likes: u64,
}

fn parse_parameter_count_hint(parameter_count: &str) -> Option<u64> {
    let normalized = parameter_count.trim().replace(',', "").to_uppercase();

    if let Some(raw) = normalized.strip_suffix('B') {
        raw.parse::<f64>()
            .ok()
            .map(|value| (value * 1_000_000_000.0).round() as u64)
    } else if let Some(raw) = normalized.strip_suffix('M') {
        raw.parse::<f64>()
            .ok()
            .map(|value| (value * 1_000_000.0).round() as u64)
    } else {
        None
    }
}

fn effective_parameters_raw(entry: &HfModelEntry) -> Option<u64> {
    entry
        .parameters_raw
        .or_else(|| parse_parameter_count_hint(&entry.parameter_count))
}

fn option_max<T: PartialOrd + Copy>(left: Option<T>, right: Option<T>) -> Option<T> {
    match (left, right) {
        (Some(left), Some(right)) => Some(if right > left { right } else { left }),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn hf_entry_rank(entry: &HfModelEntry) -> (u64, u64, usize, usize, u8, u64, u64, u32) {
    (
        entry.hf_downloads,
        entry.hf_likes,
        entry.capabilities.len(),
        entry.gguf_sources.len(),
        u8::from(entry.release_date.is_some()),
        entry.active_parameters.unwrap_or(0),
        effective_parameters_raw(entry).unwrap_or(0),
        entry.context_length,
    )
}

fn merge_exact_name_entries(
    mut primary: HfModelEntry,
    mut secondary: HfModelEntry,
) -> HfModelEntry {
    if hf_entry_rank(&secondary) > hf_entry_rank(&primary) {
        std::mem::swap(&mut primary, &mut secondary);
    }

    let primary_effective_params = effective_parameters_raw(&primary).unwrap_or(0);
    let secondary_effective_params = effective_parameters_raw(&secondary).unwrap_or(0);

    if secondary_effective_params > primary_effective_params {
        primary.parameter_count = secondary.parameter_count.clone();
    }
    primary.parameters_raw = option_max(primary.parameters_raw, secondary.parameters_raw);
    primary.min_ram_gb = primary.min_ram_gb.max(secondary.min_ram_gb);
    primary.recommended_ram_gb = primary.recommended_ram_gb.max(secondary.recommended_ram_gb);
    primary.min_vram_gb = option_max(primary.min_vram_gb, secondary.min_vram_gb);
    primary.context_length = primary.context_length.max(secondary.context_length);
    primary.is_moe |= secondary.is_moe;
    primary.num_experts = option_max(primary.num_experts, secondary.num_experts);
    primary.active_experts = option_max(primary.active_experts, secondary.active_experts);
    primary.active_parameters = option_max(primary.active_parameters, secondary.active_parameters);

    if primary.provider.is_empty() {
        primary.provider = secondary.provider.clone();
    }
    if primary.quantization.is_empty() {
        primary.quantization = secondary.quantization.clone();
    }
    if primary.use_case.is_empty() {
        primary.use_case = secondary.use_case.clone();
    }
    if primary.format == ModelFormat::default() && secondary.format != ModelFormat::default() {
        primary.format = secondary.format;
    }
    if secondary.release_date.as_deref() > primary.release_date.as_deref() {
        primary.release_date = secondary.release_date.clone();
    }

    for capability in secondary.capabilities {
        if !primary.capabilities.contains(&capability) {
            primary.capabilities.push(capability);
        }
    }

    for source in secondary.gguf_sources {
        let exists = primary
            .gguf_sources
            .iter()
            .any(|existing| existing.repo == source.repo && existing.provider == source.provider);
        if !exists {
            primary.gguf_sources.push(source);
        }
    }

    primary
}

fn dedupe_hf_entries(entries: Vec<HfModelEntry>) -> Vec<HfModelEntry> {
    let mut deduped_entries: Vec<HfModelEntry> = Vec::with_capacity(entries.len());
    let mut deduped_indices: HashMap<String, usize> = HashMap::new();

    for entry in entries {
        let key = entry.name.to_lowercase();
        if let Some(&idx) = deduped_indices.get(&key) {
            deduped_entries[idx] = merge_exact_name_entries(deduped_entries[idx].clone(), entry);
        } else {
            deduped_indices.insert(key, deduped_entries.len());
            deduped_entries.push(entry);
        }
    }

    deduped_entries
}

const HF_MODELS_JSON: &str = include_str!("../data/hf_models.json");

pub struct ModelDatabase {
    models: Vec<LlmModel>,
}

impl Default for ModelDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelDatabase {
    pub fn new() -> Self {
        let entries: Vec<HfModelEntry> =
            serde_json::from_str(HF_MODELS_JSON).expect("Failed to parse embedded hf_models.json");

        let deduped_entries = dedupe_hf_entries(entries);

        let models = deduped_entries
            .into_iter()
            .map(|e| {
                let mut model = LlmModel {
                    name: e.name,
                    provider: e.provider,
                    parameter_count: e.parameter_count,
                    parameters_raw: e.parameters_raw,
                    min_ram_gb: e.min_ram_gb,
                    recommended_ram_gb: e.recommended_ram_gb,
                    min_vram_gb: e.min_vram_gb,
                    quantization: e.quantization,
                    context_length: e.context_length,
                    use_case: e.use_case,
                    is_moe: e.is_moe,
                    num_experts: e.num_experts,
                    active_experts: e.active_experts,
                    active_parameters: e.active_parameters,
                    release_date: e.release_date,
                    gguf_sources: e.gguf_sources,
                    capabilities: e.capabilities,
                    format: e.format,
                };
                model.capabilities = Capability::infer(&model);
                model
            })
            .collect();

        ModelDatabase { models }
    }

    pub fn get_all_models(&self) -> &Vec<LlmModel> {
        &self.models
    }

    pub fn find_model(&self, query: &str) -> Vec<&LlmModel> {
        let query_lower = query.to_lowercase();
        self.models
            .iter()
            .filter(|m| {
                m.name.to_lowercase().contains(&query_lower)
                    || m.provider.to_lowercase().contains(&query_lower)
                    || m.parameter_count.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    pub fn models_fitting_system(
        &self,
        available_ram_gb: f64,
        has_gpu: bool,
        vram_gb: Option<f64>,
    ) -> Vec<&LlmModel> {
        self.models
            .iter()
            .filter(|m| {
                // Check RAM requirement
                let ram_ok = m.min_ram_gb <= available_ram_gb;

                // If model requires GPU and system has GPU, check VRAM
                if let Some(min_vram) = m.min_vram_gb {
                    if has_gpu {
                        if let Some(system_vram) = vram_gb {
                            ram_ok && min_vram <= system_vram
                        } else {
                            // GPU detected but VRAM unknown, allow but warn
                            ram_ok
                        }
                    } else {
                        // Model prefers GPU but can run on CPU with enough RAM
                        ram_ok && available_ram_gb >= m.recommended_ram_gb
                    }
                } else {
                    ram_ok
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ────────────────────────────────────────────────────────────────────
    // Quantization function tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mlx_quant_bpp_values() {
        assert_eq!(quant_bpp("mlx-4bit"), 0.55);
        assert_eq!(quant_bpp("mlx-8bit"), 1.0);
        assert_eq!(quant_speed_multiplier("mlx-4bit"), 1.15);
        assert_eq!(quant_speed_multiplier("mlx-8bit"), 0.85);
        assert_eq!(quant_quality_penalty("mlx-4bit"), -4.0);
        assert_eq!(quant_quality_penalty("mlx-8bit"), 0.0);
    }

    #[test]
    fn test_best_quant_with_mlx_hierarchy() {
        let model = LlmModel {
            name: "Test Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };

        // Large budget should return mlx-8bit (best in MLX hierarchy)
        let result = model.best_quant_for_budget_with(10.0, 4096, MLX_QUANT_HIERARCHY);
        assert!(result.is_some());
        let (quant, _) = result.unwrap();
        assert_eq!(quant, "mlx-8bit");

        // Tighter budget should fall to mlx-4bit
        let result = model.best_quant_for_budget_with(5.0, 4096, MLX_QUANT_HIERARCHY);
        assert!(result.is_some());
        let (quant, _) = result.unwrap();
        assert_eq!(quant, "mlx-4bit");
    }

    #[test]
    fn test_quant_bpp() {
        assert_eq!(quant_bpp("F32"), 4.0);
        assert_eq!(quant_bpp("F16"), 2.0);
        assert_eq!(quant_bpp("Q8_0"), 1.05);
        assert_eq!(quant_bpp("Q4_K_M"), 0.58);
        assert_eq!(quant_bpp("Q2_K"), 0.37);
        // Unknown quant defaults to Q4_K_M
        assert_eq!(quant_bpp("UNKNOWN"), 0.58);
    }

    #[test]
    fn test_quant_speed_multiplier() {
        assert_eq!(quant_speed_multiplier("F16"), 0.6);
        assert_eq!(quant_speed_multiplier("Q5_K_M"), 1.0);
        assert_eq!(quant_speed_multiplier("Q4_K_M"), 1.15);
        assert_eq!(quant_speed_multiplier("Q2_K"), 1.35);
        // Lower quant = faster inference
        assert!(quant_speed_multiplier("Q2_K") > quant_speed_multiplier("Q8_0"));
    }

    #[test]
    fn test_quant_quality_penalty() {
        assert_eq!(quant_quality_penalty("F16"), 0.0);
        assert_eq!(quant_quality_penalty("Q8_0"), 0.0);
        assert_eq!(quant_quality_penalty("Q4_K_M"), -5.0);
        assert_eq!(quant_quality_penalty("Q2_K"), -12.0);
        // Lower quant = higher quality penalty
        assert!(quant_quality_penalty("Q2_K") < quant_quality_penalty("Q8_0"));
    }

    // ────────────────────────────────────────────────────────────────────
    // LlmModel tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_params_b_from_raw() {
        let model = LlmModel {
            name: "Test Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        assert_eq!(model.params_b(), 7.0);
    }

    #[test]
    fn test_params_b_from_string() {
        let model = LlmModel {
            name: "Test Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "13B".to_string(),
            parameters_raw: None,
            min_ram_gb: 8.0,
            recommended_ram_gb: 16.0,
            min_vram_gb: Some(8.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        assert_eq!(model.params_b(), 13.0);
    }

    #[test]
    fn test_params_b_from_millions() {
        let model = LlmModel {
            name: "Test Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "500M".to_string(),
            parameters_raw: None,
            min_ram_gb: 1.0,
            recommended_ram_gb: 2.0,
            min_vram_gb: Some(1.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 2048,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        assert_eq!(model.params_b(), 0.5);
    }

    #[test]
    fn test_estimate_memory_gb() {
        let model = LlmModel {
            name: "Test Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };

        let mem = model.estimate_memory_gb("Q4_K_M", 4096);
        // 7B params * 0.58 bytes = 4.06 GB + KV cache + overhead
        assert!(mem > 4.0);
        assert!(mem < 6.0);

        // Q8_0 should require more memory
        let mem_q8 = model.estimate_memory_gb("Q8_0", 4096);
        assert!(mem_q8 > mem);
    }

    #[test]
    fn test_best_quant_for_budget() {
        let model = LlmModel {
            name: "Test Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };

        // Large budget should return best quant
        let result = model.best_quant_for_budget(10.0, 4096);
        assert!(result.is_some());
        let (quant, _) = result.unwrap();
        assert_eq!(quant, "Q8_0");

        // Medium budget should find acceptable quant
        let result = model.best_quant_for_budget(5.0, 4096);
        assert!(result.is_some());

        // Tiny budget should return None
        let result = model.best_quant_for_budget(1.0, 4096);
        assert!(result.is_none());
    }

    #[test]
    fn test_moe_active_vram_gb() {
        // Dense model should return None
        let dense_model = LlmModel {
            name: "Dense Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        assert!(dense_model.moe_active_vram_gb().is_none());

        // MoE model should calculate active VRAM
        let moe_model = LlmModel {
            name: "MoE Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "8x7B".to_string(),
            parameters_raw: Some(46_700_000_000),
            min_ram_gb: 25.0,
            recommended_ram_gb: 50.0,
            min_vram_gb: Some(25.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 32768,
            use_case: "General".to_string(),
            is_moe: true,
            num_experts: Some(8),
            active_experts: Some(2),
            active_parameters: Some(12_900_000_000),
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        let vram = moe_model.moe_active_vram_gb();
        assert!(vram.is_some());
        let vram_val = vram.unwrap();
        // Should be significantly less than full model
        assert!(vram_val > 0.0);
        assert!(vram_val < 15.0);
    }

    #[test]
    fn test_moe_offloaded_ram_gb() {
        // Dense model should return None
        let dense_model = LlmModel {
            name: "Dense Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        assert!(dense_model.moe_offloaded_ram_gb().is_none());

        // MoE model should calculate offloaded RAM
        let moe_model = LlmModel {
            name: "MoE Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "8x7B".to_string(),
            parameters_raw: Some(46_700_000_000),
            min_ram_gb: 25.0,
            recommended_ram_gb: 50.0,
            min_vram_gb: Some(25.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 32768,
            use_case: "General".to_string(),
            is_moe: true,
            num_experts: Some(8),
            active_experts: Some(2),
            active_parameters: Some(12_900_000_000),
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        let offloaded = moe_model.moe_offloaded_ram_gb();
        assert!(offloaded.is_some());
        let offloaded_val = offloaded.unwrap();
        // Should be substantial
        assert!(offloaded_val > 10.0);
    }

    // ────────────────────────────────────────────────────────────────────
    // UseCase tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_use_case_from_model_coding() {
        let model = LlmModel {
            name: "codellama-7b".to_string(),
            provider: "Meta".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "Coding".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        assert_eq!(UseCase::from_model(&model), UseCase::Coding);
    }

    #[test]
    fn test_use_case_from_model_embedding() {
        let model = LlmModel {
            name: "bge-large".to_string(),
            provider: "BAAI".to_string(),
            parameter_count: "335M".to_string(),
            parameters_raw: Some(335_000_000),
            min_ram_gb: 1.0,
            recommended_ram_gb: 2.0,
            min_vram_gb: Some(1.0),
            quantization: "F16".to_string(),
            context_length: 512,
            use_case: "Embedding".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        assert_eq!(UseCase::from_model(&model), UseCase::Embedding);
    }

    #[test]
    fn test_use_case_from_model_reasoning() {
        let model = LlmModel {
            name: "deepseek-r1-7b".to_string(),
            provider: "DeepSeek".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 8192,
            use_case: "Reasoning".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        assert_eq!(UseCase::from_model(&model), UseCase::Reasoning);
    }

    // ────────────────────────────────────────────────────────────────────
    // ModelDatabase tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_model_database_new() {
        let db = ModelDatabase::new();
        let models = db.get_all_models();
        // Should have loaded models from embedded JSON
        assert!(!models.is_empty());
    }

    #[test]
    fn test_dedupe_hf_entries_merges_duplicate_metadata() {
        let deduped = dedupe_hf_entries(vec![
            HfModelEntry {
                name: "Example/Model".to_string(),
                provider: "Example".to_string(),
                parameter_count: "18B".to_string(),
                parameters_raw: Some(18_000_000_000),
                min_ram_gb: 10.0,
                recommended_ram_gb: 18.0,
                min_vram_gb: Some(8.0),
                quantization: "Q4_K_M".to_string(),
                context_length: 32768,
                use_case: "General".to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: Some("2026-01-01".to_string()),
                gguf_sources: vec![GgufSource {
                    repo: "example/example-model-gguf".to_string(),
                    provider: "example".to_string(),
                }],
                capabilities: vec![Capability::Vision],
                format: ModelFormat::Safetensors,
                hf_downloads: 10_000,
                hf_likes: 500,
            },
            HfModelEntry {
                name: "Example/Model".to_string(),
                provider: "Example".to_string(),
                parameter_count: "20B".to_string(),
                parameters_raw: Some(20_000_000_000),
                min_ram_gb: 12.0,
                recommended_ram_gb: 24.0,
                min_vram_gb: Some(10.0),
                quantization: "Q4_K_M".to_string(),
                context_length: 65536,
                use_case: "General".to_string(),
                is_moe: true,
                num_experts: Some(64),
                active_experts: Some(8),
                active_parameters: Some(3_000_000_000),
                release_date: Some("2026-02-01".to_string()),
                gguf_sources: vec![GgufSource {
                    repo: "unsloth/example-model-gguf".to_string(),
                    provider: "unsloth".to_string(),
                }],
                capabilities: vec![Capability::ToolUse],
                format: ModelFormat::Gguf,
                hf_downloads: 100,
                hf_likes: 10,
            },
        ]);

        assert_eq!(deduped.len(), 1);
        let merged = &deduped[0];
        assert_eq!(merged.parameter_count, "20B");
        assert_eq!(merged.parameters_raw, Some(20_000_000_000));
        assert_eq!(merged.min_ram_gb, 12.0);
        assert_eq!(merged.recommended_ram_gb, 24.0);
        assert_eq!(merged.min_vram_gb, Some(10.0));
        assert_eq!(merged.context_length, 65536);
        assert!(merged.is_moe);
        assert_eq!(merged.num_experts, Some(64));
        assert_eq!(merged.active_experts, Some(8));
        assert_eq!(merged.active_parameters, Some(3_000_000_000));
        assert!(merged.capabilities.contains(&Capability::Vision));
        assert!(merged.capabilities.contains(&Capability::ToolUse));
        assert_eq!(merged.gguf_sources.len(), 2);
        assert!(
            merged
                .gguf_sources
                .iter()
                .any(|source| source.repo == "example/example-model-gguf")
        );
        assert!(
            merged
                .gguf_sources
                .iter()
                .any(|source| source.repo == "unsloth/example-model-gguf")
        );
    }

    #[test]
    fn test_model_database_deduplicates_exact_name_collisions() {
        let db = ModelDatabase::new();
        let matches: Vec<_> = db
            .get_all_models()
            .iter()
            .filter(|m| m.name == "Qwen/Qwen3-Coder-Next")
            .collect();

        assert_eq!(
            matches.len(),
            1,
            "duplicate exact model names should be collapsed"
        );

        let model = matches[0];
        assert_eq!(model.use_case, "Code generation and completion");
        assert_eq!(model.parameter_count, "80B");
        assert_eq!(model.parameters_raw, Some(80_000_000_000));
        assert_eq!(model.min_ram_gb, 44.8);
        assert_eq!(model.recommended_ram_gb, 74.6);
        assert_eq!(model.min_vram_gb, Some(41.0));
        assert!(model.is_moe);
        assert_eq!(model.num_experts, Some(64));
        assert_eq!(model.active_experts, Some(4));
        assert_eq!(model.active_parameters, Some(3_000_000_000));
        assert!(
            model
                .gguf_sources
                .iter()
                .any(|source| source.repo == "unsloth/Qwen3-Coder-Next-GGUF")
        );
    }

    #[test]
    fn test_find_model() {
        let db = ModelDatabase::new();

        // Search by name substring (case insensitive)
        let results = db.find_model("llama");
        assert!(!results.is_empty());
        assert!(
            results
                .iter()
                .any(|m| m.name.to_lowercase().contains("llama"))
        );

        // Search should be case insensitive
        let results_upper = db.find_model("LLAMA");
        assert_eq!(results.len(), results_upper.len());
    }

    #[test]
    fn test_models_fitting_system() {
        let db = ModelDatabase::new();

        // Large system should fit many models
        let fitting = db.models_fitting_system(32.0, true, Some(24.0));
        assert!(!fitting.is_empty());

        // Very small system should fit fewer or no models
        let fitting_small = db.models_fitting_system(2.0, false, None);
        assert!(fitting_small.len() < fitting.len());

        // All fitting models should meet RAM requirements
        for model in fitting_small {
            assert!(model.min_ram_gb <= 2.0);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Capability tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_capability_infer_vision() {
        let model = LlmModel {
            name: "meta-llama/Llama-3.2-11B-Vision-Instruct".to_string(),
            provider: "Meta".to_string(),
            parameter_count: "11B".to_string(),
            parameters_raw: Some(11_000_000_000),
            min_ram_gb: 6.0,
            recommended_ram_gb: 10.0,
            min_vram_gb: Some(6.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 131072,
            use_case: "Multimodal, vision and text".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        let caps = Capability::infer(&model);
        assert!(caps.contains(&Capability::Vision));
        // Also gets ToolUse because "llama-3" + "instruct"
        assert!(caps.contains(&Capability::ToolUse));
    }

    #[test]
    fn test_capability_infer_tool_use() {
        let model = LlmModel {
            name: "Qwen/Qwen3-8B".to_string(),
            provider: "Qwen".to_string(),
            parameter_count: "8B".to_string(),
            parameters_raw: Some(8_000_000_000),
            min_ram_gb: 4.5,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 32768,
            use_case: "General purpose text generation".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        let caps = Capability::infer(&model);
        assert!(caps.contains(&Capability::ToolUse));
        assert!(!caps.contains(&Capability::Vision));
    }

    #[test]
    fn test_capability_infer_none() {
        let model = LlmModel {
            name: "BAAI/bge-large-en-v1.5".to_string(),
            provider: "BAAI".to_string(),
            parameter_count: "335M".to_string(),
            parameters_raw: Some(335_000_000),
            min_ram_gb: 1.0,
            recommended_ram_gb: 2.0,
            min_vram_gb: Some(1.0),
            quantization: "F16".to_string(),
            context_length: 512,
            use_case: "Text embeddings for RAG".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
        };
        let caps = Capability::infer(&model);
        assert!(caps.is_empty());
    }

    #[test]
    fn test_capability_preserves_explicit() {
        let model = LlmModel {
            name: "some-model".to_string(),
            provider: "Test".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![Capability::Vision],
            format: ModelFormat::default(),
        };
        let caps = Capability::infer(&model);
        // Should keep the explicit Vision and not duplicate it
        assert_eq!(caps.iter().filter(|c| **c == Capability::Vision).count(), 1);
    }

    #[test]
    fn test_awq_gptq_quant_values() {
        // AWQ
        assert_eq!(quant_bpp("AWQ-4bit"), 0.5);
        assert_eq!(quant_bpp("AWQ-8bit"), 1.0);
        assert_eq!(quant_speed_multiplier("AWQ-4bit"), 1.2);
        assert_eq!(quant_speed_multiplier("AWQ-8bit"), 0.85);
        assert_eq!(quant_quality_penalty("AWQ-4bit"), -3.0);
        assert_eq!(quant_quality_penalty("AWQ-8bit"), 0.0);
        // GPTQ
        assert_eq!(quant_bpp("GPTQ-Int4"), 0.5);
        assert_eq!(quant_bpp("GPTQ-Int8"), 1.0);
        assert_eq!(quant_speed_multiplier("GPTQ-Int4"), 1.2);
        assert_eq!(quant_speed_multiplier("GPTQ-Int8"), 0.85);
        assert_eq!(quant_quality_penalty("GPTQ-Int4"), -3.0);
        assert_eq!(quant_quality_penalty("GPTQ-Int8"), 0.0);
    }

    #[test]
    fn test_model_format_prequantized() {
        assert!(ModelFormat::Awq.is_prequantized());
        assert!(ModelFormat::Gptq.is_prequantized());
        assert!(!ModelFormat::Gguf.is_prequantized());
        assert!(!ModelFormat::Mlx.is_prequantized());
        assert!(!ModelFormat::Safetensors.is_prequantized());
    }

    // ────────────────────────────────────────────────────────────────────
    // GGUF source catalog tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_gguf_source_deserialization() {
        let json = r#"{"repo": "unsloth/Llama-3.1-8B-Instruct-GGUF", "provider": "unsloth"}"#;
        let source: GgufSource = serde_json::from_str(json).unwrap();
        assert_eq!(source.repo, "unsloth/Llama-3.1-8B-Instruct-GGUF");
        assert_eq!(source.provider, "unsloth");
    }

    #[test]
    fn test_gguf_sources_default_to_empty() {
        let json = r#"{
            "name": "test/model",
            "provider": "Test",
            "parameter_count": "7B",
            "parameters_raw": 7000000000,
            "min_ram_gb": 4.0,
            "recommended_ram_gb": 8.0,
            "quantization": "Q4_K_M",
            "context_length": 4096,
            "use_case": "General"
        }"#;
        let entry: HfModelEntry = serde_json::from_str(json).unwrap();
        assert!(entry.gguf_sources.is_empty());
    }

    #[test]
    fn test_catalog_popular_models_have_gguf_sources() {
        let db = ModelDatabase::new();
        // These popular models should have gguf_sources populated in the catalog
        let expected_with_gguf = [
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ];
        for name in &expected_with_gguf {
            let model = db.get_all_models().iter().find(|m| m.name == *name);
            assert!(model.is_some(), "Model {} should exist in catalog", name);
            let model = model.unwrap();
            assert!(
                !model.gguf_sources.is_empty(),
                "Model {} should have gguf_sources but has none",
                name
            );
        }
    }

    #[test]
    fn test_catalog_gguf_sources_have_valid_repos() {
        let db = ModelDatabase::new();
        for model in db.get_all_models() {
            for source in &model.gguf_sources {
                assert!(
                    source.repo.contains('/'),
                    "GGUF source repo '{}' for model '{}' should be owner/repo format",
                    source.repo,
                    model.name
                );
                assert!(
                    !source.provider.is_empty(),
                    "GGUF source provider for model '{}' should not be empty",
                    model.name
                );
                assert!(
                    source.repo.to_uppercase().contains("GGUF"),
                    "GGUF source repo '{}' for model '{}' should contain 'GGUF'",
                    source.repo,
                    model.name
                );
            }
        }
    }

    #[test]
    fn test_catalog_has_significant_gguf_coverage() {
        let db = ModelDatabase::new();
        let total = db.get_all_models().len();
        let with_gguf = db
            .get_all_models()
            .iter()
            .filter(|m| !m.gguf_sources.is_empty())
            .count();
        // We should have at least 25% coverage after enrichment
        let coverage_pct = (with_gguf as f64 / total as f64) * 100.0;
        assert!(
            coverage_pct >= 25.0,
            "GGUF source coverage is only {:.1}% ({}/{}), expected at least 25%",
            coverage_pct,
            with_gguf,
            total
        );
    }
}
