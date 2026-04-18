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
                || name.contains("hermes")
                || (name.contains("gemma-3") && name.ends_with("-it"))
                || (name.contains("gemma-4") && name.ends_with("-it")))
        {
            caps.push(Capability::ToolUse);
        }

        caps
    }
}

/// Model weight format — determines which inference runtime to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ModelFormat {
    #[default]
    Gguf,
    Awq,
    Gptq,
    Mlx,
    Safetensors,
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
    /// Number of attention heads (for tensor-parallelism compatibility checks).
    #[serde(default)]
    pub num_attention_heads: Option<u32>,
    /// Number of key-value heads for GQA (defaults to num_attention_heads if None).
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,
    /// Total number of transformer layers. Used by the precise KV cache formula.
    #[serde(default)]
    pub num_hidden_layers: Option<u32>,
    /// Per-head dimension. Used by the precise KV cache formula. When absent,
    /// derived as `hidden_size / num_attention_heads` if both are known, or
    /// a name based heuristic otherwise.
    #[serde(default)]
    pub head_dim: Option<u32>,
    /// Attention layer composition for hybrid models (full attention + linear /
    /// Mamba style layers). When None, all layers are assumed to be full
    /// attention. Used by KV cache compression schemes (e.g. TurboQuant) that
    /// only apply to full attention layers.
    #[serde(default)]
    pub attention_layout: Option<AttentionLayout>,
    /// Model license (e.g. "apache-2.0", "mit", "llama3.1")
    #[serde(default)]
    pub license: Option<String>,
}

/// Composition of attention layers in a hybrid model.
///
/// Some recent architectures (Qwen3-Next, Jamba, Mamba style hybrids) mix
/// full attention layers with cheaper linear / state space layers. KV cache
/// compression schemes like TurboQuant only apply to the full attention
/// fraction, so we track the split here to compute honest savings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttentionLayout {
    /// Number of full self attention layers (compressible).
    pub full: u32,
    /// Number of linear / state space layers (not compressible by KV quant).
    pub linear: u32,
}

impl AttentionLayout {
    pub fn total(&self) -> u32 {
        self.full + self.linear
    }

    /// Fraction of layers that are full attention (and therefore compressible
    /// by KV quant schemes). Returns 1.0 for an all-full model.
    pub fn compressible_fraction(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            1.0
        } else {
            self.full as f64 / total as f64
        }
    }
}

/// KV cache element representation. Controls bytes per element for the
/// precise KV cache formula and (for TurboQuant) gates on runtime support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum KvQuant {
    /// fp16 / bf16, the inference default for most runtimes.
    #[default]
    #[serde(rename = "fp16")]
    Fp16,
    /// fp8 KV cache (vLLM, llama.cpp via --cache-type-k fp8 on supported builds).
    #[serde(rename = "fp8")]
    Fp8,
    /// 8 bit integer KV cache (llama.cpp `q8_0`, vLLM int8).
    #[serde(rename = "q8_0")]
    Q8_0,
    /// 4 bit integer KV cache (llama.cpp `q4_0`, vLLM int4).
    #[serde(rename = "q4_0")]
    Q4_0,
    /// TurboQuant (3 bit keys + 2 bit values + Pi/S overhead). Research
    /// integration, vLLM + CUDA only, not in upstream vLLM yet. Compression
    /// only applies to full attention layers, so hybrid models see less.
    /// See https://github.com/0xSero/turboquant
    #[serde(rename = "tq")]
    TurboQuant,
}

impl KvQuant {
    pub fn label(&self) -> &'static str {
        match self {
            KvQuant::Fp16 => "fp16",
            KvQuant::Fp8 => "fp8",
            KvQuant::Q8_0 => "q8_0",
            KvQuant::Q4_0 => "q4_0",
            KvQuant::TurboQuant => "tq",
        }
    }

    /// Bytes per KV element for non-TurboQuant variants. TurboQuant is handled
    /// per layer because it only affects the full attention slice.
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            KvQuant::Fp16 => 2.0,
            KvQuant::Fp8 => 1.0,
            KvQuant::Q8_0 => 1.0,
            KvQuant::Q4_0 => 0.5,
            // For the bookkeeping path that doesn't know about layout, assume
            // ~2.7 bits per element on the compressible slice. The real
            // computation in `precise_kv_cache_gb` handles the layout split.
            KvQuant::TurboQuant => 0.34,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "fp16" | "f16" | "bf16" | "default" => Some(KvQuant::Fp16),
            "fp8" | "f8" => Some(KvQuant::Fp8),
            "q8" | "q8_0" | "int8" => Some(KvQuant::Q8_0),
            "q4" | "q4_0" | "int4" => Some(KvQuant::Q4_0),
            "tq" | "turboquant" => Some(KvQuant::TurboQuant),
            _ => None,
        }
    }

    /// All KV quant options llmfit knows how to estimate. Order is best
    /// quality (fp16) to most compressed.
    pub fn all() -> &'static [KvQuant] {
        &[
            KvQuant::Fp16,
            KvQuant::Fp8,
            KvQuant::Q8_0,
            KvQuant::Q4_0,
            KvQuant::TurboQuant,
        ]
    }
}

impl std::fmt::Display for KvQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Returns true if a model's license matches any in the comma-separated filter string.
/// Models without a license never match.
pub fn matches_license_filter(license: &Option<String>, filter: &str) -> bool {
    let allowed: Vec<String> = filter.split(',').map(|s| s.trim().to_lowercase()).collect();
    license
        .as_ref()
        .map(|l| allowed.contains(&l.to_lowercase()))
        .unwrap_or(false)
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

    /// Returns true if the model's attention/KV heads are evenly divisible
    /// by `tp_size`, meaning it can be split across that many devices.
    /// TP=1 always returns true.
    pub fn supports_tp(&self, tp_size: u32) -> bool {
        if tp_size <= 1 {
            return true;
        }
        let (attn, kv) = self.infer_head_counts();
        attn % tp_size == 0 && kv % tp_size == 0
    }

    /// Returns all valid TP degrees in [1..=8] for this model.
    pub fn valid_tp_sizes(&self) -> Vec<u32> {
        (1..=8).filter(|&tp| self.supports_tp(tp)).collect()
    }

    /// Infer attention and KV head counts from metadata or model name heuristics.
    fn infer_head_counts(&self) -> (u32, u32) {
        if let (Some(attn), Some(kv)) = (self.num_attention_heads, self.num_key_value_heads) {
            return (attn, kv);
        }
        if let Some(attn) = self.num_attention_heads {
            return (attn, attn);
        }
        // Heuristic: infer from model name
        infer_heads_from_name(&self.name, self.params_b())
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

    /// Approximate on-disk size (GB) for a given quantization level.
    /// This is just the model weights: params_b * bytes_per_param.
    pub fn estimate_disk_gb(&self, quant: &str) -> f64 {
        self.params_b() * quant_bpp(quant)
    }

    /// Estimate memory required (GB) at a given quantization and context length.
    /// Defaults to fp16 KV cache. Use `estimate_memory_gb_with_kv` to override.
    pub fn estimate_memory_gb(&self, quant: &str, ctx: u32) -> f64 {
        self.estimate_memory_gb_with_kv(quant, ctx, KvQuant::Fp16)
    }

    /// Estimate memory required (GB) with an explicit KV cache quantization.
    /// Formula: model_weights + KV_cache + runtime_overhead
    pub fn estimate_memory_gb_with_kv(&self, quant: &str, ctx: u32, kv: KvQuant) -> f64 {
        let bpp = quant_bpp(quant);
        let params = self.params_b();
        let model_mem = params * bpp;
        let kv_cache = self.kv_cache_gb(ctx, kv);
        // Runtime overhead (CUDA/Metal context, buffers)
        let overhead = 0.5;
        model_mem + kv_cache + overhead
    }

    /// KV cache size in GB at the given context length and KV quant.
    ///
    /// Uses the precise per layer formula when `num_hidden_layers`,
    /// `num_key_value_heads`, and `head_dim` are known:
    ///
    /// `kv_bytes = 2 * n_layers * n_kv_heads * head_dim * ctx * dtype_bytes`
    ///
    /// Falls back to a coarse `params * ctx` approximation when the metadata
    /// is missing so older catalog entries don't regress.
    ///
    /// For TurboQuant, only the full attention slice (per `attention_layout`)
    /// is compressed. Linear / state space layers stay at fp16.
    pub fn kv_cache_gb(&self, ctx: u32, kv: KvQuant) -> f64 {
        let params = self.params_b();
        let layout = self.effective_attention_layout();

        // Precise path: requires layer count, KV head count, head dim.
        if let (Some(n_layers), Some(head_dim)) = (self.num_hidden_layers, self.head_dim) {
            let n_kv_heads = self
                .num_key_value_heads
                .or(self.num_attention_heads)
                .unwrap_or(8);

            let bytes_per_layer =
                |bpe: f64| -> f64 { 2.0 * n_kv_heads as f64 * head_dim as f64 * ctx as f64 * bpe };

            let total_bytes = match kv {
                KvQuant::TurboQuant => {
                    // Compressed slice (full attention) at TQ rate, rest stay fp16.
                    let full_layers = match layout {
                        Some(l) => l.full.min(n_layers),
                        None => n_layers,
                    };
                    let linear_layers = n_layers.saturating_sub(full_layers);
                    bytes_per_layer(KvQuant::TurboQuant.bytes_per_element()) * full_layers as f64
                        + bytes_per_layer(KvQuant::Fp16.bytes_per_element()) * linear_layers as f64
                }
                _ => bytes_per_layer(kv.bytes_per_element()) * n_layers as f64,
            };

            return total_bytes / 1_073_741_824.0;
        }

        // Fallback: coarse linear approximation, scaled by KV quant ratio.
        // Historical formula was 0.000008 * params_b * ctx (assumes fp16).
        let baseline_fp16 = 0.000008 * params * ctx as f64;
        let scale = match kv {
            KvQuant::Fp16 => 1.0,
            KvQuant::Fp8 | KvQuant::Q8_0 => 0.5,
            KvQuant::Q4_0 => 0.25,
            KvQuant::TurboQuant => {
                // Without layer counts we can't separate full vs linear, so
                // weight the savings by the layout if available, otherwise
                // assume an all-full dense transformer.
                let frac = layout.map(|l| l.compressible_fraction()).unwrap_or(1.0);
                let tq_ratio = KvQuant::TurboQuant.bytes_per_element() / 2.0;
                frac * tq_ratio + (1.0 - frac)
            }
        };
        baseline_fp16 * scale
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

    /// Resolved attention layout: explicit metadata if present, otherwise a
    /// best effort heuristic based on the model name. Returns `None` for
    /// plain dense transformers (which the KV estimator should treat as
    /// "all layers compressible").
    pub fn effective_attention_layout(&self) -> Option<AttentionLayout> {
        self.attention_layout
            .or_else(|| infer_attention_layout_from_name(&self.name))
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
    #[serde(default)]
    num_attention_heads: Option<u32>,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    #[serde(default)]
    num_hidden_layers: Option<u32>,
    #[serde(default)]
    head_dim: Option<u32>,
    #[serde(default)]
    license: Option<String>,
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

/// Normalize a model name/ID to a canonical slug for deduplication.
///
/// Strips the `org/` prefix, lowercases, and collapses `-`/`_`/`.` so that
/// `meta-llama/Llama-3.1-8B` and `meta-llama/llama-3.1-8b` compare equal.
pub(crate) fn canonical_slug(name: &str) -> String {
    let slug = name.split('/').next_back().unwrap_or(name);
    slug.to_lowercase().replace(['-', '_', '.'], "")
}

/// Parse the compile-time embedded JSON into a flat `Vec<LlmModel>`.
fn load_embedded() -> Vec<LlmModel> {
    let entries: Vec<HfModelEntry> =
        serde_json::from_str(HF_MODELS_JSON).expect("Failed to parse embedded hf_models.json");
    entries
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
                num_attention_heads: e.num_attention_heads,
                num_key_value_heads: e.num_key_value_heads,
                num_hidden_layers: e.num_hidden_layers,
                head_dim: e.head_dim,
                attention_layout: None,
                license: e.license,
            };
            model.capabilities = Capability::infer(&model);
            // Auto-populate attention_layout from name heuristic for known
            // hybrid families. Explicit metadata still wins (model.attention_layout
            // stays None until the scraper is taught to read it from config.json).
            if model.attention_layout.is_none() {
                model.attention_layout = infer_attention_layout_from_name(&model.name);
            }
            model
        })
        .collect()
}

impl ModelDatabase {
    /// Load only the compile-time embedded model list (no cache).
    /// Used internally by the updater to determine which models are already known.
    pub fn embedded() -> Self {
        ModelDatabase {
            models: load_embedded(),
        }
    }

    /// Load the embedded model list **and** merge any locally cached models.
    ///
    /// Cached models are appended after the embedded ones; if an ID already
    /// exists in the embedded list it is skipped to avoid duplication.
    /// Silently ignores a missing or corrupt cache file.
    pub fn new() -> Self {
        let mut models = load_embedded();

        // Merge cached models (from `llmfit update`) without duplicating.
        // canonical_slug normalizes org/ prefix, case, and separators so that
        // e.g. `meta-llama/Llama-3.1-8B` and `meta-llama/llama-3.1-8b` are
        // treated as the same model.
        let embedded_keys: std::collections::HashSet<String> =
            models.iter().map(|m| canonical_slug(&m.name)).collect();

        for cached in crate::update::load_cache() {
            if !embedded_keys.contains(&canonical_slug(&cached.name)) {
                models.push(cached);
            }
        }

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

/// Infer an attention layout from the model name for known hybrid families.
/// Returns `None` for plain dense / all-full transformers (which is the safe
/// default for the KV cache estimator: assume all layers are compressible).
///
/// The numbers here come from the published configs of each family as of
/// 2026 Q1. They're a best effort starting point and should be replaced
/// with values scraped from `config.json` whenever the metadata is available.
pub fn infer_attention_layout_from_name(name: &str) -> Option<AttentionLayout> {
    let lower = name.to_lowercase();

    // Qwen3-Next series: roughly 1 full attention layer per 4 layers,
    // remainder are linear / gated DeltaNet style. The A3B (35B total)
    // variant ships with 10 full out of 40 according to the TurboQuant
    // benchmark in 0xSero/turboquant.
    if lower.contains("qwen3-next") || lower.contains("qwen3.5-next") {
        return Some(AttentionLayout {
            full: 10,
            linear: 30,
        });
    }

    // Jamba (Mamba + Transformer hybrid). Jamba 1.5 Mini and Large both
    // use a 1:7 attention to mamba ratio in their 32 layer blocks.
    if lower.contains("jamba") {
        return Some(AttentionLayout {
            full: 4,
            linear: 28,
        });
    }

    // Zamba2 (Mamba2 + shared attention). Zamba2-7B has 2 shared attention
    // blocks and 54 mamba layers per the model card.
    if lower.contains("zamba") {
        return Some(AttentionLayout {
            full: 2,
            linear: 54,
        });
    }

    // RWKV / Mamba pure SSM models: no full attention at all. We still
    // report them so the KV estimator can short circuit. Compressible
    // fraction is 0, so KV quant savings will correctly show as zero.
    if lower.contains("mamba") || lower.contains("rwkv") {
        return Some(AttentionLayout { full: 0, linear: 1 });
    }

    None
}

/// Infer attention and KV head counts from the model name and parameter count.
/// Used as a fallback when explicit head counts are not available in the model metadata.
fn infer_heads_from_name(name: &str, params_b: f64) -> (u32, u32) {
    let name_lower = name.to_lowercase();

    // Qwen family
    if name_lower.contains("qwen") {
        if params_b > 100.0 {
            return (128, 16);
        } else if params_b > 50.0 {
            return (64, 8);
        } else if params_b > 25.0 {
            return (40, 8);
        } else if params_b > 10.0 {
            return (40, 8);
        } else if params_b > 5.0 {
            return (32, 8);
        } else {
            return (16, 4);
        }
    }

    // Llama family
    if name_lower.contains("llama") {
        if name_lower.contains("scout") || name_lower.contains("maverick") {
            return (64, 8);
        } else if params_b > 60.0 {
            return (64, 8);
        } else if params_b > 20.0 {
            return (48, 8);
        } else if params_b > 5.0 {
            return (32, 8);
        } else {
            return (16, 8);
        }
    }

    // DeepSeek family
    if name_lower.contains("deepseek") {
        if params_b > 200.0 {
            return (128, 16);
        } else if params_b > 50.0 {
            return (64, 8);
        } else if params_b > 25.0 {
            return (40, 8);
        } else if params_b > 10.0 {
            return (40, 8);
        } else {
            return (32, 8);
        }
    }

    // Mistral/Mixtral
    if name_lower.contains("mistral") || name_lower.contains("mixtral") {
        if params_b > 100.0 {
            return (96, 8);
        } else if params_b > 20.0 {
            return (32, 8);
        } else {
            return (32, 8);
        }
    }

    // Gemma
    if name_lower.contains("gemma") {
        if params_b > 20.0 {
            return (32, 16);
        } else if params_b > 5.0 {
            return (16, 8);
        } else {
            return (8, 4);
        }
    }

    // Phi
    if name_lower.contains("phi") {
        if params_b > 10.0 {
            return (40, 10);
        } else {
            return (32, 8);
        }
    }

    // MiniMax
    if name_lower.contains("minimax") {
        return (48, 8);
    }

    // Default: common pattern based on param count
    if params_b > 100.0 {
        (128, 16)
    } else if params_b > 50.0 {
        (64, 8)
    } else if params_b > 20.0 {
        (32, 8)
    } else if params_b > 5.0 {
        (32, 8)
    } else {
        (16, 4)
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
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
    #[ignore] // Requires network access to populate GGUF sources at build time
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

    // ────────────────────────────────────────────────────────────────────
    // Tensor parallelism tests
    // ────────────────────────────────────────────────────────────────────

    fn tp_test_model(
        name: &str,
        params_b: f64,
        attn_heads: Option<u32>,
        kv_heads: Option<u32>,
    ) -> LlmModel {
        LlmModel {
            name: name.to_string(),
            provider: "Test".to_string(),
            parameter_count: format!("{:.0}B", params_b),
            parameters_raw: Some((params_b * 1_000_000_000.0) as u64),
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
            num_attention_heads: attn_heads,
            num_key_value_heads: kv_heads,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
        }
    }

    #[test]
    fn test_supports_tp_with_explicit_heads() {
        let model = tp_test_model("Test-8B", 8.0, Some(32), Some(8));
        assert!(model.supports_tp(1));
        assert!(model.supports_tp(2));
        assert!(model.supports_tp(4));
        assert!(model.supports_tp(8));
        assert!(!model.supports_tp(3)); // 32 % 3 != 0
        assert!(!model.supports_tp(5));
    }

    #[test]
    fn test_supports_tp_always_true_for_1() {
        let model = tp_test_model("Tiny", 1.0, None, None);
        assert!(model.supports_tp(1));
    }

    #[test]
    fn test_valid_tp_sizes_32_8() {
        let model = tp_test_model("Test", 8.0, Some(32), Some(8));
        let sizes = model.valid_tp_sizes();
        assert!(sizes.contains(&1));
        assert!(sizes.contains(&2));
        assert!(sizes.contains(&4));
        assert!(sizes.contains(&8));
        assert!(!sizes.contains(&3));
    }

    #[test]
    fn test_valid_tp_sizes_48_heads() {
        // 48 attn heads, 8 kv heads — TP must divide both
        let model = tp_test_model("Llama-32B", 32.0, Some(48), Some(8));
        assert!(model.supports_tp(2)); // 48%2==0, 8%2==0
        assert!(!model.supports_tp(3)); // 48%3==0 but 8%3!=0
        assert!(model.supports_tp(4)); // 48%4==0, 8%4==0
        assert!(model.supports_tp(8)); // 48%8==0, 8%8==0
    }

    #[test]
    fn test_infer_heads_from_name_qwen() {
        let (attn, kv) = infer_heads_from_name("Qwen2.5-72B-Instruct", 72.0);
        assert_eq!(attn, 64);
        assert_eq!(kv, 8);
    }

    #[test]
    fn test_infer_heads_from_name_llama() {
        let (attn, kv) = infer_heads_from_name("Llama-3.1-8B", 8.0);
        assert_eq!(attn, 32);
        assert_eq!(kv, 8);
    }

    #[test]
    fn test_infer_heads_from_name_deepseek() {
        let (attn, kv) = infer_heads_from_name("DeepSeek-V3", 671.0);
        assert_eq!(attn, 128);
        assert_eq!(kv, 16);
    }

    #[test]
    fn test_supports_tp_with_inferred_heads() {
        // No explicit heads — should infer from name
        let model = tp_test_model("Llama-3.1-70B", 70.0, None, None);
        assert!(model.supports_tp(2));
        assert!(model.supports_tp(4));
        assert!(model.supports_tp(8));
    }

    // ────────────────────────────────────────────────────────────────────
    // KV cache formula + KvQuant + AttentionLayout
    // ────────────────────────────────────────────────────────────────────

    fn kv_test_model(name: &str) -> LlmModel {
        // Roughly modelled on Llama-3.1-8B: 32 layers, 32 heads, 8 KV heads,
        // head_dim 128.
        LlmModel {
            name: name.to_string(),
            provider: "Test".to_string(),
            parameter_count: "8B".to_string(),
            parameters_raw: Some(8_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 8192,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: ModelFormat::default(),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            num_hidden_layers: Some(32),
            head_dim: Some(128),
            attention_layout: None,
            license: None,
        }
    }

    #[test]
    fn test_kv_quant_from_str_round_trip() {
        for kv in KvQuant::all() {
            let parsed = KvQuant::parse(kv.label()).expect("label should parse");
            assert_eq!(parsed, *kv);
        }
        assert_eq!(KvQuant::parse("FP16"), Some(KvQuant::Fp16));
        assert_eq!(KvQuant::parse("Q4_0"), Some(KvQuant::Q4_0));
        assert_eq!(KvQuant::parse("turboquant"), Some(KvQuant::TurboQuant));
        assert_eq!(KvQuant::parse("nope"), None);
    }

    #[test]
    fn test_kv_cache_precise_formula_matches_hand_calc() {
        // 32 layers * 2 (K+V) * 8 KV heads * 128 head_dim * 8192 ctx * 2 (fp16)
        // = 1_073_741_824 bytes ≈ 1.0 GB
        let model = kv_test_model("Llama-3.1-8B");
        let kv = model.kv_cache_gb(8192, KvQuant::Fp16);
        assert!((kv - 1.0).abs() < 0.05, "expected ~1.0 GB, got {:.4}", kv);
    }

    #[test]
    fn test_kv_cache_scales_with_quant() {
        let model = kv_test_model("test");
        let fp16 = model.kv_cache_gb(8192, KvQuant::Fp16);
        let q8 = model.kv_cache_gb(8192, KvQuant::Q8_0);
        let q4 = model.kv_cache_gb(8192, KvQuant::Q4_0);
        // q8 should be ~half fp16, q4 should be ~quarter
        assert!((q8 / fp16 - 0.5).abs() < 0.01);
        assert!((q4 / fp16 - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_kv_cache_fallback_when_metadata_missing() {
        // No layer/head_dim metadata: should fall back to the linear approx
        // and still scale with KvQuant.
        let mut model = kv_test_model("nameless");
        model.num_hidden_layers = None;
        model.head_dim = None;
        let fp16 = model.kv_cache_gb(8192, KvQuant::Fp16);
        let q4 = model.kv_cache_gb(8192, KvQuant::Q4_0);
        assert!(fp16 > 0.0);
        assert!(q4 < fp16);
    }

    #[test]
    fn test_turboquant_full_attention_uses_compressed_rate() {
        // Pure dense (no layout): TQ should compress every layer.
        let model = kv_test_model("dense");
        let fp16 = model.kv_cache_gb(8192, KvQuant::Fp16);
        let tq = model.kv_cache_gb(8192, KvQuant::TurboQuant);
        let ratio = tq / fp16;
        // ~0.34 / 2.0 = 0.17 of fp16
        assert!(
            (0.10..=0.25).contains(&ratio),
            "TQ ratio on dense should be ~0.17, got {:.3}",
            ratio
        );
    }

    #[test]
    fn test_turboquant_hybrid_only_compresses_full_attention() {
        // 10 full + 30 linear layers (Qwen3.5-A3B style).
        let mut model = kv_test_model("hybrid");
        model.num_hidden_layers = Some(40);
        model.attention_layout = Some(AttentionLayout {
            full: 10,
            linear: 30,
        });
        let fp16 = model.kv_cache_gb(8192, KvQuant::Fp16);
        let tq = model.kv_cache_gb(8192, KvQuant::TurboQuant);
        let savings = 1.0 - tq / fp16;
        // Honest savings should be ~0.83 * 0.25 ≈ 21% (only the 10/40 slice
        // is compressed by ~83%). Allow a wide tolerance because the constants
        // are deliberately conservative.
        assert!(
            (0.10..=0.30).contains(&savings),
            "expected ~20% honest savings on hybrid model, got {:.3}",
            savings
        );
        // And it must be far from the dense headline of ~83%.
        assert!(savings < 0.5);
    }

    #[test]
    fn test_attention_layout_compressible_fraction() {
        let dense = AttentionLayout {
            full: 32,
            linear: 0,
        };
        assert!((dense.compressible_fraction() - 1.0).abs() < 0.0001);

        let hybrid = AttentionLayout {
            full: 10,
            linear: 30,
        };
        assert!((hybrid.compressible_fraction() - 0.25).abs() < 0.0001);

        let pure_ssm = AttentionLayout {
            full: 0,
            linear: 64,
        };
        assert!((pure_ssm.compressible_fraction() - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_infer_attention_layout_qwen3_next() {
        let layout = infer_attention_layout_from_name("Qwen/Qwen3-Next-80B-A3B");
        assert!(layout.is_some());
        let layout = layout.unwrap();
        assert!(layout.full > 0 && layout.linear > 0);
        assert!(layout.compressible_fraction() < 0.5);
    }

    #[test]
    fn test_infer_attention_layout_dense_returns_none() {
        assert!(infer_attention_layout_from_name("meta-llama/Llama-3.1-8B").is_none());
        assert!(infer_attention_layout_from_name("Qwen/Qwen2.5-7B").is_none());
    }

    #[test]
    fn test_effective_attention_layout_prefers_explicit() {
        let mut model = kv_test_model("Qwen/Qwen3-Next-80B");
        // Explicit metadata should override the heuristic
        model.attention_layout = Some(AttentionLayout {
            full: 5,
            linear: 35,
        });
        let resolved = model.effective_attention_layout().unwrap();
        assert_eq!(resolved.full, 5);
        assert_eq!(resolved.linear, 35);
    }

    #[test]
    fn test_estimate_memory_with_kv_q8_smaller_than_fp16() {
        let model = kv_test_model("Llama-3.1-8B");
        let fp16_total = model.estimate_memory_gb_with_kv("Q4_K_M", 32_768, KvQuant::Fp16);
        let q8_total = model.estimate_memory_gb_with_kv("Q4_K_M", 32_768, KvQuant::Q8_0);
        let q4_total = model.estimate_memory_gb_with_kv("Q4_K_M", 32_768, KvQuant::Q4_0);
        assert!(q8_total < fp16_total);
        assert!(q4_total < q8_total);
    }
}
