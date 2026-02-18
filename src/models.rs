use serde::{Deserialize, Serialize};

/// Quantization levels ordered from best quality to most compressed.
/// Used for dynamic quantization selection: try the best that fits.
pub const QUANT_HIERARCHY: &[&str] = &["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"];

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
        _ => 1.0,
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
        _ => -5.0,
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
}

impl LlmModel {
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
        // Try best quality first
        for &q in QUANT_HIERARCHY {
            let mem = self.estimate_memory_gb(q, ctx);
            if mem <= budget_gb {
                return Some((q, mem));
            }
        }
        // Try halving context once
        let half_ctx = ctx / 2;
        if half_ctx >= 1024 {
            for &q in QUANT_HIERARCHY {
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
#[derive(Deserialize)]
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
}

const HF_MODELS_JSON: &str = include_str!("../data/hf_models.json");

pub struct ModelDatabase {
    models: Vec<LlmModel>,
}

impl ModelDatabase {
    pub fn new() -> Self {
        let entries: Vec<HfModelEntry> =
            serde_json::from_str(HF_MODELS_JSON).expect("Failed to parse embedded hf_models.json");

        let models = entries
            .into_iter()
            .map(|e| LlmModel {
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
}
