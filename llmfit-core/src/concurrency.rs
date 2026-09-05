//! Concurrent-session capacity estimation (issue #140).
//!
//! Answers "how many simultaneous inference sessions of this model fit in the
//! memory pool at a given context length" — the server-planning counterpart to
//! [`crate::fit::ModelFit::usable_context`], which solves the same memory
//! balance for tokens instead of sessions. Model weights and the fixed runtime
//! overhead are resident once; each concurrent session adds one KV cache at the
//! chosen context:
//!
//! ```text
//! sessions(ctx) = floor( (pool - weights_resident) / kv_cache(ctx) )
//! ```
//!
//! `kv_cache(ctx)` is GQA- and layout-aware via
//! [`crate::models::LlmModel::kv_cache_gb`]; `weights_resident` is
//! [`crate::models::LlmModel::estimate_memory_gb`] at context 0 (weights plus
//! the one-time runtime overhead, no KV). This is a memory-capacity ceiling,
//! not a throughput figure: it says how many sessions can be *resident*, not how
//! fast they run under concurrent load.

use serde::Serialize;

use crate::models::{KvQuant, LlmModel};

/// Default context ladder reported when no explicit context is given: 4k to
/// 256k, doubling. Makes the memory-versus-context trade-off visible at a
/// glance (a model that hosts many sessions at 8k but only one at 128k is the
/// common real case).
pub const DEFAULT_CONTEXT_LADDER: [u32; 7] =
    [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144];

/// Concurrency at one context length.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConcurrencySlot {
    /// Context requested for this rung.
    pub requested_context: u32,
    /// Context actually used, clamped to the model's native window.
    pub effective_context: u32,
    /// True when `requested_context` exceeded the native window and was clamped.
    pub clamped: bool,
    /// KV cache size for a single session at `effective_context`, in GB.
    pub per_session_kv_gb: f64,
    /// Maximum concurrent sessions that fit alongside the resident weights.
    pub max_sessions: u32,
}

/// Full concurrency estimate for one model against one memory pool.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConcurrencyEstimate {
    /// Memory pool the sessions share, in GB.
    pub pool_gb: f64,
    /// Weights plus one-time runtime overhead, resident once, in GB.
    pub weights_resident_gb: f64,
    /// Pool left for KV caches after the resident weights, in GB.
    pub kv_budget_gb: f64,
    /// Coarse per-session recurrent-state estimate (GB) for hybrid SSM models,
    /// added to each session on top of its KV cache. Zero for pure attention
    /// models. See [`crate::models::LlmModel::recurrent_state_estimate_gb`].
    pub per_session_recurrent_gb: f64,
    /// Weight quantization the estimate assumes.
    pub quant: String,
    /// KV cache element representation the estimate assumes.
    pub kv_quant: KvQuant,
    /// Model's native context window.
    pub native_context: u32,
    /// One entry per requested context length.
    pub ladder: Vec<ConcurrencySlot>,
}

impl ConcurrencyEstimate {
    /// Largest ladder rung (by effective context) that still fits at least
    /// `users` concurrent sessions, if any.
    pub fn max_context_for(&self, users: u32) -> Option<u32> {
        self.ladder
            .iter()
            .filter(|s| s.max_sessions >= users)
            .map(|s| s.effective_context)
            .max()
    }
}

/// Estimate concurrent-session capacity for `model` sharing a `pool_gb` memory
/// pool, at each context length in `contexts`.
///
/// `pool_gb` is the pool the model actually runs from (for a GPU-resident model
/// this is [`crate::fit::ModelFit::memory_available_gb`]); `quant` is the weight
/// quantization; `kv` the KV cache representation. Contexts above the model's
/// native window are clamped to it, mirroring `usable_context`.
pub fn estimate_concurrency(
    model: &LlmModel,
    pool_gb: f64,
    quant: &str,
    kv: KvQuant,
    contexts: &[u32],
) -> ConcurrencyEstimate {
    // Weights and fixed runtime overhead are resident once (KV at context 0 is
    // zero), so estimate_memory_gb(quant, 0) is the per-process floor. Each
    // session then adds one kv_cache_gb(ctx) on top.
    let weights_resident_gb = model.estimate_memory_gb(quant, 0);
    let kv_budget_gb = (pool_gb - weights_resident_gb).max(0.0);
    let native_context = model.context_length;
    // Hybrid SSM / linear-attention models keep a fixed-size recurrent state per
    // sequence, so each concurrent session costs its KV cache plus this
    // (context-independent) estimate. Zero for pure attention models.
    let per_session_recurrent_gb = model.recurrent_state_estimate_gb();

    let ladder = contexts
        .iter()
        .map(|&requested| {
            let effective_context = requested.min(native_context);
            let per_session_kv_gb = model.kv_cache_gb(effective_context, kv);
            let per_session_total_gb = per_session_kv_gb + per_session_recurrent_gb;
            let max_sessions = if per_session_total_gb > f64::EPSILON {
                (kv_budget_gb / per_session_total_gb).floor() as u32
            } else {
                0
            };
            ConcurrencySlot {
                requested_context: requested,
                effective_context,
                clamped: requested > native_context,
                per_session_kv_gb,
                max_sessions,
            }
        })
        .collect();

    ConcurrencyEstimate {
        pool_gb,
        weights_resident_gb,
        kv_budget_gb,
        per_session_recurrent_gb,
        quant: quant.to_string(),
        kv_quant: kv,
        native_context,
        ladder,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{self, LlmModel};

    /// Model with a precise-path KV cache calibrated so that fp16 KV at an 8192
    /// context is exactly 1 GiB: 2 * 8 kv_heads * 128 head_dim * 2 bytes * 32
    /// layers = 131072 bytes/token, * 8192 = 1_073_741_824 bytes = 1 GiB.
    fn calibrated_model(context_length: u32) -> LlmModel {
        LlmModel {
            name: "Calibrated".to_string(),
            provider: "Test".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: Some(4.0),
            quantization: "Q4_K_M".to_string(),
            context_length,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            languages: vec![],
            format: models::ModelFormat::default(),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            num_hidden_layers: Some(32),
            head_dim: Some(128),
            attention_layout: None,
            license: None,
            hidden_size: Some(4096),
            moe_intermediate_size: None,
            vocab_size: Some(32000),
            shared_expert_intermediate_size: None,
            architecture: None,
        }
    }

    /// Hybrid SSM model: 64 layers, 16 full attention + 48 linear, hidden 5120,
    /// matching the measured Qwen3.5 hybrid used to fit the recurrent estimate.
    fn hybrid_ssm_model() -> LlmModel {
        let mut m = calibrated_model(262_144);
        m.num_hidden_layers = Some(64);
        m.hidden_size = Some(5120);
        m.attention_layout = Some(models::AttentionLayout {
            full: 16,
            linear: 48,
        });
        m
    }

    #[test]
    fn recurrent_estimate_calibrated_to_measured_hybrid() {
        let mib = hybrid_ssm_model().recurrent_state_estimate_gb() * 1024.0;
        assert!(
            (mib - 150.0).abs() < 10.0,
            "recurrent state should be ~150 MiB, got {mib} MiB"
        );
        // Pure attention model (no linear layers) pays no recurrent state.
        assert_eq!(calibrated_model(262_144).recurrent_state_estimate_gb(), 0.0);
    }

    #[test]
    fn recurrent_state_reduces_hybrid_sessions() {
        let hyb = hybrid_ssm_model();
        let mut hyb_no_rec = hyb.clone();
        hyb_no_rec.hidden_size = None; // forces recurrent estimate to 0, KV unchanged
        let est = estimate_concurrency(&hyb, 100.0, "Q4_K_M", KvQuant::Fp16, &[8_192]);
        let est0 = estimate_concurrency(&hyb_no_rec, 100.0, "Q4_K_M", KvQuant::Fp16, &[8_192]);
        assert!(est.per_session_recurrent_gb > 0.0);
        assert_eq!(est0.per_session_recurrent_gb, 0.0);
        assert!(
            est.ladder[0].max_sessions < est0.ladder[0].max_sessions,
            "recurrent state must reduce concurrency: {} vs {}",
            est.ladder[0].max_sessions,
            est0.ladder[0].max_sessions
        );
    }

    #[test]
    fn kv_cache_is_calibrated_to_one_gib_at_8k() {
        let m = calibrated_model(262_144);
        assert!((m.kv_cache_gb(8_192, KvQuant::Fp16) - 1.0).abs() < 1e-9);
        assert!((m.kv_cache_gb(16_384, KvQuant::Fp16) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn budget_and_weights_are_consistent() {
        let m = calibrated_model(262_144);
        let pool = 100.0;
        let est = estimate_concurrency(&m, pool, "Q4_K_M", KvQuant::Fp16, &DEFAULT_CONTEXT_LADDER);
        assert_eq!(est.weights_resident_gb, m.estimate_memory_gb("Q4_K_M", 0));
        assert!((est.kv_budget_gb - (pool - est.weights_resident_gb)).abs() < 1e-9);
        assert_eq!(est.ladder.len(), DEFAULT_CONTEXT_LADDER.len());
    }

    #[test]
    fn sessions_match_budget_over_per_session_kv() {
        let m = calibrated_model(262_144);
        let est = estimate_concurrency(&m, 100.0, "Q4_K_M", KvQuant::Fp16, &DEFAULT_CONTEXT_LADDER);
        let slot_8k = est
            .ladder
            .iter()
            .find(|s| s.effective_context == 8_192)
            .unwrap();
        assert!((slot_8k.per_session_kv_gb - 1.0).abs() < 1e-9);
        assert_eq!(slot_8k.max_sessions, est.kv_budget_gb.floor() as u32);
    }

    #[test]
    fn more_context_never_fits_more_sessions() {
        let m = calibrated_model(262_144);
        let est = estimate_concurrency(&m, 100.0, "Q4_K_M", KvQuant::Fp16, &DEFAULT_CONTEXT_LADDER);
        for w in est.ladder.windows(2) {
            assert!(
                w[1].max_sessions <= w[0].max_sessions,
                "sessions must not grow as context grows: {} then {}",
                w[0].max_sessions,
                w[1].max_sessions
            );
        }
    }

    #[test]
    fn contexts_above_native_window_are_clamped() {
        let m = calibrated_model(16_384);
        let est = estimate_concurrency(&m, 100.0, "Q4_K_M", KvQuant::Fp16, &DEFAULT_CONTEXT_LADDER);
        let over = est
            .ladder
            .iter()
            .find(|s| s.requested_context == 262_144)
            .unwrap();
        assert!(over.clamped);
        assert_eq!(over.effective_context, 16_384);
        let under = est
            .ladder
            .iter()
            .find(|s| s.requested_context == 8_192)
            .unwrap();
        assert!(!under.clamped);
        let native_rung = est
            .ladder
            .iter()
            .find(|s| s.requested_context == 16_384)
            .unwrap();
        assert_eq!(over.max_sessions, native_rung.max_sessions);
    }

    #[test]
    fn tiny_pool_yields_zero_sessions() {
        let m = calibrated_model(262_144);
        let est = estimate_concurrency(&m, 0.1, "Q4_K_M", KvQuant::Fp16, &DEFAULT_CONTEXT_LADDER);
        assert_eq!(est.kv_budget_gb, 0.0);
        assert!(est.ladder.iter().all(|s| s.max_sessions == 0));
    }

    #[test]
    fn max_context_for_users_picks_largest_fitting_rung() {
        let m = calibrated_model(262_144);
        let est = estimate_concurrency(&m, 100.0, "Q4_K_M", KvQuant::Fp16, &DEFAULT_CONTEXT_LADDER);
        let c1 = est.max_context_for(1).unwrap();
        if let Some(c_many) = est.max_context_for(8) {
            assert!(c_many <= c1);
        }
    }

    #[test]
    fn coarse_fallback_without_arch_metadata_still_estimates() {
        let mut m = calibrated_model(262_144);
        m.num_hidden_layers = None;
        m.head_dim = None;
        m.num_key_value_heads = None;
        m.num_attention_heads = None;
        let est = estimate_concurrency(&m, 100.0, "Q4_K_M", KvQuant::Fp16, &DEFAULT_CONTEXT_LADDER);
        assert!(est.ladder.iter().any(|s| s.max_sessions > 0));
        for w in est.ladder.windows(2) {
            assert!(w[1].max_sessions <= w[0].max_sessions);
        }
    }
}
