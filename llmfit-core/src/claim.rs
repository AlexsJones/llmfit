//! Kubernetes DRA claim generation.
//!
//! Turns a (model, quantization, min-tok/s) target into a ResourceClaim or
//! ResourceClaimTemplate whose CEL selector encodes the fit inequality
//! against attributes published by the llmfit.ai DRA driver (llmfit-dra).
//! The driver publishes physics inputs (memory capacity, bandwidth); this
//! module inlines the model-specific constants from the database so the
//! kube-scheduler can evaluate fit at claim time. Nothing here runs in the
//! serving path — the output is plain YAML for kubectl/GitOps.

use crate::models::{LlmModel, quant_bpp, quant_is_recognized};

/// Attribute/driver domain used by llmfit-dra.
pub const DRIVER_DOMAIN: &str = "llmfit.ai";

/// KV-cache / runtime headroom multiplier applied when the requested quant
/// differs from the database entry (whose min_vram_gb already includes it).
const WEIGHTS_HEADROOM: f64 = 1.2;

#[derive(Debug, Clone)]
pub struct ClaimTarget {
    pub min_tps: f64,
    /// Backend efficiency as a percentage (fit.rs default_efficiency = 0.55).
    pub efficiency_pct: u32,
    pub device_class: String,
    /// Emit a ResourceClaimTemplate (for pod templates) instead of a bare
    /// ResourceClaim.
    pub template: bool,
    /// Override the database entry's quantization.
    pub quant: Option<String>,
    /// Override the generated metadata.name.
    pub name: Option<String>,
}

impl Default for ClaimTarget {
    fn default() -> Self {
        Self {
            min_tps: 20.0,
            efficiency_pct: 55,
            device_class: DRIVER_DOMAIN.to_string(),
            template: false,
            quant: None,
            name: None,
        }
    }
}

/// The two constants the CEL selector needs.
#[derive(Debug, PartialEq)]
pub struct FitBounds {
    /// Device memory floor, binary gibibytes (weights + headroom).
    pub memory_gi: u64,
    /// Bandwidth floor in GB/s such that bw × efficiency / read ≥ min_tps.
    pub min_bandwidth_gbs: u64,
    /// Weights size used for the memory bound, for provenance comments.
    pub weights_gb: f64,
    /// Weights read per decoded token: all of them, or only the active
    /// experts of a MoE model.
    pub read_gb: f64,
    pub quant: String,
}

pub fn fit_bounds(model: &LlmModel, target: &ClaimTarget) -> Result<FitBounds, String> {
    if target.min_tps <= 0.0 {
        return Err("--min-tps must be > 0".to_string());
    }
    if target.efficiency_pct == 0 || target.efficiency_pct > 100 {
        return Err("--efficiency must be in 1..=100".to_string());
    }
    if let Some(q) = target.quant.as_deref()
        && !quant_is_recognized(q)
    {
        return Err(format!(
            "Unrecognized --quant '{q}'. Expected a known label such as Q4_K_M, Q6_K, Q8_0, or mlx-4bit (case-sensitive)."
        ));
    }
    let quant = target
        .quant
        .clone()
        .unwrap_or_else(|| model.quantization.clone());
    let weights_gb = model.estimate_disk_gb(&quant);
    if weights_gb <= 0.0 {
        return Err(format!(
            "cannot size model '{}' (unknown parameter count)",
            model.name
        ));
    }
    // Memory floor: the database's min_vram_gb is authoritative for the
    // entry's own quant (it already includes KV/runtime headroom); for a
    // quant override, fall back to weights × headroom.
    let memory_gb = if quant == model.quantization {
        model
            .min_vram_gb
            .unwrap_or(model.min_ram_gb)
            .max(weights_gb)
    } else {
        weights_gb * WEIGHTS_HEADROOM
    };
    // MoE decode reads only the active experts per token (fit.rs estimate_tps).
    let read_gb = match model.active_parameters.filter(|_| model.is_moe) {
        Some(active) => active as f64 / 1_000_000_000.0 * quant_bpp(&quant),
        None => weights_gb,
    };
    // tok/s ≈ bandwidth × efficiency / read  ⇒  bandwidth ≥ tps × read / eff
    let min_bw = target.min_tps * read_gb * 100.0 / f64::from(target.efficiency_pct);
    Ok(FitBounds {
        memory_gi: memory_gb.ceil() as u64,
        min_bandwidth_gbs: min_bw.ceil() as u64,
        weights_gb,
        read_gb,
        quant,
    })
}

/// DNS-label-safe name derived from the model name, e.g.
/// "Qwen2.5 32B Instruct" → "qwen2-5-32b-instruct-fit".
pub fn claim_name(model: &LlmModel) -> String {
    let mut s: String = model
        .name
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    let s = s.trim_matches('-');
    let mut base = s.chars().take(48).collect::<String>();
    base = base.trim_matches('-').to_string();
    format!("{base}-fit")
}

/// Render the resolved fit bounds as machine-readable JSON, for programmatic
/// consumers (the llmfit-dra ModelClaim controller renders its own
/// ResourceClaimTemplate from these numbers instead of scraping YAML).
/// `resolver_version` is the binary version, recorded so consumers can
/// re-resolve when the model database advances.
pub fn render_json(
    model: &LlmModel,
    target: &ClaimTarget,
    resolver_version: &str,
) -> Result<String, String> {
    let b = fit_bounds(model, target)?;
    let name = target.name.clone().unwrap_or_else(|| claim_name(model));
    let out = serde_json::json!({
        "model": model.name,
        "claimName": name,
        "quant": b.quant,
        "weightsGb": (b.weights_gb * 10.0).round() / 10.0,
        "memoryGi": b.memory_gi,
        "minBandwidthGBs": b.min_bandwidth_gbs,
        "minTps": target.min_tps,
        "efficiencyPct": target.efficiency_pct,
        "deviceClass": target.device_class,
        "resolverVersion": resolver_version,
    });
    serde_json::to_string_pretty(&out).map_err(|e| format!("JSON serialization failed: {e}"))
}

/// Render the claim YAML. Built as a template string (not serde) so the
/// output carries provenance comments explaining where every constant came
/// from — the file is meant to be committed to GitOps repos and read by
/// humans.
pub fn render(model: &LlmModel, target: &ClaimTarget) -> Result<String, String> {
    let b = fit_bounds(model, target)?;
    let name = target.name.clone().unwrap_or_else(|| claim_name(model));
    let eff = target.efficiency_pct;
    let ind = if target.template { "    " } else { "  " };
    // Continuation lines must sit at exactly the block scalar's content
    // indentation (first line = ind + 16) for clean YAML folding.
    let pad = format!("{ind}                ");
    // Every optional lookup is guarded with CEL map membership: a missing
    // attribute must mean "device does not match", not a CEL runtime error.
    // Unguarded access errors on any device without the attribute (the cpu0
    // fallback, unindexed virtual display adapters on servers with BMC
    // framebuffers) and can wrongly disqualify allocations.
    let cel = format!(
        "'memory' in device.capacity['{d}'] &&\n\
         {pad}device.capacity['{d}'].memory.compareTo(quantity('{mem}Gi')) >= 0 &&\n\
         {pad}'memoryBandwidthGBs' in device.attributes['{d}'] &&\n\
         {pad}device.attributes['{d}'].memoryBandwidthGBs >= {bw} &&\n\
         {pad}'healthy' in device.attributes['{d}'] &&\n\
         {pad}device.attributes['{d}'].healthy",
        d = DRIVER_DOMAIN,
        mem = b.memory_gi,
        bw = b.min_bandwidth_gbs,
    );
    let header = format!(
        "# Generated by llmfit claim — do not compute these constants by hand.\n\
         # model:  {name} ({params:.1}B params, {quant} ≈ {weights:.1} GB weights)\n\
         # fit:    tok/s ≈ bandwidth × {eff}% / {read:.1} GB{active}  ⇒  bandwidth ≥ {bw} GB/s for ≥ {tps} tok/s\n\
         # memory: ≥ {mem} Gi (weights + KV/runtime headroom)\n",
        name = model.name,
        params = model.params_b(),
        quant = b.quant,
        weights = b.weights_gb,
        read = b.read_gb,
        active = if b.read_gb < b.weights_gb {
            " active"
        } else {
            ""
        },
        eff = eff,
        bw = b.min_bandwidth_gbs,
        tps = target.min_tps,
        mem = b.memory_gi,
    );
    let devices = format!(
        "devices:\n\
         {i}  requests:\n\
         {i}    - name: model\n\
         {i}      exactly:\n\
         {i}        deviceClassName: {class}\n\
         {i}        selectors:\n\
         {i}          - cel:\n\
         {i}              expression: >-\n\
         {i}                {cel}",
        i = ind,
        class = target.device_class,
        cel = cel,
    );
    let body = if target.template {
        format!(
            "apiVersion: resource.k8s.io/v1\n\
             kind: ResourceClaimTemplate\n\
             metadata:\n\
             \x20 name: {name}\n\
             spec:\n\
             \x20 spec:\n\
             \x20   {devices}\n"
        )
    } else {
        format!(
            "apiVersion: resource.k8s.io/v1\n\
             kind: ResourceClaim\n\
             metadata:\n\
             \x20 name: {name}\n\
             spec:\n\
             \x20 {devices}\n"
        )
    };
    Ok(format!("{header}{body}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(quant: &str, min_vram: Option<f64>) -> LlmModel {
        serde_json::from_value(serde_json::json!({
            "name": "Test Model 7B",
            "provider": "test",
            "parameter_count": "7B",
            "parameters_raw": 7_000_000_000u64,
            "min_ram_gb": 8.0,
            "recommended_ram_gb": 10.0,
            "min_vram_gb": min_vram,
            "quantization": quant,
            "context_length": 8192,
            "use_case": "general",
        }))
        .unwrap()
    }

    #[test]
    fn render_json_golden() {
        let json = render_json(
            &model("Q4_K_M", Some(6.0)),
            &ClaimTarget::default(),
            "9.9.9-test",
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["model"], "Test Model 7B");
        assert_eq!(v["claimName"], "test-model-7b-fit");
        assert_eq!(v["quant"], "Q4_K_M");
        assert_eq!(v["memoryGi"], 6);
        assert_eq!(v["minBandwidthGBs"], 148);
        assert_eq!(v["minTps"], 20.0);
        assert_eq!(v["efficiencyPct"], 55);
        assert_eq!(v["deviceClass"], "llmfit.ai");
        assert_eq!(v["resolverVersion"], "9.9.9-test");
        // weightsGb rounded to one decimal
        assert!(v["weightsGb"].as_f64().unwrap() > 3.0);
    }

    #[test]
    fn render_json_propagates_resolution_errors() {
        let mut t = ClaimTarget::default();
        t.min_tps = 0.0;
        assert!(render_json(&model("Q4_K_M", None), &t, "x").is_err());
    }

    #[test]
    fn bounds_use_db_memory_for_entry_quant() {
        // Q4_K_M: 7B × 0.58 bpp = 4.06 GB weights; db min_vram 6.0 wins.
        let b = fit_bounds(&model("Q4_K_M", Some(6.0)), &ClaimTarget::default()).unwrap();
        assert_eq!(b.memory_gi, 6);
        // bw ≥ 20 × 4.06 / 0.55 = 147.6… → 148
        assert_eq!(b.min_bandwidth_gbs, 148);
        assert_eq!(b.quant, "Q4_K_M");
    }

    #[test]
    fn bounds_apply_headroom_on_quant_override() {
        let t = ClaimTarget {
            quant: Some("Q8_0".to_string()),
            ..ClaimTarget::default()
        };
        // Q8_0: 7B × 1.05 bpp = 7.35 GB weights; ×1.2 headroom = 8.82 → 9 Gi.
        let b = fit_bounds(&model("Q4_K_M", Some(6.0)), &t).unwrap();
        assert_eq!(b.memory_gi, 9);
        assert_eq!(b.min_bandwidth_gbs, 268); // 20 × 7.35 / 0.55 = 267.3…
    }

    #[test]
    fn bounds_reject_unknown_quant_override() {
        for q in ["q8_0", "BOGUS"] {
            let t = ClaimTarget {
                quant: Some(q.to_string()),
                ..ClaimTarget::default()
            };
            let err = fit_bounds(&model("Q4_K_M", Some(6.0)), &t).unwrap_err();
            assert!(err.contains("Unrecognized --quant"), "{err}");
        }
        for q in ["I2_S", "TQ2_0", "TQ1_0"] {
            let t = ClaimTarget {
                quant: Some(q.to_string()),
                ..ClaimTarget::default()
            };
            assert!(fit_bounds(&model("Q4_K_M", Some(6.0)), &t).is_ok(), "{q}");
        }
    }

    #[test]
    fn bounds_use_active_parameters_for_moe_bandwidth() {
        // 30B total / 3B active: memory holds all experts, but decode reads
        // only the active ones per token, as in fit.rs estimate_tps.
        let mut moe = model("Q4_K_M", None);
        moe.name = "Test MoE 30B-A3B".to_string();
        moe.parameters_raw = Some(30_000_000_000);
        moe.is_moe = true;
        moe.active_parameters = Some(3_000_000_000);
        let b = fit_bounds(&moe, &ClaimTarget::default()).unwrap();
        // 30B × 0.58 bpp = 17.4 GB weights.
        assert_eq!(b.memory_gi, 18);
        // bw ≥ 20 × (3B × 0.58) / 0.55 = 63.27… → 64
        assert_eq!(b.min_bandwidth_gbs, 64);

        // Without an active count, fall back to all weights like fit.rs.
        moe.active_parameters = None;
        let b = fit_bounds(&moe, &ClaimTarget::default()).unwrap();
        assert_eq!(b.min_bandwidth_gbs, 633); // 20 × 17.4 / 0.55 = 632.7…
    }

    #[test]
    fn bounds_reject_nonsense() {
        assert!(
            fit_bounds(
                &model("Q4_K_M", None),
                &ClaimTarget {
                    min_tps: 0.0,
                    ..ClaimTarget::default()
                }
            )
            .is_err()
        );
        assert!(
            fit_bounds(
                &model("Q4_K_M", None),
                &ClaimTarget {
                    efficiency_pct: 0,
                    ..ClaimTarget::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn name_is_dns_label_safe() {
        let mut m = model("Q4_K_M", None);
        m.name = "Qwen2.5 32B Instruct".to_string();
        assert_eq!(claim_name(&m), "qwen2-5-32b-instruct-fit");
    }

    #[test]
    fn render_claim_yaml_shape() {
        let y = render(&model("Q4_K_M", Some(6.0)), &ClaimTarget::default()).unwrap();
        assert!(y.contains("kind: ResourceClaim\n"));
        assert!(y.contains("name: test-model-7b-fit"));
        assert!(y.contains("deviceClassName: llmfit.ai"));
        assert!(y.contains("quantity('6Gi')"));
        assert!(y.contains("memoryBandwidthGBs >= 148"));
        assert!(y.contains(".healthy"));
    }

    #[test]
    fn render_guards_every_optional_lookup() {
        // Missing attributes must be a non-match, not a CEL runtime error:
        // each capacity/attribute access is preceded by an `in` guard.
        let y = render(&model("Q4_K_M", Some(6.0)), &ClaimTarget::default()).unwrap();
        for guard in [
            "'memory' in device.capacity['llmfit.ai']",
            "'memoryBandwidthGBs' in device.attributes['llmfit.ai']",
            "'healthy' in device.attributes['llmfit.ai']",
        ] {
            assert!(y.contains(guard), "missing guard: {guard}\n{y}");
        }
        // Guard must appear before the corresponding access.
        let mem_guard = y.find("'memory' in").unwrap();
        let mem_access = y.find(".memory.compareTo").unwrap();
        assert!(mem_guard < mem_access);
    }

    #[test]
    fn render_template_wraps_spec() {
        let t = ClaimTarget {
            template: true,
            ..ClaimTarget::default()
        };
        let y = render(&model("Q4_K_M", None), &t).unwrap();
        assert!(y.contains("kind: ResourceClaimTemplate\n"));
        assert!(y.contains("spec:\n  spec:\n"));
    }
}
