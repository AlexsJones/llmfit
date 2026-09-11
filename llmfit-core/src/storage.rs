//! Storage for one library of models used sequentially, at their selected quants.
//! Hardware analysis stays in `analysis`; this module has no I/O or side effects.

use crate::fit::{FitLevel, InferenceRuntime, ModelFit, rank_models_by_fit};
use std::collections::BTreeMap;

/// Generic marketed capacities, in decimal GB. Not every device offers every tier.
pub const SSD_TIERS_GB: &[u32] = &[256, 512, 1000, 2000, 4000, 8000, 16000];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageSelection {
    #[default]
    Score,
    Largest,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
#[serde(tag = "mode", content = "size_gb", rename_all = "snake_case")]
pub enum ScratchPolicy {
    #[default]
    Auto,
    Fixed(f64),
}

#[derive(Debug, Clone)]
pub struct StorageRequest {
    pub keep: usize,
    pub selection: StorageSelection,
    pub os_reserve_gb: f64,
    pub scratch: ScratchPolicy,
    pub headroom_percent: u8,
    pub perfect: bool,
}

impl Default for StorageRequest {
    fn default() -> Self {
        Self {
            keep: 3,
            selection: StorageSelection::Score,
            os_reserve_gb: 100.0,
            scratch: ScratchPolicy::Auto,
            headroom_percent: 15,
            perfect: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StorageModelEstimate {
    pub name: String,
    pub best_quant: String,
    pub fit_level: FitLevel,
    pub runtime: InferenceRuntime,
    pub score: f64,
    pub disk_size_gb: f64,
    pub effective_context_length: u32,
}

impl From<&ModelFit> for StorageModelEstimate {
    fn from(fit: &ModelFit) -> Self {
        Self {
            name: fit.model.name.clone(),
            best_quant: fit.best_quant.clone(),
            fit_level: fit.fit_level,
            runtime: fit.runtime,
            score: fit.score,
            disk_size_gb: fit.model.estimate_disk_gb(&fit.best_quant),
            effective_context_length: fit.effective_context_length,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StorageEstimate {
    pub selection: StorageSelection,
    pub keep_requested: usize,
    pub selected_count: usize,
    pub eligible_count: usize,
    pub perfect: bool,
    pub models: Vec<StorageModelEstimate>,
    pub library_gb: f64,
    pub os_reserve_gb: f64,
    pub scratch_policy: ScratchPolicy,
    pub download_scratch_gb: f64,
    pub headroom_percent: u8,
    pub need_gb: f64,
    pub target_capacity_gb: f64,
    /// None when no models were selected or no listed tier is large enough.
    pub minimum_ssd_gb: Option<u32>,
    pub suggested_ssd_gb: Option<u32>,
    pub estimate_notice: String,
    pub warnings: Vec<String>,
}

/// Parse storage into decimal GB. Hardware's legacy memory parser uses different
/// suffix conventions, so it must not be used for marketed SSD capacities.
pub fn parse_storage_size(input: &str) -> Result<f64, String> {
    let invalid = || {
        format!(
            "Invalid storage size '{input}': use a non-negative size such as 100GB, 1TB, or 1TiB"
        )
    };
    let input = input.trim();
    let split = input
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(input.len());
    let (number, suffix) = input.split_at(split);
    let number: f64 = number.parse().map_err(|_| invalid())?;
    let scale = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "g" | "gb" => 1.0,
        "m" | "mb" => 0.001,
        "t" | "tb" => 1000.0,
        "mib" => 1_048_576.0 / 1_000_000_000.0,
        "gib" => 1_073_741_824.0 / 1_000_000_000.0,
        "tib" => 1_099_511_627_776.0 / 1_000_000_000.0,
        _ => return Err(invalid()),
    };
    let gb = number * scale;
    if !gb.is_finite() || gb < 0.0 {
        return Err(invalid());
    }
    Ok(gb)
}

fn validate_nonnegative(value: f64, name: &str) -> Result<(), String> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be finite and non-negative"))
    }
}

fn next_tier(need_gb: f64) -> Option<u32> {
    SSD_TIERS_GB
        .iter()
        .copied()
        .find(|&gb| f64::from(gb) >= need_gb)
}

/// Size a library from the shared analysis builders' backend-compatible,
/// sanitization-checked fits. The caller may further narrow the search scope.
/// Counts each catalog identity once and never sums models' RAM requirements.
pub fn estimate_storage(
    fits: Vec<ModelFit>,
    request: &StorageRequest,
) -> Result<StorageEstimate, String> {
    if request.keep == 0 {
        return Err("--keep must be at least 1".to_string());
    }
    validate_nonnegative(request.os_reserve_gb, "OS reserve")?;
    if let ScratchPolicy::Fixed(gb) = request.scratch {
        validate_nonnegative(gb, "Download scratch")?;
    }
    if request.headroom_percent >= 100 {
        return Err("--headroom must be between 0 and 99 percent".to_string());
    }

    // A deterministic starting order also breaks ties in the shared stable score
    // sort. Catalog loading can otherwise expose HashMap iteration order.
    let mut unique: BTreeMap<String, ModelFit> = BTreeMap::new();
    for fit in fits.into_iter().filter(|fit| {
        fit.fit_level != FitLevel::TooTight
            && (!request.perfect || fit.fit_level == FitLevel::Perfect)
    }) {
        let row = StorageModelEstimate::from(&fit);
        if !row.disk_size_gb.is_finite() || row.disk_size_gb <= 0.0 {
            return Err(format!("Invalid weight estimate for '{}'", row.name));
        }
        if !row.score.is_finite() {
            return Err(format!("Invalid ranking score for '{}'", row.name));
        }
        // The database has already merged aliases. Preserve its full IDs:
        // HF and ONNX repositories can share a slug but store different files.
        let key = fit.model.name.clone();
        if let Some(existing) = unique.get(&key) {
            if StorageModelEstimate::from(existing) != row {
                return Err(format!(
                    "Conflicting estimates for duplicate model '{}'",
                    row.name
                ));
            }
        } else {
            unique.insert(key, fit);
        }
    }

    let eligible_count = unique.len();
    let mut ranked = rank_models_by_fit(unique.into_values().collect());
    if request.selection == StorageSelection::Largest {
        // Stable sort preserves score and identity tie-breakers. Compare the
        // actual selected weight estimates, not parameters or rounded JSON.
        ranked.sort_by(|a, b| {
            b.model
                .estimate_disk_gb(&b.best_quant)
                .total_cmp(&a.model.estimate_disk_gb(&a.best_quant))
        });
    }
    ranked.truncate(request.keep);
    let models: Vec<StorageModelEstimate> = ranked.iter().map(StorageModelEstimate::from).collect();
    let library_gb: f64 = models.iter().map(|m| m.disk_size_gb).sum();
    let download_scratch_gb = if models.is_empty() {
        0.0
    } else {
        match request.scratch {
            ScratchPolicy::Auto => models.iter().map(|m| m.disk_size_gb).fold(0.0, f64::max),
            ScratchPolicy::Fixed(gb) => gb,
        }
    };
    let need_gb = request.os_reserve_gb + library_gb + download_scratch_gb;
    let target_capacity_gb = need_gb / (1.0 - f64::from(request.headroom_percent) / 100.0);
    if !need_gb.is_finite() || !target_capacity_gb.is_finite() {
        return Err("Storage requirement exceeds the supported numeric range".to_string());
    }

    let mut warnings = Vec::new();
    let (minimum_ssd_gb, suggested_ssd_gb) = if models.is_empty() {
        warnings.push(
            "No runnable models match the filters; no SSD recommendation was made".to_string(),
        );
        (None, None)
    } else {
        if models.len() < request.keep {
            warnings.push(format!(
                "Requested {} models, but only {} are eligible",
                request.keep,
                models.len()
            ));
        }
        let minimum = next_tier(need_gb);
        let suggested = next_tier(target_capacity_gb);
        if suggested.is_none() {
            warnings.push("No listed SSD tier is large enough for the requested headroom; use target_capacity_gb to size a larger drive".to_string());
        }
        (minimum, suggested)
    };

    Ok(StorageEstimate {
        selection: request.selection,
        keep_requested: request.keep,
        selected_count: models.len(),
        eligible_count,
        perfect: request.perfect,
        models,
        library_gb,
        os_reserve_gb: request.os_reserve_gb,
        scratch_policy: request.scratch,
        download_scratch_gb,
        headroom_percent: request.headroom_percent,
        need_gb,
        target_capacity_gb,
        minimum_ssd_gb,
        suggested_ssd_gb,
        estimate_notice: "Estimated weights in decimal GB for one library copy used sequentially. Actual artifacts, auxiliary files, and runtime caches may differ. Automatic scratch covers one additional download as large as the largest selected model. Reserve and free headroom are planning allowances; current disk usage is not measured.".to_string(),
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuBackend, SystemSpecs};
    use crate::models::LlmModel;

    fn fit(name: &str, params_b: f64, quant: &str, score: f64) -> ModelFit {
        let model: LlmModel = serde_json::from_value(serde_json::json!({
            "name": name, "provider": "fixture", "parameter_count": format!("{params_b}B"),
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 1.0,
            "quantization": quant, "context_length": 8192, "use_case": "general"
        }))
        .expect("model fixture");
        let specs = SystemSpecs {
            total_ram_gb: 128.0,
            available_ram_gb: 120.0,
            total_cpu_cores: 8,
            cpu_name: "fixture".into(),
            has_gpu: true,
            gpu_vram_gb: Some(128.0),
            total_gpu_vram_gb: Some(128.0),
            gpu_available_gb: None,
            gpu_name: None,
            gpu_count: 1,
            unified_memory: false,
            backend: GpuBackend::Cuda,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        };
        let mut fit = ModelFit::analyze(&model, &specs);
        fit.best_quant = quant.to_string();
        fit.score = score;
        fit.fit_level = FitLevel::Good;
        fit
    }

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    #[test]
    fn formula_counts_weights_and_one_largest_download() {
        let mut moe = fit("fixture/moe", 120.0, "mlx-8bit", 90.0);
        moe.model.is_moe = true;
        moe.model.active_parameters = Some(5_000_000_000);
        let report = estimate_storage(
            vec![
                moe,
                fit("fixture/b", 100.0, "mlx-8bit", 80.0),
                fit("fixture/c", 80.0, "mlx-8bit", 70.0),
            ],
            &StorageRequest::default(),
        )
        .expect("report");
        close(report.library_gb, 300.0);
        close(report.download_scratch_gb, 120.0);
        close(report.need_gb, 520.0);
        close(report.target_capacity_gb, 520.0 / 0.85);
        assert_eq!(report.minimum_ssd_gb, Some(1000));
        assert_eq!(report.suggested_ssd_gb, Some(1000));
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn autoround_weights_and_scratch_choose_correct_ssd_tier() {
        // Parameter count from the reported Qwen3.6-35B-A3B regression. Eight-bit
        // weights and one download exceed 256 GB after the 200 GB reserve.
        for (quant, expected_weights, expected_tier) in [
            ("AutoRound-8bit", 34.1311488, 512),
            ("AutoRound-4bit", 17.0655744, 256),
        ] {
            let mut candidate = fit("fixture/AutoRound-35B", 34.1311488, quant, 90.0);
            candidate.model.parameters_raw = Some(34_131_148_800);
            candidate.model.format = crate::models::ModelFormat::Autoround;
            for selection in [StorageSelection::Score, StorageSelection::Largest] {
                let report = estimate_storage(
                    vec![candidate.clone()],
                    &StorageRequest {
                        keep: 1,
                        selection,
                        os_reserve_gb: 200.0,
                        headroom_percent: 0,
                        ..StorageRequest::default()
                    },
                )
                .expect("AutoRound library");
                assert_eq!(report.minimum_ssd_gb, Some(expected_tier));
                assert_eq!(report.suggested_ssd_gb, Some(expected_tier));
                close(report.library_gb, expected_weights);
                close(report.download_scratch_gb, expected_weights);
                close(report.need_gb, 200.0 + 2.0 * expected_weights);
            }
        }
    }

    #[test]
    fn selection_uses_score_or_quantized_disk_not_parameter_count() {
        let candidates = vec![
            fit("fixture/high-score", 8.0, "Q4_K_M", 95.0),
            fit("fixture/largest", 7.0, "Q8_0", 70.0),
            fit("fixture/most-params", 10.0, "Q2_K", 60.0),
        ];
        let mut request = StorageRequest {
            keep: 1,
            ..StorageRequest::default()
        };
        let score = estimate_storage(candidates.clone(), &request).expect("report");
        assert_eq!(score.models[0].name, "fixture/high-score");
        close(score.library_gb, 4.64);
        request.selection = StorageSelection::Largest;
        let largest = estimate_storage(candidates, &request).expect("report");
        assert_eq!(largest.models[0].name, "fixture/largest");
        close(largest.library_gb, 7.35);
    }

    #[test]
    fn ties_and_duplicate_identities_are_deterministic() {
        let a = fit("fixture/a", 8.0, "Q4_K_M", 90.0);
        let b = fit("fixture/b", 8.0, "Q4_K_M", 90.0);
        let duplicate = a.clone();
        for selection in [StorageSelection::Score, StorageSelection::Largest] {
            let request = StorageRequest {
                keep: 1,
                selection,
                ..StorageRequest::default()
            };
            for candidates in [
                vec![b.clone(), duplicate.clone(), a.clone()],
                vec![a.clone(), duplicate.clone(), b.clone()],
            ] {
                let report = estimate_storage(candidates, &request).expect("report");
                assert_eq!(report.eligible_count, 2);
                assert_eq!(report.models[0].name, "fixture/a");
            }
        }
        let mut conflict = a.clone();
        conflict.best_quant = "Q8_0".into();
        assert!(
            estimate_storage(vec![a, conflict], &StorageRequest::default())
                .expect_err("duplicate conflict")
                .contains("Conflicting")
        );
    }

    #[test]
    fn different_catalog_repositories_with_the_same_slug_remain_distinct() {
        let hf = fit("microsoft/phi-3-mini-4k-instruct", 3.82, "mlx-8bit", 80.0);
        let mut onnx = fit("onnx-community/Phi-3-mini-4k-instruct", 3.8, "Q8_0", 85.0);
        onnx.model.format = crate::models::ModelFormat::Onnx;
        for selection in [StorageSelection::Score, StorageSelection::Largest] {
            let request = StorageRequest {
                keep: 2,
                selection,
                ..StorageRequest::default()
            };
            let first = estimate_storage(vec![hf.clone(), onnx.clone()], &request)
                .expect("different repositories are not conflicting duplicates");
            let reversed = estimate_storage(vec![onnx.clone(), hf.clone()], &request)
                .expect("reversed catalog");
            assert_eq!(first.selected_count, 2);
            assert_eq!(first.eligible_count, 2);
            assert_eq!(first.models, reversed.models);
            close(first.library_gb, 3.82 + 3.99);
        }
    }

    #[test]
    fn filters_precede_selection_and_partial_results_are_explicit() {
        let mut tight = fit("fixture/tight", 500.0, "Q4_K_M", 100.0);
        tight.fit_level = FitLevel::TooTight;
        let mut perfect = fit("fixture/perfect", 8.0, "Q4_K_M", 80.0);
        perfect.fit_level = FitLevel::Perfect;
        let mut marginal = fit("fixture/marginal", 8.0, "Q4_K_M", 90.0);
        marginal.fit_level = FitLevel::Marginal;
        marginal.run_mode = crate::fit::RunMode::CpuOnly;
        let all = vec![tight, perfect, marginal];
        let report = estimate_storage(all.clone(), &StorageRequest::default()).expect("report");
        assert_eq!(report.selected_count, 2);
        assert_eq!(report.keep_requested, 3);
        assert_eq!(report.eligible_count, 2);
        assert_eq!(report.warnings.len(), 1);
        let report = estimate_storage(
            all,
            &StorageRequest {
                perfect: true,
                ..StorageRequest::default()
            },
        )
        .expect("report");
        assert_eq!(report.models[0].fit_level, FitLevel::Perfect);
        assert_eq!(report.selected_count, 1);
    }

    #[test]
    fn empty_results_never_recommend_a_drive() {
        let report = estimate_storage(
            vec![],
            &StorageRequest {
                scratch: ScratchPolicy::Fixed(200.0),
                ..StorageRequest::default()
            },
        )
        .expect("empty report");
        assert!(report.models.is_empty());
        close(report.library_gb, 0.0);
        close(report.download_scratch_gb, 0.0);
        close(report.need_gb, 100.0);
        assert_eq!(report.minimum_ssd_gb, None);
        assert_eq!(report.suggested_ssd_gb, None);
        assert!(!report.warnings.is_empty());
        assert!(serde_json::to_value(report).expect("JSON")["suggested_ssd_gb"].is_null());
    }

    #[test]
    fn tiers_respect_headroom_precision_and_exhaustion() {
        for (need, headroom, minimum, suggested) in [
            (500.0, 15, Some(512), Some(1000)),
            (512.0, 0, Some(512), Some(512)),
            (512.004, 0, Some(1000), Some(1000)),
            (16000.0, 0, Some(16000), Some(16000)),
            (16000.0, 15, Some(16000), None),
            (16001.0, 0, None, None),
        ] {
            let report = estimate_storage(
                vec![fit("fixture/a", 1.0, "mlx-8bit", 90.0)],
                &StorageRequest {
                    keep: 1,
                    os_reserve_gb: need - 1.0,
                    scratch: ScratchPolicy::Fixed(0.0),
                    headroom_percent: headroom,
                    ..StorageRequest::default()
                },
            )
            .expect("report");
            assert_eq!(report.minimum_ssd_gb, minimum);
            assert_eq!(report.suggested_ssd_gb, suggested);
            assert_eq!(report.warnings.is_empty(), suggested.is_some());
        }
        for &tier in SSD_TIERS_GB {
            assert_eq!(next_tier(f64::from(tier)), Some(tier));
            assert_ne!(next_tier(f64::from(tier) + 0.0001), Some(tier));
        }
    }

    #[test]
    fn explicit_scratch_replaces_automatic_allowance() {
        let report = estimate_storage(
            vec![fit("fixture/a", 8.0, "Q4_K_M", 90.0)],
            &StorageRequest {
                keep: 1,
                os_reserve_gb: 0.0,
                scratch: ScratchPolicy::Fixed(200.0),
                headroom_percent: 0,
                ..StorageRequest::default()
            },
        )
        .expect("report");
        close(report.need_gb, 204.64);
        close(report.target_capacity_gb, report.need_gb);
    }

    #[test]
    fn storage_sizes_distinguish_decimal_and_binary_units() {
        for (input, expected) in [
            ("100", 100.0),
            (" 100gB ", 100.0),
            ("1000M", 1.0),
            ("1.5T", 1500.0),
            ("1TB", 1000.0),
            ("1TiB", 1099.511627776),
            ("1GiB", 1.073741824),
            ("1MiB", 0.001048576),
            ("0G", 0.0),
        ] {
            close(parse_storage_size(input).expect("size"), expected);
        }
        for input in [
            "",
            " ",
            "-1G",
            "NaN",
            "inf",
            "1e3G",
            "1.2.3GB",
            "G",
            "1XB",
            "1GB extra",
            "١G",
        ] {
            assert!(parse_storage_size(input).is_err(), "{input}");
        }
        assert!(parse_storage_size(&format!("{}TB", "9".repeat(308))).is_err());
    }

    #[test]
    fn invalid_requests_and_nonfinite_results_are_errors() {
        for request in [
            StorageRequest {
                keep: 0,
                ..StorageRequest::default()
            },
            StorageRequest {
                headroom_percent: 100,
                ..StorageRequest::default()
            },
            StorageRequest {
                os_reserve_gb: f64::NAN,
                ..StorageRequest::default()
            },
            StorageRequest {
                os_reserve_gb: -1.0,
                ..StorageRequest::default()
            },
            StorageRequest {
                scratch: ScratchPolicy::Fixed(f64::INFINITY),
                ..StorageRequest::default()
            },
            StorageRequest {
                scratch: ScratchPolicy::Fixed(-1.0),
                ..StorageRequest::default()
            },
            StorageRequest {
                os_reserve_gb: f64::MAX,
                headroom_percent: 99,
                ..StorageRequest::default()
            },
        ] {
            assert!(estimate_storage(vec![], &request).is_err());
        }
        let mut candidate = fit("fixture/bad", 8.0, "Q4_K_M", 90.0);
        for size in ["0B", "NaNB", "infB", "-8B"] {
            candidate.model.parameter_count = size.to_string();
            assert!(estimate_storage(vec![candidate.clone()], &StorageRequest::default()).is_err());
        }
        candidate.model.parameter_count = "8B".into();
        candidate.score = f64::NAN;
        assert!(estimate_storage(vec![candidate], &StorageRequest::default()).is_err());
    }

    #[test]
    fn retaining_more_models_never_reduces_capacity_for_a_fixed_selection() {
        let fits = vec![
            fit("fixture/a", 8.0, "Q4_K_M", 90.0),
            fit("fixture/b", 12.0, "Q8_0", 80.0),
            fit("fixture/c", 20.0, "Q2_K", 70.0),
        ];
        for selection in [StorageSelection::Score, StorageSelection::Largest] {
            let mut previous = 0.0;
            for keep in 1..=5 {
                let report = estimate_storage(
                    fits.clone(),
                    &StorageRequest {
                        keep,
                        selection,
                        ..StorageRequest::default()
                    },
                )
                .expect("report");
                assert!(report.need_gb >= previous);
                previous = report.need_gb;
                assert!(f64::from(report.suggested_ssd_gb.expect("tier")) * 0.85 >= report.need_gb);
            }
        }
    }

    #[test]
    fn shared_analysis_preserves_backend_and_sanitization_gates() {
        let mut base = fit("fixture/base-8B", 8.0, "Q4_K_M", 90.0).model;
        base.min_ram_gb = 6.0;
        base.min_vram_gb = Some(6.0);
        base.recommended_ram_gb = 12.0;
        let mut mlx = base.clone();
        mlx.name = "fixture/base-8B-MLX".into();
        mlx.format = crate::models::ModelFormat::Mlx;
        let mut awq = base.clone();
        awq.name = "fixture/base-8B-AWQ".into();
        awq.format = crate::models::ModelFormat::Awq;
        awq.quantization = "AWQ-4bit".into();
        let mut draft = base.clone();
        draft.name = "fixture/EAGLE-8B".into();
        let mut tts = base.clone();
        tts.name = "fixture/voice-8B".into();
        tts.capabilities = vec![crate::models::Capability::Tts];
        let models = vec![base, mlx, awq, draft, tts];

        for (backend, unified_memory, has_gpu) in [
            (GpuBackend::Cuda, false, true),
            (GpuBackend::Rocm, false, true),
            (GpuBackend::Metal, true, true),
            (GpuBackend::CpuX86, false, false),
            (GpuBackend::CpuArm, false, false),
        ] {
            let specs = SystemSpecs {
                total_ram_gb: 128.0,
                available_ram_gb: 120.0,
                total_cpu_cores: 8,
                cpu_name: "fixture".into(),
                has_gpu,
                gpu_vram_gb: has_gpu.then_some(128.0),
                total_gpu_vram_gb: has_gpu.then_some(128.0),
                gpu_available_gb: None,
                gpu_name: None,
                gpu_count: u32::from(has_gpu),
                unified_memory,
                backend,
                gpus: vec![],
                cluster_mode: false,
                cluster_node_count: 0,
            };
            let fits: Vec<ModelFit> = crate::analysis::rankable_models(&models, &specs)
                .map(|model| ModelFit::analyze(model, &specs))
                .collect();
            if !has_gpu {
                assert!(
                    fits.iter()
                        .all(|fit| fit.run_mode == crate::fit::RunMode::CpuOnly)
                );
            }
            let report = estimate_storage(
                fits,
                &StorageRequest {
                    keep: 10,
                    ..StorageRequest::default()
                },
            )
            .expect("backend report");
            let names: Vec<&str> = report.models.iter().map(|m| m.name.as_str()).collect();
            assert!(names.contains(&"fixture/base-8B"));
            assert!(!names.contains(&"fixture/EAGLE-8B"));
            assert!(!names.contains(&"fixture/voice-8B"));
            assert_eq!(
                names.contains(&"fixture/base-8B-MLX"),
                backend == GpuBackend::Metal
            );
            assert_eq!(
                names.contains(&"fixture/base-8B-AWQ"),
                matches!(backend, GpuBackend::Cuda | GpuBackend::Rocm)
            );
            if !has_gpu {
                assert!(report.models.iter().all(|m| m.fit_level == FitLevel::Good));
            }
        }
    }

    #[test]
    fn reserve_does_not_change_model_fit_or_count_installed_models_twice() {
        let mut candidate = fit("fixture/a", 8.0, "Q4_K_M", 90.0);
        candidate.installed = true;
        let first = estimate_storage(
            vec![candidate.clone()],
            &StorageRequest {
                os_reserve_gb: 0.0,
                ..StorageRequest::default()
            },
        )
        .expect("report");
        let second = estimate_storage(
            vec![candidate],
            &StorageRequest {
                os_reserve_gb: 1000.0,
                ..StorageRequest::default()
            },
        )
        .expect("report");
        assert_eq!(first.models, second.models);
        close(first.library_gb, 4.64);
        close(second.need_gb - first.need_gb, 1000.0);
    }
}
