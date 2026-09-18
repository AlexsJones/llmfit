//! Replay the embedded benchmark data through the fit estimator and gate the
//! median error against reality, per chip class (issue #972).
//!
//! The replay covers both embedded measurement sources:
//!
//! - the localmaxxing leaderboard cache (`data/benchmark_cache.json`, read
//!   through [`llmfit_core::benchmarks`]), and
//! - the community submissions aggregated from `data/community/`.
//!
//! Both replay through the same estimator the `fit` and `plan` commands use
//! (`plan::estimate_model_plan_with_config` delegates to the fit estimator
//! whenever the GPU bandwidth is known), with the same comparability filters
//! as the in-crate calibration replay: single-request generation only, no
//! speculative decoding or MTP, models present in the catalog, and only
//! configurations that fit the GPU path. Community submissions record no
//! engine flags, so the batch and draft-acceleration filters apply to
//! leaderboard rows only.
//!
//! The gate is the median relative error `|estimate/measured - 1|` per chip
//! class (`UNIFIED`, `DISCRETE_GPU`). The overall median used by the older
//! replay is deliberately loose; a class-level median catches a systematic
//! bias that hides inside a fine overall figure. Classes whose median already
//! exceeded [`MAX_MEDIAN_ERROR`] when this gate landed are recorded in
//! [`EXCEPTIONS`] with a ceiling just above their measured baseline; remove an
//! entry once the class is back under the target.
//!
//! Refresh protocol: a weekly data refresh may move a class median a few
//! points without any estimator change. If the gate fires right after a
//! refresh, resolve it in the same PR as the data: either fix the estimator
//! or deliberately raise and record the ceiling. Never bump a ceiling
//! silently.
//!
//! Recorded baselines when the gate landed (2026-09-18; cache scraped
//! 2026-08-31): 665 rows total, 397 leaderboard + 268 community.
//!   UNIFIED:      n=184, median error 0.206 (median ratio 0.84)
//!   DISCRETE_GPU: n=481, median error 0.301 (median ratio 0.96, excepted)
//!
//! The replay is hermetic: synthetic `SystemSpecs`, embedded data only, and a
//! pinned `CalcConfig` (no environment or measurement inputs).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use serde_json::Value;

use llmfit_core::benchmarks;
use llmfit_core::fit::{CalcConfig, DEFAULT_ESTIMATION_CTX, resolve_gpu_bandwidth};
use llmfit_core::hardware::{GpuBackend, GpuInfo, SystemSpecs};
use llmfit_core::models::{LlmModel, ModelDatabase};
use llmfit_core::plan::{PlanRequest, PlanRunPath, estimate_model_plan_with_config};
use llmfit_core::providers;

/// Median relative-error ceiling per chip class.
const MAX_MEDIAN_ERROR: f64 = 0.25;

/// Classes whose median error exceeded [`MAX_MEDIAN_ERROR`] when this gate
/// landed. The recorded value is a ceiling, not the measured baseline: it
/// carries a little headroom so the weekly data refresh cannot redden `main`
/// without an estimator change. The gate fails when a class regresses past
/// its ceiling; delete the entry once the class improves below
/// [`MAX_MEDIAN_ERROR`].
///
/// `DISCRETE_GPU` measured a 0.301 median error on 2026-09-18 (481 rows:
/// 362 leaderboard, 119 community), hence the 0.35 ceiling.
const EXCEPTIONS: &[(&str, f64)] = &[("DISCRETE_GPU", 0.35)];

/// Landing-time replayed rows per gated class. Each class must keep at least
/// [`MIN_COVERAGE`] of its landing sample so a silent shrinkage (say, most
/// rows suddenly failing the memory fit) cannot pass.
const LANDING_COUNTS: &[(&str, usize)] = &[("UNIFIED", 184), ("DISCRETE_GPU", 481)];

/// Fraction of the landing sample a class must retain.
const MIN_COVERAGE: f64 = 0.6;

/// Minimum replayed rows overall, guarding against silent data shrinkage.
const MIN_TOTAL_ROWS: usize = 400;

/// Ceiling on the share of leaderboard rows whose stated quantization label
/// fell back to the catalog quantization (44% at landing). A jump here means
/// the gate would mostly be measuring label translation; revisit the fallback
/// policy before loosening.
const MAX_QUANT_FALLBACK_SHARE: f64 = 0.60;

/// One replayed measurement: the estimate/measured ratio and its provenance.
struct Sample {
    class: String,
    source: &'static str,
    ratio: f64,
}

#[derive(Default)]
struct Skips {
    unknown_model: usize,
    no_class: usize,
    no_hardware: usize,
    unsupported_quant: usize,
    quant_approximated: usize,
    no_bandwidth: usize,
    multi_gpu: usize,
    not_fitting: usize,
    implausible_tps: usize,
    ambiguous_tag: usize,
}

/// Mirror of the in-crate calibration helper: synthetic specs for a
/// leaderboard preset label like `"RTX 3090 (24 GB)"`. Returns `None` for
/// presets the calibration cannot model faithfully (CPU-only presets, or GPUs
/// without a bandwidth entry: the estimator is bandwidth-driven). Keep in
/// sync with the helper in `fit.rs` tests (test-module private, unreachable
/// from here).
fn specs_for_preset_label(label: &str) -> Option<SystemSpecs> {
    let (name, rest) = label.split_once(" (")?;
    let vram_gb: f64 = rest
        .trim_end_matches(')')
        .trim_end_matches(" GB")
        .trim()
        .parse()
        .ok()?;
    if name == "CPU Only" {
        return None;
    }
    let unified = name.starts_with("Apple");
    let backend = if unified {
        GpuBackend::Metal
    } else if name.starts_with("RX ") || name.contains("Radeon") {
        GpuBackend::Rocm
    } else {
        GpuBackend::Cuda
    };
    llmfit_core::hardware::gpu_memory_bandwidth_gbps(name)?;
    let total_ram_gb = if unified {
        vram_gb
    } else {
        (2.0 * vram_gb).max(32.0)
    };
    Some(SystemSpecs {
        total_ram_gb,
        available_ram_gb: total_ram_gb * 0.85,
        total_cpu_cores: 16,
        cpu_name: "calibration".to_string(),
        has_gpu: true,
        gpu_vram_gb: Some(vram_gb),
        total_gpu_vram_gb: Some(vram_gb),
        gpu_available_gb: None,
        gpu_name: Some(name.to_string()),
        gpu_count: 1,
        unified_memory: unified,
        backend,
        gpus: vec![GpuInfo {
            name: name.to_string(),
            vram_gb: Some(vram_gb),
            backend,
            count: 1,
            unified_memory: unified,
        }],
        cluster_mode: false,
        cluster_node_count: 0,
    })
}

/// Synthetic specs for a community submission's `hardware` payload. Returns
/// `None` when the payload cannot be modelled (missing GPU name). Multi-GPU
/// submissions are filtered out before this point: the estimator models a
/// single GPU, so tensor-parallel runs are not comparable to its roofline.
fn specs_for_community_hardware(hw: &Value) -> Option<SystemSpecs> {
    let gpu_name = hw["hardwareName"].as_str()?.to_string();
    let unified = hw["unifiedMemory"].as_bool().unwrap_or(false);
    let vram_gb = hw["vramGb"].as_f64();
    let ram_gb = hw["ramGb"].as_f64().unwrap_or(0.0).max(0.0);
    let cpu_name = hw["cpu"].as_str().unwrap_or("community").to_string();
    let cores = hw["cpuCores"].as_u64().unwrap_or(8) as usize;
    let backend = if gpu_name.contains("Apple") {
        GpuBackend::Metal
    } else if gpu_name.contains("Radeon") || gpu_name.starts_with("RX ") || gpu_name.contains("AMD")
    {
        GpuBackend::Rocm
    } else if gpu_name.contains("Intel") {
        GpuBackend::Vulkan
    } else {
        GpuBackend::Cuda
    };
    Some(SystemSpecs {
        total_ram_gb: ram_gb,
        available_ram_gb: ram_gb * 0.85,
        total_cpu_cores: cores,
        cpu_name,
        has_gpu: true,
        gpu_vram_gb: vram_gb,
        total_gpu_vram_gb: vram_gb,
        gpu_available_gb: None,
        gpu_name: Some(gpu_name.clone()),
        gpu_count: 1,
        unified_memory: unified,
        backend,
        gpus: vec![GpuInfo {
            name: gpu_name,
            vram_gb,
            backend,
            count: 1,
            unified_memory: unified,
        }],
        cluster_mode: false,
        cluster_node_count: 0,
    })
}

/// Mirror of `models`' canonical slug (org prefix stripped, lowercased,
/// punctuation removed); the original is crate-private, so keep this in sync
/// with `models::canonical_slug`.
fn canonical_slug(name: &str) -> String {
    let slug = name.split('/').next_back().unwrap_or(name);
    slug.to_lowercase().replace(['-', '_', '.'], "")
}

/// The string forms of one provider tag that participate in
/// [`providers::tag_matches_model`]: the lowercased tag itself, and the
/// basename with its `.gguf` extension and quant suffixes stripped.
fn tag_match_keys(tag: &str) -> (String, Vec<String>) {
    let lower = tag.to_lowercase();
    let stem = lower
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(&lower)
        .trim_end_matches(".gguf")
        .to_string();
    let mut stems = Vec::new();
    if let Some(base) = providers::strip_gguf_quant_suffix(&stem) {
        stems.push(base);
    }
    if let Some(base) = providers::strip_mlx_quant_suffix(&stem) {
        stems.push(base);
    }
    stems.push(stem);
    (lower, stems)
}

/// The quantization to replay a row at: the measured label when the planner
/// understands it, otherwise the model's catalog quantization (the same
/// resolution `plan` applies to an unspecified quant). Rows with a stated but
/// unrecognized label are counted in `Skips::quant_approximated`; rows with no
/// label take the catalog quantization directly.
fn resolve_quant(row_quant: &str, model: &LlmModel, skips: &mut Skips) -> Option<String> {
    if !row_quant.is_empty() {
        if let Some(quant) = llmfit_core::plan::normalize_quant(row_quant) {
            return Some(quant);
        }
        skips.quant_approximated += 1;
    }
    llmfit_core::plan::normalize_quant(&model.quantization)
}

/// Same GPU-path memory fit check as the in-crate replay: on unified memory
/// the pool is shared, otherwise the weights must fit VRAM.
fn fits_gpu(model: &LlmModel, quant: &str, ctx: u32, specs: &SystemSpecs) -> bool {
    let mem = model.estimate_memory_gb(quant, ctx);
    specs.unified_memory || specs.gpu_vram_gb.map(|v| mem <= v).unwrap_or(false)
}

/// Estimate through the public planner, which delegates to the fit estimator
/// on the bandwidth path; `None` when the GPU path has no estimate.
fn estimate_via_plan(
    model: &LlmModel,
    quant: &str,
    specs: &SystemSpecs,
    config: &CalcConfig,
) -> Option<f64> {
    let request = PlanRequest {
        context: DEFAULT_ESTIMATION_CTX,
        quant: Some(quant.to_string()),
        target_tps: None,
        kv_quant: None,
    };
    let plan = estimate_model_plan_with_config(model, &request, specs, config).ok()?;
    plan.run_paths
        .iter()
        .find(|path| path.path == PlanRunPath::Gpu)
        .and_then(|path| path.estimated_tps)
}

/// Lower median of a non-empty slice (same convention as the in-crate replay).
fn median(sorted: &[f64]) -> f64 {
    sorted[(sorted.len() - 1) / 2]
}

/// The active ceiling for a gated class.
fn ceiling_for(class: &str) -> f64 {
    EXCEPTIONS
        .iter()
        .find(|(name, _)| *name == class)
        .map(|(_, ceiling)| *ceiling)
        .unwrap_or(MAX_MEDIAN_ERROR)
}

/// Find the catalog model whose name matches a provider tag, through the same
/// matcher the fit annotation path uses. When several catalog entries match,
/// the shortest name wins (ties broken lexicographically): the catalog dedup
/// pass emits entries in hash-map order, so a "first match" pick would differ
/// between runs. Returns the match plus the total number of matches (for the
/// ambiguity count).
fn model_for_tag<'a>(tag: &str, candidates: &[&'a LlmModel]) -> (Option<&'a LlmModel>, usize) {
    let mut matched: Option<&LlmModel> = None;
    let mut match_count = 0usize;
    for model in candidates {
        if providers::tag_matches_model(tag, &model.name) {
            match_count += 1;
            let better = match matched {
                None => true,
                Some(prev) => {
                    (model.name.len(), model.name.as_str()) < (prev.name.len(), prev.name.as_str())
                }
            };
            if better {
                matched = Some(model);
            }
        }
    }
    (matched, match_count)
}

#[test]
fn estimator_median_error_within_target_per_chip_class() {
    let started = Instant::now();
    let db = ModelDatabase::embedded();
    let models = db.get_all_models();
    // Pin the RAM-bandwidth input so the replay stays strictly hermetic: the
    // automatic default (LLMFIT_DDR_BANDWIDTH or a measured sweep) feeds only
    // the CPU-offload / MoE paths, which this gate never reads.
    let config = CalcConfig {
        ddr_bandwidth_gbps: Some(50.0),
        ..CalcConfig::default()
    };

    let mut samples: Vec<Sample> = Vec::new();
    let mut skips = Skips::default();
    let mut lb_quant_rows = 0usize;

    // ── localmaxxing leaderboard cache ────────────────────────────────
    // The catalog can hold two entries with the same canonical slug (an hf
    // entry and its onnx twin); pick the shortest name, ties lexicographic,
    // instead of relying on the dedup pass's hash-map iteration order.
    let mut slug_to_model: HashMap<String, &LlmModel> = HashMap::new();
    for model in models {
        let entry = slug_to_model
            .entry(canonical_slug(&model.name))
            .or_insert(model);
        if (model.name.len(), model.name.as_str()) < (entry.name.len(), entry.name.as_str()) {
            *entry = model;
        }
    }
    for label in benchmarks::cached_preset_labels() {
        let Some(specs) = specs_for_preset_label(label) else {
            continue;
        };
        let Some(resp) = benchmarks::cached_leaderboard_for_preset(label) else {
            continue;
        };
        // Rows replay against the preset's hardware, not the row's
        // self-reported gpuName; that mirrors the in-crate replay, so minor
        // within-preset variations (e.g. XT vs XTX) and rows whose hardware
        // reports several GPUs or tensor-parallel engine flags are all part
        // of the recorded baseline.
        for row in &resp.rows {
            let class = row
                .hardware
                .as_ref()
                .and_then(|hw| hw.hw_class.clone())
                .unwrap_or_else(|| {
                    if specs.unified_memory {
                        "UNIFIED".to_string()
                    } else {
                        "DISCRETE_GPU".to_string()
                    }
                });
            if class == "CPU_ONLY" {
                continue;
            }
            let Some(measured) = row.tok_s_out.filter(|t| *t > 0.5) else {
                continue;
            };
            // Single-request generation only: batched serving measures a
            // different quantity than the estimator models.
            if row.batch_size.unwrap_or(1) > 1 {
                continue;
            }
            // Draft-accelerated runs exceed the memory-bandwidth roofline
            // that plain autoregressive estimates model.
            if row
                .engine_flags
                .as_ref()
                .is_some_and(|f| f.spec_decoding.unwrap_or(false) || f.mtp_enabled.unwrap_or(false))
            {
                continue;
            }
            let hf_id = row.hf_id();
            if hf_id.is_empty() {
                continue;
            }
            let Some(model) = slug_to_model.get(&canonical_slug(hf_id)).copied() else {
                skips.unknown_model += 1;
                continue;
            };
            lb_quant_rows += 1;
            let Some(quant) = resolve_quant(row.quantization(), model, &mut skips) else {
                skips.unsupported_quant += 1;
                continue;
            };
            let ctx = row
                .context_length
                .unwrap_or(4096)
                .min(DEFAULT_ESTIMATION_CTX);
            if !fits_gpu(model, &quant, ctx, &specs) {
                skips.not_fitting += 1;
                continue;
            }
            let Some(est) = estimate_via_plan(model, &quant, &specs, &config) else {
                // Unreachable today: the specs above are bandwidth-gated, and
                // the GPU path always carries an estimate once bandwidth is
                // known. Kept as a guard against future planner changes.
                skips.no_bandwidth += 1;
                continue;
            };
            // Same non-positive guard as the in-crate replay (unreachable
            // today; every estimator return is >= 0.1).
            if est <= 0.0 {
                continue;
            }
            samples.push(Sample {
                class,
                source: "leaderboard",
                ratio: est / measured,
            });
        }
    }

    // ── community submissions ─────────────────────────────────────────
    // First pass: keep submissions the estimator can model, and every
    // plausible (tag, measured) pair they carry. Submission payloads carry no
    // quantization field, so community rows replay at the model's catalog
    // quantization; parsing quant labels out of the tags is a possible
    // follow-up.
    struct Eligible {
        class: String,
        specs: SystemSpecs,
        results: Vec<(&'static str, f64)>,
    }
    let mut eligible: Vec<Eligible> = Vec::new();
    for submission in benchmarks::community_submissions() {
        let hw = &submission["hardware"];
        let class = hw["hwClass"].as_str().unwrap_or_default().to_string();
        if class.is_empty() {
            skips.no_class += 1;
            continue;
        }
        if class == "CPU_ONLY" {
            continue;
        }
        // The estimator models a single GPU; tensor-parallel runs across
        // several cards are not comparable to its roofline.
        if hw["gpuCount"].as_u64().unwrap_or(1) > 1 {
            skips.multi_gpu += 1;
            continue;
        }
        let Some(specs) = specs_for_community_hardware(hw) else {
            skips.no_hardware += 1;
            continue;
        };
        if resolve_gpu_bandwidth(&specs, &config).is_none() {
            skips.no_bandwidth += 1;
            continue;
        }
        let mut results = Vec::new();
        for result in submission["results"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let Some(measured) = result["avgTps"]
                .as_f64()
                .filter(|t| llmfit_core::bench::is_plausible_tps(*t))
            else {
                skips.implausible_tps += 1;
                continue;
            };
            let Some(tag) = result["model"].as_str() else {
                continue;
            };
            results.push((tag, measured));
        }
        if !results.is_empty() {
            eligible.push(Eligible {
                class,
                specs,
                results,
            });
        }
    }

    // The exact tag match is a fuzzy check that derives Ollama-style
    // candidates from each model name; running it against every catalog entry
    // for every tag would dominate the test. A model can only match if some
    // tag (or one of its stripped stems) passes the same underlying
    // installed-name checks, so build that superset once with the repo's own
    // matchers and run the exact check only across it. Set membership is
    // monotone: the prefilter cannot drop a model the exact match would take.
    // (`prefilter_matches_full_scan` below brute-forces this claim on demand.)
    let mut tag_set_lower: HashSet<String> = HashSet::new();
    let mut stem_set_all: HashSet<String> = HashSet::new();
    for submission in &eligible {
        for (tag, _) in &submission.results {
            let (lower, stems) = tag_match_keys(tag);
            tag_set_lower.insert(lower);
            stem_set_all.extend(stems);
        }
    }
    let interesting: Vec<&LlmModel> = models
        .iter()
        .filter(|m| {
            providers::is_model_installed(&m.name, &tag_set_lower)
                || providers::is_model_installed_llamacpp(&m.name, &stem_set_all)
        })
        .collect();
    let match_started = Instant::now();

    let mut tag_models: HashMap<String, Option<&LlmModel>> = HashMap::new();
    let mut est_cache: HashMap<(usize, usize), Option<f64>> = HashMap::new();
    for (sub_idx, submission) in eligible.iter().enumerate() {
        for &(tag, measured) in &submission.results {
            let model = if let Some(found) = tag_models.get(tag) {
                *found
            } else {
                let (matched, match_count) = model_for_tag(tag, &interesting);
                if match_count > 1 {
                    skips.ambiguous_tag += 1;
                }
                tag_models.insert(tag.to_string(), matched);
                matched
            };
            let Some(model) = model else {
                skips.unknown_model += 1;
                continue;
            };
            let Some(model_idx) = models.iter().position(|m| std::ptr::eq(m, model)) else {
                continue;
            };
            let est = if let Some(cached) = est_cache.get(&(sub_idx, model_idx)) {
                *cached
            } else {
                let computed = match llmfit_core::plan::normalize_quant(&model.quantization) {
                    Some(quant) => {
                        let ctx = DEFAULT_ESTIMATION_CTX;
                        if fits_gpu(model, &quant, ctx, &submission.specs) {
                            estimate_via_plan(model, &quant, &submission.specs, &config)
                        } else {
                            None
                        }
                    }
                    None => None,
                };
                est_cache.insert((sub_idx, model_idx), computed);
                computed
            };
            let Some(est) = est else {
                // Bundles rows that do not fit with a planner `None`
                // (unreachable while the bandwidth is known).
                skips.not_fitting += 1;
                continue;
            };
            if est <= 0.0 {
                continue;
            }
            samples.push(Sample {
                class: submission.class.clone(),
                source: "community",
                ratio: est / measured,
            });
        }
    }
    let match_ms = match_started.elapsed().as_millis();

    // ── aggregate, report, and gate ───────────────────────────────────
    let mut errors_by_class: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut ratios_by_class: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut errors_by_class_source: BTreeMap<(String, &'static str), Vec<f64>> = BTreeMap::new();
    for sample in &samples {
        errors_by_class
            .entry(sample.class.clone())
            .or_default()
            .push((sample.ratio - 1.0).abs());
        ratios_by_class
            .entry(sample.class.clone())
            .or_default()
            .push(sample.ratio);
        errors_by_class_source
            .entry((sample.class.clone(), sample.source))
            .or_default()
            .push((sample.ratio - 1.0).abs());
    }

    println!(
        "estimator accuracy replay: {} rows in {:.1}s (match phase {}ms)",
        samples.len(),
        started.elapsed().as_secs_f64(),
        match_ms,
    );
    println!(
        "  skips: {} unknown model, {} no chip class, {} no hardware, {} no bandwidth, {} \
         multi-GPU, {} unsupported quant, {} of {} leaderboard rows on the catalog-quant \
         fallback, {} not fitting, {} implausible tok/s, {} ambiguous tags",
        skips.unknown_model,
        skips.no_class,
        skips.no_hardware,
        skips.no_bandwidth,
        skips.multi_gpu,
        skips.unsupported_quant,
        skips.quant_approximated,
        lb_quant_rows,
        skips.not_fitting,
        skips.implausible_tps,
        skips.ambiguous_tag,
    );
    for (class, errors) in &errors_by_class {
        let mut sorted_errors = errors.clone();
        sorted_errors.sort_by(|a, b| a.partial_cmp(b).expect("finite ratios"));
        let mut sorted_ratios = ratios_by_class[class].clone();
        sorted_ratios.sort_by(|a, b| a.partial_cmp(b).expect("finite ratios"));
        let pct = |sorted: &[f64], p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize];
        let limit = ceiling_for(class);
        let median_error = median(&sorted_errors);
        println!(
            "  {class}: n={} median_error={:.3} (ceiling {:.2}, margin {:+.3}) median_ratio={:.2} \
             ratio_p10={:.2} ratio_p90={:.2}",
            errors.len(),
            median_error,
            limit,
            limit - median_error,
            median(&sorted_ratios),
            pct(&sorted_ratios, 0.10),
            pct(&sorted_ratios, 0.90),
        );
        for source in ["leaderboard", "community"] {
            let Some(source_errors) = errors_by_class_source.get(&(class.clone(), source)) else {
                continue;
            };
            let mut sorted = source_errors.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite errors"));
            println!(
                "    {source}: n={} median_error={:.3}",
                sorted.len(),
                median(&sorted),
            );
        }
    }

    assert!(
        samples.len() >= MIN_TOTAL_ROWS,
        "replay produced only {} rows; did the cache or catalog shrink?",
        samples.len()
    );
    assert!(
        lb_quant_rows > 0,
        "no leaderboard rows reached quantization resolution; the fallback-share guard would \
         silently skip"
    );
    let share = skips.quant_approximated as f64 / lb_quant_rows as f64;
    assert!(
        share <= MAX_QUANT_FALLBACK_SHARE,
        "quant-label fallback share {share:.2} exceeds the {MAX_QUANT_FALLBACK_SHARE:.2} \
         bound (recorded 0.44 at landing): the gate would mostly be measuring label \
         translation; revisit the fallback policy"
    );
    for class in errors_by_class.keys() {
        assert!(
            LANDING_COUNTS.iter().any(|(known, _)| known == class),
            "ungated chip class {class:?} appeared; add it to the gate and LANDING_COUNTS"
        );
    }
    for (class, landing_rows) in LANDING_COUNTS {
        let errors = errors_by_class
            .get(*class)
            .unwrap_or_else(|| panic!("no replayed rows for chip class {class}"));
        let floor = (*landing_rows as f64 * MIN_COVERAGE) as usize;
        assert!(
            errors.len() >= floor,
            "chip class {class} replayed only {} rows, below the {floor}-row floor ({MIN_COVERAGE:.2} \
             of the {landing_rows} landing rows)",
            errors.len()
        );
        let mut sorted = errors.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite errors"));
        let median_error = median(&sorted);
        let limit = ceiling_for(class);
        assert!(
            median_error <= limit,
            "chip class {class}: median |error| {median_error:.3} exceeds the recorded ceiling \
             {limit:.2} across {} rows; check the estimator before loosening",
            sorted.len()
        );
    }
}

/// Opt-in parity guard for the tag-matching prefilter above. Brute-forces
/// every community tag over the full catalog and fails if the prefilter drops
/// or reorders a match. Ignored by default because the full scan takes
/// minutes; run it after touching the provider matchers:
/// `cargo test -p llmfit-core --test estimator_accuracy -- --ignored`.
#[test]
#[ignore = "brute-force full-scan parity check; run when the provider matchers change"]
fn prefilter_matches_full_scan() {
    let db = ModelDatabase::embedded();
    let models = db.get_all_models();
    let mut tags: Vec<&str> = Vec::new();
    for submission in benchmarks::community_submissions() {
        for result in submission["results"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            if let Some(tag) = result["model"].as_str() {
                tags.push(tag);
            }
        }
    }
    tags.sort_unstable();
    tags.dedup();
    let mut tag_set_lower: HashSet<String> = HashSet::new();
    let mut stem_set_all: HashSet<String> = HashSet::new();
    for tag in &tags {
        let (lower, stems) = tag_match_keys(tag);
        tag_set_lower.insert(lower);
        stem_set_all.extend(stems);
    }
    let interesting: Vec<&LlmModel> = models
        .iter()
        .filter(|m| {
            providers::is_model_installed(&m.name, &tag_set_lower)
                || providers::is_model_installed_llamacpp(&m.name, &stem_set_all)
        })
        .collect();
    let all: Vec<&LlmModel> = models.iter().collect();
    for tag in &tags {
        let (fast, fast_n) = model_for_tag(tag, &interesting);
        let (slow, slow_n) = model_for_tag(tag, &all);
        assert_eq!(
            fast.map(|m| m.name.as_str()),
            slow.map(|m| m.name.as_str()),
            "prefilter resolved tag {tag:?} differently from the full scan"
        );
        assert_eq!(fast_n, slow_n, "prefilter dropped matches for tag {tag:?}");
    }
    println!(
        "prefilter parity: {} tags checked, {} prefiltered models",
        tags.len(),
        interesting.len()
    );
}
