//! Hardware profiles — named, versioned descriptions of a machine.
//!
//! `--memory` / `--ram` / `--cpu-cores` each override one detected field, which
//! is enough to fix a bad detection but not to answer "what would this model do
//! on *that* box": the throughput estimate is driven by memory bandwidth and
//! fp16 matmul throughput, neither of which is a capacity. A profile names a
//! whole machine — capacity, unified-memory topology, and the estimator inputs
//! — so the answer is reproducible and reviewable instead of a pile of flags.
//!
//! Profiles under `data/hardware/` are aggregated by `build.rs` and embedded,
//! so a merged profile ships in the next release; users can drop their own into
//! [`user_profile_dir`] without rebuilding.
//!
//! Loading is **lenient** about unknown keys so a profile written for a newer
//! llmfit still works, while [`HardwareProfile::validate_strict`] (used by
//! `llmfit hardware validate`) rejects them — that is what turns a typo into an
//! error instead of a field that silently does nothing.

use crate::fit::CalcConfig;
use crate::hardware::SystemSpecs;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The only profile schema version this build understands.
pub const SCHEMA_VERSION: u32 = 1;

/// Profiles from `data/hardware/*.json`, aggregated by build.rs.
const EMBEDDED_PROFILES_JSON: &str =
    include_str!(concat!(env!("OUT_DIR"), "/hardware_profiles.json"));

/// Run-mode keys accepted by `calibration[].run_mode`, matching
/// [`crate::fit::RunMode`] in snake_case.
const RUN_MODES: [&str; 5] = [
    "gpu",
    "tensor_parallel",
    "moe_offload",
    "cpu_offload",
    "cpu_only",
];

const MAX_RAM_GB: f64 = 16_384.0;
const MAX_BANDWIDTH_GBPS: f64 = 100_000.0;
const MAX_TFLOPS: f64 = 100_000.0;
const MAX_TPS: f64 = 100_000.0;
const MAX_RUN_MODE_FACTOR: f64 = 10.0;

type Unknown = BTreeMap<String, serde_json::Value>;

/// A machine description. See `data/hardware/schema.json` for the contract and
/// `data/hardware/README.md` for what each field affects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub schema_version: u32,
    /// Profile id, equal to the file stem.
    pub name: String,
    /// Which GPU this profile describes. Recorded for provenance only —
    /// profiles are never auto-selected, so a wrong guess can't silently
    /// replace a correct detection.
    #[serde(rename = "match", default, skip_serializing_if = "Option::is_none")]
    pub matcher: Option<ProfileMatch>,
    pub hardware: ProfileHardware,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimation: Option<ProfileEstimation>,
    /// Measured anchors with provenance. Validated but **not applied** in
    /// schema version 1: an anchor changes every estimate on the machine, so it
    /// ships as reviewable data first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calibration: Vec<CalibrationEntry>,
    #[serde(flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileMatch {
    pub gpu_name_contains: String,
    #[serde(flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileHardware {
    pub total_ram_gb: f64,
    pub unified_memory: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_memory_bandwidth_gbps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ddr_bandwidth_gbps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_compute_tflops_fp16: Option<f64>,
    #[serde(flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileEstimation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub efficiency: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_mode_factors: Option<ProfileRunModeFactors>,
    #[serde(flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: Unknown,
}

/// Per-mode speed multipliers. Every key is optional: an unset mode keeps the
/// [`crate::fit::RunModeFactors`] default rather than being zeroed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileRunModeFactors {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_parallel: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_offload: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_offload: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_only: Option<f64>,
    #[serde(flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEntry {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_mode: Option<String>,
    pub measured_tps: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: Unknown,
}

/// Where a profile came from, so output can distinguish a bundled profile from
/// a local file the user is iterating on.
#[derive(Debug, Clone, PartialEq)]
pub enum ProfileOrigin {
    /// Embedded at build time from `data/hardware/`.
    Embedded,
    /// Discovered in [`user_profile_dir`].
    User(PathBuf),
    /// Loaded from a path the user passed explicitly.
    File(PathBuf),
}

impl ProfileOrigin {
    pub fn label(&self) -> String {
        match self {
            ProfileOrigin::Embedded => "bundled".to_string(),
            ProfileOrigin::User(path) | ProfileOrigin::File(path) => path.display().to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedProfile {
    pub origin: ProfileOrigin,
    pub profile: HardwareProfile,
}

/// Every selectable profile, plus the user files that could not be loaded.
///
/// Load failures are reported rather than skipped: a profile the user wrote and
/// cannot select is a bug they need to see, not a file to ignore.
#[derive(Debug, Clone, Default)]
pub struct ProfileCatalog {
    pub profiles: Vec<LoadedProfile>,
    pub errors: Vec<(PathBuf, String)>,
}

impl HardwareProfile {
    /// Parse profile JSON, tolerating unknown keys.
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("invalid hardware profile JSON: {e}"))
    }

    /// Check the fields this build understands.
    ///
    /// Bounds are deliberately wide — the point is to reject values that would
    /// make an estimate meaningless (zero bandwidth divides the roofline to
    /// nothing, NaN propagates into every score) rather than to police
    /// plausible hardware.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "unsupported schema_version {}; this build understands {}",
                self.schema_version, SCHEMA_VERSION
            ));
        }
        validate_name(&self.name)?;

        if let Some(matcher) = &self.matcher
            && matcher.gpu_name_contains.trim().is_empty()
        {
            return Err("match.gpu_name_contains must not be empty".to_string());
        }

        let hw = &self.hardware;
        check_range("hardware.total_ram_gb", hw.total_ram_gb, MAX_RAM_GB)?;
        for (label, value) in [
            (
                "hardware.gpu_memory_bandwidth_gbps",
                hw.gpu_memory_bandwidth_gbps,
            ),
            ("hardware.ddr_bandwidth_gbps", hw.ddr_bandwidth_gbps),
        ] {
            if let Some(value) = value {
                check_range(label, value, MAX_BANDWIDTH_GBPS)?;
            }
        }
        if let Some(tflops) = hw.gpu_compute_tflops_fp16 {
            check_range("hardware.gpu_compute_tflops_fp16", tflops, MAX_TFLOPS)?;
        }

        if let Some(est) = &self.estimation {
            if let Some(eff) = est.efficiency {
                check_range("estimation.efficiency", eff, 1.0)?;
            }
            if let Some(factors) = &est.run_mode_factors {
                for (label, value) in [
                    ("gpu", factors.gpu),
                    ("tensor_parallel", factors.tensor_parallel),
                    ("moe_offload", factors.moe_offload),
                    ("cpu_offload", factors.cpu_offload),
                    ("cpu_only", factors.cpu_only),
                ] {
                    if let Some(value) = value {
                        check_range(
                            &format!("estimation.run_mode_factors.{label}"),
                            value,
                            MAX_RUN_MODE_FACTOR,
                        )?;
                    }
                }
            }
        }

        for (i, entry) in self.calibration.iter().enumerate() {
            if entry.model.trim().is_empty() {
                return Err(format!("calibration[{i}].model must not be empty"));
            }
            check_range(
                &format!("calibration[{i}].measured_tps"),
                entry.measured_tps,
                MAX_TPS,
            )?;
            if let Some(mode) = &entry.run_mode
                && !RUN_MODES.contains(&mode.as_str())
            {
                return Err(format!(
                    "calibration[{i}].run_mode '{mode}' is not one of {}",
                    RUN_MODES.join(", ")
                ));
            }
        }

        Ok(())
    }

    /// [`Self::validate`] plus a rejection of every unrecognized key.
    pub fn validate_strict(&self) -> Result<(), String> {
        self.validate()?;
        let unknown = self.unknown_keys();
        if !unknown.is_empty() {
            return Err(format!("unknown key(s): {}", unknown.join(", ")));
        }
        Ok(())
    }

    /// Dotted paths of every key this build does not recognize.
    pub fn unknown_keys(&self) -> Vec<String> {
        fn collect(map: &Unknown, prefix: &str, out: &mut Vec<String>) {
            for key in map.keys() {
                if prefix.is_empty() {
                    out.push(key.clone());
                } else {
                    out.push(format!("{prefix}.{key}"));
                }
            }
        }

        let mut out = Vec::new();
        collect(&self.unknown, "", &mut out);
        if let Some(matcher) = &self.matcher {
            collect(&matcher.unknown, "match", &mut out);
        }
        collect(&self.hardware.unknown, "hardware", &mut out);
        if let Some(est) = &self.estimation {
            collect(&est.unknown, "estimation", &mut out);
            if let Some(factors) = &est.run_mode_factors {
                collect(&factors.unknown, "estimation.run_mode_factors", &mut out);
            }
        }
        for (i, entry) in self.calibration.iter().enumerate() {
            collect(&entry.unknown, &format!("calibration[{i}]"), &mut out);
        }
        out
    }

    /// Reject a profile whose `name` disagrees with its file name, which would
    /// otherwise be selectable under a name that does not appear in listings.
    pub fn check_name_matches_stem(&self, path: &Path) -> Result<(), String> {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("{} has no readable file name", path.display()))?;
        if stem != self.name {
            return Err(format!(
                "name '{}' does not match file stem '{}'",
                self.name, stem
            ));
        }
        Ok(())
    }

    /// Apply the profile's capacity and unified-memory topology.
    pub fn apply_to_specs(&self, specs: SystemSpecs) -> SystemSpecs {
        specs.with_profile_capacity(self.hardware.total_ram_gb, self.hardware.unified_memory)
    }

    /// Apply the profile's estimator inputs, leaving unset fields untouched so
    /// a partial profile keeps the calculated defaults.
    pub fn apply_to_config(&self, config: &mut CalcConfig) {
        let hw = &self.hardware;
        if let Some(bandwidth) = hw.gpu_memory_bandwidth_gbps {
            config.gpu_bandwidth_gbps_override = Some(bandwidth);
        }
        if let Some(bandwidth) = hw.ddr_bandwidth_gbps {
            config.ddr_bandwidth_gbps = Some(bandwidth);
        }
        if let Some(tflops) = hw.gpu_compute_tflops_fp16 {
            config.gpu_compute_tflops_fp16 = Some(tflops);
        }

        let Some(est) = &self.estimation else {
            return;
        };
        if let Some(efficiency) = est.efficiency {
            config.efficiency = efficiency;
        }
        if let Some(factors) = &est.run_mode_factors {
            let target = &mut config.run_mode_factors;
            if let Some(v) = factors.gpu {
                target.gpu = v;
            }
            if let Some(v) = factors.tensor_parallel {
                target.tensor_parallel = v;
            }
            if let Some(v) = factors.moe_offload {
                target.moe_offload = v;
            }
            if let Some(v) = factors.cpu_offload {
                target.cpu_offload = v;
            }
            if let Some(v) = factors.cpu_only {
                target.cpu_only = v;
            }
        }
    }

    /// Apply both halves at once: specs describe what fits, `config` describes
    /// how fast it runs, and a profile is only coherent if both move together.
    pub fn apply(&self, specs: SystemSpecs, config: &mut CalcConfig) -> SystemSpecs {
        self.apply_to_config(config);
        self.apply_to_specs(specs)
    }
}

fn validate_name(name: &str) -> Result<(), String> {
    let mut chars = name.chars();
    let ok = match chars.next() {
        Some(first) if first.is_ascii_lowercase() || first.is_ascii_digit() => chars
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-')),
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "name '{name}' must start with a lowercase letter or digit and use only [a-z0-9._-]"
        ))
    }
}

fn check_range(label: &str, value: f64, max: f64) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 || value > max {
        return Err(format!(
            "{label} must be a finite number greater than 0 and at most {max}; got {value}"
        ));
    }
    Ok(())
}

/// Profiles embedded at build time. Invalid entries are dropped — the core test
/// suite validates `data/hardware/` against its schema, so a bad file fails CI
/// rather than reaching a release.
pub fn embedded() -> &'static [HardwareProfile] {
    static CACHE: OnceLock<Vec<HardwareProfile>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let raw: Vec<serde_json::Value> =
            serde_json::from_str(EMBEDDED_PROFILES_JSON).unwrap_or_default();
        raw.into_iter()
            .filter_map(|value| serde_json::from_value::<HardwareProfile>(value).ok())
            .filter(|profile| profile.validate().is_ok())
            .collect()
    })
}

/// Directory scanned for user-supplied profiles, alongside the update cache
/// (e.g. `~/.local/share/llmfit/hardware` on Linux).
/// `LLMFIT_HARDWARE_PROFILES` overrides the location.
pub fn user_profile_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LLMFIT_HARDWARE_PROFILES") {
        let trimmed = dir.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    Some(crate::update::cache_dir()?.join("hardware"))
}

/// Read, parse, validate, and name-check one profile file.
pub fn load_file(path: &Path) -> Result<HardwareProfile, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let profile = HardwareProfile::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    profile
        .validate()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    profile
        .check_name_matches_stem(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(profile)
}

/// Every selectable profile, sorted by name.
///
/// A user profile replaces the bundled one with the same name, so a local fix
/// wins without editing the binary's data.
pub fn catalog() -> ProfileCatalog {
    catalog_in(user_profile_dir().as_deref())
}

fn catalog_in(user_dir: Option<&Path>) -> ProfileCatalog {
    let mut catalog = ProfileCatalog::default();
    for profile in embedded() {
        catalog.profiles.push(LoadedProfile {
            origin: ProfileOrigin::Embedded,
            profile: profile.clone(),
        });
    }

    if let Some(entries) = user_dir.and_then(|dir| std::fs::read_dir(dir).ok()) {
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("json"))
            .collect();
        paths.sort();
        for path in paths {
            match load_file(&path) {
                Ok(profile) => {
                    catalog
                        .profiles
                        .retain(|loaded| loaded.profile.name != profile.name);
                    catalog.profiles.push(LoadedProfile {
                        origin: ProfileOrigin::User(path),
                        profile,
                    });
                }
                Err(err) => catalog.errors.push((path, err)),
            }
        }
    }

    catalog
        .profiles
        .sort_by(|a, b| a.profile.name.cmp(&b.profile.name));
    catalog
}

/// Look a profile up by name (user profiles shadow bundled ones).
pub fn find(name: &str) -> Option<LoadedProfile> {
    catalog()
        .profiles
        .into_iter()
        .find(|loaded| loaded.profile.name == name)
}

/// Resolve a `--profile` selector: a path when it looks like one, otherwise a
/// profile name.
pub fn resolve(selector: &str) -> Result<LoadedProfile, String> {
    let selector = selector.trim();
    if selector.is_empty() {
        return Err("profile selector must not be empty".to_string());
    }

    if looks_like_path(selector) {
        let path = PathBuf::from(selector);
        let profile = load_file(&path)?;
        return Ok(LoadedProfile {
            origin: ProfileOrigin::File(path),
            profile,
        });
    }

    find(selector).ok_or_else(|| {
        let names: Vec<String> = catalog()
            .profiles
            .iter()
            .map(|loaded| loaded.profile.name.clone())
            .collect();
        if names.is_empty() {
            format!("unknown hardware profile '{selector}'; no profiles available")
        } else {
            format!(
                "unknown hardware profile '{selector}'. Available: {}",
                names.join(", ")
            )
        }
    })
}

/// Whether a selector names a file rather than a profile.
///
/// Profile names are bare slugs, so anything with a separator, a `.json`
/// extension, or an existing file behind it is a path.
fn looks_like_path(selector: &str) -> bool {
    let path = Path::new(selector);
    selector.ends_with(".json") || path.components().count() > 1 || path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuBackend, GpuInfo};

    const MINIMAL: &str = r#"{
        "schema_version": 1,
        "name": "test-box",
        "hardware": { "total_ram_gb": 64.0, "unified_memory": false }
    }"#;

    fn full_profile() -> HardwareProfile {
        HardwareProfile::parse(
            r#"{
                "schema_version": 1,
                "name": "test-box",
                "match": { "gpu_name_contains": "Radeon 8060S" },
                "hardware": {
                    "total_ram_gb": 128.0,
                    "unified_memory": true,
                    "gpu_memory_bandwidth_gbps": 256.0,
                    "ddr_bandwidth_gbps": 250.0,
                    "gpu_compute_tflops_fp16": 29.7
                },
                "estimation": {
                    "efficiency": 0.7,
                    "run_mode_factors": { "cpu_only": 0.25 }
                },
                "calibration": [
                    { "model": "openai/gpt-oss-120b", "quant": "MXFP4",
                      "run_mode": "gpu", "measured_tps": 50.0, "source": "issue #969" }
                ]
            }"#,
        )
        .expect("full profile parses")
    }

    /// A per-test scratch directory: tests in one binary share a process, so a
    /// fixed path would race between threads.
    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("llmfit-hwprofile-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn specs_no_gpu() -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: false,
            gpu_vram_gb: None,
            total_gpu_vram_gb: None,
            gpu_available_gb: None,
            gpu_name: None,
            gpu_count: 0,
            unified_memory: false,
            backend: GpuBackend::CpuX86,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    fn specs_with_gpu() -> SystemSpecs {
        SystemSpecs {
            has_gpu: true,
            gpu_vram_gb: Some(8.0),
            total_gpu_vram_gb: Some(8.0),
            gpu_name: Some("NVIDIA RTX 3070".to_string()),
            gpu_count: 1,
            backend: GpuBackend::Cuda,
            gpus: vec![GpuInfo {
                name: "NVIDIA RTX 3070".to_string(),
                vram_gb: Some(8.0),
                backend: GpuBackend::Cuda,
                count: 1,
                unified_memory: false,
            }],
            ..specs_no_gpu()
        }
    }

    #[test]
    fn minimal_profile_parses_with_optional_sections_absent() {
        let profile = HardwareProfile::parse(MINIMAL).expect("minimal profile parses");
        assert!(profile.validate_strict().is_ok());
        assert_eq!(profile.hardware.total_ram_gb, 64.0);
        assert!(profile.matcher.is_none());
        assert!(profile.estimation.is_none());
        assert!(profile.calibration.is_empty());
        assert!(profile.hardware.gpu_memory_bandwidth_gbps.is_none());
    }

    #[test]
    fn missing_required_field_is_a_parse_error() {
        let err = HardwareProfile::parse(
            r#"{ "name": "x", "hardware": { "total_ram_gb": 8.0, "unified_memory": false } }"#,
        )
        .expect_err("schema_version is required");
        assert!(err.contains("schema_version"), "{err}");

        let err = HardwareProfile::parse(r#"{ "schema_version": 1, "name": "x" }"#)
            .expect_err("hardware is required");
        assert!(err.contains("hardware"), "{err}");
    }

    #[test]
    fn loader_tolerates_unknown_keys_but_strict_validation_rejects_them() {
        let profile = HardwareProfile::parse(
            r#"{
                "schema_version": 1,
                "name": "test-box",
                "future_top_level": 1,
                "hardware": { "total_ram_gb": 64.0, "unified_memory": false, "npu_tops": 50 },
                "estimation": { "efficiency": 0.5, "future_knob": true },
                "calibration": [{ "model": "m", "measured_tps": 10.0, "future": 1 }]
            }"#,
        )
        .expect("unknown keys must not break loading");

        assert!(profile.validate().is_ok(), "unknown keys are not fatal");
        assert_eq!(
            profile.unknown_keys(),
            vec![
                "future_top_level",
                "hardware.npu_tops",
                "estimation.future_knob",
                "calibration[0].future",
            ]
        );
        let err = profile
            .validate_strict()
            .expect_err("strict validation rejects unknown keys");
        assert!(err.contains("hardware.npu_tops"), "{err}");
    }

    #[test]
    fn wrong_schema_version_is_rejected() {
        let profile = HardwareProfile::parse(
            r#"{ "schema_version": 2, "name": "x",
                 "hardware": { "total_ram_gb": 8.0, "unified_memory": false } }"#,
        )
        .expect("parses");
        let err = profile.validate().expect_err("version 2 is unsupported");
        assert!(err.contains("schema_version 2"), "{err}");
    }

    #[test]
    fn invalid_names_are_rejected() {
        for name in ["", "Ryzen-AI", "-leading-dash", "has space", "has/slash"] {
            assert!(
                validate_name(name).is_err(),
                "'{name}' should be an invalid profile name"
            );
        }
        for name in ["a", "ryzen-ai-max-plus-395", "m3.max_128"] {
            assert!(validate_name(name).is_ok(), "'{name}' should be valid");
        }
    }

    #[test]
    fn non_finite_and_out_of_range_numbers_are_rejected() {
        for ram in ["0", "-1", "1e9"] {
            let text = format!(
                r#"{{ "schema_version": 1, "name": "x",
                      "hardware": {{ "total_ram_gb": {ram}, "unified_memory": false }} }}"#
            );
            let profile = HardwareProfile::parse(&text).expect("parses");
            assert!(
                profile.validate().is_err(),
                "total_ram_gb {ram} should be rejected"
            );
        }

        let profile = HardwareProfile::parse(
            r#"{ "schema_version": 1, "name": "x",
                 "hardware": { "total_ram_gb": 64.0, "unified_memory": false,
                               "gpu_memory_bandwidth_gbps": 0 } }"#,
        )
        .expect("parses");
        let err = profile.validate().expect_err("zero bandwidth is rejected");
        assert!(err.contains("gpu_memory_bandwidth_gbps"), "{err}");
    }

    #[test]
    fn efficiency_above_one_is_rejected() {
        let profile = HardwareProfile::parse(
            r#"{ "schema_version": 1, "name": "x",
                 "hardware": { "total_ram_gb": 64.0, "unified_memory": false },
                 "estimation": { "efficiency": 1.5 } }"#,
        )
        .expect("parses");
        let err = profile.validate().expect_err("efficiency > 1 is rejected");
        assert!(err.contains("estimation.efficiency"), "{err}");
    }

    #[test]
    fn calibration_is_validated_even_though_it_is_not_applied() {
        let profile = HardwareProfile::parse(
            r#"{ "schema_version": 1, "name": "x",
                 "hardware": { "total_ram_gb": 64.0, "unified_memory": false },
                 "calibration": [{ "model": "m", "measured_tps": 10.0,
                                   "run_mode": "teleport" }] }"#,
        )
        .expect("parses");
        let err = profile
            .validate()
            .expect_err("unknown run_mode is rejected");
        assert!(err.contains("calibration[0].run_mode"), "{err}");

        let profile = HardwareProfile::parse(
            r#"{ "schema_version": 1, "name": "x",
                 "hardware": { "total_ram_gb": 64.0, "unified_memory": false },
                 "calibration": [{ "model": "m", "measured_tps": 0 }] }"#,
        )
        .expect("parses");
        assert!(
            profile.validate().is_err(),
            "zero tok/s is not a measurement"
        );
    }

    #[test]
    fn apply_to_config_maps_every_supported_field() {
        let mut config = CalcConfig::default();
        let defaults = CalcConfig::default();
        full_profile().apply_to_config(&mut config);

        assert_eq!(config.gpu_bandwidth_gbps_override, Some(256.0));
        assert_eq!(config.ddr_bandwidth_gbps, Some(250.0));
        assert_eq!(config.gpu_compute_tflops_fp16, Some(29.7));
        assert_eq!(config.efficiency, 0.7);
        assert_eq!(config.run_mode_factors.cpu_only, 0.25);
        // Modes the profile left unset keep their defaults.
        assert_eq!(config.run_mode_factors.gpu, defaults.run_mode_factors.gpu);
        assert_eq!(
            config.run_mode_factors.moe_offload,
            defaults.run_mode_factors.moe_offload
        );
    }

    #[test]
    fn partial_profile_leaves_estimator_defaults_alone() {
        let mut config = CalcConfig::default();
        let defaults = CalcConfig::default();
        HardwareProfile::parse(MINIMAL)
            .expect("parses")
            .apply_to_config(&mut config);

        assert_eq!(config.efficiency, defaults.efficiency);
        assert_eq!(config.gpu_bandwidth_gbps_override, None);
        assert_eq!(config.ddr_bandwidth_gbps, None);
        assert_eq!(config.gpu_compute_tflops_fp16, None);
    }

    #[test]
    fn unified_profile_makes_vram_track_ram_and_synthesizes_a_gpu() {
        let specs = full_profile().apply_to_specs(specs_no_gpu());

        assert!(specs.unified_memory);
        assert!(
            specs.has_gpu,
            "a unified profile must give the GPU path a pool"
        );
        assert_eq!(specs.total_ram_gb, 128.0);
        assert_eq!(specs.gpu_vram_gb, Some(128.0));
        assert_eq!(specs.total_gpu_vram_gb, Some(128.0));
        assert!(specs.gpus.iter().all(|gpu| gpu.unified_memory));
    }

    #[test]
    fn discrete_profile_sets_ram_and_leaves_detected_vram() {
        let profile = HardwareProfile::parse(MINIMAL).expect("parses");
        let specs = profile.apply_to_specs(specs_with_gpu());

        assert!(!specs.unified_memory);
        assert_eq!(specs.total_ram_gb, 64.0);
        assert_eq!(
            specs.gpu_vram_gb,
            Some(8.0),
            "profiles carry no VRAM figure, so detection stands"
        );
    }

    #[test]
    fn apply_moves_specs_and_config_together() {
        let mut config = CalcConfig::default();
        let specs = full_profile().apply(specs_no_gpu(), &mut config);

        assert_eq!(specs.total_ram_gb, 128.0);
        assert_eq!(config.gpu_bandwidth_gbps_override, Some(256.0));
    }

    #[test]
    fn name_must_match_file_stem() {
        let profile = HardwareProfile::parse(MINIMAL).expect("parses");
        assert!(
            profile
                .check_name_matches_stem(Path::new("/tmp/test-box.json"))
                .is_ok()
        );
        let err = profile
            .check_name_matches_stem(Path::new("/tmp/other.json"))
            .expect_err("stem mismatch is rejected");
        assert!(err.contains("does not match file stem"), "{err}");
    }

    #[test]
    fn embedded_profiles_are_present_and_valid() {
        let profiles = embedded();
        assert!(
            profiles.len() >= 2,
            "expected the bundled seed profiles, got {}",
            profiles.len()
        );
        for profile in profiles {
            profile
                .validate_strict()
                .unwrap_or_else(|e| panic!("bundled profile {} is invalid: {e}", profile.name));
        }
        assert!(
            profiles
                .iter()
                .any(|p| p.name == "ryzen-ai-max-plus-395" && p.hardware.unified_memory),
            "expected a 256 GB/s-class unified profile"
        );
        assert!(
            profiles
                .iter()
                .any(|p| p.name == "nvidia-rtx-4090" && !p.hardware.unified_memory),
            "expected a discrete-GPU profile"
        );
    }

    #[test]
    fn resolve_finds_a_bundled_profile_by_name() {
        let loaded = resolve("ryzen-ai-max-plus-395").expect("bundled profile resolves");
        assert_eq!(loaded.profile.name, "ryzen-ai-max-plus-395");
        assert!(loaded.profile.hardware.unified_memory);
    }

    #[test]
    fn user_profiles_shadow_bundled_ones_and_bad_files_are_reported() {
        let dir = temp_dir("catalog");
        std::fs::write(
            dir.join("nvidia-rtx-4090.json"),
            r#"{ "schema_version": 1, "name": "nvidia-rtx-4090",
                 "hardware": { "total_ram_gb": 256.0, "unified_memory": false } }"#,
        )
        .expect("write shadow profile");
        std::fs::write(dir.join("broken.json"), "{ not json").expect("write broken profile");

        let catalog = catalog_in(Some(&dir));

        let shadowed = catalog
            .profiles
            .iter()
            .find(|loaded| loaded.profile.name == "nvidia-rtx-4090")
            .expect("shadowed profile is listed once");
        assert_eq!(shadowed.profile.hardware.total_ram_gb, 256.0);
        assert!(matches!(shadowed.origin, ProfileOrigin::User(_)));
        assert_eq!(
            catalog
                .profiles
                .iter()
                .filter(|loaded| loaded.profile.name == "nvidia-rtx-4090")
                .count(),
            1,
            "a user profile replaces the bundled one rather than duplicating it"
        );
        assert!(
            catalog
                .profiles
                .iter()
                .any(|loaded| loaded.profile.name == "ryzen-ai-max-plus-395"),
            "unshadowed bundled profiles remain"
        );
        assert_eq!(catalog.errors.len(), 1, "the broken file is reported");
        assert!(catalog.errors[0].0.ends_with("broken.json"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_rejects_an_unknown_name_and_lists_alternatives() {
        let err = resolve("no-such-box").expect_err("unknown name is an error");
        assert!(err.contains("unknown hardware profile"), "{err}");
        assert!(err.contains("nvidia-rtx-4090"), "{err}");
    }

    #[test]
    fn resolve_rejects_an_empty_selector() {
        assert!(resolve("   ").is_err());
    }

    #[test]
    fn resolve_reads_a_path_selector() {
        let dir = temp_dir("resolve");
        let path = dir.join("test-box.json");
        std::fs::write(&path, MINIMAL).expect("write profile");

        let loaded = resolve(path.to_str().expect("utf-8 temp path")).expect("path resolves");
        assert_eq!(loaded.origin, ProfileOrigin::File(path.clone()));
        assert_eq!(loaded.profile.name, "test-box");

        let renamed = dir.join("mismatch.json");
        std::fs::write(&renamed, MINIMAL).expect("write profile");
        let err = resolve(renamed.to_str().expect("utf-8 temp path"))
            .expect_err("stem mismatch is rejected on load");
        assert!(err.contains("does not match file stem"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_path_selector_reports_the_path() {
        let err =
            resolve("./definitely-missing-profile.json").expect_err("missing file is an error");
        assert!(err.contains("definitely-missing-profile.json"), "{err}");
    }

    #[test]
    fn bare_names_are_not_treated_as_paths() {
        assert!(!looks_like_path("ryzen-ai-max-plus-395"));
        assert!(looks_like_path("profile.json"));
        assert!(looks_like_path("./dir/profile"));
    }
}
