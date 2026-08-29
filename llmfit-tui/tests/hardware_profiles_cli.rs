//! `llmfit hardware` and the global `--profile` flag.
//!
//! Every test points `LLMFIT_HARDWARE_PROFILES` at a scratch directory so a
//! developer's own profiles can't change the result.

use assert_cmd::Command;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// A profile file with a bandwidth no real GPU reports, so an assertion can
/// prove the value came from the profile and not from detection.
const UNIFIED_PROFILE: &str = r#"{
  "schema_version": 1,
  "name": "test-unified",
  "hardware": {
    "total_ram_gb": 128.0,
    "unified_memory": true,
    "gpu_memory_bandwidth_gbps": 777.0,
    "gpu_compute_tflops_fp16": 42.0
  },
  "estimation": { "efficiency": 0.5 },
  "calibration": [{ "model": "openai/gpt-oss-120b", "measured_tps": 50.0 }]
}"#;

fn scratch_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("llmfit-cli-hwprofile-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn llmfit(profiles_dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("llmfit").expect("failed to locate llmfit test binary");
    cmd.env("LLMFIT_HARDWARE_PROFILES", profiles_dir);
    cmd
}

fn json_of(output: Vec<u8>) -> Value {
    serde_json::from_slice(&output).expect("command did not emit valid JSON")
}

#[test]
fn hardware_list_json_reports_bundled_profiles() {
    let dir = scratch_dir("list");
    let output = llmfit(&dir)
        .args(["--json", "hardware", "list"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = json_of(output);
    let profiles = json
        .get("profiles")
        .and_then(Value::as_array)
        .expect("list --json missing profiles array");
    assert!(
        profiles.len() >= 2,
        "expected the bundled seed profiles, got {}",
        profiles.len()
    );
    assert!(
        json.get("errors")
            .and_then(Value::as_array)
            .is_some_and(|e| e.is_empty()),
        "an empty profile directory should report no errors"
    );

    let unified = profiles
        .iter()
        .find(|p| p.get("name").and_then(Value::as_str) == Some("ryzen-ai-max-plus-395"))
        .expect("the 256 GB/s-class unified seed profile should be listed");
    assert_eq!(
        unified
            .get("gpu_memory_bandwidth_gbps")
            .and_then(Value::as_f64),
        Some(256.0)
    );
    assert_eq!(
        unified.get("unified_memory").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        unified.get("origin").and_then(Value::as_str),
        Some("bundled")
    );

    assert!(
        profiles
            .iter()
            .any(|p| p.get("unified_memory").and_then(Value::as_bool) == Some(false)),
        "expected a discrete-GPU seed profile too"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hardware_list_reports_unloadable_user_files_instead_of_hiding_them() {
    let dir = scratch_dir("list-errors");
    std::fs::write(dir.join("broken.json"), "{ not json").expect("write broken profile");

    let output = llmfit(&dir)
        .args(["--json", "hardware", "list"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = json_of(output);
    let errors = json
        .get("errors")
        .and_then(Value::as_array)
        .expect("list --json missing errors array");
    assert_eq!(errors.len(), 1, "the broken file should be reported");
    assert!(
        errors[0]
            .get("path")
            .and_then(Value::as_str)
            .is_some_and(|p| p.ends_with("broken.json"))
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hardware_show_json_reports_effective_bandwidth_and_unapplied_calibration() {
    let dir = scratch_dir("show");
    std::fs::write(dir.join("test-unified.json"), UNIFIED_PROFILE).expect("write profile");

    let output = llmfit(&dir)
        .args(["--json", "hardware", "show", "test-unified"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = json_of(output);
    let effective = json
        .get("effective")
        .and_then(Value::as_object)
        .expect("show --json missing effective object");
    assert_eq!(
        effective.get("gpu_bandwidth_gbps").and_then(Value::as_f64),
        Some(777.0),
        "the profile's bandwidth must win over the detected GPU"
    );
    assert_eq!(
        effective.get("total_ram_gb").and_then(Value::as_f64),
        Some(128.0)
    );
    assert_eq!(
        effective.get("unified_memory").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        effective.get("efficiency").and_then(Value::as_f64),
        Some(0.5)
    );
    assert_eq!(
        json.get("calibration_applied").and_then(Value::as_bool),
        Some(false),
        "schema version 1 records calibration without applying it"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hardware_show_rejects_an_unknown_profile() {
    let dir = scratch_dir("show-unknown");
    llmfit(&dir)
        .args(["hardware", "show", "no-such-profile"])
        .assert()
        .failure();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hardware_validate_rejects_unknown_keys_and_stem_mismatch() {
    let dir = scratch_dir("validate");
    let good = dir.join("test-unified.json");
    std::fs::write(&good, UNIFIED_PROFILE).expect("write profile");
    let typo = dir.join("typo.json");
    std::fs::write(
        &typo,
        r#"{ "schema_version": 1, "name": "typo",
             "hardware": { "total_ram_gb": 64.0, "unified_memory": false,
                           "gpu_bandwith_gbps": 900 } }"#,
    )
    .expect("write typo profile");
    let mismatch = dir.join("wrong-stem.json");
    std::fs::write(&mismatch, UNIFIED_PROFILE).expect("write mismatched profile");

    llmfit(&dir)
        .args(["hardware", "validate"])
        .arg(&good)
        .assert()
        .success();

    let output = llmfit(&dir)
        .args(["--json", "hardware", "validate"])
        .arg(&good)
        .arg(&typo)
        .arg(&mismatch)
        .assert()
        .failure()
        .get_output()
        .stdout
        .clone();

    let json = json_of(output);
    assert_eq!(json.get("ok").and_then(Value::as_bool), Some(false));
    let results = json
        .get("results")
        .and_then(Value::as_array)
        .expect("validate --json missing results array");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get("ok").and_then(Value::as_bool), Some(true));
    assert!(
        results[1]
            .get("error")
            .and_then(Value::as_str)
            .is_some_and(|e| e.contains("gpu_bandwith_gbps")),
        "a misspelled key must be named in the error: {:?}",
        results[1]
    );
    assert!(
        results[2]
            .get("error")
            .and_then(Value::as_str)
            .is_some_and(|e| e.contains("file stem")),
        "a name/stem mismatch must be rejected: {:?}",
        results[2]
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hardware_validate_requires_a_path() {
    let dir = scratch_dir("validate-noargs");
    llmfit(&dir)
        .args(["hardware", "validate"])
        .assert()
        .failure();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hardware_path_json_reports_the_user_directory() {
    let dir = scratch_dir("path");
    let output = llmfit(&dir)
        .args(["--json", "hardware", "path"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = json_of(output);
    assert_eq!(
        json.get("path").and_then(Value::as_str),
        Some(dir.to_str().expect("utf-8 scratch path"))
    );
    assert_eq!(json.get("exists").and_then(Value::as_bool), Some(true));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn profile_conflicts_with_the_single_field_overrides() {
    let dir = scratch_dir("conflicts");
    for conflicting in [
        vec!["--memory", "8G"],
        vec!["--ram", "16G"],
        vec!["--cpu-cores", "4"],
    ] {
        llmfit(&dir)
            .args(["--profile", "nvidia-rtx-4090"])
            .args(&conflicting)
            .args(["--json", "system"])
            .assert()
            .failure();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn profile_replaces_detected_hardware_in_the_fit_sweep() {
    let dir = scratch_dir("fit");
    std::fs::write(dir.join("test-unified.json"), UNIFIED_PROFILE).expect("write profile");

    let output = llmfit(&dir)
        .args([
            "--no-dashboard",
            "--json",
            "--profile",
            "test-unified",
            "fit",
            "--limit",
            "10",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = json_of(output);
    assert_eq!(
        json.pointer("/system/total_ram_gb").and_then(Value::as_f64),
        Some(128.0),
        "the profile's capacity should replace the detected pool"
    );
    assert_eq!(
        json.pointer("/system/unified_memory")
            .and_then(Value::as_bool),
        Some(true)
    );

    let models = json
        .get("models")
        .and_then(Value::as_array)
        .expect("fit --json missing models array");
    assert!(
        !models.is_empty(),
        "a 128 GB machine should fit some models"
    );

    let bandwidths: Vec<f64> = models
        .iter()
        .filter_map(|m| m.pointer("/estimate_basis/gpu_bandwidth_gbps"))
        .filter_map(Value::as_f64)
        .collect();
    assert!(
        !bandwidths.is_empty(),
        "a unified profile should put at least one model on the GPU path"
    );
    assert!(
        bandwidths.iter().all(|bw| *bw == 777.0),
        "every GPU-path estimate should use the profile bandwidth, got {bandwidths:?}"
    );
    assert!(
        models
            .iter()
            .filter_map(|m| m.pointer("/estimate_basis/efficiency"))
            .filter_map(Value::as_f64)
            .all(|eff| eff == 0.5),
        "the profile's efficiency should reach the estimator"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn profile_reads_a_path_selector() {
    let dir = scratch_dir("fit-path");
    let path = dir.join("test-unified.json");
    std::fs::write(&path, UNIFIED_PROFILE).expect("write profile");

    // Point the user directory elsewhere: the profile must be found through the
    // path alone, not through name lookup.
    let empty = scratch_dir("fit-path-empty");
    let output = llmfit(&empty)
        .args(["--no-dashboard", "--json", "--profile"])
        .arg(&path)
        .args(["fit", "--limit", "1"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = json_of(output);
    assert_eq!(
        json.pointer("/system/total_ram_gb").and_then(Value::as_f64),
        Some(128.0)
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&empty);
}

#[test]
fn unknown_profile_fails_instead_of_falling_back_to_detection() {
    let dir = scratch_dir("unknown");
    llmfit(&dir)
        .args([
            "--no-dashboard",
            "--json",
            "--profile",
            "no-such-profile",
            "fit",
        ])
        .assert()
        .failure();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn profile_with_force_runtime_is_refused_rather_than_half_applied() {
    let dir = scratch_dir("force-runtime");
    std::fs::write(dir.join("test-unified.json"), UNIFIED_PROFILE).expect("write profile");

    llmfit(&dir)
        .args([
            "--no-dashboard",
            "--json",
            "--profile",
            "test-unified",
            "recommend",
            "--force-runtime",
            "llamacpp",
        ])
        .assert()
        .failure();

    let _ = std::fs::remove_dir_all(&dir);
}
