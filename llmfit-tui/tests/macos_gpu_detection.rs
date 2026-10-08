//! Exercise the actual profiler subprocess flow without changing process-wide PATH.
#![cfg(target_os = "macos")]

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

const PROFILER: &str = r#"#!/bin/sh
printf '%s\n' "$*" >> "$LLMFIT_TEST_PROBE_LOG"
case "$*" in
  'SPDisplaysDataType -json')
    case "$LLMFIT_TEST_JSON_MODE" in
      fail) exit 1 ;;
      invalid) printf '{invalid json'; exit 0 ;;
      *) /bin/cat "$LLMFIT_TEST_JSON_FIXTURE" ;;
    esac ;;
  'SPDisplaysDataType')
    case "$LLMFIT_TEST_TEXT_MODE" in
      fail) exit 1 ;;
      non-utf8) printf '\377'; exit 0 ;;
      unmatched) printf 'Chipset Model: Intel HD Graphics 630\n'; exit 0 ;;
      *) /bin/cat "$LLMFIT_TEST_TEXT_FIXTURE" ;;
    esac ;;
  *) exit 1 ;;
esac
"#;

fn detect(json_fixture: &str, json_mode: &str, text_mode: &str) -> (Value, String) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let sequence = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "llmfit-profiler-{}-{nonce}-{sequence}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("create profiler directory");
    let profiler = dir.join("system_profiler");
    fs::write(&profiler, PROFILER).expect("write profiler stub");
    fs::set_permissions(&profiler, fs::Permissions::from_mode(0o755))
        .expect("make profiler executable");
    let fixtures =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../llmfit-core/tests/fixtures/hardware/macos");
    let log = dir.join("calls.txt");
    let output = Command::cargo_bin("llmfit")
        .expect("find llmfit test binary")
        // Only the controlled profiler is discoverable; host GPU utilities
        // cannot hide a missing GPU result from the code under test.
        .env("PATH", &dir)
        .env("LLMFIT_TEST_PROBE_LOG", &log)
        .env("LLMFIT_TEST_JSON_FIXTURE", fixtures.join(json_fixture))
        .env(
            "LLMFIT_TEST_TEXT_FIXTURE",
            fixtures.join("apple-a18-pro.txt"),
        )
        .env("LLMFIT_TEST_JSON_MODE", json_mode)
        .env("LLMFIT_TEST_TEXT_MODE", text_mode)
        .args(["--no-dashboard", "system", "--json"])
        .output()
        .expect("run hardware detection");
    let calls = fs::read_to_string(log).expect("read profiler calls");
    fs::remove_dir_all(dir).expect("remove profiler directory");
    assert!(output.status.success(), "{:?}", output);
    let report: Value = serde_json::from_slice(&output.stdout).expect("valid system JSON");
    (report["system"].clone(), calls)
}

fn assert_unified_gpu(system: &Value) {
    assert_eq!(system["has_gpu"], true, "{system}");
    assert_eq!(system["backend"], "Metal");
    assert_eq!(system["unified_memory"], true);
    assert_eq!(system["gpu_count"], 1);
    assert_eq!(system["gpu_vram_gb"], system["total_ram_gb"]);
    let gpus = system["gpus"].as_array().expect("GPU array");
    assert_eq!(gpus.len(), 1);
    assert_eq!(gpus[0]["unified_memory"], true);
    assert_eq!(gpus[0]["vram_gb"], system["total_ram_gb"]);
}

#[test]
fn json_apple_gpu_survives_failed_text_probe() {
    for (fixture, name) in [
        ("apple-a18-pro.json", "Apple A18 Pro"),
        ("apple-silicon.json", "Apple M2"),
    ] {
        for failure in ["fail", "non-utf8", "unmatched"] {
            let (system, calls) = detect(fixture, "success", failure);
            assert_unified_gpu(&system);
            assert_eq!(system["gpu_name"], name);
            assert_eq!(calls, "SPDisplaysDataType -json\n");
        }
    }
}

#[test]
fn json_apple_gpu_skips_successful_text_probe() {
    let (system, calls) = detect("apple-a18-pro.json", "success", "success");
    assert_unified_gpu(&system);
    assert_eq!(system["gpu_name"], "Apple A18 Pro");
    assert_eq!(calls, "SPDisplaysDataType -json\n");
}

#[test]
fn text_probe_recovers_when_json_fails() {
    for failure in ["fail", "invalid"] {
        let (system, calls) = detect("apple-a18-pro.json", failure, "success");
        assert_unified_gpu(&system);
        assert_eq!(calls, "SPDisplaysDataType -json\nSPDisplaysDataType\n");
    }
}

#[test]
fn failed_probes_do_not_invent_a_gpu() {
    let (system, _) = detect("apple-a18-pro.json", "fail", "fail");
    assert_eq!(system["has_gpu"], false);
    assert_eq!(system["gpu_count"], 0);
}
