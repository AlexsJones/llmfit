//! Exercise the public CLI with a small custom catalog and isolated profiles.
use assert_cmd::Command;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "llmfit-storage-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).expect("fixture directory");
        let models: Vec<Value> = [("small", 8), ("medium", 14), ("large", 20), ("huge", 3000), ("EAGLE", 8)]
            .into_iter().map(|(name, params)| json!({
                "name": format!("storage970/{name}-{params}B"), "provider": "Storage970",
                "parameter_count": format!("{params}B"), "parameters_raw": params as u64 * 1_000_000_000,
                "min_ram_gb": f64::from(params) * 0.58 + 0.5,
                "min_vram_gb": f64::from(params) * 0.58 + 0.5,
                "recommended_ram_gb": f64::from(params) * 2.0,
                "quantization": "Q4_K_M", "context_length": 32768, "use_case": "general"
            })).collect();
        std::fs::write(
            dir.join("models.json"),
            serde_json::to_vec(&models).expect("catalog JSON"),
        )
        .expect("catalog");
        std::fs::create_dir(dir.join("profiles")).expect("profile directory");
        for (name, unified, bandwidth) in [
            ("unified", true, 777.0),
            ("unified-fast", true, 1554.0),
            ("discrete", false, 777.0),
        ] {
            std::fs::write(
                dir.join("profiles").join(format!("{name}.json")),
                serde_json::to_vec(&json!({
                    "schema_version": 1, "name": name,
                    "hardware": {"total_ram_gb": 128.0, "unified_memory": unified,
                        "gpu_memory_bandwidth_gbps": bandwidth},
                    "estimation": {"efficiency": 0.5}
                }))
                .expect("profile JSON"),
            )
            .expect("profile");
        }
        Self { dir }
    }

    fn command(&self) -> Command {
        let mut cmd = Command::cargo_bin("llmfit").expect("test binary");
        cmd.env("LLMFIT_CUSTOM_MODELS", self.dir.join("models.json"))
            .env("LLMFIT_HARDWARE_PROFILES", self.dir.join("profiles"))
            .env_remove("OLLAMA_CONTEXT_LENGTH")
            .env("NO_COLOR", "1");
        cmd
    }

    fn json(&self, args: &[&str]) -> Value {
        let output = self
            .command()
            .args(args)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        serde_json::from_slice(&output).expect("one JSON report")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn number(value: &Value, field: &str) -> f64 {
    value[field]
        .as_f64()
        .unwrap_or_else(|| panic!("missing number {field}"))
}

#[test]
fn storage_accepts_the_complete_catalog() {
    let fixture = Fixture::new();
    for args in [
        vec!["storage", "--json"],
        vec!["--profile", "unified", "storage", "--json"],
    ] {
        let report = fixture.json(&args);
        let storage = &report["storage"];
        let models = storage["models"].as_array().expect("models");
        assert!(models.len() <= 3);
        assert_eq!(storage["selected_count"], models.len());
        assert!(number(storage, "library_gb").is_finite());
        if args.contains(&"--profile") {
            assert_eq!(models.len(), 3, "roomy profile has runnable models");
        }
    }
}

#[test]
fn storage_help_documents_the_contract() {
    let output = Command::cargo_bin("llmfit")
        .expect("binary")
        .args(["storage", "--help"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("help text");
    for option in [
        "--keep",
        "--selection",
        "--os-reserve",
        "--scratch",
        "--headroom",
        "--perfect",
        "--search",
        "--json",
    ] {
        assert!(text.contains(option), "{option}");
    }
}

#[test]
fn storage_uses_the_shared_fits_and_reports_reproducible_totals() {
    let fixture = Fixture::new();
    let fits = fixture.json(&[
        "--profile",
        "unified",
        "--max-context",
        "2048",
        "fit",
        "--providers",
        "Storage970",
        "--json",
    ]);
    let report = fixture.json(&[
        "--profile",
        "unified",
        "--max-context",
        "2048",
        "storage",
        "--search",
        "storage970",
        "--json",
    ]);
    let storage = &report["storage"];
    assert_eq!(storage["selected_count"], 3);
    assert_eq!(storage["eligible_count"], 3);
    assert_eq!(storage["selection"], "score");
    assert_eq!(report["system"]["total_ram_gb"], 128.0);
    assert_eq!(report["system"]["unified_memory"], true);
    let models = storage["models"].as_array().expect("models");
    for selected in models {
        let original = fits["models"]
            .as_array()
            .expect("fits")
            .iter()
            .find(|fit| fit["name"] == selected["name"])
            .expect("same fit");
        assert_eq!(selected["best_quant"], original["best_quant"]);
        assert_eq!(selected["fit_level"], original["fit_level"]);
        assert_eq!(selected["effective_context_length"], 2048);
        assert!((number(selected, "score") - number(original, "score")).abs() <= 0.051);
        assert!(
            (number(selected, "disk_size_gb") - number(original, "disk_size_gb")).abs() <= 0.005
        );
    }
    let sum: f64 = models.iter().map(|m| number(m, "disk_size_gb")).sum();
    let max = models
        .iter()
        .map(|m| number(m, "disk_size_gb"))
        .fold(0.0, f64::max);
    assert!((number(storage, "library_gb") - sum).abs() < 1e-9);
    assert!((number(storage, "download_scratch_gb") - max).abs() < 1e-9);
    assert!((number(storage, "need_gb") - (100.0 + sum + max)).abs() < 1e-9);
    assert!(number(storage, "suggested_ssd_gb") * 0.85 >= number(storage, "need_gb"));

    let second = fixture.json(&[
        "--profile",
        "unified",
        "--max-context",
        "2048",
        "storage",
        "--search",
        "storage970",
        "--json",
    ]);
    assert_eq!(storage, &second["storage"]);
}

#[test]
fn storage_applies_largest_perfect_search_and_custom_allowances() {
    let fixture = Fixture::new();
    let report = fixture.json(&[
        "--profile",
        "unified",
        "storage",
        "--search",
        "StOrAgE970",
        "--selection",
        "largest",
        "--perfect",
        "--keep",
        "2",
        "--os-reserve",
        "1TB",
        "--scratch",
        "1GiB",
        "--headroom",
        "0",
        "--json",
    ]);
    let storage = &report["storage"];
    assert_eq!(storage["selected_count"], 2);
    assert_eq!(storage["selection"], "largest");
    let models = storage["models"].as_array().expect("models");
    assert!(models.iter().all(|model| model["fit_level"] == "Perfect"));
    assert!(number(&models[0], "disk_size_gb") >= number(&models[1], "disk_size_gb"));
    assert_eq!(storage["os_reserve_gb"], 1000.0);
    assert!((number(storage, "download_scratch_gb") - 1.073741824).abs() < 1e-9);
    assert_eq!(storage["need_gb"], storage["target_capacity_gb"]);
}

#[test]
fn storage_handles_partial_empty_and_exhausted_tiers() {
    let fixture = Fixture::new();
    let partial = fixture.json(&[
        "--profile",
        "unified",
        "storage",
        "--search",
        "storage970",
        "--keep",
        "20",
        "--json",
    ]);
    assert_eq!(partial["storage"]["selected_count"], 3);
    assert_eq!(partial["storage"]["keep_requested"], 20);
    assert!(
        !partial["storage"]["warnings"]
            .as_array()
            .expect("warnings")
            .is_empty()
    );
    let empty = fixture.json(&[
        "--profile",
        "unified",
        "storage",
        "--search",
        "storage970/no-match",
        "--json",
    ]);
    assert_eq!(empty["storage"]["selected_count"], 0);
    assert_eq!(empty["storage"]["need_gb"], 100.0);
    assert!(empty["storage"]["suggested_ssd_gb"].is_null());
    let huge = fixture.json(&[
        "--profile",
        "unified",
        "storage",
        "--search",
        "storage970",
        "--os-reserve",
        "20TB",
        "--json",
    ]);
    assert!(huge["storage"]["minimum_ssd_gb"].is_null());
    assert!(huge["storage"]["suggested_ssd_gb"].is_null());
}

#[test]
fn storage_validates_options_before_planning() {
    let fixture = Fixture::new();
    for args in [
        ["--keep", "0"],
        ["--headroom", "100"],
        ["--selection", "invalid"],
    ] {
        fixture
            .command()
            .args(["storage", args[0], args[1]])
            .assert()
            .code(2);
    }
    for args in [
        ["--os-reserve", "NaN"],
        ["--scratch", "-1G"],
        ["--search", " "],
    ] {
        let output = fixture
            .command()
            .args(["storage", "--json", &format!("{}={}", args[0], args[1])])
            .assert()
            .code(1)
            .get_output()
            .stdout
            .clone();
        let json: Value = serde_json::from_slice(&output).expect("JSON error");
        assert!(json["error"]["message"].is_string());
    }
    fixture
        .command()
        .args(["storage", "--csv"])
        .assert()
        .code(1);
    fixture
        .command()
        .args(["storage", "--memory", "128G"])
        .assert()
        .code(2);
    fixture
        .command()
        .args(["--profile", "unified", "--memory", "128G", "storage"])
        .assert()
        .code(2);
}

#[test]
fn storage_supports_text_and_legacy_hardware_overrides() {
    let fixture = Fixture::new();
    let report = fixture.json(&[
        "--memory",
        "128G",
        "--ram",
        "128G",
        "--cpu-cores",
        "18",
        "storage",
        "--search",
        "storage970",
        "--json",
    ]);
    assert_eq!(report["system"]["cpu_cores"], 18);
    assert_eq!(report["system"]["total_ram_gb"], 128.0);
    let output = fixture
        .command()
        .args(["--profile", "unified", "storage", "--search", "storage970"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("text");
    for label in [
        "Library",
        "Scratch",
        "OS/apps reserve",
        "Required",
        "Suggested SSD",
        "storage970/",
    ] {
        assert!(text.contains(label), "missing {label}");
    }
}

#[test]
fn storage_matches_profile_config_and_environment_context() {
    let fixture = Fixture::new();
    for profile in ["discrete", "unified-fast"] {
        let fits = fixture.json(&[
            "--profile",
            profile,
            "--max-context",
            "1024",
            "fit",
            "--providers",
            "Storage970",
            "--json",
        ]);
        let output = fixture
            .command()
            .env("OLLAMA_CONTEXT_LENGTH", "1024")
            .args([
                "--profile",
                profile,
                "storage",
                "--search",
                "storage970",
                "--json",
            ])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let report: Value = serde_json::from_slice(&output).expect("JSON");
        for model in report["storage"]["models"].as_array().expect("models") {
            let original = fits["models"]
                .as_array()
                .expect("fits")
                .iter()
                .find(|m| m["name"] == model["name"])
                .expect("fit");
            assert_eq!(model["best_quant"], original["best_quant"]);
            assert!((number(model, "score") - number(original, "score")).abs() <= 0.051);
            assert_eq!(model["effective_context_length"], 1024);
        }
    }
}
