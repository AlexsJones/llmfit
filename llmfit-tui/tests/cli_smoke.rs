use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn run_json_command(args: &[&str]) -> Value {
    let output = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .env_remove("LLAMA_CPP_PATH")
        .args(args)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    serde_json::from_slice(&output).expect("command did not emit valid JSON")
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("llmfit-{name}-{}-{nanos}", std::process::id()))
}

fn create_fake_llama_cpp_bin_dir(name: &str) -> PathBuf {
    let dir = unique_temp_dir(name);
    fs::create_dir_all(&dir).expect("failed to create fake llama.cpp bin dir");
    for binary in ["llama-cli", "llama-server"] {
        let path = dir.join(binary);
        fs::write(&path, "#!/bin/sh\nexit 0\n").expect("failed to write fake llama.cpp binary");
        make_executable(&path);
    }
    dir
}

fn make_executable(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path)
            .expect("failed to stat fake llama.cpp binary")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)
            .expect("failed to mark fake llama.cpp binary executable");
    }

    #[cfg(not(unix))]
    let _ = path;
}

#[test]
fn help_includes_project_description() {
    let output = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .arg("--help")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let text = String::from_utf8(output).expect("--help output was not UTF-8");
    assert!(text.contains("Right-size LLM models to your system's hardware"));
}

#[test]
fn version_matches_package_version() {
    let output = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .arg("--version")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let text = String::from_utf8(output).expect("--version output was not UTF-8");
    assert!(text.contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn system_json_has_expected_shape() {
    let json = run_json_command(&["--no-dashboard", "--json", "system"]);
    let system = json
        .get("system")
        .and_then(Value::as_object)
        .expect("system key missing or not an object");

    assert!(system.contains_key("available_ram_gb"));
    assert!(system.contains_key("cpu_cores"));
    assert!(system.contains_key("backend"));
}

#[test]
fn llama_cpp_path_flag_makes_provider_available() {
    let dir = create_fake_llama_cpp_bin_dir("llama-cpp-path");
    let dir_str = dir.to_str().expect("temp dir path was not UTF-8");

    let json = run_json_command(&[
        "--no-dashboard",
        "--llama-cpp-path",
        dir_str,
        "--json",
        "system",
    ]);
    let llama_cpp = json
        .pointer("/providers/llama.cpp")
        .and_then(Value::as_object)
        .expect("llama.cpp provider status missing");

    assert_eq!(
        llama_cpp.get("available").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        llama_cpp.get("llama_cli_path").and_then(Value::as_str),
        Some(
            dir.join("llama-cli")
                .to_str()
                .expect("binary path was not UTF-8")
        )
    );
    assert_eq!(
        llama_cpp.get("llama_server_path").and_then(Value::as_str),
        Some(
            dir.join("llama-server")
                .to_str()
                .expect("binary path was not UTF-8")
        )
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn llama_cpp_path_flag_ignores_missing_directory() {
    let missing = unique_temp_dir("missing-llama-cpp-path");
    let missing_str = missing.to_str().expect("temp dir path was not UTF-8");

    let json = run_json_command(&[
        "--no-dashboard",
        "--llama-cpp-path",
        missing_str,
        "--json",
        "system",
    ]);

    assert!(
        json.get("system").is_some(),
        "system output should be present"
    );
}

#[test]
fn llama_cpp_path_flag_overrides_env_var() {
    let env_dir = create_fake_llama_cpp_bin_dir("llama-cpp-env");
    let flag_dir = create_fake_llama_cpp_bin_dir("llama-cpp-flag");
    let flag_dir_str = flag_dir.to_str().expect("temp dir path was not UTF-8");

    let output = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .env("LLAMA_CPP_PATH", &env_dir)
        .args([
            "--no-dashboard",
            "--llama-cpp-path",
            flag_dir_str,
            "--json",
            "system",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&output).expect("command did not emit valid JSON");
    let llama_cpp = json
        .pointer("/providers/llama.cpp")
        .and_then(Value::as_object)
        .expect("llama.cpp provider status missing");

    assert_eq!(
        llama_cpp.get("llama_cli_path").and_then(Value::as_str),
        Some(
            flag_dir
                .join("llama-cli")
                .to_str()
                .expect("binary path was not UTF-8")
        )
    );

    let _ = fs::remove_dir_all(env_dir);
    let _ = fs::remove_dir_all(flag_dir);
}

#[test]
fn llama_cpp_path_flag_works_with_help() {
    Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .args(["--llama-cpp-path", "/tmp/x", "--help"])
        .assert()
        .success();
}

#[test]
fn list_json_returns_non_empty_catalog() {
    let json = run_json_command(&["--no-dashboard", "--json", "list"]);
    let models = json
        .as_array()
        .expect("list --json output should be an array");

    assert!(!models.is_empty(), "model catalog should not be empty");
    let first = models[0]
        .as_object()
        .expect("first model entry should be a JSON object");
    assert!(first.contains_key("name"));
    assert!(first.contains_key("provider"));
}

#[test]
fn fit_json_obeys_limit_and_contains_models_field() {
    let json = run_json_command(&[
        "--no-dashboard",
        "--json",
        "--memory",
        "8G",
        "--ram",
        "16G",
        "--cpu-cores",
        "4",
        "fit",
        "--limit",
        "3",
    ]);

    let models = json
        .get("models")
        .and_then(Value::as_array)
        .expect("fit --json output missing models array");

    assert!(models.len() <= 3, "fit output exceeded requested limit");

    if let Some(first) = models.first() {
        let first = first
            .as_object()
            .expect("fit model entry should be a JSON object");
        assert!(first.contains_key("fit_level"));
        assert!(first.contains_key("run_mode"));
        assert!(first.contains_key("score"));
    }
}

#[test]
fn fit_json_returns_empty_models_when_no_perfect_matches() {
    let json = run_json_command(&[
        "--no-dashboard",
        "--json",
        "--memory",
        "1M",
        "--ram",
        "1M",
        "--cpu-cores",
        "1",
        "fit",
        "--perfect",
    ]);

    let models = json
        .get("models")
        .and_then(Value::as_array)
        .expect("fit --json output missing models array");

    assert!(
        models.is_empty(),
        "expected no perfect matches on extremely constrained hardware"
    );
}

#[test]
fn cpu_cores_parser_rejects_zero() {
    Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .args(["--cpu-cores", "0", "--json", "system"])
        .assert()
        .failure();
}
