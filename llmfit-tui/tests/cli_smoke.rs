use assert_cmd::Command;
use serde_json::Value;

fn run_json_command(args: &[&str]) -> Value {
    let output = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .args(args)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    serde_json::from_slice(&output).expect("command did not emit valid JSON")
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
