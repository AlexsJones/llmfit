use assert_cmd::Command;
use serde_json::Value;

#[test]
fn list_json_outputs_valid_json_array() {
    let output = Command::cargo_bin("llmfit")
        .expect("binary should build")
        .args(["list", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8(output).expect("stdout should be valid UTF-8");
    assert!(
        !stdout.contains("=== Available LLM Models ==="),
        "expected JSON output, got table header: {stdout}"
    );
    assert!(
        !stdout.contains("╭") && !stdout.contains("│ Status │"),
        "expected JSON output, got table rendering: {stdout}"
    );

    let parsed: Value = serde_json::from_str(&stdout).expect("stdout should parse as JSON");
    let models = parsed
        .as_array()
        .expect("list --json should output a JSON array");
    assert!(
        !models.is_empty(),
        "expected at least one model in JSON output"
    );

    let first = models[0]
        .as_object()
        .expect("each model should be a JSON object");
    assert!(
        first.contains_key("name"),
        "model JSON should include 'name'"
    );
    assert!(
        first.contains_key("provider"),
        "model JSON should include 'provider'"
    );
}
