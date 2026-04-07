use assert_cmd::Command;

#[test]
fn list_json_outputs_parseable_json_instead_of_table() {
    let assert = Command::cargo_bin("llmfit")
        .expect("llmfit binary should build for integration test")
        .args(["list", "--json"])
        .assert()
        .success();

    let output = assert.get_output();
    let stdout = String::from_utf8(output.stdout.clone()).expect("stdout should be valid UTF-8");

    assert!(
        !stdout.contains("=== Available LLM Models ==="),
        "list --json should not render the table header"
    );
    assert!(
        !stdout.contains("╭────────"),
        "list --json should not render table borders"
    );

    let models: serde_json::Value =
        serde_json::from_str(&stdout).expect("list --json should emit valid JSON");
    let models = models
        .as_array()
        .expect("list --json should emit a top-level JSON array");

    assert!(
        !models.is_empty(),
        "model database should not be empty in list --json output"
    );
    assert!(
        models
            .iter()
            .all(|model| model.get("name").and_then(|name| name.as_str()).is_some()),
        "every JSON model entry should include a string name"
    );
}
