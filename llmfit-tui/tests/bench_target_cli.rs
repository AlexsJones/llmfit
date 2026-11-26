//! What `llmfit bench` says when it cannot settle on a target. The speed
//! bench, the quality bench and the tool-call probe all resolve `--provider`
//! into a target, so each provider has to fail the same way on all of them.

use assert_cmd::Command;

/// A closed local port: every provider fails before any request is answered.
const DEAD: &str = "http://127.0.0.1:9";

/// Runs `llmfit bench` with `args`, expects exit 1, and returns stderr.
fn bench_fails(args: &[&str]) -> String {
    let out = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .env_remove("LLAMA_CPP_PATH")
        .env_remove("OLLAMA_HOST")
        .arg("--no-dashboard")
        .arg("bench")
        .args(args)
        .assert()
        .code(1)
        .get_output()
        .stderr
        .clone();
    String::from_utf8_lossy(&out).into_owned()
}

#[test]
fn the_speed_bench_names_the_provider_that_needs_a_model() {
    for (provider, message) in [
        ("ollama", "Error: model name required for ollama bench"),
        ("mlx", "Error: model name required for mlx bench"),
    ] {
        let stderr = bench_fails(&["--json", "--provider", provider, "--url", DEAD]);
        assert!(stderr.contains(message), "{provider}: {stderr}");
    }
}

#[test]
fn the_speed_bench_reports_an_unreachable_provider() {
    for (provider, message) in [
        ("vllm", "Error: Cannot read http://127.0.0.1:9/v1/models"),
        ("ferrum", "Error: Cannot read http://127.0.0.1:9/v1/models"),
        (
            "lmstudio",
            "Error: Endpoint at http://127.0.0.1:9 did not identify as LM Studio",
        ),
        (
            "llamacpp",
            "Error: could not detect model from llama-server at http://127.0.0.1:9. Use --model",
        ),
    ] {
        let stderr = bench_fails(&["--json", "--provider", provider, "--url", DEAD]);
        assert!(stderr.contains(message), "{provider}: {stderr}");
    }
}

#[test]
fn a_named_model_still_reaches_the_provider() {
    for (provider, message) in [
        ("vllm", "Error: Cannot read http://127.0.0.1:9/v1/models"),
        ("ferrum", "Error: Cannot read http://127.0.0.1:9/v1/models"),
        (
            "lmstudio",
            "Error: Endpoint at http://127.0.0.1:9 did not identify as LM Studio",
        ),
        ("llamacpp", "Error: Warmup request failed"),
    ] {
        let stderr = bench_fails(&[
            "--json",
            "--provider",
            provider,
            "--url",
            DEAD,
            "some-model",
        ]);
        assert!(stderr.contains(message), "{provider}: {stderr}");
    }
}

#[test]
fn the_quality_bench_and_the_probe_ask_for_a_model() {
    for flag in ["--quality", "--tool-calls"] {
        let stderr = bench_fails(&[flag, "--provider", "ollama"]);
        assert!(
            stderr.contains("Error: model name required for this bench"),
            "{flag}: {stderr}"
        );
    }
}
