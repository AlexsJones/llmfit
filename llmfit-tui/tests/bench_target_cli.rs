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

#[test]
fn openai_compat_needs_a_declared_engine_that_is_not_a_known_provider() {
    let compat = ["--json", "--provider", "openai_compat", "--url", DEAD];
    for (extra, message) in [
        (
            &[][..],
            "Error: --provider openai_compat needs --engine-name and --engine-version",
        ),
        (
            &["--engine-name", "capy-flash"][..],
            "Error: --provider openai_compat needs --engine-name and --engine-version",
        ),
        (
            &["--engine-name", "vLLM", "--engine-version", "0.11"][..],
            "Error: --engine-name 'vLLM' names a provider llmfit already benchmarks",
        ),
        (
            &["--engine-name", "llama.cpp", "--engine-version", "b5000"][..],
            "Error: --engine-name 'llama.cpp' names a provider llmfit already benchmarks",
        ),
        (
            &[
                "--engine-name",
                "capy-flash",
                "--engine-version",
                "0.16.0",
                "--all",
            ][..],
            "Error: --provider openai_compat benchmarks the one server at --url",
        ),
        (
            &["--engine-name", "capy-flash", "--engine-version", "0.16.0"][..],
            "Error: Cannot reach http://127.0.0.1:9/v1/models",
        ),
    ] {
        let args: Vec<&str> = compat.iter().chain(extra).copied().collect();
        let stderr = bench_fails(&args);
        assert!(stderr.contains(message), "{extra:?}: {stderr}");
    }

    let stderr = bench_fails(&[
        "--json",
        "--provider",
        "openai_compat",
        "--engine-name",
        "capy-flash",
        "--engine-version",
        "0.16.0",
    ]);
    assert!(
        stderr.contains("Error: --provider openai_compat needs the server's --url"),
        "{stderr}"
    );

    let stderr = bench_fails(&[
        "--json",
        "--provider",
        "vllm",
        "--url",
        DEAD,
        "--engine-name",
        "capy-flash",
    ]);
    assert!(
        stderr.contains(
            "--engine-name, --engine-version and --engine-repo need --provider openai_compat"
        ),
        "{stderr}"
    );
}
