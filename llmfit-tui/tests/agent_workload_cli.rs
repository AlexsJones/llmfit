//! `--agent-turns` / `--agent-turn-tokens` size memory for the context an
//! agent loop grows into (#914), rather than for the context declared.

use assert_cmd::Command;
use serde_json::Value;

/// Base estimation context (8192) plus 10 turns of 2048 tokens.
const AGENT_CONTEXT: &str = "28672";

fn llmfit(args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("llmfit").expect("failed to locate llmfit test binary");
    cmd.env_remove("LLAMA_CPP_PATH")
        .env_remove("OLLAMA_CONTEXT_LENGTH")
        .args(["--no-dashboard", "--json", "--memory", "24", "--ram", "32"])
        .args(args);
    cmd
}

/// Runs `fit` with `args`, expects clap to refuse, and returns what it said.
/// A refusal for an unknown flag would pass for the wrong reason.
fn refused(args: &[&str]) -> String {
    let out = llmfit(args)
        .arg("fit")
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&out).into_owned();
    assert!(
        !stderr.contains("unexpected argument"),
        "refused as an unknown flag: {stderr}"
    );
    stderr
}

fn fit_models(args: &[&str]) -> Value {
    let out = llmfit(args)
        .arg("fit")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&out).expect("fit --json emits JSON");
    json["models"].clone()
}

#[test]
fn the_profile_sizes_memory_like_the_context_it_works_out() {
    let agent = fit_models(&["--agent-turns", "10", "--agent-turn-tokens", "2048"]);
    let explicit = fit_models(&["--max-context", AGENT_CONTEXT]);
    assert!(
        agent.as_array().is_some_and(|m| !m.is_empty()),
        "expected fit results"
    );
    assert_eq!(agent, explicit);
}

#[test]
fn the_profile_sizes_for_more_context_than_the_default_estimate() {
    let agent = fit_models(&["--agent-turns", "10", "--agent-turn-tokens", "2048"]);
    let default = fit_models(&[]);
    assert_ne!(agent, default, "a grown context must change the estimate");
}

#[test]
fn the_assumptions_are_named_on_stderr() {
    let out = llmfit(&["--agent-turns", "10", "--agent-turn-tokens", "2048"])
        .arg("fit")
        .assert()
        .success()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&out);
    assert!(stderr.contains("10 tool turns"), "{stderr}");
    assert!(stderr.contains("2048 tokens"), "{stderr}");
    assert!(stderr.contains(AGENT_CONTEXT), "{stderr}");
}

#[test]
fn one_assumption_alone_is_refused() {
    let turns = refused(&["--agent-turns", "10"]);
    assert!(turns.contains("--agent-turn-tokens"), "{turns}");
    let tokens = refused(&["--agent-turn-tokens", "2048"]);
    assert!(tokens.contains("--agent-turns"), "{tokens}");
}

#[test]
fn the_profile_and_an_explicit_context_cap_are_mutually_exclusive() {
    let stderr = refused(&[
        "--agent-turns",
        "10",
        "--agent-turn-tokens",
        "2048",
        "--max-context",
        "4096",
    ]);
    assert!(stderr.contains("--max-context"), "{stderr}");
}

#[test]
fn zero_turns_or_zero_tokens_are_refused() {
    for args in [
        ["--agent-turns", "0", "--agent-turn-tokens", "2048"],
        ["--agent-turns", "10", "--agent-turn-tokens", "0"],
    ] {
        let stderr = refused(&args);
        assert!(stderr.contains("0 is not in 1.."), "{stderr}");
    }
}

#[test]
fn the_flags_are_accepted_after_the_subcommand() {
    llmfit(&[])
        .args(["fit", "--agent-turns", "10", "--agent-turn-tokens", "2048"])
        .assert()
        .success();
}
