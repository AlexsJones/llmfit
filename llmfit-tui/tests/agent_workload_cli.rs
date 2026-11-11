//! `--agent-turns` / `--agent-turn-tokens` size memory for the context an
//! agent loop grows into (#914), rather than for the context declared.

use assert_cmd::Command;
use serde_json::Value;

/// Base estimation context (8192) plus 10 turns of 2048 tokens.
const AGENT_CONTEXT: &str = "28672";
const AGENT_CONTEXT_TOKENS: u64 = 28_672;

fn llmfit(args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("llmfit").expect("failed to locate llmfit test binary");
    cmd.env_remove("LLAMA_CPP_PATH")
        .env_remove("OLLAMA_CONTEXT_LENGTH")
        .args(["--no-dashboard", "--json", "--memory", "24", "--ram", "32"])
        .args(args);
    cmd
}

/// Runs llmfit with `args`, expects clap or the sizing checks to refuse, and
/// returns what it said. A refusal for an unknown flag would pass for the
/// wrong reason.
fn refused(args: &[&str]) -> String {
    let out = llmfit(args).assert().failure().get_output().stderr.clone();
    let stderr = String::from_utf8_lossy(&out).into_owned();
    assert!(
        !stderr.contains("unexpected argument"),
        "refused as an unknown flag: {stderr}"
    );
    stderr
}

fn stderr_of(args: &[&str]) -> String {
    let out = llmfit(args).assert().success().get_output().stderr.clone();
    String::from_utf8_lossy(&out).into_owned()
}

fn fit_models(args: &[&str]) -> Vec<Value> {
    let out = llmfit(args)
        .arg("fit")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&out).expect("fit --json emits JSON");
    json["models"]
        .as_array()
        .expect("fit --json has a models array")
        .clone()
}

/// What the context changes about a model. Throughput and score also depend
/// on a memory bandwidth measured at every start, so two runs on a busy
/// machine never agree on them.
fn sizing(model: &Value) -> Value {
    serde_json::json!({
        "name": model["name"],
        "effective_context_length": model["effective_context_length"],
        "usable_context": model["usable_context"],
        "memory_required_gb": model["memory_required_gb"],
        "fit_level": model["fit_level"],
        "run_mode": model["run_mode"],
        "notes": model["notes"],
    })
}

fn assert_same_sizing(left: &[Value], right: &[Value]) {
    assert_eq!(left.len(), right.len(), "different number of models");
    for (l, r) in left.iter().zip(right) {
        assert_eq!(sizing(l), sizing(r), "model {}", l["name"]);
    }
}

#[test]
fn the_profile_sizes_memory_like_the_context_it_works_out() {
    let agent = fit_models(&["--agent-turns", "10", "--agent-turn-tokens", "2048"]);
    let explicit = fit_models(&["--max-context", AGENT_CONTEXT]);
    assert!(!agent.is_empty(), "expected fit results");
    assert_same_sizing(&agent, &explicit);
}

#[test]
fn the_profile_sizes_for_more_context_than_the_default_estimate() {
    let agent = fit_models(&["--agent-turns", "10", "--agent-turn-tokens", "2048"]);
    let default = fit_models(&[]);
    let grew = agent.iter().zip(&default).any(|(a, d)| {
        a["effective_context_length"].as_u64() > d["effective_context_length"].as_u64()
    });
    assert!(grew, "a grown context must raise the estimation context");
}

#[test]
fn a_model_whose_window_is_shorter_than_the_context_says_so() {
    let models = fit_models(&["--agent-turns", "10", "--agent-turn-tokens", "2048"]);
    let says = |m: &Value| {
        m["notes"].as_array().is_some_and(|n| {
            n.iter()
                .any(|n| n.as_str().is_some_and(|s| s.contains("cannot hold")))
        })
    };
    let (short, long): (Vec<&Value>, Vec<&Value>) = models
        .iter()
        .partition(|m| m["context_length"].as_u64() < Some(AGENT_CONTEXT_TOKENS));
    assert!(!short.is_empty() && !long.is_empty(), "need both kinds");
    assert!(
        short.iter().all(|m| says(m)),
        "a short window goes unflagged"
    );
    assert!(!long.iter().any(|m| says(m)), "a long window is flagged");
}

#[test]
fn the_assumptions_are_named_on_stderr() {
    let stderr = stderr_of(&["fit", "--agent-turns", "10", "--agent-turn-tokens", "2048"]);
    assert!(stderr.contains("10 tool turns"), "{stderr}");
    assert!(stderr.contains("2048 tokens"), "{stderr}");
    assert!(stderr.contains(AGENT_CONTEXT), "{stderr}");
}

#[test]
fn the_assumptions_are_named_without_a_subcommand() {
    let stderr = stderr_of(&[
        "--cli",
        "--agent-turns",
        "10",
        "--agent-turn-tokens",
        "2048",
    ]);
    assert!(stderr.contains(AGENT_CONTEXT), "{stderr}");
}

#[test]
fn one_assumption_alone_is_refused_wherever_it_is_given() {
    for args in [
        &["fit", "--agent-turns", "10"][..],
        &["--agent-turns", "10", "fit"],
    ] {
        let stderr = refused(args);
        assert!(stderr.contains("--agent-turn-tokens"), "{stderr}");
    }
    for args in [
        &["fit", "--agent-turn-tokens", "2048"][..],
        &["--agent-turn-tokens", "2048", "fit"],
    ] {
        let stderr = refused(args);
        assert!(stderr.contains("--agent-turns"), "{stderr}");
    }
}

#[test]
fn the_two_assumptions_may_be_split_across_the_subcommand() {
    for args in [
        &["--agent-turns", "10", "fit", "--agent-turn-tokens", "2048"][..],
        &["--agent-turn-tokens", "2048", "fit", "--agent-turns", "10"],
    ] {
        let stderr = stderr_of(args);
        assert!(stderr.contains(AGENT_CONTEXT), "{stderr}");
    }
}

#[test]
fn the_profile_and_an_explicit_context_cap_are_mutually_exclusive() {
    // Before the subcommand, after it, and split across it in both orders.
    for args in [
        &[
            "--agent-turns",
            "10",
            "--agent-turn-tokens",
            "2048",
            "--max-context",
            "4096",
            "fit",
        ][..],
        &[
            "fit",
            "--agent-turns",
            "10",
            "--agent-turn-tokens",
            "2048",
            "--max-context",
            "4096",
        ],
        &[
            "--max-context",
            "4096",
            "fit",
            "--agent-turns",
            "10",
            "--agent-turn-tokens",
            "2048",
        ],
        &[
            "--agent-turns",
            "10",
            "--agent-turn-tokens",
            "2048",
            "fit",
            "--max-context",
            "4096",
        ],
    ] {
        let stderr = refused(args);
        assert!(stderr.contains("--max-context"), "{args:?}: {stderr}");
        assert!(stderr.contains("--agent-turns"), "{args:?}: {stderr}");
    }
}

#[test]
fn zero_turns_or_zero_tokens_are_refused() {
    for args in [
        ["--agent-turns", "0", "--agent-turn-tokens", "2048"],
        ["--agent-turns", "10", "--agent-turn-tokens", "0"],
    ] {
        let stderr = refused(&[&args[..], &["fit"]].concat());
        assert!(stderr.contains("0 is not in 1.."), "{stderr}");
    }
}

#[test]
fn a_subcommand_that_sizes_nothing_refuses_the_profile() {
    // `plan` takes its own --context and would otherwise report a context it
    // does not use; `system` has no estimate at all.
    let plan = refused(&[
        "plan",
        "Qwen/Qwen3-8B",
        "--context",
        "4096",
        "--agent-turns",
        "10",
        "--agent-turn-tokens",
        "2048",
    ]);
    assert!(plan.contains("--context"), "{plan}");
    let system = refused(&[
        "system",
        "--agent-turns",
        "10",
        "--agent-turn-tokens",
        "2048",
    ]);
    assert!(system.contains("--agent-turns"), "{system}");
}

#[test]
fn the_flags_are_accepted_after_the_subcommand() {
    llmfit(&[])
        .args(["fit", "--agent-turns", "10", "--agent-turn-tokens", "2048"])
        .assert()
        .success();
}
