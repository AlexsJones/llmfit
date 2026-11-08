//! `llmfit bench --nodes/--tp/--pp/--interconnect` records the serving
//! layout of the benchmarked endpoint with the results (#1227).

use assert_cmd::Command;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;

const MODELS_BODY: &str = r#"{"object":"list","data":[{"id":"facebook/opt-125m","object":"model","owned_by":"vllm","root":"facebook/opt-125m","max_model_len":512}]}"#;
const CHAT_BODY: &str = r#"{"choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":3,"completion_tokens":20}}"#;

/// Answers like vLLM: `/v1/models` lists one model, any other request gets a
/// chat completion. Returns the base URL.
fn serve_vllm() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        while let Ok((stream, _)) = listener.accept() {
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut request_line = String::new();
            reader.read_line(&mut request_line).expect("request line");
            let mut length = 0;
            let mut line = String::new();
            while reader.read_line(&mut line).expect("header") > 2 {
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().expect("length");
                }
                line.clear();
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).expect("body");
            let reply = if request_line.contains("/models") {
                MODELS_BODY
            } else {
                CHAT_BODY
            };
            let mut stream = stream;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.len(),
                reply
            )
            .expect("write");
        }
    });
    format!("http://{addr}")
}

fn temp_store(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("llmfit-bench-layout-{tag}-{}", std::process::id()))
}

fn bench(args: &[&str], store: &PathBuf) -> std::process::Output {
    Command::cargo_bin("llmfit")
        .expect("llmfit binary")
        .env("LLMFIT_BENCH_STORE", store)
        .arg("bench")
        .args(args)
        .output()
        .expect("run llmfit")
}

#[test]
fn bench_records_layout_flags_in_output_and_store() {
    let store = temp_store("record");
    let url = serve_vllm();
    let out = bench(
        &[
            "--provider",
            "vllm",
            "--url",
            &url,
            "--runs",
            "1",
            "--json",
            "--nodes",
            "2",
            "--tp",
            "2",
            "--interconnect",
            "200GbE RoCE",
        ],
        &store,
    );
    let stored: Vec<Value> = std::fs::read_dir(store.join("pending"))
        .map(|dir| {
            dir.flatten()
                .filter_map(|e| std::fs::read_to_string(e.path()).ok())
                .filter_map(|s| serde_json::from_str(&s).ok())
                .collect()
        })
        .unwrap_or_default();
    let _ = std::fs::remove_dir_all(&store); // best effort: a temp dir
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let json: Value = serde_json::from_slice(&out.stdout).expect("bench --json output");
    let layout = &json["result"]["layout"];
    assert_eq!(layout["node_count"], 2);
    assert_eq!(layout["tp_size"], 2);
    assert!(layout.get("pp_size").is_none());
    assert_eq!(layout["interconnect"], "200GbE RoCE");

    assert_eq!(stored.len(), 1, "the result is saved in the test's store");
    let result = &stored[0]["results"][0];
    assert_eq!(result["provider"], "vllm");
    assert_eq!(result["nodeCount"], 2);
    assert_eq!(result["tpSize"], 2);
    assert!(result.get("ppSize").is_none());
    assert_eq!(result["interconnect"], "200GbE RoCE");
}

#[test]
fn bench_without_layout_flags_records_no_layout() {
    let store = temp_store("none");
    let url = serve_vllm();
    let out = bench(
        &["--provider", "vllm", "--url", &url, "--runs", "1", "--json"],
        &store,
    );
    let _ = std::fs::remove_dir_all(&store);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: Value = serde_json::from_slice(&out.stdout).expect("bench --json output");
    assert!(json["result"].get("layout").is_none());
}

/// Invalid layouts fail before any endpoint is contacted or result stored.
#[test]
fn bench_rejects_invalid_layout_flags() {
    let store = temp_store("invalid");
    let cases: &[(&[&str], &str)] = &[
        (&["m", "--nodes", "0"], "--nodes"),
        (&["m", "--tp", "0"], "--tp"),
        (&["m", "--pp=-1"], "--pp"),
        (&["m", "--nodes", "two"], "--nodes"),
        (&["m", "--nodes", "2"], "--tp and/or --pp"),
        (
            &["m", "--tp", "2", "--interconnect", "  "],
            "--interconnect",
        ),
        (&["--all", "--nodes", "2", "--tp", "2"], "--all"),
        (&["--quality", "--tp", "2"], "--quality"),
        (&["m", "--provider", "ollama", "--tp", "2"], "not Ollama"),
    ];
    for (args, needle) in cases {
        let out = bench(args, &store);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {stderr}");
        assert!(stderr.contains(needle), "{args:?}: {stderr}");
    }
    assert!(
        !store.join("pending").exists(),
        "no result is stored for a rejected invocation"
    );
    let _ = std::fs::remove_dir_all(&store);
}
