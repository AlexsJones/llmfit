//! `llmfit bench --provider vllm` with `VLLM_HOST` set. The vLLM provider
//! reads `VLLM_HOST`, but bench only built `http://localhost:$VLLM_PORT`, so
//! it benchmarked (or failed to reach) a different server.

use assert_cmd::Command;
use std::io::{BufRead, BufReader, Read, Write};

/// Answers `/v1/models` like vLLM and every other request like a chat
/// completion; returns host:port.
fn serve() -> String {
    const MODELS: &str = r#"{"object":"list","data":[{"id":"facebook/opt-125m","object":"model","owned_by":"vllm"}]}"#;
    const CHAT: &str = r#"{"choices":[{"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":10}}"#;
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
            let mut request = vec![0; length];
            reader.read_exact(&mut request).expect("body");
            let body = if request_line.starts_with("GET /v1/models") {
                MODELS
            } else {
                CHAT
            };
            let mut stream = stream;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .expect("write");
        }
    });
    addr.to_string()
}

#[test]
fn bench_reaches_the_vllm_host_the_provider_uses() {
    // A successful bench saves its result for `--share`: kept out of the
    // user's store, or a later `bench --share` would upload this fake one.
    let store = std::env::temp_dir().join(format!("llmfit-vllm-host-{}", std::process::id()));
    let out = Command::cargo_bin("llmfit")
        .expect("llmfit binary")
        .env("VLLM_HOST", serve())
        .env_remove("VLLM_PORT")
        .env("LLMFIT_BENCH_STORE", &store)
        .args(["bench", "--provider", "vllm", "--runs", "1", "--json"])
        .output()
        .expect("run llmfit");
    let saved = std::fs::read_dir(store.join("pending")).map(|d| d.count());
    let _ = std::fs::remove_dir_all(&store); // best effort: a temp dir
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        saved.ok(),
        Some(1),
        "the result is saved in the test's store"
    );
}
