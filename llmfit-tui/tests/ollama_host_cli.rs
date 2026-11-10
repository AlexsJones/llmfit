//! `llmfit bench --provider ollama` with `OLLAMA_HOST` in the form Ollama
//! documents (`127.0.0.1:11434`, no scheme). The bench paths read the
//! variable raw and every request failed with `http: invalid format`.

use assert_cmd::Command;
use std::io::{BufRead, BufReader, Read, Write};

/// Answers like Ollama's `/api/generate` to every request; returns host:port.
fn serve() -> String {
    const BODY: &str = r#"{"response":"hi","eval_count":10,"eval_duration":1000000000,"prompt_eval_count":5,"prompt_eval_duration":100000000,"total_duration":1200000000}"#;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        while let Ok((stream, _)) = listener.accept() {
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
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
            let mut stream = stream;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                BODY.len(),
                BODY
            )
            .expect("write");
        }
    });
    addr.to_string()
}

#[test]
fn bench_reaches_ollama_host_given_without_a_scheme() {
    // A successful bench saves its result for `--share`: kept out of the
    // user's store, or a later `bench --share` would upload this fake one.
    let store = std::env::temp_dir().join(format!("llmfit-ollama-host-{}", std::process::id()));
    let out = Command::cargo_bin("llmfit")
        .expect("llmfit binary")
        .env("OLLAMA_HOST", serve())
        .env("LLMFIT_BENCH_STORE", &store)
        .args([
            "bench",
            "m",
            "--provider",
            "ollama",
            "--runs",
            "1",
            "--json",
        ])
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
