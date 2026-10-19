//! `llmfit bench <model> --tool-calls`: the tool-call probe from the CLI,
//! against a loopback server that answers like Ollama's
//! `/v1/chat/completions`.

use assert_cmd::Command;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};

/// Answers `body` to every request; returns the base URL.
fn serve(body: &'static str) -> String {
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
                body.len(),
                body
            )
            .expect("write");
        }
    });
    format!("http://{addr}")
}

const CALLED: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"write","arguments":"{\"path\":\"notes.txt\",\"content\":\"hello\"}"}}]}}]}"#;
const IN_CONTENT: &str = r#"{"choices":[{"message":{"role":"assistant","content":"{\"name\":\"write\"}","tool_calls":null}}]}"#;

fn probe(url: &str, runs: &str) -> std::process::Output {
    Command::cargo_bin("llmfit")
        .expect("llmfit binary")
        .args([
            "bench",
            "qwen2.5-coder:7b",
            "--tool-calls",
            "--provider",
            "ollama",
            "--url",
            url,
            "--runs",
            runs,
            "--json",
        ])
        .output()
        .expect("run llmfit")
}

#[test]
fn tool_calls_reports_ok_for_a_model_that_calls_the_tool() {
    let out = probe(&serve(CALLED), "2");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: Value = serde_json::from_slice(&out.stdout).expect("JSON");
    assert_eq!(json["verdict"], "ok");
    assert_eq!(json["passed"], 2);
    assert_eq!(json["model"], "qwen2.5-coder:7b");
}

#[test]
fn tool_calls_reports_broken_for_a_model_that_prints_its_call() {
    let out = probe(&serve(IN_CONTENT), "2");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: Value = serde_json::from_slice(&out.stdout).expect("JSON");
    assert_eq!(json["verdict"], "broken");
    assert_eq!(json["attempts"][0], "no_tool_call");
}

#[test]
fn tool_calls_against_an_unreachable_server_fails() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    drop(listener);
    let out = probe(&url, "1");
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn tool_calls_text_output_names_the_verdict() {
    let url = serve(IN_CONTENT);
    let out = Command::cargo_bin("llmfit")
        .expect("llmfit binary")
        .args([
            "bench",
            "m",
            "--tool-calls",
            "--provider",
            "ollama",
            "--url",
            &url,
            "--runs",
            "1",
        ])
        .output()
        .expect("run llmfit");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("broken"), "{text}");
    assert!(text.contains("0/1"), "{text}");
}
