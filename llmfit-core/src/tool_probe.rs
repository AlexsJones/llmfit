//! Tool-call probe: does a served model really return OpenAI-style tool calls?
//!
//! The `tool_use` capability comes from the model card, but whether calls
//! reach `tool_calls` depends on the runtime and its chat template. Served by
//! Ollama, `qwen2.5-coder:7b` prints its call as JSON in `content` and
//! `tool_calls` stays `null`: an agent that picked it for tool use runs and
//! executes nothing. The probe asks for one call, several times, and reports
//! what came back.

use std::time::Duration;

use serde_json::Value;

/// The one tool offered, and the prompt that needs it.
pub const PROBE_TOOL: &str = "write";
pub const PROBE_PROMPT: &str =
    "Create the file notes.txt containing the word hello. Use the write tool.";

/// What one attempt returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Attempt {
    /// `tool_calls[0]` names the tool, with `path` and `content` arguments.
    Called,
    /// No tool call at all (`tool_calls` missing, `null` or empty).
    NoToolCall,
    /// A tool call that cannot be executed: other tool, arguments missing or
    /// not JSON.
    Malformed,
}

/// The verdict over every attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Every attempt returned a usable call.
    Ok,
    /// Some did.
    Flaky,
    /// None did.
    Broken,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolProbeResult {
    pub model: String,
    pub provider: String,
    pub attempts: Vec<Attempt>,
    pub passed: usize,
    pub verdict: Verdict,
}

/// Classifies one `/v1/chat/completions` response.
pub fn classify(response: &Value) -> Attempt {
    let calls = &response["choices"][0]["message"]["tool_calls"];
    let Some(first) = calls.as_array().and_then(|calls| calls.first()) else {
        return Attempt::NoToolCall;
    };
    if first["function"]["name"] != PROBE_TOOL {
        return Attempt::Malformed;
    }
    let arguments: Option<Value> = first["function"]["arguments"]
        .as_str()
        .and_then(|raw| serde_json::from_str(raw).ok());
    let usable = arguments.is_some_and(|args| {
        args["path"].as_str().is_some_and(|path| !path.is_empty()) && args["content"].is_string()
    });
    if usable {
        Attempt::Called
    } else {
        Attempt::Malformed
    }
}

/// The verdict over `attempts`. No attempt is no evidence: `Broken`.
pub fn verdict(attempts: &[Attempt]) -> Verdict {
    let called = attempts.iter().filter(|a| **a == Attempt::Called).count();
    if called == 0 {
        Verdict::Broken
    } else if called == attempts.len() {
        Verdict::Ok
    } else {
        Verdict::Flaky
    }
}

fn probe_request(model: &str) -> Value {
    serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": PROBE_PROMPT }],
        "tools": [{
            "type": "function",
            "function": {
                "name": PROBE_TOOL,
                "description": "Write a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }
            }
        }],
        "stream": false,
    })
}

/// Probes `model` behind an OpenAI-compatible endpoint (Ollama serves one at
/// `/v1/chat/completions` too) with `attempts` requests. A request that fails
/// is an error, not an attempt: an unreachable server says nothing about the
/// model.
pub fn probe_openai_compat(
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    provider: &str,
    attempts: usize,
) -> Result<ToolProbeResult, String> {
    if attempts == 0 {
        return Err("the tool-call probe needs at least one attempt".to_string());
    }
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let body = probe_request(model);
    let mut seen = Vec::with_capacity(attempts);
    for _ in 0..attempts {
        let mut req = ureq::post(&url)
            .config()
            .timeout_global(Some(Duration::from_secs(300)))
            .build();
        if let Some(key) = api_key {
            req = req.header("Authorization", &format!("Bearer {key}"));
        }
        let response: Value = req
            .send_json(&body)
            .map_err(|e| format!("{url} request failed: {e}"))?
            .into_body()
            .read_json()
            .map_err(|e| format!("{url} returned a body that is not JSON: {e}"))?;
        seen.push(classify(&response));
    }
    let passed = seen.iter().filter(|a| **a == Attempt::Called).count();
    Ok(ToolProbeResult {
        model: model.to_string(),
        provider: provider.to_string(),
        verdict: verdict(&seen),
        attempts: seen,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn response_with(message: Value) -> Value {
        json!({ "choices": [{ "message": message }] })
    }

    fn call(name: &str, arguments: &str) -> Value {
        response_with(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": { "name": name, "arguments": arguments }
            }]
        }))
    }

    #[test]
    fn a_call_to_the_tool_with_its_arguments_is_called() {
        let r = call("write", r#"{"path":"notes.txt","content":"hello"}"#);
        assert_eq!(classify(&r), Attempt::Called);
    }

    #[test]
    fn null_tool_calls_with_the_call_printed_in_content_is_no_tool_call() {
        // What qwen2.5-coder:7b returns when served by Ollama.
        let r = response_with(json!({
            "role": "assistant",
            "content": "{\"name\": \"write\", \"arguments\": {\"path\": \"notes.txt\"}}",
            "tool_calls": null
        }));
        assert_eq!(classify(&r), Attempt::NoToolCall);
    }

    #[test]
    fn an_empty_tool_call_list_is_no_tool_call() {
        let r = response_with(json!({ "content": "done", "tool_calls": [] }));
        assert_eq!(classify(&r), Attempt::NoToolCall);
    }

    #[test]
    fn a_missing_tool_call_field_is_no_tool_call() {
        let r = response_with(json!({ "content": "done" }));
        assert_eq!(classify(&r), Attempt::NoToolCall);
    }

    #[test]
    fn a_call_to_another_tool_is_malformed() {
        let r = call("read", r#"{"path":"notes.txt","content":"hello"}"#);
        assert_eq!(classify(&r), Attempt::Malformed);
    }

    #[test]
    fn a_call_without_its_path_is_malformed() {
        let r = call("write", r#"{"content":"hello"}"#);
        assert_eq!(classify(&r), Attempt::Malformed);
    }

    #[test]
    fn a_call_with_an_empty_path_is_malformed() {
        let r = call("write", r#"{"path":"","content":"hello"}"#);
        assert_eq!(classify(&r), Attempt::Malformed);
    }

    #[test]
    fn a_call_without_its_content_is_malformed() {
        let r = call("write", r#"{"path":"notes.txt"}"#);
        assert_eq!(classify(&r), Attempt::Malformed);
    }

    #[test]
    fn arguments_that_are_not_json_are_malformed() {
        let r = call("write", "path=notes.txt");
        assert_eq!(classify(&r), Attempt::Malformed);
    }

    #[test]
    fn a_response_without_choices_is_no_tool_call() {
        assert_eq!(classify(&json!({ "error": "boom" })), Attempt::NoToolCall);
    }

    #[test]
    fn every_attempt_called_is_ok() {
        assert_eq!(verdict(&[Attempt::Called, Attempt::Called]), Verdict::Ok);
    }

    #[test]
    fn some_attempts_called_is_flaky() {
        assert_eq!(
            verdict(&[Attempt::Called, Attempt::Malformed]),
            Verdict::Flaky
        );
    }

    #[test]
    fn no_attempt_called_is_broken() {
        assert_eq!(
            verdict(&[Attempt::NoToolCall, Attempt::Malformed]),
            Verdict::Broken
        );
    }

    #[test]
    fn no_attempt_at_all_is_broken() {
        assert_eq!(verdict(&[]), Verdict::Broken);
    }

    /// Serves `body` to every request on a loopback port and hands each
    /// request back on the channel.
    fn serve(body: &'static str) -> (String, std::sync::mpsc::Receiver<String>) {
        use std::io::{BufRead, BufReader, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (tx, rx) = std::sync::mpsc::channel();
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
                let _ = tx.send(String::from_utf8(request).expect("utf8"));
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
        (format!("http://{addr}"), rx)
    }

    const CALLED: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"write","arguments":"{\"path\":\"notes.txt\",\"content\":\"hello\"}"}}]}}]}"#;
    const IN_CONTENT: &str = r#"{"choices":[{"message":{"role":"assistant","content":"{\"name\":\"write\"}","tool_calls":null}}]}"#;

    #[test]
    fn a_model_that_always_calls_the_tool_is_ok() {
        let (url, _) = serve(CALLED);
        let r = probe_openai_compat(&url, None, "m", "Ollama", 3).expect("probe");
        assert_eq!(r.verdict, Verdict::Ok);
        assert_eq!(r.passed, 3);
        assert_eq!(r.attempts.len(), 3);
    }

    #[test]
    fn a_model_that_prints_its_call_in_content_is_broken() {
        let (url, _) = serve(IN_CONTENT);
        let r = probe_openai_compat(&url, None, "m", "Ollama", 2).expect("probe");
        assert_eq!(r.verdict, Verdict::Broken);
        assert_eq!(r.passed, 0);
    }

    #[test]
    fn the_request_offers_the_tool_to_the_model_it_names() {
        let (url, requests) = serve(CALLED);
        probe_openai_compat(&url, None, "qwen3:8b", "Ollama", 1).expect("probe");
        let body: Value = serde_json::from_str(&requests.recv().expect("request")).expect("json");
        assert_eq!(body["model"], "qwen3:8b");
        assert_eq!(body["tools"][0]["function"]["name"], PROBE_TOOL);
        assert_eq!(body["messages"][0]["content"], PROBE_PROMPT);
    }

    #[test]
    fn zero_attempts_is_refused() {
        let err = probe_openai_compat("http://127.0.0.1:9", None, "m", "Ollama", 0)
            .expect_err("zero attempts");
        assert!(err.contains("at least one"), "{err}");
    }

    #[test]
    fn an_unreachable_endpoint_is_an_error_not_a_verdict() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        drop(listener);
        let err = probe_openai_compat(&url, None, "m", "Ollama", 1).expect_err("unreachable");
        assert!(err.contains(&url), "{err}");
    }
}
