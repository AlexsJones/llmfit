//! Small OpenAI-compatible client used by interactive recommendation surfaces.
//!
//! The caller owns the conversation and the evidence supplied to the model.
//! This module only validates connection settings and performs one bounded
//! chat-completions request.

use std::net::IpAddr;
use std::time::Duration;

use http::{HeaderValue, Uri};
use serde::{Deserialize, Serialize};

pub const ADVISOR_BASE_URL_ENV: &str = "LLMFIT_ADVISOR_BASE_URL";
pub const ADVISOR_MODEL_ENV: &str = "LLMFIT_ADVISOR_MODEL";
pub const ADVISOR_API_KEY_ENV: &str = "LLMFIT_ADVISOR_API_KEY";
pub const ADVISOR_REASONING_EFFORT_ENV: &str = "LLMFIT_ADVISOR_REASONING_EFFORT";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);
const MAX_RESPONSE_BYTES: u64 = 64 * 1024;
const MAX_COMPLETION_TOKENS: u32 = 320;

/// Connection settings for an OpenAI-compatible chat-completions endpoint.
///
/// Deliberately does not implement `Debug`: the value may contain an API key.
#[derive(Clone)]
pub struct AdvisorConfig {
    base_url: String,
    model: String,
    api_key: Option<String>,
    reasoning_effort: Option<ReasoningEffort>,
}

impl AdvisorConfig {
    pub fn from_env() -> Result<Self, String> {
        let base_url = std::env::var(ADVISOR_BASE_URL_ENV).map_err(|_| {
            format!("Set {ADVISOR_BASE_URL_ENV} and {ADVISOR_MODEL_ENV} to enable the advisor")
        })?;
        let model = std::env::var(ADVISOR_MODEL_ENV).map_err(|_| {
            format!("Set {ADVISOR_BASE_URL_ENV} and {ADVISOR_MODEL_ENV} to enable the advisor")
        })?;
        let api_key = std::env::var(ADVISOR_API_KEY_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty());
        let reasoning_effort = std::env::var(ADVISOR_REASONING_EFFORT_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.parse())
            .transpose()?;

        Self::new(base_url, model, api_key).map(|mut config| {
            config.reasoning_effort = reasoning_effort;
            config
        })
    }

    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Result<Self, String> {
        let base_url = base_url.into().trim().trim_end_matches('/').to_string();
        let model = model.into().trim().to_string();
        let api_key = api_key
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        if base_url.is_empty() {
            return Err(format!("{ADVISOR_BASE_URL_ENV} cannot be empty"));
        }
        if model.is_empty() {
            return Err(format!("{ADVISOR_MODEL_ENV} cannot be empty"));
        }

        let uri: Uri = base_url
            .parse()
            .map_err(|_| format!("{ADVISOR_BASE_URL_ENV} must be a valid HTTP(S) URL"))?;
        if !matches!(uri.scheme_str(), Some("http" | "https")) || uri.authority().is_none() {
            return Err(format!("{ADVISOR_BASE_URL_ENV} must be an HTTP(S) URL"));
        }
        if uri.scheme_str() == Some("http")
            && !uri
                .authority()
                .is_some_and(|authority| is_loopback_host(authority.host()))
        {
            return Err(format!(
                "{ADVISOR_BASE_URL_ENV} must use HTTPS unless the endpoint is on localhost"
            ));
        }
        if uri
            .authority()
            .is_some_and(|authority| authority.as_str().contains('@'))
        {
            return Err(format!(
                "{ADVISOR_BASE_URL_ENV} must not contain embedded credentials"
            ));
        }
        if uri.query().is_some() {
            return Err(format!(
                "{ADVISOR_BASE_URL_ENV} must not contain a query string"
            ));
        }

        if let Some(key) = api_key.as_deref() {
            let bearer = format!("Bearer {key}");
            HeaderValue::from_str(&bearer)
                .map_err(|_| format!("{ADVISOR_API_KEY_ENV} is not a valid header value"))?;
        }

        Ok(Self {
            base_url,
            model,
            api_key,
            reasoning_effort: None,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn has_api_key(&self) -> bool {
        self.api_key.is_some()
    }

    pub fn reasoning_effort_label(&self) -> &'static str {
        self.reasoning_effort
            .map_or("provider default", ReasoningEffort::as_str)
    }

    /// Safe connection label for display. It never includes credentials.
    pub fn endpoint_label(&self) -> String {
        let Ok(uri) = self.base_url.parse::<Uri>() else {
            return "configured endpoint".to_string();
        };
        let scheme = uri.scheme_str().map_or("https", |value| value);
        let authority = uri.authority().map_or("", |value| value.as_str());
        format!("{scheme}://{authority}")
    }

    fn chat_completions_url(&self) -> String {
        if self.base_url.ends_with("/chat/completions") {
            return self.base_url.clone();
        }

        let path_is_root = self
            .base_url
            .parse::<Uri>()
            .is_ok_and(|uri| matches!(uri.path(), "" | "/"));
        if path_is_root {
            format!("{}/v1/chat/completions", self.base_url)
        } else {
            format!("{}/chat/completions", self.base_url)
        }
    }
}

fn is_loopback_host(host: &str) -> bool {
    let host = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Max,
    Xhigh,
}

impl ReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
            Self::Xhigh => "xhigh",
        }
    }
}

impl std::str::FromStr for ReasoningEffort {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "max" => Ok(Self::Max),
            "xhigh" => Ok(Self::Xhigh),
            _ => Err(format!(
                "{ADVISOR_REASONING_EFFORT_ENV} must be one of: none, minimal, low, medium, high, max, xhigh"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AdvisorRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdvisorMessage {
    pub role: AdvisorRole,
    pub content: String,
}

impl AdvisorMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: AdvisorRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: AdvisorRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: AdvisorRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [AdvisorMessage],
    stream: bool,
    temperature: f64,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: Option<serde_json::Value>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    #[serde(default)]
    reasoning_content: Option<serde_json::Value>,
}

/// Send one non-streaming chat-completions request.
pub fn complete(config: &AdvisorConfig, messages: &[AdvisorMessage]) -> Result<String, String> {
    if messages.is_empty() {
        return Err("Advisor request has no messages".to_string());
    }

    let body = ChatCompletionRequest {
        model: config.model(),
        messages,
        stream: false,
        temperature: 0.2,
        max_tokens: MAX_COMPLETION_TOKENS,
        reasoning_effort: config.reasoning_effort,
    };
    let mut request = ureq::post(config.chat_completions_url())
        .config()
        .timeout_global(Some(REQUEST_TIMEOUT))
        .max_redirects(0)
        .build();
    if let Some(api_key) = config.api_key.as_deref() {
        request = request.header("Authorization", format!("Bearer {api_key}"));
    }

    let response = request
        .send_json(&body)
        .map_err(|error| format!("Advisor request failed: {error}"))?;
    if response.status().is_redirection() {
        return Err(
            "Advisor endpoint redirects are not allowed; configure its final URL directly"
                .to_string(),
        );
    }
    let parsed: ChatCompletionResponse = response
        .into_body()
        .into_with_config()
        .limit(MAX_RESPONSE_BYTES)
        .read_json()
        .map_err(|error| format!("Advisor returned invalid JSON: {error}"))?;

    parse_response(parsed)
}

fn parse_response(response: ChatCompletionResponse) -> Result<String, String> {
    let message = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| "Advisor returned no choices".to_string())?
        .message;

    if let Some(content) = message.content {
        let text = match content {
            serde_json::Value::String(text) => text,
            serde_json::Value::Array(parts) => parts
                .into_iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                })
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        if !text.trim().is_empty() {
            return Ok(text.trim().to_string());
        }
    }

    if let Some(refusal) = message.refusal.filter(|value| !value.trim().is_empty()) {
        return Ok(refusal.trim().to_string());
    }

    if message.reasoning.as_ref().is_some_and(has_content)
        || message.reasoning_content.as_ref().is_some_and(has_content)
    {
        return Err(format!(
            "Advisor finished reasoning without a final answer. Try {ADVISOR_REASONING_EFFORT_ENV}=none or low, or choose a non-reasoning model"
        ));
    }

    Err("Advisor returned an empty response".to_string())
}

fn has_content(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::String(value) => !value.trim().is_empty(),
        serde_json::Value::Array(values) => !values.is_empty(),
        serde_json::Value::Object(values) => !values.is_empty(),
        serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    fn config(base_url: &str) -> AdvisorConfig {
        AdvisorConfig::new(base_url, "test-model", None).expect("valid test config")
    }

    fn read_http_request(stream: &mut TcpStream) -> Vec<u8> {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set test timeout");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let count = stream.read(&mut buffer).expect("read advisor request");
            request.extend_from_slice(&buffer[..count]);
            let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")?
                        .trim()
                        .parse::<usize>()
                        .ok()
                })
                .expect("request has content length");
            if request.len() >= header_end + 4 + content_length {
                return request;
            }
        }
    }

    #[test]
    fn builds_endpoint_from_host_or_api_prefix() {
        assert_eq!(
            config("http://localhost:11434").chat_completions_url(),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            config("https://openrouter.ai/api/v1").chat_completions_url(),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            config("https://generativelanguage.googleapis.com/v1beta/openai/")
                .chat_completions_url(),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        );
    }

    #[test]
    fn accepts_a_full_chat_completions_url() {
        let url = "https://example.com/custom/chat/completions";
        assert_eq!(config(url).chat_completions_url(), url);
    }

    #[test]
    fn rejects_credentials_and_queries_in_base_url() {
        assert!(AdvisorConfig::new("https://user:secret@example.com/v1", "m", None).is_err());
        assert!(AdvisorConfig::new("https://example.com/v1?key=secret", "m", None).is_err());
    }

    #[test]
    fn plain_http_is_limited_to_loopback_endpoints() {
        assert!(AdvisorConfig::new("http://127.0.0.1:11434", "m", None).is_ok());
        assert!(AdvisorConfig::new("http://[::1]:11434", "m", None).is_ok());
        assert!(AdvisorConfig::new("http://provider.example/v1", "m", None).is_err());
    }

    #[test]
    fn display_label_omits_endpoint_paths() {
        assert_eq!(
            config("https://example.com/private/path/v1").endpoint_label(),
            "https://example.com"
        );
    }

    #[test]
    fn validates_reasoning_effort() {
        assert_eq!("none".parse(), Ok(ReasoningEffort::None));
        assert_eq!("XHIGH".parse(), Ok(ReasoningEffort::Xhigh));
        assert!("lots".parse::<ReasoningEffort>().is_err());
    }

    #[test]
    fn parses_string_and_text_part_responses() {
        let string_response: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"message": {"content": " Use model A. "}}]
        }))
        .expect("valid response");
        assert_eq!(parse_response(string_response).unwrap(), "Use model A.");

        let parts_response: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"message": {"content": [
                {"type": "text", "text": "Use "},
                {"type": "text", "text": "model B."}
            ]}}]
        }))
        .expect("valid response");
        assert_eq!(parse_response(parts_response).unwrap(), "Use model B.");
    }

    #[test]
    fn rejects_missing_or_empty_choices() {
        let no_choices: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "choices": []
        }))
        .expect("valid response");
        assert!(parse_response(no_choices).is_err());

        let empty: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"message": {"content": "  "}}]
        }))
        .expect("valid response");
        assert!(parse_response(empty).is_err());

        let reasoning_only: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"message": {"content": "", "reasoning": "working"}}]
        }))
        .expect("valid response");
        assert!(
            parse_response(reasoning_only)
                .expect_err("reasoning-only response must fail")
                .contains(ADVISOR_REASONING_EFFORT_ENV)
        );
    }

    #[test]
    fn sends_openai_compatible_request_with_bearer_auth() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let address = listener.local_addr().expect("test server address");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept advisor request");
            let request = read_http_request(&mut stream);

            let body = r#"{"choices":[{"message":{"content":"Use the measured option."}}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .expect("write advisor response");
            String::from_utf8(request).expect("request is UTF-8")
        });

        let mut config = AdvisorConfig::new(
            format!("http://{address}/v1"),
            "expert-model",
            Some("test-secret".to_string()),
        )
        .expect("valid config");
        config.reasoning_effort = Some(ReasoningEffort::None);
        let response = complete(&config, &[AdvisorMessage::user("Help me choose")])
            .expect("successful completion");
        let request = server.join().expect("test server completed");

        assert_eq!(response, "Use the measured option.");
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-secret")
        );
        let (_, request_body) = request.split_once("\r\n\r\n").expect("request has a body");
        let request_json: serde_json::Value =
            serde_json::from_str(request_body).expect("request body is JSON");
        assert_eq!(request_json["model"], "expert-model");
        assert_eq!(request_json["messages"][0]["role"], "user");
        assert_eq!(request_json["reasoning_effort"], "none");
        assert_eq!(request_json["max_tokens"], MAX_COMPLETION_TOKENS);
    }

    #[test]
    fn does_not_follow_provider_redirects() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let address = listener.local_addr().expect("test server address");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept advisor request");
            let _request = read_http_request(&mut stream);
            write!(
                stream,
                "HTTP/1.1 302 Found\r\nLocation: http://{address}/redirected\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .expect("write redirect response");
            drop(stream);

            listener
                .set_nonblocking(true)
                .expect("set nonblocking listener");
            std::thread::sleep(Duration::from_millis(100));
            listener.accept().is_ok()
        });

        let result = complete(
            &config(&format!("http://{address}/v1")),
            &[AdvisorMessage::user("Help me choose")],
        );
        assert!(result.expect_err("redirect must fail").contains("redirect"));
        assert!(!server.join().expect("test server completed"));
    }
}
