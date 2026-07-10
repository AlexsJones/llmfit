//! Contribute benchmark results back to the project as a GitHub pull request.
//!
//! Every successful bench run is first recorded in a **local store** as a
//! ready-to-upload submission payload (see [`store_local`]). Sharing — now or
//! any time later — uploads everything still pending in a single PR, after
//! which the files move to `shared/` so they remain as local history but are
//! never sent twice. Declining to share therefore never discards data.
//!
//! No `gh` CLI and no server are required. Authentication uses the GitHub OAuth
//! **device flow** — the same mechanism `gh auth login` uses — with a public
//! client id, so nothing secret ships in the binary. The fork / commit / open-PR
//! steps then go through the GitHub REST API via `ureq` (already a dependency).
//!
//! A `GITHUB_TOKEN` / `GH_TOKEN` env var, or a token cached from a previous
//! device-flow login, short-circuits the interactive step — which also makes
//! `--share` usable from CI. Credentials are resolved and verified *before*
//! benchmarks run ([`preflight_auth`]) so a bad token is caught up front.
//!
//! All human-facing output goes to stderr so it never corrupts `bench --json`
//! output on stdout.

use crate::bench::BenchResult;
use crate::hardware::SystemSpecs;
use base64::Engine;
use serde::Serialize;
use serde_json::{Value, json};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const UPSTREAM_OWNER: &str = "AlexsJones";
const UPSTREAM_REPO: &str = "llmfit";
const UPSTREAM_BRANCH: &str = "main";
const SUBMISSION_DIR: &str = "llmfit-core/data/community";
const SCHEMA_VERSION: u32 = 1;
const USER_AGENT: &str = concat!("llmfit/", env!("CARGO_PKG_VERSION"));
const API: &str = "https://api.github.com";

/// Public OAuth App client id used for the device flow. This is **not** a
/// secret (device flow requires no client secret) and is safe to ship. Until
/// the real OAuth App is registered, override it with the `LLMFIT_GH_CLIENT_ID`
/// environment variable.
const DEFAULT_CLIENT_ID: &str = "REPLACE_WITH_OAUTH_APP_CLIENT_ID";

/// Options controlling a `bench --share` invocation.
pub struct ShareOptions {
    /// Print the payload that would be submitted and exit without contacting GitHub.
    pub dry_run: bool,
    /// Skip the interactive confirmation prompt (assume "yes").
    pub assume_yes: bool,
}

// ---------------------------------------------------------------------------
// Submission payload
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Submission {
    schema_version: u32,
    submitted_at_unix: u64,
    tool: ToolInfo,
    hardware: HwPayload,
    results: Vec<ResultPayload>,
}

#[derive(Serialize)]
struct ToolInfo {
    name: &'static str,
    version: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HwPayload {
    hw_class: &'static str,
    hardware_name: Option<String>,
    mem_tier_gb: Option<u32>,
    vram_gb: Option<f64>,
    gpu_count: u32,
    unified_memory: bool,
    cpu: String,
    cpu_cores: usize,
    ram_gb: f64,
    os: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResultPayload {
    model: String,
    provider: String,
    num_runs: usize,
    avg_tps: f64,
    min_tps: f64,
    max_tps: f64,
    avg_ttft_ms: Option<f64>,
    avg_total_ms: f64,
    avg_output_tokens: f64,
}

fn os_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    }
}

/// Round a memory size to the nearest common tier (matches the leaderboard's
/// coarse buckets so submissions group cleanly).
fn nearest_mem_tier(gb: f64) -> u32 {
    const TIERS: [u32; 12] = [8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256];
    let mut best = 0u32;
    let mut best_d = f64::MAX;
    for &t in &TIERS {
        let d = (gb - t as f64).abs();
        if d < best_d {
            best_d = d;
            best = t;
        }
    }
    best
}

fn build_submission(results: &[BenchResult], specs: &SystemSpecs) -> Submission {
    let hw_class = if specs.unified_memory {
        "UNIFIED"
    } else if specs.has_gpu {
        "DISCRETE_GPU"
    } else {
        "CPU_ONLY"
    };

    let mem_tier_gb = if let Some(vram) = specs.total_gpu_vram_gb {
        let t = nearest_mem_tier(vram);
        (t > 0).then_some(t)
    } else if specs.unified_memory {
        let t = nearest_mem_tier(specs.total_ram_gb);
        (t > 0).then_some(t)
    } else {
        None
    };

    let results = results
        .iter()
        .map(|r| ResultPayload {
            model: r.model.clone(),
            provider: r.provider.clone(),
            num_runs: r.summary.num_runs,
            avg_tps: round2(r.summary.avg_tps),
            min_tps: round2(r.summary.min_tps),
            max_tps: round2(r.summary.max_tps),
            avg_ttft_ms: r.summary.avg_ttft_ms.map(round2),
            avg_total_ms: round2(r.summary.avg_total_ms),
            avg_output_tokens: round2(r.summary.avg_output_tokens),
        })
        .collect();

    Submission {
        schema_version: SCHEMA_VERSION,
        submitted_at_unix: now_unix(),
        tool: ToolInfo {
            name: "llmfit",
            version: env!("CARGO_PKG_VERSION"),
        },
        hardware: HwPayload {
            hw_class,
            hardware_name: specs.gpu_name.clone(),
            mem_tier_gb,
            vram_gb: specs.total_gpu_vram_gb.map(round2),
            gpu_count: specs.gpu_count,
            unified_memory: specs.unified_memory,
            cpu: specs.cpu_name.clone(),
            cpu_cores: specs.total_cpu_cores,
            ram_gb: round2(specs.total_ram_gb),
            os: os_name(),
        },
        results,
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Slug identifying the hardware of a stored submission payload, used for the
/// branch name and submission path.
fn payload_slug(payload: &Value) -> String {
    let hw = &payload["hardware"];
    let raw = hw["hardwareName"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| format!("cpu-{}", hw["cpu"].as_str().unwrap_or("unknown")));
    let mut slug: String = raw
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "unknown".to_string()
    } else {
        slug
    }
}

/// Short stable hash of the payload, for a unique branch/file name without
/// relying on a random source.
fn short_hash(s: &str) -> String {
    // FNV-1a 64-bit.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    // Mix in the wall clock so repeated identical runs still differ.
    h ^= now_unix();
    h = h.wrapping_mul(0x100000001b3);
    format!("{:08x}", (h & 0xffff_ffff) as u32)
}

// ---------------------------------------------------------------------------
// Local benchmark store
// ---------------------------------------------------------------------------

/// A submission payload recorded in the local benchmark store.
pub struct StoredBenchmark {
    pub path: PathBuf,
    pub payload: Value,
}

impl StoredBenchmark {
    /// One line per benchmark result: `model via provider — N tok/s`.
    pub fn result_lines(&self) -> Vec<String> {
        let Some(results) = self.payload["results"].as_array() else {
            return Vec::new();
        };
        results
            .iter()
            .map(|r| {
                format!(
                    "{} via {} — {:.1} tok/s",
                    r["model"].as_str().unwrap_or("?"),
                    r["provider"].as_str().unwrap_or("?"),
                    r["avgTps"].as_f64().unwrap_or(0.0),
                )
            })
            .collect()
    }
}

/// Root of the local store. Overridable with `LLMFIT_BENCH_STORE` (useful for
/// tests and for keeping the store on a shared volume).
fn store_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LLMFIT_BENCH_STORE")
        && !dir.trim().is_empty()
    {
        return Some(PathBuf::from(dir));
    }
    Some(dirs::data_local_dir()?.join("llmfit").join("benchmarks"))
}

fn read_store(subdir: &str) -> Vec<StoredBenchmark> {
    let Some(dir) = store_root().map(|r| r.join(subdir)) else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<StoredBenchmark> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                return None;
            }
            let payload: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).ok()?).ok()?;
            Some(StoredBenchmark { path, payload })
        })
        .collect();
    // Filenames start with the unix timestamp, so path order is record order.
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Record benchmark results locally as a ready-to-upload submission payload.
/// Returns the path of the stored file.
pub fn store_local(results: &[BenchResult], specs: &SystemSpecs) -> Result<PathBuf, String> {
    if results.is_empty() {
        return Err("no benchmark results to store".to_string());
    }
    let submission = build_submission(results, specs);
    let json =
        serde_json::to_string_pretty(&submission).map_err(|e| format!("serialize failed: {e}"))?;
    let dir = store_root()
        .ok_or("no local data directory")?
        .join("pending");
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let path = dir.join(format!("{}-{}.json", now_unix(), short_hash(&json)));
    std::fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

/// Stored benchmarks not yet contributed upstream, oldest first.
pub fn pending_benchmarks() -> Vec<StoredBenchmark> {
    read_store("pending")
}

/// Stored benchmarks already contributed upstream, oldest first.
pub fn shared_benchmarks() -> Vec<StoredBenchmark> {
    read_store("shared")
}

/// Move uploaded submissions from `pending/` to `shared/` so they remain as
/// local history but are never uploaded twice. Best-effort: a file that cannot
/// be moved stays pending (worst case a duplicate submission, never data loss).
pub fn mark_shared(stored: &[StoredBenchmark]) {
    let Some(dir) = store_root().map(|r| r.join("shared")) else {
        return;
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    for s in stored {
        if let Some(name) = s.path.file_name() {
            let _ = std::fs::rename(&s.path, dir.join(name));
        }
    }
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/// Share everything in the local pending store as a single pull request.
/// Interactive CLI flow: lists the stored submissions, confirms with the
/// user, then uploads and marks them shared. Pass a pre-validated `token`
/// (from [`preflight_auth`]) to skip credential resolution here.
///
/// Returns `Ok(Some(pr_url))` on success, `Ok(None)` if the user cancelled or
/// `--dry-run` was set, and `Err(_)` on failure.
pub fn share_all_pending(
    opts: &ShareOptions,
    token: Option<String>,
) -> Result<Option<String>, String> {
    let stored = pending_benchmarks();
    if stored.is_empty() {
        return Err(
            "no local benchmarks are stored yet. Run `llmfit bench <model>` or \
             `llmfit bench --all` first."
                .to_string(),
        );
    }

    eprintln!("\n  The following stored benchmark data would be contributed:\n");
    for s in &stored {
        for line in s.result_lines() {
            eprintln!("    - {line}");
        }
    }

    if opts.dry_run {
        for s in &stored {
            if let Ok(json) = serde_json::to_string_pretty(&s.payload) {
                eprintln!("\n  --- {} ---", s.path.display());
                for line in json.lines() {
                    eprintln!("    {line}");
                }
            }
        }
        eprintln!("\n  --dry-run: nothing was submitted.");
        return Ok(None);
    }

    if !opts.assume_yes {
        let prompt = format!(
            "\n  Share all {} local benchmark submission(s) as a PR to \
             {UPSTREAM_OWNER}/{UPSTREAM_REPO}?",
            stored.len()
        );
        if !confirm(&prompt)? {
            eprintln!("  Cancelled — results remain stored locally.");
            return Ok(None);
        }
    }

    let token = match token {
        Some(t) => t,
        None => preflight_auth()?,
    };
    let pr_url = submit_stored(&stored, &token)?;
    mark_shared(&stored);
    Ok(Some(pr_url))
}

/// Non-interactive core of the share flow: fork the repo, commit each stored
/// submission to one new branch, and open a single pull request using an
/// already-resolved token. Never prompts or reads stdin, so it is safe to
/// call from a worker thread while a TUI owns the terminal. Returns the PR
/// URL; the caller is responsible for [`mark_shared`] afterwards.
pub fn submit_stored(stored: &[StoredBenchmark], token: &str) -> Result<String, String> {
    if stored.is_empty() {
        return Err("no benchmark results to share".to_string());
    }

    // Re-stamp the submission time: stored files may be days old.
    let now = now_unix();
    let files: Vec<(String, String)> = stored
        .iter()
        .map(|s| {
            let mut payload = s.payload.clone();
            payload["submittedAtUnix"] = json!(now);
            let json = serde_json::to_string_pretty(&payload)
                .map_err(|e| format!("serialize failed: {e}"))?;
            Ok((payload_slug(&payload), json))
        })
        .collect::<Result<_, String>>()?;

    let login = whoami(token)?;
    ensure_fork(token, &login)?;
    let base_sha = upstream_head_sha(token)?;

    let all_json: String = files.iter().map(|(_, j)| j.as_str()).collect();
    let hash = short_hash(&all_json);
    let slug = files[0].0.clone();
    let branch = format!("bench/{slug}-{hash}");
    create_branch(token, &login, &branch, &base_sha)?;

    for (i, (file_slug, json)) in files.iter().enumerate() {
        let path = format!("{SUBMISSION_DIR}/{file_slug}/{now}-{hash}-{i}.json");
        let message = format!("data: community benchmark ({file_slug})");
        put_file(token, &login, &branch, &path, json, &message)?;
    }

    open_pr(token, &login, &branch, stored, &slug)
}

fn confirm(prompt: &str) -> Result<bool, String> {
    eprint!("{prompt} [y/N] ");
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| format!("failed to read input: {e}"))?;
    let a = line.trim().to_lowercase();
    Ok(a == "y" || a == "yes")
}

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

/// Where a non-interactively resolved token came from. An invalid env token
/// is a hard error (the user set it explicitly); an invalid cached token is
/// silently discarded and re-acquired via the device flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenSource {
    /// `GITHUB_TOKEN` / `GH_TOKEN` environment variable.
    Env,
    /// Token cached by a previous device-flow login.
    Cache,
}

/// Resolve a GitHub token without any user interaction: env vars, then the
/// cached token from a previous device-flow login. Returns `None` when an
/// interactive login would be required.
pub fn resolve_token_noninteractive() -> Option<String> {
    resolve_token_noninteractive_with_source().map(|(t, _)| t)
}

/// [`resolve_token_noninteractive`], also reporting where the token came from.
pub fn resolve_token_noninteractive_with_source() -> Option<(String, TokenSource)> {
    for var in ["GITHUB_TOKEN", "GH_TOKEN"] {
        if let Some(t) = std::env::var(var)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            return Some((t, TokenSource::Env));
        }
    }
    read_cached_token().map(|t| (t, TokenSource::Cache))
}

/// Check a token against the GitHub API. `Ok(Some(login))` means the token is
/// valid, `Ok(None)` means GitHub rejected it (invalid/expired), `Err` means
/// the request itself failed (network, rate limit, ...).
pub fn validate_token(token: &str) -> Result<Option<String>, String> {
    let (status, body) = api("GET", &format!("{API}/user"), token, None)?;
    if status == 401 {
        return Ok(None);
    }
    if !(200..300).contains(&status) {
        return Err(api_error(status, &body));
    }
    Ok(Some(
        body["login"]
            .as_str()
            .ok_or("could not determine GitHub username")?
            .to_string(),
    ))
}

/// Drop the cached device-flow token (e.g. after GitHub reports it expired).
pub fn clear_cached_token() {
    if let Some(p) = token_path() {
        let _ = std::fs::remove_file(p);
    }
}

/// Resolve *and verify* GitHub credentials, intended to run **before** any
/// benchmarks so a missing or expired token is caught up front rather than
/// after minutes of benching. Falls back to the interactive device flow when
/// possible. Prints progress to stderr (CLI flow).
pub fn preflight_auth() -> Result<String, String> {
    if let Some((token, source)) = resolve_token_noninteractive_with_source() {
        match validate_token(&token) {
            Ok(Some(login)) => {
                eprintln!("  GitHub: authenticated as {login}");
                return Ok(token);
            }
            Ok(None) => match source {
                TokenSource::Env => {
                    return Err(
                        "the GITHUB_TOKEN/GH_TOKEN environment variable holds an invalid \
                         or expired token"
                            .to_string(),
                    );
                }
                TokenSource::Cache => {
                    eprintln!("  GitHub: cached login has expired — starting a new login");
                    clear_cached_token();
                }
            },
            Err(e) => return Err(format!("could not verify GitHub credentials: {e}")),
        }
    }
    let Some(client_id) = oauth_client_id() else {
        return Err(
            "no GitHub token found. Set GITHUB_TOKEN (or GH_TOKEN), or set \
             LLMFIT_GH_CLIENT_ID to a registered OAuth App client id to enable \
             interactive login."
                .to_string(),
        );
    };
    let token = device_flow(&client_id)?;
    if let Err(e) = write_cached_token(&token) {
        eprintln!("  Warning: could not cache token: {e}");
    }
    Ok(token)
}

/// The OAuth App client id for the device flow, or `None` when only the
/// unregistered placeholder is available (interactive login not possible).
pub fn oauth_client_id() -> Option<String> {
    let id = std::env::var("LLMFIT_GH_CLIENT_ID").unwrap_or_else(|_| DEFAULT_CLIENT_ID.to_string());
    (id != DEFAULT_CLIENT_ID).then_some(id)
}

/// Persist a token obtained via the device flow for future runs.
pub fn cache_token(token: &str) -> Result<(), String> {
    write_cached_token(token)
}

fn token_path() -> Option<std::path::PathBuf> {
    Some(dirs::config_dir()?.join("llmfit").join("github_token"))
}

fn read_cached_token() -> Option<String> {
    let p = token_path()?;
    let t = std::fs::read_to_string(p).ok()?;
    let t = t.trim().to_string();
    (!t.is_empty()).then_some(t)
}

fn write_cached_token(token: &str) -> Result<(), String> {
    let p = token_path().ok_or("no config directory")?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&p, token).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// A started device-flow authorization: show `user_code` / `verification_uri`
/// to the user, then call [`device_flow_poll`] every `interval` seconds.
pub struct DeviceAuth {
    pub user_code: String,
    pub verification_uri: String,
    pub device_code: String,
    /// Minimum seconds to wait between polls.
    pub interval: u64,
}

/// Outcome of a single device-flow poll.
pub enum DevicePoll {
    /// Authorized; carries the access token.
    Token(String),
    /// User has not authorized yet — poll again after `interval`.
    Pending,
    /// Server asked to slow down — add ~5s to the interval and poll again.
    SlowDown,
    /// Terminal failure (expired, denied, or protocol error).
    Failed(String),
}

/// Begin the GitHub OAuth device flow: request a user code for the given
/// OAuth App client id.
pub fn device_flow_start(client_id: &str) -> Result<DeviceAuth, String> {
    let resp = ureq::post("https://github.com/login/device/code")
        .config()
        .http_status_as_error(false)
        .build()
        .header("Accept", "application/json")
        .header("User-Agent", USER_AGENT)
        .send_json(json!({ "client_id": client_id, "scope": "public_repo" }))
        .map_err(|e| format!("device code request failed: {e}"))?;
    let v: Value = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("device code parse failed: {e}"))?;

    Ok(DeviceAuth {
        device_code: v["device_code"]
            .as_str()
            .ok_or("device flow: missing device_code")?
            .to_string(),
        user_code: v["user_code"].as_str().unwrap_or("").to_string(),
        verification_uri: v["verification_uri"]
            .as_str()
            .unwrap_or("https://github.com/login/device")
            .to_string(),
        interval: v["interval"].as_u64().unwrap_or(5).max(1),
    })
}

/// Poll the device flow once. The caller owns the pacing (sleep `interval`
/// seconds between calls), which lets a TUI keep its event loop responsive.
pub fn device_flow_poll(client_id: &str, device_code: &str) -> Result<DevicePoll, String> {
    let resp = ureq::post("https://github.com/login/oauth/access_token")
        .config()
        .http_status_as_error(false)
        .build()
        .header("Accept", "application/json")
        .header("User-Agent", USER_AGENT)
        .send_json(json!({
            "client_id": client_id,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }))
        .map_err(|e| format!("token poll failed: {e}"))?;
    let v: Value = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("token poll parse failed: {e}"))?;

    if let Some(tok) = v["access_token"].as_str() {
        return Ok(DevicePoll::Token(tok.to_string()));
    }
    Ok(match v["error"].as_str() {
        Some("authorization_pending") => DevicePoll::Pending,
        Some("slow_down") => DevicePoll::SlowDown,
        Some("expired_token") => {
            DevicePoll::Failed("device code expired before authorization".into())
        }
        Some("access_denied") => DevicePoll::Failed("authorization was denied".into()),
        Some(other) => DevicePoll::Failed(format!("authorization failed: {other}")),
        None => DevicePoll::Failed("unexpected response while polling for token".into()),
    })
}

/// Run the GitHub OAuth device flow, blocking until the user authorizes or the
/// code expires. Returns the access token. (CLI path; prints to stderr.)
fn device_flow(client_id: &str) -> Result<String, String> {
    let auth = device_flow_start(client_id)?;
    let mut interval = auth.interval;

    eprintln!("\n  To authorize llmfit to open a pull request on your behalf:\n");
    eprintln!("    1. Open {}", auth.verification_uri);
    eprintln!("    2. Enter code: {}\n", auth.user_code);
    eprintln!("  Waiting for authorization (Ctrl-C to cancel)...");

    loop {
        std::thread::sleep(Duration::from_secs(interval + 1));
        match device_flow_poll(client_id, &auth.device_code)? {
            DevicePoll::Token(tok) => return Ok(tok),
            DevicePoll::Pending => continue,
            DevicePoll::SlowDown => {
                interval += 5;
                continue;
            }
            DevicePoll::Failed(e) => return Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// GitHub REST API helpers
// ---------------------------------------------------------------------------

/// Issue an authenticated GitHub API request. Returns `(status, body)` where
/// `body` is parsed JSON (or `Value::Null` when there is none). Non-2xx statuses
/// are returned rather than raised so callers can react to them.
fn api(method: &str, url: &str, token: &str, body: Option<&Value>) -> Result<(u16, Value), String> {
    let auth = format!("Bearer {token}");
    let resp = match method {
        "GET" => ureq::get(url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("Authorization", &auth)
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", USER_AGENT)
            .header("X-GitHub-Api-Version", "2022-11-28")
            .call(),
        "POST" => ureq::post(url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("Authorization", &auth)
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", USER_AGENT)
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send_json(body.unwrap_or(&json!({}))),
        "PUT" => ureq::put(url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("Authorization", &auth)
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", USER_AGENT)
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send_json(body.unwrap_or(&json!({}))),
        _ => return Err(format!("unsupported method {method}")),
    }
    .map_err(|e| format!("{method} {url} failed: {e}"))?;

    let status = resp.status().as_u16();
    let val: Value = resp.into_body().read_json().unwrap_or(Value::Null);
    Ok((status, val))
}

/// Extract a human-readable error message from a GitHub error body.
fn api_error(status: u16, body: &Value) -> String {
    let msg = body["message"].as_str().unwrap_or("unknown error");
    format!("GitHub API returned {status}: {msg}")
}

fn whoami(token: &str) -> Result<String, String> {
    validate_token(token)?.ok_or_else(|| "GitHub token is invalid or expired (401)".into())
}

/// Ensure the authenticated user has a fork of the upstream repo, creating one
/// if needed and waiting until it is queryable.
fn ensure_fork(token: &str, login: &str) -> Result<(), String> {
    let fork_url = format!("{API}/repos/{login}/{UPSTREAM_REPO}");
    let (status, _) = api("GET", &fork_url, token, None)?;
    if status == 200 {
        return Ok(());
    }

    let (status, body) = api(
        "POST",
        &format!("{API}/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/forks"),
        token,
        None,
    )?;
    if !(200..300).contains(&status) {
        return Err(api_error(status, &body));
    }

    // Forking is asynchronous; poll until the fork responds.
    for _ in 0..15 {
        std::thread::sleep(Duration::from_secs(2));
        let (status, _) = api("GET", &fork_url, token, None)?;
        if status == 200 {
            return Ok(());
        }
    }
    Err("fork was not ready after waiting; try --share again shortly".into())
}

fn upstream_head_sha(token: &str) -> Result<String, String> {
    let url =
        format!("{API}/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/git/ref/heads/{UPSTREAM_BRANCH}");
    let (status, body) = api("GET", &url, token, None)?;
    if !(200..300).contains(&status) {
        return Err(api_error(status, &body));
    }
    body["object"]["sha"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "could not read upstream head sha".into())
}

fn create_branch(token: &str, login: &str, branch: &str, sha: &str) -> Result<(), String> {
    let url = format!("{API}/repos/{login}/{UPSTREAM_REPO}/git/refs");
    let body = json!({ "ref": format!("refs/heads/{branch}"), "sha": sha });
    let (status, body) = api("POST", &url, token, Some(&body))?;
    // 201 created; 422 typically means the ref already exists — acceptable.
    if status == 201 || status == 422 {
        return Ok(());
    }
    Err(api_error(status, &body))
}

fn put_file(
    token: &str,
    login: &str,
    branch: &str,
    path: &str,
    content: &str,
    message: &str,
) -> Result<(), String> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(content);
    let url = format!("{API}/repos/{login}/{UPSTREAM_REPO}/contents/{path}");
    let body = json!({
        "message": message,
        "content": encoded,
        "branch": branch,
    });
    let (status, body) = api("PUT", &url, token, Some(&body))?;
    if !(200..300).contains(&status) {
        return Err(api_error(status, &body));
    }
    Ok(())
}

fn open_pr(
    token: &str,
    login: &str,
    branch: &str,
    stored: &[StoredBenchmark],
    slug: &str,
) -> Result<String, String> {
    let title = format!("bench: community results for {slug}");
    let mut body = String::from(
        "Automated benchmark contribution from `llmfit bench --share`.\n\n\
         | Model | Provider | Avg TPS | Avg TTFT (ms) |\n\
         | --- | --- | --- | --- |\n",
    );
    for s in stored {
        let results = s.payload["results"].as_array();
        for r in results.map(|v| v.as_slice()).unwrap_or_default() {
            let ttft = r["avgTtftMs"]
                .as_f64()
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "—".to_string());
            body.push_str(&format!(
                "| {} | {} | {:.1} | {} |\n",
                r["model"].as_str().unwrap_or("?"),
                r["provider"].as_str().unwrap_or("?"),
                r["avgTps"].as_f64().unwrap_or(0.0),
                ttft
            ));
        }
    }
    body.push_str("\n_Submitted without the `gh` CLI via the GitHub device flow._\n");

    let url = format!("{API}/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/pulls");
    let payload = json!({
        "title": title,
        "head": format!("{login}:{branch}"),
        "base": UPSTREAM_BRANCH,
        "body": body,
    });
    let (status, resp) = api("POST", &url, token, Some(&payload))?;
    if !(200..300).contains(&status) {
        return Err(api_error(status, &resp));
    }
    resp["html_url"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "pull request created but no URL returned".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_tier_rounds_to_nearest() {
        assert_eq!(nearest_mem_tier(23.9), 24);
        assert_eq!(nearest_mem_tier(31.0), 32);
        assert_eq!(nearest_mem_tier(7.5), 8);
    }

    fn specs_with_gpu(name: &str) -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(24.0),
            total_gpu_vram_gb: Some(24.0),
            gpu_name: Some(name.to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: crate::hardware::GpuBackend::Cuda,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    fn sample_result() -> BenchResult {
        use crate::bench::BenchSummary;
        BenchResult {
            model: "llama3.1:8b".to_string(),
            provider: "ollama".to_string(),
            runs: vec![],
            summary: BenchSummary {
                num_runs: 3,
                avg_ttft_ms: Some(41.2),
                avg_tps: 128.44,
                min_tps: 121.0,
                max_tps: 133.7,
                avg_total_ms: 812.5,
                avg_output_tokens: 104.0,
            },
        }
    }

    #[test]
    fn slug_is_filename_safe() {
        let submission = build_submission(&[sample_result()], &specs_with_gpu("NVIDIA RTX 4090!!"));
        let payload = serde_json::to_value(&submission).unwrap();
        let slug = payload_slug(&payload);
        assert!(slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'));
        assert!(!slug.contains("--"));
        assert!(!slug.starts_with('-') && !slug.ends_with('-'));
        assert_eq!(slug, "nvidia-rtx-4090");

        // No GPU name → falls back to a cpu-derived slug.
        let mut cpu_only = specs_with_gpu("unused");
        cpu_only.gpu_name = None;
        let payload =
            serde_json::to_value(build_submission(&[sample_result()], &cpu_only)).unwrap();
        assert_eq!(payload_slug(&payload), "cpu-test-cpu");
    }

    #[test]
    fn local_store_roundtrip() {
        // LLMFIT_BENCH_STORE scopes the store to a temp dir. Env vars are
        // process-global, so this is the only test that may touch the store.
        let dir = std::env::temp_dir().join(format!("llmfit-store-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::set_var("LLMFIT_BENCH_STORE", &dir) };

        let specs = specs_with_gpu("NVIDIA GeForce RTX 4090");
        let path = store_local(&[sample_result()], &specs).unwrap();
        assert!(path.starts_with(dir.join("pending")));

        let pending = pending_benchmarks();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].payload["schemaVersion"], 1);
        assert_eq!(
            pending[0].result_lines(),
            vec!["llama3.1:8b via ollama — 128.4 tok/s".to_string()]
        );
        assert!(shared_benchmarks().is_empty());

        mark_shared(&pending);
        assert!(pending_benchmarks().is_empty());
        let shared = shared_benchmarks();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].payload["results"][0]["avgTps"], 128.44);

        unsafe { std::env::remove_var("LLMFIT_BENCH_STORE") };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn short_hash_is_hex() {
        let h = short_hash("{\"a\":1}");
        assert_eq!(h.len(), 8);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn submission_matches_published_schema() {
        use crate::bench::{BenchResult, BenchSummary};

        let result = BenchResult {
            model: "llama3.1:8b".to_string(),
            provider: "ollama".to_string(),
            runs: vec![],
            summary: BenchSummary {
                num_runs: 3,
                avg_ttft_ms: Some(41.234),
                avg_tps: 128.44,
                min_tps: 121.0,
                max_tps: 133.7,
                avg_total_ms: 812.5,
                avg_output_tokens: 104.0,
            },
        };
        // llama-server results are labeled "llamacpp" — must be schema-valid too.
        let llamacpp_result = BenchResult {
            model: "qwen2.5-7b-q4_k_m".to_string(),
            provider: "llamacpp".to_string(),
            runs: vec![],
            summary: BenchSummary {
                num_runs: 3,
                avg_ttft_ms: None,
                avg_tps: 42.5,
                min_tps: 40.0,
                max_tps: 45.0,
                avg_total_ms: 2400.0,
                avg_output_tokens: 100.0,
            },
        };

        let submission = build_submission(
            &[result, llamacpp_result],
            &specs_with_gpu("NVIDIA GeForce RTX 4090"),
        );
        let value = serde_json::to_value(&submission).unwrap();

        let schema_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/community/schema.json");
        let schema: Value =
            serde_json::from_str(&std::fs::read_to_string(&schema_path).unwrap()).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();

        let errors: Vec<String> = validator
            .iter_errors(&value)
            .map(|e| format!("  [{}] {}", e.instance_path(), e))
            .collect();
        assert!(
            errors.is_empty(),
            "generated submission violates schema:\n{}\npayload:\n{}",
            errors.join("\n"),
            serde_json::to_string_pretty(&value).unwrap()
        );

        // camelCase field names must survive serialization.
        assert_eq!(value["schemaVersion"], 1);
        assert_eq!(value["hardware"]["hwClass"], "DISCRETE_GPU");
        assert_eq!(value["hardware"]["memTierGb"], 24);
        assert_eq!(value["results"][0]["avgTps"], 128.44);
    }
}
