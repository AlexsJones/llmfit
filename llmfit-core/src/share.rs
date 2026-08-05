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

/// Public OAuth App client id used for the device flow (the "llmfit" OAuth
/// App, device flow enabled). This is **not** a secret — the device flow
/// requires no client secret, so shipping it in the binary is by design.
/// Override with the `LLMFIT_GH_CLIENT_ID` environment variable (e.g. when
/// running a fork against a different OAuth App).
const DEFAULT_CLIENT_ID: &str = "Ov23lirCd460lRfnbKyK";

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

/// Round a memory size to the nearest common tier, for the capacity a
/// submission *declares* about the machine it ran on.
///
/// The ladder is deliberately coarse so submissions group cleanly, but it has
/// a floor: capacities that shipped below 8 GB get their own rungs, otherwise
/// a 4 GB card is published as an 8 GB part and compared against hardware it
/// has nothing in common with.
///
/// Exact ties resolve downward, which understates rather than overstates. That
/// is the safe direction here, because this value is a claim about what a
/// machine actually has.
///
/// Not to be merged with `lookup_mem_tier` in `benchmarks.rs`, which looks
/// almost identical and means something else. See the note there.
fn declared_mem_tier(gb: f64) -> u32 {
    const TIERS: [u32; 16] = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256];
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
        let t = declared_mem_tier(vram);
        (t > 0).then_some(t)
    } else if specs.unified_memory {
        let t = declared_mem_tier(specs.total_ram_gb);
        (t > 0).then_some(t)
    } else {
        None
    };

    let results = results
        .iter()
        .map(|r| ResultPayload {
            // For the llamacpp provider `r.model` is the id reported by
            // llama-server, which is usually the absolute path of the loaded
            // GGUF. Store the bare file name instead (#819).
            model: strip_gguf_path(&r.model),
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

/// Whether a hardware-identity string is a placeholder emitted when detection
/// failed (e.g. a GPU reported as "N/A" because ROCm/libdrm couldn't name it).
fn is_placeholder_identity(s: &str) -> bool {
    let t = s.trim().to_lowercase();
    t.is_empty()
        || matches!(
            t.as_str(),
            "n/a" | "na" | "n-a" | "unknown" | "none" | "null" | "-"
        )
}

/// Whether a stored submission identifies its hardware well enough to be a
/// useful community contribution. The leaderboard buckets results by hardware
/// identity — GPU/accelerator name for GPU machines, CPU for CPU-only ones — so
/// a placeholder identity produces a meaningless bucket and must not be shared.
fn hardware_is_identifiable(payload: &Value) -> bool {
    let hw = &payload["hardware"];
    match hw["hwClass"].as_str().unwrap_or("") {
        "DISCRETE_GPU" | "UNIFIED" => {
            !is_placeholder_identity(hw["hardwareName"].as_str().unwrap_or(""))
        }
        // CPU-only machines are identified by their CPU (see `payload_slug`).
        _ => !is_placeholder_identity(hw["cpu"].as_str().unwrap_or("")),
    }
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
#[derive(Clone)]
pub struct StoredBenchmark {
    pub path: PathBuf,
    pub payload: Value,
}

impl StoredBenchmark {
    /// Whether this run was recorded on hardware matching `specs` (same CPU
    /// and GPU). Measurements from a previous machine configuration must not
    /// override or calibrate estimates for the current one.
    pub fn matches_hardware(&self, specs: &SystemSpecs) -> bool {
        crate::benchmarks::hardware_payload_matches(&self.payload["hardware"], specs)
    }

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

/// llama-server reports the value of its `-m/--model` argument, usually an
/// absolute filesystem path to a GGUF, as the model id in its
/// OpenAI-compatible `/v1/models` listing, and that id ends up verbatim in
/// `BenchResult::model`. Keep only the file name when the value is a path to
/// a `.gguf` file, so no machine-specific directory (often a username) leaks
/// into a stored submission (#819). Every other id shape (Ollama tags,
/// HF-style `org/model` ids from vLLM or MLX, bare file names) passes
/// through unchanged.
///
/// Only ids that look like absolute paths are stripped (leading `/` or `\\`,
/// or a Windows drive letter), so Hub-style references such as
/// `hf.co/org/repo/file.gguf` keep their namespace: they contain separators
/// and end in `.gguf` but carry nothing machine-specific. Splits on both
/// separator kinds, like `tag_matches_model` does: a Windows path can show
/// up verbatim in the listing.
fn strip_gguf_path(id: &str) -> String {
    let is_absolute = id.starts_with('/')
        || id.starts_with('\\')
        || matches!(id.as_bytes(), [_, b':', b'/' | b'\\', ..]);
    if is_absolute && id.to_ascii_lowercase().ends_with(".gguf") {
        id.rsplit(['/', '\\']).next().unwrap_or(id).to_string()
    } else {
        id.to_string()
    }
}

/// Rewrite absolute GGUF paths left in the `model` field of a stored payload.
/// New payloads are normalised in `build_submission`, but stores written by
/// older binaries still carry paths (#819). Scrubbing at load time means the
/// share listing, the `--dry-run` preview and the upload all agree, and no
/// machine-specific path leaves the machine.
fn sanitize_stored_payload(payload: &mut Value) {
    if let Some(results) = payload.get_mut("results").and_then(Value::as_array_mut) {
        for r in results {
            if let Some(model) = r["model"].as_str() {
                let stripped = strip_gguf_path(model);
                if stripped != model {
                    r["model"] = Value::String(stripped);
                }
            }
        }
    }
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
            let mut payload: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).ok()?).ok()?;
            sanitize_stored_payload(&mut payload);
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

/// Index of the user's own benchmark runs (pending and shared), used to
/// annotate fit rows: a throughput measured on THIS machine is ground truth
/// and takes priority over community medians and formula estimates.
pub struct LocalBenchIndex {
    /// (provider model tag, tok/s), newest run first.
    entries: Vec<(String, f64)>,
}

impl LocalBenchIndex {
    /// Load every stored benchmark result recorded on hardware matching
    /// `specs`. Returns `None` when nothing qualifies so callers can skip
    /// per-model lookups entirely.
    pub fn load(specs: &SystemSpecs) -> Option<Self> {
        let mut entries: Vec<(String, f64)> = Vec::new();
        for s in shared_benchmarks().into_iter().chain(pending_benchmarks()) {
            if !s.matches_hardware(specs) {
                continue;
            }
            let Some(results) = s.payload["results"].as_array() else {
                continue;
            };
            for r in results {
                if let (Some(model), Some(tps)) = (r["model"].as_str(), r["avgTps"].as_f64())
                    && tps > 0.0
                {
                    entries.push((model.to_string(), tps));
                }
            }
        }
        // Store reads are oldest-first; prefer the newest measurement.
        entries.reverse();
        (!entries.is_empty()).then_some(Self { entries })
    }

    /// Most recent locally measured tok/s for a catalog model, if any stored
    /// run's provider tag matches it.
    pub fn lookup(&self, model_hf_name: &str) -> Option<crate::benchmarks::MeasuredTps> {
        let matches: Vec<f64> = self
            .entries
            .iter()
            .filter(|(tag, _)| crate::providers::tag_matches_model(tag, model_hf_name))
            .map(|(_, tps)| *tps)
            .collect();
        Some(crate::benchmarks::MeasuredTps {
            tok_s: *matches.first()?,
            sample_count: matches.len() as u32,
            hardware_label: "this machine".to_string(),
            source: crate::benchmarks::MeasuredSource::LocalBench,
        })
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
/// Returns `Ok(Some(outcome))` on success, `Ok(None)` if the user cancelled
/// or `--dry-run` was set, and `Err(_)` on failure.
pub fn share_all_pending(
    opts: &ShareOptions,
    token: Option<String>,
) -> Result<Option<SubmitOutcome>, String> {
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
    let outcome = submit_stored(&stored, &token)?;
    mark_shared(&stored);
    Ok(Some(outcome))
}

/// Outcome of uploading stored submissions.
pub struct SubmitOutcome {
    /// PR that now contains the results — an already-open bench PR when one
    /// existed, otherwise a newly opened one. `None` when every file had
    /// already been submitted upstream and there was nothing to open a PR for.
    pub pr_url: Option<String>,
    /// Results were appended to an existing open PR instead of a new one.
    pub reused_existing_pr: bool,
    /// Files actually uploaded this time.
    pub uploaded: usize,
    /// Files skipped because a submission with the same name already exists
    /// upstream (e.g. a retry after a partially failed share).
    pub skipped: usize,
}

/// Non-interactive core of the share flow, safe to call from a worker thread
/// while a TUI owns the terminal (never prompts or reads stdin). Forks the
/// repo, then either **appends** the stored submissions to the user's
/// already-open benchmark PR (avoiding one PR per bench run) or commits them
/// to a new branch and opens one. Upstream file names mirror the local store
/// names, so re-submitting after a partial failure skips what already landed
/// instead of duplicating it. The caller is responsible for [`mark_shared`]
/// afterwards.
pub fn submit_stored(stored: &[StoredBenchmark], token: &str) -> Result<SubmitOutcome, String> {
    if stored.is_empty() {
        return Err("no benchmark results to share".to_string());
    }

    // Never contribute a result whose hardware could not be identified: the
    // community leaderboard groups submissions by hardware, so a placeholder
    // identity (e.g. a GPU reported as "N/A" when ROCm/libdrm can't name it) is
    // noise. Keep such results local rather than polluting the board.
    let stored: Vec<StoredBenchmark> = stored
        .iter()
        .filter(|s| hardware_is_identifiable(&s.payload))
        .cloned()
        .collect();
    if stored.is_empty() {
        return Err(
            "not submitting: your hardware could not be identified (GPU/accelerator \
             name reported as \"N/A\"). The community leaderboard groups results by \
             hardware, so an unidentified machine can't be contributed. Your results \
             remain stored locally."
                .to_string(),
        );
    }
    let stored = stored.as_slice();

    let now = now_unix();
    let files = prepare_files(stored, now)?;

    let login = whoami(token)?;
    ensure_fork(token, &login)?;

    // A lookup failure only means we open a fresh PR — never a lost result —
    // so it is deliberately soft.
    if let Some((branch, pr_url)) = find_open_bench_pr(token, &login).unwrap_or(None) {
        let (uploaded, skipped) = put_files(token, &login, &branch, &files)?;
        return Ok(SubmitOutcome {
            pr_url: Some(pr_url),
            reused_existing_pr: true,
            uploaded,
            skipped,
        });
    }

    let base_sha = upstream_head_sha(token)?;
    let all_json: String = files.iter().map(|(_, _, j)| j.as_str()).collect();
    let hash = short_hash(&all_json);
    let slug = files[0].0.clone();
    let branch = format!("bench/{slug}-{hash}");
    create_branch(token, &login, &branch, &base_sha)?;

    let (uploaded, skipped) = put_files(token, &login, &branch, &files)?;
    if uploaded == 0 {
        // Everything was already upstream (e.g. an earlier PR merged but the
        // local move to shared/ failed). No diff — GitHub would reject a PR.
        return Ok(SubmitOutcome {
            pr_url: None,
            reused_existing_pr: false,
            uploaded,
            skipped,
        });
    }

    let pr_url = open_pr(token, &login, &branch, stored, &slug)?;
    Ok(SubmitOutcome {
        pr_url: Some(pr_url),
        reused_existing_pr: false,
        uploaded,
        skipped,
    })
}

/// Build the upload set as `(hardware slug, file name, payload json)` per
/// stored submission, re-stamping the submission time (stored files may be
/// days old). Upstream file names reuse the local store name (record
/// timestamp + content hash): stable across retries, so a re-upload of the
/// same stored result targets the same path and is skipped, not duplicated.
fn prepare_files(
    stored: &[StoredBenchmark],
    now: u64,
) -> Result<Vec<(String, String, String)>, String> {
    stored
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut payload = s.payload.clone();
            payload["submittedAtUnix"] = json!(now);
            let json = serde_json::to_string_pretty(&payload)
                .map_err(|e| format!("serialize failed: {e}"))?;
            let name = s
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("{now}-{i}.json"));
            Ok((payload_slug(&payload), name, json))
        })
        .collect()
}

/// Commit each prepared file to `branch`, returning `(uploaded, skipped)`
/// where skipped counts files that already existed upstream.
fn put_files(
    token: &str,
    login: &str,
    branch: &str,
    files: &[(String, String, String)],
) -> Result<(usize, usize), String> {
    let (mut uploaded, mut skipped) = (0usize, 0usize);
    for (slug, name, json) in files {
        let path = format!("{SUBMISSION_DIR}/{slug}/{name}");
        let message = format!("data: community benchmark ({slug})");
        if put_file(token, login, branch, &path, json, &message)? {
            uploaded += 1;
        } else {
            skipped += 1;
        }
    }
    Ok((uploaded, skipped))
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

/// The OAuth App client id for the device flow: the `LLMFIT_GH_CLIENT_ID`
/// env override when set, otherwise the shipped default. `None` only when
/// the override is explicitly set to an empty string (opt-out of
/// interactive login, e.g. in restricted CI).
pub fn oauth_client_id() -> Option<String> {
    let id = std::env::var("LLMFIT_GH_CLIENT_ID")
        .unwrap_or_else(|_| DEFAULT_CLIENT_ID.to_string())
        .trim()
        .to_string();
    (!id.is_empty()).then_some(id)
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

/// Create `path` on `branch`. Returns `Ok(true)` when the file was written,
/// `Ok(false)` when it already exists there — creating over an existing path
/// makes GitHub answer 422 `"sha" wasn't supplied`, which for our
/// content-hash-named submissions means this exact result already landed in
/// an earlier attempt.
fn put_file(
    token: &str,
    login: &str,
    branch: &str,
    path: &str,
    content: &str,
    message: &str,
) -> Result<bool, String> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(content);
    let url = format!("{API}/repos/{login}/{UPSTREAM_REPO}/contents/{path}");
    let body = json!({
        "message": message,
        "content": encoded,
        "branch": branch,
    });
    let (status, body) = api("PUT", &url, token, Some(&body))?;
    if status == 422 && body["message"].as_str().is_some_and(|m| m.contains("sha")) {
        return Ok(false);
    }
    if !(200..300).contains(&status) {
        return Err(api_error(status, &body));
    }
    Ok(true)
}

/// First open upstream PR whose head is one of this user's `bench/…`
/// branches, as `(branch, html_url)`.
fn find_open_bench_pr(token: &str, login: &str) -> Result<Option<(String, String)>, String> {
    let url = format!("{API}/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/pulls?state=open&per_page=100");
    let (status, body) = api("GET", &url, token, None)?;
    if !(200..300).contains(&status) {
        return Err(api_error(status, &body));
    }
    Ok(open_bench_pr_in(&body, login))
}

/// Pure matcher over a GitHub pull-list response: this user's first open
/// `bench/…` PR, if any.
fn open_bench_pr_in(prs: &Value, login: &str) -> Option<(String, String)> {
    let prefix = format!("{login}:bench/");
    prs.as_array()?.iter().find_map(|pr| {
        let label = pr["head"]["label"].as_str()?;
        if !label.starts_with(&prefix) {
            return None;
        }
        Some((
            pr["head"]["ref"].as_str()?.to_string(),
            pr["html_url"].as_str()?.to_string(),
        ))
    })
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
    fn submission_declares_a_4gb_card_as_4gb() {
        // End-to-end on the payload, not just the helper: a 4 GB card used to
        // be published as an 8 GB part. Guards the whole build_submission path.
        let submission = build_submission(
            &[sample_result()],
            &specs_with_vram("NVIDIA GeForce GTX 1050 Ti", 4.0),
        );
        let value = serde_json::to_value(&submission).unwrap();
        assert_eq!(value["hardware"]["vramGb"], 4.0);
        assert_eq!(value["hardware"]["memTierGb"], 4);
    }

    #[test]
    fn declared_mem_tier_has_a_floor() {
        // The ladder used to start at 8, so anything smaller was published as
        // an 8 GB part. These are the capacities that actually shipped below
        // it: GT 710 and GTX 1050 at 2, GTX 1060 3 GB, GTX 1050 Ti at 4,
        // GTX 1060 6 GB and RTX 2060 at 6.
        assert_eq!(declared_mem_tier(2.0), 2);
        assert_eq!(declared_mem_tier(3.0), 3);
        assert_eq!(declared_mem_tier(4.0), 4);
        assert_eq!(declared_mem_tier(6.0), 6);
    }

    #[test]
    fn declared_mem_tier_ties_resolve_downward() {
        // 7 GB is equidistant from 6 and 8; understating is the safe
        // direction for a declared capacity.
        assert_eq!(declared_mem_tier(7.0), 6);
    }

    #[test]
    fn declared_mem_tier_rounds_to_nearest() {
        assert_eq!(declared_mem_tier(23.9), 24);
        assert_eq!(declared_mem_tier(31.0), 32);
        assert_eq!(declared_mem_tier(7.5), 8);
    }

    #[test]
    fn hardware_identifiable_rejects_placeholder_gpu() {
        // The exact shape that produced the meaningless `bench/n-a-…` PR: a GPU
        // machine whose accelerator name could not be read.
        let na = json!({
            "hardware": { "hwClass": "DISCRETE_GPU", "hardwareName": "N/A", "cpu": "AMD Ryzen" }
        });
        assert!(!hardware_is_identifiable(&na));

        // A real GPU name is fine.
        let good = json!({
            "hardware": { "hwClass": "UNIFIED", "hardwareName": "Apple M3 Max", "cpu": "Apple M3 Max" }
        });
        assert!(hardware_is_identifiable(&good));

        // CPU-only machines are identified by their CPU.
        let cpu_only = json!({
            "hardware": { "hwClass": "CPU_ONLY", "hardwareName": null, "cpu": "Intel Core i9-14900K" }
        });
        assert!(hardware_is_identifiable(&cpu_only));
        let cpu_unknown = json!({
            "hardware": { "hwClass": "CPU_ONLY", "hardwareName": null, "cpu": "unknown" }
        });
        assert!(!hardware_is_identifiable(&cpu_unknown));
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
            gpu_available_gb: None,
            gpu_name: Some(name.to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: crate::hardware::GpuBackend::Cuda,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    fn specs_with_vram(name: &str, vram_gb: f64) -> SystemSpecs {
        SystemSpecs {
            gpu_vram_gb: Some(vram_gb),
            total_gpu_vram_gb: Some(vram_gb),
            ..specs_with_gpu(name)
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
    fn open_bench_pr_matcher_picks_own_bench_branch_only() {
        let prs = json!([
            // Someone else's bench PR — must not match.
            {"head": {"label": "otheruser:bench/rtx-4090-aaaa", "ref": "bench/rtx-4090-aaaa"},
             "html_url": "https://github.com/AlexsJones/llmfit/pull/1"},
            // Our PR but not a bench branch — must not match.
            {"head": {"label": "me:fix/typo", "ref": "fix/typo"},
             "html_url": "https://github.com/AlexsJones/llmfit/pull/2"},
            // Ours — match.
            {"head": {"label": "me:bench/rtx-4090-bbbb", "ref": "bench/rtx-4090-bbbb"},
             "html_url": "https://github.com/AlexsJones/llmfit/pull/3"},
        ]);
        assert_eq!(
            open_bench_pr_in(&prs, "me"),
            Some((
                "bench/rtx-4090-bbbb".to_string(),
                "https://github.com/AlexsJones/llmfit/pull/3".to_string()
            ))
        );
        assert_eq!(open_bench_pr_in(&prs, "nobody"), None);
        assert_eq!(open_bench_pr_in(&json!([]), "me"), None);
        assert_eq!(
            open_bench_pr_in(&json!({"message": "rate limited"}), "me"),
            None
        );
    }

    #[test]
    fn prepare_files_restamps_and_keeps_stable_names() {
        let submission = build_submission(
            &[sample_result()],
            &specs_with_gpu("NVIDIA GeForce RTX 4090"),
        );
        let stored = StoredBenchmark {
            path: PathBuf::from("/store/pending/1752100000-abcd1234.json"),
            payload: serde_json::to_value(&submission).unwrap(),
        };

        let files = prepare_files(std::slice::from_ref(&stored), 9_999_999_999).unwrap();
        assert_eq!(files.len(), 1);
        let (slug, name, json) = &files[0];
        assert_eq!(slug, "nvidia-geforce-rtx-4090");
        // Upstream name mirrors the local store file, so retries are idempotent.
        assert_eq!(name, "1752100000-abcd1234.json");
        let payload: Value = serde_json::from_str(json).unwrap();
        assert_eq!(payload["submittedAtUnix"], 9_999_999_999u64);
        // Identical input at the same submit time → identical prepared file.
        let again = prepare_files(std::slice::from_ref(&stored), 9_999_999_999).unwrap();
        assert_eq!(&again[0].2, json);
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

        // #819: a payload written by an older binary can still carry an
        // absolute GGUF path in `model`. The store scrubs it at load time, so
        // the listing, the dry-run preview and the upload all see the file
        // name.
        let legacy = dir.join("pending").join("1700000000-legacy00.json");
        std::fs::write(
            &legacy,
            r#"{"schemaVersion":1,"results":[{"model":"/home/user/gguf/phi-4-Q4_K_M.gguf","provider":"llamacpp","avgTps":10.0}]}"#,
        )
        .unwrap();
        let with_legacy = pending_benchmarks();
        assert_eq!(with_legacy.len(), 2);
        assert_eq!(
            with_legacy[0].payload["results"][0]["model"],
            "phi-4-Q4_K_M.gguf"
        );
        std::fs::remove_file(&legacy).unwrap();

        // The local index resolves the stored run for the matching catalog
        // model (ollama tag "llama3.1:8b" ↔ HF-style name) and outranks
        // nothing else: unknown models get no local measurement.
        let idx = LocalBenchIndex::load(&specs).expect("store has one run");
        let m = idx.lookup("test/llama-3.1-8b").expect("tag should match");
        assert_eq!(m.tok_s, 128.44);
        assert_eq!(m.sample_count, 1);
        assert_eq!(m.source, crate::benchmarks::MeasuredSource::LocalBench);
        assert!(idx.lookup("test/qwen2.5-7b").is_none());

        // Runs recorded on different hardware never leak into the index.
        let other_gpu = specs_with_gpu("NVIDIA GeForce RTX 3060");
        assert!(LocalBenchIndex::load(&other_gpu).is_none());
        assert!(!pending[0].matches_hardware(&other_gpu));
        assert!(pending[0].matches_hardware(&specs));

        mark_shared(&pending);
        // Shared runs still count as local measurements.
        assert!(
            LocalBenchIndex::load(&specs)
                .and_then(|i| i.lookup("test/llama-3.1-8b"))
                .is_some()
        );
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

    #[test]
    fn strip_gguf_path_keeps_only_the_file_name_for_gguf_paths() {
        assert_eq!(
            strip_gguf_path("/home/user/gguf/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
            "SmolLM2-135M-Instruct-Q4_K_M.gguf"
        );
        assert_eq!(
            strip_gguf_path(r"C:\models\phi-4-Q4_K_M.GGUF"),
            "phi-4-Q4_K_M.GGUF"
        );
    }

    #[test]
    fn strip_gguf_path_leaves_non_path_ids_untouched() {
        // Ollama tag.
        assert_eq!(strip_gguf_path("llama3.1:8b"), "llama3.1:8b");
        // HF-style id from vLLM or MLX: contains a slash but is not a file.
        assert_eq!(
            strip_gguf_path("meta-llama/Llama-3.1-8B-Instruct"),
            "meta-llama/Llama-3.1-8B-Instruct"
        );
        // GGUF repo id: ends in "GGUF" but not ".gguf".
        assert_eq!(
            strip_gguf_path("unsloth/Qwen3-4B-GGUF"),
            "unsloth/Qwen3-4B-GGUF"
        );
        // Already a bare file name.
        assert_eq!(strip_gguf_path("model.gguf"), "model.gguf");
        // Hub-style reference: separators and a .gguf suffix, but relative,
        // so the org and repo context must survive.
        assert_eq!(
            strip_gguf_path(
                "hf.co/bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf"
            ),
            "hf.co/bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf"
        );
        // Relative filesystem path: nothing machine-specific to hide.
        assert_eq!(strip_gguf_path("models/foo.gguf"), "models/foo.gguf");
    }

    #[test]
    fn sanitize_stored_payload_scrubs_absolute_model_paths() {
        let mut payload = json!({
            "results": [
                { "model": "/home/alice/models/phi-4-Q4_K_M.gguf" },
                { "model": "llama3.1:8b" }
            ]
        });
        sanitize_stored_payload(&mut payload);
        assert_eq!(payload["results"][0]["model"], "phi-4-Q4_K_M.gguf");
        assert_eq!(payload["results"][1]["model"], "llama3.1:8b");
    }

    #[test]
    fn sanitize_stored_payload_ignores_non_object_payloads() {
        // read_store treats every pending file as untrusted: a hand-edited
        // file whose top level is not a JSON object must not panic the scrub.
        for mut payload in [json!("not an object"), json!([1, 2, 3]), Value::Null] {
            let before = payload.clone();
            sanitize_stored_payload(&mut payload);
            assert_eq!(payload, before);
        }
    }

    #[test]
    fn sanitize_stored_payload_leaves_objects_without_results_untouched() {
        // Indexing through IndexMut would insert a null `results` key here.
        let mut payload = json!({ "hardware": { "gpuModel": "RTX 2080" } });
        sanitize_stored_payload(&mut payload);
        assert_eq!(payload, json!({ "hardware": { "gpuModel": "RTX 2080" } }));
    }

    // #819: llama-server reports its `-m` argument, an absolute GGUF path, as
    // the model id. The stored payload must carry the bare file name, asserted
    // on the serialised JSON so the whole `build_submission` path is covered,
    // not just the helper.
    #[test]
    fn submission_model_strips_gguf_path_end_to_end() {
        let mut result = sample_result();
        result.model = "/home/user/gguf/SmolLM2-135M-Instruct-Q4_K_M.gguf".to_string();
        result.provider = "llamacpp".to_string();
        let payload =
            serde_json::to_value(build_submission(&[result], &specs_with_gpu("GTX 1050 Ti")))
                .unwrap();
        assert_eq!(
            payload["results"][0]["model"],
            "SmolLM2-135M-Instruct-Q4_K_M.gguf"
        );
    }
}
