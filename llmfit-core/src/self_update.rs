//! Self-update check for the `llmfit` binary itself.
//!
//! This is deliberately separate from [`crate::update`], which refreshes the
//! *model database* cache — a totally different concern with a name that
//! unfortunately collides in spirit. This module only answers "is a newer
//! `llmfit` release available, and how should the user get it?" It never
//! downloads or replaces the running binary: `llmfit` ships through too many
//! package managers (Homebrew, MacPorts, `uv`/`pip`, Docker, `cargo install`,
//! a prebuilt-binary install script, and building from source) for a single
//! self-mutating code path to be safe or honest about what it's touching —
//! most of those are owned by a package manager that would silently
//! overwrite/reconcile any file we swapped in ourselves.
//!
//! Set `LLMFIT_NO_UPDATE_CHECK=1` to skip the check entirely (CI, Docker,
//! air-gapped hosts) — every caller, including `llmfit doctor`, is expected
//! to honor [`background_check_disabled`] before calling
//! [`check_for_update`] or [`fetch_latest_version`].

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// GitHub releases API endpoint for the upstream repo. Used instead of
/// scraping the HTML releases page or crates.io (which lags behind for
/// non-`cargo install` users).
const RELEASES_API: &str = "https://api.github.com/repos/AlexsJones/llmfit/releases/latest";

/// Env var that disables the opportunistic background check.
pub const NO_UPDATE_CHECK_ENV: &str = "LLMFIT_NO_UPDATE_CHECK";

/// How the running `llmfit` binary most likely got onto this machine.
///
/// Detection is heuristic (it inspects the running executable's path) and
/// best-effort: on ambiguity this prefers [`InstallMethod::Unknown`] over
/// guessing wrong and pointing the user at a command that won't work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMethod {
    /// Homebrew (own tap or homebrew-core): path contains `/Cellar/` or
    /// `/Caskroom/`.
    Homebrew,
    /// MacPorts: path is under `/opt/local/`.
    MacPorts,
    /// `cargo install`: path is under `~/.cargo/bin/`.
    CargoInstall,
    /// `uv tool install`: path is under a `uv`-managed tool directory.
    UvTool,
    /// Running inside a container (`docker run ghcr.io/alexsjones/llmfit`).
    Docker,
    /// The prebuilt-binary install script (`install.sh`), which places the
    /// binary in `~/.local/bin` or `/usr/local/bin` outside any of the
    /// above package managers.
    InstallerScript,
    /// Couldn't tell — most likely a from-source build (`cargo build
    /// --release`) run directly out of `target/`.
    Unknown,
}

impl InstallMethod {
    /// The command to hand the user so *they* apply the upgrade. `llmfit`
    /// never runs this itself — see the module docs for why.
    pub fn upgrade_hint(self) -> &'static str {
        match self {
            InstallMethod::Homebrew => "brew upgrade llmfit",
            InstallMethod::MacPorts => "sudo port upgrade llmfit",
            InstallMethod::CargoInstall => "cargo install llmfit --force",
            InstallMethod::UvTool => "uv tool upgrade llmfit",
            InstallMethod::Docker => "docker pull ghcr.io/alexsjones/llmfit",
            InstallMethod::InstallerScript => "curl -fsSL https://llmfit.axjns.dev/install.sh | sh",
            InstallMethod::Unknown => {
                "rebuild from source (cargo build --release), or reinstall via \
                 whichever method you originally used — see \
                 https://github.com/AlexsJones/llmfit#installation"
            }
        }
    }
}

/// Detect the install method from the running executable's own path.
///
/// Never fails: an unresolvable `current_exe()` (sandboxing, deleted binary)
/// falls back to [`InstallMethod::Unknown`] rather than erroring, matching
/// this module's "best-effort, never block the user" stance.
pub fn detect_install_method() -> InstallMethod {
    // Checked ahead of the path heuristics below: inside a container the
    // binary's own path (e.g. `/usr/local/bin/llmfit`) looks identical to a
    // plain install.sh install, but `/.dockerenv` disambiguates it.
    if Path::new("/.dockerenv").exists() {
        return InstallMethod::Docker;
    }
    let Ok(exe) = std::env::current_exe() else {
        return InstallMethod::Unknown;
    };
    detect_install_method_from_path(&exe)
}

/// Path-only half of [`detect_install_method`], split out so tests can drive
/// it with arbitrary paths without touching the real filesystem.
fn detect_install_method_from_path(exe: &Path) -> InstallMethod {
    let s = exe.to_string_lossy();
    if s.contains("/Cellar/") || s.contains("/Caskroom/") || s.contains("/homebrew/") {
        InstallMethod::Homebrew
    } else if s.contains("/opt/local/") {
        InstallMethod::MacPorts
    } else if s.contains("/.cargo/bin/") {
        InstallMethod::CargoInstall
    } else if s.contains("/uv/tools/") || s.contains("/.local/share/uv/") {
        InstallMethod::UvTool
    } else if s.contains("/.local/bin/") || s == "/usr/local/bin/llmfit" {
        InstallMethod::InstallerScript
    } else {
        InstallMethod::Unknown
    }
}

/// Parsed `major.minor.patch`. Anything beyond three components (build
/// metadata, pre-release suffixes) is ignored for comparison purposes —
/// release-please tags this project as plain `vX.Y.Z`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SimpleVersion(u64, u64, u64);

/// `true` if `latest` parses to a strictly newer version than `current`.
/// Either side failing to parse (unexpected version string shape) counts as
/// "not behind" — we'd rather stay silent than nag over a parse quirk.
fn is_newer(current: &str, latest: Option<&str>) -> bool {
    match (parse_version(current), latest.and_then(parse_version)) {
        (Some(current), Some(latest)) => latest > current,
        _ => false,
    }
}

fn parse_version(v: &str) -> Option<SimpleVersion> {
    let v = v.trim().trim_start_matches('v');
    // Drop any `-rc.1` / `+build` suffix before splitting on '.'.
    let core = v.split(['-', '+']).next().unwrap_or(v);
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some(SimpleVersion(major, minor, patch))
}

/// Result of comparing the running version against the latest release.
#[derive(Debug, Clone)]
pub struct VersionCheck {
    pub current: String,
    /// `None` if the latest-version lookup failed (offline, rate-limited,
    /// GitHub outage) — never treated as "you're behind".
    pub latest: Option<String>,
    pub is_behind: bool,
    pub install_method: InstallMethod,
}

impl VersionCheck {
    /// One-line human summary, or `None` when there's nothing worth telling
    /// the user (up to date, or the lookup failed).
    pub fn notice(&self) -> Option<String> {
        if !self.is_behind {
            return None;
        }
        let latest = self.latest.as_deref()?;
        Some(format!(
            "llmfit {} is available (you have {}). Upgrade with: {}",
            latest,
            self.current,
            self.install_method.upgrade_hint()
        ))
    }
}

#[derive(serde::Deserialize)]
struct GithubRelease {
    tag_name: String,
}

/// Fetch the latest published release's version string (without the `v`
/// prefix) from the GitHub releases API. Best-effort: any network or parse
/// failure is folded into `Err` with a short reason, never a panic.
pub fn fetch_latest_version() -> Result<String, String> {
    let resp = ureq::get(RELEASES_API)
        .header("User-Agent", "llmfit-self-update-check")
        .config()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .call()
        .map_err(|e| format!("could not reach GitHub releases API: {e}"))?;
    let release = resp
        .into_body()
        .read_json::<GithubRelease>()
        .map_err(|e| format!("could not parse GitHub releases response: {e}"))?;
    Ok(release.tag_name.trim_start_matches('v').to_string())
}

/// Compare `current_version` against the latest published release and
/// report the install method the user should use to upgrade.
///
/// Never fails or blocks indefinitely: a failed lookup yields
/// `latest: None, is_behind: false` rather than propagating an error, since
/// this runs opportunistically and must never be load-bearing for any other
/// command.
pub fn check_for_update(current_version: &str) -> VersionCheck {
    let install_method = detect_install_method();
    let latest = fetch_latest_version().ok();
    let is_behind = is_newer(current_version, latest.as_deref());
    VersionCheck {
        current: current_version.to_string(),
        latest,
        is_behind,
        install_method,
    }
}

/// `true` if the opportunistic background check should be skipped, per
/// [`NO_UPDATE_CHECK_ENV`]. Every caller — including `llmfit doctor` — is
/// expected to check this before hitting the network.
pub fn background_check_disabled() -> bool {
    std::env::var(NO_UPDATE_CHECK_ENV).is_ok_and(|v| v != "0" && !v.is_empty())
}

// ── Cached, rate-limited background check ─────────────────────────────────

/// Don't hit the GitHub API more than once per day from routine command
/// invocations. `llmfit doctor` bypasses this (it wants a live answer for a
/// bug report); the opportunistic startup notice does not.
const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(serde::Serialize, serde::Deserialize)]
struct CheckCache {
    /// Unix seconds of the last successful (or attempted) lookup.
    checked_at: u64,
    /// Latest known version, if the last lookup succeeded.
    latest: Option<String>,
}

fn cache_file() -> Option<PathBuf> {
    Some(
        dirs::data_dir()?
            .join("llmfit")
            .join("self_update_cache.json"),
    )
}

fn read_cache() -> Option<CheckCache> {
    let path = cache_file()?;
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn write_cache(cache: &CheckCache) {
    let Some(path) = cache_file() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(bytes) = serde_json::to_vec(cache) {
        let _ = std::fs::write(path, bytes);
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Opportunistic, cached version check for interactive startup paths (the
/// TUI, or a CLI banner). Returns `None` when the check is disabled, the
/// cache is still fresh and says "up to date", or nothing is worth telling
/// the user yet — callers can treat `None` as "print nothing" unconditionally.
///
/// Unlike [`check_for_update`], this hits the network at most once per
/// [`CHECK_INTERVAL`]; a fresh cache entry is trusted without a fresh
/// request. A failed lookup still updates `checked_at` (so a flaky network
/// doesn't turn into a retry-every-launch loop) but leaves `latest` as
/// whatever was last known.
pub fn check_for_update_cached(current_version: &str) -> Option<VersionCheck> {
    if background_check_disabled() {
        return None;
    }
    let cached = read_cache();
    let is_fresh = cached
        .as_ref()
        .is_some_and(|c| now_secs().saturating_sub(c.checked_at) < CHECK_INTERVAL.as_secs());

    let latest = if is_fresh {
        cached.and_then(|c| c.latest)
    } else {
        let fetched = fetch_latest_version().ok();
        write_cache(&CheckCache {
            checked_at: now_secs(),
            latest: fetched.clone().or_else(|| cached.and_then(|c| c.latest)),
        });
        fetched
    };

    let install_method = detect_install_method();
    let is_behind = is_newer(current_version, latest.as_deref());
    if !is_behind {
        return None;
    }
    Some(VersionCheck {
        current: current_version.to_string(),
        latest,
        is_behind,
        install_method,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_version_basic() {
        assert_eq!(parse_version("1.1.10"), Some(SimpleVersion(1, 1, 10)));
        assert_eq!(parse_version("v1.1.10"), Some(SimpleVersion(1, 1, 10)));
    }

    #[test]
    fn test_parse_version_with_suffixes() {
        assert_eq!(parse_version("1.2.0-rc.1"), Some(SimpleVersion(1, 2, 0)));
        assert_eq!(parse_version("1.2.0+abcdef"), Some(SimpleVersion(1, 2, 0)));
    }

    #[test]
    fn test_parse_version_missing_components_default_to_zero() {
        assert_eq!(parse_version("2"), Some(SimpleVersion(2, 0, 0)));
        assert_eq!(parse_version("2.5"), Some(SimpleVersion(2, 5, 0)));
    }

    #[test]
    fn test_parse_version_invalid() {
        assert_eq!(parse_version("not-a-version"), None);
        assert_eq!(parse_version(""), None);
    }

    #[test]
    fn test_version_ordering() {
        assert!(parse_version("1.2.0") < parse_version("1.10.0"));
        assert!(parse_version("1.1.238") < parse_version("1.1.239"));
        assert!(parse_version("2.0.0") > parse_version("1.99.99"));
    }

    #[test]
    fn test_detect_install_method_homebrew_cellar() {
        assert_eq!(
            detect_install_method_from_path(Path::new(
                "/opt/homebrew/Cellar/llmfit/1.1.10/bin/llmfit"
            )),
            InstallMethod::Homebrew
        );
    }

    #[test]
    fn test_detect_install_method_cargo() {
        assert_eq!(
            detect_install_method_from_path(Path::new("/Users/tmf/.cargo/bin/llmfit")),
            InstallMethod::CargoInstall
        );
    }

    #[test]
    fn test_detect_install_method_installer_script() {
        assert_eq!(
            detect_install_method_from_path(Path::new("/Users/tmf/.local/bin/llmfit")),
            InstallMethod::InstallerScript
        );
    }

    #[test]
    fn test_detect_install_method_unknown_fallback() {
        assert_eq!(
            detect_install_method_from_path(Path::new(
                "/Users/tmf/Documents/llmfit/target/release/llmfit"
            )),
            InstallMethod::Unknown
        );
    }

    #[test]
    fn test_notice_when_up_to_date_is_none() {
        let vc = VersionCheck {
            current: "1.1.10".to_string(),
            latest: Some("1.1.10".to_string()),
            is_behind: false,
            install_method: InstallMethod::Homebrew,
        };
        assert_eq!(vc.notice(), None);
    }

    #[test]
    fn test_notice_when_behind_mentions_both_versions_and_command() {
        let vc = VersionCheck {
            current: "1.1.9".to_string(),
            latest: Some("1.1.10".to_string()),
            is_behind: true,
            install_method: InstallMethod::Homebrew,
        };
        let notice = vc.notice().expect("should have a notice");
        assert!(notice.contains("1.1.10"));
        assert!(notice.contains("1.1.9"));
        assert!(notice.contains("brew upgrade llmfit"));
    }

    #[test]
    fn test_notice_when_latest_lookup_failed_is_none_even_if_flagged_behind() {
        // Defensive: is_behind should never be true without a `latest`, but
        // notice() must not panic/unwrap its way into a bad message either way.
        let vc = VersionCheck {
            current: "1.1.9".to_string(),
            latest: None,
            is_behind: true,
            install_method: InstallMethod::Unknown,
        };
        assert_eq!(vc.notice(), None);
    }

    #[test]
    fn test_background_check_disabled() {
        // SAFETY: tests run single-threaded within this process by default
        // for env-mutating tests is not guaranteed by cargo test, so keep
        // this scoped to values that don't leak assumptions across tests.
        unsafe {
            std::env::remove_var(NO_UPDATE_CHECK_ENV);
        }
        assert!(!background_check_disabled());
        unsafe {
            std::env::set_var(NO_UPDATE_CHECK_ENV, "1");
        }
        assert!(background_check_disabled());
        unsafe {
            std::env::set_var(NO_UPDATE_CHECK_ENV, "0");
        }
        assert!(!background_check_disabled());
        unsafe {
            std::env::remove_var(NO_UPDATE_CHECK_ENV);
        }
    }
}
