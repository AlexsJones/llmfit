//! Hardware diagnostic dump for bug reports.
//!
//! `llmfit doctor` captures the raw output of every external tool the GPU
//! detection paths in [`crate::hardware`] shell out to, alongside what llmfit
//! actually detected. Users paste the dump into GitHub issues; each report
//! then doubles as a parser regression fixture (the verbatim tool output can
//! be dropped straight into `hardware.rs` tests).

use crate::hardware::SystemSpecs;
use std::fmt::Write as _;

/// Cap each captured section so a pathological tool can't flood the report.
const MAX_SECTION_BYTES: usize = 16 * 1024;

/// Run `cmd args…` and return its combined stdout/stderr, or a note that the
/// tool is unavailable. Never fails: missing tools are part of the diagnosis.
fn capture(cmd: &str, args: &[&str]) -> String {
    match std::process::Command::new(cmd).args(args).output() {
        Ok(out) => {
            let mut text = String::new();
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if !stdout.trim().is_empty() {
                text.push_str(stdout.trim_end());
            }
            if !stderr.trim().is_empty() {
                if !text.is_empty() {
                    text.push_str("\n--- stderr ---\n");
                }
                text.push_str(stderr.trim_end());
            }
            if text.is_empty() {
                text = format!("(no output, exit status: {})", out.status);
            }
            truncate(text)
        }
        Err(e) => format!("(not available: {e})"),
    }
}

fn truncate(mut text: String) -> String {
    if text.len() > MAX_SECTION_BYTES {
        // Truncate on a char boundary at or below the cap.
        let mut cut = MAX_SECTION_BYTES;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
        text.push_str("\n… (truncated)");
    }
    text
}

fn section(report: &mut String, title: &str, body: &str) {
    let _ = writeln!(report, "## {title}\n```\n{body}\n```\n");
}

/// Walk `/sys/class/drm/card*` and report the fields the sysfs detection
/// paths read: vendor, device id, driver, and dedicated VRAM if exposed.
fn sysfs_drm_summary() -> String {
    let entries = match std::fs::read_dir("/sys/class/drm") {
        Ok(e) => e,
        Err(e) => return format!("(not available: {e})"),
    };
    let mut cards: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let device = path.join("device");
        let read = |f: &str| {
            std::fs::read_to_string(device.join(f))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| "-".to_string())
        };
        let driver = std::fs::read_to_string(device.join("uevent"))
            .ok()
            .and_then(|u| {
                u.lines()
                    .find(|l| l.starts_with("DRIVER="))
                    .map(|l| l.trim_start_matches("DRIVER=").to_string())
            })
            .unwrap_or_else(|| "-".to_string());
        cards.push(format!(
            "{name}: vendor={} device={} driver={driver} mem_info_vram_total={}",
            read("vendor"),
            read("device"),
            read("mem_info_vram_total"),
        ));
    }
    if cards.is_empty() {
        "(no /sys/class/drm cardN entries)".to_string()
    } else {
        cards.sort();
        cards.join("\n")
    }
}

/// Build the full diagnostic report as Markdown.
///
/// `version` is the binary version string (core doesn't know the crate
/// version of the caller).
pub fn collect_diagnostics(version: &str) -> String {
    let mut report = String::new();
    let _ = writeln!(report, "# llmfit doctor report\n");
    let _ = writeln!(
        report,
        "Paste this whole report into a GitHub issue at \
         https://github.com/AlexsJones/llmfit/issues — the raw tool output \
         below is what lets detection bugs become regression tests. It \
         contains hardware model names and driver info only.\n"
    );
    let _ = writeln!(
        report,
        "- llmfit version: {version}\n- OS: {} ({})\n",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    // What llmfit concluded — shown first so mismatches with the raw
    // output below are immediately visible.
    let specs = SystemSpecs::detect();
    section(&mut report, "Detected by llmfit", &format!("{specs:#?}"));

    // NVIDIA
    section(
        &mut report,
        "nvidia-smi (extended query)",
        &capture(
            "nvidia-smi",
            &[
                "--query-gpu=addressing_mode,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
        ),
    );
    section(
        &mut report,
        "nvidia-smi (standard query)",
        &capture(
            "nvidia-smi",
            &[
                "--query-gpu=memory.total,name",
                "--format=csv,noheader,nounits",
            ],
        ),
    );

    // AMD ROCm
    section(
        &mut report,
        "rocm-smi --showmeminfo vram",
        &capture("rocm-smi", &["--showmeminfo", "vram"]),
    );
    section(
        &mut report,
        "rocm-smi --showproductname",
        &capture("rocm-smi", &["--showproductname"]),
    );

    if cfg!(target_os = "linux") {
        section(&mut report, "sysfs DRM cards", &sysfs_drm_summary());
        section(&mut report, "lspci (display controllers)", &{
            let full = capture("lspci", &["-nn"]);
            let filtered: Vec<&str> = full
                .lines()
                .filter(|l| {
                    let lower = l.to_lowercase();
                    lower.contains("vga")
                        || lower.contains("3d controller")
                        || lower.contains("display controller")
                        || lower.starts_with("(not available")
                })
                .collect();
            if filtered.is_empty() {
                "(no display controllers listed)".to_string()
            } else {
                filtered.join("\n")
            }
        });
    }

    if cfg!(target_os = "macos") {
        section(
            &mut report,
            "system_profiler SPDisplaysDataType",
            &capture("system_profiler", &["SPDisplaysDataType"]),
        );
    }

    if cfg!(target_os = "windows") {
        section(
            &mut report,
            "PowerShell Win32_VideoController",
            &capture(
                "powershell",
                &[
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }",
                ],
            ),
        );
        // AdapterRAM above is a uint32 and saturates at ~4 GB, so it cannot
        // confirm the size of any modern card. This registry dump carries the
        // driver-published 64-bit value that detection actually uses (#830).
        section(
            &mut report,
            "PowerShell display adapter registry VRAM",
            &capture(
                "powershell",
                &[
                    "-NoProfile",
                    "-Command",
                    crate::hardware::WINDOWS_REGISTRY_VRAM_PS_COMMAND,
                ],
            ),
        );
    }

    // Vulkan (fallback path on Linux/Windows) and NPUs — cheap to include
    // everywhere; reported as unavailable where the tool is missing.
    section(
        &mut report,
        "vulkaninfo --summary",
        &capture("vulkaninfo", &["--summary"]),
    );
    section(&mut report, "npu-smi info", &capture("npu-smi", &["info"]));

    // Provider applications that are detected by install location rather
    // than a live API probe (#731) — lets "installed but not running"
    // reports carry the evidence.
    section(
        &mut report,
        "Provider app installs",
        &format!(
            "LM Studio installed: {}\nDocker Desktop installed: {}\nollama on PATH: {}",
            crate::providers::lmstudio_app_installed(),
            crate::providers::docker_desktop_installed(),
            crate::providers::command_exists("ollama"),
        ),
    );

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_missing_tool_is_note_not_panic() {
        let out = capture("definitely-not-a-real-tool-xyz", &[]);
        assert!(out.starts_with("(not available:"), "{out}");
    }

    #[test]
    fn test_truncate_caps_section_and_marks_it() {
        let big = "x".repeat(MAX_SECTION_BYTES + 100);
        let out = truncate(big);
        assert!(out.len() <= MAX_SECTION_BYTES + 20);
        assert!(out.ends_with("(truncated)"));
    }

    #[test]
    fn test_report_contains_key_sections() {
        let report = collect_diagnostics("0.0.0-test");
        assert!(report.contains("# llmfit doctor report"));
        assert!(report.contains("llmfit version: 0.0.0-test"));
        assert!(report.contains("## Detected by llmfit"));
        assert!(report.contains("## rocm-smi --showmeminfo vram"));
        assert!(report.contains("## nvidia-smi (extended query)"));
    }
}
