//! Aggregate community benchmark submissions (data/community/<slug>/*.json)
//! and hardware profiles (data/hardware/*.json) into single JSON arrays
//! embedded in the binary. This is what closes the contribution loop: a
//! submission merged into the repo ships to every user in the next release,
//! with no CI step or network fetch involved.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=data/community");
    println!("cargo:rerun-if-changed=data/hardware");

    embed_community_benchmarks();
    embed_hardware_profiles();
}

fn embed_community_benchmarks() {
    let community_dir = Path::new("data/community");
    let mut files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(community_dir) {
        for entry in entries.flatten() {
            let slug_dir = entry.path();
            if !slug_dir.is_dir() {
                continue; // README.md, schema.json
            }
            if let Ok(subs) = fs::read_dir(&slug_dir) {
                for sub in subs.flatten() {
                    let p = sub.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("json") {
                        files.push(p);
                    }
                }
            }
        }
    }
    // Deterministic embed order regardless of directory iteration order.
    files.sort();

    let mut payloads: Vec<serde_json::Value> = Vec::new();
    for f in &files {
        let Ok(text) = fs::read_to_string(f) else {
            println!(
                "cargo:warning=community submission unreadable, skipped: {}",
                f.display()
            );
            continue;
        };
        match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(v) => payloads.push(v),
            // CI validates submissions on PR; a bad file here should never
            // happen, but a warning beats breaking every build.
            Err(e) => println!(
                "cargo:warning=community submission invalid JSON, skipped: {}: {e}",
                f.display()
            ),
        }
    }

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"))
        .join("community_benchmarks.json");
    let json = serde_json::to_string(&payloads).expect("serialize community aggregate");
    fs::write(&out, json).expect("write community aggregate");
}

/// Aggregate `data/hardware/*.json` into one array, skipping the schema and
/// README that live alongside the profiles.
fn embed_hardware_profiles() {
    let hardware_dir = Path::new("data/hardware");
    let mut files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(hardware_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue; // README.md
            }
            if path.file_stem().and_then(|s| s.to_str()) == Some("schema") {
                continue; // schema.json is the contract, not a profile
            }
            files.push(path);
        }
    }
    // Deterministic embed order regardless of directory iteration order.
    files.sort();

    let mut payloads: Vec<serde_json::Value> = Vec::new();
    for f in &files {
        let Ok(text) = fs::read_to_string(f) else {
            println!(
                "cargo:warning=hardware profile unreadable, skipped: {}",
                f.display()
            );
            continue;
        };
        match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(v) => payloads.push(v),
            // `cargo test -p llmfit-core` validates this directory against
            // schema.json, so a bad file here should never reach a release; a
            // warning beats breaking every build.
            Err(e) => println!(
                "cargo:warning=hardware profile invalid JSON, skipped: {}: {e}",
                f.display()
            ),
        }
    }

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"))
        .join("hardware_profiles.json");
    let json = serde_json::to_string(&payloads).expect("serialize hardware profile aggregate");
    fs::write(&out, json).expect("write hardware profile aggregate");
}
