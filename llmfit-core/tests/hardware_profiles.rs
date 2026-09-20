//! Gate the bundled hardware profiles: `data/hardware/*.json` is embedded by
//! build.rs, and an invalid file there is dropped at load time — so without
//! this test a malformed contribution would ship as a profile nobody can
//! select, with no build failure to point at it.

use jsonschema::Validator;
use llmfit_core::hwprofile::HardwareProfile;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn hardware_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data/hardware")
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

/// Every profile file, excluding the schema that lives alongside them.
fn profile_paths() -> Vec<PathBuf> {
    let dir = hardware_dir();
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("json"))
        .filter(|path| path.file_stem().and_then(|s| s.to_str()) != Some("schema"))
        .collect();
    paths.sort();
    paths
}

fn validator() -> Validator {
    let schema = read_json(&hardware_dir().join("schema.json"));
    jsonschema::validator_for(&schema)
        .expect("schema itself is invalid — check llmfit-core/data/hardware/schema.json")
}

#[test]
fn bundled_profiles_match_schema() {
    let validator = validator();
    let paths = profile_paths();
    assert!(paths.len() >= 2, "expected at least two seed profiles");

    for path in paths {
        let data = read_json(&path);
        let errors: Vec<String> = validator
            .iter_errors(&data)
            .take(30)
            .map(|e| format!("  [{}]  {}", e.instance_path(), e))
            .collect();
        assert!(
            errors.is_empty(),
            "{}: {} schema violation(s):\n{}",
            path.display(),
            errors.len(),
            errors.join("\n")
        );
    }
}

#[test]
fn bundled_profile_names_match_file_stems_and_validate_strictly() {
    for path in profile_paths() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let profile =
            HardwareProfile::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        profile
            .check_name_matches_stem(&path)
            .unwrap_or_else(|e| panic!("{e}"));
        profile
            .validate_strict()
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}

#[test]
fn every_bundled_profile_is_embedded_and_selectable() {
    let on_disk: Vec<String> = profile_paths()
        .iter()
        .filter_map(|path| path.file_stem().and_then(|s| s.to_str()).map(String::from))
        .collect();
    let embedded: Vec<String> = llmfit_core::hwprofile::embedded()
        .iter()
        .map(|profile| profile.name.clone())
        .collect();

    assert_eq!(
        on_disk, embedded,
        "build.rs embed drifted from data/hardware/"
    );
    for name in &on_disk {
        llmfit_core::hwprofile::resolve(name)
            .unwrap_or_else(|e| panic!("bundled profile {name} does not resolve: {e}"));
    }
}
