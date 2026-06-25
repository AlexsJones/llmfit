use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn cargo_config_exists() {
    let config_path = workspace_root().join(".cargo/config.toml");
    assert!(
        config_path.exists(),
        ".cargo/config.toml should exist at the workspace root"
    );
}

#[test]
fn cargo_config_pins_target_dir_inside_repo() {
    let config_path = workspace_root().join(".cargo/config.toml");
    let contents =
        fs::read_to_string(&config_path).expect("failed to read .cargo/config.toml");

    assert!(
        contents.contains("[build]"),
        ".cargo/config.toml should contain a [build] section"
    );
    assert!(
        contents.contains("target-dir = \"target\""),
        "build.target-dir should be pinned to \"target\""
    );
}
