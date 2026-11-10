use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    reserve_windows_main_stack();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let dist_dir = manifest_dir.join("..").join("llmfit-web").join("dist");
    println!("cargo:rerun-if-changed={}", dist_dir.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let out_file = out_dir.join("web_assets.rs");

    let generated = if dist_dir.exists() {
        let mut files = Vec::new();
        collect_files(&dist_dir, &mut files);
        files.sort();
        for file in &files {
            println!("cargo:rerun-if-changed={}", file.display());
        }
        generate_assets_from_dist(&dist_dir, &files)
    } else {
        println!(
            "cargo:warning=llmfit-web/dist not found. Falling back to placeholder embedded dashboard. Run `npm ci && npm run build` in llmfit-web."
        );
        generate_fallback_assets()
    };

    fs::write(&out_file, generated).expect("failed to write generated web_assets.rs");
}

/// Give the Windows binary the 8 MB main-thread stack Linux and macOS already
/// have. Windows defaults to 1 MB, and a debug build of `llmfit --help` needs
/// about 1 MB just to build the clap command tree, so each new flag or
/// subcommand risks a stack overflow on Windows alone.
fn reserve_windows_main_stack() {
    const STACK_BYTES: u32 = 8 * 1024 * 1024;
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/STACK:{STACK_BYTES}");
    } else {
        println!("cargo:rustc-link-arg-bins=-Wl,--stack,{STACK_BYTES}");
    }
}

fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|_| panic!("failed to read {}", dir.display()));

    for entry in entries {
        let entry = entry.expect("invalid dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn generate_assets_from_dist(dist_dir: &Path, files: &[PathBuf]) -> String {
    let mut output = String::new();
    output.push_str("#[derive(Clone, Copy)]\n");
    output.push_str("pub(crate) struct EmbeddedAsset {\n");
    output.push_str("    pub(crate) path: &'static str,\n");
    output.push_str("    pub(crate) content_type: &'static str,\n");
    output.push_str("    pub(crate) bytes: &'static [u8],\n");
    output.push_str("}\n\n");
    output.push_str("pub(crate) static EMBEDDED_WEB_ASSETS: &[EmbeddedAsset] = &[\n");

    for file in files {
        let relative = file
            .strip_prefix(dist_dir)
            .unwrap_or_else(|_| panic!("{} not under dist dir", file.display()));
        let route_path = format!("/{}", relative.to_string_lossy().replace('\\', "/"));
        let include_path = file.to_string_lossy();
        let content_type = content_type_for_path(file);

        output.push_str(&format!(
            "    EmbeddedAsset {{ path: {route_path:?}, content_type: {content_type:?}, bytes: include_bytes!({include_path:?}) }},\n"
        ));
    }

    output.push_str("];\n");
    output
}

fn generate_fallback_assets() -> String {
    let mut output = String::new();
    output.push_str("#[derive(Clone, Copy)]\n");
    output.push_str("pub(crate) struct EmbeddedAsset {\n");
    output.push_str("    pub(crate) path: &'static str,\n");
    output.push_str("    pub(crate) content_type: &'static str,\n");
    output.push_str("    pub(crate) bytes: &'static [u8],\n");
    output.push_str("}\n\n");
    output.push_str("const FALLBACK_INDEX_HTML: &[u8] = br#\"<!doctype html>\n");
    output.push_str("<html lang=\\\"en\\\">\n");
    output.push_str("  <head><meta charset=\\\"UTF-8\\\"/><meta name=\\\"viewport\\\" content=\\\"width=device-width, initial-scale=1.0\\\"/><title>llmfit</title></head>\n");
    output.push_str("  <body style=\\\"font-family: sans-serif; padding: 24px\\\">\n");
    output.push_str("    <h1>llmfit Web Dashboard</h1>\n");
    output.push_str("    <p>Frontend assets are missing.</p>\n");
    output.push_str(
        "    <p>From repo root run: <code>cd llmfit-web && npm ci && npm run build</code></p>\n",
    );
    output.push_str("    <script src=\\\"/assets/fallback.js\\\"></script>\n");
    output.push_str("  </body>\n");
    output.push_str("</html>\"#;\n");
    output.push_str("const FALLBACK_JS: &[u8] = br\"console.warn('llmfit-web dist assets not found; serving fallback page.');\";\n\n");
    output.push_str("pub(crate) static EMBEDDED_WEB_ASSETS: &[EmbeddedAsset] = &[\n");
    output.push_str("    EmbeddedAsset { path: \"/index.html\", content_type: \"text/html; charset=utf-8\", bytes: FALLBACK_INDEX_HTML },\n");
    output.push_str("    EmbeddedAsset { path: \"/assets/fallback.js\", content_type: \"text/javascript; charset=utf-8\", bytes: FALLBACK_JS },\n");
    output.push_str("];\n");
    output
}

fn content_type_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") | Some("map") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}
