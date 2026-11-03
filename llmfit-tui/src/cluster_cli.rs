//! `llmfit cluster` subcommands: interactive setup, status output, and clearing
//! the saved config. Cluster data and discovery live in `llmfit_core::cluster`;
//! this module owns the terminal prompts and printing.

use clap::Subcommand;
use std::io::{self, BufRead, Write};

use llmfit_core::cluster::{ClusterConfig, ClusterNode, RayEndpoint};

/// Sub-commands for `llmfit cluster`.
#[derive(Subcommand, Debug)]
pub(crate) enum ClusterAction {
    /// Initialize cluster configuration interactively (or via Ray Dashboard)
    Init {
        /// Ray Dashboard URL to auto-discover from (e.g. http://10.0.0.1:8265)
        #[arg(long)]
        ray_url: Option<String>,
    },
    /// Show current cluster configuration
    Status,
    /// Remove saved cluster configuration
    Clear,
}

/// Print an error for `llmfit cluster` and exit 1: a `{"error": ...}`
/// document on stdout under `--json`, plain text on stderr otherwise.
fn fail(json: bool, kind: &str, message: &str) -> ! {
    if json {
        crate::display::display_json_error(kind, message);
    } else {
        eprintln!("Error: {}", message);
    }
    std::process::exit(1);
}

/// The follow-up warnings for a config just discovered from Ray: what the
/// user still has to fill in before `--cluster` gives a meaningful answer.
fn discovery_warnings(cluster: &ClusterConfig) -> Vec<String> {
    let mut warnings = Vec::new();
    let shown_path = ClusterConfig::config_path().map_or_else(
        || "the cluster config".to_string(),
        |p| p.display().to_string(),
    );
    if cluster.total_gpu_count() == 0 {
        warnings.push(
            "Ray reports no GPUs on any node; `--cluster` needs GPU nodes and will refuse this config. \
             If Ray did not detect your GPUs, restart it with `ray start --num-gpus <N>` or set gpu_count per node."
                .to_string(),
        );
        return warnings;
    }
    if !cluster.zero_vram_gpu_nodes().is_empty() {
        warnings.push(format!(
            "Ray did not report per-GPU VRAM for some GPU nodes; the saved config has gpu_vram_gb = 0 for them. \
             Pass --memory <per-GPU VRAM> with `--cluster`, set gpu_vram_gb per node in {}, or re-run `llmfit cluster init` interactively.",
            shown_path
        ));
    }
    let unset = cluster.unset_backend_gpu_nodes();
    if !unset.is_empty() {
        warnings.push(format!(
            "Ray did not say which GPU vendor {} use{}; set `backend = \"cuda\"|\"rocm\"|\"metal\"|\"vulkan\"` per node in {}. \
             Until then `--cluster` assumes CUDA, which is wrong for AMD GPUs.",
            unset.join(", "),
            if unset.len() == 1 { "s" } else { "" },
            shown_path
        ));
    }
    warnings
}

/// The `cluster init --json` success document: the saved cluster plus the
/// warnings a terminal user would see.
fn init_json(cluster: &ClusterConfig, warnings: &[String]) -> serde_json::Value {
    let mut doc = cluster_json(cluster);
    doc["warnings"] = serde_json::json!(warnings);
    doc
}

/// "Cluster configured: 1 node, 0 GB total VRAM".
fn configured_line(cluster: &ClusterConfig) -> String {
    format!(
        "Cluster configured: {}, {:.0} GB total VRAM",
        llmfit_core::cluster::nodes_label(cluster.node_count()),
        cluster.total_vram_gb()
    )
}

/// Run a `llmfit cluster` subcommand. Exits the process with status 1 on error.
pub(crate) fn run(action: ClusterAction, json: bool) {
    match action {
        ClusterAction::Init { ray_url } => {
            let from_ray = ray_url.is_some();
            let cluster = match ray_url {
                // Non-interactive path: a --ray-url was supplied, so discover
                // the cluster straight from the Ray Dashboard API. Never drop
                // into interactive prompts here — fail loudly instead.
                Some(url) => {
                    warn_if_insecure(RayEndpoint::parse(&url).ok().as_ref());
                    let cluster = ClusterConfig::discover_from_ray_url(&url).unwrap_or_else(|e| {
                        fail(
                            json,
                            "cluster_discovery_failed",
                            &format!(
                                "could not initialize cluster from --ray-url '{}': {}",
                                url, e
                            ),
                        )
                    });
                    if let Err(e) = cluster.save() {
                        fail(json, "cluster_config_write_failed", &e);
                    }
                    cluster
                }
                None => {
                    interactive_init().unwrap_or_else(|e| fail(json, "cluster_init_failed", &e))
                }
            };
            // Only discovery leaves fields for the user to fill in; the
            // interactive prompts ask for each one.
            let warnings = if from_ray {
                discovery_warnings(&cluster)
            } else {
                Vec::new()
            };
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&init_json(&cluster, &warnings))
                        .expect("JSON serialization failed")
                );
            } else {
                println!("{}", configured_line(&cluster));
                for w in &warnings {
                    eprintln!("⚠️  {}", w);
                }
            }
        }
        ClusterAction::Status => match ClusterConfig::load() {
            Ok(Some(cluster)) if json => println!(
                "{}",
                serde_json::to_string_pretty(&cluster_json(&cluster))
                    .expect("JSON serialization failed")
            ),
            Ok(Some(cluster)) => print_cluster(&cluster),
            Ok(None) => fail(
                json,
                "cluster_not_configured",
                "No cluster configured. Run `llmfit cluster init` to set up.",
            ),
            Err(e) => fail(json, "cluster_config_invalid", &e),
        },
        ClusterAction::Clear => match ClusterConfig::remove_config() {
            Ok(()) if json => println!("{}", serde_json::json!({ "removed": true })),
            Ok(()) => println!("Cluster config removed."),
            Err(e) => fail(json, "cluster_config_remove_failed", &e),
        },
    }
}

/// Warn on stderr before discovery talks plain HTTP to a non-local host.
fn warn_if_insecure(endpoint: Option<&RayEndpoint>) {
    if let Some(w) = endpoint.and_then(RayEndpoint::insecure_warning) {
        eprintln!("⚠️  {}", w);
    }
}

/// Print cluster info to stdout.
fn print_cluster(cluster: &ClusterConfig) {
    println!();
    println!("  Cluster: {}", cluster.name);
    println!("  Nodes:   {}", cluster.node_count());
    println!();
    for node in &cluster.nodes {
        let role = if node.is_head { "HEAD" } else { "WORKER" };
        println!(
            "    {} ({}) — {} | {} | {}× GPU | {:.0} GB VRAM | {:.0} GB RAM | {} cores",
            node.hostname,
            role,
            node.ip,
            node.gpu_name,
            node.gpu_count,
            node.gpu_vram_gb,
            node.total_ram_gb,
            node.cpu_cores
        );
    }
    println!();
    println!("  Totals:");
    println!("    GPUs:     {}", cluster.total_gpu_count());
    println!("    VRAM:     {:.0} GB", cluster.total_vram_gb());
    println!("    RAM:      {:.0} GB", cluster.total_ram_gb());
    println!("    CPUs:     {} cores", cluster.total_cpu_cores());
    println!();
}

/// The `cluster status --json` document.
fn cluster_json(cluster: &ClusterConfig) -> serde_json::Value {
    serde_json::json!({
        "cluster": {
            "name": cluster.name,
            "node_count": cluster.node_count(),
            "head_ip": cluster.head_ip,
            "ray_port": cluster.ray_port,
            "total_gpus": cluster.total_gpu_count(),
            "total_ram_gb": cluster.total_ram_gb(),
            "total_vram_gb": cluster.total_vram_gb(),
            "total_cpu_cores": cluster.total_cpu_cores(),
            "nodes": cluster.nodes.iter().map(|n| serde_json::json!({
                "hostname": n.hostname,
                "ip": n.ip,
                "gpu": n.gpu_name,
                "gpu_count": n.gpu_count,
                "vram_gb": n.gpu_vram_gb,
                "ram_gb": n.total_ram_gb,
                "cpu_cores": n.cpu_cores,
                "is_head": n.is_head,
                "backend": n.backend.map(|b| b.label()),
            })).collect::<Vec<_>>(),
        }
    })
}

// ── Prompt helpers ─────────────────────────────────────────────────────────────

/// Print `prompt` (no newline), then read one line from stdin, trimmed.
fn read_line(prompt: &str) -> Result<String, String> {
    print!("{}", prompt);
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin()
        .lock()
        .read_line(&mut input)
        .map_err(|e| format!("Read error: {}", e))?;
    Ok(input.trim().to_string())
}

/// Prompt for a `u32` value with a default, warning on invalid non-empty input.
///
/// Empty input (pressing Enter) silently accepts the default.
/// A zero value for prompts where > 0 is required will warn and use the default.
fn prompt_u32(label: &str, default: u32) -> u32 {
    let input = read_line(&format!("  {} [{}]: ", label, default)).unwrap_or_default();
    match input.parse::<u32>() {
        Ok(n) if n > 0 => n,
        Ok(_) => {
            eprintln!("⚠️  Value must be > 0; using default {}", default);
            default
        }
        Err(_) if input.is_empty() => default,
        Err(_) => {
            eprintln!("⚠️  Invalid number '{}', using default {}", input, default);
            default
        }
    }
}

/// Prompt for a `f64` value with a default, warning on invalid non-empty input.
///
/// Empty input (pressing Enter) silently accepts the default.
/// Values < 0 warn and use the default.
fn prompt_f64(label: &str, default: f64) -> f64 {
    let input = read_line(&format!("  {} [{}]: ", label, default)).unwrap_or_default();
    if input.is_empty() {
        return default;
    }
    match input.parse::<f64>() {
        Ok(v) if v >= 0.0 => v,
        Ok(_) => {
            eprintln!("⚠️  Value must be >= 0; using default {}", default);
            default
        }
        Err(_) => {
            eprintln!("⚠️  Invalid number '{}', using default {}", input, default);
            default
        }
    }
}

/// Increment the last octet of an IPv4 address.
fn increment_ip(ip: &str, offset: u8) -> Result<String, String> {
    let parts: Vec<&str> = ip.rsplitn(2, '.').collect();
    if parts.len() == 2
        && let Ok(last_octet) = parts[0].parse::<u8>()
    {
        if let Some(new_octet) = last_octet.checked_add(offset) {
            return Ok(format!("{}.{}", parts[1], new_octet));
        } else {
            return Err(format!(
                "IP octet overflow: {}.{} + {} exceeds 255",
                parts[1], last_octet, offset
            ));
        }
    }
    Err(format!("Invalid IP address: {}", ip))
}

/// Validate an IP address or hostname string.
fn validate_ip_or_hostname(input: &str) -> Result<(), String> {
    if input.is_empty() || input.contains(' ') {
        return Err(format!("Invalid IP or hostname: '{}'", input));
    }
    // If it looks like an IP, validate each octet
    if input.chars().all(|c| c.is_ascii_digit() || c == '.') {
        let octets: Vec<&str> = input.split('.').collect();
        if octets.len() != 4 || octets.iter().any(|o| o.parse::<u8>().is_err()) {
            return Err(format!("Invalid IPv4 address: '{}'", input));
        }
    } else if !input
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '.')
    {
        return Err(format!("Invalid hostname: '{}'", input));
    }
    Ok(())
}

// ── Interactive cluster init ───────────────────────────────────────────────────

/// Save the config, then print where it went.
fn save_and_report(cluster: &ClusterConfig) -> Result<(), String> {
    cluster.save()?;
    if let Some(path) = ClusterConfig::config_path() {
        println!("  Saved to {}", path.display());
    }
    println!();
    Ok(())
}

/// Run the interactive cluster initialization flow.
/// Returns the created/updated `ClusterConfig`.
fn interactive_init() -> Result<ClusterConfig, String> {
    const DEFAULT_RAY_PORT: u16 = 8265;

    println!();
    println!("  === Remote GPU Cluster Setup ===");
    println!();

    // Step 1: Get head node IP
    // Interactive discovery talks plain HTTP to host:port. For an https://
    // dashboard (e.g. behind a TLS proxy) use `cluster init --ray-url`.
    println!(
        "  (Ray Dashboard over HTTPS? Use `llmfit cluster init --ray-url https://…` instead.)"
    );
    let head_ip = read_line("  Head node IP or hostname: ")?;
    if head_ip.is_empty() {
        return Err("Head node IP is required".to_string());
    }
    validate_ip_or_hostname(&head_ip)?;

    // Step 2: Try Ray Dashboard API
    let port_input = read_line(&format!("  Ray Dashboard port [{}]: ", DEFAULT_RAY_PORT))?;
    let ray_port: u16 = if port_input.is_empty() {
        DEFAULT_RAY_PORT
    } else {
        match port_input.parse::<u16>() {
            Ok(p) => p,
            Err(_) => {
                eprintln!(
                    "⚠️  Invalid port '{}', using default {}",
                    port_input, DEFAULT_RAY_PORT
                );
                DEFAULT_RAY_PORT
            }
        }
    };

    println!();
    println!(
        "  Connecting to Ray Dashboard at {}:{}...",
        head_ip, ray_port
    );

    warn_if_insecure(Some(&RayEndpoint {
        https: false,
        host: head_ip.clone(),
        port: ray_port,
    }));
    match ClusterConfig::discover_from_ray(&head_ip, ray_port) {
        Ok(mut cluster) => {
            println!("  Found {} node(s) via Ray API.", cluster.node_count());

            // Ray reports the GPU model and VRAM when the node's dashboard
            // agent can read GPU stats. Prompt once for the GPU nodes it
            // could not; CPU-only nodes (e.g. a head node) keep 0 GPUs / 0 GB.
            let gpu_nodes = cluster.nodes.iter().filter(|n| n.gpu_count > 0).count();
            let missing = cluster.zero_vram_gpu_nodes();
            if gpu_nodes > 0 {
                if missing.is_empty() {
                    println!("  Ray reported GPU model and VRAM for every GPU node.");
                } else {
                    // 0 would save a config `--cluster` then refuses.
                    let vram = Some(prompt_f64(
                        &format!(
                            "GPU VRAM per GPU for {} GPU node(s) Ray reported no VRAM for (GB)",
                            missing.len()
                        ),
                        80.0,
                    ))
                    .filter(|v| *v > 0.0)
                    .ok_or("GPU VRAM per GPU must be greater than 0")?;
                    for i in missing {
                        cluster.nodes[i].gpu_vram_gb = vram;
                    }
                }
                if cluster.nodes.len() > gpu_nodes {
                    println!(
                        "  {} node(s) report no GPUs and are counted for RAM/CPU only.",
                        cluster.nodes.len() - gpu_nodes
                    );
                }
                println!("  Mixed GPU types? Edit gpu_vram_gb per node in the saved config.");
            } else {
                println!("  Ray reports no GPUs on any node.");
            }

            print_cluster(&cluster);
            save_and_report(&cluster)?;
            Ok(cluster)
        }
        Err(e) => {
            println!("  Could not reach Ray Dashboard: {}", e);
            println!();

            // Fallback: manual config
            let count_input = read_line("  How many GPU nodes? ")?;
            let node_count: usize = match count_input.parse() {
                Ok(n) if n >= 1 => n,
                Ok(_) => {
                    return Err("Node count must be at least 1".to_string());
                }
                Err(_) => {
                    return Err(format!("'{}' is not a valid number", count_input));
                }
            };

            // VRAM is required — reject empty/non-numeric with an actionable error.
            let vram_input = read_line("  GPU VRAM per GPU (GB): ")?;
            let vram_gb: f64 = vram_input
                .parse()
                .ok()
                .filter(|v: &f64| v.is_finite() && *v > 0.0)
                .ok_or_else(|| {
                    format!(
                        "Invalid VRAM value '{}' — expected a positive number like 80",
                        vram_input
                    )
                })?;

            let gpu_name = read_line("  GPU name [GPU]: ")?;
            let gpu_name = if gpu_name.is_empty() {
                "GPU".to_string()
            } else {
                gpu_name
            };

            // RAM and CPU cores: 0 means "unknown" — use prompt helpers that accept 0.
            let ram_gb = prompt_f64("RAM per node (GB) [0 = unknown]", 0.0);

            let cpu_cores: usize = {
                let t = read_line("  CPU cores per node [0 = unknown]: ").unwrap_or_default();
                if t.is_empty() {
                    0
                } else {
                    match t.parse::<usize>() {
                        Ok(n) => n,
                        Err(_) => {
                            eprintln!("⚠️  Invalid number '{}', using 0", t);
                            0
                        }
                    }
                }
            };

            let gpus_per_node = prompt_u32("GPUs per node", 1);

            // Build nodes
            let mut nodes = Vec::with_capacity(node_count);
            for i in 0..node_count {
                let is_head = i == 0;
                let ip = if is_head {
                    head_ip.clone()
                } else {
                    // u8::try_from guards against > 256 node clusters silently wrapping.
                    let default_ip = match u8::try_from(i) {
                        Ok(offset) => {
                            increment_ip(&head_ip, offset).unwrap_or_else(|_| head_ip.clone())
                        }
                        Err(_) => {
                            eprintln!(
                                "⚠️  Cannot auto-increment IP for node {} (max 256 nodes); \
                                 using head IP as placeholder — enter the correct IP below.",
                                i + 1
                            );
                            head_ip.clone()
                        }
                    };
                    let ip = read_line(&format!("  Node {} IP [{}]: ", i + 1, default_ip))?;
                    if ip.is_empty() {
                        default_ip
                    } else {
                        validate_ip_or_hostname(&ip)?;
                        ip
                    }
                };

                nodes.push(ClusterNode {
                    hostname: format!("node-{}", i + 1),
                    ip,
                    gpu_name: gpu_name.clone(),
                    gpu_vram_gb: vram_gb,
                    total_ram_gb: ram_gb,
                    cpu_cores,
                    gpu_count: gpus_per_node,
                    unified_memory: false,
                    is_head,
                    backend: None,
                });
            }

            let cluster = ClusterConfig::from_nodes(&head_ip, nodes);

            println!();
            print_cluster(&cluster);
            save_and_report(&cluster)?;

            Ok(cluster)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(hostname: &str, gpu_count: u32, is_head: bool) -> ClusterNode {
        ClusterNode {
            hostname: hostname.to_string(),
            ip: "10.0.0.1".to_string(),
            gpu_name: "H100".to_string(),
            gpu_vram_gb: 80.0,
            total_ram_gb: 512.0,
            cpu_cores: 64,
            gpu_count,
            unified_memory: false,
            is_head,
            backend: None,
        }
    }

    #[test]
    fn discovery_asks_for_the_backend_instead_of_silently_assuming_cuda() {
        let mut cluster = ClusterConfig::from_nodes("10.0.0.1", vec![node("framework", 1, true)]);
        let warnings = discovery_warnings(&cluster);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("framework") && w.contains("backend = ")),
            "{warnings:?}"
        );
        cluster.nodes[0].backend = Some(llmfit_core::hardware::GpuBackend::Rocm);
        assert!(discovery_warnings(&cluster).is_empty());
        // The JSON document carries the same warnings and the node backend.
        cluster.nodes[0].backend = None;
        let doc = init_json(&cluster, &discovery_warnings(&cluster));
        assert_eq!(doc["warnings"].as_array().unwrap().len(), 1);
        assert!(doc["cluster"]["nodes"][0]["backend"].is_null());
    }

    #[test]
    fn configured_line_pluralises_nodes() {
        let one = ClusterConfig::from_nodes("10.0.0.1", vec![node("a", 1, true)]);
        assert_eq!(
            configured_line(&one),
            "Cluster configured: 1 node, 80 GB total VRAM"
        );
        let two =
            ClusterConfig::from_nodes("10.0.0.1", vec![node("a", 1, true), node("b", 1, false)]);
        assert_eq!(
            configured_line(&two),
            "Cluster configured: 2 nodes, 160 GB total VRAM"
        );
    }

    #[test]
    fn cluster_json_reports_totals_and_nodes() {
        let cluster = ClusterConfig::from_nodes(
            "10.0.0.1",
            vec![node("head", 0, true), node("worker", 8, false)],
        );
        let json = cluster_json(&cluster);
        let c = &json["cluster"];
        assert_eq!(c["node_count"], 2);
        assert_eq!(c["head_ip"], "10.0.0.1");
        assert_eq!(c["total_gpus"], 8);
        assert_eq!(c["total_vram_gb"], 640.0);
        assert_eq!(c["total_ram_gb"], 1024.0);
        assert_eq!(c["total_cpu_cores"], 128);
        let nodes = c["nodes"].as_array().expect("nodes array");
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0]["hostname"], "head");
        assert_eq!(nodes[0]["is_head"], true);
        assert_eq!(nodes[1]["gpu_count"], 8);
        assert_eq!(nodes[1]["vram_gb"], 80.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // increment_ip edge cases
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_increment_ip_normal() {
        assert_eq!(increment_ip("192.168.0.1", 1).unwrap(), "192.168.0.2");
    }

    #[test]
    fn test_increment_ip_no_change() {
        assert_eq!(increment_ip("192.168.0.0", 0).unwrap(), "192.168.0.0");
    }

    #[test]
    fn test_increment_ip_larger_offset() {
        assert_eq!(increment_ip("10.0.0.1", 10).unwrap(), "10.0.0.11");
    }

    #[test]
    fn test_increment_ip_max_valid() {
        assert_eq!(increment_ip("10.0.0.0", 255).unwrap(), "10.0.0.255");
    }

    #[test]
    fn test_increment_ip_boundary_255_plus_0() {
        assert_eq!(increment_ip("192.168.0.255", 0).unwrap(), "192.168.0.255");
    }

    #[test]
    fn test_increment_ip_overflow() {
        let result = increment_ip("192.168.0.254", 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("overflow"));
    }

    #[test]
    fn test_increment_ip_invalid_no_dots() {
        let result = increment_ip("invalid", 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid IP"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IP / hostname validation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_ip() {
        assert!(validate_ip_or_hostname("192.168.1.1").is_ok());
        assert!(validate_ip_or_hostname("10.0.0.1").is_ok());
    }

    #[test]
    fn test_validate_valid_hostname() {
        assert!(validate_ip_or_hostname("gpu-node-1").is_ok());
        assert!(validate_ip_or_hostname("my.cluster.local").is_ok());
    }

    #[test]
    fn test_validate_invalid_empty() {
        assert!(validate_ip_or_hostname("").is_err());
    }

    #[test]
    fn test_validate_invalid_spaces() {
        assert!(validate_ip_or_hostname("10.0.0 .1").is_err());
    }

    #[test]
    fn test_validate_invalid_ip_too_few_octets() {
        assert!(validate_ip_or_hostname("10.0.1").is_err());
    }

    #[test]
    fn test_validate_invalid_ip_octet_overflow() {
        assert!(validate_ip_or_hostname("10.0.0.256").is_err());
    }
}
