//! `llmfit cluster` subcommands: interactive setup, status output, and clearing
//! the saved config. Cluster data and discovery live in `llmfit_core::cluster`;
//! this module owns the terminal prompts and printing.

use clap::Subcommand;
use std::io::{self, BufRead, Write};

use llmfit_core::cluster::{ClusterConfig, ClusterNode};

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

/// Run a `llmfit cluster` subcommand. Exits the process with status 1 on error.
pub(crate) fn run(action: ClusterAction) {
    match action {
        ClusterAction::Init { ray_url } => match ray_url {
            // Non-interactive path: a --ray-url was supplied, so discover
            // the cluster straight from the Ray Dashboard API. Never drop
            // into interactive prompts here — fail loudly instead.
            Some(url) => match ClusterConfig::discover_from_ray_url(&url) {
                Ok(cluster) => {
                    if let Err(e) = cluster.save() {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    }
                    println!(
                        "Cluster configured: {} nodes, {:.0} GB total VRAM",
                        cluster.node_count(),
                        cluster.total_vram_gb()
                    );
                    if cluster.total_gpu_count() == 0 {
                        eprintln!(
                            "⚠️  Ray reports no GPUs on any node; `--cluster` needs GPU nodes and will refuse this config."
                        );
                    } else if !cluster.zero_vram_gpu_nodes().is_empty() {
                        eprintln!(
                            "⚠️  Ray discovery cannot report per-GPU VRAM; the saved config has gpu_vram_gb = 0."
                        );
                        eprintln!(
                            "   Pass --memory <per-GPU VRAM> with `--cluster`, set gpu_vram_gb per node in the cluster config, or re-run `llmfit cluster init` interactively."
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Error: could not initialize cluster from --ray-url '{}': {}",
                        url, e
                    );
                    std::process::exit(1);
                }
            },
            None => match interactive_init() {
                Ok(cluster) => {
                    println!(
                        "Cluster configured: {} nodes, {:.0} GB total VRAM",
                        cluster.node_count(),
                        cluster.total_vram_gb()
                    );
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            },
        },
        ClusterAction::Status => match ClusterConfig::load() {
            Ok(Some(cluster)) => print_cluster(&cluster),
            Ok(None) => {
                eprintln!("No cluster configured. Run `llmfit cluster init` to set up.");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        },
        ClusterAction::Clear => match ClusterConfig::remove_config() {
            Ok(()) => println!("Cluster config removed."),
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        },
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
    println!("    Link:     {}", cluster.interconnect_label());
    println!();
}

/// Print cluster info as JSON.
#[allow(dead_code)]
fn print_cluster_json(cluster: &ClusterConfig) {
    let json = serde_json::json!({
        "cluster": {
            "name": cluster.name,
            "node_count": cluster.node_count(),
            "head_ip": cluster.head_ip,
            "ray_port": cluster.ray_port,
            "interconnect": cluster.interconnect,
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
            })).collect::<Vec<_>>(),
        }
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).expect("JSON serialization failed")
    );
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

    match ClusterConfig::discover_from_ray(&head_ip, ray_port) {
        Ok(mut cluster) => {
            println!("  Found {} node(s) via Ray API.", cluster.node_count());

            // Prompt for VRAM once: Ray reports GPU counts but neither the
            // GPU model nor per-GPU VRAM, so there is nothing to tell mixed
            // GPU types apart. Only nodes that actually have GPUs get the
            // value; CPU-only nodes (e.g. a head node) keep 0 GPUs / 0 GB.
            let gpu_nodes = cluster.nodes.iter().filter(|n| n.gpu_count > 0).count();
            if gpu_nodes > 0 {
                let vram = prompt_f64(
                    &format!("GPU VRAM per GPU across {} GPU node(s) (GB)", gpu_nodes),
                    80.0,
                );
                for node in cluster.nodes.iter_mut().filter(|n| n.gpu_count > 0) {
                    node.gpu_vram_gb = vram;
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
            let vram_gb: f64 = vram_input.parse().map_err(|_| {
                format!(
                    "Invalid VRAM value '{}' — expected a number like 80",
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
