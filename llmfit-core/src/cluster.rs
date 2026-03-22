//! Remote GPU cluster detection and resource aggregation.
//!
//! Supports any multi-node GPU cluster (NVIDIA DGX, Lambda, RunPod,
//! bare-metal, etc.) by letting users specify node IPs and hardware
//! specs, or auto-discover them via the Ray Dashboard API.
//!
//! Config is persisted to `~/.config/llmfit/cluster.toml`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::hardware::{GpuBackend, GpuInfo, SystemSpecs};

/// Per-node hardware specs in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub hostname: String,
    pub ip: String,
    pub gpu_name: String,
    pub gpu_vram_gb: f64,
    pub total_ram_gb: f64,
    pub cpu_cores: usize,
    #[serde(default)]
    pub unified_memory: bool,
    #[serde(default)]
    pub is_head: bool,
}

/// Aggregated cluster specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub name: String,
    pub nodes: Vec<ClusterNode>,
    pub head_ip: String,
    #[serde(default = "default_ray_port")]
    pub ray_port: u16,
    #[serde(default = "default_interconnect")]
    pub interconnect: String,
}

fn default_ray_port() -> u16 {
    8265
}
fn default_interconnect() -> String {
    "ethernet".to_string()
}

impl ClusterConfig {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_gpu_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_ram_gb(&self) -> f64 {
        self.nodes.iter().map(|n| n.total_ram_gb).sum()
    }

    pub fn total_vram_gb(&self) -> f64 {
        self.nodes.iter().map(|n| n.gpu_vram_gb).sum()
    }

    pub fn total_cpu_cores(&self) -> usize {
        self.nodes.iter().map(|n| n.cpu_cores).sum()
    }

    /// Interconnect bandwidth label.
    pub fn interconnect_label(&self) -> &str {
        match self.interconnect.as_str() {
            "nvlink" => "NVLink",
            "qsfp" => "QSFP (200 Gb/s)",
            "100gbe" => "100 GbE",
            "ethernet" | "10gbe" => "10 GbE",
            other => other,
        }
    }

    /// Convert cluster config into an aggregated `SystemSpecs` so the existing
    /// fit analysis pipeline works unmodified. The cluster's total VRAM is
    /// presented as a single GPU pool (tensor-parallel across nodes).
    pub fn to_system_specs(&self) -> SystemSpecs {
        let total_vram: f64 = self.total_vram_gb();
        let total_ram: f64 = self.total_ram_gb();
        let total_cores: usize = self.total_cpu_cores();
        let node_count = self.nodes.len() as u32;

        let gpu_name = self
            .nodes
            .first()
            .map(|n| n.gpu_name.clone())
            .unwrap_or_else(|| "Unknown".into());

        let unified = self
            .nodes
            .first()
            .map(|n| n.unified_memory)
            .unwrap_or(false);

        let backend = if unified {
            GpuBackend::Cuda // unified memory NVIDIA (e.g. DGX Spark GB10)
        } else {
            GpuBackend::Cuda
        };

        let gpus = vec![GpuInfo {
            name: gpu_name.clone(),
            vram_gb: Some(self.nodes.first().map(|n| n.gpu_vram_gb).unwrap_or(0.0)),
            backend,
            count: node_count,
            unified_memory: unified,
        }];

        SystemSpecs {
            total_ram_gb: total_ram,
            available_ram_gb: total_ram * 0.85, // conservative estimate
            total_cpu_cores: total_cores,
            cpu_name: format!("Cluster ({}× nodes)", node_count),
            has_gpu: true,
            gpu_vram_gb: Some(self.nodes.first().map(|n| n.gpu_vram_gb).unwrap_or(0.0)),
            total_gpu_vram_gb: Some(total_vram),
            gpu_name: Some(format!("{} (×{})", gpu_name, node_count)),
            gpu_count: node_count,
            unified_memory: false, // cluster uses NCCL path, not unified
            backend,
            gpus,
            cluster_mode: true,
            cluster_node_count: node_count,
        }
    }

    // ── Config file persistence ────────────────────────────────────

    /// Default config path: `~/.config/llmfit/cluster.toml`.
    pub fn config_path() -> Option<PathBuf> {
        dirs_path().map(|d| d.join("cluster.toml"))
    }

    /// Load saved cluster config, if it exists.
    pub fn load() -> Option<Self> {
        let path = Self::config_path()?;
        let content = std::fs::read_to_string(&path).ok()?;
        toml::from_str(&content).ok()
    }

    /// Save cluster config to disk.
    pub fn save(&self) -> Result<(), String> {
        let path = Self::config_path().ok_or("Could not determine config directory")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create config dir: {}", e))?;
        }
        let content =
            toml::to_string_pretty(self).map_err(|e| format!("TOML serialize error: {}", e))?;
        std::fs::write(&path, content).map_err(|e| format!("Failed to write config: {}", e))?;
        Ok(())
    }

    /// Remove saved cluster config.
    pub fn remove_config() -> Result<(), String> {
        if let Some(path) = Self::config_path() {
            if path.exists() {
                std::fs::remove_file(&path)
                    .map_err(|e| format!("Failed to remove config: {}", e))?;
            }
        }
        Ok(())
    }

    // ── Discovery ──────────────────────────────────────────────────

    /// Try to discover cluster from Ray Dashboard API at the given head node.
    pub fn discover_from_ray(head_ip: &str, ray_port: u16) -> Result<Self, String> {
        let url = format!("http://{}:{}/nodes?view=summary", head_ip, ray_port);

        let resp = ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(5)))
            .build()
            .call()
            .map_err(|e| format!("Ray API request failed: {}", e))?;

        let body: serde_json::Value = resp
            .into_body()
            .read_json()
            .map_err(|e| format!("Ray API JSON parse error: {}", e))?;

        // Ray /nodes?view=summary returns { "data": { "summary": [...] } }
        let nodes_data = body
            .get("data")
            .and_then(|d: &serde_json::Value| d.get("summary"))
            .and_then(|s: &serde_json::Value| s.as_array())
            .ok_or("Unexpected Ray API response format")?;

        if nodes_data.is_empty() {
            return Err("No nodes found in Ray cluster".to_string());
        }

        let mut nodes = Vec::new();
        let mut head_found = false;

        for (i, node) in nodes_data.iter().enumerate() {
            let default_hostname = format!("node-{}", i + 1);

            let ip = node
                .get("raylet")
                .and_then(|r: &serde_json::Value| r.get("nodeManagerAddress"))
                .and_then(|a: &serde_json::Value| a.as_str())
                .unwrap_or(head_ip)
                .to_string();

            let hostname = node
                .get("hostname")
                .and_then(|h: &serde_json::Value| h.as_str())
                .unwrap_or(&default_hostname)
                .to_string();

            let gpu_count = node
                .get("resources")
                .and_then(|r: &serde_json::Value| r.get("GPU"))
                .and_then(|g: &serde_json::Value| g.as_f64())
                .unwrap_or(1.0);

            let memory_bytes = node
                .get("resources")
                .and_then(|r: &serde_json::Value| r.get("memory"))
                .and_then(|m: &serde_json::Value| m.as_f64())
                .unwrap_or(0.0);

            let total_ram = if memory_bytes > 0.0 {
                memory_bytes / (1024.0 * 1024.0 * 1024.0)
            } else {
                0.0
            };

            let cpu_cores = node
                .get("resources")
                .and_then(|r: &serde_json::Value| r.get("CPU"))
                .and_then(|c: &serde_json::Value| c.as_f64())
                .map(|c| c as usize)
                .unwrap_or(0);

            let is_head = ip == head_ip || (!head_found && i == 0);
            if is_head {
                head_found = true;
            }

            nodes.push(ClusterNode {
                hostname,
                ip,
                gpu_name: "GPU".to_string(), // Ray doesn't report GPU model
                gpu_vram_gb: gpu_count * 80.0, // rough default; user can correct
                total_ram_gb: total_ram,
                cpu_cores,
                unified_memory: false,
                is_head,
            });
        }

        // Sort: head first, then by hostname
        nodes.sort_by(|a, b| b.is_head.cmp(&a.is_head).then(a.hostname.cmp(&b.hostname)));

        Ok(ClusterConfig {
            name: format!("{}-node cluster", nodes.len()),
            head_ip: head_ip.to_string(),
            ray_port,
            interconnect: "ethernet".to_string(),
            nodes,
        })
    }

    /// Create a cluster config from manual node list.
    pub fn from_nodes(head_ip: &str, nodes: Vec<ClusterNode>) -> Self {
        ClusterConfig {
            name: format!("{}-node cluster", nodes.len()),
            head_ip: head_ip.to_string(),
            ray_port: 8265,
            interconnect: "ethernet".to_string(),
            nodes,
        }
    }

    /// Check if the Ray cluster at the configured head node is reachable.
    pub fn is_ray_reachable(&self) -> bool {
        let url = format!(
            "http://{}:{}/nodes?view=summary",
            self.head_ip, self.ray_port
        );
        ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(3)))
            .build()
            .call()
            .is_ok()
    }

    /// Display cluster info to stdout.
    pub fn display(&self) {
        println!();
        println!("  Cluster: {}", self.name);
        println!("  Nodes:   {}", self.node_count());
        println!();
        for node in &self.nodes {
            let role = if node.is_head { "HEAD" } else { "WORKER" };
            println!(
                "    {} ({}) — {} | {} | {:.0} GB VRAM | {:.0} GB RAM | {} cores",
                node.hostname, role, node.ip, node.gpu_name, node.gpu_vram_gb,
                node.total_ram_gb, node.cpu_cores
            );
        }
        println!();
        println!("  Totals:");
        println!("    GPUs:     {}", self.total_gpu_count());
        println!(
            "    VRAM:     {:.0} GB",
            self.total_vram_gb()
        );
        println!(
            "    RAM:      {:.0} GB",
            self.total_ram_gb()
        );
        println!("    CPUs:     {} cores", self.total_cpu_cores());
        println!("    Link:     {}", self.interconnect_label());
        println!();
    }

    /// Display as JSON.
    pub fn display_json(&self) {
        let json = serde_json::json!({
            "cluster": {
                "name": self.name,
                "node_count": self.node_count(),
                "head_ip": self.head_ip,
                "ray_port": self.ray_port,
                "interconnect": self.interconnect,
                "total_gpus": self.total_gpu_count(),
                "total_ram_gb": self.total_ram_gb(),
                "total_vram_gb": self.total_vram_gb(),
                "total_cpu_cores": self.total_cpu_cores(),
                "nodes": self.nodes.iter().map(|n| serde_json::json!({
                    "hostname": n.hostname,
                    "ip": n.ip,
                    "gpu": n.gpu_name,
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
}

// ── Helpers ────────────────────────────────────────────────────────

fn dirs_path() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".config").join("llmfit"))
}

/// Increment the last octet of an IPv4 address.
fn increment_ip(ip: &str, offset: u8) -> Result<String, String> {
    let parts: Vec<&str> = ip.rsplitn(2, '.').collect();
    if parts.len() == 2 {
        if let Ok(last_octet) = parts[0].parse::<u8>() {
            if let Some(new_octet) = last_octet.checked_add(offset) {
                return Ok(format!("{}.{}", parts[1], new_octet));
            } else {
                return Err(format!(
                    "IP octet overflow: {}.{} + {} exceeds 255",
                    parts[1], last_octet, offset
                ));
            }
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

// ── Interactive cluster init ───────────────────────────────────────

/// Run the interactive cluster initialization flow.
/// Returns the created/updated ClusterConfig.
pub fn interactive_init() -> Result<ClusterConfig, String> {
    use std::io::{self, BufRead, Write};

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    println!();
    println!("  === Remote GPU Cluster Setup ===");
    println!();

    // Step 1: Get head node IP
    print!("  Head node IP or hostname: ");
    stdout.flush().ok();
    let mut head_input = String::new();
    stdin
        .lock()
        .read_line(&mut head_input)
        .map_err(|e| format!("Read error: {}", e))?;
    let head_ip = head_input.trim();
    if head_ip.is_empty() {
        return Err("Head node IP is required".to_string());
    }
    validate_ip_or_hostname(head_ip)?;
    let head_ip = head_ip.to_string();

    // Step 2: Try Ray Dashboard API
    print!("  Ray Dashboard port [8265]: ");
    stdout.flush().ok();
    let mut port_input = String::new();
    stdin
        .lock()
        .read_line(&mut port_input)
        .map_err(|e| format!("Read error: {}", e))?;
    let ray_port: u16 = port_input.trim().parse().unwrap_or(default_ray_port());

    println!();
    println!(
        "  Connecting to Ray Dashboard at {}:{}...",
        head_ip, ray_port
    );

    match ClusterConfig::discover_from_ray(&head_ip, ray_port) {
        Ok(mut cluster) => {
            println!("  Found {} node(s) via Ray API.", cluster.node_count());

            // Let user set VRAM per node
            print!("  GPU VRAM per node (GB) [80]: ");
            stdout.flush().ok();
            let mut vram_input = String::new();
            stdin
                .lock()
                .read_line(&mut vram_input)
                .map_err(|e| format!("Read error: {}", e))?;
            let vram_gb: f64 = vram_input.trim().parse().unwrap_or(80.0);
            for node in &mut cluster.nodes {
                node.gpu_vram_gb = vram_gb;
            }

            cluster.display();
            cluster.save()?;
            if let Some(path) = ClusterConfig::config_path() {
                println!("  Saved to {}", path.display());
            }
            println!();
            Ok(cluster)
        }
        Err(e) => {
            println!("  Could not reach Ray Dashboard: {}", e);
            println!();

            // Fallback: manual config
            print!("  How many GPU nodes? ");
            stdout.flush().ok();
            let mut count_input = String::new();
            stdin
                .lock()
                .read_line(&mut count_input)
                .map_err(|e| format!("Read error: {}", e))?;
            let trimmed = count_input.trim();
            let node_count: usize = match trimmed.parse() {
                Ok(n) if n >= 1 => n,
                Ok(_) => {
                    return Err("Node count must be at least 1".to_string());
                }
                Err(_) => {
                    return Err(format!("'{}' is not a valid number", trimmed));
                }
            };

            print!("  GPU VRAM per node (GB): ");
            stdout.flush().ok();
            let mut vram_input = String::new();
            stdin
                .lock()
                .read_line(&mut vram_input)
                .map_err(|e| format!("Read error: {}", e))?;
            let vram_gb: f64 = vram_input
                .trim()
                .parse()
                .map_err(|_| "Invalid VRAM value")?;

            print!("  GPU name [GPU]: ");
            stdout.flush().ok();
            let mut gpu_input = String::new();
            stdin
                .lock()
                .read_line(&mut gpu_input)
                .map_err(|e| format!("Read error: {}", e))?;
            let gpu_name = gpu_input.trim();
            let gpu_name = if gpu_name.is_empty() {
                "GPU"
            } else {
                gpu_name
            };

            print!("  RAM per node (GB) [0 = unknown]: ");
            stdout.flush().ok();
            let mut ram_input = String::new();
            stdin
                .lock()
                .read_line(&mut ram_input)
                .map_err(|e| format!("Read error: {}", e))?;
            let ram_gb: f64 = ram_input.trim().parse().unwrap_or(0.0);

            print!("  CPU cores per node [0 = unknown]: ");
            stdout.flush().ok();
            let mut cores_input = String::new();
            stdin
                .lock()
                .read_line(&mut cores_input)
                .map_err(|e| format!("Read error: {}", e))?;
            let cpu_cores: usize = cores_input.trim().parse().unwrap_or(0);

            // Build nodes
            let mut nodes = Vec::with_capacity(node_count);
            for i in 0..node_count {
                let is_head = i == 0;
                let ip = if is_head {
                    head_ip.clone()
                } else {
                    let default_ip =
                        increment_ip(&head_ip, i as u8).unwrap_or_else(|_| head_ip.clone());
                    print!("  Node {} IP [{}]: ", i + 1, default_ip);
                    stdout.flush().ok();
                    let mut ip_input = String::new();
                    stdin
                        .lock()
                        .read_line(&mut ip_input)
                        .map_err(|e| format!("Read error: {}", e))?;
                    let ip = ip_input.trim();
                    if ip.is_empty() {
                        default_ip
                    } else {
                        validate_ip_or_hostname(ip)?;
                        ip.to_string()
                    }
                };

                nodes.push(ClusterNode {
                    hostname: format!("node-{}", i + 1),
                    ip,
                    gpu_name: gpu_name.to_string(),
                    gpu_vram_gb: vram_gb,
                    total_ram_gb: ram_gb,
                    cpu_cores,
                    unified_memory: false,
                    is_head,
                });
            }

            let cluster = ClusterConfig::from_nodes(&head_ip, nodes);

            println!();
            cluster.display();

            cluster.save()?;
            if let Some(path) = ClusterConfig::config_path() {
                println!("  Saved to {}", path.display());
            }
            println!();

            Ok(cluster)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ──────────────────────────────────────────────────────────────────
    // increment_ip edge cases
    // ──────────────────────────────────────────────────────────────────

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

    // ──────────────────────────────────────────────────────────────────
    // IP / hostname validation
    // ──────────────────────────────────────────────────────────────────

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

    // ──────────────────────────────────────────────────────────────────
    // ClusterConfig
    // ──────────────────────────────────────────────────────────────────

    fn make_test_cluster() -> ClusterConfig {
        ClusterConfig::from_nodes(
            "10.0.0.1",
            vec![
                ClusterNode {
                    hostname: "node-1".to_string(),
                    ip: "10.0.0.1".to_string(),
                    gpu_name: "A100".to_string(),
                    gpu_vram_gb: 80.0,
                    total_ram_gb: 256.0,
                    cpu_cores: 64,
                    unified_memory: false,
                    is_head: true,
                },
                ClusterNode {
                    hostname: "node-2".to_string(),
                    ip: "10.0.0.2".to_string(),
                    gpu_name: "A100".to_string(),
                    gpu_vram_gb: 80.0,
                    total_ram_gb: 256.0,
                    cpu_cores: 64,
                    unified_memory: false,
                    is_head: false,
                },
            ],
        )
    }

    #[test]
    fn test_cluster_totals() {
        let cluster = make_test_cluster();
        assert_eq!(cluster.node_count(), 2);
        assert_eq!(cluster.total_gpu_count(), 2);
        assert!((cluster.total_vram_gb() - 160.0).abs() < 0.01);
        assert!((cluster.total_ram_gb() - 512.0).abs() < 0.01);
        assert_eq!(cluster.total_cpu_cores(), 128);
    }

    #[test]
    fn test_cluster_to_system_specs() {
        let cluster = make_test_cluster();
        let specs = cluster.to_system_specs();
        assert!(specs.cluster_mode);
        assert_eq!(specs.cluster_node_count, 2);
        assert!(specs.has_gpu);
        assert!((specs.total_gpu_vram_gb.unwrap() - 160.0).abs() < 0.01);
        assert_eq!(specs.gpu_count, 2);
    }

    #[test]
    fn test_cluster_serialization_roundtrip() {
        let cluster = make_test_cluster();
        let toml_str = toml::to_string_pretty(&cluster).unwrap();
        let loaded: ClusterConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.node_count(), cluster.node_count());
        assert!((loaded.total_vram_gb() - cluster.total_vram_gb()).abs() < 0.01);
    }
}
