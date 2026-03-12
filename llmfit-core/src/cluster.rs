//! DGX Spark cluster detection and resource aggregation.
//!
//! Discovers multi-node DGX Spark clusters via:
//! 1. Saved config file (`~/.config/llmfit/cluster.toml`)
//! 2. Ray Dashboard API (live node enumeration)
//! 3. Interactive prompts (fallback)

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::hardware::{GpuBackend, GpuInfo, SystemSpecs};

// — DGX Spark hardware constants ——————————————————————————————————————
// Source: NVIDIA DGX Spark technical specifications (2025).
// See: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
/// Total unified memory per DGX Spark node (LPDDR5x).
const SPARK_TOTAL_RAM_GB: f64 = 128.0;
/// System reservation on GB10 — OS, drivers, runtime services, CUDA context.
/// Measured empirically: ~43 GB unavailable for model inference on stock DGX OS.
const SPARK_SYSTEM_RESERVED_GB: f64 = 43.0;
/// Usable GPU memory per node after system reservation.
const SPARK_USABLE_VRAM_GB: f64 = SPARK_TOTAL_RAM_GB - SPARK_SYSTEM_RESERVED_GB;
/// CPU cores per DGX Spark node (10 × Cortex-X925 + 10 × Cortex-A725).
const SPARK_CPU_CORES: usize = 20;
/// GPU name string for identification in hardware detection.
const SPARK_GPU_NAME: &str = "NVIDIA GB10 (Blackwell)";

/// Per-node hardware specs in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparkNode {
    pub hostname: String,
    pub ip: String,
    #[serde(default = "default_gpu_name")]
    pub gpu_name: String,
    #[serde(default = "default_vram")]
    pub gpu_vram_gb: f64,
    #[serde(default = "default_ram")]
    pub total_ram_gb: f64,
    #[serde(default = "default_cores")]
    pub cpu_cores: usize,
    #[serde(default = "default_true")]
    pub unified_memory: bool,
    #[serde(default)]
    pub is_head: bool,
}

fn default_gpu_name() -> String {
    SPARK_GPU_NAME.to_string()
}
fn default_vram() -> f64 {
    SPARK_USABLE_VRAM_GB
}
fn default_ram() -> f64 {
    SPARK_TOTAL_RAM_GB
}
fn default_cores() -> usize {
    SPARK_CPU_CORES
}
fn default_true() -> bool {
    true
}

/// Aggregated cluster specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSpecs {
    pub name: String,
    pub nodes: Vec<SparkNode>,
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
    "qsfp".to_string()
}

impl ClusterSpecs {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 1 GPU per DGX Spark node.
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
            "qsfp" => "QSFP (200 Gb/s)",
            "ethernet" | "10gbe" => "10 GbE",
            other => other,
        }
    }

    /// Convert cluster specs into an aggregated `SystemSpecs` so the existing
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

        // Present the cluster as a single multi-GPU system with CUDA backend.
        // vLLM + Ray handle tensor parallelism across nodes transparently.
        let gpus = vec![GpuInfo {
            name: gpu_name.clone(),
            vram_gb: Some(self.nodes.first().map(|n| n.gpu_vram_gb).unwrap_or(0.0)),
            backend: GpuBackend::Cuda,
            count: node_count,
            unified_memory: unified,
        }];

        SystemSpecs {
            total_ram_gb: total_ram,
            available_ram_gb: total_ram * 0.85, // conservative estimate
            total_cpu_cores: total_cores,
            cpu_name: format!(
                "{}× ARM Cortex ({}c each)",
                node_count,
                self.nodes.first().map(|n| n.cpu_cores).unwrap_or(0)
            ),
            has_gpu: true,
            gpu_vram_gb: Some(self.nodes.first().map(|n| n.gpu_vram_gb).unwrap_or(0.0)),
            total_gpu_vram_gb: Some(total_vram),
            gpu_name: Some(format!("{} (×{})", gpu_name, node_count)),
            gpu_count: node_count,
            unified_memory: false, // cluster uses CUDA/NCCL path, not unified
            backend: GpuBackend::Cuda,
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
            let default_hostname = format!("spark-{}", i + 1);

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

            // Extract GPU resources if available
            let gpu_count = node
                .get("resources")
                .and_then(|r: &serde_json::Value| r.get("GPU"))
                .and_then(|g: &serde_json::Value| g.as_f64())
                .unwrap_or(1.0);

            // Extract memory (Ray reports in bytes)
            let memory_bytes = node
                .get("resources")
                .and_then(|r: &serde_json::Value| r.get("memory"))
                .and_then(|m: &serde_json::Value| m.as_f64())
                .unwrap_or(0.0);

            let total_ram = if memory_bytes > 0.0 {
                memory_bytes / (1024.0 * 1024.0 * 1024.0)
            } else {
                SPARK_TOTAL_RAM_GB
            };

            let cpu_cores = node
                .get("resources")
                .and_then(|r: &serde_json::Value| r.get("CPU"))
                .and_then(|c: &serde_json::Value| c.as_f64())
                .map(|c| c as usize)
                .unwrap_or(SPARK_CPU_CORES);

            let is_head = ip == head_ip || (!head_found && i == 0);
            if is_head {
                head_found = true;
            }

            nodes.push(SparkNode {
                hostname,
                ip,
                gpu_name: SPARK_GPU_NAME.to_string(),
                gpu_vram_gb: SPARK_USABLE_VRAM_GB * gpu_count,
                total_ram_gb: total_ram.min(SPARK_TOTAL_RAM_GB), // cap at physical max
                cpu_cores,
                unified_memory: true,
                is_head,
            });
        }

        // Sort: head first, then by hostname
        nodes.sort_by(|a, b| b.is_head.cmp(&a.is_head).then(a.hostname.cmp(&b.hostname)));

        Ok(ClusterSpecs {
            name: format!("{}-node DGX Spark Cluster", nodes.len()),
            head_ip: head_ip.to_string(),
            ray_port,
            interconnect: "qsfp".to_string(),
            nodes,
        })
    }

    /// Create a cluster config from manual node count + head IP.
    /// Uses known DGX Spark specs for each node.
    pub fn from_manual(head_ip: &str, node_count: usize) -> Self {
        let mut nodes = Vec::with_capacity(node_count);
        for i in 0..node_count {
            let is_head = i == 0;
            // Guess worker IPs by incrementing the last octet
            let ip = if is_head {
                head_ip.to_string()
            } else {
                increment_ip(head_ip, i as u8).unwrap_or_else(|e| {
                    eprintln!("Warning: {}", e);
                    head_ip.to_string()
                })
            };

            nodes.push(SparkNode {
                hostname: format!("spark-{}", i + 1),
                ip,
                gpu_name: SPARK_GPU_NAME.to_string(),
                gpu_vram_gb: SPARK_USABLE_VRAM_GB,
                total_ram_gb: SPARK_TOTAL_RAM_GB,
                cpu_cores: SPARK_CPU_CORES,
                unified_memory: true,
                is_head,
            });
        }

        ClusterSpecs {
            name: format!("{}-node DGX Spark Cluster", node_count),
            head_ip: head_ip.to_string(),
            ray_port: 8265,
            interconnect: "qsfp".to_string(),
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
        println!(
            "  Nodes:   {} × DGX Spark (GB10 Blackwell)",
            self.node_count()
        );
        println!();
        for node in &self.nodes {
            let role = if node.is_head { "HEAD" } else { "WORKER" };
            println!(
                "    {} ({}) — {} | {:.0} GB unified | {} cores",
                node.hostname, role, node.ip, node.total_ram_gb, node.cpu_cores
            );
        }
        println!();
        println!("  Totals:");
        println!(
            "    GPUs:     {} × {}",
            self.total_gpu_count(),
            SPARK_GPU_NAME
        );
        println!(
            "    Memory:   {:.0} GB unified ({:.0} GB usable for models)",
            self.total_ram_gb(),
            self.total_vram_gb()
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

// ── Interactive cluster init ───────────────────────────────────────

/// Run the interactive cluster initialization flow.
/// Returns the created/updated ClusterSpecs.
pub fn interactive_init() -> Result<ClusterSpecs, String> {
    use std::io::{self, BufRead, Write};

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    println!();
    println!("  === DGX Spark Cluster Setup ===");
    println!();

    // Step 1: Get head node IP
    print!("  Head node IP or hostname [10.0.0.1]: ");
    stdout.flush().ok();
    let mut head_input = String::new();
    stdin
        .lock()
        .read_line(&mut head_input)
        .map_err(|e| format!("Read error: {}", e))?;
    let head_ip = head_input.trim();
    let head_ip = if head_ip.is_empty() {
        "10.0.0.1"
    } else {
        head_ip
    };
    // Validate IP format
    if !head_ip.contains('.')
        && !head_ip
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '.')
    {
        return Err(format!("Invalid IP or hostname: '{}'", head_ip));
    }
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

    match ClusterSpecs::discover_from_ray(&head_ip, ray_port) {
        Ok(cluster) => {
            println!("  Found {} node(s) via Ray API.", cluster.node_count());
            cluster.display();

            // Save
            cluster.save()?;
            if let Some(path) = ClusterSpecs::config_path() {
                println!("  Saved to {}", path.display());
            }
            println!();
            Ok(cluster)
        }
        Err(e) => {
            println!("  Could not reach Ray Dashboard: {}", e);
            println!();

            // Fallback: manual node count
            print!("  How many DGX Spark nodes? [3]: ");
            stdout.flush().ok();
            let mut count_input = String::new();
            stdin
                .lock()
                .read_line(&mut count_input)
                .map_err(|e| format!("Read error: {}", e))?;
            let trimmed = count_input.trim();
            let node_count: usize = if trimmed.is_empty() {
                3
            } else {
                match trimmed.parse() {
                    Ok(n) if n >= 1 => n,
                    Ok(_) => {
                        eprintln!("  Warning: node count must be at least 1, using 3");
                        3
                    }
                    Err(_) => {
                        eprintln!("  Warning: '{}' is not a valid number, using 3", trimmed);
                        3
                    }
                }
            };

            let mut cluster = ClusterSpecs::from_manual(&head_ip, node_count);
            cluster.ray_port = ray_port;

            // Let user correct worker IPs
            for i in 1..cluster.nodes.len() {
                let default_ip = cluster.nodes[i].ip.clone();
                print!("  spark-{} IP [{}]: ", i + 1, default_ip);
                stdout.flush().ok();
                let mut ip_input = String::new();
                stdin
                    .lock()
                    .read_line(&mut ip_input)
                    .map_err(|e| format!("Read error: {}", e))?;
                let ip = ip_input.trim();
                if !ip.is_empty() {
                    cluster.nodes[i].ip = ip.to_string();
                }
            }

            println!();
            cluster.display();

            // Save
            cluster.save()?;
            if let Some(path) = ClusterSpecs::config_path() {
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
        // 0 + 255 = 255, exactly at limit
        assert_eq!(increment_ip("10.0.0.0", 255).unwrap(), "10.0.0.255");
    }

    #[test]
    fn test_increment_ip_boundary_255_plus_0() {
        // 255 + 0 should succeed (no overflow)
        assert_eq!(increment_ip("192.168.0.255", 0).unwrap(), "192.168.0.255");
    }

    #[test]
    fn test_increment_ip_overflow_254_plus_2() {
        let result = increment_ip("192.168.0.254", 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("overflow"));
    }

    #[test]
    fn test_increment_ip_overflow_255_plus_1() {
        let result = increment_ip("192.168.0.255", 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("overflow"));
    }

    #[test]
    fn test_increment_ip_invalid_no_dots() {
        let result = increment_ip("invalid", 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid IP"));
    }

    #[test]
    fn test_increment_ip_invalid_last_octet() {
        // Last octet is not a number
        let result = increment_ip("192.168.0.abc", 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid IP"));
    }

    // ──────────────────────────────────────────────────────────────────
    // ClusterSpecs aggregation
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_cluster_specs_from_manual() {
        let cluster = ClusterSpecs::from_manual("10.0.0.1", 3);
        assert_eq!(cluster.node_count(), 3);
        assert_eq!(cluster.total_gpu_count(), 3);
        assert!(cluster.nodes[0].is_head);
        assert!(!cluster.nodes[1].is_head);
        assert_eq!(cluster.nodes[0].ip, "10.0.0.1");
        assert_eq!(cluster.nodes[1].ip, "10.0.0.2");
        assert_eq!(cluster.nodes[2].ip, "10.0.0.3");
    }

    #[test]
    fn test_cluster_specs_totals() {
        let cluster = ClusterSpecs::from_manual("10.0.0.1", 2);
        assert_eq!(cluster.total_cpu_cores(), SPARK_CPU_CORES * 2);
        assert!((cluster.total_ram_gb() - SPARK_TOTAL_RAM_GB * 2.0).abs() < 0.01);
        assert!((cluster.total_vram_gb() - SPARK_USABLE_VRAM_GB * 2.0).abs() < 0.01);
    }
}
