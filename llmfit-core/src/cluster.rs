//! Remote GPU cluster detection and resource aggregation.
//!
//! Supports any multi-node GPU cluster (NVIDIA DGX, Lambda, RunPod,
//! bare-metal, etc.) by letting users specify node IPs and hardware
//! specs, or auto-discover them via the Ray Dashboard API.
//!
//! Config is persisted to the platform-appropriate config directory:
//! - macOS: `~/Library/Application Support/llmfit/cluster.toml`
//! - Linux: `$XDG_CONFIG_HOME/llmfit/cluster.toml` (default: `~/.config/llmfit/cluster.toml`)
//! - Windows: `%APPDATA%/llmfit/cluster.toml`

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::hardware::{GpuBackend, GpuInfo, SystemSpecs, gpu_compute_capability};

/// Conservative fraction of total cluster RAM assumed available for inference,
/// accounting for OS/orchestrator overhead.
const CLUSTER_AVAILABLE_RAM_FACTOR: f64 = 0.85;

/// Serde helpers for `Option<GpuBackend>` using lowercase string representation.
///
/// Accepted values: `"cuda"`, `"metal"`, `"rocm"`, `"vulkan"`, `"sycl"`,
/// `"cpu_arm"`, `"cpu_x86"`, `"ascend"`.
///
/// Omit the field (or leave it out of the TOML) to let llmfit auto-detect; a
/// warning will be emitted when the backend cannot be determined unambiguously.
mod gpu_backend_opt {
    use super::GpuBackend;
    use serde::{Deserialize, Deserializer, Serializer};

    /// The name a backend is written as in `cluster.toml`.
    pub fn key(b: GpuBackend) -> &'static str {
        match b {
            GpuBackend::Cuda => "cuda",
            GpuBackend::Metal => "metal",
            GpuBackend::Rocm => "rocm",
            GpuBackend::Vulkan => "vulkan",
            GpuBackend::Sycl => "sycl",
            GpuBackend::CpuArm => "cpu_arm",
            GpuBackend::CpuX86 => "cpu_x86",
            GpuBackend::Ascend => "ascend",
        }
    }

    pub fn serialize<S: Serializer>(val: &Option<GpuBackend>, s: S) -> Result<S::Ok, S::Error> {
        match val {
            None => s.serialize_none(),
            Some(b) => s.serialize_some(key(*b)),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<GpuBackend>, D::Error> {
        let s: Option<String> = Option::deserialize(d)?;
        Ok(match s.as_deref() {
            None | Some("") => None,
            Some(raw) => Some(match raw.to_lowercase().as_str() {
                "cuda" | "nvidia" => GpuBackend::Cuda,
                "metal" => GpuBackend::Metal,
                "rocm" | "amd" => GpuBackend::Rocm,
                "vulkan" => GpuBackend::Vulkan,
                "sycl" | "intel" => GpuBackend::Sycl,
                "cpu_arm" | "arm" => GpuBackend::CpuArm,
                "cpu_x86" | "x86" | "cpu" => GpuBackend::CpuX86,
                "ascend" => GpuBackend::Ascend,
                other => {
                    return Err(serde::de::Error::custom(format!(
                        "unknown GPU backend '{}'; expected one of: \
                         cuda, metal, rocm, vulkan, sycl, cpu_arm, cpu_x86, ascend",
                        other
                    )));
                }
            }),
        })
    }
}

/// Per-node hardware specs in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterNode {
    pub hostname: String,
    pub ip: String,
    pub gpu_name: String,
    /// VRAM per GPU on this node (NOT total for the whole node).
    /// Multiply by `gpu_count` to get total VRAM contributed by this node.
    pub gpu_vram_gb: f64,
    pub total_ram_gb: f64,
    pub cpu_cores: usize,
    #[serde(default = "default_gpu_count")]
    pub gpu_count: u32,
    #[serde(default)]
    pub unified_memory: bool,
    #[serde(default)]
    pub is_head: bool,
    /// GPU acceleration backend for this node.
    ///
    /// If unspecified or heterogeneous across nodes, CUDA is used with a warning.
    /// Set `backend = "cuda"|"rocm"|"metal"|"vulkan"` per node in your cluster
    /// TOML to suppress that warning.
    #[serde(
        default,
        with = "gpu_backend_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub backend: Option<GpuBackend>,
}

/// Aggregated cluster specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterConfig {
    pub name: String,
    pub nodes: Vec<ClusterNode>,
    pub head_ip: String,
    #[serde(default = "default_ray_port")]
    pub ray_port: u16,
}

fn default_gpu_count() -> u32 {
    1
}

fn default_ray_port() -> u16 {
    8265
}

impl ClusterConfig {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_gpu_count(&self) -> usize {
        self.nodes.iter().map(|n| n.gpu_count as usize).sum()
    }

    pub fn total_ram_gb(&self) -> f64 {
        self.nodes.iter().map(|n| n.total_ram_gb).sum()
    }

    /// Sum of (VRAM per GPU × GPU count) across all nodes.
    ///
    /// Note: `ClusterNode::gpu_vram_gb` is **per-GPU**, not per-node total.
    pub fn total_vram_gb(&self) -> f64 {
        self.nodes
            .iter()
            .map(|n| n.gpu_vram_gb * n.gpu_count as f64)
            .sum()
    }

    pub fn total_cpu_cores(&self) -> usize {
        self.nodes.iter().map(|n| n.cpu_cores).sum()
    }

    /// Nodes that report GPUs, in config order; all nodes when none do.
    fn gpu_nodes(&self) -> Vec<&ClusterNode> {
        let with_gpus: Vec<&ClusterNode> = self.nodes.iter().filter(|n| n.gpu_count > 0).collect();
        if with_gpus.is_empty() {
            self.nodes.iter().collect()
        } else {
            with_gpus
        }
    }

    /// Indices of nodes that declare GPUs but have no per-GPU VRAM recorded.
    /// Ray discovery cannot see VRAM, so a config saved via `cluster init
    /// --ray-url` has `gpu_vram_gb = 0.0` until the user fills it in.
    pub fn zero_vram_gpu_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.gpu_count > 0 && n.gpu_vram_gb <= 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Convert cluster config into an aggregated `SystemSpecs` so the existing
    /// fit analysis pipeline works unmodified. The cluster's GPUs are presented
    /// as a single tensor-parallel pool sized by the smallest card, so on a
    /// mixed cluster `total_gpu_vram_gb` is below [`Self::total_vram_gb`].
    ///
    /// Simulation semantics (TUI Hardware Simulation, `S` key) on these
    /// aggregate specs: the VRAM field is per-GPU and applies uniformly to
    /// every GPU across all nodes, while RAM and CPU cores are cluster-wide
    /// totals. Fit and tokens/second estimates do not model interconnect
    /// overhead between nodes, so simulated multi-node throughput is
    /// optimistic. Confirming the simulation popup without edits applies the
    /// simulation availability model (90% of total RAM) rather than this
    /// aggregation's default 85% reservation.
    pub fn to_system_specs(&self) -> SystemSpecs {
        let total_ram: f64 = self.total_ram_gb();
        let total_cores: usize = self.total_cpu_cores();
        let node_count = self.nodes.len() as u32;
        let total_gpus = self.total_gpu_count() as u32;

        // GPU fields come from nodes that actually have GPUs. A CPU-only node
        // (commonly the Ray head, sorted first) must not supply the per-GPU
        // VRAM, name, backend or memory model. Fall back to all nodes only
        // when none report GPUs.
        let gpu_nodes = self.gpu_nodes();
        // Every shard runs on every GPU, so the oldest architecture decides
        // which quantization kernels the cluster can run. Report that GPU's
        // name, since kernel checks read the compute capability from it.
        let weakest = gpu_nodes
            .iter()
            // `None` sorts first: a GPU llmfit cannot identify might be the
            // oldest, so its kernel support is reported as unknown rather
            // than borrowed from a newer card.
            .min_by_key(|n| gpu_compute_capability(&n.gpu_name))
            .copied();
        let gpu_name = weakest
            .map(|n| n.gpu_name.clone())
            .unwrap_or_else(|| "Unknown".into());
        let mixed = gpu_nodes.iter().any(|n| n.gpu_name != gpu_name);
        // Tensor parallelism shards the model evenly across every GPU, so the
        // smallest card bounds what each shard can hold. Summing mixed cards
        // (2x 80 GB + 2x 24 GB = 208 GB) would overstate a pool that really
        // holds 4x 24 GB.
        let per_gpu_vram = gpu_nodes
            .iter()
            .map(|n| n.gpu_vram_gb)
            .reduce(f64::min)
            .unwrap_or(0.0);
        let usable_vram = per_gpu_vram * total_gpus as f64;

        // unified_memory is true only when ALL GPU nodes have unified memory
        // (e.g. Apple Silicon or DGX GB10 clusters where every node is
        // a system-on-chip with shared DRAM/VRAM).
        let unified_memory = !gpu_nodes.is_empty() && gpu_nodes.iter().all(|n| n.unified_memory);

        let backend = derive_cluster_backend(&gpu_nodes);

        let gpus = vec![GpuInfo {
            name: gpu_name.clone(),
            vram_gb: Some(per_gpu_vram),
            backend,
            count: total_gpus,
            unified_memory,
            // Cluster nodes report only total VRAM, not live free memory; no
            // backend to poll for a cluster-aggregated value.
            free_vram_gb: None,
            free_vram_per_card_gb: Vec::new(),
            pcie_links: Vec::new(),
        }];

        SystemSpecs {
            total_ram_gb: total_ram,
            // Use a conservative fraction — cluster OS/orchestrator (Ray, Slurm, etc.)
            // reserves meaningful overhead.
            available_ram_gb: total_ram * CLUSTER_AVAILABLE_RAM_FACTOR,
            total_cpu_cores: total_cores,
            cpu_name: format!("Cluster (×{} nodes)", node_count),
            // A config with no GPU nodes must not claim one; callers decide
            // whether a GPU-less cluster is usable.
            has_gpu: total_gpus > 0,
            gpu_vram_gb: Some(per_gpu_vram),
            total_gpu_vram_gb: Some(usable_vram),
            gpu_available_gb: None,
            gpu_name: Some(if mixed {
                format!("{} (×{}, mixed)", gpu_name, total_gpus)
            } else {
                format!("{} (×{})", gpu_name, total_gpus)
            }),
            gpu_count: total_gpus,
            unified_memory,
            backend,
            gpus,
            cluster_mode: true,
            cluster_node_count: node_count,
        }
    }

    /// The distinct backends named by GPU nodes, in config order, when there
    /// is more than one. Tensor parallelism cannot span CUDA and Metal (or any
    /// other mix), so such a cluster has no single pool to fit against. Nodes
    /// that leave `backend` unset are not counted; they fall back to CUDA with
    /// [`Self::backend_warning`].
    pub fn conflicting_backends(&self) -> Vec<&'static str> {
        let mut named: Vec<&'static str> = Vec::new();
        for backend in self.gpu_nodes().iter().filter_map(|n| n.backend) {
            let key = gpu_backend_opt::key(backend);
            if !named.contains(&key) {
                named.push(key);
            }
        }
        if named.len() > 1 { named } else { Vec::new() }
    }

    /// Warning to show when the GPU nodes do not all name the same backend,
    /// in which case [`Self::to_system_specs`] assumes CUDA.
    pub fn backend_warning(&self) -> Option<String> {
        let gpu_nodes = self.gpu_nodes();
        if gpu_nodes.is_empty() || uniform_backend(&gpu_nodes).is_some() {
            return None;
        }
        Some(
            "Cluster nodes have heterogeneous or unspecified GPU backends; defaulting to CUDA. \
             Set `backend = \"cuda\"|\"rocm\"|\"metal\"|\"vulkan\"` per node in your cluster TOML to suppress this warning."
                .to_string(),
        )
    }

    // ── Config file persistence ────────────────────────────────────────────────

    /// Default config path via the platform's config directory.
    ///
    /// - macOS: `~/Library/Application Support/llmfit/cluster.toml`
    /// - Linux: `$XDG_CONFIG_HOME/llmfit/cluster.toml` (default: `~/.config/llmfit/cluster.toml`)
    /// - Windows: `%APPDATA%/llmfit/cluster.toml`
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("llmfit").join("cluster.toml"))
    }

    /// Load the saved cluster config. `Ok(None)` means none is saved; an
    /// unreadable or malformed file is an error naming the file and, for
    /// TOML errors, the offending line, since users edit it by hand.
    pub fn load() -> Result<Option<Self>, String> {
        let Some(path) = Self::config_path() else {
            return Ok(None);
        };
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("Failed to read {}: {}", path.display(), e)),
        };
        Self::from_toml(&content)
            .map(Some)
            .map_err(|e| format!("Invalid cluster config {}: {}", path.display(), e))
    }

    /// Parse a cluster config from TOML text.
    ///
    /// Unknown keys are rejected, so a typo such as `gpu_cout = 8` is an error
    /// instead of quietly meaning one GPU. The one exception is the top-level
    /// `interconnect` key, which older versions wrote and nothing reads.
    pub fn from_toml(content: &str) -> Result<Self, String> {
        let mut table: toml::Table = toml::from_str(content).map_err(|e| e.to_string())?;
        if table.remove("interconnect").is_none() {
            // Parse the original text so errors keep their line and column.
            return toml::from_str(content).map_err(|e| e.to_string());
        }
        table.try_into().map_err(|e: toml::de::Error| e.to_string())
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
        if let Some(path) = Self::config_path()
            && path.exists()
        {
            std::fs::remove_file(&path).map_err(|e| format!("Failed to remove config: {}", e))?;
        }
        Ok(())
    }

    // ── Discovery ─────────────────────────────────────────────────────────────

    /// Try to discover cluster from Ray Dashboard API at the given head node,
    /// over plain HTTP.
    pub fn discover_from_ray(head_ip: &str, ray_port: u16) -> Result<Self, String> {
        Self::discover_from_endpoint(&RayEndpoint {
            https: false,
            host: head_ip.to_string(),
            port: ray_port,
        })
    }

    /// Discover a cluster from a Ray Dashboard URL such as
    /// `https://ray.example.com:8265`, keeping its scheme.
    pub fn discover_from_ray_url(url: &str) -> Result<Self, String> {
        Self::discover_from_endpoint(&RayEndpoint::parse(url)?)
    }

    fn discover_from_endpoint(endpoint: &RayEndpoint) -> Result<Self, String> {
        let head_ip = endpoint.host.as_str();
        let ray_port = endpoint.port;
        let url = format!("{}/nodes?view=summary", endpoint.base_url());

        let resp = ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(5)))
            .build()
            .call()
            .map_err(|e| format!("Ray API request to {} failed: {}", url, e))?;

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

        let nodes = parse_ray_nodes(nodes_data, head_ip);

        Ok(ClusterConfig {
            name: format!("{}-node cluster", nodes.len()),
            head_ip: head_ip.to_string(),
            ray_port,
            nodes,
        })
    }

    /// Create a cluster config from manual node list.
    pub fn from_nodes(head_ip: &str, nodes: Vec<ClusterNode>) -> Self {
        ClusterConfig {
            name: format!("{}-node cluster", nodes.len()),
            head_ip: head_ip.to_string(),
            ray_port: 8265,
            nodes,
        }
    }
}

// ── Module-level helpers ───────────────────────────────────────────────────────

/// A Ray Dashboard address parsed from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RayEndpoint {
    pub https: bool,
    /// Host or IP, without IPv6 brackets, so it can be compared with the
    /// `nodeManagerAddress` Ray reports.
    pub host: String,
    pub port: u16,
}

impl RayEndpoint {
    /// Parse `http://10.0.0.1:8265`, `https://ray.example.com`, a bare
    /// `10.0.0.1:9000`, or `[fd00::1]:8265`. The scheme defaults to HTTP and
    /// the port to the standard Ray Dashboard port 8265; any path is ignored.
    pub fn parse(input: &str) -> Result<Self, String> {
        let raw = input.trim();
        if raw.is_empty() {
            return Err("Ray Dashboard URL is empty".to_string());
        }
        let (https, rest) = if let Some(rest) = raw.strip_prefix("https://") {
            (true, rest)
        } else {
            (false, raw.strip_prefix("http://").unwrap_or(raw))
        };
        let authority = rest.split('/').next().unwrap_or(rest);
        let invalid = || format!("Invalid Ray Dashboard URL '{}'", input);
        let bad_port = || format!("Invalid port in Ray Dashboard URL '{}'", input);
        let parse_port = |p: &str| match p.parse::<u16>() {
            Ok(port) if port != 0 => Ok(port),
            _ => Err(bad_port()),
        };

        let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
            let (host, after) = v6.split_once(']').ok_or_else(invalid)?;
            let port = match after {
                "" => default_ray_port(),
                _ => parse_port(after.strip_prefix(':').ok_or_else(invalid)?)?,
            };
            (host, port)
        } else {
            match authority.rsplit_once(':') {
                // An unbracketed IPv6 address has several colons and no
                // unambiguous port.
                Some((host, _)) if host.contains(':') => return Err(invalid()),
                Some((host, port)) => (host, parse_port(port)?),
                None => (authority, default_ray_port()),
            }
        };
        if host.is_empty() {
            return Err(invalid());
        }
        Ok(Self {
            https,
            host: host.to_string(),
            port,
        })
    }

    /// `scheme://host:port`, bracketing IPv6 hosts.
    pub fn base_url(&self) -> String {
        let scheme = if self.https { "https" } else { "http" };
        if self.host.contains(':') {
            format!("{}://[{}]:{}", scheme, self.host, self.port)
        } else {
            format!("{}://{}:{}", scheme, self.host, self.port)
        }
    }
}

/// Turn Ray `/nodes?view=summary` entries into cluster nodes, head first.
///
/// Ray lists a `GPU` resource only on nodes that have GPUs, so a missing key
/// means zero GPUs (a CPU-only head node is common). Ray reports neither GPU
/// model nor VRAM, so `gpu_vram_gb` is left at 0.0 for the caller to fill in.
fn parse_ray_nodes(nodes_data: &[serde_json::Value], head_ip: &str) -> Vec<ClusterNode> {
    let node_ip = |node: &serde_json::Value| -> Option<String> {
        node.get("raylet")
            .and_then(|r| r.get("nodeManagerAddress"))
            .and_then(|a| a.as_str())
            .map(str::to_string)
    };
    // Head is the node whose address matches `head_ip`. Ray does not order
    // nodes, so fall back to the first entry only when nothing matches.
    let head_index = nodes_data
        .iter()
        .position(|n| node_ip(n).as_deref() == Some(head_ip))
        .unwrap_or(0);

    let mut nodes = Vec::new();

    for (i, node) in nodes_data.iter().enumerate() {
        let default_hostname = format!("node-{}", i + 1);

        let ip = node_ip(node).unwrap_or_else(|| head_ip.to_string());

        let hostname = node
            .get("hostname")
            .and_then(|h: &serde_json::Value| h.as_str())
            .unwrap_or(&default_hostname)
            .to_string();

        let gpu_count = node
            .get("resources")
            .and_then(|r: &serde_json::Value| r.get("GPU"))
            .and_then(|g: &serde_json::Value| g.as_f64())
            .unwrap_or(0.0);

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

        let is_head = i == head_index;

        let gpu_count_u32 = gpu_count as u32;
        nodes.push(ClusterNode {
            hostname,
            ip,
            // Ray doesn't report GPU model name or VRAM per GPU.
            // gpu_vram_gb is set to 0.0 here and filled in via interactive
            // prompt after discover_from_ray returns.
            gpu_name: "GPU".to_string(),
            gpu_vram_gb: 0.0,
            total_ram_gb: total_ram,
            cpu_cores,
            gpu_count: gpu_count_u32,
            unified_memory: false,
            is_head,
            backend: None, // Ray doesn't report GPU backend
        });
    }

    // Sort: head first, then by hostname
    nodes.sort_by(|a, b| b.is_head.cmp(&a.is_head).then(a.hostname.cmp(&b.hostname)));
    nodes
}

/// The backend every node names, or `None` when any node leaves it unset or
/// two nodes disagree.
fn uniform_backend(nodes: &[&ClusterNode]) -> Option<GpuBackend> {
    let first = nodes.first()?.backend?;
    nodes
        .iter()
        .all(|n| n.backend == Some(first))
        .then_some(first)
}

/// Derive the GPU backend for the whole cluster from per-node `backend` fields:
/// the shared backend when all nodes agree, otherwise CUDA. Callers surface
/// the fallback through [`ClusterConfig::backend_warning`].
pub(crate) fn derive_cluster_backend(nodes: &[&ClusterNode]) -> GpuBackend {
    uniform_backend(nodes).unwrap_or(GpuBackend::Cuda)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // ClusterConfig — totals and spec conversion
    // ─────────────────────────────────────────────────────────────────────────

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
                    gpu_count: 1,
                    unified_memory: false,
                    is_head: true,
                    backend: None,
                },
                ClusterNode {
                    hostname: "node-2".to_string(),
                    ip: "10.0.0.2".to_string(),
                    gpu_name: "A100".to_string(),
                    gpu_vram_gb: 80.0,
                    total_ram_gb: 256.0,
                    cpu_cores: 64,
                    gpu_count: 1,
                    unified_memory: false,
                    is_head: false,
                    backend: None,
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

    /// `total_vram_gb()` multiplies per-GPU VRAM by each node's `gpu_count`;
    /// summing `gpu_vram_gb` alone would count one GPU per node.
    #[test]
    fn test_total_vram_counts_all_gpus_per_node() {
        let cluster = ClusterConfig::from_nodes(
            "10.0.0.1",
            vec![ClusterNode {
                hostname: "dgx".to_string(),
                ip: "10.0.0.1".to_string(),
                gpu_name: "H100".to_string(),
                gpu_vram_gb: 80.0,
                total_ram_gb: 2000.0,
                cpu_cores: 128,
                gpu_count: 8, // 8 × 80 GB = 640 GB
                unified_memory: false,
                is_head: true,
                backend: None,
            }],
        );
        assert!(
            (cluster.total_vram_gb() - 640.0).abs() < 0.01,
            "expected 8 × 80 = 640 GB, got {}",
            cluster.total_vram_gb()
        );
        let specs = cluster.to_system_specs();
        assert!(
            (specs.total_gpu_vram_gb.unwrap() - 640.0).abs() < 0.01,
            "SystemSpecs total_gpu_vram_gb should reflect gpu_count multiplier"
        );
    }

    /// unified_memory: true only when ALL nodes have unified_memory = true.
    #[test]
    fn test_unified_memory_all_nodes() {
        let cluster = ClusterConfig::from_nodes(
            "10.0.0.1",
            vec![
                ClusterNode {
                    hostname: "m4-1".to_string(),
                    ip: "10.0.0.1".to_string(),
                    gpu_name: "Apple M4 Max".to_string(),
                    gpu_vram_gb: 128.0,
                    total_ram_gb: 128.0,
                    cpu_cores: 16,
                    gpu_count: 1,
                    unified_memory: true,
                    is_head: true,
                    backend: None,
                },
                ClusterNode {
                    hostname: "m4-2".to_string(),
                    ip: "10.0.0.2".to_string(),
                    gpu_name: "Apple M4 Max".to_string(),
                    gpu_vram_gb: 128.0,
                    total_ram_gb: 128.0,
                    cpu_cores: 16,
                    gpu_count: 1,
                    unified_memory: true,
                    is_head: false,
                    backend: None,
                },
            ],
        );
        let specs = cluster.to_system_specs();
        assert!(
            specs.unified_memory,
            "all-unified cluster should report unified_memory: true"
        );
    }

    /// unified_memory: false when even one node lacks unified memory.
    #[test]
    fn test_mixed_unified_memory() {
        let cluster = ClusterConfig::from_nodes(
            "10.0.0.1",
            vec![
                ClusterNode {
                    hostname: "mac".to_string(),
                    ip: "10.0.0.1".to_string(),
                    gpu_name: "Apple M4".to_string(),
                    gpu_vram_gb: 128.0,
                    total_ram_gb: 128.0,
                    cpu_cores: 16,
                    gpu_count: 1,
                    unified_memory: true,
                    is_head: true,
                    backend: None,
                },
                ClusterNode {
                    hostname: "dgx".to_string(),
                    ip: "10.0.0.2".to_string(),
                    gpu_name: "H100".to_string(),
                    gpu_vram_gb: 80.0,
                    total_ram_gb: 512.0,
                    cpu_cores: 64,
                    gpu_count: 1,
                    unified_memory: false,
                    is_head: false,
                    backend: None,
                },
            ],
        );
        let specs = cluster.to_system_specs();
        assert!(
            !specs.unified_memory,
            "mixed-unified cluster should NOT report unified_memory: true"
        );
    }

    /// Serde roundtrip for ClusterNode.backend (new optional field).
    #[test]
    fn test_backend_serde_roundtrip() {
        let mut cluster = make_test_cluster();
        cluster.nodes[0].backend = Some(GpuBackend::Rocm);
        // Serialized TOML should include the backend field for node-1 only
        let toml_str = toml::to_string_pretty(&cluster).unwrap();
        assert!(
            toml_str.contains("backend"),
            "serialized TOML should include backend field when set"
        );
        let loaded: ClusterConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(
            loaded.nodes[0].backend,
            Some(GpuBackend::Rocm),
            "ROCm backend should survive TOML roundtrip"
        );
        assert_eq!(
            loaded.nodes[1].backend, None,
            "node without backend should deserialize as None"
        );
    }

    /// Existing TOML without a backend field should parse cleanly (backward compat).
    #[test]
    fn test_backend_missing_in_toml_is_none() {
        let toml_str = r#"
name = "2-node cluster"
head_ip = "10.0.0.1"
ray_port = 8265
interconnect = "ethernet"

[[nodes]]
hostname = "node-1"
ip = "10.0.0.1"
gpu_name = "A100"
gpu_vram_gb = 80.0
total_ram_gb = 256.0
cpu_cores = 64
gpu_count = 1
is_head = true
"#;
        let cluster: ClusterConfig =
            ClusterConfig::from_toml(toml_str).expect("should parse without backend field");
        assert_eq!(cluster.nodes[0].backend, None);
    }

    /// Configs saved before `interconnect` was removed still load: the key is
    /// ignored rather than rejected.
    #[test]
    fn test_legacy_interconnect_key_is_ignored() {
        let toml_str = r#"
name = "1-node cluster"
head_ip = "10.0.0.1"
interconnect = "nvlink"

[[nodes]]
hostname = "node-1"
ip = "10.0.0.1"
gpu_name = "H100"
gpu_vram_gb = 80.0
total_ram_gb = 512.0
cpu_cores = 64
"#;
        let cluster: ClusterConfig =
            ClusterConfig::from_toml(toml_str).expect("legacy interconnect key should be ignored");
        assert_eq!(cluster.node_count(), 1);
        assert!(!toml::to_string(&cluster).unwrap().contains("interconnect"));
    }

    /// derive_cluster_backend returns Cuda when all backends are unspecified.
    #[test]
    fn test_derive_backend_unspecified() {
        let nodes = [ClusterNode {
            hostname: "n".to_string(),
            ip: "10.0.0.1".to_string(),
            gpu_name: "A100".to_string(),
            gpu_vram_gb: 80.0,
            total_ram_gb: 256.0,
            cpu_cores: 64,
            gpu_count: 1,
            unified_memory: false,
            is_head: true,
            backend: None,
        }];
        // Defaults to Cuda; the warning is covered by backend_warning tests.
        let backend = derive_cluster_backend(&nodes.iter().collect::<Vec<_>>());
        assert_eq!(backend, GpuBackend::Cuda);
    }

    /// derive_cluster_backend propagates a uniform explicit backend.
    #[test]
    fn test_derive_backend_uniform_rocm() {
        let nodes = [
            ClusterNode {
                hostname: "n1".to_string(),
                ip: "10.0.0.1".to_string(),
                gpu_name: "MI300X".to_string(),
                gpu_vram_gb: 192.0,
                total_ram_gb: 512.0,
                cpu_cores: 96,
                gpu_count: 8,
                unified_memory: false,
                is_head: true,
                backend: Some(GpuBackend::Rocm),
            },
            ClusterNode {
                hostname: "n2".to_string(),
                ip: "10.0.0.2".to_string(),
                gpu_name: "MI300X".to_string(),
                gpu_vram_gb: 192.0,
                total_ram_gb: 512.0,
                cpu_cores: 96,
                gpu_count: 8,
                unified_memory: false,
                is_head: false,
                backend: Some(GpuBackend::Rocm),
            },
        ];
        assert_eq!(
            derive_cluster_backend(&nodes.iter().collect::<Vec<_>>()),
            GpuBackend::Rocm
        );
    }

    /// Ray lists a `GPU` resource only on nodes that have GPUs. A CPU-only
    /// head node must come back with zero GPUs, not one phantom card.
    #[test]
    fn ray_endpoint_accepts_common_shapes() {
        let ep = |https, host: &str, port| RayEndpoint {
            https,
            host: host.to_string(),
            port,
        };
        assert_eq!(
            RayEndpoint::parse("http://10.0.0.1:8265"),
            Ok(ep(false, "10.0.0.1", 8265))
        );
        assert_eq!(
            RayEndpoint::parse("10.0.0.1:9000"),
            Ok(ep(false, "10.0.0.1", 9000))
        );
        assert_eq!(
            RayEndpoint::parse("https://ray.example.com/nodes"),
            Ok(ep(true, "ray.example.com", 8265))
        );
        assert_eq!(
            RayEndpoint::parse(" ray-head.local "),
            Ok(ep(false, "ray-head.local", 8265))
        );
        assert_eq!(
            RayEndpoint::parse("http://[fd00::1]:9000"),
            Ok(ep(false, "fd00::1", 9000))
        );
        assert_eq!(
            RayEndpoint::parse("[fd00::1]"),
            Ok(ep(false, "fd00::1", 8265))
        );
    }

    #[test]
    fn ray_endpoint_rejects_malformed_input() {
        for bad in [
            "",
            "   ",
            "http://",
            "10.0.0.1:abc",
            "10.0.0.1:0",
            "/8265",
            "fd00::1",
            "[fd00::1",
            "[fd00::1]x",
        ] {
            assert!(
                RayEndpoint::parse(bad).is_err(),
                "expected parse error for {:?}",
                bad
            );
        }
    }

    /// An `https://` dashboard URL must be requested over HTTPS, not
    /// rewritten to plain HTTP.
    #[test]
    fn ray_endpoint_base_url_keeps_scheme() {
        let url = |s| RayEndpoint::parse(s).map(|e| e.base_url());
        assert_eq!(
            url("https://ray.example.com:443"),
            Ok("https://ray.example.com:443".to_string())
        );
        assert_eq!(url("10.0.0.1"), Ok("http://10.0.0.1:8265".to_string()));
        assert_eq!(
            url("[fd00::1]:9000"),
            Ok("http://[fd00::1]:9000".to_string())
        );
    }

    #[test]
    fn test_system_specs_without_gpus_report_no_gpu() {
        let mut cfg = make_test_cluster();
        for node in &mut cfg.nodes {
            node.gpu_count = 0;
        }
        let specs = cfg.to_system_specs();
        assert!(!specs.has_gpu);
        assert_eq!(specs.gpu_count, 0);
        assert_eq!(specs.total_gpu_vram_gb, Some(0.0));
    }

    /// Kernel checks read the compute capability from `gpu_name`, so a mixed
    /// cluster must report its oldest GPU, wherever it sits in the config.
    #[test]
    fn test_system_specs_report_the_oldest_gpu_architecture() {
        let node = |host: &str, gpu: &str| ClusterNode {
            hostname: host.to_string(),
            ip: "10.0.0.1".to_string(),
            gpu_name: gpu.to_string(),
            gpu_vram_gb: 80.0,
            total_ram_gb: 256.0,
            cpu_cores: 32,
            gpu_count: 4,
            unified_memory: false,
            is_head: false,
            backend: Some(GpuBackend::Cuda),
        };
        let cfg = ClusterConfig::from_nodes(
            "10.0.0.1",
            vec![
                node("a", "NVIDIA A100-SXM4-80GB"),
                node("b", "Tesla V100-SXM2-32GB"),
            ],
        );
        let specs = cfg.to_system_specs();
        let name = specs.gpu_name.expect("cluster reports a GPU name");
        assert!(name.starts_with("Tesla V100"), "{name}");
        assert!(name.ends_with("(×8, mixed)"), "{name}");
        // V100 (7.0) cannot run AWQ (needs 7.5), so neither can the cluster.
        assert_eq!(
            crate::hardware::quant_kernel_support("AWQ-4bit", &name),
            crate::hardware::QuantKernelSupport::Unsupported
        );

        // A card llmfit cannot identify (Ray's generic "GPU") might be the
        // oldest, so the cluster must not claim the A100's kernel support.
        let unknown = ClusterConfig::from_nodes(
            "10.0.0.1",
            vec![node("a", "NVIDIA A100-SXM4-80GB"), node("b", "GPU")],
        );
        let name = unknown
            .to_system_specs()
            .gpu_name
            .expect("cluster reports a GPU name");
        assert_eq!(name, "GPU (×8, mixed)");
        assert_eq!(
            crate::hardware::quant_kernel_support("AWQ-4bit", &name),
            crate::hardware::QuantKernelSupport::Unknown
        );

        let uniform = ClusterConfig::from_nodes(
            "10.0.0.1",
            vec![
                node("a", "NVIDIA A100-SXM4-80GB"),
                node("b", "NVIDIA A100-SXM4-80GB"),
            ],
        );
        assert_eq!(
            uniform.to_system_specs().gpu_name.as_deref(),
            Some("NVIDIA A100-SXM4-80GB (×8)")
        );
    }

    /// On a mixed cluster, every GPU holds an equal tensor-parallel shard, so
    /// the pool is the smallest card times the GPU count.
    #[test]
    fn test_system_specs_pool_bounded_by_smallest_gpu() {
        let node = |host: &str, vram: f64| ClusterNode {
            hostname: host.to_string(),
            ip: "10.0.0.1".to_string(),
            gpu_name: "GPU".to_string(),
            gpu_vram_gb: vram,
            total_ram_gb: 256.0,
            cpu_cores: 32,
            gpu_count: 2,
            unified_memory: false,
            is_head: false,
            backend: Some(GpuBackend::Cuda),
        };
        let cfg = ClusterConfig::from_nodes("10.0.0.1", vec![node("a", 80.0), node("b", 24.0)]);
        assert_eq!(cfg.total_vram_gb(), 208.0);

        let specs = cfg.to_system_specs();
        assert_eq!(specs.gpu_vram_gb, Some(24.0));
        assert_eq!(specs.total_gpu_vram_gb, Some(96.0));
        assert_eq!(specs.gpu_count, 4);
    }

    #[test]
    fn conflicting_backends_ignores_unset_and_cpu_only_nodes() {
        let mut cfg = make_test_cluster();
        for node in &mut cfg.nodes {
            node.backend = Some(GpuBackend::Cuda);
        }
        assert!(cfg.conflicting_backends().is_empty());

        // Unset backends are a warning, not a conflict.
        cfg.nodes[1].backend = None;
        assert!(cfg.conflicting_backends().is_empty());

        cfg.nodes[1].backend = Some(GpuBackend::Metal);
        assert_eq!(cfg.conflicting_backends(), vec!["cuda", "metal"]);

        // A CPU-only node's backend takes no part in the GPU pool.
        cfg.nodes[1].gpu_count = 0;
        assert!(cfg.conflicting_backends().is_empty());
    }

    /// `--ram` on a unified-memory cluster is the cluster-wide total, so each
    /// GPU's VRAM is its share, not the whole pool.
    #[test]
    fn ram_override_splits_unified_pool_across_cluster_gpus() {
        let mut cfg = make_test_cluster();
        for node in &mut cfg.nodes {
            node.unified_memory = true;
            node.gpu_count = 1;
            node.gpu_vram_gb = 128.0;
            node.total_ram_gb = 128.0;
        }
        let gpus = cfg.total_gpu_count() as f64;
        let specs = cfg.to_system_specs();
        assert!(specs.unified_memory);

        let over = specs.clone().with_ram_override(64.0 * gpus);
        assert_eq!(over.gpu_vram_gb, Some(64.0));
        assert_eq!(over.total_gpu_vram_gb, Some(64.0 * gpus));
        assert_eq!(over.gpus[0].vram_gb, Some(64.0));

        let half = specs.with_ram_percent(50.0);
        assert_eq!(half.gpu_vram_gb, Some(64.0));
        assert_eq!(half.total_gpu_vram_gb, Some(64.0 * gpus));
        assert_eq!(half.gpus[0].vram_gb, Some(64.0));
    }

    #[test]
    fn backend_warning_only_when_backends_disagree_or_are_missing() {
        let mut cfg = make_test_cluster();
        for node in &mut cfg.nodes {
            node.backend = Some(GpuBackend::Rocm);
        }
        assert_eq!(cfg.backend_warning(), None);

        cfg.nodes[0].backend = Some(GpuBackend::Cuda);
        assert!(cfg.backend_warning().is_some());
        assert_eq!(cfg.to_system_specs().backend, GpuBackend::Cuda);

        cfg.nodes[0].backend = None;
        assert!(cfg.backend_warning().is_some());
    }

    /// A misspelled key must be an error, not a silently applied default:
    /// `gpu_cout = 8` would otherwise mean one GPU.
    #[test]
    fn from_toml_rejects_unknown_keys() {
        let node = |extra: &str| {
            format!(
                "name = \"c\"\nhead_ip = \"10.0.0.1\"\n{extra}\n[[nodes]]\nhostname = \"a\"\nip = \"10.0.0.1\"\ngpu_name = \"H100\"\ngpu_vram_gb = 80.0\ntotal_ram_gb = 512.0\ncpu_cores = 64\n"
            )
        };
        let typo = node("") + "gpu_cout = 8\n";
        let err = ClusterConfig::from_toml(&typo).expect_err("typo must be rejected");
        assert!(err.contains("gpu_cout"), "{err}");

        let err = ClusterConfig::from_toml(&node("raay_port = 1"))
            .expect_err("top-level typo must be rejected");
        assert!(err.contains("raay_port"), "{err}");

        // The legacy key is still accepted, and real errors still surface
        // alongside it.
        assert!(ClusterConfig::from_toml(&node("interconnect = \"qsfp\"")).is_ok());
        let err = ClusterConfig::from_toml(&(node("interconnect = \"qsfp\"") + "gpu_cout = 8\n"))
            .expect_err("typo next to legacy key must be rejected");
        assert!(err.contains("gpu_cout"), "{err}");
    }

    /// Hand-edited configs are the common case, so a parse error must say
    /// where the problem is.
    #[test]
    fn from_toml_reports_location_of_errors() {
        let err = ClusterConfig::from_toml("name = \"x\"\nhead_ip = \nnodes = []\n")
            .expect_err("missing value must not parse");
        assert!(err.contains("line 2"), "error should name the line: {err}");
    }

    #[test]
    fn test_parse_ray_nodes_cpu_only_head_has_no_gpus() {
        let summary = serde_json::json!([
            {
                "hostname": "worker-1",
                "raylet": { "nodeManagerAddress": "10.0.0.2" },
                "resources": { "CPU": 64.0, "GPU": 8.0, "memory": 549_755_813_888.0_f64 }
            },
            {
                "hostname": "head",
                "raylet": { "nodeManagerAddress": "10.0.0.1" },
                "resources": { "CPU": 16.0, "memory": 68_719_476_736.0_f64 }
            }
        ]);
        let nodes = parse_ray_nodes(summary.as_array().unwrap(), "10.0.0.1");

        assert_eq!(nodes.len(), 2);
        let head = &nodes[0];
        assert!(head.is_head);
        assert_eq!(head.hostname, "head");
        assert_eq!(head.gpu_count, 0);
        assert_eq!(head.cpu_cores, 16);
        assert_eq!(head.total_ram_gb, 64.0);

        let worker = &nodes[1];
        assert!(!worker.is_head);
        assert_eq!(worker.gpu_count, 8);
        assert_eq!(worker.gpu_vram_gb, 0.0);
        assert_eq!(worker.total_ram_gb, 512.0);

        // Once VRAM is filled in for GPU nodes, totals count only real cards.
        let mut cfg = ClusterConfig::from_nodes("10.0.0.1", nodes);
        for n in cfg.nodes.iter_mut().filter(|n| n.gpu_count > 0) {
            n.gpu_vram_gb = 80.0;
        }
        assert_eq!(cfg.total_gpu_count(), 8);
        assert_eq!(cfg.total_vram_gb(), 640.0);
    }

    /// A CPU-only head sorts first but must not supply the per-GPU fields:
    /// per-GPU VRAM, GPU name, backend and memory model come from GPU nodes.
    #[test]
    fn test_system_specs_ignore_cpu_only_head() {
        let mut cfg = make_test_cluster();
        // node-1 becomes a CPU-only head with no backend and 0 GB.
        cfg.nodes[0].gpu_count = 0;
        cfg.nodes[0].gpu_vram_gb = 0.0;
        cfg.nodes[0].gpu_name = "GPU".to_string();
        cfg.nodes[0].backend = None;
        // node-2 is the only GPU worker: 8x MI300X with an explicit backend.
        cfg.nodes[1].gpu_count = 8;
        cfg.nodes[1].gpu_vram_gb = 192.0;
        cfg.nodes[1].gpu_name = "MI300X".to_string();
        cfg.nodes[1].backend = Some(GpuBackend::Rocm);
        assert!(cfg.nodes[0].is_head);

        let specs = cfg.to_system_specs();
        assert_eq!(specs.gpu_vram_gb, Some(192.0));
        assert_eq!(specs.gpus[0].vram_gb, Some(192.0));
        assert_eq!(specs.total_gpu_vram_gb, Some(1536.0));
        assert_eq!(specs.gpu_count, 8);
        assert_eq!(specs.gpu_name.as_deref(), Some("MI300X (×8)"));
        assert_eq!(specs.backend, GpuBackend::Rocm);
        // RAM and cores still aggregate over every node, head included.
        assert_eq!(specs.total_ram_gb, 512.0);
        assert_eq!(specs.total_cpu_cores, 128);
    }
}
