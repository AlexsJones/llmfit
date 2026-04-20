use std::collections::BTreeMap;
use sysinfo::System;

/// The acceleration backend for inference speed estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum GpuBackend {
    Cuda,
    Metal,
    Rocm,
    Vulkan, // AMD/other GPUs without ROCm (e.g. Windows AMD, older AMD)
    Sycl,   // Intel oneAPI
    CpuArm,
    CpuX86,
    Ascend,
}

impl GpuBackend {
    pub fn label(&self) -> &'static str {
        match self {
            GpuBackend::Cuda => "CUDA",
            GpuBackend::Metal => "Metal",
            GpuBackend::Rocm => "ROCm",
            GpuBackend::Vulkan => "Vulkan",
            GpuBackend::Sycl => "SYCL",
            GpuBackend::CpuArm => "CPU (ARM)",
            GpuBackend::CpuX86 => "CPU (x86)",
            GpuBackend::Ascend => "NPU (Ascend)",
        }
    }
}

/// Information about a single detected GPU.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_gb: Option<f64>,
    pub backend: GpuBackend,
    pub count: u32, // >1 for same-model multi-GPU (e.g. 2x RTX 4090)
    pub unified_memory: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemSpecs {
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub total_cpu_cores: usize,
    pub cpu_name: String,
    pub has_gpu: bool,
    pub gpu_vram_gb: Option<f64>,
    /// Total VRAM across all same-model GPUs (e.g., 48GB for 2x RTX 3090).
    /// For multi-GPU inference backends (llama.cpp, vLLM), models can be split
    /// across cards, so we use total VRAM for fit scoring.
    pub total_gpu_vram_gb: Option<f64>,
    pub gpu_name: Option<String>,
    pub gpu_count: u32,
    pub unified_memory: bool,
    pub backend: GpuBackend,
    /// All detected GPUs (may span different vendors/backends).
    pub gpus: Vec<GpuInfo>,
    /// True when running in multi-node cluster mode (e.g. DGX Spark cluster).
    pub cluster_mode: bool,
    /// Number of nodes in the cluster (0 or 1 = single machine).
    pub cluster_node_count: u32,
}

impl SystemSpecs {
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let total_ram_bytes = sys.total_memory();
        let available_ram_bytes = sys.available_memory();
        let total_ram_gb = total_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_ram_gb = if available_ram_bytes == 0 && total_ram_bytes > 0 {
            // sysinfo may fail to report available memory on some platforms
            // (e.g. macOS Tahoe / newer macOS versions). Try fallbacks.
            Self::available_ram_fallback(&sys, total_ram_bytes, total_ram_gb)
        } else {
            available_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        };

        let total_cpu_cores = sys.cpus().len();
        let cpu_name = Self::detect_cpu_name(&sys);

        let gpus = Self::detect_all_gpus(total_ram_gb, &cpu_name);

        // Primary GPU = the one with the most VRAM (best for inference).
        // For fit scoring, we use the primary GPU's VRAM pool.
        let primary = gpus.first();
        let has_gpu = !gpus.is_empty();
        let gpu_vram_gb = primary.and_then(|g| g.vram_gb);
        // Total VRAM = per-card VRAM * count (for multi-GPU tensor splitting)
        let total_gpu_vram_gb = primary.and_then(|g| g.vram_gb.map(|vram| vram * g.count as f64));
        let gpu_name = primary.map(|g| g.name.clone());
        let gpu_count = primary.map(|g| g.count).unwrap_or(0);
        let unified_memory = primary.map(|g| g.unified_memory).unwrap_or(false);

        let cpu_backend =
            if cfg!(target_arch = "aarch64") || cpu_name.to_lowercase().contains("apple") {
                GpuBackend::CpuArm
            } else {
                GpuBackend::CpuX86
            };
        let backend = primary.map(|g| g.backend).unwrap_or(cpu_backend);

        SystemSpecs {
            total_ram_gb,
            available_ram_gb,
            total_cpu_cores,
            cpu_name,
            has_gpu,
            gpu_vram_gb,
            total_gpu_vram_gb,
            gpu_name,
            gpu_count,
            unified_memory,
            backend,
            gpus,
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    /// Detect all GPUs across all vendors. Returns a Vec sorted by VRAM descending
    /// (best GPU first). Unlike the old cascade, this does NOT short-circuit:
    /// a system with both NVIDIA and AMD GPUs will report both.
    fn detect_all_gpus(total_ram_gb: f64, cpu_name: &str) -> Vec<GpuInfo> {
        let mut gpus = Vec::new();

        // NVIDIA GPUs via nvidia-smi, with sysfs fallback for Linux/toolbox setups
        let nvidia = Self::detect_nvidia_gpus();
        if nvidia.is_empty() {
            if let Some(nvidia_sysfs) = Self::detect_nvidia_gpu_sysfs_info() {
                gpus.push(nvidia_sysfs);
            }
        } else {
            gpus.extend(nvidia);
        }

        // AMD GPUs via rocm-smi or sysfs
        if let Some(amd) = Self::detect_amd_gpu_rocm_info() {
            gpus.push(amd);
        } else if let Some(amd) = Self::detect_amd_gpu_sysfs_info() {
            gpus.push(amd);
        }

        // Windows WMI (catches GPUs not found by vendor-specific tools)
        for wmi_gpu in Self::detect_gpu_windows_info() {
            // Skip if we already found a GPU with the same name from a vendor tool
            let dominated = gpus.iter().any(|existing| {
                let existing_lower = existing.name.to_lowercase();
                let wmi_lower = wmi_gpu.name.to_lowercase();
                existing_lower.contains(&wmi_lower) || wmi_lower.contains(&existing_lower)
            });
            if !dominated {
                gpus.push(wmi_gpu);
            }
        }

        // AMD unified memory APUs (e.g. Ryzen AI MAX series).
        // These share the full system RAM between CPU and GPU, like Apple Silicon.
        // WMI AdapterRAM is a 32-bit field capped at ~4 GB, so we override with
        // total system RAM for these APUs.
        //
        // On Windows, BIOS GPU UMA carveouts cause sysinfo to report only the
        // CPU-accessible portion (e.g. 32 GB on a 128 GB system where 96 GB is
        // allocated to the GPU). Query total physical DIMM capacity via
        // Win32_PhysicalMemory, which reads SMBIOS and is unaffected by the
        // carveout, so model fit estimates reflect the full memory pool.
        if is_amd_unified_memory_apu(cpu_name) {
            let apu_pool_gb = detect_windows_physical_total_ram_gb().unwrap_or(total_ram_gb);
            let amd_idx = gpus.iter().position(|g| {
                let lower = g.name.to_lowercase();
                lower.contains("amd") || lower.contains("radeon")
            });
            if let Some(idx) = amd_idx {
                gpus[idx].unified_memory = true;
                gpus[idx].vram_gb = Some(apu_pool_gb);
            } else {
                // No AMD GPU found via other methods; create one.
                gpus.push(GpuInfo {
                    name: format!("{} (integrated)", cpu_name),
                    vram_gb: Some(apu_pool_gb),
                    backend: GpuBackend::Vulkan,
                    count: 1,
                    unified_memory: true,
                });
            }
        }

        // NVIDIA Grace / DGX Spark unified memory SoCs (e.g. GB10, GB20).
        // These share the full system RAM between CPU and GPU, like Apple Silicon.
        // nvidia-smi may report 0 VRAM or a small dedicated portion, so we
        // override with total system RAM and flag as unified memory.
        // Inside Docker the friendly name may be missing; we also match by PCI
        // device ID (e.g. "Device [10de:2e12]").
        let is_nvidia_unified = gpus.iter().any(|g| is_nvidia_unified_memory_gpu(&g.name));
        if is_nvidia_unified {
            for gpu in &mut gpus {
                if is_nvidia_unified_memory_gpu(&gpu.name) {
                    gpu.unified_memory = true;
                    gpu.vram_gb = Some(total_ram_gb);
                }
            }
        }

        // Intel Arc via sysfs
        if let Some(vram) = Self::detect_intel_gpu() {
            let already_found = gpus.iter().any(|g| g.name.to_lowercase().contains("intel"));
            if !already_found {
                gpus.push(GpuInfo {
                    name: "Intel Arc".to_string(),
                    vram_gb: Some(vram),
                    backend: GpuBackend::Sycl,
                    count: 1,
                    unified_memory: false,
                });
            }
        }

        // Apple Silicon (unified memory)
        if let Some(vram) = Self::detect_apple_gpu(total_ram_gb) {
            let name = if cpu_name.to_lowercase().contains("apple") {
                cpu_name.to_string()
            } else {
                "Apple Silicon".to_string()
            };
            gpus.push(GpuInfo {
                name,
                vram_gb: Some(vram),
                backend: GpuBackend::Metal,
                count: 1,
                unified_memory: true,
            });
        }

        // Ascend NPUs via npu-smi
        let ascend = Self::detect_ascend_npus();
        if !ascend.is_empty() {
            gpus.extend(ascend);
        }

        // Vulkan fallback (e.g. Android/Termux with Turnip)
        for vulkan_gpu in Self::detect_vulkan_gpu_info() {
            let dominated = gpus
                .iter()
                .any(|existing| Self::is_same_gpu_name(&existing.name, &vulkan_gpu.name));
            if !dominated {
                gpus.push(vulkan_gpu);
            }
        }

        // When both discrete and integrated GPUs are present, drop the
        // integrated GPUs so the discrete GPU becomes primary. This applies
        // globally, not just to the Windows WMI path, to handle cases where
        // an iGPU is detected via Vulkan or APU detection alongside a dGPU.
        gpus = Self::prefer_discrete_gpus(gpus);

        // Sort by VRAM descending so the best GPU is primary
        gpus.sort_by(|a, b| {
            let va = a.vram_gb.unwrap_or(0.0);
            let vb = b.vram_gb.unwrap_or(0.0);
            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
        });

        gpus
    }

    /// Detect NVIDIA GPUs via nvidia-smi. Returns one GpuInfo per unique model,
    /// with count and per-card VRAM for same-model multi-GPU setups.
    ///
    /// First tries querying `addressing_mode` to detect unified memory (Tegra/Grace
    /// Blackwell platforms). Falls back to the standard 2-column query if the field
    /// is unavailable on older nvidia-smi versions.
    fn detect_nvidia_gpus() -> Vec<GpuInfo> {
        // Try the extended query first (addressing_mode,memory.total,name).
        // On NVIDIA Tegra / Grace Blackwell, addressing_mode returns "ATS"
        // (Address Translation Services) which signals unified CPU+GPU memory.
        if let Some(gpus) = Self::try_nvidia_smi_with_addressing_mode() {
            return gpus;
        }

        // Fallback: standard 2-column query for older nvidia-smi versions
        let output = match std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.total,name")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            Ok(o) if o.status.success() => o,
            _ => return Vec::new(),
        };

        let text = match String::from_utf8(output.stdout) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        Self::parse_nvidia_smi_list(&text)
    }

    /// Try nvidia-smi with `addressing_mode` column. Returns `None` if the
    /// query fails (e.g. older driver that doesn't support the field), so the
    /// caller can fall back to the standard query.
    fn try_nvidia_smi_with_addressing_mode() -> Option<Vec<GpuInfo>> {
        let output = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=addressing_mode,memory.total,name")
            .arg("--format=csv,noheader,nounits")
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let text = String::from_utf8(output.stdout).ok()?;
        Some(Self::parse_nvidia_smi_extended(&text))
    }

    /// Parse `nvidia-smi --query-gpu=addressing_mode,memory.total,name`.
    /// Detects unified memory when addressing_mode is "ATS" and VRAM is
    /// unavailable — common on NVIDIA Tegra / Grace Blackwell (DGX Spark).
    /// Falls back to system RAM via /proc/meminfo as the unified memory pool.
    fn parse_nvidia_smi_extended(text: &str) -> Vec<GpuInfo> {
        // Track per-model: (count, per_card_vram_mb, is_unified)
        let mut grouped: BTreeMap<String, (u32, f64, bool)> = BTreeMap::new();
        let total_ram_gb = read_proc_meminfo_total_gb();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(3, ',').collect();
            if parts.len() < 3 {
                continue;
            }

            let addr_mode = parts[0].trim();
            let is_unified = addr_mode.eq_ignore_ascii_case("ATS");

            let name = parts[2].trim().to_string();
            let name = if name.is_empty() {
                "NVIDIA GPU".to_string()
            } else {
                name
            };

            let parsed_vram_mb = parts[1].trim().parse::<f64>().unwrap_or(0.0);

            let vram_mb = if parsed_vram_mb > 0.0 {
                parsed_vram_mb
            } else if is_unified {
                // Unified memory: use total system RAM as the shared pool
                total_ram_gb.unwrap_or(0.0) * 1024.0
            } else {
                estimate_vram_from_name(&name) * 1024.0
            };

            let entry = grouped.entry(name).or_insert((0, 0.0, false));
            entry.0 += 1;
            if vram_mb > entry.1 {
                entry.1 = vram_mb;
            }
            if is_unified {
                entry.2 = true;
            }
        }

        if grouped.is_empty() {
            return Vec::new();
        }

        grouped
            .into_iter()
            .map(|(name, (count, per_card_vram_mb, is_unified))| GpuInfo {
                name,
                vram_gb: if per_card_vram_mb > 0.0 {
                    Some(per_card_vram_mb / 1024.0)
                } else {
                    None
                },
                backend: GpuBackend::Cuda,
                count,
                unified_memory: is_unified,
            })
            .collect()
    }

    /// Parse `nvidia-smi --query-gpu=memory.total,name --format=csv,noheader,nounits`.
    /// Groups same-model cards and keeps per-card VRAM (never sums across cards).
    fn parse_nvidia_smi_list(text: &str) -> Vec<GpuInfo> {
        let mut grouped: BTreeMap<String, (u32, f64)> = BTreeMap::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, ',').collect();

            let name = parts
                .get(1)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("NVIDIA GPU")
                .to_string();

            let parsed_vram_mb = parts
                .first()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(0.0);
            let vram_mb = if parsed_vram_mb > 0.0 {
                parsed_vram_mb
            } else {
                estimate_vram_from_name(&name) * 1024.0
            };

            let entry = grouped.entry(name).or_insert((0, 0.0));
            entry.0 += 1;
            if vram_mb > entry.1 {
                entry.1 = vram_mb;
            }
        }

        if grouped.is_empty() {
            return Vec::new();
        }

        grouped
            .into_iter()
            .map(|(name, (count, per_card_vram_mb))| GpuInfo {
                name,
                vram_gb: if per_card_vram_mb > 0.0 {
                    Some(per_card_vram_mb / 1024.0)
                } else {
                    None
                },
                backend: GpuBackend::Cuda,
                count,
                unified_memory: false,
            })
            .collect()
    }

    /// Detect NVIDIA GPUs via Linux sysfs when nvidia-smi is unavailable.
    /// This is common in containerized environments (e.g. Toolbx) and
    /// Nouveau-based systems.
    fn detect_nvidia_gpu_sysfs_info() -> Option<GpuInfo> {
        if !cfg!(target_os = "linux") {
            return None;
        }

        let entries = std::fs::read_dir("/sys/class/drm").ok()?;
        let mut gpu_count: u32 = 0;
        let mut total_vram_bytes: u64 = 0;
        let mut slot_hints: Vec<String> = Vec::new();
        let mut backend = GpuBackend::Vulkan;

        for entry in entries.flatten() {
            let card_path = entry.path();
            let fname = card_path.file_name()?.to_str()?.to_string();
            // Only look at cardN entries, not connectors (cardN-DP-1, etc.)
            if !fname.starts_with("card") || fname.contains('-') {
                continue;
            }

            let device_path = card_path.join("device");
            let vendor_path = device_path.join("vendor");
            let Ok(vendor) = std::fs::read_to_string(&vendor_path) else {
                continue;
            };
            if vendor.trim() != "0x10de" {
                continue;
            }

            gpu_count += 1;

            if let Ok(vram_str) = std::fs::read_to_string(device_path.join("mem_info_vram_total"))
                && let Ok(vram_bytes) = vram_str.trim().parse::<u64>()
                && vram_bytes > 0
            {
                // Track the maximum per-card VRAM instead of summing across all cards.
                total_vram_bytes = total_vram_bytes.max(vram_bytes);
            }

            if let Ok(uevent) = std::fs::read_to_string(device_path.join("uevent")) {
                for line in uevent.lines() {
                    if let Some(slot) = line.strip_prefix("PCI_SLOT_NAME=") {
                        slot_hints.push(slot.to_string());
                    } else if let Some(driver) = line.strip_prefix("DRIVER=")
                        && driver.eq_ignore_ascii_case("nvidia")
                    {
                        backend = GpuBackend::Cuda;
                    }
                }
            }
        }

        if gpu_count == 0 {
            return None;
        }

        let name = Self::get_nvidia_gpu_name_lspci(&slot_hints)
            .unwrap_or_else(|| "NVIDIA GPU".to_string());

        let mut vram_gb = if total_vram_bytes > 0 {
            Some(total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else {
            None
        };

        if vram_gb.is_none() {
            let est = estimate_vram_from_name(&name);
            if est > 0.0 {
                vram_gb = Some(est);
            }
        }

        let unified_memory = is_nvidia_unified_memory_gpu(&name);

        Some(GpuInfo {
            name,
            vram_gb,
            backend,
            count: gpu_count,
            unified_memory,
        })
    }

    /// Detect AMD GPU via rocm-smi (available on Linux with ROCm installed).
    /// Parses per-card VRAM and GPU name from rocm-smi output.
    fn detect_amd_gpu_rocm_info() -> Option<GpuInfo> {
        // Try rocm-smi --showmeminfo vram for VRAM
        let vram_output = std::process::Command::new("rocm-smi")
            .arg("--showmeminfo")
            .arg("vram")
            .output()
            .ok()?;

        if !vram_output.status.success() {
            return None;
        }

        let vram_text = String::from_utf8(vram_output.stdout).ok()?;

        // Parse VRAM total from rocm-smi output.
        // Typical format includes a line like:
        //   "GPU[0] : vram Total Memory (B): 8589934592"
        // or in table format with "Total" and bytes.
        let mut per_gpu_vram_bytes: Vec<u64> = Vec::new();
        for line in vram_text.lines() {
            let lower = line.to_lowercase();
            if lower.contains("total") && !lower.contains("used") {
                // Extract the numeric value (bytes)
                if let Some(val) = line
                    .split_whitespace()
                    .filter_map(|w| w.parse::<u64>().ok())
                    .next_back()
                    && val > 0
                {
                    per_gpu_vram_bytes.push(val);
                }
            }
        }

        // Filter out integrated GPUs (iGPUs) that have very little VRAM.
        // rocm-smi reports all GPU agents including iGPUs on APUs like
        // Ryzen 9800X3D, which would otherwise inflate the GPU count.
        // Discrete GPUs have >= 2 GB VRAM; iGPUs typically show < 1 GB.
        const IGPU_VRAM_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024; // 2 GB
        let discrete_vram: Vec<u64> = per_gpu_vram_bytes
            .iter()
            .copied()
            .filter(|&v| v >= IGPU_VRAM_THRESHOLD)
            .collect();
        let (effective_vram, gpu_count) = if discrete_vram.is_empty() {
            // No discrete GPUs found; use all entries (may be an iGPU-only system)
            (per_gpu_vram_bytes, 1u32)
        } else {
            let count = discrete_vram.len() as u32;
            (discrete_vram, count)
        };

        // Try to get GPU name from rocm-smi --showproductname
        let gpu_name = std::process::Command::new("rocm-smi")
            .arg("--showproductname")
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok()
                } else {
                    None
                }
            })
            .and_then(|text| {
                // Look for "Card Series" or "Card Model" lines.
                // rocm-smi output format: "GPU[0] : Card Series: <GPU name>"
                // The GPU name is after the LAST colon, not the first.
                for line in text.lines() {
                    let lower = line.to_lowercase();
                    if lower.contains("card series") || lower.contains("card model") {
                        // Take the last colon-separated segment — that's the actual GPU name
                        if let Some(name) = line.rsplit(':').next() {
                            let name = name.trim().to_string();
                            if !name.is_empty() {
                                return Some(name);
                            }
                        }
                    }
                }
                None
            });

        let name = gpu_name.unwrap_or_else(|| "AMD GPU".to_string());
        let max_per_gpu_bytes = effective_vram.into_iter().max().unwrap_or(0);
        let vram_gb = if max_per_gpu_bytes > 0 {
            Some(max_per_gpu_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else {
            let est = estimate_vram_from_name(&name);
            if est > 0.0 { Some(est) } else { None }
        };

        Some(GpuInfo {
            name,
            vram_gb,
            backend: GpuBackend::Rocm,
            count: gpu_count,
            unified_memory: false,
        })
    }

    /// Detect AMD GPU via sysfs on Linux (works without ROCm installed).
    /// AMD vendor ID is 0x1002.
    fn detect_amd_gpu_sysfs_info() -> Option<GpuInfo> {
        if !cfg!(target_os = "linux") {
            return None;
        }

        let mut slot_hints: Vec<String> = Vec::new();
        let entries = std::fs::read_dir("/sys/class/drm").ok()?;

        for entry in entries.flatten() {
            let card_path = entry.path();
            let fname = card_path.file_name()?.to_str()?.to_string();
            // Only look at cardN entries, not cardN-DP-1 etc.
            if !fname.starts_with("card") || fname.contains('-') {
                continue;
            }

            let device_path = card_path.join("device");
            let vendor_path = device_path.join("vendor");
            if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                if vendor.trim() != "0x1002" {
                    continue;
                }
            } else {
                continue;
            }

            // Found an AMD GPU. Try to read VRAM.
            let mut vram_gb: Option<f64> = None;
            let vram_path = device_path.join("mem_info_vram_total");
            if let Ok(vram_str) = std::fs::read_to_string(&vram_path)
                && let Ok(vram_bytes) = vram_str.trim().parse::<u64>()
                && vram_bytes > 0
            {
                vram_gb = Some(vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
            }

            if let Ok(uevent) = std::fs::read_to_string(device_path.join("uevent")) {
                for line in uevent.lines() {
                    if let Some(slot) = line.strip_prefix("PCI_SLOT_NAME=") {
                        slot_hints.push(slot.to_string());
                    }
                }
            }

            // Try to get GPU name from lspci
            let gpu_name = Self::get_amd_gpu_name_lspci(&slot_hints);
            let name = gpu_name.unwrap_or_else(|| "AMD GPU".to_string());

            // If we still don't have VRAM, try to estimate from name
            if vram_gb.is_none() {
                let estimated = estimate_vram_from_name(&name);
                if estimated > 0.0 {
                    vram_gb = Some(estimated);
                }
            }

            // AMD GPU without ROCm — Vulkan is the most likely inference backend
            return Some(GpuInfo {
                name,
                vram_gb,
                backend: GpuBackend::Vulkan,
                count: 1,
                unified_memory: false,
            });
        }
        None
    }

    /// Extract AMD GPU name from lspci output.
    fn get_amd_gpu_name_lspci(slot_hints: &[String]) -> Option<String> {
        let text = Self::lspci_output()?;

        // First pass: match exact slot (e.g. "0000:01:00.0"), if available.
        for slot in slot_hints {
            for line in text.lines() {
                let lower = line.to_lowercase();
                if line.starts_with(slot)
                    && (lower.contains("vga") || lower.contains("3d") || lower.contains("display"))
                    && (lower.contains("amd") || lower.contains("ati"))
                    && let Some(model) = Self::extract_model_from_lspci_line(line)
                {
                    return Some(model);
                }
            }
        }

        // Fallback: any AMD/ATI display controller line.
        for line in text.lines() {
            let lower = line.to_lowercase();
            if (lower.contains("vga") || lower.contains("3d"))
                && (lower.contains("amd") || lower.contains("ati"))
                && let Some(model) = Self::extract_model_from_lspci_line(line)
            {
                return Some(model);
            }
        }
        None
    }

    /// Resolve NVIDIA GPU name from lspci, optionally prioritizing specific
    /// PCI slots discovered from sysfs.
    fn get_nvidia_gpu_name_lspci(slot_hints: &[String]) -> Option<String> {
        let text = Self::lspci_output()?;

        // First pass: match exact slot (e.g. "0000:01:00.0"), if available.
        for slot in slot_hints {
            for line in text.lines() {
                let lower = line.to_lowercase();
                if line.starts_with(slot)
                    && (lower.contains("vga") || lower.contains("3d") || lower.contains("display"))
                    && lower.contains("nvidia")
                    && let Some(model) = Self::extract_model_from_lspci_line(line)
                {
                    return Some(model);
                }
            }
        }

        // Fallback: any NVIDIA display controller line.
        for line in text.lines() {
            let lower = line.to_lowercase();
            if (lower.contains("vga") || lower.contains("3d") || lower.contains("display"))
                && lower.contains("nvidia")
                && let Some(model) = Self::extract_model_from_lspci_line(line)
            {
                return Some(model);
            }
        }

        None
    }

    /// Read lspci output, with host fallback for containerized environments.
    fn lspci_output() -> Option<String> {
        let local = std::process::Command::new("lspci")
            .arg("-nnD")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok());

        if local.is_some() {
            return local;
        }

        std::process::Command::new("flatpak-spawn")
            .args(["--host", "lspci", "-nnD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
    }

    /// Extract a likely model name from an lspci line.
    /// Prefers human-readable bracketed tokens (e.g. "[GeForce RTX 2060]").
    fn extract_model_from_lspci_line(line: &str) -> Option<String> {
        let mut best: Option<String> = None;
        let mut rest = line;

        while let Some(start) = rest.find('[') {
            let after = &rest[start + 1..];
            let Some(end) = after.find(']') else { break };
            let token = after[..end].trim();
            let usable = !token.is_empty()
                && !token.contains(':')
                && !token.chars().all(|c| c.is_ascii_digit());

            if usable
                && best
                    .as_ref()
                    .map(|current| token.len() > current.len())
                    .unwrap_or(true)
            {
                best = Some(token.to_string());
            }

            rest = &after[end + 1..];
        }

        if best.is_some() {
            return best;
        }

        // Fallback: text after the first ": " separator.
        line.split_once(": ")
            .map(|(_, right)| right.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Detect GPUs on Windows via WMI (Win32_VideoController).
    /// Returns all discrete GPUs found (AMD, NVIDIA, Intel, etc.).
    /// When both discrete and integrated GPUs are present, the integrated
    /// GPUs are filtered out so the discrete GPU is selected as primary.
    fn detect_gpu_windows_info() -> Vec<GpuInfo> {
        if !cfg!(target_os = "windows") {
            return Vec::new();
        }

        // Use PowerShell to query WMI — more reliable than wmic (deprecated)
        if let Ok(output) = std::process::Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg("Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }")
            .output()
            && output.status.success()
                && let Ok(text) = String::from_utf8(output.stdout) {
                    let gpus = Self::parse_windows_gpu_list(&text);
                    if !gpus.is_empty() {
                        return Self::prefer_discrete_gpus(gpus);
                    }
                }

        // Fallback to wmic for older Windows
        let gpus = Self::detect_gpu_windows_wmic_list();
        Self::prefer_discrete_gpus(gpus)
    }

    /// Fallback Windows GPU detection via wmic (works on older systems).
    fn detect_gpu_windows_wmic_list() -> Vec<GpuInfo> {
        let output = match std::process::Command::new("wmic")
            .arg("path")
            .arg("win32_VideoController")
            .arg("get")
            .arg("Name,AdapterRAM")
            .arg("/format:csv")
            .output()
        {
            Ok(o) if o.status.success() => o,
            _ => return Vec::new(),
        };

        let text = match String::from_utf8(output.stdout) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let mut gpus = Vec::new();
        // CSV format: Node,AdapterRAM,Name
        for line in text.lines().skip(1) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 3 {
                let raw_vram: u64 = parts[1].trim().parse().unwrap_or(0);
                let name = parts[2..].join(",").trim().to_string();
                let lower = name.to_lowercase();
                if lower.contains("microsoft")
                    || lower.contains("basic")
                    || lower.contains("virtual")
                {
                    continue;
                }
                let backend = Self::infer_gpu_backend(&name);
                let vram_gb = Self::resolve_wmi_vram(raw_vram, &name);
                gpus.push(GpuInfo {
                    name,
                    vram_gb,
                    backend,
                    count: 1,
                    unified_memory: false,
                });
            }
        }
        gpus
    }

    /// Parse all GPU entries from PowerShell output (Name|AdapterRAM per line).
    fn parse_windows_gpu_list(text: &str) -> Vec<GpuInfo> {
        let mut gpus = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, '|').collect();
            let name = parts[0].trim().to_string();
            let raw_vram: u64 = parts
                .get(1)
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);

            let lower = name.to_lowercase();
            if lower.contains("microsoft")
                || lower.contains("basic")
                || lower.contains("virtual")
                || lower.is_empty()
            {
                continue;
            }

            let backend = Self::infer_gpu_backend(&name);
            let vram_gb = Self::resolve_wmi_vram(raw_vram, &name);
            gpus.push(GpuInfo {
                name,
                vram_gb,
                backend,
                count: 1,
                unified_memory: false,
            });
        }
        gpus
    }

    /// When both discrete and integrated GPUs are detected on Windows,
    /// drop the integrated GPUs so the discrete GPU becomes primary.
    /// If only integrated GPUs are present, keep them all (iGPU-only systems).
    fn prefer_discrete_gpus(gpus: Vec<GpuInfo>) -> Vec<GpuInfo> {
        let discrete: Vec<GpuInfo> = gpus
            .iter()
            .filter(|g| !Self::is_integrated_gpu_name(&g.name))
            .cloned()
            .collect();

        if discrete.is_empty() {
            // No discrete GPUs found; keep everything (iGPU-only system).
            gpus
        } else {
            discrete
        }
    }

    /// Heuristic: returns true when the GPU name matches known integrated GPU
    /// patterns on Windows (Intel UHD/HD/Iris, AMD Radeon Graphics without a
    /// discrete model number like RX).
    fn is_integrated_gpu_name(name: &str) -> bool {
        let lower = name.to_lowercase();

        // Explicitly tagged as integrated (e.g. from APU detection path)
        if lower.contains("(integrated)") {
            return true;
        }

        // Intel integrated: UHD, HD Graphics, Iris (but NOT Intel Arc discrete)
        if lower.contains("intel") {
            return lower.contains("uhd")
                || lower.contains("hd graphics")
                || (lower.contains("iris") && !lower.contains("arc"));
        }

        // AMD integrated: "Radeon Graphics" or "Radeon(TM) Graphics" without
        // a discrete series identifier (RX, PRO, Vega 56/64, VII, W-series).
        if lower.contains("radeon") && lower.contains("graphics") {
            let has_discrete_tag = lower.contains("rx ")
                || lower.contains("pro ")
                || lower.contains("vega")
                || lower.contains(" vii")
                || lower.contains(" w");
            return !has_discrete_tag;
        }

        false
    }

    /// WMI AdapterRAM is a 32-bit field, capped at ~4 GB.
    /// If reported value is suspiciously low, estimate from GPU name.
    fn resolve_wmi_vram(raw_bytes: u64, name: &str) -> Option<f64> {
        let mut vram_gb = raw_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if vram_gb < 0.1 || (vram_gb <= 4.1 && estimate_vram_from_name(name) > 4.1) {
            let estimated = estimate_vram_from_name(name);
            if estimated > 0.0 {
                vram_gb = estimated;
            }
        }
        if vram_gb > 0.0 { Some(vram_gb) } else { None }
    }

    /// Infer the most likely inference backend from a GPU name string.
    fn infer_gpu_backend(name: &str) -> GpuBackend {
        let lower = name.to_lowercase();
        if lower.contains("nvidia")
            || lower.contains("geforce")
            || lower.contains("quadro")
            || lower.contains("tesla")
            || lower.contains("rtx")
        {
            GpuBackend::Cuda
        } else if lower.contains("amd") || lower.contains("radeon") || lower.contains("ati") {
            // On Windows, Vulkan is the primary inference path for AMD GPUs
            // (ROCm support on Windows is limited)
            GpuBackend::Vulkan
        } else if lower.contains("intel") || lower.contains("arc") {
            GpuBackend::Sycl
        } else {
            GpuBackend::Vulkan
        }
    }

    /// Detect Intel Arc / Intel integrated GPU via sysfs or lspci.
    /// Intel Arc GPUs (A370M, A770, etc.) have dedicated VRAM exposed via
    /// the DRM subsystem at /sys/class/drm/card*/device/. Even integrated
    /// Intel GPUs that share system RAM are useful for inference via SYCL/oneAPI.
    fn detect_intel_gpu() -> Option<f64> {
        // Try sysfs first: works for Intel discrete (Arc) GPUs on Linux.
        // Walk /sys/class/drm/card*/device/ looking for Intel vendor ID (0x8086).
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let card_path = entry.path();
                let device_path = card_path.join("device");

                // Check vendor ID matches Intel (0x8086)
                let vendor_path = device_path.join("vendor");
                if let Ok(vendor) = std::fs::read_to_string(&vendor_path)
                    && vendor.trim() != "0x8086"
                {
                    continue;
                }

                // Look for total VRAM via DRM memory info
                // Intel discrete GPUs expose this under drm/card*/device/mem_info_vram_total
                let vram_path = card_path.join("device/mem_info_vram_total");
                if let Ok(vram_str) = std::fs::read_to_string(&vram_path)
                    && let Ok(vram_bytes) = vram_str.trim().parse::<u64>()
                    && vram_bytes > 0
                {
                    let vram_gb = vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    return Some(vram_gb);
                }

                // For integrated Intel GPUs, check if it's an Arc-class device
                // by looking for "Arc" in the device name via lspci
                if let Some(text) = Self::lspci_output() {
                    for line in text.lines() {
                        let lower = line.to_lowercase();
                        if lower.contains("intel") && lower.contains("arc") {
                            // Intel Arc integrated (e.g. Arc Graphics in Meteor Lake)
                            // These share system RAM; report None for VRAM and
                            // let the caller know a GPU exists.
                            return Some(0.0);
                        }
                    }
                }
            }
        }

        // Fallback: check lspci directly for Intel Arc devices
        // (covers cases where sysfs isn't available or card dirs don't exist)
        if let Some(text) = Self::lspci_output() {
            for line in text.lines() {
                let lower = line.to_lowercase();
                if lower.contains("intel") && lower.contains("arc") {
                    return Some(0.0);
                }
            }
        }

        None
    }

    /// Detect Apple Silicon GPU via system_profiler.
    /// Returns total system RAM as VRAM since memory is unified.
    /// The unified memory pool capacity is the total RAM -- it doesn't
    /// fluctuate with current usage the way available RAM does.
    fn detect_apple_gpu(total_ram_gb: f64) -> Option<f64> {
        // system_profiler only exists on macOS
        let output = std::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let text = String::from_utf8(output.stdout).ok()?;

        // Apple Silicon GPUs show "Apple M1/M2/M3/M4" in the chipset line.
        // Discrete AMD/Intel GPUs on older Macs won't match.
        let is_apple_gpu = text.lines().any(|line| {
            let lower = line.to_lowercase();
            lower.contains("apple m") || lower.contains("apple gpu")
        });

        if is_apple_gpu {
            // Unified memory: GPU and CPU share the same RAM pool.
            // Report total RAM as the VRAM capacity.
            Some(total_ram_gb)
        } else {
            None
        }
    }

    fn has_command(command: &str) -> bool {
        let Some(path_var) = std::env::var_os("PATH") else {
            return false;
        };

        for path in std::env::split_paths(&path_var) {
            let candidate = path.join(command);
            if candidate.is_file() {
                return true;
            }

            #[cfg(target_os = "windows")]
            for ext in [".exe", ".cmd", ".bat", ".com"] {
                let candidate = path.join(format!("{command}{ext}"));
                if candidate.is_file() {
                    return true;
                }
            }
        }

        false
    }

    /// Detect GPUs via Vulkan. This is especially useful on Android/Termux,
    /// where vendor-specific Linux utilities may be unavailable.
    fn detect_vulkan_gpu_info() -> Vec<GpuInfo> {
        if !Self::has_command("vulkaninfo") {
            return Vec::new();
        }

        let output = match std::process::Command::new("vulkaninfo")
            .arg("--summary")
            .output()
        {
            Ok(o) if o.status.success() => o,
            _ => match std::process::Command::new("vulkaninfo").output() {
                Ok(o) if o.status.success() => o,
                _ => return Vec::new(),
            },
        };

        let text = String::from_utf8_lossy(&output.stdout);
        let mut grouped: BTreeMap<String, u32> = BTreeMap::new();

        for name in Self::parse_vulkan_device_names(&text) {
            if Self::is_software_vulkan_device(&name) {
                continue;
            }
            *grouped.entry(name).or_insert(0) += 1;
        }

        grouped
            .into_iter()
            .map(|(name, count)| GpuInfo {
                backend: GpuBackend::Vulkan,
                count,
                name,
                unified_memory: false,
                vram_gb: None,
            })
            .collect()
    }

    fn is_same_gpu_name(existing_name: &str, candidate_name: &str) -> bool {
        Self::normalize_gpu_name_for_dedupe(existing_name)
            == Self::normalize_gpu_name_for_dedupe(candidate_name)
    }

    fn normalize_gpu_name_for_dedupe(name: &str) -> String {
        let mut normalized = String::with_capacity(name.len());
        let mut last_was_separator = true;

        for ch in name.chars().flat_map(char::to_lowercase) {
            if ch.is_alphanumeric() {
                normalized.push(ch);
                last_was_separator = false;
            } else if !last_was_separator {
                normalized.push(' ');
                last_was_separator = true;
            }
        }

        normalized.trim().to_string()
    }

    fn parse_vulkan_device_names(text: &str) -> Vec<String> {
        let mut names = Vec::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some((key, value)) = trimmed.split_once('=')
                && key.trim().eq_ignore_ascii_case("deviceName")
            {
                let name = value.trim();
                if !name.is_empty() {
                    names.push(name.to_string());
                }
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("GPU id")
                && let Some(start) = rest.find('(')
                && let Some(end) = rest.rfind(')')
                && end > start + 1
            {
                let name = rest[start + 1..end].trim();
                if !name.is_empty() {
                    names.push(name.to_string());
                }
            }
        }

        names
    }

    fn is_software_vulkan_device(name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.contains("llvmpipe")
            || lower.contains("lavapipe")
            || lower.contains("swiftshader")
            || lower.contains("software rasterizer")
    }

    /// Detect Ascend NPUs via npu-smi. Returns a vector of NPU info.
    fn detect_ascend_npus() -> Vec<GpuInfo> {
        // 1. Get the list of IDs
        let list_output = match std::process::Command::new("npu-smi")
            .args(["info", "-l"])
            .output()
        {
            Ok(o) if o.status.success() => o,
            _ => return Vec::new(),
        };

        let list_stdout = String::from_utf8_lossy(&list_output.stdout);

        // Extracting IDs: ["0", "1", "2"...]
        let ids: Vec<String> = list_stdout
            .lines()
            .filter(|line| line.contains("NPU ID"))
            .filter_map(|line| line.split(':').next_back())
            .map(|s| s.trim().to_string())
            .collect();

        if ids.is_empty() {
            return Vec::new();
        }

        let mut npu_infos: Vec<GpuInfo> = Vec::new();
        let npu_name = "Ascend NPU";

        // 2. Loop through NPUs
        for id in &ids {
            let mem_output = std::process::Command::new("npu-smi")
                .args(["info", "-t", "memory", "-i", id])
                .output();

            if let Ok(o) = mem_output {
                let s = String::from_utf8_lossy(&o.stdout);

                // Parse HBM Capacity (e.g., from "HBM Capacity(MB) : 65536")
                let mem = s
                    .lines()
                    .find(|l| l.contains("HBM Capacity"))
                    .and_then(|l| l.split(':').next_back())
                    .and_then(|v| v.split_whitespace().next())
                    .and_then(|num| num.parse::<u64>().ok())
                    .unwrap_or(0);

                let npu_info = GpuInfo {
                    name: npu_name.to_string(),
                    vram_gb: Some((mem as f64) / 1024.0),
                    backend: GpuBackend::Ascend,
                    count: 1,
                    unified_memory: false,
                };
                npu_infos.push(npu_info);
            }
        }

        npu_infos
    }

    /// Fallback for available RAM when sysinfo returns 0.
    /// Tries total - used first, then macOS vm_stat parsing.
    fn available_ram_fallback(sys: &System, total_bytes: u64, total_gb: f64) -> f64 {
        // Try total - used from sysinfo (may also use vm_statistics64 internally)
        let used = sys.used_memory();
        if used > 0 && used < total_bytes {
            return (total_bytes - used) as f64 / (1024.0 * 1024.0 * 1024.0);
        }

        // macOS fallback: parse vm_stat output
        if let Some(avail) = Self::available_ram_from_vm_stat() {
            return avail;
        }

        // Last resort: assume 80% of total is available (conservative)
        total_gb * 0.8
    }

    /// Parse macOS `vm_stat` to compute available memory.
    /// Available ≈ (free + inactive + purgeable) * page_size
    fn available_ram_from_vm_stat() -> Option<f64> {
        let output = std::process::Command::new("vm_stat").output().ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;

        // First line: "Mach Virtual Memory Statistics: (page size of NNNNN bytes)"
        let page_size: u64 = text
            .lines()
            .next()
            .and_then(|line| {
                line.split("page size of ")
                    .nth(1)?
                    .split(' ')
                    .next()?
                    .parse()
                    .ok()
            })
            .unwrap_or(16384); // Apple Silicon default is 16 KB pages

        let mut free: u64 = 0;
        let mut inactive: u64 = 0;
        let mut purgeable: u64 = 0;

        for line in text.lines() {
            if let Some(val) = Self::parse_vm_stat_line(line, "Pages free") {
                free = val;
            } else if let Some(val) = Self::parse_vm_stat_line(line, "Pages inactive") {
                inactive = val;
            } else if let Some(val) = Self::parse_vm_stat_line(line, "Pages purgeable") {
                purgeable = val;
            }
        }

        let available_bytes = (free + inactive + purgeable) * page_size;
        if available_bytes > 0 {
            Some(available_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else {
            None
        }
    }

    /// Parse a single vm_stat line like "Pages free:    123456."
    fn parse_vm_stat_line(line: &str, key: &str) -> Option<u64> {
        if !line.starts_with(key) {
            return None;
        }
        line.split(':')
            .nth(1)?
            .trim()
            .trim_end_matches('.')
            .parse()
            .ok()
    }

    fn detect_cpu_name(sys: &System) -> String {
        if let Some(cpu_name) = sys
            .cpus()
            .iter()
            .map(|cpu| cpu.brand().trim())
            .find(|brand| !brand.is_empty() && !brand.eq_ignore_ascii_case("unknown"))
        {
            return cpu_name.to_string();
        }

        if let Some(cpu_name) = Self::read_cpu_name_from_proc_cpuinfo() {
            return cpu_name;
        }

        if let Some(cpu_name) = Self::read_android_soc_name() {
            return cpu_name;
        }

        "Unknown CPU".to_string()
    }

    fn read_cpu_name_from_proc_cpuinfo() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
            return Self::parse_cpu_name_from_cpuinfo(&text);
        }

        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    fn parse_cpu_name_from_cpuinfo(text: &str) -> Option<String> {
        for key in ["model name", "hardware", "processor", "cpu model", "model"] {
            for line in text.lines() {
                let Some((lhs, rhs)) = line.split_once(':') else {
                    continue;
                };
                if lhs.trim().eq_ignore_ascii_case(key) {
                    let candidate = rhs.trim();
                    if !candidate.is_empty() && !candidate.eq_ignore_ascii_case("unknown") {
                        return Some(candidate.to_string());
                    }
                }
            }
        }

        None
    }

    fn read_android_soc_name() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            let output = std::process::Command::new("getprop")
                .arg("ro.soc.model")
                .output()
                .ok()?;
            if !output.status.success() {
                return None;
            }

            let model = String::from_utf8(output.stdout).ok()?;
            let model = model.trim();
            if model.is_empty() {
                return None;
            }

            return Some(model.to_string());
        }

        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// Override the primary GPU's VRAM with a user-specified value (in GB).
    /// This is used by the `--memory` CLI flag when GPU autodetection fails.
    /// If no GPU was detected, this creates a synthetic GPU entry.
    pub fn with_gpu_memory_override(mut self, vram_gb: f64) -> Self {
        if self.gpus.is_empty() {
            // No GPU was detected; create a synthetic one.
            let backend = if cfg!(target_arch = "aarch64")
                || self.cpu_name.to_lowercase().contains("apple")
            {
                GpuBackend::Metal
            } else {
                GpuBackend::Cuda
            };
            self.gpus.push(GpuInfo {
                name: "User-specified GPU".to_string(),
                vram_gb: Some(vram_gb),
                backend,
                count: 1,
                unified_memory: false,
            });
            self.has_gpu = true;
            self.gpu_vram_gb = Some(vram_gb);
            self.total_gpu_vram_gb = Some(vram_gb);
            self.gpu_name = Some("User-specified GPU".to_string());
            self.gpu_count = 1;
            self.backend = backend;
        } else {
            // Override the primary (first) GPU's VRAM.
            self.gpus[0].vram_gb = Some(vram_gb);
            self.gpu_vram_gb = Some(vram_gb);
            // Update total VRAM: per-card VRAM * count.
            let count = self.gpus[0].count;
            self.total_gpu_vram_gb = Some(vram_gb * count as f64);
            self.has_gpu = true;
        }
        self
    }

    /// Override total and available system RAM with a user-specified value (in GB).
    /// Sets available RAM to 90% of the override to model typical system usage.
    /// On unified-memory systems (Apple Silicon), this also updates GPU VRAM
    /// to stay consistent — use `--memory` after `--ram` to override VRAM separately.
    pub fn with_ram_override(mut self, ram_gb: f64) -> Self {
        self.total_ram_gb = ram_gb;
        self.available_ram_gb = ram_gb * 0.9;
        if self.unified_memory {
            self.gpu_vram_gb = Some(ram_gb);
            self.total_gpu_vram_gb = Some(ram_gb);
            for gpu in &mut self.gpus {
                if gpu.unified_memory {
                    gpu.vram_gb = Some(ram_gb);
                }
            }
        }
        self
    }

    /// Override the detected CPU core count with a user-specified value.
    pub fn with_cpu_core_override(mut self, cores: usize) -> Self {
        self.total_cpu_cores = cores;
        self
    }

    pub fn display(&self) {
        println!("\n=== System Specifications ===");
        println!("CPU: {} ({} cores)", self.cpu_name, self.total_cpu_cores);
        println!("Total RAM: {:.2} GB", self.total_ram_gb);
        println!("Available RAM: {:.2} GB", self.available_ram_gb);
        println!("Backend: {}", self.backend.label());

        if self.gpus.is_empty() {
            println!("GPU: Not detected");
        } else {
            for (i, gpu) in self.gpus.iter().enumerate() {
                let prefix = if self.gpus.len() > 1 {
                    format!("GPU {}: ", i + 1)
                } else {
                    "GPU: ".to_string()
                };
                if gpu.unified_memory {
                    println!(
                        "{}{} (unified memory, {:.2} GB shared, {})",
                        prefix,
                        gpu.name,
                        gpu.vram_gb.unwrap_or(0.0),
                        gpu.backend.label(),
                    );
                } else {
                    match gpu.vram_gb {
                        Some(vram) if vram > 0.0 => {
                            if gpu.count > 1 {
                                let total_vram = vram * gpu.count as f64;
                                println!(
                                    "{}{} x{} ({:.2} GB VRAM each = {:.0} GB total, {})",
                                    prefix,
                                    gpu.name,
                                    gpu.count,
                                    vram,
                                    total_vram,
                                    gpu.backend.label()
                                );
                            } else {
                                println!(
                                    "{}{} ({:.2} GB VRAM, {})",
                                    prefix,
                                    gpu.name,
                                    vram,
                                    gpu.backend.label()
                                );
                            }
                        }
                        Some(_) => println!(
                            "{}{} (shared system memory, {})",
                            prefix,
                            gpu.name,
                            gpu.backend.label()
                        ),
                        None => println!(
                            "{}{} (VRAM unknown, {})",
                            prefix,
                            gpu.name,
                            gpu.backend.label()
                        ),
                    }
                }
            }
        }
        println!();
    }
}

/// Parse a human-readable memory size string into gigabytes.
/// Accepts formats: "32G", "32g", "32GB", "32gb", "32000M", "32000m", "32000MB", etc.
/// Returns `None` if the input is malformed.
pub fn parse_memory_size(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Split into numeric part and suffix
    let num_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());
    let (num_str, suffix) = s.split_at(num_end);
    let value: f64 = num_str.parse().ok()?;
    if value < 0.0 {
        return None;
    }

    let suffix = suffix.trim().to_lowercase();
    match suffix.as_str() {
        "g" | "gb" | "gib" | "" => Some(value),     // already in GB
        "m" | "mb" | "mib" => Some(value / 1024.0), // MB → GB
        "t" | "tb" | "tib" => Some(value * 1024.0), // TB → GB
        _ => None,
    }
}

pub fn is_running_in_wsl() -> bool {
    static IS_WSL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *IS_WSL.get_or_init(detect_running_in_wsl)
}

fn detect_running_in_wsl() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }

    if std::env::var_os("WSL_INTEROP").is_some() || std::env::var_os("WSL_DISTRO_NAME").is_some() {
        return true;
    }

    ["/proc/sys/kernel/osrelease", "/proc/version"]
        .iter()
        .any(|path| {
            std::fs::read_to_string(path)
                .map(|text| text.to_ascii_lowercase().contains("microsoft"))
                .unwrap_or(false)
        })
}

/// Check if the CPU name indicates an AMD APU with unified memory architecture.
/// These APUs share the full system RAM between CPU and GPU (like Apple Silicon).
/// Currently covers:
///  - Ryzen AI MAX / MAX+ (Strix Halo): up to 128 GB unified.
///  - Ryzen AI 9 / 7 / 5 (Strix Point, Krackan Point): configurable shared
///    memory, users can allocate most of system RAM to GPU via BIOS.
/// All Ryzen AI APUs have integrated Radeon GPUs that share system memory.
fn is_amd_unified_memory_apu(cpu_name: &str) -> bool {
    let lower = cpu_name.to_lowercase();
    // Only "Ryzen AI MAX" / "Ryzen AI MAX+" APUs have a large unified memory
    // pool shared between CPU and GPU (similar to Apple Silicon).
    // Regular Ryzen AI chips (e.g. HX 370, HX 365) have a standard small iGPU
    // and should NOT be treated as unified-memory systems.
    // Examples that match:
    //   "AMD Ryzen AI MAX+ 395 w/ Radeon 8060S"
    //   "AMD Ryzen AI MAX 390"
    if lower.contains("ryzen ai max") {
        return true;
    }
    false
}

/// Query total installed physical RAM on Windows by summing DIMM capacities
/// from WMI `Win32_PhysicalMemory`. Unlike `sysinfo::System::total_memory()`
/// or `Win32_ComputerSystem.TotalPhysicalMemory`, this reads directly from
/// SMBIOS and is unaffected by BIOS-level GPU UMA carveouts.
///
/// On AMD Ryzen AI MAX / MAX+ systems where users configure e.g. 96 GB as GPU
/// UMA in BIOS, the OS only sees the remaining ~32 GB as system RAM, causing
/// `sysinfo` to report 32 GB. `Win32_PhysicalMemory.Capacity` correctly sums
/// all installed DIMMs (e.g. 128 GB) regardless of that carveout.
///
/// Returns `None` when not on Windows, PowerShell is unavailable, or the
/// query fails; callers fall back to the sysinfo value.
fn detect_windows_physical_total_ram_gb() -> Option<f64> {
    if !cfg!(target_os = "windows") {
        return None;
    }
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let bytes: u64 = text.trim().parse().ok()?;
    if bytes == 0 {
        return None;
    }
    Some(bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// Read total system RAM from /proc/meminfo (Linux only).
/// Used as the unified memory pool on NVIDIA Tegra / Grace Blackwell platforms
/// where nvidia-smi cannot report dedicated VRAM.
fn read_proc_meminfo_total_gb() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb as f64 / (1024.0 * 1024.0));
        }
    }
    None
}

/// Estimate GPU memory bandwidth in GB/s from the GPU model name.
///
/// Token generation in LLM inference is memory-bandwidth-bound (each token
/// requires reading the full model weights once). Using per-GPU bandwidth
/// produces significantly more accurate tok/s estimates than a single
/// constant for all CUDA/ROCm/Metal devices.
///
/// References:
///  - kipply, "Transformer Inference Arithmetic" (2022)
///  - ggerganov, llama.cpp Apple Silicon benchmarks (Discussion #4167)
///  - Google, "Efficiently Scaling Transformer Inference" (arXiv:2211.05102)
///  - ggerganov, llama.cpp NVIDIA T4 benchmarks (Discussion #4225)
///
/// Returns `None` when the GPU is not recognized; callers should fall back
/// to the existing fixed-constant approach.
pub fn gpu_memory_bandwidth_gbps(name: &str) -> Option<f64> {
    let lower = name.to_lowercase();

    // ── NVIDIA Consumer (GeForce) ──────────────────────────────────
    // RTX 50 series (Blackwell)
    if lower.contains("5090") {
        return Some(1792.0);
    }
    if lower.contains("5080") {
        return Some(960.0);
    }
    if lower.contains("5070 ti") {
        return Some(896.0);
    }
    if lower.contains("5070") {
        return Some(672.0);
    }
    if lower.contains("5060 ti") {
        return Some(448.0);
    }
    if lower.contains("5060") {
        return Some(256.0);
    }

    // RTX 40 series (Ada Lovelace)
    if lower.contains("4090") {
        return Some(1008.0);
    }
    if lower.contains("4080 super") {
        return Some(736.0);
    }
    if lower.contains("4080") {
        return Some(717.0);
    }
    if lower.contains("4070 ti super") {
        return Some(672.0);
    }
    if lower.contains("4070 ti") {
        return Some(504.0);
    }
    if lower.contains("4070 super") {
        return Some(504.0);
    }
    if lower.contains("4070") {
        return Some(504.0);
    }
    if lower.contains("4060 ti") {
        return Some(288.0);
    }
    if lower.contains("4060") {
        return Some(272.0);
    }

    // RTX 30 series (Ampere)
    if lower.contains("3090 ti") {
        return Some(1008.0);
    }
    if lower.contains("3090") {
        return Some(936.0);
    }
    if lower.contains("3080 ti") {
        return Some(912.0);
    }
    if lower.contains("3080") {
        return Some(760.0);
    }
    if lower.contains("3070 ti") {
        return Some(608.0);
    }
    if lower.contains("3070") {
        return Some(448.0);
    }
    if lower.contains("3060 ti") {
        return Some(448.0);
    }
    if lower.contains("3060") {
        return Some(360.0);
    }

    // RTX 20 series (Turing)
    if lower.contains("2080 ti") {
        return Some(616.0);
    }
    if lower.contains("2080 super") {
        return Some(496.0);
    }
    if lower.contains("2080") {
        return Some(448.0);
    }
    if lower.contains("2070 super") {
        return Some(448.0);
    }
    if lower.contains("2070") {
        return Some(448.0);
    }
    if lower.contains("2060 super") {
        return Some(448.0);
    }
    if lower.contains("2060") {
        return Some(336.0);
    }

    // GTX 16 series (Turing, no RT cores)
    if lower.contains("1660 ti") {
        return Some(288.0);
    }
    if lower.contains("1660 super") {
        return Some(336.0);
    }
    if lower.contains("1660") {
        return Some(192.0);
    }
    if lower.contains("1650 super") {
        return Some(192.0);
    }
    if lower.contains("1650") {
        return Some(128.0);
    }

    // ── NVIDIA Data Center / Professional ──────────────────────────
    if lower.contains("h100 sxm") {
        return Some(3350.0);
    }
    if lower.contains("h100") {
        return Some(2039.0);
    } // PCIe
    if lower.contains("h200") {
        return Some(4800.0);
    }
    if lower.contains("a100 sxm") {
        return Some(2039.0);
    }
    if lower.contains("a100") {
        return Some(1555.0);
    } // PCIe 40GB
    if lower.contains("l40s") {
        return Some(864.0);
    }
    if lower.contains("l40") {
        return Some(864.0);
    }
    if lower.contains("l4") {
        return Some(300.0);
    }
    if lower.contains("a10g") {
        return Some(600.0);
    }
    if lower.contains("a10") {
        return Some(600.0);
    }
    if lower.contains("t4") {
        return Some(320.0);
    }
    if lower.contains("v100 sxm") {
        return Some(900.0);
    }
    if lower.contains("v100") {
        return Some(897.0);
    }
    if lower.contains("a6000") {
        return Some(768.0);
    }
    if lower.contains("a5000") {
        return Some(768.0);
    }
    if lower.contains("a4000") {
        return Some(448.0);
    }

    // ── AMD Discrete (RDNA) ────────────────────────────────────────
    // RX 9000 series (RDNA 4)
    if lower.contains("9070 xt") {
        return Some(624.0);
    }
    if lower.contains("9070") {
        return Some(488.0);
    }

    // RX 7000 series (RDNA 3)
    if lower.contains("7900 xtx") {
        return Some(960.0);
    }
    if lower.contains("7900 xt") {
        return Some(800.0);
    }
    if lower.contains("7900 gre") {
        return Some(576.0);
    }
    if lower.contains("7800 xt") {
        return Some(624.0);
    }
    if lower.contains("7700 xt") {
        return Some(432.0);
    }
    if lower.contains("7600") {
        return Some(288.0);
    }

    // RX 6000 series (RDNA 2)
    if lower.contains("6950 xt") {
        return Some(576.0);
    }
    if lower.contains("6900 xt") {
        return Some(512.0);
    }
    if lower.contains("6800 xt") {
        return Some(512.0);
    }
    if lower.contains("6800") {
        return Some(512.0);
    }
    if lower.contains("6700 xt") {
        return Some(384.0);
    }
    if lower.contains("6600 xt") {
        return Some(256.0);
    }
    if lower.contains("6600") {
        return Some(224.0);
    }

    // AMD data center (CDNA)
    if lower.contains("mi300x") {
        return Some(5300.0);
    }
    if lower.contains("mi300") {
        return Some(5300.0);
    }
    if lower.contains("mi250x") {
        return Some(3277.0);
    }
    if lower.contains("mi250") {
        return Some(3277.0);
    }
    if lower.contains("mi210") {
        return Some(1638.0);
    }
    if lower.contains("mi100") {
        return Some(1229.0);
    }

    // ── Apple Silicon (unified memory bandwidth) ───────────────────
    if lower.contains("m4 ultra") {
        return Some(819.0);
    }
    if lower.contains("m4 max") {
        return Some(546.0);
    }
    if lower.contains("m4 pro") {
        return Some(273.0);
    }
    if lower.contains("m4") {
        return Some(120.0);
    }
    if lower.contains("m3 ultra") {
        return Some(800.0);
    }
    if lower.contains("m3 max") {
        return Some(400.0);
    }
    if lower.contains("m3 pro") {
        return Some(150.0);
    }
    if lower.contains("m3") {
        return Some(100.0);
    }
    if lower.contains("m2 ultra") {
        return Some(800.0);
    }
    if lower.contains("m2 max") {
        return Some(400.0);
    }
    if lower.contains("m2 pro") {
        return Some(200.0);
    }
    if lower.contains("m2") {
        return Some(100.0);
    }
    if lower.contains("m1 ultra") {
        return Some(800.0);
    }
    if lower.contains("m1 max") {
        return Some(400.0);
    }
    if lower.contains("m1 pro") {
        return Some(200.0);
    }
    if lower.contains("m1") {
        return Some(68.0);
    }

    None
}

/// Returns the NVIDIA compute capability (major, minor) for a known GPU name.
/// Used to determine compatibility with quantization formats that require
/// specific hardware features (e.g. AWQ requires Turing+ / cc >= 7.5).
///
/// Returns `None` for non-NVIDIA GPUs or unrecognized models.
pub fn gpu_compute_capability(name: &str) -> Option<(u8, u8)> {
    let lower = name.to_lowercase();

    // ── Blackwell (RTX 50xx, B100/B200) ──────────────────────────
    if lower.contains("5090")
        || lower.contains("5080")
        || lower.contains("5070")
        || lower.contains("5060")
        || lower.contains("b200")
        || lower.contains("b100")
        || lower.contains("gb200")
        || lower.contains("gb100")
    {
        return Some((10, 0));
    }

    // ── Hopper (H100, H200) ─────────────────────────────────────
    if lower.contains("h100") || lower.contains("h200") {
        return Some((9, 0));
    }

    // ── Ada Lovelace (RTX 40xx, L4, L40/L40S) ──────────────────
    if lower.contains("4090")
        || lower.contains("4080")
        || lower.contains("4070")
        || lower.contains("4060")
        || lower.contains("l40")
        || lower.contains("l4")
    {
        return Some((8, 9));
    }

    // ── Ampere (RTX 30xx consumer = 8.6, A100/A10/A6000 = 8.0) ─
    if lower.contains("a100") {
        return Some((8, 0));
    }
    if lower.contains("3090")
        || lower.contains("3080")
        || lower.contains("3070")
        || lower.contains("3060")
        || lower.contains("a10")
        || lower.contains("a6000")
        || lower.contains("a5000")
        || lower.contains("a4000")
        || lower.contains("a2000")
        || lower.contains("a16")
    {
        return Some((8, 6));
    }

    // ── Turing (RTX 20xx, GTX 16xx, T4) ─────────────────────────
    if lower.contains("2080")
        || lower.contains("2070")
        || lower.contains("2060")
        || lower.contains("1660")
        || lower.contains("1650")
        || lower.contains("t4")
    {
        return Some((7, 5));
    }

    // ── Volta (V100, Titan V) ───────────────────────────────────
    if lower.contains("v100") || lower.contains("titan v") {
        return Some((7, 0));
    }

    // ── Pascal (P100, GTX 10xx, Titan X Pascal) ─────────────────
    if lower.contains("p100")
        || lower.contains("1080")
        || lower.contains("1070")
        || lower.contains("1060")
        || lower.contains("1050")
        || lower.contains("p40")
        || lower.contains("p4")
    {
        return Some((6, 1));
    }

    None
}

/// Minimum NVIDIA compute capability required by a quantization format
/// when running under vLLM. Based on vLLM's documented hardware support:
/// <https://docs.vllm.ai/en/latest/features/quantization/#supported-hardware>
///
/// Returns `None` for quantization formats that have no known CC restriction
/// (e.g. GGUF quants which run through llama.cpp, not vLLM).
pub fn quant_min_compute_capability(quantization: &str) -> Option<(u8, u8)> {
    match quantization {
        // AWQ requires Turing+ (int4 tensor-core kernels)
        "AWQ-4bit" | "AWQ-8bit" => Some((7, 5)),
        // GPTQ Marlin kernels require Turing+
        "GPTQ-Int4" | "GPTQ-Int8" => Some((7, 5)),
        _ => None,
    }
}

/// Check if a GPU name (including PCI device IDs from lspci) indicates an
/// NVIDIA unified memory SoC (Grace Blackwell / DGX Spark / GB-series).
/// Inside Docker, nvidia-smi may report the raw PCI device ID instead of the
/// friendly model name, e.g. "NVIDIA Corporation Device [10de:2e12] (rev a1)"
/// instead of "NVIDIA GB10".
fn is_nvidia_unified_memory_gpu(name: &str) -> bool {
    let lower = name.to_lowercase();
    // Friendly model names
    if lower.contains("gb10") || lower.contains("gb20") {
        return true;
    }
    // PCI device IDs (hex) — these are the known GB-series SoCs.
    // 10de:2e12 = GB10 (DGX Spark / Project DIGITS)
    if lower.contains("2e12") {
        return true;
    }
    false
}

/// Fallback VRAM estimation from GPU model name.
/// Used when nvidia-smi or other tools report 0 VRAM.
fn estimate_vram_from_name(name: &str) -> f64 {
    let lower = name.to_lowercase();
    // NVIDIA RTX 50 series
    if lower.contains("5090") {
        return 32.0;
    }
    if lower.contains("5080") {
        return 16.0;
    }
    if lower.contains("5070 ti") {
        return 16.0;
    }
    if lower.contains("5070") {
        return 12.0;
    }
    if lower.contains("5060 ti") {
        return 16.0;
    }
    if lower.contains("5060") {
        return 8.0;
    }
    // NVIDIA RTX 40 series
    if lower.contains("4090") {
        return 24.0;
    }
    if lower.contains("4080") {
        return 16.0;
    }
    if lower.contains("4070 ti") {
        return 12.0;
    }
    if lower.contains("4070") {
        return 12.0;
    }
    if lower.contains("4060 ti") {
        return 16.0;
    }
    if lower.contains("4060") {
        return 8.0;
    }
    // NVIDIA RTX 30 series
    if lower.contains("3090") {
        return 24.0;
    }
    if lower.contains("3080 ti") {
        return 12.0;
    }
    if lower.contains("3080") {
        return 10.0;
    }
    if lower.contains("3070") {
        return 8.0;
    }
    if lower.contains("3060 ti") {
        return 8.0;
    }
    if lower.contains("3060") {
        return 12.0;
    }
    // Data center / professional
    if lower.contains("h100") {
        return 80.0;
    }
    if lower.contains("a100") {
        return 80.0;
    }
    if lower.contains("l40") {
        return 48.0;
    }
    // NVIDIA RTX professional (Ampere) — must be checked before the broad "a10" match
    if lower.contains("a6000") {
        return 48.0;
    }
    if lower.contains("a5500") {
        return 24.0;
    }
    if lower.contains("a5000") {
        return 24.0;
    }
    if lower.contains("a4500") {
        return 20.0;
    }
    if lower.contains("a4000") {
        return 16.0;
    }
    if lower.contains("a2000") {
        return 12.0;
    }
    if lower.contains("a10") {
        return 24.0;
    }
    if lower.contains("t4") {
        return 16.0;
    }
    // NVIDIA Grace / DGX Spark unified memory SoCs.
    // Also match PCI device ID 2e12 (GB10) for Docker/container environments
    // where lspci shows "Device [10de:2e12]" instead of the friendly name.
    if lower.contains("gb10") || lower.contains("2e12") {
        return 128.0;
    }
    if lower.contains("gb20") {
        return 128.0;
    }
    // AMD RX 9000 series (RDNA 4)
    if lower.contains("9070 xt") {
        return 16.0;
    }
    if lower.contains("9070") {
        return 12.0;
    }
    if lower.contains("9060 xt") {
        return 16.0;
    }
    if lower.contains("9060") {
        return 8.0;
    }
    // AMD RX 7000 series
    if lower.contains("7900 xtx") {
        return 24.0;
    }
    if lower.contains("7900") {
        return 20.0;
    }
    if lower.contains("7800") {
        return 16.0;
    }
    if lower.contains("7700") {
        return 12.0;
    }
    if lower.contains("7600") {
        return 8.0;
    }
    // AMD RX 6000 series
    if lower.contains("6950") {
        return 16.0;
    }
    if lower.contains("6900") {
        return 16.0;
    }
    if lower.contains("6800") {
        return 16.0;
    }
    if lower.contains("6750") {
        return 12.0;
    }
    if lower.contains("6700") {
        return 12.0;
    }
    if lower.contains("6650") {
        return 8.0;
    }
    if lower.contains("6600") {
        return 8.0;
    }
    if lower.contains("6500") {
        return 4.0;
    }
    // AMD RX 5000 series
    if lower.contains("5700 xt") {
        return 8.0;
    }
    if lower.contains("5700") {
        return 8.0;
    }
    if lower.contains("5600") {
        return 6.0;
    }
    if lower.contains("5500") {
        return 4.0;
    }
    // AMD Radeon 8000 series (Ryzen AI MAX / Strix Halo integrated)
    // These are unified memory APUs; VRAM = system RAM in practice,
    // but this fallback gives a reasonable discrete estimate for name-only detection.
    if lower.contains("8060s") {
        return 32.0;
    }
    if lower.contains("8050s") {
        return 24.0;
    }
    if lower.contains("8060") && !lower.contains("8060s") {
        return 16.0;
    }
    if lower.contains("8050") && !lower.contains("8050s") {
        return 12.0;
    }
    // AMD Radeon 800M series (Ryzen AI 9 / Strix Point integrated)
    if lower.contains("890m") {
        return 16.0;
    }
    if lower.contains("880m") {
        return 12.0;
    }
    if lower.contains("870m") {
        return 8.0;
    }
    if lower.contains("860m") {
        return 8.0;
    }

    // Integrated GPUs (APU iGPUs) — must check before generic fallbacks
    // APU names like "AMD Radeon(TM) Graphics" or "Radeon Graphics" without
    // a discrete model number (RX/HD/R5/R7/R9) have very limited dedicated VRAM.
    if (lower.contains("radeon") || lower.contains("amd"))
        && !lower.contains("rx ")
        && !lower.contains("hd ")
        && !lower.contains(" r5 ")
        && !lower.contains(" r7 ")
        && !lower.contains(" r9 ")
        && !lower.contains("8060")
        && !lower.contains("8050")
        && (lower.contains("graphics") || lower.contains("igpu"))
    {
        return 0.5;
    }

    // Generic fallbacks
    if lower.contains("rtx") {
        return 8.0;
    }
    if lower.contains("gtx") {
        return 4.0;
    }
    if lower.contains("rx ") || lower.contains("radeon") {
        return 8.0;
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::SystemSpecs;

    #[test]
    fn test_parse_nvidia_smi_does_not_sum_multi_gpu_vram() {
        let text = "24564, NVIDIA GeForce RTX 4090\n24564, NVIDIA GeForce RTX 4090\n";
        let gpus = SystemSpecs::parse_nvidia_smi_list(text);

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].count, 2);
        let vram = gpus[0]
            .vram_gb
            .expect("VRAM should be parsed for RTX 4090 entries");
        // 24564 MiB ~= 23.99 GiB; must stay single-card VRAM, not 2x summed.
        assert!(vram > 23.0 && vram < 25.0, "unexpected VRAM value: {vram}");
    }

    #[test]
    fn test_parse_nvidia_smi_keeps_distinct_models() {
        let text = "24564, NVIDIA GeForce RTX 4090\n16376, NVIDIA GeForce RTX 4080\n";
        let gpus = SystemSpecs::parse_nvidia_smi_list(text);

        assert_eq!(gpus.len(), 2);
        assert!(gpus.iter().any(|g| g.name.contains("4090") && g.count == 1));
        assert!(gpus.iter().any(|g| g.name.contains("4080") && g.count == 1));
    }

    #[test]
    fn test_parse_nvidia_smi_gb10_gets_vram_estimate() {
        // DGX Spark reports GB10 with 0 VRAM from nvidia-smi
        let text = "0, NVIDIA GB10\n";
        let gpus = SystemSpecs::parse_nvidia_smi_list(text);

        assert_eq!(gpus.len(), 1);
        assert!(gpus[0].name.contains("GB10"));
        // estimate_vram_from_name should kick in and return 128GB
        let vram = gpus[0].vram_gb.expect("GB10 should have estimated VRAM");
        assert!(vram > 100.0, "GB10 VRAM should be ~128GB, got {vram}");
    }

    #[test]
    fn test_estimate_vram_gb10() {
        assert_eq!(super::estimate_vram_from_name("NVIDIA GB10"), 128.0);
        assert_eq!(super::estimate_vram_from_name("NVIDIA GB20"), 128.0);
    }

    #[test]
    fn test_estimate_vram_rtx_professional() {
        assert_eq!(super::estimate_vram_from_name("NVIDIA RTX A6000"), 48.0);
        assert_eq!(super::estimate_vram_from_name("NVIDIA RTX A5500"), 24.0);
        assert_eq!(super::estimate_vram_from_name("NVIDIA RTX A5000"), 24.0);
        assert_eq!(super::estimate_vram_from_name("NVIDIA RTX A4500"), 20.0);
        assert_eq!(super::estimate_vram_from_name("NVIDIA RTX A4000"), 16.0);
        assert_eq!(super::estimate_vram_from_name("NVIDIA RTX A2000"), 12.0);
    }

    #[test]
    fn test_parse_extended_discrete_gpu_not_unified() {
        // Discrete GPU: addressing_mode is "None", VRAM is reported normally
        let text = "None, 24564, NVIDIA GeForce RTX 4090\n";
        let gpus = SystemSpecs::parse_nvidia_smi_extended(text);

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4090");
        assert!(
            !gpus[0].unified_memory,
            "discrete GPU should not be unified"
        );
        let vram = gpus[0].vram_gb.expect("VRAM should be present");
        assert!(vram > 23.0 && vram < 25.0, "unexpected VRAM: {vram}");
    }

    #[test]
    fn test_parse_extended_tegra_unified_memory() {
        // NVIDIA Tegra / Grace Blackwell: ATS addressing, VRAM is [N/A]
        // On a real system, /proc/meminfo would provide the fallback.
        // In tests, /proc/meminfo may or may not exist.
        let text = "ATS, [N/A], NVIDIA Thor\n";
        let gpus = SystemSpecs::parse_nvidia_smi_extended(text);

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "NVIDIA Thor");
        assert!(gpus[0].unified_memory, "ATS should set unified_memory=true");
        // VRAM comes from /proc/meminfo; if unavailable, it's None
        // (on Linux test machines it will be Some, on macOS CI it will be None)
    }

    #[test]
    fn test_parse_extended_multi_gpu_discrete() {
        // Two discrete GPUs, no unified memory
        let text = "None, 24564, NVIDIA GeForce RTX 4090\nNone, 24564, NVIDIA GeForce RTX 4090\n";
        let gpus = SystemSpecs::parse_nvidia_smi_extended(text);

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].count, 2);
        assert!(!gpus[0].unified_memory);
    }

    #[test]
    fn test_gpu_bandwidth_known_gpus() {
        // Spot-check a few well-known GPUs
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 4090"),
            Some(1008.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 3060"),
            Some(360.0)
        );
        assert_eq!(super::gpu_memory_bandwidth_gbps("Tesla T4"), Some(320.0));
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA H100 SXM"),
            Some(3350.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA A100"),
            Some(1555.0)
        );
    }

    #[test]
    fn test_gpu_bandwidth_apple_silicon() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("Apple M1 Max"),
            Some(400.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("Apple M4 Pro"),
            Some(273.0)
        );
    }

    #[test]
    fn test_gpu_bandwidth_unknown_returns_none() {
        assert_eq!(super::gpu_memory_bandwidth_gbps("Some Random GPU"), None);
        assert_eq!(super::gpu_memory_bandwidth_gbps(""), None);
    }

    #[test]
    fn test_gpu_bandwidth_amd() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Radeon RX 7900 XTX"),
            Some(960.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Instinct MI300X"),
            Some(5300.0)
        );
    }

    #[test]
    fn test_parse_cpu_name_from_cpuinfo_prefers_model_name() {
        let cpuinfo = "\
processor   : 0
model name  : Qualcomm Kryo 680
Hardware    : Qualcomm Technologies, Inc SM8350
";
        assert_eq!(
            SystemSpecs::parse_cpu_name_from_cpuinfo(cpuinfo),
            Some("Qualcomm Kryo 680".to_string())
        );
    }

    #[test]
    fn test_parse_cpu_name_from_cpuinfo_uses_hardware_fallback() {
        let cpuinfo = "\
processor   : 0
Hardware    : Qualcomm Technologies, Inc SM8650
";
        assert_eq!(
            SystemSpecs::parse_cpu_name_from_cpuinfo(cpuinfo),
            Some("Qualcomm Technologies, Inc SM8650".to_string())
        );
    }

    #[test]
    fn test_parse_vulkan_device_names_from_summary_output() {
        let text = "\
GPU0:
deviceName         = Adreno (TM) 740
GPU1:
deviceName         = llvmpipe (LLVM 17.0.0, 256 bits)
";
        let names = SystemSpecs::parse_vulkan_device_names(text);
        assert_eq!(
            names,
            vec![
                "Adreno (TM) 740".to_string(),
                "llvmpipe (LLVM 17.0.0, 256 bits)".to_string()
            ]
        );
    }

    #[test]
    fn test_parse_vulkan_device_names_from_gpu_id_lines() {
        let text = "\
GPU id = 0 (Adreno (TM) 740)
GPU id = 1 (NVIDIA GeForce RTX 4090)
";
        let names = SystemSpecs::parse_vulkan_device_names(text);
        assert_eq!(
            names,
            vec![
                "Adreno (TM) 740".to_string(),
                "NVIDIA GeForce RTX 4090".to_string()
            ]
        );
    }

    #[test]
    fn test_is_software_vulkan_device() {
        assert!(SystemSpecs::is_software_vulkan_device(
            "llvmpipe (LLVM 17.0.0, 256 bits)"
        ));
        assert!(SystemSpecs::is_software_vulkan_device("SwiftShader Device"));
        assert!(!SystemSpecs::is_software_vulkan_device("Adreno (TM) 740"));
    }

    #[test]
    fn test_is_same_gpu_name_uses_normalized_exact_match() {
        assert!(SystemSpecs::is_same_gpu_name(
            "NVIDIA-GeForce RTX 4090",
            "nvidia geforce rtx 4090"
        ));
        assert!(!SystemSpecs::is_same_gpu_name("RTX", "RTX 4090"));
    }

    #[test]
    fn test_normalize_gpu_name_for_dedupe() {
        assert_eq!(
            SystemSpecs::normalize_gpu_name_for_dedupe(" Adreno (TM) 740 "),
            "adreno tm 740"
        );
    }

    // ── GpuBackend::label ────────────────────────────────────────────

    #[test]
    fn test_gpu_backend_labels() {
        assert_eq!(super::GpuBackend::Cuda.label(), "CUDA");
        assert_eq!(super::GpuBackend::Metal.label(), "Metal");
        assert_eq!(super::GpuBackend::Rocm.label(), "ROCm");
        assert_eq!(super::GpuBackend::Vulkan.label(), "Vulkan");
        assert_eq!(super::GpuBackend::Sycl.label(), "SYCL");
        assert_eq!(super::GpuBackend::CpuArm.label(), "CPU (ARM)");
        assert_eq!(super::GpuBackend::CpuX86.label(), "CPU (x86)");
        assert_eq!(super::GpuBackend::Ascend.label(), "NPU (Ascend)");
    }

    // ── parse_memory_size ────────────────────────────────────────────

    #[test]
    fn test_parse_memory_size_gb() {
        assert_eq!(super::parse_memory_size("32G"), Some(32.0));
        assert_eq!(super::parse_memory_size("32GB"), Some(32.0));
        assert_eq!(super::parse_memory_size("32GiB"), Some(32.0));
        assert_eq!(super::parse_memory_size("24g"), Some(24.0));
        assert_eq!(super::parse_memory_size("24gb"), Some(24.0));
    }

    #[test]
    fn test_parse_memory_size_mb() {
        let result = super::parse_memory_size("16384M").unwrap();
        assert!((result - 16.0).abs() < 0.01);
        let result = super::parse_memory_size("8192MB").unwrap();
        assert!((result - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_memory_size_tb() {
        let result = super::parse_memory_size("1T").unwrap();
        assert!((result - 1024.0).abs() < 0.01);
        let result = super::parse_memory_size("2TB").unwrap();
        assert!((result - 2048.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_memory_size_bare_number() {
        assert_eq!(super::parse_memory_size("16"), Some(16.0));
    }

    #[test]
    fn test_parse_memory_size_whitespace() {
        assert_eq!(super::parse_memory_size("  32G  "), Some(32.0));
    }

    #[test]
    fn test_parse_memory_size_empty() {
        assert_eq!(super::parse_memory_size(""), None);
        assert_eq!(super::parse_memory_size("  "), None);
    }

    #[test]
    fn test_parse_memory_size_invalid_suffix() {
        assert_eq!(super::parse_memory_size("32X"), None);
        assert_eq!(super::parse_memory_size("32KB"), None);
    }

    #[test]
    fn test_parse_memory_size_fractional() {
        assert_eq!(super::parse_memory_size("16.5G"), Some(16.5));
    }

    // ── with_gpu_memory_override ─────────────────────────────────────

    fn make_specs_no_gpu() -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: false,
            gpu_vram_gb: None,
            total_gpu_vram_gb: None,
            gpu_name: None,
            gpu_count: 0,
            unified_memory: false,
            backend: super::GpuBackend::CpuX86,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    fn make_specs_with_gpu() -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(8.0),
            total_gpu_vram_gb: Some(8.0),
            gpu_name: Some("NVIDIA RTX 3070".to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: super::GpuBackend::Cuda,
            gpus: vec![super::GpuInfo {
                name: "NVIDIA RTX 3070".to_string(),
                vram_gb: Some(8.0),
                backend: super::GpuBackend::Cuda,
                count: 1,
                unified_memory: false,
            }],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    #[test]
    fn test_gpu_override_creates_synthetic_gpu_when_none() {
        let specs = make_specs_no_gpu().with_gpu_memory_override(24.0);
        assert!(specs.has_gpu);
        assert_eq!(specs.gpu_vram_gb, Some(24.0));
        assert_eq!(specs.total_gpu_vram_gb, Some(24.0));
        assert_eq!(specs.gpu_count, 1);
        assert_eq!(specs.gpus.len(), 1);
        assert_eq!(specs.gpus[0].name, "User-specified GPU");
    }

    #[test]
    fn test_gpu_override_updates_existing_gpu() {
        let specs = make_specs_with_gpu().with_gpu_memory_override(24.0);
        assert_eq!(specs.gpu_vram_gb, Some(24.0));
        assert_eq!(specs.total_gpu_vram_gb, Some(24.0));
        assert_eq!(specs.gpus[0].vram_gb, Some(24.0));
        assert_eq!(specs.gpus[0].name, "NVIDIA RTX 3070");
    }

    #[test]
    fn test_gpu_override_multi_gpu_scales_total() {
        let mut specs = make_specs_with_gpu();
        specs.gpus[0].count = 2;
        let specs = specs.with_gpu_memory_override(24.0);
        assert_eq!(specs.gpu_vram_gb, Some(24.0));
        assert_eq!(specs.total_gpu_vram_gb, Some(48.0));
    }

    // ── is_amd_unified_memory_apu ────────────────────────────────────

    #[test]
    fn test_amd_unified_memory_apu_detection() {
        // Only Ryzen AI MAX / MAX+ have true unified memory
        assert!(super::is_amd_unified_memory_apu(
            "AMD Ryzen AI MAX+ 395 w/ Radeon 8060S"
        ));
        assert!(super::is_amd_unified_memory_apu("AMD Ryzen AI MAX 390"));
        // Regular Ryzen AI chips are NOT unified memory APUs
        assert!(!super::is_amd_unified_memory_apu(
            "AMD Ryzen AI 9 HX 370 w/ Radeon 890M"
        ));
        assert!(!super::is_amd_unified_memory_apu("AMD Ryzen AI 7 350"));
        assert!(!super::is_amd_unified_memory_apu("AMD Ryzen 9 7950X"));
        assert!(!super::is_amd_unified_memory_apu("Intel Core i9-14900K"));
    }

    // ── detect_windows_physical_total_ram_gb ─────────────────────────

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn test_windows_physical_total_ram_returns_none_on_non_windows() {
        // On Linux/macOS the function must return None (it is Windows-only).
        assert!(super::detect_windows_physical_total_ram_gb().is_none());
    }

    // ── bandwidth: RTX 20 series ─────────────────────────────────────

    #[test]
    fn test_bandwidth_rtx_20_series() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 2080 Ti"),
            Some(616.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 2060"),
            Some(336.0)
        );
    }

    // ── bandwidth: GTX 16 series ─────────────────────────────────────

    #[test]
    fn test_bandwidth_gtx_16_series() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce GTX 1660 Ti"),
            Some(288.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce GTX 1650"),
            Some(128.0)
        );
    }

    // ── bandwidth: RTX 50 series ─────────────────────────────────────

    #[test]
    fn test_bandwidth_rtx_50_series() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 5090"),
            Some(1792.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 5080"),
            Some(960.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 5070 Ti"),
            Some(896.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 5070"),
            Some(672.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 5060 Ti"),
            Some(448.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 5060"),
            Some(256.0)
        );
    }

    // ── bandwidth: AMD RX 6000 series ────────────────────────────────

    #[test]
    fn test_bandwidth_amd_rx_6000() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Radeon RX 6950 XT"),
            Some(576.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Radeon RX 6700 XT"),
            Some(384.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Radeon RX 6600"),
            Some(224.0)
        );
    }

    // ── bandwidth: NVIDIA professional ───────────────────────────────

    #[test]
    fn test_bandwidth_nvidia_professional() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA RTX A6000"),
            Some(768.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("NVIDIA RTX A4000"),
            Some(448.0)
        );
        assert_eq!(super::gpu_memory_bandwidth_gbps("NVIDIA L40S"), Some(864.0));
        assert_eq!(super::gpu_memory_bandwidth_gbps("NVIDIA L4"), Some(300.0));
    }

    // ── bandwidth: Apple Silicon all variants ────────────────────────

    #[test]
    fn test_bandwidth_apple_silicon_all() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("Apple M4 Ultra"),
            Some(819.0)
        );
        assert_eq!(super::gpu_memory_bandwidth_gbps("Apple M4"), Some(120.0));
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("Apple M3 Ultra"),
            Some(800.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("Apple M3 Max"),
            Some(400.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("Apple M3 Pro"),
            Some(150.0)
        );
        assert_eq!(super::gpu_memory_bandwidth_gbps("Apple M3"), Some(100.0));
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("Apple M1 Pro"),
            Some(200.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("Apple M1 Ultra"),
            Some(800.0)
        );
    }

    // ── bandwidth: AMD CDNA ──────────────────────────────────────────

    #[test]
    fn test_bandwidth_amd_cdna() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Instinct MI250X"),
            Some(3277.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Instinct MI210"),
            Some(1638.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Instinct MI100"),
            Some(1229.0)
        );
    }

    // ── bandwidth: AMD RDNA 4 ────────────────────────────────────────

    #[test]
    fn test_bandwidth_amd_rdna4() {
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Radeon RX 9070 XT"),
            Some(624.0)
        );
        assert_eq!(
            super::gpu_memory_bandwidth_gbps("AMD Radeon RX 9070"),
            Some(488.0)
        );
    }

    // ── compute capability tests ──────────────────────────────────────

    #[test]
    fn test_compute_capability_nvidia_generations() {
        // Pascal
        assert_eq!(super::gpu_compute_capability("Tesla P100"), Some((6, 1)));
        // Volta
        assert_eq!(
            super::gpu_compute_capability("Tesla V100-PCIE-16GB"),
            Some((7, 0))
        );
        // Turing
        assert_eq!(super::gpu_compute_capability("Tesla T4"), Some((7, 5)));
        assert_eq!(
            super::gpu_compute_capability("NVIDIA GeForce RTX 2080 Ti"),
            Some((7, 5))
        );
        assert_eq!(
            super::gpu_compute_capability("NVIDIA GeForce GTX 1660 Ti"),
            Some((7, 5))
        );
        // Ampere
        assert_eq!(super::gpu_compute_capability("NVIDIA A100"), Some((8, 0)));
        assert_eq!(
            super::gpu_compute_capability("NVIDIA GeForce RTX 3090"),
            Some((8, 6))
        );
        // Ada Lovelace
        assert_eq!(
            super::gpu_compute_capability("NVIDIA GeForce RTX 4090"),
            Some((8, 9))
        );
        assert_eq!(super::gpu_compute_capability("NVIDIA L40S"), Some((8, 9)));
        // Hopper
        assert_eq!(
            super::gpu_compute_capability("NVIDIA H100 SXM"),
            Some((9, 0))
        );
        // Blackwell
        assert_eq!(
            super::gpu_compute_capability("NVIDIA GeForce RTX 5090"),
            Some((10, 0))
        );
    }

    #[test]
    fn test_compute_capability_unknown_returns_none() {
        assert_eq!(super::gpu_compute_capability("Some Random GPU"), None);
        assert_eq!(super::gpu_compute_capability("Apple M4 Max"), None);
        assert_eq!(
            super::gpu_compute_capability("AMD Radeon RX 7900 XTX"),
            None
        );
    }

    #[test]
    fn test_is_integrated_gpu_name() {
        // Intel integrated
        assert!(SystemSpecs::is_integrated_gpu_name(
            "Intel(R) UHD Graphics 770"
        ));
        assert!(SystemSpecs::is_integrated_gpu_name(
            "Intel(R) HD Graphics 630"
        ));
        assert!(SystemSpecs::is_integrated_gpu_name(
            "Intel(R) Iris(R) Xe Graphics"
        ));
        assert!(SystemSpecs::is_integrated_gpu_name(
            "Intel(R) Iris(R) Plus Graphics"
        ));
        // Intel discrete should NOT match
        assert!(!SystemSpecs::is_integrated_gpu_name(
            "Intel(R) Arc(TM) A770"
        ));
        assert!(!SystemSpecs::is_integrated_gpu_name(
            "Intel(R) Arc(TM) B580"
        ));
        // Explicit "(integrated)" tag from APU detection
        assert!(SystemSpecs::is_integrated_gpu_name(
            "AMD Ryzen AI 9 HX 370 w/ Radeon 890M (integrated)"
        ));
    }

    #[test]
    fn test_is_integrated_gpu_name_amd() {
        // AMD integrated (generic "Radeon Graphics" with no RX/PRO)
        assert!(SystemSpecs::is_integrated_gpu_name(
            "AMD Radeon(TM) Graphics"
        ));
        assert!(SystemSpecs::is_integrated_gpu_name("AMD Radeon Graphics"));
        // AMD discrete should NOT match
        assert!(!SystemSpecs::is_integrated_gpu_name(
            "AMD Radeon RX 7900 XTX"
        ));
        assert!(!SystemSpecs::is_integrated_gpu_name("AMD Radeon Pro W7900"));
    }

    #[test]
    fn test_is_integrated_gpu_name_nvidia() {
        // NVIDIA GPUs are never integrated in the traditional sense
        assert!(!SystemSpecs::is_integrated_gpu_name(
            "NVIDIA GeForce RTX 4090"
        ));
        assert!(!SystemSpecs::is_integrated_gpu_name(
            "NVIDIA GeForce GTX 1650"
        ));
    }

    #[test]
    fn test_prefer_discrete_gpus_filters_integrated() {
        use super::GpuBackend;
        let gpus = vec![
            super::GpuInfo {
                name: "Intel(R) UHD Graphics 770".to_string(),
                vram_gb: Some(8.0),
                backend: GpuBackend::Vulkan,
                count: 1,
                unified_memory: false,
            },
            super::GpuInfo {
                name: "NVIDIA GeForce RTX 4090".to_string(),
                vram_gb: Some(4.0), // WMI 32-bit cap may report low value
                backend: GpuBackend::Cuda,
                count: 1,
                unified_memory: false,
            },
        ];
        let result = SystemSpecs::prefer_discrete_gpus(gpus);
        assert_eq!(result.len(), 1);
        assert!(result[0].name.contains("RTX 4090"));
    }

    #[test]
    fn test_prefer_discrete_gpus_keeps_igpu_only() {
        use super::GpuBackend;
        let gpus = vec![super::GpuInfo {
            name: "Intel(R) UHD Graphics 770".to_string(),
            vram_gb: Some(2.0),
            backend: GpuBackend::Vulkan,
            count: 1,
            unified_memory: false,
        }];
        let result = SystemSpecs::prefer_discrete_gpus(gpus);
        assert_eq!(result.len(), 1);
        assert!(result[0].name.contains("UHD"));
    }

    #[test]
    fn test_quant_min_compute_capability() {
        assert_eq!(
            super::quant_min_compute_capability("AWQ-4bit"),
            Some((7, 5))
        );
        assert_eq!(
            super::quant_min_compute_capability("AWQ-8bit"),
            Some((7, 5))
        );
        assert_eq!(
            super::quant_min_compute_capability("GPTQ-Int4"),
            Some((7, 5))
        );
        assert_eq!(
            super::quant_min_compute_capability("GPTQ-Int8"),
            Some((7, 5))
        );
        // GGUF quants have no CC restriction
        assert_eq!(super::quant_min_compute_capability("Q4_K_M"), None);
        assert_eq!(super::quant_min_compute_capability("Q8_0"), None);
    }

    #[test]
    fn test_ram_override_updates_ram_values() {
        let specs = SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(16.0),
            total_gpu_vram_gb: Some(16.0),
            gpu_name: Some("Test GPU".to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: super::GpuBackend::Cuda,
            gpus: vec![super::GpuInfo {
                name: "Test GPU".to_string(),
                vram_gb: Some(16.0),
                backend: super::GpuBackend::Cuda,
                count: 1,
                unified_memory: false,
            }],
            cluster_mode: false,
            cluster_node_count: 0,
        };

        let overridden = specs.with_ram_override(128.0);
        assert_eq!(overridden.total_ram_gb, 128.0);
        assert!((overridden.available_ram_gb - 115.2).abs() < 0.01);
        // Discrete GPU VRAM unchanged
        assert_eq!(overridden.gpu_vram_gb, Some(16.0));
        assert_eq!(overridden.total_gpu_vram_gb, Some(16.0));
    }

    #[test]
    fn test_ram_override_unified_memory_updates_gpu() {
        let specs = SystemSpecs {
            total_ram_gb: 36.0,
            available_ram_gb: 30.0,
            total_cpu_cores: 10,
            cpu_name: "Apple M2 Max".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(36.0),
            total_gpu_vram_gb: Some(36.0),
            gpu_name: Some("Apple M2 Max".to_string()),
            gpu_count: 1,
            unified_memory: true,
            backend: super::GpuBackend::Metal,
            gpus: vec![super::GpuInfo {
                name: "Apple M2 Max".to_string(),
                vram_gb: Some(36.0),
                backend: super::GpuBackend::Metal,
                count: 1,
                unified_memory: true,
            }],
            cluster_mode: false,
            cluster_node_count: 0,
        };

        let overridden = specs.with_ram_override(96.0);
        assert_eq!(overridden.total_ram_gb, 96.0);
        assert_eq!(overridden.gpu_vram_gb, Some(96.0));
        assert_eq!(overridden.total_gpu_vram_gb, Some(96.0));
        assert_eq!(overridden.gpus[0].vram_gb, Some(96.0));
    }

    #[test]
    fn test_cpu_core_override() {
        let specs = SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: false,
            gpu_vram_gb: None,
            total_gpu_vram_gb: None,
            gpu_name: None,
            gpu_count: 0,
            unified_memory: false,
            backend: super::GpuBackend::CpuX86,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        };

        let overridden = specs.with_cpu_core_override(64);
        assert_eq!(overridden.total_cpu_cores, 64);
        // Other fields unchanged
        assert_eq!(overridden.total_ram_gb, 32.0);
        assert_eq!(overridden.available_ram_gb, 24.0);
        assert!(!overridden.has_gpu);
    }
}
