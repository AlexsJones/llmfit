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
        let cpu_name = sys
            .cpus()
            .first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string());

        let gpus = Self::detect_all_gpus(available_ram_gb, &cpu_name);

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
        }
    }

    /// Detect all GPUs across all vendors. Returns a Vec sorted by VRAM descending
    /// (best GPU first). Unlike the old cascade, this does NOT short-circuit:
    /// a system with both NVIDIA and AMD GPUs will report both.
    fn detect_all_gpus(available_ram_gb: f64, cpu_name: &str) -> Vec<GpuInfo> {
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
        if let Some(vram) = Self::detect_apple_gpu(available_ram_gb) {
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
    fn detect_nvidia_gpus() -> Vec<GpuInfo> {
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

        Some(GpuInfo {
            name,
            vram_gb,
            backend,
            count: gpu_count,
            unified_memory: false,
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
        let mut gpu_count: u32 = 0;
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
                    gpu_count += 1;
                }
            }
        }

        if gpu_count == 0 {
            // rocm-smi succeeded but we couldn't parse VRAM; GPU exists though
            gpu_count = 1;
        }

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
                // Look for "Card Series" or "Card Model" lines
                for line in text.lines() {
                    let lower = line.to_lowercase();
                    if (lower.contains("card series") || lower.contains("card model"))
                        && let Some(val) = line.split(':').nth(1)
                    {
                        let name = val.trim().to_string();
                        if !name.is_empty() {
                            return Some(name);
                        }
                    }
                }
                None
            });

        let name = gpu_name.unwrap_or_else(|| "AMD GPU".to_string());
        let max_per_gpu_bytes = per_gpu_vram_bytes.into_iter().max().unwrap_or(0);
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

            // Try to get GPU name from lspci
            let gpu_name = Self::get_amd_gpu_name_lspci();
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
    fn get_amd_gpu_name_lspci() -> Option<String> {
        let text = Self::lspci_output()?;
        for line in text.lines() {
            let lower = line.to_lowercase();
            // VGA compatible controller or 3D controller with AMD/ATI
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

        // First pass: match exact slot (e.g. "01:00.0"), if available.
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
            .arg("-nn")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok());

        if local.is_some() {
            return local;
        }

        std::process::Command::new("flatpak-spawn")
            .args(["--host", "lspci", "-nn"])
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
                        return gpus;
                    }
                }

        // Fallback to wmic for older Windows
        Self::detect_gpu_windows_wmic_list()
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
            self.gpu_name = Some("User-specified GPU".to_string());
            self.gpu_count = 1;
            self.backend = backend;
        } else {
            // Override the primary (first) GPU's VRAM.
            self.gpus[0].vram_gb = Some(vram_gb);
            self.gpu_vram_gb = Some(vram_gb);
            self.has_gpu = true;
        }
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
    // Data center
    if lower.contains("h100") {
        return 80.0;
    }
    if lower.contains("a100") {
        return 80.0;
    }
    if lower.contains("l40") {
        return 48.0;
    }
    if lower.contains("a10") {
        return 24.0;
    }
    if lower.contains("t4") {
        return 16.0;
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
    // Integrated GPUs (APU iGPUs) — must check before generic fallbacks
    // APU names like "AMD Radeon(TM) Graphics" or "Radeon Graphics" without
    // a discrete model number (RX/HD/R5/R7/R9) have very limited dedicated VRAM.
    if (lower.contains("radeon") || lower.contains("amd"))
        && !lower.contains("rx ")
        && !lower.contains("hd ")
        && !lower.contains(" r5 ")
        && !lower.contains(" r7 ")
        && !lower.contains(" r9 ")
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
}
