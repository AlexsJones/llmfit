use sysinfo::System;

/// The acceleration backend for inference speed estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum GpuBackend {
    Cuda,
    Metal,
    Rocm,
    Sycl,    // Intel oneAPI
    CpuArm,
    CpuX86,
}

impl GpuBackend {
    pub fn label(&self) -> &'static str {
        match self {
            GpuBackend::Cuda => "CUDA",
            GpuBackend::Metal => "Metal",
            GpuBackend::Rocm => "ROCm",
            GpuBackend::Sycl => "SYCL",
            GpuBackend::CpuArm => "CPU (ARM)",
            GpuBackend::CpuX86 => "CPU (x86)",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemSpecs {
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub total_cpu_cores: usize,
    pub cpu_name: String,
    pub has_gpu: bool,
    pub gpu_vram_gb: Option<f64>,
    pub gpu_name: Option<String>,
    pub gpu_count: u32,
    pub unified_memory: bool,
    pub backend: GpuBackend,
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
        let cpu_name = sys.cpus()
            .first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string());

        let (has_gpu, gpu_vram_gb, gpu_name, gpu_count, unified_memory, backend) =
            Self::detect_gpu(available_ram_gb, &cpu_name);

        SystemSpecs {
            total_ram_gb,
            available_ram_gb,
            total_cpu_cores,
            cpu_name,
            has_gpu,
            gpu_vram_gb,
            gpu_name,
            gpu_count,
            unified_memory,
            backend,
        }
    }

    #[allow(clippy::type_complexity)]
    fn detect_gpu(available_ram_gb: f64, cpu_name: &str) -> (bool, Option<f64>, Option<String>, u32, bool, GpuBackend) {
        let cpu_backend = if cfg!(target_arch = "aarch64") || cpu_name.to_lowercase().contains("apple") {
            GpuBackend::CpuArm
        } else {
            GpuBackend::CpuX86
        };

        // Check for NVIDIA GPU via nvidia-smi (multi-GPU: one line per GPU)
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.total,name")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            if output.status.success() {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    let mut total_vram_mb: f64 = 0.0;
                    let mut count: u32 = 0;
                    let mut first_name: Option<String> = None;
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() { continue; }
                        let parts: Vec<&str> = line.splitn(2, ',').collect();
                        if let Some(vram_str) = parts.first() {
                            if let Ok(vram_mb) = vram_str.trim().parse::<f64>() {
                                total_vram_mb += vram_mb;
                                count += 1;
                                if first_name.is_none() {
                                    if let Some(name) = parts.get(1) {
                                        first_name = Some(name.trim().to_string());
                                    }
                                }
                            }
                        }
                    }
                    if count > 0 {
                        let mut vram_gb = total_vram_mb / 1024.0;
                        // Fallback: if nvidia-smi reports 0, estimate from GPU name
                        if vram_gb < 0.1 {
                            if let Some(ref name) = first_name {
                                vram_gb = estimate_vram_from_name(name);
                            }
                        }
                        let vram = if vram_gb > 0.0 { Some(vram_gb) } else { None };
                        return (true, vram, first_name, count, false, GpuBackend::Cuda);
                    }
                }
            }
        }

        // Check for AMD GPU via rocm-smi
        if let Ok(output) = std::process::Command::new("rocm-smi")
            .arg("--showmeminfo")
            .arg("vram")
            .output()
        {
            if output.status.success() {
                return (true, None, None, 1, false, GpuBackend::Rocm);
            }
        }

        // Check for Intel Arc GPU via sysfs (integrated or discrete)
        if let Some(vram) = Self::detect_intel_gpu() {
            return (true, Some(vram), Some("Intel Arc".to_string()), 1, false, GpuBackend::Sycl);
        }

        // Check for Apple Silicon (unified memory architecture)
        if let Some(vram) = Self::detect_apple_gpu(available_ram_gb) {
            let name = if cpu_name.to_lowercase().contains("apple") {
                Some(cpu_name.to_string())
            } else {
                Some("Apple Silicon".to_string())
            };
            return (true, Some(vram), name, 1, true, GpuBackend::Metal);
        }

        (false, None, None, 0, false, cpu_backend)
    }

    /// Estimate VRAM from a GPU model name when detection reports 0.
    /// Covers NVIDIA RTX 50/40/30 series and AMD RX 7000/6000 series.

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
                if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                    if vendor.trim() != "0x8086" {
                        continue;
                    }
                }

                // Look for total VRAM via DRM memory info
                // Intel discrete GPUs expose this under drm/card*/device/mem_info_vram_total
                let vram_path = card_path.join("device/mem_info_vram_total");
                if let Ok(vram_str) = std::fs::read_to_string(&vram_path) {
                    if let Ok(vram_bytes) = vram_str.trim().parse::<u64>() {
                        if vram_bytes > 0 {
                            let vram_gb = vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                            return Some(vram_gb);
                        }
                    }
                }

                // For integrated Intel GPUs, check if it's an Arc-class device
                // by looking for "Arc" in the device name via lspci
                if let Ok(output) = std::process::Command::new("lspci").output() {
                    if output.status.success() {
                        if let Ok(text) = String::from_utf8(output.stdout) {
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
            }
        }

        // Fallback: check lspci directly for Intel Arc devices
        // (covers cases where sysfs isn't available or card dirs don't exist)
        if let Ok(output) = std::process::Command::new("lspci").output() {
            if output.status.success() {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    for line in text.lines() {
                        let lower = line.to_lowercase();
                        if lower.contains("intel") && lower.contains("arc") {
                            return Some(0.0);
                        }
                    }
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
        let page_size: u64 = text.lines().next().and_then(|line| {
            line.split("page size of ").nth(1)?.split(' ').next()?.parse().ok()
        }).unwrap_or(16384); // Apple Silicon default is 16 KB pages

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
        line.split(':').nth(1)?.trim().trim_end_matches('.').parse().ok()
    }

    pub fn display(&self) {
        println!("\n=== System Specifications ===");
        println!("CPU: {} ({} cores)", self.cpu_name, self.total_cpu_cores);
        println!("Total RAM: {:.2} GB", self.total_ram_gb);
        println!("Available RAM: {:.2} GB", self.available_ram_gb);
        println!("Backend: {}", self.backend.label());

        if self.has_gpu {
            let gpu_label = self.gpu_name.as_deref().unwrap_or("Unknown");
            if self.unified_memory {
                println!(
                    "GPU: {} (unified memory, {:.2} GB shared)",
                    gpu_label,
                    self.gpu_vram_gb.unwrap_or(0.0)
                );
            } else {
                match self.gpu_vram_gb {
                    Some(vram) if vram > 0.0 => {
                        if self.gpu_count > 1 {
                            println!("GPU: {} x{} ({:.2} GB VRAM total)", gpu_label, self.gpu_count, vram);
                        } else {
                            println!("GPU: {} ({:.2} GB VRAM)", gpu_label, vram);
                        }
                    }
                    Some(_) => println!("GPU: {} (shared system memory)", gpu_label),
                    None => println!("GPU: {} (VRAM unknown)", gpu_label),
                }
            }
        } else {
            println!("GPU: Not detected");
        }
        println!();
    }
}

pub(crate) fn is_running_in_wsl() -> bool {
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

    ["/proc/sys/kernel/osrelease", "/proc/version"].iter().any(|path| {
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
    if lower.contains("5090") { return 32.0; }
    if lower.contains("5080") { return 16.0; }
    if lower.contains("5070 ti") { return 16.0; }
    if lower.contains("5070") { return 12.0; }
    if lower.contains("5060 ti") { return 16.0; }
    if lower.contains("5060") { return 8.0; }
    // NVIDIA RTX 40 series
    if lower.contains("4090") { return 24.0; }
    if lower.contains("4080") { return 16.0; }
    if lower.contains("4070 ti") { return 12.0; }
    if lower.contains("4070") { return 12.0; }
    if lower.contains("4060 ti") { return 16.0; }
    if lower.contains("4060") { return 8.0; }
    // NVIDIA RTX 30 series
    if lower.contains("3090") { return 24.0; }
    if lower.contains("3080 ti") { return 12.0; }
    if lower.contains("3080") { return 10.0; }
    if lower.contains("3070") { return 8.0; }
    if lower.contains("3060 ti") { return 8.0; }
    if lower.contains("3060") { return 12.0; }
    // Data center
    if lower.contains("h100") { return 80.0; }
    if lower.contains("a100") { return 80.0; }
    if lower.contains("l40") { return 48.0; }
    if lower.contains("a10") { return 24.0; }
    if lower.contains("t4") { return 16.0; }
    // AMD RX 7000 series
    if lower.contains("7900") { return 24.0; }
    if lower.contains("7800") { return 16.0; }
    if lower.contains("7700") { return 12.0; }
    if lower.contains("7600") { return 8.0; }
    // AMD RX 6000 series
    if lower.contains("6900") { return 16.0; }
    if lower.contains("6800") { return 16.0; }
    // Generic fallbacks
    if lower.contains("rtx") { return 8.0; }
    if lower.contains("gtx") { return 4.0; }
    if lower.contains("rx ") { return 8.0; }
    0.0
}
