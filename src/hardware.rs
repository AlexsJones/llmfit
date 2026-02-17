use sysinfo::System;

/// The acceleration backend for inference speed estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum GpuBackend {
    Cuda,
    Metal,
    Rocm,
    Vulkan,  // AMD/other GPUs without ROCm (e.g. Windows AMD, older AMD)
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
            GpuBackend::Vulkan => "Vulkan",
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

        // Check for AMD GPU via rocm-smi (Linux/ROCm)
        if let Some(result) = Self::detect_amd_gpu_rocm() {
            return result;
        }

        // Check for AMD GPU via sysfs on Linux
        if let Some(result) = Self::detect_amd_gpu_sysfs() {
            return result;
        }

        // Check for GPU via Windows WMI (covers AMD, NVIDIA without drivers, etc.)
        if let Some(result) = Self::detect_gpu_windows() {
            return result;
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

    /// Detect AMD GPU via rocm-smi (available on Linux with ROCm installed).
    /// Parses VRAM total and GPU name from rocm-smi output.
    #[allow(clippy::type_complexity)]
    fn detect_amd_gpu_rocm() -> Option<(bool, Option<f64>, Option<String>, u32, bool, GpuBackend)> {
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
        let mut total_vram_bytes: u64 = 0;
        let mut gpu_count: u32 = 0;
        for line in vram_text.lines() {
            let lower = line.to_lowercase();
            if lower.contains("total") && !lower.contains("used") {
                // Extract the numeric value (bytes)
                if let Some(val) = line.split_whitespace()
                    .filter_map(|w| w.parse::<u64>().ok())
                    .last()
                {
                    if val > 0 {
                        total_vram_bytes += val;
                        gpu_count += 1;
                    }
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
                    if lower.contains("card series") || lower.contains("card model") {
                        if let Some(val) = line.split(':').nth(1) {
                            let name = val.trim().to_string();
                            if !name.is_empty() {
                                return Some(name);
                            }
                        }
                    }
                }
                None
            });

        let vram_gb = if total_vram_bytes > 0 {
            Some(total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else {
            // Fallback: estimate from name
            gpu_name.as_ref().map(|n| estimate_vram_from_name(n)).filter(|&v| v > 0.0)
        };

        Some((true, vram_gb, gpu_name, gpu_count, false, GpuBackend::Rocm))
    }

    /// Detect AMD GPU via sysfs on Linux (works without ROCm installed).
    /// AMD vendor ID is 0x1002.
    #[allow(clippy::type_complexity)]
    fn detect_amd_gpu_sysfs() -> Option<(bool, Option<f64>, Option<String>, u32, bool, GpuBackend)> {
        if !cfg!(target_os = "linux") {
            return None;
        }

        let entries = std::fs::read_dir("/sys/class/drm").ok()?;
        for entry in entries.flatten() {
            let card_path = entry.path();
            let name = card_path.file_name()?.to_str()?;
            // Only look at cardN entries, not cardN-DP-1 etc.
            if !name.starts_with("card") || name.contains('-') {
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
            if let Ok(vram_str) = std::fs::read_to_string(&vram_path) {
                if let Ok(vram_bytes) = vram_str.trim().parse::<u64>() {
                    if vram_bytes > 0 {
                        vram_gb = Some(vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
                    }
                }
            }

            // Try to get GPU name from lspci
            let gpu_name = Self::get_amd_gpu_name_lspci();

            // If we still don't have VRAM, try to estimate from name
            if vram_gb.is_none() {
                if let Some(ref name) = gpu_name {
                    let estimated = estimate_vram_from_name(name);
                    if estimated > 0.0 {
                        vram_gb = Some(estimated);
                    }
                }
            }

            // AMD GPU without ROCm — Vulkan is the most likely inference backend
            return Some((true, vram_gb, gpu_name, 1, false, GpuBackend::Vulkan));
        }
        None
    }

    /// Extract AMD GPU name from lspci output.
    fn get_amd_gpu_name_lspci() -> Option<String> {
        let output = std::process::Command::new("lspci").output().ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;
        for line in text.lines() {
            let lower = line.to_lowercase();
            // VGA compatible controller or 3D controller with AMD/ATI
            if (lower.contains("vga") || lower.contains("3d")) && (lower.contains("amd") || lower.contains("ati")) {
                // Extract the part after the colon, e.g. "Advanced Micro Devices ... [Radeon RX 5700 XT]"
                if let Some(desc) = line.split("]:").last() {
                    let desc: &str = desc.trim();
                    // Try to extract the bracketed name like "[Radeon RX 5700 XT]"
                    if let Some(start) = desc.rfind('[') {
                        if let Some(end) = desc.rfind(']') {
                            if start < end {
                                return Some(desc[start + 1..end].to_string());
                            }
                        }
                    }
                    return Some(desc.to_string());
                }
            }
        }
        None
    }

    /// Detect GPU on Windows via WMI (Win32_VideoController).
    /// This works for AMD, NVIDIA (without toolkit), and Intel GPUs.
    #[allow(clippy::type_complexity)]
    fn detect_gpu_windows() -> Option<(bool, Option<f64>, Option<String>, u32, bool, GpuBackend)> {
        if !cfg!(target_os = "windows") {
            return None;
        }

        // Use PowerShell to query WMI — more reliable than wmic (deprecated)
        let output = std::process::Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg("Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }")
            .output()
            .ok()?;

        if !output.status.success() {
            // Fallback to wmic for older Windows
            return Self::detect_gpu_windows_wmic();
        }

        let text = String::from_utf8(output.stdout).ok()?;
        Self::parse_windows_gpu_entries(&text)
    }

    /// Fallback Windows GPU detection via wmic (works on older systems).
    #[allow(clippy::type_complexity)]
    fn detect_gpu_windows_wmic() -> Option<(bool, Option<f64>, Option<String>, u32, bool, GpuBackend)> {
        let output = std::process::Command::new("wmic")
            .arg("path")
            .arg("win32_VideoController")
            .arg("get")
            .arg("Name,AdapterRAM")
            .arg("/format:csv")
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let text = String::from_utf8(output.stdout).ok()?;
        // CSV format: Node,AdapterRAM,Name
        let mut best_name: Option<String> = None;
        let mut best_vram: u64 = 0;

        for line in text.lines().skip(1) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 3 {
                let vram: u64 = parts[1].trim().parse().unwrap_or(0);
                let name = parts[2..].join(",").trim().to_string();
                // Skip Microsoft Basic Display Adapter and similar virtual adapters
                let lower = name.to_lowercase();
                if lower.contains("microsoft") || lower.contains("basic") || lower.contains("virtual") {
                    continue;
                }
                if vram > best_vram || best_name.is_none() {
                    best_vram = vram;
                    best_name = Some(name);
                }
            }
        }

        if let Some(name) = best_name {
            let backend = Self::infer_gpu_backend(&name);
            let mut vram_gb = best_vram as f64 / (1024.0 * 1024.0 * 1024.0);
            // WMI AdapterRAM is capped at 4 GB (32-bit field). Estimate from name if it looks wrong.
            if vram_gb < 0.1 || (vram_gb <= 4.1 && estimate_vram_from_name(&name) > 4.1) {
                let estimated = estimate_vram_from_name(&name);
                if estimated > 0.0 {
                    vram_gb = estimated;
                }
            }
            let vram = if vram_gb > 0.0 { Some(vram_gb) } else { None };
            return Some((true, vram, Some(name), 1, false, backend));
        }

        None
    }

    /// Parse GPU entries from PowerShell output (Name|AdapterRAM per line).
    #[allow(clippy::type_complexity)]
    fn parse_windows_gpu_entries(text: &str) -> Option<(bool, Option<f64>, Option<String>, u32, bool, GpuBackend)> {
        let mut best_name: Option<String> = None;
        let mut best_vram: u64 = 0;
        let mut count: u32 = 0;

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, '|').collect();
            let name = parts[0].trim().to_string();
            let vram: u64 = parts.get(1).and_then(|v| v.trim().parse().ok()).unwrap_or(0);

            // Skip virtual/basic display adapters
            let lower = name.to_lowercase();
            if lower.contains("microsoft") || lower.contains("basic") || lower.contains("virtual") || lower.is_empty() {
                continue;
            }

            count += 1;
            if vram > best_vram || best_name.is_none() {
                best_vram = vram;
                best_name = Some(name);
            }
        }

        if let Some(name) = best_name {
            let backend = Self::infer_gpu_backend(&name);
            let mut vram_gb = best_vram as f64 / (1024.0 * 1024.0 * 1024.0);
            // WMI AdapterRAM is a 32-bit field, capped at ~4 GB.
            // If reported value is suspiciously low, estimate from GPU name.
            if vram_gb < 0.1 || (vram_gb <= 4.1 && estimate_vram_from_name(&name) > 4.1) {
                let estimated = estimate_vram_from_name(&name);
                if estimated > 0.0 {
                    vram_gb = estimated;
                }
            }
            let vram = if vram_gb > 0.0 { Some(vram_gb) } else { None };
            Some((true, vram, Some(name), count.max(1), false, backend))
        } else {
            None
        }
    }

    /// Infer the most likely inference backend from a GPU name string.
    fn infer_gpu_backend(name: &str) -> GpuBackend {
        let lower = name.to_lowercase();
        if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("quadro") || lower.contains("tesla") || lower.contains("rtx") {
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
    // AMD RX 9000 series
    if lower.contains("9070 xt") { return 16.0; }
    if lower.contains("9070") { return 12.0; }
    // AMD RX 7000 series
    if lower.contains("7900 xtx") { return 24.0; }
    if lower.contains("7900") { return 20.0; }
    if lower.contains("7800") { return 16.0; }
    if lower.contains("7700") { return 12.0; }
    if lower.contains("7600") { return 8.0; }
    // AMD RX 6000 series
    if lower.contains("6950") { return 16.0; }
    if lower.contains("6900") { return 16.0; }
    if lower.contains("6800") { return 16.0; }
    if lower.contains("6750") { return 12.0; }
    if lower.contains("6700") { return 12.0; }
    if lower.contains("6650") { return 8.0; }
    if lower.contains("6600") { return 8.0; }
    if lower.contains("6500") { return 4.0; }
    // AMD RX 5000 series
    if lower.contains("5700 xt") { return 8.0; }
    if lower.contains("5700") { return 8.0; }
    if lower.contains("5600") { return 6.0; }
    if lower.contains("5500") { return 4.0; }
    // Generic fallbacks
    if lower.contains("rtx") { return 8.0; }
    if lower.contains("gtx") { return 4.0; }
    if lower.contains("rx ") || lower.contains("radeon") { return 8.0; }
    0.0
}
