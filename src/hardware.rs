use sysinfo::System;

#[derive(Debug, Clone)]
pub struct SystemSpecs {
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub total_cpu_cores: usize,
    pub cpu_name: String,
    pub has_gpu: bool,
    pub gpu_vram_gb: Option<f64>,
    pub unified_memory: bool, // Apple Silicon: GPU shares system RAM
}

impl SystemSpecs {
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let total_ram_bytes = sys.total_memory();
        let available_ram_bytes = sys.available_memory();
        let total_ram_gb = total_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_ram_gb = available_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        let total_cpu_cores = sys.cpus().len();
        let cpu_name = sys.cpus()
            .first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string());

        let (has_gpu, gpu_vram_gb, unified_memory) = Self::detect_gpu(available_ram_gb);

        SystemSpecs {
            total_ram_gb,
            available_ram_gb,
            total_cpu_cores,
            cpu_name,
            has_gpu,
            gpu_vram_gb,
            unified_memory,
        }
    }

    fn detect_gpu(available_ram_gb: f64) -> (bool, Option<f64>, bool) {
        // Check for NVIDIA GPU via nvidia-smi
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.total")
            .arg("--format=csv,noheader,nounits")
            .output()
            && output.status.success()
                && let Ok(vram_str) = String::from_utf8(output.stdout)
                    && let Ok(vram_mb) = vram_str.trim().parse::<f64>() {
                        return (true, Some(vram_mb / 1024.0), false);
                    }

        // Check for AMD GPU via rocm-smi
        if let Ok(output) = std::process::Command::new("rocm-smi")
            .arg("--showmeminfo")
            .arg("vram")
            .output()
            && output.status.success() {
                return (true, None, false);
            }

        // Check for Intel Arc GPU via sysfs (integrated or discrete)
        if let Some(vram) = Self::detect_intel_gpu() {
            return (true, Some(vram), false);
        }

        // Check for Apple Silicon (unified memory architecture)
        if let Some(vram) = Self::detect_apple_gpu(available_ram_gb) {
            return (true, Some(vram), true);
        }

        (false, None, false)
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
    /// Returns available system RAM as VRAM since memory is unified.
    fn detect_apple_gpu(available_ram_gb: f64) -> Option<f64> {
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
            // Unified memory: GPU can use most of system RAM.
            // Report available RAM as the VRAM pool (it's shared).
            Some(available_ram_gb)
        } else {
            None
        }
    }

    pub fn display(&self) {
        println!("\n=== System Specifications ===");
        println!("CPU: {} ({} cores)", self.cpu_name, self.total_cpu_cores);
        println!("Total RAM: {:.2} GB", self.total_ram_gb);
        println!("Available RAM: {:.2} GB", self.available_ram_gb);

        if self.has_gpu {
            if self.unified_memory {
                println!(
                    "GPU: Apple Silicon (unified memory, {:.2} GB shared)",
                    self.gpu_vram_gb.unwrap_or(0.0)
                );
            } else {
                match self.gpu_vram_gb {
                    Some(vram) if vram > 0.0 => println!("GPU: Detected ({:.2} GB VRAM)", vram),
                    Some(_) => println!("GPU: Intel Arc (shared system memory)"),
                    None => println!("GPU: Detected (VRAM unknown)"),
                }
            }
        } else {
            println!("GPU: Not detected");
        }
        println!();
    }
}
