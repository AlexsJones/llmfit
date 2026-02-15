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

        // Check for Apple Silicon (unified memory architecture)
        if let Some(vram) = Self::detect_apple_gpu(available_ram_gb) {
            return (true, Some(vram), true);
        }

        (false, None, false)
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
                    Some(vram) => println!("GPU: Detected ({:.2} GB VRAM)", vram),
                    None => println!("GPU: Detected (VRAM unknown)"),
                }
            }
        } else {
            println!("GPU: Not detected");
        }
        println!();
    }
}
