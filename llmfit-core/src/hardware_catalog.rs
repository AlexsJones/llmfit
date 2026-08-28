use crate::hardware::{GpuBackend, GpuInfo, SystemSpecs};

/// A hardware configuration from the purchase catalog.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HardwareConfig {
    /// Human-readable name (e.g. "RTX 4070 Ti PC")
    pub name: &'static str,
    /// Approximate price in USD at time of catalog publication
    pub price_usd: u32,
    /// GPU VRAM in GB (None = CPU-only / integrated GPU)
    pub vram_gb: Option<f64>,
    /// System RAM in GB
    pub ram_gb: f64,
    /// CPU thread / core count
    pub cpu_cores: usize,
    /// GPU acceleration backend
    pub gpu_backend: GpuBackend,
    /// Short note for the user (e.g. "Unified memory", "Dual GPU")
    pub notes: &'static str,
}

impl HardwareConfig {
    /// Build a `SystemSpecs` that represents this hardware configuration.
    /// The specs are synthetic (not from real hardware detection) and are
    /// used to run the same fit pipeline as the regular `llmfit fit` command.
    pub fn to_specs(&self) -> SystemSpecs {
        let mut specs = SystemSpecs::synthetic(self.ram_gb, self.cpu_cores, self.gpu_backend);
        if let Some(vram) = self.vram_gb {
            let unified = matches!(self.gpu_backend, GpuBackend::Metal);
            specs.gpus.push(GpuInfo {
                name: self.name.to_string(),
                vram_gb: Some(vram),
                backend: self.gpu_backend,
                count: 1,
                unified_memory: unified,
            });
            specs.has_gpu = true;
            specs.gpu_vram_gb = Some(vram);
            specs.total_gpu_vram_gb = Some(vram);
            specs.gpu_name = Some(self.name.to_string());
            specs.gpu_count = 1;
            specs.unified_memory = unified;
        }
        specs
    }
}

/// The built-in hardware catalog.
/// Prices are approximate USD street prices as of mid-2025. Add new entries
/// in ascending price order so `configs_within_budget` preserves that order.
pub static CATALOG: &[HardwareConfig] = &[
    // ── CPU-only / mini PCs ────────────────────────────────────────────────
    HardwareConfig {
        name: "Beelink SEi12 Mini PC",
        price_usd: 200,
        vram_gb: None,
        ram_gb: 16.0,
        cpu_cores: 8,
        gpu_backend: GpuBackend::CpuX86,
        notes: "Compact x86 mini PC, CPU inference only",
    },
    HardwareConfig {
        name: "Beelink EQ12 Pro (32 GB)",
        price_usd: 280,
        vram_gb: None,
        ram_gb: 32.0,
        cpu_cores: 8,
        gpu_backend: GpuBackend::CpuX86,
        notes: "32 GB RAM, good for mid-size CPU-only models",
    },
    HardwareConfig {
        name: "Intel NUC 13 Pro (64 GB)",
        price_usd: 420,
        vram_gb: None,
        ram_gb: 64.0,
        cpu_cores: 12,
        gpu_backend: GpuBackend::CpuX86,
        notes: "High-RAM NUC for large CPU-offload models",
    },
    // ── Apple Silicon ──────────────────────────────────────────────────────
    HardwareConfig {
        name: "Mac mini M4 (16 GB)",
        price_usd: 599,
        vram_gb: Some(16.0),
        ram_gb: 16.0,
        cpu_cores: 10,
        gpu_backend: GpuBackend::Metal,
        notes: "Unified memory; GPU+CPU share the same 16 GB pool",
    },
    HardwareConfig {
        name: "Mac mini M4 Pro (24 GB)",
        price_usd: 799,
        vram_gb: Some(24.0),
        ram_gb: 24.0,
        cpu_cores: 14,
        gpu_backend: GpuBackend::Metal,
        notes: "Unified memory; 24 GB pool, excellent perf/watt",
    },
    HardwareConfig {
        name: "Mac mini M4 Pro (48 GB)",
        price_usd: 999,
        vram_gb: Some(48.0),
        ram_gb: 48.0,
        cpu_cores: 14,
        gpu_backend: GpuBackend::Metal,
        notes: "Unified memory; fits 70B models comfortably",
    },
    HardwareConfig {
        name: "MacBook Pro M4 Max (128 GB)",
        price_usd: 2499,
        vram_gb: Some(128.0),
        ram_gb: 128.0,
        cpu_cores: 16,
        gpu_backend: GpuBackend::Metal,
        notes: "Unified memory; runs virtually any open model",
    },
    // ── NVIDIA GPUs (discrete, paired with ~32 GB system RAM) ─────────────
    HardwareConfig {
        name: "RTX 3060 (12 GB) PC",
        price_usd: 400,
        vram_gb: Some(12.0),
        ram_gb: 32.0,
        cpu_cores: 8,
        gpu_backend: GpuBackend::Cuda,
        notes: "Entry CUDA GPU; good for 7B–13B models",
    },
    HardwareConfig {
        name: "RTX 4060 Ti (16 GB) PC",
        price_usd: 500,
        vram_gb: Some(16.0),
        ram_gb: 32.0,
        cpu_cores: 8,
        gpu_backend: GpuBackend::Cuda,
        notes: "16 GB VRAM at mid-range price",
    },
    HardwareConfig {
        name: "RTX 3090 (24 GB) PC",
        price_usd: 700,
        vram_gb: Some(24.0),
        ram_gb: 32.0,
        cpu_cores: 12,
        gpu_backend: GpuBackend::Cuda,
        notes: "Prosumer GPU; handles most 30B–34B models",
    },
    HardwareConfig {
        name: "RTX 4090 (24 GB) PC",
        price_usd: 1800,
        vram_gb: Some(24.0),
        ram_gb: 64.0,
        cpu_cores: 16,
        gpu_backend: GpuBackend::Cuda,
        notes: "Fastest consumer CUDA GPU available",
    },
    HardwareConfig {
        name: "2× RTX 3090 (48 GB total) PC",
        price_usd: 1400,
        vram_gb: Some(48.0),
        ram_gb: 64.0,
        cpu_cores: 16,
        gpu_backend: GpuBackend::Cuda,
        notes: "Dual-GPU tensor parallel; 48 GB VRAM total",
    },
    // ── AMD GPUs ───────────────────────────────────────────────────────────
    HardwareConfig {
        name: "RX 7800 XT (16 GB) PC",
        price_usd: 430,
        vram_gb: Some(16.0),
        ram_gb: 32.0,
        cpu_cores: 8,
        gpu_backend: GpuBackend::Rocm,
        notes: "16 GB at mid price; ROCm support on Linux",
    },
    HardwareConfig {
        name: "RX 7900 XTX (24 GB) PC",
        price_usd: 750,
        vram_gb: Some(24.0),
        ram_gb: 32.0,
        cpu_cores: 12,
        gpu_backend: GpuBackend::Rocm,
        notes: "Top AMD consumer GPU; ROCm on Linux",
    },
    // ── NVIDIA workstation / server ────────────────────────────────────────
    HardwareConfig {
        name: "NVIDIA RTX 4000 Ada (20 GB) Workstation",
        price_usd: 1250,
        vram_gb: Some(20.0),
        ram_gb: 64.0,
        cpu_cores: 16,
        gpu_backend: GpuBackend::Cuda,
        notes: "Professional GPU; ECC memory, great driver support",
    },
    HardwareConfig {
        name: "NVIDIA L4 (24 GB) Cloud Instance",
        price_usd: 2000,
        vram_gb: Some(24.0),
        ram_gb: 64.0,
        cpu_cores: 24,
        gpu_backend: GpuBackend::Cuda,
        notes: "Data-center GPU; approximate monthly cost as purchase price",
    },
];

/// Returns all catalog entries whose price is at or below `max_price_usd`.
pub fn configs_within_budget(max_price_usd: u32) -> Vec<&'static HardwareConfig> {
    CATALOG
        .iter()
        .filter(|c| c.price_usd <= max_price_usd)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_non_empty() {
        assert!(!CATALOG.is_empty());
    }

    #[test]
    fn budget_zero_returns_nothing() {
        assert!(configs_within_budget(0).is_empty());
    }

    #[test]
    fn budget_filters_correctly() {
        let results = configs_within_budget(600);
        assert!(results.iter().all(|c| c.price_usd <= 600));
        assert!(results.iter().any(|c| c.price_usd <= 600));
    }

    #[test]
    fn budget_max_returns_all() {
        let all = configs_within_budget(u32::MAX);
        assert_eq!(all.len(), CATALOG.len());
    }

    #[test]
    fn to_specs_does_not_panic() {
        for config in CATALOG {
            let _ = config.to_specs();
        }
    }
}
