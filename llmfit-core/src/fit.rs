use crate::gguf::Gguf;
use crate::hardware::{Hardware, GpuBackend, InferenceRuntime};
use crate::model::{Model, ModelArchitecture, ModelQuantization};

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use strum_macros::EnumString;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelFit {
    pub model: Model,
    pub fits: Vec<Fit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fit {
    pub quantization: ModelQuantization,
    pub in_memory: bool,
    pub vram_gb: f32,
    pub ram_gb: f32,
    pub layers_on_gpu: u32,
    pub estimated_tps: f32,
}

#[derive(Debug, Default, Clone, Copy, EnumString, Serialize, Deserialize, PartialEq, Eq)]
#[strum(serialize_all = "kebab-case")]
pub enum SortColumn {
    #[default]
    #[strum(serialize = "tps", serialize = "speed")]
    EstimatedTps,
    #[strum(serialize = "vram", serialize = "gpu-mem")]
    Vram,
    #[strum(serialize = "ram", serialize = "cpu-mem")]
    Ram,
    #[strum(serialize = "layers", serialize = "gpu-layers")]
    GpuLayers,
}

impl From<SortColumn> for fn(&Fit, &Fit) -> Ordering {
    fn from(sort_column: SortColumn) -> Self {
        match sort_column {
            SortColumn::EstimatedTps => |a, b| {
                b.estimated_tps
                    .partial_cmp(&a.estimated_tps)
                    .unwrap_or(Ordering::Equal)
            },
            SortColumn::Vram => |a, b| a.vram_gb.partial_cmp(&b.vram_gb).unwrap_or(Ordering::Equal),
            SortColumn::Ram => |a, b| a.ram_gb.partial_cmp(&b.ram_gb).unwrap_or(Ordering::Equal),
            SortColumn::GpuLayers => |a, b| b.layers_on_gpu.cmp(&a.layers_on_gpu),
        }
    }
}

pub fn backend_compatible(model_architecture: &ModelArchitecture, gpu_backend: &GpuBackend) -> bool {
    match model_architecture {
        ModelArchitecture::Llama | ModelArchitecture::Mistral => true,
        ModelArchitecture::Mixtral if *gpu_backend == GpuBackend::Nvidia => true,
        _ => false,
    }
}

fn get_performance_factor(
    gpu_backend: &GpuBackend,
    inference_runtime: &InferenceRuntime,
) -> f32 {
    // Performance factor based on GPU backend and inference runtime
    // Higher is better. These are relative values.
    match (gpu_backend, inference_runtime) {
        (GpuBackend::Nvidia, InferenceRuntime::LlamaCpp) => 100.0, // Baseline
        (GpuBackend::Nvidia, InferenceRuntime::VLlm) => 200.0,    // vLLM on NVIDIA
        (GpuBackend::Amd, _) => 80.0,                             // AMD
        (GpuBackend::Apple, _) => 70.0,                           // Apple Silicon
        (GpuBackend::Ascend, _) => 390.0,                         // Huawei Ascend 910b NPU
        (_, InferenceRuntime::VLlm) => 200.0, // vLLM fallback for non-NVIDIA
    }
}

pub fn calculate_fit(
    model: &Model,
    quantization: &ModelQuantization,
    hardware: &Hardware,
) -> Option<Fit> {
    let gguf = Gguf::from_model(model, quantization);

    let mut layers_on_gpu = 0;
    let mut vram_gb = 0.0;
    let mut ram_gb = gguf.non_tensor_data_size_gb;

    if hardware.gpu.is_some() {
        let gpu = hardware.gpu.as_ref().unwrap();
        let available_vram = gpu.vram_gb.unwrap_or(0.0) - gguf.overhead_vram_gb;

        // Check if the model is compatible with the GPU backend
        if !backend_compatible(&model.architecture, &gpu.backend) {
            return None;
        }

        // Calculate how many layers can be offloaded to the GPU
        if available_vram > gguf.layer_vram_gb(1) {
            layers_on_gpu = ((available_vram - gguf.static_vram_gb) / gguf.per_layer_vram_gb)
                .floor() as u32;
            layers_on_gpu = layers_on_gpu.min(model.params.num_layers as u32);
        }

        vram_gb = gguf.static_vram_gb + gguf.layer_vram_gb(layers_on_gpu as usize);
    }

    let layers_on_cpu = model.params.num_layers as u32 - layers_on_gpu;
    ram_gb += gguf.layer_ram_gb(layers_on_cpu as usize);

    let in_memory = ram_gb < hardware.ram_gb;

    let performance_factor = if let Some(gpu) = &hardware.gpu {
        get_performance_factor(&gpu.backend, &gpu.inference_runtime)
    } else {
        1.0 // CPU performance factor
    };

    let estimated_tps =
        (layers_on_gpu as f32 * performance_factor) + (layers_on_cpu as f32 * 1.0);

    Some(Fit {
        quantization: quantization.clone(),
        in_memory,
        vram_gb,
        ram_gb,
        layers_on_gpu,
        estimated_tps,
    })
}

pub fn get_best_fit(model: &Model, hardware: &Hardware) -> Option<ModelFit> {
    let mut fits: Vec<Fit> = model
        .quantizations
        .iter()
        .filter_map(|q| calculate_fit(model, q, hardware))
        .collect();

    if fits.is_empty() {
        return None;
    }

    fits.sort_by(|a, b| {
        b.estimated_tps
            .partial_cmp(&a.estimated_tps)
            .unwrap_or(Ordering::Equal)
    });

    Some(ModelFit {
        model: model.clone(),
        fits,
    })
}

pub fn get_fits_for_model(
    model: &Model,
    hardware: &Hardware,
    sort_column: fn(&Fit, &Fit) -> Ordering,
) -> Option<ModelFit> {
    let mut fits: Vec<Fit> = model
        .quantizations
        .iter()
        .filter_map(|q| calculate_fit(model, q, hardware))
        .collect();

    if fits.is_empty() {
        return None;
    }

    fits.sort_by(sort_column);

    Some(ModelFit {
        model: model.clone(),
        fits,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelKey(String, String);

impl ModelKey {
    pub fn from_fit(fit: &ModelFit) -> Self {
        Self(
            fit.model.id.clone(),
            fit.fits[0].quantization.name.clone(),
        )
    }
}

pub fn get_model_fits_from_db(
    db: &HashMap<String, Model>,
    hardware: &Hardware,
    filter: Option<&str>,
    limit: Option<usize>,
    sort_column: fn(&Fit, &Fit) -> Ordering,
    fit: Option<FitArg>,
) -> Vec<ModelFit> {
    let mut model_fits: Vec<ModelFit> = db
        .values()
        .filter_map(|model| {
            if let Some(f) = filter {
                if !model.id.to_lowercase().contains(&f.to_lowercase()) {
                    return None;
                }
            }
            get_fits_for_model(model, hardware, sort_column)
        })
        .collect();

    if let Some(fit_filter) = fit {
        model_fits = model_fits
            .into_iter()
            .filter(|mf| {
                let best_fit = &mf.fits[0];
                match fit_filter {
                    FitArg::Perfect => {
                        best_fit.layers_on_gpu == mf.model.params.num_layers as u32
                    }
                }
            })
            .collect();
    }

    if let Some(l) = limit {
        model_fits.truncate(l);
    }

    model_fits
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, EnumString)]
#[strum(serialize_all = "kebab-case")]
pub enum FitArg {
    Perfect,
}
