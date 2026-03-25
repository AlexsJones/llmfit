pub mod fit;
pub mod hardware;
pub mod models;
pub mod plan;
pub mod providers;

pub mod cmd {
    pub fn new<S: AsRef<std::ffi::OsStr>>(program: S) -> std::process::Command {
        let mut cmd = std::process::Command::new(program);
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        cmd
    }
}

pub use fit::{FitLevel, InferenceRuntime, ModelFit, RunMode, ScoreComponents, SortColumn};
pub use hardware::{GpuBackend, SystemSpecs};
pub use models::{Capability, LlmModel, ModelDatabase, ModelFormat, UseCase};
pub use plan::{
    HardwareEstimate, PathEstimate, PlanCurrentStatus, PlanEstimate, PlanRequest, PlanRunPath,
    UpgradeDelta, estimate_model_plan, normalize_quant, resolve_model_selector,
};
pub use providers::{
    LlamaCppProvider, LmStudioProvider, MlxProvider, ModelProvider, OllamaProvider,
};
