pub mod fit;
pub mod hardware;
pub mod models;
pub mod providers;

// Re-export key types for convenience
pub use fit::{FitLevel, ModelFit, RunMode, ScoreComponents};
pub use hardware::{GpuBackend, SystemSpecs};
pub use models::{LlmModel, ModelDatabase, UseCase};
pub use providers::{ModelProvider, OllamaProvider};
