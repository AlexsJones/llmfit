# Audio Model Support — Implementation Plan

This document describes the Rust changes needed to fully support
`pipeline_tag: automatic-speech-recognition` models (Whisper variants).

The JSON data additions (`llmfit-core/data/hf_models.json`) in this branch are ready.
The Rust integration changes below are the next step — open for discussion.

## Data changes (this branch)

`llmfit-core/data/hf_models.json` — 4 new entries with:
- `pipeline_tag: "automatic-speech-recognition"`
- `capabilities: ["audio"]`
- New fields (custom, don't break existing Rust deserialization via `#[serde(default)]`):
  - `_audio_rtf_gpu: f64` — Real-Time Factor on GPU (0.007 = 7x realtime)
  - `_audio_rtf_cpu: f64` — RTF on CPU
  - `_audio_vram_gb: f64` — VRAM needed at F16
  - `_audio_backends: [str]` — supported servers

## Rust changes needed

### 1. `llmfit-core/src/models.rs`

Add `Capability::Audio` to the `Capability` enum:

```rust
pub enum Capability {
    Vision,
    ToolUse,
    Reasoning,
    Embedding,
    Audio,   // ← new
}
```

Extend `LlmModel` deserialization to accept the new `_audio_*` fields:

```rust
// Inside LlmModel or a companion AudioMeta struct
#[serde(default)]
pub audio_rtf_gpu: Option<f64>,
#[serde(default)]
pub audio_rtf_cpu: Option<f64>,
#[serde(default)]
pub audio_vram_gb: Option<f64>,
#[serde(default)]
pub audio_backends: Vec<String>,
```

Add `UseCase::Audio` variant and detect it from `pipeline_tag`:

```rust
pub enum UseCase {
    General, Coding, Reasoning, Chat, Multimodal, Embedding,
    Audio,  // ← new
}

impl UseCase {
    pub fn from_model(model: &LlmModel) -> Self {
        // existing checks …
        if model.pipeline_tag.as_deref() == Some("automatic-speech-recognition")
            || model.capabilities.contains(&Capability::Audio)
        {
            UseCase::Audio
        } else { /* existing logic */ }
    }
}
```

### 2. `llmfit-core/src/fit.rs`

Audio models don't use tok/s — they use RTF (Real-Time Factor).
Add an `AudioFit` struct separate from `ModelFit`:

```rust
pub struct AudioFit {
    pub model: LlmModel,
    pub rtf_gpu: Option<f64>,
    pub rtf_cpu: f64,
    pub fits_vram: bool,
    pub fits_ram: bool,
    pub recommended_backend: String,
}
```

Scoring for audio: `score = accuracy_tier - latency_penalty - vram_penalty`.
Lower RTF = faster = better score.

### 3. `llmfit-core/src/providers.rs`

Add Whisper server provider detection:

```rust
/// mlx-openai-server Whisper endpoint (Apple Silicon path).
pub struct MlxWhisperProvider;
impl ModelProvider for MlxWhisperProvider {
    fn check_running(&self) -> Option<ProviderInfo> {
        probe_http("http://localhost:18000/v1/audio/transcriptions")
            .map(|_| ProviderInfo { name: "mlx-openai-server", port: 18000 })
    }
}

/// faster-whisper-server (Docker, NVIDIA/CPU path).
pub struct FasterWhisperProvider;
impl ModelProvider for FasterWhisperProvider {
    fn check_running(&self) -> Option<ProviderInfo> {
        probe_http("http://localhost:8000/health")
            .map(|_| ProviderInfo { name: "faster-whisper-server", port: 8000 })
    }
}
```

### 4. `llmfit-tui/src/main.rs` / CLI

Add `llmfit fit --kind audio` / `llmfit recommend --kind audio` to filter
to ASR models only (useful for the TLDR smart installer use case).

```bash
llmfit --json fit --kind audio -n 3
```

## Why this matters

Projects like [TLDR](https://github.com/melnikaite/tldr-free) (Chrome extension
that summarizes pages/videos) use an OpenAI-compatible Whisper backend for
audio transcription. Choosing the right Whisper model for your hardware is
just as confusing as choosing an LLM — RTF on a GTX 1660 Ti vs. Apple M3 Pro
is wildly different. This brings llmfit's hardware-aware recommendations to
the audio domain.
