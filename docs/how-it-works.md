# How llmfit Works

The scoring model, speed estimation, and the model database.

[← Back to README](../README.md)

## How it works

1. **Hardware detection** -- Reads total/available RAM via `sysinfo`, counts CPU cores, and probes for GPUs:
   - **NVIDIA** -- Multi-GPU support via `nvidia-smi`. Aggregates VRAM across all detected GPUs. Falls back to VRAM estimation from GPU model name if reporting fails.
   - **AMD** -- Detected via `rocm-smi`.
   - **Intel Arc** -- Discrete VRAM via sysfs, integrated via `lspci`.
   - **Apple Silicon** -- Unified memory via `system_profiler`. VRAM = system RAM.
   - **Ascend** -- Detected via `npu-smi`.
   - **Backend detection** -- Automatically identifies the acceleration backend (CUDA, Metal, ROCm, SYCL, CPU ARM, CPU x86, Ascend) for speed estimation.

2. **Model database** -- Hundreds models sourced from the HuggingFace API, stored in `llmfit-core/data/hf_models.json` and embedded at compile time. Memory requirements are computed from parameter counts across a quantization hierarchy (Q8_0 through Q2_K). VRAM is the primary constraint for GPU inference; system RAM is the fallback for CPU-only execution.

   **MoE support** -- Models with Mixture-of-Experts architectures (Mixtral, DeepSeek-V2/V3) are detected automatically. Only a subset of experts is active per token, so the effective VRAM requirement is much lower than total parameter count suggests. For example, Mixtral 8x7B has 46.7B total parameters but only activates ~12.9B per token, reducing VRAM from 23.9 GB to ~6.6 GB with expert offloading.

3. **Dynamic quantization** -- Instead of assuming a fixed quantization, llmfit tries the best quality quantization that fits your hardware. It walks a hierarchy from Q8_0 (best quality) down to Q2_K (most compressed), picking the highest quality that fits in available memory. If nothing fits at full context, it tries again at half context.

4. **Multi-dimensional scoring** -- Each model is scored across four dimensions (0–100 each):

   | Dimension   | What it measures                                                               |
   |-------------|--------------------------------------------------------------------------------|
   | **Quality** | Parameter count, model family reputation, quantization penalty, task alignment |
   | **Speed**   | Estimated tokens/sec based on backend, params, and quantization                |
   | **Fit**     | Memory utilization efficiency (sweet spot: 50–80% of available memory)         |
   | **Context** | Context window capability vs target for the use case                           |

   Dimensions are combined into a weighted composite score. Weights vary by use-case category (General, Coding, Reasoning, Chat, Multimodal, Embedding). For example, Chat weights Speed higher (0.35) while Reasoning weights Quality higher (0.55). Models are ranked by composite score, with unrunnable models (Too Tight) always at the bottom.

   Task alignment within the Quality dimension uses a curated per-family benchmark table ([llmfit-core/data/use_case_benchmarks.json](../llmfit-core/data/use_case_benchmarks.json), aggregated from public coding/reasoning/chat leaderboards), so a strong coding model outranks a larger generalist for `--use-case coding` even at fewer parameters. Families without an entry fall back to name-based heuristics; corrections to the table are welcome PRs.

5. **Speed estimation** -- Token generation in LLM inference is memory-bandwidth-bound: each token requires reading the full model weights once from VRAM. When the GPU model is recognized, llmfit uses its actual memory bandwidth to estimate throughput:

   Formula: `(bandwidth_GB_s / model_size_GB) × efficiency_factor`

   The efficiency factor (0.55) and per-mode speed multipliers are tunable via the Advanced Configuration popup (`A` in the TUI). The defaults account for kernel overhead, KV-cache reads, and memory controller effects. This approach is validated against published benchmarks from llama.cpp ([Apple Silicon](https://github.com/ggml-org/llama.cpp/discussions/4167), [NVIDIA T4](https://github.com/ggml-org/llama.cpp/discussions/4225)) and real-world measurements.

   The bandwidth lookup table covers ~80 GPUs across NVIDIA (consumer + datacenter), AMD (RDNA + CDNA), and Apple Silicon families.

   For unrecognized GPUs, llmfit falls back to per-backend speed constants:

   | Backend      | Speed constant |
   |--------------|----------------|
   | CUDA         | 220            |
   | Metal        | 160            |
   | ROCm         | 180            |
   | SYCL         | 100            |
   | CPU (ARM)    | 90             |
   | CPU (x86)    | 70             |
   | NPU (Ascend) | 390            |

   Fallback formula: `K / params_b × quant_speed_multiplier`, with per-mode penalties tunable via the Advanced Configuration popup (`A` in the TUI).

6. **Fit analysis** -- Each model is evaluated for memory compatibility:

   **Run modes:**
   - **GPU** -- Model fits in VRAM. Fast inference.
   - **MoE** -- Mixture-of-Experts with expert offloading. Active experts in VRAM, inactive in RAM.
   - **CPU+GPU** -- VRAM insufficient, spills to system RAM with partial GPU offload.
   - **CPU** -- No GPU. Model loaded entirely into system RAM.

   **Fit levels:**
   - **Perfect** -- Recommended memory met on GPU. Requires GPU acceleration.
   - **Good** -- Fits with headroom. Best achievable for MoE offload or CPU+GPU.
   - **Marginal** -- Tight fit, or CPU-only (CPU-only always caps here).
   - **Too Tight** -- Not enough VRAM or system RAM anywhere.

---

## Model database

The model list is generated by `scripts/scrape_hf_models.py`, a standalone Python script (stdlib only, no pip dependencies) that queries the HuggingFace REST API. Hundreds models & providers including Meta Llama, Mistral, Qwen, Google Gemma, Microsoft Phi, DeepSeek, IBM Granite, Allen Institute OLMo, xAI Grok, Cohere, BigCode, 01.ai, Upstage, TII Falcon, HuggingFace, Zhipu GLM, Moonshot Kimi, Baidu ERNIE, and more. The scraper automatically detects MoE architectures via model config (`num_local_experts`, `num_experts_per_tok`) and known architecture mappings.

Model categories span general purpose, coding (CodeLlama, StarCoder2, WizardCoder, Qwen2.5-Coder, Qwen3-Coder), reasoning (DeepSeek-R1, Orca-2), multimodal/vision (Llama 3.2 Vision, Llama 4 Scout/Maverick, Qwen2.5-VL), chat, enterprise (IBM Granite), and embedding (nomic-embed, bge).

See [MODELS.md](../MODELS.md) for the full list.

The model database is embedded at compile time, so **end users** get updates by upgrading llmfit itself (`brew upgrade llmfit`, `scoop update llmfit`, or downloading a newer release). The commands below are for **contributors** refreshing the database from source:

To refresh the model database:

```sh
# Automated update (recommended)
make update-models

# Or run the script directly
./scripts/update_models.sh

# Or manually
python3 scripts/scrape_hf_models.py
cargo build --release
```

The scraper writes `llmfit-core/data/hf_models.json`, which is baked into the binary via `include_str!`. The automated update script backs up existing data, validates JSON output, and rebuilds the binary.

By default, the scraper enriches models with known GGUF download sources from providers like [unsloth](https://huggingface.co/unsloth) and [bartowski](https://huggingface.co/bartowski). Results are cached in `data/gguf_sources_cache.json` (7-day TTL) to avoid repeated API calls. Use `--no-gguf-sources` to skip enrichment for a faster scrape.

---
