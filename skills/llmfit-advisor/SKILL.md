---
name: llmfit-advisor
description: Detect local hardware (RAM, CPU, GPU/VRAM) and recommend the best-fit local LLM models with optimal quantization, speed estimates, and fit scoring.
metadata:
  {
    "openclaw":
      {
        "emoji": "🧠",
        "requires": { "bins": ["llmfit"] },
        "install":
          [
            {
              "id": "brew",
              "kind": "brew",
              "formula": "AlexsJones/llmfit",
              "bins": ["llmfit"],
              "label": "Install llmfit (brew tap AlexsJones/llmfit && brew install llmfit)",
            },
            {
              "id": "cargo",
              "kind": "node",
              "bins": ["llmfit"],
              "label": "Install llmfit (cargo install llmfit)",
            },
          ],
      },
  }
---

# llmfit-advisor

Hardware-aware local LLM advisor. Detects your system specs (RAM, CPU, GPU/VRAM) and recommends models that actually fit, with optimal quantization and speed estimates.

## When to use (trigger phrases)

Use this skill immediately when the user asks any of:

- "what local models can I run?"
- "which LLMs fit my hardware?"
- "recommend a local model"
- "what's the best model for my GPU?"
- "can I run Llama 70B locally?"
- "configure local models"
- "set up Ollama models"
- "what models fit my VRAM?"
- "help me pick a local model for coding"

Also use this skill when:

- The user wants to configure `models.providers.ollama` or `models.providers.lmstudio`
- The user mentions running models locally and you need to know what fits
- A model recommendation is needed and the user has local inference capability (Ollama, vLLM, LM Studio)

## Quick start

### Detect hardware

```bash
llmfit --json system
```

Returns JSON with CPU, RAM, GPU name, VRAM, multi-GPU info, and whether memory is unified (Apple Silicon).

### Get top recommendations

```bash
llmfit recommend --json --limit 5
```

Returns the top 5 models ranked by a composite score (quality, speed, fit, context) with optimal quantization for the detected hardware.

### Filter by use case

```bash
llmfit recommend --json --use-case coding --limit 3
llmfit recommend --json --use-case reasoning --limit 3
llmfit recommend --json --use-case chat --limit 3
```

Valid use cases: `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding`.

### Filter by minimum fit level

```bash
llmfit recommend --json --min-fit good --limit 10
```

Valid fit levels (best to worst): `perfect`, `good`, `marginal`.

## Understanding the output

### System JSON

```json
{
  "system": {
    "cpu_name": "Apple M2 Max",
    "cpu_cores": 12,
    "total_ram_gb": 32.0,
    "available_ram_gb": 24.5,
    "has_gpu": true,
    "gpu_name": "Apple M2 Max",
    "gpu_vram_gb": 32.0,
    "gpu_count": 1,
    "backend": "Metal",
    "unified_memory": true
  }
}
```

### Recommendation JSON

Each model in the `models` array includes:

| Field | Meaning |
|---|---|
| `name` | HuggingFace model ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`) |
| `provider` | Model provider (Meta, Alibaba, Google, etc.) |
| `params_b` | Parameter count in billions |
| `score` | Composite score 0–100 (higher is better) |
| `score_components` | Breakdown: `quality`, `speed`, `fit`, `context` (each 0–100) |
| `fit_level` | `Perfect`, `Good`, `Marginal`, or `TooTight` |
| `run_mode` | `GPU`, `CPU+GPU Offload`, or `CPU Only` |
| `best_quant` | Optimal quantization for the hardware (e.g. `Q5_K_M`, `Q4_K_M`) |
| `estimated_tps` | Estimated tokens per second |
| `memory_required_gb` | VRAM/RAM needed at this quantization |
| `memory_available_gb` | Available VRAM/RAM detected |
| `utilization_pct` | How much of available memory the model uses |
| `use_case` | What the model is designed for |
| `context_length` | Maximum context window |

### Fit levels explained

- **Perfect**: Model fits comfortably with room to spare. Ideal choice.
- **Good**: Model fits but uses most available memory. Will work well.
- **Marginal**: Model barely fits. May work but expect slower performance or reduced context.
- **TooTight**: Model does not fit. Do not recommend.

### Run modes explained

- **GPU**: Full GPU inference. Fastest. Model weights loaded entirely into VRAM.
- **CPU+GPU Offload**: Some layers on GPU, rest in system RAM. Slower than pure GPU.
- **CPU Only**: All inference on CPU using system RAM. Slowest but works without GPU.

## Configuring OpenClaw with results

After getting recommendations, configure the user's local model provider.

### For Ollama

Map the HuggingFace model name to its Ollama tag. Common mappings:

| llmfit name | Ollama tag |
|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | `llama3.1:8b` |
| `meta-llama/Llama-3.3-70B-Instruct` | `llama3.3:70b` |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | `qwen2.5-coder:7b` |
| `Qwen/Qwen2.5-72B-Instruct` | `qwen2.5:72b` |
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | `deepseek-coder-v2:16b` |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | `deepseek-r1:32b` |
| `google/gemma-2-9b-it` | `gemma2:9b` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistral:7b` |
| `microsoft/Phi-3-mini-4k-instruct` | `phi3:mini` |
| `microsoft/Phi-4-mini-instruct` | `phi4-mini` |

Then update `openclaw.json`:

```json
{
  "models": {
    "providers": {
      "ollama": {
        "models": ["ollama/<ollama-tag>"]
      }
    }
  }
}
```

And optionally set as default:

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "ollama/<ollama-tag>"
      }
    }
  }
}
```

### For vLLM / LM Studio

Use the HuggingFace model name directly as the model identifier with the appropriate provider prefix (`vllm/` or `lmstudio/`).

## Workflow example

When a user asks "what local models can I run?":

1. Run `llmfit --json system` to show hardware summary
2. Run `llmfit recommend --json --limit 5` to get top picks
3. Present the recommendations with scores and fit levels
4. If the user wants to configure one, map it to the appropriate Ollama/vLLM/LM Studio tag
5. Offer to update `openclaw.json` with the chosen model

When a user asks for a specific use case like "recommend a coding model":

1. Run `llmfit recommend --json --use-case coding --limit 3`
2. Present the coding-specific recommendations
3. Offer to pull via Ollama and configure

## Notes

- llmfit detects NVIDIA GPUs (via nvidia-smi), AMD GPUs (via rocm-smi), and Apple Silicon (unified memory).
- Multi-GPU setups aggregate VRAM across cards automatically.
- The `best_quant` field tells you the optimal quantization — higher quant (Q6_K, Q8_0) means better quality if VRAM allows.
- Speed estimates (`estimated_tps`) are approximate and vary by hardware and quantization.
- Models with `fit_level: "TooTight"` should never be recommended to users.
