# Runtime Provider Integration

How llmfit detects and talks to Ollama, llama.cpp, Docker Model Runner, LM Studio, and remote instances.

[← Back to README](../README.md)

## Runtime provider integration

llmfit supports multiple local runtime providers:

- **Ollama** (daemon/API based pulls)
- **llama.cpp** (direct GGUF downloads from Hugging Face + local cache detection)
- **MLX** (Apple Silicon / mlx-community model cache + optional server) — MLX downloads map to `mlx-community/*` repos on HuggingFace, not the original model publisher
- **Docker Model Runner** (Docker Desktop's built-in model serving)
- **LM Studio** (local model server with REST API for model management + downloads)

When more than one compatible provider is available for a model, pressing `d` in the TUI opens a provider picker modal.

### Ollama integration

llmfit integrates with [Ollama](https://ollama.com) to detect which models you already have installed and to download new ones directly from the TUI.

### Requirements

- **Ollama must be installed and running** (`ollama serve` or the Ollama desktop app)
- llmfit connects to `http://localhost:11434` (Ollama's default API port)
- No configuration needed — if Ollama is running, llmfit detects it automatically

### Remote Ollama instances

To connect to Ollama running on a different machine or port, set the `OLLAMA_HOST` environment variable:

```sh
# Connect to Ollama on a specific IP and port
OLLAMA_HOST="http://192.168.1.100:11434" llmfit

# Connect via hostname  
OLLAMA_HOST="http://ollama-server:666" llmfit

# Works with all TUI and CLI commands
OLLAMA_HOST="http://192.168.1.100:11434" llmfit --cli
OLLAMA_HOST="http://192.168.1.100:11434" llmfit fit --perfect -n 5
```

This is useful for:
- Running llmfit on one machine while Ollama serves from another (e.g., GPU server + laptop client)
- Connecting to Ollama running in Docker containers with custom ports
- Using Ollama behind reverse proxies or load balancers

### How it works

On startup, llmfit queries `GET /api/tags` to list your installed Ollama models. Each installed model gets a green **✓** in the **Inst** column of the TUI. The system bar shows `Ollama: ✓ (N installed)`.

When you press `d` on a model, llmfit sends `POST /api/pull` to Ollama to download it. The row highlights with an animated progress indicator showing download progress in real-time. Once complete, the model is immediately available for use with Ollama.

If Ollama is not running, Ollama-specific operations are skipped; the TUI still supports other providers like llama.cpp where available.

### llama.cpp integration

llmfit integrates with [llama.cpp](https://github.com/ggml-org/llama.cpp) as a runtime/download provider in both TUI and CLI.

Requirements:

- `llama-cli` or `llama-server` available in `PATH` (for runtime detection)
- network access to Hugging Face for GGUF downloads

How it works:

- llmfit maps HF models to known GGUF repos (with heuristic fallbacks)
- downloads GGUF files into the local llama.cpp model cache
- marks models installed when matching GGUF files are present locally

#### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CPP_PATH` | *(none)* | Directory containing llama.cpp binaries (`llama-cli`, `llama-server`). Checked before `PATH` lookup. |
| `LLAMA_SERVER_PORT` | `8080` | Port used when probing a running `llama-server` health endpoint for runtime detection. |

If llama.cpp is installed in a non-standard location, set `LLAMA_CPP_PATH` so llmfit can find it without requiring it in your `PATH`.

### Docker Model Runner integration

llmfit integrates with [Docker Model Runner](https://docs.docker.com/desktop/features/model-runner/), Docker Desktop's built-in model serving feature.

Requirements:

- Docker Desktop with Model Runner enabled
- Default endpoint: `http://localhost:12434`

How it works:

- llmfit queries `GET /engines` to list models available in Docker Model Runner
- models are matched to the HF database using Ollama-style tag mapping (Docker Model Runner uses `ai/<tag>` naming)
- pressing `d` in the TUI pulls via `docker model pull`

### Remote Docker Model Runner instances

To connect to Docker Model Runner on a different host or port, set the `DOCKER_MODEL_RUNNER_HOST` environment variable:

```sh
DOCKER_MODEL_RUNNER_HOST="http://192.168.1.100:12434" llmfit
```

### LM Studio integration

llmfit integrates with [LM Studio](https://lmstudio.ai) as a local model server with built-in model download capabilities.

Requirements:

- LM Studio must be running with its local server enabled
- Default endpoint: `http://127.0.0.1:1234`

How it works:

- llmfit queries `GET /v1/models` to list models available in LM Studio
- pressing `d` in the TUI triggers a download via `POST /api/v1/models/download`
- download progress is tracked by polling `GET /api/v1/models/download-status`
- LM Studio accepts HuggingFace model names directly, so no name mapping is needed

### Remote LM Studio instances

To connect to LM Studio on a different host or port, set the `LMSTUDIO_HOST` environment variable:

```sh
LMSTUDIO_HOST="http://192.168.1.100:1234" llmfit
```

### API authentication

If your LM Studio instance has **Require API Key** enabled (required for MCP server access), set the `LMSTUDIO_API_KEY` environment variable to provide a Bearer token with all requests:

```sh
export LMSTUDIO_API_KEY="your-api-key-here"
llmfit
```

### Model name mapping

llmfit's database uses HuggingFace model names (e.g. `Qwen/Qwen2.5-Coder-14B-Instruct`) while Ollama uses its own naming scheme (e.g. `qwen2.5-coder:14b`). llmfit maintains an accurate mapping table between the two so that install detection and pulls resolve to the correct model. Each mapping is exact — `qwen2.5-coder:14b` maps to the Coder model, not the base `qwen2.5:14b`.

---
