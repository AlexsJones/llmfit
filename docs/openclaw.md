# OpenClaw Integration

[← Back to README](../README.md)

## OpenClaw integration

llmfit ships as an [OpenClaw](https://github.com/openclaw/openclaw) skill that lets the agent recommend hardware-appropriate local models and auto-configure Ollama/vLLM/LM Studio providers.

### Install the skill

```sh
# From the llmfit repo
./scripts/install-openclaw-skill.sh

# Or manually
cp -r skills/llmfit-advisor ~/.openclaw/skills/
```

Once installed, ask your OpenClaw agent things like:

- "What local models can I run?"
- "Recommend a coding model for my hardware"
- "Set up Ollama with the best models for my GPU"

The agent will call `llmfit recommend --json` under the hood, interpret the results, and offer to configure your `openclaw.json` with optimal model choices.

### How it works

The skill teaches the OpenClaw agent to:

1. Detect your hardware via `llmfit --json system`
2. Get ranked recommendations via `llmfit recommend --json`
3. Map HuggingFace model names to Ollama/vLLM/LM Studio tags
4. Configure `models.providers.ollama.models` in `openclaw.json`

See [skills/llmfit-advisor/SKILL.md](../skills/llmfit-advisor/SKILL.md) for the full skill definition.

---
