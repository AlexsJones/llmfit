# Custom Models

Add your own models locally without a rebuild, or contribute a model to the built-in catalog.

[← Back to README](../README.md)

### Adding your own models locally (no rebuild needed)

You don't need to modify llmfit or wait for a release to see extra models. Create a `custom_models.json` in llmfit's data directory:

- Linux: `~/.local/share/llmfit/custom_models.json`
- macOS: `~/Library/Application Support/llmfit/custom_models.json`
- Windows: `%APPDATA%\llmfit\custom_models.json`

(or point the `LLMFIT_CUSTOM_MODELS` env var at any path). The file is a JSON array using the same entry format as the built-in catalog — see [llmfit-core/data/schema.json](../llmfit-core/data/schema.json); only a few fields are required:

```json
[
  {
    "name": "my-org/My-Model-7B",
    "provider": "my-org",
    "parameter_count": "7B",
    "min_ram_gb": 5.0,
    "recommended_ram_gb": 8.0,
    "min_vram_gb": 5.0,
    "quantization": "Q4_K_M",
    "context_length": 32768,
    "use_case": "General chat"
  }
]
```

Custom entries with the same name as a catalog model **override** it; new names are added. Optional fields (`is_moe`, `num_hidden_layers`, `gguf_sources`, …) improve estimate accuracy when provided. You can also run `llmfit update` to fetch trending models from HuggingFace without a rebuild.

### Adding a model to the built-in catalog

1. Add the model's HuggingFace repo ID (e.g., `meta-llama/Llama-3.1-8B`) to the `TARGET_MODELS` list in `scripts/scrape_hf_models.py`.
2. If the model is gated (requires HuggingFace authentication to access metadata), add a fallback entry to the `FALLBACKS` list in the same script with the parameter count and context length.
3. Run the automated update script:
   ```sh
   make update-models
   # or: ./scripts/update_models.sh
   ```
4. Verify the updated model list: `./target/release/llmfit list`
5. Update [MODELS.md](../MODELS.md) by running: `python3 << 'EOF' < scripts/...` (see commit history for the generator script)
6. Open a pull request.

See [MODELS.md](../MODELS.md) for the current list and [AGENTS.md](../AGENTS.md) for architecture details.

---
