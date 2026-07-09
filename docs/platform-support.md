# Platform Support

[← Back to README](../README.md)

## Platform support

- **Linux** -- Full support. GPU detection via `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), sysfs/`lspci` (Intel Arc) and `npu-smi` (Ascend).
- **macOS (Apple Silicon)** -- Full support. Detects unified memory via `system_profiler`. VRAM = system RAM (shared pool). Models run via Metal GPU acceleration.
- **macOS (Intel)** -- RAM and CPU detection works. Discrete GPU detection if `nvidia-smi` available.
- **Windows** -- RAM and CPU detection works. NVIDIA GPU detection via `nvidia-smi` if installed.
- **Android / Termux / PRoot** -- CPU and RAM detection usually work, but GPU autodetection is not currently supported. Mobile GPUs such as Adreno typically are not visible through the desktop/server probing interfaces llmfit uses.

### GPU support

| Vendor                 | Detection method              | VRAM reporting                 |
|------------------------|-------------------------------|--------------------------------|
| NVIDIA                 | `nvidia-smi`                  | Exact dedicated VRAM           |
| AMD                    | `rocm-smi`                    | Detected (VRAM may be unknown) |
| Intel Arc (discrete)   | sysfs (`mem_info_vram_total`) | Exact dedicated VRAM           |
| Intel Arc (integrated) | `lspci`                       | Shared system memory           |
| Apple Silicon          | `system_profiler`             | Unified memory (= system RAM)  |
| Ascend                 | `npu-smi`                     | Detected (VRAM may be unknown) |

If autodetection fails or reports incorrect values, use `--memory`, `--ram`, or `--cpu-cores` to override (see [Hardware overrides](cli.md#hardware-overrides)).

### Android / Termux note

On Android setups such as **Termux + PRoot**, llmfit usually cannot see mobile GPUs through the standard Linux detection paths (`nvidia-smi`, `rocm-smi`, DRM/sysfs, `lspci`, etc.). In those environments, "no GPU detected" is expected with the current implementation.

If you still want GPU-style recommendations on a unified-memory phone or tablet, use a manual memory override:

```sh
llmfit --memory=8G fit -n 20
llmfit recommend --json --memory=8G --limit 10
```

This is a workaround for recommendation/scoring only; it does not provide true Android GPU runtime detection.

---
