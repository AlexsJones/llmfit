# Hardware profiles

A hardware profile is a named, versioned description of a machine: how much
memory it has, whether that memory is unified, and the bandwidth/compute
figures the throughput estimator needs. Pass one with `--profile` to score
models against that machine instead of the detected one:

```sh
llmfit --profile ryzen-ai-max-plus-395 fit -n 10
llmfit --profile ./my-workstation.json recommend --json
llmfit hardware list
llmfit hardware show nvidia-rtx-4090
```

`--profile` replaces the `--memory` / `--ram` / `--cpu-cores` overrides (it
conflicts with all three) because a profile describes a whole machine rather
than one field.

## Layout

```
hardware/
  schema.json          JSON Schema (draft-07) every profile must satisfy
  <name>.json          one profile; `name` must equal the file stem
```

Profiles in this directory are aggregated by `llmfit-core/build.rs` and
**embedded into the binary**, so a merged profile ships in the next release
and is available by name with no download. Users can add their own without
rebuilding by dropping files into the directory printed by:

```sh
llmfit hardware path
```

A user profile whose `name` matches an embedded one takes precedence.
`LLMFIT_HARDWARE_PROFILES` overrides that directory.

## Fields

| Field | Required | Applied to |
| --- | --- | --- |
| `schema_version` | yes | must be `1` |
| `name` | yes | must equal the file stem; `[a-z0-9][a-z0-9._-]*` |
| `match.gpu_name_contains` | no | provenance only — llmfit never auto-selects a profile |
| `hardware.total_ram_gb` | yes | `SystemSpecs` capacity (and VRAM when unified) |
| `hardware.unified_memory` | yes | `SystemSpecs::unified_memory` |
| `hardware.gpu_memory_bandwidth_gbps` | no | `CalcConfig::gpu_bandwidth_gbps_override` |
| `hardware.ddr_bandwidth_gbps` | no | `CalcConfig::ddr_bandwidth_gbps` |
| `hardware.gpu_compute_tflops_fp16` | no | `CalcConfig::gpu_compute_tflops_fp16` (prefill/TTFT) |
| `estimation.efficiency` | no | `CalcConfig::efficiency` |
| `estimation.run_mode_factors.*` | no | `CalcConfig::run_mode_factors` (per-mode, unset keys keep defaults) |
| `calibration[]` | no | **parsed and validated, not yet applied** in `schema_version` 1 |

`calibration` records measured anchors with their provenance so they can be
reviewed now and used by a later schema version; it changes no estimate today.

A discrete-GPU profile does not carry a VRAM figure, so VRAM stays as
detected on the host. Unified profiles are fully described: VRAM tracks
`total_ram_gb`.

## Format

Each file conforms to [`schema.json`](./schema.json). Example:

```json
{
  "schema_version": 1,
  "name": "example-unified-256",
  "match": { "gpu_name_contains": "Radeon 8060S" },
  "hardware": {
    "total_ram_gb": 128.0,
    "unified_memory": true,
    "gpu_memory_bandwidth_gbps": 256.0,
    "ddr_bandwidth_gbps": 256.0,
    "gpu_compute_tflops_fp16": 29.7
  },
  "estimation": {
    "efficiency": 0.6,
    "run_mode_factors": { "cpu_only": 0.25 }
  },
  "calibration": [
    {
      "model": "openai/gpt-oss-120b",
      "quant": "MXFP4",
      "run_mode": "gpu",
      "measured_tps": 50.0,
      "source": "https://github.com/AlexsJones/llmfit/issues/969"
    }
  ]
}
```

Unknown keys are **tolerated when loading** (so a profile written for a newer
llmfit still works) but **rejected by validation**, which is what catches a
typo before it silently does nothing:

```sh
llmfit hardware validate ./my-workstation.json
```

## Bundled profiles

| Name | Memory | Bandwidth | Notes |
| --- | --- | --- | --- |
| `ryzen-ai-max-plus-395` | 128 GB unified | 256 GB/s | Strix Halo APU: 256-bit LPDDR5X-8000. Radeon 8060S, 40 RDNA 3.5 CUs @ ~2.9 GHz → ~29.7 TFLOP/s packed fp16. |
| `apple-m3-max-128gb` | 128 GB unified | 400 GB/s | 40-core-GPU M3 Max. fp16 matmul throughput is left unset, so prefill/TTFT report `null` rather than a guess. |
| `nvidia-rtx-4090` | 64 GB system RAM | 1008 GB/s | Discrete Ada card: 165.2 TFLOP/s dense fp16 (tensor, fp32 accumulate). `ddr_bandwidth_gbps` is dual-channel DDR5-5600. |

## Validation

`cargo test -p llmfit-core` validates every file here against
[`schema.json`](./schema.json) and checks that each `name` matches its file
stem, so a malformed contribution fails CI rather than shipping as a profile
that cannot be selected.
