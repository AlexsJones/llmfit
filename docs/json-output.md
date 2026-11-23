# JSON output contract

Machine-readable stdout for `llmfit recommend` and `llmfit bench --json`.
GUI, dashboard, and script authors can depend on the fields, types, and
exit codes below.

[← Back to README](../README.md)

This describes the CLI documents printed by the current source
(`display_json_fits` / `display_json_fits_with_llamacpp` in
`llmfit-tui/src/display.rs`, and the `bench` printers in
`llmfit-tui/src/main.rs`). It is not the REST or MCP document. `llmfit serve`
and the MCP tools emit machine codes for `fit_level`, `run_mode`, `runtime`,
and `capabilities`. The CLI overlays display strings on those four keys.
See [CLI display strings](#cli-display-strings).

`fit --json` and `info --json` reuse the same `system` object and the same
model object as `recommend`. Their filters and envelopes differ; only the
shared fields are covered here.

## Stability

Neither document has a `version` or `schema_version` field. A frontend cannot
tell a contract change from the payload.

Until that field exists, the rules are:

- The keys, JSON types, and enumerated strings in this file are the contract.
  Renaming a key, changing its type, or changing an enumerated string is a
  breaking change and needs a note here plus a release note.
- New keys may appear. Ignore keys you do not recognize.
- Omitted keys are absent, not `null`, unless the table says the value is
  `null`. Do not require a key that this file marks as optional.
- Rounded numbers are part of the contract. Fields marked "1 decimal" or
  "2 decimals" are rounded half away from zero (`(v * 10^n).round() / 10^n`)
  before printing. Do not expect more precision than that.
- stdout on success is one pretty-printed JSON document (two-space indent)
  plus a trailing newline. Decode it with a JSON parser. Do not split on
  lines or assume minified JSON.
- stderr is not part of the document. It carries warnings, cluster banners,
  the local-benchmark save hint, and most errors. Do not parse it as JSON.
- `--share` can print extra human text on stdout after a bench document.
  See [Share and stdout](#share-and-stdout).
- `--tui` launches the interactive UI and prints neither document.

There is no machine-readable way to pin this contract to a binary yet. Record
the `llmfit --version` string next to anything you persist.

## Exit codes shared by both commands

| Code | When |
| --- | --- |
| 0 | Success. stdout is the JSON document described below. An empty `models` or `results` array is still success. |
| 1 | The command failed. See each command for whether stdout is a JSON error document or empty. |
| 2 | clap rejected the command line (unknown flag, missing value, conflicting flags such as `bench --all --tool-calls`, or a sizing-flag conflict). stderr is clap's usage error, not JSON. |

## `llmfit recommend`

JSON is the default, and this subcommand has no switch that turns it off.
`llmfit recommend` and `llmfit recommend --json` print this document.
`--json` takes no value, so `recommend --json true` and
`recommend --json false` are clap errors (exit 2). `--csv` prints CSV
instead of this document. `--output-llamacpp` adds one optional key; it
does not change the envelope.

```sh
llmfit recommend --limit 5
llmfit recommend --use-case coding --min-fit good
llmfit recommend --output-llamacpp
```

The array is the top `limit` models (default 5) after ranking. Default
`--min-fit marginal` drops `Too Tight` rows, and `too_tight` is not a legal
`--min-fit` value, so `recommend` does not return models that do not fit.
Backend-incompatible and sanitization-demoted catalog rows are removed before
ranking. On machines that are not Apple Silicon unified memory, MLX-only
models are removed. An unknown `--capability` token matches nothing and
yields an empty `models` array with exit code 0. Unknown `--use-case`,
`--min-fit`, `--runtime`, and `--force-runtime` values are errors (exit 1).

### Success document

```json
{
  "system": {},
  "models": []
}
```

### `system`

Built by `system_json`. Memory figures in this object are binary GiB
(bytes divided by 1024³), rounded to 2 decimals. The keys keep their
`_gb` suffix. A model's `memory_available_gb` is that same unit.
`disk_size_gb` is decimal GB; see the model table.

| Field | Type | Meaning |
| --- | --- | --- |
| `total_ram_gb` | number | Installed system RAM. |
| `available_ram_gb` | number | RAM llmfit treats as usable. |
| `cpu_cores` | number | Logical CPU cores (`total_cpu_cores`). Integer. |
| `cpu_name` | string | Detected CPU name. |
| `has_gpu` | boolean | A GPU was detected. |
| `gpu_vram_gb` | number or null | Per-card VRAM of the primary group (highest per-card capacity). Not the sum across cards. On unified memory this is the shared pool. |
| `gpu_available_gb` | number or null | VRAM llmfit treats as usable right now. |
| `gpu_name` | string or null | Primary GPU name. |
| `gpu_count` | number | GPU count. Integer. |
| `unified_memory` | boolean | CPU and GPU share one memory pool. |
| `backend` | string | Primary backend label. One of `CUDA`, `Metal`, `ROCm`, `Vulkan`, `SYCL`, `CPU (ARM)`, `CPU (x86)`, `NPU (Ascend)`. |
| `gpus` | array | One object per detected GPU group. |

Each `gpus[]` object:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | GPU model name. |
| `vram_gb` | number or null | Per-card VRAM, 2 decimals. Two identical 24 GiB cards report about 24 here with `count` 2. Multiply by `count` for the group total. `free_vram_gb` is already summed across the group. On unified memory this is the shared pool. |
| `backend` | string | Same labels as `system.backend`. |
| `count` | number | Cards in this group. Integer. |
| `unified_memory` | boolean | This GPU uses the system pool. |
| `free_vram_gb` | number or null | Free VRAM summed across the group, 2 decimals. `null` when the backend does not report it (Metal, Intel, Windows, name-only fallbacks, unified memory). |
| `memory_bandwidth_gbps` | number or null | Table lookup for `name`. `null` when the name is unknown or is a mobile part llmfit refuses to guess. Not passed through the 2-decimal helper. |
| `pcie_links` | array | Negotiated PCIe links. `[]` when none were detected. Always present. |

Each `pcie_links[]` object: `index` (number or null, NVIDIA card index), `generation` (number or null), `width` (number or null), `max_generation` (number or null), `max_width` (number or null). Integers when present.

`system` does not include an OS field, a hostname, cluster membership, or
the summed VRAM pool. That pool is the sum of each `gpus[].vram_gb`
multiplied by that group's `count`. `gpu_vram_gb` is only the primary
group's per-card figure. A model's `memory_available_gb` is the pool the
fit scored against. Cluster mode still fills these same keys from the
aggregated specs. A cluster banner, when printed, goes to stderr.

### `models[]`

One object per recommendation, best rank first. `name` is the catalog id
(often a Hugging Face repo id).

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Catalog model id. |
| `provider` | string | Catalog publisher. |
| `parameter_count` | string | Display size such as `8B`. Not a number. |
| `params_b` | number | Parameter count in billions, 2 decimals. |
| `context_length` | number | Native context window, in tokens. Integer. |
| `usable_context` | number | Context that fits in the memory pool. Integer. `0` when no context fits. |
| `effective_context_length` | number | Context used for the memory estimate. Integer. |
| `use_case` | string | Catalog prose, for example `General purpose text generation`. Not an enum. |
| `category` | string | Inferred bucket. One of `General`, `Coding`, `Reasoning`, `Chat`, `Multimodal`, `Embedding`. This is what `--use-case` filters on. |
| `release_date` | string or null | Catalog date string, or `null`. |
| `is_moe` | boolean | Mixture-of-experts model. |
| `fit_level` | string | CLI display string. See [CLI display strings](#cli-display-strings). |
| `fit_label` | string | Same display string as `fit_level` on this CLI document. |
| `run_mode` | string | CLI display string. |
| `run_mode_label` | string | Same display string as `run_mode` on this CLI document. |
| `score` | number | Weighted score, 0–100, 1 decimal. |
| `score_components` | object | `quality`, `speed`, `fit`, `context`. Each a number, 0–100, 1 decimal. |
| `estimated_tps` | number | Formula or calibrated generation tok/s, 1 decimal. `0.0` when llmfit has no speed basis. |
| `runtime` | string | CLI display string. |
| `runtime_label` | string | Same display string as `runtime` on this CLI document. |
| `best_quant` | string or null | Quantization chosen for this machine. `null` when a GGUF label was cleared because the repo name is a native NVFP4/MXFP4 model. |
| `quantization` | string | Catalog quantization label. |
| `format` | string | Weight format: `gguf`, `awq`, `gptq`, `autoround`, `mlx`, `safetensors`, or `onnx`. |
| `memory_required_gb` | number | Memory the chosen run mode needs, 2 decimals. The formula depends on `run_mode`. See below. |
| `memory_available_gb` | number | Pool it is scored against, 2 decimals. Binary GiB. |
| `moe_offloaded_gb` | number or null | Inactive experts left in RAM, 2 decimals. `null` except on `MoE`. On `MoE`, inactive-expert bytes / 1024³. |
| `total_memory_gb` | number | `memory_required_gb` plus `moe_offloaded_gb` (0 when null), 2 decimals. |
| `utilization_pct` | number | `memory_required / memory_available * 100`, 1 decimal. |
| `notes` | array of strings | Fit notes. May include a sentence explaining a cleared `best_quant`. |
| `gguf_sources` | array | `{ "repo": string, "provider": string }` Hugging Face GGUF repos. `[]` when none are known. |
| `capabilities` | array of strings | CLI display labels. See below. |
| `capability_ids` | array of strings | Stable codes: `vision`, `tool_use`, `audio`, `tts`. |
| `license` | string or null | Catalog license text. Several ids are comma-separated. `null` when the catalog has none. |
| `supports_tp` | array of numbers | Tensor-parallel sizes in `1..=8` the model can split across. Integers. |
| `installed` | boolean | A local provider already has this model. |
| `disk_size_gb` | number | Estimated weight storage at `best_quant`, decimal GB, 2 decimals. `params_b` (parameter count / 1e9, or the catalog size such as `7B`) times bytes per parameter. The printer only rounds that product. It is not divided by 1024³. Weights only: no KV cache, runtime overhead, or download scratch. |
| `ollama_name` | string or null | Ollama pull tag, or `null` when llmfit has no mapping. |
| `estimate_basis` | object | Inputs behind `estimated_tps`. See below. |
| `estimate_confidence` | string | Stable code: `measured_local`, `measured_community`, `calibrated`, `estimated`, `unsupported`. |
| `estimate_confidence_label` | string | Display label: `measured (this machine)`, `measured (community)`, `calibrated`, `estimated`, `unsupported — no basis`. |
| `verify_command` | string or null | A `llama-bench` invocation for a llama.cpp GPU or CPU-only fit. `null` otherwise. The `-m` path is a placeholder. |
| `measured_tps` | object or null | A real measurement when one matched. `null` otherwise. |
| `prefill_tps` | number or null | Estimated prompt-processing tok/s, 1 decimal. |
| `ttft_ms` | number or null | Estimated time to first token at `effective_context_length`, 1 decimal. |
| `tp_size` | number or null | Tensor-parallel size chosen for a cluster layout. Integer when present. |
| `layout_refusal` | object or null | Why a cluster split was refused. `null` on a normal single-machine fit. |
| `runnable` | boolean | Backend is compatible, the fit is not Too Tight, and the runtime is not `unsupported`. |
| `llamacpp_command` | string, optional | Present only with `--output-llamacpp`, and only when the runtime is llama.cpp, a GGUF source exists, and the run mode is not tensor-parallel. Absent otherwise. Not `null`. |

For `GPU`, `CPU+GPU`, `CPU`, and `TP`, `memory_required_gb` is
`estimate_memory_gb` at the chosen quant: the same decimal-GB weight product
as `disk_size_gb`, plus KV cache, plus 0.5. The precise KV term is
bytes/1024³. For `MoE`, `memory_required_gb` is active-expert bytes / 1024³,
times 1.1, and at least 0.5. That path does not use the full-model decimal
weight product, does not add KV cache, and does not add another 0.5.
Inactive experts are `moe_offloaded_gb`. A MoE model that cannot offload,
and a MoE model scored CPU-only, reports the catalog `min_ram_gb` or
`min_vram_gb` instead.

`estimate_basis`:

| Field | Type | Meaning |
| --- | --- | --- |
| `method` | string | `gpu_bandwidth_roofline`, `gpu_ddr_offload_roofline`, `backend_constant`, `cpu_constant`, or `unsupported`. |
| `gpu_bandwidth_gbps` | number or null | GPU bandwidth assumed by a roofline method. |
| `ddr_bandwidth_gbps` | number or null | System-RAM bandwidth when the estimate reads DDR. |
| `efficiency` | number | Factor applied to raw bandwidth. Not pre-rounded. |
| `assumed_context` | number | Context length the generation estimate models. Integer. |
| `local_calibration` | number or null | Scale already applied to `estimated_tps` from local or llmfit-community runs. `null` when none applied. |

`measured_tps`, when not null:

| Field | Type | Meaning |
| --- | --- | --- |
| `tok_s` | number | Measured generation tok/s. Not pre-rounded. |
| `sample_count` | number | Runs behind `tok_s`. Integer. |
| `hardware_label` | string | Hardware those runs were matched to. |
| `source` | string | `local_bench`, `community_llmfit`, or `community`. |
| `match_grade` | string, optional | `exact_pcie` or `legacy`. Omitted (not `null`) when there is no PCIe grade. |

`layout_refusal`, when not null:

| Field | Type | Meaning |
| --- | --- | --- |
| `reason` | string | `attention_heads_not_divisible` or `kv_heads_not_divisible`. |
| `tp_size` | number | Refused tensor-parallel size. Integer. |
| `requires` | string | `pipeline_parallel`. |
| `message` | string | Same explanation as the human note. |

### CLI display strings

On this CLI document the following keys are display text. The REST API and
MCP tools use the machine codes instead, and keep the display text under
`fit_label`, `run_mode_label`, `runtime_label`, and capability labels. The
CLI copies are load-bearing for existing scripts and stay as they are until
an explicit deprecation.

| Key | CLI value | Machine code (not printed on this CLI key) |
| --- | --- | --- |
| `fit_level`, `fit_label` | `Perfect`, `Good`, `Marginal`, `Too Tight` | `perfect`, `good`, `marginal`, `too_tight` |
| `run_mode`, `run_mode_label` | `GPU`, `TP`, `MoE`, `CPU+GPU`, `CPU` | `gpu`, `tensor_parallel`, `moe_offload`, `cpu_offload`, `cpu_only` |
| `runtime`, `runtime_label` | `llama.cpp`, `MLX`, `vLLM`, `bitnet.cpp`, `unsupported` | `llamacpp`, `mlx`, `vllm`, `bitnetcpp`, `unsupported` |
| `capabilities[]` | `Vision`, `Tool Use`, `Audio`, `Text-to-Speech` | `vision`, `tool_use`, `audio`, `tts` in `capability_ids` |

`Too Tight` contains a space. `recommend` does not emit that row today
because of `--min-fit`. `fit --json --show-incompatible` can emit
`runtime: "unsupported"`.

Use `capability_ids` and `estimate_confidence` when you need a stable code.
Do not switch on `fit_level` with the API's `too_tight` spelling.

### Exit codes

| Code | stdout | When |
| --- | --- | --- |
| 0 | Success document | Recommendations computed, including `models: []`. |
| 1 | Empty | Unknown `--use-case`, `--min-fit`, `--runtime`, or `--force-runtime`. stderr starts with `Unknown ...`. |
| 1 | Empty | `--profile` combined with `--force-runtime`. stderr is `Error: --profile cannot be combined with --force-runtime yet`. |
| 1 | Empty | `--profile` does not resolve. stderr is `Error: ...`. |
| 1 | JSON error document | `--cluster` was refused. `recommend` always prints that refusal as JSON. |
| 2 | Empty | clap usage error. |

The JSON error document is the only structured failure `recommend` prints:

```json
{
  "error": {
    "kind": "cluster_not_configured",
    "message": "One sentence, with the hint appended."
  }
}
```

`kind` is one of `cluster_no_gpus`, `cluster_mixed_backends`,
`cluster_vram_unset`, `cluster_not_configured`, `cluster_config_invalid`.
`message` is the problem and the hint joined into one string. `recommend`
prints this document for a cluster refusal. It does not print the
plain-text form.

Other failures (bad filters, a bad profile) are not wrapped in this object.
Check the exit code before decoding stdout.

## `llmfit bench --json`

`--json` is off by default. `llmfit bench` with no model and no action flag
opens the benchmark TUI and prints no document.

`--json` selects exactly one document. The flags are checked in this order:

| Flags | stdout |
| --- | --- |
| `--tool-calls` | One tool-probe object. No wrapper. Cannot be combined with `--all` (clap, exit 2). |
| `--quality` or `--routing` | Quality document. `--routing` adds `routing` and `runner_ups`. This does not also run the throughput benchmark. |
| anything else (`--json`, optionally a model, `--all`, `--provider`, `--url`, `--runs`) | Throughput document. |

```sh
llmfit bench --json some-model
llmfit bench --json --all
llmfit bench --json --provider lmstudio some-model
llmfit bench --quality --json --routing
llmfit bench --tool-calls --json some-model
```

`--provider` accepts `ollama`, `vllm`, `ferrum`, `lmstudio` (alias
`lm-studio`), `mlx`, and `llamacpp` (aliases `llama.cpp` and `llama-server`).
The default `auto`, and any other string, auto-detects a running server.
A typo does not fail the parse.

`--runs` defaults to 3. One warmup request is not included in `runs` or
`summary`. Counted prompts cycle through a fixed set of four prompts.

### Throughput document

A single target:

```json
{ "result": {} }
```

`--all`:

```json
{ "results": [] }
```

`result` and each `results[]` entry:

| Field | Type | Meaning |
| --- | --- | --- |
| `model` | string | Model id sent to the provider. Not necessarily a catalog id. |
| `provider` | string | `ollama`, `vllm`, `ferrum`, `mlx`, `llamacpp`, or `lmstudio`. |
| `runs` | array | Counted runs, in order. |
| `summary` | object | Aggregates over `runs`. |

`runs[]`:

| Field | Type | Meaning |
| --- | --- | --- |
| `ttft_ms` | number or null | Ollama: `prompt_eval_duration` converted from nanoseconds to milliseconds. OpenAI-compatible providers (vLLM, Ferrum, MLX, llama.cpp, LM Studio): always `null`. Streaming is not used. |
| `tps` | number | Output tokens per second for that run. Not pre-rounded. May exceed the plausible range; see `summary`. |
| `total_ms` | number | Ollama prefers native `total_duration`; otherwise wall time. OpenAI-compatible providers use wall time after the body is read. |
| `prompt_tokens` | number | Prompt tokens. Integer. |
| `output_tokens` | number | Generated tokens. Integer. |

`summary`:

| Field | Type | Meaning |
| --- | --- | --- |
| `num_runs` | number | `runs.length`. Integer. |
| `avg_ttft_ms` | number or null | Mean of the runs that reported `ttft_ms`. `null` when none did. |
| `avg_tps` | number | Mean of runs whose `tps` is finite, `> 0`, and `<= 10000`. `0.0` when none qualify. |
| `min_tps` | number | Minimum plausible `tps`, or `0.0`. |
| `max_tps` | number | Maximum plausible `tps`, or `0.0`. |
| `avg_total_ms` | number | Mean of `total_ms`, including implausible runs. |
| `avg_output_tokens` | number | Mean of `output_tokens`. |

Implausible per-run `tps` values stay in `runs`. They are left out of
`avg_tps`, `min_tps`, and `max_tps` only.

With `--all`, a target that fails is omitted. Its error is not printed when
`--json` is set. If every target fails, stdout is `{ "results": [] }` and
the exit code is still 0. If no provider is running at all, there is no
document and the exit code is 1.

### Quality document

```json
{ "quality_results": [] }
```

With `--routing`:

```json
{
  "quality_results": [],
  "routing": [],
  "runner_ups": []
}
```

`quality_results[]`:

| Field | Type | Meaning |
| --- | --- | --- |
| `model` | string | Model id that was served. |
| `provider` | string | `ollama` for Ollama. For every other provider, the display name: `vLLM`, `Ferrum`, `MLX`, `llama.cpp`, or `LM Studio`. These are not the throughput slugs. |
| `roles` | array | One aggregate per role that ran. |
| `test_results` | array | One object per test. |
| `overall_quality` | number | Mean of `roles[].quality`. |
| `overall_speed` | number | Mean of `roles[].speed`. |
| `overall_composite` | number | Mean of `roles[].composite`. |

`roles[]`: `role` (string), `quality` (number, 0–10, 1 decimal), `speed`
(number, tok/s, 1 decimal), `composite` (number, 1 decimal), `test_count`
(integer).

`test_results[]`:

| Field | Type | Meaning |
| --- | --- | --- |
| `test_name` | string | Test name from the quality config. |
| `role` | string | Role name. Defaults are the built-in Amplifier roles. `--roles` and `--quality-config` change the set. |
| `quality` | number | Rubric score, 0–10. |
| `tok_per_sec` | number | Output tok/s for that test. |
| `composite` | number | Blend of quality and speed. Not pre-rounded. |
| `response_preview` | string | First 150 characters of the reply. Empty when the test errored. |
| `ttft_ms` | number or null | Time to first token when the endpoint reported it. |
| `wall_time_sec` | number | Wall time in seconds. |
| `eval_tokens` | number | Output tokens. Integer. |
| `error` | string or null | Failure for this test. A failed test is still a result; it does not fail the process by itself. |

`routing[]` is the best model per role: `role`, `model`, `quality`, `speed`,
`composite`. `note` is omitted.

`runner_ups[]` is the second-best model per role, with the same fields plus
`note`. Every runner-up has a note: `consider for speed`, `close quality`,
`competitive composite`, or `alternative`. Roles with only one model produce
no runner-up.

### Tool-probe document

`llmfit bench --tool-calls --json` prints the probe object itself, with no
`result` wrapper:

| Field | Type | Meaning |
| --- | --- | --- |
| `model` | string | Model id that was probed. |
| `provider` | string | Display name: `Ollama`, `vLLM`, `Ferrum`, `MLX`, `llama.cpp`, or `LM Studio`. |
| `attempts` | array of strings | One entry per counted attempt, in order: `called`, `no_tool_call`, or `malformed`. |
| `passed` | number | How many attempts were `called`. Integer. |
| `verdict` | string | `ok` (every attempt called), `flaky` (some did), or `broken` (none did). |

The probe offers one `write` tool. A request that cannot be completed is an
error (exit 1), not an attempt. `--runs` is the attempt count and must be at
least 1.

### Share and stdout

Do not pass `--share` when you need a single JSON value on stdout.

- `llmfit bench --json --share` with no model and no `--all` skips the
  benchmark and prints human text only. `--json` does not apply to that path.
- `llmfit bench --json <model> --share` and `llmfit bench --json --all --share`
  print the throughput document first, then human lines such as
  `Pull request opened: ...` on stdout.
- `--dry-run` writes the submission payload to stderr and does not add those
  stdout lines. The function then returns without a PR.
- A share that fails after a successful measurement still leaves the JSON
  document on stdout, then exits 1.

Without `--share`, a successful throughput run also appends a one-line save
hint to stderr. That hint is not JSON and does not affect the exit code.

### Exit codes

| Code | stdout | When |
| --- | --- | --- |
| 0 | Throughput, quality, or tool-probe document | The selected benchmark produced a document. `--all` with some or all targets failing still exits 0, with only the successes in `results`. |
| 1 | Empty | Single-target throughput failed, including a warmup failure. stderr is `Error: ...`. |
| 1 | Empty | `--all` found no running provider. stderr names the providers to start. |
| 1 | Empty | Provider resolution failed (no model for a provider that requires one, endpoint identity mismatch, unreachable detector). stderr is `Error: ...`. |
| 1 | Empty | Quality benchmarking collected no results, or `--roles` matched nothing, or the quality config could not be read. stderr is plain text. |
| 1 | Empty | The tool probe could not complete a request, or `--runs` was 0. |
| 1 | Empty, or a throughput document already printed | `--share` could not authenticate, or the upload failed. |
| 2 | Empty | clap usage error. |

`bench --json` does not print the `{"error":{"kind","message"}}` document.
A non-zero exit with empty stdout is the failure signal, except for the
share-after-success case above.
