# Cluster mode

Check which models fit across several machines before you launch anything: a homelab of DGX Sparks, or a few multi-GPU servers running Ray and vLLM.

[← Back to README](../README.md) · [CLI & automation](cli.md) · [How it works](how-it-works.md)

> **Preview.** Cluster mode is new. If you run Ray, please help by [opening an issue](https://github.com/AlexsJones/llmfit/issues/new) with the output of `curl <dashboard>/nodes?view=summary` (add `-H "Authorization: Bearer $RAY_AUTH_TOKEN"` if your dashboard needs a token, and redact hostnames or IPs you'd rather not share).

## Why this is useful

- **Know before you launch.** Find out whether a model fits across your machines before you spend an hour starting vLLM on them.
- **Catch splits vLLM would refuse.** Most models can't be split 3 ways. On 3 × DGX Spark, llmfit tells you how many GPUs each model can actually use, and which models need a layout tensor parallelism can't provide, instead of you finding out from a vLLM startup crash.
- **One file you can review.** Ray discovery writes a single `cluster.toml` with every node's GPUs, VRAM, RAM and cores, so you don't have to tally them up by hand.
- **Works with what you already run, and changes nothing.** llmfit reads Ray's dashboard once, at setup, and plans for vLLM. It never launches, schedules or changes anything on the cluster.

## Contents

1. [Homelab: 3 × DGX Spark discovered through Ray](#1-homelab-3--dgx-spark-discovered-through-ray)
2. [A hand-written cluster.toml, without Ray](#2-a-hand-written-clustertoml-without-ray)
3. [Switching cluster mode on and off](#3-switching-cluster-mode-on-and-off)
4. [Reference](#reference)
5. [Limits today](#limits-today)

> **About the output below.** Every block was captured from a real `llmfit` build. Ray discovery was run against a mock Ray Dashboard that serves a `/nodes?view=summary` response in Ray's format and requires a token, so the hostnames, IPs and the `120 GB` VRAM value are example inputs, not measurements. Config paths are shortened to `~`. The tok/s lines are llmfit's own estimates, not benchmarks. Long output is trimmed where marked with `…`.

---

## 1. Homelab: 3 × DGX Spark discovered through Ray

### Discover the cluster

**What you run.** Point `cluster init` at the Ray Dashboard (port 8265 by default). If your cluster uses Ray token authentication, llmfit picks the token up the same way Ray does: `RAY_AUTH_TOKEN`, then the file in `RAY_AUTH_TOKEN_PATH`, then `~/.ray/auth_token`.

```
$ llmfit cluster init --ray-url http://172.30.0.2:18265
⚠️  Ray Dashboard at http://172.30.0.2:18265 uses plain HTTP to a non-local host. The dashboard accepts job submissions and any Ray auth token is sent unencrypted; prefer https://, an SSH tunnel (`ray dashboard`) or a private network.
Error: could not initialize cluster from --ray-url 'http://172.30.0.2:18265': Ray API request to http://172.30.0.2:18265/nodes?view=summary was rejected (401): the dashboard requires a token; set RAY_AUTH_TOKEN or RAY_AUTH_TOKEN_PATH, or run `ray get-auth-token`
```

The dashboard needs a token, so provide one and run it again:

```
$ export RAY_AUTH_TOKEN_PATH=~/.ray/auth_token   # or set RAY_AUTH_TOKEN directly
$ llmfit cluster init --ray-url http://172.30.0.2:18265
⚠️  Ray Dashboard at http://172.30.0.2:18265 uses plain HTTP to a non-local host. …
Cluster configured: 3 nodes, 0 GB total VRAM
⚠️  Ray did not report per-GPU VRAM for some GPU nodes; the saved config has gpu_vram_gb = 0 for them.
   Pass --memory <per-GPU VRAM> with `--cluster`, set gpu_vram_gb per node in the cluster config, or re-run `llmfit cluster init` interactively.
```

`ray get-auth-token` prints the token Ray uses on the machine you run it on (see Ray's [token authentication guide](https://docs.ray.io/en/latest/ray-security/token-auth.html)), so run it where the cluster's token lives. The token is only sent in the request's `Authorization` header. It is never written to `cluster.toml` or printed. The plain-HTTP warning appears whenever discovery uses `http://` to anything other than localhost.

**What you see.** A `cluster.toml` in your config directory (`~/.config/llmfit/cluster.toml` on Linux):

```toml
version = 1
name = "3-node cluster"
head_ip = "172.30.0.2"
ray_port = 18265

[[nodes]]
hostname = "spark-1"
ip = "172.30.0.2"
gpu_name = "NVIDIA GB10"
gpu_vram_gb = 0.0
total_ram_gb = 128.0
cpu_cores = 20
gpu_count = 1
unified_memory = false
is_head = true

[[nodes]]
hostname = "spark-2"
# … spark-2 and spark-3 look the same
```

Ray reports each node's GPU count, CPU cores, RAM and, from its per-GPU stats, the GPU model. When a card also reports its total memory (most discrete NVIDIA GPUs do), llmfit fills in `gpu_vram_gb` too. In this mock the GB10s report a name but no memory total, as a unified-memory part may, so VRAM is left at 0 for you to fill in.

**How it helps.** You get one file listing what Ray sees, and you can correct it before trusting any result. For the Sparks, set the VRAM you want vLLM to use per GPU and mark the memory as unified:

```toml
gpu_vram_gb = 120.0        # your choice per GPU; example value
unified_memory = true
backend = "cuda"           # silences the "unspecified backend" warning
```

```
$ llmfit cluster status

  Cluster: 3-node cluster
  Nodes:   3

    spark-1 (HEAD) — 172.30.0.2 | NVIDIA GB10 | 1× GPU | 120 GB VRAM | 128 GB RAM | 20 cores
    spark-2 (WORKER) — 172.30.0.12 | NVIDIA GB10 | 1× GPU | 120 GB VRAM | 128 GB RAM | 20 cores
    spark-3 (WORKER) — 172.30.0.13 | NVIDIA GB10 | 1× GPU | 120 GB VRAM | 128 GB RAM | 20 cores

  Totals:
    GPUs:     3
    VRAM:     360 GB
    RAM:      384 GB
    CPUs:     60 cores
```

### Check what fits

**What you run.** Add `--cluster` to `fit`, `info` or the other analysis commands. To look at one model, use `info`.

**What you see.** For each model, llmfit picks the largest tensor-parallel size vLLM will accept whose shards fit: TP=3 across all three Sparks if it can, otherwise TP=2 or TP=1. vLLM shards attention heads evenly across ranks, so 64 heads can't be split 3 ways. Qwen3-32B therefore runs on 2 of the 3 GPUs:

```
$ llmfit --cluster info Qwen/Qwen3-32B
📡 Using cluster config: 3 nodes, 360 GB total VRAM
…
Fit Analysis:
  Status: 🟢 Perfect
  Run Mode: TP
  Memory Utilization: 9.0% (21.5 / 240.0 GB)
Notes:
  Context capped at 8192 tokens for estimation (model supports up to 40960; use --max-context to override)
  Cluster: tensor-parallel via vLLM (TP=2), one shard per GPU on 2 of 3 GPUs; TP=3 invalid for this model (64 attention heads not divisible by 3)
…
```

A model whose heads do split 3 ways uses every GPU:

```
$ llmfit --cluster info zai-org/glm-edge-4b-chat
…
Notes:
  Cluster: tensor-parallel via vLLM (TP=3), one shard per GPU across 3 GPUs on 3 nodes
…
```

A model that only fits once it's spread over all three Sparks, but can't be split 3 ways, is refused, with the reason:

```
$ llmfit --cluster info Qwen/Qwen3-Coder-480B-A35B-Instruct
📡 Using cluster config: 3 nodes, 360 GB total VRAM
…
  Baseline Est. Speed: - (layout refused)
…
Fit Analysis:
  Status: 🔴 Too Tight
  Run Mode: TP
…
Notes:
  Context capped at 8192 tokens for estimation (model supports up to 262144; use --max-context to override)
  TP=3 invalid for this model (8 KV heads neither divisible by nor a divisor of 3) and no smaller valid split fits; pipeline-parallel layout needed (not yet modelled)
…
```

With `--json`, every cluster fit carries `tp_size` (the split it uses), and a refused one carries a `layout_refusal` you can script against instead of parsing the note. `reason` is `attention_heads_not_divisible` or `kv_heads_not_divisible`. A refused layout has no speed: the `fit` table and `info` show `-` instead of a tok/s figure, no measured tok/s is attached, and in JSON `estimated_tps` stays a number, `0.0`, with `measured_tps` set to `null`:

```
$ llmfit --cluster --json info Qwen/Qwen3-Coder-480B-A35B-Instruct | jq '.models[0] | {name, fit_level, tp_size, layout_refusal}'
{
  "name": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
  "fit_level": "Too Tight",
  "tp_size": null,
  "layout_refusal": {
    "message": "TP=3 invalid for this model (8 KV heads neither divisible by nor a divisor of 3) and no smaller valid split fits; pipeline-parallel layout needed (not yet modelled)",
    "reason": "kv_heads_not_divisible",
    "requires": "pipeline_parallel",
    "tp_size": 3
  }
}
$ llmfit --cluster --json info Qwen/Qwen3-32B | jq '.models[0] | {name, fit_level, tp_size, layout_refusal}'
{
  "name": "Qwen/Qwen3-32B",
  "fit_level": "Perfect",
  "tp_size": 2,
  "layout_refusal": null
}
```

Some catalog entries have no head counts. If such a model fits on one GPU, that single-GPU fit is still valid, and the note says the larger split wasn't checked. A model on one GPU isn't sharded, so it's shown as a plain GPU run (`tp_size` is `1` in JSON) and isn't given the tensor-parallel speed penalty:

```
$ llmfit --cluster info huihui-ai/DeepSeek-R1-0528-Qwen3-8B-abliterated
…
Fit Analysis:
  Status: 🟢 Perfect
  Run Mode: GPU
  Memory Utilization: 4.6% (5.5 / 120.0 GB)
Notes:
  Cluster: runs on a single GPU via vLLM (no tensor parallelism), 1 of 3 GPUs; TP=3 unverified: attention-head counts unknown for this model
…
```

If it needs every GPU, llmfit keeps the memory verdict and adds `TP=3 layout unverified: attention-head counts unknown for this model, so vLLM may refuse this split`.

**How it helps.** You see up front how many GPUs each model would actually use, and you learn which models need a pipeline-parallel layout before you copy weights to three machines, not from a vLLM startup error.

---

## 2. A hand-written cluster.toml, without Ray

**What you run.** No Ray, or the dashboard isn't reachable? Write the file yourself, or run `llmfit cluster init` with no flags to be prompted for the same fields. `gpu_vram_gb` is **per GPU**, and `gpu_count` is GPUs on that node:

```toml
version = 1
name = "a100-pair"
head_ip = "10.0.0.1"

[[nodes]]
hostname = "gpu-1"
ip = "10.0.0.1"
gpu_name = "NVIDIA A100-SXM4-80GB"
gpu_vram_gb = 80.0
gpu_count = 4
total_ram_gb = 512.0
cpu_cores = 64
backend = "cuda"
is_head = true

[[nodes]]
hostname = "gpu-2"
ip = "10.0.0.2"
gpu_name = "NVIDIA A100-SXM4-80GB"
gpu_vram_gb = 80.0
gpu_count = 4
total_ram_gb = 512.0
cpu_cores = 64
backend = "cuda"
```

```
$ llmfit cluster status

  Cluster: a100-pair
  Nodes:   2

    gpu-1 (HEAD) — 10.0.0.1 | NVIDIA A100-SXM4-80GB | 4× GPU | 80 GB VRAM | 512 GB RAM | 64 cores
    gpu-2 (WORKER) — 10.0.0.2 | NVIDIA A100-SXM4-80GB | 4× GPU | 80 GB VRAM | 512 GB RAM | 64 cores

  Totals:
    GPUs:     8
    VRAM:     640 GB
    RAM:      1024 GB
    CPUs:     128 cores
```

**What you see.** The same model that couldn't split 3 ways splits cleanly 8 ways, one shard per GPU:

```
$ llmfit --cluster info Qwen/Qwen3-32B
📡 Using cluster config: 2 nodes, 640 GB total VRAM
…
Fit Analysis:
  Status: 🟢 Perfect
  Run Mode: TP
  Memory Utilization: 3.4% (21.5 / 640.0 GB)
Notes:
  Context capped at 8192 tokens for estimation (model supports up to 40960; use --max-context to override)
  Cluster: tensor-parallel via vLLM (TP=8), one shard per GPU across 8 GPUs on 2 nodes
…
```

A model with fewer KV heads than GPUs in the split still runs, but vLLM gives every GPU a copy of a KV head, so the cluster holds more KV cache than one copy. llmfit counts the copies and says how much they add:

```
$ llmfit --cluster info Qwen/Qwen3-235B-A22B-GPTQ-Int4
📡 Using cluster config: 2 nodes, 640 GB total VRAM
…
Notes:
  Context capped at 8192 tokens for estimation (model supports up to 40960; use --max-context to override)
  Cluster: tensor-parallel via vLLM (TP=8), one shard per GPU across 8 GPUs on 2 nodes
  KV cache replicated: 4 KV heads across TP=8 ranks adds 1.5 GB
…
```

**How it helps.** Typos fail loudly instead of quietly changing the answer. A misspelt key is rejected with its line:

```
$ llmfit cluster status
Error: Invalid cluster config ~/.config/llmfit/cluster.toml: TOML parse error at line 10, column 1
   |
10 | gpu_cout = 4
   | ^^^^^^^^
unknown field `gpu_cout`, expected one of `hostname`, `ip`, `gpu_name`, `gpu_vram_gb`, `total_ram_gb`, `cpu_cores`, `gpu_count`, `unified_memory`, `is_head`, `backend`
```

A file written by a newer llmfit is refused rather than misread:

```
$ llmfit cluster status
Error: Invalid cluster config ~/.config/llmfit/cluster.toml: cluster config version 2 is newer than this llmfit supports (version 1); upgrade llmfit or re-create it with `llmfit cluster init`
```

---

## 3. Switching cluster mode on and off

Cluster mode is opt-in. A saved config never changes the results for the machine you're on until you pass `--cluster`:

```
$ llmfit fit
ℹ️  Cluster config detected. Pass --cluster to use it, or --no-cluster to suppress this message.
…
$ llmfit --no-cluster fit      # same results for this machine, no hint
$ llmfit --cluster fit         # score against the cluster
$ llmfit cluster clear
Cluster config removed.
```

---

## Reference

| Command / setting | What it does |
|---|---|
| `llmfit cluster init` | Interactive setup. It tries the Ray Dashboard over plain HTTP first and falls back to manual entry. For HTTPS, use `--ray-url https://…`. |
| `llmfit cluster init --ray-url <url>` | Non-interactive discovery from a Ray Dashboard URL (`http://`, `https://`, `host:port` or `[ipv6]:port`). The port defaults to 443 for `https://` and to 8265 otherwise. Only ALIVE nodes (or entries with no state) are counted, and nodes without an address are skipped. Writes `cluster.toml`. |
| `llmfit cluster status` | Prints the saved cluster. Add `--json` for machine-readable output. |
| `llmfit cluster clear` | Deletes the saved cluster config. |
| `--cluster` / `--no-cluster` | Score against the saved cluster, or skip it and hide the hint. |
| `--memory <size>` with `--cluster` | Overrides per-GPU VRAM for every GPU, for example when Ray reported none. |
| `RAY_AUTH_TOKEN`, `RAY_AUTH_TOKEN_PATH`, `~/.ray/auth_token` | Ray auth token, checked in that order and sent as `Authorization: Bearer`. An unreadable or empty `RAY_AUTH_TOKEN_PATH` is an error. |
| `cluster.toml` `version` | Schema version. Files without it load as `1`. Unknown keys are rejected. |

How llmfit sizes a cluster:

- **Memory pool:** the smallest card's VRAM × the total GPU count.
- **Split size:** the largest TP size (up to the total GPU count) that vLLM accepts for the model and whose shards fit, one shard per GPU. It's reported in the note and as `tp_size` in JSON. TP=1 is reported as a plain GPU run.
- **KV replication:** when the split is larger than the model's KV-head count, each extra KV copy is added to the memory needed.
- **Quantization support:** set by the oldest GPU.
- **RAM:** the sum across nodes, minus 15%.
- **Refused setups:** clusters that mix backends (for example CUDA and Metal) are refused.

## Limits today

- **Tensor parallelism only.** Each GPU in the split holds one shard. Real multi-node vLLM usually splits within a node with TP and across nodes with pipeline parallelism (PP). Until PP is modelled, a model that needs every GPU but can't be split that many ways shows as `… pipeline-parallel layout needed`.
- **Speed is a ceiling.** Multi-node tok/s uses a flat interconnect factor and hasn't been measured. Treat it as an upper bound.
- **Memory is optimistic.** Per-GPU runtime overhead and per-architecture KV-cache layouts aren't counted yet. KV replication is counted for standard attention; MLA models (DeepSeek V2/V3, Kimi K2) aren't adjusted.
- **Planned, not here yet:** pipeline-parallel layouts, replica counts per cluster, and an `--emit vllm` launch command.
- **No orchestration, ever.** llmfit reads Ray once, at setup. It never launches, schedules or changes anything on the cluster.
