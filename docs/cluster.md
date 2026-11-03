# Cluster mode

Check which models fit across several machines before you launch anything: a homelab of DGX Sparks, or a few multi-GPU servers running Ray and vLLM.

[← Back to README](../README.md) · [CLI & automation](cli.md) · [How it works](how-it-works.md)

## Why this is useful

- **Know before you launch.** Find out whether a model fits across your machines before you spend an hour starting vLLM on them.
- **Catch splits vLLM would refuse.** Most models can't be split 3 ways, so on 3 × DGX Spark llmfit tells you a layout is invalid instead of you finding out from a vLLM startup crash.
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

The token is only sent in the request's `Authorization` header. It is never written to `cluster.toml` or printed. The plain-HTTP warning appears whenever discovery uses `http://` to anything other than localhost.

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

Ray reports each node's GPU count, CPU cores, RAM and, from its per-GPU stats, the GPU model. When a card also reports its total memory (most discrete NVIDIA GPUs do), llmfit fills in `gpu_vram_gb` too. Here the GB10s reported a name but no memory total, so VRAM is left at 0 for you to fill in.

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

**What you run.** Add `--cluster` to `fit`, `info` or the other analysis commands. To look at one model, use `info`:

```
$ llmfit --cluster info Qwen/Qwen3-32B
📡 Using cluster config: 3 nodes, 360 GB total VRAM
…
Fit Analysis:
  Status: 🔴 Too Tight
  Run Mode: TP
  Memory Utilization: 6.0% (21.5 / 360.0 GB)
…
Notes:
  Context capped at 8192 tokens for estimation (model supports up to 40960; use --max-context to override)
  TP=3 invalid for this model (64 attention heads not divisible by 3); pipeline-parallel layout needed (not yet modelled)
  Best quantization for hardware: Q8_0 (model default: Q4_K_M)
  Baseline estimated speed: 5.3 tok/s
```

**What you see.** Memory isn't the problem here: the model needs about 6% of the pool. The problem is that vLLM shards attention heads evenly across tensor-parallel ranks, and 64 heads don't split 3 ways. llmfit reports this as **Too Tight** with the reason, so the model drops out of the "fits" list.

A model whose heads do split 3 ways fits normally:

```
$ llmfit --cluster info zai-org/glm-edge-4b-chat
…
Fit Analysis:
  Status: 🟢 Perfect
  Run Mode: TP
  Memory Utilization: 1.1% (3.9 / 360.0 GB)
Notes:
  Cluster: tensor-parallel via vLLM (TP=3), one shard per GPU across 3 GPUs on 3 nodes
…
```

When the catalog has no head counts for a model, llmfit keeps the memory verdict but says the split hasn't been checked:

```
$ llmfit --cluster info huihui-ai/DeepSeek-R1-0528-Qwen3-8B-abliterated
…
Fit Analysis:
  Status: 🟢 Perfect
…
Notes:
  TP=3 layout unverified: attention-head counts unknown for this model, so vLLM may refuse this split
  Cluster: tensor-parallel via vLLM (TP=3), one shard per GPU across 3 GPUs on 3 nodes
…
```

**How it helps.** Today, most models in the catalog that have head metadata can't run with TP=3. llmfit tells you so up front, so you don't find out from a vLLM startup error after copying weights to three machines. On a 3-node cluster, the `llmfit --cluster fit` list will often lead with "unverified" models. Read the note before you rely on one.

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
| `llmfit cluster init` | Interactive setup. It tries the Ray Dashboard first and falls back to manual entry. |
| `llmfit cluster init --ray-url <url>` | Non-interactive discovery from a Ray Dashboard URL (`http://`, `https://`, `host:port` or `[ipv6]:port`). Writes `cluster.toml`. |
| `llmfit cluster status` | Prints the saved cluster. Add `--json` for machine-readable output. |
| `llmfit cluster clear` | Deletes the saved cluster config. |
| `--cluster` / `--no-cluster` | Score against the saved cluster, or skip it and hide the hint. |
| `--memory <size>` with `--cluster` | Overrides per-GPU VRAM for every GPU, for example when Ray reported none. |
| `RAY_AUTH_TOKEN`, `RAY_AUTH_TOKEN_PATH`, `~/.ray/auth_token` | Ray auth token, checked in that order and sent as `Authorization: Bearer`. An unreadable or empty `RAY_AUTH_TOKEN_PATH` is an error. |
| `cluster.toml` `version` | Schema version. Files without it load as `1`. Unknown keys are rejected. |

How llmfit sizes a cluster:

- **Memory pool:** the smallest card's VRAM × the total GPU count.
- **Split size:** TP = the total GPU count, one shard per GPU.
- **Quantization support:** set by the oldest GPU.
- **RAM:** the sum across nodes, minus 15%.
- **Refused setups:** clusters that mix backends (for example CUDA and Metal) are refused.

## Limits today

- **Tensor parallelism only.** Every GPU holds one shard. Real multi-node vLLM usually splits within a node with TP and across nodes with pipeline parallelism (PP). Until PP is modelled, many models show as `TP=N invalid … pipeline-parallel layout needed` on clusters with an odd GPU count.
- **Speed is a ceiling.** Multi-node tok/s uses a flat interconnect factor and hasn't been measured. Treat it as an upper bound.
- **Memory is optimistic.** Per-GPU runtime overhead and per-architecture KV-cache layouts aren't counted yet.
- **Planned, not here yet:** pipeline-parallel layouts, replica counts per cluster, and an `--emit vllm` launch command.
- **No orchestration, ever.** llmfit reads Ray once, at setup. It never launches, schedules or changes anything on the cluster.
