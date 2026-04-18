# llmfit

<p align="center">
  <img src="assets/icon.svg" alt="llmfit 图标" width="128" height="128">
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <b>中文</b>
</p>

<p align="center">
  <a href="https://github.com/AlexsJones/llmfit/actions/workflows/ci.yml"><img src="https://github.com/AlexsJones/llmfit/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/llmfit"><img src="https://img.shields.io/crates/v/llmfit.svg" alt="Crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="许可证"></a>
</p>

**数百种模型与提供商，一条命令即可找出你的硬件能运行哪些模型。**

一款终端工具，根据你系统的 RAM、CPU 和 GPU 为 LLM 模型匹配合适的规格。自动检测硬件，从质量、速度、适配度和上下文四个维度为每个模型打分，告诉你哪些模型能在你的机器上流畅运行。

内置交互式 TUI（默认）和经典 CLI 模式。支持多 GPU 配置、MoE（混合专家）架构、动态量化选择、速度估算，以及本地运行时提供商（Ollama、llama.cpp、MLX、Docker Model Runner、LM Studio）。

**新功能：[硬件模拟](#硬件模拟-s)** — 在 TUI 中按 `S` 模拟不同硬件。覆盖 RAM、VRAM 和 CPU 核心数，无需离开应用即可查看哪些模型适合目标硬件。

> **姐妹项目：**
> - [sympozium](https://github.com/sympozium-ai/sympozium/) — 在 Kubernetes 中管理 Agent。
> - [llmserve](https://github.com/AlexsJones/llmserve) — 一个简单的 TUI，用于服务本地 LLM 模型。选择模型、选择后端、开始服务。
> - [llama-panel](https://github.com/AlexsJones/llama-panel) — 用于管理本地 llama-server 实例的原生 macOS 应用。

![演示](demo.gif)

---

## 安装

### Windows
```sh
scoop install llmfit
```

如果尚未安装 Scoop，请参阅 [Scoop 安装指南](https://scoop.sh/)。

### macOS / Linux

#### Homebrew
```sh
brew install llmfit
```

### MacPorts
```sh
port install llmfit
```

#### 快速安装
```sh
curl -fsSL https://llmfit.axjns.dev/install.sh | sh
```

从 GitHub 下载最新发布的二进制文件并安装到 `/usr/local/bin`（如果没有 sudo 则安装到 `~/.local/bin`）。

**安装到 `~/.local/bin`（无需 sudo）：**
```sh
curl -fsSL https://llmfit.axjns.dev/install.sh | sh -s -- --local
```

### Docker / Podman
```sh
docker run ghcr.io/alexsjones/llmfit
```
此命令会输出 `llmfit recommend` 的 JSON 结果，可以用 `jq` 进一步查询。
```
podman run ghcr.io/alexsjones/llmfit recommend --use-case coding | jq '.models[].name'
```

### 从源码构建
```sh
git clone https://github.com/AlexsJones/llmfit.git
cd llmfit
cargo build --release
# 二进制文件位于 target/release/llmfit
```

---

## 使用方法

### TUI（默认）

```sh
llmfit
```

启动交互式终端 UI。系统配置（CPU、RAM、GPU 名称、VRAM、后端）显示在顶部。模型按综合评分排序，以可滚动的表格列出。每行显示模型的评分、预估 tok/s、最佳量化方案、运行模式、内存占用和用途分类。

| 按键                       | 操作                                            |
|----------------------------|-------------------------------------------------|
| `Up` / `Down` 或 `j` / `k` | 浏览模型                                        |
| `/`                        | 进入搜索模式（按名称、提供商、参数量、用途模糊匹配） |
| `Esc` 或 `Enter`           | 退出搜索模式                                    |
| `Ctrl-U`                   | 清除搜索                                        |
| `f`                        | 切换适配度过滤：全部、可运行、完美、良好、勉强       |
| `a`                        | 切换可用性过滤：全部、GGUF 可用、已安装            |
| `s`                        | 切换排序列：评分、参数量、内存%、上下文、日期、用途   |
| `v`                        | 进入 Visual 模式（多选模型）                      |
| `V`                        | 进入 Select 模式（按列过滤）                      |
| `t`                        | 切换颜色主题（自动保存）                          |
| `p`                        | 打开 Plan 模式（硬件规划）                        |
| `P`                        | 打开提供商过滤弹窗                              |
| `U`                        | 打开用途过滤弹窗                                |
| `C`                        | 打开能力过滤弹窗                                |
| `L`                        | 打开许可证过滤弹窗                              |
| `R`                        | 打开运行时/后端过滤弹窗（llama.cpp、MLX、vLLM）  |
| `S`                        | 打开硬件模拟弹窗（覆盖 RAM/VRAM/CPU）            |
| `h`                        | 打开帮助弹窗（所有快捷键）                      |
| `m`                        | 标记选中模型用于对比                            |
| `c`                        | 打开对比视图（已标记 vs 选中）                    |
| `x`                        | 清除对比标记                                    |
| `i`                        | 切换已安装优先排序（任何已检测的运行时提供商）    |
| `d`                        | 下载选中模型（多个提供商可用时弹出选择器）        |
| `r`                        | 从运行时提供商刷新已安装模型                    |
| `Enter`                    | 切换选中模型的详情视图                          |
| `PgUp` / `PgDn`            | 滚动 10 行                                      |
| `g` / `G`                  | 跳转到顶部 / 底部                               |
| `q`                        | 退出                                            |

### 类 Vim 模式

TUI 使用类 Vim 模式，当前模式显示在左下角状态栏。当前模式决定哪些按键生效。

#### Normal 模式

默认模式。可浏览、搜索、过滤和打开各种视图。上表中的所有按键均在此模式下有效。

#### Visual 模式 (`v`)

选择连续的多个模型进行批量对比。按 `v` 在当前行设置锚点，然后用 `j`/`k` 或方向键扩展选区。选中的行会高亮显示。

| 按键               | 操作                                 |
|--------------------|--------------------------------------|
| `j` / `k` 或方向键 | 向上/下扩展选区                      |
| `c`                | 对比所有选中模型（打开多模型对比视图） |
| `m`                | 标记当前模型用于双模型对比           |
| `Esc` 或 `v`       | 退出 Visual 模式                     |

多模型对比视图以表格形式显示，行为属性（评分、tok/s、适配度、内存%、参数量、模式、上下文、量化等），列为模型。最优值会高亮显示。如果选中的模型超出屏幕宽度，可用 `h`/`l` 或方向键水平滚动。

#### Select 模式 (`V`)

按列过滤。按 `V`（shift-v）进入 Select 模式，然后用 `h`/`l` 或方向键在列标题间移动。当前列会高亮显示。按 `Enter` 或 `Space` 激活该列对应的过滤器：

| 列                        | 过滤操作                                              |
|---------------------------|-------------------------------------------------------|
| Inst                      | 切换可用性过滤                                        |
| Model                     | 进入搜索模式                                          |
| Provider                  | 打开提供商弹窗                                        |
| Params                    | 打开参数量分组弹窗（<3B、3-7B、7-14B、14-30B、30-70B、70B+） |
| Score、tok/s、Mem%、Ctx、Date | 按该列排序                                            |
| Quant                     | 打开量化弹窗                                          |
| Mode                      | 打开运行模式弹窗（GPU、MoE、CPU+GPU、CPU）                 |
| Fit                       | 切换适配度过滤                                        |
| Use Case                  | 打开用途弹窗                                          |

在 Select 模式下仍可用 `j`/`k`、方向键、`Ctrl-U`、`Ctrl-D`、`PageUp`、`PageDown`、`Home` 和 `End` 浏览行，以便在应用过滤器时查看效果。按 `Esc` 返回 Normal 模式。

### TUI Plan 模式 (`p`)

Plan 模式与常规适配分析相反：不是问"我的硬件能跑什么？"，而是估算"这个模型配置需要什么硬件？"。

在选中的行上按 `p`，然后：

| 按键                   | 操作                                     |
|------------------------|------------------------------------------|
| `Tab` / `j` / `k`      | 在可编辑字段间移动（上下文、量化、目标 TPS） |
| `Left` / `Right`       | 在当前字段内移动光标                     |
| 输入                   | 编辑当前字段                             |
| `Backspace` / `Delete` | 删除字符                                 |
| `Ctrl-U`               | 清空当前字段                             |
| `Esc` 或 `q`           | 退出 Plan 模式                           |

Plan 模式显示以下估算：
- 最低和推荐的 VRAM/RAM/CPU 核心数
- 可行的运行路径（GPU、CPU 卸载、纯 CPU）
- 达到更好适配目标所需的升级差距

### 硬件模拟 (`S`)

在 TUI 中按 `S` 打开硬件模拟弹窗。覆盖 RAM、VRAM 和 CPU 核心数，查看哪些模型适合不同的目标硬件。所有模型评分、适配等级和速度估算会根据模拟配置即时重新计算。

![硬件模拟](simulation.png)

| 按键                   | 操作                                    |
|------------------------|-----------------------------------------|
| `Tab` / `j` / `k`      | 在 RAM、VRAM、CPU 字段间切换            |
| 输入数字               | 编辑选中字段                            |
| `Enter`               | 应用模拟                                |
| `Ctrl-R`              | 重置为真实检测到的硬件                  |
| `Esc`                 | 取消并关闭                              |

模拟激活时，系统栏和状态栏会显示 `SIM` 标识。整个模型表格会反映模拟硬件，直到你重置。

### 主题

按 `t` 可在 10 种内置颜色主题间切换。选择会自动保存到 `~/.config/llmfit/theme`，下次启动时恢复。

| 主题                     | 描述                                        |
|--------------------------|---------------------------------------------|
| **Default**              | llmfit 原始配色                             |
| **Dracula**              | 深紫色背景搭配柔和色调                      |
| **Solarized**            | Ethan Schoonover 的 Solarized Dark 配色方案 |
| **Nord**                 | 极地风格，冷蓝灰色调                         |
| **Monokai**              | Monokai Pro 暖色语法配色                    |
| **Gruvbox**              | 复古风格，暖色大地色调                       |
| **Catppuccin Latte**     | 🌻 浅色主题——和谐的柔和反转配色             |
| **Catppuccin Frappé**    | 🪴 低对比度深色——柔和、内敛的美学            |
| **Catppuccin Macchiato** | 🌺 中对比度深色——温柔舒缓的色调             |
| **Catppuccin Mocha**     | 🌿 最深的暗色变体——温馨且色彩丰富           |

### Web 仪表盘

当你以非 JSON 模式运行 `llmfit` 时，会自动在后台启动 Web 仪表盘，默认监听 `0.0.0.0:8787`。可在同一网络中的任意浏览器打开：

```
http://<你的机器IP>:8787
```

你也可以通过环境变量覆盖主机或端口：

```sh
LLMFIT_DASHBOARD_HOST=0.0.0.0 LLMFIT_DASHBOARD_PORT=9000 llmfit
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLMFIT_DASHBOARD_HOST` | `0.0.0.0` | 仪表盘服务绑定的网卡地址 |
| `LLMFIT_DASHBOARD_PORT` | `8787` | 仪表盘服务绑定的端口 |

如需禁用自动启动仪表盘，添加 `--no-dashboard`：

```sh
llmfit --no-dashboard
```

### CLI 模式

使用 `--cli` 或任何子命令获取经典表格输出：

```sh
# 按适配度排序的所有模型表格
llmfit --cli

# 仅显示完美适配的模型，前 5 个
llmfit fit --perfect -n 5

# 显示检测到的系统配置
llmfit system

# 列出数据库中所有模型
llmfit list

# 按名称、提供商或参数量搜索
llmfit search "llama 8b"

# 单个模型的详细信息
llmfit info "Mistral-7B"

# 前 5 个推荐（JSON 格式，供 agent/脚本使用）
llmfit recommend --json --limit 5

# 按用途过滤推荐
llmfit recommend --json --use-case coding --limit 3

# 强制指定运行时（在 Apple Silicon 上绕过自动 MLX 选择）
llmfit recommend --force-runtime llamacpp
llmfit recommend --force-runtime llamacpp --use-case coding --limit 3

# 为特定模型配置规划所需硬件
llmfit plan "Qwen/Qwen3-4B-MLX-4bit" --context 8192
llmfit plan "Qwen/Qwen3-4B-MLX-4bit" --context 8192 --quant mlx-4bit
llmfit plan "Qwen/Qwen3-4B-MLX-4bit" --context 8192 --target-tps 25 --json

# 作为节点级 REST API 运行（供集群调度器/聚合器使用）
llmfit serve --host 0.0.0.0 --port 8787
```

### REST API (`llmfit serve`)

`llmfit serve` 启动一个 HTTP API，提供与 TUI/CLI 相同的适配/评分数据，包括过滤和节点级最优模型选择。

```sh
# 健康检查
curl http://localhost:8787/health

# 节点硬件信息
curl http://localhost:8787/api/v1/system

# 带过滤的完整适配列表
curl "http://localhost:8787/api/v1/models?min_fit=marginal&runtime=llamacpp&sort=score&limit=20"

# 关键调度端点：该节点的最佳可运行模型
curl "http://localhost:8787/api/v1/models/top?limit=5&min_fit=good&use_case=coding"

# 按模型名称/提供商搜索
curl "http://localhost:8787/api/v1/models/Mistral?runtime=any"
```

`models`/`models/top` 支持的查询参数：

- `limit`（或 `n`）：返回的最大行数
- `perfect`：`true|false`（为 `true` 时强制仅显示完美适配）
- `min_fit`：`perfect|good|marginal|too_tight`
- `runtime`：`any|mlx|llamacpp`
- `use_case`：`general|coding|reasoning|chat|multimodal|embedding`
- `provider`：提供商文本过滤（子字符串匹配）
- `search`：跨名称/提供商/参数量/用途的全文过滤
- `sort`：`score|tps|params|mem|ctx|date|use_case`
- `include_too_tight`：包含不可运行的行（`/top` 默认 `false`，`/models` 默认 `true`）
- `max_context`：每次请求的上下文长度上限，用于内存估算
- `force_runtime`：`mlx|llamacpp|vllm` — 在分析过程中覆盖自动运行时选择

本地验证 API 行为：

```sh
# 自动启动服务器并运行端点/模式/过滤断言
python3 scripts/test_api.py --spawn

# 或测试已运行的服务器
python3 scripts/test_api.py --base-url http://127.0.0.1:8787
```

### 硬件覆盖

在某些系统上硬件自动检测可能失败（例如 `nvidia-smi` 故障、虚拟机、直通配置），或者你可能想针对不同的目标硬件评估模型适配。使用 `--memory`、`--ram` 和 `--cpu-cores` 覆盖检测到的值：

```sh
# 覆盖 GPU VRAM
llmfit --memory=32G

# 覆盖系统 RAM
llmfit --ram=128G

# 覆盖 CPU 核心数
llmfit --cpu-cores=16

# 组合覆盖以模拟目标硬件
llmfit --memory=24G --ram=64G --cpu-cores=8 fit
llmfit --memory=24G --ram=64G system --json

# 适用于所有模式：TUI、CLI 和子命令
llmfit --memory=24G --cli
llmfit --memory=24G fit --perfect -n 5
llmfit --ram=64G recommend --json
```

`--memory` 和 `--ram` 支持的后缀：`G`/`GB`/`GiB`（千兆字节）、`M`/`MB`/`MiB`（兆字节）、`T`/`TB`/`TiB`（太字节）。不区分大小写。如果未检测到 GPU，`--memory` 会创建一个虚拟 GPU 条目，以便按 GPU 推理对模型评分。在统一内存系统（Apple Silicon）上，`--ram` 同时更新 VRAM；使用 `--memory` 可独立覆盖 VRAM。

### 上下文长度上限

使用 `--max-context` 限制用于内存估算的上下文长度（不改变每个模型标称的最大上下文）：

```sh
# 按 4K 上下文估算内存适配
llmfit --max-context 4096 --cli

# 适用于子命令
llmfit --max-context 8192 fit --perfect -n 5
llmfit --max-context 16384 recommend --json --limit 5
```

如果未设置 `--max-context`，llmfit 会在可用时使用 `OLLAMA_CONTEXT_LENGTH`。

### JSON 输出

在任何子命令后添加 `--json` 获取机器可读输出：

```sh
llmfit --json system     # 硬件信息（JSON）
llmfit --json fit -n 10  # 前 10 个适配结果（JSON）
llmfit recommend --json  # 前 5 个推荐（recommend 默认输出 JSON）
llmfit plan "Qwen/Qwen2.5-Coder-0.5B-Instruct" --context 8192 --json
```

`plan` 的 JSON 输出包含以下稳定字段：
- 请求参数（`context`、`quantization`、`target_tps`）
- 估算的最低/推荐硬件
- 每条路径的可行性（`gpu`、`cpu_offload`、`cpu_only`）
- 升级差距

---

## 工作原理

1. **硬件检测** -- 通过 `sysinfo` 读取总计/可用 RAM，统计 CPU 核心数，并探测 GPU：
   - **NVIDIA** -- 通过 `nvidia-smi` 支持多 GPU。聚合所有检测到的 GPU 的 VRAM。如果报告失败，则根据 GPU 型号名称估算 VRAM。
   - **AMD** -- 通过 `rocm-smi` 检测。
   - **Intel Arc** -- 独立显卡通过 sysfs 检测 VRAM，集成显卡通过 `lspci` 检测。
   - **Apple Silicon** -- 通过 `system_profiler` 检测统一内存。VRAM = 系统 RAM。
   - **Ascend** -- 通过 `npu-smi` 检测。
   - **后端检测** -- 自动识别加速后端（CUDA、Metal、ROCm、SYCL、CPU ARM、CPU x86、Ascend）用于速度估算。

2. **模型数据库** -- 数百个模型来源于 HuggingFace API，存储在 `data/hf_models.json` 中并在编译时嵌入。内存需求根据量化层级（Q8_0 到 Q2_K）的参数量计算。VRAM 是 GPU 推理的主要约束；系统 RAM 是纯 CPU 执行的后备方案。

   **MoE 支持** -- 自动检测混合专家架构（Mixtral、DeepSeek-V2/V3）的模型。每个 token 只有部分专家处于活跃状态，因此实际 VRAM 需求远低于总参数量的暗示。例如，Mixtral 8x7B 总参数量为 46.7B，但每个 token 仅激活约 12.9B，通过专家卸载将 VRAM 需求从 23.9 GB 降至约 6.6 GB。

3. **动态量化** -- llmfit 不假设固定量化，而是尝试适配你硬件的最高质量量化。它从 Q8_0（最高质量）到 Q2_K（最高压缩）逐级尝试，选择能装入可用内存的最高质量等级。如果在完整上下文下无法装入，则尝试半上下文。

4. **多维评分** -- 每个模型按四个维度评分（每个 0-100）：

   | 维度       | 衡量内容                                 |
   |------------|------------------------------------------|
   | **质量**   | 参数量、模型系列声誉、量化惩罚、任务对齐度  |
   | **速度**   | 基于后端、参数量和量化的预估 tokens/sec   |
   | **适配度** | 内存利用效率（最佳区间：可用内存的 50-80%） |
   | **上下文** | 上下文窗口能力与用途目标的对比           |

   各维度通过加权合成为综合评分。权重因用途类别而异（通用、编程、推理、对话、多模态、嵌入）。例如，对话类更侧重速度（0.35），推理类更侧重质量（0.55）。模型按综合评分排序，不可运行的模型（Too Tight）始终排在最后。

5. **速度估算** -- LLM 推理中的 token 生成受内存带宽限制：每个 token 需要从 VRAM 完整读取一次模型权重。当识别出 GPU 型号时，llmfit 使用其实际内存带宽来估算吞吐量：

   公式：`(bandwidth_GB_s / model_size_GB) × efficiency_factor`

   效率因子（0.55）考虑了内核开销、KV 缓存读取和内存控制器效应。该方法已通过 llama.cpp 的公开基准测试验证（[Apple Silicon](https://github.com/ggml-org/llama.cpp/discussions/4167)、[NVIDIA T4](https://github.com/ggml-org/llama.cpp/discussions/4225)）及实际测量数据。

   带宽查找表涵盖约 80 种 GPU，覆盖 NVIDIA（消费级 + 数据中心级）、AMD（RDNA + CDNA）和 Apple Silicon 系列。

   对于未识别的 GPU，llmfit 使用按后端的速度常量作为回退：

   | 后端         | 速度常量 |
   |--------------|----------|
   | CUDA         | 220      |
   | Metal        | 160      |
   | ROCm         | 180      |
   | SYCL         | 100      |
   | CPU (ARM)    | 90       |
   | CPU (x86)    | 70       |
   | NPU (Ascend) | 390      |

   回退公式：`K / params_b × quant_speed_multiplier`，对 CPU 卸载（0.5x）、纯 CPU（0.3x）和 MoE 专家切换（0.8x）施加惩罚。

6. **适配分析** -- 评估每个模型的内存兼容性：

   **运行模式：**
   - **GPU** -- 模型完全装入 VRAM。推理速度快。
   - **MoE** -- 混合专家 + 专家卸载。活跃专家在 VRAM 中，非活跃专家在 RAM 中。
   - **CPU+GPU** -- VRAM 不足，溢出到系统 RAM 并使用部分 GPU 卸载。
   - **CPU** -- 无 GPU。模型完全加载到系统 RAM 中。

   **适配等级：**
   - **Perfect（完美）** -- GPU 上满足推荐内存。需要 GPU 加速。
   - **Good（良好）** -- 有余量地装入。MoE 卸载或 CPU+GPU 模式的最佳等级。
   - **Marginal（勉强）** -- 装入紧张，或纯 CPU 运行（纯 CPU 始终封顶在此等级）。
   - **Too Tight（过紧）** -- VRAM 和系统 RAM 均不足。

---

## 模型数据库

模型列表由 `scripts/scrape_hf_models.py` 生成，这是一个独立的 Python 脚本（仅使用标准库，无需 pip 依赖），通过 HuggingFace REST API 查询。数百个模型和提供商，包括 Meta Llama、Mistral、Qwen、Google Gemma、Microsoft Phi、DeepSeek、IBM Granite、Allen Institute OLMo、xAI Grok、Cohere、BigCode、01.ai、Upstage、TII Falcon、HuggingFace、Zhipu GLM、Moonshot Kimi、Baidu ERNIE 等。爬虫通过模型配置（`num_local_experts`、`num_experts_per_tok`）和已知架构映射自动检测 MoE 架构。

模型类别涵盖通用、编程（CodeLlama、StarCoder2、WizardCoder、Qwen2.5-Coder、Qwen3-Coder）、推理（DeepSeek-R1、Orca-2）、多模态/视觉（Llama 3.2 Vision、Llama 4 Scout/Maverick、Qwen2.5-VL）、对话、企业级（IBM Granite）和嵌入（nomic-embed、bge）。

完整列表请参阅 [MODELS.md](MODELS.md)。

模型数据库在编译时嵌入，因此**最终用户**通过升级 llmfit 本身来获取更新（`brew upgrade llmfit`、`scoop update llmfit` 或下载更新的发布版本）。以下命令适用于从源码刷新数据库的**贡献者**：

刷新模型数据库：

```sh
# 自动更新（推荐）
make update-models

# 或直接运行脚本
./scripts/update_models.sh

# 或手动执行
python3 scripts/scrape_hf_models.py
cargo build --release
```

爬虫将结果写入 `data/hf_models.json`，通过 `include_str!` 在编译时嵌入二进制文件。自动更新脚本会备份现有数据、验证 JSON 输出并重新构建二进制文件。

默认情况下，爬虫会使用来自 [unsloth](https://huggingface.co/unsloth) 和 [bartowski](https://huggingface.co/bartowski) 等提供商的已知 GGUF 下载源来丰富模型信息。结果缓存在 `data/gguf_sources_cache.json` 中（7 天 TTL），以避免重复 API 调用。使用 `--no-gguf-sources` 可跳过丰富步骤以加快爬取速度。

---

## 项目结构

```
src/
  main.rs         -- CLI 参数解析、入口、TUI 启动
  hardware.rs     -- 系统 RAM/CPU/GPU 检测（多 GPU、后端识别）
  models.rs       -- 模型数据库、量化层级、动态量化选择
  fit.rs          -- 多维评分（Q/S/F/C）、速度估算、MoE 卸载
  providers.rs    -- 运行时提供商集成（Ollama、llama.cpp、MLX、Docker Model Runner、LM Studio）、安装检测、拉取/下载
  display.rs      -- 经典 CLI 表格渲染 + JSON 输出
  tui_app.rs      -- TUI 应用状态、过滤器、导航
  tui_ui.rs       -- TUI 渲染（ratatui）
  tui_events.rs   -- TUI 键盘事件处理（crossterm）
data/
  hf_models.json  -- 模型数据库（206 个模型）
skills/
  llmfit-advisor/ -- 用于硬件感知模型推荐的 OpenClaw 技能
scripts/
  scrape_hf_models.py        -- HuggingFace API 爬虫
  update_models.sh            -- 自动化数据库更新脚本
  install-openclaw-skill.sh   -- 安装 OpenClaw 技能
Makefile           -- 构建和维护命令
```

---

## 发布到 crates.io

`Cargo.toml` 已包含所需的元数据（描述、许可证、仓库地址）。发布步骤：

```sh
# 先进行试运行以发现问题
cargo publish --dry-run

# 正式发布（需要 crates.io API token）
cargo login
cargo publish
```

发布前请确认：

- `Cargo.toml` 中的版本号正确（每次发布时递增）。
- 仓库根目录存在 `LICENSE` 文件。如果缺失请创建：

```sh
# MIT 许可证：
curl -sL https://opensource.org/license/MIT -o LICENSE
# 或自行编写。Cargo.toml 声明 license = "MIT"。
```

- `data/hf_models.json` 已提交。它在编译时嵌入，必须存在于发布的 crate 中。
- `Cargo.toml` 中的 `exclude` 列表将 `target/`、`scripts/` 和 `demo.gif` 排除在发布的 crate 之外，以减小下载体积。

发布更新：

```sh
# 递增版本号
# 编辑 Cargo.toml: version = "0.2.0"
cargo publish
```

---

## 依赖

| Crate                  | 用途                                     |
|------------------------|------------------------------------------|
| `clap`                 | 基于 derive 宏的 CLI 参数解析            |
| `sysinfo`              | 跨平台 RAM 和 CPU 检测                   |
| `serde` / `serde_json` | 模型数据库的 JSON 反序列化               |
| `tabled`               | CLI 表格格式化                           |
| `colored`              | CLI 彩色输出                             |
| `ureq`                 | 用于运行时/提供商 API 集成的 HTTP 客户端 |
| `ratatui`              | 终端 UI 框架                             |
| `crossterm`            | ratatui 的终端输入/输出后端              |

---

## 运行时提供商集成

llmfit 支持多个本地运行时提供商：

- **Ollama**（基于守护进程/API 的拉取）
- **llama.cpp**（从 Hugging Face 直接下载 GGUF + 本地缓存检测）
- **MLX**（Apple Silicon / mlx-community 模型缓存 + 可选服务器）— MLX 下载映射到 HuggingFace 上的 `mlx-community/*` 仓库，而非原始模型发布者
- **Docker Model Runner**（Docker Desktop 内置的模型服务）
- **LM Studio**（本地模型服务器，支持 REST API 模型管理和下载）

当某个模型有多个兼容的提供商可用时，在 TUI 中按 `d` 会打开提供商选择弹窗。

### Ollama 集成

llmfit 与 [Ollama](https://ollama.com) 集成，可检测你已安装的模型并直接从 TUI 下载新模型。

### 要求

- **Ollama 必须已安装且正在运行**（`ollama serve` 或 Ollama 桌面应用）
- llmfit 连接到 `http://localhost:11434`（Ollama 默认 API 端口）
- 无需配置 -- 如果 Ollama 正在运行，llmfit 会自动检测到它

### 远程 Ollama 实例

要连接到在其他机器或端口上运行的 Ollama，设置 `OLLAMA_HOST` 环境变量：

```sh
# 连接到指定 IP 和端口的 Ollama
OLLAMA_HOST="http://192.168.1.100:11434" llmfit

# 通过主机名连接
OLLAMA_HOST="http://ollama-server:666" llmfit

# 适用于所有 TUI 和 CLI 命令
OLLAMA_HOST="http://192.168.1.100:11434" llmfit --cli
OLLAMA_HOST="http://192.168.1.100:11434" llmfit fit --perfect -n 5
```

适用场景：
- 在一台机器上运行 llmfit，而 Ollama 在另一台机器上提供服务（例如 GPU 服务器 + 笔记本客户端）
- 连接到在 Docker 容器中以自定义端口运行的 Ollama
- 使用反向代理或负载均衡器后面的 Ollama

### 工作原理

启动时，llmfit 查询 `GET /api/tags` 列出已安装的 Ollama 模型。每个已安装的模型在 TUI 的 **Inst** 列显示绿色 **✓**。系统栏显示 `Ollama: ✓ (N installed)`。

在模型上按 `d` 时，llmfit 向 Ollama 发送 `POST /api/pull` 来下载模型。该行会高亮显示并带有动画进度指示器，实时显示下载进度。下载完成后，模型可立即在 Ollama 中使用。

如果 Ollama 未运行，Ollama 相关操作会被跳过；TUI 仍然支持其他可用的提供商（如 llama.cpp）。

### llama.cpp 集成

llmfit 与 [llama.cpp](https://github.com/ggml-org/llama.cpp) 集成，在 TUI 和 CLI 中均可作为运行时/下载提供商使用。

要求：

- `llama-cli` 或 `llama-server` 在 `PATH` 中可用（用于运行时检测）
- 需要网络访问 Hugging Face 以下载 GGUF 文件

工作原理：

- llmfit 将 HF 模型映射到已知的 GGUF 仓库（带有启发式回退）
- 将 GGUF 文件下载到本地 llama.cpp 模型缓存
- 当本地存在匹配的 GGUF 文件时标记模型为已安装

#### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLAMA_CPP_PATH` | *(无)* | 包含 llama.cpp 二进制文件（`llama-cli`、`llama-server`）的目录。优先于 `PATH` 查找��� |
| `LLAMA_SERVER_PORT` | `8080` | 探测运行中的 `llama-server` 健康端点以进行运行时检测时使用的端口。 |

如果 llama.cpp 安装在非标准位置，设置 `LLAMA_CPP_PATH` 以便 llmfit 无需将其加入 `PATH` 即可找���它。

### Docker Model Runner 集成

llmfit 与 [Docker Model Runner](https://docs.docker.com/desktop/features/model-runner/) 集成，这是 Docker Desktop 内置的模型服务功能。

要求：

- Docker Desktop 已启用 Model Runner
- 默认端点：`http://localhost:12434`

工作原理：

- llmfit 查询 `GET /engines` 列出 Docker Model Runner 中可用的模型
- 使用 Ollama 风格的标签映射将模型与 HF 数据库匹配（Docker Model Runner 使用 `ai/<tag>` 命名）
- 在 TUI 中按 `d` 通过 `docker model pull` 拉取模型

### 远程 Docker Model Runner 实例

要连接到不同主机或端口的 Docker Model Runner，设置 `DOCKER_MODEL_RUNNER_HOST` 环境变量：

```sh
DOCKER_MODEL_RUNNER_HOST="http://192.168.1.100:12434" llmfit
```

### LM Studio 集成

llmfit 与 [LM Studio](https://lmstudio.ai) 集成，作为本地模型服务器，支持内置模型下载功能。

要求：

- LM Studio 必须运行且本地服务器已启用
- 默认端点：`http://127.0.0.1:1234`

工作原理：

- llmfit 查询 `GET /v1/models` 列出 LM Studio 中可用的模型
- 在 TUI 中按 `d` 通过 `POST /api/v1/models/download` 触发下载
- 通过轮询 `GET /api/v1/models/download-status` 跟踪下载进度
- LM Studio 直接接受 HuggingFace 模型名称，无需名称映射

### 远程 LM Studio 实例

要连接到不同主机或端口的 LM Studio，设置 `LMSTUDIO_HOST` 环境变量：

```sh
LMSTUDIO_HOST="http://192.168.1.100:1234" llmfit
```

### 模型名称映射

llmfit 的数据库使用 HuggingFace 模型名称（例如 `Qwen/Qwen2.5-Coder-14B-Instruct`），而 Ollama 使用自己的命名方案（例如 `qwen2.5-coder:14b`）。llmfit 维护了一个精确的映射表，确保安装检测和拉取操作解析到正确的模型。每个映射都是精确的 -- `qwen2.5-coder:14b` 映射到 Coder 模型，而不是基础的 `qwen2.5:14b`。

---

## 平台支持

- **Linux** -- 完全支持。通过 `nvidia-smi`（NVIDIA）、`rocm-smi`（AMD）、sysfs/`lspci`（Intel Arc）和 `npu-smi`（Ascend）进行 GPU 检测。
- **macOS (Apple Silicon)** -- 完全支持。通过 `system_profiler` 检测统一内存。VRAM = 系统 RAM（共享池）。模型通过 Metal GPU 加速运行。
- **macOS (Intel)** -- RAM 和 CPU 检测正常。如果 `nvidia-smi` 可用，可检测独立 GPU。
- **Windows** -- RAM 和 CPU 检测正常。如果安装了 `nvidia-smi`，可检测 NVIDIA GPU。
- **Android / Termux / PRoot** -- CPU 和 RAM 检测通常正常，但目前不支持 GPU 自动检测。Adreno 等移动 GPU 通常无法通过 llmfit 使用的桌面/服务器探测接口访问。

### GPU 支持

| 厂商            | 检测方式                      | VRAM 报告             |
|-----------------|-------------------------------|-----------------------|
| NVIDIA          | `nvidia-smi`                  | 精确的独立 VRAM       |
| AMD             | `rocm-smi`                    | 已检测（VRAM 可能未知） |
| Intel Arc（独立） | sysfs (`mem_info_vram_total`) | 精确的独立 VRAM       |
| Intel Arc（集成） | `lspci`                       | 共享系统内存          |
| Apple Silicon   | `system_profiler`             | 统一内存（= 系统 RAM）  |
| Ascend          | `npu-smi`                     | 已检测（VRAM 可能未知） |

如果自动检测失败或报告的值不正确，使用 `--memory`、`--ram` 或 `--cpu-cores` 覆盖（参见上方[硬件覆盖](#硬���覆盖)）。

### Android / Termux 说明

在 **Termux + PRoot** 等 Android 环境中，llmfit 通常无法通过标准 Linux 检测路径（`nvidia-smi`、`rocm-smi`、DRM/sysfs、`lspci` 等）检测到移动 GPU。在这些环境中，"未检测到 GPU"是当前实现的预期行为。

如果你仍希望在统一内存的手机或平板上获得 GPU 风格的推荐，可使用手动内存覆盖：

```sh
llmfit --memory=8G fit -n 20
llmfit recommend --json --memory=8G --limit 10
```

这仅是推荐/评分的变通方案；不提供真正的 Android GPU 运行时检测。

---

## 贡献

欢迎贡献，特别是添加新模型。

### 添加模型

1. 在 `scripts/scrape_hf_models.py` 的 `TARGET_MODELS` 列表中添加模型的 HuggingFace 仓库 ID（例如 `meta-llama/Llama-3.1-8B`）。
2. 如果模型有访问限制（需要 HuggingFace 身份验证才能访问元数据），在同一脚本的 `FALLBACKS` 列表中添加包含参数量和上下文长度的回退条目。
3. 运行自动更新脚本：
   ```sh
   make update-models
   # 或: ./scripts/update_models.sh
   ```
4. 验证更新后的模型列表：`./target/release/llmfit list`
5. 运行以下命令更新 [MODELS.md](MODELS.md)：`python3 << 'EOF' < scripts/...`（参见提交历史中的生成脚本）
6. 提交 Pull Request。

参见 [MODELS.md](MODELS.md) 查看当前列表，[AGENTS.md](AGENTS.md) 查看架构详情。

---

## OpenClaw 集成

llmfit 作为 [OpenClaw](https://github.com/openclaw/openclaw) 技能提供，让 agent 能够推荐适合硬件的本地模型，并自动配置 Ollama/vLLM/LM Studio 提供商。

### 安装技能

```sh
# 从 llmfit 仓库
./scripts/install-openclaw-skill.sh

# 或手动安装
cp -r skills/llmfit-advisor ~/.openclaw/skills/
```

安装后，可以向 OpenClaw agent 提问：

- "我能运行哪些本地模型？"
- "为我的硬件推荐一个编程模型"
- "用最适合我 GPU 的模型配置 Ollama"

Agent 会在后台调用 `llmfit recommend --json`，解读结果，并提议用最优的模型选择配置你的 `openclaw.json`。

### 工作原理

该技能教会 OpenClaw agent：

1. 通过 `llmfit --json system` 检测你的硬件
2. 通过 `llmfit recommend --json` 获取排序后的推荐
3. 将 HuggingFace 模型名称映射到 Ollama/vLLM/LM Studio 标签
4. 配置 `openclaw.json` 中的 `models.providers.ollama.models`

参见 [skills/llmfit-advisor/SKILL.md](skills/llmfit-advisor/SKILL.md) 查看完整技能定义。

---

## 替代方案

如果你在寻找不同的方案，可以看看 [llm-checker](https://github.com/Pavelevich/llm-checker) -- 一个带有 Ollama 集成的 Node.js CLI 工具，可以直接拉取和基准测试模型。它采用更直接的方式，通过 Ollama 在你的硬件上实际运行模型，而不是从配置参数估算。如果你已安装 Ollama 并想测试真实性能，这是个不错的选择。注意它不支持 MoE（混合专家）架构 -- 所有模型都被视为密集模型，因此 Mixtral 或 DeepSeek-V3 等模型的内存估算将反映总参数量而非较小的活跃子集。

---

## 许可证

MIT

---

*本文档由 [@JasonYeYuhe](https://github.com/JasonYeYuhe) 翻译并维护。如果您发现任何翻译问题或需要增加新特性说明，欢迎提交 Issue 或与我联系。*
