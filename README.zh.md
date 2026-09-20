# llmfit

<p align="center">
  <img src="assets/icon.svg" alt="llmfit icon" width="128" height="128">
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <b>中文</b> ·
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  <a href="https://github.com/AlexsJones/llmfit/actions/workflows/ci.yml"><img src="https://github.com/AlexsJones/llmfit/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/llmfit"><img src="https://img.shields.io/crates/v/llmfit.svg" alt="Crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://about.signpath.io"><img src="https://img.shields.io/badge/SignPath-signed-brightgreen?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgZmlsbD0id2hpdGUiIHZpZXdCb3g9IjAgMCAxNiAxNiI+PHBhdGggZD0iTTEwLjA2NyA0LjU2N2wtNC43MzQgNC43MzMtMS40LTEuNGExIDEgMCAwIDAtMS40MTQgMS40MTRsMi4xIDIuMWExIDEgMCAwIDAgMS40MTQgMGw1LjQ0LTUuNDRhMSAxIDAgMCAwLTEuNDE0LTEuNDE0eiIvPjwvc3ZnPg==" alt="Signed with SignPath"></a>
</p>

> **📊 新功能：基准测试与共享 — 来自你机器的真实数据，让所有人的估算更准确。** 下载模型、运行服务并在你的硬件上实测 tok/s — 然后直接从 TUI 将结果以 PR 形式贡献回项目。无需 `gh` CLI，也无需第三方账号。每次测试都会先保存在本地，你自己的实测数据会替换适配表中的估算值，每条合并的提交都会随下一个版本发布：相同硬件的用户无需自己运行基准测试，就能获得实测 `✓` 数据。[按步骤查看基准测试指南 →](docs/benchmarking.md)
>
> *此前：[llmfit 1.0 — 让每个数字都可验证的里程碑版本 →](https://github.com/AlexsJones/llmfit/discussions/708)*

**数百种模型与提供商，一条命令即可找出你的硬件能运行哪些模型。**

一款终端工具，根据你系统的 RAM、CPU 和 GPU 为 LLM 模型匹配合适的规格。自动检测硬件，从质量、速度、适配度和上下文四个维度为每个模型打分，告诉你哪些模型能在你的机器上流畅运行。

内置交互式 TUI（默认）和经典 CLI 模式。支持多 GPU 配置、MoE（混合专家）架构、动态量化选择、速度估算，以及本地运行时提供商（Ollama、llama.cpp、MLX、Docker Model Runner、LM Studio）。

> **姐妹项目：**
> - [sympozium](https://github.com/sympozium-ai/sympozium/) — 在 Kubernetes 中管理 Agent。
> - [llmserve](https://github.com/AlexsJones/llmserve) — 用于服务本地 LLM 模型的简单 TUI。选择模型、选择后端、开启服务。
> - [llama-panel](https://github.com/AlexsJones/llama-panel) — 用于管理本地 llama-server 实例的原生 macOS 应用。

![演示](assets/demo.gif)

## 文档导航

|  |  |
|---|---|
| **入门指南** | [安装](#安装) · [使用](#使用) · [工作原理](#工作原理) |
| **功能指南** | [TUI 指南](docs/tui.md) · [基准测试分步指南](docs/benchmarking.md) · [CLI 与自动化](docs/cli.md) · [运行时提供商](docs/providers.md) · [OpenClaw 集成](docs/openclaw.md) |
| **技术参考** | [完整工作原理](docs/how-it-works.md) · [平台与 GPU 支持](docs/platform-support.md) · [自定义模型](docs/custom-models.md) · [开发指南](docs/development.md) |
| **项目信息** | [参与贡献](#参与贡献) · [其他替代方案](#其他替代方案) · [代码签名](#代码签名) · [开源许可证](#开源许可证) |

---

## 安装

### Windows
```sh
scoop install llmfit
```

如果尚未安装 Scoop，请参阅 [Scoop 安装指南](https://scoop.sh/)。

### macOS / Linux

#### Homebrew

预编译二进制文件（推荐，适用于所有 macOS/Linux 版本）：
```sh
brew install AlexsJones/llmfit/llmfit
```

或通过 homebrew-core formula 安装（在无预编译 bottle 的 macOS 版本上会从源码构建）：
```sh
brew install llmfit
```

#### MacPorts
```sh
port install llmfit
```

#### 一键脚本安装
```sh
curl -fsSL https://llmfit.axjns.dev/install.sh | sh
```

从 GitHub 下载最新的发布二进制文件并安装至 `/usr/local/bin`（若无 sudo 权限则安装至 `~/.local/bin`）。

**无需 sudo 安装到 `~/.local/bin`：**
```sh
curl -fsSL https://llmfit.axjns.dev/install.sh | sh -s -- --local
```

### uv / pip
安装或更新 llmfit：
```sh
uv tool install -U llmfit
```

免安装直接运行：
```sh
uvx llmfit
```

你也可以像普通 Python 包一样使用 pip 或 uv 进行常规安装。

### Docker / Podman
```sh
docker run ghcr.io/alexsjones/llmfit
```
这将输出 `llmfit recommend` 命令的 JSON 结果，可结合 `jq` 进一步查询：
```sh
podman run ghcr.io/alexsjones/llmfit recommend --use-case coding | jq '.models[].name'
```
如需启动交互式 TUI 界面，请传入全局 `--tui` 参数：
```sh
docker run --rm -it ghcr.io/alexsjones/llmfit --tui
```

### 从源码构建
```sh
git clone https://github.com/AlexsJones/llmfit.git
cd llmfit
cargo build --release
# 二进制文件位于 target/release/llmfit
```

---

## 使用

```sh
llmfit          # 交互式 TUI：检测你的硬件，并对所有模型进行评分排名
```

TUI 界面顶部会显示检测到的硬件配置，并针对每个模型从适配度、速度、质量和上下文四个维度进行打分。有关导航、规划、模拟、下载、社区排行榜和基准测试的说明，请参阅 [TUI 指南](docs/tui.md)。

适用于脚本、Agent 和经典终端输出：

```sh
llmfit fit                    # 按适配度排序的所有模型表格
llmfit recommend --json       # 以 JSON 格式输出推荐模型（供 Agent/脚本调用）
llmfit info "<model>"         # 单个模型：适配分析、估算依据、验证命令
llmfit bench                  # 针对当前运行的提供商实测真实 tok/s 和 TTFT
llmfit doctor                 # 硬件检测报告（用于提交 Issue 诊断）
```

完整参考：[CLI 与自动化](docs/cli.md)。

---

## 工作原理

llmfit 会检测你的系统硬件（RAM、CPU、GPU/显存、后端），然后根据四个维度对目录中的每个模型进行评分：内存适配度、预估速度、质量和上下文。速度估算基于内存带宽模型，并结合运行时采样与真实社区实测数据进行校准 — 每个估算值都会提供输入依据，因此 `llmfit info` 可以准确展示估算所采用的假设参数以及如何在你的机器上进行验证。

详细信息（包含估算公式和模型数据库）：请参阅 [llmfit 工作原理](docs/how-it-works.md)。

---

## 参与贡献

欢迎大家参与贡献，尤其是添加新模型支持。

### 提交 PR 前

在提交更改前，请先运行 `cargo fmt`。CI 检查失败的大多数原因都是代码格式问题：

```sh
cargo fmt
```

添加模型指南（本地免重新构建添加，或添加到内置目录）：请参阅 [自定义模型](docs/custom-models.md)。

---

## 其他替代方案

如果你正在寻找不同的实现方式，可以看看 [llm-checker](https://github.com/Pavelevich/llm-checker) —— 一个集成了 Ollama 的 Node.js CLI 工具，可以直接拉取并对模型进行基准测试。它采取了更实操的方式，即直接在你的硬件上通过 Ollama 实际运行模型，而不是纯根据规格进行预估。如果你已经安装了 Ollama 并且想测试实际运行表现，这是一个不错的选择。需要注意的是，它不支持 MoE（混合专家）架构 —— 所有模型均被视为密集模型（Dense），因此像 Mixtral 或 DeepSeek-V3 这种模型的内存预估将反映总参数量，而非较小的活跃参数子集。

---

## 代码签名

llmfit 的 Windows 发布二进制文件通过 [SignPath.io](https://about.signpath.io/) 进行了数字签名（Authenticode），免费代码签名证书由 [SignPath Foundation](https://signpath.org/) 提供。

签名过程在[发布工作流](.github/workflows/release.yml)中自动进行：仅对由 GitHub Actions 在本仓库构建的工件提交签名，且签名请求需由项目维护者（[@AlexsJones](https://github.com/AlexsJones)）审批。

**代码签名政策：** 请参阅 [SignPath Foundation 代码签名政策与条款](https://signpath.org/terms)。

**隐私说明：** 除非用户或安装/操作人员明确请求，本程序不会向其他网络系统传输任何信息。llmfit 仅在您明确使用相应功能（例如下载模型、查询运行时提供商或访问社区排行榜）时才会连接外部服务。

---

## 开源许可证

MIT
