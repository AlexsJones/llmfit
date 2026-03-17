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

**支持成百上千种模型与提供商。只需一条命令，即可找出最适合你硬件的模型。**

`llmfit` 是一款可以根据你的系统 RAM、CPU 和 GPU，自动为您匹配合适尺寸 (Right-sizes) LLM 模型的终端工具。它会自动检测你的硬件，并从质量、速度、匹配度和上下文维度为每个模型打分，告诉您哪些模型在您的机器上运行得最顺畅。

该工具内置了交互式 TUI（默认）和经典的 CLI 模式。支持多 GPU 设置、混合专家 (MoE) 架构、动态量化选择、速度预估，以及本地运行时提供商（Ollama, llama.cpp, MLX, Docker Model Runner）。

> **姐妹项目：** 欢迎查看 [sympozium](https://github.com/AlexsJones/sympozium/)，用于管理 Kubernetes 中的 Agent。

![演示](demo.gif)

---

## 安装

### Windows
```sh
scoop install llmfit
```

### macOS
```sh
brew tap AlexsJones/llmfit
brew install llmfit
```

### 从源码安装 (Rust)
```sh
cargo install llmfit
```

---

## 核心功能

- **硬件检测**：自动识别你的 VRAM, RAM, CPU 核心以及 GPU 架构（包括 Apple Silicon M1-M5, NVIDIA CUDA, AMD ROCm）。
- **模型打分**：根据性能、量化损失和速度，为每个模型提供 0-100 分的适配度评分。
- **速度预测**：估算预填充 (PP) 和文本生成 (TG) 的 Token/s 速度。
- **量化建议**：自动建议最适合你显存的量化位数（4-bit, 8-bit 等）。
- **多运行时支持**：无缝对接本地的 Ollama, llama.cpp, MLX 实例。

## 使用方法

### TUI 模式 (推荐)
直接运行：
```sh
llmfit
```
使用方向键浏览，回车键查看详细分析。

### CLI 模式
搜索特定模型：
```sh
llmfit search "llama-3"
```
查看当前硬件摘要：
```sh
llmfit stats
```

## 贡献

欢迎通过提交 Issue、Pull Request 来参与贡献！

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

*本文档由 [@JasonYeYuhe](https://github.com/JasonYeYuhe) 翻译并维护。如果您发现任何翻译问题或需要增加新特性说明，欢迎提交 Issue 或与我联系。*
