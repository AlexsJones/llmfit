# Supported Models

llmfit ships with a curated database of 94 LLM models from HuggingFace. All memory estimates assume Q4_K_M quantization (0.5 bytes per parameter) unless noted otherwise.

### 01.ai

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [01-ai/Yi-6B-Chat](https://huggingface.co/01-ai/Yi-6B-Chat) | 6.1B | Q4_K_M | 4k | Instruction following, chat |
| [01-ai/Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat) | 34.4B | Q4_K_M | 4k | Instruction following, chat |

### Alibaba

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 600M | Q4_K_M | 40k | Lightweight, edge deployment |
| [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 1.5B | Q4_K_M | 32k | Code generation and completion |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | Q4_K_M | 40k | Lightweight, edge deployment |
| [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3.8B | Q4_K_M | 32k | Multimodal, vision and text |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4.0B | Q4_K_M | 40k | General purpose text generation |
| [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7.6B | Q4_K_M | 32k | Instruction following, chat |
| [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 7.6B | Q4_K_M | 32k | Code generation and completion |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8.2B | Q4_K_M | 40k | General purpose text generation |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 8.3B | Q4_K_M | 32k | Multimodal, vision and text |
| [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | 14.8B | Q4_K_M | 128k | Instruction following, chat |
| [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 14.8B | Q4_K_M | 128k | General purpose text generation |
| [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 14.8B | Q4_K_M | 32k | Code generation and completion |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 30.5B (MoE) | Q4_K_M | 40k | Efficient MoE, general purpose |
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | 32.5B | Q4_K_M | 128k | Instruction following, chat |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 32.8B | Q4_K_M | 40k | General purpose text generation |
| [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | 32.8B | Q4_K_M | 32k | Code generation and completion |
| [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | 72.7B | Q4_K_M | 32k | Instruction following, chat |
| [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | 235B (MoE) | Q4_K_M | 40k | State-of-the-art, MoE architecture |
| [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) | 480B (MoE) | Q4_K_M | 256k | Code generation and completion |

### Allen Institute

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [allenai/OLMo-2-0325-32B-Instruct](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct) | 32B | Q4_K_M | 4k | Fully open-source, instruction following |

### Ant Group

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [inclusionAI/Ling-lite](https://huggingface.co/inclusionAI/Ling-lite) | 16.8B (MoE) | Q4_K_M | 128k | Efficient MoE, general purpose |

### BAAI

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | 335M | Q4_K_M | 512 | Text embeddings for RAG |

### Baidu

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [baidu/ERNIE-4.5-300B-A47B-Paddle](https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-Paddle) | 300B (MoE) | Q4_K_M | 128k | Multilingual, reasoning |

### BigCode

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [bigcode/starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b) | 7.2B | Q4_K_M | 16k | Code generation and completion |
| [bigcode/starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b) | 15.7B | Q4_K_M | 16k | Code generation and completion |

### BigScience

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [bigscience/bloom](https://huggingface.co/bigscience/bloom) | 176B | Q4_K_M | 2k | Multilingual text generation |

### Cohere

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01) | 35B | Q4_K_M | 128k | RAG, tool use, agents |

### Community

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | 1.1B | Q4_K_M | 2k | Instruction following, chat |

### DeepSeek

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7.6B | Q4_K_M | 128k | Advanced reasoning, chain-of-thought |
| [deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | 16B (MoE) | Q4_K_M | 128k | Code generation and completion |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | 32.8B | Q4_K_M | 128k | Advanced reasoning, chain-of-thought |
| [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | 671B (MoE) | Q4_K_M | 128k | Advanced reasoning, chain-of-thought |
| [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | 685B (MoE) | Q4_K_M | 128k | State-of-the-art, MoE architecture |

### Google

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 1B | Q4_K_M | 32k | Lightweight, edge deployment |
| [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | 2.6B | Q4_K_M | 4k | General purpose text generation |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | 4B | Q4_K_M | 128k | Lightweight, general purpose |
| [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | 9.2B | Q4_K_M | 4k | General purpose text generation |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | 12B | Q4_K_M | 128k | Multimodal, vision and text |
| [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 27B | Q4_K_M | 128k | General purpose text generation |
| [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) | 27.2B | Q4_K_M | 4k | General purpose text generation |

### HuggingFace

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | 7.2B | Q4_K_M | 32k | General purpose text generation |

### IBM

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [ibm-granite/granite-4.0-h-micro](https://huggingface.co/ibm-granite/granite-4.0-h-micro) | 3B | Q4_K_M | 128k | Enterprise, hybrid Mamba/transformer |
| [ibm-granite/granite-4.0-h-tiny](https://huggingface.co/ibm-granite/granite-4.0-h-tiny) | 7B (MoE) | Q4_K_M | 128k | Enterprise, hybrid Mamba/transformer |
| [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) | 8.1B | Q4_K_M | 128k | Enterprise, instruction following |
| [ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small) | 32B (MoE) | Q4_K_M | 128k | Enterprise, hybrid Mamba/transformer |

### LMSYS

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | 7.0B | Q4_K_M | 4k | Instruction following, chat |
| [lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) | 13.0B | Q4_K_M | 4k | Instruction following, chat |

### Meituan

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [meituan/LongCat-Flash](https://huggingface.co/meituan/LongCat-Flash) | 560B (MoE) | Q4_K_M | 512k | Long context MoE |

### Meta

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1.2B | Q4_K_M | 4k | General purpose text generation |
| [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | 3.2B | Q4_K_M | 4k | General purpose text generation |
| [meta-llama/CodeLlama-7b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf) | 6.7B | Q4_K_M | 4k | Code generation and completion |
| [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | 8.0B | Q4_K_M | 4k | General purpose text generation |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8.0B | Q4_K_M | 4k | Instruction following, chat |
| [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | 10.7B | Q4_K_M | 4k | Instruction following, chat |
| [meta-llama/CodeLlama-13b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf) | 13.0B | Q4_K_M | 4k | Code generation and completion |
| [meta-llama/CodeLlama-34b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf) | 33.7B | Q4_K_M | 4k | Code generation and completion |
| [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) | 70.6B | Q4_K_M | 4k | Instruction following, chat |
| [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | 70.6B | Q4_K_M | 128k | Instruction following, chat |
| [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | 109B (MoE) | Q4_K_M | 128k | Multimodal, vision and text |
| [meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) | 400B (MoE) | Q4_K_M | 128k | Multimodal, vision and text |
| [meta-llama/Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) | 405.9B | Q4_K_M | 4k | Instruction following, chat |

### Microsoft

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [microsoft/phi-3-mini-4k-instruct](https://huggingface.co/microsoft/phi-3-mini-4k-instruct) | 3.8B | Q4_K_M | 4k | Lightweight, edge deployment |
| [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.8B | Q4_K_M | 128k | Lightweight, long context |
| [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.8B | Q4_K_M | 128k | Lightweight, edge deployment |
| [microsoft/Orca-2-7b](https://huggingface.co/microsoft/Orca-2-7b) | 7.0B | Q4_K_M | 4k | Reasoning, step-by-step solutions |
| [microsoft/Orca-2-13b](https://huggingface.co/microsoft/Orca-2-13b) | 13.0B | Q4_K_M | 4k | Reasoning, step-by-step solutions |
| [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) | 14B | Q4_K_M | 16k | Reasoning, STEM, code generation |
| [microsoft/Phi-3-medium-14b-instruct](https://huggingface.co/microsoft/Phi-3-medium-14b-instruct) | 14B | Q4_K_M | 4k | Balanced performance and size |

### Mistral AI

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 7.2B | Q4_K_M | 32k | Instruction following, chat |
| [mistralai/Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) | 8.0B | Q4_K_M | 32k | Instruction following, chat |
| [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) | 12.2B | Q4_K_M | 128k | Instruction following, chat |
| [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | 24B | Q4_K_M | 32k | Instruction following, chat |
| [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) | 24B | Q4_K_M | 128k | Multimodal, vision and text |
| [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 46.7B (MoE) | Q4_K_M | 32k | Instruction following, chat |
| [mistralai/Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) | 123B | Q4_K_M | 128k | Large-scale instruction following |
| [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) | 140.6B (MoE) | Q4_K_M | 64k | Large MoE, instruction following |

### Moonshot

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct) | 1000B (MoE) | Q4_K_M | 128k | Large MoE, reasoning |

### Nomic

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 137M | F16 | 8k | Text embeddings for RAG |

### NousResearch

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO) | 46.7B (MoE) | Q4_K_M | 32k | General purpose text generation |

### OpenChat

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [openchat/openchat-3.5-0106](https://huggingface.co/openchat/openchat-3.5-0106) | 7.0B | Q4_K_M | 8k | Instruction following, chat |

### Rednote

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [rednote-hilab/dots.llm1.inst](https://huggingface.co/rednote-hilab/dots.llm1.inst) | 142B (MoE) | Q4_K_M | 128k | MoE, general purpose |

### Stability AI

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [stabilityai/stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) | 1.6B | Q4_K_M | 4k | Instruction following, chat |

### TII

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) | 7.2B | Q4_K_M | 4k | Instruction following, chat |
| [tiiuae/Falcon3-7B-Instruct](https://huggingface.co/tiiuae/Falcon3-7B-Instruct) | 7.5B | Q4_K_M | 32k | Instruction following, chat |
| [tiiuae/Falcon3-10B-Instruct](https://huggingface.co/tiiuae/Falcon3-10B-Instruct) | 10.3B | Q4_K_M | 32k | Instruction following, chat |
| [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) | 40.0B | Q4_K_M | 2k | Instruction following, chat |
| [tiiuae/falcon-180B-chat](https://huggingface.co/tiiuae/falcon-180B-chat) | 180B | Q4_K_M | 2k | Large-scale instruction following |

### Upstage

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) | 10.7B | Q4_K_M | 4k | High-performance instruction following |

### WizardLM

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [WizardLMTeam/WizardLM-13B-V1.2](https://huggingface.co/WizardLMTeam/WizardLM-13B-V1.2) | 13.0B | Q4_K_M | 4k | Instruction following, chat |
| [WizardLMTeam/WizardCoder-15B-V1.0](https://huggingface.co/WizardLMTeam/WizardCoder-15B-V1.0) | 15.5B | Q4_K_M | 8k | Code generation and completion |

### xAI

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [xai-org/grok-1](https://huggingface.co/xai-org/grok-1) | 314B (MoE) | Q4_K_M | 8k | Large MoE, general purpose |

### Zhipu AI

| Model | Parameters | Quantization | Context | Use Case |
|-------|-----------|--------------|---------|----------|
| [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) | 9B | Q4_K_M | 128k | Multilingual, instruction following |
