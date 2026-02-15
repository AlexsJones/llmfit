# Supported Models

llmfit ships with a curated database of 48 LLM models from HuggingFace. All memory estimates assume Q4_K_M quantization (0.5 bytes per parameter) unless noted otherwise.

---

### 01.ai

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [01-ai/Yi-6B-Chat](https://huggingface.co/01-ai/Yi-6B-Chat) | 6.1B | Q4_K_M | 4k | Instruction following, chat |
| [01-ai/Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat) | 34.4B | Q4_K_M | 4k | Instruction following, chat |

### Alibaba

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7.6B | Q4_K_M | 32k | Instruction following, chat |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8.2B | Q4_K_M | 40k | General purpose text generation |
| [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | 14.8B | Q4_K_M | 128k | Instruction following, chat |
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | 32.5B | Q4_K_M | 128k | Instruction following, chat |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 32.8B | Q4_K_M | 40k | General purpose text generation |
| [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | 72.7B | Q4_K_M | 32k | Instruction following, chat |

### BAAI

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | 335M | Q4_K_M | 512 | Text embeddings for RAG |

### BigCode

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [bigcode/starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b) | 7.2B | Q4_K_M | 16k | Code generation and completion |
| [bigcode/starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b) | 15.7B | Q4_K_M | 16k | Code generation and completion |

### Cohere

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01) | 35B | Q4_K_M | 128k | RAG, tool use, agents |

### Community

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | 1.1B | Q4_K_M | 2k | Instruction following, chat |

### DeepSeek

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7.6B | Q4_K_M | 128k | Advanced reasoning, chain-of-thought |
| [deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | 16B | Q4_K_M | 128k | Code generation and completion |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | 32.8B | Q4_K_M | 128k | Advanced reasoning, chain-of-thought |
| [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | 685B | Q4_K_M | 128k | State-of-the-art, MoE architecture |

### Google

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | 2.6B | Q4_K_M | 4k | General purpose text generation |
| [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | 9.2B | Q4_K_M | 4k | General purpose text generation |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | 12B | Q4_K_M | 128k | Multimodal, vision and text |
| [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) | 27.2B | Q4_K_M | 4k | General purpose text generation |

### HuggingFace

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | 7.2B | Q4_K_M | 32k | General purpose text generation |

### Meta

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
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
| [meta-llama/Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) | 405.9B | Q4_K_M | 4k | Instruction following, chat |

### Microsoft

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [microsoft/phi-3-mini-4k-instruct](https://huggingface.co/microsoft/phi-3-mini-4k-instruct) | 3.8B | Q4_K_M | 4k | Lightweight, edge deployment |
| [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.8B | Q4_K_M | 128k | Lightweight, long context |
| [microsoft/Orca-2-7b](https://huggingface.co/microsoft/Orca-2-7b) | 7.0B | Q4_K_M | 4k | Reasoning, step-by-step solutions |
| [microsoft/Orca-2-13b](https://huggingface.co/microsoft/Orca-2-13b) | 13.0B | Q4_K_M | 4k | Reasoning, step-by-step solutions |
| [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) | 14B | Q4_K_M | 16k | Reasoning, STEM, code generation |
| [microsoft/Phi-3-medium-14b-instruct](https://huggingface.co/microsoft/Phi-3-medium-14b-instruct) | 14B | Q4_K_M | 4k | Balanced performance and size |

### Mistral AI

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 7.2B | Q4_K_M | 32k | Instruction following, chat |
| [mistralai/Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) | 8.0B | Q4_K_M | 32k | Instruction following, chat |
| [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) | 12.2B | Q4_K_M | 128k | Instruction following, chat |
| [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | 24B | Q4_K_M | 32k | Instruction following, chat |
| [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 46.7B | Q4_K_M | 32k | Instruction following, chat |

### Nomic

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 137M | F16 | 8k | Text embeddings for RAG |

### NousResearch

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO) | 46.7B | Q4_K_M | 32k | General purpose text generation |

### Stability AI

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [stabilityai/stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) | 1.6B | Q4_K_M | 4k | Instruction following, chat |

### TII

| Model | Parameters | Quantization | Context | Use Case |
|---|---|---|---|---|
| [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) | 7.2B | Q4_K_M | 4k | Instruction following, chat |
