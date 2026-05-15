#!/usr/bin/env python3
"""
Scraper for popular LLM models from Hugging Face.
Fetches model metadata and computes RAM/VRAM requirements from parameter counts.
Outputs a JSON file consumable by llmfit's models.rs.

Usage:
  python3 scrape_hf_models.py                  # Curated + top 1000 by downloads
  python3 scrape_hf_models.py --threads 8      # Same, with parallel fetches
  python3 scrape_hf_models.py -n 500           # Curated + top 500 by downloads
  python3 scrape_hf_models.py --no-discover     # Curated list only
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.request
import urllib.error

HF_API = "https://huggingface.co/api/models"

# Global auth token, set from --token flag or HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var
_hf_token: str | None = None


def _auth_headers() -> dict[str, str]:
    """Return HTTP headers with auth if a HuggingFace token is available."""
    headers = {"User-Agent": "llmfit-scraper/1.0"}
    if _hf_token:
        headers["Authorization"] = f"Bearer {_hf_token}"
    return headers

# Top text-generation models to scrape (owner/repo)
TARGET_MODELS = [
    # Meta Llama family
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",  # NEW: Multimodal vision model
    "meta-llama/Llama-3.3-70B-Instruct",
    # Meta Llama 4 (MoE)
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # Code Llama
    "meta-llama/CodeLlama-7b-Instruct-hf",  # NEW: Popular code model
    "meta-llama/CodeLlama-13b-Instruct-hf",  # NEW: Larger code model
    "meta-llama/CodeLlama-34b-Instruct-hf",  # NEW: Large code model
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Mistral-Large-Instruct-2407",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-Nemo-Instruct-2407",
    # Qwen
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",  # NEW: Ultra-lightweight coder
    "Qwen/Qwen2.5-Coder-7B-Instruct",     # NEW: Popular coder
    "Qwen/Qwen2.5-Coder-14B-Instruct",    # NEW: Mid-size coder
    "Qwen/Qwen2.5-Coder-32B-Instruct",    # NEW: Large coder
    "Qwen/Qwen2.5-VL-3B-Instruct",        # NEW: Vision-language 3B
    "Qwen/Qwen2.5-VL-7B-Instruct",        # NEW: Vision-language 7B
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Qwen/Qwen3-Coder-Next",
    # Qwen 3.5 (native multimodal, Feb 2026)
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
    # Qwen3.5 Small Series (Instruct)
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    # Qwen3.5 Small Series (Base)
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B-Base",
    # Qwen 3.5 (Claude Opus 4.6 reasoning, Feb 2026)
    "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
    "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
    "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2",
    "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF",
    "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
    "Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled",
    # Qwen 3.6 (native multimodal + hybrid attention, Apr 2026)
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.6-35B-A3B",
    # Microsoft Phi
    "microsoft/phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-14b-instruct",
    "microsoft/Phi-3.5-mini-instruct",  # NEW: Newer Phi variant
    "microsoft/phi-4",
    "microsoft/Phi-4-mini-instruct",
    # Microsoft Orca
    "microsoft/Orca-2-7b",  # NEW: Reasoning model
    "microsoft/Orca-2-13b",  # NEW: Larger reasoning model
    # Google Gemma
    "google/gemma-2-2b-it",  # NEW: Smaller variant for edge
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    # Google Gemma 4
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-31B-it",
    "google/gemma-4-26B-A4B-it",
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    # DeepSeek V4 family (MoE, hybrid attention, Apr 2026)
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Pro-Base",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4-Flash-Base",
    # Cohere
    "CohereForAI/c4ai-command-r-v01",
    # 01.ai Yi family
    "01-ai/Yi-6B-Chat",  # NEW: Popular multilingual 6B
    "01-ai/Yi-34B-Chat",  # NEW: Popular multilingual 34B
    # Upstage Solar
    "upstage/SOLAR-10.7B-Instruct-v1.0",  # NEW: High-performance 10.7B
    # TII Falcon
    "tiiuae/falcon-7b-instruct",  # NEW: Popular UAE model
    "tiiuae/falcon-40b-instruct",
    "tiiuae/falcon-180B-chat",
    "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-10B-Instruct",
    # HuggingFace Zephyr
    "HuggingFaceH4/zephyr-7b-beta",  # NEW: Very popular fine-tune
    # OpenChat
    "openchat/openchat-3.5-0106",  # NEW: Popular alternative
    # LMSYS Vicuna
    "lmsys/vicuna-7b-v1.5",  # NEW: Popular community model
    "lmsys/vicuna-13b-v1.5",  # NEW: Larger Vicuna
    # NousResearch
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",  # NEW: Popular fine-tune
    # WizardLM
    "WizardLMTeam/WizardLM-13B-V1.2",  # NEW: Popular instruction model
    # Code models
    "bigcode/starcoder2-7b",
    "bigcode/starcoder2-15b",
    "WizardLMTeam/WizardCoder-15B-V1.0",  # NEW: Code specialist
    # Small / edge models
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stabilityai/stablelm-2-1_6b-chat",
    # IBM Granite
    "ibm-granite/granite-3.1-8b-instruct",
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-h-micro",
    "ibm-granite/granite-4.0-h-small",
    # Allen Institute OLMo
    "allenai/OLMo-2-0325-32B-Instruct",
    # Zhipu GLM
    "THUDM/glm-4-9b-chat",
    # xAI Grok
    "xai-org/grok-1",
    # Moonshot Kimi
    "moonshotai/Kimi-K2-Instruct",
    # BigScience BLOOM
    "bigscience/bloom",
    # Baidu ERNIE
    "baidu/ERNIE-4.5-300B-A47B-Paddle",
    # Rednote dots.llm
    "rednote-hilab/dots.llm1.inst",
    # Meituan LongCat
    "meituan/LongCat-Flash",
    # Ant Group Ling
    "inclusionAI/Ling-lite",
    # Liquid AI LFM2 (dense)
    "LiquidAI/LFM2-350M",
    "LiquidAI/LFM2-700M",
    "LiquidAI/LFM2-1.2B",
    "LiquidAI/LFM2-2.6B",
    "LiquidAI/LFM2-2.6B-Exp",
    # Liquid AI LFM2 (MoE)
    "LiquidAI/LFM2-8B-A1B",
    "LiquidAI/LFM2-24B-A2B",
    # Liquid AI LFM2.5
    "LiquidAI/LFM2.5-1.2B-Base",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "LiquidAI/LFM2.5-1.2B-Thinking",
    "LiquidAI/LFM2.5-1.2B-JP",
    # Liquid AI LFM2 Vision-Language
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2-VL-1.6B",
    "LiquidAI/LFM2-VL-3B",
    "LiquidAI/LFM2.5-VL-1.6B",
    # Liquid AI LFM2 Audio
    "LiquidAI/LFM2-Audio-1.5B",
    "LiquidAI/LFM2.5-Audio-1.5B",
    # Liquid AI Liquid Nanos (task-specific fine-tunes)
    "LiquidAI/LFM2-1.2B-Tool",
    "LiquidAI/LFM2-1.2B-RAG",
    "LiquidAI/LFM2-1.2B-Extract",
    "LiquidAI/LFM2-350M-Extract",
    "LiquidAI/LFM2-350M-Math",
    "LiquidAI/LFM2-350M-ENJP-MT",
    "LiquidAI/LFM2-350M-PII-Extract-JP",
    "LiquidAI/LFM2-ColBERT-350M",
    "LiquidAI/LFM2-2.6B-Transcript",
    # Embeddings (useful for RAG sizing)
    "nomic-ai/nomic-embed-text-v1.5",
    "BAAI/bge-large-en-v1.5",
    # --- New models added Feb 2026 ---
    # DeepSeek V3.2 family
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V3.2-Speciale",
    # Zhipu/Z.ai GLM-5
    "zai-org/GLM-5",
    # Moonshot Kimi K2.5
    "moonshotai/Kimi-K2.5",
    # MiniMax M2.7 / M2.5
    "MiniMaxAI/MiniMax-M2.7",
    "MiniMaxAI/MiniMax-M2.5",
    # Xiaomi MiMo
    "XiaomiMiMo/MiMo-V2-Flash",
    "XiaomiMiMo/MiMo-7B-RL",
    # NVIDIA Nemotron
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    # Microsoft Phi-4 reasoning family
    "microsoft/Phi-4-reasoning",
    "microsoft/Phi-4-mini-reasoning",
    "microsoft/Phi-4-multimodal-instruct",
    # LG AI EXAONE 4.0
    "LGAI-EXAONE/EXAONE-4.0-32B",
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    # HuggingFace SmolLM3
    "HuggingFaceTB/SmolLM3-3B",
    # Google Gemma 3n (effective parameter models)
    "google/gemma-3n-E4B-it",
    "google/gemma-3n-E2B-it",
    # RWKV v7 — pure RNN/SSM, no KV cache (GGUF native via shoumenchougou)
    "shoumenchougou/RWKV7-G1f-1.5B-GGUF",
    "shoumenchougou/RWKV7-G1f-2.9B-GGUF",
    "shoumenchougou/RWKV7-G1f-7.2B-GGUF",
    "shoumenchougou/RWKV7-G1f-13.3B-GGUF",
]

# Bytes-per-parameter for different quantization levels
QUANT_BPP = {
    "F32":    4.0,
    "F16":    2.0,
    "BF16":   2.0,
    "Q8_0":   1.0,
    "Q6_K":   0.75,
    "Q5_K_M": 0.625,
    "Q4_K_M": 0.5,
    "Q4_0":   0.5,
    "Q3_K_M": 0.4375,
    "Q2_K":   0.3125,
    "AWQ-4bit": 0.5,
    "AWQ-8bit": 1.0,
    "GPTQ-Int4": 0.5,
    "GPTQ-Int8": 1.0,
}

# Overhead multiplier for runtime memory beyond just model weights
RUNTIME_OVERHEAD = 1.2  # ~20% overhead for KV cache, activations, OS

# Known MoE (Mixture of Experts) architecture configurations
MOE_CONFIGS = {
    "mixtral": {"num_experts": 8, "active_experts": 2},
    "deepseek_v2": {"num_experts": 64, "active_experts": 6},
    "deepseek_v3": {"num_experts": 256, "active_experts": 8},
    "deepseek_v4": {"num_experts": 384, "active_experts": 6},
    "qwen3_moe": {"num_experts": 128, "active_experts": 8},
    "llama4": {"num_experts": 16, "active_experts": 1},
    "grok": {"num_experts": 8, "active_experts": 2},
    "glm5": {"num_experts": 256, "active_experts": 8},
    "minimax_m2": {"num_experts": 32, "active_experts": 2},
    "mimo_v2": {"num_experts": 128, "active_experts": 8},
    "nemotron3_nano": {"num_experts": 128, "active_experts": 6},
    "qwen3_5_moe": {"num_experts": 256, "active_experts": 8},
    "qwen3_vl_moe": {"num_experts": 256, "active_experts": 8},
}

# Published active parameter counts for well-known MoE models
MOE_ACTIVE_PARAMS = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 12_900_000_000,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 39_100_000_000,
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 12_900_000_000,
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 2_400_000_000,
    "deepseek-ai/DeepSeek-V3": 37_000_000_000,
    "deepseek-ai/DeepSeek-R1": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3.2": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3.2-Speciale": 37_000_000_000,
    "deepseek-ai/DeepSeek-V4-Pro": 49_000_000_000,
    "deepseek-ai/DeepSeek-V4-Pro-Base": 49_000_000_000,
    "deepseek-ai/DeepSeek-V4-Flash": 13_000_000_000,
    "deepseek-ai/DeepSeek-V4-Flash-Base": 13_000_000_000,
    "Qwen/Qwen3-30B-A3B": 3_300_000_000,
    "Qwen/Qwen3-235B-A22B": 22_000_000_000,
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": 35_000_000_000,
    "Qwen/Qwen3-Coder-Next": 3_000_000_000,
    "Qwen/Qwen3.5-35B-A3B": 3_000_000_000,
    "Qwen/Qwen3.5-122B-A10B": 10_000_000_000,
    "Qwen/Qwen3.5-397B-A17B": 17_000_000_000,
    "Qwen/Qwen3.6-35B-A3B": 3_000_000_000,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 17_000_000_000,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 17_000_000_000,
    "xai-org/grok-1": 86_000_000_000,
    "moonshotai/Kimi-K2-Instruct": 32_000_000_000,
    "moonshotai/Kimi-K2.5": 32_000_000_000,
    "zai-org/GLM-5": 40_000_000_000,
    "MiniMaxAI/MiniMax-M2.7": 10_000_000_000,
    "MiniMaxAI/MiniMax-M2.5": 10_000_000_000,
    "XiaomiMiMo/MiMo-V2-Flash": 15_000_000_000,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 3_000_000_000,
    "LiquidAI/LFM2-8B-A1B": 1_500_000_000,
    "LiquidAI/LFM2-24B-A2B": 2_300_000_000,  # 23.8B total, 2.3B active
    "google/gemma-4-26B-A4B-it": 4_000_000_000,
}


def fetch_model_info(repo_id: str) -> dict | None:
    """Fetch model info from HuggingFace API."""
    url = f"{HF_API}/{repo_id}"
    req = urllib.request.Request(url, headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401 and not _hf_token:
            print(f"  ⚠ HTTP 401 for {repo_id} — model is gated, set HF_TOKEN to access",
                  file=sys.stderr)
        else:
            print(f"  ⚠ HTTP {e.code} for {repo_id} — skipping", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ⚠ Error fetching {repo_id}: {e}", file=sys.stderr)
        return None


def format_param_count(total_params: int) -> str:
    """Convert raw parameter count into human-readable string."""
    if total_params >= 1_000_000_000:
        val = total_params / 1_000_000_000
        return f"{val:.1f}B" if val != int(val) else f"{int(val)}B"
    elif total_params >= 1_000_000:
        val = total_params / 1_000_000
        return f"{val:.0f}M"
    else:
        return f"{total_params / 1_000:.0f}K"


def estimate_ram(total_params: int, quant: str) -> tuple[float, float]:
    """
    Estimate min RAM (Q4 quantized) and recommended RAM (comfortable headroom).
    Returns (min_ram_gb, recommended_ram_gb).
    """
    bpp = QUANT_BPP.get(quant, 0.5)
    model_size_gb = (total_params * bpp) / (1024**3)
    min_ram_gb = model_size_gb * RUNTIME_OVERHEAD
    # Recommended: enough for Q4 + generous KV cache + OS headroom
    recommended_ram_gb = model_size_gb * 2.0

    # Apply sensible floor
    min_ram_gb = max(min_ram_gb, 1.0)
    recommended_ram_gb = max(recommended_ram_gb, 2.0)

    return round(min_ram_gb, 1), round(recommended_ram_gb, 1)


def estimate_vram(total_params: int, quant: str) -> float:
    """Estimate minimum VRAM to fit model weights on GPU."""
    bpp = QUANT_BPP.get(quant, 0.5)
    model_size_gb = (total_params * bpp) / (1024**3)
    # VRAM needs to hold weights + some activation memory
    vram_gb = model_size_gb * 1.1
    return round(max(vram_gb, 0.5), 1)


def extract_arch_metadata(config: dict | None) -> dict:
    """Extract architecture fields for precise KV cache and MoE speed estimation.

    Checks top-level config and falls back to ``text_config`` for multimodal
    models (e.g. Llama 4 Scout, Qwen-VL).  Returns a dict with architecture
    fields (any may be ``None``).
    """
    cfg = config or {}
    # Try top-level first, then text_config for multimodal wrappers.
    sources = [cfg]
    if isinstance(cfg.get("text_config"), dict):
        sources.append(cfg["text_config"])

    num_hidden_layers = None
    num_attention_heads = None
    num_key_value_heads = None
    head_dim = None
    hidden_size = None
    vocab_size = None
    moe_intermediate_size = None
    shared_expert_intermediate_size = None

    for src in sources:
        if num_hidden_layers is None:
            num_hidden_layers = src.get("num_hidden_layers")
        if num_attention_heads is None:
            num_attention_heads = src.get("num_attention_heads")
        if num_key_value_heads is None:
            num_key_value_heads = src.get("num_key_value_heads")
        if head_dim is None:
            head_dim = src.get("head_dim")
        if hidden_size is None:
            hidden_size = src.get("hidden_size")
        if head_dim is None and num_attention_heads and hidden_size:
            head_dim = hidden_size // num_attention_heads
        if vocab_size is None:
            vocab_size = src.get("vocab_size")
        if moe_intermediate_size is None:
            # Prefer explicit moe_intermediate_size (Qwen, DeepSeek), fall
            # back to intermediate_size which is the per-expert FFN dim in
            # Mixtral-style MoE models that don't use a separate key.
            v = src.get("moe_intermediate_size") or src.get("intermediate_size")
            # Some models (e.g. ERNIE-4.5-VL) use a list; take first element.
            if isinstance(v, list):
                v = v[0] if v else None
            moe_intermediate_size = v
        if shared_expert_intermediate_size is None:
            v = src.get("shared_expert_intermediate_size")
            if isinstance(v, list):
                v = v[0] if v else None
            shared_expert_intermediate_size = v

    # GQA default: if num_key_value_heads missing, assume MHA
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    return {
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "moe_intermediate_size": moe_intermediate_size,
        "shared_expert_intermediate_size": shared_expert_intermediate_size,
    }


def detect_moe(repo_id: str, config: dict | None, architecture: str,
               total_params: int) -> dict:
    """Detect MoE architecture and compute active parameters."""
    result = {
        "is_moe": False,
        "num_experts": None,
        "active_experts": None,
        "active_parameters": None,
    }

    # Check config.json for MoE indicators (also check text_config for
    # multimodal models like Llama 4 that nest MoE fields there)
    num_experts = None
    active_experts = None
    if config:
        num_experts = config.get("num_local_experts") or config.get("num_experts") or config.get("n_routed_experts")
        active_experts = config.get("num_experts_per_tok") or config.get("top_k_experts")
        if (not num_experts or not active_experts) and isinstance(config.get("text_config"), dict):
            tc = config["text_config"]
            num_experts = num_experts or tc.get("num_local_experts") or tc.get("num_experts") or tc.get("n_routed_experts")
            active_experts = active_experts or tc.get("num_experts_per_tok") or tc.get("top_k_experts")

    # Check if architecture is in known MoE configs
    if architecture in MOE_CONFIGS:
        moe = MOE_CONFIGS[architecture]
        num_experts = num_experts or moe["num_experts"]
        active_experts = active_experts or moe["active_experts"]

    if num_experts and active_experts:
        result["is_moe"] = True
        result["num_experts"] = num_experts
        result["active_experts"] = active_experts

        # Use published active params if known, otherwise estimate
        if repo_id in MOE_ACTIVE_PARAMS:
            result["active_parameters"] = MOE_ACTIVE_PARAMS[repo_id]
        else:
            result["active_parameters"] = estimate_active_params(
                total_params, num_experts, active_experts)

    return result


def estimate_active_params(total_params: int, num_experts: int,
                           active_experts: int) -> int:
    """Estimate active parameters for MoE models.

    Assumes expert MLP layers are ~95% of total params and
    shared attention/embedding layers are ~5%.
    """
    shared_fraction = 0.05
    shared = int(total_params * shared_fraction)
    expert_pool = total_params - shared
    per_expert = expert_pool // num_experts
    return shared + active_experts * per_expert


def estimate_params_from_arch(config: dict | None) -> int | None:
    """Estimate total parameter count from architecture metadata.

    Uses the transformer parameter formula accounting for MoE expert weights.
    Returns None if insufficient metadata is available.
    """
    cfg = config or {}
    # Check text_config for multimodal wrappers
    for src in [cfg, cfg.get("text_config", {})]:
        hidden = src.get("hidden_size")
        layers = src.get("num_hidden_layers")
        vocab = src.get("vocab_size")
        if hidden and layers and vocab:
            break
    else:
        return None

    n_heads = src.get("num_attention_heads") or 1
    n_kv = src.get("num_key_value_heads") or n_heads
    head_dim = src.get("head_dim") or (hidden // n_heads if n_heads else hidden)

    # Attention: Q + K + V + O projections per layer
    attn = 2 * n_heads * head_dim * hidden + 2 * n_kv * head_dim * hidden

    # FFN / expert weights
    def _scalar(v, default=None):
        """Coerce list values (e.g. ERNIE-4.5-VL) to a single int."""
        if isinstance(v, list):
            return v[0] if v else default
        return v if v is not None else default

    num_experts = src.get("num_local_experts") or src.get("num_experts")
    moe_inter = _scalar(src.get("moe_intermediate_size"))
    shared_inter = _scalar(src.get("shared_expert_intermediate_size"), 0)
    intermediate = _scalar(src.get("intermediate_size"))

    if num_experts and moe_inter:
        # MoE: per-expert FFN + shared expert
        expert_ffn = num_experts * 3 * hidden * moe_inter
        shared_ffn = 3 * hidden * shared_inter if shared_inter else 0
        router = num_experts * hidden
        ffn_total = expert_ffn + shared_ffn + router
    elif intermediate:
        # Dense: standard SwiGLU FFN (gate + up + down)
        ffn_total = 3 * hidden * intermediate
    else:
        # Fallback: assume 4x hidden
        ffn_total = 4 * hidden * hidden

    per_layer = attn + ffn_total
    embedding = 2 * vocab * hidden  # embedding + lm_head

    total = layers * per_layer + embedding
    return total if total > 1_000_000 else None


def infer_use_case(repo_id: str, pipeline_tag: str | None, config: dict | None) -> str:
    """Infer a brief use-case description from model metadata."""
    rid = repo_id.lower()
    if "embed" in rid or "bge" in rid:
        return "Text embeddings for RAG"
    if "coder" in rid or "starcoder" in rid or "code" in rid:
        return "Code generation and completion"
    if "r1" in rid or "reason" in rid:
        return "Advanced reasoning, chain-of-thought"
    if "instruct" in rid or "chat" in rid:
        return "Instruction following, chat"
    if "tiny" in rid or "small" in rid or "mini" in rid:
        return "Lightweight, edge deployment"
    if pipeline_tag == "text-generation":
        return "General purpose text generation"
    return "General purpose"


def infer_context_length(config: dict | None) -> int:
    """Try to extract context length from model config."""
    if not config:
        return 4096

    # Common config keys for max sequence length
    keys_to_check = [
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
        "sliding_window",
    ]

    def _extract_from(cfg: dict) -> int | None:
        for key in keys_to_check:
            if key in cfg:
                val = cfg[key]
                if isinstance(val, int) and val > 0:
                    return val
        return None

    def _apply_rope_scaling(val: int, cfg: dict) -> int:
        """Apply RoPE scaling factor when present (e.g., Llama 4 Maverick
        has max_position_embeddings=4096 but a rope_scaling factor of 256,
        giving an effective context of 1M tokens)."""
        rope = cfg.get("rope_scaling")
        if isinstance(rope, dict) and isinstance(rope.get("factor"), (int, float)):
            scaled = int(val * rope["factor"])
            if scaled > val:
                return scaled
        return val

    # Check top-level config
    val = _extract_from(config)
    if val is not None:
        return _apply_rope_scaling(val, config)

    # For multimodal models (e.g., Qwen3.5), check text_config
    if "text_config" in config and isinstance(config["text_config"], dict):
        tc = config["text_config"]
        val = _extract_from(tc)
        if val is not None:
            return _apply_rope_scaling(val, tc)

    return 4096


def fetch_config_json(repo_id: str) -> dict | None:
    """Fetch the full config.json from a HF repo (has max_position_embeddings)."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def extract_provider(repo_id: str) -> str:
    """Map HF org name to a friendly provider name."""
    org = repo_id.split("/")[0].lower()
    mapping = {
        "meta-llama": "Meta",
        "mistralai": "Mistral AI",
        "qwen": "Alibaba",
        "microsoft": "Microsoft",
        "google": "Google",
        "deepseek-ai": "DeepSeek",
        "bigcode": "BigCode",
        "cohereforai": "Cohere",
        "tinyllama": "Community",
        "stabilityai": "Stability AI",
        "nomic-ai": "Nomic",
        "baai": "BAAI",
        "01-ai": "01.ai",  # NEW
        "upstage": "Upstage",  # NEW
        "tiiuae": "TII",  # NEW
        "huggingfaceh4": "HuggingFace",  # NEW
        "openchat": "OpenChat",  # NEW
        "lmsys": "LMSYS",  # NEW
        "nousresearch": "NousResearch",  # NEW
        "wizardlmteam": "WizardLM",  # NEW
        "liquidai": "Liquid AI",
    }
    return mapping.get(org, org)


def infer_capabilities(repo_id: str, pipeline_tag: str | None, use_case: str) -> list[str]:
    """Infer model capabilities like vision and tool use."""
    caps: list[str] = []
    rid = repo_id.lower()
    uc = use_case.lower()

    # Vision
    if (
        pipeline_tag == "image-text-to-text"
        or pipeline_tag == "any-to-any"
        or "vision" in rid
        or "-vl-" in rid
        or rid.endswith("-vl")
        or "llava" in rid
        or "onevision" in rid
        or "pixtral" in rid
        or "vision" in uc
        or "multimodal" in uc
    ):
        caps.append("vision")

    # Tool use (known families)
    if (
        "tool" in uc
        or "function call" in uc
        or "qwen3" in rid
        or "qwen2.5" in rid
        or "command-r" in rid
        or ("llama-3" in rid and "instruct" in rid)
        or ("mistral" in rid and "instruct" in rid)
        or "hermes" in rid
        or ("gemma-3" in rid and rid.endswith("-it"))
        or ("gemma-4" in rid and rid.endswith("-it"))
    ):
        caps.append("tool_use")

    return caps


def detect_quant_format(repo_id: str, config: dict | None) -> tuple[str, str]:
    """Detect quantization format and label from config.json.

    Returns (format, quant_label) where:
    - format: "gguf", "awq", "gptq", "mlx", or "safetensors"
    - quant_label: e.g. "AWQ-4bit", "GPTQ-Int4", "Q4_K_M"
    """
    if not config:
        return _detect_format_from_name(repo_id)

    quant_config = config.get("quantization_config", {})
    if not quant_config:
        return _detect_format_from_name(repo_id)

    quant_method = quant_config.get("quant_method", "")
    bits = quant_config.get("bits", quant_config.get("num_bits", 4))

    # AWQ
    if quant_method == "awq":
        label = f"AWQ-{bits}bit"
        return ("awq", label)

    # GPTQ (including gptq_marlin)
    if quant_method.startswith("gptq"):
        label = f"GPTQ-Int{bits}"
        return ("gptq", label)

    # AutoRound — pre-quantized safetensors, cannot be dynamically re-quantized
    if quant_method == "auto-round":
        label = f"AutoRound-{bits}bit"
        return ("autoround", label)

    # compressed-tensors: dig into config_groups for bits, check name for format
    if quant_method == "compressed-tensors":
        # Try to extract bits from config_groups
        config_groups = quant_config.get("config_groups", {})
        for group in config_groups.values():
            if isinstance(group, dict):
                weights = group.get("weights", {})
                if "num_bits" in weights:
                    bits = weights["num_bits"]
                    break

        name_upper = repo_id.upper()
        if "-AWQ" in name_upper:
            label = f"AWQ-{bits}bit"
            return ("awq", label)
        elif "-GPTQ" in name_upper:
            label = f"GPTQ-Int{bits}"
            return ("gptq", label)
        elif "-AUTOROUND" in name_upper:
            label = f"AutoRound-{bits}bit"
            return ("autoround", label)

    return _detect_format_from_name(repo_id)


def _detect_format_from_name(repo_id: str) -> tuple[str, str]:
    """Fallback: detect format from model name patterns."""
    name_upper = repo_id.upper()

    if "-AWQ-8BIT" in name_upper:
        return ("awq", "AWQ-8bit")
    if "-AWQ" in name_upper:
        return ("awq", "AWQ-4bit")
    if "-GPTQ-INT8" in name_upper or "-GPTQ-8BIT" in name_upper:
        return ("gptq", "GPTQ-Int8")
    if "-GPTQ" in name_upper:
        return ("gptq", "GPTQ-Int4")
    if "-AUTOROUND" in name_upper:
        return ("autoround", "AutoRound-4bit")
    if "-MLX-" in name_upper or name_upper.endswith("-MLX"):
        return ("mlx", "Q4_K_M")  # MLX uses its own quant scheme handled elsewhere

    return ("gguf", "Q4_K_M")


def scrape_model(repo_id: str) -> dict | None:
    """Scrape a single model and return an LlmModel-compatible dict."""
    info = fetch_model_info(repo_id)
    if not info:
        return None

    # Extract parameter count from safetensors metadata
    safetensors = info.get("safetensors", {})
    total_params = safetensors.get("total")
    if not total_params:
        params_by_dtype = safetensors.get("parameters", {})
        if params_by_dtype:
            total_params = max(params_by_dtype.values())

    if not total_params:
        print(f"  ⚠ No parameter count found for {repo_id}", file=sys.stderr)
        return None

    config = info.get("config", {})
    pipeline_tag = info.get("pipeline_tag")

    # Fetch full config.json for accurate context length
    full_config = fetch_config_json(repo_id)

    # Detect quantization format from config.json
    model_format, default_quant = detect_quant_format(repo_id, full_config)
    context_length = infer_context_length(full_config) if full_config else infer_context_length(config)

    # Correct parameters_raw when safetensors reports quantized element counts
    # instead of true parameter count (common in FP8/INT4/INT8 repos).
    arch_params = estimate_params_from_arch(full_config)
    if arch_params and arch_params > total_params * 2:
        total_params = arch_params

    min_ram, rec_ram = estimate_ram(total_params, default_quant)
    min_vram = estimate_vram(total_params, default_quant)

    architecture = config.get("model_type", "unknown")

    # Detect MoE architecture
    moe_info = detect_moe(repo_id, full_config, architecture, total_params)

    use_case_str = infer_use_case(repo_id, pipeline_tag, config)

    # Architecture metadata for the precise KV cache formula. All optional;
    # absent fields cause the Rust side to fall back to the linear approx.
    arch_meta = extract_arch_metadata(full_config)

    result = {
        "name": repo_id,
        "provider": extract_provider(repo_id),
        "parameter_count": format_param_count(total_params),
        "parameters_raw": total_params,
        "min_ram_gb": min_ram,
        "recommended_ram_gb": rec_ram,
        "min_vram_gb": min_vram,
        "quantization": default_quant,
        "format": model_format,
        "context_length": context_length,
        "use_case": use_case_str,
        "capabilities": infer_capabilities(repo_id, pipeline_tag, use_case_str),
        "pipeline_tag": pipeline_tag or "unknown",
        "architecture": architecture,
        "hf_downloads": info.get("downloads", 0),
        "hf_likes": info.get("likes", 0),
        "release_date": (info.get("createdAt") or "")[:10] or None,
        **arch_meta,
    }

    # Add MoE fields if detected
    if moe_info["is_moe"]:
        result["is_moe"] = True
        result["num_experts"] = moe_info["num_experts"]
        result["active_experts"] = moe_info["active_experts"]
        result["active_parameters"] = moe_info["active_parameters"]

    return result


def scrape_models_parallel(repo_ids: list[str], threads: int) -> tuple[list[dict], set[str]]:
    """Scrape a batch of models with optional parallelism.

    Returns (results, scraped_names).
    """
    results: list[dict] = []
    scraped_names: set[str] = set()
    total = len(repo_ids)

    if threads <= 1:
        for i, repo_id in enumerate(repo_ids, 1):
            print(f"[{i}/{total}] {repo_id}...")
            model = scrape_model(repo_id)
            if model:
                print(f"  ✓ {model['parameter_count']} params, "
                      f"min {model['min_ram_gb']} GB RAM, "
                      f"ctx {model['context_length']}")
                results.append(model)
                scraped_names.add(repo_id)
            # Be polite to the API in single-thread mode.
            time.sleep(0.3)
        return results, scraped_names

    print(f"Using {threads} threads for model scraping")
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # executor.map keeps output aligned with input ordering while running concurrently.
        for i, (repo_id, model) in enumerate(
            zip(repo_ids, executor.map(scrape_model, repo_ids)),
            1,
        ):
            print(f"[{i}/{total}] {repo_id}...")
            if model:
                print(f"  ✓ {model['parameter_count']} params, "
                      f"min {model['min_ram_gb']} GB RAM, "
                      f"ctx {model['context_length']}")
                results.append(model)
                scraped_names.add(repo_id)

    return results, scraped_names


# ---------------------------------------------------------------------------
# GGUF source enrichment — find pre-quantized GGUF repos for known models
# ---------------------------------------------------------------------------

# Providers known to publish high-quality GGUF quantizations
GGUF_PROVIDERS = ["unsloth", "bartowski", "ggml-org", "TheBloke", "mradermacher"]

GGUF_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "gguf_sources_cache.json")
GGUF_CACHE_MAX_AGE_DAYS = 7  # Re-check repos older than this


def _load_gguf_cache() -> dict:
    """Load the GGUF source cache from disk.

    Returns dict mapping model repo_id -> {"sources": [...], "checked": ISO timestamp}
    """
    try:
        with open(GGUF_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_gguf_cache(cache: dict):
    """Save the GGUF source cache to disk."""
    os.makedirs(os.path.dirname(GGUF_CACHE_FILE), exist_ok=True)
    with open(GGUF_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_entry_fresh(entry: dict) -> bool:
    """Check if a cache entry is still valid."""
    try:
        from datetime import datetime, timedelta, timezone
        checked = datetime.fromisoformat(entry["checked"])
        return (datetime.now(timezone.utc) - checked) < timedelta(days=GGUF_CACHE_MAX_AGE_DAYS)
    except (KeyError, ValueError):
        return False


def _model_gguf_repo_candidates(repo_id: str) -> list[tuple[str, str]]:
    """Generate candidate GGUF repo names for a model.

    Returns list of (provider, candidate_repo_id) tuples.
    e.g. for "meta-llama/Llama-3.1-8B-Instruct" →
         [("unsloth", "unsloth/Llama-3.1-8B-Instruct-GGUF"),
          ("bartowski", "bartowski/Llama-3.1-8B-Instruct-GGUF")]
    """
    model_name = repo_id.split("/")[-1]
    candidates = []
    for provider in GGUF_PROVIDERS:
        candidates.append((provider, f"{provider}/{model_name}-GGUF"))
    return candidates


def check_gguf_repo_exists(repo_id: str) -> bool:
    """Check if a HuggingFace repo exists and has GGUF files."""
    url = f"{HF_API}/{repo_id}"
    req = urllib.request.Request(url, headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            info = json.loads(resp.read().decode())
            tags = info.get("tags", [])
            return "gguf" in tags
    except Exception:
        return False


def _resolve_gguf_sources(repo_id: str) -> tuple[list[dict], list[tuple[str, bool]]]:
    """Resolve GGUF sources for a single model repo.

    Returns (sources, checks) where checks is [(candidate_repo, exists), ...].
    """
    sources: list[dict] = []
    checks: list[tuple[str, bool]] = []
    for provider, candidate_repo in _model_gguf_repo_candidates(repo_id):
        exists = check_gguf_repo_exists(candidate_repo)
        checks.append((candidate_repo, exists))
        if exists:
            sources.append({"repo": candidate_repo, "provider": provider})
        time.sleep(0.15)  # Be polite to the API
    return sources, checks


def enrich_gguf_sources(models: list[dict], threads: int = 1) -> int:
    """Add gguf_sources to models by checking GGUF provider repos.

    Uses a persistent cache to avoid re-checking repos on every scrape.
    Returns the number of models enriched.
    """
    cache = _load_gguf_cache()
    enriched = 0
    cache_hits = 0
    total = len(models)
    from datetime import datetime, timezone

    to_check: list[tuple[int, str]] = []

    for i, model in enumerate(models, 1):
        repo_id = model["name"]

        # Skip non-GGUF models (AWQ/GPTQ don't use GGUF sources)
        if model.get("format", "gguf") != "gguf":
            continue

        # Check cache first
        if repo_id in cache and _cache_entry_fresh(cache[repo_id]):
            sources = cache[repo_id]["sources"]
            cache_hits += 1
        else:
            to_check.append((i, repo_id))
            continue

        if sources:
            model["gguf_sources"] = sources
            enriched += 1

    # Resolve cache misses, optionally in parallel.
    if to_check:
        def _apply_checked_sources(idx: int, repo_id: str, sources: list[dict]):
            nonlocal enriched
            cache[repo_id] = {
                "sources": sources,
                "checked": datetime.now(timezone.utc).isoformat(),
            }
            if sources:
                models[idx - 1]["gguf_sources"] = sources
                enriched += 1

        if threads <= 1:
            for idx, repo_id in to_check:
                sources, checks = _resolve_gguf_sources(repo_id)
                print(f"  [{idx}/{total}] {repo_id}")
                for candidate_repo, exists in checks:
                    mark = "✓" if exists else "✗"
                    print(f"     {mark} {candidate_repo}")
                print(f"     -> {len(sources)} source(s)")
                _apply_checked_sources(idx, repo_id, sources)
        else:
            print(f"  Using {threads} threads for GGUF source checks")
            future_to_meta: dict[concurrent.futures.Future, tuple[int, str]] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for idx, repo_id in to_check:
                    future = executor.submit(_resolve_gguf_sources, repo_id)
                    future_to_meta[future] = (idx, repo_id)

                for future in concurrent.futures.as_completed(future_to_meta):
                    idx, repo_id = future_to_meta[future]
                    sources, checks = future.result()
                    print(f"  [{idx}/{total}] {repo_id}")
                    for candidate_repo, exists in checks:
                        mark = "✓" if exists else "✗"
                        print(f"     {mark} {candidate_repo}")
                    print(f"     -> {len(sources)} source(s)")
                    _apply_checked_sources(idx, repo_id, sources)

    _save_gguf_cache(cache)
    print(f"  Cache: {cache_hits} hits, {total - cache_hits} API checks")
    return enriched


# ---------------------------------------------------------------------------
# Auto-discovery from HuggingFace trending / most-downloaded
# ---------------------------------------------------------------------------

# Pipeline tags to search for discoverable models
DISCOVER_PIPELINES = [
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "feature-extraction",       # Embedding models (useful for RAG sizing)
]

# Orgs to skip — test fixtures and legacy mirrors only.
# Quantization/repack orgs (TheBloke, bartowski, unsloth, etc.) are kept
# because they provide popular quantised variants users actually run.
SKIP_ORGS = {
    "trl-internal-testing",   # Test fixtures
}

# Sort strategies to query — results are merged and deduplicated.
# Each strategy surfaces models that the others might miss.
DISCOVER_SORT_STRATEGIES = [
    "downloads",        # All-time most downloaded
    "trendingScore",    # Currently trending (recent velocity)
    "likes30d",         # Most liked in the last 30 days
]


def _fetch_models_page(url: str) -> tuple[list[dict], str | None]:
    """Fetch a page of models from the HuggingFace API.

    Returns (models, next_url) where next_url is parsed from the Link header
    for cursor-based pagination, or None if there are no more pages.
    """
    req = urllib.request.Request(url, headers=_auth_headers())
    with urllib.request.urlopen(req, timeout=60) as resp:
        # Parse cursor-based pagination from Link header
        next_url = None
        link_header = resp.headers.get("Link", "")
        if 'rel="next"' in link_header:
            # Format: <url>; rel="next"
            next_url = link_header.split(">")[0].lstrip("<")
        models = json.loads(resp.read().decode())
    return models, next_url


def _build_first_page_url(pipeline: str, sort: str, page_size: int) -> str:
    """Build the initial API URL for a pipeline query."""
    return (
        f"{HF_API}?"
        f"pipeline_tag={pipeline}&"
        f"sort={sort}&"
        f"direction=-1&"
        f"limit={page_size}&"
        f"expand[]=safetensors&"
        f"expand[]=config"
    )


def _estimate_params_from_config(config: dict) -> int | None:
    """Try to estimate parameter count from model config fields.

    This is a fallback for models that don't expose safetensors metadata
    in the listing API. Uses common config.json fields to estimate.
    """
    # Some configs directly state the param count
    for key in ("num_parameters", "n_params", "total_params"):
        val = config.get(key)
        if val and isinstance(val, (int, float)) and val > 1000:
            return int(val)

    # Estimate from architecture dimensions (rough but useful)
    hidden = config.get("hidden_size") or config.get("d_model")
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    vocab = config.get("vocab_size")
    intermediate = config.get("intermediate_size") or config.get("d_ff")

    if hidden and layers and vocab:
        # Rough transformer parameter estimate:
        # ~12 * L * H^2 (attention + FFN) + V * H (embeddings)
        ffn_factor = (intermediate / hidden) if intermediate else 4.0
        params = int(layers * hidden * hidden * (4 + 2 * ffn_factor) + vocab * hidden)
        if params > 1_000_000:  # sanity check: at least 1M params
            return params

    return None


def _process_listing(
    m: dict,
    curated: set[str],
    seen_ids: set[str],
    min_downloads: int,
    stats: dict,
) -> dict | None:
    """Check a single model listing against filters.

    Returns the listing with _total_params attached if accepted, else None.
    Mutates seen_ids and stats as side effects.
    """
    repo_id = m.get("id", "")
    if not repo_id or "/" not in repo_id:
        return None
    stats["total_seen"] += 1

    if repo_id in curated:
        stats["skip_curated"] += 1
        return None

    if repo_id in seen_ids:
        stats["skip_duplicate"] += 1
        return None
    seen_ids.add(repo_id)

    org = repo_id.split("/")[0]
    if org in SKIP_ORGS:
        stats["skip_org"] += 1
        return None

    downloads = m.get("downloads", 0)
    if downloads < min_downloads:
        stats["skip_downloads"] += 1
        return None

    tags = set(m.get("tags", []))
    if tags & {"adapter", "merge", "lora", "qlora"}:
        stats["skip_tags"] += 1
        return None

    # Try safetensors metadata first
    safetensors = m.get("safetensors", {})
    total_params = safetensors.get("total")
    if not total_params:
        params_by_dtype = safetensors.get("parameters", {})
        if params_by_dtype:
            total_params = max(params_by_dtype.values())

    param_source = "safetensors"

    # Fallback: fetch full config.json and estimate from arch dims.
    # Cap config fetches to avoid excessive network calls during discovery.
    config_attempts = stats["params_from_config"] + stats.get("skip_no_params", 0)
    if not total_params and config_attempts < 500:
        full_cfg = fetch_config_json(repo_id)
        if full_cfg:
            total_params = _estimate_params_from_config(full_cfg)
        param_source = "config"

    if not total_params:
        stats["skip_no_params"] += 1
        return None

    if param_source == "safetensors":
        stats["params_from_safetensors"] += 1
    else:
        stats["params_from_config"] += 1

    m["_total_params"] = total_params
    stats["accepted"] += 1
    return m


def discover_trending_models(limit: int = 30, min_downloads: int = 10000) -> list[dict]:
    """Discover popular models from HuggingFace using multiple sort strategies.

    Queries the HF API with three sort strategies (all-time downloads,
    trending score, and 30-day likes) across all pipeline types, then
    merges and deduplicates the results. This surfaces both established
    popular models and newly trending ones.

    Uses cursor-based pagination and falls back to estimating params from
    config.json when safetensors metadata is unavailable.

    Returns a list of dicts with model listing data for models NOT already
    in TARGET_MODELS.
    """
    curated = set(TARGET_MODELS)
    discovered = []
    seen_ids = set()

    PAGE_SIZE = 1000

    stats = {
        "total_seen": 0,
        "skip_curated": 0,
        "skip_duplicate": 0,
        "skip_org": 0,
        "skip_downloads": 0,
        "skip_tags": 0,
        "skip_no_params": 0,
        "params_from_safetensors": 0,
        "params_from_config": 0,
        "accepted": 0,
    }

    for sort_strategy in DISCOVER_SORT_STRATEGIES:
        strategy_accepted = 0
        # Trending/likes sorts surface newly popular models that may not
        # have high all-time downloads yet — use a lower floor for them.
        effective_min = (min_downloads if sort_strategy == "downloads"
                         else max(1000, min_downloads // 10))
        # Cap pages for non-download sorts since they aren't ordered by
        # downloads and would otherwise scan endlessly.
        max_pages = 50 if sort_strategy == "downloads" else 5

        for pipeline in DISCOVER_PIPELINES:
            next_url: str | None = _build_first_page_url(
                pipeline, sort_strategy, PAGE_SIZE
            )
            pipeline_accepted = 0
            hit_floor = False
            page_num = 0

            while len(discovered) < limit and next_url and page_num < max_pages:
                page_num += 1
                try:
                    models, next_url = _fetch_models_page(next_url)
                except Exception as e:
                    print(f"    ⚠ {pipeline} page {page_num}: {e}",
                          file=sys.stderr)
                    break

                if not models:
                    break

                below_min_this_page = 0

                for m in models:
                    result = _process_listing(
                        m, curated, seen_ids, effective_min, stats
                    )
                    if result is None:
                        # Track download-floor hits for early stop
                        downloads = m.get("downloads", 0)
                        repo_id = m.get("id", "")
                        if (repo_id and "/" in repo_id
                                and repo_id not in curated
                                and downloads < effective_min):
                            below_min_this_page += 1
                        continue

                    discovered.append(result)
                    pipeline_accepted += 1
                    strategy_accepted += 1
                    if len(discovered) >= limit:
                        break

                # For download-sorted queries, stop when most results are
                # below the threshold. For trending/likes sorts, always
                # exhaust pages since ordering isn't by downloads.
                if sort_strategy == "downloads":
                    if below_min_this_page > len(models) * 0.8:
                        hit_floor = True
                        break

                if len(models) < PAGE_SIZE:
                    break

                time.sleep(0.2)

            suffix = f", hit download floor" if hit_floor else ""
            if pipeline_accepted > 0 or page_num > 0:
                print(f"    {pipeline}: +{pipeline_accepted}"
                      f" (pages: {page_num}{suffix})")

            if len(discovered) >= limit:
                break

        print(f"  sort={sort_strategy} (min_dl={effective_min:,}): "
              f"+{strategy_accepted} new models")

        if len(discovered) >= limit:
            break

    # Print filter statistics
    print(f"\n  Discovery filter stats:")
    print(f"    Total listings seen:     {stats['total_seen']:>6}")
    print(f"    Skipped (curated dupe):  {stats['skip_curated']:>6}")
    print(f"    Skipped (seen/duplicate):{stats['skip_duplicate']:>6}")
    print(f"    Skipped (skip org):      {stats['skip_org']:>6}")
    print(f"    Skipped (low downloads): {stats['skip_downloads']:>6}")
    print(f"    Skipped (adapter/merge): {stats['skip_tags']:>6}")
    print(f"    Skipped (no params):     {stats['skip_no_params']:>6}")
    print(f"    Params from safetensors: {stats['params_from_safetensors']:>6}")
    print(f"    Params from config est.: {stats['params_from_config']:>6}")
    print(f"    Accepted:                {stats['accepted']:>6}")

    return discovered[:limit]


def _build_discovered_model(listing: dict) -> dict | None:
    """Build model dict from a listing returned by discover_trending_models.

    Only fetches config.json for accurate context length; all other metadata
    comes from the listing data already obtained via expand=safetensors.
    """
    repo_id = listing["id"]
    total_params = listing["_total_params"]
    config = listing.get("config", {})
    pipeline_tag = listing.get("pipeline_tag")

    full_config = fetch_config_json(repo_id)

    model_format, default_quant = detect_quant_format(repo_id, full_config)
    context_length = (infer_context_length(full_config) if full_config
                      else infer_context_length(config))

    # Correct parameters_raw when safetensors reports quantized element counts
    arch_params = estimate_params_from_arch(full_config)
    if arch_params and arch_params > total_params * 2:
        total_params = arch_params

    min_ram, rec_ram = estimate_ram(total_params, default_quant)
    min_vram = estimate_vram(total_params, default_quant)

    architecture = config.get("model_type", "unknown")
    moe_info = detect_moe(repo_id, full_config, architecture, total_params)
    use_case_str = infer_use_case(repo_id, pipeline_tag, config)

    # Architecture metadata for the precise KV cache formula.
    arch_meta = extract_arch_metadata(full_config)

    model = {
        "name": repo_id,
        "provider": extract_provider(repo_id),
        "parameter_count": format_param_count(total_params),
        "parameters_raw": total_params,
        "min_ram_gb": min_ram,
        "recommended_ram_gb": rec_ram,
        "min_vram_gb": min_vram,
        "quantization": default_quant,
        "format": model_format,
        "context_length": context_length,
        "use_case": use_case_str,
        "capabilities": infer_capabilities(repo_id, pipeline_tag, use_case_str),
        "pipeline_tag": pipeline_tag or "unknown",
        "architecture": architecture,
        "hf_downloads": listing.get("downloads", 0),
        "hf_likes": listing.get("likes", 0),
        "release_date": (listing.get("createdAt") or "")[:10] or None,
        **arch_meta,
        "_discovered": True,
    }

    if moe_info["is_moe"]:
        model["is_moe"] = True
        model["num_experts"] = moe_info["num_experts"]
        model["active_experts"] = moe_info["active_experts"]
        model["active_parameters"] = moe_info["active_parameters"]

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Scrape LLM model metadata from HuggingFace for llmfit."
    )
    parser.add_argument(
        "--discover", action="store_true", default=True,
        help="Auto-discover top models by download count from HuggingFace "
             "in addition to the curated TARGET_MODELS list (default: enabled)."
    )
    parser.add_argument(
        "--no-discover", action="store_false", dest="discover",
        help="Disable auto-discovery, only scrape curated TARGET_MODELS list."
    )
    parser.add_argument(
        "-n", "--discover-limit", type=int, default=1000,
        help="Max number of top-downloaded models to discover (default: 1000). "
             "Duplicates of curated models are skipped automatically."
    )
    parser.add_argument(
        "--min-downloads", type=int, default=10000,
        help="Minimum download count for discovered models (default: 10000)."
    )
    parser.add_argument(
        "--gguf-sources", action="store_true", default=True,
        help="Enrich models with known GGUF download sources from "
             "providers like unsloth and bartowski on HuggingFace (default: enabled)."
    )
    parser.add_argument(
        "--no-gguf-sources", action="store_false", dest="gguf_sources",
        help="Skip GGUF download source enrichment (faster scrape)."
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace API token for accessing gated models. "
             "Can also be set via HF_TOKEN or HUGGING_FACE_HUB_TOKEN env var."
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="Number of worker threads for parallel model metadata scraping "
             "(default: 1, which preserves current sequential behavior)."
    )
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be >= 1")

    # Resolve auth token: CLI flag > HF_TOKEN > HUGGING_FACE_HUB_TOKEN
    global _hf_token
    _hf_token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if _hf_token:
        print(f"🔑 Authenticated with HuggingFace token ({_hf_token[:4]}...{_hf_token[-4:]})")
    else:
        print("ℹ  No HF token set. Gated models will use fallback data.")
        print("   Set HF_TOKEN env var or pass --token to access gated models.\n")

    # Fallback entries for gated/auth-required models where the API
    # doesn't return safetensors metadata without a token.
    FALLBACKS = [
        {
            "name": "meta-llama/Llama-3.3-70B-Instruct",
            "provider": "Meta", "parameter_count": "70.6B",
            "parameters_raw": 70_553_706_496,
            "min_ram_gb": 39.4, "recommended_ram_gb": 65.7, "min_vram_gb": 36.1,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "mistralai/Mistral-Small-24B-Instruct-2501",
            "provider": "Mistral AI", "parameter_count": "24B",
            "parameters_raw": 24_000_000_000,
            "min_ram_gb": 13.4, "recommended_ram_gb": 22.4, "min_vram_gb": 12.3,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "mistral",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen2.5-14B-Instruct",
            "provider": "Alibaba", "parameter_count": "14.8B",
            "parameters_raw": 14_770_000_000,
            "min_ram_gb": 8.2, "recommended_ram_gb": 13.7, "min_vram_gb": 7.6,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "qwen2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen2.5-32B-Instruct",
            "provider": "Alibaba", "parameter_count": "32.5B",
            "parameters_raw": 32_510_000_000,
            "min_ram_gb": 18.2, "recommended_ram_gb": 30.3, "min_vram_gb": 16.7,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "qwen2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "microsoft/phi-3-mini-4k-instruct",
            "provider": "Microsoft", "parameter_count": "3.8B",
            "parameters_raw": 3_821_000_000,
            "min_ram_gb": 2.1, "recommended_ram_gb": 3.6, "min_vram_gb": 2.0,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Lightweight, edge deployment",
            "pipeline_tag": "text-generation", "architecture": "phi3",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "microsoft/phi-4",
            "provider": "Microsoft", "parameter_count": "14B",
            "parameters_raw": 14_000_000_000,
            "min_ram_gb": 7.8, "recommended_ram_gb": 13.0, "min_vram_gb": 7.2,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Reasoning, STEM, code generation",
            "pipeline_tag": "text-generation", "architecture": "phi",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "google/gemma-3-12b-it",
            "provider": "Google", "parameter_count": "12B",
            "parameters_raw": 12_000_000_000,
            "min_ram_gb": 6.7, "recommended_ram_gb": 11.2, "min_vram_gb": 6.1,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Multimodal, vision and text",
            "capabilities": ["vision", "tool_use"],
            "pipeline_tag": "image-text-to-text", "architecture": "gemma3",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "deepseek-ai/DeepSeek-V3",
            "provider": "DeepSeek", "parameter_count": "685B",
            "parameters_raw": 685_000_000_000,
            "min_ram_gb": 382.8, "recommended_ram_gb": 638.0, "min_vram_gb": 351.3,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "State-of-the-art, MoE architecture",
            "pipeline_tag": "text-generation", "architecture": "deepseek_v3",
            "is_moe": True, "num_experts": 256, "active_experts": 8,
            "active_parameters": 37_000_000_000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "CohereForAI/c4ai-command-r-v01",
            "provider": "Cohere", "parameter_count": "35B",
            "parameters_raw": 35_000_000_000,
            "min_ram_gb": 19.5, "recommended_ram_gb": 32.6, "min_vram_gb": 17.9,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "RAG, tool use, agents",
            "pipeline_tag": "text-generation", "architecture": "cohere",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "bigcode/starcoder2-15b",
            "provider": "BigCode", "parameter_count": "15.7B",
            "parameters_raw": 15_700_000_000,
            "min_ram_gb": 8.8, "recommended_ram_gb": 14.6, "min_vram_gb": 8.0,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "starcoder2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "nomic-ai/nomic-embed-text-v1.5",
            "provider": "Nomic", "parameter_count": "137M",
            "parameters_raw": 137_000_000,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "F16", "context_length": 8192,
            "use_case": "Text embeddings for RAG",
            "pipeline_tag": "feature-extraction", "architecture": "nomic_bert",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "provider": "DeepSeek", "parameter_count": "16B",
            "parameters_raw": 15_700_000_000,
            "min_ram_gb": 8.8, "recommended_ram_gb": 14.6, "min_vram_gb": 8.0,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "deepseek_v2",
            "is_moe": True, "num_experts": 64, "active_experts": 6,
            "active_parameters": 2_400_000_000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "microsoft/Phi-3-medium-14b-instruct",
            "provider": "Microsoft", "parameter_count": "14B",
            "parameters_raw": 14_000_000_000,
            "min_ram_gb": 7.8, "recommended_ram_gb": 13.0, "min_vram_gb": 7.2,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Balanced performance and size",
            "pipeline_tag": "text-generation", "architecture": "phi3",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        # NEW FALLBACKS for popular models
        {
            "name": "google/gemma-2-2b-it",
            "provider": "Google", "parameter_count": "2.6B",
            "parameters_raw": 2614341376,
            "min_ram_gb": 1.5, "recommended_ram_gb": 2.4, "min_vram_gb": 1.3,
            "quantization": "Q4_K_M", "context_length": 8192,
            "use_case": "Lightweight, edge deployment",
            "pipeline_tag": "text-generation", "architecture": "gemma2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "meta-llama/CodeLlama-7b-Instruct-hf",
            "provider": "Meta", "parameter_count": "7.0B",
            "parameters_raw": 7016400896,
            "min_ram_gb": 3.9, "recommended_ram_gb": 6.5, "min_vram_gb": 3.6,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "meta-llama/CodeLlama-13b-Instruct-hf",
            "provider": "Meta", "parameter_count": "13.0B",
            "parameters_raw": 13015864320,
            "min_ram_gb": 7.3, "recommended_ram_gb": 12.1, "min_vram_gb": 6.7,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "meta-llama/CodeLlama-34b-Instruct-hf",
            "provider": "Meta", "parameter_count": "34.0B",
            "parameters_raw": 34018971648,
            "min_ram_gb": 19.0, "recommended_ram_gb": 31.7, "min_vram_gb": 17.4,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "provider": "Meta", "parameter_count": "11.0B",
            "parameters_raw": 10665463808,
            "min_ram_gb": 6.0, "recommended_ram_gb": 9.9, "min_vram_gb": 5.5,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "mistralai/Ministral-8B-Instruct-2410",
            "provider": "Mistral AI", "parameter_count": "8.0B",
            "parameters_raw": 8030261248,
            "min_ram_gb": 4.5, "recommended_ram_gb": 7.5, "min_vram_gb": 4.1,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "mistral",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "mistralai/Mistral-Nemo-Instruct-2407",
            "provider": "Mistral AI", "parameter_count": "12.2B",
            "parameters_raw": 12247076864,
            "min_ram_gb": 6.8, "recommended_ram_gb": 11.4, "min_vram_gb": 6.3,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "mistral",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "microsoft/Phi-3.5-mini-instruct",
            "provider": "Microsoft", "parameter_count": "3.8B",
            "parameters_raw": 3821000000,
            "min_ram_gb": 2.1, "recommended_ram_gb": 3.6, "min_vram_gb": 2.0,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Lightweight, long context",
            "pipeline_tag": "text-generation", "architecture": "phi3",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "microsoft/Orca-2-7b",
            "provider": "Microsoft", "parameter_count": "7.0B",
            "parameters_raw": 7016400896,
            "min_ram_gb": 3.9, "recommended_ram_gb": 6.5, "min_vram_gb": 3.6,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Reasoning, step-by-step solutions",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "microsoft/Orca-2-13b",
            "provider": "Microsoft", "parameter_count": "13.0B",
            "parameters_raw": 13015864320,
            "min_ram_gb": 7.3, "recommended_ram_gb": 12.1, "min_vram_gb": 6.7,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Reasoning, step-by-step solutions",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "01-ai/Yi-6B-Chat",
            "provider": "01.ai", "parameter_count": "6.1B",
            "parameters_raw": 6061356032,
            "min_ram_gb": 3.4, "recommended_ram_gb": 5.6, "min_vram_gb": 3.1,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Multilingual, Chinese/English chat",
            "pipeline_tag": "text-generation", "architecture": "yi",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "01-ai/Yi-34B-Chat",
            "provider": "01.ai", "parameter_count": "34.4B",
            "parameters_raw": 34386780160,
            "min_ram_gb": 19.2, "recommended_ram_gb": 32.0, "min_vram_gb": 17.6,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Multilingual, Chinese/English chat",
            "pipeline_tag": "text-generation", "architecture": "yi",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "upstage/SOLAR-10.7B-Instruct-v1.0",
            "provider": "Upstage", "parameter_count": "10.7B",
            "parameters_raw": 10700000000,
            "min_ram_gb": 6.0, "recommended_ram_gb": 10.0, "min_vram_gb": 5.5,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "High-performance instruction following",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "tiiuae/falcon-7b-instruct",
            "provider": "TII", "parameter_count": "7.0B",
            "parameters_raw": 7000000000,
            "min_ram_gb": 3.9, "recommended_ram_gb": 6.5, "min_vram_gb": 3.6,
            "quantization": "Q4_K_M", "context_length": 2048,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "falcon",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "tiiuae/falcon-40b-instruct",
            "provider": "TII", "parameter_count": "40.0B",
            "parameters_raw": 40000000000,
            "min_ram_gb": 22.4, "recommended_ram_gb": 37.3, "min_vram_gb": 20.5,
            "quantization": "Q4_K_M", "context_length": 2048,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "falcon",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "HuggingFaceH4/zephyr-7b-beta",
            "provider": "HuggingFace", "parameter_count": "7.2B",
            "parameters_raw": 7241732096,
            "min_ram_gb": 4.0, "recommended_ram_gb": 6.7, "min_vram_gb": 3.7,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "mistral",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "openchat/openchat-3.5-0106",
            "provider": "OpenChat", "parameter_count": "7.0B",
            "parameters_raw": 7000000000,
            "min_ram_gb": 3.9, "recommended_ram_gb": 6.5, "min_vram_gb": 3.6,
            "quantization": "Q4_K_M", "context_length": 8192,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "mistral",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "lmsys/vicuna-7b-v1.5",
            "provider": "LMSYS", "parameter_count": "7.0B",
            "parameters_raw": 6738415616,
            "min_ram_gb": 3.8, "recommended_ram_gb": 6.3, "min_vram_gb": 3.4,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "lmsys/vicuna-13b-v1.5",
            "provider": "LMSYS", "parameter_count": "13.0B",
            "parameters_raw": 13015864320,
            "min_ram_gb": 7.3, "recommended_ram_gb": 12.1, "min_vram_gb": 6.7,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "provider": "NousResearch", "parameter_count": "46.7B",
            "parameters_raw": 46702792704,
            "min_ram_gb": 26.1, "recommended_ram_gb": 43.5, "min_vram_gb": 23.9,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "mixtral",
            "is_moe": True, "num_experts": 8, "active_experts": 2,
            "active_parameters": 12900000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "WizardLMTeam/WizardLM-13B-V1.2",
            "provider": "WizardLM", "parameter_count": "13.0B",
            "parameters_raw": 13015864320,
            "min_ram_gb": 7.3, "recommended_ram_gb": 12.1, "min_vram_gb": 6.7,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "WizardLMTeam/WizardCoder-15B-V1.0",
            "provider": "WizardLM", "parameter_count": "15.5B",
            "parameters_raw": 15515334656,
            "min_ram_gb": 8.7, "recommended_ram_gb": 14.5, "min_vram_gb": 7.9,
            "quantization": "Q4_K_M", "context_length": 8192,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "starcoder",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "provider": "Alibaba", "parameter_count": "1.5B",
            "parameters_raw": 1539938304,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.8,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "qwen2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "provider": "Alibaba", "parameter_count": "7.6B",
            "parameters_raw": 7615616000,
            "min_ram_gb": 4.3, "recommended_ram_gb": 7.1, "min_vram_gb": 3.9,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "qwen2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "provider": "Alibaba", "parameter_count": "14.7B",
            "parameters_raw": 14770000000,
            "min_ram_gb": 8.2, "recommended_ram_gb": 13.7, "min_vram_gb": 7.6,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "qwen2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "provider": "Alibaba", "parameter_count": "32.5B",
            "parameters_raw": 32510000000,
            "min_ram_gb": 18.2, "recommended_ram_gb": 30.3, "min_vram_gb": 16.7,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "qwen2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "provider": "Alibaba", "parameter_count": "3.8B",
            "parameters_raw": 3821000000,
            "min_ram_gb": 2.1, "recommended_ram_gb": 3.6, "min_vram_gb": 2.0,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "qwen2_vl",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "provider": "Alibaba", "parameter_count": "8.3B",
            "parameters_raw": 8290000000,
            "min_ram_gb": 4.6, "recommended_ram_gb": 7.7, "min_vram_gb": 4.2,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "qwen2_vl",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen3-14B",
            "provider": "Alibaba", "parameter_count": "14.8B",
            "parameters_raw": 14770000000,
            "min_ram_gb": 8.2, "recommended_ram_gb": 13.7, "min_vram_gb": 7.6,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "General purpose text generation",
            "pipeline_tag": "text-generation", "architecture": "qwen3",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        # --- New fallbacks added Feb 2026 ---
        {
            "name": "deepseek-ai/DeepSeek-V3.2",
            "provider": "DeepSeek", "parameter_count": "685B",
            "parameters_raw": 685000000000,
            "min_ram_gb": 383.2, "recommended_ram_gb": 638.7, "min_vram_gb": 351.3,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "State-of-the-art, MoE architecture",
            "pipeline_tag": "text-generation", "architecture": "deepseek_v3",
            "is_moe": True, "num_experts": 256, "active_experts": 8,
            "active_parameters": 37000000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-12-01",
        },
        {
            "name": "deepseek-ai/DeepSeek-V3.2-Speciale",
            "provider": "DeepSeek", "parameter_count": "685B",
            "parameters_raw": 685000000000,
            "min_ram_gb": 383.2, "recommended_ram_gb": 638.7, "min_vram_gb": 351.3,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Advanced reasoning, chain-of-thought",
            "pipeline_tag": "text-generation", "architecture": "deepseek_v3",
            "is_moe": True, "num_experts": 256, "active_experts": 8,
            "active_parameters": 37000000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-12-01",
        },
        {
            "name": "zai-org/GLM-5",
            "provider": "Zhipu AI", "parameter_count": "744B",
            "parameters_raw": 744000000000,
            "min_ram_gb": 416.2, "recommended_ram_gb": 693.6, "min_vram_gb": 381.4,
            "quantization": "Q4_K_M", "context_length": 200000,
            "use_case": "State-of-the-art, MoE architecture",
            "pipeline_tag": "text-generation", "architecture": "glm",
            "is_moe": True, "num_experts": 256, "active_experts": 8,
            "active_parameters": 40000000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2026-02-11",
        },
        {
            "name": "moonshotai/Kimi-K2.5",
            "provider": "Moonshot", "parameter_count": "171B",
            "parameters_raw": 171000000000,
            "min_ram_gb": 95.6, "recommended_ram_gb": 159.4, "min_vram_gb": 87.7,
            "quantization": "Q4_K_M", "context_length": 262144,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "kimi",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2026-01-26",
        },
        {
            "name": "MiniMaxAI/MiniMax-M2.7",
            "provider": "MiniMax", "parameter_count": "230B",
            "parameters_raw": 230000000000,
            "min_ram_gb": 128.6, "recommended_ram_gb": 214.4, "min_vram_gb": 117.9,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Latest flagship with enhanced reasoning and coding",
            "pipeline_tag": "text-generation", "architecture": "minimax",
            "is_moe": True, "num_experts": 32, "active_experts": 2,
            "active_parameters": 10000000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2026-03-18",
        },
        {
            "name": "MiniMaxAI/MiniMax-M2.5",
            "provider": "MiniMax", "parameter_count": "230B",
            "parameters_raw": 230000000000,
            "min_ram_gb": 128.6, "recommended_ram_gb": 214.4, "min_vram_gb": 117.9,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Coding, agentic tool use",
            "pipeline_tag": "text-generation", "architecture": "minimax",
            "is_moe": True, "num_experts": 32, "active_experts": 2,
            "active_parameters": 10000000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2026-02-11",
        },
        {
            "name": "XiaomiMiMo/MiMo-V2-Flash",
            "provider": "Xiaomi", "parameter_count": "309B",
            "parameters_raw": 309000000000,
            "min_ram_gb": 172.8, "recommended_ram_gb": 288.0, "min_vram_gb": 158.4,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Efficient reasoning, coding",
            "pipeline_tag": "text-generation", "architecture": "mimo",
            "is_moe": True, "num_experts": 128, "active_experts": 8,
            "active_parameters": 15000000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-12-01",
        },
        {
            "name": "XiaomiMiMo/MiMo-7B-RL",
            "provider": "Xiaomi", "parameter_count": "7.0B",
            "parameters_raw": 7000000000,
            "min_ram_gb": 3.9, "recommended_ram_gb": 6.5, "min_vram_gb": 3.6,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Advanced reasoning, math and code",
            "pipeline_tag": "text-generation", "architecture": "mimo",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-05-01",
        },
        {
            "name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "provider": "NVIDIA", "parameter_count": "30B",
            "parameters_raw": 30000000000,
            "min_ram_gb": 16.8, "recommended_ram_gb": 28.0, "min_vram_gb": 15.4,
            "quantization": "Q4_K_M", "context_length": 1048576,
            "use_case": "Efficient MoE, agentic tasks",
            "pipeline_tag": "text-generation", "architecture": "nemotron",
            "is_moe": True, "num_experts": 128, "active_experts": 6,
            "active_parameters": 3000000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-06-01",
        },
        {
            "name": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            "provider": "NVIDIA", "parameter_count": "9B",
            "parameters_raw": 9000000000,
            "min_ram_gb": 5.0, "recommended_ram_gb": 8.4, "min_vram_gb": 4.6,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Hybrid Mamba2, reasoning",
            "pipeline_tag": "text-generation", "architecture": "nemotron",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-06-01",
        },
        {
            "name": "microsoft/Phi-4-reasoning",
            "provider": "Microsoft", "parameter_count": "14B",
            "parameters_raw": 14000000000,
            "min_ram_gb": 7.8, "recommended_ram_gb": 13.0, "min_vram_gb": 7.2,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Advanced reasoning, math and code",
            "pipeline_tag": "text-generation", "architecture": "phi4",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-04-01",
        },
        {
            "name": "microsoft/Phi-4-mini-reasoning",
            "provider": "Microsoft", "parameter_count": "3.8B",
            "parameters_raw": 3800000000,
            "min_ram_gb": 2.1, "recommended_ram_gb": 3.5, "min_vram_gb": 1.9,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Lightweight reasoning",
            "pipeline_tag": "text-generation", "architecture": "phi4",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-04-01",
        },
        {
            "name": "microsoft/Phi-4-multimodal-instruct",
            "provider": "Microsoft", "parameter_count": "14B",
            "parameters_raw": 14000000000,
            "min_ram_gb": 7.8, "recommended_ram_gb": 13.0, "min_vram_gb": 7.2,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Multimodal, vision and audio",
            "pipeline_tag": "image-text-to-text", "architecture": "phi4",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-04-01",
        },
        {
            "name": "LGAI-EXAONE/EXAONE-4.0-32B",
            "provider": "LG AI", "parameter_count": "32B",
            "parameters_raw": 32000000000,
            "min_ram_gb": 17.9, "recommended_ram_gb": 29.8, "min_vram_gb": 16.4,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Hybrid reasoning, multilingual",
            "pipeline_tag": "text-generation", "architecture": "exaone",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-07-15",
        },
        {
            "name": "LGAI-EXAONE/EXAONE-4.0-1.2B",
            "provider": "LG AI", "parameter_count": "1.2B",
            "parameters_raw": 1200000000,
            "min_ram_gb": 0.7, "recommended_ram_gb": 1.1, "min_vram_gb": 0.6,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Lightweight, on-device",
            "pipeline_tag": "text-generation", "architecture": "exaone",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-07-15",
        },
        {
            "name": "HuggingFaceTB/SmolLM3-3B",
            "provider": "HuggingFace", "parameter_count": "3B",
            "parameters_raw": 3000000000,
            "min_ram_gb": 1.7, "recommended_ram_gb": 2.8, "min_vram_gb": 1.5,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Lightweight, multilingual reasoning",
            "pipeline_tag": "text-generation", "architecture": "smollm",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-07-08",
        },
        {
            "name": "google/gemma-3n-E4B-it",
            "provider": "Google", "parameter_count": "8B",
            "parameters_raw": 8000000000,
            "min_ram_gb": 4.5, "recommended_ram_gb": 7.5, "min_vram_gb": 4.1,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Multimodal, on-device (effective 4B)",
            "pipeline_tag": "image-text-to-text", "architecture": "gemma3n",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-06-25",
        },
        {
            "name": "google/gemma-3n-E2B-it",
            "provider": "Google", "parameter_count": "4B",
            "parameters_raw": 4000000000,
            "min_ram_gb": 2.2, "recommended_ram_gb": 3.7, "min_vram_gb": 2.1,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Multimodal, on-device (effective 2B)",
            "pipeline_tag": "image-text-to-text", "architecture": "gemma3n",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-06-25",
        },
        # Google Gemma 4 family
        {
            "name": "google/gemma-4-E2B-it",
            "provider": "Google", "parameter_count": "5.1B",
            "parameters_raw": 5100000000,
            "min_ram_gb": 2.9, "recommended_ram_gb": 4.8, "min_vram_gb": 2.6,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Multimodal, on-device (effective 2B)",
            "pipeline_tag": "image-text-to-text", "architecture": "gemma4",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-07-30",
        },
        {
            "name": "google/gemma-4-E4B-it",
            "provider": "Google", "parameter_count": "8B",
            "parameters_raw": 8000000000,
            "min_ram_gb": 4.5, "recommended_ram_gb": 7.5, "min_vram_gb": 4.1,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Multimodal, on-device (effective 4B)",
            "pipeline_tag": "image-text-to-text", "architecture": "gemma4",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-07-30",
        },
        {
            "name": "google/gemma-4-31B-it",
            "provider": "Google", "parameter_count": "31B",
            "parameters_raw": 31000000000,
            "min_ram_gb": 17.3, "recommended_ram_gb": 28.9, "min_vram_gb": 15.9,
            "quantization": "Q4_K_M", "context_length": 262144,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "gemma4",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-07-30",
        },
        {
            "name": "google/gemma-4-26B-A4B-it",
            "provider": "Google", "parameter_count": "26B",
            "parameters_raw": 26000000000,
            "min_ram_gb": 14.5, "recommended_ram_gb": 24.2, "min_vram_gb": 13.3,
            "quantization": "Q4_K_M", "context_length": 262144,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "gemma4",
            "is_moe": True, "num_experts": 128, "active_experts": 8,
            "active_parameters": 4_000_000_000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-07-30",
        },
        # Qwen3-Coder-Next (80B MoE, 3B active, Jan 2026)
        {
            "name": "Qwen/Qwen3-Coder-Next",
            "provider": "Alibaba", "parameter_count": "80B",
            "parameters_raw": 80000000000,
            "min_ram_gb": 44.8, "recommended_ram_gb": 74.6, "min_vram_gb": 41.0,
            "quantization": "Q4_K_M", "context_length": 262144,
            "use_case": "Code generation, agentic coding",
            "pipeline_tag": "text-generation", "architecture": "qwen3_next",
            "is_moe": True, "num_experts": 64, "active_experts": 4,
            "active_parameters": 3000000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2026-01-30",
        },
        {
            "name": "Qwen/Qwen3.5-27B",
            "provider": "Alibaba", "parameter_count": "27.8B",
            "parameters_raw": 27781427952,
            "min_ram_gb": 15.5, "recommended_ram_gb": 25.9, "min_vram_gb": 14.2,
            "quantization": "Q4_K_M", "context_length": 262144,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "qwen3_5",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "Qwen/Qwen3.5-35B-A3B",
            "provider": "Alibaba", "parameter_count": "36.0B",
            "parameters_raw": 35951822704,
            "min_ram_gb": 20.1, "recommended_ram_gb": 33.5, "min_vram_gb": 18.4,
            "quantization": "Q4_K_M", "context_length": 262144,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "qwen3_5_moe",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
            "is_moe": True, "num_experts": 256, "active_experts": 8,
            "active_parameters": 3_000_000_000,
        },
        {
            "name": "Qwen/Qwen3.5-122B-A10B",
            "provider": "Alibaba", "parameter_count": "125.1B",
            "parameters_raw": 125086497008,
            "min_ram_gb": 69.9, "recommended_ram_gb": 116.5, "min_vram_gb": 64.1,
            "quantization": "Q4_K_M", "context_length": 262144,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "qwen3_5_moe",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
            "is_moe": True, "num_experts": 256, "active_experts": 8,
            "active_parameters": 10_000_000_000,
        },
        {
            "name": "Qwen/Qwen3.5-397B-A17B",
            "provider": "Alibaba", "parameter_count": "403.4B",
            "parameters_raw": 403397928944,
            "min_ram_gb": 225.4, "recommended_ram_gb": 375.7, "min_vram_gb": 206.6,
            "quantization": "Q4_K_M", "context_length": 262144,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "qwen3_5_moe",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
            "is_moe": True, "num_experts": 256, "active_experts": 8,
            "active_parameters": 17_000_000_000,
        },
        # Liquid AI LFM2 dense models
        {
            "name": "LiquidAI/LFM2-350M",
            "provider": "Liquid AI", "parameter_count": "354M",
            "parameters_raw": 354483968,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Lightweight, edge deployment",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-700M",
            "provider": "Liquid AI", "parameter_count": "742M",
            "parameters_raw": 742489344,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Lightweight, edge deployment",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-1.2B",
            "provider": "Liquid AI", "parameter_count": "1.2B",
            "parameters_raw": 1170340608,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.6,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "General purpose text generation",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-2.6B",
            "provider": "Liquid AI", "parameter_count": "2.6B",
            "parameters_raw": 2569272320,
            "min_ram_gb": 1.4, "recommended_ram_gb": 2.4, "min_vram_gb": 1.3,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "General purpose text generation",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-2.6B-Exp",
            "provider": "Liquid AI", "parameter_count": "2.6B",
            "parameters_raw": 2569272320,
            "min_ram_gb": 1.4, "recommended_ram_gb": 2.4, "min_vram_gb": 1.3,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Instruction following, math, knowledge",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        # Liquid AI LFM2 MoE models
        {
            "name": "LiquidAI/LFM2-8B-A1B",
            "provider": "Liquid AI", "parameter_count": "8.3B",
            "parameters_raw": 8300000000,
            "min_ram_gb": 4.6, "recommended_ram_gb": 7.7, "min_vram_gb": 4.3,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "General purpose, edge MoE",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "is_moe": True, "num_experts": 32, "active_experts": 4,
            "active_parameters": 1500000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-24B-A2B",
            "provider": "Liquid AI", "parameter_count": "23.8B",
            "parameters_raw": 23_843_661_440,
            "min_ram_gb": 13.3, "recommended_ram_gb": 22.2, "min_vram_gb": 12.2,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Agentic tasks, RAG, summarization",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "is_moe": True, "num_experts": 32, "active_experts": 4,
            "active_parameters": 2300000000,
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        # Liquid AI LFM2.5 models
        {
            "name": "LiquidAI/LFM2.5-1.2B-Base",
            "provider": "Liquid AI", "parameter_count": "1.2B",
            "parameters_raw": 1170340608,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.6,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "General purpose text generation",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2.5-1.2B-Instruct",
            "provider": "Liquid AI", "parameter_count": "1.2B",
            "parameters_raw": 1170340608,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.6,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2.5-1.2B-Thinking",
            "provider": "Liquid AI", "parameter_count": "1.2B",
            "parameters_raw": 1170340608,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.6,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Advanced reasoning, chain-of-thought",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2.5-1.2B-JP",
            "provider": "Liquid AI", "parameter_count": "1.2B",
            "parameters_raw": 1170340608,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.6,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Japanese language, multilingual chat",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        # Liquid AI LFM2 Vision-Language models
        {
            "name": "LiquidAI/LFM2-VL-450M",
            "provider": "Liquid AI", "parameter_count": "451M",
            "parameters_raw": 450822656,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-VL-1.6B",
            "provider": "Liquid AI", "parameter_count": "1.6B",
            "parameters_raw": 1584804000,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.8,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-VL-3B",
            "provider": "Liquid AI", "parameter_count": "3.0B",
            "parameters_raw": 2998975216,
            "min_ram_gb": 1.7, "recommended_ram_gb": 2.8, "min_vram_gb": 1.5,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2.5-VL-1.6B",
            "provider": "Liquid AI", "parameter_count": "1.6B",
            "parameters_raw": 1596625904,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.8,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        # Liquid AI LFM2 Audio models
        {
            "name": "LiquidAI/LFM2-Audio-1.5B",
            "provider": "Liquid AI", "parameter_count": "1.5B",
            "parameters_raw": 1500000000,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.8,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Speech-to-speech, ASR, TTS",
            "pipeline_tag": "audio-to-audio", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2.5-Audio-1.5B",
            "provider": "Liquid AI", "parameter_count": "1.5B",
            "parameters_raw": 1500000000,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.8,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Speech-to-speech, ASR, TTS",
            "pipeline_tag": "audio-to-audio", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        # Liquid AI Liquid Nanos (task-specific fine-tunes)
        {
            "name": "LiquidAI/LFM2-1.2B-Tool",
            "provider": "Liquid AI", "parameter_count": "1.2B",
            "parameters_raw": 1170340608,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.6,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Tool calling, function calling",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-1.2B-RAG",
            "provider": "Liquid AI", "parameter_count": "1.2B",
            "parameters_raw": 1170340608,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.6,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Retrieval-augmented generation",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-1.2B-Extract",
            "provider": "Liquid AI", "parameter_count": "1.2B",
            "parameters_raw": 1170340608,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.6,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Data extraction, structured output",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-350M-Extract",
            "provider": "Liquid AI", "parameter_count": "354M",
            "parameters_raw": 354483968,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Data extraction, structured output",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-350M-Math",
            "provider": "Liquid AI", "parameter_count": "354M",
            "parameters_raw": 354483968,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Math reasoning, chain-of-thought",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-350M-ENJP-MT",
            "provider": "Liquid AI", "parameter_count": "354M",
            "parameters_raw": 354483968,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "English-Japanese translation",
            "pipeline_tag": "translation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-350M-PII-Extract-JP",
            "provider": "Liquid AI", "parameter_count": "354M",
            "parameters_raw": 354483968,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "PII extraction, Japanese",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-ColBERT-350M",
            "provider": "Liquid AI", "parameter_count": "353M",
            "parameters_raw": 353322752,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Semantic search, sentence similarity",
            "pipeline_tag": "sentence-similarity", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        {
            "name": "LiquidAI/LFM2-2.6B-Transcript",
            "provider": "Liquid AI", "parameter_count": "2.6B",
            "parameters_raw": 2569272320,
            "min_ram_gb": 1.4, "recommended_ram_gb": 2.4, "min_vram_gb": 1.3,
            "quantization": "Q4_K_M", "context_length": 128000,
            "use_case": "Meeting transcription, summarization",
            "pipeline_tag": "text-generation", "architecture": "lfm2",
            "hf_downloads": 0, "hf_likes": 0, "release_date": "2025-11-28",
        },
        # RWKV v7 G1f: GGUF-native repos — no safetensors metadata, fallback required
        {
            "name": "shoumenchougou/RWKV7-G1f-1.5B-GGUF",
            "provider": "RWKV", "parameter_count": "1.5B",
            "parameters_raw": 1_500_000_000,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.8,
            "quantization": "Q4_K_M", "format": "gguf", "context_length": 8192,
            "use_case": "General purpose text generation",
            "capabilities": [],
            "pipeline_tag": "text-generation", "architecture": "rwkv",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "shoumenchougou/RWKV7-G1f-2.9B-GGUF",
            "provider": "RWKV", "parameter_count": "2.9B",
            "parameters_raw": 2_900_000_000,
            "min_ram_gb": 1.6, "recommended_ram_gb": 2.7, "min_vram_gb": 1.5,
            "quantization": "Q4_K_M", "format": "gguf", "context_length": 8192,
            "use_case": "General purpose text generation",
            "capabilities": [],
            "pipeline_tag": "text-generation", "architecture": "rwkv",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "shoumenchougou/RWKV7-G1f-7.2B-GGUF",
            "provider": "RWKV", "parameter_count": "7.2B",
            "parameters_raw": 7_200_000_000,
            "min_ram_gb": 4.0, "recommended_ram_gb": 6.7, "min_vram_gb": 3.7,
            "quantization": "Q4_K_M", "format": "gguf", "context_length": 8192,
            "use_case": "General purpose text generation",
            "capabilities": [],
            "pipeline_tag": "text-generation", "architecture": "rwkv",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
        {
            "name": "shoumenchougou/RWKV7-G1f-13.3B-GGUF",
            "provider": "RWKV", "parameter_count": "13.3B",
            "parameters_raw": 13_300_000_000,
            "min_ram_gb": 7.4, "recommended_ram_gb": 12.4, "min_vram_gb": 6.8,
            "quantization": "Q4_K_M", "format": "gguf", "context_length": 8192,
            "use_case": "General purpose text generation",
            "capabilities": [],
            "pipeline_tag": "text-generation", "architecture": "rwkv",
            "hf_downloads": 0, "hf_likes": 0, "release_date": None,
        },
    ]

    print(f"Scraping {len(TARGET_MODELS)} curated models from HuggingFace...\n")

    results, scraped_names = scrape_models_parallel(TARGET_MODELS, args.threads)

    # Fill in fallbacks for models that couldn't be scraped
    fallback_count = 0
    for fb in FALLBACKS:
        if fb["name"] not in scraped_names:
            print(f"  + Fallback: {fb['name']} ({fb['parameter_count']})")
            results.append(fb)
            scraped_names.add(fb["name"])
            fallback_count += 1

    # Auto-discover trending models if --discover flag is set
    discovered_count = 0
    if args.discover:
        print(f"\nDiscovering top models by downloads (limit={args.discover_limit}, "
              f"min_downloads={args.min_downloads:,})...")
        trending = discover_trending_models(
            limit=args.discover_limit,
            min_downloads=args.min_downloads,
        )
        already_scraped = sum(1 for l in trending if l["id"] in scraped_names)
        print(f"\n  Discovery returned {len(trending)} candidates"
              f" ({already_scraped} already scraped)\n")

        candidates = [l for l in trending if l["id"] not in scraped_names]

        if args.threads <= 1:
            for i, listing in enumerate(candidates, 1):
                repo_id = listing["id"]
                print(f"[discover {i}/{len(candidates)}] {repo_id}...")
                model = _build_discovered_model(listing)
                if model:
                    print(f"  ✓ {model['parameter_count']} params, "
                          f"{model['hf_downloads']:,} downloads, "
                          f"ctx {model['context_length']}")
                    results.append(model)
                    scraped_names.add(repo_id)
                    discovered_count += 1
                time.sleep(0.15)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
                for i, (listing, model) in enumerate(
                    zip(candidates, executor.map(_build_discovered_model, candidates)),
                    1,
                ):
                    repo_id = listing["id"]
                    print(f"[discover {i}/{len(candidates)}] {repo_id}...")
                    if model:
                        print(f"  ✓ {model['parameter_count']} params, "
                              f"{model['hf_downloads']:,} downloads, "
                              f"ctx {model['context_length']}")
                        results.append(model)
                        scraped_names.add(repo_id)
                        discovered_count += 1

    # --- Additive merge with existing database ---
    # The database is additive: models from previous runs are preserved.
    # Freshly scraped models update existing entries; historical models
    # that are no longer in the top discovered set are kept as-is.
    output_paths = ["data/hf_models.json", "llmfit-core/data/hf_models.json"]

    # Build a map of freshly scraped models (name -> model dict)
    fresh_by_name = {m["name"]: m for m in results}

    # Load existing database and merge
    existing_count = 0
    retained_count = 0
    updated_count = 0
    for output_path in output_paths:
        if os.path.exists(output_path):
            try:
                with open(output_path) as f:
                    existing = json.load(f)
                existing_count = max(existing_count, len(existing))
                for old_model in existing:
                    name = old_model.get("name", "")
                    if name in fresh_by_name:
                        updated_count += 1
                    elif name:
                        # Historical model not in current scrape — keep it
                        results.append(old_model)
                        fresh_by_name[name] = old_model
                        scraped_names.add(name)
                        retained_count += 1
            except (json.JSONDecodeError, KeyError):
                pass
            break  # Only need to load from one path

    if existing_count:
        print(f"\nMerged with existing database ({existing_count} models):")
        print(f"  Updated: {updated_count}, Retained historical: {retained_count}")

    # Sort by parameter count
    results.sort(key=lambda m: m["parameters_raw"])

    # Enrich with GGUF download sources if requested
    gguf_enriched = 0
    if args.gguf_sources:
        print(f"\nEnriching {len(results)} models with GGUF download sources...")
        gguf_enriched = enrich_gguf_sources(results, threads=args.threads)
        print(f"  Found GGUF sources for {gguf_enriched} models")

    # Write to both locations: repo root (for reference) and llmfit-core (compiled into binary)
    for output_path in output_paths:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n✅ Wrote {len(results)} models to {', '.join(output_paths)}")
    print(f"   Curated: {len(TARGET_MODELS)}, Fallbacks: {fallback_count}, "
          f"Discovered: {discovered_count}, Retained: {retained_count}, "
          f"GGUF-sourced: {gguf_enriched}")

    # Print summary table
    print(f"\n{'Model':<50} {'Params':>8} {'Min RAM':>8} {'Rec RAM':>8} {'VRAM':>6}")
    print("─" * 84)
    for m in results:
        print(f"{m['name']:<50} {m['parameter_count']:>8} "
              f"{m['min_ram_gb']:>7.1f}G {m['recommended_ram_gb']:>7.1f}G "
              f"{m['min_vram_gb']:>5.1f}G")


if __name__ == "__main__":
    main()
