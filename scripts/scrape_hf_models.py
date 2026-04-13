#!/usr/bin/env python3
"""
Scraper for popular LLM models from Hugging Face.
Fetches model metadata and computes RAM/VRAM requirements from parameter counts.
Outputs a JSON file consumable by llmfit's models.rs.

Usage:
  python3 scrape_hf_models.py                  # Curated list only
  python3 scrape_hf_models.py --discover        # Curated + top trending models
  python3 scrape_hf_models.py --discover -n 50  # Curated + top 50 trending
"""

import argparse
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
    "Qwen/Qwen3-30B-A3B": 3_300_000_000,
    "Qwen/Qwen3-235B-A22B": 22_000_000_000,
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": 35_000_000_000,
    "Qwen/Qwen3-Coder-Next": 3_000_000_000,
    "Qwen/Qwen3.5-35B-A3B": 3_000_000_000,
    "Qwen/Qwen3.5-122B-A10B": 10_000_000_000,
    "Qwen/Qwen3.5-397B-A17B": 17_000_000_000,
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


def detect_moe(repo_id: str, config: dict | None, architecture: str,
               total_params: int) -> dict:
    """Detect MoE architecture and compute active parameters."""
    result = {
        "is_moe": False,
        "num_experts": None,
        "active_experts": None,
        "active_parameters": None,
    }

    # Check config.json for MoE indicators
    num_experts = None
    active_experts = None
    if config:
        num_experts = config.get("num_local_experts") or config.get("num_experts")
        active_experts = config.get("num_experts_per_tok") or config.get("top_k_experts")

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

    # Check top-level config
    for key in keys_to_check:
        if key in config:
            val = config[key]
            if isinstance(val, int) and val > 0:
                return val

    # For multimodal models (e.g., Qwen3.5), check text_config
    if "text_config" in config and isinstance(config["text_config"], dict):
        for key in keys_to_check:
            if key in config["text_config"]:
                val = config["text_config"][key]
                if isinstance(val, int) and val > 0:
                    return val

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

    min_ram, rec_ram = estimate_ram(total_params, default_quant)
    min_vram = estimate_vram(total_params, default_quant)

    architecture = config.get("model_type", "unknown")

    # Detect MoE architecture
    moe_info = detect_moe(repo_id, full_config, architecture, total_params)

    use_case_str = infer_use_case(repo_id, pipeline_tag, config)

    # Architecture metadata for the precise KV cache formula. All optional;
    # absent fields cause the Rust side to fall back to the linear approx.
    num_hidden_layers = (full_config or {}).get("num_hidden_layers")
    num_attention_heads = (full_config or {}).get("num_attention_heads")
    num_key_value_heads = (full_config or {}).get("num_key_value_heads") or num_attention_heads
    head_dim = (full_config or {}).get("head_dim")
    if head_dim is None and num_attention_heads:
        hidden_size = (full_config or {}).get("hidden_size")
        if hidden_size:
            head_dim = hidden_size // num_attention_heads

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
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
    }

    # Add MoE fields if detected
    if moe_info["is_moe"]:
        result["is_moe"] = True
        result["num_experts"] = moe_info["num_experts"]
        result["active_experts"] = moe_info["active_experts"]
        result["active_parameters"] = moe_info["active_parameters"]

    return result


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


def enrich_gguf_sources(models: list[dict]) -> int:
    """Add gguf_sources to models by checking GGUF provider repos.

    Uses a persistent cache to avoid re-checking repos on every scrape.
    Returns the number of models enriched.
    """
    cache = _load_gguf_cache()
    enriched = 0
    cache_hits = 0
    total = len(models)
    from datetime import datetime, timezone

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
            # Query HuggingFace
            candidates = _model_gguf_repo_candidates(repo_id)
            sources = []
            for provider, candidate_repo in candidates:
                print(f"  [{i}/{total}] Checking {candidate_repo}...", end="")
                if check_gguf_repo_exists(candidate_repo):
                    sources.append({"repo": candidate_repo, "provider": provider})
                    print(" ✓")
                else:
                    print(" ✗")
                time.sleep(0.15)  # Be polite to the API

            # Update cache
            cache[repo_id] = {
                "sources": sources,
                "checked": datetime.now(timezone.utc).isoformat(),
            }

        if sources:
            model["gguf_sources"] = sources
            enriched += 1

    _save_gguf_cache(cache)
    print(f"  Cache: {cache_hits} hits, {total - cache_hits} API checks")
    return enriched


# ---------------------------------------------------------------------------
# Auto-discovery from HuggingFace trending / most-downloaded
# ---------------------------------------------------------------------------

# Pipeline tags to search for discoverable models
DISCOVER_PIPELINES = ["text-generation", "text2text-generation", "image-text-to-text"]

# Orgs to skip — these publish many fine-tunes that clutter the list
SKIP_ORGS = {
    "TheBloke",               # GGUF repacks, not original models
    "unsloth",                # Training framework repacks
    "mlx-community",          # MLX conversions
    "bartowski",              # GGUF repacks
    "mradermacher",           # GGUF repacks
    "trl-internal-testing",   # Test fixtures
    "openai-community",       # Legacy model mirrors (gpt2 etc.)
    "distilbert",             # Distilled legacy models
}


def discover_trending_models(limit: int = 30, min_downloads: int = 10000) -> list[dict]:
    """Query HuggingFace API for top text-generation models by download count.

    Uses ?expand=safetensors to get parameter counts directly from the listing
    API, avoiding individual API calls per model (per HF team recommendation).

    Returns a list of dicts with model listing data (including safetensors
    metadata) for models NOT already in TARGET_MODELS.
    """
    curated = set(TARGET_MODELS)
    discovered = []
    seen_ids = set()

    for pipeline in DISCOVER_PIPELINES:
        # Fetch more than we need since we'll filter heavily
        fetch_limit = min(limit * 8, 10000)  # HF API max is 10000
        url = (
            f"{HF_API}?"
            f"pipeline_tag={pipeline}&"
            f"sort=downloads&"
            f"direction=-1&"
            f"limit={fetch_limit}&"
            f"expand[]=safetensors&"
            f"expand[]=config"
        )
        req = urllib.request.Request(url, headers=_auth_headers())
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                models = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  ⚠ Failed to fetch trending {pipeline} models: {e}",
                  file=sys.stderr)
            continue

        for m in models:
            repo_id = m.get("id", "")
            if not repo_id or "/" not in repo_id:
                continue

            # Skip if already curated or seen
            if repo_id in curated or repo_id in seen_ids:
                continue
            seen_ids.add(repo_id)

            # Skip known repack / converter orgs
            org = repo_id.split("/")[0]
            if org in SKIP_ORGS:
                continue

            # Skip models with too few downloads
            downloads = m.get("downloads", 0)
            if downloads < min_downloads:
                continue

            # Skip GGUF-only repos, adapters, and merges
            tags = set(m.get("tags", []))
            if tags & {"gguf", "adapter", "merge", "lora", "qlora"}:
                continue

            # Check for actual parameter count from expand=safetensors
            # (replaces old safetensors tag check — many models have data but no tag)
            safetensors = m.get("safetensors", {})
            total_params = safetensors.get("total")
            if not total_params:
                params_by_dtype = safetensors.get("parameters", {})
                if params_by_dtype:
                    total_params = max(params_by_dtype.values())
            if not total_params:
                continue  # no param data available

            # Attach param count for downstream use
            m["_total_params"] = total_params
            discovered.append(m)
            if len(discovered) >= limit:
                break

        if len(discovered) >= limit:
            break

    return discovered[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Scrape LLM model metadata from HuggingFace for llmfit."
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Auto-discover trending text-generation models from HuggingFace "
             "in addition to the curated TARGET_MODELS list."
    )
    parser.add_argument(
        "-n", "--discover-limit", type=int, default=30,
        help="Max number of trending models to discover (default: 30). "
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
    args = parser.parse_args()

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
    ]

    print(f"Scraping {len(TARGET_MODELS)} curated models from HuggingFace...\n")

    results = []
    scraped_names = set()
    for i, repo_id in enumerate(TARGET_MODELS, 1):
        print(f"[{i}/{len(TARGET_MODELS)}] {repo_id}...")
        model = scrape_model(repo_id)
        if model:
            print(f"  ✓ {model['parameter_count']} params, "
                  f"min {model['min_ram_gb']} GB RAM, "
                  f"ctx {model['context_length']}")
            results.append(model)
            scraped_names.add(repo_id)
        # Be polite to the API
        time.sleep(0.3)

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
        print(f"\nDiscovering trending models (limit={args.discover_limit}, "
              f"min_downloads={args.min_downloads})...")
        trending = discover_trending_models(
            limit=args.discover_limit,
            min_downloads=args.min_downloads,
        )
        print(f"  Found {len(trending)} new models not in curated list\n")

        for i, listing in enumerate(trending, 1):
            repo_id = listing["id"]
            if repo_id in scraped_names:
                continue
            print(f"[discover {i}/{len(trending)}] {repo_id}...")

            # Build model from listing data (param count already available
            # from expand=safetensors, avoiding an extra API call per model)
            total_params = listing["_total_params"]
            config = listing.get("config", {})
            pipeline_tag = listing.get("pipeline_tag")

            # Still need config.json for accurate context length
            full_config = fetch_config_json(repo_id)

            model_format, default_quant = detect_quant_format(repo_id, full_config)
            context_length = infer_context_length(full_config) if full_config else infer_context_length(config)

            min_ram, rec_ram = estimate_ram(total_params, default_quant)
            min_vram = estimate_vram(total_params, default_quant)

            architecture = config.get("model_type", "unknown")
            moe_info = detect_moe(repo_id, full_config, architecture, total_params)
            use_case_str = infer_use_case(repo_id, pipeline_tag, config)

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
                "_discovered": True,
            }

            if moe_info["is_moe"]:
                model["is_moe"] = True
                model["num_experts"] = moe_info["num_experts"]
                model["active_experts"] = moe_info["active_experts"]
                model["active_parameters"] = moe_info["active_parameters"]

            print(f"  ✓ {model['parameter_count']} params, "
                  f"{model['hf_downloads']:,} downloads, "
                  f"ctx {model['context_length']}")
            results.append(model)
            scraped_names.add(repo_id)
            discovered_count += 1
            time.sleep(0.15)  # Only fetching config.json now, can be faster

    # Sort by parameter count
    results.sort(key=lambda m: m["parameters_raw"])

    # Enrich with GGUF download sources if requested
    gguf_enriched = 0
    if args.gguf_sources:
        print(f"\nEnriching {len(results)} models with GGUF download sources...")
        gguf_enriched = enrich_gguf_sources(results)
        print(f"  Found GGUF sources for {gguf_enriched} models")

    # Write to both locations: repo root (for reference) and llmfit-core (compiled into binary)
    output_paths = ["data/hf_models.json", "llmfit-core/data/hf_models.json"]
    for output_path in output_paths:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n✅ Wrote {len(results)} models to {', '.join(output_paths)}")
    print(f"   Curated: {len(TARGET_MODELS)}, Fallbacks: {fallback_count}, "
          f"Discovered: {discovered_count}, GGUF-sourced: {gguf_enriched}")

    # Print summary table
    print(f"\n{'Model':<50} {'Params':>8} {'Min RAM':>8} {'Rec RAM':>8} {'VRAM':>6}")
    print("─" * 84)
    for m in results:
        print(f"{m['name']:<50} {m['parameter_count']:>8} "
              f"{m['min_ram_gb']:>7.1f}G {m['recommended_ram_gb']:>7.1f}G "
              f"{m['min_vram_gb']:>5.1f}G")


if __name__ == "__main__":
    main()
