#!/usr/bin/env python3
"""
One-time scraper for popular LLM models from Hugging Face.
Fetches model metadata and computes RAM/VRAM requirements from parameter counts.
Outputs a JSON file consumable by llmfit's models.rs.
"""

import json
import sys
import time
import urllib.request
import urllib.error

HF_API = "https://huggingface.co/api/models"

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
    # Code Llama
    "meta-llama/CodeLlama-7b-Instruct-hf",  # NEW: Popular code model
    "meta-llama/CodeLlama-13b-Instruct-hf",  # NEW: Larger code model
    "meta-llama/CodeLlama-34b-Instruct-hf",  # NEW: Large code model
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Ministral-8B-Instruct-2410",  # NEW: Smaller Mistral variant
    "mistralai/Mistral-Nemo-Instruct-2407",  # NEW: 12B mid-size model
    # Qwen
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    # Microsoft Phi
    "microsoft/phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-14b-instruct",
    "microsoft/Phi-3.5-mini-instruct",  # NEW: Newer Phi variant
    "microsoft/phi-4",
    # Microsoft Orca
    "microsoft/Orca-2-7b",  # NEW: Reasoning model
    "microsoft/Orca-2-13b",  # NEW: Larger reasoning model
    # Google Gemma
    "google/gemma-2-2b-it",  # NEW: Smaller variant for edge
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "google/gemma-3-12b-it",
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-V3",
    # Cohere
    "CohereForAI/c4ai-command-r-v01",
    # 01.ai Yi family
    "01-ai/Yi-6B-Chat",  # NEW: Popular multilingual 6B
    "01-ai/Yi-34B-Chat",  # NEW: Popular multilingual 34B
    # Upstage Solar
    "upstage/SOLAR-10.7B-Instruct-v1.0",  # NEW: High-performance 10.7B
    # TII Falcon
    "tiiuae/falcon-7b-instruct",  # NEW: Popular UAE model
    "tiiuae/falcon-40b-instruct",  # NEW: Large Falcon
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
    # Embeddings (useful for RAG sizing)
    "nomic-ai/nomic-embed-text-v1.5",
    "BAAI/bge-large-en-v1.5",
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
}

# Overhead multiplier for runtime memory beyond just model weights
RUNTIME_OVERHEAD = 1.2  # ~20% overhead for KV cache, activations, OS

# Known MoE (Mixture of Experts) architecture configurations
MOE_CONFIGS = {
    "mixtral": {"num_experts": 8, "active_experts": 2},
    "deepseek_v2": {"num_experts": 64, "active_experts": 6},
    "deepseek_v3": {"num_experts": 256, "active_experts": 8},
}

# Published active parameter counts for well-known MoE models
MOE_ACTIVE_PARAMS = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 12_900_000_000,
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 12_900_000_000,
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 2_400_000_000,
    "deepseek-ai/DeepSeek-V3": 37_000_000_000,
}


def fetch_model_info(repo_id: str) -> dict | None:
    """Fetch model info from HuggingFace API."""
    url = f"{HF_API}/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "llmfit-scraper/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
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
        num_experts = config.get("num_local_experts")
        active_experts = config.get("num_experts_per_tok")

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
    for key in [
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
        "sliding_window",
    ]:
        if key in config:
            val = config[key]
            if isinstance(val, int) and val > 0:
                return val
    return 4096


def fetch_config_json(repo_id: str) -> dict | None:
    """Fetch the full config.json from a HF repo (has max_position_embeddings)."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "llmfit-scraper/1.0"})
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
    }
    return mapping.get(org, org)


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
    default_quant = "Q4_K_M"

    # Fetch full config.json for accurate context length
    full_config = fetch_config_json(repo_id)
    context_length = infer_context_length(full_config) if full_config else infer_context_length(config)

    min_ram, rec_ram = estimate_ram(total_params, default_quant)
    min_vram = estimate_vram(total_params, default_quant)

    architecture = config.get("model_type", "unknown")

    # Detect MoE architecture
    moe_info = detect_moe(repo_id, full_config, architecture, total_params)

    result = {
        "name": repo_id,
        "provider": extract_provider(repo_id),
        "parameter_count": format_param_count(total_params),
        "parameters_raw": total_params,
        "min_ram_gb": min_ram,
        "recommended_ram_gb": rec_ram,
        "min_vram_gb": min_vram,
        "quantization": default_quant,
        "context_length": context_length,
        "use_case": infer_use_case(repo_id, pipeline_tag, config),
        "pipeline_tag": pipeline_tag or "unknown",
        "architecture": architecture,
        "hf_downloads": info.get("downloads", 0),
        "hf_likes": info.get("likes", 0),
    }

    # Add MoE fields if detected
    if moe_info["is_moe"]:
        result["is_moe"] = True
        result["num_experts"] = moe_info["num_experts"]
        result["active_experts"] = moe_info["active_experts"]
        result["active_parameters"] = moe_info["active_parameters"]

    return result


def main():
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
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "mistralai/Mistral-Small-24B-Instruct-2501",
            "provider": "Mistral AI", "parameter_count": "24B",
            "parameters_raw": 24_000_000_000,
            "min_ram_gb": 13.4, "recommended_ram_gb": 22.4, "min_vram_gb": 12.3,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "mistral",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "Qwen/Qwen2.5-14B-Instruct",
            "provider": "Alibaba", "parameter_count": "14.8B",
            "parameters_raw": 14_770_000_000,
            "min_ram_gb": 8.2, "recommended_ram_gb": 13.7, "min_vram_gb": 7.6,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "qwen2",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "Qwen/Qwen2.5-32B-Instruct",
            "provider": "Alibaba", "parameter_count": "32.5B",
            "parameters_raw": 32_510_000_000,
            "min_ram_gb": 18.2, "recommended_ram_gb": 30.3, "min_vram_gb": 16.7,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "qwen2",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "microsoft/phi-3-mini-4k-instruct",
            "provider": "Microsoft", "parameter_count": "3.8B",
            "parameters_raw": 3_821_000_000,
            "min_ram_gb": 2.1, "recommended_ram_gb": 3.6, "min_vram_gb": 2.0,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Lightweight, edge deployment",
            "pipeline_tag": "text-generation", "architecture": "phi3",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "microsoft/phi-4",
            "provider": "Microsoft", "parameter_count": "14B",
            "parameters_raw": 14_000_000_000,
            "min_ram_gb": 7.8, "recommended_ram_gb": 13.0, "min_vram_gb": 7.2,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Reasoning, STEM, code generation",
            "pipeline_tag": "text-generation", "architecture": "phi",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "google/gemma-3-12b-it",
            "provider": "Google", "parameter_count": "12B",
            "parameters_raw": 12_000_000_000,
            "min_ram_gb": 6.7, "recommended_ram_gb": 11.2, "min_vram_gb": 6.1,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "text-generation", "architecture": "gemma3",
            "hf_downloads": 0, "hf_likes": 0,
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
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "CohereForAI/c4ai-command-r-v01",
            "provider": "Cohere", "parameter_count": "35B",
            "parameters_raw": 35_000_000_000,
            "min_ram_gb": 19.5, "recommended_ram_gb": 32.6, "min_vram_gb": 17.9,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "RAG, tool use, agents",
            "pipeline_tag": "text-generation", "architecture": "cohere",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "bigcode/starcoder2-15b",
            "provider": "BigCode", "parameter_count": "15.7B",
            "parameters_raw": 15_700_000_000,
            "min_ram_gb": 8.8, "recommended_ram_gb": 14.6, "min_vram_gb": 8.0,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "starcoder2",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "nomic-ai/nomic-embed-text-v1.5",
            "provider": "Nomic", "parameter_count": "137M",
            "parameters_raw": 137_000_000,
            "min_ram_gb": 1.0, "recommended_ram_gb": 2.0, "min_vram_gb": 0.5,
            "quantization": "F16", "context_length": 8192,
            "use_case": "Text embeddings for RAG",
            "pipeline_tag": "feature-extraction", "architecture": "nomic_bert",
            "hf_downloads": 0, "hf_likes": 0,
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
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "microsoft/Phi-3-medium-14b-instruct",
            "provider": "Microsoft", "parameter_count": "14B",
            "parameters_raw": 14_000_000_000,
            "min_ram_gb": 7.8, "recommended_ram_gb": 13.0, "min_vram_gb": 7.2,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Balanced performance and size",
            "pipeline_tag": "text-generation", "architecture": "phi3",
            "hf_downloads": 0, "hf_likes": 0,
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
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "meta-llama/CodeLlama-7b-Instruct-hf",
            "provider": "Meta", "parameter_count": "7.0B",
            "parameters_raw": 7016400896,
            "min_ram_gb": 3.9, "recommended_ram_gb": 6.5, "min_vram_gb": 3.6,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "meta-llama/CodeLlama-13b-Instruct-hf",
            "provider": "Meta", "parameter_count": "13.0B",
            "parameters_raw": 13015864320,
            "min_ram_gb": 7.3, "recommended_ram_gb": 12.1, "min_vram_gb": 6.7,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "meta-llama/CodeLlama-34b-Instruct-hf",
            "provider": "Meta", "parameter_count": "34.0B",
            "parameters_raw": 34018971648,
            "min_ram_gb": 19.0, "recommended_ram_gb": 31.7, "min_vram_gb": 17.4,
            "quantization": "Q4_K_M", "context_length": 16384,
            "use_case": "Code generation and completion",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "provider": "Meta", "parameter_count": "11.0B",
            "parameters_raw": 10665463808,
            "min_ram_gb": 6.0, "recommended_ram_gb": 9.9, "min_vram_gb": 5.5,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Multimodal, vision and text",
            "pipeline_tag": "image-text-to-text", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "mistralai/Ministral-8B-Instruct-2410",
            "provider": "Mistral AI", "parameter_count": "8.0B",
            "parameters_raw": 8030261248,
            "min_ram_gb": 4.5, "recommended_ram_gb": 7.5, "min_vram_gb": 4.1,
            "quantization": "Q4_K_M", "context_length": 32768,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "mistral",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "mistralai/Mistral-Nemo-Instruct-2407",
            "provider": "Mistral AI", "parameter_count": "12.2B",
            "parameters_raw": 12247076864,
            "min_ram_gb": 6.8, "recommended_ram_gb": 11.4, "min_vram_gb": 6.3,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Instruction following, chat",
            "pipeline_tag": "text-generation", "architecture": "mistral",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "microsoft/Phi-3.5-mini-instruct",
            "provider": "Microsoft", "parameter_count": "3.8B",
            "parameters_raw": 3821000000,
            "min_ram_gb": 2.1, "recommended_ram_gb": 3.6, "min_vram_gb": 2.0,
            "quantization": "Q4_K_M", "context_length": 131072,
            "use_case": "Lightweight, long context",
            "pipeline_tag": "text-generation", "architecture": "phi3",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "microsoft/Orca-2-7b",
            "provider": "Microsoft", "parameter_count": "7.0B",
            "parameters_raw": 7016400896,
            "min_ram_gb": 3.9, "recommended_ram_gb": 6.5, "min_vram_gb": 3.6,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Reasoning, step-by-step solutions",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "microsoft/Orca-2-13b",
            "provider": "Microsoft", "parameter_count": "13.0B",
            "parameters_raw": 13015864320,
            "min_ram_gb": 7.3, "recommended_ram_gb": 12.1, "min_vram_gb": 6.7,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Reasoning, step-by-step solutions",
            "pipeline_tag": "text-generation", "architecture": "llama",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "01-ai/Yi-6B-Chat",
            "provider": "01.ai", "parameter_count": "6.1B",
            "parameters_raw": 6061356032,
            "min_ram_gb": 3.4, "recommended_ram_gb": 5.6, "min_vram_gb": 3.1,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Multilingual, Chinese/English chat",
            "pipeline_tag": "text-generation", "architecture": "yi",
            "hf_downloads": 0, "hf_likes": 0,
        },
        {
            "name": "01-ai/Yi-34B-Chat",
            "provider": "01.ai", "parameter_count": "34.4B",
            "parameters_raw": 34386780160,
            "min_ram_gb": 19.2, "recommended_ram_gb": 32.0, "min_vram_gb": 17.6,
            "quantization": "Q4_K_M", "context_length": 4096,
            "use_case": "Multilingual, Chinese/English chat",
            "pipeline_tag": "text-generation", "architecture": "yi",
            "hf_downloads": 0, "hf_likes": 0,
        },
    ]

    print(f"Scraping {len(TARGET_MODELS)} models from HuggingFace...\n")

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
            fallback_count += 1

    # Sort by parameter count
    results.sort(key=lambda m: m["parameters_raw"])

    output_path = "data/hf_models.json"
    import os
    os.makedirs("data", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Wrote {len(results)} models to {output_path}")
    print(f"   Scraped: {len(scraped_names)}, Fallbacks: {fallback_count}")

    # Print summary table
    print(f"\n{'Model':<50} {'Params':>8} {'Min RAM':>8} {'Rec RAM':>8} {'VRAM':>6}")
    print("─" * 84)
    for m in results:
        print(f"{m['name']:<50} {m['parameter_count']:>8} "
              f"{m['min_ram_gb']:>7.1f}G {m['recommended_ram_gb']:>7.1f}G "
              f"{m['min_vram_gb']:>5.1f}G")


if __name__ == "__main__":
    main()
