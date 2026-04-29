#!/usr/bin/env python3
"""
Scraper for Docker Model Runner available models.

Queries the Docker Hub API for models in the 'ai/' namespace,
cross-references against llmfit's HF model database and Ollama mapping table,
and outputs a JSON mapping of HF model names to Docker Model Runner tags.

Usage:
  python3 scripts/scrape_docker_models.py
"""

import json
import os
import sys
import urllib.request
import urllib.error

DOCKER_HUB_API = "https://hub.docker.com/v2/repositories/ai/"
PAGE_SIZE = 100

# Same mapping as OLLAMA_MAPPINGS in providers.rs.
# Maps lowercased HF repo suffix → Ollama-style tag (without ai/ prefix).
OLLAMA_MAPPINGS = {
    # Meta Llama family
    "llama-3.3-70b-instruct": "llama3.3:70b",
    "llama-3.2-11b-vision-instruct": "llama3.2-vision:11b",
    "llama-3.2-3b-instruct": "llama3.2:3b",
    "llama-3.2-3b": "llama3.2:3b",
    "llama-3.2-1b-instruct": "llama3.2:1b",
    "llama-3.2-1b": "llama3.2:1b",
    "llama-3.1-405b-instruct": "llama3.1:405b",
    "llama-3.1-405b": "llama3.1:405b",
    "llama-3.1-70b-instruct": "llama3.1:70b",
    "llama-3.1-8b-instruct": "llama3.1:8b",
    "llama-3.1-8b": "llama3.1:8b",
    "meta-llama-3-8b-instruct": "llama3:8b",
    "meta-llama-3-8b": "llama3:8b",
    "llama-2-7b-hf": "llama2:7b",
    "codellama-34b-instruct-hf": "codellama:34b",
    "codellama-13b-instruct-hf": "codellama:13b",
    "codellama-7b-instruct-hf": "codellama:7b",
    # Google Gemma
    "gemma-3-12b-it": "gemma3:12b",
    "gemma-2-27b-it": "gemma2:27b",
    "gemma-2-9b-it": "gemma2:9b",
    "gemma-2-2b-it": "gemma2:2b",
    # Microsoft Phi
    "phi-4": "phi4",
    "phi-4-mini-instruct": "phi4-mini",
    "phi-3.5-mini-instruct": "phi3.5",
    "phi-3-mini-4k-instruct": "phi3",
    "phi-3-medium-14b-instruct": "phi3:14b",
    "phi-2": "phi",
    "orca-2-7b": "orca2:7b",
    "orca-2-13b": "orca2:13b",
    # Mistral
    "mistral-7b-instruct-v0.3": "mistral:7b",
    "mistral-7b-instruct-v0.2": "mistral:7b",
    "mistral-nemo-instruct-2407": "mistral-nemo",
    "mistral-small-24b-instruct-2501": "mistral-small:24b",
    "mistral-large-instruct-2407": "mistral-large",
    "mixtral-8x7b-instruct-v0.1": "mixtral:8x7b",
    "mixtral-8x22b-instruct-v0.1": "mixtral:8x22b",
    # Qwen 2 / 2.5
    "qwen2-1.5b-instruct": "qwen2:1.5b",
    "qwen2.5-72b-instruct": "qwen2.5:72b",
    "qwen2.5-32b-instruct": "qwen2.5:32b",
    "qwen2.5-14b-instruct": "qwen2.5:14b",
    "qwen2.5-7b-instruct": "qwen2.5:7b",
    "qwen2.5-7b": "qwen2.5:7b",
    "qwen2.5-3b-instruct": "qwen2.5:3b",
    "qwen2.5-1.5b-instruct": "qwen2.5:1.5b",
    "qwen2.5-1.5b": "qwen2.5:1.5b",
    "qwen2.5-0.5b-instruct": "qwen2.5:0.5b",
    "qwen2.5-0.5b": "qwen2.5:0.5b",
    "qwen2.5-coder-32b-instruct": "qwen2.5-coder:32b",
    "qwen2.5-coder-14b-instruct": "qwen2.5-coder:14b",
    "qwen2.5-coder-7b-instruct": "qwen2.5-coder:7b",
    "qwen2.5-coder-1.5b-instruct": "qwen2.5-coder:1.5b",
    "qwen2.5-coder-0.5b-instruct": "qwen2.5-coder:0.5b",
    "qwen2.5-vl-7b-instruct": "qwen2.5vl:7b",
    "qwen2.5-vl-3b-instruct": "qwen2.5vl:3b",
    # Qwen 3
    "qwen3-235b-a22b": "qwen3:235b",
    "qwen3-32b": "qwen3:32b",
    "qwen3-30b-a3b": "qwen3:30b-a3b",
    "qwen3-30b-a3b-instruct-2507": "qwen3:30b-a3b",
    "qwen3-14b": "qwen3:14b",
    "qwen3-8b": "qwen3:8b",
    "qwen3-4b": "qwen3:4b",
    "qwen3-4b-instruct-2507": "qwen3:4b",
    "qwen3-1.7b-base": "qwen3:1.7b",
    "qwen3-0.6b": "qwen3:0.6b",
    "qwen3-coder-30b-a3b-instruct": "qwen3-coder",
    # Qwen 3.5
    "qwen3.5-27b": "qwen3.5",
    "qwen3.5-35b-a3b": "qwen3.5:35b",
    "qwen3.5-122b-a10b": "qwen3.5:122b",
    # Qwen3-Coder-Next
    "qwen3-coder-next": "qwen3-coder-next",
    # DeepSeek
    "deepseek-v3": "deepseek-v3",
    "deepseek-v3.2": "deepseek-v3",
    "deepseek-r1": "deepseek-r1",
    "deepseek-r1-0528": "deepseek-r1",
    "deepseek-r1-distill-qwen-32b": "deepseek-r1:32b",
    "deepseek-r1-distill-qwen-14b": "deepseek-r1:14b",
    "deepseek-r1-distill-qwen-7b": "deepseek-r1:7b",
    "deepseek-coder-v2-lite-instruct": "deepseek-coder-v2:16b",
    # Community / other
    "tinyllama-1.1b-chat-v1.0": "tinyllama",
    "stablelm-2-1_6b-chat": "stablelm2:1.6b",
    "yi-6b-chat": "yi:6b",
    "yi-34b-chat": "yi:34b",
    "starcoder2-7b": "starcoder2:7b",
    "starcoder2-15b": "starcoder2:15b",
    "falcon-7b-instruct": "falcon:7b",
    "falcon-40b-instruct": "falcon:40b",
    "falcon-180b-chat": "falcon:180b",
    "falcon3-7b-instruct": "falcon3:7b",
    "openchat-3.5-0106": "openchat:7b",
    "vicuna-7b-v1.5": "vicuna:7b",
    "vicuna-13b-v1.5": "vicuna:13b",
    "glm-4-9b-chat": "glm4:9b",
    "glm-4-9b-0414": "glm4:9b",
    "glm-4-32b-0414": "glm4:32b",
    "glm-z1-9b-0414": "glm-z1:9b",
    "glm-z1-32b-0414": "glm-z1:32b",
    "solar-10.7b-instruct-v1.0": "solar:10.7b",
    "zephyr-7b-beta": "zephyr:7b",
    "c4ai-command-r-v01": "command-r",
    "nous-hermes-2-mixtral-8x7b-dpo": "nous-hermes2-mixtral:8x7b",
    "hermes-3-llama-3.1-8b": "hermes3:8b",
    "nomic-embed-text-v1.5": "nomic-embed-text",
    "bge-large-en-v1.5": "bge-large",
    "smollm2-135m-instruct": "smollm2:135m",
    "smollm2-135m": "smollm2:135m",
    # Google Gemma 3n
    "gemma-3n-e4b-it": "gemma3n:e4b",
    "gemma-3n-e2b-it": "gemma3n:e2b",
    # Microsoft Phi-4 reasoning
    "phi-4-reasoning": "phi4-reasoning",
    "phi-4-mini-reasoning": "phi4-mini-reasoning",
    # DeepSeek V3.2 Speciale
    "deepseek-v3.2-speciale": "deepseek-v3",
    # Liquid AI LFM2
    "lfm2-350m": "lfm2:350m",
    "lfm2-700m": "lfm2:700m",
    "lfm2-1.2b": "lfm2:1.2b",
    "lfm2-2.6b": "lfm2:2.6b",
    "lfm2-2.6b-exp": "lfm2:2.6b",
    "lfm2-8b-a1b": "lfm2:8b-a1b",
    "lfm2-24b-a2b": "lfm2:24b",
    # Liquid AI LFM2.5
    "lfm2.5-1.2b-instruct": "lfm2.5:1.2b",
    "lfm2.5-1.2b-thinking": "lfm2.5-thinking:1.2b",
}


def fetch_docker_hub_models() -> list[str]:
    """Fetch all model names from the Docker Hub ai/ namespace."""
    models = []
    url = f"{DOCKER_HUB_API}?page_size={PAGE_SIZE}"

    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "llmfit-scraper/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"Error fetching {url}: {e}", file=sys.stderr)
            break

        for repo in data.get("results", []):
            name = repo.get("name", "")
            if name:
                models.append(name)

        url = data.get("next")

    return models


def fetch_tags_for_model(model_name: str) -> list[str]:
    """Fetch available tags for a Docker Hub ai/ model."""
    url = f"{DOCKER_HUB_API}{model_name}/tags/?page_size=100"
    req = urllib.request.Request(url, headers={"User-Agent": "llmfit-scraper/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError):
        return []

    return [t["name"] for t in data.get("results", []) if t.get("name")]


def ollama_tag_to_docker_repo(ollama_tag: str) -> str:
    """Extract the Docker Hub repo name from an Ollama tag.

    E.g. "llama3.1:8b" → "llama3.1", "phi4" → "phi4"
    """
    return ollama_tag.split(":")[0]


def lookup_ollama_tag(hf_name: str) -> str | None:
    """Mirror the Rust lookup_ollama_tag logic.

    Extract the repo suffix (after last '/'), lowercase it,
    and look it up in the OLLAMA_MAPPINGS dict.
    """
    suffix = hf_name.rsplit("/", 1)[-1].lower()
    return OLLAMA_MAPPINGS.get(suffix)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_file = os.path.join(project_root, "llmfit-core", "data", "docker_models.json")

    # Load the HF model database to get all model names
    hf_models_file = os.path.join(project_root, "llmfit-core", "data", "hf_models.json")
    with open(hf_models_file) as f:
        hf_models = json.load(f)

    print(f"Loaded {len(hf_models)} models from HF database")

    # Fetch all available Docker Hub ai/ models
    print("Fetching Docker Hub ai/ namespace...")
    docker_repos = fetch_docker_hub_models()
    # Filter out vllm/safetensors variants — these are alternative serving formats,
    # not standard Model Runner models
    docker_repos = [r for r in docker_repos if not r.endswith(("-vllm", "-safetensors"))]
    docker_repo_set = set(docker_repos)
    print(f"Found {len(docker_repos)} Docker Model Runner repos (excl. vllm/safetensors variants)")

    # Fetch tags for each available repo
    print("Fetching tags for each repo...")
    repo_tags: dict[str, list[str]] = {}
    for repo in sorted(docker_repos):
        tags = fetch_tags_for_model(repo)
        repo_tags[repo] = tags
        tag_str = ", ".join(tags[:5])
        if len(tags) > 5:
            tag_str += f", ... ({len(tags)} total)"
        print(f"  ai/{repo}: [{tag_str}]")

    # Cross-reference: for each HF model, check if its Ollama tag maps to a
    # Docker Hub repo. Uses the same lookup logic as Rust's lookup_ollama_tag().
    mappings = []
    matched = 0
    unmatched_models = []

    for model in hf_models:
        hf_name = model["name"]

        ollama_tag = lookup_ollama_tag(hf_name)
        if not ollama_tag:
            unmatched_models.append(hf_name)
            continue

        docker_repo = ollama_tag_to_docker_repo(ollama_tag)
        if docker_repo not in docker_repo_set:
            unmatched_models.append(hf_name)
            continue

        # Build the full Docker tag: ai/<repo>:<size> or ai/<repo>
        docker_tag = f"ai/{ollama_tag}"
        available_tags = repo_tags.get(docker_repo, [])

        mappings.append({
            "hf_name": hf_name,
            "docker_tag": docker_tag,
            "docker_repo": f"ai/{docker_repo}",
            "available_tags": available_tags,
        })
        matched += 1

    print()
    print(f"Matched: {matched}/{len(hf_models)} models have Docker Model Runner images")

    if unmatched_models:
        print(f"Unmatched: {len(unmatched_models)} models (no Ollama mapping or no Docker repo)")

    # Write output
    output = {
        "generated_by": "scrape_docker_models.py",
        "docker_hub_repo_count": len(docker_repos),
        "matched_model_count": matched,
        "models": mappings,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"\nWrote {output_file}")


if __name__ == "__main__":
    main()
