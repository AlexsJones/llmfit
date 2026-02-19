#!/usr/bin/env python3
"""Verify that all models in hf_models.json exist on HuggingFace and all
Ollama mappings in src/providers.rs point to valid Ollama registry entries.

Usage:
    python3 scripts/verify_models.py            # check both
    python3 scripts/verify_models.py --hf       # HuggingFace only
    python3 scripts/verify_models.py --ollama   # Ollama only

Exits with code 1 if any model is missing. Suitable for CI.
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_MODELS_PATH = REPO_ROOT / "data" / "hf_models.json"
PROVIDERS_RS_PATH = REPO_ROOT / "src" / "providers.rs"

HEADERS = {"User-Agent": "llmfit-verify/1.0"}
REQUEST_DELAY = 0.3  # seconds between requests to avoid rate limiting


def check_url(url: str) -> int:
    """GET a URL and return the HTTP status code, or -1 on error."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# HuggingFace verification
# ---------------------------------------------------------------------------

def load_hf_models() -> list[str]:
    """Return list of HF repo names from hf_models.json."""
    with open(HF_MODELS_PATH) as f:
        data = json.load(f)
    return [m["name"] for m in data]


def verify_hf(models: list[str]) -> list[str]:
    """Check each HF model exists. Returns list of missing model names."""
    missing = []
    total = len(models)
    for i, name in enumerate(models, 1):
        url = f"https://huggingface.co/api/models/{name}"
        status = check_url(url)
        if status == 200:
            print(f"  [{i}/{total}] ✓ {name}")
        else:
            print(f"  [{i}/{total}] ✗ {name} (HTTP {status})")
            missing.append(name)
        time.sleep(REQUEST_DELAY)
    return missing


# ---------------------------------------------------------------------------
# Ollama verification
# ---------------------------------------------------------------------------

def parse_ollama_tags() -> list[str]:
    """Extract unique Ollama tags from OLLAMA_MAPPINGS in providers.rs."""
    src = PROVIDERS_RS_PATH.read_text()

    # Find the OLLAMA_MAPPINGS block
    match = re.search(
        r"const OLLAMA_MAPPINGS:.*?=.*?\[(.+?)\];",
        src,
        re.DOTALL,
    )
    if not match:
        print("ERROR: Could not find OLLAMA_MAPPINGS in providers.rs")
        sys.exit(2)

    block = match.group(1)
    # Extract the second element of each tuple: ("hf_name", "ollama_tag")
    tags = re.findall(r'\(\s*"[^"]+"\s*,\s*"([^"]+)"\s*\)', block)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def verify_ollama(tags: list[str]) -> list[str]:
    """Check each Ollama tag exists. Returns list of missing tags."""
    missing = []
    total = len(tags)
    for i, tag in enumerate(tags, 1):
        url = f"https://ollama.com/library/{tag}"
        status = check_url(url)
        if status == 200:
            print(f"  [{i}/{total}] ✓ {tag}")
        else:
            print(f"  [{i}/{total}] ✗ {tag} (HTTP {status})")
            missing.append(tag)
        time.sleep(REQUEST_DELAY)
    return missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify model availability")
    parser.add_argument("--hf", action="store_true", help="Check HuggingFace only")
    parser.add_argument("--ollama", action="store_true", help="Check Ollama only")
    args = parser.parse_args()

    # Default: check both
    check_hf = args.hf or not args.ollama
    check_ollama = args.ollama or not args.hf

    failures = False

    if check_hf:
        models = load_hf_models()
        print(f"\n=== HuggingFace: checking {len(models)} models ===\n")
        missing = verify_hf(models)
        if missing:
            failures = True
            print(f"\n  ⚠ {len(missing)} HuggingFace model(s) not found:")
            for m in missing:
                print(f"    - {m}")
        else:
            print(f"\n  All {len(models)} HuggingFace models verified ✓")

    if check_ollama:
        tags = parse_ollama_tags()
        print(f"\n=== Ollama: checking {len(tags)} tags ===\n")
        missing = verify_ollama(tags)
        if missing:
            failures = True
            print(f"\n  ⚠ {len(missing)} Ollama tag(s) not found:")
            for t in missing:
                print(f"    - {t}")
        else:
            print(f"\n  All {len(tags)} Ollama tags verified ✓")

    print()
    if failures:
        print("FAIL: Some models are unavailable. Fix mappings or remove entries.")
        sys.exit(1)
    else:
        print("PASS: All models verified.")


if __name__ == "__main__":
    main()
