#!/usr/bin/env python3
"""
Scraper for localmaxxing.com benchmark data.
Fetches leaderboard results for all hardware presets and caches them locally
so the TUI has a fallback when the API is unreachable.

Usage:
  python3 scrape_benchmarks.py                    # Scrape all presets, 100 results each
  python3 scrape_benchmarks.py --limit 50         # Fewer results per preset
  python3 scrape_benchmarks.py --api-key bhk_...  # Use API key for auth
  python3 scrape_benchmarks.py --presets "RTX 4090,M4 Max"  # Specific presets only

Output:
  data/benchmark_cache.json        (repo root, for reference)
  llmfit-core/data/benchmark_cache.json  (compiled into binary)

The cache format is:
  {
    "scraped_at": "2026-04-27T15:00:00Z",
    "presets": {
      "RTX 4090 (24 GB)": { "rows": [...], "total": 47 },
      "Apple M4 Max (128 GB)": { "rows": [...], "total": 12 },
      ...
    }
  }
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

BASE_URL = "https://localmaxxing.com/api"

# Mirror of the Rust HARDWARE_PRESETS — keep in sync with benchmarks.rs
HARDWARE_PRESETS = [
    # NVIDIA consumer
    {"label": "RTX 5090 (32 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "RTX 5090", "memTier": 32},
    {"label": "RTX 5080 (16 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "RTX 5080", "memTier": 16},
    {"label": "RTX 4090 (24 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "RTX 4090", "memTier": 24},
    {"label": "RTX 4080 (16 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "RTX 4080", "memTier": 16},
    {"label": "RTX 4070 Ti (12 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "RTX 4070", "memTier": 12},
    {"label": "RTX 3090 (24 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "RTX 3090", "memTier": 24},
    {"label": "RTX 3080 (10 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "RTX 3080", "memTier": 12},
    {"label": "RTX 3060 (12 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "RTX 3060", "memTier": 12},
    # NVIDIA datacenter
    {"label": "A100 (80 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "A100", "memTier": 80},
    {"label": "A100 (40 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "A100", "memTier": 48},
    {"label": "H100 (80 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "H100", "memTier": 80},
    {"label": "L40S (48 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "L40S", "memTier": 48},
    {"label": "T4 (16 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "T4", "memTier": 16},
    # AMD
    {"label": "RX 7900 XTX (24 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "7900 XTX", "memTier": 24},
    {"label": "RX 7900 XT (20 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "7900 XT", "memTier": 24},
    {"label": "MI300X (192 GB)", "hwClass": "DISCRETE_GPU", "hardwareName": "MI300X", "memTier": 128},
    # Apple Silicon
    {"label": "Apple M4 Max (128 GB)", "hwClass": "UNIFIED", "hardwareName": "M4 Max", "memTier": 128},
    {"label": "Apple M4 Max (64 GB)", "hwClass": "UNIFIED", "hardwareName": "M4 Max", "memTier": 48},
    {"label": "Apple M4 Pro (48 GB)", "hwClass": "UNIFIED", "hardwareName": "M4 Pro", "memTier": 48},
    {"label": "Apple M4 Pro (24 GB)", "hwClass": "UNIFIED", "hardwareName": "M4 Pro", "memTier": 24},
    {"label": "Apple M3 Max (128 GB)", "hwClass": "UNIFIED", "hardwareName": "M3 Max", "memTier": 128},
    {"label": "Apple M3 Max (96 GB)", "hwClass": "UNIFIED", "hardwareName": "M3 Max", "memTier": 96},
    {"label": "Apple M2 Ultra (192 GB)", "hwClass": "UNIFIED", "hardwareName": "M2 Ultra", "memTier": 128},
    {"label": "Apple M2 Max (96 GB)", "hwClass": "UNIFIED", "hardwareName": "M2 Max", "memTier": 96},
    {"label": "Apple M2 Pro (32 GB)", "hwClass": "UNIFIED", "hardwareName": "M2 Pro", "memTier": 32},
    {"label": "Apple M1 Max (64 GB)", "hwClass": "UNIFIED", "hardwareName": "M1 Max", "memTier": 48},
    # CPU only
    {"label": "CPU Only", "hwClass": "CPU_ONLY", "hardwareName": None, "memTier": None},
]


def fetch_leaderboard(preset: dict, api_key: str | None, limit: int) -> dict:
    """Fetch leaderboard for a single hardware preset."""
    params = [f"hwClass={preset['hwClass']}"]
    if preset["hardwareName"]:
        name = preset["hardwareName"].replace(" ", "+")
        params.append(f"hardwareName={name}")
    if preset["memTier"]:
        params.append(f"memTier={preset['memTier']}")
    params.append(f"limit={limit}")

    url = f"{BASE_URL}/leaderboard?{'&'.join(params)}"

    headers = {"User-Agent": "llmfit-benchmark-scraper/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code}: {e.reason}")
        return {"rows": [], "total": 0}
    except urllib.error.URLError as e:
        print(f"    Network error: {e.reason}")
        return {"rows": [], "total": 0}
    except Exception as e:
        print(f"    Error: {e}")
        return {"rows": [], "total": 0}


def main():
    parser = argparse.ArgumentParser(
        description="Scrape localmaxxing.com benchmark data for offline cache"
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max results per hardware preset (default: 100)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="localmaxxing.com API key (or set LOCALMAXXING_API_KEY env var)"
    )
    parser.add_argument(
        "--presets", type=str, default=None,
        help="Comma-separated list of preset labels to scrape (default: all)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("LOCALMAXXING_API_KEY")

    # Filter presets if specified
    presets = HARDWARE_PRESETS
    if args.presets:
        filter_names = {s.strip().lower() for s in args.presets.split(",")}
        presets = [p for p in presets if p["label"].lower() in filter_names]
        if not presets:
            print(f"No matching presets found. Available:")
            for p in HARDWARE_PRESETS:
                print(f"  {p['label']}")
            sys.exit(1)

    print(f"Scraping {len(presets)} hardware presets from localmaxxing.com...")
    if api_key:
        print(f"  Using API key: {api_key[:8]}...")
    print()

    cache = {
        "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "presets": {},
    }

    total_results = 0
    for i, preset in enumerate(presets):
        label = preset["label"]
        print(f"  [{i+1}/{len(presets)}] {label}...", end=" ", flush=True)

        data = fetch_leaderboard(preset, api_key, args.limit)
        count = len(data.get("rows", []))
        total = data.get("total", count)

        cache["presets"][label] = {
            "rows": data.get("rows", []),
            "total": total,
        }

        total_results += count
        print(f"{count} results (total: {total})")

        # Be polite to the API
        if i < len(presets) - 1:
            time.sleep(0.5)

    # Write to both locations
    output_paths = ["data/benchmark_cache.json", "llmfit-core/data/benchmark_cache.json"]
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache, f, indent=2)

    file_size = os.path.getsize(output_paths[0])
    size_str = f"{file_size / 1024:.0f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024*1024):.1f} MB"

    print(f"\nWrote {total_results} total benchmark results ({size_str}) to:")
    for p in output_paths:
        print(f"  {p}")

    # Summary table
    print(f"\n{'Hardware':<30} {'Results':>8} {'Total':>8}")
    print("-" * 48)
    for label, data in cache["presets"].items():
        count = len(data["rows"])
        total = data["total"]
        if count > 0:
            print(f"{label:<30} {count:>8} {total:>8}")


if __name__ == "__main__":
    main()
