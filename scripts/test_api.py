#!/usr/bin/env python3
"""
Local API validation tests for llmfit serve.

Usage:
  # Test an already-running server
  python3 scripts/test_api.py --base-url http://127.0.0.1:8787

  # Spawn server automatically (from repo root)
  python3 scripts/test_api.py --spawn
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def _http_json(url: str, timeout: float = 10.0) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            body = resp.read().decode("utf-8")
            data = json.loads(body) if body else {}
            return code, data
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {"raw": body}
        return exc.code, data


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect_keys(obj: Dict[str, Any], keys: List[str], path: str) -> None:
    for key in keys:
        _assert(key in obj, f"missing key '{key}' in {path}")


def test_health(base_url: str) -> None:
    code, data = _http_json(f"{base_url}/health")
    _assert(code == 200, f"/health expected 200, got {code}")
    _expect_keys(data, ["status", "node"], "/health")
    _assert(data["status"] == "ok", "health status must be 'ok'")
    _assert(isinstance(data["node"], dict), "health node must be object")
    _expect_keys(data["node"], ["name", "os"], "/health.node")


def test_system(base_url: str) -> None:
    code, data = _http_json(f"{base_url}/api/v1/system")
    _assert(code == 200, f"/api/v1/system expected 200, got {code}")
    _expect_keys(data, ["node", "system"], "/api/v1/system")
    _expect_keys(
        data["system"],
        ["total_ram_gb", "available_ram_gb", "cpu_cores", "cpu_name", "has_gpu", "backend", "gpus"],
        "/api/v1/system.system",
    )


def test_models_envelope_and_limit(base_url: str) -> None:
    code, data = _http_json(f"{base_url}/api/v1/models?limit=3&sort=score")
    _assert(code == 200, f"/api/v1/models expected 200, got {code}")
    _expect_keys(data, ["node", "system", "total_models", "returned_models", "filters", "models"], "/api/v1/models")
    _assert(isinstance(data["models"], list), "models must be a list")
    _assert(data["returned_models"] <= 3, "returned_models must respect limit")
    _assert(len(data["models"]) == data["returned_models"], "returned_models must equal models length")


def test_top_endpoint_excludes_too_tight(base_url: str) -> None:
    code, data = _http_json(f"{base_url}/api/v1/models/top?limit=10&min_fit=marginal")
    _assert(code == 200, f"/api/v1/models/top expected 200, got {code}")
    models = data.get("models", [])
    for row in models:
        _assert(row.get("fit_level") != "too_tight", "/models/top should not include too_tight fits")


def test_filters_runtime_and_use_case(base_url: str) -> None:
    code, data = _http_json(f"{base_url}/api/v1/models?limit=10&runtime=any&use_case=general")
    _assert(code == 200, f"runtime/use_case filter query expected 200, got {code}")
    models = data.get("models", [])
    for row in models:
        category = str(row.get("category", "")).lower()
        _assert(category == "general", "use_case=general should only return General category")


def test_models_shape(base_url: str) -> None:
    code, data = _http_json(f"{base_url}/api/v1/models?limit=5")
    _assert(code == 200, f"/api/v1/models shape query expected 200, got {code}")
    models = data.get("models", [])
    if not models:
        return

    sample = models[0]
    _expect_keys(
        sample,
        [
            "name",
            "provider",
            "fit_level",
            "run_mode",
            "score",
            "estimated_tps",
            "runtime",
            "best_quant",
            "memory_required_gb",
            "memory_available_gb",
            "utilization_pct",
            "score_components",
        ],
        "/api/v1/models.models[0]",
    )
    _expect_keys(sample["score_components"], ["quality", "speed", "fit", "context"], "/score_components")


def test_name_lookup(base_url: str) -> None:
    code, data = _http_json(f"{base_url}/api/v1/models?limit=1")
    _assert(code == 200, f"seed query expected 200, got {code}")
    models = data.get("models", [])
    if not models:
        return

    raw_name = str(models[0].get("name", "")).strip()
    _assert(raw_name, "expected at least one model name")

    token = raw_name.split("/")[-1].split("-")[0] or raw_name[:8]
    path_name = urllib.parse.quote(token, safe="")

    code2, data2 = _http_json(f"{base_url}/api/v1/models/{path_name}?limit=10")
    _assert(code2 == 200, f"/api/v1/models/{{name}} expected 200, got {code2}")
    _expect_keys(data2, ["models"], "/api/v1/models/{name}")
    result_models = data2.get("models", [])

    if result_models:
        lower_token = token.lower()
        matched = any(lower_token in str(row.get("name", "")).lower() for row in result_models)
        _assert(matched, "name lookup should return at least one model matching token")


def test_invalid_filter_returns_400(base_url: str) -> None:
    code, data = _http_json(f"{base_url}/api/v1/models?min_fit=nope")
    _assert(code == 400, f"invalid min_fit expected 400, got {code}")
    _expect_keys(data, ["error"], "error response")


def test_sort_score_desc(base_url: str) -> None:
    code, data = _http_json(f"{base_url}/api/v1/models?limit=25&sort=score")
    _assert(code == 200, f"sort=score query expected 200, got {code}")

    scores: List[float] = []
    for row in data.get("models", []):
        fit_level = row.get("fit_level")
        if fit_level == "too_tight":
            continue
        score = row.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))

    for i in range(1, len(scores)):
        _assert(scores[i - 1] >= scores[i] - 1e-9, "scores should be non-increasing for sort=score")


def wait_for_health(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            code, data = _http_json(f"{base_url}/health", timeout=2.0)
            if code == 200 and data.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy at {base_url} within {timeout_s}s")


def spawn_server(base_url: str, project_root: str) -> subprocess.Popen:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8787

    cmd = [
        "cargo",
        "run",
        "-p",
        "llmfit",
        "--",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def run_all_tests(base_url: str) -> None:
    tests = [
        ("health", test_health),
        ("system", test_system),
        ("models envelope+limit", test_models_envelope_and_limit),
        ("top excludes too_tight", test_top_endpoint_excludes_too_tight),
        ("filters runtime/use_case", test_filters_runtime_and_use_case),
        ("model row shape", test_models_shape),
        ("name lookup", test_name_lookup),
        ("invalid filter 400", test_invalid_filter_returns_400),
        ("sort score desc", test_sort_score_desc),
    ]

    for name, fn in tests:
        fn(base_url)
        print(f"✓ {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run llmfit REST API validation tests")
    parser.add_argument("--base-url", default="http://127.0.0.1:8787", help="API base URL")
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn llmfit serve automatically (requires cargo in PATH)",
    )
    parser.add_argument(
        "--project-root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Project root used when --spawn is set",
    )
    args = parser.parse_args()

    proc: Optional[subprocess.Popen] = None

    try:
        if args.spawn:
            print(f"Spawning server at {args.base_url} ...")
            proc = spawn_server(args.base_url, args.project_root)
            wait_for_health(args.base_url, timeout_s=45.0)

        print(f"Running API tests against {args.base_url}")
        run_all_tests(args.base_url)
        print("\nAll API tests passed.")
        return 0

    except Exception as exc:
        print(f"\nAPI tests failed: {exc}", file=sys.stderr)
        if proc and proc.stdout:
            try:
                output = proc.stdout.read(4000)
                if output:
                    print("\nServer output:", file=sys.stderr)
                    print(output, file=sys.stderr)
            except Exception:
                pass
        return 1

    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
