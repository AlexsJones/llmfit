#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scrape_hf_models as scraper  # noqa: E402


QWEN_38_TARGETS = (
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.8-27B-FP8",
    "Qwen/Qwen3.8-2.4T-A95B",
    "Qwen/Qwen3.8-2.4T-A95B-FP8",
)


def test_curated_guard_reports_missing_entries():
    models = [{"name": QWEN_38_TARGETS[0]}]

    assert scraper.missing_curated_models(models, QWEN_38_TARGETS) == list(
        QWEN_38_TARGETS[1:]
    )


def test_qwen_38_repositories_are_curated_refresh_targets():
    assert set(QWEN_38_TARGETS).issubset(scraper.TARGET_MODELS)


if __name__ == "__main__":
    for test in (
        test_curated_guard_reports_missing_entries,
        test_qwen_38_repositories_are_curated_refresh_targets,
    ):
        test()
    print("PASS")
