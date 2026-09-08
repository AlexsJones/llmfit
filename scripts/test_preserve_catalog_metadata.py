#!/usr/bin/env python3
"""Guard: weekly scrape must not silently wipe architecture metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_hf_models import (  # noqa: E402
    ARCH_METADATA_DROP_LIMIT,
    preserve_existing_metadata,
)


def test_preserves_architecture_when_config_fetch_misses():
    old = {
        "license": "mit",
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "num_hidden_layers": 28,
        "context_length": 4194304,
        "hf_downloads": 89,
    }
    fresh = {
        "license": None,
        "num_attention_heads": None,
        "num_key_value_heads": None,
        "num_hidden_layers": None,
        "context_length": 4096,
        "hf_downloads": 0,
    }
    restored = preserve_existing_metadata(old, fresh)
    assert fresh["num_attention_heads"] == 24, restored
    assert fresh["num_key_value_heads"] == 8, restored
    assert fresh["num_hidden_layers"] == 28, restored
    assert fresh["context_length"] == 4194304, restored
    assert fresh["license"] == "mit", restored
    assert "num_attention_heads" in restored


def test_does_not_invent_heads_the_catalog_never_had():
    old = {"num_attention_heads": None, "context_length": 2048}
    fresh = {"num_attention_heads": None, "context_length": 4096}
    preserve_existing_metadata(old, fresh)
    assert fresh["num_attention_heads"] is None
    assert fresh["context_length"] == 4096


def test_mass_drop_limit_would_have_caught_2026_08_28():
    assert ARCH_METADATA_DROP_LIMIT < 1764


if __name__ == "__main__":
    tests = [
        test_preserves_architecture_when_config_fetch_misses,
        test_does_not_invent_heads_the_catalog_never_had,
        test_mass_drop_limit_would_have_caught_2026_08_28,
    ]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
