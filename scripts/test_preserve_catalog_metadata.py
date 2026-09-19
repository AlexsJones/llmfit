#!/usr/bin/env python3
"""Guard: weekly scrape must not silently wipe architecture metadata."""

import contextlib
import email.message
import io
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scrape_hf_models as shm  # noqa: E402
from scrape_hf_models import (  # noqa: E402
    ARCH_METADATA_DROP_LIMIT,
    RATE_LIMIT_MAX_RETRIES,
    RATE_LIMIT_STATS,
    detect_moe,
    estimate_params_from_arch,
    extract_arch_metadata,
    infer_context_length,
    preserve_existing_metadata,
    revalidation_priority,
    select_retained_for_revalidation,
    rate_limit_summary,
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


# --- HF rate limiting (#1039) ------------------------------------------------
# A fake clock and a scripted urlopen stand in for HuggingFace, so these run
# without network and without real sleeps.


class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, ratelimit: str | None):
        super().__init__(body)
        self.headers = email.message.Message()
        if ratelimit:
            self.headers["ratelimit"] = ratelimit


def _http_error(url: str, code: int, **headers) -> urllib.error.HTTPError:
    msg = email.message.Message()
    for key, value in headers.items():
        msg[key.replace("_", "-")] = value
    return urllib.error.HTTPError(url, code, "scripted", msg, io.BytesIO(b""))


class _FakeHF:
    """Replays a list of responses/errors and records requests and sleeps."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[str] = []
        self.sleeps: list[float] = []
        self.clock = 1_000.0

    def urlopen(self, req, timeout):
        self.requests.append(req.full_url)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds

    def now(self):
        return self.clock

    def __enter__(self):
        self._saved = (shm._urlopen, shm._sleep, shm._now)
        shm._urlopen, shm._sleep, shm._now = self.urlopen, self.sleep, self.now
        shm._reset_rate_limit_state()
        # Keep the pause/429 messages out of the test output, and available
        # to the tests that assert the scraper is never silent about them.
        self.stderr = io.StringIO()
        self._redirect = contextlib.redirect_stderr(self.stderr)
        self._redirect.__enter__()
        return self

    def __exit__(self, *exc):
        self._redirect.__exit__(*exc)
        shm._urlopen, shm._sleep, shm._now = self._saved
        shm._reset_rate_limit_state()


INFO_URL = "https://huggingface.co/api/models/org/model"
CONFIG_URL = "https://huggingface.co/org/model/resolve/main/config.json"


def test_parses_hf_ratelimit_header():
    assert shm._parse_ratelimit_header('"api";r=499;t=106') == ("api", 499, 106.0)
    assert shm._parse_ratelimit_header('"resolvers";r=0;t=12') == ("resolvers", 0, 12.0)
    assert shm._parse_ratelimit_header(None) == (None, None, None)
    assert shm._parse_ratelimit_header("garbage") == (None, None, None)


def test_429_sleeps_to_the_reset_then_retries():
    with _FakeHF(
        [
            _http_error(INFO_URL, 429, ratelimit='"api";r=0;t=42'),
            _FakeResponse(b'{"downloads": 7}', '"api";r=499;t=300'),
        ]
    ) as hf:
        info = shm.fetch_model_info("org/model")
    assert info == {"downloads": 7}
    assert len(hf.requests) == 2
    assert hf.sleeps == [42.0], hf.sleeps


def test_429_is_counted_per_caller_kind():
    with _FakeHF(
        [
            _http_error(CONFIG_URL, 429, ratelimit='"resolvers";r=0;t=5'),
            _FakeResponse(
                b'{"max_position_embeddings": 8192}', '"resolvers";r=2999;t=295'
            ),
        ]
    ):
        config = shm.fetch_config_json("org/model")
        assert config == {"max_position_embeddings": 8192}
        assert RATE_LIMIT_STATS["http_429"] == 1
        assert RATE_LIMIT_STATS["http_429_by_kind"] == {"config_json": 1}
        assert RATE_LIMIT_STATS["pauses"] == 1
        assert RATE_LIMIT_STATS["gave_up"] == 0


def test_retry_after_wins_over_ratelimit_reset():
    with _FakeHF(
        [
            _http_error(INFO_URL, 429, retry_after="90", ratelimit='"api";r=0;t=42'),
            _FakeResponse(b"{}", None),
        ]
    ) as hf:
        shm.fetch_model_info("org/model")
    assert hf.sleeps == [90.0], hf.sleeps


def test_spent_window_on_a_200_holds_the_next_request():
    with _FakeHF(
        [
            _FakeResponse(b"{}", '"api";r=0;t=30'),
            _FakeResponse(b"{}", '"api";r=499;t=300'),
        ]
    ) as hf:
        shm.fetch_model_info("org/a")
        assert hf.sleeps == []
        shm.fetch_model_info("org/b")
        assert hf.sleeps == [30.0], hf.sleeps
        assert RATE_LIMIT_STATS["http_429"] == 0
        assert RATE_LIMIT_STATS["pauses"] == 1


def test_buckets_pause_independently():
    with _FakeHF(
        [
            # spends the api window
            _FakeResponse(b"{}", '"api";r=0;t=120'),
            # config.json must not wait
            _FakeResponse(b"{}", '"resolvers";r=2999;t=120'),
        ]
    ) as hf:
        shm.fetch_model_info("org/a")
        shm.fetch_config_json("org/a")
        assert hf.sleeps == [], hf.sleeps


def test_config_json_429_after_retries_is_none_but_never_silent():
    burst = [
        _http_error(CONFIG_URL, 429, ratelimit='"resolvers";r=0;t=3')
        for _ in range(RATE_LIMIT_MAX_RETRIES + 1)
    ]
    with _FakeHF(burst) as hf:
        assert shm.fetch_config_json("org/model") is None
        assert len(hf.requests) == RATE_LIMIT_MAX_RETRIES + 1
        assert RATE_LIMIT_STATS["http_429"] == RATE_LIMIT_MAX_RETRIES + 1
        assert RATE_LIMIT_STATS["gave_up"] == 1
        summary = rate_limit_summary()
        assert "config_json" in summary and "1 request(s) still 429" in summary, summary
        assert "still rate limited after" in hf.stderr.getvalue()


def test_non_429_errors_pass_through_untouched():
    with _FakeHF([_http_error(CONFIG_URL, 404)]) as hf:
        assert shm.fetch_config_json("org/model") is None
        assert hf.sleeps == []
        assert RATE_LIMIT_STATS["http_429"] == 0
        assert rate_limit_summary().startswith("HF rate limiting this run: none")
        assert hf.stderr.getvalue() == ""


def test_bucket_named_by_the_response_drives_the_retry_wait():
    # The URL looks like "api" but HF meters it elsewhere: the retry must
    # still wait on the bucket the 429 named, not spin on the guessed one.
    with _FakeHF(
        [
            _http_error(INFO_URL, 429, ratelimit='"models";r=0;t=7'),
            _FakeResponse(b"{}", None),
        ]
    ) as hf:
        shm.fetch_model_info("org/model")
    assert hf.sleeps == [7.0], hf.sleeps


def test_wait_is_capped_against_bogus_reset_values():
    with _FakeHF(
        [
            _http_error(INFO_URL, 429, retry_after="999999"),
            _FakeResponse(b"{}", None),
        ]
    ) as hf:
        shm.fetch_model_info("org/model")
    assert hf.sleeps == [shm.RATE_LIMIT_MAX_WAIT_SECONDS], hf.sleeps


def test_in_flight_threads_report_one_pause_not_eight():
    with _FakeHF([]) as hf:
        for _ in range(8):
            shm._schedule_rate_limit_pause("api", 197.0, "HTTP 429 on model_info")
            hf.clock += 0.01  # threads land a few ms apart
        assert RATE_LIMIT_STATS["pauses"] == 1
        assert abs(RATE_LIMIT_STATS["pause_seconds"] - 197.0) < 1e-6


PROBE_URL = "https://huggingface.co/api/models/unsloth/model-GGUF"


class _OneCandidateNoDisk:
    """enrich_gguf_sources with one GGUF candidate and the cache kept in memory."""

    def __enter__(self):
        self._saved = (
            shm._load_gguf_cache,
            shm._save_gguf_cache,
            shm._model_gguf_repo_candidates,
        )
        self.cache_writes: list[dict] = []
        shm._load_gguf_cache = dict
        shm._save_gguf_cache = lambda cache: self.cache_writes.append(dict(cache))
        shm._model_gguf_repo_candidates = lambda repo_id: [
            ("unsloth", "unsloth/model-GGUF")
        ]
        self._quiet = contextlib.redirect_stdout(io.StringIO())
        self._quiet.__enter__()
        return self

    def __exit__(self, *exc):
        self._quiet.__exit__(*exc)
        (
            shm._load_gguf_cache,
            shm._save_gguf_cache,
            shm._model_gguf_repo_candidates,
        ) = self._saved


def test_exhausted_gguf_probe_is_not_cached_as_a_miss():
    burst = [
        _http_error(PROBE_URL, 429, ratelimit='"api";r=0;t=3')
        for _ in range(RATE_LIMIT_MAX_RETRIES + 1)
    ]
    model = {"name": "org/model", "format": "gguf", "parameters_raw": 1}
    with _FakeHF(burst), _OneCandidateNoDisk() as gguf:
        assert shm.enrich_gguf_sources([model], threads=1) == 0
        assert RATE_LIMIT_STATS["gave_up"] == 1
    assert "gguf_sources" not in model
    assert gguf.cache_writes == [{}], gguf.cache_writes


def test_missing_gguf_repo_is_still_cached_as_a_miss():
    model = {"name": "org/model", "format": "gguf", "parameters_raw": 1}
    with _FakeHF([_http_error(PROBE_URL, 404)]), _OneCandidateNoDisk() as gguf:
        assert shm.enrich_gguf_sources([model], threads=1) == 0
    assert "gguf_sources" not in model
    assert list(gguf.cache_writes[0]) == ["org/model"]
    assert gguf.cache_writes[0]["org/model"]["sources"] == []


def test_yarn_context_is_not_scaled_twice():
    # DeepSeek-V4: max_position_embeddings is already original * factor.
    cfg = {
        "max_position_embeddings": 1048576,
        "rope_scaling": {"type": "yarn", "factor": 16,
                         "original_max_position_embeddings": 65536},
    }
    assert infer_context_length(cfg) == 1048576
    # Kimi-K2.6 nests the same shape under text_config.
    nested = {"text_config": {
        "max_position_embeddings": 262144,
        "rope_scaling": {"type": "yarn", "factor": 64.0,
                         "original_max_position_embeddings": 4096},
    }}
    assert infer_context_length(nested) == 262144


def test_rope_factor_still_scales_a_pre_scaling_window():
    # No original_max_position_embeddings: the value is the unscaled window.
    cfg = {"max_position_embeddings": 4096,
           "rope_scaling": {"type": "linear", "factor": 4.0}}
    assert infer_context_length(cfg) == 16384
    # original * factor larger than the stated window wins.
    cfg = {"max_position_embeddings": 8192,
           "rope_scaling": {"type": "yarn", "factor": 4,
                            "original_max_position_embeddings": 8192}}
    assert infer_context_length(cfg) == 32768


def test_detects_moe_under_family_specific_key_names():
    kimi_k3 = {"text_config": {"num_experts": 896, "num_experts_per_token": 16}}
    moe = detect_moe("moonshotai/Kimi-K3", kimi_k3, "kimi_k3", 2_779_931_837_184)
    assert moe["is_moe"] and moe["num_experts"] == 896 and moe["active_experts"] == 16
    assert moe["active_parameters"] == 104_000_000_000

    step = {"text_config": {"moe_num_experts": 288, "moe_top_k": 8}}
    moe = detect_moe("stepfun-ai/Step-3.7-Flash", step, "step3p7", 201_365_316_160)
    assert moe["is_moe"] and moe["num_experts"] == 288 and moe["active_experts"] == 8
    assert moe["active_parameters"] == 11_000_000_000


def test_arch_metadata_drops_unset_sentinels():
    # inclusionAI/LLaDA-UI: -1 failed the u32 parse of the whole catalog.
    arch = extract_arch_metadata({
        "num_hidden_layers": 28, "hidden_size": 2048, "num_attention_heads": 16,
        "shared_expert_intermediate_size": -1,
    })
    assert arch["shared_expert_intermediate_size"] is None
    assert arch["num_hidden_layers"] == 28
    # 0 is a real MoE size (Qwen3-Coder has no shared expert) but never a
    # real vocab; data/schema.json draws the same line.
    arch = extract_arch_metadata({"shared_expert_intermediate_size": 0, "vocab_size": 0})
    assert arch["shared_expert_intermediate_size"] == 0
    assert arch["vocab_size"] is None
    # GLM-5.3-Flash: head_dim=0 falls back to hidden_size / heads.
    arch = extract_arch_metadata({"text_config": {
        "hidden_size": 4096, "num_attention_heads": 64, "head_dim": 0,
    }})
    assert arch["head_dim"] == 64
    # hidden_size < heads would derive 0 again; leave it null instead.
    arch = extract_arch_metadata({"hidden_size": 8, "num_attention_heads": 16, "head_dim": 0})
    assert arch["head_dim"] is None


def test_param_estimate_sees_the_same_experts_as_detection():
    base = {"hidden_size": 4096, "num_hidden_layers": 45, "vocab_size": 128896,
            "num_attention_heads": 64, "moe_intermediate_size": 1280,
            "intermediate_size": 11264}
    dense = estimate_params_from_arch(base)
    step = estimate_params_from_arch({**base, "moe_num_experts": 288, "moe_top_k": 8})
    routed = estimate_params_from_arch({**base, "n_routed_experts": 288})
    assert step == routed
    assert step > 5 * dense


def test_revalidation_ranks_uncorrectable_packed_counts_first():
    # #1045: HF reports the packed int32 element count for these, and with no
    # architecture metadata nothing downstream can correct it.
    packed = {"name": "TelperionAI/Qwen3.8-27B-INT4-AWQ-GPTQ", "format": "awq",
              "hidden_size": None, "release_date": "2026-08-20"}
    assert revalidation_priority(packed) == 0
    # The format field is "gguf" for names the format detector does not know.
    w4a16 = {"name": "RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16",
             "format": "gguf", "hidden_size": None, "release_date": "2025-09-01"}
    assert revalidation_priority(w4a16) == 0
    # Same kind of repo, but config.json was read: the correction could fire.
    assert revalidation_priority({**packed, "hidden_size": 5120}) is None
    # "8B" is a size, not a bit width.
    plain = {"name": "meta-llama/Llama-3.1-8B-Instruct", "format": "gguf",
             "hidden_size": None, "release_date": "2024-07-18"}
    assert revalidation_priority(plain) is None


def test_revalidation_flags_suspect_context_and_missing_date():
    base = {"name": "org/model", "format": "gguf", "hidden_size": 4096,
            "release_date": "2026-01-01", "context_length": 131072}
    assert revalidation_priority(base) is None
    assert revalidation_priority({**base, "context_length": 16777216}) == 1
    assert revalidation_priority({**base, "release_date": None}) == 2


def test_revalidation_selection_respects_budget_and_skips_fresh_entries():
    existing = [
        {"name": "a/dateless-popular", "release_date": None, "hf_downloads": 9_000_000},
        {"name": "b/model-AWQ", "format": "awq", "hidden_size": None,
         "release_date": "2026-01-01", "hf_downloads": 10},
        {"name": "c/model-GPTQ", "format": "gptq", "hidden_size": None,
         "release_date": "2026-01-01", "hf_downloads": 500},
        {"name": "d/rescraped-AWQ", "format": "awq", "hidden_size": None},
        {"name": "e/healthy", "hidden_size": 4096, "release_date": "2026-01-01"},
    ]
    fresh = {"d/rescraped-AWQ"}
    # Priority beats popularity; downloads order entries within a priority.
    assert select_retained_for_revalidation(existing, fresh, 2) == [
        "c/model-GPTQ", "b/model-AWQ"]
    assert select_retained_for_revalidation(existing, fresh, 10) == [
        "c/model-GPTQ", "b/model-AWQ", "a/dateless-popular"]
    assert select_retained_for_revalidation(existing, fresh, 0) == []


if __name__ == "__main__":
    tests = [
        test_preserves_architecture_when_config_fetch_misses,
        test_does_not_invent_heads_the_catalog_never_had,
        test_mass_drop_limit_would_have_caught_2026_08_28,
        test_parses_hf_ratelimit_header,
        test_429_sleeps_to_the_reset_then_retries,
        test_429_is_counted_per_caller_kind,
        test_retry_after_wins_over_ratelimit_reset,
        test_spent_window_on_a_200_holds_the_next_request,
        test_buckets_pause_independently,
        test_config_json_429_after_retries_is_none_but_never_silent,
        test_non_429_errors_pass_through_untouched,
        test_bucket_named_by_the_response_drives_the_retry_wait,
        test_wait_is_capped_against_bogus_reset_values,
        test_in_flight_threads_report_one_pause_not_eight,
        test_exhausted_gguf_probe_is_not_cached_as_a_miss,
        test_missing_gguf_repo_is_still_cached_as_a_miss,
        test_yarn_context_is_not_scaled_twice,
        test_rope_factor_still_scales_a_pre_scaling_window,
        test_detects_moe_under_family_specific_key_names,
        test_arch_metadata_drops_unset_sentinels,
        test_param_estimate_sees_the_same_experts_as_detection,
        test_revalidation_ranks_uncorrectable_packed_counts_first,
        test_revalidation_flags_suspect_context_and_missing_date,
        test_revalidation_selection_respects_budget_and_skips_fresh_entries,
    ]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
