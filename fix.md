# llmfit — Bug Tracker & Fix List

## Critical Bugs

### BUG-001: LM Studio download polling has no timeout
**File:** `llmfit-core/src/providers.rs`  
**Issue:** `GET /api/v1/models/download-status` is polled in a loop with no timeout or cancellation. If LM Studio crashes mid-download, the TUI hangs indefinitely.  
**Fix:** Add a max-retry count (e.g. 300 × 1s = 5 min) and surface a timeout error via `PullEvent::Error`.

---

### BUG-002: `available_ram_gb` can be 0 on macOS Tahoe
**File:** `llmfit-core/src/hardware.rs`  
**Issue:** `sysinfo` returns 0 for available memory on newer macOS. The fallback `available_ram_fallback()` exists but may still return 0 if all fallback paths fail, causing all models to score as TooTight.  
**Fix:** If fallback also returns 0, default to `total_ram_gb * 0.6` as a safe estimate and log a warning.

---

### BUG-003: Duplicate `hf_models.json` can diverge
**File:** `data/hf_models.json` vs `llmfit-core/data/hf_models.json`  
**Issue:** Two copies of the model database exist. `update_models.sh` only updates one. If they diverge, the binary embeds stale data.  
**Fix:** Remove `data/hf_models.json` from repo root; update all references to point to `llmfit-core/data/hf_models.json` only.

---

### BUG-004: `OLLAMA_HOST` with port-only value (`:11434`) silently ignored
**File:** `llmfit-core/src/providers.rs` — `normalize_ollama_host()`  
**Issue:** `OLLAMA_HOST=":11434"` doesn't start with `http://` and has no `://`, so it gets formatted as `http://:11434` — an invalid URL that causes all Ollama API calls to fail silently.  
**Fix:** Detect port-only values and prepend `http://localhost`.

```rust
// Before
Some(format!("http://{host}"))

// After
if host.starts_with(':') {
    Some(format!("http://localhost{host}"))
} else {
    Some(format!("http://{host}"))
}
```

---

### BUG-005: MoE detection false positives
**File:** `llmfit-core/src/models.rs` / `fit.rs`  
**Issue:** MoE detection relies on model name substring matching (e.g. "moe", "mixtral"). Dense models with "moe" in their name (e.g. a hypothetical "moebius-7b") would get incorrect expert-offload memory calculations.  
**Fix:** Prefer `num_local_experts` field from scraped JSON; only fall back to name heuristic when field is absent.

---

## Medium Bugs

### BUG-006: Score can exceed 100 for small models on high-end hardware
**File:** `llmfit-core/src/fit.rs`  
**Issue:** The Fit dimension score formula can produce values > 100 when memory utilization is very low (e.g. a 1B model on a 80GB A100). The composite score is then clamped but individual dimension scores are not.  
**Fix:** Clamp each dimension score to `[0.0, 100.0]` before combining.

---

### BUG-007: `--max-context` not respected in Plan mode
**File:** `llmfit-tui/src/main.rs`, `llmfit-core/src/plan.rs`  
**Issue:** `--max-context` CLI flag is passed to fit scoring but not forwarded to `estimate_model_plan()`. Plan mode always uses the model's full advertised context for KV-cache estimation.  
**Fix:** Thread `context_limit: Option<u32>` through `PlanRequest`.

---

### BUG-008: Web dashboard starts even with `--cli` flag
**File:** `llmfit-tui/src/main.rs`  
**Issue:** The background dashboard thread is spawned before the CLI/TUI branch is selected. Running `llmfit --cli` still binds port 8787.  
**Fix:** Move dashboard spawn inside the TUI branch, or check for `--no-dashboard` / CLI mode before spawning.

---

### BUG-009: `nvidia-smi` multi-GPU VRAM aggregation ignores heterogeneous setups
**File:** `llmfit-core/src/hardware.rs`  
**Issue:** When two different GPU models are present (e.g. RTX 3090 + RTX 4090), VRAM is aggregated but the bandwidth lookup uses only the first GPU's name. Speed estimates will be wrong.  
**Fix:** Use the minimum bandwidth across all GPUs for conservative estimation, or report per-GPU fits separately.

---

### BUG-010: TUI search is case-sensitive for provider names
**File:** `llmfit-tui/src/tui_app.rs` — `apply_filters()`  
**Issue:** Search lowercases the query but provider names in the database use mixed case (e.g. "Meta", "Google"). Searching "meta" works but "Meta" may not match depending on normalization path.  
**Fix:** Normalize both query and field to lowercase consistently before comparison.

---

### BUG-011: `llmfit serve` and TUI dashboard use different code paths
**File:** `llmfit-tui/src/serve_api.rs`, `llmfit-tui/src/main.rs`  
**Issue:** `llmfit serve` (standalone REST API) and the auto-started TUI dashboard both use `serve_api.rs` but are initialized differently. Hardware overrides (`--memory`, `--ram`) are applied in TUI mode but may not be forwarded in `serve` subcommand.  
**Fix:** Ensure `serve` subcommand reads and applies the same hardware override flags as TUI/CLI modes.

---

## Minor Bugs / Polish

### BUG-012: `g`/`G` jump keys don't reset scroll offset in detail view
**File:** `llmfit-tui/src/tui_events.rs`  
**Issue:** Pressing `G` (jump to bottom) in the model list doesn't reset the detail pane scroll position. If a long detail was open, the scroll offset persists on the new selection.  
**Fix:** Reset `detail_scroll` to 0 whenever selection changes.

---

### BUG-013: Theme file written on every `t` keypress
**File:** `llmfit-tui/src/theme.rs`  
**Issue:** Theme is saved to `~/.config/llmfit/theme` on every cycle keypress, not just on quit. On slow filesystems this causes noticeable lag.  
**Fix:** Save theme only on quit, or debounce writes.

---

### BUG-014: `llmfit list` output truncates long model names
**File:** `llmfit-tui/src/display.rs`  
**Issue:** The `tabled` crate truncates columns based on terminal width, but model names like `meta-llama/Llama-3.3-70B-Instruct` get cut without indication.  
**Fix:** Add `…` suffix when truncating, or use `--wide` flag to disable truncation.

---

### BUG-015: Docker Model Runner provider not detected on Linux
**File:** `llmfit-core/src/providers.rs`  
**Issue:** Docker Model Runner is a Docker Desktop feature. On Linux, Docker Desktop is less common. The provider check hits `http://localhost:12434` and times out (default ureq timeout), adding ~5s to startup.  
**Fix:** Check for Docker Desktop process or socket before attempting HTTP connection; skip on Linux if not found.

---

## Scraper Bugs

### BUG-016: `scrape_hf_models.py` GGUF cache TTL not enforced on corrupt cache
**File:** `scripts/scrape_hf_models.py`  
**Issue:** If `data/gguf_sources_cache.json` is corrupted (partial write, disk full), the scraper crashes with an unhandled JSON parse error instead of regenerating the cache.  
**Fix:** Wrap cache load in try/except and delete + regenerate on parse failure.

---

### BUG-017: Scraper uses hardcoded fallbacks for gated models that may be outdated
**File:** `scripts/scrape_hf_models.py` — `FALLBACKS` dict  
**Issue:** Gated model fallbacks (e.g. Llama parameter counts) are hardcoded and never auto-updated. If Meta releases a new Llama variant, it won't appear until a developer manually updates the fallback.  
**Fix:** Add a `--check-fallbacks` flag that validates fallback entries against public HF metadata where possible.
