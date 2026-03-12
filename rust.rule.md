# Rust Coding Rules for llmfit

Rules extracted from codebase analysis. Follow these when contributing.

---

## Safety & Error Handling

### NO unsafe code
Never use `unsafe` blocks. The entire codebase must be safe Rust.

### NO .unwrap() on user-facing paths
```rust
// BAD - will panic
let fit = self.selected_fit().unwrap();

// GOOD - graceful handling
let Some(fit) = self.selected_fit() else {
    self.pull_status = Some("No model selected".to_string());
    return;
};
```

Use `.expect()` only for internal invariants with descriptive messages:
```rust
// Acceptable - this should never fail if code is correct
let config = parse_config().expect("Default config must be valid");
```

### Always provide user feedback on errors
Use existing status fields to communicate errors to users:
```rust
// Use pull_status for TUI feedback
self.pull_status = Some(format!("Error: {}", e));
```

---

## TUI Architecture

### Stateless rendering
`tui_ui::draw()` must NOT mutate `App`. Pass `&App` only.
```rust
// BAD
fn draw(frame: &mut Frame, app: &mut App) { ... }

// GOOD
fn draw(frame: &mut Frame, app: &App) { ... }
```

Exception: `TableState` widget requires `&mut App` - use it only for widget requirements, never to change application state.

### Single mutation point
All `App` state changes in TUI happen in `tui_events.rs`. This is the sole place that mutates `App` in the TUI loop.

### Input modes are mutually exclusive
```rust
match app.input_mode {
    InputMode::Normal => handle_normal_mode(app, key),
    InputMode::Search => handle_search_mode(app, key),
    // Each mode handles its own keys
}
```

---

## Keybindings

### Add new keybinding pattern
1. Add handler in `tui_events.rs` (in the appropriate mode function)
2. Add method to `App` impl in `tui_app.rs`
3. Update status bar help text in `tui_ui.rs`
4. Keep keybindings mnemonic where possible (y=copy, m=mark, c=compare)

### Keybinding format
```rust
// Use comments to group related keybindings
// Navigation
KeyCode::Up | KeyCode::Char('k') => app.move_up(),
KeyCode::Down | KeyCode::Char('j') => app.move_down(),

// Actions
KeyCode::Char('y') => app.copy_selected_model_name(),
```

---

## Dependencies

### Preferred crates
- System detection: `sysinfo` (never raw platform calls)
- TUI: `ratatui` + `crossterm` (never termion or ncurses)
- CLI parsing: `clap` with derive feature (never manual arg parsing)
- Clipboard: `arboard` (cross-platform)

### Dependency policy
- Prefer well-maintained crates with minimal transitive dependencies
- No pip dependencies for Python scraper (stdlib only)

---

## Fit Analysis

### Fit level ordering (do not change)
```
Perfect > Good > Marginal > TooTight
```

### Memory hierarchy
```
GPU inference > CPU offload > CPU only
```

### VRAM vs RAM
- `min_vram_gb`: VRAM for GPU inference (weights on GPU)
- `min_ram_gb`: System RAM for CPU inference (same weights, different pool)
- Both represent the same workload on different hardware paths

### Apple Silicon (unified memory)
- VRAM = system RAM (same memory pool)
- Skip `CpuOffload` path (no separate pool to spill to)
- Check `specs.unified_memory` flag

---

## CLI vs TUI Independence

### Separate modules
- `display.rs`: CLI table output only
- `tui_*.rs`: TUI only

### CLI must work without TUI
The CLI path (`cargo run -- --cli`) must work without initializing any TUI state.

---

## Model Database

### Never manually edit hf_models.json
Regenerate it:
```sh
python3 scripts/scrape_hf_models.py && cargo build
```

### Adding a new model
1. Add HuggingFace repo ID to `TARGET_MODELS` in `scripts/scrape_hf_models.py`
2. If gated (requires HF auth), add fallback entry to `FALLBACK` dict
3. Run the scraper
4. Verify output
5. Build to verify compilation

---

## Code Style

### Match expressions for error handling
```rust
match something_that_can_fail() {
    Ok(result) => { /* use result */ },
    Err(e) => {
        self.pull_status = Some(format!("Failed: {}", e));
    }
}
```

### Use existing fields for status
Don't add new status fields if `pull_status` already serves the purpose:
```rust
// pull_status is reused for copy feedback, download status, errors, etc.
self.pull_status = Some(format!("Copied: {}", name));
```

### Method naming
- `toggle_*` - flip a boolean state
- `cycle_*` - advance to next option in a cycle
- `open_*/close_*` - enter/exit a mode or popup
- `copy_*` - copy something to clipboard
- `mark_*` - mark for later use (compare, etc.)

---

## Testing (when added)

### Unit tests location
- `fit.rs` logic tests (given known SystemSpecs and LlmModel, assert correct FitLevel)
- `models.rs` tests (verify JSON parsing, search matching)
- CLI integration tests via `assert_cmd` crate

### TUI testing
TUI is difficult to unit test. Keep rendering stateless and test state mutations in `tui_app.rs` directly.

---

## Platform Notes

### GPU detection
- NVIDIA: shells out to `nvidia-smi`
- AMD: shells out to `rocm-smi`
- Apple Silicon: uses `system_profiler SPDisplaysDataType`
- Best-effort, fails silently if unavailable

### Cross-platform
- `sysinfo` handles RAM/CPU cross-platform
- `crossterm` works on Linux, macOS, Windows
- No conditional compilation needed for RAM/CPU
