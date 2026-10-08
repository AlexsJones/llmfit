use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use ratatui::layout::{Position, Rect};
use std::time::{Duration, Instant};

use crate::tui_app::{App, InputMode};

/// Apply selection and terminal-size changes to the model table's scroll state.
/// Also used once at startup, before the first frame is drawn.
pub fn update_model_viewport(app: &mut App, terminal_area: ratatui::layout::Rect) {
    if app.show_bench
        || app.show_benchmarks
        || app.show_downloads
        || app.show_plan
        || app.show_multi_compare
        || app.show_compare
        || app.show_detail
    {
        return;
    }
    let table_area = crate::tui_ui::main_layout(terminal_area)[2];
    let viewport = crate::tui_ui::model_table_viewport(
        app.filtered_fits.len(),
        app.selected_row,
        app.table_state.offset(),
        usize::from(table_area.height.saturating_sub(3)),
        app.table_follow_selection,
    );
    app.table_state
        .select((!app.filtered_fits.is_empty()).then_some(app.selected_row));
    *app.table_state.offset_mut() = viewport.start;
}

/// Poll for and handle events. Returns true if an event was processed.
pub fn handle_events(app: &mut App) -> std::io::Result<bool> {
    let processed = handle_pending_events(app)?;
    // Refresh after input, resize events, and background-worker state changes
    let (width, height) = crossterm::terminal::size()?;
    update_model_viewport(app, ratatui::layout::Rect::new(0, 0, width, height));
    Ok(processed)
}

fn handle_pending_events(app: &mut App) -> std::io::Result<bool> {
    // Always tick the pull progress and worker messages (non-blocking)
    app.tick_pull();
    app.tick_bench();
    app.tick_bench_offer();
    app.tick_bench_fetch();

    if event::poll(Duration::from_millis(50))? {
        let key = match event::read()? {
            Event::Key(key) => key,
            Event::Mouse(mouse) => {
                let (width, height) = crossterm::terminal::size()?;
                return Ok(handle_mouse(app, mouse, Rect::new(0, 0, width, height)));
            }
            Event::Resize(_, _) => {
                app.last_model_click = None;
                return Ok(true);
            }
            _ => return Ok(false),
        };
        // Only handle Press events (ignore Release on some platforms)
        if key.kind != KeyEventKind::Press {
            return Ok(false);
        }
        handle_key(app, key);
        return Ok(true);
    }
    Ok(false)
}

fn handle_key(app: &mut App, key: KeyEvent) {
    app.last_model_click = None;
    match app.input_mode {
        InputMode::Normal => handle_normal_mode(app, key),
        InputMode::Visual => handle_visual_mode(app, key),
        InputMode::Select => handle_select_mode(app, key),
        InputMode::Search => handle_search_mode(app, key),
        InputMode::Plan => handle_plan_mode(app, key),
        InputMode::ProviderPopup => handle_provider_popup_mode(app, key),
        InputMode::UseCasePopup => handle_use_case_popup_mode(app, key),
        InputMode::CapabilityPopup => handle_capability_popup_mode(app, key),
        InputMode::DownloadProviderPopup => handle_download_provider_popup_mode(app, key),
        InputMode::QuantPopup => handle_quant_popup_mode(app, key),
        InputMode::RunModePopup => handle_run_mode_popup_mode(app, key),
        InputMode::ParamsBucketPopup => handle_params_bucket_popup_mode(app, key),
        InputMode::LicensePopup => handle_license_popup_mode(app, key),
        InputMode::RuntimePopup => handle_runtime_popup_mode(app, key),
        InputMode::HelpPopup => handle_help_popup_mode(app, key),
        InputMode::Simulation => handle_simulation_mode(app, key),
        InputMode::AdvancedConfig => handle_advanced_config_mode(app, key),
        InputMode::DownloadManager => handle_download_manager_mode(app, key),
        InputMode::FilterPopup => handle_filter_popup_mode(app, key),
        InputMode::Benchmarks => handle_benchmarks_mode(app, key),
        InputMode::BenchOffer => handle_bench_offer_mode(app, key),
    }
}

fn handle_mouse(app: &mut App, mouse: MouseEvent, terminal_area: Rect) -> bool {
    handle_mouse_at(app, mouse, terminal_area, Instant::now())
}

fn handle_mouse_at(app: &mut App, mouse: MouseEvent, terminal_area: Rect, now: Instant) -> bool {
    let previous_click = if mouse.kind == MouseEventKind::Down(MouseButton::Left) {
        app.last_model_click.take()
    } else {
        if !matches!(
            mouse.kind,
            MouseEventKind::Up(MouseButton::Left) | MouseEventKind::Moved
        ) {
            app.last_model_click = None;
        }
        None
    };
    if !mouse.modifiers.is_empty()
        || !terminal_area.contains(Position::new(mouse.column, mouse.row))
    {
        return false;
    }
    let position = Position::new(mouse.column, mouse.row);
    let layout = crate::tui_ui::main_layout(terminal_area);
    let click = matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left));
    // The visible footer belongs to the active mode, including subdialogs
    let keys_area = crate::tui_ui::status_key_area(app, layout[3]);
    if click && keys_area.contains(position) {
        let (text, _) = crate::tui_ui::status_keys_and_mode(app);
        if let Some(key) = key_hint_at(&text, usize::from(mouse.column - keys_area.x)) {
            handle_key(app, key);
            return true;
        }
        return false;
    }
    if app.bench_confirm_quit {
        return false;
    }
    if let Some((popup, count, cursor)) =
        crate::tui_ui::selection_popup(app, app.input_mode, terminal_area)
    {
        return handle_selection_popup_mouse(app, mouse, popup, count, cursor);
    }
    if !matches!(
        app.input_mode,
        InputMode::Normal
            | InputMode::Visual
            | InputMode::Select
            | InputMode::Search
            | InputMode::Benchmarks
            | InputMode::DownloadManager
            | InputMode::Plan
    ) || app.bench_hw_picker_open
    {
        return handle_popup_mouse(app, mouse, terminal_area);
    }
    if app.dm_confirm_delete {
        let popup = crate::tui_ui::centered_popup(layout[2], 50, 5);
        if click && mouse.row == popup.y + 2 {
            let name = app
                .download_history
                .records
                .get(app.dm_history_cursor)
                .map(|r| r.model_name.as_str())
                .unwrap_or("?");
            use unicode_width::UnicodeWidthStr;
            let start = popup.x + 1 + 12 + UnicodeWidthStr::width(name) as u16;
            if mouse.column == start || mouse.column == start + 2 {
                handle_key(
                    app,
                    KeyEvent::new(
                        KeyCode::Char(if mouse.column == start { 'y' } else { 'n' }),
                        KeyModifiers::NONE,
                    ),
                );
                return true;
            }
        }
        return false;
    }
    if app.dm_editing_dir {
        let config = crate::tui_ui::download_manager_layout(layout[2])[1];
        if click && mouse.row == config.y + 1 && config.contains(position) {
            app.dm_dir_cursor = crate::tui_ui::search_cursor_at(
                &app.dm_dir_input,
                app.dm_dir_cursor,
                usize::from(config.width.saturating_sub(16)),
                usize::from(mouse.column - config.x).saturating_sub(15),
            );
            return true;
        }
        return false;
    }
    if click {
        if crate::tui_ui::status_progress_area(app, layout[3])
            .is_some_and(|area| area.contains(position))
        {
            if !app.show_downloads {
                // Opening another view leaves the benchmark worker running
                app.close_bench();
                app.toggle_downloads();
            }
            return true;
        }
        if crate::tui_ui::status_model_area(app, layout[3])
            .is_some_and(|area| area.contains(position))
        {
            match app.input_mode {
                InputMode::Search => app.exit_search(),
                InputMode::Visual => app.exit_visual_mode(),
                InputMode::Select => app.exit_select_mode(),
                _ => {}
            }
            app.toggle_detail();
            return true;
        }
        let system = layout[0].inner(ratatui::layout::Margin::new(1, 1));
        if app.sim_active
            && mouse.row == system.y
            && mouse.column >= system.x
            && mouse.column < system.x + system.width.min(5)
        {
            app.open_simulation_popup();
            return true;
        }
    }
    let top = crate::tui_ui::search_and_filter_layout(layout[1]);
    if click && let Some(index) = top.iter().position(|rect| rect.contains(position)) {
        if app.input_mode == InputMode::Benchmarks {
            if index == 0 {
                app.bench_search_start();
                return true;
            }
            return false;
        }
        if !matches!(
            app.input_mode,
            InputMode::Normal | InputMode::Search | InputMode::Select | InputMode::Visual
        ) {
            return false;
        }
        let search_cursor = crate::tui_ui::search_cursor_at(
            &app.search_query,
            app.cursor_position,
            usize::from(top[0].width.saturating_sub(2)),
            usize::from(mouse.column.saturating_sub(top[0].x + 1)),
        );
        if app.input_mode == InputMode::Search {
            app.exit_search();
        }
        let key = ['/', 'P', 'U', 'C', 's', 'f', 'a', 'T', 't'][index];
        let key = if index == 5 && mouse.row == top[index].y && mouse.column >= top[index].x + 9 {
            'F'
        } else {
            key
        };
        handle_normal_mode(app, KeyEvent::new(KeyCode::Char(key), KeyModifiers::NONE));
        if index == 0 {
            app.cursor_position = search_cursor;
        }
        return true;
    }
    let table_area = layout[2];
    if app.show_downloads
        || app.show_plan
        || app.show_benchmarks
        || app.show_bench
        || app.show_multi_compare
        || app.show_compare
        || app.show_detail
    {
        return handle_view_mouse(app, mouse, table_area, previous_click, now);
    }
    if !matches!(
        app.input_mode,
        InputMode::Normal | InputMode::Visual | InputMode::Select | InputMode::Search
    ) {
        return false;
    }
    let inner = table_area.inner(ratatui::layout::Margin::new(1, 1));
    // Scrolling only moves the viewport; the selected model remains unchanged
    if mouse.column == table_area.right().saturating_sub(1)
        && table_area.contains(position)
        && app.filtered_fits.len() > usize::from(table_area.height.saturating_sub(3))
    {
        if matches!(
            mouse.kind,
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
        ) {
            scroll_model_table(
                app,
                table_area,
                if mouse.kind == MouseEventKind::ScrollUp {
                    -3
                } else {
                    3
                },
            );
            return true;
        }
        if click || matches!(mouse.kind, MouseEventKind::Drag(MouseButton::Left)) {
            if mouse.row == table_area.y {
                scroll_model_table(app, table_area, -1);
            } else if mouse.row == table_area.bottom().saturating_sub(1) {
                scroll_model_table(app, table_area, 1);
            } else {
                let capacity = usize::from(table_area.height.saturating_sub(3));
                let max_offset = app.filtered_fits.len().saturating_sub(capacity);
                *app.table_state.offset_mut() = usize::from(mouse.row - table_area.y - 1)
                    * max_offset
                    / usize::from(table_area.height.saturating_sub(3).max(1));
                app.table_follow_selection = false;
                app.enqueue_capability_probes_for_visible(capacity);
            }
            return true;
        }
        return false;
    }
    if !inner.contains(position) {
        return false;
    }
    let viewport = crate::tui_ui::model_table_viewport(
        app.filtered_fits.len(),
        app.selected_row,
        app.table_state.offset(),
        usize::from(table_area.height.saturating_sub(3)),
        app.table_follow_selection,
    );
    match mouse.kind {
        MouseEventKind::ScrollUp => scroll_model_table(app, table_area, -3),
        MouseEventKind::ScrollDown => scroll_model_table(app, table_area, 3),
        MouseEventKind::Down(MouseButton::Left) if mouse.row == inner.y => {
            let columns = crate::tui_ui::model_table_columns(table_area);
            let Some(column) = columns.iter().position(|area| area.contains(position)) else {
                return false;
            };
            if app.input_mode == InputMode::Search {
                app.exit_search();
            }
            if app.input_mode == InputMode::Visual {
                app.exit_visual_mode();
            }
            app.select_column = column;
            if !app.sort_model_table_column(column) {
                return false;
            }
        }
        MouseEventKind::Down(MouseButton::Left) => {
            let row = viewport.start + usize::from(mouse.row - inner.y - 1);
            if !viewport.contains(&row) {
                return false;
            }
            if app.input_mode == InputMode::Normal
                && is_double_click(previous_click, now, row, mouse.column)
            {
                handle_normal_mode(app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
                return true;
            }
            app.selected_row = row;
            app.table_follow_selection = true;
            app.last_model_click = Some((now, row, mouse.column));
            app.confirm_download = false;
            app.enqueue_capability_probes_for_visible(24);
        }
        _ => return false,
    }
    true
}

fn is_double_click(
    previous: Option<(Instant, usize, u16)>,
    now: Instant,
    row: usize,
    column: u16,
) -> bool {
    previous.is_some_and(|(when, previous_row, previous_column)| {
        previous_row == row
            && previous_column.abs_diff(column) <= 1
            && now.saturating_duration_since(when) <= Duration::from_millis(500)
    })
}

fn scroll_model_table(app: &mut App, table_area: Rect, delta: isize) {
    let capacity = usize::from(table_area.height.saturating_sub(3));
    let viewport = crate::tui_ui::model_table_viewport(
        app.filtered_fits.len(),
        app.selected_row,
        app.table_state.offset(),
        capacity,
        app.table_follow_selection,
    );
    let max_offset = app.filtered_fits.len().saturating_sub(capacity.max(1));
    *app.table_state.offset_mut() = viewport.start.saturating_add_signed(delta).min(max_offset);
    app.table_follow_selection = false;
    app.last_model_click = None;
    app.enqueue_capability_probes_for_visible(capacity);
}

fn wheel_key(kind: MouseEventKind) -> Option<KeyEvent> {
    let code = match kind {
        MouseEventKind::ScrollUp => KeyCode::Up,
        MouseEventKind::ScrollDown => KeyCode::Down,
        MouseEventKind::ScrollLeft => KeyCode::Left,
        MouseEventKind::ScrollRight => KeyCode::Right,
        _ => return None,
    };
    Some(KeyEvent::new(code, KeyModifiers::NONE))
}

/// Hit-test only recognized key hints, measured in terminal cells, not bytes
/// Descriptions belong to their button; aliases can be clicked individually
fn key_hint_at(text: &str, column: usize) -> Option<KeyEvent> {
    use unicode_width::UnicodeWidthStr;
    let mut start = 0;
    for group in text.split("  ") {
        let width = UnicodeWidthStr::width(group);
        let leading = group.len() - group.trim_start().len();
        if column >= start + leading && column < start + width {
            let label = group.trim().split(':').next()?.split(' ').next()?;
            let label = label.trim_matches(['[', ']']);
            let aliases = if label == "/" {
                vec!["/"]
            } else {
                label.split('/').collect()
            };
            let mut offset = start + leading;
            let mut first = None;
            for alias in aliases {
                let parts = match alias {
                    "↑↓" => vec!["↑", "↓"],
                    "←→" => vec!["←", "→"],
                    "jk" => vec!["j", "k"],
                    "hl" => vec!["h", "l"],
                    _ => vec![alias],
                };
                for part in parts {
                    let (name, modifiers) = part
                        .strip_prefix("Ctrl-")
                        .map_or((part, KeyModifiers::NONE), |name| {
                            (name, KeyModifiers::CONTROL)
                        });
                    let code = match name {
                        "Enter" => Some(KeyCode::Enter),
                        "Esc" => Some(KeyCode::Esc),
                        "Space" => Some(KeyCode::Char(' ')),
                        "Tab" => Some(KeyCode::Tab),
                        "Backspace" => Some(KeyCode::Backspace),
                        "Delete" => Some(KeyCode::Delete),
                        "↑" => Some(KeyCode::Up),
                        "↓" => Some(KeyCode::Down),
                        "←" => Some(KeyCode::Left),
                        "→" => Some(KeyCode::Right),
                        _ if name.chars().count() == 1 => name.chars().next().map(|c| {
                            KeyCode::Char(if modifiers.contains(KeyModifiers::CONTROL) {
                                c.to_ascii_lowercase()
                            } else {
                                c
                            })
                        }),
                        _ => None,
                    };
                    if let Some(code) = code {
                        let key = KeyEvent::new(code, modifiers);
                        first.get_or_insert(key);
                        if column >= offset && column < offset + UnicodeWidthStr::width(part) {
                            return Some(key);
                        }
                    }
                    offset += UnicodeWidthStr::width(part);
                }
                offset += 1;
            }
            return first;
        }
        start += width + 2;
    }
    None
}

fn handle_selection_popup_mouse(
    app: &mut App,
    mouse: MouseEvent,
    popup: Rect,
    count: usize,
    cursor: usize,
) -> bool {
    let position = Position::new(mouse.column, mouse.row);
    let click = mouse.kind == MouseEventKind::Down(MouseButton::Left);
    if !popup.contains(position) {
        return false;
    }
    if let Some(key) = wheel_key(mouse.kind) {
        handle_key(app, key);
        return true;
    }
    if !click {
        return false;
    }
    let mut body = popup.inner(ratatui::layout::Margin::new(1, 1));
    if app.input_mode == InputMode::ProviderPopup {
        if mouse.row == body.y && body.contains(position) {
            app.provider_search_cursor_position = crate::tui_ui::search_cursor_at(
                &app.provider_search,
                app.provider_search_cursor_position,
                usize::from(body.width.saturating_sub(3)),
                usize::from(mouse.column - body.x).saturating_sub(3),
            );
            return true;
        }
        body.y += 1;
        body.height = body.height.saturating_sub(1);
        // These two hints are centered in the bottom border
        let hint = " ^a: all | ^n: clear ";
        let start = popup.x
            + 1
            + popup
                .width
                .saturating_sub(2)
                .saturating_sub(hint.len() as u16)
                / 2;
        if mouse.row == popup.bottom().saturating_sub(1)
            && mouse.column >= start
            && mouse.column < start + hint.len() as u16
        {
            handle_key(
                app,
                KeyEvent::new(
                    KeyCode::Char(if mouse.column - start < 10 { 'a' } else { 'n' }),
                    KeyModifiers::CONTROL,
                ),
            );
            return true;
        }
    }
    if !body.contains(position) {
        return false;
    }
    let first = cursor.saturating_sub(usize::from(body.height).saturating_sub(1));
    let row = first + usize::from(mouse.row - body.y);
    if row >= count {
        return false;
    }
    match app.input_mode {
        InputMode::ProviderPopup => app.provider_cursor = row,
        InputMode::UseCasePopup => app.use_case_cursor = row,
        InputMode::CapabilityPopup => app.capability_cursor = row,
        InputMode::QuantPopup => app.quant_cursor = row,
        InputMode::RunModePopup => app.run_mode_cursor = row,
        InputMode::ParamsBucketPopup => app.params_bucket_cursor = row,
        InputMode::LicensePopup => app.license_cursor = row,
        InputMode::RuntimePopup => app.runtime_cursor = row,
        _ => return false,
    }
    handle_key(app, KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE));
    true
}

fn handle_popup_mouse(app: &mut App, mouse: MouseEvent, area: Rect) -> bool {
    use crate::tui_app::{AdvConfigField, FilterPopupField, SimulationField};
    use crate::tui_ui::centered_popup;
    let popup = match app.input_mode {
        InputMode::Simulation => centered_popup(area, 48, 14),
        InputMode::AdvancedConfig => centered_popup(area, 52, 17),
        InputMode::FilterPopup => centered_popup(area, 56, 21),
        InputMode::DownloadProviderPopup => centered_popup(area, 44, 8),
        InputMode::HelpPopup => centered_popup(area, 52, 32),
        InputMode::BenchOffer => centered_popup(area, 64, 14),
        InputMode::Benchmarks if app.bench_hw_picker_open => centered_popup(
            area,
            52,
            (llmfit_core::benchmarks::HardwarePreset::all().len() as u16 + 5)
                .min(area.height.saturating_sub(6)),
        ),
        _ => return false,
    };
    if !popup.contains(Position::new(mouse.column, mouse.row)) {
        return false;
    }
    if let Some(key) = wheel_key(mouse.kind) {
        if app.input_mode == InputMode::BenchOffer {
            return false;
        }
        handle_key(app, key);
        return true;
    }
    if mouse.kind != MouseEventKind::Down(MouseButton::Left) {
        return false;
    }
    let inner = popup.inner(ratatui::layout::Margin::new(1, 1));
    if !inner.contains(Position::new(mouse.column, mouse.row)) {
        return false;
    }
    let row = mouse.row - inner.y;
    let column = usize::from(mouse.column - inner.x);
    let footer = match app.input_mode {
        InputMode::Simulation => Some((
            7 + u16::from(app.specs.unified_memory) + u16::from(app.sim_active),
            "  Enter:apply  Ctrl-R:reset  Esc:close",
        )),
        InputMode::AdvancedConfig => Some((10, "  Enter:apply  Ctrl-R:reset  Esc:close")),
        InputMode::FilterPopup => Some((17, "  Space:toggle  Ctrl-U:clear  Esc:cancel")),
        _ => None,
    };
    if let Some((y, text)) = footer
        && row == y
    {
        if let Some(key) = key_hint_at(text, column) {
            handle_key(app, key);
            return true;
        }
        return false;
    }
    match app.input_mode {
        InputMode::Simulation if (1..=3).contains(&row) => {
            let (field, len) = match row {
                1 => (SimulationField::Ram, app.sim_ram_input.len()),
                2 => (SimulationField::Vram, app.sim_vram_input.len()),
                _ => (SimulationField::CpuCores, app.sim_cpu_input.len()),
            };
            app.sim_field = field;
            app.sim_cursor_position = column.saturating_sub(14).min(len);
        }
        InputMode::AdvancedConfig if (1..=8).contains(&row) => {
            let fields = [
                (AdvConfigField::Efficiency, &app.adv_config_efficiency_input),
                (AdvConfigField::FactorGpu, &app.adv_config_eff_factor_gpu),
                (
                    AdvConfigField::FactorCpuOffload,
                    &app.adv_config_eff_factor_cpu_offload,
                ),
                (AdvConfigField::FactorMoe, &app.adv_config_eff_factor_moe),
                (AdvConfigField::FactorTp, &app.adv_config_eff_factor_tp),
                (
                    AdvConfigField::FactorCpuOnly,
                    &app.adv_config_eff_factor_cpu_only,
                ),
                (
                    AdvConfigField::ContextCap,
                    &app.adv_config_context_cap_input,
                ),
                (
                    AdvConfigField::DdrBandwidth,
                    &app.adv_config_ddr_bandwidth_input,
                ),
            ];
            let (field, input) = fields[usize::from(row - 1)];
            app.adv_config_field = field;
            app.adv_config_cursor_position = column.saturating_sub(14).min(input.len());
        }
        InputMode::FilterPopup => {
            let (field, len) = match row {
                1 => (
                    FilterPopupField::ParamsMin,
                    app.filter_params_min_input.len(),
                ),
                2 => (
                    FilterPopupField::ParamsMax,
                    app.filter_params_max_input.len(),
                ),
                5 => (
                    FilterPopupField::MemPctMin,
                    app.filter_mem_pct_min_input.len(),
                ),
                6 => (
                    FilterPopupField::MemPctMax,
                    app.filter_mem_pct_max_input.len(),
                ),
                9 => (FilterPopupField::SortDirection, 0),
                12 => (FilterPopupField::FitFilter, 0),
                15 => (FilterPopupField::Availability, 0),
                _ => return false,
            };
            app.filter_field = field;
            app.filter_cursor_position = column.saturating_sub(9).min(len);
            if matches!(
                field,
                FilterPopupField::SortDirection
                    | FilterPopupField::FitFilter
                    | FilterPopupField::Availability
            ) {
                handle_key(app, KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE));
            }
        }
        InputMode::DownloadProviderPopup => {
            let first = if app.download_provider_model.is_some() {
                2
            } else {
                0
            };
            let Some(index) = row
                .checked_sub(first)
                .map(usize::from)
                .filter(|&i| i < app.download_provider_options.len())
            else {
                return false;
            };
            app.download_provider_cursor = index;
            // Clicking selects; the visible Enter action confirms the download
        }
        InputMode::Benchmarks if app.bench_hw_picker_open => {
            let first = app
                .bench_hw_picker_cursor
                .saturating_sub(usize::from(inner.height).saturating_sub(1));
            let index = first + usize::from(row);
            if index > llmfit_core::benchmarks::HardwarePreset::all().len() {
                return false;
            }
            app.bench_hw_picker_cursor = index;
            handle_key(app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        }
        InputMode::BenchOffer => {
            // Use the same wrapped paragraph as the screen: long model names
            // and terminal resizing must not move buttons away from their hits
            use ratatui::{
                buffer::Buffer,
                widgets::{Paragraph, Widget, Wrap},
            };
            let mut buffer = Buffer::empty(inner);
            Paragraph::new(crate::tui_ui::bench_offer_lines(app, &app.theme.colors()))
                .wrap(Wrap { trim: false })
                .render(inner, &mut buffer);
            let text = (inner.x..inner.right())
                .map(|x| buffer[(x, mouse.row)].symbol())
                .collect::<String>();
            use unicode_width::UnicodeWidthStr;
            let share = "Share with llmfit";
            if app.bench_offer_share_unavailable.is_none()
                && text.find(share).is_some_and(|byte| {
                    let start = UnicodeWidthStr::width(&text[..byte]);
                    (start.saturating_sub(4)..start + share.len()).contains(&column)
                })
            {
                handle_key(app, KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE));
                return true;
            }
            if (text.contains("[Enter]") || text.contains("[Space]") || text.contains("[Esc]"))
                && let Some(key) = key_hint_at(&text, column)
            {
                handle_key(app, key);
                return true;
            }
            return false;
        }
        _ => return false,
    }
    true
}

fn handle_view_mouse(
    app: &mut App,
    mouse: MouseEvent,
    area: Rect,
    previous_click: Option<(Instant, usize, u16)>,
    now: Instant,
) -> bool {
    use crate::tui_app::{DownloadManagerFocus, PlanField};
    let position = Position::new(mouse.column, mouse.row);
    if !area.contains(position) {
        return false;
    }
    if app.show_downloads {
        let chunks = crate::tui_ui::download_manager_layout(area);
        let Some(section) = chunks.iter().position(|r| r.contains(position)) else {
            return false;
        };
        let click = mouse.kind == MouseEventKind::Down(MouseButton::Left);
        if !click && wheel_key(mouse.kind).is_none() {
            return false;
        }
        app.dm_focus = [
            DownloadManagerFocus::Active,
            DownloadManagerFocus::Config,
            DownloadManagerFocus::History,
        ][section];
        if let Some(key) = wheel_key(mouse.kind) {
            handle_key(app, key);
            return true;
        }
        if section == 1 && mouse.row == chunks[1].y + 1 {
            handle_key(app, KeyEvent::new(KeyCode::Char('e'), KeyModifiers::NONE));
        }
        if section == 2 && mouse.row >= chunks[2].y + 2 {
            let first = app
                .dm_history_cursor
                .saturating_sub(usize::from(chunks[2].height.saturating_sub(3)).saturating_sub(1));
            let index = first + usize::from(mouse.row - chunks[2].y - 2);
            if index < app.download_history.records.len() {
                app.dm_history_cursor = index;
            }
        }
        return true;
    }
    if mouse.kind == MouseEventKind::Down(MouseButton::Left) {
        if app.show_bench
            && mouse.row == area.y
            && mouse.column > area.x
            && mouse.column < area.right().saturating_sub(1)
        {
            let text = crate::tui_ui::bench_title(app)
                .replace(['(', ')', ','], " ")
                .replace('=', ":");
            if let Some(key) = key_hint_at(&text, usize::from(mouse.column - area.x - 1)) {
                handle_key(app, key);
                return true;
            }
            return false;
        }
        if app.show_detail && crate::tui_ui::detail_pull_at(app, area, position) {
            // The existing picker still requires an explicit download confirmation
            app.start_download();
            return true;
        }
        if app.show_benchmarks
            && let Some((notice_area, lines, key)) = crate::tui_ui::leaderboard_notice(app, area)
            && crate::tui_ui::paragraph_action_at(lines, notice_area, position, false)
        {
            // A clicked notice acts on the view, not on the search text input
            match key {
                "Esc" => app.bench_search_clear(),
                "H" => app.open_bench_hw_picker(),
                _ => app.bench_refresh(),
            }
            return true;
        }
    }
    if let Some(key) = wheel_key(mouse.kind) {
        if app.show_multi_compare {
            let code = if matches!(
                mouse.kind,
                MouseEventKind::ScrollUp | MouseEventKind::ScrollLeft
            ) {
                KeyCode::Left
            } else {
                KeyCode::Right
            };
            handle_key(app, KeyEvent::new(code, KeyModifiers::NONE));
        } else if app.show_bench || app.show_benchmarks || app.show_plan {
            handle_key(app, key);
        } else {
            return false;
        }
        return true;
    }
    if mouse.kind != MouseEventKind::Down(MouseButton::Left) {
        return false;
    }
    let inner = area.inner(ratatui::layout::Margin::new(1, 1));
    if !inner.contains(position) {
        return false;
    }
    let row = mouse.row - inner.y;
    if app.show_plan {
        let (field, len) = match row {
            5 => (PlanField::Context, app.plan_context_input.len()),
            6 => (PlanField::Quant, app.plan_quant_input.len()),
            7 => (PlanField::KvQuant, app.plan_kv_quant_input.len()),
            8 => (PlanField::TargetTps, app.plan_target_tps_input.len()),
            _ => return false,
        };
        app.plan_field = field;
        app.plan_cursor_position = usize::from(mouse.column - inner.x)
            .saturating_sub(14)
            .min(len);
        return true;
    }
    if app.show_benchmarks && !app.bench_loading && !app.bench_entries.is_empty() {
        if row == 0 {
            app.open_bench_hw_picker();
            return true;
        }
        if row >= 2 {
            let index = app.bench_scroll + usize::from(row - 2);
            if index < app.bench_visible_indices().len() {
                app.bench_cursor = index;
                return true;
            }
        }
    }
    if app.show_bench
        && !app.bench_show_detail
        && app.bench_view_mode == crate::tui_app::BenchViewMode::Results
        && row >= 2
    {
        let index = usize::from(row - 2);
        if index < app.bench_model_status.len() {
            if is_double_click(previous_click, now, index, mouse.column) {
                handle_key(app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            } else {
                app.bench_selected_row = index;
                app.last_model_click = Some((now, index, mouse.column));
            }
            return true;
        }
    }
    false
}

fn handle_normal_mode(app: &mut App, key: KeyEvent) {
    // Handle bench quit-confirmation first (overrides all other handlers)
    if app.bench_confirm_quit {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('y') | KeyCode::Char('Y') => {
                app.bench_confirm_quit = false;
                app.close_bench();
            }
            _ => {
                app.bench_confirm_quit = false;
                app.bench_progress = format!(
                    "{}/{} tests — Benchmarking...",
                    app.bench_tests_done, app.bench_tests_total
                );
            }
        }
        return;
    }

    match key.code {
        // Quit
        KeyCode::Char('q') | KeyCode::Esc => {
            if app.show_bench {
                if app.bench_show_detail {
                    app.bench_show_detail = false;
                } else if app.bench_running {
                    app.bench_confirm_quit = true;
                    app.bench_progress =
                        "Inference bench running! Press q again to exit, any key to cancel"
                            .to_string();
                } else {
                    app.close_bench();
                }
            } else if app.show_downloads {
                app.close_downloads();
            } else if app.show_multi_compare {
                app.close_multi_compare();
            } else if app.show_detail {
                app.show_detail = false;
            } else if app.show_compare {
                app.show_compare = false;
            } else {
                app.save_filters();
                app.should_quit = true;
            }
        }

        // Live bench view navigation (only active when bench view is open)
        KeyCode::Char('j') | KeyCode::Down if app.show_bench => {
            if app.bench_show_detail {
                app.live_bench_scroll += 1;
            } else {
                let max = app.bench_model_status.len().saturating_sub(1);
                if app.bench_selected_row < max {
                    app.bench_selected_row += 1;
                }
            }
        }
        KeyCode::Char('k') | KeyCode::Up if app.show_bench => {
            if app.bench_show_detail {
                app.live_bench_scroll = app.live_bench_scroll.saturating_sub(1);
            } else if app.bench_selected_row > 0 {
                app.bench_selected_row -= 1;
            }
        }
        KeyCode::Char('r') if app.show_bench => {
            app.toggle_bench_view();
        }
        KeyCode::Enter if app.show_bench => {
            if app.bench_show_detail {
                app.bench_show_detail = false;
            } else {
                app.bench_show_detail = true;
                app.live_bench_scroll = 0;
            }
        }

        // Navigation — in multi-compare, h/l scroll columns
        KeyCode::Char('h') if app.show_multi_compare => app.multi_compare_scroll_left(),
        KeyCode::Char('l') if app.show_multi_compare => app.multi_compare_scroll_right(),
        KeyCode::Left if app.show_multi_compare => app.multi_compare_scroll_left(),
        KeyCode::Right if app.show_multi_compare => app.multi_compare_scroll_right(),

        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => app.half_page_up(),
        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => app.half_page_down(),
        KeyCode::Up | KeyCode::Char('k') => app.move_up(),
        KeyCode::Down | KeyCode::Char('j') => app.move_down(),
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        KeyCode::Home | KeyCode::Char('g') => app.cycle_top_bottom(),
        // Visual mode
        KeyCode::Char('v') => app.enter_visual_mode(),

        // Select mode
        KeyCode::Char('V') => app.enter_select_mode(),

        // Search
        KeyCode::Char('/') => app.enter_search(),

        // Fit filter
        KeyCode::Char('f') => app.cycle_fit_filter(),

        // Filter popup (range filters, sort direction, fit)
        KeyCode::Char('F') => app.open_filter_popup(),

        // Availability filter
        KeyCode::Char('a') => app.cycle_availability_filter(),

        // TP compatibility filter
        KeyCode::Char('T') => app.cycle_tp_filter(),

        // Sort column
        KeyCode::Char('s') => app.cycle_sort_column(),

        // Theme
        KeyCode::Char('t') => app.cycle_theme(),

        // Plan view
        KeyCode::Char('p') => app.open_plan_mode(),

        // Provider popup
        KeyCode::Char('P') => app.open_provider_popup(),
        KeyCode::Char('U') => app.open_use_case_popup(),
        KeyCode::Char('C') => app.open_capability_popup(),
        KeyCode::Char('L') => app.open_license_popup(),
        KeyCode::Char('R') => app.open_runtime_popup(),
        KeyCode::Char('S') => app.open_simulation_popup(),
        KeyCode::Char('h') => app.open_help_popup(),

        // Installed-first sort toggle (any provider)
        KeyCode::Char('i')
            if app.ollama_available
                || app.mlx_available
                || app.llamacpp_available
                || app.lmstudio_available
                || app.vllm_available
                || app.ramalama_available =>
        {
            app.toggle_installed_first()
        }

        // Download model via best provider (requires confirmation)
        KeyCode::Char('d')
            if app.ollama_available
                || app.mlx_available
                || app.llamacpp_available
                || app.lmstudio_available
                || app.vllm_available =>
        {
            if app.pull_active.is_none() {
                app.start_download();
            }
        }

        // Refresh installed models
        KeyCode::Char('r')
            if app.ollama_available
                || app.mlx_available
                || app.llamacpp_available
                || app.lmstudio_available
                || app.vllm_available
                || app.ramalama_available =>
        {
            app.refresh_installed()
        }

        // Download manager view
        KeyCode::Char('D') => app.toggle_downloads(),

        // Benchmarks view (localmaxxing.com community leaderboard)
        KeyCode::Char('b') => app.open_benchmarks(),

        // Live inference-bench view (llmfit bench — I=open, I again=rerun)
        KeyCode::Char('I') if app.show_bench => app.rerun_bench(),
        KeyCode::Char('I') => app.open_bench(),

        // Advanced Config popup
        KeyCode::Char('A') => app.open_advanced_config_popup(),

        // Detail view
        KeyCode::Enter => app.toggle_detail(),

        // Compare view
        KeyCode::Char('m') => app.mark_selected_for_compare(),
        KeyCode::Char('c') => app.toggle_compare_view(),
        KeyCode::Char('x') => app.clear_compare_mark(),
        KeyCode::Char('y') => app.copy_selected_model_name(),

        _ => {}
    }
}

fn handle_visual_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        // Exit visual mode
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('v') => app.exit_visual_mode(),

        // Navigation (extends selection)
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => app.half_page_up(),
        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => app.half_page_down(),
        KeyCode::Up | KeyCode::Char('k') => app.move_up(),
        KeyCode::Down | KeyCode::Char('j') => app.move_down(),
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        KeyCode::Home | KeyCode::Char('g') => app.cycle_top_bottom(),

        // Mark all selected for compare
        KeyCode::Char('m') => app.mark_selected_for_compare(),

        // Compare first and last in visual selection
        KeyCode::Char('c') => app.visual_compare(),

        _ => {}
    }
}

fn handle_select_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        // Exit select mode
        KeyCode::Esc | KeyCode::Char('q') => app.exit_select_mode(),

        // Column navigation
        KeyCode::Left | KeyCode::Char('h') => app.select_column_left(),
        KeyCode::Right | KeyCode::Char('l') => app.select_column_right(),

        // Activate filter for current column
        KeyCode::Enter | KeyCode::Char(' ') => app.activate_select_column_filter(),

        // Row navigation (still works in select mode)
        KeyCode::Up | KeyCode::Char('k') => app.move_up(),
        KeyCode::Down | KeyCode::Char('j') => app.move_down(),

        _ => {}
    }
}

fn handle_search_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Enter => app.exit_search(),

        KeyCode::Backspace => app.search_backspace(),
        KeyCode::Delete => app.search_delete(),

        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.clear_search();
        }

        KeyCode::Left => app.search_cursor_left(),
        KeyCode::Right => app.search_cursor_right(),

        KeyCode::Char(c) if allows_search_text_input(key.modifiers) => app.search_input(c),

        // Allow navigation while searching
        KeyCode::Up => app.move_up(),
        KeyCode::Down => app.move_down(),

        _ => {}
    }
}

fn handle_provider_popup_mode(app: &mut App, key: KeyEvent) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    match key.code {
        KeyCode::Esc => app.close_provider_popup(),

        KeyCode::Up if shift => app.provider_popup_up(25),
        KeyCode::Down if shift => app.provider_popup_down(25),
        KeyCode::Up => app.provider_popup_up(1),
        KeyCode::Down => app.provider_popup_down(1),

        KeyCode::Left => app.provider_search_cursor_left(),
        KeyCode::Right => app.provider_search_cursor_right(),
        KeyCode::Home => app.provider_search_cursor_home(),
        KeyCode::End => app.provider_search_cursor_end(),

        // Space toggles too (provider names never contain spaces).
        KeyCode::Enter | KeyCode::Char(' ') => app.provider_popup_toggle(),

        KeyCode::Backspace => app.provider_search_backspace(),
        KeyCode::Delete => app.provider_search_delete(),

        // Ctrl shortcuts (typing plain letters filters, so these are modified).
        KeyCode::Char('u') if ctrl => app.provider_search_clear(),
        KeyCode::Char('a') if ctrl => app.provider_popup_select_all(),
        KeyCode::Char('n') if ctrl => app.provider_popup_clear_all(),

        // Plain printable ASCII filters the provider list. Reject modified
        // character events such as macOS Option/Command-arrow artifacts.
        KeyCode::Char(c) if is_plain_provider_filter_char(c, key.modifiers) => {
            app.provider_search_input(c)
        }

        _ => {}
    }
}

fn allows_search_text_input(modifiers: KeyModifiers) -> bool {
    !modifiers.intersects(
        KeyModifiers::CONTROL
            | KeyModifiers::ALT
            | KeyModifiers::SUPER
            | KeyModifiers::HYPER
            | KeyModifiers::META,
    )
}

fn is_plain_provider_filter_char(c: char, modifiers: KeyModifiers) -> bool {
    c.is_ascii_graphic()
        && !modifiers.intersects(
            KeyModifiers::CONTROL
                | KeyModifiers::ALT
                | KeyModifiers::SUPER
                | KeyModifiers::HYPER
                | KeyModifiers::META,
        )
}

fn handle_plan_mode(app: &mut App, key: KeyEvent) {
    // Every plan field is a text input and quant names contain 'q'/'j'/'k'
    // (q4_k_m), so no plain letter may double as a binding here (#781).
    match key.code {
        KeyCode::Esc => app.close_plan_mode(),
        KeyCode::Tab | KeyCode::Down => app.plan_next_field(),
        KeyCode::BackTab | KeyCode::Up => app.plan_prev_field(),
        KeyCode::Left => app.plan_cursor_left(),
        KeyCode::Right => app.plan_cursor_right(),
        KeyCode::Backspace => app.plan_backspace(),
        KeyCode::Delete => app.plan_delete(),
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.plan_clear_field()
        }
        KeyCode::Char(c) if allows_search_text_input(key.modifiers) => app.plan_input(c),
        _ => {}
    }
}

fn handle_use_case_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('U') | KeyCode::Char('q') => app.close_use_case_popup(),

        KeyCode::Up | KeyCode::Char('k') => app.use_case_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.use_case_popup_down(),

        KeyCode::Char(' ') | KeyCode::Enter => app.use_case_popup_toggle(),

        KeyCode::Char('a') => app.use_case_popup_select_all(),

        _ => {}
    }
}

fn handle_capability_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('C') | KeyCode::Char('q') => app.close_capability_popup(),

        KeyCode::Up | KeyCode::Char('k') => app.capability_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.capability_popup_down(),

        KeyCode::Char(' ') | KeyCode::Enter => app.capability_popup_toggle(),

        KeyCode::Char('a') => app.capability_popup_select_all(),

        _ => {}
    }
}

fn handle_download_provider_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => app.close_download_provider_popup(),
        KeyCode::Up | KeyCode::Char('k') => app.download_provider_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.download_provider_popup_down(),
        KeyCode::Enter | KeyCode::Char(' ') => app.confirm_download_provider_selection(),
        _ => {}
    }
}

fn handle_quant_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => app.close_quant_popup(),

        KeyCode::Up | KeyCode::Char('k') => app.quant_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.quant_popup_down(),

        KeyCode::Char(' ') | KeyCode::Enter => app.quant_popup_toggle(),

        KeyCode::Char('a') => app.quant_popup_select_all(),

        _ => {}
    }
}

fn handle_run_mode_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => app.close_run_mode_popup(),

        KeyCode::Up | KeyCode::Char('k') => app.run_mode_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.run_mode_popup_down(),

        KeyCode::Char(' ') | KeyCode::Enter => app.run_mode_popup_toggle(),

        KeyCode::Char('a') => app.run_mode_popup_select_all(),

        _ => {}
    }
}

fn handle_params_bucket_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => app.close_params_bucket_popup(),

        KeyCode::Up | KeyCode::Char('k') => app.params_bucket_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.params_bucket_popup_down(),

        KeyCode::Char(' ') | KeyCode::Enter => app.params_bucket_popup_toggle(),

        KeyCode::Char('a') => app.params_bucket_popup_select_all(),

        _ => {}
    }
}

fn handle_license_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('L') | KeyCode::Char('q') => app.close_license_popup(),

        KeyCode::Up | KeyCode::Char('k') => app.license_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.license_popup_down(),

        KeyCode::Char(' ') | KeyCode::Enter => app.license_popup_toggle(),

        KeyCode::Char('a') => app.license_popup_select_all(),

        _ => {}
    }
}

fn handle_runtime_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('R') | KeyCode::Char('q') => app.close_runtime_popup(),

        KeyCode::Up | KeyCode::Char('k') => app.runtime_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.runtime_popup_down(),

        KeyCode::Char(' ') | KeyCode::Enter => app.runtime_popup_toggle(),

        KeyCode::Char('a') => app.runtime_popup_select_all(),

        _ => {}
    }
}

fn handle_help_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('h') | KeyCode::Char('q') => app.close_help_popup(),
        KeyCode::Up | KeyCode::Char('k') => {
            if app.help_scroll > 0 {
                app.help_scroll -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.help_scroll += 1;
        }
        _ => {}
    }
}

fn handle_simulation_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => app.close_simulation_popup(),

        // Apply simulation
        KeyCode::Enter => app.apply_simulation(),

        // Reset to real hardware
        KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.reset_simulation();
            app.close_simulation_popup();
        }

        // Field navigation
        KeyCode::Tab | KeyCode::Down | KeyCode::Char('j') => app.sim_next_field(),
        KeyCode::BackTab | KeyCode::Up | KeyCode::Char('k') => app.sim_prev_field(),

        // Cursor movement within field
        KeyCode::Left => app.sim_cursor_left(),
        KeyCode::Right => app.sim_cursor_right(),

        // Editing
        KeyCode::Backspace => app.sim_backspace(),
        KeyCode::Delete => app.sim_delete(),
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.sim_clear_field()
        }

        // Character input (digits and decimal point)
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' => app.sim_input(c),

        _ => {}
    }
}

fn handle_advanced_config_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => app.close_advanced_config_popup(),

        // Apply config changes
        KeyCode::Enter => app.apply_advanced_config(),

        // Field navigation
        KeyCode::Tab | KeyCode::Down | KeyCode::Char('j') => app.adv_config_next_field(),
        KeyCode::BackTab | KeyCode::Up | KeyCode::Char('k') => app.adv_config_prev_field(),

        // Cursor movement within field
        KeyCode::Left => app.adv_config_cursor_left(),
        KeyCode::Right => app.adv_config_cursor_right(),

        // Editing
        KeyCode::Backspace => app.adv_config_backspace(),
        KeyCode::Delete => app.adv_config_delete(),
        KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.reset_advanced_config()
        }
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.adv_config_clear_field()
        }

        // Character input (digits and decimal point)
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' => app.adv_config_input(c),

        _ => {}
    }
}

fn handle_download_manager_mode(app: &mut App, key: KeyEvent) {
    use crate::tui_app::DownloadManagerFocus;

    // Handle delete confirmation first
    if app.dm_confirm_delete {
        match key.code {
            KeyCode::Char('y') => {
                app.delete_selected_download();
                app.dm_confirm_delete = false;
            }
            _ => app.dm_confirm_delete = false,
        }
        return;
    }

    // Handle directory editing mode
    if app.dm_editing_dir {
        match key.code {
            KeyCode::Esc => {
                app.dm_editing_dir = false;
            }
            KeyCode::Enter => {
                app.apply_download_dir();
                app.dm_editing_dir = false;
            }
            KeyCode::Backspace => {
                app.dm_dir_backspace();
            }
            KeyCode::Delete => {
                app.dm_dir_delete();
            }
            KeyCode::Left => {
                app.dm_dir_cursor_left();
            }
            KeyCode::Right => {
                app.dm_dir_cursor_right();
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.dm_dir_clear();
            }
            KeyCode::Char(c) => {
                app.insert_dm_dir_char(c);
            }
            _ => {}
        }
        return;
    }

    match key.code {
        // Close
        KeyCode::Esc | KeyCode::Char('D') | KeyCode::Char('q') => app.close_downloads(),

        // Focus cycling
        KeyCode::Tab => app.dm_focus = app.dm_focus.next(),
        KeyCode::BackTab => app.dm_focus = app.dm_focus.prev(),

        // Navigation within history
        KeyCode::Up | KeyCode::Char('k') if app.dm_focus == DownloadManagerFocus::History => {
            if app.dm_history_cursor > 0 {
                app.dm_history_cursor -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') if app.dm_focus == DownloadManagerFocus::History => {
            let len = app.download_history.records.len();
            if len > 0 && app.dm_history_cursor < len - 1 {
                app.dm_history_cursor += 1;
            }
        }

        // Delete model
        KeyCode::Char('x') if app.dm_focus == DownloadManagerFocus::History => {
            if !app.download_history.records.is_empty() {
                app.dm_confirm_delete = true;
            }
        }

        // Edit download directory
        KeyCode::Char('e') if app.dm_focus == DownloadManagerFocus::Config => {
            app.start_editing_download_dir();
        }

        _ => {}
    }
}

fn handle_filter_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => app.close_filter_popup(),

        KeyCode::Enter => app.apply_filter_popup(),

        // Field navigation
        KeyCode::Tab | KeyCode::Down => app.filter_next_field(),
        KeyCode::BackTab | KeyCode::Up => app.filter_prev_field(),

        // Cursor movement within field
        KeyCode::Left => app.filter_cursor_left(),
        KeyCode::Right => app.filter_cursor_right(),

        // Editing
        KeyCode::Backspace => app.filter_backspace(),
        KeyCode::Delete => app.filter_delete(),
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if matches!(
                app.filter_field,
                crate::tui_app::FilterPopupField::SortDirection
                    | crate::tui_app::FilterPopupField::FitFilter
                    | crate::tui_app::FilterPopupField::Availability
            ) {
                return;
            }
            app.filter_clear_active_input();
        }

        // Sort direction toggle
        KeyCode::Char(' ')
            if app.filter_field == crate::tui_app::FilterPopupField::SortDirection =>
        {
            app.filter_toggle_sort_direction()
        }

        // Fit filter cycling
        KeyCode::Char(' ') if app.filter_field == crate::tui_app::FilterPopupField::FitFilter => {
            app.cycle_filter_fit()
        }

        // Availability filter cycling (All / GGUF Avail / Installed)
        KeyCode::Char(' ')
            if app.filter_field == crate::tui_app::FilterPopupField::Availability =>
        {
            app.cycle_filter_availability()
        }

        // Numeric input
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' => app.filter_input(c),

        _ => {}
    }
}

fn handle_bench_offer_mode(app: &mut App, key: KeyEvent) {
    use crate::tui_app::BenchOfferState;
    match app.bench_offer_state {
        BenchOfferState::Offer => match key.code {
            KeyCode::Enter => app.bench_offer_confirm(),
            KeyCode::Char(' ') | KeyCode::Char('s') => app.bench_offer_toggle_share(),
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('n') => app.bench_offer_dismiss(),
            _ => {}
        },
        // While running, Esc detaches the worker and drops to the leaderboard.
        BenchOfferState::Running => {
            if key.code == KeyCode::Esc {
                app.bench_offer_dismiss();
            }
        }
        BenchOfferState::Done | BenchOfferState::Error => match key.code {
            KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') => app.bench_offer_dismiss(),
            _ => {}
        },
    }
}

fn handle_benchmarks_mode(app: &mut App, key: KeyEvent) {
    // Hardware picker sub-modal takes priority when open
    if app.bench_hw_picker_open {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('H') => app.close_bench_hw_picker(),
            KeyCode::Up | KeyCode::Char('k') => app.bench_hw_picker_up(),
            KeyCode::Down | KeyCode::Char('j') => app.bench_hw_picker_down(),
            KeyCode::Enter | KeyCode::Char(' ') => app.bench_hw_picker_select(),
            _ => {}
        }
        return;
    }

    // `/` search box captures keystrokes while active
    if app.bench_search_active {
        match key.code {
            KeyCode::Esc => app.bench_search_clear(),
            KeyCode::Enter => app.bench_search_accept(),
            KeyCode::Backspace => app.bench_search_backspace(),
            KeyCode::Up => app.bench_move_up(),
            KeyCode::Down => app.bench_move_down(),
            KeyCode::Char(c) if allows_search_text_input(key.modifiers) => {
                app.bench_search_input(c)
            }
            _ => {}
        }
        return;
    }

    match key.code {
        // With a filter applied, Esc clears it first; q/b still close directly.
        KeyCode::Esc if !app.bench_search_query.is_empty() => app.bench_search_clear(),
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('b') => app.close_benchmarks(),
        KeyCode::Char('/') => app.bench_search_start(),
        KeyCode::Up | KeyCode::Char('k') => app.bench_move_up(),
        KeyCode::Down | KeyCode::Char('j') => app.bench_move_down(),
        KeyCode::Char('r') => app.bench_refresh(),
        KeyCode::Char('H') => app.open_bench_hw_picker(),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui_app::PlanField;
    use llmfit_core::hardware::{GpuBackend, SystemSpecs};

    fn plan_mode_app() -> App {
        let mut app = App::with_specs_context_and_config(
            SystemSpecs {
                total_ram_gb: 16.0,
                available_ram_gb: 12.0,
                total_cpu_cores: 8,
                cpu_name: "Test CPU".to_string(),
                has_gpu: false,
                gpu_vram_gb: None,
                total_gpu_vram_gb: None,
                gpu_available_gb: None,
                gpu_name: None,
                gpu_count: 0,
                unified_memory: false,
                backend: GpuBackend::CpuX86,
                gpus: Vec::new(),
                cluster_mode: false,
                cluster_node_count: 0,
            },
            None,
            None,
        );
        app.input_mode = InputMode::Plan;
        app.show_plan = true;
        app
    }

    fn plain(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    fn click_at(app: &mut App, area: Rect, column: u16, row: u16) -> bool {
        handle_mouse(
            app,
            mouse(MouseEventKind::Down(MouseButton::Left), column, row),
            area,
        )
    }

    fn normal_mode_app() -> App {
        let mut app = plan_mode_app();
        app.input_mode = InputMode::Normal;
        app.show_plan = false;
        app
    }

    #[test]
    fn mouse_selects_visible_rows_and_bounds_wheel_navigation() {
        let mut app = normal_mode_app();
        app.filtered_fits = (0..app.all_fits.len().min(100)).rev().collect();
        let count = app.filtered_fits.len();
        assert!(count > 20);
        app.selected_row = count - 1;
        update_model_viewport(&mut app, Rect::new(0, 0, 160, 24));

        // A resize between frames must not leave hit testing on the old offset
        let area = Rect::new(0, 0, 160, 20);
        let table = crate::tui_ui::main_layout(area)[2];
        let capacity = usize::from(table.height - 3);
        let click = mouse(
            MouseEventKind::Down(MouseButton::Left),
            table.x + 4,
            table.y + 2,
        );
        app.confirm_download = true;
        assert!(handle_mouse(&mut app, click, area));
        assert_eq!(app.selected_row, count - capacity);
        assert_eq!(
            app.selected_fit().unwrap().model.name,
            app.all_fits[app.filtered_fits[count - capacity]].model.name
        );
        assert!(!app.confirm_download);

        app.selected_row = 0;
        app.sort_column = llmfit_core::fit::SortColumn::Score;
        app.header_sort_column = None;
        let y = table.y + 1;
        let header = rendered_position_in(&mut app, area, "Score", y..y + 1);
        let up = mouse(MouseEventKind::ScrollUp, click.column, click.row);
        let down = mouse(MouseEventKind::ScrollDown, click.column, click.row);
        handle_mouse(&mut app, up, area);
        assert_eq!(app.selected_row, 0);
        handle_mouse(&mut app, down, area);
        assert_eq!(app.selected_row, 0);
        assert_eq!(app.table_state.offset(), 3);
        assert_eq!(
            rendered_position_in(&mut app, area, "Score", y..y + 1),
            header
        );
        update_model_viewport(&mut app, area);
        assert_eq!(
            app.table_state.offset(),
            3,
            "redraw must not snap back to selection"
        );
        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        update_model_viewport(&mut app, area);
        assert_eq!(app.selected_row, 1);
        assert_eq!(
            app.table_state.offset(),
            1,
            "keyboard reveals the selected row"
        );
        app.selected_row = count - 1;
        handle_mouse(&mut app, down, area);
        assert_eq!(app.selected_row, count - 1);

        // Filtering can shrink the list while a previous scroll offset remains
        app.filtered_fits.truncate(2);
        app.selected_row = 0;
        assert!(click_at(&mut app, area, click.column, click.row + 1));
        assert_eq!(app.selected_row, 1);
        assert!(!click_at(&mut app, area, click.column, click.row + 2));
        app.filtered_fits.clear();
        app.selected_row = 0;
        assert!(!handle_mouse(&mut app, click, area));
        handle_mouse(&mut app, down, area);
        handle_mouse(&mut app, up, area);
        assert_eq!(app.selected_row, 0);
    }

    #[test]
    fn mouse_headers_match_rendered_columns_with_and_without_selection() {
        use llmfit_core::fit::SortColumn;

        let mut app = normal_mode_app();
        for width in [120, 220] {
            for empty in [false, true] {
                app.filtered_fits = if empty { Vec::new() } else { vec![0, 1] };
                app.selected_row = 0;
                app.sort_column = SortColumn::Score;
                app.header_sort_column = None;
                app.sort_ascending = false;
                let area = Rect::new(0, 0, width, 24);
                update_model_viewport(&mut app, area);
                let y = crate::tui_ui::main_layout(area)[2].y + 1;
                let x = rendered_position_in(&mut app, area, "Score", y..y + 1).x;
                assert!(
                    click_at(&mut app, area, x, y),
                    "width={width}, empty={empty}, x={x}, y={y}, mode={:?}, detail={}",
                    app.input_mode,
                    app.show_detail
                );
                assert_eq!(app.sort_column, SortColumn::Score);
                assert!(app.sort_ascending, "width={width}, empty={empty}");
            }
        }
    }

    #[test]
    fn mouse_ignores_overlays_borders_and_non_action_events() {
        let mut app = plan_mode_app();
        app.show_plan = false;
        app.filtered_fits = vec![0, 1, 2];
        app.selected_row = 0;
        let area = Rect::new(0, 0, 160, 24);
        let table = crate::tui_ui::main_layout(area)[2];
        let click = mouse(
            MouseEventKind::Down(MouseButton::Left),
            table.x + 4,
            table.y + 3,
        );
        for mode in [
            InputMode::HelpPopup,
            InputMode::ProviderPopup,
            InputMode::BenchOffer,
        ] {
            app.input_mode = mode;
            assert!(!handle_mouse(&mut app, click, area));
            assert!(!handle_mouse(
                &mut app,
                mouse(MouseEventKind::ScrollDown, click.column, click.row),
                area
            ));
            assert_eq!(app.selected_row, 0);
        }
        app.input_mode = InputMode::Normal;
        app.show_detail = true;
        assert!(!handle_mouse(&mut app, click, area));
        app.show_detail = false;
        app.bench_confirm_quit = true;
        assert!(!handle_mouse(&mut app, click, area));
        app.bench_confirm_quit = false;
        for kind in [
            MouseEventKind::Moved,
            MouseEventKind::Up(MouseButton::Left),
            MouseEventKind::Drag(MouseButton::Left),
            MouseEventKind::Down(MouseButton::Right),
        ] {
            assert!(!handle_mouse(
                &mut app,
                mouse(kind, click.column, click.row),
                area
            ));
        }
        assert!(!handle_mouse(
            &mut app,
            MouseEvent {
                modifiers: KeyModifiers::CONTROL,
                ..click
            },
            area
        ));
        for (x, y) in [
            (table.x, click.row),
            (table.right() - 1, click.row),
            (click.column, table.y),
            (click.column, table.bottom() - 1),
            (0, 0),
        ] {
            assert!(!handle_mouse(&mut app, mouse(click.kind, x, y), area));
        }
        assert_eq!(app.selected_row, 0);
        assert!(!handle_mouse(&mut app, click, Rect::new(0, 0, 1, 1)));
    }

    fn rendered_position(app: &mut App, area: Rect, label: &str) -> Position {
        rendered_position_in(app, area, label, 0..area.height)
    }

    fn rendered_position_in(
        app: &mut App,
        area: Rect,
        label: &str,
        rows: std::ops::Range<u16>,
    ) -> Position {
        use ratatui::{Terminal, backend::TestBackend};
        let mut terminal =
            Terminal::new(TestBackend::new(area.width, area.height)).expect("terminal");
        terminal
            .draw(|f| crate::tui_ui::draw(f, app))
            .expect("frame");
        let buffer = terminal.backend().buffer();
        let cells = label.chars().count() as u16;
        for y in rows {
            for x in 0..area.width.saturating_sub(cells) {
                if (x..x + cells)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    == label
                {
                    // "Mode" must not match the beginning of "Model"
                    if label.ends_with(|c: char| c.is_alphanumeric())
                        && x + cells < area.width
                        && buffer[(x + cells, y)]
                            .symbol()
                            .chars()
                            .next()
                            .is_some_and(char::is_alphanumeric)
                    {
                        continue;
                    }
                    return Position::new(x, y);
                }
            }
        }
        panic!("missing visible label: {label}");
    }

    fn click_label(app: &mut App, area: Rect, label: &str) {
        let p = rendered_position(app, area, label);
        assert!(
            handle_mouse(
                app,
                mouse(MouseEventKind::Down(MouseButton::Left), p.x, p.y),
                area
            ),
            "label: {label}"
        );
    }

    #[test]
    fn mouse_top_controls_and_footer_follow_rendered_actions() {
        let area = Rect::new(0, 0, 220, 30);
        let mut app = normal_mode_app();
        click_label(&mut app, area, "Providers");
        assert_eq!(app.input_mode, InputMode::ProviderPopup);
        click_label(&mut app, area, "Esc:");
        assert_eq!(app.input_mode, InputMode::Normal);
        let fit = app.fit_filter;
        click_label(&mut app, area, "[f]");
        assert_ne!(app.fit_filter, fit);
        let fit = app.fit_filter;
        click_label(&mut app, area, "Filter");
        assert_eq!(app.input_mode, InputMode::FilterPopup);
        assert_eq!(app.fit_filter, fit);
        click_label(&mut app, area, "Esc:cancel");
        click_label(&mut app, area, "[F]");
        assert_eq!(app.input_mode, InputMode::FilterPopup);
        click_label(&mut app, area, "Esc:cancel");
        let sort = app.sort_column;
        click_label(&mut app, area, "[s]");
        assert_ne!(app.sort_column, sort);
        app.search_query = "abc".to_string();
        app.cursor_position = 3;
        click_label(&mut app, area, "Search");
        assert_eq!(app.input_mode, InputMode::Search);
        click_label(&mut app, area, "Ctrl-U:clear");
        assert!(app.search_query.is_empty());
        app.search_query = "keep".to_string();
        click_label(&mut app, area, "[s]");
        assert_eq!(app.input_mode, InputMode::Normal);
        assert_eq!(app.search_query, "keep");
    }

    #[test]
    fn mouse_headers_only_sort_and_keep_the_selected_model() {
        let area = Rect::new(0, 0, 300, 30);
        let mut app = normal_mode_app();
        let selected = app.selected_fit().expect("model").model.name.clone();
        let providers = app.selected_providers.clone();
        let use_cases = app.selected_use_cases.clone();
        let fit = app.fit_filter;
        let availability = app.availability_filter;
        let y = crate::tui_ui::main_layout(area)[2].y + 1;
        for (column, label) in [
            (1, "Inst"),
            (2, "Model"),
            (3, "Provider"),
            (4, "Params"),
            (5, "Score"),
            (6, "tok/s"),
            (7, "Quant"),
            (8, "Disk"),
            (9, "Mode"),
            (10, "Mem %"),
            (11, "Ctx"),
            (12, "Date"),
            (13, "Fit"),
            (14, "Use Case"),
        ] {
            for toggle in 0..2 {
                let ascending = app.table_sort_is_ascending();
                let p = rendered_position_in(&mut app, area, label, y..y + 1);
                assert!(click_at(&mut app, area, p.x, p.y));
                assert_eq!(app.sorted_table_column(), column);
                if toggle == 1 {
                    assert_ne!(
                        app.table_sort_is_ascending(),
                        ascending,
                        "direction: {label}"
                    );
                }
                assert_eq!(
                    app.input_mode,
                    InputMode::Normal,
                    "header must not open a filter: {label}"
                );
                assert_eq!(app.selected_fit().expect("model").model.name, selected);
                assert_eq!(app.selected_providers, providers);
                assert_eq!(app.selected_use_cases, use_cases);
                assert_eq!(app.fit_filter, fit);
                assert_eq!(app.availability_filter, availability);
                assert_eq!(app.table_state.offset(), 0);
            }
        }
        // Alphabetical model sorting orders rows rather than merely changing a label
        app.sort_model_table_column(2);
        let names = app
            .filtered_fits
            .iter()
            .map(|&i| &app.all_fits[i])
            .filter(|fit| fit.fit_level != llmfit_core::fit::FitLevel::TooTight)
            .map(|fit| fit.model.name.to_lowercase())
            .collect::<Vec<_>>();
        assert!(names.windows(2).all(|pair| pair[0] <= pair[1]));
        app.sort_model_table_column(2);
        let names = app
            .filtered_fits
            .iter()
            .map(|&i| &app.all_fits[i])
            .filter(|fit| fit.fit_level != llmfit_core::fit::FitLevel::TooTight)
            .map(|fit| fit.model.name.to_lowercase())
            .collect::<Vec<_>>();
        assert!(names.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    #[test]
    fn mouse_provider_rows_use_filtered_scrolled_indices_and_shield_table() {
        let area = Rect::new(0, 0, 80, 12);
        let mut app = plan_mode_app();
        app.show_plan = false;
        app.input_mode = InputMode::ProviderPopup;
        app.providers = (0..60)
            .map(|i| format!("{}-{i:02}", if i % 2 == 0 { "match" } else { "other" }))
            .collect();
        app.selected_providers = vec![true; 60];
        app.provider_search = "match".to_string();
        app.provider_cursor = 25;
        app.selected_row = 0;
        let model_row = app.selected_row;
        click_label(&mut app, area, "match-42");
        assert_eq!(app.provider_cursor, 21);
        assert!(!app.selected_providers[42]);
        assert_eq!(app.selected_providers.iter().filter(|&&s| !s).count(), 1);
        assert_eq!(app.selected_row, model_row);
        let (popup, _, _) =
            crate::tui_ui::selection_popup(&app, app.input_mode, area).expect("popup");
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::ScrollUp, popup.x + 2, popup.y + 2),
            area,
        );
        assert_eq!(app.provider_cursor, 20);
        assert_eq!(app.selected_row, model_row);
        assert!(!click_at(&mut app, area, 1, 5));
        app.provider_search = "no matches here".to_string();
        app.provider_cursor = 0;
        let (popup, _, _) =
            crate::tui_ui::selection_popup(&app, app.input_mode, area).expect("popup");
        assert!(!click_at(&mut app, area, popup.x + 2, popup.y + 2));
    }

    #[test]
    fn mouse_footer_respects_clipping_progress_and_key_aliases() {
        let mut app = plan_mode_app();
        app.show_plan = false;
        app.input_mode = InputMode::Search;
        app.search_query = "needle".to_string();
        app.cursor_position = 6;
        app.pull_status = Some("downloading model".to_string());
        app.pull_percent = Some(25.0);
        for width in [60, 160] {
            let area = Rect::new(0, 0, width, 24);
            let p = rendered_position(&mut app, area, "downloading");
            assert!(click_at(&mut app, area, p.x, p.y));
            assert!(app.show_downloads);
            assert_eq!(app.search_query, "needle");
            app.close_downloads();
            app.input_mode = InputMode::Search;
        }
        click_label(&mut app, Rect::new(0, 0, 160, 24), "Ctrl-U:clear");
        assert!(app.search_query.is_empty());
        assert_eq!(
            key_hint_at(" ↑↓/jk:nav", 2).expect("down").code,
            KeyCode::Down
        );
        assert_eq!(
            key_hint_at(" /:search", 1).expect("search").code,
            KeyCode::Char('/')
        );
        assert_eq!(
            key_hint_at(" Ctrl-U:clear", 8).expect("clear").modifiers,
            KeyModifiers::CONTROL
        );
        assert!(key_hint_at(" type:edit", 1).is_none());
    }

    #[test]
    fn mouse_double_click_opens_details_and_scrollbar_keeps_selection() {
        let mut app = normal_mode_app();
        app.filtered_fits = (0..app.all_fits.len().min(100)).collect();
        app.selected_row = 0;
        let area = Rect::new(0, 0, 160, 24);
        let table = crate::tui_ui::main_layout(area)[2];
        let click = mouse(MouseEventKind::Down(MouseButton::Left), 10, table.y + 2);
        let now = Instant::now();
        assert!(handle_mouse_at(&mut app, click, area, now));
        assert!(!app.show_detail, "single click only selects");
        assert!(handle_mouse_at(
            &mut app,
            click,
            area,
            now + Duration::from_millis(700)
        ));
        assert!(!app.show_detail, "two slow clicks are not a double click");
        assert!(handle_mouse_at(
            &mut app,
            click,
            area,
            now + Duration::from_millis(900)
        ));
        assert!(app.show_detail);
        handle_key(&mut app, plain('q'));
        let last_offset = app.filtered_fits.len() - usize::from(table.height - 3);
        assert!(click_at(
            &mut app,
            area,
            table.right() - 1,
            table.bottom() - 2
        ));
        assert_eq!(app.selected_row, 0);
        assert_eq!(app.table_state.offset(), last_offset);
        update_model_viewport(&mut app, area);
        assert_eq!(app.table_state.offset(), last_offset);
        assert!(handle_mouse(
            &mut app,
            mouse(
                MouseEventKind::Drag(MouseButton::Left),
                table.right() - 1,
                table.y + 1
            ),
            area
        ));
        assert_eq!(app.selected_row, 0);
        assert_eq!(app.table_state.offset(), 0);
    }

    #[test]
    fn mouse_fields_and_subdialogs_use_their_own_actions() {
        use crate::tui_app::{DownloadManagerFocus, SimulationField};
        let area = Rect::new(0, 0, 160, 30);
        let mut app = plan_mode_app();
        app.show_plan = false;
        app.open_simulation_popup();
        click_label(&mut app, area, "VRAM (GB):");
        assert_eq!(app.sim_field, SimulationField::Vram);
        click_label(&mut app, area, "Esc:close");
        app.show_downloads = true;
        app.input_mode = InputMode::DownloadManager;
        app.dm_focus = DownloadManagerFocus::Config;
        let config = crate::tui_ui::download_manager_layout(crate::tui_ui::main_layout(area)[2])[1];
        assert!(!handle_mouse(
            &mut app,
            mouse(MouseEventKind::Moved, config.x + 2, config.y),
            area
        ));
        assert_eq!(app.dm_focus, DownloadManagerFocus::Config);
        app.dm_focus = DownloadManagerFocus::History;
        assert!(!handle_mouse(
            &mut app,
            mouse(MouseEventKind::Moved, config.x + 2, config.y),
            area
        ));
        assert_eq!(app.dm_focus, DownloadManagerFocus::History);
        click_at(&mut app, area, config.x + 2, config.y);
        assert_eq!(app.dm_focus, DownloadManagerFocus::Config);
        assert!(!app.dm_editing_dir, "section border only focuses");
        click_at(&mut app, area, config.x + 2, config.y + 1);
        assert!(app.dm_editing_dir);
        app.dm_editing_dir = true;
        app.dm_dir_input = "example".to_string();
        app.dm_dir_cursor = 7;
        click_label(&mut app, area, "Ctrl-U:clear");
        assert!(app.dm_dir_input.is_empty());
        click_label(&mut app, area, "Esc:cancel");
        assert!(!app.dm_editing_dir);
        app.dm_confirm_delete = true;
        click_label(&mut app, area, "n:cancel");
        assert!(!app.dm_confirm_delete);
        app.download_history.records = vec![crate::download_history::DownloadRecord {
            model_name: "模型".to_string(),
            provider: "test".to_string(),
            result: crate::download_history::DownloadResult::Success,
            timestamp: 0,
            file_path: None,
        }];
        app.dm_history_cursor = 0;
        app.dm_confirm_delete = true;
        let p = rendered_position(&mut app, area, "? (y/n)");
        assert!(!click_at(&mut app, area, p.x, p.y));
        assert!(
            app.dm_confirm_delete,
            "punctuation must not confirm deletion"
        );
        let p = rendered_position(&mut app, area, "(y/n)");
        assert!(click_at(&mut app, area, p.x + 3, p.y));
        assert!(!app.dm_confirm_delete);
        assert_eq!(
            app.download_history.records.len(),
            1,
            "only cancel deletion in tests"
        );
        app.show_downloads = false;
        app.input_mode = InputMode::Benchmarks;
        app.bench_hw_picker_open = true;
        click_label(&mut app, area, "Esc:cancel");
        assert!(!app.bench_hw_picker_open);
        app.input_mode = InputMode::BenchOffer;
        app.bench_offer_state = crate::tui_app::BenchOfferState::Offer;
        app.bench_offer_share_unavailable = None;
        app.bench_offer_pending = 0;
        let share = app.bench_offer_share;
        let p = rendered_position(&mut app, area, "opens a PR");
        assert!(!click_at(&mut app, area, p.x, p.y));
        assert_eq!(
            app.bench_offer_share, share,
            "sharing explanation is not a toggle"
        );
        click_label(&mut app, area, "Share with llmfit");
        assert_eq!(app.bench_offer_share, !share);
        // Never confirm a run/share in tests: that starts external work
        click_label(&mut app, area, "[Esc] Skip");
        assert_ne!(app.input_mode, InputMode::BenchOffer);
    }

    #[test]
    fn mouse_status_shortcuts_and_simulation_respect_popups() {
        let area = Rect::new(0, 0, 160, 30);
        let mut app = normal_mode_app();
        let footer = crate::tui_ui::main_layout(area)[3];
        app.table_follow_selection = false;
        *app.table_state.offset_mut() = 10;
        let click = mouse(
            MouseEventKind::Down(MouseButton::Left),
            footer.x + 5,
            footer.y,
        );
        assert!(handle_mouse(&mut app, click, area));
        assert!(app.show_detail);
        app.show_detail = false;
        app.open_help_popup();
        assert!(!handle_mouse(&mut app, click, area));
        assert!(!app.show_detail);
        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        app.sim_active = true;
        click_label(&mut app, area, "SIM");
        assert_eq!(app.input_mode, InputMode::Simulation);
        click_label(&mut app, area, "Esc:close");
        app.show_plan = true;
        app.input_mode = InputMode::Plan;
        click_label(&mut app, area, "SIM");
        click_label(&mut app, area, "Esc:close");
        assert_eq!(app.input_mode, InputMode::Plan);
        app.show_plan = false;
        app.show_benchmarks = true;
        app.input_mode = InputMode::Benchmarks;
        app.bench_loading = false;
        click_label(&mut app, area, "SIM");
        click_label(&mut app, area, "Esc:close");
        assert_eq!(app.input_mode, InputMode::Benchmarks);
        app.show_benchmarks = false;
        app.input_mode = InputMode::Normal;
        app.pull_status = Some("Marked for compare".to_string());
        let p = rendered_position(&mut app, area, "Marked");
        assert!(!click_at(&mut app, area, p.x, p.y));
        assert!(
            !app.show_downloads,
            "informational status is not a download control"
        );
    }

    #[test]
    fn mouse_detail_download_hint_follows_wrapping_without_starting_a_download() {
        let mut app = normal_mode_app();
        let index = app
            .all_fits
            .iter()
            .position(|fit| !fit.model.gguf_sources.is_empty())
            .expect("GGUF model");
        app.filtered_fits = vec![index];
        app.selected_row = 0;
        app.all_fits[index].installed = false;
        // A long Unicode name changes the hint's wrapped screen position
        app.all_fits[index].model.name =
            "測試 / A long model name for wrapped metadata".to_string();
        app.llamacpp_available = true;
        app.show_detail = true;
        for width in [80, 160] {
            let area = Rect::new(0, 0, width, 60);
            let p = rendered_position(&mut app, area, "press d");
            let table = crate::tui_ui::main_layout(area)[2];
            assert!(crate::tui_ui::detail_pull_at(&app, table, p));
            let text = rendered_position(&mut app, area, "Installed:");
            assert!(!click_at(&mut app, area, text.x, text.y));
            assert!(click_at(&mut app, area, p.x, p.y));
            assert_eq!(app.input_mode, InputMode::DownloadProviderPopup);
            assert!(
                app.pull_active.is_none(),
                "click opens the picker; no download starts"
            );
            app.close_download_provider_popup();
        }
        app.llamacpp_available = false;
        let area = Rect::new(0, 0, 160, 60);
        let p = rendered_position(&mut app, area, "No runtime detected");
        assert!(!click_at(&mut app, area, p.x, p.y));
    }

    #[test]
    fn mouse_live_benchmark_uses_double_click_and_title_actions() {
        use crate::tui_app::{BenchModelState, BenchModelStatus, BenchViewMode};
        let mut app = normal_mode_app();
        app.show_bench = true;
        app.bench_model_status = vec![BenchModelStatus {
            name: "test model".to_string(),
            state: BenchModelState::Complete,
            roles_done: 0,
            roles_total: 0,
            current_role: String::new(),
            last_quality: 0.0,
            last_speed: 0.0,
        }];
        let area = Rect::new(0, 0, 160, 30);
        let p = rendered_position(&mut app, area, "test model");
        let click = mouse(MouseEventKind::Down(MouseButton::Left), p.x, p.y);
        let now = Instant::now();
        assert!(handle_mouse_at(&mut app, click, area, now));
        assert!(!app.bench_show_detail);
        assert!(handle_mouse_at(
            &mut app,
            click,
            area,
            now + Duration::from_millis(700)
        ));
        assert!(!app.bench_show_detail);
        assert!(handle_mouse_at(
            &mut app,
            click,
            area,
            now + Duration::from_millis(900)
        ));
        assert!(app.bench_show_detail);
        let y = crate::tui_ui::main_layout(area)[2].y;
        let p = rendered_position_in(&mut app, area, "Enter/q=close detail", y..y + 1);
        assert!(click_at(&mut app, area, p.x, p.y));
        assert!(!app.bench_show_detail);
        let p = rendered_position_in(&mut app, area, "r=routing", y..y + 1);
        assert!(click_at(&mut app, area, p.x, p.y));
        assert_eq!(app.bench_view_mode, BenchViewMode::Routing);
        let p = rendered_position_in(&mut app, area, "r=results", y..y + 1);
        assert!(click_at(&mut app, area, p.x, p.y));
        assert_eq!(app.bench_view_mode, BenchViewMode::Results);
        let y = crate::tui_ui::main_layout(area)[2].y;
        let p = rendered_position_in(&mut app, area, "Quality", y..y + 1);
        assert!(!click_at(&mut app, area, p.x, p.y));
        assert!(!app.should_quit);
    }

    #[test]
    fn mouse_leaderboard_notices_only_activate_the_visible_action() {
        let area = Rect::new(0, 0, 100, 30);
        let mut app = plan_mode_app();
        app.show_plan = false;
        app.show_benchmarks = true;
        app.input_mode = InputMode::Benchmarks;
        app.bench_loading = false;
        app.bench_entries.clear();
        click_label(&mut app, area, "H:pick");
        assert!(app.bench_hw_picker_open);
        click_label(&mut app, area, "Esc:");
        assert!(!app.bench_hw_picker_open);
        app.bench_error = Some("Example error".to_string());
        let p = rendered_position(&mut app, area, "r:retry");
        let table = crate::tui_ui::main_layout(area)[2];
        let (notice_area, lines, key) =
            crate::tui_ui::leaderboard_notice(&app, table).expect("error notice");
        assert_eq!(key, "r");
        // Check retry hit geometry without starting a network fetch in the test
        assert!(crate::tui_ui::paragraph_action_at(
            lines,
            notice_area,
            p,
            false
        ));
        let p = rendered_position(&mut app, area, "LOCALMAXXING");
        assert!(!click_at(&mut app, area, p.x, p.y));
        assert!(!app.bench_loading);
        app.open_bench_hw_picker();
        click_label(&mut app, area, "Esc:");
        assert!(
            !app.bench_hw_picker_open,
            "picker must also render above error notices"
        );
        app.bench_error = None;
        app.bench_loading = true;
        app.open_bench_hw_picker();
        click_label(&mut app, area, "Esc:");
        assert!(
            !app.bench_hw_picker_open,
            "picker must remain usable while loading"
        );
        app.bench_loading = false;
        app.bench_entries = vec![
            serde_json::from_value(serde_json::json!({
                "id": "test", "model": {"hf_id": "test/model"}
            }))
            .expect("leaderboard fixture"),
        ];
        app.bench_search_query = "no match".to_string();
        app.bench_search_active = true;
        click_label(&mut app, area, "Esc:clear the search");
        assert!(app.bench_search_query.is_empty());
        assert!(!app.bench_search_active);
    }

    #[test]
    fn model_viewport_updates_on_events_and_draws_leave_it_unchanged() {
        use ratatui::{Terminal, backend::TestBackend, layout::Rect};

        let mut app = plan_mode_app();
        app.input_mode = InputMode::Normal;
        app.show_plan = false;
        assert!(!app.all_fits.is_empty());
        app.filtered_fits = (0..app.all_fits.len().min(100)).collect();
        app.selected_row = app.filtered_fits.len() - 1;
        let area = Rect::new(0, 0, 160, 24);
        update_model_viewport(&mut app, area);
        let capacity = usize::from(crate::tui_ui::main_layout(area)[2].height - 3);
        assert_eq!(app.table_state.offset(), app.selected_row + 1 - capacity);

        // Resizing changes persistent scrolling only during event handling.
        let resized = Rect::new(0, 0, 160, 20);
        update_model_viewport(&mut app, resized);
        let capacity = usize::from(crate::tui_ui::main_layout(resized)[2].height - 3);
        assert_eq!(app.table_state.offset(), app.selected_row + 1 - capacity);
        let state = app.table_state.clone();
        let selected = app.selected_row;
        let tick = app.tick_count;
        let mut terminal = Terminal::new(TestBackend::new(160, 20)).expect("terminal");
        terminal
            .draw(|f| crate::tui_ui::draw(f, &mut app))
            .expect("first frame");
        let first_frame = terminal.backend().buffer().clone();
        terminal
            .draw(|f| crate::tui_ui::draw(f, &mut app))
            .expect("second frame");
        assert_eq!(terminal.backend().buffer(), &first_frame);
        assert_eq!(app.table_state, state);
        assert_eq!(app.selected_row, selected);
        assert_eq!(app.tick_count, tick);

        handle_normal_mode(&mut app, KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        update_model_viewport(&mut app, resized);
        assert_eq!(app.selected_row, selected - 1);
        assert_eq!(app.table_state.selected(), Some(selected - 1));
        app.filtered_fits.clear();
        app.selected_row = 0;
        update_model_viewport(&mut app, resized);
        assert_eq!(app.table_state.offset(), 0);
        assert_eq!(app.table_state.selected(), None);
        terminal
            .draw(|f| crate::tui_ui::draw(f, &mut app))
            .expect("empty table");
        let footer = crate::tui_ui::main_layout(resized)[3];
        let text = (0..resized.width)
            .map(|x| terminal.backend().buffer()[(x, footer.y)].symbol())
            .collect::<String>();
        assert!(text.contains("No model selected"));
        assert!(crate::tui_ui::status_model_area(&app, footer).is_none());
    }

    #[test]
    fn plan_mode_text_fields_accept_q_j_k() {
        // Regression for #781: 'q' closed the plan screen and 'j'/'k' jumped
        // fields, making quant names like q4_k_m impossible to type.
        let mut app = plan_mode_app();
        app.plan_field = PlanField::KvQuant;
        for c in ['q', '4', '_', 'k', 'j'] {
            handle_plan_mode(&mut app, plain(c));
        }
        assert_eq!(app.input_mode, InputMode::Plan, "plan must stay open");
        assert_eq!(app.plan_field, PlanField::KvQuant, "field must not change");
        assert_eq!(app.plan_kv_quant_input, "q4_kj");
    }

    #[test]
    fn plan_mode_esc_still_closes_and_tab_navigates() {
        let mut app = plan_mode_app();
        app.plan_field = PlanField::Context;
        handle_plan_mode(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.plan_field, PlanField::Quant);
        handle_plan_mode(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(app.input_mode, InputMode::Normal);
    }

    #[test]
    fn search_text_accepts_unmodified_and_shift_modified_input() {
        assert!(allows_search_text_input(KeyModifiers::NONE));
        assert!(allows_search_text_input(KeyModifiers::SHIFT));
    }

    #[test]
    fn search_text_rejects_modified_navigation_artifacts() {
        assert!(!allows_search_text_input(KeyModifiers::ALT));
        assert!(!allows_search_text_input(KeyModifiers::SUPER));
        assert!(!allows_search_text_input(KeyModifiers::CONTROL));
        assert!(!allows_search_text_input(KeyModifiers::META));
        assert!(!allows_search_text_input(KeyModifiers::HYPER));
        assert!(!allows_search_text_input(
            KeyModifiers::ALT | KeyModifiers::SHIFT
        ));
        assert!(!allows_search_text_input(
            KeyModifiers::SUPER | KeyModifiers::SHIFT
        ));
    }

    #[test]
    fn provider_filter_text_accepts_plain_ascii_graphic_chars() {
        assert!(is_plain_provider_filter_char('o', KeyModifiers::NONE));
        assert!(is_plain_provider_filter_char('O', KeyModifiers::SHIFT));
        assert!(is_plain_provider_filter_char('-', KeyModifiers::NONE));
    }

    #[test]
    fn provider_filter_text_rejects_non_ascii_space_and_modified_chars() {
        assert!(!is_plain_provider_filter_char('你', KeyModifiers::NONE));
        assert!(!is_plain_provider_filter_char(' ', KeyModifiers::NONE));
        assert!(!is_plain_provider_filter_char('b', KeyModifiers::ALT));
        assert!(!is_plain_provider_filter_char('f', KeyModifiers::ALT));
        assert!(!is_plain_provider_filter_char('a', KeyModifiers::SUPER));
        assert!(!is_plain_provider_filter_char('e', KeyModifiers::SUPER));
        assert!(!is_plain_provider_filter_char('x', KeyModifiers::CONTROL));
    }
}
