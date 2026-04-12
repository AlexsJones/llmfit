use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::time::Duration;

use crate::tui_app::{App, InputMode};

/// Poll for and handle events. Returns true if an event was processed.
pub fn handle_events(app: &mut App) -> std::io::Result<bool> {
    // Always tick the pull progress (non-blocking)
    app.tick_pull();

    if event::poll(Duration::from_millis(50))?
        && let Event::Key(key) = event::read()?
    {
        // Only handle Press events (ignore Release on some platforms)
        if key.kind != KeyEventKind::Press {
            return Ok(false);
        }
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
        }
        return Ok(true);
    }
    Ok(false)
}

fn handle_normal_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        // Quit
        KeyCode::Char('q') | KeyCode::Esc => {
            if app.show_multi_compare {
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
        KeyCode::Home | KeyCode::Char('g') => app.home(),
        KeyCode::End | KeyCode::Char('G') => app.end(),

        // Visual mode
        KeyCode::Char('v') => app.enter_visual_mode(),

        // Select mode
        KeyCode::Char('V') => app.enter_select_mode(),

        // Search
        KeyCode::Char('/') => app.enter_search(),

        // Fit filter
        KeyCode::Char('f') => app.cycle_fit_filter(),

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
                || app.lmstudio_available =>
        {
            app.toggle_installed_first()
        }

        // Download model via best provider (requires confirmation)
        KeyCode::Char('d')
            if app.ollama_available
                || app.mlx_available
                || app.llamacpp_available
                || app.lmstudio_available =>
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
                || app.lmstudio_available =>
        {
            app.refresh_installed()
        }

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
        KeyCode::Home | KeyCode::Char('g') => app.home(),
        KeyCode::End | KeyCode::Char('G') => app.end(),

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

        KeyCode::Char(c) => app.search_input(c),

        // Allow navigation while searching
        KeyCode::Up => app.move_up(),
        KeyCode::Down => app.move_down(),

        _ => {}
    }
}

fn handle_provider_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('P') | KeyCode::Char('q') => app.close_provider_popup(),

        KeyCode::Up | KeyCode::Char('k') => app.provider_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.provider_popup_down(),

        KeyCode::Char(' ') | KeyCode::Enter => app.provider_popup_toggle(),

        KeyCode::Char('a') => app.provider_popup_select_all(),
        KeyCode::Char('c') => app.provider_popup_clear_all(),

        _ => {}
    }
}

fn handle_plan_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => app.close_plan_mode(),
        KeyCode::Tab | KeyCode::Down | KeyCode::Char('j') => app.plan_next_field(),
        KeyCode::BackTab | KeyCode::Up | KeyCode::Char('k') => app.plan_prev_field(),
        KeyCode::Left => app.plan_cursor_left(),
        KeyCode::Right => app.plan_cursor_right(),
        KeyCode::Backspace => app.plan_backspace(),
        KeyCode::Delete => app.plan_delete(),
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.plan_clear_field()
        }
        KeyCode::Char(c) => app.plan_input(c),
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
