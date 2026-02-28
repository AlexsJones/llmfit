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
            InputMode::Search => handle_search_mode(app, key),
            InputMode::ProviderPopup => handle_provider_popup_mode(app, key),
            InputMode::FilterPopup => handle_filter_popup_mode(app, key),
        }
        return Ok(true);
    }
    Ok(false)
}

fn handle_normal_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        // Quit
        KeyCode::Char('q') | KeyCode::Esc => {
            if app.confirm_download {
                app.confirm_download = false;
                app.pull_status = Some("Download cancelled".to_string());
            } else if app.show_detail {
                app.show_detail = false;
            } else {
                app.should_quit = true;
            }
        }

        // Navigation
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => app.half_page_up(),
        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => app.half_page_down(),
        KeyCode::Up | KeyCode::Char('k') => app.move_up(),
        KeyCode::Down | KeyCode::Char('j') => app.move_down(),
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        KeyCode::Home | KeyCode::Char('g') => app.home(),
        KeyCode::End | KeyCode::Char('G') => app.end(),

        // Search
        KeyCode::Char('/') => app.enter_search(),

        // Fit filter
        KeyCode::Char('f') => app.cycle_fit_filter(),

        // Column filters popup
        KeyCode::Char('F') => app.open_filter_popup(),

        // Sort column
        KeyCode::Char('s') => app.cycle_sort_column(),

        // Theme
        KeyCode::Char('t') => app.cycle_theme(),

        // Provider popup
        KeyCode::Char('p') => app.open_provider_popup(),

        // Installed-first sort toggle (any provider)
        KeyCode::Char('i') if app.ollama_available || app.mlx_available => {
            app.toggle_installed_first()
        }

        // Download model via best provider (requires confirmation)
        KeyCode::Char('d') if app.ollama_available || app.mlx_available => {
            if app.confirm_download {
                // Second press: confirmed, start the download
                app.confirm_download = false;
                app.start_download();
            } else if app.pull_active.is_none() {
                // First press: show confirmation prompt
                if let Some(fit) = app.selected_fit() {
                    if fit.installed {
                        app.pull_status = Some("Already installed".to_string());
                    } else {
                        let size_est = fit.model.params_b() * 0.5; // rough Q4 estimate in GB
                        app.pull_status = Some(format!(
                            "Download {}? (~{:.1} GB) Press 'd' to confirm, Esc to cancel",
                            fit.model.name, size_est
                        ));
                        app.confirm_download = true;
                    }
                }
            }
        }

        // Refresh installed models
        KeyCode::Char('r') if app.ollama_available || app.mlx_available => app.refresh_installed(),

        // Detail view
        KeyCode::Enter => app.toggle_detail(),

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

fn handle_filter_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('F') | KeyCode::Char('q') => app.close_filter_popup(),
        KeyCode::Up | KeyCode::Char('k') => app.filter_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.filter_popup_down(),
        KeyCode::Right | KeyCode::Char('l') => app.filter_popup_adjust_right(),
        KeyCode::Left | KeyCode::Char('h') => app.filter_popup_adjust_left(),
        KeyCode::Char('r') => app.filter_popup_reset_current(),
        KeyCode::Char('R') => app.filter_popup_reset_all(),
        _ => {}
    }
}

fn handle_provider_popup_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Char('p') | KeyCode::Char('q') => app.close_provider_popup(),

        KeyCode::Up | KeyCode::Char('k') => app.provider_popup_up(),
        KeyCode::Down | KeyCode::Char('j') => app.provider_popup_down(),

        KeyCode::Char(' ') | KeyCode::Enter => app.provider_popup_toggle(),

        KeyCode::Char('a') => app.provider_popup_select_all(),

        _ => {}
    }
}
