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
        }
        return Ok(true);
    }
    Ok(false)
}

fn handle_normal_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        // Quit
        KeyCode::Char('q') | KeyCode::Esc => {
            if app.show_detail {
                app.show_detail = false;
            } else {
                app.should_quit = true;
            }
        }

        // Navigation
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

        // Provider popup
        KeyCode::Char('p') => app.open_provider_popup(),

        // Installed-first sort toggle (Ollama only)
        KeyCode::Char('i') if app.ollama_available => app.toggle_installed_first(),

        // Download model via Ollama
        KeyCode::Char('d') if app.ollama_available => app.start_download(),

        // Refresh installed models
        KeyCode::Char('r') if app.ollama_available => app.refresh_installed(),

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
