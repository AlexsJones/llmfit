use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{
        Block, Borders, Cell, Paragraph, Row, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Table, TableState, Wrap,
    },
    Frame,
};

use crate::fit::FitLevel;
use crate::tui_app::{App, FitFilter, InputMode};

pub fn draw(frame: &mut Frame, app: &mut App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // system info bar
            Constraint::Length(3), // search + filters
            Constraint::Min(10),  // main table
            Constraint::Length(1), // status bar
        ])
        .split(frame.area());

    draw_system_bar(frame, app, outer[0]);
    draw_search_and_filters(frame, app, outer[1]);

    if app.show_detail {
        draw_detail(frame, app, outer[2]);
    } else {
        draw_table(frame, app, outer[2]);
    }

    draw_status_bar(frame, app, outer[3]);
}

fn draw_system_bar(frame: &mut Frame, app: &App, area: Rect) {
    let gpu_info = if app.specs.has_gpu {
        if app.specs.unified_memory {
            format!(
                "GPU: Apple Silicon ({:.1} GB shared)",
                app.specs.gpu_vram_gb.unwrap_or(0.0)
            )
        } else {
            match app.specs.gpu_vram_gb {
                Some(vram) if vram > 0.0 => format!("GPU: {:.1} GB VRAM", vram),
                Some(_) => "GPU: Intel Arc (shared memory)".to_string(),
                None => "GPU: detected".to_string(),
            }
        }
    } else {
        "GPU: none".to_string()
    };

    let text = Line::from(vec![
        Span::styled(" CPU: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{} ({} cores)", app.specs.cpu_name, app.specs.total_cpu_cores),
            Style::default().fg(Color::White),
        ),
        Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
        Span::styled("RAM: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!(
                "{:.1} GB avail / {:.1} GB total",
                app.specs.available_ram_gb, app.specs.total_ram_gb
            ),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
        Span::styled(gpu_info, Style::default().fg(Color::Yellow)),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" llmfit ")
        .title_style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD));

    let paragraph = Paragraph::new(text).block(block);
    frame.render_widget(paragraph, area);
}

fn draw_search_and_filters(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(30),        // search
            Constraint::Length(50),      // provider filters
            Constraint::Length(20),      // fit filter
        ])
        .split(area);

    // Search box
    let search_style = match app.input_mode {
        InputMode::Search => Style::default().fg(Color::Yellow),
        InputMode::Normal => Style::default().fg(Color::DarkGray),
    };

    let search_text = if app.search_query.is_empty() && app.input_mode == InputMode::Normal {
        Line::from(Span::styled(
            "Press / to search...",
            Style::default().fg(Color::DarkGray),
        ))
    } else {
        Line::from(Span::styled(
            &app.search_query,
            Style::default().fg(Color::White),
        ))
    };

    let search_block = Block::default()
        .borders(Borders::ALL)
        .border_style(search_style)
        .title(" Search ")
        .title_style(search_style);

    let search = Paragraph::new(search_text).block(search_block);
    frame.render_widget(search, chunks[0]);

    if app.input_mode == InputMode::Search {
        frame.set_cursor_position((
            chunks[0].x + app.cursor_position as u16 + 1,
            chunks[0].y + 1,
        ));
    }

    // Provider filters  
    let mut provider_spans: Vec<Span> = Vec::new();
    for (i, provider) in app.providers.iter().enumerate() {
        if i > 0 {
            provider_spans.push(Span::styled(" ", Style::default()));
        }
        let (label, style) = if app.selected_providers[i] {
            (
                format!("[{}:{}]", i + 1, provider),
                Style::default().fg(Color::Green),
            )
        } else {
            (
                format!("[{}:{}]", i + 1, provider),
                Style::default().fg(Color::DarkGray),
            )
        };
        provider_spans.push(Span::styled(label, style));
    }

    let provider_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Providers ")
        .title_style(Style::default().fg(Color::DarkGray));

    let providers = Paragraph::new(Line::from(provider_spans)).block(provider_block);
    frame.render_widget(providers, chunks[1]);

    // Fit filter
    let fit_style = match app.fit_filter {
        FitFilter::All => Style::default().fg(Color::White),
        FitFilter::Runnable => Style::default().fg(Color::Green),
        FitFilter::Perfect => Style::default().fg(Color::Green),
        FitFilter::Good => Style::default().fg(Color::Yellow),
        FitFilter::Marginal => Style::default().fg(Color::Magenta),
    };

    let fit_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Fit [f] ")
        .title_style(Style::default().fg(Color::DarkGray));

    let fit_text = Paragraph::new(Line::from(Span::styled(
        app.fit_filter.label(),
        fit_style,
    )))
    .block(fit_block);
    frame.render_widget(fit_text, chunks[2]);
}

fn fit_color(level: FitLevel) -> Color {
    match level {
        FitLevel::Perfect => Color::Green,
        FitLevel::Good => Color::Yellow,
        FitLevel::Marginal => Color::Magenta,
        FitLevel::TooTight => Color::Red,
    }
}

fn fit_indicator(level: FitLevel) -> &'static str {
    match level {
        FitLevel::Perfect => "●",
        FitLevel::Good => "●",
        FitLevel::Marginal => "●",
        FitLevel::TooTight => "●",
    }
}

fn draw_table(frame: &mut Frame, app: &mut App, area: Rect) {
    let header_cells = [
        "", "Model", "Provider", "Params", "VRAM", "RAM", "Mode", "Mem %", "Ctx", "Fit", "Use Case",
    ]
    .iter()
    .map(|h| {
        Cell::from(*h).style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
    });
    let header = Row::new(header_cells).height(1);

    let rows: Vec<Row> = app
        .filtered_fits
        .iter()
        .map(|&idx| {
            let fit = &app.all_fits[idx];
            let color = fit_color(fit.fit_level);

            let vram_text = fit
                .model
                .min_vram_gb
                .map(|v| format!("{:.1} GB", v))
                .unwrap_or_else(|| "-".to_string());

            let mode_color = match fit.run_mode {
                crate::fit::RunMode::Gpu => Color::Green,
                crate::fit::RunMode::CpuOffload => Color::Yellow,
                crate::fit::RunMode::CpuOnly => Color::DarkGray,
            };

            Row::new(vec![
                Cell::from(fit_indicator(fit.fit_level)).style(Style::default().fg(color)),
                Cell::from(fit.model.name.clone()).style(Style::default().fg(Color::White)),
                Cell::from(fit.model.provider.clone())
                    .style(Style::default().fg(Color::DarkGray)),
                Cell::from(fit.model.parameter_count.clone())
                    .style(Style::default().fg(Color::White)),
                Cell::from(vram_text)
                    .style(Style::default().fg(Color::White)),
                Cell::from(format!("{:.1} GB", fit.model.min_ram_gb))
                    .style(Style::default().fg(Color::White)),
                Cell::from(fit.run_mode_text().to_string())
                    .style(Style::default().fg(mode_color)),
                Cell::from(format!("{:.0}%", fit.utilization_pct))
                    .style(Style::default().fg(color)),
                Cell::from(format!("{}k", fit.model.context_length / 1000))
                    .style(Style::default().fg(Color::DarkGray)),
                Cell::from(fit.fit_text().to_string()).style(Style::default().fg(color)),
                Cell::from(truncate_str(&fit.model.use_case, 30))
                    .style(Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(2),  // indicator
        Constraint::Min(20),    // model name
        Constraint::Length(12), // provider
        Constraint::Length(8),  // params
        Constraint::Length(9),  // vram
        Constraint::Length(9),  // ram
        Constraint::Length(7),  // mode
        Constraint::Length(6),  // mem %
        Constraint::Length(5),  // ctx
        Constraint::Length(10), // fit
        Constraint::Min(12),   // use case
    ];

    let count_text = format!(
        " Models ({}/{}) ",
        app.filtered_fits.len(),
        app.all_fits.len()
    );

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(count_text)
                .title_style(Style::default().fg(Color::White)),
        )
        .row_highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    let mut state = TableState::default();
    if !app.filtered_fits.is_empty() {
        state.select(Some(app.selected_row));
    }

    frame.render_stateful_widget(table, area, &mut state);

    // Scrollbar
    if app.filtered_fits.len() > (area.height as usize).saturating_sub(3) {
        let mut scrollbar_state = ScrollbarState::new(app.filtered_fits.len())
            .position(app.selected_row);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓")),
            area,
            &mut scrollbar_state,
        );
    }
}

fn draw_detail(frame: &mut Frame, app: &App, area: Rect) {
    let fit = match app.selected_fit() {
        Some(f) => f,
        None => {
            let block = Block::default()
                .borders(Borders::ALL)
                .title(" No model selected ");
            frame.render_widget(block, area);
            return;
        }
    };

    let color = fit_color(fit.fit_level);

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Model:       ", Style::default().fg(Color::DarkGray)),
            Span::styled(&fit.model.name, Style::default().fg(Color::White).bold()),
        ]),
        Line::from(vec![
            Span::styled("  Provider:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(&fit.model.provider, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Parameters:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                &fit.model.parameter_count,
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Quantization:", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!(" {}", fit.model.quantization),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Context:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{} tokens", fit.model.context_length),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Use Case:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(&fit.model.use_case, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  ── System Fit ──",
            Style::default().fg(Color::Cyan),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Fit Level:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{} {}", fit_indicator(fit.fit_level), fit.fit_text()),
                Style::default().fg(color).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Run Mode:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                fit.run_mode_text(),
                Style::default().fg(Color::White).bold(),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  -- Memory --",
            Style::default().fg(Color::Cyan),
        )),
        Line::from(""),
    ];

    if let Some(vram) = fit.model.min_vram_gb {
        let vram_label = if app.specs.has_gpu {
            if app.specs.unified_memory {
                if let Some(sys_vram) = app.specs.gpu_vram_gb {
                    format!("  (shared: {:.1} GB)", sys_vram)
                } else {
                    "  (shared memory)".to_string()
                }
            } else if let Some(sys_vram) = app.specs.gpu_vram_gb {
                format!("  (system: {:.1} GB)", sys_vram)
            } else {
                "  (system: unknown)".to_string()
            }
        } else {
            "  (no GPU)".to_string()
        };
        lines.push(Line::from(vec![
            Span::styled("  Min VRAM:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1} GB", vram),
                Style::default().fg(Color::White),
            ),
            Span::styled(vram_label, Style::default().fg(Color::DarkGray)),
        ]));
    }

    lines.extend_from_slice(&[
        Line::from(vec![
            Span::styled("  Min RAM:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1} GB", fit.model.min_ram_gb),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                format!("  (system: {:.1} GB avail)", app.specs.available_ram_gb),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Rec RAM:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1} GB", fit.model.recommended_ram_gb),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Mem Usage:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}%", fit.utilization_pct),
                Style::default().fg(color),
            ),
            Span::styled(
                format!("  ({:.1} / {:.1} GB)", fit.memory_required_gb, fit.memory_available_gb),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ]);

    lines.push(Line::from(""));
    if !fit.notes.is_empty() {
        lines.push(Line::from(Span::styled(
            "  ── Notes ──",
            Style::default().fg(Color::Cyan),
        )));
        lines.push(Line::from(""));
        for note in &fit.notes {
            lines.push(Line::from(Span::styled(
                format!("  {}", note),
                Style::default().fg(Color::White),
            )));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(format!(" {} ", fit.model.name))
        .title_style(Style::default().fg(Color::White).bold());

    let paragraph = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let (keys, mode_text) = match app.input_mode {
        InputMode::Normal => {
            let detail_key = if app.show_detail { "Enter:table" } else { "Enter:detail" };
            (
                format!(
                    " ↑↓/jk:navigate  {}  /:search  f:fit filter  1-{}:providers  q:quit",
                    detail_key,
                    app.providers.len()
                ),
                "NORMAL",
            )
        }
        InputMode::Search => ("  Type to search  Esc:done  Ctrl-U:clear".to_string(), "SEARCH"),
    };

    let status_line = Line::from(vec![
        Span::styled(
            format!(" {} ", mode_text),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Green)
                .bold(),
        ),
        Span::styled(keys, Style::default().fg(Color::DarkGray)),
    ]);

    frame.render_widget(Paragraph::new(status_line), area);
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}
