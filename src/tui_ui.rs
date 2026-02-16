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
use crate::hardware::is_running_in_wsl;
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
        let gpu_label = app.specs.gpu_name.as_deref().unwrap_or("Unknown");
        let backend = app.specs.backend.label();
        if app.specs.unified_memory {
            format!(
                "GPU: {} ({:.1} GB shared, {})",
                gpu_label,
                app.specs.gpu_vram_gb.unwrap_or(0.0),
                backend
            )
        } else {
            match app.specs.gpu_vram_gb {
                Some(vram) if vram > 0.0 => {
                    if app.specs.gpu_count > 1 {
                        format!("GPU: {} x{} ({:.1} GB, {})", gpu_label, app.specs.gpu_count, vram, backend)
                    } else {
                        format!("GPU: {} ({:.1} GB, {})", gpu_label, vram, backend)
                    }
                }
                Some(_) => format!("GPU: {} (shared, {})", gpu_label, backend),
                None => format!("GPU: {} ({})", gpu_label, backend),
            }
        }
    } else {
        format!("GPU: none ({})", app.specs.backend.label())
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
                "{:.1} GB avail / {:.1} GB total{}",
                app.specs.available_ram_gb,
                app.specs.total_ram_gb,
                if is_running_in_wsl() { " (WSL)" } else { "" }
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
        "", "Model", "Provider", "Params", "Score", "tok/s", "Quant", "Mode", "Mem %", "Ctx", "Fit", "Use Case",
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

            let mode_color = match fit.run_mode {
                crate::fit::RunMode::Gpu => Color::Green,
                crate::fit::RunMode::MoeOffload => Color::Cyan,
                crate::fit::RunMode::CpuOffload => Color::Yellow,
                crate::fit::RunMode::CpuOnly => Color::DarkGray,
            };

            let score_color = if fit.score >= 70.0 {
                Color::Green
            } else if fit.score >= 50.0 {
                Color::Yellow
            } else {
                Color::Red
            };

            let tps_text = if fit.estimated_tps >= 100.0 {
                format!("{:.0}", fit.estimated_tps)
            } else if fit.estimated_tps >= 10.0 {
                format!("{:.1}", fit.estimated_tps)
            } else {
                format!("{:.1}", fit.estimated_tps)
            };

            Row::new(vec![
                Cell::from(fit_indicator(fit.fit_level)).style(Style::default().fg(color)),
                Cell::from(fit.model.name.clone()).style(Style::default().fg(Color::White)),
                Cell::from(fit.model.provider.clone())
                    .style(Style::default().fg(Color::DarkGray)),
                Cell::from(fit.model.parameter_count.clone())
                    .style(Style::default().fg(Color::White)),
                Cell::from(format!("{:.0}", fit.score))
                    .style(Style::default().fg(score_color)),
                Cell::from(tps_text)
                    .style(Style::default().fg(Color::White)),
                Cell::from(fit.best_quant.clone())
                    .style(Style::default().fg(Color::DarkGray)),
                Cell::from(fit.run_mode_text().to_string())
                    .style(Style::default().fg(mode_color)),
                Cell::from(format!("{:.0}%", fit.utilization_pct))
                    .style(Style::default().fg(color)),
                Cell::from(format!("{}k", fit.model.context_length / 1000))
                    .style(Style::default().fg(Color::DarkGray)),
                Cell::from(fit.fit_text().to_string()).style(Style::default().fg(color)),
                Cell::from(fit.use_case.label().to_string())
                    .style(Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(2),  // indicator
        Constraint::Min(20),    // model name
        Constraint::Length(12), // provider
        Constraint::Length(8),  // params
        Constraint::Length(6),  // score
        Constraint::Length(6),  // tok/s
        Constraint::Length(7),  // quant
        Constraint::Length(7),  // mode
        Constraint::Length(6),  // mem %
        Constraint::Length(5),  // ctx
        Constraint::Length(10), // fit
        Constraint::Min(10),   // use case
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
            Span::styled("  Best Quant:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!(" {} (for this hardware)", fit.best_quant),
                Style::default().fg(Color::Green),
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
        Line::from(vec![
            Span::styled("  Category:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(fit.use_case.label(), Style::default().fg(Color::Cyan)),
        ]),
    ];

    // Scoring section
    let score_color = if fit.score >= 70.0 {
        Color::Green
    } else if fit.score >= 50.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    lines.extend_from_slice(&[
        Line::from(""),
        Line::from(Span::styled(
            "  ── Score Breakdown ──",
            Style::default().fg(Color::Cyan),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Overall:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1} / 100", fit.score),
                Style::default().fg(score_color).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Quality:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", fit.score_components.quality),
                Style::default().fg(Color::White),
            ),
            Span::styled("  Speed: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", fit.score_components.speed),
                Style::default().fg(Color::White),
            ),
            Span::styled("  Fit: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", fit.score_components.fit),
                Style::default().fg(Color::White),
            ),
            Span::styled("  Context: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", fit.score_components.context),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Est. Speed:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1} tok/s", fit.estimated_tps),
                Style::default().fg(Color::White),
            ),
        ]),
    ]);

    // MoE Architecture section
    if fit.model.is_moe {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  ── MoE Architecture ──",
            Style::default().fg(Color::Cyan),
        )));
        lines.push(Line::from(""));

        if let (Some(num_experts), Some(active_experts)) =
            (fit.model.num_experts, fit.model.active_experts)
        {
            lines.push(Line::from(vec![
                Span::styled("  Experts:     ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{} active / {} total per token", active_experts, num_experts),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
        }

        if let Some(active_vram) = fit.model.moe_active_vram_gb() {
            lines.push(Line::from(vec![
                Span::styled("  Active VRAM: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1} GB", active_vram),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(
                    format!("  (vs {:.1} GB full model)", fit.model.min_vram_gb.unwrap_or(0.0)),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
        }

        if let Some(offloaded) = fit.moe_offloaded_gb {
            lines.push(Line::from(vec![
                Span::styled("  Offloaded:   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1} GB inactive experts in RAM", offloaded),
                    Style::default().fg(Color::Yellow),
                ),
            ]));
        }

        if fit.run_mode == crate::fit::RunMode::MoeOffload {
            lines.push(Line::from(vec![
                Span::styled("  Strategy:    ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Expert offloading (active in VRAM, inactive in RAM)",
                    Style::default().fg(Color::Green),
                ),
            ]));
        } else if fit.run_mode == crate::fit::RunMode::Gpu {
            lines.push(Line::from(vec![
                Span::styled("  Strategy:    ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "All experts loaded in VRAM (optimal)",
                    Style::default().fg(Color::Green),
                ),
            ]));
        }
    }

    lines.extend_from_slice(&[
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
    ]);

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
