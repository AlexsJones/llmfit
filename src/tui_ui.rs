use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{
        Block, Borders, Cell, Clear, Paragraph, Row, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Table, TableState, Wrap,
    },
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
            Constraint::Min(10),   // main table
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

    // Draw provider popup on top if active
    if app.input_mode == InputMode::ProviderPopup {
        draw_provider_popup(frame, app);
    }
}

fn draw_system_bar(frame: &mut Frame, app: &App, area: Rect) {
    let gpu_info = if app.specs.gpus.is_empty() {
        format!("GPU: none ({})", app.specs.backend.label())
    } else {
        // Show the primary GPU (best VRAM, used for fit scoring) in full.
        // If there are additional GPUs, append "+N more".
        let primary = &app.specs.gpus[0];
        let backend = primary.backend.label();
        let primary_str = if primary.unified_memory {
            format!(
                "{} ({:.1} GB shared, {})",
                primary.name,
                primary.vram_gb.unwrap_or(0.0),
                backend
            )
        } else {
            match primary.vram_gb {
                Some(vram) if vram > 0.0 => {
                    if primary.count > 1 {
                        format!(
                            "{} x{} ({:.1} GB, {})",
                            primary.name, primary.count, vram, backend
                        )
                    } else {
                        format!("{} ({:.1} GB, {})", primary.name, vram, backend)
                    }
                }
                Some(_) => format!("{} (shared, {})", primary.name, backend),
                None => format!("{} ({})", primary.name, backend),
            }
        };
        let extra = app.specs.gpus.len() - 1;
        if extra > 0 {
            format!("GPU: {} +{} more", primary_str, extra)
        } else {
            format!("GPU: {}", primary_str)
        }
    };

    let ollama_info = if app.ollama_available {
        format!("Ollama: ✓ ({} installed)", app.ollama_installed.len() / 2)
    } else {
        "Ollama: ✗".to_string()
    };
    let ollama_color = if app.ollama_available { Color::Green } else { Color::DarkGray };

    let text = Line::from(vec![
        Span::styled(" CPU: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!(
                "{} ({} cores)",
                app.specs.cpu_name, app.specs.total_cpu_cores
            ),
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
        Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
        Span::styled(ollama_info, Style::default().fg(ollama_color)),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" llmfit ")
        .title_style(
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(text).block(block);
    frame.render_widget(paragraph, area);
}

fn draw_search_and_filters(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(30),    // search
            Constraint::Length(24), // provider summary
            Constraint::Length(20), // fit filter
        ])
        .split(area);

    // Search box
    let search_style = match app.input_mode {
        InputMode::Search => Style::default().fg(Color::Yellow),
        InputMode::Normal | InputMode::ProviderPopup => Style::default().fg(Color::DarkGray),
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

    // Provider filter summary (press 'p' to open popup)
    let active_count = app.selected_providers.iter().filter(|&&s| s).count();
    let total_count = app.providers.len();
    let provider_text = if active_count == total_count {
        "All".to_string()
    } else {
        format!("{}/{}", active_count, total_count)
    };
    let provider_color = if active_count == total_count {
        Color::Green
    } else if active_count == 0 {
        Color::Red
    } else {
        Color::Yellow
    };

    let provider_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Providers (p) ")
        .title_style(Style::default().fg(Color::DarkGray));

    let providers = Paragraph::new(Line::from(Span::styled(
        format!(" {}", provider_text),
        Style::default().fg(provider_color),
    )))
    .block(provider_block);
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

    let fit_text = Paragraph::new(Line::from(Span::styled(app.fit_filter.label(), fit_style)))
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

/// Build a compact animated download indicator for the "Inst" column.
/// Shows a block-character progress bar when percentage is known,
/// or an animated spinner when waiting.
fn pull_indicator(percent: Option<f64>, tick: u64) -> String {
    const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let spin = SPINNER[(tick as usize / 3) % SPINNER.len()];

    match percent {
        Some(pct) => {
            // 3-char block bar: each char = ~33%, using ░▒▓█
            const BLOCKS: &[char] = &[' ', '░', '▒', '▓', '█'];
            let filled = pct / 100.0 * 3.0; // 0..3
            let mut bar = String::with_capacity(5);
            bar.push(spin);
            for i in 0..3 {
                let level = (filled - i as f64).clamp(0.0, 1.0);
                let idx = (level * 4.0).round() as usize;
                bar.push(BLOCKS[idx]);
            }
            bar
        }
        None => format!(" {} ", spin),
    }
}

fn draw_table(frame: &mut Frame, app: &mut App, area: Rect) {
    let header_cells = [
        "", "Inst", "Model", "Provider", "Params", "Score", "tok/s", "Quant", "Mode", "Mem %", "Ctx",
        "Fit", "Use Case",
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

            let is_pulling = app.pull_active.is_some()
                && app.pull_model_name.as_deref() == Some(&fit.model.name);

            let installed_icon = if fit.installed {
                " ✓".to_string()
            } else if is_pulling {
                pull_indicator(app.pull_percent, app.tick_count)
            } else {
                " ".to_string()
            };
            let installed_color = if fit.installed {
                Color::Green
            } else if is_pulling {
                Color::Yellow
            } else {
                Color::DarkGray
            };

            let row_style = if is_pulling {
                Style::default().bg(Color::Rgb(50, 50, 0))
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(fit_indicator(fit.fit_level)).style(Style::default().fg(color)),
                Cell::from(installed_icon).style(Style::default().fg(installed_color)),
                Cell::from(fit.model.name.clone()).style(Style::default().fg(Color::White)),
                Cell::from(fit.model.provider.clone()).style(Style::default().fg(Color::DarkGray)),
                Cell::from(fit.model.parameter_count.clone())
                    .style(Style::default().fg(Color::White)),
                Cell::from(format!("{:.0}", fit.score)).style(Style::default().fg(score_color)),
                Cell::from(tps_text).style(Style::default().fg(Color::White)),
                Cell::from(fit.best_quant.clone()).style(Style::default().fg(Color::DarkGray)),
                Cell::from(fit.run_mode_text().to_string()).style(Style::default().fg(mode_color)),
                Cell::from(format!("{:.0}%", fit.utilization_pct))
                    .style(Style::default().fg(color)),
                Cell::from(format!("{}k", fit.model.context_length / 1000))
                    .style(Style::default().fg(Color::DarkGray)),
                Cell::from(fit.fit_text().to_string()).style(Style::default().fg(color)),
                Cell::from(fit.use_case.label().to_string())
                    .style(Style::default().fg(Color::DarkGray)),
            ])
            .style(row_style)
        })
        .collect();

    let widths = [
        Constraint::Length(2),  // indicator
        Constraint::Length(5),  // installed / pull %
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
        Constraint::Min(10),    // use case
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
        let mut scrollbar_state =
            ScrollbarState::new(app.filtered_fits.len()).position(app.selected_row);
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
        Line::from(vec![
            Span::styled("  Installed:   ", Style::default().fg(Color::DarkGray)),
            if fit.installed {
                Span::styled("✓ Yes (Ollama)", Style::default().fg(Color::Green).bold())
            } else if app.ollama_available {
                Span::styled("✗ No  (press d to pull)", Style::default().fg(Color::DarkGray))
            } else {
                Span::styled("- Ollama not running", Style::default().fg(Color::DarkGray))
            },
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
                    format!(
                        "{} active / {} total per token",
                        active_experts, num_experts
                    ),
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
                    format!(
                        "  (vs {:.1} GB full model)",
                        fit.model.min_vram_gb.unwrap_or(0.0)
                    ),
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
            Span::styled(format!("{:.1} GB", vram), Style::default().fg(Color::White)),
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
                format!(
                    "  ({:.1} / {:.1} GB)",
                    fit.memory_required_gb, fit.memory_available_gb
                ),
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

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn draw_provider_popup(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Size the popup: width fits longest provider name, height fits up to 20 rows
    let max_name_len = app.providers.iter().map(|p| p.len()).max().unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4)); // "[x] name" + padding
    let popup_height = (app.providers.len() as u16 + 2).min(area.height.saturating_sub(4)); // +2 for border

    // Center the popup
    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    // Clear the area behind the popup
    frame.render_widget(Clear, popup_area);

    // Build the list of provider lines
    let inner_height = popup_height.saturating_sub(2) as usize; // minus borders
    let total = app.providers.len();

    // Scroll so the cursor is always visible
    let scroll_offset = if app.provider_cursor >= inner_height {
        app.provider_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .providers
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_providers[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.provider_cursor;

            let style = if is_cursor {
                if app.selected_providers[i] {
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD)
                        .bg(Color::DarkGray)
                } else {
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD)
                        .bg(Color::DarkGray)
                }
            } else if app.selected_providers[i] {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::DarkGray)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_providers.iter().filter(|&&s| s).count();
    let title = format!(" Providers ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(title)
        .title_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    // If a download is in progress, show the progress bar
    if let Some(status) = &app.pull_status {
        let progress_text = if let Some(pct) = app.pull_percent {
            format!(" {} [{:.0}%] ", status, pct)
        } else {
            format!(" {} ", status)
        };

        let (keys, mode_text) = match app.input_mode {
            InputMode::Normal => {
                let detail_key = if app.show_detail { "Enter:table" } else { "Enter:detail" };
                let ollama_keys = if app.ollama_available {
                    let installed_key = if app.installed_first { "i:all" } else { "i:installed↑" };
                    format!("  {}  d:pull  r:refresh", installed_key)
                } else {
                    String::new()
                };
                (
                    format!(
                        " ↑↓/jk:nav  {}  /:search  f:fit{}  p:providers  q:quit",
                        detail_key,
                        ollama_keys,
                    ),
                    "NORMAL",
                )
            }
            InputMode::Search => ("  Type to search  Esc:done  Ctrl-U:clear".to_string(), "SEARCH"),
            InputMode::ProviderPopup => (
                "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
                "PROVIDERS",
            ),
        };

        // Split into two lines: keys + progress
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(20), Constraint::Length(progress_text.len() as u16 + 2)])
            .split(area);

        let status_line = Line::from(vec![
            Span::styled(
                format!(" {} ", mode_text),
                Style::default().fg(Color::Black).bg(Color::Green).bold(),
            ),
            Span::styled(keys, Style::default().fg(Color::DarkGray)),
        ]);
        frame.render_widget(Paragraph::new(status_line), chunks[0]);

        let pull_color = if app.pull_active.is_some() { Color::Yellow } else { Color::Green };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(progress_text, Style::default().fg(pull_color)))),
            chunks[1],
        );
        return;
    }

    let (keys, mode_text) = match app.input_mode {
        InputMode::Normal => {
            let detail_key = if app.show_detail {
                "Enter:table"
            } else {
                "Enter:detail"
            };
            let ollama_keys = if app.ollama_available {
                let installed_key = if app.installed_first { "i:all" } else { "i:installed↑" };
                format!("  {}  d:pull  r:refresh", installed_key)
            } else {
                String::new()
            };
            (
                format!(
                    " ↑↓/jk:nav  {}  /:search  f:fit{}  p:providers  q:quit",
                    detail_key,
                    ollama_keys,
                ),
                "NORMAL",
            )
        }
        InputMode::Search => (
            "  Type to search  Esc:done  Ctrl-U:clear".to_string(),
            "SEARCH",
        ),
        InputMode::ProviderPopup => (
            "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
            "PROVIDERS",
        ),
    };

    let status_line = Line::from(vec![
        Span::styled(
            format!(" {} ", mode_text),
            Style::default().fg(Color::Black).bg(Color::Green).bold(),
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
