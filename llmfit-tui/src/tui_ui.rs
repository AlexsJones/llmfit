use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Cell, Clear, Paragraph, Row, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Table, TableState, Wrap,
    },
};

use crate::theme::ThemeColors;
use crate::tui_app::{App, FitFilter, InputMode};
use llmfit_core::fit::FitLevel;
use llmfit_core::fit::SortColumn;
use llmfit_core::hardware::is_running_in_wsl;
use llmfit_core::providers;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let tc = app.theme.colors();

    // Fill background if theme specifies one
    if tc.bg != Color::Reset {
        let bg_block = Block::default().style(Style::default().bg(tc.bg));
        frame.render_widget(bg_block, frame.area());
    }

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // system info bar
            Constraint::Length(3), // search + filters
            Constraint::Min(10),   // main table
            Constraint::Length(1), // status bar
        ])
        .split(frame.area());

    draw_system_bar(frame, app, outer[0], &tc);
    draw_search_and_filters(frame, app, outer[1], &tc);

    if app.show_detail {
        draw_detail(frame, app, outer[2], &tc);
    } else {
        draw_table(frame, app, outer[2], &tc);
    }

    draw_status_bar(frame, app, outer[3], &tc);

    // Draw provider popup on top if active
    if app.input_mode == InputMode::ProviderPopup {
        draw_provider_popup(frame, app, &tc);
    }
}

fn draw_system_bar(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let gpu_info = if app.specs.gpus.is_empty() {
        format!("GPU: none ({})", app.specs.backend.label())
    } else {
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
                        let total_vram = vram * primary.count as f64;
                        format!(
                            "{} x{} ({:.1} GB each = {:.0} GB total, {})",
                            primary.name, primary.count, vram, total_vram, backend
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
    let ollama_color = if app.ollama_available {
        tc.good
    } else {
        tc.muted
    };

    let mlx_info = if app.mlx_available {
        format!("MLX: ✓ ({} installed)", app.mlx_installed.len())
    } else if !app.mlx_installed.is_empty() {
        format!("MLX: ({} cached)", app.mlx_installed.len())
    } else {
        "MLX: ✗".to_string()
    };
    let mlx_color = if app.mlx_available {
        tc.good
    } else if !app.mlx_installed.is_empty() {
        tc.warning
    } else {
        tc.muted
    };

    let text = Line::from(vec![
        Span::styled(" CPU: ", Style::default().fg(tc.muted)),
        Span::styled(
            format!(
                "{} ({} cores)",
                app.specs.cpu_name, app.specs.total_cpu_cores
            ),
            Style::default().fg(tc.fg),
        ),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled("RAM: ", Style::default().fg(tc.muted)),
        Span::styled(
            format!(
                "{:.1} GB avail / {:.1} GB total{}",
                app.specs.available_ram_gb,
                app.specs.total_ram_gb,
                if is_running_in_wsl() { " (WSL)" } else { "" }
            ),
            Style::default().fg(tc.accent),
        ),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(gpu_info, Style::default().fg(tc.accent_secondary)),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(ollama_info, Style::default().fg(ollama_color)),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(mlx_info, Style::default().fg(mlx_color)),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" llmfit ")
        .title_style(Style::default().fg(tc.title).add_modifier(Modifier::BOLD));

    let paragraph = Paragraph::new(text).block(block);
    frame.render_widget(paragraph, area);
}

fn draw_search_and_filters(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(30),    // search
            Constraint::Length(24), // provider summary
            Constraint::Length(18), // sort column
            Constraint::Length(20), // fit filter
            Constraint::Length(16), // theme
        ])
        .split(area);

    // Search box
    let search_style = match app.input_mode {
        InputMode::Search => Style::default().fg(tc.accent_secondary),
        InputMode::Normal | InputMode::ProviderPopup => Style::default().fg(tc.muted),
    };

    let search_text = if app.search_query.is_empty() && app.input_mode == InputMode::Normal {
        Line::from(Span::styled(
            "Press / to search...",
            Style::default().fg(tc.muted),
        ))
    } else {
        Line::from(Span::styled(&app.search_query, Style::default().fg(tc.fg)))
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

    // Provider filter summary
    let active_count = app.selected_providers.iter().filter(|&&s| s).count();
    let total_count = app.providers.len();
    let provider_text = if active_count == total_count {
        "All".to_string()
    } else {
        format!("{}/{}", active_count, total_count)
    };
    let provider_color = if active_count == total_count {
        tc.good
    } else if active_count == 0 {
        tc.error
    } else {
        tc.warning
    };

    let provider_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" Providers (p) ")
        .title_style(Style::default().fg(tc.muted));

    let providers = Paragraph::new(Line::from(Span::styled(
        format!(" {}", provider_text),
        Style::default().fg(provider_color),
    )))
    .block(provider_block);
    frame.render_widget(providers, chunks[1]);

    // Sort column
    let sort_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" Sort [s] ")
        .title_style(Style::default().fg(tc.muted));

    let sort_text = Paragraph::new(Line::from(Span::styled(
        format!(" {}", app.sort_column.label()),
        Style::default().fg(tc.accent),
    )))
    .block(sort_block);
    frame.render_widget(sort_text, chunks[2]);

    // Fit filter
    let fit_style = match app.fit_filter {
        FitFilter::All => Style::default().fg(tc.fg),
        FitFilter::Runnable => Style::default().fg(tc.good),
        FitFilter::Perfect => Style::default().fg(tc.good),
        FitFilter::Good => Style::default().fg(tc.warning),
        FitFilter::Marginal => Style::default().fg(tc.fit_marginal),
        FitFilter::TooTight => Style::default().fg(tc.error),
    };

    let fit_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" Fit [f] ")
        .title_style(Style::default().fg(tc.muted));

    let fit_text = Paragraph::new(Line::from(Span::styled(app.fit_filter.label(), fit_style)))
        .block(fit_block);
    frame.render_widget(fit_text, chunks[3]);

    // Theme indicator
    let theme_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" Theme [t] ")
        .title_style(Style::default().fg(tc.muted));

    let theme_text = Paragraph::new(Line::from(Span::styled(
        format!(" {}", app.theme.label()),
        Style::default().fg(tc.info),
    )))
    .block(theme_block);
    frame.render_widget(theme_text, chunks[4]);
}

fn fit_color(level: FitLevel, tc: &ThemeColors) -> Color {
    match level {
        FitLevel::Perfect => tc.fit_perfect,
        FitLevel::Good => tc.fit_good,
        FitLevel::Marginal => tc.fit_marginal,
        FitLevel::TooTight => tc.fit_tight,
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
fn pull_indicator(percent: Option<f64>, tick: u64) -> String {
    const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let spin = SPINNER[(tick as usize / 3) % SPINNER.len()];

    match percent {
        Some(pct) => {
            const BLOCKS: &[char] = &[' ', '░', '▒', '▓', '█'];
            let filled = pct / 100.0 * 3.0;
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

fn draw_table(frame: &mut Frame, app: &mut App, area: Rect, tc: &ThemeColors) {
    let sort_col = app.sort_column;
    let header_names = [
        "", "Inst", "Model", "Provider", "Params", "Score", "tok/s", "Quant", "Mode", "Mem %",
        "Ctx", "Date", "Fit", "Use Case",
    ];
    let sort_col_idx: Option<usize> = match sort_col {
        SortColumn::Score => Some(5),
        SortColumn::Params => Some(4),
        SortColumn::MemPct => Some(9),
        SortColumn::Ctx => Some(10),
        SortColumn::ReleaseDate => Some(11),
        SortColumn::UseCase => Some(13),
    };
    let header_cells = header_names.iter().enumerate().map(|(i, h)| {
        if sort_col_idx == Some(i) {
            Cell::from(format!("{} ▼", h)).style(
                Style::default()
                    .fg(tc.accent_secondary)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Cell::from(*h).style(Style::default().fg(tc.accent).add_modifier(Modifier::BOLD))
        }
    });
    let header = Row::new(header_cells).height(1);

    let rows: Vec<Row> = app
        .filtered_fits
        .iter()
        .map(|&idx| {
            let fit = &app.all_fits[idx];
            let color = fit_color(fit.fit_level, tc);

            let mode_color = match fit.run_mode {
                llmfit_core::fit::RunMode::Gpu => tc.mode_gpu,
                llmfit_core::fit::RunMode::MoeOffload => tc.mode_moe,
                llmfit_core::fit::RunMode::CpuOffload => tc.mode_offload,
                llmfit_core::fit::RunMode::CpuOnly => tc.mode_cpu,
            };

            let score_color = if fit.score >= 70.0 {
                tc.score_high
            } else if fit.score >= 50.0 {
                tc.score_mid
            } else {
                tc.score_low
            };

            #[allow(clippy::if_same_then_else)]
            let tps_text = if fit.estimated_tps >= 100.0 {
                format!("{:.0}", fit.estimated_tps)
            } else if fit.estimated_tps >= 10.0 {
                format!("{:.1}", fit.estimated_tps)
            } else {
                format!("{:.1}", fit.estimated_tps)
            };

            let is_pulling = app.pull_active.is_some()
                && app.pull_model_name.as_deref() == Some(&fit.model.name);
            let has_ollama = providers::has_ollama_mapping(&fit.model.name);

            let installed_icon = if fit.installed {
                " ✓".to_string()
            } else if is_pulling {
                pull_indicator(app.pull_percent, app.tick_count)
            } else if !has_ollama {
                " —".to_string()
            } else {
                " ".to_string()
            };
            #[allow(clippy::if_same_then_else)]
            let installed_color = if fit.installed {
                tc.good
            } else if is_pulling {
                tc.warning
            } else if !has_ollama {
                tc.muted
            } else {
                tc.muted
            };

            let row_style = if is_pulling {
                Style::default().bg(Color::Rgb(50, 50, 0))
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(fit_indicator(fit.fit_level)).style(Style::default().fg(color)),
                Cell::from(installed_icon).style(Style::default().fg(installed_color)),
                Cell::from(fit.model.name.clone()).style(Style::default().fg(tc.fg)),
                Cell::from(fit.model.provider.clone()).style(Style::default().fg(tc.muted)),
                Cell::from(fit.model.parameter_count.clone()).style(Style::default().fg(tc.fg)),
                Cell::from(format!("{:.0}", fit.score)).style(Style::default().fg(score_color)),
                Cell::from(tps_text).style(Style::default().fg(tc.fg)),
                Cell::from(fit.best_quant.clone()).style(Style::default().fg(tc.muted)),
                Cell::from(fit.run_mode_text().to_string()).style(Style::default().fg(mode_color)),
                Cell::from(format!("{:.0}%", fit.utilization_pct))
                    .style(Style::default().fg(color)),
                Cell::from(format!("{}k", fit.model.context_length / 1000))
                    .style(Style::default().fg(tc.muted)),
                Cell::from(
                    fit.model
                        .release_date
                        .as_deref()
                        .and_then(|d| d.get(..7))
                        .unwrap_or("\u{2014}")
                        .to_string(),
                )
                .style(Style::default().fg(tc.muted)),
                Cell::from(fit.fit_text().to_string()).style(Style::default().fg(color)),
                Cell::from(fit.use_case.label().to_string()).style(Style::default().fg(tc.muted)),
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
        Constraint::Length(8),  // date (YYYY-MM)
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
                .border_style(Style::default().fg(tc.border))
                .title(count_text)
                .title_style(Style::default().fg(tc.fg)),
        )
        .row_highlight_style(
            Style::default()
                .bg(tc.highlight_bg)
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

fn draw_detail(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
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

    let color = fit_color(fit.fit_level, tc);

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Model:       ", Style::default().fg(tc.muted)),
            Span::styled(&fit.model.name, Style::default().fg(tc.fg).bold()),
        ]),
        Line::from(vec![
            Span::styled("  Provider:    ", Style::default().fg(tc.muted)),
            Span::styled(&fit.model.provider, Style::default().fg(tc.fg)),
        ]),
        Line::from(vec![
            Span::styled("  Parameters:  ", Style::default().fg(tc.muted)),
            Span::styled(&fit.model.parameter_count, Style::default().fg(tc.fg)),
        ]),
        Line::from(vec![
            Span::styled("  Quantization:", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.model.quantization),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Best Quant:  ", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {} (for this hardware)", fit.best_quant),
                Style::default().fg(tc.good),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Context:     ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} tokens", fit.model.context_length),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Use Case:    ", Style::default().fg(tc.muted)),
            Span::styled(&fit.model.use_case, Style::default().fg(tc.fg)),
        ]),
        Line::from(vec![
            Span::styled("  Category:    ", Style::default().fg(tc.muted)),
            Span::styled(fit.use_case.label(), Style::default().fg(tc.accent)),
        ]),
        Line::from(vec![
            Span::styled("  Released:    ", Style::default().fg(tc.muted)),
            Span::styled(
                fit.model.release_date.as_deref().unwrap_or("Unknown"),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Runtime:     ", Style::default().fg(tc.muted)),
            Span::styled(
                fit.runtime_text(),
                Style::default().fg(if fit.runtime == llmfit_core::fit::InferenceRuntime::Mlx {
                    tc.accent
                } else {
                    tc.fg
                }),
            ),
            Span::styled(
                format!(" (est. ~{:.1} tok/s)", fit.estimated_tps),
                Style::default().fg(tc.muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Installed:   ", Style::default().fg(tc.muted)),
            {
                let ollama_installed =
                    providers::is_model_installed(&fit.model.name, &app.ollama_installed);
                let mlx_installed =
                    providers::is_model_installed_mlx(&fit.model.name, &app.mlx_installed);
                let any_available = app.ollama_available || app.mlx_available;

                if ollama_installed && mlx_installed {
                    Span::styled("✓ Ollama  ✓ MLX", Style::default().fg(tc.good).bold())
                } else if ollama_installed {
                    Span::styled("✓ Ollama", Style::default().fg(tc.good).bold())
                } else if mlx_installed {
                    Span::styled("✓ MLX", Style::default().fg(tc.good).bold())
                } else if any_available {
                    Span::styled("✗ No  (press d to pull)", Style::default().fg(tc.muted))
                } else {
                    Span::styled("- No provider running", Style::default().fg(tc.muted))
                }
            },
        ]),
    ];

    // Scoring section
    let score_color = if fit.score >= 70.0 {
        tc.score_high
    } else if fit.score >= 50.0 {
        tc.score_mid
    } else {
        tc.score_low
    };
    lines.extend_from_slice(&[
        Line::from(""),
        Line::from(Span::styled(
            "  ── Score Breakdown ──",
            Style::default().fg(tc.accent),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Overall:     ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} / 100", fit.score),
                Style::default().fg(score_color).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Quality:     ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.0}", fit.score_components.quality),
                Style::default().fg(tc.fg),
            ),
            Span::styled("  Speed: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.0}", fit.score_components.speed),
                Style::default().fg(tc.fg),
            ),
            Span::styled("  Fit: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.0}", fit.score_components.fit),
                Style::default().fg(tc.fg),
            ),
            Span::styled("  Context: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.0}", fit.score_components.context),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Est. Speed:  ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} tok/s", fit.estimated_tps),
                Style::default().fg(tc.fg),
            ),
        ]),
    ]);

    // MoE Architecture section
    if fit.model.is_moe {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  ── MoE Architecture ──",
            Style::default().fg(tc.accent),
        )));
        lines.push(Line::from(""));

        if let (Some(num_experts), Some(active_experts)) =
            (fit.model.num_experts, fit.model.active_experts)
        {
            lines.push(Line::from(vec![
                Span::styled("  Experts:     ", Style::default().fg(tc.muted)),
                Span::styled(
                    format!(
                        "{} active / {} total per token",
                        active_experts, num_experts
                    ),
                    Style::default().fg(tc.accent),
                ),
            ]));
        }

        if let Some(active_vram) = fit.model.moe_active_vram_gb() {
            lines.push(Line::from(vec![
                Span::styled("  Active VRAM: ", Style::default().fg(tc.muted)),
                Span::styled(
                    format!("{:.1} GB", active_vram),
                    Style::default().fg(tc.accent),
                ),
                Span::styled(
                    format!(
                        "  (vs {:.1} GB full model)",
                        fit.model.min_vram_gb.unwrap_or(0.0)
                    ),
                    Style::default().fg(tc.muted),
                ),
            ]));
        }

        if let Some(offloaded) = fit.moe_offloaded_gb {
            lines.push(Line::from(vec![
                Span::styled("  Offloaded:   ", Style::default().fg(tc.muted)),
                Span::styled(
                    format!("{:.1} GB inactive experts in RAM", offloaded),
                    Style::default().fg(tc.warning),
                ),
            ]));
        }

        if fit.run_mode == llmfit_core::fit::RunMode::MoeOffload {
            lines.push(Line::from(vec![
                Span::styled("  Strategy:    ", Style::default().fg(tc.muted)),
                Span::styled(
                    "Expert offloading (active in VRAM, inactive in RAM)",
                    Style::default().fg(tc.good),
                ),
            ]));
        } else if fit.run_mode == llmfit_core::fit::RunMode::Gpu {
            lines.push(Line::from(vec![
                Span::styled("  Strategy:    ", Style::default().fg(tc.muted)),
                Span::styled(
                    "All experts loaded in VRAM (optimal)",
                    Style::default().fg(tc.good),
                ),
            ]));
        }
    }

    lines.extend_from_slice(&[
        Line::from(""),
        Line::from(Span::styled(
            "  ── System Fit ──",
            Style::default().fg(tc.accent),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Fit Level:   ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} {}", fit_indicator(fit.fit_level), fit.fit_text()),
                Style::default().fg(color).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Run Mode:    ", Style::default().fg(tc.muted)),
            Span::styled(fit.run_mode_text(), Style::default().fg(tc.fg).bold()),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  -- Memory --",
            Style::default().fg(tc.accent),
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
            Span::styled("  Min VRAM:    ", Style::default().fg(tc.muted)),
            Span::styled(format!("{:.1} GB", vram), Style::default().fg(tc.fg)),
            Span::styled(vram_label, Style::default().fg(tc.muted)),
        ]));
    }

    lines.extend_from_slice(&[
        Line::from(vec![
            Span::styled("  Min RAM:     ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} GB", fit.model.min_ram_gb),
                Style::default().fg(tc.fg),
            ),
            Span::styled(
                format!("  (system: {:.1} GB avail)", app.specs.available_ram_gb),
                Style::default().fg(tc.muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Rec RAM:     ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} GB", fit.model.recommended_ram_gb),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Mem Usage:   ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1}%", fit.utilization_pct),
                Style::default().fg(color),
            ),
            Span::styled(
                format!(
                    "  ({:.1} / {:.1} GB)",
                    fit.memory_required_gb, fit.memory_available_gb
                ),
                Style::default().fg(tc.muted),
            ),
        ]),
    ]);

    lines.push(Line::from(""));
    if !fit.notes.is_empty() {
        lines.push(Line::from(Span::styled(
            "  ── Notes ──",
            Style::default().fg(tc.accent),
        )));
        lines.push(Line::from(""));
        for note in &fit.notes {
            lines.push(Line::from(Span::styled(
                format!("  {}", note),
                Style::default().fg(tc.fg),
            )));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(format!(" {} ", fit.model.name))
        .title_style(Style::default().fg(tc.fg).bold());

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn draw_provider_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app.providers.iter().map(|p| p.len()).max().unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.providers.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.providers.len();

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
                        .fg(tc.good)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                } else {
                    Style::default()
                        .fg(tc.fg)
                        .add_modifier(Modifier::BOLD)
                        .bg(tc.highlight_bg)
                }
            } else if app.selected_providers[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_providers.iter().filter(|&&s| s).count();
    let title = format!(" Providers ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    // If a download is in progress, show the progress bar
    if let Some(status) = &app.pull_status {
        let progress_text = if let Some(pct) = app.pull_percent {
            format!(" {} [{:.0}%] ", status, pct)
        } else {
            format!(" {} ", status)
        };

        let (keys, mode_text) = match app.input_mode {
            InputMode::Normal => {
                let detail_key = if app.show_detail {
                    "Enter:table"
                } else {
                    "Enter:detail"
                };
                let ollama_keys = if app.ollama_available {
                    let installed_key = if app.installed_first {
                        "i:all"
                    } else {
                        "i:installed↑"
                    };
                    format!("  {}  d:pull  r:refresh", installed_key)
                } else {
                    String::new()
                };
                (
                    format!(
                        " ↑↓/jk:nav  {}  /:search  f:fit  s:sort  t:theme{}  p:providers  q:quit",
                        detail_key, ollama_keys,
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

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(20),
                Constraint::Length(progress_text.len() as u16 + 2),
            ])
            .split(area);

        let status_line = Line::from(vec![
            Span::styled(
                format!(" {} ", mode_text),
                Style::default().fg(tc.status_fg).bg(tc.status_bg).bold(),
            ),
            Span::styled(keys, Style::default().fg(tc.muted)),
        ]);
        frame.render_widget(Paragraph::new(status_line), chunks[0]);

        let pull_color = if app.pull_active.is_some() {
            tc.warning
        } else {
            tc.good
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                progress_text,
                Style::default().fg(pull_color),
            ))),
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
            let any_provider = app.ollama_available || app.mlx_available;
            let ollama_keys = if any_provider {
                let installed_key = if app.installed_first {
                    "i:all"
                } else {
                    "i:installed↑"
                };
                format!("  {}  d:pull  r:refresh", installed_key)
            } else {
                String::new()
            };
            (
                format!(
                    " ↑↓/jk:nav  {}  /:search  f:fit  s:sort  t:theme{}  p:providers  q:quit",
                    detail_key, ollama_keys,
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
            Style::default().fg(tc.status_fg).bg(tc.status_bg).bold(),
        ),
        Span::styled(keys, Style::default().fg(tc.muted)),
    ]);

    frame.render_widget(Paragraph::new(status_line), area);
}
