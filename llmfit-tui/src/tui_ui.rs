use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Cell, Clear, Paragraph, Row, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Table, TableState, Wrap,
    },
};

use crate::download_history::DownloadResult;
use crate::theme::ThemeColors;
use crate::tui_app::{
    AdvConfigField, App, AvailabilityFilter, DL_DOCKER, DL_LLAMACPP, DL_LMSTUDIO, DL_OLLAMA,
    DownloadCapability, DownloadManagerFocus, DownloadProvider, FitFilter, InputMode, PlanField,
    SimulationField,
};
use llmfit_core::fit::{FitLevel, ModelFit, SortColumn};
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
            Constraint::Length(4), // system info bar (2 rows)
            Constraint::Length(3), // search + filters
            Constraint::Min(10),   // main table
            Constraint::Length(2), // status bar (model name + keybindings)
        ])
        .split(frame.area());

    draw_system_bar(frame, app, outer[0], &tc);
    draw_search_and_filters(frame, app, outer[1], &tc);

    if app.show_downloads {
        draw_downloads(frame, app, outer[2], &tc);
    } else if app.show_plan {
        draw_plan(frame, app, outer[2], &tc);
    } else if app.show_multi_compare {
        draw_multi_compare(frame, app, outer[2], &tc);
    } else if app.show_compare {
        draw_compare(frame, app, outer[2], &tc);
    } else if app.show_detail {
        draw_detail(frame, app, outer[2], &tc);
    } else {
        draw_table(frame, app, outer[2], &tc);
    }

    draw_status_bar(frame, app, outer[3], &tc);

    // Draw popup overlays on top if active
    if app.input_mode == InputMode::ProviderPopup {
        draw_provider_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::UseCasePopup {
        draw_use_case_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::CapabilityPopup {
        draw_capability_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::DownloadProviderPopup {
        draw_download_provider_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::QuantPopup {
        draw_quant_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::RunModePopup {
        draw_run_mode_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::ParamsBucketPopup {
        draw_params_bucket_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::LicensePopup {
        draw_license_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::RuntimePopup {
        draw_runtime_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::HelpPopup {
        draw_help_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::Simulation {
        draw_simulation_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::AdvancedConfig {
        draw_advanced_config_popup(frame, app, &tc);
    } else if app.input_mode == InputMode::FilterPopup {
        draw_filter_popup(frame, app, &tc);
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
        format!("Ollama: ✓ ({} installed)", app.ollama_installed_count)
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

    let llamacpp_info = if app.llamacpp_available {
        if app.llamacpp_detection_hint.is_empty() {
            format!("llama.cpp: ✓ ({} models)", app.llamacpp_installed_count)
        } else {
            format!("llama.cpp: ✓ ({})", app.llamacpp_detection_hint)
        }
    } else if !app.llamacpp_installed.is_empty() {
        format!("llama.cpp: ({} cached)", app.llamacpp_installed_count)
    } else {
        format!("llama.cpp: ✗ ({})", app.llamacpp_detection_hint)
    };
    let llamacpp_color = if app.llamacpp_available {
        tc.good
    } else if !app.llamacpp_installed.is_empty() {
        tc.warning
    } else {
        tc.muted
    };

    let docker_mr_info = if app.docker_mr_available {
        format!("Docker: ✓ ({} models)", app.docker_mr_installed_count)
    } else {
        "Docker: ✗".to_string()
    };
    let docker_mr_color = if app.docker_mr_available {
        tc.good
    } else {
        tc.muted
    };

    let lmstudio_info = if app.lmstudio_available {
        format!("LM Studio: ✓ ({} models)", app.lmstudio_installed_count)
    } else {
        "LM Studio: ✗".to_string()
    };
    let lmstudio_color = if app.lmstudio_available {
        tc.good
    } else {
        tc.muted
    };

    let mut hw_spans = Vec::new();
    if app.sim_active {
        hw_spans.push(Span::styled(
            " SIM ",
            Style::default().fg(tc.bg).bg(tc.warning).bold(),
        ));
    }
    hw_spans.extend([
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
    ]);
    let hardware_line = Line::from(hw_spans);

    let mut provider_spans = vec![
        Span::styled(" ", Style::default()),
        Span::styled(ollama_info, Style::default().fg(ollama_color)),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(mlx_info, Style::default().fg(mlx_color)),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(llamacpp_info, Style::default().fg(llamacpp_color)),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(docker_mr_info, Style::default().fg(docker_mr_color)),
        Span::styled("  │  ", Style::default().fg(tc.muted)),
        Span::styled(lmstudio_info, Style::default().fg(lmstudio_color)),
    ];

    if app.backend_hidden_count > 0 {
        provider_spans.push(Span::styled("  │  ", Style::default().fg(tc.muted)));
        provider_spans.push(Span::styled(
            format!(
                "{} model{} hidden (incompatible backend)",
                app.backend_hidden_count,
                if app.backend_hidden_count == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            Style::default().fg(tc.muted),
        ));
    }

    let provider_line = Line::from(provider_spans);

    let text = Text::from(vec![hardware_line, provider_line]);

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
            Constraint::Length(18), // provider summary
            Constraint::Length(18), // use-case summary
            Constraint::Length(16), // capability summary
            Constraint::Length(18), // sort column
            Constraint::Length(20), // fit filter
            Constraint::Length(20), // availability filter
            Constraint::Length(14), // TP filter
            Constraint::Length(16), // theme
        ])
        .split(area);

    // Search box
    let search_style = match app.input_mode {
        InputMode::Search => Style::default().fg(tc.accent_secondary),
        InputMode::Normal
        | InputMode::Plan
        | InputMode::ProviderPopup
        | InputMode::UseCasePopup
        | InputMode::CapabilityPopup
        | InputMode::DownloadProviderPopup
        | InputMode::Visual
        | InputMode::Select
        | InputMode::QuantPopup
        | InputMode::RunModePopup
        | InputMode::ParamsBucketPopup
        | InputMode::LicensePopup
        | InputMode::RuntimePopup
        | InputMode::HelpPopup
        | InputMode::Simulation
        | InputMode::AdvancedConfig
        | InputMode::DownloadManager
        | InputMode::FilterPopup => Style::default().fg(tc.muted),
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
        .title(" Providers (P) ")
        .title_style(Style::default().fg(tc.muted));

    let providers = Paragraph::new(Line::from(Span::styled(
        format!(" {}", provider_text),
        Style::default().fg(provider_color),
    )))
    .block(provider_block);
    frame.render_widget(providers, chunks[1]);

    // Use-case filter summary
    let active_count = app.selected_use_cases.iter().filter(|&&s| s).count();
    let total_count = app.use_cases.len();
    let use_case_text = if active_count == total_count {
        "All".to_string()
    } else {
        format!("{}/{}", active_count, total_count)
    };
    let use_case_color = if active_count == total_count {
        tc.good
    } else if active_count == 0 {
        tc.error
    } else {
        tc.warning
    };

    let use_case_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" Use Case (U) ")
        .title_style(Style::default().fg(tc.muted));

    let use_cases = Paragraph::new(Line::from(Span::styled(
        format!(" {}", use_case_text),
        Style::default().fg(use_case_color),
    )))
    .block(use_case_block);
    frame.render_widget(use_cases, chunks[2]);

    // Capability filter summary
    let active_cap_count = app.selected_capabilities.iter().filter(|&&s| s).count();
    let total_cap_count = app.capabilities.len();
    let cap_text = if active_cap_count == total_cap_count {
        "All".to_string()
    } else {
        format!("{}/{}", active_cap_count, total_cap_count)
    };
    let cap_color = if active_cap_count == total_cap_count {
        tc.good
    } else if active_cap_count == 0 {
        tc.error
    } else {
        tc.warning
    };

    let cap_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" Caps (C) ")
        .title_style(Style::default().fg(tc.muted));

    let caps = Paragraph::new(Line::from(Span::styled(
        format!(" {}", cap_text),
        Style::default().fg(cap_color),
    )))
    .block(cap_block);
    frame.render_widget(caps, chunks[3]);

    // Sort column
    let sort_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" Sort [s] ")
        .title_style(Style::default().fg(tc.muted));

    let sort_text = Paragraph::new(Line::from(Span::styled(
        format!(
            " {} {}",
            app.sort_column.label(),
            if app.sort_ascending { "↑" } else { "↓" }
        ),
        Style::default().fg(tc.accent),
    )))
    .block(sort_block);
    frame.render_widget(sort_text, chunks[4]);

    // Fit + Filter indicator [f/F]
    let has_range_filters = !app.filter_params_min_input.is_empty()
        || !app.filter_params_max_input.is_empty()
        || !app.filter_mem_pct_min_input.is_empty()
        || !app.filter_mem_pct_max_input.is_empty();

    let fit_color = if has_range_filters || app.fit_filter != FitFilter::All {
        match app.fit_filter {
            FitFilter::All => tc.accent,
            FitFilter::Runnable | FitFilter::Perfect | FitFilter::TurboQuantFit => tc.good,
            FitFilter::Good => tc.warning,
            FitFilter::Marginal => tc.fit_marginal,
            FitFilter::TooTight => tc.error,
        }
    } else {
        tc.fg
    };

    let fit_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" Fit [f] Filter [F] ")
        .title_style(Style::default().fg(tc.muted));

    let mut parts: Vec<&str> = vec![app.fit_filter.label()];
    if !app.filter_params_min_input.is_empty() || !app.filter_params_max_input.is_empty() {
        parts.push("R");
    }
    if !app.filter_mem_pct_min_input.is_empty() || !app.filter_mem_pct_max_input.is_empty() {
        parts.push("M");
    }
    let fit_text = Paragraph::new(Line::from(Span::styled(
        parts.join(" "),
        Style::default().fg(fit_color),
    )))
    .block(fit_block);
    frame.render_widget(fit_text, chunks[5]);

    // Availability filter
    let avail_style = match app.availability_filter {
        AvailabilityFilter::All => Style::default().fg(tc.fg),
        AvailabilityFilter::HasGguf => Style::default().fg(tc.info),
        AvailabilityFilter::Installed => Style::default().fg(tc.good),
    };

    let avail_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" Avail [a] ")
        .title_style(Style::default().fg(tc.muted));

    let avail_text = Paragraph::new(Line::from(Span::styled(
        app.availability_filter.label(),
        avail_style,
    )))
    .block(avail_block);
    frame.render_widget(avail_text, chunks[6]);

    // TP filter
    use crate::tui_app::TpFilter;
    let tp_style = match app.tp_filter {
        TpFilter::All => Style::default().fg(tc.fg),
        _ => Style::default().fg(tc.accent),
    };
    let tp_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(" TP [T] ")
        .title_style(Style::default().fg(tc.muted));
    let tp_text =
        Paragraph::new(Line::from(Span::styled(app.tp_filter.label(), tp_style))).block(tp_block);
    frame.render_widget(tp_text, chunks[7]);

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
    frame.render_widget(theme_text, chunks[8]);
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
        "", "Inst", "Model", "Provider", "Params", "Score", "tok/s*", "Quant", "Disk", "Mode",
        "Mem %", "Ctx", "Date", "Fit", "Use Case",
    ];
    let sort_col_idx: Option<usize> = match sort_col {
        SortColumn::Score => Some(5),
        SortColumn::Tps => Some(6),
        SortColumn::Params => Some(4),
        SortColumn::MemPct => Some(10),
        SortColumn::Ctx => Some(11),
        SortColumn::ReleaseDate => Some(12),
        SortColumn::UseCase => Some(14),
        SortColumn::Provider => Some(3),
    };
    let in_select_mode = app.input_mode == InputMode::Select;
    let header_cells = header_names.iter().enumerate().map(|(i, h)| {
        if in_select_mode && app.select_column == i {
            Cell::from(format!("▸{}◂", h)).style(
                Style::default()
                    .fg(tc.fg)
                    .bg(tc.accent_secondary)
                    .add_modifier(Modifier::BOLD),
            )
        } else if sort_col_idx == Some(i) {
            let arrow = if app.sort_ascending { "▲" } else { "▼" };
            Cell::from(format!("{} {}", h, arrow)).style(
                Style::default()
                    .fg(tc.accent_secondary)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Cell::from(*h).style(Style::default().fg(tc.accent).add_modifier(Modifier::BOLD))
        }
    });
    let header = Row::new(header_cells).height(1);

    let visible_rows = (area.height as usize).saturating_sub(3).max(1);
    let total_rows = app.filtered_fits.len();
    let viewport_start = if total_rows <= visible_rows || app.selected_row < visible_rows {
        0
    } else {
        app.selected_row + 1 - visible_rows
    };
    let viewport_end = (viewport_start + visible_rows).min(total_rows);

    let visual_range = app.visual_range();
    let rows: Vec<Row> = app
        .filtered_fits
        .iter()
        .enumerate()
        .skip(viewport_start)
        .take(viewport_end.saturating_sub(viewport_start))
        .map(|(row_idx, &idx)| {
            let fit = &app.all_fits[idx];
            let color = fit_color(fit.fit_level, tc);

            let mode_color = match fit.run_mode {
                llmfit_core::fit::RunMode::Gpu => tc.mode_gpu,
                llmfit_core::fit::RunMode::TensorParallel => tc.mode_gpu,
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
            let capability = app.download_capability_for(&fit.model.name);

            let installed_icon = if fit.installed {
                " ✓".to_string()
            } else if is_pulling {
                pull_indicator(app.pull_percent, app.tick_count)
            } else {
                match capability {
                    DownloadCapability::Unknown => " …".to_string(),
                    DownloadCapability::Known(flags) => {
                        if flags == 0 {
                            " —".to_string()
                        } else {
                            let mut s = String::new();
                            if flags & DL_OLLAMA != 0 {
                                s.push('O');
                            }
                            if flags & DL_LLAMACPP != 0 {
                                s.push('L');
                            }
                            if flags & DL_DOCKER != 0 {
                                s.push('D');
                            }
                            if flags & DL_LMSTUDIO != 0 {
                                s.push('S');
                            }
                            format!("{:>2}", s)
                        }
                    }
                }
            };
            let installed_color = if fit.installed {
                tc.good
            } else if is_pulling {
                tc.warning
            } else {
                match capability {
                    DownloadCapability::Unknown => tc.muted,
                    DownloadCapability::Known(0) => tc.muted,
                    DownloadCapability::Known(_) => tc.info,
                }
            };

            let in_visual_range = visual_range
                .as_ref()
                .map(|r| r.contains(&row_idx))
                .unwrap_or(false);
            let row_style = if is_pulling {
                Style::default().bg(Color::Rgb(50, 50, 0))
            } else if in_visual_range {
                Style::default().bg(Color::Rgb(40, 40, 80))
            } else {
                Style::default()
            };

            let marker = if app.compare_mark_model.as_deref() == Some(fit.model.name.as_str()) {
                format!("{}*", fit_indicator(fit.fit_level))
            } else {
                fit_indicator(fit.fit_level).to_string()
            };

            Row::new(vec![
                Cell::from(marker).style(Style::default().fg(color)),
                Cell::from(installed_icon).style(Style::default().fg(installed_color)),
                Cell::from(fit.model.name.clone()).style(Style::default().fg(tc.fg)),
                Cell::from(fit.model.provider.clone()).style(Style::default().fg(tc.muted)),
                Cell::from(fit.model.parameter_count.clone()).style(Style::default().fg(tc.fg)),
                Cell::from(format!("{:.0}", fit.score)).style(Style::default().fg(score_color)),
                Cell::from(tps_text).style(Style::default().fg(tc.fg)),
                Cell::from(fit.best_quant.clone()).style(Style::default().fg(tc.muted)),
                Cell::from(format!(
                    "{:.1}G",
                    fit.model.estimate_disk_gb(&fit.best_quant)
                ))
                .style(Style::default().fg(tc.muted)),
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
        Constraint::Length(8),  // score
        Constraint::Length(8),  // tok/s
        Constraint::Length(10), // quant (AWQ-4bit, GPTQ-Int4, GPTQ-Int8)
        Constraint::Length(6),  // disk
        Constraint::Length(7),  // mode
        Constraint::Length(7),  // mem %
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
        state.select(Some(app.selected_row.saturating_sub(viewport_start)));
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

fn draw_compare(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let Some((left, right)) = app.selected_compare_pair() else {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(" Compare ")
            .title_style(Style::default().fg(tc.title).add_modifier(Modifier::BOLD));
        let body = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Compare requires two different models.",
                Style::default().fg(tc.warning),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  1) Move to a model and press m (mark).",
                Style::default().fg(tc.muted),
            )),
            Line::from(Span::styled(
                "  2) Move to another model and press c (compare).",
                Style::default().fg(tc.muted),
            )),
            Line::from(Span::styled(
                "  3) Press c again to return.",
                Style::default().fg(tc.muted),
            )),
        ])
        .block(block);
        frame.render_widget(body, area);
        return;
    };

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(10)])
        .split(area);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(sections[1]);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(" Compare ", Style::default().fg(tc.accent).bold()),
        Span::styled(
            format!("{}  vs  {}", left.model.name, right.model.name),
            Style::default().fg(tc.fg),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border)),
    );
    frame.render_widget(title, sections[0]);

    let score_delta = right.score - left.score;
    let tps_delta = right.estimated_tps - left.estimated_tps;
    let mem_delta = right.utilization_pct - left.utilization_pct;
    let params_delta = right.model.params_b() - left.model.params_b();
    let ctx_delta = right.model.context_length as i64 - left.model.context_length as i64;

    let score_hint = if score_delta > 0.05 {
        " ↑"
    } else if score_delta < -0.05 {
        " ↓"
    } else {
        " ="
    };
    let tps_hint = if tps_delta > 0.05 {
        " ↑"
    } else if tps_delta < -0.05 {
        " ↓"
    } else {
        " ="
    };
    let mem_hint = if mem_delta > 0.05 {
        " ↑"
    } else if mem_delta < -0.05 {
        " ↓"
    } else {
        " ="
    };
    let params_hint = if params_delta > 0.01 {
        " ↑"
    } else if params_delta < -0.01 {
        " ↓"
    } else {
        " ="
    };
    let ctx_hint = if ctx_delta > 0 {
        " ↑"
    } else if ctx_delta < 0 {
        " ↓"
    } else {
        " ="
    };

    let score_style = Style::default().fg(if score_delta >= 0.0 {
        tc.good
    } else {
        tc.warning
    });
    let tps_style = Style::default().fg(if tps_delta >= 0.0 {
        tc.good
    } else {
        tc.warning
    });
    let mem_style = Style::default().fg(if mem_delta <= 0.0 {
        tc.good
    } else {
        tc.warning
    });
    let params_style = Style::default().fg(if params_delta >= 0.0 {
        tc.good
    } else {
        tc.warning
    });
    let ctx_style = Style::default().fg(if ctx_delta >= 0 { tc.good } else { tc.warning });

    let legend = Paragraph::new(Line::from(Span::styled(
        "  Delta hints: ↑ value increased, ↓ value decreased (for Mem%, lower is better)",
        Style::default().fg(tc.muted),
    )));
    frame.render_widget(legend, sections[0]);

    let left_metrics = CompareMetrics {
        score: format!("{:.1}", left.score),
        score_style: Style::default().fg(tc.score_high),
        tps: format!("{:.1}", left.estimated_tps),
        tps_style: Style::default().fg(tc.fg),
        mem: format!("{:.1}%", left.utilization_pct),
        mem_style: Style::default().fg(fit_color(left.fit_level, tc)),
        params: left.model.parameter_count.clone(),
        params_style: Style::default().fg(tc.fg),
        context: format!(" {} tokens", left.model.context_length),
        context_style: Style::default().fg(tc.fg),
    };

    let right_metrics = CompareMetrics {
        score: format!("{:.1} ({:+.1}){}", right.score, score_delta, score_hint),
        score_style,
        tps: format!("{:.1} ({:+.1}){}", right.estimated_tps, tps_delta, tps_hint),
        tps_style,
        mem: format!(
            "{:.1}% ({:+.1}%){}",
            right.utilization_pct, mem_delta, mem_hint
        ),
        mem_style,
        params: format!(
            "{} ({:+.2}B){}",
            right.model.parameter_count, params_delta, params_hint
        ),
        params_style,
        context: format!(
            " {} tokens ({:+}){}",
            right.model.context_length, ctx_delta, ctx_hint
        ),
        context_style: ctx_style,
    };

    render_compare_panel(
        frame,
        cols[0],
        tc,
        " Marked (baseline) ",
        left,
        &left_metrics,
    );
    render_compare_panel(
        frame,
        cols[1],
        tc,
        " Selected (delta vs baseline) ",
        right,
        &right_metrics,
    );
}

struct CompareMetrics {
    score: String,
    score_style: Style,
    tps: String,
    tps_style: Style,
    mem: String,
    mem_style: Style,
    params: String,
    params_style: Style,
    context: String,
    context_style: Style,
}

fn compare_badges(fit: &ModelFit) -> String {
    let mut tags = Vec::new();
    if fit.model.is_moe {
        tags.push("MoE");
    }
    if fit.run_mode == llmfit_core::fit::RunMode::MoeOffload {
        tags.push("Offload");
    }
    if !fit.notes.is_empty() {
        tags.push("Notes");
    }
    if tags.is_empty() {
        "-".to_string()
    } else {
        tags.join(", ")
    }
}

fn render_compare_panel(
    frame: &mut Frame,
    area: Rect,
    tc: &ThemeColors,
    title: &str,
    fit: &ModelFit,
    metrics: &CompareMetrics,
) {
    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Model: ", Style::default().fg(tc.muted)),
            Span::styled(fit.model.name.clone(), Style::default().fg(tc.fg).bold()),
        ]),
        Line::from(vec![
            Span::styled("  Provider:", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.model.provider),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Use:    ", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.use_case.label()),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Released:", Style::default().fg(tc.muted)),
            Span::styled(
                format!(
                    " {}",
                    fit.model.release_date.as_deref().unwrap_or("Unknown")
                ),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  License:", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.model.license.as_deref().unwrap_or("Unknown")),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Score: ", Style::default().fg(tc.muted)),
            Span::styled(metrics.score.clone(), metrics.score_style),
        ]),
        Line::from(vec![
            Span::styled("  Fit:   ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} {}", fit_indicator(fit.fit_level), fit.fit_text()),
                Style::default().fg(fit_color(fit.fit_level, tc)),
            ),
        ]),
        Line::from(vec![
            Span::styled("  tok/s: ", Style::default().fg(tc.muted)),
            Span::styled(metrics.tps.clone(), metrics.tps_style),
        ]),
        Line::from(vec![
            Span::styled("  Mem%:  ", Style::default().fg(tc.muted)),
            Span::styled(metrics.mem.clone(), metrics.mem_style),
        ]),
        Line::from(vec![
            Span::styled("  Disk:  ", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {:.1} GB", fit.model.estimate_disk_gb(&fit.best_quant)),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Runtime:", Style::default().fg(tc.muted)),
            Span::styled(
                format!(" {}", fit.runtime_text()),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Mode:   ", Style::default().fg(tc.muted)),
            Span::styled(fit.run_mode_text(), Style::default().fg(tc.fg)),
        ]),
        Line::from(vec![
            Span::styled("  Params: ", Style::default().fg(tc.muted)),
            Span::styled(metrics.params.clone(), metrics.params_style),
        ]),
        Line::from(vec![
            Span::styled("  Context:", Style::default().fg(tc.muted)),
            Span::styled(metrics.context.clone(), metrics.context_style),
        ]),
        Line::from(vec![
            Span::styled("  Quant:  ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} (default {})", fit.best_quant, fit.model.quantization),
                Style::default().fg(tc.good),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Badges: ", Style::default().fg(tc.muted)),
            Span::styled(compare_badges(fit), Style::default().fg(tc.info)),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(tc.border))
                .title(title)
                .title_style(Style::default().fg(tc.accent_secondary)),
        ),
        area,
    );
}

fn draw_multi_compare(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    if app.compare_models.is_empty() {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(" Compare ")
            .title_style(Style::default().fg(tc.title).add_modifier(Modifier::BOLD));
        let body = Paragraph::new("  No models selected for comparison.").block(block);
        frame.render_widget(body, area);
        return;
    }

    let models: Vec<&ModelFit> = app
        .compare_models
        .iter()
        .filter_map(|&idx| app.all_fits.get(idx))
        .collect();

    if models.len() < 2 {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(" Compare ")
            .title_style(Style::default().fg(tc.title).add_modifier(Modifier::BOLD));
        let body = Paragraph::new("  Need at least 2 models to compare.").block(block);
        frame.render_widget(body, area);
        return;
    }

    // Attribute rows: label, value extractor, color logic
    struct AttrRow {
        label: &'static str,
        values: Vec<String>,
        styles: Vec<Style>,
    }

    let label_width: u16 = 12;
    // How many model columns can we fit?
    let available_width = area.width.saturating_sub(label_width + 3); // borders + label col
    let col_width: u16 = 20;
    let max_visible = (available_width / col_width).max(1) as usize;
    let scroll = app
        .compare_scroll
        .min(models.len().saturating_sub(max_visible));
    let visible_models: Vec<&ModelFit> = models
        .iter()
        .skip(scroll)
        .take(max_visible)
        .copied()
        .collect();
    let n = visible_models.len();

    // Find best/worst for highlighting
    let best_score = models.iter().map(|m| m.score).fold(f64::MIN, f64::max);
    let best_tps = models
        .iter()
        .map(|m| m.estimated_tps)
        .fold(f64::MIN, f64::max);
    let best_mem = models
        .iter()
        .map(|m| m.utilization_pct)
        .fold(f64::MAX, f64::min); // lower is better
    let best_ctx = models
        .iter()
        .map(|m| m.model.context_length)
        .max()
        .unwrap_or(0);

    let mut rows: Vec<AttrRow> = Vec::new();

    // Model name
    rows.push(AttrRow {
        label: "Model",
        values: visible_models
            .iter()
            .map(|m| truncate_str(&m.model.name, col_width as usize - 1))
            .collect(),
        styles: vec![Style::default().fg(tc.fg).add_modifier(Modifier::BOLD); n],
    });

    // Provider
    rows.push(AttrRow {
        label: "Provider",
        values: visible_models
            .iter()
            .map(|m| m.model.provider.clone())
            .collect(),
        styles: vec![Style::default().fg(tc.muted); n],
    });

    // Score
    rows.push(AttrRow {
        label: "Score",
        values: visible_models
            .iter()
            .map(|m| format!("{:.1}", m.score))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                if (m.score - best_score).abs() < 0.1 {
                    Style::default().fg(tc.good).add_modifier(Modifier::BOLD)
                } else if m.score >= 70.0 {
                    Style::default().fg(tc.score_high)
                } else if m.score >= 50.0 {
                    Style::default().fg(tc.score_mid)
                } else {
                    Style::default().fg(tc.score_low)
                }
            })
            .collect(),
    });

    // tok/s
    rows.push(AttrRow {
        label: "tok/s",
        values: visible_models
            .iter()
            .map(|m| format!("{:.1}", m.estimated_tps))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                if (m.estimated_tps - best_tps).abs() < 0.1 {
                    Style::default().fg(tc.good).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(tc.fg)
                }
            })
            .collect(),
    });

    // Fit
    rows.push(AttrRow {
        label: "Fit",
        values: visible_models
            .iter()
            .map(|m| format!("{} {}", fit_indicator(m.fit_level), m.fit_text()))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| Style::default().fg(fit_color(m.fit_level, tc)))
            .collect(),
    });

    // Mem %
    rows.push(AttrRow {
        label: "Mem %",
        values: visible_models
            .iter()
            .map(|m| format!("{:.1}%", m.utilization_pct))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                if (m.utilization_pct - best_mem).abs() < 0.1 {
                    Style::default().fg(tc.good).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(fit_color(m.fit_level, tc))
                }
            })
            .collect(),
    });

    // Disk
    rows.push(AttrRow {
        label: "Disk",
        values: visible_models
            .iter()
            .map(|m| format!("{:.1} GB", m.model.estimate_disk_gb(&m.best_quant)))
            .collect(),
        styles: vec![Style::default().fg(tc.muted); n],
    });

    // Params
    rows.push(AttrRow {
        label: "Params",
        values: visible_models
            .iter()
            .map(|m| m.model.parameter_count.clone())
            .collect(),
        styles: vec![Style::default().fg(tc.fg); n],
    });

    // Mode
    rows.push(AttrRow {
        label: "Mode",
        values: visible_models
            .iter()
            .map(|m| m.run_mode_text().to_string())
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                let c = match m.run_mode {
                    llmfit_core::fit::RunMode::Gpu => tc.mode_gpu,
                    llmfit_core::fit::RunMode::TensorParallel => tc.mode_gpu,
                    llmfit_core::fit::RunMode::MoeOffload => tc.mode_moe,
                    llmfit_core::fit::RunMode::CpuOffload => tc.mode_offload,
                    llmfit_core::fit::RunMode::CpuOnly => tc.mode_cpu,
                };
                Style::default().fg(c)
            })
            .collect(),
    });

    // Context
    rows.push(AttrRow {
        label: "Context",
        values: visible_models
            .iter()
            .map(|m| format!("{}k", m.model.context_length / 1000))
            .collect(),
        styles: visible_models
            .iter()
            .map(|m| {
                if m.model.context_length == best_ctx {
                    Style::default().fg(tc.good).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(tc.muted)
                }
            })
            .collect(),
    });

    // Quant
    rows.push(AttrRow {
        label: "Quant",
        values: visible_models
            .iter()
            .map(|m| m.best_quant.clone())
            .collect(),
        styles: vec![Style::default().fg(tc.muted); n],
    });

    // Use Case
    rows.push(AttrRow {
        label: "Use Case",
        values: visible_models
            .iter()
            .map(|m| m.use_case.label().to_string())
            .collect(),
        styles: vec![Style::default().fg(tc.muted); n],
    });

    // License
    rows.push(AttrRow {
        label: "License",
        values: visible_models
            .iter()
            .map(|m| m.model.license.as_deref().unwrap_or("Unknown").to_string())
            .collect(),
        styles: vec![Style::default().fg(tc.muted); n],
    });

    // Runtime
    rows.push(AttrRow {
        label: "Runtime",
        values: visible_models
            .iter()
            .map(|m| m.runtime_text().to_string())
            .collect(),
        styles: vec![Style::default().fg(tc.fg); n],
    });

    // Build the table
    let mut header_cells = vec![Cell::from("").style(Style::default().fg(tc.accent).bold())];
    for (i, m) in visible_models.iter().enumerate() {
        let name = truncate_str(&m.model.name, col_width as usize - 1);
        let style = if i == 0 && scroll == 0 {
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(tc.accent).add_modifier(Modifier::BOLD)
        };
        header_cells.push(Cell::from(name).style(style));
    }
    let header = Row::new(header_cells).height(1);

    let table_rows: Vec<Row> = rows
        .iter()
        .enumerate()
        .map(|(row_idx, attr)| {
            let mut cells =
                vec![Cell::from(attr.label).style(Style::default().fg(tc.muted).bold())];
            for (col_idx, (val, style)) in attr.values.iter().zip(attr.styles.iter()).enumerate() {
                let _ = col_idx;
                cells.push(Cell::from(val.as_str()).style(*style));
            }
            let bg = if row_idx % 2 == 0 {
                Style::default()
            } else {
                Style::default().bg(Color::Rgb(25, 25, 35))
            };
            Row::new(cells).style(bg)
        })
        .collect();

    let mut widths = vec![Constraint::Length(label_width)];
    for _ in 0..n {
        widths.push(Constraint::Length(col_width));
    }

    let scroll_info = if models.len() > max_visible {
        format!(" Compare ({}/{})  ←/→ scroll ", models.len(), models.len())
    } else {
        format!(" Compare ({} models) ", models.len())
    };

    let table = Table::new(table_rows, widths).header(header).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(scroll_info)
            .title_style(
                Style::default()
                    .fg(tc.accent_secondary)
                    .add_modifier(Modifier::BOLD),
            ),
    );

    frame.render_widget(table, area);
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}~", &s[..max_len.saturating_sub(1)])
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
            Span::styled("  Capabilities:", Style::default().fg(tc.muted)),
            Span::styled(
                if fit.model.capabilities.is_empty() {
                    " None".to_string()
                } else {
                    format!(
                        " {}",
                        fit.model
                            .capabilities
                            .iter()
                            .map(|c| c.label())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                },
                Style::default().fg(tc.info),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Released:    ", Style::default().fg(tc.muted)),
            Span::styled(
                fit.model.release_date.as_deref().unwrap_or("Unknown"),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  License:     ", Style::default().fg(tc.muted)),
            Span::styled(
                fit.model.license.as_deref().unwrap_or("Unknown"),
                Style::default().fg(tc.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Runtime:     ", Style::default().fg(tc.muted)),
            Span::styled(
                fit.runtime_text(),
                Style::default().fg(match fit.runtime {
                    llmfit_core::fit::InferenceRuntime::Mlx => tc.accent,
                    llmfit_core::fit::InferenceRuntime::Vllm => tc.accent_secondary,
                    _ => tc.fg,
                }),
            ),
            Span::styled(
                format!(" (baseline est. ~{:.1} tok/s)", fit.estimated_tps),
                Style::default().fg(tc.muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Installed:   ", Style::default().fg(tc.muted)),
            {
                let mut installed_providers = Vec::new();
                if providers::is_model_installed(&fit.model.name, &app.ollama_installed) {
                    installed_providers.push("Ollama");
                }
                if providers::is_model_installed_mlx(&fit.model.name, &app.mlx_installed) {
                    installed_providers.push("MLX");
                }
                if providers::is_model_installed_llamacpp(&fit.model.name, &app.llamacpp_installed)
                {
                    installed_providers.push("llama.cpp");
                }
                if providers::is_model_installed_docker_mr(
                    &fit.model.name,
                    &app.docker_mr_installed,
                ) {
                    installed_providers.push("Docker");
                }
                if providers::is_model_installed_lmstudio(&fit.model.name, &app.lmstudio_installed)
                {
                    installed_providers.push("LM Studio");
                }
                let any_available = app.ollama_available
                    || app.mlx_available
                    || app.llamacpp_available
                    || app.docker_mr_available
                    || app.lmstudio_available;

                if !installed_providers.is_empty() {
                    let label = installed_providers
                        .iter()
                        .map(|p| format!("✓ {p}"))
                        .collect::<Vec<_>>()
                        .join("  ");
                    Span::styled(label, Style::default().fg(tc.good).bold())
                } else if any_available {
                    Span::styled("✗ No  (press d to pull)", Style::default().fg(tc.muted))
                } else {
                    Span::styled("- No runtime detected", Style::default().fg(tc.muted))
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
            Span::styled("  Baseline Est:", Style::default().fg(tc.muted)),
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
        Line::from(vec![
            Span::styled("  Disk (est):  ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} GB", fit.model.estimate_disk_gb(&fit.best_quant)),
                Style::default().fg(tc.fg),
            ),
            Span::styled(
                format!("  (at {})", fit.best_quant),
                Style::default().fg(tc.muted),
            ),
        ]),
    ]);

    // Disk size breakdown per quant level
    let quants: &[&str] = if fit.best_quant.starts_with("mlx") {
        &["mlx-8bit", "mlx-4bit"]
    } else {
        &["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
    };
    let mut disk_spans: Vec<Span> = vec![Span::styled(
        "  Disk/quant:  ",
        Style::default().fg(tc.muted),
    )];
    for (i, &q) in quants.iter().enumerate() {
        if i > 0 {
            disk_spans.push(Span::styled("  ", Style::default()));
        }
        let size = fit.model.estimate_disk_gb(q);
        let text = format!("{}: {:.1}G", q, size);
        let style = if q == fit.best_quant {
            Style::default().fg(tc.good).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(tc.muted)
        };
        disk_spans.push(Span::styled(text, style));
    }
    lines.push(Line::from(disk_spans));

    if fit.model.params_b() > 0.0 {
        lines.push(Line::from(Span::styled(
            "  -- VRAM by Context --",
            Style::default().fg(tc.accent),
        )));

        let display_quant = fit.best_quant.as_str();
        let quant = display_quant
            .split_whitespace()
            .next()
            .unwrap_or(display_quant);
        let available_gpu_vram = app.specs.gpu_vram_gb;
        let available_ram = app.specs.available_ram_gb;

        for &ctx in &[4096_u32, 8192, 16384, 32768, 65536, 131072] {
            if ctx > fit.model.context_length {
                continue;
            }

            let mem_gb = fit.model.estimate_memory_gb(quant, ctx);
            let mem_color = if available_gpu_vram.is_some_and(|vram| mem_gb <= vram) {
                tc.good
            } else if mem_gb <= available_ram {
                tc.warning
            } else {
                tc.error
            };

            let ctx_label = format!("{}K", ctx / 1024);
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:>4} ctx:   ", ctx_label),
                    Style::default().fg(tc.muted),
                ),
                Span::styled(
                    format!("{:>6.1} GB", mem_gb),
                    Style::default().fg(mem_color),
                ),
            ]));
        }
    }

    // Build right-pane content (GGUF sources + notes)
    let has_right_pane =
        !fit.model.gguf_sources.is_empty() || !fit.notes.is_empty() || fit.fits_with_turboquant;

    // Pre-compute right pane inner width for line-wrapping decisions
    // (45% of area minus 2 border columns)
    let right_inner_width = (area.width as usize * 45 / 100).saturating_sub(2);

    let mut right_lines: Vec<Line> = vec![Line::from("")];

    if !fit.model.gguf_sources.is_empty() {
        right_lines.push(Line::from(Span::styled(
            "  ── GGUF Downloads ──",
            Style::default().fg(tc.accent),
        )));
        right_lines.push(Line::from(""));
        for src in &fit.model.gguf_sources {
            let provider_str = format!("  📦 {:<12}", src.provider);
            let url_str = format!("hf.co/{}", src.repo);
            // Visual width: "  📦 " = 5 display cols (📦 is 2-wide), plus padded provider
            let provider_visual_width = 5 + src.provider.len().max(12);
            if provider_visual_width + url_str.len() <= right_inner_width {
                // Fits on one line
                right_lines.push(Line::from(vec![
                    Span::styled(provider_str, Style::default().fg(tc.info)),
                    Span::styled(url_str, Style::default().fg(tc.fg)),
                ]));
            } else {
                // Too wide: put URL on its own indented line
                right_lines.push(Line::from(Span::styled(
                    provider_str,
                    Style::default().fg(tc.info),
                )));
                right_lines.push(Line::from(Span::styled(
                    format!("       {}", url_str),
                    Style::default().fg(tc.fg),
                )));
            }
        }
        right_lines.push(Line::from(""));
        right_lines.push(Line::from(Span::styled(
            "  llmfit download \\".to_string(),
            Style::default().fg(tc.muted),
        )));
        right_lines.push(Line::from(Span::styled(
            format!("    {} \\", fit.model.gguf_sources[0].repo),
            Style::default().fg(tc.muted),
        )));
        right_lines.push(Line::from(Span::styled(
            format!("    --quant {}", fit.best_quant),
            Style::default().fg(tc.muted),
        )));
        right_lines.push(Line::from(""));
    }

    if !fit.notes.is_empty() {
        right_lines.push(Line::from(Span::styled(
            "  ── Notes ──",
            Style::default().fg(tc.accent),
        )));
        right_lines.push(Line::from(""));
        for note in &fit.notes {
            right_lines.push(Line::from(Span::styled(
                format!("  {}", note),
                Style::default().fg(tc.fg),
            )));
        }
    }

    if fit.fits_with_turboquant {
        right_lines.push(Line::from(""));
        right_lines.push(Line::from(Span::styled(
            "  TurboQuant+: Would fit with 9.8x KV compression",
            Style::default().fg(tc.good).add_modifier(Modifier::BOLD),
        )));
        right_lines.push(Line::from(Span::styled(
            "  (github.com/0xSero/turboquant)",
            Style::default().fg(tc.muted),
        )));
    }

    // Track the left pane area for cursor positioning
    let left_area;

    if has_right_pane {
        // Split into left (model info) and right (downloads + notes) panes
        let h_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);

        left_area = h_layout[0];

        let left_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(format!(" {} ", fit.model.name))
            .title_style(Style::default().fg(tc.fg).bold());

        let left_paragraph = Paragraph::new(lines)
            .block(left_block)
            .wrap(Wrap { trim: false });
        frame.render_widget(left_paragraph, h_layout[0]);

        let right_title = if !fit.model.gguf_sources.is_empty() {
            " 📦 Downloads & Notes "
        } else {
            " Notes "
        };
        let right_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(right_title)
            .title_style(Style::default().fg(tc.info).bold());

        let right_paragraph = Paragraph::new(right_lines)
            .block(right_block)
            .wrap(Wrap { trim: false });
        frame.render_widget(right_paragraph, h_layout[1]);
    } else {
        left_area = area;

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

    if app.input_mode == InputMode::Plan {
        let (row_offset, label_len) = match app.plan_field {
            PlanField::Context => (5u16, "  Context:    ".len() as u16),
            PlanField::Quant => (6u16, "  Quant:      ".len() as u16),
            PlanField::KvQuant => (7u16, "  KV Quant:   ".len() as u16),
            PlanField::TargetTps => (8u16, "  Target TPS: ".len() as u16),
        };
        let x = left_area.x + 1 + label_len + app.plan_cursor_position as u16;
        let y = left_area.y + 1 + row_offset;
        if x < left_area.x + left_area.width.saturating_sub(1)
            && y < left_area.y + left_area.height.saturating_sub(1)
        {
            frame.set_cursor_position((x, y));
        }
    }
}

fn draw_plan(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let Some(model_name) = app.plan_model_name() else {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.border))
            .title(" Planner ");
        frame.render_widget(block, area);
        return;
    };

    let field_style = |field: PlanField| {
        if app.input_mode == InputMode::Plan && app.plan_field == field {
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(tc.fg)
        }
    };

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Model: ", Style::default().fg(tc.muted)),
            Span::styled(model_name, Style::default().fg(tc.fg).bold()),
        ]),
        Line::from(vec![
            Span::styled("  Note: ", Style::default().fg(tc.muted)),
            Span::styled(
                "Estimate-based using current llmfit fit/speed heuristics.",
                Style::default().fg(tc.warning),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Inputs (editable)",
            Style::default().fg(tc.accent),
        )),
        Line::from(vec![
            Span::styled("  Context:    ", Style::default().fg(tc.muted)),
            Span::styled(
                if app.plan_context_input.is_empty() {
                    "<required>"
                } else {
                    app.plan_context_input.as_str()
                },
                field_style(PlanField::Context),
            ),
            Span::styled(" tokens", Style::default().fg(tc.muted)),
        ]),
        Line::from(vec![
            Span::styled("  Quant:      ", Style::default().fg(tc.muted)),
            Span::styled(
                if app.plan_quant_input.is_empty() {
                    "<auto>"
                } else {
                    app.plan_quant_input.as_str()
                },
                field_style(PlanField::Quant),
            ),
        ]),
        Line::from(vec![
            Span::styled("  KV Quant:   ", Style::default().fg(tc.muted)),
            Span::styled(
                if app.plan_kv_quant_input.is_empty() {
                    "<fp16>"
                } else {
                    app.plan_kv_quant_input.as_str()
                },
                field_style(PlanField::KvQuant),
            ),
            Span::styled(
                "  (fp16, fp8, q8_0, q4_0, tq)",
                Style::default().fg(tc.muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Target TPS: ", Style::default().fg(tc.muted)),
            Span::styled(
                if app.plan_target_tps_input.is_empty() {
                    "<none>"
                } else {
                    app.plan_target_tps_input.as_str()
                },
                field_style(PlanField::TargetTps),
            ),
            Span::styled(" tok/s", Style::default().fg(tc.muted)),
        ]),
        Line::from(""),
    ];

    if let Some(err) = &app.plan_error {
        lines.push(Line::from(vec![
            Span::styled("  Error: ", Style::default().fg(tc.error)),
            Span::styled(err, Style::default().fg(tc.error).bold()),
        ]));
    } else if let Some(plan) = &app.plan_estimate {
        lines.push(Line::from(vec![
            Span::styled("  Active KV: ", Style::default().fg(tc.muted)),
            Span::styled(plan.kv_quant.label(), Style::default().fg(tc.fg).bold()),
        ]));
        lines.push(Line::from(" "));
        lines.push(Line::from(Span::styled(
            "  Minimum Hardware",
            Style::default().fg(tc.accent),
        )));
        lines.push(Line::from(vec![
            Span::styled("  VRAM: ", Style::default().fg(tc.muted)),
            Span::styled(
                plan.minimum
                    .vram_gb
                    .map(|v| format!("{v:.1} GB"))
                    .unwrap_or_else(|| "n/a".to_string()),
                Style::default().fg(tc.fg),
            ),
            Span::styled("   RAM: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} GB", plan.minimum.ram_gb),
                Style::default().fg(tc.fg),
            ),
            Span::styled("   CPU: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} cores", plan.minimum.cpu_cores),
                Style::default().fg(tc.fg),
            ),
        ]));
        lines.push(Line::from(" "));
        lines.push(Line::from(Span::styled(
            "  Recommended Hardware",
            Style::default().fg(tc.accent),
        )));
        lines.push(Line::from(vec![
            Span::styled("  VRAM: ", Style::default().fg(tc.muted)),
            Span::styled(
                plan.recommended
                    .vram_gb
                    .map(|v| format!("{v:.1} GB"))
                    .unwrap_or_else(|| "n/a".to_string()),
                Style::default().fg(tc.fg),
            ),
            Span::styled("   RAM: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{:.1} GB", plan.recommended.ram_gb),
                Style::default().fg(tc.fg),
            ),
            Span::styled("   CPU: ", Style::default().fg(tc.muted)),
            Span::styled(
                format!("{} cores", plan.recommended.cpu_cores),
                Style::default().fg(tc.fg),
            ),
        ]));
        lines.push(Line::from(" "));
        lines.push(Line::from(Span::styled(
            "  Run Paths",
            Style::default().fg(tc.accent),
        )));

        for path in &plan.run_paths {
            let path_color = if path.feasible { tc.good } else { tc.error };
            let status = if path.feasible { "yes" } else { "no" };
            lines.push(Line::from(vec![
                Span::styled("  - ", Style::default().fg(tc.muted)),
                Span::styled(path.path.label(), Style::default().fg(tc.fg).bold()),
                Span::styled(": ", Style::default().fg(tc.muted)),
                Span::styled(status, Style::default().fg(path_color)),
                Span::styled("  tps=", Style::default().fg(tc.muted)),
                Span::styled(
                    path.estimated_tps
                        .map(|t| format!("{t:.1}"))
                        .unwrap_or_else(|| "-".to_string()),
                    Style::default().fg(tc.fg),
                ),
                Span::styled("  fit=", Style::default().fg(tc.muted)),
                Span::styled(
                    path.fit_level
                        .map(|f| match f {
                            FitLevel::Perfect => "Perfect",
                            FitLevel::Good => "Good",
                            FitLevel::Marginal => "Marginal",
                            FitLevel::TooTight => "Too Tight",
                        })
                        .unwrap_or("-"),
                    Style::default().fg(path_color),
                ),
            ]));
        }

        lines.push(Line::from(" "));
        lines.push(Line::from(Span::styled(
            "  Upgrade Deltas",
            Style::default().fg(tc.accent),
        )));
        if plan.upgrade_deltas.is_empty() {
            lines.push(Line::from(Span::styled(
                "  - none required",
                Style::default().fg(tc.good),
            )));
        } else {
            for delta in &plan.upgrade_deltas {
                lines.push(Line::from(Span::styled(
                    format!("  - {}", delta.description),
                    Style::default().fg(tc.fg),
                )));
            }
        }

        if !plan.kv_alternatives.is_empty() {
            lines.push(Line::from(" "));
            lines.push(Line::from(Span::styled(
                "  KV Cache Alternatives",
                Style::default().fg(tc.accent),
            )));
            lines.push(Line::from(Span::styled(
                format!(
                    "  {:<8} {:>10} {:>10} {:>10}",
                    "kv", "kv (GB)", "total", "savings"
                ),
                Style::default().fg(tc.muted),
            )));
            for alt in &plan.kv_alternatives {
                let label = if alt.supported {
                    alt.kv_quant.label().to_string()
                } else {
                    format!("{} (n/a)", alt.kv_quant.label())
                };
                let savings_str = if alt.savings_fraction > 0.0 {
                    format!("-{:.0}%", alt.savings_fraction * 100.0)
                } else {
                    "-".to_string()
                };
                let row_color = if !alt.supported {
                    tc.muted
                } else if alt.kv_quant == plan.kv_quant {
                    tc.good
                } else {
                    tc.fg
                };
                lines.push(Line::from(Span::styled(
                    format!(
                        "  {:<8} {:>10.2} {:>10.2} {:>10}",
                        label, alt.kv_cache_gb, alt.memory_required_gb, savings_str
                    ),
                    Style::default().fg(row_color),
                )));
                if let Some(note) = &alt.note {
                    lines.push(Line::from(Span::styled(
                        format!("            {}", note),
                        Style::default().fg(tc.muted),
                    )));
                }
            }
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.border))
        .title(format!(" Plan: {} ", model_name))
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
        .style(Style::default().bg(tc.bg))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        )
        .title_bottom(
            Line::from(vec![
                Span::styled(
                    " a",
                    Style::default()
                        .fg(tc.accent_secondary)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(": all | ", Style::default().fg(tc.muted)),
                Span::styled(
                    "c",
                    Style::default()
                        .fg(tc.accent_secondary)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(": clear ", Style::default().fg(tc.muted)),
            ])
            .centered(),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_use_case_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app
        .use_cases
        .iter()
        .map(|uc| uc.label().len())
        .max()
        .unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.use_cases.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.use_cases.len();

    let scroll_offset = if app.use_case_cursor >= inner_height {
        app.use_case_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .use_cases
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, use_case)| {
            let checkbox = if app.selected_use_cases[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.use_case_cursor;

            let style = if is_cursor {
                if app.selected_use_cases[i] {
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
            } else if app.selected_use_cases[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(
                format!(" {} {}", checkbox, use_case.label()),
                style,
            ))
        })
        .collect();

    let active_count = app.selected_use_cases.iter().filter(|&&s| s).count();
    let title = format!(" Use Cases ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_capability_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app
        .capabilities
        .iter()
        .map(|c| c.label().len())
        .max()
        .unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.capabilities.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.capabilities.len();

    let scroll_offset = if app.capability_cursor >= inner_height {
        app.capability_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .capabilities
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, cap)| {
            let checkbox = if app.selected_capabilities[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.capability_cursor;

            let style = if is_cursor {
                if app.selected_capabilities[i] {
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
            } else if app.selected_capabilities[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(
                format!(" {} {}", checkbox, cap.label()),
                style,
            ))
        })
        .collect();

    let active_count = app.selected_capabilities.iter().filter(|&&s| s).count();
    let title = format!(" Capabilities ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_download_provider_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();
    let popup_width = 44.min(area.width.saturating_sub(4));
    let popup_height = 8.min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let mut lines = Vec::new();
    if let Some(name) = &app.download_provider_model {
        lines.push(Line::from(Span::styled(
            format!(" Model: {}", name),
            Style::default().fg(tc.muted),
        )));
        lines.push(Line::from(""));
    }

    for (i, provider) in app.download_provider_options.iter().enumerate() {
        let label = match provider {
            DownloadProvider::Ollama => "Ollama",
            DownloadProvider::Mlx => "MLX",
            DownloadProvider::LlamaCpp => "llama.cpp",
            DownloadProvider::DockerModelRunner => "Docker Model Runner",
            DownloadProvider::LmStudio => "LM Studio",
        };
        let is_cursor = i == app.download_provider_cursor;
        let prefix = if is_cursor { ">" } else { " " };
        let style = if is_cursor {
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD)
                .bg(tc.highlight_bg)
        } else {
            Style::default().fg(tc.fg)
        };
        lines.push(Line::from(Span::styled(
            format!(" {} {}", prefix, label),
            style,
        )));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(" Download With ")
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn status_keys_and_mode(app: &App) -> (String, String) {
    match app.input_mode {
        InputMode::Normal => {
            if app.show_multi_compare {
                return (
                    " ←/→/hl:scroll  q/Esc:close".to_string(),
                    "COMPARE".to_string(),
                );
            }
            let detail_key = if app.show_detail {
                "Enter:table"
            } else {
                "Enter:detail"
            };
            let any_provider = app.ollama_available
                || app.mlx_available
                || app.llamacpp_available
                || app.docker_mr_available
                || app.lmstudio_available;
            let ollama_keys = if any_provider {
                let installed_key = if app.installed_first {
                    "i:all"
                } else {
                    "i:installed↑"
                };
                format!("  {}  d:pull  D:downloads  r:refresh", installed_key)
            } else {
                String::new()
            };
            (
                format!(
                    " S:simulate  A:config  h:help  {}  /:search  f:fit  F:filter  s:sort{}  P:providers  U:use cases  C:caps  R:runtime  q:quit",
                    detail_key, ollama_keys,
                ),
                if app.sim_active {
                    "NORMAL [SIM]".to_string()
                } else {
                    "NORMAL".to_string()
                },
            )
        }
        InputMode::Visual => {
            let count = app.visual_selection_count();
            (
                format!(
                    " ↑↓/jk:extend  c:compare  m:mark  Esc:exit  ({} selected)",
                    count
                ),
                "VISUAL".to_string(),
            )
        }
        InputMode::Select => {
            let header_names = [
                "", "Inst", "Model", "Provider", "Params", "Score", "tok/s*", "Quant", "Mode",
                "Mem %", "Ctx", "Date", "Fit", "Use Case",
            ];
            let col_name = header_names.get(app.select_column).unwrap_or(&"");
            (
                format!(" ←/→:column  ↑↓:nav  Enter:filter [{}]  Esc:exit", col_name),
                "SELECT".to_string(),
            )
        }
        InputMode::Search => (
            "  Type to search  Esc:done  Ctrl-U:clear".to_string(),
            "SEARCH".to_string(),
        ),
        InputMode::Plan => (
            "  Tab/jk:field  ←/→:cursor  type:edit  Backspace/Delete  Ctrl-U:clear  Esc:close"
                .to_string(),
            "PLAN".to_string(),
        ),
        InputMode::ProviderPopup => (
            "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
            "PROVIDERS".to_string(),
        ),
        InputMode::UseCasePopup => (
            "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
            "USE CASES".to_string(),
        ),
        InputMode::CapabilityPopup => (
            "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
            "CAPABILITIES".to_string(),
        ),
        InputMode::DownloadProviderPopup => (
            "  ↑↓/jk:choose  Enter:download  Esc:cancel".to_string(),
            "DOWNLOAD".to_string(),
        ),
        InputMode::QuantPopup => (
            "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
            "QUANT".to_string(),
        ),
        InputMode::RunModePopup => (
            "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
            "RUN MODE".to_string(),
        ),
        InputMode::ParamsBucketPopup => (
            "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
            "PARAMS".to_string(),
        ),
        InputMode::LicensePopup => (
            "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
            "LICENSE".to_string(),
        ),
        InputMode::RuntimePopup => (
            "  ↑↓/jk:navigate  Space:toggle  a:all/none  Esc:close".to_string(),
            "RUNTIME".to_string(),
        ),
        InputMode::HelpPopup => (
            "  ↑↓/jk:scroll  Esc/h/q:close".to_string(),
            "HELP".to_string(),
        ),
        InputMode::Simulation => (
            "  Tab/jk:field  type:edit  Enter:apply  Ctrl-R:reset  Esc:close".to_string(),
            "SIMULATION".to_string(),
        ),
        InputMode::AdvancedConfig => (
            "  Tab/jk:field  type:edit  Enter:apply  Ctrl-R:reset  Esc:close".to_string(),
            "ADV CONFIG".to_string(),
        ),
        InputMode::DownloadManager => (
            "  Tab:section  jk:navigate  x:delete  e:edit dir  D/Esc:close".to_string(),
            "DOWNLOADS".to_string(),
        ),
        InputMode::FilterPopup => (
            "  Tab/jk:nav  type:range  Space:toggle  Enter:apply  Ctrl-U:clear  Esc:close"
                .to_string(),
            "FILTER".to_string(),
        ),
    }
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let (keys, mode_text) = status_keys_and_mode(app);

    // Split into 2 rows: selected model name + keybindings
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);

    // Row 0: selected model full name
    let model_line = if !app.show_detail
        && !app.show_compare
        && !app.show_multi_compare
        && !app.show_plan
        && !app.show_downloads
    {
        if let Some(&idx) = app.filtered_fits.get(app.selected_row) {
            let fit = &app.all_fits[idx];
            Line::from(vec![
                Span::styled(" ▶ ", Style::default().fg(tc.accent).bold()),
                Span::styled(
                    fit.model.name.clone(),
                    Style::default().fg(tc.fg).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {}  {}", fit.model.parameter_count, fit.model.provider),
                    Style::default().fg(tc.muted),
                ),
            ])
        } else {
            Line::from(Span::styled(
                " No model selected",
                Style::default().fg(tc.muted),
            ))
        }
    } else {
        Line::from("")
    };
    frame.render_widget(Paragraph::new(model_line), rows[0]);

    // Row 1: keybindings (with download progress if active)
    if let Some(status) = &app.pull_status {
        let progress_text = if let Some(pct) = app.pull_percent {
            format!(" {} [{:.0}%] ", status, pct)
        } else {
            format!(" {} ", status)
        };

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(20),
                Constraint::Length(progress_text.len() as u16 + 2),
            ])
            .split(rows[1]);

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

    let status_line = Line::from(vec![
        Span::styled(
            format!(" {} ", mode_text),
            Style::default().fg(tc.status_fg).bg(tc.status_bg).bold(),
        ),
        Span::styled(keys, Style::default().fg(tc.muted)),
    ]);

    frame.render_widget(Paragraph::new(status_line), rows[1]);
}

fn draw_quant_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app.quants.iter().map(|q| q.len()).max().unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.quants.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.quants.len();

    let scroll_offset = if app.quant_cursor >= inner_height {
        app.quant_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .quants
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_quants[i] { "[x]" } else { "[ ]" };
            let is_cursor = i == app.quant_cursor;

            let style = if is_cursor {
                if app.selected_quants[i] {
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
            } else if app.selected_quants[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_quants.iter().filter(|&&s| s).count();
    let title = format!(" Quant ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_run_mode_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app.run_modes.iter().map(|m| m.len()).max().unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.run_modes.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.run_modes.len();

    let scroll_offset = if app.run_mode_cursor >= inner_height {
        app.run_mode_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .run_modes
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_run_modes[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.run_mode_cursor;

            let style = if is_cursor {
                if app.selected_run_modes[i] {
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
            } else if app.selected_run_modes[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_run_modes.iter().filter(|&&s| s).count();
    let title = format!(" Run Mode ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_params_bucket_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app
        .params_buckets
        .iter()
        .map(|b| b.len())
        .max()
        .unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.params_buckets.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.params_buckets.len();

    let scroll_offset = if app.params_bucket_cursor >= inner_height {
        app.params_bucket_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .params_buckets
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_params_buckets[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.params_bucket_cursor;

            let style = if is_cursor {
                if app.selected_params_buckets[i] {
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
            } else if app.selected_params_buckets[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_params_buckets.iter().filter(|&&s| s).count();
    let title = format!(" Params ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_help_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let popup_width = 52.min(area.width.saturating_sub(4));
    let popup_height = (area.height - 4).min(32);

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    // Entries: ("key", "description") — empty key = blank line, key without leading spaces = section header
    let help_entries: Vec<(&str, &str)> = vec![
        ("Navigation", ""),
        ("  ↑ / k", "Move up"),
        ("  ↓ / j", "Move down"),
        ("  Enter", "Toggle detail view"),
        ("  /", "Search"),
        ("  Ctrl-U", "Clear search"),
        ("", ""),
        ("Filters", ""),
        ("  f", "Cycle fit filter"),
        ("  F", "Filter popup (range, sort dir)"),
        ("  a", "Cycle availability filter"),
        ("  T", "Cycle tensor-parallel filter"),
        ("  P", "Provider filter"),
        ("  U", "Use case filter"),
        ("  C", "Capability filter"),
        ("  L", "License filter"),
        ("  R", "Runtime/backend filter"),
        ("", ""),
        ("Sorting & Display", ""),
        ("  s", "Cycle sort column"),
        ("  i", "Toggle installed-first sort"),
        ("  t", "Cycle theme"),
        ("", ""),
        ("Actions", ""),
        ("  S", "Hardware simulation"),
        ("  A", "Advanced configuration"),
        ("  d", "Download/pull model"),
        ("  r", "Refresh installed models"),
        ("  p", "Plan mode"),
        ("  y", "Copy model name"),
        ("", ""),
        ("Comparison", ""),
        ("  m", "Mark model for compare"),
        ("  c", "Compare marked models"),
        ("  x", "Clear marked models"),
        ("  v", "Visual select mode"),
        ("  V", "Column select mode"),
        ("", ""),
        ("General", ""),
        ("  h", "This help screen"),
        ("  q / Esc", "Quit / close popup"),
    ];

    let all_lines: Vec<Line> = help_entries
        .iter()
        .map(|(key, desc)| {
            if key.is_empty() {
                Line::from("")
            } else if desc.is_empty() && !key.starts_with(' ') {
                // Section header
                Line::from(Span::styled(
                    format!(" {}", key),
                    Style::default()
                        .fg(tc.accent_secondary)
                        .add_modifier(Modifier::BOLD),
                ))
            } else {
                Line::from(vec![
                    Span::styled(
                        format!(" {:<14}", key),
                        Style::default().fg(tc.fg).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(*desc, Style::default().fg(tc.muted)),
                ])
            }
        })
        .collect();

    let inner_height = popup_height.saturating_sub(2) as usize;
    let max_scroll = all_lines.len().saturating_sub(inner_height);
    let scroll = app.help_scroll.min(max_scroll);

    let visible: Vec<Line> = all_lines
        .into_iter()
        .skip(scroll)
        .take(inner_height)
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .title(" Key Bindings ")
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(visible).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_runtime_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app.runtimes.iter().map(|r| r.len()).max().unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.runtimes.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.runtimes.len();

    let scroll_offset = if app.runtime_cursor >= inner_height {
        app.runtime_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .runtimes
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_runtimes[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.runtime_cursor;

            let style = if is_cursor {
                if app.selected_runtimes[i] {
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
            } else if app.selected_runtimes[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_runtimes.iter().filter(|&&s| s).count();
    let title = format!(" Runtime ({}/{}) ", active_count, total);

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

fn draw_license_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let max_name_len = app.licenses.iter().map(|l| l.len()).max().unwrap_or(10);
    let popup_width = (max_name_len as u16 + 10).min(area.width.saturating_sub(4));
    let popup_height = (app.licenses.len() as u16 + 2).min(area.height.saturating_sub(4));

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_height = popup_height.saturating_sub(2) as usize;
    let total = app.licenses.len();

    let scroll_offset = if app.license_cursor >= inner_height {
        app.license_cursor - inner_height + 1
    } else {
        0
    };

    let lines: Vec<Line> = app
        .licenses
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
        .map(|(i, name)| {
            let checkbox = if app.selected_licenses[i] {
                "[x]"
            } else {
                "[ ]"
            };
            let is_cursor = i == app.license_cursor;

            let style = if is_cursor {
                if app.selected_licenses[i] {
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
            } else if app.selected_licenses[i] {
                Style::default().fg(tc.good)
            } else {
                Style::default().fg(tc.muted)
            };

            Line::from(Span::styled(format!(" {} {}", checkbox, name), style))
        })
        .collect();

    let active_count = app.selected_licenses.iter().filter(|&&s| s).count();
    let title = format!(" License ({}/{}) ", active_count, total);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(title)
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn draw_simulation_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let popup_width = 48u16.min(area.width.saturating_sub(4));
    let popup_height = 14u16.min(area.height.saturating_sub(4));
    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(" Hardware Simulation ")
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let fields = [
        ("  RAM (GB):", &app.sim_ram_input, SimulationField::Ram),
        ("  VRAM (GB):", &app.sim_vram_input, SimulationField::Vram),
        (
            "  CPU Cores:",
            &app.sim_cpu_input,
            SimulationField::CpuCores,
        ),
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    for (label, value, field) in &fields {
        let is_active = app.sim_field == *field;
        let label_style = if is_active {
            Style::default().fg(tc.accent).bold()
        } else {
            Style::default().fg(tc.fg)
        };
        let value_style = if is_active {
            Style::default().fg(tc.fg).bg(tc.highlight_bg)
        } else {
            Style::default().fg(tc.fg)
        };

        let display_val = if value.is_empty() && is_active {
            "_".to_string()
        } else {
            format!("{:<16}", value)
        };

        lines.push(Line::from(vec![
            Span::styled(format!("{:<14}", label), label_style),
            Span::styled(display_val, value_style),
        ]));
    }

    lines.push(Line::from(""));

    // Show real hardware for reference
    let real_vram = app
        .real_specs
        .gpu_vram_gb
        .map(|v| format!("{:.1}", v))
        .unwrap_or_else(|| "none".to_string());
    lines.push(Line::from(Span::styled(
        format!(
            "  Real: {:.1} GB RAM, {} GB VRAM, {} cores",
            app.real_specs.total_ram_gb, real_vram, app.real_specs.total_cpu_cores,
        ),
        Style::default().fg(tc.muted),
    )));

    if app.specs.unified_memory {
        lines.push(Line::from(Span::styled(
            "  (unified memory: RAM also affects VRAM)",
            Style::default().fg(tc.muted),
        )));
    }

    if app.sim_active {
        lines.push(Line::from(Span::styled(
            "  Currently simulating",
            Style::default().fg(tc.warning),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Enter:apply  Ctrl-R:reset  Esc:close",
        Style::default().fg(tc.muted),
    )));

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);

    // Draw cursor in the active field
    let field_row = match app.sim_field {
        SimulationField::Ram => 1,
        SimulationField::Vram => 2,
        SimulationField::CpuCores => 3,
    };
    let cursor_x = inner.x + 14 + app.sim_cursor_position as u16;
    let cursor_y = inner.y + field_row;
    if cursor_x < inner.x + inner.width {
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

fn draw_advanced_config_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    let area = frame.area();

    let popup_width = 52u16.min(area.width.saturating_sub(4));
    let popup_height = 16u16.min(area.height.saturating_sub(4));
    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(" Advanced Configuration ")
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    // Field definitions: (label, input_ref, field_type)
    let fields: Vec<(&str, &str, AdvConfigField)> = vec![
        (
            "  Efficiency:",
            &app.adv_config_efficiency_input,
            AdvConfigField::Efficiency,
        ),
        (
            "  GPU factor:",
            &app.adv_config_eff_factor_gpu,
            AdvConfigField::FactorGpu,
        ),
        (
            "  CPU Offload:",
            &app.adv_config_eff_factor_cpu_offload,
            AdvConfigField::FactorCpuOffload,
        ),
        (
            "  MoE Offload:",
            &app.adv_config_eff_factor_moe,
            AdvConfigField::FactorMoe,
        ),
        (
            "  Tensor Par:",
            &app.adv_config_eff_factor_tp,
            AdvConfigField::FactorTp,
        ),
        (
            "  CPU Only:",
            &app.adv_config_eff_factor_cpu_only,
            AdvConfigField::FactorCpuOnly,
        ),
        (
            "  Context cap:",
            &app.adv_config_context_cap_input,
            AdvConfigField::ContextCap,
        ),
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    for (label, value, field) in &fields {
        let is_active = app.adv_config_field == *field;
        let label_style = if is_active {
            Style::default().fg(tc.accent).bold()
        } else {
            Style::default().fg(tc.fg)
        };
        let value_style = if is_active {
            Style::default().fg(tc.fg).bg(tc.highlight_bg)
        } else {
            Style::default().fg(tc.fg)
        };

        let display_val = if value.is_empty() && is_active {
            "_".to_string()
        } else {
            format!("{:<16}", value)
        };

        lines.push(Line::from(vec![
            Span::styled(format!("{:<14}", label), label_style),
            Span::styled(display_val, value_style),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Enter:apply  Ctrl-R:reset  Esc:close",
        Style::default().fg(tc.muted),
    )));

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);

    // Draw cursor in the active field
    let field_row = match app.adv_config_field {
        AdvConfigField::Efficiency => 1,
        AdvConfigField::FactorGpu => 2,
        AdvConfigField::FactorCpuOffload => 3,
        AdvConfigField::FactorMoe => 4,
        AdvConfigField::FactorTp => 5,
        AdvConfigField::FactorCpuOnly => 6,
        AdvConfigField::ContextCap => 7,
    };
    let cursor_x = inner.x + 14 + app.adv_config_cursor_position as u16;
    let cursor_y = inner.y + field_row;
    if cursor_x < inner.x + inner.width {
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

// ---------------------------------------------------------------------------
// Download Manager view
// ---------------------------------------------------------------------------

fn draw_downloads(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // Active download
            Constraint::Length(3), // Config
            Constraint::Min(6),    // History
        ])
        .split(area);

    draw_dm_active(frame, app, chunks[0], tc);
    draw_dm_config(frame, app, chunks[1], tc);
    draw_dm_history(frame, app, chunks[2], tc);

    // Show delete confirmation overlay
    if app.dm_confirm_delete {
        let popup_width = 50u16.min(area.width.saturating_sub(4));
        let popup_height = 5u16;
        let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
        let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);
        frame.render_widget(Clear, popup_area);

        let model_name = app
            .download_history
            .records
            .get(app.dm_history_cursor)
            .map(|r| r.model_name.as_str())
            .unwrap_or("?");

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(tc.error))
            .title(" Confirm Delete ");
        let text = Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![
                Span::raw("  Delete "),
                Span::styled(model_name, Style::default().fg(tc.fg).bold()),
                Span::raw("? (y/n)"),
            ]),
        ])
        .block(block);
        frame.render_widget(text, popup_area);
    }

    // Show cursor when editing directory
    if app.dm_editing_dir {
        let inner = chunks[1].inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 1,
        });
        let cursor_x = inner.x + 14 + app.dm_dir_cursor as u16;
        let cursor_y = inner.y;
        if cursor_x < inner.x + inner.width {
            frame.set_cursor_position((cursor_x, cursor_y));
        }
    }
}

fn draw_dm_active(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let focused = app.dm_focus == DownloadManagerFocus::Active;
    let border_style = if focused {
        Style::default().fg(tc.accent)
    } else {
        Style::default().fg(tc.border)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Active Download ");

    if app.pull_active.is_some() {
        let model = app.pull_model_name.as_deref().unwrap_or("unknown");
        let status = app.pull_status.as_deref().unwrap_or("");
        let pct = app.pull_percent.unwrap_or(0.0);

        // Build a text-based progress bar
        let bar_width = area.width.saturating_sub(6) as usize;
        let filled = ((pct / 100.0) * bar_width as f64) as usize;
        let empty = bar_width.saturating_sub(filled);
        let bar = format!("[{}{}]", "█".repeat(filled), "░".repeat(empty));

        let lines = vec![
            Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(model, Style::default().fg(tc.fg).bold()),
            ]),
            Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(bar, Style::default().fg(tc.accent)),
                Span::styled(format!(" {:.0}%", pct), Style::default().fg(tc.fg)),
            ]),
            Line::from(Span::styled(
                format!("  {}", status),
                Style::default().fg(tc.muted),
            )),
        ];
        frame.render_widget(Paragraph::new(lines).block(block), area);
    } else {
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No active download",
                Style::default().fg(tc.muted),
            )),
        ];
        frame.render_widget(Paragraph::new(lines).block(block), area);
    }
}

fn draw_dm_config(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let focused = app.dm_focus == DownloadManagerFocus::Config;
    let border_style = if focused {
        Style::default().fg(tc.accent)
    } else {
        Style::default().fg(tc.border)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Config ");

    let dir_display = if app.dm_editing_dir {
        app.dm_dir_input.as_str()
    } else {
        // Show current models dir from the llamacpp provider
        // We use the public function as a fallback display
        ""
    };

    let line = if app.dm_editing_dir {
        Line::from(vec![
            Span::styled("  Models dir:  ", Style::default().fg(tc.muted)),
            Span::styled(dir_display, Style::default().fg(tc.fg)),
            Span::styled("█", Style::default().fg(tc.accent)),
        ])
    } else {
        Line::from(vec![
            Span::styled("  Models dir:  ", Style::default().fg(tc.muted)),
            Span::styled(
                app.llamacpp_models_dir().display().to_string(),
                Style::default().fg(tc.fg),
            ),
            if focused {
                Span::styled("  [e]dit", Style::default().fg(tc.accent))
            } else {
                Span::raw("")
            },
        ])
    };

    frame.render_widget(Paragraph::new(vec![line]).block(block), area);
}

fn draw_dm_history(frame: &mut Frame, app: &App, area: Rect, tc: &ThemeColors) {
    let focused = app.dm_focus == DownloadManagerFocus::History;
    let border_style = if focused {
        Style::default().fg(tc.accent)
    } else {
        Style::default().fg(tc.border)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(format!(
            " History ({}) ",
            app.download_history.records.len()
        ));

    if app.download_history.records.is_empty() {
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No download history",
                Style::default().fg(tc.muted),
            )),
        ];
        frame.render_widget(Paragraph::new(lines).block(block), area);
        return;
    }

    // Build table rows (newest first)
    let header = Row::new(vec![
        Cell::from("  Model").style(Style::default().fg(tc.accent).bold()),
        Cell::from("Provider").style(Style::default().fg(tc.accent).bold()),
        Cell::from("Status").style(Style::default().fg(tc.accent).bold()),
        Cell::from("Date").style(Style::default().fg(tc.accent).bold()),
    ]);

    let rows: Vec<Row> = app
        .download_history
        .records
        .iter()
        .rev()
        .enumerate()
        .map(|(display_idx, record)| {
            let (status_text, status_color) = match &record.result {
                DownloadResult::Success => ("✓ Done", tc.good),
                DownloadResult::Error(_) => ("✗ Error", tc.error),
            };
            let date = format_epoch(record.timestamp);

            let style = if display_idx == app.dm_history_cursor {
                Style::default().bg(tc.highlight_bg)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(format!("  {}", record.model_name)).style(Style::default().fg(tc.fg)),
                Cell::from(record.provider.clone()).style(Style::default().fg(tc.muted)),
                Cell::from(status_text).style(Style::default().fg(status_color)),
                Cell::from(date).style(Style::default().fg(tc.muted)),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Min(30),
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Length(12),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .highlight_symbol("▶ ");

    frame.render_widget(table, area);
}

/// Format epoch seconds as a simple date string.
fn format_epoch(epoch: u64) -> String {
    // Simple date formatting without external crate
    let secs_per_day: u64 = 86400;
    let days = epoch / secs_per_day;

    // Days since 1970-01-01
    let mut y = 1970i32;
    let mut remaining = days;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366u64
        } else {
            365
        };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0usize;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        m += 1;
    }
    format!("{:04}-{:02}-{:02}", y, m + 1, remaining + 1)
}

fn draw_filter_popup(frame: &mut Frame, app: &App, tc: &ThemeColors) {
    use crate::tui_app::{FilterPopupField, FitFilter};

    let area = frame.area();
    let popup_width = 56u16.min(area.width.saturating_sub(4));
    let popup_height = 18u16.min(area.height.saturating_sub(4));
    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tc.accent_secondary))
        .style(Style::default().bg(tc.bg))
        .title(" Filter [F] ")
        .title_style(
            Style::default()
                .fg(tc.accent_secondary)
                .add_modifier(Modifier::BOLD),
        );

    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let mut lines: Vec<Line> = Vec::new();

    // Helper closures
    let label_style = |active: bool| {
        if active {
            Style::default().fg(tc.accent).bold()
        } else {
            Style::default().fg(tc.fg)
        }
    };
    let value_style = |active: bool| {
        if active {
            Style::default().fg(tc.fg).bg(tc.highlight_bg)
        } else {
            Style::default().fg(tc.muted)
        }
    };

    // Parameters (B)
    lines.push(Line::from(Span::styled(
        "  Parameters (B):",
        Style::default().fg(tc.accent).bold(),
    )));

    let is_min = app.filter_field == FilterPopupField::ParamsMin;
    let min_val = if app.filter_params_min_input.is_empty() {
        "any".to_string()
    } else {
        app.filter_params_min_input.clone()
    };
    lines.push(Line::from(vec![
        Span::styled("    Min:", label_style(is_min)),
        Span::styled(format!(" {:>12}", min_val), value_style(is_min)),
    ]));

    let is_max = app.filter_field == FilterPopupField::ParamsMax;
    let max_val = if app.filter_params_max_input.is_empty() {
        "any".to_string()
    } else {
        app.filter_params_max_input.clone()
    };
    lines.push(Line::from(vec![
        Span::styled("    Max:", label_style(is_max)),
        Span::styled(format!(" {:>12}", max_val), value_style(is_max)),
    ]));

    // Memory Usage (%)
    lines.push(Line::from(Span::styled(
        "  Memory Usage (%):",
        Style::default().fg(tc.accent).bold(),
    )));

    let is_mem_min = app.filter_field == FilterPopupField::MemPctMin;
    let mem_min_val = if app.filter_mem_pct_min_input.is_empty() {
        "any".to_string()
    } else {
        format!("{}%", app.filter_mem_pct_min_input)
    };
    lines.push(Line::from(vec![
        Span::styled("    Min:", label_style(is_mem_min)),
        Span::styled(format!(" {:>12}", mem_min_val), value_style(is_mem_min)),
    ]));

    let is_mem_max = app.filter_field == FilterPopupField::MemPctMax;
    let mem_max_val = if app.filter_mem_pct_max_input.is_empty() {
        "any".to_string()
    } else {
        format!("{}%", app.filter_mem_pct_max_input)
    };
    lines.push(Line::from(vec![
        Span::styled("    Max:", label_style(is_mem_max)),
        Span::styled(format!(" {:>12}", mem_max_val), value_style(is_mem_max)),
    ]));

    // Sort Direction
    lines.push(Line::from(Span::styled(
        "  Sort:",
        Style::default().fg(tc.accent).bold(),
    )));

    let is_sort = app.filter_field == FilterPopupField::SortDirection;
    let dir_text = if app.filter_sort_ascending {
        "Ascending ↑"
    } else {
        "Descending ↓"
    };
    let sort_val_style = if is_sort {
        Style::default().fg(tc.info).bg(tc.highlight_bg)
    } else {
        Style::default().fg(tc.accent)
    };
    lines.push(Line::from(vec![
        Span::styled("    Direction:", label_style(is_sort)),
        Span::styled(format!(" {:>12}", dir_text), sort_val_style),
    ]));

    // Fit Filter
    lines.push(Line::from(Span::styled(
        "  Fit Filter:",
        Style::default().fg(tc.accent).bold(),
    )));

    let is_fit = app.filter_field == FilterPopupField::FitFilter;
    let fit_color = match app.fit_filter {
        FitFilter::All => tc.fg,
        FitFilter::Runnable | FitFilter::Perfect | FitFilter::TurboQuantFit => tc.good,
        FitFilter::Good => tc.warning,
        FitFilter::Marginal => tc.fit_marginal,
        FitFilter::TooTight => tc.error,
    };
    let fit_val_style = if is_fit {
        Style::default().fg(fit_color).bg(tc.highlight_bg)
    } else {
        Style::default().fg(fit_color)
    };
    lines.push(Line::from(vec![
        Span::styled("    Fit:", label_style(is_fit)),
        Span::styled(format!(" {:>12}", app.fit_filter.label()), fit_val_style),
    ]));

    // Footer
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Space:toggle  Ctrl-U:clear  Esc:cancel",
        Style::default().fg(tc.muted),
    )));

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);

    // Draw cursor for text input fields
    let field_row: u16 = match app.filter_field {
        FilterPopupField::ParamsMin => 1,
        FilterPopupField::ParamsMax => 2,
        FilterPopupField::MemPctMin => 4,
        FilterPopupField::MemPctMax => 5,
        FilterPopupField::SortDirection => 7,
        FilterPopupField::FitFilter => 9,
    };

    let label_width: u16 = 14;
    let cursor_x = inner.x + label_width + app.filter_cursor_position as u16 + 1;
    let cursor_y = inner.y + field_row;
    if cursor_x < inner.x + inner.width {
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}
