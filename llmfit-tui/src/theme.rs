use ratatui::style::Color;
use std::fs;
use std::path::PathBuf;

/// Available color themes for the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Default,
    IndustrialDark,
    IndustrialLight,
    Nord,
    Gruvbox,
    Solarized,
}

impl Theme {
    pub fn label(&self) -> &'static str {
        match self {
            Theme::Default => "Default",
            Theme::IndustrialDark => "Industrial Dark",
            Theme::IndustrialLight => "Industrial Light",
            Theme::Nord => "Nord",
            Theme::Gruvbox => "Gruvbox",
            Theme::Solarized => "Solarized",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            Theme::Default => Theme::IndustrialDark,
            Theme::IndustrialDark => Theme::IndustrialLight,
            Theme::IndustrialLight => Theme::Nord,
            Theme::Nord => Theme::Gruvbox,
            Theme::Gruvbox => Theme::Solarized,
            Theme::Solarized => Theme::Default,
        }
    }

    pub fn colors(&self) -> ThemeColors {
        match self {
            Theme::Default => default_colors(),
            Theme::IndustrialDark => industrial_dark_colors(),
            Theme::IndustrialLight => industrial_light_colors(),
            Theme::Nord => nord_colors(),
            Theme::Gruvbox => gruvbox_colors(),
            Theme::Solarized => solarized_colors(),
        }
    }

    /// Path to the config file: ~/.config/llmfit/theme
    fn config_path() -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()?;
        Some(
            PathBuf::from(home)
                .join(".config")
                .join("llmfit")
                .join("theme"),
        )
    }

    /// Save the current theme to disk.
    pub fn save(&self) {
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(&path, self.label());
        }
    }

    /// Load the saved theme from disk, falling back to Default.
    pub fn load() -> Self {
        Self::config_path()
            .and_then(|path| fs::read_to_string(path).ok())
            .map(|s| Self::from_label(s.trim()))
            .unwrap_or(Theme::Default)
    }

    fn from_label(s: &str) -> Self {
        match s {
            "Industrial Dark" => Theme::IndustrialDark,
            "Industrial Light" => Theme::IndustrialLight,
            "Nord" => Theme::Nord,
            "Gruvbox" => Theme::Gruvbox,
            "Solarized" => Theme::Solarized,
            _ => Theme::Default,
        }
    }
}

/// All semantic colors used throughout the TUI, mapped from each theme.
pub struct ThemeColors {
    // General
    pub bg: Color,
    pub fg: Color,
    pub muted: Color,
    pub border: Color,
    pub title: Color,
    pub highlight_bg: Color,

    // Accent colors
    pub accent: Color,
    pub accent_secondary: Color,

    // Status colors
    pub good: Color,
    pub warning: Color,
    pub error: Color,
    pub info: Color,

    // Score colors
    pub score_high: Color,
    pub score_mid: Color,
    pub score_low: Color,

    // Fit levels
    pub fit_perfect: Color,
    pub fit_good: Color,
    pub fit_marginal: Color,
    pub fit_tight: Color,

    // Run modes
    pub mode_gpu: Color,
    pub mode_moe: Color,
    pub mode_offload: Color,
    pub mode_cpu: Color,

    // Status bar
    pub status_bg: Color,
    pub status_fg: Color,
}

fn default_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Reset,
        fg: Color::Reset,
        muted: Color::DarkGray,
        border: Color::DarkGray,
        title: Color::Rgb(245, 158, 11),
        highlight_bg: Color::Reset,

        accent: Color::Rgb(245, 158, 11),
        accent_secondary: Color::Rgb(16, 185, 129),

        good: Color::Rgb(16, 185, 129),
        warning: Color::Rgb(245, 158, 11),
        error: Color::Rgb(239, 68, 68),
        info: Color::Rgb(14, 165, 233),

        score_high: Color::Rgb(16, 185, 129),
        score_mid: Color::Rgb(245, 158, 11),
        score_low: Color::Rgb(239, 68, 68),

        fit_perfect: Color::Rgb(16, 185, 129),
        fit_good: Color::Rgb(14, 165, 233),
        fit_marginal: Color::Rgb(245, 158, 11),
        fit_tight: Color::Rgb(239, 68, 68),

        mode_gpu: Color::Rgb(16, 185, 129),
        mode_moe: Color::Rgb(14, 165, 233),
        mode_offload: Color::Rgb(245, 158, 11),
        mode_cpu: Color::DarkGray,

        status_bg: Color::Rgb(245, 158, 11),
        status_fg: Color::Black,
    }
}

fn industrial_dark_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(9, 9, 11),
        fg: Color::Rgb(250, 250, 250),
        muted: Color::Rgb(113, 113, 122),
        border: Color::Rgb(39, 39, 42),
        title: Color::Rgb(245, 158, 11),
        highlight_bg: Color::Rgb(39, 39, 42),

        accent: Color::Rgb(245, 158, 11),
        accent_secondary: Color::Rgb(217, 119, 6),

        good: Color::Rgb(16, 185, 129),
        warning: Color::Rgb(245, 158, 11),
        error: Color::Rgb(239, 68, 68),
        info: Color::Rgb(14, 165, 233),

        score_high: Color::Rgb(16, 185, 129),
        score_mid: Color::Rgb(245, 158, 11),
        score_low: Color::Rgb(239, 68, 68),

        fit_perfect: Color::Rgb(16, 185, 129),
        fit_good: Color::Rgb(14, 165, 233),
        fit_marginal: Color::Rgb(245, 158, 11),
        fit_tight: Color::Rgb(239, 68, 68),

        mode_gpu: Color::Rgb(16, 185, 129),
        mode_moe: Color::Rgb(14, 165, 233),
        mode_offload: Color::Rgb(217, 119, 6),
        mode_cpu: Color::Rgb(113, 113, 122),

        status_bg: Color::Rgb(245, 158, 11),
        status_fg: Color::Rgb(9, 9, 11),
    }
}

fn industrial_light_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(255, 255, 255),
        fg: Color::Rgb(9, 9, 11),
        muted: Color::Rgb(161, 161, 170),
        border: Color::Rgb(228, 228, 231),
        title: Color::Rgb(217, 119, 6),
        highlight_bg: Color::Rgb(228, 228, 231),

        accent: Color::Rgb(217, 119, 6),
        accent_secondary: Color::Rgb(245, 158, 11),

        good: Color::Rgb(5, 150, 105),
        warning: Color::Rgb(217, 119, 6),
        error: Color::Rgb(220, 38, 38),
        info: Color::Rgb(2, 132, 199),

        score_high: Color::Rgb(5, 150, 105),
        score_mid: Color::Rgb(217, 119, 6),
        score_low: Color::Rgb(220, 38, 38),

        fit_perfect: Color::Rgb(5, 150, 105),
        fit_good: Color::Rgb(2, 132, 199),
        fit_marginal: Color::Rgb(217, 119, 6),
        fit_tight: Color::Rgb(220, 38, 38),

        mode_gpu: Color::Rgb(5, 150, 105),
        mode_moe: Color::Rgb(2, 132, 199),
        mode_offload: Color::Rgb(217, 119, 6),
        mode_cpu: Color::Rgb(161, 161, 170),

        status_bg: Color::Rgb(217, 119, 6),
        status_fg: Color::Rgb(255, 255, 255),
    }
}

fn nord_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(46, 52, 64),
        fg: Color::Rgb(216, 222, 233),
        muted: Color::Rgb(76, 86, 106),
        border: Color::Rgb(67, 76, 94),
        title: Color::Rgb(163, 190, 140),
        highlight_bg: Color::Rgb(59, 66, 82),

        accent: Color::Rgb(136, 192, 208),
        accent_secondary: Color::Rgb(235, 203, 139),

        good: Color::Rgb(163, 190, 140),
        warning: Color::Rgb(235, 203, 139),
        error: Color::Rgb(191, 97, 106),
        info: Color::Rgb(136, 192, 208),

        score_high: Color::Rgb(163, 190, 140),
        score_mid: Color::Rgb(235, 203, 139),
        score_low: Color::Rgb(191, 97, 106),

        fit_perfect: Color::Rgb(163, 190, 140),
        fit_good: Color::Rgb(136, 192, 208),
        fit_marginal: Color::Rgb(235, 203, 139),
        fit_tight: Color::Rgb(191, 97, 106),

        mode_gpu: Color::Rgb(163, 190, 140),
        mode_moe: Color::Rgb(136, 192, 208),
        mode_offload: Color::Rgb(235, 203, 139),
        mode_cpu: Color::Rgb(76, 86, 106),

        status_bg: Color::Rgb(129, 161, 193),
        status_fg: Color::Rgb(46, 52, 64),
    }
}

fn gruvbox_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(40, 40, 40),
        fg: Color::Rgb(235, 219, 178),
        muted: Color::Rgb(146, 131, 116),
        border: Color::Rgb(80, 73, 69),
        title: Color::Rgb(184, 187, 38),
        highlight_bg: Color::Rgb(60, 56, 54),

        accent: Color::Rgb(131, 165, 152),
        accent_secondary: Color::Rgb(250, 189, 47),

        good: Color::Rgb(184, 187, 38),
        warning: Color::Rgb(250, 189, 47),
        error: Color::Rgb(251, 73, 52),
        info: Color::Rgb(131, 165, 152),

        score_high: Color::Rgb(184, 187, 38),
        score_mid: Color::Rgb(250, 189, 47),
        score_low: Color::Rgb(251, 73, 52),

        fit_perfect: Color::Rgb(184, 187, 38),
        fit_good: Color::Rgb(131, 165, 152),
        fit_marginal: Color::Rgb(250, 189, 47),
        fit_tight: Color::Rgb(251, 73, 52),

        mode_gpu: Color::Rgb(184, 187, 38),
        mode_moe: Color::Rgb(131, 165, 152),
        mode_offload: Color::Rgb(250, 189, 47),
        mode_cpu: Color::Rgb(146, 131, 116),

        status_bg: Color::Rgb(214, 93, 14),
        status_fg: Color::Rgb(40, 40, 40),
    }
}

fn solarized_colors() -> ThemeColors {
    ThemeColors {
        bg: Color::Rgb(0, 43, 54),
        fg: Color::Rgb(131, 148, 150),
        muted: Color::Rgb(88, 110, 117),
        border: Color::Rgb(88, 110, 117),
        title: Color::Rgb(133, 153, 0),
        highlight_bg: Color::Rgb(7, 54, 66),

        accent: Color::Rgb(38, 139, 210),
        accent_secondary: Color::Rgb(181, 137, 0),

        good: Color::Rgb(133, 153, 0),
        warning: Color::Rgb(181, 137, 0),
        error: Color::Rgb(220, 50, 47),
        info: Color::Rgb(38, 139, 210),

        score_high: Color::Rgb(133, 153, 0),
        score_mid: Color::Rgb(181, 137, 0),
        score_low: Color::Rgb(220, 50, 47),

        fit_perfect: Color::Rgb(133, 153, 0),
        fit_good: Color::Rgb(38, 139, 210),
        fit_marginal: Color::Rgb(181, 137, 0),
        fit_tight: Color::Rgb(220, 50, 47),

        mode_gpu: Color::Rgb(133, 153, 0),
        mode_moe: Color::Rgb(42, 161, 152),
        mode_offload: Color::Rgb(181, 137, 0),
        mode_cpu: Color::Rgb(88, 110, 117),

        status_bg: Color::Rgb(38, 139, 210),
        status_fg: Color::Rgb(253, 246, 227),
    }
}
