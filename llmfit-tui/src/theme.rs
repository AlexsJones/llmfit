use ratatui::style::Color;
use std::fs;
use std::path::PathBuf;

/// Available color themes for the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Default,
    Dracula,
    Solarized,
    Nord,
    Monokai,
    Gruvbox,
}

impl Theme {
    pub fn label(&self) -> &'static str {
        match self {
            Theme::Default => "Default",
            Theme::Dracula => "Dracula",
            Theme::Solarized => "Solarized",
            Theme::Nord => "Nord",
            Theme::Monokai => "Monokai",
            Theme::Gruvbox => "Gruvbox",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            Theme::Default => Theme::Dracula,
            Theme::Dracula => Theme::Solarized,
            Theme::Solarized => Theme::Nord,
            Theme::Nord => Theme::Monokai,
            Theme::Monokai => Theme::Gruvbox,
            Theme::Gruvbox => Theme::Default,
        }
    }

    pub fn colors(&self) -> ThemeColors {
        match self {
            Theme::Default => default_colors(),
            Theme::Dracula => dracula_colors(),
            Theme::Solarized => solarized_colors(),
            Theme::Nord => nord_colors(),
            Theme::Monokai => monokai_colors(),
            Theme::Gruvbox => gruvbox_colors(),
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
            "Dracula" => Theme::Dracula,
            "Solarized" => Theme::Solarized,
            "Nord" => Theme::Nord,
            "Monokai" => Theme::Monokai,
            "Gruvbox" => Theme::Gruvbox,
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
        fg: Color::White,
        muted: Color::DarkGray,
        border: Color::DarkGray,
        title: Color::Green,
        highlight_bg: Color::Rgb(40, 40, 70),

        accent: Color::Cyan,
        accent_secondary: Color::Yellow,

        good: Color::Green,
        warning: Color::Yellow,
        error: Color::Red,
        info: Color::Cyan,

        score_high: Color::Green,
        score_mid: Color::Yellow,
        score_low: Color::Red,

        fit_perfect: Color::Green,
        fit_good: Color::Yellow,
        fit_marginal: Color::Magenta,
        fit_tight: Color::Red,

        mode_gpu: Color::Green,
        mode_moe: Color::Cyan,
        mode_offload: Color::Yellow,
        mode_cpu: Color::DarkGray,

        status_bg: Color::Green,
        status_fg: Color::Black,
    }
}

fn dracula_colors() -> ThemeColors {
    // Dracula: dark purple bg, pastel accents
    ThemeColors {
        bg: Color::Rgb(40, 42, 54),
        fg: Color::Rgb(248, 248, 242),
        muted: Color::Rgb(98, 114, 164),
        border: Color::Rgb(68, 71, 90),
        title: Color::Rgb(80, 250, 123),
        highlight_bg: Color::Rgb(68, 71, 90),

        accent: Color::Rgb(139, 233, 253),
        accent_secondary: Color::Rgb(241, 250, 140),

        good: Color::Rgb(80, 250, 123),
        warning: Color::Rgb(241, 250, 140),
        error: Color::Rgb(255, 85, 85),
        info: Color::Rgb(139, 233, 253),

        score_high: Color::Rgb(80, 250, 123),
        score_mid: Color::Rgb(241, 250, 140),
        score_low: Color::Rgb(255, 85, 85),

        fit_perfect: Color::Rgb(80, 250, 123),
        fit_good: Color::Rgb(241, 250, 140),
        fit_marginal: Color::Rgb(189, 147, 249),
        fit_tight: Color::Rgb(255, 85, 85),

        mode_gpu: Color::Rgb(80, 250, 123),
        mode_moe: Color::Rgb(139, 233, 253),
        mode_offload: Color::Rgb(241, 250, 140),
        mode_cpu: Color::Rgb(98, 114, 164),

        status_bg: Color::Rgb(189, 147, 249),
        status_fg: Color::Rgb(40, 42, 54),
    }
}

fn solarized_colors() -> ThemeColors {
    // Solarized Dark
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
        fit_good: Color::Rgb(181, 137, 0),
        fit_marginal: Color::Rgb(211, 54, 130),
        fit_tight: Color::Rgb(220, 50, 47),

        mode_gpu: Color::Rgb(133, 153, 0),
        mode_moe: Color::Rgb(42, 161, 152),
        mode_offload: Color::Rgb(181, 137, 0),
        mode_cpu: Color::Rgb(88, 110, 117),

        status_bg: Color::Rgb(38, 139, 210),
        status_fg: Color::Rgb(253, 246, 227),
    }
}

fn nord_colors() -> ThemeColors {
    // Nord: cool blue-gray palette
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
        fit_good: Color::Rgb(235, 203, 139),
        fit_marginal: Color::Rgb(180, 142, 173),
        fit_tight: Color::Rgb(191, 97, 106),

        mode_gpu: Color::Rgb(163, 190, 140),
        mode_moe: Color::Rgb(136, 192, 208),
        mode_offload: Color::Rgb(235, 203, 139),
        mode_cpu: Color::Rgb(76, 86, 106),

        status_bg: Color::Rgb(129, 161, 193),
        status_fg: Color::Rgb(46, 52, 64),
    }
}

fn monokai_colors() -> ThemeColors {
    // Monokai Pro
    ThemeColors {
        bg: Color::Rgb(39, 40, 34),
        fg: Color::Rgb(248, 248, 242),
        muted: Color::Rgb(117, 113, 94),
        border: Color::Rgb(73, 72, 62),
        title: Color::Rgb(166, 226, 46),
        highlight_bg: Color::Rgb(73, 72, 62),

        accent: Color::Rgb(102, 217, 239),
        accent_secondary: Color::Rgb(230, 219, 116),

        good: Color::Rgb(166, 226, 46),
        warning: Color::Rgb(230, 219, 116),
        error: Color::Rgb(249, 38, 114),
        info: Color::Rgb(102, 217, 239),

        score_high: Color::Rgb(166, 226, 46),
        score_mid: Color::Rgb(230, 219, 116),
        score_low: Color::Rgb(249, 38, 114),

        fit_perfect: Color::Rgb(166, 226, 46),
        fit_good: Color::Rgb(230, 219, 116),
        fit_marginal: Color::Rgb(174, 129, 255),
        fit_tight: Color::Rgb(249, 38, 114),

        mode_gpu: Color::Rgb(166, 226, 46),
        mode_moe: Color::Rgb(102, 217, 239),
        mode_offload: Color::Rgb(230, 219, 116),
        mode_cpu: Color::Rgb(117, 113, 94),

        status_bg: Color::Rgb(253, 151, 31),
        status_fg: Color::Rgb(39, 40, 34),
    }
}

fn gruvbox_colors() -> ThemeColors {
    // Gruvbox Dark
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
        fit_good: Color::Rgb(250, 189, 47),
        fit_marginal: Color::Rgb(211, 134, 155),
        fit_tight: Color::Rgb(251, 73, 52),

        mode_gpu: Color::Rgb(184, 187, 38),
        mode_moe: Color::Rgb(131, 165, 152),
        mode_offload: Color::Rgb(250, 189, 47),
        mode_cpu: Color::Rgb(146, 131, 116),

        status_bg: Color::Rgb(214, 93, 14),
        status_fg: Color::Rgb(40, 40, 40),
    }
}
