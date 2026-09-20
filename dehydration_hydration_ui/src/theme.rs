//! Light / dark theme preference, shared by every VENUS rust tool.
//!
//! The preference lives in one file (`~/.config/venus_rust_tools/theme`,
//! containing `dark` or `light`) so switching the theme in any of the tools
//! switches all of them — the next time each one starts. Dark is the default:
//! it is what every tool shipped with before the preference existed.
//!
//! This module is deliberately self-contained (egui + std only) so it can be
//! copied verbatim into the other tools' crates.

use std::path::PathBuf;

/// The preference file, under `$XDG_CONFIG_HOME` (or `~/.config`).
fn pref_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
        })?;
    Some(base.join("venus_rust_tools").join("theme"))
}

/// The saved preference, or dark when there is none (or it is unreadable).
pub fn load() -> egui::Theme {
    match pref_path().and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(s) if s.trim().eq_ignore_ascii_case("light") => egui::Theme::Light,
        _ => egui::Theme::Dark,
    }
}

/// Persist the preference. Best effort: a read-only home directory only
/// costs the user their choice on the next start, not an error dialog.
pub fn save(theme: egui::Theme) {
    let Some(path) = pref_path() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(
        path,
        match theme {
            egui::Theme::Light => "light\n",
            egui::Theme::Dark => "dark\n",
        },
    );
}

/// A sun / moon button that flips the theme of the whole application and
/// saves the choice for every VENUS rust tool. Drop it anywhere in a toolbar.
pub fn toggle_button(ui: &mut egui::Ui) {
    let (icon, tip, next) = match ui.ctx().theme() {
        egui::Theme::Dark => ("☀", "Switch to the light theme", egui::Theme::Light),
        egui::Theme::Light => ("🌙", "Switch to the dark theme", egui::Theme::Dark),
    };
    if ui
        .button(icon)
        .on_hover_text(format!("{tip} (applies to all the VENUS rust tools)"))
        .clicked()
    {
        ui.ctx().set_theme(next);
        save(next);
    }
}
