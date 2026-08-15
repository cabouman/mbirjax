//! Recently used dataset folders, persisted across sessions.
//!
//! One folder path per line, most recent first, capped at [`MAX_RECENT`].
//! Lives next to the shared theme preference
//! (`~/.config/venus_rust_tools/dehydration_hydration_recent`), so it follows
//! the user, not the machine or the working directory.

use std::path::{Path, PathBuf};

pub const MAX_RECENT: usize = 5;

/// The persistence file, under `$XDG_CONFIG_HOME` (or `~/.config`).
fn list_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join("venus_rust_tools").join("dehydration_hydration_recent"))
}

/// The saved list, most recent first. Missing/unreadable file → empty.
pub fn load() -> Vec<PathBuf> {
    let Some(path) = list_path() else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .take(MAX_RECENT)
        .collect()
}

/// Put `dir` at the front of the list (deduplicated, capped) and persist.
/// Returns the updated list. Best effort: an unwritable config directory
/// only costs the history, not an error dialog.
pub fn add(dir: &Path) -> Vec<PathBuf> {
    let mut list = load();
    list.retain(|p| p != dir);
    list.insert(0, dir.to_path_buf());
    list.truncate(MAX_RECENT);

    if let Some(path) = list_path() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let text: String = list
            .iter()
            .map(|p| format!("{}\n", p.display()))
            .collect();
        let _ = std::fs::write(path, text);
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Point XDG_CONFIG_HOME at a scratch dir so the tests never touch the
    /// real preference file. Serialized because the env var is process-wide.
    fn with_tmp_config<F: FnOnce()>(tag: &str, f: F) {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("dehydration_recent_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let old = std::env::var_os("XDG_CONFIG_HOME");
        unsafe { std::env::set_var("XDG_CONFIG_HOME", &dir) };
        f();
        match old {
            Some(v) => unsafe { std::env::set_var("XDG_CONFIG_HOME", v) },
            None => unsafe { std::env::remove_var("XDG_CONFIG_HOME") },
        }
    }

    #[test]
    fn add_dedupes_and_caps_at_five() {
        with_tmp_config("cap", || {
            for name in ["a", "b", "c", "d", "e", "f"] {
                add(Path::new(name));
            }
            // "b" again: moves to the front without duplicating.
            let list = add(Path::new("b"));
            assert_eq!(
                list,
                ["b", "f", "e", "d", "c"]
                    .iter()
                    .map(PathBuf::from)
                    .collect::<Vec<_>>()
            );
            assert_eq!(load(), list, "list must persist");
        });
    }

    #[test]
    fn load_without_file_is_empty() {
        with_tmp_config("empty", || {
            assert!(load().is_empty());
        });
    }
}
