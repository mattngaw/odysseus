use std::path::Path;

/// Linux comm can be available when the exe symlink is access-restricted.
/// It is a short task label, not an executable path or command line.
pub(super) fn kernel_process_name(comm: &str) -> Option<String> {
    executable_name(Path::new(comm.trim_end_matches('\n')))
}

pub(super) fn executable_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let name = name.strip_suffix(" (deleted)").unwrap_or(name);
    let safe: String = name
        .chars()
        .take(64)
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+') {
                c
            } else {
                '_'
            }
        })
        .collect();
    (!safe.is_empty()).then_some(safe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_directories_and_controls_from_executable_names() {
        assert_eq!(
            executable_name(Path::new("/home/private-user/env/bin/python3.12")),
            Some("python3.12".into())
        );
        assert_eq!(
            executable_name(Path::new("/tmp/engine (deleted)")),
            Some("engine".into())
        );
        assert_eq!(
            executable_name(Path::new("/tmp/<name>\n\u{1b}")),
            Some("_name___".into())
        );
        assert!(executable_name(Path::new("/")).is_none());
        assert_eq!(
            executable_name(Path::new(&"x".repeat(200))).unwrap().len(),
            64
        );
    }

    #[test]
    fn kernel_labels_drop_the_proc_newline_and_apply_the_same_public_filter() {
        assert_eq!(kernel_process_name("python\n"), Some("python".into()));
        assert_eq!(kernel_process_name("/tmp/<task>\n"), Some("_task_".into()));
        assert_eq!(kernel_process_name("\n"), None);
    }
}
