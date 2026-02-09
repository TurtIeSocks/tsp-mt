use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

static SHUTDOWN_WORKDIRS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
static SHUTDOWN_HOOK_INSTALLED: OnceLock<()> = OnceLock::new();

fn shutdown_workdirs() -> &'static Mutex<HashSet<PathBuf>> {
    SHUTDOWN_WORKDIRS.get_or_init(|| Mutex::new(HashSet::new()))
}

fn install_shutdown_hook_once() {
    SHUTDOWN_HOOK_INSTALLED.get_or_init(|| {
        if let Err(err) = ctrlc::set_handler(|| {
            let work_dirs: Vec<PathBuf> = match shutdown_workdirs().lock() {
                Ok(guard) => guard.iter().cloned().collect(),
                Err(_) => Vec::new(),
            };

            for work_dir in work_dirs {
                cleanup_workdir(&work_dir);
            }
        }) {
            log::warn!("cleanup: failed to install shutdown hook err={err}");
        }
    });
}

pub fn register_workdir_for_shutdown_cleanup(work_dir: &Path) {
    install_shutdown_hook_once();
    if let Ok(mut guard) = shutdown_workdirs().lock() {
        guard.insert(work_dir.to_path_buf());
    }
}

fn unregister_workdir_for_shutdown_cleanup(work_dir: &Path) {
    if let Some(set) = SHUTDOWN_WORKDIRS.get()
        && let Ok(mut guard) = set.lock()
    {
        guard.remove(work_dir);
    }
}

pub fn cleanup_workdir(work_dir: &Path) {
    if !work_dir.exists() {
        unregister_workdir_for_shutdown_cleanup(work_dir);
        return;
    }

    if let Err(err) = fs::remove_dir_all(work_dir) {
        log::warn!(
            "cleanup: failed to remove workdir={} err={err}",
            work_dir.display()
        );
    } else {
        unregister_workdir_for_shutdown_cleanup(work_dir);
        log::debug!("cleanup: removed workdir={}", work_dir.display());
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::cleanup_workdir;

    fn unique_temp_dir(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("tsp-mt-tests-{name}-{nanos}"))
    }

    #[test]
    fn cleanup_workdir_removes_existing_directory() {
        let dir = unique_temp_dir("cleanup");
        fs::create_dir_all(&dir).expect("create temp dir");
        fs::write(dir.join("marker.txt"), b"ok").expect("write marker");

        cleanup_workdir(&dir);

        assert!(!dir.exists());
    }

    #[test]
    fn cleanup_workdir_ignores_missing_directory() {
        let dir = unique_temp_dir("missing");
        cleanup_workdir(&dir);
        assert!(!dir.exists());
    }
}
