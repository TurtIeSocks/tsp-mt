use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use crate::LKHNode;

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

pub(crate) fn register_workdir_for_shutdown_cleanup(work_dir: &Path) {
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

pub(crate) fn cleanup_workdir(work_dir: &Path) {
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

pub fn tour_distance(points: &[LKHNode]) -> (f64, f64, i32) {
    if points.len() < 2 {
        log::info!(
            "metrics: n={} total_m=0 longest_m=0 avg_m=0 spike_threshold_m=0 spikes=0",
            points.len()
        );
        return (0.0, 0.0, 0);
    }

    let mut total = 0.0;
    let mut longest = 0.0;
    let n = points.len();

    // OPEN: only edges i -> i+1
    for i in 0..(points.len() - 1) {
        let d = points[i].dist(&points[i + 1]);
        total += d;
        if d > longest {
            longest = d;
        }
    }
    let avg_edge = total / ((points.len() - 1) as f64);
    let threshold = avg_edge * 10.0;

    // Spike threshold: 10× average edge length (OPEN edges count = n-1)
    let mut spikes = 0;
    for i in 0..(points.len() - 1) {
        let d_l = points[i].dist(&points[(i + 1) % n]);
        if d_l > threshold {
            spikes += 1;
        }
        // let d_r = points[i].dist(&points[(i - 1) % n]);
        // if d_l > threshold && d_r > threshold {
        //     spikes += 1;
        // }
    }

    log::info!(
        "metrics: n={} total_m={total:.0} longest_m={longest:.0} avg_m={avg_edge:.0} spike_threshold_m={threshold:.0} spikes={spikes}",
        points.len()
    );

    (total, longest, spikes)
}
