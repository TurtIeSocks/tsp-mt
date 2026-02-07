macro_rules! timed_pass {
    ($label:expr, $route:expr, $body:block) => {{
        let label = format!("{}", $label);
        let start = std::time::Instant::now();
        let start_thread = start;
        let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let running_thread = std::sync::Arc::clone(&running);
        let label_thread = label.clone();

        let handle = std::thread::spawn(move || {
            let mut stderr = std::io::stderr();
            while running_thread.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = std::io::Write::write_fmt(
                    &mut stderr,
                    format_args!(
                        "\r\x1b[2KRunning {}: {:.2}s",
                        label_thread,
                        start_thread.elapsed().as_secs_f32()
                    ),
                );
                let _ = std::io::Write::flush(&mut stderr);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });

        $body
        running.store(false, std::sync::atomic::Ordering::Relaxed);
        let _ = handle.join();

        let mut stderr = std::io::stderr();
        let _ = std::io::Write::write_fmt(&mut stderr, format_args!("\r\x1b[2K"));
        let _ = std::io::Write::flush(&mut stderr);

        eprintln!("Finished {}: {:.2}", label, start.elapsed().as_secs_f32());
        $crate::utils::measure_distance_open($route);
        eprintln!();
    }};
}

pub(crate) use timed_pass;
