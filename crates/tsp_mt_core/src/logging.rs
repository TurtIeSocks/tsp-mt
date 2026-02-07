use std::{fs::File, io::Write};

use env_logger::{Builder, Target, fmt::Formatter};
use log::Level;

use crate::Result;
use crate::options::{LogFormat, SolverOptions};

pub fn init_logger(options: &SolverOptions) -> Result<()> {
    let log_format = options.log_format;
    let log_timestamp = options.log_timestamp;

    let mut builder = Builder::new();
    builder
        .filter_level(options.log_level.to_filter())
        .write_style(env_logger::WriteStyle::Never)
        .format(move |buf: &mut Formatter, record| {
            if log_timestamp {
                write!(buf, "{} ", buf.timestamp_millis())?;
            }

            match log_format {
                LogFormat::Compact => {
                    writeln!(buf, "{} {}", level_tag(record.level()), record.args())
                }
                LogFormat::Pretty => {
                    writeln!(
                        buf,
                        "{} [{}] {}",
                        level_tag(record.level()),
                        record.target(),
                        record.args()
                    )
                }
            }
        });

    if let Some(log_path) = options.log_output_path() {
        let log_file = File::create(log_path).map_err(|e| {
            crate::Error::other(format!(
                "failed to create log output file {}: {e}",
                log_path.display()
            ))
        })?;
        builder.target(Target::Pipe(Box::new(log_file)));
    } else {
        builder.target(Target::Stderr);
    }

    builder
        .try_init()
        .map_err(|e| crate::Error::other(format!("logger init failed: {e}")))
}

fn level_tag(level: Level) -> &'static str {
    match level {
        Level::Error => "ERROR",
        Level::Warn => "WARN",
        Level::Info => "INFO",
        Level::Debug => "DEBUG",
        Level::Trace => "TRACE",
    }
}
