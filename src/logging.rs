use std::io::{self, Write};

use env_logger::{Builder, Target, fmt::Formatter};
use log::Level;

use crate::options::{LogFormat, SolverOptions};

pub fn init_logger(options: &SolverOptions) -> io::Result<()> {
    let log_format = options.log_format;
    let log_timestamp = options.log_timestamp;

    let mut builder = Builder::new();
    builder
        .filter_level(options.log_level.to_filter())
        .write_style(env_logger::WriteStyle::Never)
        .target(Target::Stderr)
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

    builder.try_init().map_err(io::Error::other)
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
