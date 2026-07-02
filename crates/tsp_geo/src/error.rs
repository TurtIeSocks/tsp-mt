use alloc::string::String;

use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[cfg(feature = "std")]
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = core::result::Result<T, Error>;

impl Error {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    pub fn invalid_data(message: impl Into<String>) -> Self {
        Self::InvalidData(message.into())
    }

    pub fn other(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }
}
