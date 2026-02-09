use thiserror::Error;

#[derive(Debug, Error)]
pub enum LkhError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("{context}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")]
    ProcessFailed {
        context: String,
        stdout: String,
        stderr: String,
    },
    #[error("{0}")]
    Other(String),
}

pub type LkhResult<T> = std::result::Result<T, LkhError>;

impl LkhError {
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
