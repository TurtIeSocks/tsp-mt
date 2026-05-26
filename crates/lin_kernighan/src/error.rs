use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("solver: {0}")]
    Solver(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    pub fn invalid_data(message: impl Into<String>) -> Self {
        Self::InvalidData(message.into())
    }

    pub fn solver(message: impl Into<String>) -> Self {
        Self::Solver(message.into())
    }
}
