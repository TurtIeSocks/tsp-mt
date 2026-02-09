//! Error and result types used by the `lkh` crate.
//!
//! # Example
//!
//! ```no_run
//! use lkh::{LkhError, LkhResult};
//!
//! fn validate(value: i32) -> LkhResult<()> {
//!     if value < 0 {
//!         return Err(LkhError::invalid_input("value must be non-negative"));
//!     }
//!     Ok(())
//! }
//! ```
//!
use thiserror::Error;

/// Unified error type for this crate.
#[derive(Debug, Error)]
pub enum LkhError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("already assigned: {0}")]
    AlreadyAssigned(String),
    #[error("{context}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")]
    ProcessFailed {
        context: String,
        stdout: String,
        stderr: String,
    },
    #[error("{0}")]
    Other(String),
}

/// Convenient result alias for APIs in this crate.
pub type LkhResult<T> = std::result::Result<T, LkhError>;

impl LkhError {
    /// Constructs an `InvalidInput` error.
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Constructs an `InvalidData` error.
    pub fn invalid_data(message: impl Into<String>) -> Self {
        Self::InvalidData(message.into())
    }

    /// Constructs an `Other` error.
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }
}

#[macro_export]
macro_rules! with_methods_error {
    ($target:ident) => {
        impl From<$target> for $crate::error::LkhError {
            fn from(value: $target) -> Self {
                $crate::error::LkhError::AlreadyAssigned(value.to_string())
            }
        }
    };
}
