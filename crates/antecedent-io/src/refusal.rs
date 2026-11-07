//! Shared construction of code/detail/message IO refusals.

use crate::IoError;

pub(crate) fn refused(code: &'static str, detail: &str, message: &str) -> IoError {
    IoError::Refused { code, message: format!("{detail}: {message}") }
}
