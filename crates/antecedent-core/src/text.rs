//! Small text predicates shared by the validators of every crate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Whether `text` is empty or only whitespace.
///
/// The one definition of "blank" a declaration's identifier, label or
/// justification is checked with, so no two validators disagree on it.
#[must_use]
pub fn is_blank(text: &str) -> bool {
    text.trim().is_empty()
}

#[cfg(test)]
mod tests {
    use super::is_blank;

    #[test]
    fn blank_is_empty_or_whitespace_only() {
        assert!(is_blank(""));
        assert!(is_blank(" \t\n"));
        assert!(!is_blank("a"));
        assert!(!is_blank(" a "));
    }
}
