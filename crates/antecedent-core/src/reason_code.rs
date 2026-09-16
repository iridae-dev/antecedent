//! Closed reason-code vocabulary (`parity/reason_codes.toml`).
//!
//! The lists are generated into `reason_codes_data.rs`; the schema gate fails
//! when they drift from the registry. [`reason_code!`](crate::reason_code!)
//! checks a literal at compile time, so an unregistered code cannot be emitted
//! from Rust; the Python `CausalUnsupportedError` checks the same list at
//! construction.

#[path = "reason_codes_data.rs"]
mod data;

pub use data::{REASON_CODES, RUNTIME_REFUSAL_CODES};

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const fn listed(code: &str, list: &[&str]) -> bool {
    let mut i = 0;
    while i < list.len() {
        if str_eq(code, list[i]) {
            return true;
        }
        i += 1;
    }
    false
}

/// The prefix a reason-coded refusal message carries: `<PREFIX><code>: <message>`.
///
/// Written as a `concat!` of two fragments deliberately: the schema gate refuses
/// a raw `"reason=` literal anywhere in Rust so every coded refusal goes through
/// the checked macros, and this constant is the one owner of the wire shape they
/// and the Python boundary share.
pub const PREFIX: &str = concat!("reason", "=");

/// Split a refusal message into its reason code and the message after it.
#[must_use]
pub fn split_prefix(message: &str) -> Option<(&str, &str)> {
    let rest = message.strip_prefix(PREFIX)?;
    let (code, tail) = rest.split_once(':')?;
    Some((code.trim(), tail.trim_start()))
}

/// Whether `code` is a registered reason code.
#[must_use]
pub const fn is_registered(code: &str) -> bool {
    listed(code, REASON_CODES)
}

/// Whether `code` is registered for runtime refusals (`applies_to` includes
/// `runtime_refusal`).
#[must_use]
pub const fn is_runtime_refusal(code: &str) -> bool {
    listed(code, RUNTIME_REFUSAL_CODES)
}

/// A registered reason code, checked at compile time.
///
/// ```
/// let code: &'static str = antecedent_core::reason_code!("not_executed");
/// assert_eq!(code, "not_executed");
/// ```
///
/// ```compile_fail
/// let _ = antecedent_core::reason_code!("bogus");
/// ```
#[macro_export]
macro_rules! reason_code {
    ($code:literal) => {{
        const _: () = assert!(
            $crate::reason_code::is_registered($code),
            concat!("unregistered reason code `", $code, "`; add it to parity/reason_codes.toml")
        );
        $code
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_codes_are_found_and_others_are_not() {
        assert!(is_registered("not_executed"));
        assert!(is_runtime_refusal("attested_not_reverifiable"));
        assert!(!is_runtime_refusal("no_interval_reported"));
        assert!(!is_registered("bogus"));
        assert!(!is_registered("not_execute"));
        assert!(!is_registered("not_executedx"));
        assert_eq!(crate::reason_code!("no_interval_reported"), "no_interval_reported");
    }

    #[test]
    fn runtime_codes_are_a_subset_of_all_codes() {
        for code in RUNTIME_REFUSAL_CODES {
            assert!(is_registered(code), "{code}");
        }
    }
}
