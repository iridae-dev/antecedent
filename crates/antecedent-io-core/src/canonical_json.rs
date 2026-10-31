//! The one canonical digest of a JSON value, shared with host languages.
//!
//! A host builds a JSON document for something it cannot hash natively (a
//! provider spec, a request mapping) and asks Rust for its digest, so no host
//! language owns an encoding. The digest is a domain-separated BLAKE3 stream
//! through [`antecedent_core::CanonicalWriter`]: object keys are sorted, every
//! value carries a type tag, and an integer `1` differs from a float `1.0`.
//! It is independent of whitespace, key order and the host's number formatting.

use antecedent_core::CanonicalWriter;
use serde_json::Value;

use crate::error::IoError;

/// Largest JSON document hashed, in bytes.
pub const MAX_CANONICAL_JSON_BYTES: usize = 16 * 1024 * 1024;
/// Default deepest nesting hashed.
pub const MAX_CANONICAL_JSON_DEPTH: usize = 64;

/// Bounds on a document handed to the canonical digest. Both stay refusals; the defaults are
/// [`MAX_CANONICAL_JSON_BYTES`] and [`MAX_CANONICAL_JSON_DEPTH`]. Documents parsed from text
/// are additionally subject to the JSON parser's own recursion limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalJsonLimits {
    /// Largest document, in bytes.
    pub max_bytes: usize,
    /// Deepest nesting.
    pub max_depth: usize,
}

impl Default for CanonicalJsonLimits {
    fn default() -> Self {
        Self { max_bytes: MAX_CANONICAL_JSON_BYTES, max_depth: MAX_CANONICAL_JSON_DEPTH }
    }
}

fn write(
    stream: &mut CanonicalWriter,
    value: &Value,
    depth: usize,
    max_depth: usize,
) -> Result<(), IoError> {
    if depth > max_depth {
        return Err(IoError::Convert("canonical JSON nests too deeply".into()));
    }
    match value {
        Value::Null => {
            stream.word(0);
        }
        Value::Bool(flag) => {
            stream.word(1).word(u64::from(*flag));
        }
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                stream.word(2).word(u64::from_le_bytes(int.to_le_bytes()));
            } else if let Some(uint) = number.as_u64() {
                stream.word(3).word(uint);
            } else if let Some(real) = number.as_f64().filter(|real| real.is_finite()) {
                stream.word(4).real(real);
            } else {
                return Err(IoError::Convert("canonical JSON number is not finite".into()));
            }
        }
        Value::String(text) => {
            stream.word(5).text(text);
        }
        Value::Array(items) => {
            stream.word(6).word(items.len() as u64);
            for item in items {
                write(stream, item, depth + 1, max_depth)?;
            }
        }
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            stream.word(7).word(entries.len() as u64);
            for (key, item) in entries {
                stream.text(key);
                write(stream, item, depth + 1, max_depth)?;
            }
        }
    }
    Ok(())
}

// serde_json represents an integer outside i64/u64 as f64. Reject its raw
// integer token before parsing, since rounding would merge different inputs.
// Strings (including escaped quotes) are skipped; malformed numeric tokens
// and other syntax are diagnosed by serde_json. Decimal/exponent tokens retain
// the existing f64 path.
fn reject_out_of_range_integer_tokens(json: &str) -> Result<(), IoError> {
    let bytes = json.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                i += 1;
                while i < bytes.len() {
                    match bytes[i] {
                        b'\\' => i = i.saturating_add(2),
                        b'"' => {
                            i += 1;
                            break;
                        }
                        _ => i += 1,
                    }
                }
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                i += 1;
                while i < bytes.len()
                    && !matches!(
                        bytes[i],
                        b' ' | b'\t' | b'\r' | b'\n' | b',' | b':' | b'[' | b']' | b'{' | b'}'
                    )
                {
                    i += 1;
                }
                let token = &bytes[start..i];
                let digits = token.strip_prefix(b"-").unwrap_or(token);
                let boundary = start == 0
                    || matches!(
                        bytes[start - 1],
                        b' ' | b'\t' | b'\r' | b'\n' | b',' | b':' | b'[' | b'{'
                    );
                let integer = boundary
                    && !digits.is_empty()
                    && digits.iter().all(u8::is_ascii_digit)
                    && (digits.len() == 1 || digits[0] != b'0');
                if integer {
                    // The token contains only ASCII digits and an optional leading minus.
                    let token = &json[start..i];
                    let valid = if token.starts_with('-') {
                        token.parse::<i64>().is_ok()
                    } else {
                        token.parse::<u64>().is_ok()
                    };
                    if !valid {
                        return Err(IoError::Convert(
                            "canonical JSON integer is outside i64/u64 range".into(),
                        ));
                    }
                }
            }
            _ => i += 1,
        }
    }
    Ok(())
}

/// Digest of `json` under `domain`, as 64 lowercase hex characters.
///
/// # Errors
/// A blank domain, an oversized or unparseable document, a non-finite number,
/// excessive nesting, or an integer outside i64/u64 refuses.
pub fn canonical_json_digest(domain: &str, json: &str) -> Result<String, IoError> {
    canonical_json_digest_with(domain, json, &CanonicalJsonLimits::default())
}

/// [`canonical_json_digest`] under the caller's bounds.
///
/// # Errors
/// As [`canonical_json_digest`], with `limits` in place of the defaults.
pub fn canonical_json_digest_with(
    domain: &str,
    json: &str,
    limits: &CanonicalJsonLimits,
) -> Result<String, IoError> {
    if json.len() > limits.max_bytes {
        return Err(IoError::TooLarge);
    }
    reject_out_of_range_integer_tokens(json)?;
    let value: Value = serde_json::from_str(json)
        .map_err(|error| IoError::Convert(format!("canonical JSON is invalid: {error}")))?;
    canonical_json_value_digest_with(domain, &value, limits)
}

/// [`canonical_json_digest`] of a value already parsed, so a Rust caller digests
/// what a host language would digest from the same document. A value parsed
/// elsewhere may already have lost the distinction between out-of-range integers;
/// use [`canonical_json_digest`] for raw JSON text.
///
/// # Errors
/// A blank domain, a non-finite number or excessive nesting refuses.
pub fn canonical_json_value_digest(domain: &str, value: &Value) -> Result<String, IoError> {
    canonical_json_value_digest_with(domain, value, &CanonicalJsonLimits::default())
}

/// [`canonical_json_value_digest`] under the caller's bounds.
///
/// # Errors
/// As [`canonical_json_value_digest`], with `limits` in place of the defaults.
pub fn canonical_json_value_digest_with(
    domain: &str,
    value: &Value,
    limits: &CanonicalJsonLimits,
) -> Result<String, IoError> {
    if domain.trim().is_empty() {
        return Err(IoError::Convert("canonical digest needs a domain".into()));
    }
    let mut stream = CanonicalWriter::new(domain);
    write(&mut stream, value, 0, limits.max_depth)?;
    Ok(stream.finish_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_ignores_key_order_and_whitespace_but_not_value_type_or_domain() {
        let a = canonical_json_digest("d", r#"{"b": [1, 2.5], "a": "x"}"#).unwrap();
        let b = canonical_json_digest("d", r#"{"a":"x","b":[1,2.5]}"#).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert_ne!(a, canonical_json_digest("e", r#"{"a":"x","b":[1,2.5]}"#).unwrap());
        assert_ne!(
            canonical_json_digest("d", "[1]").unwrap(),
            canonical_json_digest("d", "[1.0]").unwrap()
        );
        assert_ne!(
            canonical_json_digest("d", r#"{"a":"1"}"#).unwrap(),
            canonical_json_digest("d", r#"{"a":1}"#).unwrap()
        );
    }

    #[test]
    fn raw_integer_range_is_checked_before_parsing_and_valid_encoding_is_unchanged() {
        for json in ["18446744073709551616", "18446744073709551617", "-9223372036854775809"] {
            assert!(canonical_json_digest("d", json).is_err(), "{json}");
        }
        for json in ["1-2", "--1", "1+2", "01", "+18446744073709551616"] {
            assert!(
                matches!(
                    canonical_json_digest("d", json),
                    Err(IoError::Convert(message)) if message.starts_with("canonical JSON is invalid:")
                ),
                "{json}"
            );
        }
        for json in ["-0", "1.0", "1e4", "-1E-3", r#"{"α":-0.0}"#] {
            assert!(canonical_json_digest("d", json).is_ok(), "{json}");
        }
        // A literal in a string is not a number, even after an escaped quote.
        assert!(canonical_json_digest("d", r#"{"s":"\\\"18446744073709551616"}"#).is_ok());
        let input = r#"{"a":[-9223372036854775808,18446744073709551615,1.5]}"#;
        let mut original = CanonicalWriter::new("d");
        original.word(7).word(1).text("a").word(6).word(3);
        original.word(2).word(u64::from_le_bytes(i64::MIN.to_le_bytes()));
        original.word(3).word(u64::MAX);
        original.word(4).real(1.5);
        assert_eq!(canonical_json_digest("d", input).unwrap(), original.finish_hex());
        assert_ne!(
            canonical_json_digest("d", "18446744073709551615").unwrap(),
            canonical_json_digest("d", "18446744073709551616.0").unwrap()
        );
    }

    #[test]
    fn digest_refuses_blank_domain_garbage_and_runaway_nesting() {
        assert!(canonical_json_digest(" ", "{}").is_err());
        assert!(canonical_json_digest("d", "{").is_err());
        assert!(canonical_json_digest("d", "NaN").is_err());
        let deep = "[".repeat(100) + &"]".repeat(100);
        assert!(canonical_json_digest("d", &deep).is_err());
    }
}
