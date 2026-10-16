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
const MAX_DEPTH: usize = 64;

fn write(stream: &mut CanonicalWriter, value: &Value, depth: usize) -> Result<(), IoError> {
    if depth > MAX_DEPTH {
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
                write(stream, item, depth + 1)?;
            }
        }
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            stream.word(7).word(entries.len() as u64);
            for (key, item) in entries {
                stream.text(key);
                write(stream, item, depth + 1)?;
            }
        }
    }
    Ok(())
}

/// Digest of `json` under `domain`, as 64 lowercase hex characters.
///
/// # Errors
/// A blank domain, an oversized or unparseable document, a non-finite number
/// or excessive nesting refuses.
pub fn canonical_json_digest(domain: &str, json: &str) -> Result<String, IoError> {
    if json.len() > MAX_CANONICAL_JSON_BYTES {
        return Err(IoError::TooLarge);
    }
    let value: Value = serde_json::from_str(json)
        .map_err(|error| IoError::Convert(format!("canonical JSON is invalid: {error}")))?;
    canonical_json_value_digest(domain, &value)
}

/// [`canonical_json_digest`] of a value already parsed, so a Rust caller digests
/// what a host language would digest from the same document.
///
/// # Errors
/// A blank domain, a non-finite number or excessive nesting refuses.
pub fn canonical_json_value_digest(domain: &str, value: &Value) -> Result<String, IoError> {
    if domain.trim().is_empty() {
        return Err(IoError::Convert("canonical digest needs a domain".into()));
    }
    let mut stream = CanonicalWriter::new(domain);
    write(&mut stream, value, 0)?;
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
    fn digest_refuses_blank_domain_garbage_and_runaway_nesting() {
        assert!(canonical_json_digest(" ", "{}").is_err());
        assert!(canonical_json_digest("d", "{").is_err());
        assert!(canonical_json_digest("d", "NaN").is_err());
        let deep = "[".repeat(100) + &"]".repeat(100);
        assert!(canonical_json_digest("d", &deep).is_err());
    }
}
