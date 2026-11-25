//! Byte buffer with the same frames as [`antecedent_core::CanonicalWriter`].
//!
//! `blake3` of the buffer is the writer's digest. The buffer is how a child
//! codec keeps the bytes the writer hashes and then throws away.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CanonicalWriter;

/// Framing calls a lowered-layer identity and its byte buffer share.
pub(crate) trait CanonPut {
    /// One little-endian `u64`.
    fn word(&mut self, word: u64) -> &mut Self;
    /// Length-prefixed raw bytes.
    fn bytes(&mut self, bytes: &[u8]) -> &mut Self;
    /// Length-prefixed UTF-8.
    fn text(&mut self, text: &str) -> &mut Self;
    /// A real as its IEEE-754 bits.
    fn real(&mut self, value: f64) -> &mut Self;
    /// A presence word, then the text when it is present.
    fn opt_text(&mut self, value: Option<&str>) -> &mut Self;
    /// A count, then each text.
    fn texts<S: AsRef<str>>(&mut self, values: &[S]) -> &mut Self;
}

impl CanonPut for CanonicalWriter {
    fn word(&mut self, word: u64) -> &mut Self {
        CanonicalWriter::word(self, word)
    }

    fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        CanonicalWriter::bytes(self, bytes)
    }

    fn text(&mut self, text: &str) -> &mut Self {
        CanonicalWriter::text(self, text)
    }

    fn real(&mut self, value: f64) -> &mut Self {
        CanonicalWriter::real(self, value)
    }

    fn opt_text(&mut self, value: Option<&str>) -> &mut Self {
        CanonicalWriter::opt_text(self, value)
    }

    fn texts<S: AsRef<str>>(&mut self, values: &[S]) -> &mut Self {
        CanonicalWriter::texts(self, values)
    }
}

/// The framed bytes [`CanonicalWriter`] would have hashed, including its tag.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct CanonBuf {
    body: Vec<u8>,
}

#[cfg(test)]
impl CanonBuf {
    /// Start with `tag` framed as text, the same first frame as `CanonicalWriter::new`.
    pub(crate) fn new(tag: &str) -> Self {
        let mut buf = Self { body: Vec::new() };
        buf.text(tag);
        buf
    }

    /// The framed bytes.
    pub(crate) fn into_vec(self) -> Vec<u8> {
        self.body
    }

    fn put_word(&mut self, word: u64) {
        self.body.extend_from_slice(&word.to_le_bytes());
    }
}

#[cfg(test)]
impl CanonPut for CanonBuf {
    fn word(&mut self, word: u64) -> &mut Self {
        self.put_word(word);
        self
    }

    fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.put_word(bytes.len() as u64);
        self.body.extend_from_slice(bytes);
        self
    }

    fn text(&mut self, text: &str) -> &mut Self {
        self.bytes(text.as_bytes())
    }

    fn real(&mut self, value: f64) -> &mut Self {
        self.word(value.to_bits())
    }

    fn opt_text(&mut self, value: Option<&str>) -> &mut Self {
        match value {
            Some(text) => self.word(1).text(text),
            None => self.word(0),
        }
    }

    fn texts<S: AsRef<str>>(&mut self, values: &[S]) -> &mut Self {
        self.word(values.len() as u64);
        for value in values {
            self.text(value.as_ref());
        }
        self
    }
}

/// BLAKE3 of framed bytes. This is [`CanonicalWriter::finish`] when the frames match.
#[cfg(test)]
pub(crate) fn framed_digest(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample<S: CanonPut>(sink: &mut S) {
        sink.text("ab").word(3).bytes(&[1, 2]).real(-0.0).opt_text(Some("z")).opt_text(None);
        sink.texts(&["p", "q"]);
    }

    #[test]
    fn the_buffer_hashes_to_the_writer_digest() {
        let mut writer = CanonicalWriter::new("antecedent.lowered_program.canon_buffer");
        sample(&mut writer);
        let mut buf = CanonBuf::new("antecedent.lowered_program.canon_buffer");
        sample(&mut buf);
        assert_eq!(framed_digest(&buf.into_vec()), writer.finish());
    }
}
