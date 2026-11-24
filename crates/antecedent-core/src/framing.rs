//! Canonical framing kernels: the byte layout every identity is written in before hashing.
//!
//! Each kernel appends the frame of one value to a [`FrameSink`]. [`crate::CanonicalWriter`]
//! feeds BLAKE3 through these kernels and nothing else, so its byte stream is a composition of
//! the frames below:
//!
//! | Frame | Bytes |
//! |---|---|
//! | word | the `u64`, little-endian (8 bytes) |
//! | byte string, text, tag | its length as a word, then the bytes |
//! | presence | the word `0` or `1` |
//! | real | the word of its IEEE-754 bit pattern |
//! | ordered list | the count as a word, then each item's frame in the order given |
//! | set of digests | sorted ascending, repeats removed, then framed as an ordered list |
//! | multiset of digests or `u32` values | sorted ascending, repeats kept, then framed as an ordered list |
//!
//! The kernels are pure functions over integers, byte arrays and slices so that Charon and
//! Aeneas translate them (`formal/rust/extract.sh`, extraction `Framing`); the Lean proofs in
//! `formal/rust/AntecedentRust/Framing.lean` show each frame is uniquely decodable, that set
//! framing ignores order and repeats, multiset framing ignores order only, and ordered framing
//! ignores neither. The two sorts are the standard library's `sort_unstable`, which the
//! translation treats as an ascending sort.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Receiver of framed bytes, in the order they are framed.
pub(crate) trait FrameSink {
    /// Append `bytes`.
    fn put(&mut self, bytes: &[u8]);
}

/// One `u64` word, little-endian.
pub(crate) fn frame_word<S: FrameSink>(sink: &mut S, word: u64) {
    sink.put(&word.to_le_bytes());
}

/// A length or count as its word.
pub(crate) fn frame_count<S: FrameSink>(sink: &mut S, count: usize) {
    frame_word(sink, count as u64);
}

/// A byte string (raw bytes, UTF-8 text or a domain tag): its length as a word, then the bytes.
pub(crate) fn frame_bytes<S: FrameSink>(sink: &mut S, payload: &[u8]) {
    frame_count(sink, payload.len());
    sink.put(payload);
}

/// The presence word of an optional value: `1` when present, `0` when absent.
pub(crate) fn frame_presence<S: FrameSink>(sink: &mut S, present: bool) {
    frame_word(sink, u64::from(present));
}

/// A real as the word of its IEEE-754 bit pattern (`f64::to_bits`), so `-0.0` and `0.0` differ
/// and NaN payloads are kept.
pub(crate) fn frame_real_bits<S: FrameSink>(sink: &mut S, bits: u64) {
    frame_word(sink, bits);
}

/// An ordered list of words: the count, then each word in the order given.
pub(crate) fn frame_words<S: FrameSink>(sink: &mut S, values: &[u64]) {
    frame_count(sink, values.len());
    let mut i = 0;
    while i < values.len() {
        frame_word(sink, values[i]);
        i += 1;
    }
}

/// An ordered list of `u32` values: the count, then each value widened to one word.
pub(crate) fn frame_u32s<S: FrameSink>(sink: &mut S, values: &[u32]) {
    frame_count(sink, values.len());
    let mut i = 0;
    while i < values.len() {
        frame_word(sink, u64::from(values[i]));
        i += 1;
    }
}

/// An ordered list of 32-byte digests: the count, then each digest as a byte string.
pub(crate) fn frame_digests<S: FrameSink>(sink: &mut S, digests: &[[u8; 32]]) {
    frame_count(sink, digests.len());
    let mut i = 0;
    while i < digests.len() {
        frame_bytes(sink, &digests[i]);
        i += 1;
    }
}

/// Sort digests ascending (lexicographic on bytes).
pub(crate) fn sort_digests(digests: &mut [[u8; 32]]) {
    digests.sort_unstable();
}

/// Sort `u32` values ascending.
pub(crate) fn sort_u32s(values: &mut [u32]) {
    values.sort_unstable();
}

/// The digests of a sorted list with each repeat after its first occurrence removed.
pub(crate) fn dedup_sorted_digests(sorted: &[[u8; 32]]) -> Vec<[u8; 32]> {
    let mut distinct: Vec<[u8; 32]> = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        if i == 0 || sorted[i - 1] != sorted[i] {
            distinct.push(sorted[i]);
        }
        i += 1;
    }
    distinct
}

/// A set of digests: sorted, repeats removed, then framed as an ordered list. Neither the
/// order of `digests` nor a repeated digest reaches the bytes.
pub(crate) fn frame_digest_set<S: FrameSink>(sink: &mut S, mut digests: Vec<[u8; 32]>) {
    sort_digests(&mut digests);
    let distinct = dedup_sorted_digests(&digests);
    frame_digests(sink, &distinct);
}

/// A multiset of digests: sorted with repeats kept, then framed as an ordered list. The order
/// of `digests` does not reach the bytes; how often a digest repeats does.
pub(crate) fn frame_digest_multiset<S: FrameSink>(sink: &mut S, mut digests: Vec<[u8; 32]>) {
    sort_digests(&mut digests);
    frame_digests(sink, &digests);
}

/// A multiset of `u32` values: sorted with repeats kept, then framed as an ordered list.
pub(crate) fn frame_u32_multiset<S: FrameSink>(sink: &mut S, mut values: Vec<u32>) {
    sort_u32s(&mut values);
    frame_u32s(sink, &values);
}

#[cfg(test)]
mod tests {
    use super::*;

    impl FrameSink for Vec<u8> {
        fn put(&mut self, bytes: &[u8]) {
            self.extend_from_slice(bytes);
        }
    }

    fn framed(write: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut out = Vec::new();
        write(&mut out);
        out
    }

    fn word(w: u64) -> Vec<u8> {
        w.to_le_bytes().to_vec()
    }

    #[test]
    fn frames_have_the_documented_bytes() {
        assert_eq!(framed(|s| frame_word(s, 0x0102_0304_0506_0708)), [8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(framed(|s| frame_bytes(s, b"ab")), [word(2), b"ab".to_vec()].concat());
        assert_eq!(framed(|s| frame_presence(s, true)), word(1));
        assert_eq!(framed(|s| frame_presence(s, false)), word(0));
        assert_ne!(
            framed(|s| frame_real_bits(s, (-0.0_f64).to_bits())),
            framed(|s| frame_real_bits(s, 0.0_f64.to_bits()))
        );
        assert_eq!(framed(|s| frame_words(s, &[3, 1])), [word(2), word(3), word(1)].concat());
        assert_eq!(framed(|s| frame_u32s(s, &[7])), [word(1), word(7)].concat());
    }

    #[test]
    fn set_framing_ignores_order_and_repeats_and_multiset_framing_keeps_repeats() {
        let (a, b) = ([1_u8; 32], [2_u8; 32]);
        let set = |items: Vec<[u8; 32]>| framed(|s| frame_digest_set(s, items));
        let multiset = |items: Vec<[u8; 32]>| framed(|s| frame_digest_multiset(s, items));
        assert_eq!(set(vec![b, a, b]), set(vec![a, b]));
        assert_eq!(set(vec![a, b]), [word(2), word(32), a.to_vec(), word(32), b.to_vec()].concat());
        assert_eq!(multiset(vec![b, a]), multiset(vec![a, b]));
        assert_ne!(multiset(vec![a, b, b]), multiset(vec![a, b]));
        let u32_multiset = |items: Vec<u32>| framed(|s| frame_u32_multiset(s, items));
        assert_eq!(u32_multiset(vec![3, 1, 3]), framed(|s| frame_u32s(s, &[1, 3, 3])));
        assert_ne!(framed(|s| frame_words(s, &[1, 2])), framed(|s| frame_words(s, &[2, 1])));
    }

    /// The writer as it was before the kernels: words, length prefixes and the set encodings
    /// hashed directly. The kernels must leave every digest unchanged.
    struct Legacy(blake3::Hasher);

    impl Legacy {
        fn new(tag: &str) -> Self {
            let mut legacy = Self(blake3::Hasher::new());
            legacy.text(tag);
            legacy
        }
        fn word(&mut self, word: u64) {
            self.0.update(&word.to_le_bytes());
        }
        fn bytes(&mut self, bytes: &[u8]) {
            self.word(bytes.len() as u64);
            self.0.update(bytes);
        }
        fn text(&mut self, text: &str) {
            self.bytes(text.as_bytes());
        }
        fn real(&mut self, value: f64) {
            self.word(value.to_bits());
        }
        fn finish(&self) -> [u8; 32] {
            *self.0.finalize().as_bytes()
        }
    }

    /// A deterministic stream of test values.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 11
        }
        fn below(&mut self, n: usize) -> usize {
            usize::try_from(self.next() % n as u64).unwrap()
        }
    }

    fn digest(seed: usize) -> [u8; 32] {
        *blake3::hash(&seed.to_le_bytes()).as_bytes()
    }

    #[test]
    fn the_writer_hashes_the_same_bytes_as_before_the_kernels() {
        use crate::CanonicalWriter;
        let reals =
            [0.0, -0.0, 1.5, f64::NAN, f64::from_bits(0x7ff8_0000_0000_0001), f64::INFINITY];
        let mut rng = Lcg(7);
        for case in 0..200_u64 {
            let mut new = CanonicalWriter::new("antecedent.framing.equality");
            let mut old = Legacy::new("antecedent.framing.equality");
            for _ in 0..rng.below(12) {
                match rng.below(14) {
                    0 => {
                        let w = rng.next() ^ (case << 50);
                        new.word(w);
                        old.word(w);
                    }
                    1 => {
                        let text = "x".repeat(rng.below(40));
                        new.text(&text);
                        old.text(&text);
                    }
                    2 => {
                        let bytes: Vec<u8> =
                            (0..rng.below(9000)).map(|i| u8::try_from(i % 256).unwrap()).collect();
                        new.bytes(&bytes);
                        old.bytes(&bytes);
                        let mut checked = CanonicalWriter::new("t");
                        checked.try_bytes(&bytes, || Ok::<(), ()>(())).unwrap();
                        assert_eq!(
                            checked.finish(),
                            CanonicalWriter::new("t").bytes(&bytes).finish()
                        );
                    }
                    3 => {
                        let value = reals[rng.below(reals.len())];
                        new.real(value);
                        old.real(value);
                    }
                    4 => {
                        let values: Vec<f64> = (0..rng.below(5)).map(|i| reals[i]).collect();
                        new.reals(&values);
                        old.word(values.len() as u64);
                        values.iter().for_each(|&v| old.real(v));
                    }
                    5 => {
                        let values: Vec<u32> = (0..rng.below(6))
                            .map(|_| u32::try_from(rng.next() >> 21).unwrap())
                            .collect();
                        new.u32s(&values);
                        old.word(values.len() as u64);
                        values.iter().for_each(|&v| old.word(u64::from(v)));
                    }
                    6 => {
                        let values: Vec<u64> = (0..rng.below(6)).map(|_| rng.next()).collect();
                        new.words(&values);
                        old.word(values.len() as u64);
                        values.iter().for_each(|&v| old.word(v));
                    }
                    7 => {
                        let values: Vec<String> =
                            (0..rng.below(4)).map(|i| "ab".repeat(i)).collect();
                        new.texts(&values);
                        old.word(values.len() as u64);
                        values.iter().for_each(|v| old.text(v));
                    }
                    8 => {
                        let value = (rng.below(2) == 0).then_some("opt");
                        new.opt_text(value);
                        match value {
                            Some(text) => {
                                old.word(1);
                                old.text(text);
                            }
                            None => old.word(0),
                        }
                    }
                    9 => {
                        let value = (rng.below(2) == 0).then(|| rng.next());
                        new.opt_word(value);
                        match value {
                            Some(w) => {
                                old.word(1);
                                old.word(w);
                            }
                            None => old.word(0),
                        }
                    }
                    10 => {
                        let value = (rng.below(2) == 0).then(|| reals[rng.below(6)]);
                        new.opt_real(value);
                        match value {
                            Some(v) => {
                                old.word(1);
                                old.real(v);
                            }
                            None => old.word(0),
                        }
                    }
                    11 => {
                        // A set of item digests (SetOf): sorted, then deduplicated.
                        let items: Vec<[u8; 32]> =
                            (0..rng.below(7)).map(|_| digest(rng.below(4))).collect();
                        let mut sorted = items.clone();
                        sorted.sort_unstable();
                        sorted.dedup();
                        new.digest_set(items);
                        old.word(sorted.len() as u64);
                        sorted.iter().for_each(|d| old.bytes(d));
                    }
                    12 => {
                        // A multiset of item digests (encode_set): sorted, repeats kept.
                        let items: Vec<[u8; 32]> =
                            (0..rng.below(7)).map(|_| digest(rng.below(4))).collect();
                        let mut sorted = items.clone();
                        sorted.sort_unstable();
                        new.digest_multiset(items);
                        old.word(sorted.len() as u64);
                        sorted.iter().for_each(|d| old.bytes(d));
                    }
                    _ => {
                        // A variable multiset (encode_variable_set): ids sorted, repeats kept.
                        let ids: Vec<crate::ids::VariableId> = (0..rng.below(7))
                            .map(|_| {
                                crate::ids::VariableId::from_raw(
                                    u32::try_from(rng.below(5)).unwrap(),
                                )
                            })
                            .collect();
                        let mut sorted = ids.clone();
                        sorted.sort_unstable();
                        crate::canonical_encode::encode_variable_set(&mut new, &ids);
                        old.word(sorted.len() as u64);
                        sorted.iter().for_each(|id| old.word(u64::from(id.raw())));
                    }
                }
            }
            assert_eq!(new.finish(), old.finish(), "case {case}");
        }
    }
}
