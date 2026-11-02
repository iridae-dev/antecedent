//! Identity of the units a table's rows describe.
//!
//! A table's content digest says which values it holds, not which units they belong to: two
//! tables with the same row count and schema can describe different participants, or the same
//! ones in another order. A [`UnitKey`] is the declared identity of the row-to-unit assignment,
//! shared by every declaration that binds units (participant labels, a network's unit table)
//! and carried beside the data, never inside its content digest.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::error::DataError;

const SHA256_BLOCK_BYTES: usize = 64;
const SHA256_LENGTH_BYTES: usize = 8;
const SHA256_INITIAL: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];
const SHA256_ROUNDS: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// SHA-256 as lowercase hex. The digest the Python identifier binding already exports, so a
/// key encoded here and a binding encoded there name the same assignment.
fn sha256_hex(message: &[u8]) -> String {
    let mut padded = message.to_vec();
    padded.push(0x80);
    while padded.len() % SHA256_BLOCK_BYTES != SHA256_BLOCK_BYTES - SHA256_LENGTH_BYTES {
        padded.push(0);
    }
    let bits = u64::try_from(message.len()).unwrap_or(u64::MAX).wrapping_mul(8);
    padded.extend_from_slice(&bits.to_be_bytes());

    let mut state = SHA256_INITIAL;
    for block in padded.chunks_exact(SHA256_BLOCK_BYTES) {
        let mut schedule = [0_u32; 64];
        for (word, bytes) in schedule.iter_mut().zip(block.chunks_exact(4)) {
            *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for i in 16..64 {
            let lo = schedule[i - 15].rotate_right(7)
                ^ schedule[i - 15].rotate_right(18)
                ^ (schedule[i - 15] >> 3);
            let hi = schedule[i - 2].rotate_right(17)
                ^ schedule[i - 2].rotate_right(19)
                ^ (schedule[i - 2] >> 10);
            schedule[i] =
                schedule[i - 16].wrapping_add(lo).wrapping_add(schedule[i - 7]).wrapping_add(hi);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for (round, word) in SHA256_ROUNDS.iter().zip(schedule) {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ (!e & g);
            let first =
                h.wrapping_add(sum1).wrapping_add(choice).wrapping_add(*round).wrapping_add(word);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let second = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(first);
            d = c;
            c = b;
            b = a;
            a = first.wrapping_add(second);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    state.iter().fold(String::with_capacity(64), |mut hex, word| {
        let _ = write!(hex, "{word:08x}");
        hex
    })
}

/// The row count, the distinct unit labels in first-seen order, and every row's label code,
/// the last two as digests.
///
/// Labels that differ in text, or in which row carries them, give another key. Equality is the
/// identity of the assignment. The digests are those of the Python identifier binding
/// (SHA-256 of the labels joined by NUL, and of the little-endian `u32` codes), so a key and a
/// binding encoded from the same labels agree.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct UnitKey {
    rows: usize,
    labels: String,
    codes: String,
}

impl UnitKey {
    /// A key from digests an external encoder produced over `rows` rows.
    #[must_use]
    pub fn new(rows: usize, labels: impl Into<String>, codes: impl Into<String>) -> Self {
        Self { rows, labels: labels.into(), codes: codes.into() }
    }

    /// Encode one label per row: codes are assigned in first-seen order, and the key digests
    /// the distinct labels and the per-row codes.
    ///
    /// # Errors
    ///
    /// [`DataError::EmptyInput`] for no rows, or [`DataError::Overflow`] for more distinct
    /// labels than fit a 32-bit code.
    pub fn from_row_labels<S: AsRef<str>>(
        labels: impl IntoIterator<Item = S>,
    ) -> Result<Self, DataError> {
        let mut index: HashMap<String, u32> = HashMap::new();
        let mut distinct: Vec<String> = Vec::new();
        let mut codes: Vec<u8> = Vec::new();
        let mut rows = 0_usize;
        for label in labels {
            let label = label.as_ref();
            let code = if let Some(code) = index.get(label) {
                *code
            } else {
                let code = u32::try_from(distinct.len())
                    .map_err(|_| DataError::overflow("distinct unit labels as u32 codes"))?;
                index.insert(label.to_owned(), code);
                distinct.push(label.to_owned());
                code
            };
            codes.extend_from_slice(&code.to_le_bytes());
            rows += 1;
        }
        if rows == 0 {
            return Err(DataError::empty("unit labels"));
        }
        let labels = sha256_hex(distinct.join("\0").as_bytes());
        let codes = sha256_hex(&codes);
        Ok(Self { rows, labels, codes })
    }

    /// Rows the key labels.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Digest of the distinct labels in first-seen order.
    #[must_use]
    pub fn labels_digest(&self) -> &str {
        &self.labels
    }

    /// Digest of every row's label code.
    #[must_use]
    pub fn codes_digest(&self) -> &str {
        &self.codes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_digest_is_sha256() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn the_key_names_labels_and_their_row_assignment() {
        let a = UnitKey::from_row_labels(["p1", "p2", "p1"]).unwrap();
        assert_eq!(a, UnitKey::from_row_labels(["p1", "p2", "p1"]).unwrap());
        assert_ne!(a, UnitKey::from_row_labels(["p1", "p1", "p2"]).unwrap());
        assert_ne!(a, UnitKey::from_row_labels(["p1", "p3", "p1"]).unwrap());
        assert_eq!(a.rows(), 3);
    }

    #[test]
    fn a_table_carries_a_key_only_for_its_own_rows() {
        let values = [1.0, 2.0, 3.0];
        let table = crate::TabularData::from_f64_columns([("y", values.as_slice())]).unwrap();
        assert!(table.unit_key().is_none());
        let key = UnitKey::from_row_labels(["a", "b", "c"]).unwrap();
        let keyed = table.clone().with_unit_key(key.clone()).unwrap();
        assert_eq!(keyed.unit_key(), Some(&key));
        assert_eq!(keyed.storage().content_digest(), table.storage().content_digest());
        let short = UnitKey::from_row_labels(["a", "b"]).unwrap();
        assert!(table.with_unit_key(short).is_err());
    }

    #[test]
    fn no_rows_have_no_key() {
        assert!(UnitKey::from_row_labels(std::iter::empty::<&str>()).is_err());
    }
}
