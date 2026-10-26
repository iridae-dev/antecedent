//! A bounds-checked little-endian cursor over a byte slice.
//!
//! The binary section decoders of the artifacts (the spill payload of a capacity run, the
//! stream artifact header) read fixed-width words off a slice and refuse a short slice.
//! The refusal is the decoder's own error type, so the cursor takes a constructor for it
//! once and every read that runs past the end returns it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Cursor over `bytes` whose reads fail with `E`.
#[derive(Clone, Debug)]
pub struct ByteCursor<'a, E> {
    rest: &'a [u8],
    short: fn() -> E,
}

impl<'a, E> ByteCursor<'a, E> {
    /// A cursor at the start of `bytes`; a read past the end returns `short()`.
    #[must_use]
    pub const fn new(bytes: &'a [u8], short: fn() -> E) -> Self {
        Self { rest: bytes, short }
    }

    /// Bytes not yet read.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.rest.len()
    }

    /// The next `len` bytes.
    ///
    /// # Errors
    ///
    /// The cursor's short-read error when fewer than `len` bytes remain.
    pub fn take(&mut self, len: usize) -> Result<&'a [u8], E> {
        if self.rest.len() < len {
            return Err((self.short)());
        }
        let (head, tail) = self.rest.split_at(len);
        self.rest = tail;
        Ok(head)
    }

    /// The next `N` bytes as an array.
    ///
    /// # Errors
    ///
    /// The cursor's short-read error when fewer than `N` bytes remain.
    pub fn array<const N: usize>(&mut self) -> Result<[u8; N], E> {
        let bytes = self.take(N)?;
        let mut out = [0_u8; N];
        out.copy_from_slice(bytes);
        Ok(out)
    }

    /// The next byte.
    ///
    /// # Errors
    ///
    /// As [`Self::take`].
    pub fn u8(&mut self) -> Result<u8, E> {
        Ok(self.array::<1>()?[0])
    }

    /// The next little-endian `u16`.
    ///
    /// # Errors
    ///
    /// As [`Self::take`].
    pub fn u16(&mut self) -> Result<u16, E> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    /// The next little-endian `u32`.
    ///
    /// # Errors
    ///
    /// As [`Self::take`].
    pub fn u32(&mut self) -> Result<u32, E> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    /// The next little-endian `u64`.
    ///
    /// # Errors
    ///
    /// As [`Self::take`].
    pub fn u64(&mut self) -> Result<u64, E> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// The next little-endian IEEE-754 `f64`, bit pattern preserved.
    ///
    /// # Errors
    ///
    /// As [`Self::take`].
    pub fn f64(&mut self) -> Result<f64, E> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    /// Succeeds only when every byte was read; otherwise returns `trailing`.
    ///
    /// # Errors
    ///
    /// `trailing` when bytes remain.
    pub fn finish(self, trailing: E) -> Result<(), E> {
        if self.rest.is_empty() { Ok(()) } else { Err(trailing) }
    }
}

/// Cursor over decoded `f64` words whose reads fail with `E`, for the numbers sections the
/// artifacts store as one flat run of reals.
#[derive(Clone, Debug)]
pub struct RealCursor<'a, E> {
    rest: &'a [f64],
    short: fn() -> E,
}

impl<'a, E> RealCursor<'a, E> {
    /// A cursor at the start of `values`; a read past the end returns `short()`.
    #[must_use]
    pub const fn new(values: &'a [f64], short: fn() -> E) -> Self {
        Self { rest: values, short }
    }

    /// Values not yet read.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.rest.len()
    }

    /// The next value.
    ///
    /// # Errors
    ///
    /// The cursor's short-read error when no value remains.
    pub fn take(&mut self) -> Result<f64, E> {
        let (&first, tail) = self.rest.split_first().ok_or_else(|| (self.short)())?;
        self.rest = tail;
        Ok(first)
    }

    /// The next `len` values, borrowed.
    ///
    /// # Errors
    ///
    /// The cursor's short-read error when fewer than `len` values remain.
    pub fn take_slice(&mut self, len: usize) -> Result<&'a [f64], E> {
        if self.rest.len() < len {
            return Err((self.short)());
        }
        let (head, tail) = self.rest.split_at(len);
        self.rest = tail;
        Ok(head)
    }
}

#[cfg(test)]
mod tests {
    use super::{ByteCursor, RealCursor};

    #[test]
    fn real_cursor_refuses_a_short_run_without_consuming() {
        let values = [1.0, 2.0, 3.0];
        let mut cursor = RealCursor::new(&values, || "short");
        assert_eq!(cursor.take(), Ok(1.0));
        assert_eq!(cursor.take_slice(3), Err("short"));
        assert_eq!(cursor.remaining(), 2);
        assert_eq!(cursor.take_slice(2), Ok(&values[1..]));
        assert_eq!(cursor.take(), Err("short"));
    }

    #[derive(Debug, PartialEq)]
    enum Fault {
        Short,
        Trailing,
    }

    #[test]
    fn reads_words_in_order_and_refuses_a_short_or_long_slice() {
        let mut bytes = vec![7_u8];
        bytes.extend_from_slice(&0x0102_u16.to_le_bytes());
        bytes.extend_from_slice(&0x0304_0506_u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&(-0.0_f64).to_le_bytes());
        let mut cursor = ByteCursor::new(&bytes, || Fault::Short);
        assert_eq!(cursor.u8(), Ok(7));
        assert_eq!(cursor.u16(), Ok(0x0102));
        assert_eq!(cursor.u32(), Ok(0x0304_0506));
        assert_eq!(cursor.u64(), Ok(u64::MAX));
        assert_eq!(cursor.f64().unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(cursor.remaining(), 0);
        assert_eq!(cursor.u8(), Err(Fault::Short));
        assert_eq!(cursor.finish(Fault::Trailing), Ok(()));

        let mut cursor = ByteCursor::new(&bytes[..2], || Fault::Short);
        assert_eq!(cursor.u32(), Err(Fault::Short));
        assert_eq!(cursor.remaining(), 2, "a refused read consumes nothing");
        assert_eq!(cursor.finish(Fault::Trailing), Err(Fault::Trailing));
    }
}
