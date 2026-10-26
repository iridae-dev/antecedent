//! Schema-directed CBOR decoding: quote real collection storage before constructing it.
//!
//! No generic CBOR value tree, encoded-size multiplier or untrusted size-hint allocation.
//! The first pass checks the entire schema while borrowing input. The second allocates
//! each string/vector at its validated length. One owner holds their aggregate lease.

use super::*;
use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;
use std::mem::size_of;
use std::time::Duration;

/// An independently verified baseline and admission for its decoded collections.
///
/// This owner intentionally has no `Clone` or consuming wire accessor: borrow it or
/// share the owner with `Arc`, so a decoded buffer cannot outlive its admission.
pub struct AdmittedStalenessBaseline {
    wire: StalenessBaselineWire,
    _storage: AdmissionLease,
}

impl AdmittedStalenessBaseline {
    /// Borrow the verified wire while its actual collection storage remains funded.
    #[must_use]
    pub const fn wire(&self) -> &StalenessBaselineWire {
        &self.wire
    }
}

fn invalid(message: &'static str) -> IoError {
    IoError::Convert(message.into())
}

fn add(a: usize, b: usize) -> Result<usize, IoError> {
    a.checked_add(b).ok_or_else(|| invalid("staleness decoded storage overflows usize"))
}

fn lease(ctx: &ExecutionContext, bytes: usize) -> Result<AdmissionLease, IoError> {
    let bytes =
        u64::try_from(bytes).map_err(|_| invalid("staleness decoded storage overflows u64"))?;
    ctx.acquire_admission_bounded(0, bytes, Duration::ZERO).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => IoError::StreamCancelled,
        other => IoError::StreamBudgetRefused { message: other.to_string() },
    })
}

impl StalenessBaselineWire {
    /// Decode with schema-aware admission, retaining the lease with the actual wire.
    ///
    /// The format, accepted historical versions and seal are unchanged. This funds
    /// decoded collection payloads and the temporary seal string, not caller input,
    /// later report conversion, or new-data assessment workspace.
    pub fn consume_admitted(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<AdmittedStalenessBaseline, IoError> {
        if bytes.len() > STALENESS_BASELINE_MAX_BYTES {
            return Err(invalid("staleness baseline exceeds the byte limit"));
        }
        let body = bytes
            .strip_prefix(STALENESS_BASELINE_MAGIC)
            .ok_or_else(|| invalid("bytes are not a staleness baseline"))?;
        let mut scan = Decoder { bytes: body, position: 0, ctx };
        let storage = StalenessBaselineWire::scan(&mut scan)?;
        scan.end()?;
        let storage = lease(ctx, storage)?;
        let mut decode = Decoder { bytes: body, position: 0, ctx };
        let wire = StalenessBaselineWire::read(&mut decode)?;
        decode.end()?;
        // expected_seal serializes a borrowed seal-cleared wire, without cloning
        // any profiles. Only its fixed-size 64-byte hex return is allocated.
        let scratch = lease(ctx, 64)?;
        wire.verify()?;
        drop(scratch);
        decode.cancelled()?;
        Ok(AdmittedStalenessBaseline { wire, _storage: storage })
    }
}

#[derive(Clone, Copy)]
struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
    ctx: &'a ExecutionContext,
}

#[derive(Clone, Copy)]
struct Header {
    major: u8,
    additional: u8,
    value: u64,
}

impl Decoder<'_> {
    fn cancelled(&self) -> Result<(), IoError> {
        if self.ctx.cancellation.is_cancelled() { Err(IoError::StreamCancelled) } else { Ok(()) }
    }

    fn take(&mut self, length: usize) -> Result<&[u8], IoError> {
        self.cancelled()?;
        let end = add(self.position, length)?;
        let result = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| invalid("truncated staleness baseline"))?;
        self.position = end;
        Ok(result)
    }

    fn raw_header(&mut self) -> Result<Header, IoError> {
        let byte = self.take(1)?[0];
        let additional = byte & 31;
        let value = match additional {
            0..=23 => u64::from(additional),
            24..=27 => {
                let length = 1usize << (additional - 24);
                self.take(length)?.iter().fold(0u64, |value, byte| (value << 8) | u64::from(*byte))
            }
            31 => 0,
            _ => return Err(invalid("invalid staleness CBOR header")),
        };
        let additional =
            if byte >> 5 == 7 && additional == 24 && value < 24 { value as u8 } else { additional };
        Ok(Header { major: byte >> 5, additional, value })
    }

    fn header(&mut self) -> Result<Header, IoError> {
        for _ in 0..256 {
            let header = self.raw_header()?;
            if header.major != 6 {
                return Ok(header);
            }
            if header.additional == 31 {
                break;
            }
        }
        Err(invalid("staleness CBOR tag nesting exceeds limit"))
    }

    fn null(&mut self) -> Result<bool, IoError> {
        let mut probe = *self;
        let header = probe.header()?;
        if header.major == 7 && matches!(header.additional, 22 | 23) {
            *self = probe;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn end(&self) -> Result<(), IoError> {
        self.cancelled()?;
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(invalid("staleness baseline has trailing bytes"))
        }
    }

    fn sequence(
        &mut self,
        major: u8,
        mut element: impl FnMut(&mut Self) -> Result<(), IoError>,
    ) -> Result<usize, IoError> {
        let header = self.header()?;
        if header.major != major {
            return Err(invalid("staleness CBOR collection has wrong type"));
        }
        if header.additional != 31 {
            let length = usize::try_from(header.value)
                .map_err(|_| invalid("staleness CBOR collection length exceeds usize"))?;
            for _ in 0..length {
                element(self)?;
            }
            return Ok(length);
        }
        let mut length = 0;
        loop {
            self.cancelled()?;
            if self.bytes.get(self.position) == Some(&0xff) {
                self.position += 1;
                return Ok(length);
            }
            element(self)?;
            length = add(length, 1)?;
        }
    }

    fn text(&mut self, mut chunk: impl FnMut(&str) -> Result<(), IoError>) -> Result<(), IoError> {
        let header = self.header()?;
        if header.major != 3 {
            return Err(invalid("staleness CBOR text has wrong type"));
        }
        if header.additional != 31 {
            let length = usize::try_from(header.value)
                .map_err(|_| invalid("staleness CBOR text exceeds usize"))?;
            let text = std::str::from_utf8(self.take(length)?)
                .map_err(|_| invalid("staleness CBOR text is not UTF-8"))?;
            return chunk(text);
        }
        loop {
            if self.bytes.get(self.position) == Some(&0xff) {
                self.position += 1;
                return Ok(());
            }
            let part = self.raw_header()?;
            if part.major != 3 || part.additional == 31 {
                return Err(invalid("invalid indefinite staleness CBOR text"));
            }
            let length = usize::try_from(part.value)
                .map_err(|_| invalid("staleness CBOR text exceeds usize"))?;
            let text = std::str::from_utf8(self.take(length)?)
                .map_err(|_| invalid("staleness CBOR text is not UTF-8"))?;
            chunk(text)?;
        }
    }

    fn key(&mut self) -> Result<Key, IoError> {
        let mut key = Key { bytes: [0; 32], length: 0 };
        self.text(|part| {
            let end = add(key.length, part.len())?;
            let target = key
                .bytes
                .get_mut(key.length..end)
                .ok_or_else(|| invalid("unknown staleness baseline field"))?;
            target.copy_from_slice(part.as_bytes());
            key.length = end;
            Ok(())
        })?;
        Ok(key)
    }

    fn record(
        &mut self,
        fields: &[&'static str],
        mut field: impl FnMut(&str, &mut Self) -> Result<(), IoError>,
    ) -> Result<(), IoError> {
        let mut probe = *self;
        if probe.header()?.major == 4 {
            let mut index = 0;
            self.sequence(4, |d| {
                let name = fields
                    .get(index)
                    .ok_or_else(|| invalid("staleness record has too many fields"))?;
                field(name, d)?;
                index += 1;
                Ok(())
            })?;
        } else {
            self.sequence(5, |d| {
                let key = d.key()?;
                field(key.text(), d)
            })?;
        }
        Ok(())
    }
}

struct Key {
    bytes: [u8; 32],
    length: usize,
}

impl Key {
    fn text(&self) -> &str {
        // Each chunk was checked above; joining UTF-8 chunks stays UTF-8.
        std::str::from_utf8(&self.bytes[..self.length]).unwrap_or("")
    }
}

trait Wire: Sized {
    /// Dynamic payload bytes only; inline storage belongs to the parent vector.
    fn scan(decoder: &mut Decoder<'_>) -> Result<usize, IoError>;
    fn read(decoder: &mut Decoder<'_>) -> Result<Self, IoError>;
}

impl Wire for String {
    fn scan(d: &mut Decoder<'_>) -> Result<usize, IoError> {
        let mut length = 0;
        d.text(|part| {
            length = add(length, part.len())?;
            Ok(())
        })?;
        Ok(length)
    }
    fn read(d: &mut Decoder<'_>) -> Result<Self, IoError> {
        let mut probe = *d;
        let length = Self::scan(&mut probe)?;
        let mut value = String::new();
        value.try_reserve_exact(length).map_err(|error| IoError::StreamBudgetRefused {
            message: format!("staleness string allocation failed: {error}"),
        })?;
        d.text(|part| {
            value.push_str(part);
            Ok(())
        })?;
        Ok(value)
    }
}

impl<T: Wire> Wire for Vec<T> {
    fn scan(d: &mut Decoder<'_>) -> Result<usize, IoError> {
        let mut nested = 0;
        let count = d.sequence(4, |d| {
            nested = add(nested, T::scan(d)?)?;
            Ok(())
        })?;
        let inline = count
            .checked_mul(size_of::<T>())
            .ok_or_else(|| invalid("staleness vector storage overflows usize"))?;
        add(inline, nested)
    }
    fn read(d: &mut Decoder<'_>) -> Result<Self, IoError> {
        let mut probe = *d;
        let count = probe.sequence(4, |d| T::scan(d).map(|_| ()))?;
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|error| IoError::StreamBudgetRefused {
            message: format!("staleness vector allocation failed: {error}"),
        })?;
        d.sequence(4, |d| {
            values.push(T::read(d)?);
            Ok(())
        })?;
        Ok(values)
    }
}

impl<T: Wire> Wire for Option<T> {
    fn scan(d: &mut Decoder<'_>) -> Result<usize, IoError> {
        if d.null()? { Ok(0) } else { T::scan(d) }
    }
    fn read(d: &mut Decoder<'_>) -> Result<Self, IoError> {
        if d.null()? { Ok(None) } else { T::read(d).map(Some) }
    }
}

macro_rules! scalar {
    ($ty:ty, $d:ident, $body:block) => {
        impl Wire for $ty {
            fn scan(d: &mut Decoder<'_>) -> Result<usize, IoError> { Self::read(d).map(|_| 0) }
            fn read($d: &mut Decoder<'_>) -> Result<Self, IoError> $body
        }
    };
}

scalar!(u64, d, {
    let h = d.header()?;
    if h.major == 0 && h.additional != 31 {
        Ok(h.value)
    } else {
        Err(invalid("staleness field requires an unsigned integer"))
    }
});
scalar!(u32, d, {
    Self::try_from(u64::read(d)?).map_err(|_| invalid("staleness integer exceeds u32"))
});
scalar!(bool, d, {
    let h = d.header()?;
    match (h.major, h.additional) {
        (7, 20) => Ok(false),
        (7, 21) => Ok(true),
        _ => Err(invalid("staleness field requires a boolean")),
    }
});
scalar!(f64, d, {
    let h = d.header()?;
    match (h.major, h.additional) {
        (0, 0..=27) => Ok(h.value as f64),
        (1, 0..=27) => Ok(-1.0 - h.value as f64),
        (7, 25) => {
            let bits = h.value as u16;
            let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
            let exponent = (bits >> 10) & 31;
            let fraction = bits & 1023;
            if exponent == 31 {
                return Ok(f64::from_bits(
                    (u64::from(bits & 0x8000) << 48)
                        | 0x7ff0_0000_0000_0000
                        | if fraction == 0 {
                            0
                        } else {
                            0x0008_0000_0000_0000 | (u64::from(fraction) << 42)
                        },
                ));
            }
            Ok(sign
                * match exponent {
                    0 => f64::from(fraction) * 2f64.powi(-24),
                    _ => f64::from(1024 + fraction) * 2f64.powi(i32::from(exponent) - 25),
                })
        }
        (7, 26) => Ok(f64::from(f32::from_bits(h.value as u32))),
        (7, 27) => Ok(f64::from_bits(h.value)),
        _ => Err(invalid("staleness field requires a real number")),
    }
});

impl<A: Wire, B: Wire> Wire for (A, B) {
    fn scan(d: &mut Decoder<'_>) -> Result<usize, IoError> {
        let mut bytes = 0;
        let mut index = 0;
        d.sequence(4, |d| {
            bytes = add(
                bytes,
                match index {
                    0 => A::scan(d)?,
                    1 => B::scan(d)?,
                    _ => return Err(invalid("staleness tuple exceeds two elements")),
                },
            )?;
            index += 1;
            Ok(())
        })?;
        if index != 2 {
            return Err(invalid("staleness tuple requires two elements"));
        }
        Ok(bytes)
    }
    fn read(d: &mut Decoder<'_>) -> Result<Self, IoError> {
        let (mut a, mut b) = (None, None);
        d.sequence(4, |d| {
            if a.is_none() {
                a = Some(A::read(d)?);
            } else if b.is_none() {
                b = Some(B::read(d)?);
            } else {
                return Err(invalid("staleness tuple exceeds two elements"));
            }
            Ok(())
        })?;
        Ok((
            a.ok_or_else(|| invalid("missing staleness tuple element"))?,
            b.ok_or_else(|| invalid("missing staleness tuple element"))?,
        ))
    }
}

macro_rules! required {
    ($value:ident) => {
        $value.ok_or_else(|| invalid(concat!("missing staleness field: ", stringify!($value))))?
    };
    ($value:ident, default) => {
        $value.unwrap_or_default()
    };
}

macro_rules! record {
    ($ty:ident { $( $field:ident: $value:ty $(= $default:ident)? ),* $(,)? }) => {
        impl Wire for $ty {
            fn scan(d: &mut Decoder<'_>) -> Result<usize, IoError> {
                $(let mut $field = None;)*
                let mut bytes = 0;
                d.record(&[$(stringify!($field)),*], |key, d| {
                    match key {
                        $(stringify!($field) => {
                            if $field.is_some() { return Err(invalid("duplicate staleness field")); }
                            bytes = add(bytes, <$value>::scan(d)?)?;
                            $field = Some(());
                        })*
                        _ => return Err(invalid("unknown staleness baseline field")),
                    }
                    Ok(())
                })?;
                $(let _ = required!($field $(, $default)?);)*
                Ok(bytes)
            }
            fn read(d: &mut Decoder<'_>) -> Result<Self, IoError> {
                $(let mut $field = None;)*
                d.record(&[$(stringify!($field)),*], |key, d| {
                    match key {
                        $(stringify!($field) => {
                            if $field.is_some() { return Err(invalid("duplicate staleness field")); }
                            $field = Some(<$value>::read(d)?);
                        })*
                        _ => return Err(invalid("unknown staleness baseline field")),
                    }
                    Ok(())
                })?;
                Ok(Self { $($field: required!($field $(, $default)?)),* })
            }
        }
    };
}

record!(NumericProfileWire { mean: f64, m2: f64, min: f64, max: f64,
    edges: Vec<f64>, bin_counts: Vec<u64> });
record!(LevelCountWire { label: String, count: u64 });
record!(StalenessColumnWire { name: String, value_type: String, storage: String,
    role: String, n_valid: u64, n_missing: u64, numeric: Option<NumericProfileWire> = default,
    levels: Option<Vec<LevelCountWire>> = default });
record!(JointProfileWire { columns: Vec<String>, n: u64, mean: Vec<f64>, m2: Vec<f64>,
    subsample: Vec<f64>, subsample_rows: u64 });
record!(MissingPatternWire { columns: Vec<String>, patterns: Vec<(u64,u64)>,
    wide_patterns: Vec<(Vec<u64>,u64)> = default });
record!(OverlapSummaryWire {
    n: u64,
    extreme_fraction: f64,
    ess_ratio: f64,
    min_propensity: f64,
    max_propensity: f64,
    max_weight: f64,
    max_weighted_smd: f64,
    epsilon: f64
});
record!(RecordedCheckWire {
    name: String,
    level: String,
    statistic: f64,
    threshold: f64,
    direction: String,
    passed: bool
});
record!(OutcomeModelWire { design: Vec<String>, outcome: String, n: u64,
    xtx: Vec<f64>, xty: Vec<f64>, yty: f64 });
record!(StalenessBaselineWire { version: u32, required_features: Vec<String>, kind: String,
    plan_identity: String, snapshot: String, policy: String, treatment: Option<String> = default,
    outcome: Option<String> = default, adjustment: Vec<String>, bayesian: bool, period_column: Option<String> = default,
    rows: u64, complete_rows: u64, arms: Vec<LevelCountWire>, columns: Vec<StalenessColumnWire>,
    joint: Option<JointProfileWire> = default, missing: Option<MissingPatternWire> = default,
    overlap: Option<OverlapSummaryWire> = default, checks: Vec<RecordedCheckWire>,
    outcome_model: Option<OutcomeModelWire> = default, seal: String });
