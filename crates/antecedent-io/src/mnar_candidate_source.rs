//! Source-bound reconstruction of the declared-shift MNAR candidate.
//!
//! This wire is not a causal claim, measured artifact or interval authority.
//! A consumer reconstructs the exact native source and its typed operator graph;
//! caller-supplied identities only constrain that reconstruction.
use crate::IoError;
use antecedent_core::{
    ExecutionContext,
    mnar_fixed_shift::{
        FixedShiftProgram, FixedShiftSource, SamplingPremise, SourceError, SourceRow,
    },
};

const MAGIC: &[u8] = b"ANTECEDENT-MNAR-SOURCE\x03";
const MAX_POPULATION_BYTES: usize = 256;
// Population length, treatment, outcome, treated shift and row count.
const HEADER_BYTES: usize = 4 + 4 + 4 + 8 + 4;
// Subject, treatment flag, outcome flag and outcome bits.
const ROW_BYTES: usize = 8 + 1 + 1 + 8;

fn invalid(message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("mnar_fixed_shift.invalid_source: {message}"),
    }
}

fn source_refusal(error: SourceError) -> IoError {
    let detail = match error {
        SourceError::InvalidDeclaration => "mnar_fixed_shift.invalid_declaration",
        SourceError::DuplicateSubject => "mnar_fixed_shift.duplicate_subject",
        SourceError::InvalidOutcome => "mnar_fixed_shift.invalid_outcome",
        SourceError::InsufficientObservations => "mnar_fixed_shift.insufficient_observations",
        SourceError::ConstantOutcome => "mnar_fixed_shift.constant_outcome",
        SourceError::InvalidShift => "mnar_fixed_shift.invalid_shift",
    };
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("{detail}: {error}"),
    }
}

fn read_word(bytes: &[u8], pos: &mut usize) -> Result<u32, IoError> {
    let end = pos.checked_add(4).ok_or_else(|| invalid("offset overflow"))?;
    let value = u32::from_le_bytes(
        bytes
            .get(*pos..end)
            .ok_or_else(|| invalid("truncated word"))?
            .try_into()
            .map_err(|_| invalid("word"))?,
    );
    *pos = end;
    Ok(value)
}

fn admission_error(ctx: &ExecutionContext, message: impl std::fmt::Display) -> IoError {
    if ctx.cancellation.is_cancelled() {
        IoError::Refused {
            code: antecedent_core::reason_code!("execution_cancelled"),
            message: format!("mnar_fixed_shift.source_admission_cancelled: {message}"),
        }
    } else {
        IoError::Refused {
            code: antecedent_core::reason_code!("execution_resource_limit"),
            message: format!("mnar_fixed_shift.source_admission_resource_limit: {message}"),
        }
    }
}

/// Identities a caller retains separately from source bytes. These are not authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpectedMnarSource {
    /// Original source including row values and missingness.
    pub source: [u8; 32],
    /// Fixed model-conditional operation derived from the source declaration.
    pub program: [u8; 32],
}

/// Encoded candidate source with its admission lease and independent expectation.
pub struct EncodedMnarSource {
    bytes: Vec<u8>,
    expected: ExpectedMnarSource,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl EncodedMnarSource {
    /// Original encoded source.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Caller-retainable source and program constraint; not an authority.
    #[must_use]
    pub const fn expectation(&self) -> ExpectedMnarSource {
        self.expected
    }
}

/// Reconstructed native candidate and memory reservation; no interval is published.
pub struct ReconstructedMnarSource {
    source: FixedShiftSource,
    program: FixedShiftProgram,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl ReconstructedMnarSource {
    /// Validated original source.
    #[must_use]
    pub fn source(&self) -> &FixedShiftSource {
        &self.source
    }
    /// Native operators independently reconstructed from that source.
    #[must_use]
    pub fn program(&self) -> &FixedShiftProgram {
        &self.program
    }
}

/// Encode a validated source and return separately retainable identities.
///
/// # Errors
/// The context refuses admission for the encoded size, or cancels.
pub fn encode(
    source: &FixedShiftSource,
    ctx: &ExecutionContext,
) -> Result<EncodedMnarSource, IoError> {
    let program =
        FixedShiftProgram::compile(source).map_err(|_| invalid("operator declaration rejected"))?;
    let len = MAGIC
        .len()
        .checked_add(HEADER_BYTES)
        .and_then(|n| n.checked_add(source.population().len()))
        .and_then(|n| {
            source.rows().len().checked_mul(ROW_BYTES).and_then(|rows| n.checked_add(rows))
        })
        .ok_or_else(|| invalid("source size overflow"))?;
    let lease = ctx.acquire_admission(0, len as u64).map_err(|e| admission_error(ctx, e))?;
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(
        &u32::try_from(source.population().len())
            .map_err(|_| invalid("population too long"))?
            .to_le_bytes(),
    );
    out.extend_from_slice(source.population().as_bytes());
    out.extend_from_slice(&source.treatment().to_le_bytes());
    out.extend_from_slice(&source.outcome().to_le_bytes());
    out.extend_from_slice(&source.treated_shift().to_bits().to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(source.rows().len()).map_err(|_| invalid("too many rows"))?.to_le_bytes(),
    );
    for row in source.rows() {
        if ctx.cancellation.is_cancelled() {
            return Err(admission_error(ctx, "source export cancelled"));
        }
        out.extend_from_slice(&row.subject_id.to_le_bytes());
        out.push(u8::from(row.treated));
        out.push(u8::from(row.outcome.is_some()));
        out.extend_from_slice(&row.outcome.map_or(0, f64::to_bits).to_le_bytes());
    }
    Ok(EncodedMnarSource {
        bytes: out,
        expected: ExpectedMnarSource {
            source: source.source_digest(),
            program: *program.identity(),
        },
        _lease: lease,
    })
}

/// Decode original source rows, rebuild the native program and check both independently
/// retained identities. A successful result carries no measured or supported authority.
///
/// # Errors
/// Invalid/corrupt source, identity mismatch, admission refusal, or cancellation.
pub fn reconstruct(
    bytes: &[u8],
    expected: &ExpectedMnarSource,
    ctx: &ExecutionContext,
) -> Result<ReconstructedMnarSource, IoError> {
    let decoded = decode_source(bytes, ctx)?;
    if decoded.source.source_digest() != expected.source
        || decoded.program.identity() != &expected.program
    {
        return Err(invalid("source or operation differs from retained expectation"));
    }
    Ok(decoded)
}

/// Decode and independently reconstruct the bounded native source, without granting authority.
///
/// # Errors
/// Malformed source, cancellation, or resource refusal.
pub fn decode_source(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<ReconstructedMnarSource, IoError> {
    if !bytes.starts_with(MAGIC) {
        return Err(invalid("invalid source header"));
    }
    let lease = ctx
        .acquire_admission(0, (bytes.len() as u64).saturating_mul(4))
        .map_err(|e| admission_error(ctx, e))?;
    let mut pos = MAGIC.len();
    let name_len = read_word(bytes, &mut pos)? as usize;
    if name_len == 0 || name_len > MAX_POPULATION_BYTES {
        return Err(invalid("population length"));
    }
    let end = pos.checked_add(name_len).ok_or_else(|| invalid("offset overflow"))?;
    let population =
        std::str::from_utf8(bytes.get(pos..end).ok_or_else(|| invalid("truncated population"))?)
            .map_err(|_| invalid("population is not UTF-8"))?
            .to_owned();
    pos = end;
    let treatment = read_word(bytes, &mut pos)?;
    let outcome = read_word(bytes, &mut pos)?;
    let shift_end = pos.checked_add(8).ok_or_else(|| invalid("offset overflow"))?;
    let treated_shift = f64::from_bits(u64::from_le_bytes(
        bytes
            .get(pos..shift_end)
            .ok_or_else(|| invalid("truncated treated shift"))?
            .try_into()
            .map_err(|_| invalid("treated shift"))?,
    ));
    pos = shift_end;
    let count = read_word(bytes, &mut pos)? as usize;
    if count == 0
        || count.checked_mul(ROW_BYTES).and_then(|n| pos.checked_add(n)) != Some(bytes.len())
    {
        return Err(invalid("row count or encoded length"));
    }
    let mut rows = Vec::with_capacity(count);
    for chunk in bytes[pos..].chunks_exact(ROW_BYTES) {
        if ctx.cancellation.is_cancelled() {
            return Err(admission_error(ctx, "source reconstruction cancelled"));
        }
        let subject_id =
            u64::from_le_bytes(chunk[..8].try_into().map_err(|_| invalid("subject identity"))?);
        let treated = match chunk[8] {
            0 => false,
            1 => true,
            _ => return Err(invalid("treatment flag")),
        };
        let bits =
            u64::from_le_bytes(chunk[10..18].try_into().map_err(|_| invalid("outcome payload"))?);
        let value = match chunk[9] {
            0 if bits == 0 => None,
            1 => Some(f64::from_bits(bits)),
            _ => return Err(invalid("outcome flag or noncanonical missing value")),
        };
        rows.push(SourceRow { subject_id, treated, outcome: value });
    }
    let source = FixedShiftSource::new(
        population,
        treatment,
        outcome,
        SamplingPremise::IndependentRandomizedArms,
        treated_shift,
        rows,
    )
    .map_err(source_refusal)?;
    let program = FixedShiftProgram::compile(&source)
        .map_err(|_| invalid("operator declaration rejected"))?;
    if !program.verifies(&source) {
        return Err(invalid("source or operation failed reconstruction"));
    }
    Ok(ReconstructedMnarSource { source, program, _lease: lease })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(control: (usize, usize), treated: (usize, usize)) -> Vec<SourceRow> {
        let arm = |first: u64, is_treated: bool, (assigned, observed): (usize, usize)| {
            (0..assigned)
                .map(move |i| SourceRow {
                    subject_id: first + i as u64,
                    treated: is_treated,
                    outcome: (i < observed).then(|| {
                        f64::from(u32::try_from((i * 7) % 11).unwrap()) / 3.0
                            + f64::from(u8::from(is_treated))
                    }),
                })
                .collect::<Vec<_>>()
        };
        let mut rows = arm(0, false, control);
        rows.extend(arm(100_000, true, treated));
        rows
    }

    fn source(shift: f64) -> FixedShiftSource {
        FixedShiftSource::new(
            "population".into(),
            0,
            1,
            SamplingPremise::IndependentRandomizedArms,
            shift,
            rows((30, 24), (40, 28)),
        )
        .unwrap()
    }

    #[test]
    fn source_reconstruction_checks_original_rows_and_native_program() {
        let ctx = ExecutionContext::for_tests(71);
        let original = source(0.75);
        let artifact = encode(&original, &ctx).unwrap();
        let expected = artifact.expectation();
        let bytes = artifact.bytes();
        let reconstructed = reconstruct(bytes, &expected, &ctx).unwrap();
        assert_eq!(reconstructed.source(), &original);
        assert!(reconstructed.program().verifies(&original));
        let mut edited = bytes.to_vec();
        let last = edited.len() - 1;
        edited[last] ^= 1;
        assert!(reconstruct(&edited, &expected, &ctx).is_err());
        let wrong = ExpectedMnarSource { source: [7; 32], ..expected };
        assert!(reconstruct(bytes, &wrong, &ctx).is_err());
        assert!(reconstruct(&bytes[..bytes.len() - 1], &expected, &ctx).is_err());
    }

    #[test]
    fn declared_shift_travels_in_the_wire_and_the_expectation() {
        let ctx = ExecutionContext::for_tests(72);
        let a = encode(&source(0.5), &ctx).unwrap();
        let b = encode(&source(-0.5), &ctx).unwrap();
        assert_ne!(a.expectation(), b.expectation());
        assert_ne!(a.bytes(), b.bytes());
        let decoded = decode_source(b.bytes(), &ctx).unwrap();
        assert!((decoded.source().treated_shift() + 0.5).abs() < 1e-15);
        assert!(reconstruct(b.bytes(), &a.expectation(), &ctx).is_err());
    }

    #[test]
    fn encoded_length_is_exact_and_follows_the_declared_row_count() {
        let ctx = ExecutionContext::for_tests(73);
        let small = encode(&source(0.0), &ctx).unwrap();
        assert_eq!(
            small.bytes().len(),
            MAGIC.len() + HEADER_BYTES + "population".len() + 70 * ROW_BYTES
        );
        let large = FixedShiftSource::new(
            "p".repeat(MAX_POPULATION_BYTES),
            0,
            1,
            SamplingPremise::IndependentRandomizedArms,
            0.0,
            rows((12_000, 4000), (12_000, 4000)),
        )
        .unwrap();
        let encoded = encode(&large, &ctx).unwrap();
        assert_eq!(
            encoded.bytes().len(),
            MAGIC.len() + HEADER_BYTES + MAX_POPULATION_BYTES + 24_000 * ROW_BYTES
        );
        assert!(decode_source(encoded.bytes(), &ctx).is_ok());
    }

    #[test]
    fn decoding_refuses_with_the_specific_source_error() {
        let ctx = ExecutionContext::for_tests(74);
        let refusal = |bytes: &[u8]| match decode_source(bytes, &ctx) {
            Err(IoError::Refused { message, .. }) => message,
            _ => panic!("expected a refusal"),
        };
        let encoded = encode(&source(0.0), &ctx).unwrap();
        let header = MAGIC.len() + 4 + "population".len();
        let mut nan = encoded.bytes().to_vec();
        nan[header + 8..header + 16].copy_from_slice(&f64::NAN.to_bits().to_le_bytes());
        assert!(refusal(&nan).starts_with("mnar_fixed_shift.invalid_shift:"));
        // A duplicated subject identity.
        let mut duplicate = encoded.bytes().to_vec();
        let first = header + 16 + 4;
        let second = first + ROW_BYTES;
        let copy = duplicate[first..first + 8].to_vec();
        duplicate[second..second + 8].copy_from_slice(&copy);
        assert!(refusal(&duplicate).starts_with("mnar_fixed_shift.duplicate_subject:"));
        // A count that disagrees with the encoded length.
        let mut short = encoded.bytes().to_vec();
        short[header + 16..header + 20].copy_from_slice(&69u32.to_le_bytes());
        assert!(refusal(&short).starts_with("mnar_fixed_shift.invalid_source:"));
    }
}
