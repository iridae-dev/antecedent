//! Shared compiler and source-replay contract for one bounded categorical candidate.
//!
//! This module validates declarations and derives identities only. It performs no inference,
//! proves no independence or causal claim, and constructs no measured authority. The caller
//! explicitly declares independent subjects and supplies one distinct subject ID per row, the
//! declared treatment levels (the first is the reference) and a label for every row.
use std::collections::BTreeSet;

use crate::identity::{CanonicalWriter, SemanticDigest};
use crate::sampling_design::SamplingDesign;

const MAGIC: &[u8] = b"ACPC\x02";
/// Fewest declared treatment levels.
pub const MIN_LEVELS: usize = 2;
/// Most declared treatment levels; the level index is one byte on the wire.
pub const MAX_LEVELS: usize = 8;
/// Fewest rows the conjugate posterior is admitted on.
pub const MIN_ROWS: usize = 30;
/// Most rows: one row per declared subject, bounded by the source byte cap below.
pub const MAX_ROWS: usize = 4_096;
/// Fewest rows of every declared level; a level with fewer has no estimable mean.
pub const MIN_LEVEL_ROWS: usize = 2;
/// Longest snapshot, level label or subject identity, in bytes.
pub const MAX_TEXT_BYTES: usize = 256;
/// Exact public byte bound for the original-source representation: the largest admitted
/// design with every text at its longest.
pub const MAX_SOURCE_BYTES: usize = MAGIC.len()
    + (2 + MAX_TEXT_BYTES)
    + 1
    + MAX_LEVELS * (2 + MAX_TEXT_BYTES)
    + 4
    + MAX_ROWS * (8 + 1 + 2 + MAX_TEXT_BYTES);
/// Fixed prior precision of each level coefficient.
pub const LEVEL_PRECISION: f64 = 1.0;
/// Fixed prior precision of the intercept.
pub const NUISANCE_PRECISION: f64 = 0.5;
/// Fixed inverse-gamma shape.
pub const VARIANCE_SHAPE: f64 = 2.0;
/// Fixed inverse-gamma scale.
pub const VARIANCE_SCALE: f64 = 1.0;
/// Fixed equal-tailed credible level.
pub const CREDIBLE_LEVEL: f64 = 0.95;
/// Name of the fixed prior in the lowered statistical program.
pub const PRIOR_NAME: &str = "normal_inverse_gamma_lp1_np0_5_shape2_scale1";

/// Bytes admitted for one execution: the dummy-coded design, the outcome and a fixed overhead.
#[must_use]
pub fn admission_bytes(rows: usize, levels: usize) -> u64 {
    let rows = u64::try_from(rows).unwrap_or(u64::MAX);
    let width = u64::try_from(levels).unwrap_or(u64::MAX).saturating_add(1);
    32_768_u64.saturating_add(rows.saturating_mul(width.saturating_mul(8)))
}

/// A validated, bounded design source and its shared compiler identity.
#[derive(Clone, Debug)]
pub struct CategoricalCandidateSource {
    outcomes: Vec<f64>,
    levels: Vec<String>,
    declared_levels: Vec<String>,
    subject_ids: Vec<String>,
    snapshot: String,
    source: [u8; 32],
    query: [u8; 32],
    program: SemanticDigest,
}

impl CategoricalCandidateSource {
    /// Validate the declared design and derive all identities from the actual data.
    ///
    /// # Errors
    /// A row is nonfinite, IDs repeat, source strings are invalid, the design is not
    /// independent-subject, the row or level counts leave their bounds, a row names an
    /// undeclared level or a declared level has too few rows.
    pub fn new(
        outcomes: Vec<f64>,
        levels: Vec<String>,
        declared_levels: Vec<String>,
        subject_ids: Vec<String>,
        snapshot: String,
        design: SamplingDesign,
    ) -> Result<Self, &'static str> {
        let rows = outcomes.len();
        if design != SamplingDesign::IndependentSubjectTrajectories {
            return Err("categorical candidate requires independent subject trajectories");
        }
        if !(MIN_ROWS..=MAX_ROWS).contains(&rows)
            || levels.len() != rows
            || subject_ids.len() != rows
        {
            return Err("categorical candidate row count is outside its bounds or misaligned");
        }
        if outcomes.iter().any(|value| !value.is_finite()) {
            return Err("categorical candidate outcome is not finite");
        }
        let valid_text = |text: &String| !text.is_empty() && text.len() <= MAX_TEXT_BYTES;
        if !(MIN_LEVELS..=MAX_LEVELS).contains(&declared_levels.len())
            || !declared_levels.iter().all(valid_text)
            || declared_levels.iter().collect::<BTreeSet<_>>().len() != declared_levels.len()
        {
            return Err("categorical candidate declared levels are invalid");
        }
        let mut counts = vec![0_usize; declared_levels.len()];
        for level in &levels {
            let Some(index) = declared_levels.iter().position(|declared| declared == level) else {
                return Err("categorical candidate row names an undeclared level");
            };
            counts[index] += 1;
        }
        if counts.iter().any(|&count| count < MIN_LEVEL_ROWS) {
            return Err("categorical candidate declared level has too few rows");
        }
        if !subject_ids.iter().all(valid_text)
            || subject_ids.iter().collect::<BTreeSet<_>>().len() != rows
        {
            return Err("categorical candidate subject identities are invalid or repeated");
        }
        if !valid_text(&snapshot) {
            return Err("categorical candidate snapshot is invalid");
        }
        let mut source = CanonicalWriter::new("antecedent.categorical_candidate.source.v2");
        source
            .text(&snapshot)
            .reals(&outcomes)
            .texts(&levels)
            .texts(&declared_levels)
            .text(design.as_str())
            .texts(&subject_ids);
        let source = source.finish();
        let mut query = CanonicalWriter::new("antecedent.categorical_candidate.declaration.v2");
        query
            .text("declared_levels_first_is_reference")
            .word(declared_levels.len() as u64)
            .text("gaussian_normal_inverse_gamma")
            .real(LEVEL_PRECISION)
            .real(NUISANCE_PRECISION)
            .real(VARIANCE_SHAPE)
            .real(VARIANCE_SCALE)
            .real(CREDIBLE_LEVEL)
            .text("level_minus_reference")
            .text("one_row_per_declared_independent_subject")
            .bytes(&source);
        let query = query.finish();
        // The dummy-coded design has an intercept and one column per non-reference level.
        let program = program_identity(&source, &query, declared_levels.len() as u64);
        Ok(Self {
            outcomes,
            levels,
            declared_levels,
            subject_ids,
            snapshot,
            source,
            query,
            program,
        })
    }

    /// Original finite outcomes, in row order.
    #[must_use]
    pub fn outcomes(&self) -> &[f64] {
        &self.outcomes
    }
    /// Original per-row level labels, in row order.
    #[must_use]
    pub fn levels(&self) -> &[String] {
        &self.levels
    }
    /// Declared treatment levels; the first is the reference.
    #[must_use]
    pub fn declared_levels(&self) -> &[String] {
        &self.declared_levels
    }
    /// The reference level every reported contrast is taken against.
    #[must_use]
    pub fn reference(&self) -> &str {
        &self.declared_levels[0]
    }
    /// Number of rows, equal to the number of declared subjects.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.outcomes.len()
    }
    /// Declared distinct subject identities, in row order.
    #[must_use]
    pub fn subject_ids(&self) -> &[String] {
        &self.subject_ids
    }
    /// Caller-declared source snapshot identity.
    #[must_use]
    pub fn snapshot(&self) -> &str {
        &self.snapshot
    }
    /// Identity of the original data and declared subject design.
    #[must_use]
    pub const fn source_digest(&self) -> [u8; 32] {
        self.source
    }
    /// Identity of the query and model declaration.
    #[must_use]
    pub const fn query_digest(&self) -> [u8; 32] {
        self.query
    }
    /// Expected identity of the complete six-layer lowered program.
    #[must_use]
    pub const fn program_identity(&self) -> SemanticDigest {
        self.program
    }

    /// Canonical bounded source bytes; these are data and carry no scientific authority.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MAX_SOURCE_BYTES.min(64 + self.rows() * 32));
        bytes.extend_from_slice(MAGIC);
        put_text(&mut bytes, &self.snapshot);
        // Validated at construction: at most MAX_LEVELS declared levels and MAX_ROWS rows.
        bytes.push(u8::try_from(self.declared_levels.len()).unwrap_or(u8::MAX));
        for level in &self.declared_levels {
            put_text(&mut bytes, level);
        }
        bytes.extend_from_slice(&u32::try_from(self.rows()).unwrap_or(u32::MAX).to_le_bytes());
        for value in &self.outcomes {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        for level in &self.levels {
            let index = self.declared_levels.iter().position(|declared| declared == level);
            bytes.push(index.and_then(|index| u8::try_from(index).ok()).unwrap_or(u8::MAX));
        }
        for subject in &self.subject_ids {
            put_text(&mut bytes, subject);
        }
        bytes
    }

    /// Decode an untrusted source and revalidate the entire typed declaration.
    ///
    /// # Errors
    /// Invalid, oversized, truncated, noncanonical, unsupported or scientifically out-of-scope source.
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() > MAX_SOURCE_BYTES {
            return Err("categorical candidate source exceeds its byte bound");
        }
        let mut reader = Reader { bytes, offset: 0 };
        if reader.take(MAGIC.len())? != MAGIC {
            return Err("unsupported categorical candidate source version");
        }
        let snapshot = reader.text()?;
        let level_count = usize::from(reader.take(1)?[0]);
        if !(MIN_LEVELS..=MAX_LEVELS).contains(&level_count) {
            return Err("categorical candidate level count is outside its bounds");
        }
        let mut declared_levels = Vec::with_capacity(level_count);
        for _ in 0..level_count {
            declared_levels.push(reader.text()?);
        }
        let raw: [u8; 4] = reader.take(4)?.try_into().map_err(|_| "invalid row count")?;
        let rows = usize::try_from(u32::from_le_bytes(raw)).map_err(|_| "invalid row count")?;
        if !(MIN_ROWS..=MAX_ROWS).contains(&rows) {
            return Err("categorical candidate row count is outside its bounds");
        }
        let mut outcomes = Vec::with_capacity(rows);
        for _ in 0..rows {
            let raw: [u8; 8] = reader.take(8)?.try_into().map_err(|_| "invalid outcome")?;
            outcomes.push(f64::from_bits(u64::from_le_bytes(raw)));
        }
        let mut levels = Vec::with_capacity(rows);
        for &index in reader.take(rows)? {
            let level = declared_levels
                .get(usize::from(index))
                .ok_or("invalid categorical candidate level")?;
            levels.push(level.clone());
        }
        let mut subject_ids = Vec::with_capacity(rows);
        for _ in 0..rows {
            subject_ids.push(reader.text()?);
        }
        if reader.offset != bytes.len() {
            return Err("trailing categorical candidate source bytes");
        }
        let compiled = Self::new(
            outcomes,
            levels,
            declared_levels,
            subject_ids,
            snapshot,
            SamplingDesign::IndependentSubjectTrajectories,
        )?;
        if compiled.encode() != bytes {
            return Err("noncanonical categorical candidate source");
        }
        Ok(compiled)
    }
}

fn put_text(bytes: &mut Vec<u8>, text: &str) {
    // All fields were validated at construction to contain at most MAX_TEXT_BYTES bytes.
    let len = u16::try_from(text.len()).unwrap_or(u16::MAX);
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], &'static str> {
        let end = self.offset.checked_add(len).ok_or("candidate source length overflow")?;
        let value = self.bytes.get(self.offset..end).ok_or("truncated candidate source")?;
        self.offset = end;
        Ok(value)
    }
    fn text(&mut self) -> Result<String, &'static str> {
        let raw: [u8; 2] = self.take(2)?.try_into().map_err(|_| "invalid text length")?;
        let len = usize::from(u16::from_le_bytes(raw));
        if len == 0 || len > MAX_TEXT_BYTES {
            return Err("invalid candidate text length");
        }
        Ok(std::str::from_utf8(self.take(len)?)
            .map_err(|_| "candidate text is not UTF-8")?
            .to_owned())
    }
}

// Independently reproduce the precise semantic identities written by the six-layer compiler.
// The main compiler must compare its ordinary LoweredProgram identity with this result. Any
// future compiler-layer change therefore refuses until this contract is changed in lockstep.
// `columns` is the encoded design width (intercept plus one column per non-reference level);
// row counts are excluded from the statistical identity by the compiler itself.
fn program_identity(source: &[u8; 32], query: &[u8; 32], columns: u64) -> SemanticDigest {
    let mut statistical = CanonicalWriter::new("antecedent.lowered_program.statistical.v2");
    statistical
        .opt_text(None)
        .opt_text(None)
        .text("categorical_posterior_candidate_v1")
        .word(1)
        .text("outcome")
        .text("native.categorical_normal_inverse_gamma")
        .word(0)
        .opt_word(None)
        .word(0)
        .word(0)
        .word(1)
        .text("conjugate")
        .word(0)
        .text(PRIOR_NAME)
        .opt_word(Some(columns))
        .text("quoted")
        .word(0);
    let statistical = statistical.finish();
    let mut evidence = CanonicalWriter::new("antecedent.lowered_program.evidence");
    evidence.word(1).text("observations").word(1).bytes(source).text("single").word(0);
    let evidence = evidence.finish();
    let source_hex = SemanticDigest::from_bytes(*source).to_hex();
    let mut program = CanonicalWriter::new("antecedent.lowered_program");
    program
        .text("CategoricalPosteriorCandidate")
        .text("complete")
        .text("query_functional")
        .text("association")
        .opt_word(None)
        .bytes(query)
        .text("compiled_categorical_posterior_conjugate_candidate")
        .text("tabular")
        .text("not_applicable")
        .word(0)
        .opt_text(Some(source_hex.as_str()))
        .opt_text(None)
        .word(0)
        .text("not_claimed")
        .opt_text(None)
        .bytes(&statistical)
        .bytes(&evidence)
        .text("posterior");
    SemanticDigest::from_bytes(program.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(counts: &[usize]) -> (Vec<String>, Vec<String>) {
        let declared: Vec<String> = (0..counts.len()).map(|i| format!("level-{i}")).collect();
        let rows = counts
            .iter()
            .enumerate()
            .flat_map(|(i, &count)| (0..count).map(|_| declared[i].clone()).collect::<Vec<_>>())
            .collect();
        (declared, rows)
    }

    fn build(
        counts: &[usize],
        design: SamplingDesign,
    ) -> Result<CategoricalCandidateSource, &'static str> {
        let (declared, levels) = labels(counts);
        let rows = levels.len();
        CategoricalCandidateSource::new(
            (0..rows).map(|i| f64::from(u32::try_from(i).unwrap()) * 0.5).collect(),
            levels,
            declared,
            (0..rows).map(|i| format!("subject-{i}")).collect(),
            "source-a".into(),
            design,
        )
    }

    fn source() -> CategoricalCandidateSource {
        build(&[20, 20], SamplingDesign::IndependentSubjectTrajectories).unwrap()
    }

    #[test]
    fn source_round_trips_and_each_subject_is_part_of_the_compiler_identity() {
        let original = source();
        let encoded = original.encode();
        let replayed = CategoricalCandidateSource::decode(&encoded).unwrap();
        assert_eq!(original.source_digest(), replayed.source_digest());
        assert_eq!(original.program_identity(), replayed.program_identity());
        let mut ids = original.subject_ids().to_vec();
        ids[0] = "different-subject".into();
        let changed = CategoricalCandidateSource::new(
            original.outcomes().to_vec(),
            original.levels().to_vec(),
            original.declared_levels().to_vec(),
            ids,
            original.snapshot().into(),
            SamplingDesign::IndependentSubjectTrajectories,
        )
        .unwrap();
        assert_ne!(original.program_identity(), changed.program_identity());
        assert!(CategoricalCandidateSource::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn general_shapes_round_trip_and_the_level_count_is_part_of_the_program() {
        let mut identities = BTreeSet::new();
        for counts in [
            vec![15, 15],
            vec![10, 10, 10],
            vec![2, 2, 26],
            vec![4, 4, 4, 4, 4, 4, 4, 4],
            vec![3, 2, 5, 40, 2, 7, 9, 2],
            vec![MAX_ROWS - 2, 2],
        ] {
            let original = build(&counts, SamplingDesign::IndependentSubjectTrajectories).unwrap();
            assert_eq!(original.declared_levels().len(), counts.len());
            assert_eq!(original.rows(), counts.iter().sum::<usize>());
            let encoded = original.encode();
            assert!(encoded.len() <= MAX_SOURCE_BYTES);
            let replayed = CategoricalCandidateSource::decode(&encoded).unwrap();
            assert_eq!(replayed.encode(), encoded);
            assert_eq!(original.program_identity(), replayed.program_identity());
            identities.insert(original.program_identity().to_hex());
        }
        assert_eq!(identities.len(), 6);
    }

    #[test]
    fn the_source_byte_cap_holds_the_largest_admitted_design() {
        let (declared, levels) = {
            let declared: Vec<String> =
                (0..MAX_LEVELS).map(|i| format!("{i}{}", "l".repeat(MAX_TEXT_BYTES - 1))).collect();
            let rows = (0..MAX_ROWS).map(|i| declared[i % MAX_LEVELS].clone()).collect();
            (declared, rows)
        };
        let widest = CategoricalCandidateSource::new(
            vec![1.5; MAX_ROWS],
            levels,
            declared,
            (0..MAX_ROWS).map(|i| format!("{i:0>MAX_TEXT_BYTES$}")).collect(),
            "s".repeat(MAX_TEXT_BYTES),
            SamplingDesign::IndependentSubjectTrajectories,
        )
        .unwrap();
        assert_eq!(widest.encode().len(), MAX_SOURCE_BYTES);
        assert!(CategoricalCandidateSource::decode(&widest.encode()).is_ok());
        let mut over = widest.encode();
        over.push(0);
        assert!(CategoricalCandidateSource::decode(&over).is_err());
    }

    #[test]
    fn each_violation_is_refused() {
        let design = SamplingDesign::IndependentSubjectTrajectories;
        // Row bounds: one below the minimum, one above the maximum.
        assert!(build(&[14, 15], design).is_err());
        assert!(build(&[MAX_ROWS - 1, 2], design).is_err());
        // A level with too few rows, and level-count bounds.
        assert!(build(&[29, 1], design).is_err());
        assert!(build(&[30], design).is_err());
        assert!(build(&[4; 9], design).is_err());
        // A dependent design.
        assert!(build(&[20, 20], SamplingDesign::DependentTimeSeries).is_err());
        let original = source();
        let new =
            |outcomes: Vec<f64>, levels: Vec<String>, declared: Vec<String>, ids: Vec<String>| {
                CategoricalCandidateSource::new(
                    outcomes,
                    levels,
                    declared,
                    ids,
                    "source-a".into(),
                    design,
                )
            };
        let parts = || {
            (
                original.outcomes().to_vec(),
                original.levels().to_vec(),
                original.declared_levels().to_vec(),
                original.subject_ids().to_vec(),
            )
        };
        let (o, l, d, i) = parts();
        assert!(new(o, l, d, i).is_ok());
        // Nonfinite outcomes.
        for bad in [f64::NAN, f64::INFINITY] {
            let (mut o, l, d, i) = parts();
            o[3] = bad;
            assert!(new(o, l, d, i).is_err());
        }
        // Repeated subjects.
        let (o, l, d, mut i) = parts();
        i[1] = i[0].clone();
        assert!(new(o, l, d, i).is_err());
        // An undeclared row label and repeated or empty declared levels.
        let (o, mut l, d, i) = parts();
        l[0] = "other".into();
        assert!(new(o, l, d, i).is_err());
        let (o, l, mut d, i) = parts();
        d[1] = d[0].clone();
        assert!(new(o, l, d, i).is_err());
        let (o, mut l, mut d, i) = parts();
        d[0] = String::new();
        l.iter_mut().filter(|level| level.as_str() == "level-0").for_each(String::clear);
        assert!(new(o, l, d, i).is_err());
        // Row misalignment.
        let (o, l, d, mut i) = parts();
        i.pop();
        assert!(new(o, l, d, i).is_err());
    }

    #[test]
    fn noncanonical_or_hostile_bytes_are_refused() {
        let encoded = source().encode();
        for cut in [0, 4, 5, 10, encoded.len() / 2, encoded.len() - 1] {
            assert!(CategoricalCandidateSource::decode(&encoded[..cut]).is_err());
        }
        // Level count of one and nine; a row count below its minimum; an out-of-range level index.
        for (offset, value) in [(5 + 2 + "source-a".len(), 1_u8), (5 + 2 + "source-a".len(), 9)] {
            let mut bytes = encoded.clone();
            bytes[offset] = value;
            assert!(CategoricalCandidateSource::decode(&bytes).is_err());
        }
        let mut bytes = encoded.clone();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        assert!(CategoricalCandidateSource::decode(&bytes).is_err());
        let rows_offset = 5 + 2 + "source-a".len() + 1 + 2 * (2 + "level-0".len());
        let mut bytes = encoded.clone();
        bytes[rows_offset] = 1;
        assert!(CategoricalCandidateSource::decode(&bytes).is_err());
        let mut bytes = encoded.clone();
        bytes[rows_offset + 4 + 8 * 40] = 7;
        assert!(CategoricalCandidateSource::decode(&bytes).is_err());
        let mut bytes = encoded;
        bytes.extend_from_slice(&[0; 3]);
        assert!(CategoricalCandidateSource::decode(&bytes).is_err());
    }
}
