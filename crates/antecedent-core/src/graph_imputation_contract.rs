//! Shared compiler and source-replay contract for the graph-selection imputation candidate.
//!
//! This module validates a declaration and derives identities only. It performs no imputation,
//! graph scoring or inference and constructs no measured authority. The compiler states, for
//! every supported graph, the operation its retained child preparation selected: the
//! adjustment set and the identification status. A consumer re-derives each of those from the
//! graph itself and refuses unless they are equal, so the compiler's checked operations are
//! replayed rather than recompiled from the source alone.
//!
//! The declared support is the structural support further restricted by [`SUPPORT_RULE`]: a
//! graph whose treatment-to-outcome effect has more than one minimal back-door adjustment set
//! names no single operation and is excluded. Compiler and consumer derive the exclusion from the
//! identifier's own output, the retained atoms are exactly the graphs that remain, and the
//! number excluded is part of the contract, so a consumer cannot be given a different set.
use std::collections::BTreeSet;

use crate::identification::IdentificationStatus;
use crate::identity::CanonicalWriter;

/// Rule that restricts the structural support to graphs naming a single operation.
pub const SUPPORT_RULE: &str = "unique_minimal_backdoor_adjustment_set";
/// Variable count above which the exact graph enumeration is not attempted.
pub const MAX_VARIABLES: usize = 4;
/// Fewest rows of a replayable source.
pub const MIN_ROWS: usize = 4;
/// Most rows of a replayable source.
pub const MAX_ROWS: usize = 10_000;
/// Longest variable name.
pub const MAX_NAME_BYTES: usize = 256;
/// Most graph atoms: the labeled DAGs on four nodes.
pub const MAX_ATOMS: usize = 543;
/// Exact public byte bound of the encoded source.
pub const MAX_SOURCE_BYTES: usize =
    4096 + MAX_VARIABLES * (MAX_ROWS * 8 + MAX_NAME_BYTES + 16) + MAX_ATOMS * 64;

const MAGIC: &[u8] = b"ANTECEDENT-GRAPH-CANDIDATE\x01";

/// Structural support declared before any observation is scored.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphSupport {
    /// Treatment has no parents and a direct arrow to the outcome. The empty set identifies the
    /// same total effect in every graph, so the graphs differ only in how the other variables
    /// are modeled.
    TreatmentRootDirectEffect,
    /// Treatment has a direct arrow to the outcome and may have parents. Adjustment sets differ
    /// between graphs, so the mixture carries uncertainty about which variables confound.
    TreatmentDirectEffect,
}

impl GraphSupport {
    /// Stable wire tag.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::TreatmentRootDirectEffect => 1,
            Self::TreatmentDirectEffect => 2,
        }
    }
    /// Inverse of [`Self::tag`].
    #[must_use]
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::TreatmentRootDirectEffect),
            2 => Some(Self::TreatmentDirectEffect),
            _ => None,
        }
    }
    /// Stable label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TreatmentRootDirectEffect => "treatment_root_direct_effect",
            Self::TreatmentDirectEffect => "treatment_direct_effect",
        }
    }
    /// Inverse of [`Self::as_str`].
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        [Self::TreatmentRootDirectEffect, Self::TreatmentDirectEffect]
            .into_iter()
            .find(|support| support.as_str() == label)
    }
}

/// The operation a compiler retained for one graph: its identification status and the
/// adjustment set of the g-computation, in design-column order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtomOperation {
    /// Exact adjacency mask of the graph.
    pub mask: u64,
    /// Identification status of the graph's effect proof.
    pub status: IdentificationStatus,
    /// Dense raw ids of the adjusted variables.
    pub adjustment: Vec<u32>,
}

fn status_tag(status: IdentificationStatus) -> Option<u8> {
    match status {
        IdentificationStatus::NonparametricallyIdentified => Some(0),
        IdentificationStatus::IdentifiedUnderParametricRestrictions => Some(1),
        _ => None,
    }
}

fn status_from_tag(tag: u8) -> Option<IdentificationStatus> {
    match tag {
        0 => Some(IdentificationStatus::NonparametricallyIdentified),
        1 => Some(IdentificationStatus::IdentifiedUnderParametricRestrictions),
        _ => None,
    }
}

/// Every declared fact of a replayable source, before the compiler's program identity.
#[derive(Clone, Debug)]
pub struct GraphImputationParts {
    /// Variable names in table order.
    pub names: Vec<String>,
    /// Column domains in table order: zero continuous, one binary.
    pub domains: Vec<u8>,
    /// Original columns, missing cells as NaN.
    pub columns: Vec<Vec<f64>>,
    /// Treatment variable raw id.
    pub treatment: u32,
    /// Outcome variable raw id.
    pub outcome: u32,
    /// Declared graph support.
    pub support: GraphSupport,
    /// Completed tables.
    pub imputations: u32,
    /// Chained-equation sweeps.
    pub iterations: u32,
    /// Imputation seed.
    pub imputation_seed: u64,
    /// Child effect draws per graph.
    pub posterior_draws: u32,
    /// Conjugate-model prior scale.
    pub prior_scale: f64,
    /// Seed of the child execution streams.
    pub execution_seed: u64,
    /// Content identity of the native table the compiler held.
    pub data_digest: [u8; 32],
    /// The compiler's retained operation of every retained graph, in mask order.
    pub atoms: Vec<AtomOperation>,
    /// Structurally supported graphs that [`SUPPORT_RULE`] excluded.
    pub excluded: u32,
}

impl GraphImputationParts {
    /// Row count.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, Vec::len)
    }

    /// Check every bound a replay depends on.
    ///
    /// # Errors
    /// The declaration lies outside the bounded cell.
    pub fn validate(&self) -> Result<(), &'static str> {
        let n = self.names.len();
        let rows = self.rows();
        if !(2..=MAX_VARIABLES).contains(&n)
            || self.domains.len() != n
            || self.columns.len() != n
            || !(MIN_ROWS..=MAX_ROWS).contains(&rows)
            || self.columns.iter().any(|column| column.len() != rows)
            || self.columns.iter().flatten().any(|value| value.is_infinite())
            || self.domains.iter().any(|&domain| domain > 1)
            || self.names.iter().any(|name| name.is_empty() || name.len() > MAX_NAME_BYTES)
            || self.names.iter().collect::<BTreeSet<_>>().len() != n
        {
            return Err("graph imputation source shape lies outside the bounded cell");
        }
        let (treatment, outcome) = (self.treatment as usize, self.outcome as usize);
        if treatment >= n
            || outcome >= n
            || treatment == outcome
            || self.domains[treatment] != 1
            || !(2..=64).contains(&self.imputations)
            || !(1..=64).contains(&self.iterations)
            || !(100..=4096).contains(&self.posterior_draws)
            || !self.prior_scale.is_finite()
            || self.prior_scale <= 0.0
        {
            return Err("graph imputation declaration lies outside the bounded cell");
        }
        let edge_bits = n * (n - 1);
        let mut previous = None;
        if self.atoms.is_empty() || self.atoms.len() > MAX_ATOMS {
            return Err("graph imputation atom count lies outside the bounded cell");
        }
        if self.excluded as usize > MAX_ATOMS {
            return Err("graph imputation excluded graph count lies outside the bounded cell");
        }
        for atom in &self.atoms {
            let mut seen = BTreeSet::new();
            if previous.is_some_and(|last| last >= atom.mask)
                || atom.mask >= 1u64 << edge_bits
                || status_tag(atom.status).is_none()
                || atom.adjustment.len() > n
                || atom.adjustment.iter().any(|&v| {
                    v as usize >= n || v == self.treatment || v == self.outcome || !seen.insert(v)
                })
            {
                return Err("graph imputation atom operation is invalid");
            }
            previous = Some(atom.mask);
        }
        Ok(())
    }

    /// Identity of the original named columns, domains and missing-cell bit patterns.
    #[must_use]
    pub fn source_digest(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("antecedent.graph_candidate.source.v1");
        writer.word(self.names.len() as u64);
        for ((name, domain), column) in self.names.iter().zip(&self.domains).zip(&self.columns) {
            writer.text(name).word(u64::from(*domain)).word(column.len() as u64);
            for value in column {
                writer.word(value.to_bits());
            }
        }
        writer.finish()
    }

    /// Identity of the whole declaration and of every retained child operation. The compiler
    /// seals this into its lowered program, so the program commits to the operations a
    /// consumer re-derives.
    #[must_use]
    pub fn contract_digest(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("antecedent.graph_candidate.contract.v1");
        writer
            .text(self.support.as_str())
            .text(SUPPORT_RULE)
            .word(u64::from(self.excluded))
            .bytes(&self.source_digest())
            .bytes(&self.data_digest)
            .word(u64::from(self.treatment))
            .word(u64::from(self.outcome))
            .word(u64::from(self.imputations))
            .word(u64::from(self.iterations))
            .word(self.imputation_seed)
            .word(u64::from(self.posterior_draws))
            .word(self.prior_scale.to_bits())
            .word(self.execution_seed)
            .word(self.atoms.len() as u64);
        for atom in &self.atoms {
            writer
                .word(atom.mask)
                .word(u64::from(status_tag(atom.status).unwrap_or(u8::MAX)))
                .word(atom.adjustment.len() as u64);
            for &variable in &atom.adjustment {
                writer.word(u64::from(variable));
            }
        }
        writer.finish()
    }
}

/// A validated source with the compiler's lowered program identity.
#[derive(Clone, Debug)]
pub struct GraphImputationSource {
    parts: GraphImputationParts,
    program: [u8; 32],
}

impl GraphImputationSource {
    /// Validate the declaration and attach the compiler's program identity.
    ///
    /// # Errors
    /// The declaration lies outside the bounded cell.
    pub fn new(parts: GraphImputationParts, program: [u8; 32]) -> Result<Self, &'static str> {
        parts.validate()?;
        Ok(Self { parts, program })
    }
    /// The declared facts.
    #[must_use]
    pub const fn parts(&self) -> &GraphImputationParts {
        &self.parts
    }
    /// The compiler's lowered program identity.
    #[must_use]
    pub const fn program(&self) -> [u8; 32] {
        self.program
    }

    /// Canonical bounded bytes; these are data and carry no scientific authority.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let parts = &self.parts;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(u8::try_from(parts.names.len()).unwrap_or(u8::MAX));
        bytes.extend_from_slice(&u32::try_from(parts.rows()).unwrap_or(u32::MAX).to_le_bytes());
        bytes.extend_from_slice(&parts.treatment.to_le_bytes());
        bytes.extend_from_slice(&parts.outcome.to_le_bytes());
        bytes.push(parts.support.tag());
        bytes.extend_from_slice(&parts.imputations.to_le_bytes());
        bytes.extend_from_slice(&parts.iterations.to_le_bytes());
        bytes.extend_from_slice(&parts.imputation_seed.to_le_bytes());
        bytes.extend_from_slice(&parts.posterior_draws.to_le_bytes());
        bytes.extend_from_slice(&parts.prior_scale.to_bits().to_le_bytes());
        bytes.extend_from_slice(&parts.execution_seed.to_le_bytes());
        bytes.extend_from_slice(&parts.data_digest);
        bytes.extend_from_slice(&self.program);
        bytes.extend_from_slice(&parts.excluded.to_le_bytes());
        bytes
            .extend_from_slice(&u32::try_from(parts.atoms.len()).unwrap_or(u32::MAX).to_le_bytes());
        for atom in &parts.atoms {
            bytes.extend_from_slice(&atom.mask.to_le_bytes());
            bytes.push(status_tag(atom.status).unwrap_or(u8::MAX));
            bytes.push(u8::try_from(atom.adjustment.len()).unwrap_or(u8::MAX));
            for variable in &atom.adjustment {
                bytes.extend_from_slice(&variable.to_le_bytes());
            }
        }
        for (name, domain) in parts.names.iter().zip(&parts.domains) {
            bytes.push(*domain);
            bytes.extend_from_slice(&u16::try_from(name.len()).unwrap_or(u16::MAX).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
        }
        for column in &parts.columns {
            for value in column {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        bytes
    }

    /// Decode an untrusted source and revalidate the entire declaration.
    ///
    /// # Errors
    /// Invalid, oversized, truncated, noncanonical or out-of-scope source.
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() > MAX_SOURCE_BYTES || !bytes.starts_with(MAGIC) {
            return Err("invalid graph imputation source header or size");
        }
        let mut reader = Reader { bytes, offset: MAGIC.len() };
        let n = usize::from(reader.byte()?);
        let rows = reader.u32()? as usize;
        if !(2..=MAX_VARIABLES).contains(&n) || !(MIN_ROWS..=MAX_ROWS).contains(&rows) {
            return Err("graph imputation source shape lies outside the bounded cell");
        }
        let treatment = reader.u32()?;
        let outcome = reader.u32()?;
        let support =
            GraphSupport::from_tag(reader.byte()?).ok_or("unknown graph imputation support")?;
        let imputations = reader.u32()?;
        let iterations = reader.u32()?;
        let imputation_seed = reader.u64()?;
        let posterior_draws = reader.u32()?;
        let prior_scale = f64::from_bits(reader.u64()?);
        let execution_seed = reader.u64()?;
        let data_digest = reader.digest()?;
        let program = reader.digest()?;
        let excluded = reader.u32()?;
        let atom_count = reader.u32()? as usize;
        if atom_count == 0 || atom_count > MAX_ATOMS {
            return Err("graph imputation atom count lies outside the bounded cell");
        }
        let mut atoms = Vec::with_capacity(atom_count);
        for _ in 0..atom_count {
            let mask = reader.u64()?;
            let status = status_from_tag(reader.byte()?).ok_or("unknown identification status")?;
            let length = usize::from(reader.byte()?);
            if length > n {
                return Err("graph imputation adjustment set is too long");
            }
            let mut adjustment = Vec::with_capacity(length);
            for _ in 0..length {
                adjustment.push(reader.u32()?);
            }
            atoms.push(AtomOperation { mask, status, adjustment });
        }
        let mut names = Vec::with_capacity(n);
        let mut domains = Vec::with_capacity(n);
        for _ in 0..n {
            domains.push(reader.byte()?);
            let length = usize::from(reader.u16()?);
            if length == 0 || length > MAX_NAME_BYTES {
                return Err("graph imputation variable name length is invalid");
            }
            names.push(
                std::str::from_utf8(reader.take(length)?)
                    .map_err(|_| "graph imputation variable name is not UTF-8")?
                    .to_owned(),
            );
        }
        let column_bytes = rows.checked_mul(8).ok_or("graph imputation column size overflow")?;
        let mut columns = Vec::with_capacity(n);
        for _ in 0..n {
            let raw = reader.take(column_bytes)?;
            columns.push(
                raw.chunks_exact(8)
                    .map(|chunk| {
                        let mut word = [0u8; 8];
                        word.copy_from_slice(chunk);
                        f64::from_bits(u64::from_le_bytes(word))
                    })
                    .collect(),
            );
        }
        if reader.offset != bytes.len() {
            return Err("trailing graph imputation source bytes");
        }
        let source = Self::new(
            GraphImputationParts {
                names,
                domains,
                columns,
                treatment,
                outcome,
                support,
                imputations,
                iterations,
                imputation_seed,
                posterior_draws,
                prior_scale,
                execution_seed,
                data_digest,
                atoms,
                excluded,
            },
            program,
        )?;
        if source.encode() != bytes {
            return Err("noncanonical graph imputation source");
        }
        Ok(source)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], &'static str> {
        let end = self.offset.checked_add(len).ok_or("graph imputation length overflow")?;
        let value = self.bytes.get(self.offset..end).ok_or("truncated graph imputation source")?;
        self.offset = end;
        Ok(value)
    }
    fn byte(&mut self) -> Result<u8, &'static str> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, &'static str> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().map_err(|_| "invalid u16")?))
    }
    fn u32(&mut self) -> Result<u32, &'static str> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| "invalid u32")?))
    }
    fn u64(&mut self) -> Result<u64, &'static str> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| "invalid u64")?))
    }
    fn digest(&mut self) -> Result<[u8; 32], &'static str> {
        self.take(32)?.try_into().map_err(|_| "invalid digest")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts() -> GraphImputationParts {
        GraphImputationParts {
            names: vec!["t".into(), "y".into(), "x".into()],
            domains: vec![1, 0, 0],
            columns: vec![
                vec![0.0, 1.0, 0.0, 1.0],
                vec![0.1, f64::NAN, 0.3, 1.2],
                vec![0.5, -0.5, 1.5, 0.0],
            ],
            treatment: 0,
            outcome: 1,
            support: GraphSupport::TreatmentDirectEffect,
            imputations: 2,
            iterations: 2,
            imputation_seed: 7,
            posterior_draws: 100,
            prior_scale: 1.0,
            execution_seed: 9,
            data_digest: [3; 32],
            atoms: vec![
                AtomOperation {
                    mask: 1,
                    status: IdentificationStatus::NonparametricallyIdentified,
                    adjustment: vec![],
                },
                AtomOperation {
                    mask: 5,
                    status: IdentificationStatus::NonparametricallyIdentified,
                    adjustment: vec![2],
                },
            ],
            excluded: 3,
        }
    }

    #[test]
    fn source_round_trips_and_binds_every_retained_operation() {
        let original = GraphImputationSource::new(parts(), [5; 32]).unwrap();
        let bytes = original.encode();
        let replayed = GraphImputationSource::decode(&bytes).unwrap();
        assert_eq!(replayed.parts().contract_digest(), original.parts().contract_digest());
        assert_eq!(replayed.program(), [5; 32]);
        let mut changed = parts();
        changed.atoms[1].adjustment.clear();
        assert_ne!(changed.contract_digest(), parts().contract_digest());
        let mut other_data = parts();
        other_data.columns[2][0] = 0.6;
        assert_ne!(other_data.contract_digest(), parts().contract_digest());
        let mut other_exclusion = parts();
        other_exclusion.excluded = 4;
        assert_ne!(other_exclusion.contract_digest(), parts().contract_digest());
        assert_eq!(replayed.parts().excluded, 3);
    }

    #[test]
    fn truncated_trailing_or_edited_bytes_are_refused() {
        let bytes = GraphImputationSource::new(parts(), [5; 32]).unwrap().encode();
        assert!(GraphImputationSource::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(GraphImputationSource::decode(&trailing).is_err());
        let mut edited = bytes;
        edited[MAGIC.len() + 1 + 4 + 4 + 4] = 9;
        assert!(GraphImputationSource::decode(&edited).is_err());
    }

    #[test]
    fn invalid_declarations_are_refused() {
        let mut bad = parts();
        bad.domains[0] = 0;
        assert!(bad.validate().is_err());
        let mut bad = parts();
        bad.atoms[1].mask = 1;
        assert!(bad.validate().is_err());
        let mut bad = parts();
        bad.atoms[0].adjustment = vec![0];
        assert!(bad.validate().is_err());
        let mut bad = parts();
        bad.columns[1][0] = f64::INFINITY;
        assert!(bad.validate().is_err());
    }
}
