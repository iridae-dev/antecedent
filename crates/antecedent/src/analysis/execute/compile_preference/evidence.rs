//! Whether separately estimated outcomes rest on shared observations.
//!
//! Studies whose outcomes are joined into one law are judged by the evidence ledger, the same
//! overlap judgment choice inference and the composition bundle use: each study is a source
//! whose observations are its rows, identified by the content identity of the table the rows
//! came from and the row's index in that table ([`ObservationId::from_key`]). A study's table is
//! its own source unless it was taken from a larger one: a column projection declares its
//! [`RowProvenance`] (the source table's content identity and, when the rows were selected,
//! their original indices), so two studies that read different columns of the same rows share
//! those rows. The content identity binds the table's columns and rows, not its schema roles or
//! the analysis compiled over it, so two studies of one table share every row whatever outcome
//! each targets; studies of different tables share none.
//!
//! Studies of different tables may still observe the same subjects. A study that declares a
//! [`UnitKey`] names its rows by unit identifiers in a declared identifier space, and two keyed
//! studies in one space are judged by the units they share (each unit counted once, masked
//! rows excluded). Keys in different spaces, or a key on one study only, leave the overlap of
//! different tables unknown, and that is refused, never assumed away.
//!
//! Studies that share observations are not independent evidence. Under distinct evidence they
//! are refused (`evidence_double_counted`). Under shared evidence the caller declares, for
//! every pair that overlaps, the covariance between the two studies' estimates; the
//! propagation chain receives it as the cross-study block of its declared covariance. The
//! within-study blocks are each study's own posterior covariance, known only after fitting,
//! so the chain checks the whole matrix when it draws.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::ExternalRefusal;
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, Overlap, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_core::reason_details::{decision_evidence, preference_compile};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest, VariableId};
use antecedent_data::{OwnedColumn, TableView, TabularData};

use super::outcomes::OutcomeDependence;
use super::{BoundStudy, CANCELLED_STAGE, PreferenceRouteError, STAGE_COMPILE};
use crate::CausalError;
use crate::analysis::builder::DataInput;

/// Domain of the observation identities a study's rows carry in the ledger.
const ROW_DOMAIN: &str = "antecedent.preference_route.study_rows";
/// Domain of the observation identities a study's unit identifiers carry in the ledger.
const UNIT_DOMAIN: &str = "antecedent.preference_route.study_units";
/// Tag of the digest naming a set of unit identifiers in a chain factor.
const UNIT_FACTOR_TAG: &str = "antecedent.preference_route.unit_factor";
/// Tag of the digest naming a set of rows of one source table in a chain factor.
const ROW_FACTOR_TAG: &str = "antecedent.preference_route.row_factor";
/// Longest namespace or unit label, in bytes.
const MAX_UNIT_TEXT_BYTES: usize = 4096;

/// The declared covariance between the estimates of two studies that share observations.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossStudyCovariance {
    /// Index of the first study, in declaration order.
    pub first: usize,
    /// Index of the second study, after the first.
    pub second: usize,
    /// Row-major covariance between every coordinate of the first study's response (rows,
    /// in its grid order) and every coordinate of the second's (columns). Every term is
    /// finite and nonzero: the studies share their rows, and the chain reads a shared factor
    /// with a zero term as an undeclared overlap.
    pub covariance: Vec<f64>,
}

/// Separately estimated outcomes whose studies share observations, joined into one law under
/// a declared cross-study covariance.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedEvidence {
    /// Aligned rows of the joined law.
    pub draws: usize,
    /// Seed of the family table.
    pub seed: u64,
    /// The covariance of every pair of studies that share observations.
    pub cross: Vec<CrossStudyCovariance>,
}

// ---------------------------------------------------------------------------------------------
// Declarations on a study
// ---------------------------------------------------------------------------------------------

/// Where the rows of a study's table came from: the table it was taken from and, when rows
/// were selected, each row's index in that table.
///
/// A study built on a column projection of a larger table declares the larger table here, so
/// its rows are the same observations as the rows of any other projection of that table. The
/// identity is compared as declared text; [`Self::of_table`] uses the stored content identity
/// of a table, the same identity a study of that table carries itself.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowProvenance {
    table: Arc<str>,
    rows: Option<Arc<[u64]>>,
}

impl RowProvenance {
    /// Rows taken, in order and all of them, from `source`: a column projection of it.
    #[must_use]
    pub fn of_table(source: &TabularData) -> Self {
        Self {
            table: Arc::from(
                SemanticDigest::from_bytes(source.storage().content_digest()).to_hex(),
            ),
            rows: None,
        }
    }

    /// Rows taken from the table whose content identity is `table`: with `rows`, the index in
    /// that table of each row of the study's table, in order; without, all its rows in order.
    #[must_use]
    pub fn new(table: &str, rows: Option<Vec<u64>>) -> Self {
        Self { table: Arc::from(table), rows: rows.map(Arc::from) }
    }

    /// Content identity of the table the rows came from.
    #[must_use]
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Index in that table of each row, when the rows were selected.
    #[must_use]
    pub fn rows(&self) -> Option<&[u64]> {
        self.rows.as_deref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum UnitKeySource {
    Column(VariableId),
    Labels(Arc<[String]>),
}

/// The unit identifiers of a study's rows in a declared identifier space.
///
/// Two studies of different tables are compared by the units they share only when both declare
/// a key in the same namespace (for example `hospital_mrn`); keys in different namespaces are
/// never compared. A key is a column of the study's table (an integer, real or boolean column
/// that no row leaves missing and that is neither the treatment nor the outcome), or one label
/// per row of the table. The label of a row is canonical, type-tagged text: `text:<value>`,
/// `int64:<decimal>`, `float64:<16 lowercase hex digits of the bits>` or `bool:<0|1>`, so the
/// text `"1"` and the integer `1` are different units; a key column yields the same labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnitKey {
    namespace: Arc<str>,
    source: UnitKeySource,
}

impl UnitKey {
    /// The identifiers in column `variable` of the study's table, in identifier space
    /// `namespace`.
    #[must_use]
    pub fn column(variable: VariableId, namespace: &str) -> Self {
        Self { namespace: Arc::from(namespace), source: UnitKeySource::Column(variable) }
    }

    /// One canonical label per row of the study's table, in identifier space `namespace`.
    #[must_use]
    pub fn labels(namespace: &str, labels: Vec<String>) -> Self {
        Self { namespace: Arc::from(namespace), source: UnitKeySource::Labels(Arc::from(labels)) }
    }

    /// The identifier space.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// The key column, when the key is a column of the study's table.
    #[must_use]
    pub const fn variable(&self) -> Option<VariableId> {
        match &self.source {
            UnitKeySource::Column(variable) => Some(*variable),
            UnitKeySource::Labels(_) => None,
        }
    }
}

/// A study's declared row provenance and unit key, checked against its table at build.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EvidenceRows {
    pub(crate) provenance: Option<RowProvenance>,
    pub(crate) units: Option<ResolvedUnits>,
}

/// A unit key resolved to one canonical label per row of the study's table.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedUnits {
    pub(crate) namespace: Arc<str>,
    pub(crate) variable: Option<VariableId>,
    pub(crate) labels: Arc<[String]>,
}

fn declaration_error(detail: String) -> CausalError {
    CausalError::SchemaMismatch { detail }
}

/// Canonical labels of a key column, one per row.
fn column_labels(data: &TabularData, variable: VariableId) -> Result<Vec<String>, String> {
    let column = data
        .storage()
        .columns()
        .get(variable.as_usize())
        .ok_or_else(|| format!("unit key column {variable:?} is not a column of the table"))?;
    let missing = || format!("unit key column {variable:?} leaves a row without an identifier");
    match column {
        OwnedColumn::Int64(column) => (0..column.values.len())
            .map(|row| {
                column
                    .validity
                    .is_valid(row)
                    .then(|| format!("int64:{}", column.values[row]))
                    .ok_or_else(missing)
            })
            .collect(),
        OwnedColumn::Float64(column) => {
            let values = column.values.as_slice();
            (0..values.len())
                .map(|row| {
                    let value = values[row];
                    (column.validity.is_valid(row) && value.is_finite())
                        .then(|| format!("float64:{:016x}", value.to_bits()))
                        .ok_or_else(missing)
                })
                .collect()
        }
        OwnedColumn::Boolean(column) => (0..column.values.len())
            .map(|row| {
                column
                    .validity
                    .is_valid(row)
                    .then(|| format!("bool:{}", u8::from(column.values[row] != 0)))
                    .ok_or_else(missing)
            })
            .collect(),
        _ => Err(format!(
            "unit key column {variable:?} is not an integer, real or boolean column; declare \
             text identifiers as one label per row"
        )),
    }
}

impl EvidenceRows {
    /// Check the declared provenance and unit key against the study's table and resolve the
    /// key's labels. `None` when neither is declared.
    ///
    /// # Errors
    /// A declaration on data that is not one table; provenance with a blank table identity or
    /// a row index per row that does not match the table; a blank namespace; a key column the
    /// table does not have, of an unsupported type, or with a missing identifier; a label list
    /// whose length is not the table's row count, or a blank or overlong label.
    pub(crate) fn declare(
        unit_key: Option<UnitKey>,
        provenance: Option<RowProvenance>,
        data: &DataInput,
    ) -> Result<Option<Self>, CausalError> {
        if unit_key.is_none() && provenance.is_none() {
            return Ok(None);
        }
        let DataInput::Tabular(table) = data else {
            return Err(CausalError::Unsupported {
                message: "a unit key or row provenance describes the rows of one table",
            });
        };
        let rows = table.row_count();
        if let Some(provenance) = &provenance {
            if provenance.table.trim().is_empty() || provenance.table.len() > MAX_UNIT_TEXT_BYTES {
                return Err(declaration_error(
                    "row provenance names its source table by a non-blank content identity"
                        .to_owned(),
                ));
            }
            if provenance.rows.as_ref().is_some_and(|indices| indices.len() != rows) {
                return Err(declaration_error(format!(
                    "row provenance gives a source index per row; the table has {rows} rows"
                )));
            }
        }
        let units = match unit_key {
            None => None,
            Some(key) => {
                if key.namespace.trim().is_empty() || key.namespace.len() > MAX_UNIT_TEXT_BYTES {
                    return Err(declaration_error(
                        "a unit key names its identifier space by a non-blank namespace".to_owned(),
                    ));
                }
                let (variable, labels) = match &key.source {
                    UnitKeySource::Column(variable) => (
                        Some(*variable),
                        Arc::from(column_labels(table, *variable).map_err(declaration_error)?),
                    ),
                    UnitKeySource::Labels(labels) => {
                        if labels.len() != rows {
                            return Err(declaration_error(format!(
                                "a unit key gives one label per row; {} labels for {rows} rows",
                                labels.len()
                            )));
                        }
                        if labels.iter().any(|label| {
                            label.trim().is_empty() || label.len() > MAX_UNIT_TEXT_BYTES
                        }) {
                            return Err(declaration_error(
                                "every unit label is non-blank text".to_owned(),
                            ));
                        }
                        (None, Arc::clone(labels))
                    }
                };
                Some(ResolvedUnits { namespace: key.namespace, variable, labels })
            }
        };
        Ok(Some(Self { provenance, units }))
    }
}

/// The units a study's analysis reads: the distinct labels of its unmasked rows.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct StudyUnits {
    pub(super) namespace: Arc<str>,
    pub(super) labels: BTreeSet<String>,
}

/// The distinct unit labels of the rows `data` keeps for analysis.
pub(super) fn study_units(units: &ResolvedUnits, data: &TabularData) -> StudyUnits {
    let mask = data.storage().analysis_mask();
    let labels = units
        .labels
        .iter()
        .enumerate()
        .filter(|(row, _)| mask.is_none_or(|mask| mask.is_valid(*row)))
        .map(|(_, label)| label.clone())
        .collect();
    StudyUnits { namespace: Arc::clone(&units.namespace), labels }
}

// ---------------------------------------------------------------------------------------------
// Judgement
// ---------------------------------------------------------------------------------------------

fn invalid(offending: String, supplied: impl Into<String>) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        STAGE_COMPILE,
        preference_compile::CROSS_STUDY_COVARIANCE_INVALID,
    )
    .with_offending(offending)
    .with_supplied(supplied)
    .with_remedy(
        "declare one finite, nonzero covariance block per pair of studies that share \
         observations, rows of the first study's coordinates by columns of the second's",
    )
    .into()
}

fn pair(studies: &[BoundStudy], first: usize, second: usize) -> String {
    format!("{} and {}", name(studies, first), name(studies, second))
}

fn name(studies: &[BoundStudy], index: usize) -> String {
    studies.get(index).map_or_else(|| format!("study {index}"), |s| s.binding.outcome_id.clone())
}

/// Check every declared block's indices, shape and terms.
fn check_blocks(
    studies: &[BoundStudy],
    cross: &[CrossStudyCovariance],
) -> Result<(), PreferenceRouteError> {
    for (index, block) in cross.iter().enumerate() {
        let (Some(first), Some(second)) = (studies.get(block.first), studies.get(block.second))
        else {
            return Err(invalid(
                format!("block {index}"),
                format!("studies {} and {} of {}", block.first, block.second, studies.len()),
            ));
        };
        if block.first >= block.second {
            return Err(invalid(
                pair(studies, block.first, block.second),
                "the first study is not before the second",
            ));
        }
        if cross[..index].iter().any(|b| (b.first, b.second) == (block.first, block.second)) {
            return Err(invalid(pair(studies, block.first, block.second), "a repeated block"));
        }
        let shape = first.coordinates.len() * second.coordinates.len();
        if block.covariance.len() != shape {
            return Err(invalid(
                pair(studies, block.first, block.second),
                format!("{} terms for a {shape}-term block", block.covariance.len()),
            ));
        }
        if !block.covariance.iter().all(|term| term.is_finite() && term.abs() > 0.0) {
            return Err(invalid(
                pair(studies, block.first, block.second),
                "a non-finite or zero term",
            ));
        }
    }
    Ok(())
}

fn ledger_refusal(
    error: antecedent_contract::evidence_ledger::LedgerError,
    offending: Option<String>,
) -> PreferenceRouteError {
    let mut refusal =
        ExternalRefusal::new(error.code, STAGE_COMPILE, error.detail).with_supplied(error.message);
    if let Some(offending) = offending {
        refusal = refusal.with_offending(offending);
    }
    refusal.into()
}

/// The ledger of the studies' rows, one observation set per study: each row is identified by
/// the table it came from and its index there.
fn ledger_of(
    studies: &[BoundStudy],
    ctx: &ExecutionContext,
) -> Result<(EvidenceLedger, Vec<SourceId>), PreferenceRouteError> {
    let mut ledger = EvidenceLedger::new();
    let mut sources = Vec::with_capacity(studies.len());
    for study in studies {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
        }
        let table = study.origin.as_bytes();
        let rows = (0..study.rows).map(|row| {
            let index = study
                .origin_rows
                .as_ref()
                .and_then(|indices| indices.get(row).copied())
                .unwrap_or(row as u64);
            let mut key = Vec::with_capacity(table.len() + 8);
            key.extend_from_slice(table);
            key.extend_from_slice(&index.to_le_bytes());
            ObservationId::from_key(ROW_DOMAIN, &key)
        });
        let declaration =
            SourceDeclaration::new(study.binding.outcome_id.clone(), SourceKind::ObservationSet)
                .with_observations(rows);
        let source = ledger
            .add_source(declaration)
            .map_err(|error| ledger_refusal(error, Some(study.binding.outcome_id.clone())))?;
        sources.push(source);
    }
    Ok((ledger, sources))
}

/// The observation identity of a unit label in `namespace`.
fn unit_observation(namespace: &str, label: &str) -> ObservationId {
    let mut key = Vec::with_capacity(8 + namespace.len() + label.len());
    key.extend_from_slice(&(namespace.len() as u64).to_le_bytes());
    key.extend_from_slice(namespace.as_bytes());
    key.extend_from_slice(label.as_bytes());
    ObservationId::from_key(UNIT_DOMAIN, &key)
}

/// The ledger of the keyed studies' units, one observation set per keyed study.
fn unit_ledger_of(
    studies: &[BoundStudy],
    ctx: &ExecutionContext,
) -> Result<(EvidenceLedger, Vec<Option<SourceId>>), PreferenceRouteError> {
    let mut ledger = EvidenceLedger::new();
    let mut sources = Vec::with_capacity(studies.len());
    for study in studies {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
        }
        let Some(units) = &study.units else {
            sources.push(None);
            continue;
        };
        let observations =
            units.labels.iter().map(|label| unit_observation(&units.namespace, label));
        let declaration =
            SourceDeclaration::new(study.binding.outcome_id.clone(), SourceKind::ObservationSet)
                .with_observations(observations);
        let source = ledger
            .add_source(declaration)
            .map_err(|error| ledger_refusal(error, Some(study.binding.outcome_id.clone())))?;
        sources.push(Some(source));
    }
    Ok((ledger, sources))
}

/// What a pair of studies shares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SharedBasis {
    /// Rows of the table both studies' rows came from.
    Rows,
    /// Units of one identifier space, both studies keyed in it.
    Units,
}

/// A pair of studies that shares observations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SharedPair {
    pub(super) first: usize,
    pub(super) second: usize,
    pub(super) basis: SharedBasis,
}

/// Every pair of studies the ledger found sharing observations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct StudyOverlaps {
    pub(super) pairs: Vec<SharedPair>,
}

/// How the overlap of two studies is judged and what was found.
enum Judged {
    Overlap(Overlap, SharedBasis),
    NotComparable(String),
}

/// A ledger of every study's rows and each study's source in it.
type RowLedger<'a> = (&'a EvidenceLedger, &'a [SourceId]);
/// A ledger of keyed studies' units and each keyed study's source in it.
type UnitLedger<'a> = (&'a EvidenceLedger, &'a [Option<SourceId>]);

fn judge_pair(
    studies: &[BoundStudy],
    (i, j): (usize, usize),
    (rows, units): (RowLedger<'_>, UnitLedger<'_>),
) -> Result<Judged, PreferenceRouteError> {
    let (a, b) = (&studies[i], &studies[j]);
    let by_rows = || -> Result<Judged, PreferenceRouteError> {
        let overlap = rows.0.overlap(rows.1[i], rows.1[j]).map_err(|e| ledger_refusal(e, None))?;
        Ok(Judged::Overlap(overlap, SharedBasis::Rows))
    };
    if a.origin == b.origin {
        return by_rows();
    }
    match (&a.units, &b.units) {
        (None, None) => by_rows(),
        (Some(left), Some(right)) if left.namespace == right.namespace => {
            let (Some(left), Some(right)) = (units.1[i], units.1[j]) else {
                return Ok(Judged::NotComparable("a unit key was not entered".to_owned()));
            };
            let overlap = units.0.overlap(left, right).map_err(|e| ledger_refusal(e, None))?;
            Ok(Judged::Overlap(overlap, SharedBasis::Units))
        }
        (Some(left), Some(right)) => Ok(Judged::NotComparable(format!(
            "unit keys of different identifier spaces ({} and {}) are not compared",
            left.namespace, right.namespace
        ))),
        (Some(_), None) => Ok(Judged::NotComparable(format!(
            "{} declares no unit key, so its overlap with a keyed study of another table is \
             unknown",
            name(studies, j)
        ))),
        (None, Some(_)) => Ok(Judged::NotComparable(format!(
            "{} declares no unit key, so its overlap with a keyed study of another table is \
             unknown",
            name(studies, i)
        ))),
    }
}

/// Judge whether the studies' outcomes may be joined under `dependence`: every pair the
/// ledger finds overlapping is refused under distinct evidence and needs a declared block
/// under shared evidence; a block for a pair that shares nothing is refused. Studies of one
/// source table are judged by its rows, whatever keys they declare; keyed studies of different
/// tables by the units of their identifier space; unkeyed studies of different tables share
/// nothing. Returns the pairs that share observations.
///
/// # Errors
/// `evidence_double_counted` (`preference_compile.studies_share_evidence`) for studies that
/// share observations without a declared block; `evidence_overlap_unknown`
/// (`decision_evidence.dependence_unknown`) for an overlap the ledger cannot compute or one
/// between a keyed study and a study of another table without a comparable key;
/// `invalid_argument` (`preference_compile.cross_study_covariance_invalid`) for a malformed
/// block or one naming studies that share nothing; a ledger refusal; cancellation.
pub(super) fn judge(
    studies: &[BoundStudy],
    dependence: OutcomeDependence,
    cross: &[CrossStudyCovariance],
    ctx: &ExecutionContext,
) -> Result<StudyOverlaps, PreferenceRouteError> {
    check_blocks(studies, cross)?;
    let (rows, row_sources) = ledger_of(studies, ctx)?;
    let (units, unit_sources) = unit_ledger_of(studies, ctx)?;
    let mut overlaps = StudyOverlaps::default();
    for i in 0..studies.len() {
        for j in (i + 1)..studies.len() {
            let declared = cross.iter().any(|block| (block.first, block.second) == (i, j));
            let judged =
                judge_pair(studies, (i, j), ((&rows, &row_sources), (&units, &unit_sources)))?;
            let (overlap, basis) = match judged {
                Judged::Overlap(overlap, basis) => (overlap, basis),
                Judged::NotComparable(reason) => {
                    return Err(ExternalRefusal::new(
                        antecedent_core::reason_code!("evidence_overlap_unknown"),
                        STAGE_COMPILE,
                        decision_evidence::DEPENDENCE_UNKNOWN,
                    )
                    .with_offending(pair(studies, i, j))
                    .with_supplied(reason)
                    .with_remedy(
                        "declare unit keys of one identifier space on both studies, or estimate \
                         the outcomes jointly in one design",
                    )
                    .into());
                }
            };
            match overlap {
                Overlap::Independent if declared => {
                    return Err(invalid(pair(studies, i, j), "studies that share no observations"));
                }
                Overlap::Independent => {}
                Overlap::SharedObservations { .. }
                    if declared
                        && matches!(dependence, OutcomeDependence::SharedEvidence { .. }) =>
                {
                    overlaps.pairs.push(SharedPair { first: i, second: j, basis });
                }
                Overlap::SharedObservations { shared } => {
                    let supplied = match (&basis, &studies[i].units) {
                        (SharedBasis::Units, Some(units)) => {
                            format!("{shared} shared units in namespace {}", units.namespace)
                        }
                        _ => format!("{shared} shared observations"),
                    };
                    return Err(ExternalRefusal::new(
                        antecedent_core::reason_code!("evidence_double_counted"),
                        STAGE_COMPILE,
                        preference_compile::STUDIES_SHARE_EVIDENCE,
                    )
                    .with_offending(pair(studies, i, j))
                    .with_expected("studies that share no observations, or a declared block")
                    .with_supplied(supplied)
                    .with_remedy(
                        "declare the studies' cross-study covariance as shared evidence, or \
                         estimate the outcomes jointly in one design",
                    )
                    .into());
                }
                other => {
                    let reason = match other {
                        Overlap::Unknown { reason } => reason,
                        _ => "the observation overlap is not recognised",
                    };
                    return Err(ExternalRefusal::new(
                        antecedent_core::reason_code!("evidence_overlap_unknown"),
                        STAGE_COMPILE,
                        decision_evidence::DEPENDENCE_UNKNOWN,
                    )
                    .with_offending(pair(studies, i, j))
                    .with_supplied(reason)
                    .into());
                }
            }
        }
    }
    Ok(overlaps)
}

// ---------------------------------------------------------------------------------------------
// Chain factors
// ---------------------------------------------------------------------------------------------

/// Digest naming a set of unit labels of one identifier space.
fn unit_factor(prefix: &str, namespace: &str, labels: impl Iterator<Item = String>) -> String {
    let labels: Vec<String> = labels.collect();
    let mut writer = CanonicalWriter::new(UNIT_FACTOR_TAG);
    writer.text(namespace).texts(&labels);
    format!("{prefix}:{}", writer.finish_hex())
}

/// Digest naming a set of row indices of the source table `origin`.
fn row_factor(prefix: &str, origin: &str, rows: &BTreeSet<u64>) -> String {
    let mut writer = CanonicalWriter::new(ROW_FACTOR_TAG);
    writer.text(origin).word(u64::try_from(rows.len()).unwrap_or(u64::MAX));
    for row in rows {
        writer.word(*row);
    }
    format!("{prefix}:{}", writer.finish_hex())
}

/// The indices in its source table of the rows a study reads.
fn source_rows(study: &BoundStudy) -> BTreeSet<u64> {
    match &study.origin_rows {
        Some(rows) => rows.iter().copied().collect(),
        None => (0..study.rows).map(|row| row as u64).collect(),
    }
}

/// The row factors of every study's chain sources.
///
/// A study without a unit key declares the content identity of the table its rows came from,
/// as it always has, unless some study of the same source table selected rows of it: then,
/// like a keyed study, it declares one factor per study it shares rows with, naming the shared
/// rows (`shared_rows:<digest>`), or `rows:<digest>` naming its own rows when it shares none,
/// so studies of disjoint rows of one table are not read as sharing it. A keyed study declares one factor per study it shares observations with:
/// the source table's identity for a pair of one table, and `shared_units:<digest>` naming the
/// shared units of its identifier space otherwise; a keyed study that shares nothing declares
/// `units:<digest>` naming its own units. The chain rule then sees exactly the pairwise
/// overlaps the ledger found, and a declared block is required exactly where two studies share
/// a factor.
pub(super) fn row_factors(studies: &[BoundStudy], overlaps: &StudyOverlaps) -> Vec<Vec<String>> {
    studies
        .iter()
        .enumerate()
        .map(|(index, study)| {
            let Some(units) = &study.units else {
                let selected = studies.iter().any(|other| {
                    other.units.is_none()
                        && other.origin == study.origin
                        && other.origin_rows.is_some()
                });
                if !selected {
                    return vec![study.origin.clone()];
                }
                let mine = source_rows(study);
                let mut factors = Vec::new();
                for shared in &overlaps.pairs {
                    let other = if shared.first == index {
                        shared.second
                    } else if shared.second == index {
                        shared.first
                    } else {
                        continue;
                    };
                    let factor = match studies.get(other) {
                        Some(theirs) if theirs.units.is_none() && theirs.origin == study.origin => {
                            let both = mine.intersection(&source_rows(theirs)).copied().collect();
                            row_factor("shared_rows", &study.origin, &both)
                        }
                        _ => study.origin.clone(),
                    };
                    if !factors.contains(&factor) {
                        factors.push(factor);
                    }
                }
                if factors.is_empty() {
                    factors.push(row_factor("rows", &study.origin, &mine));
                }
                return factors;
            };
            let mut factors = Vec::new();
            for shared in &overlaps.pairs {
                let other = if shared.first == index {
                    shared.second
                } else if shared.second == index {
                    shared.first
                } else {
                    continue;
                };
                let factor =
                    match (&shared.basis, studies.get(other).and_then(|s| s.units.as_ref())) {
                        (SharedBasis::Units, Some(theirs)) => unit_factor(
                            "shared_units",
                            &units.namespace,
                            units.labels.intersection(&theirs.labels).cloned(),
                        ),
                        _ => study.origin.clone(),
                    };
                if !factors.contains(&factor) {
                    factors.push(factor);
                }
            }
            if factors.is_empty() {
                factors.push(unit_factor("units", &units.namespace, units.labels.iter().cloned()));
            }
            factors
        })
        .collect()
}
