//! Whether separately estimated outcomes rest on shared observations.
//!
//! Studies whose outcomes are joined into one law are judged by the evidence ledger, the same
//! overlap judgment choice inference and the composition bundle use: each study is a source
//! whose observations are the rows its analysis reads (masked rows excluded), identified by the
//! identity of the table the rows came from and the row's index in that table
//! ([`ObservationId::from_key`]). A study's table is its own source unless it was taken from a
//! larger one: a study of a column projection or a row selection declares its
//! [`RowProvenance`] (the source table's identity and, when the rows were selected, their
//! original indices), so two studies that read different columns of the same rows share those
//! rows. The provenance is carried beside the table, never in it: [`RowProvenance::of_table`]
//! names every row of a source table, a column projection keeps its source's provenance, and
//! [`RowProvenance::select_rows`] takes rows of a table together with their provenance. The
//! identity binds the table's columns and rows ([`rows_digest`]), not its analysis mask, its
//! weights, its schema roles or the analysis compiled over it, so two studies of one table share
//! every row both analyse whatever outcome, mask or weighting each reads.
//!
//! Studies of different tables may still observe the same subjects. A study that declares a
//! [`UnitKey`] names its rows by unit identifiers in a declared identifier space, and two keyed
//! studies in one space are judged by the units they share (each unit counted once, masked
//! rows excluded), whether or not their rows come from one table: they share evidence when
//! they share units or rows. Labels of one space must carry the same type tags on both
//! studies; keys in different spaces, or a key on one study only, leave the overlap of
//! different tables unknown. Each is refused, never assumed away.
//!
//! Studies of different tables without unit keys are not compared by the ledger: their rows
//! are other observations only on the caller's declaration that the studies rest on distinct
//! evidence (or, under shared evidence, by the absence of a declared block for the pair), and
//! only when both studies declare their row provenance. A study without declared provenance
//! names its own table, which a projection or row selection of another study's source table
//! also does, so the pair's overlap is unknown and it is refused
//! (`preference_compile.row_provenance_undeclared`, basis `undeclared`); a decision over one
//! study, or over studies of one table, needs no declaration. The plan states the basis of
//! every pair's judgement: `shared_rows`, `shared_units`, `disjoint_rows`, `disjoint_units` or
//! `declared_distinct`.
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
use antecedent_data::{DataError, OwnedColumn, TableView, TabularData};

use super::outcomes::OutcomeDependence;
use super::{BoundStudy, CANCELLED_STAGE, PreferenceRouteError, STAGE_COMPILE};
use crate::CausalError;
use crate::analysis::builder::DataInput;

/// Domain of the observation identities a study's rows carry in the ledger.
const ROW_DOMAIN: &str = "antecedent.preference_route.study_rows.v1";
/// Domain of the observation identities a study's unit identifiers carry in the ledger.
const UNIT_DOMAIN: &str = "antecedent.preference_route.study_units.v1";
/// Tag of the digest naming a set of unit identifiers in a chain factor.
const UNIT_FACTOR_TAG: &str = "antecedent.preference_route.unit_factor.v1";
/// Tag of the digest naming a set of rows of one source table in a chain factor.
const ROW_FACTOR_TAG: &str = "antecedent.preference_route.row_factor.v1";
/// Longest namespace or unit label, in bytes.
const MAX_UNIT_TEXT_BYTES: usize = 4096;

/// Basis of a pair the ledger found sharing rows of one source table.
pub(super) const PAIR_SHARED_ROWS: &str = "shared_rows";
/// Basis of a pair the ledger found sharing units of one identifier space.
pub(super) const PAIR_SHARED_UNITS: &str = "shared_units";
/// Basis of a pair of one source table whose analysed rows are disjoint.
pub(super) const PAIR_DISJOINT_ROWS: &str = "disjoint_rows";
/// Basis of a pair keyed in one identifier space whose units are disjoint.
pub(super) const PAIR_DISJOINT_UNITS: &str = "disjoint_units";
/// Basis of a pair of different tables without unit keys: distinct on the caller's declaration.
pub(super) const PAIR_DECLARED_DISTINCT: &str = "declared_distinct";
/// Basis of a pair of different tables without unit keys, at least one of which declares no
/// row provenance: whether they share rows is unknown, and the pair is refused.
pub(super) const PAIR_UNDECLARED: &str = "undeclared";

/// The declared covariance between the estimates of two studies that share observations.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossStudyCovariance {
    /// Index of the first study, in declaration order.
    pub first: usize,
    /// Index of the second study, after the first.
    pub second: usize,
    /// Row-major covariance between every coordinate of the first study's response (rows,
    /// in its grid order) and every coordinate of the second's (columns). Every term is
    /// finite. A term may be zero, declaring those two coordinates' estimates uncorrelated
    /// although the studies share observations; a block whose every term is zero is refused,
    /// since it declares the shared observations contribute nothing, which is what distinct
    /// evidence declares for studies that share none. The whole matrix is checked for
    /// positive semi-definiteness when it is bound to the studies' posterior covariances.
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

/// The identity of the observations a table's rows record: the digest of its typed columns and
/// their validity, without its analysis mask or weights. It equals the table's content digest
/// when the table carries neither, and the content digest of the same columns with neither
/// otherwise, so the same rows under another mask or weighting are the same observations.
///
/// # Errors
/// A failure to rebuild the columns without the mask and weights.
pub(crate) fn rows_digest(data: &TabularData) -> Result<[u8; 32], CausalError> {
    let storage = data.storage();
    if storage.analysis_mask().is_none() && storage.weights().is_none() {
        return Ok(storage.content_digest());
    }
    let rows = storage.derive(storage.schema().clone(), storage.columns().to_vec(), None, None)?;
    Ok(rows.content_digest())
}

/// Where the rows of a study's table came from: the table it was taken from and, when rows
/// were selected, each row's index in that table.
///
/// A study built on a column projection of a larger table declares the larger table here, so
/// its rows are the same observations as the rows of any other projection of that table. The
/// provenance is carried beside the table, never in it: a column projection keeps every row in
/// order, so it keeps its source's provenance, and [`Self::select_rows`] takes rows of a table
/// with the provenance of the rows taken. The identity is compared as declared text;
/// [`Self::of_table`] uses the identity of a table's rows (its columns and validity, without a
/// mask or weights), the same identity a study of that table carries itself.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowProvenance {
    table: Arc<str>,
    rows: Option<Arc<[u64]>>,
}

impl RowProvenance {
    /// Every row of `source`, in order: the provenance of `source` itself and of any column
    /// projection of it.
    ///
    /// # Errors
    /// A failure to digest the rows of `source`.
    pub fn of_table(source: &TabularData) -> Result<Self, CausalError> {
        Ok(Self {
            table: Arc::from(SemanticDigest::from_bytes(rows_digest(source)?).to_hex()),
            rows: None,
        })
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

    /// The rows `rows` of `table`, whose rows this provenance describes, in that order (a row
    /// may repeat), with every column, the analysis mask and the weights gathered through
    /// them, and the provenance of the rows taken: their indices in the source table, composed
    /// through this provenance's own indices when it has them. The declared unit key of
    /// `table` is dropped.
    ///
    /// # Errors
    /// An empty selection; a row index past the table; a provenance whose row indices do not
    /// give one per row of `table`; gathered weights that are not a valid weighting.
    pub fn select_rows(
        &self,
        table: &TabularData,
        rows: &[usize],
    ) -> Result<(TabularData, Self), CausalError> {
        if rows.is_empty() {
            return Err(
                DataError::EmptySelection { context: "row selection: no rows requested" }.into()
            );
        }
        let storage = table.storage();
        if rows.iter().any(|&row| row >= storage.row_count()) {
            return Err(DataError::InvalidValue {
                what: "row selection",
                requirement: "exceeds the table",
            }
            .into());
        }
        let indices = match &self.rows {
            None => rows
                .iter()
                .map(|&row| {
                    u64::try_from(row).map_err(|_| DataError::Overflow { what: "row index" })
                })
                .collect::<Result<Vec<u64>, DataError>>()?,
            Some(parent) => {
                if parent.len() != storage.row_count() {
                    return Err(declaration_error(format!(
                        "row provenance gives {} source indices for a table of {} rows",
                        parent.len(),
                        storage.row_count()
                    )));
                }
                rows.iter().map(|&row| parent[row]).collect()
            }
        };
        let columns = (0..storage.columns().len())
            .map(|index| {
                let id = VariableId::from_raw(
                    u32::try_from(index)
                        .map_err(|_| DataError::Overflow { what: "column index" })?,
                );
                storage.column(id)?.gather_rows(rows)
            })
            .collect::<Result<Vec<_>, DataError>>()?;
        let mask = storage.analysis_mask().map(|mask| mask.gather_rows(rows)).transpose()?;
        let weights = storage
            .weights()
            .map(|weights| rows.iter().map(|&row| weights[row]).collect::<Arc<[f64]>>());
        let derived = storage.derive(storage.schema().clone(), columns, mask, weights)?;
        let provenance = Self { table: Arc::clone(&self.table), rows: Some(Arc::from(indices)) };
        Ok((TabularData::new(derived), provenance))
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
/// per row of the table. Units are compared by canonical, type-tagged labels: `text:<value>`,
/// `int64:<decimal>`, `float64:<16 lowercase hex digits of the bits>` or `bool:<0|1>`, so the
/// text `"1"` and the integer `1` are different units; a key column yields such labels. The type
/// of supplied labels is stated by the constructor: [`UnitKey::text_labels`] reads each identifier
/// as text whatever it contains (`"2023:017"` is the text `2023:017`), and
/// [`UnitKey::canonical_labels`] takes labels already in the canonical form, each checked at
/// build.
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

    /// One text identifier per row of the study's table, in identifier space `namespace`. Each
    /// identifier is text whatever characters it holds, a `:` included; it becomes the canonical
    /// label `text:<identifier>`.
    #[must_use]
    pub fn text_labels(namespace: &str, identifiers: Vec<String>) -> Self {
        let labels: Vec<String> = identifiers.into_iter().map(|id| format!("text:{id}")).collect();
        Self { namespace: Arc::from(namespace), source: UnitKeySource::Labels(Arc::from(labels)) }
    }

    /// One canonical, type-tagged label per row of the study's table (as the type documentation
    /// spells them), in identifier space `namespace`. A label not in that form is refused at
    /// build: an untagged identifier is never read as a tag.
    #[must_use]
    pub fn canonical_labels(namespace: &str, labels: Vec<String>) -> Self {
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

/// Whether `label` is in the canonical, type-tagged form: `text:` and a non-blank identifier,
/// `int64:` and the decimal a 64-bit integer prints as, `float64:` and 16 lowercase hex digits of
/// a finite real's bits, or `bool:0` / `bool:1`.
fn canonical_label(label: &str) -> bool {
    let Some((tag, value)) = label.split_once(':') else {
        return false;
    };
    match tag {
        "text" => !value.trim().is_empty(),
        "int64" => value.parse::<i64>().is_ok_and(|number| number.to_string() == value),
        "float64" => {
            value.len() == 16
                && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                && u64::from_str_radix(value, 16).is_ok_and(|bits| f64::from_bits(bits).is_finite())
        }
        "bool" => value == "0" || value == "1",
        _ => false,
    }
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
                        if let Some(label) = labels.iter().find(|label| !canonical_label(label)) {
                            return Err(declaration_error(format!(
                                "unit label {label:?} is not canonical, type-tagged text \
                                 (text:<identifier>, int64:<decimal>, float64:<16 lowercase hex \
                                 digits> or bool:<0|1>); declare raw text identifiers with \
                                 UnitKey::text_labels"
                            )));
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
    /// The type tags of the labels (the text before the first `:`, empty for a label without
    /// one), so labels of one space stored under different types are not read as other units.
    pub(super) tags: BTreeSet<String>,
}

/// The type tag of a canonical unit label: the text before its first `:`, or empty.
fn label_tag(label: &str) -> &str {
    label.split_once(':').map_or("", |(tag, _)| tag)
}

/// The distinct unit labels of the rows `data` keeps for analysis.
pub(super) fn study_units(units: &ResolvedUnits, data: &TabularData) -> StudyUnits {
    let mask = data.storage().analysis_mask();
    let labels: BTreeSet<String> = units
        .labels
        .iter()
        .enumerate()
        .filter(|(row, _)| mask.is_none_or(|mask| mask.is_valid(*row)))
        .map(|(_, label)| label.clone())
        .collect();
    let tags = labels.iter().map(|label| label_tag(label).to_owned()).collect();
    StudyUnits { namespace: Arc::clone(&units.namespace), labels, tags }
}

/// The rows of `data` its analysis reads, when an analysis mask leaves some out.
pub(super) fn analysed_rows(data: &TabularData) -> Option<Arc<[usize]>> {
    let mask = data.storage().analysis_mask()?;
    Some((0..data.row_count()).filter(|row| mask.is_valid(*row)).collect())
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
        "declare one finite covariance block, not all zero, per pair of studies that share \
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
        if !block.covariance.iter().all(|term| term.is_finite()) {
            return Err(invalid(pair(studies, block.first, block.second), "a non-finite term"));
        }
        if !block.covariance.iter().any(|term| term.abs() > 0.0) {
            return Err(invalid(
                pair(studies, block.first, block.second),
                "a block whose every term is zero",
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

/// The index in its source table of every row a study's analysis reads, in row order: masked
/// rows are not observations the study rests on.
fn analysed_source_rows(study: &BoundStudy) -> Vec<u64> {
    let index = |row: usize| {
        study
            .origin_rows
            .as_ref()
            .and_then(|indices| indices.get(row).copied())
            .unwrap_or(row as u64)
    };
    match &study.analysed {
        Some(rows) => rows.iter().map(|&row| index(row)).collect(),
        None => (0..study.rows).map(index).collect(),
    }
}

/// The ledger of the studies' analysed rows, one observation set per study: each row is
/// identified by the table it came from and its index there.
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
        let rows = analysed_source_rows(study).into_iter().map(|index| {
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

/// Every pair of studies the ledger found sharing observations, and the basis of the
/// judgement of every pair.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct StudyOverlaps {
    pub(super) pairs: Vec<SharedPair>,
    /// `(first, second, basis)` of every pair of studies, in pair order: one of
    /// [`PAIR_SHARED_ROWS`], [`PAIR_SHARED_UNITS`], [`PAIR_DISJOINT_ROWS`],
    /// [`PAIR_DISJOINT_UNITS`] or [`PAIR_DECLARED_DISTINCT`]; a pair whose basis would be
    /// [`PAIR_UNDECLARED`] is refused.
    pub(super) bases: Vec<(usize, usize, &'static str)>,
}

/// How the overlap of two studies is judged and what was found: the overlap, what a shared
/// overlap shares, and the basis stated when nothing is shared.
enum Judged {
    Overlap(Overlap, SharedBasis, &'static str),
    NotComparable(String),
    LabelTypesDiffer(String),
    /// Studies of different tables, without unit keys, at least one without declared row
    /// provenance: the named studies may be projections of one source table.
    Undeclared(String),
}

/// A ledger of every study's rows and each study's source in it.
type RowLedger<'a> = (&'a EvidenceLedger, &'a [SourceId]);
/// A ledger of keyed studies' units and each keyed study's source in it.
type UnitLedger<'a> = (&'a EvidenceLedger, &'a [Option<SourceId>]);

fn tags_of(units: &StudyUnits) -> String {
    units.tags.iter().map(String::as_str).collect::<Vec<_>>().join(", ")
}

fn judge_pair(
    studies: &[BoundStudy],
    (i, j): (usize, usize),
    (rows, units): (RowLedger<'_>, UnitLedger<'_>),
) -> Result<Judged, PreferenceRouteError> {
    let (a, b) = (&studies[i], &studies[j]);
    let by_rows = |independent: &'static str| -> Result<Judged, PreferenceRouteError> {
        let overlap = rows.0.overlap(rows.1[i], rows.1[j]).map_err(|e| ledger_refusal(e, None))?;
        Ok(Judged::Overlap(overlap, SharedBasis::Rows, independent))
    };
    if let (Some(left), Some(right)) = (&a.units, &b.units)
        && left.namespace == right.namespace
    {
        if left.tags != right.tags {
            return Ok(Judged::LabelTypesDiffer(format!(
                "labels of namespace {} typed {} and {}",
                left.namespace,
                tags_of(left),
                tags_of(right)
            )));
        }
        let (Some(left), Some(right)) = (units.1[i], units.1[j]) else {
            return Ok(Judged::NotComparable("a unit key was not entered".to_owned()));
        };
        let overlap = units.0.overlap(left, right).map_err(|e| ledger_refusal(e, None))?;
        // Studies of one source table that share no unit may still share rows.
        if a.origin == b.origin && matches!(overlap, Overlap::Independent) {
            return by_rows(PAIR_DISJOINT_ROWS);
        }
        return Ok(Judged::Overlap(overlap, SharedBasis::Units, PAIR_DISJOINT_UNITS));
    }
    if a.origin == b.origin {
        return by_rows(PAIR_DISJOINT_ROWS);
    }
    match (&a.units, &b.units) {
        (None, None) if !(a.provenance_declared && b.provenance_declared) => {
            let undeclared: Vec<String> = [(i, a), (j, b)]
                .into_iter()
                .filter(|(_, study)| !study.provenance_declared)
                .map(|(index, _)| name(studies, index))
                .collect();
            Ok(Judged::Undeclared(undeclared.join(" and ")))
        }
        (None, None) => by_rows(PAIR_DECLARED_DISTINCT),
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
/// under shared evidence; a block for a pair that shares nothing is refused. Studies keyed in
/// one identifier space are judged by their units and, when their rows come from one source
/// table, by its rows as well (they share evidence when either overlaps); other studies of one
/// source table by its analysed rows; unkeyed studies of different tables are distinct on the
/// caller's declaration only when both declare their row provenance, and refused otherwise.
/// Returns the pairs that share observations and the basis of every
/// pair's judgement.
///
/// # Errors
/// `evidence_double_counted` (`preference_compile.studies_share_evidence`) for studies that
/// share observations without a declared block; `evidence_overlap_unknown`
/// (`decision_evidence.dependence_unknown`) for an overlap the ledger cannot compute or one
/// between a keyed study and a study of another table without a comparable key, and
/// (`preference_compile.unit_label_types_differ`) for studies keyed in one space whose labels
/// carry different type tags, and (`preference_compile.row_provenance_undeclared`, basis
/// `undeclared`) for unkeyed studies of different tables when either declares no row
/// provenance; `invalid_argument`
/// (`preference_compile.cross_study_covariance_invalid`) for a malformed block or one naming
/// studies that share nothing or whose every term is zero; a ledger refusal; cancellation.
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
            let (overlap, basis, independent) = match judged {
                Judged::Overlap(overlap, basis, independent) => (overlap, basis, independent),
                Judged::LabelTypesDiffer(reason) => {
                    return Err(ExternalRefusal::new(
                        antecedent_core::reason_code!("evidence_overlap_unknown"),
                        STAGE_COMPILE,
                        preference_compile::UNIT_LABEL_TYPES_DIFFER,
                    )
                    .with_offending(pair(studies, i, j))
                    .with_supplied(reason)
                    .with_remedy(
                        "store the unit identifiers under one type on both studies (an integer \
                         identifier read as a real after missing values, for example), or \
                         declare labels of one type",
                    )
                    .into());
                }
                Judged::Undeclared(undeclared) => {
                    return Err(ExternalRefusal::new(
                        antecedent_core::reason_code!("evidence_overlap_unknown"),
                        STAGE_COMPILE,
                        preference_compile::ROW_PROVENANCE_UNDECLARED,
                    )
                    .with_offending(pair(studies, i, j))
                    .with_expected(
                        "a declared row provenance on both studies of different tables, or unit \
                         keys of one identifier space",
                    )
                    .with_supplied(format!(
                        "basis {PAIR_UNDECLARED}: {undeclared} declares no row provenance"
                    ))
                    .with_remedy(
                        "declare each study's row provenance (RowProvenance::of_table on the \
                         table it was taken from, or RowProvenance::select_rows for selected \
                         rows), or declare unit keys of one identifier space",
                    )
                    .into());
                }
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
                Overlap::Independent => overlaps.bases.push((i, j, independent)),
                Overlap::SharedObservations { .. }
                    if declared
                        && matches!(dependence, OutcomeDependence::SharedEvidence { .. }) =>
                {
                    let shared = match &basis {
                        SharedBasis::Rows => PAIR_SHARED_ROWS,
                        SharedBasis::Units => PAIR_SHARED_UNITS,
                    };
                    overlaps.bases.push((i, j, shared));
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

/// The indices in its source table of the rows a study's analysis reads.
fn source_rows(study: &BoundStudy) -> BTreeSet<u64> {
    analysed_source_rows(study).into_iter().collect()
}

/// The row factors of every study's chain sources.
///
/// A study declares one factor per study it shares observations with. A pair that shares
/// units of one identifier space is named by those units (`shared_units:<digest>`). A pair
/// that shares rows of one source table is named by that table's identity, or, when some study
/// of the table reads only some of its rows (a row selection or an analysis mask), by the rows
/// both read (`shared_rows:<digest>`), so studies of disjoint rows of one table are not read as
/// sharing it. A study that shares nothing declares its own units (`units:<digest>`) when it is
/// keyed, else the rows it reads of a partly read table (`rows:<digest>`), else its table's
/// identity. Both studies of a pair compute the same factor, so the chain rule sees exactly
/// the pairwise overlaps the ledger found, and a declared block is required exactly where two
/// studies share a factor.
pub(super) fn row_factors(studies: &[BoundStudy], overlaps: &StudyOverlaps) -> Vec<Vec<String>> {
    let partly_read = |origin: &str| {
        studies.iter().any(|study| {
            study.origin == origin && (study.origin_rows.is_some() || study.analysed.is_some())
        })
    };
    studies
        .iter()
        .enumerate()
        .map(|(index, study)| {
            let mut factors = Vec::new();
            for shared in &overlaps.pairs {
                let other = if shared.first == index {
                    shared.second
                } else if shared.second == index {
                    shared.first
                } else {
                    continue;
                };
                let Some(theirs) = studies.get(other) else {
                    continue;
                };
                let factor = match (&shared.basis, &study.units, &theirs.units) {
                    (SharedBasis::Units, Some(mine), Some(their_units)) => unit_factor(
                        "shared_units",
                        &mine.namespace,
                        mine.labels.intersection(&their_units.labels).cloned(),
                    ),
                    _ if partly_read(&study.origin) => {
                        let both = source_rows(study)
                            .intersection(&source_rows(theirs))
                            .copied()
                            .collect();
                        row_factor("shared_rows", &study.origin, &both)
                    }
                    _ => study.origin.clone(),
                };
                if !factors.contains(&factor) {
                    factors.push(factor);
                }
            }
            if factors.is_empty() {
                factors.push(match &study.units {
                    Some(units) => {
                        unit_factor("units", &units.namespace, units.labels.iter().cloned())
                    }
                    None if partly_read(&study.origin) => {
                        row_factor("rows", &study.origin, &source_rows(study))
                    }
                    None => study.origin.clone(),
                });
            }
            factors
        })
        .collect()
}
