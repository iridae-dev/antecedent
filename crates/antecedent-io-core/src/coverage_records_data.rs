//! Coverage records loaded from `data/coverage_records.json`.
//!
//! `parity/coverage_records.toml` is the registry. `scripts/generate_support_matrix_docs.py`
//! writes the snapshot, including attesting ids and identity aliases. Rows are parsed
//! on first use and leaked so the structs stay [`Copy`] with `'static` fields.
//!
//! Every row was emitted by a coverage test through `CoverageTally::for_record`
//! and collected by `scripts/collect_coverage_records.py`.

#![allow(missing_docs)]

use std::ops::Deref;
use std::sync::OnceLock;

use antecedent_core::IntervalMethod;
use serde::Deserialize;

const SNAPSHOT: &str = include_str!("../data/coverage_records.json");

/// Copy handle for a leaked slice. The static itself is const; the slice is built
/// on first use.
///
/// `iter` and `IntoIterator` return `'static` items. A [`std::sync::LazyLock`]
/// deref would borrow the lock and could not satisfy `&'static` row lookups.
#[derive(Clone, Copy, Debug)]
pub struct StaticTable<T: 'static> {
    load: fn() -> &'static [T],
}

impl<T: 'static> StaticTable<T> {
    /// `'static` iterator. Prefer this over the [`Deref`] to `[T]`, whose items
    /// are bound to the borrow of the handle.
    #[must_use = "consume the iterator to visit the coverage records"]
    pub fn iter(&self) -> std::slice::Iter<'static, T> {
        (self.load)().iter()
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        (self.load)().len()
    }

    /// Whether the table has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        (self.load)().is_empty()
    }
}

impl<T: 'static> Deref for StaticTable<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        (self.load)()
    }
}

impl<T: 'static> IntoIterator for StaticTable<T> {
    type Item = &'static T;
    type IntoIter = std::slice::Iter<'static, T>;

    fn into_iter(self) -> Self::IntoIter {
        (self.load)().iter()
    }
}

impl<T: 'static> IntoIterator for &StaticTable<T> {
    type Item = &'static T;
    type IntoIter = std::slice::Iter<'static, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// One measured coverage record. Key fields first, then the measured scope,
/// then the measurement and its provenance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverageRecord {
    pub id: &'static str,
    pub query: &'static str,
    pub graph_class: &'static str,
    pub structure: &'static str,
    pub modality: &'static str,
    pub inference: &'static str,
    pub estimator: &'static str,
    pub interval_method: &'static str,
    pub se_kind: &'static str,
    pub dependence: &'static str,
    pub posterior: &'static str,
    pub functional: &'static str,
    pub identification: &'static str,
    pub nominal: f64,
    pub n_min: u64,
    pub n_max: u64,
    pub replicates_min: u32,
    pub posterior_draws_min: u32,
    pub unidentified_mass_max: f64,
    pub observed: f64,
    pub mcse: f64,
    pub replicates: u32,
    pub boundary: bool,
    /// The measurement at each sample-size grid point, smallest `n` first.
    /// `n_min..n_max` spans these points; the record is a boundary when any
    /// point is.
    pub grid: &'static [CoverageGridPoint],
    /// The same sample sizes re-measured with fresh seeds, smallest `n`
    /// first; empty when the record was never independently confirmed.
    pub confirmation: &'static [CoverageGridPoint],
    /// Contiguous sample-size span `(n_lo, n_hi)` of a boundary record whose
    /// every grid point passed in the measurement and again in the
    /// confirmation. Rows in it are not governed by the failing points outside it.
    pub confirmed_passing_range: Option<(u64, u64)>,
    pub dgp: &'static str,
    pub test: &'static str,
    pub calibration_sha: &'static str,
}

/// One sample-size grid point of a coverage record: the rows it measured and
/// the coverage it measured there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverageGridPoint {
    pub point: u8,
    pub n_min: u64,
    pub n_max: u64,
    pub observed: f64,
    pub mcse: f64,
    pub replicates: u32,
    pub boundary: bool,
}

const fn table<T: 'static>(load: fn() -> &'static [T]) -> StaticTable<T> {
    StaticTable { load }
}

pub static RECORDS: StaticTable<CoverageRecord> = table(records);

/// Exact identity-only migrations with reviewed original-engine replay evidence.
/// All other construction axes, measured bounds and attestation remain mandatory.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FunctionalIdentityAlias {
    pub(crate) record_id: &'static str,
    pub(crate) measurement_sha: &'static str,
    pub(crate) original_functional: &'static str,
    pub(crate) canonical_functional: &'static str,
}

pub(crate) static FUNCTIONAL_IDENTITY_ALIASES: StaticTable<FunctionalIdentityAlias> =
    table(functional_identity_aliases);

/// Ids of the records in [`RECORDS`] that attest the tree this snapshot was
/// generated from (`scripts/calibration_facets.py`: no facet drifted since the
/// record's `calibration_sha`, or a valid replay waiver covers it). Regenerated
/// with the registry; a `calibrated` slot requires its governing record to be
/// listed.
pub static ATTESTING_RECORD_IDS: StaticTable<&str> = table(attesting_record_ids);

pub static INTERVAL_METHOD_REASONS: StaticTable<(IntervalMethod, Option<&'static str>)> =
    table(interval_method_reasons);

fn records() -> &'static [CoverageRecord] {
    registry().records
}

fn functional_identity_aliases() -> &'static [FunctionalIdentityAlias] {
    registry().aliases
}

fn attesting_record_ids() -> &'static [&'static str] {
    registry().attesting
}

fn interval_method_reasons() -> &'static [(IntervalMethod, Option<&'static str>)] {
    registry().reasons
}

struct Registry {
    records: &'static [CoverageRecord],
    aliases: &'static [FunctionalIdentityAlias],
    attesting: &'static [&'static str],
    reasons: &'static [(IntervalMethod, Option<&'static str>)],
}

fn registry() -> &'static Registry {
    static CELL: OnceLock<Registry> = OnceLock::new();
    CELL.get_or_init(|| {
        let raw: RawRegistry = serde_json::from_str(SNAPSHOT).expect("coverage records snapshot");
        Registry {
            records: leak_slice(raw.records.into_iter().map(CoverageRecord::from_raw).collect()),
            aliases: leak_slice(
                raw.functional_identity_aliases
                    .into_iter()
                    .map(FunctionalIdentityAlias::from_raw)
                    .collect(),
            ),
            attesting: leak_strs(raw.attesting_record_ids),
            reasons: leak_slice(
                raw.interval_method_reasons.into_iter().map(RawReason::into_pair).collect(),
            ),
        }
    })
}

#[derive(Deserialize)]
struct RawRegistry {
    records: Vec<RawRecord>,
    functional_identity_aliases: Vec<RawAlias>,
    attesting_record_ids: Vec<String>,
    interval_method_reasons: Vec<RawReason>,
}

#[derive(Deserialize)]
struct RawRecord {
    id: String,
    query: String,
    graph_class: String,
    structure: String,
    modality: String,
    inference: String,
    estimator: String,
    interval_method: String,
    se_kind: String,
    dependence: String,
    posterior: String,
    functional: String,
    identification: String,
    nominal: String,
    n_min: u64,
    n_max: u64,
    replicates_min: u32,
    posterior_draws_min: u32,
    unidentified_mass_max: String,
    observed: String,
    mcse: String,
    replicates: u32,
    boundary: bool,
    grid: Vec<RawGrid>,
    #[serde(default)]
    confirmation: Vec<RawGrid>,
    #[serde(default)]
    confirmed_passing_range: Option<[u64; 2]>,
    dgp: String,
    test: String,
    calibration_sha: String,
}

#[derive(Deserialize)]
struct RawGrid {
    point: u8,
    n_min: u64,
    n_max: u64,
    observed: String,
    mcse: String,
    replicates: u32,
    boundary: bool,
}

#[derive(Deserialize)]
struct RawAlias {
    record_id: String,
    measurement_sha: String,
    original_functional: String,
    canonical_functional: String,
}

#[derive(Deserialize)]
struct RawReason {
    method: String,
    reason: Option<String>,
}

impl CoverageRecord {
    fn from_raw(raw: RawRecord) -> Self {
        let id = leak_str(raw.id);
        Self {
            id,
            query: leak_str(raw.query),
            graph_class: leak_str(raw.graph_class),
            structure: leak_str(raw.structure),
            modality: leak_str(raw.modality),
            inference: leak_str(raw.inference),
            estimator: leak_str(raw.estimator),
            interval_method: leak_str(raw.interval_method),
            se_kind: leak_str(raw.se_kind),
            dependence: leak_str(raw.dependence),
            posterior: leak_str(raw.posterior),
            functional: leak_str(raw.functional),
            identification: leak_str(raw.identification),
            nominal: parse_f64(&raw.nominal, id, "nominal"),
            n_min: raw.n_min,
            n_max: raw.n_max,
            replicates_min: raw.replicates_min,
            posterior_draws_min: raw.posterior_draws_min,
            unidentified_mass_max: parse_f64(
                &raw.unidentified_mass_max,
                id,
                "unidentified_mass_max",
            ),
            observed: parse_f64(&raw.observed, id, "observed"),
            mcse: parse_f64(&raw.mcse, id, "mcse"),
            replicates: raw.replicates,
            boundary: raw.boundary,
            grid: leak_grid(raw.grid, id),
            confirmation: leak_grid(raw.confirmation, id),
            confirmed_passing_range: raw.confirmed_passing_range.map(|[lo, hi]| (lo, hi)),
            dgp: leak_str(raw.dgp),
            test: leak_str(raw.test),
            calibration_sha: leak_str(raw.calibration_sha),
        }
    }
}

impl FunctionalIdentityAlias {
    fn from_raw(raw: RawAlias) -> Self {
        Self {
            record_id: leak_str(raw.record_id),
            measurement_sha: leak_str(raw.measurement_sha),
            original_functional: leak_str(raw.original_functional),
            canonical_functional: leak_str(raw.canonical_functional),
        }
    }
}

impl RawReason {
    fn into_pair(self) -> (IntervalMethod, Option<&'static str>) {
        (interval_method(&self.method), self.reason.map(leak_str))
    }
}

fn leak_grid(points: Vec<RawGrid>, id: &str) -> &'static [CoverageGridPoint] {
    if points.is_empty() {
        return &[];
    }
    leak_slice(
        points
            .into_iter()
            .map(|point| CoverageGridPoint {
                point: point.point,
                n_min: point.n_min,
                n_max: point.n_max,
                observed: parse_f64(&point.observed, id, "grid.observed"),
                mcse: parse_f64(&point.mcse, id, "grid.mcse"),
                replicates: point.replicates,
                boundary: point.boundary,
            })
            .collect(),
    )
}

fn interval_method(name: &str) -> IntervalMethod {
    match name {
        "AnalyticSe" => IntervalMethod::AnalyticSe,
        "BootstrapSe" => IntervalMethod::BootstrapSe,
        "PosteriorQuantile" => IntervalMethod::PosteriorQuantile,
        "UnitPosteriorQuantile" => IntervalMethod::UnitPosteriorQuantile,
        "IdentifiedSet" => IntervalMethod::IdentifiedSet,
        "CircularBlockSe" => IntervalMethod::CircularBlockSe,
        "SimultaneousBand" => IntervalMethod::SimultaneousBand,
        "AndersonRubin" => IntervalMethod::AndersonRubin,
        "None" => IntervalMethod::None,
        other => panic!("coverage snapshot: unknown interval method {other}"),
    }
}

fn parse_f64(text: &str, id: &str, field: &str) -> f64 {
    text.parse::<f64>()
        .unwrap_or_else(|_| panic!("coverage snapshot: {id} {field} is not an f64: {text}"))
}

fn leak_str(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

fn leak_slice<T>(items: Vec<T>) -> &'static [T] {
    Box::leak(items.into_boxed_slice())
}

fn leak_strs(items: Vec<String>) -> &'static [&'static str] {
    if items.is_empty() {
        return &[];
    }
    leak_slice(items.into_iter().map(leak_str).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_record_count_and_id_round_trip() {
        let raw: serde_json::Value = serde_json::from_str(SNAPSHOT).expect("coverage snapshot");
        let rows = raw["records"].as_array().expect("records array");
        assert_eq!(RECORDS.len(), rows.len());
        let id = rows[0]["id"].as_str().expect("id");
        let loaded = record_for(id).expect("id round-trip");
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.query, rows[0]["query"].as_str().expect("query"));
        assert_eq!(
            loaded.mcse.to_bits(),
            rows[0]["mcse"].as_str().expect("mcse").parse::<f64>().expect("mcse number").to_bits()
        );
        assert_eq!(loaded.grid.len(), rows[0]["grid"].as_array().expect("grid").len());
        let expected_attesting: Vec<&str> = raw["attesting_record_ids"]
            .as_array()
            .expect("attesting ID array")
            .iter()
            .map(|id| id.as_str().expect("attesting ID"))
            .collect();
        assert_eq!(ATTESTING_RECORD_IDS.iter().copied().collect::<Vec<_>>(), expected_attesting);
        assert!(method_listed(IntervalMethod::AnalyticSe));
        assert!(method_listed(IntervalMethod::None));
    }

    /// Same shape as the facade's record lookup: the iterator item must be `'static`.
    fn record_for(id: &str) -> Option<&'static CoverageRecord> {
        RECORDS.iter().find(|row| row.id == id)
    }

    /// Same shape as the facade test over [`INTERVAL_METHOD_REASONS`].
    fn method_listed(method: IntervalMethod) -> bool {
        let has_record = RECORDS.iter().any(|row| row.interval_method == method.as_str());
        INTERVAL_METHOD_REASONS
            .iter()
            .any(|(item, reason)| *item == method && (has_record || reason.is_some()))
    }
}
