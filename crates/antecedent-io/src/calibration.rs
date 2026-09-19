//! Calibration match: one reported interval against the coverage records.
//!
//! The only implementation of the rule. The runtime claim
//! (`antecedent::StudyResult::claim`) and the independent consumer
//! ([`crate::verify_contract_against_body`]) both call [`calibration_slot`]
//! on the same [`CalibrationBasisWire`], which the claim carries and
//! `claim_id` covers. Records come from the generated
//! [`crate::coverage_records_data::RECORDS`] table
//! (`parity/coverage_records.toml`, emitted by the coverage tests through
//! `CoverageTally::for_record` and collected by
//! `scripts/collect_coverage_records.py`).
//!
//! Vocabulary:
//!
//! - `calibrated`: a non-boundary record measured this construction (every key
//!   field equal, including the nominal level and the identification label)
//!   and the execution lies inside what it measured: its row count inside the
//!   record's measured row-count range (the span of its sample-size grid points,
//!   never extrapolated beyond them), at least as many successful resampling
//!   replicates and posterior draws, and no more non-identified mass. A record
//!   is a boundary when any of its grid points is: a construction that failed
//!   at one measured sample size is not calibrated anywhere in the range.
//! - `scope_not_assessed`: a record measured the construction but the
//!   execution is outside its scope (the reason names which bound), or the
//!   covering record is a named boundary / under-coverage measurement
//!   (`boundary_record`, with its observed coverage).
//! - `unavailable`: no record measured this construction, level or
//!   identification; or no interval was reported.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::contract_section::CalibrationSlotWire;
use crate::coverage_records_data::{CoverageGridPoint, CoverageRecord, RECORDS};

/// No interval was reported.
pub const NO_INTERVAL_REPORTED: &str = "no_interval_reported";
/// No coverage record measured this construction.
pub const CONSTRUCTION_NOT_MEASURED: &str = "estimator_grid_not_measured";
/// Records measured the construction, but at another nominal level.
pub const LEVEL_NOT_MEASURED: &str = "interval_level_not_measured";
/// Records measured the construction, but under another identification label.
pub const IDENTIFICATION_NOT_MEASURED: &str = "identification_not_measured";
/// The covering record is a named boundary (under-coverage) measurement.
pub const BOUNDARY_RECORD: &str = "boundary_record";
/// The execution's row count lies outside every measured range.
pub const SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE: &str = "sample_size_outside_measured_range";
/// Fewer resampling replicates succeeded than the record measured.
pub const REPLICATES_BELOW_MEASURED: &str = "resampling_replicates_below_measured";
/// Fewer posterior draws than the record measured.
pub const POSTERIOR_DRAWS_BELOW_MEASURED: &str = "posterior_draws_below_measured";
/// More non-identified structural mass than the record measured.
pub const UNIDENTIFIED_MASS_ABOVE_MEASURED: &str = "unidentified_mass_above_measured";
/// The claim carries no match basis to re-derive the slot from.
pub const BASIS_MISSING: &str = "calibration_basis_missing";

/// Every reason code the matcher can emit (all listed in `parity/reason_codes.toml`).
pub const REASON_CODES: [&str; 10] = [
    NO_INTERVAL_REPORTED,
    CONSTRUCTION_NOT_MEASURED,
    LEVEL_NOT_MEASURED,
    IDENTIFICATION_NOT_MEASURED,
    BOUNDARY_RECORD,
    SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE,
    REPLICATES_BELOW_MEASURED,
    POSTERIOR_DRAWS_BELOW_MEASURED,
    UNIDENTIFIED_MASS_ABOVE_MEASURED,
    BASIS_MISSING,
];

/// Tolerance for comparing nominal levels and structural masses.
const TOLERANCE: f64 = 1e-9;

/// Construction of one reported interval: the fields a coverage record must
/// equal to describe it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CalibrationKeyWire {
    /// Support-matrix query axis name.
    pub query: String,
    /// Support-matrix graph axis, including `CoDetermined` / `Unknown`.
    pub graph_class: String,
    /// `fixed` (explicit or accepted structure) or `graph_posterior`.
    pub structure: String,
    /// Data modality the execution ran on (`tabular`, `series`, `panel`,
    /// `event`, `multi_env`): a record binds only to the modality it measured.
    pub modality: String,
    /// `Frequentist` or `Bayesian`.
    pub inference: String,
    /// Resolved plan estimator (`logical_plan.estimator`); empty when none.
    pub estimator: String,
    /// `IntervalMethod::as_str()`.
    pub interval_method: String,
    /// Analytic SE kind recorded by the estimator; empty when not analytic.
    pub se_kind: String,
    /// `iid`, `panel_cluster`, or `circular_block:<family>`.
    pub dependence: String,
    /// Posterior construction (`<backend>.<likelihood>.<prior>`); empty for
    /// Frequentist executions.
    pub posterior: String,
    /// Target population, outcome functional, contrast, horizon and policy
    /// the support axis query name leaves open.
    pub functional: String,
    /// Nominal level of the reported interval.
    pub level: f64,
    /// `point` (all structural mass identified under an identified status) or
    /// `partial`, followed by `+<rule>` for every derivation rule in
    /// [`CONSTRUCTION_RULES`] the identification used ([`identification_key`]):
    /// an interval over a different identified adjustment is a different
    /// construction and binds only to records measured for it.
    pub identification: String,
}

/// Derivation rules that make an identification a distinct interval
/// construction for coverage matching. A parent-adjusted temporal pulse
/// (`temporal.parent_adjustment`) fits a different adjustment set than the
/// unfolding-derived one on the same query, so its coverage is separate
/// evidence.
pub const CONSTRUCTION_RULES: &[&str] = &["temporal.parent_adjustment"];

/// The calibration key's `identification` field: the `point` / `partial`
/// label plus `+<rule>` for each [`CONSTRUCTION_RULES`] entry among
/// `derivation_rules`, in [`CONSTRUCTION_RULES`] order.
#[must_use]
pub fn identification_key<'a>(
    label: &str,
    derivation_rules: impl IntoIterator<Item = &'a str>,
) -> String {
    let used: Vec<&str> = derivation_rules.into_iter().collect();
    let mut key = label.to_string();
    for rule in CONSTRUCTION_RULES {
        if used.contains(rule) {
            key.push('+');
            key.push_str(rule);
        }
    }
    key
}

/// Execution facts a record's scope is checked against.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CalibrationScopeWire {
    /// Data-snapshot rows.
    pub row_count: u64,
    /// Resampling replicates that succeeded, when resampling-based.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicates_ok: Option<u32>,
    /// Posterior draws, when posterior-based.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_draws: Option<u32>,
    /// Structural mass that is not identified (unidentified + unevaluable +
    /// incomplete search).
    pub unidentified_mass: f64,
}

/// Match key plus scope facts of one reported interval.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CalibrationBasisWire {
    /// Construction.
    pub key: CalibrationKeyWire,
    /// Execution facts.
    pub scope: CalibrationScopeWire,
}

impl CalibrationKeyWire {
    fn same_construction(&self, record: &CoverageRecord) -> bool {
        record.query == self.query
            && record.graph_class == self.graph_class
            && record.structure == self.structure
            && record.modality == self.modality
            && record.inference == self.inference
            && record.estimator == self.estimator
            && record.interval_method == self.interval_method
            && record.se_kind == self.se_kind
            && record.dependence == self.dependence
            && record.posterior == self.posterior
            && record.functional == self.functional
    }
}

/// Calibration slot of one reported interval (secondary list empty).
#[must_use]
pub fn calibration_slot(basis: &CalibrationBasisWire) -> CalibrationSlotWire {
    calibration_slot_in(basis, RECORDS)
}

/// Calibration slot of the primary interval, with one secondary slot per
/// further reported interval, in order.
#[must_use]
pub fn calibration_slots(bases: &[CalibrationBasisWire]) -> CalibrationSlotWire {
    let Some((primary, rest)) = bases.split_first() else {
        return CalibrationSlotWire::unavailable(BASIS_MISSING);
    };
    let mut slot = calibration_slot(primary);
    slot.secondary = rest.iter().map(calibration_slot).collect();
    slot
}

/// [`calibration_slot`] against an explicit record table.
#[must_use]
pub fn calibration_slot_in(
    basis: &CalibrationBasisWire,
    records: &[CoverageRecord],
) -> CalibrationSlotWire {
    let key = &basis.key;
    let scope = &basis.scope;
    let unavailable = |reason: &str| CalibrationSlotWire::unavailable(reason).with_basis(basis);
    if key.interval_method == "none" {
        return unavailable(NO_INTERVAL_REPORTED);
    }
    let construction: Vec<&CoverageRecord> =
        records.iter().filter(|record| key.same_construction(record)).collect();
    if construction.is_empty() {
        return unavailable(CONSTRUCTION_NOT_MEASURED);
    }
    let at_level: Vec<&CoverageRecord> = construction
        .into_iter()
        .filter(|record| (record.nominal - key.level).abs() <= TOLERANCE)
        .collect();
    if at_level.is_empty() {
        return unavailable(LEVEL_NOT_MEASURED);
    }
    let measured: Vec<&CoverageRecord> =
        at_level.into_iter().filter(|record| record.identification == key.identification).collect();
    if measured.is_empty() {
        return unavailable(IDENTIFICATION_NOT_MEASURED);
    }
    let covering: Vec<&CoverageRecord> = measured
        .iter()
        .copied()
        .filter(|record| {
            let (lo, hi) = measured_range(record);
            lo <= scope.row_count && scope.row_count <= hi
        })
        .collect();
    let Some(governing) = worst(&covering) else {
        let nearest = measured
            .iter()
            .copied()
            .min_by_key(|record| {
                let (lo, hi) = measured_range(record);
                (lo.saturating_sub(scope.row_count)).max(scope.row_count.saturating_sub(hi))
            })
            .expect("measured is non-empty");
        return CalibrationSlotWire::from_record(nearest, "scope_not_assessed")
            .with_reason(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE)
            .with_basis(basis);
    };
    let boundaries: Vec<&CoverageRecord> =
        covering.iter().copied().filter(|record| is_boundary(record)).collect();
    if let Some(boundary) = worst(&boundaries) {
        let mut slot = CalibrationSlotWire::from_record(boundary, "scope_not_assessed")
            .with_reason(BOUNDARY_RECORD)
            .with_basis(basis);
        // The coverage that makes it a boundary: the worst failing grid point.
        if let Some(point) = boundary
            .grid
            .iter()
            .filter(|point| point.boundary)
            .min_by(|a, b| a.observed.total_cmp(&b.observed))
        {
            slot.observed = Some(point.observed);
        }
        return slot;
    }
    let outside = if covering.iter().any(|record| {
        record.replicates_min > 0 && scope.replicates_ok.unwrap_or(0) < record.replicates_min
    }) {
        Some(REPLICATES_BELOW_MEASURED)
    } else if covering.iter().any(|record| {
        record.posterior_draws_min > 0
            && scope.posterior_draws.unwrap_or(0) < record.posterior_draws_min
    }) {
        Some(POSTERIOR_DRAWS_BELOW_MEASURED)
    } else if covering
        .iter()
        .any(|record| scope.unidentified_mass > record.unidentified_mass_max + TOLERANCE)
    {
        Some(UNIDENTIFIED_MASS_ABOVE_MEASURED)
    } else {
        None
    };
    match outside {
        Some(reason) => CalibrationSlotWire::from_record(governing, "scope_not_assessed")
            .with_reason(reason)
            .with_basis(basis),
        None => CalibrationSlotWire::from_record(governing, "calibrated").with_basis(basis),
    }
}

/// Row-count range a record measured: the span of its sample-size grid points
/// (`n_min` of the smallest point to `n_max` of the largest), or its own
/// `n_min..n_max` for a record measured at a single sample size.
#[must_use]
pub fn measured_range(record: &CoverageRecord) -> (u64, u64) {
    let points: &[CoverageGridPoint] = record.grid;
    match (points.iter().map(|p| p.n_min).min(), points.iter().map(|p| p.n_max).max()) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => (record.n_min, record.n_max),
    }
}

/// Whether a record is a boundary measurement: flagged itself, or under-covering
/// (outside its nominal band, or a named boundary) at any grid point. A failing
/// point is never averaged into a pass over the range.
#[must_use]
pub fn is_boundary(record: &CoverageRecord) -> bool {
    record.boundary || record.grid.iter().any(|point| point.boundary)
}

/// Lowest observed coverage among `records` (ties broken by id): the record
/// that governs a slot when several measurements cover one execution.
fn worst<'a>(records: &[&'a CoverageRecord]) -> Option<&'a CoverageRecord> {
    records.iter().copied().min_by(|a, b| a.observed.total_cmp(&b.observed).then(a.id.cmp(b.id)))
}

/// Re-derive a claim's slot (primary and secondary) from the bases it carries.
///
/// A slot without a basis re-derives to `unavailable` / `calibration_basis_missing`.
/// Caller-attested evidence that records `attested_not_reverifiable` is
/// uncalibrated by construction, not a missing match key.
#[must_use]
pub fn rederive_calibration(slot: &CalibrationSlotWire) -> CalibrationSlotWire {
    let Some(primary) = slot.basis.as_ref() else {
        if slot.status == "unavailable"
            && slot.reason.as_deref()
                == Some(antecedent_core::reason_code!("attested_not_reverifiable"))
        {
            return CalibrationSlotWire::unavailable("attested_not_reverifiable");
        }
        return CalibrationSlotWire::unavailable(BASIS_MISSING);
    };
    let mut bases = vec![primary.clone()];
    for secondary in &slot.secondary {
        let Some(reported) = secondary.basis.as_ref() else {
            let mut out = calibration_slots(&bases);
            out.secondary.push(CalibrationSlotWire::unavailable(BASIS_MISSING));
            return out;
        };
        bases.push(reported.clone());
    }
    calibration_slots(&bases)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &'static str) -> CoverageRecord {
        CoverageRecord {
            id,
            query: "AverageEffect",
            graph_class: "Dag",
            structure: "fixed",
            modality: "tabular",
            inference: "Frequentist",
            estimator: "aipw",
            interval_method: "bootstrap_se",
            se_kind: "",
            dependence: "iid",
            posterior: "",
            functional: "all_observed.mean",
            identification: "point",
            nominal: 0.95,
            n_min: 500,
            n_max: 500,
            replicates_min: 200,
            posterior_draws_min: 0,
            unidentified_mass_max: 0.0,
            observed: 0.948,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
            grid: &[],
            dgp: "crates/antecedent/tests/x.rs::dgp",
            test: "crates/antecedent/tests/x.rs::test",
            calibration_sha: "0123456789abcdef0123456789abcdef01234567",
        }
    }

    fn basis() -> CalibrationBasisWire {
        CalibrationBasisWire {
            key: CalibrationKeyWire {
                query: "AverageEffect".into(),
                graph_class: "Dag".into(),
                structure: "fixed".into(),
                modality: "tabular".into(),
                inference: "Frequentist".into(),
                estimator: "aipw".into(),
                interval_method: "bootstrap_se".into(),
                se_kind: String::new(),
                dependence: "iid".into(),
                posterior: String::new(),
                functional: "all_observed.mean".into(),
                level: 0.95,
                identification: "point".into(),
            },
            scope: CalibrationScopeWire {
                row_count: 500,
                replicates_ok: Some(200),
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        }
    }

    #[test]
    fn inside_scope_is_calibrated_against_the_worst_covering_record() {
        let mut low = record("cov.low");
        low.observed = 0.94;
        let slot = calibration_slot_in(&basis(), &[record("cov.high"), low]);
        assert_eq!(slot.status, "calibrated");
        assert_eq!(slot.record_id.as_deref(), Some("cov.low"));
        assert_eq!(slot.observed, Some(0.94));
        assert_eq!(slot.reason, None);
    }

    #[test]
    fn too_few_successful_replicates_are_not_calibrated() {
        let mut b = basis();
        b.scope.replicates_ok = Some(2);
        let slot = calibration_slot_in(&b, &[record("cov.a")]);
        assert_eq!(slot.status, "scope_not_assessed");
        assert_eq!(slot.reason.as_deref(), Some(REPLICATES_BELOW_MEASURED));
    }

    #[test]
    fn level_and_identification_mismatches_are_unavailable() {
        let mut b = basis();
        b.key.level = 0.9;
        assert_eq!(
            calibration_slot_in(&b, &[record("cov.a")]).reason.as_deref(),
            Some(LEVEL_NOT_MEASURED)
        );
        let mut b = basis();
        b.key.identification = "partial".into();
        let slot = calibration_slot_in(&b, &[record("cov.a")]);
        assert_eq!(slot.status, "unavailable");
        assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
    }

    #[test]
    fn parent_adjustment_binds_only_to_records_measured_for_it() {
        assert_eq!(identification_key("point", ["backdoor.criterion", "temporal.unfold"]), "point");
        let parent = identification_key(
            "point",
            ["temporal.parent_adjustment", "backdoor.adjustment_set", "temporal.unfold"],
        );
        assert_eq!(parent, "point+temporal.parent_adjustment");

        let unfolding = record("cov.unfolding");
        let mut measured = record("cov.parent");
        measured.identification = "point+temporal.parent_adjustment";

        let mut b = basis();
        b.key.identification = parent;
        let slot = calibration_slot_in(&b, &[unfolding]);
        assert_eq!(slot.status, "unavailable");
        assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
        let slot = calibration_slot_in(&b, &[unfolding, measured]);
        assert_eq!(slot.status, "calibrated");
        assert_eq!(slot.record_id.as_deref(), Some("cov.parent"));

        // And an unfolding-identified result never binds to the parent record.
        let slot = calibration_slot_in(&basis(), &[measured]);
        assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
    }

    #[test]
    fn row_count_outside_the_measured_range_is_not_assessed() {
        let mut b = basis();
        b.scope.row_count = 1200;
        let slot = calibration_slot_in(&b, &[record("cov.a")]);
        assert_eq!(slot.status, "scope_not_assessed");
        assert_eq!(slot.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
        assert_eq!(slot.scope_n, Some(500));
    }

    const GRID: [CoverageGridPoint; 3] = [
        CoverageGridPoint {
            point: 0,
            n_min: 250,
            n_max: 250,
            observed: 0.945,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
        },
        CoverageGridPoint {
            point: 1,
            n_min: 500,
            n_max: 500,
            observed: 0.948,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
        },
        CoverageGridPoint {
            point: 2,
            n_min: 1000,
            n_max: 1000,
            observed: 0.951,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
        },
    ];
    static FAILING_SMALL: [CoverageGridPoint; 3] = {
        let mut grid = GRID;
        grid[0].observed = 0.90;
        grid[0].boundary = true;
        grid
    };

    fn gridded(id: &'static str, grid: &'static [CoverageGridPoint]) -> CoverageRecord {
        let mut out = record(id);
        out.n_min = 250;
        out.n_max = 1000;
        out.grid = grid;
        out
    }

    #[test]
    fn a_passing_grid_is_calibrated_across_its_whole_measured_range() {
        for n in [250, 251, 500, 777, 1000] {
            let mut b = basis();
            b.scope.row_count = n;
            let slot = calibration_slot_in(&b, &[gridded("cov.grid", &GRID)]);
            assert_eq!(slot.status, "calibrated", "n = {n}: {slot:?}");
            assert_eq!((slot.scope_n, slot.scope_n_max), (Some(250), Some(1000)));
        }
        for n in [249, 1001, 5000] {
            let mut b = basis();
            b.scope.row_count = n;
            let slot = calibration_slot_in(&b, &[gridded("cov.grid", &GRID)]);
            assert_eq!(slot.status, "scope_not_assessed", "n = {n}: {slot:?}");
            assert_eq!(slot.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
        }
    }

    #[test]
    fn the_range_is_the_grid_span_not_a_contradicting_summary() {
        let mut stale = gridded("cov.grid", &GRID);
        stale.n_min = 500;
        stale.n_max = 500;
        let mut b = basis();
        b.scope.row_count = 300;
        assert_eq!(calibration_slot_in(&b, &[stale]).status, "calibrated");
    }

    #[test]
    fn a_grid_that_fails_at_any_point_is_a_boundary_over_the_whole_range() {
        // Even at the passing base point: a failure at 250 rows is not averaged
        // into a pass at 500.
        for n in [250, 500, 1000] {
            let mut b = basis();
            b.scope.row_count = n;
            let slot = calibration_slot_in(&b, &[gridded("cov.grid", &FAILING_SMALL)]);
            assert_eq!(slot.status, "scope_not_assessed", "n = {n}: {slot:?}");
            assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
            assert_eq!(slot.observed, Some(0.90), "the failing point's coverage is reported");
        }
    }

    #[test]
    fn a_covering_boundary_record_is_reported_with_its_coverage() {
        let mut boundary = record("cov.boundary");
        boundary.boundary = true;
        boundary.observed = 0.885;
        let slot = calibration_slot_in(&basis(), &[record("cov.a"), boundary]);
        assert_eq!(slot.status, "scope_not_assessed");
        assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
        assert_eq!(slot.record_id.as_deref(), Some("cov.boundary"));
        assert_eq!(slot.observed, Some(0.885));
    }

    #[test]
    fn no_interval_and_no_record_are_unavailable_with_codes() {
        let mut b = basis();
        b.key.interval_method = "none".into();
        assert_eq!(
            calibration_slot_in(&b, &[record("cov.a")]).reason.as_deref(),
            Some(NO_INTERVAL_REPORTED)
        );
        assert_eq!(
            calibration_slot_in(&basis(), &[]).reason.as_deref(),
            Some(CONSTRUCTION_NOT_MEASURED)
        );
    }

    #[test]
    fn rederive_reproduces_primary_and_secondary_slots() {
        let mut secondary = basis();
        secondary.key.interval_method = "identified_set".into();
        let slot = calibration_slots(&[basis(), secondary]);
        assert_eq!(slot.secondary.len(), 1);
        assert_eq!(rederive_calibration(&slot), slot);
        let mut forged = slot.clone();
        forged.status = "calibrated".into();
        forged.secondary[0].status = "calibrated".into();
        assert_ne!(rederive_calibration(&forged), forged);
    }

    #[test]
    fn attested_external_estimate_stays_uncalibrated_without_a_basis() {
        let slot = CalibrationSlotWire::unavailable("attested_not_reverifiable");
        assert_eq!(rederive_calibration(&slot), slot);
        assert_eq!(
            rederive_calibration(&CalibrationSlotWire::unavailable("forged")).reason.as_deref(),
            Some(BASIS_MISSING)
        );
    }
}
