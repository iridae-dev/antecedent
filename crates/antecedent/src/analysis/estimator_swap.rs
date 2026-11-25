//! Swap a listed estimator onto a program that already carries a cached identification product.
//!
//! Admission is the cell row in `parity/estimator_validity.toml`: the estimator must be in that
//! row's `listed` estimators and in its `valid` estimators. The identification digest and the
//! identification-product digest are the bytes already on the program. This module does not call
//! `identification_product_wire`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::OnceLock;

use antecedent_io::ProgramIdentityWire;

use crate::EstimatorId;

const VALIDITY: &str = include_str!("../../../../parity/estimator_validity.toml");

/// Why a cached identification product was not reused under a new estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EstimatorSwapRefusal {
    /// The program has no identification-product digest to reuse.
    NoCachedIdentification,
    /// `cell` is not a coordinate in `parity/estimator_validity.toml`.
    UnknownCell,
    /// The estimator is absent from the cell's listed estimators, or it is not valid for the cell.
    EstimatorNotAdmitted,
}

struct ValidityRow {
    valid: Vec<String>,
    listed: Vec<String>,
}

fn rows() -> &'static HashMap<String, ValidityRow> {
    static ROWS: OnceLock<HashMap<String, ValidityRow>> = OnceLock::new();
    ROWS.get_or_init(parse_validity)
}

fn parse_validity() -> HashMap<String, ValidityRow> {
    let mut rows = HashMap::new();
    let mut coordinate = None;
    let mut valid = Vec::new();
    let mut listed = Vec::new();
    let mut which = List::None;
    for line in VALIDITY.lines() {
        let line = line.trim();
        if line == "[[cell]]" {
            store(&mut rows, &mut coordinate, &mut valid, &mut listed);
            which = List::None;
            continue;
        }
        if let Some(rest) = line.strip_prefix("coordinate = \"") {
            coordinate = rest.strip_suffix('"').map(str::to_owned);
            continue;
        }
        if line.starts_with("valid =") {
            which = List::Valid;
            continue;
        }
        if line.starts_with("listed =") {
            which = List::Listed;
            continue;
        }
        if line.starts_with("unlisted_valid") || line.starts_with("listed_not_valid") {
            which = List::None;
            continue;
        }
        let Some(name) = quoted_name(line) else { continue };
        match which {
            List::Valid => valid.push(name.to_owned()),
            List::Listed => listed.push(name.to_owned()),
            List::None => {}
        }
    }
    store(&mut rows, &mut coordinate, &mut valid, &mut listed);
    rows
}

fn store(
    rows: &mut HashMap<String, ValidityRow>,
    coordinate: &mut Option<String>,
    valid: &mut Vec<String>,
    listed: &mut Vec<String>,
) {
    if let Some(coordinate) = coordinate.take() {
        rows.insert(
            coordinate,
            ValidityRow { valid: std::mem::take(valid), listed: std::mem::take(listed) },
        );
    }
}

fn quoted_name(line: &str) -> Option<&str> {
    line.trim().trim_end_matches(',').strip_prefix('"')?.strip_suffix('"')
}

enum List {
    None,
    Valid,
    Listed,
}

/// Reuse `program`'s cached identification under `estimator` when that estimator is listed and
/// valid for `cell`.
///
/// The returned program copies `identification` and `identification_product`. Its estimator
/// commitment is `estimator`, so `program_digest` changes when the commitment did.
///
/// # Errors
///
/// [`EstimatorSwapRefusal::NoCachedIdentification`] when the program has no identification
/// product. [`EstimatorSwapRefusal::UnknownCell`] when `cell` is not in the validity file.
/// [`EstimatorSwapRefusal::EstimatorNotAdmitted`] when `estimator` is absent from the cell's
/// listed estimators or is not valid for the cell.
pub(crate) fn swap_estimator(
    program: &ProgramIdentityWire,
    cell: &str,
    estimator: EstimatorId,
) -> Result<ProgramIdentityWire, EstimatorSwapRefusal> {
    if program.identification_product.is_none() {
        return Err(EstimatorSwapRefusal::NoCachedIdentification);
    }
    let Some(row) = rows().get(cell) else {
        return Err(EstimatorSwapRefusal::UnknownCell);
    };
    let name = estimator.as_str();
    let listed = row.listed.iter().any(|item| item == name);
    let valid = row.valid.iter().any(|item| item == name);
    if !listed || !valid {
        return Err(EstimatorSwapRefusal::EstimatorNotAdmitted);
    }
    let mut swapped = program.clone();
    let name = name.to_owned();
    swapped.commitments.estimator = Some(name.clone());
    swapped.commitments.resolved_estimator = Some(name);
    Ok(swapped)
}

#[cfg(test)]
mod tests {
    use antecedent_core::IDENTITY_FORMAT;
    use antecedent_io::{InferentialCommitmentsWire, ProgramIdentityWire, program_digest};

    use super::{EstimatorSwapRefusal, swap_estimator};
    use crate::EstimatorId;

    const DERIVATIVE: &str = "AverageDerivative:TemporalAdmg:accepted:Bayesian:cheap";
    const CURVE: &str = "ResponseCurve:Admg:accepted:Bayesian:cheap";
    const IDENTIFICATION: [u8; 32] = [0x11; 32];
    const PRODUCT: [u8; 32] = [0x22; 32];

    fn program(estimator: &str) -> ProgramIdentityWire {
        ProgramIdentityWire {
            format: IDENTITY_FORMAT,
            target: [0x33; 32],
            identification: IDENTIFICATION,
            identification_product: Some(PRODUCT),
            completion_budget: None,
            commitments: InferentialCommitmentsWire {
                format: IDENTITY_FORMAT,
                estimator: Some(estimator.to_owned()),
                resolved_estimator: Some(estimator.to_owned()),
                identifier: None,
                inference: "bayesian".to_owned(),
                validation_suite: None,
                interval_method: "analytic_se".to_owned(),
                se_kind: None,
                prior_required: true,
            },
            temporal_attribution_rows: None,
            functional_program: None,
            checked_aipw_lowering: None,
            checked_frontdoor_lowering: None,
            checked_iv_lowering: None,
            checked_linear_adjustment_lowering: None,
            checked_functional_response_grid: None,
            checked_nested_counterfactual: None,
            checked_dr_lowering: None,
            graph_posterior_programs: None,
            structure_provenance: None,
        }
    }

    #[test]
    fn swapping_a_listed_estimator_changes_the_program_digest_only() {
        let base = program("response.riesz_ade");
        let swapped =
            swap_estimator(&base, DERIVATIVE, EstimatorId::TemporalResponseBayesian).unwrap();
        let base_program = program_digest(&base).unwrap();
        let swapped_program = program_digest(&swapped).unwrap();
        assert_ne!(base_program, swapped_program);
        assert_eq!(swapped.identification, IDENTIFICATION);
        assert_eq!(swapped.identification_product, Some(PRODUCT));
        assert_eq!(swapped.commitments.estimator.as_deref(), Some("response.temporal.bayesian"));
        assert_eq!(
            swapped.commitments.resolved_estimator.as_deref(),
            Some("response.temporal.bayesian")
        );
    }

    #[test]
    fn an_estimator_absent_from_the_cell_is_refused() {
        let base = program("functional.effect");
        assert_eq!(
            swap_estimator(&base, DERIVATIVE, EstimatorId::LinearAdjustmentAte),
            Err(EstimatorSwapRefusal::EstimatorNotAdmitted)
        );
        assert_eq!(
            swap_estimator(&base, CURVE, EstimatorId::ResponseBayesian),
            Err(EstimatorSwapRefusal::EstimatorNotAdmitted)
        );
        let admitted = swap_estimator(&base, CURVE, EstimatorId::FunctionalEffect).unwrap();
        assert_eq!(admitted.identification, base.identification);
        assert_eq!(admitted.identification_product, base.identification_product);
        assert_eq!(
            swap_estimator(&base, "not-a-cell", EstimatorId::FunctionalEffect),
            Err(EstimatorSwapRefusal::UnknownCell)
        );
        let mut uncached = base;
        uncached.identification_product = None;
        assert_eq!(
            swap_estimator(&uncached, CURVE, EstimatorId::FunctionalEffect),
            Err(EstimatorSwapRefusal::NoCachedIdentification)
        );
    }
}
