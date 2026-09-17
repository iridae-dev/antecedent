//! Bind facade executions to record-keyed coverage tallies.
//!
//! The construction a coverage record describes is read from the runtime's
//! own calibration match key ([`StudyResult::calibration_bases`]), the same
//! key [`StudyResult::claim`] matches against the registry. A test therefore
//! cannot record a construction the facade does not report under that key.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code)]

use antecedent::{CausalContract, PreparedStudy, Study, StudyResult};

use super::calibration::{Construction, CoverageTally, ScopeFacts};

/// Every reported interval's construction and scope facts, primary first.
///
/// # Panics
///
/// When the result's structural masses do not conserve.
#[must_use]
pub fn constructions(
    contract: &CausalContract,
    result: &StudyResult,
) -> Vec<(Construction, ScopeFacts)> {
    result
        .calibration_bases(contract)
        .expect("calibration bases")
        .into_iter()
        .map(|basis| {
            let key = basis.key;
            (
                Construction {
                    query: key.query,
                    graph_class: key.graph_class,
                    structure: key.structure,
                    modality: key.modality,
                    inference: key.inference,
                    estimator: key.estimator,
                    interval_method: key.interval_method,
                    se_kind: key.se_kind,
                    dependence: key.dependence,
                    posterior: key.posterior,
                    functional: key.functional,
                    identification: key.identification,
                    reported_level: key.level,
                },
                ScopeFacts {
                    row_count: basis.scope.row_count,
                    replicates_ok: basis.scope.replicates_ok,
                    posterior_draws: basis.scope.posterior_draws,
                    unidentified_mass: basis.scope.unidentified_mass,
                },
            )
        })
        .collect()
}

/// Bind `result`, executed from `contract`'s program, to `tally`: the
/// reported interval whose method the tally's [`super::calibration::RecordKey`] declared.
///
/// # Panics
///
/// When the execution reported no interval of that method.
pub fn bind_contract(tally: &mut CoverageTally, contract: &CausalContract, result: &StudyResult) {
    let wanted = tally.record_interval().expect("bind needs a record-keyed tally");
    let reported = constructions(contract, result);
    let methods: Vec<String> =
        reported.iter().map(|(construction, _)| construction.interval_method.clone()).collect();
    let (construction, scope) = reported
        .into_iter()
        .find(|(construction, _)| construction.interval_method == wanted)
        .unwrap_or_else(|| {
            panic!("the execution reported no {wanted} interval (it reported {methods:?})")
        });
    tally.bind(&construction, scope);
}

/// [`bind_contract`] for a result of [`Study::run`] on `study`.
///
/// # Panics
///
/// When inspection fails or the interval was not reported.
pub fn bind(tally: &mut CoverageTally, study: &Study, result: &StudyResult) {
    let contract = study.inspect().expect("inspect");
    bind_contract(tally, &contract, result);
}

/// [`bind_contract`] for a result of [`PreparedStudy::estimate`].
///
/// # Panics
///
/// When the contract cannot be built or the interval was not reported.
pub fn bind_prepared(tally: &mut CoverageTally, prepared: &PreparedStudy, result: &StudyResult) {
    let contract = prepared.contract().expect("contract");
    bind_contract(tally, &contract, result);
}

/// Bind the same execution to several tallies (e.g. one test scoring two
/// levels of the same interval).
pub fn bind_all(tallies: &mut [&mut CoverageTally], study: &Study, result: &StudyResult) {
    let contract = study.inspect().expect("inspect");
    for tally in tallies {
        bind_contract(tally, &contract, result);
    }
}
