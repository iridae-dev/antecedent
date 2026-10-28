//! The finite program search reads a retained response law or mean grid and nothing else, so the
//! data-side declarations of a study reach it only as the law's own draw weights and support
//! mask. Known answers: unequal draw weights set the expectation and variance exactly (the
//! weighted sums are binary-exact), a zero-weight draw contributes nothing, a negative weight is
//! refused at the artifact, a masked coordinate and a non-finite draw are refused and never
//! dropped, and a candidate outside its declared control bounds is excluded, never valued.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_design::decision_contract::{
    ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
};
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
use antecedent_design::program_search::{
    FiniteProgramDomain, PreparedFiniteProgramSearch, ProgramCandidate, ProgramControl,
    ProgramFeasibility, ProgramSearchError, ProgramSearchScope, ProgramSearchSource,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};

fn quantity(regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "USD".into(),
        population_id: "observed".into(),
        regime_id: regime.into(),
        horizon: 2,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn contract(objective: ProgramObjective) -> ProgramSearchContract {
    ProgramSearchContract {
        space: ProgramSearchSpace {
            id: "two-channel schedules".into(),
            variables: vec!["response".into()],
        },
        objective,
        constraints: vec![],
    }
}

fn domain() -> FiniteProgramDomain {
    let control = |raw, offset| ProgramControl {
        variable: VariableId::from_raw(raw),
        offset,
        lower: 0.,
        upper: 4.,
        unit_cost: 1.,
    };
    FiniteProgramDomain {
        controls: vec![control(0, 0), control(1, 1)],
        budget: Some(5.),
        restrictions: vec![],
    }
}

/// reference (0, 0), a (4, 0), joint (4, 4) which the budget of 5 excludes, and out (9, 0) which
/// the control bound of 4 excludes.
fn candidates() -> Vec<ProgramCandidate> {
    [("reference", [0., 0.]), ("a", [4., 0.]), ("joint", [4., 4.]), ("out", [9., 0.])]
        .into_iter()
        .map(|(id, controls)| ProgramCandidate {
            id: id.into(),
            controls: controls.to_vec(),
            inputs: vec![quantity(id)],
        })
        .collect()
}

fn search(objective: ProgramObjective) -> PreparedFiniteProgramSearch {
    PreparedFiniteProgramSearch::compile(
        contract(objective),
        domain(),
        candidates(),
        ProgramSearchScope {
            population: "observed".into(),
            horizon: 2,
            utility_units: "USD".into(),
            max_candidates: 32,
        },
        &ExecutionContext::for_tests(19),
    )
    .unwrap()
}

fn coordinates() -> Vec<ScientificQuantity> {
    candidates().iter().map(|c| c.inputs[0].clone()).collect()
}

/// A joint finite law over the four coordinates; `draws` is draw-major.
fn law_metadata(
    draws: usize,
    weights: Option<Vec<f64>>,
    supported: Option<Vec<bool>>,
) -> DistributionMetadata {
    let identity = DistributionIdentity::new(
        antecedent_io_core::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior,
        &coordinates(),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "finite-joint-law".into(),
            provider_id: "compiled-model".into(),
            rng_id: "no-sampling".into(),
            snapshot_id: "source".into(),
            causal_contract_id: "target".into(),
        },
    )
    .unwrap();
    DistributionMetadata {
        version: 1,
        identity,
        axes: ["draw".into(), "quantity".into()],
        shape: [draws, 4],
        weights,
        supported,
        calibration: DistributionCalibration::Exact,
        trust: DistributionTrust::Unverified,
        legacy_posterior: None,
        legacy_bindings: None,
    }
}

/// Two draws: reference 0 and 0; a 3 and 1; joint 8 and 8; out 5 and 5.
const DRAWS: [f64; 8] = [0., 3., 8., 5., 0., 1., 8., 5.];

fn value_of(rows: &[antecedent_design::program_search::ProgramSearchRow], id: &str) -> Option<f64> {
    rows.iter().find(|row| row.id == id).unwrap().value
}

#[test]
fn cells_01_draw_weights_of_the_retained_law_set_expected_utility_and_variance_exactly() {
    let ctx = ExecutionContext::for_tests(19);
    let weighted =
        DistributionArtifact::new(law_metadata(2, Some(vec![3., 1.]), None), DRAWS.to_vec())
            .unwrap();
    let mean = search(ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)))
        .execute(ProgramSearchSource::Law(&weighted), &ctx)
        .unwrap();
    // E[a] = (3 * 3 + 1 * 1) / 4 = 2.5, not the unweighted 2.0.
    assert_eq!(value_of(&mean.rows, "a"), Some(2.5));
    assert_eq!(value_of(&mean.rows, "reference"), Some(0.));
    assert_eq!(mean.best, vec!["a"]);
    let unweighted =
        DistributionArtifact::new(law_metadata(2, None, None), DRAWS.to_vec()).unwrap();
    let flat = search(ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)))
        .execute(ProgramSearchSource::Law(&unweighted), &ctx)
        .unwrap();
    assert_eq!(value_of(&flat.rows, "a"), Some(2.0));
    // Var[a] = 0.75 * 0.25 + 0.25 * 2.25 = 0.75 under the weights, 1 without them.
    let variance = search(ProgramObjective::Variance(UtilityExpr::Input(0)))
        .execute(ProgramSearchSource::Law(&weighted), &ctx)
        .unwrap();
    assert_eq!(value_of(&variance.rows, "a"), Some(0.75));
    assert_eq!(value_of(&variance.rows, "reference"), Some(0.));
    // A zero-weight draw contributes nothing: all mass on the first draw gives E[a] = 3, Var = 0.
    let point =
        DistributionArtifact::new(law_metadata(2, Some(vec![1., 0.]), None), DRAWS.to_vec())
            .unwrap();
    let on_first = search(ProgramObjective::Variance(UtilityExpr::Input(0)))
        .execute(ProgramSearchSource::Law(&point), &ctx)
        .unwrap();
    assert_eq!(value_of(&on_first.rows, "a"), Some(0.));
    let on_first = search(ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)))
        .execute(ProgramSearchSource::Law(&point), &ctx)
        .unwrap();
    assert_eq!(value_of(&on_first.rows, "a"), Some(3.));
    // Negative, non-finite and wrongly counted weights never reach the search.
    for bad in [vec![3., -1.], vec![f64::NAN, 1.], vec![1., 1., 1.], vec![0., 0.]] {
        assert!(
            DistributionArtifact::new(law_metadata(2, Some(bad), None), DRAWS.to_vec()).is_err()
        );
    }
}

#[test]
fn cells_01_masked_coordinates_and_non_finite_draws_are_refused_never_dropped() {
    let ctx = ExecutionContext::for_tests(19);
    // The feasible candidate `a` (coordinate 1) is masked as unsupported: the whole search is
    // refused with the typed cause, and no row carries a value.
    let masked = DistributionArtifact::new(
        law_metadata(2, None, Some(vec![true, false, true, true])),
        DRAWS.to_vec(),
    )
    .unwrap();
    let error = search(ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)))
        .execute(ProgramSearchSource::Law(&masked), &ctx)
        .expect_err("a masked response coordinate must refuse the search");
    assert!(
        matches!(&error, ProgramSearchError::Invalid(detail) if detail.contains("UnsupportedCoordinate")),
        "{error:?}"
    );
    // A missing (non-finite) draw cannot be carried by a law at all.
    for bad in [f64::NAN, f64::INFINITY] {
        let mut draws = DRAWS.to_vec();
        draws[1] = bad;
        assert!(DistributionArtifact::new(law_metadata(2, None, None), draws).is_err());
    }
    // A mean grid with a missing (non-finite) mean is refused as a source, never ranked.
    let grid = MeanSource {
        coordinates: coordinates(),
        means: vec![0., f64::NAN, 8., 5.],
        estimate: MeanEstimate::Frequentist,
        provider_id: "compiled-model".into(),
        snapshot_id: "source".into(),
        causal_contract_id: "target".into(),
        rng_id: "no-sampling".into(),
    };
    let outcome = search(ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)))
        .execute(ProgramSearchSource::Means(&grid), &ctx);
    assert!(
        matches!(&outcome, Err(ProgramSearchError::Invalid(detail)) if detail.contains("a mean source has one finite mean per coordinate")),
        "{outcome:?}"
    );
}

#[test]
fn cells_01_a_candidate_outside_its_control_bounds_is_excluded_and_never_valued() {
    let ctx = ExecutionContext::for_tests(19);
    let law = DistributionArtifact::new(law_metadata(2, None, None), DRAWS.to_vec()).unwrap();
    let result = search(ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)))
        .execute(ProgramSearchSource::Law(&law), &ctx)
        .unwrap();
    let out = result.rows.iter().find(|row| row.id == "out").unwrap();
    assert!(
        matches!(&out.feasibility, ProgramFeasibility::Operational(reason) if reason.as_str() == "control bounds"),
        "{:?}",
        out.feasibility
    );
    assert_eq!(out.value, None);
    let joint = result.rows.iter().find(|row| row.id == "joint").unwrap();
    assert!(
        matches!(&joint.feasibility, ProgramFeasibility::Operational(reason) if reason.as_str() == "budget")
    );
    assert_eq!(joint.value, None);
    // The high-valued excluded programs never win: 8 and 5 beat the feasible 2 but are not ranked.
    assert_eq!(result.best, vec!["a"]);
    assert_eq!(result.continuous_gap, None);
}
