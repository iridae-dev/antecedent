//! Program binding: coordinates and regimes derived by hand from a declared dose grid.
//!
//! A program with the grid {1e-5, 1e-4, 0.125, 2.5, 100000, 1234567, 1e7} mg requests one
//! coordinate per dose. Each regime is `do(schema:dose=<g>)` where `<g>` is the six-significant-
//! digit `%g` rendering, derived here by hand: 1e-05, 0.0001, 0.125, 2.5, 100000, 1.23457e+06,
//! 1e+07. A grid entry differing from another only below the sixth digit still names the same
//! regime label but is a different point: its exact bits differ, so a claim for one is refused
//! against the other with `dose_grid_changed`, and the durable identities differ.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    CheckedCausalContract, DistributionMeaning, ExternalCapability, ExternalProgramClaim,
    ExternalResponse, ExternalResult, ExternalResultHeader, ExternalScientificObject,
    ExternalTrustState, ExternalUncertaintyMeaning, LawProviderContract, ProgramBinding,
    ProviderObjectIdentity, QuantityRole, bind_external_result_to_program,
    check_external_against_program, dose_label,
};
use antecedent_core::IdentificationStatus;

fn program(grid: Vec<f64>) -> ProgramBinding {
    ProgramBinding {
        graph_id: "graph:g1".into(),
        contract_id: "contract:c1".into(),
        treatment_id: "schema:dose".into(),
        outcome_id: "schema:y".into(),
        population_id: "target".into(),
        intervention_kind: "do".into(),
        horizon: 0,
        dose_grid: grid,
        dose_units: "mg".into(),
        outcome_units: "mmHg".into(),
        functional_id: "mean".into(),
        transform_id: "identity".into(),
    }
}

#[test]
fn the_requested_coordinates_are_the_hand_derived_g_format_regimes_in_grid_order() {
    let grid = vec![1e-5, 1e-4, 0.125, 2.5, 100_000.0, 1_234_567.0, 1e7];
    let p = program(grid);
    let want = [
        "do(schema:dose=1e-05)",
        "do(schema:dose=0.0001)",
        "do(schema:dose=0.125)",
        "do(schema:dose=2.5)",
        "do(schema:dose=100000)",
        "do(schema:dose=1.23457e+06)",
        "do(schema:dose=1e+07)",
    ];
    let quantities = p.expected_quantities();
    assert_eq!(quantities.len(), want.len());
    for (q, regime) in quantities.iter().zip(want) {
        assert_eq!(q.regime_id, regime);
        assert_eq!(q.variable_id, "schema:y");
        assert_eq!(q.role, QuantityRole::Outcome);
        assert_eq!(q.units, "mmHg");
        assert_eq!(q.population_id, "target");
        assert_eq!(q.horizon, 0);
        assert_eq!(q.functional_id, "mean");
        assert_eq!(q.transform_id, "identity");
    }
    // The faithful claim passes with exactly seven coordinates.
    let claim = ExternalProgramClaim::declared_by(&p);
    assert_eq!(check_external_against_program(&p, &claim).unwrap().coordinates, 7);
    assert_eq!(dose_label(-0.75), "-0.75");
    assert_eq!(dose_label(0.0), "0");
}

#[test]
fn a_dose_one_ulp_off_shares_the_label_but_is_a_different_program() {
    let near = 0.1 + 0.2; // 0.30000000000000004
    let a = program(vec![near]);
    let b = program(vec![0.3]);
    // Six significant digits cannot tell them apart.
    assert_eq!(dose_label(near), "0.3");
    assert_eq!(a.expected_quantities()[0].regime_id, b.expected_quantities()[0].regime_id);
    assert_eq!(a.expected_quantities()[0].regime_id, "do(schema:dose=0.3)");
    // The exact bits are what the program and its identity cover.
    assert_ne!(a.identity(), b.identity());
    let claim_for_a = ExternalProgramClaim::declared_by(&a);
    let refusal = check_external_against_program(&b, &claim_for_a).unwrap_err();
    assert_eq!(refusal.detail, "program_binding.dose_grid_changed");
    assert_eq!(refusal.offending.as_deref(), Some("doses"));
    assert_eq!(refusal.expected.as_deref(), Some("0.3"));
    assert_eq!(refusal.supplied.as_deref(), Some("0.3"));
}

fn contract_for(p: &ProgramBinding) -> CheckedCausalContract {
    CheckedCausalContract {
        graph_id: p.graph_id.clone(),
        identification: IdentificationStatus::NonparametricallyIdentified,
        estimand: p.expected_quantities(),
        accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
        required_evidence_ids: vec![],
        required_assumption_ids: vec![],
        equivalences: vec![],
    }
}

fn response_for(p: &ProgramBinding, values: Vec<f64>) -> ExternalResult {
    let quantities = p.expected_quantities();
    ExternalResult::Response(ExternalResponse {
        header: ExternalResultHeader {
            object: ExternalScientificObject::Law(LawProviderContract {
                identity: ProviderObjectIdentity {
                    provider_id: "lab".into(),
                    object_id: "curve".into(),
                    version_id: "v1".into(),
                    snapshot_id: "snap".into(),
                    request_id: "req".into(),
                },
                quantities: quantities.clone(),
                meaning: DistributionMeaning::InterventionalPredictive,
                capabilities: vec![ExternalCapability::Sample, ExternalCapability::Mean],
            }),
            graph_id: p.graph_id.clone(),
            quantities,
            evidence_ids: vec![],
            assumption_ids: vec![],
            trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
        },
        values,
        uncertainty: ExternalUncertaintyMeaning::None,
        point_support: None,
    })
}

/// A result bound to the program keeps one value per coordinate in grid order: the value at
/// dose 2.5 is the fourth entry, so the response `[8, 6, 4, 2]` over `{0.5, 1, 1.5, 2.5}`
/// reads 2 at the largest dose.
#[test]
fn a_bound_result_carries_one_value_per_hand_listed_coordinate() {
    let p = program(vec![0.5, 1.0, 1.5, 2.5]);
    let claim = ExternalProgramClaim::declared_by(&p);
    let bound = bind_external_result_to_program(
        &p,
        &claim,
        &contract_for(&p),
        &response_for(&p, vec![8.0, 6.0, 4.0, 2.0]),
    )
    .unwrap();
    let values = bound.claim().values().unwrap();
    let regimes: Vec<String> =
        p.expected_quantities().into_iter().map(|q| q.regime_id).collect();
    assert_eq!(
        regimes,
        ["do(schema:dose=0.5)", "do(schema:dose=1)", "do(schema:dose=1.5)", "do(schema:dose=2.5)"]
    );
    let at = |regime: &str| values[regimes.iter().position(|r| r == regime).unwrap()];
    assert_eq!(at("do(schema:dose=2.5)"), 2.0);
    assert_eq!(at("do(schema:dose=0.5)"), 8.0);
    assert!(!bound.claim().is_native_estimation());
}
