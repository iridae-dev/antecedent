//! C1: an external claim is bound to the actual identified program and request.
//!
//! Every refusal is asserted by its exact registered code and detail; every
//! expected value is derived by hand from the declared program below.

use antecedent_core::{
    CheckedCausalContract, DistributionMeaning, ExternalCapability, ExternalProgramClaim,
    ExternalRefusal, ExternalResponse, ExternalResult, ExternalResultHeader,
    ExternalScientificObject, ExternalTrustState, ExternalUncertaintyMeaning, IdentificationStatus,
    LawProviderContract, ProgramBinding, ProviderObjectIdentity, bind_external_result_to_program,
    check_external_against_program, dose_label,
};

fn binding() -> ProgramBinding {
    ProgramBinding {
        graph_id: "graph:g1".into(),
        contract_id: "contract:c1".into(),
        treatment_id: "schema:dose".into(),
        outcome_id: "schema:y".into(),
        population_id: "target".into(),
        intervention_kind: "do".into(),
        horizon: 0,
        dose_grid: vec![0.5, 1.0, 2.0],
        dose_units: "mg".into(),
        outcome_units: "mmHg".into(),
        functional_id: "mean".into(),
        transform_id: "identity".into(),
    }
}

fn detail(refusal: &ExternalRefusal) -> (&'static str, &str, Option<&str>) {
    (refusal.code, refusal.detail.as_str(), refusal.offending.as_deref())
}

fn refused(mutate: impl FnOnce(&mut ExternalProgramClaim)) -> ExternalRefusal {
    let program = binding();
    let mut claim = ExternalProgramClaim::declared_by(&program);
    mutate(&mut claim);
    check_external_against_program(&program, &claim).unwrap_err()
}

#[test]
fn c1_binding_faithful_claim_passes_with_derived_coordinates() {
    let program = binding();
    let claim = ExternalProgramClaim::declared_by(&program);
    let regimes: Vec<&str> = claim.quantities.iter().map(|q| q.regime_id.as_str()).collect();
    assert_eq!(regimes, ["do(schema:dose=0.5)", "do(schema:dose=1)", "do(schema:dose=2)"]);
    let checked = check_external_against_program(&program, &claim).unwrap();
    assert_eq!(checked.coordinates, 3);
    assert_eq!(checked.identity, program.identity());
    assert_eq!(program.identity().len(), 64);
    assert_eq!(dose_label(2.0), "2");
}

#[test]
fn c1_binding_quantities_override_is_refused_with_its_coordinate() {
    let units = refused(|c| c.quantities[1].units = "kPa".into());
    assert_eq!(
        detail(&units),
        (
            "quantity_semantics_mismatch",
            "program_binding.quantities_override_mismatch",
            Some("coordinate[1]")
        )
    );
    let horizon = refused(|c| c.quantities[2].horizon = 3);
    assert_eq!(horizon.detail, "program_binding.quantities_override_mismatch");
    let functional = refused(|c| c.quantities[0].functional_id = "risk".into());
    assert_eq!(functional.detail, "program_binding.quantities_override_mismatch");
    let transform = refused(|c| c.quantities[0].transform_id = "log".into());
    assert_eq!(transform.detail, "program_binding.quantities_override_mismatch");
}

#[test]
fn c1_binding_treatment_and_outcome_substitution_is_refused() {
    let outcome = refused(|c| c.outcome_id = "schema:z".into());
    assert_eq!(
        detail(&outcome),
        (
            "external_binding_mismatch",
            "program_binding.treatment_outcome_substitution",
            Some("outcome_id")
        )
    );
    assert_eq!(outcome.expected.as_deref(), Some("schema:y"));
    assert_eq!(outcome.supplied.as_deref(), Some("schema:z"));
    let treatment = refused(|c| c.treatment_id = "schema:other".into());
    assert_eq!(detail(&treatment).1, "program_binding.treatment_outcome_substitution");
    assert_eq!(treatment.offending.as_deref(), Some("treatment_id"));
    // Substituting the variable inside a coordinate is the same failure.
    let variable = refused(|c| c.quantities[0].variable_id = "schema:z".into());
    assert_eq!(detail(&variable).1, "program_binding.treatment_outcome_substitution");
    assert_eq!(variable.offending.as_deref(), Some("coordinate[0]"));
    // So is intervening on a different treatment in a coordinate's regime.
    let regime = refused(|c| c.quantities[1].regime_id = "do(schema:other=1)".into());
    assert_eq!(detail(&regime).1, "program_binding.treatment_outcome_substitution");
}

#[test]
fn c1_binding_population_mismatch_is_refused() {
    let declared = refused(|c| c.population_id = "source".into());
    assert_eq!(
        detail(&declared),
        (
            "quantity_semantics_mismatch",
            "program_binding.population_mismatch",
            Some("population_id")
        )
    );
    assert_eq!(declared.expected.as_deref(), Some("target"));
    let coordinate = refused(|c| c.quantities[2].population_id = "source".into());
    assert_eq!(
        detail(&coordinate),
        (
            "quantity_semantics_mismatch",
            "program_binding.population_mismatch",
            Some("coordinate[2]")
        )
    );
}

#[test]
fn c1_binding_dose_grid_change_is_refused() {
    let doses = refused(|c| c.doses[1] = 1.5);
    assert_eq!(
        detail(&doses),
        ("quantity_semantics_mismatch", "program_binding.dose_grid_changed", Some("doses"))
    );
    assert_eq!(doses.expected.as_deref(), Some("0.5,1,2"));
    assert_eq!(doses.supplied.as_deref(), Some("0.5,1.5,2"));
    let units = refused(|c| c.dose_units = "g".into());
    assert_eq!(units.detail, "program_binding.dose_grid_changed");
    assert_eq!(units.offending.as_deref(), Some("dose_units"));
    let count = refused(|c| {
        c.quantities.pop();
    });
    assert_eq!(count.detail, "program_binding.dose_grid_changed");
    assert_eq!((count.expected.as_deref(), count.supplied.as_deref()), (Some("3"), Some("2")));
    let regime = refused(|c| c.quantities[1].regime_id = "do(schema:dose=3)".into());
    assert_eq!(detail(&regime).1, "program_binding.dose_grid_changed");
    assert_eq!(regime.expected.as_deref(), Some("do(schema:dose=1)"));
}

#[test]
fn c1_binding_contract_identity_mismatch_is_refused() {
    let contract = refused(|c| c.contract_id = "contract:other".into());
    assert_eq!(
        detail(&contract),
        (
            "external_binding_mismatch",
            "program_binding.contract_identity_mismatch",
            Some("contract_id")
        )
    );
    let graph = refused(|c| c.graph_id = "graph:g2".into());
    assert_eq!(detail(&graph).1, "program_binding.contract_identity_mismatch");
    assert_eq!(graph.offending.as_deref(), Some("graph_id"));
    // A well-formed identity of a different program does not certify this one.
    let mut other = binding();
    other.dose_grid = vec![0.5, 1.0, 4.0];
    let stale = refused(|c| c.declared_identity = other.identity());
    assert_eq!(
        detail(&stale),
        (
            "external_binding_mismatch",
            "program_binding.contract_identity_mismatch",
            Some("declared_identity")
        )
    );
    assert_eq!(stale.expected, Some(binding().identity()));
}

#[test]
fn c1_binding_graph_only_identity_is_insufficient() {
    let program = binding();
    for declared in [
        ProgramBinding::graph_only_identity(&program.graph_id),
        program.graph_id.clone(),
        "graph:00ff".to_owned(),
        "contract:g1".to_owned(),
    ] {
        let refusal = refused(|c| c.declared_identity = declared.clone());
        assert_eq!(
            detail(&refusal),
            (
                "external_binding_mismatch",
                "program_binding.graph_only_identity",
                Some("declared_identity")
            ),
            "{declared}"
        );
        assert_eq!(refusal.expected, Some(program.identity()));
    }
    // Two different questions on one graph share the graph-only identity but not ours.
    let mut other = binding();
    other.outcome_id = "schema:z".into();
    assert_ne!(program.identity(), other.identity());
    assert_eq!(
        ProgramBinding::graph_only_identity(&program.graph_id),
        ProgramBinding::graph_only_identity(&other.graph_id)
    );
}

#[test]
fn c1_binding_identity_changes_when_any_one_field_changes() {
    let base = binding().identity();
    let mut seen = vec![base.clone()];
    for field in 0..13 {
        let mut b = binding();
        match field {
            0 => b.graph_id = "graph:g9".into(),
            1 => b.contract_id = "contract:c9".into(),
            2 => b.treatment_id = "schema:t2".into(),
            3 => b.outcome_id = "schema:y2".into(),
            4 => b.population_id = "source".into(),
            5 => b.intervention_kind = "shift".into(),
            6 => b.horizon = 1,
            7 => b.dose_grid[2] = 2.25,
            8 => b.dose_grid.push(4.0),
            9 => b.dose_units = "g".into(),
            10 => b.outcome_units = "kPa".into(),
            11 => b.functional_id = "risk".into(),
            _ => b.transform_id = "log".into(),
        }
        let identity = b.identity();
        assert!(!seen.contains(&identity), "field {field} did not change the identity");
        seen.push(identity);
    }
    assert_eq!(binding().identity(), base);
}

#[test]
fn c1_binding_invalid_binding_is_refused_before_any_comparison() {
    let mut program = binding();
    program.dose_grid = vec![1.0, 1.0];
    let claim = ExternalProgramClaim::declared_by(&program);
    let refusal = check_external_against_program(&program, &claim).unwrap_err();
    assert_eq!(
        detail(&refusal),
        ("invalid_argument", "program_binding.invalid_binding", Some("dose_grid"))
    );
    program.dose_grid = vec![1.0];
    program.population_id = " ".into();
    let refusal = program.validate().unwrap_err();
    assert_eq!(refusal.offending.as_deref(), Some("population_id"));
    assert!(refusal.validate().is_ok());
}

fn contract_for(program: &ProgramBinding) -> CheckedCausalContract {
    CheckedCausalContract {
        graph_id: program.graph_id.clone(),
        identification: IdentificationStatus::NonparametricallyIdentified,
        estimand: program.expected_quantities(),
        accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
        required_evidence_ids: vec![],
        required_assumption_ids: vec![],
        equivalences: vec![],
    }
}

fn response_for(program: &ProgramBinding) -> ExternalResult {
    let quantities = program.expected_quantities();
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
            graph_id: program.graph_id.clone(),
            quantities,
            evidence_ids: vec![],
            assumption_ids: vec![],
            trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
        },
        values: vec![10.0, 12.0, 15.0],
        uncertainty: ExternalUncertaintyMeaning::None,
        point_support: None,
    })
}

#[test]
fn c1_binding_binds_a_result_to_the_program_and_stays_external() {
    let program = binding();
    let claim = ExternalProgramClaim::declared_by(&program);
    let bound = bind_external_result_to_program(
        &program,
        &claim,
        &contract_for(&program),
        &response_for(&program),
    )
    .unwrap();
    assert_eq!(bound.program_identity(), program.identity());
    assert!(!bound.claim().is_native_estimation());
    assert_eq!(bound.claim().values(), Some(&[10.0, 12.0, 15.0][..]));
}

#[test]
fn c1_binding_refuses_a_contract_that_describes_another_request() {
    let program = binding();
    let claim = ExternalProgramClaim::declared_by(&program);
    let result = response_for(&program);
    let mut contract = contract_for(&program);
    contract.estimand[0].units = "kPa".into();
    let refusal =
        bind_external_result_to_program(&program, &claim, &contract, &result).unwrap_err();
    assert_eq!(
        detail(&refusal),
        (
            "quantity_semantics_mismatch",
            "program_binding.quantities_override_mismatch",
            Some("coordinate[0]")
        )
    );
    let mut contract = contract_for(&program);
    contract.graph_id = "graph:other".into();
    let refusal =
        bind_external_result_to_program(&program, &claim, &contract, &result).unwrap_err();
    assert_eq!(refusal.detail, "program_binding.contract_identity_mismatch");
    let mut contract = contract_for(&program);
    contract.estimand.pop();
    let refusal =
        bind_external_result_to_program(&program, &claim, &contract, &result).unwrap_err();
    assert_eq!(refusal.detail, "program_binding.dose_grid_changed");
    // A program refusal comes before any contract or result check.
    let mut wrong = ExternalProgramClaim::declared_by(&program);
    wrong.population_id = "source".into();
    let refusal =
        bind_external_result_to_program(&program, &wrong, &contract_for(&program), &result)
            .unwrap_err();
    assert_eq!(refusal.detail, "program_binding.population_mismatch");
}
