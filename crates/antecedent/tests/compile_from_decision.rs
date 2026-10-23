//! Compile that starts from a decision problem or a value-of-information request.
//!
//! The lowered program's layer four carries the decision. A carrier the statistical
//! realization cannot publish is refused with a typed refusal naming the next
//! attempt. An EVSI request quotes its Monte Carlo store and refuses before
//! allocation when the admission pool cannot hold it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::design_ranking::{
    DesignRankingError, DesignRankingRequestWire, compile_for_study, evaluate_for_study,
};
use antecedent::analysis::{
    BAYESIAN_REALIZATION, DISTRIBUTIONAL_CARRIER, DecisionCarrierError, DecisionCompileRequest,
    MissingCarrier,
};
use antecedent::planner::PhysicalExecutionPlan;
use antecedent::planner::lowered::CarrierDemand;
use antecedent::{
    BayesianConfig, CausalError, CompileCause, Dag, InferenceMode, RefuteSuite, Study,
};
use antecedent_contract::{EvidencePort, QuantityRole, ScientificQuantity};
use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, SemanticDigest};
use antecedent_data::TabularData;
use antecedent_design::{AffineUtility, DecisionProblem, DesignError, Utility};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde_json::json;

const TREATMENT: [f64; 8] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
const OUTCOME: [f64; 8] = [1.1, 3.0, 0.9, 3.1, 1.0, 2.9, 1.0, 3.0];

fn study(bayesian: bool) -> Study {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let columns: [(&str, &[f64]); 2] = [("t", &TREATMENT), ("y", &OUTCOME)];
    let data = TabularData::try_from_schema_f64(schema.clone(), columns).unwrap();
    let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let mut builder = Study::tabular(data).graph(graph).query(query);
    if bayesian {
        builder = builder.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(256).prior_scale(1.0e6),
        ));
    }
    builder.refute(RefuteSuite::None).bootstrap_replicates(0).build().unwrap()
}

/// Act (`-0.5 + slope * effect`) or wait (zero).
fn affine_problem(slope: f64) -> DecisionProblem<usize, f64> {
    let utility = AffineUtility::new(vec![0.0, -0.5], vec![0.0, slope]).unwrap();
    DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![])
}

/// A utility that declares no affine form.
struct Quadratic;

impl Utility<usize, f64> for Quadratic {
    fn evaluate_batch(
        &self,
        actions: &[usize],
        outcomes: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        for (row, &action) in actions.iter().enumerate() {
            for (col, &state) in outcomes.iter().enumerate() {
                out[row * outcomes.len() + col] = action as f64 * state * state;
            }
        }
        Ok(())
    }
}

fn request(carrier: CarrierDemand) -> DecisionCompileRequest {
    DecisionCompileRequest::new(carrier)
        .with_contract_identity("launch-decision")
        .with_utility_unit("net_margin")
}

fn decision_identity(plan: &PhysicalExecutionPlan) -> SemanticDigest {
    plan.lowered().expect("a compiled plan carries a lowered program").identity()
}

fn carrier_refusal(error: &CausalError) -> DecisionCarrierError {
    let CausalError::Rejected(failure) = error.peeled() else {
        panic!("expected a typed compile refusal, got {error}");
    };
    let CompileCause::Decision(refusal) = failure.cause() else {
        panic!("expected a decision-carrier refusal, got {error}");
    };
    *refusal
}

fn ranking_wire(candidates: &[&str]) -> DesignRankingRequestWire {
    let quantity = ScientificQuantityWire::from(&ScientificQuantity {
        variable_id: "theta".into(),
        variable_name: "theta".into(),
        role: QuantityRole::Outcome,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "decision-state".into(),
        horizon: 0,
        functional_id: "state".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    });
    let candidates: Vec<_> = candidates
        .iter()
        .map(|id| {
            json!({
                "semantic_id": id,
                "sample_size": 200,
                "cost": {"amount": 1.0, "unit": "utility"},
                "signal": {
                    "prior_id": "decision-prior",
                    "state_quantity": quantity,
                    "observation_quantity": quantity,
                    "rng_seed": 3,
                    "evidence_lineage": ["pilot"],
                    "conditional_independence": "iid_given_state"
                },
                "provider": {"kind": "binomial"}
            })
        })
        .collect();
    serde_json::from_value(json!({
        "decision": {
            "contract_identity": "launch-ranking",
            "utility_unit": "utility",
            "action_ids": ["wait", "act"],
            "intercepts": [0.0, -0.5],
            "slopes": [0.0, 1.0],
            "prior": {"kind": "draws", "states": [0.25, 0.75]}
        },
        "candidates": candidates,
        "prior_observation_ids": ["pilot-b", "pilot-a"],
        "rng_seed": 3,
        "source_digests": ["pilot-prior"]
    }))
    .unwrap()
}

#[test]
fn bayesian_cell_compiles_from_a_decision_with_the_posterior_carrier() {
    let ctx = ExecutionContext::for_tests(1);
    let study = study(true);
    let plan = study
        .compile_for_decision(
            &affine_problem(2.0),
            &request(CarrierDemand::ParameterPosterior),
            &ctx,
        )
        .unwrap();
    let lowered = plan.lowered().unwrap();
    let decision = lowered.evidence().decision().expect("layer four carries the decision");
    assert_eq!(decision.carrier(), CarrierDemand::ParameterPosterior);
    assert_eq!(decision.actions(), 2);
    assert_eq!(decision.constraints(), 0);
    assert_eq!(decision.chance_threshold().to_bits(), 1.0_f64.to_bits());
    assert_eq!(decision.utility_unit(), Some("net_margin"));
    assert_eq!(decision.contract_identity(), Some("launch-decision"));
    assert_eq!(decision.utility().coefficients(), Some(&[(0.0, 0.0), (-0.5, 2.0)][..]));
    assert!(decision.information_value().is_none());

    // A decision changes layer four, so the program identity differs from the estimate's.
    let estimate = study.compile(&ctx).unwrap();
    assert!(estimate.lowered().unwrap().evidence().decision().is_none());
    assert_ne!(decision_identity(&estimate), lowered.identity());

    // The plan still executes: dispatch reads the same route cell.
    let result = study.execute(&plan, &ctx).unwrap();
    assert_eq!(result.lowered_program_identity(), Some(lowered.identity()));
}

#[test]
fn frequentist_cell_refuses_a_parameter_posterior_and_names_the_next_attempt() {
    let ctx = ExecutionContext::for_tests(1);
    let study = study(false);
    let error = study
        .compile_for_decision(
            &affine_problem(2.0),
            &request(CarrierDemand::ParameterPosterior),
            &ctx,
        )
        .unwrap_err();
    assert_eq!(error.reason_code(), Some("distribution_meaning_mismatch"));
    assert!(error.to_string().contains("next attempt:"), "{error}");
    let refusal = carrier_refusal(&error);
    assert_eq!(refusal.demanded, CarrierDemand::ParameterPosterior);
    assert_eq!(refusal.missing_carrier(), Some(MissingCarrier::ParameterPosterior));
    assert_eq!(refusal.realization, "frequentist");
    assert_eq!(refusal.next_attempt, BAYESIAN_REALIZATION);
    let port = refusal.port.expect("a posterior demand is an evidence-port request");
    assert_eq!(port.source_port, EvidencePort::ConfidenceRegion);
    assert_eq!(port.legal_ports, &[EvidencePort::ParameterPosterior]);

    // The weaker carrier an affine utility needs is supplied.
    let mean = study
        .compile_for_decision(&affine_problem(2.0), &request(CarrierDemand::Mean), &ctx)
        .unwrap();
    assert_eq!(
        mean.lowered().unwrap().evidence().decision().unwrap().carrier(),
        CarrierDemand::Mean
    );

    // A mean does not carry a utility that declares no affine form.
    let opaque: DecisionProblem<usize, f64> =
        DecisionProblem::new(vec![0, 1], Arc::new(Quadratic), vec![]);
    let error =
        study.compile_for_decision(&opaque, &request(CarrierDemand::Mean), &ctx).unwrap_err();
    assert_eq!(error.reason_code(), Some("decision_contract_unsatisfied"));
    assert_eq!(carrier_refusal(&error).next_attempt, DISTRIBUTIONAL_CARRIER);

    // An outcome law is refused on both realizations of an average-effect route.
    for bayesian in [false, true] {
        let error = self::study(bayesian)
            .compile_for_decision(&affine_problem(2.0), &request(CarrierDemand::OutcomeLaw), &ctx)
            .unwrap_err();
        assert_eq!(carrier_refusal(&error).missing_carrier(), Some(MissingCarrier::OutcomeLaw));
    }
}

#[test]
fn same_decision_keeps_its_identity_and_a_new_utility_changes_it() {
    let ctx = ExecutionContext::for_tests(1);
    let study = study(true);
    let carrier = request(CarrierDemand::ParameterPosterior);
    let first = study.compile_for_decision(&affine_problem(2.0), &carrier, &ctx).unwrap();
    let again = study.compile_for_decision(&affine_problem(2.0), &carrier, &ctx).unwrap();
    assert_eq!(decision_identity(&first), decision_identity(&again));

    let new_utility = study.compile_for_decision(&affine_problem(3.0), &carrier, &ctx).unwrap();
    assert_ne!(decision_identity(&first), decision_identity(&new_utility));

    let new_unit = study
        .compile_for_decision(
            &affine_problem(2.0),
            &carrier.clone().with_utility_unit("revenue"),
            &ctx,
        )
        .unwrap();
    assert_ne!(decision_identity(&first), decision_identity(&new_unit));

    // A utility change does not change the statistical program.
    assert_eq!(
        first.lowered().unwrap().statistical().identity(),
        new_utility.lowered().unwrap().statistical().identity()
    );
}

#[test]
fn compile_from_an_evsi_request_fills_the_value_of_information_program() {
    let ctx = ExecutionContext::for_tests(1);
    let study = study(true);
    let wire = ranking_wire(&["study-b", "study-a"]);
    let plan = compile_for_study(&study, &wire, &ctx).unwrap();
    let decision = plan.lowered().unwrap().evidence().decision().unwrap();
    assert_eq!(decision.carrier(), CarrierDemand::ParameterPosterior);
    assert_eq!(decision.contract_identity(), Some("launch-ranking"));
    assert_eq!(decision.utility_unit(), Some("utility"));
    let evsi = decision.information_value().expect("an EVSI compile carries the request");
    let candidates: Vec<&str> = evsi.candidates().iter().map(|id| &**id).collect();
    assert_eq!(candidates, ["study-a", "study-b"]);
    let prior: Vec<&str> = evsi.prior_observation_ids().iter().map(|id| &**id).collect();
    assert_eq!(prior, ["pilot-a", "pilot-b"]);
    assert_eq!(evsi.rng_seed(), 3);
    assert!(evsi.mc_error_tolerance() > 0.0);
    let schedule = evsi.rank();
    let draws = u64::from(schedule.max_batches().max(schedule.min_batches().max(1)))
        * u64::from(schedule.batch_size());
    let store = evsi.monte_carlo();
    assert_eq!(store.candidates(), 2);
    assert_eq!(store.draws(), draws);
    assert_eq!(store.tasks(), 2 * draws);
    assert_eq!(store.total_bytes(), 2 * draws * 8 + draws * 8);

    // Declaration order does not change the compiled program.
    let reordered =
        compile_for_study(&study, &ranking_wire(&["study-a", "study-b"]), &ctx).unwrap();
    assert_eq!(decision_identity(&plan), decision_identity(&reordered));

    // The evaluation is bound to that program.
    let evaluated = evaluate_for_study(&study, &wire, &ctx).unwrap();
    assert_eq!(evaluated.lowered_program_identity(), Some(decision_identity(&plan)));

    // A sampling realization is not a state belief.
    let refused = compile_for_study(&self::study(false), &wire, &ctx).unwrap_err();
    let DesignRankingError::Compile(error) = &refused else {
        panic!("expected a compile refusal, got {refused}");
    };
    assert_eq!(carrier_refusal(error).next_attempt, BAYESIAN_REALIZATION);
    let external = refused.refusal().expect("a carrier refusal is structured");
    assert_eq!(external.code, "distribution_meaning_mismatch");
    assert_eq!(external.remedy, Some(BAYESIAN_REALIZATION.sentence()));
}

#[test]
fn evsi_refuses_before_allocation_when_the_pool_cannot_hold_the_store() {
    let ctx = ExecutionContext::for_tests(1).with_memory_limit(Some(64));
    let study = study(true);
    let refused = compile_for_study(&study, &ranking_wire(&["study-a"]), &ctx).unwrap_err();
    let DesignRankingError::Compile(error) = &refused else {
        panic!("expected a compile refusal, got {refused}");
    };
    assert!(matches!(error, CausalError::Resource { .. }), "{error}");
    assert!(error.to_string().contains("evsi Monte Carlo store"), "{error}");
    assert!(error.to_string().contains("next attempt:"), "{error}");
    // Nothing was leased from the pool.
    assert_eq!(ctx.admission().available().bytes(), 64);
}
