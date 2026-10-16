//! Support labels and horizon draws at the native claim boundary.
//!
//! Hand examples.
//!
//! * A joint-draw response whose coordinates are labelled
//!   `[Supported, Extrapolative]` exports a support mask `[true, false]`: only a
//!   `Supported` coordinate is read as supported by the aligned law, the same
//!   reading the mean route gives it. Extrapolation is never promoted.
//! * A Bayesian temporal mean curve over two horizons is two separate
//!   conjugate fits, so its per-horizon draws are independent marginals. No
//!   aligned joint law may be exported from it: the claim either cannot be bound
//!   to a one-dimensional dose grid or exposes no joint law, and a quantile
//!   (nonlinear) functional is refused with `source_not_supplied`.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::{
    NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{ProgramBinding, QuantityRole, ScientificQuantity};
use antecedent_core::{
    AssumptionSet, CausalQuery, CausalResponse, CausalSchema, CausalSchemaBuilder,
    ContinuousDomain, CredibleDraws, ExecutionContext, GridSpec, IdentificationStatus,
    IntervalInterpretation, Lag, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, SupportRegion, SupportReport, SupportStatus,
    TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, SourceRepresentation,
    StructuralPolicy, UtilityExpr,
};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_io::distribution_artifact::DistributionCalibration;

fn schema() -> CausalSchema {
    CausalSchemaBuilder::new()
        .continuous("dose")
        .treatment()
        .continuous("y")
        .unit("mmHg")
        .outcome()
        .build()
        .unwrap()
}

fn context() -> NativeResponseContext {
    NativeResponseContext {
        program: ProgramBinding {
            graph_id: "graph:g".into(),
            contract_id: "contract:c".into(),
            treatment_id: "dose".into(),
            outcome_id: "y".into(),
            population_id: "target".into(),
            intervention_kind: "do".into(),
            horizon: 0,
            dose_grid: vec![1.0, 2.0],
            dose_units: "mg".into(),
            outcome_units: "mmHg".into(),
            functional_id: "mean".into(),
            transform_id: "identity".into(),
        },
        snapshot_id: "snap-1".into(),
        rng_id: "pcg:seed=7".into(),
        calibration: DistributionCalibration::Unmeasured,
    }
}

fn quantity(dose: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: format!("do(dose={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn quantile_contract() -> DecisionContract {
    let action = |id: &str, dose: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity(dose)],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![action("A", "1"), action("B", "2")],
        utility_units: "util".into(),
        criterion: DecisionCriterion::Quantile { p: 0.5 },
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn joint_response(labels: Vec<SupportStatus>) -> CausalResponse {
    let schema = schema();
    let grid = [1.0, 2.0];
    CausalResponse {
        estimand: ResponseFunctional::MeanCurve {
            outcome: schema.id_of("y").unwrap(),
            treatment: ContinuousDomain::new(
                schema.id_of("dose").unwrap(),
                GridSpec::Values(Arc::from(grid.to_vec())),
            ),
        },
        identification_status: IdentificationStatus::NonparametricallyIdentified,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid: Arc::from(grid.to_vec()),
            dimension: 1,
            mean: Arc::from(vec![3.0, 5.0]),
        }),
        uncertainty: ResponseUncertainty::PointwiseBand {
            level: 0.9,
            lower: Arc::from(vec![2.0, 4.0]),
            upper: Arc::from(vec![4.0, 6.0]),
            interpretation: IntervalInterpretation::Credible,
            draws: Some(CredibleDraws::columns(2, &[vec![2.0, 4.0], vec![4.0, 6.0]])),
        },
        support: SupportReport {
            status: SupportStatus::Extrapolative,
            query_region: SupportRegion {
                minima: Arc::from(vec![1.0]),
                maxima: Arc::from(vec![2.0]),
            },
            diagnostics: vec![],
            warnings: vec![],
            point_status: Some(Arc::from(labels)),
        },
        assumptions: AssumptionSet::new(),
        provenance_id: "op-1".into(),
        horizon_identification: None,
        interaction_structurally_zero: false,
    }
}

#[test]
fn an_extrapolative_coordinate_is_not_exported_as_supported() {
    let response = joint_response(vec![SupportStatus::Supported, SupportStatus::Extrapolative]);
    let claim = NativeResponseClaim::from_response(&response, &schema(), &context()).unwrap();
    let contract = quantile_contract();
    let requirement = contract.source_requirement().unwrap().unwrap();
    let input = claim.into_decision_source(&requirement).unwrap();
    assert_eq!(input.representation, SourceRepresentation::JointDraws);
    let NativeDecisionSource::JointLaw(artifact) = &input.source else {
        panic!("a quantile requirement is answered by an aligned joint law");
    };
    assert_eq!(artifact.metadata().supported.as_deref(), Some(&[true, false][..]));
}

// ---------------------------------------------------- multi-horizon Bayesian responses

fn series() -> TimeSeriesData {
    let n = 240usize;
    let treatment: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            _ => -1.0,
        })
        .collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| treatment[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| treatment[j])
        })
        .collect();
    TimeSeriesData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())], 1)
        .unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    graph
}

#[test]
fn multi_horizon_bayesian_responses_supply_no_aligned_joint_law() {
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(-1), None).unwrap());
    let study = Study::series(series())
        .graph(graph())
        .query(CausalQuery::Response(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(17)).unwrap();
    let response = result.response().expect("temporal response");

    // Separate per-horizon fits retain no draws that could be read as one law.
    if let ResponseUncertainty::PointwiseBand { draws, .. } = &response.uncertainty {
        assert!(draws.is_none(), "independent horizon draws must not be retained");
    }

    let contract = quantile_contract();
    let requirement = contract.source_requirement().unwrap().unwrap();
    match NativeResponseClaim::from_response(response, &schema(), &context()) {
        // The two-horizon grid is not the one-dimensional dose grid of the program.
        Err(refusal) => assert!(refusal.validate().is_ok()),
        Ok(claim) => {
            assert!(!claim.has_joint_law());
            let refusal = claim.decision_source(&requirement).unwrap_err();
            assert_eq!(refusal.code, "decision_contract_unsatisfied");
            assert_eq!(refusal.detail, "native_claims.source_not_supplied");
            assert_eq!(refusal.expected.as_deref(), Some("joint_draws,marginal_draws"));
            assert_eq!(refusal.supplied.as_deref(), Some("mean"));
        }
    }
}
