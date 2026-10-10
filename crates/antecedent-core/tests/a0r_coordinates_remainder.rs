//! A0 remainder: a native response names its own coordinates, or refuses with a
//! typed `coordinate_support.*` detail. Expected coordinates are derived by hand.

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalResponse, ContinuousDomain, ExternalRefusal, GridSpec,
    HorizonIdentification, IdentificationStatus, Intervention, InterventionSequence,
    MechanismOverride, ProgramBinding, QuantityRole, ResponseCoordinateLabels, ResponseFunctional,
    ResponseIdentification, ResponseUncertainty, ResponseValue, SequencedIntervention,
    StochasticPolicy, SupportRegion, SupportReport, SupportStatus, TemporalPolicy, Value,
    VariableId, response_coordinates,
};

const A: u32 = 0;
const B: u32 = 1;
const Y: u32 = 2;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn name_of(id: VariableId) -> Option<String> {
    match id.raw() {
        A => Some("a".to_owned()),
        B => Some("b".to_owned()),
        Y => Some("y".to_owned()),
        _ => None,
    }
}

fn labels() -> ResponseCoordinateLabels<'static> {
    ResponseCoordinateLabels {
        outcome_units: "mmHg",
        population_id: "target",
        transform_id: "identity",
    }
}

fn response(
    estimand: ResponseFunctional,
    estimate: ResponseValue,
    horizons: Option<Vec<u32>>,
) -> CausalResponse {
    CausalResponse {
        estimand,
        identification_status: IdentificationStatus::NonparametricallyIdentified,
        estimate: ResponseIdentification::PointIdentified(estimate),
        uncertainty: ResponseUncertainty::None,
        support: SupportReport {
            status: SupportStatus::Supported,
            query_region: SupportRegion { minima: Arc::from([0.0]), maxima: Arc::from([1.0]) },
            diagnostics: Vec::new(),
            warnings: Vec::new(),
            point_status: None,
        },
        assumptions: AssumptionSet::new(),
        provenance_id: Arc::from("a0r.test"),
        horizon_identification: horizons.map(|steps| {
            steps
                .into_iter()
                .map(|horizon| HorizonIdentification {
                    horizon,
                    status: IdentificationStatus::NonparametricallyIdentified,
                    method: Arc::from("temporal.backdoor.unfolded"),
                    adjustment: Arc::from([]),
                })
                .collect()
        }),
        interaction_structurally_zero: false,
    }
}

fn curve(doses: &[f64]) -> ResponseFunctional {
    ResponseFunctional::MeanCurve {
        outcome: v(Y),
        treatment: ContinuousDomain::new(v(A), GridSpec::Values(Arc::from(doses.to_vec()))),
    }
}

fn surface(len: usize) -> ResponseValue {
    ResponseValue::Surface {
        grid: Arc::from(vec![0.0; len]),
        dimension: 1,
        mean: Arc::from(vec![0.0; len]),
    }
}

fn intervention(interventions: Vec<Intervention>) -> CausalResponse {
    response(
        ResponseFunctional::InterventionResponse {
            outcome: v(Y),
            interventions: Arc::from(interventions),
        },
        ResponseValue::Scalar(1.0),
        None,
    )
}

fn refusal_of(response: &CausalResponse) -> ExternalRefusal {
    let refusal = response_coordinates(response, &name_of, &labels()).unwrap_err();
    // Every refusal is a well-formed, registered `<namespace>.<snake_case>` envelope.
    assert_eq!(refusal.validate(), Ok(()), "{refusal:?}");
    assert!(refusal.detail.starts_with("coordinate_support."), "{}", refusal.detail);
    refusal
}

#[test]
fn a0r_static_curve_coordinates_are_hand_derived_and_match_the_program_binding() {
    let doses = [0.0, 0.5, 2.0];
    let derived =
        response_coordinates(&response(curve(&doses), surface(3), None), &name_of, &labels())
            .unwrap();
    let regimes: Vec<_> = derived.iter().map(|q| q.regime_id.as_str()).collect();
    assert_eq!(regimes, ["do(a=0)", "do(a=0.5)", "do(a=2)"]);
    for quantity in &derived {
        assert_eq!(quantity.variable_id, "y");
        assert_eq!(quantity.variable_name, "y");
        assert_eq!(quantity.role, QuantityRole::Outcome);
        assert_eq!(quantity.units, "mmHg");
        assert_eq!(quantity.population_id, "target");
        assert_eq!((quantity.horizon, quantity.functional_id.as_str()), (0, "mean"));
        assert_eq!(quantity.transform_id, "identity");
        assert!(quantity.conditioning.is_empty());
    }
    // The C1 program binding derives the same coordinates for the same request.
    let binding = ProgramBinding {
        graph_id: "graph:g".into(),
        contract_id: "contract:c".into(),
        treatment_id: "a".into(),
        outcome_id: "y".into(),
        population_id: "target".into(),
        intervention_kind: "do".into(),
        horizon: 0,
        dose_grid: doses.to_vec(),
        dose_units: "mg".into(),
        outcome_units: "mmHg".into(),
        functional_id: "mean".into(),
        transform_id: "identity".into(),
    };
    assert_eq!(derived, binding.expected_quantities());
}

#[test]
fn a0r_units_are_required_and_never_inferred() {
    let r = response(curve(&[0.0, 1.0]), surface(2), None);
    for blank in [
        ResponseCoordinateLabels { outcome_units: " ", ..labels() },
        ResponseCoordinateLabels { population_id: "", ..labels() },
        ResponseCoordinateLabels { transform_id: "\t", ..labels() },
    ] {
        let refusal = response_coordinates(&r, &name_of, &blank).unwrap_err();
        assert_eq!(refusal.code, "invalid_argument");
        assert_eq!(refusal.detail, "coordinate_support.units_required");
        assert_eq!(refusal.validate(), Ok(()));
    }
}

#[test]
fn a0r_temporal_surface_coordinates_are_dose_major() {
    let r = response(curve(&[0.0, 1.0]), surface(4), Some(vec![1, 3]));
    let derived = response_coordinates(&r, &name_of, &labels()).unwrap();
    let cells: Vec<_> = derived.iter().map(|q| (q.regime_id.as_str(), q.horizon)).collect();
    // Cell `d * n_horizons + h`, the layout of the surface mean and its point_status.
    assert_eq!(cells, [("do(a=0)", 1), ("do(a=0)", 3), ("do(a=1)", 1), ("do(a=1)", 3)]);
}

#[test]
fn a0r_joint_set_and_shift_response_has_one_coordinate_naming_every_intervention() {
    let joint = intervention(vec![
        Intervention::set(v(A), Value::f64(1.0)),
        Intervention::shift(v(B), Value::f64(0.5)),
    ]);
    let derived = response_coordinates(&joint, &name_of, &labels()).unwrap();
    assert_eq!(derived.len(), 1, "a scalar value has exactly one coordinate");
    assert_eq!(derived[0].regime_id, "do(a=1) & shift(b=0.5)");
    assert_eq!((derived[0].horizon, derived[0].functional_id.as_str()), (0, "mean"));

    let shifted = intervention(vec![Intervention::shift(v(A), Value::Int64(2))]);
    let derived = response_coordinates(&shifted, &name_of, &labels()).unwrap();
    assert_eq!(derived[0].regime_id, "shift(a=2)");
}

#[test]
fn a0r_what_has_no_exact_regime_refuses_instead_of_being_scalarized() {
    let cases = [
        (
            intervention(vec![Intervention::stochastic(
                v(A),
                StochasticPolicy::gaussian(0.25, 0.1),
            )]),
            "stochastic",
        ),
        (
            intervention(vec![Intervention::soft(v(A), MechanismOverride::additive_shift(0.1))]),
            "soft",
        ),
        (
            intervention(vec![Intervention::sequence(InterventionSequence::new(vec![
                SequencedIntervention::new(
                    Intervention::set(v(A), Value::f64(1.0)),
                    TemporalPolicy::pulse(-1),
                ),
            ]))]),
            "sequence",
        ),
        (intervention(vec![Intervention::set(v(A), Value::Category(2))]), "set"),
        (intervention(vec![Intervention::set(v(A), Value::symbolic_intervention())]), "set"),
        (intervention(vec![Intervention::shift(v(A), Value::f64(f64::NAN))]), "shift"),
    ];
    for (case, kind) in cases {
        let refusal = refusal_of(&case);
        assert_eq!(refusal.code, "route_not_supported");
        assert_eq!(refusal.detail, "coordinate_support.regime_not_describable");
        assert_eq!(refusal.supplied.as_deref(), Some(kind));
    }
}

#[test]
fn a0r_derivative_functionals_and_temporal_paths_refuse_with_their_own_detail() {
    let derivative = response(
        ResponseFunctional::PointDerivative {
            outcome: v(Y),
            treatment: v(A),
            at: 0.5,
            order: 1,
            scale: antecedent_core::DerivativeScale::Identity,
        },
        ResponseValue::Scalar(1.0),
        None,
    );
    let refusal = refusal_of(&derivative);
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "coordinate_support.functional_not_described");
    assert_eq!(refusal.supplied.as_deref(), Some("point_derivative"));

    let jacobian = response(
        ResponseFunctional::Jacobian {
            outcomes: Arc::from([v(Y)]),
            treatments: Arc::from([v(A), v(B)]),
            at: Arc::from([0.0, 0.0]),
            scale: antecedent_core::DerivativeScale::Identity,
        },
        ResponseValue::Jacobian { outcomes: 1, treatments: 2, values: Arc::from([1.0, 2.0]) },
        None,
    );
    assert_eq!(refusal_of(&jacobian).supplied.as_deref(), Some("jacobian"));

    let path = response(
        ResponseFunctional::InterventionResponse {
            outcome: v(Y),
            interventions: Arc::from([Intervention::set(v(A), Value::f64(1.0))]),
        },
        ResponseValue::Vector(Arc::from([1.0, 2.0])),
        Some(vec![1, 2]),
    );
    assert_eq!(refusal_of(&path).detail, "coordinate_support.temporal_path_not_described");
}

#[test]
fn a0r_a_variable_the_caller_cannot_name_refuses() {
    let unknown = ResponseFunctional::MeanCurve {
        outcome: v(99),
        treatment: ContinuousDomain::new(v(A), GridSpec::Values(Arc::from([0.0, 1.0]))),
    };
    let refusal = refusal_of(&response(unknown, surface(2), None));
    assert_eq!(refusal.code, "unknown_variable");
    assert_eq!(refusal.detail, "coordinate_support.unknown_variable");
    assert_eq!(refusal.supplied.as_deref(), Some("99"));
}
