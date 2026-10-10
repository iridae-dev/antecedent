//! A0 remainder: the coordinates the facade getter reads off a native response are
//! identical to the C1 program-bound getter's, and to a hand derivation.
//!
//! Hand example. Outcome `y` (mmHg) under `do(dose = d)` for `d in {0.5, 1, 2}` in
//! the `target` population: three coordinates whose regimes read `do(dose=0.5)`,
//! `do(dose=1)` and `do(dose=2)`, functional `mean`, horizon 0, identity scale.

use std::sync::Arc;

use antecedent::analysis::native_claims::{NativeResponseContext, native_response_coordinates};
use antecedent_core::{
    AssumptionSet, CausalResponse, CausalSchema, CausalSchemaBuilder, ContinuousDomain, GridSpec,
    IdentificationStatus, ProgramBinding, QuantityRole, ResponseCoordinateLabels,
    ResponseFunctional, ResponseIdentification, ResponseUncertainty, ResponseValue,
    ScientificQuantity, SupportRegion, SupportReport, SupportStatus, VariableId,
    response_coordinates,
};
use antecedent_io::distribution_artifact::DistributionCalibration;
use antecedent_io::quantity_wire::ScientificQuantityWire;

const GRID: [f64; 3] = [0.5, 1.0, 2.0];

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

fn response() -> CausalResponse {
    let schema = schema();
    CausalResponse {
        estimand: ResponseFunctional::MeanCurve {
            outcome: schema.id_of("y").unwrap(),
            treatment: ContinuousDomain::new(
                schema.id_of("dose").unwrap(),
                GridSpec::Values(Arc::from(GRID.to_vec())),
            ),
        },
        identification_status: IdentificationStatus::NonparametricallyIdentified,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid: Arc::from(GRID.to_vec()),
            dimension: 1,
            mean: Arc::from(vec![1.0, 2.0, 3.0]),
        }),
        uncertainty: ResponseUncertainty::None,
        support: SupportReport {
            status: SupportStatus::Supported,
            query_region: SupportRegion { minima: Arc::from([0.5]), maxima: Arc::from([2.0]) },
            diagnostics: vec![],
            warnings: vec![],
            point_status: Some(Arc::from(vec![SupportStatus::Supported; 3])),
        },
        assumptions: AssumptionSet::new(),
        provenance_id: "op-1".into(),
        horizon_identification: None,
        interaction_structurally_zero: false,
    }
}

fn program() -> ProgramBinding {
    ProgramBinding {
        graph_id: "graph:g".into(),
        contract_id: "contract:c".into(),
        treatment_id: "dose".into(),
        outcome_id: "y".into(),
        population_id: "target".into(),
        intervention_kind: "do".into(),
        horizon: 0,
        dose_grid: GRID.to_vec(),
        dose_units: "mg".into(),
        outcome_units: "mmHg".into(),
        functional_id: "mean".into(),
        transform_id: "identity".into(),
    }
}

fn by_hand(dose: &str) -> ScientificQuantity {
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

#[test]
fn a0r_facade_getter_equals_the_c1_program_getter_and_the_hand_derivation() {
    let schema = schema();
    let name_of = |id: VariableId| schema.get(id).ok().map(|variable| variable.name.to_string());
    let labels = ResponseCoordinateLabels {
        outcome_units: "mmHg",
        population_id: "target",
        transform_id: "identity",
    };
    let facade = response_coordinates(&response(), &name_of, &labels).unwrap();
    let context = NativeResponseContext {
        program: program(),
        snapshot_id: "snap-1".into(),
        rng_id: "pcg:seed=7".into(),
        calibration: DistributionCalibration::Unmeasured,
    };
    let c1 = native_response_coordinates(&response(), &schema, &context).unwrap();
    let hand = vec![by_hand("0.5"), by_hand("1"), by_hand("2")];
    assert_eq!(facade, hand);
    assert_eq!(facade, c1);
}

#[test]
fn a0r_facade_coordinates_survive_the_python_wire_unchanged() {
    // The PyO3 getter returns these wire objects as JSON; the round trip must be exact.
    let schema = schema();
    let name_of = |id: VariableId| schema.get(id).ok().map(|variable| variable.name.to_string());
    let labels = ResponseCoordinateLabels {
        outcome_units: "mmHg",
        population_id: "target",
        transform_id: "identity",
    };
    let derived = response_coordinates(&response(), &name_of, &labels).unwrap();
    let wires: Vec<ScientificQuantityWire> = derived.iter().map(Into::into).collect();
    let json = serde_json::to_string(&wires).unwrap();
    let back: Vec<ScientificQuantityWire> = serde_json::from_str(&json).unwrap();
    let restored: Vec<ScientificQuantity> =
        back.into_iter().map(|wire| ScientificQuantity::try_from(wire).unwrap()).collect();
    assert_eq!(restored, derived);
    assert_eq!(wires[1].regime_id, "do(dose=1)");
    assert_eq!(wires[1].role, "outcome");
}
