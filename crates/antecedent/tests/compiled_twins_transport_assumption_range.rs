//! The compiled transport assumption range and `transport_assumption_range` give the same
//! result bit for bit. The hand bands are the closed form already stated by the design test.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use antecedent::analysis::transport_assumption_program::compile_transport_assumption_range;
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_design::transport_sensitivity::{
    TransportPremise, TransportRoute, TransportSensitivitySpec, ViolationFamily,
    transport_assumption_range,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn same<T: std::fmt::Debug>(compiled: &T, direct: &T) {
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
}

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "effect".into(),
        variable_name: "effect".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn spec(route: TransportRoute, value: f64, family: ViolationFamily) -> TransportSensitivitySpec {
    TransportSensitivitySpec {
        route,
        quantity: quantity(),
        point_value: value,
        family,
        grid_points: 3,
        source_population: "source".into(),
        target_population: "target".into(),
        query_binding: "derivation:1".into(),
        provider_snapshot: "snapshot:1".into(),
        causal_contract_id: "contract:1".into(),
    }
}

#[test]
fn compiled_transport_assumption_range_equals_the_kernel() {
    let ctx = ctx();
    let additive =
        spec(TransportRoute::CohortDecision, 3.0, ViolationFamily::AdditiveShift { bound: 2.0 });
    let direct = transport_assumption_range(&additive).unwrap();
    let program = compile_transport_assumption_range(additive, &ctx).unwrap();
    assert_eq!(program.lowered().route(), "TransportAssumptionRange");
    let support = program.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(support.unit(), "transport.assumption_range");
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    // v = 3, additive bound 2, grid {0, 1, 2}: [3 - s, 3 + s].
    assert_eq!(compiled.grid(), &[0.0, 1.0, 2.0]);
    assert_eq!(compiled.lower(), &[3.0, 2.0, 1.0]);
    assert_eq!(compiled.upper(), &[3.0, 4.0, 5.0]);
    assert_eq!(compiled.premise(), TransportPremise::SelectionDiagramInvariance);

    let proportional = spec(
        TransportRoute::MeasurementTransport,
        -4.0,
        ViolationFamily::ProportionalShift { bound: 0.5 },
    );
    let direct = transport_assumption_range(&proportional).unwrap();
    let compiled =
        compile_transport_assumption_range(proportional, &ctx).unwrap().execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    assert_eq!(compiled.grid(), &[0.0, 0.25, 0.5]);
    assert_eq!(compiled.lower(), &[-4.0, -5.0, -6.0]);
    assert_eq!(compiled.upper(), &[-4.0, -3.0, -2.0]);
}
