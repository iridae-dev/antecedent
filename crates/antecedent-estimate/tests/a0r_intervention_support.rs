//! A0 remainder: a frequentist `InterventionResponse` (set, shifted or stochastic
//! policy) answers one scalar, so it carries exactly one coordinate and one support
//! label, equal to its summary. It is never scalarized: the label is the summary,
//! and a policy with no exact regime refuses to name a coordinate.
//!
//! Oracle: the regimes are written out by hand (`do(a=0.25)`, `shift(a=0.25)`); the
//! label count is the value count (one scalar), and the summary is `Extrapolative`
//! because policy support is not certified (see the estimator's own support detail).
#![allow(clippy::cast_precision_loss, reason = "deterministic design")]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, IdentificationStatus, Intervention, ResponseCoordinateLabels,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue, StochasticPolicy,
    SupportStatus, Value, VariableId, response_coordinates,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseEstimator;

fn data() -> TabularData {
    let n = 500_usize;
    let (mut a, mut y, mut x) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        let noise = ((i * 37 % 101) as f64 / 100.0 - 0.5) * 0.3;
        let treatment = 0.7 * z + noise;
        x.push(z);
        a.push(treatment);
        y.push(1.0 + 2.0 * treatment + 0.8 * z + 0.05 * (i as f64).sin());
    }
    TabularData::from_f64_columns([("a", a.as_slice()), ("y", y.as_slice()), ("x", x.as_slice())])
        .unwrap()
}

fn name_of(id: VariableId) -> Option<String> {
    ["a", "y", "x"].get(id.as_usize()).map(|name| (*name).to_owned())
}

fn estimate(data: &TabularData, intervention: Intervention) -> antecedent_core::CausalResponse {
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([intervention]),
    });
    ContinuousResponseEstimator::new([VariableId::from_raw(2)])
        .estimate_identified(
            data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap()
}

#[test]
fn a0r_set_and_shift_policies_carry_one_label_equal_to_their_summary() {
    let data = data();
    let a = VariableId::from_raw(0);
    let labels = ResponseCoordinateLabels {
        outcome_units: "mmHg",
        population_id: "target",
        transform_id: "identity",
    };
    for (intervention, regime) in [
        (Intervention::set(a, Value::f64(0.25)), "do(a=0.25)"),
        (Intervention::shift(a, Value::f64(0.25)), "shift(a=0.25)"),
    ] {
        let response = estimate(&data, intervention);
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(_)) = &response.estimate
        else {
            panic!("a policy answers one scalar");
        };
        assert_eq!(
            response.support.point_status.as_deref(),
            Some(&[SupportStatus::Extrapolative][..]),
            "{regime}"
        );
        assert_eq!(response.support.status, SupportStatus::Extrapolative);
        let coordinates = response_coordinates(&response, &name_of, &labels).unwrap();
        assert_eq!(coordinates.len(), 1, "one scalar, one coordinate");
        assert_eq!(coordinates[0].regime_id, regime);
    }
}

#[test]
fn a0r_a_stochastic_policy_is_labelled_but_refuses_to_name_a_regime() {
    let data = data();
    let response = estimate(
        &data,
        Intervention::stochastic(VariableId::from_raw(0), StochasticPolicy::gaussian(0.25, 0.01)),
    );
    assert_eq!(response.support.point_status.as_deref(), Some(&[SupportStatus::Extrapolative][..]));
    let labels = ResponseCoordinateLabels {
        outcome_units: "mmHg",
        population_id: "target",
        transform_id: "identity",
    };
    let refusal = response_coordinates(&response, &name_of, &labels).unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "coordinate_support.regime_not_describable");
}
