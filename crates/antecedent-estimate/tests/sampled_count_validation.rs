//! Validation scan equivalence for sampled count tables.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{RegimeId, VariableId};
use antecedent_estimate::counterfactual_transport_sampled::{
    SampledAxis, SampledRegimeCounts, SampledRegimeCountsParts, SamplingDesign,
};

fn parts() -> SampledRegimeCountsParts {
    SampledRegimeCountsParts {
        population: Arc::from("population"),
        regime: RegimeId::from_raw(0),
        interventions: Vec::new(),
        axes: vec![
            SampledAxis { variable: VariableId::from_raw(1), levels: vec![0.0] },
            SampledAxis { variable: VariableId::from_raw(2), levels: vec![1.0] },
        ],
        counts: vec![3],
        snapshot_identity: Arc::from("snapshot"),
        design: SamplingDesign::IndependentUnits,
        structural_zeros: Vec::new(),
    }
}

#[test]
fn axis_and_strata_scans_preserve_refusal_precedence_and_order() {
    let message = |p| SampledRegimeCounts::try_new(p).unwrap_err().to_string();
    let mut bad = parts();
    bad.axes[0].levels.clear();
    bad.axes[1].variable = bad.axes[0].variable;
    assert!(message(bad).contains("axis has no levels"));
    let mut bad = parts();
    bad.axes[1].variable = bad.axes[0].variable;
    bad.axes[1].levels.clear();
    assert!(message(bad).contains("variable is measured on two axes"));
    for variables in [
        vec![],
        vec![VariableId::from_raw(1), VariableId::from_raw(1)],
        vec![VariableId::from_raw(9)],
    ] {
        let mut bad = parts();
        bad.design = SamplingDesign::Stratified { variables };
        assert!(message(bad).contains("stratification variables are distinct measured variables"));
    }
    let mut bad = parts();
    bad.interventions.push((VariableId::from_raw(2), 1.0));
    assert!(message(bad).contains("intervenes on a measured axis"));
    let mut valid = parts();
    valid.interventions.push((VariableId::from_raw(3), 1.0));
    valid.design = SamplingDesign::Stratified {
        variables: vec![VariableId::from_raw(2), VariableId::from_raw(1)],
    };
    let expected_axes = valid.axes.clone();
    let expected_design = valid.design.clone();
    let table = SampledRegimeCounts::try_new(valid).unwrap();
    assert_eq!(table.axes(), expected_axes);
    assert_eq!(table.design(), &expected_design);
    assert_eq!(table.counts(), &[3]);
    assert_eq!(table.interventions(), &[(VariableId::from_raw(3), 1.0)]);
}
