//! Route-native diagnostics reach `CausalError::route_diagnostics` from the code paths
//! that detect them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::CausalError;
use antecedent_core::diagnostic::RouteKind;
use antecedent_core::{CausalRng, DeclaredSamplingDesign, SamplingDesignDeclaration};
use antecedent_data::{ResamplingPlan, fill_declared_plan};

#[test]
fn a_dependent_series_resampled_by_an_iid_plan_is_a_dependent_resampling_check() {
    let declaration =
        SamplingDesignDeclaration::new(DeclaredSamplingDesign::DependentTimeSeries, true);
    let mut out = Vec::new();
    let error = fill_declared_plan(
        declaration,
        ResamplingPlan::IidBootstrap,
        8,
        None,
        &mut CausalRng::from_seed(5),
        &mut out,
    )
    .expect_err("an iid plan drops the series dependence");
    let records = CausalError::from(error).route_diagnostics();
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(records[0].kind(), Some(RouteKind::DependentResampling));
    assert!(records[0].failed());
    assert!(records[0].unevaluable_mass().is_unreported());
}
