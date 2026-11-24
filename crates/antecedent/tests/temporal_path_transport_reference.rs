//! Time-ordered path transport against the independent finite-SCM oracle.
//!
//! The fixture is the mediation model with a covariate shift in `Z` (`Z` at time 0, `X` at 1,
//! `M` at 2, `Y` at 3). The oracle enumerates `P*(Y_{x, M_{x'}} = 1)` from the target
//! population's own structural equations; the route only ever sees the supplied regime laws, so
//! the exact contrast equals the enumerated direct effect to rounding, and a plug-in on a very
//! large table drawn at the exact cell probabilities (counts proportional to them) converges to
//! the same value. The exact route and its replay are pinned by
//! `counterfactual_transport_lifecycle.rs`; this file adds the sampled route, the time-order
//! refusal for it, and the support unit of each route.

#![allow(
    dead_code,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "the shared oracle fixtures are included whole; table counts round nonnegative cell probabilities times a unit count"
)]

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use std::sync::Arc;

use antecedent::CounterfactualTransportOptions;
use antecedent::analysis::temporal_path_transport::{
    TemporalPathSpec, compile_sampled_temporal_path_transport, compile_temporal_path_transport,
    consume_sampled_temporal_path_transport,
};
use antecedent_core::ExecutionContext;
use antecedent_estimate::counterfactual_transport_sampled::{
    OverlapDeclaration, SampledAxis, SampledRegimeCounts, SampledRegimeCountsParts,
    SampledTransportLimits, SampledTransportRequest, SamplingDesign, UnseenCellRule,
};
use antecedent_expr::ExactDiscreteLaw;
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, Target, all_cases};

/// Units in each sampled table; the counts are the exact cell probabilities times this.
const UNITS: u64 = 1 << 20;

fn case() -> Case {
    all_cases().into_iter().find(|c| c.name == "nonlinear_mediation_covariate_shift").unwrap()
}

fn target<'a>(case: &'a Case, label: &str) -> &'a Target {
    case.targets.iter().find(|t| t.label == label).unwrap()
}

fn spec(case: &Case, times: [(&str, u32); 4]) -> TemporalPathSpec {
    let id = |name: &str| adapter::variable_id(case, name).raw();
    TemporalPathSpec {
        treatment: id("X"),
        outcome: id("Y"),
        control_bits: 0.0_f64.to_bits(),
        active_bits: 1.0_f64.to_bits(),
        outcome_level_bits: 1.0_f64.to_bits(),
        times: times.into_iter().map(|(name, time)| (id(name), time)).collect(),
        active_edges: vec![(id("X"), id("Y"))],
    }
}

fn options(case: &Case) -> CounterfactualTransportOptions {
    CounterfactualTransportOptions {
        variable_names: case.graph.observed.iter().map(|name| (*name).to_owned()).collect(),
        ..Default::default()
    }
}

/// A table whose counts are the law's cell probabilities times `UNITS`, rounded, with the
/// rounding remainder put on the largest cell.
fn table_of(law: &ExactDiscreteLaw) -> SampledRegimeCounts {
    let mut counts: Vec<u64> =
        law.probabilities().iter().map(|p| (p * UNITS as f64).round() as u64).collect();
    let total: u64 = counts.iter().sum();
    let largest = (0..counts.len()).max_by_key(|cell| counts[*cell]).unwrap();
    counts[largest] = counts[largest] + UNITS - total;
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(law.population()),
        regime: law.regime(),
        interventions: law
            .interventions()
            .iter()
            .map(|a| (a.variable, a.value.as_f64().unwrap()))
            .collect(),
        axes: law
            .axes()
            .iter()
            .map(|axis| SampledAxis {
                variable: axis.variable,
                levels: axis.values.iter().map(|value| value.as_f64().unwrap()).collect(),
            })
            .collect(),
        counts,
        snapshot_identity: Arc::from(law.snapshot_identity()),
        design: SamplingDesign::IndependentUnits,
        structural_zeros: law
            .probabilities()
            .iter()
            .enumerate()
            .filter_map(|(cell, p)| (*p == 0.0).then_some(cell))
            .collect(),
    })
    .unwrap()
}

fn sampled_request(case: &Case) -> SampledTransportRequest {
    let adapted = adapter::catalog(case);
    let data = adapter::laws(case, &adapted);
    let mut limits = SampledTransportLimits::default();
    limits.evaluation.operations = 40_000_000;
    SampledTransportRequest {
        tables: data.laws().iter().map(table_of).collect(),
        overlap: OverlapDeclaration::all_independent(),
        unseen_rule: UnseenCellRule::Refuse,
        limits,
    }
}

const ORDER: [(&str, u32); 4] = [("Z", 0), ("X", 1), ("M", 2), ("Y", 3)];

fn compile_sampled(
    case: &Case,
    times: [(&str, u32); 4],
    ctx: &ExecutionContext,
) -> Result<
    antecedent::analysis::temporal_path_transport::CompiledSampledTemporalPathTransport,
    antecedent::analysis::temporal_path_transport::TemporalPathTransportError,
> {
    let adapted = adapter::catalog(case);
    compile_sampled_temporal_path_transport(
        adapter::problem(case),
        Arc::from("target"),
        Arc::from(adapter::sources(case, &adapter::evidence_controllable(case))),
        adapted.catalog,
        sampled_request(case),
        spec(case, times),
        options(case),
        ctx,
    )
}

#[test]
fn the_sampled_path_contrast_converges_to_the_enumerated_direct_effect() {
    let case = case();
    let active = target(&case, "Y_{x1,M_{x0}}").expected_value.unwrap();
    let baseline = target(&case, "Y_{x0,M_{x0}}").expected_value.unwrap();
    let ctx = ExecutionContext::for_tests(7);
    let plan = compile_sampled(&case, ORDER, &ctx).unwrap();
    assert_eq!(plan.plan_view().route().as_str(), "TemporalPathTransport");
    let result = plan.execute(&ctx).unwrap();
    let estimate = &result.estimate;
    // Counts are the exact probabilities to within 1 / 2^21 per cell, so the plug-in equals the
    // enumerated truth to well inside 1e-4.
    assert!((estimate.active - active).abs() < 1e-4, "{} vs {active}", estimate.active);
    assert!((estimate.baseline - baseline).abs() < 1e-4, "{} vs {baseline}", estimate.baseline);
    assert!((estimate.contrast - (active - baseline)).abs() < 1e-4);
    assert!((estimate.contrast - (estimate.active - estimate.baseline)).abs() < 1e-12);
    assert!(estimate.standard_error.is_some() && estimate.sampled_units > 0);
    assert_eq!(estimate.calibration, "not_measured");

    let bytes = plan.export(&result, &ctx).unwrap();
    let replay =
        consume_sampled_temporal_path_transport(&bytes, plan.lowered().identity(), &ctx).unwrap();
    assert_eq!(replay.estimate.contrast.to_bits(), estimate.contrast.to_bits());
}

#[test]
fn an_arrow_that_does_not_advance_time_is_refused_for_the_sampled_route() {
    let case = case();
    let ctx = ExecutionContext::for_tests(7);
    let flat = [("Z", 0), ("X", 1), ("M", 1), ("Y", 3)];
    assert!(compile_sampled(&case, flat, &ctx).is_err());
}

#[test]
fn the_path_routes_report_and_consult_the_support_unit_of_their_record() {
    let case = case();
    let ctx = ExecutionContext::for_tests(7);
    let sampled = compile_sampled(&case, ORDER, &ctx).unwrap();
    sampled.execute(&ctx).unwrap();

    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    let exact = compile_temporal_path_transport(
        adapter::problem(&case),
        Arc::from("target"),
        Arc::from(adapter::sources(&case, &adapter::evidence_controllable(&case))),
        adapted.catalog,
        data,
        spec(&case, ORDER),
        options(&case),
        &ctx,
    )
    .unwrap();
    exact.execute(&ctx).unwrap();

    let refused = ctx.would_refuse();
    for plan_view in [sampled.plan_view(), exact.plan_view()] {
        let view = plan_view.unit_support().expect("bound to its capability record");
        assert_eq!(view.unit(), "transport.temporal_path_specific");
        assert_eq!(view.supported(), view.status().as_str() == "supported");
        assert_eq!(view.supported(), view.missing_evidence().is_empty());
        assert!(!view.supported() || refused.is_empty(), "{refused:?}");
    }
    assert!(refused.iter().all(|entry| entry.code == "not_supported"), "{refused:?}");
}
