//! Bayesian panel difference-in-differences, synthetic-control and local-ratio routes through
//! the public study facade. The routes are explicitly unlicensed and uncalibrated: the checks
//! below compare each posterior with an independent closed form or known truth, read the
//! retained artifact with an independent loader, and refuse every boundary with its next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, InferenceMode, PanelDidQuery, Study, StudyResult};
use antecedent_core::{Assumption, AssumptionStatus, SyntheticControlQuery, VariableId};
use antecedent_data::TabularData;

fn noise(i: usize) -> f64 {
    ((i * 7) % 11) as f64 / 10.0 - 0.5
}

fn bayes(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().prior_scale(100.0).n_draws(draws))
}

/// Closed-form Normal-Inverse-Gamma posterior mean and sd of the group coefficient of
/// `y ~ 1 + g`, prior `beta | s2 ~ N(0, s2 scale^2)`, `s2 ~ InvGamma(a, b)`.
fn nig_group_posterior(y: &[f64], g: &[bool], scale: f64, a0: f64, b0: f64) -> (f64, f64) {
    let n = y.len() as f64;
    let n1 = g.iter().filter(|v| **v).count() as f64;
    let prec = 1.0 / (scale * scale);
    let (m00, m01, m11) = (n + prec, n1, n1 + prec);
    let det = m00 * m11 - m01 * m01;
    let inv = [[m11 / det, -m01 / det], [-m01 / det, m00 / det]];
    let sy: f64 = y.iter().sum();
    let sy1: f64 = y.iter().zip(g).filter(|(_, v)| **v).map(|(v, _)| *v).sum();
    let mean = [inv[0][0] * sy + inv[0][1] * sy1, inv[1][0] * sy + inv[1][1] * sy1];
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let quad =
        mean[0] * (m00 * mean[0] + m01 * mean[1]) + mean[1] * (m01 * mean[0] + m11 * mean[1]);
    let an = a0 + n / 2.0;
    let bn = b0 + 0.5 * (yy - quad);
    (mean[1], (bn / (an - 1.0) * inv[1][1]).sqrt())
}

fn panel_fixture() -> (TabularData, PanelDidQuery, Vec<f64>, Vec<bool>) {
    let (mut subjects, mut treated, mut post, mut outcome) = (vec![], vec![], vec![], vec![]);
    let (mut changes, mut groups) = (vec![], vec![]);
    for j in 0..24 {
        let t = j >= 12;
        let before = 1.0 + 0.3 * (j % 3) as f64 + noise(j);
        let change = 0.7 + 2.0 * f64::from(t) + noise(j + 3);
        for after in [false, true] {
            subjects.push(Arc::<str>::from(format!("s{j}")));
            treated.push(t);
            post.push(after);
            outcome.push(if after { before + change } else { before });
        }
        changes.push(change);
        groups.push(t);
    }
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let clusters = subjects.clone();
    let query = PanelDidQuery::new(VariableId::from_raw(0), treated, post, subjects, clusters);
    (data, query, changes, groups)
}

fn run(
    data: &TabularData,
    query: impl Into<antecedent_core::CausalQuery>,
    inference: InferenceMode,
    seed: u64,
) -> Result<(antecedent::PreparedStudy, StudyResult, ExecutionContext), antecedent::CausalError> {
    let ctx = ExecutionContext::for_tests(seed);
    let prepared =
        Study::tabular(data.clone()).query(query).inference(inference).build()?.prepare(&ctx)?;
    let result = prepared.estimate(data, &ctx)?;
    Ok((prepared, result, ctx))
}

fn effect_summary(result: &StudyResult) -> (f64, f64, f64, f64) {
    let posterior = result.posterior.as_ref().unwrap();
    let col = posterior.effect_column().unwrap();
    let s = &posterior.summaries;
    (s.mean[col], s.sd[col], s.q025[col], s.q975[col])
}

#[test]
fn panel_did_posterior_matches_the_closed_form_and_keeps_premises_declared() {
    let (data, query, changes, groups) = panel_fixture();
    let (prepared, result, ctx) = run(&data, query, bayes(20_000), 11).unwrap();
    let (mean, sd) = nig_group_posterior(&changes, &groups, 100.0, 1e-3, 1e-3);
    let (got_mean, got_sd, lower, upper) = effect_summary(&result);
    assert!((got_mean - mean).abs() < 0.02, "{got_mean} vs {mean}");
    assert!((got_sd - sd).abs() < 0.03 * sd, "{got_sd} vs {sd}");
    assert!(lower < mean && mean < upper);
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.panel_did"));
    assert_eq!(result.support_status, None, "the Bayesian design route is unlicensed");
    assert!(result.panel_did().is_none());
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, Assumption::Custom { id, .. } if id.as_ref() == "parallel_trends"
    ) && record.status == AssumptionStatus::Declared));
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "identification.quasi.bayesian.parallel_trends_declared")
    );
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.quasi.bayesian.unlicensed")
    );

    // Independent loader: decode the retained artifact and recompute the summaries from its draws.
    let bytes = prepared.encode_contracted_result(&result, "bayes-panel-did", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let posterior_bytes = body.posterior_artifact.clone().unwrap();
    let (wire, draws) = antecedent_io::decode_causal_posterior_bytes(&posterior_bytes).unwrap();
    assert_eq!(wire.n_draws, 20_000);
    let first: &[f64] = &draws[..usize::try_from(wire.n_draws).unwrap()];
    let recomputed = first.iter().sum::<f64>() / first.len() as f64;
    assert!((recomputed - got_mean).abs() < 1e-9);
    assert!(body.panel_did.is_none());

    // A forged artifact that drops the posterior (no design section either) or carries both is refused.
    let mut dropped = body.clone();
    dropped.posterior_artifact = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &dropped,
            header.variable_names.clone(),
            "d"
        )
        .is_err()
    );
    let mut corrupted = body;
    corrupted.posterior_artifact = Some(vec![1, 2, 3]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(&corrupted, header.variable_names, "c")
            .is_err()
    );
}

#[test]
fn panel_did_refuses_shared_clusters_and_foreign_backends_with_next_attempts() {
    let (data, query, _, _) = panel_fixture();
    let mut shared = query.clone();
    let clusters: Vec<Arc<str>> =
        (0..shared.clusters.len()).map(|i| Arc::<str>::from(format!("c{}", (i / 2) % 4))).collect();
    shared.clusters = clusters.into();
    let error = run(&data, shared, bayes(2_000), 3).unwrap_err().to_string();
    assert!(error.contains("reason=sampling_dependence_unknown"), "{error}");
    assert!(error.contains("bayesian_did_dependent_units"), "{error}");
    assert!(error.contains("Next attempt"), "{error}");

    let laplace = InferenceMode::Bayesian(BayesianConfig::laplace());
    let error = run(&data, query.clone(), laplace, 3).unwrap_err().to_string();
    assert!(error.contains("conjugate"), "{error}");
}

fn ratio_fixture(strong: bool) -> TabularData {
    let (mut x, mut t, mut y) = (vec![], vec![], vec![]);
    for i in 0..200 {
        let r = (i as f64 - 100.0) / 100.0;
        let a = f64::from(r >= 0.0);
        let dose = if strong { 0.2 + 0.6 * a + 0.05 * noise(i) } else { 0.3 * noise(i) };
        x.push(r);
        t.push(dose);
        y.push(1.0 + 2.0 * dose + 0.3 * r + 0.05 * noise(i + 3));
    }
    TabularData::from_f64_columns([("x", x.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn ratio_query() -> antecedent_core::LocalPolynomialRatioQuery {
    antecedent_core::LocalPolynomialRatioQuery {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(1),
        running: VariableId::from_raw(0),
        cutoff: 0.0,
        bandwidth: 0.8,
        kink: false,
        options: None,
    }
}

#[test]
fn local_ratio_recovers_the_known_ratio_and_withholds_the_interval_when_weak() {
    let data = ratio_fixture(true);
    let (prepared, result, ctx) = run(&data, ratio_query(), bayes(8_000), 5).unwrap();
    let (median, _, lower, upper) = effect_summary(&result);
    assert!(lower < 2.0 && 2.0 < upper, "{lower} {upper}");
    assert!((result.effect() - 2.0).abs() < 0.2, "{}", result.effect());
    assert!((median - 2.0).abs() < 0.3);
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.local_ratio"));
    assert!(result.interval.is_some());
    assert!(!result.diagnostics.iter().any(|d| d.code.as_ref().ends_with("weak_first_stage")));
    let bytes = prepared.encode_contracted_result(&result, "bayes-ratio", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert!(body.local_polynomial_ratio.is_none() && body.posterior_artifact.is_some());

    let weak = ratio_fixture(false);
    let (_, result, _) = run(&weak, ratio_query(), bayes(8_000), 5).unwrap();
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.quasi.bayesian_local_ratio.weak_first_stage"
                && d.message.contains("Next attempt"))
    );
    assert!(result.interval.is_none(), "a weak first stage publishes no interval");
    let posterior = result.posterior.as_ref().unwrap();
    assert!(posterior.effect_column().is_none(), "no ratio draws are published");
}

fn synthetic_fixture() -> (TabularData, SyntheticControlQuery) {
    let (mut outcome, mut units, mut periods) = (vec![], vec![], vec![]);
    for unit in 0..6 {
        for t in 1..=12_i64 {
            let donor = |j: usize| {
                j as f64
                    + 0.3 * t as f64 * (j % 3) as f64
                    + (usize::try_from(t).unwrap() * (j + 3) * 7 % 13) as f64 / 13.0
            };
            let value = if unit == 5 {
                0.5 * donor(0)
                    + 0.5 * donor(1)
                    + 0.02 * noise(usize::try_from(t).unwrap())
                    + if t >= 9 { 3.0 } else { 0.0 }
            } else {
                donor(unit)
            };
            outcome.push(value);
            units.push(Arc::<str>::from(if unit == 5 {
                "treated".to_string()
            } else {
                format!("d{unit}")
            }));
            periods.push(t);
        }
    }
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    (data, SyntheticControlQuery::new(VariableId::from_raw(0), units, periods, "treated", 9))
}

#[test]
fn synthetic_control_posterior_recovers_the_effect_and_records_convergence() {
    let (data, query) = synthetic_fixture();
    let (prepared, result, ctx) = run(&data, query.clone(), bayes(4_000), 9).unwrap();
    let (mean, _, lower, upper) = effect_summary(&result);
    assert!((mean - 3.0).abs() < 0.6, "{mean}");
    assert!(lower < 3.0 && 3.0 < upper);
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.synthetic_control"));
    let diagnostics = &result.posterior.as_ref().unwrap().diagnostics;
    assert_eq!(diagnostics.n_chains, Some(4));
    assert!(diagnostics.rhat_max.unwrap() <= 1.05);
    assert!(diagnostics.ess_bulk_min.unwrap() >= 100.0);
    assert!(diagnostics.notes.iter().any(|n| n.starts_with("donor_weight unit=d0")));
    let bytes = prepared.encode_contracted_result(&result, "bayes-sc", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert!(body.synthetic_control.is_none() && body.posterior_artifact.is_some());

    let error = run(&data, query.clone().difference_in_differences(), bayes(4_000), 9)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("reason=route_not_supported") && error.contains("Next attempt"),
        "{error}"
    );
    let error = run(&data, query.with_augmentation(1.0), bayes(4_000), 9).unwrap_err().to_string();
    assert!(error.contains("Next attempt"), "{error}");
}
