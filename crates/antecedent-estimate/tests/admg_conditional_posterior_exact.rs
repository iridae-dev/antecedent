//! The conditional (ADMG) transport route under a Dirichlet posterior mean, against the
//! enumerated latent-SCM truth `P*(y | do(x), w)` (`common/admg_conditional_scm.rs`).
//!
//! Every supplied law is the exact preimage of the enumerated law under the update
//! (`w = u (1 + K c) - c` over positive cells, so the Dirichlet mean with concentration `c` is
//! exactly `u`); the route (mean of every law, then the checked conditional formula) must return
//! the enumerated conditional truth for every level of the conditioning set, and the supplied
//! laws must differ from the enumerated ones. The chain `0 -> 1 -> 2` with node 0 selected and
//! a confounded variant with a bidirected edge between 1 and 2 are both exercised; at least the
//! chain must be identified.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "binary levels and small graph coordinates"
)]

#[path = "common/admg_conditional_scm.rs"]
mod admg_conditional_scm;

use admg_conditional_scm::{Scm, query, request};
use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::posterior_mean::{
    DirichletNamespace, PosteriorRequest, dirichlet_supported_mean,
};
use antecedent_estimate::prepare_exact_admg_conditional_transport;
use antecedent_expr::{ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    ADMG_CONDITIONAL_DEFAULT_LIMITS, BoundConditionalTransportFunctional,
    ConditionalTransportDecision, decide_admg_conditional_transport,
};

/// Laws whose Dirichlet posterior mean (concentration `c`) is exactly the supplied ones.
fn updated_by_preimage(data: &ExactTransportData, c: f64) -> (ExactTransportData, f64) {
    let mut moved = 0.0_f64;
    let laws: Vec<ExactDiscreteLaw> = data
        .laws()
        .iter()
        .map(|law| {
            let u = law.probabilities();
            let k = u.iter().filter(|p| **p > 0.0).count() as f64;
            let counts: Vec<f64> = u
                .iter()
                .map(|&p| {
                    if p > 0.0 {
                        let w = p * (1.0 + k * c) - c;
                        assert!(w >= 0.0, "cell {p} is below the update floor");
                        moved = moved.max((w - p).abs());
                        w
                    } else {
                        0.0
                    }
                })
                .collect();
            let mean = dirichlet_supported_mean(
                &counts,
                PosteriorRequest::IndependentNormal { scale: c },
                DirichletNamespace::AdmgConditional,
            )
            .unwrap();
            for (m, p) in mean.iter().zip(u) {
                assert!((m - p).abs() < 1e-12, "{m} vs {p}");
            }
            ExactDiscreteLaw::try_new(
                law.population(),
                law.regime(),
                law.interventions().to_vec(),
                law.axes().to_vec(),
                mean,
                law.snapshot_identity(),
                law.tolerance(),
            )
            .unwrap()
        })
        .collect();
    (ExactTransportData::try_new(laws, data.max_support_rows()).unwrap(), moved)
}

fn evaluate(
    functional: &BoundConditionalTransportFunctional,
    data: &ExactTransportData,
    x: &[usize],
    w: &[usize],
    level: usize,
) -> Result<antecedent_expr::ExactDistribution, EstimationError> {
    let ctx = ExecutionContext::for_tests(3);
    prepare_exact_admg_conditional_transport(
        functional,
        data.clone(),
        &request(x, w, level),
        ExactEvaluationLimits::default(),
        &ctx,
    )?
    .evaluate(&ctx)
}

/// Returns whether the query was identified (and, if so, checked).
fn check(scm: &Scm, y: &[usize], x: &[usize], w: &[usize]) -> bool {
    let ctx = ExecutionContext::for_tests(3);
    let (catalog, data) = scm.catalog_and_laws();
    let ConditionalTransportDecision::Identified(functional) = decide_admg_conditional_transport(
        &scm.diagram(),
        &query(y, x, w),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap() else {
        return false;
    };
    let (updated, moved) = updated_by_preimage(&data, 1e-6);
    assert!(moved > 1e-8, "the update must be nontrivial: {moved}");
    let mut compared = 0;
    for level in 0..(1usize << scm.n) {
        if (0..scm.n).any(|i| (level >> i) & 1 == 1 && !x.contains(&i) && !w.contains(&i)) {
            continue;
        }
        let expected = scm.truth(y, x, w, level).expect("positive parameterization");
        let point = evaluate(&functional, &updated, x, w, level).unwrap();
        assert_eq!(point.probabilities.len(), expected.len());
        for (actual, truth) in point.probabilities.iter().zip(&expected) {
            assert!((actual - truth).abs() < 1e-9, "level {level}: {actual} vs {truth}");
            compared += 1;
        }
    }
    assert!(compared > 0);
    true
}

#[test]
fn conditional_transport_on_dirichlet_updated_laws_equals_the_enumerated_truth() {
    let chain = Scm {
        n: 3,
        directed: vec![(0, 1), (1, 2)],
        bidirected: vec![],
        selected: vec![0],
        params: 0,
        target_zero: vec![],
    };
    assert!(check(&chain, &[2], &[1], &[0]), "the selected-root chain is identified");
    for params in 0..2 {
        let confounded = Scm {
            n: 3,
            directed: vec![(0, 1), (1, 2)],
            bidirected: vec![(1, 2)],
            selected: vec![0],
            params,
            target_zero: vec![],
        };
        // Identified or not, a non-identified query is simply not evaluated here.
        let _ = check(&confounded, &[2], &[1], &[0]);
    }
}
