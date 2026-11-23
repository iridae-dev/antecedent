//! The binary-observation recovery route under a Dirichlet posterior mean, against the
//! enumerated truth of the generating m-graph.
//!
//! The pattern counts are the exact preimage of the model's observed law under the Dirichlet
//! update: with concentration `c` per positive cell and `K` positive cells, the posterior mean
//! of `w` is `(w + c) / (1 + K c)`, so `w = u (1 + K c) - c` has posterior mean exactly the
//! observed law `u`. The observed law and the target law are marginals of the enumerated joint
//! (`support/recovery_scm.rs`, independent of the library). The route (Dirichlet mean, then the
//! recovery formula) must return the enumerated target law, and the supplied counts must differ
//! visibly from the law they update to, so a wrong normalisation or a skipped update fails.
#![allow(clippy::cast_possible_truncation, reason = "test indices are small")]

mod support {
    pub mod recovery_scm;
}

use antecedent_core::{ExecutionContext, RegimeId};
use antecedent_estimate::evaluate_exact_recovery;
use antecedent_estimate::posterior_mean::{
    DirichletNamespace, PosteriorRequest, dirichlet_supported_mean,
};
use antecedent_expr::{ExactDiscreteLaw, LawTolerance};
use antecedent_identify::{
    RecoveryDecision, RecoveryDerivation, RecoveryLimits, decide_observation_recovery,
};
use support::recovery_scm::MModel;

fn derive(model: &MModel, ctx: &ExecutionContext) -> Box<RecoveryDerivation> {
    match decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        None,
        RecoveryLimits::default(),
        ctx,
    )
    .unwrap()
    {
        RecoveryDecision::Recovered(d) => d,
        RecoveryDecision::NonRecoverable(_) => panic!("expected recoverable"),
    }
}

/// Counts whose Dirichlet posterior mean (concentration `c`) is `u`.
fn preimage(u: &[f64], c: f64) -> Vec<f64> {
    let k = u.iter().filter(|p| **p > 0.0).count() as f64;
    u.iter()
        .map(|&p| {
            if p > 0.0 {
                let w = p * (1.0 + k * c) - c;
                assert!(w >= 0.0, "cell {p} is below the update floor");
                w
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn binary_recovery_of_the_dirichlet_updated_pattern_law_equals_the_enumerated_truth() {
    let ctx = ExecutionContext::for_tests(1);
    for seed in [21_u64, 5] {
        let Some(model) = MModel::new(1, 1, &[(1, 0)], &[vec![1]], seed) else { continue };
        let derivation = derive(&model, &ctx);
        let observed = model.observed_law();
        let c = 1e-3;
        let counts = preimage(observed.probabilities(), c);
        let moved: f64 = counts
            .iter()
            .zip(observed.probabilities())
            .map(|(w, u)| (w - u).abs())
            .fold(0.0, f64::max);
        assert!(moved > 1e-4, "the update must be nontrivial: {moved}");
        let mean = dirichlet_supported_mean(
            &counts,
            PosteriorRequest::IndependentNormal { scale: c },
            DirichletNamespace::BinaryRecovery,
        )
        .unwrap();
        for (m, u) in mean.iter().zip(observed.probabilities()) {
            assert!((m - u).abs() < 1e-12, "{m} vs {u}");
        }
        let updated = ExactDiscreteLaw::try_new(
            observed.population(),
            RegimeId::from_raw(7),
            observed.interventions().to_vec(),
            observed.axes().to_vec(),
            mean,
            observed.snapshot_identity(),
            LawTolerance::default(),
        )
        .unwrap();
        let recovered = evaluate_exact_recovery(&derivation, &updated, &ctx).unwrap();
        let truth = model.truth();
        assert_eq!(recovered.law().probabilities().len(), truth.len());
        for (got, want) in recovered.law().probabilities().iter().zip(&truth) {
            assert!((got - want).abs() < 1e-9, "seed {seed}: {got} vs {want}");
        }
    }
}
