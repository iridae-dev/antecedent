//! Frozen degree-2 known-Gaussian whole-method design for the closed learned posterior.
//! The original Bayesian basis learner is refitted for every independent data sample.
//! Source rows are independent and disjoint; target covariates are supplied/fixed.
//! Correct invariant treatment effects, declared varying blocks, and known residual
//! variance are required. Weak declared Gaussian priors have variance 1000. This is
//! a finite fixed-parameter frequentist coverage grid, not universal prior calibration.
//! Actual joint target-effect draws supply type-7 intervals; fitting/drawing runs anew
//! for every replicate. Ordinary tests do not run these measurement experiments.
#![allow(clippy::cast_precision_loss, reason = "bounded independent simulation sizes")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
use antecedent_core::ExecutionContext;
use antecedent_estimate::joint_bayesian_transport::{
    GaussianPrior, JointPriors, JointTransportOptions, PriorProvenance, SourceDependence,
    SourceSharing, TransportGraphClass, VaryingBlock,
};
use antecedent_estimate::learned_joint_transport::{
    LearnedJointModel, fit_learned_joint_transport,
};
use calibration::{
    CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim, quantile_interval,
};

#[path = "common/joint_transport_design.rs"]
mod transport_design;
use transport_design::{DRAWS, FEATURE, proof, source, target};
fn model(varying: VaryingBlock, sharing: SourceSharing) -> LearnedJointModel {
    let (q, r) = if varying == VaryingBlock::Intercept { (5, 1) } else { (3, 3) };
    LearnedJointModel {
        basis_degree: 2,
        graph: TransportGraphClass::FixedDag,
        features: vec![FEATURE],
        varying,
        sharing,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(q, 0., 1000., PriorProvenance::Declared),
            varying: GaussianPrior::isotropic(r, 0., 1000., PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.,
        conflict_z_threshold: 3.,
    }
}
fn measure(
    test: &'static str,
    expected_id: &str,
    count: usize,
    varying: VaryingBlock,
    sharing: SourceSharing,
) {
    let n = grid_n(300);
    let model = model(varying, sharing);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "known_quadratic_gaussian_invariant_effect_fixed_target",
            interval: "posterior_quantile",
        },
        0.95,
    );
    let results = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0x44cb_7000 + rep);
        let sources: Vec<_> = (0..count)
            .map(|s| source(n, s, seed + 10_000 * s as u64, varying, sharing, true))
            .collect();
        let options = JointTransportOptions { draws: DRAWS, seed: seed + 100_000 };
        fit_learned_joint_transport(
            &proof(),
            &model,
            &sources,
            Some(&target()),
            &options,
            &ExecutionContext::for_tests(seed),
        )
        .map(|fit| {
            let basis = fit.target_calibration_basis(0.95).expect("actual posterior binding");
            let column = fit.draws.column("effect.target").expect("actual target effect column");
            let interval = quantile_interval(&fit.draws.coordinate(column), 0.95);
            (basis, interval)
        })
    });
    // Target mean(x)=.2, mean(x²)=.1225. True effect is 2+.4x+.2x²; source intercepts/slopes
    // never changes the target effect. This oracle never uses fitted coefficients.
    for result in results {
        match result {
            Ok((basis, interval)) => {
                candidate::bind(&mut tally, &basis);
                tally.record(interval, 2.1045);
            }
            Err(_) => tally.skip(),
        }
    }
    assert_eq!(tally.record_id().as_deref(), Some(expected_id));
    tally.assert();
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn learned_degree2_gaussian_one_source_intercept_l95() {
    measure(
        "learned_degree2_gaussian_one_source_intercept_l95",
        "cov.classical_transport.dag.bayesian.posterior_quantile.l95.learned_degree2_gaussian_one_source_intercept_l95",
        1,
        VaryingBlock::Intercept,
        SourceSharing::IndependentVaryingBlocks,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn learned_degree2_gaussian_two_independent_intercepts_l95() {
    measure(
        "learned_degree2_gaussian_two_independent_intercepts_l95",
        "cov.classical_transport.dag.bayesian.posterior_quantile.l95.learned_degree2_gaussian_two_independent_intercepts_l95",
        2,
        VaryingBlock::Intercept,
        SourceSharing::IndependentVaryingBlocks,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn learned_degree2_gaussian_two_shared_intercepts_l95() {
    measure(
        "learned_degree2_gaussian_two_shared_intercepts_l95",
        "cov.classical_transport.dag.bayesian.posterior_quantile.l95.learned_degree2_gaussian_two_shared_intercepts_l95",
        2,
        VaryingBlock::Intercept,
        SourceSharing::SharedVaryingBlock,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn learned_degree2_gaussian_two_varying_covariates_l95() {
    measure(
        "learned_degree2_gaussian_two_varying_covariates_l95",
        "cov.classical_transport.dag.bayesian.posterior_quantile.l95.learned_degree2_gaussian_two_varying_covariates_l95",
        2,
        VaryingBlock::InterceptAndCovariates,
        SourceSharing::IndependentVaryingBlocks,
    );
}

#[test]
fn candidate_uses_checked_standardization() {
    proof();
}
