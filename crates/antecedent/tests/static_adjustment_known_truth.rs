//! Known-truth evidence for the static `AverageEffect` adjustment estimators on every
//! coordinate they claim: graph class (`Dag`, `Cpdag`, `Pag`, tier `CoDetermined`/`Unknown`),
//! structure source (`explicit`, `accepted`, `graph_posterior`) and validation suite.
//!
//! The design is exactly balanced and noise free (see `known_truth_cells`). The treatment share
//! depends on `z` (1/3 and 2/3), so the naive contrast is `2 + 2/3` and only adjustment
//! recovers the structural coefficient 2. Because `y = 2 t + z` exactly:
//!
//! * the arm-specific linear outcome models of AIPW, DML, the DR-learner and the linear and
//!   Gaussian-identity adjustments are exact, every residual vanishes, and the score
//!   average is the contrast `mu(1, z) - mu(0, z) = 2` whatever the cross-fitted propensity;
//! * the logistic propensity on `z` (and on `z` with the quarter-splitting nuisance
//!   indicators) reproduces the cell proportions 1/3 and 2/3 exactly, so the Hajek-weighted arm
//!   means both equal the population mean of `z` (0) and the weighted contrast is 2;
//! * matching on the score or on the covariates finds a donor with the same `z`, so every
//!   matched contrast is 2; with two strata the propensity quantile cut falls between the two
//!   `z` cells, so each stratum is one `z` cell and its raw difference of means is 2.
//!
//! The causal forest reports the cross-fitted ridge AIPW marginal, whose ridge penalty is not
//! zero, so its design carries an outcome that is constant within arm (`y = 2 t`, `z` only
//! causes `t`): the ridge fit of a constant is the constant, the honest leaf contrasts are 2, and
//! the marginal is exact.
//!
//! The expected value is computed in the test from the table (`stratified_truth`), never from
//! a production estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "known_truth_cells/mod.rs"]
mod support;

use antecedent::{EstimatorId, EstimatorSpec, InferenceMode};
use antecedent_estimate::{
    CausalForest, DmlAte, DrLearner, GlmAdjustmentAte, PropensityStratification,
};
use antecedent_learn::{LearnerSpec, LinearSpec};
use antecedent_stats::GlmFamily;
use support::{
    Design, UNKNOWN_TRUTH, assert_coordinate, confounded_truth, for_each_coordinate,
    on_large_stack, run_average_effect, stratified_truth_with, table_with, unknown_table_with,
};

/// Exact designs: floating-point round-off of a handful of solves.
const EXACT: f64 = 1e-9;
/// Estimators whose propensity comes from an iterative logistic fit stopped at the library's
/// convergence tolerance.
const ITERATIVE: f64 = 1e-6;

fn linear() -> LearnerSpec {
    LearnerSpec::Linear(LinearSpec::default())
}

fn constant_arm(t: f64, _z: f64) -> f64 {
    2.0 * t
}

fn constant_arm_unknown(_era: f64, t: f64) -> f64 {
    t
}

fn check_family(
    family: &'static str,
    estimator_id: &'static str,
    spec: EstimatorSpec,
    design: Design,
    truth: f64,
    unknown_truth: [f64; 2],
    tolerance: f64,
) {
    on_large_stack(move || {
        for_each_coordinate(family, |coordinate| {
            let result =
                run_average_effect(coordinate, &spec, &InferenceMode::Frequentist, &design, 31);
            assert_coordinate(coordinate, estimator_id, &result, truth, &unknown_truth, tolerance);
        });
    });
}

fn standard_tables() -> Design {
    Design::confounded()
}

#[test]
fn aipw_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = confounded_truth();
    check_family(
        "aipw|AverageEffect",
        "aipw",
        EstimatorSpec::Default(EstimatorId::Aipw),
        standard_tables(),
        truth,
        UNKNOWN_TRUTH,
        EXACT,
    );
}

#[test]
fn dml_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = confounded_truth();
    let spec = EstimatorSpec::Dml(Box::new(DmlAte::new().with_outcome(linear()).with_folds(4)));
    check_family("dml|AverageEffect", "dml", spec, standard_tables(), truth, UNKNOWN_TRUTH, EXACT);
}

#[test]
fn dr_learner_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = confounded_truth();
    let spec =
        EstimatorSpec::DrLearner(Box::new(DrLearner::new().with_outcome(linear()).with_folds(4)));
    check_family(
        "dr.learner|AverageEffect",
        "dr.learner",
        spec,
        standard_tables(),
        truth,
        UNKNOWN_TRUTH,
        EXACT,
    );
}

#[test]
fn causal_forest_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = stratified_truth_with(constant_arm);
    assert!((truth - 2.0).abs() < 1e-12, "forest design truth {truth}");
    let spec = EstimatorSpec::CausalForest(Box::new(CausalForest::new().with_n_trees(100)));
    check_family(
        "causal.forest|AverageEffect",
        "causal.forest",
        spec,
        Design {
            main: table_with(constant_arm),
            unknown: unknown_table_with(constant_arm_unknown),
            ..Design::confounded()
        },
        truth,
        UNKNOWN_TRUTH,
        EXACT,
    );
}

#[test]
fn glm_adjustment_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    // The identity-link member of the GLM family on a noise-free linear law is exact; the
    // logit link needs a binary outcome and is covered by its own design.
    let truth = confounded_truth();
    let spec = EstimatorSpec::GlmAdjustment(Box::new(
        GlmAdjustmentAte::new()
            .with_family(GlmFamily::GaussianIdentity)
            .with_bootstrap_replicates(0),
    ));
    check_family(
        "glm.adjustment|AverageEffect",
        "glm.adjustment",
        spec,
        standard_tables(),
        truth,
        UNKNOWN_TRUTH,
        EXACT,
    );
}

#[test]
fn propensity_weighting_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = confounded_truth();
    check_family(
        "propensity.weighting|AverageEffect",
        "propensity.weighting",
        EstimatorSpec::Default(EstimatorId::PropensityWeighting),
        standard_tables(),
        truth,
        UNKNOWN_TRUTH,
        ITERATIVE,
    );
}

#[test]
fn propensity_matching_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = confounded_truth();
    check_family(
        "propensity.matching|AverageEffect",
        "propensity.matching",
        EstimatorSpec::Default(EstimatorId::PropensityMatching),
        standard_tables(),
        truth,
        UNKNOWN_TRUTH,
        ITERATIVE,
    );
}

#[test]
fn propensity_stratification_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = confounded_truth();
    // Two strata cut the fitted scores between the two `z` cells (each holds half the rows).
    let spec = EstimatorSpec::PropensityStratification(Box::new(
        PropensityStratification::new().with_n_strata(2).with_bootstrap_replicates(0),
    ));
    check_family(
        "propensity.stratification|AverageEffect",
        "propensity.stratification",
        spec,
        standard_tables(),
        truth,
        UNKNOWN_TRUTH,
        EXACT,
    );
}

#[test]
fn distance_matching_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = confounded_truth();
    check_family(
        "distance.matching|AverageEffect",
        "distance.matching",
        EstimatorSpec::Default(EstimatorId::DistanceMatching),
        standard_tables(),
        truth,
        UNKNOWN_TRUTH,
        EXACT,
    );
}

#[test]
fn linear_adjustment_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = confounded_truth();
    check_family(
        "linear.adjustment.ate|AverageEffect",
        "linear.adjustment.ate",
        EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
        standard_tables(),
        truth,
        UNKNOWN_TRUTH,
        EXACT,
    );
}
