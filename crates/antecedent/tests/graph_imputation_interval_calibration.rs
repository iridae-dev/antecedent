//! Whole-method candidate measurement; no coverage authority before review and attestation.
#![cfg(feature = "calibration-internal")]

#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;

use antecedent::analysis::graph_imputation::{
    GraphImputationSpec, GraphImputationSupport, graph_imputation_calibration_candidate,
    graph_imputation_prior_predictive_calibration_candidate,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{ImputationSpec, TabularData};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim};
use std::sync::Arc;

/// The protocol every cell runs: its parameters are part of the reported construction, so a
/// record measured here covers an execution that declares the same ones.
fn spec(support: GraphImputationSupport, variables: usize) -> GraphImputationSpec {
    GraphImputationSpec {
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(1),
        support,
        imputation: ImputationSpec::new().with_imputations(8).with_iterations(10).with_seed(219),
        domains: Arc::from(&[1u8, 0, 0, 0][..variables]),
        posterior_draws: 500,
        prior_scale: 1.0,
        execution_seed: 223,
    }
}

/// One original dataset. Treatment is Bernoulli(logistic(`confounding` * (X + Z / 2))), so zero
/// is randomized and a positive value makes the covariates confound. The outcome is
/// `truth * T + 0.4 * X (+ 0.4 * Z) + N(0, 1)`, with the covariate coefficient raised to 0.8
/// where they confound. Outcome observation probability is 0.9 for control and 0.7 for
/// treated units. With four variables, Z is a second covariate; with three it is absent.
fn dataset(
    rng: &mut candidate::Generator,
    n: usize,
    variables: usize,
    confounding: f64,
    truth: f64,
) -> Option<TabularData> {
    let covariate_effect = if confounding == 0.0 { 0.4 } else { 0.8 };
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut x_column = Vec::with_capacity(n);
    let mut z_column = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.normal();
        let z = if variables == 4 { rng.normal() } else { 0.0 };
        let probability = if confounding == 0.0 {
            0.5
        } else {
            1.0 / (1.0 + (-confounding * (x + 0.5 * z)).exp())
        };
        let t = rng.binary(probability) as f64;
        let y = truth * t + covariate_effect * (x + z) + rng.normal();
        treatment.push(t);
        x_column.push(x);
        z_column.push(z);
        outcome.push(if rng.uniform() < if t == 0.0 { 0.9 } else { 0.7 } { y } else { f64::NAN });
    }
    let mut columns = vec![("t", treatment), ("y", outcome), ("x", x_column)];
    if variables == 4 {
        columns.push(("z", z_column));
    }
    TabularData::from_f64_columns(columns.iter().map(|(name, values)| (*name, values.as_slice())))
        .ok()
}

/// T randomized/root, X independent of T, Y=0.8T+0.4X+N(0,1); missing Y depends only on T.
#[test]
#[ignore = "calibration: run through scripts/measure_calibration.sh"]
fn graph_imputation_mar_bic_nearest_rank_candidate() {
    let n = grid_n(200);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x6a_3400 + replicate);
        let mut rng = candidate::Generator::new(seed);
        let source = dataset(&mut rng, n, 3, 0.0, 0.8)?;
        graph_imputation_calibration_candidate(
            source,
            &spec(GraphImputationSupport::TreatmentRootDirectEffect, 3),
            &ExecutionContext::for_tests(seed),
        )
        .ok()
    });
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "graph_imputation_mar_bic_nearest_rank_candidate",
            dgp: "randomized_binary_covariate_linear_normal_mar_by_treatment",
            interval: "nearest_rank_95",
        },
        0.95,
    );
    for result in results {
        match result {
            Some((lower, upper, basis)) => {
                candidate::bind(&mut tally, &basis);
                tally.record(Some((lower, upper)), 0.8);
            }
            None => tally.skip(),
        }
    }
    tally.assert();
}

/// Prior-predictive trial of one cell of the grid. The treatment effect is drawn from
/// Uniform(-1, 1) once per independent original dataset and is the causal truth. Coverage
/// averages over that declared effect distribution; it is not pointwise or uniform coverage.
/// The cell fixes the variable count (the graph size) and whether the covariates confound; the
/// sample size runs over the standard grid. Every failed original-data replicate fails the cell
/// rather than conditioning the coverage calculation on success.
fn prior_predictive_cell(
    test: &'static str,
    dgp: &'static str,
    variables: usize,
    confounding: f64,
    seed_base: u64,
) {
    // Where the covariates confound, the declared background knowledge is that they are
    // pre-treatment: without it the confounder graph is Markov equivalent to its mediator and
    // collider orientations, which adjust for nothing, and no fixed-graph truth is covered.
    let support = if confounding == 0.0 {
        GraphImputationSupport::TreatmentRootDirectEffect
    } else {
        GraphImputationSupport::PreTreatmentCovariates
    };
    let n = grid_n(200);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(seed_base + replicate);
        let mut effect_rng = candidate::Generator::new(seed ^ 0x5052_494f_525f_4245);
        let truth = 2.0 * effect_rng.uniform() - 1.0;
        let mut rng = candidate::Generator::new(seed);
        let source = dataset(&mut rng, n, variables, confounding, truth)?;
        let candidate = graph_imputation_prior_predictive_calibration_candidate(
            source,
            &spec(support, variables),
            &ExecutionContext::for_tests(seed),
        )
        .ok()?;
        Some((truth, candidate))
    });
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: "nearest_rank_95" }, 0.95);
    let mut skipped = 0u64;
    for result in results {
        if let Some((truth, (lower, upper, basis))) = result {
            candidate::bind(&mut tally, &basis);
            tally.record(Some((lower, upper)), truth);
        } else {
            skipped += 1;
            tally.skip();
        }
    }
    assert_eq!(skipped, 0, "prior-predictive coverage cannot condition on successful fits");
    tally.assert();
}

/// Three variables, randomized root treatment: the empty adjustment set in every graph.
#[test]
#[ignore = "calibration: run through scripts/measure_calibration.sh"]
fn graph_imputation_prior_predictive_uniform_effect_candidate() {
    prior_predictive_cell(
        "graph_imputation_prior_predictive_uniform_effect_candidate",
        "randomized_root_normal_covariate_beta_uniform_minus1_plus1_mar_by_treatment",
        3,
        0.0,
        0x6a_3500,
    );
}

/// Three variables, the covariate confounds and is declared pre-treatment: adjustment sets differ
/// between graphs. Every graph has a unique minimal back-door set, so the support rule excludes
/// none.
#[test]
#[ignore = "calibration: run through scripts/measure_calibration.sh"]
fn graph_imputation_prior_predictive_confounded_candidate() {
    prior_predictive_cell(
        "graph_imputation_prior_predictive_confounded_candidate",
        "confounded_normal_covariate_beta_uniform_minus1_plus1_mar_by_treatment",
        3,
        1.0,
        0x6a_3600,
    );
}

/// Four variables, randomized root treatment: a larger graph, still no confounding.
#[test]
#[ignore = "calibration: run through scripts/measure_calibration.sh"]
fn graph_imputation_prior_predictive_four_variable_candidate() {
    prior_predictive_cell(
        "graph_imputation_prior_predictive_four_variable_candidate",
        "randomized_root_two_normal_covariates_beta_uniform_minus1_plus1_mar_by_treatment",
        4,
        0.0,
        0x6a_3700,
    );
}

/// Four variables, both covariates confound and are declared pre-treatment: the largest graph
/// with differing adjustment sets. The support is the pre-treatment-covariate graphs whose
/// treatment-to-outcome effect has a unique minimal back-door adjustment set; graphs with two minimal sets are excluded and the mixture weights
/// the retained graphs only, so a record measured here covers that support rule and no other.
#[test]
#[ignore = "calibration: run through scripts/measure_calibration.sh"]
fn graph_imputation_prior_predictive_four_variable_confounded_candidate() {
    prior_predictive_cell(
        "graph_imputation_prior_predictive_four_variable_confounded_candidate",
        "confounded_two_normal_covariates_beta_uniform_minus1_plus1_mar_by_treatment",
        4,
        1.0,
        0x6a_3800,
    );
}
