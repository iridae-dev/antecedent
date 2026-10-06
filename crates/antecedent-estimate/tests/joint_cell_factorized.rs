//! Factorized joint-cell AIPW (2.2 E5): the factorized cell propensities and cell means are
//! checked against a known joint law enumerated by hand, the family keeps its supported cells
//! when one is empty or rare, ordering sensitivity is reported, a declared `antecedent-learn`
//! learner supplies the nuisances out of fold (point only), the factorization reproduces the
//! empirical joint frequencies of a hand-counted law, and every closed route (undeclared
//! learner, lasso, interval, covariance) refuses with its reason code.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test fixtures index small literals and build tables element by element"
)]

use antecedent_core::{ExecutionContext, StreamDomain, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{
    CellStatus, EstimationError, FactorizedJointConfig, FactorizedJointFit, JointContrast,
    LearnerSpec, LinearSpec, RidgeTuning, ScoreTable, declared_joint_nuisance,
    fit_factorized_joint_cells, orderings_for, provenance_withholds_interval,
    refuse_joint_inference, refuse_joint_learner_inference,
};
use antecedent_kernels::standard_normal;

/// `P(cell | z)` for three binary components (cell bit `j` is `t_j`), enumerated by hand: each
/// row sums to one. Row 0 is `z = 0`, row 1 is `z = 1`.
fn law_three() -> Vec<Vec<f64>> {
    vec![
        vec![0.20, 0.10, 0.15, 0.05, 0.15, 0.10, 0.15, 0.10],
        vec![0.05, 0.15, 0.10, 0.20, 0.10, 0.15, 0.05, 0.20],
    ]
}

/// Two components, every cell populated.
fn law_two() -> Vec<Vec<f64>> {
    vec![vec![0.4, 0.2, 0.2, 0.2], vec![0.1, 0.3, 0.2, 0.4]]
}

/// Two components, the joint cell (1, 1) has probability zero.
fn law_empty_cell() -> Vec<Vec<f64>> {
    vec![vec![0.4, 0.3, 0.3, 0.0], vec![0.2, 0.4, 0.4, 0.0]]
}

struct Sample {
    z: Vec<f64>,
    t: Vec<Vec<f64>>,
    y: Vec<f64>,
}

/// `z ~ Bernoulli(1/2)`, the cell drawn from `law[z]`, and
/// `Y = 0.4 cell + 1.0 t0 t1 + 0.5 z + 0.3 N(0, 1)`, so `E[Y^{do(cell)}] = 0.4 cell + t0 t1 + 0.25`.
fn draw(n: usize, seed: u64, law: &[Vec<f64>], k: usize) -> Sample {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xE5);
    let mut sample = Sample { z: Vec::new(), t: vec![Vec::new(); k], y: Vec::new() };
    for _ in 0..n {
        let z = rng.next_f64() < 0.5;
        let probs = &law[usize::from(z)];
        let u = rng.next_f64();
        let mut cumulative = 0.0;
        let mut cell = probs.iter().rposition(|&p| p > 0.0).unwrap();
        for (c, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                cell = c;
                break;
            }
        }
        sample.z.push(f64::from(u8::from(z)));
        for (j, column) in sample.t.iter_mut().enumerate() {
            column.push(f64::from(u8::try_from((cell >> j) & 1).unwrap()));
        }
        let t0t1 = f64::from(u8::from(cell & 3 == 3));
        sample.y.push(
            0.4 * cell as f64
                + t0t1
                + 0.5 * f64::from(u8::from(z))
                + 0.3 * standard_normal(&mut rng),
        );
    }
    sample
}

fn frame(sample: &Sample, extra: &[(&str, Vec<f64>)]) -> TabularData {
    let mut columns: Vec<(String, Vec<f64>)> =
        sample.t.iter().enumerate().map(|(j, t)| (format!("t{j}"), t.clone())).collect();
    columns.push(("y".to_string(), sample.y.clone()));
    columns.push(("z".to_string(), sample.z.clone()));
    columns.extend(extra.iter().map(|(name, values)| ((*name).to_string(), values.clone())));
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    TabularData::from_f64_columns(borrowed).unwrap()
}

fn id(data: &TabularData, name: &str) -> VariableId {
    data.schema().id_of(name).unwrap()
}

fn config() -> FactorizedJointConfig {
    let mut config = FactorizedJointConfig::new(RidgeTuning::default());
    config.seed = 7;
    config
}

fn run(
    data: &TabularData,
    k: usize,
    adjustment: &[&str],
    orderings: &[Vec<usize>],
    config: &FactorizedJointConfig,
) -> Result<FactorizedJointFit, EstimationError> {
    let treatments: Vec<VariableId> = (0..k).map(|j| id(data, &format!("t{j}"))).collect();
    let adjustment: Vec<VariableId> = adjustment.iter().map(|name| id(data, name)).collect();
    fit_factorized_joint_cells(
        data,
        &treatments,
        id(data, "y"),
        &adjustment,
        orderings,
        config,
        &ExecutionContext::for_tests(3),
    )
}

fn refused(error: EstimationError) -> (&'static str, String) {
    let EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    (code, message)
}

fn estimate_of(fit: &FactorizedJointFit, cell: usize) -> f64 {
    match &fit.cells[cell].status {
        CellStatus::Supported(estimate) => estimate.estimate,
        CellStatus::Unsupported(refusal) => panic!("cell {cell} unsupported: {refusal:?}"),
    }
}

/// The factorized cell propensities and cell means match the hand-enumerated joint law, the
/// enumerated propensities normalize to one, and the family is stable across every ordering.
#[test]
fn factorized_propensities_and_means_match_the_enumerated_law() {
    let law = law_three();
    let n = 12_000;
    let sample = draw(n, 11, &law, 3);
    let data = frame(&sample, &[]);
    let orderings = orderings_for(3, &[0, 1, 2], true).unwrap();
    assert_eq!(orderings.len(), 6);
    let fit = run(&data, 3, &["z"], &orderings, &config()).unwrap();

    // Oracle 1: cross-fitted cell propensities against the enumerated law, row by row.
    let propensities = &fit.scores.propensities;
    assert_eq!(fit.scores.n_columns(), 8);
    for (j, column) in fit.scores.columns.iter().enumerate() {
        let arm = column.arm as usize;
        let mut worst = 0.0_f64;
        for i in 0..n {
            let oracle = law[usize::from(sample.z[i] > 0.5)][arm];
            worst = worst.max((propensities[j * n + i] - oracle).abs());
        }
        assert!(worst < 0.05, "cell {arm}: worst propensity error {worst}");
    }

    // Oracle 2: the true cell means 0.4 c + 1{t0 = t1 = 1} + 0.25.
    for cell in 0..8 {
        let truth = 0.4 * cell as f64 + f64::from(u8::from(cell & 3 == 3)) + 0.25;
        let estimate = estimate_of(&fit, cell);
        assert!((estimate - truth).abs() < 0.05, "cell {cell}: {estimate} vs {truth}");
        let CellStatus::Supported(report) = &fit.cells[cell].status else { unreachable!() };
        assert!(report.ess > 100.0 && report.propensity_min > 0.0);
    }

    // Normalization holds exactly for every ordering, and every cell was enumerable.
    assert_eq!(fit.normalization.len(), 6);
    for check in &fit.normalization {
        assert_eq!(check.cells_enumerated, 8);
        assert!(check.max_abs_error.unwrap() < 1e-9, "{check:?}");
    }

    // Ordering sensitivity: every ordering supports every cell and they agree closely.
    assert_eq!(fit.sensitivity.orderings, orderings);
    for spread in &fit.sensitivity.cells {
        assert!(spread.estimates.iter().all(Option::is_some));
        assert!(spread.spread.unwrap() < 0.05, "{spread:?}");
    }
    assert!(!fit.sensitivity.disagreement);

    // A contrast is the difference of the reported cell means, and the three-component
    // family has no 2x2 interaction.
    let contrast = fit.contrast_point(JointContrast::CellMinusControl(7)).unwrap();
    assert!((contrast - (estimate_of(&fit, 7) - estimate_of(&fit, 0))).abs() < 1e-9);
    assert!((contrast - 3.8).abs() < 0.1, "{contrast}");
    let (code, message) = refused(fit.contrast_point(JointContrast::Interaction).unwrap_err());
    assert_eq!(code, "joint_cell_unsupported");
    assert!(message.contains("joint_cells.contrast_cell_unsupported"), "{message}");
}

/// Two components with a genuine interaction: the 2x2 interaction contrast recovers 1.
#[test]
fn the_two_by_two_interaction_recovers_the_known_value() {
    let sample = draw(10_000, 12, &law_two(), 2);
    let data = frame(&sample, &[]);
    let fit = run(&data, 2, &["z"], &orderings_for(2, &[0, 1], true).unwrap(), &config()).unwrap();
    let interaction = fit.contrast_point(JointContrast::Interaction).unwrap();
    let by_hand =
        estimate_of(&fit, 0) - estimate_of(&fit, 1) - estimate_of(&fit, 2) + estimate_of(&fit, 3);
    assert!((interaction - by_hand).abs() < 1e-9);
    assert!((interaction - 1.0).abs() < 0.1, "{interaction}");
}

/// The score table is the artifact: its provenance marks the penalized propensity (so no
/// interval or covariance is published) and it round-trips through the wire form with the
/// same scores, columns and folds.
#[test]
fn the_score_table_is_a_point_and_score_artifact_that_round_trips() {
    let sample = draw(3_000, 13, &law_two(), 2);
    let data = frame(&sample, &[]);
    let fit = run(&data, 2, &["z"], &orderings_for(2, &[0, 1], false).unwrap(), &config()).unwrap();
    let table = &fit.scores;
    assert!(provenance_withholds_interval(&table.nuisance_provenance));
    assert!(table.nuisance_provenance.starts_with("joint_cell.factorized.crossfit"));
    assert_eq!(table.propensity_clip, Some(0.01));
    assert_eq!(table.intervened.len(), 2);
    let back = ScoreTable::from_wire(table.to_wire()).unwrap();
    assert_eq!(back.scores, table.scores);
    assert_eq!(back.columns, table.columns);
    assert_eq!(back.fold_ids, table.fold_ids);
    assert_eq!(back.propensities, table.propensities);
    assert_eq!(back.nuisance_provenance, table.nuisance_provenance);
    // The retained scores average to the reported cell means.
    for (j, column) in table.columns.iter().enumerate() {
        let mean = table.column(j).unwrap().iter().sum::<f64>() / table.n_rows as f64;
        assert!((mean - estimate_of(&fit, column.arm as usize)).abs() < 1e-12);
    }
    let (code, message) = refused(refuse_joint_inference());
    assert_eq!(code, "penalized_interval_not_licensed");
    assert!(message.contains("joint_cells.interval_withheld"), "{message}");
}

/// A family with one empty cell keeps the other cells: the empty cell is refused alone with
/// `arm_not_populated`, its siblings' constant conditionals are counted, and contrasts that
/// need it refuse while the rest evaluate.
#[test]
fn a_family_with_one_empty_cell_keeps_the_rest() {
    let sample = draw(8_000, 14, &law_empty_cell(), 2);
    let data = frame(&sample, &[]);
    let fit = run(&data, 2, &["z"], &orderings_for(2, &[0, 1], true).unwrap(), &config()).unwrap();
    for cell in 0..3 {
        let truth = 0.4 * cell as f64 + 0.25;
        assert!((estimate_of(&fit, cell) - truth).abs() < 0.06, "cell {cell}");
    }
    assert_eq!(fit.cells[3].rows, 0);
    let CellStatus::Unsupported(refusal) = &fit.cells[3].status else {
        panic!("the empty cell must be refused")
    };
    assert_eq!(refusal.code, "arm_not_populated");
    assert_eq!(refusal.detail, "joint_cells.cell_empty");
    assert!(fit.degenerate_conditionals > 0);
    assert_eq!(fit.scores.n_columns(), 3);
    assert!((fit.contrast_point(JointContrast::CellMinusControl(1)).unwrap() - 0.4).abs() < 0.06);
    for contrast in [JointContrast::CellMinusControl(3), JointContrast::Interaction] {
        let (code, _) = refused(fit.contrast_point(contrast).unwrap_err());
        assert_eq!(code, "joint_cell_unsupported");
    }
    // The empty cell is unsupported under every ordering, so it is not flagged as disagreement.
    assert!(!fit.sensitivity.cells[3].flagged);
    for check in &fit.normalization {
        assert!(check.max_abs_error.unwrap() < 1e-9);
    }
}

/// A rare joint cell (three rows) is refused individually because its outcome model has no
/// residual degrees of freedom; cells whose conditionals do not pass through it stay.
#[test]
fn a_rare_joint_cell_is_refused_individually() {
    let mut sample = draw(1_200, 15, &law_empty_cell(), 2);
    for i in 0..3 {
        sample.t[0][i] = 1.0;
        sample.t[1][i] = 1.0;
    }
    let data = frame(&sample, &[]);
    let fit = run(&data, 2, &["z"], &orderings_for(2, &[0, 1], false).unwrap(), &config()).unwrap();
    assert_eq!(fit.cells[3].rows, 3);
    let CellStatus::Unsupported(refusal) = &fit.cells[3].status else {
        panic!("the rare cell must be refused")
    };
    assert_eq!(refusal.code, "joint_cell_unsupported");
    assert_eq!(refusal.detail, "joint_cells.outcome_model_unsupported");
    for cell in [0, 2] {
        assert!(matches!(fit.cells[cell].status, CellStatus::Supported(_)), "cell {cell}");
    }
}

/// Two exactly duplicated treatment columns can only take cells 0 and 3; the impossible cells
/// are refused individually and the two real cells are estimated.
#[test]
fn exact_duplicate_treatments_leave_two_cells() {
    let n = 4_000;
    let mut rng = ExecutionContext::for_tests(16).rng.stream_for(StreamDomain::Estimate, 0xD5);
    let (mut z, mut t, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let zi = rng.next_f64() < 0.5;
        let ti = rng.next_f64() < if zi { 0.7 } else { 0.3 };
        z.push(f64::from(u8::from(zi)));
        t.push(f64::from(u8::from(ti)));
        y.push(
            f64::from(u8::from(ti))
                + 0.5 * f64::from(u8::from(zi))
                + 0.3 * standard_normal(&mut rng),
        );
    }
    let sample = Sample { z, t: vec![t.clone(), t], y };
    let data = frame(&sample, &[]);
    let fit = run(&data, 2, &["z"], &orderings_for(2, &[0, 1], true).unwrap(), &config()).unwrap();
    assert!((estimate_of(&fit, 0) - 0.25).abs() < 0.05);
    assert!((estimate_of(&fit, 3) - 1.25).abs() < 0.05);
    for cell in [1, 2] {
        let CellStatus::Unsupported(refusal) = &fit.cells[cell].status else {
            panic!("cell {cell} cannot occur and must be refused")
        };
        assert_eq!(refusal.code, "arm_not_populated");
    }
}

/// A rank-deficient adjustment design (an exact duplicate covariate) leaves no outcome model
/// that can be fit; the family refuses rather than dropping the column.
#[test]
fn a_rank_deficient_design_is_refused_without_dropping_a_column() {
    let sample = draw(2_000, 17, &law_two(), 2);
    let data = frame(&sample, &[("z_copy", sample.z.clone())]);
    let error =
        run(&data, 2, &["z", "z_copy"], &orderings_for(2, &[0, 1], false).unwrap(), &config())
            .unwrap_err();
    let (code, message) = refused(error);
    assert_eq!(code, "joint_cell_unsupported");
    assert!(message.contains("joint_cells.no_supported_cell"), "{message}");
}

/// Zero ordering tolerance flags any cell whose estimates differ at all across orderings, and
/// the tolerance is reported.
#[test]
fn ordering_disagreement_is_flagged_against_the_declared_tolerance() {
    let sample = draw(3_000, 18, &law_two(), 2);
    let data = frame(&sample, &[]);
    let mut strict = config();
    strict.ordering_tolerance_sd = 0.0;
    let orderings = orderings_for(2, &[0, 1], true).unwrap();
    let fit = run(&data, 2, &["z"], &orderings, &strict).unwrap();
    assert!(fit.sensitivity.tolerance.abs() < f64::EPSILON);
    assert!(fit.sensitivity.disagreement);
    assert!(fit.sensitivity.cells.iter().any(|c| c.flagged && c.spread.unwrap() > 0.0));
    let lax = FactorizedJointConfig { ordering_tolerance_sd: 5.0, ..config() };
    let fit = run(&data, 2, &["z"], &orderings, &lax).unwrap();
    assert!(fit.sensitivity.tolerance > 0.0 && !fit.sensitivity.disagreement);
}

/// The declared ordering is first, every permutation appears once, and a malformed ordering
/// is refused.
#[test]
fn orderings_are_validated_and_enumerated() {
    let all = orderings_for(3, &[2, 0, 1], true).unwrap();
    assert_eq!(all[0], vec![2, 0, 1]);
    assert_eq!(all.len(), 6);
    let mut sorted = all.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 6);
    assert_eq!(orderings_for(2, &[1, 0], false).unwrap(), vec![vec![1, 0]]);
    for bad in [vec![0, 0], vec![0], vec![0, 2], vec![0, 1, 2]] {
        let (code, message) = refused(orderings_for(2, &bad, false).unwrap_err());
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("joint_cells.ordering"), "{message}");
    }
}

/// Declarations outside the bounded cell are refused before any fit: more than three
/// components, a repeated or oversized ordering list, an invalid tuning, a non-binary
/// treatment, and a cancelled context.
#[test]
fn declarations_outside_the_bounded_cell_are_refused() {
    let sample = draw(600, 19, &law_two(), 2);
    let data = frame(&sample, &[]);
    let ok = orderings_for(2, &[0, 1], false).unwrap();
    let t0 = id(&data, "t0");
    let t1 = id(&data, "t1");
    let y = id(&data, "y");
    let z = id(&data, "z");
    let ctx = ExecutionContext::for_tests(3);
    let fit =
        |treatments: &[VariableId], orderings: &[Vec<usize>], config: &FactorizedJointConfig| {
            fit_factorized_joint_cells(&data, treatments, y, &[z], orderings, config, &ctx)
        };

    let (_, message) = refused(fit(&[t0, t1, t0, t1], &ok, &config()).unwrap_err());
    assert!(message.contains("joint_cells.component_count"), "{message}");
    let (_, message) = refused(fit(&[t0, t1], &[vec![0, 1], vec![0, 1]], &config()).unwrap_err());
    assert!(message.contains("joint_cells.ordering"), "{message}");
    let (_, message) = refused(fit(&[t0, t0], &ok, &config()).unwrap_err());
    assert!(message.contains("joint_cells.columns"), "{message}");
    for edit in [
        FactorizedJointConfig { folds: 1, ..config() },
        FactorizedJointConfig { clip: 0.0, ..config() },
        FactorizedJointConfig { clip: 0.5, ..config() },
        FactorizedJointConfig { min_cell_ess: 0.5, ..config() },
        FactorizedJointConfig { normalization_tolerance: 0.0, ..config() },
        FactorizedJointConfig { ordering_tolerance_sd: -1.0, ..config() },
    ] {
        let (code, message) = refused(fit(&[t0, t1], &ok, &edit).unwrap_err());
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("joint_cells.config"), "{message}");
    }

    let mut bad = sample;
    bad.t[0][5] = 2.0;
    let bad_data = frame(&bad, &[]);
    let (_, message) = refused(
        fit_factorized_joint_cells(
            &bad_data,
            &[id(&bad_data, "t0"), id(&bad_data, "t1")],
            id(&bad_data, "y"),
            &[id(&bad_data, "z")],
            &ok,
            &config(),
            &ctx,
        )
        .unwrap_err(),
    );
    assert!(message.contains("joint_cells.not_binary"), "{message}");

    let cancelled = ExecutionContext::for_tests(3);
    cancelled.cancellation.cancel();
    let (code, message) = refused(
        fit_factorized_joint_cells(&data, &[t0, t1], y, &[z], &ok, &config(), &cancelled)
            .unwrap_err(),
    );
    assert_eq!(code, "cancelled_no_claim");
    assert!(message.contains("joint_cells.cancelled"), "{message}");
}

/// Ridge-logistic is the default route, `random_forest` and `gradient_boosted_trees` declare a
/// learner, and lasso, an undeclared machine-learning name, an unknown name and an interval or
/// covariance request over either route are closed with their registered reason codes.
#[test]
fn undeclared_learners_lasso_and_intervals_are_closed() {
    assert_eq!(declared_joint_nuisance("ridge_logistic").unwrap(), None);
    assert!(matches!(
        declared_joint_nuisance("random_forest").unwrap(),
        Some(LearnerSpec::RandomForest(_))
    ));
    assert!(matches!(
        declared_joint_nuisance("gradient_boosted_trees").unwrap(),
        Some(LearnerSpec::GradientBoostedTrees(_))
    ));
    let (code, _) = refused(declared_joint_nuisance("lasso").unwrap_err());
    assert_eq!(code, "selection_inference_not_licensed");
    for provider in ["ml", "gradient_boosting", "neural_network", "forest", "boosting"] {
        let (code, message) = refused(declared_joint_nuisance(provider).unwrap_err());
        assert_eq!(code, "ml_nuisance_not_licensed");
        assert!(message.contains("joint_cells.ml_learner_not_declared"), "{message}");
    }
    let (code, message) = refused(declared_joint_nuisance("typo").unwrap_err());
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("joint_cells.nuisance_provider"), "{message}");
    let (code, _) = refused(refuse_joint_inference());
    assert_eq!(code, "penalized_interval_not_licensed");
    let (code, message) = refused(refuse_joint_learner_inference());
    assert_eq!(code, "ml_nuisance_not_licensed");
    assert!(message.contains("joint_cells.ml_interval_withheld"), "{message}");
    assert!(message.contains("remainder and coverage license"), "{message}");
}

fn learner_config() -> FactorizedJointConfig {
    FactorizedJointConfig { learner: Some(LearnerSpec::Linear(LinearSpec {})), ..config() }
}

/// A declared learner (OLS outcome models and unpenalized-logistic conditionals through the
/// `antecedent-learn` contract) recovers the known cell means, interaction and propensities.
/// Both nuisance models are correctly specified here (`z` is binary, so a logistic in `z` is
/// saturated and the outcome is linear in `z`), so the AIPW cell mean has standard error near
/// `0.17 / sqrt(n p_cell)`, about 0.004 at n = 12 000 with `p_cell >= 0.15`; the 0.05 and 0.1
/// tolerances are an order of magnitude wider and exist to keep the test from being flaky,
/// not to claim a rate. The table is marked so no interval is published.
#[test]
fn a_declared_learner_recovers_the_known_effect_point_only() {
    let law = law_two();
    let n = 12_000;
    let sample = draw(n, 31, &law, 2);
    let data = frame(&sample, &[]);
    let orderings = orderings_for(2, &[0, 1], true).unwrap();
    let fit = run(&data, 2, &["z"], &orderings, &learner_config()).unwrap();
    for cell in 0..4 {
        let truth = 0.4 * cell as f64 + f64::from(u8::from(cell == 3)) + 0.25;
        assert!((estimate_of(&fit, cell) - truth).abs() < 0.05, "cell {cell}");
    }
    let interaction = fit.contrast_point(JointContrast::Interaction).unwrap();
    assert!((interaction - 1.0).abs() < 0.1, "{interaction}");
    let propensities = &fit.scores.propensities;
    for (j, column) in fit.scores.columns.iter().enumerate() {
        let worst = (0..n)
            .map(|i| {
                let oracle = law[usize::from(sample.z[i] > 0.5)][column.arm as usize];
                (propensities[j * n + i] - oracle).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(worst < 0.05, "cell {}: worst propensity error {worst}", column.arm);
    }
    for check in &fit.normalization {
        assert!(check.max_abs_error.unwrap() < 1e-9, "{check:?}");
    }
    assert!(!fit.sensitivity.disagreement);

    let table = &fit.scores;
    assert!(provenance_withholds_interval(&table.nuisance_provenance));
    assert!(table.nuisance_provenance.starts_with("joint_cell.factorized.crossfit.learner_prefix"));
    assert!(table.nuisance_provenance.contains(";nuisance=ml_joint_cell;outcome=linear;"));
    assert!(
        table.nuisance_provenance.contains(";learners=linear/faer/"),
        "{}",
        table.nuisance_provenance
    );
    assert!(table.nuisance_provenance.contains(";fold_seed=7;ctx_seed="));
    assert!(!table.nuisance_provenance.contains(";propensity=ridge_logistic"));
    let back = ScoreTable::from_wire(table.to_wire()).unwrap();
    assert_eq!(back.scores, table.scores);
    assert_eq!(back.nuisance_provenance, table.nuisance_provenance);
    assert!(provenance_withholds_interval(&back.nuisance_provenance));
}

/// The same declaration, data, config seed and context replay bit-identically; a different
/// config seed changes the fold plan and the recorded seed.
#[test]
fn seeded_learner_fits_replay_bit_identically() {
    let sample = draw(3_000, 32, &law_two(), 2);
    let data = frame(&sample, &[]);
    let orderings = orderings_for(2, &[0, 1], false).unwrap();
    let a = run(&data, 2, &["z"], &orderings, &learner_config()).unwrap();
    let b = run(&data, 2, &["z"], &orderings, &learner_config()).unwrap();
    assert_eq!(bits(&a.scores.scores), bits(&b.scores.scores));
    assert_eq!(bits(&a.scores.propensities), bits(&b.scores.propensities));
    assert_eq!(a.scores.fold_ids, b.scores.fold_ids);
    assert_eq!(a.scores.nuisance_provenance, b.scores.nuisance_provenance);
    let other = FactorizedJointConfig { seed: 8, ..learner_config() };
    let c = run(&data, 2, &["z"], &orderings, &other).unwrap();
    assert_ne!(a.scores.fold_ids, c.scores.fold_ids);
    assert_ne!(a.scores.nuisance_provenance, c.scores.nuisance_provenance);
    assert!(c.scores.nuisance_provenance.contains(";fold_seed=8;"));
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Declaring a learner is a per-call option: the default ridge route is bit-identical before
/// and after a learner fit in the same process and keeps its ridge provenance.
#[test]
fn declaring_a_learner_leaves_the_default_route_bit_identical() {
    let sample = draw(3_000, 33, &law_two(), 2);
    let data = frame(&sample, &[]);
    let orderings = orderings_for(2, &[0, 1], true).unwrap();
    assert!(config().learner.is_none());
    let before = run(&data, 2, &["z"], &orderings, &config()).unwrap();
    let learned = run(&data, 2, &["z"], &orderings, &learner_config()).unwrap();
    let after = run(&data, 2, &["z"], &orderings, &config()).unwrap();
    assert_eq!(bits(&before.scores.scores), bits(&after.scores.scores));
    assert_eq!(bits(&before.scores.propensities), bits(&after.scores.propensities));
    assert_eq!(before.scores.fold_ids, after.scores.fold_ids);
    assert_eq!(before.scores.nuisance_provenance, after.scores.nuisance_provenance);
    assert!(before.scores.nuisance_provenance.contains(";propensity=ridge_logistic"));
    assert!(!before.scores.nuisance_provenance.contains("ml_joint_cell"));
    assert_ne!(before.scores.scores, learned.scores.scores);
}

/// A learner with no declared identity or no resolvable provider is refused before any fit:
/// `Auto` hides the learner actually fit, and a forest needs the `ml-forest` feature.
#[test]
fn an_auto_or_unavailable_learner_is_refused_before_any_fit() {
    let sample = draw(600, 34, &law_two(), 2);
    let data = frame(&sample, &[]);
    let orderings = orderings_for(2, &[0, 1], false).unwrap();
    let auto = FactorizedJointConfig { learner: Some(LearnerSpec::Auto), ..config() };
    let (code, message) = refused(run(&data, 2, &["z"], &orderings, &auto).unwrap_err());
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("joint_cells.learner"), "{message}");
    #[cfg(not(feature = "ml-forest"))]
    {
        let forest = FactorizedJointConfig {
            learner: declared_joint_nuisance("random_forest").unwrap(),
            ..config()
        };
        let (code, message) = refused(run(&data, 2, &["z"], &orderings, &forest).unwrap_err());
        assert_eq!(code, "route_not_supported");
        assert!(message.contains("joint_cells.learner_unavailable"), "{message}");
    }
}

/// A sample whose rows realize `counts[cell]` exactly (cell bit `j` is component `j`), with no
/// adjustment covariate and a deterministic outcome.
fn counted_sample(counts: &[usize], k: usize) -> Sample {
    let mut sample = Sample { z: Vec::new(), t: vec![Vec::new(); k], y: Vec::new() };
    for (cell, &count) in counts.iter().enumerate() {
        for r in 0..count {
            for (j, column) in sample.t.iter_mut().enumerate() {
                column.push(f64::from(u8::try_from((cell >> j) & 1).unwrap()));
            }
            sample.z.push(0.0);
            sample.y.push(0.4 * cell as f64 + 0.1 * (r % 7) as f64);
        }
    }
    sample
}

/// Independent verification of the factorization. On a hand-counted law with no covariate the
/// prefix-stratum conditionals are saturated (an intercept per observed prefix), and every cell
/// count is a multiple of the five folds, so the cell-stratified fold plan puts exactly
/// `count / 5` of each cell in each fold: a training set then has the same cell proportions as
/// the whole table, and each conditional's out-of-fold probability is the training frequency
/// `count(prefix, value) / count(prefix)`. The oracle is the empirical joint frequency counted
/// directly in the test; the chain-rule product under every ordering must equal it (to the
/// logistic solver's tolerance, `1e-5`), and the enumerated cell propensities must sum to one.
#[test]
fn the_factorized_product_reproduces_the_empirical_joint_frequencies() {
    for (k, counts) in
        [(2, vec![200_usize, 100, 150, 50]), (3, vec![100_usize, 50, 75, 25, 75, 50, 75, 50])]
    {
        let sample = counted_sample(&counts, k);
        let data = frame(&sample, &[]);
        let n = sample.y.len();
        assert_eq!(n, counts.iter().sum::<usize>());
        // Direct count of the joint cells from the treatment columns.
        let mut tally = vec![0usize; 1 << k];
        for i in 0..n {
            let cell: usize = (0..k).map(|j| usize::from(sample.t[j][i] > 0.5) << j).sum();
            tally[cell] += 1;
        }
        assert_eq!(tally, counts);
        let orderings = orderings_for(k, &(0..k).collect::<Vec<_>>(), true).unwrap();
        let fit = run(&data, k, &[], &orderings, &learner_config()).unwrap();
        assert_eq!(fit.scores.n_columns(), 1 << k);
        let propensities = &fit.scores.propensities;
        for (j, column) in fit.scores.columns.iter().enumerate() {
            let oracle = tally[column.arm as usize] as f64 / n as f64;
            let worst =
                (0..n).map(|i| (propensities[j * n + i] - oracle).abs()).fold(0.0_f64, f64::max);
            assert!(worst < 1e-5, "k = {k}, cell {}: {worst}", column.arm);
        }
        for i in 0..n {
            let total: f64 = (0..1 << k).map(|j| propensities[j * n + i]).sum();
            assert!((total - 1.0).abs() < 1e-9, "k = {k}, row {i}: sums to {total}");
        }
        assert_eq!(fit.normalization.len(), orderings.len());
        for check in &fit.normalization {
            assert_eq!(check.cells_enumerated, 1 << k);
            assert!(check.max_abs_error.unwrap() < 1e-9, "{check:?}");
        }
        // Every ordering gives the same estimate of every cell (the identity holds for any
        // ordering on this saturated law).
        for spread in &fit.sensitivity.cells {
            assert!(spread.spread.unwrap() < 1e-4, "{spread:?}");
        }
    }
}

/// A random-forest learner recovers the known means. Runs only in a build with the forest
/// provider (`ml-forest`); the tolerance is loose because a forest's piecewise-constant fit
/// of a binary covariate is exact up to bootstrap noise in the leaf means, a 0.15 band is
/// several standard errors at this size, and no rate is claimed.
#[cfg(feature = "ml-forest")]
#[test]
fn a_random_forest_learner_recovers_the_known_means() {
    let sample = draw(6_000, 35, &law_two(), 2);
    let data = frame(&sample, &[]);
    let config = FactorizedJointConfig {
        learner: declared_joint_nuisance("random_forest").unwrap(),
        ..config()
    };
    let fit = run(&data, 2, &["z"], &orderings_for(2, &[0, 1], false).unwrap(), &config).unwrap();
    for cell in 0..4 {
        let truth = 0.4 * cell as f64 + f64::from(u8::from(cell == 3)) + 0.25;
        assert!((estimate_of(&fit, cell) - truth).abs() < 0.15, "cell {cell}");
    }
    assert!(fit.scores.nuisance_provenance.contains("forest"));
    assert!(provenance_withholds_interval(&fit.scores.nuisance_provenance));
}
