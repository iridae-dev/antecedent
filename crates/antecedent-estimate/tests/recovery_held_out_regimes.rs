//! Whole-procedure validation of experimental recovery on entire held-out
//! intervention regimes.
//!
//! Every case runs the full procedure: simulate clamp experiments from a known
//! cyclic linear truth with non-zero disturbances and large intercepts, fit the
//! recovery, draw its uncertainty law, retain the law with an intervention that
//! no training regime ran, and compile every draw into the certified equilibrium.
//! The prediction is scored against the truth solved independently and against a
//! fresh sample of the held-out regime that the fit never saw. Holding out rows
//! of a known regime would not exercise this; the held-out regimes here clamp
//! sets of units (including joint clamps) that no training experiment clamped.
//!
//! The ordinary tests check recovery and the shape of the reported uncertainty
//! on fixed seeds. The `#[ignore]`d designs score the repeated-sampling coverage
//! of the reported interval for the held-out interventional mean; they are run
//! by the calibration measurement and nothing here records a coverage number.
//! Inadequate designs (uncovered or rank-deficient constraints) and failed
//! replicates are scored as withheld mass, never as a filled-in answer.
#![allow(clippy::cast_precision_loss, reason = "bounded row and draw counts")]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use antecedent_core::{
    CausalRng, DeclaredSamplingDesign, ExecutionContext, ResamplingRule, Value, VariableId,
};
use antecedent_estimate::experimental_fit::{
    BootstrapSpec, DrawStatus, ExperimentRows, ExperimentalFitOptions, ExperimentalRecoveryFit,
    RecoveredEstimate, asymptotic_normal_recovery, bootstrap_experimental_recovery,
    fit_experimental_recovery,
};
use antecedent_estimate::recovery::{
    CertifiedInterventionEvaluation, RecoveryPlanParts, RetainedRecoveryPlan,
    consume_recovery_evidence, consume_retained_recovery_plan,
};
use antecedent_identify::experiment_moments::{RegimeSample, UnitColumn};
use antecedent_identify::experiment_regime::{
    DependencePremise, ExperimentRegime, PopulationId, RegimeContract, RegimeIntervention,
    SourceIdentity, UnitMask,
};
use antecedent_identify::experimental_recovery::{
    DirectCoefficient, DrawAdmissibility, InadmissibleCoefficientDraw,
    RecoveredCoefficientUncertainty, UnresolvedReason,
};
use antecedent_kernels::standard_normal;
use calibration::{CoverageTally, grid_n, grid_seed, map_replicates, n_sim, quantile_interval};

const EVIDENCE: &str = "held-out-regime-evidence";
const SOURCE: &str = "clamp-experiment-recovery";

fn node_at(index: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(index).expect("small node index"))
}

fn coefficient(into: usize, from: usize) -> DirectCoefficient {
    DirectCoefficient { into: node_at(into), from: node_at(from) }
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

/// `x = K x + c + ε`, `K[into * n + from]`, independent disturbances with sd `sd`.
struct Truth {
    k: Vec<f64>,
    c: Vec<f64>,
    sd: Vec<f64>,
}

impl Truth {
    fn n(&self) -> usize {
        self.c.len()
    }

    fn effect(&self, into: usize, from: usize) -> f64 {
        self.k[into * self.n() + from]
    }
}

/// Cycles 0 <-> 1 and 1 <-> 2, plus a negative edge 0 -> 2, with large intercepts.
fn cyclic_truth() -> Truth {
    let mut k = vec![0.0; 9];
    k[1] = 0.3; // 1 -> 0
    k[3] = 0.4; // 0 -> 1
    k[5] = 0.2; // 2 -> 1
    k[6] = -0.3; // 0 -> 2
    k[7] = 0.5; // 1 -> 2
    Truth { k, c: vec![5.0, -3.0, 10.0], sd: vec![1.0, 0.7, 1.2] }
}

/// A null regime and one single-unit clamp per node.
const SINGLE: &[&[usize]] = &[&[], &[0], &[1], &[2]];

/// Overlapping multi-unit clamps: some constraint blocks are overdetermined.
const MULTIPLE: &[&[usize]] = &[&[0, 1], &[1, 2], &[0, 2], &[0]];

/// Joint clamps no `SINGLE` regime ran.
const HELD_OUT_JOINT: &[&[usize]] = &[&[0, 1], &[1, 2], &[0, 2]];

const CLAMP_SD: f64 = 1.5;

fn clamp_mean(unit: usize) -> f64 {
    2.0 + 3.0 * unit as f64
}

/// Values the held-out intervention sets, distinct from every training clamp law.
const PROBE_VALUES: [f64; 3] = [4.0, -2.0, 7.0];

fn solve_dense(mut matrix: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|a, b| matrix[a * n + col].abs().total_cmp(&matrix[b * n + col].abs()))
            .expect("non-empty pivot range");
        for c in 0..n {
            matrix.swap(col * n + c, pivot * n + c);
        }
        rhs.swap(col, pivot);
        for row in (col + 1)..n {
            let factor = matrix[row * n + col] / matrix[col * n + col];
            for c in col..n {
                matrix[row * n + c] -= factor * matrix[col * n + c];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = ((row + 1)..n).map(|c| matrix[row * n + c] * x[c]).sum();
        x[row] = (rhs[row] - tail) / matrix[row * n + row];
    }
    x
}

/// Equilibrium of `truth` with `clamped` held at `values` and every disturbance at its mean.
fn truth_equilibrium(truth: &Truth, clamp: &[(usize, f64)]) -> Vec<f64> {
    let n = truth.n();
    let free: Vec<usize> = (0..n).filter(|unit| clamp.iter().all(|(c, _)| c != unit)).collect();
    let f = free.len();
    let mut x = vec![0.0; n];
    for &(unit, value) in clamp {
        x[unit] = value;
    }
    let mut matrix = vec![0.0; f * f];
    let mut rhs = vec![0.0; f];
    for (p, &row) in free.iter().enumerate() {
        rhs[p] = truth.c[row] + clamp.iter().map(|&(u, v)| truth.effect(row, u) * v).sum::<f64>();
        for (q, &col) in free.iter().enumerate() {
            matrix[p * f + q] = if p == q { 1.0 } else { 0.0 } - truth.effect(row, col);
        }
    }
    if f > 0 {
        let solved = solve_dense(matrix, rhs, f);
        for (p, &row) in free.iter().enumerate() {
            x[row] = solved[p];
        }
    }
    x
}

/// Columns of `rows` draws of the regime that clamps `clamped`.
fn simulate(truth: &Truth, clamped: &[usize], rows: usize, rng: &mut CausalRng) -> Vec<Vec<f64>> {
    let n = truth.n();
    let free: Vec<usize> = (0..n).filter(|unit| !clamped.contains(unit)).collect();
    let f = free.len();
    let mut columns = vec![Vec::with_capacity(rows); n];
    let mut x = vec![0.0; n];
    for _ in 0..rows {
        for &unit in clamped {
            x[unit] = clamp_mean(unit) + CLAMP_SD * standard_normal(rng);
        }
        let mut matrix = vec![0.0; f * f];
        let mut rhs = vec![0.0; f];
        for (p, &row) in free.iter().enumerate() {
            let pushed: f64 = clamped.iter().map(|&unit| truth.effect(row, unit) * x[unit]).sum();
            rhs[p] = truth.c[row] + truth.sd[row] * standard_normal(rng) + pushed;
            for (q, &col) in free.iter().enumerate() {
                matrix[p * f + q] = if p == q { 1.0 } else { 0.0 } - truth.effect(row, col);
            }
        }
        if f > 0 {
            let solved = solve_dense(matrix, rhs, f);
            for (p, &row) in free.iter().enumerate() {
                x[row] = solved[p];
            }
        }
        for (column, value) in columns.iter_mut().zip(&x) {
            column.push(*value);
        }
    }
    columns
}

fn contract(n: usize, designs: &[&[usize]]) -> RegimeContract {
    RegimeContract {
        regimes: designs
            .iter()
            .enumerate()
            .map(|(index, clamped)| ExperimentRegime {
                targets: clamped
                    .iter()
                    .map(|&unit| RegimeIntervention { unit: node_at(unit), value: Value::Int64(0) })
                    .collect(),
                units: (0..n).map(node_at).collect(),
                population: PopulationId::from_raw(1),
                premise: DependencePremise::Independence,
                source: SourceIdentity::from_raw(u64::try_from(index).expect("index") + 1),
            })
            .collect(),
        mask: UnitMask::default(),
    }
}

/// Experiment rows that own their columns.
struct Dataset {
    contract: RegimeContract,
    data: Vec<Vec<Vec<f64>>>,
}

impl Dataset {
    fn simulate(truth: &Truth, designs: &[&[usize]], rows: usize, seed: u64) -> Self {
        let mut rng = CausalRng::from_seed(seed);
        let data = designs.iter().map(|clamped| simulate(truth, clamped, rows, &mut rng)).collect();
        Self { contract: contract(truth.n(), designs), data }
    }

    fn from_columns(n: usize, designs: &[&[usize]], data: Vec<Vec<Vec<f64>>>) -> Self {
        Self { contract: contract(n, designs), data }
    }

    fn with_rows<R>(&self, body: impl FnOnce(&ExperimentRows<'_>) -> R) -> R {
        let columns: Vec<Vec<UnitColumn<'_>>> = self
            .data
            .iter()
            .map(|regime| {
                regime
                    .iter()
                    .enumerate()
                    .map(|(unit, values)| UnitColumn {
                        unit: node_at(unit),
                        values: values.as_slice(),
                    })
                    .collect()
            })
            .collect();
        let samples: Vec<RegimeSample<'_>> = columns
            .iter()
            .zip(&self.contract.regimes)
            .map(|(regime, declared)| RegimeSample {
                columns: regime.as_slice(),
                source: declared.source,
            })
            .collect();
        body(&ExperimentRows { contract: &self.contract, samples: &samples })
    }

    fn fit(&self) -> ExperimentalRecoveryFit {
        self.with_rows(|rows| {
            fit_experimental_recovery(rows, &ExperimentalFitOptions::default())
                .expect("the rows fit")
        })
    }

    fn bootstrap(&self, replicates: usize, batch: usize, seed: u64) -> Option<RecoveredEstimate> {
        let spec = BootstrapSpec {
            design: DeclaredSamplingDesign::IndependentSubjectTrajectories,
            rule: ResamplingRule::RowSubjectIid,
            replicates,
            seed,
            sampling_units: None,
            block_length: None,
            batch,
        };
        self.with_rows(|rows| {
            bootstrap_experimental_recovery(
                rows,
                &ExperimentalFitOptions::default(),
                &spec,
                &ExecutionContext::for_tests(0),
            )
            .ok()
        })
    }
}

/// Retain `law` with a hard clamp that no training regime ran.
fn plan_for(
    law: &RecoveredCoefficientUncertainty,
    clamp: &[(usize, f64)],
    outcome: usize,
) -> RetainedRecoveryPlan {
    RetainedRecoveryPlan::try_new(RecoveryPlanParts {
        uncertainty: law.clone(),
        clamp_nodes: clamp.iter().map(|&(unit, _)| unit).collect(),
        clamp_values: clamp.iter().map(|&(_, value)| value).collect(),
        outcome,
        source_id: SOURCE.to_owned(),
        certifying_evidence: vec![EVIDENCE.to_owned()],
    })
    .expect("the clamp and outcome are coordinates of the law")
}

fn evaluate(
    law: &RecoveredCoefficientUncertainty,
    clamp: &[(usize, f64)],
    outcome: usize,
) -> CertifiedInterventionEvaluation {
    consume_retained_recovery_plan(&plan_for(law, clamp, outcome))
}

fn clamp_at(units: &[usize], value_of: impl Fn(usize) -> f64) -> Vec<(usize, f64)> {
    units.iter().map(|&unit| (unit, value_of(unit))).collect()
}

fn free_outcomes(clamped: &[usize]) -> impl Iterator<Item = usize> + '_ {
    (0..3).filter(|unit| !clamped.contains(unit))
}

/// Spread of the per-draw interventional mean: the reported coefficient uncertainty.
fn draw_mean_spread(evaluation: &CertifiedInterventionEvaluation) -> f64 {
    let mixture = evaluation.outcome_mixture().expect("certified draws");
    assert!(mixture.certified_mass > 0.0);
    mixture.between_draw_variance.sqrt()
}

/// Score one held-out prediction against the truth solved independently.
fn assert_matches_truth(
    law: &RecoveredCoefficientUncertainty,
    truth: &Truth,
    clamp: &[(usize, f64)],
    outcome: usize,
    sigmas: f64,
) -> f64 {
    let expected = truth_equilibrium(truth, clamp)[outcome];
    let evaluation = evaluate(law, clamp, outcome);
    close(evaluation.withheld_mass(), 0.0, 1e-12, "withheld mass");
    close(evaluation.total_mass(), 1.0, 1e-9, "total mass");
    let mixture = evaluation.outcome_mixture().expect("certified draws");
    close(mixture.certified_mass, 1.0, 1e-9, "certified mass");
    let spread = draw_mean_spread(&evaluation);
    assert!(
        spread > 1e-4 && spread < 0.5,
        "coefficient uncertainty of the held-out mean {spread} at clamp {clamp:?}, outcome {outcome}"
    );
    close(
        mixture.mean,
        expected,
        sigmas * spread + 1e-3,
        &format!("held-out mean at clamp {clamp:?}, outcome {outcome}"),
    );
    spread
}

#[test]
fn single_clamp_recovery_predicts_entire_held_out_joint_regimes() {
    let truth = cyclic_truth();
    let fit = Dataset::simulate(&truth, SINGLE, 20_000, 101).fit();
    assert!(fit.goal_closed());
    let estimate = asymptotic_normal_recovery(&fit, 600, 7).expect("draws");
    for clamped in HELD_OUT_JOINT {
        assert!(
            !SINGLE.contains(clamped),
            "{clamped:?} is a training regime; the check would hold out rows, not a regime"
        );
        let clamp = clamp_at(clamped, |unit| PROBE_VALUES[unit]);
        for outcome in free_outcomes(clamped) {
            assert_matches_truth(estimate.law(), &truth, &clamp, outcome, 5.0);
        }
    }
}

#[test]
fn held_out_prediction_matches_a_fresh_sample_of_the_unseen_regime() {
    let truth = cyclic_truth();
    let fit = Dataset::simulate(&truth, SINGLE, 20_000, 103).fit();
    let estimate = asymptotic_normal_recovery(&fit, 600, 11).expect("draws");
    let mut rng = CausalRng::from_seed(990_001);
    for clamped in HELD_OUT_JOINT {
        // The mean of every free unit is linear in the clamped units, so the sample
        // mean of the regime with clamp law mean `clamp_mean` is the equilibrium at
        // `clamp_mean`. The sample comes from a stream the fit never saw.
        let columns = simulate(&truth, clamped, 40_000, &mut rng);
        let clamp = clamp_at(clamped, clamp_mean);
        for outcome in free_outcomes(clamped) {
            let column = &columns[outcome];
            let n = column.len() as f64;
            let sample_mean = column.iter().sum::<f64>() / n;
            let sample_variance =
                column.iter().map(|v| (v - sample_mean) * (v - sample_mean)).sum::<f64>() / n;
            let sample_se = (sample_variance / n).sqrt();
            let evaluation = evaluate(estimate.law(), &clamp, outcome);
            let mixture = evaluation.outcome_mixture().expect("certified draws");
            let spread = mixture.between_draw_variance.sqrt();
            let combined = (spread * spread + sample_se * sample_se).sqrt();
            close(
                mixture.mean,
                sample_mean,
                5.0 * combined + 1e-3,
                &format!("prediction against the unseen regime {clamped:?}, outcome {outcome}"),
            );
        }
    }
}

#[test]
fn overlapping_joint_training_predicts_held_out_single_clamps() {
    let truth = cyclic_truth();
    let fit = Dataset::simulate(&truth, MULTIPLE, 20_000, 107).fit();
    assert!(fit.goal_closed());
    assert!(fit.block_fits().iter().any(|block| block.rows > block.column_rank));
    let estimate = asymptotic_normal_recovery(&fit, 600, 13).expect("draws");
    let held_out: &[&[usize]] = &[&[1], &[2]];
    for clamped in held_out {
        assert!(!MULTIPLE.contains(clamped));
        let clamp = clamp_at(clamped, |unit| PROBE_VALUES[unit]);
        for outcome in free_outcomes(clamped) {
            assert_matches_truth(estimate.law(), &truth, &clamp, outcome, 5.0);
        }
    }
}

#[test]
fn held_out_uncertainty_shrinks_with_the_square_root_of_the_rows() {
    let truth = cyclic_truth();
    let clamped: &[usize] = &[0, 1];
    let clamp = clamp_at(clamped, |unit| PROBE_VALUES[unit]);
    let spread_at = |rows: usize, seed: u64| {
        let fit = Dataset::simulate(&truth, SINGLE, rows, seed).fit();
        let estimate = asymptotic_normal_recovery(&fit, 800, 5).expect("draws");
        assert_matches_truth(estimate.law(), &truth, &clamp, 2, 5.0)
    };
    let small = spread_at(2_000, 211);
    let large = spread_at(32_000, 213);
    let ratio = small / large;
    assert!((3.0..=5.5).contains(&ratio), "16x the rows should cut the spread 4x: {ratio}");
}

#[test]
fn bootstrap_law_predicts_a_held_out_joint_regime_with_comparable_uncertainty() {
    let truth = cyclic_truth();
    let data = Dataset::simulate(&truth, SINGLE, 1_500, 301);
    let fit = data.fit();
    let asymptotic = asymptotic_normal_recovery(&fit, 600, 3).expect("draws");
    let bootstrap = data.bootstrap(300, 16, 17).expect("bootstrap");
    let clamped: &[usize] = &[0, 2];
    let clamp = clamp_at(clamped, |unit| PROBE_VALUES[unit]);
    let wald = assert_matches_truth(asymptotic.law(), &truth, &clamp, 1, 5.0);
    let resampled = assert_matches_truth(bootstrap.law(), &truth, &clamp, 1, 5.0);
    let ratio = resampled / wald;
    assert!((0.7..=1.45).contains(&ratio), "bootstrap spread {resampled} against {wald}");
}

/// A design that never clamps node 2 leaves the equations into nodes 0 and 1 without
/// the pair coverage they need. Only the equation into 2 is covered.
fn uncovered_design() -> (Dataset, Vec<(usize, f64)>, usize) {
    let designs: &[&[usize]] = &[&[], &[0], &[1]];
    let data = Dataset::simulate(&cyclic_truth(), designs, 4_000, 401);
    (data, clamp_at(&[0, 1], |unit| PROBE_VALUES[unit]), 2)
}

/// Two regimes that clamp node 0 and node 1 with node 1 an exact multiple of node 0
/// in the first regime: the block into node 2 has rank one.
fn rank_deficient_design() -> (Dataset, Vec<(usize, f64)>, usize) {
    let mut rng = CausalRng::from_seed(97);
    let rows = 400;
    let mut regime0 = vec![Vec::new(), Vec::new(), Vec::new()];
    let mut regime1 = vec![Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..rows {
        let x0 = 1.0 + standard_normal(&mut rng);
        let x1 = 2.0 * x0;
        let x2 = 0.25 * x0 + 0.5 * x1 + 4.0 + 0.1 * standard_normal(&mut rng);
        regime0[0].push(x0);
        regime0[1].push(x1);
        regime0[2].push(x2);
        let y1 = -1.0 + standard_normal(&mut rng);
        let y0 = 0.5 * y1;
        let y2 = 0.25 * y0 + 0.5 * y1 + 4.0 + 0.1 * standard_normal(&mut rng);
        regime1[0].push(y0);
        regime1[1].push(y1);
        regime1[2].push(y2);
    }
    let designs: &[&[usize]] = &[&[0], &[1]];
    (Dataset::from_columns(3, designs, vec![regime0, regime1]), vec![(2, 1.0)], 0)
}

#[test]
fn inadequate_designs_withhold_every_draw_from_the_held_out_decision() {
    for (name, (data, clamp, outcome)) in
        [("uncovered", uncovered_design()), ("rank deficient", rank_deficient_design())]
    {
        let fit = data.fit();
        assert!(!fit.goal_closed(), "{name}: the design cannot close the goal");
        assert!(!fit.unresolved().is_empty(), "{name}: unresolved directions are kept");
        let estimate = asymptotic_normal_recovery(&fit, 64, 9).expect("draws");
        for (draw, status) in estimate.law().draws().iter().zip(estimate.draw_status()) {
            assert_eq!(
                draw.admissibility(),
                DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::UnresolvedDirections),
                "{name}"
            );
            assert!(matches!(status, DrawStatus::Unresolved { .. }), "{name}: {status:?}");
        }
        close(estimate.law().inadmissible_mass(), 1.0, 1e-9, "inadmissible mass");
        let plan = plan_for(estimate.law(), &clamp, outcome);
        let evaluation = consume_retained_recovery_plan(&plan);
        close(evaluation.withheld_mass(), 1.0, 1e-9, &format!("{name}: withheld mass"));
        assert!(evaluation.outcome_mixture().is_none(), "{name}: no interval from no draw");
        for draw in evaluation.draws() {
            assert_eq!(
                draw.inadmissible_reason(),
                Some(InadmissibleCoefficientDraw::UnresolvedDirections)
            );
        }
        let decision = consume_recovery_evidence(&plan, EVIDENCE, 0.0).expect("finite threshold");
        assert_eq!(decision.choice(), None, "{name}: no action from withheld mass");
        close(decision.withheld_mass(), 1.0, 1e-9, &format!("{name}: decision withheld mass"));
        close(decision.mass_account().total(), 1.0, 1e-9, "decision mass");
    }
}

#[test]
fn an_uncovered_design_resolves_only_the_covered_equation_and_names_the_rest() {
    let truth = cyclic_truth();
    let (data, _, _) = uncovered_design();
    let fit = data.fit();
    let missing: Vec<VariableId> = fit
        .unresolved()
        .iter()
        .filter_map(|direction| match &direction.reason {
            UnresolvedReason::IncompletePairCoverage { target, missing_clamps } => {
                assert_eq!(missing_clamps, &vec![node_at(2)]);
                Some(*target)
            }
            UnresolvedReason::DeficientConstraintRank { .. } => None,
        })
        .collect();
    assert_eq!(missing, vec![node_at(0), node_at(1)]);
    // The covered equation is still recovered, to its own sampling error.
    for from in [0, 1] {
        let estimate = fit.coefficient(coefficient(2, from)).expect("the covered equation");
        close(estimate, truth.effect(2, from), 0.05, "covered direct effect");
    }
    for (into, from) in [(0, 1), (0, 2), (1, 0), (1, 2)] {
        assert!(fit.coefficient(coefficient(into, from)).is_none(), "{into} <- {from} was filled");
    }
}

#[test]
fn failed_replicates_are_withheld_mass_in_the_held_out_decision() {
    // Three rows per regime with a repeated clamp level: some resamples make a clamp
    // constant, and that replicate's recovery fails.
    let designs: &[&[usize]] = &[&[0], &[1]];
    let data = Dataset::from_columns(
        2,
        designs,
        vec![
            vec![vec![1.0, 1.0, 2.0], vec![0.4, 0.9, 1.7]],
            vec![vec![0.2, -0.1, 1.1], vec![0.5, 0.5, 3.0]],
        ],
    );
    let estimate = data.bootstrap(40, 4, 17).expect("bootstrap");
    let failed = estimate.failed_mass();
    assert!(failed > 0.0 && failed < 1.0, "failed mass {failed}");
    let plan = plan_for(estimate.law(), &[(0, 1.0)], 1);
    let evaluation = consume_retained_recovery_plan(&plan);
    close(evaluation.total_mass(), 1.0, 1e-12, "total mass");
    assert!(evaluation.withheld_mass() >= failed - 1e-12, "failed mass must stay withheld");
    for (draw, status) in evaluation.draws().iter().zip(estimate.draw_status()) {
        if matches!(status, DrawStatus::Failed { .. }) {
            assert_eq!(draw.inadmissible_reason(), Some(InadmissibleCoefficientDraw::Marked));
        }
    }
    if let Some(mixture) = evaluation.outcome_mixture() {
        close(mixture.certified_mass + mixture.withheld_mass, 1.0, 1e-12, "mixture mass");
        assert!(mixture.withheld_mass >= failed - 1e-12);
    }
    let decision = consume_recovery_evidence(&plan, EVIDENCE, 0.0).expect("finite threshold");
    assert_eq!(decision.choice(), None, "no single action while mass is withheld");
    assert!(decision.withheld_mass() >= failed - 1e-12);
    close(decision.mass_account().total(), 1.0, 1e-12, "decision mass");
}

// -- Calibration designs ------------------------------------------------------------------------
//
// Each replicate simulates fresh clamp experiments, runs the whole recovery, draws the
// uncertainty law, compiles every draw under a joint intervention no training regime ran,
// and reports the equal-tailed interval of the per-draw interventional mean. A replicate
// that withholds any mass reports no interval and counts as a miss. Nothing is measured
// at implementation time.

#[derive(Clone, Copy)]
enum Law {
    AsymptoticNormal,
    Bootstrap,
}

const ASYMPTOTIC_DRAWS: usize = 500;
const BOOTSTRAP_REPLICATES: usize = 200;
const REPORTED_LEVEL: f64 = 0.95;

fn held_out_interval(
    law: Law,
    rows: usize,
    seed: u64,
    clamp: &[(usize, f64)],
    outcome: usize,
) -> Option<(f64, f64)> {
    let data = Dataset::simulate(&cyclic_truth(), SINGLE, rows, seed);
    let fit =
        data.with_rows(|r| fit_experimental_recovery(r, &ExperimentalFitOptions::default()).ok())?;
    let estimate = match law {
        Law::AsymptoticNormal => {
            asymptotic_normal_recovery(&fit, ASYMPTOTIC_DRAWS, seed ^ 0x5eed).ok()?
        }
        Law::Bootstrap => data.bootstrap(BOOTSTRAP_REPLICATES, 16, seed ^ 0x5eed)?,
    };
    let evaluation = evaluate(estimate.law(), clamp, outcome);
    if evaluation.withheld_mass() > 1e-12 {
        return None;
    }
    let means: Vec<f64> = evaluation
        .draws()
        .iter()
        .filter_map(|draw| draw.outcome_moments(outcome))
        .map(|moments| moments.mean)
        .collect();
    quantile_interval(&means, REPORTED_LEVEL)
}

fn held_out_coverage(name: &str, law: Law, seed_base: u64, clamped: &[usize], outcome: usize) {
    let rows = grid_n(400);
    let clamp = clamp_at(clamped, |unit| PROBE_VALUES[unit]);
    let truth = truth_equilibrium(&cyclic_truth(), &clamp)[outcome];
    let mut tally = CoverageTally::new(name, REPORTED_LEVEL);
    let intervals = map_replicates(n_sim(), |replicate| {
        held_out_interval(law, rows, grid_seed(seed_base + replicate), &clamp, outcome)
    });
    for interval in intervals {
        tally.record(interval, truth);
    }
    tally.assert();
}

#[test]
fn a_calibration_replicate_reports_a_finite_interval_around_the_truth() {
    // Ordinary-CI smoke of the harness path: no coverage tally, one replicate per law.
    let clamp = clamp_at(&[0, 1], |unit| PROBE_VALUES[unit]);
    let truth = truth_equilibrium(&cyclic_truth(), &clamp)[2];
    for law in [Law::AsymptoticNormal, Law::Bootstrap] {
        let (lo, hi) = held_out_interval(law, 4_000, 9_001, &clamp, 2).expect("an interval");
        assert!(lo < hi && hi - lo < 1.0, "[{lo}, {hi}]");
        assert!(lo - 0.25 < truth && truth < hi + 0.25, "[{lo}, {hi}] against {truth}");
    }
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn held_out_joint_regime_mean_covers_with_asymptotic_normal_draws_l95() {
    held_out_coverage(
        "recovery_held_out_joint_asymptotic_normal",
        Law::AsymptoticNormal,
        0x71_0000,
        &[0, 1],
        2,
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn held_out_joint_regime_mean_covers_with_the_whole_procedure_bootstrap_l95() {
    held_out_coverage("recovery_held_out_joint_bootstrap", Law::Bootstrap, 0x72_0000, &[0, 2], 1);
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn held_out_joint_regime_mean_of_a_cycle_member_covers_l95() {
    // Clamping nodes 1 and 2 leaves node 0 free, so the outcome is the one unit that
    // the two-way feedback with node 1 feeds back into.
    held_out_coverage(
        "recovery_held_out_joint_feedback_member",
        Law::AsymptoticNormal,
        0x73_0000,
        &[1, 2],
        0,
    );
}
