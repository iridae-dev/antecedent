// SPDX-License-Identifier: MIT OR Apache-2.0

//! Simulated-truth tests of the experimental recovery fit and its uncertainty laws.

use super::*;

use antecedent_core::{CausalRng, Value};
use antecedent_identify::experiment_regime::{
    DependencePremise, ExperimentRegime, PopulationId, RegimeIntervention, SourceIdentity, UnitMask,
};
use antecedent_identify::experimental_recovery::{DrawAdmissibility, InadmissibleCoefficientDraw};

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

/// Cycles 0 ↔ 1 and 1 ↔ 2, plus a negative edge 0 → 2, with large intercepts.
fn cyclic_truth() -> Truth {
    let mut k = vec![0.0; 9];
    k[1] = 0.3; // 1 → 0
    k[3] = 0.4; // 0 → 1
    k[5] = 0.2; // 2 → 1
    k[6] = -0.3; // 0 → 2
    k[7] = 0.5; // 1 → 2
    Truth { k, c: vec![5.0, -3.0, 10.0], sd: vec![1.0, 0.7, 1.2] }
}

/// Lower-triangular (acyclic) support: 0 → 1, 0 → 2, 1 → 2.
fn acyclic_truth() -> Truth {
    let mut k = vec![0.0; 9];
    k[3] = 0.5; // 0 → 1
    k[6] = 0.25; // 0 → 2
    k[7] = 0.5; // 1 → 2
    Truth { k, c: vec![40.0, 3.0, -7.0], sd: vec![2.0, 1.0, 0.8] }
}

/// A null regime and one single-unit clamp per node.
const SINGLE: &[&[usize]] = &[&[], &[0], &[1], &[2]];

/// Overlapping multi-unit clamps: some blocks are overdetermined.
const MULTIPLE: &[&[usize]] = &[&[0, 1], &[1, 2], &[0, 2], &[0]];

const CLAMP_SD: f64 = 1.5;

fn clamp_mean(unit: usize) -> f64 {
    2.0 + 3.0 * unit as f64
}

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
                let identity = if p == q { 1.0 } else { 0.0 };
                matrix[p * f + q] = identity - truth.effect(row, col);
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

    fn fit(&self, options: &ExperimentalFitOptions) -> ExperimentalRecoveryFit {
        self.with_rows(|rows| fit_experimental_recovery(rows, options).expect("the rows fit"))
    }

    fn bootstrap(
        &self,
        spec: &BootstrapSpec<'_>,
        ctx: &ExecutionContext,
    ) -> Result<RecoveredEstimate, ExperimentalFitError> {
        self.with_rows(|rows| {
            bootstrap_experimental_recovery(rows, &ExperimentalFitOptions::default(), spec, ctx)
        })
    }
}

fn off_diagonal(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|into| (0..n).filter(move |from| *from != into).map(move |f| (into, f)))
        .collect()
}

fn assert_recovers(truth: &Truth, fit: &ExperimentalRecoveryFit, sigmas: f64) {
    for (into, from) in off_diagonal(truth.n()) {
        let estimate = fit.coefficient(coefficient(into, from)).expect("resolved coefficient");
        let se = fit.standard_error(coefficient(into, from)).expect("standard error");
        assert!(se > 0.0 && se < 0.05, "standard error {se} of {into} <- {from}");
        close(estimate, truth.effect(into, from), sigmas * se, "direct effect");
    }
    for unit in 0..truth.n() {
        let intercept = fit.intercept()[unit].expect("resolved intercept");
        close(intercept, truth.c[unit], 0.4, "intercept");
        let variance = fit.disturbance_variance()[unit].expect("resolved variance");
        let expected = truth.sd[unit] * truth.sd[unit];
        close(variance, expected, 0.05 * expected, "disturbance variance");
    }
}

#[test]
fn single_clamp_experiments_recover_a_cyclic_truth_with_intercepts() {
    let truth = cyclic_truth();
    let fit = Dataset::simulate(&truth, SINGLE, 20_000, 11).fit(&ExperimentalFitOptions::default());
    assert!(fit.goal_closed());
    assert!(fit.unresolved().is_empty());
    assert_recovers(&truth, &fit, 5.0);
    let stability = fit.diagnostics().stability.as_ref().expect("complete matrix");
    assert!(stability.weakly_stable());
    assert!(stability.family_admitted);
    assert_eq!(stability.experiments.len(), SINGLE.len());
    assert!(fit.diagnostics().flags.is_empty(), "{:?}", fit.diagnostics().flags);
    let correlation = fit.diagnostics().largest_residual_correlation.expect("jointly passive rows");
    assert!(correlation.correlation.abs() < 0.05, "{correlation:?}");
    // The null regime feeds the disturbance estimate: node 0 is passive in three regimes.
    assert_eq!(fit.residual_rows()[0], 60_000);
}

#[test]
fn multiple_clamp_experiments_recover_under_both_weightings() {
    let truth = cyclic_truth();
    let data = Dataset::simulate(&truth, MULTIPLE, 20_000, 23);
    for weighting in [ConstraintWeighting::Efficient, ConstraintWeighting::Unweighted] {
        let fit =
            data.fit(&ExperimentalFitOptions { weighting, ..ExperimentalFitOptions::default() });
        assert!(fit.goal_closed());
        assert!(
            fit.block_fits().iter().any(|block| block.rows > block.column_rank),
            "some block is overdetermined"
        );
        assert_recovers(&truth, &fit, 5.0);
    }
}

#[test]
fn coefficient_intervals_cover_near_their_nominal_rate() {
    let truth = cyclic_truth();
    let estimators =
        [CoefficientCovarianceEstimator::Sandwich, CoefficientCovarianceEstimator::ModelBased];
    let mut covered = [0_usize; 2];
    let mut total = 0_usize;
    for seed in 0..300_u64 {
        let data = Dataset::simulate(&truth, SINGLE, 400, 5_000 + seed);
        for (slot, covariance) in estimators.iter().enumerate() {
            let options = ExperimentalFitOptions {
                covariance: *covariance,
                ..ExperimentalFitOptions::default()
            };
            let fit = data.fit(&options);
            for (into, from) in off_diagonal(3) {
                let estimate = fit.coefficient(coefficient(into, from)).expect("coefficient");
                let se = fit.standard_error(coefficient(into, from)).expect("standard error");
                if (estimate - truth.effect(into, from)).abs() <= 1.959_964 * se {
                    covered[slot] += 1;
                }
            }
        }
        total += 6;
    }
    for (slot, hits) in covered.iter().enumerate() {
        let rate = *hits as f64 / total as f64;
        assert!((0.92..=0.975).contains(&rate), "estimator {slot} covers at {rate}");
    }
}

#[test]
fn disturbance_divisor_and_intercept_match_a_hand_regression() {
    let designs: &[&[usize]] = &[&[0], &[1]];
    let x0 = vec![1.0, 2.0, 3.0, 4.0, 10.0];
    let x1 = vec![2.0, 1.0, 4.0, 3.0, 9.0];
    let data = vec![
        vec![x0.clone(), x1.clone()],
        vec![vec![1.2, 0.7, 2.2, 1.1, 0.4], vec![0.5, 1.5, -1.0, 2.0, 3.0]],
    ];
    let dataset = Dataset::from_columns(2, designs, data);
    // Node 1 is passive only in regime 0: its row is the regression of x1 on x0 there.
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (m0, m1) = (mean(&x0), mean(&x1));
    let sxy: f64 = x0.iter().zip(&x1).map(|(a, b)| (a - m0) * (b - m1)).sum();
    let sxx: f64 = x0.iter().map(|a| (a - m0) * (a - m0)).sum();
    let slope = sxy / sxx;
    let residuals: Vec<f64> = x0.iter().zip(&x1).map(|(a, b)| b - slope * a).collect();
    let intercept = mean(&residuals);
    let ssr: f64 = residuals.iter().map(|e| (e - intercept) * (e - intercept)).sum();
    for (divisor, denominator) in
        [(DisturbanceDivisor::RowCount, 5.0), (DisturbanceDivisor::RowCountMinusOne, 4.0)]
    {
        let fit =
            dataset.fit(&ExperimentalFitOptions { divisor, ..ExperimentalFitOptions::default() });
        close(fit.coefficient(coefficient(1, 0)).expect("slope"), slope, 1e-12, "slope");
        close(fit.intercept()[1].expect("intercept"), intercept, 1e-12, "intercept");
        let variance = fit.disturbance_variance()[1].expect("variance");
        close(variance, ssr / denominator, 1e-12, "residual variance");
        assert_eq!(fit.residual_rows()[1], 5);
    }
    let by_n =
        dataset.fit(&ExperimentalFitOptions::default()).disturbance_variance()[1].expect("n");
    let by_n_minus_one = dataset
        .fit(&ExperimentalFitOptions {
            divisor: DisturbanceDivisor::RowCountMinusOne,
            ..ExperimentalFitOptions::default()
        })
        .disturbance_variance()[1]
        .expect("n - 1");
    close(by_n_minus_one / by_n, 1.25, 1e-12, "divisor ratio");
}

#[test]
fn acyclic_support_matches_independent_dag_regressions() {
    let truth = acyclic_truth();
    // Each regime clamps every node but one, so each row is the regression of the
    // passive node on all others in that regime.
    let designs: &[&[usize]] = &[&[1, 2], &[0, 2], &[0, 1]];
    let dataset = Dataset::simulate(&truth, designs, 5_000, 31);
    let fit = dataset.fit(&ExperimentalFitOptions::default());
    assert!(fit.goal_closed());
    for target in 0..3 {
        let columns = &dataset.data[target];
        let others: Vec<usize> = (0..3).filter(|unit| *unit != target).collect();
        let rows = columns[0].len() as f64;
        let mean = |values: &[f64]| values.iter().sum::<f64>() / rows;
        let means: Vec<f64> = columns.iter().map(|column| mean(column)).collect();
        let cross = |a: usize, b: usize| -> f64 {
            columns[a].iter().zip(&columns[b]).map(|(x, y)| (x - means[a]) * (y - means[b])).sum()
        };
        let (p, q) = (others[0], others[1]);
        let (spp, sqq, spq) = (cross(p, p), cross(q, q), cross(p, q));
        let (spy, sqy) = (cross(p, target), cross(q, target));
        let determinant = spp * sqq - spq * spq;
        let bp = (sqq * spy - spq * sqy) / determinant;
        let bq = (spp * sqy - spq * spy) / determinant;
        let estimate_p = fit.coefficient(coefficient(target, p)).expect("coefficient");
        let estimate_q = fit.coefficient(coefficient(target, q)).expect("coefficient");
        close(estimate_p, bp, 1e-12 * bp.abs().max(1.0), "first regression slope");
        close(estimate_q, bq, 1e-12 * bq.abs().max(1.0), "second regression slope");
        let intercept = means[target] - bp * means[p] - bq * means[q];
        let fitted = fit.intercept()[target].expect("intercept");
        close(fitted, intercept, 1e-12 * means[target].abs().max(1.0), "regression intercept");
        let ssr: f64 = (0..columns[0].len())
            .map(|row| {
                let e =
                    columns[target][row] - bp * columns[p][row] - bq * columns[q][row] - intercept;
                e * e
            })
            .sum();
        let variance = fit.disturbance_variance()[target].expect("variance");
        close(variance, ssr / rows, 1e-12 * variance.max(1.0), "regression residual variance");
    }
    // Upper-triangular coordinates are absent from the truth and are estimated near zero.
    for (into, from) in [(0, 1), (0, 2), (1, 2)] {
        let estimate = fit.coefficient(coefficient(into, from)).expect("coefficient");
        let se = fit.standard_error(coefficient(into, from)).expect("standard error");
        close(estimate, 0.0, 5.0 * se, "absent edge");
    }
    for (into, from) in [(1, 0), (2, 0), (2, 1)] {
        let estimate = fit.coefficient(coefficient(into, from)).expect("coefficient");
        let se = fit.standard_error(coefficient(into, from)).expect("standard error");
        close(estimate, truth.effect(into, from), 5.0 * se, "present edge");
    }
}

/// Regimes clamp 0 and 1 only. Rows 0 and 1 are uncovered (2 is never clamped).
/// `K[1,0] = 2` and `K[0,1] = 0.5` without disturbance make the block into 2
/// exactly rank one: `[[1, 2], [0.5, 1]]`.
fn deficient_dataset() -> Dataset {
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
    Dataset::from_columns(3, &[&[0], &[1]], vec![regime0, regime1])
}

#[test]
fn rank_deficient_and_uncovered_rows_keep_their_unresolved_directions() {
    let fit = deficient_dataset().fit(&ExperimentalFitOptions::default());
    assert!(!fit.goal_closed());
    assert_eq!(fit.blocks().len(), 1);
    let AssembledCoefficientRecovery::NullDirections { target, column_rank, directions, .. } =
        &fit.blocks()[0]
    else {
        panic!("the block into 2 is rank deficient: {:?}", fit.blocks()[0]);
    };
    assert_eq!(*target, node_at(2));
    assert_eq!(*column_rank, 1);
    assert_eq!(directions.len(), 1);
    assert_eq!(fit.unresolved().len(), 3);
    let covered: Vec<VariableId> = fit
        .unresolved()
        .iter()
        .filter_map(|direction| match &direction.reason {
            UnresolvedReason::IncompletePairCoverage { target, missing_clamps } => {
                assert_eq!(missing_clamps, &vec![node_at(2)]);
                assert_eq!(direction.support.len(), 2);
                Some(*target)
            }
            UnresolvedReason::DeficientConstraintRank { target, .. } => {
                assert_eq!(*target, node_at(2));
                assert_eq!(direction.support.len(), 2, "the null direction moves both sources");
                None
            }
        })
        .collect();
    assert_eq!(covered, vec![node_at(0), node_at(1)]);
    for (into, from) in off_diagonal(3) {
        assert!(fit.coefficient(coefficient(into, from)).is_none());
    }
    assert!(fit.intercept().iter().all(Option::is_none));
    assert!(fit.disturbance_variance().iter().all(Option::is_none));
    // The identified projection of the row into 2 is along (1, 2) / √5.
    let identified = fit.block_fits()[0].identified.as_ref().expect("identified projection");
    assert_eq!(identified.basis.len(), 1);
    let basis = &identified.basis[0];
    close((basis[0] + 2.0 * basis[1]).abs() / 5.0_f64.sqrt(), 1.0, 1e-9, "row-space basis");
    let projected: Vec<f64> = basis.iter().map(|weight| weight * identified.estimate[0]).collect();
    close(projected[0], 0.25, 0.01, "projection onto the first source");
    close(projected[1], 0.5, 0.02, "projection onto the second source");
    let flags = &fit.diagnostics().flags;
    assert!(flags.contains(&FitFlag::IncompleteCoverage { target: node_at(0) }));
    assert!(flags.contains(&FitFlag::IncompleteCoverage { target: node_at(1) }));
    assert!(flags.contains(&FitFlag::RankDeficientConstraint {
        target: node_at(2),
        column_rank: 1,
        width: 2
    }));
    assert!(fit.diagnostics().stability.is_none());
    // Every draw carries the directions and is inadmissible; nothing was filled.
    let estimate = asymptotic_normal_recovery(&fit, 8, 3).expect("draws");
    for (draw, status) in estimate.law().draws().iter().zip(estimate.draw_status()) {
        assert!(draw.coefficients().is_empty());
        assert_eq!(draw.unresolved().len(), 3);
        assert_eq!(
            draw.admissibility(),
            DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::UnresolvedDirections)
        );
        assert_eq!(status, &DrawStatus::Unresolved { directions: 3 });
    }
}

#[test]
fn a_strong_prior_does_not_repair_a_null_direction() {
    let fit = deficient_dataset().fit(&ExperimentalFitOptions::default());
    let known = PosteriorSpec {
        prior: RecoveryPrior {
            coefficients: CoefficientPrior::Normal { mean: 0.0, variance: 1e-8 },
            disturbance: DisturbancePrior::Known { variances: vec![1.0, 1.0, 0.01] },
        },
        draws: 16,
        burn_in: 0,
        seed: 5,
    };
    let estimate = posterior_recovery(&fit, &known).expect("posterior");
    assert_eq!(estimate.law_meaning(), Some(DrawLawMeaning::ParameterPosterior));
    for draw in estimate.law().draws() {
        assert!(draw.coefficients().iter().all(|(c, _)| c.into != node_at(2)));
        assert!(draw.unresolved().iter().any(|direction| matches!(
            direction.reason,
            UnresolvedReason::DeficientConstraintRank { .. }
        )));
        assert!(draw.is_inadmissible());
    }
    let identified = estimate.identified_posteriors();
    assert_eq!(identified.len(), 1);
    assert_eq!(identified[0].basis.len(), 1);
    assert_eq!(identified[0].mean.len(), 1);
    // The prior acts on the identified projection only: it shrinks it toward zero.
    assert!(identified[0].mean[0].abs() < 0.01, "{:?}", identified[0].mean);
    let unknown = PosteriorSpec {
        prior: RecoveryPrior {
            coefficients: CoefficientPrior::Normal { mean: 0.0, variance: 1e-6 },
            disturbance: DisturbancePrior::InverseWishart {
                degrees_of_freedom: 2.0,
                scale: vec![1.0, 1.0, 1.0],
            },
        },
        draws: 8,
        burn_in: 4,
        seed: 5,
    };
    let estimate = posterior_recovery(&fit, &unknown).expect("posterior");
    assert!(estimate.identified_posteriors().is_empty());
    assert!(estimate.law().draws().iter().all(|draw| draw.coefficients().is_empty()));
    assert!(
        estimate
            .law()
            .draws()
            .iter()
            .all(antecedent_identify::experimental_recovery::JointRecoveryDraw::is_inadmissible)
    );
}

fn draw_moments(estimate: &RecoveredEstimate, target: DirectCoefficient) -> (f64, f64) {
    let values: Vec<f64> = estimate
        .law()
        .draws()
        .iter()
        .filter_map(|draw| draw.coefficients().iter().find(|(c, _)| *c == target).map(|(_, v)| *v))
        .collect();
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (count - 1.0);
    (mean, variance.sqrt())
}

#[test]
fn asymptotic_draws_follow_the_joint_covariance_and_replay() {
    let truth = cyclic_truth();
    let fit = Dataset::simulate(&truth, SINGLE, 5_000, 41).fit(&ExperimentalFitOptions::default());
    let estimate = asymptotic_normal_recovery(&fit, 4_000, 9).expect("draws");
    assert_eq!(estimate.law_meaning(), Some(DrawLawMeaning::SamplingLaw));
    assert_eq!(estimate.origin(), &RecoveredDrawOrigin::AsymptoticNormal { draws: 4_000, seed: 9 });
    close(estimate.law().total_mass(), 1.0, 1e-9, "total mass");
    assert!(estimate.law().draws().iter().all(|draw| !draw.is_inadmissible()));
    assert!(estimate.draw_status().iter().all(|status| matches!(
        status,
        DrawStatus::Complete { stability: Some(stability) } if stability.weakly_stable()
    )));
    for (into, from) in off_diagonal(3) {
        let target = coefficient(into, from);
        let (mean, sd) = draw_moments(&estimate, target);
        let se = fit.standard_error(target).expect("standard error");
        close(mean, fit.coefficient(target).expect("estimate"), 0.1 * se, "draw mean");
        close(sd / se, 1.0, 0.1, "draw spread");
    }
    let again = asymptotic_normal_recovery(&fit, 4_000, 9).expect("draws");
    assert_eq!(again, estimate);
    let other = asymptotic_normal_recovery(&fit, 4_000, 10).expect("draws");
    assert_ne!(other.law(), estimate.law());
    // The per-block covariances feed the block-wise generator unchanged.
    let point = fit.point_draw().expect("point");
    let covariances = fit.block_covariances();
    assert_eq!(covariances.len(), 3);
    let generation = crate::recovery::DrawGeneration {
        nodes: fit.nodes(),
        blocks: fit.blocks(),
        covariances: &covariances,
        intercept: &point.intercept,
        noise_mean: &point.noise_mean,
        noise_covariance: &point.noise_covariance,
        draws: 4,
        seed: 1,
    };
    assert_eq!(crate::recovery::asymptotic_normal_draws(&generation).expect("draws").len(), 4);
}

#[test]
fn posterior_over_resolved_rows_matches_the_efficient_fit() {
    let truth = cyclic_truth();
    let data = Dataset::simulate(&truth, SINGLE, 5_000, 43);
    let fit = data.fit(&ExperimentalFitOptions {
        covariance: CoefficientCovarianceEstimator::ModelBased,
        ..ExperimentalFitOptions::default()
    });
    let variances: Vec<f64> =
        fit.disturbance_variance().iter().map(|v| v.expect("variance")).collect();
    let known = PosteriorSpec {
        prior: RecoveryPrior {
            coefficients: CoefficientPrior::Flat,
            disturbance: DisturbancePrior::Known { variances: variances.clone() },
        },
        draws: 4_000,
        burn_in: 0,
        seed: 3,
    };
    let estimate = posterior_recovery(&fit, &known).expect("posterior");
    assert!(estimate.law().draws().iter().all(|draw| !draw.is_inadmissible()));
    for (into, from) in off_diagonal(3) {
        let target = coefficient(into, from);
        let (mean, sd) = draw_moments(&estimate, target);
        let se = fit.standard_error(target).expect("standard error");
        close(mean, fit.coefficient(target).expect("estimate"), 0.1 * se, "posterior mean");
        close(sd / se, 1.0, 0.1, "posterior spread");
    }
    let gibbs = PosteriorSpec {
        prior: RecoveryPrior {
            coefficients: CoefficientPrior::Normal { mean: 0.0, variance: 100.0 },
            disturbance: DisturbancePrior::InverseWishart {
                degrees_of_freedom: 2.0,
                scale: vec![1.0, 1.0, 1.0],
            },
        },
        draws: 2_000,
        burn_in: 100,
        seed: 3,
    };
    let estimate = posterior_recovery(&fit, &gibbs).expect("posterior");
    assert_eq!(estimate.law().draws().len(), 2_000);
    for (unit, expected) in variances.iter().enumerate() {
        let mean = estimate
            .law()
            .draws()
            .iter()
            .map(|draw| draw.noise_covariance()[unit * 3 + unit])
            .sum::<f64>()
            / 2_000.0;
        close(mean, *expected, 0.05 * expected, "posterior disturbance variance");
        let intercept =
            estimate.law().draws().iter().map(|draw| draw.intercept()[unit]).sum::<f64>() / 2_000.0;
        close(intercept, fit.intercept()[unit].expect("intercept"), 0.2, "posterior intercept");
    }
    for (into, from) in off_diagonal(3) {
        let target = coefficient(into, from);
        let (mean, _) = draw_moments(&estimate, target);
        let se = fit.standard_error(target).expect("standard error");
        close(mean, fit.coefficient(target).expect("estimate"), 0.2 * se, "Gibbs mean");
    }
    assert_eq!(posterior_recovery(&fit, &gibbs).expect("posterior"), estimate);
}

#[test]
fn a_bad_prior_is_refused() {
    let fit =
        Dataset::simulate(&cyclic_truth(), SINGLE, 200, 2).fit(&ExperimentalFitOptions::default());
    let spec = PosteriorSpec {
        prior: RecoveryPrior {
            coefficients: CoefficientPrior::Flat,
            disturbance: DisturbancePrior::Known { variances: vec![1.0, 0.0, 1.0] },
        },
        draws: 4,
        burn_in: 0,
        seed: 1,
    };
    assert_eq!(
        posterior_recovery(&fit, &spec),
        Err(ExperimentalFitError::Prior(PriorFault::KnownVariance { node: 1 }))
    );
}

fn bootstrap_spec(
    design: DeclaredSamplingDesign,
    rule: ResamplingRule,
    replicates: usize,
    batch: usize,
) -> BootstrapSpec<'static> {
    BootstrapSpec {
        design,
        rule,
        replicates,
        seed: 17,
        sampling_units: None,
        block_length: None,
        batch,
    }
}

#[test]
fn a_row_level_bootstrap_is_refused_under_a_trajectory_design() {
    let data = Dataset::simulate(&cyclic_truth(), SINGLE, 100, 3);
    let ctx = ExecutionContext::for_tests(0);
    let row_level = bootstrap_spec(
        DeclaredSamplingDesign::IndependentSubjectTrajectories,
        ResamplingRule::RowIid,
        8,
        1,
    );
    assert!(matches!(
        data.bootstrap(&row_level, &ctx),
        Err(ExperimentalFitError::SamplingDesign(SamplingDesignError::FailedAttempt { .. }))
    ));
    let subject_on_panel =
        bootstrap_spec(DeclaredSamplingDesign::ClusteredPanel, ResamplingRule::RowSubjectIid, 8, 1);
    assert!(matches!(
        data.bootstrap(&subject_on_panel, &ctx),
        Err(ExperimentalFitError::SamplingDesign(_))
    ));
    let unlabelled =
        bootstrap_spec(DeclaredSamplingDesign::ClusteredPanel, ResamplingRule::WholeCluster, 8, 1);
    assert_eq!(data.bootstrap(&unlabelled, &ctx), Err(ExperimentalFitError::MissingSamplingUnits));
    let no_block = bootstrap_spec(
        DeclaredSamplingDesign::DependentTimeSeries,
        ResamplingRule::ContiguousBlock,
        8,
        1,
    );
    assert_eq!(
        data.bootstrap(&no_block, &ctx),
        Err(ExperimentalFitError::BlockLength { rows: 100 })
    );
}

#[test]
fn whole_procedure_bootstrap_spread_matches_the_sandwich() {
    let truth = cyclic_truth();
    let data = Dataset::simulate(&truth, SINGLE, 1_500, 59);
    let fit = data.fit(&ExperimentalFitOptions::default());
    let spec = bootstrap_spec(
        DeclaredSamplingDesign::IndependentSubjectTrajectories,
        ResamplingRule::RowSubjectIid,
        300,
        16,
    );
    let estimate = data.bootstrap(&spec, &ExecutionContext::for_tests(0)).expect("bootstrap");
    assert_eq!(estimate.law_meaning(), Some(DrawLawMeaning::SamplingLaw));
    assert!(matches!(estimate.origin(), RecoveredDrawOrigin::Bootstrap { replicates: 300, .. }));
    close(estimate.failed_mass(), 0.0, 0.0, "failed mass");
    close(estimate.law().total_mass(), 1.0, 1e-9, "total mass");
    for (into, from) in off_diagonal(3) {
        let target = coefficient(into, from);
        let (mean, sd) = draw_moments(&estimate, target);
        let se = fit.standard_error(target).expect("standard error");
        close(mean, fit.coefficient(target).expect("estimate"), 0.4 * se, "bootstrap mean");
        assert!((0.8..=1.25).contains(&(sd / se)), "bootstrap spread {sd} against {se}");
    }
}

#[test]
fn bootstrap_draws_whole_subjects_and_whole_clusters() {
    let labels: &[u64] = &[7, 7, 3, 3, 9, 9];
    let units: &[&[u64]] = &[labels];
    let values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let column = [UnitColumn { unit: node_at(0), values: &values }];
    let samples = [RegimeSample { columns: &column, source: SourceIdentity::from_raw(1) }];
    for rule in [ResamplingRule::RowSubjectIid, ResamplingRule::WholeCluster] {
        let spec = BootstrapSpec {
            sampling_units: Some(units),
            ..bootstrap_spec(DeclaredSamplingDesign::ClusteredPanel, rule, 1, 1)
        };
        let plan = ResamplingPlan::new(rule, &samples, &spec).expect("plan");
        for replicate in 0..20 {
            let mut rng = keyed_stream(BOOTSTRAP_REPLICATE_OPERATION, 0, replicate, 1);
            let rows = plan.draw(0, &mut rng);
            assert_eq!(rows.len(), 6);
            for pair in rows.chunks(2) {
                assert_eq!(pair[0] / 2, pair[1] / 2, "a trajectory is drawn whole: {rows:?}");
                assert_eq!(pair[0] + 1, pair[1]);
            }
        }
    }
    let block = BootstrapSpec {
        block_length: Some(3),
        ..bootstrap_spec(
            DeclaredSamplingDesign::DependentTimeSeries,
            ResamplingRule::ContiguousBlock,
            1,
            1,
        )
    };
    let plan =
        ResamplingPlan::new(ResamplingRule::ContiguousBlock, &samples, &block).expect("plan");
    let mut rng = keyed_stream(BOOTSTRAP_REPLICATE_OPERATION, 0, 0, 1);
    let rows = plan.draw(0, &mut rng);
    assert_eq!(rows.len(), 6);
    for triple in rows.chunks(3) {
        assert_eq!(triple[1], triple[0] + 1);
        assert_eq!(triple[2], triple[0] + 2);
    }
}

#[test]
fn failed_bootstrap_replicates_stay_as_failed_mass() {
    // Three rows per regime with a repeated clamp level: some resamples make the
    // clamp constant, and that replicate's assembly refuses it.
    let designs: &[&[usize]] = &[&[0], &[1]];
    let data = vec![
        vec![vec![1.0, 1.0, 2.0], vec![0.4, 0.9, 1.7]],
        vec![vec![0.2, -0.1, 1.1], vec![0.5, 0.5, 3.0]],
    ];
    let dataset = Dataset::from_columns(2, designs, data);
    let spec = bootstrap_spec(
        DeclaredSamplingDesign::IndependentSubjectTrajectories,
        ResamplingRule::RowSubjectIid,
        40,
        4,
    );
    let estimate = dataset.bootstrap(&spec, &ExecutionContext::for_tests(0)).expect("bootstrap");
    assert_eq!(estimate.law().draws().len(), 40);
    close(estimate.law().total_mass(), 1.0, 1e-12, "total mass");
    let failed = estimate.failed_mass();
    assert!(failed > 0.0 && failed < 1.0, "failed mass {failed}");
    close(estimate.law().inadmissible_mass(), failed, 1e-12, "failed mass is inadmissible");
    for (draw, status) in estimate.law().draws().iter().zip(estimate.draw_status()) {
        match status {
            DrawStatus::Failed { error } => {
                assert!(matches!(
                    error,
                    ExperimentalFitError::Assembly(
                        MomentAssemblyRefusal::DegenerateIntervention { .. }
                    )
                ));
                assert_eq!(
                    draw.admissibility(),
                    DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::Marked)
                );
                close(draw.weight(), 1.0 / 40.0, 1e-15, "failed weight");
            }
            DrawStatus::Complete { .. } => assert!(!draw.is_inadmissible()),
            other => panic!("unexpected status {other:?}"),
        }
    }
}

#[test]
fn bootstrap_law_does_not_depend_on_threads_or_batches() {
    let data = Dataset::simulate(&cyclic_truth(), SINGLE, 300, 71);
    let design = DeclaredSamplingDesign::IndependentSubjectTrajectories;
    let serial = data
        .bootstrap(
            &bootstrap_spec(design, ResamplingRule::RowSubjectIid, 24, 1),
            &ExecutionContext::for_tests(0),
        )
        .expect("serial");
    for (threads, batch) in [(4, 3), (3, 64), (2, 5)] {
        let parallel = data
            .bootstrap(
                &bootstrap_spec(design, ResamplingRule::RowSubjectIid, 24, batch),
                &ExecutionContext::production(99, threads),
            )
            .expect("parallel");
        assert_eq!(parallel, serial, "threads {threads}, batch {batch}");
    }
    let reseeded = data
        .bootstrap(
            &BootstrapSpec {
                seed: 18,
                ..bootstrap_spec(design, ResamplingRule::RowSubjectIid, 24, 1)
            },
            &ExecutionContext::for_tests(0),
        )
        .expect("reseeded");
    assert_ne!(reseeded.law(), serial.law());
}

#[test]
fn a_refused_contract_is_not_fit() {
    let mut dataset = Dataset::simulate(&cyclic_truth(), SINGLE, 50, 1);
    dataset.contract.regimes[1].source = dataset.contract.regimes[0].source;
    let refused = dataset
        .with_rows(|rows| fit_experimental_recovery(rows, &ExperimentalFitOptions::default()));
    assert!(matches!(refused, Err(ExperimentalFitError::Regime(_))));
}

#[test]
fn supplied_laws_declare_no_meaning() {
    let fit =
        Dataset::simulate(&cyclic_truth(), SINGLE, 500, 4).fit(&ExperimentalFitOptions::default());
    let law = RecoveredCoefficientUncertainty::retain(
        fit.nodes(),
        vec![fit.point_draw().expect("point")],
    )
    .expect("law");
    let supplied = RecoveredEstimate::supplied(law);
    assert_eq!(supplied.origin(), &RecoveredDrawOrigin::Supplied);
    assert_eq!(supplied.law_meaning(), None);
    assert_eq!(supplied.draw_status(), &[DrawStatus::Complete { stability: None }]);
}
