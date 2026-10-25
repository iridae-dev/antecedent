// SPDX-License-Identifier: MIT OR Apache-2.0

//! Noisy simulated-truth cases for experiment moment assembly.
//!
//! The exact fixtures elsewhere have zero residual variance, so a dropped
//! intercept or an `n - 1` divisor changes nothing they check. These cases draw
//! samples from a linear system with non-zero residual variance and large
//! intercepts, compare the assembled moments with closed-form truth, and predict
//! a regime the assembly never saw.

use antecedent_core::{Value, VariableId};

use crate::experiment_moments::{
    AssembledExperimentConstraints, MissingPair, MomentAssemblyRefusal, RegimeSample,
    TargetConstraintBlock, UnitColumn, assemble_experiment_constraints,
};
use crate::experiment_regime::{
    AcceptedRegimeContract, DependencePremise, ExperimentRegime, PopulationId, RegimeContract,
    RegimeContractDecision, RegimeIntervention, SourceIdentity, UnitMask, validate_regime_contract,
};
use crate::experimental_recovery::{
    AssembledCoefficientRecovery, WeakStability, recover_assembled_coefficients,
};

fn unit(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn regime(units: &[u32], targets: &[u32], source: u64) -> ExperimentRegime {
    ExperimentRegime {
        targets: targets
            .iter()
            .map(|&raw| RegimeIntervention { unit: unit(raw), value: Value::Int64(0) })
            .collect(),
        units: units.iter().copied().map(unit).collect(),
        population: PopulationId::from_raw(1),
        premise: DependencePremise::Independence,
        source: SourceIdentity::from_raw(source),
    }
}

fn accepted(regimes: Vec<ExperimentRegime>) -> AcceptedRegimeContract {
    let decision = validate_regime_contract(&RegimeContract { regimes, mask: UnitMask::default() });
    let RegimeContractDecision::Accepted(contract) = decision else {
        panic!("the fixture regime contract was refused");
    };
    contract
}

/// Deterministic standard-normal stream (LCG uniforms through Box-Muller).
struct Normals {
    state: u64,
}

impl Normals {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Open interval (0, 1) so the logarithm is finite.
        (((self.state >> 11) as f64) + 0.5) / ((1_u64 << 53) as f64)
    }

    fn draw(&mut self) -> f64 {
        let u = self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

// Structural system. Every equation has a non-zero residual and a large intercept.
//   x ~ N(40, 2^2)
//   y = 3 + 0.5 x + e_y, e_y ~ N(0, 1)
//   z = -7 + 0.25 x + 0.5 y + e_z, e_z ~ N(0, 0.8^2)
// A clamped unit is drawn independently: x ~ N(40, 2^2), y ~ N(10, 1.5^2), z ~ N(-20, 3^2).
const B_YX: f64 = 0.5;
const B_ZX: f64 = 0.25;
const B_ZY: f64 = 0.5;

struct Columns {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

fn simulate(rows: usize, clamped: &[u32], seed: u64) -> Columns {
    let mut noise = Normals::new(seed);
    let (mut x, mut y, mut z) =
        (Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let xv = 40.0 + 2.0 * noise.draw();
        let yv = if clamped.contains(&1) {
            10.0 + 1.5 * noise.draw()
        } else {
            3.0 + B_YX * xv + noise.draw()
        };
        let zv = if clamped.contains(&2) {
            -20.0 + 3.0 * noise.draw()
        } else {
            -7.0 + B_ZX * xv + B_ZY * yv + 0.8 * noise.draw()
        };
        x.push(xv);
        y.push(yv);
        z.push(zv);
    }
    Columns { x, y, z }
}

fn columns(data: &Columns) -> [UnitColumn<'_>; 3] {
    [
        UnitColumn { unit: unit(0), values: &data.x },
        UnitColumn { unit: unit(1), values: &data.y },
        UnitColumn { unit: unit(2), values: &data.z },
    ]
}

const ROWS: usize = 40_000;

/// The four training regimes: clamp {x}, {y, z}, {x, z}, {x, y}.
const TRAINING: [(&[u32], u64); 4] = [(&[0], 1), (&[1, 2], 2), (&[0, 2], 3), (&[0, 1], 4)];

fn assemble_training() -> AssembledExperimentConstraints {
    assemble_designs(&TRAINING, 1000)
}

/// Simulate one sample per `(clamp set, source)` design and assemble the constraints.
fn try_assemble_designs(
    designs: &[(&[u32], u64)],
    seed: u64,
) -> Result<AssembledExperimentConstraints, MomentAssemblyRefusal> {
    let contract = accepted(
        designs.iter().map(|(targets, source)| regime(&[0, 1, 2], targets, *source)).collect(),
    );
    let data: Vec<Columns> = designs
        .iter()
        .enumerate()
        .map(|(index, (targets, _))| simulate(ROWS, targets, seed + index as u64))
        .collect();
    let cols: Vec<[UnitColumn<'_>; 3]> = data.iter().map(columns).collect();
    let samples: Vec<RegimeSample<'_>> = cols
        .iter()
        .zip(designs)
        .map(|(c, (_, source))| RegimeSample {
            columns: c.as_slice(),
            source: SourceIdentity::from_raw(*source),
        })
        .collect();
    assemble_experiment_constraints(&contract, &samples, u64::MAX)
}

fn assemble_designs(designs: &[(&[u32], u64)], seed: u64) -> AssembledExperimentConstraints {
    try_assemble_designs(designs, seed).expect("covering independent regimes assemble")
}

fn block(assembled: &AssembledExperimentConstraints, target: u32) -> &TargetConstraintBlock {
    assembled.blocks.iter().find(|block| block.target == unit(target)).expect("target block")
}

fn row(block: &TargetConstraintBlock, regime: usize, probe: u32) -> (&[f64], f64) {
    let slot = block
        .rows
        .iter()
        .position(|row| row.regime == regime && row.probe == unit(probe))
        .expect("constraint row");
    let width = block.sources.len();
    (&block.coefficients[slot * width..(slot + 1) * width], block.rhs[slot])
}

fn near(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: actual {actual}, expected {expected}");
}

#[test]
fn noisy_moments_match_the_closed_form_and_are_centered() {
    let assembled = assemble_training();
    // Regime 0 clamps x only. Closed form for x ~ N(40, 4), the system above.
    let r0 = &assembled.moments[0];
    let truth_mean = [40.0, 23.0, 14.5];
    // Var x = 4, Cov(x, y) = 2, Var y = 2, Cov(x, z) = 2, Cov(y, z) = 1.5, Var z = 1.89.
    let truth_cov = [4.0, 2.0, 2.0, 2.0, 2.0, 1.5, 2.0, 1.5, 1.89];
    for (i, expected) in truth_mean.iter().enumerate() {
        near(r0.mean[i], *expected, 0.1, "regime 0 mean");
    }
    for (index, expected) in truth_cov.iter().enumerate() {
        near(r0.covariance[index], *expected, 0.12, "regime 0 covariance");
    }
    // Regime 3 clamps x and y independently. Var x = 4, Var y = 2.25, Cov(x, y) = 0,
    // Cov(x, z) = 1, Cov(y, z) = 1.125, Var z = 1.4525.
    let r3 = &assembled.moments[3];
    let truth_cov = [4.0, 0.0, 1.0, 0.0, 2.25, 1.125, 1.0, 1.125, 1.4525];
    for (index, expected) in truth_cov.iter().enumerate() {
        near(r3.covariance[index], *expected, 0.12, "regime 3 covariance");
    }
    near(r3.mean[2], 8.0, 0.1, "regime 3 mean of z");
    // A moment about zero instead of the mean would be near mean^2 = 1600.
    for moments in &assembled.moments {
        assert!(moments.covariance.iter().all(|c| c.abs() < 20.0));
        assert_eq!(moments.rows, ROWS);
    }
}

#[test]
fn noisy_constraint_rows_recover_the_direct_effects() {
    let assembled = assemble_training();
    let into_z = block(&assembled, 2);
    assert_eq!(into_z.sources, vec![unit(0), unit(1)]);
    let tolerance = 0.03;
    // Regime 0 clamps x; y is passive and follows x with slope B_YX. The total
    // effect of x on z is B_ZX + B_ZY * B_YX.
    let (coefficients, rhs) = row(into_z, 0, 0);
    near(coefficients[0], 1.0, 0.0, "probe coefficient");
    near(coefficients[1], B_YX, tolerance, "effect of x on passive y");
    near(rhs, B_ZX + B_ZY * B_YX, tolerance, "total effect of x on z");
    // Regime 3 clamps x and y, so the two rows are the direct effects themselves.
    let (probe_x, rhs_x) = row(into_z, 3, 0);
    assert_eq!(probe_x, &[1.0, 0.0]);
    near(rhs_x, B_ZX, tolerance, "direct effect of x on z");
    let (probe_y, rhs_y) = row(into_z, 3, 1);
    assert_eq!(probe_y, &[0.0, 1.0]);
    near(rhs_y, B_ZY, tolerance, "direct effect of y on z");
    // Every row of the block is satisfied by the true direct effects.
    for (index, _) in into_z.rows.iter().enumerate() {
        let width = into_z.sources.len();
        let lhs = into_z.coefficients[index * width] * B_ZX
            + into_z.coefficients[index * width + 1] * B_ZY;
        near(lhs, into_z.rhs[index], tolerance, "row consistency with the truth");
    }
    // x has no parents: the effects of y and z on x are zero in every regime that clamps them.
    let into_x = block(&assembled, 0);
    for (index, rhs) in into_x.rhs.iter().enumerate() {
        near(*rhs, 0.0, tolerance, &format!("effect into x, row {index}"));
    }
}

/// Slope of `response` on `regressor` with both centered.
fn slope(regressor: &[f64], response: &[f64]) -> f64 {
    let n = regressor.len() as f64;
    let mr = regressor.iter().sum::<f64>() / n;
    let ms = response.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (a, b) in regressor.iter().zip(response) {
        cov += (a - mr) * (b - ms);
        var += (a - mr) * (a - mr);
    }
    cov / var
}

#[test]
fn constraints_fit_on_training_regimes_predict_a_held_out_regime() {
    let assembled = assemble_training();
    let into_z = block(&assembled, 2);
    let (_, direct_x) = row(into_z, 3, 0);
    let (_, direct_y) = row(into_z, 3, 1);

    // Held-out regime: clamp y alone. No training regime has this clamp set, and its
    // sample comes from a stream the assembly never saw.
    let held_out = simulate(ROWS, &[1], 99_991);
    // Prediction from the training constraints. With y clamped, x is untouched and
    // independent of y, so z responds to y with the direct effect.
    let observed = slope(&held_out.y, &held_out.z);
    near(observed, direct_y, 0.03, "held-out slope of z on y");
    near(observed, B_ZY, 0.03, "held-out slope against the truth");
    // The covariance of (y, z) predicted from the direct effect and the held-out
    // clamp variance, with x contributing nothing to it.
    let n = held_out.y.len() as f64;
    let my = held_out.y.iter().sum::<f64>() / n;
    let mz = held_out.z.iter().sum::<f64>() / n;
    let var_y = held_out.y.iter().map(|v| (v - my) * (v - my)).sum::<f64>() / n;
    let cov_yz =
        held_out.y.iter().zip(&held_out.z).map(|(a, b)| (a - my) * (b - mz)).sum::<f64>() / n;
    near(cov_yz, direct_y * var_y, 0.1, "held-out Cov(y, z)");
    // A wrong effect for x is also visible: x does not move z through y when y is clamped.
    let x_slope = slope(&held_out.x, &held_out.z);
    near(x_slope, direct_x, 0.03, "held-out slope of z on x");
}

#[test]
fn small_sample_moments_use_the_row_count_divisor_and_the_mean() {
    // Two units, one clamp each, five rows with non-zero residual variance.
    let contract = accepted(vec![regime(&[0, 1], &[0], 1), regime(&[0, 1], &[1], 2)]);
    let x = [1.0, 2.0, 3.0, 4.0, 10.0];
    let y = [2.0, 1.0, 4.0, 3.0, 9.0];
    let cols = [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
    let samples = [
        RegimeSample { columns: &cols, source: SourceIdentity::from_raw(1) },
        RegimeSample { columns: &cols, source: SourceIdentity::from_raw(2) },
    ];
    let assembled =
        assemble_experiment_constraints(&contract, &samples, u64::MAX).expect("assembles");
    // Hand values with divisor n = 5: sum of squares about the mean 4 is 50, about the
    // mean 3.8 is 38.8, and the cross products sum to 42.
    let expected_mean = [4.0, 3.8];
    let expected_cov = [10.0, 8.4, 8.4, 7.76];
    for moments in &assembled.moments {
        assert_eq!(moments.rows, 5);
        for (a, b) in moments.mean.iter().zip(expected_mean) {
            near(*a, b, 1e-12, "mean");
        }
        for (a, b) in moments.covariance.iter().zip(expected_cov) {
            near(*a, b, 1e-12, "covariance with divisor n");
            // The n - 1 divisor would give 1.25 times these values.
            assert!((a - b * 1.25).abs() > 0.5);
        }
    }
    // The slope is divisor-free: Cov(x, y) / Var(x).
    let into_y = block(&assembled, 1);
    let (coefficients, rhs) = row(into_y, 0, 0);
    assert_eq!(coefficients, &[1.0]);
    near(rhs, 0.84, 1e-10, "experimental effect of x on y");
}

#[test]
fn integer_samples_match_the_exact_rational_moments() {
    // Independent formula: with integer data the covariance is (n sum(xy) - sum(x)
    // sum(y)) / n^2 exactly, a different computation from a centered two-pass sum.
    let contract = accepted(vec![regime(&[0, 1], &[0], 1), regime(&[0, 1], &[1], 2)]);
    let mut noise = Normals::new(5);
    for rows in [3_usize, 7, 12] {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a rounded standard normal times 20 is far inside the i64 range"
        )]
        let draw = |noise: &mut Normals| -> Vec<i64> {
            (0..rows).map(|_| (noise.draw() * 20.0).round() as i64 + 1000).collect()
        };
        let xs = draw(&mut noise);
        let ys = draw(&mut noise);
        let x: Vec<f64> = xs.iter().map(|&v| v as f64).collect();
        let y: Vec<f64> = ys.iter().map(|&v| v as f64).collect();
        let cols =
            [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
        let samples = [
            RegimeSample { columns: &cols, source: SourceIdentity::from_raw(1) },
            RegimeSample { columns: &cols, source: SourceIdentity::from_raw(2) },
        ];
        let assembled =
            assemble_experiment_constraints(&contract, &samples, u64::MAX).expect("assembles");
        let n = rows as i128;
        let pairs = [(&xs, &xs), (&xs, &ys), (&ys, &xs), (&ys, &ys)];
        for moments in &assembled.moments {
            for (index, (a, b)) in pairs.iter().enumerate() {
                let sum_a: i128 = a.iter().map(|&v| i128::from(v)).sum();
                let sum_b: i128 = b.iter().map(|&v| i128::from(v)).sum();
                let sum_ab: i128 =
                    a.iter().zip(b.iter()).map(|(&p, &q)| i128::from(p) * i128::from(q)).sum();
                let exact = (n * sum_ab - sum_a * sum_b) as f64 / (n * n) as f64;
                near(moments.covariance[index], exact, 1e-9, "exact rational covariance");
            }
        }
    }
}

// Whole-procedure checks on entire held-out intervention regimes. The design below
// clamps one unit at a time, so no joint clamp is ever sampled by the assembly.

/// Clamp x, y or z alone. Every direct effect is identified from these.
const SINGLES: [(&[u32], u64); 3] = [(&[0], 11), (&[1], 12), (&[2], 13)];

/// Structural direct effects, `truth[into * 3 + from]`.
fn truth_matrix() -> [f64; 9] {
    let mut k = [0.0; 9];
    k[3] = B_YX;
    k[6] = B_ZX;
    k[7] = B_ZY;
    k
}

/// Solve `a x = b` for `n` unknowns and `cols` right-hand sides stored row-major.
fn solve_small(mut a: Vec<f64>, mut b: Vec<f64>, n: usize, cols: usize) -> Vec<f64> {
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|p, q| a[*p * n + col].abs().total_cmp(&a[*q * n + col].abs()))
            .expect("a pivot row");
        for c in 0..n {
            a.swap(col * n + c, pivot * n + c);
        }
        for c in 0..cols {
            b.swap(col * cols + c, pivot * cols + c);
        }
        for row in (col + 1)..n {
            let factor = a[row * n + col] / a[col * n + col];
            for c in col..n {
                a[row * n + c] -= factor * a[col * n + c];
            }
            for c in 0..cols {
                b[row * cols + c] -= factor * b[col * cols + c];
            }
        }
    }
    let mut x = vec![0.0; n * cols];
    for c in 0..cols {
        for row in (0..n).rev() {
            let tail: f64 = ((row + 1)..n).map(|k| a[row * n + k] * x[k * cols + c]).sum();
            x[row * cols + c] = (b[row * cols + c] - tail) / a[row * n + row];
        }
    }
    x
}

fn free_units(clamped: &[usize]) -> Vec<usize> {
    (0..3).filter(|unit| !clamped.contains(unit)).collect()
}

/// Total effect of each clamped unit on each passive unit when `clamped` is held,
/// `(I - K_FF)^-1 K_FJ`, laid out `[passive][clamped]`.
fn clamp_response(k: &[f64; 9], clamped: &[usize]) -> Vec<f64> {
    let free = free_units(clamped);
    let (f, j) = (free.len(), clamped.len());
    let mut a = vec![0.0; f * f];
    let mut b = vec![0.0; f * j];
    for (p, &row) in free.iter().enumerate() {
        for (q, &col) in free.iter().enumerate() {
            a[p * f + q] = if p == q { 1.0 } else { 0.0 } - k[row * 3 + col];
        }
        for (q, &col) in clamped.iter().enumerate() {
            b[p * j + q] = k[row * 3 + col];
        }
    }
    solve_small(a, b, f, j)
}

/// Least-squares response of each passive column on the clamped columns of a
/// sample, laid out `[passive][clamped]`.
fn empirical_response(data: &Columns, clamped: &[usize]) -> Vec<f64> {
    let cols = [&data.x, &data.y, &data.z];
    let n = data.x.len() as f64;
    let mean = |column: &Vec<f64>| column.iter().sum::<f64>() / n;
    let cross = |a: usize, b: usize| {
        let (ma, mb) = (mean(cols[a]), mean(cols[b]));
        cols[a].iter().zip(cols[b]).map(|(u, v)| (u - ma) * (v - mb)).sum::<f64>() / n
    };
    let free = free_units(clamped);
    let (f, j) = (free.len(), clamped.len());
    let mut sxx = vec![0.0; j * j];
    let mut sxy = vec![0.0; j * f];
    for (p, &a) in clamped.iter().enumerate() {
        for (q, &b) in clamped.iter().enumerate() {
            sxx[p * j + q] = cross(a, b);
        }
        for (q, &b) in free.iter().enumerate() {
            sxy[p * f + q] = cross(a, b);
        }
    }
    let beta = solve_small(sxx, sxy, j, f);
    let mut out = vec![0.0; f * j];
    for p in 0..j {
        for q in 0..f {
            out[q * j + p] = beta[p * f + q];
        }
    }
    out
}

/// Direct-effect matrix `K[into * 3 + from]` read from the recovered blocks. A
/// target without a recovered block keeps zeros and is the caller's to check.
fn recovered_matrix(assembled: &AssembledExperimentConstraints) -> [f64; 9] {
    let mut k = [0.0; 9];
    for block in &assembled.blocks {
        let AssembledCoefficientRecovery::Coefficients { target, sources, values } =
            recover_assembled_coefficients(block, None, WeakStability::NotDeclared)
        else {
            panic!("the block into {:?} did not solve", block.target);
        };
        let into = usize::try_from(target.raw()).expect("small unit");
        for (source, value) in sources.iter().zip(&values) {
            let from = usize::try_from(source.raw()).expect("small unit");
            k[into * 3 + from] = *value;
        }
    }
    k
}

fn clamp_units(clamped: &[usize]) -> Vec<u32> {
    clamped.iter().map(|&unit| u32::try_from(unit).expect("small unit")).collect()
}

#[test]
fn recovery_from_single_clamps_predicts_entire_held_out_joint_regimes() {
    let assembled = assemble_designs(&SINGLES, 3000);
    assert!(assembled.uncovered.is_empty());
    let recovered = recovered_matrix(&assembled);
    let truth = truth_matrix();
    for (index, expected) in truth.iter().enumerate() {
        near(recovered[index], *expected, 0.03, &format!("recovered direct effect {index}"));
    }
    // None of these clamp sets is a training regime. Each sample is a fresh stream.
    let held_out: [(&[usize], u64); 3] = [(&[0, 1], 77_001), (&[1, 2], 77_002), (&[0, 2], 77_003)];
    for (clamped, seed) in held_out {
        let predicted = clamp_response(&recovered, clamped);
        let exact = clamp_response(&truth, clamped);
        let sample = simulate(ROWS, &clamp_units(clamped), seed);
        let observed = empirical_response(&sample, clamped);
        assert_eq!(predicted.len(), observed.len());
        for ((value, truth_value), sample_value) in predicted.iter().zip(&exact).zip(&observed) {
            near(*value, *truth_value, 0.03, "held-out response against the truth");
            near(*value, *sample_value, 0.03, "held-out response against the sample");
        }
    }
}

#[test]
fn a_wrong_direct_effect_is_visible_in_the_held_out_joint_regime() {
    // The held-out comparison must be able to fail: bias the effect of y on z by 0.2
    // and the joint clamp of x and y no longer matches the sample it never saw.
    let assembled = assemble_designs(&SINGLES, 3000);
    let mut biased = recovered_matrix(&assembled);
    biased[7] += 0.2;
    let clamped: &[usize] = &[0, 1];
    let observed = empirical_response(&simulate(ROWS, &clamp_units(clamped), 77_001), clamped);
    let predicted = clamp_response(&biased, clamped);
    assert!((predicted[1] - observed[1]).abs() > 0.1, "{predicted:?} against {observed:?}");
}

#[test]
fn inadequate_designs_refuse_or_leave_pairs_uncovered_and_fill_nothing() {
    // One clamp alone covers no target: y and z are never clamped.
    match try_assemble_designs(&[(&[0], 21)], 4000) {
        Err(MomentAssemblyRefusal::InadequateInterventionCoverage { missing }) => {
            assert_eq!(missing.len(), 4);
            for (from, into) in [(1, 0), (1, 2), (2, 0), (2, 1)] {
                assert!(missing.contains(&MissingPair { from: unit(from), into: unit(into) }));
            }
        }
        other => panic!("one clamp cannot cover any target: {other:?}"),
    }
    // Clamping x and z leaves y never clamped: only the equation into y is covered.
    let partial = assemble_designs(&[(&[0], 22), (&[2], 23)], 4100);
    assert_eq!(
        partial.uncovered,
        vec![
            MissingPair { from: unit(1), into: unit(0) },
            MissingPair { from: unit(1), into: unit(2) },
        ]
    );
    assert!(partial.blocks.iter().all(|block| block.target == unit(1)));
    assert_eq!(partial.blocks.len(), 1);
    let recovered = recovered_matrix(&partial);
    near(recovered[3], B_YX, 0.03, "effect of x on y");
    near(recovered[5], 0.0, 0.03, "effect of z on y");
    // The equations into x and z have no block, so nothing was filled for them.
    for entry in [1, 2, 6, 7] {
        assert!(recovered[entry].abs() < f64::EPSILON, "entry {entry} was filled");
    }
}
