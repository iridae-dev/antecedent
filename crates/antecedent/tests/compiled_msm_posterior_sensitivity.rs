//! The posterior sensitivity ranges of a marginal structural model's period parameters.
//!
//! A longitudinal regime study fitted with the marginal structural model method reads its own
//! weighted design (observed subjects, stabilized weights) as a weighted likelihood with the
//! weights normalized to the effective sample size, under the reference prior, and draws its
//! conjugate posterior under a keyed stream of the context seed. The twins call
//! `component_evalue_posterior_range` and `component_ovb_posterior_range` on the same draws and
//! assert bit equality of the ranges. The independent check writes the weighted least-squares
//! normal equations here and compares the draws with the closed form of the conjugate posterior:
//! the parameters centred at the weighted least-squares fit with covariance
//! `RSS / (n_eff - p - 2) (X^T W X)^-1`, and `E[sigma^2] = RSS / (n_eff - p - 2)`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "bits are compared exactly; the fixture is small and uses model notation"
)]

use std::sync::Arc;

use antecedent::analysis::sensitivity_range::{SensitivityRangeOutcome, SensitivityRangeRequest};
use antecedent::analysis::target_validation::result_quadrant;
use antecedent::{PreparedStudy, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, LongitudinalRegimeMethod, LongitudinalRegimeQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_validate::target_native::{RefutationTarget, TargetFamily};
use antecedent_validate::{
    AssumptionRange, ComponentModel, EffectScale, OvbPosterior, component_evalue_posterior_range,
    component_ovb_posterior_range,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x4D53_4D50)
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    antecedent::support::caller_thread()
        .stack_size(16 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// `SplitMix64` uniform stream.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, w) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * w).cos()
    }

    fn bernoulli(&mut self, p: f64) -> f64 {
        if self.next() < p { 1.0 } else { 0.0 }
    }
}

fn same_range(compiled: &AssumptionRange, direct: &AssumptionRange) {
    assert_eq!(compiled.target.identity(), direct.target.identity(), "target");
    assert_eq!(compiled.assumption, direct.assumption);
    assert_eq!(compiled.tested.len(), direct.tested.len());
    for (a, b) in compiled.tested.iter().zip(direct.tested.iter()) {
        assert_eq!(a.parameter.to_bits(), b.parameter.to_bits());
        assert_eq!(a.target.to_bits(), b.target.to_bits(), "target at {}", a.parameter);
        assert_eq!(a.test.map(f64::to_bits), b.test.map(f64::to_bits), "test at {}", a.parameter);
    }
    assert_eq!(compiled.break_even.map(f64::to_bits), direct.break_even.map(f64::to_bits));
    assert_eq!(
        compiled.break_even_significance.map(f64::to_bits),
        direct.break_even_significance.map(f64::to_bits)
    );
    assert_eq!(compiled.identity, direct.identity);
}

fn close(a: f64, b: f64, relative: f64) -> bool {
    (a - b).abs() <= relative * a.abs().max(b.abs()).max(1e-12)
}

/// A two-period sequentially randomized trial with known, history-dependent probabilities,
/// fitted with the marginal structural model method.
struct Regime {
    y: Vec<f64>,
    l1: Vec<f64>,
    l2: Vec<f64>,
    query: LongitudinalRegimeQuery,
}

impl Regime {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = Stream(seed);
        let (mut y, mut l1, mut l2) = (vec![], vec![], vec![]);
        let (mut history, mut probabilities) = (vec![], vec![]);
        for _ in 0..n {
            let first = s.normal();
            let p1 = if first > 0.0 { 0.7 } else { 0.3 };
            let a1 = s.bernoulli(p1) == 1.0;
            let second = 0.5 * f64::from(a1) + s.normal();
            let p2 = if second > 0.0 { 0.65 } else { 0.35 };
            let a2 = s.bernoulli(p2) == 1.0;
            y.push(1.0 + 2.0 * f64::from(a1) + 1.5 * f64::from(a2) + 0.8 * first + s.normal());
            l1.push(first);
            l2.push(second);
            history.extend([a1, a2]);
            probabilities.extend([p1, p2]);
        }
        let query = LongitudinalRegimeQuery {
            calibration_design: None,
            outcome: v(0),
            method: LongitudinalRegimeMethod::MarginalStructuralModel,
            period_outcome_predictions: None,
            known_fixed_outcome_predictions: false,
            stabilizing_numerator_probabilities: Some(Arc::from([0.5, 0.5])),
            q_predictions: None,
            observation_history: None,
            prediction_fold_ids: None,
            periods: 2,
            treatment_history: history.into(),
            regime_actions: vec![false; 2 * n].into(),
            treatment_probabilities: probabilities.into(),
            censoring_probabilities: vec![1.0; 2 * n].into(),
            outcome_observed: vec![true; n].into(),
            subject_ids: (0..n)
                .map(|i| Arc::<str>::from(format!("s{i}")))
                .collect::<Vec<_>>()
                .into(),
            fold_ids: (0..n).map(|i| (i % 5) as u32).collect::<Vec<_>>().into(),
            excluded_fold_predictions: false,
            probabilities_known_by_design: true,
            minimum_probability: 0.01,
            rule_id: None,
            rule_version: None,
            rule_provenance: None,
        };
        Self { y, l1, l2, query }
    }

    fn study(&self) -> (PreparedStudy, StudyResult) {
        let data = TabularData::from_f64_columns([
            ("outcome", self.y.as_slice()),
            ("l1", self.l1.as_slice()),
            ("l2", self.l2.as_slice()),
        ])
        .unwrap();
        let context = ctx();
        let prepared = Study::tabular(data.clone())
            .query(CausalQuery::LongitudinalRegime(self.query.clone()))
            .build()
            .unwrap()
            .prepare(&context)
            .unwrap();
        let result = prepared.estimate(&data, &context).unwrap();
        (prepared, result)
    }

    /// Exposures and stabilized weights, written out here.
    fn design(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = self.y.len();
        let mut exposures = vec![Vec::with_capacity(n), Vec::with_capacity(n)];
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let mut weight = 1.0;
            for t in 0..2 {
                let j = 2 * i + t;
                let a = self.query.treatment_history[j];
                let p = self.query.treatment_probabilities[j];
                weight *= if a { 0.5 / p } else { 0.5 / (1.0 - p) };
                exposures[t].push(f64::from(a));
            }
            weights.push(weight);
        }
        (exposures, weights)
    }
}

/// The inverse of a small symmetric positive definite matrix by Gauss-Jordan elimination.
fn inverse(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut out = row.clone();
            out.extend((0..p).map(|j| if i == j { 1.0 } else { 0.0 }));
            out
        })
        .collect();
    for k in 0..p {
        let pivot = (k..p).max_by(|&x, &z| a[x][k].abs().total_cmp(&a[z][k].abs())).unwrap();
        a.swap(k, pivot);
        let scale = a[k][k];
        for j in 0..2 * p {
            a[k][j] /= scale;
        }
        for i in 0..p {
            if i != k {
                let factor = a[i][k];
                for j in 0..2 * p {
                    a[i][j] -= factor * a[k][j];
                }
            }
        }
    }
    a.into_iter().map(|row| row[p..].to_vec()).collect()
}

fn target(result: &StudyResult) -> RefutationTarget {
    RefutationTarget::new(
        TargetFamily::JointContrast,
        result_quadrant(result),
        ["period[0]", "period[1]"],
        *result.lowered_program_identity().unwrap().as_bytes(),
    )
    .unwrap()
}

#[test]
fn compiled_msm_posterior_ranges_equal_the_direct_component_ranges_on_its_draws() {
    on_large_stack(|| {
        let fixture = Regime::new(600, 45);
        let (study, result) = fixture.study();
        let request =
            SensitivityRangeRequest::MsmParameterEValuePosterior { level: 0.9, draws: 800 };
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MsmParameterPosterior { posterior, evalue, range } = outcome
        else {
            panic!("an MSM posterior outcome");
        };
        let (summary, direct) = component_evalue_posterior_range(
            target(&result),
            &posterior.standardized(),
            EffectScale::StandardizedMeanDifference,
            0.9,
            ComponentModel::MarginalStructural,
        )
        .unwrap();
        assert_eq!(format!("{:?}", evalue.unwrap()), format!("{summary:?}"));
        same_range(&range, &direct);
        // The draws are the context seed's: the same request draws the same posterior.
        let again = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MsmParameterPosterior { posterior: repeated, .. } = again
        else {
            panic!("an MSM posterior outcome");
        };
        assert_eq!(repeated, posterior);

        let request = SensitivityRangeRequest::MsmParametersOvbPosterior {
            level: 0.9,
            r2yz_extreme: 0.5,
            draws: 800,
        };
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MsmParameterPosterior { posterior, evalue, range } = outcome
        else {
            panic!("an MSM posterior outcome");
        };
        assert!(evalue.is_none());
        let direct = component_ovb_posterior_range(
            target(&result),
            &OvbPosterior {
                coefficients: &posterior.coefficients,
                bias_scale: &posterior.bias_scale(),
            },
            0.9,
            0.5,
            ComponentModel::MarginalStructural,
        )
        .unwrap();
        same_range(&range, &direct);
        // The weights are the route's stabilized weights.
        let (_, weights) = fixture.design();
        for (a, b) in posterior.weights.iter().zip(&weights) {
            assert!(close(*a, *b, 1e-12));
        }
        // Too few draws are refused.
        let refused = study.sensitivity_range(
            &result,
            &SensitivityRangeRequest::MsmParameterEValuePosterior { level: 0.9, draws: 1 },
            &ctx(),
        );
        assert!(refused.is_err());
    });
}

#[test]
fn the_msm_posterior_is_the_conjugate_weighted_least_squares_posterior() {
    on_large_stack(|| {
        let fixture = Regime::new(600, 45);
        let (study, result) = fixture.study();
        let draws = 20_000;
        let request = SensitivityRangeRequest::MsmParameterEValuePosterior { level: 0.95, draws };
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MsmParameterPosterior { posterior, .. } = outcome else {
            panic!("an MSM posterior outcome");
        };

        // The weighted design, written out here: weights normalized to the effective size.
        let (exposures, weights) = fixture.design();
        let n = weights.len();
        let total: f64 = weights.iter().sum();
        let effective = total * total / weights.iter().map(|w| w * w).sum::<f64>();
        let normalized: Vec<f64> = weights.iter().map(|w| w * effective / total).collect();
        let row = |i: usize| [1.0, exposures[0][i], exposures[1][i]];
        let mut gram = vec![vec![0.0; 3]; 3];
        let mut moment = [0.0; 3];
        for i in 0..n {
            let x = row(i);
            for a in 0..3 {
                moment[a] += normalized[i] * x[a] * fixture.y[i];
                for b in 0..3 {
                    gram[a][b] += normalized[i] * x[a] * x[b];
                }
            }
        }
        let inv = inverse(&gram);
        let beta: Vec<f64> =
            (0..3).map(|a| (0..3).map(|b| inv[a][b] * moment[b]).sum::<f64>()).collect();
        let rss: f64 = (0..n)
            .map(|i| {
                let x = row(i);
                let fitted: f64 = (0..3).map(|a| x[a] * beta[a]).sum();
                normalized[i] * (fixture.y[i] - fitted).powi(2)
            })
            .sum();
        let dof = effective - 3.0;

        // The fit, the residual sum of squares, the degrees of freedom and the exposures'
        // residual scales are the closed form.
        assert!(close(posterior.dof, dof, 1e-12), "{} against {dof}", posterior.dof);
        assert!(close(posterior.rss, rss, 1e-9), "{} against {rss}", posterior.rss);
        for t in 0..2 {
            assert!(close(posterior.estimates[t], beta[t + 1], 1e-9));
            // Frisch-Waugh: RSS(A_t ~ 1 + A_-t) = 1 / [(X^T W X)^-1]_tt.
            let scale = (1.0 / (inv[t + 1][t + 1] * dof)).sqrt();
            assert!(close(posterior.exposure_scale[t], scale, 1e-9));
        }

        // The draws follow the conjugate posterior: a Student-t centred at the fit with scale
        // matrix RSS / dof (X^T W X)^-1, and sigma^2 = RSS / chi2_dof.
        let d = draws as f64;
        let mean_variance = rss / (dof - 2.0);
        for t in 0..2 {
            let column = &posterior.coefficients[t];
            let mean = column.iter().sum::<f64>() / d;
            let variance = column.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / (d - 1.0);
            let expected = mean_variance * inv[t + 1][t + 1];
            assert!(
                (mean - beta[t + 1]).abs() <= 4.0 * (expected / d).sqrt(),
                "period {t}: mean {mean} against {}",
                beta[t + 1]
            );
            assert!(close(variance, expected, 0.05), "period {t}: {variance} against {expected}");
        }
        let sigma2 = posterior.sigma.iter().map(|s| s * s).sum::<f64>() / d;
        assert!(close(sigma2, mean_variance, 0.02), "{sigma2} against {mean_variance}");
        // The E-value reads each parameter draw over the sigma of the same draw.
        let standardized = posterior.standardized();
        for t in 0..2 {
            for k in [0, draws / 2, draws - 1] {
                assert_eq!(
                    standardized[t][k].to_bits(),
                    (posterior.coefficients[t][k] / posterior.sigma[k]).to_bits()
                );
            }
        }
        // The route's own period effects are the centre of the posterior.
        let published = result.longitudinal_regime().unwrap().period_effects.clone().unwrap();
        for t in 0..2 {
            assert!(close(posterior.estimates[t], published[t], 1e-9));
        }
    });
}
