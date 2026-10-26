//! Independent case-weight perturbation oracle shared by the weighting tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::overlap::IpwTarget;

// Independent case-weight perturbation oracle: refit the logistic MLE and
// weighted means after infinitesimally changing each observation's mass.
pub(super) fn oracle_fit(
    t: &[f64],
    y: &[f64],
    x: &[f64],
    mass: &[f64],
    target: IpwTarget,
    clip: Option<f64>,
    trim: Option<f64>,
) -> (f64, Vec<f64>) {
    let mut beta = [0.0_f64; 2];
    for _ in 0..100 {
        let mut score = [0.0; 2];
        let mut information = [0.0; 3];
        for i in 0..t.len() {
            let e = antecedent_kernels::sigmoid(beta[0] + beta[1] * x[i]);
            let residual = mass[i] * (t[i] - e);
            let v = mass[i] * e * (1.0 - e);
            score[0] += residual;
            score[1] += residual * x[i];
            information[0] += v;
            information[1] += v * x[i];
            information[2] += v * x[i] * x[i];
        }
        let determinant = information[0] * information[2] - information[1].powi(2);
        let step = [
            (information[2] * score[0] - information[1] * score[1]) / determinant,
            (information[0] * score[1] - information[1] * score[0]) / determinant,
        ];
        beta[0] += step[0];
        beta[1] += step[1];
        if step[0].abs().max(step[1].abs()) < 1e-13 {
            break;
        }
    }
    let e: Vec<f64> =
        x.iter().map(|x| antecedent_kernels::sigmoid(beta[0] + beta[1] * x)).collect();
    let mut numerator = [0.0; 2];
    let mut denominator = [0.0; 2];
    for i in 0..t.len() {
        if trim.is_some_and(|c| e[i] < c || e[i] > 1.0 - c) {
            continue;
        }
        let score = clip.map_or(e[i], |c| crate::overlap::clip_propensity(e[i], c));
        let arm = usize::from(t[i] > 0.5);
        let w = mass[i] * target.weight(t[i], score);
        numerator[arm] += w * y[i];
        denominator[arm] += w;
    }
    (numerator[1] / denominator[1] - numerator[0] / denominator[0], e)
}
