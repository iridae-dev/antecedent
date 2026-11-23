// SPDX-License-Identifier: MIT OR Apache-2.0
//! Limited-memory inverse-curvature products for local smooth cyclic likelihoods.
//! The two-loop update is the same standard L-BFGS construction used by the
//! repository's NOTEARS solver; this helper admits no graph or likelihood claim.
use crate::util::dot;

#[derive(Default)]
pub(crate) struct InverseCurvature {
    history: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)>,
}
impl InverseCurvature {
    pub(crate) fn update(
        &mut self,
        old: &[f64],
        new: &[f64],
        old_score: &[f64],
        new_score: &[f64],
    ) {
        let s: Vec<_> = new.iter().zip(old).map(|(a, b)| a - b).collect();
        let y: Vec<_> = old_score.iter().zip(new_score).map(|(a, b)| a - b).collect();
        let curvature = dot(&s, &y);
        let scale = (dot(&s, &s) * dot(&y, &y)).sqrt();
        if curvature.is_finite() && curvature > 1e-12 * scale && curvature > 0.0 {
            if self.history.len() == 12 {
                self.history.pop_front();
            }
            self.history.push_back((s, y, 1.0 / curvature));
        }
    }
    pub(crate) fn direction(&mut self, score: &[f64]) -> (Vec<f64>, bool) {
        if self.history.is_empty() {
            return (score.to_vec(), false);
        }
        let mut q = score.to_vec();
        let mut alpha = Vec::new();
        for (s, y, rho) in self.history.iter().rev() {
            let a = rho * dot(s, &q);
            alpha.push(a);
            for (q, y) in q.iter_mut().zip(y) {
                *q -= a * y;
            }
        }
        let Some((s, y, _)) = self.history.back() else {
            return (score.to_vec(), false);
        };
        let gamma = dot(s, y) / dot(y, y);
        let mut output: Vec<_> = q.iter().map(|q| gamma * q).collect();
        for ((s, y, rho), a) in self.history.iter().zip(alpha.into_iter().rev()) {
            let beta = rho * dot(y, &output);
            for (r, s) in output.iter_mut().zip(s) {
                *r += s * (a - beta);
            }
        }
        if output.iter().any(|v| !v.is_finite()) || dot(&output, score) <= 0.0 {
            self.history.clear();
            (score.to_vec(), false)
        } else {
            (output, true)
        }
    }
}
