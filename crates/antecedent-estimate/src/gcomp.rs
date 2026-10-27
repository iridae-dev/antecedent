//! Shared g-computation contrast kernel (GLM + Bayesian).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::GlmFamily;

/// A column-major design with its treatment column and the two intervention values a
/// g-computation contrast compares.
///
/// The one description of "predict every row with the treatment column set to `active`,
/// then to `control`" shared by the GLM adjustment, its delta-method gradient and the
/// Bayesian posterior-draw evaluator.
#[derive(Clone, Copy, Debug)]
pub struct GcompContrast<'a> {
    /// Mean/link family of the fitted model.
    pub family: GlmFamily,
    /// Column-major design (`nrows * ncols`), treatment column included.
    pub x_colmajor: &'a [f64],
    /// Row count.
    pub nrows: usize,
    /// Column count.
    pub ncols: usize,
    /// Treatment column index within the design.
    pub t_col: usize,
    /// Active intervention value.
    pub active: f64,
    /// Control intervention value.
    pub control: f64,
}

/// Rows processed together by the column-outer kernels, so the per-row accumulators stay
/// in L1 while each design column is streamed once.
const ROW_BLOCK: usize = 1024;

/// Linear predictors `η_r = Σ_c x[c·nrows + r] · coefficients[c]` of every row of a
/// column-major `nrows × ncols` design.
///
/// Each design column is read contiguously. Every row accumulates its terms in column
/// order from `0.0`, the order of a row-wise dot product, so the values are bit-identical
/// to it.
///
/// # Panics
///
/// `coefficients` is shorter than `ncols` or `x_colmajor` shorter than `nrows * ncols`.
#[must_use]
pub(crate) fn linear_predictors(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    coefficients: &[f64],
) -> Vec<f64> {
    assert!(coefficients.len() >= ncols, "one coefficient per design column");
    assert!(x_colmajor.len() >= nrows * ncols, "design shorter than nrows * ncols");
    let mut eta = vec![0.0; nrows];
    for (column, &coef) in x_colmajor.chunks_exact(nrows.max(1)).zip(coefficients).take(ncols) {
        for (slot, &x) in eta.iter_mut().zip(column) {
            *slot += x * coef;
        }
    }
    eta
}

impl GcompContrast<'_> {
    /// Per-row mean-scale contrast `μ(T=active, Z) − μ(T=control, Z)`.
    ///
    /// Column-outer: each design column is streamed once into the two arms' linear
    /// predictors. Row `r` accumulates the same terms in the same order as
    /// the row-first single-row contrast, so the values are bit-identical to it.
    #[must_use]
    pub fn diffs(&self, coefficients: &[f64]) -> Vec<f64> {
        assert!(coefficients.len() >= self.ncols, "one coefficient per design column");
        let n = self.nrows;
        let mut eta_active = vec![0.0; n];
        let mut eta_control = vec![0.0; n];
        for (c, &coef) in coefficients.iter().enumerate().take(self.ncols) {
            if c == self.t_col {
                let (active, control) = (self.active * coef, self.control * coef);
                for (a, k) in eta_active.iter_mut().zip(&mut eta_control) {
                    *a += active;
                    *k += control;
                }
            } else {
                let column = &self.x_colmajor[c * n..(c + 1) * n];
                for ((a, k), &x) in eta_active.iter_mut().zip(&mut eta_control).zip(column) {
                    let term = x * coef;
                    *a += term;
                    *k += term;
                }
            }
        }
        eta_active
            .iter()
            .zip(&eta_control)
            .map(|(&active, &control)| {
                self.family.mean_from_eta(active) - self.family.mean_from_eta(control)
            })
            .collect()
    }

    /// Linear predictors of one row under the active and control interventions.
    ///
    /// Column order is the row-wise dot product: treatment terms use `active` and `control`,
    /// and every other column adds `x * β` to both arms. The row-first contrast and the
    /// GLM delta-method gradient both start from this pair.
    #[must_use]
    pub(crate) fn contrast_row_etas(&self, coefficients: &[f64], row: usize) -> (f64, f64) {
        assert!(coefficients.len() >= self.ncols, "one coefficient per design column");
        let mut eta_active = 0.0;
        let mut eta_control = 0.0;
        for (c, &coef) in coefficients.iter().enumerate().take(self.ncols) {
            if c == self.t_col {
                eta_active += self.active * coef;
                eta_control += self.control * coef;
            } else {
                let val = self.x_colmajor[c * self.nrows + row];
                eta_active += val * coef;
                eta_control += val * coef;
            }
        }
        (eta_active, eta_control)
    }

    /// Single-row mean-scale contrast; the row-first oracle of the column-outer kernels.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn row_contrast(&self, coefficients: &[f64], row: usize) -> f64 {
        let (eta_active, eta_control) = self.contrast_row_etas(coefficients, row);
        self.family.mean_from_eta(eta_active) - self.family.mean_from_eta(eta_control)
    }

    /// Row-averaged mean-scale contrast over every row.
    ///
    /// The identity link needs no data pass: only the treatment column differs, so the
    /// average is exactly `(active − control) β_T`. Nonlinear links share one
    /// residualized `η` (covariates only) between the two arms, accumulated column-outer
    /// over blocks of rows; each row adds its covariate terms in column order and the
    /// rows are summed in row order, so the value is bit-identical to a row-by-row walk.
    #[must_use]
    pub fn mean_contrast(&self, beta: &[f64]) -> f64 {
        if matches!(self.family, GlmFamily::GaussianIdentity) {
            return (self.active - self.control) * beta[self.t_col];
        }
        assert!(beta.len() >= self.ncols, "one coefficient per design column");
        let beta_t = beta[self.t_col];
        let n = self.nrows;
        let mut sum = 0.0;
        let mut eta = [0.0_f64; ROW_BLOCK];
        let mut start = 0;
        while start < n {
            let len = ROW_BLOCK.min(n - start);
            let block = &mut eta[..len];
            block.fill(0.0);
            for (c, &b) in beta.iter().enumerate().take(self.ncols) {
                if c == self.t_col {
                    continue;
                }
                let column = &self.x_colmajor[c * n + start..c * n + start + len];
                for (slot, &x) in block.iter_mut().zip(column) {
                    *slot += x * b;
                }
            }
            for &e in block.iter() {
                sum += self.family.mean_from_eta(e + beta_t * self.active)
                    - self.family.mean_from_eta(e + beta_t * self.control);
            }
            start += len;
        }
        sum / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic column-major design with an intercept and `ncols - 1` smooth columns.
    fn design(nrows: usize, ncols: usize) -> Vec<f64> {
        let mut x = vec![0.0; nrows * ncols];
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        for (index, value) in x.iter_mut().enumerate() {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *value = if index < nrows { 1.0 } else { ((state >> 33) as f64 / 2f64.powi(31)) - 0.5 };
        }
        x
    }

    /// Row-by-row references of the column-outer kernels.
    fn row_first_diffs(contrast: &GcompContrast<'_>, beta: &[f64]) -> Vec<f64> {
        (0..contrast.nrows).map(|row| contrast.row_contrast(beta, row)).collect()
    }

    fn row_first_mean_contrast(contrast: &GcompContrast<'_>, beta: &[f64]) -> f64 {
        let beta_t = beta[contrast.t_col];
        let mut sum = 0.0;
        for r in 0..contrast.nrows {
            let mut eta = 0.0;
            for (c, &b) in beta.iter().enumerate().take(contrast.ncols) {
                if c == contrast.t_col {
                    continue;
                }
                eta += contrast.x_colmajor[c * contrast.nrows + r] * b;
            }
            sum += contrast.family.mean_from_eta(eta + beta_t * contrast.active)
                - contrast.family.mean_from_eta(eta + beta_t * contrast.control);
        }
        sum / contrast.nrows as f64
    }

    #[test]
    fn column_outer_kernels_are_bit_identical_to_the_row_first_walk() {
        // 2 500 rows: two full row blocks and a partial one; the treatment column sits in
        // the middle so skipped-column handling is exercised.
        let (nrows, ncols) = (2_500, 6);
        let x = design(nrows, ncols);
        let beta: Vec<f64> = (0..ncols).map(|c| 0.3 - 0.11 * c as f64).collect();
        for family in [
            GlmFamily::GaussianIdentity,
            GlmFamily::BinomialLogit,
            GlmFamily::BinomialProbit,
            GlmFamily::PoissonLog,
        ] {
            let contrast = GcompContrast {
                family,
                x_colmajor: &x,
                nrows,
                ncols,
                t_col: 2,
                active: 1.0,
                control: 0.0,
            };
            let diffs = contrast.diffs(&beta);
            let reference = row_first_diffs(&contrast, &beta);
            assert!(diffs.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()));
            if !matches!(family, GlmFamily::GaussianIdentity) {
                assert_eq!(
                    contrast.mean_contrast(&beta).to_bits(),
                    row_first_mean_contrast(&contrast, &beta).to_bits(),
                    "{family:?}"
                );
            }
            let eta = linear_predictors(&x, nrows, ncols, &beta);
            for (row, &value) in eta.iter().enumerate() {
                let mut by_row = 0.0;
                for (c, &b) in beta.iter().enumerate() {
                    by_row += x[c * nrows + row] * b;
                }
                assert_eq!(value.to_bits(), by_row.to_bits());
            }
        }
    }

    #[test]
    fn standardized_logit_contrast_matches_the_hand_formula_and_interaction_columns_are_not_intervened() {
        // Two rows, columns [intercept, T, Z]; T is observed at an arbitrary value that must
        // not matter because it is replaced by the intervention.
        let x = [1.0, 1.0, 0.7, 0.2, -1.0, 1.0];
        let beta = [0.2, 0.7, 0.5];
        let contrast = GcompContrast {
            family: GlmFamily::BinomialLogit,
            x_colmajor: &x,
            nrows: 2,
            ncols: 3,
            t_col: 1,
            active: 1.0,
            control: 0.0,
        };
        let logistic = |eta: f64| 1.0 / (1.0 + (-eta).exp());
        let expected = 0.5
            * ((logistic(0.2 + 0.7 - 0.5) - logistic(0.2 - 0.5))
                + (logistic(0.2 + 0.7 + 0.5) - logistic(0.2 + 0.5)));
        assert!((contrast.mean_contrast(&beta) - expected).abs() < 1e-14);
        let diffs = contrast.diffs(&beta);
        assert!(((diffs[0] + diffs[1]) / 2.0 - expected).abs() < 1e-14);

        // Counterexample: a treatment-derived design column (T*Z) is held at its observed
        // value, so the identity-link shortcut equals (active - control) * beta_T and omits
        // the interaction term of the true do(T) contrast (beta_T + beta_TZ * mean Z). The
        // kernel is therefore valid only for main-effect designs, which the design compiler
        // produces (intercept, treatment, covariate main effects).
        let z = [-1.0, 1.0];
        let observed_t = [1.0, 0.0];
        let design: Vec<f64> = [1.0, 1.0]
            .into_iter()
            .chain(observed_t)
            .chain(z)
            .chain([observed_t[0] * z[0], observed_t[1] * z[1]])
            .collect();
        let with_interaction = GcompContrast {
            family: GlmFamily::GaussianIdentity,
            x_colmajor: &design,
            nrows: 2,
            ncols: 4,
            t_col: 1,
            active: 1.0,
            control: 0.0,
        };
        let coefficients = [0.0, 1.0, 0.0, 2.0];
        let kernel = with_interaction.mean_contrast(&coefficients);
        let true_do_contrast = 1.0 + 2.0 * (z[0] + z[1]) / 2.0;
        assert!((kernel - 1.0).abs() < 1e-14);
        assert!((true_do_contrast - 1.0).abs() < 1e-14, "mean Z is zero here");
        // With a nonzero mean covariate the shortcut and the true intervention part ways.
        let z_shift = [0.5, 1.5];
        let design_shift: Vec<f64> = [1.0, 1.0]
            .into_iter()
            .chain(observed_t)
            .chain(z_shift)
            .chain([observed_t[0] * z_shift[0], observed_t[1] * z_shift[1]])
            .collect();
        let shifted = GcompContrast { x_colmajor: &design_shift, ..with_interaction };
        let true_shifted = 1.0 + 2.0 * (z_shift[0] + z_shift[1]) / 2.0;
        assert!((shifted.mean_contrast(&coefficients) - 1.0).abs() < 1e-14);
        assert!((true_shifted - 3.0).abs() < 1e-14);
        assert!((shifted.mean_contrast(&coefficients) - true_shifted).abs() > 1.0);
    }

    fn timed<R>(label: &str, reps: usize, mut work: impl FnMut() -> R) -> R {
        let start = std::time::Instant::now();
        let mut last = None;
        for _ in 0..reps {
            last = Some(std::hint::black_box(work()));
        }
        println!("{label}: {:.3} ms/call", start.elapsed().as_secs_f64() * 1e3 / reps as f64);
        last.unwrap()
    }

    #[test]
    #[ignore = "timing harness: cargo test --release -p antecedent-estimate --lib timing_ -- --ignored --nocapture"]
    fn timing_gcomp_contrast_kernels() {
        let (nrows, ncols) = (200_000, 10);
        let x = design(nrows, ncols);
        let beta: Vec<f64> = (0..ncols).map(|c| 0.05 * (c as f64 + 1.0)).collect();
        for family in [GlmFamily::BinomialLogit, GlmFamily::PoissonLog] {
            let contrast = GcompContrast {
                family,
                x_colmajor: &x,
                nrows,
                ncols,
                t_col: 1,
                active: 1.0,
                control: 0.0,
            };
            timed(&format!("mean_contrast {family:?}"), 20, || contrast.mean_contrast(&beta));
            timed(&format!("diffs {family:?}"), 20, || contrast.diffs(&beta));
        }
    }
}
