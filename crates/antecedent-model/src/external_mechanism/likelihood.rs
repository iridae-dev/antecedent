//! Equilibrium likelihood and analytic scores for resolved mechanisms.

use antecedent_core::ExecutionContext;

use crate::nonlinear_equilibrium::{EquilibriumError, RegimeRow, cholesky};

use super::{
    DerivativeDeclaration, ExternalFitParameter, LikelihoodTerms, ResolvedFeedbackSystem,
    ResolvedKind, invalid,
};

/// Per-row quantities shared by the likelihood and its scores.
struct RowTerms {
    free: Vec<usize>,
    blocks: Vec<Vec<usize>>,
    gradients: Vec<Vec<f64>>,
    whitened_residual_weights: Vec<f64>,
    data: f64,
    log_determinant: f64,
}

fn log_abs_determinant(mut a: Vec<f64>, k: usize) -> Result<f64, EquilibriumError> {
    let mut sum = 0.0;
    for col in 0..k {
        let mut pivot = col;
        for r in col + 1..k {
            if a[r * k + col].abs() > a[pivot * k + col].abs() {
                pivot = r;
            }
        }
        let p = a[pivot * k + col];
        if p == 0.0 || !p.is_finite() {
            return Err(EquilibriumError::Singular);
        }
        if pivot != col {
            for c in 0..k {
                a.swap(pivot * k + c, col * k + c);
            }
        }
        sum += p.abs().ln();
        for r in col + 1..k {
            let factor = a[r * k + col] / p;
            if factor != 0.0 {
                for c in col..k {
                    a[r * k + c] -= factor * a[col * k + c];
                }
            }
        }
    }
    Ok(sum)
}

impl ResolvedFeedbackSystem {
    #[allow(
        clippy::float_cmp,
        reason = "observed hard clamps must equal the declared structural value exactly"
    )]
    fn check_row(&self, row: &RegimeRow) -> Result<Vec<usize>, EquilibriumError> {
        let n = self.system.order();
        if row.values.len() != n || row.values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("state/noise dimension or nonfinite"));
        }
        let mut seen = vec![false; n];
        for c in &row.clamps {
            if c.variable >= n || !c.value.is_finite() || seen[c.variable] {
                return Err(invalid("clamp target, value, or duplicate"));
            }
            seen[c.variable] = true;
            if row.values[c.variable] != c.value {
                return Err(invalid("observed value differs from known clamp"));
            }
        }
        Ok((0..n).filter(|&i| !seen[i]).collect())
    }

    fn row_terms(&self, row: &RegimeRow) -> Result<RowTerms, EquilibriumError> {
        let n = self.system.order();
        let free = self.check_row(row)?;
        if free.is_empty() {
            return Ok(RowTerms {
                free,
                blocks: Vec::new(),
                gradients: Vec::new(),
                whitened_residual_weights: Vec::new(),
                data: 0.0,
                log_determinant: 0.0,
            });
        }
        let bound = self.contraction_bound(&row.clamps)?;
        if bound.is_nan() || bound >= 1.0 {
            return Err(EquilibriumError::Uncertified { bound });
        }
        let k = free.len();
        let mut values = vec![0.0; n];
        let mut gradients: Vec<Vec<f64>> = vec![Vec::new(); n];
        for &i in &free {
            let e = &self.equations[i];
            let inputs: Vec<f64> = e.reads.iter().map(|&r| row.values[r]).collect();
            let (v, g) = e.evaluate(&inputs)?;
            values[i] = v;
            gradients[i] = g;
        }
        let residual: Vec<f64> = free.iter().map(|&i| row.values[i] - values[i]).collect();
        let cov: Vec<f64> = free
            .iter()
            .flat_map(|&i| free.iter().map(move |&j| self.system.noise_covariance[i * n + j]))
            .collect();
        let l = cholesky(&cov, k)?;
        let mut z = residual;
        for i in 0..k {
            for j in 0..i {
                z[i] -= l[i * k + j] * z[j];
            }
            z[i] /= l[i * k + i];
        }
        // w = Sigma^-1 r = L^-T z.
        let mut w = z.clone();
        for i in (0..k).rev() {
            for j in i + 1..k {
                w[i] -= l[j * k + i] * w[j];
            }
            w[i] /= l[i * k + i];
        }
        let noise_logdet = 2.0 * (0..k).map(|i| l[i * k + i].ln()).sum::<f64>();
        let data = -0.5
            * (z.iter().map(|x| x * x).sum::<f64>()
                + noise_logdet
                + k as f64 * (2.0 * std::f64::consts::PI).ln());
        let mut log_determinant = 0.0;
        let blocks = self.blocks(&row.clamps)?;
        for block in &blocks {
            let m = block.len();
            let mut matrix = vec![0.0; m * m];
            for (a, &i) in block.iter().enumerate() {
                matrix[a * m + a] = 1.0;
                for (g, &read) in self.equations[i].reads.iter().enumerate() {
                    if let Some(b) = block.iter().position(|&j| j == read) {
                        matrix[a * m + b] -= gradients[i][g];
                    }
                }
            }
            log_determinant += log_abs_determinant(matrix, m)?;
        }
        Ok(RowTerms {
            free,
            blocks,
            gradients,
            whitened_residual_weights: w,
            data,
            log_determinant,
        })
    }

    /// Joint equilibrium log-likelihood terms of known-regime rows. Requires a declared
    /// global contraction for every free block, so one inverse branch is never normalized as
    /// the density of a multiple-equilibrium system.
    pub fn likelihood_terms(
        &self,
        rows: &[RegimeRow],
        ctx: &ExecutionContext,
    ) -> Result<LikelihoodTerms, EquilibriumError> {
        let n = self.system.order();
        let _lease = ctx
            .acquire_serial_numerical_admission(
                u64::try_from(64 * n * n + 256 * n).map_err(|_| invalid("resource size"))?,
            )
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let mut total = LikelihoodTerms { data: 0.0, log_determinant: 0.0 };
        for row in rows {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            let terms = self.row_terms(row)?;
            total.data += terms.data;
            total.log_determinant += terms.log_determinant;
        }
        if !total.total().is_finite() {
            return Err(invalid("nonfinite likelihood"));
        }
        Ok(total)
    }

    /// Analytic derivative of the log-determinant term `log |det(I - J)|` with respect to
    /// fitted scalars: `d/dtheta = -sum_j (M^-1)_{j,e} dJ_{e,j}/dtheta` with `M = I - J` of the
    /// equation's block, from the implementation's declared mixed derivatives. `None` when a
    /// parameter target's mechanism is not declared analytic or supplies no mixed derivative;
    /// the caller then falls back to a central difference. An intercept does not move `J`.
    pub fn determinant_term_score(
        &self,
        rows: &[RegimeRow],
        targets: &[ExternalFitParameter],
        ctx: &ExecutionContext,
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        self.system.parameter_values(targets)?;
        let mut score = vec![0.0; targets.len()];
        for row in rows {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            let terms = self.row_terms(row)?;
            for (t, target) in targets.iter().enumerate() {
                let ExternalFitParameter::Parameter { equation, parameter } = *target else {
                    continue;
                };
                if !terms.free.contains(&equation) {
                    continue;
                }
                let e = &self.equations[equation];
                let ResolvedKind::External { parameters, implementation, spec, .. } = &e.kind
                else {
                    return Err(invalid("parameter target is not external"));
                };
                if spec.derivative != DerivativeDeclaration::Analytic {
                    return Ok(None);
                }
                let inputs: Vec<f64> = e.reads.iter().map(|&r| row.values[r]).collect();
                let Some(mixed) =
                    implementation.input_gradient_parameter_derivative(&inputs, parameters)?
                else {
                    return Ok(None);
                };
                if mixed.len() != parameters.len()
                    || mixed[parameter].len() != inputs.len()
                    || mixed[parameter].iter().any(|v| !v.is_finite())
                {
                    return Err(invalid("external mixed derivative shape or nonfinite"));
                }
                let block = terms
                    .blocks
                    .iter()
                    .find(|b| b.contains(&equation))
                    .ok_or_else(|| invalid("equation outside every block"))?;
                let m = block.len();
                let mut matrix = vec![0.0; m * m];
                for (a, &i) in block.iter().enumerate() {
                    matrix[a * m + a] = 1.0;
                    for (g, &read) in self.equations[i].reads.iter().enumerate() {
                        if let Some(b) = block.iter().position(|&j| j == read) {
                            matrix[a * m + b] -= terms.gradients[i][g];
                        }
                    }
                }
                let position = block.iter().position(|&j| j == equation).unwrap_or(0);
                let mut unit = vec![0.0; m];
                unit[position] = 1.0;
                if !antecedent_kernels::gauss_jordan_solve(&mut matrix, &mut unit, m, 1) {
                    return Err(EquilibriumError::Singular);
                }
                for (g, &read) in e.reads.iter().enumerate() {
                    if let Some(b) = block.iter().position(|&j| j == read) {
                        score[t] -= unit[b] * mixed[parameter][g];
                    }
                }
            }
        }
        Ok(Some(score))
    }

    /// Analytic derivative of the Gaussian data term with respect to fitted scalars, or
    /// `None` when a target is a parameter whose implementation supplies no parameter
    /// gradient. The log-determinant term is not covered here.
    pub fn data_term_score(
        &self,
        rows: &[RegimeRow],
        targets: &[ExternalFitParameter],
        ctx: &ExecutionContext,
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        self.system.parameter_values(targets)?;
        let mut score = vec![0.0; targets.len()];
        for row in rows {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            let terms = self.row_terms(row)?;
            for (t, target) in targets.iter().enumerate() {
                match *target {
                    ExternalFitParameter::Intercept { equation } => {
                        if let Some(pos) = terms.free.iter().position(|&i| i == equation) {
                            score[t] += terms.whitened_residual_weights[pos];
                        }
                    }
                    ExternalFitParameter::Parameter { equation, parameter } => {
                        let Some(pos) = terms.free.iter().position(|&i| i == equation) else {
                            continue;
                        };
                        let e = &self.equations[equation];
                        let ResolvedKind::External { parameters, implementation, .. } = &e.kind
                        else {
                            return Err(invalid("parameter target is not external"));
                        };
                        let inputs: Vec<f64> = e.reads.iter().map(|&r| row.values[r]).collect();
                        let Some(g) = implementation.parameter_gradient(&inputs, parameters)?
                        else {
                            return Ok(None);
                        };
                        if g.len() != parameters.len() || g.iter().any(|v| !v.is_finite()) {
                            return Err(invalid("external parameter gradient shape or nonfinite"));
                        }
                        score[t] += terms.whitened_residual_weights[pos] * g[parameter];
                    }
                }
            }
        }
        Ok(Some(score))
    }
}
