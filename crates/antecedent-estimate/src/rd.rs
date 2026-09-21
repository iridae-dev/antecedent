//! Sharp regression discontinuity estimator.
//!
//! **Estimand.** The average effect for units at the cutoff,
//! `τ_c = lim_{r↓c} E[Y | R = r] − lim_{r↑c} E[Y | R = r]`
//! ([`antecedent_core::TargetPopulation::LocalAtCutoff`]). It is not the population average effect and not
//! the average effect over the bandwidth window: with an effect that varies in `R` those
//! are three different numbers, and only `τ_c` is identified by the design. A query that
//! names any other target population is refused.
//!
//! **Sharpness is checked, not assumed.** The design says `T = 1{R ≥ cutoff}`. The
//! treatment column is read and every complete row is compared with that rule; one
//! violation refuses the problem (`rd_assignment_not_sharp`). With imperfect compliance
//! the outcome jump is an intent-to-treat contrast, not the effect of the treatment, and
//! reporting it under the treatment's name would be wrong.
//!
//! The effect is the coefficient on `T` in a local-linear OLS of `Y` on
//! `[1, T, (R − c), T·(R − c)]`, restricted to rows within `bandwidth` of the cutoff.
//!
//! Bandwidth is explicit configuration — no data-driven bandwidth selector
//! (Imbens–Kalyanaraman, cross-validation, etc.) is implemented yet.
//!
//! The window is a rectangular (uniform-kernel) local-linear fit: every row within
//! `bandwidth` of the cutoff gets equal weight, and rows outside it get none. This means:
//! - **Boundary bias is larger than a triangular-kernel local-linear fit would give.** A
//!   triangular (or other kernel that downweights rows farther from the cutoff) reduces the
//!   influence of observations near the edge of the window, which is where local-linear
//!   extrapolation error is largest; the uniform kernel here does not.
//! - **A too-wide caller-supplied `bandwidth` biases the estimate with no diagnostic.**
//!   Since there is no data-driven selector, nothing warns the caller that a wide window is
//!   pulling in curvature away from the cutoff and biasing the local-linear approximation.
//!   Callers are responsible for choosing (and, ideally, sensitivity-checking across) a
//!   defensible bandwidth themselves.
//!
//! Uses the dedicated method tag `"rd.sharp"` rather than `backdoor.adjustment`, since RD
//! identification does not rely on a backdoor adjustment set: [`prepare`](SharpRegressionDiscontinuity::prepare)
//! accepts any [`IdentifiedEstimand`] carrying that tag, including a synthetic one built for
//! tests via `IdentifiedEstimand::backdoor("rd.sharp", ..)`.
//!
//! Analytic SE: [`AnalyticSeKind::Hc1`] (the default) is the residual sandwich
//! `n/(n−p) (XᵀX)⁻¹ Xᵀ diag(e²) X (XᵀX)⁻¹` read at the jump coefficient. It
//! stays valid when the outcome variance differs across the cutoff or along the
//! running variable, which is the usual situation in RD and why
//! heteroskedasticity-robust variances are the standard for local-linear RD
//! (Calonico, Cattaneo & Titiunik 2014 use HC-type or nearest-neighbour
//! residual variances). It is the conventional SE of the uniform-kernel fit: it
//! assumes independent rows and ignores smoothing bias (no robust bias
//! correction). Hc0/Hc2/Hc3 are available through
//! [`SharpRegressionDiscontinuity::with_se_kind`]; the classical
//! `σ²(XᵀX)⁻¹` ([`AnalyticSeKind::Homoskedastic`]) is an explicit opt-in that
//! assumes one outcome variance inside the window on both sides of the cutoff.
//!
//! Positivity is not meaningful for RD — it is not a propensity-based method — so
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, matching
//! [`crate::adjustment::LinearAdjustmentAte`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::se::AnalyticSeKind;
use crate::util::{BootstrapSeResult, stats_err};

/// Local-linear RD design column count: `[1, T, (R-c), T·(R-c)]`.
const RD_NCOLS: usize = 4;
/// Column index of the treatment indicator within the RD design.
const RD_TREATMENT_COL: usize = 1;

/// Prepared sharp-RD problem: local-linear design windowed to `|R − cutoff| ≤ bandwidth`.
#[derive(Clone, Debug)]
pub struct PreparedRdProblem {
    /// Column-major `[1, T, (R-c), T·(R-c)]` design, restricted to the bandwidth window.
    pub matrix: Arc<[f64]>,
    /// Row count within the bandwidth window.
    pub nrows: usize,
    /// Outcome, length `nrows`.
    pub outcome: Arc<[f64]>,
    /// Estimand method tag (always `"rd.sharp"`).
    pub method: Arc<str>,
    /// Cutoff applied.
    pub cutoff: f64,
    /// Bandwidth applied.
    pub bandwidth: f64,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Complete rows on which `T = 1{R ≥ cutoff}` was checked and held (every one of them;
    /// a single violation refuses the problem).
    pub assignment_verified_rows: usize,
}

/// Estimation workspace (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct RdWorkspace {
    /// OLS scratch.
    pub ols: LeastSquaresWorkspace,
}

/// Sharp regression discontinuity estimator of the average effect for units at the
/// cutoff ([`antecedent_core::TargetPopulation::LocalAtCutoff`]) — not a population average effect.
///
/// `running_variable`, `cutoff`, and `bandwidth` are explicit configuration; there is no
/// data-driven bandwidth selector.
#[derive(Clone, Debug)]
pub struct SharpRegressionDiscontinuity {
    /// Dense linear-algebra backend.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// Running (assignment) variable.
    pub running_variable: VariableId,
    /// Discontinuity cutoff.
    pub cutoff: f64,
    /// Symmetric bandwidth around the cutoff (`|R − cutoff| ≤ bandwidth` is retained).
    pub bandwidth: f64,
    /// Analytic SE kind: HC1 residual sandwich (default), HC0/HC2/HC3, or the
    /// classical homoskedastic formula on explicit opt-in. Cluster / multiway /
    /// HAC kinds are refused (the RD problem carries no labels).
    pub se_kind: AnalyticSeKind,
}

impl SharpRegressionDiscontinuity {
    /// Construct with explicit running variable, cutoff, and bandwidth.
    ///
    /// Defaults: 200 bootstrap replicates, explicit overlap override, HC1
    /// analytic SE.
    #[must_use]
    pub fn new(running_variable: VariableId, cutoff: f64, bandwidth: f64) -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            running_variable,
            cutoff,
            bandwidth,
            se_kind: AnalyticSeKind::Hc1,
        }
    }

    /// Set the analytic SE kind. `Hc1` (default) and `Hc0`/`Hc2`/`Hc3` use the
    /// residual sandwich and stay valid under heteroskedasticity; `Homoskedastic`
    /// assumes a constant outcome variance in the window. Label-based kinds are
    /// refused at `fit`.
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }

    /// Set the dense linear-algebra backend.
    #[must_use]
    pub const fn with_backend(mut self, backend: FaerBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the number of bootstrap replicates used for the bootstrap standard error.
    ///
    /// Defaults to 200. Set to `0` to skip bootstrapping and report only the analytic SE.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Set the overlap policy. Must remain [`OverlapPolicy::ExplicitOverride`] — RD is not a
    /// propensity-based method.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the running (assignment) variable.
    ///
    /// Overridden by the estimand's own `rd_design` when present.
    #[must_use]
    pub const fn with_running_variable(mut self, running_variable: VariableId) -> Self {
        self.running_variable = running_variable;
        self
    }

    /// Set the discontinuity cutoff.
    ///
    /// Overridden by the estimand's own `rd_design` when present.
    #[must_use]
    pub const fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Set the symmetric bandwidth around the cutoff (`|R − cutoff| ≤ bandwidth` is
    /// retained). Must be positive.
    ///
    /// Overridden by the estimand's own `rd_design` when present.
    #[must_use]
    pub const fn with_bandwidth(mut self, bandwidth: f64) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Prepare the windowed local-linear design from tabular data, identified estimand, and
    /// query.
    ///
    /// Accepts any estimand tagged `"rd.sharp"` (including a synthetic one built via
    /// `IdentifiedEstimand::backdoor("rd.sharp", ..)` for tests).
    ///
    /// # Errors
    ///
    /// Overlap policy is not `ExplicitOverride`, incompatible estimand, unsupported query,
    /// a target population other than the design's cutoff (`population_not_estimable`), a
    /// treatment column that is not `1{R ≥ cutoff}` on every complete row
    /// (`rd_assignment_not_sharp`), missing/invalid data columns, no rows within the
    /// bandwidth window, or a window with only one treatment arm represented.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedRdProblem, EstimationError> {
        crate::util::require_explicit_override(
            self.overlap,
            "SharpRegressionDiscontinuity requires ExplicitOverride overlap policy",
        )?;
        if estimand.method_kind().ok() != Some(antecedent_expr::EstimandMethod::RdSharp) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "SharpRegressionDiscontinuity expects an \"rd.sharp\" estimand",
            });
        }
        // Prefer design params packaged on the estimand when present.
        let (running_variable, cutoff, bandwidth) = if let Some(d) = estimand.rd_design {
            (d.running_variable, d.cutoff, d.bandwidth)
        } else {
            (self.running_variable, self.cutoff, self.bandwidth)
        };
        if bandwidth <= 0.0 {
            return Err(EstimationError::unsupported("bandwidth must be positive"));
        }
        query.validate()?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::unsupported("sharp RD does not support effect modifiers"));
        }
        // The jump is the effect for units at the cutoff and for no other population, so
        // the query has to ask for exactly that; it is never relabelled here.
        if !query.target_population.is_local_at_cutoff(running_variable, cutoff) {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("population_not_estimable"),
                format!(
                    "sharp RD estimates the average effect for units at the cutoff only \
                     (TargetPopulation::LocalAtCutoff on running variable {running_variable:?} \
                     at {cutoff}); the query targets {:?}, which this design does not identify",
                    query.target_population
                ),
            ));
        }
        // The sharp-RD estimand is the outcome jump at the cutoff for the 0/1 crossing
        // indicator `T = 1{R ≥ c}` — a local ATE, not a per-unit-of-treatment slope. Scaling
        // the jump by arbitrary query levels (e.g. levels 0/2 doubling the reported effect)
        // would be semantically wrong, so require the canonical binary coding and report the
        // raw jump.
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        if (active - 1.0).abs() > 1e-12 || control.abs() > 1e-12 {
            return Err(EstimationError::unsupported(
                "sharp RD requires binary treatment levels coded active=1.0, control=0.0; the                  RD estimand is the raw outcome jump at the cutoff for the 0/1 crossing                  indicator and does not scale with query levels",
            ));
        }

        let ids = [query.outcome, running_variable, query.treatment];
        let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
        let outcome_full =
            data.float64_masked(query.outcome, &row_mask).map_err(EstimationError::from)?;
        let running_full =
            data.float64_masked(running_variable, &row_mask).map_err(EstimationError::from)?;
        let treatment_full =
            data.float64_masked(query.treatment, &row_mask).map_err(EstimationError::from)?;
        let assignment_verified_rows =
            verify_sharp_assignment(&treatment_full, &running_full, cutoff)?;

        let mut y_sel = Vec::new();
        let mut centered_sel = Vec::new();
        let mut treated_sel = Vec::new();
        for i in 0..running_full.len() {
            let centered = running_full[i] - cutoff;
            if centered.abs() <= bandwidth {
                y_sel.push(outcome_full[i]);
                centered_sel.push(centered);
                treated_sel.push(if centered >= 0.0 { 1.0 } else { 0.0 });
            }
        }
        let nrows = y_sel.len();
        if nrows == 0 {
            return Err(EstimationError::data_msg(
                "no rows within the bandwidth window of the cutoff",
            ));
        }
        let has_treated = treated_sel.iter().any(|&t| t > 0.5);
        let has_control = treated_sel.iter().any(|&t| t < 0.5);
        if !has_treated || !has_control {
            return Err(EstimationError::data_msg(
                "bandwidth window must contain rows on both sides of the cutoff",
            ));
        }

        let matrix = build_rd_matrix(&treated_sel, &centered_sel);

        Ok(PreparedRdProblem {
            matrix: Arc::from(matrix),
            nrows,
            outcome: Arc::from(y_sel),
            method: Arc::clone(&estimand.method),
            cutoff,
            bandwidth,
            overlap: self.overlap,
            assignment_verified_rows,
        })
    }

    /// Fit the local-linear OLS and return the raw jump at the cutoff, with optional
    /// bootstrap. The query levels are constrained to 0/1 in `prepare`, so no level scaling
    /// is applied.
    ///
    /// # Errors
    ///
    /// Backend/rank failure.
    pub fn fit(
        &self,
        problem: &PreparedRdProblem,
        workspace: &mut RdWorkspace,
        ctx: &ExecutionContext,
        mut assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let fit = self
            .backend
            .least_squares(
                &problem.matrix,
                problem.nrows,
                RD_NCOLS,
                &problem.outcome,
                &mut workspace.ols,
            )
            .map_err(stats_err)?;
        let ate = fit.coefficients[RD_TREATMENT_COL];
        let se_analytic = match self.se_kind {
            AnalyticSeKind::Homoskedastic => {
                let n = problem.nrows as f64;
                let p = RD_NCOLS as f64;
                let sigma2 = fit.rss / (n - p).max(1.0);
                analytic_se_treatment(&problem.matrix, problem.nrows, sigma2)
            }
            AnalyticSeKind::Hc0
            | AnalyticSeKind::Hc1
            | AnalyticSeKind::Hc2
            | AnalyticSeKind::Hc3 => crate::se::residual_sandwich_coef_se(
                self.se_kind,
                &problem.matrix,
                problem.nrows,
                RD_NCOLS,
                &fit.residuals,
                RD_TREATMENT_COL,
                None,
                None,
                None,
            )?
            .unwrap_or(f64::NAN),
            _ => {
                return Err(EstimationError::unsupported(
                    "sharp RD supports Homoskedastic or HC0-HC3 analytic SEs; cluster, multiway, and HAC kinds need labels the RD problem does not carry",
                ));
            }
        };

        let (id, description) = if matches!(self.se_kind, AnalyticSeKind::Homoskedastic) {
            (
                "rd.sharp.homoskedastic_se",
                "se_analytic is the classical sigma^2 (X'X)^-1 jump-coefficient SE, selected explicitly: it assumes one outcome variance on both sides of the cutoff and along the running variable inside the window; the default se_kind Hc1 and se_bootstrap do not",
            )
        } else {
            (
                "rd.sharp.conventional_robust_se",
                "se_analytic is the heteroskedasticity-robust residual-sandwich SE of the conventional uniform-kernel local-linear jump: it allows the outcome variance to differ across the cutoff and along the running variable, assumes independent rows, and ignores smoothing bias (no robust bias correction), so intervals are valid only when the bandwidth makes that bias negligible",
            )
        };
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::ParametricRestriction(
                antecedent_core::ParametricAssumption {
                    id: Arc::from(id),
                    description: Arc::from(description),
                },
            ),
            source: antecedent_core::AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("rd.sharp"),
            },
            scope: antecedent_core::AssumptionScope::Estimation,
            status: antecedent_core::AssumptionStatus::Declared,
        });

        // `prepare` compared every complete row with the threshold rule, so the declared
        // sharpness assumption is now supported by the data it speaks about.
        if problem.assignment_verified_rows > 0 {
            for record in &mut assumptions.entries {
                let is_sharpness = matches!(
                    &record.assumption,
                    antecedent_core::Assumption::Custom { id, .. }
                        if id.as_ref() == "rd.sharp_assignment"
                );
                if is_sharpness {
                    record.status = antecedent_core::AssumptionStatus::Supported;
                }
            }
        }

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx)?)
        };

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_se_kind(self.se_kind)
            .with_bootstrap(boot))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedRdProblem,
        workspace: &mut RdWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let n = problem.nrows;
        let _ = workspace;
        crate::util::bootstrap_se_with_scratch(
            self.bootstrap_replicates,
            ctx,
            0x5D0C_u64,
            n,
            || (RdWorkspace::default(), vec![0.0; n * RD_NCOLS], vec![0.0; n]),
            |(ws, x_boot, y_boot), idx| {
                crate::util::gather_bootstrap_vector(y_boot, &problem.outcome, idx);
                crate::util::gather_bootstrap_design(x_boot, &problem.matrix, n, RD_NCOLS, idx);
                match self.backend.least_squares(x_boot, n, RD_NCOLS, y_boot, &mut ws.ols) {
                    Ok(fit) => Ok(Some(fit.coefficients[RD_TREATMENT_COL])),
                    Err(_) => Ok(None),
                }
            },
        )
    }
}

/// Check `T = 1{R ≥ cutoff}` on every row; return the number of rows checked.
///
/// Sharpness is a statement about the whole assignment rule, so rows outside the
/// bandwidth window count too: a treated unit far below the cutoff says the rule is not
/// the one the design declares.
fn verify_sharp_assignment(
    treatment: &[f64],
    running: &[f64],
    cutoff: f64,
) -> Result<usize, EstimationError> {
    let n = treatment.len();
    let mut not_binary = 0usize;
    let mut off_rule = 0usize;
    for (&t, &r) in treatment.iter().zip(running) {
        let treated = (t - 1.0).abs() <= 1e-12;
        let control = t.abs() <= 1e-12;
        if !treated && !control {
            not_binary += 1;
        } else if treated != (r >= cutoff) {
            off_rule += 1;
        }
    }
    if not_binary > 0 {
        return Err(EstimationError::refused(
            antecedent_core::reason_code!("rd_assignment_not_sharp"),
            format!(
                "sharp RD needs a 0/1 treatment column to check against T = 1{{R >= {cutoff}}}; \
                 {not_binary} of {n} rows hold another value"
            ),
        ));
    }
    if off_rule > 0 {
        let share = off_rule as f64 / n as f64;
        return Err(EstimationError::refused(
            antecedent_core::reason_code!("rd_assignment_not_sharp"),
            format!(
                "treatment is not the threshold rule T = 1{{R >= {cutoff}}}: {off_rule} of {n} rows \
                 ({:.1}%) are treated below the cutoff or untreated at or above it. The outcome \
                 jump is then an intent-to-treat contrast, not the effect of the treatment; a \
                 fuzzy design needs the jump in treatment as well, which this estimator does \
                 not compute. Check the running variable, the cutoff, and which side is treated",
                100.0 * share
            ),
        ));
    }
    Ok(n)
}

/// Build the column-major `[1, T, (R-c), T·(R-c)]` local-linear design.
fn build_rd_matrix(treated: &[f64], centered: &[f64]) -> Vec<f64> {
    let n = treated.len();
    let mut matrix = vec![0.0; n * RD_NCOLS];
    for r in 0..n {
        matrix[r] = 1.0;
        matrix[n + r] = treated[r];
        matrix[2 * n + r] = centered[r];
        matrix[3 * n + r] = treated[r] * centered[r];
    }
    matrix
}

fn analytic_se_treatment(x_colmajor: &[f64], nrows: usize, sigma2: f64) -> f64 {
    let Some(inv) = crate::util::xtx_inverse(x_colmajor, nrows, RD_NCOLS) else {
        return f64::NAN;
    };
    (sigma2 * inv[RD_TREATMENT_COL * RD_NCOLS + RD_TREATMENT_COL].max(0.0)).sqrt()
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use antecedent_core::StreamDomain;

    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet,
        TargetPopulation, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::ExprId;
    use antecedent_expr::IdentifiedEstimand;

    use super::*;
    use crate::overlap::OverlapPolicy;

    /// `R ~ U(-1, 1)`, `T = 1{R ≥ 0}`, `Y = 2 + 0.5R + 3T − 0.8T·R + noise`. Jump at cutoff = 3.
    fn sharp_rd_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x8D15_u64);
        let mut t = vec![0.0; n];
        let mut r = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let ri = 2.0 * rng.next_f64() - 1.0;
            let ti = if ri >= 0.0 { 1.0 } else { 0.0 };
            let noise = (rng.next_f64() - 0.5) * 0.2;
            t[i] = ti;
            r[i] = ri;
            y[i] = 2.0 + 0.5 * ri + 3.0 * ti - 0.8 * ti * ri + noise;
        }
        (table_tyr(t, y, r), rd_estimand())
    }

    fn rd_estimand() -> IdentifiedEstimand {
        IdentifiedEstimand::backdoor("rd.sharp", Arc::from([]), ExprId::from_raw(0))
    }

    /// Columns `t` (id 0), `y` (id 1), `r` (id 2).
    fn table_tyr(t: Vec<f64>, y: Vec<f64>, r: Vec<f64>) -> TabularData {
        let n = t.len();
        let mut b = CausalSchemaBuilder::new();
        for (name, role) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("r", RoleHint::Context),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let cols = [t, y, r]
            .into_iter()
            .enumerate()
            .map(|(i, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(u32::try_from(i).unwrap()),
                        Arc::from(values),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TabularData::new(storage)
    }

    /// The cutoff-population query of the design `(R = id 2, c)`.
    fn local_query(cutoff: f64) -> AverageEffectQuery {
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_target_population(TargetPopulation::local_at_cutoff(
                VariableId::from_raw(2),
                cutoff,
            ))
    }

    // ---- known truth: three different effects, one of which the design identifies
    //
    // R has density f(r) = 2(r + 1)/9 on [−1, 2] (R = 3√U − 1), T = 1{R ≥ 0},
    // Y = g(R) + T·τ(R) + ε with a curved baseline g(r) = 1 + 0.5r + 0.8r² + r³ and a
    // heterogeneous effect τ(r) = 2 + 6r. Closed forms:
    //   effect at the cutoff        τ(0)                 = 2
    //   window average (h = 0.4)    2 + 6·E[R | |R| ≤ h] = 2 + 6·h²/3 = 2.32
    //   population average effect   2 + 6·E[R]           = 2 + 6·1    = 8
    // (E[R | |R| ≤ h] = ∫ r(r+1) / ∫ (r+1) over [−h, h] = h²/3; E[R] = 1.)
    const TAU_AT_CUTOFF: f64 = 2.0;
    const TAU_WINDOW_AVERAGE: f64 = 2.32;
    const TAU_POPULATION: f64 = 8.0;
    const KNOWN_TRUTH_BANDWIDTH: f64 = 0.4;

    fn heterogeneous_curved_scm(n: usize, seed: u64) -> TabularData {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x8D16_u64);
        let (mut t, mut y, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let ri = 3.0 * rng.next_f64().sqrt() - 1.0;
            let ti = if ri >= 0.0 { 1.0 } else { 0.0 };
            let baseline = 1.0 + 0.5 * ri + 0.8 * ri * ri + ri * ri * ri;
            let noise = (rng.next_f64() - 0.5) * 0.4;
            t[i] = ti;
            r[i] = ri;
            y[i] = baseline + ti * (2.0 + 6.0 * ri) + noise;
        }
        table_tyr(t, y, r)
    }

    /// The estimator targets the effect at the cutoff, not the window average and not the
    /// population average, on a design where the three are different known numbers.
    ///
    /// The heterogeneity `6r` is linear, so the `T·(R − c)` column absorbs it exactly. The
    /// quadratic baseline term biases both one-sided intercepts equally and cancels; the
    /// cubic term leaves a smoothing bias of about `−0.4·h³ = −0.026` at `h = 0.4`, which
    /// is why the tolerance is 0.06 rather than sampling error alone (SE ≈ 0.006 here).
    #[test]
    fn targets_the_cutoff_effect_not_the_window_or_population_average() {
        let data = heterogeneous_curved_scm(60_000, 11);
        let est = SharpRegressionDiscontinuity {
            bootstrap_replicates: 0,
            ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, KNOWN_TRUTH_BANDWIDTH)
        };
        let prep = est.prepare(&data, &rd_estimand(), &local_query(0.0)).unwrap();
        let mut ws = RdWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.se_analytic < 0.02, "se={}", effect.se_analytic);
        assert!((effect.ate - TAU_AT_CUTOFF).abs() < 0.06, "jump={}", effect.ate);
        assert!((effect.ate - TAU_WINDOW_AVERAGE).abs() > 0.25, "jump={}", effect.ate);
        assert!((effect.ate - TAU_POPULATION).abs() > 5.0, "jump={}", effect.ate);
    }

    /// The number is the cutoff effect, so a query that names any other population —
    /// including the default population-wide one — is refused rather than relabelled.
    #[test]
    fn refuses_a_population_the_design_does_not_identify() {
        let data = heterogeneous_curved_scm(2_000, 12);
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 0.4);
        let population_wide =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let other_cutoff = local_query(0.25);
        let other_running = population_wide.clone().with_target_population(
            TargetPopulation::local_at_cutoff(VariableId::from_raw(1), 0.0),
        );
        for query in [population_wide, other_cutoff, other_running] {
            let err = est.prepare(&data, &rd_estimand(), &query).unwrap_err();
            assert!(
                matches!(err, EstimationError::Refused { code: "population_not_estimable", .. }),
                "err={err:?}"
            );
        }
        assert!(est.prepare(&data, &rd_estimand(), &local_query(0.0)).is_ok());
    }

    /// Fuzzy compliance: P(T=1 | R ≥ 0) = 0.75, P(T=1 | R < 0) = 0.25, effect of T = 3.
    /// The outcome jump at the cutoff is the intent-to-treat contrast 0.5·3 = 1.5, not
    /// the effect of T. The treatment column shows the rule is not sharp, so refuse.
    #[test]
    fn refuses_when_the_treatment_column_is_not_the_threshold_rule() {
        let n = 4_000;
        let mut rng =
            ExecutionContext::for_tests(13).rng.stream_for(StreamDomain::Estimate, 0x8D17_u64);
        let (mut t, mut y, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let ri = 2.0 * rng.next_f64() - 1.0;
            let p = if ri >= 0.0 { 0.75 } else { 0.25 };
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            t[i] = ti;
            r[i] = ri;
            y[i] = 1.0 + 0.5 * ri + 3.0 * ti + (rng.next_f64() - 0.5) * 0.2;
        }
        let data = table_tyr(t, y, r);
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0);
        let err = est.prepare(&data, &rd_estimand(), &local_query(0.0)).unwrap_err();
        let EstimationError::Refused { code, message } = err else {
            panic!("expected a refusal, got {err:?}");
        };
        assert_eq!(code, "rd_assignment_not_sharp");
        // About a quarter of rows on each side break the rule.
        assert!(message.contains("of 4000 rows"), "{message}");
    }

    /// A treatment column that is not 0/1 cannot be checked against the rule.
    #[test]
    fn refuses_a_non_binary_treatment_column() {
        let t = vec![0.0, 2.0, 0.0, 2.0];
        let r = vec![-0.5, 0.5, -0.2, 0.3];
        let y = vec![1.0, 4.0, 1.1, 4.2];
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0);
        let err = est.prepare(&table_tyr(t, y, r), &rd_estimand(), &local_query(0.0)).unwrap_err();
        assert!(
            matches!(err, EstimationError::Refused { code: "rd_assignment_not_sharp", .. }),
            "err={err:?}"
        );
    }

    /// Verified sharpness is evidence: the declared assumption becomes `Supported`.
    #[test]
    fn verified_assignment_marks_the_sharpness_assumption_supported() {
        let (data, estimand) = sharp_rd_scm(800, 7);
        let est = SharpRegressionDiscontinuity {
            bootstrap_replicates: 0,
            ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
        };
        let prep = est.prepare(&data, &estimand, &local_query(0.0)).unwrap();
        assert_eq!(prep.assignment_verified_rows, 800);
        let mut declared = AssumptionSet::new();
        for id in ["rd.continuity", "rd.sharp_assignment"] {
            declared.push(antecedent_core::AssumptionRecord {
                assumption: antecedent_core::Assumption::Custom {
                    id: Arc::from(id),
                    description: Arc::from(""),
                },
                source: antecedent_core::AssumptionSource::AlgorithmDefault {
                    algorithm: Arc::from("rd.sharp"),
                },
                scope: antecedent_core::AssumptionScope::Identification,
                status: antecedent_core::AssumptionStatus::Declared,
            });
        }
        let mut ws = RdWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), declared).unwrap();
        let status = |id: &str| {
            effect
                .assumptions
                .entries
                .iter()
                .find(|r| {
                    matches!(&r.assumption, antecedent_core::Assumption::Custom { id: i, .. } if i.as_ref() == id)
                })
                .map(|r| r.status)
        };
        assert_eq!(
            status("rd.sharp_assignment"),
            Some(antecedent_core::AssumptionStatus::Supported)
        );
        assert_eq!(status("rd.continuity"), Some(antecedent_core::AssumptionStatus::Declared));
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(31)
    }

    #[test]
    fn recovers_jump_of_three() {
        let (data, estimand) = sharp_rd_scm(6000, 1);
        let est = SharpRegressionDiscontinuity {
            bootstrap_replicates: 30,
            ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
        };
        let query = local_query(0.0);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = RdWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 3.0).abs() < 0.5, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    /// HC1 is the default analytic SE; the homoskedastic SE is an explicit
    /// opt-in with its own assumption record, and the jump is unchanged.
    #[test]
    fn hc1_is_the_default_se_and_homoskedastic_is_opt_in() {
        let (data, estimand) = sharp_rd_scm(800, 4);
        let query = local_query(0.0);
        let robust = SharpRegressionDiscontinuity {
            bootstrap_replicates: 0,
            ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
        };
        assert_eq!(robust.se_kind, AnalyticSeKind::Hc1);
        let explicit_hc1 = robust.clone().with_se_kind(AnalyticSeKind::Hc1);
        let classical = robust.clone().with_se_kind(AnalyticSeKind::Homoskedastic);
        let prep = robust.prepare(&data, &estimand, &query).unwrap();
        let mut ws = RdWorkspace::default();
        let a = classical.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let b = robust.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let c = explicit_hc1.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert_eq!(a.ate.to_bits(), b.ate.to_bits());
        assert_eq!(b.se_analytic.to_bits(), c.se_analytic.to_bits());
        assert!(b.se_analytic.is_finite() && b.se_analytic > 0.0);
        assert!((a.se_analytic - b.se_analytic).abs() > 1e-9);
        let declares = |e: &EffectEstimate, id: &str| {
            e.assumptions.entries.iter().any(|r| {
                matches!(
                    &r.assumption,
                    antecedent_core::Assumption::ParametricRestriction(p) if p.id.as_ref() == id
                )
            })
        };
        assert!(declares(&a, "rd.sharp.homoskedastic_se"));
        assert!(!declares(&a, "rd.sharp.conventional_robust_se"));
        assert!(declares(&b, "rd.sharp.conventional_robust_se"));
        assert!(!declares(&b, "rd.sharp.homoskedastic_se"));
        let cluster = classical.with_se_kind(AnalyticSeKind::Cluster);
        assert!(cluster.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).is_err());
    }

    #[test]
    fn rejects_non_rd_estimand() {
        let (data, mut estimand) = sharp_rd_scm(200, 2);
        estimand.method = Arc::from("backdoor.adjustment");
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::IncompatibleEstimand { .. }));
    }

    #[test]
    fn rejects_require_diagnostics_overlap() {
        let (data, estimand) = sharp_rd_scm(200, 3);
        let est = SharpRegressionDiscontinuity {
            overlap: OverlapPolicy::require_diagnostics(),
            ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn rejects_non_binary_treatment_levels() {
        // Levels 0/2 must be refused rather than doubling the reported jump: the sharp-RD
        // estimand is the raw outcome jump at the cutoff for the 0/1 crossing indicator.
        let (data, estimand) = sharp_rd_scm(200, 6);
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0);
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            2.0,
        )
        .with_target_population(TargetPopulation::local_at_cutoff(VariableId::from_raw(2), 0.0));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Unsupported { .. }), "err={err:?}");
    }

    #[test]
    fn rejects_empty_bandwidth_window() {
        // Every unit sits below a cutoff of 100, so a sharp rule treats no one.
        let r: Vec<f64> = (0..200).map(|i| f64::from(i) / 100.0 - 1.0).collect();
        let data = table_tyr(vec![0.0; 200], r.clone(), r);
        let estimand = rd_estimand();
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 100.0, 0.01);
        let query = local_query(100.0);
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Data(_)));
    }

    #[test]
    fn rejects_unsupported_target_population() {
        let (data, estimand) = sharp_rd_scm(200, 5);
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Refused { .. }));
    }
}
