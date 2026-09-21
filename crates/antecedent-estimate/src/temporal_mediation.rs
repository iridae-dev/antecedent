//! Linear temporal mediation effects.
//!
//! Path-product decomposition on lagged samples: total = direct + mediated
//! under a linear SEM with a single mediator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, Diagnostic, ExecutionContext, IdentificationStatus, Lag, MediationContrast,
    MediationQuery, TemporalNodeKey,
};
use antecedent_data::{LaggedColumn, LaggedSampleWorkspace, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::util::{coefficient_variance, ols_sigma2};

mod shared;
pub use shared::{
    PreparedTemporalMediation, SharedMediationBlockSe, shared_mediation_block_bootstrap,
};

/// Temporal mediation effect estimate with optional decomposition.
#[derive(Clone, Debug)]
pub struct TemporalMediationEstimate {
    /// Requested contrast estimate.
    pub effect: EffectEstimate,
    /// Total effect (when computed).
    pub total: Option<f64>,
    /// Direct effect (when computed).
    pub direct: Option<f64>,
    /// Mediated / indirect effect (when computed).
    pub mediated: Option<f64>,
}

/// One scalar posterior summary within a horizon-specific mediation decomposition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediationPosteriorSummary {
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub standard_deviation: f64,
    /// Equal-tail 2.5% quantile.
    pub q025: f64,
    /// Equal-tail 97.5% quantile.
    pub q975: f64,
}

/// Honest uncertainty retained for one temporal mediation horizon.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TemporalMediationUncertainty {
    /// Sampling standard error for the requested contrast. The decomposition
    /// components do not acquire unsupported iid uncertainty.
    FrequentistPointwise {
        /// Standard error, absent when the estimator cannot justify one.
        standard_error: Option<f64>,
    },
    /// Shared circular-block bootstrap SEs for every contrast at this horizon.
    ///
    /// One circular-block replicate of lag-aligned rows refits all three mechanism
    /// regressions, so Total, Direct and Mediated SEs are mutually consistent.
    /// `requested` repeats the requested contrast's SE.
    FrequentistBlockBootstrap {
        /// Requested contrast SE.
        requested: Option<f64>,
        /// Shared-replicate SEs for Total, Direct and Mediated.
        block: crate::temporal_mediation::TemporalMediationBlockSe,
    },
    /// Separate per-horizon posterior summaries. This is not a joint posterior
    /// over horizons.
    BayesianPointwise {
        /// Requested contrast.
        requested: MediationPosteriorSummary,
        /// Total effect.
        total: MediationPosteriorSummary,
        /// Direct effect.
        direct: MediationPosteriorSummary,
        /// Mediated effect.
        mediated: MediationPosteriorSummary,
        /// Draw count used at this horizon.
        n_draws: usize,
        /// Inference backend identifier.
        backend: Arc<str>,
    },
    /// No justified uncertainty was available.
    Unavailable,
}

/// Closed interval over completion-specific mediation point effects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalMediationIdentifiedSet {
    /// Minimum completion-specific point effect.
    pub lower: f64,
    /// Maximum completion-specific point effect.
    pub upper: f64,
}

/// One independently identified and estimated temporal mediation horizon.
#[derive(Clone, Debug)]
pub struct TemporalMediationSlice {
    /// Requested horizon.
    pub horizon: u32,
    /// Identification status at this horizon.
    pub identification_status: IdentificationStatus,
    /// Identifier/estimand method.
    pub method: Arc<str>,
    /// Horizon-specific unfolded adjustment set `S(h)` shared by the mediator,
    /// outcome and reduced-form regressions: the `T→Y` back-door set `I(h)` plus
    /// the mediator/outcome parents that confound `M→Y` (unfolded-window keys).
    pub adjustment: Arc<[TemporalNodeKey]>,
    /// Requested effect and decomposition.
    pub estimate: TemporalMediationEstimate,
    /// Pointwise uncertainty semantics for this horizon.
    pub uncertainty: TemporalMediationUncertainty,
    /// Structural completion range, when the input is an incomplete graph class.
    pub identified_set: Option<TemporalMediationIdentifiedSet>,
    /// Horizon-local identification and estimation diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

/// Durable multi-horizon temporal mediation result.
#[derive(Clone, Debug)]
pub struct TemporalMediationGrid {
    /// Slices in requested horizon order.
    pub slices: Arc<[TemporalMediationSlice]>,
    /// Always false until a shared dynamic posterior defines cross-horizon dependence.
    pub joint_posterior: bool,
}

/// Linear temporal mediation estimator (two-stage / path-product).
#[derive(Clone, Debug)]
pub struct TemporalMediationEstimator {
    /// Linear algebra backend.
    pub backend: FaerBackend,
    /// When true, [`MediationContrast::NaturalDirect`] / [`MediationContrast::NaturalIndirect`]
    /// are treated as their controlled counterparts (linear alias).
    pub allow_natural_controlled_alias: bool,
    /// When true, publish iid analytic SEs for every contrast (homoskedastic OLS
    /// coefficient SE for Total/Direct, Sobel for Mediated). Default false: lagged
    /// rows of one series are serially dependent, so `se_analytic` is NaN for all
    /// three contrasts unless this is set. The name is historical.
    pub allow_iid_sobel_se: bool,
}

impl Default for TemporalMediationEstimator {
    fn default() -> Self {
        Self {
            backend: FaerBackend,
            allow_natural_controlled_alias: false,
            allow_iid_sobel_se: false,
        }
    }
}

impl TemporalMediationEstimator {
    /// Create with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the linear algebra backend.
    #[must_use]
    pub const fn with_backend(mut self, backend: FaerBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set whether [`MediationContrast::NaturalDirect`] / [`MediationContrast::NaturalIndirect`]
    /// are treated as their controlled counterparts (linear alias).
    ///
    /// Defaults to `false`: natural contrasts are refused unless explicitly enabled, since
    /// they only alias the controlled direct/indirect effects under a linear SEM.
    #[must_use]
    pub const fn with_allow_natural_controlled_alias(mut self, allow: bool) -> Self {
        self.allow_natural_controlled_alias = allow;
        self
    }

    /// Publish iid analytic SEs for every contrast (anti-conservative under serial
    /// correlation). Default leaves `se_analytic` NaN for Total, Direct and Mediated.
    #[must_use]
    pub const fn with_allow_iid_sobel_se(mut self, allow: bool) -> Self {
        self.allow_iid_sobel_se = allow;
        self
    }

    /// Estimate mediation contrasts from lag-aligned series.
    ///
    /// Treatment at lag `h` (the query's first requested horizon; default 1),
    /// mediator and outcome contemporaneous unless the unfolded path says otherwise.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, multi-mediator sets, or OLS failures.
    pub fn estimate(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        ctx: &ExecutionContext,
    ) -> Result<TemporalMediationEstimate, EstimationError> {
        self.estimate_with_extras(data, estimand, query, &[], ctx)
    }

    /// Refit the mediation contrast with additional contemporaneous covariates
    /// in both mechanism regressions (used by mediation-native sensitivity checks).
    pub fn estimate_with_extras(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        extra: &[antecedent_core::VariableId],
        ctx: &ExecutionContext,
    ) -> Result<TemporalMediationEstimate, EstimationError> {
        self.estimate_with_adjustment(data, estimand, query, &[], extra, ctx)
    }

    /// Fit with graph-derived lagged baseline covariates and optional RCC columns.
    ///
    /// Point estimates only, plus iid analytic SEs when [`Self::allow_iid_sobel_se`]
    /// is set (NaN otherwise, for every contrast). Dependence-honest uncertainty
    /// comes from [`Self::estimate_with_block_bootstrap`].
    #[allow(clippy::too_many_arguments)] // Keep existing estimator entry points source-compatible.
    pub fn estimate_with_adjustment(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        adjustment: &[LaggedColumn],
        extra: &[antecedent_core::VariableId],
        ctx: &ExecutionContext,
    ) -> Result<TemporalMediationEstimate, EstimationError> {
        let (mediator, delta) = self.validate(estimand, query)?;
        let design = Self::prepare_design(data, mediator, query, adjustment, extra, ctx)?;
        let fit = self.fit_design(&design, None, delta)?;
        Ok(self.estimate_from_fit(query, &fit))
    }

    fn estimate_from_fit(
        &self,
        query: &MediationQuery,
        fit: &ContrastFit,
    ) -> TemporalMediationEstimate {
        let (total, direct, mediated) = (fit.total, fit.direct, fit.mediated);

        let point = match query.contrast {
            MediationContrast::Total => total,
            MediationContrast::Direct | MediationContrast::NaturalDirect => direct,
            MediationContrast::Mediated | MediationContrast::NaturalIndirect => mediated,
        };
        // Lagged rows of one series are serially dependent, so every iid analytic
        // SE (OLS coefficient SE for Total/Direct, Sobel for Mediated) is refused
        // unless the caller opts in.
        let se_analytic =
            if self.allow_iid_sobel_se { fit.iid_se(query.contrast) } else { f64::NAN };

        let mut assumptions = AssumptionSet::default();
        if matches!(
            query.contrast,
            MediationContrast::NaturalDirect | MediationContrast::NaturalIndirect
        ) {
            assumptions.push(antecedent_core::AssumptionRecord {
                assumption: antecedent_core::Assumption::Custom {
                    id: Arc::from("natural_controlled_alias"),
                    description: Arc::from(
                        "natural direct/indirect effects are aliased to controlled \
                         direct/mediated effects under linear temporal mediation",
                    ),
                },
                source: antecedent_core::AssumptionSource::AlgorithmDefault {
                    algorithm: Arc::from("temporal_mediation"),
                },
                scope: antecedent_core::AssumptionScope::Estimation,
                status: antecedent_core::AssumptionStatus::Declared,
            });
        }

        TemporalMediationEstimate {
            effect: EffectEstimate::new(
                point,
                se_analytic,
                assumptions,
                crate::overlap::OverlapPolicy::ExplicitOverride,
            ),
            total: Some(total),
            direct: Some(direct),
            mediated: Some(mediated),
        }
    }

    /// Point estimates plus one shared circular-block bootstrap for Total, Direct
    /// and Mediated.
    ///
    /// Each replicate resamples circular blocks of consecutive lag-aligned rows
    /// ([`crate::temporal_block::row_block_bootstrap_vec`]) and refits all three
    /// mechanism regressions on it, so the three contrast SEs come from the same
    /// replicates. The block length is
    /// [`crate::temporal_block::dependence_block_length`] over the Total, Direct
    /// and Mediated influence scores and every mechanism's normal-equation scores
    /// (structural span = deepest design lag + 1), the rule the single-window
    /// Pulse / Sustained and class-envelope paths use, and every SE carries the
    /// [`crate::temporal_block::circular_fixed_b_scale`] of that length and the
    /// [`crate::temporal_block::kernel_bias_scale`] of the contrast influences. The
    /// short-series effective-row count reads the persistence probes instead
    /// (see `MediationDesign::persistence_probes`).
    /// `effect.se_bootstrap` is the requested contrast's SE; iid analytic SEs keep
    /// the [`Self::allow_iid_sobel_se`] gate.
    ///
    /// # Errors
    ///
    /// Validation or point-fit failures. Failed replicates are counted, not raised.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_with_block_bootstrap(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        adjustment: &[LaggedColumn],
        replicates: u32,
        stream_base: u64,
        ctx: &ExecutionContext,
    ) -> Result<(TemporalMediationEstimate, TemporalMediationBlockSe), EstimationError> {
        let (mediator, delta) = self.validate(estimand, query)?;
        let design = Self::prepare_design(data, mediator, query, adjustment, &[], ctx)?;
        let point = self.fit_design(&design, None, delta)?;
        let mut estimate = self.estimate_from_fit(query, &point);
        let structural_span = mediation_design_max_lag(query, adjustment) as usize + 1;
        let scores = design.contrast_scores(self.backend, &point);
        let influence_refs: Vec<&[f64]> =
            scores.iter().flat_map(|s| s.iter().map(Vec::as_slice)).collect();
        // Every mechanism's normal-equation scores (intercept = residual series)
        // join the contrast scores, as on the single-window effect path. The scan
        // reads every score; without replicates no interval is published and the
        // rule length is reported instead.
        let block_length = if replicates > 0 {
            let normal_scores = shared::mechanism_normal_scores(&point);
            let score_refs: Vec<&[f64]> = influence_refs
                .iter()
                .copied()
                .chain(normal_scores.iter().map(Vec::as_slice))
                .collect();
            crate::temporal_block::dependence_block_length(structural_span, design.n, &score_refs)
        } else {
            antecedent_data::circular_block_length(structural_span, design.n)
        };
        // The short-series statistic reads persistence probes (residual × centred
        // regressor), not the partialled influence functions: the failure it
        // predicts is a persistent treatment or mediator, which partialling on
        // lagged design columns removes from the influence while the interval
        // still under-covers (docs/short-series-thresholds.md measures both).
        let probes = design.persistence_probes(&point);
        let contrast_scores: Vec<&[f64]> = probes.iter().map(Vec::as_slice).collect();
        let mut block = TemporalMediationBlockSe {
            total: None,
            direct: None,
            mediated: None,
            replicates_ok: 0,
            replicates_attempted: 0,
            block_length,
            rows: design.n,
            effective_rows: crate::temporal_block::score_effective_rows(
                &contrast_scores,
                block_length,
            ),
            kernel_bias: crate::temporal_block::kernel_bias_scale(&influence_refs, block_length),
        };
        if replicates > 0 {
            let boot = crate::temporal_block::row_block_bootstrap_vec(
                design.n,
                block.block_length,
                replicates,
                stream_base,
                ctx,
                |rows| {
                    self.fit_design(&design, Some(rows), delta)
                        .ok()
                        .map(|fit| vec![fit.total, fit.direct, fit.mediated])
                },
            )
            .with_kernel_bias(&influence_refs);
            let [total, direct, mediated] = [0, 1, 2].map(|k| boot.se_result(k));
            block.total = total.se;
            block.direct = direct.se;
            block.mediated = mediated.se;
            block.replicates_ok = total.replicates_ok;
            block.replicates_attempted = boot.attempted;
            let requested = match query.contrast {
                MediationContrast::Total => total,
                MediationContrast::Direct | MediationContrast::NaturalDirect => direct,
                MediationContrast::Mediated | MediationContrast::NaturalIndirect => mediated,
            };
            estimate.effect = estimate.effect.with_bootstrap(Some(requested));
        }
        Ok((estimate, block))
    }

    fn validate(
        &self,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
    ) -> Result<(antecedent_core::VariableId, f64), EstimationError> {
        query.validate()?;
        if matches!(
            query.contrast,
            MediationContrast::NaturalDirect | MediationContrast::NaturalIndirect
        ) && !self.allow_natural_controlled_alias
        {
            return Err(EstimationError::unsupported(
                "NaturalDirect/NaturalIndirect require allow_natural_controlled_alias; \
                 natural effects alias controlled effects in linear temporal mediation",
            ));
        }
        if !(estimand.method_kind().ok().is_some_and(|m| {
            m.is_temporal_mediation() || m == antecedent_expr::EstimandMethod::FrontDoor
        })) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "TemporalMediationEstimator expects temporal_mediation.* or frontdoor",
            });
        }
        if estimand.mediators.len() != 1 {
            return Err(EstimationError::unsupported(
                "TemporalMediationEstimator supports exactly one mediator",
            ));
        }
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        let delta = active - control;
        if delta == 0.0 {
            return Err(EstimationError::unsupported(
                "active and control treatment levels must differ",
            ));
        }
        Ok((estimand.mediators[0], delta))
    }

    /// Prepare the lag-aligned `[T, M, Y, adjustment…, extra…]` columns.
    fn prepare_design(
        data: &TimeSeriesData,
        mediator: antecedent_core::VariableId,
        query: &MediationQuery,
        adjustment: &[LaggedColumn],
        extra: &[antecedent_core::VariableId],
        ctx: &ExecutionContext,
    ) -> Result<MediationDesign, EstimationError> {
        let mut cols = vec![
            LaggedColumn { variable: query.treatment, lag: Lag::from_raw(treatment_lag(query)) },
            LaggedColumn { variable: mediator, lag: Lag::CONTEMPORANEOUS },
            LaggedColumn { variable: query.outcome, lag: Lag::CONTEMPORANEOUS },
        ];
        cols.extend_from_slice(adjustment);
        cols.extend(
            extra.iter().map(|&variable| LaggedColumn { variable, lag: Lag::CONTEMPORANEOUS }),
        );
        let ncols = cols.len();
        let max_lag = cols.iter().map(|c| c.lag.raw()).max().unwrap_or(1);
        let plan =
            data.plan_lagged_sample(max_lag, Arc::from(cols)).map_err(EstimationError::from)?;
        let mut ws = LaggedSampleWorkspace::default();
        let prep =
            plan.prepare(data, &mut ws, &ctx.kernel_policy).map_err(EstimationError::from)?;
        let n = prep.n;
        if n < 4 {
            return Err(EstimationError::data_msg("insufficient effective samples for mediation"));
        }
        let mut columns = Vec::with_capacity(n * ncols);
        for c in 0..ncols {
            columns.extend_from_slice(prep.column(c));
        }
        Ok(MediationDesign { columns, n, n_extra: ncols - 3 })
    }

    /// Fit the three mechanism regressions on the design, or on the replicate
    /// rows `rows` (`rows[r]` = source row of replicate row `r`).
    fn fit_design(
        &self,
        design: &MediationDesign,
        rows: Option<&[usize]>,
        delta: f64,
    ) -> Result<ContrastFit, EstimationError> {
        let n = design.n;
        let gathered;
        let columns = match rows {
            None => &design.columns,
            Some(rows) => {
                gathered = design
                    .columns
                    .chunks_exact(n)
                    .flat_map(|col| rows.iter().map(move |&r| col[r]))
                    .collect::<Vec<f64>>();
                &gathered
            }
        };
        let column = |c: usize| &columns[c * n..(c + 1) * n];
        let (t, m, y) = (column(0), column(1), column(2));
        let n_extra = design.n_extra;
        let extras: Vec<_> = (0..n_extra).map(|i| column(3 + i)).collect();
        // Stage 1: M ~ [1, T] → a = β_T
        let (a, _intercept_m, design_a, sigma2_a, resid_a) =
            ols_two_col(self.backend, t, m, &extras)?;
        // Stage 2: Y ~ [1, T, M] → c' = β_T (direct), b = β_M
        let (c_prime, b, design_b, sigma2_b, resid_b) =
            ols_three_col(self.backend, t, m, y, &extras)?;
        // Reduced form: Y ~ [1, T] → c = total
        let (c, _intercept_y, design_c, sigma2_c, resid_c) =
            ols_two_col(self.backend, t, y, &extras)?;
        Ok(ContrastFit {
            total: c * delta,
            direct: c_prime * delta,
            mediated: a * b * delta,
            a,
            b,
            delta,
            n,
            n_extra,
            designs: [design_a, design_b, design_c],
            sigma2: [sigma2_a, sigma2_b, sigma2_c],
            residuals: [resid_a, resid_b, resid_c],
        })
    }
}

/// Owned lag-aligned mediation columns `[T, M, Y, adjustment…, extra…]`, column-major.
struct MediationDesign {
    columns: Vec<f64>,
    n: usize,
    n_extra: usize,
}

impl MediationDesign {
    fn column(&self, c: usize) -> &[f64] {
        &self.columns[c * self.n..(c + 1) * self.n]
    }

    /// Persistence probes of the Total, Direct and Mediated contrasts: each
    /// mechanism residual times the centred treatment (and, for the mediated
    /// path, the centred mediator), from the residuals of the point fit. They
    /// are not influence functions (see [`Self::contrast_scores`]);
    /// they feed only the short-series effective-row statistic, whose threshold
    /// was measured on them.
    fn persistence_probes(&self, fit: &ContrastFit) -> [Vec<f64>; 3] {
        let (t, m) = (self.column(0), self.column(1));
        let [r_a, r_b, r_c] = &fit.residuals;
        let centered = |x: &[f64]| {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|v| v - mean).collect::<Vec<f64>>()
        };
        let (tc, mc) = (centered(t), centered(m));
        let total = r_c.iter().zip(&tc).map(|(e, x)| e * x).collect();
        let direct = r_b.iter().zip(&tc).map(|(e, x)| e * x).collect();
        let mediated =
            (0..self.n).map(|r| fit.b * r_a[r] * tc[r] + fit.a * r_b[r] * mc[r]).collect();
        [total, direct, mediated]
    }

    /// OLS residuals of `outcome` on `[1, regressors…, extras…]` over the design's rows.
    fn partial_residuals(
        &self,
        backend: FaerBackend,
        regressors: &[&[f64]],
        outcome: &[f64],
    ) -> Option<Vec<f64>> {
        let mut design = vec![1.0; self.n];
        for column in regressors {
            design.extend_from_slice(column);
        }
        for i in 0..self.n_extra {
            design.extend_from_slice(self.column(3 + i));
        }
        let ncols = design.len() / self.n;
        ols_fit_with_residuals(backend, &design, ncols, outcome)
            .ok()
            .map(|(_, residuals)| residuals)
    }

    /// Influence scores of the Total, Direct and Mediated contrasts on the
    /// lag-aligned rows, or `None` when a mechanism regression fails.
    ///
    /// A regression coefficient's score is its residual times the regressor
    /// residualized on the other regressors (Frisch–Waugh), over that
    /// residualized regressor's second moment: `T` on `[1, extras]` for the
    /// total effect and for `a`, `T` on `[1, M, extras]` for the direct effect,
    /// `M` on `[1, T, extras]` for `b`. The product `a·b` takes the delta method
    /// `b·ψ_a + a·ψ_b`, which needs the second-moment scaling to weigh its two
    /// terms correctly.
    fn contrast_scores(&self, backend: FaerBackend, fit: &ContrastFit) -> Option<[Vec<f64>; 3]> {
        let (t, m) = (self.column(0), self.column(1));
        let [r_a, r_b, r_c] = &fit.residuals;
        // Coefficient score: residual × partialled regressor / its second moment.
        let score = |residual: &[f64], partialled: &[f64]| -> Option<Vec<f64>> {
            let moment = partialled.iter().map(|x| x * x).sum::<f64>() / self.n as f64;
            (moment > 0.0 && moment.is_finite())
                .then(|| residual.iter().zip(partialled).map(|(e, x)| e * x / moment).collect())
        };
        let t_given_extras = self.partial_residuals(backend, &[], t)?;
        let t_given_m = self.partial_residuals(backend, &[m], t)?;
        let total = score(r_c, &t_given_extras)?;
        let direct = score(r_b, &t_given_m)?;
        let psi_a = score(r_a, &t_given_extras)?;
        // `M` partialled on `[1, T, extras]` is the `M ~ T` residual itself.
        let psi_b = score(r_b, r_a)?;
        let mediated: Vec<f64> =
            psi_a.iter().zip(&psi_b).map(|(sa, sb)| fit.b * sa + fit.a * sb).collect();
        Some([total, direct, mediated])
    }
}

/// Shared circular-block SEs for one temporal mediation horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalMediationBlockSe {
    /// Total-effect SE.
    pub total: Option<f64>,
    /// Direct-effect SE.
    pub direct: Option<f64>,
    /// Mediated-effect SE.
    pub mediated: Option<f64>,
    /// Replicates where all three contrasts were finite.
    pub replicates_ok: u32,
    /// Replicates evaluated.
    pub replicates_attempted: u32,
    /// Circular-block length in lag-aligned rows
    /// ([`crate::temporal_block::dependence_block_length`]; the plain rule
    /// length when [`Self::replicates_attempted`] is 0, since the score scan
    /// only runs for a bootstrap).
    pub block_length: usize,
    /// Lag-aligned rows resampled.
    pub rows: usize,
    /// Effective rows of the three contrasts' estimating scores at
    /// [`Self::block_length`] (the smallest,
    /// [`crate::temporal_block::score_effective_rows`]; `NaN` without scores).
    pub effective_rows: f64,
    /// Bartlett kernel-bias factor of the three contrast influences (and, for
    /// a mixture, their weighted mixtures) at [`Self::block_length`]
    /// ([`crate::temporal_block::kernel_bias_scale`]), applied to every SE
    /// together with the fixed-b factor.
    pub kernel_bias: f64,
}

fn treatment_lag(query: &MediationQuery) -> u32 {
    query.horizons.first().copied().filter(|&h| h >= 1).unwrap_or(1)
}

/// Deepest lag in the mediation design (treatment lag or an adjustment lag).
fn mediation_design_max_lag(query: &MediationQuery, adjustment: &[LaggedColumn]) -> u32 {
    adjustment.iter().map(|c| c.lag.raw()).fold(treatment_lag(query), u32::max)
}

/// Three mechanism fits on one prepared design.
struct ContrastFit {
    total: f64,
    direct: f64,
    mediated: f64,
    a: f64,
    b: f64,
    delta: f64,
    n: usize,
    n_extra: usize,
    /// Column-major designs for `M ~ T`, `Y ~ T + M`, `Y ~ T`.
    designs: [Vec<f64>; 3],
    sigma2: [f64; 3],
    /// OLS residuals of the same three regressions, in the same order.
    residuals: [Vec<f64>; 3],
}

impl ContrastFit {
    /// Homoskedastic iid SE for `contrast` (Sobel for the mediated product).
    fn iid_se(&self, contrast: MediationContrast) -> f64 {
        let (n, k, delta) = (self.n, self.n_extra, self.delta);
        let [design_a, design_b, design_c] = &self.designs;
        let [sigma2_a, sigma2_b, sigma2_c] = self.sigma2;
        let var = match contrast {
            MediationContrast::Total => coefficient_variance(design_c, n, 2 + k, 1, sigma2_c),
            MediationContrast::Direct | MediationContrast::NaturalDirect => {
                coefficient_variance(design_b, n, 3 + k, 1, sigma2_b)
            }
            MediationContrast::Mediated | MediationContrast::NaturalIndirect => {
                let var_a = coefficient_variance(design_a, n, 2 + k, 1, sigma2_a);
                let var_b = coefficient_variance(design_b, n, 3 + k, 2, sigma2_b);
                // Sobel: SE(ab) ≈ sqrt(b² Var(a) + a² Var(b)).
                self.b * self.b * var_a + self.a * self.a * var_b
            }
        };
        (var * delta * delta).max(0.0).sqrt()
    }
}

/// Returns `(slope_x, intercept, design [1,x], σ²)`.
fn ols_two_col(
    backend: FaerBackend,
    x: &[f64],
    y: &[f64],
    extra: &[&[f64]],
) -> Result<(f64, f64, Vec<f64>, f64, Vec<f64>), EstimationError> {
    let n = x.len();
    let mut design = vec![0.0; n * 2];
    for i in 0..n {
        design[i] = 1.0;
        design[n + i] = x[i];
    }
    for column in extra {
        design.extend_from_slice(column);
    }
    let (coef, residuals) = ols_fit_with_residuals(backend, &design, 2 + extra.len(), y)?;
    let sigma2 = ols_sigma2(&design, n, 2 + extra.len(), y, &coef);
    Ok((coef[1], coef[0], design, sigma2, residuals))
}

/// Returns `(c' = β_T, b = β_M, design [1,T,M], σ², residuals)`.
fn ols_three_col(
    backend: FaerBackend,
    t: &[f64],
    m: &[f64],
    y: &[f64],
    extra: &[&[f64]],
) -> Result<(f64, f64, Vec<f64>, f64, Vec<f64>), EstimationError> {
    let n = t.len();
    let mut design = vec![0.0; n * 3];
    for i in 0..n {
        design[i] = 1.0;
        design[n + i] = t[i];
        design[2 * n + i] = m[i];
    }
    for column in extra {
        design.extend_from_slice(column);
    }
    let (coef, residuals) = ols_fit_with_residuals(backend, &design, 3 + extra.len(), y)?;
    let sigma2 = ols_sigma2(&design, n, 3 + extra.len(), y, &coef);
    Ok((coef[1], coef[2], design, sigma2, residuals))
}

#[cfg(test)]
fn ols_fit(
    backend: FaerBackend,
    design_colmajor: &[f64],
    ncols: usize,
    y: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    Ok(ols_fit_with_residuals(backend, design_colmajor, ncols, y)?.0)
}

/// `(coefficients, residuals)` of one least-squares fit.
fn ols_fit_with_residuals(
    backend: FaerBackend,
    design_colmajor: &[f64],
    ncols: usize,
    y: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
    let mut ws = LeastSquaresWorkspace::default();
    let fit = backend
        .least_squares(design_colmajor, y.len(), ncols, y, &mut ws)
        .map_err(crate::util::stats_err)?;
    Ok((fit.coefficients, fit.residuals))
}

/// Temporal effect surface: direct, total, mediated, and (optional) conditional effects.
#[derive(Clone, Debug)]
pub struct TemporalEffectSurface {
    /// Total effect.
    pub total: f64,
    /// Direct effect.
    pub direct: f64,
    /// Mediated effect.
    pub mediated: f64,
    /// Optional conditional effect at a modifier level (same as total when unmodified).
    pub conditional: Option<f64>,
}

impl TemporalMediationEstimator {
    /// Convenience: return the full direct/total/mediated/conditional effect surface.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::estimate`].
    pub fn effect_surface(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        ctx: &ExecutionContext,
    ) -> Result<TemporalEffectSurface, EstimationError> {
        let est = self.estimate(data, estimand, query, ctx)?;
        TemporalEffectSurface::from_components(est.total, est.direct, est.mediated)
    }
}

impl TemporalEffectSurface {
    /// Assemble the surface from the estimator's optional components.
    ///
    /// A component the estimator did not compute is a refusal, never a value:
    /// substituting zero would publish "not estimated" as "no effect", and
    /// substituting the requested contrast would publish a Direct or Mediated
    /// estimate as the total.
    ///
    /// # Errors
    ///
    /// [`EstimationError::Unsupported`] when any of the three components is absent.
    pub fn from_components(
        total: Option<f64>,
        direct: Option<f64>,
        mediated: Option<f64>,
    ) -> Result<Self, EstimationError> {
        let (Some(total), Some(direct), Some(mediated)) = (total, direct, mediated) else {
            return Err(EstimationError::unsupported(
                "temporal effect surface needs the total, direct and mediated effects; \
                 the estimator did not compute all three",
            ));
        };
        Ok(Self { total, direct, mediated, conditional: None })
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, MediationContrast, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use antecedent_expr::{CausalExprArena, IdentifiedEstimand};

    use super::*;

    fn mediated_series(n: usize) -> (TimeSeriesData, MediationQuery, IdentifiedEstimand) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "m", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut t = vec![0.0; n];
        let mut m = vec![0.0; n];
        let mut y = vec![0.0; n];
        for (i, value) in t.iter_mut().enumerate() {
            *value = (0.071 * i as f64).sin() + 0.35 * (0.137 * i as f64).cos();
        }
        for i in 1..n {
            m[i] = 0.8 * t[i - 1] + 0.12 * (0.43 * i as f64).sin();
            y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + 0.09 * (0.29 * i as f64).cos();
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(m),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let mut arena = CausalExprArena::new();
        let functional = arena.temporal_mediation_ate(
            q.treatment,
            q.outcome,
            &q.mediators,
            antecedent_core::Value::f64(1.0),
            antecedent_core::Value::f64(0.0),
        );
        let estimand = IdentifiedEstimand::temporal_mediation(
            "temporal_mediation.mediated",
            Arc::clone(&q.mediators),
            functional,
        );
        (data, q, estimand)
    }

    #[test]
    fn recovers_positive_mediated_effect() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/temporal_mediation_grid/expected.json"
        ))
        .unwrap();
        let (data, q, estimand) = mediated_series(fixture["data"]["n"].as_u64().unwrap() as usize);
        let est = TemporalMediationEstimator::new()
            .estimate(&data, &estimand, &q, &ExecutionContext::for_tests(1))
            .unwrap();
        let tolerance = fixture["acceptance"]["atol"].as_f64().unwrap();
        for (actual, field) in [
            (est.total.unwrap(), "total"),
            (est.direct.unwrap(), "direct"),
            (est.mediated.unwrap(), "mediated"),
        ] {
            let expected = fixture["reference"][field].as_f64().unwrap();
            assert!((actual - expected).abs() <= tolerance, "{field}: {actual} != {expected}");
        }
        assert!(
            est.effect.se_analytic.is_nan(),
            "iid Sobel SE is refused on lagged rows unless allow_iid_sobel_se"
        );
        let with_sobel = TemporalMediationEstimator::new()
            .with_allow_iid_sobel_se(true)
            .estimate(&data, &estimand, &q, &ExecutionContext::for_tests(1))
            .unwrap();
        let expected_se = fixture["reference"]["se_mediated_sobel"].as_f64().unwrap();
        assert!(
            (with_sobel.effect.se_analytic - expected_se).abs() <= tolerance,
            "se_mediated_sobel: {} != {expected_se}",
            with_sobel.effect.se_analytic
        );
        // total = c*delta, direct = c'*delta, mediated = a*b*delta come from three separate
        // OLS fits, but T is identical across the reduced-form and full regressions, so
        // c = c' + a*b holds exactly in-sample by Frisch-Waugh-Lovell. This is a guaranteed
        // identity today, not a live bug -- pin it as a cheap guard against a future change
        // (switching to WLS, regularizing one fit, altering a design matrix) silently
        // breaking it.
        let total = est.total.unwrap();
        let direct = est.direct.unwrap();
        let mediated = est.mediated.unwrap();
        assert!(
            (total - (direct + mediated)).abs() < 1e-9,
            "FWL identity violated: total={total} direct={direct} mediated={mediated} \
             direct+mediated={}",
            direct + mediated
        );
    }

    #[test]
    fn effect_surface_refuses_an_absent_component_instead_of_reporting_zero() {
        let full = TemporalEffectSurface::from_components(Some(1.5), Some(0.5), Some(1.0))
            .expect("all three components present");
        assert_eq!((full.total, full.direct, full.mediated), (1.5, 0.5, 1.0));
        assert_eq!(full.conditional, None);
        for missing in 0..3 {
            let pick = |i: usize, v: f64| (i != missing).then_some(v);
            let err =
                TemporalEffectSurface::from_components(pick(0, 1.5), pick(1, 0.5), pick(2, 1.0))
                    .expect_err("an absent component must be refused, not zero-filled");
            assert!(matches!(err, EstimationError::Unsupported { .. }));
        }
    }

    #[test]
    fn natural_contrast_without_flag_errors() {
        let (data, mut q, estimand) = mediated_series(300);
        q.contrast = MediationContrast::NaturalIndirect;
        let err = TemporalMediationEstimator::new()
            .estimate(&data, &estimand, &q, &ExecutionContext::for_tests(1))
            .unwrap_err();
        assert!(matches!(err, EstimationError::Unsupported { .. }));
    }

    #[test]
    fn point_fit_residuals_feed_the_normal_equation_scores_without_a_refit() {
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| (0.37 * i as f64).sin()).collect();
        let t: Vec<f64> = (0..n).map(|i| 0.8 * x[i] + 0.3 * (1.3 * i as f64).cos()).collect();
        let m: Vec<f64> =
            (0..n).map(|i| 0.6 * t[i] + 0.4 * x[i] + 0.2 * (2.1 * i as f64).sin()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 0.5 * t[i] + 0.7 * m[i] - 0.3 * x[i] + 0.25 * (0.9 * i as f64).cos())
            .collect();
        let design =
            MediationDesign { columns: [t.as_slice(), &m, &y, &x].concat(), n, n_extra: 1 };
        let estimator = TemporalMediationEstimator::new();
        let fit = estimator.fit_design(&design, None, 1.0).unwrap();
        let scores = shared::mechanism_normal_scores(&fit);
        // M ~ [1, T, X] (3 columns), Y ~ [1, T, M, X] (4), Y ~ [1, T, X] (3).
        assert_eq!(scores.len(), 10);
        // Normal equations: every score series of an OLS fit sums to zero, and each is the
        // column times the residual of *that* regression.
        let cols = |c: &[&[f64]]| -> Vec<Vec<f64>> {
            std::iter::once(vec![1.0; n]).chain(c.iter().map(|s| s.to_vec())).collect()
        };
        let regressions = [
            (cols(&[&t, &x]), &fit.residuals[0]),
            (cols(&[&t, &m, &x]), &fit.residuals[1]),
            (cols(&[&t, &x]), &fit.residuals[2]),
        ];
        let mut k = 0;
        for (columns, residual) in &regressions {
            for column in columns {
                let total: f64 = scores[k].iter().sum();
                assert!(total.abs() < 1e-8, "score {k} sums to {total}");
                for r in 0..n {
                    assert!((scores[k][r] - column[r] * residual[r]).abs() < 1e-12);
                }
                k += 1;
            }
        }
    }

    #[test]
    fn contrast_scores_are_the_partialled_coefficient_influence_functions() {
        // An extra regressor correlated with T: centring T alone is not the
        // coefficient's influence function, residualizing it on the extras is.
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| (0.37 * i as f64).sin()).collect();
        let t: Vec<f64> = (0..n).map(|i| 0.8 * x[i] + 0.3 * (1.3 * i as f64).cos()).collect();
        let m: Vec<f64> =
            (0..n).map(|i| 0.6 * t[i] + 0.4 * x[i] + 0.2 * (2.1 * i as f64).sin()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 0.5 * t[i] + 0.7 * m[i] - 0.3 * x[i] + 0.25 * (0.9 * i as f64).cos())
            .collect();
        let design =
            MediationDesign { columns: [t.as_slice(), &m, &y, &x].concat(), n, n_extra: 1 };
        let estimator = TemporalMediationEstimator::new();
        let fit = estimator.fit_design(&design, None, 1.0).unwrap();
        let [total, direct, mediated] = design.contrast_scores(estimator.backend, &fit).unwrap();
        // ψ_i = n·[(X'X)⁻¹X']_{k,i}·e_i: the k-th coefficient of regressing the unit
        // vector e_i on the design, times row i's residual, times n.
        let influence = |regressors: &[&[f64]], outcome: &[f64], k: usize| -> Vec<f64> {
            let mut matrix = vec![1.0; n];
            for column in regressors {
                matrix.extend_from_slice(column);
            }
            let ncols = 1 + regressors.len();
            let coef = ols_fit(FaerBackend, &matrix, ncols, outcome).unwrap();
            (0..n)
                .map(|i| {
                    let fitted = (0..ncols).map(|c| matrix[c * n + i] * coef[c]).sum::<f64>();
                    let mut unit = vec![0.0; n];
                    unit[i] = 1.0;
                    let row = ols_fit(FaerBackend, &matrix, ncols, &unit).unwrap()[k];
                    n as f64 * row * (outcome[i] - fitted)
                })
                .collect()
        };
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(u, v)| (u - v).abs() < 1e-8);
        assert!(close(&total, &influence(&[&t, &x], &y, 1)));
        assert!(close(&direct, &influence(&[&t, &m, &x], &y, 1)));
        let psi_a = influence(&[&t, &x], &m, 1);
        let psi_b = influence(&[&t, &m, &x], &y, 2);
        let delta: Vec<f64> =
            psi_a.iter().zip(&psi_b).map(|(a, b)| fit.b * a + fit.a * b).collect();
        assert!(close(&mediated, &delta));
    }
}
