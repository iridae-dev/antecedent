//! Temporal dose-over-horizon / policy-path response estimation (ADR 0021).
//!
//! Reuses temporal-backdoor identification and linear g-computation on the
//! unfolded design. No new identification algorithm.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalResponse, ContinuousDomain, ExecutionContext, GridSpec,
    IdentificationStatus, Intervention, MechanismOverride, ObservationSpec, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TargetPopulation, TemporalEffectQuery,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TableView, TemporalIndexer, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::LeastSquaresWorkspace;

use crate::adjustment::LinearAdjustmentAte;
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::temporal_adjustment::TemporalLinearAdjustment;
use crate::temporal_block::common_time_window;
use crate::util::range;

use self::assumptions::with_pointwise_homoskedastic_ols_assumption;
use self::bootstrap::{
    bayesian_bootstrap_surface, bayesian_bootstrap_surface_gapped, bootstrap_surface,
    bootstrap_surface_gapped,
};
use self::support::{intervention_support, mean_curve_support};
use self::surface::{
    CellEval, FittedHorizon, FittedSurface, SeProvenance, checked_surface_cells,
    flatten_dose_horizon_grid, gapped_horizon_rows, horizon_identification_of, horizon_rows,
    horizon_scores,
};

mod assumptions;
mod band;
mod bayes;
mod block;
mod bootstrap;
mod plan;
mod support;
mod surface;
#[cfg(test)]
mod tests;

pub use assumptions::TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC;
#[cfg(test)]
pub(crate) use band::SIMULTANEOUS_BAND_WITHHELD;
pub(crate) use band::max_deviation_band_columns;
pub use band::{
    MaxDeviationBand, SIMULTANEOUS_BAND_CRITICAL, SIMULTANEOUS_BAND_LOWER,
    SIMULTANEOUS_BAND_MIN_REPLICATES, SIMULTANEOUS_BAND_UPPER, TEMPORAL_RESPONSE_BAND_WITHHELD,
    clear_simultaneous_band, max_deviation_band, publish_simultaneous_band,
};
#[cfg(test)]
pub(crate) use block::temporal_block_length;
pub use block::{
    ResponseBlockLength, TEMPORAL_RESPONSE_BLOCK_LENGTH, TEMPORAL_RESPONSE_FEW_REPLICATES,
    block_dispersion_inflation, disclose_response_block_bootstrap, inflate_replicates,
};
pub use bootstrap::circular_block_positions_into;
pub(crate) use plan::plan_temporal_intervention;
pub use plan::{TemporalInterventionPlan, plan_from_response_query};
pub use surface::{JOINT_HORIZON_LAYOUT, JOINT_HORIZON_REPLICATES, PreparedTemporalSurface};

/// Temporal response estimator: dose × horizon surfaces and temporal intervention responses.
#[derive(Clone, Debug)]
pub struct TemporalResponseEstimator {
    /// Shared OLS machinery (bootstrap off for the surface path by default).
    pub inner: LinearAdjustmentAte,
    /// Fit each horizon on its intact lag windows when the series has missing time points.
    /// The joint bootstrap then resamples blocks of the original window axis.
    pub allow_gaps: bool,
    /// The declared premise that the outcome innovation is serially independent, with the
    /// Ljung-Box gate that can refute it. Under it a single-horizon, one-step Bayesian curve is
    /// the untempered full-likelihood posterior; without it, or at any other horizon, the
    /// Bayesian curve is the long-run-tempered generalized posterior.
    pub white_innovation: Option<crate::white_innovation_posterior::InnovationWhitenessGate>,
    /// Draw the surface replicates as Dirichlet block-weight draws (the posterior over the
    /// surface) instead of circular-block bootstrap counts; the band then reads as credible.
    /// Needs bootstrap replicates; on a series with missing time points the weights fall on blocks
    /// of the original time axis and only on the times observed in every horizon.
    pub block_weight_posterior: bool,
    /// Publish the joint replicates behind the band as the [`JOINT_HORIZON_REPLICATES`] and
    /// [`JOINT_HORIZON_LAYOUT`] support diagnostics, so a consumer can compose horizons from
    /// realizations coupled across them.
    pub retain_joint_replicates: bool,
}

/// A temporal query's per-horizon identification and the aggregate identification status
/// and required assumptions the Bayesian temporal estimators publish with their response.
#[derive(Clone, Debug)]
pub struct TemporalIdentification<'a> {
    /// One identified estimand with its unfolded-graph indexer per requested horizon.
    pub horizons: &'a [(&'a IdentifiedEstimand, &'a TemporalIndexer)],
    /// Aggregate identification status; every horizon must be point identified.
    pub status: IdentificationStatus,
    /// Assumptions the identification requires, carried onto the response.
    pub assumptions: AssumptionSet,
}

/// The data, identifications and variables one surface fit is run on.
#[derive(Clone, Copy)]
struct SurfaceProblem<'a> {
    data: &'a TimeSeriesData,
    identifications: &'a [(&'a IdentifiedEstimand, &'a TemporalIndexer)],
    outcome: VariableId,
    treatment: VariableId,
    temporal: &'a TemporalResponseSpec,
}

impl Default for TemporalResponseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalResponseEstimator {
    /// Defaults: explicit-override overlap, no bootstrap. Pointwise SEs come from the
    /// linear-functional variance `cbar(a)' Sigma cbar(a)` of the standardized mean,
    /// not from the `β_T` coefficient SE alone.
    #[must_use]
    pub fn new() -> Self {
        let mut inner = LinearAdjustmentAte::new();
        inner.inference.bootstrap_replicates = 0;
        inner.overlap = OverlapPolicy::ExplicitOverride;
        Self {
            inner,
            allow_gaps: false,
            white_innovation: None,
            block_weight_posterior: false,
            retain_joint_replicates: false,
        }
    }

    /// Declare serially independent outcome innovations, refutable by `gate`, so a one-step
    /// single-horizon Bayesian curve reports the untempered full-likelihood posterior instead
    /// of the tempered generalized posterior. The declaration is never established here; a
    /// rejecting residual test refuses the fit and names the tempered posterior.
    #[must_use]
    pub const fn with_white_innovation(
        mut self,
        gate: crate::white_innovation_posterior::InnovationWhitenessGate,
    ) -> Self {
        self.white_innovation = Some(gate);
        self
    }

    /// Admit a series with missing time points: windows that read a gap are dropped.
    #[must_use]
    pub const fn with_allow_gaps(mut self, allow_gaps: bool) -> Self {
        self.allow_gaps = allow_gaps;
        self
    }

    /// Estimate a temporal [`ResponseQuery`] on series data.
    ///
    /// `identifications` must be aligned with `query.temporal.horizons`: one
    /// `(estimand, indexer)` pair per requested horizon, already identified.
    /// Reusing a max-horizon estimand at a shorter target is not valid when
    /// confounding is horizon-dependent.
    ///
    /// # Errors
    ///
    /// Missing temporal attachment, unsupported functional/intervention, length
    /// mismatch, or fit failures.
    pub fn estimate(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        // This estimator is a public Rust entry point, not merely an internal
        // continuation from the planner. Enforce the complete ResponseQuery
        // contract here so callers cannot bypass treatment/outcome, observation,
        // or intervention validation and still receive a numerical response.
        query.validate()?;
        if query.observation != ObservationSpec::Complete {
            return Err(EstimationError::unsupported(
                "TemporalResponseEstimator requires complete observations; apply a licensed observation correction first",
            ));
        }
        if !matches!(
            identification_status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "temporal response estimation requires point identification",
            });
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| {
            EstimationError::unsupported(
                "TemporalResponseEstimator requires ResponseQuery.temporal (ADR 0021)",
            )
        })?;
        temporal.validate()?;
        if identifications.len() != temporal.horizons.len() {
            return Err(EstimationError::unsupported(
                "temporal response identification must be supplied once per requested horizon",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::TargetPopulation);
        }
        let assumptions = with_pointwise_homoskedastic_ols_assumption(assumptions);
        let mut response = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => self.estimate_mean_curve(
                &SurfaceProblem {
                    data,
                    identifications,
                    outcome: *outcome,
                    treatment: treatment.variable,
                    temporal,
                },
                &treatment.grid.values()?,
                identification_status,
                assumptions,
                ctx,
            ),
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                match plan_temporal_intervention(interventions, temporal)? {
                    TemporalInterventionPlan::Single { treatment, level, shift } => self
                        .estimate_intervention_curve(
                            &SurfaceProblem {
                                data,
                                identifications,
                                outcome: *outcome,
                                treatment,
                                temporal,
                            },
                            level,
                            shift,
                            identification_status,
                            assumptions,
                            ctx,
                        ),
                    TemporalInterventionPlan::Sequential { .. }
                    | TemporalInterventionPlan::Mechanisms { .. } => {
                        Err(EstimationError::unsupported(
                            "multi-step and joint Sequence overlays require the unfolded \
                             sequential estimator (Study temporal response path)",
                        ))
                    }
                }
            }
            _ => Err(EstimationError::unsupported(
                "temporal response is licensed only for MeanCurve and InterventionResponse",
            )),
        }?;
        // Preserve the exact query estimand. Reconstructing it in the numerical
        // helpers changed Linspace into Values and erased a licensed single-step
        // Sequence into a plain Set/Soft intervention.
        response.estimand = query.functional.clone();
        Ok(response)
    }

    fn estimate_mean_curve(
        &self,
        problem: &SurfaceProblem<'_>,
        doses: &[f64],
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        let SurfaceProblem { outcome, treatment, temporal, .. } = *problem;
        if doses.is_empty() {
            return Err(EstimationError::unsupported("dose grid must be non-empty"));
        }
        let n_h = temporal.horizons.len();
        checked_surface_cells(doses.len(), n_h)?;
        let surface = self.fit_surface(problem, identification_status, ctx)?;

        // Layout: value[d * n_horizons + h] — dose major, then horizon.
        let cells: Vec<(usize, CellEval)> = doses
            .iter()
            .flat_map(|&dose| (0..n_h).map(move |h| (h, CellEval::Dose(dose))))
            .collect();
        let values = surface.cells(&cells);
        let mut support = mean_curve_support(doses, temporal, &surface.ranges);
        let assumptions = surface.finish(
            &mut support,
            &values,
            assumptions,
            self.inner.inference.bootstrap_replicates,
        );
        let uncertainty = values.published_band();

        Ok(CausalResponse {
            estimand: ResponseFunctional::MeanCurve {
                outcome,
                treatment: ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from(doses.to_vec())),
                ),
            },
            identification_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(flatten_dose_horizon_grid(doses, &temporal.horizons)?),
                dimension: 2,
                mean: Arc::from(values.mean),
            }),
            uncertainty,
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.gcomp"),
            horizon_identification: Some(Arc::from(surface.identification)),
            interaction_structurally_zero: false,
        })
    }

    fn estimate_intervention_curve(
        &self,
        problem: &SurfaceProblem<'_>,
        level: Option<f64>,
        shift: f64,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        let SurfaceProblem { outcome, treatment, temporal, .. } = *problem;
        // Linear-in-dose, no treatment×covariate interaction: the fitted model is
        // mu_hat(d) = beta_t * d + base_mean, with base_mean independent of d. So
        // averaging g-comp at observed A_i + delta over i collapses exactly to a
        // single evaluation at Abar + delta — an O(n) loop is not needed. A bootstrap
        // replicate evaluates at its own resampled Abar* + delta.
        let surface = self.fit_surface(problem, identification_status, ctx)?;
        let eval = level.map_or(CellEval::Shift(shift), CellEval::Dose);
        let cells: Vec<(usize, CellEval)> =
            (0..surface.horizons.len()).map(|h| (h, eval)).collect();
        let eval_levels: Vec<f64> = surface
            .horizons
            .iter()
            .map(|fitted| level.unwrap_or_else(|| fitted.treatment_mean() + shift))
            .collect();
        let values = surface.cells(&cells);

        let grid: Vec<f64> = temporal.horizons.iter().map(|h| f64::from(*h)).collect();
        let mut support =
            intervention_support(&eval_levels, level, shift, temporal, &surface.ranges);
        let assumptions = surface.finish(
            &mut support,
            &values,
            assumptions,
            self.inner.inference.bootstrap_replicates,
        );
        let uncertainty = values.published_band();

        Ok(CausalResponse {
            estimand: ResponseFunctional::InterventionResponse {
                outcome,
                interventions: Arc::from(vec![if let Some(level) = level {
                    Intervention::set(treatment, Value::f64(level))
                } else {
                    Intervention::soft(treatment, MechanismOverride::additive_shift(shift))
                }]),
            },
            identification_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(grid),
                dimension: 1,
                mean: Arc::from(values.mean),
            }),
            uncertainty,
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.intervention_gcomp"),
            horizon_identification: Some(Arc::from(surface.identification)),
            interaction_structurally_zero: false,
        })
    }

    /// Fit each horizon with that horizon's identified estimand, retain that
    /// horizon's lag-aligned treatment range and identification record, then run
    /// the joint surface bootstrap when replicates were requested.
    fn fit_surface(
        &self,
        problem: &SurfaceProblem<'_>,
        identification_status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<FittedSurface, EstimationError> {
        let SurfaceProblem { data, identifications, temporal, .. } = *problem;
        let mut ols_ws = LeastSquaresWorkspace::default();
        let mut horizons = Vec::with_capacity(temporal.horizons.len());
        let mut ranges = Vec::with_capacity(temporal.horizons.len());
        let mut identification = Vec::with_capacity(temporal.horizons.len());

        for (i, &horizon) in temporal.horizons.iter().enumerate() {
            let (estimand, indexer) = identifications[i];
            let fitted = self.fit_horizon(problem, estimand, horizon, indexer, ctx, &mut ols_ws)?;
            ranges.push(range(&fitted.prepared.treatment));
            identification.push(horizon_identification_of(
                horizon,
                estimand,
                indexer,
                identification_status,
            )?);
            horizons.push(fitted);
        }

        let replicates = self.inner.inference.bootstrap_replicates;
        let span = identifications
            .iter()
            .map(|(_, indexer)| indexer.history() as usize + indexer.horizon() as usize)
            .max()
            .unwrap_or(1);
        let series_rows = data.row_count();
        let rows = horizon_rows(&horizons, series_rows);
        let gapped = horizons.iter().any(|fitted| fitted.selection.is_some());
        let window = if gapped {
            let gapped_rows = gapped_horizon_rows(&horizons, &rows);
            let start = gapped_rows.iter().map(|d| d.first_time).max().unwrap_or(0);
            let end =
                gapped_rows.iter().map(|d| d.first_time + d.complete.len()).min().unwrap_or(0);
            end.saturating_sub(start)
        } else {
            common_time_window(&rows).map_or(0, |(_, len)| len)
        };
        // The lengthening reads every horizon's scores (a refit and a Politis–White
        // scan each); without replicates no band is published and the rule suffices. The
        // scores of a gapped fit are not consecutive, so its block length is the
        // structural rule.
        let scores = if replicates > 0 && !gapped { horizon_scores(&horizons) } else { Vec::new() };
        let score_refs: Vec<&[f64]> = scores.iter().map(Vec::as_slice).collect();
        let block = ResponseBlockLength::new(span, window, &score_refs);
        let weighted = self.block_weight_posterior;
        let bootstrap = if weighted && gapped {
            bayesian_bootstrap_surface_gapped(
                &horizons,
                &gapped_horizon_rows(&horizons, &rows),
                block.length,
                replicates,
                ctx,
            )
        } else if weighted {
            bayesian_bootstrap_surface(&horizons, &rows, block.length, replicates, ctx)
        } else if gapped {
            bootstrap_surface_gapped(
                &horizons,
                &gapped_horizon_rows(&horizons, &rows),
                block.length,
                replicates,
                ctx,
            )
        } else {
            bootstrap_surface(&horizons, &rows, block.length, replicates, ctx)
        };
        let se_provenance = SeProvenance {
            fell_back: replicates > 0 && bootstrap.is_none(),
            cancelled: bootstrap.as_ref().is_some_and(|b| b.cancelled),
        };
        Ok(FittedSurface {
            horizons,
            ranges,
            identification,
            bootstrap,
            se_provenance,
            block,
            weighted,
            retain_joint: self.retain_joint_replicates,
        })
    }

    fn fit_horizon(
        &self,
        problem: &SurfaceProblem<'_>,
        estimand: &IdentifiedEstimand,
        horizon_steps: u32,
        indexer: &TemporalIndexer,
        ctx: &ExecutionContext,
        ols_ws: &mut LeastSquaresWorkspace,
    ) -> Result<FittedHorizon, EstimationError> {
        let SurfaceProblem { data, treatment, outcome, temporal, .. } = *problem;
        let pulse_query = TemporalEffectQuery {
            treatment,
            outcome,
            policy: temporal.policy.clone(),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(1.0)),
            horizon_steps,
            max_history_lag: temporal.max_history_lag,
            target_population: TargetPopulation::AllObserved,
            effect_modifiers: std::sync::Arc::from([]),
        };
        pulse_query.validate()?;
        // Multi-step Sustained/Dynamic: temporal linear adjustment currently refuses.
        // For 0.7 licensed Pulse (and single-step Sustained) this passes; multi-step
        // policies fail closed here rather than estimating a one-node proxy.
        let adj = TemporalLinearAdjustment { inner: self.inner.clone() };
        if self.allow_gaps {
            let (prepared, selection) =
                adj.prepare_with_gaps(data, estimand, &pulse_query, indexer, &ctx.kernel_policy)?;
            let mut fitted = FittedHorizon::fit(prepared, ols_ws)?;
            fitted.selection = selection.complete.iter().any(|intact| !intact).then_some(selection);
            return Ok(fitted);
        }
        let prepared =
            adj.prepare(data, estimand, &pulse_query, indexer, None, &ctx.kernel_policy)?;
        FittedHorizon::fit(prepared, ols_ws)
    }

    /// Fit every horizon of a complete-data `MeanCurve` / single Set-Shift
    /// `InterventionResponse` once, for tuple-level replicate refits.
    ///
    /// Used by outer bootstraps that must replace the outcome per replicate (the
    /// observation-adjusted pseudo-outcome is refit on every replicate): see
    /// [`PreparedTemporalSurface::replicate`]. Cells follow the layout of
    /// [`Self::estimate`].
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::estimate`] (Sequence / mechanism overlays refuse).
    pub fn prepare_surface(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        query: &ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<PreparedTemporalSurface, EstimationError> {
        query.validate()?;
        let temporal = query.temporal.as_ref().ok_or_else(|| {
            EstimationError::unsupported(
                "temporal surface replicates require ResponseQuery.temporal",
            )
        })?;
        if identifications.len() != temporal.horizons.len() {
            return Err(EstimationError::unsupported(
                "temporal response identification must be supplied once per requested horizon",
            ));
        }
        let n_h = temporal.horizons.len();
        let (treatment, outcome, cells) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                let doses = treatment.grid.values()?;
                checked_surface_cells(doses.len(), n_h)?;
                let cells = doses
                    .iter()
                    .flat_map(|&dose| (0..n_h).map(move |h| (h, CellEval::Dose(dose))))
                    .collect();
                (treatment.variable, *outcome, cells)
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                match plan_temporal_intervention(interventions, temporal)? {
                    TemporalInterventionPlan::Single { treatment, level, shift } => {
                        let eval = level.map_or(CellEval::Shift(shift), CellEval::Dose);
                        (treatment, *outcome, (0..n_h).map(|h| (h, eval)).collect())
                    }
                    _ => {
                        return Err(EstimationError::unsupported(
                            "Sequence overlays have no single-surface tuple replicate",
                        ));
                    }
                }
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "temporal response is licensed only for MeanCurve and InterventionResponse",
                ));
            }
        };
        let problem = SurfaceProblem { data, identifications, outcome, treatment, temporal };
        let mut ols_ws = LeastSquaresWorkspace::default();
        let horizons = temporal
            .horizons
            .iter()
            .zip(identifications)
            .map(|(&horizon, &(estimand, indexer))| {
                self.fit_horizon(&problem, estimand, horizon, indexer, ctx, &mut ols_ws)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if horizons.iter().any(|fitted| fitted.selection.is_some()) {
            return Err(EstimationError::unsupported(
                "the tuple-level replicate refits read consecutive lag-aligned rows; a series with gaps is fit by the joint surface bootstrap only",
            ));
        }
        Ok(PreparedTemporalSurface { horizons, cells, series_rows: data.row_count() })
    }

    /// Draw the surface replicates as Dirichlet block-weight draws (see
    /// [`Self::block_weight_posterior`]).
    #[must_use]
    pub const fn with_block_weight_posterior(mut self, on: bool) -> Self {
        self.block_weight_posterior = on;
        self
    }

    /// Publish the joint replicates behind the band (see [`Self::retain_joint_replicates`]).
    #[must_use]
    pub const fn with_joint_replicates(mut self, on: bool) -> Self {
        self.retain_joint_replicates = on;
        self
    }

    /// Copy the `Study` bootstrap / replicate count onto the shared OLS machinery.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.inner.inference.bootstrap_replicates = replicates;
        self
    }
}
