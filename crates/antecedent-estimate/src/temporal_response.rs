//! Temporal dose-over-horizon / policy-path response estimation (ADR 0021).
//!
//! Reuses temporal-backdoor identification and linear g-computation on the
//! unfolded design. No new identification algorithm.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalResponse, ContinuousDomain, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, GridSpec, HorizonIdentification, IdentificationStatus,
    Intervention, InterventionSequence, MAX_TEMPORAL_RESPONSE_CELLS, MechanismOverride,
    ObservationSpec, ParametricAssumption, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, SupportDiagnostic, SupportRegion,
    SupportReport, SupportStatus, TargetPopulation, TemporalEffectQuery, TemporalNodeKey,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{ResamplingPlan, TemporalIndexer, TimeSeriesData, fill_resample_index_batch};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, SandwichKind,
    coefficient_covariance, normal_ppf,
};

use crate::adjustment::{LinearAdjustmentAte, PreparedEstimationProblem};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::temporal_adjustment::TemporalLinearAdjustment;
use crate::temporal_sequential::{SequentialMechanismOverlay, SequentialNodeOverlay};
use crate::util::{BOOTSTRAP_MAX_FAILURE_FRAC, range, sample_std};

/// Licensed temporal `InterventionResponse` overlay.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalInterventionPlan {
    /// One treatment schedule from [`TemporalResponseSpec::policy`] (plain Set / Shift / Soft).
    Single {
        /// Intervened variable.
        treatment: VariableId,
        /// Hard set / Soft constant.
        level: Option<f64>,
        /// Additive shift when `level` is `None`.
        shift: f64,
    },
    /// Explicit Sequence: one overlay per intervened unfolded node.
    Sequential {
        /// Licensed Set / Soft constant / Soft shift overlays.
        overlays: Vec<SequentialNodeOverlay>,
    },
    /// Deterministic mean mechanisms, evaluated through the unfolded engine.
    Mechanisms {
        /// Multiplicative or bounded-mean overlays, optionally mixed with Set/Shift.
        overlays: Vec<SequentialMechanismOverlay>,
    },
}

impl TemporalInterventionPlan {
    /// Sequential mean-mechanism overlays; single Set/Shift keeps its direct path.
    #[must_use]
    pub fn mechanism_overlays(&self) -> Option<Vec<SequentialMechanismOverlay>> {
        match self {
            Self::Single { .. } => None,
            Self::Sequential { overlays } => {
                Some(overlays.iter().copied().map(Into::into).collect())
            }
            Self::Mechanisms { overlays } => Some(overlays.clone()),
        }
    }
    /// Treatment nodes the identifier must cover.
    #[must_use]
    pub fn identification_schedule(&self, spec: &TemporalResponseSpec) -> Vec<(VariableId, i32)> {
        match self {
            Self::Single { treatment, .. } => spec
                .policy
                .active_offsets()
                .map(|offsets| offsets.iter().map(|&offset| (*treatment, offset)).collect())
                .unwrap_or_default(),
            Self::Sequential { overlays } => {
                overlays.iter().map(|overlay| (overlay.variable, overlay.offset)).collect()
            }
            Self::Mechanisms { overlays } => overlays
                .iter()
                .map(|overlay| (overlay.node.variable, overlay.node.offset))
                .collect(),
        }
    }
}

/// Resample stream base for the per-horizon coefficient bootstrap.
///
/// Fixed rather than per-horizon so every horizon resamples the same rows: the surface is
/// read across horizons, and common random numbers keep those pointwise SEs comparable.
const HORIZON_BOOTSTRAP_STREAM: u64 = 0x7E50_u64;

/// Index of the treatment column in the compiled design (col0 = intercept, col1 = treatment).
/// Verified against `CompiledDesign::linear_adjustment`.
const TREATMENT_COL: usize = 1;

/// Per-horizon lag-aligned observed treatment `(min, max)`.
type HorizonTreatmentRange = (f64, f64);

/// Resolved Sequence leaf: variable, fixed level, shift, and active offsets.
type SequenceLeaf = (VariableId, Option<f64>, f64, Arc<[i32]>);

/// What [`TemporalResponseEstimator::run_per_horizon`] returns: the caller's
/// per-horizon payloads alongside the lag-aligned treatment range, the
/// identification record retained for each requested horizon, and where the
/// pointwise SEs actually came from.
type PerHorizonRun<T> =
    (Vec<T>, Vec<HorizonTreatmentRange>, Vec<HorizonIdentification>, SeProvenance);

/// Where a surface's pointwise SEs came from, so the band can say so.
///
/// A requested bootstrap that degenerates to the analytic SE, or one truncated by
/// cancellation, is reported rather than passed off as a completed bootstrap.
#[derive(Clone, Copy, Debug, Default)]
struct SeProvenance {
    /// Horizons where a bootstrap was requested but yielded no usable draws.
    fell_back: usize,
    /// Horizons whose bootstrap was cut short by cooperative cancellation.
    cancelled: usize,
}

impl SeProvenance {
    /// Warning describing a degraded bootstrap, or `None` when SEs are as requested.
    fn warning(self, replicates: u32) -> Option<Diagnostic> {
        if replicates == 0 || (self.fell_back == 0 && self.cancelled == 0) {
            return None;
        }
        let Self { fell_back, cancelled } = self;
        Some(Diagnostic::new(
            "estimate.temporal_response.bootstrap_degraded",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "requested {replicates} bootstrap replicates for the response surface: \
                 {fell_back} horizon(s) reported the analytic OLS linear-functional SE \
                 instead (too few resamples survived), {cancelled} horizon(s) were \
                 truncated by cancellation"
            ),
        ))
    }
}

/// Temporal response estimator: dose × horizon surfaces and temporal intervention responses.
#[derive(Clone, Debug)]
pub struct TemporalResponseEstimator {
    /// Shared OLS machinery (bootstrap off for the surface path by default).
    pub inner: LinearAdjustmentAte,
}

impl Default for TemporalResponseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

fn with_pointwise_homoskedastic_ols_assumption(mut assumptions: AssumptionSet) -> AssumptionSet {
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("ols.linear_additive.gcomp"),
            description: Arc::from(
                "Temporal response levels use linear additive g-computation on each unfolded horizon. The numerical surface is model-dependent when the conditional outcome response is nonlinear or contains treatment-covariate interactions.",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("ols.homoskedastic.pointwise"),
            description: Arc::from(
                "Pointwise 95% band from the delta-method SE of the g-computed level, using the \
                 full homoskedastic OLS coefficient covariance. Not a simultaneous band. Serially \
                 correlated or heteroskedastic innovations can make nominal coverage optimistic.",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions
}

impl TemporalResponseEstimator {
    /// Defaults: explicit-override overlap, no bootstrap. Pointwise SEs come from the
    /// linear-functional variance `cbar(a)' Sigma cbar(a)` of the standardized mean,
    /// not from the `β_T` coefficient SE alone.
    #[must_use]
    pub fn new() -> Self {
        let mut inner = LinearAdjustmentAte::new();
        inner.bootstrap_replicates = 0;
        inner.overlap = OverlapPolicy::ExplicitOverride;
        Self { inner }
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
                data,
                identifications,
                *outcome,
                treatment.variable,
                &treatment.grid.values()?,
                temporal,
                identification_status,
                assumptions,
                ctx,
            ),
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                match plan_temporal_intervention(interventions, temporal)? {
                    TemporalInterventionPlan::Single { treatment, level, shift } => self
                        .estimate_intervention_curve(
                            data,
                            identifications,
                            *outcome,
                            treatment,
                            level,
                            shift,
                            temporal,
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

    /// Bayesian Gaussian linear-additive response on the identified unfolded
    /// design at each horizon. Intervals are pointwise posterior quantiles.
    #[allow(clippy::too_many_lines)]
    pub fn estimate_bayesian(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        mut assumptions: AssumptionSet,
        estimator: &crate::BayesianGComputationAte,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        query.validate()?;
        if query.observation != antecedent_core::ObservationSpec::Complete
            || query.target_population != TargetPopulation::AllObserved
            || estimator.likelihood != antecedent_prob::BayesLikelihood::GaussianIdentity
        {
            return Err(EstimationError::unsupported(
                "Bayesian temporal response requires complete observations, AllObserved, and GaussianIdentity",
            ));
        }
        if !matches!(
            identification_status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::unsupported(
                "Bayesian temporal response requires point identification at every horizon",
            ));
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| {
            EstimationError::unsupported("missing temporal response specification")
        })?;
        if identifications.len() != temporal.horizons.len() {
            return Err(EstimationError::unsupported("one identification required per horizon"));
        }
        let (outcome, treatment, doses, intervention) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                (*outcome, treatment.variable, treatment.grid.values()?, None)
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                match plan_temporal_intervention(interventions, temporal)? {
                    TemporalInterventionPlan::Single { treatment, level, shift } => {
                        (*outcome, treatment, Vec::new(), Some((level, shift)))
                    }
                    TemporalInterventionPlan::Sequential { .. }
                    | TemporalInterventionPlan::Mechanisms { .. } => {
                        return Err(EstimationError::unsupported(
                            "Bayesian multi-step Sequence uses sequential mechanism overlays, \
                             not response.temporal.bayesian / bayesian.gcomp",
                        ));
                    }
                }
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "Bayesian temporal response supports only curves and intervention responses",
                ));
            }
        };
        let mut rows = Vec::new();
        let mut ranges = Vec::new();
        let mut horizons = Vec::new();
        let mut levels = Vec::new();
        for (h, (&horizon_steps, &(estimand, indexer))) in
            temporal.horizons.iter().zip(identifications).enumerate()
        {
            let pulse = TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, Value::f64(0.0)),
                active: Intervention::set(treatment, Value::f64(1.0)),
                horizon_steps,
                max_history_lag: temporal.max_history_lag,
                target_population: TargetPopulation::AllObserved,
            };
            let prep = TemporalLinearAdjustment::new().prepare(
                data,
                estimand,
                &pulse,
                indexer,
                None,
                &ctx.kernel_policy,
            )?;
            let n = prep.design.nrows;
            let t = &prep.design.matrix[n..2 * n];
            ranges.push((
                t.iter().copied().fold(f64::INFINITY, f64::min),
                t.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ));
            horizons.push(horizon_identification_of(
                horizon_steps,
                estimand,
                indexer,
                identification_status,
            )?);
            let mut est = estimator.clone();
            est.seed = est.seed.wrapping_add(h as u64);
            let bprep = crate::BayesianGComputationAte::from_prepared_estimation(&prep);
            let posterior = est.fit(
                &bprep,
                identification_status,
                &mut crate::BayesianGCompWorkspace::default(),
                ctx,
            )?;
            if h == 0 {
                assumptions.entries.extend(posterior.assumptions.entries.clone());
            }
            let mut weights = design_column_means(&prep.design);
            let grid = if let Some((level, shift)) = intervention {
                vec![level.unwrap_or(weights[1] + shift)]
            } else {
                doses.clone()
            };
            if intervention.is_some() {
                levels.push(grid[0]);
            }
            let mut row = Vec::new();
            for dose in grid {
                weights[1] = dose;
                row.push(crate::bayesian::linear_response_summary(&posterior, &weights, 0.95)?);
            }
            rows.push(row);
        }
        let mut mean = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for d in 0..if intervention.is_some() { 1 } else { doses.len() } {
            for row in &rows {
                let (m, lo, hi, _) = row[d];
                mean.push(m);
                lower.push(lo);
                upper.push(hi);
            }
        }
        let (grid, dimension, support) = if let Some((level, shift)) = intervention {
            (
                temporal.horizons.iter().map(|&h| f64::from(h)).collect(),
                1,
                intervention_support(&levels, level, shift, temporal, &ranges),
            )
        } else {
            (
                flatten_dose_horizon_grid(&doses, &temporal.horizons)?,
                2,
                mean_curve_support(&doses, temporal, &ranges),
            )
        };
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption { id: Arc::from("bayesian.temporal_response.linear_additive"),
                description: Arc::from("Gaussian linear-additive unfolded outcome model at each horizon; pointwise posterior intervals conditional on observed adjustment distribution; independent residual likelihood; no joint dose-horizon band") }),
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("response.temporal.bayesian") }, scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared,
        });
        Ok(CausalResponse {
            estimand: query.functional.clone(),
            identification_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(grid),
                dimension,
                mean: Arc::from(mean),
            }),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.response.temporal.bayesian"),
            horizon_identification: Some(Arc::from(horizons)),
            interaction_structurally_zero: false,
        })
    }

    fn estimate_mean_curve(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        outcome: VariableId,
        treatment: VariableId,
        doses: &[f64],
        temporal: &TemporalResponseSpec,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        if doses.is_empty() {
            return Err(EstimationError::unsupported("dose grid must be non-empty"));
        }
        let n_h = temporal.horizons.len();
        let cells = checked_surface_cells(doses.len(), n_h)?;
        let mut mean = Vec::with_capacity(cells);
        let mut lower = Vec::with_capacity(mean.capacity());
        let mut upper = Vec::with_capacity(mean.capacity());

        // Layout: value[d * n_horizons + h] — dose major, then horizon.
        let (per_horizon, horizon_ranges, horizon_identification, se_provenance) = self
            .run_per_horizon(
                data,
                identifications,
                treatment,
                outcome,
                temporal,
                identification_status,
                ctx,
                |fitted| doses.iter().map(|&dose| fitted.mean_and_se_at(dose)).collect::<Vec<_>>(),
            )?;

        let z = normal_ppf(0.975);
        for d_idx in 0..doses.len() {
            for row in &per_horizon {
                let (yhat, se) = row[d_idx];
                mean.push(yhat);
                lower.push(yhat - z * se);
                upper.push(yhat + z * se);
            }
        }

        let mut support = mean_curve_support(doses, temporal, &horizon_ranges);
        support.warnings.extend(se_provenance.warning(self.inner.bootstrap_replicates));

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
                mean: Arc::from(mean),
            }),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.gcomp"),
            horizon_identification: Some(Arc::from(horizon_identification)),
            interaction_structurally_zero: false,
        })
    }

    fn estimate_intervention_curve(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        outcome: VariableId,
        treatment: VariableId,
        level: Option<f64>,
        shift: f64,
        temporal: &TemporalResponseSpec,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        // Linear-in-dose, no treatment×covariate interaction: the fitted model is
        // mu_hat(d) = beta_t * d + base_mean, with base_mean independent of d. So
        // averaging g-comp at observed A_i + delta over i collapses exactly to a
        // single evaluation at Abar + delta — an O(n) loop is not needed.
        let (per_horizon, horizon_ranges, horizon_identification, se_provenance) = self
            .run_per_horizon(
                data,
                identifications,
                treatment,
                outcome,
                temporal,
                identification_status,
                ctx,
                |fitted| {
                    let eval_at = level.unwrap_or_else(|| fitted.treatment_mean() + shift);
                    let (yhat, se) = fitted.mean_and_se_at(eval_at);
                    (eval_at, yhat, se)
                },
            )?;

        let z = normal_ppf(0.975);
        let mut eval_levels = Vec::with_capacity(per_horizon.len());
        let mut mean = Vec::with_capacity(per_horizon.len());
        let mut lower = Vec::with_capacity(per_horizon.len());
        let mut upper = Vec::with_capacity(per_horizon.len());
        for (eval_at, yhat, se) in per_horizon {
            eval_levels.push(eval_at);
            mean.push(yhat);
            lower.push(yhat - z * se);
            upper.push(yhat + z * se);
        }

        let grid: Vec<f64> = temporal.horizons.iter().map(|h| f64::from(*h)).collect();
        let mut support =
            intervention_support(&eval_levels, level, shift, temporal, &horizon_ranges);
        support.warnings.extend(se_provenance.warning(self.inner.bootstrap_replicates));

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
                mean: Arc::from(mean),
            }),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.intervention_gcomp"),
            horizon_identification: Some(Arc::from(horizon_identification)),
            interaction_structurally_zero: false,
        })
    }

    /// Shared per-horizon scaffold: fit each horizon with that horizon's
    /// identified estimand, retain that horizon's lag-aligned treatment range,
    /// and let the caller turn each [`FittedHorizon`] into whatever per-horizon
    /// payload its response shape needs.
    fn run_per_horizon<T>(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        treatment: VariableId,
        outcome: VariableId,
        temporal: &TemporalResponseSpec,
        identification_status: IdentificationStatus,
        ctx: &ExecutionContext,
        mut per_horizon: impl FnMut(&FittedHorizon) -> T,
    ) -> Result<PerHorizonRun<T>, EstimationError> {
        let mut ols_ws = LeastSquaresWorkspace::default();
        let mut results = Vec::with_capacity(temporal.horizons.len());
        let mut horizon_ranges = Vec::with_capacity(temporal.horizons.len());
        let mut horizon_identification = Vec::with_capacity(temporal.horizons.len());
        let mut se_provenance = SeProvenance::default();

        for (i, &horizon) in temporal.horizons.iter().enumerate() {
            let (estimand, indexer) = identifications[i];
            let fitted = self.fit_horizon(
                data,
                estimand,
                treatment,
                outcome,
                temporal,
                horizon,
                indexer,
                ctx,
                &mut ols_ws,
            )?;
            horizon_ranges.push(range(&fitted.prepared.treatment));
            horizon_identification.push(horizon_identification_of(
                horizon,
                estimand,
                indexer,
                identification_status,
            )?);
            se_provenance.fell_back +=
                usize::from(fitted.bootstrap_fell_back(self.inner.bootstrap_replicates));
            se_provenance.cancelled += usize::from(fitted.bootstrap_cancelled());
            results.push(per_horizon(&fitted));
        }

        Ok((results, horizon_ranges, horizon_identification, se_provenance))
    }

    fn fit_horizon(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        treatment: VariableId,
        outcome: VariableId,
        temporal: &TemporalResponseSpec,
        horizon_steps: u32,
        indexer: &TemporalIndexer,
        ctx: &ExecutionContext,
        ols_ws: &mut LeastSquaresWorkspace,
    ) -> Result<FittedHorizon, EstimationError> {
        let pulse_query = TemporalEffectQuery {
            treatment,
            outcome,
            policy: temporal.policy.clone(),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(1.0)),
            horizon_steps,
            max_history_lag: temporal.max_history_lag,
            target_population: TargetPopulation::AllObserved,
        };
        pulse_query.validate()?;
        // Multi-step Sustained/Dynamic: temporal linear adjustment currently refuses.
        // For 0.7 licensed Pulse (and single-step Sustained) this passes; multi-step
        // policies fail closed here rather than estimating a one-node proxy.
        let adj = TemporalLinearAdjustment { inner: self.inner.clone() };
        let prepared =
            adj.prepare(data, estimand, &pulse_query, indexer, None, &ctx.kernel_policy)?;
        FittedHorizon::fit(prepared, ols_ws, self.inner.bootstrap_replicates, ctx)
    }

    /// Copy the `Study` bootstrap / replicate count onto the shared OLS machinery.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.inner.bootstrap_replicates = replicates;
        self
    }
}

/// Per-horizon OLS fit: coefficients, coefficient covariance, and design column means.
///
/// `mu_hat(dose) = coefs' cbar(dose)`, where `cbar(dose)` is the vector of design
/// column means with the treatment column's mean replaced by `dose`. Because the
/// fitted model is linear in the treatment column with no treatment×covariate
/// interaction, this is an O(p) evaluation per dose (no re-scan of the design),
/// and `Var(mu_hat(dose)) = cbar(dose)' Sigma cbar(dose)` is exact (not merely the
/// variance of the `beta_T` coefficient).
struct FittedHorizon {
    prepared: PreparedEstimationProblem,
    coefs: Vec<f64>,
    /// p x p coefficient covariance, row-major.
    cov: Vec<f64>,
    /// Design column means (length p); index `TREATMENT_COL` is `Abar`.
    column_means: Vec<f64>,
    /// Bootstrap coefficient draws when `inner.bootstrap_replicates > 0` and enough
    /// replicates survived; `None` means [`Self::mean_and_se_at`] reports the analytic SE.
    bootstrap: Option<HorizonBootstrap>,
}

impl FittedHorizon {
    fn fit(
        prepared: PreparedEstimationProblem,
        ols_ws: &mut LeastSquaresWorkspace,
        bootstrap_replicates: u32,
        ctx: &ExecutionContext,
    ) -> Result<Self, EstimationError> {
        let n = prepared.design.nrows;
        let p = prepared.design.ncols;
        let fit = FaerBackend
            .least_squares(&prepared.design.matrix, n, p, &prepared.design.outcome, ols_ws)
            .map_err(EstimationError::from)?;
        let cov = coefficient_covariance(
            &prepared.design.matrix,
            n,
            p,
            &fit.residuals,
            SandwichKind::Homoskedastic,
        )
        .map_err(EstimationError::from)?;
        let column_means = design_column_means(&prepared.design);
        let bootstrap =
            bootstrap_horizon_coefs(&prepared, n, p, bootstrap_replicates, ctx, ols_ws)?;
        Ok(Self { prepared, coefs: fit.coefficients, cov, column_means, bootstrap })
    }

    /// Whether a bootstrap was requested but produced no usable draws, so
    /// [`Self::mean_and_se_at`] fell back to the analytic SE.
    const fn bootstrap_fell_back(&self, replicates: u32) -> bool {
        replicates > 0 && self.bootstrap.is_none()
    }

    /// Whether cancellation truncated the bootstrap that produced this horizon's SEs.
    fn bootstrap_cancelled(&self) -> bool {
        self.bootstrap.as_ref().is_some_and(|b| b.cancelled)
    }

    fn treatment_mean(&self) -> f64 {
        self.column_means[TREATMENT_COL]
    }

    /// `(mu_hat(dose), se(dose))` via OLS point estimate and bootstrap or analytic SE.
    fn mean_and_se_at(&self, dose: f64) -> (f64, f64) {
        let p = self.coefs.len();
        let mut cbar = self.column_means.clone();
        cbar[TREATMENT_COL] = dose;
        let mut mu = 0.0;
        for (&coef, &c) in self.coefs.iter().zip(cbar.iter()) {
            mu += coef * c;
        }
        let se = if let Some(boots) = &self.bootstrap {
            let mus: Vec<f64> = boots
                .coefs
                .iter()
                .map(|beta| beta.iter().zip(cbar.iter()).map(|(&b, &c)| b * c).sum::<f64>())
                .collect();
            sample_std(&mus)
        } else {
            let mut var = 0.0;
            for (i, &ci) in cbar.iter().enumerate() {
                let row = &self.cov[i * p..i * p + p];
                let row_sum: f64 =
                    row.iter().zip(cbar.iter()).map(|(&cov_ij, &cj)| cov_ij * cj).sum();
                var += ci * row_sum;
            }
            var.max(0.0).sqrt()
        };
        (mu, se)
    }
}

/// Retained bootstrap coefficient draws for one horizon.
///
/// The surface evaluates `cbar(a)'β` at many doses, so the whole coefficient vector is
/// kept per replicate rather than collapsing to a single SE the way [`crate::util::bootstrap_se`]
/// does. Failure accounting matches that helper: too few survivors, or more than
/// [`BOOTSTRAP_MAX_FAILURE_FRAC`] soft failures, means no bootstrap SE is reported.
struct HorizonBootstrap {
    coefs: Vec<Vec<f64>>,
    cancelled: bool,
}

/// IID bootstrap coefficient draws for one horizon's OLS fit.
///
/// Returns `Ok(None)` when no bootstrap was requested, when cancellation stopped the loop
/// before two replicates survived, or when singular resamples pushed the failure fraction
/// past the crate-wide threshold. Callers fall back to the analytic linear-functional SE
/// and must say so — a silently-analytic SE reported as a bootstrap SE is a lie about the
/// uncertainty's provenance.
fn bootstrap_horizon_coefs(
    prepared: &PreparedEstimationProblem,
    n: usize,
    p: usize,
    replicates: u32,
    ctx: &ExecutionContext,
    ols_ws: &mut LeastSquaresWorkspace,
) -> Result<Option<HorizonBootstrap>, EstimationError> {
    if replicates == 0 || n == 0 {
        return Ok(None);
    }
    let n_rep = replicates as usize;
    let mut indexes = vec![0u32; n * n_rep];
    fill_resample_index_batch(
        ResamplingPlan::IidBootstrap,
        n,
        n_rep,
        None,
        ctx,
        HORIZON_BOOTSTRAP_STREAM,
        &mut indexes,
    )
    .map_err(EstimationError::from)?;
    let mut x_boot = vec![0.0; n * p];
    let mut y_boot = vec![0.0; n];
    let mut coefs = Vec::with_capacity(n_rep);
    let mut cancelled = false;
    for r in 0..n_rep {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        let sl = &indexes[r * n..(r + 1) * n];
        for (i, &src) in sl.iter().enumerate() {
            let src = src as usize;
            y_boot[i] = prepared.design.outcome[src];
            for c in 0..p {
                x_boot[c * n + i] = prepared.design.matrix[c * n + src];
            }
        }
        if let Ok(fit) = FaerBackend.least_squares(&x_boot, n, p, &y_boot, ols_ws) {
            coefs.push(fit.coefficients);
        }
        if let Some(progress) = &ctx.progress {
            progress.report((r + 1) as f64 / n_rep as f64, "bootstrap");
        }
    }
    if coefs.len() < 2 {
        return Ok(None);
    }
    // Unattempted replicates after cancellation are not failures (mirrors
    // `finalize_bootstrap_se_ex`); singular resamples among those attempted are.
    if !cancelled {
        let failed = replicates.saturating_sub(u32::try_from(coefs.len()).unwrap_or(u32::MAX));
        if f64::from(failed) / f64::from(replicates) > BOOTSTRAP_MAX_FAILURE_FRAC {
            return Ok(None);
        }
    }
    Ok(Some(HorizonBootstrap { coefs, cancelled }))
}

fn design_column_means(design: &CompiledDesign) -> Vec<f64> {
    let n = design.nrows;
    let p = design.ncols;
    let mut means = vec![0.0; p];
    for (col, mean) in means.iter_mut().enumerate() {
        let start = col * n;
        let sum: f64 = design.matrix[start..start + n].iter().sum();
        *mean = sum / n as f64;
    }
    means
}

fn checked_surface_cells(doses: usize, horizons: usize) -> Result<usize, EstimationError> {
    let cells = doses
        .checked_mul(horizons)
        .filter(|cells| *cells <= MAX_TEMPORAL_RESPONSE_CELLS)
        .ok_or_else(|| {
        EstimationError::data_msg(
            "temporal response dose-by-horizon cell count exceeds the materialization limit",
        )
    })?;
    Ok(cells)
}

fn flatten_dose_horizon_grid(doses: &[f64], horizons: &[u32]) -> Result<Vec<f64>, EstimationError> {
    let cells = checked_surface_cells(doses.len(), horizons.len())?;
    let capacity = cells.checked_mul(2).ok_or_else(|| {
        EstimationError::data_msg("temporal response coordinate grid size overflow")
    })?;
    let mut grid = Vec::with_capacity(capacity);
    for &dose in doses {
        for &h in horizons {
            grid.push(dose);
            grid.push(f64::from(h));
        }
    }
    Ok(grid)
}

fn named_adjustment(
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
) -> Result<Vec<TemporalNodeKey>, EstimationError> {
    estimand
        .adjustment_set
        .iter()
        .map(|&dense| {
            indexer.key_of(dense.raw()).map_err(|e| EstimationError::data_msg(e.to_string()))
        })
        .collect()
}

fn horizon_identification_of(
    horizon: u32,
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
    status: IdentificationStatus,
) -> Result<HorizonIdentification, EstimationError> {
    Ok(HorizonIdentification {
        horizon,
        status,
        method: Arc::clone(&estimand.method),
        adjustment: Arc::from(named_adjustment(estimand, indexer)?),
    })
}

fn cell_against_range(dose: f64, observed_min: f64, observed_max: f64) -> SupportStatus {
    if !observed_min.is_finite() || !observed_max.is_finite() {
        SupportStatus::Extrapolative
    } else if dose < observed_min || dose > observed_max {
        SupportStatus::OutsideEmpiricalSupport
    } else {
        SupportStatus::Supported
    }
}

/// Surface summary over the same geometry as the estimate.
///
/// All cells supported → [`SupportStatus::Supported`]. Mixed supported /
/// unsupported cells → [`SupportStatus::Extrapolative`] (partially
/// extrapolative). No cell supported → [`SupportStatus::OutsideEmpiricalSupport`],
/// unless every cell was unassessable (non-finite range), which stays
/// extrapolative.
fn summarize_surface_support(points: &[SupportStatus]) -> SupportStatus {
    let n = points.len();
    let n_supported = points.iter().filter(|status| **status == SupportStatus::Supported).count();
    if n == 0 {
        return SupportStatus::Extrapolative;
    }
    if n_supported == n {
        return SupportStatus::Supported;
    }
    if n_supported > 0 {
        return SupportStatus::Extrapolative;
    }
    if points.iter().any(|status| *status == SupportStatus::OutsideEmpiricalSupport) {
        SupportStatus::OutsideEmpiricalSupport
    } else {
        SupportStatus::Extrapolative
    }
}

fn mean_curve_support(
    doses: &[f64],
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
) -> SupportReport {
    let mut point_status = Vec::with_capacity(doses.len().saturating_mul(horizon_ranges.len()));
    for &dose in doses {
        for &(lo, hi) in horizon_ranges {
            point_status.push(cell_against_range(dose, lo, hi));
        }
    }
    assemble_temporal_support(doses, temporal, horizon_ranges, point_status)
}

fn intervention_support(
    eval_levels: &[f64],
    level: Option<f64>,
    shift: f64,
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
) -> SupportReport {
    let point_status: Vec<SupportStatus> = eval_levels
        .iter()
        .zip(horizon_ranges.iter())
        .map(|(&dose, &(lo, hi))| {
            if level.is_some() {
                cell_against_range(dose, lo, hi)
            } else {
                // A shift intervention evaluates the factual treatment law at
                // A + delta, not just at E[A] + delta. The latter can sit inside
                // [min(A), max(A)] while a large fraction of shifted rows are
                // outside it. Classify the whole shifted interval instead.
                shifted_range_against_range(shift, lo, hi)
            }
        })
        .collect();
    let shifted_extrapolation = level.is_none()
        && point_status.iter().any(|status| *status == SupportStatus::Extrapolative);
    let mut report = assemble_temporal_support(eval_levels, temporal, horizon_ranges, point_status);
    // InterventionResponse has one result cell per horizon. Reusing the mean
    // surface assembler used to claim an H x H dose-by-horizon layout even
    // though both the estimate and point_status contain only H cells.
    if let Some(layout) = report
        .diagnostics
        .iter_mut()
        .find(|diagnostic| diagnostic.id.as_ref() == "response.temporal.dose_horizon_layout")
    {
        layout.id = Arc::from("response.temporal.intervention_horizon_layout");
        layout.values = Arc::from([temporal.horizons.len() as f64]);
        layout.detail = Arc::from("one intervention-response cell per requested horizon");
    }
    if level.is_none() {
        let mut shifted_ranges = Vec::with_capacity(horizon_ranges.len().saturating_mul(2));
        for &(lo, hi) in horizon_ranges {
            shifted_ranges.push(lo + shift);
            shifted_ranges.push(hi + shift);
        }
        let shifted_min = shifted_ranges.iter().step_by(2).copied().fold(f64::INFINITY, f64::min);
        let shifted_max =
            shifted_ranges.iter().skip(1).step_by(2).copied().fold(f64::NEG_INFINITY, f64::max);
        report.query_region.minima =
            Arc::from([shifted_min, f64::from(temporal.horizons.first().copied().unwrap_or(1))]);
        report.query_region.maxima =
            Arc::from([shifted_max, f64::from(temporal.horizons.last().copied().unwrap_or(1))]);
        report.diagnostics.push(SupportDiagnostic {
            id: Arc::from("response.temporal.shifted_treatment_range"),
            values: Arc::from(shifted_ranges),
            detail: Arc::from(
                "per-horizon support requested by A + delta as [min_0+delta, max_0+delta, …]",
            ),
        });
    }
    if shifted_extrapolation {
        report.warnings.push(Diagnostic::new(
            "response.temporal.shift_distribution_extrapolative",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "the shifted treatment distribution extends beyond at least one horizon's observed treatment range",
        ));
    }
    report
}

fn shifted_range_against_range(shift: f64, observed_min: f64, observed_max: f64) -> SupportStatus {
    if !shift.is_finite() || !observed_min.is_finite() || !observed_max.is_finite() {
        return SupportStatus::Extrapolative;
    }
    if shift == 0.0 {
        return SupportStatus::Supported;
    }
    let shifted_min = observed_min + shift;
    let shifted_max = observed_max + shift;
    if !shifted_min.is_finite() || !shifted_max.is_finite() {
        SupportStatus::Extrapolative
    } else if shifted_max < observed_min || shifted_min > observed_max {
        SupportStatus::OutsideEmpiricalSupport
    } else {
        SupportStatus::Extrapolative
    }
}

fn assemble_temporal_support(
    doses: &[f64],
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
    point_status: Vec<SupportStatus>,
) -> SupportReport {
    let status = summarize_surface_support(&point_status);
    let mixed = point_status.iter().any(|s| *s == SupportStatus::Supported)
        && point_status.iter().any(|s| *s != SupportStatus::Supported);
    let mut range_values = Vec::with_capacity(horizon_ranges.len().saturating_mul(2));
    for &(lo, hi) in horizon_ranges {
        range_values.push(lo);
        range_values.push(hi);
    }
    let mut warnings = Vec::new();
    if mixed {
        warnings.push(Diagnostic::new(
            "response.temporal.partial_horizon_support",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "some requested (dose, horizon) cells sit outside that horizon's lag-aligned \
             treatment range; inspect support.point_status",
        ));
    } else if status == SupportStatus::OutsideEmpiricalSupport {
        warnings.push(Diagnostic::new(
            "response.outside_empirical_support",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "no requested (dose, horizon) cell sits inside that horizon's lag-aligned \
             treatment range",
        ));
    }
    SupportReport {
        status,
        query_region: SupportRegion {
            minima: Arc::from(vec![
                doses.iter().copied().fold(f64::INFINITY, f64::min),
                f64::from(temporal.horizons.first().copied().unwrap_or(1)),
            ]),
            maxima: Arc::from(vec![
                doses.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                f64::from(temporal.horizons.last().copied().unwrap_or(1)),
            ]),
        },
        diagnostics: vec![
            SupportDiagnostic {
                id: Arc::from("response.temporal.dose_horizon_layout"),
                values: Arc::from(vec![doses.len() as f64, temporal.horizons.len() as f64]),
                detail: Arc::from(
                    "row-major dose × horizon surface: value[d * n_horizons + h]; \
                     grid stores [dose_d, horizon_h] pairs",
                ),
            },
            SupportDiagnostic {
                id: Arc::from("response.temporal.horizon_treatment_range"),
                values: Arc::from(range_values),
                detail: Arc::from(
                    "per-horizon lag-aligned treatment range as [min_0, max_0, min_1, max_1, …]",
                ),
            },
        ],
        warnings,
        point_status: Some(Arc::from(point_status)),
    }
}

/// Plan overlays for a temporal [`ResponseQuery`], if it is an `InterventionResponse`.
///
/// # Errors
///
/// Unlicensed Sequence / Soft forms.
pub fn plan_from_response_query(
    query: &ResponseQuery,
) -> Result<Option<TemporalInterventionPlan>, EstimationError> {
    let Some(temporal) = query.temporal.as_ref() else {
        return Ok(None);
    };
    match &query.functional {
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            Ok(Some(plan_temporal_intervention(interventions, temporal)?))
        }
        _ => Ok(None),
    }
}

/// Classify a temporal `InterventionResponse` as a single-node overlay or a
/// sequential schedule. Nested Sequence stays refused. Multi-step never
/// collapses to the last step.
///
/// # Errors
///
/// Empty, nested, stochastic, unlicensed Soft, or ambiguous schedules.
pub fn plan_temporal_intervention(
    interventions: &[Intervention],
    spec: &TemporalResponseSpec,
) -> Result<TemporalInterventionPlan, EstimationError> {
    if interventions.is_empty() {
        return Err(EstimationError::unsupported(
            "intervention response requires at least one intervention",
        ));
    }
    if interventions.len() > 1 {
        return Err(EstimationError::unsupported(
            "temporal InterventionResponse supports one primary intervention \
             (use Sequence for multi-step or joint policies)",
        ));
    }
    if let Some(plan) = plan_mean_mechanisms(&interventions[0], spec)? {
        return Ok(plan);
    }
    match &interventions[0] {
        Intervention::Sequence(seq) => plan_sequence(seq, spec, 0),
        other => {
            let (treatment, level, shift) = resolve_one(other, 0)?;
            Ok(TemporalInterventionPlan::Single { treatment, level, shift })
        }
    }
}

fn plan_mean_mechanisms(
    intervention: &Intervention,
    spec: &TemporalResponseSpec,
) -> Result<Option<TemporalInterventionPlan>, EstimationError> {
    let leaves = match intervention {
        Intervention::Sequence(sequence) => {
            sequence.steps.iter().map(|step| &step.intervention).collect::<Vec<_>>()
        }
        other => vec![other],
    };
    if !leaves.iter().any(|leaf| matches!(leaf,
        Intervention::Soft { mechanism, .. } if matches!(mechanism.family_id.as_ref(), "multiplicative" | "truncated_shift"))) {
        return Ok(None);
    }
    let mut modifiers = Vec::new();
    let mut replacements = Vec::new();
    for leaf in leaves {
        let mut multiplier = 1.0;
        let mut bounds = None;
        let mut replacement = leaf.clone();
        if let Intervention::Soft { variable, mechanism } = leaf {
            match mechanism.family_id.as_ref() {
                "multiplicative" => {
                    if mechanism.parameters.len() != 1 || !mechanism.parameters[0].is_finite() {
                        return Err(EstimationError::unsupported(
                            "multiplicative requires one finite mean multiplier",
                        ));
                    }
                    multiplier = mechanism.parameters[0];
                    replacement =
                        Intervention::soft(*variable, MechanismOverride::additive_shift(0.0));
                }
                "truncated_shift" => {
                    let p = &mechanism.parameters;
                    if p.len() != 3 || p.iter().any(|v| !v.is_finite()) || p[1] > p[2] {
                        return Err(EstimationError::unsupported(
                            "truncated_shift requires finite [shift, lower, upper] with lower <= upper",
                        ));
                    }
                    bounds = Some((p[1], p[2]));
                    replacement =
                        Intervention::soft(*variable, MechanismOverride::additive_shift(p[0]));
                }
                _ => {}
            }
        }
        modifiers.push((multiplier, bounds));
        replacements.push(replacement);
    }
    let surrogate = if let Intervention::Sequence(sequence) = intervention {
        let mut sequence = sequence.clone();
        let mut steps = sequence.steps.to_vec();
        for (step, replacement) in steps.iter_mut().zip(replacements) {
            step.intervention = replacement;
        }
        sequence.steps = Arc::from(steps);
        Intervention::Sequence(sequence)
    } else {
        replacements.remove(0)
    };
    let base = plan_temporal_intervention(&[surrogate], spec)?;
    let nodes = match base {
        TemporalInterventionPlan::Single { treatment, level, shift } => spec
            .policy
            .active_offsets()
            .map_err(|e| EstimationError::data_msg(e.to_string()))?
            .iter()
            .map(|&offset| SequentialNodeOverlay { variable: treatment, offset, level, shift })
            .collect::<Vec<_>>(),
        TemporalInterventionPlan::Sequential { overlays } => overlays,
        TemporalInterventionPlan::Mechanisms { .. } => {
            unreachable!("surrogate contains only Set/Shift")
        }
    };
    let overlays = nodes
        .into_iter()
        .enumerate()
        .map(|(index, node)| {
            let (multiplier, bounds) = modifiers[if modifiers.len() == 1 { 0 } else { index }];
            SequentialMechanismOverlay { node, multiplier, bounds }
        })
        .collect();
    Ok(Some(TemporalInterventionPlan::Mechanisms { overlays }))
}

/// `depth` counts levels of `Intervention::Sequence` nesting already entered.
/// `resolve_sequence` is only reachable from a Sequence itself, so a `depth > 0`
/// arrival there means a Sequence nested inside a Sequence — refused explicitly
/// rather than silently recursing into a leaf (ADR 0021 fail-closed contract).
fn resolve_one(
    iv: &Intervention,
    depth: usize,
) -> Result<(VariableId, Option<f64>, f64), EstimationError> {
    let finite_numeric = |value: &Value, kind: &'static str| {
        value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
            EstimationError::unsupported(match kind {
                "set" => "intervention Set requires a finite numeric value",
                _ => "intervention Shift requires a finite numeric delta",
            })
        })
    };
    let one_finite_parameter = |mechanism: &MechanismOverride, family: &'static str| {
        if mechanism.parameters.len() != 1 || !mechanism.parameters[0].is_finite() {
            return Err(EstimationError::unsupported(match family {
                "constant" => "Soft(constant) requires exactly one finite parameter",
                _ => "Soft(additive_shift) requires exactly one finite parameter",
            }));
        }
        Ok(mechanism.parameters[0])
    };
    match iv {
        Intervention::Set { variable, value } => {
            let level = finite_numeric(value, "set")?;
            Ok((*variable, Some(level), 0.0))
        }
        Intervention::Shift { variable, delta } => {
            let d = finite_numeric(delta, "shift")?;
            Ok((*variable, None, d))
        }
        Intervention::Soft { variable, mechanism } => match mechanism.family_id.as_ref() {
            "constant" => {
                let level = one_finite_parameter(mechanism, "constant")?;
                Ok((*variable, Some(level), 0.0))
            }
            "additive_shift" => {
                let d = one_finite_parameter(mechanism, "additive_shift")?;
                Ok((*variable, None, d))
            }
            other => Err(EstimationError::data_msg(format!(
                "Soft mechanism family `{other}` is not licensed for temporal InterventionResponse; \
                 use constant or additive_shift"
            ))),
        },
        Intervention::Sequence(seq) => {
            if depth > 0 {
                return Err(EstimationError::unsupported(
                    "Intervention::Sequence nested inside a Sequence is not licensed for \
                     temporal InterventionResponse",
                ));
            }
            if seq.steps.len() != 1 {
                return Err(EstimationError::unsupported(
                    "multi-step Sequence must be planned as sequential overlays; \
                     refuse rather than collapse to the last step",
                ));
            }
            resolve_one(&seq.steps[0].intervention, depth + 1)
        }
        Intervention::Stochastic { .. } => Err(EstimationError::unsupported(
            "stochastic interventions are not licensed on the temporal InterventionResponse path",
        )),
        other => Err(EstimationError::data_msg(format!(
            "unsupported intervention variant for temporal response: {other:?}"
        ))),
    }
}

fn plan_sequence(
    seq: &InterventionSequence,
    spec: &TemporalResponseSpec,
    depth: usize,
) -> Result<TemporalInterventionPlan, EstimationError> {
    if seq.is_empty() {
        return Err(EstimationError::unsupported("empty Intervention::Sequence"));
    }
    if depth > 0 {
        return Err(EstimationError::unsupported(
            "Intervention::Sequence nested inside a Sequence is not licensed for \
             temporal InterventionResponse",
        ));
    }
    let origin = spec.treatment_offset()?;
    let mut leaves = Vec::with_capacity(seq.steps.len());
    for step in seq.steps.iter() {
        if matches!(step.intervention, Intervention::Sequence(_)) {
            return Err(EstimationError::unsupported(
                "Intervention::Sequence nested inside a Sequence is not licensed for \
                 temporal InterventionResponse",
            ));
        }
        let (variable, level, shift) = resolve_one(&step.intervention, depth + 1)?;
        let offsets =
            step.temporal.active_offsets().map_err(|e| EstimationError::data_msg(e.to_string()))?;
        leaves.push((variable, level, shift, offsets));
    }
    if leaves.len() == 1 {
        let (variable, level, shift, offsets) = &leaves[0];
        // Pulse(0) / a singleton window at 0 is the implicit Sequence shorthand
        // and attaches to the spec origin. Any other native step policy — Pulse(-2),
        // Sustained(-3, -1), or a window that already includes time 0 among other
        // times — is executed as written.
        let resolved = match offsets.as_ref() {
            [0] => Arc::<[i32]>::from([origin]),
            _ => Arc::clone(offsets),
        };
        let overlays = resolved
            .iter()
            .map(|&offset| SequentialNodeOverlay {
                variable: *variable,
                offset,
                level: *level,
                shift: *shift,
            })
            .collect();
        return Ok(TemporalInterventionPlan::Sequential { overlays });
    }
    let offsets = sequence_overlay_offsets(origin, &leaves)?;
    let mut overlays = Vec::with_capacity(leaves.len());
    let mut seen = Vec::new();
    for ((variable, level, shift, _), offset) in leaves.iter().zip(offsets) {
        let key = (*variable, offset);
        if seen.contains(&key) {
            return Err(EstimationError::unsupported(
                "Sequence assigns the same (variable, time) twice; refuse rather than collapse",
            ));
        }
        seen.push(key);
        overlays.push(SequentialNodeOverlay {
            variable: *variable,
            offset,
            level: *level,
            shift: *shift,
        });
    }
    Ok(TemporalInterventionPlan::Sequential { overlays })
}

/// Multi-step same variable → consecutive times ending at the spec origin.
/// Distinct variables, each once → joint at the origin.
/// Explicit distinct Pulse offsets are honored.
fn sequence_overlay_offsets(
    origin: i32,
    leaves: &[SequenceLeaf],
) -> Result<Vec<i32>, EstimationError> {
    let n = i32::try_from(leaves.len())
        .map_err(|_| EstimationError::unsupported("Sequence is too long"))?;
    let mut explicit = Vec::with_capacity(leaves.len());
    for (_, _, _, offsets) in leaves {
        let [at] = offsets.as_ref() else {
            return Err(EstimationError::unsupported(
                "each Sequence step must be a Pulse or single-time window",
            ));
        };
        explicit.push(*at);
    }
    let same_var = leaves.iter().all(|(variable, _, _, _)| *variable == leaves[0].0);
    let all_default = explicit.iter().all(|&at| at == 0);
    let all_origin = explicit.iter().all(|&at| at == origin);
    let distinct_explicit = {
        let mut seen = explicit.clone();
        seen.sort_unstable();
        seen.dedup();
        seen.len() == explicit.len() && !(all_default || all_origin)
    };
    if distinct_explicit {
        return Ok(explicit);
    }
    if same_var && (all_default || all_origin) {
        // Consecutive times ending at the policy origin. Two Pulse{0} (or
        // Pulse{origin}) steps are a two-step policy, not last-step collapse.
        return Ok((0..n).map(|i| origin - (n - 1 - i)).collect());
    }
    let unique_vars = {
        let mut vars: Vec<VariableId> =
            leaves.iter().map(|(variable, _, _, _)| *variable).collect();
        vars.sort_by_key(|variable| variable.raw());
        vars.dedup();
        vars.len() == leaves.len()
    };
    if unique_vars && (all_default || all_origin) {
        return Ok(vec![origin; leaves.len()]);
    }
    Err(EstimationError::unsupported(
        "Sequence schedule is ambiguous; use distinct Pulse offsets or a same-variable \
         consecutive policy",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::TemporalPolicy;

    #[test]
    fn soft_constant_resolves_to_set() {
        let v = VariableId::from_raw(0);
        let (t, level, shift) =
            resolve_one(&Intervention::soft(v, MechanismOverride::constant(1.5)), 0).unwrap();
        assert_eq!(t, v);
        assert_eq!(level, Some(1.5));
        assert!(shift.abs() < f64::EPSILON);
    }

    #[test]
    fn soft_unknown_family_refuses() {
        let v = VariableId::from_raw(0);
        let err = resolve_one(
            &Intervention::soft(v, MechanismOverride::named("linear_gaussian", vec![1.0])),
            0,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not licensed"));
    }

    #[test]
    fn non_finite_and_ambiguous_soft_parameters_fail_closed() {
        let v = VariableId::from_raw(0);
        let non_finite = resolve_one(&Intervention::set(v, Value::f64(f64::NAN)), 0).unwrap_err();
        assert!(non_finite.to_string().contains("finite numeric"));

        let too_many = resolve_one(
            &Intervention::soft(v, MechanismOverride::named("additive_shift", vec![1.0, 2.0])),
            0,
        )
        .unwrap_err();
        assert!(too_many.to_string().contains("exactly one finite parameter"));
    }

    #[test]
    fn multi_step_sequence_is_consecutive_overlays_not_last_step() {
        use antecedent_core::SequencedIntervention;

        let v = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let seq = InterventionSequence {
            steps: Arc::from(vec![
                SequencedIntervention {
                    intervention: Intervention::set(v, Value::f64(0.0)),
                    temporal: TemporalPolicy::pulse(0),
                },
                SequencedIntervention {
                    intervention: Intervention::set(v, Value::f64(5.0)),
                    temporal: TemporalPolicy::pulse(0),
                },
            ]),
        };
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(seq)], &spec).unwrap()
        else {
            panic!("expected sequential overlays");
        };
        assert_eq!(overlays.len(), 2);
        assert_eq!(overlays[0].offset, -2);
        assert_eq!(overlays[0].level, Some(0.0));
        assert_eq!(overlays[1].offset, -1);
        assert_eq!(overlays[1].level, Some(5.0));
        assert_ne!(overlays[0].level, overlays[1].level);
    }

    #[test]
    fn implicit_single_step_pulse_zero_attaches_to_spec_origin() {
        use antecedent_core::SequencedIntervention;

        let variable = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let sequence = InterventionSequence::new([SequencedIntervention {
            intervention: Intervention::set(variable, Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        }]);
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(sequence)], &spec).unwrap()
        else {
            panic!("an implicit Pulse(0) Sequence must resolve to overlays");
        };
        assert_eq!(
            overlays,
            vec![SequentialNodeOverlay { variable, offset: -1, level: Some(1.0), shift: 0.0 }]
        );
    }

    #[test]
    fn sequence_step_order_is_the_schedule_not_a_set() {
        use antecedent_core::SequencedIntervention;

        let variable = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let forward = InterventionSequence::new([
            SequencedIntervention {
                intervention: Intervention::set(variable, Value::f64(0.0)),
                temporal: TemporalPolicy::pulse(0),
            },
            SequencedIntervention {
                intervention: Intervention::set(variable, Value::f64(5.0)),
                temporal: TemporalPolicy::pulse(0),
            },
        ]);
        let reversed = InterventionSequence::new([
            SequencedIntervention {
                intervention: Intervention::set(variable, Value::f64(5.0)),
                temporal: TemporalPolicy::pulse(0),
            },
            SequencedIntervention {
                intervention: Intervention::set(variable, Value::f64(0.0)),
                temporal: TemporalPolicy::pulse(0),
            },
        ]);
        let TemporalInterventionPlan::Sequential { overlays: a } =
            plan_temporal_intervention(&[Intervention::Sequence(forward)], &spec).unwrap()
        else {
            panic!("expected sequential overlays");
        };
        let TemporalInterventionPlan::Sequential { overlays: b } =
            plan_temporal_intervention(&[Intervention::Sequence(reversed)], &spec).unwrap()
        else {
            panic!("expected sequential overlays");
        };
        assert_eq!(a[0].offset, -2);
        assert_eq!(a[1].offset, -1);
        assert_eq!(a[0].level, Some(0.0));
        assert_eq!(a[1].level, Some(5.0));
        assert_eq!(b[0].offset, -2);
        assert_eq!(b[1].offset, -1);
        assert_eq!(b[0].level, Some(5.0));
        assert_eq!(b[1].level, Some(0.0));
        assert_ne!(a, b);
    }

    #[test]
    fn single_step_sequence_preserves_explicit_pulse_offset() {
        use antecedent_core::SequencedIntervention;

        let variable = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let sequence = InterventionSequence::new([SequencedIntervention {
            intervention: Intervention::set(variable, Value::f64(2.0)),
            temporal: TemporalPolicy::pulse(-2),
        }]);
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(sequence)], &spec).unwrap()
        else {
            panic!("an explicit Sequence must resolve to overlays");
        };
        assert_eq!(
            overlays,
            vec![SequentialNodeOverlay { variable, offset: -2, level: Some(2.0), shift: 0.0 }]
        );
    }

    #[test]
    fn single_step_sequence_expands_explicit_sustained_policy() {
        use antecedent_core::SequencedIntervention;

        let variable = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let sequence = InterventionSequence::new([SequencedIntervention {
            intervention: Intervention::shift(variable, Value::f64(0.5)),
            temporal: TemporalPolicy::sustained(-3, -1),
        }]);
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(sequence)], &spec).unwrap()
        else {
            panic!("an explicit Sequence must resolve to overlays");
        };
        assert_eq!(
            overlays.iter().map(|overlay| overlay.offset).collect::<Vec<_>>(),
            vec![-3, -2, -1]
        );
        assert!(
            overlays.iter().all(
                |overlay| overlay.level.is_none() && (overlay.shift - 0.5).abs() < f64::EPSILON
            )
        );
    }

    #[test]
    fn nested_sequence_fails_closed() {
        use antecedent_core::SequencedIntervention;

        let v = VariableId::from_raw(0);
        let inner = InterventionSequence {
            steps: Arc::from(vec![SequencedIntervention {
                intervention: Intervention::set(v, Value::f64(0.0)),
                temporal: antecedent_core::TemporalPolicy::pulse(0),
            }]),
        };
        let outer = InterventionSequence {
            steps: Arc::from(vec![SequencedIntervention {
                intervention: Intervention::Sequence(inner),
                temporal: antecedent_core::TemporalPolicy::pulse(0),
            }]),
        };
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let err = plan_temporal_intervention(&[Intervention::Sequence(outer)], &spec).unwrap_err();
        assert!(err.to_string().contains("not licensed"));
    }

    // ---- GAP1: uncertainty is computed but was never asserted anywhere ----
    //
    // The band was fixed from a wrong formula (ATE-coefficient SE scaled by dose,
    // which gave a ZERO-WIDTH 95% interval at dose 0) to the correct linear-functional
    // variance `cbar(a)' Sigma cbar(a)`. These tests would fail against the old formula.

    use antecedent_core::{
        CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        ValidityBitmap,
    };
    use antecedent_graph::{TemporalDag, ensure_lagged};
    use antecedent_identify::TemporalBackdoorIdentifier;

    /// Deterministic AR(2)-ish series: `t` is a mildly autocorrelated continuous
    /// treatment, `y` depends on `t` lagged 1 and 2 steps. Non-degenerate `t` mean.
    fn synthetic_series(n: usize) -> (TimeSeriesData, TemporalDag) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "y"] {
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
        let mut y = vec![0.0; n];
        for i in 2..n {
            t[i] = 0.3 + 0.2 * t[i - 1] + 0.05 * (i as f64).sin();
            y[i] = 1.0 + 2.0 * t[i - 1] + 3.0 * t[i - 2] + 0.01 * (i as f64).cos();
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
        let mut graph = TemporalDag::empty();
        let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t1, y0).unwrap();
        graph.insert_directed(t2, y0).unwrap();
        (data, graph)
    }

    fn identify(graph: &TemporalDag, horizon_steps: u32) -> (IdentifiedEstimand, TemporalIndexer) {
        let id_query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_horizon_steps(horizon_steps)
                .with_policy(TemporalPolicy::pulse(0));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(graph, &id_query).unwrap();
        let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
        (estimand, id_res.indexer)
    }

    #[test]
    fn public_entry_point_revalidates_query_and_identification_status() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
        let invalid_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(0),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([-1.0, 1.0])),
            ),
        })
        .with_temporal(temporal.clone());
        let (data, _) = synthetic_series(100);
        let err = TemporalResponseEstimator::new()
            .estimate(
                &data,
                &[],
                &invalid_query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(7),
            )
            .unwrap_err();
        assert!(err.to_string().contains("same variable"), "got {err}");

        let valid_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([-1.0, 1.0])),
            ),
        })
        .with_temporal(temporal);
        let err = TemporalResponseEstimator::new()
            .estimate(
                &data,
                &[],
                &valid_query,
                IdentificationStatus::NotIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(7),
            )
            .unwrap_err();
        assert!(err.to_string().contains("point identification"), "got {err}");
    }

    #[test]
    fn response_preserves_the_exact_caller_estimand() {
        let (data, graph) = synthetic_series(300);
        let (estimand, indexer) = identify(&graph, 1);
        let temporal =
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Linspace { start: -1.0, end: 1.0, points: 3 },
            ),
        })
        .with_temporal(temporal);
        let result = TemporalResponseEstimator::new()
            .estimate(
                &data,
                &[(&estimand, &indexer)],
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(7),
            )
            .unwrap();
        assert_eq!(result.estimand, query.functional);
    }

    /// (a) `lower < mean < upper` strictly for every cell of the dose x horizon surface,
    /// and the band is symmetric about the mean.
    /// (b) A dose of exactly 0.0 in the grid produces a STRICTLY POSITIVE band width —
    /// the direct regression guard for the old zero-width-at-dose-0 bug.
    #[test]
    fn uncertainty_band_strict_and_zero_dose_has_positive_width() {
        let (data, graph) = synthetic_series(400);
        let (estimand, indexer) = identify(&graph, 4);
        let doses = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let n_h = 3;
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 2, 4], TemporalPolicy::pulse(0), None).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from(doses.clone())),
            ),
        })
        .with_temporal(temporal);
        let est = TemporalResponseEstimator::new();
        let result = est
            .estimate(
                &data,
                &[(&estimand, &indexer), (&estimand, &indexer), (&estimand, &indexer)],
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(7),
            )
            .unwrap();

        assert!(
            result.assumptions.entries.iter().any(|r| matches!(
                &r.assumption,
                Assumption::ParametricRestriction(p) if p.id.as_ref() == "ols.homoskedastic.pointwise"
            )),
            "temporal response must record the homoskedastic pointwise OLS assumption"
        );
        assert!(result.assumptions.entries.iter().any(|r| matches!(
            &r.assumption,
            Assumption::ParametricRestriction(p) if p.id.as_ref() == "ols.linear_additive.gcomp"
        )));

        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            &result.estimate
        else {
            panic!("expected point-identified surface");
        };
        let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &result.uncertainty else {
            panic!(
                "expected PointwiseBand uncertainty on the temporal MeanCurve path — this is \
                 exactly the GAP1 regression this test guards against"
            );
        };
        assert_eq!(mean.len(), doses.len() * n_h);
        assert_eq!(mean.len(), lower.len());
        assert_eq!(mean.len(), upper.len());

        for i in 0..mean.len() {
            assert!(lower[i] < mean[i], "cell {i}: lower {} not < mean {}", lower[i], mean[i]);
            assert!(mean[i] < upper[i], "cell {i}: mean {} not < upper {}", mean[i], upper[i]);
            // Symmetric about the mean by construction (mean +/- z*se); assert it holds.
            assert!(
                (mean[i] - lower[i] - (upper[i] - mean[i])).abs() < 1e-9,
                "cell {i}: band not symmetric about mean (lower half {}, upper half {})",
                mean[i] - lower[i],
                upper[i] - mean[i]
            );
        }

        // (b): dose == 0.0 must produce a strictly positive band width. Under the old
        // (buggy) formula — ATE-coefficient SE scaled by dose — the width at dose 0.0
        // was exactly zero.
        let zero_idx = doses.iter().position(|&d| d == 0.0).unwrap();
        for h in 0..n_h {
            let idx = zero_idx * n_h + h;
            let width = upper[idx] - lower[idx];
            assert!(
                width > 1e-9,
                "dose=0.0 horizon-slot {h}: band width {width} is not strictly positive \
                 (regression guard: old formula gave a zero-width interval at dose 0)"
            );
        }
    }

    /// (c) Recompute `se(dose)` independently — build `cbar` from the fitted design's
    /// column means with the treatment entry replaced by `dose`, then form the quadratic
    /// form against the coefficient covariance directly (own loop, not `mean_and_se_at`)
    /// — and check it matches production. A regression to the old ATE-coefficient-SE
    /// formula would diverge from this independent recompute.
    #[test]
    fn uncertainty_se_matches_independent_quadratic_form_recompute() {
        let (data, graph) = synthetic_series(300);
        let (estimand, indexer) = identify(&graph, 3);
        let temporal =
            TemporalResponseSpec::new(vec![3u32], TemporalPolicy::pulse(0), None).unwrap();
        let est = TemporalResponseEstimator::new();
        let mut ws = LeastSquaresWorkspace::default();
        let fitted = est
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                3,
                &indexer,
                &ExecutionContext::for_tests(9),
                &mut ws,
            )
            .unwrap();

        let p = fitted.coefs.len();
        let z = normal_ppf(0.975);
        for &dose in &[-2.0, -0.5, 0.0, 0.5, 1.0, 3.0] {
            let mut cbar = fitted.column_means.clone();
            cbar[TREATMENT_COL] = dose;
            let mut expected_var = 0.0;
            for i in 0..p {
                for j in 0..p {
                    expected_var += cbar[i] * fitted.cov[i * p + j] * cbar[j];
                }
            }
            let expected_se = expected_var.max(0.0).sqrt();
            let (mu, actual_se) = fitted.mean_and_se_at(dose);

            assert!(
                (actual_se - expected_se).abs() <= 1e-9_f64.max(1e-9 * expected_se),
                "dose={dose}: production se {actual_se} != independently recomputed se {expected_se}"
            );
            // Half-width equals normal_ppf(0.975) * se, for the independently recomputed se.
            let expected_lower = mu - z * expected_se;
            let expected_upper = mu + z * expected_se;
            assert!((expected_upper - mu - z * expected_se).abs() < 1e-9);
            assert!((mu - expected_lower - z * expected_se).abs() < 1e-9);
        }
    }

    /// (d) se(dose) grows as the dose moves away from the observed treatment mean — the
    /// standard widening of a regression band away from the design centroid. Under the
    /// old buggy formula (se proportional to |dose|, minimized at dose == 0.0), this
    /// fails whenever the fitted treatment mean is nonzero, since `se(0.0)` would then be
    /// smaller than `se(treatment_mean)`.
    #[test]
    fn uncertainty_se_widens_away_from_treatment_mean() {
        let (data, graph) = synthetic_series(400);
        let (estimand, indexer) = identify(&graph, 4);
        let temporal =
            TemporalResponseSpec::new(vec![2u32], TemporalPolicy::pulse(0), None).unwrap();
        let est = TemporalResponseEstimator::new();
        let mut ws = LeastSquaresWorkspace::default();
        let fitted = est
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                2,
                &indexer,
                &ExecutionContext::for_tests(7),
                &mut ws,
            )
            .unwrap();

        let center = fitted.treatment_mean();
        assert!(
            center.abs() > 1e-3,
            "test fixture assumes a non-degenerate (nonzero) treatment mean; got {center}"
        );
        let (_, se_center) = fitted.mean_and_se_at(center);
        let (_, se_near) = fitted.mean_and_se_at(center + 1.0);
        let (_, se_far) = fitted.mean_and_se_at(center + 3.0);
        let (_, se_near_neg) = fitted.mean_and_se_at(center - 1.0);
        let (_, se_zero) = fitted.mean_and_se_at(0.0);

        assert!(
            se_center < se_near,
            "se should grow moving away from center: {se_center} vs {se_near}"
        );
        assert!(
            se_near < se_far,
            "se should keep growing further from center: {se_near} vs {se_far}"
        );
        assert!(
            se_center < se_near_neg,
            "se should grow symmetrically on the other side of the center: {se_center} vs {se_near_neg}"
        );
        assert!(
            se_center < se_zero,
            "se at the treatment mean ({center}) should be smaller than se at dose=0.0 ({se_zero}); \
             the old bug's minimum was at dose 0, not at the design centroid"
        );
    }

    // ---- GAP2: refusal paths were unasserted ----

    /// (f) A `Sequence` spanning multiple target variables must refuse. Exercised directly
    /// against `resolve_sequence` (rather than end-to-end through `Study`) because a
    /// cross-variable `Sequence` has no unique `primary_variable`, so the facade already
    /// refuses earlier ("no treatment/outcome pair") before temporal resolution runs.
    #[test]
    fn sequence_multiple_target_variables_fails_closed() {
        use antecedent_core::SequencedIntervention;

        let v0 = VariableId::from_raw(0);
        let v1 = VariableId::from_raw(1);
        let seq = InterventionSequence {
            steps: Arc::from(vec![
                SequencedIntervention {
                    intervention: Intervention::set(v0, Value::f64(1.0)),
                    temporal: antecedent_core::TemporalPolicy::pulse(0),
                },
                SequencedIntervention {
                    intervention: Intervention::set(v1, Value::f64(2.0)),
                    temporal: antecedent_core::TemporalPolicy::pulse(0),
                },
            ]),
        };
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(seq)], &spec).unwrap()
        else {
            panic!("joint Sequence must be sequential overlays");
        };
        assert_eq!(overlays.len(), 2);
        assert_eq!(overlays[0].variable, v0);
        assert_eq!(overlays[1].variable, v1);
        assert_eq!(overlays[0].offset, -1);
        assert_eq!(overlays[1].offset, -1);
    }

    /// (d) Empty dose grid must refuse with a specific message, not silently produce an
    /// empty (or garbage) surface.
    #[test]
    fn empty_dose_grid_refuses() {
        let (data, graph) = synthetic_series(300);
        let (estimand, indexer) = identify(&graph, 2);
        let temporal =
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
        let est = TemporalResponseEstimator::new();
        let err = est
            .estimate_mean_curve(
                &data,
                &[(&estimand, &indexer)],
                VariableId::from_raw(1),
                VariableId::from_raw(0),
                &[],
                &temporal,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap_err();
        assert!(err.to_string().contains("dose grid must be non-empty"), "unexpected error: {err}");
    }

    /// The union of per-horizon treatment ranges would call dose 1.5 supported
    /// here; the cell grid must not.
    #[test]
    fn surface_support_uses_per_horizon_ranges_not_union() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 2, 8], TemporalPolicy::pulse(0), None).unwrap();
        let ranges = [(-2.0, 2.0), (-1.0, 1.0), (-0.4, 0.4)];
        let report = mean_curve_support(&[1.5], &temporal, &ranges);
        assert_eq!(report.status, SupportStatus::Extrapolative);
        assert_eq!(
            report.point_status.as_ref().map(AsRef::as_ref),
            Some(
                [
                    SupportStatus::Supported,
                    SupportStatus::OutsideEmpiricalSupport,
                    SupportStatus::OutsideEmpiricalSupport,
                ]
                .as_slice()
            )
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.temporal.partial_horizon_support")
        );
        let ranges_diag = report
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == "response.temporal.horizon_treatment_range")
            .expect("horizon treatment range diagnostic");
        assert_eq!(ranges_diag.values.as_ref(), [-2.0, 2.0, -1.0, 1.0, -0.4, 0.4].as_slice());
    }

    #[test]
    fn surface_support_all_outside_and_all_supported() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 2], TemporalPolicy::pulse(0), None).unwrap();
        let ranges = [(-1.0, 1.0), (-0.5, 0.5)];
        let outside = mean_curve_support(&[10.0], &temporal, &ranges);
        assert_eq!(outside.status, SupportStatus::OutsideEmpiricalSupport);
        assert!(
            outside
                .point_status
                .as_ref()
                .unwrap()
                .iter()
                .all(|s| { *s == SupportStatus::OutsideEmpiricalSupport })
        );
        assert!(
            outside
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.outside_empirical_support")
        );

        let inside = mean_curve_support(&[0.0], &temporal, &ranges);
        assert_eq!(inside.status, SupportStatus::Supported);
        assert!(inside.warnings.is_empty());
        assert!(
            inside.point_status.as_ref().unwrap().iter().all(|s| *s == SupportStatus::Supported)
        );
    }

    #[test]
    fn intervention_support_is_one_cell_per_horizon() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 2, 8], TemporalPolicy::pulse(0), None).unwrap();
        let ranges = [(-2.0, 2.0), (-1.0, 1.0), (-0.4, 0.4)];
        let report = intervention_support(&[1.5, 1.5, 1.5], Some(1.5), 0.0, &temporal, &ranges);
        assert_eq!(report.status, SupportStatus::Extrapolative);
        assert_eq!(report.point_status.as_ref().unwrap().len(), 3);
        let layout = report
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.id.as_ref() == "response.temporal.intervention_horizon_layout"
            })
            .expect("intervention layout");
        assert_eq!(layout.values.as_ref(), &[3.0]);
    }

    #[test]
    fn shift_support_checks_the_shifted_law_not_only_its_mean() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
        let ranges = [(0.0, 1.0)];

        // E[A] + 0.4 = 0.9 is inside the observed range, but the shifted
        // treatment law spans [0.4, 1.4] and therefore extrapolates.
        let partial = intervention_support(&[0.9], None, 0.4, &temporal, &ranges);
        assert_eq!(partial.status, SupportStatus::Extrapolative);
        assert_eq!(partial.point_status.as_deref(), Some(&[SupportStatus::Extrapolative][..]));
        assert!(partial.warnings.iter().any(|warning| {
            warning.code.as_ref() == "response.temporal.shift_distribution_extrapolative"
        }));
        assert_eq!(partial.query_region.minima.as_ref(), &[0.4, 1.0]);
        assert_eq!(partial.query_region.maxima.as_ref(), &[1.4, 1.0]);
        let shifted = partial
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.id.as_ref() == "response.temporal.shifted_treatment_range"
            })
            .expect("shifted treatment range");
        assert_eq!(shifted.values.as_ref(), &[0.4, 1.4]);

        // A disjoint shifted law gets the stronger outside-support status.
        let outside = intervention_support(&[2.5], None, 2.0, &temporal, &ranges);
        assert_eq!(outside.status, SupportStatus::OutsideEmpiricalSupport);
        assert_eq!(
            outside.point_status.as_deref(),
            Some(&[SupportStatus::OutsideEmpiricalSupport][..])
        );

        let identity = intervention_support(&[0.5], None, 0.0, &temporal, &ranges);
        assert_eq!(identity.status, SupportStatus::Supported);
    }

    /// Extreme T only at the end of the series. A longer-horizon pulse looks
    /// further back, so that spike is inside the h=1 treatment column and
    /// outside a long-horizon window.
    fn spike_then_quiet_series(n: usize) -> (TimeSeriesData, TemporalDag) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "y"] {
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
        let mut y = vec![0.0; n];
        for (i, t_i) in t.iter_mut().enumerate() {
            *t_i = if i >= n.saturating_sub(7) { 10.0 } else { 0.05 * (i as f64).sin() };
        }
        for i in 2..n {
            y[i] = 1.0 + 2.0 * t[i - 1] + 3.0 * t[i - 2];
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
        let mut graph = TemporalDag::empty();
        let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t1, y0).unwrap();
        graph.insert_directed(t2, y0).unwrap();
        (data, graph)
    }

    #[test]
    fn late_treatment_spike_is_horizon_specific_support() {
        let (data, graph) = spike_then_quiet_series(80);
        let (estimand, indexer) = identify(&graph, 8);
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 8], TemporalPolicy::pulse(0), None).unwrap();
        let est = TemporalResponseEstimator::new();
        let mut ws = LeastSquaresWorkspace::default();
        let ctx = ExecutionContext::for_tests(11);
        let short = est
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                1,
                &indexer,
                &ctx,
                &mut ws,
            )
            .unwrap();
        let long = est
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                8,
                &indexer,
                &ctx,
                &mut ws,
            )
            .unwrap();
        let short_range = range(&short.prepared.treatment);
        let long_range = range(&long.prepared.treatment);
        assert!(
            short_range.1 > long_range.1 + 1.0,
            "long horizon should miss the late spike: short={short_range:?} long={long_range:?}"
        );
        let dose = long_range.1 + (short_range.1 - long_range.1) * 0.5;
        let result = est
            .estimate_mean_curve(
                &data,
                &[(&estimand, &indexer), (&estimand, &indexer)],
                VariableId::from_raw(1),
                VariableId::from_raw(0),
                &[0.0, dose],
                &temporal,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ctx,
            )
            .unwrap();
        assert_eq!(result.support.status, SupportStatus::Extrapolative);
        let cells = result.support.point_status.as_ref().expect("temporal point_status");
        // dose-major: (0, h=1), (0, h=8), (dose, h=1), (dose, h=8)
        assert_eq!(cells[0], SupportStatus::Supported);
        assert_eq!(cells[1], SupportStatus::Supported);
        assert_eq!(cells[2], SupportStatus::Supported);
        assert_eq!(cells[3], SupportStatus::OutsideEmpiricalSupport);
        // The union envelope would have classified `dose` as supported.
        assert!(dose >= short_range.0 && dose <= short_range.1);
        assert!(dose < long_range.0 || dose > long_range.1);
    }

    #[test]
    fn bootstrap_se_differs_from_analytic_ols_se() {
        let (data, graph) = synthetic_series(240);
        let (estimand, indexer) = identify(&graph, 3);
        let temporal =
            TemporalResponseSpec::new(vec![3u32], TemporalPolicy::pulse(0), None).unwrap();
        let mut ols_ws = LeastSquaresWorkspace::default();
        let analytic = TemporalResponseEstimator::new()
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                3,
                &indexer,
                &ExecutionContext::for_tests(13),
                &mut ols_ws,
            )
            .unwrap();
        let boot = TemporalResponseEstimator::new()
            .with_bootstrap_replicates(40)
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                3,
                &indexer,
                &ExecutionContext::for_tests(13),
                &mut ols_ws,
            )
            .unwrap();
        let (_, se_a) = analytic.mean_and_se_at(1.0);
        let (_, se_b) = boot.mean_and_se_at(1.0);
        assert!(se_a.is_finite() && se_a > 0.0, "analytic se={se_a}");
        assert!(se_b.is_finite() && se_b > 0.0, "bootstrap se={se_b}");
        assert!(
            (se_a - se_b).abs() > 1e-6,
            "Study bootstrap must change the surface SE (analytic={se_a}, bootstrap={se_b})"
        );
        let (mu_a, _) = analytic.mean_and_se_at(1.0);
        let (mu_b, _) = boot.mean_and_se_at(1.0);
        assert!(
            (mu_a - mu_b).abs() < 1e-12,
            "point estimate must stay full-sample OLS (analytic={mu_a}, boot={mu_b})"
        );
    }
}
