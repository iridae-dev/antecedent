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
    Intervention, InterventionSequence, MechanismOverride, ParametricAssumption,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    SupportDiagnostic, SupportRegion, SupportReport, SupportStatus, TargetPopulation,
    TemporalEffectQuery, TemporalNodeKey, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TemporalIndexer, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, SandwichKind,
    coefficient_covariance, normal_ppf,
};

use crate::adjustment::{LinearAdjustmentAte, PreparedEstimationProblem};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::temporal_adjustment::TemporalLinearAdjustment;
use crate::util::range;

/// Index of the treatment column in the compiled design (col0 = intercept, col1 = treatment).
/// Verified against `CompiledDesign::linear_adjustment`.
const TREATMENT_COL: usize = 1;

/// Per-horizon lag-aligned observed treatment `(min, max)`.
type HorizonTreatmentRange = (f64, f64);

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
        match &query.functional {
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
                let (treatment, level, shift) = resolve_temporal_intervention(interventions)?;
                self.estimate_intervention_curve(
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
                )
            }
            _ => Err(EstimationError::unsupported(
                "temporal response is licensed only for MeanCurve and InterventionResponse",
            )),
        }
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
        let mut mean = Vec::with_capacity(doses.len().saturating_mul(n_h));
        let mut lower = Vec::with_capacity(mean.capacity());
        let mut upper = Vec::with_capacity(mean.capacity());

        // Layout: value[d * n_horizons + h] — dose major, then horizon.
        let (per_horizon, horizon_ranges, horizon_identification) = self.run_per_horizon(
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
                grid: Arc::from(flatten_dose_horizon_grid(doses, &temporal.horizons)),
                dimension: 2,
                mean: Arc::from(mean),
            }),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            },
            support: mean_curve_support(doses, temporal, &horizon_ranges),
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.gcomp"),
            horizon_identification: Some(Arc::from(horizon_identification)),
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
        let (per_horizon, horizon_ranges, horizon_identification) = self.run_per_horizon(
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
            support: intervention_support(&eval_levels, temporal, &horizon_ranges),
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.intervention_gcomp"),
            horizon_identification: Some(Arc::from(horizon_identification)),
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
    ) -> Result<(Vec<T>, Vec<HorizonTreatmentRange>, Vec<HorizonIdentification>), EstimationError>
    {
        let mut ols_ws = LeastSquaresWorkspace::default();
        let mut results = Vec::with_capacity(temporal.horizons.len());
        let mut horizon_ranges = Vec::with_capacity(temporal.horizons.len());
        let mut horizon_identification = Vec::with_capacity(temporal.horizons.len());

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
            results.push(per_horizon(&fitted));
        }

        Ok((results, horizon_ranges, horizon_identification))
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
        FittedHorizon::fit(prepared, ols_ws)
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
}

impl FittedHorizon {
    fn fit(
        prepared: PreparedEstimationProblem,
        ols_ws: &mut LeastSquaresWorkspace,
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
        Ok(Self { prepared, coefs: fit.coefficients, cov, column_means })
    }

    fn treatment_mean(&self) -> f64 {
        self.column_means[TREATMENT_COL]
    }

    /// `(mu_hat(dose), se(dose))` via the exact linear-functional closed form.
    fn mean_and_se_at(&self, dose: f64) -> (f64, f64) {
        let p = self.coefs.len();
        let mut cbar = self.column_means.clone();
        cbar[TREATMENT_COL] = dose;
        let mut mu = 0.0;
        for (&coef, &c) in self.coefs.iter().zip(cbar.iter()) {
            mu += coef * c;
        }
        let mut var = 0.0;
        for (i, &ci) in cbar.iter().enumerate() {
            let row = &self.cov[i * p..i * p + p];
            let row_sum: f64 = row.iter().zip(cbar.iter()).map(|(&cov_ij, &cj)| cov_ij * cj).sum();
            var += ci * row_sum;
        }
        let se = var.max(0.0).sqrt();
        (mu, se)
    }
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

fn flatten_dose_horizon_grid(doses: &[f64], horizons: &[u32]) -> Vec<f64> {
    let mut grid = Vec::with_capacity(doses.len().saturating_mul(horizons.len()).saturating_mul(2));
    for &dose in doses {
        for &h in horizons {
            grid.push(dose);
            grid.push(f64::from(h));
        }
    }
    grid
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
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
) -> SupportReport {
    let point_status: Vec<SupportStatus> = eval_levels
        .iter()
        .zip(horizon_ranges.iter())
        .map(|(&dose, &(lo, hi))| cell_against_range(dose, lo, hi))
        .collect();
    assemble_temporal_support(eval_levels, temporal, horizon_ranges, point_status)
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

fn resolve_temporal_intervention(
    interventions: &[Intervention],
) -> Result<(VariableId, Option<f64>, f64), EstimationError> {
    if interventions.is_empty() {
        return Err(EstimationError::unsupported(
            "intervention response requires at least one intervention",
        ));
    }
    if interventions.len() > 1 {
        return Err(EstimationError::unsupported(
            "temporal InterventionResponse supports one primary intervention \
             (use Sequence for multi-step policies on one variable)",
        ));
    }
    resolve_one(&interventions[0], 0)
}

/// `depth` counts levels of `Intervention::Sequence` nesting already entered.
/// `resolve_sequence` is only reachable from a Sequence itself, so a `depth > 0`
/// arrival there means a Sequence nested inside a Sequence — refused explicitly
/// rather than silently recursing into a leaf (ADR 0021 fail-closed contract).
fn resolve_one(
    iv: &Intervention,
    depth: usize,
) -> Result<(VariableId, Option<f64>, f64), EstimationError> {
    match iv {
        Intervention::Set { variable, value } => {
            let level = value.as_f64().ok_or_else(|| {
                EstimationError::unsupported("intervention Set requires a numeric value")
            })?;
            Ok((*variable, Some(level), 0.0))
        }
        Intervention::Shift { variable, delta } => {
            let d = delta.as_f64().ok_or_else(|| {
                EstimationError::unsupported("intervention Shift requires a numeric delta")
            })?;
            Ok((*variable, None, d))
        }
        Intervention::Soft { variable, mechanism } => match mechanism.family_id.as_ref() {
            "constant" => {
                let level = mechanism.parameters.first().copied().ok_or_else(|| {
                    EstimationError::unsupported("Soft(constant) requires a parameter")
                })?;
                Ok((*variable, Some(level), 0.0))
            }
            "additive_shift" => {
                let d = mechanism.parameters.first().copied().ok_or_else(|| {
                    EstimationError::unsupported("Soft(additive_shift) requires a parameter")
                })?;
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
            resolve_sequence(seq, depth + 1)
        }
        Intervention::Stochastic { .. } => Err(EstimationError::unsupported(
            "stochastic interventions are not licensed on the temporal InterventionResponse path",
        )),
        other => Err(EstimationError::data_msg(format!(
            "unsupported intervention variant for temporal response: {other:?}"
        ))),
    }
}

/// A single-step `Sequence` is licensed and resolves as its one step. More than one step
/// must fail closed rather than silently collapsing to the last step's (level, shift) —
/// see ADR 0021's "multi-step policies are refused" contract. The original cross-variable
/// diagnostic is preserved when it applies (checked before the generic multi-step refusal).
fn resolve_sequence(
    seq: &InterventionSequence,
    depth: usize,
) -> Result<(VariableId, Option<f64>, f64), EstimationError> {
    if seq.is_empty() {
        return Err(EstimationError::unsupported("empty Intervention::Sequence"));
    }
    if seq.steps.len() > 1 {
        let mut first_var: Option<VariableId> = None;
        for step in seq.steps.iter() {
            let (var, _, _) = resolve_one(&step.intervention, depth)?;
            if let Some(fv) = first_var {
                if fv != var {
                    return Err(EstimationError::unsupported(
                        "Sequence with multiple target variables is not licensed for temporal \
                         InterventionResponse",
                    ));
                }
            } else {
                first_var = Some(var);
            }
        }
        return Err(EstimationError::unsupported(
            "multi-step Sequence intervention policies are not licensed for temporal \
             InterventionResponse (ADR 0021 fail-closed contract)",
        ));
    }
    resolve_one(&seq.steps[0].intervention, depth)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn multi_step_sequence_fails_closed() {
        use antecedent_core::SequencedIntervention;

        let v = VariableId::from_raw(0);
        let seq = InterventionSequence {
            steps: Arc::from(vec![
                SequencedIntervention {
                    intervention: Intervention::set(v, Value::f64(0.0)),
                    temporal: antecedent_core::TemporalPolicy::pulse(0),
                },
                SequencedIntervention {
                    intervention: Intervention::set(v, Value::f64(5.0)),
                    temporal: antecedent_core::TemporalPolicy::pulse(0),
                },
            ]),
        };
        let err = resolve_sequence(&seq, 0).unwrap_err();
        assert!(err.to_string().contains("not licensed"));
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
        let err = resolve_one(&Intervention::Sequence(outer), 0).unwrap_err();
        assert!(err.to_string().contains("not licensed"));
    }

    // ---- GAP1: uncertainty is computed but was never asserted anywhere ----
    //
    // The band was fixed from a wrong formula (ATE-coefficient SE scaled by dose,
    // which gave a ZERO-WIDTH 95% interval at dose 0) to the correct linear-functional
    // variance `cbar(a)' Sigma cbar(a)`. These tests would fail against the old formula.

    use antecedent_core::{
        CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, TemporalPolicy,
        ValueType,
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
        let err = resolve_sequence(&seq, 0).unwrap_err();
        assert!(err.to_string().contains("multiple target variables"), "unexpected error: {err}");
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
        let report = intervention_support(&[1.5, 1.5, 1.5], &temporal, &ranges);
        assert_eq!(report.status, SupportStatus::Extrapolative);
        assert_eq!(report.point_status.as_ref().unwrap().len(), 3);
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
}
