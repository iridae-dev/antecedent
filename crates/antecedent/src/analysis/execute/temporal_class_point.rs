//! TemporalCpdag/TemporalPag conditional, derivative, and cross-world points.
//!
//! Each identified completion uses the existing temporal linear producer. The
//! published point is the completion-weighted mean. Enumeration is not a
//! posterior probability.
//!
//! Uncertainty (see `temporal_point_uncertainty`): the frequentist interval is
//! the shared circular-block bootstrap of the whole frozen-weight procedure,
//! every regression of every completion refit on the same resampled series
//! times, with an Imbens–Manski interval for the identified set over the
//! completions from the same replicates. The Bayesian side draws every
//! completion's slope posterior; one completion publishes its posterior, and
//! several publish an interval for the identified set over the completions
//! (enumeration weights are not posterior mass, so the completion posteriors are
//! not mixed).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::series_weights::refuse_unread_series_weights;
use super::temporal_path::CrossWorldContrast;
use super::temporal_point_uncertainty::{
    self as point_uncertainty, FrozenAtom, SlopeAtom, SlopeSource,
};
use super::{
    Arc, CausalError, CausalPosterior, CausalQuery, CausalResponse, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecuteRun, ExecutionContext, GraphClass,
    GraphPosterior, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedKeyWeights, IdentifierId,
    InferenceMode, Instant, Intervention, OverlapPolicy, PhysicalExecutionPlan,
    PosteriorMassSummary, RefuteSuite, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, Strategy, StudyResult, TableView, TemporalEffectQuery,
    TemporalIndexer, TimeSeriesData, VariableId, elapsed_ns, envelope_to_identification_result_for,
    identification_status_ok_for_case, identify_cached_diagnostic, report_identify_compute,
    resolve_posterior_status, select_estimand,
};
mod cross_world;
mod derivative;
mod posterior;

use cross_world::{collect_class_cross_world, cross_world_from, cross_world_note};
use derivative::{
    ClassDerivatives, class_derivative_diagnostics, class_derivative_responses,
    derivative_evaluator, derivative_on_envelope, derivative_probe, derivative_structural_mixture,
    response_center, retain_graph_dependent_derivatives,
};

use crate::planner::RouteDefaults;
use antecedent_core::SupportStatus;

const CONDITIONAL_PRIOR_REFUSAL: &str = "temporal conditional effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not supported";
const DERIVATIVE_PRIOR_REFUSAL: &str = "temporal derivative uses the isotropic conjugate prior scale; a mapped coefficient prior is not supported";
const CROSS_WORLD_PRIOR_REFUSAL: &str = "temporal cross-world effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not supported";

/// The identified completions of one envelope whose own regression could not be fit, and the
/// first cause. Their mass is unevaluable, not lost: it stays in the structural mixture.
#[derive(Default)]
struct SkippedCompletions {
    count: usize,
    first_cause: Option<String>,
}

impl SkippedCompletions {
    /// Count `error` as one unfittable completion, or return it when it ends the run or
    /// describes the request rather than the completion.
    fn record(&mut self, error: CausalError) -> Result<(), CausalError> {
        if !completion_cannot_be_fit(&error) {
            return Err(error);
        }
        self.count += 1;
        self.first_cause.get_or_insert_with(|| error.to_string());
        Ok(())
    }

    fn absorb(&mut self, other: &Self) {
        self.count += other.count;
        if self.first_cause.is_none() {
            self.first_cause.clone_from(&other.first_cause);
        }
    }

    /// The refusal for an envelope none of whose identified completions could be fit.
    fn none_estimable(&self) -> CausalError {
        let message = match &self.first_cause {
            Some(cause) => format!(
                "temporal class envelope had no estimable identified completions: all {} \
                 identified completions failed to fit (first: {cause})",
                self.count
            ),
            None => "temporal class envelope had no estimable identified completions".into(),
        };
        CausalError::Compile { message }
    }

    fn diagnostic(&self) -> Option<Diagnostic> {
        let cause = self.first_cause.as_deref()?;
        Some(Diagnostic::new(
            "estimate.temporal_class.completions_skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "{} identified completion(s) could not be fit and are excluded from the \
                 published point and its uncertainty; their mass is unevaluable (first: {cause})",
                self.count
            ),
        ))
    }
}

/// The identified completions of one class envelope, each with its recorded
/// regressions, frozen weight and point.
struct ClassPoints<'a> {
    atoms: Vec<SlopeAtom<'a>>,
    skipped: SkippedCompletions,
    weights: Vec<f64>,
    points: Vec<f64>,
    /// Point by envelope case (structural mixture), `None` where not evaluated.
    values: Vec<Option<f64>>,
    estimand: IdentifiedEstimand,
    assumptions: antecedent_core::AssumptionSet,
    /// First identified completion, used to name the cross-world functional.
    primary_identification: IdentificationResult,
    primary_indexer: TemporalIndexer,
}

impl ClassPoints<'_> {
    fn point(&self) -> Result<f64, CausalError> {
        let pairs: Vec<(f64, f64)> =
            self.weights.iter().copied().zip(self.points.iter().copied()).collect();
        weighted_mean(&pairs)
    }
}

/// How a class point route reports its uncertainty: the inference mode, the
/// bootstrap budget, the structural lag span and the RNG stream bases.
#[derive(Clone, Copy)]
pub(super) struct ClassPointUncertainty<'a> {
    pub(super) inference: &'a InferenceMode,
    pub(super) bootstrap_replicates: u32,
    pub(super) structural_span: usize,
    pub(super) block_stream: u64,
    pub(super) posterior_stream: u64,
}

/// The uncertainty a scalar class point route publishes beside its point.
pub(super) struct ClassScalarOutcome {
    pub(super) estimate: EffectEstimate,
    pub(super) posterior: Option<CausalPosterior>,
    pub(super) identified_set: Option<antecedent_estimate::IdentifiedSetInterval>,
    pub(super) diagnostics: Vec<Diagnostic>,
    pub(super) replicates_ok: Option<u32>,
    pub(super) cancelled: bool,
}

/// An assembled class conditional fit.
pub(super) struct ClassConditionalFit {
    pub(super) estimand: IdentifiedEstimand,
    pub(super) diagnostics: Vec<Diagnostic>,
    pub(super) structural: crate::result::StructuralResponseMixture,
    pub(super) outcome: ClassScalarOutcome,
}

fn isotropic_prior_scale(
    inference: &InferenceMode,
    message: &'static str,
) -> Result<Option<f64>, CausalError> {
    match inference {
        InferenceMode::Bayesian(cfg) => {
            if cfg.has_transfer() {
                return Err(CausalError::Unsupported { message });
            }
            Ok(Some(cfg.prior_scale))
        }
        InferenceMode::Frequentist => Ok(None),
    }
}

fn weighted_mean(values: &[(f64, f64)]) -> Result<f64, CausalError> {
    let mut weighted = 0.0;
    let mut total = 0.0;
    for &(weight, point) in values {
        if !point.is_finite() {
            return Err(CausalError::Compile {
                message: "temporal class completion produced a non-finite point".into(),
            });
        }
        weighted += weight * point;
        total += weight;
    }
    if !matches!(total.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        return Err(CausalError::Compile {
            message: "temporal class envelope had no estimable identified completions".into(),
        });
    }
    Ok(weighted / total)
}

fn weighted_derivative_mean(values: &[(f64, f64)]) -> Result<f64, CausalError> {
    if values.iter().all(|(_, point)| point.is_nan()) && !values.is_empty() {
        return Ok(f64::NAN);
    }
    weighted_mean(values)
}

fn derivative_response_finite(response: &CausalResponse) -> bool {
    match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => value.is_finite(),
        ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) => {
            values.iter().all(|value| value.is_finite())
        }
        ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }) => {
            values.iter().all(|value| value.is_finite())
        }
        _ => false,
    }
}

fn require_identified_mass(
    envelope: &antecedent_identify::IdentificationEnvelope<
        antecedent_identify::TemporalCompletionGraph,
    >,
) -> Result<(), CausalError> {
    if matches!(envelope.status, IdentificationStatus::NotIdentified)
        || envelope.identified_weight.0 <= 0.0
    {
        return Err(CausalError::not_identified(
            envelope.status,
            envelope.truncated_completions > 0,
            "temporal class query not identified (no identified mass in envelope)",
        ));
    }
    Ok(())
}

/// Every identified completion's recorded regressions and point.
fn collect_class_points<'a>(
    bundle: &'a antecedent_identify::TemporalClassEnvelope,
    mut fit: impl FnMut(
        &'a antecedent_identify::GraphIdentificationCase<
            antecedent_identify::TemporalCompletionGraph,
        >,
        &'a TemporalIndexer,
    ) -> Result<(SlopeAtom<'a>, f64), CausalError>,
) -> Result<ClassPoints<'a>, CausalError> {
    let envelope = &bundle.envelope;
    require_identified_mass(envelope)?;
    let mut atoms = Vec::new();
    let mut weights = Vec::new();
    let mut points = Vec::new();
    let mut values = vec![None; envelope.cases.len()];
    let mut primary = None;
    let mut skipped = SkippedCompletions::default();
    for (index, (case, indexer)) in envelope.cases.iter().zip(bundle.indexers.iter()).enumerate() {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let (atom, point) = match fit(case, indexer) {
            Ok(fitted) => fitted,
            // A completion whose own regression cannot be fit is unevaluable mass, not a
            // reason to lose the other completions; a request error or a stop of the run
            // ends the run.
            Err(error) => {
                skipped.record(error)?;
                continue;
            }
        };
        atoms.push(atom);
        weights.push(case.weight.0);
        points.push(point);
        values[index] = Some(point);
        if primary.is_none() {
            primary = Some((case.result.clone(), indexer.clone()));
        }
    }
    let (primary_identification, primary_indexer) =
        primary.ok_or_else(|| skipped.none_estimable())?;
    let estimand = primary_identification.estimands.first().cloned().ok_or_else(|| {
        CausalError::Compile { message: "temporal class completion has no estimand".into() }
    })?;
    Ok(ClassPoints {
        atoms,
        skipped,
        weights,
        points,
        values,
        estimand,
        assumptions: primary_identification.required_assumptions.clone(),
        primary_identification,
        primary_indexer,
    })
}

/// Whether `error` is a numerical failure of one completion's own regression (a singular
/// design, or too few intact aligned rows for its lags). Everything else (a request error such
/// as an unknown variable or an unsupported option, cancellation, a resource or admission
/// limit) holds for every completion or ends the run, and is not skipped.
fn completion_cannot_be_fit(error: &CausalError) -> bool {
    use antecedent_data::DataError;
    use antecedent_estimate::EstimationError;
    use antecedent_stats::StatsError;
    matches!(
        error,
        CausalError::Estimate(
            EstimationError::Stats(
                StatsError::RankDeficient { .. } | StatsError::NonPositiveVariance { .. }
            ) | EstimationError::Data(
                DataError::IncompleteSeries { .. }
                    | DataError::EmptySelection { .. }
                    | DataError::InvalidArgument { .. }
            )
        )
    )
}

/// The structural lag span of a class envelope's completions.
fn envelope_span(bundle: &antecedent_identify::TemporalClassEnvelope) -> usize {
    super::temporal_class_block_span(bundle.indexers.iter())
}

/// The uncertainty of a scalar frozen-weight point over `atoms`.
///
/// Frequentist: the shared circular-block replicates of the whole procedure give
/// the point's SE and, over several completions, an Imbens–Manski interval for the
/// identified set. Bayesian: one completion publishes its posterior; several
/// publish an interval for the identified set over the completion posteriors.
#[allow(
    clippy::too_many_arguments,
    reason = "the frozen atoms, their weights and points, the published point and its assumptions, the cap flag, the reporting plan and the context are distinct inputs"
)]
#[allow(
    clippy::too_many_lines,
    reason = "one completion-mixture uncertainty: the sampling and Bayesian branches publish over the same frozen atoms"
)]
pub(super) fn class_scalar_uncertainty(
    atoms: &[FrozenAtom],
    weights: &[f64],
    points: &[f64],
    point: f64,
    assumptions: antecedent_core::AssumptionSet,
    truncated: bool,
    plan: ClassPointUncertainty<'_>,
    ctx: &ExecutionContext,
    evaluate: impl Fn(usize, &dyn SlopeSource) -> Result<Vec<f64>, CausalError>,
) -> Result<ClassScalarOutcome, CausalError> {
    let family = antecedent_estimate::CircularBlockFamily::Mixture;
    if let InferenceMode::Bayesian(cfg) = plan.inference {
        let draws = point_uncertainty::point_posterior_draws(
            atoms,
            point_uncertainty::PointPosteriorPlan {
                prior_scale: cfg.prior_scale,
                draws: cfg.n_draws,
                structural_span: plan.structural_span,
                stream_base: plan.posterior_stream,
            },
            ctx,
            evaluate,
        )?;
        let mut diagnostics =
            vec![point_uncertainty::posterior_diagnostic(&draws, "temporal class point")];
        let columns: Vec<Vec<f64>> =
            draws.atom_draws.iter().map(|atom| atom.iter().map(|draw| draw[0]).collect()).collect();
        if columns.len() == 1 {
            let posterior = point_uncertainty::scalar_posterior(
                &columns[0],
                "class_point",
                IdentificationStatus::NonparametricallyIdentified,
                assumptions.clone(),
                draws.tempering,
            )?;
            let sd = posterior.summaries.sd[0];
            return Ok(ClassScalarOutcome {
                estimate: EffectEstimate::new(
                    point,
                    sd,
                    assumptions,
                    OverlapPolicy::ExplicitOverride,
                ),
                posterior: Some(posterior),
                identified_set: None,
                diagnostics,
                replicates_ok: None,
                cancelled: false,
            });
        }
        let means: Vec<f64> =
            columns.iter().map(|column| column.iter().sum::<f64>() / column.len() as f64).collect();
        let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        let rows = atoms.iter().map(FrozenAtom::rows).min().unwrap_or(0);
        let identified_set = antecedent_estimate::imbens_manski_posterior_draws(
            &means,
            &refs,
            rows,
            super::IDENTIFIED_SET_INTERVAL_LEVEL,
        )
        .map(|interval| interval.with_truncated(truncated));
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_class.point_posteriors_not_mixed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "completion enumeration is not posterior mass, so the completion posteriors are not \
             mixed; the published interval is for the identified set over the completions",
        ));
        diagnostics.extend(
            identified_set
                .as_ref()
                .into_iter()
                .flat_map(super::identified_set_interval_diagnostics),
        );
        return Ok(ClassScalarOutcome {
            estimate: EffectEstimate::new(
                point,
                f64::NAN,
                assumptions,
                OverlapPolicy::ExplicitOverride,
            ),
            posterior: None,
            identified_set,
            diagnostics,
            replicates_ok: None,
            cancelled: false,
        });
    }
    let replicates = point_uncertainty::point_block_replicates(
        atoms,
        point_uncertainty::PointBlockPlan {
            replicates: plan.bootstrap_replicates,
            stream_base: plan.block_stream,
            structural_span: plan.structural_span,
        },
        true,
        ctx,
        evaluate,
        |values| point_uncertainty::weighted_outputs(values, weights),
    )?;
    let identified_set = replicates
        .as_ref()
        .filter(|_| atoms.len() > 1)
        .and_then(|replicates| {
            replicates
                .shared_block()
                .identified_set_interval(points, super::IDENTIFIED_SET_INTERVAL_LEVEL)
        })
        .map(|interval| interval.with_truncated(truncated));
    let mut diagnostics = point_uncertainty::block_diagnostics(
        replicates.as_ref(),
        plan.bootstrap_replicates,
        family,
        "every regression of every identified completion is refit on each replicate and the \
         frozen completion weights are reapplied",
    );
    diagnostics.extend(
        identified_set.as_ref().into_iter().flat_map(super::identified_set_interval_diagnostics),
    );
    Ok(ClassScalarOutcome {
        estimate: point_uncertainty::block_scalar_estimate(
            point,
            replicates.as_ref(),
            assumptions,
            family,
        ),
        posterior: None,
        identified_set,
        diagnostics,
        replicates_ok: replicates.as_ref().map(point_uncertainty::PointBlockReplicates::completed),
        cancelled: replicates.as_ref().is_some_and(|replicates| replicates.draws.cancelled),
    })
}

/// The uncertainty of a frozen-weight class derivative response over `atoms`,
/// attached to `response` (whose point is `center`).
fn class_response_uncertainty(
    atoms: &[FrozenAtom],
    weights: &[f64],
    (response, center): (&mut CausalResponse, &[f64]),
    plan: ClassPointUncertainty<'_>,
    ctx: &ExecutionContext,
    evaluate: impl Fn(usize, &dyn SlopeSource) -> Result<Vec<f64>, CausalError>,
) -> Result<(Vec<Diagnostic>, Option<u32>, bool), CausalError> {
    if let InferenceMode::Bayesian(cfg) = plan.inference {
        let draws = point_uncertainty::point_posterior_draws(
            atoms,
            point_uncertainty::PointPosteriorPlan {
                prior_scale: cfg.prior_scale,
                draws: cfg.n_draws,
                structural_span: plan.structural_span,
                stream_base: plan.posterior_stream,
            },
            ctx,
            evaluate,
        )?;
        let mut diagnostics =
            vec![point_uncertainty::posterior_diagnostic(&draws, "temporal class derivative")];
        if draws.atom_draws.len() == 1 {
            point_uncertainty::attach_posterior_response_band(
                response,
                &draws.atom_draws[0],
                center,
            );
        } else {
            let level = crate::result::REPORTED_SE_INTERVAL_LEVEL;
            let width = center.len();
            let mut lower = vec![f64::INFINITY; width];
            let mut upper = vec![f64::NEG_INFINITY; width];
            for atom in &draws.atom_draws {
                for j in 0..width {
                    let column: Vec<f64> = atom.iter().map(|draw| draw[j]).collect();
                    let (_, _, low, high) = point_uncertainty::draw_summary(&column, level);
                    lower[j] = lower[j].min(low);
                    upper[j] = upper[j].max(high);
                }
            }
            response.uncertainty = ResponseUncertainty::IdentifiedEnvelopeBand {
                level,
                lower_outer: Arc::from(lower),
                upper_outer: Arc::from(upper),
                interpretation: antecedent_core::IntervalInterpretation::Credible,
            };
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_class.point_posteriors_not_mixed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "completion enumeration is not posterior mass, so the completion posteriors are \
                 not mixed; the band is the union of the completions' equal-tailed credible \
                 intervals, an envelope for the identified set",
            ));
        }
        return Ok((diagnostics, None, false));
    }
    let replicates = point_uncertainty::point_block_replicates(
        atoms,
        point_uncertainty::PointBlockPlan {
            replicates: plan.bootstrap_replicates,
            stream_base: plan.block_stream,
            structural_span: plan.structural_span,
        },
        false,
        ctx,
        evaluate,
        |values| point_uncertainty::weighted_outputs(values, weights),
    )?;
    if let Some(replicates) = &replicates {
        point_uncertainty::attach_block_response_band(response, replicates, center);
    }
    let diagnostics = point_uncertainty::block_diagnostics(
        replicates.as_ref(),
        plan.bootstrap_replicates,
        antecedent_estimate::CircularBlockFamily::Mixture,
        "every outcome regression of every identified completion is refit on each replicate and \
         the frozen completion weights are reapplied",
    );
    Ok((
        diagnostics,
        replicates.as_ref().map(point_uncertainty::PointBlockReplicates::completed),
        replicates.as_ref().is_some_and(|replicates| replicates.draws.cancelled),
    ))
}

/// Every identified completion's conditional design (over the modifier set) and point.
fn collect_class_conditional<'a>(
    data: &'a TimeSeriesData,
    query: &TemporalEffectQuery,
    bundle: &'a antecedent_identify::TemporalClassEnvelope,
    policy: antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<ClassPoints<'a>, CausalError> {
    refuse_unread_series_weights(data, "the class-aware conditional effect")?;
    collect_class_points(bundle, |case, indexer| {
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let atom =
            SlopeAtom::conditional(data, &estimand, query, indexer, policy, prior_scale, false)?;
        let point = atom.conditional_effect()?;
        Ok((atom, point))
    })
}

/// The conditional point of one class envelope with its uncertainty.
pub(super) fn assemble_temporal_class_conditional(
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    bundle: &antecedent_identify::TemporalClassEnvelope,
    graph_class: GraphClass,
    refute: RefuteSuite,
    (inference, bootstrap_replicates): (&InferenceMode, u32),
    ctx: &ExecutionContext,
) -> Result<ClassConditionalFit, CausalError> {
    if query.effect_modifiers.is_empty()
        || query.is_multi_step_sustained()
        || matches!(query.policy, antecedent_core::TemporalPolicy::Dynamic { .. })
    {
        return Err(CausalError::Unsupported {
            message: "temporal conditional effect fits a modifier set on a single-step pulse or sustained policy",
        });
    }
    if refute != RefuteSuite::None {
        return Err(CausalError::Unsupported {
            message: "temporal conditional effect is point-only; cheap/full refuters are not supported",
        });
    }
    let prior_scale = isotropic_prior_scale(inference, CONDITIONAL_PRIOR_REFUSAL)?;
    let fitted = collect_class_conditional(data, query, bundle, ctx.kernel_policy, prior_scale)?;
    let point = fitted.point()?;
    let envelope = &bundle.envelope;
    let mut diagnostics =
        vec![super::temporal_path::temporal_class_envelope_diagnostic(envelope, graph_class)];
    diagnostics.push(Diagnostic::new(
        "estimate.temporal.conditional_marginal",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "published value is the completion-weighted marginal effect over the modifier set; its \
         interval refits every completion's interaction regression and modifier means",
    ));
    diagnostics.extend(super::temporal_effect_core::modifier_set_coverage_note(
        query.effect_modifiers.len(),
    ));
    diagnostics.extend(fitted.skipped.diagnostic());
    let mut structural = super::temporal_path::temporal_class_structural_mixture(
        envelope,
        crate::result::StructuralWeightBasis::CompletionEnumeration,
        None,
        &fitted.values,
    );
    let frozen: Vec<FrozenAtom> = fitted.atoms.into_iter().map(SlopeAtom::freeze).collect();
    let outcome = class_scalar_uncertainty(
        &frozen,
        &fitted.weights,
        &fitted.points,
        point,
        fitted.assumptions,
        envelope.truncated_completions > 0,
        ClassPointUncertainty {
            inference,
            bootstrap_replicates,
            structural_span: envelope_span(bundle),
            block_stream: point_uncertainty::CONDITIONAL_BLOCK_STREAM,
            posterior_stream: point_uncertainty::CONDITIONAL_POSTERIOR_STREAM,
        },
        ctx,
        |_, source| source.conditional_effect().map(|value| vec![value]),
    )?;
    structural.identified_set_interval = outcome.identified_set;
    diagnostics.extend(outcome.diagnostics.iter().cloned());
    Ok(ClassConditionalFit { estimand: fitted.estimand, diagnostics, structural, outcome })
}

impl super::Study {
    /// The class envelope a derivative response reads: the prepared handle's
    /// certificate for the probe's horizon (or its default), then the one the
    /// plan's lowered program retained, else a fresh identification.
    fn derivative_class_envelope(
        &self,
        identifier_id: IdentifierId,
        probe: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<antecedent_identify::TemporalClassEnvelope>, bool), CausalError> {
        Ok(if let Some((cache, prepared)) = self.reuse_temporal_class_identification(physical) {
            if let Some((_, envelope)) =
                cache.by_horizon.iter().find(|(horizon, _)| *horizon == probe.horizon_steps)
            {
                (Arc::clone(envelope), prepared)
            } else {
                (Arc::clone(&cache.envelope), prepared)
            }
        } else {
            report_identify_compute(ctx);
            (
                self.identify_temporal_class_within(identifier_id, probe, budget, ctx)?
                    .envelope
                    .clone(),
                false,
            )
        })
    }

    /// The uncertainty plan of a class point route of this study.
    fn class_point_plan(
        &self,
        structural_span: usize,
        (block_stream, posterior_stream): (u64, u64),
    ) -> ClassPointUncertainty<'_> {
        ClassPointUncertainty {
            inference: &self.settings.inference,
            bootstrap_replicates: self.settings.bootstrap_replicates,
            structural_span,
            block_stream,
            posterior_stream,
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one class derivative execution: the completions, the mixture and its band stay together"
    )]
    pub(crate) fn execute_temporal_class_derivative_within(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.require_supported_temporal_observation()?;
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware temporal derivative requires TemporalResponseSpec".into(),
        })?;
        if temporal.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon",
            });
        }
        let prior_scale =
            isotropic_prior_scale(&self.settings.inference, DERIVATIVE_PRIOR_REFUSAL)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let probe = super::temporal_path::unit_contrast_query(
            temporal,
            (treatment, outcome),
            temporal.horizons[0],
            query.target_population.clone(),
        );
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let (bundle, identify_cached) =
            self.derivative_class_envelope(identifier_id, &probe, physical, budget, ctx)?;
        let offsets = (probe.try_treatment_offset()?, probe.outcome_offset()?);
        let envelope = &bundle.envelope;
        let ClassDerivatives { atoms, responses, values, estimand, skipped } =
            class_derivative_responses(
                data,
                query,
                &bundle,
                offsets,
                (ctx.kernel_policy, prior_scale),
            )?;
        let mut response = mix_derivative_responses(&responses)?;
        let center = response_center(&response)?;
        let weights: Vec<f64> = responses.iter().map(|(weight, _)| *weight).collect();
        let frozen: Vec<FrozenAtom> = atoms.into_iter().map(SlopeAtom::freeze).collect();
        let (uncertainty_diagnostics, replicates_ok, cancelled) = class_response_uncertainty(
            &frozen,
            &weights,
            (&mut response, &center),
            self.class_point_plan(
                envelope_span(&bundle),
                (
                    point_uncertainty::DERIVATIVE_BLOCK_STREAM,
                    point_uncertainty::DERIVATIVE_POSTERIOR_STREAM,
                ),
            ),
            ctx,
            derivative_evaluator(&query.functional),
        )?;
        let mut diagnostics =
            class_derivative_diagnostics(envelope, self.graph.class(), identify_cached, &response);
        diagnostics.extend(skipped.diagnostic());
        diagnostics.extend(uncertainty_diagnostics);
        // A derivative's cheap/full suite is its refuter state on every identified
        // completion's Pulse witness at the one horizon (unsupported).
        let (refutations, predictive_checks) = if self.settings.refute == RefuteSuite::None {
            (Vec::new(), Vec::new())
        } else {
            let mut sites = Vec::new();
            for (case_index, (case, indexer)) in
                envelope.cases.iter().zip(bundle.indexers.iter()).enumerate()
            {
                if !super::identification_status_ok_for_case(case.result.status)
                    || case.result.estimands.is_empty()
                {
                    continue;
                }
                let estimand =
                    super::select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
                sites.push(super::temporal_response_refutation::PulseWitnessSite {
                    key: super::temporal_path::class_completion_key(0, case_index),
                    horizon: temporal.horizons[0],
                    weight: case.weight.0,
                    witness: probe.clone(),
                    estimand,
                    indexer: indexer.clone(),
                    status: case.result.status,
                    assumptions: case.result.required_assumptions.clone(),
                });
            }
            let (reports, checks, notes) =
                super::temporal_response_refutation::refute_temporal_derivative(
                    self, data, query, &response, &sites, ctx,
                )?;
            diagnostics.extend(notes);
            (reports, checks)
        };
        let scalar = super::response_path::response_scalar_summary(&response).0;
        let estimate = EffectEstimate::new(
            scalar,
            f64::NAN,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let identification =
            envelope_to_identification_result_for(envelope, CausalQuery::Response(query.clone()));
        let estimator_id = EstimatorId::temporal_response_for(&self.settings.inference);
        let structural = derivative_structural_mixture(envelope, &values);
        response.identification_status = identification.status;
        retain_graph_dependent_derivatives(&mut response, &structural);
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(replicates_ok)
            .cancelled(cancelled)
            .extras(IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: Arc::unwrap_or_clone(bundle),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                response: Some(response),
                predictive_checks,
                ..Default::default()
            }),
        );
        result.posterior = None;
        result.rebind_interval(self.settings.inference.is_bayesian());
        Ok(result)
    }
}

/// The mass every positive-weight graph-posterior atom carries, kept by kind.
///
/// Each atom contributes its own completion masses (identified and evaluated, unidentified,
/// identified but unevaluable, subsampled out) scaled by its posterior weight, and the graphs
/// the posterior proved unidentified add their mass beside them, so the four masses account for
/// all posterior mass. An atom with no completions contributes its whole weight as unidentified
/// mass. The status is the most conservative status of any atom.
#[derive(Default)]
struct PosteriorAtomMasses {
    weight: f64,
    identified: f64,
    unidentified: f64,
    subsampled: f64,
    atoms: Vec<crate::result::StructuralResponseAtom>,
    truncated_atoms: usize,
    partial_scope: bool,
    statuses: Vec<IdentificationStatus>,
    skipped: SkippedCompletions,
}

impl PosteriorAtomMasses {
    fn add(
        &mut self,
        weight: f64,
        status: IdentificationStatus,
        mixture: crate::result::StructuralResponseMixture,
    ) {
        let carried = mixture.identified_mass
            + mixture.unidentified_mass
            + mixture.unevaluable_mass
            + mixture.subsampled_out_mass;
        self.weight += weight;
        if carried > 0.0 {
            self.identified += weight * mixture.identified_mass;
            self.unidentified += weight * mixture.unidentified_mass;
            self.subsampled += weight * mixture.subsampled_out_mass;
        } else {
            self.unidentified += weight;
        }
        self.truncated_atoms += mixture.truncated_atoms;
        self.partial_scope |= !mixture.full_mass_scope;
        self.statuses.push(status);
        self.atoms.extend(mixture.atoms);
    }

    /// The posterior-level mixture, its most conservative atom status and the mass summary the
    /// status ladder reads, with `graph_unidentified` the mass of graphs proved unidentified.
    fn finish(
        self,
        graph_unidentified: f64,
    ) -> (crate::result::StructuralResponseMixture, IdentificationStatus, PosteriorMassSummary)
    {
        let total = self.weight + graph_unidentified;
        let (identified, unidentified, subsampled, unevaluable) = if total > 0.0 {
            let identified = self.identified / total;
            let unidentified = (self.unidentified + graph_unidentified) / total;
            let subsampled = self.subsampled / total;
            // The remainder, so the four masses conserve exactly under division.
            let unevaluable = (1.0 - identified - unidentified - subsampled).max(0.0);
            (identified, unidentified, subsampled, unevaluable)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for atom in &self.atoms {
            if let Some(ResponseValue::Scalar(value)) = &atom.value {
                lo = lo.min(*value);
                hi = hi.max(*value);
            }
        }
        let status = super::most_conservative_identification_status(self.statuses);
        let mixture = crate::result::StructuralResponseMixture {
            weight_basis: crate::result::StructuralWeightBasis::CompletionEnumeration,
            atoms: self.atoms,
            identified_mass: identified,
            unidentified_mass: unidentified,
            unevaluable_mass: unevaluable,
            subsampled_out_mass: subsampled,
            identified_set: (lo.is_finite() && hi.is_finite())
                .then(|| super::scalar_identified_set(lo, hi)),
            identified_set_interval: None,
            conditional_on_identified: None,
            full_mass_scope: !self.partial_scope,
            truncated_atoms: self.truncated_atoms,
            aggregation_policy: None,
        };
        let summary = PosteriorMassSummary {
            unidentified,
            unevaluable,
            subsampled,
            ..PosteriorMassSummary::default()
        };
        (mixture, status, summary)
    }
}

/// Every identified completion of every graph-posterior atom, frozen for one
/// shared uncertainty procedure: weight = atom posterior weight times the
/// completion's normalized enumeration weight.
#[derive(Default)]
struct PooledAtoms {
    atoms: Vec<FrozenAtom>,
    weights: Vec<f64>,
    points: Vec<f64>,
    truncated: bool,
    spans: Vec<usize>,
}

impl PooledAtoms {
    fn push(&mut self, fitted: ClassPoints<'_>, weight: f64, truncated: bool) {
        let total: f64 = fitted.weights.iter().sum();
        if total.is_nan() || total <= 0.0 {
            return;
        }
        self.truncated |= truncated;
        self.spans.push(super::temporal_class_block_span(std::iter::once(&fitted.primary_indexer)));
        for ((atom, completion_weight), point) in
            fitted.atoms.into_iter().zip(fitted.weights.iter()).zip(fitted.points.iter())
        {
            self.atoms.push(atom.freeze());
            self.weights.push(weight * completion_weight / total);
            self.points.push(*point);
        }
    }
}

/// The lag-1 pulse probe a cross-world contrast identifies on.
/// The one-step pulse probe a temporal cross-world query identifies on, when the query names
/// a contrast this route estimates. It depends on the query alone, so compile can identify it.
pub(super) fn cross_world_identification_probe(query: &CausalQuery) -> Option<TemporalEffectQuery> {
    let (treatment, outcome, ..) = cross_world_endpoints(query).ok()?;
    Some(cross_world_probe(treatment, outcome))
}

pub(super) fn cross_world_probe(treatment: VariableId, outcome: VariableId) -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(treatment, outcome, 1.0)
        .with_policy(antecedent_core::TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

fn mix_derivative_responses(
    responses: &[(f64, CausalResponse)],
) -> Result<CausalResponse, CausalError> {
    let (_, first) = responses.first().ok_or_else(|| CausalError::Compile {
        message: "temporal class derivative had no completion response".into(),
    })?;
    let mut mixed = first.clone();
    let value = match &first.estimate {
        ResponseIdentification::PointIdentified(value) => mix_response_value(responses, value)?,
        _ => {
            return Err(CausalError::Compile {
                message: "temporal class derivative completion is not point identified".into(),
            });
        }
    };
    mixed.estimate = ResponseIdentification::PointIdentified(value);
    mixed.uncertainty = ResponseUncertainty::None;
    if responses.iter().any(|(_, response)| response.support.status != SupportStatus::Supported) {
        mixed.support.status = SupportStatus::OutsideEmpiricalSupport;
        for (_, response) in responses.iter().skip(1) {
            mixed.support.warnings.extend(response.support.warnings.iter().cloned());
        }
    }
    Ok(mixed)
}

fn mix_response_value(
    responses: &[(f64, CausalResponse)],
    prototype: &ResponseValue,
) -> Result<ResponseValue, CausalError> {
    match prototype {
        ResponseValue::Scalar(_) => {
            let pairs = scalar_pairs(responses)?;
            Ok(ResponseValue::Scalar(weighted_mean(&pairs)?))
        }
        ResponseValue::Vector(values) => {
            let width = values.len();
            let mut mixed = vec![0.0; width];
            let mut total = 0.0;
            for (weight, response) in responses {
                let ResponseIdentification::PointIdentified(ResponseValue::Vector(row)) =
                    &response.estimate
                else {
                    return Err(shape_error());
                };
                if row.len() != width {
                    return Err(shape_error());
                }
                for (slot, value) in mixed.iter_mut().zip(row.iter()) {
                    *slot += weight * value;
                }
                total += weight;
            }
            if total <= 0.0 {
                return Err(shape_error());
            }
            for slot in &mut mixed {
                *slot /= total;
            }
            Ok(ResponseValue::Vector(std::sync::Arc::from(mixed)))
        }
        ResponseValue::Jacobian { outcomes, treatments, values } => {
            let width = values.len();
            let mut mixed = vec![0.0; width];
            let mut total = 0.0;
            for (weight, response) in responses {
                let ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
                    outcomes: rows,
                    treatments: cols,
                    values: row,
                }) = &response.estimate
                else {
                    return Err(shape_error());
                };
                if *rows != *outcomes || *cols != *treatments || row.len() != width {
                    return Err(shape_error());
                }
                for (slot, value) in mixed.iter_mut().zip(row.iter()) {
                    *slot += weight * value;
                }
                total += weight;
            }
            if total <= 0.0 {
                return Err(shape_error());
            }
            for slot in &mut mixed {
                *slot /= total;
            }
            Ok(ResponseValue::Jacobian {
                outcomes: *outcomes,
                treatments: *treatments,
                values: std::sync::Arc::from(mixed),
            })
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal linear derivative does not estimate this functional",
        }),
    }
}

fn scalar_pairs(responses: &[(f64, CausalResponse)]) -> Result<Vec<(f64, f64)>, CausalError> {
    responses
        .iter()
        .map(|(weight, response)| match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(point)) => {
                Ok((*weight, *point))
            }
            _ => Err(shape_error()),
        })
        .collect()
}

fn shape_error() -> CausalError {
    CausalError::Compile {
        message: "temporal class completions disagree on the derivative shape".into(),
    }
}

pub(super) fn cross_world_endpoints(
    query: &CausalQuery,
) -> Result<(VariableId, VariableId, Option<VariableId>, f64, f64), CausalError> {
    match query {
        CausalQuery::NestedCounterfactual(q) => {
            Ok((q.treatment, q.outcome, Some(q.mediator), q.active_value(), q.control_value()))
        }
        CausalQuery::PathSpecific(q) => {
            let first = super::temporal_path::require_path_nodes(&q.path_nodes)?;
            Ok((
                q.treatment,
                q.outcome,
                Some(first),
                super::temporal_path::temporal_cross_world_hard_level(&q.active)?,
                super::temporal_path::temporal_cross_world_hard_level(&q.control)?,
            ))
        }
        CausalQuery::Counterfactual(q) => {
            if q.outcomes.len() != 1 {
                return Err(CausalError::Unsupported {
                    message: "temporal counterfactual needs one outcome",
                });
            }
            if q.interventions.len() != 1 || !q.joint_control.is_empty() {
                return Err(CausalError::Unsupported {
                    message: "temporal counterfactual needs one hard intervention in each world",
                });
            }
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).ok_or(
                    CausalError::Unsupported {
                        message: "temporal counterfactual needs one hard intervention",
                    },
                )?;
            let outcome = *q.outcomes.first().ok_or(CausalError::Unsupported {
                message: "temporal counterfactual needs one outcome",
            })?;
            Ok((
                treatment,
                outcome,
                None,
                super::temporal_path::temporal_cross_world_hard_level(&q.interventions[0])?,
                super::temporal_path::temporal_cross_world_hard_level(&q.control)?,
            ))
        }
        CausalQuery::Distribution(q) => {
            if q.outcomes.len() != 1 {
                return Err(CausalError::Unsupported {
                    message: "temporal interventional distribution needs one outcome",
                });
            }
            if q.interventions.len() != 1 {
                return Err(CausalError::Unsupported {
                    message: "temporal interventional distribution needs one hard intervention",
                });
            }
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).ok_or(
                    CausalError::Unsupported {
                        message: "temporal interventional distribution needs one hard intervention",
                    },
                )?;
            let outcome = *q.outcomes.first().ok_or(CausalError::Unsupported {
                message: "temporal interventional distribution needs one outcome",
            })?;
            if !q.conditioning.is_empty() {
                return Err(CausalError::Unsupported {
                    message: "temporal interventional distribution is unconditional",
                });
            }
            Ok((
                treatment,
                outcome,
                None,
                super::temporal_path::temporal_cross_world_hard_level(&q.interventions[0])?,
                0.0,
            ))
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal cross-world route does not estimate this query",
        }),
    }
}

pub(super) fn temporal_class_point_family(query: &CausalQuery) -> bool {
    match query {
        CausalQuery::TemporalEffect(effect) => !effect.effect_modifiers.is_empty(),
        CausalQuery::Response(response) => {
            response.temporal.is_some()
                && matches!(
                    response.functional,
                    ResponseFunctional::AverageDerivative { .. }
                        | ResponseFunctional::PointDerivative { .. }
                        | ResponseFunctional::DirectionalDerivative { .. }
                        | ResponseFunctional::Jacobian { .. }
                )
        }
        CausalQuery::NestedCounterfactual(_)
        | CausalQuery::PathSpecific(_)
        | CausalQuery::Counterfactual(_)
        | CausalQuery::Distribution(_) => true,
        _ => false,
    }
}

pub(super) fn cross_world_plan_variables(
    query: &CausalQuery,
) -> Result<(VariableId, VariableId), CausalError> {
    let (treatment, outcome, _, _, _) = cross_world_endpoints(query)?;
    Ok((treatment, outcome))
}

fn load_class_posterior_cache(
    study: &super::Study,
    gp: &GraphPosterior,
    data: &TimeSeriesData,
    probe: &TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> Result<
    (Arc<crate::analysis::prepared::CachedTemporalClassPosteriorIdentification>, bool),
    CausalError,
> {
    if let Some(cache) = study.temporal_class_posterior_cache() {
        return Ok((Arc::clone(cache), true));
    }
    let variables =
        data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
    let cache = crate::analysis::prepared::build_temporal_class_posterior_identification_cache(
        gp,
        &variables,
        probe,
        study.declared.max_completions,
        ctx,
    )?;
    Ok((Arc::new(cache), false))
}

#[cfg(test)]
mod posterior_mass_tests {
    use super::*;

    #[test]
    fn vector_completion_mass_and_order_preserve_unidentified_and_unevaluable_cases() {
        use antecedent_identify::{
            GraphIdentificationCase, ProbabilityMass, TemporalCompletionGraph,
        };

        let query = CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            1.0,
        ));
        let case = |index: u32, identified: bool| {
            let mut graph = antecedent_graph::TemporalDag::empty();
            antecedent_graph::ensure_lagged(
                &mut graph,
                VariableId::from_raw(index),
                antecedent_core::Lag::CONTEMPORANEOUS,
            )
            .unwrap();
            let result = if identified {
                IdentificationResult::identified(
                    query.clone(),
                    Vec::new(),
                    antecedent_expr::CausalExprArena::new(),
                    antecedent_identify::DerivationTrace::default(),
                    antecedent_core::AssumptionSet::default(),
                    antecedent_identify::IdentificationPerformanceRecord::default(),
                )
            } else {
                IdentificationResult::not_identified(
                    query.clone(),
                    antecedent_identify::DerivationTrace::default(),
                    antecedent_core::AssumptionSet::default(),
                    antecedent_identify::IdentificationPerformanceRecord::default(),
                )
            };
            GraphIdentificationCase {
                graph: TemporalCompletionGraph::Dag(graph),
                result,
                weight: ProbabilityMass(0.25),
            }
        };
        let mut envelope = antecedent_identify::IdentificationEnvelope::from_cases(vec![
            case(0, true),
            case(1, false),
            case(2, true),
            case(3, true),
        ]);
        envelope.truncated_completions = 1;
        let first = ResponseValue::Vector(Arc::from([1.0, 2.0]));
        let last = ResponseValue::Vector(Arc::from([3.0, 4.0]));
        let values = [Some(first.clone()), None, None, Some(last.clone())];
        let structural = derivative_structural_mixture(&envelope, &values);
        assert!((structural.identified_mass - 0.5).abs() < 1e-12);
        assert!((structural.unidentified_mass - 0.25).abs() < 1e-12);
        assert!((structural.unevaluable_mass - 0.25).abs() < 1e-12);
        assert!(!structural.full_mass_scope);
        assert_eq!(structural.atoms.len(), envelope.cases.len());
        for (index, atom) in structural.atoms.iter().enumerate() {
            assert_eq!(atom.graph_key, envelope.cases[index].graph.fingerprint());
            assert_eq!(atom.value, values[index]);
        }
        assert!(structural.identified_set.is_none());

        let mut response = CausalResponse {
            estimand: ResponseFunctional::DirectionalDerivative {
                outcomes: Arc::from([VariableId::from_raw(1)]),
                treatments: Arc::from([VariableId::from_raw(0)]),
                at: Arc::from([0.0]),
                direction: Arc::from([1.0]),
            },
            identification_status: IdentificationStatus::GraphDependent,
            estimate: ResponseIdentification::PointIdentified(first.clone()),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from([0.0, 1.0]),
                upper: Arc::from([2.0, 3.0]),
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            },
            support: antecedent_core::SupportReport {
                status: SupportStatus::Supported,
                query_region: antecedent_core::SupportRegion {
                    minima: Arc::from([0.0]),
                    maxima: Arc::from([0.0]),
                },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
                point_status: None,
            },
            assumptions: antecedent_core::AssumptionSet::default(),
            provenance_id: Arc::from("test"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        };
        retain_graph_dependent_derivatives(&mut response, &structural);
        assert_eq!(
            response.estimate,
            ResponseIdentification::GraphDependent(vec![
                (structural.atoms[0].graph_key, first),
                (structural.atoms[3].graph_key, last),
            ])
        );
        assert!(matches!(response.uncertainty, ResponseUncertainty::None));
    }

    fn mixture(
        values: &[Option<f64>],
        masses: [f64; 3],
    ) -> crate::result::StructuralResponseMixture {
        let [identified, unidentified, unevaluable] = masses;
        crate::result::StructuralResponseMixture {
            weight_basis: crate::result::StructuralWeightBasis::CompletionEnumeration,
            atoms: values
                .iter()
                .enumerate()
                .map(|(index, value)| crate::result::StructuralResponseAtom {
                    graph_key: index as u64,
                    weight: 1.0 / values.len() as f64,
                    status: IdentificationStatus::NonparametricallyIdentified,
                    value: value.map(ResponseValue::Scalar),
                    posterior: None,
                    response: None,
                })
                .collect(),
            identified_mass: identified,
            unidentified_mass: unidentified,
            unevaluable_mass: unevaluable,
            subsampled_out_mass: 0.0,
            identified_set: None,
            identified_set_interval: None,
            conditional_on_identified: None,
            full_mass_scope: true,
            truncated_atoms: 0,
            aggregation_policy: None,
        }
    }

    #[test]
    fn every_atoms_completion_mass_reaches_the_published_mixture_and_status() {
        let identified = IdentificationStatus::NonparametricallyIdentified;
        let mut masses = PosteriorAtomMasses::default();
        masses.add(0.5, identified, mixture(&[Some(1.0); 4], [1.0, 0.0, 0.0]));
        masses.add(0.5, identified, mixture(&[Some(3.0), None, None, None], [0.25, 0.75, 0.0]));
        let (published, status, summary) = masses.finish(0.0);
        assert!((published.identified_mass - 0.625).abs() < 1e-12);
        assert!((published.unidentified_mass - 0.375).abs() < 1e-12);
        assert_eq!(published.atoms.len(), 8);
        let set = published.identified_set.expect("the range over every atom's completions");
        assert_eq!((set.lower[0], set.upper[0]), (1.0, 3.0));
        assert_eq!(
            resolve_posterior_status(
                status,
                crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
                summary,
            ),
            IdentificationStatus::GraphDependent
        );
    }

    #[test]
    fn zero_completion_atoms_and_unidentified_graphs_conserve_the_posterior_mass() {
        let identified = IdentificationStatus::NonparametricallyIdentified;
        let mut masses = PosteriorAtomMasses::default();
        masses.add(0.4, identified, mixture(&[Some(1.0)], [1.0, 0.0, 0.0]));
        masses.add(0.3, identified, mixture(&[], [0.0, 0.0, 0.0]));
        masses.add(0.1, identified, mixture(&[None], [0.0, 0.0, 1.0]));
        let (published, _, _) = masses.finish(0.2);
        assert!((published.identified_mass - 0.4).abs() < 1e-12);
        assert!((published.unidentified_mass - 0.5).abs() < 1e-12);
        assert!((published.unevaluable_mass - 0.1).abs() < 1e-12);
        let sum = published.identified_mass
            + published.unidentified_mass
            + published.unevaluable_mass
            + published.subsampled_out_mass;
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn only_a_numerical_failure_of_one_completion_is_skipped() {
        use antecedent_data::DataError;
        use antecedent_estimate::EstimationError;
        use antecedent_stats::StatsError;
        let singular = CausalError::from(EstimationError::Stats(StatsError::RankDeficient {
            rank: 1,
            ncols: 3,
        }));
        assert!(completion_cannot_be_fit(&singular));
        let too_few = CausalError::from(EstimationError::data_msg("too few aligned rows"));
        assert!(completion_cannot_be_fit(&too_few));
        let unknown = CausalError::from(EstimationError::Data(DataError::UnknownVariable {
            id: VariableId::from_raw(9),
        }));
        assert!(!completion_cannot_be_fit(&unknown));
        let unsupported = CausalError::from(EstimationError::unsupported("a request option"));
        assert!(!completion_cannot_be_fit(&unsupported));
        assert!(!completion_cannot_be_fit(&CausalError::Cancelled { stage: "temporal class" }));
        assert!(!completion_cannot_be_fit(&CausalError::Resource { message: "limit".into() }));
    }

    #[test]
    fn skipped_completions_are_counted_and_a_request_error_is_returned() {
        use antecedent_data::DataError;
        use antecedent_estimate::EstimationError;
        let mut skipped = SkippedCompletions::default();
        assert!(skipped.diagnostic().is_none());
        skipped.record(CausalError::from(EstimationError::data_msg("too few rows"))).unwrap();
        skipped.record(CausalError::from(EstimationError::data_msg("another"))).unwrap();
        assert_eq!(skipped.count, 2);
        let message = skipped.diagnostic().expect("a note for skipped completions").message;
        assert!(message.contains("2 identified completion(s)"), "{message}");
        assert!(skipped.none_estimable().to_string().contains("too few rows"));
        let unknown = CausalError::from(EstimationError::Data(DataError::UnknownVariable {
            id: VariableId::from_raw(9),
        }));
        let returned = skipped.record(unknown).expect_err("a request error ends the run");
        assert!(returned.to_string().contains("unknown variable"), "{returned}");
        assert_eq!(skipped.count, 2);
    }
}
