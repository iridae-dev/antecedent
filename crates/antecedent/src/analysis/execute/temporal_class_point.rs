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
use crate::planner::RouteDefaults;
use antecedent_core::SupportStatus;

const CONDITIONAL_PRIOR_REFUSAL: &str = "temporal conditional effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed";
const DERIVATIVE_PRIOR_REFUSAL: &str = "temporal derivative uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed";
const CROSS_WORLD_PRIOR_REFUSAL: &str = "temporal cross-world effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed";

/// The identified completions of one class envelope, each with its recorded
/// regressions, frozen weight and point.
struct ClassPoints<'a> {
    atoms: Vec<SlopeAtom<'a>>,
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
    for (index, (case, indexer)) in envelope.cases.iter().zip(bundle.indexers.iter()).enumerate() {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let (atom, point) = fit(case, indexer)?;
        atoms.push(atom);
        weights.push(case.weight.0);
        points.push(point);
        values[index] = Some(point);
        if primary.is_none() {
            primary = Some((case.result.clone(), indexer.clone()));
        }
    }
    let (primary_identification, primary_indexer) =
        primary.ok_or_else(|| CausalError::Compile {
            message: "temporal class envelope had no estimable identified completions".into(),
        })?;
    let estimand = primary_identification.estimands.first().cloned().ok_or_else(|| {
        CausalError::Compile { message: "temporal class completion has no estimand".into() }
    })?;
    Ok(ClassPoints {
        atoms,
        weights,
        points,
        values,
        estimand,
        assumptions: primary_identification.required_assumptions.clone(),
        primary_identification,
        primary_indexer,
    })
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

/// Every identified completion's one-modifier conditional design and point.
fn collect_class_conditional<'a>(
    data: &'a TimeSeriesData,
    query: &TemporalEffectQuery,
    bundle: &'a antecedent_identify::TemporalClassEnvelope,
    policy: antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<ClassPoints<'a>, CausalError> {
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
    if query.effect_modifiers.len() != 1
        || query.is_multi_step_sustained()
        || matches!(query.policy, antecedent_core::TemporalPolicy::Dynamic { .. })
    {
        return Err(CausalError::Unsupported {
            message: "temporal conditional effect fits one modifier on a single-step pulse or sustained policy",
        });
    }
    if refute != RefuteSuite::None {
        return Err(CausalError::Unsupported {
            message: "temporal conditional effect is point-only; cheap/full refuters are not licensed",
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
        "published value is the completion-weighted one-modifier marginal effect; its interval \
         refits every completion's interaction regression and modifier mean",
    ));
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

/// The refusal message for a cross-world query whose back-door set is not empty.
fn cross_world_backdoor_refusal(query: &CausalQuery) -> &'static str {
    match query {
        CausalQuery::NestedCounterfactual(_) => {
            "temporal nested effect is licensed when the back-door set is empty"
        }
        CausalQuery::PathSpecific(_) => {
            "temporal path-specific effect is licensed when the back-door set is empty"
        }
        CausalQuery::Counterfactual(_) => {
            "temporal counterfactual is licensed when the back-door set is empty"
        }
        CausalQuery::Distribution(_) => {
            "temporal interventional mean is licensed when the back-door set is empty"
        }
        _ => "temporal cross-world effect is licensed when the back-door set is empty",
    }
}

/// Every identified completion's cross-world regressions and point.
fn collect_class_cross_world<'a>(
    data: &'a TimeSeriesData,
    query: &CausalQuery,
    bundle: &'a antecedent_identify::TemporalClassEnvelope,
    (contrast, offsets): (CrossWorldContrast, (i32, i32)),
    policy: antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<ClassPoints<'a>, CausalError> {
    collect_class_points(bundle, |case, indexer| {
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        if !estimand.adjustment_set.is_empty() {
            let message = cross_world_backdoor_refusal(query);
            return Err(CausalError::Unsupported { message });
        }
        let atom = SlopeAtom::new(data, &estimand, indexer, offsets, policy, prior_scale);
        let point = cross_world_from(query, &atom, contrast)?;
        Ok((atom, point))
    })
}

/// The cross-world point of `query` from the slopes `source` fits.
fn cross_world_from(
    query: &CausalQuery,
    source: &dyn SlopeSource,
    contrast: CrossWorldContrast,
) -> Result<f64, CausalError> {
    let fit = |outcome: VariableId, treatments: &[VariableId], contemporaneous: &[VariableId]| {
        source.slopes(outcome, treatments, contemporaneous)
    };
    super::temporal_path::cross_world_point(query, &fit, contrast)
}

/// The diagnostics of a class derivative response: the envelope, cache
/// provenance, the derivative note and the support warnings.
fn class_derivative_diagnostics(
    envelope: &antecedent_identify::IdentificationEnvelope<
        antecedent_identify::TemporalCompletionGraph,
    >,
    graph_class: GraphClass,
    identify_cached: bool,
    response: &CausalResponse,
) -> Vec<Diagnostic> {
    let mut diagnostics =
        vec![super::temporal_path::temporal_class_envelope_diagnostic(envelope, graph_class)];
    if identify_cached {
        diagnostics.push(identify_cached_diagnostic());
    }
    diagnostics.push(derivative_note());
    diagnostics.extend(response.support.warnings.iter().cloned());
    diagnostics
}

fn derivative_note() -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal.linear_derivative",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "published value is the completion-weighted slope of the linear unfolded temporal fit; \
         its band refits every completion's outcome regressions",
    )
}

/// The weighted completion derivative responses with their recorded regressions,
/// the finite scalar summaries by completion and the first completion's estimand.
struct ClassDerivatives<'a> {
    atoms: Vec<SlopeAtom<'a>>,
    responses: Vec<(f64, CausalResponse)>,
    values: Vec<Option<f64>>,
    estimand: IdentifiedEstimand,
}

/// The linear slope of each identified completion with its weight, the
/// finite scalar summaries by completion, and the first completion's estimand.
fn class_derivative_responses<'a>(
    data: &'a TimeSeriesData,
    query: &ResponseQuery,
    bundle: &'a antecedent_identify::TemporalClassEnvelope,
    offsets: (i32, i32),
    (policy, prior_scale): (antecedent_core::KernelPolicy, Option<f64>),
) -> Result<ClassDerivatives<'a>, CausalError> {
    let envelope = &bundle.envelope;
    require_identified_mass(envelope)?;
    let mut atoms = Vec::new();
    let mut responses = Vec::new();
    let mut values = vec![None; envelope.cases.len()];
    let mut primary_estimand = None;
    for (index, (case, indexer)) in envelope.cases.iter().zip(bundle.indexers.iter()).enumerate() {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let atom = SlopeAtom::new(data, &estimand, indexer, offsets, policy, prior_scale);
        let response = super::temporal_path::temporal_linear_derivative_response(
            data,
            query,
            &case.result,
            &atom,
        )?;
        let scalar = super::response_path::response_scalar_summary(&response).0;
        if scalar.is_finite() {
            values[index] = Some(scalar);
        }
        if primary_estimand.is_none() {
            primary_estimand = Some(estimand);
        }
        atoms.push(atom);
        responses.push((case.weight.0, response));
    }
    let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
        message: "temporal class envelope had no estimable identified completions".into(),
    })?;
    Ok(ClassDerivatives { atoms, responses, values, estimand })
}

/// The derivative of every replicate or draw of one completion.
fn derivative_evaluator(
    functional: &ResponseFunctional,
) -> impl Fn(usize, &dyn SlopeSource) -> Result<Vec<f64>, CausalError> + '_ {
    move |_: usize, source: &dyn SlopeSource| {
        let value = super::temporal_path::temporal_linear_derivative_value(functional, source)?;
        point_uncertainty::response_values(&value).ok_or_else(|| CausalError::Compile {
            message: "temporal derivative replicate has no numeric value".into(),
        })
    }
}

/// The flattened point of a mixed derivative response.
fn response_center(response: &CausalResponse) -> Result<Vec<f64>, CausalError> {
    match &response.estimate {
        ResponseIdentification::PointIdentified(value) => point_uncertainty::response_values(value),
        _ => None,
    }
    .ok_or_else(|| CausalError::Compile {
        message: "temporal class derivative point has no numeric value".into(),
    })
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
            (self.identify_temporal_class(identifier_id, probe)?.envelope.clone(), false)
        })
    }

    /// The uncertainty plan of a class point route of this study.
    fn class_point_plan(
        &self,
        structural_span: usize,
        (block_stream, posterior_stream): (u64, u64),
    ) -> ClassPointUncertainty<'_> {
        ClassPointUncertainty {
            inference: &self.inference,
            bootstrap_replicates: self.bootstrap_replicates,
            structural_span,
            block_stream,
            posterior_stream,
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one class derivative execution: the completions, the mixture and its band stay together"
    )]
    pub(super) fn execute_temporal_class_derivative(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.require_licensed_temporal_observation()?;
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware temporal derivative requires TemporalResponseSpec".into(),
        })?;
        if temporal.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon",
            });
        }
        let prior_scale = isotropic_prior_scale(&self.inference, DERIVATIVE_PRIOR_REFUSAL)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let probe = super::temporal_path::unit_contrast_query(
            temporal,
            (treatment, outcome),
            temporal.horizons[0],
            query.target_population.clone(),
        );
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let (bundle, identify_cached) =
            self.derivative_class_envelope(identifier_id, &probe, physical, ctx)?;
        let offsets = (probe.try_treatment_offset()?, probe.outcome_offset());
        let envelope = &bundle.envelope;
        let ClassDerivatives { atoms, responses, values, estimand } = class_derivative_responses(
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
        diagnostics.extend(uncertainty_diagnostics);
        // A derivative's cheap/full suite is its refuter state on every identified
        // completion's Pulse witness at the one horizon (unlicensed).
        let (refutations, predictive_checks) = if self.refute == RefuteSuite::None {
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
        let estimator_id = EstimatorId::temporal_response_for(&self.inference);
        let structural = super::temporal_path::temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &values,
        );
        response.identification_status = identification.status;
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
        result.rebind_interval(self.inference.is_bayesian());
        Ok(result)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one class cross-world execution: the completions, the mixture and its interval stay together"
    )]
    pub(super) fn execute_temporal_class_cross_world(
        &self,
        data: &TimeSeriesData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal cross-world effects are point-only; cheap/full refuters are not licensed",
            });
        }
        let prior_scale = isotropic_prior_scale(&self.inference, CROSS_WORLD_PRIOR_REFUSAL)?;
        let (treatment, outcome, mediator, active, control) = cross_world_endpoints(&self.query)?;
        let contrast = CrossWorldContrast { treatment, outcome, mediator, active, control };
        let probe = cross_world_probe(treatment, outcome);
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        // The probe is identified once, at compile, and the program retains its certificate;
        // a plan that retained none identifies here.
        let (bundle, identify_cached) =
            if let Some(crate::planner::lowered::RetainedExecutionProof::CrossWorldClass {
                envelope,
            }) = crate::analysis::lowering::plan_retained_execution_proof(self, physical)
            {
                (Arc::clone(envelope), false)
            } else {
                report_identify_compute(ctx);
                (self.identify_temporal_class(identifier_id, &probe)?.envelope.clone(), false)
            };
        let offsets = (probe.try_treatment_offset()?, probe.outcome_offset());
        let query = &self.query;
        let fitted = collect_class_cross_world(
            data,
            query,
            &bundle,
            (contrast, offsets),
            ctx.kernel_policy,
            prior_scale,
        )?;
        let point = fitted.point()?;
        let identification = super::temporal_path::template_scoped_cross_world(
            fitted.primary_identification.clone(),
            &fitted.primary_indexer,
            self.query.clone(),
            contrast,
        );
        let estimand = identification.estimands[0].clone();
        let envelope = &bundle.envelope;
        let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        )];
        diagnostics.push(cross_world_note());
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let mut structural = super::temporal_path::temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &fitted.values,
        );
        let frozen: Vec<FrozenAtom> = fitted.atoms.into_iter().map(SlopeAtom::freeze).collect();
        let outcome_uncertainty = class_scalar_uncertainty(
            &frozen,
            &fitted.weights,
            &fitted.points,
            point,
            identification.required_assumptions.clone(),
            envelope.truncated_completions > 0,
            self.class_point_plan(
                envelope_span(&bundle),
                (
                    point_uncertainty::CROSS_WORLD_BLOCK_STREAM,
                    point_uncertainty::CROSS_WORLD_POSTERIOR_STREAM,
                ),
            ),
            ctx,
            |_, source| cross_world_from(query, source, contrast).map(|value| vec![value]),
        )?;
        structural.identified_set_interval = outcome_uncertainty.identified_set;
        diagnostics.extend(outcome_uncertainty.diagnostics.iter().cloned());
        let posterior = outcome_uncertainty.posterior;
        let n_draws = posterior
            .as_ref()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                outcome_uncertainty.estimate,
                Strategy::new(identifier_id, EstimatorId::TemporalLinearAdjustment),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(outcome_uncertainty.replicates_ok)
            .cancelled(outcome_uncertainty.cancelled)
            .extras(IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: Arc::unwrap_or_clone(bundle),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                n_draws,
                posterior,
                ..Default::default()
            }),
        );
        Ok(result)
    }

    /// Graph-posterior conditional, derivative, and cross-world points.
    ///
    /// Each atom runs the same producer as the explicit class route. The
    /// published scalar is the posterior-weighted mean of those
    /// completion-weighted points. A single atom publishes that atom's
    /// completion-weighted mean. The uncertainty treats every identified
    /// completion of every atom as one frozen-weight atom (posterior weight times
    /// normalized completion weight) of one shared procedure.
    pub(super) fn execute_temporal_class_graph_posterior_family(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match &self.query {
            CausalQuery::TemporalEffect(_) => {
                self.execute_posterior_conditional(data, gp, physical, ctx)
            }
            CausalQuery::Response(_) => self.execute_posterior_derivative(data, gp, physical, ctx),
            _ => self.execute_posterior_cross_world(data, gp, physical, ctx),
        }
    }

    fn execute_posterior_conditional(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let CausalQuery::TemporalEffect(query) = &self.query else { unreachable!() };
        if query.effect_modifiers.len() != 1
            || query.is_multi_step_sustained()
            || matches!(query.policy, antecedent_core::TemporalPolicy::Dynamic { .. })
        {
            return Err(CausalError::Unsupported {
                message: "temporal conditional effect fits one modifier on a single-step pulse or sustained policy",
            });
        }
        if self.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal conditional effect is point-only; cheap/full refuters are not licensed",
            });
        }
        let prior_scale = isotropic_prior_scale(&self.inference, CONDITIONAL_PRIOR_REFUSAL)?;
        let (cache, identify_cached) = load_class_posterior_cache(self, gp, data, query, ctx)?;
        let mut pairs = Vec::new();
        let mut pooled = PooledAtoms::default();
        let mut chosen = None;
        let key_weights = IdentifiedKeyWeights::new(&cache.graphs);
        for atom in cache.class_atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let fitted = collect_class_conditional(
                data,
                query,
                &atom.envelope,
                ctx.kernel_policy,
                prior_scale,
            )?;
            let point = fitted.point()?;
            if !point.is_finite() {
                continue;
            }
            pairs.push((weight, point));
            let envelope = &atom.envelope.envelope;
            if chosen.is_none() {
                let mut diagnostics =
                    vec![super::temporal_path::temporal_class_envelope_diagnostic(
                        envelope,
                        self.graph.class(),
                    )];
                diagnostics.push(Diagnostic::new(
                    "estimate.temporal.conditional_marginal",
                    DiagnosticKind::Execution,
                    DiagnosticSeverity::Info,
                    "published value is the completion-weighted one-modifier marginal effect; its \
                     interval refits every completion's interaction regression and modifier mean",
                ));
                chosen = Some((
                    fitted.assumptions.clone(),
                    fitted.estimand.clone(),
                    envelope_to_identification_result_for(
                        envelope,
                        CausalQuery::TemporalEffect(query.clone()),
                    ),
                    diagnostics,
                    super::temporal_path::temporal_class_structural_mixture(
                        envelope,
                        crate::result::StructuralWeightBasis::CompletionEnumeration,
                        None,
                        &fitted.values,
                    ),
                    envelope.status,
                ));
            }
            pooled.push(fitted, weight, envelope.truncated_completions > 0);
        }
        self.finish_posterior_point(
            ExecuteRun { physical, ctx, started, identify_cached },
            &cache,
            (&pairs, pooled, (None, false)),
            chosen,
            (query.treatment, query.outcome),
            (
                EstimatorId::TemporalLinearAdjustment,
                (
                    point_uncertainty::CONDITIONAL_BLOCK_STREAM,
                    point_uncertainty::CONDITIONAL_POSTERIOR_STREAM,
                ),
            ),
            &|_: usize, source: &dyn SlopeSource| {
                source.conditional_effect().map(|value| vec![value])
            },
        )
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one graph-posterior derivative execution: the atoms, the mixture and its band stay together"
    )]
    fn execute_posterior_derivative(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let CausalQuery::Response(query) = &self.query else { unreachable!() };
        query.require_licensed_temporal_observation()?;
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware temporal derivative requires TemporalResponseSpec".into(),
        })?;
        if temporal.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon",
            });
        }
        let probe = derivative_probe(query)?;
        let (cache, identify_cached) = load_class_posterior_cache(self, gp, data, &probe, ctx)?;
        let prior_scale = isotropic_prior_scale(&self.inference, DERIVATIVE_PRIOR_REFUSAL)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let offsets = (probe.try_treatment_offset()?, probe.outcome_offset());
        let mut pairs = Vec::new();
        let mut responses = Vec::new();
        let mut pooled_atoms = Vec::new();
        let mut pooled_weights = Vec::new();
        let mut spans = Vec::new();
        let mut chosen = None;
        let mut witness_states = super::posterior_witness_state::AtomWitnessStates::default();
        let key_weights = IdentifiedKeyWeights::new(&cache.graphs);
        for atom in cache.class_atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let derived = derivative_on_envelope(
                data,
                query,
                &atom.envelope,
                self.graph.class(),
                (offsets, ctx.kernel_policy, prior_scale),
            )?;
            let scalar = super::response_path::response_scalar_summary(&derived.response).0;
            if !derivative_response_finite(&derived.response) {
                continue;
            }
            pairs.push((weight, scalar));
            responses.push((weight, derived.response.clone()));
            if self.refute != RefuteSuite::None {
                // Each atom's derivative is refuted on its own identified completions' Pulse
                // witnesses against its own response, then the reports mix by posterior weight.
                let sites = super::posterior_witness_state::class_completion_sites(
                    &atom.envelope,
                    &probe,
                    temporal.horizons[0],
                )?;
                witness_states.push_derivative(
                    self,
                    data,
                    (atom.key, weight),
                    (query, &derived.response),
                    &sites,
                    ctx,
                )?;
            }
            let total: f64 = derived.weights.iter().sum();
            for (completion, completion_weight) in
                derived.atoms.into_iter().zip(derived.weights.iter())
            {
                pooled_atoms.push(completion.freeze());
                pooled_weights.push(weight * completion_weight / total);
            }
            spans.push(envelope_span(&atom.envelope));
            if chosen.is_none() {
                chosen = Some((
                    derived.identification.required_assumptions.clone(),
                    derived.estimand,
                    derived.identification,
                    derived.diagnostics,
                    derived.structural,
                    derived.status,
                ));
            }
        }
        let mut response = mix_derivative_responses(&responses)?;
        let center = response_center(&response)?;
        let (uncertainty_diagnostics, replicates_ok, cancelled) = class_response_uncertainty(
            &pooled_atoms,
            &pooled_weights,
            (&mut response, &center),
            self.class_point_plan(
                spans.iter().copied().max().unwrap_or(1),
                (
                    point_uncertainty::DERIVATIVE_BLOCK_STREAM,
                    point_uncertainty::DERIVATIVE_POSTERIOR_STREAM,
                ),
            ),
            ctx,
            derivative_evaluator(&query.functional),
        )?;
        let chosen = chosen.map(|mut chosen| {
            chosen.3.extend(uncertainty_diagnostics);
            chosen
        });
        let result = self.finish_posterior_point(
            ExecuteRun { physical, ctx, started, identify_cached },
            &cache,
            (&pairs, PooledAtoms::default(), (replicates_ok, cancelled)),
            chosen,
            (treatment, outcome),
            (
                EstimatorId::temporal_response_for(&self.inference),
                (
                    point_uncertainty::DERIVATIVE_BLOCK_STREAM,
                    point_uncertainty::DERIVATIVE_POSTERIOR_STREAM,
                ),
            ),
            &derivative_evaluator(&query.functional),
        )?;
        response.identification_status = result.identification.status;
        let mut result = result;
        result.payload = crate::result::ResultPayload::Response(response);
        result.posterior = None;
        if let Some(structural) = result.structural_response.as_mut() {
            structural.identified_set_interval = None;
        }
        if self.refute != RefuteSuite::None {
            let (reports, checks, notes) = witness_states.finish();
            result.refutations = reports;
            result.predictive_checks = checks;
            result.diagnostics.extend(notes);
        }
        result.rebind_interval(self.inference.is_bayesian());
        Ok(result)
    }

    fn execute_posterior_cross_world(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal cross-world effects are point-only; cheap/full refuters are not licensed",
            });
        }
        let (treatment, outcome, mediator, active, control) = cross_world_endpoints(&self.query)?;
        let contrast = CrossWorldContrast { treatment, outcome, mediator, active, control };
        let probe = cross_world_probe(treatment, outcome);
        let (cache, identify_cached) = load_class_posterior_cache(self, gp, data, &probe, ctx)?;
        let prior_scale = isotropic_prior_scale(&self.inference, CROSS_WORLD_PRIOR_REFUSAL)?;
        let offsets = (probe.try_treatment_offset()?, probe.outcome_offset());
        let mut pairs = Vec::new();
        let mut pooled = PooledAtoms::default();
        let mut chosen = None;
        let key_weights = IdentifiedKeyWeights::new(&cache.graphs);
        let query = &self.query;
        for atom in cache.class_atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let fitted = collect_class_cross_world(
                data,
                query,
                &atom.envelope,
                (contrast, offsets),
                ctx.kernel_policy,
                prior_scale,
            )?;
            let point = fitted.point()?;
            if !point.is_finite() {
                continue;
            }
            let envelope = &atom.envelope.envelope;
            if chosen.is_none() {
                let identification = super::temporal_path::template_scoped_cross_world(
                    fitted.primary_identification.clone(),
                    &fitted.primary_indexer,
                    self.query.clone(),
                    contrast,
                );
                let estimand = identification.estimands[0].clone();
                let diagnostics = vec![
                    super::temporal_path::temporal_class_envelope_diagnostic(
                        envelope,
                        self.graph.class(),
                    ),
                    cross_world_note(),
                ];
                let structural = super::temporal_path::temporal_class_structural_mixture(
                    envelope,
                    crate::result::StructuralWeightBasis::CompletionEnumeration,
                    None,
                    &fitted.values,
                );
                chosen = Some((
                    identification.required_assumptions.clone(),
                    estimand,
                    identification,
                    diagnostics,
                    structural,
                    envelope.status,
                ));
            }
            pairs.push((weight, point));
            pooled.push(fitted, weight, envelope.truncated_completions > 0);
        }
        self.finish_posterior_point(
            ExecuteRun { physical, ctx, started, identify_cached },
            &cache,
            (&pairs, pooled, (None, false)),
            chosen,
            (treatment, outcome),
            (
                EstimatorId::TemporalLinearAdjustment,
                (
                    point_uncertainty::CROSS_WORLD_BLOCK_STREAM,
                    point_uncertainty::CROSS_WORLD_POSTERIOR_STREAM,
                ),
            ),
            &|_: usize, source: &dyn SlopeSource| {
                cross_world_from(query, source, contrast).map(|value| vec![value])
            },
        )
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the run, the cache, the pooled completions, the chosen atom's claim, the pair, the estimator with its streams and the functional are distinct inputs"
    )]
    fn finish_posterior_point(
        &self,
        run: ExecuteRun<'_>,
        cache: &crate::analysis::prepared::CachedTemporalClassPosteriorIdentification,
        (pairs, pooled, accounting): (&[(f64, f64)], PooledAtoms, (Option<u32>, bool)),
        chosen: Option<(
            antecedent_core::AssumptionSet,
            IdentifiedEstimand,
            IdentificationResult,
            Vec<Diagnostic>,
            crate::result::StructuralResponseMixture,
            IdentificationStatus,
        )>,
        (treatment, outcome): (VariableId, VariableId),
        (estimator_id, streams): (EstimatorId, (u64, u64)),
        evaluate: &dyn Fn(usize, &dyn SlopeSource) -> Result<Vec<f64>, CausalError>,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let (assumptions, estimand, mut identification, mut diagnostics, mut structural, atom_status) =
            chosen.ok_or_else(|| CausalError::Compile {
                message: format!(
                    "temporal class graph-posterior point had no estimable atom (atoms={}, unidentified={})",
                    cache.class_atoms.len(),
                    cache.graphs.unidentified_mass()
                ),
            })?;
        let point = weighted_derivative_mean(pairs)?;
        let unidentified = cache.graphs.unidentified_mass();
        identification.status = resolve_posterior_status(
            atom_status,
            crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
            PosteriorMassSummary { unidentified, ..PosteriorMassSummary::default() },
        );
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        if pairs.len() > 1 {
            diagnostics.push(Diagnostic::new(
                "estimate.graph_posterior.temporal_class_point",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "published value is the posterior-weighted mean of each atom's completion-weighted point; completion enumeration is not posterior probability",
            ));
        }
        let PooledAtoms { atoms, weights, points, truncated, spans } = pooled;
        let (estimate, posterior, replicates_ok, cancelled) = if atoms.is_empty() {
            (
                EffectEstimate::new(point, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride),
                None,
                accounting.0,
                accounting.1,
            )
        } else {
            let outcome_uncertainty = class_scalar_uncertainty(
                &atoms,
                &weights,
                &points,
                point,
                assumptions,
                truncated,
                self.class_point_plan(spans.iter().copied().max().unwrap_or(1), streams),
                ctx,
                evaluate,
            )?;
            structural.identified_set_interval = outcome_uncertainty.identified_set;
            diagnostics.extend(outcome_uncertainty.diagnostics.iter().cloned());
            (
                outcome_uncertainty.estimate,
                outcome_uncertainty.posterior,
                outcome_uncertainty.replicates_ok,
                outcome_uncertainty.cancelled,
            )
        };
        let n_draws = posterior
            .as_ref()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let result = self.finish_identified_execute(
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
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(replicates_ok)
            .cancelled(cancelled)
            .extras(IdentifiedExecuteExtras {
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                n_draws,
                posterior,
                ..Default::default()
            }),
        );
        Ok(result)
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

fn cross_world_note() -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal.cross_world",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "published value is the completion-weighted linear unfolded cross-world point; its \
         interval refits every completion's regressions of the contrast",
    )
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
            if q.path_nodes.len() != 1 {
                return Err(CausalError::Unsupported {
                    message: "temporal path-specific effect is the one-mediator edge product",
                });
            }
            Ok((
                q.treatment,
                q.outcome,
                Some(q.path_nodes[0]),
                super::temporal_path::temporal_cross_world_hard_level(&q.active)?,
                super::temporal_path::temporal_cross_world_hard_level(&q.control)?,
            ))
        }
        CausalQuery::Counterfactual(q) => {
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
        CausalQuery::TemporalEffect(effect) => effect.effect_modifiers.len() == 1,
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
    if let Some(cache) = study.temporal_class_posterior_identification_cache.as_ref() {
        return Ok((Arc::clone(cache), true));
    }
    let variables =
        data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
    let cache = crate::analysis::prepared::build_temporal_class_posterior_identification_cache(
        gp,
        &variables,
        probe,
        study.max_completions,
        ctx,
    )?;
    Ok((Arc::new(cache), false))
}

fn derivative_probe(query: &ResponseQuery) -> Result<TemporalEffectQuery, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "class-aware temporal derivative requires TemporalResponseSpec".into(),
    })?;
    let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
    Ok(super::temporal_path::unit_contrast_query(
        temporal,
        (treatment, outcome),
        temporal.horizons[0],
        query.target_population.clone(),
    ))
}

/// One class envelope's mixed derivative with its completions' recorded
/// regressions and weights, estimand, identification, diagnostics, structural
/// mixture and status.
struct ClassDerivative<'a> {
    response: CausalResponse,
    atoms: Vec<SlopeAtom<'a>>,
    weights: Vec<f64>,
    estimand: IdentifiedEstimand,
    identification: IdentificationResult,
    diagnostics: Vec<Diagnostic>,
    structural: crate::result::StructuralResponseMixture,
    status: IdentificationStatus,
}

fn derivative_on_envelope<'a>(
    data: &'a TimeSeriesData,
    query: &ResponseQuery,
    bundle: &'a antecedent_identify::TemporalClassEnvelope,
    graph_class: GraphClass,
    (offsets, policy, prior_scale): ((i32, i32), antecedent_core::KernelPolicy, Option<f64>),
) -> Result<ClassDerivative<'a>, CausalError> {
    let envelope = &bundle.envelope;
    let ClassDerivatives { atoms, responses, values, estimand } =
        class_derivative_responses(data, query, bundle, offsets, (policy, prior_scale))?;
    let weights: Vec<f64> = responses.iter().map(|(weight, _)| *weight).collect();
    let response = mix_derivative_responses(&responses)?;
    let mut diagnostics =
        vec![super::temporal_path::temporal_class_envelope_diagnostic(envelope, graph_class)];
    diagnostics.push(derivative_note());
    diagnostics.extend(response.support.warnings.iter().cloned());
    let identification =
        envelope_to_identification_result_for(envelope, CausalQuery::Response(query.clone()));
    let structural = super::temporal_path::temporal_class_structural_mixture(
        envelope,
        crate::result::StructuralWeightBasis::CompletionEnumeration,
        None,
        &values,
    );
    Ok(ClassDerivative {
        response,
        atoms,
        weights,
        estimand,
        identification,
        diagnostics,
        structural,
        status: envelope.status,
    })
}
