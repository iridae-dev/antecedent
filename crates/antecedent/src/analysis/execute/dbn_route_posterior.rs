// SPDX-License-Identifier: MIT OR Apache-2.0
//! Conditional effects, single-horizon derivatives and cross-world contrasts over a
//! supplied dynamic-Bayesian graph posterior on a series.
//!
//! Every atom of the posterior is a `TemporalDag`: a contemporaneous DAG with lagged parents.
//! Each atom is identified on its own graph by the temporal back-door identifier and answers the
//! query with the linear unfolded producer of the explicit `TemporalDag` cell: the fitted
//! interaction slope and modifier mean of a conditional effect, the slope of the outcome
//! regression a derivative, elasticity, Jacobian or directional derivative reads, or the
//! regressions of a cross-world contrast (counterfactual, interventional mean, natural direct
//! effect or path-restricted effect). Every regression an atom's point reads is recorded, so the
//! uncertainty of the mixture is computed on all atoms at once:
//!
//! * Frequentist: every regression of every atom is refit on the same circular-block resample of
//!   the series times, the frozen posterior weights are reapplied to the atoms' values of that
//!   replicate, and the standard error is the spread of the draw-wise mixture;
//! * Bayesian: the coefficients of every regression of every atom are one joint draw, draw `d`
//!   of every atom enters draw `d` of the functional, and the pooled posterior is the
//!   draw-wise posterior-weighted mixture of the atoms' draws.
//!
//! An atom the identifier refuses is unidentified posterior mass, and an atom whose regressions
//! cannot be fitted or whose functional is not finite is unevaluable mass; neither is
//! redistributed, so the published value is conditional on the atoms that identify and evaluate
//! and the result states the mass it does not cover.

use antecedent_data::TableView;
use std::collections::HashMap;

use antecedent_core::{CausalQuery, ResponseQuery, TemporalEffectQuery};
use antecedent_discovery::{AdjacencyMask, GraphPosteriorAtomKind};
use antecedent_identify::TemporalBackdoorIdentifier;

use super::atom_mix::{AtomBoundRule, AtomMixInput, MixedAtomEvals, mix_atom_evals};
use super::temporal_path::{
    CrossWorldContrast, cross_world_point, template_scoped_cross_world,
    temporal_linear_derivative_response, temporal_linear_derivative_value, unit_contrast_query,
};
use super::temporal_point_uncertainty::{
    self as point_uncertainty, FrozenAtom, SlopeAtom, SlopeSource,
};
use super::transport_posterior::ScalarAtomEval;
use super::{
    Arc, CausalError, CausalResponse, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, ExecutionContext, GraphIdentFlag, GraphPosterior,
    IdentificationResult, IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifiedKeyWeights, IdentifierId, InferenceMode, Instant,
    OverlapPolicy, PhysicalExecutionPlan, PosteriorMassSummary, RefuteSuite,
    ResponseIdentification, ResponseValue, Strategy, StudyResult, TemporalIndexer, TimeSeriesData,
    VariableId, WeightedGraphSamples, elapsed_ns, identification_status_ok_for_case,
    provenance_ids, resolve_posterior_status, select_estimand,
};
use crate::analysis::identification_cache::CachedGraphPosteriorAtomIdentification;
use crate::analysis::proof_support::graph_posterior_withheld_diagnostic;

/// Method tag every posterior atom's estimand carries.
const ATOM_ESTIMAND_METHOD: &str = "graph_posterior.dbn_route";

/// Stable code of the disclosure every dynamic graph-posterior point result carries.
pub(crate) const DBN_ROUTE_POSTERIOR_CODE: &str = "estimate.graph_posterior.dbn_route";

const CONDITIONAL_PRIOR_REFUSAL: &str = "a temporal conditional effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed";
const DERIVATIVE_PRIOR_REFUSAL: &str = "a temporal derivative uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed";
const CROSS_WORLD_PRIOR_REFUSAL: &str = "a temporal cross-world effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed";

/// Whether a posterior is a dynamic-Bayesian posterior over `TemporalDag` atoms.
#[must_use]
pub(crate) fn dbn_atoms(posterior: &GraphPosterior) -> bool {
    posterior.atom_kind == GraphPosteriorAtomKind::Dag
        && posterior.lag_masks.is_some()
        && posterior.max_lag.is_some()
}

/// Whether a series query is answered atom by atom over `TemporalDag` atoms: a
/// conditional (modifier-set) pulse or sustained contrast, the slope of a derivative at one or several horizons, or a
/// cross-world contrast.
#[must_use]
pub(crate) fn dbn_route_query(query: &CausalQuery) -> bool {
    match query {
        CausalQuery::TemporalEffect(effect) => !effect.effect_modifiers.is_empty(),
        CausalQuery::Response(response) => {
            super::response_functional_is_derivative(&response.functional)
                && response.temporal.as_ref().is_some_and(|spec| !spec.horizons.is_empty())
        }
        CausalQuery::NestedCounterfactual(_)
        | CausalQuery::PathSpecific(_)
        | CausalQuery::Counterfactual(_)
        | CausalQuery::Distribution(_) => true,
        _ => false,
    }
}

/// What the point of each atom is.
enum DbnPoint<'a> {
    Conditional(&'a TemporalEffectQuery),
    Derivative { query: &'a ResponseQuery, probe: TemporalEffectQuery },
    CrossWorld { contrast: CrossWorldContrast, probe: TemporalEffectQuery, direct: bool },
}

impl<'a> DbnPoint<'a> {
    /// The points of a query: one, or for a derivative one per requested horizon.
    fn of_all(query: &'a CausalQuery) -> Result<Vec<Self>, CausalError> {
        let CausalQuery::Response(response) = query else {
            return Ok(vec![Self::of(query)?]);
        };
        response.require_licensed_temporal_observation()?;
        let temporal = response.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "a temporal derivative requires TemporalResponseSpec".into(),
        })?;
        if temporal.horizons.is_empty() {
            return Err(CausalError::Compile {
                message: "a temporal derivative names at least one horizon".into(),
            });
        }
        if temporal.horizons.len() > 1
            && !matches!(
                response.functional,
                antecedent_core::ResponseFunctional::PointDerivative { .. }
                    | antecedent_core::ResponseFunctional::AverageDerivative { .. }
            )
        {
            return Err(CausalError::Unsupported {
                message: "a Jacobian or directional derivative is the local slope at one horizon; \
                          name one horizon, or a point or average derivative over several",
            });
        }
        let (treatment, outcome) =
            super::response_path::response_primary_pair(&response.functional)?;
        Ok(temporal
            .horizons
            .iter()
            .map(|&horizon| Self::Derivative {
                query: response,
                probe: unit_contrast_query(
                    temporal,
                    (treatment, outcome),
                    horizon,
                    response.target_population.clone(),
                ),
            })
            .collect())
    }

    fn of(query: &'a CausalQuery) -> Result<Self, CausalError> {
        match query {
            CausalQuery::TemporalEffect(effect) => {
                if effect.effect_modifiers.is_empty()
                    || effect.is_multi_step_sustained()
                    || matches!(effect.policy, antecedent_core::TemporalPolicy::Dynamic { .. })
                {
                    return Err(CausalError::Unsupported {
                        message: "a temporal conditional effect fits a modifier set on a single-step \
                                  pulse or sustained policy",
                    });
                }
                Ok(Self::Conditional(effect))
            }
            CausalQuery::NestedCounterfactual(_)
            | CausalQuery::PathSpecific(_)
            | CausalQuery::Counterfactual(_)
            | CausalQuery::Distribution(_) => {
                let (treatment, outcome, mediator, active, control) =
                    super::temporal_class_point::cross_world_endpoints(query)?;
                Ok(Self::CrossWorld {
                    contrast: CrossWorldContrast { treatment, outcome, mediator, active, control },
                    probe: super::temporal_class_point::cross_world_probe(treatment, outcome),
                    direct: matches!(
                        query,
                        CausalQuery::NestedCounterfactual(_) | CausalQuery::PathSpecific(_)
                    ),
                })
            }
            _ => Err(CausalError::Compile {
                message: "a dynamic graph-posterior point answers a conditional \
                          effect, a derivative or a cross-world contrast"
                    .into(),
            }),
        }
    }

    const fn label(&self) -> &'static str {
        match self {
            Self::Conditional(_) => "temporal conditional effect",
            Self::Derivative { .. } => "temporal derivative",
            Self::CrossWorld { .. } => "temporal cross-world effect",
        }
    }

    fn treatment_outcome(&self) -> (VariableId, VariableId) {
        match self {
            Self::Conditional(query) => (query.treatment, query.outcome),
            Self::Derivative { probe, .. } => (probe.treatment, probe.outcome),
            Self::CrossWorld { contrast, .. } => (contrast.treatment, contrast.outcome),
        }
    }

    /// The query the temporal back-door identifier decides for an atom.
    fn identification_query(&self) -> &TemporalEffectQuery {
        match self {
            Self::Conditional(query) => query,
            Self::Derivative { probe, .. } | Self::CrossWorld { probe, .. } => probe,
        }
    }
}

/// One distinct template of the posterior, identified on its own graph.
struct IdentifiedTemplate {
    key: u64,
    identification: IdentificationResult,
    indexer: TemporalIndexer,
    estimand: IdentifiedEstimand,
}

/// The estimand an atom contributes: the target, whatever back-door set produced it.
fn target_estimand(estimand: &IdentifiedEstimand) -> IdentifiedEstimand {
    let mut target = estimand.clone();
    target.method = Arc::from(ATOM_ESTIMAND_METHOD);
    target.adjustment_set = Arc::from([]);
    target.instruments = Arc::from([]);
    target.mediators = Arc::from([]);
    target
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

/// The mixed derivative answer: the primary atom's response with the mixed value in its shape,
/// the identification status of the mixture and no interval until a band is attached.
fn mixed_derivative_response(
    mut base: CausalResponse,
    center: &[f64],
    (functional, status): (&antecedent_core::ResponseFunctional, IdentificationStatus),
) -> CausalResponse {
    let template = match &base.estimate {
        ResponseIdentification::PointIdentified(value) => Some(value.clone()),
        _ => None,
    };
    let value = match template {
        Some(ResponseValue::Vector(_)) => ResponseValue::Vector(Arc::from(center.to_vec())),
        Some(ResponseValue::Jacobian { outcomes, treatments, .. }) => {
            ResponseValue::Jacobian { outcomes, treatments, values: Arc::from(center.to_vec()) }
        }
        _ => ResponseValue::Scalar(center[0]),
    };
    base.estimand = functional.clone();
    base.identification_status = status;
    base.estimate = ResponseIdentification::PointIdentified(value);
    base.uncertainty = antecedent_core::ResponseUncertainty::None;
    base.provenance_id = Arc::from("estimate.response.graph_posterior.dbn_route");
    base
}

/// One horizon's fit of a dynamic posterior: every identified template's regressions frozen, the
/// atoms' values mixed by posterior weight and the masses the mixture accounts for.
struct DbnLane<'q> {
    point: DbnPoint<'q>,
    templates: Vec<IdentifiedTemplate>,
    /// Position in `templates` of each frozen regression set.
    positions: Vec<usize>,
    frozen: Vec<FrozenAtom>,
    /// Frozen posterior weight of each frozen atom.
    weights: Vec<f64>,
    responses: Vec<Option<CausalResponse>>,
    mixed: MixedAtomEvals,
    center: Vec<f64>,
    primary_position: usize,
    /// Circular-block span the contributing templates need.
    span: usize,
}

impl DbnLane<'_> {
    fn primary(&self) -> &IdentifiedTemplate {
        &self.templates[self.primary_position]
    }
}

/// Refutation reports, predictive checks and their accompanying diagnostic disclosures.
type LaneRefutation = (
    Vec<antecedent_validate::RefutationReport>,
    Vec<super::PredictiveCheckReport>,
    Vec<Diagnostic>,
);

impl super::Study {
    /// Fit one horizon (or the one point) of a dynamic posterior: identify every distinct
    /// template on its own graph, fit its regressions and mix the values.
    ///
    /// # Errors
    ///
    /// No atom identifies or evaluates the point, or a cancelled run.
    #[allow(
        clippy::too_many_lines,
        reason = "one mixture: the templates, their regressions and the masses read one frozen posterior"
    )]
    fn fit_dbn_lane<'q>(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        (keys, variables): (&[u64], &[VariableId]),
        point_kind: DbnPoint<'q>,
        prior_scale: Option<f64>,
        ctx: &ExecutionContext,
    ) -> Result<DbnLane<'q>, CausalError> {
        let label = point_kind.label();
        // Identify every distinct template on its own graph.
        let mut templates: Vec<IdentifiedTemplate> = Vec::new();
        let mut unidentified_keys: std::collections::HashSet<u64> =
            std::collections::HashSet::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for &key in keys.iter().take(gp.n_graphs) {
            if !seen.insert(key) {
                continue;
            }
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let found = crate::analysis::prepared::temporal_dag_from_dbn_atom(gp, key, variables)
                .ok()
                .and_then(|graph| {
                    let identifier = if matches!(point_kind, DbnPoint::Conditional(_)) {
                        TemporalBackdoorIdentifier::new().with_parent_adjustment_fallback()
                    } else {
                        TemporalBackdoorIdentifier::new()
                    };
                    identifier.identify_temporal(&graph, point_kind.identification_query()).ok()
                });
            let template = found.and_then(|found| {
                if !identification_status_ok_for_case(found.result.status)
                    || found.result.estimands.is_empty()
                {
                    return None;
                }
                let estimand =
                    select_estimand(&found.result, EstimatorId::TemporalLinearAdjustment).ok()?;
                Some(IdentifiedTemplate {
                    key,
                    identification: found.result,
                    indexer: found.indexer,
                    estimand,
                })
            });
            match template {
                Some(template) => templates.push(template),
                None => {
                    unidentified_keys.insert(key);
                }
            }
        }
        let flags: Vec<GraphIdentFlag> = keys
            .iter()
            .map(|key| {
                if unidentified_keys.contains(key) {
                    GraphIdentFlag::Unidentified
                } else {
                    GraphIdentFlag::Identified
                }
            })
            .collect();
        let graphs = WeightedGraphSamples::new(gp.weights.to_vec(), flags, keys.to_vec())?;
        let unidentified_mass = graphs.unidentified_mass();
        let key_weights = IdentifiedKeyWeights::new(&graphs);
        if templates.is_empty() {
            return Err(CausalError::not_identified(
                IdentificationStatus::NotIdentified,
                false,
                &format!("no atom of the graph posterior identifies the {label}"),
            ));
        }

        // Every identified template's regressions and point.
        let offsets = match &point_kind {
            DbnPoint::Conditional(_) => (0, 0),
            DbnPoint::Derivative { probe, .. } | DbnPoint::CrossWorld { probe, .. } => {
                (probe.try_treatment_offset()?, probe.outcome_offset())
            }
        };
        let mut slope_atoms: Vec<(usize, SlopeAtom<'_>)> = Vec::new();
        let mut values: Vec<Option<Vec<f64>>> = Vec::with_capacity(templates.len());
        let mut responses: Vec<Option<CausalResponse>> = Vec::with_capacity(templates.len());
        let mut first_refusal: Option<CausalError> = None;
        for (position, template) in templates.iter().enumerate() {
            let built = match &point_kind {
                DbnPoint::Conditional(query) => SlopeAtom::conditional(
                    data,
                    &template.estimand,
                    query,
                    &template.indexer,
                    ctx.kernel_policy,
                    prior_scale,
                    false,
                ),
                DbnPoint::Derivative { .. } => Ok(SlopeAtom::new(
                    data,
                    &template.estimand,
                    &template.indexer,
                    offsets,
                    ctx.kernel_policy,
                    prior_scale,
                )),
                DbnPoint::CrossWorld { direct, .. } => {
                    let mut used = template.estimand.clone();
                    if *direct {
                        used.adjustment_set = Arc::from([]);
                    }
                    Ok(SlopeAtom::new(
                        data,
                        &used,
                        &template.indexer,
                        offsets,
                        ctx.kernel_policy,
                        prior_scale,
                    ))
                }
            };
            let point = built.and_then(|atom| {
                let (value, response) = match &point_kind {
                    DbnPoint::Conditional(_) => (vec![atom.conditional_effect()?], None),
                    DbnPoint::Derivative { query, .. } => {
                        let response = temporal_linear_derivative_response(
                            data,
                            query,
                            &template.identification,
                            &atom,
                        )?;
                        let value = match &response.estimate {
                            ResponseIdentification::PointIdentified(value) => {
                                point_uncertainty::response_values(value)
                            }
                            _ => None,
                        }
                        .ok_or_else(|| CausalError::Compile {
                            message: "a temporal derivative atom has no numeric value".into(),
                        })?;
                        (value, Some(response))
                    }
                    DbnPoint::CrossWorld { contrast, .. } => {
                        let fit =
                            |outcome: VariableId,
                             treatments: &[VariableId],
                             contemporaneous: &[VariableId]| {
                                atom.slopes(outcome, treatments, contemporaneous)
                            };
                        (vec![cross_world_point(&self.query, &fit, *contrast)?], None)
                    }
                };
                Ok((atom, value, response))
            });
            match point {
                Ok((atom, value, response))
                    if !value.is_empty() && value.iter().all(|each| each.is_finite()) =>
                {
                    values.push(Some(value));
                    responses.push(response);
                    slope_atoms.push((position, atom));
                }
                Ok(_) => {
                    values.push(None);
                    responses.push(None);
                }
                Err(error @ CausalError::Cancelled { .. }) => return Err(error),
                Err(error) => {
                    if first_refusal.is_none()
                        && matches!(
                            error,
                            CausalError::Unsupported { .. } | CausalError::Support { .. }
                        )
                    {
                        first_refusal = Some(error);
                    }
                    values.push(None);
                    responses.push(None);
                }
            }
        }
        if slope_atoms.is_empty() {
            return Err(first_refusal.unwrap_or_else(|| {
                CausalError::not_identified(
                    IdentificationStatus::NotIdentified,
                    false,
                    &format!("no atom of the graph posterior evaluates the {label}"),
                )
            }));
        }

        // The structural mixture and its masses; a vector answer is accounted by its first
        // coordinate and mixed coordinate by coordinate below.
        let evals: Vec<ScalarAtomEval> = templates
            .iter()
            .zip(&values)
            .map(|(template, value)| ScalarAtomEval {
                key: template.key,
                status: template.identification.status,
                estimand: target_estimand(&template.estimand),
                value: value.as_ref().and_then(|each| each.first().copied()),
            })
            .collect();
        let proofs: Vec<CachedGraphPosteriorAtomIdentification> = templates
            .iter()
            .map(|template| CachedGraphPosteriorAtomIdentification {
                key: template.key,
                estimand: target_estimand(&template.estimand),
                identification: template.identification.clone(),
                transport: None,
            })
            .collect();
        let mut mixed = mix_atom_evals(
            &evals,
            AtomMixInput {
                graphs: &graphs,
                proofs: &proofs,
                unidentified_mass,
                full_mass_scope: true,
                truncated_atoms: 0,
                bounds: AtomBoundRule::Point,
                estimand_fallback: true,
            },
            |n, mixable| {
                format!(
                    "{label} graph-posterior mix: no evaluable atom value (evals={n}, \
                     mixable={mixable})"
                )
            },
            || format!("{label} graph-posterior mix: missing estimand"),
        )?;
        mixed.identification.status = resolve_posterior_status(
            mixed.identification.status,
            mixed.policy,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                unevaluable: mixed.mixture.unevaluable_mass,
                ..PosteriorMassSummary::default()
            },
        );

        let weights: Vec<f64> = slope_atoms
            .iter()
            .map(|(position, _)| key_weights.get(templates[*position].key))
            .collect();
        let atom_values: Vec<Vec<f64>> = slope_atoms
            .iter()
            .map(|(position, _)| values[*position].clone().unwrap_or_default())
            .collect();
        let center = point_uncertainty::weighted_outputs(&atom_values, &weights).ok_or_else(
            || CausalError::Compile {
                message: format!(
                    "{label} graph-posterior mix: the atoms' values differ in shape or carry no \
                     mass"
                ),
            },
        )?;
        let primary_position = slope_atoms
            .iter()
            .zip(&weights)
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map_or(slope_atoms[0].0, |(slope, _)| slope.0);
        let span = super::temporal_class_block_span(
            slope_atoms.iter().map(|(position, _)| &templates[*position].indexer),
        );
        let positions: Vec<usize> = slope_atoms.iter().map(|(position, _)| *position).collect();
        let frozen: Vec<FrozenAtom> =
            slope_atoms.into_iter().map(|(_, atom)| SlopeAtom::freeze(atom)).collect();
        Ok(DbnLane {
            point: point_kind,
            templates,
            positions,
            frozen,
            weights,
            responses,
            mixed,
            center,
            primary_position,
            span,
        })
    }

    /// Run the derivative refuter state on every contributing atom of one horizon and mix the
    /// reports by posterior weight.
    fn refute_dbn_lane(
        &self,
        data: &TimeSeriesData,
        lane: &DbnLane<'_>,
        ctx: &ExecutionContext,
    ) -> Result<LaneRefutation, CausalError> {
        let DbnPoint::Derivative { query, probe } = &lane.point else {
            return Err(CausalError::Unsupported {
                message: "a temporal conditional or cross-world point over a graph posterior is \
                          validated by the functional refuter state of the whole procedure",
            });
        };
        let mut states = super::posterior_witness_state::AtomWitnessStates::default();
        for (&position, &weight) in lane.positions.iter().zip(&lane.weights) {
            let template = &lane.templates[position];
            let Some(response) = lane.responses[position].as_ref() else { continue };
            let site = super::temporal_response_refutation::PulseWitnessSite {
                key: template.key,
                horizon: probe.horizon_steps,
                weight: 1.0,
                witness: probe.clone(),
                estimand: template.estimand.clone(),
                indexer: template.indexer.clone(),
                status: template.identification.status,
                assumptions: template.identification.required_assumptions.clone(),
            };
            states.push_derivative(
                self,
                data,
                (template.key, weight),
                (query, response),
                &[site],
                ctx,
            )?;
        }
        Ok(states.finish())
    }

    /// Mix a conditional effect, a derivative (at one or several horizons) or a
    /// cross-world contrast over the `TemporalDag` atoms of a dynamic graph posterior.
    ///
    /// Several horizons are fitted separately (each has its own identification, atoms and
    /// masses, and horizons are never mixed) and their atoms' regressions are refit together on
    /// the same circular-block resamples, or drawn jointly, so the band across horizons is one
    /// joint band.
    ///
    /// # Errors
    ///
    /// A validation suite on a conditional or cross-world point, a mapped coefficient prior, no
    /// atom that identifies and evaluates the point at a horizon, or a cancelled run.
    #[allow(
        clippy::too_many_lines,
        reason = "one mixture: the horizons, their regressions, the masses and the shared interval read one frozen posterior"
    )]
    pub(super) fn execute_dbn_route_graph_posterior(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let points = DbnPoint::of_all(&self.query)?;
        let first_point = points.first().ok_or(CausalError::Compile {
            message: "dynamic graph posterior has no requested point".into(),
        })?;
        let derivative = matches!(first_point, DbnPoint::Derivative { .. });
        if self.refute != RefuteSuite::None && !derivative {
            return Err(CausalError::Unsupported {
                message: "a temporal conditional or cross-world point over a graph posterior is \
                          validated by the functional refuter state of the whole procedure, not \
                          by an atom's executor",
            });
        }
        let prior_scale = isotropic_prior_scale(
            &self.inference,
            match first_point {
                DbnPoint::Conditional(_) => CONDITIONAL_PRIOR_REFUSAL,
                DbnPoint::Derivative { .. } => DERIVATIVE_PRIOR_REFUSAL,
                DbnPoint::CrossWorld { .. } => CROSS_WORLD_PRIOR_REFUSAL,
            },
        )?;
        let label = first_point.label();
        let variables: Vec<VariableId> =
            data.schema().variables().iter().map(|variable| variable.id).collect();

        // Samples that carry one (contemporaneous, lag) template share one atom.
        let lag_masks = gp.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
            message: "a dynamic graph posterior carries per-atom lag masks".into(),
        })?;
        let mut first_of: HashMap<(AdjacencyMask, u64), usize> = HashMap::new();
        let mut keys: Vec<u64> = Vec::with_capacity(gp.n_graphs);
        for index in 0..gp.n_graphs {
            let first = *first_of.entry((gp.atoms[index], lag_masks[index])).or_insert(index);
            keys.push(u64::try_from(first).map_err(|_| CausalError::Compile {
                message: "a dynamic graph posterior has too many atoms for atom keys".into(),
            })?);
        }

        let mut lanes: Vec<DbnLane<'_>> = Vec::with_capacity(points.len());
        for point in points {
            lanes.push(self.fit_dbn_lane(
                data,
                gp,
                (&keys, &variables),
                point,
                prior_scale,
                ctx,
            )?);
        }
        let lane_count = lanes.len();
        let width = lanes[0].center.len();

        // Per-atom refuter states, one horizon at a time (horizons are not mixed).
        let mut refuted = (Vec::new(), Vec::new(), Vec::new());
        if derivative && self.refute != RefuteSuite::None {
            for lane in &lanes {
                let (reports, checks, notes) = self.refute_dbn_lane(data, lane, ctx)?;
                refuted.0.extend(reports);
                refuted.1.extend(checks);
                refuted.2.extend(notes);
            }
        }

        // The horizon with the least identified mass names the result's masses.
        let primary_lane = lanes
            .iter()
            .enumerate()
            .min_by(|left, right| {
                left.1
                    .mixed
                    .mixture
                    .identified_mass
                    .total_cmp(&right.1.mixed.mixture.identified_mass)
            })
            .map_or(0, |(index, _)| index);

        // Every horizon's atoms enter one shared resample (or one joint draw); the weights are
        // reapplied within each horizon.
        let mut frozen: Vec<FrozenAtom> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut lane_of: Vec<usize> = Vec::new();
        let mut contributing = 0;
        let mut structural_span = 1;
        let mut center: Vec<f64> = Vec::with_capacity(lane_count * width);
        for (index, lane) in lanes.iter_mut().enumerate() {
            contributing += lane.frozen.len();
            structural_span = structural_span.max(lane.span);
            center.extend_from_slice(&lane.center);
            for (atom, weight) in lane.frozen.drain(..).zip(&lane.weights) {
                frozen.push(atom);
                weights.push(*weight);
                lane_of.push(index);
            }
        }
        let lane_totals: Vec<f64> = (0..lane_count)
            .map(|index| {
                weights
                    .iter()
                    .zip(&lane_of)
                    .filter(|(_, lane)| **lane == index)
                    .map(|(w, _)| *w)
                    .sum()
            })
            .collect();
        let stack = |values: &[Vec<f64>]| -> Option<Vec<f64>> {
            let mut out = Vec::with_capacity(lane_count * width);
            for index in 0..lane_count {
                let (rows, row_weights): (Vec<Vec<f64>>, Vec<f64>) = values
                    .iter()
                    .zip(&weights)
                    .zip(&lane_of)
                    .filter(|(_, lane)| **lane == index)
                    .map(|((row, weight), _)| (row.clone(), *weight))
                    .unzip();
                out.extend(point_uncertainty::weighted_outputs(&rows, &row_weights)?);
            }
            Some(out)
        };
        let point = center[0];
        let primary = lanes[primary_lane].primary();
        let assumptions = primary.identification.required_assumptions.clone();
        let query = &self.query;
        let point_kind = &lanes[0].point;
        let evaluate = |_: usize, source: &dyn SlopeSource| -> Result<Vec<f64>, CausalError> {
            match point_kind {
                DbnPoint::Conditional(_) => source.conditional_effect().map(|value| vec![value]),
                DbnPoint::Derivative { query: response_query, .. } => {
                    let value =
                        temporal_linear_derivative_value(&response_query.functional, source)?;
                    point_uncertainty::response_values(&value).ok_or_else(|| CausalError::Compile {
                        message: "a temporal derivative replicate has no numeric value".into(),
                    })
                }
                DbnPoint::CrossWorld { contrast, .. } => {
                    let fit = |outcome: VariableId,
                               treatments: &[VariableId],
                               contemporaneous: &[VariableId]| {
                        source.slopes(outcome, treatments, contemporaneous)
                    };
                    cross_world_point(query, &fit, *contrast).map(|value| vec![value])
                }
            }
        };
        let mut diagnostics: Vec<Diagnostic> = Vec::new();
        if let DbnPoint::Conditional(effect) = point_kind {
            diagnostics.extend(super::temporal_effect_core::modifier_set_coverage_note(
                effect.effect_modifiers.len(),
            ));
        }
        let (block_stream, posterior_stream) = match point_kind {
            DbnPoint::Conditional(_) => (
                point_uncertainty::CONDITIONAL_BLOCK_STREAM,
                point_uncertainty::CONDITIONAL_POSTERIOR_STREAM,
            ),
            DbnPoint::Derivative { .. } => (
                point_uncertainty::DERIVATIVE_BLOCK_STREAM,
                point_uncertainty::DERIVATIVE_POSTERIOR_STREAM,
            ),
            DbnPoint::CrossWorld { .. } => (
                point_uncertainty::CROSS_WORLD_BLOCK_STREAM,
                point_uncertainty::CROSS_WORLD_POSTERIOR_STREAM,
            ),
        };
        // The mixed derivative answer, shaped as the primary atom's at each horizon (several
        // horizons are one value per horizon), once its band is set.
        let mut response = if derivative {
            let mut per_horizon = Vec::with_capacity(lane_count);
            for lane in &lanes {
                let DbnPoint::Derivative { query: response_query, .. } = &lane.point else {
                    continue;
                };
                let base = lane.responses[lane.primary_position].clone().ok_or_else(|| {
                    CausalError::Compile {
                        message: "the primary dynamic atom carries no derivative response".into(),
                    }
                })?;
                per_horizon.push(mixed_derivative_response(
                    base,
                    &lane.center,
                    (&response_query.functional, lane.mixed.identification.status),
                ));
            }
            if per_horizon.len() == 1 {
                per_horizon.pop()
            } else {
                let DbnPoint::Derivative { query: response_query, .. } = point_kind else {
                    unreachable!("derivative lanes are derivative points")
                };
                let mut combined = super::temporal_path::combine_horizon_derivatives(
                    &response_query.functional,
                    &per_horizon,
                )?;
                combined.identification_status = lanes[primary_lane].mixed.identification.status;
                Some(combined)
            }
        } else {
            None
        };
        let mut posterior = None;
        let mut replicates_ok = None;
        let mut cancelled = false;
        let mut estimate = if let InferenceMode::Bayesian(cfg) = &self.inference {
            let draws = point_uncertainty::point_posterior_draws(
                &frozen,
                point_uncertainty::PointPosteriorPlan {
                    prior_scale: cfg.prior_scale,
                    draws: cfg.n_draws,
                    structural_span,
                    stream_base: posterior_stream,
                },
                ctx,
                evaluate,
            )?;
            diagnostics.push(point_uncertainty::posterior_diagnostic(&draws, label));
            let n_draws = draws.atom_draws.first().map_or(0, Vec::len);
            let pooled: Vec<Vec<f64>> = (0..n_draws)
                .map(|draw| {
                    let mut row = Vec::with_capacity(lane_count * width);
                    for (index, &lane_total) in lane_totals.iter().enumerate().take(lane_count) {
                        for coordinate in 0..width {
                            row.push(
                                draws
                                    .atom_draws
                                    .iter()
                                    .zip(&weights)
                                    .zip(&lane_of)
                                    .filter(|(_, lane)| **lane == index)
                                    .map(|((atom, weight), _)| {
                                        weight / lane_total * atom[draw][coordinate]
                                    })
                                    .sum::<f64>(),
                            );
                        }
                    }
                    row
                })
                .collect();
            if contributing > 1 {
                diagnostics.push(Diagnostic::new(
                    DBN_ROUTE_POSTERIOR_CODE,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "pooled posterior: the coefficients of every regression of every atom (at \
                     every horizon) are one joint draw, draw d of every atom enters draw d of \
                     the functional, and draw d of the mixture is the posterior-weighted sum of \
                     the atoms' draw d within each horizon",
                ));
            }
            if let Some(response) = response.as_mut() {
                point_uncertainty::attach_posterior_response_band(response, &pooled, &center);
            }
            let first_column: Vec<f64> = pooled.iter().map(|draw| draw[0]).collect();
            let pooled_posterior = point_uncertainty::scalar_posterior(
                &first_column,
                "dbn_graph_posterior_point",
                lanes[primary_lane].mixed.identification.status,
                assumptions.clone(),
                draws.tempering,
            )?;
            let sd = pooled_posterior.summaries.sd[0];
            if width == 1 && lane_count == 1 {
                let mut pooled_posterior = pooled_posterior;
                pooled_posterior.unidentified_mass =
                    lanes[primary_lane].mixed.mixture.unidentified_mass;
                pooled_posterior.unevaluable_mass =
                    lanes[primary_lane].mixed.mixture.unevaluable_mass;
                posterior = Some(pooled_posterior);
            }
            EffectEstimate::new(point, sd, assumptions, OverlapPolicy::ExplicitOverride)
        } else {
            let replicates = point_uncertainty::point_block_replicates(
                &frozen,
                point_uncertainty::PointBlockPlan {
                    replicates: self.bootstrap_replicates,
                    stream_base: block_stream,
                    structural_span,
                },
                false,
                ctx,
                evaluate,
                stack,
            )?;
            diagnostics.extend(point_uncertainty::block_diagnostics(
                replicates.as_ref(),
                self.bootstrap_replicates,
                antecedent_estimate::CircularBlockFamily::Mixture,
                "every regression of every contributing atom (at every horizon) is refit on each \
                 replicate and the frozen posterior weights are reapplied to the atoms' values \
                 within each horizon",
            ));
            replicates_ok =
                replicates.as_ref().map(point_uncertainty::PointBlockReplicates::completed);
            cancelled = replicates.as_ref().is_some_and(|replicates| replicates.draws.cancelled);
            if let (Some(response), Some(replicates)) = (response.as_mut(), replicates.as_ref()) {
                point_uncertainty::attach_block_response_band(response, replicates, &center);
            }
            point_uncertainty::block_scalar_estimate(
                point,
                replicates.as_ref(),
                assumptions,
                antecedent_estimate::CircularBlockFamily::Mixture,
            )
        };
        if let Some(response) = response.as_ref() {
            // A derivative's scalar is the response's own summary; a vector has none.
            let (scalar, _) = super::response_path::response_scalar_summary(response);
            estimate.ate = scalar;
        }

        // The identification and estimand the result names are the primary atom's.
        let primary_lane_ref = &lanes[primary_lane];
        let primary = primary_lane_ref.primary();
        let (treatment, outcome) = primary_lane_ref.point.treatment_outcome();
        let mut identification = primary.identification.clone();
        let mut estimand = primary.estimand.clone();
        if let DbnPoint::CrossWorld { contrast, .. } = &primary_lane_ref.point {
            identification = template_scoped_cross_world(
                identification,
                &primary.indexer,
                self.query.clone(),
                *contrast,
            );
            estimand = identification.estimands[0].clone();
        }
        identification.status = primary_lane_ref.mixed.identification.status;
        let mixture = primary_lane_ref.mixed.mixture.clone();
        let policy = primary_lane_ref.mixed.policy;
        if lane_count > 1 {
            let listed = lanes
                .iter()
                .map(|lane| {
                    let horizon = match &lane.point {
                        DbnPoint::Derivative { probe, .. } => probe.horizon_steps,
                        _ => 0,
                    };
                    format!(
                        "h{horizon}:identified={}:unidentified={}:unevaluable={}",
                        lane.mixed.mixture.identified_mass,
                        lane.mixed.mixture.unidentified_mass,
                        lane.mixed.mixture.unevaluable_mass
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            diagnostics.push(Diagnostic::new(
                DBN_ROUTE_POSTERIOR_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "each horizon is identified, fitted and mixed on its own and horizons are \
                     never mixed; the structural mixture is that of the horizon with the least \
                     identified mass and the band is joint across horizons (masses: {listed})"
                ),
            ));
        }
        let n_draws = posterior
            .as_ref()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        diagnostics.push(
            Diagnostic::new(
                DBN_ROUTE_POSTERIOR_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "each TemporalDag atom was identified on its own graph and answered the \
                     {label} with the linear unfolded producer of the explicit cell; the atoms \
                     are mixed by posterior probability over the {contributing} atom(s) that \
                     identify and evaluate it; unidentified_mass={} and unevaluable_mass={} are \
                     never redistributed",
                    mixture.unidentified_mass, mixture.unevaluable_mass,
                ),
            )
            .with_fields([
                ("atoms", contributing.to_string()),
                ("unidentified_mass", mixture.unidentified_mass.to_string()),
                ("unevaluable_mass", mixture.unevaluable_mass.to_string()),
            ]),
        );
        super::push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            policy,
            mixture.identified_mass,
            mixture.unidentified_mass,
            mixture.unevaluable_mass,
            mixture.subsampled_out_mass,
        );
        if self.inference.is_bayesian() && posterior.is_none() {
            diagnostics.push(graph_posterior_withheld_diagnostic());
        }
        diagnostics.extend(refuted.2);
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    EstimatorId::TemporalLinearAdjustment,
                ),
                treatment,
                outcome,
            )
            .refutations(refuted.0)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(replicates_ok)
            .cancelled(cancelled)
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.dbn_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.dbn_posterior.point",
                    "estimate.temporal.linear.adjustment",
                )),
                diagnostics: Some(diagnostics),
                structural_response: Some(mixture),
                response,
                n_draws,
                posterior,
                predictive_checks: refuted.1,
                ..Default::default()
            }),
        );
        if self.inference.is_frequentist() {
            result.posterior = None;
        }
        result.rebind_interval(self.inference.is_bayesian());
        Ok(result)
    }
}
