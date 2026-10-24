//! Lowering of the static average-effect, conditional-effect, interventional-distribution
//! and path-specific cells.
//!
//! Layer one states the query's scientific target: the contrast of its two intervention
//! worlds, conditioned on the effect modifiers for a conditional effect; the mediation
//! target of a path-restricted effect; or the typed interventional-distribution
//! functional, which no world-indexed form states. Layer two identifies once at compile
//! through the prepare-side identification and retains it, so preparation installs it;
//! a sharp regression discontinuity identifies on each execution. Layer three binds the
//! resolved identifier and estimator, the realization the inference mode selects (an
//! identified-set envelope when frequentist identification is partial), the fitted models
//! with their roles and variables, the cross-fitting split, and the resampling the
//! configured estimator performs under the declared sampling design. Layer five quotes
//! the adjustment design a linear adjustment fits and states, with the reason, when a
//! route fits no single quoted adjustment design. Layer six names the claim kind the
//! retained identification licenses.
//!
//! A query whose actions have no portable world-indexed statement (a callback action)
//! lowers to its typed functional, keyed by the canonical digest of the query.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchema, Conditioning, CrossWorldPremise,
    DeclaredSamplingDesign, GraphFamily, IdentificationStatus, InferenceSelection, Intervention,
    InterventionContrast, InterventionWorld, InterventionalDistributionQuery, MediationArms,
    MediationComponent, MediationTarget, ModelAssumptions, ObservationUnit, OutcomeFunctional,
    OutcomeQuantity, PartialIdentificationEnvelope, PathSpecificEffectQuery, QuantityUnits,
    ScientificTarget, ScientificTargetError, ScientificWorldId, TargetForm, TargetPopulation,
    TimeFrame, VariableId, WorldAction, WorldBinding, WorldCollection,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::util::SamplingDeclaration;

use super::{LoweringContext, quote_adjustment_width, retain_identification};
use crate::Study;
use crate::accepted::GraphClass;
use crate::analysis::IdentificationCache;
use crate::analysis::compiled_linear::CompiledLinearExecution;
use crate::analysis::execute::RouteInput;
use crate::analysis::prepared::CachedStaticIdentification;
use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;
use crate::planner::lowered::{
    ClaimDemand, DesignShape, FunctionalKind, IdentificationLayer, LoweredParts, LoweredProgram,
    LoweringDepth, PhysicalLayer, ProblemTarget, ScientificProblem, UncertaintyReading,
    WidthSource, linear_gaussian_physical_operators, linear_gaussian_program_operators,
};
use crate::planner::statistical::{
    DeclaredProgram, FoldPlan, ModelRole, ModelSlot, Realization, ResampleUnit, StatisticalProgram,
};
use crate::{EstimatorId, IdentifierId, InferenceMode};

const CONTROL_WORLD: u32 = 0;
const ACTIVE_WORLD: u32 = 1;
/// Observation unit of a static tabular target.
const ROW_UNIT: &str = "row";
/// Why a discrete plug-in route has no design width.
const PLUG_IN_FITS_NO_DESIGN: &str =
    "discrete plug-in evaluation of identified factors fits no design matrix";
/// Why a linear adjustment with an uncoded categorical adjustment variable is not quoted.
const UNCODED_ADJUSTMENT: &str = "a categorical adjustment variable has no declared contrast \
                                  coding; the fit encodes it and sizes the design at execution";
/// Why a linear adjustment without one compile-time adjustment set is not quoted.
const ADJUSTMENT_SET_PER_GRAPH: &str = "no single adjustment set is fixed at compile (a class or \
                                        mixed graph chooses one per member graph, or \
                                        identification was deferred); the design is sized at \
                                        execution";
/// Why a regression-discontinuity route is not quoted.
const LOCAL_FITS_AT_EXECUTION: &str =
    "two one-sided local fits inside the bandwidth, sized at execution";
/// Why a weighting, matching, instrument, learner or Bayesian route is not quoted.
const NUISANCE_MODELS: &str = "the estimator fits its propensity, outcome, instrument or \
                               learner models on their own bases, sized at execution; no single \
                               adjustment design is quoted";

/// The query of one static effect cell, with the rows it reads.
#[derive(Clone, Copy)]
enum EffectQuery<'a> {
    /// Average contrast of two intervention worlds.
    Average(&'a AverageEffectQuery),
    /// The same contrast conditioned on its effect modifiers.
    Conditional(&'a AverageEffectQuery),
    /// Law of the outcomes under the query's interventions.
    Distribution(&'a InterventionalDistributionQuery),
    /// Effect along the paths through the declared intermediate nodes.
    PathSpecific(&'a PathSpecificEffectQuery),
}

impl<'a> EffectQuery<'a> {
    fn of(input: &RouteInput<'a>) -> Option<(Self, &'a TabularData)> {
        Some(match *input {
            RouteInput::StaticAteDag { data, query, .. }
            | RouteInput::StaticAtePag { data, query, .. }
            | RouteInput::StaticAteCpdag { data, query, .. }
            | RouteInput::StaticAteMixed { data, query } => (Self::Average(query), data),
            RouteInput::ConditionalDag { data, query, .. }
            | RouteInput::ConditionalCpdag { data, query, .. }
            | RouteInput::ConditionalPag { data, query, .. }
            | RouteInput::ConditionalMixed { data, query } => {
                (Self::Conditional(&query.inner), data)
            }
            RouteInput::DistributionDag { data, query, .. }
            | RouteInput::DistributionMixed { data, query } => (Self::Distribution(query), data),
            RouteInput::PathSpecificDag { data, query, .. } => (Self::PathSpecific(query), data),
            _ => return None,
        })
    }

    /// Treatment, outcome, effect modifiers and declared intermediate nodes.
    fn roles(self) -> Roles {
        match self {
            Self::Average(query) => Roles {
                treatment: Some(query.treatment),
                outcome: Some(query.outcome),
                ..Roles::default()
            },
            Self::Conditional(query) => Roles {
                treatment: Some(query.treatment),
                outcome: Some(query.outcome),
                modifiers: query.effect_modifiers.to_vec(),
                ..Roles::default()
            },
            Self::Distribution(query) => Roles {
                treatment: query.interventions.first().and_then(Intervention::primary_variable),
                outcome: query.outcomes.first().copied(),
                modifiers: query.conditioning.to_vec(),
                ..Roles::default()
            },
            Self::PathSpecific(query) => Roles {
                treatment: Some(query.treatment),
                outcome: Some(query.outcome),
                mediators: query.path_nodes.to_vec(),
                ..Roles::default()
            },
        }
    }
}

/// Variables the fitted models read, by role.
#[derive(Clone, Debug, Default)]
struct Roles {
    treatment: Option<VariableId>,
    outcome: Option<VariableId>,
    adjustment: Vec<VariableId>,
    instruments: Vec<VariableId>,
    mediators: Vec<VariableId>,
    modifiers: Vec<VariableId>,
}

impl Roles {
    /// Adopt the roles of the selected estimand.
    fn bind(&mut self, cache: &CachedStaticIdentification) {
        let estimand = &cache.estimand;
        self.adjustment = estimand.adjustment_set.to_vec();
        self.instruments = estimand.instruments.to_vec();
        if !estimand.mediators.is_empty() {
            self.mediators = estimand.mediators.to_vec();
        }
    }

    fn treatment(&self) -> Vec<VariableId> {
        self.treatment.into_iter().collect()
    }

    fn joined(parts: &[&[VariableId]]) -> Vec<VariableId> {
        let mut joined = Vec::new();
        for part in parts {
            for &variable in *part {
                if !joined.contains(&variable) {
                    joined.push(variable);
                }
            }
        }
        joined
    }
}

/// Lower one cell of this family.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    let Some((query, data)) = cx.input.as_ref().and_then(EffectQuery::of) else {
        return Err(super::not_lowered_by("static effect", cx));
    };
    let study = cx.study;
    let plan = cx.plan;
    let record = &plan.logical.record;
    let linear = plan.linear.as_deref();
    let identifier =
        record.identifier.as_deref().and_then(|name| name.parse::<IdentifierId>().ok());
    let estimator = record.estimator.as_deref().and_then(|name| name.parse::<EstimatorId>().ok());

    let target = match linear {
        Some(linear) => ProblemTarget::Target(linear.program.target().clone()),
        None => problem_target(study, query, data.schema())
            .unwrap_or_else(|_| unportable_target(study, query)),
    };
    let source_digest = linear.map(|linear| linear.program.source_digest());
    let scientific = ScientificProblem {
        target,
        plan_id: Arc::clone(&record.plan_id),
        data: record.data_classification,
        graph_class: study.graph.class().as_str(),
        query_variables: Arc::clone(&record.query_variables),
        source_digest,
        support_coordinate: crate::analysis::contract::matrix_coordinate(study).map(Arc::from),
        licensed: matches!(study.support_status, Some(crate::support::CellStatus::Licensed)),
    };

    let identification = identification_layer(cx, linear, estimator)?;
    let caches = match &identification {
        IdentificationLayer::Retained(retained) => retained.caches(),
        IdentificationLayer::Deferred { .. }
        | IdentificationLayer::PerExecution { .. }
        | IdentificationLayer::NonGraphical(_)
        | IdentificationLayer::NotClaimed { .. } => &[],
    };
    let static_cache = caches.iter().find_map(|cache| match cache {
        IdentificationCache::Static(cache) => Some(cache.as_ref()),
        _ => None,
    });
    let status = identification_status(caches);
    let mut roles = query.roles();
    if let Some(cache) = static_cache {
        roles.bind(cache);
    }
    let bayesian = matches!(study.inference, InferenceMode::Bayesian(_));
    let spec = configured_spec(study);

    let design = design_shape(cx, linear, estimator, static_cache, spec)?;
    let (replicates, sampling_design) = resampling_request(study, linear, estimator, spec);
    let mut statistical = StatisticalProgram::declared(DeclaredProgram {
        identifier,
        estimator,
        inference: &study.inference,
        bootstrap_replicates: if bayesian { 0 } else { replicates },
        sampling_design,
        design,
        operators: match linear {
            Some(linear) => linear.program.operators().to_vec(),
            None => linear_gaussian_program_operators()?,
        },
    })?;
    statistical.models = Arc::from(model_slots(study, estimator, spec, &roles));
    statistical.folds = fold_plan(study, estimator, spec);
    if !bayesian && status == Some(IdentificationStatus::PartiallyIdentified) {
        let name = estimator.map_or("unnamed", |estimator| estimator.as_str());
        statistical.realization = Realization::Envelope(
            PartialIdentificationEnvelope::new(format!("identified_set.{name}"))
                .map_err(|error| CausalError::Compile { message: error.to_string() })?,
        );
    }

    let physical = PhysicalLayer {
        operators: Arc::from(physical_operators(cx, linear, design, &statistical)?),
        design,
    };
    let claim = ClaimDemand {
        uncertainty: match statistical.realization() {
            Realization::Frequentist(_) => UncertaintyReading::Sampling,
            Realization::Bayesian(_) | Realization::GeneralizedBayes(_) => {
                UncertaintyReading::Posterior
            }
            Realization::Envelope(_) => UncertaintyReading::Envelope,
            Realization::ExactPoint { .. } => UncertaintyReading::None,
        },
        kind: status.map(claim_kind),
    };
    let evidence = super::decision::evidence_program(cx, source_digest, &statistical)?;
    Ok(LoweredParts {
        cell: cx.cell,
        depth: LoweringDepth::Complete,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

/// Layer one for a route that retained no linear operation.
fn problem_target(
    study: &Study,
    query: EffectQuery<'_>,
    schema: &CausalSchema,
) -> Result<ProblemTarget, CausalError> {
    let family = graph_family(study);
    let inference = match study.inference {
        InferenceMode::Frequentist => InferenceSelection::Frequentist,
        InferenceMode::Bayesian(_) => InferenceSelection::Bayesian,
    };
    Ok(match query {
        EffectQuery::Average(query) => ProblemTarget::Target(contrast_target(
            query,
            Conditioning::unconditional(),
            family,
            inference,
            schema,
        )?),
        EffectQuery::Conditional(query) => ProblemTarget::Target(contrast_target(
            query,
            Conditioning::on(query.effect_modifiers.iter().copied()),
            family,
            inference,
            schema,
        )?),
        // A law over the outcomes, not a functional of one outcome: no world form states it.
        EffectQuery::Distribution(query) => super::query_functional(
            study,
            FunctionalKind::InterventionalDistribution,
            query.outcomes.first().copied(),
        ),
        EffectQuery::PathSpecific(query) => {
            ProblemTarget::Target(path_specific_target(query, family, inference, schema)?)
        }
    })
}

/// Layer one for a query whose worlds the target algebra refuses (a callback action
/// or another non-portable intervention): its typed functional, keyed by the query.
fn unportable_target(study: &Study, query: EffectQuery<'_>) -> ProblemTarget {
    let (kind, outcome) = match query {
        EffectQuery::Average(query) | EffectQuery::Conditional(query) => {
            (FunctionalKind::InterventionContrast, Some(query.outcome))
        }
        EffectQuery::Distribution(query) => {
            (FunctionalKind::InterventionalDistribution, query.outcomes.first().copied())
        }
        EffectQuery::PathSpecific(query) => (FunctionalKind::PathSpecific, Some(query.outcome)),
    };
    super::query_functional(study, kind, outcome)
}

/// Graph family the target is stated under. A co-determined tier background states
/// its own family; an unknown within-tier order keeps the supplied graph's.
fn graph_family(study: &Study) -> GraphFamily {
    let codetermined = |background: &antecedent_graph::TieredBackground| {
        background.within_tier == antecedent_graph::WithinTier::CoDetermined
    };
    if study.tiered.as_ref().is_some_and(codetermined) {
        return GraphFamily::CoDetermined;
    }
    match study.graph.class() {
        GraphClass::Dag => GraphFamily::Dag,
        GraphClass::Admg => GraphFamily::Admg,
        GraphClass::Cpdag => GraphFamily::Cpdag,
        GraphClass::Pag => GraphFamily::Pag,
        GraphClass::TemporalDag => GraphFamily::TemporalDag,
        GraphClass::TemporalCpdag => GraphFamily::TemporalCpdag,
        GraphClass::TemporalPag => GraphFamily::TemporalPag,
        GraphClass::TemporalAdmg => GraphFamily::TemporalAdmg,
        GraphClass::RandomizedTrial => GraphFamily::Graphless,
    }
}

/// Contrast of the query's active and control worlds of `functional` of the outcome.
fn contrast_target(
    query: &AverageEffectQuery,
    conditioning: Conditioning,
    family: GraphFamily,
    inference: InferenceSelection,
    schema: &CausalSchema,
) -> Result<ScientificTarget, CausalError> {
    let binding = world_binding(
        query.outcome,
        query.outcome_functional.clone(),
        &query.target_population,
        conditioning,
        schema,
    )?;
    let time = TimeFrame::static_decision();
    let control = world(CONTROL_WORLD, &query.control, true, &binding, &time)?;
    let active = world(ACTIVE_WORLD, &query.active, false, &binding, &time)?;
    let worlds = WorldCollection::new([control, active]).map_err(target_error)?;
    let contrast = InterventionContrast::difference(
        worlds,
        ScientificWorldId::from_raw(ACTIVE_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
    )
    .map_err(target_error)?;
    Ok(ScientificTarget::new(
        TargetForm::Contrast(contrast),
        ModelAssumptions::new(family, AssumptionSet::new()),
        inference,
    ))
}

/// The path-restricted natural effect. The identifier reads a mediation query's
/// mediators as its path nodes, so declared nodes state the mediated component of a
/// mediation target with its cross-world premises. No declared node selects every
/// directed path: the total effect, the contrast of the two worlds.
fn path_specific_target(
    query: &PathSpecificEffectQuery,
    family: GraphFamily,
    inference: InferenceSelection,
    schema: &CausalSchema,
) -> Result<ScientificTarget, CausalError> {
    if query.path_nodes.is_empty() {
        let total = AverageEffectQuery::new(
            query.treatment,
            query.outcome,
            Vec::<VariableId>::new(),
            query.control.clone(),
            query.active.clone(),
            query.target_population.clone(),
        );
        return contrast_target(&total, Conditioning::unconditional(), family, inference, schema);
    }
    let binding = world_binding(
        query.outcome,
        OutcomeFunctional::Mean,
        &query.target_population,
        Conditioning::unconditional(),
        schema,
    )?;
    let arms = MediationArms::new(
        query.treatment,
        query.path_nodes.iter().copied(),
        query.control.clone(),
        query.active.clone(),
        MediationComponent::Mediated,
    )
    .map_err(target_error)?;
    let mediation = MediationTarget::new(
        binding,
        TimeFrame::static_decision(),
        None,
        arms,
        [
            CrossWorldPremise::Composition,
            CrossWorldPremise::CrossWorldIndependence,
            CrossWorldPremise::PathSpecificNested,
        ],
    )
    .map_err(target_error)?;
    Ok(ScientificTarget::new(
        TargetForm::Mediation(mediation),
        ModelAssumptions::new(family, AssumptionSet::new()),
        inference,
    ))
}

fn world_binding(
    outcome: VariableId,
    functional: OutcomeFunctional,
    population: &TargetPopulation,
    conditioning: Conditioning,
    schema: &CausalSchema,
) -> Result<WorldBinding, CausalError> {
    let units = QuantityUnits::new(schema.get(outcome)?.unit.as_deref().unwrap_or("unspecified"))
        .map_err(target_error)?;
    WorldBinding::new(
        population.clone(),
        ObservationUnit::new(ROW_UNIT).map_err(target_error)?,
        OutcomeQuantity::new(outcome, functional, units).map_err(target_error)?,
        conditioning,
    )
    .map_err(target_error)
}

fn world(
    raw: u32,
    intervention: &Intervention,
    reference: bool,
    binding: &WorldBinding,
    time: &TimeFrame,
) -> Result<InterventionWorld, CausalError> {
    let action = WorldAction::at_decision([intervention.clone()]).map_err(target_error)?;
    InterventionWorld::new(
        ScientificWorldId::from_raw(raw),
        action,
        reference,
        binding.clone(),
        time.clone(),
    )
    .map_err(target_error)
}

#[allow(clippy::needless_pass_by_value, reason = "Result::map_err consumes its error")]
fn target_error(error: ScientificTargetError) -> CausalError {
    CausalError::Compile { message: format!("static effect scientific target: {error}") }
}

/// Layer two. Compile identifies what preparation would and retains it. A refusal
/// leaves the identification to the stage that consumes it, which reports the same
/// refusal it always has, so compile answers as it did before.
fn identification_layer(
    cx: &LoweringContext<'_>,
    linear: Option<&CompiledLinearExecution>,
    estimator: Option<EstimatorId>,
) -> Result<IdentificationLayer, CausalError> {
    if linear.is_none()
        && matches!(estimator, Some(EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear))
    {
        return Ok(IdentificationLayer::PerExecution {
            reason: super::RD_IDENTIFIES_PER_EXECUTION,
        });
    }
    let product = linear.map(|linear| linear.program.identification().clone());
    match retain_identification(cx, product) {
        Ok(layer) => Ok(layer),
        Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. })) => Err(error),
        Err(error) if linear.is_some() => Err(error),
        Err(error) => Ok(match cx.study.not_identified_product(&error) {
            Some(failure) => super::identification_refused_with(failure),
            None => super::identification_refused_at_compile(),
        }),
    }
}

/// Status of the identification the program retained, from the product the route reads.
fn identification_status(caches: &[IdentificationCache]) -> Option<IdentificationStatus> {
    caches.iter().find_map(|cache| match cache {
        IdentificationCache::Static(cache) => Some(cache.identification.status),
        IdentificationCache::Pag(cache) => Some(cache.identification.status),
        IdentificationCache::Cpdag(cache) => Some(cache.identification.status),
        _ => None,
    })
}

/// The claim an identification status licenses, as the claim reader classifies it.
const fn claim_kind(status: IdentificationStatus) -> ClaimKind {
    match status {
        IdentificationStatus::NonparametricallyIdentified
        | IdentificationStatus::IdentifiedUnderParametricRestrictions
        | IdentificationStatus::IdentifiedUnderPriorRestrictions => ClaimKind::Point,
        IdentificationStatus::PartiallyIdentified => ClaimKind::Bounds,
        IdentificationStatus::GraphDependent => ClaimKind::Mixture,
        IdentificationStatus::NotIdentified => ClaimKind::Incomplete,
    }
}

/// The estimator configuration the caller supplied, when it is more than an id.
fn configured_spec(study: &Study) -> Option<&EstimatorSpec> {
    study.estimator_spec.as_deref().filter(|spec| !matches!(spec, EstimatorSpec::Default(_)))
}

/// Replicates and sampling design of the bootstrap the route runs.
///
/// A retained linear operation owns its fitter. A configured estimator owns its
/// replicate count and design. An id-selected estimator takes the study's count,
/// except the instrument, matching and learner fits, which publish their own
/// uncertainty without the bootstrap.
fn resampling_request(
    study: &Study,
    linear: Option<&CompiledLinearExecution>,
    estimator: Option<EstimatorId>,
    spec: Option<&EstimatorSpec>,
) -> (u32, Option<DeclaredSamplingDesign>) {
    let declared = |declaration: SamplingDeclaration| match declaration {
        SamplingDeclaration::Declared(design) => Some(design),
        SamplingDeclaration::Undeclared => study.sampling_design,
    };
    if let Some(linear) = linear {
        return (linear.fitter.bootstrap_replicates, declared(linear.fitter.sampling_design));
    }
    if let Some(spec) = spec {
        return match spec {
            EstimatorSpec::LinearAdjustmentAte(config) => {
                (config.bootstrap_replicates, declared(config.sampling_design))
            }
            EstimatorSpec::PropensityWeighting(config) => {
                (config.bootstrap_replicates, declared(config.sampling_design))
            }
            EstimatorSpec::PropensityStratification(config) => {
                (config.bootstrap_replicates, declared(config.sampling_design))
            }
            EstimatorSpec::Aipw(config) => {
                (config.bootstrap_replicates, declared(config.sampling_design))
            }
            EstimatorSpec::GlmAdjustment(config) => {
                (config.bootstrap_replicates, declared(config.sampling_design))
            }
            EstimatorSpec::FrontDoorTwoStage(config) => {
                (config.bootstrap_replicates, declared(config.sampling_design))
            }
            EstimatorSpec::PropensityMatching(config) => {
                (config.bootstrap_replicates, study.sampling_design)
            }
            EstimatorSpec::DistanceMatching(config) => {
                (config.bootstrap_replicates, study.sampling_design)
            }
            EstimatorSpec::IvWald(config) => (config.bootstrap_replicates, study.sampling_design),
            EstimatorSpec::Iv2Sls(config) => (config.bootstrap_replicates, study.sampling_design),
            EstimatorSpec::Dml(_)
            | EstimatorSpec::DrLearner(_)
            | EstimatorSpec::CausalForest(_)
            | EstimatorSpec::Default(_) => (0, study.sampling_design),
        };
    }
    let unresampled = matches!(
        estimator,
        Some(
            EstimatorId::IvWald
                | EstimatorId::Iv2Sls
                | EstimatorId::PropensityMatching
                | EstimatorId::DistanceMatching
                | EstimatorId::Dml
                | EstimatorId::DrLearner
                | EstimatorId::CausalForest
        )
    );
    (if unresampled { 0 } else { study.bootstrap_replicates }, study.sampling_design)
}

/// Cross-fitting split of the estimators that fit nuisances out of fold.
fn fold_plan(
    study: &Study,
    estimator: Option<EstimatorId>,
    spec: Option<&EstimatorSpec>,
) -> Option<FoldPlan> {
    let unit = ResampleUnit::of_design(study.sampling_design);
    let (folds, unit) = match (estimator?, spec) {
        (EstimatorId::Dml, Some(EstimatorSpec::Dml(config))) => {
            (config.folds, if config.units.spec.is_some() { ResampleUnit::Cluster } else { unit })
        }
        (EstimatorId::Dml, _) => (antecedent_estimate::DmlAte::new().folds, unit),
        (EstimatorId::DrLearner, Some(EstimatorSpec::DrLearner(config))) => {
            (config.folds, if config.units.spec.is_some() { ResampleUnit::Cluster } else { unit })
        }
        (EstimatorId::DrLearner, _) => (antecedent_estimate::DrLearner::new().folds, unit),
        (EstimatorId::Aipw, spec) => {
            let overlap = match spec {
                Some(EstimatorSpec::Aipw(config)) => config.overlap,
                _ => study
                    .overlap_policy
                    .unwrap_or_else(|| antecedent_estimate::AipwAte::new().overlap),
            };
            (antecedent_estimate::CheckedAipwProcedure::for_overlap(overlap).folds(), unit)
        }
        (EstimatorId::BayesianRobustAte, _) => (
            antecedent_estimate::bayesian_robust_ate::BayesianRobustAteOptions::default().folds,
            ResampleUnit::Row,
        ),
        _ => return None,
    };
    (folds > 0).then_some(FoldPlan { folds, unit, embargo: None })
}

/// Models the estimator fits, with the variables each reads and predicts.
#[allow(clippy::too_many_lines, reason = "one exhaustive match over the static-effect estimators")]
fn model_slots(
    study: &Study,
    estimator: Option<EstimatorId>,
    spec: Option<&EstimatorSpec>,
    roles: &Roles,
) -> Vec<ModelSlot> {
    let Some(estimator) = estimator else {
        return Vec::new();
    };
    let kernel = estimator.kernel_label();
    let t = roles.treatment();
    let y = roles.outcome;
    let treatment = roles.treatment;
    let z = roles.adjustment.as_slice();
    let m = roles.mediators.as_slice();
    let iv = roles.instruments.as_slice();
    let x = roles.modifiers.as_slice();
    let tz = Roles::joined(&[&t, z]);
    match estimator {
        EstimatorId::LinearAdjustmentAte
        | EstimatorId::GlmAdjustment
        | EstimatorId::BayesianGcomp
        | EstimatorId::BayesianBasisGcomp => vec![slot(ModelRole::Outcome, kernel, tz, y)],
        EstimatorId::PropensityWeighting | EstimatorId::PropensityMatching => {
            vec![slot(ModelRole::Propensity, "glm.logit", z.to_vec(), treatment)]
        }
        EstimatorId::PropensityStratification => {
            let mut slots = vec![slot(ModelRole::Propensity, "glm.logit", z.to_vec(), treatment)];
            let adjusted = matches!(
                spec,
                Some(EstimatorSpec::PropensityStratification(config)) if config.regression_adjustment
            );
            if adjusted {
                slots.push(slot(ModelRole::Outcome, "ols", tz, y));
            }
            slots
        }
        EstimatorId::DistanceMatching => vec![slot(ModelRole::Nuisance, kernel, z.to_vec(), None)],
        EstimatorId::Aipw => vec![
            slot(ModelRole::Propensity, "glm.logit", z.to_vec(), treatment),
            slot(ModelRole::Outcome, "ols", tz, y),
        ],
        EstimatorId::BayesianRobustAte => vec![
            slot(ModelRole::Propensity, "ridge_logistic", z.to_vec(), treatment),
            slot(ModelRole::Outcome, "ridge", tz, y),
        ],
        EstimatorId::Dml => {
            let (outcome, propensity) = if let Some(EstimatorSpec::Dml(config)) = spec {
                (config.outcome, config.treatment)
            } else {
                let defaults = antecedent_estimate::DmlAte::new();
                (defaults.outcome, defaults.treatment)
            };
            vec![
                slot(ModelRole::Propensity, &propensity.identity(), z.to_vec(), treatment),
                slot(ModelRole::Outcome, &outcome.identity(), z.to_vec(), y),
            ]
        }
        EstimatorId::DrLearner => {
            let (outcome, propensity, last) = if let Some(EstimatorSpec::DrLearner(config)) = spec {
                (config.outcome, config.treatment, config.final_learner)
            } else {
                let defaults = antecedent_estimate::DrLearner::new();
                (defaults.outcome, defaults.treatment, defaults.final_learner)
            };
            vec![
                slot(ModelRole::Propensity, &propensity.identity(), z.to_vec(), treatment),
                slot(ModelRole::Outcome, &outcome.identity(), tz, y),
                slot(ModelRole::Nuisance, &last.identity(), x.to_vec(), None),
            ]
        }
        EstimatorId::CausalForest => {
            vec![slot(ModelRole::Outcome, kernel, Roles::joined(&[&t, x, z]), y)]
        }
        EstimatorId::ConditionalLinearAdjustment | EstimatorId::BayesianConditional => {
            vec![slot(ModelRole::Outcome, kernel, Roles::joined(&[&t, z, x]), y)]
        }
        EstimatorId::FrontDoorTwoStage | EstimatorId::FrontDoorFunctional => {
            let mut slots: Vec<ModelSlot> = m
                .iter()
                .map(|&mediator| slot(ModelRole::Mediator, kernel, t.clone(), Some(mediator)))
                .collect();
            slots.push(slot(ModelRole::Outcome, kernel, Roles::joined(&[m, &t]), y));
            slots
        }
        EstimatorId::IvWald => vec![
            slot(ModelRole::Instrument, kernel, iv.to_vec(), treatment),
            slot(ModelRole::Outcome, kernel, iv.to_vec(), y),
        ],
        EstimatorId::Iv2Sls | EstimatorId::BayesianIvJointLinear => vec![
            slot(ModelRole::Instrument, kernel, Roles::joined(&[iv, z]), treatment),
            slot(ModelRole::Outcome, kernel, tz, y),
        ],
        EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear => {
            let running: Vec<VariableId> =
                study.rd.map(|config| config.running_variable).into_iter().collect();
            vec![slot(ModelRole::Outcome, kernel, running, y)]
        }
        EstimatorId::StaticMediationLinear => {
            let mut slots: Vec<ModelSlot> = m
                .iter()
                .map(|&mediator| slot(ModelRole::Mediator, kernel, tz.clone(), Some(mediator)))
                .collect();
            slots.push(slot(ModelRole::Outcome, kernel, Roles::joined(&[&t, m, z]), y));
            slots
        }
        // A discrete plug-in evaluates the identified factors; it fits no model.
        _ => Vec::new(),
    }
}

fn slot(
    role: ModelRole,
    family: &str,
    inputs: Vec<VariableId>,
    output: Option<VariableId>,
) -> ModelSlot {
    ModelSlot { role, family: Arc::from(family), inputs: Arc::from(inputs), output }
}

/// Layer five's design. A retained linear operation quotes its own design; a linear
/// adjustment on one adjustment set quotes `[1 | T | Z…]` from the data; every other
/// route states what it fits instead of one quoted adjustment design and carries no
/// rows or columns; the physical record keeps the planner's sizing.
fn design_shape(
    cx: &LoweringContext<'_>,
    linear: Option<&CompiledLinearExecution>,
    estimator: Option<EstimatorId>,
    static_cache: Option<&CachedStaticIdentification>,
    spec: Option<&EstimatorSpec>,
) -> Result<DesignShape, CausalError> {
    if let Some(linear) = linear {
        return quote_adjustment_width(
            cx,
            &linear.identification.estimand.adjustment_set,
            &linear.fitter.contrasts,
        );
    }
    Ok(match (estimator, static_cache) {
        // One DAG, supplied or projected from a latent-free mixed graph: one adjustment set.
        (Some(EstimatorId::LinearAdjustmentAte), Some(cache))
            if cx.study.graph.class() == GraphClass::Dag || cx.plan.static_graph().is_some() =>
        {
            let contrasts = match spec {
                Some(EstimatorSpec::LinearAdjustmentAte(config)) => config.contrasts.clone(),
                _ => Vec::new(),
            };
            // A categorical adjustment without a coding has no quotable width.
            quote_adjustment_width(cx, &cache.estimand.adjustment_set, &contrasts)
                .unwrap_or_else(|_| DesignShape::not_design_based(UNCODED_ADJUSTMENT))
        }
        (Some(EstimatorId::FunctionalDistribution | EstimatorId::FunctionalEffect), _) => {
            DesignShape::not_design_based(PLUG_IN_FITS_NO_DESIGN)
        }
        (Some(EstimatorId::LinearAdjustmentAte), _) => {
            DesignShape::not_design_based(ADJUSTMENT_SET_PER_GRAPH)
        }
        (Some(EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear), _) => {
            DesignShape::not_design_based(LOCAL_FITS_AT_EXECUTION)
        }
        _ => DesignShape::not_design_based(NUISANCE_MODELS),
    })
}

/// Physical read, fit and intervene contracts sized from the plan the program installs:
/// a quoted design re-plans the record first, as installation does.
fn physical_operators(
    cx: &LoweringContext<'_>,
    linear: Option<&CompiledLinearExecution>,
    design: DesignShape,
    statistical: &StatisticalProgram,
) -> Result<Vec<antecedent_core::PhysicalOperator>, CausalError> {
    if let Some(linear) = linear {
        return Ok(linear.program.physical_operators().to_vec());
    }
    let resample = statistical.resampling().is_some();
    if design.source() == WidthSource::Quoted {
        let mut planned = cx.plan.clone();
        planned.requote(design, cx.ctx)?;
        return linear_gaussian_physical_operators(&planned, resample);
    }
    linear_gaussian_physical_operators(cx.plan, resample)
}

#[cfg(test)]
#[path = "static_effect_tests.rs"]
mod tests;
