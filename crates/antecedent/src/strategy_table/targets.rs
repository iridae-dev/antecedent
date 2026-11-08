//! What each estimator estimates: the declarative target of every [`EstimatorId`], and the one
//! validity check that reads it.
//!
//! An estimator either estimates the quantity a query asks for or it does not. That is a
//! scientific fact about the estimator, so a mismatch refuses in every build (it is never
//! `not_supported`, which says an estimator that does answer the question lacks evidence for
//! the cell). The declaration is [`EstimatorTarget`]:
//!
//! - `query_kinds`: the [`CausalQueryKind`]s the estimator can estimate;
//! - `inference`: the inference mode it implements ([`InferenceTarget`]);
//! - `design`: for an estimator whose target depends on the query's own fields, the design
//!   variants it answers ([`DesignTarget`]).
//!
//! [`estimator_targets`] reads a built query, [`estimator_targets_cell`] reads a support-matrix
//! cell; both evaluate the same declaration without executing anything, so the cell reading
//! is the under-support inventory (`parity/estimator_validity.toml`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};

use antecedent_core::{
    CausalQuery, CausalQueryKind as K, LongitudinalRegimeMethod, RandomizationDesign,
    ResponseFunctional, TemporalAssignment, TemporalPolicy,
};

use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::support::{
    SupportCell, SupportGraphClass, SupportInference, SupportQuery, SupportStructure,
    SupportValidation,
};

use antecedent_expr::EstimandMethod;

use super::{EstimatorId, IdentifierId, estimand_compatible_with_estimator, validate_static_pair};

const _: () = assert!(
    crate::error::is_runtime_refusal_code("strategy_incompatible"),
    "`strategy_incompatible` is not a runtime_refusal code"
);
const _: () = assert!(
    crate::error::is_runtime_refusal_code("estimator_inference_mismatch"),
    "`estimator_inference_mismatch` is not a runtime_refusal code"
);

/// The inference mode an estimator implements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceTarget {
    /// Sampling-distribution interval; refuses a Bayesian request.
    Frequentist,
    /// Posterior; refuses a Frequentist request.
    Bayesian,
    /// The estimator applies the requested mode at execution.
    Both,
}

/// Randomized-experiment design a design estimator answers (a classification of
/// [`RandomizationDesign`] and the query's covariate / receipt declarations).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RandomizedShape {
    /// Independent Bernoulli assignment, intention to treat.
    Bernoulli,
    /// Independent Bernoulli assignment with a received-treatment record (CACE / LATE).
    BernoulliCace,
    /// Independent Bernoulli assignment with a fixed pre-assignment covariate adjustment.
    BernoulliCuped,
    /// Independent Bernoulli assignment with fitted ANCOVA covariates.
    BernoulliAncova,
    /// Complete randomization.
    Complete,
    /// Stratified randomization.
    Stratified,
    /// 2x2 factorial randomization.
    Factorial2x2,
    /// Cluster randomization without a fixed covariate adjustment.
    Cluster,
    /// Cluster randomization with a fixed pre-assignment covariate adjustment.
    ClusterCuped,
    /// Three or more independently assigned actions.
    MultiArm,
    /// Switchback sequences.
    Switchback,
}

/// Frozen inputs a policy-value estimator answers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyShape {
    /// Fixed binary policy, no outcome model (inverse probability).
    Ipw,
    /// Fixed binary policy with held-out outcome models (doubly robust).
    DoublyRobust,
    /// Fixed multi-action policy.
    MultiAction,
}

/// Assignment record of a temporal-design query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemporalDesignShape {
    /// Arm, factorial, interference or dose assignment: excursion contrasts.
    Excursion,
    /// Static regimes on subject trajectories: regime survival.
    RegimeSurvival,
}

/// Functional (and attachment) of a response query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseShape {
    /// [`ResponseFunctional::MeanCurve`].
    MeanCurve,
    /// [`ResponseFunctional::AverageDerivative`].
    AverageDerivative,
    /// [`ResponseFunctional::PointDerivative`] (every reporting scale).
    PointDerivative,
    /// [`ResponseFunctional::DirectionalDerivative`].
    DirectionalDerivative,
    /// [`ResponseFunctional::Jacobian`].
    Jacobian,
    /// [`ResponseFunctional::InterventionResponse`].
    InterventionResponse,
    /// A temporal response (any functional) attached to a time-indexed treatment.
    Temporal,
}

/// The design variants an estimator answers, for estimators whose target depends on the
/// query's own fields and not only on its kind.
#[derive(Clone, Copy, Debug)]
pub enum DesignTarget {
    /// The query kind alone fixes the target.
    Any,
    /// A [`K::RandomizedEffect`] with one of these designs.
    Randomized(&'static [RandomizedShape]),
    /// A [`K::PolicyValue`] with one of these frozen-input shapes.
    Policy(&'static [PolicyShape]),
    /// A [`K::LongitudinalRegime`] with one of these methods.
    Longitudinal(&'static [LongitudinalRegimeMethod]),
    /// A [`K::TemporalDesign`] with one of these assignment records.
    TemporalDesign(&'static [TemporalDesignShape]),
    /// A [`K::TemporalEffect`] with one of these window shapes, per inference mode.
    Temporal {
        /// Shapes answered under frequentist inference.
        frequentist: TemporalAccept,
        /// Shapes answered under Bayesian inference.
        bayesian: TemporalAccept,
    },
    /// A [`K::Response`] with one of these shapes, per inference mode (a Bayesian level curve
    /// is `response.bayesian`; a Bayesian derivative keeps the derivative estimator).
    Response {
        /// Shapes answered under frequentist inference.
        frequentist: &'static [ResponseShape],
        /// Shapes answered under Bayesian inference.
        bayesian: &'static [ResponseShape],
    },
}

/// What an estimator estimates.
#[derive(Clone, Copy, Debug)]
pub struct EstimatorTarget {
    /// The query kinds it can estimate.
    pub query_kinds: &'static [K],
    /// The inference mode it implements.
    pub inference: InferenceTarget,
    /// The design variants it answers within those kinds.
    pub design: DesignTarget,
    /// The graph classes it runs on.
    pub graphs: GraphScope,
    /// Per-kind replacements of [`Self::graphs`] (an identifier-native plug-in answers a
    /// response or a conditional effect on an ADMG only).
    pub kind_graphs: &'static [(K, GraphScope)],
    /// Kinds (and the classes) on which a graph-posterior structure is not executed by this
    /// estimator's route (the posterior is evaluated per atom by a different estimator).
    pub posterior_not: &'static [(K, &'static [GraphClass])],
    /// Kinds and classes on which this estimator runs only over a graph posterior (the fixed
    /// structure is taken by another estimator).
    pub posterior_only: &'static [(K, &'static [GraphClass])],
    /// Kinds (and the classes, `None` for every class and structure) on which the route runs a
    /// different estimator under Bayesian inference, so a selection of this one would be
    /// silently replaced. A class-specific entry describes the fixed-structure plan only; over
    /// a graph posterior the route is not replaced.
    pub bayesian_not: &'static [(K, Option<&'static [GraphClass]>)],
    /// Kinds on which a tier background is not run (the route needs a supplied DAG or another
    /// estimator runs the closure).
    pub tier_not: &'static [K],
    /// Classes on which an effect-modified temporal window runs, on a fixed structure only;
    /// empty for no restriction.
    pub conditional_fixed_on: &'static [GraphClass],
    /// The validation suites the estimator's route implements.
    pub validation: ValidationTarget,
}

/// The validation suites an estimator's route implements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationTarget {
    /// Every suite.
    Any,
    /// Only the `none` suite: the route never reads the requested refuters or predictive
    /// checks, so a `cheap` or `full` suite would silently run none.
    NoneOnly,
}

/// Where a question is asked: the identification route the estimator must be able to run on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Site {
    /// The graph class the compiler dispatches on (a tier background reads as a DAG).
    pub class: GraphClass,
    /// The structure is a tier background identified by generalized adjustment.
    pub tier: bool,
    /// The structure is a graph posterior (a mixture of atoms), not one fixed graph.
    pub posterior: bool,
}

/// The effect-window shapes of a temporal effect query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemporalPolicyShape {
    /// A pulse at one offset.
    Pulse,
    /// A sustained schedule of one active step.
    SustainedOneStep,
    /// A sustained schedule of several active steps.
    SustainedMultiStep,
}

/// The temporal-effect shapes accepted, by whether the query declares effect modifiers.
#[derive(Clone, Copy, Debug)]
pub struct TemporalAccept {
    /// Shapes accepted with no effect modifiers.
    pub plain: &'static [TemporalPolicyShape],
    /// Shapes accepted with effect modifiers.
    pub conditional: &'static [TemporalPolicyShape],
}

/// The graph classes an estimator runs on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphScope {
    /// Any class (identifier-native plug-ins, structural models, graphless designs).
    Any,
    /// Static graphs: not a time-indexed class (a temporal ADMG is read through its
    /// contemporaneous projection, so it counts).
    Static,
    /// Time-indexed graphs.
    Temporal,
    /// A DAG only.
    Dag,
    /// An ADMG only.
    Admg,
    /// An ADMG or a temporal ADMG.
    AdmgFamily,
    /// Any class except a temporal DAG.
    NotTemporalDag,
    /// Exactly these classes.
    Of(&'static [GraphClass]),
}

impl GraphScope {
    /// Whether an estimator of this scope runs on `class`.
    #[must_use]
    pub fn allows(self, class: GraphClass) -> bool {
        match self {
            Self::Any => true,
            Self::Static => matches!(
                class,
                GraphClass::Dag
                    | GraphClass::Admg
                    | GraphClass::Cpdag
                    | GraphClass::Pag
                    | GraphClass::TemporalAdmg
                    | GraphClass::RandomizedTrial
            ),
            Self::Temporal => matches!(
                class,
                GraphClass::TemporalDag
                    | GraphClass::TemporalCpdag
                    | GraphClass::TemporalPag
                    | GraphClass::TemporalAdmg
            ),
            Self::Dag => matches!(class, GraphClass::Dag),
            Self::Admg => matches!(class, GraphClass::Admg),
            Self::AdmgFamily => matches!(class, GraphClass::Admg | GraphClass::TemporalAdmg),
            Self::NotTemporalDag => !matches!(class, GraphClass::TemporalDag),
            Self::Of(classes) => classes.contains(&class),
        }
    }
}

const AVERAGE: &[K] = &[K::AverageEffect];
const CONDITIONAL: &[K] = &[K::ConditionalEffect];
const TEMPORAL_EFFECT: &[K] = &[K::TemporalEffect];
const MEDIATION: &[K] = &[K::Mediation];
const RANDOMIZED: &[K] = &[K::RandomizedEffect];
const POLICY: &[K] = &[K::PolicyValue];
const LONGITUDINAL: &[K] = &[K::LongitudinalRegime];
const SURVIVAL: &[K] = &[K::Survival];
const DOSE: &[K] = &[K::ContinuousDoseResponse];
const TRANSPORT: &[K] = &[K::Transport];
const INTERFERENCE: &[K] = &[K::Interference];
const IV_BOUNDS: &[K] = &[K::IvBounds];
const TEMPORAL_DESIGN: &[K] = &[K::TemporalDesign];
const HISTORY_MATCH: &[K] = &[K::HistoryMatch];
const SEQUENTIAL_CONTRAST: &[K] = &[K::SequentialContrast];
const RESPONSE: &[K] = &[K::Response];
/// The lag-0 average effect on a temporal graph is compiled as a pulse; the temporal
/// estimators therefore answer both readings, as the cells' evidence lists record.
const TEMPORAL_FAMILY: &[K] = &[
    K::TemporalEffect,
    K::AverageEffect,
    K::Counterfactual,
    K::Distribution,
    K::NestedCounterfactual,
    K::PathSpecific,
];
const TEMPORAL_BAYESIAN_GCOMP: &[K] = &[K::TemporalEffect, K::AverageEffect, K::Counterfactual];
/// The plug-in of an identified functional (general ID) evaluates every identified query
/// whose estimand it can evaluate; its graph-class limits are the compiler's.
const FUNCTIONAL_EFFECT: &[K] = &[
    K::AverageEffect,
    K::ConditionalEffect,
    K::Counterfactual,
    K::NestedCounterfactual,
    K::Mediation,
    K::PathSpecific,
    K::Response,
    K::TemporalEffect,
];
const STATIC_MEDIATION: &[K] = &[K::Mediation, K::NestedCounterfactual];
const BASIS_GCOMP: &[K] = &[K::AverageEffect, K::ConditionalEffect];
/// Fitted structural mechanisms: counterfactuals, attribution and simulation on a fitted
/// model (`analysis/execute/attribution_path.rs`, `checked_attribution.rs`).
const GCM: &[K] = &[
    K::Counterfactual,
    K::AnomalyAttribution,
    K::ChangeAttribution,
    K::MechanismChange,
    K::UnitChange,
    K::Attribute,
    K::Simulate,
];
const GCM_ANOMALY: &[K] = &[K::AnomalyAttribution, K::Attribute];
const GCM_CHANGE: &[K] = &[K::ChangeAttribution];

use RandomizedShape as R;
use ResponseShape as S;

const ARM_FREQUENTIST: &[RandomizedShape] = &[R::Bernoulli, R::MultiArm];
const NEYMAN: &[RandomizedShape] = &[R::Complete, R::Stratified, R::Factorial2x2, R::Cluster];
/// Arm-exchangeable likelihoods: Bernoulli, complete, multi-arm and 2x2 factorial assignment.
/// Cluster, stratified and switchback designs have no arm-exchangeable likelihood.
const ARM_BAYESIAN: &[RandomizedShape] = &[
    R::Bernoulli,
    R::BernoulliCace,
    R::BernoulliCuped,
    R::BernoulliAncova,
    R::Complete,
    R::Factorial2x2,
    R::MultiArm,
];

const LEVEL_SHAPES: &[ResponseShape] = &[S::MeanCurve, S::InterventionResponse];

const fn frequentist(query_kinds: &'static [K], design: DesignTarget) -> EstimatorTarget {
    EstimatorTarget {
        query_kinds,
        inference: InferenceTarget::Frequentist,
        design,
        graphs: GraphScope::Any,
        kind_graphs: &[],
        posterior_not: &[],
        posterior_only: &[],
        bayesian_not: &[],
        tier_not: &[],
        conditional_fixed_on: &[],
        validation: ValidationTarget::Any,
    }
}

const fn bayesian(query_kinds: &'static [K], design: DesignTarget) -> EstimatorTarget {
    EstimatorTarget {
        query_kinds,
        inference: InferenceTarget::Bayesian,
        design,
        graphs: GraphScope::Any,
        kind_graphs: &[],
        posterior_not: &[],
        posterior_only: &[],
        bayesian_not: &[],
        tier_not: &[],
        conditional_fixed_on: &[],
        validation: ValidationTarget::Any,
    }
}

const fn both(query_kinds: &'static [K], design: DesignTarget) -> EstimatorTarget {
    EstimatorTarget {
        query_kinds,
        inference: InferenceTarget::Both,
        design,
        graphs: GraphScope::Any,
        kind_graphs: &[],
        posterior_not: &[],
        posterior_only: &[],
        bayesian_not: &[],
        tier_not: &[],
        conditional_fixed_on: &[],
        validation: ValidationTarget::Any,
    }
}

const ANY: DesignTarget = DesignTarget::Any;

use TemporalPolicyShape as P;
const ONE_STEP_OR_PULSE: &[TemporalPolicyShape] = &[P::Pulse, P::SustainedOneStep];
const MULTI_STEP: &[TemporalPolicyShape] = &[P::SustainedMultiStep];
const NO_SHAPE: &[TemporalPolicyShape] = &[];
const NO_TEMPORAL: TemporalAccept = TemporalAccept { plain: NO_SHAPE, conditional: NO_SHAPE };
/// The single-window estimators: a pulse or one-step sustained effect, no effect modifiers.
const SINGLE_WINDOW: TemporalAccept =
    TemporalAccept { plain: ONE_STEP_OR_PULSE, conditional: NO_SHAPE };

const PLAIN_TEMPORAL: &[GraphClass] =
    &[GraphClass::TemporalDag, GraphClass::TemporalCpdag, GraphClass::TemporalPag];
const FIXED_TEMPORAL_DAG: &[GraphClass] = &[GraphClass::TemporalDag];
const TEMPORAL_MEDIATION_CLASSES: &[GraphClass] =
    &[GraphClass::TemporalDag, GraphClass::TemporalCpdag];
/// Every class except a plain time-indexed one: where the identified-functional plug-in and
/// the structural mechanisms of a static or contemporaneous reading run.
const NOT_PLAIN_TEMPORAL: &[GraphClass] = &[
    GraphClass::Dag,
    GraphClass::Admg,
    GraphClass::Cpdag,
    GraphClass::Pag,
    GraphClass::TemporalAdmg,
];
const ALL_CLASSES: &[GraphClass] = &[
    GraphClass::Dag,
    GraphClass::Admg,
    GraphClass::Cpdag,
    GraphClass::Pag,
    GraphClass::TemporalDag,
    GraphClass::TemporalCpdag,
    GraphClass::TemporalPag,
    GraphClass::TemporalAdmg,
    GraphClass::RandomizedTrial,
];
/// Where the plug-in executes a counterfactual (ID*), nested or path-specific effect, a
/// mediation, a conditional effect or a response (`compile_logical_counterfactual_class`,
/// `execute_path_specific_mixed`, `execute_mediation_mixed`).
const ID_COUNTERFACTUAL: &[GraphClass] =
    &[GraphClass::Admg, GraphClass::Pag, GraphClass::TemporalAdmg];
/// The nested effect: the mixed and class graphs, and a DAG only over a graph posterior (a fixed
/// tabular DAG takes `mediation.linear`).
const NESTED_PLUGIN: &[GraphClass] = &[
    GraphClass::Dag,
    GraphClass::Admg,
    GraphClass::Cpdag,
    GraphClass::Pag,
    GraphClass::TemporalAdmg,
];
const PLUGIN_MEDIATION: &[GraphClass] = &[
    GraphClass::Admg,
    GraphClass::Cpdag,
    GraphClass::Pag,
    GraphClass::TemporalAdmg,
    GraphClass::TemporalPag,
];
const STRUCTURAL_COUNTERFACTUAL: &[GraphClass] = &[GraphClass::Dag, GraphClass::Cpdag];
const ONLY_DAG: &[GraphClass] = &[GraphClass::Dag];
const ONLY_TEMPORAL_ADMG: &[GraphClass] = &[GraphClass::TemporalAdmg];

/// The target of every estimator. An exhaustive `match`, so a new [`EstimatorId`] does not
/// compile until its row says what it estimates.
#[allow(
    clippy::too_many_lines,
    reason = "one declarative row per EstimatorId in an exhaustive match; each row is independent"
)]
const fn declared(id: EstimatorId) -> EstimatorTarget {
    match id {
        // Static adjustment-family point estimators: the marginal average effect only. A
        // modifier-wise effect is `conditional.*`; responses and temporal queries have their own.
        EstimatorId::LinearAdjustmentAte
        | EstimatorId::PropensityWeighting
        | EstimatorId::PropensityMatching
        | EstimatorId::PropensityStratification
        | EstimatorId::DistanceMatching
        | EstimatorId::Aipw
        | EstimatorId::GlmAdjustment
        | EstimatorId::Dml
        | EstimatorId::DrLearner
        | EstimatorId::CausalForest
        | EstimatorId::FrontDoorTwoStage
        | EstimatorId::FrontDoorFunctional
        | EstimatorId::IvWald
        | EstimatorId::Iv2Sls
        | EstimatorId::RandomizedQuantile
        | EstimatorId::RdSharp => frequentist(AVERAGE, ANY),
        EstimatorId::BayesianGcomp
        | EstimatorId::BayesianRobustAte
        | EstimatorId::BayesianRandomizedQuantile
        | EstimatorId::BayesianIvJointLinear
        | EstimatorId::BayesianRdLocalLinear => bayesian(AVERAGE, ANY),
        EstimatorId::BayesianBasisGcomp => bayesian(BASIS_GCOMP, ANY),
        EstimatorId::ConditionalLinearAdjustment => frequentist(CONDITIONAL, ANY),
        EstimatorId::BayesianConditional => bayesian(CONDITIONAL, ANY),

        // Temporal effects (`temporal_effect_procedure`, `temporal_response_for`,
        // `temporal_mediation_for`). The linear temporal adjustment also executes the
        // counterfactual, distribution, nested and path-specific readings on a temporal graph
        // (`parity/support_cells.toml` evidence lists) and, under either inference mode, the
        // placeholder the builder binds for a temporal effect.
        EstimatorId::TemporalLinearAdjustment => both(
            TEMPORAL_FAMILY,
            DesignTarget::Temporal {
                frequentist: TemporalAccept {
                    plain: ONE_STEP_OR_PULSE,
                    conditional: ONE_STEP_OR_PULSE,
                },
                bayesian: TemporalAccept { plain: NO_SHAPE, conditional: ONE_STEP_OR_PULSE },
            },
        ),
        EstimatorId::BayesianTemporalGcomp => bayesian(
            TEMPORAL_BAYESIAN_GCOMP,
            DesignTarget::Temporal { frequentist: NO_TEMPORAL, bayesian: SINGLE_WINDOW },
        ),
        // A multi-step sustained schedule runs sequential g-computation in either mode.
        EstimatorId::TemporalSequentialGcomp => both(
            TEMPORAL_EFFECT,
            DesignTarget::Temporal {
                frequentist: TemporalAccept { plain: MULTI_STEP, conditional: MULTI_STEP },
                bayesian: TemporalAccept { plain: MULTI_STEP, conditional: MULTI_STEP },
            },
        ),
        EstimatorId::TemporalGlmLinkContrast | EstimatorId::TemporalLaggedInstrument => {
            frequentist(
                TEMPORAL_EFFECT,
                DesignTarget::Temporal { frequentist: SINGLE_WINDOW, bayesian: NO_TEMPORAL },
            )
        }
        EstimatorId::BayesianTemporalGlmLinkContrast
        | EstimatorId::BayesianTemporalLaggedInstrument
        | EstimatorId::BayesianTemporalWhiteInnovation => bayesian(
            TEMPORAL_EFFECT,
            DesignTarget::Temporal { frequentist: NO_TEMPORAL, bayesian: SINGLE_WINDOW },
        ),
        EstimatorId::TemporalMediation => frequentist(MEDIATION, ANY),
        EstimatorId::BayesianTemporalMediation => bayesian(MEDIATION, ANY),
        // Temporal response (dose over horizon, policy path): any functional, temporal attachment.
        EstimatorId::TemporalResponseGcomp => frequentist(
            RESPONSE,
            DesignTarget::Response { frequentist: &[S::Temporal], bayesian: &[] },
        ),
        EstimatorId::TemporalResponseBayesian => bayesian(
            RESPONSE,
            DesignTarget::Response { frequentist: &[], bayesian: &[S::Temporal] },
        ),

        // Identifier-native plug-ins: they apply the requested mode at execution.
        EstimatorId::FunctionalDistribution => both(&[K::Distribution], ANY),
        EstimatorId::FunctionalEffect => both(
            FUNCTIONAL_EFFECT,
            DesignTarget::Response { frequentist: LEVEL_SHAPES, bayesian: LEVEL_SHAPES },
        ),
        // Static linear structural mediation also evaluates the path-specific natural effects
        // and the fixed-contract nested natural direct effect (`estimator_data` compat:
        // `PathSpecificNatural`; the nested cells list `mediation.linear`).
        EstimatorId::StaticMediationLinear => both(STATIC_MEDIATION, ANY),

        // Static responses. A level curve is `kennedy_dr` (frequentist) or `response.bayesian`;
        // a Bayesian derivative keeps the derivative estimator (`static_response_for`,
        // `default_for_response`); a joint intervention is `cell.aipw` / `intervention_gcomp`
        // (frequentist) or `response.bayesian`.
        EstimatorId::ResponseKennedyDr => both(
            RESPONSE,
            DesignTarget::Response {
                frequentist: &[S::MeanCurve, S::PointDerivative],
                bayesian: &[S::PointDerivative],
            },
        ),
        EstimatorId::ResponseRieszAde => both(
            RESPONSE,
            DesignTarget::Response {
                frequentist: &[S::AverageDerivative],
                bayesian: &[S::AverageDerivative],
            },
        ),
        EstimatorId::ResponseGamDerivative => both(
            RESPONSE,
            DesignTarget::Response {
                frequentist: &[S::DirectionalDerivative, S::Jacobian],
                bayesian: &[S::DirectionalDerivative, S::Jacobian],
            },
        ),
        EstimatorId::ResponseInterventionGcomp | EstimatorId::CellAipw => frequentist(
            RESPONSE,
            DesignTarget::Response { frequentist: &[S::InterventionResponse], bayesian: &[] },
        ),
        EstimatorId::ResponseBayesian => {
            bayesian(RESPONSE, DesignTarget::Response { frequentist: &[], bayesian: LEVEL_SHAPES })
        }

        // Fitted structural mechanisms (abduction-action-prediction) and their Bayesian
        // bootstrap attribution (`checked_attribution.rs`: anomaly / model target uses
        // `gcm.fit.bayesian`, change uses `gcm.attribution.bayesian`).
        EstimatorId::GcmFit => both(GCM, ANY),
        EstimatorId::GcmFitBayesian => bayesian(GCM_ANOMALY, ANY),
        EstimatorId::GcmAttributionBayesian => bayesian(GCM_CHANGE, ANY),

        // Transport and interference (`refuse_unsupported_transport`, interference executors).
        EstimatorId::TransportTrialIpw => frequentist(TRANSPORT, ANY),
        EstimatorId::TransportTrialBayesianBootstrap => bayesian(TRANSPORT, ANY),
        EstimatorId::InterferenceHtHajek
        | EstimatorId::InterferenceClusterNeyman
        | EstimatorId::InterferenceSaturationExact
        | EstimatorId::InterferenceObservationalIpw => frequentist(INTERFERENCE, ANY),
        EstimatorId::InterferenceBayesianGaussian
        | EstimatorId::InterferenceClusterBayesianBootstrap => bayesian(INTERFERENCE, ANY),

        // Randomized experiments (`randomized_shape`): each design has one frequentist
        // estimator; the Bayesian arm contrasts answer the arm-exchangeable designs.
        EstimatorId::RandomizedHt => {
            frequentist(RANDOMIZED, DesignTarget::Randomized(ARM_FREQUENTIST))
        }
        EstimatorId::RandomizedFixedCupedHt => {
            frequentist(RANDOMIZED, DesignTarget::Randomized(&[R::BernoulliCuped]))
        }
        EstimatorId::RandomizedClusterFixedCuped => {
            frequentist(RANDOMIZED, DesignTarget::Randomized(&[R::ClusterCuped]))
        }
        EstimatorId::RandomizedAncova => {
            frequentist(RANDOMIZED, DesignTarget::Randomized(&[R::BernoulliAncova]))
        }
        EstimatorId::RandomizedWaldCace => {
            frequentist(RANDOMIZED, DesignTarget::Randomized(&[R::BernoulliCace]))
        }
        EstimatorId::RandomizedSwitchbackHt => {
            frequentist(RANDOMIZED, DesignTarget::Randomized(&[R::Switchback]))
        }
        EstimatorId::RandomizedNeyman => frequentist(RANDOMIZED, DesignTarget::Randomized(NEYMAN)),
        EstimatorId::BayesianRandomizedFinitePopulation
        | EstimatorId::BayesianRandomizedSuperpopulation
        | EstimatorId::BayesianComplierEffect => {
            bayesian(RANDOMIZED, DesignTarget::Randomized(ARM_BAYESIAN))
        }
        EstimatorId::BayesianSwitchback => {
            bayesian(RANDOMIZED, DesignTarget::Randomized(&[R::Switchback]))
        }

        // Policy value (`policy_shape`).
        EstimatorId::RandomizedIpwPolicy => {
            frequentist(POLICY, DesignTarget::Policy(&[PolicyShape::Ipw]))
        }
        EstimatorId::RandomizedDrPolicy => {
            frequentist(POLICY, DesignTarget::Policy(&[PolicyShape::DoublyRobust]))
        }
        EstimatorId::RandomizedMultiActionIpwPolicy => {
            frequentist(POLICY, DesignTarget::Policy(&[PolicyShape::MultiAction]))
        }
        EstimatorId::BayesianBinaryPolicyValue => bayesian(
            POLICY,
            DesignTarget::Policy(&[
                PolicyShape::Ipw,
                PolicyShape::DoublyRobust,
                PolicyShape::MultiAction,
            ]),
        ),

        // Survival and continuous dose.
        EstimatorId::RandomizedSurvivalProductLimit => frequentist(SURVIVAL, ANY),
        EstimatorId::BayesianPiecewiseSurvival => bayesian(SURVIVAL, ANY),
        EstimatorId::ContinuousDoseKernel => frequentist(DOSE, ANY),
        EstimatorId::BayesianDosePolicy => bayesian(DOSE, ANY),

        // Longitudinal regimes: one estimator per method; the Bayesian sequential regime value
        // is the twin of the sequentially augmented value.
        EstimatorId::LongitudinalIpwRegime => {
            frequentist(LONGITUDINAL, DesignTarget::Longitudinal(&[LongitudinalRegimeMethod::Ipw]))
        }
        EstimatorId::LongitudinalGFormulaRegime => frequentist(
            LONGITUDINAL,
            DesignTarget::Longitudinal(&[LongitudinalRegimeMethod::GFormula]),
        ),
        EstimatorId::LongitudinalSequentialDrRegime => frequentist(
            LONGITUDINAL,
            DesignTarget::Longitudinal(&[LongitudinalRegimeMethod::SequentialDoublyRobust]),
        ),
        EstimatorId::LongitudinalMarginalStructuralModel => frequentist(
            LONGITUDINAL,
            DesignTarget::Longitudinal(&[LongitudinalRegimeMethod::MarginalStructuralModel]),
        ),
        EstimatorId::BayesianSequentialRegimeValue => bayesian(
            LONGITUDINAL,
            DesignTarget::Longitudinal(&[LongitudinalRegimeMethod::SequentialDoublyRobust]),
        ),

        // Quasi-experimental panel and regression-discontinuity designs.
        EstimatorId::PanelDid => frequentist(&[K::PanelDid], ANY),
        EstimatorId::BayesianPanelDid => bayesian(&[K::PanelDid], ANY),
        EstimatorId::SyntheticControl => frequentist(&[K::SyntheticControl], ANY),
        EstimatorId::BayesianSyntheticControl => bayesian(&[K::SyntheticControl], ANY),
        EstimatorId::LocalRatio => frequentist(&[K::LocalPolynomialRatio], ANY),
        EstimatorId::BayesianLocalRatio => bayesian(&[K::LocalPolynomialRatio], ANY),

        // Instrumental-variable bounds, history matching, sequential contrast.
        EstimatorId::IvBoundsSharp => frequentist(IV_BOUNDS, ANY),
        EstimatorId::BayesianIvBounds => bayesian(IV_BOUNDS, ANY),
        EstimatorId::PeriodHistoryMatching => frequentist(HISTORY_MATCH, ANY),
        EstimatorId::BayesianPeriodHistoryOutcomeModel => bayesian(HISTORY_MATCH, ANY),
        EstimatorId::SequentialCrossfitContrastWald => frequentist(SEQUENTIAL_CONTRAST, ANY),
        EstimatorId::BayesianSequentialCrossfitContrast => bayesian(SEQUENTIAL_CONTRAST, ANY),

        // Temporal design: excursion estimators answer arm / dose designs, regime survival
        // answers a survival design (`temporal_design_shape`).
        EstimatorId::TemporalDesignExcursion => frequentist(
            TEMPORAL_DESIGN,
            DesignTarget::TemporalDesign(&[TemporalDesignShape::Excursion]),
        ),
        EstimatorId::BayesianTemporalDesignExcursion => bayesian(
            TEMPORAL_DESIGN,
            DesignTarget::TemporalDesign(&[TemporalDesignShape::Excursion]),
        ),
        EstimatorId::TemporalRegimeSurvival => frequentist(
            TEMPORAL_DESIGN,
            DesignTarget::TemporalDesign(&[TemporalDesignShape::RegimeSurvival]),
        ),
        EstimatorId::BayesianTemporalRegimeSurvival => bayesian(
            TEMPORAL_DESIGN,
            DesignTarget::TemporalDesign(&[TemporalDesignShape::RegimeSurvival]),
        ),
    }
}

/// The target of every estimator: [`declared`] (kinds, inference, design) with the graph
/// scope of the estimator's family.
///
/// - Static adjustment, front-door, IV, RD, conditional and static-response estimators run on
///   static classes only.
/// - The temporal family runs on time-indexed classes only.
/// - `functional.effect` runs anywhere, except that it answers a response on an ADMG or
///   temporal ADMG, a conditional effect on an ADMG, and a mediation on anything but a
///   temporal DAG (`refuse_incompatible_estimator_spec` before it was folded in here).
/// - `bayesian.basis.gcomp` answers a conditional effect on a DAG only.
/// - Mediation estimators other than the temporal ones refuse a temporal DAG.
/// - Everything else (structural models, transport, interference, graphless designs,
///   `functional.distribution`) is not graph-class scoped.
#[allow(
    clippy::too_many_lines,
    reason = "one match arm per estimator family; the scoping table reads as one declaration"
)]
pub(super) const fn estimator_target(id: EstimatorId) -> EstimatorTarget {
    let mut target = declared(id);
    match id {
        EstimatorId::LinearAdjustmentAte
        | EstimatorId::PropensityWeighting
        | EstimatorId::PropensityMatching
        | EstimatorId::PropensityStratification
        | EstimatorId::DistanceMatching
        | EstimatorId::Aipw
        | EstimatorId::GlmAdjustment
        | EstimatorId::Dml
        | EstimatorId::DrLearner
        | EstimatorId::CausalForest
        | EstimatorId::BayesianGcomp
        | EstimatorId::BayesianRobustAte
        | EstimatorId::RandomizedQuantile
        | EstimatorId::BayesianRandomizedQuantile
        | EstimatorId::ConditionalLinearAdjustment
        | EstimatorId::BayesianConditional
        | EstimatorId::ResponseKennedyDr
        | EstimatorId::ResponseRieszAde
        | EstimatorId::ResponseGamDerivative
        | EstimatorId::ResponseBayesian => target.graphs = GraphScope::Static,
        // A tier closure's joint response runs `cell.aipw` or `response.bayesian` only.
        EstimatorId::ResponseInterventionGcomp => {
            target.graphs = GraphScope::Static;
            target.tier_not = &[K::Response];
        }
        // A DAG graph-posterior atom is identified by backdoor adjustment, so an estimator that
        // consumes a front-door, instrument or discontinuity estimand does not run on it.
        EstimatorId::FrontDoorTwoStage
        | EstimatorId::FrontDoorFunctional
        | EstimatorId::IvWald
        | EstimatorId::Iv2Sls
        | EstimatorId::RdSharp
        | EstimatorId::BayesianIvJointLinear
        | EstimatorId::BayesianRdLocalLinear => {
            target.graphs = GraphScope::Static;
            target.posterior_not = &[(K::AverageEffect, ONLY_DAG)];
        }
        // The quadratic-basis g-computation needs a supplied DAG: it refuses a class graph and a
        // tier background.
        EstimatorId::BayesianBasisGcomp => {
            target.graphs = GraphScope::Of(ONLY_DAG);
            target.tier_not = &[K::AverageEffect, K::ConditionalEffect];
        }
        // The cell-saturated AIPW requires one static DAG (a tier closure reads as a DAG).
        EstimatorId::CellAipw => target.graphs = GraphScope::Of(ONLY_DAG),
        // Temporal family. The linear adjustment is the executed estimator of a frequentist
        // temporal contrast and of the cross-world and effect-modified plans; a Bayesian
        // pulse, sustained or lag-0 contrast, and a Bayesian counterfactual on a temporal DAG,
        // run `bayesian.temporal.gcomp` instead, so selecting the linear estimator there would
        // be silently replaced. A contrast on a temporal ADMG is evaluated by `functional.effect`.
        EstimatorId::TemporalLinearAdjustment => {
            target.graphs = GraphScope::Of(PLAIN_TEMPORAL);
            target.bayesian_not =
                &[(K::AverageEffect, None), (K::Counterfactual, Some(FIXED_TEMPORAL_DAG))];
        }
        // Sequential g-computation adjusts for a named modifier on a fixed temporal DAG; the class
        // routes and graph posteriors refuse effect modifiers.
        EstimatorId::TemporalSequentialGcomp => {
            target.graphs = GraphScope::Of(PLAIN_TEMPORAL);
            target.conditional_fixed_on = FIXED_TEMPORAL_DAG;
        }
        // The cross-world counterfactual compiles under the linear adjustment on every plain
        // class; the Bayesian twin runs it on a fixed temporal DAG only.
        EstimatorId::BayesianTemporalGcomp => {
            target.graphs = GraphScope::Of(PLAIN_TEMPORAL);
            target.kind_graphs = &[(K::Counterfactual, GraphScope::Of(FIXED_TEMPORAL_DAG))];
            target.posterior_not = &[(K::Counterfactual, FIXED_TEMPORAL_DAG)];
        }
        // Fixed temporal DAG only; a graph posterior mixes atoms and is not read.
        EstimatorId::TemporalGlmLinkContrast
        | EstimatorId::BayesianTemporalGlmLinkContrast
        | EstimatorId::TemporalLaggedInstrument
        | EstimatorId::BayesianTemporalLaggedInstrument
        | EstimatorId::BayesianTemporalWhiteInnovation => {
            target.graphs = GraphScope::Of(FIXED_TEMPORAL_DAG);
            target.posterior_not = &[(K::TemporalEffect, ALL_CLASSES)];
        }
        EstimatorId::TemporalMediation | EstimatorId::BayesianTemporalMediation => {
            target.graphs = GraphScope::Of(TEMPORAL_MEDIATION_CLASSES);
        }
        EstimatorId::TemporalResponseGcomp | EstimatorId::TemporalResponseBayesian => {
            target.graphs = GraphScope::Temporal;
            target.posterior_not = &[(K::Response, ONLY_TEMPORAL_ADMG)];
        }
        // The identified-functional plug-in: a temporal contrast only on a temporal ADMG
        // (`ensure_temporal_effect_selection`), a counterfactual by ID* on an ADMG, PAG or
        // temporal ADMG (a DAG or CPDAG counterfactual is the structural-mechanism route), the
        // nested effect where the mediation route does not take it (a tabular DAG takes
        // `mediation.linear` only), a mediation on a mixed or class graph, a path-specific
        // effect anywhere but a plain temporal class. A DAG graph posterior is evaluated per
        // atom by the target's own estimator.
        EstimatorId::FunctionalEffect => {
            target.kind_graphs = &[
                (K::Response, GraphScope::AdmgFamily),
                (K::ConditionalEffect, GraphScope::AdmgFamily),
                (K::Mediation, GraphScope::Of(PLUGIN_MEDIATION)),
                (K::Counterfactual, GraphScope::Of(ID_COUNTERFACTUAL)),
                (K::NestedCounterfactual, GraphScope::Of(NESTED_PLUGIN)),
                (K::PathSpecific, GraphScope::Of(NOT_PLAIN_TEMPORAL)),
                (K::AverageEffect, GraphScope::Of(NOT_PLAIN_TEMPORAL)),
                (K::TemporalEffect, GraphScope::Of(ONLY_TEMPORAL_ADMG)),
            ];
            target.posterior_not = &[(K::AverageEffect, ONLY_DAG)];
            target.posterior_only = &[(K::NestedCounterfactual, ONLY_DAG)];
        }
        EstimatorId::FunctionalDistribution => target.graphs = GraphScope::Of(NOT_PLAIN_TEMPORAL),
        // Static linear mediation: the fixed three-node tabular DAG only; nested effects are
        // evaluated per atom by `functional.effect` over a graph posterior.
        EstimatorId::StaticMediationLinear => {
            target.graphs = GraphScope::Of(ONLY_DAG);
            target.posterior_not = &[(K::NestedCounterfactual, ONLY_DAG)];
        }
        // Frequentist attribution and the structural counterfactual (a DAG or CPDAG). A
        // Bayesian anomaly or change attribution always runs its Bayesian twin.
        EstimatorId::GcmFit => {
            target.kind_graphs = &[(K::Counterfactual, GraphScope::Of(STRUCTURAL_COUNTERFACTUAL))];
            target.bayesian_not = &[(K::AnomalyAttribution, None), (K::ChangeAttribution, None)];
        }
        _ => {}
    }
    // The graph-posterior routes run one fixed estimator per query (the sealed linear
    // adjustment or `linear.adjustment.ate` for a frequentist average effect,
    // `bayesian.gcomp` for a Bayesian one, the default response estimator for a response,
    // `functional.effect` on ADMG atoms), so an explicit selection of any other estimator would
    // be silently replaced.
    match id {
        EstimatorId::PropensityWeighting
        | EstimatorId::PropensityMatching
        | EstimatorId::PropensityStratification
        | EstimatorId::DistanceMatching
        | EstimatorId::Aipw
        | EstimatorId::GlmAdjustment
        | EstimatorId::Dml
        | EstimatorId::DrLearner
        | EstimatorId::CausalForest
        | EstimatorId::BayesianRobustAte
        | EstimatorId::RandomizedQuantile
        | EstimatorId::BayesianRandomizedQuantile
        | EstimatorId::BayesianIvJointLinear
        | EstimatorId::BayesianRdLocalLinear => {
            target.posterior_not = &[(K::AverageEffect, ALL_CLASSES)];
        }
        EstimatorId::BayesianBasisGcomp => {
            target.posterior_not =
                &[(K::AverageEffect, ALL_CLASSES), (K::ConditionalEffect, ALL_CLASSES)];
        }
        EstimatorId::CellAipw => target.posterior_not = &[(K::Response, ALL_CLASSES)],
        _ => {}
    }
    // These routes never read the requested validation suite.
    match id {
        EstimatorId::BayesianRobustAte
        | EstimatorId::BayesianBasisGcomp
        | EstimatorId::BayesianRdLocalLinear
        | EstimatorId::RandomizedQuantile
        | EstimatorId::BayesianRandomizedQuantile
        | EstimatorId::BayesianIvJointLinear => target.validation = ValidationTarget::NoneOnly,
        _ => {}
    }
    target
}

impl EstimatorId {
    /// What this estimator estimates.
    #[must_use]
    pub const fn target(self) -> EstimatorTarget {
        estimator_target(self)
    }
}

/// The refusal of an estimator that does not estimate the requested quantity: scientific, so
/// it refuses in every build and never goes through the support gate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScientificRefusal {
    /// The registered runtime-refusal code (`strategy_incompatible` or
    /// `estimator_inference_mismatch`).
    pub code: &'static str,
    /// What does not match.
    pub detail: String,
}

impl std::fmt::Display for ScientificRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}: {}", crate::error::REASON_PREFIX, self.code, self.detail)
    }
}

impl std::error::Error for ScientificRefusal {}

impl From<ScientificRefusal> for CausalError {
    fn from(refusal: ScientificRefusal) -> Self {
        CausalError::Unsupported { message: intern(refusal.to_string()) }
    }
}

/// Interned wire message, for the `&'static str` an unsupported refusal carries. The set of
/// distinct messages is bounded by the estimator, query-kind and design-shape combinations.
fn intern(text: String) -> &'static str {
    static INTERNED: OnceLock<Mutex<BTreeSet<&'static str>>> = OnceLock::new();
    let mut set = INTERNED
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = set.get(text.as_str()) {
        return found;
    }
    let leaked: &'static str = Box::leak(text.into_boxed_str());
    set.insert(leaked);
    leaked
}

fn mismatch(estimator: EstimatorId, detail: &str) -> ScientificRefusal {
    ScientificRefusal {
        code: "strategy_incompatible",
        detail: format!("estimator {} {detail}", estimator.as_str()),
    }
}

/// The refusal of an estimator selected under the inference mode it does not implement.
///
/// The text keeps the wire form the builder has always reported for the mismatch.
#[must_use]
pub fn inference_mismatch(estimator: EstimatorId, requested_bayesian: bool) -> ScientificRefusal {
    let name = estimator.as_str();
    let detail = if requested_bayesian {
        format!(
            "estimator {name} is Frequentist, so it does not run under inference=Bayesian (it \
             would report a sampling interval under a Bayesian label). Omit estimator= to use \
             the Bayesian estimator, or use inference=Frequentist to run {name}"
        )
    } else {
        format!(
            "estimator {name} is Bayesian, so it does not run under inference=Frequentist. \
             Omit estimator= to use the Frequentist estimator, or use inference=Bayesian to \
             run {name}"
        )
    };
    ScientificRefusal { code: "estimator_inference_mismatch", detail }
}

/// The part of a query that selects among an estimator's design variants.
enum Facet {
    None,
    Randomized(RandomizedShape),
    Policy(PolicyShape),
    Longitudinal(LongitudinalRegimeMethod),
    TemporalDesign(TemporalDesignShape),
    /// Every shape the query can be read as (a static functional on a temporal class is read as
    /// either).
    Response(Vec<ResponseShape>),
    /// Every window shape the query can be read as, and whether it declares effect modifiers.
    Temporal {
        policies: Vec<TemporalPolicyShape>,
        conditional: bool,
    },
}

fn randomized_shape(query: &antecedent_core::RandomizedEffectQuery) -> RandomizedShape {
    match &query.design {
        RandomizationDesign::Bernoulli => {
            if query.received_treatment.is_some() {
                R::BernoulliCace
            } else if query.fixed_cuped.is_some() {
                R::BernoulliCuped
            } else if !query.ancova_covariates.is_empty() {
                R::BernoulliAncova
            } else {
                R::Bernoulli
            }
        }
        RandomizationDesign::Cluster { .. } if query.fixed_cuped.is_some() => R::ClusterCuped,
        RandomizationDesign::Cluster { .. } => R::Cluster,
        RandomizationDesign::Complete { .. } => R::Complete,
        RandomizationDesign::Stratified { .. } => R::Stratified,
        RandomizationDesign::Factorial2x2 { .. } => R::Factorial2x2,
        RandomizationDesign::MultiArm { .. } => R::MultiArm,
        RandomizationDesign::Switchback { .. } => R::Switchback,
    }
}

fn response_shape(functional: &ResponseFunctional) -> ResponseShape {
    match functional {
        ResponseFunctional::MeanCurve { .. } => S::MeanCurve,
        ResponseFunctional::AverageDerivative { .. } => S::AverageDerivative,
        ResponseFunctional::PointDerivative { .. } => S::PointDerivative,
        ResponseFunctional::DirectionalDerivative { .. } => S::DirectionalDerivative,
        ResponseFunctional::Jacobian { .. } => S::Jacobian,
        ResponseFunctional::InterventionResponse { .. } => S::InterventionResponse,
    }
}

fn facet_of(query: &CausalQuery, site: Option<Site>) -> Facet {
    match query {
        CausalQuery::RandomizedEffect(randomized) => {
            Facet::Randomized(randomized_shape(randomized))
        }
        CausalQuery::PolicyValue(policy) => Facet::Policy(if policy.multi_action.is_some() {
            PolicyShape::MultiAction
        } else if policy.mu0.is_empty() {
            PolicyShape::Ipw
        } else {
            PolicyShape::DoublyRobust
        }),
        CausalQuery::LongitudinalRegime(regime) => Facet::Longitudinal(regime.method),
        CausalQuery::TemporalDesign(design) => Facet::TemporalDesign(
            if matches!(design.assignment, TemporalAssignment::RegimeSurvival { .. }) {
                TemporalDesignShape::RegimeSurvival
            } else {
                TemporalDesignShape::Excursion
            },
        ),
        CausalQuery::TemporalEffect(effect) => Facet::Temporal {
            policies: vec![match &effect.policy {
                TemporalPolicy::Pulse { .. } => P::Pulse,
                TemporalPolicy::Sustained { .. } if effect.is_multi_step_sustained() => {
                    P::SustainedMultiStep
                }
                TemporalPolicy::Sustained { .. } => P::SustainedOneStep,
                TemporalPolicy::Dynamic { active_at, .. } if active_at.len() == 1 => P::Pulse,
                _ => P::SustainedMultiStep,
            }],
            conditional: !effect.effect_modifiers.is_empty(),
        },
        CausalQuery::Response(response) => {
            let functional = response_shape(&response.functional);
            // A level curve or joint response on a temporal ADMG reads the contemporaneous
            // projection, not the lagged template.
            let projected = site.is_some_and(|site| site.class == GraphClass::TemporalAdmg)
                && matches!(functional, S::MeanCurve | S::InterventionResponse);
            Facet::Response(vec![if response.is_temporal() && !projected {
                S::Temporal
            } else {
                functional
            }])
        }
        _ => Facet::None,
    }
}

fn design_mismatch(
    estimator: EstimatorId,
    read: &str,
    allowed: &dyn std::fmt::Debug,
) -> ScientificRefusal {
    mismatch(
        estimator,
        &format!(
            "does not estimate this design: the query is read as {read}; it answers {allowed:?}"
        ),
    )
}

fn check_design(
    estimator: EstimatorId,
    target: EstimatorTarget,
    bayesian_mode: bool,
    facet: &Facet,
) -> Result<(), ScientificRefusal> {
    match (target.design, facet) {
        (DesignTarget::Randomized(allowed), Facet::Randomized(shape))
            if !allowed.contains(shape) =>
        {
            Err(design_mismatch(estimator, &format!("{shape:?}"), &allowed))
        }
        (DesignTarget::Policy(allowed), Facet::Policy(shape)) if !allowed.contains(shape) => {
            Err(design_mismatch(estimator, &format!("{shape:?}"), &allowed))
        }
        (DesignTarget::Longitudinal(allowed), Facet::Longitudinal(method))
            if !allowed.contains(method) =>
        {
            Err(design_mismatch(estimator, &format!("{method:?}"), &allowed))
        }
        (DesignTarget::TemporalDesign(allowed), Facet::TemporalDesign(shape))
            if !allowed.contains(shape) =>
        {
            Err(design_mismatch(estimator, &format!("{shape:?}"), &allowed))
        }
        (
            DesignTarget::Temporal { frequentist, bayesian },
            Facet::Temporal { policies, conditional },
        ) => {
            let accept = if bayesian_mode { bayesian } else { frequentist };
            let allowed = if *conditional { accept.conditional } else { accept.plain };
            if policies.iter().any(|policy| allowed.contains(policy)) {
                Ok(())
            } else {
                let mode = if bayesian_mode { "Bayesian" } else { "Frequentist" };
                let modifiers = if *conditional { "with" } else { "without" };
                Err(design_mismatch(
                    estimator,
                    &format!("{policies:?} {modifiers} effect modifiers under {mode} inference"),
                    &allowed,
                ))
            }
        }
        (DesignTarget::Response { frequentist, bayesian }, Facet::Response(shapes)) => {
            let allowed = if bayesian_mode { bayesian } else { frequentist };
            if shapes.iter().any(|shape| allowed.contains(shape)) {
                Ok(())
            } else {
                let mode = if bayesian_mode { "Bayesian" } else { "Frequentist" };
                Err(design_mismatch(
                    estimator,
                    &format!("{shapes:?} under {mode} inference"),
                    &allowed,
                ))
            }
        }
        _ => Ok(()),
    }
}

/// The identifiers whose estimands the compiler can produce for an average or conditional
/// effect on `class` (the cell's identification route; `parity/supported_routes.toml` records
/// `backdoor.adjustment` on a DAG, `general.id` on an ADMG and `generalized.adjustment` on a
/// CPDAG or PAG). A time-indexed class has its own unfolded identifiers, which only the
/// temporal estimators read.
const fn route_identifiers(class: GraphClass) -> &'static [IdentifierId] {
    match class {
        GraphClass::Dag => &[
            IdentifierId::BackdoorAdjustment,
            IdentifierId::BackdoorEfficient,
            IdentifierId::Frontdoor,
            IdentifierId::Iv,
            IdentifierId::RdSharp,
            IdentifierId::GeneralId,
            IdentifierId::Auto,
        ],
        GraphClass::Admg | GraphClass::TemporalAdmg => &[IdentifierId::GeneralId],
        GraphClass::Cpdag | GraphClass::Pag => &[IdentifierId::GeneralizedAdjustment],
        _ => &[],
    }
}

/// The estimand methods an identifier returns.
const fn identifier_methods(identifier: IdentifierId) -> &'static [EstimandMethod] {
    match identifier {
        IdentifierId::BackdoorAdjustment | IdentifierId::GeneralizedAdjustment => {
            &[EstimandMethod::BackdoorAdjustment]
        }
        IdentifierId::BackdoorEfficient => &[EstimandMethod::BackdoorEfficient],
        IdentifierId::Frontdoor => &[EstimandMethod::FrontDoor],
        IdentifierId::Iv => &[EstimandMethod::Iv],
        IdentifierId::RdSharp => &[EstimandMethod::RdSharp],
        IdentifierId::GeneralId => &[EstimandMethod::GeneralId],
        IdentifierId::Auto => &[
            EstimandMethod::BackdoorAdjustment,
            EstimandMethod::FrontDoor,
            EstimandMethod::Iv,
            EstimandMethod::RdSharp,
            EstimandMethod::GeneralId,
        ],
        _ => &[],
    }
}

/// Whether a static estimator can run on the identification route of `site`: some identifier
/// of the route is on the estimator's allow-list and returns an estimand the estimator
/// consumes. An estimator with no static identifier allow-list (the temporal and
/// identifier-native families) is not an adjustment-route estimator and passes.
fn runs_on_route(estimator: EstimatorId, site: Site) -> bool {
    let static_route =
        IdentifierId::ALL.iter().any(|id| validate_static_pair(*id, estimator).is_ok());
    if !static_route {
        return true;
    }
    route_of(site).iter().any(|identifier| {
        validate_static_pair(*identifier, estimator).is_ok()
            && identifier_methods(*identifier)
                .iter()
                .any(|method| estimand_compatible_with_estimator(*method, &estimator))
    })
}

/// A tier background identifies by generalized adjustment whatever class it is read as.
fn route_of(site: Site) -> &'static [IdentifierId] {
    if site.tier { &[IdentifierId::GeneralizedAdjustment] } else { route_identifiers(site.class) }
}

fn check_graph(
    estimator: EstimatorId,
    target: EstimatorTarget,
    kind: K,
    site: Site,
) -> Result<(), ScientificRefusal> {
    let class = site.class;
    let scope = target
        .kind_graphs
        .iter()
        .find(|(scoped, _)| *scoped == kind)
        .map_or(target.graphs, |(_, scope)| *scope);
    if !scope.allows(class) {
        return Err(mismatch(
            estimator,
            &format!(
                "does not run on a {} graph for a {kind:?} query (it runs on: {scope:?})",
                class.as_str()
            ),
        ));
    }
    if site.posterior
        && target
            .posterior_not
            .iter()
            .any(|(scoped, classes)| *scoped == kind && classes.contains(&class))
    {
        return Err(mismatch(
            estimator,
            &format!(
                "does not run on a graph posterior over {} graphs for a {kind:?} query: the \
                 posterior is evaluated per atom by another estimator, or the estimator needs \
                 one fixed graph",
                class.as_str()
            ),
        ));
    }
    if site.tier && target.tier_not.contains(&kind) {
        return Err(mismatch(
            estimator,
            &format!("does not run a {kind:?} query on a tier background"),
        ));
    }
    if !site.posterior
        && target
            .posterior_only
            .iter()
            .any(|(scoped, classes)| *scoped == kind && classes.contains(&class))
    {
        return Err(mismatch(
            estimator,
            &format!(
                "runs a {kind:?} query on a {} graph only over a graph posterior (a fixed \
                 structure is taken by another estimator)",
                class.as_str()
            ),
        ));
    }
    let adjustment_route = matches!(kind, K::AverageEffect | K::ConditionalEffect)
        && (site.tier
            || !matches!(
                class,
                GraphClass::TemporalDag
                    | GraphClass::TemporalCpdag
                    | GraphClass::TemporalPag
                    | GraphClass::RandomizedTrial
            ));
    if adjustment_route && !runs_on_route(estimator, site) {
        return Err(mismatch(
            estimator,
            &format!(
                "cannot consume the estimand identification returns on a {} graph \
                 (identification route {:?})",
                class.as_str(),
                route_of(site)
            ),
        ));
    }
    Ok(())
}

/// The one validity check: kind, design variant, graph class and route, then inference mode.
/// `site` is `None` where the graph is not a premise of the question (a placeholder estimator
/// the builder bound itself).
fn check(
    estimator: EstimatorId,
    kind: K,
    bayesian_mode: bool,
    facet: &Facet,
    site: Option<Site>,
) -> Result<(), ScientificRefusal> {
    let target = estimator_target(estimator);
    if !target.query_kinds.contains(&kind) {
        return Err(mismatch(
            estimator,
            &format!(
                "does not estimate this query: it is a {kind:?} query and the estimator \
                 estimates {:?}",
                target.query_kinds
            ),
        ));
    }
    check_design(estimator, target, bayesian_mode, facet)?;
    if let Some(site) = site {
        check_graph(estimator, target, kind, site)?;
        if let Facet::Temporal { conditional: true, .. } = facet
            && !target.conditional_fixed_on.is_empty()
            && (site.posterior || !target.conditional_fixed_on.contains(&site.class))
        {
            return Err(mismatch(
                estimator,
                &format!(
                    "does not take effect modifiers on a {} graph{}: it adjusts for a \
                         named modifier on a fixed temporal DAG only",
                    site.class.as_str(),
                    if site.posterior { " posterior" } else { "" }
                ),
            ));
        }
        // A level curve or joint response posterior over ADMG atoms is evaluated per atom by
        // the identified-functional plug-in (`response_path.rs`), whatever was selected.
        if let Facet::Response(shapes) = facet
            && site.posterior
            && matches!(site.class, GraphClass::Admg | GraphClass::TemporalAdmg)
            && shapes.iter().any(|shape| matches!(shape, S::MeanCurve | S::InterventionResponse))
            && estimator != EstimatorId::FunctionalEffect
        {
            return Err(mismatch(
                estimator,
                "does not run a level or joint response over a graph posterior of ADMG \
                     atoms: each atom is evaluated by functional.effect",
            ));
        }
    }
    // The route runs another estimator under Bayesian inference for this kind: selecting this
    // one would be silently replaced, so it is refused.
    let replaced = bayesian_mode
        && target.bayesian_not.iter().any(|(scoped, classes)| {
            *scoped == kind
                && match (classes, site) {
                    (None, _) => true,
                    (Some(classes), Some(site)) => !site.posterior && classes.contains(&site.class),
                    (Some(_), None) => false,
                }
        });
    if replaced {
        return Err(inference_mismatch(estimator, true));
    }
    match (target.inference, bayesian_mode) {
        (InferenceTarget::Frequentist, true) => Err(inference_mismatch(estimator, true)),
        (InferenceTarget::Bayesian, false) => Err(inference_mismatch(estimator, false)),
        _ => Ok(()),
    }
}

/// Whether `estimator` estimates what `query` asks for under `inference` at `site`
/// (`None` skips the graph rules).
///
/// Scientific: a mismatch refuses in every build, before and independent of the support gate
/// (which is about evidence for an estimator that does answer the question). It reads the
/// declaration only and runs nothing.
///
/// # Errors
///
/// [`ScientificRefusal`] with `strategy_incompatible` when the estimator does not estimate this
/// query kind or design variant, `estimator_inference_mismatch` when it does not implement the
/// requested inference mode.
pub fn estimator_targets(
    query: &CausalQuery,
    inference: &InferenceMode,
    site: Option<Site>,
    estimator: EstimatorId,
) -> Result<(), ScientificRefusal> {
    check(
        estimator,
        query.kind(),
        matches!(inference, InferenceMode::Bayesian(_)),
        &facet_of(query, site),
        site,
    )
}

/// The query kind a support-matrix query axis value names.
const fn cell_kind(query: SupportQuery) -> K {
    match query {
        SupportQuery::AverageEffect => K::AverageEffect,
        SupportQuery::ConditionalEffect => K::ConditionalEffect,
        SupportQuery::Counterfactual => K::Counterfactual,
        SupportQuery::NestedCounterfactualEffect => K::NestedCounterfactual,
        SupportQuery::InterventionalDistribution => K::Distribution,
        SupportQuery::PathSpecificEffect => K::PathSpecific,
        SupportQuery::MediationEffect | SupportQuery::TemporalMediationEffect => K::Mediation,
        SupportQuery::PulseEffect
        | SupportQuery::SustainedEffect
        | SupportQuery::TemporalConditionalPulse
        | SupportQuery::TemporalConditionalSustained => K::TemporalEffect,
        SupportQuery::ResponseCurve
        | SupportQuery::AverageDerivative
        | SupportQuery::PointDerivative
        | SupportQuery::SemiElasticity
        | SupportQuery::Elasticity
        | SupportQuery::DirectionalDerivative
        | SupportQuery::ResponseJacobian
        | SupportQuery::InterventionResponse => K::Response,
        SupportQuery::TransportQuery => K::Transport,
        SupportQuery::InterferenceQuery => K::Interference,
        SupportQuery::AnomalyAttribution => K::AnomalyAttribution,
        SupportQuery::ChangeAttribution => K::ChangeAttribution,
    }
}

/// The graph class a cell's graph axis names (the tier-background extras read as a DAG).
const fn cell_graph_class(class: SupportGraphClass) -> GraphClass {
    match class {
        SupportGraphClass::Dag | SupportGraphClass::CoDetermined | SupportGraphClass::Unknown => {
            GraphClass::Dag
        }
        SupportGraphClass::Admg => GraphClass::Admg,
        SupportGraphClass::Cpdag => GraphClass::Cpdag,
        SupportGraphClass::Pag => GraphClass::Pag,
        SupportGraphClass::TemporalDag => GraphClass::TemporalDag,
        SupportGraphClass::TemporalCpdag => GraphClass::TemporalCpdag,
        SupportGraphClass::TemporalPag => GraphClass::TemporalPag,
        SupportGraphClass::TemporalAdmg => GraphClass::TemporalAdmg,
    }
}

/// The response shapes a response cell is read as: its functional, plus the temporal
/// attachment on a temporal class (a `TemporalAdmg` carries both readings).
fn cell_response_shapes(cell: SupportCell) -> Vec<ResponseShape> {
    let functional = match cell.query {
        SupportQuery::ResponseCurve => S::MeanCurve,
        SupportQuery::AverageDerivative => S::AverageDerivative,
        SupportQuery::PointDerivative | SupportQuery::SemiElasticity | SupportQuery::Elasticity => {
            S::PointDerivative
        }
        SupportQuery::DirectionalDerivative => S::DirectionalDerivative,
        SupportQuery::ResponseJacobian => S::Jacobian,
        _ => S::InterventionResponse,
    };
    match cell.graph_class {
        SupportGraphClass::TemporalDag
        | SupportGraphClass::TemporalCpdag
        | SupportGraphClass::TemporalPag => vec![S::Temporal],
        // A level curve or joint response on a temporal ADMG reads the contemporaneous
        // projection (the plug-in); only a derivative reads the lagged template.
        SupportGraphClass::TemporalAdmg
            if matches!(functional, S::MeanCurve | S::InterventionResponse) =>
        {
            vec![functional]
        }
        SupportGraphClass::TemporalAdmg => vec![functional, S::Temporal],
        _ => vec![functional],
    }
}

/// [`estimator_targets`] for a support-matrix cell: whether `estimator` estimates the
/// quantity the cell's query names under the cell's inference mode. Runs nothing; this is the
/// reading the generated `parity/estimator_validity.toml` records.
///
/// A response cell is read at its functional's granularity (and as temporal on a temporal
/// class); a temporal-effect cell is read at its window shape (a sustained cell covers both
/// one-step and multi-step schedules) and its effect-modifier flag; structure and tier are
/// premises of the site.
///
/// # Errors
///
/// As [`estimator_targets`].
pub fn estimator_targets_cell(
    cell: SupportCell,
    estimator: EstimatorId,
) -> Result<(), ScientificRefusal> {
    let kind = cell_kind(cell.query);
    let facet = match cell.query {
        SupportQuery::PulseEffect => {
            Facet::Temporal { policies: vec![P::Pulse], conditional: false }
        }
        SupportQuery::TemporalConditionalPulse => {
            Facet::Temporal { policies: vec![P::Pulse], conditional: true }
        }
        // A sustained cell covers one-step and multi-step schedules.
        SupportQuery::SustainedEffect => Facet::Temporal {
            policies: vec![P::SustainedOneStep, P::SustainedMultiStep],
            conditional: false,
        },
        SupportQuery::TemporalConditionalSustained => Facet::Temporal {
            policies: vec![P::SustainedOneStep, P::SustainedMultiStep],
            conditional: true,
        },
        _ if kind == K::Response => Facet::Response(cell_response_shapes(cell)),
        _ => Facet::None,
    };
    let site = Site {
        class: cell_graph_class(cell.graph_class),
        tier: matches!(
            cell.graph_class,
            SupportGraphClass::CoDetermined | SupportGraphClass::Unknown
        ),
        posterior: cell.structure == SupportStructure::GraphPosterior,
    };
    check(estimator, kind, cell.inference == SupportInference::Bayesian, &facet, Some(site))?;
    estimator_validation(estimator, cell.validation)
}

/// Whether the estimator's route implements the validation `suite`.
///
/// # Errors
///
/// `strategy_incompatible` when the route never reads the suite (it would silently run none).
pub fn estimator_validation(
    estimator: EstimatorId,
    suite: SupportValidation,
) -> Result<(), ScientificRefusal> {
    if estimator.target().validation == ValidationTarget::NoneOnly
        && suite != SupportValidation::None
    {
        return Err(mismatch(
            estimator,
            &format!(
                "does not implement the {suite:?} validation suite: its route reads no refuter \
                 or predictive check, so the suite is not implemented for this estimator; run \
                 it with the `none` suite"
            ),
        ));
    }
    Ok(())
}
