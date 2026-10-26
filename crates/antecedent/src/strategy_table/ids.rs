//! Identifier / estimator ids, defaults, and pair allowlists.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::OnceLock;

use antecedent_core::{AssumptionRecord, IdentificationStatus};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_identify::IdentificationResult;

use crate::error::CausalError;

/// Wire names in [`IdentifierId::ALL`] order, derived from [`IdentifierId::as_str`].
fn identifier_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| IdentifierId::ALL.iter().map(IdentifierId::as_str).collect()).as_slice()
}

/// Wire names in [`EstimatorId::ALL`] order, derived from [`EstimatorId::as_str`].
fn estimator_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| EstimatorId::ALL.iter().map(EstimatorId::as_str).collect()).as_slice()
}

/// Parse a closed-set id through a name index built once from [`ALL`] / [`as_str`], so a new
/// variant cannot compile into `estimator_data` / `identifier_data` and still miss `FromStr`.
fn parse_closed<T: Copy>(
    s: &str,
    index: &'static HashMap<&'static str, T>,
    kind: &'static str,
    expected: &'static [&'static str],
) -> Result<T, UnknownStrategy> {
    index.get(s).copied().ok_or_else(|| UnknownStrategy {
        kind,
        got: s.to_string(),
        expected,
    })
}

/// Name -> id for every identifier, built once.
fn identifier_index() -> &'static HashMap<&'static str, IdentifierId> {
    static INDEX: OnceLock<HashMap<&'static str, IdentifierId>> = OnceLock::new();
    INDEX.get_or_init(|| IdentifierId::ALL.iter().map(|id| (id.as_str(), *id)).collect())
}

/// Name -> id for every estimator, built once.
fn estimator_index() -> &'static HashMap<&'static str, EstimatorId> {
    static INDEX: OnceLock<HashMap<&'static str, EstimatorId>> = OnceLock::new();
    INDEX.get_or_init(|| EstimatorId::ALL.iter().map(|id| (id.as_str(), *id)).collect())
}

/// Declare a closed-set id enum together with its `ALL` catalog, both
/// generated from one variant list.
///
/// `FromStr` walks `ALL`, so a variant added to the enum but missing from a
/// hand-maintained catalog would compile and then fail to parse. Generating
/// both from the same list makes that omission impossible.
macro_rules! closed_id_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident, )+
        }
        all_doc = $all_doc:literal;
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $( $(#[$vmeta])* $variant, )+
        }

        impl $name {
            #[doc = $all_doc]
            pub const ALL: &'static [$name] = &[ $( Self::$variant, )+ ];
        }
    };
}

closed_id_enum! {
/// Closed set of identification strategies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum IdentifierId {
    /// Classic backdoor adjustment-set search.
    BackdoorAdjustment,
    /// Efficient (optimal) backdoor adjustment.
    BackdoorEfficient,
    /// Front-door identification.
    Frontdoor,
    /// Instrumental-variable identification.
    Iv,
    /// Sharp regression discontinuity.
    RdSharp,
    /// Temporal unfolded backdoor.
    TemporalBackdoorUnfolded,
    /// Lagged instrument on the unfolded window graph.
    TemporalIvUnfolded,
    /// Class-aware / generalized adjustment (PAG-safe).
    GeneralizedAdjustment,
    /// Shpitser–Pearl general ID (semi-Markovian).
    GeneralId,
    /// Path-restricted natural effects.
    PathSpecificNatural,
    /// Pairwise backdoor identification for response functionals.
    ResponseBackdoor,
    /// Structural-mechanism identification for staged counterfactuals.
    GcmParametric,
    /// Conservative single-source sID (direct / S-admissible / singleton c-component).
    TransportSid,
    /// Design-based interference: assignment design identifies the exposure contrast.
    InterferenceDesign,
    /// Randomized trial assignment directly identifies the ITT.
    RandomizedDesign,
    /// Conditional dose response under supplied density and group exchangeability.
    ContinuousDoseExchangeability,
    /// `AutoIdentifier` — all applicable estimands, no silent estimator choice.
    Auto,
}
all_doc = "Every closed-set identifier, in declaration order (powers [`UnknownStrategy::expected`]).";
}

/// Per-identifier data-only facts backing [`IdentifierId::as_str`],
/// [`IdentifierId::is_dag_only`], and [`identify_provenance_step`].
///
/// Purely descriptive — the real behavioral dispatch lives in
/// [`identify_static_query_with_rd`] / [`identify_pag`] / [`identify_admg`], which stay
/// exhaustive `match`es and are not table-driven.
pub(super) struct IdentifierData {
    name: &'static str,
    is_dag_only: bool,
    pub(super) provenance: (&'static str, &'static str),
}

/// Provenance pair for the common case where the family and step names coincide.
const fn same_step(name: &'static str) -> (&'static str, &'static str) {
    (name, name)
}

pub(super) const fn identifier_data(id: IdentifierId) -> IdentifierData {
    match id {
        IdentifierId::BackdoorAdjustment => IdentifierData {
            name: "backdoor.adjustment",
            is_dag_only: true,
            provenance: same_step("identify.backdoor"),
        },
        IdentifierId::BackdoorEfficient => IdentifierData {
            name: "backdoor.efficient",
            is_dag_only: true,
            provenance: same_step("identify.efficient_backdoor"),
        },
        IdentifierId::Frontdoor => IdentifierData {
            name: "frontdoor",
            is_dag_only: true,
            provenance: same_step("identify.frontdoor"),
        },
        IdentifierId::Iv => {
            IdentifierData { name: "iv", is_dag_only: true, provenance: same_step("identify.iv") }
        }
        IdentifierId::RdSharp => IdentifierData {
            name: "rd.sharp",
            is_dag_only: true,
            provenance: ("identify.rd_design", "identify.rd_sharp"),
        },
        IdentifierId::TemporalBackdoorUnfolded => IdentifierData {
            name: "temporal.backdoor.unfolded",
            is_dag_only: true,
            provenance: ("identify.temporal_backdoor", "identify.temporal_backdoor_unfolded"),
        },
        IdentifierId::TemporalIvUnfolded => IdentifierData {
            name: "temporal.iv.unfolded",
            is_dag_only: true,
            provenance: ("identify.temporal_iv", "identify.temporal_iv_unfolded"),
        },
        IdentifierId::GeneralizedAdjustment => IdentifierData {
            name: "generalized.adjustment",
            is_dag_only: false,
            provenance: same_step("identify.generalized_adjustment"),
        },
        IdentifierId::GeneralId => IdentifierData {
            name: "general.id",
            is_dag_only: true,
            provenance: same_step("identify.general_id"),
        },
        IdentifierId::PathSpecificNatural => IdentifierData {
            name: "path_specific.natural",
            is_dag_only: true,
            provenance: same_step("identify.path_specific"),
        },
        IdentifierId::GcmParametric => IdentifierData {
            name: "gcm.parametric",
            is_dag_only: true,
            provenance: same_step("counterfactual.aap"),
        },
        IdentifierId::TransportSid => IdentifierData {
            name: "transport.sid",
            is_dag_only: false,
            provenance: same_step("identify.transport.sid"),
        },
        IdentifierId::InterferenceDesign => IdentifierData {
            name: "interference.design",
            is_dag_only: true,
            provenance: same_step("identify.interference.design"),
        },
        IdentifierId::RandomizedDesign => IdentifierData {
            name: "randomized.design",
            is_dag_only: false,
            provenance: same_step("identify.randomized.design"),
        },
        IdentifierId::ContinuousDoseExchangeability => IdentifierData {
            name: "policy.continuous_dose_exchangeability",
            is_dag_only: false,
            provenance: same_step("identify.policy.continuous_dose"),
        },
        IdentifierId::ResponseBackdoor => IdentifierData {
            name: "response.backdoor",
            is_dag_only: true,
            provenance: ("identify.response", "identify.response_backdoor"),
        },
        IdentifierId::Auto => IdentifierData {
            name: "auto",
            is_dag_only: true,
            provenance: same_step("identify.auto"),
        },
    }
}

impl IdentifierId {
    /// Canonical wire id.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        identifier_data(*self).name
    }

    /// Whether this identifier requires a DAG (not a raw PAG).
    #[must_use]
    pub const fn is_dag_only(&self) -> bool {
        identifier_data(*self).is_dag_only
    }
}

impl FromStr for IdentifierId {
    type Err = UnknownStrategy;

    /// Parse a wire / builder id string.
    ///
    /// # Errors
    ///
    /// [`UnknownStrategy`] when `s` does not match any closed-set identifier name.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_closed(s, identifier_index(), "identifier", identifier_names())
    }
}

impl std::fmt::Display for IdentifierId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

closed_id_enum! {
/// Closed set of estimators.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum EstimatorId {
    /// OLS g-computation / linear adjustment ATE.
    LinearAdjustmentAte,
    /// Inverse-probability weighting.
    PropensityWeighting,
    /// Propensity-score matching.
    PropensityMatching,
    /// Propensity stratification.
    PropensityStratification,
    /// Covariate distance matching.
    DistanceMatching,
    /// Augmented IPW.
    Aipw,
    /// GLM (logit) adjustment.
    GlmAdjustment,
    /// Linear front-door two-stage (product of coefficients).
    FrontDoorTwoStage,
    /// Plug-in of the nonparametric front-door functional.
    FrontDoorFunctional,
    /// Wald IV.
    IvWald,
    /// Two-stage least squares.
    Iv2Sls,
    /// Bayesian joint linear Gaussian IV with fixed unit disturbance loading.
    BayesianIvJointLinear,
    /// Sharp local-linear RD.
    RdSharp,
    /// Bayesian local-linear sharp RD with fixed bandwidth.
    BayesianRdLocalLinear,
    /// Bayesian g-computation.
    BayesianGcomp,
    /// Bayesian quadratic-basis g-computation with an explicit shrinkage prior.
    BayesianBasisGcomp,
    /// Cross-fitted orthogonal ATE with shared Bayesian-bootstrap nuisance refits.
    BayesianRobustAte,
    /// Bayesian Gaussian interaction model, averaged over observed modifiers.
    BayesianConditional,
    /// Temporal linear adjustment.
    TemporalLinearAdjustment,
    /// Bayesian g-computation on the lag-aligned temporal adjustment design.
    BayesianTemporalGcomp,
    /// Untempered Gaussian-likelihood posterior of a single-step temporal linear effect, under a
    /// declared serially independent innovation. Unlicensed; calibration unmeasured.
    BayesianTemporalWhiteInnovation,
    /// Mean-scale contrast of a Bernoulli logit temporal model over independent panel subjects:
    /// the unpenalized maximum-likelihood fit with a delta-method standard error. Unlicensed;
    /// calibration unmeasured.
    TemporalGlmLinkContrast,
    /// Laplace posterior of the same mean-scale contrast under a Gaussian coefficient prior.
    /// Unlicensed; calibration unmeasured.
    BayesianTemporalGlmLinkContrast,
    /// Wald contrast of a lagged instrument on the unfolded window graph: the effect of treatment
    /// under the declared exclusion and exogeneity of the lag. Unlicensed; calibration unmeasured.
    TemporalLaggedInstrument,
    /// Reduced-form ratio posterior of the same lagged-instrument effect. Unlicensed; calibration
    /// unmeasured.
    BayesianTemporalLaggedInstrument,
    /// Sequential g-computation of a sustained treatment window.
    TemporalSequentialGcomp,
    /// Discrete plug-in evaluation of an identified interventional distribution.
    FunctionalDistribution,
    /// Discrete plug-in evaluation of an identified scalar functional (ATE / path NE).
    FunctionalEffect,
    /// Static linear structural mediation.
    StaticMediationLinear,
    /// Conditional linear adjustment (effect modifiers).
    ConditionalLinearAdjustment,
    /// Temporal linear mediation (path-product).
    TemporalMediation,
    /// Bayesian response.temporal.bayesian estimator.
    TemporalResponseBayesian,
    /// Bayesian response.bayesian estimator.
    ResponseBayesian,
    /// Bayesian temporal.mediation.bayesian estimator.
    BayesianTemporalMediation,
    /// Kennedy-style doubly robust continuous response curve / point derivative.
    ResponseKennedyDr,
    /// Riesz-representer average derivative estimator.
    ResponseRieszAde,
    /// Low-dimensional GAM plug-in Jacobian / directional derivative.
    ResponseGamDerivative,
    /// Additive-GAM g-computation for static, shifted, or stochastic interventions.
    ResponseInterventionGcomp,
    /// Cell-saturated AIPW for discrete joint `Set` interventions.
    CellAipw,
    /// Temporal dose-over-horizon / policy-path g-computation (ADR 0021).
    TemporalResponseGcomp,
    /// Fitted additive GCM mechanisms with abduction–action–prediction ITE.
    GcmFit,
    /// Bayesian-bootstrap anomaly attribution under a fitted GCM.
    GcmFitBayesian,
    /// Bayesian-bootstrap distribution-change attribution under a fitted GCM.
    GcmAttributionBayesian,
    /// Dahabreh trial-to-target IPW (Direct / S-admissible standardize only).
    TransportTrialIpw,
    /// Bayesian-bootstrap trial-to-target IPW conditional on supplied probabilities.
    TransportTrialBayesianBootstrap,
    /// Horvitz–Thompson / Hájek exposure contrast under a known assignment design.
    InterferenceHtHajek,
    /// Cluster-randomized total exposure contrast under partial interference.
    InterferenceClusterNeyman,
    /// Exact two-stage saturation exposure contrast, point and variance proxy only.
    InterferenceSaturationExact,
    /// Supplied-propensity observational network exposure point contrast.
    InterferenceObservationalIpw,
    /// Bayesian finite-network Gaussian potential-outcome exposure contrast.
    InterferenceBayesianGaussian,
    /// Cluster-level Bayesian-bootstrap posterior of the cluster total exposure contrast
    /// (unlicensed, uncalibrated).
    InterferenceClusterBayesianBootstrap,
    /// Horvitz--Thompson design-based Bernoulli ITT.
    RandomizedHt,
    /// Local Epanechnikov-kernel response with supplied inverse dose density.
    ContinuousDoseKernel,
    /// Bernoulli ITT adjusted by a declared pre-assignment covariate with fixed coefficient.
    RandomizedFixedCupedHt,
    /// Fitted multi-covariate ANCOVA under independent Bernoulli assignment.
    RandomizedAncova,
    /// Wald CACE/LATE under independent Bernoulli encouragement.
    RandomizedWaldCace,
    /// Switchback unit-period HT with independent-sequence variance.
    RandomizedSwitchbackHt,
    /// Difference-in-means ITT for complete or stratified randomization.
    RandomizedNeyman,
    /// Doubly robust held-out policy value under known randomized propensities.
    RandomizedDrPolicy,
    /// Fixed-policy inverse-probability value under known randomized propensities.
    RandomizedIpwPolicy,
    /// Fixed multi-action policy IPW value under known randomized action probabilities.
    RandomizedMultiActionIpwPolicy,
    /// Randomized product-limit survival/RMST or competing-risk incidence.
    RandomizedSurvivalProductLimit,
    /// Prespecified sequential inverse-probability regime value.
    LongitudinalIpwRegime,
    /// Prespecified sequential g-formula value from supplied conditional rewards.
    LongitudinalGFormulaRegime,
    /// Sequentially augmented regime value from supplied Q scores.
    LongitudinalSequentialDrRegime,
    /// Additive binary marginal structural model with stabilized IPTW.
    LongitudinalMarginalStructuralModel,
    /// Cross-fitted DML / AIPW average treatment effect.
    Dml,
    /// Doubly robust CATE learner (DRLearner).
    DrLearner,
    /// Native honest causal forest CATE.
    CausalForest,
    /// Bayesian panel difference-in-differences posterior (unlicensed, uncalibrated).
    BayesianPanelDid,
    /// Bayesian simplex synthetic-control posterior (unlicensed, uncalibrated).
    BayesianSyntheticControl,
    /// Bayesian fuzzy-jump and kink local-ratio posterior (unlicensed, uncalibrated).
    BayesianLocalRatio,
    /// Piecewise-exponential posterior of the randomized survival and restricted-mean
    /// contrast (unlicensed, uncalibrated).
    BayesianPiecewiseSurvival,
    /// Dirichlet-multinomial grid-distribution posterior of the randomized potential-outcome
    /// quantiles and their difference (unlicensed, uncalibrated).
    BayesianRandomizedQuantile,
    /// Per-group polynomial dose-curve posterior of a fixed group-to-dose policy value
    /// (unlicensed, uncalibrated).
    BayesianDosePolicy,
    /// Posterior of the arm contrasts of a randomized trial over the trial's own units, the same
    /// target as the design-based row (unlicensed, uncalibrated).
    BayesianRandomizedFinitePopulation,
    /// Posterior of the arm contrasts of a randomized trial in the population the units are
    /// sampled from (unlicensed, uncalibrated).
    BayesianRandomizedSuperpopulation,
    /// Bayesian-bootstrap posterior of the complier effect under known Bernoulli encouragement
    /// (unlicensed, uncalibrated).
    BayesianComplierEffect,
    /// Posterior of the switchback direct effect under a declared carryover and within-sequence
    /// dependence (unlicensed, uncalibrated).
    BayesianSwitchback,
    /// Posterior of a fixed binary policy's value under known randomization (unlicensed,
    /// uncalibrated).
    BayesianBinaryPolicyValue,
    /// Posterior of a prespecified sequential regime's value under known sequential
    /// randomization (unlicensed, uncalibrated).
    BayesianSequentialRegimeValue,
    /// Normal interval of the observational cross-fit sequential contrast (unlicensed,
    /// uncalibrated).
    SequentialCrossfitContrastWald,
    /// Modular posterior of the observational cross-fit sequential contrast (unlicensed,
    /// uncalibrated).
    BayesianSequentialCrossfitContrast,
    /// Frequentist within-period history-matched contrast (unlicensed, uncalibrated).
    PeriodHistoryMatching,
    /// Bayesian outcome-model effect on the treated of a period-and-history design
    /// (unlicensed, uncalibrated). A model-based estimand, not a matching posterior.
    BayesianPeriodHistoryOutcomeModel,
    /// Cluster-robust Horvitz-Thompson excursion contrasts of a sequentially randomized
    /// unit-period design (unlicensed, uncalibrated).
    TemporalDesignExcursion,
    /// Cluster Bayesian-bootstrap posterior of the same excursion contrasts (unlicensed,
    /// uncalibrated).
    BayesianTemporalDesignExcursion,
    /// Inverse-probability weighted regime survival with a whole-procedure subject bootstrap
    /// (unlicensed, uncalibrated).
    TemporalRegimeSurvival,
    /// Joint-likelihood parametric g-formula posterior of regime survival (unlicensed,
    /// uncalibrated).
    BayesianTemporalRegimeSurvival,
    /// Sharp Balke-Pearl bounds on the average effect of a binary treatment under a binary
    /// instrument, with an Imbens-Manski row-bootstrap interval (unlicensed, uncalibrated).
    IvBoundsSharp,
    /// Bayesian-bootstrap posterior of the same bounds' endpoints (unlicensed, uncalibrated).
    BayesianIvBounds,
}
all_doc = "Every closed-set estimator, in declaration order (powers [`UnknownStrategy::expected`]).";
}

impl EstimatorId {
    /// Default response estimator for `functional` when the caller did not override it.
    #[must_use]
    pub const fn default_for_response(functional: &antecedent_core::ResponseFunctional) -> Self {
        use antecedent_core::ResponseFunctional;
        match functional {
            ResponseFunctional::MeanCurve { .. } | ResponseFunctional::PointDerivative { .. } => {
                Self::ResponseKennedyDr
            }
            ResponseFunctional::AverageDerivative { .. } => Self::ResponseRieszAde,
            ResponseFunctional::DirectionalDerivative { .. }
            | ResponseFunctional::Jacobian { .. } => Self::ResponseGamDerivative,
            ResponseFunctional::InterventionResponse { .. } => Self::ResponseInterventionGcomp,
        }
    }

    /// Pulse or sustained temporal effect procedure: sequential g-computation
    /// for a multi-step sustained schedule, otherwise the single-step Bayesian
    /// g-computation or frequentist linear adjustment.
    #[must_use]
    pub(crate) fn temporal_effect_procedure(
        query: &antecedent_core::TemporalEffectQuery,
        bayesian: bool,
    ) -> Self {
        if query.is_multi_step_sustained() {
            Self::TemporalSequentialGcomp
        } else if bayesian {
            Self::BayesianTemporalGcomp
        } else {
            Self::TemporalLinearAdjustment
        }
    }

    /// [`Self::temporal_effect_procedure`] for an inference mode.
    #[must_use]
    pub(crate) fn temporal_effect_for(
        query: &antecedent_core::TemporalEffectQuery,
        inference: &crate::InferenceMode,
    ) -> Self {
        Self::temporal_effect_procedure(
            query,
            matches!(inference, crate::InferenceMode::Bayesian(_)),
        )
    }

    /// Temporal response estimator of an inference mode.
    #[must_use]
    pub(crate) const fn temporal_response_for(inference: &crate::InferenceMode) -> Self {
        match inference {
            crate::InferenceMode::Frequentist => Self::TemporalResponseGcomp,
            crate::InferenceMode::Bayesian(_) => Self::TemporalResponseBayesian,
        }
    }

    /// Static response estimator of an inference mode: the functional's
    /// default under frequentist inference, the Bayesian response otherwise.
    #[must_use]
    pub(crate) const fn static_response_for(
        functional: &antecedent_core::ResponseFunctional,
        inference: &crate::InferenceMode,
    ) -> Self {
        use antecedent_core::ResponseFunctional;
        match inference {
            crate::InferenceMode::Frequentist => Self::default_for_response(functional),
            // A Bayesian derivative keeps the derivative estimator's id, as on a Dag: the
            // posterior is read from the same local fit, not from the Bayesian level curve.
            crate::InferenceMode::Bayesian(_) => match functional {
                ResponseFunctional::AverageDerivative { .. }
                | ResponseFunctional::PointDerivative { .. }
                | ResponseFunctional::DirectionalDerivative { .. }
                | ResponseFunctional::Jacobian { .. } => Self::default_for_response(functional),
                _ => Self::ResponseBayesian,
            },
        }
    }

    /// Temporal mediation estimator of an inference mode.
    #[must_use]
    pub(crate) const fn temporal_mediation_for(inference: &crate::InferenceMode) -> Self {
        match inference {
            crate::InferenceMode::Frequentist => Self::TemporalMediation,
            crate::InferenceMode::Bayesian(_) => Self::BayesianTemporalMediation,
        }
    }
}

/// Per-estimator data-only facts backing [`EstimatorId::as_str`],
/// [`EstimatorId::parallel_task_dimension`], [`EstimatorId::kernel_label`], and
/// [`estimate_provenance_step`].
///
/// The static-ATE identifier allow-list is [`Self::compatible`]. Fit dispatch in
/// [`estimate_static_effect`] / [`estimate_static_effect_default`] and
/// [`estimand_compatible_with_estimator`] stay exhaustive `match`es.
pub(super) struct EstimatorData {
    name: &'static str,
    parallel_task_dimension: &'static str,
    kernel_label: &'static str,
    pub(super) provenance: (&'static str, &'static str),
    /// Identifiers [`validate_static_pair`] accepts with this estimator.
    compatible: &'static [IdentifierId],
}

const ADJUSTMENT: &[IdentifierId] = &[
    IdentifierId::BackdoorAdjustment,
    IdentifierId::BackdoorEfficient,
    IdentifierId::GeneralizedAdjustment,
    IdentifierId::Auto,
];
const BACKDOOR_AND_AUTO: &[IdentifierId] =
    &[IdentifierId::BackdoorAdjustment, IdentifierId::BackdoorEfficient, IdentifierId::Auto];
const FRONTDOOR_AND_AUTO: &[IdentifierId] = &[IdentifierId::Frontdoor, IdentifierId::Auto];
const IV_AND_AUTO: &[IdentifierId] = &[IdentifierId::Iv, IdentifierId::Auto];
const RD: &[IdentifierId] = &[IdentifierId::RdSharp];
const RD_AND_AUTO: &[IdentifierId] = &[IdentifierId::RdSharp, IdentifierId::Auto];
const INTERFERENCE: &[IdentifierId] = &[IdentifierId::InterferenceDesign];
const RANDOMIZED: &[IdentifierId] = &[IdentifierId::RandomizedDesign];
const DOSE: &[IdentifierId] = &[IdentifierId::ContinuousDoseExchangeability];
const GENERAL_ID: &[IdentifierId] = &[IdentifierId::GeneralId];
const NONE: &[IdentifierId] = &[];

// Keeping this as an exhaustive `match` rather than an indexed table preserves the compile
// error when a new variant is added and its row is forgotten.
#[allow(
    clippy::too_many_lines,
    reason = "one data row per EstimatorId in an exhaustive match; each row is independent, so the length is the variant count"
)]
pub(super) const fn estimator_data(id: EstimatorId) -> EstimatorData {
    match id {
        EstimatorId::LinearAdjustmentAte => EstimatorData {
            compatible: ADJUSTMENT,
            name: "linear.adjustment.ate",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer",
            provenance: ("estimate.linear_adjustment", "estimate.linear_adjustment_ate"),
        },
        EstimatorId::PropensityWeighting => EstimatorData {
            compatible: ADJUSTMENT,
            name: "propensity.weighting",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ipw",
            provenance: ("estimate.propensity", "estimate.propensity_weighting"),
        },
        EstimatorId::PropensityMatching => EstimatorData {
            compatible: ADJUSTMENT,
            name: "propensity.matching",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "matching",
            provenance: ("estimate.propensity", "estimate.propensity_matching"),
        },
        EstimatorId::PropensityStratification => EstimatorData {
            compatible: ADJUSTMENT,
            name: "propensity.stratification",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "propensity.stratification",
            provenance: ("estimate.propensity", "estimate.propensity_stratification"),
        },
        EstimatorId::DistanceMatching => EstimatorData {
            compatible: ADJUSTMENT,
            name: "distance.matching",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "matching",
            provenance: ("estimate.matching", "estimate.distance_matching"),
        },
        EstimatorId::Aipw => EstimatorData {
            compatible: ADJUSTMENT,
            name: "aipw",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "aipw",
            provenance: same_step("estimate.aipw"),
        },
        EstimatorId::GlmAdjustment => EstimatorData {
            compatible: ADJUSTMENT,
            name: "glm.adjustment",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "glm.logit",
            provenance: ("estimate.glm_adjustment", "estimate.glm_adjustment_ate"),
        },
        EstimatorId::FrontDoorTwoStage => EstimatorData {
            compatible: FRONTDOOR_AND_AUTO,
            name: "frontdoor.linear_two_stage",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "frontdoor.linear_two_stage",
            provenance: ("estimate.frontdoor", "estimate.frontdoor_two_stage"),
        },
        EstimatorId::FrontDoorFunctional => EstimatorData {
            compatible: FRONTDOOR_AND_AUTO,
            name: "frontdoor.functional",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "frontdoor.functional",
            provenance: same_step("estimate.frontdoor_functional"),
        },
        EstimatorId::IvWald => EstimatorData {
            compatible: IV_AND_AUTO,
            name: "iv.wald",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "iv.wald",
            provenance: ("estimate.iv", "estimate.wald_iv"),
        },
        EstimatorId::Iv2Sls => EstimatorData {
            compatible: IV_AND_AUTO,
            name: "iv.2sls",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "2sls",
            provenance: ("estimate.iv", "estimate.two_stage_least_squares"),
        },
        EstimatorId::BayesianIvJointLinear => EstimatorData {
            compatible: IV_AND_AUTO,
            name: "iv.bayesian_joint_linear",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_iv.joint_fixed_loading",
            provenance: ("estimate.bayesian_iv", "estimate.bayesian_iv_joint_fixed_loading"),
        },
        EstimatorId::RdSharp => EstimatorData {
            compatible: RD,
            name: "rd.sharp",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "rd.local_linear",
            provenance: ("estimate.rd", "estimate.rd_sharp"),
        },
        EstimatorId::BayesianRdLocalLinear => EstimatorData {
            compatible: RD_AND_AUTO,
            name: "rd.bayesian_local_linear",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_rd.local_linear",
            provenance: ("estimate.bayesian_rd", "estimate.bayesian_rd_local_linear"),
        },
        EstimatorId::BayesianConditional => EstimatorData {
            compatible: ADJUSTMENT,
            name: "conditional.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.conditional",
            provenance: same_step("estimate.bayesian_conditional"),
        },
        EstimatorId::BayesianGcomp => EstimatorData {
            compatible: ADJUSTMENT,
            name: "bayesian.gcomp",
            parallel_task_dimension: "analysis",
            kernel_label: "ols.faer",
            provenance: same_step("estimate.bayesian_gcomp"),
        },
        EstimatorId::BayesianBasisGcomp => EstimatorData {
            compatible: ADJUSTMENT,
            name: "bayesian.basis.gcomp",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_basis_gcomp.quadratic_tz_z2",
            provenance: same_step("estimate.bayesian_basis_gcomp"),
        },
        EstimatorId::BayesianRobustAte => EstimatorData {
            compatible: BACKDOOR_AND_AUTO,
            name: "bayesian.robust_ate",
            parallel_task_dimension: "bootstrap.draw",
            kernel_label: "bayesian_bootstrap.aipw",
            provenance: same_step("estimate.bayesian_robust_ate"),
        },
        EstimatorId::TemporalSequentialGcomp => EstimatorData {
            compatible: NONE,
            name: "temporal.sequential.gcomp",
            parallel_task_dimension: "mechanism",
            kernel_label: "ols.faer.temporal.sequential",
            provenance: same_step("estimate.temporal_sequential"),
        },
        EstimatorId::TemporalLinearAdjustment => EstimatorData {
            compatible: NONE,
            name: "temporal.linear.adjustment",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer.temporal",
            provenance: ("estimate.temporal_linear", "estimate.temporal_linear_adjustment"),
        },
        EstimatorId::BayesianTemporalGcomp => EstimatorData {
            compatible: NONE,
            name: "bayesian.temporal.gcomp",
            parallel_task_dimension: "analysis",
            kernel_label: "ols.faer.temporal",
            provenance: ("estimate.bayesian_temporal_gcomp", "estimate.bayesian.temporal.gcomp"),
        },
        EstimatorId::TemporalGlmLinkContrast => EstimatorData {
            compatible: NONE,
            name: "temporal.glm_link_contrast",
            parallel_task_dimension: "analysis",
            kernel_label: "glm.faer.temporal.link_contrast",
            provenance: (
                "estimate.temporal_glm_link_contrast",
                "estimate.temporal.glm_link_contrast",
            ),
        },
        EstimatorId::TemporalLaggedInstrument => EstimatorData {
            compatible: NONE,
            name: "temporal.lagged_instrument",
            parallel_task_dimension: "analysis",
            kernel_label: "iv.wald.temporal",
            provenance: (
                "estimate.temporal_lagged_instrument",
                "estimate.temporal.lagged_instrument",
            ),
        },
        EstimatorId::BayesianTemporalLaggedInstrument => EstimatorData {
            compatible: NONE,
            name: "bayesian.temporal.lagged_instrument",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.iv.reduced_form_ratio.temporal",
            provenance: (
                "estimate.bayesian_temporal_lagged_instrument",
                "estimate.bayesian.temporal.lagged_instrument",
            ),
        },
        EstimatorId::BayesianTemporalGlmLinkContrast => EstimatorData {
            compatible: NONE,
            name: "bayesian.temporal.glm_link_contrast",
            parallel_task_dimension: "analysis",
            kernel_label: "laplace.temporal.link_contrast",
            provenance: (
                "estimate.bayesian_temporal_glm_link_contrast",
                "estimate.bayesian.temporal.glm_link_contrast",
            ),
        },
        EstimatorId::BayesianTemporalWhiteInnovation => EstimatorData {
            compatible: NONE,
            name: "bayesian.temporal.white_innovation_linear",
            parallel_task_dimension: "analysis",
            kernel_label: "ols.faer.temporal.white_innovation",
            provenance: (
                "estimate.bayesian_temporal_white_innovation",
                "estimate.bayesian.temporal.white_innovation_linear",
            ),
        },
        EstimatorId::FunctionalDistribution => EstimatorData {
            compatible: NONE,
            name: "functional.distribution",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "functional.distribution",
            provenance: same_step("estimate.functional_distribution"),
        },
        EstimatorId::StaticMediationLinear => EstimatorData {
            compatible: NONE,
            name: "mediation.linear",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer.mediation",
            provenance: same_step("estimate.mediation.linear"),
        },
        EstimatorId::FunctionalEffect => EstimatorData {
            compatible: GENERAL_ID,
            name: "functional.effect",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "functional.effect",
            provenance: same_step("estimate.functional_effect"),
        },
        EstimatorId::ConditionalLinearAdjustment => EstimatorData {
            compatible: ADJUSTMENT,
            name: "conditional.linear.adjustment",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer.conditional",
            provenance: ("estimate.conditional_linear", "estimate.conditional_linear_adjustment"),
        },
        EstimatorId::BayesianTemporalMediation => EstimatorData {
            compatible: NONE,
            name: "temporal.mediation.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.gaussian",
            provenance: (
                "estimate.bayesian_temporal_mediation",
                "estimate.bayesian_temporal_mediation",
            ),
        },
        EstimatorId::ResponseBayesian => EstimatorData {
            compatible: NONE,
            name: "response.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.gaussian",
            provenance: same_step("estimate.response.bayesian"),
        },
        EstimatorId::TemporalResponseBayesian => EstimatorData {
            compatible: NONE,
            name: "response.temporal.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.gaussian",
            provenance: (
                "estimate.response.temporal.bayesian",
                "estimate.response.temporal.bayesian",
            ),
        },
        EstimatorId::TemporalMediation => EstimatorData {
            compatible: NONE,
            name: "temporal.mediation",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer.temporal",
            provenance: same_step("estimate.temporal_mediation"),
        },
        EstimatorId::ResponseKennedyDr => EstimatorData {
            compatible: NONE,
            name: "response.kennedy_dr",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "response.kennedy_dr.gam",
            provenance: same_step("estimate.response.kennedy_dr"),
        },
        EstimatorId::ResponseRieszAde => EstimatorData {
            compatible: NONE,
            name: "response.riesz_ade",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "response.riesz_ade.gam",
            provenance: same_step("estimate.response.riesz_ade"),
        },
        EstimatorId::ResponseGamDerivative => EstimatorData {
            compatible: NONE,
            name: "response.gam_derivative",
            parallel_task_dimension: "outcome",
            kernel_label: "response.gam.derivative",
            provenance: same_step("estimate.response.gam_derivative"),
        },
        EstimatorId::ResponseInterventionGcomp => EstimatorData {
            compatible: NONE,
            name: "response.intervention_gcomp",
            parallel_task_dimension: "intervention",
            kernel_label: "response.gam.intervention_gcomp",
            provenance: (
                "estimate.response.intervention_gcomp",
                "estimate.response.intervention_gcomp",
            ),
        },
        EstimatorId::CellAipw => EstimatorData {
            compatible: NONE,
            name: "cell.aipw",
            parallel_task_dimension: "cell",
            kernel_label: "aipw.cell",
            provenance: same_step("estimate.cell.aipw"),
        },
        EstimatorId::TemporalResponseGcomp => EstimatorData {
            compatible: NONE,
            name: "temporal.response.gcomp",
            parallel_task_dimension: "horizon",
            kernel_label: "ols.faer.temporal.response",
            provenance: same_step("estimate.temporal_response.gcomp"),
        },
        EstimatorId::GcmFit => EstimatorData {
            compatible: NONE,
            name: "gcm.fit",
            parallel_task_dimension: "analysis",
            kernel_label: "gcm.aap",
            provenance: same_step("estimate.gcm.fit"),
        },
        EstimatorId::GcmFitBayesian => EstimatorData {
            compatible: NONE,
            name: "gcm.fit.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "gcm.attribution.bayesian",
            provenance: same_step("estimate.gcm.fit.bayesian"),
        },
        EstimatorId::GcmAttributionBayesian => EstimatorData {
            compatible: NONE,
            name: "gcm.attribution.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "gcm.attribution.bayesian",
            provenance: same_step("estimate.gcm.attribution.bayesian"),
        },
        EstimatorId::TransportTrialIpw => EstimatorData {
            compatible: NONE,
            name: "transport.trial_ipw",
            parallel_task_dimension: "analysis",
            kernel_label: "transport.trial_ipw",
            provenance: same_step("estimate.transport.trial_ipw"),
        },
        EstimatorId::TransportTrialBayesianBootstrap => EstimatorData {
            compatible: NONE,
            name: "transport.trial_bayesian_bootstrap",
            parallel_task_dimension: "analysis",
            kernel_label: "transport.trial_bayesian_bootstrap",
            provenance: (
                "estimate.transport.trial_bayesian_bootstrap",
                "estimate.transport.trial_bayesian_bootstrap",
            ),
        },
        EstimatorId::InterferenceHtHajek => EstimatorData {
            compatible: INTERFERENCE,
            name: "interference.ht_hajek",
            parallel_task_dimension: "analysis",
            kernel_label: "interference.ht_hajek",
            provenance: same_step("estimate.interference.ht_hajek"),
        },
        EstimatorId::InterferenceClusterNeyman => EstimatorData {
            compatible: INTERFERENCE,
            name: "interference.cluster_neyman",
            parallel_task_dimension: "analysis",
            kernel_label: "interference.cluster_neyman",
            provenance: (
                "estimate.interference.cluster_neyman",
                "estimate.interference.cluster_neyman",
            ),
        },
        EstimatorId::InterferenceSaturationExact => EstimatorData {
            compatible: INTERFERENCE,
            name: "interference.saturation_exact",
            parallel_task_dimension: "analysis",
            kernel_label: "interference.saturation_exact",
            provenance: (
                "estimate.interference.saturation_exact",
                "estimate.interference.saturation_exact",
            ),
        },
        EstimatorId::InterferenceObservationalIpw => EstimatorData {
            compatible: INTERFERENCE,
            name: "interference.observational_ipw",
            parallel_task_dimension: "analysis",
            kernel_label: "interference.observational_ipw",
            provenance: (
                "estimate.interference.observational_ipw",
                "estimate.interference.observational_ipw",
            ),
        },
        EstimatorId::InterferenceBayesianGaussian => EstimatorData {
            compatible: INTERFERENCE,
            name: "interference.bayesian_gaussian",
            parallel_task_dimension: "analysis",
            kernel_label: "interference.bayesian_gaussian",
            provenance: (
                "estimate.interference.bayesian_gaussian",
                "estimate.interference.bayesian_gaussian",
            ),
        },
        EstimatorId::InterferenceClusterBayesianBootstrap => EstimatorData {
            compatible: INTERFERENCE,
            name: "interference.cluster_bayesian_bootstrap",
            parallel_task_dimension: "analysis",
            kernel_label: "interference.cluster_bayesian_bootstrap",
            provenance: (
                "estimate.interference.cluster_bayesian_bootstrap",
                "estimate.interference.cluster_bayesian_bootstrap",
            ),
        },
        EstimatorId::BayesianComplierEffect => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.complier_effect",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.complier_effect",
            provenance: same_step("estimate.bayesian_complier_effect"),
        },
        EstimatorId::RandomizedHt => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.ht_itt",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.ht_itt",
            provenance: same_step("estimate.randomized.ht_itt"),
        },
        EstimatorId::ContinuousDoseKernel => EstimatorData {
            compatible: DOSE,
            name: "policy.triangular_kernel_inverse_density",
            parallel_task_dimension: "analysis",
            kernel_label: "policy.triangular_kernel_inverse_density",
            provenance: same_step("estimate.policy.continuous_dose"),
        },
        EstimatorId::RandomizedFixedCupedHt => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.fixed_cuped_ht_itt",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.fixed_cuped_ht_itt",
            provenance: (
                "estimate.randomized.fixed_cuped_ht_itt",
                "estimate.randomized.fixed_cuped_ht_itt",
            ),
        },
        EstimatorId::RandomizedAncova => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.ancova_itt",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.ancova_itt",
            provenance: same_step("estimate.randomized.ancova_itt"),
        },
        EstimatorId::RandomizedWaldCace => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.wald_cace_late",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.wald_cace_late",
            provenance: (
                "estimate.randomized.wald_cace_late",
                "estimate.randomized.wald_cace_late",
            ),
        },
        EstimatorId::RandomizedSwitchbackHt => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.switchback_ht_itt",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.switchback_ht_itt",
            provenance: (
                "estimate.randomized.switchback_ht_itt",
                "estimate.randomized.switchback_ht_itt",
            ),
        },
        EstimatorId::RandomizedNeyman => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.neyman_itt",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.neyman_itt",
            provenance: same_step("estimate.randomized.neyman_itt"),
        },
        EstimatorId::RandomizedDrPolicy => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.dr_policy",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.dr_policy",
            provenance: (
                "estimate.policy_value.doubly_robust",
                "estimate.policy_value.doubly_robust",
            ),
        },
        EstimatorId::RandomizedIpwPolicy => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.ipw_policy",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.ipw_policy",
            provenance: same_step("estimate.policy_value.ipw"),
        },
        EstimatorId::RandomizedMultiActionIpwPolicy => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.multi_action_ipw_policy",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.multi_action_ipw_policy",
            provenance: (
                "estimate.policy_value.multi_action_ipw",
                "estimate.policy_value.multi_action_ipw",
            ),
        },
        EstimatorId::RandomizedSurvivalProductLimit => EstimatorData {
            compatible: RANDOMIZED,
            name: "randomized.survival_product_limit",
            parallel_task_dimension: "analysis",
            kernel_label: "randomized.survival_product_limit",
            provenance: same_step("estimate.survival.product_limit"),
        },
        EstimatorId::LongitudinalIpwRegime => EstimatorData {
            compatible: RANDOMIZED,
            name: "longitudinal.ipw_regime",
            parallel_task_dimension: "subject",
            kernel_label: "longitudinal.ipw_regime",
            provenance: same_step("estimate.longitudinal.ipw_regime"),
        },
        EstimatorId::LongitudinalGFormulaRegime => EstimatorData {
            compatible: RANDOMIZED,
            name: "longitudinal.g_formula_regime",
            parallel_task_dimension: "subject",
            kernel_label: "longitudinal.g_formula_regime",
            provenance: (
                "estimate.longitudinal.g_formula_regime",
                "estimate.longitudinal.g_formula_regime",
            ),
        },
        EstimatorId::LongitudinalSequentialDrRegime => EstimatorData {
            compatible: RANDOMIZED,
            name: "longitudinal.sequential_dr_regime",
            parallel_task_dimension: "subject",
            kernel_label: "longitudinal.sequential_dr_regime",
            provenance: (
                "estimate.longitudinal.sequential_dr_regime",
                "estimate.longitudinal.sequential_dr_regime",
            ),
        },
        EstimatorId::LongitudinalMarginalStructuralModel => EstimatorData {
            compatible: RANDOMIZED,
            name: "longitudinal.marginal_structural_model",
            parallel_task_dimension: "subject",
            kernel_label: "longitudinal.marginal_structural_model",
            provenance: (
                "estimate.longitudinal.marginal_structural_model",
                "estimate.longitudinal.marginal_structural_model",
            ),
        },
        EstimatorId::Dml => EstimatorData {
            compatible: ADJUSTMENT,
            name: "dml",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "dml",
            provenance: same_step("estimate.dml"),
        },
        EstimatorId::DrLearner => EstimatorData {
            compatible: ADJUSTMENT,
            name: "dr.learner",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "dr.learner",
            provenance: same_step("estimate.dr.learner"),
        },
        EstimatorId::CausalForest => EstimatorData {
            compatible: ADJUSTMENT,
            name: "causal.forest",
            parallel_task_dimension: "forest.tree",
            kernel_label: "causal.forest",
            provenance: same_step("estimate.causal.forest"),
        },
        EstimatorId::BayesianPanelDid => EstimatorData {
            compatible: NONE,
            name: "bayesian.panel_did",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_did.normal_inverse_gamma",
            provenance: same_step("estimate.bayesian_panel_did"),
        },
        EstimatorId::BayesianSyntheticControl => EstimatorData {
            compatible: NONE,
            name: "bayesian.synthetic_control",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_synthetic_control.pairwise_metropolis",
            provenance: same_step("estimate.bayesian_synthetic_control"),
        },
        EstimatorId::BayesianBinaryPolicyValue => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.binary_policy_value",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.binary_policy_value",
            provenance: same_step("estimate.bayesian_binary_policy_value"),
        },
        EstimatorId::BayesianSequentialRegimeValue => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.sequential_regime_value",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.sequential_regime_value",
            provenance: same_step("estimate.bayesian_sequential_regime_value"),
        },
        EstimatorId::BayesianPiecewiseSurvival => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.piecewise_survival",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_survival.piecewise_exponential_gamma",
            provenance: same_step("estimate.bayesian_piecewise_survival"),
        },
        EstimatorId::BayesianRandomizedQuantile => EstimatorData {
            compatible: BACKDOOR_AND_AUTO,
            name: "bayesian.randomized_quantile",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_quantile.dirichlet_grid_inversion",
            provenance: same_step("estimate.bayesian_randomized_quantile"),
        },
        EstimatorId::BayesianDosePolicy => EstimatorData {
            compatible: DOSE,
            name: "bayesian.continuous_dose_policy",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_dose_policy.group_polynomial_conjugate",
            provenance: same_step("estimate.bayesian_continuous_dose_policy"),
        },
        EstimatorId::PeriodHistoryMatching => EstimatorData {
            compatible: RANDOMIZED,
            name: "matching.period_history_contrast",
            parallel_task_dimension: "analysis",
            kernel_label: "matching.period_history_contrast",
            provenance: same_step("estimate.period_history_matching"),
        },
        EstimatorId::BayesianPeriodHistoryOutcomeModel => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.period_history_outcome_model",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_gaussian.period_history_outcome_model",
            provenance: same_step("estimate.bayesian_period_history_outcome_model"),
        },
        EstimatorId::TemporalDesignExcursion => EstimatorData {
            compatible: RANDOMIZED,
            name: "temporal_design.excursion",
            parallel_task_dimension: "analysis",
            kernel_label: "temporal_design.cluster_ratio_ht",
            provenance: same_step("estimate.temporal_design_excursion"),
        },
        EstimatorId::BayesianTemporalDesignExcursion => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.temporal_design.excursion",
            parallel_task_dimension: "analysis",
            kernel_label: "temporal_design.cluster_bayesian_bootstrap",
            provenance: same_step("estimate.bayesian_temporal_design_excursion"),
        },
        EstimatorId::TemporalRegimeSurvival => EstimatorData {
            compatible: RANDOMIZED,
            name: "temporal_design.regime_survival",
            parallel_task_dimension: "analysis",
            kernel_label: "temporal_design.ipw_hazard_subject_bootstrap",
            provenance: same_step("estimate.temporal_regime_survival"),
        },
        EstimatorId::BayesianTemporalRegimeSurvival => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.temporal_design.regime_survival",
            parallel_task_dimension: "analysis",
            kernel_label: "temporal_design.joint_likelihood_g_formula",
            provenance: same_step("estimate.bayesian_temporal_regime_survival"),
        },
        EstimatorId::IvBoundsSharp => EstimatorData {
            compatible: RANDOMIZED,
            name: "iv_bounds.balke_pearl",
            parallel_task_dimension: "analysis",
            kernel_label: "iv_bounds.balke_pearl_row_bootstrap",
            provenance: same_step("estimate.iv_bounds_sharp"),
        },
        EstimatorId::BayesianIvBounds => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.iv_bounds.endpoint_bootstrap",
            parallel_task_dimension: "analysis",
            kernel_label: "iv_bounds.bayesian_bootstrap_endpoints",
            provenance: same_step("estimate.bayesian_iv_bounds"),
        },
        EstimatorId::SequentialCrossfitContrastWald => EstimatorData {
            compatible: RANDOMIZED,
            name: "sequential.crossfit_contrast.wald",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "sequential.crossfit_contrast",
            provenance: same_step("estimate.sequential_crossfit_contrast_wald"),
        },
        EstimatorId::BayesianSequentialCrossfitContrast => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.sequential_crossfit_contrast",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "sequential.crossfit_contrast",
            provenance: same_step("estimate.bayesian_sequential_crossfit_contrast"),
        },
        EstimatorId::BayesianLocalRatio => EstimatorData {
            compatible: NONE,
            name: "bayesian.local_ratio",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian_local_ratio.joint_local_linear",
            provenance: same_step("estimate.bayesian_local_ratio"),
        },
        EstimatorId::BayesianRandomizedFinitePopulation => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.randomized_finite_population",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.randomized_arm_contrast",
            provenance: same_step("estimate.bayesian_randomized_arm_contrast"),
        },
        EstimatorId::BayesianSwitchback => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.switchback_effect",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.switchback_effect",
            provenance: same_step("estimate.bayesian_switchback_effect"),
        },
        EstimatorId::BayesianRandomizedSuperpopulation => EstimatorData {
            compatible: RANDOMIZED,
            name: "bayesian.randomized_superpopulation",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.randomized_arm_contrast",
            provenance: same_step("estimate.bayesian_randomized_arm_contrast"),
        },
    }
}

impl EstimatorId {
    /// Canonical wire id.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        estimator_data(*self).name
    }

    /// Parallel-task dimension label for physical planning.
    #[must_use]
    pub const fn parallel_task_dimension(&self) -> &'static str {
        estimator_data(*self).parallel_task_dimension
    }

    /// Dense-kernel label recorded on the physical plan.
    #[must_use]
    pub const fn kernel_label(&self) -> &'static str {
        estimator_data(*self).kernel_label
    }
}

impl FromStr for EstimatorId {
    type Err = UnknownStrategy;

    /// Parse a wire / builder id string.
    ///
    /// # Errors
    ///
    /// [`UnknownStrategy`] when `s` does not match any closed-set estimator name.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_closed(s, estimator_index(), "estimator", estimator_names())
    }
}

impl std::fmt::Display for EstimatorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when a strategy name does not match any known strategy.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("unknown {kind} `{got}`; expected one of: {}", .expected.join(", "))]
pub struct UnknownStrategy {
    /// Which strategy family failed to parse (`"identifier"` or `"estimator"`).
    pub kind: &'static str,
    /// The name that failed to parse.
    pub got: String,
    /// Every accepted name, for the error message.
    pub expected: &'static [&'static str],
}

impl From<UnknownStrategy> for CausalError {
    fn from(e: UnknownStrategy) -> Self {
        crate::compile_reason!("unknown_strategy", "{e}")
    }
}

/// Default identifier as a closed enum.
pub const DEFAULT_IDENTIFIER_ID: IdentifierId = IdentifierId::BackdoorAdjustment;

/// Default estimator as a closed enum.
pub const DEFAULT_ESTIMATOR_ID: EstimatorId = EstimatorId::LinearAdjustmentAte;

/// Default distribution identifier enum.
pub const DEFAULT_DISTRIBUTION_IDENTIFIER_ID: IdentifierId = IdentifierId::GeneralId;
/// Default distribution estimator enum.
pub const DEFAULT_DISTRIBUTION_ESTIMATOR_ID: EstimatorId = EstimatorId::FunctionalDistribution;

/// Default response identifier enum.
pub const DEFAULT_RESPONSE_IDENTIFIER_ID: IdentifierId = IdentifierId::ResponseBackdoor;
/// Default response-curve estimator enum.
pub const DEFAULT_RESPONSE_ESTIMATOR_ID: EstimatorId = EstimatorId::ResponseKennedyDr;

/// Compile-time allowlist for continuous-response identify/estimate pairs.
///
/// # Errors
///
/// Incompatible identifier/estimator pair.
pub fn validate_response_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    if identifier != IdentifierId::ResponseBackdoor
        || !matches!(
            estimator,
            EstimatorId::ResponseKennedyDr
                | EstimatorId::ResponseRieszAde
                | EstimatorId::ResponseGamDerivative
                | EstimatorId::ResponseInterventionGcomp
                | EstimatorId::CellAipw
                | EstimatorId::ResponseBayesian
        )
    {
        return Err(CausalError::Compile {
            message: format!(
                "Response requires identifier response.backdoor and a response.* estimator (got {:?} / {:?})",
                identifier.as_str(),
                estimator.as_str()
            ),
        });
    }
    Ok(())
}

/// Compile-time allowlist for class-aware Cpdag/Pag response (same theory as ATE).
///
/// # Errors
///
/// Incompatible identifier/estimator pair.
pub fn validate_class_response_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    if identifier != IdentifierId::GeneralizedAdjustment
        || !matches!(
            estimator,
            EstimatorId::ResponseKennedyDr
                | EstimatorId::ResponseRieszAde
                | EstimatorId::ResponseGamDerivative
                | EstimatorId::ResponseInterventionGcomp
                | EstimatorId::CellAipw
                | EstimatorId::ResponseBayesian
        )
    {
        return Err(CausalError::Compile {
            message: format!(
                "Cpdag/Pag/Admg response requires identifier generalized.adjustment and a \
                 response.* estimator (kennedy_dr, riesz_ade, gam_derivative, \
                 intervention_gcomp or bayesian) (got {:?} / {:?})",
                identifier.as_str(),
                estimator.as_str()
            ),
        });
    }
    Ok(())
}

/// Compile-time allowlist of identifier/estimator pairs for the static ATE path.
///
/// The pairs live on each estimator's compatible-identifier list. `BayesianRobustAte`
/// is not an adjustment estimator: it is sealed only for an explicit retained DAG, so
/// `generalized.adjustment` is refused while backdoor and `auto` accept it.
///
/// # Errors
///
/// Unknown ids or incompatible pairs.
pub fn validate_static_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    let supported = estimator_data(estimator).compatible.contains(&identifier);
    if !supported {
        return Err(crate::compile_reason!(
            "strategy_incompatible",
            "identifier {:?} is not compatible with estimator {:?}",
            identifier.as_str(),
            estimator.as_str()
        ));
    }
    Ok(())
}

/// Default path-specific identifier enum.
pub const DEFAULT_PATH_IDENTIFIER_ID: IdentifierId = IdentifierId::PathSpecificNatural;
/// Default path-specific estimator enum.
pub const DEFAULT_PATH_ESTIMATOR_ID: EstimatorId = EstimatorId::FunctionalEffect;

/// Default PAG identifier enum.
pub const DEFAULT_PAG_IDENTIFIER_ID: IdentifierId = IdentifierId::GeneralizedAdjustment;
/// Default PAG estimator enum.
pub const DEFAULT_PAG_ESTIMATOR_ID: EstimatorId = EstimatorId::LinearAdjustmentAte;

/// Default ADMG identifier enum.
pub const DEFAULT_ADMG_IDENTIFIER_ID: IdentifierId = IdentifierId::GeneralId;
/// Default ADMG estimator enum.
pub const DEFAULT_ADMG_ESTIMATOR_ID: EstimatorId = EstimatorId::FunctionalEffect;

/// Default conditional identifier enum.
pub const DEFAULT_CONDITIONAL_IDENTIFIER_ID: IdentifierId = IdentifierId::BackdoorAdjustment;
/// Default conditional estimator enum.
pub const DEFAULT_CONDITIONAL_ESTIMATOR_ID: EstimatorId = EstimatorId::ConditionalLinearAdjustment;

/// Whether an identification status is acceptable when estimands are present.
#[must_use]
pub fn identification_status_acceptable(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::GraphDependent
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
    )
}

/// Gate identification: reject `NotIdentified` and empty estimands.
///
/// # Errors
///
/// Effect not identified or no estimand returned.
pub fn require_identified(result: &IdentificationResult) -> Result<(), CausalError> {
    if matches!(result.status, IdentificationStatus::NotIdentified)
        || result.estimands.is_empty()
        || !identification_status_acceptable(result.status)
    {
        return Err(CausalError::not_identified(
            result.status,
            antecedent_identify::search_truncated(result),
            "effect not identified",
        ));
    }
    Ok(())
}

/// Whether an estimand method is compatible with an estimator.
#[must_use]
#[allow(clippy::too_many_lines, reason = "one exhaustive match over every estimator id")]
pub fn estimand_compatible_with_estimator(method: EstimandMethod, estimator: &EstimatorId) -> bool {
    match estimator {
        EstimatorId::LinearAdjustmentAte
        | EstimatorId::PropensityWeighting
        | EstimatorId::PropensityMatching
        | EstimatorId::PropensityStratification
        | EstimatorId::DistanceMatching
        | EstimatorId::Aipw
        | EstimatorId::GlmAdjustment
        | EstimatorId::BayesianGcomp
        | EstimatorId::BayesianBasisGcomp
        | EstimatorId::BayesianConditional
        | EstimatorId::ConditionalLinearAdjustment
        | EstimatorId::ResponseKennedyDr
        | EstimatorId::ResponseRieszAde
        | EstimatorId::ResponseGamDerivative
        | EstimatorId::ResponseInterventionGcomp
        | EstimatorId::CellAipw
        | EstimatorId::Dml
        | EstimatorId::DrLearner
        | EstimatorId::CausalForest
        | EstimatorId::ResponseBayesian
        | EstimatorId::BayesianRandomizedQuantile
        | EstimatorId::BayesianRobustAte => method.is_backdoor_family(),
        EstimatorId::FrontDoorTwoStage | EstimatorId::FrontDoorFunctional => {
            matches!(method, EstimandMethod::FrontDoor)
        }
        EstimatorId::IvWald | EstimatorId::Iv2Sls | EstimatorId::BayesianIvJointLinear => {
            matches!(method, EstimandMethod::Iv)
        }
        EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear => {
            matches!(method, EstimandMethod::RdSharp)
        }
        EstimatorId::TemporalLinearAdjustment
        | EstimatorId::BayesianTemporalGcomp
        | EstimatorId::BayesianTemporalWhiteInnovation
        | EstimatorId::TemporalGlmLinkContrast
        | EstimatorId::BayesianTemporalGlmLinkContrast
        | EstimatorId::TemporalSequentialGcomp => {
            matches!(method, EstimandMethod::TemporalBackdoorUnfolded)
        }
        EstimatorId::TemporalLaggedInstrument | EstimatorId::BayesianTemporalLaggedInstrument => {
            matches!(method, EstimandMethod::TemporalIvUnfolded)
        }
        EstimatorId::TemporalResponseGcomp | EstimatorId::TemporalResponseBayesian => {
            matches!(
                method,
                EstimandMethod::TemporalBackdoorUnfolded | EstimandMethod::BackdoorAdjustment
            )
        }
        EstimatorId::TemporalMediation | EstimatorId::BayesianTemporalMediation => {
            method.is_temporal_mediation() || matches!(method, EstimandMethod::FrontDoor)
        }
        EstimatorId::FunctionalDistribution => matches!(method, EstimandMethod::GeneralId),
        EstimatorId::StaticMediationLinear => matches!(method, EstimandMethod::PathSpecificNatural),
        EstimatorId::FunctionalEffect => {
            matches!(method, EstimandMethod::PathSpecificNatural | EstimandMethod::GeneralId)
        }
        EstimatorId::GcmFit
        | EstimatorId::GcmFitBayesian
        | EstimatorId::GcmAttributionBayesian
        | EstimatorId::TransportTrialIpw
        | EstimatorId::TransportTrialBayesianBootstrap
        | EstimatorId::InterferenceHtHajek
        | EstimatorId::InterferenceClusterNeyman
        | EstimatorId::InterferenceSaturationExact
        | EstimatorId::InterferenceObservationalIpw
        | EstimatorId::InterferenceBayesianGaussian
        | EstimatorId::InterferenceClusterBayesianBootstrap
        | EstimatorId::ContinuousDoseKernel
        | EstimatorId::RandomizedSurvivalProductLimit
        | EstimatorId::LongitudinalIpwRegime
        | EstimatorId::LongitudinalGFormulaRegime
        | EstimatorId::LongitudinalSequentialDrRegime
        | EstimatorId::LongitudinalMarginalStructuralModel
        | EstimatorId::BayesianPanelDid
        | EstimatorId::BayesianSyntheticControl
        | EstimatorId::BayesianSequentialRegimeValue
        | EstimatorId::SequentialCrossfitContrastWald
        | EstimatorId::BayesianSequentialCrossfitContrast
        | EstimatorId::PeriodHistoryMatching
        | EstimatorId::BayesianPeriodHistoryOutcomeModel
        | EstimatorId::TemporalDesignExcursion
        | EstimatorId::BayesianTemporalDesignExcursion
        | EstimatorId::TemporalRegimeSurvival
        | EstimatorId::BayesianTemporalRegimeSurvival
        | EstimatorId::IvBoundsSharp
        | EstimatorId::BayesianIvBounds
        | EstimatorId::BayesianPiecewiseSurvival
        | EstimatorId::BayesianDosePolicy
        | EstimatorId::BayesianLocalRatio => false,
        EstimatorId::RandomizedHt
        | EstimatorId::RandomizedFixedCupedHt
        | EstimatorId::RandomizedAncova
        | EstimatorId::RandomizedWaldCace
        | EstimatorId::RandomizedSwitchbackHt
        | EstimatorId::BayesianRandomizedFinitePopulation
        | EstimatorId::BayesianRandomizedSuperpopulation
        | EstimatorId::BayesianComplierEffect
        | EstimatorId::BayesianSwitchback
        | EstimatorId::RandomizedNeyman => {
            matches!(method, EstimandMethod::RandomizedItt)
        }
        EstimatorId::RandomizedDrPolicy
        | EstimatorId::BayesianBinaryPolicyValue
        | EstimatorId::RandomizedIpwPolicy
        | EstimatorId::RandomizedMultiActionIpwPolicy => {
            matches!(method, EstimandMethod::RandomizedPolicyValue)
        }
    }
}

/// Select a single estimand matching the estimator (no silent Auto `.first()`).
///
/// # Errors
///
/// No estimand, or multiple estimands without a unique estimator-compatible match.
pub fn select_estimand(
    identification: &IdentificationResult,
    estimator: EstimatorId,
) -> Result<IdentifiedEstimand, CausalError> {
    select_estimand_index(identification, estimator).map(|i| identification.estimands[i].clone())
}

/// Select the estimand as [`select_estimand`] does and narrow the identification to it.
///
/// A multi-strategy identification lists alternatives whose status and assumptions
/// differ. The returned identification carries exactly the selected estimand's claim, so
/// what reaches the estimate and the result is the claim of the strategy actually used.
///
/// # Errors
///
/// No estimand, or multiple estimands without a unique estimator-compatible match.
pub fn select_claim(
    identification: IdentificationResult,
    estimator: EstimatorId,
) -> Result<(IdentificationResult, IdentifiedEstimand), CausalError> {
    let index = select_estimand_index(&identification, estimator)?;
    let estimand = identification.estimands[index].clone();
    let mut claim = if identification.estimand_claims.is_empty() {
        identification
    } else {
        identification.narrowed_to(index).unwrap_or(identification)
    };
    if let Some(restriction) = estimator.claim_restriction() {
        restrict_claim(&mut claim, &restriction);
    }
    Ok((claim, estimand))
}

impl EstimatorId {
    /// Restriction under which this estimator's target equals the identified functional, when
    /// it needs one.
    ///
    /// An estimator that evaluates the identified functional keeps the identifier's claim; how
    /// it models a regression inside that functional is an estimation-scope record, because the
    /// regression is a feature of the observed law that a richer model can fit without changing
    /// the target. An estimator that computes a *different* functional of the observed law is
    /// the queried effect only under a restriction on the structural model, and that
    /// restriction is part of the identification claim: the constant-effect (or monotonicity)
    /// restriction of a Wald ratio, recorded by the IV identifier, the restriction under
    /// which a product of regression coefficients is the front-door effect, and the restriction
    /// under which `total − direct` is the pure natural indirect effect the path-specific
    /// identifier certifies, recorded here because those identifiers cannot know which
    /// estimator will run.
    #[must_use]
    pub fn claim_restriction(self) -> Option<AssumptionRecord> {
        match self {
            Self::FrontDoorTwoStage => Some(antecedent_estimate::linear_path_product_restriction()),
            Self::StaticMediationLinear => {
                Some(antecedent_estimate::linear_no_interaction_restriction())
            }
            _ => None,
        }
    }
}

/// Add `restriction` to a claim and weaken a nonparametric status to a parametric one.
/// Statuses that are already weaker are kept.
fn restrict_claim(claim: &mut IdentificationResult, restriction: &AssumptionRecord) {
    let weaken = |status: &mut IdentificationStatus| {
        if *status == IdentificationStatus::NonparametricallyIdentified {
            *status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
        }
    };
    weaken(&mut claim.status);
    claim.required_assumptions.extend_unique([restriction]);
    for own in &mut claim.estimand_claims {
        weaken(&mut own.status);
        own.required_assumptions.extend_unique([restriction]);
    }
}

fn select_estimand_index(
    identification: &IdentificationResult,
    estimator: EstimatorId,
) -> Result<usize, CausalError> {
    let estimands = &identification.estimands;
    if estimands.is_empty() {
        return Err(CausalError::Compile { message: "no estimand returned".into() });
    }
    if estimands.len() == 1 {
        return Ok(0);
    }
    let matches: Vec<usize> = estimands
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            if e.is_adjustment_shaped() {
                return estimand_compatible_with_estimator(
                    EstimandMethod::BackdoorAdjustment,
                    &estimator,
                );
            }
            e.method_kind()
                .map(|m| estimand_compatible_with_estimator(m, &estimator))
                .unwrap_or(false)
        })
        .map(|(i, _)| i)
        .collect();
    if matches.len() == 1 {
        return Ok(matches[0]);
    }
    // Auto (and the response identifiers) can list a criterion estimand (backdoor adjustment,
    // front-door, path-specific, ...) beside the general-ID functional for the same query: ID
    // is the general fallback, so when exactly one estimator-compatible match is criterion-
    // shaped and the rest are general.id, the criterion match is the more specific claim and
    // wins. This does not fire when every match is general.id (for example a MeanCurve's one
    // general-ID estimand per grid level): that is a genuine ambiguity between distinct
    // functionals, not a criterion-vs-fallback tiebreak, and still errors below.
    let non_general: Vec<usize> = matches
        .iter()
        .copied()
        .filter(|&i| estimands[i].method_kind() != Ok(EstimandMethod::GeneralId))
        .collect();
    if non_general.len() == 1 {
        return Ok(non_general[0]);
    }
    // Auto lists every strategy that identified the query (criterion estimands and the general
    // ID functional); name them so a caller sees which methods the estimator matched.
    let methods: Vec<&str> = estimands.iter().map(|e| e.method.as_ref()).collect();
    Err(CausalError::Compile {
        message: format!(
            "identifier returned {} estimands ({}); select an explicit identifier or an estimator \
             that uniquely matches one method (got estimator {:?}, matching {})",
            estimands.len(),
            methods.join(", "),
            estimator.as_str(),
            matches.len()
        ),
    })
}

/// Allowlist for interventional-distribution identify+estimate.
///
/// # Errors
///
/// Incompatible identifier/estimator pair.
pub fn validate_distribution_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    let supported = matches!(
        (&identifier, &estimator),
        (IdentifierId::GeneralId | IdentifierId::Auto, EstimatorId::FunctionalDistribution)
    );
    if !supported {
        return Err(CausalError::Compile {
            message: format!(
                "Distribution requires identifier general.id|auto with estimator \
                 functional.distribution (got {:?} / {:?})",
                identifier.as_str(),
                estimator.as_str()
            ),
        });
    }
    Ok(())
}

/// Allowlist for path-specific natural-effect identify+estimate.
///
/// # Errors
///
/// Incompatible identifier/estimator pair.
pub fn validate_path_specific_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    let supported = matches!(
        (&identifier, &estimator),
        (IdentifierId::PathSpecificNatural | IdentifierId::Auto, EstimatorId::FunctionalEffect)
    );
    if !supported {
        return Err(CausalError::Compile {
            message: format!(
                "PathSpecific requires identifier path_specific.natural|auto with estimator \
                 functional.effect (got {:?} / {:?})",
                identifier.as_str(),
                estimator.as_str()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod names {
    use super::{EstimatorId, IdentifierId};

    #[test]
    fn identifier_from_str_roundtrips_all() {
        for id in IdentifierId::ALL {
            assert_eq!(id.as_str().parse::<IdentifierId>().unwrap(), *id);
        }
        assert!("not.an.identifier".parse::<IdentifierId>().is_err());
    }

    #[test]
    fn estimator_from_str_roundtrips_all() {
        for id in EstimatorId::ALL {
            assert_eq!(id.as_str().parse::<EstimatorId>().unwrap(), *id);
        }
        assert!("not.an.estimator".parse::<EstimatorId>().is_err());
    }

    /// `ALL` is generated from the enum's own variant list, so every variant
    /// is present exactly once; distinct wire names keep `FromStr` injective.
    #[test]
    fn catalogs_have_distinct_wire_names() {
        let identifiers: std::collections::HashSet<_> =
            IdentifierId::ALL.iter().map(IdentifierId::as_str).collect();
        assert_eq!(identifiers.len(), IdentifierId::ALL.len());
        let estimators: std::collections::HashSet<_> =
            EstimatorId::ALL.iter().map(EstimatorId::as_str).collect();
        assert_eq!(estimators.len(), EstimatorId::ALL.len());
    }
}
