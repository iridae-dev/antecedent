//! Prior and inference binding identity wire forms.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Deserialize, IdentityDomain, IoError, PayloadDigestWire, SemanticDigest, Serialize, digest_wire,
};

/// Coefficient / residual prior contents.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PriorSetIdentityWire {
    /// Prior specifications in declaration order.
    pub specs: Vec<PriorSpecIdentityWire>,
    /// Contrast coding for categorical predictors, when declared.
    pub contrast: Option<String>,
    /// Categorical predictor variable ids.
    pub categorical: Vec<u32>,
    /// Prior-restriction assumption ids.
    pub restrictions: Vec<String>,
}

/// One prior specification.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PriorSpecIdentityWire {
    /// Gaussian coefficient prior: mean and variance vectors.
    GaussianCoefficients {
        /// Coefficient means.
        mean: PayloadDigestWire,
        /// Coefficient variances.
        variance: PayloadDigestWire,
    },
    /// Inverse-gamma residual variance prior.
    ResidualInvGamma {
        /// Shape bits.
        shape_bits: u64,
        /// Scale bits.
        scale_bits: u64,
    },
    /// Known residual variance bits.
    KnownResidualVariance(u64),
    /// Correlation of a dense coefficient prior scale `V0` (row-major).
    CoefficientCorrelation {
        /// Dimension `p`.
        dim: u64,
        /// Row-major `p × p` correlation entries.
        matrix: PayloadDigestWire,
    },
}

/// Transferred-prior mapping.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PriorMappingIdentityWire {
    /// Identical coefficient subspace.
    IdenticalCoefficientSubspace,
    /// Effect functional bridge.
    EffectFunctional {
        /// Source posterior quantity name.
        source_quantity: String,
    },
    /// Named parameter pairs `(source, target)`, a sorted and deduplicated set.
    NamedParameters {
        /// Source → target parameter pairs.
        pairs: Vec<(String, String)>,
    },
}

impl PriorMappingIdentityWire {
    /// Canonical mapping (named pairs sorted and deduplicated).
    #[must_use]
    pub fn from_mapping(mapping: &crate::PriorMapping) -> Self {
        match mapping {
            crate::PriorMapping::IdenticalCoefficientSubspace => Self::IdenticalCoefficientSubspace,
            crate::PriorMapping::EffectFunctional { source_quantity } => {
                Self::EffectFunctional { source_quantity: source_quantity.to_string() }
            }
            crate::PriorMapping::NamedParameters { pairs } => {
                let mut pairs: Vec<(String, String)> = pairs
                    .iter()
                    .map(|(source, target)| (source.to_string(), target.to_string()))
                    .collect();
                pairs.sort();
                pairs.dedup();
                Self::NamedParameters { pairs }
            }
        }
    }
}

/// One external prior-bank source retained for conflict re-evaluation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalPriorSourceIdentityWire {
    /// Caller-stable source id.
    pub id: String,
    /// Mapped source prior.
    pub prior: PriorSetIdentityWire,
    /// Power-prior alpha bits.
    pub alpha_bits: u64,
    /// Mixture weight bits, when a mixture component.
    pub mixture_weight_bits: Option<u64>,
    /// Declared prior-strength ESS bits.
    pub ess_bits: Option<u64>,
}

/// External prior-bank composition inputs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalComposeIdentityWire {
    /// Sources in composition order.
    pub sources: Vec<ExternalPriorSourceIdentityWire>,
    /// Conflict policy `(p_min bits, kl_scale bits)`, when re-shrinking.
    pub conflict_policy: Option<(u64, u64)>,
    /// Original composition baseline, when retained for future conflict re-shrink.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<PriorSetIdentityWire>,
}

/// Bayesian configuration bound into the inference layer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BayesianBindingWire {
    /// Backend (`conjugate_gaussian`, `laplace`, `hmc`).
    pub backend: String,
    /// Likelihood / link.
    pub likelihood: String,
    /// Posterior draws requested.
    pub n_draws: u64,
    /// Isotropic prior scale bits, when no explicit prior or artifact is set.
    pub prior_scale_bits: Option<u64>,
    /// Explicit prior contents.
    pub prior: Option<PriorSetIdentityWire>,
    /// Posterior-artifact bytes used for a deferred mapped hydrate.
    pub prior_artifact: Option<PayloadDigestWire>,
    /// Concrete source snapshot recorded by a checked prior-source producer.
    /// This records ownership; distinct full snapshots do not prove absence of partial overlap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_source_snapshot: Option<String>,
    /// Mapping for the prior artifact.
    pub prior_mapping: Option<PriorMappingIdentityWire>,
    /// External prior-bank composition.
    pub external_compose: Option<ExternalComposeIdentityWire>,
    /// Numerical draw controls requested on the posterior (`measured`); absent when none
    /// were, so a configuration without them keeps its identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draw_controls: Option<String>,
    /// Canonical text of the declared randomized-design model (hazard partition and prior, or
    /// dose curve and prior); absent when none was declared, so a configuration without one
    /// keeps its identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub design_model: Option<String>,
    /// Declared white-innovation premise and its Ljung-Box gate, as
    /// `ljung_box:max_lag=<n>:level_bits=<hex>`; absent when none was declared, so a
    /// configuration without it keeps its identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub white_innovation: Option<String>,
    /// Declared bounds of the retained posterior predictive replicates, as
    /// `n_sims=<n>:max_rows=<n>`; absent when none were requested, so a configuration
    /// without them keeps its identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predictive_replicates: Option<String>,
}

/// Discovery / estimation window split.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SplitIdentityWire {
    /// Discovery window `[start, end)`.
    pub discovery: (u64, u64),
    /// Estimation window `[start, end)`.
    pub estimation: (u64, u64),
    /// Gap between the windows.
    pub gap: u64,
    /// Series length the split was validated against.
    pub series_len: u64,
}

/// Observation-mechanism estimator options.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservationOptionsWire {
    /// Selected-outcome correction (`ipw` / `aipw`).
    pub selected_correction: String,
    /// Observation-probability floor bits.
    pub observation_probability_floor_bits: u64,
    /// Censoring-survival floor bits.
    pub censoring_survival_floor_bits: u64,
    /// Cross-fitting folds.
    pub crossfit_folds: u64,
}

/// Inference-binding payload: numeric knobs, prior contents, and resampling.
///
/// Independent of structure: a different graph with the same numeric
/// configuration has the same inference binding and a different program.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceBindingWire {
    /// Identity format.
    pub format: u16,
    /// `frequentist` or `bayesian`; must equal the program commitments.
    pub inference: String,
    /// Bootstrap replicates (frequentist).
    pub bootstrap_replicates: u32,
    /// Bayesian backend, likelihood, draws, and prior contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bayesian: Option<BayesianBindingWire>,
    /// Validation suite id; must equal the program commitments.
    pub validation_suite: Option<String>,
    /// Overlap policy tag.
    pub overlap_policy: Option<String>,
    /// Typed estimator configuration, when the caller supplied one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator_spec: Option<EstimatorSpecWire>,
    /// Response options, when a response surface is configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_options: Option<ResponseOptionsWire>,
    /// Observation-mechanism estimator options.
    pub observation_options: ObservationOptionsWire,
    /// Discovery / estimation split, when declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<SplitIdentityWire>,
    /// The temporal response retains the joint replicates of its surface (frequentist block
    /// bootstrap counts, or Bayesian block-weight draws), so horizons can be composed from
    /// realizations coupled across them. Absent when not requested.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub joint_horizons: bool,
}

/// Overlap policy of a configured estimator.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OverlapPolicyWire {
    /// Caller overrides overlap diagnostics.
    ExplicitOverride,
    /// Diagnostics required, with optional clip / trim bits.
    RequireDiagnostics {
        /// Propensity clip bits.
        clip_bits: Option<u64>,
        /// Propensity trim bits.
        trim_bits: Option<u64>,
    },
}

/// GLM fitting options of a configured estimator.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlmOptionsWire {
    /// Maximum IRLS iterations.
    pub max_iter: u32,
    /// Convergence tolerance bits.
    pub tol_bits: u64,
    /// Negative-binomial dispersion policy.
    pub nb_alpha: String,
    /// Ridge applied on separation, as bits.
    pub ridge_on_separation_bits: Option<u64>,
}

/// Scalar configuration of one configured estimator.
///
/// Data-sized fields (cluster ids, multiway ids, panel times, registry
/// weights) are [`PayloadDigestWire`]s computed once when the study is built.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EstimatorConfigWire {
    /// Linear-algebra backend.
    pub backend: String,
    /// Bootstrap replicates.
    pub bootstrap_replicates: u32,
    /// Overlap policy.
    pub overlap: OverlapPolicyWire,
    /// GLM options, when the estimator fits a GLM.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glm: Option<GlmOptionsWire>,
    /// Analytic SE kind, when configurable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub se_kind: Option<String>,
    /// Caliper bits, when matching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caliper_bits: Option<u64>,
    /// Caliper scale, when matching on the propensity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caliper_scale: Option<String>,
    /// Propensity strata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_strata: Option<u32>,
    /// GLM outcome family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Linear fit kind and penalty bits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_kind: Option<(String, Option<u64>)>,
    /// Cluster ids, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_ids: Option<PayloadDigestWire>,
    /// Multiway cluster ids, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiway_ids: Option<PayloadDigestWire>,
    /// Panel times, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel_times: Option<PayloadDigestWire>,
    /// Population registry contents, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_registry: Option<PayloadDigestWire>,
    /// Cross-fit folds (`dml` / `dr.learner`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub folds: Option<u32>,
    /// Outcome nuisance spec name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// Treatment nuisance spec name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment: Option<String>,
    /// DML score name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<String>,
    /// `dr.learner` final-stage spec name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_learner: Option<String>,
    /// Full outcome learner configuration, including hyperparameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_config: Option<String>,
    /// Full treatment learner configuration, including hyperparameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment_config: Option<String>,
    /// Full final-stage learner configuration, including hyperparameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_learner_config: Option<String>,
    /// Causal-forest tree count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_trees: Option<u32>,
    /// Causal-forest minimum leaf size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_leaf: Option<u32>,
    /// Causal-forest maximum depth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<u32>,
    /// Causal-forest honesty flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub honesty: Option<bool>,
    /// Declared exclusion of treatment rows at neither declared level; absent when the
    /// estimator refuses such rows (the default), so existing digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other_levels: Option<String>,
    /// Digest of the declared sampling weights of every source row; absent when none are
    /// declared, so existing digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_weights: Option<PayloadDigestWire>,
    /// Canonical key of a requested bootstrap interval (level, wave size, minimum replicates
    /// and Monte Carlo tolerances); absent when none is requested, so existing digests are
    /// unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_interval: Option<String>,
}

/// Portable estimator-spec identity (mirrors `antecedent::EstimatorSpec` variants).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EstimatorSpecWire {
    /// Id only.
    Default(String),
    /// Caller-configured OLS / linear adjustment.
    LinearAdjustmentAte(EstimatorConfigWire),
    /// Caller-configured inverse-probability weighting.
    PropensityWeighting(EstimatorConfigWire),
    /// Caller-configured propensity-score matching.
    PropensityMatching(EstimatorConfigWire),
    /// Caller-configured propensity stratification.
    PropensityStratification(EstimatorConfigWire),
    /// Caller-configured covariate distance matching.
    DistanceMatching(EstimatorConfigWire),
    /// Caller-configured augmented IPW.
    Aipw(EstimatorConfigWire),
    /// Caller-configured GLM adjustment.
    GlmAdjustment(EstimatorConfigWire),
    /// Caller-configured front-door two-stage.
    FrontDoorTwoStage(EstimatorConfigWire),
    /// Caller-configured Wald IV.
    IvWald(EstimatorConfigWire),
    /// Caller-configured two-stage least squares.
    Iv2Sls(EstimatorConfigWire),
    /// Caller-configured DML / AIPW.
    Dml(EstimatorConfigWire),
    /// Caller-configured DR-Learner.
    DrLearner(EstimatorConfigWire),
    /// Caller-configured causal forest.
    CausalForest(EstimatorConfigWire),
}

/// Portable response-surface options (every field that changes the estimate).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseOptionsWire {
    /// Cross-fitting folds.
    pub folds: u64,
    /// Nuisance spline basis count.
    pub nuisance_basis: u64,
    /// Nuisance roughness penalty bits.
    pub nuisance_lambda_bits: u64,
    /// Kernel bandwidth bits, when set (normal reference otherwise).
    pub bandwidth_bits: Option<u64>,
    /// Weak-support ESS threshold bits.
    pub minimum_local_ess_bits: u64,
    /// Pointwise confidence level bits.
    pub confidence_level_bits: u64,
    /// Simultaneous-band multiplier replicates, when requested.
    pub simultaneous_replicates: Option<u32>,
    /// Multiplier seed for the simultaneous band.
    pub multiplier_seed: u64,
}

/// Digest an inference binding.
///
/// # Errors
///
/// CBOR encode failure.
pub fn inference_binding_digest(wire: &InferenceBindingWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::InferenceBinding, wire)
}
