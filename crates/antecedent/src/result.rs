//! Analysis result artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_attribution::{
    AnomalyScores, ChangeAttributionResult, MechanismChangeDetection, UnitChangeResult,
};
use antecedent_core::{
    CausalResponse, Diagnostic, ExecutionPerformanceRecord, IdentificationStatus,
    LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord, ProvenanceGraph, ResponseEnvelope,
    ResponseValue, VariableId,
};
use antecedent_estimate::{
    CausalPosterior, EffectEstimate, InterferenceEstimate, InterventionalDistributionEstimate,
    TemporalMediationEstimate, TemporalMediationGrid, TransportEffectEstimate,
};
use antecedent_identify::{IdentificationResult, IdentifiedEstimand};
use antecedent_io::{AnalysisTraceWire, DerivationStepWire, assumptions_to_wire};
use antecedent_validate::{PredictiveCheckReport, RefutationReport};

use crate::gcm::IteResult;

/// Identification certificate retained from the actual execution, including class atoms.
#[derive(Clone, Debug)]
pub struct AnalysisIdentification {
    /// Full point or completion-envelope artifact in its original coordinates.
    pub identification: crate::Identification,
    /// Query whose functional was estimated.
    pub query: antecedent_core::CausalQuery,
    /// Supplied graph class, preserved even when fully oriented.
    pub graph_class: crate::GraphClass,
}

/// Row-weight target population of a retarget, bound to the data snapshot and
/// score table it was computed on.
///
/// The `RowWeights` population references [`Self::target_weights`], so the
/// target identity and every claim over the result name exactly this
/// weighting. The weights travel in the exported contract so a consumer can
/// re-derive the identity.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RowWeightsBinding {
    /// Weights in score-table row order.
    pub weights: std::sync::Arc<[f64]>,
    /// Declared parents of the weights.
    pub depends_on: std::sync::Arc<[VariableId]>,
    /// Rehashable target-weights identity payload.
    pub identity: antecedent_io::TargetWeightsIdentityWire,
    /// Digest of [`Self::identity`].
    pub target_weights: antecedent_core::SemanticDigest,
}

impl StudyResult {
    /// Target population a row-weight retarget recorded, when it differs from
    /// the prepared query.
    #[must_use]
    pub fn retarget_population(&self) -> Option<antecedent_core::TargetPopulation> {
        self.row_weights.as_ref().map(RowWeightsBinding::population)
    }
}

impl RowWeightsBinding {
    /// The population this binding defines.
    #[must_use]
    pub fn population(&self) -> antecedent_core::TargetPopulation {
        antecedent_core::TargetPopulation::RowWeights {
            weights: *self.target_weights.as_bytes(),
            depends_on: std::sync::Arc::clone(&self.depends_on),
        }
    }
}

/// Meaning of structural atom weights retained on a response result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StructuralWeightBasis {
    /// Probabilities supplied by a graph posterior.
    PosteriorProbability,
    /// Enumeration weights over CPDAG/PAG completions; not posterior probabilities.
    CompletionEnumeration,
    /// Caller-declared mass over class members. Not a graph posterior and not
    /// completion enumeration.
    CallerSuppliedClassPrior,
}

impl StructuralWeightBasis {
    /// Snake-case name used by the bindings and the result artifact wire format.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PosteriorProbability => "posterior_probability",
            Self::CompletionEnumeration => "completion_enumeration",
            Self::CallerSuppliedClassPrior => "caller_supplied_class_prior",
        }
    }
}

impl From<StructuralWeightBasis> for antecedent_io::StructuralWeightBasisWire {
    fn from(basis: StructuralWeightBasis) -> Self {
        match basis {
            StructuralWeightBasis::PosteriorProbability => Self::PosteriorProbability,
            StructuralWeightBasis::CompletionEnumeration => Self::CompletionEnumeration,
            StructuralWeightBasis::CallerSuppliedClassPrior => Self::CallerSuppliedClassPrior,
        }
    }
}

/// How identified structural atoms may be combined.
///
/// Mass-weighting is licensed only when every contributing atom names the same
/// estimand. Completion enumeration is a [`StructuralWeightBasis`], not a
/// reason to treat class members as posterior probability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StructuralAggregationPolicy {
    /// Identified atoms share one estimand identity; a probability-weighted
    /// mean of their values is the same functional.
    SameEstimandWeightedMean,
    /// Identified atoms share one partially identified estimand; publish the
    /// identified set, not a fake scalar mixture.
    IdentifiedSetEnvelope,
    /// Identified atoms name different estimands. No scalar mixture.
    GraphDependentAtoms,
}

impl StructuralAggregationPolicy {
    /// Snake-case name used in diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SameEstimandWeightedMean => "same_estimand_weighted_mean",
            Self::IdentifiedSetEnvelope => "identified_set_envelope",
            Self::GraphDependentAtoms => "graph_dependent_atoms",
        }
    }
}

/// One structural atom in a graph-dependent response.
#[derive(Clone, Debug)]
pub struct StructuralResponseAtom {
    /// Stable key within this result.
    pub graph_key: u64,
    /// Raw atom weight in the declared basis.
    pub weight: f64,
    /// Structural identification/evaluation status.
    pub status: IdentificationStatus,
    /// Numerical response when identified and evaluable.
    pub value: Option<ResponseValue>,
    /// Completion-conditional posterior, without assigning structural probabilities.
    pub posterior: Option<CausalPosterior>,
    /// Full function-valued atom, including its conditional sampling uncertainty.
    pub response: Option<CausalResponse>,
}

/// Structural uncertainty retained separately from sampling uncertainty.
#[derive(Clone, Debug)]
pub struct StructuralResponseMixture {
    /// Interpretation of [`StructuralResponseAtom::weight`].
    pub weight_basis: StructuralWeightBasis,
    /// Every examined structural atom, including nonidentified/unevaluable atoms.
    pub atoms: Vec<StructuralResponseAtom>,
    /// Fraction of total weight with an evaluable identified response.
    pub identified_mass: f64,
    /// Fraction proved or conservatively treated as structurally unidentified.
    pub unidentified_mass: f64,
    /// Fraction identified in theory but not evaluable by the selected estimator.
    pub unevaluable_mass: f64,
    /// Fraction on identified atoms that were never evaluated because the
    /// Interactive latency tier's graph budget left them out of the stratified
    /// subsample. Neither unidentified nor a failed estimate; the four masses
    /// sum to one. Zero outside the Interactive tier.
    pub subsampled_out_mass: f64,
    /// Pointwise range over identified atom point responses.
    pub identified_set: Option<ResponseEnvelope>,
    /// Interval for a scalar [`Self::identified_set`] that adds sampling
    /// uncertainty to the point bounds (C-3). Published on class-aware
    /// temporal Pulse / Sustained effects. Frequentist: an Imbens–Manski interval
    /// with per-completion endpoints from the shared circular-block replicates,
    /// covering the true effect with asymptotic probability at least the stated
    /// level whenever it is one retained identified completion's effect.
    /// Bayesian: product-posterior envelope quantiles from independently seeded
    /// per-completion posterior draws (every retained completion's posterior puts
    /// at most `1 − Φ(c)` of its mass outside each endpoint). When the completion
    /// enumeration was capped the interval is still published, flagged
    /// `truncated`, with a warning diagnostic.
    pub identified_set_interval: Option<antecedent_estimate::IdentifiedSetInterval>,
    /// Probability-weighted summary when a scalar `SameEstimandWeightedMean` mix is
    /// published (`E[τ | identified]`). Retained unidentified mass does not
    /// suppress this field; consumers should read `identification.status` and
    /// `unidentified_mass` alongside it.
    pub conditional_on_identified: Option<ResponseValue>,
    /// Whether reported mass covers the full class rather than a capped subset.
    pub full_mass_scope: bool,
    /// Atoms whose identification search was capped before a determination.
    pub truncated_atoms: usize,
}

/// End-to-end analysis result.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct StudyResult {
    /// Logical plan record.
    pub logical_plan: LogicalAnalysisPlanRecord,
    /// Physical plan record.
    pub physical_plan: PhysicalExecutionPlanRecord,
    /// Full identification artifact.
    pub identification: IdentificationResult,
    /// Complete identification certificate, when supplied by the execution path.
    pub certificate: Option<AnalysisIdentification>,
    /// Primary estimand used for estimation.
    pub estimand: IdentifiedEstimand,
    /// Point estimate + uncertainty (frequentist, or Bayesian posterior mean summary).
    ///
    /// For [`CausalQuery::Distribution`](antecedent_core::CausalQuery::Distribution) this holds the
    /// interventional mean of the first numeric outcome when defined (`ate` field), else NaN.
    /// Its `se_bootstrap` is the mean's bootstrap SE; do not form `ate ± z·se` for a
    /// probability — the bounded per-atom intervals are
    /// [`InterventionalDistributionEstimate::atom_uncertainty`], and a binary outcome's
    /// mean interval is [`InterventionalDistributionEstimate::mean_interval`].
    pub estimate: EffectEstimate,
    /// Function-valued causal response for [`CausalQuery::Response`](antecedent_core::CausalQuery::Response).
    pub response: Option<CausalResponse>,
    /// Structural atom/mass result for class-aware or graph-posterior responses.
    pub structural_response: Option<StructuralResponseMixture>,
    /// Full interventional distribution when the query was
    /// [`CausalQuery::Distribution`](antecedent_core::CausalQuery::Distribution).
    pub distribution: Option<InterventionalDistributionEstimate>,
    /// Bayesian posterior when `InferenceMode::Bayesian` was used.
    pub posterior: Option<CausalPosterior>,
    /// Temporal / static mediation decomposition when the query was mediation.
    pub mediation: Option<TemporalMediationEstimate>,
    /// Horizon-indexed temporal mediation decomposition. Present for temporal
    /// mediation, including single-horizon results; it does not imply a joint posterior.
    pub mediation_grid: Option<TemporalMediationGrid>,
    /// Unit-level ITE when the query was counterfactual.
    pub counterfactual: Option<IteResult>,
    /// Anomaly scores when the query was anomaly attribution.
    pub anomaly: Option<Vec<AnomalyScores>>,
    /// Change-attribution result.
    pub change_attribution: Option<ChangeAttributionResult>,
    /// Mechanism-change detections.
    pub mechanism_change: Option<Vec<MechanismChangeDetection>>,
    /// Unit-change attribution.
    pub unit_change: Option<UnitChangeResult>,
    /// Trial-to-target transport estimate when the query was transport.
    pub transport: Option<TransportEffectEstimate>,
    /// Design-based interference estimate when the query was interference.
    pub interference: Option<InterferenceEstimate>,
    /// Refutation reports (may be empty).
    pub refutations: Vec<RefutationReport>,
    /// Prior/posterior predictive check reports (Bayesian path; may be empty).
    pub predictive_checks: Vec<PredictiveCheckReport>,
    /// Diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Provenance.
    pub provenance: ProvenanceGraph,
    /// Support-matrix evidence contract that produced this result.
    ///
    /// `licensed` and `allowed_unlicensed` both yield a successful study.
    /// Downstream consumers must not treat a number as licensed unless this is
    /// [`crate::support::CellStatus::Licensed`]. `None` when the query is not on
    /// the public axis.
    pub support_status: Option<crate::support::CellStatus>,
    /// How the caller supplied structure. Graph-posterior mixtures are never a
    /// single adjustment set, even when every identified atom happens to agree.
    pub structure_source: crate::support::StructureSource,
    /// Performance record.
    pub performance: ExecutionPerformanceRecord,
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Candidate-selection screen recorded for a prepared batch family.
    pub candidate_selection: Option<crate::analysis::CandidateSelection>,
    /// How this execution formed its reported interval.
    pub interval: Option<IntervalBinding>,
    /// Row-weight population recorded by a nonconstant retarget.
    pub row_weights: Option<RowWeightsBinding>,
    /// Names of caller-supplied custom validators that ran on this execution.
    pub custom_validator_names: Vec<std::sync::Arc<str>>,
    /// Prepared contract this result was executed under, stamped by the
    /// [`crate::PreparedStudy`] entry point that produced it. `None` for
    /// results no prepared handle executed; those cannot be exported as claims.
    pub executed_contract: Option<ExecutedContract>,
    /// Population bindings in force, when the query names a predicate or a
    /// custom target distribution. Encoding the query needs them.
    pub population_registry: Option<antecedent_core::PopulationRegistry>,
}

/// Prepared contract a result was executed under.
///
/// Export recompiles the contract on the handle and refuses unless every
/// identity layer — including the data snapshot the result was computed on —
/// matches this stamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutedContract {
    /// Every contract identity layer, including the executed data snapshot.
    pub identities: antecedent_core::ContractIdentities,
    /// Validation suite whose refutations the result carries.
    pub refute: crate::RefuteSuite,
}

/// Nominal level of the interval a standard-error result reports.
///
/// An [`EffectEstimate`] carries a standard error, not endpoints; every
/// rendering of it (Python views, HTML, the analysis-result artifact's
/// `standard_error`) forms the two-sided `estimate ± 1.96·SE` interval.
/// Posterior summaries report the equal-tailed `q025` / `q975` interval at the
/// same level.
pub const REPORTED_SE_INTERVAL_LEVEL: f64 = 0.95;

/// Two-sided normal critical value of [`REPORTED_SE_INTERVAL_LEVEL`].
///
/// Every path that forms `estimate ± z·SE` at the published level reads this
/// one quantile rather than writing `normal_ppf(0.975)` again:
/// `0.5 + 0.95 / 2.0` is exactly `0.975` in binary64, so the value is
/// bit-identical to the literal it replaces.
#[must_use]
pub fn reported_se_interval_z() -> f64 {
    antecedent_stats::normal_ppf(0.5 + REPORTED_SE_INTERVAL_LEVEL / 2.0)
}

/// How one reported interval was formed: the construction the calibration
/// match key describes.
///
/// Derived from the result's own content by
/// [`StudyResult::primary_interval_binding`] and
/// [`StudyResult::identified_set_interval_binding`], so it always describes
/// what the execution reported, never what was requested.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct IntervalBinding {
    /// Interval method.
    pub method: antecedent_core::IntervalMethod,
    /// Analytic SE kind recorded by the estimator, when the method is analytic.
    pub se_kind: Option<antecedent_estimate::AnalyticSeKind>,
    /// Nominal coverage level of the reported interval.
    pub level: f64,
    /// Dependence rule of the construction: `iid`, `panel_cluster`, or
    /// `circular_block:<family>` ([`antecedent_estimate::CircularBlockFamily::tag`]).
    pub dependence: &'static str,
    /// Resampling replicates that succeeded (bootstrap, circular block, or
    /// simultaneous-band multipliers), when the interval is resampling-based.
    pub replicates_ok: Option<u32>,
    /// Resampling replicates that failed, when reported.
    pub replicates_failed: Option<u32>,
    /// Posterior draws behind a posterior interval.
    pub posterior_draws: Option<u32>,
}

impl IntervalBinding {
    const fn new(
        method: antecedent_core::IntervalMethod,
        level: f64,
        dependence: &'static str,
    ) -> Self {
        Self {
            method,
            se_kind: None,
            level,
            dependence,
            replicates_ok: None,
            replicates_failed: None,
            posterior_draws: None,
        }
    }

    /// Binding of an execution that reported no interval.
    #[must_use]
    pub const fn none(dependence: &'static str) -> Self {
        Self::new(antecedent_core::IntervalMethod::None, f64::NAN, dependence)
    }
}

/// Dependence label of a circular-block family.
#[must_use]
pub const fn circular_block_dependence(
    family: antecedent_estimate::CircularBlockFamily,
) -> &'static str {
    match family {
        antecedent_estimate::CircularBlockFamily::SingleWindow => "circular_block:single_window",
        antecedent_estimate::CircularBlockFamily::Mediation => "circular_block:mediation",
        antecedent_estimate::CircularBlockFamily::Sequential => "circular_block:sequential",
        antecedent_estimate::CircularBlockFamily::Mixture => "circular_block:mixture",
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Published scalar standard error and the label of its normal interval.
///
/// One owner of the facade's SE / `± z·SE` rule. Wire `standard_error`,
/// [`StudyResult::primary_interval_binding`], the Uncertainty slot, and the
/// Python copies all read this; they do not prefer `se_bootstrap` on their own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedScalarUncertainty {
    /// The one scalar SE the facade publishes, when licensed and positive-finite.
    pub standard_error: Option<f64>,
    /// Interval method for the normal interval formed from [`Self::standard_error`]:
    /// [`IntervalMethod::AnalyticSe`], [`IntervalMethod::BootstrapSe`], or
    /// [`IntervalMethod::None`]. Circular-block labeling is applied by
    /// [`StudyResult::primary_interval_binding`] when a block family is recorded.
    pub method: antecedent_core::IntervalMethod,
    /// Nominal level of the normal interval, or `NaN` when nothing was published.
    pub level: f64,
    /// Why a scalar SE was withheld, when the estimate is an IV result that
    /// cannot publish a Wald or bootstrap SE.
    pub withheld_reason: Option<&'static str>,
}

impl PublishedScalarUncertainty {
    /// Select the one published scalar SE for an [`EffectEstimate`].
    ///
    /// Rules:
    /// 1. Only a positive-finite SE is published (`Some` alone is not enough).
    /// 2. When first-stage diagnostics are present and `se_analytic` is
    ///    non-finite, do not fall through to `se_bootstrap` — withhold (weak
    ///    instrument / IV uncertainty not licensed as a Wald or bootstrap SE).
    ///    A later integration prefers an Anderson–Rubin interval when
    ///    `FirstStageDiagnostics` carries one; until that field exists this
    ///    still withholds rather than publishing `± 1.96·se_bootstrap`.
    /// 3. Otherwise prefer a positive-finite bootstrap SE over a positive-finite
    ///    analytic SE (AIPW and peers keep today's preference). IV and NN
    ///    matching are expected not to emit a competing finite pair after their
    ///    estimator-side fixes.
    #[must_use]
    pub fn select(estimate: &EffectEstimate) -> Self {
        use antecedent_core::IntervalMethod as M;

        if let Some(diagnostics) = &estimate.first_stage_diagnostics {
            // Prefer Anderson–Rubin when diagnostics carry one (sibling adds
            // `anderson_rubin: Option<(f64, f64, f64)>`). Until that field
            // exists, `anderson_rubin_interval` is always `None` and a
            // non-finite analytic SE withholds rather than publishing bootstrap.
            if let Some((_lower, _upper, level)) = anderson_rubin_interval(diagnostics) {
                return Self {
                    standard_error: None,
                    method: M::None,
                    level,
                    withheld_reason: Some(
                        "Anderson–Rubin interval preferred; Wald/bootstrap SE not licensed",
                    ),
                };
            }
            if !estimate.se_analytic.is_finite() {
                return Self::withhold_iv();
            }
            if positive_finite(estimate.se_analytic) {
                return Self {
                    standard_error: Some(estimate.se_analytic),
                    method: M::AnalyticSe,
                    level: REPORTED_SE_INTERVAL_LEVEL,
                    withheld_reason: None,
                };
            }
            return Self::withhold_iv();
        }

        if estimate.se_bootstrap.is_some_and(positive_finite) {
            return Self {
                standard_error: estimate.se_bootstrap,
                method: M::BootstrapSe,
                level: REPORTED_SE_INTERVAL_LEVEL,
                withheld_reason: None,
            };
        }
        if positive_finite(estimate.se_analytic) {
            return Self {
                standard_error: Some(estimate.se_analytic),
                method: M::AnalyticSe,
                level: REPORTED_SE_INTERVAL_LEVEL,
                withheld_reason: None,
            };
        }
        Self {
            standard_error: None,
            method: M::None,
            level: f64::NAN,
            withheld_reason: None,
        }
    }

    const fn withhold_iv() -> Self {
        Self {
            standard_error: None,
            method: antecedent_core::IntervalMethod::None,
            level: f64::NAN,
            withheld_reason: Some(
                "Wald/bootstrap SE not licensed (weak instrument or IV uncertainty withheld)",
            ),
        }
    }
}

/// Anderson–Rubin interval `(lower, upper, level)` when first-stage diagnostics
/// carry one. Sibling work adds the field; until it exists this always returns
/// `None` so the selector withholds instead of publishing a bootstrap SE.
fn anderson_rubin_interval(
    diagnostics: &antecedent_estimate::FirstStageDiagnostics,
) -> Option<(f64, f64, f64)> {
    // When `FirstStageDiagnostics.anderson_rubin` lands, return it here
    // (and set it explicitly in tests). Keep the diagnostics borrow so the
    // integration site stays typed against the live struct.
    let _ = diagnostics;
    None
}

fn draws_u32(n: usize) -> Option<u32> {
    u32::try_from(n).ok()
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn count_u32(value: f64) -> Option<u32> {
    (value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX)).then_some(value as u32)
}

fn support_values<'a>(response: &'a CausalResponse, id: &str) -> Option<&'a [f64]> {
    response
        .support
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.id.as_ref() == id)
        .map(|diagnostic| diagnostic.values.as_ref())
}

/// A Frequentist temporal response band built from circular-block replicates
/// of the lag-aligned surface (support diagnostic `response.temporal.block_length`).
fn temporal_response_band(response: &CausalResponse, level: f64) -> Option<IntervalBinding> {
    support_values(response, antecedent_estimate::TEMPORAL_RESPONSE_BLOCK_LENGTH)?;
    let mut binding = IntervalBinding::new(
        antecedent_core::IntervalMethod::CircularBlockSe,
        level,
        "circular_block:response",
    );
    if let Some(counts) = support_values(response, "response.observation_block_bootstrap") {
        binding.replicates_ok = counts.get(1).and_then(|n| count_u32(*n));
    }
    Some(binding)
}

impl StudyResult {
    /// Dependence rule the execution's data imposes before any circular-block
    /// family: panel executions cluster by unit, every other modality is `iid`
    /// unless a circular-block construction names its family.
    fn base_dependence(&self) -> &'static str {
        if self.logical_plan.data_classification == antecedent_core::DataClassification::Panel {
            "panel_cluster"
        } else {
            "iid"
        }
    }

    fn posterior_draw_count(&self) -> Option<u32> {
        self.posterior.as_ref().and_then(|posterior| draws_u32(posterior.draws.n_draws))
    }

    /// Construction of the primary interval this execution reported.
    ///
    /// `bayesian` is the program's inference family: a Bayesian response's
    /// band is a posterior band even when the draws live in a referenced
    /// artifact rather than on the result.
    ///
    /// One owner of the rule, read by the claim's calibration match key:
    ///
    /// 1. A function-valued response reports its `ResponseUncertainty`
    ///    (`None` → [`IntervalMethod::None`](antecedent_core::IntervalMethod::None);
    ///    simultaneous band; identified-envelope band; posterior; a temporal
    ///    circular-block band; otherwise the influence-function band).
    /// 2. An interventional distribution with a binary outcome reports its
    ///    bounded mean interval.
    /// 3. A posterior reports its `q025` / `q975` interval.
    /// 4. A standard error reports `estimate ± 1.96·SE` from
    ///    [`PublishedScalarUncertainty::select`]: the published bootstrap SE
    ///    when that is the selected SE (circular-block when a family is
    ///    recorded), else the published analytic SE with the estimator's
    ///    recorded kind. A bootstrap interval here is a normal interval from
    ///    the bootstrap SE, not a percentile interval (replicates are not
    ///    retained). A requested bootstrap that reported no positive-finite SE
    ///    is not a bootstrap interval; an IV result that withheld Wald/bootstrap
    ///    uncertainty is [`IntervalMethod::None`](antecedent_core::IntervalMethod::None).
    /// 5. When no finite scalar interval exists, a reported identified-set
    ///    interval is the primary interval; otherwise nothing was reported.
    #[must_use]
    pub fn primary_interval_binding(&self, bayesian: bool) -> IntervalBinding {
        use antecedent_core::{IntervalMethod as M, ResponseUncertainty as U};
        let base = self.base_dependence();
        if let Some(response) = &self.response {
            // A Bayesian response publishes credible bands; the draws behind a
            // band are in the referenced posterior artifact, not on the result.
            let posterior = bayesian || self.posterior.is_some();
            let mut binding = match &response.uncertainty {
                U::None => return IntervalBinding::none(base),
                U::Posterior { .. } => {
                    IntervalBinding::new(M::PosteriorQuantile, REPORTED_SE_INTERVAL_LEVEL, base)
                }
                U::Scalar { level, .. } | U::PointwiseBand { level, .. } if posterior => {
                    IntervalBinding::new(M::PosteriorQuantile, *level, base)
                }
                U::Scalar { level, .. } | U::PointwiseBand { level, .. } => {
                    temporal_response_band(response, *level)
                        .unwrap_or_else(|| IntervalBinding::new(M::AnalyticSe, *level, base))
                }
                U::SimultaneousBand { level, replicates, .. } => {
                    let mut binding = IntervalBinding::new(M::SimultaneousBand, *level, base);
                    binding.replicates_ok = Some(*replicates);
                    binding
                }
                U::IdentifiedEnvelopeBand { level, .. } => {
                    IntervalBinding::new(M::IdentifiedSet, *level, base)
                }
            };
            if binding.method == M::PosteriorQuantile {
                binding.posterior_draws = self.posterior_draw_count();
            }
            return binding;
        }
        if let Some(distribution) = &self.distribution {
            if self.posterior.is_none() {
                match distribution.mean_interval {
                    Some(antecedent_estimate::ProbabilityInterval::Bounded { level, .. }) => {
                        let mut binding = IntervalBinding::new(M::BootstrapSe, level, base);
                        binding.replicates_ok = distribution.bootstrap_replicates_ok;
                        binding.replicates_failed = distribution.bootstrap_replicates_failed;
                        return binding;
                    }
                    Some(antecedent_estimate::ProbabilityInterval::Unavailable(_)) => {
                        return IntervalBinding::none(base);
                    }
                    None => {}
                }
            }
        }
        if self.posterior.is_some() {
            let mut binding =
                IntervalBinding::new(M::PosteriorQuantile, REPORTED_SE_INTERVAL_LEVEL, base);
            binding.posterior_draws = self.posterior_draw_count();
            return binding;
        }
        let estimate = &self.estimate;
        if estimate.ate.is_finite() {
            let published = PublishedScalarUncertainty::select(estimate);
            match published.method {
                M::BootstrapSe => {
                    let (method, dependence) = match estimate.block_family {
                        Some(family) => (M::CircularBlockSe, circular_block_dependence(family)),
                        None => (M::BootstrapSe, base),
                    };
                    let mut binding = IntervalBinding::new(method, published.level, dependence);
                    binding.replicates_ok = estimate.bootstrap_replicates_ok;
                    binding.replicates_failed = estimate.bootstrap_replicates_failed;
                    return binding;
                }
                M::AnalyticSe => {
                    let mut binding = IntervalBinding::new(M::AnalyticSe, published.level, base);
                    binding.se_kind = estimate.se_kind;
                    return binding;
                }
                M::None if published.withheld_reason.is_some() => {
                    return IntervalBinding::none(base);
                }
                _ => {}
            }
        }
        self.identified_set_interval_binding().unwrap_or_else(|| IntervalBinding::none(base))
    }

    /// Construction of a reported identified-set interval
    /// (`structural_response.identified_set_interval`), when one was reported.
    ///
    /// A class-aware scalar can report both a frozen-weight point with its
    /// shared-block SE and an Imbens–Manski interval for the identified set;
    /// each states its own calibration.
    #[must_use]
    pub fn identified_set_interval_binding(&self) -> Option<IntervalBinding> {
        use antecedent_estimate::IdentifiedSetIntervalMethod as Method;
        let structural = self.structural_response.as_ref()?;
        let interval = structural.identified_set_interval.as_ref()?;
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::IdentifiedSet,
            interval.level,
            self.base_dependence(),
        );
        match interval.method {
            Method::ImbensManskiSharedBlock => {
                binding.dependence =
                    circular_block_dependence(antecedent_estimate::CircularBlockFamily::Mixture);
                binding.replicates_ok = self.estimate.bootstrap_replicates_ok;
                binding.replicates_failed = self.estimate.bootstrap_replicates_failed;
            }
            Method::ProductPosteriorEnvelopeQuantile => {
                binding.posterior_draws = structural
                    .atoms
                    .iter()
                    .filter_map(|atom| atom.posterior.as_ref())
                    .filter_map(|posterior| draws_u32(posterior.draws.n_draws))
                    .min()
                    .or_else(|| self.posterior_draw_count());
            }
            _ => {}
        }
        Some(binding)
    }

    /// Construction of a counterfactual's published per-unit intervals
    /// (`counterfactual.unit_effect_intervals`), when the execution formed them.
    #[must_use]
    pub fn unit_effect_interval_binding(&self) -> Option<IntervalBinding> {
        let intervals = self.counterfactual.as_ref()?.unit_effect_intervals.as_ref()?;
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::UnitPosteriorQuantile,
            intervals.level,
            self.base_dependence(),
        );
        binding.posterior_draws = self.posterior_draw_count();
        Some(binding)
    }

    /// Construction of a temporal response's published simultaneous band
    /// (support diagnostic `response.simultaneous_band.critical`), when one
    /// accompanies a pointwise band.
    #[must_use]
    pub fn simultaneous_band_binding(&self, bayesian: bool) -> Option<IntervalBinding> {
        let response = self.response.as_ref()?;
        if matches!(
            response.uncertainty,
            antecedent_core::ResponseUncertainty::SimultaneousBand { .. }
        ) {
            return None;
        }
        let critical = support_values(response, antecedent_estimate::SIMULTANEOUS_BAND_CRITICAL)?;
        let level = *critical.first()?;
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::SimultaneousBand,
            level,
            self.primary_interval_binding(bayesian).dependence,
        );
        binding.replicates_ok = critical.get(2).and_then(|n| count_u32(*n));
        if bayesian || self.posterior.is_some() {
            binding.posterior_draws = binding.replicates_ok.take();
        }
        Some(binding)
    }

    /// Every interval this execution reported: the primary interval first,
    /// then an identified-set interval and a simultaneous band when they are
    /// reported beside it. Each states its own calibration on the claim.
    #[must_use]
    pub fn reported_interval_bindings(&self, bayesian: bool) -> Vec<IntervalBinding> {
        let primary = self.primary_interval_binding(bayesian);
        let mut out = vec![primary];
        for extra in [
            self.identified_set_interval_binding(),
            self.simultaneous_band_binding(bayesian),
            self.unit_effect_interval_binding(),
        ]
        .into_iter()
        .flatten()
        {
            if out.iter().all(|seen| seen.method != extra.method) {
                out.push(extra);
            }
        }
        out
    }

    /// Re-derive [`Self::interval`] from the reported content.
    pub(crate) fn rebind_interval(&mut self, bayesian: bool) {
        self.interval = Some(self.primary_interval_binding(bayesian));
    }

    /// Primary scalar effect for display and tests.
    ///
    /// Prefer this over reading [`EffectEstimate::ate`] directly when the query may be a
    /// distribution, mediation, or counterfactual: returns the interventional mean,
    /// mediation total, or mean ITE when present, otherwise the estimate's `ate` field.
    #[must_use]
    pub fn effect(&self) -> f64 {
        if let Some(dist) = &self.distribution {
            return dist.mean;
        }
        if let Some(med) = &self.mediation {
            if let Some(total) = med.total {
                return total;
            }
        }
        if let Some(cf) = &self.counterfactual {
            return cf.mean_ite;
        }
        self.estimate.ate
    }

    /// Borrow the logical plan record (semantics).
    #[must_use]
    pub fn logical_plan(&self) -> &LogicalAnalysisPlanRecord {
        &self.logical_plan
    }

    /// Borrow the physical plan record (layouts / kernels / batching).
    #[must_use]
    pub fn physical_plan(&self) -> &PhysicalExecutionPlanRecord {
        &self.physical_plan
    }

    /// Build a durable analysis-trace wire payload (assumptions + derivation).
    #[must_use]
    pub fn analysis_trace_wire(&self) -> AnalysisTraceWire {
        AnalysisTraceWire {
            assumptions: assumptions_to_wire(&self.estimate.assumptions),
            derivation: self
                .identification
                .derivation
                .steps
                .iter()
                .map(|s| DerivationStepWire {
                    rule: s.rule.to_string(),
                    detail: s.detail.to_string(),
                })
                .collect(),
            method: self.estimand.method.to_string(),
            adjustment_set: self.estimand.adjustment_set.iter().map(|id| id.raw()).collect(),
            support_status: self
                .support_status
                .map(crate::support::CellStatus::as_str)
                .map(str::to_string),
            allowlist_reason: self
                .support_status
                .and_then(crate::support::CellStatus::allowlist_reason)
                .map(str::to_string),
            allowlist_parent: self
                .support_status
                .and_then(crate::support::CellStatus::allowlist_parent)
                .map(str::to_string),
        }
    }
}

#[cfg(test)]
mod weight_basis_tests {
    use super::StructuralWeightBasis;

    #[test]
    fn as_str_matches_the_wire_spelling() {
        for basis in [
            StructuralWeightBasis::PosteriorProbability,
            StructuralWeightBasis::CompletionEnumeration,
            StructuralWeightBasis::CallerSuppliedClassPrior,
        ] {
            let wire = antecedent_io::StructuralWeightBasisWire::from(basis);
            assert_eq!(serde_json::to_value(wire).unwrap(), basis.as_str());
        }
    }
}

#[cfg(test)]
mod published_scalar_uncertainty_tests {
    use antecedent_core::{AssumptionSet, IntervalMethod};
    use antecedent_estimate::{EffectEstimate, FirstStageDiagnostics, OverlapPolicy};

    use super::PublishedScalarUncertainty;

    fn estimate(ate: f64, se_analytic: f64, se_bootstrap: Option<f64>) -> EffectEstimate {
        let mut estimate =
            EffectEstimate::new(ate, se_analytic, AssumptionSet::default(), OverlapPolicy::ExplicitOverride);
        estimate.se_bootstrap = se_bootstrap;
        estimate
    }

    fn weak_first_stage() -> FirstStageDiagnostics {
        // When `anderson_rubin` appears on this struct, set it explicitly here.
        FirstStageDiagnostics {
            f_statistic: 5.0,
            df1: 1,
            df2: 98,
            partial_r2: 0.05,
        }
    }

    #[test]
    fn weak_iv_non_finite_analytic_withholds_bootstrap() {
        let estimate = estimate(1.0, f64::NAN, Some(0.2))
            .with_first_stage_diagnostics(Some(weak_first_stage()));
        let published = PublishedScalarUncertainty::select(&estimate);
        assert!(published.standard_error.is_none());
        assert_eq!(published.method, IntervalMethod::None);
        assert!(published.withheld_reason.is_some());
    }

    #[test]
    fn finite_analytic_without_bootstrap_publishes_analytic() {
        let estimate = estimate(1.0, 0.15, None);
        let published = PublishedScalarUncertainty::select(&estimate);
        assert_eq!(published.standard_error, Some(0.15));
        assert_eq!(published.method, IntervalMethod::AnalyticSe);
        assert!(published.withheld_reason.is_none());
    }

    #[test]
    fn both_finite_without_first_stage_prefers_bootstrap() {
        let estimate = estimate(1.0, 0.15, Some(0.2));
        let published = PublishedScalarUncertainty::select(&estimate);
        assert_eq!(published.standard_error, Some(0.2));
        assert_eq!(published.method, IntervalMethod::BootstrapSe);
        assert!(published.withheld_reason.is_none());
    }
}
