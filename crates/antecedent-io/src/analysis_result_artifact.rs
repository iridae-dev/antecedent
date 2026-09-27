//! Composite analysis-result artifact.
//!
//! This additive container keeps response, posterior, mediation-grid, and
//! structural-mixture axes together without extending the exhaustive legacy
//! `CausalPayloadKind` enum.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

#[path = "support_graphless_data.rs"]
mod graphless_data;

use crate::{
    ArtifactKind, ArtifactManifest, AssumptionRecordWire, CausalQueryWire, CausalResponseWire,
    CompressPolicy, DiagnosticWire, EncodedArtifact, IdentificationResultWire, IoError,
    ProvenanceWire, RefutationReportWire, STABLE_FORMAT, SemanticVersion, from_cbor,
    pack_section_shared, read_and_migrate, to_cbor,
};

const ARTIFACT_KIND: &str = "analysis_result";
const HEADER_SECTION: &str = "analysis_result.header";
const BODY_SECTION: &str = "analysis_result.body";

/// One posterior interval summary in a temporal mediation slice.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MediationPosteriorSummaryWire {
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub standard_deviation: f64,
    /// Lower equal-tail quantile.
    pub q025: f64,
    /// Upper equal-tail quantile.
    pub q975: f64,
}

/// Pointwise uncertainty for one mediation horizon.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TemporalMediationUncertaintyWire {
    /// Frequentist pointwise standard error.
    FrequentistPointwise {
        /// Standard error for the requested contrast.
        standard_error: Option<f64>,
    },
    /// Per-horizon posterior summaries.
    BayesianPointwise {
        /// Requested contrast.
        requested: MediationPosteriorSummaryWire,
        /// Total effect.
        total: MediationPosteriorSummaryWire,
        /// Direct effect.
        direct: MediationPosteriorSummaryWire,
        /// Mediated effect.
        mediated: MediationPosteriorSummaryWire,
        /// Draw count.
        n_draws: u64,
        /// Inference backend.
        backend: String,
    },
    /// No justified uncertainty.
    Unavailable,
}

/// One horizon in a temporal mediation grid.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TemporalMediationSliceWire {
    /// Requested horizon.
    pub horizon: u32,
    /// Identification status at this horizon.
    pub identification_status: crate::IdentificationStatusWire,
    /// Identifier method.
    pub method: String,
    /// Horizon-specific adjustment set.
    pub adjustment: Vec<crate::HorizonAdjustmentNodeWire>,
    /// Requested effect.
    pub effect: f64,
    /// Total effect.
    pub total: Option<f64>,
    /// Direct effect.
    pub direct: Option<f64>,
    /// Mediated effect.
    pub mediated: Option<f64>,
    /// Pointwise uncertainty.
    pub uncertainty: TemporalMediationUncertaintyWire,
    /// Completion-specific structural interval.
    pub identified_set: Option<[f64; 2]>,
    /// Horizon-local diagnostics.
    pub diagnostics: Vec<DiagnosticWire>,
}

/// Durable horizon-indexed temporal mediation result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TemporalMediationGridWire {
    /// Horizon slices in query order.
    pub slices: Vec<TemporalMediationSliceWire>,
    /// Whether one joint posterior generated all horizons.
    pub joint_posterior: bool,
}

/// Meaning of structural atom weights.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StructuralWeightBasisWire {
    /// Graph-posterior probability.
    PosteriorProbability,
    /// Completion-enumeration weight.
    CompletionEnumeration,
    /// Caller-declared mass over class members.
    CallerSuppliedClassPrior,
}

/// One graph/completion response atom.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StructuralResponseAtomWire {
    /// Stable atom key.
    pub graph_key: u64,
    /// Raw atom weight.
    pub weight: f64,
    /// Atom identification status.
    pub identification_status: crate::IdentificationStatusWire,
    /// Numerical response when evaluable.
    pub value: Option<crate::ResponseValueWire>,
    /// Completion-conditional posterior artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_artifact: Option<Vec<u8>>,
    /// Full response with conditional sampling uncertainty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<crate::CausalResponseWire>,
}

/// Construction of an [`IdentifiedSetIntervalWire`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IdentifiedSetIntervalMethodWire {
    /// Frequentist Imbens–Manski interval from shared circular-block replicates.
    ImbensManskiSharedBlock,
    /// Product-posterior envelope quantiles at Imbens–Manski tails.
    #[serde(alias = "imbens_manski_posterior_draws")]
    ProductPosteriorEnvelopeQuantile,
}

/// Interval for the identified set of a class-aware scalar effect. Frequentist
/// (`imbens_manski_shared_block`): covers the true effect with asymptotic
/// probability at least `level` whenever it is one retained identified
/// completion's effect. Bayesian (`product_posterior_envelope_quantile`): every
/// retained completion's posterior puts at most `1 − Φ(critical_value)` of its
/// mass outside each endpoint. Mirrors [`antecedent_estimate::IdentifiedSetInterval`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IdentifiedSetIntervalWire {
    /// Nominal coverage of the true (completion-specific) effect.
    pub level: f64,
    /// Lower endpoint of the interval.
    pub lower: f64,
    /// Upper endpoint of the interval.
    pub upper: f64,
    /// Estimated lower bound `min_g θ̂_g`.
    pub bound_lower: f64,
    /// Estimated upper bound `max_g θ̂_g`.
    pub bound_upper: f64,
    /// Endpoint SD: `lower = bound_lower − critical_value · lower_se`.
    pub lower_se: f64,
    /// Endpoint SD: `upper = bound_upper + critical_value · upper_se`.
    pub upper_se: f64,
    /// Imbens–Manski critical value.
    pub critical_value: f64,
    /// Whether the estimated width passed the moment-selection threshold.
    pub width_retained: bool,
    /// Identified completions whose effects enter the set (every fitted
    /// completion, in both constructions).
    pub completions: u64,
    /// Replicates or posterior draws behind the SDs.
    pub replicates: u64,
    /// Construction.
    pub method: IdentifiedSetIntervalMethodWire,
    /// The completion enumeration (or its equivalence audit) was capped: the
    /// set spans retained completions only. Omitted when false; artifacts
    /// written before the field existed decode as `false`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

impl IdentifiedSetIntervalMethodWire {
    /// Wire tag for an in-memory construction.
    ///
    /// # Errors
    ///
    /// A construction this wire version has no tag for.
    pub fn try_from_method(
        method: antecedent_estimate::IdentifiedSetIntervalMethod,
    ) -> Result<Self, IoError> {
        use antecedent_estimate::IdentifiedSetIntervalMethod as Method;
        match method {
            Method::ImbensManskiSharedBlock => Ok(Self::ImbensManskiSharedBlock),
            Method::ProductPosteriorEnvelopeQuantile => Ok(Self::ProductPosteriorEnvelopeQuantile),
            other => Err(IoError::Convert(format!(
                "identified-set interval construction {other:?} has no wire tag"
            ))),
        }
    }

    /// In-memory construction for this wire tag.
    #[must_use]
    pub const fn method(self) -> antecedent_estimate::IdentifiedSetIntervalMethod {
        use antecedent_estimate::IdentifiedSetIntervalMethod as Method;
        match self {
            Self::ImbensManskiSharedBlock => Method::ImbensManskiSharedBlock,
            Self::ProductPosteriorEnvelopeQuantile => Method::ProductPosteriorEnvelopeQuantile,
        }
    }
}

/// Encode an identified-set interval.
///
/// # Errors
///
/// A construction with no wire tag (never silently relabelled).
pub fn identified_set_interval_to_wire(
    interval: &antecedent_estimate::IdentifiedSetInterval,
) -> Result<IdentifiedSetIntervalWire, IoError> {
    Ok(IdentifiedSetIntervalWire {
        level: interval.level,
        lower: interval.lower,
        upper: interval.upper,
        bound_lower: interval.bound_lower,
        bound_upper: interval.bound_upper,
        lower_se: interval.lower_se,
        upper_se: interval.upper_se,
        critical_value: interval.critical_value,
        width_retained: interval.width_retained,
        completions: u64::try_from(interval.completions).unwrap_or(u64::MAX),
        replicates: u64::try_from(interval.replicates).unwrap_or(u64::MAX),
        method: IdentifiedSetIntervalMethodWire::try_from_method(interval.method)?,
        truncated: interval.truncated,
    })
}

/// Decode and validate an identified-set interval.
///
/// # Errors
///
/// Non-finite values, a level outside `(0, 1)`, inverted endpoints or bounds,
/// negative SDs or critical value, no completions, or fewer than two replicates.
pub fn identified_set_interval_from_wire(
    wire: &IdentifiedSetIntervalWire,
) -> Result<antecedent_estimate::IdentifiedSetInterval, IoError> {
    let finite = [
        wire.level,
        wire.lower,
        wire.upper,
        wire.bound_lower,
        wire.bound_upper,
        wire.lower_se,
        wire.upper_se,
        wire.critical_value,
    ]
    .iter()
    .all(|v| v.is_finite());
    if !(finite && wire.level > 0.0 && wire.level < 1.0)
        || wire.lower > wire.upper
        || wire.bound_lower > wire.bound_upper
        || wire.lower_se < 0.0
        || wire.upper_se < 0.0
        || wire.critical_value < 0.0
        || wire.completions == 0
        || wire.replicates < 2
    {
        return Err(IoError::Convert(
            "identified-set interval must be finite with level in (0, 1), ordered endpoints \
             and bounds, nonnegative SDs, at least one completion and two replicates"
                .into(),
        ));
    }
    let count = |v: u64| {
        usize::try_from(v)
            .map_err(|_| IoError::Convert("identified-set interval count overflows".into()))
    };
    Ok(antecedent_estimate::IdentifiedSetInterval {
        level: wire.level,
        lower: wire.lower,
        upper: wire.upper,
        bound_lower: wire.bound_lower,
        bound_upper: wire.bound_upper,
        lower_se: wire.lower_se,
        upper_se: wire.upper_se,
        critical_value: wire.critical_value,
        width_retained: wire.width_retained,
        completions: count(wire.completions)?,
        replicates: count(wire.replicates)?,
        method: wire.method.method(),
        truncated: wire.truncated,
    })
}

/// Structural-mixture response metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StructuralResponseMixtureWire {
    /// Meaning of atom weights.
    pub weight_basis: StructuralWeightBasisWire,
    /// Examined structural atoms.
    pub atoms: Vec<StructuralResponseAtomWire>,
    /// Evaluable identified mass.
    pub identified_mass: f64,
    /// Structurally unidentified mass.
    pub unidentified_mass: f64,
    /// Identified but numerically unevaluable mass.
    pub unevaluable_mass: f64,
    /// Identified mass the Interactive latency tier left out of its graph
    /// subsample and never evaluated. Absent (zero) outside that tier and on
    /// artifacts written before the field existed.
    #[serde(default, skip_serializing_if = "is_zero_mass")]
    pub subsampled_out_mass: f64,
    /// Pointwise identified set.
    pub identified_set: Option<crate::ResponseEnvelopeWire>,
    /// Imbens–Manski interval for a scalar identified set (format 0.5). Absent on
    /// format-0.4 artifacts and whenever no interval was computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identified_set_interval: Option<IdentifiedSetIntervalWire>,
    /// Conditional summary, only for posterior-probability weights.
    pub conditional_on_identified: Option<crate::ResponseValueWire>,
    /// Whether masses cover the full structural support.
    pub full_mass_scope: bool,
    /// Number of capped atom searches.
    pub truncated_atoms: u64,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's skip_serializing_if passes `&T`.
fn is_zero_mass(mass: &f64) -> bool {
    *mass == 0.0
}

/// A temporal identification certificate with its explicit unfolded variable namespace.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TemporalIdentificationWire {
    /// Requested outcome horizon.
    pub horizon: u32,
    /// Dense identification id indexes this map to a base-schema variable and offset.
    pub variables: Vec<crate::HorizonAdjustmentNodeWire>,
    /// Horizon-specific identification and derivation, using dense ids above.
    pub identification: IdentificationResultWire,
}

/// One probability atom from an interventional distribution result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistributionAtomWire {
    /// Outcome assignment in query outcome order.
    pub outcomes: Vec<(u32, crate::ValueWire)>,
    /// Conditioning assignment in query conditioning order.
    #[serde(default)]
    pub conditioning: Vec<(u32, crate::ValueWire)>,
    /// Probability mass for this atom.
    pub probability: f64,
}

/// Atom distribution reported by a checked interventional-distribution execution.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InterventionalDistributionWire {
    /// Outcome and conditioning probability atoms.
    pub atoms: Vec<DistributionAtomWire>,
}

/// Held-out doubly robust policy answer retained separately from scalar effects.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PolicyValueWire {
    /// Net policy value.
    pub policy_value: f64,
    /// Net reference value.
    pub reference_value: f64,
    /// Paired incremental value.
    pub incremental_value: f64,
    /// Reference minus policy value.
    pub relative_value_gap: f64,
    /// Treatment rate.
    pub treatment_rate: f64,
    /// Total policy cost.
    pub total_cost: f64,
    /// Policy, reference, and paired incremental row-score SEs.
    pub policy_standard_error: f64,
    /// Reference value SE.
    pub reference_standard_error: f64,
    /// Paired incremental value SE.
    pub incremental_standard_error: f64,
    /// Pointwise 95% interval for policy value when licensed by the retained route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_interval_95: Option<[f64; 2]>,
    /// Paired pointwise 95% interval for policy minus reference value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incremental_interval_95: Option<[f64; 2]>,
    /// Prediction ownership declaration.
    pub prediction_ownership: String,
    /// Propensity support range.
    pub propensity_min: f64,
    /// Maximum propensity.
    pub propensity_max: f64,
    /// Explicit uncertainty semantics.
    pub uncertainty: String,
    /// Exact graphless policy license, absent when no policy row matches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphless_support_status: Option<String>,
    /// Held-out randomized uplift by descending frozen score bin.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uplift_bins: Vec<UpliftBinWire>,
    /// Point-only action effects versus control within fixed baseline strata.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub multi_action_cate: Vec<MultiActionCateWire>,
    /// Finite-class regret relative to a prespecified selected candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regret: Option<FixedCandidateRegretWire>,
}

/// Paired candidate contrasts and simultaneous regret endpoints.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FixedCandidateRegretWire {
    /// Estimated net value of each fixed candidate.
    pub candidate_values: Vec<f64>,
    /// Paired candidate-minus-selected standard errors.
    pub contrast_standard_errors: Vec<f64>,
    /// Estimated best-in-class value gap.
    pub regret: f64,
    /// Bonferroni simultaneous 95% bounds.
    pub interval_95: [f64; 2],
    /// Selected candidate index.
    pub selected_index: usize,
}

/// One conditional dose-response cell with explicit local support diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContinuousDosePointWire {
    /// Baseline group label.
    pub baseline_group: String,
    /// Prespecified target dose.
    pub target_dose: f64,
    /// Point response.
    pub response: f64,
    /// Number of local rows.
    pub local_rows: usize,
    /// Effective local sample size.
    pub effective_sample_size: f64,
    /// Lowest supplied local density.
    pub minimum_dose_density: f64,
    /// Largest normalized local weight.
    pub maximum_normalized_weight: f64,
    /// Descriptive local outcome SD, not a standard error.
    pub local_outcome_sd: f64,
}

/// Point-only conditional continuous-dose response grid.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContinuousDoseResponseWire {
    /// Group-target cells in stable group and target order.
    pub points: Vec<ContinuousDosePointWire>,
    /// Prespecified kernel bandwidth.
    pub bandwidth: f64,
    /// Caller-declared density provenance.
    pub density_provenance: String,
    /// Explicit absence of interval inference.
    pub uncertainty: String,
    /// Fixed group-to-dose policy value, when requested by the query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_policy: Option<DosePolicyValueWire>,
    /// Exact graphless fixed-dose policy license, absent for off-axis results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphless_support_status: Option<String>,
}

/// Portable paired value of a fixed kernel-smoothed dose policy.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DosePolicyValueWire {
    /// Frozen policy group-dose map in group order.
    pub policy_doses: Vec<(String, f64)>,
    /// Frozen reference group-dose map in group order.
    pub reference_doses: Vec<(String, f64)>,
    /// Kernel-smoothed value under the fixed policy.
    pub policy_value: f64,
    /// Kernel-smoothed value under the fixed reference.
    pub reference_value: f64,
    /// Paired value difference.
    pub incremental_value: f64,
    /// Independent-row policy variance.
    pub policy_variance: f64,
    /// Independent-row reference variance.
    pub reference_variance: f64,
    /// Paired independent-row incremental variance.
    pub incremental_variance: f64,
    /// Policy pointwise interval, when supported.
    pub policy_interval_95: Option<[f64; 2]>,
    /// Reference pointwise interval, when supported.
    pub reference_interval_95: Option<[f64; 2]>,
    /// Incremental pointwise interval, when supported.
    pub incremental_interval_95: Option<[f64; 2]>,
    /// Smallest local window row count across requested doses.
    pub minimum_local_rows: usize,
    /// Smallest local effective sample size across requested doses.
    pub minimum_effective_sample_size: f64,
    /// Largest local normalized weight across requested doses.
    pub maximum_normalized_weight: f64,
    /// Smallest supplied observed-dose density across requested local windows.
    pub minimum_dose_density: f64,
}

/// One randomized multi-action conditional effect.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MultiActionCateWire {
    /// Pre-treatment group label.
    pub group: String,
    /// Action compared with control.
    pub action: String,
    /// Point estimate of the action minus control contrast.
    pub effect: f64,
    /// Paired action-minus-control row-score standard error.
    #[serde(default)]
    pub standard_error: f64,
    /// Pointwise interval when randomized group support is sufficient.
    #[serde(default)]
    pub interval_95: Option<[f64; 2]>,
    /// Total rows in the group.
    pub evaluation_rows: usize,
    /// Rows assigned the action.
    pub observed_action_rows: usize,
    /// Rows assigned control.
    pub observed_control_rows: usize,
}

/// One retained held-out randomized uplift bin.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UpliftBinWire {
    /// Zero-based descending score rank.
    pub rank: usize,
    /// Randomized inverse-probability contrast.
    pub effect: f64,
    /// Independent-subject row-score standard error.
    pub standard_error: f64,
    /// Evaluation subjects in the bin.
    pub evaluation_rows: usize,
    /// Pointwise 95% interval when fixed-rank and observed assignment support pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_95: Option<[f64; 2]>,
}

/// Cohort, period, event time, effect, treated count, control count, SE, cluster count.
type EventTimeEffect = (i64, i64, i64, f64, usize, usize, f64, usize);

/// Panel `DiD` result section; uncertainty is an SE only and carries no interval claim.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PanelDidWire {
    /// Difference in mean subject-level changes.
    pub effect: f64,
    /// Cluster-robust standard error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_error: Option<f64>,
    /// Pointwise normal interval for a sufficiently supported scalar `DiD` contrast.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_95: Option<[f64; 2]>,
    /// Treated subject count.
    pub treated_subjects: usize,
    /// Comparison subject count.
    pub comparison_subjects: usize,
    /// Total distinct inference clusters.
    pub clusters: usize,
    /// Explicit uncertainty semantics tag.
    pub uncertainty: String,
    /// Exact graphless two-period `DiD` license, absent off-axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphless_support_status: Option<String>,
    /// Cohort, period, event time, effect, treated count, control count, SE, cluster count.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_time_effects: Vec<EventTimeEffect>,
    /// Optional post-adoption pointwise intervals aligned with `event_time_effects`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_time_intervals_95: Vec<Option<[f64; 2]>>,
    /// Propensity range, effective control count, and caller cross-fit declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub augmented: Option<(f64, f64, f64, bool)>,
}

/// Synthetic-control point result and donor-support diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SyntheticControlWire {
    /// Average post-intervention treated-minus-synthetic outcome.
    pub effect: f64,
    /// Root mean squared pre-treatment fit gap.
    pub pre_treatment_rmse: f64,
    /// Donor weights in stable unit-name order.
    pub donor_weights: Vec<(String, f64)>,
    /// Leave-one-donor-out placebo effects in donor order.
    pub placebo_effects: Vec<f64>,
    /// Uncalibrated descriptive placebo rank.
    pub placebo_rank: f64,
    /// Effective donor count from the weight concentration.
    pub effective_donors: f64,
    /// Number of observed pre-treatment periods.
    pub n_pre_periods: usize,
    /// Number of observed post-treatment periods.
    pub n_post_periods: usize,
    /// Explicit point-only uncertainty statement.
    pub uncertainty: String,
    /// Exact sharp-null p-value under declared uniform single-unit assignment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub randomization_p_value: Option<f64>,
    /// Constant additive effect tested by the exact Fisher p-value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub randomization_null_effect: Option<f64>,
    /// Absolute post-gap statistics for every candidate treated unit.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub randomization_statistics: Vec<(String, f64)>,
    /// Unadjusted simplex gap for augmented synthetic control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unadjusted_effect: Option<f64>,
    /// Donor ridge prediction difference subtracted from the simplex gap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_model_correction: Option<f64>,
    /// Positive donor outcome-model ridge penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub augmentation_ridge: Option<f64>,
}

/// Point-only synthetic `DiD` result with unit and time simplex weights.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SyntheticDidWire {
    /// Post-treatment difference-in-differences contrast.
    pub effect: f64,
    /// Pre-treatment root mean squared treated-versus-synthetic gap.
    pub pre_treatment_rmse: f64,
    /// Convex donor weights in stable unit order.
    pub donor_weights: Vec<(String, f64)>,
    /// Convex pre-period weights in calendar order.
    pub time_weights: Vec<(i64, f64)>,
    /// Number of donor units.
    pub n_donors: usize,
    /// Number of pre-treatment periods.
    pub n_pre_periods: usize,
    /// Number of post-treatment periods.
    pub n_post_periods: usize,
    /// Explicit no-interval uncertainty tag.
    pub uncertainty: String,
    /// Exact sharp-null p-value under declared uniform single-unit assignment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub randomization_p_value: Option<f64>,
    /// Constant additive effect tested by the exact Fisher p-value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub randomization_null_effect: Option<f64>,
    /// Absolute synthetic-DiD contrast for every candidate treated unit.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub randomization_statistics: Vec<(String, f64)>,
}

/// Retained local ratio result and its fixed-bandwidth normal interval.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LocalPolynomialRatioWire {
    /// Ratio of local outcome and treatment contrasts.
    pub effect: f64,
    /// Local outcome contrast.
    pub reduced_form: f64,
    /// Local treatment contrast.
    pub first_stage: f64,
    /// Prespecified running-variable cutoff.
    pub cutoff: f64,
    /// Prespecified triangular-kernel bandwidth.
    pub bandwidth: f64,
    /// Whether the contrast is a regression kink.
    pub kink: bool,
    /// Local observations below the cutoff.
    pub n_left: usize,
    /// Local observations on or above the cutoff.
    pub n_right: usize,
    /// HC0 delta-method standard error.
    pub standard_error: f64,
    /// Pointwise normal lower endpoint when the ratio interval is finite.
    #[serde(default)]
    pub ci_lower: Option<f64>,
    /// Pointwise normal upper endpoint when the ratio interval is finite.
    #[serde(default)]
    pub ci_upper: Option<f64>,
    /// Descriptive HC0 reduced-form standard error.
    pub reduced_form_standard_error: f64,
    /// Descriptive HC0 first-stage standard error.
    pub first_stage_standard_error: f64,
    /// Interval construction or legacy point-only uncertainty semantics.
    pub uncertainty: String,
}

/// Retained randomized ITT design metadata and design-aware variance.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RandomizedEffectWire {
    /// Intention-to-treat contrast.
    pub effect: f64,
    /// `itt` or `cace_late`.
    #[serde(default = "default_randomized_itt")]
    pub estimand: String,
    /// Outcome assignment effect for CACE/LATE.
    #[serde(default)]
    pub intention_to_treat_effect: Option<f64>,
    /// Treatment-receipt first stage for CACE/LATE.
    #[serde(default)]
    pub first_stage_effect: Option<f64>,
    /// Row-aligned observed receipt for CACE/LATE.
    #[serde(default)]
    pub received_treatment: Option<Vec<bool>>,
    /// Exact two-sided Fisher sharp-null p-value, when requested.
    #[serde(default)]
    pub randomization_p_value: Option<f64>,
    /// Number of complete-design allocations exhaustively enumerated.
    #[serde(default)]
    pub randomization_allocations: Option<u64>,
    /// Second-factor marginal effect under fixed four-cell randomization.
    #[serde(default)]
    pub second_factor_effect: Option<f64>,
    /// Interaction contrast: primary effect at B=1 minus primary effect at B=0.
    #[serde(default)]
    pub factorial_interaction: Option<f64>,
    /// Conservative second-factor marginal-effect variance.
    #[serde(default)]
    pub second_factor_variance: Option<f64>,
    /// Conservative interaction variance.
    #[serde(default)]
    pub factorial_interaction_variance: Option<f64>,
    /// Ordered multi-arm labels, HT means, variance contributions, and support.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub multi_arm_values: Vec<(String, f64, f64, usize)>,
    /// Design variance estimate or conservative bound, as labeled by uncertainty.
    pub variance: f64,
    /// Primary contrast SE when a pointwise interval is reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_error: Option<f64>,
    /// Pointwise 95% primary contrast interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_95: Option<[f64; 2]>,
    /// Pointwise 95% secondary factorial main-effect interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second_factor_interval_95: Option<[f64; 2]>,
    /// Pointwise 95% factorial interaction interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factorial_interaction_interval_95: Option<[f64; 2]>,
    /// Pointwise action-versus-reference intervals aligned to multi-arm labels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub multi_arm_intervals_95: Vec<Option<[f64; 2]>>,
    /// Assignment design name.
    pub assignment_design: String,
    /// Row-aligned assignment unit labels.
    pub assignment_units: Vec<String>,
    /// Row-aligned outcome unit labels.
    pub outcome_units: Vec<String>,
    /// Row-aligned block labels, when stratified.
    pub blocks: Vec<String>,
    /// Row-aligned period labels, when switchback.
    pub periods: Vec<String>,
    /// Control and treatment arm labels.
    pub treatment_arms: (String, String),
    /// Observed control assignment units or unit-periods.
    pub control_units: usize,
    /// Observed treatment assignment units or unit-periods.
    pub treatment_units: usize,
    /// Smallest declared assignment probability.
    pub minimum_assignment_probability: f64,
    /// Explicit no-interval uncertainty contract.
    pub uncertainty: String,
    /// Exact graphless license status; absent for off-axis results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphless_support_status: Option<String>,
}

fn default_randomized_itt() -> String { "itt".into() }

/// Randomized arm event-time curves and optional scalar pointwise intervals.
/// Simultaneous 95% treatment-minus-control band on the reported event grid.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SurvivalDifferenceBandWire {
    /// Event-time grid, including zero and the restriction horizon.
    pub times: Vec<f64>,
    /// Estimated curve difference at each time.
    pub difference: Vec<f64>,
    /// Simultaneous lower endpoints.
    pub lower: Vec<f64>,
    /// Simultaneous upper endpoints.
    pub upper: Vec<f64>,
    /// Subject-bootstrap draws satisfying support.
    pub replicates_ok: u32,
}

/// Retained randomized survival or competing-risk curve and uncertainty.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SurvivalWire {
    /// Shared time grid, including zero and the restriction horizon.
    pub times: Vec<f64>,
    /// Control-arm survival or target-cause cumulative incidence.
    pub control: Vec<f64>,
    /// Treated-arm survival or target-cause cumulative incidence.
    pub treated: Vec<f64>,
    /// Control restricted mean for a survival query.
    pub rmst_control: Option<f64>,
    /// Treated restricted mean for a survival query.
    pub rmst_treated: Option<f64>,
    /// Positive target cause for cumulative incidence.
    pub target_cause: Option<i64>,
    /// Shared restriction horizon.
    pub tau: f64,
    /// Smallest event risk set in either arm.
    pub minimum_event_risk_set: Option<usize>,
    /// Explicit uncertainty semantics for the scalar contrasts.
    pub uncertainty: String,
    /// Pointwise RMST treatment-minus-control interval when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rmst_difference_interval: Option<[f64; 2]>,
    /// Pointwise treatment-minus-control survival/CIF interval at tau.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub difference_at_tau_interval: Option<[f64; 2]>,
    /// Requested arm-stratified subject-bootstrap replicates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_replicates_requested: Option<u32>,
    /// Supported arm-stratified subject-bootstrap replicates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_replicates_ok: Option<u32>,
    /// Realized control and treated subject counts for graphless support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignment_counts: Option<[usize; 2]>,
    /// Exact graphless scalar-interval license; absent for off-axis results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphless_support_status: Option<String>,
    /// Caller-supplied fixed censoring function; absent for unweighted estimates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub censoring_survival_provenance: Option<String>,
    /// Simultaneous curve-difference band, distinct from scalar pointwise intervals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub difference_band: Option<SurvivalDifferenceBandWire>,
    /// Why an explicit bootstrap request did not produce a curve band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_unavailable_reason: Option<String>,
}

/// Subject-owned sequential inverse-probability regime value.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalRegimeWire {
    /// Named point method, defaulting to IPW for older artifacts.
    #[serde(default = "default_longitudinal_result_method")]
    pub method: String,
    /// Exact graphless sequential-DR license; absent for off-axis results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphless_support_status: Option<String>,
    /// Identity of materialized caller rule; source code is not replayable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_id: Option<String>,
    /// Stable caller-declared rule version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_version: Option<String>,
    /// Caller-declared source or provenance of the rule implementation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_provenance: Option<String>,
    /// Horvitz--Thompson regime value.
    pub value: f64,
    /// Independent-subject IPW score SE or subject-clustered MSM intercept SE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_standard_error: Option<f64>,
    /// Pointwise 95% randomized IPW value or MSM intercept interval, when supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_interval_95: Option<[f64; 2]>,
    /// Pointwise 95% subject-clustered intervals for additive MSM period effects.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub period_intervals_95: Vec<[f64; 2]>,
    /// Explicit reason an interval was withheld.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_reason: Option<String>,
    /// Effective sample size among matching observed histories.
    pub effective_sample_size: f64,
    /// Fraction of enrolled subjects with matching observed histories.
    pub matched_observed_fraction: f64,
    /// Maximum inverse-probability trajectory weight.
    pub maximum_weight: f64,
    /// Minimum prescribed action probability.
    pub minimum_action_probability: f64,
    /// Minimum remaining-uncensored probability.
    pub minimum_censoring_probability: f64,
    /// Explicitly point only.
    pub uncertainty: String,
    /// Known randomized probabilities or caller-declared excluded-fold predictions.
    pub probability_ownership: String,
    /// Additive MSM period coefficients; absent for regime-value methods.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub period_effects: Vec<f64>,
    /// Pointwise CR1 subject-clustered standard errors for period coefficients.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub standard_errors: Vec<f64>,
    /// Stabilizing treatment-one numerator probabilities for MSM.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stabilizing_numerator_probabilities: Vec<f64>,
    /// Observed subjects used by MSM; zero for regime-value methods.
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub observed_subjects: usize,
}

#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde skip_serializing_if requires an fn(&T) -> bool signature")]
fn is_zero_usize(value: &usize) -> bool { *value == 0 }

fn default_longitudinal_result_method() -> String {
    "ipw".into()
}

/// One independent-cluster 95% pointwise exposure-contrast interval.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InterferencePointwiseIntervalWire {
    /// Lower and upper endpoints.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Cluster-level Neyman standard error and Welch degrees of freedom.
    pub standard_error: f64,
    /// Welch--Satterthwaite degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Control/treated or low/high first-stage independent cluster counts.
    pub first_stage_arm_clusters: [usize; 2],
}

/// Support and pointwise inference attached only to randomized interference.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InterferenceInferenceWire {
    /// Construction identifier.
    pub method: String,
    /// Exact graphless interference license (`licensed`), when a published
    /// interval and cluster support match a licensed row. Absent otherwise, and
    /// on legacy artifacts written before the interference license existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphless_support_status: Option<String>,
    /// Pointwise interval when support passes.
    pub interval: Option<InterferencePointwiseIntervalWire>,
    /// Reason the otherwise valid point estimate has no interval.
    pub interval_unavailable_reason: Option<String>,
    /// Realized exposure support by units and independent clusters.
    pub from_exposed_units: usize,
    /// Units observed at the active exposure.
    pub to_exposed_units: usize,
    /// Clusters observed at the baseline exposure.
    pub from_exposed_clusters: usize,
    /// Clusters observed at the active exposure.
    pub to_exposed_clusters: usize,
}

/// Composite result body. Every scientific axis is independently optional.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnalysisResultWire {
    /// Original query.
    pub query: CausalQueryWire,
    /// Structural identification and derivation.
    pub identification: IdentificationResultWire,
    /// Namespace for the primary identification; absent means the base schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identification_variables: Option<Vec<crate::HorizonAdjustmentNodeWire>>,
    /// Full horizon-specific certificates and namespaces, when prepared temporally.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub temporal_identification: Vec<TemporalIdentificationWire>,
    /// Scalar estimate when one exists; function-valued results have no scalar placeholder.
    pub estimate: Option<f64>,
    /// Policy-value answer, never encoded as an ATE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_value: Option<PolicyValueWire>,
    /// Conditional continuous-dose response; never encoded as policy value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuous_dose_response: Option<ContinuousDoseResponseWire>,
    /// Balanced two-period panel `DiD` metadata and cluster uncertainty semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel_did: Option<PanelDidWire>,
    /// Synthetic-control result with donor and placebo diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthetic_control: Option<SyntheticControlWire>,
    /// Synthetic `DiD` point result and fitted simplex weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthetic_did: Option<SyntheticDidWire>,
    /// Fixed-window fuzzy RD or regression-kink ratio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_polynomial_ratio: Option<LocalPolynomialRatioWire>,
    /// Retained randomized experiment design and variance semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub randomized_effect: Option<RandomizedEffectWire>,
    /// Randomized survival or competing-risk curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub survival: Option<SurvivalWire>,
    /// Subject-owned prespecified longitudinal regime point value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub longitudinal_regime: Option<LongitudinalRegimeWire>,
    /// Independent-cluster exposure support and pointwise inference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interference_inference: Option<InterferenceInferenceWire>,
    /// Full atom result for an interventional-distribution query. Absent on older
    /// artifacts; such artifacts remain readable but cannot verify as a checked
    /// distribution execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interventional_distribution: Option<InterventionalDistributionWire>,
    /// Scalar standard error when justified. Absent when the published interval
    /// is an Anderson–Rubin set (`interval_lower` / `interval_upper`) or when
    /// no scalar SE is licensed.
    pub standard_error: Option<f64>,
    /// Lower endpoint of a published Anderson–Rubin set. May be infinite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_lower: Option<f64>,
    /// Upper endpoint of a published Anderson–Rubin set. May be infinite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_upper: Option<f64>,
    /// Estimation assumptions.
    pub assumptions: Vec<AssumptionRecordWire>,
    /// Execution and scientific diagnostics.
    pub diagnostics: Vec<DiagnosticWire>,
    /// Validation reports.
    pub refutations: Vec<RefutationReportWire>,
    /// Response axis.
    pub response: Option<CausalResponseWire>,
    /// Nested canonical posterior artifact bytes, including draws when requested.
    pub posterior_artifact: Option<Vec<u8>>,
    /// Horizon-indexed mediation axis.
    pub mediation_grid: Option<TemporalMediationGridWire>,
    /// Structural-mixture response axis.
    pub structural_response: Option<StructuralResponseMixtureWire>,
    /// Per-unit counterfactual effects, when the execution computed them.
    ///
    /// Absent on every other result, so their bodies (and digests) are
    /// unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_effects: Option<UnitEffectsWire>,
    /// Complete-case-aligned CATE point predictions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cate: Option<Vec<f64>>,
    /// Portable fitted effect, bound into the result body identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitted_effect: Option<antecedent_estimate::FittedEffect>,
    /// Licensed pointwise CATE standard errors, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cate_se: Option<Vec<f64>>,
    /// Forest leaf-dispersion diagnostic per row. Not a standard error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cate_leaf_dispersion: Option<Vec<f64>>,
    /// Held-out outcome R².
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_oof_r2: Option<f64>,
    /// Held-out treatment probability log loss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment_oof_logloss: Option<f64>,
    /// Number of nuisance cross-fitting folds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossfit_folds: Option<usize>,
    /// Master seed used for nuisance cross-fitting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossfit_seed: Option<u64>,
    /// Fitted learner (spec, implementation, version), in fit order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub learner_provenance: Vec<(String, String, String)>,
}

/// Per-unit counterfactual effects an execution reported.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UnitEffectsWire {
    /// One effect per unit, in row order.
    pub effects: Vec<f64>,
    /// Every unit carries the same effect by construction of the fitted mechanism.
    pub homogeneous: bool,
    /// Per-unit interval bounds, when the execution formed them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervals: Option<UnitEffectIntervalsWire>,
    /// Per-unit extrapolation flags aligned with [`Self::effects`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extrapolative: Option<Vec<bool>>,
}

/// Level-tagged per-unit interval bounds aligned with [`UnitEffectsWire::effects`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UnitEffectIntervalsWire {
    /// Lower bound per unit.
    pub lower: Vec<f64>,
    /// Upper bound per unit.
    pub upper: Vec<f64>,
    /// Nominal level the bounds were read at.
    pub level: f64,
    /// Construction id.
    pub method: String,
}

/// Composite artifact header.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnalysisResultHeader {
    /// Variable names in raw-id order.
    pub variable_names: Vec<String>,
}

/// Encode a composite result artifact without a contract section.
///
/// # Errors
///
/// Returns an error when CBOR/container encoding fails.
pub fn encode_analysis_result_artifact(
    result: &AnalysisResultWire,
    variable_names: Vec<String>,
    artifact_id: &str,
) -> Result<EncodedArtifact, IoError> {
    encode_analysis_result_artifact_with_contract(result, variable_names, artifact_id, None)
}

/// Encode a composite result, optionally attaching [`crate::CONTRACT_SECTION`].
///
/// The section is additive. Old readers ignore it; this encoder does not raise
/// `minimum_reader_version`. A present contract is validated and bound to the
/// body query and header names before write.
///
/// # Errors
///
/// Returns an error when validation, contract binding, or CBOR encoding fails.
pub fn encode_analysis_result_artifact_with_contract(
    result: &AnalysisResultWire,
    variable_names: Vec<String>,
    artifact_id: &str,
    contract: Option<&crate::AnalysisResultContractWire>,
) -> Result<EncodedArtifact, IoError> {
    validate_result(result, &variable_names, false)?;
    if let Some(contract) = contract {
        crate::validate_contract_section(contract)?;
        let header = AnalysisResultHeader { variable_names: variable_names.clone() };
        let unresolved =
            crate::contract_section::verify_contract_for_encoding(&header, result, contract);
        if !unresolved.is_empty() {
            return Err(IoError::Convert(format!(
                "contract does not verify against analysis_result body: {}",
                unresolved.join(",")
            )));
        }
    }
    let header = to_cbor(&AnalysisResultHeader { variable_names })?;
    let body = to_cbor(result)?;
    let (header_descriptor, header_section) = pack_section_shared(
        HEADER_SECTION,
        "application/cbor",
        header.into(),
        CompressPolicy::Auto,
    );
    let (body_descriptor, body_section) =
        pack_section_shared(BODY_SECTION, "application/cbor", body.into(), CompressPolicy::Auto);
    let mut sections_desc = vec![header_descriptor, body_descriptor];
    let mut sections = vec![header_section, body_section];
    if let Some(contract) = contract {
        let payload = to_cbor(contract)?;
        let (descriptor, section) = pack_section_shared(
            crate::CONTRACT_SECTION,
            "application/cbor",
            payload.into(),
            CompressPolicy::Auto,
        );
        sections_desc.push(descriptor);
        sections.push(section);
    }
    Ok(EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: STABLE_FORMAT,
            minimum_reader_version: STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(antecedent_core::VERSION)?,
            artifact_id: artifact_id.into(),
            sections: sections_desc,
            provenance: ProvenanceWire { note: "composite analysis result".into() },
        },
        sections,
    })
}

/// Decode a composite result artifact.
///
/// # Errors
///
/// Returns an error for the wrong artifact kind, missing sections, or invalid CBOR.
pub fn decode_analysis_result_artifact(
    bytes: &[u8],
) -> Result<(EncodedArtifact, AnalysisResultHeader, AnalysisResultWire), IoError> {
    let artifact = read_and_migrate(bytes)?;
    if artifact.manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into()) {
        return Err(IoError::Convert("expected an analysis_result artifact".into()));
    }
    let header = artifact
        .sections
        .iter()
        .find(|section| section.id == HEADER_SECTION)
        .ok_or_else(|| IoError::Convert(format!("missing section `{HEADER_SECTION}`")))?;
    let body = artifact
        .sections
        .iter()
        .find(|section| section.id == BODY_SECTION)
        .ok_or_else(|| IoError::Convert(format!("missing section `{BODY_SECTION}`")))?;
    let decoded_header: AnalysisResultHeader = from_cbor(&header.data)?;
    let decoded_body: AnalysisResultWire = from_cbor(&body.data)?;
    validate_result(&decoded_body, &decoded_header.variable_names, true)?;
    Ok((artifact, decoded_header, decoded_body))
}

// allow(too_many_lines): single dispatch over every result-variant validator; splitting hides the total contract
#[allow(clippy::too_many_lines)]
fn validate_result(
    result: &AnalysisResultWire,
    variable_names: &[String],
    allow_legacy_graphless_missing: bool,
) -> Result<(), IoError> {
    use crate::causal_artifact::{
        validate_query_ids, validate_response_result, validate_variable_names,
    };
    if let Some(model) = &result.fitted_effect {
        // Unsupported optional prediction codecs do not erase the scientific claim.
        // The sealed body still binds their bytes; prediction loading rejects them.
        if model.version == 1 && model.predictor.version == 1 {
            model.validate().map_err(|e| IoError::Convert(e.to_string()))?;
        }
        if model.features.iter().any(|v| *v as usize >= variable_names.len()) {
            return Err(IoError::Convert("fitted effect names an unknown feature".into()));
        }
    }
    validate_variable_names(variable_names)?;
    validate_query_ids(&result.query, variable_names.len())?;
    if let Some(inference) = &result.interference_inference {
        let crate::CausalQueryWire::Interference(query) = &result.query else {
            return Err(IoError::Convert("interference inference is attached to a different query".into()));
        };
        let expected = match &query.assignment {
            crate::AssignmentDesignWire::ClusterRandomization { .. } => "cluster_total_neyman_welch",
            crate::AssignmentDesignWire::TwoStageSaturation { .. } => "saturation_cluster_neyman_welch",
            crate::AssignmentDesignWire::ObservedExposure { .. } => "observational_known_exposure_cluster_t",
            _ => return Err(IoError::Convert("independent-cluster inference requires a cluster or supplied-exposure interference design".into())),
        };
        let minimum_first_stage_clusters = if matches!(&query.assignment,
            crate::AssignmentDesignWire::TwoStageSaturation { .. }) { 24 } else { 8 };
        if inference.method != expected || inference.from_exposed_units == 0
            || inference.to_exposed_units == 0 || inference.from_exposed_clusters == 0
            || inference.to_exposed_clusters == 0
            || (inference.interval.is_some() == inference.interval_unavailable_reason.is_some())
        {
            return Err(IoError::Convert("interference inference method, support, or interval status is inconsistent".into()));
        }
        if let Some(interval) = &inference.interval {
            if let crate::AssignmentDesignWire::ObservedExposure { provenance, .. } = &query.assignment {
                if provenance != "known"
                    || inference.from_exposed_clusters < 8 || inference.to_exposed_clusters < 8
                    || interval.degrees_of_freedom < 29.0
                {
                    return Err(IoError::Convert("observational exposure interval requires known probabilities and independent-cluster support".into()));
                }
            }
            if !interval.lower.is_finite() || !interval.upper.is_finite()
                || interval.lower >= interval.upper
                || !interval.standard_error.is_finite() || interval.standard_error <= 0.0
                || !interval.degrees_of_freedom.is_finite() || interval.degrees_of_freedom <= 0.0
                || interval.first_stage_arm_clusters.iter().any(|&count| count < minimum_first_stage_clusters)
                || inference.from_exposed_clusters < 8 || inference.to_exposed_clusters < 8
                || result.estimate.is_none_or(|effect| effect < interval.lower || effect > interval.upper)
            {
                return Err(IoError::Convert("interference pointwise interval violates its support or scalar claim".into()));
            }
        }
        // A published, support-passing interval on one of the three off-axis
        // cluster designs is exactly the licensed condition; every invalid or
        // unsupported interval was already refused above, so recomputing the
        // license from the published interval catches a forged status.
        let graphless_licensed = inference.interval.is_some();
        if inference.graphless_support_status.as_deref() != graphless_licensed.then_some("licensed")
            && !(allow_legacy_graphless_missing && inference.graphless_support_status.is_none())
        {
            return Err(IoError::Convert(
                "interference graphless license does not match the published interval support".into(),
            ));
        }
    }
    if matches!(&result.query, crate::CausalQueryWire::RandomizedEffect(query)
        if matches!(query.design, crate::RandomizationDesignWire::Switchback))
        && result.randomized_effect.is_none()
    {
        return Err(IoError::Convert(
            "switchback artifact is missing its randomized result section".into(),
        ));
    }
    if let Some(randomized) = &result.randomized_effect {
        let crate::CausalQueryWire::RandomizedEffect(query) = &result.query else {
            return Err(IoError::Convert(
                "randomized result section is attached to a different query".into(),
            ));
        };
        let (design, point_uncertainty, interval_uncertainty) = match &query.design {
            crate::RandomizationDesignWire::Bernoulli
                if query.estimand == crate::RandomizedEstimandWire::CaceLate =>
            {
                ("bernoulli", "bernoulli_wald_cace_influence_variance_no_interval",
                    Some("bernoulli_wald_cace_influence_normal_interval"))
            }
            crate::RandomizationDesignWire::Bernoulli
                if query.estimand == crate::RandomizedEstimandWire::TreatmentOnTreated =>
            {
                ("bernoulli", "bernoulli_one_sided_tot_influence_variance_no_interval",
                    Some("bernoulli_one_sided_tot_influence_normal_interval"))
            }
            crate::RandomizationDesignWire::Bernoulli if query.fixed_cuped.is_some() => {
                ("bernoulli", "bernoulli_fixed_cuped_ht_conservative_variance_no_interval", Some("bernoulli_fixed_cuped_ht_score_normal_interval"))
            }
            crate::RandomizationDesignWire::Bernoulli if !query.ancova_covariates.is_empty() => {
                ("bernoulli", "bernoulli_ancova_hc0_variance_no_interval", Some("bernoulli_ancova_hc0_normal_interval"))
            }
            crate::RandomizationDesignWire::Bernoulli => {
                ("bernoulli", "bernoulli_ht_design_variance_no_interval", Some("bernoulli_ht_score_normal_interval"))
            }
            crate::RandomizationDesignWire::Complete { .. } => {
                ("complete", "complete_neyman_variance_upper_bound_no_interval", Some("complete_neyman_normal_interval"))
            }
            crate::RandomizationDesignWire::Cluster { .. } => {
                ("cluster", "cluster_neyman_variance_upper_bound_no_interval", Some("cluster_neyman_normal_interval"))
            }
            crate::RandomizationDesignWire::Stratified => {
                ("stratified", "stratified_neyman_variance_upper_bound_no_interval", Some("stratified_neyman_normal_interval"))
            }
            crate::RandomizationDesignWire::Factorial2x2 => {
                ("factorial_2x2", "factorial_cell_neyman_variance_upper_bound_no_interval", Some("factorial_cell_neyman_pointwise_normal_intervals"))
            }
            crate::RandomizationDesignWire::Switchback => {
                ("switchback", "switchback_independent_sequence_sandwich_variance_no_interval", Some("switchback_independent_sequence_student_interval"))
            }
            crate::RandomizationDesignWire::MultiArm => {
                ("multi_arm", "multi_arm_covariance_free_variance_bound_no_interval", Some("multi_arm_ht_score_pointwise_normal_intervals"))
            }
        };
        let (control, treated) =
            if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
                (
                    query.multi_arm_assignment.iter().filter(|&&arm| arm == 0).count(),
                    query.multi_arm_assignment.iter().filter(|&&arm| arm == 1).count(),
                )
            } else if matches!(query.design, crate::RandomizationDesignWire::Cluster { .. }) {
                let mut clusters = std::collections::BTreeMap::new();
                for (unit, assignment) in
                    query.assignment_units.iter().zip(&query.realized_assignment)
                {
                    clusters.insert(unit, *assignment);
                }
                (
                    clusters.values().filter(|assigned| !**assigned).count(),
                    clusters.values().filter(|assigned| **assigned).count(),
                )
            } else {
                (
                    query.realized_assignment.iter().filter(|assigned| !**assigned).count(),
                    query.realized_assignment.iter().filter(|assigned| **assigned).count(),
                )
            };
        let minimum_probability = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
            query.multi_arm_probabilities.iter().flat_map(|row| row.iter().copied()).fold(f64::INFINITY, f64::min)
        } else {
            query.assignment_probabilities.iter().copied()
                .map(|p| if matches!(query.design, crate::RandomizationDesignWire::Switchback) {
                    p.min(1.0 - p)
                } else { p })
                .fold(f64::INFINITY, f64::min)
        };
        let expected_allocations = if query.exact_randomization_test {
            let n = query.realized_assignment.len() as u64;
            let k = treated as u64;
            Some((0..k).fold(1_u64, |count, i| count * (n - i) / (i + 1)))
        } else { None };
        let interval_supported = match &query.design {
            crate::RandomizationDesignWire::Bernoulli =>
                matches!(query.estimand, crate::RandomizedEstimandWire::Itt
                    | crate::RandomizedEstimandWire::CaceLate
                    | crate::RandomizedEstimandWire::TreatmentOnTreated)
                    && query.ancova_covariates.len() <= 2
                    && query.realized_assignment.len() >= 400
                    && control >= 30 && treated >= 30
                    && query.assignment_probabilities.iter().all(|p| *p + 1e-12 >= 0.2 && *p <= 0.8 + 1e-12)
                    && (query.ancova_covariates.is_empty()
                        || query.assignment_probabilities.iter().all(|p|
                            (*p - query.assignment_probabilities[0]).abs() <= 1e-12)),
            crate::RandomizationDesignWire::Complete { .. }
            | crate::RandomizationDesignWire::Cluster { .. } => control >= 30 && treated >= 30,
            crate::RandomizationDesignWire::Stratified => {
                let mut blocks = std::collections::BTreeMap::<&str, (usize, usize)>::new();
                for (block, assigned) in query.blocks.iter().zip(&query.realized_assignment) {
                    let counts = blocks.entry(block).or_default();
                    if *assigned { counts.1 += 1; } else { counts.0 += 1; }
                }
                blocks.len() >= 4 && control >= 60 && treated >= 60
                    && blocks.values().all(|(c, t)| *c >= 15 && *t >= 15)
            }
            crate::RandomizationDesignWire::Factorial2x2 => query.factorial_cell_counts
                .is_some_and(|counts| counts.iter().all(|count| *count >= 30)),
            crate::RandomizationDesignWire::MultiArm =>
                query.multi_arm_assignment.len() >= 400
                    && query.multi_arm_probabilities.iter().all(|row|
                        row.iter().all(|p| *p + 1e-12 >= 0.2))
                    && (0..query.multi_arm_labels.len()).all(|arm|
                        query.multi_arm_assignment.iter().filter(|&&assigned| assigned == arm).count() >= 30),
            crate::RandomizationDesignWire::Switchback => {
                let mut sizes = std::collections::BTreeMap::<&str, usize>::new();
                for sequence in &query.assignment_units {
                    *sizes.entry(sequence.as_str()).or_default() += 1;
                }
                sizes.len() >= 30 && control >= 30 && treated >= 30
                    && sizes.values().next().is_some_and(|first| *first > 0
                        && sizes.values().all(|size| size == first))
                    && minimum_probability + 1e-12 >= 0.2
            },
        };
        let has_interval = randomized.interval_95.is_some();
        let graphless_method = match query.design {
            crate::RandomizationDesignWire::Bernoulli
                if query.estimand == crate::RandomizedEstimandWire::TreatmentOnTreated =>
                Some(("complier_effect", "bernoulli_one_sided", "wald_ratio_influence", "unit", "pointwise_95_normal_interval")),
            crate::RandomizationDesignWire::Bernoulli
                if query.estimand == crate::RandomizedEstimandWire::CaceLate =>
                Some(("complier_effect", "bernoulli", "wald_ratio_influence", "unit", "pointwise_95_normal_interval")),
            crate::RandomizationDesignWire::Bernoulli if query.fixed_cuped.is_some() =>
                Some(("randomized_effect", "bernoulli", "fixed_cuped_ht_score", "unit", "pointwise_95_normal_interval")),
            crate::RandomizationDesignWire::Bernoulli if !query.ancova_covariates.is_empty() =>
                Some(("randomized_effect", "bernoulli", "ancova_hc0", "unit", "pointwise_95_normal_interval")),
            crate::RandomizationDesignWire::Bernoulli => Some(("randomized_effect", "bernoulli", "independent_action_ht_score", "unit", "pointwise_95_normal_interval")),
            crate::RandomizationDesignWire::Complete { .. } => Some(("randomized_effect", "complete", "neyman_difference_in_means", "unit", "pointwise_95_normal_interval")),
            crate::RandomizationDesignWire::Cluster { .. } => Some(("randomized_effect", "cluster", "neyman_unit_weighted_cluster_totals", "cluster", "pointwise_95_normal_interval")),
            crate::RandomizationDesignWire::Stratified => Some(("randomized_effect", "stratified", "blocked_neyman_difference_in_means", "unit", "pointwise_95_normal_interval")),
            crate::RandomizationDesignWire::Factorial2x2 => Some(("randomized_effect", "factorial_2x2", "fixed_cell_neyman_contrasts", "unit", "three_pointwise_95_normal_intervals")),
            crate::RandomizationDesignWire::MultiArm => Some(("randomized_effect", "multi_arm", "independent_action_ht_scores", "unit", "all_action_pointwise_95_normal_intervals")),
            crate::RandomizationDesignWire::Switchback =>
                Some(("randomized_effect", "switchback", "independent_sequence_ht_score", "sequence", "pointwise_95_student_interval")),
        };
        let mut block_counts = std::collections::BTreeMap::<&str, (usize, usize)>::new();
        let support_blocks = if matches!(query.design, crate::RandomizationDesignWire::Switchback) {
            &query.assignment_units
        } else { &query.blocks };
        for (block, assigned) in support_blocks.iter().zip(&query.realized_assignment) {
            let counts = block_counts.entry(block.as_str()).or_default();
            if *assigned { counts.1 += 1; } else { counts.0 += 1; }
        }
        let min_block_arm = block_counts.values().flat_map(|(c, t)| [*c, *t]).min().unwrap_or(0);
        let balanced_sequences = matches!(query.design, crate::RandomizationDesignWire::Switchback)
            && block_counts.values().next().is_some_and(|first| {
                let size = first.0 + first.1;
                size > 0 && block_counts.values().all(|counts| counts.0 + counts.1 == size)
            });
        let min_factorial_cell = query.factorial_cell_counts
            .map_or(0, |counts| *counts.iter().min().unwrap_or(&0));
        let min_action_rows = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
            (0..query.multi_arm_labels.len()).map(|arm|
                query.multi_arm_assignment.iter().filter(|&&assigned| assigned == arm).count())
                .min().unwrap_or(0)
        } else { control.min(treated) };
        let min_probability = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
            query.multi_arm_probabilities.iter().flat_map(|row| row.iter().copied())
                .fold(f64::INFINITY, f64::min)
        } else {
            query.assignment_probabilities.iter().copied().map(|p| p.min(1.0 - p))
                .fold(f64::INFINITY, f64::min)
        };
        let reported_intervals = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
            randomized.multi_arm_intervals_95.iter().filter(|interval| interval.is_some()).count()
        } else {
            usize::from(has_interval)
                + usize::from(randomized.second_factor_interval_95.is_some())
                + usize::from(randomized.factorial_interaction_interval_95.is_some())
        };
        let all_reported_intervals = match query.design {
            crate::RandomizationDesignWire::Factorial2x2 => reported_intervals == 3,
            crate::RandomizationDesignWire::MultiArm =>
                randomized.multi_arm_intervals_95.len() == query.multi_arm_labels.len()
                    && reported_intervals + 1 == query.multi_arm_labels.len(),
            _ => has_interval,
        };
        let graphless_licensed = graphless_method.is_some_and(|(family, design_key, method, assignment_unit, claim)| {
            graphless_data::LICENSES.iter().any(|row| {
                row.family == family
                    && row.design == design_key
                    && row.method == method
                    && row.inference_claim == claim
                    && row.assignment_unit == assignment_unit
                    && control >= row.min_assignment_units_per_arm
                    && treated >= row.min_assignment_units_per_arm
                    && query.realized_assignment.len() >= row.min_rows
                    && block_counts.len() >= row.min_blocks
                    && min_block_arm >= row.min_block_arm
                    && min_factorial_cell >= row.min_factorial_cell
                    && min_action_rows >= row.min_action_rows
                    && min_probability + 1e-12 >= row.min_probability
                    && reported_intervals >= row.min_reported_intervals
                    && (!row.requires_balanced_sequences || balanced_sequences)
                    && (row.max_covariates == 0 || query.ancova_covariates.len() <= row.max_covariates)
                    && (!row.all_reported_intervals || all_reported_intervals)
                    && has_interval
            })
        });
        let expected_graphless_status = graphless_licensed.then_some("licensed");
        let expected_uncertainty = if has_interval {
            interval_uncertainty.unwrap_or(point_uncertainty)
        } else { point_uncertainty };
        let z95 = if matches!(query.design, crate::RandomizationDesignWire::Switchback) {
            antecedent_stats::student_t_ppf(0.975, (block_counts.len() - 1) as f64)
        } else { 1.959_963_984_540_054 };
        let primary_interval_valid = match (randomized.standard_error, randomized.interval_95) {
            (None, None) => result.standard_error.is_none(),
            (Some(se), Some([lower, upper])) => {
                interval_supported && interval_uncertainty.is_some()
                    && se.is_finite() && se > 0.0
                    && result.standard_error.is_some_and(|top| (top - se).abs() <= 1e-10)
                    && lower.is_finite() && upper.is_finite()
                    && (lower - (randomized.effect - z95 * se)).abs() <= 1e-8
                    && (upper - (randomized.effect + z95 * se)).abs() <= 1e-8
                    && (matches!(query.design, crate::RandomizationDesignWire::Bernoulli | crate::RandomizationDesignWire::MultiArm)
                        || (se * se - randomized.variance).abs() <= 1e-8)
            }
            _ => false,
        };
        let extra_intervals_valid = if matches!(query.design, crate::RandomizationDesignWire::Factorial2x2) {
            match (randomized.second_factor_effect, randomized.second_factor_variance,
                randomized.second_factor_interval_95, randomized.factorial_interaction,
                randomized.factorial_interaction_variance, randomized.factorial_interaction_interval_95) {
                (Some(second), Some(second_var), Some([sl, su]), Some(interaction), Some(interaction_var), Some([il, iu])) if has_interval =>
                    (sl - (second - z95 * second_var.sqrt())).abs() <= 1e-8
                    && (su - (second + z95 * second_var.sqrt())).abs() <= 1e-8
                    && (il - (interaction - z95 * interaction_var.sqrt())).abs() <= 1e-8
                    && (iu - (interaction + z95 * interaction_var.sqrt())).abs() <= 1e-8,
                (_, _, None, _, _, None) if !has_interval => true,
                _ => false,
            }
        } else {
            randomized.second_factor_interval_95.is_none()
                && randomized.factorial_interaction_interval_95.is_none()
        };
        let multi_arm_intervals_valid = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
            if has_interval {
                randomized.multi_arm_intervals_95.len() == query.multi_arm_labels.len()
                    && randomized.multi_arm_values.len() == query.multi_arm_labels.len()
                    && randomized.multi_arm_intervals_95.len() >= 2
                    && randomized.multi_arm_intervals_95[0].is_none()
                    && randomized.multi_arm_intervals_95.iter().enumerate().skip(1).all(|(arm, interval)|
                        interval.is_some_and(|[lower, upper]| {
                            let contrast = randomized.multi_arm_values[arm].1 - randomized.multi_arm_values[0].1;
                            lower.is_finite() && upper.is_finite() && lower < upper
                                && ((lower + upper) / 2.0 - contrast).abs() <= 1e-8
                        }))
                    && randomized.multi_arm_intervals_95[1] == randomized.interval_95
            } else { randomized.multi_arm_intervals_95.is_empty()
                || randomized.multi_arm_intervals_95.iter().all(Option::is_none) }
        } else { randomized.multi_arm_intervals_95.is_empty() };
        if result.estimate != Some(randomized.effect)
            || !primary_interval_valid
            || result.interval_lower.is_some()
            || result.interval_upper.is_some()
            || !randomized.effect.is_finite()
            || !randomized.variance.is_finite()
            || randomized.variance < 0.0
            || randomized.assignment_design != design
            || (randomized.graphless_support_status.as_deref() != expected_graphless_status
                && !(allow_legacy_graphless_missing
                    && randomized.graphless_support_status.is_none()))
            || randomized.uncertainty != expected_uncertainty
            || !extra_intervals_valid
            || !multi_arm_intervals_valid
            || randomized.estimand
                != if query.estimand == crate::RandomizedEstimandWire::CaceLate {
                    "cace_late"
                } else if query.estimand == crate::RandomizedEstimandWire::TreatmentOnTreated {
                    "treatment_on_treated"
                } else if matches!(query.design, crate::RandomizationDesignWire::Factorial2x2) {
                    "factorial_primary_main_effect"
                } else if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
                    "multi_arm_itt"
                } else {
                    "itt"
                }
            || randomized.received_treatment != query.received_treatment
            || randomized.randomization_allocations != expected_allocations
            || (matches!(query.design, crate::RandomizationDesignWire::Factorial2x2)
                != (randomized.second_factor_effect.is_some()
                    && randomized.factorial_interaction.is_some()
                    && randomized.second_factor_variance.is_some()
                    && randomized.factorial_interaction_variance.is_some()))
            || randomized.second_factor_effect.is_some_and(|value| !value.is_finite())
            || randomized.factorial_interaction.is_some_and(|value| !value.is_finite())
            || randomized.second_factor_variance.is_some_and(|value| !value.is_finite() || value < 0.0 || (value - randomized.variance).abs() > 1e-12)
            || randomized.factorial_interaction_variance.is_some_and(|value| !value.is_finite() || value < 0.0 || (value - 4.0 * randomized.variance).abs() > 1e-12)
            || (matches!(query.design, crate::RandomizationDesignWire::MultiArm)
                == randomized.multi_arm_values.is_empty())
            || (matches!(query.design, crate::RandomizationDesignWire::MultiArm) && (
                randomized.multi_arm_values.len() != query.multi_arm_labels.len()
                || randomized.multi_arm_values.iter().enumerate().any(|(i, (label, value, variance, support))|
                    label != &query.multi_arm_labels[i] || !value.is_finite() || !variance.is_finite() || *variance < 0.0
                    || *support != query.multi_arm_assignment.iter().filter(|&&arm| arm == i).count())
                || randomized.multi_arm_values.get(1).is_none_or(|(_, value, variance, _)|
                    (randomized.effect - (value - randomized.multi_arm_values[0].1)).abs() > 1e-10
                    || (randomized.variance - 2.0 * (variance + randomized.multi_arm_values[0].2)).abs() > 1e-10)
            ))
            || (query.exact_randomization_test
                && randomized.randomization_p_value.is_none_or(|p| !p.is_finite()
                    || p < 1.0 / expected_allocations.unwrap() as f64 || p > 1.0))
            || (!query.exact_randomization_test && randomized.randomization_p_value.is_some())
            || (query.estimand != crate::RandomizedEstimandWire::Itt
                && (randomized.intention_to_treat_effect.is_none_or(|value| !value.is_finite())
                    || randomized
                        .first_stage_effect
                        .is_none_or(|value| !value.is_finite() || value <= f64::EPSILON)
                    || (randomized.effect
                        - randomized.intention_to_treat_effect.unwrap()
                            / randomized.first_stage_effect.unwrap())
                    .abs()
                        > 1e-10))
            || (query.estimand == crate::RandomizedEstimandWire::Itt
                && (randomized.intention_to_treat_effect.is_some()
                    || randomized.first_stage_effect.is_some()))
            || randomized.assignment_units != query.assignment_units
            || randomized.outcome_units != query.outcome_units
            || randomized.blocks != query.blocks
            || randomized.periods != query.periods
            || randomized.treatment_arms != query.treatment_arms
            || randomized.control_units != control
            || randomized.treatment_units != treated
            || (randomized.minimum_assignment_probability - minimum_probability).abs() > 1e-12
        {
            return Err(IoError::Convert("invalid randomized payload or fabricated interval".into()));
        }
    }
    if matches!(result.query, crate::CausalQueryWire::PanelDid(_)) && result.panel_did.is_none() {
        return Err(IoError::Convert(
            "panel DiD artifact is missing its design-specific result section".into(),
        ));
    }
    if let crate::CausalQueryWire::SyntheticControl(query) = &result.query {
        if query.difference_in_differences {
            if result.synthetic_did.is_none() || result.synthetic_control.is_some() {
                return Err(IoError::Convert("synthetic DiD artifact requires its distinct weight result section".into()));
            }
        } else if result.synthetic_control.is_none() || result.synthetic_did.is_some() {
            return Err(IoError::Convert("synthetic-control artifact requires its donor-support result section".into()));
        }
    }
    if matches!(result.query, crate::CausalQueryWire::LocalPolynomialRatio(_))
        && result.local_polynomial_ratio.is_none()
    {
        return Err(IoError::Convert("local ratio artifact is missing its design-specific result section".into()));
    }
    if let Some(fit) = &result.local_polynomial_ratio {
        let crate::CausalQueryWire::LocalPolynomialRatio(query) = &result.query else {
            return Err(IoError::Convert("local ratio result requires matching query".into()));
        };
        let legacy_point_only = fit.uncertainty == "rbc_point_with_unvalidated_hc0_standard_error_no_interval";
        let interval_available = !legacy_point_only && fit.standard_error > 0.0;
        let z = antecedent_stats::normal_ppf(0.975);
        let lower = fit.effect - z * fit.standard_error;
        let upper = fit.effect + z * fit.standard_error;
        let tolerance = 1e-10 * (1.0 + fit.effect.abs() + (z * fit.standard_error).abs());
        let bounds_match = |bounds: Option<f64>, expected: f64| {
            if interval_available {
                bounds.is_some_and(|value| value.is_finite() && (value - expected).abs() <= tolerance)
            } else { bounds.is_none() }
        };
        #[allow(clippy::float_cmp, reason = "wire fields must equal the frozen query bytes exactly; any drift is a real mismatch")]
        if fit.cutoff != query.cutoff
            || fit.bandwidth != query.bandwidth
            || fit.kink != query.kink
            || result.estimate != Some(fit.effect)
            || (if interval_available { result.standard_error != Some(fit.standard_error) } else { result.standard_error.is_some() })
            || !bounds_match(fit.ci_lower, lower)
            || !bounds_match(fit.ci_upper, upper)
            || result.interval_lower.is_some()
            || result.interval_upper.is_some()
            || !fit.effect.is_finite()
            || !fit.reduced_form.is_finite()
            || !fit.first_stage.is_finite()
            || fit.first_stage.abs() < 1e-12
            || !fit.standard_error.is_finite()
            || fit.standard_error < 0.0
            || !fit.reduced_form_standard_error.is_finite()
            || fit.reduced_form_standard_error < 0.0
            || !fit.first_stage_standard_error.is_finite()
            || fit.first_stage_standard_error < 0.0
            || fit.n_left < 3
            || fit.n_right < 3
            || (!legacy_point_only && fit.uncertainty != "rbc_hc0_delta_normal_fixed_bandwidth")
        {
            return Err(IoError::Convert("invalid local ratio support or fabricated interval".into()));
        }
    }
    if matches!(result.query, crate::CausalQueryWire::Survival(_)) && result.survival.is_none() {
        return Err(IoError::Convert(
            "survival artifact is missing its curve result section".into(),
        ));
    }
    if matches!(result.query, crate::CausalQueryWire::LongitudinalRegime(_))
        && result.longitudinal_regime.is_none()
    {
        return Err(IoError::Convert(
            "longitudinal regime artifact is missing its value section".into(),
        ));
    }
    if let Some(regime) = &result.longitudinal_regime {
        let crate::CausalQueryWire::LongitudinalRegime(query) = &result.query else {
            return Err(IoError::Convert(
                "longitudinal regime result is attached to a different query".into(),
            ));
        };
        let eligible_ipw = query.method == "ipw"
            && query.probabilities_known_by_design
            && query.subject_ids.len() >= 500
            && regime.matched_observed_fraction * query.subject_ids.len() as f64 >= 50.0
            && regime.effective_sample_size >= 50.0
            && regime.value_standard_error.is_some_and(|se| se.is_finite() && se > 0.0);
        let eligible_msm = query.method == "marginal_structural_model"
            && query.probabilities_known_by_design
            && query.subject_ids.len() >= 300
            && regime.observed_subjects >= 200
            && regime.effective_sample_size >= 150.0
            && regime.value_standard_error.is_some_and(|se| se.is_finite() && se > 0.0)
            && regime.standard_errors.len() == query.periods
            && regime.standard_errors.iter().all(|se| se.is_finite() && *se > 0.0);
        let n = query.subject_ids.len();
        let dr_cells = n.checked_mul(query.periods);
        let dr_matching_subjects = if matches!(query.periods, 2 | 3)
            && dr_cells.is_some_and(|cells| query.treatment_history.len() == cells
                && query.regime_actions.len() == cells)
            && query.outcome_observed.len() == n {
            (0..n).filter(|&i| query.outcome_observed[i]
                && (0..query.periods).all(|t| {
                    let j = query.periods * i + t;
                    query.treatment_history[j] == query.regime_actions[j]
                })).count()
        } else { 0 };
        let dr_min_action = query.regime_actions.iter().zip(&query.treatment_probabilities)
            .map(|(action, p)| if *action { *p } else { 1.0 - p })
            .fold(1.0_f64, f64::min);
        let dr_min_censor = query.censoring_probabilities.iter().copied().fold(1.0_f64, f64::min);
        let eligible_dr = query.method == "sequential_dr"
            && query.probabilities_known_by_design
            && query.excluded_fold_predictions
            && query.prediction_fold_ids == query.fold_ids
            && (query.periods == 2 && n >= 300
                && query.outcome_observed.iter().filter(|&&observed| observed).count() >= 200
                && dr_matching_subjects >= 50
                && dr_min_action >= 0.4 && dr_min_censor >= 0.85
                || query.periods == 3 && n >= 800
                && query.outcome_observed.iter().filter(|&&observed| observed).count() >= 600
                && dr_matching_subjects >= 60
                && dr_min_action >= 0.5 && dr_min_censor >= 0.9)
            && dr_cells.is_some_and(|cells| query.q_predictions.len() == cells)
            && regime.value_standard_error.is_some_and(|se| se.is_finite() && se > 0.0);
        let g_formula_expected = if query.method == "g_formula"
            && query.known_fixed_outcome_predictions
            && query.probabilities_known_by_design {
            antecedent_estimate::longitudinal_regime::evaluate_g_formula_value(
                &query.period_outcome_predictions, &query.regime_actions,
                &query.treatment_probabilities, &query.censoring_probabilities,
                n, query.periods, query.minimum_probability,
            ).ok().and_then(|summary| {
                antecedent_estimate::longitudinal_regime::g_formula_fixed_q_pointwise_interval_95(
                    &summary, &query.period_outcome_predictions, n, query.periods,
                ).map(|interval| (summary.value, interval))
            })
        } else { None };
        let expected_reason = match query.method.as_str() {
            "ipw" if regime.value_interval_95.is_none() => Some("insufficient_independent_subject_support_or_degenerate_score"),
            "g_formula" if !query.known_fixed_outcome_predictions => Some("prediction_model_uncertainty_not_accounted"),
            "g_formula" if !query.probabilities_known_by_design => Some("known_sequential_randomization_required_for_fixed_q_interval"),
            "g_formula" if g_formula_expected.is_none() => Some("insufficient_two_period_subject_support_or_degenerate_fixed_q_score"),
            "sequential_dr" if regime.value_interval_95.is_none() => Some("insufficient_calibrated_horizon_or_trajectory_support_for_sequential_dr_interval"),
            "marginal_structural_model" if regime.value_interval_95.is_none() => Some("insufficient_independent_subject_support_for_msm_intervals"),
            _ => None,
        };
        let expected_uncertainty = if query.method == "marginal_structural_model" {
            if regime.value_interval_95.is_some() { "pointwise_subject_clustered_cr1_95" }
            else { "pointwise_subject_clustered_cr1_no_interval" }
        } else if query.method == "sequential_dr" && regime.value_interval_95.is_some() {
            "pointwise_subject_score_conditional_excluded_fold_q_95"
        } else if query.method == "g_formula" && regime.value_interval_95.is_some() {
            "pointwise_subject_score_conditional_fixed_known_q_95"
        } else if regime.value_interval_95.is_some() {
            "pointwise_subject_score_95"
        } else { "point_only_no_interval" };
        #[allow(clippy::float_cmp, reason = "wire interval bounds must equal the recomputed values exactly; drift is a real mismatch")]
        let interval_valid = match regime.value_interval_95 {
            Some(bounds) => {
                let se = regime.value_standard_error.unwrap_or(f64::NAN);
                let span = antecedent_stats::normal_ppf(0.975) * se;
                let tolerance = 1e-10 * (1.0 + regime.value.abs() + span.abs());
                (eligible_ipw || eligible_msm || eligible_dr || g_formula_expected.is_some())
                    && g_formula_expected.is_none_or(|(value, (expected_se, expected_bounds))| {
                        (regime.value - value).abs() <= 1e-10
                            && (se - expected_se).abs() <= 1e-10
                            && bounds == expected_bounds
                    })
                    && bounds[0].is_finite() && bounds[1].is_finite()
                    && (bounds[0] - (regime.value - span)).abs() <= tolerance
                    && (bounds[1] - (regime.value + span)).abs() <= tolerance
            }
            None => true,
        };
        let period_intervals_valid = if regime.period_intervals_95.is_empty() {
            regime.value_interval_95.is_none() || query.method != "marginal_structural_model"
        } else {
            eligible_msm && regime.value_interval_95.is_some()
                && regime.period_intervals_95.len() == query.periods
                && regime.period_intervals_95.iter().zip(&regime.period_effects)
                    .zip(&regime.standard_errors).all(|((bounds, center), se)| {
                        let span = antecedent_stats::normal_ppf(0.975) * se;
                        let tolerance = 1e-10 * (1.0 + center.abs() + span.abs());
                        bounds[0].is_finite() && bounds[1].is_finite()
                            && (bounds[0] - (center - span)).abs() <= tolerance
                            && (bounds[1] - (center + span)).abs() <= tolerance
                    })
        };
        let graphless_dr_licensed = query.method == "sequential_dr"
            && matches!(query.periods, 2 | 3)
            && eligible_dr
            && regime.value_interval_95.is_some()
            && graphless_data::LICENSES.iter().any(|row| {
                row.family == "longitudinal_regime"
                    && row.design == if query.periods == 2 {
                        "known_sequential_randomized_two_period"
                    } else {
                        "known_sequential_randomized_three_period"
                    }
                    && row.method == "subject_excluded_q_sequential_dr_scores"
                    && row.inference_claim == "conditional_q_pointwise_95_normal_interval"
            });
        if regime.graphless_support_status.as_deref() != graphless_dr_licensed.then_some("licensed")
            && !(allow_legacy_graphless_missing && regime.graphless_support_status.is_none())
        {
            return Err(IoError::Convert(
                "longitudinal regime graphless license does not match evidenced support".into(),
            ));
        }
        if regime.method != query.method
            || regime.rule_id != query.rule_id
            || regime.rule_version != query.rule_version
            || regime.rule_provenance != query.rule_provenance
            || result.estimate.is_some()
            || result.standard_error.is_some()
            || result.interval_lower.is_some()
            || result.interval_upper.is_some()
            || !regime.value.is_finite()
            || regime.value_standard_error.is_some_and(|se| !se.is_finite() || se < 0.0
                || !(matches!(query.method.as_str(), "ipw" | "marginal_structural_model")
                    || query.method == "g_formula" && g_formula_expected.is_some()
                    || query.method == "sequential_dr" && regime.value_interval_95.is_some()))
            || !interval_valid
            || !period_intervals_valid
            || query.method == "sequential_dr" && regime.interval_reason.as_deref() != expected_reason
            || regime.interval_reason.as_deref().is_some_and(|reason| Some(reason) != expected_reason)
            || regime.value_interval_95.is_some() && regime.interval_reason.is_some()
            || !regime.effective_sample_size.is_finite()
            || regime.effective_sample_size <= 0.0
            || regime.effective_sample_size > query.subject_ids.len() as f64
            || !regime.matched_observed_fraction.is_finite()
            || !(0.0 < regime.matched_observed_fraction && regime.matched_observed_fraction <= 1.0)
            || !regime.maximum_weight.is_finite()
            || regime.maximum_weight <= 0.0
            || !regime.minimum_action_probability.is_finite()
            || regime.minimum_action_probability < query.minimum_probability
            || !regime.minimum_censoring_probability.is_finite()
            || regime.minimum_censoring_probability < query.minimum_probability
            || regime.uncertainty != expected_uncertainty
            || regime.probability_ownership
                != if query.probabilities_known_by_design {
                    "known_sequential_randomization"
                } else {
                    "caller_declared_subject_excluded_fold_predictions"
                }
        {
            return Err(IoError::Convert(
                "invalid longitudinal regime payload or fabricated interval".into(),
            ));
        }
        if query.method == "marginal_structural_model" {
            if regime.period_effects.len() != query.periods
                || regime.standard_errors.len() != query.periods
                || regime.stabilizing_numerator_probabilities != query.stabilizing_numerator_probabilities
                || regime.observed_subjects <= query.periods + 1
                || regime.observed_subjects > query.subject_ids.len()
                || regime.period_effects.iter().any(|v| !v.is_finite())
                || regime.standard_errors.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(IoError::Convert("invalid longitudinal MSM coefficients or uncertainty".into()));
            }
        } else if !regime.period_effects.is_empty() || !regime.standard_errors.is_empty()
            || !regime.stabilizing_numerator_probabilities.is_empty() || regime.observed_subjects != 0 {
            return Err(IoError::Convert("MSM-only result fields attached to a regime value".into()));
        }
    }
    crate::causal_query_from_wire(&result.query)?;
    let identification_count = match &result.identification_variables {
        Some(variables) => {
            validate_temporal_namespace(variables, variable_names.len())?;
            variables.len()
        }
        None => variable_names.len(),
    };
    validate_identification_namespace(&result.identification, identification_count)?;
    let mut horizons = std::collections::BTreeSet::new();
    for horizon in &result.temporal_identification {
        if horizon.horizon == 0 || !horizons.insert(horizon.horizon) {
            return Err(IoError::Convert(
                "temporal identification horizons must be positive and unique".into(),
            ));
        }
        validate_temporal_namespace(&horizon.variables, variable_names.len())?;
        validate_identification_namespace(&horizon.identification, horizon.variables.len())?;
        crate::identification_from_wire(&horizon.identification)?;
    }
    crate::identification_from_wire(&result.identification)?;
    validate_interventional_distribution(result)?;
    if result.identification_variables.is_none() && result.identification.query != result.query {
        return Err(IoError::Convert(
            "identification.query does not match the enclosing query".into(),
        ));
    }
    if let crate::CausalQueryWire::TemporalEffect { horizon_steps, .. } = &result.query {
        if !result.temporal_identification.is_empty()
            && !result.temporal_identification.iter().any(|item| item.horizon == *horizon_steps)
        {
            return Err(IoError::Convert(
                "temporal certificate horizon does not match the enclosing query".into(),
            ));
        }
    }
    if result.estimate.is_some_and(|estimate| !estimate.is_finite()) {
        return Err(IoError::Convert(
            "analysis scalar estimate must be finite when present".into(),
        ));
    }
    if matches!(result.query, crate::CausalQueryWire::ContinuousDoseResponse(_))
        != result.continuous_dose_response.is_some()
    {
        return Err(IoError::Convert("continuous-dose query and result section must appear together".into()));
    }
    if let Some(fit) = &result.continuous_dose_response {
        let crate::CausalQueryWire::ContinuousDoseResponse(query) = &result.query else {
            return Err(IoError::Convert("continuous-dose result is attached to a different query".into()));
        };
        let groups: std::collections::BTreeSet<&str> = query.baseline_groups.iter().map(String::as_str).collect();
        let policy_requested = query.fixed_policy.is_some();
        let policy_interval = fit.fixed_policy.as_ref().is_some_and(|value| value.incremental_interval_95.is_some());
        let expected_uncertainty = if policy_interval {
            "fixed_group_kernel_smoothed_paired_pointwise_95_normal_intervals"
        } else if policy_requested {
            "fixed_group_kernel_smoothed_paired_variance_no_interval"
        } else { "point_only_no_interval" };
        #[allow(clippy::float_cmp, reason = "wire fields must equal the frozen query bytes exactly; any drift is a real mismatch")]
        if result.estimate.is_some() || result.standard_error.is_some()
            || result.interval_lower.is_some() || result.interval_upper.is_some()
            || fit.bandwidth != query.bandwidth || fit.density_provenance != query.density_provenance
            || fit.uncertainty != expected_uncertainty
            || fit.fixed_policy.is_some() != policy_requested
            || fit.points.len() != groups.len() * query.target_doses.len()
            || (!policy_requested && fit.graphless_support_status.is_some())
        {
            return Err(IoError::Convert("continuous-dose result must match its query and uncertainty".into()));
        }
        for (index, point) in fit.points.iter().enumerate() {
            let group = groups.iter().nth(index / query.target_doses.len()).copied().unwrap_or("");
            let target = query.target_doses[index % query.target_doses.len()];
            #[allow(clippy::float_cmp, reason = "target dose must equal the frozen query grid value exactly")]
            if point.baseline_group != group || point.target_dose != target
                || point.local_rows < query.min_local_support
                || ![point.response, point.effective_sample_size, point.minimum_dose_density,
                    point.maximum_normalized_weight, point.local_outcome_sd].iter().all(|value| value.is_finite())
                || point.effective_sample_size <= 0.0 || point.minimum_dose_density <= 0.0
                || !(0.0..=1.0).contains(&point.maximum_normalized_weight)
                || point.local_outcome_sd < 0.0
            {
                return Err(IoError::Convert("continuous-dose point or local support is invalid".into()));
            }
        }
        if let Some(value) = &fit.fixed_policy {
            let Some((policy_doses, reference_doses)) = &query.fixed_policy else {
                return Err(IoError::Convert("continuous-dose policy maps are missing".into()));
            };
            let mut expected_policy = policy_doses.clone();
            let mut expected_reference = reference_doses.clone();
            expected_policy.sort_by(|a, b| a.0.cmp(&b.0));
            expected_reference.sort_by(|a, b| a.0.cmp(&b.0));
            let minimum_group_rows = groups.iter().map(|group| query.baseline_groups.iter()
                .filter(|observed| observed.as_str() == *group).count()).min().unwrap_or(0);
            let intervals = [value.policy_interval_95, value.reference_interval_95,
                value.incremental_interval_95];
            let variances = [value.policy_variance, value.reference_variance,
                value.incremental_variance];
            let centers = [value.policy_value, value.reference_value, value.incremental_value];
            let supported = graphless_data::LICENSES.iter().any(|row| {
                row.family == "continuous_dose_policy"
                    && row.design == "fixed_group_kernel"
                    && row.method == "inverse_density_kernel_paired_scores"
                    && row.inference_claim == "policy_reference_incremental_pointwise_95_normal_intervals"
                    && row.assignment_unit == "unit"
                    && query.baseline_groups.len() >= row.min_rows
                    && minimum_group_rows >= row.min_group_rows
                    && value.minimum_local_rows >= row.min_local_rows
                    && value.minimum_effective_sample_size + 1e-12 >= row.min_effective_sample_size
                    && (row.max_normalized_weight == 0.0
                        || value.maximum_normalized_weight <= row.max_normalized_weight + 1e-12)
                    && value.minimum_dose_density + 1e-12 >= row.min_dose_density
                    && (!row.requires_known_density || query.density_provenance == "known")
                    && row.min_reported_intervals == 3 && row.all_reported_intervals
                    && variances.iter().all(|variance| *variance > 0.0)
            });
            if value.policy_doses != expected_policy || value.reference_doses != expected_reference
                || !centers.iter().chain(&variances).all(|number| number.is_finite())
                || (value.policy_value - value.reference_value - value.incremental_value).abs() > 1e-8
                || variances.iter().any(|variance| *variance < 0.0)
                || value.minimum_local_rows < query.min_local_support
                || !value.minimum_effective_sample_size.is_finite()
                || value.minimum_effective_sample_size <= 0.0
                || !value.maximum_normalized_weight.is_finite()
                || !(0.0..=1.0).contains(&value.maximum_normalized_weight)
                || !value.minimum_dose_density.is_finite() || value.minimum_dose_density <= 0.0
                || policy_interval != supported
                || (fit.graphless_support_status.as_deref() != supported.then_some("licensed")
                    && !(allow_legacy_graphless_missing && fit.graphless_support_status.is_none()))
                || intervals.iter().any(|interval| interval.is_some() != policy_interval)
                || intervals.iter().zip(centers).zip(variances).any(|((interval, center), variance)| {
                    interval.is_some_and(|[lower, upper]| {
                        let radius = 1.959_963_984_540_054 * variance.sqrt();
                        !lower.is_finite() || !upper.is_finite()
                            || (lower - (center - radius)).abs() > 1e-8
                            || (upper - (center + radius)).abs() > 1e-8
                    })
                })
            {
                return Err(IoError::Convert("continuous-dose policy value or paired interval is invalid".into()));
            }
        }
    }
    if let Some(policy) = &result.policy_value {
        let expected_uncertainty = match &result.query {
            crate::CausalQueryWire::PolicyValue(query) if query.multi_action.is_some() => {
                "multi_action_ipw_row_score_standard_error_independent_subjects"
            }
            crate::CausalQueryWire::PolicyValue(query) if query.mu0.is_empty() && query.mu1.is_empty() => {
                "ipw_row_score_standard_error_independent_subjects"
            }
            crate::CausalQueryWire::PolicyValue(_) => "row_score_standard_error_independent_subjects",
            _ => "invalid_policy_value_query",
        };
        let expected_interval = match &result.query {
            crate::CausalQueryWire::PolicyValue(query) => {
                let (n, policy_matches, reference_matches) = if let Some(multi) = &query.multi_action {
                    (multi.assignment.len(),
                     multi.assignment.iter().zip(&multi.actions).filter(|(a, b)| a == b).count(),
                     multi.assignment.iter().zip(&multi.reference).filter(|(a, b)| a == b).count())
                } else {
                    (query.assignment.len(),
                     query.assignment.iter().zip(&query.actions).filter(|(a, b)| a == b).count(),
                     query.assignment.iter().zip(&query.reference).filter(|(a, b)| a == b).count())
                };
                let multi_constraints_couple = query.multi_action.as_ref().is_some_and(|multi| {
                    let n = multi.assignment.len();
                    [(&multi.capacities, &multi.costs, multi.budget),
                     (&multi.reference_capacities, &multi.reference_costs, multi.reference_budget)]
                        .into_iter().any(|(capacities, costs, budget)| {
                            capacities.iter().any(|&limit| limit < n)
                                || budget.is_some_and(|limit| {
                                    let maximum_cost = costs.iter().copied().fold(0.0_f64, f64::max);
                                    limit + 1e-12 < n as f64 * maximum_cost
                                })
                        })
                });
                n >= (if query.multi_action.is_some() || !query.mu0.is_empty() { 300 } else { 120 })
                    && policy_matches >= 10 && reference_matches >= 10
                    && !query.global_constraints_present && !multi_constraints_couple
                    && (query.multi_action.is_some() || query.mu0.is_empty()
                        || query.disjoint_training_subjects || query.crossfit_fold_ownership_valid)
                    && policy.policy_standard_error > 0.0 && policy.incremental_standard_error > 0.0
            }
            _ => false,
        };
        let z = antecedent_stats::normal_ppf(0.975);
        let bounds_match = |bounds: Option<[f64; 2]>, center: f64, se: f64| {
            let Some(bounds) = bounds else { return true; };
            let span = z * se;
            let tolerance = 1e-10 * (1.0 + center.abs() + span.abs());
            expected_interval && (bounds[0] - (center - span)).abs() <= tolerance
                && (bounds[1] - (center + span)).abs() <= tolerance
        };
        if result.estimate.is_some()
            || ![
                policy.policy_value,
                policy.reference_value,
                policy.incremental_value,
                policy.relative_value_gap,
                policy.treatment_rate,
                policy.total_cost,
                policy.policy_standard_error,
                policy.reference_standard_error,
                policy.incremental_standard_error,
                policy.propensity_min,
                policy.propensity_max,
            ]
            .iter()
            .all(|value| value.is_finite())
            || !(0.0..=1.0).contains(&policy.treatment_rate)
            || policy.total_cost < 0.0
            || policy.policy_standard_error < 0.0
            || policy.reference_standard_error < 0.0
            || policy.incremental_standard_error < 0.0
            || policy.policy_interval_95.is_some() != policy.incremental_interval_95.is_some()
            || policy.policy_interval_95.iter().chain(policy.incremental_interval_95.iter())
                .any(|bounds| !bounds.iter().all(|value| value.is_finite()) || bounds[0] > bounds[1])
            || !bounds_match(policy.policy_interval_95, policy.policy_value, policy.policy_standard_error)
            || !bounds_match(policy.incremental_interval_95, policy.incremental_value, policy.incremental_standard_error)
            || !(0.0..=1.0).contains(&policy.propensity_min)
            || !(0.0..=1.0).contains(&policy.propensity_max)
            || policy.propensity_min > policy.propensity_max
            || policy.prediction_ownership.is_empty()
            || policy.uncertainty != expected_uncertainty
        {
            return Err(IoError::Convert(
                "invalid policy-value payload or fabricated scalar effect".into(),
            ));
        }
        let crate::CausalQueryWire::PolicyValue(query) = &result.query else {
            return Err(IoError::Convert("policy uplift bins require a policy-value query".into()));
        };
        if let Some(regret) = &policy.regret {
            let Some(design) = &query.regret else {
                return Err(IoError::Convert("finite-class regret requires its bound candidate query".into()));
            };
            let k = design.candidates.len();
            let n = query.assignment.len();
            if n < 400 || !(2..=16).contains(&k) || design.selected_index >= k
                || regret.selected_index != design.selected_index
                || regret.candidate_values.len() != k || regret.contrast_standard_errors.len() != k
                || query.global_constraints_present || query.multi_action.is_some()
                || !query.mu0.is_empty() || !query.uplift_bins.is_empty()
                || query.propensity.iter().any(|p| !(0.2..=0.8).contains(p))
                || design.training_subject_ids.is_empty()
                || design.training_subject_ids.iter().any(|id| query.evaluation_subject_ids.contains(id))
                || design.candidates.iter().any(|candidate| candidate.len() != n
                    || query.assignment.iter().zip(candidate).filter(|(a, b)| a == b).count() < 50)
                || design.candidates[design.selected_index] != query.actions
            {
                return Err(IoError::Convert("finite-class regret support or ownership is invalid".into()));
            }
            let selected = regret.candidate_values[regret.selected_index];
            let critical = antecedent_stats::normal_ppf(1.0 - 0.05 / (2.0 * (k - 1) as f64));
            let mut lower: f64 = 0.0;
            let mut upper: f64 = 0.0;
            let mut point: f64 = 0.0;
            for (j, (&value, &se)) in regret.candidate_values.iter()
                .zip(&regret.contrast_standard_errors).enumerate() {
                if !value.is_finite() || !se.is_finite() || se < 0.0
                    || (j == regret.selected_index && se != 0.0) {
                    return Err(IoError::Convert("finite-class regret candidate score is invalid".into()));
                }
                if j == regret.selected_index { continue; }
                let difference = value - selected;
                let span = critical * se;
                point = point.max(difference);
                lower = lower.max(difference - span);
                upper = upper.max(difference + span);
            }
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * (1.0 + a.abs() + b.abs());
            if !close(selected, policy.policy_value) || !close(point, regret.regret)
                || !close(lower, regret.interval_95[0]) || !close(upper, regret.interval_95[1]) {
                return Err(IoError::Convert("finite-class regret interval does not match paired candidate contrasts".into()));
            }
        } else if query.regret.is_some() {
            return Err(IoError::Convert("finite-class regret query requires a regret result".into()));
        }
        if policy.uplift_bins.len() != query.uplift_bin_count
            || (!query.uplift_bins.is_empty() && query.uplift_bins.len() != query.assignment.len())
            || policy.uplift_bins.iter().enumerate().any(|(rank, bin)| {
                let rows = query.uplift_bins.iter().enumerate()
                    .filter(|&(_, &group)| group == rank).collect::<Vec<_>>();
                let treated = rows.iter().filter(|&&(i, _)| query.assignment[i]).count();
                let controls = rows.len() - treated;
                let interval_valid = bin.interval_95.is_none_or(|bounds| {
                    let span = z * bin.standard_error;
                    let tolerance = 1e-10 * (1.0 + bin.effect.abs() + span.abs());
                    rows.len() >= 300 && treated >= 50 && controls >= 50
                        && !query.uplift_training_subject_ids.is_empty()
                        && query.uplift_training_subject_ids.iter()
                            .all(|id| !query.evaluation_subject_ids.contains(id))
                        && bin.standard_error > 0.0
                        && bounds.iter().all(|value| value.is_finite())
                        && (bounds[0] - (bin.effect - span)).abs() <= tolerance
                        && (bounds[1] - (bin.effect + span)).abs() <= tolerance
                });
                bin.rank != rank || bin.evaluation_rows != rows.len()
                    || !bin.effect.is_finite() || !bin.standard_error.is_finite()
                    || bin.standard_error < 0.0 || !interval_valid
            })
        {
            return Err(IoError::Convert("invalid policy uplift-bin support or interval".into()));
        }
        if let Some(multi) = &query.multi_action {
            let k = multi.action_labels.len();
            if k < 2 {
                return Err(IoError::Convert("multi-action CATE requires a control and an action".into()));
            }
            let mut groups = multi.cate_groups.clone();
            groups.sort_unstable();
            groups.dedup();
            if policy.multi_action_cate.len() != groups.len() * k.saturating_sub(1)
                || policy.multi_action_cate.iter().enumerate().any(|(index, point)| {
                    let group = &groups[index / (k - 1)];
                    let action = index % (k - 1) + 1;
                    let rows = multi.cate_groups.iter().enumerate()
                        .filter(|(_, label)| *label == group).map(|(row, _)| row).collect::<Vec<_>>();
                    let observed_action = rows.iter().filter(|&&row| multi.assignment[row] == action).count();
                    let observed_control = rows.iter().filter(|&&row| multi.assignment[row] == 0).count();
                    let strong_overlap = rows.iter().all(|&row| {
                        multi.propensities[row * k] >= 0.2
                            && multi.propensities[row * k + action] >= 0.2
                    });
                    let interval_valid = point.interval_95.is_none_or(|bounds| {
                        let span = z * point.standard_error;
                        let tolerance = 1e-10 * (1.0 + point.effect.abs() + span.abs());
                        rows.len() >= 300 && observed_action >= 50 && observed_control >= 50
                            && strong_overlap && point.standard_error > 0.0
                            && bounds.iter().all(|value| value.is_finite())
                            && (bounds[0] - (point.effect - span)).abs() <= tolerance
                            && (bounds[1] - (point.effect + span)).abs() <= tolerance
                    });
                    point.group != group.as_str() || point.action != multi.action_labels[action].as_str()
                        || point.evaluation_rows != rows.len()
                        || point.observed_action_rows != observed_action
                        || point.observed_control_rows != observed_control
                        || !point.effect.is_finite() || !point.standard_error.is_finite()
                        || point.standard_error < 0.0 || !interval_valid
                })
            {
                return Err(IoError::Convert("invalid multi-action CATE support or interval".into()));
            }
        } else if !policy.multi_action_cate.is_empty() {
            return Err(IoError::Convert("multi-action CATE requires a multi-action query".into()));
        }
        let (design, method, claim) = antecedent_core::policy_graphless_coordinate(
            query.multi_action.is_some(), query.mu0.is_empty(),
            !policy.uplift_bins.is_empty(), !policy.multi_action_cate.is_empty(),
            query.regret.is_some(),
            !query.mu0.is_empty() && query.crossfit_fold_ownership_valid
                && !query.disjoint_training_subjects,
        );
        let (treated, control, min_action_rows, min_probability, policy_matches, reference_matches, uncoupled) =
            if let Some(multi) = &query.multi_action {
                let n = multi.assignment.len();
                let k = multi.action_labels.len();
                let treated = multi.assignment.iter().filter(|&&action| action != 0).count();
                let policy_matches = multi.assignment.iter().zip(&multi.actions).filter(|(a, b)| a == b).count();
                let reference_matches = multi.assignment.iter().zip(&multi.reference).filter(|(a, b)| a == b).count();
                let coupled = [(&multi.capacities, &multi.costs, multi.budget),
                    (&multi.reference_capacities, &multi.reference_costs, multi.reference_budget)]
                    .into_iter().any(|(capacities, costs, budget)| {
                        capacities.iter().any(|&limit| limit < n)
                            || budget.is_some_and(|limit| limit + 1e-12 < n as f64
                                * costs.iter().copied().fold(0.0_f64, f64::max))
                    });
                (treated, n - treated,
                 (0..k).map(|action| multi.assignment.iter().filter(|&&a| a == action).count()).min().unwrap_or(0),
                 multi.propensities.iter().copied().fold(f64::INFINITY, f64::min),
                 policy_matches, reference_matches, !query.global_constraints_present && !coupled)
            } else {
                let n = query.assignment.len();
                let treated = query.assignment.iter().filter(|&&assigned| assigned).count();
                (treated, n - treated, treated.min(n - treated),
                 query.propensity.iter().copied().map(|p| p.min(1.0 - p)).fold(f64::INFINITY, f64::min),
                 query.assignment.iter().zip(&query.actions).filter(|(a, b)| a == b).count(),
                 query.assignment.iter().zip(&query.reference).filter(|(a, b)| a == b).count(),
                 !query.global_constraints_present)
            };
        let min_bin_rows = policy.uplift_bins.iter().map(|bin| bin.evaluation_rows).min().unwrap_or(0);
        let min_bin_arm_rows = if policy.uplift_bins.is_empty() { 0 } else {
            (0..policy.uplift_bins.len()).flat_map(|bin| {
                let treated = query.uplift_bins.iter().enumerate()
                    .filter(|&(row, &rank)| rank == bin && query.assignment[row]).count();
                [treated, policy.uplift_bins[bin].evaluation_rows - treated]
            }).min().unwrap_or(0)
        };
        let scalar_intervals = usize::from(policy.policy_interval_95.is_some())
            + usize::from(policy.incremental_interval_95.is_some());
        let reported_intervals = scalar_intervals
            + policy.uplift_bins.iter().filter(|bin| bin.interval_95.is_some()).count()
            + policy.multi_action_cate.iter().filter(|point| point.interval_95.is_some()).count();
        let all_reported_intervals = scalar_intervals == 2
            && policy.uplift_bins.iter().all(|bin| bin.interval_95.is_some())
            && policy.multi_action_cate.iter().all(|point| point.interval_95.is_some());
        let min_group_rows = policy.multi_action_cate.iter().map(|point| point.evaluation_rows).min().unwrap_or(0);
        let min_group_arm_rows = policy.multi_action_cate.iter()
            .flat_map(|point| [point.observed_action_rows, point.observed_control_rows]).min().unwrap_or(0);
        let licensed = graphless_data::LICENSES.iter().any(|row| {
            row.family == "policy_value" && row.design == design && row.method == method
                && row.inference_claim == claim && row.assignment_unit == "unit"
                && query.evaluation_subject_ids.len() >= row.min_rows
                && treated >= row.min_assignment_units_per_arm
                && control >= row.min_assignment_units_per_arm
                && min_action_rows >= row.min_action_rows
                && min_probability + 1e-12 >= row.min_probability
                && policy_matches >= row.min_policy_matches
                && reference_matches >= row.min_reference_matches
                && min_bin_rows >= row.min_bin_rows && min_bin_arm_rows >= row.min_bin_arm_rows
                && min_group_rows >= row.min_group_rows && min_group_arm_rows >= row.min_group_arm_rows
                && reported_intervals >= row.min_reported_intervals
                && (!row.all_reported_intervals || all_reported_intervals)
                && (!row.requires_uncoupled_constraints || uncoupled)
                && (!row.requires_disjoint_nuisance_training || query.disjoint_training_subjects)
                && (!row.requires_rank_ownership || (!query.uplift_training_subject_ids.is_empty()
                    && query.uplift_training_subject_ids.iter()
                        .all(|id| !query.evaluation_subject_ids.contains(id))))
                && scalar_intervals == 2
        });
        if policy.graphless_support_status.as_deref().is_some_and(|status|
            Some(status) != licensed.then_some("licensed")) {
            return Err(IoError::Convert("policy graphless support status does not match exact design and interval evidence".into()));
        }
    }
    if let Some(did) = &result.panel_did {
        let crate::CausalQueryWire::PanelDid(query) = &result.query else {
            return Err(IoError::Convert(
                "panel DiD result section is attached to a different query".into(),
            ));
        };
        let mut subjects = std::collections::BTreeMap::<&str, (bool, &str)>::new();
        let mut group_clusters: [std::collections::BTreeSet<&str>; 2] = Default::default();
        let mut cell_clusters: [[std::collections::BTreeSet<&str>; 2]; 2] = Default::default();
        let mut duplicate_subject = false;
        for i in 0..query.treated.len() {
            let subject = query.subjects[i].as_str();
            let cluster = query.clusters[i].as_str();
            match subjects.insert(subject, (query.treated[i], cluster)) {
                Some((group, old_cluster))
                    if group != query.treated[i] || old_cluster != cluster =>
                {
                    return Err(IoError::Convert(
                        "panel DiD query changes treatment or cluster within a subject".into(),
                    ));
                }
                Some(_) => duplicate_subject = true,
                None => {}
            }
            group_clusters[usize::from(query.treated[i])].insert(cluster);
            cell_clusters[usize::from(query.treated[i])][usize::from(query.post[i])]
                .insert(cluster);
        }
        let representative = did.event_time_effects.iter().find(|effect| effect.2 >= 0);
        #[allow(clippy::float_cmp, reason = "representative effect/SE must equal the sealed event-time rows exactly")]
        let event_study_valid = if query.staggered_event_study {
            let mut unit_metadata = std::collections::BTreeMap::<&str, (i64, &str)>::new();
            let mut observed_periods = std::collections::BTreeSet::new();
            for i in 0..query.subjects.len() {
                let subject = query.subjects[i].as_str();
                let metadata = (query.cohorts[i], query.clusters[i].as_str());
                if unit_metadata.insert(subject, metadata).is_some_and(|old| old != metadata) {
                    return Err(IoError::Convert("event study changes cohort or cluster within subject".into()));
                }
                observed_periods.insert(query.periods[i]);
            }
            let adoption_cohorts: std::collections::BTreeSet<_> = unit_metadata.values()
                .map(|(cohort, _)| *cohort).filter(|cohort| *cohort > 0).collect();
            let expected_keys: std::collections::BTreeSet<_> = adoption_cohorts.iter()
                .flat_map(|cohort| observed_periods.iter().filter(move |period| **period != *cohort - 1)
                    .map(move |period| (*cohort, *period))).collect();
            let actual_keys: std::collections::BTreeSet<_> = did.event_time_effects.iter()
                .map(|effect| (effect.0, effect.1)).collect();
            let control_count = unit_metadata.values().filter(|(cohort, _)| *cohort == 0).count();
            !did.event_time_effects.is_empty()
                && actual_keys == expected_keys
                && actual_keys.len() == did.event_time_effects.len()
                && representative.is_some()
                && did.event_time_effects.iter().all(|effect| {
                    let (cohort, period, event_time, point, treated, controls, se, clusters) = *effect;
                    let expected_treated = unit_metadata.values().filter(|(g, _)| *g == cohort).count();
                    let expected_clusters = unit_metadata.values()
                        .filter(|(g, _)| *g == cohort || *g == 0)
                        .map(|(_, cluster)| *cluster)
                        .collect::<std::collections::BTreeSet<_>>().len();
                    cohort > 0 && period > 0 && event_time == period - cohort
                        && event_time != -1 && point.is_finite() && se.is_finite() && se >= 0.0
                        && treated == expected_treated && controls == control_count
                        && clusters == expected_clusters && treated >= 2 && controls >= 2 && clusters >= 4
                })
                && representative.is_some_and(|effect| did.effect == effect.3
                    && did.standard_error == Some(effect.6))
        } else { did.event_time_effects.is_empty() };
        let event_intervals_valid = if query.staggered_event_study {
            if did.event_time_intervals_95.is_empty() {
                did.interval_95.is_none()
                    && did.uncertainty == "cluster_robust_standard_error_no_interval"
            } else if did.event_time_intervals_95.len() != did.event_time_effects.len() {
                false
            } else {
                let mut clusters_by_cohort = std::collections::BTreeMap::<i64, std::collections::BTreeSet<&str>>::new();
                for (cohort, cluster) in query.cohorts.iter().zip(&query.clusters) {
                    clusters_by_cohort.entry(*cohort).or_default().insert(cluster);
                }
                let controls = clusters_by_cohort.get(&0).map_or(0, std::collections::BTreeSet::len);
                let valid = did.event_time_effects.iter().zip(&did.event_time_intervals_95)
                    .all(|(effect, interval)| {
                        let treated = clusters_by_cohort.get(&effect.0).map_or(0, std::collections::BTreeSet::len);
                        let supported = effect.2 >= 0 && treated >= 24 && controls >= 24
                            && effect.7 >= 48 && effect.6.is_finite() && effect.6 > 0.0;
                        match interval {
                            Some(bounds) if supported => {
                                let span = antecedent_stats::normal_ppf(0.975) * effect.6;
                                let tolerance = 1e-10 * (1.0 + effect.3.abs() + span.abs());
                                bounds[0].is_finite() && bounds[1].is_finite()
                                    && (bounds[0] - (effect.3 - span)).abs() <= tolerance
                                    && (bounds[1] - (effect.3 + span)).abs() <= tolerance
                            }
                            None => !supported,
                            _ => false,
                        }
                    });
                let representative_interval = did.event_time_effects.iter().position(|effect| effect.2 >= 0)
                    .and_then(|index| did.event_time_intervals_95[index]);
                let expected_uncertainty = if did.event_time_intervals_95.iter().any(Option::is_some) {
                    "event_time_pointwise_normal_intervals_independent_clusters"
                } else { "cluster_robust_standard_error_no_interval" };
                valid && did.interval_95 == representative_interval
                    && did.uncertainty == expected_uncertainty
            }
        } else { did.event_time_intervals_95.is_empty() };
        let (treated_subjects, comparison_subjects) = if query.staggered_event_study {
            representative.map_or((0, 0), |effect| (effect.4, effect.5))
        } else if let Some((target, _)) = query.staggered_target {
            let mut cohort_by_subject = std::collections::BTreeMap::new();
            for (subject, cohort) in query.subjects.iter().zip(&query.cohorts) {
                if cohort_by_subject.insert(subject.as_str(), *cohort).is_some_and(|old| old != *cohort) {
                    return Err(IoError::Convert("staggered DiD query changes adoption cohort within subject".into()));
                }
            }
            (
                cohort_by_subject.values().filter(|cohort| **cohort == target).count(),
                cohort_by_subject.values().filter(|cohort| **cohort == 0).count(),
            )
        } else {
            let treated_subjects = subjects.values().filter(|(treated, _)| *treated).count();
            (treated_subjects, subjects.len() - treated_subjects)
        };
        let clusters = if query.staggered_event_study {
            representative.map_or(0, |effect| effect.7)
        } else if let Some((target, _)) = query.staggered_target {
            query.clusters.iter().zip(&query.cohorts)
                .filter(|(_, cohort)| **cohort == 0 || **cohort == target)
                .map(|(cluster, _)| cluster)
                .collect::<std::collections::BTreeSet<_>>().len()
        } else {
            query.clusters.iter().collect::<std::collections::BTreeSet<_>>().len()
        };
        let augmented_valid = match (&query.augmented, &did.augmented) {
            (Some((pre, propensity, prediction, declared)), Some((p_min, p_max, ess, recorded))) => {
                let variables = [query.outcome, *pre, *propensity, *prediction];
                variables.iter().collect::<std::collections::BTreeSet<_>>().len() == 4
                    && !duplicate_subject && query.post.iter().all(|post| *post)
                    && query.periods.is_empty() && query.cohorts.is_empty()
                    && query.staggered_target.is_none() && !query.staggered_event_study
                    && !query.repeated_cross_section
                    && *declared == *recorded
                    && p_min.is_finite() && p_max.is_finite()
                    && *p_min > 0.0 && *p_min <= *p_max && *p_max < 1.0
                    && ess.is_finite() && *ess > 0.0 && *ess <= comparison_subjects as f64 + 1e-9
                    && did.standard_error.is_none()
                    && did.uncertainty == "point_only_no_standard_error"
            }
            (None, None) => true,
            _ => false,
        };
        let interval_supported = query.augmented.is_none()
            && query.staggered_target.is_none()
            && !query.staggered_event_study
            && if query.repeated_cross_section {
                !duplicate_subject && cell_clusters.iter().flatten().all(|members| members.len() >= 30)
            } else {
                group_clusters.iter().all(|members| members.len() >= 30)
            };
        let interval_valid = if query.staggered_event_study {
            let representative_scalar_valid = match (did.interval_95, did.standard_error, result.standard_error) {
                (Some(_), Some(section_se), Some(reported_se)) =>
                    section_se.is_finite() && section_se > 0.0
                        && (reported_se - section_se).abs() <= 1e-10 * (1.0 + section_se.abs()),
                (None, _, None) => true,
                _ => false,
            };
            representative_scalar_valid && event_intervals_valid
        } else { match (did.interval_95, did.standard_error) {
            (Some(bounds), Some(se)) if interval_supported && se.is_finite() && se > 0.0 => {
                let radius = 1.959_963_984_540_054 * se;
                let tolerance = 1e-8 * (1.0 + did.effect.abs() + radius.abs());
                bounds.iter().all(|value| value.is_finite())
                    && (bounds[0] - (did.effect - radius)).abs() <= tolerance
                    && (bounds[1] - (did.effect + radius)).abs() <= tolerance
                    && result.standard_error.is_some_and(|reported| (reported - se).abs() <= tolerance)
                    && did.uncertainty == "cluster_robust_normal_interval_independent_clusters"
            }
            (None, _) => result.standard_error.is_none()
                && (query.augmented.is_some() || did.uncertainty == "cluster_robust_standard_error_no_interval"),
            _ => false,
        }};
        let did_design = if query.repeated_cross_section { "repeated_cross_section_2x2" } else { "panel_2x2" };
        let did_method = if query.repeated_cross_section { "four_cell_cluster_scores_cr1" } else { "cluster_change_scores_cr1" };
        let min_cell_clusters = cell_clusters.iter().flatten().map(std::collections::BTreeSet::len).min().unwrap_or(0);
        let did_licensed = query.augmented.is_none() && query.staggered_target.is_none()
            && !query.staggered_event_study && interval_valid && did.interval_95.is_some()
            && graphless_data::LICENSES.iter().any(|row| {
                row.family == "difference_in_differences" && row.design == did_design
                    && row.method == did_method && row.inference_claim == "pointwise_95_normal_interval"
                    && row.assignment_unit == "cluster"
                    && group_clusters[0].len() >= row.min_assignment_units_per_arm
                    && group_clusters[1].len() >= row.min_assignment_units_per_arm
                    && (!query.repeated_cross_section || (!duplicate_subject
                        && 4 >= row.min_blocks && min_cell_clusters >= row.min_block_arm))
                    && 1 >= row.min_reported_intervals
            });
        if result.estimate != Some(did.effect)
            || !interval_valid
            || !did.effect.is_finite()
            || did.standard_error.is_some_and(|se| !se.is_finite() || se < 0.0)
            || (query.augmented.is_none() && did.standard_error.is_none())
            || did.treated_subjects != treated_subjects
            || did.comparison_subjects != comparison_subjects
            || did.clusters != clusters
            || !event_study_valid
            || !event_intervals_valid
            || !augmented_valid
            || did.graphless_support_status.as_deref().is_some_and(|status|
                Some(status) != did_licensed.then_some("licensed"))
            || (query.repeated_cross_section
                && (duplicate_subject
                    || cell_clusters.iter().flatten().any(|members| members.len() < 2)))
            || (!query.repeated_cross_section && query.staggered_target.is_none() && !query.staggered_event_study && query.augmented.is_none()
                && group_clusters.iter().any(|members| members.len() < 2))
            || result.interval_lower.is_some()
            || result.interval_upper.is_some()
        {
            return Err(IoError::Convert(
                "invalid panel DiD payload or fabricated interval".into(),
            ));
        }
    }
    if let Some(fit) = &result.synthetic_control {
        let crate::CausalQueryWire::SyntheticControl(query) = &result.query else {
            return Err(IoError::Convert("synthetic-control result is attached to a different query".into()));
        };
        if query.difference_in_differences {
            return Err(IoError::Convert("synthetic-control result is attached to a synthetic DiD query".into()));
        }
        let donors: std::collections::BTreeSet<&str> = query.units.iter()
            .map(String::as_str).filter(|unit| *unit != query.treated_unit.as_str()).collect();
        let pre: std::collections::BTreeSet<i64> = query.periods.iter().copied()
            .filter(|period| *period < query.intervention_period).collect();
        let post: std::collections::BTreeSet<i64> = query.periods.iter().copied()
            .filter(|period| *period >= query.intervention_period).collect();
        let weight_names: Vec<&str> = fit.donor_weights.iter().map(|(unit, _)| unit.as_str()).collect();
        let squared_mass: f64 = fit.donor_weights.iter().map(|(_, weight)| weight * weight).sum();
        #[allow(clippy::float_cmp, reason = "randomization null effect must equal the frozen sharp-null query value exactly")]
        if result.estimate != Some(fit.effect)
            || result.standard_error.is_some()
            || result.interval_lower.is_some() || result.interval_upper.is_some()
            || !fit.effect.is_finite()
            || !fit.pre_treatment_rmse.is_finite() || fit.pre_treatment_rmse < 0.0
            || !fit.placebo_rank.is_finite() || !(0.0..=1.0).contains(&fit.placebo_rank)
            || !fit.effective_donors.is_finite() || fit.effective_donors < 1.0
            || fit.donor_weights.len() != donors.len() || donors.len() < 3
            || weight_names != donors.iter().copied().collect::<Vec<_>>()
            || fit.donor_weights.iter().any(|(_, weight)| !weight.is_finite() || *weight < 0.0)
            || (fit.donor_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() > 1e-8
            || (fit.effective_donors - 1.0 / squared_mass).abs() > 1e-8
            || fit.placebo_effects.len() != donors.len()
            || fit.placebo_effects.iter().any(|effect| !effect.is_finite())
            || fit.n_pre_periods != pre.len() || fit.n_post_periods != post.len()
            || (!query.uniform_unit_randomization && (
                (query.augmentation_ridge.is_none() && fit.uncertainty != "point_only_with_unlicensed_placebo_rank")
                || fit.randomization_p_value.is_some() || fit.randomization_null_effect.is_some()
                || !fit.randomization_statistics.is_empty()))
            || (query.augmentation_ridge.is_none() && (
                fit.unadjusted_effect.is_some() || fit.outcome_model_correction.is_some()
                || fit.augmentation_ridge.is_some()))
            || (query.augmentation_ridge.is_some() && (
                (!query.uniform_unit_randomization && fit.uncertainty != "point_only_augmented_no_interval")
                || fit.augmentation_ridge != query.augmentation_ridge
                || !fit.unadjusted_effect.is_some_and(f64::is_finite)
                || !fit.outcome_model_correction.is_some_and(f64::is_finite)
                || (fit.unadjusted_effect.unwrap_or(f64::NAN)
                    - fit.outcome_model_correction.unwrap_or(f64::NAN) - fit.effect).abs() > 1e-8))
            || (query.uniform_unit_randomization && (
                fit.randomization_null_effect.is_some_and(|effect|
                    !effect.is_finite() || effect != query.sharp_null_effect.unwrap_or(0.0))
                || (query.sharp_null_effect.is_some() && fit.randomization_null_effect.is_none())
                || !query.sharp_null_effect.unwrap_or(0.0).is_finite()
                ||
                (query.augmentation_ridge.is_none()
                    && fit.uncertainty != "point_only_with_exact_unit_randomization_p_value_no_interval")
                || (query.augmentation_ridge.is_some()
                    && fit.uncertainty != "point_only_augmented_with_exact_unit_randomization_p_value_no_interval")
                || fit.randomization_statistics.len() != donors.len() + 1
                || fit.randomization_statistics.len() > 32
                || fit.randomization_statistics.iter().map(|(unit, _)| unit.as_str()).collect::<Vec<_>>()
                    != query.units.iter().map(String::as_str).collect::<std::collections::BTreeSet<_>>().into_iter().collect::<Vec<_>>()
                || fit.randomization_statistics.iter().any(|(_, statistic)| !statistic.is_finite() || *statistic < 0.0)
                || fit.randomization_p_value != fit.randomization_statistics.iter()
                    .find(|(unit, _)| unit == &query.treated_unit)
                    .map(|(_, observed)| fit.randomization_statistics.iter()
                        .filter(|(_, statistic)| statistic >= observed).count() as f64
                        / fit.randomization_statistics.len() as f64)))
        {
            return Err(IoError::Convert("invalid synthetic-control payload or fabricated interval".into()));
        }
    }
    if let Some(fit) = &result.synthetic_did {
        let crate::CausalQueryWire::SyntheticControl(query) = &result.query else {
            return Err(IoError::Convert("synthetic DiD result is attached to a different query".into()));
        };
        let donors: std::collections::BTreeSet<&str> = query.units.iter().map(String::as_str)
            .filter(|unit| *unit != query.treated_unit.as_str()).collect();
        let pre: std::collections::BTreeSet<i64> = query.periods.iter().copied()
            .filter(|period| *period < query.intervention_period).collect();
        let post: std::collections::BTreeSet<i64> = query.periods.iter().copied()
            .filter(|period| *period >= query.intervention_period).collect();
        #[allow(clippy::float_cmp, reason = "randomization null effect must equal the frozen sharp-null query value exactly")]
        if !query.difference_in_differences
            || result.estimate != Some(fit.effect)
            || result.standard_error.is_some() || result.interval_lower.is_some() || result.interval_upper.is_some()
            || !fit.effect.is_finite() || !fit.pre_treatment_rmse.is_finite() || fit.pre_treatment_rmse < 0.0
            || donors.len() < 2 || pre.len() < 2 || post.is_empty()
            || fit.n_donors != donors.len() || fit.n_pre_periods != pre.len() || fit.n_post_periods != post.len()
            || fit.donor_weights.iter().map(|(unit, _)| unit.as_str()).collect::<Vec<_>>() != donors.iter().copied().collect::<Vec<_>>()
            || fit.donor_weights.iter().any(|(_, weight)| !weight.is_finite() || *weight < 0.0)
            || (fit.donor_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() > 1e-8
            || fit.time_weights.iter().map(|(period, _)| *period).collect::<Vec<_>>() != pre.iter().copied().collect::<Vec<_>>()
            || fit.time_weights.iter().any(|(_, weight)| !weight.is_finite() || *weight < 0.0)
            || (fit.time_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() > 1e-8
            || (!query.uniform_unit_randomization && (
                fit.uncertainty != "point_only_no_interval"
                || fit.randomization_p_value.is_some() || fit.randomization_null_effect.is_some()
                || !fit.randomization_statistics.is_empty()))
            || (query.uniform_unit_randomization && (
                fit.randomization_null_effect.is_some_and(|effect|
                    !effect.is_finite() || effect != query.sharp_null_effect.unwrap_or(0.0))
                || (query.sharp_null_effect.is_some() && fit.randomization_null_effect.is_none())
                || !query.sharp_null_effect.unwrap_or(0.0).is_finite()
                ||
                fit.uncertainty != "point_only_with_exact_unit_randomization_p_value_no_interval"
                || fit.randomization_statistics.len() != donors.len() + 1
                || fit.randomization_statistics.len() > 32
                || fit.randomization_statistics.iter().map(|(unit, _)| unit.as_str()).collect::<Vec<_>>()
                    != query.units.iter().map(String::as_str).collect::<std::collections::BTreeSet<_>>().into_iter().collect::<Vec<_>>()
                || fit.randomization_statistics.iter().any(|(_, statistic)| !statistic.is_finite() || *statistic < 0.0)
                || fit.randomization_p_value != fit.randomization_statistics.iter()
                    .find(|(unit, _)| unit == &query.treated_unit)
                    .map(|(_, observed)| fit.randomization_statistics.iter()
                        .filter(|(_, statistic)| statistic >= observed).count() as f64
                        / fit.randomization_statistics.len() as f64)))
        {
            return Err(IoError::Convert("invalid synthetic DiD weights, support, or fabricated interval".into()));
        }
    }
    if let Some(curve) = &result.survival {
        let crate::CausalQueryWire::Survival(query) = &result.query else {
            return Err(IoError::Convert(
                "survival result section is attached to a different query".into(),
            ));
        };
        let point_only = curve.uncertainty == "point_only_no_interval"
            && curve.rmst_difference_interval.is_none()
            && curve.difference_at_tau_interval.is_none()
            && curve.bootstrap_replicates_requested.is_none()
            && curve.bootstrap_replicates_ok.is_none();
        let pointwise_bootstrap = curve.uncertainty == "subject_stratified_percentile_bootstrap_pointwise_95"
            && curve.bootstrap_replicates_requested.is_some_and(|n| (199..=100_000).contains(&n))
            && curve.bootstrap_replicates_ok.is_some_and(|ok| {
                let requested = curve.bootstrap_replicates_requested.unwrap_or(0);
                ok >= 199 && ok >= requested.saturating_sub(requested / 10) && ok <= requested
            })
            && curve.difference_at_tau_interval.is_some_and(|limits| {
                limits[0].is_finite() && limits[1].is_finite()
                    && -1.0 <= limits[0] && limits[0] <= limits[1] && limits[1] <= 1.0
            })
            && if query.target_cause.is_some() {
                curve.rmst_difference_interval.is_none()
            } else {
                curve.rmst_difference_interval.is_some_and(|limits| {
                    limits[0].is_finite() && limits[1].is_finite()
                        && -curve.tau <= limits[0] && limits[0] <= limits[1] && limits[1] <= curve.tau
                })
            };
        let band_valid = match (&curve.difference_band, &curve.band_unavailable_reason) {
            (Some(band), None) => {
                let requested = curve.bootstrap_replicates_requested.unwrap_or(0);
                query.delayed_entry.is_none() && query.censoring_columns.is_empty()
                    && pointwise_bootstrap && requested >= 399
                    && band.replicates_ok >= 399
                    && band.replicates_ok >= requested.saturating_sub(requested / 10)
                    && band.replicates_ok <= requested
                    && band.times == curve.times
                    && band.difference.len() == curve.times.len()
                    && band.lower.len() == curve.times.len()
                    && band.upper.len() == curve.times.len()
                    && band.difference.iter().zip(&curve.treated).zip(&curve.control)
                        .all(|((&difference, &treated), &control)| (difference - (treated - control)).abs() <= 1e-10)
                    && band.lower.iter().zip(&band.difference).zip(&band.upper)
                        .all(|((&lower, &difference), &upper)| {
                            lower.is_finite() && difference.is_finite() && upper.is_finite()
                                && -1.0 <= lower && lower <= difference && difference <= upper && upper <= 1.0
                        })
            }
            (None, Some(reason)) => pointwise_bootstrap && !reason.trim().is_empty(),
            (None, None) => true,
            (Some(_), Some(_)) => false,
        };
        let scalar_licensed = query.target_cause.is_none()
            && query.delayed_entry.is_none()
            && query.censoring_columns.is_empty()
            && pointwise_bootstrap
            && curve.bootstrap_replicates_ok.is_some_and(|ok| ok >= 299)
            && curve.assignment_counts.is_some_and(|[control, treated]| {
                graphless_data::LICENSES.iter().any(|row| {
                    row.family == "survival"
                        && row.design == "two_arm_individual_randomized"
                        && row.method == "arm_stratified_subject_bootstrap_product_limit"
                        && row.inference_claim == "rmst_and_horizon_survival_pointwise_95_percentile_intervals"
                        && row.assignment_unit == "unit"
                        && control >= row.min_assignment_units_per_arm
                        && treated >= row.min_assignment_units_per_arm
                        && control.saturating_add(treated) >= row.min_rows
                        && row.min_reported_intervals <= 2
                })
            });
        #[allow(clippy::float_cmp, reason = "survival curve endpoints and tau must equal the frozen query/boundary values exactly")]
        if result.estimate.is_some()
            || result.standard_error.is_some()
            || result.interval_lower.is_some()
            || result.interval_upper.is_some()
            || curve.censoring_survival_provenance.as_deref()
                != (!query.censoring_columns.is_empty()).then_some("caller_supplied_fixed_not_fitted_or_verified")
            || !(point_only || pointwise_bootstrap)
            || curve.graphless_support_status.as_deref() != scalar_licensed.then_some("licensed")
            || (curve.assignment_counts.is_some_and(|counts| counts.contains(&0)))
            || !band_valid
            || curve.tau != query.tau
            || curve.target_cause != query.target_cause
            || curve.times.len() < 2
            || curve.control.len() != curve.times.len()
            || curve.treated.len() != curve.times.len()
            || curve.times[0] != 0.0
            || curve.times.last() != Some(&curve.tau)
            || curve.times.windows(2).any(|w| !w[0].is_finite() || w[0] >= w[1])
            || !curve.tau.is_finite()
            || curve
                .control
                .iter()
                .chain(curve.treated.iter())
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || curve.minimum_event_risk_set == Some(0)
            || (query.target_cause.is_some()
                && (curve.rmst_control.is_some()
                    || curve.rmst_treated.is_some()
                    || curve.control[0] != 0.0
                    || curve.treated[0] != 0.0))
            || (query.target_cause.is_none()
                && (curve.rmst_control.is_none()
                    || curve.rmst_treated.is_none()
                    || curve.control[0] != 1.0
                    || curve.treated[0] != 1.0))
            || [curve.control.as_slice(), curve.treated.as_slice()].into_iter().any(|arm| {
                arm.windows(2)
                    .any(|w| if query.target_cause.is_some() { w[0] > w[1] } else { w[0] < w[1] })
            })
            || curve.rmst_control.is_some_and(|v| !v.is_finite() || !(0.0..=curve.tau).contains(&v))
            || curve.rmst_treated.is_some_and(|v| !v.is_finite() || !(0.0..=curve.tau).contains(&v))
        {
            return Err(IoError::Convert(
                "invalid survival payload or fabricated uncertainty".into(),
            ));
        }
    }
    if result.estimate.is_some() && !licenses_scalar_estimate(&result.identification.status) {
        return Err(IoError::Convert(
            "identification status does not license a scalar estimate".into(),
        ));
    }
    if result.standard_error.is_some_and(|se| !se.is_finite() || se < 0.0) {
        return Err(IoError::Convert(
            "analysis standard error must be finite and nonnegative".into(),
        ));
    }
    if let Some(response) = &result.response {
        validate_response_result(response, variable_names.len())?;
    }
    if let Some(posterior) = &result.posterior_artifact {
        crate::decode_causal_posterior_bytes(posterior)?;
    }
    if let Some(grid) = &result.mediation_grid {
        validate_mediation_grid(grid, variable_names.len())?;
    }
    if let Some(unit_effects) = &result.unit_effects {
        validate_unit_effects(unit_effects)?;
    }
    validate_cate_and_learner_metrics(result)?;
    if let Some(structural) = &result.structural_response {
        validate_structural_response(structural, variable_names.len())?;
    }
    Ok(())
}

fn validate_interventional_distribution(result: &AnalysisResultWire) -> Result<(), IoError> {
    let query = match &result.query {
        CausalQueryWire::Distribution(query) => query,
        _ if result.interventional_distribution.is_some() => {
            return Err(IoError::Convert(
                "interventional distribution atoms require a distribution query".into(),
            ));
        }
        _ => return Ok(()),
    };
    let Some(distribution) = &result.interventional_distribution else {
        // Older artifacts remain readable. The independent contract consumer
        // reports the missing executable result payload as a dependency.
        return Ok(());
    };
    if distribution.atoms.is_empty() {
        return Err(IoError::Convert("interventional distribution has no atoms".into()));
    }
    let expected_outcomes = &query.outcomes;
    let expected_conditioning = &query.conditioning;
    let mut groups: Vec<(Vec<(u32, crate::ValueWire)>, f64)> = Vec::new();
    let mut numeric_mean = 0.0;
    for atom in &distribution.atoms {
        if !atom.probability.is_finite()
            || atom.probability < 0.0
            || atom.outcomes.iter().map(|(id, _)| *id).collect::<Vec<_>>() != *expected_outcomes
            || atom.conditioning.iter().map(|(id, _)| *id).collect::<Vec<_>>()
                != *expected_conditioning
        {
            return Err(IoError::Convert(
                "interventional distribution atom does not match its query or has invalid mass"
                    .into(),
            ));
        }
        if atom
            .outcomes
            .iter()
            .chain(atom.conditioning.iter())
            .any(|(_, value)| matches!(value, crate::ValueWire::Float64(v) if !v.is_finite()))
        {
            return Err(IoError::Convert(
                "interventional distribution atom contains a non-finite value".into(),
            ));
        }
        if expected_outcomes.len() == 1 {
            if let Some(value) = atom.outcomes[0].1.to_value().as_f64() {
                numeric_mean += value * atom.probability;
            } else {
                numeric_mean = f64::NAN;
            }
        } else {
            numeric_mean = f64::NAN;
        }
        if let Some((_, mass)) =
            groups.iter_mut().find(|(conditioning, _)| *conditioning == atom.conditioning)
        {
            *mass += atom.probability;
        } else {
            groups.push((atom.conditioning.clone(), atom.probability));
        }
        if distribution
            .atoms
            .iter()
            .filter(|other| {
                other.conditioning == atom.conditioning && other.outcomes == atom.outcomes
            })
            .count()
            != 1
        {
            return Err(IoError::Convert(
                "interventional distribution contains duplicate atoms".into(),
            ));
        }
    }
    if groups.iter().any(|(_, mass)| (mass - 1.0).abs() > 1e-9) {
        return Err(IoError::Convert(
            "interventional distribution atom masses must sum to one per conditioning assignment"
                .into(),
        ));
    }
    if expected_conditioning.is_empty()
        && groups.len() == 1
        && expected_outcomes.len() == 1
        && numeric_mean.is_finite()
        && result.estimate.is_some_and(|estimate| {
            (estimate - numeric_mean).abs() > 1e-9 * (1.0 + numeric_mean.abs())
        })
    {
        return Err(IoError::Convert(
            "interventional distribution atoms disagree with the reported mean".into(),
        ));
    }
    Ok(())
}

fn validate_structural_response(
    structural: &StructuralResponseMixtureWire,
    n_variables: usize,
) -> Result<(), IoError> {
    if let Some(interval) = &structural.identified_set_interval {
        identified_set_interval_from_wire(interval)?;
    }
    if !(0.0..=1.0).contains(&structural.subsampled_out_mass) {
        return Err(IoError::Convert(
            "structural subsampled_out_mass must be a fraction in [0, 1]".into(),
        ));
    }
    let total = structural.identified_mass
        + structural.unidentified_mass
        + structural.unevaluable_mass
        + structural.subsampled_out_mass;
    if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
        return Err(IoError::Convert("structural masses must sum to one".into()));
    }
    if !structural.atoms.is_empty() {
        let weight: f64 = structural.atoms.iter().map(|atom| atom.weight).sum();
        if !weight.is_finite() || (weight - 1.0).abs() > 1e-9 {
            return Err(IoError::Convert("inconsistent atom totals".into()));
        }
    }
    for atom in &structural.atoms {
        if !atom.weight.is_finite() || atom.weight < 0.0 {
            return Err(IoError::Convert(
                "structural atom weight must be finite and nonnegative".into(),
            ));
        }
        if let Some(value) = &atom.value {
            crate::response_value_from_wire(value)?;
        }
        if let Some(response) = &atom.response {
            crate::causal_artifact::validate_response_result(response, n_variables)?;
        }
        if let Some(posterior) = &atom.posterior_artifact {
            crate::decode_causal_posterior_bytes(posterior)?;
        }
    }
    Ok(())
}

/// A point scalar is licensed only when identification does not refuse the estimand.
/// `not_identified` never licenses one; partial / graph-dependent mixtures may still
/// publish a conditional or retained identified-mass summary.
fn licenses_scalar_estimate(status: &str) -> bool {
    status != "not_identified"
}

fn validate_mediation_grid(
    grid: &TemporalMediationGridWire,
    base_variable_count: usize,
) -> Result<(), IoError> {
    if grid.slices.is_empty() {
        return Err(IoError::Convert("mediation grid must contain at least one slice".into()));
    }
    let mut horizons = std::collections::BTreeSet::new();
    for slice in &grid.slices {
        if slice.horizon == 0 || !horizons.insert(slice.horizon) {
            return Err(IoError::Convert(
                "mediation grid horizons must be positive and unique".into(),
            ));
        }
        if slice.method.trim().is_empty() {
            return Err(IoError::Convert("mediation grid method must be non-blank".into()));
        }
        for node in &slice.adjustment {
            if node.variable as usize >= base_variable_count {
                return Err(IoError::Convert(
                    "mediation grid adjustment names an unknown base variable".into(),
                ));
            }
        }
        let optionals = [slice.total, slice.direct, slice.mediated];
        if !slice.effect.is_finite() || optionals.iter().any(|v| v.is_some_and(|x| !x.is_finite()))
        {
            return Err(IoError::Convert(
                "mediation grid effects must be finite when present".into(),
            ));
        }
        if let Some([lower, upper]) = slice.identified_set {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(IoError::Convert(
                    "mediation grid identified set must be finite and ordered".into(),
                ));
            }
        }
        validate_mediation_uncertainty(&slice.uncertainty)?;
    }
    Ok(())
}

fn validate_mediation_uncertainty(
    uncertainty: &TemporalMediationUncertaintyWire,
) -> Result<(), IoError> {
    match uncertainty {
        TemporalMediationUncertaintyWire::FrequentistPointwise { standard_error } => {
            if standard_error.is_some_and(|se| !se.is_finite() || se < 0.0) {
                return Err(IoError::Convert(
                    "mediation grid standard error must be finite and nonnegative".into(),
                ));
            }
        }
        TemporalMediationUncertaintyWire::BayesianPointwise {
            requested,
            total,
            direct,
            mediated,
            n_draws,
            backend,
        } => {
            for summary in [requested, total, direct, mediated] {
                if ![summary.mean, summary.standard_deviation, summary.q025, summary.q975]
                    .iter()
                    .all(|v| v.is_finite())
                    || summary.standard_deviation < 0.0
                    || summary.q025 > summary.q975
                {
                    return Err(IoError::Convert(
                        "mediation grid posterior summary must be finite and ordered".into(),
                    ));
                }
            }
            if *n_draws == 0 || backend.trim().is_empty() {
                return Err(IoError::Convert(
                    "mediation grid Bayesian uncertainty needs draws and a backend".into(),
                ));
            }
        }
        TemporalMediationUncertaintyWire::Unavailable => {}
    }
    Ok(())
}

fn validate_unit_effects(unit_effects: &UnitEffectsWire) -> Result<(), IoError> {
    if unit_effects.effects.is_empty()
        || unit_effects.effects.iter().any(|effect| !effect.is_finite())
    {
        return Err(IoError::Convert("unit effects must be a non-empty finite vector".into()));
    }
    let n = unit_effects.effects.len();
    if let Some(intervals) = &unit_effects.intervals {
        if intervals.lower.len() != n
            || intervals.upper.len() != n
            || intervals.lower.iter().any(|v| !v.is_finite())
            || intervals.upper.iter().any(|v| !v.is_finite())
            || intervals.lower.iter().zip(&intervals.upper).any(|(lo, hi)| lo > hi)
            || !(0.0..=1.0).contains(&intervals.level)
            || !intervals.level.is_finite()
            || intervals.method.trim().is_empty()
        {
            return Err(IoError::Convert(
                "unit effect intervals must match effects length with finite ordered bounds, \
                 level in [0, 1], and a non-blank method"
                    .into(),
            ));
        }
    }
    if unit_effects.extrapolative.as_ref().is_some_and(|flags| flags.len() != n) {
        return Err(IoError::Convert(
            "unit effect extrapolative flags must match effects length".into(),
        ));
    }
    Ok(())
}

fn validate_cate_and_learner_metrics(result: &AnalysisResultWire) -> Result<(), IoError> {
    if result.cate.is_none() && (result.cate_se.is_some() || result.cate_leaf_dispersion.is_some())
    {
        return Err(IoError::Convert(
            "cate standard errors and leaf dispersion require cate point predictions".into(),
        ));
    }
    if let (Some(cate), Some(dispersion)) = (&result.cate, &result.cate_leaf_dispersion) {
        if dispersion.len() != cate.len() || dispersion.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(IoError::Convert(
                "cate leaf dispersion must match cate length and be finite nonnegative".into(),
            ));
        }
    }
    match (&result.cate, &result.cate_se) {
        (None, _) => {}
        (Some(cate), cate_se) => {
            if cate.is_empty() || cate.iter().any(|v| !v.is_finite()) {
                return Err(IoError::Convert(
                    "cate predictions must be a non-empty finite vector".into(),
                ));
            }
            if let Some(se) = cate_se {
                if se.len() != cate.len() || se.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(IoError::Convert(
                        "cate standard errors must match cate length and be finite nonnegative"
                            .into(),
                    ));
                }
            }
        }
    }
    if result.outcome_oof_r2.is_some_and(|v| !v.is_finite()) {
        return Err(IoError::Convert("outcome_oof_r2 must be finite when present".into()));
    }
    if result.treatment_oof_logloss.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return Err(IoError::Convert(
            "treatment_oof_logloss must be finite and nonnegative".into(),
        ));
    }
    if result.crossfit_folds.is_some_and(|folds| folds < 2) {
        return Err(IoError::Convert("crossfit_folds must be at least two when present".into()));
    }
    if result.learner_provenance.iter().any(|(spec, implementation, version)| {
        spec.trim().is_empty() || implementation.trim().is_empty() || version.trim().is_empty()
    }) {
        return Err(IoError::Convert("learner provenance entries must be non-blank".into()));
    }
    Ok(())
}

fn validate_identification_namespace(
    identification: &IdentificationResultWire,
    count: usize,
) -> Result<(), IoError> {
    crate::causal_artifact::validate_query_ids(&identification.query, count)?;
    let ids = identification
        .arena
        .var_sets
        .iter()
        .flatten()
        .copied()
        .chain(identification.arena.interventions.iter().flatten().map(|value| value.variable))
        .chain(
            identification
                .estimands
                .iter()
                .flat_map(|estimand| {
                    estimand
                        .adjustment_set
                        .iter()
                        .chain(&estimand.instruments)
                        .chain(&estimand.mediators)
                })
                .copied(),
        )
        .chain(
            identification
                .estimands
                .iter()
                .filter_map(|estimand| estimand.rd_design.as_ref())
                .map(|design| design.running_variable),
        )
        .chain(identification.hedge.iter().flat_map(crate::HedgeCertificateWire::variables));
    if ids.into_iter().any(|id| id as usize >= count) {
        return Err(IoError::Convert(
            "identification variable is outside its declared namespace".into(),
        ));
    }
    Ok(())
}

fn validate_temporal_namespace(
    variables: &[crate::HorizonAdjustmentNodeWire],
    base_count: usize,
) -> Result<(), IoError> {
    let mut keys = std::collections::BTreeSet::new();
    if variables.is_empty() {
        return Err(IoError::Convert("empty temporal variable namespace".into()));
    }
    for node in variables {
        if node.variable as usize >= base_count || !keys.insert((node.variable, node.offset)) {
            return Err(IoError::Convert(
                "temporal namespace contains an unknown base variable or duplicate node".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> AnalysisResultWire {
        let query = serde_json::json!({"response": {
            "functional": {"average_derivative": {"outcome": 1, "treatment": 0, "weighting": "observed"}},
            "target_population": "all_observed", "observation": "complete", "observation_assumptions": []
        }});
        serde_json::from_value(serde_json::json!({
            "query": query,
            "identification": {"status": "nonparametrically_identified", "query": query,
                "estimands": [], "arena": {"var_sets": [], "interventions": [], "lists": [], "nodes": []},
                "derivation": [], "required_assumptions": [], "diagnostics": [], "candidates_examined": 0, "sets_returned": 0},
            "estimate": 1.0, "standard_error": 0.2, "assumptions": [], "diagnostics": [], "refutations": [],
            "response": null, "posterior_artifact": null, "mediation_grid": null, "structural_response": null
        })).unwrap()
    }

    fn distribution_fixture() -> AnalysisResultWire {
        let query = antecedent_core::CausalQuery::Distribution(
            antecedent_core::InterventionalDistributionQuery::new(
                antecedent_core::VariableId::from_raw(1),
                [antecedent_core::Intervention::set(
                    antecedent_core::VariableId::from_raw(0),
                    antecedent_core::Value::f64(1.0),
                )],
            ),
        );
        let query = crate::causal_query_to_wire(&query).unwrap();
        let mut result = fixture();
        result.query = query.clone();
        result.identification.query = query;
        result.estimate = Some(0.7);
        result.interventional_distribution = Some(InterventionalDistributionWire {
            atoms: vec![
                DistributionAtomWire {
                    outcomes: vec![(1, crate::ValueWire::Float64(0.0))],
                    conditioning: Vec::new(),
                    probability: 0.3,
                },
                DistributionAtomWire {
                    outcomes: vec![(1, crate::ValueWire::Float64(1.0))],
                    conditioning: Vec::new(),
                    probability: 0.7,
                },
            ],
        });
        result
    }

    #[test]
    fn panel_did_result_artifact_round_trips_design_uncertainty_semantics() {
        let domain = antecedent_core::PanelDidQuery::new(
            antecedent_core::VariableId::from_raw(1),
            [true, true, true, true, false, false, false, false],
            [false, true, false, true, false, true, false, true],
            ["a", "a", "b", "b", "c", "c", "d", "d"].map(std::sync::Arc::<str>::from),
            ["x", "x", "y", "y", "z", "z", "w", "w"].map(std::sync::Arc::<str>::from),
        );
        let query =
            crate::causal_query_to_wire(&antecedent_core::CausalQuery::PanelDid(domain)).unwrap();
        let mut result = fixture();
        result.query = query.clone();
        result.identification.query = query;
        result.estimate = Some(2.0);
        result.standard_error = None;
        result.panel_did = Some(PanelDidWire {
            effect: 2.0,
            standard_error: Some(0.4),
            interval_95: None,
            treated_subjects: 2,
            comparison_subjects: 2,
            clusters: 4,
            uncertainty: "cluster_robust_standard_error_no_interval".into(),
            graphless_support_status: None,
            event_time_effects: vec![],
            event_time_intervals_95: vec![],
            augmented: None,
        });
        let encoded = encode_analysis_result_artifact(
            &result,
            vec!["treatment".into(), "outcome".into()],
            "panel-did-result",
        )
        .unwrap();
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn repeated_cross_section_did_result_round_trips_without_interval() {
        let domain = antecedent_core::PanelDidQuery::repeated_cross_section(
            antecedent_core::VariableId::from_raw(1),
            [false, false, false, false, true, true, true, true],
            [false, false, true, true, false, false, true, true],
            (0..8).map(|i| std::sync::Arc::<str>::from(format!("s{i}"))).collect::<Vec<_>>(),
            (0..8).map(|i| std::sync::Arc::<str>::from(format!("c{i}"))).collect::<Vec<_>>(),
        );
        let query =
            crate::causal_query_to_wire(&antecedent_core::CausalQuery::PanelDid(domain)).unwrap();
        let mut result = fixture();
        result.query = query.clone();
        result.identification.query = query;
        result.estimate = Some(2.0);
        result.standard_error = None;
        result.panel_did = Some(PanelDidWire {
            effect: 2.0,
            standard_error: Some((16.0_f64 / 7.0).sqrt()),
            interval_95: None,
            treated_subjects: 4,
            comparison_subjects: 4,
            clusters: 8,
            uncertainty: "cluster_robust_standard_error_no_interval".into(),
            graphless_support_status: None,
            event_time_effects: vec![],
            event_time_intervals_95: vec![],
            augmented: None,
        });
        let encoded = encode_analysis_result_artifact(
            &result,
            vec!["treatment".into(), "outcome".into()],
            "rcs-did-result",
        )
        .unwrap();
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);
        result.interval_lower = Some(0.0);
        assert!(
            encode_analysis_result_artifact(
                &result,
                vec!["treatment".into(), "outcome".into()],
                "rcs-invalid-interval"
            )
            .is_err()
        );
    }

    #[test]
    fn staggered_did_result_round_trips_selected_cohort_counts_without_interval() {
        let mut units = Vec::new();
        let mut clusters = Vec::new();
        let mut periods = Vec::new();
        let mut cohorts = Vec::new();
        for unit in 0..10 {
            for period in 1..=4 {
                units.push(std::sync::Arc::<str>::from(format!("u{unit}")));
                clusters.push(std::sync::Arc::<str>::from(format!("c{unit}")));
                periods.push(period);
                cohorts.push(if unit < 4 { 0 } else if unit < 8 { 3 } else { 4 });
            }
        }
        let query = crate::causal_query_to_wire(&antecedent_core::CausalQuery::PanelDid(
            antecedent_core::PanelDidQuery::staggered_group_time(
                antecedent_core::VariableId::from_raw(1), units, clusters, periods, cohorts, 3, 4,
            ),
        )).unwrap();
        let mut result = fixture();
        result.query = query.clone();
        result.identification.query = query;
        result.estimate = Some(4.0);
        result.standard_error = None;
        result.panel_did = Some(PanelDidWire {
            effect: 4.0,
            standard_error: Some(0.5),
            interval_95: None,
            treated_subjects: 4,
            comparison_subjects: 4,
            clusters: 8,
            uncertainty: "cluster_robust_standard_error_no_interval".into(),
            graphless_support_status: None,
            event_time_effects: vec![],
            event_time_intervals_95: vec![],
            augmented: None,
        });
        let encoded = encode_analysis_result_artifact(
            &result, vec!["treatment".into(), "outcome".into()], "staggered-did-result",
        ).unwrap();
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn distribution_atoms_are_bound_to_query_and_reported_mean() {
        let mut result = distribution_fixture();
        validate_interventional_distribution(&result).unwrap();

        result.interventional_distribution.as_mut().unwrap().atoms[0].probability = 0.2;
        assert!(validate_interventional_distribution(&result).is_err());

        let mut result = distribution_fixture();
        result.interventional_distribution.as_mut().unwrap().atoms[1].outcomes[0].0 = 0;
        assert!(validate_interventional_distribution(&result).is_err());

        let mut result = distribution_fixture();
        result.estimate = Some(0.6);
        assert!(validate_interventional_distribution(&result).is_err());

        let mut result = distribution_fixture();
        result.interventional_distribution.as_mut().unwrap().atoms[1].probability = f64::NAN;
        assert!(validate_interventional_distribution(&result).is_err());
    }

    #[test]
    fn composite_roundtrip_validates_names_ids_and_nested_posterior() {
        let result = fixture();
        let names = vec!["a".into(), "y".into()];
        let artifact = encode_analysis_result_artifact(&result, names.clone(), "review").unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (_, header, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);
        assert_eq!(header.variable_names, names);
        assert!(encode_analysis_result_artifact(&result, vec!["a".into()], "review").is_err());
        assert!(
            encode_analysis_result_artifact(&result, vec!["a".into(), "a".into()], "review")
                .is_err()
        );
        let mut invalid = result;
        invalid.posterior_artifact = Some(vec![1, 2, 3]);
        assert!(encode_analysis_result_artifact(&invalid, names, "review").is_err());
    }
    #[test]
    fn temporal_identification_uses_its_own_namespace_and_preserves_offsets() {
        let mut result = fixture();
        // Base query remains ids 0,1; its unfolded certificate uses ids 4,5.
        let mut wire = serde_json::to_value(&result.identification).unwrap();
        wire["query"]["response"]["functional"]["average_derivative"]["treatment"] = 4.into();
        wire["query"]["response"]["functional"]["average_derivative"]["outcome"] = 5.into();
        result.identification = serde_json::from_value(wire).unwrap();
        let variables: Vec<_> = (-2..=0)
            .flat_map(|offset| {
                (0..2).map(move |variable| crate::HorizonAdjustmentNodeWire { variable, offset })
            })
            .collect();
        result.identification_variables = Some(variables.clone());
        result.temporal_identification.push(TemporalIdentificationWire {
            horizon: 1,
            variables,
            identification: result.identification.clone(),
        });
        let names = vec!["x".into(), "y".into()];
        let artifact = encode_analysis_result_artifact(&result, names.clone(), "temporal").unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);
        result.identification_variables.as_mut().unwrap()[5].variable = 2;
        assert!(encode_analysis_result_artifact(&result, names.clone(), "invalid").is_err());
        result.identification_variables = None;
        assert!(encode_analysis_result_artifact(&result, names, "missing").is_err());
    }
    fn interval() -> antecedent_estimate::IdentifiedSetInterval {
        antecedent_estimate::IdentifiedSetInterval {
            level: 0.9,
            lower: 0.41,
            upper: 1.12,
            bound_lower: 0.52,
            bound_upper: 0.98,
            lower_se: 0.061,
            upper_se: 0.083,
            critical_value: 1.31,
            width_retained: true,
            completions: 2,
            replicates: 199,
            method: antecedent_estimate::IdentifiedSetIntervalMethod::ImbensManskiSharedBlock,
            truncated: false,
        }
    }

    #[test]
    fn every_identified_set_construction_has_its_own_wire_tag() {
        let mut tags = Vec::new();
        for method in antecedent_estimate::IdentifiedSetIntervalMethod::ALL {
            let original = antecedent_estimate::IdentifiedSetInterval { method, ..interval() };
            let wire = identified_set_interval_to_wire(&original).unwrap();
            assert_eq!(identified_set_interval_from_wire(&wire).unwrap().method, method);
            tags.push(serde_json::to_value(wire.method).unwrap());
        }
        tags.dedup();
        assert_eq!(tags.len(), antecedent_estimate::IdentifiedSetIntervalMethod::ALL.len());
    }

    #[test]
    fn truncated_identified_set_interval_round_trips_and_defaults_to_false() {
        let original = antecedent_estimate::IdentifiedSetInterval { truncated: true, ..interval() };
        let wire = identified_set_interval_to_wire(&original).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["truncated"], true);
        assert_eq!(identified_set_interval_from_wire(&wire).unwrap(), original);
        // An untruncated interval omits the field, and a body without it decodes
        // as untruncated.
        let plain = identified_set_interval_to_wire(&interval()).unwrap();
        let json = serde_json::to_value(&plain).unwrap();
        assert!(json.get("truncated").is_none());
        let decoded: IdentifiedSetIntervalWire = serde_json::from_value(json).unwrap();
        assert!(!identified_set_interval_from_wire(&decoded).unwrap().truncated);
    }

    fn with_structural(interval: Option<IdentifiedSetIntervalWire>) -> AnalysisResultWire {
        let mut result = fixture();
        result.structural_response = Some(StructuralResponseMixtureWire {
            weight_basis: StructuralWeightBasisWire::CompletionEnumeration,
            atoms: Vec::new(),
            identified_mass: 1.0,
            unidentified_mass: 0.0,
            unevaluable_mass: 0.0,
            subsampled_out_mass: 0.0,
            identified_set: None,
            identified_set_interval: interval,
            conditional_on_identified: None,
            full_mass_scope: true,
            truncated_atoms: 0,
        });
        result
    }

    #[test]
    fn subsampled_out_mass_is_optional_on_the_wire_and_round_trips() {
        let names: Vec<String> = vec!["a".into(), "y".into()];
        // Zero is omitted, and a body without the field decodes as zero.
        let plain = with_structural(None);
        let json = serde_json::to_value(&plain).unwrap();
        assert!(json["structural_response"].get("subsampled_out_mass").is_none());
        let decoded: AnalysisResultWire = serde_json::from_value(json).unwrap();
        assert!(decoded.structural_response.unwrap().subsampled_out_mass.abs() < f64::EPSILON);

        let mut result = with_structural(None);
        if let Some(structural) = result.structural_response.as_mut() {
            structural.identified_mass = 0.4;
            structural.subsampled_out_mass = 0.6;
        }
        let artifact = encode_analysis_result_artifact(&result, names.clone(), "sub").unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);

        if let Some(structural) = result.structural_response.as_mut() {
            structural.subsampled_out_mass = 1.5;
        }
        assert!(encode_analysis_result_artifact(&result, names, "bad").is_err());
    }

    #[test]
    fn identified_set_interval_round_trips_and_validates() {
        let original = interval();
        let wire = identified_set_interval_to_wire(&original).unwrap();
        assert_eq!(identified_set_interval_from_wire(&wire).unwrap(), original);
        let result = with_structural(Some(wire.clone()));
        let names = vec!["a".into(), "y".into()];
        let artifact = encode_analysis_result_artifact(&result, names.clone(), "set").unwrap();
        assert_eq!(artifact.manifest.format_version, STABLE_FORMAT);
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);
        let restored = decoded.structural_response.unwrap().identified_set_interval.unwrap();
        assert_eq!(identified_set_interval_from_wire(&restored).unwrap(), original);
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(
            json["structural_response"]["identified_set_interval"]["method"],
            "imbens_manski_shared_block"
        );

        for corrupt in [
            IdentifiedSetIntervalWire { level: 1.0, ..wire.clone() },
            IdentifiedSetIntervalWire { lower: 1.5, ..wire.clone() },
            IdentifiedSetIntervalWire { upper_se: -0.1, ..wire.clone() },
            IdentifiedSetIntervalWire { critical_value: f64::NAN, ..wire.clone() },
            IdentifiedSetIntervalWire { replicates: 1, ..wire.clone() },
            IdentifiedSetIntervalWire { completions: 0, ..wire.clone() },
        ] {
            assert!(identified_set_interval_from_wire(&corrupt).is_err());
            assert!(
                encode_analysis_result_artifact(
                    &with_structural(Some(corrupt)),
                    names.clone(),
                    "bad"
                )
                .is_err()
            );
        }
    }

    #[test]
    fn format_0_4_structural_result_migrates_without_an_interval() {
        // A 0.4 writer had no interval field: the body omits it entirely.
        let result = with_structural(None);
        let body = serde_json::to_value(&result).unwrap();
        assert!(body["structural_response"].get("identified_set_interval").is_none());
        let mut artifact =
            encode_analysis_result_artifact(&result, vec!["a".into(), "y".into()], "old").unwrap();
        artifact.manifest.format_version = crate::FormatVersion { major: 0, minor: 4 };
        artifact.manifest.minimum_reader_version = crate::FormatVersion { major: 0, minor: 4 };
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (migrated, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
        assert_eq!(decoded, result);
        assert!(decoded.structural_response.unwrap().identified_set_interval.is_none());
    }

    #[test]
    fn function_valued_result_has_no_scalar_and_survives_json_bridge() {
        let mut result = fixture();
        assert_eq!(result.estimate, Some(1.0)); // Legacy numeric scalar remains readable.
        result.estimate = None;
        result.standard_error = None;
        let json = serde_json::to_value(&result).unwrap();
        assert!(json["estimate"].is_null());
        let restored: AnalysisResultWire = serde_json::from_value(json).unwrap();
        let artifact =
            encode_analysis_result_artifact(&restored, vec!["x".into(), "y".into()], "functional")
                .unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        assert_eq!(decode_analysis_result_artifact(&bytes).unwrap().2, result);
        result.estimate = Some(f64::NAN);
        assert!(
            encode_analysis_result_artifact(&result, vec!["x".into(), "y".into()], "invalid")
                .is_err()
        );
    }

    #[test]
    fn validate_result_refuses_wrong_query_ids_and_keeps_the_valid_counterpart() {
        let names = vec!["a".into(), "y".into()];
        let ok = fixture();
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "ok").is_ok());
        let mut wrong = ok;
        let mut query = serde_json::to_value(&wrong.identification.query).unwrap();
        query["response"]["functional"]["average_derivative"]["treatment"] = 1.into();
        query["response"]["functional"]["average_derivative"]["outcome"] = 0.into();
        wrong.identification.query = serde_json::from_value(query).unwrap();
        let err = encode_analysis_result_artifact(&wrong, names, "wrong").unwrap_err();
        assert!(err.to_string().contains("enclosing query"), "{err}");
    }

    fn pulse_query_wire(horizon: u32) -> crate::CausalQueryWire {
        use crate::query_wire::causal_query_to_wire;
        use antecedent_core::{CausalQuery, TemporalEffectQuery, VariableId};
        causal_query_to_wire(&CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_horizon_steps(horizon),
        ))
        .unwrap()
    }

    #[test]
    fn validate_result_refuses_changed_horizon_and_keeps_the_matching_certificate() {
        let names = vec!["x".into(), "y".into()];
        let mut ok = fixture();
        ok.query = pulse_query_wire(1);
        ok.identification.query = ok.query.clone();
        ok.temporal_identification.push(TemporalIdentificationWire {
            horizon: 1,
            variables: vec![
                crate::HorizonAdjustmentNodeWire { variable: 0, offset: -1 },
                crate::HorizonAdjustmentNodeWire { variable: 1, offset: 0 },
            ],
            identification: ok.identification.clone(),
        });
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "horizon-ok").is_ok());
        let mut stale = ok;
        stale.temporal_identification[0].horizon = 2;
        let err = encode_analysis_result_artifact(&stale, names, "horizon-stale").unwrap_err();
        assert!(err.to_string().contains("horizon"), "{err}");
    }

    #[test]
    fn validate_result_refuses_inconsistent_atom_totals_and_keeps_unit_mass() {
        let names = vec!["a".into(), "y".into()];
        let mut ok = with_structural(None);
        ok.structural_response.as_mut().unwrap().atoms.push(StructuralResponseAtomWire {
            graph_key: 1,
            weight: 1.0,
            identification_status: crate::IdentificationStatusWire::NonparametricallyIdentified,
            value: None,
            posterior_artifact: None,
            response: None,
        });
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "atoms-ok").is_ok());
        let mut bad = ok;
        bad.structural_response.as_mut().unwrap().atoms[0].weight = 0.4;
        let err = encode_analysis_result_artifact(&bad, names, "atoms-bad").unwrap_err();
        assert!(err.to_string().contains("atom totals"), "{err}");
    }

    #[test]
    fn validate_result_refuses_inconsistent_masses_and_keeps_unit_mass() {
        let names = vec!["a".into(), "y".into()];
        let ok = with_structural(None);
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "mass-ok").is_ok());
        let mut bad = ok;
        bad.structural_response.as_mut().unwrap().identified_mass = 0.5;
        let err = encode_analysis_result_artifact(&bad, names, "mass-bad").unwrap_err();
        assert!(err.to_string().contains("masses must sum to one"), "{err}");
    }

    #[test]
    fn validate_result_refuses_scalar_estimate_under_not_identified() {
        let names = vec!["a".into(), "y".into()];
        let mut result = fixture();
        result.identification.status = "not_identified".into();
        assert_eq!(result.estimate, Some(1.0));
        let err = encode_analysis_result_artifact(&result, names.clone(), "uid").unwrap_err();
        assert!(err.to_string().contains("does not license a scalar estimate"), "{err}");
        result.estimate = None;
        result.standard_error = None;
        assert!(encode_analysis_result_artifact(&result, names, "uid-ok").is_ok());
    }

    fn mediation_slice() -> TemporalMediationSliceWire {
        TemporalMediationSliceWire {
            horizon: 1,
            identification_status: crate::IdentificationStatusWire::NonparametricallyIdentified,
            method: "front_door".into(),
            adjustment: vec![],
            effect: 0.5,
            total: Some(0.5),
            direct: Some(0.2),
            mediated: Some(0.3),
            uncertainty: TemporalMediationUncertaintyWire::Unavailable,
            identified_set: None,
            diagnostics: vec![],
        }
    }

    #[test]
    fn validate_result_refuses_malformed_mediation_grid() {
        let names = vec!["a".into(), "y".into()];
        let mut ok = fixture();
        ok.mediation_grid = Some(TemporalMediationGridWire {
            slices: vec![mediation_slice()],
            joint_posterior: false,
        });
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "grid-ok").is_ok());

        let mut bad = ok.clone();
        bad.mediation_grid.as_mut().unwrap().slices[0].effect = f64::NAN;
        let err = encode_analysis_result_artifact(&bad, names.clone(), "grid-nan").unwrap_err();
        assert!(err.to_string().contains("mediation grid"), "{err}");

        let mut inverted = ok;
        inverted.mediation_grid.as_mut().unwrap().slices[0].identified_set = Some([1.0, 0.0]);
        let err = encode_analysis_result_artifact(&inverted, names, "grid-set").unwrap_err();
        assert!(err.to_string().contains("identified set"), "{err}");
    }

    #[test]
    fn validate_result_refuses_malformed_cate() {
        let names = vec!["a".into(), "y".into()];
        let mut ok = fixture();
        ok.cate = Some(vec![0.1, 0.2]);
        ok.cate_se = Some(vec![0.05, 0.04]);
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "cate-ok").is_ok());

        let mut bad_len = ok.clone();
        bad_len.cate_se = Some(vec![0.05]);
        let err = encode_analysis_result_artifact(&bad_len, names.clone(), "cate-len").unwrap_err();
        assert!(err.to_string().contains("cate"), "{err}");

        let mut bad_dispersion = ok.clone();
        bad_dispersion.cate_leaf_dispersion = Some(vec![0.05, -1.0]);
        let err = encode_analysis_result_artifact(&bad_dispersion, names.clone(), "cate-disp")
            .unwrap_err();
        assert!(err.to_string().contains("leaf dispersion"), "{err}");

        let mut bad_nan = ok;
        bad_nan.cate = Some(vec![0.1, f64::NAN]);
        let err = encode_analysis_result_artifact(&bad_nan, names, "cate-nan").unwrap_err();
        assert!(err.to_string().contains("cate"), "{err}");
    }
}
