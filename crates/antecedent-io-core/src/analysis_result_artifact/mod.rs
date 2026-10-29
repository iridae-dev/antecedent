//! Composite analysis-result artifact.
//!
//! This additive container keeps response, posterior, mediation-grid, and
//! structural-mixture axes together without extending the exhaustive legacy
//! `CausalPayloadKind` enum.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::{
    ArtifactKind, ArtifactManifest, AssumptionRecordWire, CausalQueryWire, CausalResponseWire,
    CompressPolicy, DiagnosticWire, EncodedArtifact, IdentificationResultWire, IoError,
    ProvenanceWire, RefutationReportWire, STABLE_FORMAT, SemanticVersion, from_cbor,
    pack_section_shared, read_and_migrate, to_cbor,
};

const ARTIFACT_KIND: &str = crate::artifact_kinds::ANALYSIS_RESULT;
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
    /// Frequentist Imbens–Manski interval whose SDs come from the completions' joint
    /// row influence.
    ImbensManskiInfluence,
}

/// Interval for the identified set of a class-aware scalar effect. Frequentist
/// (`imbens_manski_shared_block`): covers the true effect with asymptotic
/// probability at least `level` whenever it is one retained identified
/// completion's effect. Bayesian (`product_posterior_envelope_quantile`): every
/// retained completion's posterior puts at most `1 − Φ(critical_value)` of its
/// mass outside each endpoint. Mirrors [`antecedent_stats::IdentifiedSetInterval`].
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
        method: antecedent_stats::IdentifiedSetIntervalMethod,
    ) -> Result<Self, IoError> {
        use antecedent_stats::IdentifiedSetIntervalMethod as Method;
        match method {
            Method::ImbensManskiSharedBlock => Ok(Self::ImbensManskiSharedBlock),
            Method::ProductPosteriorEnvelopeQuantile => Ok(Self::ProductPosteriorEnvelopeQuantile),
            Method::ImbensManskiInfluence => Ok(Self::ImbensManskiInfluence),
            other => Err(IoError::Convert(format!(
                "identified-set interval construction {other:?} has no wire tag"
            ))),
        }
    }

    /// In-memory construction for this wire tag.
    #[must_use]
    pub const fn method(self) -> antecedent_stats::IdentifiedSetIntervalMethod {
        use antecedent_stats::IdentifiedSetIntervalMethod as Method;
        match self {
            Self::ImbensManskiSharedBlock => Method::ImbensManskiSharedBlock,
            Self::ProductPosteriorEnvelopeQuantile => Method::ProductPosteriorEnvelopeQuantile,
            Self::ImbensManskiInfluence => Method::ImbensManskiInfluence,
        }
    }
}

/// Encode an identified-set interval.
///
/// # Errors
///
/// A construction with no wire tag (never silently relabelled).
pub fn identified_set_interval_to_wire(
    interval: &antecedent_stats::IdentifiedSetInterval,
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
        completions: crate::error::fit_u64(interval.completions)?,
        replicates: crate::error::fit_u64(interval.replicates)?,
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
) -> Result<antecedent_stats::IdentifiedSetInterval, IoError> {
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
    Ok(antecedent_stats::IdentifiedSetInterval {
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

#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde's skip_serializing_if passes `&T`")]
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
    /// Support read from an unbalanced panel; present exactly when the query declared it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_window: Option<ObservedWindowWire>,
}

/// The treated unit's observed window and the donors it admitted.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservedWindowWire {
    /// Observed pre-intervention periods of the treated unit, ascending.
    pub pre_periods: Vec<i64>,
    /// Observed post-intervention periods of the treated unit, ascending.
    pub post_periods: Vec<i64>,
    /// Donors observed in every window period.
    pub donors_used: usize,
    /// Excluded units with the number of window periods each misses, in unit order.
    pub donors_dropped: Vec<(String, usize)>,
    /// Observed unit-periods no estimate read.
    pub unread_rows: usize,
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
    /// Exact graphless license status; absent for off-axis or legacy results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphless_support_status: Option<String>,
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

fn default_randomized_itt() -> String {
    "itt".into()
}

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
    /// Exact graphless simultaneous-band license (`licensed`), when the
    /// published band and arm support match a licensed band row. Absent for
    /// off-axis bands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphless_support_status: Option<String>,
}

/// The retained posterior of an unlicensed Bayesian randomized-design row.
///
/// Every coordinate has a posterior mean and an equal-tailed 95% credible interval; the
/// covariance across coordinates is the covariance of the retained joint draws, so draw `k` of
/// every coordinate comes from one posterior sample. The calibration is always `unmeasured`
/// and the row is never licensed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PosteriorRowWire {
    /// Estimator id of the row, for example `bayesian.piecewise_survival`.
    pub estimator: String,
    /// The target the posterior answers.
    pub target: String,
    /// Always `posterior`.
    pub uncertainty: String,
    /// Calibration status of every interval (`unmeasured`).
    pub calibration: String,
    /// Always `false`: these rows are not licensed.
    pub licensed: bool,
    /// Name of every coordinate, in order.
    pub coordinates: Vec<String>,
    /// Posterior mean of each coordinate.
    pub mean: Vec<f64>,
    /// Lower endpoint of each coordinate's equal-tailed 95% credible interval.
    pub lower: Vec<f64>,
    /// Upper endpoint of each coordinate's equal-tailed 95% credible interval.
    pub upper: Vec<f64>,
    /// Joint covariance across the coordinates, row-major, from the retained draws.
    pub covariance: Option<Vec<f64>>,
    /// Exact posterior mean (known residual variance only).
    pub exact_mean: Option<Vec<f64>>,
    /// Exact posterior covariance, row-major (known residual variance only).
    pub exact_covariance: Option<Vec<f64>>,
    /// Posterior draws behind the intervals.
    pub draws: usize,
    /// Named scalar summaries.
    pub extras: std::collections::BTreeMap<String, f64>,
    /// Named vector summaries.
    pub vectors: std::collections::BTreeMap<String, Vec<f64>>,
}

/// What every stored posterior row's uncertainty is.
pub const POSTERIOR_ROW_UNCERTAINTY: &str = "posterior";
/// Calibration status of every posterior row.
pub const POSTERIOR_ROW_CALIBRATION: &str = "unmeasured";

impl PosteriorRowWire {
    /// An empty posterior of `estimator` over `coordinates`, unlicensed and uncalibrated.
    #[must_use]
    pub fn new(estimator: &str, target: String, coordinates: Vec<String>, draws: usize) -> Self {
        Self {
            estimator: estimator.to_owned(),
            target,
            uncertainty: POSTERIOR_ROW_UNCERTAINTY.to_owned(),
            calibration: POSTERIOR_ROW_CALIBRATION.to_owned(),
            licensed: false,
            coordinates,
            mean: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
            covariance: None,
            exact_mean: None,
            exact_covariance: None,
            draws,
            extras: std::collections::BTreeMap::new(),
            vectors: std::collections::BTreeMap::new(),
        }
    }

    /// Set the means and the credible intervals.
    #[must_use]
    pub fn with_intervals(mut self, mean: Vec<f64>, intervals: &[[f64; 2]]) -> Self {
        self.mean = mean;
        self.lower = intervals.iter().map(|&[lower, _]| lower).collect();
        self.upper = intervals.iter().map(|&[_, upper]| upper).collect();
        self
    }

    /// Whether every stored number is finite.
    #[must_use]
    pub fn all_finite(&self) -> bool {
        let optional = |values: &Option<Vec<f64>>| {
            values.as_ref().is_none_or(|v| v.iter().all(|x| x.is_finite()))
        };
        self.mean.iter().chain(&self.lower).chain(&self.upper).all(|v| v.is_finite())
            && optional(&self.covariance)
            && optional(&self.exact_mean)
            && optional(&self.exact_covariance)
            && self.extras.values().all(|v| v.is_finite())
            && self.vectors.values().flatten().all(|v| v.is_finite())
    }

    /// The shape and marker checks every consumer applies before trusting the row.
    ///
    /// # Errors
    /// A licensed row, a changed uncertainty or calibration marker, misaligned arrays or a
    /// non-finite value.
    pub fn validate(&self) -> Result<(), crate::IoError> {
        let width = self.coordinates.len();
        let square =
            |values: &Option<Vec<f64>>| values.as_ref().is_none_or(|v| v.len() == width * width);
        if self.uncertainty != POSTERIOR_ROW_UNCERTAINTY
            || self.calibration != POSTERIOR_ROW_CALIBRATION
            || self.licensed
            || self.estimator.is_empty()
            || width == 0
            || self.mean.len() != width
            || self.lower.len() != width
            || self.upper.len() != width
            || !square(&self.covariance)
            || !square(&self.exact_covariance)
            || self.exact_mean.as_ref().is_some_and(|v| v.len() != width)
            || self.draws < 2
            || !self.all_finite()
            || self.lower.iter().zip(&self.upper).any(|(lower, upper)| lower > upper)
        {
            return Err(crate::IoError::Convert(
                "a posterior row must be an unlicensed, unmeasured-calibration posterior with aligned finite arrays".into(),
            ));
        }
        Ok(())
    }
}

/// Retained observational two-decision regime contrast and its uncertainty.
///
/// The cross-fit scores are kept so a loader recomputes the means, the standard error and the
/// interval. The result is unlicensed and its repeated-sampling calibration is unmeasured.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SequentialContrastWire {
    /// Active regime actions.
    pub active_regime: [u8; 2],
    /// Control regime actions.
    pub control_regime: [u8; 2],
    /// Cross-fit active minus control.
    pub contrast: f64,
    /// Cross-fit active-regime mean.
    pub active_mean: f64,
    /// Cross-fit control-regime mean.
    pub control_mean: f64,
    /// Subjects scored.
    pub evaluated_rows: u64,
    /// Smallest regime-action probability used in a score.
    pub minimum_action_probability: f64,
    /// Declared positivity floor.
    pub min_probability: f64,
    /// Standard error of the mean contrast score (normal-interval report).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_error: Option<f64>,
    /// Normal interval at `coverage` (normal-interval report).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<[f64; 2]>,
    /// Nominal central coverage of the interval.
    pub coverage: f64,
    /// Cross-fit active score of each subject.
    pub active_scores: Vec<f64>,
    /// Cross-fit control score of each subject.
    pub control_scores: Vec<f64>,
    /// Active minus control score of each subject.
    pub contrast_scores: Vec<f64>,
    /// `normal_interval_calibration_unmeasured` or the modular posterior label.
    pub uncertainty: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Always `false`.
    pub licensed: bool,
}

const fn is_zero_lag(lag: &usize) -> bool {
    *lag == 0
}

/// Retained within-period history-matched contrast, its matched pairs and standard error.
///
/// The pairs and the within-arm variance estimates of their units are kept so a loader recomputes
/// the contrast, the weight total and the Abadie-Imbens standard error. The periods of the pairs
/// are kept so a loader recomputes the Newey-West standard error and interval over the matched
/// periods too. The result is unlicensed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HistoryMatchWire {
    /// Sampling-weighted mean of treated-minus-matched-control outcomes.
    pub contrast: f64,
    /// Abadie-Imbens standard error; absent when a matched unit or donor has no same-arm
    /// neighbour in its period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_error: Option<f64>,
    /// Sum of sampling weights on the matched treated units.
    pub weight_total: f64,
    /// Row indices of matched treated units.
    pub matched_queries: Vec<usize>,
    /// Row index of the control each matched query used.
    pub matched_donors: Vec<usize>,
    /// Outcome difference of each matched pair.
    pub differences: Vec<f64>,
    /// Sampling weight of each matched treated unit.
    pub pair_weights: Vec<f64>,
    /// Within-arm outcome variance estimate of each matched treated unit, in pair order; empty
    /// exactly when `standard_error` is absent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub query_variances: Vec<f64>,
    /// Within-arm outcome variance estimate of the control each matched query used, in pair
    /// order; empty exactly when `standard_error` is absent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub donor_variances: Vec<f64>,
    /// Treated rows with no control inside their period and the caliper.
    pub unmatched_treated: Vec<usize>,
    /// Periods in which at least one treated unit matched.
    pub periods_matched: usize,
    /// Calendar period of each matched treated unit, in pair order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pair_periods: Vec<i64>,
    /// Newey-West standard error over the matched periods; absent when fewer than two periods
    /// hold a match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_standard_error: Option<f64>,
    /// Lower endpoint of the 95% normal interval around the contrast.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_lower: Option<f64>,
    /// Upper endpoint of that interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_upper: Option<f64>,
    /// Bartlett truncation lag of the Newey-West estimate; zero when it is absent.
    #[serde(default, skip_serializing_if = "is_zero_lag")]
    pub serial_bandwidth: usize,
    /// The caliper, when one was declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caliper: Option<f64>,
    /// Always `abadie_imbens_and_newey_west_period_interval_calibration_unmeasured`.
    pub uncertainty: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Always `false`.
    pub licensed: bool,
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

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if requires an fn(&T) -> bool signature"
)]
fn is_zero_usize(value: &usize) -> bool {
    *value == 0
}

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
    /// Optional native execution v1 facts, separate from scientific licensing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_execution: Option<crate::native_execution_wire::NativeExecutionWire>,
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
    /// Posterior of an unlicensed Bayesian randomized-design row, in place of the
    /// frequentist section of the same query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_row: Option<PosteriorRowWire>,
    /// Subject-owned prespecified longitudinal regime point value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub longitudinal_regime: Option<LongitudinalRegimeWire>,
    /// Observational two-decision regime contrast, unlicensed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequential_contrast: Option<SequentialContrastWire>,
    /// Within-period history-matched contrast, unlicensed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_match: Option<HistoryMatchWire>,
    /// Sharp bounds on the average effect under a binary instrument, unlicensed. Absent on
    /// every other result, so their bodies and digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iv_bounds: Option<IvBoundsWire>,
    /// Noncausal association of the observed data, unlicensed. Absent on every other result,
    /// so their bodies and digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub association: Option<AssociationWire>,
    /// Rollout design effects with whole-procedure uncertainty, unlicensed. Absent on every
    /// other result, so their bodies and digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quasi_rollout: Option<QuasiRolloutWire>,
    /// Sequentially randomized unit-period design or regime survival, unlicensed. Absent on
    /// every other result, so their bodies and digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_design: Option<TemporalDesignWire>,
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
    /// Joint score-family covariance retained by the producing execution. Its values
    /// are sealed evidence; a standalone scalar program cannot replay the other members.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint_covariance: Option<crate::JointCovarianceWire>,
    /// Nominal simultaneous interval of this member (lower, upper, level).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simultaneous_interval: Option<(f64, f64, f64)>,
    /// Family BH and BY adjusted p-values, when published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjusted_p_values: Option<(f64, f64)>,
    /// Declared score-family contrast (value, standard error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family_contrast: Option<(f64, f64)>,
    /// Nominal simultaneous contrast interval (lower, upper, level).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family_contrast_interval: Option<(f64, f64, f64)>,
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
    /// Per-outcome counterfactual contrasts of a query with several outcomes or a joint
    /// baseline, in place of [`Self::unit_effects`].
    ///
    /// Absent on every other result, so their bodies (and digests) are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counterfactual_contrasts: Option<CounterfactualContrastsWire>,
    /// Complete-case-aligned CATE point predictions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cate: Option<std::sync::Arc<[f64]>>,
    /// Portable fitted effect, bound into the result body identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitted_effect: Option<std::sync::Arc<antecedent_learn::FittedEffect>>,
    /// Licensed pointwise CATE standard errors, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cate_se: Option<std::sync::Arc<[f64]>>,
    /// Forest leaf-dispersion diagnostic per row. Not a standard error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cate_leaf_dispersion: Option<std::sync::Arc<[f64]>>,
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
    validate::validate_result(result, &variable_names, false)?;
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
    validate::validate_result(&decoded_body, &decoded_header.variable_names, true)?;
    Ok((artifact, decoded_header, decoded_body))
}

mod association;
mod counterfactual_contrasts;
mod graphless;
mod iv_bounds;
mod quasi_rollout;
mod temporal_design;
mod validate;

pub use association::{
    ASSOCIATION_ANALYSES, ASSOCIATION_ANALYTIC_UNCERTAINTY, ASSOCIATION_NO_UNCERTAINTY,
    ASSOCIATION_POSTERIOR_UNCERTAINTY, ASSOCIATION_RESAMPLING_UNCERTAINTY,
    AssociationPosteriorWire, AssociationPredictionWire, AssociationResamplingWire,
    AssociationTermWire, AssociationWire,
};
pub(crate) use counterfactual_contrasts::validate_outcome_contrasts;
pub use counterfactual_contrasts::{CounterfactualContrastsWire, OutcomeContrastWire};
pub use iv_bounds::{
    IV_BOUNDS_BAYESIAN_METHOD, IV_BOUNDS_FREQUENTIST_METHOD, IV_BOUNDS_POINT_UNCERTAINTY,
    IvBoundsIntervalWire, IvBoundsPosteriorWire, IvBoundsWire,
};
pub use quasi_rollout::{DesignEffectWire, QuasiRolloutWire, rederive_rollout_summary};
pub use temporal_design::TemporalDesignWire;

pub(crate) use validate::validate_interventional_distribution;

#[cfg(test)]
mod tests;
