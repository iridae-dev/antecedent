//! Effect-estimate and refutation wire types and their conversions.
//!
//! Re-exported from [`crate::analysis_wire`], which joins them to the
//! identification and diagnostic wires of `antecedent-io-core`.
//!
//! Older readers that deny unknown fields may refuse records containing the optional
//! uncertainty and provenance metadata added to this wire.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_estimate::{
    BootstrapAccounting, ClipSensitivity, EffectEstimate, FirstStageDiagnostics, OverlapPolicy,
    OverlapReport, PropensityInterval,
};
use antecedent_validate::RefutationReport;
use serde::{Deserialize, Serialize};

pub use antecedent_io_core::analysis_wire::JointCovarianceWire;
use antecedent_io_core::analysis_wire::RefutationReportWire;

use crate::error::IoError;
use crate::trace::{AssumptionRecordWire, assumptions_from_wire, assumptions_to_wire};

/// Analytic standard-error method, including the requested HAC lag.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticSeKindWire {
    /// IID or homoskedastic.
    Homoskedastic,
    /// HC0 sandwich.
    Hc0,
    /// HC1 sandwich.
    Hc1,
    /// HC2 sandwich.
    Hc2,
    /// HC3 sandwich.
    Hc3,
    /// Cluster-robust.
    Cluster,
    /// Multiway cluster-robust.
    Multiway,
    /// Newey-West with its lag.
    NeweyWest {
        /// Lag.
        lag: u64,
    },
    /// Panel cluster HAC with its lag.
    PanelClusterHac {
        /// Lag.
        lag: u64,
    },
}

/// Circular-block SE family.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BlockFamilyWire {
    /// One window.
    SingleWindow,
    /// Mediation.
    Mediation,
    /// Sequential.
    Sequential,
    /// Mixture.
    Mixture,
    /// Whole-procedure row block.
    RowBlock,
}

/// Circular-block resampling geometry.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BlockResamplingWire {
    /// Block length in aligned rows.
    pub block_length: u64,
    /// Aligned rows resampled.
    pub rows: u64,
    /// Kernel-bias scaling factor.
    pub kernel_bias: f64,
}

/// Effect estimate wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EffectEstimateWire {
    /// ATE.
    pub ate: f64,
    /// Analytic SE.
    pub se_analytic: f64,
    /// Analytic SE method, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analytic_kind: Option<AnalyticSeKindWire>,
    /// Student-t reference degrees of freedom, or normal reference if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analytic_reference_df: Option<f64>,
    /// Circular-block resampling geometry, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_resampling: Option<BlockResamplingWire>,
    /// Circular-block SE family, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_family: Option<BlockFamilyWire>,
    /// Observations that informed the fit, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_obs: Option<u64>,
    /// Bootstrap SE.
    pub se_bootstrap: Option<f64>,
    /// Bootstrap ok.
    pub bootstrap_replicates_ok: Option<u32>,
    /// Bootstrap failed.
    pub bootstrap_replicates_failed: Option<u32>,
    /// Whether bootstrap stopped cooperatively after cancellation.
    #[serde(default)]
    pub bootstrap_cancelled: bool,
    /// Whether adaptive bootstrap stopped after convergence.
    #[serde(default)]
    pub bootstrap_early_stopped: bool,
    /// Assumptions.
    pub assumptions: Vec<AssumptionRecordWire>,
    /// Overlap policy tag.
    pub overlap_policy: EstimateOverlapPolicyWire,
    /// Clip.
    pub overlap_clip: Option<f64>,
    /// Trim.
    pub overlap_trim: Option<f64>,
    /// Propensity-overlap evidence, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlap_report: Option<OverlapReportWire>,
    /// Weak-instrument first-stage evidence, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_stage_diagnostics: Option<FirstStageDiagnosticsWire>,
    /// Retained memory.
    pub retained_memory_bytes: Option<u64>,
    /// Cross-fitted AIPW score table (retarget / joint cells / exceedance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_table: Option<ScoreTableWire>,
    /// Per-arm / per-threshold `F_a(c)` after monotone rearrangement, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exceedance_cdf: Option<Vec<f64>>,
    /// Whether exceedance means were isotonically rearranged (cov/bands stay raw).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub monotone_rearranged: bool,
    /// Joint covariance of score means or shared-row claims.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint_covariance: Option<JointCovarianceWire>,
    /// Simultaneous score-family bands and weighted support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_inference: Option<ScoreInferenceWire>,
    /// Declared canonical scenario effects, in estimand order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario_effects: Option<Vec<f64>>,
    /// Simultaneous batch interval (lower, upper, level).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simultaneous_interval: Option<(f64, f64, f64)>,
    /// BH and BY adjusted p-values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjusted_p_values: Option<(f64, f64)>,
    /// Declared family contrast `(value, se)` when batch FDR tested a contrast.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family_contrast: Option<(f64, f64)>,
    /// Contrast-family simultaneous interval `(lower, upper, level)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family_contrast_interval: Option<(f64, f64, f64)>,
    /// Simultaneous scenario intervals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario_intervals: Option<Vec<(f64, f64)>>,
    /// Additive joint-response disclosure: interaction is structurally zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interaction_structurally_zero: Option<bool>,
    /// Per-unit effects are homogeneous by construction of the selected mechanisms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_effects_homogeneous: Option<bool>,
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
    /// Actual fitted learner (spec, implementation, version), in fit order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub learner_provenance: Vec<(String, String, String)>,
    /// Point E-value for a named no-latent premise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evalue: Option<f64>,
    /// Threshold the E-value refuter judged [`EffectEstimateWire::evalue`] against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evalue_threshold: Option<f64>,
    /// Candidate-selection provenance, including screen/estimate row splits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_selection: Option<CandidateSelectionWire>,
    /// How the columns the estimator read were prepared, when it recorded a receipt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation: Option<PreparationWire>,
}

/// Overlap policy of an estimate on the wire; its clip and trim ride beside it on
/// [`EffectEstimateWire`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EstimateOverlapPolicyWire {
    /// Overlap diagnostics are required (with the optional clip and trim).
    RequireDiagnostics,
    /// The caller overrode the overlap requirement.
    ExplicitOverride,
}

/// Why a first-stage / Anderson-Rubin uncertainty statement was withheld.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyWithheldWire {
    /// The Anderson-Rubin set needs homoskedastic errors.
    AndersonRubinRequiresHomoskedastic,
    /// The Anderson-Rubin set is a union of disjoint intervals.
    AndersonRubinSetIsUnion,
    /// The Anderson-Rubin set is empty.
    AndersonRubinSetEmpty,
    /// The Anderson-Rubin set needs excluded instruments.
    AndersonRubinRequiresExcludedInstruments,
    /// The Anderson-Rubin level is not in `(0, 1)`.
    AndersonRubinInvalidLevel,
    /// The Anderson-Rubin critical value could not be computed.
    AndersonRubinCriticalValueFailed,
}

impl UncertaintyWithheldWire {
    /// The estimator's tag for this reason.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::AndersonRubinRequiresHomoskedastic => "anderson_rubin_requires_homoskedastic",
            Self::AndersonRubinSetIsUnion => "anderson_rubin_set_is_union",
            Self::AndersonRubinSetEmpty => "anderson_rubin_set_empty",
            Self::AndersonRubinRequiresExcludedInstruments => {
                "anderson_rubin_requires_excluded_instruments"
            }
            Self::AndersonRubinInvalidLevel => "anderson_rubin_invalid_level",
            Self::AndersonRubinCriticalValueFailed => "anderson_rubin_critical_value_failed",
        }
    }

    /// The reason an estimator tag names, or `None` for a tag this format does not define.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        [
            Self::AndersonRubinRequiresHomoskedastic,
            Self::AndersonRubinSetIsUnion,
            Self::AndersonRubinSetEmpty,
            Self::AndersonRubinRequiresExcludedInstruments,
            Self::AndersonRubinInvalidLevel,
            Self::AndersonRubinCriticalValueFailed,
        ]
        .into_iter()
        .find(|reason| reason.tag() == tag)
    }
}

/// Preparation receipt summary recorded on an estimate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparationWire {
    /// Stable name of the preparation backend.
    pub backend: String,
    /// Stable name of the end-to-end route bound.
    pub route_bound: String,
    /// Rows the estimator read; absent when it was handed a scan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_admitted: Option<u64>,
    /// Lower-case hex digest of the preparation plan.
    pub receipt_digest: String,
    /// Payload bytes the preparation admitted; absent when the writer recorded none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admitted_bytes: Option<u64>,
    /// Byte budget the payload was compared with; absent when none was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_bytes: Option<u64>,
    /// What the estimator was given (`finite`, `materialized_from_scan`, `scan`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<String>,
}

/// Screen / estimate split recorded on a batch estimate artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CandidateSelectionWire {
    /// Screen run id.
    pub screen_id: String,
    /// Procedure wire name.
    pub procedure: String,
    /// Winning family index.
    pub winner_index: Option<usize>,
    /// Family size.
    pub family_size: usize,
    /// Screen-half row indexes.
    pub screen_rows: Vec<u64>,
    /// Estimate-half row indexes.
    pub estimate_rows: Vec<u64>,
    /// Whether the halves are disjoint.
    pub disjoint: bool,
}

/// Simultaneous inference over a declared score family.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
#[serde(deny_unknown_fields)]
pub struct ScoreInferenceWire {
    pub raw_means: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub level: f64,
    pub critical_value: f64,
    pub event_n_eff: Vec<f64>,
    pub threshold_supported: Vec<bool>,
    pub n_eff: f64,
    pub n_eff_by_arm: Vec<f64>,
    pub propensity_range: Option<(f64, f64)>,
    pub overlap_ok: bool,
}

/// Artifact form of a cross-fitted AIPW score table.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScoreTableWire {
    /// Observed cells on retained rows.
    #[serde(default)]
    pub observed_arm: Vec<u32>,
    /// Raw out-of-fold propensities in score-column order.
    #[serde(default)]
    pub propensities: Vec<f64>,
    /// Original outcomes for threshold support.
    #[serde(default)]
    pub observed_outcome: Vec<f64>,
    /// Complete-case row count.
    pub n_rows: u64,
    /// Original row indexes.
    pub row_index: Vec<u64>,
    /// Fold ids.
    pub fold_ids: Vec<u32>,
    /// Fold count.
    pub n_folds: u32,
    /// Column-major scores.
    pub scores: Vec<f64>,
    /// `(arm, threshold)` column keys.
    pub columns: Vec<ScoreColumnWire>,
    /// Adjustment variable raw ids.
    pub adjustment_set: Vec<u32>,
    /// Nuisance provenance.
    pub nuisance_provenance: String,
    /// Propensity clip the estimator applied (`None`: unclipped or a legacy artifact).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub propensity_clip: Option<f64>,
    /// Treatment raw id.
    pub treatment: u32,
    /// Extra intervened raw ids.
    pub intervened: Vec<u32>,
}

/// Score-table column key.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScoreColumnWire {
    /// Arm or cell mask.
    pub arm: u32,
    /// Exceedance threshold (`None` = mean).
    pub threshold: Option<f64>,
}

/// Closed excluded propensity interval.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
#[serde(deny_unknown_fields)]
pub struct PropensityIntervalWire {
    pub low: f64,
    pub high: f64,
}

/// Sensitivity of overlap evidence to clipping thresholds.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
#[serde(deny_unknown_fields)]
pub struct ClipSensitivityWire {
    pub thresholds: Vec<f64>,
    pub ess: Vec<f64>,
    pub treated_ess: Vec<f64>,
    pub control_ess: Vec<f64>,
    pub extreme_weight_counts: Vec<u32>,
}

/// Propensity overlap evidence retained on an effect estimate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
#[serde(deny_unknown_fields)]
pub struct OverlapReportWire {
    pub propensity_min: f64,
    pub propensity_max: f64,
    pub ess: Option<f64>,
    pub extreme_weight_count: u32,
    pub excluded_fraction: f64,
    pub target_population_support: f64,
    pub excluded_regions: Vec<PropensityIntervalWire>,
    pub clip: Option<f64>,
    pub trim: Option<f64>,
    pub retained_fraction: f64,
    pub clip_sensitivity: Option<ClipSensitivityWire>,
}

/// Weak-instrument first-stage evidence.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
#[serde(deny_unknown_fields)]
pub struct FirstStageDiagnosticsWire {
    pub f_statistic: f64,
    pub df1: u64,
    pub df2: u64,
    pub partial_r2: f64,
    /// Homoskedastic Anderson–Rubin set `(lower, upper, level)`; endpoints may be infinite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anderson_rubin: Option<(f64, f64, f64)>,
    /// Why AR / licensed IV uncertainty was withheld.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty_withheld: Option<UncertaintyWithheldWire>,
}

fn kind_to_wire(kind: antecedent_estimate::AnalyticSeKind) -> AnalyticSeKindWire {
    use antecedent_estimate::AnalyticSeKind as K;
    match kind {
        K::Homoskedastic => AnalyticSeKindWire::Homoskedastic,
        K::Hc0 => AnalyticSeKindWire::Hc0,
        K::Hc1 => AnalyticSeKindWire::Hc1,
        K::Hc2 => AnalyticSeKindWire::Hc2,
        K::Hc3 => AnalyticSeKindWire::Hc3,
        K::Cluster => AnalyticSeKindWire::Cluster,
        K::Multiway => AnalyticSeKindWire::Multiway,
        K::NeweyWest { lag } => {
            AnalyticSeKindWire::NeweyWest { lag: crate::width::lossless_u64(lag) }
        }
        K::PanelClusterHac { lag } => {
            AnalyticSeKindWire::PanelClusterHac { lag: crate::width::lossless_u64(lag) }
        }
    }
}

fn kind_from_wire(
    kind: AnalyticSeKindWire,
) -> Result<antecedent_estimate::AnalyticSeKind, IoError> {
    use antecedent_estimate::AnalyticSeKind as K;
    Ok(match kind {
        AnalyticSeKindWire::Homoskedastic => K::Homoskedastic,
        AnalyticSeKindWire::Hc0 => K::Hc0,
        AnalyticSeKindWire::Hc1 => K::Hc1,
        AnalyticSeKindWire::Hc2 => K::Hc2,
        AnalyticSeKindWire::Hc3 => K::Hc3,
        AnalyticSeKindWire::Cluster => K::Cluster,
        AnalyticSeKindWire::Multiway => K::Multiway,
        AnalyticSeKindWire::NeweyWest { lag } => {
            K::NeweyWest { lag: usize::try_from(lag).map_err(|_| IoError::TooLarge)? }
        }
        AnalyticSeKindWire::PanelClusterHac { lag } => {
            K::PanelClusterHac { lag: usize::try_from(lag).map_err(|_| IoError::TooLarge)? }
        }
    })
}

fn family_to_wire(family: antecedent_estimate::CircularBlockFamily) -> BlockFamilyWire {
    use antecedent_estimate::CircularBlockFamily as F;
    match family {
        F::SingleWindow => BlockFamilyWire::SingleWindow,
        F::Mediation => BlockFamilyWire::Mediation,
        F::Sequential => BlockFamilyWire::Sequential,
        F::Mixture => BlockFamilyWire::Mixture,
        F::RowBlock => BlockFamilyWire::RowBlock,
    }
}

fn family_from_wire(family: BlockFamilyWire) -> antecedent_estimate::CircularBlockFamily {
    use antecedent_estimate::CircularBlockFamily as F;
    match family {
        BlockFamilyWire::SingleWindow => F::SingleWindow,
        BlockFamilyWire::Mediation => F::Mediation,
        BlockFamilyWire::Sequential => F::Sequential,
        BlockFamilyWire::Mixture => F::Mixture,
        BlockFamilyWire::RowBlock => F::RowBlock,
    }
}

/// Encode effect estimate.
#[must_use]
pub fn effect_estimate_to_wire(e: &EffectEstimate) -> EffectEstimateWire {
    let (overlap_policy, overlap_clip, overlap_trim) = match e.provenance.overlap.bounds() {
        None => (EstimateOverlapPolicyWire::ExplicitOverride, None, None),
        Some((clip, trim)) => (EstimateOverlapPolicyWire::RequireDiagnostics, clip, trim),
    };
    EffectEstimateWire {
        ate: e.ate,
        se_analytic: e.se_analytic(),
        analytic_kind: e.uncertainty.analytic.kind.map(kind_to_wire),
        analytic_reference_df: e.uncertainty.analytic.reference_df,
        block_resampling: e.uncertainty.block.resampling.map(|r| BlockResamplingWire {
            block_length: crate::width::lossless_u64(r.block_length),
            rows: crate::width::lossless_u64(r.rows),
            kernel_bias: r.kernel_bias,
        }),
        block_family: e.uncertainty.block.family.map(family_to_wire),
        n_obs: e.provenance.n_obs,
        se_bootstrap: e.se_bootstrap(),
        bootstrap_replicates_ok: e.uncertainty.bootstrap.replicates_ok,
        bootstrap_replicates_failed: e.uncertainty.bootstrap.replicates_failed,
        bootstrap_cancelled: e.uncertainty.bootstrap.cancelled,
        bootstrap_early_stopped: e.uncertainty.bootstrap.early_stopped,
        assumptions: assumptions_to_wire(&e.provenance.assumptions),
        overlap_policy,
        overlap_clip,
        overlap_trim,
        overlap_report: e.diagnostics.overlap_report.as_ref().map(overlap_report_to_wire),
        first_stage_diagnostics: e.diagnostics.first_stage.as_ref().map(|diagnostic| {
            FirstStageDiagnosticsWire {
                f_statistic: diagnostic.f_statistic,
                df1: crate::width::lossless_u64(diagnostic.df1),
                df2: crate::width::lossless_u64(diagnostic.df2),
                partial_r2: diagnostic.partial_r2,
                anderson_rubin: diagnostic.anderson_rubin,
                // The estimator names the closed set `UncertaintyWithheldWire` defines; a tag
                // outside it has no wire spelling the decoder would accept, so it is not written.
                uncertainty_withheld: diagnostic
                    .uncertainty_withheld
                    .and_then(UncertaintyWithheldWire::from_tag),
            }
        }),
        retained_memory_bytes: e.provenance.retained_memory_bytes,
        simultaneous_interval: e.family.simultaneous_interval,
        adjusted_p_values: e.family.adjusted_p_values,
        family_contrast: e.family.contrast,
        family_contrast_interval: e.family.contrast_interval,
        unit_effects_homogeneous: e.heterogeneity.unit_effects_homogeneous.then_some(true),
        cate: e.heterogeneity.cate.as_ref().map(|v| v.to_vec()),
        fitted_effect: e.heterogeneity.fitted_effect.as_deref().cloned(),
        cate_se: e.heterogeneity.cate_se.as_ref().map(|v| v.to_vec()),
        cate_leaf_dispersion: e.heterogeneity.cate_leaf_dispersion.as_ref().map(|v| v.to_vec()),
        outcome_oof_r2: e.crossfit.outcome_oof_r2,
        treatment_oof_logloss: e.crossfit.treatment_oof_logloss,
        crossfit_folds: e.crossfit.folds,
        crossfit_seed: e.crossfit.seed,
        learner_provenance: e
            .crossfit
            .learner_provenance
            .iter()
            .map(|p| (p.spec.clone(), p.implementation.clone(), p.version.clone()))
            .collect(),
        evalue: e.diagnostics.evalue,
        evalue_threshold: e.diagnostics.evalue_threshold,
        preparation: e.provenance.preparation.as_deref().map(|p| PreparationWire {
            backend: p.backend.clone(),
            route_bound: p.route_bound.clone(),
            rows_admitted: p.rows_admitted,
            receipt_digest: p.receipt_digest.clone(),
            admitted_bytes: p.admitted_bytes,
            budget_bytes: p.budget_bytes,
            delivery: p.delivery.clone(),
        }),
        candidate_selection: e.family.candidate_selection.as_ref().map(|s| {
            CandidateSelectionWire {
                screen_id: s.screen_id.to_string(),
                procedure: s.procedure.to_string(),
                winner_index: s.winner_index,
                family_size: s.family_size,
                screen_rows: s.screen_rows.iter().map(|id| id.get()).collect(),
                estimate_rows: s.estimate_rows.iter().map(|id| id.get()).collect(),
                disjoint: s.disjoint,
            }
        }),
        scenario_effects: e.scenarios.effects.as_ref().map(|v| v.to_vec()),
        scenario_intervals: e.scenarios.intervals.as_ref().map(|v| v.to_vec()),
        joint_covariance: e
            .joint
            .covariance
            .as_ref()
            .map(|c| JointCovarianceWire { dim: c.dim, values: c.values.to_vec() }),
        score_inference: e.scores.inference.as_ref().map(|s| ScoreInferenceWire {
            raw_means: s.raw_means.clone(),
            lower: s.lower.clone(),
            upper: s.upper.clone(),
            level: s.level,
            critical_value: s.critical_value,
            event_n_eff: s.event_n_eff.clone(),
            threshold_supported: s.threshold_supported.clone(),
            n_eff: s.support.n_eff,
            n_eff_by_arm: s.support.n_eff_by_arm.clone(),
            propensity_range: s.support.propensity_range,
            overlap_ok: s.support.overlap_ok,
        }),
        score_table: e.scores.table.as_ref().map(score_table_to_wire),
        exceedance_cdf: e.joint.exceedance_cdf.as_ref().map(|v| v.to_vec()),
        monotone_rearranged: e.joint.monotone_rearranged,
        interaction_structurally_zero: e.joint.interaction_structurally_zero.then_some(true),
    }
}

fn score_table_to_wire(table: &antecedent_estimate::ScoreTable) -> ScoreTableWire {
    let wire = table.to_wire();
    ScoreTableWire {
        observed_arm: table.observed_arm.to_vec(),
        propensities: table.propensities.to_vec(),
        observed_outcome: table.observed_outcome.to_vec(),
        n_rows: wire.n_rows,
        row_index: wire.row_index,
        fold_ids: wire.fold_ids,
        n_folds: wire.n_folds,
        scores: wire.scores,
        columns: wire
            .columns
            .into_iter()
            .map(|c| ScoreColumnWire { arm: c.arm, threshold: c.threshold })
            .collect(),
        adjustment_set: wire.adjustment_set,
        nuisance_provenance: wire.nuisance_provenance,
        propensity_clip: wire.propensity_clip,
        treatment: wire.treatment,
        intervened: wire.intervened,
    }
}

fn score_table_from_wire(wire: ScoreTableWire) -> Result<antecedent_estimate::ScoreTable, IoError> {
    let domain = antecedent_estimate::ScoreTableWire {
        observed_arm: wire.observed_arm,
        propensities: wire.propensities,
        observed_outcome: wire.observed_outcome,
        n_rows: wire.n_rows,
        row_index: wire.row_index,
        fold_ids: wire.fold_ids,
        n_folds: wire.n_folds,
        scores: wire.scores,
        columns: wire
            .columns
            .into_iter()
            .map(|c| antecedent_estimate::ScoreColumn { arm: c.arm, threshold: c.threshold })
            .collect(),
        adjustment_set: wire.adjustment_set,
        nuisance_provenance: wire.nuisance_provenance,
        propensity_clip: wire.propensity_clip,
        treatment: wire.treatment,
        intervened: wire.intervened,
    };
    antecedent_estimate::ScoreTable::from_wire(domain).map_err(|e| IoError::Convert(e.to_string()))
}

fn overlap_report_to_wire(report: &OverlapReport) -> OverlapReportWire {
    OverlapReportWire {
        propensity_min: report.propensity_min,
        propensity_max: report.propensity_max,
        ess: report.ess,
        extreme_weight_count: report.extreme_weight_count,
        excluded_fraction: report.excluded_fraction,
        target_population_support: report.target_population_support,
        excluded_regions: report
            .excluded_regions
            .iter()
            .map(|region| PropensityIntervalWire { low: region.low, high: region.high })
            .collect(),
        clip: report.clip,
        trim: report.trim,
        retained_fraction: report.retained_fraction,
        clip_sensitivity: report.clip_sensitivity.as_ref().map(|sensitivity| ClipSensitivityWire {
            thresholds: sensitivity.thresholds.to_vec(),
            ess: sensitivity.ess.to_vec(),
            treated_ess: sensitivity.treated_ess.to_vec(),
            control_ess: sensitivity.control_ess.to_vec(),
            extreme_weight_counts: sensitivity.extreme_weight_counts.to_vec(),
        }),
    }
}

/// Decode an effect estimate without dropping its assumptions or diagnostic evidence.
///
/// This clones the wire; [`effect_estimate_from_owned_wire`] consumes it and moves the large
/// score, scenario and CATE columns into the estimate instead.
///
/// # Errors
///
/// Invalid assumption labels or scopes.
pub fn effect_estimate_from_wire(w: &EffectEstimateWire) -> Result<EffectEstimate, IoError> {
    effect_estimate_from_owned_wire(w.clone())
}

/// [`effect_estimate_from_wire`] for an owned wire: its vectors become the estimate's
/// columns without a copy.
///
/// # Errors
///
/// Invalid assumption labels or scopes.
pub fn effect_estimate_from_owned_wire(w: EffectEstimateWire) -> Result<EffectEstimate, IoError> {
    validate_effect_estimate_wire(&w)?;
    let overlap = match w.overlap_policy {
        EstimateOverlapPolicyWire::RequireDiagnostics => {
            OverlapPolicy::RequireDiagnostics { clip: w.overlap_clip, trim: w.overlap_trim }
        }
        EstimateOverlapPolicyWire::ExplicitOverride => OverlapPolicy::ExplicitOverride,
    };
    let overlap_report = w.overlap_report.map(overlap_report_from_wire).transpose()?;
    let first_stage = w.first_stage_diagnostics.as_ref().map(first_stage_from_wire).transpose()?;
    let mut estimate = EffectEstimate::from_parts(
        w.ate,
        w.se_analytic,
        BootstrapAccounting {
            se: w.se_bootstrap,
            replicates_ok: w.bootstrap_replicates_ok,
            replicates_failed: w.bootstrap_replicates_failed,
            cancelled: w.bootstrap_cancelled,
            early_stopped: w.bootstrap_early_stopped,
        },
        assumptions_from_wire(&w.assumptions)?,
        overlap,
        overlap_report,
        w.retained_memory_bytes,
    )
    .with_first_stage_diagnostics(first_stage);
    if w.analytic_reference_df.is_some_and(|df| !df.is_finite() || df < 0.0) {
        return Err(IoError::Convert(
            "analytic reference degrees of freedom must be nonnegative and finite".into(),
        ));
    }
    estimate.uncertainty.analytic.kind = w.analytic_kind.map(kind_from_wire).transpose()?;
    estimate.uncertainty.analytic.reference_df = w.analytic_reference_df;
    estimate.uncertainty.block.resampling = w
        .block_resampling
        .map(|r| {
            if !r.kernel_bias.is_finite() {
                return Err(IoError::Convert("block kernel bias must be finite".into()));
            }
            Ok(antecedent_estimate::BlockResampling {
                block_length: usize::try_from(r.block_length).map_err(|_| IoError::TooLarge)?,
                rows: usize::try_from(r.rows).map_err(|_| IoError::TooLarge)?,
                kernel_bias: r.kernel_bias,
            })
        })
        .transpose()?;
    estimate.uncertainty.block.family = w.block_family.map(family_from_wire);
    estimate.provenance.n_obs = w.n_obs;

    if let Some(table) = w.score_table {
        estimate = estimate.with_score_table(Some(score_table_from_wire(table)?));
    }
    if let Some(cdf) = w.exceedance_cdf {
        estimate = estimate.with_exceedance_cdf(Some(cdf.into()));
    }
    estimate = estimate.with_monotone_rearranged(w.monotone_rearranged);
    if let Some(c) = w.joint_covariance {
        if c.dim == 0
            || c.values.len() != c.dim.saturating_mul(c.dim)
            || c.values.iter().any(|v| !v.is_finite())
        {
            return Err(IoError::Convert("invalid joint covariance".into()));
        }
        estimate.joint.covariance =
            Some(antecedent_estimate::JointCovariance { dim: c.dim, values: c.values.into() });
    }
    estimate.scores.inference = w.score_inference.map(score_inference_from_wire).transpose()?;
    estimate.family.simultaneous_interval = w.simultaneous_interval;
    estimate.family.adjusted_p_values = w.adjusted_p_values;
    estimate.family.contrast = w.family_contrast;
    estimate.family.contrast_interval = w.family_contrast_interval;
    estimate.diagnostics.evalue = w.evalue;
    estimate.diagnostics.evalue_threshold = w.evalue_threshold;
    estimate.provenance.preparation = w.preparation.map(|p| {
        std::sync::Arc::new(
            antecedent_estimate::PreparationSummary::from_parts(
                p.backend,
                p.route_bound,
                p.rows_admitted,
                p.receipt_digest,
            )
            .with_resources(p.admitted_bytes, p.budget_bytes, p.delivery),
        )
    });
    estimate.family.candidate_selection =
        w.candidate_selection.as_ref().map(candidate_selection_from_wire).transpose()?;
    estimate.scenarios.effects = w.scenario_effects.map(Into::into);
    estimate.scenarios.intervals = w.scenario_intervals.map(Into::into);
    estimate.joint.interaction_structurally_zero = w.interaction_structurally_zero.unwrap_or(false);
    estimate.heterogeneity.unit_effects_homogeneous = w.unit_effects_homogeneous.unwrap_or(false);
    if let Some(model) = w.fitted_effect {
        model.validate().map_err(|e| IoError::Convert(e.to_string()))?;
        estimate.heterogeneity.fitted_effect = Some(std::sync::Arc::new(model));
    }
    estimate.heterogeneity.cate = w.cate.map(Into::into);
    estimate.heterogeneity.cate_se = w.cate_se.map(Into::into);
    estimate.heterogeneity.cate_leaf_dispersion = w.cate_leaf_dispersion.map(Into::into);
    estimate.crossfit.outcome_oof_r2 = w.outcome_oof_r2;
    estimate.crossfit.treatment_oof_logloss = w.treatment_oof_logloss;
    estimate.crossfit.folds = w.crossfit_folds;
    estimate.crossfit.seed = w.crossfit_seed;
    estimate.crossfit.learner_provenance = w
        .learner_provenance
        .into_iter()
        .map(|(spec, implementation, version)| antecedent_estimate::LearnerProvenance {
            spec,
            implementation,
            version,
        })
        .collect();
    Ok(estimate)
}

// Infinite interval endpoints are meaningful for unbounded confidence sets.
fn valid_interval((lo, hi, level): (f64, f64, f64)) -> bool {
    !lo.is_nan() && !hi.is_nan() && lo <= hi && level.is_finite() && level > 0.0 && level < 1.0
}

fn validate_effect_estimate_wire(w: &EffectEstimateWire) -> Result<(), IoError> {
    let nonnegative = |v: f64| v.is_finite() && v >= 0.0;
    let probability = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
    let finite_values = |v: &Vec<f64>| v.iter().all(|x| x.is_finite());
    let nonnegative_values = |v: &Vec<f64>| v.iter().all(|&x| nonnegative(x));
    // Graph-posterior envelopes can withhold the scalar without a marker in this wire.
    if !(w.ate.is_finite() || w.ate.is_nan())
        // AnalyticSe documents NaN as the sentinel for withheld uncertainty.
        || !(w.se_analytic.is_nan() || nonnegative(w.se_analytic))
        || w.se_bootstrap.is_some_and(|v| !nonnegative(v))
        || (w.se_bootstrap.is_some() && w.bootstrap_replicates_ok.is_some_and(|n| n < 2))
        || w.overlap_clip.is_some_and(|v| !v.is_finite() || v <= 0.0 || v >= 0.5)
        || w.overlap_trim.is_some_and(|v| !v.is_finite() || v <= 0.0 || v >= 0.5)
        || w.evalue.is_some_and(|v| v.is_nan() || v < 1.0)
        || w.evalue_threshold.is_some_and(f64::is_nan)
        || w.adjusted_p_values.is_some_and(|(a, b)| !probability(a) || !probability(b))
        || w.simultaneous_interval.is_some_and(|v| !valid_interval(v))
        || w.family_contrast_interval.is_some_and(|v| !valid_interval(v))
        || w.family_contrast.is_some_and(|(value, se)| !value.is_finite() || !nonnegative(se))
        || w.scenario_intervals.as_ref().is_some_and(|v| v.iter().any(|&(lo, hi)| lo.is_nan() || hi.is_nan() || lo > hi))
        || w.scenario_effects.as_ref().is_some_and(|v| !finite_values(v))
        || w.cate.as_ref().is_some_and(|v| !finite_values(v))
        || w.cate_se.as_ref().is_some_and(|v| !nonnegative_values(v))
        || w.cate_leaf_dispersion.as_ref().is_some_and(|v| !nonnegative_values(v))
        || w.cate.as_ref().zip(w.cate_se.as_ref()).is_some_and(|(a, b)| a.len() != b.len())
        || w.cate.as_ref().zip(w.cate_leaf_dispersion.as_ref()).is_some_and(|(a, b)| a.len() != b.len())
        || w.exceedance_cdf.as_ref().is_some_and(|v| !finite_values(v))
        || w.outcome_oof_r2.is_some_and(|v| !v.is_finite() || v > 1.0)
        || w.treatment_oof_logloss.is_some_and(|v| !nonnegative(v))
        || w.crossfit_folds.is_some_and(|v| v < 2)
    {
        return Err(IoError::Convert("invalid effect estimate evidence".into()));
    }
    Ok(())
}

fn first_stage_from_wire(
    diagnostic: &FirstStageDiagnosticsWire,
) -> Result<FirstStageDiagnostics, IoError> {
    if !diagnostic.f_statistic.is_finite()
        || diagnostic.f_statistic < 0.0
        || !diagnostic.partial_r2.is_finite()
        || !(0.0..=1.0).contains(&diagnostic.partial_r2)
        || diagnostic.df1 == 0
        || diagnostic.df2 == 0
        || diagnostic.anderson_rubin.is_some_and(|v| !valid_interval(v))
        || (diagnostic.anderson_rubin.is_some() && diagnostic.uncertainty_withheld.is_some())
    {
        return Err(IoError::Convert("invalid first-stage diagnostic evidence".into()));
    }
    Ok(FirstStageDiagnostics {
        f_statistic: diagnostic.f_statistic,
        df1: usize::try_from(diagnostic.df1).map_err(|_| IoError::TooLarge)?,
        df2: usize::try_from(diagnostic.df2).map_err(|_| IoError::TooLarge)?,
        partial_r2: diagnostic.partial_r2,
        anderson_rubin: diagnostic.anderson_rubin,
        uncertainty_withheld: diagnostic.uncertainty_withheld.map(UncertaintyWithheldWire::tag),
    })
}

fn score_inference_from_wire(
    s: ScoreInferenceWire,
) -> Result<antecedent_estimate::scores::ScoreInference, IoError> {
    let n = s.raw_means.len();
    if n == 0
        || s.lower.len() != n
        || s.upper.len() != n
        || s.event_n_eff.len() != n
        || s.threshold_supported.len() != n
        || s.raw_means.iter().any(|v| !v.is_finite())
        || s.lower.iter().zip(&s.upper).zip(&s.threshold_supported).zip(&s.raw_means).any(
            |(((&lo, &hi), &supported), &mean)| {
                if supported {
                    lo.is_nan() || hi.is_nan() || lo > mean || mean > hi
                } else {
                    !((lo.is_nan() && hi.is_nan()) || (!lo.is_nan() && !hi.is_nan() && lo <= hi))
                }
            },
        )
        || s.critical_value.is_nan()
        || s.critical_value < 0.0
        || s.event_n_eff.iter().any(|v| !v.is_finite() || *v < 0.0)
        || !s.n_eff.is_finite()
        || s.n_eff < 0.0
        || s.n_eff_by_arm.iter().any(|v| !v.is_finite() || *v < 0.0)
        || s.propensity_range.is_some_and(|(lo, hi)| {
            !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi > 1.0 || lo > hi
        })
        || !s.level.is_finite()
        || s.level <= 0.0
        || s.level >= 1.0
    {
        return Err(IoError::Convert("invalid score inference".into()));
    }
    Ok(antecedent_estimate::scores::ScoreInference {
        raw_means: s.raw_means,
        lower: s.lower,
        upper: s.upper,
        level: s.level,
        critical_value: s.critical_value,
        event_n_eff: s.event_n_eff,
        threshold_supported: s.threshold_supported,
        support: antecedent_estimate::WeightedSupport {
            n_eff: s.n_eff,
            n_eff_by_arm: s.n_eff_by_arm,
            propensity_range: s.propensity_range,
            overlap_ok: s.overlap_ok,
        },
    })
}

fn candidate_selection_from_wire(
    s: &CandidateSelectionWire,
) -> Result<antecedent_estimate::CandidateSelectionRecord, IoError> {
    let screen: std::collections::BTreeSet<_> = s.screen_rows.iter().collect();
    let estimate: std::collections::BTreeSet<_> = s.estimate_rows.iter().collect();
    if s.winner_index.is_some_and(|i| i >= s.family_size)
        || screen.len() != s.screen_rows.len()
        || estimate.len() != s.estimate_rows.len()
        || (s.disjoint
            && (screen.is_empty() || estimate.is_empty() || !screen.is_disjoint(&estimate)))
    {
        return Err(IoError::Convert("invalid candidate selection provenance".into()));
    }
    Ok(antecedent_estimate::CandidateSelectionRecord {
        screen_id: std::sync::Arc::from(s.screen_id.as_str()),
        procedure: std::sync::Arc::from(s.procedure.as_str()),
        winner_index: s.winner_index,
        family_size: s.family_size,
        screen_rows: s
            .screen_rows
            .iter()
            .copied()
            .map(antecedent_data::GlobalId::new)
            .collect::<Vec<_>>()
            .into(),
        estimate_rows: s
            .estimate_rows
            .iter()
            .copied()
            .map(antecedent_data::GlobalId::new)
            .collect::<Vec<_>>()
            .into(),
        disjoint: s.disjoint,
    })
}

fn overlap_report_from_wire(wire: OverlapReportWire) -> Result<OverlapReport, IoError> {
    let finite = [
        wire.propensity_min,
        wire.propensity_max,
        wire.excluded_fraction,
        wire.target_population_support,
        wire.retained_fraction,
    ]
    .into_iter()
    .chain(wire.ess)
    .all(f64::is_finite);
    if !finite
        || wire.propensity_min < 0.0
        || wire.propensity_min > wire.propensity_max
        || wire.propensity_max > 1.0
        || wire.ess.is_some_and(|value| value < 0.0)
        || !(0.0..=1.0).contains(&wire.excluded_fraction)
        || !(0.0..=1.0).contains(&wire.target_population_support)
        || !(0.0..=1.0).contains(&wire.retained_fraction)
        || wire.excluded_regions.iter().any(|region| {
            !region.low.is_finite()
                || !region.high.is_finite()
                || region.low < 0.0
                || region.low > region.high
                || region.high > 1.0
        })
    {
        return Err(IoError::Convert("invalid propensity-overlap evidence".into()));
    }
    if let Some(sensitivity) = &wire.clip_sensitivity {
        let len = sensitivity.thresholds.len();
        if len == 0
            || sensitivity.ess.len() != len
            || sensitivity.treated_ess.len() != len
            || sensitivity.control_ess.len() != len
            || sensitivity.extreme_weight_counts.len() != len
            || sensitivity
                .thresholds
                .iter()
                .chain(&sensitivity.ess)
                .chain(&sensitivity.treated_ess)
                .chain(&sensitivity.control_ess)
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(IoError::Convert("invalid overlap clip-sensitivity evidence".into()));
        }
    }
    Ok(OverlapReport {
        propensity_min: wire.propensity_min,
        propensity_max: wire.propensity_max,
        ess: wire.ess,
        extreme_weight_count: wire.extreme_weight_count,
        excluded_fraction: wire.excluded_fraction,
        target_population_support: wire.target_population_support,
        excluded_regions: wire
            .excluded_regions
            .into_iter()
            .map(|region| PropensityInterval { low: region.low, high: region.high })
            .collect::<Vec<_>>()
            .into(),
        clip: wire.clip,
        trim: wire.trim,
        retained_fraction: wire.retained_fraction,
        clip_sensitivity: wire.clip_sensitivity.map(|sensitivity| ClipSensitivity {
            thresholds: sensitivity.thresholds.into(),
            ess: sensitivity.ess.into(),
            treated_ess: sensitivity.treated_ess.into(),
            control_ess: sensitivity.control_ess.into(),
            extreme_weight_counts: sensitivity.extreme_weight_counts.into(),
        }),
    })
}

/// Encode refutation.
#[must_use]
pub fn refutation_to_wire(r: &RefutationReport) -> RefutationReportWire {
    RefutationReportWire {
        refuter: r.refuter.to_string(),
        original_ate: r.original_ate,
        refuted_ate: r.refuted_ate,
        comparison: r.comparison,
        informative: r.informative,
        passed: r.passed,
        failure_condition: r.failure_condition.as_ref().map(ToString::to_string),
        replicates: r.replicates,
    }
}

/// Decode refutation.
#[must_use]
pub fn refutation_from_wire(w: &RefutationReportWire) -> RefutationReport {
    RefutationReport::new(
        Arc::from(w.refuter.as_str()),
        w.original_ate,
        w.refuted_ate,
        w.comparison,
        w.informative,
        w.passed,
        w.failure_condition.as_ref().map(|s| Arc::<str>::from(s.as_str())),
        w.replicates,
    )
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind,
        DiagnosticSeverity, IdentificationStatus, VariableId,
    };
    use antecedent_expr::CausalExprArena;
    use antecedent_expr::IdentifiedEstimand;
    use antecedent_graph::DenseNodeId;
    use antecedent_identify::{
        DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
        IdentificationResultParts,
    };

    use super::*;
    use crate::trace::{AssumptionRecordWire, AssumptionTagWire};
    use antecedent_io_core::analysis_wire::*;

    #[test]
    fn effect_wire_rejects_malformed_evidence_and_preserves_unbounded_values() {
        let effect =
            EffectEstimate::new(1.0, 0.2, AssumptionSet::new(), OverlapPolicy::ExplicitOverride);
        let valid = effect_estimate_to_wire(&effect);
        let mutations: &[fn(&mut EffectEstimateWire)] = &[
            |w| w.ate = f64::INFINITY,
            |w| w.se_analytic = -1.0,
            |w| w.se_analytic = f64::INFINITY,
            |w| w.se_bootstrap = Some(f64::NAN),
            |w| {
                w.se_bootstrap = Some(0.1);
                w.bootstrap_replicates_ok = Some(1);
            },
            |w| w.overlap_clip = Some(0.5),
            |w| w.overlap_trim = Some(0.0),
            |w| w.evalue = Some(f64::NAN),
            |w| w.evalue_threshold = Some(f64::NAN),
            |w| w.adjusted_p_values = Some((0.1, 1.1)),
            |w| w.simultaneous_interval = Some((2.0, 1.0, 0.95)),
            |w| w.family_contrast_interval = Some((0.0, 1.0, 1.0)),
            |w| w.family_contrast = Some((1.0, -0.1)),
            |w| w.scenario_intervals = Some(vec![(f64::NAN, 1.0)]),
            |w| w.scenario_effects = Some(vec![f64::INFINITY]),
            |w| w.cate = Some(vec![f64::NAN]),
            |w| w.cate_se = Some(vec![-0.1]),
            |w| w.cate_leaf_dispersion = Some(vec![f64::INFINITY]),
            |w| {
                w.cate = Some(vec![1.0]);
                w.cate_se = Some(vec![0.1, 0.2]);
            },
            |w| w.exceedance_cdf = Some(vec![f64::NAN]),
            |w| w.outcome_oof_r2 = Some(1.1),
            |w| w.treatment_oof_logloss = Some(-0.1),
            |w| w.crossfit_folds = Some(1),
        ];
        for (index, mutate) in mutations.iter().enumerate() {
            let mut wire = valid.clone();
            mutate(&mut wire);
            assert!(effect_estimate_from_wire(&wire).is_err(), "mutation {index}");
        }
        let mut unbounded = valid;
        unbounded.evalue = Some(f64::INFINITY);
        unbounded.evalue_threshold = Some(0.0);
        unbounded.simultaneous_interval = Some((f64::NEG_INFINITY, f64::INFINITY, 0.95));
        unbounded.scenario_intervals = Some(vec![(f64::NEG_INFINITY, 1.0)]);
        unbounded.exceedance_cdf = Some(vec![-0.1, 1.1]);
        assert!(effect_estimate_from_wire(&unbounded).is_ok());

        // IV and point-only estimators use NaN for withheld analytic uncertainty,
        // including routes with no analytic-kind or first-stage marker.
        let mut withheld = unbounded.clone();
        withheld.se_analytic = f64::NAN;
        assert!(withheld.analytic_kind.is_none());
        let decoded = effect_estimate_from_wire(&withheld).unwrap();
        assert!(effect_estimate_to_wire(&decoded).se_analytic.is_nan());
        withheld.first_stage_diagnostics = Some(FirstStageDiagnosticsWire {
            f_statistic: 2.0,
            df1: 1,
            df2: 10,
            partial_r2: 0.2,
            anderson_rubin: Some((f64::NEG_INFINITY, f64::INFINITY, 0.95)),
            uncertainty_withheld: None,
        });
        let decoded = effect_estimate_from_wire(&withheld).unwrap();
        assert!(effect_estimate_to_wire(&decoded).se_analytic.is_nan());

        // Scenario envelopes and exceedance grids do not publish a scalar ATE.
        let mut envelope = withheld.clone();
        envelope.ate = f64::NAN;
        envelope.scenario_effects = None;
        envelope.exceedance_cdf = None;
        assert!(
            effect_estimate_to_wire(&effect_estimate_from_wire(&envelope).unwrap()).ate.is_nan()
        );
        envelope.exceedance_cdf = None;
        envelope.scenario_effects = Some(vec![1.0, 2.0]);
        assert!(
            effect_estimate_to_wire(&effect_estimate_from_wire(&envelope).unwrap()).ate.is_nan()
        );
        envelope.scenario_effects = None;
        envelope.exceedance_cdf = Some(vec![0.2, 0.4]);
        assert!(
            effect_estimate_to_wire(&effect_estimate_from_wire(&envelope).unwrap()).ate.is_nan()
        );
    }

    #[test]
    fn score_and_first_stage_wire_validate_support_and_sets() {
        let score = ScoreInferenceWire {
            raw_means: vec![0.2, 0.4],
            lower: vec![f64::NEG_INFINITY, f64::NAN],
            upper: vec![f64::INFINITY, f64::NAN],
            level: 0.95,
            critical_value: f64::INFINITY,
            event_n_eff: vec![2.0, 0.0],
            threshold_supported: vec![true, false],
            n_eff: 2.0,
            n_eff_by_arm: vec![1.0, 1.0],
            propensity_range: Some((0.0, 1.0)),
            overlap_ok: true,
        };
        assert!(score_inference_from_wire(score.clone()).is_ok());
        let mutations: &[fn(&mut ScoreInferenceWire)] = &[
            |s| s.critical_value = f64::NAN,
            |s| s.event_n_eff[0] = -1.0,
            |s| s.n_eff = f64::INFINITY,
            |s| s.n_eff_by_arm[0] = -1.0,
            |s| s.propensity_range = Some((0.8, 0.2)),
            |s| s.lower[0] = 0.3,
            |s| s.upper[1] = 1.0,
        ];
        for (index, mutate) in mutations.iter().enumerate() {
            let mut invalid = score.clone();
            mutate(&mut invalid);
            assert!(score_inference_from_wire(invalid).is_err(), "mutation {index}");
        }
        let diagnostic = FirstStageDiagnosticsWire {
            f_statistic: 1.0,
            df1: 1,
            df2: 2,
            partial_r2: 0.2,
            anderson_rubin: Some((f64::NEG_INFINITY, f64::INFINITY, 0.95)),
            uncertainty_withheld: None,
        };
        assert!(first_stage_from_wire(&diagnostic).is_ok());
        let mut invalid = diagnostic.clone();
        invalid.anderson_rubin = Some((2.0, 1.0, 0.95));
        assert!(first_stage_from_wire(&invalid).is_err());
        invalid = diagnostic;
        invalid.uncertainty_withheld = Some(UncertaintyWithheldWire::AndersonRubinSetIsUnion);
        assert!(first_stage_from_wire(&invalid).is_err());
    }

    #[test]
    fn learner_outputs_survive_wire_round_trip() {
        let mut effect = EffectEstimate::new(
            2.0,
            0.1,
            AssumptionSet::new(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        )
        .with_cate(Some(vec![1.0, 3.0].into()))
        .with_cate_se(Some(vec![0.2, 0.4].into()))
        .with_cate_leaf_dispersion(Some(vec![0.3, 0.5].into()));
        effect.crossfit.learner_provenance.push(antecedent_estimate::LearnerProvenance {
            spec: "ridge".into(),
            implementation: "faer".into(),
            version: "0.24".into(),
        });
        effect.crossfit.outcome_oof_r2 = Some(0.8);
        effect.crossfit.treatment_oof_logloss = Some(0.4);
        effect.crossfit.folds = Some(5);
        effect.crossfit.seed = Some(42);
        effect.uncertainty.analytic.kind =
            Some(antecedent_estimate::AnalyticSeKind::NeweyWest { lag: 3 });
        effect.uncertainty.analytic.reference_df = Some(7.0);
        effect.uncertainty.block.resampling = Some(antecedent_estimate::BlockResampling {
            block_length: 4,
            rows: 40,
            kernel_bias: 1.2,
        });
        effect.uncertainty.block.family = Some(antecedent_estimate::CircularBlockFamily::Mediation);
        effect.provenance.n_obs = Some(40);
        let wire = effect_estimate_to_wire(&effect);
        let bytes = serde_json::to_vec(&wire).unwrap();
        let decoded: EffectEstimateWire = serde_json::from_slice(&bytes).unwrap();
        let restored = effect_estimate_from_wire(&decoded).unwrap();
        assert_eq!(restored.uncertainty.analytic, effect.uncertainty.analytic);
        assert_eq!(restored.uncertainty.block, effect.uncertainty.block);
        assert_eq!(restored.provenance.n_obs, effect.provenance.n_obs);
        let critical = |e: &EffectEstimate| {
            antecedent_stats::student_t_ppf(0.975, e.uncertainty.analytic.reference_df.unwrap())
        };
        assert_eq!(
            (
                restored.ate - critical(&restored) * restored.se_analytic(),
                restored.ate + critical(&restored) * restored.se_analytic()
            ),
            (
                effect.ate - critical(&effect) * effect.se_analytic(),
                effect.ate + critical(&effect) * effect.se_analytic()
            )
        );
        let mut legacy = serde_json::to_value(&wire).unwrap();
        for field in
            ["analytic_kind", "analytic_reference_df", "block_resampling", "block_family", "n_obs"]
        {
            legacy.as_object_mut().unwrap().remove(field);
        }
        let legacy: EffectEstimateWire = serde_json::from_value(legacy).unwrap();
        let legacy = effect_estimate_from_wire(&legacy).unwrap();
        assert_eq!(legacy.uncertainty.analytic.kind, None);
        assert_eq!(legacy.uncertainty.analytic.reference_df, None);
        assert_eq!(legacy.uncertainty.block, antecedent_estimate::BlockSeRecord::default());
        assert_eq!(legacy.provenance.n_obs, None);
        let mut zero_df = wire.clone();
        zero_df.analytic_reference_df = Some(0.0);
        let zero_df = effect_estimate_from_wire(&zero_df).unwrap();
        assert_eq!(zero_df.uncertainty.analytic.reference_df, Some(0.0));
        assert_eq!(restored.crossfit.outcome_oof_r2, effect.crossfit.outcome_oof_r2);
        assert_eq!(restored.crossfit.treatment_oof_logloss, effect.crossfit.treatment_oof_logloss);
        assert_eq!(restored.crossfit.folds, effect.crossfit.folds);
        assert_eq!(restored.crossfit.seed, effect.crossfit.seed);
        assert_eq!(restored.heterogeneity.cate, effect.heterogeneity.cate);
        assert_eq!(restored.heterogeneity.cate_se, effect.heterogeneity.cate_se);
        assert_eq!(
            restored.heterogeneity.cate_leaf_dispersion,
            effect.heterogeneity.cate_leaf_dispersion
        );
        assert_eq!(restored.crossfit.learner_provenance, effect.crossfit.learner_provenance);
    }

    fn empty_id_result(status: IdentificationStatus) -> IdentificationResult {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        IdentificationResult::from_parts(IdentificationResultParts {
            status,
            query: CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(t, y)),
            estimands: Vec::new(),
            arena: CausalExprArena::new(),
            derivation: DerivationTrace::default(),
            required_assumptions: AssumptionSet::default(),
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord::default(),
            hedge: None,
        })
    }

    fn estimand_wire(functional: u32, rd_design: Option<RdDesignWire>) -> IdentifiedEstimandWire {
        IdentifiedEstimandWire {
            method: "backdoor".into(),
            adjustment_set: Vec::new(),
            instruments: Vec::new(),
            mediators: Vec::new(),
            functional,
            rd_design,
        }
    }

    #[test]
    fn dangling_estimand_functional_is_rejected_not_panicked_on() {
        let mut wire =
            identification_to_wire(&empty_id_result(IdentificationStatus::NotIdentified)).unwrap();
        assert!(wire.arena.nodes.is_empty());
        wire.estimands = vec![estimand_wire(4_000_000_000, None)];
        let error = identification_from_wire(&wire).unwrap_err().to_string();
        assert!(error.contains("outside its expression arena"), "{error}");
        // Index equal to the arena length is the first out-of-range value.
        wire.estimands = vec![estimand_wire(0, None)];
        assert!(identification_from_wire(&wire).is_err());
    }

    #[test]
    fn rd_design_needs_finite_cutoff_and_positive_bandwidth() {
        let mut arena = CausalExprArena::new();
        let root = arena.backdoor_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[],
            antecedent_core::Value::Bool(true),
            antecedent_core::Value::Bool(false),
        );
        let result = IdentificationResult::from_parts(IdentificationResultParts {
            status: IdentificationStatus::NonparametricallyIdentified,
            query: CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            )),
            estimands: Vec::new(),
            arena,
            derivation: DerivationTrace::default(),
            required_assumptions: AssumptionSet::default(),
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord::default(),
            hedge: None,
        });
        let mut wire = identification_to_wire(&result).unwrap();
        for (cutoff, bandwidth) in
            [(f64::NAN, 1.0), (0.5, 0.0), (0.5, -1.0), (0.5, f64::INFINITY), (0.5, f64::NAN)]
        {
            wire.estimands = vec![estimand_wire(
                root.raw(),
                Some(RdDesignWire { running_variable: 0, cutoff, bandwidth }),
            )];
            assert!(identification_from_wire(&wire).is_err(), "{cutoff} {bandwidth}");
        }
        wire.estimands = vec![estimand_wire(
            root.raw(),
            Some(RdDesignWire { running_variable: 0, cutoff: 0.5, bandwidth: 0.25 }),
        )];
        assert!(identification_from_wire(&wire).is_ok());
    }

    fn bow_hedge() -> antecedent_identify::HedgeCertificate {
        // Bow graph X -> Y with X <-> Y: F = {X, Y}, F' = {Y}.
        let ids = |raw: &[u32]| -> Arc<[VariableId]> {
            raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>().into()
        };
        let dense = |raw: &[u32]| -> Arc<[DenseNodeId]> {
            raw.iter().copied().map(DenseNodeId::from_raw).collect::<Vec<_>>().into()
        };
        antecedent_identify::HedgeCertificate {
            f: ids(&[0, 1]),
            f_prime: ids(&[1]),
            f_dense: dense(&[0, 1]),
            f_prime_dense: dense(&[1]),
            problem: Some(antecedent_identify::HedgeProblem {
                variables: ids(&[0, 1]),
                directed: vec![(0, 1)].into(),
                bidirected: vec![(0, 1)].into(),
                treatments: ids(&[0]),
                outcomes: ids(&[1]),
            }),
        }
    }

    #[test]
    fn carried_hedge_is_reverified_on_load() {
        let mut result = empty_id_result(IdentificationStatus::NotIdentified);
        result.hedge = Some(bow_hedge());
        let wire = identification_to_wire(&result).unwrap();
        // Untouched: verifies (X -> Y with X <-> Y really is a hedge).
        assert!(identification_from_wire(&wire).is_ok());
        // Remove the confounding edge: F is no longer bidirected-connected, so it is no hedge.
        let mut no_confounding = wire.clone();
        no_confounding.hedge.as_mut().unwrap().problem.as_mut().unwrap().bidirected.clear();
        let err = identification_from_wire(&no_confounding).unwrap_err().to_string();
        assert!(err.contains("not a hedge"), "{err}");
        // The witness cannot certify a query whose treatment it does not contain.
        let mut wrong_treatment = wire;
        let problem = wrong_treatment.hedge.as_mut().unwrap().problem.as_mut().unwrap();
        problem.treatments = vec![1];
        problem.outcomes = vec![0];
        assert!(identification_from_wire(&wrong_treatment).is_err());
    }

    #[test]
    fn hedge_witness_survives_save_and_load() {
        let mut result = empty_id_result(IdentificationStatus::NotIdentified);
        result.hedge = Some(bow_hedge());
        let wire = identification_to_wire(&result).unwrap();
        assert!(wire.hedge.is_some());
        let bytes = serde_json::to_vec(&wire).unwrap();
        let decoded: IdentificationResultWire = serde_json::from_slice(&bytes).unwrap();
        let back = identification_from_wire(&decoded).unwrap();
        assert_eq!(back.hedge, Some(bow_hedge()));
        // A refusal without a witness stays without one.
        let plain =
            identification_to_wire(&empty_id_result(IdentificationStatus::NotIdentified)).unwrap();
        assert!(identification_from_wire(&plain).unwrap().hedge.is_none());
    }

    #[test]
    fn hedge_witness_is_rejected_when_malformed_or_on_an_identified_result() {
        let mut result = empty_id_result(IdentificationStatus::NotIdentified);
        result.hedge = Some(bow_hedge());
        let wire = identification_to_wire(&result).unwrap();

        let mut identified = wire.clone();
        identified.status = "nonparametrically_identified".into();
        assert!(identification_from_wire(&identified).is_err());

        let mut not_nested = wire.clone();
        not_nested.hedge.as_mut().unwrap().f_prime = vec![7];
        assert!(identification_from_wire(&not_nested).is_err());

        let mut empty_inner = wire.clone();
        let hedge = empty_inner.hedge.as_mut().unwrap();
        hedge.f_prime.clear();
        hedge.f_prime_dense.clear();
        assert!(identification_from_wire(&empty_inner).is_err());

        let mut unordered = wire;
        unordered.hedge.as_mut().unwrap().f_dense = vec![1, 0];
        assert!(identification_from_wire(&unordered).is_err());
    }

    #[test]
    fn restricted_status_wire_round_trips() {
        for status in [
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            IdentificationStatus::IdentifiedUnderPriorRestrictions,
        ] {
            let wire = identification_to_wire(&empty_id_result(status)).unwrap();
            let back = identification_from_wire(&wire).unwrap();
            assert_eq!(back.status, status);
        }
    }

    #[test]
    fn undetermined_is_not_a_wire_status() {
        let mut wire =
            identification_to_wire(&empty_id_result(IdentificationStatus::NotIdentified)).unwrap();
        wire.status = "undetermined".into();
        let err = identification_from_wire(&wire).unwrap_err();
        assert!(err.to_string().contains("unknown IdentificationStatus `undetermined`"), "{err}");
    }

    fn descriptive_assumption() -> AssumptionRecordWire {
        AssumptionRecordWire {
            assumption: AssumptionTagWire::Custom {
                id: "stable.assumption.id".into(),
                description: Some("scientifically meaningful description".into()),
            },
            source: "derived:diagnostic-7".into(),
            scope: "variables:[0,1]".into(),
            status: "contradicted".into(),
        }
    }

    #[test]
    fn every_withheld_reason_round_trips_through_its_tag_and_the_wire_spelling() {
        for tag in [
            "anderson_rubin_requires_homoskedastic",
            "anderson_rubin_set_is_union",
            "anderson_rubin_set_empty",
            "anderson_rubin_requires_excluded_instruments",
            "anderson_rubin_invalid_level",
            "anderson_rubin_critical_value_failed",
        ] {
            let reason = UncertaintyWithheldWire::from_tag(tag).expect("defined reason");
            assert_eq!(reason.tag(), tag);
            assert_eq!(serde_json::to_string(&reason).unwrap(), format!("\"{tag}\""));
        }
        assert_eq!(UncertaintyWithheldWire::from_tag("other"), None);
        assert_eq!(
            serde_json::to_string(&EstimateOverlapPolicyWire::RequireDiagnostics).unwrap(),
            "\"require_diagnostics\""
        );
        assert_eq!(
            serde_json::to_string(&EstimateOverlapPolicyWire::ExplicitOverride).unwrap(),
            "\"explicit_override\""
        );
    }

    #[test]
    fn effect_estimate_preserves_assumption_evidence() {
        let wire = EffectEstimateWire {
            ate: 1.0,
            se_analytic: 0.1,
            analytic_kind: None,
            analytic_reference_df: None,
            block_resampling: None,
            block_family: None,
            n_obs: None,
            se_bootstrap: Some(0.2),
            bootstrap_replicates_ok: Some(99),
            bootstrap_replicates_failed: Some(1),
            bootstrap_cancelled: true,
            bootstrap_early_stopped: true,
            assumptions: vec![descriptive_assumption()],
            overlap_policy: EstimateOverlapPolicyWire::RequireDiagnostics,
            overlap_clip: Some(0.01),
            overlap_trim: Some(0.02),
            overlap_report: Some(OverlapReportWire {
                propensity_min: 0.01,
                propensity_max: 0.98,
                ess: Some(42.0),
                extreme_weight_count: 2,
                excluded_fraction: 0.1,
                target_population_support: 0.9,
                excluded_regions: vec![PropensityIntervalWire { low: 0.0, high: 0.02 }],
                clip: Some(0.01),
                trim: Some(0.02),
                retained_fraction: 0.9,
                clip_sensitivity: Some(ClipSensitivityWire {
                    thresholds: vec![0.005, 0.01],
                    ess: vec![40.0, 42.0],
                    treated_ess: vec![20.0, 21.0],
                    control_ess: vec![20.0, 21.0],
                    extreme_weight_counts: vec![3, 2],
                }),
            }),
            first_stage_diagnostics: Some(FirstStageDiagnosticsWire {
                f_statistic: 12.0,
                df1: 1,
                df2: 98,
                partial_r2: 0.2,
                anderson_rubin: Some((1.0, 3.0, 0.95)),
                uncertainty_withheld: None,
            }),
            retained_memory_bytes: Some(4096),
            score_table: None,
            joint_covariance: None,
            score_inference: None,
            scenario_effects: None,
            simultaneous_interval: None,
            adjusted_p_values: None,
            family_contrast: None,
            family_contrast_interval: None,
            scenario_intervals: None,
            exceedance_cdf: None,
            monotone_rearranged: false,
            interaction_structurally_zero: None,
            unit_effects_homogeneous: None,
            cate: None,
            fitted_effect: None,
            cate_se: None,
            cate_leaf_dispersion: None,
            outcome_oof_r2: None,
            treatment_oof_logloss: None,
            crossfit_folds: None,
            crossfit_seed: None,
            learner_provenance: Vec::new(),
            evalue: None,
            evalue_threshold: None,
            candidate_selection: None,
            preparation: Some(PreparationWire {
                backend: "direct_arrow".into(),
                route_bound: "rows_only".into(),
                rows_admitted: Some(100),
                receipt_digest: "ab".repeat(32),
                admitted_bytes: Some(2_400),
                budget_bytes: Some(1 << 20),
                delivery: Some("finite".into()),
            }),
        };
        // A present fitted model must survive conversion or be refused, including
        // unknown versions at either layer; never silently erase it.
        let mut fitted = wire.clone();
        fitted.fitted_effect = Some(antecedent_estimate::FittedEffect {
            version: 1,
            features: vec![0],
            intercept: true,
            predictor: antecedent_learn::PortablePredictor {
                version: 1,
                columns: 2,
                provenance: antecedent_learn::learner::LearnerProvenance {
                    spec: "linear".into(),
                    implementation: "test".into(),
                    version: "1".into(),
                },
                model: antecedent_learn::PredictionMap::Linear {
                    coefficients: vec![1.0, 2.0],
                    logistic: false,
                },
            },
            support: None,
        });
        let converted = effect_estimate_from_wire(&fitted).unwrap();
        assert_eq!(effect_estimate_to_wire(&converted).fitted_effect, fitted.fitted_effect);
        for predictor_version in [false, true] {
            let mut changed = fitted.clone();
            let model = changed.fitted_effect.as_mut().unwrap();
            if predictor_version {
                model.predictor.version = 2;
            } else {
                model.version = 2;
            }
            assert!(effect_estimate_from_owned_wire(changed).is_err());
        }
        let domain = effect_estimate_from_wire(&wire).unwrap();
        assert_eq!(effect_estimate_to_wire(&domain), wire);
        let summary = domain.provenance.preparation.as_ref().unwrap();
        assert_eq!(summary.rows_admitted, Some(100));
        assert_eq!(summary.admitted_bytes, Some(2_400));
        assert_eq!(summary.budget_bytes, Some(1 << 20));
        assert_eq!(summary.delivery.as_deref(), Some("finite"));
        // A writer that recorded no resource facts round-trips as none, never as zero.
        let mut bare = wire.clone();
        if let Some(p) = bare.preparation.as_mut() {
            p.admitted_bytes = None;
            p.budget_bytes = None;
            p.delivery = None;
        }
        let bare_domain = effect_estimate_from_wire(&bare).unwrap();
        assert_eq!(effect_estimate_to_wire(&bare_domain), bare);
        let bare_json = serde_json::to_string(&bare).unwrap();
        assert!(!bare_json.contains("admitted_bytes") && !bare_json.contains("delivery"));

        let mut without = wire;
        without.preparation = None;
        let domain = effect_estimate_from_wire(&without).unwrap();
        assert!(domain.provenance.preparation.is_none());
        assert_eq!(effect_estimate_to_wire(&domain), without);
        let json = serde_json::to_string(&without).unwrap();
        assert!(!json.contains("preparation"), "an absent receipt must emit no key");
        let old: EffectEstimateWire = serde_json::from_str(&json).unwrap();
        assert_eq!(old, without);
    }

    #[test]
    fn identification_result_preserves_required_assumption_evidence() {
        let mut wire = identification_to_wire(&empty_id_result(
            IdentificationStatus::NonparametricallyIdentified,
        ))
        .unwrap();
        wire.required_assumptions = vec![descriptive_assumption()];
        let domain = identification_from_wire(&wire).unwrap();
        assert_eq!(
            identification_to_wire(&domain).unwrap().required_assumptions,
            wire.required_assumptions
        );
    }

    #[test]
    fn diagnostic_preserves_structured_evidence_fields() {
        let diagnostic = Diagnostic {
            code: Arc::from("support.overlap"),
            kind: DiagnosticKind::Scientific,
            severity: DiagnosticSeverity::Warning,
            message: Arc::from("weak overlap"),
            artifact_id: Some(Arc::from("artifact-7")),
            fields: Arc::from([
                (Arc::from("minimum_probability"), Arc::from("0.01")),
                (Arc::from("effective_sample_size"), Arc::from("12.5")),
            ]),
        };
        let wire = diagnostic_to_wire(&diagnostic);
        assert_eq!(diagnostic_from_wire(&wire).unwrap(), diagnostic);
    }

    #[test]
    fn support_kind_survives_the_wire_independent_of_its_code() {
        let diagnostic = Diagnostic::new(
            "estimate.propensity.floor",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "propensity floored",
        );
        let wire = diagnostic_to_wire(&diagnostic);
        assert_eq!(wire.kind, "support");
        assert_eq!(diagnostic_from_wire(&wire).unwrap().kind, DiagnosticKind::Support);
    }

    #[test]
    fn distribution_and_path_specific_query_identification_wire() {
        use antecedent_core::{
            Intervention, InterventionalDistributionQuery, PathSpecificEffectQuery, Value,
        };

        let dist_q = CausalQuery::Distribution(
            InterventionalDistributionQuery::new(
                VariableId::from_raw(1),
                [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
            )
            .with_conditioning([VariableId::from_raw(2)]),
        );
        let mut dist = empty_id_result(IdentificationStatus::NonparametricallyIdentified);
        dist.query = dist_q;
        let wire = identification_to_wire(&dist).unwrap();
        let back = identification_from_wire(&wire).unwrap();
        assert!(matches!(
            back.query,
            CausalQuery::Distribution(q) if q.conditioning.len() == 1
        ));

        let path_q = CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
                .with_path_nodes([VariableId::from_raw(1)]),
        );
        let mut path = empty_id_result(IdentificationStatus::NonparametricallyIdentified);
        path.query = path_q;
        // The estimand's functional must resolve inside the result's own expression arena
        // (identification_from_wire now rejects a dangling functional); give it one node.
        let mut arena = CausalExprArena::new();
        let root = arena.backdoor_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            &[],
            antecedent_core::Value::Bool(true),
            antecedent_core::Value::Bool(false),
        );
        path.arena = arena;
        path.estimands.push(IdentifiedEstimand::new(
            Arc::from("path_specific.natural"),
            Arc::from([]),
            Arc::from([]),
            Arc::from([]),
            root,
            None,
        ));
        let wire = identification_to_wire(&path).unwrap();
        assert_eq!(wire.estimands[0].method, "path_specific.natural");
        let back = identification_from_wire(&wire).unwrap();
        assert!(matches!(
            back.query,
            CausalQuery::PathSpecific(q) if q.path_nodes.len() == 1
        ));
        assert_eq!(back.estimands[0].method.as_ref(), "path_specific.natural");
    }
}
