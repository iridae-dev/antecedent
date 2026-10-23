//! The point estimate an estimator reports, grouped by what is read and written together.
//!
//! [`EffectEstimate`] is the scalar plus a small set of named groups: its
//! [`EstimateUncertainty`], the [`EstimateProvenance`] it was fitted under, [`EstimateDiagnostics`], the
//! [`CrossfitRecord`], the exported [`EstimateScores`], batch [`FamilyInference`], canonical
//! [`ScenarioBand`], [`JointRecord`] and effect-modification [`EstimateHeterogeneity`] outputs.
//! Every group other than the point is `None`/empty/false for an estimator that does not
//! produce it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::AssumptionSet;
use antecedent_stats::FirstStageDiagnostics;

use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::se::AnalyticSeKind;

/// Bootstrap bookkeeping reported beside a point estimate.
///
/// The all-default value means no bootstrap ran: no SE, no replicate counts, not
/// cancelled and not early-stopped.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BootstrapAccounting {
    /// Bootstrap standard error, when requested and enough replicates survived.
    pub se: Option<f64>,
    /// Successful replicates when bootstrap was requested.
    pub replicates_ok: Option<u32>,
    /// Soft-failed replicates when bootstrap was requested.
    pub replicates_failed: Option<u32>,
    /// The loop observed cooperative cancellation (partial replicates).
    pub cancelled: bool,
    /// Adaptive early stop (SE relative change).
    pub early_stopped: bool,
}

impl BootstrapAccounting {
    /// Accounting for a run that was cancelled or not, with no bootstrap SE or counts.
    #[must_use]
    pub fn cancelled(cancelled: bool) -> Self {
        Self { cancelled, ..Self::default() }
    }
}

/// Circular-block geometry an estimator used for its one-series bootstrap SE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockResampling {
    /// Circular-block length in lag-aligned rows.
    pub block_length: usize,
    /// Lag-aligned rows resampled.
    pub rows: usize,
    /// Bartlett kernel-bias factor of the target influence at `block_length`
    /// (`antecedent_estimate::kernel_bias_scale`), applied to the SE together
    /// with the fixed-b factor.
    pub kernel_bias: f64,
}

/// Screen / estimate split recorded on a batch result artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSelectionRecord {
    /// Caller-stable screen run id, or `"unrecorded"`.
    pub screen_id: Arc<str>,
    /// Selection rule wire name (`max_t`, `bh`, `by`, `unrecorded`).
    pub procedure: Arc<str>,
    /// Winning query index in the supplied family, or `None` when ranking is unavailable.
    pub winner_index: Option<usize>,
    /// Family size.
    pub family_size: usize,
    /// Row indexes used to screen candidates.
    pub screen_rows: Arc<[antecedent_data::GlobalId]>,
    /// Row indexes used to estimate the surfaced family.
    pub estimate_rows: Arc<[antecedent_data::GlobalId]>,
    /// Whether the two halves are disjoint.
    pub disjoint: bool,
}

/// Analytic standard error of the reported scalar and how it was formed.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct AnalyticSe {
    /// Analytic IID standard error (homoskedastic unless [`Self::kind`] says otherwise).
    ///
    /// NaN when the estimator publishes none (weak instrument, ridge / lasso).
    pub se: f64,
    /// Analytic SE formula behind [`Self::se`], recorded by the estimator that computed
    /// it. `None` when the SE is not an estimator-configured analytic kind
    /// (influence-function envelopes, bootstrap-only paths, posterior summaries).
    pub kind: Option<AnalyticSeKind>,
    /// Degrees of freedom of the Student-t reference the analytic SE is published with
    /// (`G - 1` for a cluster-DML result, `min(G_a, G_b) - 1` dyadic); `None` is the
    /// normal reference.
    pub reference_df: Option<f64>,
}

/// Circular-block provenance of a one-series bootstrap SE.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct BlockSeRecord {
    /// Circular-block geometry of the SE (the block length the estimator chose and the
    /// lag-aligned rows it resampled).
    pub resampling: Option<BlockResampling>,
    /// Circular-block SE family behind a one-series bootstrap SE, recorded by the path
    /// that ran the circular-block bootstrap.
    pub family: Option<crate::temporal_block::CircularBlockFamily>,
}

/// Every standard error a point estimate was published with.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct EstimateUncertainty {
    /// Analytic standard error.
    pub analytic: AnalyticSe,
    /// Bootstrap standard error and replicate accounting.
    pub bootstrap: BootstrapAccounting,
    /// Circular-block provenance of [`Self::bootstrap`].
    pub block: BlockSeRecord,
}

/// What the estimate was fitted under: carried premises and the analysed sample.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EstimateProvenance {
    /// Assumptions carried from identification.
    pub assumptions: AssumptionSet,
    /// Overlap policy recorded on the artifact.
    pub overlap: OverlapPolicy,
    /// Observations that actually informed the fit (complete-case / trimmed /
    /// lag-aligned rows). Distinct from the input snapshot row count.
    pub n_obs: Option<u64>,
    /// Estimated retained-memory cost of fitted scratch (bytes), when known.
    pub retained_memory_bytes: Option<u64>,
    /// How the columns the estimator read were prepared, when it admitted them through the
    /// preparation path. A physical fact about the run, so it is not part of any identity.
    pub preparation: Option<Arc<PreparationSummary>>,
}

/// Plain summary of the preparation receipt behind an estimate.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct PreparationSummary {
    /// Stable name of the preparation backend.
    pub backend: String,
    /// Stable name of the end-to-end route bound.
    pub route_bound: String,
    /// Rows the estimator read. `None` when it was handed a scan.
    pub rows_admitted: Option<u64>,
    /// Lower-case hex digest of the preparation plan.
    pub receipt_digest: String,
}

impl PreparationSummary {
    /// Summarize `receipt`.
    #[must_use]
    pub fn from_receipt(receipt: &antecedent_data::materialize::PreparationReceipt) -> Self {
        let route_bound = match receipt.route_bound() {
            antecedent_data::materialize::RouteBound::Proved { .. } => "proved",
            antecedent_data::materialize::RouteBound::RowsOnly { .. } => "rows_only",
            antecedent_data::materialize::RouteBound::NotAnEndToEndRoute => {
                "not_an_end_to_end_route"
            }
        };
        Self {
            backend: receipt.path().as_str().to_owned(),
            route_bound: route_bound.to_owned(),
            rows_admitted: receipt.proved_rows(),
            receipt_digest: receipt.digest().to_owned(),
        }
    }

    /// Plain summary from its parts (required outside this crate because the type is
    /// `#[non_exhaustive]`).
    #[must_use]
    pub fn from_parts(
        backend: String,
        route_bound: String,
        rows_admitted: Option<u64>,
        receipt_digest: String,
    ) -> Self {
        Self { backend, route_bound, rows_admitted, receipt_digest }
    }
}

/// Estimator-side diagnostics and sensitivity summaries recorded beside the point.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct EstimateDiagnostics {
    /// Propensity overlap diagnostics when computed.
    pub overlap_report: Option<OverlapReport>,
    /// Weak-instrument diagnostic (IV / 2SLS estimators only); `None` for every other
    /// estimator family. When `f_statistic < 10`, the analytic SE is NaN.
    pub first_stage: Option<FirstStageDiagnostics>,
    /// Point E-value for the reported effect when a named no-latent premise is in force.
    pub evalue: Option<f64>,
    /// Threshold [`Self::evalue`] was judged against, when it came from an
    /// E-value refuter that reported a pass/fail verdict. `None` when no refuter
    /// ran, or when the E-value is an attached premise diagnostic with no gate.
    pub evalue_threshold: Option<f64>,
    /// What a penalized propensity (ridge or lasso) or a GLM-to-penalized fallback recorded
    /// beside the point: the failed GLM fit and the fallback that replaced it, the support a
    /// lasso kept on each fold, and the refit bootstrap's per-replicate penalties. `None` for
    /// every other estimator and for an unpenalized AIPW that needed no fallback.
    pub penalized: Option<Box<crate::propensity::PenalizedReport>>,
    /// The treatment levels a propensity-based estimate contrasts, the rows at neither level
    /// that were excluded, and the target population that restriction implies. `None` for
    /// every estimator that does not recode a treatment column to declared levels.
    pub treatment_levels: Option<crate::propensity::DeclaredTreatmentLevels>,
    /// What declared sampling weights did to the analysis: the rows used, the zero-weight rows
    /// excluded, the Kish effective sample size and the unnormalized weight sum. `None` when
    /// no weights were declared, so unweighted results are unchanged.
    pub sampling_weights: Option<crate::sampling_weights::SamplingWeightRecord>,
}

/// Nuisance cross-fitting record.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct CrossfitRecord {
    /// Number of nuisance cross-fitting folds.
    pub folds: Option<usize>,
    /// Master seed used for nuisance cross-fitting.
    pub seed: Option<u64>,
    /// Actual fitted learner identities, in fold order (control, treated, propensity
    /// for AIPW; outcome folds then treatment folds for PLR; final CATE fit last).
    pub learner_provenance: Vec<antecedent_learn::LearnerProvenance>,
    /// Held-out outcome R².
    pub outcome_oof_r2: Option<f64>,
    /// Held-out treatment probability log loss.
    pub treatment_oof_logloss: Option<f64>,
}

/// Per-row score exports of the reported scalar.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct EstimateScores {
    /// Cross-fitted AIPW score table when the estimator exported one.
    pub table: Option<crate::scores::ScoreTable>,
    /// Simultaneous score-family bands and target-local support.
    pub inference: Option<crate::scores::ScoreInference>,
    /// Per-row influence for the reported scalar (shared-row joint IF / envelopes).
    pub influence: Option<Arc<[f64]>>,
}

/// Scientific target and identification retained by an executed batch-family contrast.
///
/// This binds the contrast to the actual source identification and observed world support.
/// It does not grant calibration or an additional interval/decision license.
#[derive(Clone, Debug)]
pub struct CompiledFamilyContrast {
    target: antecedent_core::ScientificTarget,
    identification: antecedent_identify::CompiledIdentificationProduct,
}

impl CompiledFamilyContrast {
    /// Retain a product already bound by the compiler to this exact target.
    ///
    /// # Errors
    /// Refuses an unbound, mismatched or incompletely supported scientific target.
    pub fn new(
        target: antecedent_core::ScientificTarget,
        identification: antecedent_identify::CompiledIdentificationProduct,
    ) -> Result<Self, antecedent_identify::ProductRefusal> {
        identification.validate_target(&target)?;
        Ok(Self { target, identification })
    }

    /// Full inference-independent contrast target, including its explicit reference world.
    #[must_use]
    pub fn target(&self) -> &antecedent_core::ScientificTarget {
        &self.target
    }

    /// Actual derivation and complete world/assembled support obligations.
    #[must_use = "the retained proof determines which contrast claims are permitted"]
    pub fn identification(&self) -> &antecedent_identify::CompiledIdentificationProduct {
        &self.identification
    }
}

/// Batch-family inference over a fixed family of cells or contrasts.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct FamilyInference {
    /// Checked scientific binding required before a joint-cell family contrast is exposed.
    /// Native retention only: older numerical artifact formats omit this binding and do not
    /// gain current scientific authority when decoded.
    pub compiled_contrast: Option<Arc<CompiledFamilyContrast>>,
    /// Simultaneous batch interval (lower, upper, level).
    ///
    /// Joint-cell batches keep this on cell **levels**. Contrast-family bands
    /// live on [`Self::contrast_interval`].
    pub simultaneous_interval: Option<(f64, f64, f64)>,
    /// BH and BY adjusted p-values for the fixed batch family.
    pub adjusted_p_values: Option<(f64, f64)>,
    /// Declared family contrast `(value, se)` when batch inference tested a
    /// contrast rather than [`EffectEstimate::ate`]. `None` for average-effect claims
    /// (`ate` is already the contrast) and when no cell contrast was declared.
    pub contrast: Option<(f64, f64)>,
    /// Simultaneous interval `(lower, upper, level)` for [`Self::contrast`].
    ///
    /// Formed from contrast-family max-t, not the cell-level covariance used
    /// by [`Self::simultaneous_interval`]. `None` when no contrast was declared
    /// or contrast max-t could not form.
    pub contrast_interval: Option<(f64, f64, f64)>,
    /// Bayesian-bootstrap posterior of [`Self::contrast`]'s functional, requested by the
    /// batch. Every member of the family shares each draw's row weights, so draw `d` of
    /// every member is one joint posterior realization of the whole family. Conditional on
    /// the frozen cross-fit nuisances; native retention only.
    pub contrast_posterior: Option<crate::score_posterior::ScorePosterior>,
    /// Candidate-selection screen recorded on a batch family (artifact payload).
    pub candidate_selection: Option<CandidateSelectionRecord>,
}

/// Effects and simultaneous intervals under declared canonical orientation scenarios.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ScenarioBand {
    /// Effects under declared canonical orientation scenarios, in estimand order.
    pub effects: Option<Arc<[f64]>>,
    /// Simultaneous intervals for canonical scenarios; not bounds on all completions.
    pub intervals: Option<Arc<[(f64, f64)]>>,
}

/// Joint influence-function outputs across arms, thresholds, cells or claims.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct JointRecord {
    /// Joint IF covariance across arms / thresholds / cells / claims.
    pub covariance: Option<crate::joint_if::JointCovariance>,
    /// Per-threshold `F_a(c) = 1 - P(Y(a) > c)` after monotone rearrangement.
    pub exceedance_cdf: Option<Arc<[f64]>>,
    /// `true` when `exceedance_cdf` / grid means were isotonically rearranged.
    /// Joint covariance and score-inference bands always describe the raw AIPW scores.
    pub monotone_rearranged: bool,
    /// Whether an additive joint path makes the interaction contrast structurally zero.
    pub interaction_structurally_zero: bool,
}

/// Effect-modification outputs of a heterogeneous-effect estimator.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct EstimateHeterogeneity {
    /// Per-row CATE when a heterogeneous-effect estimator produced one.
    pub cate: Option<Arc<[f64]>>,
    /// Retained portable CATE model, separate from the marginal estimate.
    pub fitted_effect: Option<Arc<crate::FittedEffect>>,
    /// Per-row CATE standard errors, when a licensed pointwise formula produced them.
    ///
    /// Linear DR-Learner finals use the HC0 sandwich of the orthogonal scores.
    /// Nonlinear or penalized CATE regressions, and forests (see
    /// [`Self::cate_leaf_dispersion`]), withhold this field rather than inventing an SE.
    pub cate_se: Option<Arc<[f64]>>,
    /// Per-row leaf-dispersion diagnostic of an honest forest: the square root of the
    /// mean over trees of the two-sample leaf variance of the row's leaf.
    ///
    /// This is NOT a standard error. Trees are half-samples of the same rows, so they
    /// are strongly positively correlated and the tree-averaged CATE does not have this
    /// variance; `cate ± 1.96·cate_leaf_dispersion` is not a confidence interval.
    /// Read it as "how noisy the typical honest leaf around this row is".
    pub cate_leaf_dispersion: Option<Arc<[f64]>>,
    /// Whether the fitted mechanisms on every treatment → outcome path admit no
    /// effect modification, so the reported per-unit effects are equal by
    /// construction of the selected mechanism families, not by measurement.
    pub unit_effects_homogeneous: bool,
}

/// Point estimate with uncertainty.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EffectEstimate {
    /// Reported scalar.
    ///
    /// For average-effect claims this is the ATE contrast. For `cell.aipw`
    /// joint cells it is the requested interventional **level**
    /// (`E[Y | do(A=a,…)]` or a single-threshold `F_a(c)`), not a contrast.
    /// Testing this field against zero is only valid when it is a contrast.
    /// Family p-values on joint cells use [`FamilyInference::contrast`].
    pub ate: f64,
    /// Analytic, bootstrap and block standard errors.
    pub uncertainty: EstimateUncertainty,
    /// Carried premises and the analysed sample.
    pub provenance: EstimateProvenance,
    /// Overlap, weak-instrument, E-value and penalization diagnostics.
    pub diagnostics: EstimateDiagnostics,
    /// Nuisance cross-fitting record.
    pub crossfit: CrossfitRecord,
    /// Exported score table, score bands and influence.
    pub scores: EstimateScores,
    /// Batch-family simultaneous inference.
    pub family: FamilyInference,
    /// Canonical orientation-scenario effects and intervals.
    pub scenarios: ScenarioBand,
    /// Joint covariance and exceedance outputs.
    pub joint: JointRecord,
    /// Per-row CATE outputs.
    pub heterogeneity: EstimateHeterogeneity,
}

impl EffectEstimate {
    /// Construct a point estimate without bootstrap accounting.
    #[must_use]
    pub fn new(
        ate: f64,
        se_analytic: f64,
        assumptions: AssumptionSet,
        overlap: OverlapPolicy,
    ) -> Self {
        Self::from_parts(
            ate,
            se_analytic,
            BootstrapAccounting::default(),
            assumptions,
            overlap,
            None,
            None,
        )
    }

    /// Full constructor (required outside this crate because the type is `#[non_exhaustive]`).
    #[must_use]
    pub fn from_parts(
        ate: f64,
        se_analytic: f64,
        bootstrap: BootstrapAccounting,
        assumptions: AssumptionSet,
        overlap: OverlapPolicy,
        overlap_report: Option<OverlapReport>,
        retained_memory_bytes: Option<u64>,
    ) -> Self {
        Self {
            ate,
            uncertainty: EstimateUncertainty {
                analytic: AnalyticSe { se: se_analytic, kind: None, reference_df: None },
                bootstrap,
                block: BlockSeRecord::default(),
            },
            provenance: EstimateProvenance {
                assumptions,
                overlap,
                n_obs: None,
                retained_memory_bytes,
                preparation: None,
            },
            diagnostics: EstimateDiagnostics { overlap_report, ..EstimateDiagnostics::default() },
            crossfit: CrossfitRecord::default(),
            scores: EstimateScores::default(),
            family: FamilyInference::default(),
            scenarios: ScenarioBand::default(),
            joint: JointRecord::default(),
            heterogeneity: EstimateHeterogeneity::default(),
        }
    }

    /// Analytic standard error of [`Self::ate`] (NaN when none is published).
    #[must_use]
    pub fn se_analytic(&self) -> f64 {
        self.uncertainty.analytic.se
    }

    /// Bootstrap standard error of [`Self::ate`], when requested and enough replicates survived.
    #[must_use]
    pub fn se_bootstrap(&self) -> Option<f64> {
        self.uncertainty.bootstrap.se
    }

    /// Record the analysis sample size that produced this interval.
    #[must_use]
    pub fn with_n_obs(mut self, n_obs: u64) -> Self {
        self.provenance.n_obs = Some(n_obs);
        self
    }

    /// Attach a weak-instrument diagnostic (IV / 2SLS estimators only).
    #[must_use]
    pub fn with_first_stage_diagnostics(
        mut self,
        diagnostics: Option<FirstStageDiagnostics>,
    ) -> Self {
        self.diagnostics.first_stage = diagnostics;
        self
    }

    /// Attach propensity overlap diagnostics (propensity-based estimators only).
    #[must_use]
    pub fn with_overlap_report(mut self, overlap_report: Option<OverlapReport>) -> Self {
        self.diagnostics.overlap_report = overlap_report;
        self
    }

    /// Record the declared treatment levels a propensity-based estimate contrasts.
    #[must_use]
    pub fn with_treatment_levels(
        mut self,
        levels: crate::propensity::DeclaredTreatmentLevels,
    ) -> Self {
        self.diagnostics.treatment_levels = Some(levels);
        self
    }

    /// Record what declared sampling weights did to this estimate.
    #[must_use]
    pub fn with_sampling_weights(
        mut self,
        record: Option<crate::sampling_weights::SamplingWeightRecord>,
    ) -> Self {
        self.diagnostics.sampling_weights = record;
        self
    }

    /// Attach a cross-fitted score table.
    #[must_use]
    pub fn with_score_table(mut self, table: Option<crate::scores::ScoreTable>) -> Self {
        self.scores.table = table;
        self
    }

    /// Attach joint IF covariance.
    #[must_use]
    pub fn with_joint_covariance(mut self, cov: Option<crate::joint_if::JointCovariance>) -> Self {
        self.joint.covariance = cov;
        self
    }

    /// Attach per-threshold CDF values `F_a(c)`.
    #[must_use]
    pub fn with_exceedance_cdf(mut self, cdf: Option<Arc<[f64]>>) -> Self {
        self.joint.exceedance_cdf = cdf;
        self
    }

    /// Record whether exceedance means were isotonically rearranged.
    ///
    /// When `true`, the simultaneous interval is cleared: those intervals
    /// must not mix rearranged means with raw covariance.
    #[must_use]
    pub fn with_monotone_rearranged(mut self, rearranged: bool) -> Self {
        self.joint.monotone_rearranged = rearranged;
        if rearranged {
            self.family.simultaneous_interval = None;
        }
        self
    }

    /// Attach a per-row influence sequence for the reported scalar.
    #[must_use]
    pub fn with_influence(mut self, influence: Option<Arc<[f64]>>) -> Self {
        self.scores.influence = influence;
        self
    }

    /// Record the analytic SE formula behind the analytic standard error.
    #[must_use]
    pub fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.uncertainty.analytic.kind = Some(se_kind);
        self
    }

    /// Attach a per-row CATE (`DrLearner` / `CausalForest`).
    #[must_use]
    pub fn with_cate(mut self, cate: Option<Arc<[f64]>>) -> Self {
        self.heterogeneity.cate = cate;
        self
    }

    /// Attach licensed per-row CATE standard errors, or withhold them.
    #[must_use]
    pub fn with_cate_se(mut self, cate_se: Option<Arc<[f64]>>) -> Self {
        self.heterogeneity.cate_se = cate_se;
        self
    }

    /// Attach the forest leaf-dispersion diagnostic (not a standard error).
    #[must_use]
    pub fn with_cate_leaf_dispersion(mut self, dispersion: Option<Arc<[f64]>>) -> Self {
        self.heterogeneity.cate_leaf_dispersion = dispersion;
        self
    }

    /// Record the circular-block SE family behind a one-series bootstrap SE.
    #[must_use]
    pub fn with_block_family(mut self, family: crate::temporal_block::CircularBlockFamily) -> Self {
        self.uncertainty.block.family = Some(family);
        self
    }

    /// Attach the estimated retained-memory cost of fitted scratch (bytes), when known.
    #[must_use]
    pub fn with_retained_memory_bytes(mut self, retained_memory_bytes: Option<u64>) -> Self {
        self.provenance.retained_memory_bytes = retained_memory_bytes;
        self
    }

    /// Record the preparation receipt the estimator's columns were admitted under.
    #[must_use]
    pub fn with_preparation(mut self, preparation: Option<Arc<PreparationSummary>>) -> Self {
        self.provenance.preparation = preparation;
        self
    }

    /// Attach bootstrap SE accounting (or clear when bootstrap was skipped).
    #[must_use]
    pub fn with_bootstrap(mut self, boot: Option<crate::util::BootstrapSeResult>) -> Self {
        self.uncertainty.bootstrap = match boot {
            None => BootstrapAccounting::default(),
            Some(b) => BootstrapAccounting {
                se: b.se,
                replicates_ok: Some(b.replicates_ok),
                replicates_failed: Some(b.replicates_failed),
                cancelled: b.cancelled,
                early_stopped: b.early_stopped,
            },
        };
        self
    }
}
