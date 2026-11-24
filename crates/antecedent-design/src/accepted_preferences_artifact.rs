//! Durable records of accepted preferences.
//!
//! [`AcceptedPreferencesArtifact`] stores accepted preferences: the declared value model, the
//! basis (the learned set with its relations and ranges, or the estimate with its measure), the
//! learning options and decider selection needed to relearn it, the version, the acknowledged
//! review items and the identity. It uses the container, wires and replay rules of the decision
//! record ([`crate::preference_artifact::PreferenceDecisionArtifact`]): a checksummed container
//! of kind `accepted_preferences_v1`, a CBOR body read back bit for bit, loading only under the
//! identity the consumer retained, and replay by relearning, comparing labels, counts and
//! identities exactly and every real within
//! [`crate::preference_artifact::REPLAY_RELATIVE_TOLERANCE`]. A declared basis replays without
//! records.

// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_io_core::convert::to_cbor;
use antecedent_io_core::distribution_artifact::DistributionCalibration;
use antecedent_stats::{ConditionalLogitOptions, ShapeSearchOptions};
use serde::{Deserialize, Serialize};

use crate::accepted_preferences::{
    AcceptedPreferences, AcceptedPreferencesError, LearningOptions, PendingReview, PreferenceBasis,
    select_estimate, select_learned, stated_in_scope,
};
use crate::choice_record::ChoiceRecordSet;
use crate::decision_artifact::decode_section;
use crate::preference_artifact_error::PreferenceArtifactError;
use crate::preference_choice_model::{
    ChoiceModelError, ChoiceModelEstimate, ChoiceModelOptions, CovarianceKind, ParameterEstimate,
    ParameterIntervalMethod,
};
use crate::preference_decision_artifact::{
    BODY_VERSION, LpWire, MeasureWire, ModelWire, SetWire, close, close_all, close_pair, encode,
    ensure, lp_from, lp_wire, measure_from, measure_wire, model_from, model_wire, observe,
    set_from, set_wire,
};
use crate::preference_model::PreferenceSet;
use crate::preference_set_learning::{
    BindingChoice, CellStatus, DEFAULT_MAX_SHAPE_BOXES, DEFAULT_MIN_SHAPE_WIDTH_FRACTION,
    LearnedPreferenceSet, LearnedRegion, ParameterRange, RangeBasis, RegionCell, SetLearningError,
    SetLearningOptions,
};

const ACCEPTED_KIND: &str = "accepted_preferences_v1";
const ACCEPTED_SECTION: &str = "accepted_preferences_body";

use PreferenceArtifactError as E;

/// A refused relearning, with the learner's cancellation reported as the replay's.
fn relearn_error(error: AcceptedPreferencesError) -> PreferenceArtifactError {
    match error {
        AcceptedPreferencesError::SetLearning(SetLearningError::Cancelled)
        | AcceptedPreferencesError::ChoiceModel(ChoiceModelError::Cancelled) => E::Cancelled,
        other => E::Accepted(other),
    }
}

/// `enclosure` is written only when set (an outer range under a value with shape parameters).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RangeWire {
    parameter_id: String,
    declared: (f64, f64),
    learned: (f64, f64),
    varied_in_records: bool,
    narrowed: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    enclosure: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CellWire {
    shape_lower: Vec<f64>,
    shape_upper: Vec<f64>,
    inside: bool,
    linear_outer: Vec<(f64, f64)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LearnedRegionWire {
    shape_parameters: Vec<usize>,
    linear_parameters: Vec<usize>,
    cells: Vec<CellWire>,
    boxes_used: usize,
    min_width_fraction: f64,
    budget_exhausted: bool,
}

fn region_wire(region: &LearnedRegion) -> LearnedRegionWire {
    LearnedRegionWire {
        shape_parameters: region.shape_parameters.clone(),
        linear_parameters: region.linear_parameters.clone(),
        cells: region
            .cells
            .iter()
            .map(|c| CellWire {
                shape_lower: c.shape_lower.clone(),
                shape_upper: c.shape_upper.clone(),
                inside: c.status == CellStatus::Inside,
                linear_outer: c.linear_outer.clone(),
            })
            .collect(),
        boxes_used: region.boxes_used,
        min_width_fraction: region.min_width_fraction,
        budget_exhausted: region.budget_exhausted,
    }
}

fn region_from(wire: LearnedRegionWire) -> LearnedRegion {
    LearnedRegion {
        shape_parameters: wire.shape_parameters,
        linear_parameters: wire.linear_parameters,
        cells: wire
            .cells
            .into_iter()
            .map(|c| RegionCell {
                shape_lower: c.shape_lower,
                shape_upper: c.shape_upper,
                status: if c.inside { CellStatus::Inside } else { CellStatus::Boundary },
                linear_outer: c.linear_outer,
            })
            .collect(),
        boxes_used: wire.boxes_used,
        min_width_fraction: wire.min_width_fraction,
        budget_exhausted: wire.budget_exhausted,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingWire {
    record_id: String,
    rejected_option: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LearnedWire {
    model_identity: String,
    records_digest: String,
    decider: Option<String>,
    set: SetWire,
    ranges: Vec<RangeWire>,
    binding: Vec<BindingWire>,
    tolerance: f64,
    n_records: usize,
    assumptions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    region: Option<LearnedRegionWire>,
}

fn learned_wire(learned: &LearnedPreferenceSet) -> LearnedWire {
    LearnedWire {
        model_identity: learned.model_identity.clone(),
        records_digest: learned.records_digest.clone(),
        decider: learned.decider.clone(),
        set: set_wire(&learned.set),
        ranges: learned
            .ranges
            .iter()
            .map(|r| RangeWire {
                parameter_id: r.parameter_id.clone(),
                declared: r.declared,
                learned: r.learned,
                varied_in_records: r.varied_in_records,
                narrowed: r.narrowed,
                enclosure: r.basis == RangeBasis::Enclosure,
            })
            .collect(),
        binding: learned
            .binding
            .iter()
            .map(|b| BindingWire {
                record_id: b.record_id.clone(),
                rejected_option: b.rejected_option.clone(),
            })
            .collect(),
        tolerance: learned.tolerance,
        n_records: learned.n_records,
        assumptions: learned.assumptions.clone(),
        region: learned.region.as_ref().map(region_wire),
    }
}

fn learned_from(wire: LearnedWire) -> LearnedPreferenceSet {
    LearnedPreferenceSet {
        model_identity: wire.model_identity,
        records_digest: wire.records_digest,
        decider: wire.decider,
        set: set_from(wire.set),
        ranges: wire
            .ranges
            .into_iter()
            .map(|r| ParameterRange {
                parameter_id: r.parameter_id,
                declared: r.declared,
                learned: r.learned,
                varied_in_records: r.varied_in_records,
                narrowed: r.narrowed,
                basis: if r.enclosure { RangeBasis::Enclosure } else { RangeBasis::LinearProgram },
            })
            .collect(),
        binding: wire
            .binding
            .into_iter()
            .map(|b| BindingChoice { record_id: b.record_id, rejected_option: b.rejected_option })
            .collect(),
        tolerance: wire.tolerance,
        n_records: wire.n_records,
        assumptions: wire.assumptions,
        region: wire.region.map(region_from),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EstimateParameterWire {
    parameter_id: String,
    estimate: f64,
    standard_error: f64,
    interval: (f64, f64),
    declared: (f64, f64),
    inside_declared: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    profile_interval: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CovarianceWire {
    Model,
    ClusteredByDecider,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EstimateWire {
    identity: String,
    model_identity: String,
    records_digest: String,
    decider: Option<String>,
    consistency: f64,
    consistency_standard_error: f64,
    parameters: Vec<EstimateParameterWire>,
    not_learnable: Vec<String>,
    log_likelihood: f64,
    null_log_likelihood: f64,
    n_records: usize,
    n_deciders: usize,
    covariance_kind: CovarianceWire,
    coefficient_covariance: Vec<f64>,
    level: f64,
    interval_calibration: DistributionCalibration,
    measure: MeasureWire,
    share_outside_declared: f64,
    assumptions: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    shape_parameters: Vec<String>,
}

fn estimate_wire(estimate: &ChoiceModelEstimate) -> EstimateWire {
    EstimateWire {
        identity: estimate.identity.clone(),
        model_identity: estimate.model_identity.clone(),
        records_digest: estimate.records_digest.clone(),
        decider: estimate.decider.clone(),
        consistency: estimate.consistency,
        consistency_standard_error: estimate.consistency_standard_error,
        parameters: estimate
            .parameters
            .iter()
            .map(|p| EstimateParameterWire {
                parameter_id: p.parameter_id.clone(),
                estimate: p.estimate,
                standard_error: p.standard_error,
                interval: p.interval,
                declared: p.declared,
                inside_declared: p.inside_declared,
                profile_interval: p.interval_method == ParameterIntervalMethod::ProfileLikelihood,
            })
            .collect(),
        not_learnable: estimate.not_learnable.clone(),
        log_likelihood: estimate.log_likelihood,
        null_log_likelihood: estimate.null_log_likelihood,
        n_records: estimate.n_records,
        n_deciders: estimate.n_deciders,
        covariance_kind: match estimate.covariance_kind {
            CovarianceKind::Model => CovarianceWire::Model,
            CovarianceKind::ClusteredByDecider => CovarianceWire::ClusteredByDecider,
        },
        coefficient_covariance: estimate.coefficient_covariance.clone(),
        level: estimate.level,
        interval_calibration: estimate.interval_calibration,
        measure: measure_wire(&estimate.measure),
        share_outside_declared: estimate.share_outside_declared,
        assumptions: estimate.assumptions.clone(),
        shape_parameters: estimate.shape_parameters.clone(),
    }
}

fn estimate_from(wire: EstimateWire) -> ChoiceModelEstimate {
    ChoiceModelEstimate {
        identity: wire.identity,
        model_identity: wire.model_identity,
        records_digest: wire.records_digest,
        decider: wire.decider,
        consistency: wire.consistency,
        consistency_standard_error: wire.consistency_standard_error,
        parameters: wire
            .parameters
            .into_iter()
            .map(|p| ParameterEstimate {
                parameter_id: p.parameter_id,
                estimate: p.estimate,
                standard_error: p.standard_error,
                interval: p.interval,
                declared: p.declared,
                inside_declared: p.inside_declared,
                interval_method: if p.profile_interval {
                    ParameterIntervalMethod::ProfileLikelihood
                } else {
                    ParameterIntervalMethod::Wald
                },
            })
            .collect(),
        not_learnable: wire.not_learnable,
        log_likelihood: wire.log_likelihood,
        null_log_likelihood: wire.null_log_likelihood,
        n_records: wire.n_records,
        n_deciders: wire.n_deciders,
        covariance_kind: match wire.covariance_kind {
            CovarianceWire::Model => CovarianceKind::Model,
            CovarianceWire::ClusteredByDecider => CovarianceKind::ClusteredByDecider,
        },
        coefficient_covariance: wire.coefficient_covariance,
        level: wire.level,
        interval_calibration: wire.interval_calibration,
        measure: measure_from(wire.measure),
        share_outside_declared: wire.share_outside_declared,
        assumptions: wire.assumptions,
        shape_parameters: wire.shape_parameters,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BasisWire {
    Declared,
    CompatibleSet(Box<LearnedWire>),
    ChoiceEstimate(Box<EstimateWire>),
}

fn basis_wire(basis: &PreferenceBasis) -> BasisWire {
    match basis {
        PreferenceBasis::Declared => BasisWire::Declared,
        PreferenceBasis::CompatibleSet(learned) => {
            BasisWire::CompatibleSet(Box::new(learned_wire(learned)))
        }
        PreferenceBasis::ChoiceEstimate(estimate) => {
            BasisWire::ChoiceEstimate(Box::new(estimate_wire(estimate)))
        }
    }
}

fn basis_from(wire: BasisWire) -> PreferenceBasis {
    match wire {
        BasisWire::Declared => PreferenceBasis::Declared,
        BasisWire::CompatibleSet(learned) => {
            PreferenceBasis::CompatibleSet(Box::new(learned_from(*learned)))
        }
        BasisWire::ChoiceEstimate(estimate) => {
            PreferenceBasis::ChoiceEstimate(Box::new(estimate_from(*estimate)))
        }
    }
}

/// Written only when the shape subdivision options differ from their defaults.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegionOptionsWire {
    max_boxes: usize,
    min_width_fraction: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SetOptionsWire {
    tolerance: f64,
    conflict_search_size: usize,
    lp: LpWire,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    region: Option<RegionOptionsWire>,
}

/// Written only when the shape search options differ from their defaults.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShapeSearchWire {
    grid_points: usize,
    max_dimensions: usize,
    profile_points: usize,
    max_refinements: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FitWire {
    max_iterations: usize,
    gradient_tolerance: f64,
    separation_bound: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChoiceOptionsWire {
    level: f64,
    n_draws: usize,
    seed: u64,
    fit: FitWire,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shape_search: Option<ShapeSearchWire>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LearningWire {
    CompatibleSet(SetOptionsWire),
    ChoiceEstimate(ChoiceOptionsWire),
}

fn learning_wire(learning: &LearningOptions) -> LearningWire {
    match learning {
        LearningOptions::CompatibleSet(o) => LearningWire::CompatibleSet(SetOptionsWire {
            tolerance: o.tolerance,
            conflict_search_size: o.conflict_search_size,
            lp: lp_wire(o.lp),
            region: (o.max_boxes != DEFAULT_MAX_SHAPE_BOXES
                || o.min_width_fraction.total_cmp(&DEFAULT_MIN_SHAPE_WIDTH_FRACTION).is_ne())
            .then_some(RegionOptionsWire {
                max_boxes: o.max_boxes,
                min_width_fraction: o.min_width_fraction,
            }),
        }),
        LearningOptions::ChoiceEstimate(o) => LearningWire::ChoiceEstimate(ChoiceOptionsWire {
            level: o.level,
            n_draws: o.n_draws,
            seed: o.seed,
            fit: FitWire {
                max_iterations: o.fit.max_iterations,
                gradient_tolerance: o.fit.gradient_tolerance,
                separation_bound: o.fit.separation_bound,
            },
            shape_search: (o.shape_search != ShapeSearchOptions::default()).then_some(
                ShapeSearchWire {
                    grid_points: o.shape_search.grid_points,
                    max_dimensions: o.shape_search.max_dimensions,
                    profile_points: o.shape_search.profile_points,
                    max_refinements: o.shape_search.max_refinements,
                },
            ),
        }),
    }
}

fn learning_from(wire: LearningWire) -> LearningOptions {
    match wire {
        LearningWire::CompatibleSet(o) => LearningOptions::CompatibleSet(SetLearningOptions {
            tolerance: o.tolerance,
            conflict_search_size: o.conflict_search_size,
            lp: lp_from(o.lp),
            max_boxes: o.region.map_or(DEFAULT_MAX_SHAPE_BOXES, |r| r.max_boxes),
            min_width_fraction: o
                .region
                .map_or(DEFAULT_MIN_SHAPE_WIDTH_FRACTION, |r| r.min_width_fraction),
        }),
        LearningWire::ChoiceEstimate(o) => LearningOptions::ChoiceEstimate(ChoiceModelOptions {
            level: o.level,
            n_draws: o.n_draws,
            seed: o.seed,
            fit: ConditionalLogitOptions {
                max_iterations: o.fit.max_iterations,
                gradient_tolerance: o.fit.gradient_tolerance,
                separation_bound: o.fit.separation_bound,
            },
            shape_search: o.shape_search.map_or_else(ShapeSearchOptions::default, |w| {
                ShapeSearchOptions {
                    grid_points: w.grid_points,
                    max_dimensions: w.max_dimensions,
                    profile_points: w.profile_points,
                    max_refinements: w.max_refinements,
                }
            }),
        }),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PendingWire {
    ParameterNotVaried { parameter_id: String },
    ParameterNotNarrowed { parameter_id: String },
    ToleranceApplied { tolerance: f64 },
    ParameterNotLearnable { parameter_id: String },
    EstimateOutsideDeclared { parameter_id: String },
    DrawsOutsideDeclared { share: f64 },
    StatedComparisonsUsed { count: usize },
}

fn pending_wire(item: &PendingReview) -> PendingWire {
    match item {
        PendingReview::ParameterNotVaried { parameter_id } => {
            PendingWire::ParameterNotVaried { parameter_id: parameter_id.clone() }
        }
        PendingReview::ParameterNotNarrowed { parameter_id } => {
            PendingWire::ParameterNotNarrowed { parameter_id: parameter_id.clone() }
        }
        PendingReview::ToleranceApplied { tolerance } => {
            PendingWire::ToleranceApplied { tolerance: *tolerance }
        }
        PendingReview::ParameterNotLearnable { parameter_id } => {
            PendingWire::ParameterNotLearnable { parameter_id: parameter_id.clone() }
        }
        PendingReview::EstimateOutsideDeclared { parameter_id } => {
            PendingWire::EstimateOutsideDeclared { parameter_id: parameter_id.clone() }
        }
        PendingReview::DrawsOutsideDeclared { share } => {
            PendingWire::DrawsOutsideDeclared { share: *share }
        }
        PendingReview::StatedComparisonsUsed { count } => {
            PendingWire::StatedComparisonsUsed { count: *count }
        }
    }
}

fn pending_from(wire: PendingWire) -> PendingReview {
    match wire {
        PendingWire::ParameterNotVaried { parameter_id } => {
            PendingReview::ParameterNotVaried { parameter_id }
        }
        PendingWire::ParameterNotNarrowed { parameter_id } => {
            PendingReview::ParameterNotNarrowed { parameter_id }
        }
        PendingWire::ToleranceApplied { tolerance } => {
            PendingReview::ToleranceApplied { tolerance }
        }
        PendingWire::ParameterNotLearnable { parameter_id } => {
            PendingReview::ParameterNotLearnable { parameter_id }
        }
        PendingWire::EstimateOutsideDeclared { parameter_id } => {
            PendingReview::EstimateOutsideDeclared { parameter_id }
        }
        PendingWire::DrawsOutsideDeclared { share } => {
            PendingReview::DrawsOutsideDeclared { share }
        }
        PendingWire::StatedComparisonsUsed { count } => {
            PendingReview::StatedComparisonsUsed { count }
        }
    }
}

/// Canonical identity of accepted preferences: the declared model identity, then the CBOR wire
/// of the basis and of the learning options, the number of stated comparisons, the version and
/// the CBOR wire of the acknowledged items.
pub(crate) fn accepted_identity(
    declared_identity: &str,
    basis: &PreferenceBasis,
    learning: Option<&LearningOptions>,
    stated_comparisons: usize,
    version: u32,
    acknowledged: &[PendingReview],
) -> String {
    let acknowledged: Vec<PendingWire> = acknowledged.iter().map(pending_wire).collect();
    let mut out = CanonicalWriter::new("antecedent.accepted_preferences.v1");
    out.text(declared_identity)
        .bytes(&to_cbor(&basis_wire(basis)).unwrap_or_default())
        .bytes(&to_cbor(&learning.map(learning_wire)).unwrap_or_default())
        .word(u64::try_from(stated_comparisons).unwrap_or(u64::MAX))
        .word(u64::from(version))
        .bytes(&to_cbor(&acknowledged).unwrap_or_default());
    out.finish_hex()
}

// ---------------------------------------------------------------------------------------------
// Accepted preferences artifact
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptedBody {
    version: u16,
    identity: String,
    declared: ModelWire,
    basis: BasisWire,
    learning: Option<LearningWire>,
    stated_comparisons: usize,
    accepted_version: u32,
    acknowledged: Vec<PendingWire>,
}

fn accepted_body(accepted: &AcceptedPreferences) -> AcceptedBody {
    AcceptedBody {
        version: BODY_VERSION,
        identity: accepted.identity().to_owned(),
        declared: model_wire(accepted.declared_model()),
        basis: basis_wire(accepted.basis()),
        learning: accepted.learning_options().as_ref().map(learning_wire),
        stated_comparisons: accepted.stated_comparisons(),
        accepted_version: accepted.version(),
        acknowledged: accepted.acknowledged().iter().map(pending_wire).collect(),
    }
}

fn accepted_from_body(body: AcceptedBody) -> Result<AcceptedPreferences, PreferenceArtifactError> {
    let acknowledged: Vec<PendingReview> =
        body.acknowledged.into_iter().map(pending_from).collect();
    let accepted = AcceptedPreferences::from_parts(
        model_from(body.declared),
        basis_from(body.basis),
        body.learning.map(learning_from),
        body.stated_comparisons,
        body.accepted_version,
        &acknowledged,
    )
    .map_err(E::Accepted)?;
    if accepted.identity() != body.identity {
        return Err(E::StoredIdentityMismatch);
    }
    Ok(accepted)
}

/// Accepted preferences in a durable, identity-bound record.
#[derive(Clone, Debug, PartialEq)]
pub struct AcceptedPreferencesArtifact {
    accepted: AcceptedPreferences,
}

impl AcceptedPreferencesArtifact {
    /// Hold accepted preferences for storage.
    #[must_use]
    pub fn new(accepted: AcceptedPreferences) -> Self {
        Self { accepted }
    }

    /// The accepted preferences.
    #[must_use]
    pub fn accepted(&self) -> &AcceptedPreferences {
        &self.accepted
    }

    /// Identity of the accepted preferences; the identity a consumer retains.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.accepted.identity()
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An encoding failure, an oversized payload or a blank artifact id.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, PreferenceArtifactError> {
        encode(
            ACCEPTED_KIND,
            ACCEPTED_SECTION,
            artifact_id,
            "accepted_preferences",
            &accepted_body(&self.accepted),
        )
    }

    /// Load under the consumer's independently retained identity.
    ///
    /// # Errors
    /// A corrupt container, another kind or an unsupported version; content whose parts do not
    /// belong together; an identity recomputed from the content that differs from the stored
    /// one (a resealed edit) or from `expected_identity`.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: &str,
    ) -> Result<Self, PreferenceArtifactError> {
        let body: AcceptedBody =
            decode_section(bytes, ACCEPTED_KIND, ACCEPTED_SECTION).map_err(E::Container)?;
        let accepted = accepted_from_body(body)?;
        if accepted.identity() != expected_identity {
            return Err(E::IdentityExpected);
        }
        Ok(Self { accepted })
    }

    /// Relearn from `records` with the stored options and decider selection and require the
    /// stored basis to be reproduced: the records digest, the number of stated comparisons in
    /// the selected scope and every label, count and identity exactly, and every real within
    /// [`crate::preference_artifact::REPLAY_RELATIVE_TOLERANCE`]. A declared basis replays without
    /// reading the records.
    ///
    /// # Errors
    /// Records other than the ones the basis was learned from; a refused relearning; any
    /// differing field.
    pub fn replay(&self, records: &ChoiceRecordSet) -> Result<(), PreferenceArtifactError> {
        self.replay_in(records, None)
    }

    /// [`Self::replay`] under an execution context: the relearning reads the context's
    /// cancellation token as the learner or the estimator does.
    ///
    /// # Errors
    /// As [`Self::replay`]; [`PreferenceArtifactError::Cancelled`] once the token is set.
    pub fn replay_with_context(
        &self,
        records: &ChoiceRecordSet,
        ctx: &ExecutionContext,
    ) -> Result<(), PreferenceArtifactError> {
        self.replay_in(records, Some(ctx))
    }

    fn replay_in(
        &self,
        records: &ChoiceRecordSet,
        ctx: Option<&ExecutionContext>,
    ) -> Result<(), PreferenceArtifactError> {
        observe(ctx)?;
        let model = self.accepted.declared_model();
        let decider = self.accepted.decider();
        let stated = || {
            ensure(
                stated_in_scope(records, decider) == self.accepted.stated_comparisons(),
                "stated_comparisons",
            )
        };
        match (self.accepted.basis(), self.accepted.learning_options()) {
            (PreferenceBasis::Declared, _) => Ok(()),
            (PreferenceBasis::CompatibleSet(stored), Some(LearningOptions::CompatibleSet(o))) => {
                let fresh =
                    select_learned(model, records, decider, o, ctx).map_err(relearn_error)?;
                compare_learned(stored, &fresh)?;
                stated()
            }
            (PreferenceBasis::ChoiceEstimate(stored), Some(LearningOptions::ChoiceEstimate(o))) => {
                let fresh =
                    select_estimate(model, records, decider, o, ctx).map_err(relearn_error)?;
                compare_estimate(stored, &fresh)?;
                stated()
            }
            _ => Err(E::Accepted(AcceptedPreferencesError::OptionsMismatch { option: "basis" })),
        }
    }
}

fn compare_set(
    stored: &PreferenceSet,
    fresh: &PreferenceSet,
) -> Result<(), PreferenceArtifactError> {
    ensure(stored.parameters == fresh.parameters, "parameters")?;
    ensure(stored.relations.len() == fresh.relations.len(), "relations")?;
    for (s, f) in stored.relations.iter().zip(&fresh.relations) {
        let same_terms = s.terms.len() == f.terms.len()
            && s.terms.iter().zip(&f.terms).all(|(x, y)| x.0 == y.0 && close(x.1, y.1));
        ensure(s.id == f.id && s.sense == f.sense && s.origin == f.origin, "relations")?;
        ensure(same_terms && close(s.bound, f.bound), "relation_terms")?;
    }
    ensure(stored.value_relations.len() == fresh.value_relations.len(), "value_relations")?;
    for (s, f) in stored.value_relations.iter().zip(&fresh.value_relations) {
        ensure(s.id == f.id && s.origin == f.origin, "value_relations")?;
        let same_rows = close_all(&s.chosen.rows, &f.chosen.rows)
            && close_all(&s.chosen.weights, &f.chosen.weights)
            && close_all(&s.rejected.rows, &f.rejected.rows)
            && close_all(&s.rejected.weights, &f.rejected.weights)
            && close(s.tolerance, f.tolerance);
        ensure(same_rows, "value_relation_rows")?;
    }
    Ok(())
}

fn compare_region(
    stored: Option<&LearnedRegion>,
    fresh: Option<&LearnedRegion>,
) -> Result<(), PreferenceArtifactError> {
    match (stored, fresh) {
        (None, None) => Ok(()),
        (Some(s), Some(f)) => {
            ensure(
                s.shape_parameters == f.shape_parameters
                    && s.linear_parameters == f.linear_parameters
                    && s.boxes_used == f.boxes_used
                    && s.budget_exhausted == f.budget_exhausted
                    && s.min_width_fraction.total_cmp(&f.min_width_fraction).is_eq(),
                "region",
            )?;
            ensure(s.cells.len() == f.cells.len(), "region_cells")?;
            for (a, b) in s.cells.iter().zip(&f.cells) {
                let same = a.status == b.status
                    && close_all(&a.shape_lower, &b.shape_lower)
                    && close_all(&a.shape_upper, &b.shape_upper)
                    && a.linear_outer.len() == b.linear_outer.len()
                    && a.linear_outer.iter().zip(&b.linear_outer).all(|(x, y)| close_pair(*x, *y));
                ensure(same, "region_cells")?;
            }
            Ok(())
        }
        _ => Err(E::ReplayMismatch { field: "region" }),
    }
}

fn compare_learned(
    stored: &LearnedPreferenceSet,
    fresh: &LearnedPreferenceSet,
) -> Result<(), PreferenceArtifactError> {
    if stored.records_digest != fresh.records_digest {
        return Err(E::ReplayRecordsDiffer);
    }
    ensure(stored.model_identity == fresh.model_identity, "model_identity")?;
    ensure(stored.decider == fresh.decider, "decider")?;
    ensure(stored.tolerance.total_cmp(&fresh.tolerance).is_eq(), "tolerance")?;
    ensure(stored.n_records == fresh.n_records, "n_records")?;
    compare_set(&stored.set, &fresh.set)?;
    ensure(stored.ranges.len() == fresh.ranges.len(), "ranges")?;
    for (s, f) in stored.ranges.iter().zip(&fresh.ranges) {
        ensure(s.parameter_id == f.parameter_id, "ranges")?;
        ensure(
            s.varied_in_records == f.varied_in_records
                && s.narrowed == f.narrowed
                && s.basis == f.basis,
            "range_labels",
        )?;
        ensure(close_pair(s.declared, f.declared) && close_pair(s.learned, f.learned), "ranges")?;
    }
    ensure(stored.binding == fresh.binding, "binding")?;
    compare_region(stored.region.as_ref(), fresh.region.as_ref())?;
    ensure(stored.assumptions == fresh.assumptions, "assumptions")
}

fn compare_estimate(
    stored: &ChoiceModelEstimate,
    fresh: &ChoiceModelEstimate,
) -> Result<(), PreferenceArtifactError> {
    if stored.records_digest != fresh.records_digest {
        return Err(E::ReplayRecordsDiffer);
    }
    ensure(stored.identity == fresh.identity, "estimate_identity")?;
    ensure(stored.model_identity == fresh.model_identity, "model_identity")?;
    ensure(stored.decider == fresh.decider, "decider")?;
    ensure(close(stored.consistency, fresh.consistency), "consistency")?;
    ensure(
        close(stored.consistency_standard_error, fresh.consistency_standard_error),
        "consistency_standard_error",
    )?;
    ensure(stored.parameters.len() == fresh.parameters.len(), "parameters")?;
    for (s, f) in stored.parameters.iter().zip(&fresh.parameters) {
        ensure(s.parameter_id == f.parameter_id, "parameters")?;
        ensure(s.inside_declared == f.inside_declared, "inside_declared")?;
        ensure(s.interval_method == f.interval_method, "interval_method")?;
        let numbers = close(s.estimate, f.estimate)
            && close(s.standard_error, f.standard_error)
            && close_pair(s.interval, f.interval)
            && close_pair(s.declared, f.declared);
        ensure(numbers, "parameter_estimates")?;
    }
    ensure(stored.not_learnable == fresh.not_learnable, "not_learnable")?;
    ensure(close(stored.log_likelihood, fresh.log_likelihood), "log_likelihood")?;
    ensure(close(stored.null_log_likelihood, fresh.null_log_likelihood), "null_log_likelihood")?;
    ensure(stored.n_records == fresh.n_records && stored.n_deciders == fresh.n_deciders, "counts")?;
    ensure(stored.covariance_kind == fresh.covariance_kind, "covariance_kind")?;
    ensure(
        close_all(&stored.coefficient_covariance, &fresh.coefficient_covariance),
        "coefficient_covariance",
    )?;
    ensure(stored.level.total_cmp(&fresh.level).is_eq(), "level")?;
    ensure(stored.interval_calibration == fresh.interval_calibration, "interval_calibration")?;
    let (kept, redrawn) = (&stored.measure, &fresh.measure);
    ensure(
        kept.points.len() == redrawn.points.len()
            && kept.points.iter().zip(&redrawn.points).all(|(x, y)| close_all(x, y)),
        "measure_points",
    )?;
    ensure(
        kept.weights == redrawn.weights
            && kept.labels == redrawn.labels
            && kept.origin == redrawn.origin
            && kept.unresolved == redrawn.unresolved,
        "measure",
    )?;
    ensure(
        close(stored.share_outside_declared, fresh.share_outside_declared),
        "share_outside_declared",
    )?;
    ensure(stored.shape_parameters == fresh.shape_parameters, "shape_parameters")?;
    ensure(stored.assumptions == fresh.assumptions, "assumptions")
}

#[cfg(test)]
mod tests {
    use antecedent_io_core::error::IoError;

    use super::*;
    use crate::preference_model::{PreferenceExpr, ValueModel};

    #[test]
    fn a_resealed_accepted_body_is_refused() {
        use crate::preference_model::{Attribute as A, PreferenceParameter as Q};
        let model = ValueModel {
            attributes: vec![A { id: "benefit".into(), units: "events".into() }],
            preferences: PreferenceSet::new(
                vec![Q {
                    id: "w".into(),
                    units: "utility/event".into(),
                    lower: 0.0,
                    upper: 1.0,
                    assumption_id: "w-judgment".into(),
                }],
                vec![],
            ),
            value: PreferenceExpr::product(
                PreferenceExpr::Parameter(0),
                PreferenceExpr::Attribute(0),
            ),
            utility_units: "utility".into(),
        };
        let accepted = AcceptedPreferences::declared(model).unwrap();
        let mut body = accepted_body(&accepted);
        body.accepted_version = 7;
        let bytes =
            encode(ACCEPTED_KIND, ACCEPTED_SECTION, "accepted", "accepted_preferences", &body)
                .unwrap();
        let error =
            AcceptedPreferencesArtifact::from_bytes(&bytes, accepted.identity()).unwrap_err();
        assert_eq!(error, E::StoredIdentityMismatch);
        assert_eq!(
            error.detail(),
            antecedent_core::reason_details::preference_artifact::IDENTITY_RESEALED
        );
        let io: IoError = error.into();
        assert!(matches!(io, IoError::Refused { .. }));
    }
}
