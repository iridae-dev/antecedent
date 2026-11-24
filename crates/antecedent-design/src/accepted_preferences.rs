//! Reviewed, versioned preferences reused across decisions.
//!
//! [`AcceptedPreferences`] holds a value model together with the basis its admissible
//! preferences rest on: the declared ranges and relations alone ([`PreferenceBasis::Declared`]),
//! the set compatible with recorded choices ([`PreferenceBasis::CompatibleSet`]), or an estimate
//! from choices read as noisy ([`PreferenceBasis::ChoiceEstimate`]). Preferences are declared or
//! learned once, reviewed, and then reused by every decision built from the handle; nothing
//! relearns them implicitly. Every change returns a new handle with the version increased by
//! one: acknowledging review items ([`AcceptedPreferences::acknowledge`]) and explicit
//! relearning from records ([`AcceptedPreferences::relearn_compatible_set`],
//! [`AcceptedPreferences::relearn_estimate`]), which clears the acknowledgements.
//!
//! [`AcceptedPreferences::pending`] lists what a reviewer should look at before relying on the
//! preferences, derived from the basis each time and less what was already acknowledged. For a
//! compatible set: each parameter no record varied (it keeps its declared range), each
//! parameter records varied whose range did not narrow, then a positive tolerance. For an
//! estimate: each parameter that is not learnable or whose point estimate lies outside its
//! declared range, then a positive share of measure draws dropped outside the declared set.
//! For either learned basis, when the records it was learned from include stated comparisons
//! (what someone said they would choose, not what they chose), their count comes last.
//! Parameter items come in model parameter order and the set-wide items after them.
//!
//! A learned basis is accepted only when it belongs to the declared model: the learned set keeps
//! the declared parameters with their ranges, ids, units and assumptions and every declared
//! relation, every relation it adds carries a choice origin, an estimate's parameter entries
//! carry the declared ranges, and the records the caller supplies are the records the basis was
//! learned from (the same digest). A learned object the caller supplies
//! ([`AcceptedPreferences::from_compatible_set`], [`AcceptedPreferences::from_estimate`]) is
//! also learned again from those records and options and must be reproduced, so content edited
//! after learning is refused. The number of stated comparisons is counted from those records,
//! never taken from the caller.
//!
//! [`decide_with_accepted`] builds the contract over the accepted model and passes the accepted
//! measure (an estimate's) to the route-choosing evaluation.
//!
//! The handle retains everything needed to replay the learning (the declared model, the learning
//! options and the decision-maker selected), so a stored handle can be relearned from the same
//! records and compared. A learned set is an assumption range, never a probability; only an
//! estimate carries a measure over the set, and that measure is the normal approximation to the
//! sampling distribution of the estimate, not a posterior.

// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::ExternalRefusal;
use antecedent_core::ExecutionContext;
use antecedent_io_core::distribution_artifact::DistributionArtifact;

use crate::choice_record::{ChoiceRecordSet, Pooling};
use crate::decision_contract::{DecisionCriterion, HardConstraint, StructuralPolicy};
use crate::preference_choice_model::{
    ChoiceModelError, ChoiceModelEstimate, ChoiceModelOptions, EstimateOutcome, estimate_all,
};
use crate::preference_contract::{PreferenceAction, PreferenceContract, PreferencePolicy};
use crate::preference_decision::PreferenceMeasure;
use crate::preference_model::{PreferenceModelError, RelationOrigin, ValueModel};
use crate::preference_refusal::{ACCEPT, refusal};
use crate::preference_route_choice::{
    PreferenceEvaluation, PreferenceEvaluationError, PreferenceEvaluationOptions, decide_on_draws,
    decide_preferences,
};
use crate::preference_set_learning::{
    ChoiceConflict, LearnedPreferenceSet, SetLearningError, SetLearningOptions, SetOutcome, learn,
};
use crate::signal::at_stage;

/// What the accepted preference set rests on.
#[derive(Clone, Debug, PartialEq)]
pub enum PreferenceBasis {
    /// The declared ranges and relations alone.
    Declared,
    /// The declared set narrowed by recorded choices.
    CompatibleSet(Box<LearnedPreferenceSet>),
    /// An estimate from recorded choices read as noisy, with its measure over the set.
    ChoiceEstimate(Box<ChoiceModelEstimate>),
}

/// Something a reviewer should look at before relying on the accepted preferences.
#[derive(Clone, Debug, PartialEq)]
pub enum PendingReview {
    /// No record varied the parameter: it stays at its declared range. A parameter the
    /// declaration fixes (`lower == upper`) never raises this item.
    ParameterNotVaried {
        /// Parameter id.
        parameter_id: String,
    },
    /// Records varied the parameter but its range did not narrow. A parameter the declaration
    /// fixes never raises this item.
    ParameterNotNarrowed {
        /// Parameter id.
        parameter_id: String,
    },
    /// The compatible set was learned with a positive tolerance.
    ToleranceApplied {
        /// The tolerance, in utility units.
        tolerance: f64,
    },
    /// The estimate leaves the parameter unresolved: no record varied it.
    ParameterNotLearnable {
        /// Parameter id.
        parameter_id: String,
    },
    /// The point estimate lies outside the parameter's declared range.
    EstimateOutsideDeclared {
        /// Parameter id.
        parameter_id: String,
    },
    /// A share of the estimate's measure draws fell outside the declared set and was dropped.
    DrawsOutsideDeclared {
        /// The dropped share.
        share: f64,
    },
    /// The records the preferences were learned from include stated comparisons: evidence of
    /// what someone said they would choose rather than of what they chose.
    StatedComparisonsUsed {
        /// Number of stated comparisons among the records in the selected scope.
        count: usize,
    },
}

/// Options a learned basis was produced with, retained for replay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LearningOptions {
    /// Options of the compatible-set learner.
    CompatibleSet(SetLearningOptions),
    /// Options of the noisy-choice estimate.
    ChoiceEstimate(ChoiceModelOptions),
}

/// The parts of a preference contract other than its value model, which the accepted
/// preferences supply.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceContractParts {
    /// Candidate actions.
    pub actions: Vec<PreferenceAction>,
    /// Scoring criterion at each parameter value.
    pub criterion: DecisionCriterion,
    /// Hard constraints over model attributes.
    pub constraints: Vec<HardConstraint>,
    /// Population the decision concerns.
    pub target_population: String,
    /// Horizon the decision concerns.
    pub horizon: u32,
    /// Structural-uncertainty policy.
    pub structural_policy: StructuralPolicy,
    /// Preference-uncertainty policy.
    pub preference_policy: PreferencePolicy,
}

/// Reviewed, versioned preferences. Immutable: every change returns a new handle.
#[derive(Clone, Debug, PartialEq)]
pub struct AcceptedPreferences {
    declared: ValueModel,
    declared_identity: String,
    model: ValueModel,
    basis: PreferenceBasis,
    learning: Option<LearningOptions>,
    stated_comparisons: usize,
    version: u32,
    acknowledged: Vec<PendingReview>,
    identity: String,
}

/// Why accepted preferences are refused.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum AcceptedPreferencesError {
    /// The declared value model is invalid.
    #[error("value model: {0}")]
    Model(PreferenceModelError),
    /// The learned object was produced from a different value model.
    #[error("the learned preferences belong to model {found}, not {expected}")]
    ModelMismatch {
        /// Identity of the declared model.
        expected: String,
        /// Model identity the learned object names.
        found: String,
    },
    /// The learning options do not match the learned object.
    #[error("learning option {option} does not match the learned preferences")]
    OptionsMismatch {
        /// Which option differs.
        option: &'static str,
    },
    /// The learned object does not keep what the declared model declares: a learned set without
    /// the declared parameters, ranges or relations, or an estimate whose parameter entries name
    /// other parameters or ranges.
    #[error("the learned preferences do not keep the declared {part}")]
    DeclarationNotKept {
        /// What is not kept: `parameters`, `relations`, `value_relations` or
        /// `estimate_parameters`.
        part: &'static str,
    },
    /// The records supplied are not the records the learned object was learned from.
    #[error("the records are not the ones the preferences were learned from")]
    RecordsMismatch {
        /// Digest the learned object names.
        expected: String,
        /// Digest of the supplied records in the learned object's scope.
        found: String,
    },
    /// A learned set adds a linear or value relation that no recorded choice implies: every
    /// relation beyond the declared ones carries a choice origin.
    #[error("relation {relation} is not implied by a recorded choice")]
    AddedRelationNotFromChoices {
        /// Id of the relation.
        relation: String,
    },
    /// Learning again from the supplied model, records and options does not reproduce the
    /// learned object (within the replay tolerance): its content was edited after learning, or
    /// it was learned under other options.
    #[error("the learned preferences are not reproduced from the records: {field} differs")]
    LearnedNotReproduced {
        /// The first field that differs.
        field: &'static str,
    },
    /// The compatible-set learner refused its input.
    #[error("compatible set: {0}")]
    SetLearning(SetLearningError),
    /// The noisy-choice estimate refused, for the pooled records or the selected decider.
    #[error("choice estimate: {0}")]
    ChoiceModel(ChoiceModelError),
    /// No decider was selected but the records are not pooled.
    #[error("the records are read per decider; name the decider")]
    PooledNotAvailable,
    /// The selected decider has no outcome in the records.
    #[error("no outcome for decider {decider}")]
    DeciderUnknown {
        /// The decider named.
        decider: String,
    },
    /// The selected records contradict the declared preference set.
    #[error("the recorded choices contradict the declared preferences")]
    Contradictory {
        /// The decider, `None` for the pooled records.
        decider: Option<String>,
        /// The conflict found.
        conflict: ChoiceConflict,
    },
    /// The learner could certify neither that the selected records are compatible with the
    /// declared set nor that they contradict it (a value with shape parameters only).
    #[error("compatibility of the recorded choices with the declared preferences is undetermined")]
    Undetermined {
        /// The decider, `None` for the pooled records.
        decider: Option<String>,
        /// Shape boxes left undecided when the subdivision stopped.
        boundary_boxes: usize,
    },
    /// No item was given to acknowledge.
    #[error("no review item was given")]
    NoItems,
    /// An item was given twice.
    #[error("review item {item:?} is repeated")]
    RepeatedItem {
        /// The repeated item.
        item: PendingReview,
    },
    /// An item is not pending: it does not arise from the basis or is already acknowledged.
    #[error("review item {item:?} is not pending")]
    NotPending {
        /// The item.
        item: PendingReview,
    },
    /// A version below one.
    #[error("version {version} is below one")]
    VersionInvalid {
        /// The version given.
        version: u32,
    },
    /// The version cannot be increased further.
    #[error("the version cannot be increased further")]
    VersionExhausted,
}

impl AcceptedPreferencesError {
    /// Stable detail id, prefixed `accepted_preferences.`; a wrapped refusal keeps its own
    /// detail.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Model(inner) => inner.detail(),
            Self::ModelMismatch { .. } => reason_details::accepted_preferences::MODEL_MISMATCH,
            Self::OptionsMismatch { .. } => reason_details::accepted_preferences::OPTIONS_MISMATCH,
            Self::DeclarationNotKept { .. } => {
                reason_details::accepted_preferences::DECLARATION_NOT_KEPT
            }
            Self::RecordsMismatch { .. } => reason_details::accepted_preferences::RECORDS_MISMATCH,
            Self::AddedRelationNotFromChoices { .. } => {
                reason_details::accepted_preferences::ADDED_RELATION_NOT_FROM_CHOICES
            }
            Self::LearnedNotReproduced { .. } => {
                reason_details::accepted_preferences::LEARNED_NOT_REPRODUCED
            }
            Self::SetLearning(inner) => inner.detail(),
            Self::ChoiceModel(inner) => inner.detail(),
            Self::PooledNotAvailable => reason_details::accepted_preferences::POOLED_NOT_AVAILABLE,
            Self::DeciderUnknown { .. } => reason_details::accepted_preferences::DECIDER_UNKNOWN,
            Self::Contradictory { .. } => {
                reason_details::accepted_preferences::CHOICES_CONTRADICTORY
            }
            Self::Undetermined { .. } => reason_details::accepted_preferences::SET_UNDETERMINED,
            Self::NoItems => reason_details::accepted_preferences::ACKNOWLEDGE_EMPTY,
            Self::RepeatedItem { .. } => reason_details::accepted_preferences::ACKNOWLEDGE_REPEATED,
            Self::NotPending { .. } => reason_details::accepted_preferences::NOT_PENDING,
            Self::VersionInvalid { .. } => reason_details::accepted_preferences::VERSION_INVALID,
            Self::VersionExhausted => reason_details::accepted_preferences::VERSION_EXHAUSTED,
        }
    }
}

use AcceptedPreferencesError as E;

impl AcceptedPreferences {
    /// Accept the declared preference set as it stands. Version 1, nothing pending.
    ///
    /// # Errors
    /// An invalid model.
    pub fn declared(model: ValueModel) -> Result<Self, AcceptedPreferencesError> {
        let declared_identity = model.identity().map_err(E::Model)?;
        Ok(Self::build(
            model,
            declared_identity,
            PreferenceBasis::Declared,
            Learning::none(),
            1,
            Vec::new(),
        ))
    }

    /// Accept a compatible set learned from `model` and `records` with `options`. Version 1.
    ///
    /// `records` must be the records the set was learned from: the digest of the records in the
    /// set's scope (the named decider's records for a per-decider set, every record otherwise)
    /// must be the set's records digest. The number of stated comparisons among them is counted
    /// here; a positive count is a pending review item.
    ///
    /// The set itself is verified, not trusted: every relation it adds to the declared ones must
    /// carry a choice origin, and the set is learned again from `model`, `records` and `options`
    /// and must be reproduced field by field (reals within the replay tolerance of
    /// [`crate::preference_artifact::REPLAY_RELATIVE_TOLERANCE`]). Recomputing, rather than
    /// checking a digest the learner stored, is what catches an edit: a digest over the content
    /// can be recomputed by whoever edits the content. The cost is one run of the learner,
    /// paid by this entry point only; [`learn_compatible_set_accepted`] accepts what it has just
    /// learned without a second run.
    ///
    /// # Errors
    /// An invalid model; a learned set whose model identity is not `model`'s; a learned set that
    /// does not keep the declared parameters, ranges or relations, or that adds a relation with
    /// no choice origin; options whose tolerance differs from the one the set was learned with;
    /// records other than the ones the set was learned from; a set the learner does not
    /// reproduce from them; a refusal of the learner.
    pub fn from_compatible_set(
        model: &ValueModel,
        learned: LearnedPreferenceSet,
        options: SetLearningOptions,
        records: &ChoiceRecordSet,
    ) -> Result<Self, AcceptedPreferencesError> {
        check_records(&learned.records_digest, records, learned.decider.as_deref())?;
        check_belongs(
            model,
            &PreferenceBasis::CompatibleSet(Box::new(learned.clone())),
            &LearningOptions::CompatibleSet(options),
        )?;
        check_added_relations(model, &learned)?;
        let fresh = select_learned(model, records, learned.decider.as_deref(), options, None)?;
        crate::accepted_preferences_artifact::compare_learned(&learned, &fresh)
            .map_err(|error| not_reproduced(&error))?;
        Self::accept_set(model, learned, options, records)
    }

    /// Accept a set the caller has just learned from `model`, `records` and `options`.
    fn accept_set(
        model: &ValueModel,
        learned: LearnedPreferenceSet,
        options: SetLearningOptions,
        records: &ChoiceRecordSet,
    ) -> Result<Self, AcceptedPreferencesError> {
        let stated_comparisons = stated_in_scope(records, learned.decider.as_deref());
        Self::from_parts(
            model.clone(),
            PreferenceBasis::CompatibleSet(Box::new(learned)),
            Some(LearningOptions::CompatibleSet(options)),
            stated_comparisons,
            1,
            &[],
        )
    }

    /// Accept an estimate made from `model` and `records` with `options`. Version 1.
    ///
    /// `records` must be the records the estimate was made from: their digest must be the
    /// estimate's records digest. The number of stated comparisons among them, in the
    /// estimate's scope (the named decider's records, or every record for the pooled fit), is
    /// counted here; a positive count is a pending review item.
    ///
    /// The estimate is verified as [`Self::from_compatible_set`] verifies a set: it is made
    /// again from `model`, `records` and `options` (the seed included, so the measure draws are
    /// the same) and must be reproduced field by field. [`estimate_accepted`] accepts what it
    /// has just estimated without a second run.
    ///
    /// # Errors
    /// An invalid model; an estimate whose model identity is not `model`'s; an estimate whose
    /// parameter entries name other parameters or ranges than the declared ones; options whose
    /// level differs from the estimate's; records other than the ones the estimate was made from;
    /// an estimate not reproduced from them; a refusal of the estimator.
    pub fn from_estimate(
        model: &ValueModel,
        estimate: ChoiceModelEstimate,
        options: ChoiceModelOptions,
        records: &ChoiceRecordSet,
    ) -> Result<Self, AcceptedPreferencesError> {
        check_records(&estimate.records_digest, records, None)?;
        check_belongs(
            model,
            &PreferenceBasis::ChoiceEstimate(Box::new(estimate.clone())),
            &LearningOptions::ChoiceEstimate(options),
        )?;
        let fresh = select_estimate(model, records, estimate.decider.as_deref(), options, None)?;
        crate::accepted_preferences_artifact::compare_estimate(&estimate, &fresh)
            .map_err(|error| not_reproduced(&error))?;
        Self::accept_estimate(model, estimate, options, records)
    }

    /// Accept an estimate the caller has just made from `model`, `records` and `options`.
    fn accept_estimate(
        model: &ValueModel,
        estimate: ChoiceModelEstimate,
        options: ChoiceModelOptions,
        records: &ChoiceRecordSet,
    ) -> Result<Self, AcceptedPreferencesError> {
        let stated_comparisons = stated_in_scope(records, estimate.decider.as_deref());
        Self::from_parts(
            model.clone(),
            PreferenceBasis::ChoiceEstimate(Box::new(estimate)),
            Some(LearningOptions::ChoiceEstimate(options)),
            stated_comparisons,
            1,
            &[],
        )
    }

    /// Rebuild a handle from its retained parts, checking that they belong together: the basis
    /// belongs to the model and keeps its declarations, the options match the basis, a declared
    /// basis counts no stated comparisons, the version is at least one and every acknowledged
    /// item arises from the basis, once.
    pub(crate) fn from_parts(
        declared: ValueModel,
        basis: PreferenceBasis,
        learning: Option<LearningOptions>,
        stated_comparisons: usize,
        version: u32,
        acknowledged: &[PendingReview],
    ) -> Result<Self, AcceptedPreferencesError> {
        let declared_identity = declared.identity().map_err(E::Model)?;
        check_basis(&declared_identity, &basis, learning.as_ref())?;
        check_declarations(&declared, &basis)?;
        if matches!(basis, PreferenceBasis::Declared) && stated_comparisons > 0 {
            return Err(E::OptionsMismatch { option: "stated_comparisons" });
        }
        if version == 0 {
            return Err(E::VersionInvalid { version });
        }
        let learning = Learning { options: learning, stated_comparisons };
        let all = review_items(&declared, &basis, stated_comparisons);
        let acknowledged = canonical_acknowledged(&all, &[], acknowledged)?;
        Ok(Self::build(declared, declared_identity, basis, learning, version, acknowledged))
    }

    fn build(
        declared: ValueModel,
        declared_identity: String,
        basis: PreferenceBasis,
        learning: Learning,
        version: u32,
        acknowledged: Vec<PendingReview>,
    ) -> Self {
        let model = match &basis {
            PreferenceBasis::CompatibleSet(learned) => {
                ValueModel { preferences: learned.set.clone(), ..declared.clone() }
            }
            PreferenceBasis::Declared | PreferenceBasis::ChoiceEstimate(_) => declared.clone(),
        };
        let identity = crate::accepted_preferences_artifact::accepted_identity(
            &declared_identity,
            &basis,
            learning.options.as_ref(),
            learning.stated_comparisons,
            version,
            &acknowledged,
        );
        Self {
            declared,
            declared_identity,
            model,
            basis,
            learning: learning.options,
            stated_comparisons: learning.stated_comparisons,
            version,
            acknowledged,
            identity,
        }
    }

    fn retained_learning(&self) -> Learning {
        Learning { options: self.learning, stated_comparisons: self.stated_comparisons }
    }

    /// The value model as declared.
    #[must_use]
    pub fn declared_model(&self) -> &ValueModel {
        &self.declared
    }

    /// Identity of the declared value model.
    #[must_use]
    pub fn declared_identity(&self) -> &str {
        &self.declared_identity
    }

    /// The value model with the accepted preference set: the learned set for a compatible set,
    /// the declared set otherwise.
    #[must_use]
    pub fn model(&self) -> &ValueModel {
        &self.model
    }

    /// What the accepted set rests on.
    #[must_use]
    pub fn basis(&self) -> &PreferenceBasis {
        &self.basis
    }

    /// The estimate's measure over the set; `None` unless the basis is an estimate.
    #[must_use]
    pub fn measure(&self) -> Option<&PreferenceMeasure> {
        match &self.basis {
            PreferenceBasis::ChoiceEstimate(estimate) => Some(&estimate.measure),
            PreferenceBasis::Declared | PreferenceBasis::CompatibleSet(_) => None,
        }
    }

    /// The options a learned basis was produced with; `None` for a declared basis.
    #[must_use]
    pub fn learning_options(&self) -> Option<LearningOptions> {
        self.learning
    }

    /// Number of stated comparisons among the records the preferences were learned from, in
    /// the selected scope; zero for a declared basis.
    #[must_use]
    pub fn stated_comparisons(&self) -> usize {
        self.stated_comparisons
    }

    /// The decider whose records were selected; `None` for a declared basis or pooled records.
    #[must_use]
    pub fn decider(&self) -> Option<&str> {
        match &self.basis {
            PreferenceBasis::Declared => None,
            PreferenceBasis::CompatibleSet(learned) => learned.decider.as_deref(),
            PreferenceBasis::ChoiceEstimate(estimate) => estimate.decider.as_deref(),
        }
    }

    /// Version, starting at 1 and increased by one with each change.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Canonical identity over the declared model identity, the basis content, the learning
    /// options, the version and the acknowledged items.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Review items not yet acknowledged, in the order the module documentation states.
    #[must_use]
    pub fn pending(&self) -> Vec<PendingReview> {
        review_items(&self.declared, &self.basis, self.stated_comparisons)
            .into_iter()
            .filter(|item| !self.acknowledged.contains(item))
            .collect()
    }

    /// Acknowledged review items, in the same order as [`Self::pending`] lists them.
    #[must_use]
    pub fn acknowledged(&self) -> &[PendingReview] {
        &self.acknowledged
    }

    /// A new handle, version + 1, with `items` acknowledged.
    ///
    /// # Errors
    /// No items, an item given twice, an item that is not pending, or a version that cannot
    /// be increased.
    pub fn acknowledge(&self, items: &[PendingReview]) -> Result<Self, AcceptedPreferencesError> {
        if items.is_empty() {
            return Err(E::NoItems);
        }
        let all = review_items(&self.declared, &self.basis, self.stated_comparisons);
        let acknowledged = canonical_acknowledged(&all, &self.acknowledged, items)?;
        let version = self.version.checked_add(1).ok_or(E::VersionExhausted)?;
        Ok(Self::build(
            self.declared.clone(),
            self.declared_identity.clone(),
            self.basis.clone(),
            self.retained_learning(),
            version,
            acknowledged,
        ))
    }

    /// Relearn the compatible set from the declared model and `records`: a new handle with the
    /// version increased by one and the acknowledgements cleared.
    ///
    /// # Errors
    /// As [`learn_compatible_set_accepted`], or a version that cannot be increased.
    pub fn relearn_compatible_set(
        &self,
        records: &ChoiceRecordSet,
        decider: Option<&str>,
        options: SetLearningOptions,
    ) -> Result<Self, AcceptedPreferencesError> {
        self.relearn_set(records, decider, options, None)
    }

    /// [`Self::relearn_compatible_set`] under an execution context, read as
    /// [`crate::preference_set_learning::learn_compatible_set_with_context`] reads it.
    ///
    /// # Errors
    /// As [`Self::relearn_compatible_set`]; a cancelled run is the learner's `Cancelled` refusal.
    pub fn relearn_compatible_set_with_context(
        &self,
        records: &ChoiceRecordSet,
        decider: Option<&str>,
        options: SetLearningOptions,
        ctx: &ExecutionContext,
    ) -> Result<Self, AcceptedPreferencesError> {
        self.relearn_set(records, decider, options, Some(ctx))
    }

    fn relearn_set(
        &self,
        records: &ChoiceRecordSet,
        decider: Option<&str>,
        options: SetLearningOptions,
        ctx: Option<&ExecutionContext>,
    ) -> Result<Self, AcceptedPreferencesError> {
        let version = self.version.checked_add(1).ok_or(E::VersionExhausted)?;
        let learned = select_learned(&self.declared, records, decider, options, ctx)?;
        let learning = Learning {
            options: Some(LearningOptions::CompatibleSet(options)),
            stated_comparisons: stated_in_scope(records, decider),
        };
        Ok(Self::build(
            self.declared.clone(),
            self.declared_identity.clone(),
            PreferenceBasis::CompatibleSet(Box::new(learned)),
            learning,
            version,
            Vec::new(),
        ))
    }

    /// Re-estimate from the declared model and `records`: a new handle with the version
    /// increased by one and the acknowledgements cleared.
    ///
    /// # Errors
    /// As [`estimate_accepted`], or a version that cannot be increased.
    pub fn relearn_estimate(
        &self,
        records: &ChoiceRecordSet,
        decider: Option<&str>,
        options: ChoiceModelOptions,
    ) -> Result<Self, AcceptedPreferencesError> {
        self.reestimate(records, decider, options, None)
    }

    /// [`Self::relearn_estimate`] under an execution context, read as
    /// [`crate::preference_choice_model::estimate_choice_model_with_context`] reads it.
    ///
    /// # Errors
    /// As [`Self::relearn_estimate`]; a cancelled run is the estimator's `Cancelled` refusal.
    pub fn relearn_estimate_with_context(
        &self,
        records: &ChoiceRecordSet,
        decider: Option<&str>,
        options: ChoiceModelOptions,
        ctx: &ExecutionContext,
    ) -> Result<Self, AcceptedPreferencesError> {
        self.reestimate(records, decider, options, Some(ctx))
    }

    fn reestimate(
        &self,
        records: &ChoiceRecordSet,
        decider: Option<&str>,
        options: ChoiceModelOptions,
        ctx: Option<&ExecutionContext>,
    ) -> Result<Self, AcceptedPreferencesError> {
        let version = self.version.checked_add(1).ok_or(E::VersionExhausted)?;
        let estimate = select_estimate(&self.declared, records, decider, options, ctx)?;
        let learning = Learning {
            options: Some(LearningOptions::ChoiceEstimate(options)),
            stated_comparisons: stated_in_scope(records, decider),
        };
        Ok(Self::build(
            self.declared.clone(),
            self.declared_identity.clone(),
            PreferenceBasis::ChoiceEstimate(Box::new(estimate)),
            learning,
            version,
            Vec::new(),
        ))
    }

    /// Assumptions the accepted preferences stand on, sorted and distinct: the parameter
    /// assumptions, and for a learned basis the record set's assumptions and the learning
    /// assumption.
    #[must_use]
    pub fn assumptions(&self) -> Vec<String> {
        let mut all: BTreeSet<String> =
            self.declared.preferences.parameters.iter().map(|p| p.assumption_id.clone()).collect();
        match &self.basis {
            PreferenceBasis::Declared => {}
            PreferenceBasis::CompatibleSet(learned) => all.extend(learned.assumptions.clone()),
            PreferenceBasis::ChoiceEstimate(estimate) => all.extend(estimate.assumptions.clone()),
        }
        all.into_iter().collect()
    }

    /// A preference contract over the accepted model.
    #[must_use]
    pub fn contract(&self, parts: PreferenceContractParts) -> PreferenceContract {
        PreferenceContract {
            model: self.model.clone(),
            actions: parts.actions,
            criterion: parts.criterion,
            constraints: parts.constraints,
            target_population: parts.target_population,
            horizon: parts.horizon,
            structural_policy: parts.structural_policy,
            preference_policy: parts.preference_policy,
        }
    }
}

/// Learn the compatible set from `model` and `records` and accept it. `decider: None` selects
/// the pooled outcome; `Some(id)` selects that decider's own outcome.
///
/// # Errors
/// A refusal of the learner; `None` for records that are not pooled; a decider with no
/// outcome; a contradictory outcome for the selected scope.
pub fn learn_compatible_set_accepted(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    decider: Option<&str>,
    options: SetLearningOptions,
) -> Result<AcceptedPreferences, AcceptedPreferencesError> {
    accept_learned(model, records, decider, options, None)
}

/// [`learn_compatible_set_accepted`] under an execution context, read as
/// [`crate::preference_set_learning::learn_compatible_set_with_context`] reads it.
///
/// # Errors
/// As [`learn_compatible_set_accepted`]; a cancelled run is the learner's `Cancelled` refusal.
pub fn learn_compatible_set_accepted_with_context(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    decider: Option<&str>,
    options: SetLearningOptions,
    ctx: &ExecutionContext,
) -> Result<AcceptedPreferences, AcceptedPreferencesError> {
    accept_learned(model, records, decider, options, Some(ctx))
}

fn accept_learned(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    decider: Option<&str>,
    options: SetLearningOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<AcceptedPreferences, AcceptedPreferencesError> {
    let learned = select_learned(model, records, decider, options, ctx)?;
    AcceptedPreferences::accept_set(model, learned, options, records)
}

/// Estimate from `model` and `records` and accept the estimate. `decider: None` selects the
/// pooled estimate; `Some(id)` selects that decider's own estimate.
///
/// # Errors
/// A refusal of the estimate (for the pooled records or the selected decider); `None` for
/// records that are not pooled; a decider with no outcome.
pub fn estimate_accepted(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    decider: Option<&str>,
    options: ChoiceModelOptions,
) -> Result<AcceptedPreferences, AcceptedPreferencesError> {
    accept_estimate(model, records, decider, options, None)
}

/// [`estimate_accepted`] under an execution context, read as
/// [`crate::preference_choice_model::estimate_choice_model_with_context`] reads it.
///
/// # Errors
/// As [`estimate_accepted`]; a cancelled run is the estimator's `Cancelled` refusal.
pub fn estimate_accepted_with_context(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    decider: Option<&str>,
    options: ChoiceModelOptions,
    ctx: &ExecutionContext,
) -> Result<AcceptedPreferences, AcceptedPreferencesError> {
    accept_estimate(model, records, decider, options, Some(ctx))
}

fn accept_estimate(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    decider: Option<&str>,
    options: ChoiceModelOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<AcceptedPreferences, AcceptedPreferencesError> {
    let estimate = select_estimate(model, records, decider, options, ctx)?;
    AcceptedPreferences::accept_estimate(model, estimate, options, records)
}

/// Decide with accepted preferences: the contract is built over the accepted model, and the
/// accepted measure (an estimate's) is passed when there is one.
///
/// # Errors
/// As [`decide_preferences`].
pub fn decide_with_accepted(
    accepted: &AcceptedPreferences,
    parts: PreferenceContractParts,
    source: &DistributionArtifact,
    options: PreferenceEvaluationOptions,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    let contract = accepted.contract(parts);
    decide_preferences(&contract, source, accepted.measure(), options)
}

/// [`decide_with_accepted`] under an execution context: the route that answers reads the
/// context's cancellation token at its work boundaries.
///
/// # Errors
/// As [`decide_with_accepted`]; a cancelled run is the answering route's `Cancelled` refusal.
pub fn decide_with_accepted_with_context(
    accepted: &AcceptedPreferences,
    parts: PreferenceContractParts,
    source: &DistributionArtifact,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    let contract = accepted.contract(parts);
    decide_on_draws(&contract, source, accepted.measure(), options, Some(ctx))
}

/// How a basis was learned: the options, and the number of stated comparisons among the
/// records in the selected scope.
#[derive(Clone, Copy, Debug)]
struct Learning {
    options: Option<LearningOptions>,
    stated_comparisons: usize,
}

impl Learning {
    const fn none() -> Self {
        Self { options: None, stated_comparisons: 0 }
    }
}

/// Stated comparisons among the records a learner reads for `decider`: every record for the
/// pooled scope, that decider's records otherwise.
pub(crate) fn stated_in_scope(records: &ChoiceRecordSet, decider: Option<&str>) -> usize {
    records
        .records
        .iter()
        .filter(|r| r.belief_source.is_stated())
        .filter(|r| decider.is_none_or(|name| r.decider == name))
        .count()
}

pub(crate) fn select_learned(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    decider: Option<&str>,
    options: SetLearningOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<LearnedPreferenceSet, AcceptedPreferencesError> {
    let result = learn(model, records, options, ctx).map_err(E::SetLearning)?;
    let outcome = match decider {
        None => result.pooled.ok_or(E::PooledNotAvailable)?,
        Some(name) => result
            .per_decider
            .into_iter()
            .find(|d| d.decider == name)
            .map(|d| d.outcome)
            .ok_or_else(|| E::DeciderUnknown { decider: name.to_owned() })?,
    };
    match outcome {
        SetOutcome::Learned(learned) => Ok(learned),
        SetOutcome::Contradictory(conflict) => {
            Err(E::Contradictory { decider: decider.map(str::to_owned), conflict })
        }
        SetOutcome::Undetermined(undetermined) => Err(E::Undetermined {
            decider: decider.map(str::to_owned),
            boundary_boxes: undetermined.region.cells.len(),
        }),
    }
}

pub(crate) fn select_estimate(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    decider: Option<&str>,
    options: ChoiceModelOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<ChoiceModelEstimate, AcceptedPreferencesError> {
    let result = estimate_all(model, records, options, ctx).map_err(E::ChoiceModel)?;
    match decider {
        None => result.pooled.map(|estimate| *estimate).ok_or(E::PooledNotAvailable),
        Some(name) => {
            let outcome = result
                .per_decider
                .into_iter()
                .find(|(d, _)| d == name)
                .map(|(_, outcome)| outcome)
                .ok_or_else(|| E::DeciderUnknown { decider: name.to_owned() })?;
            match outcome {
                EstimateOutcome::Estimated(estimate) => Ok(*estimate),
                EstimateOutcome::Refused(error) => Err(E::ChoiceModel(error)),
            }
        }
    }
}

/// The basis must belong to the declared model and the options to the basis.
fn check_basis(
    declared_identity: &str,
    basis: &PreferenceBasis,
    learning: Option<&LearningOptions>,
) -> Result<(), AcceptedPreferencesError> {
    let mismatch = |found: &str| E::ModelMismatch {
        expected: declared_identity.to_owned(),
        found: found.to_owned(),
    };
    match (basis, learning) {
        (PreferenceBasis::Declared, None) => Ok(()),
        (
            PreferenceBasis::CompatibleSet(learned),
            Some(LearningOptions::CompatibleSet(options)),
        ) => {
            if learned.model_identity != declared_identity {
                return Err(mismatch(&learned.model_identity));
            }
            if options.tolerance.total_cmp(&learned.tolerance).is_ne() {
                return Err(E::OptionsMismatch { option: "tolerance" });
            }
            Ok(())
        }
        (
            PreferenceBasis::ChoiceEstimate(estimate),
            Some(LearningOptions::ChoiceEstimate(options)),
        ) => {
            if estimate.model_identity != declared_identity {
                return Err(mismatch(&estimate.model_identity));
            }
            if options.level.total_cmp(&estimate.level).is_ne() {
                return Err(E::OptionsMismatch { option: "level" });
            }
            Ok(())
        }
        _ => Err(E::OptionsMismatch { option: "basis" }),
    }
}

/// The digest of the records in a learned object's scope must be the digest it names: the named
/// decider's records alone for a per-decider compatible set (`decider`), every record otherwise.
fn check_records(
    expected: &str,
    records: &ChoiceRecordSet,
    decider: Option<&str>,
) -> Result<(), AcceptedPreferencesError> {
    let digest = match decider {
        None => records.digest(),
        Some(name) => ChoiceRecordSet {
            attributes: records.attributes.clone(),
            population: records.population.clone(),
            records: records.records.iter().filter(|r| r.decider == name).cloned().collect(),
            pooling: Pooling::PerDecider,
        }
        .digest(),
    };
    let found = digest.map_err(|error| E::SetLearning(SetLearningError::Records(error)))?;
    if found == expected {
        Ok(())
    } else {
        Err(E::RecordsMismatch { expected: expected.to_owned(), found })
    }
}

/// The checks of [`AcceptedPreferences::from_parts`] that need no relearning: the basis belongs
/// to the model and keeps its declarations, and the options match it. Run before relearning, so
/// a learned object of another model or options is refused as such.
fn check_belongs(
    model: &ValueModel,
    basis: &PreferenceBasis,
    learning: &LearningOptions,
) -> Result<(), AcceptedPreferencesError> {
    let identity = model.identity().map_err(E::Model)?;
    check_basis(&identity, basis, Some(learning))?;
    check_declarations(model, basis)
}

/// A learned object that the learner does not reproduce names the first field that differs.
fn not_reproduced(
    error: &crate::preference_artifact::PreferenceArtifactError,
) -> AcceptedPreferencesError {
    use crate::preference_artifact::PreferenceArtifactError as A;
    match error {
        A::ReplayMismatch { field } => E::LearnedNotReproduced { field },
        // The records digest was checked before learning again, so only a digest the learner
        // itself computes differently remains.
        _ => E::LearnedNotReproduced { field: "records_digest" },
    }
}

/// Every linear or value relation a learned set holds beyond the declared ones is implied by a
/// recorded choice.
fn check_added_relations(
    declared: &ValueModel,
    learned: &LearnedPreferenceSet,
) -> Result<(), AcceptedPreferencesError> {
    let from_choice = |origin: &RelationOrigin| matches!(origin, RelationOrigin::Choice { .. });
    let declared_set = &declared.preferences;
    for relation in &learned.set.relations {
        if !declared_set.relations.contains(relation) && !from_choice(&relation.origin) {
            return Err(E::AddedRelationNotFromChoices { relation: relation.id.clone() });
        }
    }
    for relation in &learned.set.value_relations {
        if !declared_set.value_relations.contains(relation) && !from_choice(&relation.origin) {
            return Err(E::AddedRelationNotFromChoices { relation: relation.id.clone() });
        }
    }
    Ok(())
}

/// A learned basis keeps what the declared model declares. A learned set keeps the declared
/// parameters exactly (ids, units, ranges and assumptions, in order), every declared linear
/// relation and every declared value relation; the relations it adds come from the records. An
/// estimate's parameter entries name declared parameters with their declared ranges, and its
/// not-learnable parameters are declared ones.
fn check_declarations(
    declared: &ValueModel,
    basis: &PreferenceBasis,
) -> Result<(), AcceptedPreferencesError> {
    let parameters = &declared.preferences.parameters;
    match basis {
        PreferenceBasis::Declared => Ok(()),
        PreferenceBasis::CompatibleSet(learned) => {
            let set = &learned.set;
            if &set.parameters != parameters {
                return Err(E::DeclarationNotKept { part: "parameters" });
            }
            if !declared.preferences.relations.iter().all(|r| set.relations.contains(r)) {
                return Err(E::DeclarationNotKept { part: "relations" });
            }
            if !declared.preferences.value_relations.iter().all(|r| set.value_relations.contains(r))
            {
                return Err(E::DeclarationNotKept { part: "value_relations" });
            }
            Ok(())
        }
        PreferenceBasis::ChoiceEstimate(estimate) => {
            let entries_kept = estimate.parameters.iter().all(|entry| {
                parameters.iter().any(|p| {
                    p.id == entry.parameter_id
                        && p.lower.total_cmp(&entry.declared.0).is_eq()
                        && p.upper.total_cmp(&entry.declared.1).is_eq()
                })
            });
            let not_learnable_kept =
                estimate.not_learnable.iter().all(|id| parameters.iter().any(|p| &p.id == id));
            if entries_kept && not_learnable_kept {
                Ok(())
            } else {
                Err(E::DeclarationNotKept { part: "estimate_parameters" })
            }
        }
    }
}

/// Every review item the basis raises, before acknowledgements, in the documented order.
///
/// A parameter the declaration fixes (`lower == upper`) is a value stated outright: it raises
/// no item for not being varied or narrowed by the records.
fn review_items(
    declared: &ValueModel,
    basis: &PreferenceBasis,
    stated_comparisons: usize,
) -> Vec<PendingReview> {
    let fixed = |id: &str| {
        declared
            .preferences
            .parameters
            .iter()
            .any(|p| p.id == id && p.lower.total_cmp(&p.upper).is_eq())
    };
    let mut items = Vec::new();
    match basis {
        PreferenceBasis::Declared => {}
        PreferenceBasis::CompatibleSet(learned) => {
            for range in &learned.ranges {
                if fixed(&range.parameter_id) {
                    continue;
                }
                if !range.varied_in_records {
                    items.push(PendingReview::ParameterNotVaried {
                        parameter_id: range.parameter_id.clone(),
                    });
                } else if !range.narrowed {
                    items.push(PendingReview::ParameterNotNarrowed {
                        parameter_id: range.parameter_id.clone(),
                    });
                }
            }
            if learned.tolerance > 0.0 {
                items.push(PendingReview::ToleranceApplied { tolerance: learned.tolerance });
            }
        }
        PreferenceBasis::ChoiceEstimate(estimate) => {
            for parameter in &declared.preferences.parameters {
                let id = &parameter.id;
                if estimate.not_learnable.contains(id) {
                    items.push(PendingReview::ParameterNotLearnable { parameter_id: id.clone() });
                } else if estimate
                    .parameters
                    .iter()
                    .any(|p| p.parameter_id == *id && !p.inside_declared)
                {
                    items.push(PendingReview::EstimateOutsideDeclared { parameter_id: id.clone() });
                }
            }
            if estimate.share_outside_declared > 0.0 {
                items.push(PendingReview::DrawsOutsideDeclared {
                    share: estimate.share_outside_declared,
                });
            }
        }
    }
    if stated_comparisons > 0 && !matches!(basis, PreferenceBasis::Declared) {
        items.push(PendingReview::StatedComparisonsUsed { count: stated_comparisons });
    }
    items
}

/// `existing` plus `added`, each item checked to be pending and given once, in the order of
/// `all`.
fn canonical_acknowledged(
    all: &[PendingReview],
    existing: &[PendingReview],
    added: &[PendingReview],
) -> Result<Vec<PendingReview>, AcceptedPreferencesError> {
    for (position, item) in added.iter().enumerate() {
        if added[..position].contains(item) {
            return Err(E::RepeatedItem { item: item.clone() });
        }
        if !all.contains(item) || existing.contains(item) {
            return Err(E::NotPending { item: item.clone() });
        }
    }
    Ok(all.iter().filter(|item| existing.contains(item) || added.contains(item)).cloned().collect())
}

impl AcceptedPreferencesError {
    /// Structured refusal for a failure to accept, acknowledge or version preferences.
    ///
    /// A value-model, set-learning or choice-model refusal keeps its own refusal. A
    /// contradiction states the tolerance needed as `expected` and the conflicting records as
    /// `supplied`.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), ACCEPT, self.detail())
                .remedy(remedy)
        };
        let unsatisfied = |remedy| {
            refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                ACCEPT,
                self.detail(),
            )
            .remedy(remedy)
        };
        match self {
            Self::Model(error) => at_stage(error.to_refusal(), ACCEPT),
            Self::SetLearning(error) => at_stage(error.to_refusal(), ACCEPT),
            Self::ChoiceModel(error) => at_stage(error.to_refusal(), ACCEPT),
            Self::ModelMismatch { expected, found } => {
                unsatisfied("accept the learned preferences under the model they were learned from")
                    .expected(expected.clone())
                    .supplied(found.clone())
                    .build()
            }
            Self::OptionsMismatch { option } => unsatisfied(
                "accept the learned preferences with the options they were learned with",
            )
            .offending(*option)
            .build(),
            Self::DeclarationNotKept { part } => unsatisfied(
                "accept the learned preferences as the learner returned them, under the model \
                 they were learned from",
            )
            .offending(*part)
            .build(),
            Self::RecordsMismatch { expected, found } => {
                unsatisfied("supply the records the preferences were learned from")
                    .expected(expected.clone())
                    .supplied(found.clone())
                    .build()
            }
            Self::AddedRelationNotFromChoices { relation } => unsatisfied(
                "accept the learned preferences as the learner returned them; declare an \
                 analyst relation in the value model instead",
            )
            .offending(relation.clone())
            .build(),
            Self::LearnedNotReproduced { field } => unsatisfied(
                "accept the learned preferences as the learner returned them, with the records \
                 and options they were learned from",
            )
            .offending(*field)
            .build(),
            Self::PooledNotAvailable => {
                invalid("name the decider whose preferences to accept").build()
            }
            Self::DeciderUnknown { decider } => {
                invalid("name a decider the records hold").offending(decider.clone()).build()
            }
            Self::Contradictory { decider, conflict } => unsatisfied(
                "raise the tolerance to the stated value or remove one of the named records",
            )
            .offending_opt(decider.clone())
            .expected(format!("tolerance >= {}", conflict.tolerance_needed))
            .supplied(conflict.records.join(","))
            .build(),
            Self::Undetermined { decider, boundary_boxes } => unsatisfied(
                "raise SetLearningOptions::max_boxes or lower min_width_fraction, or narrow the \
                 declared range of the shape parameters",
            )
            .offending_opt(decider.clone())
            .supplied(format!("{boundary_boxes} undecided shape boxes"))
            .build(),
            Self::NoItems => invalid("name at least one pending review item").build(),
            Self::RepeatedItem { item } => {
                invalid("name each review item once").offending(format!("{item:?}")).build()
            }
            Self::NotPending { item } => invalid("name only review items that are pending")
                .offending(format!("{item:?}"))
                .build(),
            Self::VersionInvalid { version } => invalid("supply a version of at least one")
                .expected("1")
                .supplied(version.to_string())
                .build(),
            Self::VersionExhausted => {
                invalid("accept the preferences anew instead of revising them").build()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preference_model::{Attribute, PreferenceExpr, PreferenceParameter, PreferenceSet};

    fn model() -> ValueModel {
        ValueModel {
            attributes: vec![Attribute { id: "benefit".into(), units: "events".into() }],
            preferences: PreferenceSet::new(
                vec![PreferenceParameter {
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
        }
    }

    #[test]
    fn a_version_below_one_is_refused() {
        let error =
            AcceptedPreferences::from_parts(model(), PreferenceBasis::Declared, None, 0, 0, &[])
                .unwrap_err();
        assert_eq!(error, E::VersionInvalid { version: 0 });
        assert_eq!(
            error.detail(),
            antecedent_core::reason_details::accepted_preferences::VERSION_INVALID
        );
    }

    #[test]
    fn the_last_version_cannot_be_increased() {
        let last = AcceptedPreferences::from_parts(
            model(),
            PreferenceBasis::Declared,
            None,
            0,
            u32::MAX,
            &[],
        )
        .unwrap();
        let records = ChoiceRecordSet {
            attributes: model().attributes,
            population: "p".into(),
            records: vec![],
            pooling: crate::choice_record::Pooling::PerDecider,
        };
        let error =
            last.relearn_compatible_set(&records, None, SetLearningOptions::default()).unwrap_err();
        assert_eq!(error, E::VersionExhausted);
        assert_eq!(
            error.detail(),
            antecedent_core::reason_details::accepted_preferences::VERSION_EXHAUSTED
        );
        let error =
            last.relearn_estimate(&records, None, ChoiceModelOptions::default()).unwrap_err();
        assert_eq!(error, E::VersionExhausted);
    }

    #[test]
    fn acknowledged_items_follow_the_order_of_the_full_list() {
        let a = PendingReview::ParameterNotVaried { parameter_id: "a".into() };
        let b = PendingReview::ParameterNotVaried { parameter_id: "b".into() };
        let c = PendingReview::ToleranceApplied { tolerance: 0.5 };
        let all = vec![a.clone(), b.clone(), c.clone()];
        let merged =
            canonical_acknowledged(&all, std::slice::from_ref(&c), std::slice::from_ref(&a))
                .unwrap();
        assert_eq!(merged, vec![a.clone(), c.clone()]);
        assert_eq!(
            canonical_acknowledged(&all, std::slice::from_ref(&c), std::slice::from_ref(&c)),
            Err(E::NotPending { item: c })
        );
        assert_eq!(
            canonical_acknowledged(&all, &[], &[b.clone(), b.clone()]),
            Err(E::RepeatedItem { item: b })
        );
    }

    #[test]
    fn a_basis_with_options_of_the_other_kind_is_refused() {
        let error = AcceptedPreferences::from_parts(
            model(),
            PreferenceBasis::Declared,
            Some(LearningOptions::CompatibleSet(SetLearningOptions::default())),
            0,
            1,
            &[],
        )
        .unwrap_err();
        assert_eq!(error, E::OptionsMismatch { option: "basis" });
    }

    #[test]
    fn a_declared_basis_with_stated_comparisons_is_refused() {
        let error =
            AcceptedPreferences::from_parts(model(), PreferenceBasis::Declared, None, 2, 1, &[])
                .unwrap_err();
        assert_eq!(error, E::OptionsMismatch { option: "stated_comparisons" });
    }
}
