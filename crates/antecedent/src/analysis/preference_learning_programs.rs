//! Compiled learning of preferences from recorded choices.
//!
//! Recorded choices between options with stated or estimated attribute values say which
//! trade-offs the decision-makers who chose could have held. Each learner compiles here to a
//! program of its own family and runs the same kernel the design crate exposes, on the same
//! inputs, so every range, conflict, estimate and interval is the kernel's:
//!
//! | entry | family | kernel |
//! |---|---|---|
//! | [`compile_compatible_set`] | `PreferenceCompatibleSet` | `learn_compatible_set_with_context`; accepting reads `learn_compatible_set_accepted_with_context` |
//! | [`compile_compatible_set_relearn`] | `PreferenceCompatibleSet` | the same learner on the accepted preferences' declared model; accepting reads `AcceptedPreferences::relearn_compatible_set_with_context` |
//! | [`compile_choice_model`] | `PreferenceChoiceModel` | `estimate_choice_model_with_context`; accepting reads `estimate_accepted_with_context` |
//! | [`compile_choice_model_relearn`] | `PreferenceChoiceModel` | the same estimator on the accepted preferences' declared model; accepting reads `AcceptedPreferences::relearn_estimate_with_context` |
//!
//! The choice model's intervals are its interval claim (unit `preference_learning.choice_model`),
//! gated like every interval route; the compatible set is an assumption range, never a
//! probability (unit `preference_learning.compatible_set`). Accepting a learned result (one
//! decider's outcome, or the pooled one) is a reading of the same program: the accepted
//! preferences are not a support unit of their own.
//!
//! A program's identity covers the model, the records (their canonical digest, then their ids
//! in declaration order), the options and, when relearning, the accepted preferences revised.
//! A model or record set the learner would refuse has no identity: it is refused at compile
//! with the learner's own error ([`TerminalError::Kernel`]). The learner checks its options
//! before the model and records, so a declaration that is invalid in both is reported by its
//! model or records at compile, where the kernel alone would report the options. The compiler's
//! own refusals (re-entry, a release build's support gate) are [`TerminalError::Refused`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_design::accepted_preferences::{
    AcceptedPreferences, AcceptedPreferencesError, estimate_accepted_with_context,
    learn_compatible_set_accepted_with_context,
};
use antecedent_design::choice_record::ChoiceRecordSet;
use antecedent_design::preference_choice_model::{
    ChoiceModelError, ChoiceModelOptions, ChoiceModelResult, estimate_choice_model_with_context,
};
use antecedent_design::preference_model::ValueModel;
use antecedent_design::preference_set_learning::{
    SetLearningError, SetLearningOptions, SetLearningResult, learn_compatible_set_with_context,
};
use antecedent_stats::linear_program::LpOptions;
use antecedent_stats::{ConditionalLogitOptions, ShapeSearchOptions};

use crate::analysis::supplied_identity::{Canonical, Declaration};
pub use crate::analysis::supplied_terminal::TerminalError;
use crate::analysis::supplied_terminal::{
    Reading, TerminalSpec, admit_compile, compiled_terminal, lower,
};
use crate::planner::lowered::{Composition, EvidenceRole, FunctionalKind, ProgramCell};

const COMPATIBLE_SET_TAG: &str = "antecedent.preference_compatible_set.v1";
const CHOICE_MODEL_TAG: &str = "antecedent.preference_choice_model.v1";

/// The options of the compatible-set learner. Every field is destructured, so an option added
/// to the learner cannot be left out of the identity.
struct SetOptions<'a>(&'a SetLearningOptions);

impl Canonical for SetOptions<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let SetLearningOptions {
            tolerance,
            conflict_search_size,
            lp,
            max_boxes,
            min_width_fraction,
        } = self.0;
        let LpOptions { feasibility_tolerance, optimality_tolerance, max_pivots } = lp;
        out.real(*tolerance)
            .word(*conflict_search_size as u64)
            .real(*feasibility_tolerance)
            .real(*optimality_tolerance)
            .word(*max_pivots as u64)
            .word(*max_boxes as u64)
            .real(*min_width_fraction);
    }
}

/// The options of the choice-model estimator. Every field is destructured.
struct ModelOptions<'a>(&'a ChoiceModelOptions);

impl Canonical for ModelOptions<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let ChoiceModelOptions { level, n_draws, seed, fit, shape_search } = self.0;
        let ConditionalLogitOptions { max_iterations, gradient_tolerance, separation_bound } = fit;
        let ShapeSearchOptions { grid_points, max_dimensions, profile_points, max_refinements } =
            shape_search;
        out.real(*level)
            .word(*n_draws as u64)
            .word(*seed)
            .word(*max_iterations as u64)
            .real(*gradient_tolerance)
            .real(*separation_bound)
            .word(*grid_points as u64)
            .word(*max_dimensions as u64)
            .word(*profile_points as u64)
            .word(*max_refinements as u64);
    }
}

/// The declared model and records of a learner, or the learner's refusal of a model or record
/// set that has no identity.
fn declare_inputs(
    declaration: &mut Declaration,
    model: &ValueModel,
    records: &ChoiceRecordSet,
    prior: Option<&AcceptedPreferences>,
) -> Result<(), InputRefusal> {
    let model_identity = model.identity().map_err(InputRefusal::Model)?;
    let records_digest = records.digest().map_err(InputRefusal::Records)?;
    // The digest reads the records as a set; their declared order is kept beside it, since the
    // per-decider outcomes and the conflicts are reported in the records' order.
    let ids: Vec<&str> = records.records.iter().map(|record| record.id.as_str()).collect();
    declaration
        .field("model", model_identity.as_str())
        .field("records", records_digest.as_str())
        .field("record_ids", &ids)
        .field("revises", &prior.map(AcceptedPreferences::identity));
    Ok(())
}

/// Why a learner's declared inputs have no identity.
enum InputRefusal {
    Model(antecedent_design::preference_model::PreferenceModelError),
    Records(antecedent_design::choice_record::ChoiceRecordError),
}

// --- the compatible set -----------------------------------------------------------------------

compiled_terminal! {
    /// A compiled learning of the preference parameters compatible with recorded choices.
    #[derive(Clone, Debug)]
    CompiledPreferenceCompatibleSet {
        model: ValueModel,
        records: ChoiceRecordSet,
        options: SetLearningOptions,
        prior: Option<AcceptedPreferences>,
    }
}

/// Compile the learning of the parameters of `model` compatible with `records`.
///
/// # Errors
/// A model or record set the learner refuses (`TerminalError::Kernel`, the learner's own
/// error); `public_builder_reentry` inside a lowered dispatch.
pub fn compile_compatible_set(
    model: ValueModel,
    records: ChoiceRecordSet,
    options: SetLearningOptions,
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceCompatibleSet, TerminalError<SetLearningError>> {
    compile_set(model, records, options, None, ctx)
}

/// Compile the relearning of `accepted` from `records`: the learner runs on the accepted
/// preferences' declared model, and accepting gives the accepted preferences' next version.
///
/// # Errors
/// As [`compile_compatible_set`].
pub fn compile_compatible_set_relearn(
    accepted: AcceptedPreferences,
    records: ChoiceRecordSet,
    options: SetLearningOptions,
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceCompatibleSet, TerminalError<SetLearningError>> {
    let model = accepted.declared_model().clone();
    compile_set(model, records, options, Some(accepted), ctx)
}

fn compile_set(
    model: ValueModel,
    records: ChoiceRecordSet,
    options: SetLearningOptions,
    prior: Option<AcceptedPreferences>,
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceCompatibleSet, TerminalError<SetLearningError>> {
    admit_compile(ctx)?;
    let mut declaration = Declaration::new(COMPATIBLE_SET_TAG);
    declare_inputs(&mut declaration, &model, &records, prior.as_ref()).map_err(|refusal| {
        TerminalError::Kernel(match refusal {
            InputRefusal::Model(error) => SetLearningError::Model(error),
            InputRefusal::Records(error) => SetLearningError::Records(error),
        })
    })?;
    declaration.field("options", &SetOptions(&options));
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::PreferenceCompatibleSet,
        plan_id: "supplied.preference_compatible_set.v1",
        functional: FunctionalKind::IdentifiedSet,
        graph_class: "supplied_choice_records",
        question: declaration.finish(),
        evaluator: "preference_set_learning::learn_compatible_set_with_context",
        reading: Reading::Envelope,
        role: EvidenceRole::Observations,
        composition: Composition::Single,
        decision: None,
        claim: Some(antecedent_contract::ClaimKind::Bounds),
    })?;
    Ok(CompiledPreferenceCompatibleSet { lowered, model, records, options, prior })
}

impl CompiledPreferenceCompatibleSet {
    /// The declared value model.
    #[must_use]
    pub const fn model(&self) -> &ValueModel {
        &self.model
    }

    /// The declared records.
    #[must_use]
    pub const fn records(&self) -> &ChoiceRecordSet {
        &self.records
    }

    /// The accepted preferences this learning revises, when it relearns them.
    #[must_use]
    pub const fn revises(&self) -> Option<&AcceptedPreferences> {
        self.prior.as_ref()
    }

    /// Learn the compatible set: pooled and per decider, with the smallest conflicts of
    /// contradictory records. The learner reads the context's cancellation token before every
    /// linear program.
    ///
    /// # Errors
    /// The learner's own refusals (invalid options, records the model cannot read, a
    /// linear-program failure, cancellation), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SetLearningResult, TerminalError<SetLearningError>> {
        let scoped = self.enter(ctx)?;
        learn_compatible_set_with_context(&self.model, &self.records, self.options, &scoped)
            .map_err(TerminalError::Kernel)
    }

    /// Learn and accept the outcome of `decider` (`None`: the pooled outcome): version one of
    /// new accepted preferences, or, when relearning, the next version of the accepted
    /// preferences revised, with their acknowledgements cleared.
    ///
    /// # Errors
    /// The learner's refusals; no pooled outcome for records that are not pooled; a decider
    /// with no outcome; a contradictory outcome; a version that cannot be increased; or the
    /// compiler's.
    pub fn accept(
        &self,
        decider: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<AcceptedPreferences, TerminalError<AcceptedPreferencesError>> {
        let scoped = self.enter(ctx)?;
        match &self.prior {
            None => learn_compatible_set_accepted_with_context(
                &self.model,
                &self.records,
                decider,
                self.options,
                &scoped,
            ),
            Some(prior) => prior.relearn_compatible_set_with_context(
                &self.records,
                decider,
                self.options,
                &scoped,
            ),
        }
        .map_err(TerminalError::Kernel)
    }
}

// --- the choice model -------------------------------------------------------------------------

compiled_terminal! {
    /// A compiled estimate of the preference parameters from choices read as noisy.
    #[derive(Clone, Debug)]
    CompiledPreferenceChoiceModel {
        model: ValueModel,
        records: ChoiceRecordSet,
        options: ChoiceModelOptions,
        prior: Option<AcceptedPreferences>,
    }
}

/// Compile the estimate of the parameters of `model` from `records` read as noisy choices.
///
/// # Errors
/// A model or record set the estimator refuses (`TerminalError::Kernel`, the estimator's own
/// error); `public_builder_reentry` inside a lowered dispatch.
pub fn compile_choice_model(
    model: ValueModel,
    records: ChoiceRecordSet,
    options: ChoiceModelOptions,
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceChoiceModel, TerminalError<ChoiceModelError>> {
    compile_model(model, records, options, None, ctx)
}

/// Compile the re-estimate of `accepted` from `records`: the estimator runs on the accepted
/// preferences' declared model, and accepting gives the accepted preferences' next version.
///
/// # Errors
/// As [`compile_choice_model`].
pub fn compile_choice_model_relearn(
    accepted: AcceptedPreferences,
    records: ChoiceRecordSet,
    options: ChoiceModelOptions,
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceChoiceModel, TerminalError<ChoiceModelError>> {
    let model = accepted.declared_model().clone();
    compile_model(model, records, options, Some(accepted), ctx)
}

fn compile_model(
    model: ValueModel,
    records: ChoiceRecordSet,
    options: ChoiceModelOptions,
    prior: Option<AcceptedPreferences>,
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceChoiceModel, TerminalError<ChoiceModelError>> {
    admit_compile(ctx)?;
    let mut declaration = Declaration::new(CHOICE_MODEL_TAG);
    declare_inputs(&mut declaration, &model, &records, prior.as_ref()).map_err(|refusal| {
        TerminalError::Kernel(match refusal {
            InputRefusal::Model(error) => ChoiceModelError::Model(error),
            InputRefusal::Records(error) => ChoiceModelError::Records(error),
        })
    })?;
    declaration.field("options", &ModelOptions(&options));
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::PreferenceChoiceModel,
        plan_id: "supplied.preference_choice_model.v1",
        functional: FunctionalKind::Association,
        graph_class: "supplied_choice_records",
        question: declaration.finish(),
        evaluator: "preference_choice_model::estimate_choice_model_with_context",
        reading: Reading::Sampling,
        role: EvidenceRole::Observations,
        composition: Composition::Single,
        decision: None,
        // A choice model is a regression of recorded choices on option attributes: its
        // functional is an association, so it states an associational claim, never a
        // point that reads as an intervention effect.
        claim: Some(antecedent_contract::ClaimKind::Associational),
    })?;
    Ok(CompiledPreferenceChoiceModel { lowered, model, records, options, prior })
}

impl CompiledPreferenceChoiceModel {
    /// The declared value model.
    #[must_use]
    pub const fn model(&self) -> &ValueModel {
        &self.model
    }

    /// The declared records.
    #[must_use]
    pub const fn records(&self) -> &ChoiceRecordSet {
        &self.records
    }

    /// The accepted preferences this estimate revises, when it re-estimates them.
    #[must_use]
    pub const fn revises(&self) -> Option<&AcceptedPreferences> {
        self.prior.as_ref()
    }

    /// Estimate the parameters, pooled and per decider, with their intervals and the measure
    /// over the preference set they imply. The estimator reads the context's cancellation
    /// token before every fit, Newton step and block of measure draws.
    ///
    /// # Errors
    /// The estimator's own refusals (invalid options, records the model cannot read, a fit
    /// that does not identify or converge, cancellation), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ChoiceModelResult, TerminalError<ChoiceModelError>> {
        let scoped = self.enter(ctx)?;
        estimate_choice_model_with_context(&self.model, &self.records, self.options, &scoped)
            .map_err(TerminalError::Kernel)
    }

    /// Estimate and accept the estimate of `decider` (`None`: the pooled estimate): version one
    /// of new accepted preferences, or, when re-estimating, the next version of the accepted
    /// preferences revised, with their acknowledgements cleared.
    ///
    /// # Errors
    /// The estimator's refusals; no pooled estimate for records that are not pooled; a decider
    /// with no estimate; a version that cannot be increased; or the compiler's.
    pub fn accept(
        &self,
        decider: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<AcceptedPreferences, TerminalError<AcceptedPreferencesError>> {
        let scoped = self.enter(ctx)?;
        match &self.prior {
            None => estimate_accepted_with_context(
                &self.model,
                &self.records,
                decider,
                self.options,
                &scoped,
            ),
            Some(prior) => {
                prior.relearn_estimate_with_context(&self.records, decider, self.options, &scoped)
            }
        }
        .map_err(TerminalError::Kernel)
    }
}
