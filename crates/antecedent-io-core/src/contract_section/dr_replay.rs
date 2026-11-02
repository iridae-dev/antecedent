//! Replay of a checked doubly-robust (DR-learner) average effect from a full-source witness.
//!
//! A contract that declares a [`CheckedDrLoweringWire`] says which unweighted, all-observed,
//! binary-treatment cross-fitted fit produced the body. With the complete original table in
//! the artifact (the source witness, bound to the contract by the data-snapshot content
//! digest), an independent verifier can run that fit again with
//! [`antecedent_learn::dr_replay::replay_dr_ate`] and compare what comes out with what the body
//! reports: the point estimate bit for bit, the portable fitted effect (predictor, ordered
//! feature ids, intercept, support box), the cross-fit record, and the CATE vector when one is
//! published. Only a replay that reproduces every one of them clears the obligation to name
//! the missing estimator operation.
//!
//! The master seed of the fold plan is the one the body records (`crossfit_seed`), an
//! execution fact covered by the claim's result digest; it is not part of the program
//! declaration. The analytic standard error and the pointwise CATE standard errors are
//! recomputed and compared bit for bit; no bootstrap or interval is published for this route
//! (the DR-learner fit takes no replicate count), so a body that publishes an interval, a joint
//! covariance or a leaf dispersion is outside the operation and is not cleared.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use antecedent_core::{CancellationToken, Determinism, ExecutionContext, KernelPolicy};
use antecedent_learn::LearnerSpec;
use antecedent_learn::dr_replay::{
    DrOverlap, DrReplayError, DrReplayInput, DrReplaySpec, MAX_REPLAY_FOLDS, replay_dr_ate,
};

use super::AnalysisResultContractWire;
use super::consume::cbor_eq;
use super::replay::{contract_uses_estimator, is_set_value, overlap_policy_tag};
use crate::contract_reason::ContractReason;
use crate::contract_wire::{
    CHECKED_DR_LOWERING_FORMAT, CHECKED_DR_PROCEDURE, CheckedDrLoweringWire, CheckedLearnerWire,
};
use crate::source_witness::{
    SourceWitnessError, SourceWitnessLimits, SourceWitnessWire, verify_source_witness,
};
use crate::{AnalysisResultHeader, AnalysisResultWire};

/// Working memory a replay may admit when the caller sets no other bound.
const DEFAULT_REPLAY_MEMORY_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Bounds a source replay runs under: what a witness may declare, how much working memory the
/// fits may admit, and the token that stops the work.
#[derive(Clone, Debug)]
pub struct SourceReplayBounds {
    /// Limits a witness is held to before any table is allocated for it.
    pub witness: SourceWitnessLimits,
    /// Working-memory capacity of the replay's admission pool, in bytes.
    pub memory_bytes: u64,
    /// Cancellation token of the replay.
    pub cancellation: CancellationToken,
}

impl Default for SourceReplayBounds {
    fn default() -> Self {
        Self {
            witness: SourceWitnessLimits::default(),
            memory_bytes: DEFAULT_REPLAY_MEMORY_BYTES,
            cancellation: CancellationToken::new(),
        }
    }
}

/// Why a source replay did not clear the estimator operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrReplayRefusal {
    /// The declaration is not a well-formed lowering; the text names the fault.
    Lowering(&'static str),
    /// The declaration disagrees with the contract it sits in; the text names the fault.
    Binding(&'static str),
    /// The witness is not usable: malformed, over its limits, or not the snapshotted table.
    Source(SourceWitnessError),
    /// The declaration or the witnessed table lies outside the operation the replay supports.
    Unsupported(&'static str),
    /// The replay ran (or could not run the fit the artifact reports) and the result differs
    /// from the body; the text names what.
    Mismatch(&'static str),
    /// The cancellation token stopped the replay. Nothing was verified.
    Cancelled,
    /// The resource bounds refused the replay. Nothing was verified.
    Resource(String),
}

impl DrReplayRefusal {
    /// The stable unresolved reason this refusal publishes.
    #[must_use]
    pub const fn reason(&self) -> ContractReason {
        match self {
            Self::Lowering(_) => ContractReason::ProgramCheckedDrLowering,
            Self::Binding(_) => ContractReason::ProgramCheckedDrBinding,
            Self::Source(_) => ContractReason::DependenciesCheckedDrReplaySource,
            Self::Unsupported(_) => ContractReason::DependenciesCheckedDrReplayUnsupported,
            Self::Mismatch(_) => ContractReason::DependenciesCheckedDrReplayMismatch,
            Self::Cancelled | Self::Resource(_) => {
                ContractReason::DependenciesCheckedDrReplayResource
            }
        }
    }
}

impl fmt::Display for DrReplayRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lowering(what) => write!(f, "malformed doubly-robust lowering: {what}"),
            Self::Binding(what) => {
                write!(f, "doubly-robust lowering disagrees with the contract: {what}")
            }
            Self::Source(error) => write!(f, "{error}"),
            Self::Unsupported(what) => write!(f, "unsupported doubly-robust replay: {what}"),
            Self::Mismatch(what) => {
                write!(f, "doubly-robust replay does not reproduce the body: {what}")
            }
            Self::Cancelled => write!(f, "the doubly-robust replay was cancelled"),
            Self::Resource(what) => {
                write!(f, "the doubly-robust replay was refused by its bounds: {what}")
            }
        }
    }
}

impl std::error::Error for DrReplayRefusal {}

/// The state of the source witness a verification holds.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SourceWitnessState<'a> {
    /// A decoded witness, not yet bound to the snapshot.
    Present(&'a SourceWitnessWire),
    /// A witness section that could not be used; the replay cannot run.
    Unusable(&'a SourceWitnessError),
}

/// A source section as a verification holds it: decoded, or the reason it cannot be used.
#[derive(Clone, Debug)]
pub(crate) enum HeldSource {
    /// A decoded witness whose declared sizes are inside the limits.
    Present(SourceWitnessWire),
    /// A section the replay cannot use.
    Unusable(SourceWitnessError),
}

impl HeldSource {
    /// The state to hand to a verification, replayed under `bounds`.
    pub(crate) fn replay<'a>(&'a self, bounds: &'a SourceReplayBounds) -> SourceReplay<'a> {
        SourceReplay {
            state: match self {
                Self::Present(witness) => SourceWitnessState::Present(witness),
                Self::Unusable(error) => SourceWitnessState::Unusable(error),
            },
            bounds,
        }
    }
}

/// The artifact's source section, or `None` when it carries none.
///
/// The section's decoded byte length is checked against `limits` before any of it is
/// parsed, and the parsed declared sizes are checked again; every fault is an unusable
/// witness, never a silent absence.
pub(crate) fn hold_source_section(
    artifact: &crate::EncodedArtifact,
    limits: &SourceWitnessLimits,
) -> Option<HeldSource> {
    let mut found =
        artifact.sections.iter().filter(|section| section.id == crate::SOURCE_WITNESS_SECTION);
    let section = found.next()?;
    if found.next().is_some() {
        return Some(HeldSource::Unusable(SourceWitnessError::Malformed(
            "more than one source section",
        )));
    }
    if !matches!(u64::try_from(section.data.len()), Ok(bytes) if bytes <= limits.max_bytes) {
        return Some(HeldSource::Unusable(SourceWitnessError::TooLarge));
    }
    let witness: SourceWitnessWire = match crate::from_cbor(&section.data) {
        Ok(witness) => witness,
        Err(_) => {
            return Some(HeldSource::Unusable(SourceWitnessError::Malformed(
                "the source section does not decode",
            )));
        }
    };
    Some(match witness.check_declared(limits) {
        Ok(()) => HeldSource::Present(witness),
        Err(error) => HeldSource::Unusable(error),
    })
}

/// A source witness state with the bounds its replay runs under.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SourceReplay<'a> {
    pub(crate) state: SourceWitnessState<'a>,
    pub(crate) bounds: &'a SourceReplayBounds,
}

/// A declaration read into the types the replay takes.
struct Declaration<'a> {
    lowering: &'a CheckedDrLoweringWire,
    folds: usize,
    outcome: LearnerSpec,
    treatment: LearnerSpec,
    final_learner: LearnerSpec,
    overlap: DrOverlap,
}

fn learner_of(wire: &CheckedLearnerWire) -> Result<LearnerSpec, DrReplayRefusal> {
    wire.to_spec()
        .filter(|spec| spec.validate().is_ok())
        .ok_or(DrReplayRefusal::Lowering("a learner is unknown or its parameters are invalid"))
}

fn read_declaration(
    contract: &AnalysisResultContractWire,
) -> Result<Declaration<'_>, DrReplayRefusal> {
    let lowering = contract
        .program
        .as_ref()
        .and_then(|program| program.checked_dr_lowering.as_ref())
        .ok_or(DrReplayRefusal::Lowering("the program declares no doubly-robust lowering"))?;
    if lowering.format != CHECKED_DR_LOWERING_FORMAT
        || lowering.procedure != CHECKED_DR_PROCEDURE
        || lowering.population != "all_observed"
    {
        return Err(DrReplayRefusal::Lowering("unknown format, procedure or population"));
    }
    let folds = usize::try_from(lowering.folds)
        .ok()
        .filter(|folds| (2..=MAX_REPLAY_FOLDS).contains(folds))
        .ok_or(DrReplayRefusal::Lowering("the fold count is outside the supported range"))?;
    let overlap = DrOverlap {
        clip: lowering.overlap_clip_bits.map(f64::from_bits),
        trim: lowering.overlap_trim_bits.map(f64::from_bits),
        declared: lowering.overlap_declared,
    };
    overlap
        .validate()
        .map_err(|_| DrReplayRefusal::Lowering("the overlap bounds lie outside (0, 0.5)"))?;
    let unique = lowering.adjustment.iter().copied().collect::<std::collections::BTreeSet<_>>();
    if unique.len() != lowering.adjustment.len()
        || lowering.treatment == lowering.outcome
        || unique.contains(&lowering.treatment)
        || unique.contains(&lowering.outcome)
    {
        return Err(DrReplayRefusal::Lowering("the roles are not distinct variables"));
    }
    Ok(Declaration {
        lowering,
        folds,
        outcome: learner_of(&lowering.outcome_learner)?,
        treatment: learner_of(&lowering.treatment_learner)?,
        final_learner: learner_of(&lowering.final_learner)?,
        overlap,
    })
}

/// The declaration must describe the contract's own query, estimator, estimand and inference
/// choices: the program digest binds the declaration, and these checks bind the declaration
/// back to everything else the contract says.
#[allow(clippy::too_many_lines)]
fn check_binding(
    contract: &AnalysisResultContractWire,
    declaration: &Declaration<'_>,
) -> Result<(), DrReplayRefusal> {
    let lowering = declaration.lowering;
    if !contract_uses_estimator(contract, "dr.learner") {
        return Err(DrReplayRefusal::Binding(
            "the contract does not name the dr.learner estimator",
        ));
    }
    let program = contract
        .program
        .as_ref()
        .ok_or(DrReplayRefusal::Binding("the contract carries no program"))?;
    if program.commitments.inference != "frequentist" || program.commitments.prior_required {
        return Err(DrReplayRefusal::Binding("the program is not a frequentist commitment"));
    }
    let query_matches = match &contract.target.query {
        crate::CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            control,
            active,
            target_population: crate::TargetPopulationWire::AllObserved,
            outcome_functional: crate::query_wire::OutcomeFunctionalWire::Mean,
            effect_modifiers,
        } => {
            *treatment == lowering.treatment
                && *outcome == lowering.outcome
                && effect_modifiers.is_empty()
                && is_set_value(control, lowering.treatment, 0.0)
                && is_set_value(active, lowering.treatment, 1.0)
                && lowering.active_level_above_control
        }
        _ => false,
    };
    if !query_matches {
        return Err(DrReplayRefusal::Binding(
            "the target is not the all-observed mean contrast of the declared 0/1 treatment",
        ));
    }
    let estimand_matches = contract.identification_product.as_ref().is_some_and(|product| {
        product.estimands.iter().any(|estimand| {
            estimand.adjustment_set == lowering.adjustment
                && estimand.instruments.is_empty()
                && estimand.mediators.is_empty()
        })
    });
    if !estimand_matches {
        return Err(DrReplayRefusal::Binding(
            "no identified estimand has the declared adjustment set",
        ));
    }
    if !super::replay::carried_adjustment_is_backdoor(
        contract,
        lowering.treatment,
        lowering.outcome,
        &lowering.adjustment,
    ) {
        return Err(DrReplayRefusal::Binding(
            "the declared adjustment set is not a backdoor set of the carried graph",
        ));
    }
    let binding = contract
        .inference_binding
        .as_ref()
        .ok_or(DrReplayRefusal::Binding("the contract carries no inference binding"))?;
    if binding.inference != "frequentist" || binding.bayesian.is_some() {
        return Err(DrReplayRefusal::Binding("the inference binding is not frequentist"));
    }
    let overlap_tag = overlap_policy_tag(&crate::OverlapPolicyWire::RequireDiagnostics {
        clip_bits: lowering.overlap_clip_bits,
        trim_bits: lowering.overlap_trim_bits,
    });
    let choices_match = match binding.estimator_spec.as_ref() {
        Some(crate::EstimatorSpecWire::DrLearner(config)) => {
            config.folds.map(u64::from) == Some(lowering.folds)
                && config.outcome_config.as_deref() == Some(declaration.outcome.identity().as_str())
                && config.treatment_config.as_deref()
                    == Some(declaration.treatment.identity().as_str())
                && config.final_learner_config.as_deref()
                    == Some(declaration.final_learner.identity().as_str())
                && config.overlap
                    == crate::OverlapPolicyWire::RequireDiagnostics {
                        clip_bits: lowering.overlap_clip_bits,
                        trim_bits: lowering.overlap_trim_bits,
                    }
                && config.cluster_ids.is_none()
                && config.multiway_ids.is_none()
                && config.panel_times.is_none()
                && config.population_registry.is_none()
                && config.sampling_weights.is_none()
                && config.score.is_none()
        }
        Some(crate::EstimatorSpecWire::Default(name)) if name == "dr.learner" => binding
            .overlap_policy
            .as_deref()
            .map_or(!lowering.overlap_declared, |tag| tag == overlap_tag),
        None => binding
            .overlap_policy
            .as_deref()
            .map_or(!lowering.overlap_declared, |tag| tag == overlap_tag),
        Some(_) => false,
    };
    if !choices_match {
        return Err(DrReplayRefusal::Binding(
            "the inference binding's estimator configuration differs from the declaration",
        ));
    }
    Ok(())
}

/// The context the replay runs under: the seed the body records, the kernel and determinism
/// policy the execution identity records, one thread, and the caller's memory bound and
/// cancellation token.
fn replay_context(
    seed: u64,
    contract: &AnalysisResultContractWire,
    bounds: &SourceReplayBounds,
) -> ExecutionContext {
    let mut ctx =
        ExecutionContext::production(seed, 1).with_memory_limit(Some(bounds.memory_bytes));
    ctx.cancellation = bounds.cancellation.clone();
    if let Some(execution) = &contract.execution {
        ctx.kernel_policy = KernelPolicy {
            allow_portable_optimized: execution.kernel.allow_portable_optimized,
            allow_arch_simd: execution.kernel.allow_arch_simd,
            force_scalar: execution.kernel.force_scalar,
        };
        ctx.determinism = if execution.determinism == "prefer_fast" {
            Determinism::PreferFast
        } else {
            Determinism::Strict
        };
    }
    ctx
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the helper retains the error or fixture callback signature"
)]
fn refusal_of(error: DrReplayError) -> DrReplayRefusal {
    match error {
        DrReplayError::Cancelled => DrReplayRefusal::Cancelled,
        DrReplayError::Admission(error) => DrReplayRefusal::Resource(error.to_string()),
        DrReplayError::Unsupported(what) | DrReplayError::Shape(what) => {
            DrReplayRefusal::Unsupported(what)
        }
        DrReplayError::ArmNotPopulated => {
            DrReplayRefusal::Mismatch("a treatment arm is empty in the witnessed table")
        }
        DrReplayError::Overlap(_) => {
            DrReplayRefusal::Mismatch("the overlap handling refuses the fit the artifact reports")
        }
        DrReplayError::Learner(_) => {
            DrReplayRefusal::Mismatch("a learner refuses the fit the artifact reports")
        }
    }
}

/// Replay the declared cross-fitted doubly-robust fit from the witnessed table and compare it
/// with the body.
///
/// The witness is first bound to the contract's data snapshot
/// ([`verify_source_witness`]); the declaration is then read back against the contract; the
/// complete-case columns are the witness columns of the declared variables (every row, since a
/// witness carries only finite, all-valid cells), with the source row index as the row id.
///
/// # Errors
///
/// [`DrReplayRefusal::Cancelled`] when `bounds.cancellation` stopped the work,
/// [`DrReplayRefusal::Resource`] when the memory bound refused it, [`DrReplayRefusal::Source`]
/// for a witness that does not bind to the snapshot, [`DrReplayRefusal::Lowering`],
/// [`DrReplayRefusal::Binding`] and [`DrReplayRefusal::Unsupported`] for a declaration or table
/// the replay does not take, and [`DrReplayRefusal::Mismatch`] when the replay differs from the
/// body.
#[allow(
    clippy::too_many_lines,
    reason = "the validation keeps its related contract checks together"
)]
pub fn verify_checked_dr_replay(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
    witness: &SourceWitnessWire,
    bounds: &SourceReplayBounds,
) -> Result<(), DrReplayRefusal> {
    if bounds.cancellation.is_cancelled() {
        return Err(DrReplayRefusal::Cancelled);
    }
    let declaration = read_declaration(contract)?;
    check_binding(contract, &declaration)?;
    let lowering = declaration.lowering;
    let snapshot = contract
        .data_snapshot
        .as_ref()
        .ok_or(DrReplayRefusal::Binding("the contract carries no data snapshot"))?;
    if snapshot.modality != "tabular"
        || snapshot.missing.is_some()
        || snapshot.participants.is_some()
        || snapshot.interference.is_some()
        || snapshot.sampling_design.is_some()
    {
        return Err(DrReplayRefusal::Unsupported(
            "the snapshot is not a plain tabular table with the default missing-data handling",
        ));
    }
    if body.interval_lower.is_some()
        || body.interval_upper.is_some()
        || body.joint_covariance.is_some()
        || body.cate_leaf_dispersion.is_some()
    {
        return Err(DrReplayRefusal::Unsupported(
            "the body publishes an interval or covariance the replay does not recompute",
        ));
    }
    let published = body
        .estimate
        .filter(|value| value.is_finite())
        .ok_or(DrReplayRefusal::Mismatch("the body publishes no finite point estimate"))?;
    let seed = match (body.crossfit_seed, body.crossfit_folds) {
        (Some(seed), Some(folds)) if u64::try_from(folds) == Ok(lowering.folds) => seed,
        _ => {
            return Err(DrReplayRefusal::Mismatch(
                "the body's cross-fit record differs from the declared fold count",
            ));
        }
    };
    let observation = contract
        .observation
        .as_ref()
        .ok_or(DrReplayRefusal::Binding("the contract carries no observation identity"))?;
    crate::verify_source_witness_schema(witness, &header.variable_names, observation)
        .map_err(DrReplayRefusal::Source)?;
    verify_source_witness(witness, &header.variable_names, snapshot, &bounds.witness)
        .map_err(DrReplayRefusal::Source)?;

    let column = |id: u32| -> Result<&[f64], DrReplayRefusal> {
        usize::try_from(id)
            .ok()
            .and_then(|index| witness.columns.get(index))
            .map(|column| column.values.as_slice())
            .ok_or(DrReplayRefusal::Binding("a declared variable is not a column of the table"))
    };
    let treatment = column(lowering.treatment)?;
    let outcome = column(lowering.outcome)?;
    let covariates: Vec<&[f64]> =
        lowering.adjustment.iter().map(|&id| column(id)).collect::<Result<_, _>>()?;
    #[allow(
        clippy::float_cmp,
        reason = "the declared treatment levels are exactly 0 and 1, so the check is on exact values"
    )]
    let binary = treatment.iter().all(|&value| value == 0.0 || value == 1.0);
    if !binary {
        return Err(DrReplayRefusal::Unsupported("the witnessed treatment is not exactly 0 or 1"));
    }
    let rows = u64::try_from(treatment.len())
        .map_err(|_| DrReplayRefusal::Source(SourceWitnessError::TooLarge))?;
    let row_ids: Vec<u64> = (0..rows).collect();

    let ctx = replay_context(seed, contract, bounds);
    let replay = replay_dr_ate(
        DrReplayInput {
            treatment,
            outcome,
            covariates: &covariates,
            row_ids: &row_ids,
            intercept: lowering.intercept,
        },
        DrReplaySpec {
            folds: declaration.folds,
            seed,
            outcome: declaration.outcome,
            treatment: declaration.treatment,
            final_learner: declaration.final_learner,
            overlap: declaration.overlap,
            active_level_above_control: lowering.active_level_above_control,
        },
        &ctx,
    )
    .map_err(refusal_of)?;

    if replay.ate.to_bits() != published.to_bits() {
        return Err(DrReplayRefusal::Mismatch("the point estimate"));
    }
    let rebuilt = replay
        .fitted_effect(lowering.adjustment.clone())
        .map_err(|_| DrReplayRefusal::Mismatch("the replayed fitted effect is invalid"))?;
    match (rebuilt.as_ref(), body.fitted_effect.as_deref()) {
        (Some(replayed), Some(reported)) if cbor_eq(replayed, reported) => {}
        (None, None) => {}
        _ => return Err(DrReplayRefusal::Mismatch("the fitted effect model")),
    }
    // The analytic standard error is published only when positive and finite; nothing else
    // (no bootstrap, no interval) is published for this route, so presence and bits must agree.
    let replayed_error = Some(replay.standard_error).filter(|se| se.is_finite() && *se > 0.0);
    match (replayed_error, body.standard_error) {
        (Some(replayed), Some(reported)) if replayed.to_bits() == reported.to_bits() => {}
        (None, None) => {}
        _ => return Err(DrReplayRefusal::Mismatch("the standard error")),
    }
    // The nuisance diagnostics are part of what the estimator published, so they replay too.
    let same_bits = |replayed: Option<f64>, reported: Option<f64>| match (replayed, reported) {
        (Some(left), Some(right)) => left.to_bits() == right.to_bits(),
        (None, None) => true,
        _ => false,
    };
    if !same_bits(replay.outcome_oof_r2, body.outcome_oof_r2) {
        return Err(DrReplayRefusal::Mismatch("the outcome out-of-fold R-squared"));
    }
    if !same_bits(replay.treatment_oof_logloss, body.treatment_oof_logloss) {
        return Err(DrReplayRefusal::Mismatch("the treatment out-of-fold log loss"));
    }
    match (replay.cate_se.as_deref(), body.cate_se.as_deref()) {
        (Some(replayed), Some(reported))
            if replayed.len() == reported.len()
                && replayed.iter().zip(reported).all(|(l, r)| l.to_bits() == r.to_bits()) => {}
        (None, None) => {}
        _ => return Err(DrReplayRefusal::Mismatch("the pointwise CATE standard errors")),
    }
    if let Some(reported) = body.cate.as_deref() {
        let same = reported.len() == replay.cate.len()
            && reported
                .iter()
                .zip(&replay.cate)
                .all(|(left, right)| left.to_bits() == right.to_bits());
        if !same {
            return Err(DrReplayRefusal::Mismatch("the CATE vector"));
        }
    }
    Ok(())
}

/// Run the replay when a source is held and the contract declares a lowering: `Ok(true)` when
/// the replay reproduced the body, `Ok(false)` when there is nothing to replay (no source, or
/// no declaration), and the refusal otherwise.
pub(in crate::contract_section) fn replay_if_declared(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
    source: Option<&SourceReplay<'_>>,
) -> Result<bool, DrReplayRefusal> {
    let Some(source) = source else { return Ok(false) };
    if contract.program.as_ref().is_none_or(|program| program.checked_dr_lowering.is_none()) {
        return Ok(false);
    }
    match source.state {
        SourceWitnessState::Unusable(error) => Err(DrReplayRefusal::Source(error.clone())),
        SourceWitnessState::Present(witness) => {
            verify_checked_dr_replay(header, body, contract, witness, source.bounds).map(|()| true)
        }
    }
}
