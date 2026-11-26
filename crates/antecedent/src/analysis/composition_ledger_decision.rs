//! Decision input selected from the active results of a replayed evidence ledger.

use antecedent_contract::ScientificQuantity;
use antecedent_contract::evidence_ledger::ResultId;
use antecedent_core::ExecutionContext;
use antecedent_design::composition_verifiers::consume_ledger_for_bundle;
use antecedent_design::decision_artifact::bound_mean_result_to_bytes;
use antecedent_design::decision_contract::DecisionContract;
use antecedent_design::decision_eval::{DecisionResult, evaluate_contract_on_means};

pub use antecedent_design::ledger_pool_selection::LedgerPoolSelection;

use super::composition::CompositionBundle;
use super::composition::{BundleError, NodeFailure};
use super::composition_cohort_decision::{
    AffineDecisionSource, CompiledAffineDecisionBundleRecomputation, check_mean_requirement,
    compile_affine_decision_bundle, consume_portable_affine_decision_bundle,
};
use super::composition_decision_terminal::{
    DecisionBundleRecomputationReceipt, DecisionBundleRecomputeError,
};
use crate::support::unit::{SupportKey, require_program, require_program_unscoped};
use crate::support::units_generated::COMPOSITION_BUNDLE;
use antecedent_core::SemanticDigest;

pub(super) const COMPILE: &str =
    "antecedent.composition.compile_ledger_pooled_decision_bundle_recomputation";
pub(super) const CONSUME: &str =
    "antecedent.composition.consume_portable_ledger_pooled_decision_bundle_recomputation";
const SELECT: &str = "antecedent.composition.select_ledger_pooled_result";
const PRODUCE: &str = "antecedent.composition.ledger_pooled_decision_bytes";

pub(super) fn node_error(error: NodeFailure) -> DecisionBundleRecomputeError {
    BundleError::Refused { stage: error.stage, reason: error.reason, offending: None }.into()
}

/// Select a live point pooling result from the independently retained ledger.
///
/// The caller declares the quantity, including its population, regime and units.
/// The selected result carries a scoped identity that survives unrelated revisions.
///
/// # Errors
/// Invalid ledger, changed identity, exhausted resources, stale result, missing
/// numerical synthesis, or an assumption range.
pub fn select_ledger_pooled_result(
    ledger_bytes: &[u8],
    expected_ledger: &str,
    result: ResultId,
    quantity: ScientificQuantity,
    ctx: &ExecutionContext,
) -> Result<LedgerPoolSelection, DecisionBundleRecomputeError> {
    require_program(ctx, &SupportKey::Unit(COMPOSITION_BUNDLE), SELECT)?;
    let ledger =
        consume_ledger_for_bundle(ledger_bytes, Some(expected_ledger), ctx).map_err(node_error)?;
    LedgerPoolSelection::select(&ledger, result, quantity).map_err(node_error)
}

pub(super) fn score_selection(
    contract: &DecisionContract,
    selection: &LedgerPoolSelection,
) -> Result<DecisionResult, DecisionBundleRecomputeError> {
    check_mean_requirement(contract)?;
    evaluate_contract_on_means(contract, &selection.mean_source())
        .map_err(|error| DecisionBundleRecomputeError::Evaluation(error.to_string()))
}

/// Export the affine decision bound to its selected pooled result and declared quantity.
///
/// Connect the ledger to the selected result, and the selected result and decision
/// contract to the decision node before compiling its bundle.
///
/// # Errors
/// Invalid contract, unsupported coordinates, law-valued decision, or encoding failure.
pub fn ledger_pooled_decision_bytes(
    contract: &DecisionContract,
    selection: &LedgerPoolSelection,
    artifact_id: &str,
) -> Result<Vec<u8>, DecisionBundleRecomputeError> {
    require_program_unscoped(&SupportKey::Unit(COMPOSITION_BUNDLE), PRODUCE)?;
    let decision = score_selection(contract, selection)?;
    Ok(bound_mean_result_to_bytes(&decision, &selection.identity(), artifact_id)?)
}

/// A compiled ledger → selected point pool → affine decision.
pub type CompiledLedgerPooledDecisionBundleRecomputation =
    CompiledAffineDecisionBundleRecomputation;

/// Compile an affine decision over a ledger pool selected in its retained bundle.
///
/// # Errors
/// Missing, ambiguous or referenced required graph nodes or a law-valued contract.
pub fn compile_ledger_pooled_decision_bundle_recomputation(
    bundle: CompositionBundle,
    terminal: &str,
) -> Result<CompiledLedgerPooledDecisionBundleRecomputation, DecisionBundleRecomputeError> {
    compile_affine_decision_bundle(bundle, terminal, AffineDecisionSource::LedgerPool)
}

/// Rebuild the ledger, its selected result and the re-scored decision from a portable bundle.
///
/// # Errors
/// A changed identity, unreplayable evidence, a retired result or a different decision.
pub fn consume_portable_ledger_pooled_decision_bundle_recomputation(
    bytes: &[u8],
    expected_bundle: &str,
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<DecisionBundleRecomputationReceipt, DecisionBundleRecomputeError> {
    consume_portable_affine_decision_bundle(
        bytes,
        expected_bundle,
        expected_program,
        ctx,
        AffineDecisionSource::LedgerPool,
    )
}
