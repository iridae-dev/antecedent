// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of a class-aware (Cpdag / Pag, explicit or accepted) intervention level.
//!
//! Each evaluated completion is an atom of the identified set. The plugin-level suite
//! runs on every atom's own level against that atom's own adjustment estimand (overlap
//! under `cheap`; overlap plus per-completion sampling stability under `full`), exactly
//! as the graph-posterior cells of these classes do. Reports are mixed into one verdict
//! only when every atom answers the same level; otherwise each atom's reports are kept.
//! A disagreement diagnostic states how far the completions' levels spread. The state
//! executes and is unlicensed.

use super::atom_mix::{ScalarRefuterMix, finish_atom_refutations};
use super::response_path::response_scalar_summary;
use super::{
    AverageEffectQuery, CausalError, CausalResponse, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimationWorkspace, EstimatorId, ExecutionContext,
    OverlapPolicy, RefutationRun, TabularData, VariableId,
};

/// Relative tolerance under which completion levels count as one level.
const SAME_LEVEL_TOLERANCE: f64 = 1e-9;

/// Run the per-completion refuter state of a class-aware intervention level.
///
/// # Errors
///
/// Cancellation or a validator failure.
pub(super) fn refute_class_intervention_atoms<G>(
    study: &super::Study,
    data: &TabularData,
    envelope: &antecedent_identify::IdentificationEnvelope<G>,
    weighted: &[(u64, f64, CausalResponse)],
    (treatment, outcome): (VariableId, VariableId),
    estimator: EstimatorId,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
    let mut workspace = EstimationWorkspace::default();
    let mut per_atom = Vec::new();
    let mut diagnostics = Vec::new();
    let mut levels = Vec::new();
    for (key, weight, response) in weighted {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let Some(case) = usize::try_from(*key).ok().and_then(|index| envelope.cases.get(index))
        else {
            continue;
        };
        let Some(estimand) = case.result.estimands.first() else { continue };
        let (level, se) = response_scalar_summary(response);
        if !level.is_finite() || *weight <= 0.0 {
            continue;
        }
        levels.push(level);
        let estimate = EffectEstimate::new(
            level,
            se,
            case.result.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let (reports, notes) = RefutationRun {
            data,
            estimand,
            query: &ate_query,
            estimate: &estimate,
            suite: study.refute,
            estimator: estimator.as_str(),
            custom: &[],
            temporal: None,
        }
        .plugin_level(&mut workspace, ctx)?;
        diagnostics.extend(notes);
        per_atom.push((*weight, reports));
    }
    let lower = levels.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let spread = if levels.is_empty() { f64::NAN } else { upper - lower };
    let same_level = spread.is_finite()
        && spread <= SAME_LEVEL_TOLERANCE * lower.abs().max(upper.abs()).max(1.0);
    diagnostics.push(
        Diagnostic::new(
            "refute.response.class_envelope_disagreement",
            DiagnosticKind::Scientific,
            if same_level { DiagnosticSeverity::Info } else { DiagnosticSeverity::Warning },
            format!(
                "{} evaluated completion(s); completion levels span [{lower}, {upper}] \
                 (spread {spread}). {}",
                levels.len(),
                if same_level {
                    "Every completion answers the same level, so the per-completion refuter \
                     reports are mixed by completion weight."
                } else {
                    "The completions disagree, so each completion's refuter reports are kept \
                     on their own; no mixed verdict is formed over an identified set."
                }
            ),
        )
        .with_fields([
            ("completions", levels.len().to_string()),
            ("lower", lower.to_string()),
            ("upper", upper.to_string()),
        ]),
    );
    diagnostics.push(Diagnostic::new(
        "refute.response.class_state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "class-aware intervention-level refuter state: plugin-level overlap (cheap) and \
         overlap plus per-completion sampling stability (full) on each completion's own \
         level and estimand. It executes unlicensed: no coverage or evidence record names it, \
         and the verdicts are checks, not a calibration claim.",
    ));
    Ok(finish_atom_refutations(
        per_atom,
        diagnostics,
        same_level,
        ScalarRefuterMix::Effect,
        None,
        false,
    ))
}
