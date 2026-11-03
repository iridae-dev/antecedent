//! Internal mixture of retained graph-atom executions for one completed table.
//!
//! This is not an authority or a public interval route. Its caller must be a lowered native
//! operation that binds the atom plans, original data, graph weights and resulting receipt.
use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{AssumptionSet, ExecutionContext, IdentificationStatus};
use antecedent_estimate::bayesian::CausalPosterior;
use antecedent_estimate::{EnvelopeOptions, GraphEffectDraws, aggregate_effect_envelope};
use antecedent_prob::{GraphIdentFlag, InferenceDiagnostics, WeightedGraphSamples};

/// Why a graph was not included. Failed and skipped are deliberately not identification flags.
#[derive(Clone, Debug)]
pub(crate) enum AtomOutcome {
    Identified {
        /// Native child effect draws, from this exact precompiled graph operation.
        effect_draws: Arc<[f64]>,
        /// Child model assumptions; merged in graph order into the composite disclosure.
        assumptions: AssumptionSet,
    },
    Unidentified,
    Failed {
        reason: String,
    },
    #[allow(dead_code, reason = "the candidate refuses skipped mass and tests that boundary")]
    Skipped {
        reason: String,
    },
}

/// One exact graph mask and its execution outcome, under the completed-table BIC weights.
#[derive(Clone, Debug)]
pub(crate) struct AtomOutcomeRecord {
    pub(crate) graph_mask: u64,
    pub(crate) weight: f64,
    pub(crate) outcome: AtomOutcome,
}

/// A bounded refusal retains both numerical-failure and skipped masses, with original reasons.
#[derive(Debug)]
#[allow(dead_code, reason = "reasons are asserted through Debug by the mixing oracle tests")]
pub(crate) enum MixError {
    Invalid(&'static str),
    Unavailable { failed_mass: f64, skipped_mass: f64, reasons: Vec<(u64, String)> },
    Cancelled,
    Resource(String),
    Estimation(String),
}

/// The output holds its memory admission lease for as long as its posterior is retained.
pub(crate) struct MixedGraphTable {
    pub(crate) posterior: CausalPosterior,
    _lease: antecedent_core::execution::AdmissionLease,
}

/// Reuse the native envelope's exact moment-matched graph-mixture algorithm. The bounded first
/// cell refuses failed or skipped graphs rather than misclassifying them as unidentified.
/// True non-identification is carried as its own mass by the envelope. The supplied atoms must
/// be native child executions, not caller-constructed distribution claims.
#[allow(
    clippy::too_many_lines,
    reason = "the bounded mixture validates mass and execution outcomes together"
)]
pub(crate) fn mix_completed_table(
    atoms: &[AtomOutcomeRecord],
    ctx: &ExecutionContext,
) -> Result<MixedGraphTable, MixError> {
    if ctx.cancellation.is_cancelled() {
        return Err(MixError::Cancelled);
    }
    if atoms.is_empty() || atoms.len() > 25 {
        return Err(MixError::Invalid("invalid graph count"));
    }
    let mut keys = BTreeSet::new();
    let mut total = 0.0;
    let mut failed_mass = 0.0;
    let mut skipped_mass = 0.0;
    let mut reasons = Vec::new();
    let mut value_count: u64 = 0;
    let mut draw_count = None;
    for atom in atoms {
        if !keys.insert(atom.graph_mask)
            || !atom.weight.is_finite()
            || !(0.0..=1.0).contains(&atom.weight)
        {
            return Err(MixError::Invalid("invalid weight or duplicate graph"));
        }
        total += atom.weight;
        match &atom.outcome {
            AtomOutcome::Identified { effect_draws, .. } => {
                if effect_draws.len() < 2 || effect_draws.iter().any(|v| !v.is_finite()) {
                    return Err(MixError::Invalid("missing or nonfinite effect draws"));
                }
                if draw_count.is_some_and(|n| n != effect_draws.len()) {
                    return Err(MixError::Invalid("different child draw counts"));
                }
                draw_count = Some(effect_draws.len());
                value_count = value_count
                    .checked_add(effect_draws.len() as u64)
                    .ok_or(MixError::Invalid("draw count overflow"))?;
            }
            AtomOutcome::Unidentified => {}
            AtomOutcome::Failed { reason } => {
                if reason.len() > 512 {
                    return Err(MixError::Invalid("failure reason exceeds bound"));
                }
                failed_mass += atom.weight;
                reasons.push((atom.graph_mask, reason.clone()));
            }
            AtomOutcome::Skipped { reason } => {
                if reason.len() > 512 {
                    return Err(MixError::Invalid("skip reason exceeds bound"));
                }
                skipped_mass += atom.weight;
                reasons.push((atom.graph_mask, reason.clone()));
            }
        }
    }
    if (total - 1.0).abs() > 1e-12 {
        return Err(MixError::Invalid("weights are not normalized"));
    }
    if failed_mass > 0.0 || skipped_mass > 0.0 {
        return Err(MixError::Unavailable { failed_mass, skipped_mass, reasons });
    }
    // Account for child draw copies, mixture/sample workspaces, per-atom metadata and output.
    let bytes = value_count
        .checked_mul(8)
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add((atoms.len() as u64).saturating_mul(1024)))
        .and_then(|v| v.checked_add(4096))
        .ok_or(MixError::Invalid("mixture admission overflow"))?;
    let lease = ctx.acquire_admission(0, bytes).map_err(|e| {
        if ctx.cancellation.is_cancelled() {
            MixError::Cancelled
        } else {
            MixError::Resource(e.to_string())
        }
    })?;
    let mut weights = Vec::with_capacity(atoms.len());
    let mut flags = Vec::with_capacity(atoms.len());
    let mut masks = Vec::with_capacity(atoms.len());
    let mut draws = Vec::new();
    let mut assumptions = AssumptionSet::new();
    for atom in atoms {
        if ctx.cancellation.is_cancelled() {
            return Err(MixError::Cancelled);
        }
        weights.push(atom.weight);
        masks.push(atom.graph_mask);
        match &atom.outcome {
            AtomOutcome::Identified { effect_draws, assumptions: child } => {
                flags.push(GraphIdentFlag::Identified);
                draws.push(GraphEffectDraws {
                    graph_key: atom.graph_mask,
                    effect_draws: Arc::clone(effect_draws),
                });
                assumptions.extend_unique(child.entries.iter());
            }
            AtomOutcome::Unidentified => flags.push(GraphIdentFlag::Unidentified),
            // Zero-weight failures are still refused below: no missing native execution is
            // silently made into an identification verdict.
            AtomOutcome::Failed { .. } | AtomOutcome::Skipped { .. } => {
                return Err(MixError::Unavailable { failed_mass, skipped_mass, reasons });
            }
        }
    }
    let graphs = WeightedGraphSamples::new(weights, flags, masks)
        .map_err(|e| MixError::Estimation(e.to_string()))?;
    let mut posterior = aggregate_effect_envelope(
        &graphs,
        &draws,
        InferenceDiagnostics::analytic("graph_imputation.bic_weighted_graph_mixture"),
        EnvelopeOptions::default(),
    )
    .map_err(|e| MixError::Estimation(e.to_string()))?;
    posterior.assumptions.extend_unique(assumptions.entries.iter());
    // The native envelope preserves true unidentified mass and does not promote it to a point.
    if posterior.unidentified_mass > 0.0 {
        posterior.identification = IdentificationStatus::GraphDependent;
    }
    if ctx.cancellation.is_cancelled() {
        return Err(MixError::Cancelled);
    }
    Ok(MixedGraphTable { posterior, _lease: lease })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(mask: u64, weight: f64, base: f64) -> AtomOutcomeRecord {
        let draws: Vec<f64> = (0..100).map(|i| base + 2.0 * f64::from(i % 2)).collect();
        AtomOutcomeRecord {
            graph_mask: mask,
            weight,
            outcome: AtomOutcome::Identified {
                effect_draws: Arc::from(draws),
                assumptions: AssumptionSet::new(),
            },
        }
    }

    #[test]
    fn native_mixture_matches_independent_moments_and_keeps_unidentified_mass() {
        let ctx = ExecutionContext::for_tests(17);
        let atoms = [
            atom(1, 0.6, 0.0),
            atom(2, 0.2, 8.0),
            AtomOutcomeRecord { graph_mask: 3, weight: 0.2, outcome: AtomOutcome::Unidentified },
        ];
        let result = mix_completed_table(&atoms, &ctx).unwrap();
        // Conditional weights are 3/4,1/4: mean 3; population variance 13.
        assert!((result.posterior.summaries.mean[0] - 3.0).abs() < 1e-10);
        assert!((result.posterior.summaries.sd[0] - 13.0f64.sqrt()).abs() < 1e-10);
        assert!((result.posterior.unidentified_mass - 0.2).abs() < 1e-12);
        assert!(result.posterior.unevaluable_mass.abs() < f64::EPSILON);
        assert!(result.posterior.subsampled_out_mass.abs() < f64::EPSILON);
        assert_eq!(result.posterior.identification, IdentificationStatus::GraphDependent);
    }

    #[test]
    fn failures_are_not_relabelled_and_mutations_refuse() {
        let ctx = ExecutionContext::for_tests(17);
        let mut atoms = [
            atom(1, 0.6, 0.0),
            atom(2, 0.2, 8.0),
            AtomOutcomeRecord {
                graph_mask: 3,
                weight: 0.2,
                outcome: AtomOutcome::Failed { reason: "fit failed".into() },
            },
        ];
        match mix_completed_table(&atoms, &ctx) {
            Err(MixError::Unavailable { failed_mass, skipped_mass, reasons }) => {
                assert!((failed_mass - 0.2).abs() < 1e-12);
                assert!(skipped_mass.abs() < f64::EPSILON);
                assert_eq!(reasons, vec![(3, "fit failed".into())]);
            }
            _ => panic!("failed atom must refuse as failed"),
        }
        atoms[2].outcome = AtomOutcome::Skipped { reason: "budget".into() };
        assert!(matches!(mix_completed_table(&atoms, &ctx), Err(MixError::Unavailable { .. })));
        atoms[2].outcome = AtomOutcome::Unidentified;
        atoms[2].graph_mask = 2;
        assert!(matches!(mix_completed_table(&atoms, &ctx), Err(MixError::Invalid(_))));
        atoms[2].graph_mask = 3;
        atoms[2].weight = f64::NAN;
        assert!(matches!(mix_completed_table(&atoms, &ctx), Err(MixError::Invalid(_))));
    }
}
