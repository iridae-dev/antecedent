//! Seal evaluated rankings and their retained signal laws.

use std::collections::BTreeMap;

use antecedent_contract::ProviderObjectIdentity;
use antecedent_io_core::quantity_wire::ScientificQuantityWire;

use crate::decision::{DecisionProblem, DecisionTable};
use crate::error::DesignError;
use crate::evsi::{EvsiCandidate, EvsiCandidateRow, EvsiReport, EvsiRequest};
use crate::preposterior::{DecisionPrior, DecisionSignal};
use crate::ranking::RankingRecord;
use crate::signal::{
    PreparedLaw, PreparedSignal, SignalReceipt, SignalRequest, SignalUpdateMode, recall_prepared,
};

use super::{
    CostMapWire, DESIGN_RANKING_ARTIFACT_KIND, DESIGN_RANKING_ARTIFACT_VERSION,
    DESIGN_RANKING_CALIBRATION, DecisionWire, DesignRankingArtifactError,
    DesignRankingArtifactWire, EvsiCandidateWire, LawWire, PriorWire, ProviderWire, RankConfigWire,
    Result, SearchWire, SignalRequestWire, UtilityWire,
};

/// Inputs of a ranking evaluation, retained by [`seal`].
pub struct SealInputs<'a, A> {
    /// Decision problem the EVSI values information for.
    pub problem: &'a DecisionProblem<A, f64>,
    /// Belief the decision is made under.
    pub prior: &'a DecisionPrior<f64>,
    /// The evaluated request (candidates and their providers).
    pub request: &'a EvsiRequest,
    /// The evaluation result.
    pub report: &'a EvsiReport,
    /// Digests of the source distributions (prior sources and external laws).
    pub source_digests: &'a [String],
}

fn decision_wire<A: Clone>(inputs: &SealInputs<'_, A>) -> Result<DecisionWire> {
    let problem = inputs.problem;
    let (prior, utility, admissible) = match inputs.prior {
        DecisionPrior::Draws(states) => {
            let table = DecisionTable::build(problem, states)?;
            let rows: Vec<Vec<f64>> = (0..problem.actions.len())
                .map(|a| (0..table.n_outcomes).map(|k| table.utility(a, k)).collect())
                .collect();
            let admissible =
                (0..problem.actions.len()).map(|a| table.admissible.contains(&a)).collect();
            (PriorWire::Draws { states: states.clone() }, UtilityWire::Table { rows }, admissible)
        }
        DecisionPrior::Normal { mean, variance } => {
            let coefficients = problem
                .utility
                .affine_in_state(&problem.actions)
                .filter(|c| c.len() == problem.actions.len())
                .ok_or_else(|| {
                    DesignError::Config(
                        "a normal belief needs a utility that declares affine coefficients".into(),
                    )
                })?;
            (
                PriorWire::Normal { mean: *mean, variance: *variance },
                UtilityWire::Affine { coefficients },
                vec![true; problem.actions.len()],
            )
        }
    };
    Ok(DecisionWire {
        contract_identity: inputs.request.decision_contract_identity.clone(),
        utility_unit: inputs.request.utility_unit.clone(),
        action_ids: inputs.request.action_ids.clone(),
        prior,
        utility,
        admissible,
    })
}

fn likelihood_law(
    family: &str,
    mode: SignalUpdateMode,
    signal: &dyn DecisionSignal<f64>,
    states: Option<&[f64]>,
    n: u64,
) -> Result<LawWire> {
    let mismatch = || DesignRankingArtifactError::InvalidWire("signal law is not retainable");
    match family {
        "gaussian_mean" => Ok(LawWire::GaussianMean {
            noise_variance: signal.gaussian_noise_variance().ok_or_else(mismatch)?,
        }),
        "binomial" => Ok(LawWire::Binomial),
        _ => {
            let states = states.ok_or_else(mismatch)?;
            let statistics = signal.finite_support(n).ok_or_else(mismatch)?;
            let mut table = Vec::with_capacity(statistics.len());
            for &y in &statistics {
                let mut log_lik = vec![0.0; states.len()];
                signal.log_likelihood(y, n, states, &mut log_lik)?;
                table.push(log_lik.iter().map(|ll| ll.exp()).collect::<Vec<f64>>());
            }
            if mode == SignalUpdateMode::ExternalPosterior {
                let k = states.len() as f64;
                let predictive: Vec<f64> =
                    table.iter().map(|row| row.iter().sum::<f64>() / k).collect();
                let posterior = table
                    .iter()
                    .zip(&predictive)
                    .map(|(row, p)| {
                        row.iter().map(|l| if *p > 0.0 { l / (k * p) } else { 1.0 / k }).collect()
                    })
                    .collect();
                Ok(LawWire::FinitePosterior { statistics, predictive, posterior })
            } else {
                Ok(LawWire::FiniteLikelihood { statistics, probabilities: table })
            }
        }
    }
}

pub(super) fn request_wire(request: &SignalRequest) -> SignalRequestWire {
    let mut lineage = request.evidence_lineage.clone();
    lineage.sort();
    SignalRequestWire {
        candidate_id: request.candidate_id.clone(),
        prior_id: request.prior_id.clone(),
        state_quantity: ScientificQuantityWire::from(&request.state_quantity),
        observation_quantity: ScientificQuantityWire::from(&request.observation_quantity),
        rng_seed: request.rng_seed,
        evidence_lineage: lineage,
        conditional_independence: request.conditional_independence.clone(),
        max_sample_size: request.limits.max_sample_size,
        max_support: request.limits.max_support as u64,
    }
}

pub(super) fn provider_wire(identity: &ProviderObjectIdentity) -> ProviderWire {
    ProviderWire {
        provider_id: identity.provider_id.clone(),
        object_id: identity.object_id.clone(),
        version_id: identity.version_id.clone(),
        snapshot_id: identity.snapshot_id.clone(),
        request_id: identity.request_id.clone(),
    }
}

fn law_wire(
    receipt: &SignalReceipt,
    prepared: &PreparedSignal,
    sample_size: u64,
) -> Result<LawWire> {
    match &prepared.law {
        PreparedLaw::Likelihood(signal) => likelihood_law(
            &receipt.family,
            receipt.update_mode,
            signal.as_ref(),
            prepared.bound_states.as_deref(),
            sample_size,
        ),
        PreparedLaw::DecisionValues(values) => Ok(LawWire::DecisionValues {
            branch_probabilities: values.branch_probabilities.clone(),
            action_ids: values.action_ids.clone(),
            values: values.values.clone(),
        }),
    }
}

/// The law from evaluation when its receipt is still on the row; otherwise a
/// fresh prepare that must answer with that same receipt identity.
fn prepared_for_seal(
    declared: &EvsiCandidate,
    row: &EvsiCandidateRow,
    laws: Option<&[PreparedSignal]>,
) -> Result<PreparedSignal> {
    if let Some(laws) = laws {
        if declared.signal_request.fingerprint() != row.signal_receipt.request_fingerprint {
            return Err(DesignRankingArtifactError::SignalInconsistent {
                candidate: row.semantic_id.clone(),
                slot: "evaluated_request_changed",
            });
        }
        return laws.iter().find(|law| law.receipt == row.signal_receipt).cloned().ok_or_else(
            || DesignRankingArtifactError::SignalInconsistent {
                candidate: row.semantic_id.clone(),
                slot: "evaluated_law_missing",
            },
        );
    }
    if let Some(retained) = recall_prepared(&row.signal_receipt.identity())
        .filter(|retained| retained.receipt == row.signal_receipt)
    {
        // Debug and test builds compare the retained law with a fresh prepare.
        // The wires match, so the artifact bytes do not depend on which one is
        // used. Release keeps the retained law and does not prepare again.
        #[cfg(debug_assertions)]
        {
            let again = declared.provider.prepare(&declared.signal_request)?;
            assert_eq!(
                law_wire(&row.signal_receipt, &retained, row.sample_size)?,
                law_wire(&row.signal_receipt, &again, row.sample_size)?,
                "retained signal law changed the ranking artifact"
            );
        }
        return Ok(retained);
    }
    let prepared = declared.provider.prepare(&declared.signal_request)?;
    if prepared.receipt.identity() != row.signal_receipt.identity() {
        return Err(DesignRankingArtifactError::SignalInconsistent {
            candidate: row.semantic_id.clone(),
            slot: "provider_not_reproducible",
        });
    }
    Ok(prepared)
}

/// Seal an evaluated ranking as a durable artifact body.
///
/// The law prepared while evaluating the row is reused when its receipt is still
/// the one on the row. Providers are deterministic functions of the exact request,
/// so a miss prepares again and still requires that receipt identity. Either way
/// the law wire is the one [`law_wire`] builds from that receipt.
///
/// # Errors
///
/// A provider that no longer answers its request identically, a law that cannot be
/// retained, a duplicated declared semantic id, or an empty report.
pub fn seal<A: Clone>(inputs: &SealInputs<'_, A>) -> Result<DesignRankingArtifactWire> {
    seal_with_laws(inputs, None, None)
}

/// Seal the exact laws of a contextual evaluation under caller cancellation.
/// This path never consults the process memo or re-prepares a provider.
///
/// # Errors
/// As [`seal`], or [`DesignError::Cancelled`] when the caller cancels publication.
pub fn seal_with_context<A: Clone>(
    inputs: &SealInputs<'_, A>,
    evaluation: &crate::evsi::EvaluatedEvsi,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<DesignRankingArtifactWire> {
    if inputs.report != evaluation.report() {
        return Err(DesignRankingArtifactError::InvalidWire("evaluation report differs"));
    }
    seal_with_laws(inputs, Some(&evaluation.laws), Some(&ctx.cancellation))
}

fn seal_with_laws<A: Clone>(
    inputs: &SealInputs<'_, A>,
    laws: Option<&[PreparedSignal]>,
    cancellation: Option<&antecedent_core::CancellationToken>,
) -> Result<DesignRankingArtifactWire> {
    if cancellation.is_some_and(antecedent_core::CancellationToken::is_cancelled) {
        return Err(DesignError::Cancelled.into());
    }
    let report = inputs.report;
    let ranking = RankingRecord::from_evsi(report, inputs.source_digests)?;
    let decision = decision_wire(inputs)?;
    let mut declared_by_id = BTreeMap::new();
    for candidate in &inputs.request.candidates {
        // A later declaration with the same id is a different provider. Keeping the
        // first would seal that provider's law under the shared id.
        if declared_by_id.insert(candidate.semantic_id.as_str(), candidate).is_some() {
            return Err(DesignRankingArtifactError::InvalidWire("duplicate candidate"));
        }
    }
    let mut candidates = Vec::with_capacity(report.candidates.len());
    for row in &report.candidates {
        let declared = declared_by_id
            .get(row.semantic_id.as_str())
            .copied()
            .ok_or(DesignRankingArtifactError::InvalidWire("report names an unknown candidate"))?;
        if cancellation.is_some_and(antecedent_core::CancellationToken::is_cancelled) {
            return Err(DesignError::Cancelled.into());
        }
        let prepared = prepared_for_seal(declared, row, laws)?;
        let law = law_wire(&row.signal_receipt, &prepared, row.sample_size)?;
        candidates.push(EvsiCandidateWire::from_scored(row, declared, law));
    }
    let config = &inputs.request.rank_config;
    let mut wire = DesignRankingArtifactWire {
        version: DESIGN_RANKING_ARTIFACT_VERSION,
        kind: DESIGN_RANKING_ARTIFACT_KIND.to_owned(),
        calibration: DESIGN_RANKING_CALIBRATION.to_owned(),
        basis: ranking.basis.as_str().to_owned(),
        decision,
        cost_map: report.cost_map.as_ref().map(|m| CostMapWire {
            cost_unit: m.cost_unit.clone(),
            utility_unit: m.utility_unit.clone(),
            utility_per_cost: m.utility_per_cost,
        }),
        rng_seed: report.rng_seed,
        mc_error_tolerance: inputs.request.mc_error_tolerance,
        tie_tolerance: inputs.request.tie_tolerance,
        max_candidates: inputs.request.max_candidates as u64,
        rank_config: RankConfigWire {
            min_batches: config.min_batches,
            max_batches: config.max_batches,
            batch_size: config.batch_size,
            rank_uncertainty_threshold: config.rank_uncertainty_threshold,
        },
        prior_observation_ids: inputs.request.prior_observation_ids.clone(),
        source_digests: ranking.source_digests.clone(),
        bayes_action: report.bayes_action as u64,
        prior_expected_utility: report.prior_expected_utility,
        evpi: report.evpi,
        candidates,
        ties: report.ties.clone(),
        search: SearchWire {
            supplied: report.search.supplied as u64,
            evaluated: report.search.evaluated as u64,
            truncated: report.search.truncated,
            unevaluated_ids: report.search.unevaluated_ids.clone(),
        },
        ranking_identity: ranking.identity(),
        digest: String::new(),
    };
    wire.reseal()?;
    if cancellation.is_some_and(antecedent_core::CancellationToken::is_cancelled) {
        return Err(DesignError::Cancelled.into());
    }
    Ok(wire)
}
