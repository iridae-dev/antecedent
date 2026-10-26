//! Selective execution of the fixed-population temporal row.
//!
//! The one-shot [`super::evaluate_temporal_counterfactual`] abduces and
//! predicts in one call. [`TemporalExecutor`] runs the same arithmetic as
//! separate stages and retains each factual history's abduction under the
//! digest of the snapshot, graph, mechanism fit and factual histories. A
//! request that changes only the named action histories reuses the abduction
//! (and the per-unit exogenous-draw digests) and replays the new worlds; a
//! changed snapshot, graph, fit coefficient or factual value is a different
//! key and abduces again. Results are bit-identical to the one-shot function.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::ExecutionContext;

use super::{
    MAX_HORIZON, MAX_WORLDS, MECHANISM_CLASS, POLL_EVERY, SharedAbductionReceipt,
    TemporalCounterfactualError, TemporalCounterfactualResult, TemporalCounterfactualSpec,
    UnitCounterfactual, UnitDrawDigest, abduce_unit, check_inputs, compile_fit, digest,
    digest_draw, digest_factual, digest_fit, digest_graph, digest_history, poll, replay,
    validate_graph,
};
use crate::selective::stages::{ExecutionStage, StageReceipt, StageStatus, WorkCounters};
use crate::temporal_cross_world::FactualUnitHistory;

/// Most abductions a temporal executor retains.
pub const MAX_RETAINED: usize = 64;

#[derive(Debug)]
struct RetainedHistories {
    noise: Vec<[f64; 3]>,
    unit_draws: Vec<UnitDrawDigest>,
}

/// One staged execution.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalExecution {
    /// The answer, bit-identical to the one-shot evaluator.
    pub result: TemporalCounterfactualResult,
    /// Stage identities and statuses.
    pub stages: Vec<StageReceipt>,
    /// Work observed in this execution.
    pub work: WorkCounters,
}

/// A selective executor of the temporal row.
#[derive(Debug, Default)]
pub struct TemporalExecutor {
    retained: BTreeMap<String, Arc<RetainedHistories>>,
    counters: WorkCounters,
}

fn abduction_key(
    spec: &TemporalCounterfactualSpec,
    graph: &str,
    fit: &str,
    factual: &str,
) -> String {
    let mut w = digest("selective_abduction");
    w.text(spec.snapshot.as_str()).text(graph).text(fit).text(factual);
    w.finish_hex()
}

impl TemporalExecutor {
    /// An executor with nothing retained.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cumulative observed work of successful executions.
    #[must_use]
    pub fn counters(&self) -> WorkCounters {
        self.counters
    }

    /// Number of retained abductions.
    #[must_use]
    pub fn retained_len(&self) -> usize {
        self.retained.len()
    }

    /// Execute a request, reusing a retained abduction whose key matches.
    ///
    /// All scientific validation of the request (graph, fit, pairing, times)
    /// runs first and is identical to the one-shot evaluator's; a failure
    /// leaves the retained state unchanged.
    ///
    /// # Errors
    /// The one-shot evaluator's refusals, or cancellation.
    #[allow(
        clippy::too_many_lines,
        reason = "one staged execution whose checks, retention and outcome assembly share the same run state"
    )]
    pub fn execute(
        &mut self,
        spec: &TemporalCounterfactualSpec,
        ctx: &ExecutionContext,
    ) -> Result<TemporalExecution, TemporalCounterfactualError> {
        poll(ctx)?;
        validate_graph(&spec.graph)?;
        let mechs = compile_fit(&spec.graph, &spec.fit)?;
        let mut factual: Vec<&FactualUnitHistory> = spec.factual.iter().collect();
        factual.sort_by(|a, b| a.unit.cmp(&b.unit));
        check_inputs(spec, &factual)?;

        let (graph_digest, fit_digest, factual_digest) =
            (digest_graph(&spec.graph), digest_fit(&spec.fit), digest_factual(&factual));
        let key = abduction_key(spec, &graph_digest, &fit_digest, &factual_digest);
        let mut counters = self.counters;
        let before = counters;
        let (retained, reused) = if let Some(found) = self.retained.get(&key) {
            counters.abductions_reused += factual.len() as u64;
            (Arc::clone(found), true)
        } else {
            let mut noise = Vec::with_capacity(factual.len());
            let mut unit_draws = Vec::with_capacity(factual.len());
            for (index, unit) in factual.iter().enumerate() {
                if index % POLL_EVERY == 0 {
                    poll(ctx)?;
                }
                let draw = abduce_unit(&mechs, unit)?;
                counters.abductions += 1;
                unit_draws.push(UnitDrawDigest {
                    unit: unit.unit.clone(),
                    history: unit.history.clone(),
                    digest: digest_draw(&spec.snapshot, unit, &draw),
                });
                noise.push(draw);
            }
            (Arc::new(RetainedHistories { noise, unit_draws }), false)
        };

        let mut units = Vec::with_capacity(factual.len());
        let (mut sum_plus, mut sum_minus) = (0.0, 0.0);
        for (index, unit) in factual.iter().enumerate() {
            if index % POLL_EVERY == 0 {
                poll(ctx)?;
            }
            let noise = &retained.noise[index];
            let plus_outcome = replay(&mechs, noise, &spec.plus.actions);
            let minus_outcome = replay(&mechs, noise, &spec.minus.actions);
            counters.world_predictions += 2;
            sum_plus += plus_outcome;
            sum_minus += minus_outcome;
            units.push(UnitCounterfactual {
                unit: unit.unit.clone(),
                history: unit.history.clone(),
                factual_outcome: unit.values[super::TemporalNode::Outcome.index()],
                plus_outcome,
                minus_outcome,
            });
        }
        poll(ctx)?;
        counters.summaries += 1;
        let n = units.len() as f64;
        let unit_draws = retained.unit_draws.clone();
        let mut receipt = SharedAbductionReceipt {
            snapshot: spec.snapshot.clone(),
            graph_digest,
            fit_digest: fit_digest.clone(),
            factual_digest,
            plus_digest: digest_history(&spec.plus),
            minus_digest: digest_history(&spec.minus),
            exogenous_digest: SharedAbductionReceipt::exogenous_of(&unit_draws),
            unit_draws,
            n_units: units.len(),
            n_worlds: MAX_WORLDS,
            horizon: MAX_HORIZON,
            shared_by_both_worlds: true,
            digest: String::new(),
        };
        receipt.digest = receipt.recompute_digest();
        let result = TemporalCounterfactualResult {
            mechanism_class: MECHANISM_CLASS,
            units,
            mean_plus: sum_plus / n,
            mean_minus: sum_minus / n,
            mean_contrast: (sum_plus - sum_minus) / n,
            receipt,
        };
        let status = if reused { StageStatus::Reused } else { StageStatus::Computed };
        let mut prediction = digest("selective_prediction");
        prediction
            .text(&result.receipt.exogenous_digest)
            .text(&result.receipt.plus_digest)
            .text(&result.receipt.minus_digest);
        let stage = |stage, identity: &str, status| StageReceipt {
            stage,
            identity: identity.to_owned(),
            status,
        };
        let stages = vec![
            stage(ExecutionStage::MechanismFitting, &fit_digest, StageStatus::Declared),
            stage(ExecutionStage::FactualAbduction, &key, status),
            stage(ExecutionStage::ExogenousDraws, &result.receipt.exogenous_digest, status),
            stage(
                ExecutionStage::ActionPrediction,
                &prediction.finish_hex(),
                StageStatus::Computed,
            ),
            stage(ExecutionStage::SummaryDecision, &result.receipt.digest, StageStatus::Computed),
        ];
        // Commit only after every stage succeeded.
        if !reused {
            self.retained.insert(key, retained);
            while self.retained.len() > MAX_RETAINED {
                if let Some(first) = self.retained.keys().next().cloned() {
                    self.retained.remove(&first);
                }
            }
        }
        self.counters = counters;
        Ok(TemporalExecution { result, stages, work: counters.since(before) })
    }
}

use super::general::{
    GeneralAbductionReceipt, GeneralFactualHistory, GeneralTemporalResult, GeneralTemporalSpec,
    GeneralUnitCounterfactual, abduce_unit as abduce_general, check_fit, check_inputs as
    check_general_inputs, digest_draw as digest_general_draw, digest_factual as
    digest_general_factual, digest_fit as digest_general_fit, digest_world, node_count,
    outcome_node, replay as replay_general,
};
use super::{invalid, unsupported};

#[derive(Debug)]
struct RetainedGeneral {
    /// Abduced residuals, `generating nodes` per unit, in unit-id order.
    noise: Vec<f64>,
    width: usize,
    unit_draws: Vec<UnitDrawDigest>,
}

/// One staged execution of the general temporal row.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralTemporalExecution {
    /// The answer, bit-identical to the one-shot general evaluator.
    pub result: GeneralTemporalResult,
    /// Stage identities and statuses.
    pub stages: Vec<StageReceipt>,
    /// Work observed in this execution.
    pub work: WorkCounters,
}

/// A selective executor of the general temporal row (any horizon, any number of worlds).
///
/// Retains each factual set's abduction under the digest of the snapshot, the horizon and
/// mechanism fit and the factual histories; a request that changes only the named action
/// histories reuses it and replays the new worlds.
#[derive(Debug, Default)]
pub struct GeneralTemporalExecutor {
    retained: BTreeMap<String, Arc<RetainedGeneral>>,
    counters: WorkCounters,
}

fn general_key(snapshot: &str, fit: &str, factual: &str) -> String {
    let mut w = digest("selective_general_abduction");
    w.text(snapshot).text(fit).text(factual);
    w.finish_hex()
}

impl GeneralTemporalExecutor {
    /// An executor with nothing retained.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cumulative observed work of successful executions.
    #[must_use]
    pub fn counters(&self) -> WorkCounters {
        self.counters
    }

    /// Number of retained abductions.
    #[must_use]
    pub fn retained_len(&self) -> usize {
        self.retained.len()
    }

    /// Execute a request, reusing a retained abduction whose key matches. Validation is the
    /// one-shot evaluator's and runs first; a failure leaves the retained state unchanged.
    ///
    /// # Errors
    /// The one-shot general evaluator's refusals, or cancellation.
    #[allow(clippy::too_many_lines, reason = "one staged execution sharing one run state")]
    pub fn execute(
        &mut self,
        spec: &GeneralTemporalSpec,
        ctx: &ExecutionContext,
    ) -> Result<GeneralTemporalExecution, TemporalCounterfactualError> {
        poll(ctx)?;
        if spec.horizon == 0 {
            return Err(invalid("temporal_counterfactual.invalid_graph", Some("horizon".to_owned())));
        }
        if spec.latent_confounding {
            return Err(unsupported("temporal_counterfactual.latent_confounding", None, None));
        }
        let mechs = check_fit(spec.horizon, &spec.fit)?;
        let mut factual: Vec<&GeneralFactualHistory> = spec.factual.iter().collect();
        factual.sort_by(|a, b| a.unit.cmp(&b.unit));
        check_general_inputs(spec, &factual)?;

        let fit_digest = digest_general_fit(spec.horizon, &spec.fit);
        let factual_digest = digest_general_factual(&factual);
        let key = general_key(spec.snapshot.as_str(), &fit_digest, &factual_digest);
        let mut counters = self.counters;
        let before = counters;
        let width = mechs.len();
        let (retained, reused) = if let Some(found) = self.retained.get(&key) {
            counters.abductions_reused += factual.len() as u64;
            (Arc::clone(found), true)
        } else {
            let mut noise = vec![0.0; factual.len() * width];
            let mut unit_draws = Vec::with_capacity(factual.len());
            for (index, unit) in factual.iter().enumerate() {
                if index % POLL_EVERY == 0 {
                    poll(ctx)?;
                }
                let slot = &mut noise[index * width..(index + 1) * width];
                abduce_general(&mechs, unit, slot)?;
                counters.abductions += 1;
                unit_draws.push(UnitDrawDigest {
                    unit: unit.unit.clone(),
                    history: unit.history.clone(),
                    digest: digest_general_draw(&spec.snapshot, unit, slot),
                });
            }
            (Arc::new(RetainedGeneral { noise, width, unit_draws }), false)
        };

        let n_worlds = spec.worlds.len();
        let mut scratch = vec![0.0; node_count(spec.horizon)];
        let mut sums = vec![0.0; n_worlds];
        let mut units = Vec::with_capacity(factual.len());
        for (index, unit) in factual.iter().enumerate() {
            if index % POLL_EVERY == 0 {
                poll(ctx)?;
            }
            let noise = &retained.noise[index * retained.width..(index + 1) * retained.width];
            let mut outcomes = Vec::with_capacity(n_worlds);
            for (sum, world) in sums.iter_mut().zip(&spec.worlds) {
                let outcome = replay_general(&mechs, noise, &world.actions, &mut scratch);
                *sum += outcome;
                outcomes.push(outcome);
            }
            counters.world_predictions += n_worlds as u64;
            units.push(GeneralUnitCounterfactual {
                unit: unit.unit.clone(),
                history: unit.history.clone(),
                factual_outcome: unit.values[outcome_node(spec.horizon)],
                outcomes,
            });
        }
        poll(ctx)?;
        counters.summaries += 1;
        let n = units.len() as f64;
        let unit_draws = retained.unit_draws.clone();
        let mut receipt = GeneralAbductionReceipt {
            snapshot: spec.snapshot.clone(),
            fit_digest: fit_digest.clone(),
            factual_digest,
            world_digests: spec.worlds.iter().map(digest_world).collect(),
            exogenous_digest: GeneralAbductionReceipt::exogenous_of(&unit_draws),
            unit_draws,
            n_units: units.len(),
            n_worlds,
            horizon: spec.horizon,
            shared_by_all_worlds: true,
            digest: String::new(),
        };
        receipt.digest = receipt.recompute_digest();
        let result = GeneralTemporalResult {
            mechanism_class: MECHANISM_CLASS,
            world_names: spec.worlds.iter().map(|w| w.name.clone()).collect(),
            units,
            world_means: sums.iter().map(|s| s / n).collect(),
            receipt,
        };
        let status = if reused { StageStatus::Reused } else { StageStatus::Computed };
        let mut prediction = digest("selective_general_prediction");
        prediction.text(&result.receipt.exogenous_digest);
        for world in &result.receipt.world_digests {
            prediction.text(world);
        }
        let stage = |stage, identity: &str, status| StageReceipt {
            stage,
            identity: identity.to_owned(),
            status,
        };
        let stages = vec![
            stage(ExecutionStage::MechanismFitting, &fit_digest, StageStatus::Declared),
            stage(ExecutionStage::FactualAbduction, &key, status),
            stage(ExecutionStage::ExogenousDraws, &result.receipt.exogenous_digest, status),
            stage(
                ExecutionStage::ActionPrediction,
                &prediction.finish_hex(),
                StageStatus::Computed,
            ),
            stage(ExecutionStage::SummaryDecision, &result.receipt.digest, StageStatus::Computed),
        ];
        if !reused {
            self.retained.insert(key, retained);
            while self.retained.len() > MAX_RETAINED {
                if let Some(first) = self.retained.keys().next().cloned() {
                    self.retained.remove(&first);
                }
            }
        }
        self.counters = counters;
        Ok(GeneralTemporalExecution { result, stages, work: counters.since(before) })
    }
}

#[cfg(test)]
mod general_tests {
    use super::*;
    use crate::temporal_cross_world::general::{
        GeneralActionHistory, GeneralMechanismFit, GeneralNodeMechanism, covariate_node,
        evaluate_general_temporal_counterfactual,
    };
    use crate::temporal_cross_world::{HistoryId, SnapshotId, UnitId};

    fn spec(horizon: usize, worlds: &[Vec<f64>]) -> GeneralTemporalSpec {
        let ids: Vec<UnitId> = (0..4).map(|i| UnitId::new(format!("u{i}"))).collect();
        let nodes = node_count(horizon);
        let mut mechanisms = Vec::new();
        for slice in 0..horizon {
            let node = covariate_node(slice);
            let parents = if node == 0 { vec![] } else { vec![(node - 1, 0.5)] };
            mechanisms.push(GeneralNodeMechanism {
                node,
                intercept: 1.0,
                parent_coefficients: parents,
                noise_halfwidth: None,
            });
        }
        mechanisms.push(GeneralNodeMechanism {
            node: nodes - 1,
            intercept: 0.0,
            parent_coefficients: vec![(nodes - 2, 2.0)],
            noise_halfwidth: None,
        });
        GeneralTemporalSpec {
            horizon,
            latent_confounding: false,
            fit: GeneralMechanismFit { fit_id: "fit".into(), mechanisms },
            snapshot: SnapshotId::new("snap"),
            factual: ids
                .iter()
                .enumerate()
                .map(|(i, u)| GeneralFactualHistory {
                    unit: u.clone(),
                    history: HistoryId::new(format!("h{i}")),
                    times: (0..horizon as u32).collect(),
                    values: (0..nodes).map(|k| ((i + 1) * (k + 2)) as f64 * 0.1).collect(),
                })
                .collect(),
            worlds: worlds
                .iter()
                .enumerate()
                .map(|(w, a)| GeneralActionHistory {
                    name: format!("w{w}"),
                    times: (0..horizon as u32).collect(),
                    actions: a.clone(),
                    units: ids.clone(),
                })
                .collect(),
        }
    }

    #[test]
    fn staged_general_execution_matches_the_one_shot_and_reuses_abduction() {
        let ctx = ExecutionContext::for_tests(3);
        let first = spec(4, &[vec![0.0; 4], vec![1.0; 4], vec![1.0, 0.0, 1.0, 0.0]]);
        let one_shot = evaluate_general_temporal_counterfactual(&first, &ctx).unwrap();
        let mut exec = GeneralTemporalExecutor::new();
        let a = exec.execute(&first, &ctx).unwrap();
        assert_eq!(a.result, one_shot);
        assert_eq!(a.work.abductions, 4);
        assert_eq!(a.stages[1].status, StageStatus::Computed);
        // Only the worlds change: the abduction is reused.
        let second = spec(4, &[vec![1.0; 4], vec![0.5; 4]]);
        let b = exec.execute(&second, &ctx).unwrap();
        assert_eq!(b.result, evaluate_general_temporal_counterfactual(&second, &ctx).unwrap());
        assert_eq!((b.work.abductions, b.work.abductions_reused), (0, 4));
        assert_eq!(b.stages[1].status, StageStatus::Reused);
        assert_eq!(exec.retained_len(), 1);
    }
}
