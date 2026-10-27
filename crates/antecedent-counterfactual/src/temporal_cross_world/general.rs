//! Shared-abduction temporal counterfactual for any horizon and any number of worlds.
//!
//! The fixed two-slice route of the parent module is the special case `horizon = 2` with the
//! two named histories `plus` and `minus`. This executor drops both limits: the graph is the
//! fixed, fully observed, Markovian chain of slices
//! `L0 -> A0 -> L1 -> A1 -> ... -> L(k-1) -> A(k-1) -> Y` (a covariate and an action per
//! slice, then the final outcome), every non-action node has an additive-noise linear
//! mechanism `V = intercept + sum_p coef_p * parent_p + U_V` over any earlier nodes, each unit's
//! exogenous history `(U_L0, ..., U_L(k-1), U_Y)` is recovered exactly from its factual
//! trajectory once, and every requested action history (world) is replayed against that same
//! history. Cost is linear in `units x worlds x nodes`, bounded by [`MAX_WORK`].
//!
//! Node `i` of a slice-`t` layout is `2t` (covariate) or `2t + 1` (action); the outcome is
//! `2k`. For `k = 2` these are exactly the indices of [`super::TemporalNode`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::ExecutionContext;

use super::{
    HistoryId, POLL_EVERY, RefusingWitness, SnapshotId, TemporalCounterfactualError,
    UnitDrawDigest, UnitId, digest, invalid, poll, unsupported,
};

/// Most `units x (worlds + 1) x nodes` evaluations one call may perform. A work budget, not a
/// structural count: any horizon, world count or unit count within it is accepted.
pub const MAX_WORK: u64 = 1 << 32;

/// Number of nodes for `horizon` slices: a covariate and an action per slice, then the outcome.
#[must_use]
pub const fn node_count(horizon: usize) -> usize {
    2 * horizon + 1
}

/// Index of the covariate of `slice`.
#[must_use]
pub const fn covariate_node(slice: usize) -> usize {
    2 * slice
}

/// Index of the action of `slice`.
#[must_use]
pub const fn action_node(slice: usize) -> usize {
    2 * slice + 1
}

/// Index of the final outcome for `horizon` slices.
#[must_use]
pub const fn outcome_node(horizon: usize) -> usize {
    2 * horizon
}

const fn is_action(node: usize) -> bool {
    node % 2 == 1
}

/// The additive-noise linear mechanism of one non-action node.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralNodeMechanism {
    /// Node index it generates (a covariate or the outcome).
    pub node: usize,
    /// Intercept.
    pub intercept: f64,
    /// Parent coefficients over strictly earlier nodes.
    pub parent_coefficients: Vec<(usize, f64)>,
    /// Optional half-width of the noise support; a larger abduced residual is a refuting
    /// witness.
    pub noise_halfwidth: Option<f64>,
}

/// The mechanism fit: one mechanism for every covariate and the outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralMechanismFit {
    /// Stable identity of the fit.
    pub fit_id: String,
    /// The mechanisms.
    pub mechanisms: Vec<GeneralNodeMechanism>,
}

/// One unit's factual trajectory over all nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralFactualHistory {
    /// The unit.
    pub unit: UnitId,
    /// The unit's factual history id.
    pub history: HistoryId,
    /// Observation time of each slice.
    pub times: Vec<u32>,
    /// Observed values of all `2 * horizon + 1` nodes, in node order.
    pub values: Vec<f64>,
}

/// A named action history and the units it is requested for.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralActionHistory {
    /// Name of the history (the world).
    pub name: String,
    /// Time of each slice's action; must equal the factual observation times.
    pub times: Vec<u32>,
    /// The action of each slice.
    pub actions: Vec<f64>,
    /// The units this world is evaluated for; must pair with the factual units.
    pub units: Vec<UnitId>,
}

/// The general temporal counterfactual request.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralTemporalSpec {
    /// Number of slices (at least one).
    pub horizon: usize,
    /// Whether the graph has latent confounding; `true` is refused.
    pub latent_confounding: bool,
    /// The mechanism fit.
    pub fit: GeneralMechanismFit,
    /// Snapshot the factual histories come from.
    pub snapshot: SnapshotId,
    /// Factual histories, one per unit.
    pub factual: Vec<GeneralFactualHistory>,
    /// The named action histories (at least one).
    pub worlds: Vec<GeneralActionHistory>,
}

/// One unit's counterfactual outcomes in every world.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralUnitCounterfactual {
    /// The unit.
    pub unit: UnitId,
    /// The unit's factual history.
    pub history: HistoryId,
    /// The factual final outcome.
    pub factual_outcome: f64,
    /// Final outcome under each world, in the order of [`GeneralTemporalSpec::worlds`].
    pub outcomes: Vec<f64>,
}

/// Receipt that every world was replayed against one shared abduced history per unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneralAbductionReceipt {
    /// Snapshot identity.
    pub snapshot: SnapshotId,
    /// Digest of the horizon and the mechanism fit (identity, structure and coefficients).
    pub fit_digest: String,
    /// Digest of the factual histories (order independent).
    pub factual_digest: String,
    /// Digest of each action history, in request order.
    pub world_digests: Vec<String>,
    /// Per-unit exogenous-draw digests, sorted by unit id.
    pub unit_draws: Vec<UnitDrawDigest>,
    /// Digest over all unit draws.
    pub exogenous_digest: String,
    /// Number of units.
    pub n_units: usize,
    /// Number of worlds.
    pub n_worlds: usize,
    /// Horizon in slices.
    pub horizon: usize,
    /// Every world read the same draw for every unit (always true for this route).
    pub shared_by_all_worlds: bool,
    /// Overall receipt digest.
    pub digest: String,
}

impl GeneralAbductionReceipt {
    pub(super) fn exogenous_of(unit_draws: &[UnitDrawDigest]) -> String {
        let mut d = digest("general_exogenous");
        d.word(unit_draws.len() as u64);
        for draw in unit_draws {
            d.text(draw.unit.as_str());
            d.text(draw.history.as_str());
            d.text(&draw.digest);
        }
        d.finish_hex()
    }

    /// Recompute the overall digest from the retained components; bit-identical to
    /// [`Self::digest`] for an untampered receipt.
    #[must_use]
    pub fn recompute_digest(&self) -> String {
        let mut d = digest("general_receipt");
        d.text(self.snapshot.as_str());
        d.text(&self.fit_digest);
        d.text(&self.factual_digest);
        d.word(self.world_digests.len() as u64);
        for world in &self.world_digests {
            d.text(world);
        }
        d.text(&Self::exogenous_of(&self.unit_draws));
        d.word(self.n_units as u64);
        d.word(self.n_worlds as u64);
        d.word(self.horizon as u64);
        d.word(u64::from(self.shared_by_all_worlds));
        d.finish_hex()
    }

    /// Whether the retained digests recompute from the retained components.
    #[must_use]
    pub fn is_self_consistent(&self) -> bool {
        self.exogenous_digest == Self::exogenous_of(&self.unit_draws)
            && self.digest == self.recompute_digest()
    }
}

/// The general temporal counterfactual answer.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralTemporalResult {
    /// Mechanism class ([`super::MECHANISM_CLASS`]).
    pub mechanism_class: &'static str,
    /// World names, in request order.
    pub world_names: Vec<String>,
    /// Per-unit outcomes, sorted by unit id.
    pub units: Vec<GeneralUnitCounterfactual>,
    /// Mean final outcome under each world, in request order.
    pub world_means: Vec<f64>,
    /// The shared-abduction receipt.
    pub receipt: GeneralAbductionReceipt,
}

impl GeneralTemporalResult {
    /// Sample mean contrast `mean(world plus) - mean(world minus)`, or `None` when an index is
    /// out of range.
    #[must_use]
    pub fn mean_contrast(&self, plus: usize, minus: usize) -> Option<f64> {
        Some(self.world_means.get(plus)? - self.world_means.get(minus)?)
    }
}

fn node_value(mech: &GeneralNodeMechanism, values: &[f64]) -> f64 {
    mech.intercept + mech.parent_coefficients.iter().map(|&(p, c)| c * values[p]).sum::<f64>()
}

pub(super) fn check_fit<'a>(
    horizon: usize,
    fit: &'a GeneralMechanismFit,
) -> Result<Vec<&'a GeneralNodeMechanism>, TemporalCounterfactualError> {
    let mismatch = |what: String| invalid("temporal_counterfactual.fit_mismatch", Some(what));
    let nodes = node_count(horizon);
    if fit.fit_id.trim().is_empty() || fit.mechanisms.len() != horizon + 1 {
        return Err(mismatch("mechanisms".to_owned()));
    }
    // Generating nodes in order: every covariate, then the outcome.
    let generating: Vec<usize> =
        (0..horizon).map(covariate_node).chain([outcome_node(horizon)]).collect();
    let mut ordered = Vec::with_capacity(generating.len());
    for node in generating {
        let mut hits = fit.mechanisms.iter().filter(|m| m.node == node);
        let mech = match (hits.next(), hits.next()) {
            (Some(m), None) => m,
            _ => return Err(mismatch(format!("node {node}"))),
        };
        let mut parents: Vec<usize> = mech.parent_coefficients.iter().map(|&(p, _)| p).collect();
        parents.sort_unstable();
        let structured = parents.iter().all(|&p| p < node && p < nodes)
            && parents.windows(2).all(|w| w[0] != w[1]);
        let finite = mech.intercept.is_finite()
            && mech.parent_coefficients.iter().all(|&(_, c)| c.is_finite())
            && mech.noise_halfwidth.is_none_or(|h| h.is_finite() && h > 0.0);
        if !structured || !finite {
            return Err(mismatch(format!("node {node}")));
        }
        ordered.push(mech);
    }
    Ok(ordered)
}

pub(super) fn digest_fit(horizon: usize, fit: &GeneralMechanismFit) -> String {
    let mut d = digest("general_fit");
    d.word(horizon as u64);
    d.text(&fit.fit_id);
    let mut mechs: Vec<&GeneralNodeMechanism> = fit.mechanisms.iter().collect();
    mechs.sort_by_key(|m| m.node);
    for mech in mechs {
        d.word(mech.node as u64);
        d.real(mech.intercept);
        let mut coefs = mech.parent_coefficients.clone();
        coefs.sort_by_key(|&(p, _)| p);
        d.word(coefs.len() as u64);
        for (p, c) in coefs {
            d.word(p as u64);
            d.real(c);
        }
        match mech.noise_halfwidth {
            Some(h) => {
                d.word(1);
                d.real(h);
            }
            None => {
                d.word(0);
            }
        }
    }
    d.finish_hex()
}

pub(super) fn digest_factual(factual: &[&GeneralFactualHistory]) -> String {
    let mut d = digest("general_factual");
    d.word(factual.len() as u64);
    for unit in factual {
        d.text(unit.unit.as_str());
        d.text(unit.history.as_str());
        d.word(unit.times.len() as u64);
        for &t in &unit.times {
            d.word(u64::from(t));
        }
        d.reals(&unit.values);
    }
    d.finish_hex()
}

pub(super) fn digest_world(world: &GeneralActionHistory) -> String {
    let mut d = digest("general_action_history");
    d.text(&world.name);
    d.word(world.times.len() as u64);
    for &t in &world.times {
        d.word(u64::from(t));
    }
    d.reals(&world.actions);
    let mut units: Vec<&UnitId> = world.units.iter().collect();
    units.sort();
    d.word(units.len() as u64);
    for unit in units {
        d.text(unit.as_str());
    }
    d.finish_hex()
}

pub(super) fn digest_draw(
    snapshot: &SnapshotId,
    unit: &GeneralFactualHistory,
    noise: &[f64],
) -> String {
    let mut d = digest("general_unit_draw");
    d.text(snapshot.as_str());
    d.text(unit.unit.as_str());
    d.text(unit.history.as_str());
    d.word(noise.len() as u64);
    d.reals(noise);
    d.finish_hex()
}

fn unpaired(
    world: &str,
    unit: &UnitId,
    history: Option<&HistoryId>,
) -> TemporalCounterfactualError {
    unsupported(
        "temporal_counterfactual.unpaired_histories",
        Some(unit.as_str().to_owned()),
        Some(RefusingWitness {
            unit: Some(unit.clone()),
            history: history.cloned(),
            world: Some(world.to_owned()),
            node: None,
            residual: None,
            bound: None,
        }),
    )
}

pub(super) fn check_inputs(
    spec: &GeneralTemporalSpec,
    factual: &[&GeneralFactualHistory],
) -> Result<(), TemporalCounterfactualError> {
    let nodes = node_count(spec.horizon);
    if factual.is_empty() {
        return Err(unsupported("temporal_counterfactual.shared_history_missing", None, None));
    }
    if spec.worlds.is_empty() {
        return Err(invalid("temporal_counterfactual.history_name_invalid", None));
    }
    let work = (factual.len() as u64)
        .saturating_mul(spec.worlds.len() as u64 + 1)
        .saturating_mul(nodes as u64);
    if work > MAX_WORK {
        return Err(invalid(
            "temporal_counterfactual.too_many_units",
            Some(format!("work {work} exceeds {MAX_WORK}")),
        ));
    }
    let mut names = BTreeSet::new();
    for world in &spec.worlds {
        if world.name.trim().is_empty() || !names.insert(world.name.as_str()) {
            return Err(invalid(
                "temporal_counterfactual.history_name_invalid",
                Some(world.name.clone()),
            ));
        }
        if world.actions.len() != spec.horizon
            || world.times.len() != spec.horizon
            || !world.actions.iter().all(|a| a.is_finite())
        {
            return Err(invalid(
                "temporal_counterfactual.non_finite_history",
                Some(world.name.clone()),
            ));
        }
    }
    let mut seen: BTreeSet<&UnitId> = BTreeSet::new();
    for unit in factual {
        if !seen.insert(&unit.unit) {
            return Err(unpaired("factual", &unit.unit, Some(&unit.history)));
        }
        if unit.values.len() != nodes
            || unit.times.len() != spec.horizon
            || !unit.values.iter().all(|v| v.is_finite())
        {
            return Err(invalid(
                "temporal_counterfactual.non_finite_history",
                Some(unit.unit.as_str().to_owned()),
            ));
        }
    }
    for world in &spec.worlds {
        let mut units = BTreeSet::new();
        for unit in &world.units {
            if !units.insert(unit) {
                return Err(unpaired(&world.name, unit, None));
            }
        }
        if let Some(missing) = factual.iter().find(|f| !units.contains(&f.unit)) {
            return Err(unpaired(&world.name, &missing.unit, Some(&missing.history)));
        }
        if let Some(extra) = units.iter().find(|u| !seen.contains(**u)) {
            return Err(unpaired(&world.name, extra, None));
        }
        if let Some(unit) = factual.iter().find(|f| f.times != world.times) {
            return Err(unsupported(
                "temporal_counterfactual.time_misaligned",
                Some(unit.unit.as_str().to_owned()),
                Some(RefusingWitness {
                    unit: Some(unit.unit.clone()),
                    history: Some(unit.history.clone()),
                    world: Some(world.name.clone()),
                    node: None,
                    residual: None,
                    bound: None,
                }),
            ));
        }
    }
    Ok(())
}

/// Abduce one unit's exogenous history, one residual per generating node.
pub(super) fn abduce_unit(
    mechs: &[&GeneralNodeMechanism],
    unit: &GeneralFactualHistory,
    noise: &mut [f64],
) -> Result<(), TemporalCounterfactualError> {
    for (slot, mech) in mechs.iter().enumerate() {
        let residual = unit.values[mech.node] - node_value(mech, &unit.values);
        let refuted =
            !residual.is_finite() || mech.noise_halfwidth.is_some_and(|h| residual.abs() > h);
        if refuted {
            return Err(unsupported(
                "temporal_counterfactual.refuting_history",
                Some(format!("{}: node {}", unit.unit.as_str(), mech.node)),
                Some(RefusingWitness {
                    unit: Some(unit.unit.clone()),
                    history: Some(unit.history.clone()),
                    world: None,
                    node: None,
                    residual: Some(residual),
                    bound: mech.noise_halfwidth,
                }),
            ));
        }
        noise[slot] = residual;
    }
    Ok(())
}

/// Replay one action history against an abduced exogenous history; returns the final outcome.
pub(super) fn replay(
    mechs: &[&GeneralNodeMechanism],
    noise: &[f64],
    actions: &[f64],
    values: &mut [f64],
) -> f64 {
    let mut generated = 0usize;
    for node in 0..values.len() {
        values[node] = if is_action(node) {
            actions[node / 2]
        } else {
            let value = node_value(mechs[generated], values) + noise[generated];
            generated += 1;
            value
        };
    }
    values[values.len() - 1]
}

/// Abduce each unit's exogenous history once, then replay every named action history
/// against that same history.
///
/// Cancellation is polled before validation and every 1024 units.
///
/// # Errors
///
/// The refusals of [`super::evaluate_temporal_counterfactual`] (unpaired or misaligned
/// histories, a refuting factual history, a mismatched fit, non-finite values) for the
/// general layout, `temporal_counterfactual.invalid_graph` for a zero horizon, and
/// [`TemporalCounterfactualError::Cancelled`].
pub fn evaluate_general_temporal_counterfactual(
    spec: &GeneralTemporalSpec,
    ctx: &ExecutionContext,
) -> Result<GeneralTemporalResult, TemporalCounterfactualError> {
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
    check_inputs(spec, &factual)?;

    let n_worlds = spec.worlds.len();
    let mut noise = vec![0.0; mechs.len()];
    let mut scratch = vec![0.0; node_count(spec.horizon)];
    let mut units = Vec::with_capacity(factual.len());
    let mut unit_draws = Vec::with_capacity(factual.len());
    let mut sums = vec![0.0; n_worlds];
    for (index, unit) in factual.iter().enumerate() {
        if index % POLL_EVERY == 0 {
            poll(ctx)?;
        }
        abduce_unit(&mechs, unit, &mut noise)?;
        let mut outcomes = Vec::with_capacity(n_worlds);
        for (sum, world) in sums.iter_mut().zip(&spec.worlds) {
            let outcome = replay(&mechs, &noise, &world.actions, &mut scratch);
            *sum += outcome;
            outcomes.push(outcome);
        }
        unit_draws.push(UnitDrawDigest {
            unit: unit.unit.clone(),
            history: unit.history.clone(),
            digest: digest_draw(&spec.snapshot, unit, &noise),
        });
        units.push(GeneralUnitCounterfactual {
            unit: unit.unit.clone(),
            history: unit.history.clone(),
            factual_outcome: unit.values[outcome_node(spec.horizon)],
            outcomes,
        });
    }
    poll(ctx)?;
    let n = units.len() as f64;
    let mut receipt = GeneralAbductionReceipt {
        snapshot: spec.snapshot.clone(),
        fit_digest: digest_fit(spec.horizon, &spec.fit),
        factual_digest: digest_factual(&factual),
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
    Ok(GeneralTemporalResult {
        mechanism_class: super::MECHANISM_CLASS,
        world_names: spec.worlds.iter().map(|w| w.name.clone()).collect(),
        units,
        world_means: sums.iter().map(|s| s / n).collect(),
        receipt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_cross_world::{
        FactualUnitHistory, NamedActionHistory, NodeMechanism, TemporalCounterfactualSpec,
        TemporalGraph, TemporalMechanismFit, TemporalNode, evaluate_temporal_counterfactual,
    };

    fn units(n: usize) -> Vec<UnitId> {
        (0..n).map(|i| UnitId::new(format!("u{i:03}"))).collect()
    }

    /// Generate a factual trajectory of the horizon-`k` chain from fixed noise.
    fn factual_for(horizon: usize, i: usize) -> Vec<f64> {
        let nodes = node_count(horizon);
        let mut values = vec![0.0; nodes];
        for node in 0..nodes {
            values[node] = if is_action(node) {
                ((i + node) % 2) as f64
            } else {
                let prior: f64 = if node == 0 { 0.0 } else { values[node - 1] + 0.25 * values[0] };
                0.5 + 0.3 * prior + 0.1 * ((i * 7 + node * 3) % 11) as f64
            };
        }
        values
    }

    fn chain_fit(horizon: usize) -> GeneralMechanismFit {
        let mut mechanisms = Vec::new();
        for slice in 0..horizon {
            let node = covariate_node(slice);
            let parents = if node == 0 { vec![] } else { vec![(node - 1, 0.3), (0, 0.25)] };
            mechanisms.push(GeneralNodeMechanism {
                node,
                intercept: 0.5,
                parent_coefficients: parents,
                noise_halfwidth: None,
            });
        }
        let out = outcome_node(horizon);
        mechanisms.push(GeneralNodeMechanism {
            node: out,
            intercept: 0.5,
            parent_coefficients: vec![(out - 1, 0.3), (0, 0.25)],
            noise_halfwidth: None,
        });
        GeneralMechanismFit { fit_id: "fit".into(), mechanisms }
    }

    fn general_spec(horizon: usize, n: usize, actions: &[Vec<f64>]) -> GeneralTemporalSpec {
        let ids = units(n);
        GeneralTemporalSpec {
            horizon,
            latent_confounding: false,
            fit: chain_fit(horizon),
            snapshot: SnapshotId::new("snap"),
            factual: ids
                .iter()
                .enumerate()
                .map(|(i, unit)| GeneralFactualHistory {
                    unit: unit.clone(),
                    history: HistoryId::new(format!("h{i}")),
                    times: (0..horizon as u32).collect(),
                    values: factual_for(horizon, i),
                })
                .collect(),
            worlds: actions
                .iter()
                .enumerate()
                .map(|(w, a)| GeneralActionHistory {
                    name: format!("world_{w}"),
                    times: (0..horizon as u32).collect(),
                    actions: a.clone(),
                    units: ids.clone(),
                })
                .collect(),
        }
    }

    #[test]
    fn two_slice_special_case_matches_the_fixed_route() {
        let n = 6;
        let general = general_spec(2, n, &[vec![1.0, 1.0], vec![0.0, 0.0]]);
        let mechanism = |node: TemporalNode, index: usize| {
            let g = &general.fit.mechanisms.iter().find(|m| m.node == index).unwrap();
            let all = [
                TemporalNode::Covariate0,
                TemporalNode::Action0,
                TemporalNode::Covariate1,
                TemporalNode::Action1,
                TemporalNode::Outcome,
            ];
            NodeMechanism {
                node,
                intercept: g.intercept,
                parent_coefficients: g
                    .parent_coefficients
                    .iter()
                    .map(|&(p, c)| (all[p], c))
                    .collect(),
                noise_halfwidth: None,
            }
        };
        let fixed = TemporalCounterfactualSpec {
            graph: TemporalGraph { horizon: 2, edges: Vec::new(), latent_confounding: false },
            fit: TemporalMechanismFit {
                fit_id: "fit".into(),
                mechanisms: vec![
                    mechanism(TemporalNode::Covariate0, 0),
                    mechanism(TemporalNode::Covariate1, 2),
                    mechanism(TemporalNode::Outcome, 4),
                ],
            },
            snapshot: SnapshotId::new("snap"),
            factual: general
                .factual
                .iter()
                .map(|f| FactualUnitHistory {
                    unit: f.unit.clone(),
                    history: f.history.clone(),
                    times: [f.times[0], f.times[1]],
                    values: [f.values[0], f.values[1], f.values[2], f.values[3], f.values[4]],
                })
                .collect(),
            plus: NamedActionHistory {
                name: "world_0".into(),
                times: [0, 1],
                actions: [1.0, 1.0],
                units: units(n),
            },
            minus: NamedActionHistory {
                name: "world_1".into(),
                times: [0, 1],
                actions: [0.0, 0.0],
                units: units(n),
            },
        };
        // The fixed route requires the graph's parents to equal the mechanisms' parents.
        let mut fixed = fixed;
        fixed.graph.edges = fixed
            .fit
            .mechanisms
            .iter()
            .flat_map(|m| m.parent_coefficients.iter().map(move |(p, _)| (*p, m.node)))
            .collect();
        let ctx = ExecutionContext::for_tests(1);
        let expected = evaluate_temporal_counterfactual(&fixed, &ctx).unwrap();
        let got = evaluate_general_temporal_counterfactual(&general, &ctx).unwrap();
        for (a, b) in expected.units.iter().zip(&got.units) {
            assert_eq!(a.plus_outcome, b.outcomes[0]);
            assert_eq!(a.minus_outcome, b.outcomes[1]);
        }
        assert_eq!(expected.mean_plus, got.world_means[0]);
        assert!(got.receipt.is_self_consistent());
    }

    #[test]
    fn longer_horizon_and_many_worlds_share_one_abduction() {
        let horizon = 5;
        let worlds: Vec<Vec<f64>> =
            (0..7).map(|w| (0..horizon).map(|t| ((w + t) % 3) as f64).collect()).collect();
        let spec = general_spec(horizon, 9, &worlds);
        let ctx = ExecutionContext::for_tests(1);
        let result = evaluate_general_temporal_counterfactual(&spec, &ctx).unwrap();
        assert_eq!(result.receipt.n_worlds, 7);
        assert_eq!(result.receipt.horizon, horizon);
        assert!(result.receipt.is_self_consistent());
        // Replaying the factual actions as a world reproduces the factual outcome exactly.
        let mut with_factual = spec.clone();
        with_factual.worlds.truncate(1);
        with_factual.worlds[0].actions =
            (0..horizon).map(|t| with_factual.factual[0].values[action_node(t)]).collect();
        with_factual.factual.truncate(1);
        with_factual.worlds[0].units = vec![with_factual.factual[0].unit.clone()];
        let replayed = evaluate_general_temporal_counterfactual(&with_factual, &ctx).unwrap();
        let unit = &replayed.units[0];
        assert!((unit.outcomes[0] - unit.factual_outcome).abs() < 1e-12);
        assert!(result.mean_contrast(0, 1).is_some());
        assert!(result.mean_contrast(0, 7).is_none());
    }

    #[test]
    fn unpaired_world_units_are_refused() {
        let mut spec = general_spec(3, 4, &[vec![0.0, 1.0, 0.0]]);
        let dropped = spec.worlds[0].units.pop().expect("the world has units");
        let err = evaluate_general_temporal_counterfactual(&spec, &ExecutionContext::for_tests(1))
            .unwrap_err();
        assert!(err.refusal().is_some());
        // The witness names the world and the factual unit it no longer pairs with.
        let witness = err.witness().expect("an unpaired history carries a witness");
        assert_eq!(witness.world.as_deref(), Some(spec.worlds[0].name.as_str()));
        assert_eq!(witness.unit.as_ref(), Some(&dropped));
        assert!(witness.history.is_some());
    }
}
