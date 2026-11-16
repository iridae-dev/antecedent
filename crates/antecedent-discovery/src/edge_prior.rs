//! Per-edge and soft-tier factors of the graph prior.
//!
//! The independent-Bernoulli prior of [`crate::GraphPrior::edge_inclusion`] gives every candidate
//! edge the same inclusion probability. Two further modular factors let the prior say more
//! without changing any algorithm, because both keep the prior a product over directed edges
//! (and so a product over the parent sets of the nodes):
//!
//! * [`EdgePriors`]: a sparse map `(from, to) → p` of inclusion probabilities, with a default for
//!   the edges it does not list. Edge `i → j` contributes `ln p_ij` when present and
//!   `ln (1 - p_ij)` when absent.
//! * [`SoftTiers`]: a declared log-penalty on every edge that runs against a tier order. A
//!   present edge from a later tier into an earlier one contributes `-penalty`; nothing else
//!   changes. Hard tiers ([`crate::DiscoveryConstraints::tiers`]) forbid those same edges
//!   outright, and a hard constraint always wins: a forbidden edge has no prior factor at all.
//!
//! A prior that sets neither factor never reaches this module's arithmetic: the samplers and the
//! exact dynamic program keep their global-probability path unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use antecedent_core::VariableId;

use crate::error::DiscoveryError;

/// The refusal every per-edge prior check raises, under the edge-prior detail.
fn refuse(message: &str) -> DiscoveryError {
    DiscoveryError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("structure_posterior.edge_prior_probability: {message}"),
    }
}

/// A sparse per-edge inclusion probability map.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgePriors {
    /// `(from, to, p)`: the prior probability that the directed edge `from → to` is present.
    pub edges: Arc<[(VariableId, VariableId, f64)]>,
    /// Inclusion probability of the edges not listed. `None` keeps the prior's global
    /// [`crate::GraphPrior::edge_inclusion`] for them, or a neutral factor when that is unset.
    pub default: Option<f64>,
}

impl EdgePriors {
    /// A map of listed edges; unlisted edges keep the prior's global probability.
    ///
    /// Probabilities and names are checked when a posterior is run against its variables.
    #[must_use]
    pub fn new(edges: impl Into<Arc<[(VariableId, VariableId, f64)]>>) -> Self {
        Self { edges: edges.into(), default: None }
    }

    /// Set the inclusion probability of the unlisted edges.
    #[must_use]
    pub fn with_default(mut self, probability: f64) -> Self {
        self.default = Some(probability);
        self
    }
}

/// A prior penalty on edges that run against a declared tier order, short of forbidding them.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftTiers {
    /// Variable tiers, earliest first. A variable in several tiers keeps its earliest; a
    /// variable in none is never penalised.
    pub tiers: Arc<[Arc<[VariableId]>]>,
    /// Log-penalty `>= 0` charged to a present edge from a later tier into an earlier one: the
    /// prior odds of that edge are multiplied by `exp(-log_penalty)`.
    pub log_penalty: f64,
}

impl SoftTiers {
    /// Tiers with a declared log-penalty (`0` penalises nothing, larger values approach a hard
    /// tier).
    #[must_use]
    pub fn with_log_penalty(tiers: impl Into<Arc<[Arc<[VariableId]>]>>, log_penalty: f64) -> Self {
        Self { tiers: tiers.into(), log_penalty }
    }

    /// Tiers where an edge against the order keeps `factor in (0, 1)` of the prior odds it would
    /// have had along the order: the log-penalty is `-ln factor`.
    ///
    /// # Errors
    ///
    /// `factor` outside `(0, 1)`.
    pub fn with_odds_factor(
        tiers: impl Into<Arc<[Arc<[VariableId]>]>>,
        factor: f64,
    ) -> Result<Self, DiscoveryError> {
        if !(factor > 0.0 && factor < 1.0) {
            return Err(refuse(
                "the soft-tier odds factor of an edge against the order must lie in (0, 1)",
            ));
        }
        Ok(Self { tiers: tiers.into(), log_penalty: -factor.ln() })
    }

    /// Earliest tier index of every listed variable.
    fn tier_index(&self) -> HashMap<VariableId, usize> {
        let mut index = HashMap::new();
        for (tier_index, tier) in self.tiers.iter().enumerate() {
            for &variable in tier.iter() {
                index.entry(variable).or_insert(tier_index);
            }
        }
        index
    }
}

/// Reject a per-edge map or soft-tier declaration the prior cannot honour.
///
/// `forbidden(i, j)` is the hard-constraint verdict (forbidden list and hard tiers) on the
/// directed edge between dense indices.
pub(crate) fn validate(
    edge_priors: Option<&EdgePriors>,
    soft_tiers: Option<&SoftTiers>,
    variables: &[VariableId],
    forbidden: impl Fn(usize, usize) -> bool,
) -> Result<(), DiscoveryError> {
    let position = |v: VariableId| variables.iter().position(|&candidate| candidate == v);
    let known = |what: &str, v: VariableId| {
        position(v).ok_or_else(|| {
            refuse(&format!(
                "{what} names variable {} which is not among the posterior's variables",
                v.raw()
            ))
        })
    };
    let probability = |p: f64| p > 0.0 && p < 1.0;
    if let Some(edge_priors) = edge_priors {
        if edge_priors.default.is_some_and(|p| !probability(p)) {
            return Err(refuse(
                "the default inclusion probability of unlisted edges must lie in (0, 1)",
            ));
        }
        let mut seen = HashSet::new();
        for &(from, to, p) in edge_priors.edges.iter() {
            let i = known("a per-edge prior", from)?;
            let j = known("a per-edge prior", to)?;
            if i == j {
                return Err(refuse(&format!(
                    "a per-edge prior names the self edge on variable {}",
                    from.raw()
                )));
            }
            if !probability(p) {
                return Err(refuse(&format!(
                    "the per-edge inclusion probability of {} -> {} must lie in (0, 1)",
                    from.raw(),
                    to.raw()
                )));
            }
            if !seen.insert((i, j)) {
                return Err(refuse(&format!(
                    "the per-edge prior lists {} -> {} more than once",
                    from.raw(),
                    to.raw()
                )));
            }
            if forbidden(i, j) {
                return Err(refuse(&format!(
                    "the per-edge prior lists {} -> {}, which the constraints forbid",
                    from.raw(),
                    to.raw()
                )));
            }
        }
    }
    if let Some(soft) = soft_tiers {
        if !(soft.log_penalty.is_finite() && soft.log_penalty >= 0.0) {
            return Err(refuse("the soft-tier log-penalty must be finite and non-negative"));
        }
        for tier in soft.tiers.iter() {
            for &variable in tier.iter() {
                known("a soft tier", variable)?;
            }
        }
    }
    Ok(())
}

/// `(ln p, ln (1 - p))` of the directed edge `from → to`, per ordered pair, row-major
/// `from * n + to`, with the soft-tier penalty added to the present-edge term.
///
/// Unlisted edges take the map's default, then the global `edge_inclusion`, then a neutral
/// `(0, 0)`.
pub(crate) fn pair_logs(
    n: usize,
    variables: &[VariableId],
    global: Option<f64>,
    edge_priors: Option<&EdgePriors>,
    soft_tiers: Option<&SoftTiers>,
) -> Vec<(f64, f64)> {
    let logs = |p: f64| (p.ln(), (1.0 - p).ln());
    let base = edge_priors.and_then(|map| map.default).or(global).map_or((0.0, 0.0), logs);
    let mut table = vec![base; n * n];
    if let Some(map) = edge_priors {
        for &(from, to, p) in map.edges.iter() {
            let i = variables.iter().position(|&v| v == from);
            let j = variables.iter().position(|&v| v == to);
            if let (Some(i), Some(j)) = (i, j) {
                table[i * n + j] = logs(p);
            }
        }
    }
    if let Some(soft) = soft_tiers {
        let index = soft.tier_index();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if let (Some(&from_tier), Some(&to_tier)) =
                    (index.get(&variables[i]), index.get(&variables[j]))
                    && from_tier > to_tier
                {
                    table[i * n + j].0 -= soft.log_penalty;
                }
            }
        }
    }
    table
}
