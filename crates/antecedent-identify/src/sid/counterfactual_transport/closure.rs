//! The transport closure of single-world c-factors.
//!
//! Every single-world factor the counterfactual recursion needs is
//! `Q[S] = P(S | do(V \ S))` for a bidirected-connected variable set `S` (Tian and
//! Pearl 2002). This module computes, by forward chaining over `(population,
//! variable set)` states, which such factors of the *target* population are
//! derivable from supplied (or declared) regime laws, and records one derivation
//! per factor.
//!
//! Item semantics. An item `(population, M)` stands for the law
//! `Q^population[M] = P^population(M | do(V \ M))`, a function of `M` and of the
//! parents of `M` outside `M`. The derivation steps are:
//!
//! * `Regime`: a supplied law `P(M | do(Z))` with `M` closed under ancestors in
//!   the graph restricted to `V \ Z` is `Q[M]`, because every parent of `M`
//!   outside `M` is then in `Z` (Tian and Pearl 2002, Lemma 1).
//! * `Marginal`: for `M'` closed under ancestors in `G[M]`,
//!   `Q[M'] = sum_{M \ M'} Q[M]` (Tian and Pearl 2002, Lemma 1).
//! * `Extract`: for a district `D` of `G[M]` (bidirected-connected within `M`),
//!   `Q[D] = prod_{i in D} Q[M](v_i | v_j : j in M precedes i)` (Tian and Pearl 2002,
//!   Lemma 3, c-component factorization).
//! * `Transfer`: `Q^source[M] = Q^target[M]` when no selection target of the source
//!   lies in `M`, because `Q[M]` is a function of the mechanisms of the variables
//!   in `M` and of the laws of the exogenous terms they read, none of which is
//!   selected (a selection diagram's definition).
//!
//! The closure is a sound, bounded search; absence of an item is not a
//! non-transportability claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{RegimeId, SearchStop};

use super::CounterfactualTransportProblem as Problem;

/// Where an item's law comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ItemSource {
    /// A supplied (`Some`) or declared-available (`None`) regime law
    /// `P(M | do(intervened))`.
    Regime {
        /// The catalog regime; `None` for a declared but absent regime.
        regime: Option<RegimeId>,
        /// Bit set of the regime's intervened variables.
        intervened: u32,
    },
    /// A marginal of the item `from` over an ancestrally closed subset.
    Marginal {
        /// Source item index.
        from: usize,
    },
    /// The factor of one district of the item `from`.
    Extract {
        /// Source item index.
        from: usize,
    },
    /// The same law in the target population (no selection target in `M`).
    Transfer {
        /// Source item index.
        from: usize,
    },
}

/// One derived law.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Item {
    /// Population index (0 is the target).
    pub(super) population: usize,
    /// Bit set of the item's variables.
    pub(super) mask: u32,
    /// How it was derived.
    pub(super) source: ItemSource,
    /// Derivation depth.
    pub(super) depth: usize,
}

/// A regime seed: a population, the intervened set and the measured set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Seed {
    /// Population index.
    pub(super) population: usize,
    /// Intervened variable set.
    pub(super) intervened: u32,
    /// Measured variable set (disjoint from `intervened`).
    pub(super) measured: u32,
    /// The catalog regime, `None` for a declared candidate.
    pub(super) regime: Option<RegimeId>,
}

/// Derived items in discovery order with an index by `(population, mask)`.
#[derive(Clone, Debug, Default)]
pub(super) struct Closure {
    /// Items in discovery order.
    pub(super) items: Vec<Item>,
    /// `(population, mask) ->` first derivation.
    pub(super) index: BTreeMap<(usize, u32), usize>,
}

impl Closure {
    /// The target item of `mask`, if derived.
    pub(super) fn target_item(&self, mask: u32) -> Option<usize> {
        self.index.get(&(0, mask)).copied()
    }
}

fn bits(mask: u32) -> impl Iterator<Item = usize> {
    (0..32usize).filter(move |i| (mask >> *i) & 1 == 1)
}

/// Ancestors of `v` (inclusive) using only directed edges inside `within`.
fn ancestors_within(problem: &Problem, within: u32, v: usize) -> u32 {
    let mut acc = 1u32 << v;
    loop {
        let mut next = acc;
        for u in bits(acc) {
            next |= problem.parent_mask[u] & within;
        }
        if next == acc {
            return acc;
        }
        acc = next;
    }
}

/// Whether `subset` is closed under ancestors in the graph induced on `within`.
fn ancestral_in(problem: &Problem, within: u32, subset: u32) -> bool {
    bits(subset).all(|v| ancestors_within(problem, within, v) & !subset == 0)
}

/// Districts (bidirected-connected components) of the graph induced on `mask`.
pub(super) fn districts_of(problem: &Problem, mask: u32) -> Vec<u32> {
    let mut remaining = mask;
    let mut out = Vec::new();
    while remaining != 0 {
        let start = remaining.trailing_zeros() as usize;
        let mut component = 1u32 << start;
        loop {
            let mut next = component;
            for v in bits(component) {
                next |= problem.bidirected_mask[v] & mask;
            }
            if next == component {
                break;
            }
            component = next;
        }
        out.push(component);
        remaining &= !component;
    }
    out
}

/// Whether the measured set of a regime is a legitimate `Q[M]`: closed under
/// ancestors in the graph restricted to the non-intervened variables.
pub(super) fn regime_measured_is_factor(problem: &Problem, intervened: u32, measured: u32) -> bool {
    let free = problem.full_mask() & !intervened;
    measured != 0
        && measured & intervened == 0
        && measured & !free == 0
        && ancestral_in(problem, free, measured)
}

/// Close the seeds under marginalization, district extraction and transfer.
///
/// `selected[p]` is the bit set of selection targets of population `p`
/// (population 0, the target, has none). The `seeds` are closed first, so every
/// item they derive keeps a derivation that cites supplied regimes only; the
/// `extra` seeds (declared but absent regimes) are then added and the closure
/// continues, deriving only what the supplied closure did not. `charge` is called
/// once per processed item with `(depth, estimated live bytes)` and may stop the
/// search.
pub(super) fn close(
    problem: &Problem,
    selected: &[u32],
    seeds: &[Seed],
    extra: &[Seed],
    charge: &mut dyn FnMut(usize, u64) -> Result<(), SearchStop>,
) -> Result<Closure, SearchStop> {
    let mut closure = Closure::default();
    let mut head = 0usize;
    let mut pending: Vec<&[Seed]> = vec![extra, seeds];
    while let Some(group) = pending.pop() {
        for seed in group {
            if seed.measured == 0 {
                continue;
            }
            let key = (seed.population, seed.measured);
            if closure.index.contains_key(&key) {
                continue;
            }
            let id = closure.items.len();
            closure.items.push(Item {
                population: seed.population,
                mask: seed.measured,
                source: ItemSource::Regime { regime: seed.regime, intervened: seed.intervened },
                depth: 0,
            });
            closure.index.insert(key, id);
        }
        while head < closure.items.len() {
            let item = closure.items[head];
            let bytes = (closure.items.len() as u64).saturating_mul(64);
            charge(item.depth, bytes)?;
            let mut produced: Vec<Item> = Vec::new();
            if item.population != 0 && item.mask & selected[item.population] == 0 {
                produced.push(Item {
                    population: 0,
                    mask: item.mask,
                    source: ItemSource::Transfer { from: head },
                    depth: item.depth + 1,
                });
            }
            // Proper non-empty ancestrally closed subsets, descending numerically.
            let mut sub = item.mask.wrapping_sub(1) & item.mask;
            while sub != 0 {
                if ancestral_in(problem, item.mask, sub) {
                    produced.push(Item {
                        population: item.population,
                        mask: sub,
                        source: ItemSource::Marginal { from: head },
                        depth: item.depth + 1,
                    });
                }
                sub = sub.wrapping_sub(1) & item.mask;
            }
            for district in districts_of(problem, item.mask) {
                if district != item.mask {
                    produced.push(Item {
                        population: item.population,
                        mask: district,
                        source: ItemSource::Extract { from: head },
                        depth: item.depth + 1,
                    });
                }
            }
            for new in produced {
                let key = (new.population, new.mask);
                if let std::collections::btree_map::Entry::Vacant(slot) = closure.index.entry(key) {
                    slot.insert(closure.items.len());
                    closure.items.push(new);
                }
            }
            head += 1;
        }
    }
    Ok(closure)
}

/// Item indices reachable from `roots` through derivation links, ascending.
pub(super) fn reachable(closure: &Closure, roots: &[usize]) -> Vec<usize> {
    let mut seen = vec![false; closure.items.len()];
    let mut stack: Vec<usize> = roots.to_vec();
    while let Some(i) = stack.pop() {
        if seen[i] {
            continue;
        }
        seen[i] = true;
        match closure.items[i].source {
            ItemSource::Regime { .. } => {}
            ItemSource::Marginal { from }
            | ItemSource::Extract { from }
            | ItemSource::Transfer { from } => stack.push(from),
        }
    }
    (0..seen.len()).filter(|i| seen[*i]).collect()
}
