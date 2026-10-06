//! Exact enumerated latent-variable structural causal models for the
//! counterfactual-identification tests (2.2B X8).
//!
//! Every observed variable `V` takes levels `0..card(V)` and is a deterministic
//! function of its observed parents, the binary latent of every bidirected edge
//! touching it, and its own exogenous response type (a finite distribution over
//! lookup tables). The exogenous terms are shared by every world, so a
//! counterfactual is computed by enumerating every exogenous state and solving
//! each world in topological order. Nothing here uses the identification code.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::result_large_err,
    dead_code,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities; exact bit comparisons are the assertion"
)]

/// Deterministic generator (splitmix64).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One response type: its probability and the level it assigns to every
/// configuration of (observed parents, then latents of the variable's
/// bidirected edges), row-major with the last input fastest.
#[derive(Clone, Debug)]
pub struct ResponseType {
    pub probability: f64,
    pub table: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct LatentScm {
    pub cards: Vec<usize>,
    pub parents: Vec<Vec<usize>>,
    /// Bidirected edges, each one binary latent `(a, b, P(U = 1))`.
    pub latents: Vec<(usize, usize, f64)>,
    pub types: Vec<Vec<ResponseType>>,
}

impl LatentScm {
    /// Topological order of the observed variables.
    pub fn topo(&self) -> Vec<usize> {
        let n = self.cards.len();
        let mut order = Vec::new();
        let mut placed = vec![false; n];
        while order.len() < n {
            let next = (0..n)
                .find(|&v| !placed[v] && self.parents[v].iter().all(|&p| placed[p]))
                .expect("acyclic");
            placed[next] = true;
            order.push(next);
        }
        order
    }

    /// Latents touching `v`, in declaration order.
    pub fn latents_of(&self, v: usize) -> Vec<usize> {
        (0..self.latents.len())
            .filter(|&k| self.latents[k].0 == v || self.latents[k].1 == v)
            .collect()
    }

    /// Inputs per configuration of `v`'s table.
    pub fn table_size(&self, v: usize) -> usize {
        self.parents[v].iter().map(|&p| self.cards[p]).product::<usize>()
            * (1usize << self.latents_of(v).len())
    }

    /// Solve one world: `set[v] = Some(level)` fixes `v`.
    pub fn solve(&self, latent: &[usize], types: &[usize], set: &[Option<usize>]) -> Vec<usize> {
        let mut values = vec![0usize; self.cards.len()];
        for v in self.topo() {
            if let Some(level) = set[v] {
                values[v] = level;
                continue;
            }
            let mut index = 0usize;
            for &p in &self.parents[v] {
                index = index * self.cards[p] + values[p];
            }
            for k in self.latents_of(v) {
                index = index * 2 + latent[k];
            }
            values[v] = self.types[v][types[v]].table[index];
        }
        values
    }

    /// Every exogenous state with its probability.
    pub fn states(&self) -> Vec<(f64, Vec<usize>, Vec<usize>)> {
        let mut out = Vec::new();
        let k = self.latents.len();
        let type_cards: Vec<usize> = self.types.iter().map(Vec::len).collect();
        for latent_mask in 0..(1usize << k) {
            let latent: Vec<usize> = (0..k).map(|i| (latent_mask >> i) & 1).collect();
            let p_latent: f64 = (0..k)
                .map(|i| if latent[i] == 1 { self.latents[i].2 } else { 1.0 - self.latents[i].2 })
                .product();
            let mut types = vec![0usize; self.cards.len()];
            loop {
                let p_types: f64 =
                    types.iter().enumerate().map(|(v, &t)| self.types[v][t].probability).product();
                out.push((p_latent * p_types, latent.clone(), types.clone()));
                let mut i = types.len();
                loop {
                    if i == 0 {
                        break;
                    }
                    i -= 1;
                    types[i] += 1;
                    if types[i] < type_cards[i] {
                        break;
                    }
                    types[i] = 0;
                }
                if types.iter().all(|&t| t == 0) {
                    break;
                }
            }
        }
        out
    }

    /// The observational joint, row-major over the variables, last fastest.
    pub fn observational(&self) -> Vec<f64> {
        let size: usize = self.cards.iter().product();
        let mut joint = vec![0.0; size];
        let none = vec![None; self.cards.len()];
        for (p, latent, types) in self.states() {
            let values = self.solve(&latent, &types, &none);
            joint[self.flat(&values)] += p;
        }
        joint
    }

    pub fn flat(&self, values: &[usize]) -> usize {
        values.iter().zip(&self.cards).fold(0, |acc, (&v, &c)| acc * c + v)
    }

    /// `P(Y_{x = active} = level, X = observed)` for every level of `Y`, from one
    /// pass over the exogenous states.
    pub fn ett_numerators(&self, x: usize, active: usize, observed: usize, y: usize) -> Vec<f64> {
        let natural = vec![None; self.cards.len()];
        let mut treated = natural.clone();
        treated[x] = Some(active);
        let mut out = vec![0.0; self.cards[y]];
        for (p, latent, types) in self.states() {
            if self.solve(&latent, &types, &natural)[x] == observed {
                out[self.solve(&latent, &types, &treated)[y]] += p;
            }
        }
        out
    }

    /// `P(Y_{x = active} = level, X = observed)` for every active level, observed
    /// level and outcome level (`[active][observed][level]`), from one pass over
    /// the exogenous states.
    pub fn ett_numerator_table(&self, x: usize, y: usize) -> Vec<Vec<Vec<f64>>> {
        let natural = vec![None; self.cards.len()];
        let mut out = vec![vec![vec![0.0; self.cards[y]]; self.cards[x]]; self.cards[x]];
        for (p, latent, types) in self.states() {
            let observed = self.solve(&latent, &types, &natural)[x];
            for active in 0..self.cards[x] {
                let mut treated = natural.clone();
                treated[x] = Some(active);
                out[active][observed][self.solve(&latent, &types, &treated)[y]] += p;
            }
        }
        out
    }

    /// Number of exogenous states `states()` enumerates.
    pub fn state_count(&self) -> usize {
        self.types.iter().map(Vec::len).product::<usize>() << self.latents.len()
    }

    /// `P(Y_{x = active} = y, X = observed)`.
    pub fn ett_numerator(
        &self,
        x: usize,
        active: usize,
        observed: usize,
        y: usize,
        level: usize,
    ) -> f64 {
        self.ett_numerators(x, active, observed, y)[level]
    }

    /// `P(Y_{x = active} = level | X = observed)`.
    pub fn ett(&self, x: usize, active: usize, observed: usize, y: usize, level: usize) -> f64 {
        let numerators = self.ett_numerators(x, active, observed, y);
        numerators[level] / numerators.iter().sum::<f64>()
    }

    /// A random model on the graph with `types_per_variable` response types each
    /// (positive probabilities) and latent probabilities in (0.2, 0.8).
    pub fn random(
        rng: &mut Rng,
        cards: &[usize],
        directed: &[(usize, usize)],
        bidirected: &[(usize, usize)],
        types_per_variable: usize,
    ) -> Self {
        let n = cards.len();
        let mut parents = vec![Vec::new(); n];
        for &(a, b) in directed {
            parents[b].push(a);
        }
        for p in &mut parents {
            p.sort_unstable();
        }
        let latents = bidirected.iter().map(|&(a, b)| (a, b, 0.2 + 0.6 * rng.unit())).collect();
        let mut scm = Self { cards: cards.to_vec(), parents, latents, types: Vec::new() };
        let mut types = Vec::with_capacity(n);
        for v in 0..n {
            let size = scm.table_size(v);
            let weights: Vec<f64> = (0..types_per_variable).map(|_| 0.1 + rng.unit()).collect();
            let total: f64 = weights.iter().sum();
            // Random lookup tables, plus one constant type per level with a small
            // share, so every level of every variable has positive probability
            // under every configuration and the observational joint is positive.
            let constant_share = 0.04;
            let mut list: Vec<ResponseType> = weights
                .into_iter()
                .map(|w| ResponseType {
                    probability: (1.0 - constant_share * cards[v] as f64) * w / total,
                    table: (0..size).map(|_| rng.below(cards[v])).collect(),
                })
                .collect();
            list.extend((0..cards[v]).map(|level| ResponseType {
                probability: constant_share,
                table: vec![level; size],
            }));
            types.push(list);
        }
        scm.types = types;
        scm
    }
}
