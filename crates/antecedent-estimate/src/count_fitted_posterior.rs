//! Posterior interval of a conditional probability of a count-fitted finite table model.
//!
//! A declared model's posterior probability is an exact number with no sampling
//! distribution. When the tables were estimated from a sample, the number inherits the
//! estimation uncertainty of the tables. This module fits each node's conditional from
//! per-parent-configuration counts (retaining the row count `n`), places an independent
//! Dirichlet posterior over every conditional, and reports the equal-tailed posterior
//! quantiles of `P(target = level | evidence)` by Monte Carlo over joint draws of all
//! tables, each draw conditioned through exact enumeration of the finite joint law.
//!
//! A node may instead be declared (a fixed, known conditional such as a measured signal
//! with known sensitivity and specificity); declared nodes carry no sampling
//! uncertainty and are not redrawn.
//!
//! Scope of the guarantee: independent rows, the model structure correct, every cell
//! counted from complete rows. Conditioning events of small probability and sparse
//! parent configurations are named boundaries of the measured coverage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "small dense finite-table arithmetic over bounded cardinalities"
)]

use std::sync::Arc;

use antecedent_core::CalibrationBasis;

use crate::error::EstimationError;

/// Estimator identity of the coverage record.
pub const ESTIMATOR_ID: &str = "observation.count_fitted_tables.dirichlet";
/// Interval method identity of the coverage record.
pub const INTERVAL_METHOD: &str = "posterior_quantile";
/// Functional identity of the coverage record.
pub const FUNCTIONAL_ID: &str = "conditional_probability";
/// Largest number of joint assignments one exact enumeration may visit.
pub const MAX_ASSIGNMENTS: usize = 1 << 20;
/// Largest number of posterior draws.
pub const MAX_DRAWS: usize = 100_000;

fn unsupported(message: &'static str) -> EstimationError {
    EstimationError::Unsupported { message }
}

/// How one node's conditional distribution is known.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeTable {
    /// Counts, row-major over parent configurations (first parent varies slowest),
    /// each row `cardinality` long; estimated with a Dirichlet posterior.
    Counts(Vec<u64>),
    /// A declared conditional: row-major probabilities, rows summing to one.
    Declared(Vec<f64>),
}

/// One variable of the finite model.
#[derive(Clone, Debug, PartialEq)]
pub struct CountNode {
    /// Number of levels.
    pub cardinality: usize,
    /// Parent node indices, each smaller than this node's own index.
    pub parents: Vec<usize>,
    /// The conditional distribution of this node given its parents.
    pub table: NodeTable,
}

/// A conditional probability request.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorQuery {
    /// Node whose level probability is reported.
    pub target: usize,
    /// The level of the target.
    pub level: usize,
    /// Observed `(node, level)` evidence.
    pub evidence: Vec<(usize, usize)>,
}

/// The posterior interval and its provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorInterval {
    /// The conditional probability under the posterior-mean tables.
    pub point: f64,
    /// Lower equal-tailed posterior quantile.
    pub lower: f64,
    /// Upper equal-tailed posterior quantile.
    pub upper: f64,
    /// Nominal credible mass.
    pub level: f64,
    /// Rows the count tables were fitted from.
    pub rows: u64,
    /// Posterior draws used.
    pub draws: u32,
}

/// A topologically ordered finite model with count-fitted or declared conditionals.
#[derive(Clone, Debug, PartialEq)]
pub struct CountFittedModel {
    nodes: Vec<CountNode>,
    rows: u64,
}

impl CountFittedModel {
    /// Validate the structure and the table shapes.
    ///
    /// `rows` is the number of observations the count tables were fitted from.
    ///
    /// # Errors
    /// A parent that is not earlier, a table of the wrong shape, a declared row that is
    /// not a probability vector, no node, a cardinality below two, or an enumeration
    /// larger than [`MAX_ASSIGNMENTS`].
    pub fn new(nodes: Vec<CountNode>, rows: u64) -> Result<Self, EstimationError> {
        if nodes.is_empty() {
            return Err(unsupported("count-fitted model: no nodes"));
        }
        let mut assignments = 1_usize;
        for (index, node) in nodes.iter().enumerate() {
            if node.cardinality < 2 {
                return Err(unsupported("count-fitted model: a node needs two or more levels"));
            }
            assignments = assignments
                .checked_mul(node.cardinality)
                .filter(|&a| a <= MAX_ASSIGNMENTS)
                .ok_or_else(|| unsupported("count-fitted model: enumeration is too large"))?;
            if node.parents.iter().any(|&p| p >= index) {
                return Err(unsupported("count-fitted model: parents must precede their child"));
            }
            let configurations: usize =
                node.parents.iter().map(|&p| nodes[p].cardinality).product();
            let expected = configurations * node.cardinality;
            match &node.table {
                NodeTable::Counts(counts) if counts.len() == expected => {}
                NodeTable::Declared(probabilities) if probabilities.len() == expected => {
                    for row in probabilities.chunks(node.cardinality) {
                        let sum: f64 = row.iter().sum();
                        if row.iter().any(|p| !p.is_finite() || *p < 0.0)
                            || (sum - 1.0).abs() > 1e-9
                        {
                            return Err(unsupported(
                                "count-fitted model: a declared row is not a probability vector",
                            ));
                        }
                    }
                }
                _ => return Err(unsupported("count-fitted model: a table has the wrong shape")),
            }
        }
        Ok(Self { nodes, rows })
    }

    /// Rows the count tables were fitted from.
    #[must_use]
    pub const fn rows(&self) -> u64 {
        self.rows
    }

    /// The nodes in topological order.
    #[must_use]
    pub fn nodes(&self) -> &[CountNode] {
        &self.nodes
    }

    fn config_index(&self, node: &CountNode, assignment: &[usize]) -> usize {
        node.parents.iter().fold(0, |acc, &p| acc * self.nodes[p].cardinality + assignment[p])
    }

    /// `P(target = level | evidence)` under per-node probability tables, by exact
    /// enumeration. `None` when the evidence has zero probability.
    fn conditional(&self, tables: &[Vec<f64>], query: &PosteriorQuery) -> Option<f64> {
        let n = self.nodes.len();
        let mut assignment = vec![0_usize; n];
        let (mut joint, mut evidence_mass) = (0.0_f64, 0.0_f64);
        loop {
            if query.evidence.iter().all(|&(node, level)| assignment[node] == level) {
                let mut p = 1.0;
                for (index, node) in self.nodes.iter().enumerate() {
                    let row = self.config_index(node, &assignment);
                    p *= tables[index][row * node.cardinality + assignment[index]];
                    if p == 0.0 {
                        break;
                    }
                }
                evidence_mass += p;
                if assignment[query.target] == query.level {
                    joint += p;
                }
            }
            // Odometer increment, last node fastest.
            let mut position = n;
            loop {
                if position == 0 {
                    return (evidence_mass > 0.0).then(|| joint / evidence_mass);
                }
                position -= 1;
                assignment[position] += 1;
                if assignment[position] < self.nodes[position].cardinality {
                    break;
                }
                assignment[position] = 0;
            }
        }
    }

    fn validate_query(&self, query: &PosteriorQuery) -> Result<(), EstimationError> {
        let bad = |node: usize, level: usize| match self.nodes.get(node) {
            Some(n) => level >= n.cardinality,
            None => true,
        };
        if bad(query.target, query.level) || query.evidence.iter().any(|&(n, l)| bad(n, l)) {
            return Err(unsupported("count-fitted posterior: a node or level is out of range"));
        }
        Ok(())
    }

    /// The point value under the posterior-mean tables (counts plus `prior_alpha`).
    ///
    /// # Errors
    /// An out-of-range query or evidence of zero probability.
    pub fn point(&self, query: &PosteriorQuery, prior_alpha: f64) -> Result<f64, EstimationError> {
        self.validate_query(query)?;
        let tables: Vec<Vec<f64>> = self
            .nodes
            .iter()
            .map(|node| match &node.table {
                NodeTable::Declared(p) => p.clone(),
                NodeTable::Counts(counts) => counts
                    .chunks(node.cardinality)
                    .flat_map(|row| {
                        let total: f64 = row.iter().map(|&c| c as f64 + prior_alpha).sum();
                        row.iter().map(move |&c| (c as f64 + prior_alpha) / total)
                    })
                    .collect(),
            })
            .collect();
        self.conditional(&tables, query)
            .ok_or_else(|| unsupported("count-fitted posterior: the evidence has zero probability"))
    }

    /// Equal-tailed posterior interval of `P(target = level | evidence)`.
    ///
    /// `seed` fixes the Monte Carlo stream; the result is a deterministic function of
    /// the counts, the query, the prior, the level, the draw count and the seed.
    ///
    /// # Errors
    /// A level outside `(0, 1)`, a non-positive prior, a draw count outside
    /// `[2, MAX_DRAWS]`, an out-of-range query, or evidence of zero probability.
    #[allow(
        clippy::nonminimal_bool,
        reason = "a NaN level or width must fail the open-interval test"
    )]
    pub fn posterior_interval(
        &self,
        query: &PosteriorQuery,
        level: f64,
        prior_alpha: f64,
        draws: u32,
        seed: u64,
    ) -> Result<PosteriorInterval, EstimationError> {
        if !(level.is_finite() && level > 0.0 && level < 1.0)
            || !(prior_alpha.is_finite() && prior_alpha > 0.0)
            || draws < 2
            || draws as usize > MAX_DRAWS
        {
            return Err(unsupported("count-fitted posterior: invalid level, prior or draw count"));
        }
        let point = self.point(query, prior_alpha)?;
        let mut rng = SplitMix(seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut values = Vec::with_capacity(draws as usize);
        let mut tables: Vec<Vec<f64>> = self
            .nodes
            .iter()
            .map(|node| match &node.table {
                NodeTable::Declared(p) => p.clone(),
                NodeTable::Counts(counts) => vec![0.0; counts.len()],
            })
            .collect();
        for _ in 0..draws {
            for (node, table) in self.nodes.iter().zip(tables.iter_mut()) {
                if let NodeTable::Counts(counts) = &node.table {
                    for (row_counts, row) in
                        counts.chunks(node.cardinality).zip(table.chunks_mut(node.cardinality))
                    {
                        let mut total = 0.0;
                        for (slot, &c) in row.iter_mut().zip(row_counts) {
                            *slot = rng.gamma(c as f64 + prior_alpha);
                            total += *slot;
                        }
                        for slot in row.iter_mut() {
                            *slot /= total;
                        }
                    }
                }
            }
            let value = self.conditional(&tables, query).ok_or_else(|| {
                unsupported("count-fitted posterior: the evidence has zero probability")
            })?;
            values.push(value);
        }
        values.sort_by(f64::total_cmp);
        let quantile = |p: f64| {
            let h = p * (values.len() - 1) as f64;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "h is within [0, len - 1]"
            )]
            let lo = h.floor() as usize;
            let fraction = h - lo as f64;
            values[lo] + fraction * (values[(lo + 1).min(values.len() - 1)] - values[lo])
        };
        Ok(PosteriorInterval {
            point,
            lower: quantile((1.0 - level) / 2.0),
            upper: quantile(1.0 - (1.0 - level) / 2.0),
            level,
            rows: self.rows,
            draws,
        })
    }
}

struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn normal(&mut self) -> f64 {
        let u = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }

    /// Gamma(shape, 1) by Marsaglia and Tsang, with the boost for shape below one.
    fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.uniform();
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

/// The calibration basis of a count-fitted posterior interval at `rows` fitted rows.
///
/// One definition serves the harness and any facade, so a record can only describe the
/// construction a facade reports.
#[must_use]
pub fn calibration_basis(rows: u64, level: f64, draws: u32) -> CalibrationBasis {
    basis_with_structure("fixed", rows, level, draws)
}

/// The calibration basis of a named-boundary probe of the same construction; its
/// structure differs, so no gated design shares its key.
#[must_use]
pub fn boundary_calibration_basis(
    boundary: &str,
    rows: u64,
    level: f64,
    draws: u32,
) -> CalibrationBasis {
    basis_with_structure(&format!("boundary.{boundary}"), rows, level, draws)
}

fn basis_with_structure(structure: &str, rows: u64, level: f64, draws: u32) -> CalibrationBasis {
    let a = |text: &str| Arc::<str>::from(text);
    CalibrationBasis::new(
        [
            a("ObservationConditionalProbability"),
            a("NotApplicable"),
            a(structure),
            a("count_fitted_finite_tables"),
            a("Bayesian"),
            a(ESTIMATOR_ID),
            a(INTERVAL_METHOD),
            a(""),
            a("independent_rows"),
            a("dirichlet.uniform_prior"),
            a(FUNCTIONAL_ID),
        ],
        level,
        a("conditioning"),
        rows,
        None,
        Some(draws),
        0.0,
    )
}
