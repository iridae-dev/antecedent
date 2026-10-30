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
    clippy::cast_possible_truncation,
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
/// Maximum logical conditional-lookup work of a point or posterior request.
/// Each assignment charges one lookup plus every parent-index operation per
/// node, and every evidence comparison. The posterior budget includes its point
/// evaluation. This is a resource bound, not a measured timing guarantee.
pub const MAX_ENUMERATION_WORK: usize = 100_000_000;
/// Posterior-draw counts the calibration measured, ascending. An execution is
/// governed by the largest class its draw count reaches ([`draw_class`]); the
/// first is the supported minimum.
pub const MEASURED_DRAW_CLASSES: [u32; 2] = [2000, 8000];

fn unsupported(message: &'static str) -> EstimationError {
    EstimationError::Unsupported { message }
}

fn check_cancelled() -> Result<(), EstimationError> {
    if antecedent_core::numerical_admission::is_cancelled() {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("transport_budget_cancel"),
            "count_fitted_posterior.cancelled",
            "count-fitted posterior request was cancelled",
        ));
    }
    Ok(())
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
    /// Monte Carlo standard error of the lower endpoint: the posterior
    /// quantile is estimated from `draws` independent draws, so this is
    /// numerical error of the report, not posterior uncertainty.
    pub lower_mcse: f64,
    /// Monte Carlo standard error of the upper endpoint.
    pub upper_mcse: f64,
}

impl PosteriorInterval {
    /// The larger endpoint Monte Carlo standard error as a fraction of the
    /// interval width; infinite for an interval of zero width with nonzero
    /// error, zero when both are zero.
    #[must_use]
    pub fn mcse_width_fraction(&self) -> f64 {
        endpoint_mcse_width_fraction(self.lower_mcse, self.upper_mcse, self.upper - self.lower)
    }
}

/// Largest accepted ratio of an interval endpoint's Monte Carlo standard error to the
/// interval width. The measured count-fitted design sits near 0.015 at the minimum draw
/// count; five times that or more is a sign the draws, not the posterior, set the
/// endpoints.
pub const MAX_ENDPOINT_MCSE_FRACTION: f64 = 0.05;

/// The larger endpoint Monte Carlo standard error as a fraction of the interval `width`;
/// infinite for an interval of zero width with nonzero error, zero when both errors are
/// zero.
#[must_use]
pub fn endpoint_mcse_width_fraction(lower_mcse: f64, upper_mcse: f64, width: f64) -> f64 {
    let worst = lower_mcse.max(upper_mcse);
    if worst <= 0.0 { 0.0 } else { worst / width }
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
    /// A parent that is not earlier or is repeated, a table of the wrong shape, a declared row that is
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
            let mut seen = vec![false; index];
            for &parent in &node.parents {
                if seen[parent] {
                    return Err(unsupported("count-fitted model: duplicate parent"));
                }
                seen[parent] = true;
            }
            let expected = node
                .parents
                .iter()
                .try_fold(node.cardinality, |size, &parent| {
                    size.checked_mul(nodes[parent].cardinality)
                })
                .ok_or_else(|| {
                    unsupported("count-fitted model: conditional table shape overflows")
                })?;
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
    fn conditional(
        &self,
        tables: &[Vec<f64>],
        query: &PosteriorQuery,
    ) -> Result<Option<f64>, EstimationError> {
        let n = self.nodes.len();
        let mut assignment = vec![0_usize; n];
        let (mut joint, mut evidence_mass) = (0.0_f64, 0.0_f64);
        let mut visited = 0usize;
        loop {
            if visited % 4096 == 0 {
                check_cancelled()?;
            }
            visited += 1;
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
                    return Ok((evidence_mass > 0.0).then(|| joint / evidence_mass));
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

    fn check_work(
        &self,
        query: &PosteriorQuery,
        evaluations: usize,
    ) -> Result<(), EstimationError> {
        check_cancelled()?;
        let assignments =
            self.nodes.iter().try_fold(1usize, |n, node| n.checked_mul(node.cardinality));
        let per_assignment = self.nodes.iter().try_fold(query.evidence.len(), |work, node| {
            work.checked_add(node.parents.len())?.checked_add(1)
        });
        let work = assignments
            .zip(per_assignment)
            .and_then(|(n, per)| n.checked_mul(per))
            .and_then(|per| per.checked_mul(evaluations));
        if work.is_none_or(|work| work > MAX_ENUMERATION_WORK) {
            return Err(crate::util::refuse(
                antecedent_core::reason_code!("transport_budget_cancel"),
                "count_fitted_posterior.work_limit",
                "count-fitted posterior request exceeds the logical enumeration-work limit",
            ));
        }
        Ok(())
    }

    /// The point value under the posterior-mean tables (counts plus `prior_alpha`).
    ///
    /// # Errors
    /// An out-of-range query, evidence of zero probability, cancellation, or work
    /// exceeding [`MAX_ENUMERATION_WORK`].
    pub fn point(&self, query: &PosteriorQuery, prior_alpha: f64) -> Result<f64, EstimationError> {
        self.validate_query(query)?;
        self.check_work(query, 1)?;
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
        self.conditional(&tables, query)?
            .ok_or_else(|| unsupported("count-fitted posterior: the evidence has zero probability"))
    }

    /// Equal-tailed posterior interval of `P(target = level | evidence)`.
    ///
    /// `seed` fixes the Monte Carlo stream; the result is a deterministic function of
    /// the counts, the query, the prior, the level, the draw count and the seed.
    ///
    /// # Errors
    /// A level outside `(0, 1)`, a non-positive prior, a draw count outside
    /// `[2, MAX_DRAWS]`, an out-of-range query, evidence of zero probability,
    /// cancellation, or work exceeding [`MAX_ENUMERATION_WORK`].
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
        self.check_work(query, draws as usize + 1)?;
        let point = self.point(query, prior_alpha)?;
        let mut rng = SplitMix(seed ^ 0x9e37_79b9_7f4a_7c15);
        let _admitted = crate::util::admit_bound_array::<f64>(draws as usize, 1)?;
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
            check_cancelled()?;
            for (node, table) in self.nodes.iter().zip(tables.iter_mut()) {
                if let NodeTable::Counts(counts) = &node.table {
                    for (row_counts, row) in
                        counts.chunks(node.cardinality).zip(table.chunks_mut(node.cardinality))
                    {
                        // Normalise in log space: a gamma draw of shape below one underflows to
                        // zero with substantial probability (shape 1e-3: nearly half of the
                        // draws), and a row of zeros has no normalisation.
                        for (slot, &c) in row.iter_mut().zip(row_counts) {
                            *slot = rng.ln_gamma_draw(c as f64 + prior_alpha);
                        }
                        let top = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let mut total = 0.0;
                        for slot in row.iter_mut() {
                            *slot = (*slot - top).exp();
                            total += *slot;
                        }
                        for slot in row.iter_mut() {
                            *slot /= total;
                        }
                    }
                }
            }
            let value = self.conditional(&tables, query)?.ok_or_else(|| {
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
        let (lower_p, upper_p) = ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0);
        Ok(PosteriorInterval {
            point,
            lower: quantile(lower_p),
            upper: quantile(upper_p),
            level,
            rows: self.rows,
            draws,
            lower_mcse: quantile_mcse(&values, lower_p),
            upper_mcse: quantile_mcse(&values, upper_p),
        })
    }
}

/// Monte Carlo standard error of the `p`-quantile of `sorted` independent draws.
///
/// The rank of the sample quantile has standard deviation `sqrt(m p (1 - p))`, so the
/// quantile's standard error is half the spread of the draws between the ranks one
/// standard deviation either side of `m p` (the distribution-free order-statistic
/// estimate; it needs no density estimate at the quantile).
#[must_use]
pub fn quantile_mcse(sorted: &[f64], p: f64) -> f64 {
    let m = sorted.len() as f64;
    let spread = (m * p * (1.0 - p)).sqrt();
    let last = sorted.len() - 1;
    #[allow(
        clippy::cast_sign_loss,
        reason = "the ranks are clamped to [0, len - 1] before the conversion"
    )]
    let rank = |h: f64| h.clamp(0.0, last as f64) as usize;
    (sorted[rank((m * p + spread).ceil())] - sorted[rank((m * p - spread).floor())]) / 2.0
}

/// The measured draw class of an execution with `draws` posterior draws: the largest
/// [`MEASURED_DRAW_CLASSES`] entry not above `draws` (the first below the supported
/// minimum, which is refused before it is looked up).
#[must_use]
pub fn draw_class(draws: u32) -> u32 {
    MEASURED_DRAW_CLASSES
        .iter()
        .rev()
        .copied()
        .find(|&class| class <= draws)
        .unwrap_or(MEASURED_DRAW_CLASSES[0])
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

    /// `ln` of a Gamma(shape, 1) draw by Marsaglia and Tsang, with the boost for shape below
    /// one done on the logarithm (`ln G(shape + 1) + ln(u) / shape`), so a small shape never
    /// underflows.
    fn ln_gamma_draw(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.uniform();
            return self.ln_gamma_draw(shape + 1.0) + u.ln() / shape;
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
                return d.ln() + v.ln();
            }
        }
    }
}

/// The calibration basis of a count-fitted posterior interval at `rows` fitted rows.
///
/// `draws` selects the measured draw class ([`draw_class`]). One definition serves the harness and any facade, so a record can only describe the
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
    // The supported minimum keeps its original key; each larger measured draw class is
    // its own construction, so a record at 8000 draws never gates 2000-draw executions.
    let class = draw_class(draws);
    let posterior = if class == MEASURED_DRAW_CLASSES[0] {
        a("dirichlet.uniform_prior")
    } else {
        Arc::from(format!("dirichlet.uniform_prior.draws_{class}"))
    };
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
            posterior,
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

#[cfg(test)]
#[allow(
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    reason = "small closed-form oracle arithmetic over exponents below twelve"
)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_parents_are_refused_before_table_shape_arithmetic() {
        let result = CountFittedModel::new(
            vec![
                CountNode { cardinality: 2, parents: vec![], table: NodeTable::Counts(vec![1, 1]) },
                CountNode {
                    cardinality: 2,
                    parents: vec![0; 64],
                    table: NodeTable::Counts(vec![]),
                },
            ],
            2,
        );
        assert!(result.unwrap_err().to_string().contains("duplicate parent"));
    }

    #[test]
    fn posterior_enumeration_work_is_bounded_before_point_evaluation() {
        let model = CountFittedModel::new(
            (0..20)
                .map(|_| CountNode {
                    cardinality: 2,
                    parents: vec![],
                    table: NodeTable::Declared(vec![0.5, 0.5]),
                })
                .collect(),
            2,
        )
        .unwrap();
        let query = PosteriorQuery { target: 0, level: 1, evidence: vec![] };
        let error = model.posterior_interval(&query, 0.95, 1.0, MAX_DRAWS as u32, 1).unwrap_err();
        assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
        assert!(error.to_string().contains("work_limit"));

        let nodes = (0..3)
            .map(|i| CountNode {
                cardinality: 2,
                parents: (0..i).collect(),
                table: NodeTable::Declared(vec![0.5; 2usize.pow((i + 1) as u32)]),
            })
            .collect();
        let model = CountFittedModel::new(nodes, 2).unwrap();
        // Eight assignments, charging 1+2+3 parent-index/lookup operations.
        let allowed = MAX_ENUMERATION_WORK / 48;
        assert!(model.check_work(&query, allowed).is_ok());
        assert!(model.check_work(&query, allowed + 1).is_err());
        assert!((model.point(&query, 1.0).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn count_fitted_requests_preserve_workflow_cancellation() {
        let (model, query) = beta_four_eight();
        let context = antecedent_core::ExecutionContext::for_tests(1);
        context.cancellation.cancel();
        let error = antecedent_core::numerical_admission::with_context(&context, || {
            model.posterior_interval(&query, 0.95, 1.0, 20, 1)
        })
        .unwrap_err();
        assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
        assert!(error.to_string().contains("cancelled"));
    }

    /// A single node counted `[7, 3]` (three in level 1) under the uniform prior: `P(Y = 1)` is exactly
    /// `Beta(4, 8)`, whose quantiles and density are closed forms.
    fn beta_four_eight() -> (CountFittedModel, PosteriorQuery) {
        let nodes = vec![CountNode {
            cardinality: 2,
            parents: vec![],
            table: NodeTable::Counts(vec![7, 3]),
        }];
        (
            CountFittedModel::new(nodes, 10).expect("model"),
            PosteriorQuery { target: 0, level: 1, evidence: vec![] },
        )
    }

    /// Regularised incomplete beta `I_x(4, 8)` as the upper binomial tail with
    /// `n = 11` trials: `P(Bin(11, x) >= 4)`.
    fn beta_cdf(x: f64) -> f64 {
        let choose =
            |k: u32| (1..=k).fold(1.0, |acc, i| acc * f64::from(11 - k + i) / f64::from(i));
        (4..=11_u32).map(|j| choose(j) * x.powi(j as i32) * (1.0 - x).powi(11 - j as i32)).sum()
    }

    fn beta_pdf(x: f64) -> f64 {
        // B(4, 8) = 3! 7! / 11!
        let beta = 6.0 * 5040.0 / 39_916_800.0;
        x.powi(3) * (1.0 - x).powi(7) / beta
    }

    fn beta_quantile(p: f64) -> f64 {
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if beta_cdf(mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    #[test]
    fn the_reported_endpoint_error_matches_the_closed_form_quantile_error() {
        let (model, query) = beta_four_eight();
        for draws in [2000_u32, 8000, 32_000] {
            let interval =
                model.posterior_interval(&query, 0.95, 1.0, draws, 11).expect("interval");
            for (p, reported, endpoint) in [
                (0.025, interval.lower_mcse, interval.lower),
                (0.975, interval.upper_mcse, interval.upper),
            ] {
                let truth = beta_quantile(p);
                // sqrt(p (1 - p) / m) / f(q): the standard error of a sample quantile.
                let oracle = (p * (1.0 - p) / f64::from(draws)).sqrt() / beta_pdf(truth);
                assert!(
                    (0.5 * oracle..=1.6 * oracle).contains(&reported),
                    "draws {draws} p {p}: reported {reported} vs closed form {oracle}"
                );
                // The endpoint itself is within a few standard errors of the true quantile.
                assert!((endpoint - truth).abs() < 4.0 * oracle, "draws {draws} p {p}");
            }
        }
    }

    #[test]
    fn more_draws_shrink_the_endpoint_error_and_its_width_fraction() {
        let (model, query) = beta_four_eight();
        let fraction = |draws| {
            model
                .posterior_interval(&query, 0.95, 1.0, draws, 3)
                .expect("interval")
                .mcse_width_fraction()
        };
        let (small, large) = (fraction(2000), fraction(32_000));
        assert!(small > large, "{small} vs {large}");
        assert!(small < 0.05, "the supported minimum stays inside the declared tolerance: {small}");
    }

    #[test]
    fn a_degenerate_interval_has_no_error_fraction_to_report() {
        let interval = PosteriorInterval {
            point: 0.5,
            lower: 0.5,
            upper: 0.5,
            level: 0.95,
            rows: 0,
            draws: 2,
            lower_mcse: 0.0,
            upper_mcse: 0.0,
        };
        assert!((interval.mcse_width_fraction() - 0.0).abs() < f64::EPSILON);
        let noisy = PosteriorInterval { upper_mcse: 0.1, ..interval };
        assert!(noisy.mcse_width_fraction().is_infinite());
    }

    #[test]
    fn each_measured_draw_class_is_its_own_construction() {
        assert_eq!(MEASURED_DRAW_CLASSES[0], 2000);
        for (draws, class) in [(2000, 2000), (3000, 2000), (4000, 2000), (7999, 2000)] {
            assert_eq!(draw_class(draws), class, "{draws}");
        }
        for draws in [8000, 8001, 50_000, 100_000] {
            assert_eq!(draw_class(draws), 8000, "{draws}");
        }
        // Below the minimum is refused before lookup; the class is still defined.
        assert_eq!(draw_class(1999), 2000);

        let posterior = |draws| calibration_basis(300, 0.95, draws).posterior;
        assert_eq!(&*posterior(2000), "dirichlet.uniform_prior");
        assert_eq!(posterior(3000), posterior(2000));
        assert_eq!(&*posterior(8000), "dirichlet.uniform_prior.draws_8000");
        assert_eq!(posterior(50_000), posterior(8000));
        assert_eq!(calibration_basis(300, 0.95, 3000).posterior_draws, Some(3000));
    }
}
