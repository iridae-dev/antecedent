//! Trial-to-target IPW under a declared sampling design: observation weights, whole clusters,
//! or the consecutive times of one series.
//!
//! The contrast is the ratio of two weighted sums over all `n` rows,
//! `ipw = Σ_i w_i ψ_i / Σ_i w_i (1 − S_i)` with trial membership `S_i`, the trial score
//! `ψ_i = S_i · (±1) · (1 − s_i) / (s_i · e_i^±) · y_i`, selection probability `s_i` and
//! treatment probability `e_i`, both known. The row weight `w_i` is the number of population
//! units the row stands for (`1` when the rows are unweighted); a row of weight zero stands for
//! no unit, is not read, and leaves every number as if it had been deleted.
//!
//! * **Variance.** The linearization of the ratio has the per-row influence
//!   `w_i (ψ_i − ipw (1 − S_i))`. The independent-row error is the root of its sum of squares
//!   over `Σ w_i (1 − S_i)`; a declared cluster is the primary sampling unit, so the cluster
//!   error sums the influence inside each cluster before squaring and carries the Arellano
//!   factor `G / (G − 1)` of the repo's other cluster-robust errors; a dependent series takes
//!   the Newey–West (Bartlett) long-run variance of the influence. With unit weights the
//!   independent-row and series errors are [`crate::transport::trial_to_target_ipw_se`] and
//!   [`crate::transport::trial_to_target_ipw_hac_se`], and with every row its own cluster the
//!   cluster error is the independent-row error times `sqrt(n / (n − 1))`.
//! * **Bayesian bootstrap.** Each draw puts one `Exp(1)` weight on every row of the trial (or on
//!   every cluster, or every contiguous block of rows, shared by the trial rows inside it),
//!   multiplies it by the row's sampling weight, and re-forms the trial mean of the scores; the
//!   trial and target weight totals stay fixed. Without sampling weights this is the Dirichlet
//!   law over the trial rows, clusters or blocks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::CausalRng;
use antecedent_identify::TransportIdentification;

use crate::EstimationError;
use crate::transport::{TransportOverlapReport, diagnostic, require_dahabreh_compatible_formula};
use crate::util::DrawSpec;

/// How the rows depend on each other.
#[derive(Clone, Copy, Debug)]
pub enum RowDependence<'a> {
    /// Independent rows.
    Independent,
    /// The rows are the consecutive times of one series: the Newey–West lag of the error and
    /// the block length of the bootstrap.
    SeriesBlocks(usize),
    /// Cluster label of each row (aligned to the rows); a whole cluster is the sampling unit.
    Clusters(&'a [u32]),
}

/// The aligned columns of one trial-to-target contrast and the weights of its rows.
#[derive(Clone, Copy, Debug)]
pub struct TrialToTargetRows<'a> {
    /// Outcome (read on trial rows only).
    pub outcome: &'a [f64],
    /// Treatment (read on trial rows only).
    pub treatment: &'a [bool],
    /// Trial membership.
    pub trial: &'a [bool],
    /// P(S = 1 | X) on every row.
    pub selection_probability: &'a [f64],
    /// P(A = 1 | X, S = 1) on trial rows.
    pub treatment_probability: &'a [f64],
    /// Sampling weight of every row; `None` is all ones.
    pub weights: Option<&'a [f64]>,
}

/// The rows that are read (weight above zero), and the trial and target weight totals.
struct Active {
    read: Vec<bool>,
    trial_weight: f64,
    target_weight: f64,
}

impl TrialToTargetRows<'_> {
    fn weight(&self, row: usize) -> f64 {
        self.weights.map_or(1.0, |weights| weights[row])
    }

    /// Lengths, weights, probabilities and outcomes of the rows that are read.
    fn validate(&self, dependence: RowDependence<'_>) -> Result<Active, EstimationError> {
        let n = self.outcome.len();
        if n == 0
            || self.treatment.len() != n
            || self.trial.len() != n
            || self.selection_probability.len() != n
            || self.treatment_probability.len() != n
            || self.weights.is_some_and(|weights| weights.len() != n)
        {
            return Err(EstimationError::data_msg("transport input length mismatch"));
        }
        if let Some(weights) = self.weights {
            if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return Err(crate::sampling_weights::invalid(
                    "transport sampling weights must be finite and non-negative",
                ));
            }
        }
        if let RowDependence::Clusters(ids) = dependence {
            if ids.len() != n {
                return Err(EstimationError::data_msg("transport cluster labels length mismatch"));
            }
        }
        let mut read = vec![true; n];
        let (mut trial_weight, mut target_weight) = (0.0, 0.0);
        for i in 0..n {
            let w = self.weight(i);
            if w <= 0.0 {
                read[i] = false;
                continue;
            }
            let (s, e) = (self.selection_probability[i], self.treatment_probability[i]);
            let treatment_out_of_range = self.trial[i] && (!e.is_finite() || e <= 0.0 || e >= 1.0);
            if !s.is_finite() || s <= 0.0 || s >= 1.0 || treatment_out_of_range {
                return Err(EstimationError::data_msg(
                    "selection and treatment probabilities must lie strictly inside (0,1)",
                ));
            }
            if self.trial[i] {
                if !self.outcome[i].is_finite() {
                    return Err(EstimationError::data_msg("trial outcomes must be finite"));
                }
                trial_weight += w;
            } else {
                target_weight += w;
            }
        }
        if trial_weight <= 0.0 || target_weight <= 0.0 {
            return Err(EstimationError::data_msg(
                "transport requires source-trial and target-population rows",
            ));
        }
        if let RowDependence::Clusters(ids) = dependence {
            let mut trial_clusters = std::collections::BTreeSet::new();
            let mut clusters = std::collections::BTreeSet::new();
            for i in (0..n).filter(|&i| read[i]) {
                clusters.insert(ids[i]);
                if self.trial[i] {
                    trial_clusters.insert(ids[i]);
                }
            }
            if clusters.len() < 2 || trial_clusters.len() < 2 {
                return Err(EstimationError::data_msg(
                    "a cluster-robust transport error needs at least two clusters, and the trial \
                     rows in at least two of them: one cluster carries no between-cluster variation",
                ));
            }
        }
        Ok(Active { read, trial_weight, target_weight })
    }

    /// Score `ψ_i` of a trial row.
    fn score(&self, row: usize) -> f64 {
        let (s, e) = (self.selection_probability[row], self.treatment_probability[row]);
        let arm = if self.treatment[row] { e } else { 1.0 - e };
        let sign = if self.treatment[row] { 1.0 } else { -1.0 };
        sign * (1.0 - s) / s / arm * self.outcome[row]
    }

    /// The weighted trial-to-target IPW contrast.
    ///
    /// # Errors
    ///
    /// An uncertified identification, or invalid rows or weights.
    pub fn ipw(&self, identification: &TransportIdentification) -> Result<f64, EstimationError> {
        require_dahabreh_compatible_formula(identification, "trial-to-target effect")?;
        let active = self.validate(RowDependence::Independent)?;
        let numerator: f64 = (0..self.outcome.len())
            .filter(|&i| active.read[i] && self.trial[i])
            .map(|i| self.weight(i) * self.score(i))
            .sum();
        Ok(numerator / active.target_weight)
    }

    /// Selection and treatment overlap of the rows that are read, the weights of each being the
    /// inverse-probability weight times the row's sampling weight.
    ///
    /// # Errors
    ///
    /// Invalid rows or weights.
    pub fn overlap(&self) -> Result<TransportOverlapReport, EstimationError> {
        self.validate(RowDependence::Independent)?;
        let (mut selection_p, mut treatment_p) = (Vec::new(), Vec::new());
        let (mut selection_w, mut treatment_w) = (Vec::new(), Vec::new());
        for i in (0..self.outcome.len()).filter(|&i| self.trial[i] && self.weight(i) > 0.0) {
            let (s, e) = (self.selection_probability[i], self.treatment_probability[i]);
            let arm = if self.treatment[i] { e } else { 1.0 - e };
            selection_p.push(s);
            treatment_p.push(e);
            selection_w.push(self.weight(i) * (1.0 - s) / s);
            treatment_w.push(self.weight(i) / arm);
        }
        Ok(TransportOverlapReport {
            selection: diagnostic(&selection_p, &selection_w),
            treatment: diagnostic(&treatment_p, &treatment_w),
        })
    }

    /// Standard error of [`Self::ipw`] under `dependence`: the linearized ratio error described
    /// in the module docs.
    ///
    /// # Errors
    ///
    /// Invalid rows, weights or clusters, or a non-finite `ipw`.
    pub fn ipw_se(&self, ipw: f64, dependence: RowDependence<'_>) -> Result<f64, EstimationError> {
        let active = self.validate(dependence)?;
        if !ipw.is_finite() {
            return Err(EstimationError::data_msg("transport IPW estimate must be finite"));
        }
        let n = self.outcome.len();
        let influence: Vec<f64> = (0..n)
            .map(|i| {
                if !active.read[i] {
                    0.0
                } else if self.trial[i] {
                    self.weight(i) * self.score(i)
                } else {
                    -ipw * self.weight(i)
                }
            })
            .collect();
        let variance = match dependence {
            RowDependence::Independent => influence.iter().map(|v| v * v).sum::<f64>(),
            RowDependence::SeriesBlocks(lag) => {
                let mut variance: f64 = influence.iter().map(|v| v * v).sum();
                for k in 1..=lag.min(n.saturating_sub(1)) {
                    let weight = 1.0 - k as f64 / (lag as f64 + 1.0);
                    let cross: f64 =
                        influence.iter().zip(&influence[k..]).map(|(a, b)| a * b).sum();
                    variance += 2.0 * weight * cross;
                }
                variance.max(0.0)
            }
            RowDependence::Clusters(ids) => {
                // The influence sums to zero over the rows, so the cluster sums are not
                // re-centred.
                let rows: Vec<usize> = (0..n).filter(|&i| active.read[i]).collect();
                let values: Vec<f64> = rows.iter().map(|&i| influence[i]).collect();
                let labels: Vec<u32> = rows.iter().map(|&i| ids[i]).collect();
                let (meat, groups) = antecedent_stats::cluster_meat_scalar(&values, &labels, 0.0)
                    .map_err(|_| {
                    EstimationError::data_msg("transport cluster influence failed to form meat")
                })?;
                let g = groups as f64;
                g / (g - 1.0) * meat
            }
        };
        Ok(variance.sqrt() / active.target_weight)
    }

    /// Bayesian-bootstrap draws of the contrast: see the module docs. The certificate is
    /// checked anew.
    ///
    /// # Errors
    ///
    /// An uncertified identification, invalid rows, weights or clusters, fewer than two draws,
    /// or a trial without both treatment arms among the rows that are read.
    pub fn bayesian_bootstrap(
        &self,
        identification: &TransportIdentification,
        dependence: RowDependence<'_>,
        sampling: DrawSpec,
    ) -> Result<Vec<f64>, EstimationError> {
        let DrawSpec { draws: n_draws, seed } = sampling;
        require_dahabreh_compatible_formula(identification, "Bayesian trial-to-target effect")?;
        let active = self.validate(dependence)?;
        if n_draws < 2 {
            return Err(EstimationError::data_msg(
                "Bayesian trial-to-target effect requires at least two draws",
            ));
        }
        let n = self.outcome.len();
        let source_rows: Vec<usize> = (0..n).filter(|&i| active.read[i] && self.trial[i]).collect();
        if !source_rows.iter().any(|&i| self.treatment[i])
            || !source_rows.iter().any(|&i| !self.treatment[i])
        {
            return Err(EstimationError::data_msg(
                "Bayesian trial-to-target effect requires both observed treatment arms",
            ));
        }
        let scores: Vec<f64> = source_rows.iter().map(|&i| self.score(i)).collect();
        let row_weights: Vec<f64> = source_rows.iter().map(|&i| self.weight(i)).collect();
        // Dense cluster number of every row that is read, in order of first appearance.
        let mut cluster_index: BTreeMap<u32, usize> = BTreeMap::new();
        let mut row_cluster = vec![0_usize; n];
        if let RowDependence::Clusters(ids) = dependence {
            for &row in &source_rows {
                let next = cluster_index.len();
                row_cluster[row] = *cluster_index.entry(ids[row]).or_insert(next);
            }
        }
        let mut rng = CausalRng::from_seed(seed);
        let _admitted = crate::util::admit_bound_array::<f64>(n_draws, 1)?;
        let mut draws = Vec::with_capacity(n_draws);
        for _ in 0..n_draws {
            let shared = match dependence {
                RowDependence::Independent => antecedent_kernels::bayesian_bootstrap_weights(
                    &mut rng,
                    source_rows.len(),
                    false,
                ),
                RowDependence::SeriesBlocks(block) => {
                    let block = block.clamp(1, n.max(1));
                    let blocks = antecedent_kernels::bayesian_bootstrap_weights(
                        &mut rng,
                        n.div_ceil(block),
                        false,
                    );
                    source_rows.iter().map(|&row| blocks[row / block]).collect()
                }
                RowDependence::Clusters(_) => {
                    let clusters = antecedent_kernels::bayesian_bootstrap_weights(
                        &mut rng,
                        cluster_index.len(),
                        false,
                    );
                    source_rows.iter().map(|&row| clusters[row_cluster[row]]).collect()
                }
            };
            let weights: Vec<f64> = if self.weights.is_some() {
                shared.iter().zip(&row_weights).map(|(g, w)| g * w).collect()
            } else {
                shared
            };
            let normalizer: f64 = weights.iter().sum();
            if !normalizer.is_finite() || normalizer <= 0.0 {
                return Err(EstimationError::stats_msg(
                    "Bayesian trial-to-target row weights failed to normalize",
                ));
            }
            let numerator: f64 = weights.iter().zip(&scores).map(|(w, score)| w * score).sum();
            let draw = numerator * active.trial_weight / (normalizer * active.target_weight);
            if !draw.is_finite() {
                return Err(EstimationError::stats_msg(
                    "Bayesian trial-to-target effect has a non-finite draw",
                ));
            }
            draws.push(draw);
        }
        Ok(draws)
    }
}
