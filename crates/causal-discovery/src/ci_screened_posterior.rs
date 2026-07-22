//! Candidate-edge posterior after CI screening.
//!
//! Runs static PC (or a caller-supplied CI test) to obtain an undirected
//! skeleton, then structure-MCMC restricted to those pairs. Optional
//! Bayes-factor / posterior-dependence soft proposal weights are recorded as
//! diagnostics notes (not hard truth).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::too_many_lines)]

use std::collections::HashSet;
use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::TabularData;
use causal_state::GraphScoreFamily;
use causal_stats::{
    BayesFactorCi, ConditionalIndependence, FdrAdjustment, PartialCorrelation,
    PosteriorDependenceCi,
};

use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::graph_posterior::{GraphPosterior, GraphPosteriorEngine, GraphPrior};
use crate::pc::Pc;
use crate::structure_mcmc::StructureMcmc;

/// Soft CI weight source for screened proposals (informational / proposal bias).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum CiSoftWeight {
    /// No soft weights (uniform among screened candidates).
    #[default]
    None,
    /// Log Bayes factor for dependence (`BayesFactorCi` statistic).
    BayesFactor,
    /// Posterior probability of dependence.
    PosteriorDependence,
}

/// CI-screened candidate-edge posterior (PC skeleton → structure MCMC).
#[derive(Clone)]
pub struct CiScreenedPosterior {
    /// Constraints shared by PC screening and MCMC prior.
    pub constraints: DiscoveryConstraints,
    /// CI test for PC skeleton screening.
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// FDR for PC (`None` = off).
    pub fdr: Option<FdrAdjustment>,
    /// Soft weight diagnostic (does not replace the Gaussian-BIC posterior).
    pub soft_weight: CiSoftWeight,
    /// Structure MCMC schedule.
    pub mcmc: StructureMcmc,
}

impl std::fmt::Debug for CiScreenedPosterior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CiScreenedPosterior")
            .field("constraints", &self.constraints)
            .field("ci", &"<dyn ConditionalIndependence>")
            .field("fdr", &self.fdr)
            .field("soft_weight", &self.soft_weight)
            .field("mcmc", &self.mcmc)
            .finish()
    }
}

impl Default for CiScreenedPosterior {
    fn default() -> Self {
        Self::new()
    }
}

impl CiScreenedPosterior {
    /// Default: `ParCorr` PC screen + structure MCMC.
    #[must_use]
    pub fn new() -> Self {
        Self {
            constraints: DiscoveryConstraints {
                temporal: crate::constraints::TemporalConstraints {
                    max_lag: causal_core::Lag::CONTEMPORANEOUS,
                    min_lag: causal_core::Lag::CONTEMPORANEOUS,
                },
                ..DiscoveryConstraints::default()
            },
            ci: Arc::new(PartialCorrelation),
            fdr: Some(FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            soft_weight: CiSoftWeight::None,
            mcmc: StructureMcmc::new().with_schedule(2, 300, 600, 1),
        }
    }

    /// Attach constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// CI test for screening.
    #[must_use]
    pub fn with_ci(mut self, ci: Arc<dyn ConditionalIndependence + Send + Sync>) -> Self {
        self.ci = ci;
        self
    }

    /// Soft-weight diagnostic mode.
    #[must_use]
    pub fn with_soft_weight(mut self, soft: CiSoftWeight) -> Self {
        self.soft_weight = soft;
        self
    }

    /// MCMC schedule.
    #[must_use]
    pub fn with_mcmc(mut self, mcmc: StructureMcmc) -> Self {
        self.mcmc = mcmc;
        self
    }

    /// Run screened posterior search.
    ///
    /// # Errors
    ///
    /// PC, MCMC, or empty-skeleton failures.
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        let mut prior = prior.clone();
        prior.constraints = self.constraints.clone();

        let pairs = pc_skeleton_pairs(self, data, variables, workspace, ctx)?;
        if pairs.is_empty() {
            return Err(DiscoveryError::unsupported("CI screening produced empty skeleton"));
        }

        let soft_note = soft_weight_note(self.soft_weight, data, variables, &pairs, ctx)?;

        let pair_arc: Arc<[(u32, u32)]> = Arc::from(pairs);
        let mcmc = self.mcmc.clone().with_candidate_pairs(Arc::clone(&pair_arc));
        let mut post = mcmc.run(data, variables, &prior, score_family, workspace, ctx)?;
        if let Some(note) = soft_note {
            post.diagnostics.notes.push(note);
        }
        post.diagnostics.notes.push(Arc::from(format!("ci_screened_pairs={}", pair_arc.len())));
        Ok(post)
    }
}

impl GraphPosteriorEngine for CiScreenedPosterior {
    fn infer_graphs(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        let mut ws = DiscoveryWorkspace::default();
        self.run(data, variables, prior, score_family, &mut ws, ctx)
    }
}

fn pc_skeleton_pairs(
    eng: &CiScreenedPosterior,
    data: &TabularData,
    variables: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<Vec<(u32, u32)>, DiscoveryError> {
    let pc = Pc::new()
        .with_constraints(eng.constraints.clone())
        .with_fdr_adjustment(eng.fdr)
        .with_ci(Arc::clone(&eng.ci));
    let result = pc.run(data, variables, workspace, ctx)?;
    let mut pairs = HashSet::new();
    for e in result.evidence.graph.edges() {
        let lo = e.a.raw().min(e.b.raw());
        let hi = e.a.raw().max(e.b.raw());
        pairs.insert((lo, hi));
    }
    let mut v: Vec<_> = pairs.into_iter().collect();
    v.sort_unstable();
    Ok(v)
}

fn soft_weight_note(
    mode: CiSoftWeight,
    data: &TabularData,
    variables: &[VariableId],
    pairs: &[(u32, u32)],
    ctx: &ExecutionContext,
) -> Result<Option<Arc<str>>, DiscoveryError> {
    if matches!(mode, CiSoftWeight::None) || pairs.is_empty() {
        return Ok(None);
    }
    let cols = crate::pc::collect_float_columns(data, variables)?;
    let n = cols[0].len();
    let col_refs: Vec<&[f64]> = cols.iter().map(std::convert::AsRef::as_ref).collect();
    let mut ws = causal_stats::CiWorkspace::default();
    let mut scores = Vec::with_capacity(pairs.len());
    for &(lo, hi) in pairs {
        let q = causal_stats::CiQuery { x: lo as usize, y: hi as usize, z_start: 0, z_len: 0 };
        let req = causal_stats::CiBatchRequest {
            columns: &col_refs,
            queries: &[q],
            z_flat: &[],
            significance: causal_stats::SignificanceMethod::Analytic,
            confidence: causal_stats::ConfidenceMethod::None,
        };
        let out = match mode {
            CiSoftWeight::BayesFactor => {
                use causal_stats::ConditionalIndependenceTest;
                BayesFactorCi::new()
                    .test_batch_adhoc(&req, &mut ws, ctx)
                    .map_err(DiscoveryError::from)?
            }
            CiSoftWeight::PosteriorDependence => {
                use causal_stats::ConditionalIndependenceTest;
                PosteriorDependenceCi::new()
                    .test_batch_adhoc(&req, &mut ws, ctx)
                    .map_err(DiscoveryError::from)?
            }
            CiSoftWeight::None => unreachable!(),
        };
        scores.push(out.results[0].statistic);
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    Ok(Some(Arc::from(format!(
        "soft_weight={mode:?} n_pairs={} mean_stat={mean:.4} n_rows={n}",
        pairs.len()
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
    use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};

    fn chain_data(n_rows: usize) -> (TabularData, Vec<VariableId>) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["a", "b", "c"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let vars: Vec<_> = (0..3).map(VariableId::from_raw).collect();
        let mut rng = causal_core::CausalRng::from_seed(21);
        let mut a = vec![0.0; n_rows];
        let mut bb = vec![0.0; n_rows];
        let mut c = vec![0.0; n_rows];
        for i in 0..n_rows {
            a[i] = rng.next_f64() * 2.0 - 1.0;
            bb[i] = 1.5 * a[i] + 0.2 * (rng.next_f64() * 2.0 - 1.0);
            c[i] = 1.2 * bb[i] + 0.2 * (rng.next_f64() * 2.0 - 1.0);
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(vars[0], Arc::from(a), ValidityBitmap::all_valid(n_rows))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(vars[1], Arc::from(bb), ValidityBitmap::all_valid(n_rows))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(vars[2], Arc::from(c), ValidityBitmap::all_valid(n_rows))
                    .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        (TabularData::new(storage), vars)
    }

    #[test]
    fn screened_posterior_runs() {
        let (data, vars) = chain_data(180);
        let eng = CiScreenedPosterior::new()
            .with_soft_weight(CiSoftWeight::BayesFactor)
            .with_mcmc(StructureMcmc::new().with_schedule(2, 150, 300, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let post = eng
            .run(&data, &vars, &GraphPrior::uniform(), GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap();
        assert!(post.n_graphs >= 1);
        assert!(post.diagnostics.notes.iter().any(|n| n.contains("ci_screened_pairs")));
    }
}
