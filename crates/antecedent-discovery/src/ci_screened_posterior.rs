//! Candidate-edge posterior after CI screening.
//!
//! Runs static PC (or a caller-supplied CI test) to obtain an undirected
//! skeleton, then structure-MCMC restricted to those pairs. Optional
//! Bayes-factor / posterior-dependence soft proposal weights are recorded as
//! diagnostics notes (not hard truth).
//!
//! The screen is a frequentist filter, not an oracle: a pair the CI test fails
//! to reject is excluded from the MCMC proposal set, so its posterior marginal
//! is bounded by the search space, not by evidence against the edge. The
//! screening constraints and the caller's [`GraphPrior`] constraints are
//! merged (required/forbidden unioned, `max_parents` taken as the stricter of
//! the two), never one silently replacing the other; a real disagreement
//! (a link required by one side and forbidden by the other, or incompatible
//! tiers) is refused rather than resolved in the dark. When the caller states
//! an explicit nonzero prior edge-inclusion probability, screened-out pairs
//! keep a small floor probability instead of an exact 0.0 — this crate has no
//! Type-II / power estimate for the screening test, so the floor is the
//! documented conservative approximation `min(caller_prior, screen_alpha)`,
//! not a principled Bayesian update.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::too_many_lines)]

use std::collections::HashSet;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_state::GraphScoreFamily;
use antecedent_stats::{
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
                    max_lag: antecedent_core::Lag::CONTEMPORANEOUS,
                    min_lag: antecedent_core::Lag::CONTEMPORANEOUS,
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
        let effective = merge_screening_constraints(&self.constraints, &prior.constraints)?;
        let merged_caller_constraints = !prior.constraints.required.is_empty()
            || !prior.constraints.forbidden.is_empty()
            || prior.constraints.max_parents.is_some()
            || !prior.constraints.tiers.is_empty();

        let mut prior = prior.clone();
        prior.constraints = effective.clone();

        let pairs = pc_skeleton_pairs(&effective, self, data, variables, workspace, ctx)?;
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
        if merged_caller_constraints {
            post.diagnostics.notes.push(Arc::from(
                "ci_screened_prior_constraints_merged=caller required/forbidden/max_parents/tiers \
                 combined with screening constraints, not overwritten",
            ));
        }
        apply_screened_out_floor(&mut post, &effective, variables, &pair_arc, prior.edge_inclusion);
        Ok(post)
    }
}

/// Merge the engine's own screening constraints with a caller-supplied prior's
/// constraints.
///
/// `required` and `forbidden` are unioned rather than one replacing the
/// other, `max_parents` takes the stricter (smaller) bound when both sides
/// state one, and `tiers` is taken from whichever side declares it (or either,
/// if identical). All other fields (temporal window, alpha, significance
/// method, `mask_type`, …) are screening configuration and come from the
/// engine, since the caller's posterior request must use the same screen that
/// produced the candidate pairs.
///
/// # Errors
///
/// A link required by one side and forbidden by the other, or non-empty,
/// unequal tier partitions on both sides — these are real disagreements that
/// must be resolved by the caller, not silently arbitrated.
fn merge_screening_constraints(
    screen: &DiscoveryConstraints,
    caller: &DiscoveryConstraints,
) -> Result<DiscoveryConstraints, DiscoveryError> {
    let mut required: Vec<_> = screen.required.iter().copied().collect();
    for link in caller.required.iter() {
        if !required.contains(link) {
            required.push(*link);
        }
    }
    let mut forbidden: Vec<_> = screen.forbidden.iter().copied().collect();
    for link in caller.forbidden.iter() {
        if !forbidden.contains(link) {
            forbidden.push(*link);
        }
    }
    if required.iter().any(|r| forbidden.contains(r)) {
        return Err(DiscoveryError::unsupported(
            "CI-screened posterior: a link is required by one of the screening/prior \
             constraint sets and forbidden by the other",
        ));
    }
    let tiers = match (screen.tiers.is_empty(), caller.tiers.is_empty()) {
        (_, true) => screen.tiers.clone(),
        (true, false) => caller.tiers.clone(),
        (false, false) => {
            let same = screen.tiers.len() == caller.tiers.len()
                && screen
                    .tiers
                    .iter()
                    .zip(caller.tiers.iter())
                    .all(|(a, b)| a.as_ref() == b.as_ref());
            if !same {
                return Err(DiscoveryError::unsupported(
                    "CI-screened posterior: screening and prior constraints declare \
                     different, non-empty variable tiers",
                ));
            }
            screen.tiers.clone()
        }
    };
    let max_parents = match (screen.max_parents, caller.max_parents) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    Ok(DiscoveryConstraints {
        required: Arc::from(required),
        forbidden: Arc::from(forbidden),
        tiers,
        max_parents,
        ..screen.clone()
    })
}

/// Replace an exact-0.0 posterior marginal on a screened-out pair with a
/// small floor when the caller stated an explicit nonzero prior belief about
/// edge inclusion.
///
/// A pair the CI screen excludes from the MCMC proposal set can never be
/// sampled, so [`GraphPosterior::edge_marginals`] is identically 0.0 for it
/// regardless of what the prior said — that is a search-space artifact, not
/// evidence the edge is absent. Without a Type-II / power estimate for the
/// screening test (this crate does not compute one), the honest fallback is a
/// documented conservative floor rather than a silent hard zero:
/// `min(caller's stated edge_inclusion, screen alpha)`, split evenly across
/// both edge directions since screening does not determine orientation. Pairs
/// the caller (or the engine) explicitly forbade keep their exact 0.0 — that
/// zero was asked for, not imposed by the screen.
fn apply_screened_out_floor(
    post: &mut GraphPosterior,
    effective: &DiscoveryConstraints,
    variables: &[VariableId],
    screened_in: &[(u32, u32)],
    edge_inclusion_prior: Option<f64>,
) {
    let Some(p) = edge_inclusion_prior else { return };
    if p <= 0.0 {
        return;
    }
    let floor = p.min(effective.alpha);
    if floor <= 0.0 {
        return;
    }
    let n = variables.len();
    let screened_in: HashSet<(u32, u32)> = screened_in.iter().copied().collect();
    let forbidden_idx: HashSet<(u32, u32)> = effective
        .forbidden
        .iter()
        .filter_map(|link| {
            let si = variables.iter().position(|v| *v == link.source)?;
            let ti = variables.iter().position(|v| *v == link.target)?;
            Some((si.min(ti) as u32, si.max(ti) as u32))
        })
        .collect();

    let mut floored = 0usize;
    let mut marginals = post.edge_marginals.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let key = (i as u32, j as u32);
            if screened_in.contains(&key) || forbidden_idx.contains(&key) {
                continue;
            }
            let half = floor / 2.0;
            let cell_ij = i * n + j;
            let cell_ji = j * n + i;
            if marginals[cell_ij] < half {
                marginals[cell_ij] = half;
            }
            if marginals[cell_ji] < half {
                marginals[cell_ji] = half;
            }
            floored += 1;
        }
    }
    if floored > 0 {
        post.edge_marginals = Arc::from(marginals);
        post.diagnostics.notes.push(Arc::from(format!(
            "ci_screened_floor pairs={floored} floor={floor:.4} \
             (min of caller edge_inclusion prior and screen alpha; screened-out \
             pairs are reported as unlikely, not certainly absent)"
        )));
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
    constraints: &DiscoveryConstraints,
    eng: &CiScreenedPosterior,
    data: &TabularData,
    variables: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<Vec<(u32, u32)>, DiscoveryError> {
    let pc = Pc::new()
        .with_constraints(constraints.clone())
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
    let mut ws = antecedent_stats::CiWorkspace::default();
    let mut scores = Vec::with_capacity(pairs.len());
    for &(lo, hi) in pairs {
        let q = antecedent_stats::CiQuery { x: lo as usize, y: hi as usize, z_start: 0, z_len: 0 };
        let req = antecedent_stats::CiBatchRequest {
            columns: &col_refs,
            queries: &[q],
            z_flat: &[],
            significance: antecedent_stats::SignificanceMethod::Analytic,
            confidence: antecedent_stats::ConfidenceMethod::None,
        };
        let out = match mode {
            CiSoftWeight::BayesFactor => {
                use antecedent_stats::ConditionalIndependenceTest;
                BayesFactorCi::new()
                    .test_batch_adhoc(&req, &mut ws, ctx)
                    .map_err(DiscoveryError::from)?
            }
            CiSoftWeight::PosteriorDependence => {
                use antecedent_stats::ConditionalIndependenceTest;
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
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};

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
        let mut rng = antecedent_core::CausalRng::from_seed(21);
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

    /// `a—c` is screened out of the PC skeleton by the `a ⊥ c | b` chain
    /// separation. A caller who supplied a nonzero `edge_inclusion` prior for
    /// every pair believed something about that edge too; the CI screen's
    /// failure to reject independence is evidence, not proof, so the reported
    /// marginal must never be forced to an exact 0.0 regardless of that prior.
    #[test]
    fn screened_out_edge_keeps_nonzero_marginal_under_explicit_prior() {
        let (data, vars) = chain_data(180);
        let eng = CiScreenedPosterior::new()
            .with_mcmc(StructureMcmc::new().with_schedule(2, 150, 300, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let prior = GraphPrior::bernoulli_edges(0.3).unwrap();
        let post =
            eng.run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &mut ws, &ctx).unwrap();
        let n = 3;
        let ac = post.edge_marginals[2] + post.edge_marginals[2 * n];
        assert!(ac > 0.0, "screened-out a—c marginal reported as exact 0.0 (ac={ac})");
        assert!(
            post.diagnostics.notes.iter().any(|note| note.contains("ci_screened_floor")),
            "no diagnostic disclosed the screened-out floor: {:?}",
            post.diagnostics.notes
        );
    }

    /// A caller's explicit `required` link is an explicit prior belief, not a
    /// suggestion. `infer_graphs` must not silently discard it by replacing the
    /// whole constraint set with the engine's own screening constraints.
    #[test]
    fn caller_required_link_is_not_discarded_by_screening_constraints() {
        let (data, vars) = chain_data(180);
        let eng = CiScreenedPosterior::new()
            .with_mcmc(StructureMcmc::new().with_schedule(2, 150, 300, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let mut prior = GraphPrior::uniform();
        prior.constraints.required = Arc::from([crate::graph_posterior::static_link(&vars, 0, 2)]);
        let post =
            eng.run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &mut ws, &ctx).unwrap();
        assert!(
            post.diagnostics.notes.iter().any(|n| n.contains("ci_screened_pairs=3")),
            "caller's required a—c link was dropped from the screened candidate set: {:?}",
            post.diagnostics.notes
        );
    }

    /// When the engine's own screening constraints and the caller's prior
    /// disagree outright (one forbids what the other requires), overwriting
    /// one with the other hides a real conflict. `infer_graphs` must refuse.
    #[test]
    fn conflicting_required_and_forbidden_constraints_are_refused() {
        let (data, vars) = chain_data(180);
        let eng = CiScreenedPosterior::new()
            .with_constraints(DiscoveryConstraints {
                forbidden: Arc::from([crate::graph_posterior::static_link(&vars, 0, 2)]),
                ..DiscoveryConstraints::default()
            })
            .with_mcmc(StructureMcmc::new().with_schedule(2, 150, 300, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let mut prior = GraphPrior::uniform();
        prior.constraints.required = Arc::from([crate::graph_posterior::static_link(&vars, 0, 2)]);
        let err = eng.run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &mut ws, &ctx);
        assert!(err.is_err(), "conflicting required/forbidden constraints were silently resolved");
    }
}
