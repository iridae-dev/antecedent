//! Candidate-edge posterior after CI screening.
//!
//! Runs static PC (or a caller-supplied CI test) to obtain an undirected
//! skeleton, then structure-MCMC restricted to those pairs. Optional
//! Bayes-factor / posterior-dependence soft proposal weights are recorded as
//! diagnostics notes (not hard truth).
//!
//! The screen is a frequentist filter, not an oracle: a pair the CI test fails
//! to reject is excluded from the MCMC proposal set, so its posterior marginal
//! is bounded by the search space, not by evidence against the edge. A test that
//! fails to reject has only shown the dependence is not detectable at that sample
//! size, and a weak true edge fails it with the test's Type-II probability (for a
//! partial correlation of 0.35 at 75 rows, about one replicate in eight); once
//! the pair is dropped the posterior can never put mass on the true structure,
//! whatever its credible set claims. With [`CiScreenedPosterior::retain_undecided`]
//! a dropped pair therefore stays proposable unless the marginal likelihood
//! itself decisively supports independence (the Gaussian-BIC odds rule `removal_is_decisive`). The
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
    /// Keep a pair the CI test dropped in the proposal set unless the Gaussian-BIC evidence
    /// for independence, given the separating set that dropped it, is decisive
    /// (`removal_is_decisive`). `false` restricts the posterior to the CI skeleton exactly.
    pub retain_undecided: bool,
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
            .field("retain_undecided", &self.retain_undecided)
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
            fdr: Some(crate::pipeline::static_bh_fdr()),
            soft_weight: CiSoftWeight::None,
            retain_undecided: false,
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

    /// Keep CI-dropped pairs whose independence the marginal likelihood does not decisively
    /// support ([`Self::retain_undecided`]).
    #[must_use]
    pub fn with_retain_undecided(mut self, retain: bool) -> Self {
        self.retain_undecided = retain;
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
        prior.validate_for(variables)?;
        let effective = merge_screening_constraints(&self.constraints, &prior.constraints)?;
        let merged_caller_constraints = !prior.constraints.required.is_empty()
            || !prior.constraints.forbidden.is_empty()
            || prior.constraints.max_parents.is_some()
            || !prior.constraints.tiers.is_empty();

        let mut prior = prior.clone();
        prior.constraints = effective.clone();

        let (mut pairs, sepsets) =
            pc_skeleton_pairs(&effective, self, data, variables, workspace, ctx)?;
        let readmitted = if self.retain_undecided {
            readmit_undecided_pairs(&mut pairs, &sepsets, &effective, data, variables)?
        } else {
            Vec::new()
        };
        if pairs.is_empty() {
            return Err(DiscoveryError::unsupported("CI screening produced empty skeleton"));
        }

        let soft_note = soft_weight_note(self.soft_weight, data, variables, &pairs, ctx)?;

        let pair_arc: Arc<[(u32, u32)]> = Arc::from(pairs);
        let mcmc = self.mcmc.clone().with_candidate_pairs(Arc::clone(&pair_arc));
        let mut post = mcmc.run(data, variables, &prior, score_family, workspace, ctx)?;
        if let Some(note) = soft_note {
            post.push_note(note);
        }
        post.push_note(format!("ci_screened_pairs={}", pair_arc.len()));
        if !readmitted.is_empty() {
            post.push_note(format!(
                "ci_screen_retained pairs=[{}]: the CI test dropped them but the Gaussian-BIC \
                 evidence for independence is not decisive (odds under {SCREEN_DECISIVE_ODDS}:1), \
                 so the sampler may still propose them",
                readmitted.join(", ")
            ));
        }
        if merged_caller_constraints {
            post.push_note(
                "ci_screened_prior_constraints_merged=caller required/forbidden/max_parents/tiers \
                 combined with screening constraints, not overwritten",
            );
        }
        if let Some(note) = screened_out_note(data, variables, &pair_arc) {
            post.push_note(note);
        }
        apply_screened_out_floor(
            &mut post,
            &effective,
            variables,
            &pair_arc,
            prior.edge_inclusion,
        )?;
        Ok(post)
    }
}

/// The exact edge marginals of a variable set a CI screen picked, with the screen's edges as
/// per-edge priors.
///
/// Every number is conditional on the screen: the variables outside `variables` were left out,
/// and a pair the screen did not keep carries the low prior probability it was given, so a
/// marginal here is not the marginal of the full posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct ScreenedMarginals {
    /// The variables the exact computation ran on, in the order of the input variables. The
    /// marginals are indexed by position in this list.
    pub variables: Arc<[VariableId]>,
    /// Exact marginals over `variables` under the screen-derived per-edge prior.
    pub marginals: crate::exact_marginals::ExactDagMarginals,
    /// Pairs of `variables` the screen kept (undirected).
    pub screened_in_pairs: usize,
    /// Statement that the marginals are conditional on the screen.
    pub conditional_on_screen: Arc<str>,
}

impl CiScreenedPosterior {
    /// Screen a wide variable set, then compute exact edge marginals on a small set around the
    /// variables of interest.
    ///
    /// The CI screen runs on all `variables`. The posterior set is `focus` plus the variables
    /// nearest to it in the screen's skeleton (breadth first, ties by position), at most
    /// `max_variables` of them. On that set the exact dynamic program runs with the screen as a
    /// per-edge prior: both directions of a pair the screen kept get `in_probability`, every
    /// other pair gets `out_probability`. Edges the caller's own [`GraphPrior::edge_priors`] list
    /// keep the caller's probability. The exact ceiling
    /// ([`crate::EXACT_MARGINAL_MAX_NODES`]) then applies to the selected set, per question,
    /// not to the number of variables screened.
    ///
    /// # Errors
    ///
    /// An empty or unknown `focus`, more focus variables than `max_variables`, a
    /// `max_variables` above the exact ceiling, probabilities outside `(0, 1)`, an invalid
    /// prior, screening failures, or the exact computation's refusals.
    #[allow(clippy::too_many_arguments, reason = "one screened question, stated field by field")]
    pub fn screen_then_exact_marginals(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        focus: &[VariableId],
        max_variables: usize,
        prior: &GraphPrior,
        in_probability: f64,
        out_probability: f64,
        score_family: GraphScoreFamily,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<ScreenedMarginals, DiscoveryError> {
        prior.validate_for(variables)?;
        let bad = |p: f64| !(p > 0.0 && p < 1.0);
        if bad(in_probability) || bad(out_probability) {
            return Err(DiscoveryError::Refused {
                code: antecedent_core::reason_code!("invalid_argument"),
                message: "structure_posterior.edge_prior_probability: the screened-in and \
                          screened-out edge probabilities must lie in (0, 1)"
                    .to_owned(),
            });
        }
        if max_variables == 0 || max_variables > crate::exact_marginals::EXACT_MARGINAL_MAX_NODES {
            return Err(DiscoveryError::unsupported(
                "screen-then-exact marginals take between 1 and EXACT_MARGINAL_MAX_NODES \
                 (15) variables per question",
            ));
        }
        let focus_idx: Vec<usize> = focus
            .iter()
            .map(|v| {
                variables.iter().position(|c| c == v).ok_or_else(|| {
                    DiscoveryError::unsupported(
                        "a focus variable is not among the variables screened",
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        if focus_idx.is_empty() || focus_idx.len() > max_variables {
            return Err(DiscoveryError::unsupported(
                "screen-then-exact marginals need at least one focus variable and no more \
                 than max_variables",
            ));
        }

        let effective = merge_screening_constraints(&self.constraints, &prior.constraints)?;
        let (mut pairs, sepsets) =
            pc_skeleton_pairs(&effective, self, data, variables, workspace, ctx)?;
        if self.retain_undecided {
            readmit_undecided_pairs(&mut pairs, &sepsets, &effective, data, variables)?;
        }
        let n = variables.len();
        let mut neighbours = vec![Vec::new(); n];
        for &(a, b) in &pairs {
            neighbours[a as usize].push(b as usize);
            neighbours[b as usize].push(a as usize);
        }
        // Breadth-first from the focus: nearest variables first, ties by position.
        let mut chosen = vec![false; n];
        let mut frontier: Vec<usize> = Vec::new();
        for &i in &focus_idx {
            if !chosen[i] {
                chosen[i] = true;
                frontier.push(i);
            }
        }
        let mut count = frontier.len();
        while !frontier.is_empty() && count < max_variables {
            let mut next: Vec<usize> = frontier
                .iter()
                .flat_map(|&i| neighbours[i].iter().copied())
                .filter(|&j| !chosen[j])
                .collect();
            next.sort_unstable();
            next.dedup();
            next.truncate(max_variables - count);
            for &j in &next {
                chosen[j] = true;
            }
            count += next.len();
            frontier = next;
        }
        let selected: Vec<usize> = (0..n).filter(|&i| chosen[i]).collect();
        let subset: Vec<VariableId> = selected.iter().map(|&i| variables[i]).collect();

        let kept: HashSet<(usize, usize)> =
            pairs.iter().map(|&(a, b)| (a as usize, b as usize)).collect();
        let index = crate::constraints::ConstraintIndex::compile(&effective);
        let caller: std::collections::HashMap<(VariableId, VariableId), f64> = prior
            .edge_priors
            .as_ref()
            .map(|map| map.edges.iter().map(|&(a, b, p)| ((a, b), p)).collect())
            .unwrap_or_default();
        let mut edges = Vec::new();
        let mut screened_in_pairs = 0usize;
        for (pos_a, &a) in selected.iter().enumerate() {
            for (pos_b, &b) in selected.iter().enumerate() {
                if a == b {
                    continue;
                }
                if pos_a < pos_b && kept.contains(&(a.min(b), a.max(b))) {
                    screened_in_pairs += 1;
                }
                if index
                    .is_forbidden(&effective, crate::graph_posterior::static_link(variables, a, b))
                {
                    continue;
                }
                let screened = kept.contains(&(a.min(b), a.max(b)));
                let p = caller.get(&(variables[a], variables[b])).copied().unwrap_or(if screened {
                    in_probability
                } else {
                    out_probability
                });
                edges.push((variables[a], variables[b], p));
            }
        }
        let subset_prior = GraphPrior {
            constraints: effective,
            edge_inclusion: prior.edge_inclusion,
            edge_priors: Some(crate::edge_prior::EdgePriors::new(edges)),
            soft_tiers: prior.soft_tiers.as_ref().map(|soft| crate::edge_prior::SoftTiers {
                tiers: soft
                    .tiers
                    .iter()
                    .map(|tier| {
                        Arc::from(
                            tier.iter().copied().filter(|v| subset.contains(v)).collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<Arc<[VariableId]>>>()
                    .into(),
                log_penalty: soft.log_penalty,
            }),
        };
        let mut marginals = crate::exact_marginals::exact_dag_marginals(
            data,
            &subset,
            &subset_prior,
            score_family,
            ctx,
        )?;
        let statement = format!(
            "conditional on the CI screen: the exact marginals cover {} of {} screened variables \
             ({} of their pairs kept by the screen, each given prior probability {in_probability} \
             per direction against {out_probability} for the rest); variables outside the set \
             are not conditioned on and no marginal here is the marginal of the full posterior",
            subset.len(),
            n,
            screened_in_pairs,
        );
        let selected: Arc<[VariableId]> = Arc::from(subset);
        let statement: Arc<str> = Arc::from(statement);
        marginals.variables = Some(Arc::clone(&selected));
        marginals.conditional_on_screen = Some(Arc::clone(&statement));
        Ok(ScreenedMarginals {
            variables: selected,
            marginals,
            screened_in_pairs,
            conditional_on_screen: statement,
        })
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

/// Note naming every pair the CI screen excluded from the MCMC proposal set.
///
/// Their marginals are conditional on the screen: without an explicit `edge_inclusion` prior
/// they are exactly 0.0 because the sampler never proposes them, which is a bound on the
/// search space, not posterior evidence that the edge is absent.
fn screened_out_note(
    data: &TabularData,
    variables: &[VariableId],
    screened_in: &[(u32, u32)],
) -> Option<Arc<str>> {
    use antecedent_data::TableView;
    let name = |i: usize| {
        data.schema()
            .get(variables[i])
            .map_or_else(|_| variables[i].raw().to_string(), |v| v.name.to_string())
    };
    let screened_in: HashSet<(u32, u32)> = screened_in.iter().copied().collect();
    let n = variables.len();
    let out: Vec<String> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .filter(|&(i, j)| {
            !screened_in.contains(&(crate::indexing::dense_u32(i), crate::indexing::dense_u32(j)))
        })
        .map(|(i, j)| format!("{}-{}", name(i), name(j)))
        .collect();
    (!out.is_empty()).then(|| {
        Arc::from(format!(
            "ci_screened_out pairs=[{}]: never proposed by MCMC, so their edge marginals are \
             conditional on the CI screen (0.0 unless an edge_inclusion prior sets a floor), \
             not evidence the edges are absent",
            out.join(", ")
        ))
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
///
/// # Errors
///
/// The posterior refuses the replacement marginals (length mismatch).
fn apply_screened_out_floor(
    post: &mut GraphPosterior,
    effective: &DiscoveryConstraints,
    variables: &[VariableId],
    screened_in: &[(u32, u32)],
    edge_inclusion_prior: Option<f64>,
) -> Result<(), DiscoveryError> {
    let Some(p) = edge_inclusion_prior else { return Ok(()) };
    if p <= 0.0 {
        return Ok(());
    }
    let floor = p.min(effective.alpha);
    if floor <= 0.0 {
        return Ok(());
    }
    let n = variables.len();
    let screened_in: HashSet<(u32, u32)> = screened_in.iter().copied().collect();
    let forbidden_idx: HashSet<(u32, u32)> = effective
        .forbidden
        .iter()
        .filter_map(|link| {
            let si = variables.iter().position(|v| *v == link.source)?;
            let ti = variables.iter().position(|v| *v == link.target)?;
            Some((crate::indexing::dense_u32(si.min(ti)), crate::indexing::dense_u32(si.max(ti))))
        })
        .collect();

    let mut floored = 0usize;
    let mut marginals = post.edge_marginals().to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let key = (crate::indexing::dense_u32(i), crate::indexing::dense_u32(j));
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
        // The orientation marginals are the edge marginals (one stored accumulator), so
        // flooring them once floors both and `to_weighted_samples` cannot report a
        // screened-out edge as floored in one field and exactly 0 in the other.
        post.replace_edge_marginals(marginals)?;
        post.push_note(format!(
            "ci_screened_floor pairs={floored} floor={floor:.4} \
             (min of caller edge_inclusion prior and screen alpha; screened-out \
             pairs are reported as unlikely, not certainly absent)"
        ));
    }
    Ok(())
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
) -> Result<(Vec<(u32, u32)>, crate::result::PcSepsets), DiscoveryError> {
    // The screen reads the uncapped skeleton. The posterior enforces the parent limit through
    // its own prior.
    let mut screening = constraints.clone();
    screening.max_parents = None;
    let pc = Pc::new()
        .with_constraints(screening)
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
    Ok((v, result.sepsets))
}

/// Odds against an edge at which the marginal likelihood is said to decide for independence:
/// 19:1 is the posterior odds that leave an edge at most 5% probability under even prior odds,
/// the mass a 95% credible set may omit.
const SCREEN_DECISIVE_ODDS: f64 = 19.0;

/// Whether the Gaussian-BIC evidence decisively supports `x ⊥ y | S` over the edge `x — y`.
///
/// `rho` is the sample partial correlation of the pair given the separating set and `n` the row
/// count. Adding the edge to a Gaussian node already conditioned on `S` changes the BIC score
/// by `-(n/2)·ln(1 - rho²) - ½·ln n`: the log-likelihood gain of one coefficient less its
/// penalty. The edge is decisively absent when that is below `-ln(19)`. Even a partial
/// correlation of exactly zero loses only `½·ln n`, so no sample under 361 rows can decide: at
/// those sizes a CI test that fails to reject has not shown independence, only low power.
#[must_use]
pub(crate) fn removal_is_decisive(n: usize, rho: f64) -> bool {
    if n < 2 || !rho.is_finite() {
        return false;
    }
    let nf = n as f64;
    let gain = -0.5 * nf * (1.0 - rho * rho).max(f64::MIN_POSITIVE).ln() - 0.5 * nf.ln();
    gain < -SCREEN_DECISIVE_ODDS.ln()
}

/// Sample partial correlation of columns `i` and `j` given `given`, `None` when a value is not
/// finite or the correlation matrix is singular.
fn partial_correlation_of(cols: &[&[f64]], i: usize, j: usize, given: &[usize]) -> Option<f64> {
    let idx: Vec<usize> = [i, j].into_iter().chain(given.iter().copied()).collect();
    let k = idx.len();
    let n = cols[i].len();
    let means: Vec<f64> = idx.iter().map(|&c| cols[c].iter().sum::<f64>() / n as f64).collect();
    let mut a = vec![vec![0.0_f64; 2 * k]; k];
    for p in 0..k {
        for q in 0..k {
            let mut s = 0.0;
            for r in 0..n {
                s += (cols[idx[p]][r] - means[p]) * (cols[idx[q]][r] - means[q]);
            }
            a[p][q] = s;
        }
    }
    let sd: Vec<f64> = (0..k).map(|p| a[p][p].sqrt()).collect();
    if sd.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return None;
    }
    for p in 0..k {
        for q in 0..k {
            a[p][q] /= sd[p] * sd[q];
        }
        a[p][k + p] = 1.0;
    }
    // Gauss-Jordan inverse with partial pivoting.
    for col in 0..k {
        let piv = (col..k).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        let d = a[col][col];
        for v in &mut a[col] {
            *v /= d;
        }
        for r in 0..k {
            if r != col {
                let f = a[r][col];
                if f != 0.0 {
                    for c in 0..2 * k {
                        let t = a[col][c];
                        a[r][c] -= f * t;
                    }
                }
            }
        }
    }
    let (p00, p11, p01) = (a[0][k], a[1][k + 1], a[0][k + 1]);
    let rho = -p01 / (p00 * p11).sqrt();
    rho.is_finite().then_some(rho)
}

/// Add back every pair the CI test dropped unless the marginal likelihood decisively supports
/// the independence that dropped it (`removal_is_decisive`); returns the pairs added, named.
///
/// The separating set is the one the screen recorded for the pair (the empty set when the pair
/// was dropped without one, as by the family-wise correction). A pair the constraints forbid
/// stays out, and a pair whose partial correlation cannot be computed stays in.
fn readmit_undecided_pairs(
    pairs: &mut Vec<(u32, u32)>,
    sepsets: &crate::result::PcSepsets,
    constraints: &DiscoveryConstraints,
    data: &TabularData,
    variables: &[VariableId],
) -> Result<Vec<String>, DiscoveryError> {
    use antecedent_core::Lag;
    use antecedent_data::TableView;
    let cols = crate::pc::collect_float_columns(data, variables)?;
    let col_refs: Vec<&[f64]> = cols.iter().map(std::convert::AsRef::as_ref).collect();
    let n = col_refs.first().map_or(0, |c| c.len());
    let kept: HashSet<(u32, u32)> = pairs.iter().copied().collect();
    let forbidden: HashSet<(usize, usize)> = constraints
        .forbidden
        .iter()
        .filter_map(|link| {
            let si = variables.iter().position(|v| *v == link.source)?;
            let ti = variables.iter().position(|v| *v == link.target)?;
            Some((si, ti))
        })
        .collect();
    let name = |i: usize| {
        data.schema()
            .get(variables[i])
            .map_or_else(|_| variables[i].raw().to_string(), |v| v.name.to_string())
    };
    let mut added = Vec::new();
    let p = variables.len();
    for i in 0..p {
        for j in (i + 1)..p {
            let key = (crate::indexing::dense_u32(i), crate::indexing::dense_u32(j));
            if kept.contains(&key) || forbidden.contains(&(i, j)) || forbidden.contains(&(j, i)) {
                continue;
            }
            let lag = Lag::CONTEMPORANEOUS;
            let sep = sepsets
                .get(&(variables[i], lag, variables[j], lag))
                .or_else(|| sepsets.get(&(variables[j], lag, variables[i], lag)));
            let given: Option<Vec<usize>> = sep.map_or_else(
                || Some(Vec::new()),
                |s| s.iter().map(|(v, _)| variables.iter().position(|w| w == v)).collect(),
            );
            let decisive = given
                .and_then(|g| partial_correlation_of(&col_refs, i, j, &g))
                .is_some_and(|rho| removal_is_decisive(n, rho));
            if !decisive {
                pairs.push(key);
                added.push(format!("{}-{}", name(i), name(j)));
            }
        }
    }
    pairs.sort_unstable();
    Ok(added)
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

    fn chain_data(n_rows: usize) -> (TabularData, Vec<VariableId>) {
        let mut rng = antecedent_core::CausalRng::from_seed(21);
        let mut a = vec![0.0; n_rows];
        let mut bb = vec![0.0; n_rows];
        let mut c = vec![0.0; n_rows];
        for i in 0..n_rows {
            a[i] = rng.next_f64() * 2.0 - 1.0;
            bb[i] = 1.5 * a[i] + 0.2 * (rng.next_f64() * 2.0 - 1.0);
            c[i] = 1.2 * bb[i] + 0.2 * (rng.next_f64() * 2.0 - 1.0);
        }
        let data = crate::test_tables::continuous_table(&["a", "b", "c"], vec![a, bb, c]);
        let vars: Vec<_> = (0..3).map(VariableId::from_raw).collect();
        (data, vars)
    }

    #[test]
    fn screened_posterior_runs() {
        // 2 chains / 300 draws were refused by the per-edge MCSE gate for about a third of
        // seeds, so this passed only on a lucky one and flipped with last-bit platform
        // differences; 4 chains / 1200 draws clear it for every seed tried.
        let (data, vars) = chain_data(180);
        let eng = CiScreenedPosterior::new()
            .with_soft_weight(CiSoftWeight::BayesFactor)
            .with_mcmc(StructureMcmc::new().with_schedule(4, 150, 1200, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let post = eng
            .run(&data, &vars, &GraphPrior::uniform(), GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap();
        assert!(post.n_graphs() >= 1);
        assert!(post.diagnostics().notes.iter().any(|n| n.contains("ci_screened_pairs")));
    }

    /// A weak true edge dropped by a conditional test (here the partial correlation 0.35 at 75
    /// rows, power about 0.87) must stay proposable: nothing at that size can decide for
    /// independence, while a large sample with a null partial correlation can.
    #[test]
    fn removal_needs_decisive_evidence_of_independence() {
        assert!(!removal_is_decisive(75, 0.3465));
        assert!(!removal_is_decisive(75, 0.0));
        assert!(!removal_is_decisive(300, 0.0));
        assert!(removal_is_decisive(1000, 0.0));
        assert!(!removal_is_decisive(1000, 0.1));
        assert!(!removal_is_decisive(100_000, 0.05));
        assert!(!removal_is_decisive(10, f64::NAN));
    }

    #[test]
    fn partial_correlation_matches_the_closed_form() {
        let (data, vars) = chain_data(400);
        let cols = crate::pc::collect_float_columns(&data, &vars).unwrap();
        let refs: Vec<&[f64]> = cols.iter().map(std::convert::AsRef::as_ref).collect();
        let r = |x: usize, y: usize| partial_correlation_of(&refs, x, y, &[]).unwrap();
        let (rab, rbc, rac) = (r(0, 1), r(1, 2), r(0, 2));
        let closed = (rac - rab * rbc) / ((1.0 - rab * rab) * (1.0 - rbc * rbc)).sqrt();
        let got = partial_correlation_of(&refs, 0, 2, &[1]).unwrap();
        assert!((got - closed).abs() < 1e-9, "{got} vs {closed}");
    }

    /// At 180 rows the screen's drop of `a—c` is not decisive, so a retaining screen keeps all
    /// three pairs.
    #[test]
    fn retaining_screen_keeps_pairs_it_cannot_decide() {
        let (data, vars) = chain_data(180);
        let eng = CiScreenedPosterior::new()
            .with_retain_undecided(true)
            .with_mcmc(StructureMcmc::new().with_schedule(4, 400, 1200, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let post = eng
            .run(&data, &vars, &GraphPrior::uniform(), GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap();
        assert!(
            post.diagnostics().notes.iter().any(|n| n.contains("ci_screened_pairs=3")),
            "{:?}",
            post.diagnostics().notes
        );
    }

    /// With the default prior (no `edge_inclusion`), `a—c` is screened out and its marginal is
    /// exactly 0.0 because MCMC never proposes it. The posterior must name that pair so the
    /// 0.0 is not read as evidence of absence.
    #[test]
    fn screened_out_pairs_are_named_under_the_default_prior() {
        let (data, vars) = chain_data(180);
        let eng = CiScreenedPosterior::new()
            .with_mcmc(StructureMcmc::new().with_schedule(4, 150, 1200, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let post = eng
            .run(&data, &vars, &GraphPrior::uniform(), GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap();
        let note = post
            .diagnostics()
            .notes
            .iter()
            .find(|n| n.starts_with("ci_screened_out"))
            .unwrap_or_else(|| panic!("{:?}", post.diagnostics().notes));
        assert!(note.contains("a-c"), "{note}");
        assert!(!note.contains("a-b") && !note.contains("b-c"), "{note}");
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
            .with_mcmc(StructureMcmc::new().with_schedule(4, 150, 1200, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let prior = GraphPrior::bernoulli_edges(0.3).unwrap();
        let post =
            eng.run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &mut ws, &ctx).unwrap();
        let n = 3;
        let ac = post.edge_marginals()[2] + post.edge_marginals()[2 * n];
        assert!(ac > 0.0, "screened-out a—c marginal reported as exact 0.0 (ac={ac})");
        assert!(
            post.diagnostics().notes.iter().any(|note| note.contains("ci_screened_floor")),
            "no diagnostic disclosed the screened-out floor: {:?}",
            post.diagnostics().notes
        );
        let samples = post.to_weighted_samples().unwrap();
        assert_eq!(samples.edge_marginals.as_deref(), Some(post.edge_marginals().as_ref()));
        assert_eq!(
            samples.orientation_marginals.as_deref(),
            Some(post.orientation_marginals().as_ref())
        );
        assert_eq!(post.edge_marginals().as_ref(), post.orientation_marginals().as_ref());
        let half = ac / 2.0;
        assert!((post.orientation_marginals()[2] - half).abs() < 1e-12);
        assert!((post.orientation_marginals()[2 * n] - half).abs() < 1e-12);
    }

    /// A caller's explicit `required` link is an explicit prior belief, not a
    /// suggestion. `infer_graphs` must not silently discard it by replacing the
    /// whole constraint set with the engine's own screening constraints.
    #[test]
    fn caller_required_link_is_not_discarded_by_screening_constraints() {
        let (data, vars) = chain_data(180);
        // The caller's required link widens the screened candidate set from 2 pairs to 3
        // (unlike the other tests in this module), so the old 2 chains / 300 draws no longer
        // cleared the per-edge MCSE gate reliably. More chains and draws converges that larger
        // space properly instead of loosening the gate.
        let eng = CiScreenedPosterior::new()
            .with_mcmc(StructureMcmc::new().with_schedule(4, 400, 1200, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let mut prior = GraphPrior::uniform();
        prior.constraints.required = Arc::from([crate::graph_posterior::static_link(&vars, 0, 2)]);
        let post =
            eng.run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &mut ws, &ctx).unwrap();
        assert!(
            post.diagnostics().notes.iter().any(|n| n.contains("ci_screened_pairs=3")),
            "caller's required a—c link was dropped from the screened candidate set: {:?}",
            post.diagnostics().notes
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
            .with_mcmc(StructureMcmc::new().with_schedule(4, 150, 1200, 1));
        let ctx = ExecutionContext::for_tests(5);
        let mut ws = DiscoveryWorkspace::default();
        let mut prior = GraphPrior::uniform();
        prior.constraints.required = Arc::from([crate::graph_posterior::static_link(&vars, 0, 2)]);
        let err = eng.run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &mut ws, &ctx);
        assert!(err.is_err(), "conflicting required/forbidden constraints were silently resolved");
    }
}
