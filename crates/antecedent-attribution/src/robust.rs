//! Robust distribution-change attribution, a regression-hybrid variant of the
//! Budhathoki, Janzing, Bloebaum & Ng (2021) mechanism-Shapley decomposition.
//!
//! Uses fitted mechanism hybrids (same topology as
//! [`distribution_change`](crate::distribution_change::distribution_change)) with a
//! structural-mean payoff `v(S) = E[f(X) | S]`: the expectation is taken over the actual
//! fitted distribution of every node, not a single point. Linear-family mechanisms compose
//! the registry-selected fit directly (whatever family the registry picked — plain
//! `LinearGaussian`, `HierarchicalLinear` shrinkage, `Bvar`, or `Constant` — arithmetic
//! composition of linear coefficients is exact regardless of family, so nothing is refit).
//! Nonlinear slots integrate `E[f(X)]` by Monte Carlo: ancestral sampling with real draws
//! from each node's structural noise, topologically composed exactly as
//! [`distribution_change`](crate::distribution_change::distribution_change) samples, using
//! common random numbers across coalitions (the RNG is reseeded to the same stream on every
//! call to [`CoalitionPayoff::value`]) so that only the swapped mechanisms differ between
//! masks, keeping the Shapley differences numerically stable at a practical sample count.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ChangeAttributionQuery, ComponentId, ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::DenseNodeId;
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
    MechanismWorkspace, ParentBatch, SelectionPolicy, evaluate_column, sample_noise_column,
};

use crate::change_common::{ROBUST_STREAM, attribution_stream};
use crate::coalition::full_coalition_mask;
use crate::distribution_change::mechanism_players;
use crate::error::AttributionError;
use crate::prep::{
    require_mechanism_components, require_shapley_config, resolve_change_populations,
    resolve_outcome_dense,
};
use crate::result::ChangeAttributionResult;
use crate::shapley::{CoalitionPayoff, estimate_shapley};

/// Options for the robust estimator.
#[derive(Clone, Debug)]
pub struct RobustChangeOptions {
    /// Cap on units used for regression / evaluation.
    pub max_rows: usize,
    /// Monte Carlo draws used by the nonlinear path to integrate `E[f(X)]` per
    /// coalition. Ignored when every player's mechanism is linear-family (that
    /// path composes fitted coefficients exactly, with no sampling error).
    pub n_samples: usize,
    /// RNG seed for the nonlinear path's Monte Carlo integration.
    pub seed: u64,
}

impl Default for RobustChangeOptions {
    fn default() -> Self {
        Self { max_rows: 10_000, n_samples: 4_000, seed: 0 }
    }
}

/// Robust attribution via mechanism hybrids + Shapley.
///
/// # Errors
///
/// Fit / size / Shapley failures.
pub fn distribution_change_robust(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &RobustChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    robust_impl(graph_model, data, query, options, None, ctx)
}

/// [`distribution_change_robust`] after refitting both populations under `refit`, the
/// mechanism families held at the ones selected on the unweighted populations: one replicate
/// of the row bootstrap (counts) or one draw of the posterior over the mechanisms (row
/// weights).
///
/// # Errors
///
/// As [`distribution_change_robust`], and weights that do not match the resolved populations.
pub fn distribution_change_robust_with_refit(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &RobustChangeOptions,
    refit: &crate::distribution_change::PopulationRefit<'_>,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    robust_impl(graph_model, data, query, options, Some(refit), ctx)
}

fn robust_impl(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &RobustChangeOptions,
    refit: Option<&crate::distribution_change::PopulationRefit<'_>>,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    query.validate()?;
    require_mechanism_components(
        query.components,
        "distribution_change_robust requires AttributionComponents::Mechanisms",
    )?;

    let (baseline_data, comparison_data) = resolve_change_populations(data, query)?;

    if baseline_data.row_count() > options.max_rows
        || comparison_data.row_count() > options.max_rows
    {
        return Err(AttributionError::SizeLimit {
            kind: "units",
            requested: baseline_data.row_count().max(comparison_data.row_count()),
            max: options.max_rows,
        });
    }

    let outcome_dense = resolve_outcome_dense(graph_model, query.outcome)?;
    let players = mechanism_players(graph_model, outcome_dense, query.max_components)?;

    let (baseline_mechs, comparison_mechs) = match refit {
        None => {
            let (baseline_mechs, _) = MechanismRegistry::standard().assign_and_fit(
                graph_model,
                &baseline_data,
                SelectionPolicy::BestScore,
            )?;
            let (comparison_mechs, _) = MechanismRegistry::standard().assign_and_fit(
                graph_model,
                &comparison_data,
                SelectionPolicy::BestScore,
            )?;
            (baseline_mechs, comparison_mechs)
        }
        Some(refit) => crate::distribution_change::refit_populations(
            graph_model,
            &baseline_data,
            &comparison_data,
            refit,
            ctx,
        )?,
    };

    let all_linear = players.iter().all(|c| {
        model_slot_is_linear(&baseline_mechs, graph_model, c.variable())
            && model_slot_is_linear(&comparison_mechs, graph_model, c.variable())
    });

    let approximation = require_shapley_config(
        &query.allocation,
        "distribution_change_robust currently supports Shapley allocation",
    )?;
    // `n_samples` is the budget the caller declared for each coalition. The
    // linear payoff does not draw, but a declared budget past the ceiling is
    // still refused, the same rule `distribution_change` uses.
    crate::shapley::check_coalition_sample_budget(
        players.len(),
        &query.allocation,
        options.n_samples.max(1),
    )?;

    let (v0, v_full, estimate) = if all_linear {
        let mut payoff = RobustLinearPayoff {
            model: graph_model,
            baseline: &baseline_data,
            players: players.clone(),
            outcome: query.outcome,
            fitted: Vec::new(),
            column_means: Vec::new(),
            outcome_idx: None,
            n_rows: 0,
        };
        payoff.fit(&baseline_mechs, &comparison_mechs)?;
        let v0 = payoff.value(0)?;
        let full = full_coalition_mask(players.len())?;
        let v_full = payoff.value(full)?;
        let estimate = estimate_shapley(&players, approximation, &mut payoff, ctx)?;
        (v0, v_full, estimate)
    } else {
        let _scratch = admit_payoff_scratch(ctx, options.n_samples, graph_model.n_nodes())?;
        let mut payoff = RobustMechanismPayoff::new(
            graph_model.clone(),
            baseline_mechs,
            comparison_mechs,
            &players,
            outcome_dense,
            McSampling { n_samples: options.n_samples, seed: options.seed },
            ctx,
        );
        let v0 = payoff.value(0)?;
        let full = full_coalition_mask(players.len())?;
        let v_full = payoff.value(full)?;
        let estimate = estimate_shapley(&players, approximation, &mut payoff, ctx)?;
        (v0, v_full, estimate)
    };
    let total_change = v_full - v0;
    // `estimate` here always comes from `estimate_shapley` (never `sequential_allocate`,
    // enforced above by `require_shapley_config`), which always returns an empty
    // `interactions` vec — so deriving `interactions` via `pack_change_result` is
    // equivalent to the previously hardcoded `Arc::from([])`.
    Ok(crate::change_common::pack_change_result(
        query.outcome,
        total_change,
        estimate,
        Arc::from([]),
        Arc::from([]),
    ))
}

fn model_slot_is_linear(
    store: &CompiledMechanismStore,
    model: &CompiledCausalModel,
    var: VariableId,
) -> bool {
    let Some(d) = model.dense_of(var) else {
        return false;
    };
    matches!(
        &store.slots[d.as_usize()],
        MechanismSlot::LinearGaussian { .. }
            | MechanismSlot::HierarchicalLinear { .. }
            | MechanismSlot::Bvar { .. }
            | MechanismSlot::Constant { .. }
    )
}

/// Where a regression input column comes from at payoff-evaluation time.
enum ParentSource {
    /// Predicted column of an earlier player (index into the player list).
    Player(usize),
    /// Mean of a non-player parent column (index into `column_means`).
    Column(usize),
}

struct NodeRegression {
    baseline_beta: Vec<f64>,
    comparison_beta: Vec<f64>,
    /// Per-parent input source, resolved once at fit time (was an O(k) player
    /// scan plus a full `float64_values` column copy per row per parent).
    sources: Vec<ParentSource>,
}

struct RobustLinearPayoff<'a> {
    model: &'a CompiledCausalModel,
    baseline: &'a TabularData,
    players: Vec<ComponentId>,
    outcome: VariableId,
    fitted: Vec<NodeRegression>,
    /// Mean of each non-player parent column. An affine mechanism's mean is the
    /// mechanism of the parent means, so the row loop is not required.
    column_means: Vec<f64>,
    /// Index of the outcome among the players, if present.
    outcome_idx: Option<usize>,
    n_rows: usize,
}

impl RobustLinearPayoff<'_> {
    /// Compose `baseline_beta`/`comparison_beta` from the registry-selected fits
    /// (`baseline_mechs`/`comparison_mechs`) instead of refitting by OLS: whichever
    /// linear family the registry picked for a node (plain `LinearGaussian`,
    /// `HierarchicalLinear` shrinkage, `Bvar`, or `Constant`) already has an
    /// intercept + parent coefficients suitable for this composition, and
    /// `all_linear` (the caller's guard before constructing this payoff) has
    /// already confirmed every player is one of those four families.
    fn fit(
        &mut self,
        baseline_mechs: &CompiledMechanismStore,
        comparison_mechs: &CompiledMechanismStore,
    ) -> Result<(), AttributionError> {
        self.fitted.clear();
        self.column_means.clear();
        self.n_rows = self.baseline.row_count();
        self.outcome_idx = self.players.iter().position(|c| c.variable() == self.outcome);
        let mut column_vars: Vec<VariableId> = Vec::new();
        for (i, &comp) in self.players.iter().enumerate() {
            let dense = self
                .model
                .dense_of(comp.variable())
                .ok_or_else(|| AttributionError::missing_var("component", comp.variable()))?;
            let gather = self
                .model
                .gather_for(dense)
                .ok_or(AttributionError::MissingArtifact("missing gather"))?;
            let parents: Vec<VariableId> = gather
                .parents
                .iter()
                .map(|&p| self.model.output_layout.variables[p.as_usize()])
                .collect();
            let mut sources = Vec::with_capacity(parents.len());
            for &p in &parents {
                if let Some(pj) = self.players.iter().position(|c| c.variable() == p) {
                    // Players come from `mechanism_players` in topological order,
                    // so a parent that is itself a player always precedes `i`.
                    if pj >= i {
                        return Err(AttributionError::unsupported(
                            "robust linear payoff: player parent is not in topological order",
                        ));
                    }
                    sources.push(ParentSource::Player(pj));
                } else {
                    let ci = if let Some(ci) = column_vars.iter().position(|&v| v == p) {
                        ci
                    } else {
                        column_vars.push(p);
                        let col = self.baseline.float64_values(p)?;
                        let mean = if col.is_empty() {
                            0.0
                        } else {
                            col.iter().sum::<f64>() / col.len() as f64
                        };
                        self.column_means.push(mean);
                        self.column_means.len() - 1
                    };
                    sources.push(ParentSource::Column(ci));
                }
            }
            let baseline_beta =
                linear_beta(&baseline_mechs.slots[dense.as_usize()], parents.len())?;
            let comparison_beta =
                linear_beta(&comparison_mechs.slots[dense.as_usize()], parents.len())?;
            self.fitted.push(NodeRegression { baseline_beta, comparison_beta, sources });
        }
        Ok(())
    }
}

/// Extract `[intercept, coeffs...]` from a linear-family fitted slot.
///
/// # Errors
///
/// The slot is not one of the linear families this payoff supports, or the
/// fitted coefficient count does not match the gather's parent count (a fit /
/// topology mismatch, not something to silently zero-fill).
fn linear_beta(slot: &MechanismSlot, n_parents: usize) -> Result<Vec<f64>, AttributionError> {
    let (intercept, coeffs): (f64, &[f64]) = match slot {
        MechanismSlot::LinearGaussian { intercept, coeffs, .. }
        | MechanismSlot::HierarchicalLinear { intercept, coeffs, .. }
        | MechanismSlot::Bvar { intercept, coeffs, .. } => (*intercept, coeffs.as_ref()),
        MechanismSlot::Constant { value } => (*value, &[]),
        _ => {
            return Err(AttributionError::unsupported(
                "robust linear payoff: non-linear-family slot",
            ));
        }
    };
    if coeffs.len() != n_parents {
        return Err(AttributionError::unsupported(
            "robust linear payoff: fitted coefficient count does not match parent count",
        ));
    }
    let mut beta = Vec::with_capacity(n_parents + 1);
    beta.push(intercept);
    beta.extend_from_slice(coeffs);
    Ok(beta)
}

impl CoalitionPayoff for RobustLinearPayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let Some(outcome_idx) = self.outcome_idx else {
            return Err(AttributionError::unsupported(
                "robust payoff: outcome is not among Shapley players",
            ));
        };
        // An empty population has no outcome mean. A positive row count is linear,
        // so the column mean is the composition of parent means.
        if self.baseline.row_count() == 0 {
            return Ok(0.0);
        }
        let mut means = vec![0.0; self.players.len()];
        for i in 0..self.players.len() {
            let fit = &self.fitted[i];
            let beta =
                if mask & (1u64 << i) != 0 { &fit.comparison_beta } else { &fit.baseline_beta };
            let mut yhat = beta[0];
            for (pi, src) in fit.sources.iter().enumerate() {
                let x = match *src {
                    ParentSource::Player(pj) => means[pj],
                    ParentSource::Column(ci) => self.column_means[ci],
                };
                yhat += beta.get(pi + 1).copied().unwrap_or(0.0) * x;
            }
            means[i] = yhat;
        }
        Ok(means[outcome_idx])
    }
}

/// Nonlinear robust payoff: hybrid mechanisms, Monte Carlo integration of `E[f(X)]`.
///
/// Every node — root or not — is *sampled*, not plugged in at a point: for each of
/// `n_samples` draws, every node in topological order draws real structural noise from
/// its fitted mechanism (`sample_noise_column`) and is evaluated on that noise plus its
/// already-sampled parents (`evaluate_column`), exactly the ancestral walk
/// `sample_with_overlay_into` performs in `antecedent-model`. A root's fitted mechanism
/// *is* its marginal, so this integrates over the root's actual location *and* spread —
/// unlike a single ε=0 evaluation, a variance or shape change with an unchanged mean
/// moves the outcome here, because it moves the sampled column.
///
/// All per-coalition allocations are hoisted to construction: the working value matrix
/// (`n_samples * n_nodes`), the gathered-parent scratch (disjoint from the mechanism
/// workspace, so no per-node `to_vec` copy is needed to satisfy the borrow checker — same
/// layout as `sample_with_overlay`'s hoisted `parent_buf` in
/// `antecedent-model/src/sample.rs`), the sampled-noise scratch, and the dense-node →
/// player map (was an O(n) `dense_of` scan per player). Instead of cloning the full model
/// with a rebuilt hybrid store per coalition, each node's slot is chosen directly from the
/// baseline/comparison store — the same slot the hybrid store would have held (players
/// here are all mechanism-kind).
///
/// Every node gets its own stream, keyed by its dense id, and that stream is restarted for
/// every [`CoalitionPayoff::value`] call. Thus changing one node's mechanism cannot shift
/// the noise draws for later nodes (common random numbers across coalitions), which keeps
/// Monte Carlo Shapley differences from acquiring artificial interactions merely because
/// two mechanism families consume different numbers of random draws.
struct RobustMechanismPayoff<'a> {
    template: CompiledCausalModel,
    baseline: CompiledMechanismStore,
    comparison: CompiledMechanismStore,
    outcome: DenseNodeId,
    ws: MechanismWorkspace,
    /// Player index per dense node (aligned with the player bit order).
    node_player: Vec<Option<usize>>,
    /// Working value matrix (`n_samples * n_nodes`), overwritten every coalition.
    values: Vec<f64>,
    /// Gathered-parent scratch reused across nodes and coalitions.
    parent_scratch: Vec<f64>,
    /// Sampled structural-noise scratch reused across nodes and coalitions.
    noise_scratch: Vec<f64>,
    n_samples: usize,
    seed: u64,
    ctx: &'a ExecutionContext,
}

/// Monte Carlo sample count and stream seed of a robust mechanism payoff.
#[derive(Clone, Copy)]
struct McSampling {
    n_samples: usize,
    seed: u64,
}

/// Lease the sample matrix and noise column each payoff holds: one for the caller and one
/// per forked worker, before any is allocated.
fn admit_payoff_scratch(
    ctx: &ExecutionContext,
    n_samples: usize,
    n_nodes: usize,
) -> Result<antecedent_core::execution::AdmissionLease, AttributionError> {
    let holders = usize::try_from(ctx.parallelism.max_threads.get()).unwrap_or(usize::MAX);
    let bytes = n_samples
        .max(1)
        .checked_mul(n_nodes.saturating_add(1))
        .and_then(|cells| cells.checked_mul(size_of::<f64>()))
        .and_then(|bytes| bytes.checked_mul(holders.max(1)))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| AttributionError::Budget {
            message: "robust payoff scratch size overflow".to_owned(),
        })?;
    ctx.acquire_admission(0, bytes).map_err(|error| AttributionError::Budget {
        message: format!("admission pool cannot fund the robust payoff scratch: {error}"),
    })
}

impl<'a> RobustMechanismPayoff<'a> {
    fn new(
        template: CompiledCausalModel,
        baseline: CompiledMechanismStore,
        comparison: CompiledMechanismStore,
        players: &[ComponentId],
        outcome: DenseNodeId,
        sampling: McSampling,
        ctx: &'a ExecutionContext,
    ) -> Self {
        let McSampling { n_samples, seed } = sampling;
        let n_nodes = template.n_nodes();
        let n_samples = n_samples.max(1);
        let mut node_player = vec![None; n_nodes];
        for (i, comp) in players.iter().enumerate() {
            if let Some(dense) = template.dense_of(comp.variable()) {
                node_player[dense.as_usize()] = Some(i);
            }
        }
        Self {
            template,
            baseline,
            comparison,
            outcome,
            ws: MechanismWorkspace::default(),
            node_player,
            values: vec![0.0; n_samples * n_nodes],
            parent_scratch: Vec::new(),
            noise_scratch: vec![0.0; n_samples],
            n_samples,
            seed,
            ctx,
        }
    }
}

impl CoalitionPayoff for RobustMechanismPayoff<'_> {
    fn fork<'a>(&'a self) -> Option<Box<dyn CoalitionPayoff + Send + 'a>> {
        Some(Box::new(RobustMechanismPayoff {
            template: self.template.clone(),
            baseline: self.baseline.clone(),
            comparison: self.comparison.clone(),
            outcome: self.outcome,
            ws: MechanismWorkspace::default(),
            node_player: self.node_player.clone(),
            values: vec![0.0; self.values.len()],
            parent_scratch: Vec::new(),
            noise_scratch: vec![0.0; self.noise_scratch.len()],
            n_samples: self.n_samples,
            seed: self.seed,
            ctx: self.ctx,
        }))
    }

    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let n = self.n_samples;
        // Ancestral Monte Carlo walk (`parent_gathers` is aligned with `node_order`,
        // so this iteration is the `node_order` + `gather_for` walk without the O(n)
        // lookups): every node, root or not, draws real noise and is evaluated on it.
        for gather in self.template.parent_gathers.iter() {
            let node = gather.child;
            let n_parents = gather.n_parents();
            let need = n_parents.max(1).saturating_mul(n);
            if self.parent_scratch.len() < need {
                self.parent_scratch.resize(need, 0.0);
            }
            gather.gather(&self.values, n, &mut self.parent_scratch);
            let parents = ParentBatch {
                n_rows: n,
                n_parents,
                values: &self.parent_scratch[..n_parents.saturating_mul(n)],
            };
            let use_comparison =
                matches!(self.node_player[node.as_usize()], Some(i) if mask & (1u64 << i) != 0);
            let slot =
                if use_comparison { self.comparison.get(node) } else { self.baseline.get(node) };
            // A node-local stream prevents an upstream mechanism's draw count from
            // shifting every downstream node's noise when comparing coalitions.
            let node_seed = self
                .seed
                .wrapping_add((node.as_usize() as u64).wrapping_mul(antecedent_core::GOLDEN_GAMMA));
            let mut rng = attribution_stream(self.ctx, ROBUST_STREAM, node_seed);
            sample_noise_column(slot, n, &mut rng, &mut self.noise_scratch)?;
            let out = &mut self.values[node.as_usize() * n..(node.as_usize() + 1) * n];
            evaluate_column(slot, parents, &self.noise_scratch[..n], out, &mut self.ws)?;
        }
        let col = &self.values[self.outcome.as_usize() * n..(self.outcome.as_usize() + 1) * n];
        Ok(col.iter().sum::<f64>() / n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_common::DifferenceMeasure;
    use crate::distribution_change::{DistributionChangeOptions, distribution_change};
    use antecedent_core::{
        AllocationMethod, CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint,
        ShapleyConfig, SmallRoleSet, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::{Dag, DenseNodeId};

    /// `X -> Y` with deterministic (zero-noise) linear mechanisms: baseline `Y=X`,
    /// comparison `Y=X+2`. `X`'s marginal distribution is identical across both
    /// halves (values cycle `i % n_half`), isolating the change to `Y`'s mechanism.
    fn deterministic_linear_chain() -> (CompiledCausalModel, TabularData) {
        let n = 60usize;
        let n_half = 30usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % n_half) as f64 * 0.1;
            xv.push(x);
            yv.push(if i < n_half { x } else { x + 2.0 });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        (model, data)
    }

    /// Cross-check: on deterministic (zero-noise) linear data, the fast OLS-composition
    /// path (`RobustLinearPayoff`, exact arithmetic on fitted coefficients) and the
    /// full sampled `distribution_change` (Monte Carlo draws from the fitted
    /// mechanisms) must agree, since the mechanism's fitted residual noise is
    /// negligible. This is the only real check of `distribution_change_robust`'s
    /// numeric output against an independent computation — the sole prior test
    /// (`robust_linear_still_runs`) only asserted `total_change.abs() > 0.5`.
    #[test]
    fn robust_matches_distribution_change_on_deterministic_linear_data() {
        let (model, data) = deterministic_linear_chain();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&model, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = model.with_mechanisms(store);
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::Rows(Arc::from((0..30).collect::<Vec<_>>())),
            PopulationSelector::Rows(Arc::from((30..60).collect::<Vec<_>>())),
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let ctx = ExecutionContext::for_tests(1);

        let robust_result = distribution_change_robust(
            &model,
            &data,
            &query,
            &RobustChangeOptions::default(),
            &ctx,
        )
        .unwrap();

        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 4_000,
            seed: 17,
        };
        let sampled_result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();

        assert!(
            (robust_result.total_change - sampled_result.total_change).abs() < 0.1,
            "robust={} sampled={}",
            robust_result.total_change,
            sampled_result.total_change
        );

        let robust_y = robust_result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(1))
            .expect("y contribution (robust)")
            .contribution;
        let sampled_y = sampled_result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(1))
            .expect("y contribution (sampled)")
            .contribution;
        assert!(
            (robust_y - sampled_y).abs() < 0.1,
            "y contribution mismatch: robust={robust_y} sampled={sampled_y}"
        );
        // The +2 intercept shift on Y should dominate; X's marginal is identical
        // across both populations, so it should contribute ~nothing in either path.
        assert!(robust_y > 1.0, "robust_y={robust_y}");
        assert!(sampled_y > 1.0, "sampled_y={sampled_y}");
    }

    #[test]
    fn robust_linear_still_runs() {
        let n = 60usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % 30) as f64 * 0.1;
            xv.push(x);
            yv.push(if i < 30 { x } else { x + 2.0 });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&model, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = model.with_mechanisms(store);
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::Rows(Arc::from((0..30).collect::<Vec<_>>())),
            PopulationSelector::Rows(Arc::from((30..60).collect::<Vec<_>>())),
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let ctx = ExecutionContext::for_tests(1);
        let result = distribution_change_robust(
            &model,
            &data,
            &query,
            &RobustChangeOptions::default(),
            &ctx,
        )
        .unwrap();
        assert!(result.total_change.abs() > 0.5);
    }

    /// `.robust(true)` used to drop `with_sampling`. A non-mean measure is refused,
    /// and an explicit mean-diff sample budget does not change the exact linear result
    /// or replace the robust default when sampling was not set.
    #[test]
    fn robust_builder_honors_sampling_without_changing_the_linear_mean() {
        let (model, data) = deterministic_linear_chain();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&model, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = model.with_mechanisms(store);
        let ctx = ExecutionContext::for_tests(1);
        let baseline = PopulationSelector::Rows(Arc::from((0..30).collect::<Vec<_>>()));
        let comparison = PopulationSelector::Rows(Arc::from((30..60).collect::<Vec<_>>()));
        let run = |sampling: Option<(DifferenceMeasure, usize, u64)>| {
            let mut built = crate::builder::ChangeAttribution::new()
                .outcome(VariableId::from_raw(1))
                .baseline(baseline.clone())
                .comparison(comparison.clone())
                .robust(true);
            if let Some((measure, n, seed)) = sampling {
                built = built.with_sampling(measure, n, seed);
            }
            built.run(&model, &data, &ctx)
        };
        let refused = run(Some((DifferenceMeasure::GaussianKl, 10, 1))).unwrap_err();
        assert!(matches!(refused, AttributionError::Unsupported { .. }), "{refused:?}");
        let plain = run(None).unwrap();
        let sampled = run(Some((DifferenceMeasure::MeanDiff, 1_000, 9))).unwrap();
        assert_eq!(plain.total_change.to_bits(), sampled.total_change.to_bits());
    }

    /// Swapping a zero-effect root from a constant to a Gaussian family must not change
    /// the downstream mechanism's Monte Carlo noise. The structural truth is zero: Y has
    /// coefficient 0 on X and identical noise under both populations. This catches a
    /// sequential-stream bug where X's family consumes a different number of random draws
    /// and shifts Y's sample, creating a spurious finite-sample Shapley interaction.
    #[test]
    fn robust_node_noise_is_stable_when_another_family_changes_draw_count() {
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let template = CompiledCausalModel::compile(graph).unwrap();
        let x_constant = MechanismSlot::Constant { value: 0.0 };
        let x_gaussian =
            MechanismSlot::LinearGaussian { intercept: 0.0, coeffs: Arc::from([]), sigma: 1.0 };
        let y =
            MechanismSlot::LinearGaussian { intercept: 0.0, coeffs: Arc::from([0.0]), sigma: 1.0 };
        let baseline = CompiledMechanismStore { slots: Arc::from([x_constant, y.clone()]) };
        let comparison = CompiledMechanismStore { slots: Arc::from([x_gaussian, y]) };
        let players = [
            ComponentId::from_variable(VariableId::from_raw(0)),
            ComponentId::from_variable(VariableId::from_raw(1)),
        ];
        let ctx = ExecutionContext::for_tests(17);
        let mut payoff = RobustMechanismPayoff::new(
            template,
            baseline,
            comparison,
            &players,
            DenseNodeId::from_raw(1),
            McSampling { n_samples: 128, seed: 23 },
            &ctx,
        );

        let unchanged_x = payoff.value(0).unwrap();
        let changed_x_only = payoff.value(1).unwrap();
        assert_eq!(unchanged_x.to_bits(), changed_x_only.to_bits());
    }

    /// `X -> Y`, `Y = X²` exactly (zero-noise quadratic mechanism), `X`'s mean held
    /// fixed at 0 while its standard deviation moves 1 → 2 between baseline and
    /// comparison. Independent truth: `f(x) = x²` is exactly quadratic, so for *any*
    /// distribution shape `E[f(X)] = Var(X) + E[X]²` — a closed form that needs no
    /// simulation and holds regardless of what family fits `X`'s marginal. With
    /// `E[X] = 0` unchanged, the true swap-X-only payoff moves `Var(X): 1 → 4`, i.e.
    /// `Δ = 3` exactly.
    ///
    /// `RobustMechanismPayoff::value` is exercised directly (bypassing
    /// `distribution_change_robust`'s hardcoded `MechanismRegistry::standard()`,
    /// which has no basis/GP family and so can never select a genuinely nonlinear
    /// slot) with a hand-built mechanism store: `X` is `LinearGaussian` (its fitted
    /// marginal), `Y` is `LinearBasis` with a single `Power{degree: 2}` term, i.e.
    /// the mechanism is literally `Y = X²` with `sigma = 0`.
    #[test]
    fn nonlinear_payoff_attributes_ancestor_variance_change() {
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let template = CompiledCausalModel::compile(g).unwrap();

        // φ(x) = x² (standardized with center 0, scale 1, so φ(x) = x²  exactly).
        let quadratic_basis = antecedent_model::ParentBasis::new(
            1,
            Arc::from([0.0]),
            Arc::from([1.0]),
            Arc::from([Arc::from([]) as Arc<[f64]>]),
            Arc::from([antecedent_model::BasisTerm::Power { parent: 0, degree: 2 }]),
        )
        .unwrap();
        let y_slot = MechanismSlot::LinearBasis {
            intercept: 0.0,
            basis: quadratic_basis,
            coeffs: Arc::from([1.0]),
            sigma: 0.0,
        };
        let x_slot = |sigma: f64| MechanismSlot::LinearGaussian {
            intercept: 0.0,
            coeffs: Arc::from([]),
            sigma,
        };

        let baseline = CompiledMechanismStore { slots: Arc::from([x_slot(1.0), y_slot.clone()]) };
        // Only X's marginal changes (σ: 1 → 2); Y's mechanism (Y = X²) is untouched,
        // isolating the effect to X's variance with its mean held at 0 in both.
        let comparison = CompiledMechanismStore { slots: Arc::from([x_slot(2.0), y_slot]) };

        let players = vec![
            ComponentId::from_variable(VariableId::from_raw(0)),
            ComponentId::from_variable(VariableId::from_raw(1)),
        ];
        let outcome = DenseNodeId::from_raw(1);
        let ctx = ExecutionContext::for_tests(7);
        let mut payoff = RobustMechanismPayoff::new(
            template,
            baseline,
            comparison,
            &players,
            outcome,
            McSampling { n_samples: 20_000, seed: 11 },
            &ctx,
        );

        let v_baseline = payoff.value(0).unwrap();
        // Swap only X (player index 0) to its comparison (higher-variance) marginal;
        // Y stays at its baseline (identical, here) mechanism.
        let v_x_swapped = payoff.value(1).unwrap();
        let observed = v_x_swapped - v_baseline;
        let true_delta = 3.0; // Var(X): 4 - 1, means both 0.

        assert!(
            (observed - true_delta).abs() < 0.2,
            "nonlinear payoff should integrate E[f(X)] over X's actual (changed) \
             variance, not plug in a single point: observed Δ={observed}, true Δ={true_delta}"
        );
    }

    #[test]
    fn robust_builder_honors_sampling_and_refuses_other_measures() {
        use crate::builder::ChangeAttribution;
        let (model, data) = deterministic_linear_chain();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&model, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = model.with_mechanisms(store);
        let ctx = ExecutionContext::for_tests(1);
        let outcome = VariableId::from_raw(1);
        let baseline = PopulationSelector::Rows(Arc::from((0..30).collect::<Vec<_>>()));
        let comparison = PopulationSelector::Rows(Arc::from((30..60).collect::<Vec<_>>()));
        let err = ChangeAttribution::new()
            .outcome(outcome)
            .baseline(baseline.clone())
            .comparison(comparison.clone())
            .robust(true)
            .with_sampling(DifferenceMeasure::VarianceDiff, 10, 1)
            .run(&model, &data, &ctx)
            .unwrap_err();
        assert!(matches!(err, AttributionError::Unsupported { .. }), "{err:?}");

        let err = ChangeAttribution::new()
            .outcome(outcome)
            .baseline(baseline)
            .comparison(comparison)
            .allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() })
            .robust(true)
            .with_sampling(DifferenceMeasure::MeanDiff, 30_000_000, 1)
            .run(&model, &data, &ctx)
            .unwrap_err();
        assert!(matches!(err, AttributionError::SizeLimit { .. }), "{err:?}");
    }
}
