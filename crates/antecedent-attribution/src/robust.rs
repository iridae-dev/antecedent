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

    let all_linear = players.iter().all(|c| {
        model_slot_is_linear(&baseline_mechs, graph_model, c.variable())
            && model_slot_is_linear(&comparison_mechs, graph_model, c.variable())
    });

    let approximation = require_shapley_config(
        &query.allocation,
        "distribution_change_robust currently supports Shapley allocation",
    )?;

    let (v0, v_full, estimate) = if all_linear {
        let mut payoff = RobustLinearPayoff {
            model: graph_model,
            baseline: &baseline_data,
            players: players.clone(),
            outcome: query.outcome,
            fitted: Vec::new(),
            columns: Vec::new(),
            node_pred: Vec::new(),
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
        let mut payoff = RobustMechanismPayoff::new(
            graph_model.clone(),
            baseline_mechs,
            comparison_mechs,
            &players,
            outcome_dense,
            options.n_samples,
            options.seed,
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
    /// Owned baseline data column (index into `RobustLinearPayoff::columns`).
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
    /// Non-player parent columns from the baseline population, fetched once.
    columns: Vec<Vec<f64>>,
    /// Flat `k × n` prediction scratch reused across coalitions.
    node_pred: Vec<f64>,
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
        self.columns.clear();
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
                    assert!(pj < i, "player parent must precede its child in topo order");
                    sources.push(ParentSource::Player(pj));
                } else {
                    let ci = if let Some(ci) = column_vars.iter().position(|&v| v == p) {
                        ci
                    } else {
                        column_vars.push(p);
                        self.columns.push(self.baseline.float64_values(p)?);
                        self.columns.len() - 1
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
        self.node_pred = vec![0.0; self.players.len() * self.n_rows];
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
        let n = self.n_rows;
        let Some(outcome_idx) = self.outcome_idx else {
            return Err(AttributionError::unsupported(
                "robust payoff: outcome is not among Shapley players",
            ));
        };
        for i in 0..self.players.len() {
            let fit = &self.fitted[i];
            let beta =
                if mask & (1u64 << i) != 0 { &fit.comparison_beta } else { &fit.baseline_beta };
            // Earlier players' predictions (`done`) feed the current column (`col`).
            let (done, rest) = self.node_pred.split_at_mut(i * n);
            let col = &mut rest[..n];
            for r in 0..n {
                let mut yhat = beta[0];
                for (pi, src) in fit.sources.iter().enumerate() {
                    let x = match *src {
                        ParentSource::Player(pj) => done[pj * n + r],
                        ParentSource::Column(ci) => self.columns[ci][r],
                    };
                    yhat += beta.get(pi + 1).copied().unwrap_or(0.0) * x;
                }
                col[r] = yhat;
            }
        }
        let col = &self.node_pred[outcome_idx * n..(outcome_idx + 1) * n];
        Ok(col.iter().sum::<f64>() / n.max(1) as f64)
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
/// The RNG is reseeded to the *same* stream at the start of every [`CoalitionPayoff::value`]
/// call (common random numbers), so two coalitions differ only in which mechanisms were
/// swapped, not in which random draws were used — this is what keeps the Shapley
/// differences of two Monte Carlo estimates numerically usable at a practical sample count,
/// the same technique `distribution_change`'s `MechanismSwapPayoff` uses.
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

impl<'a> RobustMechanismPayoff<'a> {
    fn new(
        template: CompiledCausalModel,
        baseline: CompiledMechanismStore,
        comparison: CompiledMechanismStore,
        players: &[ComponentId],
        outcome: DenseNodeId,
        n_samples: usize,
        seed: u64,
        ctx: &'a ExecutionContext,
    ) -> Self {
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
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let n = self.n_samples;
        // Common random numbers: fresh stream from the same seed every call, so
        // only the swapped mechanisms (via `mask`) differ between coalitions.
        let mut rng = self.ctx.rng.stream(0x526F_6275_7374_4D43_u64.wrapping_add(self.seed));
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
            template, baseline, comparison, &players, outcome, 20_000, 11, &ctx,
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
}
