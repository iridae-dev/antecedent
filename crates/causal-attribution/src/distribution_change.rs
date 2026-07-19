//! Distribution-change attribution (pinned baseline-GCM parity; ).
//!
//! Fits mechanisms on baseline and comparison populations, then attributes the
//! change in the outcome marginal to mechanism replacements via Shapley values
//! (Budhathoki et al. 2021).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    AllocationMethod, AttributionComponents, ChangeAttributionQuery, ComponentId, ExecutionContext,
    ShapleyConfig, VariableId,
};
use causal_data::{TableView, TabularData};
use causal_graph::{BitSet, DenseNodeId, GraphWorkspace};
use causal_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
    MechanismWorkspace, SelectionPolicy, sample_observational,
};
use causal_stats::mean_var;

use crate::change_common::{measure_value, run_change_allocation, total_change, ChangeOptions};
use crate::error::AttributionError;
use crate::prep::{
    require_mechanism_or_joint, resolve_change_populations, resolve_outcome_dense,
};
use crate::result::ChangeAttributionResult;
use crate::shapley::CoalitionPayoff;

pub use crate::change_common::DifferenceMeasure;

/// Options for distribution-change attribution.
#[derive(Clone, Debug)]
pub struct DistributionChangeOptions {
    /// Difference measure on the outcome samples.
    pub measure: DifferenceMeasure,
    /// Samples drawn per coalition evaluation.
    pub n_samples: usize,
    /// RNG seed for sampling.
    pub seed: u64,
}

impl Default for DistributionChangeOptions {
    fn default() -> Self {
        let o = ChangeOptions::default_mean();
        Self { measure: o.measure, n_samples: o.n_samples, seed: o.seed }
    }
}

/// Attribute distributional change between baseline and comparison populations.
///
/// `graph_model` supplies structure; mechanisms are fit separately on each
/// population subset. Only mechanism components that are ancestors of the
/// outcome (inclusive) participate, unless `query.components` restricts further.
///
/// # Errors
///
/// Query validation, fit/sample failures, or Shapley size limits.
pub fn distribution_change(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &DistributionChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    query.validate()?;
    require_mechanism_or_joint(query.components)?;
    if matches!(query.components, AttributionComponents::All) {
        return Err(AttributionError::unsupported(
            "AttributionComponents::All requires dual graphs; use ChangeAttribution::run_structure \
             for Structure, or InputsAndMechanisms for joint input+mechanism change",
        ));
    }

    let (baseline_data, comparison_data) = resolve_change_populations(data, query)?;

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

    let outcome_dense = resolve_outcome_dense(graph_model, query.outcome)?;

    let (players, player_kinds) = joint_players(
        graph_model,
        outcome_dense,
        query.max_components,
        query.components,
    )?;
    if players.is_empty() {
        return Err(AttributionError::invalid_input("no components to attribute"));
    }

    let mut payoff = MechanismSwapPayoff {
        template: graph_model.clone(),
        baseline: baseline_mechs,
        comparison: comparison_mechs,
        baseline_data,
        comparison_data,
        players: players.clone(),
        player_kinds,
        outcome: outcome_dense,
        measure: options.measure,
        n_samples: options.n_samples,
        seed: options.seed,
        ctx,
        ws: MechanismWorkspace::default(),
        baseline_law: None,
    };

    let v0 = payoff.value(0)?;
    let full_mask = (1u64 << players.len()) - 1;
    let v_full = payoff.value(full_mask)?;
    let total = total_change(options.measure, v0, v_full);

    run_change_allocation(
        query.outcome,
        &players,
        &query.allocation,
        &mut payoff,
        total,
        Arc::from([]),
        ctx,
        Some(graph_model),
    )
}

/// Convenience: Shapley Monte Carlo distribution-change with defaults.
///
/// # Errors
///
/// See [`distribution_change`].
pub fn distribution_change_shapley(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
    baseline: causal_core::PopulationSelector,
    comparison: causal_core::PopulationSelector,
    shapley: ShapleyConfig,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    let query = ChangeAttributionQuery::new(outcome, baseline, comparison)
        .with_components(AttributionComponents::Mechanisms)
        .with_allocation(AllocationMethod::Shapley { approximation: shapley });
    distribution_change(graph_model, data, &query, &DistributionChangeOptions::default(), ctx)
}

pub(crate) fn mechanism_players(
    model: &CompiledCausalModel,
    outcome: DenseNodeId,
    max_components: usize,
) -> Result<Vec<ComponentId>, AttributionError> {
    let (players, _) =
        joint_players(model, outcome, max_components, AttributionComponents::Mechanisms)?;
    Ok(players)
}

/// Kind of Shapley player in joint change attribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PlayerKind {
    /// Swap fitted mechanism for this node.
    Mechanism,
    /// Replace observational draws with comparison/baseline empirical values.
    Input,
    /// Both mechanism swap and empirical input mix.
    Both,
}

pub(crate) fn joint_players(
    model: &CompiledCausalModel,
    outcome: DenseNodeId,
    max_components: usize,
    components: AttributionComponents,
) -> Result<(Vec<ComponentId>, Vec<PlayerKind>), AttributionError> {
    let mut ws = GraphWorkspace::default();
    let mut anc = BitSet::default();
    model.graph.ancestors_of(&[outcome], &mut anc, &mut ws);

    let mut players = Vec::new();
    let mut kinds = Vec::new();

    if matches!(
        components,
        AttributionComponents::Mechanisms
            | AttributionComponents::InputsAndMechanisms
            | AttributionComponents::All
    ) {
        for gather in model.parent_gathers.iter() {
            let node = gather.child;
            if !anc.contains(node) {
                continue;
            }
            let var = model.output_layout.variables[node.as_usize()];
            players.push(ComponentId::from_variable(var));
            kinds.push(PlayerKind::Mechanism);
        }
    }

    if matches!(
        components,
        AttributionComponents::Inputs
            | AttributionComponents::InputsAndMechanisms
            | AttributionComponents::All
    ) {
        if let Some(gather) = model.gather_for(outcome) {
            for &p in gather.parents.iter() {
                let var = model.output_layout.variables[p.as_usize()];
                let comp = ComponentId::from_variable(var);
                if let Some(idx) = players.iter().position(|&c| c == comp) {
                    kinds[idx] = PlayerKind::Both;
                } else {
                    players.push(comp);
                    kinds.push(PlayerKind::Input);
                }
            }
        }
    }

    if players.len() > max_components {
        return Err(AttributionError::SizeLimit {
            kind: "components",
            requested: players.len(),
            max: max_components,
        });
    }
    Ok((players, kinds))
}

/// Build a mechanism store that uses comparison slots for mechanism bits set in `mask`.
pub(crate) fn hybrid_mechanisms(
    baseline: &CompiledMechanismStore,
    comparison: &CompiledMechanismStore,
    model: &CompiledCausalModel,
    players: &[ComponentId],
    kinds: &[PlayerKind],
    mask: u64,
) -> CompiledMechanismStore {
    let n = model.n_nodes();
    let mut slots: Vec<MechanismSlot> = (0..n).map(|i| baseline.slots[i].clone()).collect();
    for (i, comp) in players.iter().enumerate() {
        if mask & (1u64 << i) == 0 {
            continue;
        }
        if matches!(kinds[i], PlayerKind::Input) {
            continue;
        }
        if let Some(dense) = model.dense_of(comp.variable()) {
            slots[dense.as_usize()] = comparison.slots[dense.as_usize()].clone();
        }
    }
    CompiledMechanismStore { slots: Arc::from(slots) }
}

struct MechanismSwapPayoff<'a> {
    template: CompiledCausalModel,
    baseline: CompiledMechanismStore,
    comparison: CompiledMechanismStore,
    baseline_data: TabularData,
    comparison_data: TabularData,
    players: Vec<ComponentId>,
    player_kinds: Vec<PlayerKind>,
    outcome: DenseNodeId,
    measure: DifferenceMeasure,
    n_samples: usize,
    seed: u64,
    ctx: &'a ExecutionContext,
    ws: MechanismWorkspace,
    /// Cached `(μ₀, σ₀²)` of the all-baseline outcome law for KL payoffs.
    baseline_law: Option<(f64, f64)>,
}

impl CoalitionPayoff for MechanismSwapPayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        if matches!(self.measure, DifferenceMeasure::GaussianKl) && self.baseline_law.is_none() {
            let (mu0, var0) = self.sample_outcome_law(0)?;
            self.baseline_law = Some((mu0, var0));
            if mask == 0 {
                return Ok(0.0);
            }
        }
        let (mu, var) = self.sample_outcome_law(mask)?;
        measure_value(self.measure, mask, mu, var, self.baseline_law)
    }
}

impl MechanismSwapPayoff<'_> {
    fn sample_outcome_law(&mut self, mask: u64) -> Result<(f64, f64), AttributionError> {
        use causal_core::{Intervention, Value};

        let store = hybrid_mechanisms(
            &self.baseline,
            &self.comparison,
            &self.template,
            &self.players,
            &self.player_kinds,
            mask,
        );
        let model = self.template.clone().with_mechanisms(store);
        let mut rng = self.ctx.rng.stream(0xDC01_u64.wrapping_add(self.seed));

        // Hard-set input/both players to the mean of the selected population.
        let mut interventions = Vec::new();
        for (i, &comp) in self.players.iter().enumerate() {
            if !matches!(self.player_kinds[i], PlayerKind::Input | PlayerKind::Both) {
                continue;
            }
            let data = if mask & (1u64 << i) != 0 {
                &self.comparison_data
            } else {
                &self.baseline_data
            };
            let col = data.float64_values(comp.variable())?;
            let mean = col.iter().sum::<f64>() / col.len().max(1) as f64;
            interventions.push(Intervention::set(comp.variable(), Value::f64(mean)));
        }

        let batch = if interventions.is_empty() {
            sample_observational(
                &model,
                self.n_samples.max(1),
                &mut rng,
                &mut self.ws,
                self.ctx,
            )?
        } else {
            causal_model::sample_interventional(
                &model,
                &interventions,
                self.n_samples.max(1),
                &mut rng,
                &mut self.ws,
                self.ctx,
            )?
        };
        let col = batch.column(self.outcome.as_usize())?;
        let (mu, var) = mean_var(col);
        Ok((mu, var.max(1e-12)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        AllocationMethod, AttributionComponents, CachePolicy, CausalSchemaBuilder, MeasurementSpec,
        PopulationSelector, RoleHint, ShapleyConfig, SmallRoleSet, ToleranceClass, ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};
    use causal_model::{MechanismRegistry, SelectionPolicy};

    fn two_period_chain() -> (CompiledCausalModel, TabularData) {
        // X → Y; baseline Y = X; comparison Y = X + 5 (mechanism change on Y only).
        let n = 80usize;
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
            let x = (i % 40) as f64 * 0.1;
            xv.push(x);
            if i < 40 {
                yv.push(1.0 + 2.0 * x);
            } else {
                yv.push(6.0 + 2.0 * x); // +5 intercept shift
            }
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

    #[test]
    fn attributes_mechanism_shift_to_y() {
        let (model, data) = two_period_chain();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 400,
            seed: 3,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        assert!(result.total_change > 3.0, "total={}", result.total_change);
        let y_contrib = result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(1))
            .expect("y component");
        let x_contrib = result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(0))
            .map_or(0.0, |c| c.contribution);
        assert!(
            y_contrib.contribution.abs() > x_contrib.abs(),
            "y={} x={} all={:?}",
            y_contrib.contribution,
            x_contrib,
            result.contributions
        );
        // Exact Shapley efficiency: Σφ = total_change (payoff uses CRN across coalitions).
        let phi_sum: f64 = result.contributions.iter().map(|c| c.contribution).sum();
        assert!(
            (phi_sum - result.total_change).abs() < 1e-6
                || ToleranceClass::MonteCarlo.close(phi_sum, result.total_change),
            "efficiency: Σφ={phi_sum} total={}",
            result.total_change
        );
    }

    #[test]
    fn exact_shapley_efficiency_sum_phi_equals_total_change() {
        let (model, data) = two_period_chain();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 800,
            seed: 11,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        let phi_sum: f64 = result.contributions.iter().map(|c| c.contribution).sum();
        assert!(
            (phi_sum - result.total_change).abs() < 1e-6
                || ToleranceClass::MonteCarlo.close(phi_sum, result.total_change),
            "Σφ={phi_sum} total_change={}",
            result.total_change
        );
        assert!(result.total_change.is_finite() && result.total_change.abs() > 1.0);
    }

    #[test]
    fn inputs_and_mechanisms_runs() {
        let (model, data) = two_period_chain();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_components(AttributionComponents::InputsAndMechanisms)
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let ctx = ExecutionContext::for_tests(1);
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 200,
            seed: 5,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        assert!(result.total_change.is_finite());
        assert!(!result.contributions.is_empty());
    }

    #[test]
    fn path_based_allocation_fills_breakdown() {
        let (model, data) = two_period_chain();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&model, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = model.with_mechanisms(store);
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_allocation(AllocationMethod::PathBased);
        let ctx = ExecutionContext::for_tests(1);
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 200,
            seed: 7,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        assert!(!result.path_breakdown.is_empty(), "path_breakdown should be populated");
        assert!(result.total_change.is_finite());
    }
}
