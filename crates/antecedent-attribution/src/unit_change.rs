//! Per-unit change attribution via shared exogenous noise.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ComponentId, ExecutionContext, UnitChangeQuery, VariableId};
use antecedent_counterfactual::{AbductionMissingPolicy, CounterfactualEngine};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::DenseNodeId;
use antecedent_model::{CompiledCausalModel, MechanismWorkspace, ParentBatch, evaluate_column};

use crate::change_common::{UNIT_STREAM, stream_tag};
use crate::error::AttributionError;
use crate::prep::{require_input_components, require_shapley_config, resolve_outcome_dense};
use crate::result::{ComputeBudget, UnitChangeResult};
use crate::shapley::{CoalitionPayoff, estimate_shapley};

/// Attribute per-unit outcome change to input / mechanism components.
///
/// Abduces exogenous noise once, then evaluates the outcome mechanism on
/// coalition-mixed parent values with that fixed noise (the factual-vs-reference,
/// counterfactual Shapley decomposition of Budhathoki, Michailidis & Janzing 2022).
/// Shapley values therefore attribute the real mechanism payoff, not a linear
/// surrogate.
///
/// # Errors
///
/// Size limits, out-of-range `unit_rows`, abduction, or Shapley failures.
pub fn unit_change(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &UnitChangeQuery,
    ctx: &ExecutionContext,
) -> Result<UnitChangeResult, AttributionError> {
    query.validate()?;
    let n_all = data.row_count();
    let rows: Vec<usize> = match &query.unit_rows {
        Some(r) => r.to_vec(),
        None => (0..n_all).collect(),
    };
    for &row in &rows {
        if row >= n_all {
            return Err(AttributionError::PopulationOutOfRange {
                kind: "row",
                index: row,
                limit: n_all,
            });
        }
    }
    if rows.len() > query.max_units {
        return Err(AttributionError::SizeLimit {
            kind: "units",
            requested: rows.len(),
            max: query.max_units,
        });
    }

    require_input_components(
        query.components,
        "unit_change requires AttributionComponents::Inputs",
    )?;

    let outcome_dense = resolve_outcome_dense(model, query.outcome)?;
    let gather = model
        .gather_for(outcome_dense)
        .ok_or(AttributionError::MissingArtifact("missing outcome gather"))?;
    let parents: Vec<VariableId> =
        gather.parents.iter().map(|&p| model.output_layout.variables[p.as_usize()]).collect();
    if parents.is_empty() {
        return Err(AttributionError::unsupported("unit_change requires parents of the outcome"));
    }
    let players: Vec<ComponentId> =
        parents.iter().copied().map(ComponentId::from_variable).collect();

    let engine = CounterfactualEngine::from_ref(model);
    let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx)?;

    // Reference parent means.
    let mut parent_means = Vec::with_capacity(parents.len());
    for &p in &parents {
        let col = data.float64_values(p)?;
        parent_means.push(col.iter().sum::<f64>() / col.len().max(1) as f64);
    }

    let mut all_contrib = vec![0.0; rows.len() * players.len()];
    let mut mean_phi = vec![0.0; players.len()];
    let mut budget = ComputeBudget::default();
    let mut cache_stats = crate::result::CacheStats::default();
    // Per-player Σ_u se_{u,j}² over the units that reported a permutation standard error.
    let mut sum_se2 = vec![0.0; players.len()];
    let mut n_se = 0usize;

    let approximation =
        require_shapley_config(&query.allocation, "unit_change supports Shapley allocation")?;

    // Parent columns read once; the per-row loop needs one scalar per column,
    // and `float64_values` copies the whole column on every call.
    let parent_cols: Vec<Vec<f64>> =
        parents.iter().map(|&p| data.float64_values(p)).collect::<Result<Vec<_>, _>>()?;
    for (ui, &row) in rows.iter().enumerate() {
        let factual: Vec<f64> = parent_cols.iter().map(|c| c[row]).collect();
        let noise = exo.noise[outcome_dense.as_usize() * exo.n_units + row];

        let mut payoff = UnitPayoff {
            model,
            outcome: outcome_dense,
            factual,
            reference: parent_means.clone(),
            noise,
            parent_buf: vec![0.0; parents.len().max(1)],
            out_buf: vec![0.0; 1],
            noise_buf: vec![0.0; 1],
            ws: MechanismWorkspace::default(),
        };

        // Each unit draws its own permutations. A shared seed would make every unit see the
        // identical permutation sequence, so unit estimates would be positively correlated
        // and the pooled standard error below (which assumes independence) badly optimistic.
        let unit_config = (*approximation).with_seed(unit_seed(approximation.seed, row));
        let est = estimate_shapley(&players, &unit_config, &mut payoff, ctx)?;
        budget.evaluations += est.budget.evaluations;
        budget.samples += est.budget.samples;
        cache_stats.hits += est.cache_stats.hits;
        cache_stats.misses += est.cache_stats.misses;
        cache_stats.saturated |= est.cache_stats.saturated;
        if let Some(ses) = &est.component_mc_stderr {
            for (acc, se) in sum_se2.iter_mut().zip(ses) {
                *acc += se * se;
            }
            n_se += 1;
        }
        for (j, v) in est.values.iter().enumerate() {
            all_contrib[ui * players.len() + j] = *v;
            mean_phi[j] += *v;
        }
    }

    let nu = rows.len().max(1) as f64;
    for v in &mut mean_phi {
        *v /= nu;
    }
    // SE of each player's mean over independent per-unit estimates: √(Σ_u se_{u,j}²) / n.
    let component_se: Option<Vec<f64>> = (n_se > 0).then(|| pooled_stderr(&sum_se2, nu));
    let mc_stderr = component_se.as_ref().map(|se| se.iter().sum::<f64>() / se.len() as f64);
    cache_stats.entries = cache_stats.hits + cache_stats.misses;

    Ok(UnitChangeResult {
        outcome: query.outcome,
        unit_rows: Arc::from(rows),
        components: Arc::from(players),
        contributions: Arc::from(all_contrib),
        mean_contributions: Arc::from(mean_phi),
        budget,
        monte_carlo_stderr: mc_stderr,
        component_mc_stderr: component_se.map(Arc::from),
        cache_stats,
    })
}

/// Permutation seed of one unit: distinct rows get distinct streams (the map is a bijection
/// of the row for a fixed base seed), so per-unit estimates are independent.
fn unit_seed(base: u64, row: usize) -> u64 {
    stream_tag(stream_tag(UNIT_STREAM, base), row as u64)
}

/// Standard error of each player's mean over `n_units` independent unit estimates, given
/// `Σ_u se_{u,j}²` per player.
fn pooled_stderr(sum_se2: &[f64], n_units: f64) -> Vec<f64> {
    sum_se2.iter().map(|s| s.sqrt() / n_units).collect()
}

struct UnitPayoff<'a> {
    model: &'a CompiledCausalModel,
    outcome: DenseNodeId,
    factual: Vec<f64>,
    reference: Vec<f64>,
    noise: f64,
    parent_buf: Vec<f64>,
    out_buf: Vec<f64>,
    noise_buf: Vec<f64>,
    ws: MechanismWorkspace,
}

impl CoalitionPayoff for UnitPayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let n_par = self.factual.len();
        for i in 0..n_par {
            self.parent_buf[i] =
                if mask & (1u64 << i) != 0 { self.factual[i] } else { self.reference[i] };
        }
        self.noise_buf[0] = self.noise;
        let parents =
            ParentBatch { n_rows: 1, n_parents: n_par, values: &self.parent_buf[..n_par] };
        evaluate_column(
            self.model.mechanisms.get(self.outcome),
            parents,
            &self.noise_buf,
            &mut self.out_buf,
            &mut self.ws,
        )?;
        Ok(self.out_buf[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        AllocationMethod, CausalSchemaBuilder, MeasurementSpec, RoleHint, ShapleyConfig,
        SmallRoleSet, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_model::{MechanismRegistry, SelectionPolicy};
    use serde::Deserialize;

    #[test]
    fn unit_change_attributes_parent() {
        #[derive(Deserialize)]
        struct Fixture {
            unit_case: UnitCase,
        }
        #[derive(Deserialize)]
        struct UnitCase {
            x: Vec<f64>,
            contributions: Vec<f64>,
            mean_contribution: f64,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../conformance/attribution/mechanism_unit_change/expected.json"
        ))
        .unwrap();
        let n = fixture.unit_case.x.len();
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
        let xv = fixture.unit_case.x.clone();
        let yv: Vec<f64> = xv.iter().map(|x| 2.0 * x).collect();
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
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = compiled.with_mechanisms(store);
        let q = UnitChangeQuery::new(VariableId::from_raw(1), 20)
            .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let result = unit_change(&model, &data, &q, &ExecutionContext::for_tests(1)).unwrap();
        assert_eq!(result.components.len(), 1);
        assert_eq!(result.contributions.len(), fixture.unit_case.contributions.len());
        for (actual, expected) in
            result.contributions.iter().zip(fixture.unit_case.contributions.iter())
        {
            assert!((actual - expected).abs() < 1e-10, "actual={actual} expected={expected}");
        }
        assert!((result.mean_contributions[0] - fixture.unit_case.mean_contribution).abs() < 1e-10);
    }

    #[test]
    fn units_draw_independent_permutation_streams() {
        // Distinct rows → distinct seeds, for several base seeds: the estimates of two units
        // cannot share permutations, which is what the pooled standard error assumes.
        for base in [0u64, 1, 7, u64::MAX] {
            let seeds: std::collections::HashSet<u64> =
                (0..5_000usize).map(|row| unit_seed(base, row)).collect();
            assert_eq!(seeds.len(), 5_000, "base seed {base}");
            assert!(!seeds.contains(&base));
        }
        assert_ne!(unit_seed(1, 0), unit_seed(2, 0));
    }

    #[test]
    fn pooled_stderr_is_per_player_root_sum_of_squares_over_n() {
        // Player 0: unit se 3 and 4 over 2 units → √(9 + 16)/2 = 2.5; player 1: 0.
        let se = pooled_stderr(&[25.0, 0.0], 2.0);
        assert!((se[0] - 2.5).abs() < 1e-15);
        assert_eq!(se[1], 0.0);
    }

    /// `unit_rows` equal to `n` (first past the last valid index) must be a typed
    /// [`AttributionError::PopulationOutOfRange`], never a panic on column index.
    #[test]
    fn unit_change_rejects_out_of_range_unit_row() {
        let n = 8usize;
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
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let yv: Vec<f64> = xv.iter().map(|x| 2.0 * x).collect();
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
        // Bounds check runs before abduction / Shapley; an unfitted model is enough.
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = UnitChangeQuery::new(VariableId::from_raw(1), 20)
            .with_unit_rows([n])
            .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let err = unit_change(&model, &data, &q, &ExecutionContext::for_tests(1)).unwrap_err();
        assert_eq!(err, AttributionError::PopulationOutOfRange { kind: "row", index: n, limit: n });
    }
}
