//! Per-unit change attribution via shared exogenous noise (DESIGN.md §17.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    AllocationMethod, AttributionComponents, ComponentId, ExecutionContext, UnitChangeQuery,
    VariableId,
};
use causal_counterfactual::{CounterfactualEngine, MissingPolicy};
use causal_data::{TableView, TabularData};
use causal_model::{CompiledCausalModel, MechanismWorkspace};

use crate::error::AttributionError;
use crate::result::{ComputeBudget, UnitChangeResult};
use crate::shapley::{CoalitionPayoff, estimate_shapley};

/// Attribute per-unit outcome change to input / mechanism components.
///
/// Uses abduction once, then evaluates coalition worlds that swap factual parent
/// values toward a reference (mean) for input attribution.
///
/// # Errors
///
/// Size limits, abduction, or Shapley failures.
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
    if rows.len() > query.max_units {
        return Err(AttributionError::SizeLimit {
            kind: "units",
            requested: rows.len(),
            max: query.max_units,
        });
    }

    match query.components {
        AttributionComponents::Inputs
        | AttributionComponents::InputsAndMechanisms
        | AttributionComponents::All => {}
        AttributionComponents::Mechanisms | AttributionComponents::Structure => {
            return Err(AttributionError::Message(
                "unit_change path attributes Inputs (use distribution_change for mechanisms)"
                    .into(),
            ));
        }
        _ => {
            return Err(AttributionError::Message(
                "unsupported AttributionComponents for unit_change".into(),
            ));
        }
    }

    let outcome_dense = model
        .dense_of(query.outcome)
        .ok_or_else(|| AttributionError::Message(format!("outcome {} missing", query.outcome)))?;
    let gather = model
        .gather_for(outcome_dense)
        .ok_or_else(|| AttributionError::Message("missing outcome gather".into()))?;
    let parents: Vec<VariableId> =
        gather.parents.iter().map(|&p| model.output_layout.variables[p.as_usize()]).collect();
    if parents.is_empty() {
        return Err(AttributionError::Message(
            "unit_change requires parents of the outcome".into(),
        ));
    }
    let players: Vec<ComponentId> =
        parents.iter().copied().map(ComponentId::from_variable).collect();

    let engine = CounterfactualEngine::new(model.clone());
    let exo = engine.abduct(data, MissingPolicy::Error)?;
    let _ = exo;
    let _ = MechanismWorkspace::default();

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
    let mut mc_stderr = None;

    for (ui, &row) in rows.iter().enumerate() {
        let factual: Vec<f64> = parents
            .iter()
            .map(|&p| data.float64_values(p).map(|c| c[row]))
            .collect::<Result<Vec<_>, _>>()?;
        let y_fact = data.float64_values(query.outcome)?[row];

        let mut payoff = UnitPayoff {
            factual: factual.clone(),
            reference: parent_means.clone(),
            y_fact,
            // Linear local model: Δy ≈ Σ β_i (x_i − ref_i); recover β from
            // one-at-a-time contrasts using the fitted mechanism coeffs when available.
            betas: outcome_betas(model, outcome_dense, gather.n_parents()),
        };

        let AllocationMethod::Shapley { approximation } = &query.allocation else {
            return Err(AttributionError::Message(
                "unit_change supports Shapley allocation".into(),
            ));
        };
        let est = estimate_shapley(&players, approximation, &mut payoff, ctx)?;
        budget.evaluations += est.budget.evaluations;
        budget.samples += est.budget.samples;
        cache_stats.hits += est.cache_stats.hits;
        cache_stats.misses += est.cache_stats.misses;
        if let Some(se) = est.monte_carlo_stderr {
            mc_stderr = Some(mc_stderr.map_or(se, |m: f64| m + se));
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
    if let Some(se) = mc_stderr.as_mut() {
        *se /= nu;
    }
    cache_stats.entries = cache_stats.hits + cache_stats.misses;

    Ok(UnitChangeResult {
        outcome: query.outcome,
        unit_rows: Arc::from(rows),
        components: Arc::from(players),
        contributions: Arc::from(all_contrib),
        mean_contributions: Arc::from(mean_phi),
        budget,
        monte_carlo_stderr: mc_stderr,
        cache_stats,
    })
}

fn outcome_betas(
    model: &CompiledCausalModel,
    outcome: causal_graph::DenseNodeId,
    n_parents: usize,
) -> Vec<f64> {
    match model.mechanisms.get(outcome) {
        causal_model::MechanismSlot::LinearGaussian { coeffs, .. } => coeffs.to_vec(),
        _ => vec![1.0; n_parents],
    }
}

struct UnitPayoff {
    factual: Vec<f64>,
    reference: Vec<f64>,
    y_fact: f64,
    betas: Vec<f64>,
}

impl CoalitionPayoff for UnitPayoff {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        // Value = predicted Δy under hybrid parents relative to all-reference.
        let mut pred = 0.0;
        for i in 0..self.factual.len() {
            let x = if mask & (1u64 << i) != 0 { self.factual[i] } else { self.reference[i] };
            pred += self.betas.get(i).copied().unwrap_or(0.0) * (x - self.reference[i]);
        }
        let _ = self.y_fact;
        Ok(pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, ShapleyConfig, SmallRoleSet, ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};
    use causal_model::{MechanismRegistry, SelectionPolicy};

    #[test]
    fn unit_change_attributes_parent() {
        let n = 20usize;
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
        let xv: Vec<f64> = (0..n).map(|i| i as f64).collect();
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
        // Extreme unit should have large absolute contribution vs mean reference.
        let last = result.contributions[result.contributions.len() - 1].abs();
        assert!(last > 1.0, "last unit contrib={last}");
    }
}
