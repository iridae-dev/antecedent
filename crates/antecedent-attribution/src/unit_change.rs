//! Per-unit change attribution via shared exogenous noise.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ComponentId, ExecutionContext, UnitChangeQuery, VariableId};
use antecedent_counterfactual::{AbductionMissingPolicy, CounterfactualEngine};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::DenseNodeId;
use antecedent_model::{CompiledCausalModel, MechanismWorkspace, ParentBatch, evaluate_column};

use crate::error::AttributionError;
use crate::prep::{require_input_components, require_shapley_config, resolve_outcome_dense};
use crate::result::{ComputeBudget, UnitChangeResult};
use crate::shapley::{CoalitionPayoff, estimate_shapley};

/// Attribute per-unit outcome change to input / mechanism components.
///
/// Abduces exogenous noise once, then evaluates the outcome mechanism on
/// coalition-mixed parent values with that fixed noise (Budhathoki-style factual
/// vs reference decomposition). Shapley values therefore attribute the real
/// mechanism payoff, not a linear surrogate.
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
    let exo = engine.abduct(data, AbductionMissingPolicy::Error)?;

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
    let mut sum_se2 = 0.0;
    let mut n_se = 0usize;

    let approximation =
        require_shapley_config(&query.allocation, "unit_change supports Shapley allocation")?;

    for (ui, &row) in rows.iter().enumerate() {
        let factual: Vec<f64> = parents
            .iter()
            .map(|&p| data.float64_values(p).map(|c| c[row]))
            .collect::<Result<Vec<_>, _>>()?;
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

        let est = estimate_shapley(&players, approximation, &mut payoff, ctx)?;
        budget.evaluations += est.budget.evaluations;
        budget.samples += est.budget.samples;
        cache_stats.hits += est.cache_stats.hits;
        cache_stats.misses += est.cache_stats.misses;
        if let Some(se) = est.monte_carlo_stderr {
            sum_se2 += se * se;
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
    // SE of the mean of independent per-unit estimates: √(Σ se_u²) / n.
    let mc_stderr = if n_se > 0 { Some(sum_se2.sqrt() / nu) } else { None };
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
