//! Feature relevance under interventions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{
    ComponentId, ExecutionContext, Intervention, ShapleyConfig, Value, VariableId,
};
use causal_data::{TableView, TabularData};
use causal_model::{CompiledCausalModel, MechanismWorkspace, sample_interventional};

use crate::error::AttributionError;
use crate::result::FeatureRelevance;
use crate::shapley::{CoalitionPayoff, estimate_shapley};

/// Score interventional relevance of each `feature` for `outcome` via Shapley values.
///
/// For coalition `S`, `v(S) = E[Y | do(Xᵢ = μᵢ + δ ∀ i∈S)]` (features outside `S` free under
/// ancestral sampling). `δ` is the shift from each feature's empirical mean. Shapley values
/// `φᵢ` attribute the change from the empty intervention to intervening on all listed
/// features; the reported score is `|φᵢ|`.
///
/// # Errors
///
/// Sampling / size failures.
pub fn feature_relevance(
    model: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
    features: &[VariableId],
    delta: f64,
    n_samples: usize,
    max_features: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<FeatureRelevance>, AttributionError> {
    if features.len() > max_features {
        return Err(AttributionError::SizeLimit {
            kind: "features",
            requested: features.len(),
            max: max_features,
        });
    }
    if features.len() > 64 {
        return Err(AttributionError::SizeLimit {
            kind: "features",
            requested: features.len(),
            max: 64,
        });
    }
    let outcome_dense = model
        .dense_of(outcome)
        .ok_or_else(|| AttributionError::missing_var("outcome", outcome))?;

    let mut means = Vec::with_capacity(features.len());
    for &feat in features {
        let col = data.float64_values(feat)?;
        means.push(col.iter().sum::<f64>() / col.len().max(1) as f64);
    }

    let players: Vec<ComponentId> =
        features.iter().copied().map(ComponentId::from_variable).collect();
    let mut payoff = FeaturePayoff {
        model,
        features,
        means: &means,
        delta,
        outcome: outcome_dense,
        n_samples: n_samples.max(1),
        ctx,
        ws: MechanismWorkspace::default(),
        seed: 0xFEA7_u64,
    };
    let shapley = ShapleyConfig::exact();
    let est = estimate_shapley(&players, &shapley, &mut payoff, ctx)?;
    let mut out: Vec<FeatureRelevance> = features
        .iter()
        .zip(est.values.iter())
        .map(|(&feature, &phi)| FeatureRelevance { feature, outcome, score: phi.abs() })
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

struct FeaturePayoff<'a> {
    model: &'a CompiledCausalModel,
    features: &'a [VariableId],
    means: &'a [f64],
    delta: f64,
    outcome: causal_graph::DenseNodeId,
    n_samples: usize,
    ctx: &'a ExecutionContext,
    ws: MechanismWorkspace,
    seed: u64,
}

impl CoalitionPayoff for FeaturePayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let mut interventions = Vec::new();
        for (i, &feat) in self.features.iter().enumerate() {
            if mask & (1u64 << i) != 0 {
                interventions
                    .push(Intervention::set(feat, Value::f64(self.means[i] + self.delta)));
            }
        }
        // Common random numbers across coalitions (fixed seed).
        let mut rng = self.ctx.rng.stream(self.seed);
        let batch = sample_interventional(
            self.model,
            &interventions,
            self.n_samples,
            &mut rng,
            &mut self.ws,
            self.ctx,
        )?;
        let col = batch.column(self.outcome.as_usize())?;
        Ok(col.iter().sum::<f64>() / col.len().max(1) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};
    use causal_model::{MechanismRegistry, SelectionPolicy};

    #[test]
    fn shapley_relevance_ranks_causal_parent() {
        let n = 40usize;
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
            "z",
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
        let zv: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.3).sin()).collect();
        let yv: Vec<f64> = xv.iter().zip(zv.iter()).map(|(x, _)| 3.0 * x).collect();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(zv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = compiled.with_mechanisms(store);
        let scores = feature_relevance(
            &model,
            &data,
            VariableId::from_raw(2),
            &[VariableId::from_raw(0), VariableId::from_raw(1)],
            1.0,
            80,
            8,
            &ExecutionContext::for_tests(2),
        )
        .unwrap();
        assert_eq!(scores[0].feature, VariableId::from_raw(0));
        assert!(scores[0].score >= scores[1].score);
    }
}
