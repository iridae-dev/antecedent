//! Feature relevance under interventions (DESIGN.md §17.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{ExecutionContext, Intervention, Value, VariableId};
use causal_data::{TableView, TabularData};
use causal_model::{CompiledCausalModel, MechanismWorkspace, sample_interventional};

use crate::error::AttributionError;
use crate::result::FeatureRelevance;

/// Score interventional relevance of each `feature` for `outcome`.
///
/// Relevance = |E[Y | do(X=μ+δ/2)] − E[Y | do(X=μ−δ/2)]|.
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
    let outcome_dense = model
        .dense_of(outcome)
        .ok_or_else(|| AttributionError::Message(format!("outcome {outcome} missing")))?;
    let mut rng = causal_core::CausalRng::from_seed(0);
    let mut ws = MechanismWorkspace::default();
    let mut out = Vec::with_capacity(features.len());
    for &feat in features {
        let col = data.float64_values(feat)?;
        let mean = col.iter().sum::<f64>() / col.len().max(1) as f64;
        let hi = sample_interventional(
            model,
            &[Intervention::set(feat, Value::f64(mean + 0.5 * delta))],
            n_samples.max(1),
            &mut rng,
            &mut ws,
            ctx,
        )?;
        let lo = sample_interventional(
            model,
            &[Intervention::set(feat, Value::f64(mean - 0.5 * delta))],
            n_samples.max(1),
            &mut rng,
            &mut ws,
            ctx,
        )?;
        let hi_m =
            hi.column(outcome_dense.as_usize())?.iter().sum::<f64>() / n_samples.max(1) as f64;
        let lo_m =
            lo.column(outcome_dense.as_usize())?.iter().sum::<f64>() / n_samples.max(1) as f64;
        out.push(FeatureRelevance { feature: feat, outcome, score: (hi_m - lo_m).abs() });
    }
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}
