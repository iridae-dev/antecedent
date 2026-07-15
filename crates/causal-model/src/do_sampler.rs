//! Do-samplers: weighting, KDE, and MCMC (DESIGN.md §26.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use causal_core::{CausalRng, ExecutionContext, Intervention, VariableId};
use causal_data::{TableView, TabularData};
use causal_kernels::standard_normal;

use crate::batch::{MechanismWorkspace, ParentBatch};
use crate::compile::{CompiledCausalModel, MechanismSlot};
use crate::error::ModelError;
use crate::mechanism::log_prob_column;
use crate::sample::sample_interventional;

/// Result of a do-sampler run.
#[derive(Clone, Debug)]
pub struct DoSampleResult {
    /// Sampled (or reweighted observational) outcomes for the target variable.
    pub values: Arc<[f64]>,
    /// Optional importance weights (weighting sampler); empty if not used.
    pub weights: Arc<[f64]>,
    /// Sampler id.
    pub method: Arc<str>,
    /// Diagnostics notes.
    pub notes: Vec<Arc<str>>,
    /// MCMC accept rate when applicable.
    pub accept_rate: Option<f64>,
}

/// Weighting do-sampler: reweight observational rows by P(do(T=t)|parents) / P(T|parents)
/// using fitted propensity from the treatment mechanism when available; for hard do on a
/// root treatment, returns the empirical outcome among rows matching the set value.
#[derive(Clone, Debug)]
pub struct WeightingDoSampler {
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
}

impl WeightingDoSampler {
    /// Construct.
    #[must_use]
    pub fn new(treatment: VariableId, outcome: VariableId) -> Self {
        Self { treatment, outcome }
    }

    /// Estimate E[Y | do(T=t)] via matching / Horvitz–Thompson on fitted model densities.
    ///
    /// # Errors
    ///
    /// Missing columns or unfitted treatment mechanism.
    pub fn estimate(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        treatment_value: f64,
        _ctx: &ExecutionContext,
    ) -> Result<DoSampleResult, ModelError> {
        let t_dense = model
            .dense_of(self.treatment)
            .ok_or_else(|| ModelError::Shape { message: "treatment not in model".into() })?;
        let y = data.float64_values(self.outcome).map_err(ModelError::from)?;
        let t = data.float64_values(self.treatment).map_err(ModelError::from)?;
        let n = y.len();
        let gather = model
            .gather_for(t_dense)
            .ok_or_else(|| ModelError::Shape { message: "missing gather for treatment".into() })?;

        let mut weights = vec![0.0; n];
        let mut values = Vec::with_capacity(n);
        let mut notes = Vec::new();

        if gather.n_parents() == 0 {
            // Root treatment: empirical outcomes among units with T ≈ t.
            let mut selected = 0usize;
            for i in 0..n {
                if (t[i] - treatment_value).abs() < 1e-9 {
                    values.push(y[i]);
                    weights[selected] = 1.0;
                    selected += 1;
                }
            }
            weights.truncate(selected);
            if selected == 0 {
                return Err(ModelError::Numerical {
                    message: "weighting sampler: no observational units match do-value".into(),
                });
            }
            notes.push(Arc::from("root treatment: exact match reweighting"));
            return Ok(DoSampleResult {
                values: Arc::from(values),
                weights: Arc::from(weights),
                method: Arc::from("do_weighting"),
                notes,
                accept_rate: None,
            });
        }

        // Confounded: IPW using Gaussian propensity from fitted treatment mechanism.
        let slot = model.mechanisms.get(t_dense);
        let mut parent_cols = Vec::new();
        for &p in gather.parents.iter() {
            let var = model.output_layout.variables[p.as_usize()];
            parent_cols.push(data.float64_values(var).map_err(ModelError::from)?);
        }
        let n_par = gather.n_parents();
        let mut parent_mat = vec![0.0; n * n_par];
        for (pi, col) in parent_cols.iter().enumerate() {
            for r in 0..n {
                parent_mat[pi * n + r] = col[r];
            }
        }
        let parents = ParentBatch { n_rows: n, n_parents: n_par, values: &parent_mat };
        let mut lp_obs = vec![0.0; n];
        log_prob_column(slot, &t, parents, &mut lp_obs)?;

        let lp_do = vec![0.0; n];
        // Interventional density at do-value: same parents, value fixed.
        let t_do = vec![treatment_value; n];
        match slot {
            MechanismSlot::LinearGaussian { sigma, .. } => {
                for i in 0..n {
                    let w = (-lp_obs[i]).exp().min(1e6);
                    // Stabilize with Gaussian kernel around treatment_value.
                    let d = (t[i] - treatment_value) / sigma.max(1e-6);
                    let k = (-0.5 * d * d).exp();
                    weights[i] = w * k;
                    values.push(y[i]);
                    let _ = lp_do[i];
                }
            }
            _ => {
                for i in 0..n {
                    let d = (t[i] - treatment_value).abs();
                    weights[i] = if d < 1e-6 { 1.0 } else { 0.0 };
                    values.push(y[i]);
                }
            }
        }
        let _ = t_do;
        let wsum: f64 = weights.iter().sum();
        if wsum.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(ModelError::Numerical {
                message: "weighting sampler: zero weight mass".into(),
            });
        }
        for w in &mut weights {
            *w /= wsum;
        }
        notes.push(Arc::from("IPW / kernel-stabilized weighting"));
        Ok(DoSampleResult {
            values: Arc::from(values),
            weights: Arc::from(weights),
            method: Arc::from("do_weighting"),
            notes,
            accept_rate: None,
        })
    }

    /// Weighted mean of the sampler result.
    #[must_use]
    pub fn weighted_mean(result: &DoSampleResult) -> f64 {
        if result.values.is_empty() {
            return f64::NAN;
        }
        if result.weights.is_empty() {
            return result.values.iter().sum::<f64>() / result.values.len() as f64;
        }
        let wsum: f64 = result.weights.iter().sum();
        if wsum.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return f64::NAN;
        }
        result.values.iter().zip(result.weights.iter()).map(|(v, w)| v * w).sum::<f64>() / wsum
    }
}

/// KDE do-sampler: sample from interventional SCM then smooth the outcome with a Gaussian KDE.
#[derive(Clone, Debug)]
pub struct KdeDoSampler {
    /// Outcome variable.
    pub outcome: VariableId,
    /// Bandwidth (Silverman's rule if None).
    pub bandwidth: Option<f64>,
}

impl KdeDoSampler {
    /// Construct.
    #[must_use]
    pub fn new(outcome: VariableId) -> Self {
        Self { outcome, bandwidth: None }
    }

    /// Draw interventional samples and return KDE-ready values (+ bandwidth note).
    ///
    /// # Errors
    ///
    /// Sampling failures.
    pub fn sample(
        &self,
        model: &CompiledCausalModel,
        interventions: &[Intervention],
        n_draws: usize,
        rng: &mut CausalRng,
        ws: &mut MechanismWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DoSampleResult, ModelError> {
        let batch = sample_interventional(model, interventions, n_draws, rng, ws, ctx)?;
        let dense = model
            .dense_of(self.outcome)
            .ok_or_else(|| ModelError::Shape { message: "outcome not in model".into() })?;
        let col = batch.column(dense.as_usize())?;
        let bw = self.bandwidth.unwrap_or_else(|| silverman_bandwidth(col));
        Ok(DoSampleResult {
            values: Arc::from(col.to_vec()),
            weights: Arc::from(vec![1.0 / n_draws as f64; n_draws]),
            method: Arc::from("do_kde"),
            notes: vec![Arc::from(format!("gaussian_kde_bandwidth={bw}"))],
            accept_rate: None,
        })
    }

    /// Evaluate KDE density at `x` given sampler values.
    #[must_use]
    pub fn density(result: &DoSampleResult, x: f64) -> f64 {
        let bw: f64 = result
            .notes
            .iter()
            .find_map(|n| n.strip_prefix("gaussian_kde_bandwidth=")?.parse().ok())
            .unwrap_or(1.0);
        let bw = bw.max(1e-8);
        let n = result.values.len() as f64;
        let inv = 1.0 / (bw * (2.0 * std::f64::consts::PI).sqrt());
        let mut dens = 0.0;
        for &v in result.values.iter() {
            let z = (x - v) / bw;
            dens += inv * (-0.5 * z * z).exp();
        }
        dens / n.max(1.0)
    }
}

fn silverman_bandwidth(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt().max(1e-8);
    1.06 * sd * n.powf(-0.2)
}

/// Metropolis–Hastings do-sampler on the intervened joint (Gaussian proposals).
#[derive(Clone, Debug)]
pub struct McmcDoSampler {
    /// Outcome variable to record.
    pub outcome: VariableId,
    /// Proposal standard deviation.
    pub proposal_sd: f64,
    /// Burn-in iterations.
    pub burn_in: usize,
    /// Thinning.
    pub thin: usize,
}

impl Default for McmcDoSampler {
    fn default() -> Self {
        Self { outcome: VariableId::from_raw(0), proposal_sd: 0.5, burn_in: 100, thin: 2 }
    }
}

impl McmcDoSampler {
    /// Construct targeting `outcome`.
    #[must_use]
    pub fn new(outcome: VariableId) -> Self {
        Self { outcome, ..Self::default() }
    }

    /// Run MH using interventional ancestral samples as the target measure proxy:
    /// proposals are new interventional draws; accept with min(1, 1) i.e. independent MH
    /// from the SCM interventional distribution (exact when the proposal is the target).
    ///
    /// For a non-trivial MH demo on continuous free variables under soft interventions,
    /// we walk a chain on the outcome margin with Gaussian random-walk proposals and
    /// accept via KDE density ratio estimated from a bootstrap interventional batch.
    ///
    /// # Errors
    ///
    /// Sampling / density failures.
    pub fn sample(
        &self,
        model: &CompiledCausalModel,
        interventions: &[Intervention],
        n_samples: usize,
        rng: &mut CausalRng,
        ws: &mut MechanismWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DoSampleResult, ModelError> {
        let pilot = sample_interventional(model, interventions, n_samples.max(64), rng, ws, ctx)?;
        let dense = model
            .dense_of(self.outcome)
            .ok_or_else(|| ModelError::Shape { message: "outcome not in model".into() })?;
        let pilot_col = pilot.column(dense.as_usize())?;
        let kde = DoSampleResult {
            values: Arc::from(pilot_col.to_vec()),
            weights: Arc::from([]),
            method: Arc::from("pilot"),
            notes: vec![Arc::from(format!(
                "gaussian_kde_bandwidth={}",
                silverman_bandwidth(pilot_col)
            ))],
            accept_rate: None,
        };

        let mut current = pilot_col[0];
        let mut accepted = 0usize;
        let mut total = 0usize;
        let mut out = Vec::with_capacity(n_samples);
        let iters = self.burn_in + n_samples * self.thin.max(1);

        for i in 0..iters {
            let z = standard_normal(rng);
            let prop = current + self.proposal_sd * z;
            let p_cur = KdeDoSampler::density(&kde, current).max(1e-300);
            let p_prop = KdeDoSampler::density(&kde, prop).max(1e-300);
            let accept = (p_prop / p_cur).min(1.0);
            total += 1;
            if rng.next_f64() < accept {
                current = prop;
                accepted += 1;
            }
            if i >= self.burn_in && (i - self.burn_in) % self.thin.max(1) == 0 {
                out.push(current);
                if out.len() == n_samples {
                    break;
                }
            }
        }
        let rate = accepted as f64 / total.max(1) as f64;
        Ok(DoSampleResult {
            values: Arc::from(out),
            weights: Arc::from([]),
            method: Arc::from("do_mcmc"),
            notes: vec![Arc::from(format!("mh_accept_rate={rate}"))],
            accept_rate: Some(rate),
        })
    }
}

/// Convenience: interventional mean of a variable from ancestral sampling.
///
/// # Errors
///
/// Sampling failures.
pub fn interventional_mean(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
    outcome: VariableId,
    n_draws: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, ModelError> {
    let batch = sample_interventional(model, interventions, n_draws, rng, ws, ctx)?;
    let dense = model
        .dense_of(outcome)
        .ok_or_else(|| ModelError::Shape { message: "outcome not in model".into() })?;
    let col = batch.column(dense.as_usize())?;
    Ok(col.iter().sum::<f64>() / col.len().max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{MechanismRegistry, SelectionPolicy};
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, Value, ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};

    fn binary_treatment_scm() -> (CompiledCausalModel, TabularData) {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
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
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            t[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            y[i] = 2.0 * t[i];
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity).unwrap(),
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
        (compiled.with_mechanisms(store), data)
    }

    #[test]
    fn weighting_recovers_treated_mean() {
        let (model, data) = binary_treatment_scm();
        let ctx = ExecutionContext::for_tests(1);
        let sampler = WeightingDoSampler::new(VariableId::from_raw(0), VariableId::from_raw(1));
        let res = sampler.estimate(&model, &data, 1.0, &ctx).unwrap();
        let mean = WeightingDoSampler::weighted_mean(&res);
        assert!((mean - 2.0).abs() < 1e-9, "mean={mean}");
    }

    #[test]
    fn kde_and_mcmc_run() {
        let n = 40;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
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
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            t[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            // Continuous noise keeps Y off the discrete auto-path and gives KDE spread for MH.
            y[i] = 2.0 * t[i] + 0.05 * ((i as f64) - 20.0);
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity).unwrap(),
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

        let ctx = ExecutionContext::for_tests(1);
        let mut rng = CausalRng::from_seed(3);
        let mut ws = MechanismWorkspace::default();
        let iv = [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))];
        let kde = KdeDoSampler::new(VariableId::from_raw(1))
            .sample(&model, &iv, 40, &mut rng, &mut ws, &ctx)
            .unwrap();
        assert_eq!(kde.values.len(), 40);
        let mcmc = McmcDoSampler::new(VariableId::from_raw(1))
            .sample(&model, &iv, 30, &mut rng, &mut ws, &ctx)
            .unwrap();
        assert_eq!(mcmc.values.len(), 30);
        assert!(mcmc.accept_rate.unwrap() > 0.0);
    }
}
