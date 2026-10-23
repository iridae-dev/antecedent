//! Pooled differential measurement error carried to a target population.
//!
//! Python passes each source population's validation counts, the target's optional own
//! validation counts and the target's recorded positives; the random-effects pooling, the
//! propagation draws and every refusal are Rust's
//! (`antecedent::analysis::transported_measurement::transport_pooled_differential_prevalence`).
//! The interval is an uncalibrated simulation interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::transported_measurement::{
    PooledDifferentialRequest, PooledRate, PopulationValidation,
    transport_pooled_differential_prevalence,
};
use pyo3::prelude::*;

type Counts = (u64, u64, u64, u64);

/// Corrected target prevalence with the pooled differential error propagated.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PooledDifferentialSection {
    pub target_population: String,
    pub prevalence: f64,
    pub interval: (f64, f64),
    /// `(pooled mean logit, its variance, between-population variance, target rate mean,
    /// target rate standard deviation)`.
    pub sensitivity: (f64, f64, f64, f64, f64),
    /// Same layout as `sensitivity`.
    pub specificity: (f64, f64, f64, f64, f64),
    pub discarded_draws: usize,
    pub truncated_draws: usize,
    pub draws: Vec<f64>,
    pub calibration: String,
}

const fn rate(value: &PooledRate) -> (f64, f64, f64, f64, f64) {
    (
        value.mean_logit,
        value.mean_variance,
        value.between_population_variance,
        value.target_rate,
        value.target_rate_sd,
    )
}

fn validation(population: String, counts: Counts) -> PopulationValidation {
    PopulationValidation {
        population,
        true_positive: counts.0,
        false_negative: counts.1,
        true_negative: counts.2,
        false_positive: counts.3,
    }
}

/// Correct the target's recorded prevalence with sensitivity and specificity pooled across
/// differentially validated populations.
#[pyfunction]
#[pyo3(signature = (sources, target_population, observed_positive, observed_total, *,
    target_validation=None, draws=4000, seed=1, threads=None))]
#[allow(clippy::too_many_arguments)]
fn measurement_pooled_differential_transport(
    py: Python<'_>,
    sources: Vec<(String, u64, u64, u64, u64)>,
    target_population: String,
    observed_positive: u64,
    observed_total: u64,
    target_validation: Option<Counts>,
    draws: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PooledDifferentialSection> {
    if threads == Some(0) {
        return Err(crate::value_err("threads must be positive"));
    }
    let ctx = antecedent_core::ExecutionContext::production(
        seed,
        threads.unwrap_or_else(antecedent_core::default_user_threads),
    );
    let sources: Vec<PopulationValidation> = sources
        .into_iter()
        .map(|(population, tp, fn_, tn, fp)| validation(population, (tp, fn_, tn, fp)))
        .collect();
    let target = target_validation.map(|counts| validation(target_population.clone(), counts));
    crate::detach_catch(py, move || {
        let fit = transport_pooled_differential_prevalence(
            PooledDifferentialRequest {
                sources: &sources,
                target_validation: target.as_ref(),
                target_population: &target_population,
                observed_positive,
                observed_total,
                draws,
            },
            &ctx,
        )
        .map_err(crate::py_err)?;
        Ok(PooledDifferentialSection {
            target_population: fit.target_population.clone(),
            prevalence: fit.prevalence,
            interval: fit.interval,
            sensitivity: rate(&fit.sensitivity),
            specificity: rate(&fit.specificity),
            discarded_draws: fit.discarded_draws,
            truncated_draws: fit.truncated_draws,
            draws: fit.draws,
            calibration: fit.calibration.to_owned(),
        })
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PooledDifferentialSection>()?;
    m.add_function(wrap_pyfunction!(measurement_pooled_differential_transport, m)?)
}
