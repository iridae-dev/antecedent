//! Estimand-level methods on a fitted GCM: arrow strengths, population do-contrasts,
//! multi-outcome shared-abduction contrasts, simultaneous counterfactuals, the three
//! do-samplers and the model evaluation report.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent::CausalError;
use antecedent::gcm::{
    CounterfactualEngine, DoSampleResult, KdeDoSampler, McmcDoSampler, MechanismWorkspace,
    ModelEvaluator, WeightingDoSampler, arrow_strengths, counterfactual_contrasts,
    interventional_mean, population_do_contrast, simultaneous_hard_counterfactual,
};
use antecedent_core::{Intervention, StreamDomain, Value, VariableId};
use antecedent_data::{TableView, TabularData};
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::gcm_api::PyFittedGcm;
use crate::{detach_catch, py_err, py_execution_context, py_msg, resolve_user_threads};

/// One edge's model-implied strength `coefficient^2 * Var(parent)`.
#[pyclass(skip_from_py_object, name = "ArrowStrength")]
#[derive(Clone, Debug)]
pub struct PyArrowStrength {
    /// Parent variable.
    #[pyo3(get)]
    pub parent: String,
    /// Child variable.
    #[pyo3(get)]
    pub child: String,
    /// The cited causal strength in nats, `0.5 * ln(1 + coefficient^2 * Var(parent) / sigma^2)`
    /// with `sigma^2` the child's noise variance: the divergence of the joint from the joint with
    /// this edge cut. Non-negative and scale-free.
    #[pyo3(get)]
    pub strength: f64,
    /// `coefficient^2 * Var(parent)`, the variance the edge adds to the child.
    #[pyo3(get)]
    pub variance_contribution: f64,
    /// The edge coefficient, carrying the sign the strength discards.
    #[pyo3(get)]
    pub coefficient: f64,
}

#[pymethods]
impl PyArrowStrength {
    fn __repr__(&self) -> String {
        format!(
            "ArrowStrength({} -> {}, strength={}, variance_contribution={}, coefficient={})",
            self.parent, self.child, self.strength, self.variance_contribution, self.coefficient
        )
    }
}

/// `E[child | do(parent = mu + delta/2)] - E[child | do(parent = mu - delta/2)]` with its
/// Monte Carlo standard error (common random numbers across the two arms).
#[pyclass(skip_from_py_object, name = "PopulationDoContrast")]
#[derive(Clone, Debug)]
pub struct PyPopulationDoContrast {
    /// Signed contrast.
    #[pyo3(get)]
    pub contrast: f64,
    /// Standard error of the paired per-draw differences.
    #[pyo3(get)]
    pub stderr: f64,
    /// Parent variable.
    #[pyo3(get)]
    pub parent: String,
    /// Child variable.
    #[pyo3(get)]
    pub child: String,
    /// The width `delta` of the contrast.
    #[pyo3(get)]
    pub delta: f64,
}

#[pymethods]
impl PyPopulationDoContrast {
    fn __repr__(&self) -> String {
        format!(
            "PopulationDoContrast({} -> {}, contrast={}, stderr={})",
            self.parent, self.child, self.contrast, self.stderr
        )
    }
}

/// The unit-level contrast of one outcome between two worlds.
#[pyclass(skip_from_py_object, name = "OutcomeContrast")]
pub struct PyOutcomeContrast {
    /// The outcome the contrast reads.
    #[pyo3(get)]
    pub outcome: String,
    /// Mean of the unit effects.
    #[pyo3(get)]
    pub mean_ite: f64,
    /// Per-unit effects, aligned with the factual rows.
    #[pyo3(get)]
    pub unit_effects: Py<PyArray1<f64>>,
}

#[pymethods]
impl PyOutcomeContrast {
    fn __repr__(&self) -> String {
        format!("OutcomeContrast(outcome={:?}, mean_ite={})", self.outcome, self.mean_ite)
    }
}

/// Contrasts of several outcomes between two simultaneous-intervention worlds that share one
/// abduced noise state.
#[pyclass(skip_from_py_object, name = "ContrastSet")]
pub struct PyContrastSet {
    /// One contrast per requested outcome, in the order requested.
    #[pyo3(get)]
    pub contrasts: Vec<Py<PyOutcomeContrast>>,
    /// How noise was obtained.
    #[pyo3(get)]
    pub noise_inference: String,
}

#[pymethods]
impl PyContrastSet {
    /// The contrast of `outcome`, or `None` when it was not requested.
    fn contrast(&self, py: Python<'_>, outcome: &str) -> Option<Py<PyOutcomeContrast>> {
        self.contrasts.iter().find(|c| c.borrow(py).outcome == outcome).map(|c| c.clone_ref(py))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let names: Vec<String> =
            self.contrasts.iter().map(|c| c.borrow(py).outcome.clone()).collect();
        format!("ContrastSet(outcomes={names:?}, noise_inference={:?})", self.noise_inference)
    }
}

/// The result of a do-sampler run.
#[pyclass(skip_from_py_object, name = "DoSample")]
pub struct PyDoSample {
    inner: DoSampleResult,
    /// Weighted mean of the outcome (a plain mean when the sampler carries no weights).
    #[pyo3(get)]
    pub mean: f64,
    /// Sampler id: `do_weighting`, `do_kde` or `do_mcmc`.
    #[pyo3(get)]
    pub method: String,
    /// Diagnostic notes.
    #[pyo3(get)]
    pub notes: Vec<String>,
    /// Metropolis-Hastings accept rate, MCMC only.
    #[pyo3(get)]
    pub accept_rate: Option<f64>,
    /// Gaussian KDE bandwidth, when the result carries a density estimate.
    #[pyo3(get)]
    pub bandwidth: Option<f64>,
    /// Effective sample size of the normalized importance weights, confounded branch only.
    #[pyo3(get)]
    pub ess: Option<f64>,
}

#[pymethods]
impl PyDoSample {
    /// Sampled outcome values.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.values)
    }

    /// Per-value weights; empty when the sampler carries none.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.weights)
    }

    /// The Gaussian-KDE density of the outcome at `x`, over the sampled values.
    fn density(&self, x: f64) -> f64 {
        KdeDoSampler::density(&self.inner, x)
    }

    fn __len__(&self) -> usize {
        self.inner.values.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "DoSample(method={:?}, n={}, mean={})",
            self.method,
            self.inner.values.len(),
            self.mean
        )
    }
}

/// The model evaluation report: likelihood, residual and local-Markov checks and a
/// shuffled-outcome baseline.
#[pyclass(skip_from_py_object, name = "ModelEvaluation")]
#[derive(Clone, Debug)]
pub struct PyModelEvaluation {
    /// In-sample mean log-likelihood; `-inf` when a row has zero probability.
    #[pyo3(get)]
    pub in_sample_loglik: f64,
    /// Observed rows with zero probability under the model.
    #[pyo3(get)]
    pub zero_probability_rows: usize,
    /// Mean absolute residual for invertible nodes.
    #[pyo3(get)]
    pub mean_abs_residual: f64,
    /// Residual-independence p-values against non-parent covariates.
    #[pyo3(get)]
    pub residual_independence_p: Vec<f64>,
    /// Local Markov check p-values.
    #[pyo3(get)]
    pub local_markov_p: Vec<f64>,
    /// Mean log-likelihood under shuffled outcomes.
    #[pyo3(get)]
    pub permutation_loglik: f64,
    /// Whether the model is falsified at `alpha` after Bonferroni correction.
    #[pyo3(get)]
    pub falsified: bool,
    /// Significance level used.
    #[pyo3(get)]
    pub alpha: f64,
    /// Notes.
    #[pyo3(get)]
    pub notes: Vec<String>,
}

#[pymethods]
impl PyModelEvaluation {
    fn __repr__(&self) -> String {
        format!(
            "ModelEvaluation(in_sample_loglik={}, falsified={}, alpha={})",
            self.in_sample_loglik, self.falsified, self.alpha
        )
    }
}

/// Hard `do` values by variable name.
fn hard_interventions(
    data: &TabularData,
    values: &HashMap<String, f64>,
) -> PyResult<Vec<Intervention>> {
    let mut pairs = Vec::with_capacity(values.len());
    for (name, value) in values {
        pairs.push((data.schema().id_of(name).map_err(py_err)?, *value));
    }
    pairs.sort_by_key(|(id, _)| id.raw());
    Ok(pairs.into_iter().map(|(id, v)| Intervention::set(id, Value::f64(v))).collect())
}

fn variable_of(data: &TabularData, name: &str) -> PyResult<VariableId> {
    data.schema().id_of(name).map_err(py_err)
}

#[pymethods]
impl PyFittedGcm {
    /// The model-implied strength of every edge: `coefficient^2 * Var(parent)`.
    ///
    /// Defined for linear-family mechanisms; a child with a non-linear mechanism refuses
    /// rather than reporting an approximation.
    fn arrow_strengths(&self) -> PyResult<Vec<PyArrowStrength>> {
        let strengths =
            arrow_strengths(&self.inner.model).map_err(|e| py_err(CausalError::from(e)))?;
        let name = |id: VariableId| {
            self.names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("V{}", id.raw()))
        };
        Ok(strengths
            .into_iter()
            .map(|s| PyArrowStrength {
                parent: name(s.parent),
                child: name(s.child),
                strength: s.strength,
                variance_contribution: s.variance_contribution,
                coefficient: s.coefficient,
            })
            .collect())
    }

    /// Population do-contrast of `parent` on `child`: the child's interventional mean at
    /// `mean(parent) + delta/2` minus the one at `mean(parent) - delta/2`, common random
    /// numbers across the arms. `max_units` is the most observed rows the contrast will accept
    /// (default: all); a larger table is refused, not truncated.
    #[pyo3(signature = (parent, child, delta, *, max_units=None, seed=0, threads=None))]
    fn population_do_contrast(
        &self,
        py: Python<'_>,
        parent: String,
        child: String,
        delta: f64,
        max_units: Option<usize>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<PyPopulationDoContrast> {
        let model = self.inner.clone();
        let data = Arc::clone(&self.data);
        let (parent_id, child_id) = (variable_of(&data, &parent)?, variable_of(&data, &child)?);
        let result = detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let cap = max_units.unwrap_or_else(|| data.row_count());
            population_do_contrast(&model.model, &data, parent_id, child_id, delta, cap, &ctx)
                .map_err(|e| py_err(CausalError::from(e)))
        })?;
        Ok(PyPopulationDoContrast {
            contrast: result.contrast,
            stderr: result.stderr,
            parent,
            child,
            delta,
        })
    }

    /// Abduce once, then the unit-level contrast of every listed outcome between two worlds,
    /// each world a *set* of simultaneous hard interventions (`active` and `control` map
    /// variable names to levels). Every outcome is read from the same two replays.
    #[pyo3(signature = (outcomes, active, control, *, seed=0, threads=None))]
    fn counterfactual_contrasts(
        &self,
        py: Python<'_>,
        outcomes: Vec<String>,
        active: HashMap<String, f64>,
        control: HashMap<String, f64>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<PyContrastSet> {
        let model = self.inner.model.clone();
        let data = Arc::clone(&self.data);
        let ids = outcomes.iter().map(|o| variable_of(&data, o)).collect::<PyResult<Vec<_>>>()?;
        let active = hard_interventions(&data, &active)?;
        let control = hard_interventions(&data, &control)?;
        let (rows, noise) = detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let result = counterfactual_contrasts(model, &data, &ids, (&active, &control), &ctx)
                .map_err(py_err)?;
            let rows: Vec<(f64, Vec<f64>)> = result
                .contrasts
                .iter()
                .map(|c| (c.mean_ite, c.unit_effects.as_ref().to_vec()))
                .collect();
            Ok::<_, PyErr>((rows, format!("{:?}", result.noise_inference)))
        })?;
        let mut contrasts = Vec::with_capacity(rows.len());
        for (outcome, (mean_ite, effects)) in outcomes.into_iter().zip(rows) {
            contrasts.push(Py::new(
                py,
                PyOutcomeContrast {
                    outcome,
                    mean_ite,
                    unit_effects: PyArray1::from_vec(py, effects).unbind(),
                },
            )?);
        }
        Ok(PyContrastSet { contrasts, noise_inference: noise })
    }

    /// Mean of `outcome` over every unit when the `outer` and `inner` hard interventions hold
    /// simultaneously on the abduced factual noise (`Y_{outer, inner}`). The two sets must
    /// target disjoint variables.
    #[pyo3(signature = (outcome, outer, inner, *, seed=0, threads=None))]
    fn simultaneous_counterfactual(
        &self,
        py: Python<'_>,
        outcome: String,
        outer: HashMap<String, f64>,
        inner: HashMap<String, f64>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<f64> {
        let model = self.inner.model.clone();
        let data = Arc::clone(&self.data);
        let outcome_id = variable_of(&data, &outcome)?;
        let outer = hard_interventions(&data, &outer)?;
        let inner = hard_interventions(&data, &inner)?;
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let engine = CounterfactualEngine::new(model);
            let mut ws = MechanismWorkspace::default();
            simultaneous_hard_counterfactual(
                &engine, &data, &outer, &inner, outcome_id, &mut ws, &ctx,
            )
            .map_err(|e| py_err(CausalError::from(e)))
        })
    }

    /// Mean of `outcome` under hard `interventions`, from `n` ancestral draws.
    #[pyo3(signature = (outcome, interventions, n, *, seed=0, threads=None))]
    fn interventional_mean(
        &self,
        py: Python<'_>,
        outcome: String,
        interventions: HashMap<String, f64>,
        n: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<f64> {
        let model = self.inner.model.clone();
        let data = Arc::clone(&self.data);
        let outcome_id = variable_of(&data, &outcome)?;
        let ints = hard_interventions(&data, &interventions)?;
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let mut rng = ctx.rng.stream_for(StreamDomain::Counterfactual, 0x1A1E);
            let mut ws = MechanismWorkspace::default();
            interventional_mean(&model, &ints, outcome_id, n, &mut rng, &mut ws, &ctx)
                .map_err(|e| py_err(CausalError::from(e)))
        })
    }

    /// Run a named do-sampler for `outcome` under hard `interventions`.
    ///
    /// `method` is `"weighting"` (reweights the observed rows toward `do(T = t)`; needs exactly
    /// one intervention and ignores `n`), `"kde"` (ancestral draws smoothed by a Gaussian
    /// kernel; `bandwidth` overrides Silverman) or `"mcmc"` (random-walk Metropolis-Hastings
    /// against the KDE of pilot draws; `proposal_sd`, `burn_in` and `thin` override the
    /// defaults). For a root treatment `root_match_tolerance` is the half-width of the window
    /// of observed rows matched to the do-value (the default is exact matching up to 1e-9).
    /// Any other method raises.
    #[pyo3(signature = (method, outcome, interventions, n=1000, *, bandwidth=None, proposal_sd=None, burn_in=None, thin=None, root_match_tolerance=None, seed=0, threads=None))]
    #[allow(clippy::too_many_arguments, reason = "one keyword per sampler control")]
    fn do_sample(
        &self,
        py: Python<'_>,
        method: String,
        outcome: String,
        interventions: HashMap<String, f64>,
        n: usize,
        bandwidth: Option<f64>,
        proposal_sd: Option<f64>,
        burn_in: Option<usize>,
        thin: Option<usize>,
        root_match_tolerance: Option<f64>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<PyDoSample> {
        let model = self.inner.model.clone();
        let data = Arc::clone(&self.data);
        let outcome_id = variable_of(&data, &outcome)?;
        let ints = hard_interventions(&data, &interventions)?;
        if !matches!(method.as_str(), "weighting" | "kde" | "mcmc") {
            return Err(py_msg(format!(
                "unknown do-sampler {method:?}; choose \"weighting\", \"kde\" or \"mcmc\""
            )));
        }
        let weighting_level = if method == "weighting" {
            match interventions.iter().collect::<Vec<_>>().as_slice() {
                [(name, level)] => Some((variable_of(&data, name)?, **level)),
                _ => {
                    return Err(py_msg(
                        "the weighting do-sampler targets one hard intervention do(T = t); \
                         pass exactly one",
                    ));
                }
            }
        } else {
            None
        };
        let result = detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let mut rng = ctx.rng.stream_for(StreamDomain::Counterfactual, 0xD05A);
            let mut ws = MechanismWorkspace::default();
            match (method.as_str(), weighting_level) {
                ("weighting", Some((treatment, level))) => {
                    let mut sampler = WeightingDoSampler::new(treatment, outcome_id);
                    if let Some(tolerance) = root_match_tolerance {
                        sampler = sampler.with_root_match_tolerance(tolerance);
                    }
                    sampler
                        .estimate(&model, &data, level, &ctx)
                        .map_err(|e| py_err(CausalError::from(e)))
                }
                ("kde", _) => {
                    let mut sampler = KdeDoSampler::new(outcome_id);
                    sampler.bandwidth = bandwidth;
                    sampler
                        .sample(&model, &ints, n, &mut rng, &mut ws, &ctx)
                        .map_err(|e| py_err(CausalError::from(e)))
                }
                _ => {
                    let mut sampler = McmcDoSampler::new(outcome_id);
                    if let Some(v) = proposal_sd {
                        sampler.proposal_sd = v;
                    }
                    if let Some(v) = burn_in {
                        sampler.burn_in = v;
                    }
                    if let Some(v) = thin {
                        sampler.thin = v;
                    }
                    sampler
                        .sample(&model, &ints, n, &mut rng, &mut ws, &ctx)
                        .map_err(|e| py_err(CausalError::from(e)))
                }
            }
        })?;
        Ok(PyDoSample {
            mean: WeightingDoSampler::weighted_mean(&result),
            method: result.method.to_string(),
            notes: result.notes.iter().map(ToString::to_string).collect(),
            accept_rate: result.accept_rate,
            bandwidth: result.bandwidth,
            ess: result.ess,
            inner: result,
        })
    }

    /// Evaluate the fitted model on its own data: in-sample log-likelihood, residual
    /// independence, local-Markov checks and a shuffled-outcome likelihood baseline.
    #[pyo3(signature = (*, alpha=0.05, n_permutations=20, seed=0, threads=None))]
    fn evaluate(
        &self,
        py: Python<'_>,
        alpha: f64,
        n_permutations: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<PyModelEvaluation> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(crate::value_err(format!("alpha must lie in (0, 1), got {alpha}")));
        }
        let model = self.inner.model.clone();
        let data = Arc::clone(&self.data);
        let report = detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            ModelEvaluator { alpha, n_permutations, seed }
                .evaluate(&model, &data, &ctx)
                .map_err(|e| py_err(CausalError::from(e)))
        })?;
        Ok(PyModelEvaluation {
            in_sample_loglik: report.in_sample_loglik,
            zero_probability_rows: report.zero_probability_rows,
            mean_abs_residual: report.mean_abs_residual,
            residual_independence_p: report.residual_independence_p.to_vec(),
            local_markov_p: report.local_markov_p.to_vec(),
            permutation_loglik: report.permutation_loglik,
            falsified: report.falsified,
            alpha: report.alpha,
            notes: report.notes.iter().map(ToString::to_string).collect(),
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArrowStrength>()?;
    m.add_class::<PyPopulationDoContrast>()?;
    m.add_class::<PyOutcomeContrast>()?;
    m.add_class::<PyContrastSet>()?;
    m.add_class::<PyDoSample>()?;
    m.add_class::<PyModelEvaluation>()?;
    Ok(())
}
