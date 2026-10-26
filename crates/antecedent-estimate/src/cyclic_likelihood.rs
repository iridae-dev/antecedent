// SPDX-License-Identifier: MIT OR Apache-2.0
//! Joint likelihood estimation of specified nonlinear equilibrium families.
//! Known intervention targets are fixed design. The reported parameter model is
//! conditional on the specified structure and covariance, not nonparametric ID.
//! Calibration remains unmeasured; optimization is local and its status explicit.

use antecedent_core::ExecutionContext;
#[cfg(test)]
use antecedent_model::nonlinear_equilibrium::SolvePolicy;
use antecedent_model::nonlinear_equilibrium::{EquilibriumError, EquilibriumModel, RegimeRow};
use antecedent_prob::metropolis::{MetropolisOptions, sample_metropolis};
use serde::{Deserialize, Serialize};

/// Coordinates eligible for likelihood estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicParameter {
    /// An equation intercept.
    Intercept {
        /// Dense equation index.
        equation: usize,
    },
    /// A local parameter of a compositional mechanism.
    ExpressionParameter {
        /// Dense equation index.
        equation: usize,
        /// Local parameter index.
        parameter: usize,
    },
    /// An incoming mechanism coefficient, including self-feedback.
    Weight {
        /// Dense equation index.
        equation: usize,
        /// Dense parent index.
        parent: usize,
    },
}

/// Bounded deterministic coordinate-pattern optimization policy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicFitOptions {
    /// Initial coordinate displacement.
    pub initial_step: f64,
    /// Stop after every coordinate has been checked at this displacement.
    pub tolerance: f64,
    /// Maximum full coordinate sweeps.
    pub max_sweeps: usize,
    /// L1 penalty on weight coordinates only; zero is ordinary likelihood.
    pub l1_penalty: f64,
}
impl Default for CyclicFitOptions {
    fn default() -> Self {
        Self { initial_step: 0.2, tolerance: 1e-5, max_sweeps: 500, l1_penalty: 0.0 }
    }
}

/// A retained, replayable local optimum attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicFit {
    /// Fitted structural mechanisms; covariance is fixed to its declared value.
    pub model: EquilibriumModel,
    /// Unpenalized joint log likelihood.
    pub log_likelihood: f64,
    /// Penalized objective, maximized.
    pub objective: f64,
    /// Actual complete coordinate sweeps or analytic-gradient steps.
    pub sweeps: usize,
    /// The declared coordinate displacement or per-observation score tolerance was reached.
    pub converged: bool,
    /// Likelihood evaluations actually attempted.
    pub evaluations: usize,
}

/// Conditional-structure Bayesian parameter draws and diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicPosterior {
    /// Chain-major parameter vectors.
    pub samples: Vec<Vec<f64>>,
    /// Number of independent chains.
    pub chains: usize,
    /// Retained draws in each chain.
    pub draws_per_chain: usize,
    /// Actual accepted retained proposals per chain.
    pub accepted: Vec<usize>,
    /// Maximum rank/folded split R-hat, absent if undefined.
    pub max_rhat: Option<f64>,
    /// Minimum bulk ESS.
    pub min_bulk_ess: f64,
    /// Minimum tail ESS.
    pub min_tail_ess: f64,
    /// R-hat <= 1.01, bulk/tail ESS >= 100, all chains moved.
    pub diagnostics_pass: bool,
}

fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}

/// Read a parameter vector, rejecting duplicates and non-estimable coordinates.
pub fn parameter_values(
    model: &EquilibriumModel,
    parameters: &[CyclicParameter],
) -> Result<Vec<f64>, EquilibriumError> {
    model.validate()?;
    if parameters.is_empty() {
        return Err(invalid("parameter count"));
    }
    let mut out = Vec::with_capacity(parameters.len());
    for (k, &p) in parameters.iter().enumerate() {
        if parameters[..k].contains(&p) {
            return Err(invalid("duplicate parameter"));
        }
        out.push(match p {
            CyclicParameter::ExpressionParameter { equation, parameter } => *model
                .equations
                .get(equation)
                .and_then(|e| e.parameters.get(parameter))
                .ok_or_else(|| invalid("expression parameter index"))?,
            CyclicParameter::Intercept { equation } => {
                model.equations.get(equation).ok_or_else(|| invalid("equation index"))?.intercept
            }
            CyclicParameter::Weight { equation, parent } => {
                if model.equations.get(equation).is_some_and(|e| e.expression.is_some()) {
                    return Err(invalid("compositional mechanisms use expression parameters"));
                }
                *model
                    .equations
                    .get(equation)
                    .and_then(|e| e.weights.get(parent))
                    .ok_or_else(|| invalid("weight index"))?
            }
        });
    }
    Ok(out)
}

/// Bind a checked parameter vector onto a model template.
pub fn with_parameters(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    values: &[f64],
) -> Result<EquilibriumModel, EquilibriumError> {
    parameter_values(template, parameters)?;
    if values.len() != parameters.len() || values.iter().any(|x| !x.is_finite()) {
        return Err(invalid("parameter vector"));
    }
    let mut model = template.clone();
    for (&p, &v) in parameters.iter().zip(values) {
        match p {
            CyclicParameter::ExpressionParameter { equation, parameter } => {
                model.equations[equation].parameters[parameter] = v
            }
            CyclicParameter::Intercept { equation } => model.equations[equation].intercept = v,
            CyclicParameter::Weight { equation, parent } => {
                model.equations[equation].weights[parent] = v
            }
        }
    }
    Ok(model)
}

/// Fit all rows jointly. Known intervention regimes and covariance are retained.
/// A plateau or constrained local optimum is not a global optimum certificate.
pub fn fit_cyclic_likelihood(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    rows: &[RegimeRow],
    options: CyclicFitOptions,
    ctx: &ExecutionContext,
) -> Result<CyclicFit, EquilibriumError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::AdjustedFit,
        || fit_cyclic_likelihood_unobserved(template, parameters, rows, options, ctx),
    )
}

fn fit_cyclic_likelihood_unobserved(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    rows: &[RegimeRow],
    options: CyclicFitOptions,
    ctx: &ExecutionContext,
) -> Result<CyclicFit, EquilibriumError> {
    if rows.is_empty()
        || !options.initial_step.is_finite()
        || options.initial_step <= 0.0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || options.tolerance >= options.initial_step
        || options.max_sweeps == 0
        || options.max_sweeps > 100_000
        || !options.l1_penalty.is_finite()
        || options.l1_penalty < 0.0
    {
        return Err(invalid("fit data or policy"));
    }
    let values = parameter_values(template, parameters)?;
    let evaluate = |v: &[f64]| -> Result<f64, EquilibriumError> {
        let m = with_parameters(template, parameters, v)?;
        let penalty: f64 = parameters
            .iter()
            .zip(v)
            .filter_map(|(p, v)| matches!(p, CyclicParameter::Weight { .. }).then_some(v.abs()))
            .sum();
        Ok(m.log_likelihood(rows, ctx)? - options.l1_penalty * penalty)
    };
    let (values, objective, sweeps, evaluations, converged) =
        optimize_coordinates(values, options, ctx, evaluate)?;
    let model = with_parameters(template, parameters, &values)?;
    let log_likelihood = model.log_likelihood(rows, ctx)?;
    Ok(CyclicFit { model, log_likelihood, objective, sweeps, converged, evaluations })
}

pub(crate) fn optimize_coordinates<F>(
    mut values: Vec<f64>,
    options: CyclicFitOptions,
    ctx: &ExecutionContext,
    evaluate: F,
) -> Result<(Vec<f64>, f64, usize, usize, bool), EquilibriumError>
where
    F: Fn(&[f64]) -> Result<f64, EquilibriumError>,
{
    let mut objective = evaluate(&values)?;
    let mut evaluations = 1;
    let mut step = options.initial_step;
    let mut sweeps = 0;
    while step > options.tolerance && sweeps < options.max_sweeps {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        sweeps += 1;
        let mut changed = false;
        for coordinate in 0..values.len() {
            let old = values[coordinate];
            let mut best_value = old;
            for sign in [-1.0, 1.0] {
                values[coordinate] = old + sign * step;
                evaluations += 1;
                let next = match evaluate(&values) {
                    Ok(next) => next,
                    Err(EquilibriumError::Uncertified { .. } | EquilibriumError::Singular) => {
                        f64::NEG_INFINITY
                    }
                    Err(e) => return Err(e),
                };
                if next > objective {
                    objective = next;
                    best_value = values[coordinate];
                    changed = true;
                }
            }
            values[coordinate] = best_value;
        }
        if !changed {
            step *= 0.5;
        }
    }
    Ok((values, objective, sweeps, evaluations, step <= options.tolerance))
}

/// Sample a zero-mean independent normal parameter prior, restricted to the
/// template's certified-bijection likelihood domain. `prior_sd` is fixed before fitting.
/// Structural selection and covariance uncertainty are not included.
pub fn sample_cyclic_posterior(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    rows: &[RegimeRow],
    prior_sd: f64,
    options: MetropolisOptions,
    ctx: &ExecutionContext,
) -> Result<CyclicPosterior, EquilibriumError> {
    if !prior_sd.is_finite() || prior_sd <= 0.0 || rows.is_empty() {
        return Err(invalid("prior or rows"));
    }
    // Validate once so malformed rows cannot be silently treated as prior support boundaries.
    template.log_likelihood(rows, ctx)?;
    let initial = parameter_values(template, parameters)?;
    let log_density = |values: &[f64]| {
        let Ok(m) = with_parameters(template, parameters, values) else {
            return f64::NAN;
        };
        match m.log_likelihood(rows, ctx) {
            Ok(ll) => ll - 0.5 * values.iter().map(|v| (v / prior_sd).powi(2)).sum::<f64>(),
            Err(EquilibriumError::Uncertified { .. } | EquilibriumError::Singular) => {
                f64::NEG_INFINITY
            }
            Err(_) => f64::NAN,
        }
    };
    let result = sample_metropolis(&initial, options, ctx, log_density)
        .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    let p = parameters.len();
    let summary = antecedent_prob::mcmc_summary(&result.samples, options.chains, options.draws, p);
    let moved =
        antecedent_prob::all_chains_moved(&result.samples, options.chains, options.draws, p);
    let max_rhat = summary.max_rhat.is_finite().then_some(summary.max_rhat);
    let diagnostics_pass = moved
        && max_rhat.is_some_and(|r| r <= 1.01)
        && summary.min_bulk_ess >= 100.0
        && summary.min_tail_ess >= 100.0;
    Ok(CyclicPosterior {
        samples: result.samples.chunks_exact(p).map(<[f64]>::to_vec).collect(),
        chains: options.chains,
        draws_per_chain: options.draws,
        accepted: result.accepted,
        max_rhat,
        min_bulk_ess: summary.min_bulk_ess,
        min_tail_ess: summary.min_tail_ess,
        diagnostics_pass,
    })
}

/// Declared independent sampling unit for intervention-stratified resampling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CyclicBootstrapDesign {
    /// Replicate rows at each fixed target/value assignment are IID.
    #[default]
    FixedAssignments,
    /// Entire randomized assignment/outcome rows are IID within each target set.
    RandomizedWithinTargets,
}
pub(crate) fn bootstrap_groups(
    rows: &[RegimeRow],
    design: CyclicBootstrapDesign,
) -> std::collections::BTreeMap<Vec<(usize, u64)>, Vec<usize>> {
    let mut groups = std::collections::BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        let mut key: Vec<_> = row
            .clamps
            .iter()
            .map(|c| {
                (
                    c.variable,
                    if design == CyclicBootstrapDesign::FixedAssignments {
                        c.value.to_bits()
                    } else {
                        0
                    },
                )
            })
            .collect();
        key.sort_unstable();
        groups.entry(key).or_insert_with(Vec::new).push(i);
    }
    groups
}
/// Resample replicated fixed intervention assignments, preserving the existing API.
pub fn bootstrap_cyclic_fit(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    rows: &[RegimeRow],
    options: CyclicFitOptions,
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<Vec<f64>>>, EquilibriumError> {
    bootstrap_cyclic_fit_with_design(
        template,
        parameters,
        rows,
        options,
        replicates,
        CyclicBootstrapDesign::FixedAssignments,
        ctx,
    )
}
/// IID-within-known-regime bootstrap parameter attempts. Failures retain their
/// replicate slots; this raw sampling law has no nominal coverage claim.
pub fn bootstrap_cyclic_fit_with_design(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    rows: &[RegimeRow],
    options: CyclicFitOptions,
    replicates: usize,
    design: CyclicBootstrapDesign,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<Vec<f64>>>, EquilibriumError> {
    if replicates == 0 || replicates > 10_000 {
        return Err(invalid("bootstrap replicate count"));
    }
    template.log_likelihood(rows, ctx)?;
    let groups = bootstrap_groups(rows, design);
    let bytes = rows
        .len()
        .checked_mul(template.order() * 16 + 128)
        .and_then(|v| v.checked_add(replicates * parameters.len() * 16))
        .ok_or_else(|| invalid("bootstrap allocation overflow"))?;
    let _lease = ctx
        .acquire_admission(0, u64::try_from(bytes).map_err(|_| invalid("bootstrap allocation"))?)
        .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    let mut result = Vec::with_capacity(replicates);
    for r in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        let mut rng =
            ctx.rng.stream(antecedent_core::derive_stream(0x4359_434c_424f_4f54, r as u64));
        let mut sampled = Vec::with_capacity(rows.len());
        for indices in groups.values() {
            for _ in indices {
                // Rejection sampling avoids modulo bias for finite regime sizes.
                let size = indices.len() as u64;
                let ceiling = u64::MAX - u64::MAX % size;
                let mut draw = rng.next_u64();
                while draw >= ceiling {
                    draw = rng.next_u64();
                }
                let index = usize::try_from(draw % size).map_err(|_| invalid("row index"))?;
                sampled.push(rows[indices[index]].clone());
            }
        }
        match fit_cyclic_likelihood(template, parameters, &sampled, options, ctx) {
            Ok(fit) if fit.converged => {
                result.push(Some(parameter_values(&fit.model, parameters)?))
            }
            Ok(_)
            | Err(
                EquilibriumError::Uncertified { .. }
                | EquilibriumError::Singular
                | EquilibriumError::Nonconvergence { .. },
            ) => result.push(None),
            Err(e) => return Err(e),
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_model::nonlinear_equilibrium::{Activation, Clamp, Equation};
    fn independent_panel() -> (EquilibriumModel, Vec<RegimeRow>) {
        let model = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.35],
                    activation: Activation::Tanh,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![-0.25, 0.0],
                    activation: Activation::Tanh,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![0.04, 0.0, 0.0, 0.04],
        };
        let mut rng = antecedent_core::RngFactory::from_seed(98).stream(111);
        let mut rows = Vec::new();
        // Independent reference generator: direct scalar recurrence, no production solver.
        for k in 0..1200 {
            let ua = 0.2 * antecedent_kernels::standard_normal(&mut rng);
            let ub = 0.2 * antecedent_kernels::standard_normal(&mut rng);
            let clamps = if k % 3 == 0 {
                vec![Clamp { variable: 0, value: if k % 2 == 0 { 1.0 } else { -1.0 } }]
            } else {
                vec![]
            };
            let mut a = 0.0_f64;
            let mut b = 0.0_f64;
            for _ in 0..100 {
                let next_a =
                    if clamps.is_empty() { (0.35 * b).tanh() + ua } else { clamps[0].value };
                let next_b = (-0.25 * a).tanh() + ub;
                a = next_a;
                b = next_b;
            }
            rows.push(RegimeRow { values: vec![a, b], clamps });
        }
        (model, rows)
    }
    #[test]
    fn cyclic_likelihood_recovers_from_interventions_and_predicts_unseen_joint_regime() {
        let (truth, rows) = independent_panel();
        let parameters = [
            CyclicParameter::Weight { equation: 0, parent: 1 },
            CyclicParameter::Weight { equation: 1, parent: 0 },
        ];
        let template = with_parameters(&truth, &parameters, &[0.0, 0.0]).unwrap();
        let ctx = ExecutionContext::for_tests(10);
        let fit =
            fit_cyclic_likelihood(&template, &parameters, &rows, CyclicFitOptions::default(), &ctx)
                .unwrap();
        assert!(fit.converged);
        let values = parameter_values(&fit.model, &parameters).unwrap();
        assert!((values[0] - 0.35).abs() < 0.06, "{values:?}");
        assert!((values[1] + 0.25).abs() < 0.06, "{values:?}");
        let clamps = [Clamp { variable: 0, value: 2.0 }, Clamp { variable: 1, value: -2.0 }];
        let outcome =
            fit.model.solve(&[0.0; 2], &clamps, &[0.0; 2], SolvePolicy::default(), &ctx).unwrap();
        assert_eq!(outcome.values, vec![2.0, -2.0]);
        let unseen = fit
            .model
            .solve(&[0.0; 2], &clamps[..1], &[0.0; 2], SolvePolicy::default(), &ctx)
            .unwrap();
        assert!((unseen.values[1] - (-0.5_f64).tanh()).abs() < 0.1);
        let failed = fit_cyclic_likelihood(
            &template,
            &parameters,
            &rows,
            CyclicFitOptions { max_sweeps: 1, ..Default::default() },
            &ctx,
        )
        .unwrap();
        assert!(!failed.converged);
    }
    #[test]
    fn cyclic_likelihood_posterior_and_bootstrap_keep_status_and_replay() {
        let (truth, mut rows) = independent_panel();
        rows.truncate(32);
        let parameters = [CyclicParameter::Weight { equation: 1, parent: 0 }];
        let ctx = ExecutionContext::for_tests(33);
        let options = MetropolisOptions { chains: 2, warmup: 16, draws: 16, proposal_scale: 0.04 };
        let posterior =
            sample_cyclic_posterior(&truth, &parameters, &rows, 1.0, options, &ctx).unwrap();
        assert_eq!(posterior.samples.len(), 32);
        assert!(!posterior.diagnostics_pass);
        let repeat =
            sample_cyclic_posterior(&truth, &parameters, &rows, 1.0, options, &ctx).unwrap();
        assert_eq!(posterior, repeat);
        let bootstrap = bootstrap_cyclic_fit(
            &truth,
            &parameters,
            &rows,
            CyclicFitOptions { max_sweeps: 1, ..Default::default() },
            3,
            &ctx,
        )
        .unwrap();
        assert_eq!(bootstrap.len(), 3);
        assert!(bootstrap.iter().any(Option::is_none));
    }
}

/// Conditional latent-state/noise posterior for partial or noisy factual observation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicAbductionPosterior {
    /// Chain-major latent factual states, with exact observations fixed.
    pub states: Vec<Vec<f64>>,
    /// The corresponding additive structural disturbances, reused across worlds.
    pub disturbances: Vec<Vec<f64>>,
    /// Number of chains.
    pub chains: usize,
    /// Draws per chain.
    pub draws_per_chain: usize,
    /// Maximum rank/folded split R-hat of latent coordinates (absent if undefined).
    pub max_rhat: Option<f64>,
    /// Minimum bulk ESS.
    pub min_bulk_ess: f64,
    /// Minimum tail ESS.
    pub min_tail_ess: f64,
    /// Same explicit quality gate as the parameter posterior.
    pub diagnostics_pass: bool,
}

/// Sample factual latent states under an explicit observation model. Variance
/// zero denotes exact observation; positive variance is independent Gaussian
/// measurement error. `None` denotes an unobserved variable. Model/parameters
/// remain specified; this is not joint parameter/structure inference.
pub fn abduce_cyclic_partial(
    model: &EquilibriumModel,
    observations: &[Option<f64>],
    measurement_variances: &[f64],
    initial: &[f64],
    options: MetropolisOptions,
    ctx: &ExecutionContext,
) -> Result<CyclicAbductionPosterior, EquilibriumError> {
    model.validate()?;
    let n = model.order();
    if observations.len() != n
        || measurement_variances.len() != n
        || initial.len() != n
        || initial.iter().any(|v| !v.is_finite())
        || observations.iter().flatten().any(|v| !v.is_finite())
        || measurement_variances.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(invalid("partial abduction observation model"));
    }
    let latent: Vec<_> =
        (0..n).filter(|&i| observations[i].is_none() || measurement_variances[i] > 0.0).collect();
    if latent.is_empty() {
        return Err(invalid("all observations exact: use direct additive abduction"));
    }
    let mut base = initial.to_vec();
    for i in 0..n {
        if let Some(value) = observations[i] {
            if measurement_variances[i] == 0.0 {
                base[i] = value;
            }
        }
    }
    let initial_parameters: Vec<_> = latent.iter().map(|&i| base[i]).collect();
    // Refuse malformed or uncertified density before constructing the sampler.
    model.log_likelihood(&[RegimeRow { values: base.clone(), clamps: vec![] }], ctx)?;
    let target = |values: &[f64]| {
        let mut state = base.clone();
        for (&i, &v) in latent.iter().zip(values) {
            state[i] = v;
        }
        match model.log_likelihood(&[RegimeRow { values: state.clone(), clamps: vec![] }], ctx) {
            Ok(mut lp) => {
                for i in 0..n {
                    if let Some(observed) = observations[i] {
                        if measurement_variances[i] > 0.0 {
                            lp -= 0.5 * (state[i] - observed).powi(2) / measurement_variances[i];
                        }
                    }
                }
                lp
            }
            Err(_) => f64::NAN,
        }
    };
    let sampled = sample_metropolis(&initial_parameters, options, ctx, target)
        .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    let diagnostic = antecedent_prob::mcmc_summary(
        &sampled.samples,
        options.chains,
        options.draws,
        latent.len(),
    );
    let moved = antecedent_prob::all_chains_moved(
        &sampled.samples,
        options.chains,
        options.draws,
        latent.len(),
    );
    let mut states = Vec::new();
    let mut disturbances = Vec::new();
    for draw in sampled.samples.chunks_exact(latent.len()) {
        let mut state = base.clone();
        for (&i, &v) in latent.iter().zip(draw) {
            state[i] = v;
        }
        disturbances.push(model.abduce(&state)?);
        states.push(state);
    }
    let max_rhat = diagnostic.max_rhat.is_finite().then_some(diagnostic.max_rhat);
    Ok(CyclicAbductionPosterior {
        states,
        disturbances,
        chains: options.chains,
        draws_per_chain: options.draws,
        max_rhat,
        min_bulk_ess: diagnostic.min_bulk_ess,
        min_tail_ess: diagnostic.min_tail_ess,
        diagnostics_pass: moved
            && max_rhat.is_some_and(|v| v <= 1.01)
            && diagnostic.min_bulk_ess >= 100.0
            && diagnostic.min_tail_ess >= 100.0,
    })
}

/// Conditional sparse equilibrium structure-learning attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicStructureFit {
    /// Unpenalized refit on the selected topology.
    pub fit: CyclicFit,
    /// Selected original directed coefficient edges `(parent,equation)`.
    pub directed_edges: Vec<(usize, usize)>,
    /// Explicit coefficient threshold used after penalized fitting.
    pub threshold: f64,
    /// L1 penalty used in the selection stage.
    pub l1_penalty: f64,
    /// Optional evaluation on separately supplied whole held-out intervention regimes.
    pub heldout_log_likelihood: Option<f64>,
    /// Selection-stage convergence, separate from conditional refit convergence.
    pub selection_converged: bool,
}

/// Likelihood-evaluation work, in multiply-adds, one coordinate-search sweep may spend before the
/// structure fit uses the gradient optimizer instead.
///
/// A sweep over the `P = n^2` candidate coordinates evaluates the likelihood `2P` times, and one
/// evaluation solves the order-`n` equilibrium system for every row (about `rows * n^3`
/// multiply-adds). The gradient fit spends a small constant number of evaluations per iteration,
/// so it is the cheaper optimizer exactly when the sweep is expensive. The optimizer is thus
/// chosen from the parameter count and the row count rather than from the variable count alone.
const COORDINATE_SWEEP_WORK_LIMIT: f64 = 1.5e9;

/// Whether the structure fit over `order` variables and `rows` training rows uses the gradient
/// optimizer: the coordinate search's work per sweep exceeds [`COORDINATE_SWEEP_WORK_LIMIT`].
/// Both optimizers return the same [`CyclicFit`] shape.
fn prefers_gradient_fit(order: usize, rows: usize) -> bool {
    let n = order as f64;
    let parameters = n * n;
    2.0 * parameters * rows as f64 * n * n * n > COORDINATE_SWEEP_WORK_LIMIT
}

/// Fit all off-diagonal coefficients with an explicit L1 penalty, threshold,
/// then refit selected coefficients/intercepts without penalty. This learns a
/// parametric candidate topology, not a nonparametrically identified graph.
/// Held-out rows only evaluate the final fitted procedure; they never tune it.
pub fn learn_cyclic_structure(
    template: &EquilibriumModel,
    training: &[RegimeRow],
    heldout: &[RegimeRow],
    options: CyclicFitOptions,
    threshold: f64,
    ctx: &ExecutionContext,
) -> Result<CyclicStructureFit, EquilibriumError> {
    template.validate()?;
    let n = template.order();
    if !threshold.is_finite() || threshold <= 0.0 || options.l1_penalty <= 0.0 {
        return Err(invalid("structure search needs a positive threshold and L1 penalty"));
    }
    // The complete candidate topology has one weight per ordered pair plus an intercept per
    // equation; its state is leased against the context instead of capping the order.
    let _lease = crate::cyclic_dynamics::admit_parameters(ctx, n.saturating_mul(n))?;
    let gradient = prefers_gradient_fit(n, training.len());
    let mut parameters = Vec::new();
    let mut starting = template.clone();
    for i in 0..n {
        parameters.push(CyclicParameter::Intercept { equation: i });
        // The complete candidate topology is explicit; all starts are zero,
        // avoiding an unidentified arbitrary starting feedback graph.
        for j in 0..n {
            if i != j {
                parameters.push(CyclicParameter::Weight { equation: i, parent: j });
                starting.equations[i].weights[j] = if gradient
                    && starting.equations[i].activation
                        == antecedent_model::nonlinear_equilibrium::Activation::Relu
                {
                    1e-3 * ((i * n + j + 1) as f64).sin()
                } else {
                    0.0
                };
            }
        }
    }
    let fit_large = |model: &EquilibriumModel,
                     parameters: &[CyclicParameter],
                     penalty: f64|
     -> Result<CyclicFit, EquilibriumError> {
        let fitted = crate::cyclic_gradient::fit_cyclic_gradient(
            model,
            parameters,
            training,
            crate::cyclic_gradient::CyclicGradientPolicy {
                learning_rate: options.initial_step,
                score_tolerance: options.tolerance * training.len() as f64,
                max_iterations: options.max_sweeps,
                l1_penalty: penalty,
            },
            ctx,
        )?;
        Ok(CyclicFit {
            model: fitted.model,
            log_likelihood: fitted.log_likelihood,
            objective: fitted.objective,
            sweeps: fitted.iterations,
            converged: fitted.converged,
            evaluations: fitted.evaluations,
        })
    };
    let selected = if gradient {
        fit_large(&starting, &parameters, options.l1_penalty)?
    } else {
        fit_cyclic_likelihood(&starting, &parameters, training, options, ctx)?
    };
    let mut model = selected.model.clone();
    let mut active = Vec::new();
    let mut edges = Vec::new();
    for i in 0..n {
        active.push(CyclicParameter::Intercept { equation: i });
        for j in 0..n {
            if i != j {
                if model.equations[i].weights[j].abs() >= threshold {
                    edges.push((j, i));
                    active.push(CyclicParameter::Weight { equation: i, parent: j });
                } else {
                    model.equations[i].weights[j] = 0.0;
                }
            }
        }
    }
    let fit = if gradient {
        fit_large(&model, &active, 0.0)?
    } else {
        fit_cyclic_likelihood(
            &model,
            &active,
            training,
            CyclicFitOptions { l1_penalty: 0.0, ..options },
            ctx,
        )?
    };
    let heldout_log_likelihood =
        if heldout.is_empty() { None } else { Some(fit.model.log_likelihood(heldout, ctx)?) };
    Ok(CyclicStructureFit {
        fit,
        directed_edges: edges,
        threshold,
        l1_penalty: options.l1_penalty,
        heldout_log_likelihood,
        selection_converged: selected.converged,
    })
}

#[cfg(test)]
mod optimizer_choice_tests {
    use super::prefers_gradient_fit;

    #[test]
    fn the_optimizer_follows_the_sweep_work_not_the_variable_count_alone() {
        // A small system stays on the coordinate search at any realistic row count.
        assert!(!prefers_gradient_fit(3, 100_000));
        // The same order is gradient-fitted once the rows make a sweep expensive.
        assert!(!prefers_gradient_fit(8, 1_000));
        assert!(prefers_gradient_fit(8, 1_000_000));
        // A wide system with few rows is cheap enough for the coordinate search.
        assert!(!prefers_gradient_fit(12, 500));
        assert!(prefers_gradient_fit(12, 50_000));
        assert!(prefers_gradient_fit(40, 1_000));
    }
}

#[cfg(test)]
mod partial_tests {
    use super::*;
    use antecedent_model::nonlinear_equilibrium::{Activation, Equation};
    #[test]
    fn partial_abduction_matches_analytic_conditional_gaussian() {
        let model = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.4],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![0.5, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        };
        let posterior = abduce_cyclic_partial(
            &model,
            &[Some(1.0), None],
            &[0.0, 0.0],
            &[1.0, 0.0],
            MetropolisOptions { chains: 4, warmup: 400, draws: 1200, proposal_scale: 0.8 },
            &ExecutionContext::for_tests(193),
        )
        .unwrap();
        let mean =
            posterior.states.iter().map(|x| x[1]).sum::<f64>() / posterior.states.len() as f64;
        let variance = posterior.states.iter().map(|x| (x[1] - mean).powi(2)).sum::<f64>()
            / posterior.states.len() as f64;
        assert!((mean - 0.9 / 1.16).abs() < 0.1, "{mean}");
        let expected_variance = 1.25 / 0.64 - (0.9 / 0.64_f64).powi(2) / (1.16 / 0.64);
        assert!((variance - expected_variance).abs() < 0.15, "{variance} vs {expected_variance}");
        for (state, noise) in posterior.states.iter().zip(&posterior.disturbances) {
            assert_eq!(state[0].to_bits(), 1.0_f64.to_bits());
            assert!((noise[1] - (state[1] - 0.5)).abs() < 1e-12);
        }
    }
}

/// Whole-selection bootstrap under fixed replicated assignments.
#[allow(clippy::too_many_arguments, reason = "preserves the explicit existing fitting API")]
pub fn bootstrap_cyclic_structure(
    template: &EquilibriumModel,
    training: &[RegimeRow],
    heldout: &[RegimeRow],
    options: CyclicFitOptions,
    threshold: f64,
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<CyclicStructureFit>>, EquilibriumError> {
    bootstrap_cyclic_structure_with_design(
        template,
        training,
        heldout,
        options,
        threshold,
        replicates,
        CyclicBootstrapDesign::FixedAssignments,
        ctx,
    )
}
/// Repeat the entire sparse selection/refit procedure under IID-within-regime
/// resampling. Model/edge-selection variation is retained, not replaced by
/// conditional-on-selected-model standard errors. No coverage claim is made.
#[allow(
    clippy::too_many_arguments,
    reason = "declared sampling design extends the existing whole-selection API"
)]
pub fn bootstrap_cyclic_structure_with_design(
    template: &EquilibriumModel,
    training: &[RegimeRow],
    heldout: &[RegimeRow],
    options: CyclicFitOptions,
    threshold: f64,
    replicates: usize,
    design: CyclicBootstrapDesign,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<CyclicStructureFit>>, EquilibriumError> {
    let groups = bootstrap_groups(training, design);
    let bytes = training
        .len()
        .checked_mul(template.order() * 16 + 128)
        .and_then(|v| v.checked_add(replicates * (template.order() * template.order() * 32 + 1024)))
        .ok_or_else(|| invalid("whole-selection bootstrap size overflow"))?;
    let _lease = ctx
        .acquire_admission(0, bytes as u64)
        .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    let mut output = Vec::with_capacity(replicates);
    for r in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        let mut rng =
            ctx.rng.stream(antecedent_core::derive_stream(0x4359_434c_5345_4c42, r as u64));
        let mut rows = Vec::with_capacity(training.len());
        for indices in groups.values() {
            for _ in indices {
                let size = indices.len() as u64;
                let ceiling = u64::MAX - u64::MAX % size;
                let mut draw = rng.next_u64();
                while draw >= ceiling {
                    draw = rng.next_u64();
                }
                rows.push(
                    training[indices
                        [usize::try_from(draw % size).map_err(|_| invalid("bootstrap index"))?]]
                    .clone(),
                );
            }
        }
        match learn_cyclic_structure(template, &rows, heldout, options, threshold, ctx) {
            Ok(fit) if fit.selection_converged && fit.fit.converged => output.push(Some(fit)),
            Ok(_)
            | Err(
                EquilibriumError::Singular
                | EquilibriumError::Uncertified { .. }
                | EquilibriumError::Nonconvergence { .. },
            ) => output.push(None),
            Err(e) => return Err(e),
        }
    }
    Ok(output)
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use antecedent_model::nonlinear_equilibrium::{Activation, Clamp, Equation};
    #[test]
    fn selected_nonlinear_model_predicts_a_whole_unseen_joint_intervention_regime() {
        let template = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.35, 0.0],
                    activation: Activation::Tanh,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![-0.25, 0.0, 0.0],
                    activation: Activation::Tanh,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "y".into(),
                    intercept: 0.0,
                    weights: vec![0.3, -0.2, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04],
        };
        let ctx = ExecutionContext::for_tests(337);
        let mut rng = ctx.rng.stream(991);
        let mut training = Vec::new();
        let mut heldout = Vec::new();
        for i in 0..660 {
            let ua = 0.2 * antecedent_kernels::standard_normal(&mut rng);
            let ub = 0.2 * antecedent_kernels::standard_normal(&mut rng);
            let uy = 0.2 * antecedent_kernels::standard_normal(&mut rng);
            let clamps = if i >= 600 {
                vec![Clamp { variable: 0, value: 0.7 }, Clamp { variable: 1, value: -0.6 }]
            } else if i % 3 == 0 {
                vec![Clamp { variable: 0, value: if i % 2 == 0 { 1.0 } else { -1.0 } }]
            } else if i % 3 == 1 {
                vec![Clamp { variable: 1, value: if i % 2 == 0 { 1.0 } else { -1.0 } }]
            } else {
                vec![]
            };
            let mut a = 0.0_f64;
            let mut b = 0.0_f64;
            // Independent scalar reference generator, no production solver.
            for _ in 0..100 {
                let aa = clamps
                    .iter()
                    .find(|c| c.variable == 0)
                    .map_or((0.35 * b).tanh() + ua, |c| c.value);
                let bb = clamps
                    .iter()
                    .find(|c| c.variable == 1)
                    .map_or((-0.25 * a).tanh() + ub, |c| c.value);
                a = aa;
                b = bb;
            }
            let row = RegimeRow { values: vec![a, b, 0.3 * a - 0.2 * b + uy], clamps };
            if i < 600 {
                training.push(row);
            } else {
                heldout.push(row);
            }
        }
        let options = CyclicFitOptions {
            l1_penalty: 1.0,
            tolerance: 1e-4,
            max_sweeps: 500,
            ..Default::default()
        };
        let selected =
            learn_cyclic_structure(&template, &training, &heldout, options, 0.08, &ctx).unwrap();
        assert!(selected.selection_converged && selected.fit.converged);
        assert!(selected.heldout_log_likelihood.unwrap().is_finite());
        let predicted = selected
            .fit
            .model
            .solve(&[0.0; 3], &heldout[0].clamps, &[0.0; 3], SolvePolicy::default(), &ctx)
            .unwrap();
        assert!((predicted.values[2] - 0.33).abs() < 0.07, "{predicted:?}");
        let failed = bootstrap_cyclic_structure(
            &template,
            &training[..20],
            &[],
            CyclicFitOptions { max_sweeps: 1, ..options },
            0.08,
            2,
            &ctx,
        )
        .unwrap();
        assert_eq!(failed.len(), 2);
        assert!(failed.iter().any(Option::is_none));
    }
}

/// Joint parameter/factual-state posterior under the declared structural and
/// measurement laws. Diagnostics in `factual` cover all sampled coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicJointAbduction {
    /// Chain-major parameter vectors, aligned with factual/noise draws.
    pub parameter_draws: Vec<Vec<f64>>,
    /// Factual states and shared disturbances; diagnostics cover parameters and latent states jointly.
    pub factual: CyclicAbductionPosterior,
}

/// Joint abduction updates parameters using both the independent training rows
/// and factual measurement evidence. It does not freeze the training posterior
/// before conditioning the factual state. Exact observations are fixed; missing
/// or noisy states are sampled alongside parameters. Structure/covariance remain
/// specified, with a normal parameter prior restricted to contractive support.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one explicit joint density combines training, prior and factual measurement evidence"
)]
pub fn abduce_cyclic_joint(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    training: &[RegimeRow],
    prior_sd: f64,
    observations: &[Option<f64>],
    measurement_variances: &[f64],
    initial: &[f64],
    options: MetropolisOptions,
    ctx: &ExecutionContext,
) -> Result<CyclicJointAbduction, EquilibriumError> {
    template.validate()?;
    let n = template.order();
    if !prior_sd.is_finite()
        || prior_sd <= 0.0
        || training.is_empty()
        || observations.len() != n
        || measurement_variances.len() != n
        || initial.len() != n
        || initial.iter().any(|v| !v.is_finite())
        || observations.iter().flatten().any(|v| !v.is_finite())
        || measurement_variances.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(invalid("joint abduction prior/training/observation model"));
    }
    template.log_likelihood(training, ctx)?;
    let mut start = parameter_values(template, parameters)?;
    let p = start.len();
    let latent: Vec<_> =
        (0..n).filter(|&i| observations[i].is_none() || measurement_variances[i] > 0.0).collect();
    let mut base = initial.to_vec();
    for i in 0..n {
        if let Some(value) = observations[i] {
            if measurement_variances[i] == 0.0 {
                base[i] = value;
            }
        }
    }
    template.log_likelihood(&[RegimeRow { values: base.clone(), clamps: vec![] }], ctx)?;
    start.extend(latent.iter().map(|&i| base[i]));
    let dimension = start.len();
    let density = |joint: &[f64]| {
        let Ok(model) = with_parameters(template, parameters, &joint[..p]) else {
            return f64::NAN;
        };
        let mut state = base.clone();
        for (&i, &value) in latent.iter().zip(&joint[p..]) {
            state[i] = value;
        }
        let likelihood = model.log_likelihood(training, ctx).and_then(|training_lp| {
            model
                .log_likelihood(&[RegimeRow { values: state.clone(), clamps: vec![] }], ctx)
                .map(|factual_lp| training_lp + factual_lp)
        });
        match likelihood {
            Ok(mut lp) => {
                lp -= 0.5 * joint[..p].iter().map(|v| (v / prior_sd).powi(2)).sum::<f64>();
                for i in 0..n {
                    if let Some(value) = observations[i] {
                        if measurement_variances[i] > 0.0 {
                            lp -= 0.5 * (state[i] - value).powi(2) / measurement_variances[i];
                        }
                    }
                }
                lp
            }
            Err(EquilibriumError::Uncertified { .. } | EquilibriumError::Singular) => {
                f64::NEG_INFINITY
            }
            Err(_) => f64::NAN,
        }
    };
    let sampled = sample_metropolis(&start, options, ctx, density)
        .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    let summary =
        antecedent_prob::mcmc_summary(&sampled.samples, options.chains, options.draws, dimension);
    let moved = antecedent_prob::all_chains_moved(
        &sampled.samples,
        options.chains,
        options.draws,
        dimension,
    );
    let bytes = options
        .chains
        .checked_mul(options.draws)
        .and_then(|draws| draws.checked_mul((p + 2 * n) * 8 + 128))
        .ok_or_else(|| invalid("joint abduction retention overflow"))?;
    let _retention = ctx
        .acquire_admission(0, bytes as u64)
        .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    let mut parameter_draws = Vec::new();
    let mut states = Vec::new();
    let mut disturbances = Vec::new();
    for draw in sampled.samples.chunks_exact(dimension) {
        let values = draw[..p].to_vec();
        let model = with_parameters(template, parameters, &values)?;
        let mut state = base.clone();
        for (&i, &value) in latent.iter().zip(&draw[p..]) {
            state[i] = value;
        }
        disturbances.push(model.abduce(&state)?);
        states.push(state);
        parameter_draws.push(values);
    }
    let max_rhat = summary.max_rhat.is_finite().then_some(summary.max_rhat);
    Ok(CyclicJointAbduction {
        parameter_draws,
        factual: CyclicAbductionPosterior {
            states,
            disturbances,
            chains: options.chains,
            draws_per_chain: options.draws,
            max_rhat,
            min_bulk_ess: summary.min_bulk_ess,
            min_tail_ess: summary.min_tail_ess,
            diagnostics_pass: moved
                && max_rhat.is_some_and(|r| r <= 1.01)
                && summary.min_bulk_ess >= 100.0
                && summary.min_tail_ess >= 100.0,
        },
    })
}

#[cfg(test)]
mod joint_abduction_tests {
    use super::*;
    use antecedent_model::nonlinear_equilibrium::{Activation, Equation};
    #[test]
    fn joint_abduction_matches_conjugate_noisy_evidence_and_parameter_noise_coupling() {
        let model = EquilibriumModel {
            equations: vec![Equation {
                id: "x".into(),
                intercept: 0.0,
                weights: vec![0.0],
                activation: Activation::Identity,
                expression: None,
                parameters: vec![],
            }],
            noise_covariance: vec![1.0],
        };
        let rows = [0.0, 1.0]
            .into_iter()
            .map(|x| RegimeRow { values: vec![x], clamps: vec![] })
            .collect::<Vec<_>>();
        let posterior = abduce_cyclic_joint(
            &model,
            &[CyclicParameter::Intercept { equation: 0 }],
            &rows,
            1.0,
            &[Some(2.0)],
            &[0.5],
            &[1.5],
            MetropolisOptions { chains: 4, warmup: 800, draws: 3000, proposal_scale: 0.6 },
            &ExecutionContext::for_tests(933),
        )
        .unwrap();
        let count = posterior.parameter_draws.len() as f64;
        let mean = posterior.parameter_draws.iter().map(|v| v[0]).sum::<f64>() / count;
        let variance =
            posterior.parameter_draws.iter().map(|v| (v[0] - mean).powi(2)).sum::<f64>() / count;
        let factual = posterior.factual.states.iter().map(|v| v[0]).sum::<f64>() / count;
        assert!((mean - 7.0 / 11.0).abs() < 0.06, "{mean}");
        assert!((variance - 3.0 / 11.0).abs() < 0.05, "{variance}");
        assert!((factual - 17.0 / 11.0).abs() < 0.06, "{factual}");
        for ((theta, state), noise) in posterior
            .parameter_draws
            .iter()
            .zip(&posterior.factual.states)
            .zip(&posterior.factual.disturbances)
        {
            assert!((theta[0] + noise[0] - state[0]).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod design_tests {
    use super::*;
    #[test]
    fn randomized_assignments_retain_joint_row_sampling_units() {
        use antecedent_model::nonlinear_equilibrium::Clamp;
        let rows = vec![
            RegimeRow { values: vec![1.0, 2.0], clamps: vec![Clamp { variable: 0, value: 1.0 }] },
            RegimeRow { values: vec![3.0, 4.0], clamps: vec![Clamp { variable: 0, value: 3.0 }] },
        ];
        assert_eq!(bootstrap_groups(&rows, CyclicBootstrapDesign::FixedAssignments).len(), 2);
        assert_eq!(
            bootstrap_groups(&rows, CyclicBootstrapDesign::RandomizedWithinTargets)
                .values()
                .next()
                .unwrap(),
            &vec![0, 1]
        );
    }
}
