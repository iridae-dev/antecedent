// SPDX-License-Identifier: MIT OR Apache-2.0
//! Nonlinear additive-noise equilibrium SCMs, compiled by SCCs.
//!
//! `x_i = intercept_i + activation_i(sum_j weight_ij x_j) + u_i`.
//! Built-ins have global derivative bounds, so a block infinity-norm bound below
//! one proves contraction on R^k for every fixed parent/noise input. A numerical
//! Newton result has no such interpretation. Likelihood uses the free-system
//! Jacobian; hard clamps retain the marginal, not conditional, disturbance law.

use antecedent_core::ExecutionContext;
use antecedent_graph::{CyclicDigraph, DenseNodeId};
use antecedent_kernels::gauss_jordan_solve;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum order admitted by this dense implementation.
pub const MAX_ORDER: usize = 128;

/// Serializable scalar mechanisms with global Lipschitz constant one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// Identity; the linear family is a specialization.
    Identity,
    /// Smooth bounded saturation.
    Tanh,
    /// Smooth non-monotone response.
    Sin,
    /// Piecewise-linear response; its almost-everywhere derivative uses zero at the kink.
    Relu,
}
impl Activation {
    pub(crate) fn value(self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Tanh => x.tanh(),
            Self::Sin => x.sin(),
            Self::Relu => x.max(0.0),
        }
    }
    pub(crate) fn derivative(self, x: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Tanh => 1.0 - x.tanh().powi(2),
            Self::Sin => x.cos(),
            Self::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Serializable mechanism. Row `i` is the mechanism for variable `i`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Equation {
    /// Stable mechanism identity, unique within a model.
    pub id: String,
    /// Additive intercept outside the activation.
    pub intercept: f64,
    /// Incoming coefficients in dense variable order, including self-feedback.
    pub weights: Vec<f64>,
    /// Scalar nonlinearity.
    pub activation: Activation,
    /// Optional compositional mechanism; legacy weights must be zero when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<crate::mechanism_expr::MechanismExpression>,
    /// Local finite parameters for the compositional mechanism.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<f64>,
}
impl Equation {
    /// Validate a mechanism independently of equilibrium or dynamic semantics.
    pub fn validate(&self, order: usize) -> Result<(), EquilibriumError> {
        if self.id.trim().is_empty()
            || self.id.len() > 256
            || !self.intercept.is_finite()
            || self.weights.len() != order
            || !finite(&self.weights)
            || !finite(&self.parameters)
        {
            return Err(invalid("mechanism identity, shape or coefficients"));
        }
        if let Some(expression) = &self.expression {
            if self.activation != Activation::Identity || self.weights.iter().any(|&x| x != 0.0) {
                return Err(invalid(
                    "compositional mechanism requires zero legacy weights and identity activation",
                ));
            }
            expression.compile(order, self.parameters.len())?;
        } else if !self.parameters.is_empty() {
            return Err(invalid("parameters require a compositional expression"));
        }
        Ok(())
    }
    /// Endogenous coordinates actually read by this mechanism.
    pub fn reads(&self) -> Result<Vec<usize>, EquilibriumError> {
        if let Some(expression) = &self.expression {
            Ok(expression.compile(self.weights.len(), self.parameters.len())?.reads().to_vec())
        } else {
            Ok(self
                .weights
                .iter()
                .enumerate()
                .filter_map(|(i, &w)| (w != 0.0).then_some(i))
                .collect())
        }
    }
    /// Scalar structural value before adding the joint disturbance.
    pub fn structural_value(&self, state: &[f64]) -> Result<f64, EquilibriumError> {
        if state.len() != self.weights.len() {
            return Err(invalid("mechanism state shape"));
        }
        let value = if let Some(expression) = &self.expression {
            expression
                .compile(state.len(), self.parameters.len())?
                .evaluate(state, &self.parameters)?
                .value()
        } else {
            self.activation.value(dot(&self.weights, state))
        };
        Ok(self.intercept + value)
    }
    /// Endogenous derivatives in original variable order.
    pub fn input_derivatives(&self, state: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        if state.len() != self.weights.len() {
            return Err(invalid("mechanism state shape"));
        }
        if let Some(expression) = &self.expression {
            Ok(expression
                .compile(state.len(), self.parameters.len())?
                .evaluate(state, &self.parameters)?
                .gradient()
                .to_vec())
        } else {
            let d = self.activation.derivative(dot(&self.weights, state));
            Ok(self.weights.iter().map(|w| d * w).collect())
        }
    }
    /// Absolute derivative bounds valid over all endogenous states.
    pub fn derivative_bounds(&self) -> Result<Vec<f64>, EquilibriumError> {
        if let Some(expression) = &self.expression {
            expression
                .compile(self.weights.len(), self.parameters.len())?
                .derivative_bounds(&self.parameters)
        } else {
            Ok(self.weights.iter().map(|w| w.abs()).collect())
        }
    }
}

/// Specified structural model; construction/deserialization does not establish identification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquilibriumModel {
    /// One equation per variable.
    pub equations: Vec<Equation>,
    /// Joint zero-mean Gaussian disturbance covariance, row-major.
    pub noise_covariance: Vec<f64>,
}

/// Hard-clamp intervention, not conditioning on observations.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clamp {
    /// Dense variable index.
    pub variable: usize,
    /// Fixed structural value.
    pub value: f64,
}

/// Explicit numerical policy retained alongside an execution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolvePolicy {
    /// Absolute equation-residual tolerance.
    pub tolerance: f64,
    /// Maximum iterations per block.
    pub max_iterations: usize,
    /// Require global block contraction; otherwise use damped Newton.
    pub require_contraction: bool,
}
impl Default for SolvePolicy {
    fn default() -> Self {
        Self { tolerance: 1e-10, max_iterations: 1000, require_contraction: true }
    }
}

/// What the execution actually established.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquilibriumResult {
    /// One equilibrium in dense order.
    pub values: Vec<f64>,
    /// Maximum free-equation residual.
    pub residual: f64,
    /// Total iterations, across blocks.
    pub iterations: usize,
    /// True only when every free block has a contraction or verified affine-bijection certificate.
    pub globally_unique: bool,
    /// Existence is proved by a structural certificate, independently of residual convergence.
    pub existence_certified: bool,
    /// True after a finite residual check; this alone proves neither existence nor uniqueness.
    pub numerical_converged: bool,
    /// No causal dynamics are declared by a static equilibrium SCM; stability is therefore absent.
    pub dynamic_stability: Option<bool>,
    /// Floating-point infinity-norm condition estimate of the full free residual Jacobian.
    /// Absent for an empty, singular or nonfinite system; not a rigorous conditioning bound.
    pub jacobian_condition_estimate: Option<f64>,
    /// SCCs of the intervened execution in source-before-sink order.
    pub blocks: Vec<Vec<usize>>,
}

/// A sampled equilibrium row and its known intervention regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeRow {
    /// Observed values in dense order.
    pub values: Vec<f64>,
    /// Replaced mechanisms.
    pub clamps: Vec<Clamp>,
}

/// Explicit failure; no failed solve becomes a causal answer.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EquilibriumError {
    /// Malformed shape, non-finite input, duplicate identity, or invalid policy.
    #[error("invalid equilibrium input: {0}")]
    Invalid(String),
    /// A requested certificate was not earned.
    #[error("free block is not certified contractive (bound {bound})")]
    Uncertified {
        /// Computed global Lipschitz upper bound.
        bound: f64,
    },
    /// An iteration stopped without a verified root.
    #[error("equilibrium solver did not converge; residual {residual}")]
    Nonconvergence {
        /// Last free-equation residual.
        residual: f64,
    },
    /// A singular free-system Jacobian or covariance.
    #[error("singular equilibrium matrix")]
    Singular,
    /// Shared resource admission or cancellation failed.
    #[error("equilibrium resource refusal: {0}")]
    Resource(String),
}

fn invalid(message: &str) -> EquilibriumError {
    EquilibriumError::Invalid(message.into())
}
fn finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}
pub(crate) fn check_cancel(ctx: &ExecutionContext) -> Result<(), EquilibriumError> {
    if ctx.cancellation.is_cancelled() {
        Err(EquilibriumError::Resource("cancelled".into()))
    } else {
        Ok(())
    }
}

impl EquilibriumModel {
    /// Validate structural shape, unique mechanism ids and positive-definite joint noise.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        let n = self.equations.len();
        if n == 0 || n > MAX_ORDER || self.noise_covariance.len() != n * n {
            return Err(invalid("model order/covariance shape"));
        }
        let mut ids = std::collections::BTreeSet::new();
        for equation in &self.equations {
            equation.validate(n)?;
            if !ids.insert(&equation.id) {
                return Err(invalid("duplicate mechanism identity"));
            }
        }
        cholesky(&self.noise_covariance, n)?;
        Ok(())
    }

    /// Number of endogenous variables.
    #[must_use]
    pub fn order(&self) -> usize {
        self.equations.len()
    }

    /// Structural mean at one state; joint noise is added by the execution.
    pub fn mechanism_values(&self, state: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        self.validate()?;
        self.check_vector(state)?;
        self.equations.iter().map(|e| e.structural_value(state)).collect()
    }

    fn check_vector(&self, values: &[f64]) -> Result<(), EquilibriumError> {
        if values.len() != self.order() || !finite(values) {
            return Err(invalid("state/noise dimension or nonfinite"));
        }
        Ok(())
    }

    pub(crate) fn free_indices(&self, clamps: &[Clamp]) -> Result<Vec<usize>, EquilibriumError> {
        let n = self.order();
        let mut seen = vec![false; n];
        for c in clamps {
            if c.variable >= n || !c.value.is_finite() || seen[c.variable] {
                return Err(invalid("clamp target, value, or duplicate"));
            }
            seen[c.variable] = true;
        }
        Ok((0..n).filter(|&i| !seen[i]).collect())
    }

    /// Compile the free-variable graph into SCC solve blocks.
    pub fn blocks(&self, clamps: &[Clamp]) -> Result<Vec<Vec<usize>>, EquilibriumError> {
        self.validate()?;
        let free = self.free_indices(clamps)?;
        let count = u32::try_from(free.len()).map_err(|_| invalid("graph order"))?;
        let mut graph = CyclicDigraph::with_variables(count);
        for (to, &i) in free.iter().enumerate() {
            let reads = self.equations[i].reads()?;
            for (from, &j) in free.iter().enumerate() {
                if i != j && reads.contains(&j) {
                    graph
                        .insert_directed(
                            DenseNodeId::from_raw(
                                u32::try_from(from).map_err(|_| invalid("node id"))?,
                            ),
                            DenseNodeId::from_raw(
                                u32::try_from(to).map_err(|_| invalid("node id"))?,
                            ),
                        )
                        .map_err(|e| invalid(&e.to_string()))?;
                }
            }
        }
        Ok(graph
            .condensation()
            .components()
            .iter()
            .map(|c| c.iter().map(|id| free[id.as_usize()]).collect())
            .collect())
    }

    /// Solve a specified noise realization. Initialization is numerical, not a selection law.
    /// Uncertified Newton roots are reported only as numerical solutions.
    #[allow(
        clippy::too_many_lines,
        reason = "one bounded SCC solve preserves contraction and numerical failure distinctions"
    )]
    pub fn solve(
        &self,
        noise: &[f64],
        clamps: &[Clamp],
        initial: &[f64],
        policy: SolvePolicy,
        ctx: &ExecutionContext,
    ) -> Result<EquilibriumResult, EquilibriumError> {
        self.validate()?;
        self.check_vector(noise)?;
        self.check_vector(initial)?;
        if !policy.tolerance.is_finite()
            || policy.tolerance <= 0.0
            || policy.max_iterations == 0
            || policy.max_iterations > 1_000_000
        {
            return Err(invalid("solve policy"));
        }
        let n = self.order();
        // Dense Newton matrix plus vectors, graph, covariance validation, and returned state.
        let tape_nodes = self
            .equations
            .iter()
            .filter_map(|e| {
                e.expression
                    .as_ref()
                    .map(|expr| expr.compile(n, e.parameters.len()).map(|t| t.nodes()))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let bytes = u64::try_from(16 * n * n + 128 * n + tape_nodes * (2 * n + 8) * 8)
            .map_err(|_| invalid("resource size"))?;
        let _lease = ctx
            .acquire_admission(1, bytes)
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let plan = crate::compile::compile_equilibrium(self, clamps)?;
        let blocks: Vec<Vec<usize>> =
            plan.blocks.iter().map(|block| block.variables().to_vec()).collect();
        let mut values = initial.to_vec();
        for c in clamps {
            values[c.variable] = c.value;
        }
        let mut total = 0;
        let mut globally_unique = true;
        for block in &blocks {
            let bound = self.block_bound(block);
            let contraction = bound < 1.0;
            globally_unique &= contraction || self.affine_certificate(block);
            if policy.require_contraction && !contraction {
                return Err(EquilibriumError::Uncertified { bound });
            }
            let mut converged = false;
            for _ in 0..policy.max_iterations {
                check_cancel(ctx)?;
                let residual = self.residual(&values, noise, block);
                let norm = max_abs(&residual);
                if !norm.is_finite() {
                    return Err(EquilibriumError::Nonconvergence { residual: norm });
                }
                if norm <= policy.tolerance {
                    converged = true;
                    break;
                }
                total += 1;
                if contraction {
                    for (&i, &r) in block.iter().zip(&residual) {
                        values[i] -= r;
                    }
                } else {
                    let mut jac = self.residual_jacobian(&values, block);
                    let mut step = residual;
                    if !gauss_jordan_solve(&mut jac, &mut step, block.len(), 1) {
                        return Err(EquilibriumError::Singular);
                    }
                    let old = values.clone();
                    let mut accepted = false;
                    let mut scale = 1.0;
                    for _ in 0..40 {
                        for (&i, &delta) in block.iter().zip(&step) {
                            values[i] = old[i] - scale * delta;
                        }
                        let next = max_abs(&self.residual(&values, noise, block));
                        if next.is_finite() && next < norm {
                            accepted = true;
                            break;
                        }
                        scale *= 0.5;
                    }
                    if !accepted {
                        return Err(EquilibriumError::Nonconvergence { residual: norm });
                    }
                }
            }
            let residual = max_abs(&self.residual(&values, noise, block));
            if !converged && residual > policy.tolerance {
                return Err(EquilibriumError::Nonconvergence { residual });
            }
        }
        let free = self.free_indices(clamps)?;
        let residual = max_abs(&self.residual(&values, noise, &free));
        if !finite(&values) || !residual.is_finite() || residual > policy.tolerance {
            return Err(EquilibriumError::Nonconvergence { residual });
        }
        let jacobian_condition_estimate = self.condition_estimate(&values, &free);
        Ok(EquilibriumResult {
            values,
            residual,
            iterations: total,
            globally_unique,
            existence_certified: globally_unique,
            numerical_converged: true,
            dynamic_stability: None,
            jacobian_condition_estimate,
            blocks,
        })
    }

    fn condition_estimate(&self, values: &[f64], free: &[usize]) -> Option<f64> {
        let n = free.len();
        if n == 0 {
            return None;
        }
        let mut matrix = self.residual_jacobian(values, free);
        let norm = matrix
            .chunks_exact(n)
            .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let mut inverse = vec![0.0; n * n];
        for i in 0..n {
            inverse[i * n + i] = 1.0;
        }
        if !gauss_jordan_solve(&mut matrix, &mut inverse, n, n) {
            return None;
        }
        let inverse_norm = (0..n)
            .map(|i| (0..n).map(|j| inverse[j * n + i].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let estimate = norm * inverse_norm;
        estimate.is_finite().then_some(estimate)
    }

    pub(crate) fn affine_certificate(&self, block: &[usize]) -> bool {
        if block.iter().any(|&i| {
            self.equations[i].expression.is_some()
                || self.equations[i].activation != Activation::Identity
        }) {
            return false;
        }
        let weights: Vec<_> = block
            .iter()
            .flat_map(|&i| block.iter().map(move |&j| self.equations[i].weights[j]))
            .collect();
        crate::affine_certificate::certifies_affine_bijection(&weights, block.len())
    }
    fn block_bound(&self, block: &[usize]) -> f64 {
        // Positive summation error is bounded conservatively before comparing to one.
        let inflation = 1.0 + (block.len() + 1) as f64 * f64::EPSILON;
        block
            .iter()
            .map(|&i| {
                let bounds = self.equations[i]
                    .derivative_bounds()
                    .unwrap_or_else(|_| vec![f64::INFINITY; self.order()]);
                block.iter().map(|&j| bounds[j]).sum::<f64>() * inflation
            })
            .fold(0.0, f64::max)
    }

    pub(crate) fn residual(&self, state: &[f64], noise: &[f64], free: &[usize]) -> Vec<f64> {
        free.iter()
            .map(|&i| {
                let e = &self.equations[i];
                state[i] - e.structural_value(state).unwrap_or(f64::NAN) - noise[i]
            })
            .collect()
    }

    pub(crate) fn residual_jacobian(&self, state: &[f64], free: &[usize]) -> Vec<f64> {
        let mut jac = Vec::with_capacity(free.len() * free.len());
        for &i in free {
            let e = &self.equations[i];
            let derivatives =
                e.input_derivatives(state).unwrap_or_else(|_| vec![f64::NAN; state.len()]);
            for &j in free {
                jac.push((if i == j { 1.0 } else { 0.0 }) - derivatives[j]);
            }
        }
        jac
    }

    /// Recover additive structural noise from a fully observed factual state.
    /// This conditions on a specified model; it does not identify that model from data.
    pub fn abduce(&self, factual: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        let means = self.mechanism_values(factual)?;
        Ok(factual.iter().zip(means).map(|(x, m)| x - m).collect())
    }

    /// Abduce once and reuse the same disturbances in the intervention world.
    pub fn counterfactual(
        &self,
        factual: &[f64],
        clamps: &[Clamp],
        policy: SolvePolicy,
        ctx: &ExecutionContext,
    ) -> Result<EquilibriumResult, EquilibriumError> {
        self.solve(&self.abduce(factual)?, clamps, factual, policy, ctx)
    }

    /// Joint equilibrium log density conditional on known clamps. Includes
    /// `log |det(I - J_f,FF)|` and uses marginal Gaussian disturbance covariance.
    /// Global contraction is required here to avoid treating one inverse branch
    /// of a multiple-equilibrium system as a normalized density.
    #[allow(
        clippy::float_cmp,
        reason = "observed hard clamps must equal the declared structural value exactly"
    )]
    pub fn log_likelihood(
        &self,
        rows: &[RegimeRow],
        ctx: &ExecutionContext,
    ) -> Result<f64, EquilibriumError> {
        self.validate()?;
        let n = self.order();
        let tape_nodes = self
            .equations
            .iter()
            .filter_map(|e| {
                e.expression
                    .as_ref()
                    .map(|expr| expr.compile(n, e.parameters.len()).map(|t| t.nodes()))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let _lease = ctx
            .acquire_admission(
                1,
                u64::try_from(64 * n * n + 128 * n + tape_nodes * (2 * n + 8) * 8)
                    .map_err(|_| invalid("resource size"))?,
            )
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let mut sum = 0.0;
        for row in rows {
            check_cancel(ctx)?;
            self.check_vector(&row.values)?;
            let free = self.free_indices(&row.clamps)?;
            for c in &row.clamps {
                if row.values[c.variable] != c.value {
                    return Err(invalid("observed value differs from known clamp"));
                }
            }
            for block in self.blocks(&row.clamps)? {
                let bound = self.block_bound(&block);
                if bound >= 1.0 && !self.affine_certificate(&block) {
                    return Err(EquilibriumError::Uncertified { bound });
                }
            }
            if free.is_empty() {
                continue;
            }
            let k = free.len();
            let cov: Vec<_> = free
                .iter()
                .flat_map(|&i| free.iter().map(move |&j| self.noise_covariance[i * n + j]))
                .collect();
            let l = cholesky(&cov, k)?;
            let mut z = self.residual(&row.values, &vec![0.0; n], &free);
            for i in 0..k {
                for j in 0..i {
                    z[i] -= l[i * k + j] * z[j];
                }
                z[i] /= l[i * k + i];
            }
            let noise_logdet = 2.0 * (0..k).map(|i| l[i * k + i].ln()).sum::<f64>();
            let jac = self.residual_jacobian(&row.values, &free);
            let logdet = log_abs_determinant(jac, k)?;
            sum += logdet
                - 0.5 * (dot(&z, &z) + noise_logdet + k as f64 * (2.0 * std::f64::consts::PI).ln());
        }
        if !sum.is_finite() {
            return Err(invalid("nonfinite likelihood"));
        }
        Ok(sum)
    }
}

/// Analytic likelihood score for each mechanism, conditional on the declared graph.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismLikelihoodScore {
    /// Score of the additive intercept.
    pub intercept: f64,
    /// Legacy coefficient scores; empty for a compositional mechanism.
    pub weights: Vec<f64>,
    /// Local compositional parameter scores; empty for a legacy mechanism.
    pub parameters: Vec<f64>,
}
impl EquilibriumModel {
    /// Joint likelihood and exact differentiation through its free-system Jacobian.
    /// The same contraction and surviving-disturbance conditions as `log_likelihood` apply.
    #[allow(
        clippy::too_many_lines,
        reason = "one score pass keeps free-system noise and Jacobian derivatives together"
    )]
    pub fn likelihood_score(
        &self,
        rows: &[RegimeRow],
        ctx: &ExecutionContext,
    ) -> Result<(f64, Vec<MechanismLikelihoodScore>), EquilibriumError> {
        let likelihood = self.log_likelihood(rows, ctx)?;
        let n = self.order();
        let mut scores: Vec<_> = self
            .equations
            .iter()
            .map(|e| MechanismLikelihoodScore {
                intercept: 0.0,
                weights: if e.expression.is_none() { vec![0.0; n] } else { vec![] },
                parameters: vec![0.0; e.parameters.len()],
            })
            .collect();
        let tape_nodes = self
            .equations
            .iter()
            .filter_map(|e| {
                e.expression
                    .as_ref()
                    .map(|expression| expression.compile(n, e.parameters.len()).map(|t| t.nodes()))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let bytes = u64::try_from(96 * n * n + 128 * n + tape_nodes * (3 * n + 10) * 8)
            .map_err(|_| invalid("score resources"))?;
        let _lease = ctx
            .acquire_admission(1, bytes)
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        for row in rows {
            check_cancel(ctx)?;
            let free = self.free_indices(&row.clamps)?;
            let k = free.len();
            if k == 0 {
                continue;
            }
            let covariance: Vec<_> = free
                .iter()
                .flat_map(|&i| free.iter().map(move |&j| self.noise_covariance[i * n + j]))
                .collect();
            let lower = cholesky(&covariance, k)?;
            let mut noise_score = self.residual(&row.values, &vec![0.0; n], &free);
            for i in 0..k {
                for j in 0..i {
                    noise_score[i] -= lower[i * k + j] * noise_score[j];
                }
                noise_score[i] /= lower[i * k + i];
            }
            for i in (0..k).rev() {
                for j in i + 1..k {
                    noise_score[i] -= lower[j * k + i] * noise_score[j];
                }
                noise_score[i] /= lower[i * k + i];
            }
            // Matrix is row-major; right-hand sides and the inverse are column-major.
            let mut matrix = self.residual_jacobian(&row.values, &free);
            let mut inverse = vec![0.0; k * k];
            for i in 0..k {
                inverse[i * k + i] = 1.0;
            }
            if !gauss_jordan_solve(&mut matrix, &mut inverse, k, k) {
                return Err(EquilibriumError::Singular);
            }
            for (local_i, &i) in free.iter().enumerate() {
                let equation = &self.equations[i];
                let score = &mut scores[i];
                score.intercept += noise_score[local_i];
                let mut gradient_adjoint = vec![0.0; n];
                for (local_j, &j) in free.iter().enumerate() {
                    gradient_adjoint[j] = -inverse[local_i * k + local_j];
                }
                if let Some(expression) = &equation.expression {
                    let tape = expression.compile(n, equation.parameters.len())?;
                    let evaluation = tape.evaluate(&row.values, &equation.parameters)?;
                    let params =
                        tape.parameter_score(&evaluation, noise_score[local_i], &gradient_adjoint)?;
                    for (sum, value) in score.parameters.iter_mut().zip(params) {
                        *sum += value;
                    }
                } else {
                    let z = dot(&equation.weights, &row.values);
                    let first = equation.activation.derivative(z);
                    let second = match equation.activation {
                        Activation::Identity | Activation::Relu => 0.0,
                        Activation::Tanh => -2.0 * z.tanh() * first,
                        Activation::Sin => -z.sin(),
                    };
                    let contracted: f64 =
                        gradient_adjoint.iter().zip(&equation.weights).map(|(a, w)| a * w).sum();
                    for (j, adjoint) in gradient_adjoint.iter().enumerate() {
                        score.weights[j] += noise_score[local_i] * first * row.values[j]
                            + second * row.values[j] * contracted
                            + first * adjoint;
                    }
                }
            }
        }
        if scores.iter().any(|s| {
            !s.intercept.is_finite()
                || s.weights.iter().chain(&s.parameters).any(|v| !v.is_finite())
        }) {
            return Err(invalid("nonfinite likelihood score"));
        }
        Ok((likelihood, scores))
    }
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|x| if x.is_finite() { x.abs() } else { f64::INFINITY }).fold(0.0, f64::max)
}
pub(crate) fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, EquilibriumError> {
    if a.len() != n * n || !finite(a) {
        return Err(invalid("covariance shape or nonfinite"));
    }
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            if (a[i * n + j] - a[j * n + i]).abs()
                > 1e-12 * (1.0 + a[i * n + j].abs().max(a[j * n + i].abs()))
            {
                return Err(invalid("asymmetric covariance"));
            }
            let residual = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            if i == j {
                if residual <= 0.0 {
                    return Err(EquilibriumError::Singular);
                }
                l[i * n + j] = residual.sqrt();
            } else {
                l[i * n + j] = residual / l[j * n + j];
            }
        }
    }
    Ok(l)
}

pub(crate) fn log_abs_determinant(mut a: Vec<f64>, n: usize) -> Result<f64, EquilibriumError> {
    let mut value = 0.0;
    for j in 0..n {
        let best = (j..n)
            .max_by(|&i, &k| a[i * n + j].abs().total_cmp(&a[k * n + j].abs()))
            .ok_or(EquilibriumError::Singular)?;
        if a[best * n + j].abs() <= 1e-14 {
            return Err(EquilibriumError::Singular);
        }
        for k in 0..n {
            a.swap(j * n + k, best * n + k);
        }
        let pivot = a[j * n + j];
        value += pivot.abs().ln();
        for i in j + 1..n {
            let scale = a[i * n + j] / pivot;
            for k in j + 1..n {
                a[i * n + k] -= scale * a[j * n + k];
            }
        }
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model(a: f64, b: f64, activation: Activation) -> EquilibriumModel {
        EquilibriumModel {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.0,
                    weights: vec![0.0, a],
                    activation,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![b, 0.0],
                    activation,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        }
    }
    #[test]
    fn nonlinear_equilibrium_linear_truth_clamps_and_shared_noise() {
        let m = model(0.4, 0.5, Activation::Identity);
        let ctx = ExecutionContext::for_tests(1);
        let r = m.solve(&[1.0, 0.0], &[], &[0.0; 2], SolvePolicy::default(), &ctx).unwrap();
        assert!((r.values[1] - 0.625).abs() < 1e-9);
        assert!(r.globally_unique);
        let cf = m
            .counterfactual(
                &r.values,
                &[Clamp { variable: 0, value: 2.0 }],
                SolvePolicy::default(),
                &ctx,
            )
            .unwrap();
        assert!((cf.values[1] - 1.0).abs() < 1e-9);
        assert_eq!(cf.blocks, vec![vec![1]]);
        let no_action = m.counterfactual(&r.values, &[], SolvePolicy::default(), &ctx).unwrap();
        assert!(no_action.values.iter().zip(r.values).all(|(a, b)| (a - b).abs() < 1e-9));
    }
    #[test]
    fn nonlinear_equilibrium_saturation_and_dag_limit() {
        let m = model(0.4, 0.5, Activation::Tanh);
        let r = m
            .solve(
                &[0.7, -0.2],
                &[],
                &[10.0, -10.0],
                SolvePolicy::default(),
                &ExecutionContext::for_tests(2),
            )
            .unwrap();
        assert_eq!(r.blocks, vec![vec![0, 1]]);
        let means = m.mechanism_values(&r.values).unwrap();
        assert!((r.values[0] - means[0] - 0.7).abs() < 1e-9);
        let mut dag = m;
        dag.equations[0].weights[1] = 0.0;
        let r = dag
            .solve(
                &[0.7, -0.2],
                &[],
                &[0.0; 2],
                SolvePolicy::default(),
                &ExecutionContext::for_tests(3),
            )
            .unwrap();
        assert_eq!(r.blocks, vec![vec![0], vec![1]]);
        assert!((r.values[1] - (0.35_f64.tanh() - 0.2)).abs() < 1e-10);
    }
    #[test]
    fn nonlinear_equilibrium_noncontractive_affine_bijection_has_an_independent_certificate() {
        let m = model(2.0, -2.0, Activation::Identity);
        let ctx = ExecutionContext::for_tests(3);
        assert!(matches!(
            m.solve(&[1.0, 0.0], &[], &[0.0; 2], SolvePolicy::default(), &ctx),
            Err(EquilibriumError::Uncertified { .. })
        ));
        let p = SolvePolicy { require_contraction: false, ..SolvePolicy::default() };
        let r = m.solve(&[1.0, 0.0], &[], &[0.0; 2], p, &ctx).unwrap();
        assert!(r.globally_unique);
        assert!(r.existence_certified);
        assert!(r.numerical_converged);
        assert_eq!(r.dynamic_stability, None);
        assert!((r.jacobian_condition_estimate.unwrap() - 1.8).abs() < 1e-12);
        assert!((r.values[0] - 0.2).abs() < 1e-12);
        assert!((r.values[1] + 0.4).abs() < 1e-12);
    }
    #[test]
    fn nonlinear_equilibrium_likelihood_keeps_jacobian_and_marginal_noise() {
        let mut m = model(0.4, 0.5, Activation::Identity);
        m.noise_covariance = vec![1.0, 0.6, 0.6, 1.0];
        let ctx = ExecutionContext::for_tests(4);
        let ll =
            m.log_likelihood(&[RegimeRow { values: vec![0.0; 2], clamps: vec![] }], &ctx).unwrap();
        let expected = 0.8_f64.ln() - (2.0 * std::f64::consts::PI).ln() - 0.5 * 0.64_f64.ln();
        assert!((ll - expected).abs() < 1e-12);
        let ll = m
            .log_likelihood(
                &[RegimeRow {
                    values: vec![2.0, 2.0],
                    clamps: vec![Clamp { variable: 0, value: 2.0 }],
                }],
                &ctx,
            )
            .unwrap();
        assert!((ll + 0.5 * (1.0 + (2.0 * std::f64::consts::PI).ln())).abs() < 1e-12);
    }
    #[test]
    fn nonlinear_equilibrium_multiroots_do_not_gain_density_license() {
        let m = model(2.0, 2.0, Activation::Tanh);
        let ctx = ExecutionContext::for_tests(5);
        let p = SolvePolicy { require_contraction: false, ..SolvePolicy::default() };
        let positive = m.solve(&[0.0; 2], &[], &[1.0; 2], p, &ctx).unwrap();
        let negative = m.solve(&[0.0; 2], &[], &[-1.0; 2], p, &ctx).unwrap();
        assert!(positive.values[0] > 0.9 && negative.values[0] < -0.9);
        assert!(!positive.globally_unique);
        assert!(matches!(
            m.log_likelihood(&[RegimeRow { values: positive.values, clamps: vec![] }], &ctx),
            Err(EquilibriumError::Uncertified { .. })
        ));
    }
    #[test]
    fn nonlinear_equilibrium_invalid_cancelled_budget_and_nonconvergence() {
        let mut m = model(0.4, 0.5, Activation::Identity);
        let ctx = ExecutionContext::for_tests(6);
        let p = SolvePolicy { max_iterations: 1, ..SolvePolicy::default() };
        assert!(matches!(
            m.solve(&[1.0, 0.0], &[], &[0.0; 2], p, &ctx),
            Err(EquilibriumError::Nonconvergence { .. })
        ));
        assert!(
            m.solve(&[0.0; 2], &[Clamp { variable: 0, value: 0.0 }; 2], &[0.0; 2], p, &ctx)
                .is_err()
        );
        let short = ExecutionContext::for_tests(1)
            .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, 1));
        assert!(matches!(
            m.solve(&[0.0; 2], &[], &[0.0; 2], p, &short),
            Err(EquilibriumError::Resource(_))
        ));
        ctx.cancellation.cancel();
        assert!(matches!(
            m.solve(&[0.0; 2], &[], &[0.0; 2], p, &ctx),
            Err(EquilibriumError::Resource(_))
        ));
        m.equations[0].weights[0] = f64::NAN;
        assert!(m.validate().is_err());
    }
}

/// A numerical/certificate failure retained in its original world/draw position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquilibriumSolveFailure {
    /// `uncertified`, `nonconvergence`, or `singular`; none means nonexistence.
    pub failure: String,
    /// Numerical/certificate diagnostic, not a mathematical impossibility claim.
    pub detail: String,
}

/// Draw-aligned answer that never drops failed or uncertified response mass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EquilibriumOutcome {
    /// A solved state with its actual certificate/iteration diagnostics.
    Solved(EquilibriumResult),
    /// A retained unsuccessful attempt; no renormalization over successful draws.
    Failed(EquilibriumSolveFailure),
}
impl EquilibriumModel {
    /// Attempt one response while retaining numerical/certificate failures.
    /// Malformed input, cancellation and resource refusals still abort the run.
    pub fn solve_outcome(
        &self,
        noise: &[f64],
        clamps: &[Clamp],
        initial: &[f64],
        policy: SolvePolicy,
        ctx: &ExecutionContext,
    ) -> Result<EquilibriumOutcome, EquilibriumError> {
        match self.solve(noise, clamps, initial, policy, ctx) {
            Ok(solution) => Ok(EquilibriumOutcome::Solved(solution)),
            Err(
                error @ (EquilibriumError::Uncertified { .. }
                | EquilibriumError::Nonconvergence { .. }
                | EquilibriumError::Singular),
            ) => {
                let failure = match &error {
                    EquilibriumError::Uncertified { .. } => "uncertified",
                    EquilibriumError::Nonconvergence { .. } => "nonconvergence",
                    _ => "singular",
                };
                Ok(EquilibriumOutcome::Failed(EquilibriumSolveFailure {
                    failure: failure.into(),
                    detail: error.to_string(),
                }))
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod compositional_tests {
    use super::*;
    use crate::mechanism_expr::MechanismExpression as E;
    fn expression() -> E {
        E::Multiply {
            factors: vec![
                E::Parameter { parameter: 0 },
                E::Tanh {
                    argument: Box::new(E::Add {
                        terms: vec![
                            E::Multiply {
                                factors: vec![
                                    E::Parameter { parameter: 1 },
                                    E::Variable { variable: 0 },
                                ],
                            },
                            E::Multiply {
                                factors: vec![
                                    E::Parameter { parameter: 2 },
                                    E::Variable { variable: 1 },
                                ],
                            },
                        ],
                    }),
                },
            ],
        }
    }
    fn model() -> EquilibriumModel {
        EquilibriumModel {
            equations: vec![
                Equation {
                    id: "self-and-cross".into(),
                    intercept: 0.1,
                    weights: vec![0.0; 2],
                    activation: Activation::Identity,
                    expression: Some(expression()),
                    parameters: vec![0.3, 0.2, -0.4],
                },
                Equation {
                    id: "return".into(),
                    intercept: -0.2,
                    weights: vec![0.5, 0.1],
                    activation: Activation::Sin,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![1.0, 0.3, 0.3, 0.8],
        }
    }
    #[test]
    fn composed_self_feedback_scores_include_free_system_jacobian_and_joint_noise() {
        let ctx = ExecutionContext::for_tests(19);
        let m = model();
        let rows = vec![
            RegimeRow { values: vec![0.4, -0.7], clamps: vec![] },
            RegimeRow { values: vec![-0.2, 1.0], clamps: vec![Clamp { variable: 1, value: 1.0 }] },
        ];
        let (_, scores) = m.likelihood_score(&rows, &ctx).unwrap();
        for (i, score) in scores.iter().enumerate() {
            for j in 0..score.parameters.len() + score.weights.len() + 1 {
                let mut high = m.clone();
                let mut low = m.clone();
                let expected = if j == 0 {
                    high.equations[i].intercept += 1e-6;
                    low.equations[i].intercept -= 1e-6;
                    score.intercept
                } else if !score.parameters.is_empty() {
                    high.equations[i].parameters[j - 1] += 1e-6;
                    low.equations[i].parameters[j - 1] -= 1e-6;
                    score.parameters[j - 1]
                } else {
                    high.equations[i].weights[j - 1] += 1e-6;
                    low.equations[i].weights[j - 1] -= 1e-6;
                    score.weights[j - 1]
                };
                let numerical = (high.log_likelihood(&rows, &ctx).unwrap()
                    - low.log_likelihood(&rows, &ctx).unwrap())
                    / 2e-6;
                assert!(
                    (expected - numerical).abs() < 2e-8,
                    "equation {i} coordinate {j}: analytic {expected} numerical {numerical}"
                );
            }
        }
        let result = m.solve(&[0.1, -0.1], &[], &[0.0; 2], SolvePolicy::default(), &ctx).unwrap();
        assert!(result.globally_unique);
        assert!(result.residual < 1e-9);
        let singleton = EquilibriumModel {
            equations: vec![Equation {
                id: "self".into(),
                intercept: 1.0,
                weights: vec![0.5],
                activation: Activation::Identity,
                expression: None,
                parameters: vec![],
            }],
            noise_covariance: vec![1.0],
        };
        let plan = crate::system_compile::compile_equilibrium(&singleton, &[]).unwrap();
        assert!(matches!(
            plan.blocks[0],
            crate::system_compile::ExecutionBlock::SimultaneousEquations { .. }
        ));
        assert!(
            (singleton.solve(&[0.0], &[], &[0.0], SolvePolicy::default(), &ctx).unwrap().values[0]
                - 2.0)
                .abs()
                < 1e-8
        );
    }
}
