//! Distribution providers for compiled expression evaluation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use causal_core::{Value, VariableId};

use crate::{DomainRef, InterventionAssignment};

/// Evaluation context (optional posterior draw index).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EvalContext {
    /// Posterior draw index when evaluating against a draw-indexed provider.
    pub draw: Option<usize>,
}

/// Variable → value binding for density / outcome lookup.
#[derive(Clone, Debug, Default)]
pub struct Assignment {
    /// Sorted by variable id.
    entries: Vec<(VariableId, Value)>,
}

impl Assignment {
    /// Empty assignment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from unsorted pairs (sorted + last-wins on duplicate vars).
    #[must_use]
    pub fn from_pairs(pairs: impl IntoIterator<Item = (VariableId, Value)>) -> Self {
        let mut entries: Vec<(VariableId, Value)> = pairs.into_iter().collect();
        entries.sort_by_key(|(v, _)| v.raw());
        entries.dedup_by_key(|(v, _)| *v);
        Self { entries }
    }

    /// Insert or replace a binding.
    pub fn set(&mut self, var: VariableId, value: Value) {
        match self.entries.binary_search_by_key(&var.raw(), |(v, _)| v.raw()) {
            Ok(i) => self.entries[i].1 = value,
            Err(i) => self.entries.insert(i, (var, value)),
        }
    }

    /// Borrow value for `var`, if present.
    #[must_use]
    pub fn get(&self, var: VariableId) -> Option<&Value> {
        self.entries
            .binary_search_by_key(&var.raw(), |(v, _)| v.raw())
            .ok()
            .map(|i| &self.entries[i].1)
    }

    /// All bindings, sorted.
    #[must_use]
    pub fn entries(&self) -> &[(VariableId, Value)] {
        &self.entries
    }

    /// Extend with another assignment (other wins on conflict).
    pub fn extend_from(&mut self, other: &Assignment) {
        for (v, val) in &other.entries {
            self.set(*v, val.clone());
        }
    }

    /// Restrict to the given variables (order of `vars` preserved in returned values).
    pub fn values_for(&self, vars: &[VariableId]) -> Result<Vec<Value>, EvalError> {
        let mut out = Vec::with_capacity(vars.len());
        for &v in vars {
            let Some(val) = self.get(v) else {
                return Err(EvalError::MissingBinding(v));
            };
            out.push(val.clone());
        }
        Ok(out)
    }
}

/// Resolved distribution factor identity (no string keys).
#[derive(Clone, Debug)]
pub struct FactorSpec<'a> {
    /// Factor variables.
    pub variables: &'a [VariableId],
    /// Conditioning variables.
    pub conditioned_on: &'a [VariableId],
    /// Hard intervention assignments.
    pub intervention: &'a [InterventionAssignment],
    /// Observational vs interventional domain.
    pub domain: DomainRef,
}

/// Errors from compiling or evaluating causal expressions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvalError {
    /// Continuous `IntegralOut` without quadrature nodes and without discrete support.
    UnsupportedIntegralOut,
    /// Provider has no entry for the requested factor / assignment.
    MissingTableEntry,
    /// Required variable binding absent from the assignment.
    MissingBinding(VariableId),
    /// Provider reported empty support for a summed variable.
    EmptySupport(VariableId),
    /// Division by zero while evaluating a ratio.
    DivisionByZero,
    /// Posterior draw index out of range.
    DrawOutOfRange {
        /// Requested draw.
        draw: usize,
        /// Number of draws available.
        n_draws: usize,
    },
    /// Support row length does not match requested variable count.
    SupportShape {
        /// Expected arity.
        expected: usize,
        /// Actual arity.
        actual: usize,
    },
    /// Empirical provider used where posterior draws are required (or vice versa).
    ProviderKind(&'static str),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedIntegralOut => {
                write!(
                    f,
                    "IntegralOut requires provider quadrature nodes or discrete support"
                )
            }
            Self::MissingTableEntry => write!(f, "missing probability table entry"),
            Self::MissingBinding(v) => write!(f, "missing binding for V{}", v.raw()),
            Self::EmptySupport(v) => write!(f, "empty support for V{}", v.raw()),
            Self::DivisionByZero => write!(f, "division by zero in ratio"),
            Self::DrawOutOfRange { draw, n_draws } => {
                write!(f, "draw {draw} out of range (n_draws={n_draws})")
            }
            Self::SupportShape { expected, actual } => {
                write!(f, "support row arity {actual} != expected {expected}")
            }
            Self::ProviderKind(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for EvalError {}

/// Provides densities, discrete supports, and outcome values for evaluation.
pub trait DistributionProvider {
    /// Probability / density mass for a factor under an assignment.
    ///
    /// # Errors
    ///
    /// Missing table entries, bad draw index, or shape errors.
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError>;

    /// Discrete support for variables (cartesian rows of values aligned to `vars`).
    ///
    /// # Errors
    ///
    /// Empty domains or unsupported queries.
    fn support(
        &self,
        vars: &[VariableId],
        ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError>;

    /// Optional continuous quadrature nodes `(assignment_row, Lebesgue weight)` for
    /// [`crate::ExprNode::IntegralOut`].
    ///
    /// Returning `Ok(None)` asks the evaluator to fall back to discrete [`Self::support`].
    ///
    /// # Errors
    ///
    /// Provider-specific continuous-integration failures.
    fn quadrature(
        &self,
        _vars: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Option<Arc<[(Arc<[Value]>, f64)]>>, EvalError> {
        Ok(None)
    }

    /// Outcome function value (identity: the bound value of `var`).
    ///
    /// # Errors
    ///
    /// Missing binding or non-numeric value.
    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError>;

    /// Number of posterior draws, or `None` for a single empirical world.
    fn n_draws(&self) -> Option<usize>;
}

/// Canonical key for a factor table row.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FactorKey {
    variables: Arc<[VariableId]>,
    conditioned_on: Arc<[VariableId]>,
    intervention: Arc<[InterventionAssignment]>,
    domain: DomainRef,
    /// Concatenation of values for `variables` then `conditioned_on`.
    values: Arc<[Value]>,
}

fn factor_key(spec: &FactorSpec<'_>, assignment: &Assignment) -> Result<FactorKey, EvalError> {
    let mut values = assignment.values_for(spec.variables)?;
    values.extend(assignment.values_for(spec.conditioned_on)?);
    Ok(FactorKey {
        variables: Arc::from(spec.variables),
        conditioned_on: Arc::from(spec.conditioned_on),
        intervention: Arc::from(spec.intervention.to_vec()),
        domain: spec.domain,
        values: Arc::from(values),
    })
}

/// Tabular empirical distribution provider (discrete factors + domains).
#[derive(Clone, Debug, Default)]
pub struct EmpiricalTableProvider {
    domains: HashMap<VariableId, Arc<[Value]>>,
    tables: HashMap<FactorKey, f64>,
}

impl EmpiricalTableProvider {
    /// Empty provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare discrete domain for a variable.
    pub fn set_domain(&mut self, var: VariableId, values: impl IntoIterator<Item = Value>) {
        let mut v: Vec<Value> = values.into_iter().collect();
        // Stable unique by hash equality.
        let mut seen = std::collections::HashSet::new();
        v.retain(|x| seen.insert(x.clone()));
        self.domains.insert(var, Arc::from(v));
    }

    /// Insert a factor probability for the given spec + assignment.
    ///
    /// # Errors
    ///
    /// Missing bindings for factor variables / conditions.
    pub fn insert_probability(
        &mut self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        probability: f64,
    ) -> Result<(), EvalError> {
        let key = factor_key(spec, assignment)?;
        self.tables.insert(key, probability);
        Ok(())
    }
}

impl DistributionProvider for EmpiricalTableProvider {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        let key = factor_key(spec, assignment)?;
        self.tables.get(&key).copied().ok_or(EvalError::MissingTableEntry)
    }

    fn support(
        &self,
        vars: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        if vars.is_empty() {
            return Ok(Arc::from(vec![Arc::from(Vec::<Value>::new())]));
        }
        let mut rows: Vec<Vec<Value>> = vec![Vec::new()];
        for &v in vars {
            let domain = self.domains.get(&v).ok_or(EvalError::EmptySupport(v))?;
            if domain.is_empty() {
                return Err(EvalError::EmptySupport(v));
            }
            let mut next = Vec::with_capacity(rows.len() * domain.len());
            for prefix in &rows {
                for val in domain.iter() {
                    let mut row = prefix.clone();
                    row.push(val.clone());
                    next.push(row);
                }
            }
            rows = next;
        }
        let out: Vec<Arc<[Value]>> = rows.into_iter().map(Arc::from).collect();
        Ok(Arc::from(out))
    }

    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        let value = assignment.get(var).ok_or(EvalError::MissingBinding(var))?;
        value.as_f64().ok_or(EvalError::MissingBinding(var))
    }

    fn n_draws(&self) -> Option<usize> {
        None
    }
}

/// Draw-indexed posterior provider: one [`EmpiricalTableProvider`] per draw.
#[derive(Clone, Debug, Default)]
pub struct PosteriorDrawProvider {
    draws: Vec<EmpiricalTableProvider>,
}

impl PosteriorDrawProvider {
    /// Empty posterior provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from per-draw empirical tables.
    #[must_use]
    pub fn from_draws(draws: Vec<EmpiricalTableProvider>) -> Self {
        Self { draws }
    }

    /// Number of draws.
    #[must_use]
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Whether there are no draws.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    fn table(&self, ctx: &EvalContext) -> Result<&EmpiricalTableProvider, EvalError> {
        let draw = ctx
            .draw
            .ok_or(EvalError::ProviderKind("PosteriorDrawProvider requires EvalContext.draw"))?;
        self.draws.get(draw).ok_or(EvalError::DrawOutOfRange { draw, n_draws: self.draws.len() })
    }
}

impl DistributionProvider for PosteriorDrawProvider {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        self.table(ctx)?.probability(spec, assignment, ctx)
    }

    fn support(
        &self,
        vars: &[VariableId],
        ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        self.table(ctx)?.support(vars, ctx)
    }

    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        self.table(ctx)?.outcome(var, assignment, ctx)
    }

    fn n_draws(&self) -> Option<usize> {
        Some(self.draws.len())
    }
}

/// Independent Gaussian density provider with Gauss–Hermite quadrature.
///
/// `probability` returns the product of univariate N(μ, σ²) densities for factor
/// variables. [`Self::quadrature`] returns product Gauss–Hermite nodes in
/// Lebesgue measure (suitable for ∫ body(x) dx near each Gaussian mode).
#[derive(Clone, Debug, Default)]
pub struct GaussianDensityProvider {
    /// Per-variable (mean, variance).
    params: HashMap<VariableId, (f64, f64)>,
}

impl GaussianDensityProvider {
    /// Empty provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare an independent Gaussian for `var` with mean `mean` and variance `variance`.
    ///
    /// # Panics
    ///
    /// Never panics; non-positive variance is rejected by returning early no-op... actually
    /// we clamp: variance must be > 0 or the insert is skipped. Prefer validating at call sites.
    pub fn set_gaussian(&mut self, var: VariableId, mean: f64, variance: f64) {
        if variance > 0.0 && variance.is_finite() && mean.is_finite() {
            self.params.insert(var, (mean, variance));
        }
    }
}

/// Physicists' Gauss–Hermite nodes/weights for ∫ e^{-t²} g(t) dt (n = 5).
const GH5_NODES: [f64; 5] = [
    -2.020_182_870_456_085_6,
    -0.958_572_464_613_818_5,
    0.0,
    0.958_572_464_613_818_5,
    2.020_182_870_456_085_6,
];
const GH5_WEIGHTS: [f64; 5] = [
    0.019_953_242_059_045_913,
    0.393_619_323_152_241_35,
    0.945_308_720_482_941_9,
    0.393_619_323_152_241_35,
    0.019_953_242_059_045_913,
];

impl DistributionProvider for GaussianDensityProvider {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        let mut dens = 1.0;
        for &v in spec.variables {
            let (mean, var) = self.params.get(&v).copied().ok_or(EvalError::EmptySupport(v))?;
            let x = assignment
                .get(v)
                .and_then(Value::as_f64)
                .ok_or(EvalError::MissingBinding(v))?;
            let inv_sqrt = (2.0 * std::f64::consts::PI * var).sqrt().recip();
            let z = (x - mean) / var.sqrt();
            dens *= inv_sqrt * (-0.5 * z * z).exp();
        }
        Ok(dens)
    }

    fn support(
        &self,
        vars: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        if vars.is_empty() {
            return Ok(Arc::from(vec![Arc::from(Vec::<Value>::new())]));
        }
        Err(EvalError::EmptySupport(vars[0]))
    }

    fn quadrature(
        &self,
        vars: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Option<Arc<[(Arc<[Value]>, f64)]>>, EvalError> {
        if vars.is_empty() {
            return Ok(Some(Arc::from([(Arc::from(Vec::<Value>::new()), 1.0)])));
        }
        // Product GH: start with empty prefix.
        let mut nodes: Vec<(Vec<Value>, f64)> = vec![(Vec::new(), 1.0)];
        for &v in vars {
            let (mean, variance) =
                self.params.get(&v).copied().ok_or(EvalError::EmptySupport(v))?;
            let sigma = variance.sqrt();
            let scale = sigma * std::f64::consts::SQRT_2;
            let mut next = Vec::with_capacity(nodes.len() * GH5_NODES.len());
            for (prefix, w0) in &nodes {
                for (i, &t) in GH5_NODES.iter().enumerate() {
                    let x = mean + scale * t;
                    // GH computes ∫ e^{-t²} g(t) dt. For Lebesgue ∫ h(x) dx with
                    // x = μ + σ√2 t we need g(t) = h(x) σ√2 e^{t²}, so the node
                    // weight applied to h(x) is w_i · σ√2 · e^{t²}.
                    let w = w0 * GH5_WEIGHTS[i] * scale * (t * t).exp();
                    let mut row = prefix.clone();
                    row.push(Value::f64(x));
                    next.push((row, w));
                }
            }
            nodes = next;
        }
        let out: Vec<(Arc<[Value]>, f64)> =
            nodes.into_iter().map(|(row, w)| (Arc::from(row), w)).collect();
        Ok(Some(Arc::from(out)))
    }

    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        let value = assignment.get(var).ok_or(EvalError::MissingBinding(var))?;
        value.as_f64().ok_or(EvalError::MissingBinding(var))
    }

    fn n_draws(&self) -> Option<usize> {
        None
    }
}
