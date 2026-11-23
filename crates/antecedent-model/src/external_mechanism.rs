// SPDX-License-Identifier: MIT OR Apache-2.0
//! Registered external cyclic mechanisms that can take part in portable scientific artifacts.
//!
//! A portable record never embeds executable code. It names a mechanism by a versioned
//! implementation identifier, the BLAKE3 digest of the implementation's content, the
//! declared dependencies, and the declarations a solver needs: the input signature, the
//! parameter schema, how derivatives are obtained and any global Lipschitz bounds.
//!
//! Replaying a record needs a [`TrustedMechanismResolver`] supplied by the caller. The
//! resolver maps a recorded specification to an implementation; the implementation is run
//! only after its identifier, version, content digest, dependencies and declarations equal
//! the recorded ones. No resolver, no implementation, or any difference refuses with a typed
//! reason code. A system that contains such a mechanism is `verified_external`, never native.

use crate::local_cyclic::{LocalCyclicMechanism, LocalCyclicModel};
use crate::nonlinear_equilibrium::invalid_equilibrium_input as invalid;
use crate::nonlinear_equilibrium::{
    Clamp, Equation, EquilibriumError, EquilibriumResult, MAX_ORDER, SolvePolicy, check_cancel,
    cholesky, local_positions,
};
use crate::sparse_equilibrium::{
    CsrMatrix, FactorError, SolveEvidence, SolverBudgetCode, SparseLu, infinity_condition_estimate,
    nonnegative_spectral_upper_bound,
};
use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

mod likelihood;
mod ode;
mod registry;
pub use ode::{ExternalOdeSystem, ExternalOdeTrajectory, ResolvedOdeSystem};
use registry::text_ok;
pub use registry::{
    DerivativeDeclaration, DifferenceScheme, ExternalDependency, ExternalMechanismImplementation,
    ExternalMechanismRefusal, ExternalMechanismSpec, ExternalRefusalKind, MAX_EXTERNAL_ARITY,
    MAX_EXTERNAL_DEPENDENCIES, MechanismRegistry, MechanismTrust, TrustedMechanismResolver,
    verify_resolution,
};

/// Identifier of one fitted scalar of an external-capable system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExternalFitParameter {
    /// An equation intercept.
    Intercept {
        /// Dense equation index.
        equation: usize,
    },
    /// A parameter of an external mechanism, by schema position.
    Parameter {
        /// Dense equation index.
        equation: usize,
        /// Position in the mechanism's parameter schema.
        parameter: usize,
    },
}

impl ExternalFitParameter {
    /// Dense equation index of the target.
    #[must_use]
    pub const fn equation(self) -> usize {
        match self {
            Self::Intercept { equation } | Self::Parameter { equation, .. } => equation,
        }
    }
}

/// One equation of an external-capable system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "origin", rename_all = "snake_case", deny_unknown_fields)]
#[allow(
    clippy::large_enum_variant,
    reason = "the equation representation preserves its established owned API"
)]
pub enum SystemEquation {
    /// A built-in, serializable library mechanism.
    Builtin {
        /// The equation.
        equation: Equation,
    },
    /// A registered external mechanism, recorded by identity only.
    External {
        /// Stable mechanism identity, unique within the system.
        id: String,
        /// Additive intercept outside the external value.
        intercept: f64,
        /// Fitted parameter values, in schema order.
        parameters: Vec<f64>,
        /// Recorded identity and declarations.
        mechanism: ExternalMechanismSpec,
    },
}
impl SystemEquation {
    fn id(&self) -> &str {
        match self {
            Self::Builtin { equation } => &equation.id,
            Self::External { id, .. } => id,
        }
    }
}

fn validate_equations(equations: &[SystemEquation]) -> Result<(), EquilibriumError> {
    let n = equations.len();
    if n == 0 {
        return Err(invalid("external system needs at least one equation"));
    }
    if n > MAX_ORDER {
        return Err(crate::sparse_equilibrium::SolverBudgetCode::ModelOrder
            .refusal(n as u64, MAX_ORDER as u64));
    }
    let mut ids = std::collections::BTreeSet::new();
    for equation in equations {
        if !ids.insert(equation.id()) {
            return Err(invalid("duplicate mechanism identity"));
        }
        match equation {
            SystemEquation::Builtin { equation } => equation.validate(n)?,
            SystemEquation::External { id, intercept, parameters, mechanism } => {
                if !text_ok(id) || !intercept.is_finite() {
                    return Err(invalid("external mechanism identity or intercept"));
                }
                mechanism.validate()?;
                if mechanism.reads.iter().any(|&r| r >= n) {
                    return Err(invalid(
                        "external mechanism reads a coordinate outside the system",
                    ));
                }
                if parameters.len() != mechanism.parameter_names.len()
                    || parameters.iter().any(|p| !p.is_finite())
                {
                    return Err(ExternalMechanismRefusal::declaration(
                        "parameter values must match the declared parameter schema and be finite",
                    )
                    .into());
                }
            }
        }
    }
    Ok(())
}

fn trust_of(equations: &[SystemEquation]) -> MechanismTrust {
    if equations.iter().any(|e| matches!(e, SystemEquation::External { .. })) {
        MechanismTrust::VerifiedExternal
    } else {
        MechanismTrust::Native
    }
}

type Implementations = Vec<Option<Arc<dyn ExternalMechanismImplementation>>>;

fn resolve_implementations(
    equations: &[SystemEquation],
    resolver: Option<&dyn TrustedMechanismResolver>,
) -> Result<Implementations, ExternalMechanismRefusal> {
    equations
        .iter()
        .map(|equation| match equation {
            SystemEquation::Builtin { .. } => Ok(None),
            SystemEquation::External { mechanism, .. } => {
                let resolver =
                    resolver.ok_or_else(|| ExternalMechanismRefusal::missing(mechanism))?;
                let implementation = resolver
                    .resolve_checked(mechanism)
                    .map_err(|failure| ExternalMechanismRefusal::failed(mechanism, &failure))?
                    .ok_or_else(|| ExternalMechanismRefusal::unavailable(mechanism))?;
                verify_resolution(mechanism, implementation.as_ref())?;
                Ok(Some(implementation))
            }
        })
        .collect()
}

/// A verified evidence record for one external mechanism of a resolved system.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalMechanismEvidence {
    /// Equation identity in the system.
    pub equation: String,
    /// Versioned implementation id.
    pub implementation_id: String,
    /// Implementation version.
    pub version: String,
    /// Content digest the resolved implementation was verified against.
    pub content_digest: String,
    /// Identity over all declarations.
    pub identity: String,
    /// Always `verified_external`.
    pub trust: MechanismTrust,
    /// `analytic`, `finite_difference_forward` or `finite_difference_central`.
    pub derivative_basis: String,
    /// `declared` when global Lipschitz bounds were declared, else `absent`.
    pub lipschitz_basis: String,
}

struct ResolvedEquation {
    id: String,
    reads: Vec<usize>,
    n: usize,
    kind: ResolvedKind,
}
#[allow(
    clippy::large_enum_variant,
    reason = "the equation representation preserves its established owned API"
)]
enum ResolvedKind {
    Builtin(Equation),
    External {
        intercept: f64,
        parameters: Vec<f64>,
        spec: ExternalMechanismSpec,
        implementation: Arc<dyn ExternalMechanismImplementation>,
    },
}
impl ResolvedEquation {
    fn build(
        equation: &SystemEquation,
        implementation: Option<Arc<dyn ExternalMechanismImplementation>>,
        n: usize,
    ) -> Result<Self, EquilibriumError> {
        Ok(match (equation, implementation) {
            (SystemEquation::Builtin { equation }, _) => Self {
                id: equation.id.clone(),
                reads: equation.reads()?,
                n,
                kind: ResolvedKind::Builtin(equation.clone()),
            },
            (
                SystemEquation::External { id, intercept, parameters, mechanism },
                Some(implementation),
            ) => Self {
                id: id.clone(),
                reads: mechanism.reads.clone(),
                n,
                kind: ResolvedKind::External {
                    intercept: *intercept,
                    parameters: parameters.clone(),
                    spec: mechanism.clone(),
                    implementation,
                },
            },
            (SystemEquation::External { .. }, None) => {
                return Err(invalid("external mechanism without a resolved implementation"));
            }
        })
    }

    fn external_value(
        spec_inputs: &[f64],
        parameters: &[f64],
        implementation: &dyn ExternalMechanismImplementation,
    ) -> Result<f64, EquilibriumError> {
        let value = implementation.value(spec_inputs, parameters)?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(invalid("external mechanism returned a nonfinite value"))
        }
    }

    /// Value at inputs aligned to `reads`.
    fn value(&self, inputs: &[f64]) -> Result<f64, EquilibriumError> {
        match &self.kind {
            ResolvedKind::Builtin(equation) => {
                let mut state = vec![0.0; self.n];
                for (&r, &x) in self.reads.iter().zip(inputs) {
                    state[r] = x;
                }
                equation.structural_value(&state)
            }
            ResolvedKind::External { intercept, parameters, implementation, .. } => {
                Ok(intercept + Self::external_value(inputs, parameters, implementation.as_ref())?)
            }
        }
    }

    /// Value and derivatives with respect to the inputs, aligned to `reads`.
    fn evaluate(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), EquilibriumError> {
        if inputs.len() != self.reads.len() || inputs.iter().any(|v| !v.is_finite()) {
            return Err(invalid("external mechanism inputs"));
        }
        match &self.kind {
            ResolvedKind::Builtin(equation) => {
                let mut state = vec![0.0; self.n];
                for (&r, &x) in self.reads.iter().zip(inputs) {
                    state[r] = x;
                }
                let value = equation.structural_value(&state)?;
                let full = equation.input_derivatives(&state)?;
                Ok((value, self.reads.iter().map(|&r| full[r]).collect()))
            }
            ResolvedKind::External { intercept, parameters, spec, implementation } => {
                let value = Self::external_value(inputs, parameters, implementation.as_ref())?;
                let gradient = match spec.derivative {
                    DerivativeDeclaration::Analytic => {
                        let g = implementation.input_gradient(inputs, parameters)?.ok_or_else(
                            || {
                                EquilibriumError::from(ExternalMechanismRefusal::declaration(
                                    "the mechanism declares analytic derivatives but its implementation supplies no input gradient",
                                ))
                            },
                        )?;
                        if g.len() != inputs.len() || g.iter().any(|v| !v.is_finite()) {
                            return Err(invalid("external analytic gradient shape or nonfinite"));
                        }
                        g
                    }
                    DerivativeDeclaration::FiniteDifference { step, scheme } => {
                        let mut g = Vec::with_capacity(inputs.len());
                        for j in 0..inputs.len() {
                            let h = step * inputs[j].abs().max(1.0);
                            let mut plus = inputs.to_vec();
                            plus[j] += h;
                            let f_plus =
                                Self::external_value(&plus, parameters, implementation.as_ref())?;
                            g.push(match scheme {
                                DifferenceScheme::Forward => (f_plus - value) / h,
                                DifferenceScheme::Central => {
                                    let mut minus = inputs.to_vec();
                                    minus[j] -= h;
                                    let f_minus = Self::external_value(
                                        &minus,
                                        parameters,
                                        implementation.as_ref(),
                                    )?;
                                    (f_plus - f_minus) / (2.0 * h)
                                }
                            });
                        }
                        g
                    }
                };
                Ok((intercept + value, gradient))
            }
        }
    }

    /// Global derivative bounds over all `n` coordinates; infinity where undeclared.
    fn bounds(&self) -> Result<Vec<f64>, EquilibriumError> {
        match &self.kind {
            ResolvedKind::Builtin(equation) => equation.derivative_bounds(),
            ResolvedKind::External { spec, .. } => {
                let mut out = vec![0.0; self.n];
                for (k, &r) in self.reads.iter().enumerate() {
                    out[r] = spec.input_lipschitz.as_ref().map_or(f64::INFINITY, |b| b[k]);
                }
                Ok(out)
            }
        }
    }
}

struct Adapter {
    equation: Arc<ResolvedEquation>,
}
impl LocalCyclicMechanism for Adapter {
    fn id(&self) -> &str {
        &self.equation.id
    }
    fn reads(&self) -> &[usize] {
        &self.equation.reads
    }
    fn evaluate(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), EquilibriumError> {
        self.equation.evaluate(inputs)
    }
}

fn evidence_of(equations: &[SystemEquation]) -> Vec<ExternalMechanismEvidence> {
    equations
        .iter()
        .filter_map(|e| match e {
            SystemEquation::External { id, mechanism, .. } => Some(ExternalMechanismEvidence {
                equation: id.clone(),
                implementation_id: mechanism.implementation_id.clone(),
                version: mechanism.version.clone(),
                content_digest: mechanism.content_digest.clone(),
                identity: mechanism.identity(),
                trust: MechanismTrust::VerifiedExternal,
                derivative_basis: match mechanism.derivative {
                    DerivativeDeclaration::Analytic => "analytic".into(),
                    DerivativeDeclaration::FiniteDifference { scheme, .. } => match scheme {
                        DifferenceScheme::Forward => "finite_difference_forward".into(),
                        DifferenceScheme::Central => "finite_difference_central".into(),
                    },
                },
                lipschitz_basis: if mechanism.input_lipschitz.is_some() {
                    "declared".into()
                } else {
                    "absent".into()
                },
            }),
            SystemEquation::Builtin { .. } => None,
        })
        .collect()
}

/// Likelihood of regime rows split into the data-fit and Jacobian terms.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodTerms {
    /// Gaussian disturbance term `-0.5 (z'z + log|S| + k log 2 pi)`.
    pub data: f64,
    /// `log |det(I - J_FF)|` summed over rows.
    pub log_determinant: f64,
}
impl LikelihoodTerms {
    /// Joint equilibrium log density.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.data + self.log_determinant
    }
}

/// An additive-noise feedback system whose mechanisms are built-in or registered external.
/// Serializable and portable: it records identities, never code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalFeedbackSystem {
    /// One equation per variable.
    pub equations: Vec<SystemEquation>,
    /// Joint zero-mean Gaussian disturbance covariance, row-major.
    pub noise_covariance: Vec<f64>,
}

impl ExternalFeedbackSystem {
    /// Number of endogenous variables.
    #[must_use]
    pub fn order(&self) -> usize {
        self.equations.len()
    }
    /// Validate shapes, identities, declarations and the disturbance covariance.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        validate_equations(&self.equations)?;
        let n = self.order();
        if self.noise_covariance.len() != n * n {
            return Err(invalid("model order/covariance shape"));
        }
        cholesky(&self.noise_covariance, n)?;
        Ok(())
    }
    /// `verified_external` when any mechanism is external; native only when none is.
    #[must_use]
    pub fn trust(&self) -> MechanismTrust {
        trust_of(&self.equations)
    }
    /// The recorded identities of every external mechanism.
    #[must_use]
    pub fn external_specs(&self) -> Vec<(&str, &ExternalMechanismSpec)> {
        self.equations
            .iter()
            .filter_map(|e| match e {
                SystemEquation::External { id, mechanism, .. } => Some((id.as_str(), mechanism)),
                SystemEquation::Builtin { .. } => None,
            })
            .collect()
    }
    /// Current values of fitted scalars.
    pub fn parameter_values(
        &self,
        targets: &[ExternalFitParameter],
    ) -> Result<Vec<f64>, EquilibriumError> {
        if targets.is_empty() {
            return Err(invalid("parameter count"));
        }
        let mut out = Vec::with_capacity(targets.len());
        for (k, &target) in targets.iter().enumerate() {
            if targets[..k].contains(&target) {
                return Err(invalid("duplicate parameter"));
            }
            let equation =
                self.equations.get(target.equation()).ok_or_else(|| invalid("equation index"))?;
            out.push(match (target, equation) {
                (
                    ExternalFitParameter::Intercept { .. },
                    SystemEquation::External { intercept, .. },
                ) => *intercept,
                (ExternalFitParameter::Intercept { .. }, SystemEquation::Builtin { equation }) => {
                    equation.intercept
                }
                (
                    ExternalFitParameter::Parameter { parameter, .. },
                    SystemEquation::External { parameters, .. },
                ) => *parameters.get(parameter).ok_or_else(|| invalid("parameter index"))?,
                _ => return Err(invalid("parameter target is not a fitted scalar of this system")),
            });
        }
        Ok(out)
    }
    /// A copy with the given fitted scalars replaced.
    pub fn with_parameters(
        &self,
        targets: &[ExternalFitParameter],
        values: &[f64],
    ) -> Result<Self, EquilibriumError> {
        self.parameter_values(targets)?;
        if values.len() != targets.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("parameter vector"));
        }
        let mut out = self.clone();
        for (&target, &value) in targets.iter().zip(values) {
            match (target, &mut out.equations[target.equation()]) {
                (
                    ExternalFitParameter::Intercept { .. },
                    SystemEquation::External { intercept, .. },
                ) => *intercept = value,
                (ExternalFitParameter::Intercept { .. }, SystemEquation::Builtin { equation }) => {
                    equation.intercept = value
                }
                (
                    ExternalFitParameter::Parameter { parameter, .. },
                    SystemEquation::External { parameters, .. },
                ) => parameters[parameter] = value,
                _ => return Err(invalid("parameter target")),
            }
        }
        Ok(out)
    }

    /// Resolve and verify every external mechanism. Without an external mechanism no
    /// resolver is needed; with one, a missing resolver, a missing implementation or any
    /// difference from the record is a typed refusal and nothing runs.
    pub fn resolve(
        &self,
        resolver: Option<&dyn TrustedMechanismResolver>,
    ) -> Result<ResolvedFeedbackSystem, EquilibriumError> {
        self.validate()?;
        let implementations = resolve_implementations(&self.equations, resolver)?;
        ResolvedFeedbackSystem::from_parts(self.clone(), implementations)
    }
}

/// A feedback system whose external mechanisms were resolved and verified.
pub struct ResolvedFeedbackSystem {
    system: ExternalFeedbackSystem,
    implementations: Implementations,
    equations: Vec<Arc<ResolvedEquation>>,
    local: LocalCyclicModel,
}

fn external_solver_resources(
    n: usize,
    free_count: usize,
    jacobian_nnz: usize,
    blocks: &[Vec<usize>],
    policy: SolvePolicy,
) -> Result<(usize, u64), EquilibriumError> {
    if jacobian_nnz > policy.max_nnz {
        return Err(
            SolverBudgetCode::JacobianNnz.refusal(jacobian_nnz as u64, policy.max_nnz as u64)
        );
    }
    let largest_block = blocks.iter().map(Vec::len).max().unwrap_or(0);
    if largest_block > policy.max_block_size {
        return Err(
            SolverBudgetCode::BlockSize.refusal(largest_block as u64, policy.max_block_size as u64)
        );
    }
    let fill = policy
        .max_factor_fill
        .min(jacobian_nnz.saturating_mul(8).saturating_add(64 * largest_block));
    let bytes = u64::try_from(
        16usize
            .saturating_mul(n)
            .saturating_mul(n)
            .saturating_add(128 * n)
            .saturating_add(jacobian_nnz.saturating_mul(32))
            .saturating_add((free_count + 1).saturating_mul(16))
            .saturating_add(fill.saturating_mul(24)),
    )
    .map_err(|_| invalid("resource size"))?;
    if bytes > policy.max_bytes {
        return Err(SolverBudgetCode::Memory.refusal(bytes, policy.max_bytes));
    }
    Ok((largest_block, bytes))
}

impl ResolvedFeedbackSystem {
    fn from_parts(
        system: ExternalFeedbackSystem,
        implementations: Implementations,
    ) -> Result<Self, EquilibriumError> {
        let n = system.order();
        let equations = system
            .equations
            .iter()
            .zip(&implementations)
            .map(|(e, i)| ResolvedEquation::build(e, i.clone(), n).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let local = LocalCyclicModel {
            mechanisms: equations
                .iter()
                .map(|e| {
                    Box::new(Adapter { equation: Arc::clone(e) }) as Box<dyn LocalCyclicMechanism>
                })
                .collect(),
        };
        Ok(Self { system, implementations, equations, local })
    }

    /// The portable system this resolution verified.
    #[must_use]
    pub fn system(&self) -> &ExternalFeedbackSystem {
        &self.system
    }
    /// `verified_external` when any mechanism is external; there is no route to native.
    #[must_use]
    pub fn trust(&self) -> MechanismTrust {
        self.system.trust()
    }
    /// Verified evidence of each external mechanism.
    #[must_use]
    pub fn evidence(&self) -> Vec<ExternalMechanismEvidence> {
        evidence_of(&self.system.equations)
    }
    /// The same verified implementations with other fitted scalars.
    pub fn with_parameters(
        &self,
        targets: &[ExternalFitParameter],
        values: &[f64],
    ) -> Result<Self, EquilibriumError> {
        let system = self.system.with_parameters(targets, values)?;
        Self::from_parts(system, self.implementations.clone())
    }

    /// Structural means at one state; joint noise is added by the execution.
    pub fn mechanism_values(&self, state: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        if state.len() != self.system.order() || state.iter().any(|v| !v.is_finite()) {
            return Err(invalid("state dimension or nonfinite"));
        }
        self.equations
            .iter()
            .map(|e| {
                let inputs: Vec<f64> = e.reads.iter().map(|&r| state[r]).collect();
                e.value(&inputs)
            })
            .collect()
    }

    /// Post-clamp SCC schedule.
    pub fn blocks(&self, clamps: &[Clamp]) -> Result<Vec<Vec<usize>>, EquilibriumError> {
        Ok(self.local.inspect(clamps)?.blocks)
    }

    /// Largest block infinity-norm bound of the free system from declared global
    /// derivative bounds; infinite when a bound is undeclared. Below one proves a
    /// contraction on R^k for every disturbance, conditional on the declarations.
    pub fn contraction_bound(&self, clamps: &[Clamp]) -> Result<f64, EquilibriumError> {
        let blocks = self.blocks(clamps)?;
        let bounds: Vec<Vec<f64>> =
            self.equations.iter().map(|e| e.bounds()).collect::<Result<_, _>>()?;
        let mut worst = 0.0_f64;
        for block in &blocks {
            for &i in block {
                let sum: f64 = block.iter().map(|&j| bounds[i][j]).sum();
                worst = worst.max(sum);
            }
        }
        Ok(worst)
    }

    fn reads_of(&self) -> Vec<&[usize]> {
        self.equations.iter().map(|e| e.reads.as_slice()).collect()
    }

    fn residual(
        &self,
        state: &[f64],
        noise: &[f64],
        rows: &[usize],
    ) -> Result<Vec<f64>, EquilibriumError> {
        rows.iter()
            .map(|&i| {
                let e = &self.equations[i];
                let inputs: Vec<f64> = e.reads.iter().map(|&r| state[r]).collect();
                Ok(state[i] - e.value(&inputs)? - noise[i])
            })
            .collect()
    }

    /// `I - df/dx` over `rows_of`, local to that set, from the verified mechanisms' derivatives.
    fn sparse_jacobian(
        &self,
        state: &[f64],
        rows_of: &[usize],
    ) -> Result<CsrMatrix, EquilibriumError> {
        let local = local_positions(rows_of);
        let mut rows = Vec::with_capacity(rows_of.len());
        for (a, &i) in rows_of.iter().enumerate() {
            let e = &self.equations[i];
            let inputs: Vec<f64> = e.reads.iter().map(|&r| state[r]).collect();
            let (_, gradient) = e.evaluate(&inputs)?;
            let mut row = vec![(a, 1.0)];
            for (g, &read) in e.reads.iter().enumerate() {
                if let Some(&b) = local.get(&read) {
                    row.push((b, -gradient[g]));
                }
            }
            rows.push(row);
        }
        CsrMatrix::from_rows(rows_of.len(), rows)
    }

    /// Declared global derivative-bound matrix of a block.
    fn block_bound_matrix(&self, block: &[usize]) -> Result<CsrMatrix, EquilibriumError> {
        let local = local_positions(block);
        let mut rows = Vec::with_capacity(block.len());
        for &i in block {
            let bounds = self.equations[i].bounds()?;
            rows.push(
                self.equations[i]
                    .reads
                    .iter()
                    .filter_map(|j| local.get(j).map(|&b| (b, bounds[*j])))
                    .collect(),
            );
        }
        CsrMatrix::from_rows(block.len(), rows)
    }

    fn record_jacobian_condition(
        &self,
        values: &[f64],
        free: &[usize],
        policy: SolvePolicy,
        all_contractive: bool,
        evidence: &mut SolveEvidence,
        mut tick: impl FnMut() -> Result<(), EquilibriumError>,
    ) -> Result<Option<f64>, EquilibriumError> {
        // Free-Jacobian conditioning at the returned state.
        let mut jacobian_condition_estimate = None;
        let mut factored = "not_applicable";
        if !free.is_empty() {
            let jacobian = self.sparse_jacobian(values, free)?;
            evidence.jacobian_diagonal_dominance_margin =
                jacobian.diagonal_dominance_margin().filter(|m| m.is_finite());
            let factors = match SparseLu::factor(&jacobian, policy.max_factor_fill, &mut tick) {
                Ok(lu) => {
                    SparseLu::factor(&jacobian.transpose(), policy.max_factor_fill, &mut tick)
                        .map(|lut| (lu, lut))
                }
                Err(error) => Err(error),
            };
            match factors {
                Ok((lu, lut)) => {
                    evidence.factorizations += 2;
                    evidence.peak_factor_fill =
                        evidence.peak_factor_fill.max(lu.fill()).max(lut.fill());
                    jacobian_condition_estimate = infinity_condition_estimate(&jacobian, &lu, &lut);
                    factored = "factorized";
                }
                Err(FactorError::Singular) => factored = "singular",
                Err(FactorError::NonFinite) => factored = "nonfinite",
                Err(FactorError::FillBudget(_)) => factored = "factor_fill_budget",
                Err(FactorError::Interrupted(error)) => return Err(error),
            }
        }
        evidence.jacobian_nonsingular_basis = if free.is_empty() {
            "not_applicable".into()
        } else if factored != "factorized" {
            factored.into()
        } else if all_contractive {
            "contraction".into()
        } else if evidence.jacobian_diagonal_dominance_margin.is_some_and(|m| m > 0.0) {
            "strict_diagonal_dominance".into()
        } else {
            "factorization".into()
        };
        Ok(jacobian_condition_estimate)
    }

    fn certify_block_contraction(
        &self,
        block: &[usize],
        reads: &[&[usize]],
        policy: SolvePolicy,
        evidence: &mut SolveEvidence,
    ) -> Result<bool, EquilibriumError> {
        // Positive summation error is bounded conservatively before comparing to one.
        let inflation = 1.0 + (block.len() + 1) as f64 * f64::EPSILON;
        let matrix = self.block_bound_matrix(block)?;
        let row_sum = matrix.infinity_norm() * inflation;
        let (spectral, spectral_iterations) = if row_sum < 1.0 || block.len() == 1 {
            (row_sum, 0)
        } else {
            let found = nonnegative_spectral_upper_bound(&matrix, policy.spectral_iterations);
            (found.upper.min(row_sum), found.iterations)
        };
        let bound = row_sum.min(spectral);
        let retained = |x: f64| if x.is_finite() { x } else { f64::MAX };
        evidence.row_sum_bound = evidence.row_sum_bound.max(retained(row_sum));
        evidence.spectral_bound = evidence.spectral_bound.max(retained(spectral));
        evidence.spectral_iterations += spectral_iterations;
        evidence.contraction_bound = evidence.contraction_bound.max(retained(bound));
        if block.len() > 1 || reads[block[0]].contains(&block[0]) {
            evidence.cyclic_block_count += 1;
        }
        let contraction = bound < 1.0;
        if policy.require_contraction && !contraction {
            return Err(EquilibriumError::Uncertified { bound });
        }
        Ok(contraction)
    }

    /// Numerical SCC solve under the same budgets, admission, cancellation and evidence as the
    /// built-in sparse solver: Jacobian nonzeros, block size, memory, factor fill, iteration
    /// count and wall clock. Existence and uniqueness are marked only when the declared global
    /// bounds give a block contraction (row-sum or spectral bound) below one; otherwise the
    /// root is numerical and, with `require_contraction`, refused.
    #[allow(
        clippy::too_many_lines,
        reason = "one budgeted block solve retains admission, certificates and evidence together"
    )]
    pub fn solve(
        &self,
        noise: &[f64],
        clamps: &[Clamp],
        initial: &[f64],
        policy: SolvePolicy,
        ctx: &ExecutionContext,
    ) -> Result<EquilibriumResult, EquilibriumError> {
        policy.validate()?;
        let n = self.system.order();
        for v in [noise, initial] {
            if v.len() != n || v.iter().any(|x| !x.is_finite()) {
                return Err(invalid("state/noise dimension or nonfinite"));
            }
        }
        let blocks = self.blocks(clamps)?;
        let mut is_free = vec![true; n];
        for c in clamps {
            if !c.value.is_finite() {
                return Err(invalid("clamp value"));
            }
            is_free[c.variable] = false;
        }
        let free: Vec<usize> = (0..n).filter(|&i| is_free[i]).collect();
        let reads = self.reads_of();
        let jacobian_nnz: usize = free
            .iter()
            .map(|&i| reads[i].iter().filter(|&&j| j != i && is_free[j]).count() + 1)
            .sum();
        let (largest_block, bytes) =
            external_solver_resources(n, free.len(), jacobian_nnz, &blocks, policy)?;
        let _lease = ctx
            .acquire_serial_numerical_admission(bytes)
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let started = Instant::now();
        let mut checks = 0usize;
        let tick = |checks: &mut usize| -> Result<(), EquilibriumError> {
            *checks += 1;
            check_cancel(ctx)?;
            if let Some(limit) = policy.max_wall_clock_ms {
                let elapsed = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                if elapsed >= limit {
                    return Err(SolverBudgetCode::WallClock.refusal(elapsed, limit));
                }
            }
            Ok(())
        };
        let mut evidence = SolveEvidence {
            free_variables: free.len(),
            block_count: blocks.len(),
            largest_block,
            jacobian_nnz,
            estimated_bytes: bytes,
            admitted_bytes: bytes,
            ..SolveEvidence::default()
        };
        let mut values = initial.to_vec();
        for c in clamps {
            values[c.variable] = c.value;
        }
        let max_abs = |v: &[f64]| {
            v.iter()
                .map(|x| if x.is_finite() { x.abs() } else { f64::INFINITY })
                .fold(0.0, f64::max)
        };
        let factor_failure = |error: FactorError, residual: f64| match error {
            FactorError::Singular => EquilibriumError::Singular,
            FactorError::NonFinite => EquilibriumError::Nonconvergence { residual },
            FactorError::FillBudget(required) => {
                SolverBudgetCode::FactorFill.refusal(required as u64, policy.max_factor_fill as u64)
            }
            FactorError::Interrupted(error) => error,
        };
        let mut total = 0usize;
        let mut all_contractive = true;
        for block in &blocks {
            tick(&mut checks)?;
            let contraction =
                self.certify_block_contraction(block, &reads, policy, &mut evidence)?;
            all_contractive &= contraction;
            let mut converged = false;
            for _ in 0..policy.max_iterations {
                tick(&mut checks)?;
                let residual = self.residual(&values, noise, block)?;
                let norm = max_abs(&residual);
                if !norm.is_finite() {
                    return Err(EquilibriumError::Nonconvergence { residual: norm });
                }
                if norm <= policy.tolerance {
                    converged = true;
                    break;
                }
                total += 1;
                if total > policy.max_total_iterations {
                    return Err(SolverBudgetCode::IterationBudget
                        .refusal(total as u64, policy.max_total_iterations as u64));
                }
                if contraction {
                    evidence.contraction_iterations += 1;
                    for (&i, &r) in block.iter().zip(&residual) {
                        values[i] -= r;
                    }
                } else {
                    evidence.newton_steps += 1;
                    let jac = self.sparse_jacobian(&values, block)?;
                    let lu =
                        SparseLu::factor(&jac, policy.max_factor_fill, &mut || tick(&mut checks))
                            .map_err(|e| factor_failure(e, norm))?;
                    evidence.factorizations += 1;
                    evidence.peak_factor_fill = evidence.peak_factor_fill.max(lu.fill());
                    let step = lu.solve(&residual);
                    if step.iter().any(|v| !v.is_finite()) {
                        return Err(EquilibriumError::Nonconvergence { residual: norm });
                    }
                    let old = values.clone();
                    let mut accepted = false;
                    let mut scale = 1.0;
                    for _ in 0..40 {
                        for (&i, &delta) in block.iter().zip(&step) {
                            values[i] = old[i] - scale * delta;
                        }
                        let next = max_abs(&self.residual(&values, noise, block)?);
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
            let residual = max_abs(&self.residual(&values, noise, block)?);
            if !converged && residual > policy.tolerance {
                return Err(EquilibriumError::Nonconvergence { residual });
            }
        }
        let residual = max_abs(&self.residual(&values, noise, &free)?);
        if values.iter().any(|v| !v.is_finite())
            || !residual.is_finite()
            || residual > policy.tolerance
        {
            return Err(EquilibriumError::Nonconvergence { residual });
        }
        let jacobian_condition_estimate = self.record_jacobian_condition(
            &values,
            &free,
            policy,
            all_contractive,
            &mut evidence,
            || tick(&mut checks),
        )?;
        evidence.unique_root_basis = if all_contractive { "contraction" } else { "none" }.into();
        if all_contractive
            && blocks.iter().any(|block| {
                block.iter().any(|&i| {
            matches!(
                &self.system.equations[i],
                SystemEquation::External { mechanism, .. } if mechanism.input_lipschitz.is_some()
            ) && reads[i].iter().any(|j| block.contains(j))
        })
            })
        {
            evidence.contraction_bound_provenance = Some("declared_external_lipschitz".into());
        }
        evidence.ill_conditioned =
            jacobian_condition_estimate.is_some_and(|k| k > policy.condition_warning);
        evidence.budget_checks = checks;
        Ok(EquilibriumResult {
            values,
            residual,
            iterations: total,
            globally_unique: all_contractive,
            existence_certified: all_contractive,
            numerical_converged: true,
            dynamic_stability: None,
            jacobian_condition_estimate,
            blocks,
            evidence,
        })
    }

    /// Exact additive abduction at a fully observed state.
    pub fn abduce(&self, factual: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        let means = self.mechanism_values(factual)?;
        Ok(factual.iter().zip(means).map(|(x, m)| x - m).collect())
    }

    /// Abduce once and reuse the disturbances in the intervened world.
    pub fn counterfactual(
        &self,
        factual: &[f64],
        clamps: &[Clamp],
        policy: SolvePolicy,
        ctx: &ExecutionContext,
    ) -> Result<EquilibriumResult, EquilibriumError> {
        let noise = self.abduce(factual)?;
        self.solve(&noise, clamps, factual, policy, ctx)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn solver_resource_limits_refuse_nnz_block_and_memory_excess() {
        let blocks = vec![vec![0, 1]];
        let mut policy = SolvePolicy::default();
        assert!(external_solver_resources(2, 2, 4, &blocks, policy).is_ok());
        policy.max_nnz = 3;
        assert!(external_solver_resources(2, 2, 4, &blocks, policy).is_err());
        policy.max_nnz = 4;
        policy.max_block_size = 1;
        assert!(external_solver_resources(2, 2, 4, &blocks, policy).is_err());
        policy.max_block_size = 2;
        policy.max_bytes = 0;
        assert!(external_solver_resources(2, 2, 4, &blocks, policy).is_err());
    }
}
