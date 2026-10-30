// SPDX-License-Identifier: MIT OR Apache-2.0
//! Local differentiable mechanisms for arbitrary feedback topology.
//! Callbacks declare their reads and supply derivatives. Their purity and
//! derivative correctness are caller obligations; numerical roots carry no
//! existence, uniqueness, stability or portability certificate.
use crate::nonlinear_equilibrium::{Clamp, EquilibriumError, EquilibriumResult, MAX_ORDER};
use antecedent_core::ExecutionContext;
use antecedent_graph::{CyclicDigraph, DenseNodeId};
use antecedent_kernels::gauss_jordan_solve;
use serde::{Deserialize, Serialize};

/// A process-local, pure differentiable mechanism with explicit input ordering.
pub trait LocalCyclicMechanism: Send + Sync {
    /// Stable scientific mechanism identity, unique within this model.
    fn id(&self) -> &str;
    /// Endogenous coordinates read, in callback input order; self-reads are allowed.
    fn reads(&self) -> &[usize];
    /// Structural value and derivatives with respect to exactly those inputs.
    fn evaluate(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), EquilibriumError>;
}
/// Bounded numerical policy. It does not supply an equilibrium selection law.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalCyclicPolicy {
    /// Absolute free-equation residual tolerance.
    pub tolerance: f64,
    /// Maximum Newton iterations per SCC.
    pub max_iterations: usize,
    /// Total callback calls across residuals, derivatives and backtracking.
    pub max_evaluations: usize,
}
impl Default for LocalCyclicPolicy {
    fn default() -> Self {
        Self { tolerance: 1e-10, max_iterations: 1000, max_evaluations: 100_000 }
    }
}
/// Inspected local execution, with its explicit portability limits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalCyclicInspection {
    /// Stable mechanism identities, in variable order.
    pub mechanisms: Vec<String>,
    /// Original input ordering for each callback.
    pub reads: Vec<Vec<usize>>,
    /// Post-clamp free SCC schedule.
    pub blocks: Vec<Vec<usize>>,
    /// Always false: arbitrary callback code is not a replayable built-in.
    pub portable: bool,
    /// Caller obligations that execution cannot verify.
    pub assumptions: Vec<String>,
}
/// A specified process-local additive-noise equilibrium model.
pub struct LocalCyclicModel {
    /// Exactly one mechanism per endogenous variable.
    pub mechanisms: Vec<Box<dyn LocalCyclicMechanism>>,
}
fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}
impl LocalCyclicModel {
    /// Validate identities, read ordering and dimensions, without executing callbacks.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        let n = self.mechanisms.len();
        if n == 0 || n > MAX_ORDER {
            return Err(invalid("local cyclic model order"));
        }
        let mut ids = std::collections::BTreeSet::new();
        for mechanism in &self.mechanisms {
            if mechanism.id().trim().is_empty()
                || mechanism.id().len() > 256
                || !ids.insert(mechanism.id())
                || mechanism
                    .reads()
                    .iter()
                    .enumerate()
                    .any(|(i, r)| *r >= n || mechanism.reads()[..i].contains(r))
            {
                return Err(invalid("local mechanism identity/reads"));
            }
        }
        Ok(())
    }
    fn free(&self, clamps: &[Clamp]) -> Result<Vec<usize>, EquilibriumError> {
        let n = self.mechanisms.len();
        let mut clamped = vec![false; n];
        for clamp in clamps {
            if clamp.variable >= n || !clamp.value.is_finite() || clamped[clamp.variable] {
                return Err(invalid("local clamp"));
            }
            clamped[clamp.variable] = true;
        }
        Ok((0..n).filter(|&i| !clamped[i]).collect())
    }
    /// Compile original callback dependencies using the shared SCC graph engine.
    pub fn inspect(&self, clamps: &[Clamp]) -> Result<LocalCyclicInspection, EquilibriumError> {
        self.validate()?;
        let free = self.free(clamps)?;
        let count = u32::try_from(free.len()).map_err(|_| invalid("local graph order"))?;
        let mut graph = CyclicDigraph::with_variables(count);
        for (to, &i) in free.iter().enumerate() {
            for (from, &j) in free.iter().enumerate() {
                if i != j && self.mechanisms[i].reads().contains(&j) {
                    graph
                        .insert_directed(
                            DenseNodeId::from_raw(
                                u32::try_from(from).map_err(|_| invalid("node"))?,
                            ),
                            DenseNodeId::from_raw(u32::try_from(to).map_err(|_| invalid("node"))?),
                        )
                        .map_err(|e| EquilibriumError::Invalid(e.to_string()))?;
                }
            }
        }
        let blocks = graph
            .condensation()
            .components()
            .iter()
            .map(|component| component.iter().map(|node| free[node.raw() as usize]).collect())
            .collect();
        Ok(LocalCyclicInspection {
            mechanisms: self.mechanisms.iter().map(|m| m.id().to_owned()).collect(),
            reads: self.mechanisms.iter().map(|m| m.reads().to_vec()).collect(),
            blocks,
            portable: false,
            assumptions: vec![
                "callbacks are pure deterministic functions of declared inputs".into(),
                "supplied local derivatives are correct at evaluated states".into(),
                "a numerical root is not a uniqueness or equilibrium-selection certificate".into(),
            ],
        })
    }
    fn evaluate(
        &self,
        i: usize,
        state: &[f64],
        evaluations: &mut usize,
        limit: usize,
        ctx: &ExecutionContext,
    ) -> Result<(f64, Vec<f64>), EquilibriumError> {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        if *evaluations >= limit {
            return Err(EquilibriumError::Resource("local callback evaluation budget".into()));
        }
        *evaluations += 1;
        let mechanism = &self.mechanisms[i];
        let inputs: Vec<_> = mechanism.reads().iter().map(|&j| state[j]).collect();
        let (value, gradient) = mechanism.evaluate(&inputs)?;
        if !value.is_finite()
            || gradient.len() != inputs.len()
            || gradient.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "local callback returned nonfinite value or malformed derivatives",
            ));
        }
        Ok((value, gradient))
    }
    fn residual(
        &self,
        state: &[f64],
        noise: &[f64],
        block: &[usize],
        evaluations: &mut usize,
        policy: LocalCyclicPolicy,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<f64>, Vec<f64>), EquilibriumError> {
        let mut residual = Vec::with_capacity(block.len());
        let mut jacobian = Vec::with_capacity(block.len() * block.len());
        for &i in block {
            let (value, gradient) =
                self.evaluate(i, state, evaluations, policy.max_evaluations, ctx)?;
            residual.push(state[i] - value - noise[i]);
            for &j in block {
                let derivative = self.mechanisms[i]
                    .reads()
                    .iter()
                    .position(|&v| v == j)
                    .map_or(0.0, |k| gradient[k]);
                jacobian.push((if i == j { 1.0 } else { 0.0 }) - derivative);
            }
        }
        Ok((residual, jacobian))
    }
    /// Damped Newton execution with bounded residual checks and no uniqueness claim.
    pub fn solve(
        &self,
        noise: &[f64],
        clamps: &[Clamp],
        initial: &[f64],
        policy: LocalCyclicPolicy,
        ctx: &ExecutionContext,
    ) -> Result<EquilibriumResult, EquilibriumError> {
        let inspection = self.inspect(clamps)?;
        let n = self.mechanisms.len();
        if noise.len() != n
            || initial.len() != n
            || noise.iter().chain(initial).any(|v| !v.is_finite())
            || !policy.tolerance.is_finite()
            || policy.tolerance <= 0.0
            || policy.max_iterations == 0
            || policy.max_iterations > 100_000
            || policy.max_evaluations == 0
            || policy.max_evaluations > 1_000_000
        {
            return Err(invalid("local solve inputs/policy"));
        }
        let bytes = u64::try_from(64 * n * n + 256 * n).map_err(|_| invalid("local resources"))?;
        let _lease = ctx
            .acquire_serial_numerical_admission(bytes)
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let mut values = initial.to_vec();
        for clamp in clamps {
            values[clamp.variable] = clamp.value;
        }
        let mut evaluations = 0;
        let mut iterations = 0;
        for block in &inspection.blocks {
            let mut converged = false;
            for _ in 0..policy.max_iterations {
                let (residual, mut jacobian) =
                    self.residual(&values, noise, block, &mut evaluations, policy, ctx)?;
                let norm = max_abs(&residual);
                if norm <= policy.tolerance {
                    converged = true;
                    break;
                }
                if !norm.is_finite() {
                    return Err(EquilibriumError::Nonconvergence { residual: norm });
                }
                let mut direction: Vec<_> = residual.iter().map(|r| -r).collect();
                if !gauss_jordan_solve(&mut jacobian, &mut direction, block.len(), 1) {
                    return Err(EquilibriumError::Singular);
                }
                let mut step = 1.0;
                let mut accepted = None;
                for _ in 0..48 {
                    let mut next = values.clone();
                    for (&i, &delta) in block.iter().zip(&direction) {
                        next[i] += step * delta;
                    }
                    let (candidate, _) =
                        self.residual(&next, noise, block, &mut evaluations, policy, ctx)?;
                    if max_abs(&candidate) < norm {
                        accepted = Some(next);
                        break;
                    }
                    step *= 0.5;
                }
                values = accepted.ok_or(EquilibriumError::Nonconvergence { residual: norm })?;
                iterations += 1;
            }
            if !converged {
                let (residual, _) =
                    self.residual(&values, noise, block, &mut evaluations, policy, ctx)?;
                if max_abs(&residual) > policy.tolerance {
                    return Err(EquilibriumError::Nonconvergence { residual: max_abs(&residual) });
                }
            }
        }
        let free = self.free(clamps)?;
        let (residual, _) = self.residual(&values, noise, &free, &mut evaluations, policy, ctx)?;
        let norm = max_abs(&residual);
        if norm > policy.tolerance || !norm.is_finite() {
            return Err(EquilibriumError::Nonconvergence { residual: norm });
        }
        Ok(EquilibriumResult {
            values,
            residual: norm,
            iterations,
            globally_unique: false,
            existence_certified: false,
            numerical_converged: true,
            dynamic_stability: None,
            jacobian_condition_estimate: None,
            blocks: inspection.blocks,
            evidence: crate::sparse_equilibrium::SolveEvidence::default(),
        })
    }
    /// Exact additive abduction under the specified callbacks, without learning them.
    pub fn abduce(
        &self,
        factual: &[f64],
        policy: LocalCyclicPolicy,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, EquilibriumError> {
        self.validate()?;
        if factual.len() != self.mechanisms.len()
            || factual.iter().any(|v| !v.is_finite())
            || policy.max_evaluations == 0
            || policy.max_evaluations > 1_000_000
        {
            return Err(invalid("local factual state/policy"));
        }
        let mut evaluations = 0;
        factual
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                self.evaluate(i, factual, &mut evaluations, policy.max_evaluations, ctx)
                    .map(|(value, _)| x - value)
            })
            .collect()
    }
    /// Refuse replay export: stable callback names do not serialize executable code.
    pub fn export(&self) -> Result<Vec<u8>, EquilibriumError> {
        Err(invalid(
            "local callback code is not portable; use serializable built-in expressions for independent artifact replay",
        ))
    }
}
fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|v| if v.is_finite() { v.abs() } else { f64::INFINITY }).fold(0.0, f64::max)
}

/// Adapter for an existing additive `DynamicMechanism` and its declared local derivatives.
/// Evaluates the existing column API at zero structural noise. The declaration
/// that the original mechanism is additive must be checked by its caller.
pub struct DynamicCyclicAdapter<M, G> {
    /// Stable mechanism identity.
    pub id: String,
    /// Endogenous parent order used by the existing column mechanism.
    pub reads: Vec<usize>,
    /// Existing mechanism implementation, reused without conversion or refitting.
    pub mechanism: M,
    /// Correct derivatives of its zero-noise value in declared input order.
    pub derivatives: G,
}
impl<M, G> LocalCyclicMechanism for DynamicCyclicAdapter<M, G>
where
    M: crate::compile::DynamicMechanism,
    G: Fn(&[f64]) -> Result<Vec<f64>, EquilibriumError> + Send + Sync,
{
    fn id(&self) -> &str {
        &self.id
    }
    fn reads(&self) -> &[usize] {
        &self.reads
    }
    fn evaluate(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), EquilibriumError> {
        let mut output = [0.0];
        self.mechanism
            .evaluate_column(
                crate::batch::ParentBatch { n_rows: 1, n_parents: inputs.len(), values: inputs },
                &[0.0],
                &mut output,
                &mut crate::batch::MechanismWorkspace::default(),
            )
            .map_err(|e| EquilibriumError::Invalid(e.to_string()))?;
        Ok((output[0], (self.derivatives)(inputs)?))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Mechanism {
        id: String,
        reads: Vec<usize>,
        intercept: f64,
        weights: Vec<f64>,
    }
    impl LocalCyclicMechanism for Mechanism {
        fn id(&self) -> &str {
            &self.id
        }
        fn reads(&self) -> &[usize] {
            &self.reads
        }
        fn evaluate(&self, x: &[f64]) -> Result<(f64, Vec<f64>), EquilibriumError> {
            Ok((
                self.intercept + x.iter().zip(&self.weights).map(|(x, w)| x * w).sum::<f64>(),
                self.weights.clone(),
            ))
        }
    }
    #[test]
    fn local_feedback_preserves_self_reads_abduction_and_numerical_only_status() {
        let model = LocalCyclicModel {
            mechanisms: vec![
                Box::new(Mechanism {
                    id: "a".into(),
                    reads: vec![0, 1],
                    intercept: 1.0,
                    weights: vec![0.2, 0.4],
                }),
                Box::new(Mechanism {
                    id: "b".into(),
                    reads: vec![0],
                    intercept: 0.0,
                    weights: vec![0.5],
                }),
            ],
        };
        let ctx = ExecutionContext::for_tests(1);
        let policy = LocalCyclicPolicy::default();
        let result = model.solve(&[0.0, 0.0], &[], &[0.0, 0.0], policy, &ctx).unwrap();
        assert!((result.values[0] - 1.0 / 0.6).abs() < 1e-10);
        assert!(
            !result.globally_unique && !result.existence_certified && result.numerical_converged
        );
        let noise = model.abduce(&[0.3, 0.7], policy, &ctx).unwrap();
        let cf = model
            .solve(&noise, &[Clamp { variable: 0, value: 2.0 }], &[0.3, 0.7], policy, &ctx)
            .unwrap();
        assert!((cf.values[1] - 1.55).abs() < 1e-10);
        assert!(model.export().is_err());
        assert!(
            model
                .solve(
                    &[0.0, 0.0],
                    &[],
                    &[0.0, 0.0],
                    LocalCyclicPolicy { max_evaluations: 1, ..policy },
                    &ctx
                )
                .is_err()
        );
        ctx.cancellation.cancel();
        assert!(matches!(
            model.solve(&[0.0, 0.0], &[], &[0.0, 0.0], policy, &ctx),
            Err(EquilibriumError::Resource(_))
        ));
    }
}
