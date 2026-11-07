//! Continuous-time systems driven by registered external mechanisms.

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

use crate::continuous_cyclic::{OdePolicy, OdeTrajectory, TimedClamp};
use crate::nonlinear_equilibrium::EquilibriumError;

use super::{
    ExternalMechanismEvidence, MechanismTrust, ResolvedEquation, SystemEquation,
    TrustedMechanismResolver, evidence_of, invalid, resolve_implementations, trust_of,
    validate_equations,
};

/// Continuous-time drift `dx_i/dt = f_i(x) - decay_i x_i` with built-in or registered
/// external mechanisms. Self-dependence is allowed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalOdeSystem {
    /// One drift mechanism per variable.
    pub equations: Vec<SystemEquation>,
    /// Additional linear decay per variable.
    pub decay: Vec<f64>,
}

/// A finite-horizon external-mechanism trajectory and its standing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalOdeTrajectory {
    /// States and step counts.
    pub trajectory: OdeTrajectory,
    /// Global Lipschitz upper bound of the drift (infinity-norm) from declared bounds; absent
    /// when any bound is undeclared. A declaration, not a measurement.
    pub drift_lipschitz: Option<f64>,
    /// `verified_external` when any mechanism is external.
    pub trust: MechanismTrust,
    /// Verified evidence of each external mechanism.
    pub mechanisms: Vec<ExternalMechanismEvidence>,
}

/// An ODE system with verified implementations.
pub struct ResolvedOdeSystem {
    system: ExternalOdeSystem,
    equations: Vec<Arc<ResolvedEquation>>,
}

impl ExternalOdeSystem {
    /// Validate shapes, identities and declarations.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        let n = self.equations.len();
        if n == 0 || n > 128 || self.decay.len() != n || self.decay.iter().any(|x| !x.is_finite()) {
            return Err(invalid("ODE shape/decay"));
        }
        validate_equations(&self.equations)
    }
    /// Resolve and verify every external mechanism, as for a static system.
    pub fn resolve(
        &self,
        resolver: Option<&dyn TrustedMechanismResolver>,
    ) -> Result<ResolvedOdeSystem, EquilibriumError> {
        self.validate()?;
        let implementations = resolve_implementations(&self.equations, resolver)?;
        let n = self.equations.len();
        let equations = self
            .equations
            .iter()
            .zip(implementations)
            .map(|(e, i)| ResolvedEquation::build(e, i, n).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ResolvedOdeSystem { system: self.clone(), equations })
    }
}

impl ResolvedOdeSystem {
    /// Declared drift at `state`.
    pub fn drift(&self, state: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        if state.len() != self.equations.len() || state.iter().any(|v| !v.is_finite()) {
            return Err(invalid("ODE state"));
        }
        self.equations
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let inputs: Vec<f64> = e.reads.iter().map(|&r| state[r]).collect();
                e.value(&inputs).map(|v| v - self.system.decay[i] * state[i])
            })
            .collect()
    }

    /// Infinity-norm global Lipschitz bound of the drift from declared bounds.
    pub fn drift_lipschitz(&self) -> Result<Option<f64>, EquilibriumError> {
        let mut worst = 0.0_f64;
        for (i, e) in self.equations.iter().enumerate() {
            let mut bounds = e.bounds()?;
            bounds[i] += self.system.decay[i].abs();
            let sum: f64 = bounds.iter().sum();
            if !sum.is_finite() {
                return Ok(None);
            }
            worst = worst.max(sum);
        }
        Ok(Some(worst))
    }

    /// Dormand-Prince trajectory with timed hard clamps. A declared Lipschitz bound caps the
    /// first proposed step at its reciprocal; error control then governs every step.
    pub fn trajectory(
        &self,
        initial: &[f64],
        times: &[f64],
        schedule: &[TimedClamp],
        policy: OdePolicy,
        ctx: &ExecutionContext,
    ) -> Result<ExternalOdeTrajectory, EquilibriumError> {
        let lipschitz = self.drift_lipschitz()?;
        let mut policy = policy;
        if let Some(l) = lipschitz.filter(|l| *l > 0.0) {
            policy.initial_step = policy.initial_step.min(1.0 / l);
        }
        let trajectory =
            crate::continuous_cyclic::integrate(initial, times, schedule, policy, ctx, |state| {
                self.drift(state)
            })?;
        Ok(ExternalOdeTrajectory {
            trajectory,
            drift_lipschitz: lipschitz,
            trust: trust_of(&self.system.equations),
            mechanisms: evidence_of(&self.system.equations),
        })
    }
}
