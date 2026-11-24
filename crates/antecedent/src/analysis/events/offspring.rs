//! Compiled offspring functionals of a catalog or marked Hawkes kernel.
//!
//! The direct mass on a horizon, the lifetime direct mass, the filter-ODE horizon descendants
//! and the lifetime resolvent descendants of one initiating event are functionals of the
//! conditional intensity. They are not an intervention effect. Each query compiles to a program
//! of the excitation route before any functional is evaluated.

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;

use super::hawkes::{ExcitationOffspring, HawkesParameters};
use super::marked::{MarkPrototype, MarkedOffspring, MarkedParameters};
use super::{EventError, PreparedKernel, admit, digest};
use crate::planner::lowered::EventProcessRoute::Excitation;

/// Compiled offspring functionals of one catalog source type.
pub struct PreparedExcitationOffspring {
    parameters: HawkesParameters,
    source: usize,
    horizon: f64,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

/// Compiled offspring functionals of one marked prototype.
pub struct PreparedMarkedOffspring {
    parameters: MarkedParameters,
    prototypes: Vec<MarkPrototype>,
    prototype: usize,
    horizon: f64,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

fn check_query(
    index: usize,
    count: usize,
    horizon: f64,
    message: &'static str,
) -> Result<(), EventError> {
    if index >= count || !horizon.is_finite() || horizon <= 0.0 {
        return Err(EventError::invalid(message));
    }
    Ok(())
}

impl crate::Study {
    /// Compile the offspring functionals of `source` on `[0, horizon)` before evaluating them.
    ///
    /// # Errors
    /// The parameters are not a stable catalog model, the source is outside the catalog, the
    /// horizon is not a positive finite duration, or a resource refusal.
    pub fn prepare_excitation_offspring(
        parameters: HawkesParameters,
        source: usize,
        horizon: f64,
        ctx: &ExecutionContext,
    ) -> Result<PreparedExcitationOffspring, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        parameters.validate(parameters.baseline.len())?;
        check_query(
            source,
            parameters.baseline.len(),
            horizon,
            "offspring query needs a catalog type and a positive finite horizon",
        )?;
        let lease = admit(ctx, 0, 8_192)?;
        let identity =
            digest("antecedent.events.excitation_offspring.v1", &(&parameters, source, horizon))?;
        let plan = PreparedKernel::on(Excitation, identity, "excitation_offspring")?;
        Ok(PreparedExcitationOffspring { parameters, source, horizon, plan, _lease: lease })
    }

    /// Compile the offspring functionals of one marked prototype before evaluating them.
    ///
    /// # Errors
    /// The kernel is not a stable model on these prototypes, the prototype is outside them, the
    /// horizon is not a positive finite duration, or a resource refusal.
    pub fn prepare_marked_offspring(
        parameters: MarkedParameters,
        prototypes: Vec<MarkPrototype>,
        prototype: usize,
        horizon: f64,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMarkedOffspring, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        parameters.validate(&prototypes)?;
        check_query(
            prototype,
            prototypes.len(),
            horizon,
            "marked offspring query needs a prototype and a positive finite horizon",
        )?;
        let lease = admit(ctx, 0, 8_192 + prototypes.len() as u64 * 64)?;
        let identity = digest(
            "antecedent.events.marked_offspring.v1",
            &(&parameters, &prototypes, prototype, horizon),
        )?;
        let plan = PreparedKernel::on(Excitation, identity, "marked_offspring")?;
        Ok(PreparedMarkedOffspring {
            parameters,
            prototypes,
            prototype,
            horizon,
            plan,
            _lease: lease,
        })
    }
}

impl PreparedExcitationOffspring {
    /// Inspect the compiled query without evaluating it.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Evaluate the direct, lifetime and descendant offspring of the source type.
    ///
    /// # Errors
    /// The filter ODE cannot resolve the horizon, cancellation, or an unsupported release scope.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<ExcitationOffspring, EventError> {
        let _scoped = super::scoped_execution(ctx, &self.plan)?;
        self.parameters.excitation_offspring(self.source, self.horizon)
    }
}

impl PreparedMarkedOffspring {
    /// Inspect the compiled query without evaluating it.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Evaluate the direct, lifetime and descendant offspring of the prototype.
    ///
    /// # Errors
    /// The filter ODE cannot resolve the horizon, cancellation, or an unsupported release scope.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<MarkedOffspring, EventError> {
        let _scoped = super::scoped_execution(ctx, &self.plan)?;
        self.parameters.excitation_offspring(&self.prototypes, self.prototype, self.horizon)
    }
}
