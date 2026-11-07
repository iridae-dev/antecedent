// SPDX-License-Identifier: MIT OR Apache-2.0
//! Checked surrogate-transport mechanism sensitivity and kernel scans.

use super::{
    BoundZTransportFunctional, DiscreteKernelSensitivity, DiscreteKernelSensitivityError,
    DiscreteKernelSensitivityResult, Error, ExactTransportData, ExprNode, RegimeId,
    SelectionDiagram,
};

/// Auditable result for outcome-kernel contamination of the checked z formula.
#[derive(Clone, Debug, PartialEq)]
pub struct ZTransportMechanismSensitivityResult {
    /// Stable identity of the checked derivation's inputs.
    pub query_binding: String,
    /// Provider snapshot used by the baseline formula.
    pub provider_snapshot: String,
    /// Source law regime used by the formula.
    pub source_regime: RegimeId,
    /// Baseline, exact range, tipping point, and witnesses.
    pub response: DiscreteKernelSensitivityResult,
}

/// Refusal from the narrow z outcome-kernel sensitivity contract.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ZTransportSensitivityError {
    /// Proof does not independently verify against the supplied diagram.
    #[error("z-transport derivation failed independent verification")]
    InvalidProof,
    /// The execution context was cancelled before the range was computed.
    #[error("z sensitivity cancelled")]
    Cancelled,
    /// Formula is not a checked outcome conditional times its shared parent marginal.
    #[error("z formula does not expose one shared outcome-kernel factor")]
    IncompatibleFormula,
    /// Exact source provider does not match the formula's population, regime, world, or snapshot.
    #[error("source provider does not match the checked z formula binding")]
    ProviderMismatch,
    /// Outcome, treatment, or shared parent has unsupported finite numeric support.
    #[error("z sensitivity requires finite numeric outcome and binary treatment domains")]
    UnsupportedDomain,
    /// Source law cannot supply every outcome kernel stratum, or its shared
    /// parent marginal does not have unit mass.
    #[error(
        "source law has an empty outcome-kernel stratum or a parent marginal without unit mass"
    )]
    IncompleteKernel,
    /// Contamination fraction or response threshold is invalid.
    #[error("{0}")]
    InvalidSensitivity(DiscreteKernelSensitivityError),
}

/// Largest admitted `|sum - 1|` of the shared parent marginal (the
/// evaluator's kernel-row tolerance).
const Z_SENSITIVITY_UNIT_MASS_TOLERANCE: f64 = 1e-10;

/// Evaluate the checked z formula under a single coherent outcome-kernel contamination.
///
/// The compatible formula is `sum_w P_source(Y | w, X, do(Z)) P_source(w | do(Z))`.
/// The same conditional kernel is used to form both treatment arms. The returned
/// range is an assumption range over replacement outcome distributions.
///
/// # Errors
///
/// Returns the typed refusal of the proof, formula, provider, domain or kernel check.
pub fn z_transport_mechanism_sensitivity(
    diagram: &SelectionDiagram,
    functional: &BoundZTransportFunctional,
    data: &ExactTransportData,
    max_fraction: f64,
    decision_threshold: Option<f64>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportMechanismSensitivityResult, ZTransportSensitivityError> {
    let kernels = z_surrogate_kernels(diagram, functional, data, ctx)?;
    // The stratum weights are the law's raw parent mass, which a caller's loose
    // `LawTolerance` can leave far from one; the baseline and range would then
    // be scaled by it. Unit mass is enforced, not repaired.
    let total = kernels.parent_marginal.iter().sum::<f64>();
    if !(total.is_finite() && (total - 1.0).abs() <= Z_SENSITIVITY_UNIT_MASS_TOLERANCE) {
        return Err(ZTransportSensitivityError::IncompleteKernel);
    }
    let response = DiscreteKernelSensitivity {
        source_kernel: kernels.kernels,
        outcome_values: kernels.outcome_values,
        stratum_contrast_weights: kernels.weights,
        max_fraction,
        decision_threshold,
    }
    .evaluate()
    .map_err(ZTransportSensitivityError::InvalidSensitivity)?;
    Ok(ZTransportMechanismSensitivityResult {
        query_binding: kernels.query_binding,
        provider_snapshot: kernels.provider_snapshot,
        source_regime: kernels.source_regime,
        response,
    })
}

/// The checked surrogate formula's source factors: the outcome kernel per
/// `(w, x)` stratum (w-major, control arm first), its signed contrast weights,
/// the shared parent marginal and the outcome values, with the binding.
#[derive(Clone, Debug)]
pub(crate) struct ZSurrogateKernels {
    /// `P(Y | w, x, do(z))`, one row per `(w, x)` stratum, w-major.
    pub(crate) kernels: Vec<Vec<f64>>,
    /// `-P(w | do(z))` for the control arm and `+P(w | do(z))` for the active arm.
    pub(crate) weights: Vec<f64>,
    /// `P(w | do(z))` per parent level.
    pub(crate) parent_marginal: Vec<f64>,
    /// Numeric outcome value per outcome category.
    pub(crate) outcome_values: Vec<f64>,
    /// Stable identity of the checked derivation's inputs.
    pub(crate) query_binding: String,
    /// Provider snapshot used by the baseline formula.
    pub(crate) provider_snapshot: String,
    /// Source law regime used by the formula.
    pub(crate) source_regime: RegimeId,
}

/// Why a metered read of the surrogate factors stopped.
#[derive(Debug)]
pub(crate) enum ScanStop<E> {
    /// The proof, formula, provider, domain or kernel check refused.
    Kernel(ZTransportSensitivityError),
    /// A caller's hook refused (a declared bound or a budget charge).
    Hook(E),
}

impl<E> From<ZTransportSensitivityError> for ScanStop<E> {
    fn from(error: ZTransportSensitivityError) -> Self {
        Self::Kernel(error)
    }
}

/// [`z_surrogate_kernels_metered`] with no hooks: the one-factor route.
pub(crate) fn z_surrogate_kernels(
    diagram: &SelectionDiagram,
    functional: &BoundZTransportFunctional,
    data: &ExactTransportData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZSurrogateKernels, ZTransportSensitivityError> {
    z_surrogate_kernels_metered::<std::convert::Infallible>(
        diagram,
        functional,
        data,
        ctx,
        &mut |_, _| Ok(()),
        &mut |_| Ok(()),
    )
    .map_err(|stop| match stop {
        ScanStop::Kernel(error) => error,
        ScanStop::Hook(never) => match never {},
    })
}

/// Check the proof, formula shape and provider binding of the registered
/// surrogate z formula and read its source factors from the exact law.
///
/// `shape(parent_levels, outcome_categories)` runs once the law's axes are
/// read and before any pass over its cells, so a caller can refuse a declared
/// bound before the scan; `level(w)` runs before the passes of each parent
/// level (three passes over every cell), so a caller can charge the scan.
/// Either hook's error stops the read as [`ScanStop::Hook`].
// Keep the checked binding and factor validation together so a sensitivity
// result is built only after the complete proof/provider contract passes.
#[allow(
    clippy::too_many_lines,
    reason = "metered evaluation of every surrogate kernel; the stages share the meter and the formula"
)]
pub(crate) fn z_surrogate_kernels_metered<E>(
    diagram: &SelectionDiagram,
    functional: &BoundZTransportFunctional,
    data: &ExactTransportData,
    ctx: &antecedent_core::ExecutionContext,
    shape: &mut dyn FnMut(usize, usize) -> Result<(), E>,
    level: &mut dyn FnMut(usize) -> Result<(), E>,
) -> Result<ZSurrogateKernels, ScanStop<E>> {
    let cancelled = || {
        if ctx.cancellation.is_cancelled() {
            Err(ZTransportSensitivityError::Cancelled)
        } else {
            Ok(())
        }
    };
    cancelled()?;
    let derivation = functional.derivation();
    let query = derivation.query();
    // The bound functional was verified when it was derived; the input identity
    // is what it still has to prove against this diagram.
    derivation
        .check_inputs(diagram, query)
        .map_err(|_| ZTransportSensitivityError::InvalidProof)?;
    let Some(confounder) = derivation.confounder() else {
        return Err(ZTransportSensitivityError::IncompatibleFormula.into());
    };

    let arena = functional.arena();
    // The registered surrogate factorization cites exactly one source regime.
    let [regime] = functional.cited_regimes() else {
        return Err(ZTransportSensitivityError::IncompatibleFormula.into());
    };
    let regime = *regime;
    let ExprNode::SumOut { expr, .. } = arena.node(functional.root()) else {
        return Err(ZTransportSensitivityError::IncompatibleFormula.into());
    };
    let ExprNode::Product(factors) = arena.node(*expr) else {
        return Err(ZTransportSensitivityError::IncompatibleFormula.into());
    };
    let leaves = arena.list(*factors);
    if leaves.len() != 2 {
        return Err(ZTransportSensitivityError::IncompatibleFormula.into());
    }
    let mut outcome_leaf = None;
    let mut parent_leaf = None;
    for id in leaves {
        let ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            population,
            regime: leaf_regime,
            ..
        } = arena.node(*id)
        else {
            return Err(ZTransportSensitivityError::IncompatibleFormula.into());
        };
        let mut formula_interventions = arena.intervention_set(*intervention);
        let mut query_interventions =
            query.experiment_assignment.iter().map(|a| a.variable).collect::<Vec<_>>();
        formula_interventions.sort_unstable();
        query_interventions.sort_unstable();
        if *leaf_regime != Some(regime)
            || arena.population(*population) != query.source.as_ref()
            || formula_interventions != query_interventions
        {
            return Err(ZTransportSensitivityError::IncompatibleFormula.into());
        }
        let vars = arena.var_set(*variables);
        let cond = arena.var_set(*conditioned_on);
        if vars == query.outcomes.as_ref()
            && query.treatments.iter().all(|v| cond.contains(v))
            && cond.contains(&confounder)
        {
            outcome_leaf = Some(*id);
        } else if vars == [confounder] && cond.is_empty() {
            parent_leaf = Some(*id);
        } else {
            return Err(ZTransportSensitivityError::IncompatibleFormula.into());
        }
    }
    if outcome_leaf.is_none()
        || parent_leaf.is_none()
        || query.outcomes.len() != 1
        || query.treatments.len() != 1
    {
        return Err(ZTransportSensitivityError::IncompatibleFormula.into());
    }

    let law = data
        .laws()
        .iter()
        .find(|law| {
            law.population() == query.source.as_ref()
                && law.regime() == regime
                && law.interventions().len() == query.experiment_assignment.len()
                && query.experiment_assignment.iter().all(|expected| {
                    law.interventions().iter().any(|actual| {
                        actual.variable == expected.variable && actual.value == expected.value
                    })
                })
        })
        .ok_or(ZTransportSensitivityError::ProviderMismatch)?;
    if !functional.catalog().bindings.iter().any(|binding| {
        binding.regime == law.regime()
            && binding.snapshot_identity.as_ref() == law.snapshot_identity()
    }) {
        return Err(ZTransportSensitivityError::ProviderMismatch.into());
    }
    let y = query.outcomes[0];
    let x = query.treatments[0];
    let w = confounder;
    let axis_pos = |variable| law.axes().iter().position(|axis| axis.variable == variable);
    let (yi, xi, wi) = (axis_pos(y), axis_pos(x), axis_pos(w));
    let (yi, xi, wi) = (
        yi.ok_or(ZTransportSensitivityError::ProviderMismatch)?,
        xi.ok_or(ZTransportSensitivityError::ProviderMismatch)?,
        wi.ok_or(ZTransportSensitivityError::ProviderMismatch)?,
    );
    let y_values = law.axes()[yi]
        .values
        .iter()
        .map(antecedent_core::Value::as_f64)
        .collect::<Option<Vec<_>>>()
        .ok_or(ZTransportSensitivityError::UnsupportedDomain)?;
    let x_levels = law.axes()[xi].values.len();
    let w_levels = law.axes()[wi].values.len();
    if x_levels != 2
        || w_levels == 0
        || y_values.is_empty()
        || law.snapshot_identity().trim().is_empty()
    {
        return Err(ZTransportSensitivityError::UnsupportedDomain.into());
    }
    shape(w_levels, y_values.len()).map_err(ScanStop::Hook)?;
    let dims: Vec<usize> = law.axes().iter().map(|a| a.values.len()).collect();
    let strides: Vec<usize> = (0..dims.len()).map(|i| dims[i + 1..].iter().product()).collect();
    let mut kernels = Vec::with_capacity(w_levels * 2);
    let mut weights = Vec::with_capacity(w_levels * 2);
    let mut parent_marginal = Vec::with_capacity(w_levels);
    for wl in 0..w_levels {
        cancelled()?;
        level(wl).map_err(ScanStop::Hook)?;
        let mut parent_mass = 0.0;
        for row in 0..law.probabilities().len() {
            if (row / strides[wi]) % dims[wi] == wl {
                parent_mass += law.probabilities()[row];
            }
        }
        for xl in 0..2 {
            let mut joint = vec![0.0; y_values.len()];
            let mut mass = 0.0;
            for row in 0..law.probabilities().len() {
                if (row / strides[wi]) % dims[wi] == wl && (row / strides[xi]) % dims[xi] == xl {
                    let yl = (row / strides[yi]) % dims[yi];
                    joint[yl] += law.probabilities()[row];
                    mass += law.probabilities()[row];
                }
            }
            if mass <= 0.0 {
                return Err(ZTransportSensitivityError::IncompleteKernel.into());
            }
            kernels.push(joint.into_iter().map(|p| p / mass).collect());
            weights.push(parent_mass * if xl == 0 { -1.0 } else { 1.0 });
        }
        parent_marginal.push(parent_mass);
    }
    Ok(ZSurrogateKernels {
        kernels,
        weights,
        parent_marginal,
        outcome_values: y_values,
        query_binding: format!("{}:{}:{}", query.source, query.target, functional.root().raw()),
        provider_snapshot: law.snapshot_identity().to_owned(),
        source_regime: regime,
    })
}
