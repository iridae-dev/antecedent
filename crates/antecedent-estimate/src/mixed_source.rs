//! Exact execution of a checked mixed-source formula.
//!
//! The formula is a checked expression whose every leaf names the catalog
//! regime of the study that supplies it, so the exact provider reads each leaf
//! from that regime's law and nowhere else. The compiled plan is frozen at
//! preparation; evaluating it never searches. No interval is published on this
//! route: exact laws carry no sampling uncertainty and the route claims a point.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::transport::{BoundZFormula, compile_exact_z_transport, validate_exact_laws};

impl BoundZFormula for antecedent_identify::BoundMixedSourceFunctional {
    fn arena(&self) -> &antecedent_expr::CausalExprArena {
        self.arena()
    }
    fn root(&self) -> antecedent_expr::ExprId {
        self.root()
    }
    fn catalog(&self) -> &antecedent_core::EvidenceCatalog {
        self.catalog()
    }
    fn cited_regimes(&self) -> &[antecedent_core::RegimeId] {
        self.cited_regimes()
    }
    fn outcomes(&self) -> &std::sync::Arc<[antecedent_core::VariableId]> {
        &self.derivation().query().outcomes
    }
    fn treatments(&self) -> &[antecedent_core::VariableId] {
        &self.derivation().query().treatments
    }
}

/// Evaluate a checked mixed-source functional against exact laws. Each leaf reads
/// only the law of the regime it cites; the result is a point distribution with
/// no sampling standard errors or intervals.
///
/// # Errors
/// Provider/catalog disagreement, a request that does not bind the treatments,
/// a request at a level no cited experiment supplies, missing support, or
/// resource limits.
pub fn evaluate_exact_mixed_source(
    functional: &antecedent_identify::BoundMixedSourceFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactDistribution, antecedent_expr::EvalError> {
    prepare_exact_mixed_source(functional, data, request, limits, ctx)?.evaluate(ctx)
}

/// Validate the supplied laws against the frozen catalog and compile the checked
/// formula once, without evaluating it.
///
/// # Errors
/// Provider/catalog disagreement, a request outside the cited levels, or a
/// compile-time resource limit.
pub fn prepare_exact_mixed_source(
    functional: &antecedent_identify::BoundMixedSourceFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactEvaluationPlan, antecedent_expr::EvalError> {
    validate_exact_laws(functional.catalog(), &data)?;
    compile_exact_z_transport(functional, data, request, limits, ctx)
}
