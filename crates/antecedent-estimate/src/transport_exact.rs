// SPDX-License-Identifier: MIT OR Apache-2.0
//! Checked exact transport plan preparation and law validation.

use super::BoundZFormula;

/// Evaluate a checked, catalog-bound transport functional against supplied exact laws.
/// The returned distribution carries no sampling standard errors or intervals.
///
/// # Errors
/// Inconsistent evidence/provider metadata, missing support, invalid mass, or budgets.
pub fn evaluate_exact_transport(
    functional: &antecedent_identify::BoundTransportFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactDistribution, antecedent_expr::EvalError> {
    prepare_exact_transport(functional, data, request, limits, ctx)?.evaluate(ctx)
}

/// Evaluate the currently supported point-only z-surrogate functional against
/// exact or empirical joint-law tables. Empirical inputs produce a plug-in
/// point estimate; this route publishes no interval or sampling guarantee.
///
/// # Errors
/// Request, catalog, provider, support, or resource validation failures.
pub fn evaluate_exact_z_transport(
    functional: &antecedent_identify::BoundZTransportFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactDistribution, antecedent_expr::EvalError> {
    prepare_exact_z_transport(functional, data, request, limits, ctx)?.evaluate(ctx)
}

/// Evaluate a checked multi-source limited-experiment functional against exact
/// laws, or empirical tables as a plug-in point. Each leaf reads only the law of
/// the regime it is bound to, so every factor is evaluated from the source whose
/// premises certified it; there is no separate estimator route.
///
/// # Errors
/// Provider/catalog disagreement, a request outside the cited levels, missing
/// support, or resource limits.
pub fn evaluate_exact_mz_transport(
    functional: &antecedent_identify::BoundMzTransportFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactDistribution, antecedent_expr::EvalError> {
    prepare_exact_mz_transport(functional, data, request, limits, ctx)?.evaluate(ctx)
}

/// Validate the supplied laws against the frozen catalog and compile the checked
/// mz formula once, without evaluating it.
///
/// # Errors
/// Provider/catalog disagreement, a request outside the cited levels, or a
/// compile-time resource limit.
pub fn prepare_exact_mz_transport(
    functional: &antecedent_identify::BoundMzTransportFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactEvaluationPlan, antecedent_expr::EvalError> {
    validate_exact_laws(functional.catalog(), &data)?;
    compile_exact_z_transport(functional, data, request, limits, ctx)
}

/// Validate source joint-law providers and compile the checked zTR formula.
///
/// # Errors
/// Provider/catalog disagreement, missing assignment, or resource limit.
pub fn prepare_exact_z_transport(
    functional: &antecedent_identify::BoundZTransportFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactEvaluationPlan, antecedent_expr::EvalError> {
    validate_exact_laws(functional.catalog(), &data)?;
    compile_exact_z_transport(functional, data, request, limits, ctx)
}

/// Compile the checked zTR formula against laws whose provider contract was
/// already validated (a bootstrap or posterior draw rebuilds validated tables
/// with new probabilities and need not revalidate them).
///
/// # Errors
/// A request that does not bind the certified treatments at a cited level, or
/// a compile-time resource limit.
pub(crate) fn compile_exact_z_transport(
    functional: &dyn BoundZFormula,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactEvaluationPlan, antecedent_expr::EvalError> {
    use antecedent_core::same_intervention_level;
    use antecedent_expr::{EvalError, ExactEvaluationPlan, ExactEvaluationSpec, LawTolerance};

    require_treatment_request(functional.treatments(), &request)?;
    // A source factor that exchanged a treatment cites the experiment at the
    // concrete level recorded in the proof, whatever the root shape (a bare
    // direct-exchange distribution, or a summed product on the recursive
    // route). Such a factor answers only the target intervention at that
    // level; a request at another level would silently be answered with the
    // proof's level, so it is refused here rather than evaluated.
    for (treatment, cited) in cited_treatment_levels(functional) {
        let requested = request.get(treatment);
        match cited {
            Some(level)
                if requested.is_none_or(|value| !same_intervention_level(value, &level)) =>
            {
                return Err(EvalError::ProviderKind(
                    "exact zTR request must match the concrete source intervention its formula cites",
                ));
            }
            None if !request_level_is_cited(functional, treatment, requested) => {
                return Err(EvalError::ProviderKind(
                    "exact zTR request names a treatment level no cited source experiment supplies",
                ));
            }
            _ => {}
        }
    }
    // A factor whose exchanged coordinate is bound by an enclosing summation
    // names no single regime; the concrete world selects its law.
    ExactEvaluationPlan::compile(
        ExactEvaluationSpec {
            arena: functional.arena(),
            root: functional.root(),
            outcomes: functional.outcomes().clone(),
            request,
            limits,
            tolerance: LawTolerance::default(),
        },
        data.with_world_bound_leaves(functional.cited_regimes()),
        ctx,
    )
}

/// The request must bind exactly the certified treatment coordinates.
fn require_treatment_request(
    treatments: &[antecedent_core::VariableId],
    request: &antecedent_expr::Assignment,
) -> Result<(), antecedent_expr::EvalError> {
    if request.entries().len() != treatments.len()
        || treatments.iter().any(|variable| request.get(*variable).is_none())
    {
        return Err(antecedent_expr::EvalError::ProviderKind(
            "exact request must bind precisely the certified treatment coordinates",
        ));
    }
    Ok(())
}

/// Check every supplied law against the evidence regime it claims to realize:
/// the regime must supply a measured population law (available, not a model
/// artifact, not a selected sample), be joint and unconditioned, the law's
/// interventions and axes must be the regime's, every catalog binding of the
/// regime must name the law's snapshot, and every declared coordinate domain
/// must agree with the law's axis values. Shared by the classical and z routes.
///
/// # Errors
/// The first provider/catalog disagreement found.
fn check_provider_cancelled() -> Result<(), antecedent_expr::EvalError> {
    if antecedent_core::numerical_admission::is_cancelled() {
        return Err(antecedent_expr::EvalError::ExactLaw(Box::new(
            antecedent_expr::ExactLawError {
                kind: "cancelled",
                population: std::sync::Arc::from(""),
                regime: None,
                variables: std::sync::Arc::from([]),
                conditioning: std::sync::Arc::from([]),
                interventions: std::sync::Arc::from([]),
            },
        )));
    }
    Ok(())
}

fn checked_text_equal(a: &str, b: &str) -> Result<bool, antecedent_expr::EvalError> {
    check_provider_cancelled()?;
    if a.len() != b.len() {
        return Ok(false);
    }
    for (a, b) in a.as_bytes().chunks(512).zip(b.as_bytes().chunks(512)) {
        check_provider_cancelled()?;
        if a != b {
            return Ok(false);
        }
    }
    Ok(true)
}

fn checked_value_equal(
    a: &antecedent_core::Value,
    b: &antecedent_core::Value,
) -> Result<bool, antecedent_expr::EvalError> {
    if let (antecedent_core::Value::Label(a), antecedent_core::Value::Label(b)) = (a, b) {
        checked_text_equal(a, b)
    } else {
        check_provider_cancelled()?;
        Ok(a == b)
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the bounded estimation procedure keeps its related calculations together"
)]
pub(crate) fn validate_exact_laws(
    catalog: &antecedent_core::EvidenceCatalog,
    data: &antecedent_expr::ExactTransportData,
) -> Result<(), antecedent_expr::EvalError> {
    use antecedent_core::{DistributionAvailability, VariableDomain};
    use antecedent_expr::EvalError;
    for law in data.laws() {
        check_provider_cancelled()?;
        let mut selected = None;
        for candidate in catalog.regimes.iter() {
            check_provider_cancelled()?;
            if candidate.id == law.regime()
                && checked_text_equal(&candidate.population, law.population())?
            {
                selected = Some(candidate);
                break;
            }
        }
        let regime = selected
            .ok_or(EvalError::ProviderKind("exact provider names an unknown evidence regime"))?;
        if !regime.supplies_population_law()
            || !matches!(regime.distribution, DistributionAvailability::Joint)
            || !regime.conditioned_on.is_empty()
            || law.interventions().len() != regime.interventions.len()
        {
            return Err(EvalError::ProviderKind(
                "exact provider disagrees with its evidence regime",
            ));
        }
        for assignment in law.interventions() {
            check_provider_cancelled()?;
            let mut found = false;
            for chunk in regime.interventions.chunks(512) {
                check_provider_cancelled()?;
                if chunk.contains(&assignment.variable) {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(EvalError::ProviderKind(
                    "exact provider disagrees with its evidence regime",
                ));
            }
        }
        for axis in law.axes() {
            check_provider_cancelled()?;
            let mut found = false;
            for chunk in regime.measured.chunks(512) {
                check_provider_cancelled()?;
                if chunk.contains(&axis.variable) {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(EvalError::ProviderKind(
                    "exact provider disagrees with its evidence regime",
                ));
            }
        }
        for required in regime.intervention_values.iter() {
            check_provider_cancelled()?;
            let mut found = false;
            for assignment in law.interventions() {
                check_provider_cancelled()?;
                if assignment.variable == required.variable
                    && checked_value_equal(&assignment.value, &required.value)?
                {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(EvalError::ProviderKind(
                    "exact provider disagrees with its evidence regime",
                ));
            }
        }
        for binding in catalog.bindings.iter() {
            check_provider_cancelled()?;
            if binding.regime == regime.id
                && !checked_text_equal(&binding.snapshot_identity, law.snapshot_identity())?
            {
                return Err(EvalError::ProviderKind(
                    "exact provider snapshot does not match catalog binding",
                ));
            }
        }
        for axis in law.axes() {
            for env in catalog.environments.iter() {
                check_provider_cancelled()?;
                for coordinate in env.variables.iter() {
                    check_provider_cancelled()?;
                    if coordinate.variable != axis.variable {
                        continue;
                    }
                    let valid = match coordinate.domain {
                        VariableDomain::Unspecified => true,
                        VariableDomain::Continuous => false,
                        VariableDomain::Binary => {
                            let mut seen = [false; 2];
                            for chunk in axis.values.chunks(512) {
                                check_provider_cancelled()?;
                                for value in chunk {
                                    seen[0] |= value.as_f64() == Some(0.0);
                                    seen[1] |= value.as_f64() == Some(1.0);
                                }
                            }
                            axis.values.len() == 2 && seen[0] && seen[1]
                        }
                        VariableDomain::Categorical { cardinality } => {
                            let mut all =
                                usize::try_from(cardinality).ok() == Some(axis.values.len());
                            if all {
                                for level in 0..cardinality {
                                    check_provider_cancelled()?;
                                    let mut found = false;
                                    for chunk in axis.values.chunks(512) {
                                        check_provider_cancelled()?;
                                        if chunk
                                            .iter()
                                            .any(|v| v.as_f64() == Some(f64::from(level)))
                                        {
                                            found = true;
                                            break;
                                        }
                                    }
                                    if !found {
                                        all = false;
                                        break;
                                    }
                                }
                            }
                            all
                        }
                        VariableDomain::Count => {
                            let mut all = true;
                            for chunk in axis.values.chunks(512) {
                                check_provider_cancelled()?;
                                if !chunk.iter().all(|v| {
                                    v.as_f64().is_some_and(|v| v >= 0.0 && v.fract() == 0.0)
                                }) {
                                    all = false;
                                    break;
                                }
                            }
                            all
                        }
                    };
                    if !valid {
                        return Err(EvalError::ProviderKind(
                            "exact provider domain disagrees with evidence coordinates",
                        ));
                    }
                }
            }
        }
    }
    check_provider_cancelled()?;
    Ok(())
}

/// Whether a cited source regime supplies the requested level of `treatment`.
fn request_level_is_cited(
    functional: &dyn BoundZFormula,
    treatment: antecedent_core::VariableId,
    requested: Option<&antecedent_core::Value>,
) -> bool {
    use antecedent_core::same_intervention_level;
    let Some(requested) = requested else { return false };
    functional
        .cited_regimes()
        .iter()
        .filter_map(|id| functional.catalog().regimes.iter().find(|regime| regime.id == *id))
        .filter(|regime| regime.interventions.contains(&treatment))
        .any(|regime| {
            regime.intervention_values.is_empty()
                || regime.intervention_values.iter().any(|actual| {
                    actual.variable == treatment
                        && same_intervention_level(&actual.value, requested)
                })
        })
}

/// Every treatment coordinate a reachable source factor intervenes on, with its
/// concrete level, or `None` when the factor leaves it symbolic for the request
/// to bind.
fn cited_treatment_levels(
    functional: &dyn BoundZFormula,
) -> Vec<(antecedent_core::VariableId, Option<antecedent_core::Value>)> {
    use antecedent_expr::ExprNode;
    let arena = functional.arena();
    let treatments = functional.treatments();
    let mut pending = vec![functional.root()];
    let mut seen = std::collections::BTreeSet::new();
    let mut cited = Vec::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution { intervention, .. } => {
                for assignment in arena.intervention_assignments(*intervention) {
                    if treatments.contains(&assignment.variable)
                        && !cited.iter().any(|(v, _)| *v == assignment.variable)
                    {
                        let level = (!assignment.is_symbolic()).then(|| assignment.value.clone());
                        cited.push((assignment.variable, level));
                    }
                }
            }
            ExprNode::Kernel { body, .. } => pending.push(*body),
            ExprNode::Product(list) => pending.extend(arena.list(*list)),
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                pending.push(*expr);
            }
            ExprNode::Ratio { numerator, denominator } => {
                pending.extend([*numerator, *denominator]);
            }
            ExprNode::Expectation { distribution, .. } => pending.push(*distribution),
            ExprNode::Contrast { left, right, .. } => pending.extend([*left, *right]),
        }
    }
    cited
}

/// Validate exact providers and compile without evaluating any probabilities.
///
/// # Errors
/// Provider contract, coverage, or resource limit violation.
pub fn prepare_exact_transport(
    functional: &antecedent_identify::BoundTransportFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactEvaluationPlan, antecedent_expr::EvalError> {
    let query = functional.derivation().query();
    require_treatment_request(&query.treatments, &request)?;
    validate_exact_laws(functional.catalog(), &data)?;
    antecedent_expr::ExactEvaluationPlan::compile(
        antecedent_expr::ExactEvaluationSpec {
            arena: functional.arena(),
            root: functional.root(),
            outcomes: query.outcomes.clone(),
            request,
            limits,
            tolerance: antecedent_expr::LawTolerance::default(),
        },
        data,
        ctx,
    )
}
