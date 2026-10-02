//! Arrival and candidate checks shared by the evidence planners.
//!
//! The z-transport planner and the X6 study planner both freeze a base catalog,
//! propose regimes as a catalog delta, and later accept a real catalog in which
//! the proposed evidence arrived. These checks are the one implementation of
//! "the base is preserved", "the proposed regime arrived with exactly the
//! proposed shape" and "a proposed level lies in its declared domain"; each
//! planner maps the message into its own typed error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{
    EvidenceCatalog, EvidenceCatalogDelta, EvidenceKind, EvidenceRegime, InterventionAssignment,
    RegimeKind, VariableDomain, VariableId,
};

/// Every proposed regime is present in `actual` as available evidence with
/// exactly the proposed kind, population, interventions, levels, measured set,
/// conditioning, sampling selection, law origin and availability, and is bound
/// to some provider snapshot. A selected-sample law or a fitted-model artifact
/// is therefore never the proposed regime unless it was proposed.
pub(crate) fn validate_actual_delta(
    delta: &EvidenceCatalogDelta,
    actual: &EvidenceCatalog,
) -> Result<(), String> {
    for expected in delta.proposed_regimes.iter() {
        let Some(found) = actual.regimes.iter().find(|r| r.id == expected.id) else {
            return Err(format!("arriving catalog lacks proposed regime {}", expected.id.raw()));
        };
        if found.evidence_kind != EvidenceKind::Available
            || found.kind != expected.kind
            || found.population != expected.population
            || !same_vars(&found.interventions, &expected.interventions)
            || !same_assignments(&found.intervention_values, &expected.intervention_values)
            || !same_vars(&found.measured, &expected.measured)
            || !same_vars(&found.conditioned_on, &expected.conditioned_on)
            || found.selection != expected.selection
            || found.origin != expected.origin
            || found.distribution != expected.distribution
        {
            return Err(format!("arriving regime {} differs from proposal", expected.id.raw()));
        }
        if !actual.bindings.iter().any(|b| b.regime == found.id) {
            return Err("arriving regime has no provider binding".into());
        }
    }
    Ok(())
}

/// `actual` keeps every environment, regime and binding of `base` unchanged.
pub(crate) fn validate_base_preserved(
    base: &EvidenceCatalog,
    actual: &EvidenceCatalog,
) -> Result<(), String> {
    if base.environments != actual.environments || base.target_sampling != actual.target_sampling {
        return Err("arrival catalog does not preserve the proposal base catalog".into());
    }
    for regime in base.regimes.iter() {
        if !actual.regimes.contains(regime) {
            return Err(format!("base regime {} changed or disappeared", regime.id.raw()));
        }
    }
    for binding in base.bindings.iter() {
        if !actual.bindings.contains(binding) {
            return Err("base provider binding changed or disappeared".into());
        }
    }
    Ok(())
}

/// Every declared level of an experimental regime names each intervention once
/// and lies in the domain its population declares for that variable.
pub(crate) fn valid_intervention_values(
    catalog: &EvidenceCatalog,
    regime: &EvidenceRegime,
) -> bool {
    if regime.kind == RegimeKind::Experimental
        && regime.intervention_values.len() != regime.interventions.len()
    {
        return false;
    }
    let Some(environment) = catalog.environments.iter().find(|e| e.identity == regime.population)
    else {
        return false;
    };
    regime.intervention_values.iter().all(|assignment| {
        let Some(coordinate) =
            environment.variables.iter().find(|c| c.variable == assignment.variable)
        else {
            return false;
        };
        let Some(value) = assignment.value.as_f64() else { return false };
        match coordinate.domain {
            VariableDomain::Unspecified | VariableDomain::Continuous => value.is_finite(),
            // These are discrete domain membership checks: approximate equality
            // would admit values that are not members of the declared support.
            #[expect(clippy::float_cmp, reason = "binary support membership requires exact values")]
            VariableDomain::Binary => value == 0.0 || value == 1.0,
            VariableDomain::Count => value >= 0.0 && value.fract() == 0.0,
            VariableDomain::Categorical { cardinality } => {
                value >= 0.0 && value < f64::from(cardinality) && value.fract() == 0.0
            }
        }
    })
}

/// Equal intervention levels, independent of order.
pub(crate) fn same_assignments(a: &[InterventionAssignment], b: &[InterventionAssignment]) -> bool {
    a.len() == b.len()
        && a.iter().all(|x| {
            b.iter().any(|y| {
                x.variable == y.variable
                    && antecedent_core::same_intervention_level(&x.value, &y.value)
            })
        })
}

/// Equal variable sets, independent of order.
pub(crate) fn same_vars(a: &[VariableId], b: &[VariableId]) -> bool {
    a.iter().collect::<BTreeSet<_>>() == b.iter().collect::<BTreeSet<_>>()
}
