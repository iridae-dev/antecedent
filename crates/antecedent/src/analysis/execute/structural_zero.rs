// SPDX-License-Identifier: MIT OR Apache-2.0
//! The structural zero of a static derivative whose treatment cannot reach its outcome.
//!
//! A do-intervention on a variable with no directed path to the outcome leaves the outcome's
//! law unchanged, so the derivative of the interventional mean in that treatment is exactly zero
//! whatever the observed association. The plug-in estimators regress the outcome on the
//! treatment and the adjustment set and read the slope of that regression, which is an
//! association here; the published value is the structural zero instead, with a degenerate band
//! and a typed note (the static counterpart of the temporal ADMG route's
//! `response.temporal_admg.no_directed_path`).

use std::sync::Arc;

use antecedent_core::{
    CausalResponse, Diagnostic, DiagnosticKind, DiagnosticSeverity, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, VariableId,
};
use antecedent_graph::{Dag, DenseNodeId};

/// Diagnostic code of a derivative coordinate with no directed path from treatment to outcome.
pub(crate) const NO_DIRECTED_PATH_CODE: &str = "response.dag.no_directed_path";

/// Whether the directed part of `graph` has no path from `treatment` to `outcome`.
fn unreachable(graph: &Dag, treatment: VariableId, outcome: VariableId) -> Option<bool> {
    let (t, y) = (treatment.raw() as usize, outcome.raw() as usize);
    if t >= graph.node_count() || y >= graph.node_count() {
        return None;
    }
    let (from, to) = (
        DenseNodeId::from_raw(u32::try_from(t).ok()?),
        DenseNodeId::from_raw(u32::try_from(y).ok()?),
    );
    Some(from != to && !graph.reaches(from, to))
}

/// Replace every derivative coordinate with no directed treatment-to-outcome path by the
/// structural zero, collapse its band, and note it. A directional derivative is a combination of
/// its cells, so it is zeroed only when every cell is.
pub(crate) fn apply(response: &mut CausalResponse, graph: &Dag, query: &ResponseQuery) {
    let (outcomes, treatments) = (query.functional.outcome_ids(), query.functional.treatment_ids());
    if outcomes.is_empty() || treatments.is_empty() {
        return;
    }
    // Row-major outcomes-by-treatments flags.
    let mut flags = Vec::with_capacity(outcomes.len() * treatments.len());
    for &outcome in &outcomes {
        for &treatment in &treatments {
            let Some(flag) = unreachable(graph, treatment, outcome) else { return };
            flags.push(flag);
        }
    }
    if !flags.iter().any(|&flag| flag) {
        return;
    }
    let all = flags.iter().all(|&flag| flag);
    let zeroed: Vec<bool> = match (&query.functional, &response.estimate) {
        (
            ResponseFunctional::PointDerivative { .. }
            | ResponseFunctional::AverageDerivative { .. },
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(_)),
        ) => vec![all],
        (
            ResponseFunctional::Jacobian { .. },
            ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }),
        ) if values.len() == flags.len() => flags.clone(),
        (
            ResponseFunctional::DirectionalDerivative { .. },
            ResponseIdentification::PointIdentified(ResponseValue::Vector(values)),
        ) if all => vec![true; values.len()],
        _ => return,
    };
    if !zeroed.iter().any(|&z| z) {
        return;
    }
    let zero_values = |values: &Arc<[f64]>| -> Arc<[f64]> {
        values.iter().zip(&zeroed).map(|(&value, &z)| if z { 0.0 } else { value }).collect()
    };
    response.estimate = match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(_)) => {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(0.0))
        }
        ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
            outcomes,
            treatments,
            values,
        }) => ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
            outcomes: *outcomes,
            treatments: *treatments,
            values: zero_values(values),
        }),
        ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) => {
            ResponseIdentification::PointIdentified(ResponseValue::Vector(zero_values(values)))
        }
        other => other.clone(),
    };
    response.uncertainty = match &response.uncertainty {
        ResponseUncertainty::Scalar { level, interpretation, .. } if zeroed[0] => {
            ResponseUncertainty::Scalar {
                standard_error: 0.0,
                level: *level,
                lower: 0.0,
                upper: 0.0,
                interpretation: *interpretation,
                draws: None,
            }
        }
        ResponseUncertainty::PointwiseBand { level, lower, upper, interpretation, .. }
            if lower.len() == zeroed.len() =>
        {
            ResponseUncertainty::PointwiseBand {
                level: *level,
                lower: zero_values(lower),
                upper: zero_values(upper),
                interpretation: *interpretation,
                draws: None,
            }
        }
        other => other.clone(),
    };
    response.support.warnings.push(Diagnostic::new(
        NO_DIRECTED_PATH_CODE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "{} of {} derivative coordinates have no directed path from the treatment to the \
             outcome in the declared graph, so their do-effect is exactly zero; the published \
             value is that structural zero, not the regression slope, which is a pure \
             association there",
            zeroed.iter().filter(|&&z| z).count(),
            zeroed.len()
        ),
    ));
}
