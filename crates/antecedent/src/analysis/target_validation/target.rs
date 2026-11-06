//! Result target classification and coordinate projection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, DataClassification, RefutationTarget, ResponseFunctional,
    ResponseIdentification, ResponseValue, StudyResult, TargetFamily, TargetQuadrant, VariableId,
};

/// The family of the quantity `result` reports.
#[must_use]
pub fn result_target_family(result: &StudyResult) -> TargetFamily {
    if result.mediation().is_some() || result.mediation_grid().is_some() {
        return TargetFamily::Mediation;
    }
    if let Some(response) = result.response() {
        return match &response.estimand {
            ResponseFunctional::AverageDerivative { .. }
            | ResponseFunctional::PointDerivative { .. }
            | ResponseFunctional::DirectionalDerivative { .. }
            | ResponseFunctional::Jacobian { .. } => TargetFamily::ResponseDerivative,
            // A response level remains a world value even when it has one coordinate or
            // is evaluated at a temporal horizon. It is not an average-effect contrast.
            ResponseFunctional::MeanCurve { .. }
            | ResponseFunctional::InterventionResponse { .. } => TargetFamily::WorldValued,
        };
    }
    if result.distribution().is_some() {
        return TargetFamily::InterventionalLaw;
    }
    if result.anomaly().is_some() || result.change_attribution().is_some() {
        return TargetFamily::Attribution;
    }
    // A counterfactual reports its population mean: the per-unit effects number as many as the
    // scored units, far more coordinates than any replicate law can estimate a covariance for.
    TargetFamily::ScalarAverageEffect
}

/// The per-component vector of an anomaly or change attribution: for each scored target the
/// mean score and the mean Shapley contribution of every noise component; for a change
/// attribution the total change and every component's contribution.
fn attribution_values(result: &StudyResult) -> Result<(Vec<Arc<str>>, Vec<f64>), CausalError> {
    let mut labels: Vec<Arc<str>> = Vec::new();
    let mut values = Vec::new();
    if let Some(scores) = result.anomaly() {
        for score in scores {
            let rows = score.rows.len();
            let width = score.noise_components.len();
            if rows == 0 || score.noise_contributions.len() != rows * width {
                return Err(CausalError::Unsupported {
                    message: "an anomaly attribution without scored rows has no vector to refute",
                });
            }
            let target = score.target.raw();
            labels.push(Arc::from(format!("anomaly[{target}].score")));
            values.push(score.scores.iter().sum::<f64>() / rows as f64);
            // One row-major pass accumulates every component's sum (same per-component
            // addition order as a column walk, without the strided access).
            let mut component_sums = vec![0.0_f64; width];
            for row in score.noise_contributions.chunks_exact(width.max(1)) {
                for (sum, value) in component_sums.iter_mut().zip(row) {
                    *sum += value;
                }
            }
            for (component, sum) in score.noise_components.iter().zip(&component_sums) {
                labels.push(Arc::from(format!(
                    "anomaly[{target}].component[{}]",
                    component.variable().raw()
                )));
                values.push(sum / rows as f64);
            }
        }
    } else if let Some(change) = result.change_attribution() {
        labels.push(Arc::from("change.total"));
        values.push(change.total_change);
        for contribution in change.contributions.iter() {
            labels.push(Arc::from(format!(
                "change.component[{}]",
                contribution.component.variable().raw()
            )));
            values.push(contribution.contribution);
        }
    }
    if values.is_empty() {
        return Err(CausalError::Unsupported {
            message: "the attribution result carries no component vector to refute",
        });
    }
    Ok((labels, values))
}

/// The atoms of an interventional distribution in a coordinate order that does not depend on
/// the rows: an estimate lists outcome levels in order of first appearance, so a resample or a
/// permutation that changes the first row would otherwise exchange the coordinates of the
/// vector. Atoms are ordered by conditioning assignment, then by outcome value (numerically
/// where the values are numbers).
pub(crate) fn canonical_law_atoms(
    atoms: &[antecedent_estimate::DistributionAtom],
) -> Vec<antecedent_estimate::DistributionAtom> {
    use std::cmp::Ordering;
    let compare = |left: &[(VariableId, antecedent_core::Value)],
                   right: &[(VariableId, antecedent_core::Value)]| {
        for ((_, a), (_, b)) in left.iter().zip(right) {
            let order = match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => x.total_cmp(&y),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => format!("{a:?}").cmp(&format!("{b:?}")),
            };
            if order != Ordering::Equal {
                return order;
            }
        }
        Ordering::Equal
    };
    let mut sorted = atoms.to_vec();
    sorted.sort_by(|a, b| {
        compare(&a.conditioning, &b.conditioning).then_with(|| compare(&a.outcomes, &b.outcomes))
    });
    sorted
}

/// The probability atoms of an interventional distribution, all but the last (the others
/// determine it, and a vector that sums to one has a singular replicate covariance). A
/// distribution with a single atom, or a mean only, is read through its mean.
fn law_values(result: &StudyResult) -> Result<(Vec<Arc<str>>, Vec<f64>), CausalError> {
    let distribution = result.distribution().ok_or(CausalError::Unsupported {
        message: "no interventional distribution payload to refute",
    })?;
    let atoms = &canonical_law_atoms(&distribution.atoms);
    if atoms.len() >= 2 {
        let kept = atoms.len() - 1;
        let labels = (0..kept).map(|i| Arc::from(format!("atom[{i}]"))).collect();
        let values = atoms[..kept].iter().map(|atom| atom.probability).collect();
        return Ok((labels, values));
    }
    if distribution.mean.is_finite() {
        return Ok((vec![Arc::from("mean")], vec![distribution.mean]));
    }
    Err(CausalError::Unsupported {
        message: "the interventional distribution has neither two atoms nor a finite mean to refute",
    })
}

fn labelled(prefix: &str, values: &[f64]) -> (Vec<Arc<str>>, Vec<f64>) {
    let labels = (0..values.len()).map(|i| Arc::from(format!("{prefix}[{i}]"))).collect();
    (labels, values.to_vec())
}

/// Labels and values of the quantity `result` reports, in a stable coordinate order.
///
/// # Errors
///
/// A result whose target has no finite point value (a partially identified envelope, a
/// graph-dependent response, a distribution, a mediation decomposition with a missing
/// component, or no scalar estimate).
#[allow(
    clippy::type_complexity,
    reason = "the public signature returns the family, labels and values as one tuple"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the target-specific branches preserve a single checked operation"
)]
pub fn result_target_values(
    result: &StudyResult,
) -> Result<(TargetFamily, Vec<Arc<str>>, Vec<f64>), CausalError> {
    let family = result_target_family(result);
    let refuse = |message: &'static str| CausalError::Unsupported { message };
    let (labels, values) = match family {
        TargetFamily::Mediation => {
            let m = result.mediation().ok_or_else(|| {
                refuse(
                    "a multi-horizon mediation grid is refuted one horizon at a time; next \
                     attempt: prepare one horizon per study",
                )
            })?;
            let total = m.total.ok_or_else(|| refuse("mediation total effect is unavailable"))?;
            let direct =
                m.direct.ok_or_else(|| refuse("mediation direct effect is unavailable"))?;
            let mediated =
                m.mediated.ok_or_else(|| refuse("mediation indirect effect is unavailable"))?;
            (
                vec![Arc::<str>::from("total"), Arc::from("direct"), Arc::from("indirect")],
                vec![total, direct, mediated],
            )
        }
        TargetFamily::ResponseDerivative
        | TargetFamily::WorldValued
        | TargetFamily::TemporalEffect
            if result.response().is_some() =>
        {
            let response = result.response().ok_or_else(|| refuse("no response payload"))?;
            let value = match &response.estimate {
                ResponseIdentification::PointIdentified(value) => value,
                // A graph posterior that loses mass to atoms the identifier refuses publishes
                // its value conditional on the atoms that identify; the refits recompute that
                // same conditional value.
                ResponseIdentification::GraphDependent(_) => result
                    .structural_response
                    .as_ref()
                    .and_then(|mixture| mixture.conditional_on_identified.as_ref())
                    .ok_or_else(|| {
                        refuse(
                            "only a point-identified response has a target vector to refute; \
                             next attempt: refute each identified graph's response separately",
                        )
                    })?,
                _ => {
                    return Err(refuse(
                        "only a point-identified response has a target vector to refute; next \
                         attempt: refute each identified graph's response separately",
                    ));
                }
            };
            match value {
                ResponseValue::Scalar(v) => (vec![Arc::<str>::from("value")], vec![*v]),
                ResponseValue::Surface { mean, .. } => labelled("grid", mean),
                ResponseValue::Vector(values) => labelled("coordinate", values),
                ResponseValue::Jacobian { values, .. } => labelled("jacobian", values),
                ResponseValue::Envelope(_) => {
                    return Err(refuse(
                        "an envelope is a set, not a point target; next attempt: the \
                         assumption-range sensitivity analyses of the bounds",
                    ));
                }
            }
        }
        TargetFamily::WorldValued => {
            return Err(refuse(
                "an interventional distribution is refuted through its mean response; next \
                 attempt: query the interventional mean as an InterventionResponse",
            ));
        }
        TargetFamily::Attribution => attribution_values(result)?,
        TargetFamily::InterventionalLaw => law_values(result)?,
        TargetFamily::ConditionalEffect => {
            let ite = result
                .counterfactual()
                .ok_or_else(|| refuse("no counterfactual payload to refute"))?;
            labelled("unit", &ite.unit_effects)
        }
        _ if result.counterfactual_contrasts().is_some() => {
            let set = result
                .counterfactual_contrasts()
                .ok_or_else(|| refuse("no counterfactual payload to refute"))?;
            (
                set.contrasts
                    .iter()
                    .map(|contrast| {
                        Arc::<str>::from(format!("mean_ite[{}]", contrast.outcome.raw()))
                    })
                    .collect(),
                set.contrasts.iter().map(|contrast| contrast.mean_ite).collect(),
            )
        }
        _ if result.counterfactual().is_some() => {
            let ite = result
                .counterfactual()
                .ok_or_else(|| refuse("no counterfactual payload to refute"))?;
            (vec![Arc::<str>::from("mean_ite")], vec![ite.mean_ite])
        }
        _ => {
            let effect = result
                .estimate
                .as_effect()
                .ok_or_else(|| refuse("the result reports no scalar effect to refute"))?;
            (vec![Arc::<str>::from("ate")], vec![effect.ate])
        }
    };
    if values.iter().any(|v| !v.is_finite()) {
        return Err(refuse("the result's target vector is not finite"));
    }
    Ok((family, labels, values))
}

/// Quadrant of `result`: its data classification, retained structure for multiple
/// environments, and whether it carries a posterior.
#[must_use]
pub fn result_quadrant(result: &StudyResult) -> TargetQuadrant {
    if let Some(quadrant) = result.lowered_quadrant {
        return quadrant;
    }
    TargetQuadrant {
        temporal: temporal_refutation_data(
            result.logical_plan.data_classification,
            result.certificate.as_ref().map(|certificate| certificate.graph_class.as_str()),
        ),
        bayesian: result.posterior.is_some(),
    }
}

// Multi-environment is a partitioning declaration, not a declaration of iid rows:
// both static populations and lagged systems use it. Read the retained structure
// class rather than labeling every environment collection static (or temporal).
pub(super) fn temporal_refutation_data(
    data: DataClassification,
    graph_class: Option<&str>,
) -> bool {
    match data {
        DataClassification::Tabular => false,
        DataClassification::Temporal | DataClassification::Panel | DataClassification::Event => {
            true
        }
        DataClassification::MultiEnvironment => [
            crate::GraphClass::TemporalDag,
            crate::GraphClass::TemporalCpdag,
            crate::GraphClass::TemporalPag,
            crate::GraphClass::TemporalAdmg,
        ]
        .iter()
        .any(|class| Some(class.as_str()) == graph_class),
    }
}

/// The refutation target of `result`, defined by the lowered program that produced it.
///
/// # Errors
///
/// A result no lowered dispatch produced, or one whose target has no point value
/// ([`result_target_values`]).
pub fn result_refutation_target(result: &StudyResult) -> Result<RefutationTarget, CausalError> {
    let source = result.lowered_program_identity().ok_or(CausalError::Unsupported {
        message: "a refutation target is defined by the lowered program that produced the \
                  result; next attempt: execute through Study::compile or a prepared handle",
    })?;
    let (family, labels, _) = result_target_values(result)?;
    RefutationTarget::new(family, result_quadrant(result), labels, *source.as_bytes())
        .map_err(CausalError::from)
}
