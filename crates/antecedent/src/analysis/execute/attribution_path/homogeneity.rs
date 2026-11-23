//! Path-specific assessment of counterfactual effect homogeneity.

use super::{
    Dag, DenseNodeId, Diagnostic, DiagnosticKind, DiagnosticSeverity, TableView, TabularData,
    VariableId,
};

/// Why the published per-unit effects do, or do not, vary.
#[derive(Debug)]
pub(super) enum HomogeneityVerdict {
    /// Some mechanism on a `treatment → outcome` path can modify the effect.
    Heterogeneous,
    /// Every path mechanism is additively separable, and no family that could
    /// have modified the effect was fit anywhere on those paths: the per-unit
    /// effects are equal by construction of the candidate set. Carries the
    /// outcome's selected family.
    Structural(crate::gcm::MechanismFamily),
    /// Every path mechanism is additively separable, but a heterogeneity-capable
    /// family *was* fit on some path node and lost on validation score: equal
    /// per-unit effects are a finding about the data, not a property fixed in
    /// advance.
    Empirical {
        /// `(node variable, rejected family, its score, selected family, selected score)`.
        rejected: Vec<RejectedHeterogeneity>,
    },
}

/// A heterogeneity-capable family scored on a path node and not selected.
#[derive(Debug)]
pub(super) struct RejectedHeterogeneity {
    variable: VariableId,
    family: crate::gcm::MechanismFamily,
    score: f64,
    selected: crate::gcm::MechanismFamily,
    selected_score: f64,
}

/// Assignments whose mechanisms are re-evaluated in the acted world: strictly
/// downstream of the treatment and upstream of (or at) the outcome.
pub(super) fn path_assignments<'a>(
    graph: &Dag,
    assignments: &'a [crate::gcm::MechanismAssignment],
    treatment: VariableId,
    outcome: VariableId,
) -> Option<(Vec<&'a crate::gcm::MechanismAssignment>, DenseNodeId, DenseNodeId)> {
    let (on_path, treatment_nodes, outcome_node) =
        path_assignments_over(graph, assignments, &[treatment], outcome)?;
    Some((on_path, treatment_nodes[0], outcome_node))
}

/// [`path_assignments`] for several simultaneously intervened treatments: the assignments
/// strictly downstream of at least one treatment and upstream of (or at) the outcome. Also
/// returns each treatment's node, in the order given.
pub(super) fn path_assignments_over<'a>(
    graph: &Dag,
    assignments: &'a [crate::gcm::MechanismAssignment],
    treatments: &[VariableId],
    outcome: VariableId,
) -> Option<(Vec<&'a crate::gcm::MechanismAssignment>, Vec<DenseNodeId>, DenseNodeId)> {
    let node_of =
        |variable: VariableId| assignments.iter().find(|a| a.variable == variable).map(|a| a.node);
    let treatment_nodes: Vec<DenseNodeId> =
        treatments.iter().map(|&treatment| node_of(treatment)).collect::<Option<_>>()?;
    let outcome_node = node_of(outcome)?;
    let on_path = assignments
        .iter()
        .filter(|a| {
            !treatment_nodes.contains(&a.node)
                && treatment_nodes.iter().any(|&node| graph.reaches(node, a.node))
                && graph.reaches(a.node, outcome_node)
        })
        .collect();
    Some((on_path, treatment_nodes, outcome_node))
}

/// Classify the per-unit effects as structurally equal, empirically equal, or
/// free to vary.
///
/// A unit effect can only vary across units when some mechanism on a directed
/// `treatment → outcome` path bends with that unit's other parent values or its
/// abducted disturbance. When every such mechanism is additive and linear in its
/// parents, the two worlds differ by a fixed composition of slopes: the disturbances
/// cancel and `Y(a) − Y(a0)` is a constant.
///
/// That constant is disclosed as a property of the mechanism only when it *is*
/// one — when no family that could have represented effect modification was
/// successfully scored on any path node (none was in the registry for that node,
/// or every such family failed to fit, e.g. a single-parent node where no
/// cross-parent product exists). When such a family was scored and lost, the
/// data chose additivity, and the verdict records what was rejected instead.
pub(super) fn homogeneity_verdict(
    graph: &Dag,
    assignments: &[crate::gcm::MechanismAssignment],
    treatment: VariableId,
    outcome: VariableId,
) -> HomogeneityVerdict {
    homogeneity_verdict_over(graph, assignments, &[treatment], outcome)
}

/// [`homogeneity_verdict`] for one outcome under several simultaneous treatments: the
/// mechanisms on a directed path from *any* treatment to the outcome are the ones that can
/// modify the effect, so the verdict is structural only when all of them are additively
/// separable.
pub(super) fn homogeneity_verdict_over(
    graph: &Dag,
    assignments: &[crate::gcm::MechanismAssignment],
    treatments: &[VariableId],
    outcome: VariableId,
) -> HomogeneityVerdict {
    let Some((on_path, _, outcome_node)) =
        path_assignments_over(graph, assignments, treatments, outcome)
    else {
        return HomogeneityVerdict::Heterogeneous;
    };
    let mut outcome_family = None;
    let mut rejected = Vec::new();
    for assignment in &on_path {
        if !assignment.fitted.admits_no_effect_modification() {
            return HomogeneityVerdict::Heterogeneous;
        }
        if assignment.node == outcome_node {
            outcome_family = Some(assignment.selected);
        }
        let selected_score = assignment
            .candidates
            .iter()
            .find(|c| c.family == assignment.selected)
            .map_or(f64::NAN, |c| c.score);
        for candidate in assignment.candidates.iter() {
            // `candidates` holds only families that scored; failures live in
            // `failed_families` and never gave the data a say.
            if candidate.family != assignment.selected
                && candidate.family.can_modify_effects()
                && candidate.score.is_finite()
            {
                rejected.push(RejectedHeterogeneity {
                    variable: assignment.variable,
                    family: candidate.family,
                    score: candidate.score,
                    selected: assignment.selected,
                    selected_score,
                });
            }
        }
    }
    match outcome_family {
        None => HomogeneityVerdict::Heterogeneous,
        Some(_) if !rejected.is_empty() => HomogeneityVerdict::Empirical { rejected },
        Some(family) => HomogeneityVerdict::Structural(family),
    }
}

/// Record that equal per-unit effects were chosen by validation score over a
/// family that could have made them differ.
///
/// Info, not Warning: nothing about how the number reads is hidden — the
/// selection is auditable in `gcm.counterfactual.mechanisms`, and this names the
/// rejected alternatives and the score margin so a reader does not have to parse
/// that dump to learn that heterogeneity was tested for.
pub(super) fn heterogeneity_rejected_diagnostic(
    data: &TabularData,
    rejected: &[RejectedHeterogeneity],
) -> Diagnostic {
    let name = |variable: VariableId| {
        data.schema()
            .get(variable)
            .map_or_else(|_| format!("v{}", variable.raw()), |v| v.name.to_string())
    };
    let parts: Vec<String> = rejected
        .iter()
        .map(|r| {
            format!(
                "{}: {} (score {:.6}) lost to {} (score {:.6})",
                name(r.variable),
                r.family.id(),
                r.score,
                r.selected.id(),
                r.selected_score
            )
        })
        .collect();
    let families: Vec<&str> = rejected.iter().map(|r| r.family.id()).collect();
    Diagnostic::new(
        "gcm.counterfactual.heterogeneity_rejected",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "heterogeneity-capable mechanism families were fit and lost on validation score, so \
             equal unit_effects are an empirical finding rather than a property of the mechanism: \
             {}",
            parts.join("; ")
        ),
    )
    .with_fields([("families", families.join(","))])
}
