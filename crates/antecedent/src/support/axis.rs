//! Axis naming: which matrix query name and graph-class label a query and a
//! supplied graph occupy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, DerivativeScale, ResponseFunctional, TemporalPolicy};

use crate::accepted::{AcceptedGraph, GraphClass, GraphRef};

/// Matrix query name for `query` on `graph_class`, if the query is on the public axis.
#[must_use]
pub fn query_axis_name(query: &CausalQuery, graph_class: GraphClass) -> Option<&'static str> {
    match query {
        CausalQuery::AverageEffect(_) => Some("AverageEffect"),
        CausalQuery::ConditionalEffect(_) => Some("ConditionalEffect"),
        CausalQuery::Counterfactual(_) => Some("Counterfactual"),
        CausalQuery::NestedCounterfactual(_) => Some("NestedCounterfactualEffect"),
        CausalQuery::Distribution(_) => Some("InterventionalDistribution"),
        CausalQuery::PathSpecific(_) => Some("PathSpecificEffect"),
        CausalQuery::Mediation(_) => match graph_class {
            GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag => {
                Some("TemporalMediationEffect")
            }
            GraphClass::Dag
            | GraphClass::Admg
            | GraphClass::Cpdag
            | GraphClass::Pag
            | GraphClass::TemporalAdmg => Some("MediationEffect"),
            GraphClass::RandomizedTrial => None,
        },
        CausalQuery::TemporalEffect(q) => match &q.policy {
            TemporalPolicy::Pulse { .. } if q.effect_modifiers.is_empty() => Some("PulseEffect"),
            TemporalPolicy::Sustained { .. } if q.effect_modifiers.is_empty() => {
                Some("SustainedEffect")
            }
            TemporalPolicy::Pulse { .. } => Some("TemporalConditionalPulse"),
            TemporalPolicy::Sustained { .. } => Some("TemporalConditionalSustained"),
            // A dynamic rule is classified by its evaluated schedule shape so it
            // cannot dodge the matrix: one active step rides the Pulse cell
            // (the engine estimates it as a rule-tagged pulse), any longer
            // schedule is a sustained intervention and hits the Sustained
            // refusal. Mirrors `refuse_multi_step_schedule` in
            // antecedent-estimate, which already refuses multi-step Dynamic.
            TemporalPolicy::Dynamic { .. } if !q.effect_modifiers.is_empty() => None,
            TemporalPolicy::Dynamic { active_at, .. } => {
                if active_at.len() == 1 {
                    Some("PulseEffect")
                } else {
                    Some("SustainedEffect")
                }
            }
            // `TemporalPolicy` is `#[non_exhaustive]`; every variant that
            // exists is classified above, so this covers only later additions.
            _ => None,
        },
        CausalQuery::Response(q) => {
            let temporal_graph = matches!(
                graph_class,
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            );
            // Static response on a temporal graph (and temporal attachment on a
            // static graph) are typed impossibilities, not matrix cells.
            if q.is_temporal() != temporal_graph {
                return None;
            }
            match &q.functional {
                ResponseFunctional::MeanCurve { .. } => Some("ResponseCurve"),
                ResponseFunctional::AverageDerivative { .. } => Some("AverageDerivative"),
                ResponseFunctional::PointDerivative { scale, .. } => match scale {
                    DerivativeScale::Identity => Some("PointDerivative"),
                    DerivativeScale::LogTreatment | DerivativeScale::LogOutcome => {
                        Some("SemiElasticity")
                    }
                    DerivativeScale::LogLog => Some("Elasticity"),
                },
                ResponseFunctional::DirectionalDerivative { .. } => Some("DirectionalDerivative"),
                ResponseFunctional::Jacobian { .. } => Some("ResponseJacobian"),
                ResponseFunctional::InterventionResponse { .. } => Some("InterventionResponse"),
            }
        }
        CausalQuery::Transport(_) => Some("TransportQuery"),
        CausalQuery::Interference(_) => Some("InterferenceQuery"),
        CausalQuery::AnomalyAttribution(_) => Some("AnomalyAttribution"),
        CausalQuery::ChangeAttribution(_) => Some("ChangeAttribution"),
        // Off the public axis: the mechanism-change and unit-change queries
        // that exist today, and any variant added after this release.
        //
        // This cannot be made exhaustive. `CausalQuery` is `#[non_exhaustive]`,
        // so a downstream crate is required by the
        // compiler to keep a wildcard here, and clippy refuses a named arm
        // beside it with the same body. The wire-side classifier that used to
        // shadow this table is gone — a consumer reads the `query_kind` the
        // producer stamped from here — so there is no second table to drift
        // from; what is left is that a new query kind lands off-axis silently.
        // `parity/support_axes.toml` is the list that says which names the
        // matrix expects, and `gate_parity_schema.sh` checks it.
        _ => None,
    }
}

pub(super) fn naming_class(matrix: &str) -> Option<GraphClass> {
    Some(match matrix {
        "Dag" | "CoDetermined" | "Unknown" => GraphClass::Dag,
        "Admg" => GraphClass::Admg,
        "Cpdag" => GraphClass::Cpdag,
        "Pag" => GraphClass::Pag,
        "TemporalDag" => GraphClass::TemporalDag,
        "TemporalCpdag" => GraphClass::TemporalCpdag,
        "TemporalPag" => GraphClass::TemporalPag,
        "TemporalAdmg" => GraphClass::TemporalAdmg,
        _ => return None,
    })
}

/// The graph class the engine actually dispatches on, for support-matrix
/// classification. Collapsing a non-Dag class to `Dag`/`TemporalDag` here is
/// classification only — it never changes what `Study::compile`/`execute` or
/// `identify_with` run; those still match on `AcceptedGraph::class()` directly.
/// This exists so the cell an honest caller sees matches the engine's real
/// behavior instead of the raw class tag, per each collapse's one-line
/// justification below. Each collapse is scoped to exactly the query family
/// `analysis/execute/compile.rs` wires it for — applying it more broadly would
/// license cells the engine cannot actually run (see the `ResponseCurve` note).
///
/// - **ADMG with no bidirected edges, under `AverageEffect`**: `compile.rs`'s
///   `(AverageEffect, GraphClass::Admg)` arm and `dispatch.rs`'s
///   `GraphClass::Admg` arm both branch on [`Admg::has_bidirected`](antecedent_graph::Admg::has_bidirected) and run the
///   *static DAG* path when it is false — the Dag cell's license is the
///   honest claim. Bidirected `CausalQuery::Response` now compiles through
///   `general.id` + `functional.effect`; a DAG-coerced ADMG (no bidirected
///   edges) still falls through to the Dag response path. The collapse must
///   not fire for a supplied bidirected Admg response. `CoDetermined` /
///   `Unknown` are classification-only matrix extras ([`matrix_graph_class`]),
///   not [`GraphClass`] variants.
/// - **Cpdag, under `AverageEffect`**: undirected marks are MEC information.
///   Completing a CPDAG to a DAG is a `Dag` cell only when the *caller*
///   supplies a `Dag`. A supplied `Cpdag` stays `Cpdag` even when it has a
///   unique orientation.
/// - **`TemporalCpdag` / `TemporalPag`**: supplied classes remain class-shaped,
///   including fully oriented inputs. A directed MAG edge is not a DAG edge:
///   visibility must still be certified before adjustment.
///
/// Not collapsed: a static [`Pag`](antecedent_graph::Pag)'s circle marks are information the
/// class-aware generalized-adjustment identifier is built to consume, never
/// incompleteness ([`AcceptedGraph::pag`] is infallible for exactly this
/// reason), so there is no Dag-equivalent Pag case.
#[must_use]
pub(crate) fn effective_graph_class(graph: &AcceptedGraph, query: &CausalQuery) -> GraphClass {
    match (graph.graph_ref(), query) {
        (GraphRef::Admg(admg), CausalQuery::AverageEffect(_)) => {
            if admg.has_bidirected() {
                GraphClass::Admg
            } else {
                GraphClass::Dag
            }
        }
        (GraphRef::TemporalAdmg(temporal), CausalQuery::AverageEffect(_)) => {
            match temporal.project_preserving_functional() {
                Ok(admg) if admg.has_bidirected() => GraphClass::TemporalAdmg,
                Ok(_) => GraphClass::Dag,
                Err(_) => GraphClass::TemporalAdmg,
            }
        }
        (GraphRef::Cpdag(_), CausalQuery::AverageEffect(_)) => GraphClass::Cpdag,
        (graph, _) => graph.class(),
    }
}

/// Matrix graph axis, including classification-only tier-rule extras.
///
/// A `TieredBackground` is not a supplied `Admg`/`Pag`. `CoDetermined` is a known
/// closure; Unknown is a two-scenario envelope. Classification only — execute
/// still materializes the closure ADMG or PAG.
#[must_use]
pub(crate) fn matrix_graph_class(
    graph: &AcceptedGraph,
    query: &CausalQuery,
    tiered: Option<&antecedent_graph::TieredBackground>,
) -> &'static str {
    match tiered.map(|b| b.within_tier) {
        Some(antecedent_graph::WithinTier::CoDetermined) => "CoDetermined",
        Some(antecedent_graph::WithinTier::Unknown) => "Unknown",
        None => effective_graph_class(graph, query).as_str(),
    }
}
