//! Identification caches of the transport and interference queries over a graph posterior.
//!
//! A `TransportQuery` reads a selection diagram, so every atom of a Dag, Admg or TemporalAdmg
//! posterior is run through sID on its own graph with the declared selection targets: the
//! atom is identified when sID certifies a formula, and it is unidentified mass when sID
//! does not. An `InterferenceQuery` is identified by the known assignment mechanism and the
//! exposure mapping, so no edge of any atom enters the identification: every atom that is a
//! member of the posterior's graph class carries the one design identification, and an atom
//! that is not a member (a cyclic mask, an unreadable mark) is unidentified mass.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, InterferenceQuery, TransportQuery, VariableId};
use antecedent_discovery::adjacency_mask::{
    admg_from_adjacency_mask, cpdag_from_adjacency_mask, dag_from_adjacency_mask,
    pag_from_adjacency_mask,
};
use antecedent_discovery::{AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{Admg, SelectionDiagram};
use antecedent_identify::{IdentificationResult, TransportIdentification, TransportIdentifier};
use antecedent_prob::GraphIdentFlag;

use super::{
    CachedGraphPosteriorAtomIdentification, CachedGraphPosteriorIdentification,
    collect_graph_posterior_atoms, identify_unique_adjacency_masks, map_posterior_graphs,
};
use crate::error::CausalError;

/// Refuse a posterior that carries DBN lag structure: a static exposure query reads the
/// contemporaneous masks alone, which would drop every lagged parent.
fn refuse_lagged_atoms(posterior: &GraphPosterior) -> Result<(), CausalError> {
    if posterior.lag_masks.is_some() || posterior.max_lag.is_some() {
        return Err(CausalError::Unsupported {
            message: "a static transport or interference query over a graph posterior requires \
                      static atoms; this posterior carries DBN lag structure",
        });
    }
    Ok(())
}

/// The selection-diagram carrier of one atom: the atom's ADMG, with a DAG atom read as the
/// ADMG with no bidirected edge.
fn atom_admg(mask: AdjacencyMask, n_vars: usize, kind: GraphPosteriorAtomKind) -> Option<Admg> {
    match kind {
        GraphPosteriorAtomKind::Dag => {
            let dag = dag_from_adjacency_mask(mask, n_vars).ok()?;
            Admg::from_dag(&dag).ok()
        }
        GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => {
            admg_from_adjacency_mask(mask, n_vars).ok()
        }
        _ => None,
    }
}

/// One atom's sID proof: the certified formula and the identification it states.
type TransportAtom = (IdentifiedEstimand, IdentificationResult, Arc<TransportIdentification>);

/// Run sID on the selection diagram of one atom. `None` when the atom is not a member of
/// the graph class, the diagram does not build, sID errors, or no formula is certified.
fn transport_atom(
    mask: AdjacencyMask,
    n_vars: usize,
    kind: GraphPosteriorAtomKind,
    query: &TransportQuery,
    selection_targets: &[VariableId],
) -> Option<TransportAtom> {
    let admg = atom_admg(mask, n_vars, kind)?;
    let diagram = SelectionDiagram::try_new(admg, selection_targets.to_vec()).ok()?;
    let identified = TransportIdentifier::new().identify(&diagram, query).ok()?;
    if !matches!(identified, TransportIdentification::Transportable { .. }) {
        return None;
    }
    let (identification, estimand) = crate::Study::transport_posterior_atom(query, &identified)?;
    Some((estimand, identification, Arc::new(identified)))
}

/// Identify every atom of a Dag, Admg or TemporalAdmg posterior with sID on its own
/// selection diagram.
///
/// An atom with a certified formula is an identified atom that retains the formula; an atom
/// sID does not certify (a population-varying outcome ancestor in a graph that needs general
/// c-component recursion, or a missing source experiment) stays in the samples as
/// unidentified mass.
///
/// # Errors
///
/// A posterior with DBN lag structure or class atoms, or a cancelled run.
pub(crate) fn build_transport_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &TransportQuery,
    selection_targets: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    refuse_lagged_atoms(posterior)?;
    if !matches!(
        posterior.atom_kind,
        GraphPosteriorAtomKind::Dag
            | GraphPosteriorAtomKind::Admg
            | GraphPosteriorAtomKind::TemporalAdmg
    ) {
        return Err(CausalError::Unsupported {
            message: "TransportQuery over a graph posterior reads a selection diagram per atom, \
                      so its atoms must be Dag, Admg or TemporalAdmg graphs",
        });
    }
    crate::analysis::execute::report_identify_compute(ctx);
    let n_vars = posterior.n_vars;
    let kind = posterior.atom_kind;
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, _inner| {
        Ok::<_, CausalError>(transport_atom(mask, n_vars, kind, query, selection_targets))
    })?;
    let (graphs, atoms) = collect_graph_posterior_atoms(posterior, ctx, |i, first_for_key| {
        let Some(Some((estimand, identification, transport))) = by_mask.get(&posterior.atoms[i])
        else {
            return Ok((GraphIdentFlag::Unidentified, None));
        };
        let atom = first_for_key.then(|| CachedGraphPosteriorAtomIdentification {
            key: posterior.graph_keys[i],
            estimand: estimand.clone(),
            identification: identification.clone(),
            transport: Some(Arc::clone(transport)),
        });
        Ok((GraphIdentFlag::Identified, atom))
    })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}

/// Whether sample `index` of `posterior` is a member of the posterior's graph class: a DAG
/// mask that is acyclic, an ADMG mask, a CPDAG mask, or a PAG mask with its marks.
fn atom_is_member(posterior: &GraphPosterior, index: usize) -> bool {
    let mask = posterior.atoms[index];
    let n_vars = posterior.n_vars;
    match posterior.atom_kind {
        GraphPosteriorAtomKind::Dag => dag_from_adjacency_mask(mask, n_vars).is_ok(),
        GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => {
            admg_from_adjacency_mask(mask, n_vars).is_ok()
        }
        GraphPosteriorAtomKind::Cpdag => cpdag_from_adjacency_mask(mask, n_vars).is_ok(),
        GraphPosteriorAtomKind::Pag => {
            let mark =
                posterior.mark_atoms.as_ref().map_or(AdjacencyMask::EMPTY, |marks| marks[index]);
            pag_from_adjacency_mask(mask, mark, n_vars).is_ok()
        }
        _ => false,
    }
}

/// The interference design identification for every member atom of a posterior of any
/// static atom kind.
///
/// The exposure contrast is identified by the known assignment mechanism and the exposure
/// mapping on the fixed network, so the identification is the same for every atom: the
/// posterior moves no edge into it. An atom that is not a member of the graph class is
/// unidentified mass.
///
/// # Errors
///
/// A posterior with DBN lag structure, or a cancelled run.
pub(crate) fn build_interference_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &InterferenceQuery,
    inference: &crate::InferenceMode,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    refuse_lagged_atoms(posterior)?;
    crate::analysis::execute::report_identify_compute(ctx);
    let (identification, estimand) =
        crate::analysis::execute::CheckedInterferenceOperation::design_identification(
            query, inference,
        );
    let member = map_posterior_graphs(posterior, ctx, |index, _inner| {
        Ok::<_, CausalError>(atom_is_member(posterior, index))
    })?;
    let (graphs, atoms) = collect_graph_posterior_atoms(posterior, ctx, |i, first_for_key| {
        if !member[i] {
            return Ok((GraphIdentFlag::Unidentified, None));
        }
        let atom = first_for_key.then(|| CachedGraphPosteriorAtomIdentification {
            key: posterior.graph_keys[i],
            estimand: estimand.clone(),
            identification: identification.clone(),
            transport: None,
        });
        Ok((GraphIdentFlag::Identified, atom))
    })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}
