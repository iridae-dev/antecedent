//! Identification caches of the transport and interference queries over a graph posterior.
//!
//! A `TransportQuery` reads a selection diagram, so every atom of a Dag, Admg or `TemporalAdmg`
//! posterior is run through sID on its own graph with the declared selection targets: the
//! atom is identified when sID certifies a formula, and it is unidentified mass when sID
//! does not. An atom of a Cpdag or Pag posterior names a class of selection diagrams, so it
//! is run through the class envelope of sID (every DAG completion of a Cpdag, every ADMG
//! sharing the m-separation model of a MAG completion of a Pag): the atom is identified when
//! some member certifies a formula, and the members the class leaves uncertified are
//! disclosed on the atom. An `InterferenceQuery` is identified by the known assignment mechanism and the
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
use antecedent_identify::{
    ClassTransportEnvelope, IdentificationResult, TransportIdentification, TransportIdentifier,
};
use antecedent_prob::GraphIdentFlag;

use super::{
    CachedGraphPosteriorAtomIdentification, CachedGraphPosteriorIdentification,
    collect_graph_posterior_atoms, identify_unique_adjacency_masks, map_posterior_graphs,
};
use crate::error::CausalError;

/// Refuse a posterior that carries DBN lag structure: a static exposure query reads the
/// contemporaneous masks alone, which would drop every lagged parent.
fn refuse_lagged_atoms(posterior: &GraphPosterior) -> Result<(), CausalError> {
    if posterior.lag_masks().is_some() || posterior.max_lag().is_some() {
        return Err(CausalError::Unsupported {
            message: "a static transport or interference query over a graph posterior requires \
                      static atoms; this posterior carries DBN lag structure",
        });
    }
    Ok(())
}

/// The selection-diagram carrier of one atom: the atom's ADMG, with a DAG atom read as the
/// ADMG with no bidirected edge.
fn atom_admg(mask: &AdjacencyMask, n_vars: usize, kind: GraphPosteriorAtomKind) -> Option<Admg> {
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
/// the graph class, the diagram does not build, or no formula is certified. Errors propagate.
fn transport_atom(
    mask: &AdjacencyMask,
    n_vars: usize,
    kind: GraphPosteriorAtomKind,
    query: &TransportQuery,
    selection_targets: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<Option<TransportAtom>, CausalError> {
    let Some(admg) = atom_admg(mask, n_vars, kind) else { return Ok(None) };
    let Ok(diagram) = SelectionDiagram::try_new(admg, selection_targets.to_vec()) else {
        return Ok(None);
    };
    let identified = TransportIdentifier::new().identify_with_context(&diagram, query, ctx)?;
    if !matches!(identified, TransportIdentification::Transportable { .. }) {
        return Ok(None);
    }
    let Some((identification, estimand)) =
        crate::Study::transport_posterior_atom(query, &identified)
    else {
        return Ok(None);
    };
    Ok(Some((estimand, identification, Arc::new(identified))))
}

/// Stable code of the disclosure a class atom carries about the members of its class.
pub(crate) const TRANSPORT_CLASS_ATOM_CODE: &str = "identify.graph_posterior.transport_class_atom";

/// The atom a class envelope of sID outcomes states: the first member that certifies a formula
/// the trial-to-target weighting can evaluate stands for the atom (every evaluable member
/// reads the one transported contrast), else the first certified member, else nothing. The
/// members the class leaves uncertified are disclosed on the atom's identification.
fn class_envelope_atom<G>(
    class: &ClassTransportEnvelope<G>,
    query: &TransportQuery,
) -> Option<TransportAtom> {
    let envelope = &class.envelope;
    let certified = |index: usize| {
        crate::analysis::execute::identification_status_ok_for_case(
            envelope.cases[index].result.status,
        ) && matches!(class.proofs[index], TransportIdentification::Transportable { .. })
    };
    if class.proofs.len() != envelope.cases.len() {
        return None;
    }
    let representative = (0..class.proofs.len())
        .find(|&index| {
            certified(index) && crate::analysis::execute::ipw_estimable(&class.proofs[index])
        })
        .or_else(|| (0..class.proofs.len()).find(|&index| certified(index)))?;
    let proof = &class.proofs[representative];
    let (mut identification, estimand) = crate::Study::transport_posterior_atom(query, proof)?;
    // Members that certify different formulas leave the class partially identified, as they
    // do on a supplied class; the weighting reads each formula through its certificate, so the
    // identified set over the evaluated members is the one transported contrast.
    if envelope.status == antecedent_identify::IdentificationStatus::PartiallyIdentified {
        identification.status = antecedent_identify::IdentificationStatus::PartiallyIdentified;
    }
    let evaluable = (0..class.proofs.len())
        .filter(|&index| {
            certified(index) && crate::analysis::execute::ipw_estimable(&class.proofs[index])
        })
        .count();
    identification.diagnostics.push(
        antecedent_core::Diagnostic::new(
            TRANSPORT_CLASS_ATOM_CODE,
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            format!(
                "this atom stands for a class of selection diagrams: {} case(s), {} certified \
                 and {} evaluable by the trial-to-target weighting; identified_weight={}, \
                 unidentified_weight={}; completion weights are enumeration weights, not \
                 posterior probabilities",
                envelope.cases.len(),
                (0..class.proofs.len()).filter(|&index| certified(index)).count(),
                evaluable,
                envelope.identified_weight.0,
                envelope.unidentified_weight.0,
            ),
        )
        .with_fields([
            ("cases", envelope.cases.len().to_string()),
            ("identified_weight", envelope.identified_weight.0.to_string()),
            ("unidentified_weight", envelope.unidentified_weight.0.to_string()),
        ]),
    );
    Some((estimand, identification, Arc::new(proof.clone())))
}

/// Run sID on the class of selection diagrams one Cpdag or Pag atom stands for. `None` when
/// the atom is not a member of the class, the class does not build, or no member certifies a
/// formula.
fn class_transport_atom(
    posterior: &GraphPosterior,
    index: usize,
    query: &TransportQuery,
    selection_targets: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<Option<TransportAtom>, CausalError> {
    let mask = &posterior.atoms()[index];
    let n_vars = posterior.n_vars();
    let identifier = crate::strategy_table::class_identifier_in_context(Some(ctx));
    match posterior.atom_kind() {
        GraphPosteriorAtomKind::Cpdag => {
            let Ok(cpdag) = cpdag_from_adjacency_mask(mask, n_vars) else { return Ok(None) };
            let class =
                identifier.identify_cpdag_transport_envelope(&cpdag, selection_targets, query)?;
            Ok(class_envelope_atom(&class, query))
        }
        GraphPosteriorAtomKind::Pag => {
            let mark = posterior.mark_atom(index);
            let Ok(pag) = pag_from_adjacency_mask(mask, &mark, n_vars) else { return Ok(None) };
            let class =
                identifier.identify_pag_transport_envelope(&pag, selection_targets, query)?;
            Ok(class_envelope_atom(&class, query))
        }
        _ => Ok(None),
    }
}

/// Identify every atom of a transport posterior with sID on its own selection diagram.
///
/// An atom with a certified formula is an identified atom that retains the formula; an atom
/// sID does not certify (a population-varying outcome ancestor in a graph that needs general
/// c-component recursion, or a missing source experiment) stays in the samples as
/// unidentified mass. An atom of a Cpdag or Pag posterior is identified through the class
/// envelope of its selection diagrams, as above.
///
/// # Errors
///
/// A posterior with DBN lag structure or temporal-class atoms, or a cancelled run.
pub(crate) fn build_transport_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &TransportQuery,
    selection_targets: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    refuse_lagged_atoms(posterior)?;
    let kind = posterior.atom_kind();
    if matches!(kind, GraphPosteriorAtomKind::Cpdag | GraphPosteriorAtomKind::Pag) {
        return build_class_transport_cache(posterior, query, selection_targets, ctx);
    }
    if !matches!(
        kind,
        GraphPosteriorAtomKind::Dag
            | GraphPosteriorAtomKind::Admg
            | GraphPosteriorAtomKind::TemporalAdmg
    ) {
        return Err(CausalError::Unsupported {
            message: "TransportQuery over a graph posterior reads a selection diagram per atom, \
                      so its atoms must be Dag, Admg, TemporalAdmg, Cpdag or Pag graphs",
        });
    }
    crate::analysis::execute::report_identify_compute(ctx);
    let n_vars = posterior.n_vars();
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, inner| {
        transport_atom(mask, n_vars, kind, query, selection_targets, inner)
    })?;
    let (graphs, atoms) = collect_graph_posterior_atoms(posterior, ctx, |i, first_for_key| {
        let Some(Some((estimand, identification, transport))) = by_mask.get(&posterior.atoms()[i])
        else {
            return Ok((GraphIdentFlag::Unidentified, None));
        };
        let atom = first_for_key.then(|| CachedGraphPosteriorAtomIdentification {
            key: posterior.graph_keys()[i],
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

/// The transport identification of a Cpdag or Pag posterior: one class envelope of sID outcomes
/// per distinct graph key, the first sample of each key standing for it.
fn build_class_transport_cache(
    posterior: &GraphPosterior,
    query: &TransportQuery,
    selection_targets: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    crate::analysis::execute::report_identify_compute(ctx);
    let mut first_of_key: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut firsts = Vec::new();
    for index in 0..posterior.n_graphs() {
        if let std::collections::hash_map::Entry::Vacant(slot) =
            first_of_key.entry(posterior.graph_keys()[index])
        {
            slot.insert(index);
            firsts.push(index);
        }
    }
    let solved = ctx.map_indexed(firsts.len(), |position, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        class_transport_atom(posterior, firsts[position], query, selection_targets, inner)
    })?;
    let by_key: std::collections::HashMap<u64, Option<TransportAtom>> =
        firsts.iter().map(|&index| posterior.graph_keys()[index]).zip(solved).collect();
    let (graphs, atoms) = collect_graph_posterior_atoms(posterior, ctx, |i, first_for_key| {
        let Some(Some((estimand, identification, transport))) =
            by_key.get(&posterior.graph_keys()[i])
        else {
            return Ok((GraphIdentFlag::Unidentified, None));
        };
        let atom = first_for_key.then(|| CachedGraphPosteriorAtomIdentification {
            key: posterior.graph_keys()[i],
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
    let mask = &posterior.atoms()[index];
    let n_vars = posterior.n_vars();
    match posterior.atom_kind() {
        GraphPosteriorAtomKind::Dag => dag_from_adjacency_mask(mask, n_vars).is_ok(),
        GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => {
            admg_from_adjacency_mask(mask, n_vars).is_ok()
        }
        GraphPosteriorAtomKind::Cpdag => cpdag_from_adjacency_mask(mask, n_vars).is_ok(),
        GraphPosteriorAtomKind::Pag => {
            let mark = posterior.mark_atom(index);
            pag_from_adjacency_mask(mask, &mark, n_vars).is_ok()
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
            key: posterior.graph_keys()[i],
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
