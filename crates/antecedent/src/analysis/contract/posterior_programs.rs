//! Per-atom programs of a functional result answered over a graph posterior.
//!
//! A path-restricted effect, a natural direct effect or an interventional distribution over a
//! graph posterior is answered on every distinct graph by the explicit executor of that
//! graph's structure and mixed by posterior weight, so no single program reproduces it. The
//! contract instead binds what each atom was answered by: the checked program of a DAG or ADMG
//! atom, or the identification product of a class atom, with the atom's weight, the posterior
//! samples it aggregates and the mixing rule, under one digest chain an independent consumer
//! recomputes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{CausalQuery, ExecutionContext};
use antecedent_discovery::adjacency_mask::{admg_from_adjacency_mask, dag_from_adjacency_mask};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_graph::Admg;
use antecedent_io::{GraphPosteriorAtomProgramWire, GraphPosteriorProgramsWire};

use crate::EstimatorId;
use crate::accepted::AcceptedGraph;
use crate::error::CausalError;
use crate::support::StructureSource;

use super::super::builder::{DataInput, RefuteSuite};
use super::super::execute::Study;
use super::program_wire;
use super::{CheckedProgramBinding, contract_identification, identification_product_wire, io_err};

/// Whether `study` is a static graph posterior answering a functional effect or distribution
/// per atom.
fn answers_per_atom(study: &Study, resolved_estimator: Option<&str>) -> bool {
    study.graph_posterior.as_ref().is_some_and(|posterior| {
        posterior.lag_masks.is_none()
            && posterior.max_lag.is_none()
            && matches!(study.data, DataInput::Tabular(_))
    }) && match (&study.query, resolved_estimator) {
        (
            CausalQuery::PathSpecific(_) | CausalQuery::NestedCounterfactual(_),
            Some("functional.effect"),
        )
        | (CausalQuery::Distribution(_), Some("functional.distribution")) => true,
        _ => false,
    }
}

/// The accepted structure of one single-structure posterior sample, `None` when it is not a
/// member of the posterior's graph class. A natural direct effect on a DAG atom is answered
/// through the mixed-graph executor, so its structure is the DAG read as an ADMG.
fn atom_structure(posterior: &GraphPosterior, index: usize, nested: bool) -> Option<AcceptedGraph> {
    let mask = posterior.atoms[index];
    let n_vars = posterior.n_vars;
    match posterior.atom_kind {
        GraphPosteriorAtomKind::Dag => {
            let dag = dag_from_adjacency_mask(mask, n_vars).ok()?;
            if nested {
                Some(AcceptedGraph::admg(Admg::from_dag(&dag).ok()?))
            } else {
                Some(AcceptedGraph::dag(dag))
            }
        }
        GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => {
            Some(AcceptedGraph::admg(admg_from_adjacency_mask(mask, n_vars).ok()?))
        }
        _ => None,
    }
}

/// The program or class product one atom's explicit cell is answered by. An atom whose
/// identifier refuses the query carries neither.
fn atom_answer(
    study: &Study,
    posterior: &GraphPosterior,
    index: usize,
    selected: EstimatorId,
    ctx: &ExecutionContext,
) -> Result<
    (
        Option<antecedent_io::FunctionalProgramWire>,
        Option<antecedent_io::IdentificationProductWire>,
    ),
    CausalError,
> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
    }
    // A class is answered per completion: its identification product states which
    // completions identified and how the class weight splits between them.
    if matches!(posterior.atom_kind, GraphPosteriorAtomKind::Cpdag | GraphPosteriorAtomKind::Pag) {
        crate::analysis::execute::report_identify_compute(ctx);
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        let class = study.class_atom_identification(posterior, index)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        let Some(class) = class else {
            return Ok((None, None));
        };
        let mut product =
            identification_product_wire(&class.result, class.truncated_completions > 0)
                .map_err(|err| io_err(&err))?;
        product.envelope = Some(antecedent_io::IdentificationEnvelopeWire {
            cases: class.cases as u64,
            identified_weight_bits: class.identified_weight.to_bits(),
            unidentified_weight_bits: class.unidentified_weight.to_bits(),
            truncated_completions: class.truncated_completions as u64,
        });
        return Ok((None, Some(product)));
    }
    let nested = matches!(study.query, CausalQuery::NestedCounterfactual(_));
    let Some(graph) = atom_structure(posterior, index, nested) else {
        return Ok((None, None));
    };
    let mut atom = study.clone();
    atom.graph_posterior = None;
    atom.graph = Arc::new(graph);
    atom.structure_source = StructureSource::Explicit;
    atom.refute = RefuteSuite::None;
    atom.custom_validators.clear();
    // The atom's contract needs its compiled identification, not a sampled-data
    // execution handle. Full preparation would also scan empirical factors and
    // bind their providers, making contract inspection perform hidden execution.
    let plan = match atom.compile(ctx) {
        Ok(plan) => plan,
        Err(error) if error.is_identification_refusal() => return Ok((None, None)),
        Err(error) => return Err(error),
    };
    let cell = match atom.with_identification_caches(&plan, ctx) {
        Ok(cell) => cell,
        Err(error) if error.is_identification_refusal() => return Ok((None, None)),
        Err(error) => return Err(error),
    };
    let Some(cached) = contract_identification(&cell) else {
        return Ok((None, None));
    };
    let program = program_wire::functional_program(
        CheckedProgramBinding::None,
        &cell,
        Some(&cached),
        Some(selected),
    )?;
    Ok((program, None))
}

/// The sealed per-atom programs of `study`, or `None` when it does not answer per atom.
///
/// Atoms are the distinct graphs of the posterior in order of first appearance; an atom's
/// weight is the sum of its samples'.
///
/// # Errors
///
/// An atom's explicit cell fails to prepare for a reason other than identification, or the
/// payload cannot be encoded.
pub(crate) fn graph_posterior_programs(
    study: &Study,
    resolved_estimator: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<Option<GraphPosteriorProgramsWire>, CausalError> {
    if !answers_per_atom(study, resolved_estimator) {
        return Ok(None);
    }
    let Some(posterior) = study.graph_posterior.as_ref() else {
        return Ok(None);
    };
    let selected = if matches!(study.query, CausalQuery::Distribution(_)) {
        EstimatorId::FunctionalDistribution
    } else {
        EstimatorId::FunctionalEffect
    };
    let mut order: Vec<u64> = Vec::new();
    let mut samples: HashMap<u64, Vec<usize>> = HashMap::new();
    for index in 0..posterior.n_graphs {
        let key = posterior.graph_keys[index];
        samples
            .entry(key)
            .or_insert_with(|| {
                order.push(key);
                Vec::new()
            })
            .push(index);
    }
    let mut atoms = Vec::with_capacity(order.len());
    for key in order {
        let members = &samples[&key];
        let weight: f64 = members.iter().map(|&index| posterior.weights[index]).sum();
        let (functional_program, identification_product) =
            atom_answer(study, posterior, members[0], selected, ctx)?;
        atoms.push(GraphPosteriorAtomProgramWire {
            graph_key: key,
            weight_bits: weight.to_bits(),
            samples: members.iter().map(|&index| index as u64).collect(),
            functional_program,
            identification_product,
        });
    }
    GraphPosteriorProgramsWire::seal(atoms).map(Some).map_err(|err| io_err(&err))
}
