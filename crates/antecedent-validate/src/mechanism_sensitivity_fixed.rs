// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fixed-graph transport mechanism sensitivity.

use super::{
    BTreeMap, BTreeSet, BoundTransportFunctional, ClassicalTransportQuery, DenseNodeId,
    DiscreteKernelSensitivity, DiscreteKernelSensitivityError, Error, EvidenceKind, ExprNode,
    FixedGraphConditionalMechanismSensitivityResult, FixedGraphMechanismSensitivityResult,
    FixedGraphMechanismSensitivitySpec, FixedGraphRootMechanismSensitivityResult, NodeRef,
    RegimeId, SelectionDiagram, SidLimits, VariableId, verify_classical_transport,
};

/// Typed refusal for the fixed-graph sensitivity route.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FixedGraphSensitivityError {
    /// Query does not name one outcome and one treatment parent.
    #[error("requires one outcome and one treatment parent")]
    InvalidQuery,
    /// Graph is outside the fixed fully observed DAG contract.
    #[error("requires a fixed fully observed discrete DAG")]
    UnsupportedGraph,
    /// Checked transport proof does not match the supplied query and graph.
    #[error("transport proof does not check against the declared graph and query")]
    InvalidTransportProof,
    /// Checked expression does not use the declared source outcome factor.
    #[error("checked transport expression does not bind the declared source outcome factor")]
    UnboundOutcomeFactor,
    /// Numerical snapshots do not match available catalog bindings.
    #[error("declared numerical snapshot does not match an available catalog binding")]
    SnapshotMismatch,
    /// Parent support, kernel, or target laws are incomplete or invalid.
    #[error("kernel or target parent laws are incomplete or invalid")]
    InvalidParentLaw,
    /// The finite kernel calculation rejected its parameters.
    #[error("{0}")]
    Kernel(DiscreteKernelSensitivityError),
}

/// Maximum parent-stratum product support for the explicit discrete route.
pub(crate) const FIXED_GRAPH_SENSITIVITY_MAX_STRATA: usize = 1_000_000;

/// Evaluate outcome-kernel contamination around a checked transport result.
///
/// The function verifies the derivation, fixed graph, source factor, catalog
/// regimes, and snapshot identities. It derives signed stratum weights as target
/// active-arm parent mass minus control-arm parent mass.
///
/// # Errors
/// Returns a typed refusal for unsupported graphs, mismatched proofs or bindings,
/// incomplete finite support, and invalid probabilities.
// This staged validation derives the target contrast from checked graph,
// kernel, population, and snapshot evidence before optimizing the range.
pub fn fixed_graph_mechanism_sensitivity(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    functional: &BoundTransportFunctional,
    spec: &FixedGraphMechanismSensitivitySpec,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<FixedGraphMechanismSensitivityResult, FixedGraphSensitivityError> {
    fixed_graph_mechanism_sensitivity_inner(diagram, query, functional, spec, None, ctx)
}

/// Evaluate one-factor contamination restricted to a single treatment-level
/// slice of the checked outcome mechanism; the other arm remains fixed.
///
/// This additive entry point preserves the original sensitivity spec and
/// function contract for callers that use shared-across-arms contamination.
pub fn fixed_graph_treatment_level_sensitivity(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    functional: &BoundTransportFunctional,
    spec: &FixedGraphMechanismSensitivitySpec,
    perturbed_treatment_level: usize,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<FixedGraphMechanismSensitivityResult, FixedGraphSensitivityError> {
    fixed_graph_mechanism_sensitivity_inner(
        diagram,
        query,
        functional,
        spec,
        Some(perturbed_treatment_level),
        ctx,
    )
}

fn static_dense(
    graph: &antecedent_graph::Admg,
    variable: VariableId,
) -> Result<DenseNodeId, FixedGraphSensitivityError> {
    graph
        .nodes()
        .iter()
        .position(|node| *node == NodeRef::Static(variable))
        .and_then(|i| u32::try_from(i).ok())
        .map(DenseNodeId::from_raw)
        .ok_or(FixedGraphSensitivityError::UnsupportedGraph)
}

/// Parents, strata, and cardinalities of one checked outcome mechanism.
struct OutcomeParentFrame {
    outcome: VariableId,
    parents: Vec<VariableId>,
    treatment_position: usize,
    other_positions: Vec<usize>,
    other_cardinalities: Vec<usize>,
    other_strata: Vec<Vec<usize>>,
    all_strata: Vec<Vec<usize>>,
}

struct KernelContaminationInput<'a> {
    frame: &'a OutcomeParentFrame,
    query: &'a ClassicalTransportQuery,
    functional: &'a BoundTransportFunctional,
    spec: &'a FixedGraphMechanismSensitivitySpec,
    kernels: &'a BTreeMap<Vec<usize>, Vec<f64>>,
    weights: &'a [f64],
    indices: &'a BTreeMap<Vec<usize>, usize>,
    perturbed_treatment_level: Option<usize>,
}

fn fixed_graph_mechanism_sensitivity_inner(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    functional: &BoundTransportFunctional,
    spec: &FixedGraphMechanismSensitivitySpec,
    perturbed_treatment_level: Option<usize>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<FixedGraphMechanismSensitivityResult, FixedGraphSensitivityError> {
    let frame = outcome_parent_frame(diagram, query, functional, spec, ctx)?;
    let kernels = source_outcome_kernels(&frame, spec)?;
    let contrast = contrast_weights(&frame, spec)?;
    agree_parent_laws(&frame, spec, &contrast.control, &contrast.active)?;
    evaluate_kernel_contamination(&KernelContaminationInput {
        frame: &frame,
        query,
        functional,
        spec,
        kernels: &kernels,
        weights: &contrast.weights,
        indices: &contrast.indices,
        perturbed_treatment_level,
    })
}

fn outcome_parent_frame(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    functional: &BoundTransportFunctional,
    spec: &FixedGraphMechanismSensitivitySpec,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<OutcomeParentFrame, FixedGraphSensitivityError> {
    if query.outcomes.len() != 1 || query.treatments.len() != 1 {
        return Err(FixedGraphSensitivityError::InvalidQuery);
    }
    let outcome = query.outcomes[0];
    let treatment = query.treatments[0];
    let graph = diagram.causal_graph();
    if functional.derivation().query() != query
        || graph.has_bidirected()
        || graph.nodes().iter().any(|node| !matches!(node, NodeRef::Static(_)))
    {
        return Err(FixedGraphSensitivityError::UnsupportedGraph);
    }
    verify_classical_transport(diagram, query, functional.derivation(), SidLimits::default(), ctx)
        .map_err(|_| FixedGraphSensitivityError::InvalidTransportProof)?;
    let dense = static_dense(graph, outcome)?;
    let treatment_dense = static_dense(graph, treatment)?;
    let mut parents: Vec<_> = graph
        .parents(dense)
        .iter()
        .map(|parent| match graph.nodes()[parent.as_usize()] {
            NodeRef::Static(variable) => Ok(variable),
            _ => Err(FixedGraphSensitivityError::UnsupportedGraph),
        })
        .collect::<Result<_, _>>()?;
    parents.sort_unstable();
    if diagram.selection_targets().iter().any(|selected| {
        *selected != outcome && *selected != treatment && parents.contains(selected)
    }) {
        return Err(FixedGraphSensitivityError::UnsupportedGraph);
    }
    if !parents.contains(&treatment)
        || spec.parent_cardinalities.len() != parents.len()
        || spec.outcome_values.is_empty()
        || spec.treatment_levels[0] == spec.treatment_levels[1]
        || spec.source_kernel_snapshot.trim().is_empty()
        || spec.target_parent_snapshot.trim().is_empty()
    {
        return Err(FixedGraphSensitivityError::InvalidQuery);
    }
    for (i, parent) in parents.iter().enumerate() {
        let cardinality = spec.parent_cardinalities[i];
        if cardinality == 0
            || (*parent == treatment
                && spec.treatment_levels.iter().any(|level| *level >= cardinality))
        {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
    }

    let other_positions: Vec<_> = parents
        .iter()
        .enumerate()
        .filter_map(|(i, parent)| (*parent != treatment).then_some(i))
        .collect();
    if parents.iter().filter(|parent| **parent != treatment).any(|parent| {
        let parent_dense = static_dense(graph, *parent).ok();
        parent_dense.is_some_and(|parent_dense| graph.reaches(treatment_dense, parent_dense))
    }) {
        return Err(FixedGraphSensitivityError::UnsupportedGraph);
    }
    let other_cardinalities: Vec<_> =
        other_positions.iter().map(|i| spec.parent_cardinalities[*i]).collect();
    let other_strata = cartesian_levels(&other_cardinalities)?;
    let treatment_position = parents
        .iter()
        .position(|parent| *parent == treatment)
        .ok_or(FixedGraphSensitivityError::InvalidQuery)?;
    let mut all_strata = cartesian_levels(&spec.parent_cardinalities)?;
    all_strata.retain(|stratum| spec.treatment_levels.contains(&stratum[treatment_position]));
    Ok(OutcomeParentFrame {
        outcome,
        parents,
        treatment_position,
        other_positions,
        other_cardinalities,
        other_strata,
        all_strata,
    })
}

fn source_outcome_kernels(
    frame: &OutcomeParentFrame,
    spec: &FixedGraphMechanismSensitivitySpec,
) -> Result<BTreeMap<Vec<usize>, Vec<f64>>, FixedGraphSensitivityError> {
    let mut kernels = BTreeMap::<Vec<usize>, Vec<f64>>::new();
    for row in &spec.source_kernel {
        if row.parent_levels.len() != frame.parents.len()
            || row.parent_levels.iter().zip(&spec.parent_cardinalities).any(|(x, n)| x >= n)
            || row.outcome_probabilities.len() != spec.outcome_values.len()
            || row.outcome_probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
            || (row.outcome_probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-10
            || kernels
                .insert(row.parent_levels.clone(), row.outcome_probabilities.clone())
                .is_some()
        {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
    }
    if kernels.len() != frame.all_strata.len()
        || frame.all_strata.iter().any(|stratum| !kernels.contains_key(stratum))
    {
        return Err(FixedGraphSensitivityError::InvalidParentLaw);
    }
    Ok(kernels)
}

struct ContrastWeights {
    weights: Vec<f64>,
    indices: BTreeMap<Vec<usize>, usize>,
    control: BTreeMap<Vec<usize>, f64>,
    active: BTreeMap<Vec<usize>, f64>,
}

fn contrast_weights(
    frame: &OutcomeParentFrame,
    spec: &FixedGraphMechanismSensitivitySpec,
) -> Result<ContrastWeights, FixedGraphSensitivityError> {
    let mut target_arms = BTreeMap::<usize, BTreeMap<Vec<usize>, f64>>::new();
    for row in &spec.target_parent_law {
        if !spec.treatment_levels.contains(&row.treatment_level)
            || row.parent_levels.len() != frame.other_positions.len()
            || row.parent_levels.iter().zip(&frame.other_cardinalities).any(|(x, n)| x >= n)
            || !row.probability.is_finite()
            || row.probability < 0.0
        {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
        if target_arms
            .entry(row.treatment_level)
            .or_default()
            .insert(row.parent_levels.clone(), row.probability)
            .is_some()
        {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
    }
    let indices: BTreeMap<_, _> = frame
        .all_strata
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, stratum)| (stratum, index))
        .collect();
    let mut weights = vec![0.0; frame.all_strata.len()];
    for (arm_index, treatment_level) in spec.treatment_levels.iter().copied().enumerate() {
        let arm = target_arms
            .get(&treatment_level)
            .ok_or(FixedGraphSensitivityError::InvalidParentLaw)?;
        if arm.len() != frame.other_strata.len()
            || frame.other_strata.iter().any(|stratum| !arm.contains_key(stratum))
            || (arm.values().sum::<f64>() - 1.0).abs() > 1e-10
        {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
        for stratum in &frame.other_strata {
            let mut full = vec![0; frame.parents.len()];
            for (position, level) in frame.other_positions.iter().zip(stratum) {
                full[*position] = *level;
            }
            full[frame.treatment_position] = treatment_level;
            let index = indices.get(&full).ok_or(FixedGraphSensitivityError::InvalidParentLaw)?;
            weights[*index] += if arm_index == 0 { -arm[stratum] } else { arm[stratum] };
        }
    }
    let control = target_arms
        .remove(&spec.treatment_levels[0])
        .ok_or(FixedGraphSensitivityError::InvalidParentLaw)?;
    let active = target_arms
        .remove(&spec.treatment_levels[1])
        .ok_or(FixedGraphSensitivityError::InvalidParentLaw)?;
    Ok(ContrastWeights { weights, indices, control, active })
}

fn agree_parent_laws(
    frame: &OutcomeParentFrame,
    spec: &FixedGraphMechanismSensitivitySpec,
    control: &BTreeMap<Vec<usize>, f64>,
    active: &BTreeMap<Vec<usize>, f64>,
) -> Result<(), FixedGraphSensitivityError> {
    let mut source_parent_law = BTreeMap::<Vec<usize>, f64>::new();
    for row in &spec.source_parent_law {
        if row.parent_levels.len() != frame.other_positions.len()
            || row.parent_levels.iter().zip(&frame.other_cardinalities).any(|(x, n)| x >= n)
            || !row.probability.is_finite()
            || row.probability < 0.0
            || source_parent_law.insert(row.parent_levels.clone(), row.probability).is_some()
        {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
    }
    if source_parent_law.len() != frame.other_strata.len()
        || frame.other_strata.iter().any(|stratum| !source_parent_law.contains_key(stratum))
        || (source_parent_law.values().sum::<f64>() - 1.0).abs() > 1e-10
        || frame.other_strata.iter().any(|stratum| {
            (control[stratum] - active[stratum]).abs() > 1e-10
                || (control[stratum] - source_parent_law[stratum]).abs() > 1e-10
        })
    {
        return Err(FixedGraphSensitivityError::InvalidParentLaw);
    }
    Ok(())
}

fn evaluate_kernel_contamination(
    input: &KernelContaminationInput<'_>,
) -> Result<FixedGraphMechanismSensitivityResult, FixedGraphSensitivityError> {
    let KernelContaminationInput {
        frame,
        query,
        functional,
        spec,
        kernels,
        weights,
        indices,
        perturbed_treatment_level,
    } = input;
    let outcome = frame.outcome;
    validate_sensitivity_bindings(functional, query, outcome, &frame.parents, spec)?;
    if !expression_uses_source_outcome_factor(functional, query, outcome, spec.source_kernel_regime)
    {
        return Err(FixedGraphSensitivityError::UnboundOutcomeFactor);
    }
    let mut sensitivity_weights = weights.to_vec();
    let mut fixed_contribution = 0.0;
    if let Some(perturbed_level) = perturbed_treatment_level {
        if !spec.treatment_levels.contains(perturbed_level) {
            return Err(FixedGraphSensitivityError::InvalidQuery);
        }
        for (stratum, weight) in frame.all_strata.iter().zip(&mut sensitivity_weights) {
            if stratum[frame.treatment_position] != *perturbed_level {
                *weight = 0.0;
                let mean = kernels[stratum]
                    .iter()
                    .zip(&spec.outcome_values)
                    .map(|(probability, value)| probability * value)
                    .sum::<f64>();
                fixed_contribution += weights[indices[stratum]] * mean;
            }
        }
    }
    let mut response = DiscreteKernelSensitivity {
        source_kernel: frame.all_strata.iter().map(|s| kernels[s].clone()).collect(),
        outcome_values: spec.outcome_values.clone(),
        stratum_contrast_weights: sensitivity_weights,
        max_fraction: spec.max_fraction,
        decision_threshold: spec.decision_threshold.map(|threshold| threshold - fixed_contribution),
    }
    .evaluate()
    .map_err(FixedGraphSensitivityError::Kernel)?;
    response.baseline += fixed_contribution;
    response.minimum += fixed_contribution;
    response.maximum += fixed_contribution;
    Ok(FixedGraphMechanismSensitivityResult {
        estimand: format!(
            "target active-minus-control mean response for {outcome} under source-to-target kernel contamination{}",
            perturbed_treatment_level
                .map(|level| format!(" at treatment level {level}"))
                .unwrap_or_default()
        ),
        outcome,
        parents: frame.parents.clone(),
        source_population: query.source.to_string(),
        target_population: query.target.to_string(),
        source_kernel_binding: (spec.source_kernel_regime, spec.source_kernel_snapshot.clone()),
        target_parent_binding: (spec.target_parent_regime, spec.target_parent_snapshot.clone()),
        assumptions: vec![
            "fixed fully observed DAG; no latent bidirected edges".into(),
            if perturbed_treatment_level.is_some() {
                "only one declared treatment-level slice of the discrete outcome mechanism is contaminated; the other response arm is fixed".into()
            } else {
                "only the discrete outcome mechanism is contaminated".into()
            },
            "all other outcome-parent mechanisms are invariant across populations".into(),
            "source and target non-treatment outcome-parent laws agree under both interventions"
                .into(),
        ],
        response,
    })
}

/// Evaluate bounded contamination of one isolated root-parent mechanism.
///
/// The existing checked contract supplies complete source outcome kernels and
/// source/target parent laws. This route additionally requires the named
/// mechanism to be a root, to have no directed path to another non-treatment
/// outcome parent, and to factor independently from those parents in the
/// supplied joint law. Under those conditions its marginal is the complete
/// conditional mechanism, so replacing it preserves the checked SCM factorization.
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive root-mechanism contamination routine over its simplex vertices"
)]
pub fn fixed_graph_root_mechanism_sensitivity(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    functional: &BoundTransportFunctional,
    spec: &FixedGraphMechanismSensitivitySpec,
    mechanism: VariableId,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<FixedGraphRootMechanismSensitivityResult, FixedGraphSensitivityError> {
    let mut baseline_spec = spec.clone();
    baseline_spec.max_fraction = 0.0;
    baseline_spec.decision_threshold = None;
    let checked =
        fixed_graph_mechanism_sensitivity(diagram, query, functional, &baseline_spec, ctx)?;
    let graph = diagram.causal_graph();
    let dense = graph
        .nodes()
        .iter()
        .position(|node| *node == NodeRef::Static(mechanism))
        .and_then(|i| u32::try_from(i).ok())
        .map(DenseNodeId::from_raw)
        .ok_or(FixedGraphSensitivityError::UnsupportedGraph)?;
    let treatment = query.treatments[0];
    if mechanism == treatment {
        return Err(FixedGraphSensitivityError::InvalidQuery);
    }
    let parents = &checked.parents;
    let mechanism_position = parents
        .iter()
        .position(|parent| *parent == mechanism)
        .ok_or(FixedGraphSensitivityError::InvalidQuery)?;
    let mechanism_cardinality = spec.parent_cardinalities[mechanism_position];
    let treatment_position = parents
        .iter()
        .position(|parent| *parent == treatment)
        .ok_or(FixedGraphSensitivityError::InvalidQuery)?;
    let other_positions: Vec<_> = parents
        .iter()
        .enumerate()
        .filter_map(|(position, parent)| {
            (*parent != treatment && *parent != mechanism).then_some(position)
        })
        .collect();
    if !graph.parents(dense).is_empty()
        || parents.iter().enumerate().any(|(position, parent)| {
            position != mechanism_position
                && *parent != treatment
                && graph
                    .nodes()
                    .iter()
                    .position(|node| *node == NodeRef::Static(*parent))
                    .and_then(|i| u32::try_from(i).ok())
                    .map(DenseNodeId::from_raw)
                    .is_some_and(|other| graph.reaches(dense, other))
        })
    {
        return Err(FixedGraphSensitivityError::UnsupportedGraph);
    }

    let source_joint: BTreeMap<Vec<usize>, f64> = spec
        .source_parent_law
        .iter()
        .map(|row| (row.parent_levels.clone(), row.probability))
        .collect();
    // Source parent-law rows omit the treatment coordinate.
    let local = |position: usize| position - usize::from(position > treatment_position);
    let mechanism_local = local(mechanism_position);
    let other_local: Vec<_> = other_positions.iter().map(|position| local(*position)).collect();
    let mut mechanism_mass = vec![0.0; mechanism_cardinality];
    let mut other_mass = BTreeMap::<Vec<usize>, f64>::new();
    for (levels, probability) in &source_joint {
        mechanism_mass[levels[mechanism_local]] += probability;
        let other: Vec<_> = other_local.iter().map(|position| levels[*position]).collect();
        *other_mass.entry(other).or_default() += probability;
    }
    for (levels, probability) in &source_joint {
        let other: Vec<_> = other_local.iter().map(|position| levels[*position]).collect();
        if (probability - mechanism_mass[levels[mechanism_local]] * other_mass[&other]).abs()
            > 1e-10
        {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
    }

    let kernel_map: BTreeMap<Vec<usize>, &[f64]> = spec
        .source_kernel
        .iter()
        .map(|row| (row.parent_levels.clone(), row.outcome_probabilities.as_slice()))
        .collect();
    let other_cardinalities: Vec<_> =
        other_positions.iter().map(|position| spec.parent_cardinalities[*position]).collect();
    let other_strata = cartesian_levels(&other_cardinalities)?;
    let mut mechanism_effects = Vec::with_capacity(mechanism_cardinality);
    for mechanism_level in 0..mechanism_cardinality {
        let mut effect = 0.0;
        for other in &other_strata {
            let mut control = vec![0; parents.len()];
            let mut active = vec![0; parents.len()];
            control[treatment_position] = spec.treatment_levels[0];
            active[treatment_position] = spec.treatment_levels[1];
            control[mechanism_position] = mechanism_level;
            active[mechanism_position] = mechanism_level;
            for (position, level) in other_positions.iter().zip(other) {
                control[*position] = *level;
                active[*position] = *level;
            }
            let other_probability = other_mass.get(other).copied().unwrap_or(0.0);
            let control_mean = kernel_map[&control]
                .iter()
                .zip(&spec.outcome_values)
                .map(|(probability, value)| probability * value)
                .sum::<f64>();
            let active_mean = kernel_map[&active]
                .iter()
                .zip(&spec.outcome_values)
                .map(|(probability, value)| probability * value)
                .sum::<f64>();
            effect += other_probability * (active_mean - control_mean);
        }
        mechanism_effects.push(effect);
    }
    let range = DiscreteKernelSensitivity {
        source_kernel: vec![mechanism_mass],
        outcome_values: mechanism_effects,
        stratum_contrast_weights: vec![1.0],
        max_fraction: spec.max_fraction,
        decision_threshold: spec.decision_threshold,
    }
    .evaluate()
    .map_err(FixedGraphSensitivityError::Kernel)?;
    if (range.baseline - checked.response.baseline).abs() > 1e-9 {
        return Err(FixedGraphSensitivityError::InvalidParentLaw);
    }
    Ok(FixedGraphRootMechanismSensitivityResult {
        estimand: format!(
            "target active-minus-control mean response for {} under contamination of root mechanism {mechanism}",
            checked.outcome
        ),
        outcome: checked.outcome,
        mechanism,
        source_population: checked.source_population,
        target_population: checked.target_population,
        source_kernel_binding: checked.source_kernel_binding,
        target_parent_binding: checked.target_parent_binding,
        baseline: range.baseline,
        minimum: range.minimum,
        maximum: range.maximum,
        tipping_fraction: range.tipping_fraction,
        fraction_domain: range.receipt.fraction_domain,
        minimizing_mechanism_level: range.receipt.minimizing_outcome_by_stratum[0],
        maximizing_mechanism_level: range.receipt.maximizing_outcome_by_stratum[0],
        assumptions: vec![
            "fixed fully observed DAG; no latent bidirected edges".into(),
            "only the named isolated root mechanism changes across populations".into(),
            "the root mechanism is independent of the other non-treatment outcome parents, as checked from the supplied joint law".into(),
            "the outcome mechanism and all other parent mechanisms remain invariant".into(),
        ],
    })
}

/// Evaluate a one-factor conditional mechanism deviation for a categorical
/// non-treatment parent of the outcome.
///
/// The graph contract requires every non-treatment outcome parent to have all
/// its directed parents inside that same observed parent set. Complete source
/// and target joint parent laws then identify each categorical conditional
/// mechanism. The route checks their factorization against the fixed DAG and
/// perturbs one factor with row-wise bounded simplex contamination, propagating
/// the changed joint law through downstream parent mechanisms.
#[allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    reason = "one cohesive per-stratum simplex-contamination routine; the exact == locates the vertex achieving the extremal slope just computed by fold(f64::max/min)"
)]
pub fn fixed_graph_conditional_mechanism_sensitivity(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    functional: &BoundTransportFunctional,
    spec: &FixedGraphMechanismSensitivitySpec,
    mechanism: VariableId,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<FixedGraphConditionalMechanismSensitivityResult, FixedGraphSensitivityError> {
    let mut baseline_spec = spec.clone();
    baseline_spec.max_fraction = 0.0;
    baseline_spec.decision_threshold = None;
    let checked =
        fixed_graph_mechanism_sensitivity(diagram, query, functional, &baseline_spec, ctx)?;
    if !spec.max_fraction.is_finite() || !(0.0..=1.0).contains(&spec.max_fraction) {
        return Err(FixedGraphSensitivityError::Kernel(
            DiscreteKernelSensitivityError::InvalidFraction,
        ));
    }
    if spec.decision_threshold.is_some_and(|threshold| !threshold.is_finite()) {
        return Err(FixedGraphSensitivityError::Kernel(
            DiscreteKernelSensitivityError::InvalidContrast,
        ));
    }

    let graph = diagram.causal_graph();
    let treatment = query.treatments[0];
    let parents = &checked.parents;
    let treatment_position = parents
        .iter()
        .position(|parent| *parent == treatment)
        .ok_or(FixedGraphSensitivityError::InvalidQuery)?;
    let mechanism_position = parents
        .iter()
        .position(|parent| *parent == mechanism)
        .filter(|_| mechanism != treatment)
        .ok_or(FixedGraphSensitivityError::InvalidQuery)?;
    let non_treatment_positions: Vec<_> = parents
        .iter()
        .enumerate()
        .filter_map(|(position, parent)| (*parent != treatment).then_some(position))
        .collect();
    let non_treatment_variables: BTreeSet<_> =
        non_treatment_positions.iter().map(|position| parents[*position]).collect();
    let dense_ids: BTreeMap<_, _> = graph
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(index, node)| match node {
            NodeRef::Static(variable) => {
                u32::try_from(index).ok().map(|raw| (*variable, DenseNodeId::from_raw(raw)))
            }
            _ => None,
        })
        .collect();
    let mut factor_parents = BTreeMap::<VariableId, Vec<VariableId>>::new();
    for variable in &non_treatment_variables {
        let dense =
            dense_ids.get(variable).copied().ok_or(FixedGraphSensitivityError::UnsupportedGraph)?;
        let mut graph_parents: Vec<_> = graph
            .parents(dense)
            .iter()
            .map(|parent| match graph.nodes()[parent.as_usize()] {
                NodeRef::Static(id) => Ok(id),
                _ => Err(FixedGraphSensitivityError::UnsupportedGraph),
            })
            .collect::<Result<_, _>>()?;
        graph_parents.sort_unstable();
        if graph_parents.iter().any(|parent| !non_treatment_variables.contains(parent)) {
            return Err(FixedGraphSensitivityError::UnsupportedGraph);
        }
        factor_parents.insert(*variable, graph_parents);
    }
    let selected_parents =
        factor_parents.get(&mechanism).ok_or(FixedGraphSensitivityError::UnsupportedGraph)?;

    let source_joint: BTreeMap<Vec<usize>, f64> = spec
        .source_parent_law
        .iter()
        .map(|row| (row.parent_levels.clone(), row.probability))
        .collect();
    let full_non_treatment_cardinalities: Vec<_> = non_treatment_positions
        .iter()
        .map(|position| spec.parent_cardinalities[*position])
        .collect();
    let all_parent_rows = cartesian_levels(&full_non_treatment_cardinalities)?;
    if source_joint.len() != all_parent_rows.len()
        || all_parent_rows.iter().any(|levels| !source_joint.contains_key(levels))
    {
        return Err(FixedGraphSensitivityError::InvalidParentLaw);
    }
    let position_by_variable: BTreeMap<_, _> = non_treatment_positions
        .iter()
        .enumerate()
        .map(|(local, position)| (parents[*position], local))
        .collect();
    let mut conditional_tables = BTreeMap::<VariableId, BTreeMap<Vec<usize>, Vec<f64>>>::new();
    for variable in &non_treatment_variables {
        let graph_parents = &factor_parents[variable];
        let parent_positions: Vec<_> =
            graph_parents.iter().map(|parent| position_by_variable[parent]).collect();
        let parent_cardinalities: Vec<_> = graph_parents
            .iter()
            .map(|parent| {
                let position = parents.iter().position(|item| item == parent).unwrap();
                spec.parent_cardinalities[position]
            })
            .collect();
        let variable_position = position_by_variable[variable];
        let variable_cardinality =
            spec.parent_cardinalities[parents.iter().position(|item| item == variable).unwrap()];
        let contexts = cartesian_levels(&parent_cardinalities)?;
        let mut table = BTreeMap::new();
        for context in contexts {
            let mut probabilities = vec![0.0; variable_cardinality];
            let mut denominator = 0.0;
            for (levels, probability) in &source_joint {
                if parent_positions
                    .iter()
                    .zip(&context)
                    .all(|(position, level)| levels[*position] == *level)
                {
                    denominator += probability;
                    probabilities[levels[variable_position]] += probability;
                }
            }
            if denominator <= 0.0 {
                return Err(FixedGraphSensitivityError::InvalidParentLaw);
            }
            for probability in &mut probabilities {
                *probability /= denominator;
            }
            table.insert(context, probabilities);
        }
        conditional_tables.insert(*variable, table);
    }
    for levels in &all_parent_rows {
        let mut product = 1.0;
        for variable in &non_treatment_variables {
            let graph_parents = &factor_parents[variable];
            let context: Vec<_> =
                graph_parents.iter().map(|parent| levels[position_by_variable[parent]]).collect();
            let variable_position = position_by_variable[variable];
            product *= conditional_tables[variable][&context][levels[variable_position]];
        }
        if (product - source_joint[levels]).abs() > 1e-10 {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
    }

    let selected_parent_positions: Vec<_> =
        selected_parents.iter().map(|parent| position_by_variable[parent]).collect();
    let selected_parent_cardinalities: Vec<_> = selected_parents
        .iter()
        .map(|parent| {
            let position = parents.iter().position(|item| item == parent).unwrap();
            spec.parent_cardinalities[position]
        })
        .collect();
    let mechanism_cardinality = spec.parent_cardinalities[mechanism_position];
    let contexts = cartesian_levels(&selected_parent_cardinalities)?;
    let kernel_map: BTreeMap<Vec<usize>, &[f64]> = spec
        .source_kernel
        .iter()
        .map(|row| (row.parent_levels.clone(), row.outcome_probabilities.as_slice()))
        .collect();
    let mut minimizing_levels = Vec::with_capacity(contexts.len());
    let mut maximizing_levels = Vec::with_capacity(contexts.len());
    let mut baseline = 0.0;
    let mut min_slope = 0.0;
    let mut max_slope = 0.0;
    for context in &contexts {
        let mut replacement_values = vec![0.0; mechanism_cardinality];
        for levels in &all_parent_rows {
            if !selected_parent_positions
                .iter()
                .zip(context)
                .all(|(position, level)| levels[*position] == *level)
            {
                continue;
            }
            let mechanism_level = levels[position_by_variable[&mechanism]];
            let mut nonselected_probability = 1.0;
            for variable in &non_treatment_variables {
                if *variable == mechanism {
                    continue;
                }
                let graph_parents = &factor_parents[variable];
                let parent_context: Vec<_> = graph_parents
                    .iter()
                    .map(|parent| levels[position_by_variable[parent]])
                    .collect();
                nonselected_probability *= conditional_tables[variable][&parent_context]
                    [levels[position_by_variable[variable]]];
            }
            let mut control = vec![0; parents.len()];
            let mut active = vec![0; parents.len()];
            for (local, position) in non_treatment_positions.iter().enumerate() {
                control[*position] = levels[local];
                active[*position] = levels[local];
            }
            control[treatment_position] = spec.treatment_levels[0];
            active[treatment_position] = spec.treatment_levels[1];
            let control_mean = kernel_map[&control]
                .iter()
                .zip(&spec.outcome_values)
                .map(|(probability, value)| probability * value)
                .sum::<f64>();
            let active_mean = kernel_map[&active]
                .iter()
                .zip(&spec.outcome_values)
                .map(|(probability, value)| probability * value)
                .sum::<f64>();
            replacement_values[mechanism_level] +=
                nonselected_probability * (active_mean - control_mean);
        }
        let source_row = &conditional_tables[&mechanism][context];
        let source_contribution = source_row
            .iter()
            .zip(&replacement_values)
            .map(|(probability, value)| probability * value)
            .sum::<f64>();
        baseline += source_contribution;
        let min_value = replacement_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_value = replacement_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        min_slope += min_value - source_contribution;
        max_slope += max_value - source_contribution;
        minimizing_levels
            .push(replacement_values.iter().position(|value| *value == min_value).unwrap_or(0));
        maximizing_levels
            .push(replacement_values.iter().position(|value| *value == max_value).unwrap_or(0));
    }
    if (baseline - checked.response.baseline).abs() > 1e-9 {
        return Err(FixedGraphSensitivityError::InvalidParentLaw);
    }
    let minimum = baseline + spec.max_fraction * min_slope;
    let maximum = baseline + spec.max_fraction * max_slope;
    let tipping_fraction = spec.decision_threshold.and_then(|threshold| {
        if (baseline - threshold).abs() <= 1e-12 {
            Some(0.0)
        } else {
            let slope = if threshold < baseline { min_slope } else { max_slope };
            crate::tipping::slope_fraction(baseline, threshold, slope).and_then(|fraction| {
                crate::tipping::fraction_in_sloppy_domain(fraction, spec.max_fraction)
            })
        }
    });
    Ok(FixedGraphConditionalMechanismSensitivityResult {
        estimand: format!(
            "target active-minus-control mean response for {} under contamination of conditional mechanism {mechanism}",
            checked.outcome
        ),
        outcome: checked.outcome,
        mechanism,
        source_population: checked.source_population,
        target_population: checked.target_population,
        source_kernel_binding: checked.source_kernel_binding,
        target_parent_binding: checked.target_parent_binding,
        baseline,
        minimum,
        maximum,
        tipping_fraction,
        fraction_domain: [0.0, spec.max_fraction],
        minimizing_mechanism_level_by_parent_stratum: minimizing_levels,
        maximizing_mechanism_level_by_parent_stratum: maximizing_levels,
        assumptions: vec![
            "fixed fully observed DAG; no latent bidirected edges".into(),
            "all parents of every non-treatment outcome parent are contained in the supplied joint outcome-parent set".into(),
            "supplied joint parent law factorizes over the fixed DAG conditional mechanisms".into(),
            "only the named conditional mechanism changes; every other parent and outcome mechanism remains fixed".into(),
            "row-wise contamination uses a normalized categorical replacement distribution per parent stratum".into(),
        ],
    })
}

fn validate_sensitivity_bindings(
    functional: &BoundTransportFunctional,
    query: &ClassicalTransportQuery,
    outcome: VariableId,
    parents: &[VariableId],
    spec: &FixedGraphMechanismSensitivitySpec,
) -> Result<(), FixedGraphSensitivityError> {
    let catalog = functional.catalog();
    let source_ok = catalog.regimes.iter().any(|regime| {
        regime.id == spec.source_kernel_regime
            && regime.evidence_kind == EvidenceKind::Available
            && regime.population.as_ref() == query.source.as_ref()
            && regime.measured.contains(&outcome)
            && parents.iter().all(|parent| {
                regime.measured.contains(parent) || regime.interventions.contains(parent)
            })
            && regime.interventions.contains(&query.treatments[0])
            && catalog.bindings.iter().any(|binding| {
                binding.regime == regime.id
                    && binding.snapshot_identity.as_ref() == spec.source_kernel_snapshot
            })
    });
    let target_ok = catalog.regimes.iter().any(|regime| {
        regime.id == spec.target_parent_regime
            && regime.evidence_kind == EvidenceKind::Available
            && regime.population.as_ref() == query.target.as_ref()
            && regime.kind == antecedent_core::RegimeKind::Observational
            && catalog
                .target_sampling
                .is_some_and(antecedent_core::TargetSampling::represents_target_law)
            && parents
                .iter()
                .filter(|parent| **parent != query.treatments[0])
                .all(|parent| regime.measured.contains(parent))
            && catalog.bindings.iter().any(|binding| {
                binding.regime == regime.id
                    && binding.snapshot_identity.as_ref() == spec.target_parent_snapshot
            })
    });
    if source_ok && target_ok { Ok(()) } else { Err(FixedGraphSensitivityError::SnapshotMismatch) }
}

fn cartesian_levels(
    cardinalities: &[usize],
) -> Result<Vec<Vec<usize>>, FixedGraphSensitivityError> {
    let mut rows = vec![Vec::new()];
    for cardinality in cardinalities {
        if *cardinality == 0 {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
        let next_len = rows
            .len()
            .checked_mul(*cardinality)
            .filter(|length| *length <= FIXED_GRAPH_SENSITIVITY_MAX_STRATA)
            .ok_or(FixedGraphSensitivityError::InvalidParentLaw)?;
        let mut next = Vec::with_capacity(next_len);
        for row in &rows {
            for level in 0..*cardinality {
                let mut expanded = row.clone();
                expanded.push(level);
                next.push(expanded);
            }
        }
        rows = next;
    }
    Ok(rows)
}

fn expression_uses_source_outcome_factor(
    functional: &BoundTransportFunctional,
    query: &ClassicalTransportQuery,
    outcome: VariableId,
    regime: RegimeId,
) -> bool {
    let arena = functional.arena();
    let mut pending = vec![functional.root()];
    let mut visited = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                population,
                regime: leaf_regime,
                ..
            } => {
                let represented: BTreeSet<_> = arena
                    .var_set(*variables)
                    .iter()
                    .chain(arena.var_set(*conditioned_on))
                    .chain(arena.intervention_set(*intervention).iter())
                    .copied()
                    .collect();
                if arena.population(*population) == query.source.as_ref()
                    && *leaf_regime == Some(regime)
                    && represented.contains(&outcome)
                    && query.treatments.iter().all(|parent| represented.contains(parent))
                {
                    return true;
                }
            }
            ExprNode::Product(list) => pending.extend(arena.list(*list)),
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                pending.push(*expr)
            }
            ExprNode::Ratio { numerator, denominator } => {
                pending.extend([*numerator, *denominator])
            }
            ExprNode::Expectation { distribution, .. } => pending.push(*distribution),
            ExprNode::Contrast { left, right, .. } => pending.extend([*left, *right]),
            ExprNode::Kernel { body, .. } => pending.push(*body),
        }
    }
    false
}
