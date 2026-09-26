//! Exact sensitivity ranges for a finite discrete outcome mechanism.
//!
//! The declared source kernel is contaminated once, with a single replacement
//! distribution per parent stratum shared across all response arms:
//! `K_delta(y|s) = (1-delta) K_source(y|s) + delta R(y|s)`. The reported range
//! is an assumption range over every such `R`, not a sampling interval or a
//! sharp causal bound without the declared graph and kernel assumptions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use antecedent_core::{EvidenceKind, RegimeId, VariableId};
use antecedent_expr::ExactTransportData;
use antecedent_expr::ExprNode;
use antecedent_graph::{DenseNodeId, NodeRef, SelectionDiagram};
use antecedent_identify::{
    BoundTransportFunctional, BoundZTransportFunctional, ClassicalTransportQuery, SidLimits,
    verify_classical_transport,
};

/// Inputs for discrete outcome-kernel contamination on a fixed graph.
///
/// `source_kernel` is a row-major matrix indexed by parent stratum then outcome.
/// `stratum_contrast_weights` defines the response contrast (for example, the
/// difference in target stratum probabilities under two interventions). A
/// single contaminated kernel is used for every intervention arm.
#[derive(Clone, Debug)]
pub struct DiscreteKernelSensitivity {
    /// Source probabilities `P(Y=y | parents=s)`, row-major by stratum.
    pub source_kernel: Vec<Vec<f64>>,
    /// Outcome utility/response for each ordered outcome category.
    pub outcome_values: Vec<f64>,
    /// Contrast coefficient for each parent stratum.
    pub stratum_contrast_weights: Vec<f64>,
    /// Largest contamination fraction, in `[0, 1]`.
    pub max_fraction: f64,
    /// Optional decision threshold for the response contrast.
    pub decision_threshold: Option<f64>,
}

/// Exact optimization witness for the two extremal response values.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteKernelOptimizationReceipt {
    /// Outcome category selected for the minimizing replacement kernel in each stratum.
    pub minimizing_outcome_by_stratum: Vec<usize>,
    /// Outcome category selected for the maximizing replacement kernel in each stratum.
    pub maximizing_outcome_by_stratum: Vec<usize>,
    /// Declared contamination domain `[0, max_fraction]`.
    pub fraction_domain: [f64; 2],
    /// Short description of the exact simplex-vertex optimization.
    pub method: &'static str,
}

/// Assumption range and baseline for one-factor discrete kernel contamination.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteKernelSensitivityResult {
    /// Response contrast under the unmodified source kernel.
    pub baseline: f64,
    /// Minimum response over the declared contamination domain.
    pub minimum: f64,
    /// Maximum response over the declared contamination domain.
    pub maximum: f64,
    /// Smallest contamination fraction for which the interval contains the threshold.
    pub tipping_fraction: Option<f64>,
    /// These extrema are assumption ranges, not sampling intervals.
    pub interval_interpretation: &'static str,
    /// Exact optimization witness.
    pub receipt: DiscreteKernelOptimizationReceipt,
}

/// Invalid finite-kernel sensitivity inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscreteKernelSensitivityError {
    /// No strata or categories were supplied, or dimensions disagree.
    InvalidDimensions,
    /// Probabilities or response values are nonfinite, or a kernel row is not a distribution.
    InvalidKernel,
    /// The contamination fraction is outside `[0, 1]`.
    InvalidFraction,
    /// A contrast weight or decision threshold is nonfinite.
    InvalidContrast,
}

impl fmt::Display for DiscreteKernelSensitivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidDimensions => {
                "kernel, outcome, and stratum dimensions must agree and be nonempty"
            }
            Self::InvalidKernel => "source kernel rows must be finite probability distributions",
            Self::InvalidFraction => "maximum contamination fraction must be in [0, 1]",
            Self::InvalidContrast => {
                "contrast weights, outcome values, and threshold must be finite"
            }
        })
    }
}

impl std::error::Error for DiscreteKernelSensitivityError {}

/// One source outcome-kernel row indexed by the outcome's ordered graph parents.
#[derive(Clone, Debug)]
pub struct SourceOutcomeKernelRow {
    /// Categorical index for each parent, in ascending variable order.
    pub parent_levels: Vec<usize>,
    /// Outcome probabilities in the declared category order.
    pub outcome_probabilities: Vec<f64>,
}

/// Source-population mass for each non-treatment outcome-parent stratum.
#[derive(Clone, Debug)]
pub struct SourceParentLawRow {
    /// Category index for each outcome parent other than treatment.
    pub parent_levels: Vec<usize>,
    /// Source probability mass of this stratum.
    pub probability: f64,
}

/// One target parent-law row under a treatment intervention.
#[derive(Clone, Debug)]
pub struct TargetParentLawRow {
    /// Treatment category index.
    pub treatment_level: usize,
    /// Category index for each outcome parent other than treatment.
    pub parent_levels: Vec<usize>,
    /// Target mass of this parent stratum under the intervention.
    pub probability: f64,
}

/// Fixed-graph one-factor mechanism sensitivity contract.
#[derive(Clone, Debug)]
pub struct FixedGraphMechanismSensitivitySpec {
    /// Numeric outcome value corresponding to each probability column.
    pub outcome_values: Vec<f64>,
    /// Cardinalities of the outcome parents, in ascending variable order.
    pub parent_cardinalities: Vec<usize>,
    /// Control and active treatment category indices.
    pub treatment_levels: [usize; 2],
    /// Maximum outcome-kernel contamination fraction.
    pub max_fraction: f64,
    /// Optional response threshold for tipping analysis.
    pub decision_threshold: Option<f64>,
    /// Available source regime and snapshot for the kernel values.
    pub source_kernel_regime: RegimeId,
    /// Snapshot identity for the source kernel values.
    pub source_kernel_snapshot: String,
    /// Available target regime and snapshot for the parent-law values.
    pub target_parent_regime: RegimeId,
    /// Snapshot identity for the target parent-law values.
    pub target_parent_snapshot: String,
    /// Complete source response kernel over all outcome-parent strata.
    pub source_kernel: Vec<SourceOutcomeKernelRow>,
    /// Source distribution of the non-treatment outcome parents.
    pub source_parent_law: Vec<SourceParentLawRow>,
    /// Target parent distributions for both treatment intervention worlds.
    pub target_parent_law: Vec<TargetParentLawRow>,
}

/// Auditable fixed-graph mechanism sensitivity result.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedGraphMechanismSensitivityResult {
    /// Human-readable target estimand.
    pub estimand: String,
    /// Outcome and directed parents from the checked graph.
    pub outcome: VariableId,
    /// Directed outcome parents in their required stratum order.
    pub parents: Vec<VariableId>,
    /// Source and target population identities from the checked query.
    pub source_population: String,
    /// Target population identity from the checked query.
    pub target_population: String,
    /// Bound source kernel regime and snapshot.
    pub source_kernel_binding: (RegimeId, String),
    /// Bound target parent regime and snapshot.
    pub target_parent_binding: (RegimeId, String),
    /// Assumptions and support restrictions defining the result's scope.
    pub assumptions: Vec<String>,
    /// Baseline, exact assumption range, tipping point, and optimization witness.
    pub response: DiscreteKernelSensitivityResult,
}

/// Auditable result for outcome-kernel contamination of the checked z formula.
#[derive(Clone, Debug, PartialEq)]
pub struct ZTransportMechanismSensitivityResult {
    /// Stable identity of the checked derivation's inputs.
    pub query_binding: String,
    /// Provider snapshot used by the baseline formula.
    pub provider_snapshot: String,
    /// Source law regime used by the formula.
    pub source_regime: RegimeId,
    /// Baseline, exact range, tipping point, and witnesses.
    pub response: DiscreteKernelSensitivityResult,
}

/// Refusal from the narrow z outcome-kernel sensitivity contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZTransportSensitivityError {
    /// Proof does not independently verify against the supplied diagram.
    InvalidProof,
    /// The execution context was cancelled before the range was computed.
    Cancelled,
    /// Formula is not a checked outcome conditional times its shared parent marginal.
    IncompatibleFormula,
    /// Exact source provider does not match the formula's population, regime, world, or snapshot.
    ProviderMismatch,
    /// Outcome, treatment, or shared parent has unsupported finite numeric support.
    UnsupportedDomain,
    /// Source law cannot supply every outcome kernel stratum.
    IncompleteKernel,
    /// Contamination fraction or response threshold is invalid.
    InvalidSensitivity(DiscreteKernelSensitivityError),
}

impl fmt::Display for ZTransportSensitivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidProof => "z-transport derivation failed independent verification",
            Self::IncompatibleFormula => {
                "z formula does not expose one shared outcome-kernel factor"
            }
            Self::ProviderMismatch => {
                "source provider does not match the checked z formula binding"
            }
            Self::UnsupportedDomain => {
                "z sensitivity requires finite numeric outcome and binary treatment domains"
            }
            Self::IncompleteKernel => "source law has an empty outcome-kernel stratum",
            Self::Cancelled => "z sensitivity cancelled",
            Self::InvalidSensitivity(error) => return write!(f, "{error}"),
        })
    }
}
impl std::error::Error for ZTransportSensitivityError {}

/// Evaluate the checked z formula under a single coherent outcome-kernel contamination.
///
/// The compatible formula is `sum_w P_source(Y | w, X, do(Z)) P_source(w | do(Z))`.
/// The same conditional kernel is used to form both treatment arms. The returned
/// range is an assumption range over replacement outcome distributions.
// Keep the checked binding and factor validation together so the returned
// sensitivity result is built only after the complete proof/provider contract passes.
#[allow(clippy::too_many_lines)]
pub fn z_transport_mechanism_sensitivity(
    diagram: &SelectionDiagram,
    functional: &BoundZTransportFunctional,
    data: &ExactTransportData,
    max_fraction: f64,
    decision_threshold: Option<f64>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportMechanismSensitivityResult, ZTransportSensitivityError> {
    let cancelled = || {
        if ctx.cancellation.is_cancelled() {
            Err(ZTransportSensitivityError::Cancelled)
        } else {
            Ok(())
        }
    };
    cancelled()?;
    let derivation = functional.derivation();
    let query = derivation.query();
    // The bound functional was verified when it was derived; the input identity
    // is what it still has to prove against this diagram.
    derivation
        .check_inputs(diagram, query)
        .map_err(|_| ZTransportSensitivityError::InvalidProof)?;
    let Some(confounder) = derivation.confounder() else {
        return Err(ZTransportSensitivityError::IncompatibleFormula);
    };

    let arena = functional.arena();
    // The registered surrogate factorization cites exactly one source regime.
    let [regime] = functional.cited_regimes() else {
        return Err(ZTransportSensitivityError::IncompatibleFormula);
    };
    let regime = *regime;
    let ExprNode::SumOut { expr, .. } = arena.node(functional.root()) else {
        return Err(ZTransportSensitivityError::IncompatibleFormula);
    };
    let ExprNode::Product(factors) = arena.node(*expr) else {
        return Err(ZTransportSensitivityError::IncompatibleFormula);
    };
    let leaves = arena.list(*factors);
    if leaves.len() != 2 {
        return Err(ZTransportSensitivityError::IncompatibleFormula);
    }
    let mut outcome_leaf = None;
    let mut parent_leaf = None;
    for id in leaves {
        let ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            population,
            regime: leaf_regime,
            ..
        } = arena.node(*id)
        else {
            return Err(ZTransportSensitivityError::IncompatibleFormula);
        };
        let mut formula_interventions = arena.intervention_set(*intervention);
        let mut query_interventions =
            query.experiment_assignment.iter().map(|a| a.variable).collect::<Vec<_>>();
        formula_interventions.sort_unstable();
        query_interventions.sort_unstable();
        if *leaf_regime != Some(regime)
            || arena.population(*population) != query.source.as_ref()
            || formula_interventions != query_interventions
        {
            return Err(ZTransportSensitivityError::IncompatibleFormula);
        }
        let vars = arena.var_set(*variables);
        let cond = arena.var_set(*conditioned_on);
        if vars == query.outcomes.as_ref()
            && query.treatments.iter().all(|v| cond.contains(v))
            && cond.contains(&confounder)
        {
            outcome_leaf = Some(*id);
        } else if vars == [confounder] && cond.is_empty() {
            parent_leaf = Some(*id);
        } else {
            return Err(ZTransportSensitivityError::IncompatibleFormula);
        }
    }
    if outcome_leaf.is_none()
        || parent_leaf.is_none()
        || query.outcomes.len() != 1
        || query.treatments.len() != 1
    {
        return Err(ZTransportSensitivityError::IncompatibleFormula);
    }

    let law = data
        .laws()
        .iter()
        .find(|law| {
            law.population() == query.source.as_ref()
                && law.regime() == regime
                && law.interventions().len() == query.experiment_assignment.len()
                && query.experiment_assignment.iter().all(|expected| {
                    law.interventions().iter().any(|actual| {
                        actual.variable == expected.variable && actual.value == expected.value
                    })
                })
        })
        .ok_or(ZTransportSensitivityError::ProviderMismatch)?;
    if !functional.catalog().bindings.iter().any(|binding| {
        binding.regime == law.regime()
            && binding.snapshot_identity.as_ref() == law.snapshot_identity()
    }) {
        return Err(ZTransportSensitivityError::ProviderMismatch);
    }
    let y = query.outcomes[0];
    let x = query.treatments[0];
    let w = confounder;
    let axis_pos = |variable| law.axes().iter().position(|axis| axis.variable == variable);
    let (yi, xi, wi) = (axis_pos(y), axis_pos(x), axis_pos(w));
    let (yi, xi, wi) = (
        yi.ok_or(ZTransportSensitivityError::ProviderMismatch)?,
        xi.ok_or(ZTransportSensitivityError::ProviderMismatch)?,
        wi.ok_or(ZTransportSensitivityError::ProviderMismatch)?,
    );
    let y_values = law.axes()[yi]
        .values
        .iter()
        .map(antecedent_core::Value::as_f64)
        .collect::<Option<Vec<_>>>()
        .ok_or(ZTransportSensitivityError::UnsupportedDomain)?;
    let x_levels = law.axes()[xi].values.len();
    let w_levels = law.axes()[wi].values.len();
    if x_levels != 2
        || w_levels == 0
        || y_values.is_empty()
        || law.snapshot_identity().trim().is_empty()
    {
        return Err(ZTransportSensitivityError::UnsupportedDomain);
    }
    let dims: Vec<usize> = law.axes().iter().map(|a| a.values.len()).collect();
    let strides: Vec<usize> = (0..dims.len()).map(|i| dims[i + 1..].iter().product()).collect();
    let mut kernels = Vec::with_capacity(w_levels * 2);
    let mut weights = Vec::with_capacity(w_levels * 2);
    for wl in 0..w_levels {
        cancelled()?;
        let mut parent_mass = 0.0;
        for row in 0..law.probabilities().len() {
            if (row / strides[wi]) % dims[wi] == wl {
                parent_mass += law.probabilities()[row];
            }
        }
        for xl in 0..2 {
            let mut joint = vec![0.0; y_values.len()];
            let mut mass = 0.0;
            for row in 0..law.probabilities().len() {
                if (row / strides[wi]) % dims[wi] == wl && (row / strides[xi]) % dims[xi] == xl {
                    let yl = (row / strides[yi]) % dims[yi];
                    joint[yl] += law.probabilities()[row];
                    mass += law.probabilities()[row];
                }
            }
            if mass <= 0.0 {
                return Err(ZTransportSensitivityError::IncompleteKernel);
            }
            kernels.push(joint.into_iter().map(|p| p / mass).collect());
            weights.push(parent_mass * if xl == 0 { -1.0 } else { 1.0 });
        }
    }
    let response = DiscreteKernelSensitivity {
        source_kernel: kernels,
        outcome_values: y_values,
        stratum_contrast_weights: weights,
        max_fraction,
        decision_threshold,
    }
    .evaluate()
    .map_err(ZTransportSensitivityError::InvalidSensitivity)?;
    Ok(ZTransportMechanismSensitivityResult {
        query_binding: format!("{}:{}:{}", query.source, query.target, functional.root().raw()),
        provider_snapshot: law.snapshot_identity().to_owned(),
        source_regime: regime,
        response,
    })
}

/// Typed refusal for the fixed-graph sensitivity route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedGraphSensitivityError {
    /// Query does not name one outcome and one treatment parent.
    InvalidQuery,
    /// Graph is outside the fixed fully observed DAG contract.
    UnsupportedGraph,
    /// Checked transport proof does not match the supplied query and graph.
    InvalidTransportProof,
    /// Checked expression does not use the declared source outcome factor.
    UnboundOutcomeFactor,
    /// Numerical snapshots do not match available catalog bindings.
    SnapshotMismatch,
    /// Parent support, kernel, or target laws are incomplete or invalid.
    InvalidParentLaw,
    /// The finite kernel calculation rejected its parameters.
    Kernel(DiscreteKernelSensitivityError),
}

/// Maximum parent-stratum product support for the explicit discrete route.
pub const FIXED_GRAPH_SENSITIVITY_MAX_STRATA: usize = 1_000_000;

impl fmt::Display for FixedGraphSensitivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidQuery => "requires one outcome and one treatment parent",
            Self::UnsupportedGraph => "requires a fixed fully observed discrete DAG",
            Self::InvalidTransportProof => {
                "transport proof does not check against the declared graph and query"
            }
            Self::UnboundOutcomeFactor => {
                "checked transport expression does not bind the declared source outcome factor"
            }
            Self::SnapshotMismatch => {
                "declared numerical snapshot does not match an available catalog binding"
            }
            Self::InvalidParentLaw => "kernel or target parent laws are incomplete or invalid",
            Self::Kernel(error) => return write!(f, "{error}"),
        })
    }
}

impl std::error::Error for FixedGraphSensitivityError {}

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
#[allow(clippy::too_many_lines)]
pub fn fixed_graph_mechanism_sensitivity(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    functional: &BoundTransportFunctional,
    spec: &FixedGraphMechanismSensitivitySpec,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<FixedGraphMechanismSensitivityResult, FixedGraphSensitivityError> {
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
    let dense = graph
        .nodes()
        .iter()
        .position(|node| *node == NodeRef::Static(outcome))
        .and_then(|i| u32::try_from(i).ok())
        .map(DenseNodeId::from_raw)
        .ok_or(FixedGraphSensitivityError::UnsupportedGraph)?;
    let treatment_dense = graph
        .nodes()
        .iter()
        .position(|node| *node == NodeRef::Static(treatment))
        .and_then(|i| u32::try_from(i).ok())
        .map(DenseNodeId::from_raw)
        .ok_or(FixedGraphSensitivityError::UnsupportedGraph)?;
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
        let parent_dense = graph
            .nodes()
            .iter()
            .position(|node| *node == NodeRef::Static(*parent))
            .and_then(|i| u32::try_from(i).ok())
            .map(DenseNodeId::from_raw);
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
    let mut kernels = BTreeMap::<Vec<usize>, Vec<f64>>::new();
    for row in &spec.source_kernel {
        if row.parent_levels.len() != parents.len()
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
    if kernels.len() != all_strata.len()
        || all_strata.iter().any(|stratum| !kernels.contains_key(stratum))
    {
        return Err(FixedGraphSensitivityError::InvalidParentLaw);
    }
    let mut target_arms = BTreeMap::<usize, BTreeMap<Vec<usize>, f64>>::new();
    for row in &spec.target_parent_law {
        if !spec.treatment_levels.contains(&row.treatment_level)
            || row.parent_levels.len() != other_positions.len()
            || row.parent_levels.iter().zip(&other_cardinalities).any(|(x, n)| x >= n)
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
    let indices: BTreeMap<_, _> =
        all_strata.iter().cloned().enumerate().map(|(index, stratum)| (stratum, index)).collect();
    let mut weights = vec![0.0; all_strata.len()];
    for (arm_index, treatment_level) in spec.treatment_levels.iter().copied().enumerate() {
        let arm = target_arms
            .get(&treatment_level)
            .ok_or(FixedGraphSensitivityError::InvalidParentLaw)?;
        if arm.len() != other_strata.len()
            || other_strata.iter().any(|stratum| !arm.contains_key(stratum))
            || (arm.values().sum::<f64>() - 1.0).abs() > 1e-10
        {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
        for stratum in &other_strata {
            let mut full = vec![0; parents.len()];
            for (position, level) in other_positions.iter().zip(stratum) {
                full[*position] = *level;
            }
            full[treatment_position] = treatment_level;
            let index = indices.get(&full).ok_or(FixedGraphSensitivityError::InvalidParentLaw)?;
            weights[*index] += if arm_index == 0 { -arm[stratum] } else { arm[stratum] };
        }
    }
    let control = target_arms.get(&spec.treatment_levels[0]).expect("validated control arm");
    let active = target_arms.get(&spec.treatment_levels[1]).expect("validated active arm");
    let mut source_parent_law = BTreeMap::<Vec<usize>, f64>::new();
    for row in &spec.source_parent_law {
        if row.parent_levels.len() != other_positions.len()
            || row.parent_levels.iter().zip(&other_cardinalities).any(|(x, n)| x >= n)
            || !row.probability.is_finite()
            || row.probability < 0.0
            || source_parent_law.insert(row.parent_levels.clone(), row.probability).is_some()
        {
            return Err(FixedGraphSensitivityError::InvalidParentLaw);
        }
    }
    if source_parent_law.len() != other_strata.len()
        || other_strata.iter().any(|stratum| !source_parent_law.contains_key(stratum))
        || (source_parent_law.values().sum::<f64>() - 1.0).abs() > 1e-10
        || other_strata.iter().any(|stratum| {
            (control[stratum] - active[stratum]).abs() > 1e-10
                || (control[stratum] - source_parent_law[stratum]).abs() > 1e-10
        })
    {
        return Err(FixedGraphSensitivityError::InvalidParentLaw);
    }

    validate_sensitivity_bindings(functional, query, outcome, &parents, spec)?;
    if !expression_uses_source_outcome_factor(functional, query, outcome, spec.source_kernel_regime)
    {
        return Err(FixedGraphSensitivityError::UnboundOutcomeFactor);
    }
    let response = DiscreteKernelSensitivity {
        source_kernel: all_strata.iter().map(|s| kernels[s].clone()).collect(),
        outcome_values: spec.outcome_values.clone(),
        stratum_contrast_weights: weights,
        max_fraction: spec.max_fraction,
        decision_threshold: spec.decision_threshold,
    }
    .evaluate()
    .map_err(FixedGraphSensitivityError::Kernel)?;
    Ok(FixedGraphMechanismSensitivityResult {
        estimand: format!(
            "target active-minus-control mean response for {outcome} under source-to-target kernel contamination"
        ),
        outcome,
        parents,
        source_population: query.source.to_string(),
        target_population: query.target.to_string(),
        source_kernel_binding: (spec.source_kernel_regime, spec.source_kernel_snapshot.clone()),
        target_parent_binding: (spec.target_parent_regime, spec.target_parent_snapshot.clone()),
        assumptions: vec![
            "fixed fully observed DAG; no latent bidirected edges".into(),
            "only the discrete outcome mechanism is contaminated".into(),
            "all other outcome-parent mechanisms are invariant across populations".into(),
            "source and target non-treatment outcome-parent laws agree under both interventions"
                .into(),
        ],
        response,
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

impl DiscreteKernelSensitivity {
    /// Evaluate exact extrema over all replacement distributions in each stratum.
    ///
    /// Since the response is linear in each `R(.|s)`, an extremum occurs at a
    /// simplex vertex. The method therefore checks outcome categories exactly;
    /// it does not discretize the contamination fraction.
    pub fn evaluate(
        &self,
    ) -> Result<DiscreteKernelSensitivityResult, DiscreteKernelSensitivityError> {
        let strata = self.source_kernel.len();
        let categories = self.outcome_values.len();
        if strata == 0
            || categories == 0
            || self.stratum_contrast_weights.len() != strata
            || self.source_kernel.iter().any(|row| row.len() != categories)
        {
            return Err(DiscreteKernelSensitivityError::InvalidDimensions);
        }
        if !self.max_fraction.is_finite() || !(0.0..=1.0).contains(&self.max_fraction) {
            return Err(DiscreteKernelSensitivityError::InvalidFraction);
        }
        if self.outcome_values.iter().any(|x| !x.is_finite())
            || self.stratum_contrast_weights.iter().any(|x| !x.is_finite())
            || self.decision_threshold.is_some_and(|x| !x.is_finite())
        {
            return Err(DiscreteKernelSensitivityError::InvalidContrast);
        }
        for row in &self.source_kernel {
            if row.iter().any(|p| !p.is_finite() || *p < 0.0)
                || (row.iter().sum::<f64>() - 1.0).abs() > 1e-10
            {
                return Err(DiscreteKernelSensitivityError::InvalidKernel);
            }
        }

        let baseline = self
            .source_kernel
            .iter()
            .zip(&self.stratum_contrast_weights)
            .map(|(row, weight)| {
                weight * row.iter().zip(&self.outcome_values).map(|(p, y)| p * y).sum::<f64>()
            })
            .sum::<f64>();

        // The same R is used in both contrast arms. `stratum_contrast_weights`
        // already contains their signed target-mass difference, so choosing
        // one outcome per stratum is the exact shared-kernel optimization.
        let mut low_slope = 0.0;
        let mut high_slope = 0.0;
        let mut low_witness = Vec::with_capacity(strata);
        let mut high_witness = Vec::with_capacity(strata);
        for (row, weight) in self.source_kernel.iter().zip(&self.stratum_contrast_weights) {
            let source_mean = row.iter().zip(&self.outcome_values).map(|(p, y)| p * y).sum::<f64>();
            let (low_i, low_y) = self
                .outcome_values
                .iter()
                .copied()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("nonempty categories");
            let (high_i, high_y) = self
                .outcome_values
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("nonempty categories");
            if *weight >= 0.0 {
                low_slope += weight * (low_y - source_mean);
                high_slope += weight * (high_y - source_mean);
                low_witness.push(low_i);
                high_witness.push(high_i);
            } else {
                low_slope += weight * (high_y - source_mean);
                high_slope += weight * (low_y - source_mean);
                low_witness.push(high_i);
                high_witness.push(low_i);
            }
        }
        let end_low = baseline + self.max_fraction * low_slope;
        let end_high = baseline + self.max_fraction * high_slope;
        let minimum = baseline.min(end_low);
        let maximum = baseline.max(end_high);
        let tipping_fraction = self.decision_threshold.and_then(|threshold| {
            if !(minimum..=maximum).contains(&threshold) {
                return None;
            }
            if exact_threshold_matches_baseline(threshold, baseline) {
                return Some(0.0);
            }
            let slope = if threshold < baseline { low_slope } else { high_slope };
            let fraction = (threshold - baseline) / slope;
            (fraction >= 0.0 && fraction <= self.max_fraction).then_some(fraction)
        });

        Ok(DiscreteKernelSensitivityResult {
            baseline,
            minimum,
            maximum,
            tipping_fraction,
            interval_interpretation: "assumption range under declared discrete outcome-kernel contamination; not a sampling interval",
            receipt: DiscreteKernelOptimizationReceipt {
                minimizing_outcome_by_stratum: low_witness,
                maximizing_outcome_by_stratum: high_witness,
                fraction_domain: [0.0, self.max_fraction],
                method: "linear objective over product of outcome simplexes; exact simplex-vertex extrema",
            },
        })
    }
}

// A zero tipping fraction is exact only when the declared threshold is exactly
// the computed baseline; applying a tolerance would change the reported quantity.
#[allow(
    clippy::float_cmp,
    reason = "a threshold equal to the baseline is an exact user-supplied coincidence"
)]
fn exact_threshold_matches_baseline(threshold: f64, baseline: f64) -> bool {
    threshold == baseline
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
        EvidenceRegime, RegimeBinding, RegimeKind, SamplingDesign, TargetSampling,
        VariableCoordinate, VariableDomain,
    };
    use antecedent_graph::{Admg, DenseNodeId};
    use antecedent_identify::{
        CatalogTransportResult, SidLimits, ZTransportQuery, bind_z_transport_catalog,
        identify_catalog_transport, identify_z_transport,
    };
    use std::sync::Arc;

    fn fixture(max_fraction: f64, threshold: Option<f64>) -> DiscreteKernelSensitivity {
        // Two arms have stratum masses (.8,.2) and (.2,.8), producing
        // signed contrast weights (.6,-.6). The shared contaminant is optimized
        // once per stratum, not independently per arm.
        DiscreteKernelSensitivity {
            source_kernel: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
            outcome_values: vec![0.0, 1.0],
            stratum_contrast_weights: vec![0.6, -0.6],
            max_fraction,
            decision_threshold: threshold,
        }
    }

    // The zero-contamination contract requires the extrema to equal baseline
    // exactly; a tolerance could hide an unintended perturbation.
    #[allow(clippy::float_cmp, reason = "tests pin bitwise-identical replayed values")]
    fn assert_exact_float_eq(actual: f64, expected: f64) {
        assert_eq!(actual, expected);
    }

    #[test]
    fn zero_fraction_returns_baseline_and_threshold_tips_exactly() {
        let result = fixture(0.0, Some(0.3)).evaluate().unwrap();
        assert!((result.baseline + 0.3).abs() < 1e-12);
        assert_exact_float_eq(result.minimum, result.baseline);
        assert_exact_float_eq(result.maximum, result.baseline);
        assert_eq!(result.tipping_fraction, None);
        assert!(result.interval_interpretation.contains("not a sampling interval"));
    }

    #[test]
    fn range_is_nested_and_uses_shared_kernel_witnesses() {
        let narrow = fixture(0.2, Some(-0.4)).evaluate().unwrap();
        let broad = fixture(0.6, Some(-0.4)).evaluate().unwrap();
        assert!(broad.minimum <= narrow.minimum);
        assert!(broad.maximum >= narrow.maximum);
        assert_eq!(broad.receipt.minimizing_outcome_by_stratum, vec![0, 1]);
        assert_eq!(broad.receipt.maximizing_outcome_by_stratum, vec![1, 0]);
        assert!(broad.tipping_fraction.is_some_and(|d| d <= 0.6));
    }

    #[test]
    // Keep construction, provider binding, and changed-input checks together
    // because they jointly verify this single staged z-route contract.
    #[allow(clippy::too_many_lines)]
    fn checked_z_formula_sensitivity_binds_its_provider_and_delta_domain() {
        use antecedent_expr::{
            DiscreteAxis, ExactDiscreteLaw, InterventionAssignment, LawTolerance,
        };
        let (w, z, x, y) = (
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            VariableId::from_raw(3),
        );
        let mut graph = Admg::with_variables(4);
        for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        for (a, b) in [(0, 3), (1, 3), (1, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([y]),
            treatments: Arc::from([x]),
            controllable: Arc::from([z]),
            experiment_assignment: Arc::from([antecedent_core::InterventionAssignment {
                variable: z,
                value: antecedent_core::Value::Bool(false),
            }]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let antecedent_identify::ZTransportResult::Identified(proof) = identify_z_transport(
            &diagram,
            &query,
            SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap() else {
            panic!("expected checked z formula")
        };
        let variables = [w, z, x, y];
        let env = antecedent_core::Environment::try_new(
            "source",
            variables.map(|variable| VariableCoordinate {
                variable,
                domain: VariableDomain::Binary,
                unit: None,
            }),
            Arc::<[VariableId]>::from([]),
        )
        .unwrap();
        let measured: Arc<[VariableId]> = Arc::from(variables);
        let regimes = [false, true].map(|level| {
            antecedent_core::EvidenceRegime::try_new(
                RegimeId::from_raw(u32::from(level)),
                RegimeKind::Experimental,
                EvidenceKind::Available,
                [z],
                [antecedent_core::InterventionAssignment {
                    variable: z,
                    value: antecedent_core::Value::Bool(level),
                }],
                Arc::clone(&measured),
                "source",
                antecedent_core::DistributionAvailability::Joint,
            )
            .unwrap()
        });
        let bindings = [false, true].map(|level| RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(u32::from(level)),
            snapshot_identity: Arc::from(if level { "z-true" } else { "source-snapshot" }),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let catalog = EvidenceCatalog::try_new([env], regimes, bindings, None).unwrap();
        let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        let mut probabilities = Vec::with_capacity(8);
        for row in 0..8 {
            let xl = (row / 2) % 2;
            let yl = row % 2;
            probabilities.push(0.25 * if yl == xl { 0.7 } else { 0.3 });
        }
        let law = ExactDiscreteLaw::try_new(
            "source",
            RegimeId::from_raw(0),
            [InterventionAssignment::concrete(z, antecedent_core::Value::Bool(false))],
            [w, x, y].map(|variable| DiscreteAxis {
                variable,
                values: Arc::from([
                    antecedent_core::Value::Bool(false),
                    antecedent_core::Value::Bool(true),
                ]),
            }),
            probabilities,
            "source-snapshot",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([law], 32).unwrap();
        let baseline = z_transport_mechanism_sensitivity(
            &diagram,
            &functional,
            &data,
            0.0,
            None,
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!((baseline.response.baseline - 0.4).abs() < 1e-12);
        assert_exact_float_eq(baseline.response.minimum, baseline.response.baseline);
        assert_exact_float_eq(baseline.response.maximum, baseline.response.baseline);
        assert_eq!(baseline.provider_snapshot, "source-snapshot");
        let tipping = z_transport_mechanism_sensitivity(
            &diagram,
            &functional,
            &data,
            0.0,
            Some(baseline.response.baseline),
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(tipping.response.tipping_fraction, Some(0.0));
    }

    #[test]
    fn rejects_non_distributions_and_invalid_fraction() {
        let mut invalid = fixture(0.2, None);
        invalid.source_kernel[0] = vec![0.2, 0.2];
        assert_eq!(invalid.evaluate().unwrap_err(), DiscreteKernelSensitivityError::InvalidKernel);
        let invalid = fixture(1.2, None);
        assert_eq!(
            invalid.evaluate().unwrap_err(),
            DiscreteKernelSensitivityError::InvalidFraction
        );
    }

    fn checked_transport_fixture()
    -> (SelectionDiagram, ClassicalTransportQuery, antecedent_identify::BoundTransportFunctional)
    {
        let u = VariableId::from_raw(0);
        let x = VariableId::from_raw(1);
        let y = VariableId::from_raw(2);
        let coordinate =
            |variable| VariableCoordinate { variable, domain: VariableDomain::Binary, unit: None };
        let source =
            Environment::try_new("source", [coordinate(u), coordinate(x), coordinate(y)], [x])
                .unwrap();
        let target = Environment::try_new("target", [coordinate(u)], []).unwrap();
        let source_regime = EvidenceRegime::try_new(
            RegimeId::from_raw(1),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [x],
            [],
            [u, y],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let target_regime = EvidenceRegime::try_new(
            RegimeId::from_raw(2),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [u],
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let binding = |regime, snapshot: &str| RegimeBinding {
            dataset_identity: None,
            regime,
            snapshot_identity: Arc::from(snapshot),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::UnknownDependence,
        };
        let catalog = EvidenceCatalog::try_new(
            [source, target],
            [source_regime, target_regime],
            [
                binding(RegimeId::from_raw(1), "source-snapshot"),
                binding(RegimeId::from_raw(2), "target-snapshot"),
            ],
            Some(TargetSampling::RepresentativeSample),
        )
        .unwrap();
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [x]).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([y]),
            treatments: Arc::from([x]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        let CatalogTransportResult::Identified(functional) =
            identify_catalog_transport(&diagram, &query, &catalog, SidLimits::default(), &ctx)
                .unwrap()
        else {
            panic!("source intervention should identify the fixed-DAG response");
        };
        (diagram, query, *functional)
    }

    #[test]
    fn fixed_graph_route_derives_stratum_weights_and_matches_known_truth() {
        let (diagram, query, functional) = checked_transport_fixture();
        let spec = FixedGraphMechanismSensitivitySpec {
            outcome_values: vec![0.0, 1.0],
            parent_cardinalities: vec![2, 2],
            treatment_levels: [0, 1],
            max_fraction: 0.2,
            decision_threshold: Some(0.3),
            source_kernel_regime: RegimeId::from_raw(1),
            source_kernel_snapshot: "source-snapshot".into(),
            target_parent_regime: RegimeId::from_raw(2),
            target_parent_snapshot: "target-snapshot".into(),
            source_kernel: vec![
                SourceOutcomeKernelRow {
                    parent_levels: vec![0, 0],
                    outcome_probabilities: vec![0.8, 0.2],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![0, 1],
                    outcome_probabilities: vec![0.2, 0.8],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![1, 0],
                    outcome_probabilities: vec![0.7, 0.3],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![1, 1],
                    outcome_probabilities: vec![0.3, 0.7],
                },
            ],
            source_parent_law: vec![
                SourceParentLawRow { parent_levels: vec![0], probability: 0.75 },
                SourceParentLawRow { parent_levels: vec![1], probability: 0.25 },
            ],
            target_parent_law: vec![
                TargetParentLawRow {
                    treatment_level: 0,
                    parent_levels: vec![0],
                    probability: 0.75,
                },
                TargetParentLawRow {
                    treatment_level: 0,
                    parent_levels: vec![1],
                    probability: 0.25,
                },
                TargetParentLawRow {
                    treatment_level: 1,
                    parent_levels: vec![0],
                    probability: 0.75,
                },
                TargetParentLawRow {
                    treatment_level: 1,
                    parent_levels: vec![1],
                    probability: 0.25,
                },
            ],
        };
        let result = fixed_graph_mechanism_sensitivity(
            &diagram,
            &query,
            &functional,
            &spec,
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(result.parents, [VariableId::from_raw(0), VariableId::from_raw(1)]);
        assert!((result.response.baseline - 0.55).abs() < 1e-12);
        assert!((result.response.minimum - 0.24).abs() < 1e-12);
        assert!((result.response.maximum - 0.64).abs() < 1e-12);
        assert!((result.response.tipping_fraction.unwrap() - (0.25 / 1.55)).abs() < 1e-12);
    }

    #[test]
    fn fixed_graph_route_refuses_changed_proof_and_incomplete_parent_law() {
        let (diagram, query, functional) = checked_transport_fixture();
        let mut spec = FixedGraphMechanismSensitivitySpec {
            outcome_values: vec![0.0, 1.0],
            parent_cardinalities: vec![2, 2],
            treatment_levels: [0, 1],
            max_fraction: 0.2,
            decision_threshold: None,
            source_kernel_regime: RegimeId::from_raw(1),
            source_kernel_snapshot: "source-snapshot".into(),
            target_parent_regime: RegimeId::from_raw(2),
            target_parent_snapshot: "target-snapshot".into(),
            source_kernel: vec![
                SourceOutcomeKernelRow {
                    parent_levels: vec![0, 0],
                    outcome_probabilities: vec![0.8, 0.2],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![0, 1],
                    outcome_probabilities: vec![0.2, 0.8],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![1, 0],
                    outcome_probabilities: vec![0.7, 0.3],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![1, 1],
                    outcome_probabilities: vec![0.3, 0.7],
                },
            ],
            source_parent_law: vec![
                SourceParentLawRow { parent_levels: vec![0], probability: 0.75 },
                SourceParentLawRow { parent_levels: vec![1], probability: 0.25 },
            ],
            target_parent_law: vec![
                TargetParentLawRow {
                    treatment_level: 0,
                    parent_levels: vec![0],
                    probability: 0.75,
                },
                TargetParentLawRow {
                    treatment_level: 0,
                    parent_levels: vec![1],
                    probability: 0.25,
                },
                TargetParentLawRow {
                    treatment_level: 1,
                    parent_levels: vec![0],
                    probability: 0.75,
                },
                TargetParentLawRow {
                    treatment_level: 1,
                    parent_levels: vec![1],
                    probability: 0.25,
                },
            ],
        };
        spec.source_kernel_snapshot = "stale-snapshot".into();
        assert_eq!(
            fixed_graph_mechanism_sensitivity(
                &diagram,
                &query,
                &functional,
                &spec,
                &antecedent_core::ExecutionContext::for_tests(1),
            )
            .unwrap_err(),
            FixedGraphSensitivityError::SnapshotMismatch
        );
        spec.source_kernel_snapshot = "source-snapshot".into();
        spec.target_parent_law.clear();
        assert!(matches!(
            fixed_graph_mechanism_sensitivity(
                &diagram,
                &query,
                &functional,
                &spec,
                &antecedent_core::ExecutionContext::for_tests(1),
            ),
            Err(FixedGraphSensitivityError::InvalidParentLaw)
        ));
    }
}
