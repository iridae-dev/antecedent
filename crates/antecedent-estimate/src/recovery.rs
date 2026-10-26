//! Exact evaluation of a checked observation-recovery formula and of
//! the downstream effect identified from the recovered law.
//!
//! The recovery formula is evaluated by the shared compiled evaluator over the
//! one named observed pattern law, once per cell of `X ∪ O`, at `R = 1`,
//! `X* = x`, `O = o`. Before any cell is evaluated the exact table is checked
//! against its catalog distribution (population, regime, snapshot, axes and
//! levels), against the deterministic proxy model (no mass on `R_i = 1, X*_i = ?`
//! or `R_i = 0, X*_i ∈ {0, 1}`), and for positivity of every complete-case cell.
//! The result is a derived law: its descriptor carries
//! [`antecedent_core::LawOrigin::Recovered`] provenance and its exact table's
//! snapshot identity is `recovered:<derivation identity>` (see [`RecoveredLaw`]
//! for exactly where the provenance lives). This is graph-licensed recovery, not
//! MAR or inverse-probability weighting: no complete-case fallback exists, and
//! `ObservationAssumption` is not consulted. Exact laws only; counted laws are
//! refused as `cell_not_licensed`.
//!
//! The same module evaluates a retained experimental-recovery law under a hard
//! clamp. Identify cannot depend on the model crate, so the certified
//! equilibrium is compiled here. Each draw keeps its own disturbance. Failed
//! and inadmissible draws stay in the evaluation at their weights.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{CatalogDistribution, ExecutionContext, Value, VariableId};
use antecedent_expr::{
    Assignment, DiscreteAxis, EvalContext, ExactDiscreteLaw, ExactDistribution,
    ExactEvaluationLimits, ExactEvaluationPlan, ExactEvaluationSpec, ExactTransportData, LawOrigin,
};
use antecedent_identify::experimental_recovery::{
    AssembledCoefficientRecovery, DirectCoefficient, DrawAdmissibility,
    InadmissibleCoefficientDraw, JointRecoveryDraw, RecoveredCoefficientUncertainty,
    RecoveryDrawInput, UncertaintyFault, joint_draw_from_blocks,
};
#[cfg(test)]
use antecedent_identify::experimental_recovery::{UnresolvedDirection, UnresolvedReason};
use antecedent_identify::recovery::MISSING_LEVEL;
use antecedent_identify::{
    RECOVERY_MAX_OBSERVED_CELLS, RecoveryDerivation, RecoveryDetail, RecoveryError,
};
use antecedent_model::cyclic_component::{CyclicComponent, CyclicComponentError};
use antecedent_model::intervention_request::{
    InterventionRequest, InterventionRequestError, hard_clamp_of, revalidate_solvability,
};
use antecedent_model::linear_solution::LinearSolutionError;
use antecedent_model::scc_compile::{
    CompileBudget, GaussianDisturbance, LinearSupportRoute, SccCompileError,
    compile_routed_gaussian_equilibrium,
};
use antecedent_model::{HardClamp, LinearStructuralMap, assess};
use thiserror::Error;

/// A recovered full law `P(X(1), O)`: a derived law with provenance.
///
/// Where the provenance lives, precisely: [`Self::descriptor`] carries
/// [`antecedent_core::LawOrigin::Recovered`] with the derivation identity, so no
/// catalog route treats it as measured. The exact table [`Self::law`] is the
/// evaluator's input for the downstream effect; its snapshot identity is
/// `recovered:<derivation identity>` (never the observed table's snapshot), its
/// population is the recovered population (it is that population's law), and
/// its regime id is the observed pattern law's regime it was derived from (as
/// the descriptor's `source_regime` is). Its `SuppliedExact` origin is the
/// exact-law provider tag of `antecedent_expr`, which has no derived-law origin:
/// read on its own, the table does not say it was recovered; the snapshot label
/// and the descriptor do.
#[derive(Clone, Debug)]
pub struct RecoveredLaw {
    scientific_derivation: Arc<str>,
    law: ExactDiscreteLaw,
    descriptor: CatalogDistribution,
    derivation: String,
}

impl RecoveredLaw {
    pub(crate) fn scientific_derivation_identity(&self) -> &str {
        &self.scientific_derivation
    }
    /// The recovered law over `X ∪ O` (axes sorted by variable, last fastest),
    /// with snapshot identity `recovered:<derivation identity>`.
    #[must_use]
    pub const fn law(&self) -> &ExactDiscreteLaw {
        &self.law
    }
    /// Catalog descriptor with `LawOrigin::Recovered` provenance.
    #[must_use]
    pub const fn descriptor(&self) -> &CatalogDistribution {
        &self.descriptor
    }
    /// Identity of the derivation that produced it.
    #[must_use]
    pub fn derivation_identity(&self) -> &str {
        &self.derivation
    }
}

fn refuse(detail: RecoveryDetail, message: impl Into<String>) -> RecoveryError {
    RecoveryError::new(detail, message)
}

/// Binary level of a value: exactly 0 or 1 in any numeric encoding.
fn binary_level(value: &Value) -> Option<u8> {
    if matches!(value, Value::Label(_)) {
        return None;
    }
    match value.as_f64() {
        Some(0.0) => Some(0),
        Some(1.0) => Some(1),
        _ => None,
    }
}

fn is_missing(value: &Value) -> bool {
    matches!(value, Value::Label(label) if label.as_ref() == MISSING_LEVEL)
}

/// Per axis: `levels[axis][index]` is `Some(0|1)` or `None` for `?`.
struct Axes {
    levels: Vec<Vec<Option<u8>>>,
    strides: Vec<usize>,
    /// Axis position of every variable.
    at: BTreeMap<VariableId, usize>,
}

impl Axes {
    fn level(&self, cell: usize, axis: usize) -> Option<u8> {
        self.levels[axis][(cell / self.strides[axis]) % self.levels[axis].len()]
    }
    /// Position of a level on an axis.
    fn position(&self, axis: usize, level: Option<u8>) -> usize {
        self.levels[axis].iter().position(|l| *l == level).unwrap_or(0)
    }
}

/// Check the table against the derivation's observed distribution and the
/// deterministic proxy model, and return its decoded axes.
#[allow(
    clippy::too_many_lines,
    reason = "one pass: identity, bounds, axes, levels and the proxy model"
)]
fn check_observed(
    derivation: &RecoveryDerivation,
    observed: &ExactDiscreteLaw,
) -> Result<Axes, RecoveryError> {
    use RecoveryDetail::{BoundsExceeded, EmpiricalNotLicensed, InvalidObservedLaw};
    if observed.origin() != LawOrigin::SuppliedExact {
        return Err(refuse(
            EmpiricalNotLicensed,
            format!(
                "a {} law is a sampled provider; the recovery route takes exact laws only",
                observed.origin().as_str()
            ),
        ));
    }
    let query = derivation.query();
    let descriptor = derivation.observed();
    if observed.probabilities().len() > RECOVERY_MAX_OBSERVED_CELLS {
        return Err(refuse(
            BoundsExceeded,
            format!("more than {RECOVERY_MAX_OBSERVED_CELLS} observed-law cells"),
        ));
    }
    if observed.population() != query.population.as_ref()
        || observed.regime() != query.observed_regime
        || !observed.interventions().is_empty()
    {
        return Err(refuse(
            InvalidObservedLaw,
            "the exact table is not the named observed pattern law (population or regime)",
        ));
    }
    if let Some(snapshot) = &descriptor.snapshot {
        if observed.snapshot_identity() != snapshot.as_ref() {
            return Err(refuse(
                InvalidObservedLaw,
                "the exact table's snapshot is not the one the catalog binds",
            ));
        }
    }
    let proxies: BTreeMap<VariableId, usize> =
        query.partially_observed.iter().enumerate().map(|(i, p)| (p.proxy, i)).collect();
    let mut variables = Vec::new();
    let mut levels = Vec::new();
    for axis in observed.axes() {
        let proxy = proxies.contains_key(&axis.variable);
        let mut decoded = Vec::new();
        for value in axis.values.iter() {
            decoded.push(match (binary_level(value), proxy && is_missing(value)) {
                (Some(level), _) => Some(level),
                (None, true) => None,
                (None, false) => {
                    return Err(refuse(
                        InvalidObservedLaw,
                        format!("{:?} has a level outside its domain", axis.variable),
                    ));
                }
            });
        }
        let mut sorted = decoded.clone();
        sorted.sort_unstable();
        let expected: &[Option<u8>] =
            if proxy { &[None, Some(0), Some(1)] } else { &[Some(0), Some(1)] };
        if sorted != expected {
            return Err(refuse(
                InvalidObservedLaw,
                format!(
                    "{:?} must have exactly the levels {}",
                    axis.variable,
                    if proxy { "{0, 1, ?}" } else { "{0, 1}" }
                ),
            ));
        }
        variables.push(axis.variable);
        levels.push(decoded);
    }
    let mut sorted_variables = variables.clone();
    sorted_variables.sort_unstable();
    if sorted_variables != query.observed() {
        return Err(refuse(
            InvalidObservedLaw,
            "the exact table's axes are not exactly R, X* and O",
        ));
    }
    let mut strides = vec![1usize; variables.len()];
    for i in (0..variables.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * levels[i + 1].len();
    }
    let at = variables.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    let axes = Axes { levels, strides, at };
    for (cell, p) in observed.probabilities().iter().enumerate() {
        if *p == 0.0 {
            continue;
        }
        for pair in query.partially_observed.iter() {
            let r = axes.level(cell, axes.at[&pair.response]);
            let x = axes.level(cell, axes.at[&pair.proxy]);
            if (r == Some(1)) == x.is_none() {
                return Err(refuse(
                    InvalidObservedLaw,
                    "the exact table puts mass on a cell the deterministic proxy excludes",
                ));
            }
        }
    }
    Ok(axes)
}

/// Evaluate the checked recovery formula on the named observed pattern law.
///
/// # Errors
///
/// A sampled provider (`recovery.empirical_not_licensed`), a table that is not
/// the named law or violates the proxy model or does not normalize under the
/// m-graph (`recovery.invalid_observed_law`), a complete-case cell with no mass
/// (`recovery.positivity`), or cancellation (`recovery.budget`).
pub fn evaluate_exact_recovery(
    derivation: &RecoveryDerivation,
    observed: &ExactDiscreteLaw,
    ctx: &ExecutionContext,
) -> Result<RecoveredLaw, RecoveryError> {
    let axes = check_observed(derivation, observed)?;
    let query = derivation.query();
    let substantive = query.substantive();
    let responses: Vec<VariableId> = query.partially_observed.iter().map(|p| p.response).collect();
    let proxy_of: BTreeMap<VariableId, VariableId> =
        query.partially_observed.iter().map(|p| (p.variable, p.proxy)).collect();
    // The observed coordinate standing for each substantive variable.
    let coordinate = |v: VariableId| proxy_of.get(&v).copied().unwrap_or(v);
    let data = ExactTransportData::try_new(vec![observed.clone()], RECOVERY_MAX_OBSERVED_CELLS)
        .map_err(|e| refuse(RecoveryDetail::InvalidObservedLaw, e.to_string()))?;
    let evaluator = derivation
        .arena()
        .compile(derivation.root())
        .map_err(|e| refuse(RecoveryDetail::InvalidDerivation, e.to_string()))?;
    let value_at = |v: VariableId, level: Option<u8>| {
        let axis = axes.at[&v];
        observed.axes()[axis].values[axes.position(axis, level)].clone()
    };
    let cells = 1usize << substantive.len();
    let mut probabilities = Vec::with_capacity(cells);
    for cell in 0..cells {
        if ctx.cancellation.is_cancelled() {
            return Err(refuse(RecoveryDetail::Budget, "recovery evaluation cancelled"));
        }
        // Last substantive variable varies fastest.
        let level = |i: usize| u8::from((cell >> (substantive.len() - 1 - i)) & 1 == 1);
        let mut assignment = Assignment::new();
        let mut complete_case = 0usize;
        for r in &responses {
            assignment.set(*r, value_at(*r, Some(1)));
            complete_case += axes.strides[axes.at[r]] * axes.position(axes.at[r], Some(1));
        }
        for (i, v) in substantive.iter().enumerate() {
            let c = coordinate(*v);
            assignment.set(c, value_at(c, Some(level(i))));
            complete_case += axes.strides[axes.at[&c]] * axes.position(axes.at[&c], Some(level(i)));
        }
        if observed.probabilities()[complete_case] <= 0.0 {
            return Err(refuse(
                RecoveryDetail::Positivity,
                format!("the complete-case cell {cell} of the observed law has no mass"),
            ));
        }
        let p = evaluator
            .evaluate_with(derivation.arena(), &data, &EvalContext::default(), &assignment)
            .map_err(|e| refuse(RecoveryDetail::InvalidObservedLaw, e.to_string()))?;
        probabilities.push(p);
    }
    let tolerance = observed.tolerance();
    let total = probabilities.iter().sum::<f64>();
    if probabilities
        .iter()
        .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0 + tolerance.absolute + tolerance.relative)
        || (total - 1.0).abs() > tolerance.absolute + tolerance.relative
    {
        return Err(refuse(
            RecoveryDetail::InvalidObservedLaw,
            format!(
                "the recovered masses do not form a law (total {total}); the observed law is not the law of any model of the m-graph"
            ),
        ));
    }
    let recovered_axes: Vec<DiscreteAxis> = substantive
        .iter()
        .map(|v| {
            let c = coordinate(*v);
            DiscreteAxis {
                variable: *v,
                values: Arc::from([value_at(c, Some(0)), value_at(c, Some(1))]),
            }
        })
        .collect();
    let derivation_identity = derivation.identity();
    let law = ExactDiscreteLaw::try_new(
        Arc::clone(&query.population),
        query.observed_regime,
        Vec::new(),
        recovered_axes,
        probabilities,
        // Provenance from the derivation, never the observed table's snapshot.
        format!("recovered:{derivation_identity}"),
        tolerance,
    )
    .map_err(|e| refuse(RecoveryDetail::InvalidObservedLaw, e.to_string()))?;
    Ok(RecoveredLaw {
        scientific_derivation: derivation.scientific_identity(),
        law,
        descriptor: derivation.recovered_descriptor(),
        derivation: derivation_identity,
    })
}

/// Evaluate the downstream effect on a recovered law, only after its population,
/// variables, derivation and support match the effect's handoff contract.
///
/// # Errors
///
/// No effect in the derivation (`recovery.invalid_query`), a law recovered by
/// another derivation, of another population, over other variables or without
/// strictly positive support (`recovery.handoff_mismatch`), an invalid request
/// (`recovery.invalid_query`) or cancellation (`recovery.budget`).
pub fn evaluate_recovered_effect(
    derivation: &RecoveryDerivation,
    recovered: &RecoveredLaw,
    request: Assignment,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<ExactDistribution, RecoveryError> {
    use RecoveryDetail::{Budget, HandoffMismatch, InvalidQuery};
    let Some(effect) = derivation.effect() else {
        return Err(refuse(InvalidQuery, "the derivation identified no downstream effect"));
    };
    if recovered.derivation != derivation.identity() {
        return Err(refuse(HandoffMismatch, "the law was recovered by another derivation"));
    }
    let law = &recovered.law;
    if law.population() != derivation.query().population.as_ref() {
        return Err(refuse(HandoffMismatch, "the recovered law describes another population"));
    }
    let variables: Vec<VariableId> = law.axes().iter().map(|a| a.variable).collect();
    if variables != derivation.query().substantive() {
        return Err(refuse(HandoffMismatch, "the recovered law's variables are not X and O"));
    }
    if law.probabilities().iter().any(|p| *p <= 0.0) {
        return Err(refuse(
            HandoffMismatch,
            "the recovered law is not strictly positive; the effect's conditionals need support",
        ));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(refuse(Budget, "effect evaluation cancelled"));
    }
    let data = ExactTransportData::try_new(vec![law.clone()], RECOVERY_MAX_OBSERVED_CELLS)
        .map_err(|e| refuse(HandoffMismatch, e.to_string()))?
        .with_world_bound_leaves(vec![law.regime()]);
    let plan = ExactEvaluationPlan::compile(
        ExactEvaluationSpec {
            arena: effect.arena(),
            root: effect.root(),
            outcomes: effect.outcomes().to_vec().into(),
            request,
            limits,
            tolerance: law.tolerance(),
        },
        data,
        ctx,
    )
    .map_err(|e| refuse(InvalidQuery, format!("effect request does not compile: {e}")))?;
    plan.evaluate(ctx).map_err(|e| {
        if ctx.cancellation.is_cancelled() {
            refuse(Budget, "effect evaluation cancelled")
        } else {
            refuse(InvalidQuery, format!("effect evaluation failed: {e}"))
        }
    })
}

const RECOVERED_EQUILIBRIUM: &str = "recovered-equilibrium";
#[cfg(test)]
const ARTIFACT_MAGIC: &[u8; 4] = b"RCVY";
/// Layout version written after the magic. A reader refuses every other version.
#[cfg(test)]
const ARTIFACT_VERSION: u32 = 1;
/// Bytes of the trailing integrity word.
#[cfg(test)]
const ARTIFACT_SEAL_BYTES: usize = 8;
/// Smallest encoding of one draw: weight, mark, and five empty counts.
#[cfg(test)]
const DRAW_MIN_BYTES: usize = 8 + 1 + 4 + 4 + 4 + 4 + 4;
/// Smallest encoding of one unresolved direction: tag, target, one count, support count.
#[cfg(test)]
const DIRECTION_MIN_BYTES: usize = 1 + 4 + 4 + 4;
/// How far the draw weights may sit from one before the plan is refused.
const WEIGHT_SUM_TOLERANCE: f64 = 1.0e-9;

/// Why a retained recovery plan was not evaluated.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum RecoveryInterventionFault {
    /// The coefficient law itself was refused.
    #[error(transparent)]
    Uncertainty(#[from] UncertaintyFault),
    /// The bytes are not a retained recovery plan.
    #[error("artifact does not encode a retained recovery plan")]
    Artifact,
    /// An index does not fit in the artifact encoding.
    #[error("an index does not fit in the artifact")]
    IndexOverflow,
    /// The outcome index is outside the node set.
    #[error("outcome index {outcome} is outside dimension {dimension}")]
    Outcome {
        /// Requested outcome.
        outcome: usize,
        /// Node count.
        dimension: usize,
    },
    /// A clamp index is outside the node set.
    #[error("clamp index {node} is outside dimension {dimension}")]
    ClampNode {
        /// Clamp index.
        node: usize,
        /// Node count.
        dimension: usize,
    },
    /// Clamp indices and values have different lengths.
    #[error("clamp has {nodes} nodes and {values} values")]
    ClampShape {
        /// Index count.
        nodes: usize,
        /// Value count.
        values: usize,
    },
    /// The same node is clamped twice.
    #[error("clamp index {node} is repeated")]
    DuplicateClamp {
        /// Repeated index.
        node: usize,
    },
    /// A clamp value is NaN or infinite.
    #[error("a clamp value is not finite")]
    NonFiniteClamp,
    /// The plan has no source id.
    #[error("the source id is empty")]
    EmptySource,
    /// The decision threshold is NaN or infinite.
    #[error("the decision threshold is not finite")]
    Threshold,
    /// The artifact was written in a layout this reader does not know.
    #[error("artifact layout version {found} is not supported")]
    UnsupportedVersion {
        /// Version the artifact claims.
        found: u32,
    },
    /// The artifact bytes do not match their integrity word.
    #[error("artifact bytes do not match their integrity word")]
    Integrity,
    /// A draw weight is NaN, infinite, or negative.
    #[error("the weight of draw {draw} is not a mass")]
    WeightNotMass {
        /// Index of the draw.
        draw: usize,
    },
    /// The draw weights do not sum to one.
    #[error("the draw weights sum to {total}, not one")]
    WeightsNotNormalized {
        /// Sum of the weights.
        total: f64,
    },
    /// No draws were requested.
    #[error("at least one draw must be requested")]
    DrawCount,
    /// A `Coefficients` block has no covariance.
    #[error("coefficient block {block} has no covariance")]
    MissingCovariance {
        /// Index of the block.
        block: usize,
    },
    /// A covariance does not belong to a coefficient block, is repeated, is the
    /// wrong shape, or is not a finite symmetric positive semidefinite matrix.
    #[error("covariance for block {block} is not a covariance of that coefficient block")]
    BlockCovariance {
        /// Index of the block.
        block: usize,
    },
}

/// Pieces of a retained experimental-recovery plan.
#[derive(Clone, Debug)]
pub struct RecoveryPlanParts {
    /// Joint coefficient and disturbance law.
    pub uncertainty: RecoveredCoefficientUncertainty,
    /// Clamped node indices, in node order.
    pub clamp_nodes: Vec<usize>,
    /// Clamp values, aligned with `clamp_nodes`.
    pub clamp_values: Vec<f64>,
    /// Outcome coordinate of the compiled equilibrium.
    pub outcome: usize,
    /// Source of the recovered model. It does not certify the component.
    pub source_id: String,
    /// Evidence that may certify the component. Never a substitute for a draw.
    pub certifying_evidence: Vec<String>,
}

/// A recovery law retained with the intervention it will be compiled under.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "dropping the plan drops the joint draws"]
pub struct RetainedRecoveryPlan {
    uncertainty: RecoveredCoefficientUncertainty,
    clamp_nodes: Vec<usize>,
    clamp_values: Vec<f64>,
    outcome: usize,
    source_id: String,
    certifying_evidence: Vec<String>,
}

impl RetainedRecoveryPlan {
    /// Retain `parts` after checking the clamp and the outcome index.
    ///
    /// # Errors
    ///
    /// [`RecoveryInterventionFault`] when the clamp or the outcome is not a
    /// coordinate of the law, when the source id is empty, when a draw weight is
    /// not a finite non-negative mass, or when the weights do not sum to one.
    pub fn try_new(parts: RecoveryPlanParts) -> Result<Self, RecoveryInterventionFault> {
        let dimension = parts.uncertainty.nodes().len();
        if parts.outcome >= dimension {
            return Err(RecoveryInterventionFault::Outcome { outcome: parts.outcome, dimension });
        }
        if parts.source_id.is_empty() {
            return Err(RecoveryInterventionFault::EmptySource);
        }
        check_clamp(dimension, &parts.clamp_nodes, &parts.clamp_values)?;
        check_weights(&parts.uncertainty)?;
        Ok(Self {
            uncertainty: parts.uncertainty,
            clamp_nodes: parts.clamp_nodes,
            clamp_values: parts.clamp_values,
            outcome: parts.outcome,
            source_id: parts.source_id,
            certifying_evidence: parts.certifying_evidence,
        })
    }

    /// Coefficient law, including inadmissible draws.
    pub const fn uncertainty(&self) -> &RecoveredCoefficientUncertainty {
        &self.uncertainty
    }

    /// Outcome coordinate.
    #[must_use]
    pub const fn outcome(&self) -> usize {
        self.outcome
    }
}

/// Why a mechanism matrix could not be completed from a draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum MechanismFault {
    /// The intercept does not match the nodes.
    #[error("intercept does not match the nodes")]
    InterceptShape,
    /// The matrix size overflows.
    #[error("mechanism matrix size overflows")]
    Overflow,
    /// A coefficient names a node outside the law.
    #[error("coefficient names a node outside the law")]
    UnknownNode,
    /// A coefficient lies on the structural diagonal.
    #[error("coefficient lies on the structural diagonal")]
    Diagonal,
    /// A coefficient was named twice.
    #[error("coefficient was named twice")]
    Repeated,
    /// An off-diagonal coordinate was not named.
    #[error("mechanism matrix was not complete")]
    Incomplete,
}

/// Why presented evidence does not certify the component.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum EvidenceRefusal {
    /// The evidence id is empty.
    #[error("the evidence id is empty")]
    Empty,
    /// The evidence is the source of the model. A source does not certify itself.
    #[error("the source id is not evidence")]
    IsSource,
    /// The evidence is not among those the plan cites.
    #[error("the evidence is not among those the plan cites")]
    NotCited,
}

/// Why one admissible draw was not certified. Each cause stays distinct.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum CertificationFault {
    /// The draw's coefficients are not a complete mechanism matrix.
    #[error("mechanism matrix: {0}")]
    Mechanism(MechanismFault),
    /// The structural map or the open clamp was refused.
    #[error("structural map was refused: {0}")]
    StructuralMap(LinearSolutionError),
    /// The draw's disturbance is not a Gaussian law.
    #[error("disturbance was not a Gaussian law: {0}")]
    Disturbance(SccCompileError),
    /// The hard clamp, or the compile under it, was refused. A singular clamp is
    /// [`InterventionRequestError::NotSolvable`].
    #[error("intervention request was refused: {0}")]
    Request(InterventionRequestError),
    /// The family certificate did not establish `ρ(|K|) < 1`. The equilibrium may
    /// still be unique; it is not certified stable.
    #[error("family certificate was not admitted (ρ(|K|) bound {rho_abs_upper})")]
    FamilyNotAdmitted {
        /// Upper bound on `ρ(|K|)` that failed to be below one.
        rho_abs_upper: f64,
    },
    /// The component was not admitted.
    #[error("component was not admitted: {0}")]
    Component(CyclicComponentError),
    /// The presented evidence does not certify the component.
    #[error("evidence does not certify the component: {0}")]
    Evidence(EvidenceRefusal),
}

/// Moments of the outcome coordinate on one certified draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutcomeMoments {
    /// Equilibrium mean of the outcome.
    pub mean: f64,
    /// Equilibrium variance of the outcome under the draw's own disturbance.
    pub equilibrium_variance: f64,
    /// Variance of the draw's own disturbance at the outcome coordinate.
    pub disturbance_variance: f64,
}

/// One draw after certified intervention evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedRecoveryDraw {
    weight: f64,
    outcome: EvaluatedOutcome,
}

impl EvaluatedRecoveryDraw {
    /// Weight copied from the coefficient law.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// Equilibrium, or the failure that was kept.
    #[must_use]
    pub const fn outcome(&self) -> &EvaluatedOutcome {
        &self.outcome
    }

    /// Compiled mean and covariance, when this draw was certified.
    #[must_use]
    pub fn equilibrium(&self) -> Option<(&[f64], &[f64])> {
        match &self.outcome {
            EvaluatedOutcome::Equilibrium { mean, covariance, .. } => Some((mean, covariance)),
            EvaluatedOutcome::Inadmissible(_) | EvaluatedOutcome::NotCertified(_) => None,
        }
    }

    /// This draw's own disturbance mean and covariance, when it was certified.
    #[must_use]
    pub fn disturbance(&self) -> Option<(&[f64], &[f64])> {
        match &self.outcome {
            EvaluatedOutcome::Equilibrium { disturbance_mean, disturbance_covariance, .. } => {
                Some((disturbance_mean, disturbance_covariance))
            }
            EvaluatedOutcome::Inadmissible(_) | EvaluatedOutcome::NotCertified(_) => None,
        }
    }

    /// Mean, equilibrium variance and disturbance variance of coordinate `outcome`,
    /// when the draw was certified and all three are finite.
    #[must_use]
    pub fn outcome_moments(&self, outcome: usize) -> Option<OutcomeMoments> {
        let (mean, covariance) = self.equilibrium()?;
        let (_, disturbance) = self.disturbance()?;
        let n = mean.len();
        let mean = *mean.get(outcome)?;
        let at = outcome.checked_mul(n)?.checked_add(outcome)?;
        let equilibrium_variance = *covariance.get(at)?;
        let disturbance_variance = *disturbance.get(at)?;
        let finite = mean.is_finite()
            && equilibrium_variance.is_finite()
            && equilibrium_variance >= 0.0
            && disturbance_variance.is_finite()
            && disturbance_variance >= 0.0;
        finite.then_some(OutcomeMoments { mean, equilibrium_variance, disturbance_variance })
    }

    /// Inadmissible status, when the coefficient law already refused the draw.
    #[must_use]
    pub const fn inadmissible_reason(&self) -> Option<InadmissibleCoefficientDraw> {
        match self.outcome {
            EvaluatedOutcome::Inadmissible(reason) => Some(reason),
            EvaluatedOutcome::Equilibrium { .. } | EvaluatedOutcome::NotCertified(_) => None,
        }
    }

    /// Why certification stopped, when the draw was admissible and not certified.
    #[must_use]
    pub const fn certification_fault(&self) -> Option<&CertificationFault> {
        match &self.outcome {
            EvaluatedOutcome::NotCertified(fault) => Some(fault),
            EvaluatedOutcome::Equilibrium { .. } | EvaluatedOutcome::Inadmissible(_) => None,
        }
    }
}

/// Which solver produced a certified equilibrium.
///
/// Specified models remain model-conditional. The label grants no scientific license.
#[derive(Clone, Debug, PartialEq)]
pub enum SolvedBy {
    /// Topological DAG evaluation of an acyclic free support.
    Acyclic,
    /// The DAG algorithm refused a directed cycle; the cyclic license then compiled it.
    Cyclic {
        /// Existing next attempt named by the DAG refusal.
        next_attempt: String,
    },
}

/// Result of compiling one draw. Failures stay attached to the draw.
#[derive(Clone, Debug, PartialEq)]
pub enum EvaluatedOutcome {
    /// Certified Gaussian equilibrium under the retained hard clamp.
    Equilibrium {
        /// Equilibrium mean, in node order.
        mean: Vec<f64>,
        /// Row-major equilibrium covariance under the draw's own disturbance.
        covariance: Vec<f64>,
        /// Disturbance mean of this draw, in node order.
        disturbance_mean: Vec<f64>,
        /// Row-major disturbance covariance of this draw.
        disturbance_covariance: Vec<f64>,
        /// Algorithm that produced the equilibrium. An acyclic support is not a cyclic block.
        solved_by: SolvedBy,
    },
    /// The coefficient law marked the draw inadmissible. No matrix was completed.
    Inadmissible(InadmissibleCoefficientDraw),
    /// Compilation or evidence did not certify the draw. The weight remains.
    NotCertified(CertificationFault),
}

/// Mean and variance of the outcome over the certified draws.
///
/// By the law of total variance, `variance()` is the mean equilibrium variance
/// within draws plus the variance of the draw means between draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutcomeMixture {
    /// Mass of the draws that entered the mixture.
    pub certified_mass: f64,
    /// Mass that did not enter it: inadmissible, uncertified, or non-finite.
    pub withheld_mass: f64,
    /// Weighted mean of the draw means.
    pub mean: f64,
    /// Weighted mean of the equilibrium variances: noise inside each draw.
    pub within_draw_variance: f64,
    /// Weighted variance of the draw means: uncertainty in the coefficients.
    pub between_draw_variance: f64,
    /// Weighted mean of the disturbance variances at the outcome coordinate.
    pub disturbance_variance: f64,
}

impl OutcomeMixture {
    /// Total outcome variance across draws and within them.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.within_draw_variance + self.between_draw_variance
    }
}

/// Certified intervention evaluation of every retained draw.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "dropping the evaluation drops failed mass"]
pub struct CertifiedInterventionEvaluation {
    outcome: usize,
    draws: Vec<EvaluatedRecoveryDraw>,
}

impl CertifiedInterventionEvaluation {
    /// Outcome coordinate the decision reads.
    #[must_use]
    pub const fn outcome(&self) -> usize {
        self.outcome
    }

    /// Draws in law order.
    #[must_use]
    pub fn draws(&self) -> &[EvaluatedRecoveryDraw] {
        &self.draws
    }

    /// Mass of every draw, certified or not.
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        self.draws.iter().map(EvaluatedRecoveryDraw::weight).sum()
    }

    /// Mass whose status is inadmissible or uncertified.
    #[must_use]
    pub fn withheld_mass(&self) -> f64 {
        self.draws
            .iter()
            .filter(|draw| draw.equilibrium().is_none())
            .map(EvaluatedRecoveryDraw::weight)
            .sum()
    }

    /// Outcome mean and variance over the certified draws, with the withheld mass
    /// reported beside them. `None` when no draw has finite outcome moments.
    #[must_use]
    pub fn outcome_mixture(&self) -> Option<OutcomeMixture> {
        let mut certified = 0.0;
        let mut withheld = 0.0;
        let mut moments = Vec::new();
        for draw in &self.draws {
            match draw.outcome_moments(self.outcome) {
                Some(found) => {
                    certified += draw.weight();
                    moments.push((draw.weight(), found));
                }
                None => withheld += draw.weight(),
            }
        }
        if certified <= 0.0 {
            return None;
        }
        let mut mean = 0.0;
        let mut within = 0.0;
        let mut disturbance = 0.0;
        for (weight, found) in &moments {
            mean += weight * found.mean;
            within += weight * found.equilibrium_variance;
            disturbance += weight * found.disturbance_variance;
        }
        mean /= certified;
        within /= certified;
        disturbance /= certified;
        let between = moments
            .iter()
            .map(|(weight, found)| weight * (found.mean - mean) * (found.mean - mean))
            .sum::<f64>()
            / certified;
        Some(OutcomeMixture {
            certified_mass: certified,
            withheld_mass: withheld,
            mean,
            within_draw_variance: within,
            between_draw_variance: between,
            disturbance_variance: disturbance,
        })
    }
}

/// Where the mass of a decision went. Every unit of weight is in exactly one place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassAccount {
    /// Certified draws whose outcome mean is strictly above the threshold.
    pub above: f64,
    /// Certified draws whose outcome mean is finite and not strictly above.
    pub at_or_below: f64,
    /// Inadmissible, uncertified, or without a finite outcome mean.
    pub withheld: f64,
}

impl MassAccount {
    /// Mass in all three places.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.above + self.at_or_below + self.withheld
    }
}

/// Which side of a threshold holds the certified mass.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ThresholdChoice {
    /// Every certified draw is strictly above the threshold, and nothing was withheld.
    Above,
    /// Every certified draw is at or below the threshold, and nothing was withheld.
    AtOrBelow,
}

/// Decision read from a certified evaluation. Masses are computed from the draws.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "dropping the decision drops the withheld mass"]
pub struct RecoveryEvidenceDecision {
    evaluation: CertifiedInterventionEvaluation,
    threshold: f64,
}

impl RecoveryEvidenceDecision {
    /// Evaluation the decision was read from.
    pub const fn evaluation(&self) -> &CertifiedInterventionEvaluation {
        &self.evaluation
    }

    /// Mass of certified draws whose outcome mean is strictly above the threshold.
    #[must_use]
    pub fn above_threshold(&self) -> f64 {
        self.mass_account().above
    }

    /// Mass of certified draws whose outcome mean is finite and not strictly above.
    #[must_use]
    pub fn at_or_below_threshold(&self) -> f64 {
        self.mass_account().at_or_below
    }

    /// Mass that was inadmissible, uncertified, or had no finite outcome mean.
    #[must_use]
    pub fn withheld_mass(&self) -> f64 {
        self.mass_account().withheld
    }

    /// A single action, only when one side holds every unit of mass.
    #[must_use]
    pub fn choice(&self) -> Option<ThresholdChoice> {
        let MassAccount { above, at_or_below: below, withheld } = self.mass_account();
        if withheld > 0.0 {
            return None;
        }
        if above > 0.0 && is_zero(below) {
            return Some(ThresholdChoice::Above);
        }
        if below > 0.0 && is_zero(above) {
            return Some(ThresholdChoice::AtOrBelow);
        }
        None
    }

    /// Where every unit of mass went. Weights were validated when the plan was
    /// retained, so each draw lands in exactly one of the three places and the
    /// total is the evaluation's total.
    #[must_use]
    pub fn mass_account(&self) -> MassAccount {
        let mut account = MassAccount { above: 0.0, at_or_below: 0.0, withheld: 0.0 };
        for draw in self.evaluation.draws() {
            let weight = draw.weight();
            match draw.equilibrium().and_then(|(mean, _)| mean.get(self.evaluation.outcome())) {
                Some(value) if value.is_finite() && *value > self.threshold => {
                    account.above += weight;
                }
                Some(value) if value.is_finite() => account.at_or_below += weight,
                Some(_) | None => account.withheld += weight,
            }
        }
        account
    }
}

/// Compile every draw of `plan` under its hard clamp.
///
/// Admissible draws are compiled with that draw's disturbance. Inadmissible
/// draws are copied into the evaluation and are not replaced by another draw.
#[must_use = "the evaluation is the certified intervention law"]
pub fn consume_retained_recovery_plan(
    plan: &RetainedRecoveryPlan,
) -> CertifiedInterventionEvaluation {
    let draws = plan.uncertainty.draws().iter().map(|draw| evaluate_draw(plan, draw)).collect();
    CertifiedInterventionEvaluation { outcome: plan.outcome, draws }
}

/// Decide `plan` from certifying evidence.
///
/// Evidence equal to the source id, or absent from the certifying list, does
/// not certify. Those draws are withheld and are not dropped. Admitted evidence
/// compiles the retained plan and partitions the outcome means. A choice is
/// returned only when one side of `threshold` holds the entire mass.
///
/// # Errors
///
/// [`RecoveryInterventionFault::Threshold`] when `threshold` is not finite.
#[must_use = "the decision carries withheld mass"]
pub fn consume_recovery_evidence(
    plan: &RetainedRecoveryPlan,
    evidence_id: &str,
    threshold: f64,
) -> Result<RecoveryEvidenceDecision, RecoveryInterventionFault> {
    if !threshold.is_finite() {
        return Err(RecoveryInterventionFault::Threshold);
    }
    let evaluation = match check_evidence(plan, evidence_id) {
        Ok(()) => consume_retained_recovery_plan(plan),
        Err(refusal) => withhold_for_evidence(plan, refusal),
    };
    Ok(RecoveryEvidenceDecision { evaluation, threshold })
}

/// Encode `plan` for an independent consumer.
///
/// # Errors
///
/// [`RecoveryInterventionFault::IndexOverflow`] when a length does not fit in `u32`.
///
/// The layout is the magic, a version word, the body, and a trailing 64-bit
/// FNV-1a word over everything before it. The word detects corruption; it is
/// not authentication.
#[cfg(test)]
pub fn export_retained_recovery_plan(
    plan: &RetainedRecoveryPlan,
) -> Result<Vec<u8>, RecoveryInterventionFault> {
    let mut buf = Vec::new();
    buf.extend_from_slice(ARTIFACT_MAGIC);
    write_u32(&mut buf, ARTIFACT_VERSION)?;
    write_u32(&mut buf, u32::try_from(plan.uncertainty.nodes().len())?)?;
    for node in plan.uncertainty.nodes() {
        write_u32(&mut buf, node.raw())?;
    }
    write_u32(&mut buf, u32::try_from(plan.clamp_nodes.len())?)?;
    for (&node, &value) in plan.clamp_nodes.iter().zip(&plan.clamp_values) {
        write_u32(&mut buf, u32::try_from(node)?)?;
        write_f64(&mut buf, value);
    }
    write_u32(&mut buf, u32::try_from(plan.outcome)?)?;
    write_str(&mut buf, &plan.source_id)?;
    write_u32(&mut buf, u32::try_from(plan.certifying_evidence.len())?)?;
    for evidence in &plan.certifying_evidence {
        write_str(&mut buf, evidence)?;
    }
    write_u32(&mut buf, u32::try_from(plan.uncertainty.draws().len())?)?;
    for draw in plan.uncertainty.draws() {
        write_draw(&mut buf, draw)?;
    }
    let seal = checksum(&buf);
    buf.extend_from_slice(&seal.to_le_bytes());
    Ok(buf)
}

/// Rebuild a plan from `bytes` and decide it.
///
/// The consumer compiles the equilibrium again. It does not read a stored mean.
///
/// # Errors
///
/// [`RecoveryInterventionFault`] when `bytes` is not a plan, or the threshold
/// is not finite.
#[must_use = "the decision is recomputed from the artifact"]
#[cfg(test)]
pub fn consume_recovery_artifact(
    bytes: &[u8],
    evidence_id: &str,
    threshold: f64,
) -> Result<RecoveryEvidenceDecision, RecoveryInterventionFault> {
    let plan = read_plan(bytes)?;
    consume_recovery_evidence(&plan, evidence_id, threshold)
}

/// Asymptotic covariance of the estimated values of one `Coefficients` block.
#[derive(Clone, Debug, PartialEq)]
pub struct CoefficientCovariance {
    /// Index of the block in [`DrawGeneration::blocks`].
    pub block: usize,
    /// Row-major covariance of the block's values, `sources × sources`.
    pub covariance: Vec<f64>,
}

/// What [`asymptotic_normal_draws`] draws from.
#[derive(Clone, Copy, Debug)]
pub struct DrawGeneration<'a> {
    /// Nodes of the recovered law, strictly increasing.
    pub nodes: &'a [VariableId],
    /// Recovered blocks. `NullDirections` blocks are carried into every draw as
    /// unresolved directions; `Inconsistent` and `NotAConstraint` blocks are refused.
    pub blocks: &'a [AssembledCoefficientRecovery],
    /// One covariance for each `Coefficients` block.
    pub covariances: &'a [CoefficientCovariance],
    /// Intercept shared by every draw, in node order.
    pub intercept: &'a [f64],
    /// Disturbance mean shared by every draw. It is not perturbed.
    pub noise_mean: &'a [f64],
    /// Disturbance covariance shared by every draw. It is not perturbed.
    pub noise_covariance: &'a [f64],
    /// Number of draws; each carries weight `1 / draws`.
    pub draws: usize,
    /// Seed. The same inputs and seed give the same draws.
    pub seed: u64,
}

/// Draw coefficient values from their asymptotic normal law.
///
/// Each draw adds `L z` to the values of every `Coefficients` block, where `L` is
/// a Cholesky factor of the block's covariance and `z` is standard normal. The
/// draws are weighted equally, so the weights sum to one, and they are paired
/// with the shared disturbance through [`joint_draw_from_blocks`]. The result is
/// ready for [`RecoveredCoefficientUncertainty::retain`].
///
/// # Errors
///
/// [`RecoveryInterventionFault::DrawCount`] for zero draws,
/// [`RecoveryInterventionFault::MissingCovariance`] for a `Coefficients` block
/// without a covariance, [`RecoveryInterventionFault::BlockCovariance`] for a
/// covariance that is misplaced, repeated, mis-shaped or not positive
/// semidefinite, or the block faults of [`joint_draw_from_blocks`].
pub fn asymptotic_normal_draws(
    spec: &DrawGeneration<'_>,
) -> Result<Vec<RecoveryDrawInput>, RecoveryInterventionFault> {
    if spec.draws == 0 {
        return Err(RecoveryInterventionFault::DrawCount);
    }
    let factors = block_factors(spec)?;
    let weight = 1.0 / draw_count_as_f64(spec.draws);
    let mut draws = Vec::with_capacity(spec.draws);
    for index in 0..spec.draws {
        let mut stream = SplitMix64::new(spec.seed, index);
        let blocks: Vec<AssembledCoefficientRecovery> = spec
            .blocks
            .iter()
            .enumerate()
            .map(|(at, block)| match (block, factors.get(&at)) {
                (
                    AssembledCoefficientRecovery::Coefficients { target, sources, values },
                    Some(factor),
                ) => AssembledCoefficientRecovery::Coefficients {
                    target: *target,
                    sources: sources.clone(),
                    values: perturbed(values, factor, &mut stream),
                },
                (other, _) => other.clone(),
            })
            .collect();
        let draw = joint_draw_from_blocks(
            spec.nodes,
            &blocks,
            spec.intercept.to_vec(),
            spec.noise_mean.to_vec(),
            spec.noise_covariance.to_vec(),
            weight,
        )?;
        draws.push(RecoveryDrawInput {
            coefficients: draw.coefficients().to_vec(),
            intercept: draw.intercept().to_vec(),
            noise_mean: draw.noise_mean().to_vec(),
            noise_covariance: draw.noise_covariance().to_vec(),
            unresolved: draw.unresolved().to_vec(),
            weight: draw.weight(),
            marked_inadmissible: draw.marked_inadmissible(),
        });
    }
    Ok(draws)
}

#[allow(clippy::cast_precision_loss, reason = "a draw count far below 2^53")]
const fn draw_count_as_f64(count: usize) -> f64 {
    count as f64
}

fn block_factors(
    spec: &DrawGeneration<'_>,
) -> Result<BTreeMap<usize, Vec<f64>>, RecoveryInterventionFault> {
    let mut factors = BTreeMap::new();
    for entry in spec.covariances {
        let Some(AssembledCoefficientRecovery::Coefficients { sources, .. }) =
            spec.blocks.get(entry.block)
        else {
            return Err(RecoveryInterventionFault::BlockCovariance { block: entry.block });
        };
        let Some(factor) = cholesky_semidefinite(&entry.covariance, sources.len()) else {
            return Err(RecoveryInterventionFault::BlockCovariance { block: entry.block });
        };
        if factors.insert(entry.block, factor).is_some() {
            return Err(RecoveryInterventionFault::BlockCovariance { block: entry.block });
        }
    }
    for (block, recovered) in spec.blocks.iter().enumerate() {
        if matches!(recovered, AssembledCoefficientRecovery::Coefficients { .. })
            && !factors.contains_key(&block)
        {
            return Err(RecoveryInterventionFault::MissingCovariance { block });
        }
    }
    Ok(factors)
}

fn perturbed(values: &[f64], factor: &[f64], stream: &mut SplitMix64) -> Vec<f64> {
    let k = values.len();
    let z: Vec<f64> = (0..k).map(|_| stream.standard_normal()).collect();
    (0..k)
        .map(|row| values[row] + (0..=row).map(|col| factor[row * k + col] * z[col]).sum::<f64>())
        .collect()
}

/// Lower Cholesky factor of a symmetric positive semidefinite `k × k` matrix, or
/// `None` when it is not finite, not symmetric, or has a negative direction.
fn cholesky_semidefinite(covariance: &[f64], k: usize) -> Option<Vec<f64>> {
    if k == 0 || k.checked_mul(k)? != covariance.len() || covariance.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let scale = (0..k).map(|i| covariance[i * k + i].abs()).fold(0.0_f64, f64::max);
    let tolerance = 1.0e-12 * scale.max(f64::MIN_POSITIVE);
    for row in 0..k {
        for col in 0..row {
            if (covariance[row * k + col] - covariance[col * k + row]).abs() > tolerance {
                return None;
            }
        }
    }
    let mut factor = vec![0.0; k * k];
    antecedent_kernels::cholesky::cholesky_psd_into(covariance, k, 1.0e-12, &mut factor).ok()?;
    Some(factor)
}

/// `SplitMix64` stream, one per draw, so a draw does not depend on how many
/// draws were requested.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64, index: usize) -> Self {
        let mut stream = Self { state: seed ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) };
        stream.next_u64();
        stream
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform on `(0, 1]`.
    #[allow(clippy::cast_precision_loss, reason = "53 significant bits are kept on purpose")]
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / 9_007_199_254_740_992.0
    }

    /// Standard normal by Box-Muller.
    fn standard_normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        radius * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

fn check_clamp(
    dimension: usize,
    nodes: &[usize],
    values: &[f64],
) -> Result<(), RecoveryInterventionFault> {
    if nodes.len() != values.len() {
        return Err(RecoveryInterventionFault::ClampShape {
            nodes: nodes.len(),
            values: values.len(),
        });
    }
    let mut seen = BTreeSet::new();
    for (&node, &value) in nodes.iter().zip(values) {
        if node >= dimension {
            return Err(RecoveryInterventionFault::ClampNode { node, dimension });
        }
        if !seen.insert(node) {
            return Err(RecoveryInterventionFault::DuplicateClamp { node });
        }
        if !value.is_finite() {
            return Err(RecoveryInterventionFault::NonFiniteClamp);
        }
    }
    Ok(())
}

fn check_evidence(plan: &RetainedRecoveryPlan, evidence_id: &str) -> Result<(), EvidenceRefusal> {
    if evidence_id.is_empty() {
        return Err(EvidenceRefusal::Empty);
    }
    if evidence_id == plan.source_id {
        return Err(EvidenceRefusal::IsSource);
    }
    if plan.certifying_evidence.iter().any(|cited| cited == evidence_id) {
        Ok(())
    } else {
        Err(EvidenceRefusal::NotCited)
    }
}

fn check_weights(
    uncertainty: &RecoveredCoefficientUncertainty,
) -> Result<(), RecoveryInterventionFault> {
    let mut total = 0.0;
    for (draw, retained) in uncertainty.draws().iter().enumerate() {
        let weight = retained.weight();
        if !(weight.is_finite() && weight >= 0.0) {
            return Err(RecoveryInterventionFault::WeightNotMass { draw });
        }
        total += weight;
    }
    if (total - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
        return Err(RecoveryInterventionFault::WeightsNotNormalized { total });
    }
    Ok(())
}

fn withhold_for_evidence(
    plan: &RetainedRecoveryPlan,
    refusal: EvidenceRefusal,
) -> CertifiedInterventionEvaluation {
    let draws = plan
        .uncertainty
        .draws()
        .iter()
        .map(|draw| EvaluatedRecoveryDraw {
            weight: draw.weight(),
            outcome: match draw.admissibility() {
                DrawAdmissibility::Inadmissible(reason) => EvaluatedOutcome::Inadmissible(reason),
                DrawAdmissibility::Admissible => {
                    EvaluatedOutcome::NotCertified(CertificationFault::Evidence(refusal))
                }
            },
        })
        .collect();
    CertifiedInterventionEvaluation { outcome: plan.outcome, draws }
}

fn evaluate_draw(plan: &RetainedRecoveryPlan, draw: &JointRecoveryDraw) -> EvaluatedRecoveryDraw {
    let weight = draw.weight();
    if let DrawAdmissibility::Inadmissible(reason) = draw.admissibility() {
        return EvaluatedRecoveryDraw { weight, outcome: EvaluatedOutcome::Inadmissible(reason) };
    }
    let outcome = match compile_draw(plan, draw) {
        Ok((mean, covariance, solved_by)) => EvaluatedOutcome::Equilibrium {
            mean,
            covariance,
            disturbance_mean: draw.noise_mean().to_vec(),
            disturbance_covariance: draw.noise_covariance().to_vec(),
            solved_by,
        },
        Err(fault) => EvaluatedOutcome::NotCertified(fault),
    };
    EvaluatedRecoveryDraw { weight, outcome }
}

/// Compile one draw. A singular clamp is checked before the family certificate,
/// so it is reported as a clamp fault and not as a failed certificate.
///
/// An acyclic free support uses the DAG algorithm. A directed cycle is refused
/// by that algorithm and then compiled under the cyclic license. Specified
/// models remain model-conditional; compilation grants no scientific license.
fn compile_draw(
    plan: &RetainedRecoveryPlan,
    draw: &JointRecoveryDraw,
) -> Result<(Vec<f64>, Vec<f64>, SolvedBy), CertificationFault> {
    let (coefficients, intercept) =
        mechanism_matrix(plan.uncertainty.nodes(), draw.coefficients(), draw.intercept())
            .map_err(CertificationFault::Mechanism)?;
    let model = LinearStructuralMap::try_new(coefficients, intercept)
        .map_err(CertificationFault::StructuralMap)?;
    let disturbance =
        GaussianDisturbance::try_new(draw.noise_mean().to_vec(), draw.noise_covariance().to_vec())
            .map_err(CertificationFault::Disturbance)?;
    let request =
        InterventionRequest::hard_clamp(model.dimension(), &plan.clamp_nodes, &plan.clamp_values)
            .map_err(CertificationFault::Request)?;
    revalidate_solvability(&model, &request).map_err(CertificationFault::Request)?;
    let open = HardClamp::none(model.dimension()).map_err(CertificationFault::StructuralMap)?;
    let assessment = assess(&model, &open).map_err(CertificationFault::StructuralMap)?;
    if !assessment.family.is_admitted() {
        return Err(CertificationFault::FamilyNotAdmitted {
            rho_abs_upper: assessment.family.rho_abs_upper(),
        });
    }
    let evidence: Vec<&str> = plan.certifying_evidence.iter().map(String::as_str).collect();
    let component = CyclicComponent::try_new(
        RECOVERED_EQUILIBRIUM,
        &plan.source_id,
        &evidence,
        &model,
        &assessment.family,
    )
    .map_err(CertificationFault::Component)?;
    let (clamp, values) = hard_clamp_of(&request).map_err(CertificationFault::Request)?;
    let routed = compile_routed_gaussian_equilibrium(
        &component,
        &model,
        &disturbance,
        clamp,
        &values,
        CompileBudget::unlimited(),
    )
    .map_err(InterventionRequestError::Compile)
    .map_err(CertificationFault::Request)?;
    let solved_by = match routed.route() {
        LinearSupportRoute::Acyclic { .. } => SolvedBy::Acyclic,
        LinearSupportRoute::Cyclic { next_attempt, .. } => {
            SolvedBy::Cyclic { next_attempt: next_attempt.clone() }
        }
    };
    let equilibrium = routed.equilibrium();
    Ok((equilibrium.mean().to_vec(), equilibrium.covariance().to_vec(), solved_by))
}

fn mechanism_matrix(
    nodes: &[VariableId],
    coefficients: &[(DirectCoefficient, f64)],
    intercept: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), MechanismFault> {
    let n = nodes.len();
    if intercept.len() != n {
        return Err(MechanismFault::InterceptShape);
    }
    let Some(len) = n.checked_mul(n) else {
        return Err(MechanismFault::Overflow);
    };
    let mut matrix = vec![0.0; len];
    let mut filled = vec![false; len];
    for (coefficient, value) in coefficients {
        let Some(into) = nodes.iter().position(|node| *node == coefficient.into) else {
            return Err(MechanismFault::UnknownNode);
        };
        let Some(from) = nodes.iter().position(|node| *node == coefficient.from) else {
            return Err(MechanismFault::UnknownNode);
        };
        if into == from {
            return Err(MechanismFault::Diagonal);
        }
        let index = into * n + from;
        if filled[index] {
            return Err(MechanismFault::Repeated);
        }
        filled[index] = true;
        matrix[index] = *value;
    }
    for into in 0..n {
        for from in 0..n {
            if into != from && !filled[into * n + from] {
                return Err(MechanismFault::Incomplete);
            }
        }
    }
    Ok((matrix, intercept.to_vec()))
}

fn is_zero(value: f64) -> bool {
    value.to_bits() == 0.0_f64.to_bits() || value.to_bits() == (-0.0_f64).to_bits()
}

#[allow(clippy::unnecessary_wraps, reason = "callers chain it with fallible length conversions")]
#[cfg(test)]
fn write_u32(buf: &mut Vec<u8>, value: u32) -> Result<(), RecoveryInterventionFault> {
    buf.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
fn write_f64(buf: &mut Vec<u8>, value: f64) {
    buf.extend_from_slice(&value.to_bits().to_le_bytes());
}

#[cfg(test)]
fn write_str(buf: &mut Vec<u8>, value: &str) -> Result<(), RecoveryInterventionFault> {
    let len = u32::try_from(value.len()).map_err(|_| RecoveryInterventionFault::IndexOverflow)?;
    write_u32(buf, len)?;
    buf.extend_from_slice(value.as_bytes());
    Ok(())
}

#[cfg(test)]
fn write_draw(
    buf: &mut Vec<u8>,
    draw: &JointRecoveryDraw,
) -> Result<(), RecoveryInterventionFault> {
    write_f64(buf, draw.weight());
    buf.push(u8::from(draw.marked_inadmissible()));
    write_u32(buf, u32::try_from(draw.coefficients().len())?)?;
    for (coefficient, value) in draw.coefficients() {
        write_u32(buf, coefficient.into.raw())?;
        write_u32(buf, coefficient.from.raw())?;
        write_f64(buf, *value);
    }
    write_slice(buf, draw.intercept())?;
    write_slice(buf, draw.noise_mean())?;
    write_slice(buf, draw.noise_covariance())?;
    write_u32(buf, u32::try_from(draw.unresolved().len())?)?;
    for direction in draw.unresolved() {
        write_direction(buf, direction)?;
    }
    Ok(())
}

#[cfg(test)]
fn write_slice(buf: &mut Vec<u8>, values: &[f64]) -> Result<(), RecoveryInterventionFault> {
    write_u32(buf, u32::try_from(values.len())?)?;
    for value in values {
        write_f64(buf, *value);
    }
    Ok(())
}

#[cfg(test)]
fn write_direction(
    buf: &mut Vec<u8>,
    direction: &UnresolvedDirection,
) -> Result<(), RecoveryInterventionFault> {
    match &direction.reason {
        UnresolvedReason::IncompletePairCoverage { target, missing_clamps } => {
            buf.push(1);
            write_u32(buf, target.raw())?;
            write_u32(buf, u32::try_from(missing_clamps.len())?)?;
            for node in missing_clamps {
                write_u32(buf, node.raw())?;
            }
        }
        UnresolvedReason::DeficientConstraintRank { target, column_rank, width } => {
            buf.push(2);
            write_u32(buf, target.raw())?;
            write_u32(buf, u32::try_from(*column_rank)?)?;
            write_u32(buf, u32::try_from(*width)?)?;
        }
    }
    write_u32(buf, u32::try_from(direction.support.len())?)?;
    for coefficient in &direction.support {
        write_u32(buf, coefficient.into.raw())?;
        write_u32(buf, coefficient.from.raw())?;
    }
    Ok(())
}

/// 64-bit FNV-1a over `bytes`.
#[cfg(test)]
fn checksum(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
fn read_plan(bytes: &[u8]) -> Result<RetainedRecoveryPlan, RecoveryInterventionFault> {
    let header = ARTIFACT_MAGIC.len() + 4;
    if bytes.len() < header + ARTIFACT_SEAL_BYTES
        || &bytes[..ARTIFACT_MAGIC.len()] != ARTIFACT_MAGIC
    {
        return Err(RecoveryInterventionFault::Artifact);
    }
    let mut version = [0; 4];
    version.copy_from_slice(&bytes[ARTIFACT_MAGIC.len()..header]);
    let found = u32::from_le_bytes(version);
    if found != ARTIFACT_VERSION {
        return Err(RecoveryInterventionFault::UnsupportedVersion { found });
    }
    let (sealed, seal) = bytes.split_at(bytes.len() - ARTIFACT_SEAL_BYTES);
    let mut word = [0; ARTIFACT_SEAL_BYTES];
    word.copy_from_slice(seal);
    if u64::from_le_bytes(word) != checksum(sealed) {
        return Err(RecoveryInterventionFault::Integrity);
    }
    let mut reader = Reader { rest: &sealed[header..] };
    let node_count = reader.count(4)?;
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(VariableId::from_raw(reader.u32()?));
    }
    let clamp_count = reader.count(12)?;
    let mut clamp_nodes = Vec::with_capacity(clamp_count);
    let mut clamp_values = Vec::with_capacity(clamp_count);
    for _ in 0..clamp_count {
        clamp_nodes.push(reader.u32()? as usize);
        clamp_values.push(reader.f64()?);
    }
    let outcome = reader.u32()? as usize;
    let source_id = reader.string()?;
    let evidence_count = reader.count(4)?;
    let mut certifying_evidence = Vec::with_capacity(evidence_count);
    for _ in 0..evidence_count {
        certifying_evidence.push(reader.string()?);
    }
    let draw_count = reader.count(DRAW_MIN_BYTES)?;
    let mut inputs = Vec::with_capacity(draw_count);
    for _ in 0..draw_count {
        inputs.push(reader.draw()?);
    }
    if !reader.rest.is_empty() {
        return Err(RecoveryInterventionFault::Artifact);
    }
    let uncertainty = RecoveredCoefficientUncertainty::retain(&nodes, inputs)?;
    RetainedRecoveryPlan::try_new(RecoveryPlanParts {
        uncertainty,
        clamp_nodes,
        clamp_values,
        outcome,
        source_id,
        certifying_evidence,
    })
}

#[cfg(test)]
struct Reader<'a> {
    rest: &'a [u8],
}

#[cfg(test)]
impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], RecoveryInterventionFault> {
        if self.rest.len() < len {
            return Err(RecoveryInterventionFault::Artifact);
        }
        let (head, tail) = self.rest.split_at(len);
        self.rest = tail;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, RecoveryInterventionFault> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn f64(&mut self) -> Result<f64, RecoveryInterventionFault> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(f64::from_bits(u64::from_le_bytes(bytes)))
    }

    /// A count of elements that each take at least `element` bytes. A count the
    /// remaining bytes cannot hold is refused before anything is allocated.
    fn count(&mut self, element: usize) -> Result<usize, RecoveryInterventionFault> {
        let count =
            usize::try_from(self.u32()?).map_err(|_| RecoveryInterventionFault::Artifact)?;
        let needed = count.checked_mul(element).ok_or(RecoveryInterventionFault::Artifact)?;
        if needed > self.rest.len() {
            return Err(RecoveryInterventionFault::Artifact);
        }
        Ok(count)
    }

    fn flag(&mut self) -> Result<bool, RecoveryInterventionFault> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(RecoveryInterventionFault::Artifact),
        }
    }

    fn string(&mut self) -> Result<String, RecoveryInterventionFault> {
        let len = self.count(1)?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| RecoveryInterventionFault::Artifact)
    }

    fn draw(&mut self) -> Result<RecoveryDrawInput, RecoveryInterventionFault> {
        let weight = self.f64()?;
        let marked = self.flag()?;
        let coefficient_count = self.count(16)?;
        let mut coefficients = Vec::with_capacity(coefficient_count);
        for _ in 0..coefficient_count {
            let into = VariableId::from_raw(self.u32()?);
            let from = VariableId::from_raw(self.u32()?);
            coefficients.push((DirectCoefficient { into, from }, self.f64()?));
        }
        Ok(RecoveryDrawInput {
            coefficients,
            intercept: self.floats()?,
            noise_mean: self.floats()?,
            noise_covariance: self.floats()?,
            unresolved: self.directions()?,
            weight,
            marked_inadmissible: marked,
        })
    }

    fn floats(&mut self) -> Result<Vec<f64>, RecoveryInterventionFault> {
        let count = self.count(8)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.f64()?);
        }
        Ok(values)
    }

    fn directions(&mut self) -> Result<Vec<UnresolvedDirection>, RecoveryInterventionFault> {
        let count = self.count(DIRECTION_MIN_BYTES)?;
        let mut directions = Vec::with_capacity(count);
        for _ in 0..count {
            directions.push(self.direction()?);
        }
        Ok(directions)
    }

    fn direction(&mut self) -> Result<UnresolvedDirection, RecoveryInterventionFault> {
        let tag = self.take(1)?[0];
        let reason = match tag {
            1 => {
                let target = VariableId::from_raw(self.u32()?);
                let missing = self.nodes()?;
                UnresolvedReason::IncompletePairCoverage { target, missing_clamps: missing }
            }
            2 => UnresolvedReason::DeficientConstraintRank {
                target: VariableId::from_raw(self.u32()?),
                column_rank: self.u32()? as usize,
                width: self.u32()? as usize,
            },
            _ => return Err(RecoveryInterventionFault::Artifact),
        };
        let support_count = self.count(8)?;
        let mut support = Vec::with_capacity(support_count);
        for _ in 0..support_count {
            support.push(DirectCoefficient {
                into: VariableId::from_raw(self.u32()?),
                from: VariableId::from_raw(self.u32()?),
            });
        }
        Ok(UnresolvedDirection { support, reason })
    }

    fn nodes(&mut self) -> Result<Vec<VariableId>, RecoveryInterventionFault> {
        let count = self.count(4)?;
        let mut nodes = Vec::with_capacity(count);
        for _ in 0..count {
            nodes.push(VariableId::from_raw(self.u32()?));
        }
        Ok(nodes)
    }
}

impl From<std::num::TryFromIntError> for RecoveryInterventionFault {
    fn from(_: std::num::TryFromIntError) -> Self {
        Self::IndexOverflow
    }
}

#[cfg(test)]
mod recovery_intervention_tests {
    use super::{
        ARTIFACT_MAGIC, ARTIFACT_VERSION, CertificationFault, CoefficientCovariance,
        DrawGeneration, EvaluatedOutcome, EvidenceRefusal, RecoveryInterventionFault,
        RecoveryPlanParts, RetainedRecoveryPlan, ThresholdChoice, asymptotic_normal_draws,
        checksum, consume_recovery_artifact, consume_recovery_evidence,
        consume_retained_recovery_plan, export_retained_recovery_plan, read_plan,
    };
    use antecedent_core::VariableId;
    use antecedent_identify::experimental_recovery::{
        AssembledCoefficientRecovery, DirectCoefficient, InadmissibleCoefficientDraw,
        RecoveredCoefficientUncertainty, RecoveryDrawInput, UncertaintyFault,
        joint_draw_from_blocks,
    };
    use antecedent_model::intervention_request::InterventionRequestError;
    use antecedent_model::linear_solution::SolvabilityRefusal;
    use antecedent_model::scc_compile::SccCompileError;

    fn node(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn coefficient(into: u32, from: u32, value: f64) -> (DirectCoefficient, f64) {
        (DirectCoefficient { into: node(into), from: node(from) }, value)
    }

    fn noise(variance: f64) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0, 0.0], vec![1.0, 0.3, 0.3, variance])
    }

    fn draw_with(
        b: f64,
        variance: f64,
        weight: f64,
        marked_inadmissible: bool,
    ) -> RecoveryDrawInput {
        let (noise_mean, noise_covariance) = noise(variance);
        RecoveryDrawInput {
            coefficients: vec![coefficient(0, 1, 0.4), coefficient(1, 0, b)],
            intercept: vec![0.0, 0.0],
            noise_mean,
            noise_covariance,
            unresolved: Vec::new(),
            weight,
            marked_inadmissible,
        }
    }

    fn draw(variance: f64, weight: f64, marked_inadmissible: bool) -> RecoveryDrawInput {
        draw_with(0.5, variance, weight, marked_inadmissible)
    }

    fn parts(uncertainty: RecoveredCoefficientUncertainty) -> RecoveryPlanParts {
        RecoveryPlanParts {
            uncertainty,
            clamp_nodes: vec![0],
            clamp_values: vec![2.0],
            outcome: 1,
            source_id: "clamp-sample".to_owned(),
            certifying_evidence: vec!["solve-note".to_owned()],
        }
    }

    fn plan(draws: Vec<RecoveryDrawInput>) -> RetainedRecoveryPlan {
        let uncertainty =
            RecoveredCoefficientUncertainty::retain(&[node(0), node(1)], draws).expect("law");
        RetainedRecoveryPlan::try_new(parts(uncertainty)).expect("plan")
    }

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1.0e-8
    }

    /// Where the first draw begins in the artifact of `plan`: the header, the two
    /// nodes, one clamp, the outcome, the source and evidence strings, and the
    /// draw count.
    const DRAWS_AT: usize = 8 + (4 + 8) + (4 + 12) + 4 + (4 + 12) + (4 + 4 + 10) + 4;

    /// `bytes` with a fresh trailing integrity word, as a writer would seal them.
    fn sealed(mut body: Vec<u8>) -> Vec<u8> {
        let seal = checksum(&body);
        body.extend_from_slice(&seal.to_le_bytes());
        body
    }

    fn header() -> Vec<u8> {
        let mut bytes = ARTIFACT_MAGIC.to_vec();
        bytes.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
        bytes
    }

    #[test]
    fn retained_plan_keeps_parameter_noise_dependence_and_inadmissible_mass() {
        let retained =
            plan(vec![draw(1.0, 0.4, false), draw(4.0, 0.4, false), draw(1.0, 0.2, true)]);
        let evaluation = consume_retained_recovery_plan(&retained);
        assert_eq!(evaluation.draws().len(), 3);
        let (mean, covariance) = evaluation.draws()[0].equilibrium().expect("first equilibrium");
        assert!(close(mean[0], 2.0));
        assert!(close(mean[1], 1.0));
        assert!(close(covariance[3], 1.0));
        let (_, other) = evaluation.draws()[1].equilibrium().expect("second equilibrium");
        assert!(close(other[3], 4.0));
        assert!((covariance[3] - other[3]).abs() > 1.0);
        assert_eq!(
            evaluation.draws()[2].inadmissible_reason(),
            Some(InadmissibleCoefficientDraw::Marked)
        );
        assert!(evaluation.draws()[2].equilibrium().is_none());
        assert!(close(evaluation.withheld_mass(), 0.2));
        assert!(close(evaluation.total_mass(), 1.0));
    }

    #[test]
    fn evidence_decision_keeps_inadmissible_mass_and_withholds_the_choice() {
        let retained = plan(vec![draw(1.0, 0.5, false), draw(1.0, 0.5, true)]);
        let decision = consume_recovery_evidence(&retained, "solve-note", 0.0).expect("decision");
        let (mean, _) = decision.evaluation().draws()[0].equilibrium().expect("compiled mean");
        assert!(close(mean[1], 1.0));
        assert!(close(decision.above_threshold(), 0.5));
        assert!(close(decision.at_or_below_threshold(), 0.0));
        assert!(close(decision.withheld_mass(), 0.5));
        assert!(close(decision.mass_account().total(), decision.evaluation().total_mass()));
        assert!(decision.choice().is_none());
        let refused = consume_recovery_evidence(&retained, "clamp-sample", 0.0).expect("source");
        assert!(refused.evaluation().draws().iter().all(|draw| draw.equilibrium().is_none()));
        assert!(close(refused.withheld_mass(), 1.0));
        assert!(refused.choice().is_none());
        assert_eq!(
            refused.evaluation().draws()[0].certification_fault(),
            Some(&CertificationFault::Evidence(EvidenceRefusal::IsSource))
        );
        let uncited = consume_recovery_evidence(&retained, "other-note", 0.0).expect("uncited");
        assert_eq!(
            uncited.evaluation().draws()[0].certification_fault(),
            Some(&CertificationFault::Evidence(EvidenceRefusal::NotCited))
        );
        let empty = consume_recovery_evidence(&retained, "", 0.0).expect("empty");
        assert_eq!(
            empty.evaluation().draws()[0].certification_fault(),
            Some(&CertificationFault::Evidence(EvidenceRefusal::Empty))
        );
        assert_eq!(
            empty.evaluation().draws()[1].inadmissible_reason(),
            Some(InadmissibleCoefficientDraw::Marked)
        );
        let chosen =
            consume_recovery_evidence(&plan(vec![draw(1.0, 1.0, false)]), "solve-note", 0.0)
                .expect("certified choice");
        assert_eq!(chosen.choice(), Some(ThresholdChoice::Above));
        assert!(close(chosen.above_threshold(), 1.0));
        assert!(close(chosen.withheld_mass(), 0.0));
    }

    #[test]
    fn artifact_consumer_recomputes_the_certified_equilibrium() {
        let retained = plan(vec![draw(1.0, 1.0, false)]);
        let bytes = export_retained_recovery_plan(&retained).expect("artifact");
        let decision = consume_recovery_artifact(&bytes, "solve-note", 0.5).expect("consume");
        let (mean, covariance) =
            decision.evaluation().draws()[0].equilibrium().expect("recomputed");
        assert!(close(mean[1], 1.0));
        assert!(close(covariance[3], 1.0));
        assert_eq!(decision.choice(), Some(ThresholdChoice::Above));
        let mut marker = Vec::new();
        marker.extend_from_slice(&1_u32.to_le_bytes());
        marker.extend_from_slice(&0_u32.to_le_bytes());
        let at = DRAWS_AT
            + bytes[DRAWS_AT..]
                .windows(marker.len())
                .position(|window| window == marker)
                .expect("K[1, 0]");
        let mut tampered = bytes.clone();
        tampered[at + 8..at + 16].copy_from_slice(&0.0_f64.to_bits().to_le_bytes());
        // A flipped coefficient no longer matches the integrity word.
        assert_eq!(
            consume_recovery_artifact(&tampered, "solve-note", 0.5).expect_err("tampered"),
            RecoveryInterventionFault::Integrity
        );
        // A writer that seals the changed coefficient is recomputed, not read back.
        let resealed = sealed(tampered[..tampered.len() - 8].to_vec());
        let recomputed = consume_recovery_artifact(&resealed, "solve-note", 0.5).expect("resealed");
        let (mean, _) = recomputed.evaluation().draws()[0].equilibrium().expect("recompiled");
        assert!(close(mean[1], 0.0));
        assert_eq!(recomputed.choice(), Some(ThresholdChoice::AtOrBelow));
    }

    #[test]
    fn retained_plan_round_trips_through_bytes_in_a_fresh_load() {
        let retained = plan(vec![
            draw(1.0, 0.3, false),
            draw_with(0.25, 2.0, 0.3, false),
            draw(1.0, 0.2, true),
            draw(1.0, 0.2, false),
        ]);
        let bytes = export_retained_recovery_plan(&retained).expect("artifact");
        let loaded = read_plan(&bytes).expect("fresh load");
        assert_eq!(loaded, retained);
        assert_eq!(export_retained_recovery_plan(&loaded).expect("re-export"), bytes);
        assert_eq!(
            consume_retained_recovery_plan(&loaded),
            consume_retained_recovery_plan(&retained)
        );
        let decision = consume_recovery_artifact(&bytes, "solve-note", 0.0).expect("consume");
        assert!(close(decision.mass_account().total(), 1.0));
        assert!(close(decision.withheld_mass(), 0.2));
    }

    #[test]
    fn unresolved_draw_is_not_compiled_into_a_mechanism_matrix() {
        let (noise_mean, noise_covariance) = noise(1.0);
        let block = AssembledCoefficientRecovery::NullDirections {
            target: node(1),
            sources: vec![node(0)],
            column_rank: 0,
            directions: vec![vec![1.0]],
            full_rank_was_asserted: true,
        };
        let draw = joint_draw_from_blocks(
            &[node(0), node(1)],
            &[block],
            vec![0.0, 0.0],
            noise_mean,
            noise_covariance,
            1.0,
        )
        .expect("unresolved draw");
        assert!(draw.coefficients().is_empty());
        let retained = plan(vec![RecoveryDrawInput {
            coefficients: draw.coefficients().to_vec(),
            intercept: draw.intercept().to_vec(),
            noise_mean: draw.noise_mean().to_vec(),
            noise_covariance: draw.noise_covariance().to_vec(),
            unresolved: draw.unresolved().to_vec(),
            weight: draw.weight(),
            marked_inadmissible: draw.marked_inadmissible(),
        }]);
        let evaluation = consume_retained_recovery_plan(&retained);
        assert!(evaluation.draws()[0].equilibrium().is_none());
        assert_eq!(
            evaluation.draws()[0].inadmissible_reason(),
            Some(InadmissibleCoefficientDraw::UnresolvedDirections)
        );
        assert!(close(evaluation.withheld_mass(), 1.0));
        let bytes = export_retained_recovery_plan(&retained).expect("artifact");
        let decision = consume_recovery_artifact(&bytes, "solve-note", 0.0).expect("consume");
        assert!(decision.evaluation().draws()[0].equilibrium().is_none());
        assert_eq!(
            decision.evaluation().draws()[0].inadmissible_reason(),
            Some(InadmissibleCoefficientDraw::UnresolvedDirections)
        );
        assert!(close(decision.withheld_mass(), 1.0));
        assert!(decision.choice().is_none());
    }

    fn three_node_draw(coefficients: [f64; 6], weight: f64) -> RecoveryDrawInput {
        let pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
        RecoveryDrawInput {
            coefficients: pairs
                .iter()
                .zip(coefficients)
                .map(|(&(into, from), value)| coefficient(into, from, value))
                .collect(),
            intercept: vec![0.0; 3],
            noise_mean: vec![0.0; 3],
            noise_covariance: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            unresolved: Vec::new(),
            weight,
            marked_inadmissible: false,
        }
    }

    fn three_node_plan(draws: Vec<RecoveryDrawInput>) -> RetainedRecoveryPlan {
        let nodes = [node(0), node(1), node(2)];
        let uncertainty = RecoveredCoefficientUncertainty::retain(&nodes, draws).expect("law");
        RetainedRecoveryPlan::try_new(RecoveryPlanParts {
            uncertainty,
            clamp_nodes: vec![2],
            clamp_values: vec![1.0],
            outcome: 0,
            source_id: "clamp-sample".to_owned(),
            certifying_evidence: vec!["solve-note".to_owned()],
        })
        .expect("plan")
    }

    #[test]
    fn singular_clamp_draw_keeps_its_fault_and_its_mass() {
        // A = B + C, B = A, C = A. do(C) leaves the singular pair (A, B).
        let retained = three_node_plan(vec![
            three_node_draw([0.1, 0.1, 0.1, 0.0, 0.1, 0.0], 0.5),
            three_node_draw([1.0, 1.0, 1.0, 0.0, 1.0, 0.0], 0.5),
        ]);
        let evaluation = consume_retained_recovery_plan(&retained);
        assert!(evaluation.draws()[0].equilibrium().is_some());
        assert!(matches!(
            evaluation.draws()[1].certification_fault(),
            Some(CertificationFault::Request(InterventionRequestError::NotSolvable {
                refusal: SolvabilityRefusal::Singular { members },
            })) if members == &vec![0, 1]
        ));
        assert!(close(evaluation.withheld_mass(), 0.5));
        let decision = consume_recovery_evidence(&retained, "solve-note", 0.0).expect("decision");
        assert!(close(decision.withheld_mass(), 0.5));
        assert!(close(decision.mass_account().total(), 1.0));
        assert!(decision.choice().is_none());
    }

    #[test]
    fn unstable_equilibrium_draw_is_not_certified_and_keeps_its_mass() {
        // K = [[0, 2], [2, 0]] is uniquely solvable, so the clamp is not the
        // fault. rho(|K|) = 2, so the family certificate is.
        let unstable = RecoveryDrawInput {
            coefficients: vec![coefficient(0, 1, 2.0), coefficient(1, 0, 2.0)],
            ..draw(1.0, 0.25, false)
        };
        let retained = plan(vec![draw(1.0, 0.75, false), unstable]);
        let evaluation = consume_retained_recovery_plan(&retained);
        assert!(evaluation.draws()[0].equilibrium().is_some());
        assert!(matches!(
            evaluation.draws()[1].certification_fault(),
            Some(CertificationFault::FamilyNotAdmitted { rho_abs_upper }) if *rho_abs_upper >= 1.0
        ));
        assert!(matches!(
            evaluation.draws()[1].outcome(),
            EvaluatedOutcome::NotCertified(CertificationFault::FamilyNotAdmitted { .. })
        ));
        assert!(close(evaluation.withheld_mass(), 0.25));
    }

    #[test]
    fn non_gaussian_disturbance_stays_distinct_from_the_clamp_and_the_family() {
        let indefinite = RecoveryDrawInput {
            noise_covariance: vec![1.0, 2.0, 2.0, 1.0],
            ..draw(1.0, 0.5, false)
        };
        let retained = plan(vec![draw(1.0, 0.5, false), indefinite]);
        let evaluation = consume_retained_recovery_plan(&retained);
        assert!(matches!(
            evaluation.draws()[1].certification_fault(),
            Some(CertificationFault::Request(InterventionRequestError::Compile(
                SccCompileError::NotPositiveSemidefinite { .. }
            )))
        ));
        assert!(close(evaluation.withheld_mass(), 0.5));
    }

    #[test]
    fn plan_refuses_weights_that_are_not_masses_or_do_not_sum_to_one() {
        let nodes = [node(0), node(1)];
        let retain = |draws| RecoveredCoefficientUncertainty::retain(&nodes, draws).expect("law");
        let heavy = retain(vec![draw(1.0, 1.0, false), draw(1.0, 1.0, false)]);
        assert!(matches!(
            RetainedRecoveryPlan::try_new(parts(heavy)),
            Err(RecoveryInterventionFault::WeightsNotNormalized { total }) if close(total, 2.0)
        ));
        let light = retain(vec![draw(1.0, 0.5, false), draw(1.0, 0.25, true)]);
        assert!(matches!(
            RetainedRecoveryPlan::try_new(parts(light)),
            Err(RecoveryInterventionFault::WeightsNotNormalized { .. })
        ));
        for bad in [f64::NAN, f64::INFINITY, -0.5] {
            let law = retain(vec![draw(1.0, 1.0, false), draw(1.0, bad, false)]);
            assert_eq!(
                RetainedRecoveryPlan::try_new(parts(law)).expect_err("bad weight"),
                RecoveryInterventionFault::WeightNotMass { draw: 1 }
            );
        }
    }

    #[test]
    fn evaluation_exposes_within_draw_noise_and_between_draw_spread() {
        let retained =
            plan(vec![draw_with(0.5, 1.0, 0.5, false), draw_with(0.25, 3.0, 0.5, false)]);
        let evaluation = consume_retained_recovery_plan(&retained);
        let moments = evaluation.draws()[1].outcome_moments(1).expect("moments");
        assert!(close(moments.mean, 0.5));
        assert!(close(moments.equilibrium_variance, 3.0));
        assert!(close(moments.disturbance_variance, 3.0));
        let (_, disturbance) = evaluation.draws()[1].disturbance().expect("disturbance");
        assert!(close(disturbance[1], 0.3));
        let mixture = evaluation.outcome_mixture().expect("mixture");
        assert!(close(mixture.certified_mass, 1.0));
        assert!(close(mixture.mean, 0.75));
        assert!(close(mixture.within_draw_variance, 2.0));
        assert!(close(mixture.between_draw_variance, 0.0625));
        assert!(close(mixture.variance(), 2.0625));
        assert!(close(mixture.disturbance_variance, 2.0));
    }

    #[test]
    fn mixture_reports_withheld_mass_beside_the_certified_moments() {
        let retained = plan(vec![draw(1.0, 0.8, false), draw(1.0, 0.2, true)]);
        let mixture = consume_retained_recovery_plan(&retained).outcome_mixture().expect("mixture");
        assert!(close(mixture.certified_mass, 0.8));
        assert!(close(mixture.withheld_mass, 0.2));
    }

    #[test]
    fn huge_counts_with_a_tiny_buffer_are_refused_before_allocating() {
        let counts_at = |build: &dyn Fn(&mut Vec<u8>)| {
            let mut body = header();
            build(&mut body);
            sealed(body)
        };
        // Four billion nodes in a buffer that holds none.
        let nodes = counts_at(&|body| body.extend_from_slice(&u32::MAX.to_le_bytes()));
        assert_eq!(read_plan(&nodes).expect_err("nodes"), RecoveryInterventionFault::Artifact);
        // A valid node list, then four billion clamps.
        let clamps = counts_at(&|body| {
            body.extend_from_slice(&1_u32.to_le_bytes());
            body.extend_from_slice(&0_u32.to_le_bytes());
            body.extend_from_slice(&u32::MAX.to_le_bytes());
        });
        assert_eq!(read_plan(&clamps).expect_err("clamps"), RecoveryInterventionFault::Artifact);
        // Counts whose byte size overflows nothing but exceeds what remains.
        for claimed in [1_u32 << 31, 1 << 28, 1 << 20, 4] {
            let bytes = counts_at(&|body| body.extend_from_slice(&claimed.to_le_bytes()));
            assert_eq!(read_plan(&bytes).expect_err("count"), RecoveryInterventionFault::Artifact);
        }
        // A string longer than the buffer.
        let string = counts_at(&|body| {
            for word in [1_u32, 0, 0, 0] {
                body.extend_from_slice(&word.to_le_bytes());
            }
            body.extend_from_slice(&u32::MAX.to_le_bytes());
        });
        assert_eq!(read_plan(&string).expect_err("string"), RecoveryInterventionFault::Artifact);
    }

    #[test]
    fn huge_inner_counts_in_a_draw_are_refused() {
        let retained = plan(vec![draw(1.0, 1.0, false)]);
        let bytes = export_retained_recovery_plan(&retained).expect("artifact");
        // A draw is its weight, its mark, then the coefficient count.
        let coefficient_count_at = DRAWS_AT + 8 + 1;
        let mut body = bytes[..bytes.len() - 8].to_vec();
        body[coefficient_count_at..coefficient_count_at + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            read_plan(&sealed(body)).expect_err("coefficients"),
            RecoveryInterventionFault::Artifact
        );
    }

    #[test]
    fn mark_byte_other_than_zero_or_one_is_refused() {
        let retained = plan(vec![draw(1.0, 1.0, false)]);
        let bytes = export_retained_recovery_plan(&retained).expect("artifact");
        let mark_at = DRAWS_AT + 8;
        assert_eq!(bytes[mark_at], 0);
        for mark in [2_u8, 0x80, 0xff] {
            let mut body = bytes[..bytes.len() - 8].to_vec();
            body[mark_at] = mark;
            assert_eq!(
                read_plan(&sealed(body)).expect_err("mark"),
                RecoveryInterventionFault::Artifact
            );
        }
        let mut body = bytes[..bytes.len() - 8].to_vec();
        body[mark_at] = 1;
        let marked = read_plan(&sealed(body)).expect("a one is a mark");
        assert!(marked.uncertainty().draws()[0].marked_inadmissible());
    }

    #[test]
    fn every_truncation_and_every_flipped_byte_is_refused() {
        let retained = plan(vec![draw(1.0, 0.5, false), draw(2.0, 0.5, true)]);
        let bytes = export_retained_recovery_plan(&retained).expect("artifact");
        for end in 0..bytes.len() {
            assert!(read_plan(&bytes[..end]).is_err(), "prefix of {end} bytes");
        }
        for at in 0..bytes.len() {
            let mut flipped = bytes.clone();
            flipped[at] ^= 0x01;
            assert!(read_plan(&flipped).is_err(), "flip at {at}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(read_plan(&longer).is_err());
    }

    #[test]
    fn version_and_magic_are_checked_before_the_body() {
        let retained = plan(vec![draw(1.0, 1.0, false)]);
        let bytes = export_retained_recovery_plan(&retained).expect("artifact");
        let mut other = bytes[..bytes.len() - 8].to_vec();
        other[4..8].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(
            read_plan(&sealed(other)).expect_err("version"),
            RecoveryInterventionFault::UnsupportedVersion { found: 2 }
        );
        let mut wrong = bytes[..bytes.len() - 8].to_vec();
        wrong[0] = b'X';
        assert_eq!(
            read_plan(&sealed(wrong)).expect_err("magic"),
            RecoveryInterventionFault::Artifact
        );
    }

    fn blocks() -> Vec<AssembledCoefficientRecovery> {
        vec![
            AssembledCoefficientRecovery::Coefficients {
                target: node(0),
                sources: vec![node(1)],
                values: vec![0.4],
            },
            AssembledCoefficientRecovery::Coefficients {
                target: node(1),
                sources: vec![node(0)],
                values: vec![0.5],
            },
        ]
    }

    fn covariances() -> Vec<CoefficientCovariance> {
        vec![
            CoefficientCovariance { block: 0, covariance: vec![0.01] },
            CoefficientCovariance { block: 1, covariance: vec![0.04] },
        ]
    }

    fn generation<'a>(
        nodes: &'a [VariableId],
        blocks: &'a [AssembledCoefficientRecovery],
        covariances: &'a [CoefficientCovariance],
        noise: &'a (Vec<f64>, Vec<f64>),
        draws: usize,
        seed: u64,
    ) -> DrawGeneration<'a> {
        DrawGeneration {
            nodes,
            blocks,
            covariances,
            intercept: &[0.0, 0.0],
            noise_mean: &noise.0,
            noise_covariance: &noise.1,
            draws,
            seed,
        }
    }

    #[test]
    fn generated_draws_are_seeded_equally_weighted_and_spread_by_the_covariance() {
        let nodes = [node(0), node(1)];
        let (blocks, covariances, noise) = (blocks(), covariances(), noise(1.0));
        let spec = generation(&nodes, &blocks, &covariances, &noise, 400, 17);
        let draws = asymptotic_normal_draws(&spec).expect("draws");
        assert_eq!(draws, asymptotic_normal_draws(&spec).expect("again"));
        let other = generation(&nodes, &blocks, &covariances, &noise, 400, 18);
        assert_ne!(draws, asymptotic_normal_draws(&other).expect("other seed"));
        assert_eq!(draws.len(), 400);
        assert!(draws.iter().all(|draw| close(draw.weight, 1.0 / 400.0)));
        assert!(draws.iter().all(|draw| draw.noise_covariance == noise.1));
        let into_a: Vec<f64> = draws.iter().map(|draw| draw.coefficients[0].1).collect();
        let mean = into_a.iter().sum::<f64>() / 400.0;
        let variance = into_a.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 399.0;
        assert!((mean - 0.4).abs() < 0.03, "mean {mean}");
        assert!((variance - 0.01).abs() < 0.004, "variance {variance}");
        // The first 100 draws do not depend on how many were requested.
        let fewer = generation(&nodes, &blocks, &covariances, &noise, 100, 17);
        let prefix = asymptotic_normal_draws(&fewer).expect("fewer");
        assert_eq!(prefix[0].coefficients, draws[0].coefficients);
    }

    #[test]
    fn generated_draws_feed_the_certified_evaluation_with_spread_between_draws() {
        let nodes = [node(0), node(1)];
        let (blocks, covariances, noise) = (blocks(), covariances(), noise(1.0));
        let spec = generation(&nodes, &blocks, &covariances, &noise, 200, 5);
        let draws = asymptotic_normal_draws(&spec).expect("draws");
        let uncertainty = RecoveredCoefficientUncertainty::retain(&nodes, draws).expect("law");
        let retained = RetainedRecoveryPlan::try_new(parts(uncertainty)).expect("plan");
        let evaluation = consume_retained_recovery_plan(&retained);
        assert!(close(evaluation.total_mass(), 1.0));
        let mixture = evaluation.outcome_mixture().expect("mixture");
        assert!(mixture.between_draw_variance > 0.0);
        assert!(close(mixture.within_draw_variance, 1.0));
        assert!((mixture.mean - 1.0).abs() < 0.2, "mean {}", mixture.mean);
    }

    #[test]
    fn draw_generation_refuses_what_it_cannot_draw() {
        let nodes = [node(0), node(1)];
        let (blocks, covariances, noise) = (blocks(), covariances(), noise(1.0));
        let spec = generation(&nodes, &blocks, &covariances, &noise, 0, 1);
        assert_eq!(
            asymptotic_normal_draws(&spec).expect_err("no draws"),
            RecoveryInterventionFault::DrawCount
        );
        let spec = generation(&nodes, &blocks, &covariances[..1], &noise, 2, 1);
        assert_eq!(
            asymptotic_normal_draws(&spec).expect_err("missing"),
            RecoveryInterventionFault::MissingCovariance { block: 1 }
        );
        let indefinite = vec![
            CoefficientCovariance { block: 0, covariance: vec![-0.01] },
            covariances[1].clone(),
        ];
        let spec = generation(&nodes, &blocks, &indefinite, &noise, 2, 1);
        assert_eq!(
            asymptotic_normal_draws(&spec).expect_err("negative variance"),
            RecoveryInterventionFault::BlockCovariance { block: 0 }
        );
        let repeated = vec![covariances[0].clone(), covariances[0].clone(), covariances[1].clone()];
        let spec = generation(&nodes, &blocks, &repeated, &noise, 2, 1);
        assert_eq!(
            asymptotic_normal_draws(&spec).expect_err("repeated"),
            RecoveryInterventionFault::BlockCovariance { block: 0 }
        );
        let misshaped = vec![
            CoefficientCovariance { block: 0, covariance: vec![0.01, 0.0, 0.0, 0.01] },
            covariances[1].clone(),
        ];
        let spec = generation(&nodes, &blocks, &misshaped, &noise, 2, 1);
        assert_eq!(
            asymptotic_normal_draws(&spec).expect_err("shape"),
            RecoveryInterventionFault::BlockCovariance { block: 0 }
        );
        let unsolved =
            vec![blocks[0].clone(), AssembledCoefficientRecovery::Inconsistent { column_rank: 1 }];
        let spec = generation(&nodes, &unsolved, &covariances[..1], &noise, 2, 1);
        assert_eq!(
            asymptotic_normal_draws(&spec).expect_err("inconsistent block"),
            RecoveryInterventionFault::Uncertainty(UncertaintyFault::BlockInconsistent {
                block: 1,
                column_rank: 1,
            })
        );
    }

    #[test]
    fn correlated_coefficients_are_drawn_jointly() {
        let nodes = [node(0), node(1), node(2)];
        let blocks = vec![AssembledCoefficientRecovery::Coefficients {
            target: node(0),
            sources: vec![node(1), node(2)],
            values: vec![0.1, 0.1],
        }];
        let covariances =
            vec![CoefficientCovariance { block: 0, covariance: vec![0.04, 0.036, 0.036, 0.04] }];
        let noise = (vec![0.0; 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let spec = DrawGeneration {
            nodes: &nodes,
            blocks: &blocks,
            covariances: &covariances,
            intercept: &[0.0; 3],
            noise_mean: &noise.0,
            noise_covariance: &noise.1,
            draws: 2000,
            seed: 3,
        };
        let draws = asymptotic_normal_draws(&spec).expect("draws");
        let (mut sum_a, mut sum_b) = (0.0, 0.0);
        for draw in &draws {
            sum_a += draw.coefficients[0].1;
            sum_b += draw.coefficients[1].1;
        }
        let (mean_a, mean_b) = (sum_a / 2000.0, sum_b / 2000.0);
        let covariance = draws
            .iter()
            .map(|draw| (draw.coefficients[0].1 - mean_a) * (draw.coefficients[1].1 - mean_b))
            .sum::<f64>()
            / 1999.0;
        assert!((covariance - 0.036).abs() < 0.008, "covariance {covariance}");
    }

    #[test]
    fn a_rank_deficient_draw_is_not_a_joint_intervention_mechanism() {
        let nodes = [node(0), node(1), node(2)];
        let block = AssembledCoefficientRecovery::NullDirections {
            target: node(2),
            sources: vec![node(0), node(1)],
            column_rank: 1,
            directions: vec![vec![-1.0, 1.0]],
            full_rank_was_asserted: true,
        };
        let draw = joint_draw_from_blocks(
            &nodes,
            &[block],
            vec![0.0; 3],
            vec![0.0; 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            1.0,
        )
        .expect("null directions stay in the law");
        assert!(
            antecedent_identify::experimental_recovery::as_recovered_mechanism(&draw).is_none()
        );
        let uncertainty = RecoveredCoefficientUncertainty::retain(
            &nodes,
            vec![RecoveryDrawInput {
                coefficients: draw.coefficients().to_vec(),
                intercept: draw.intercept().to_vec(),
                noise_mean: draw.noise_mean().to_vec(),
                noise_covariance: draw.noise_covariance().to_vec(),
                unresolved: draw.unresolved().to_vec(),
                weight: 1.0,
                marked_inadmissible: false,
            }],
        )
        .expect("law");
        let plan = RetainedRecoveryPlan::try_new(RecoveryPlanParts {
            uncertainty,
            clamp_nodes: vec![0, 1],
            clamp_values: vec![2.0, 1.25],
            outcome: 2,
            source_id: "held-out-joint".to_owned(),
            certifying_evidence: vec!["solve-note".to_owned()],
        })
        .expect("plan");
        let evaluation = consume_retained_recovery_plan(&plan);
        assert!(evaluation.draws()[0].equilibrium().is_none());
        assert_eq!(
            evaluation.draws()[0].inadmissible_reason(),
            Some(InadmissibleCoefficientDraw::UnresolvedDirections)
        );
    }
}
