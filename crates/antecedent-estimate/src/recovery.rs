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
    DirectCoefficient, DrawAdmissibility, InadmissibleCoefficientDraw, JointRecoveryDraw,
    RecoveredCoefficientUncertainty, RecoveryDrawInput, UncertaintyFault, UnresolvedDirection,
    UnresolvedReason,
};
use antecedent_identify::recovery::MISSING_LEVEL;
use antecedent_identify::{
    RECOVERY_MAX_OBSERVED_CELLS, RecoveryDerivation, RecoveryDetail, RecoveryError,
};
use antecedent_model::cyclic_component::{CyclicComponent, CyclicComponentError};
use antecedent_model::intervention_request::{InterventionRequest, compile_under_request};
use antecedent_model::scc_compile::{CompileBudget, GaussianDisturbance};
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
const ARTIFACT_MAGIC: &[u8; 4] = b"RCVY";

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
    /// coordinate of the law, or when the source id is empty.
    pub fn try_new(parts: RecoveryPlanParts) -> Result<Self, RecoveryInterventionFault> {
        let dimension = parts.uncertainty.nodes().len();
        if parts.outcome >= dimension {
            return Err(RecoveryInterventionFault::Outcome { outcome: parts.outcome, dimension });
        }
        if parts.source_id.is_empty() {
            return Err(RecoveryInterventionFault::EmptySource);
        }
        check_clamp(dimension, &parts.clamp_nodes, &parts.clamp_values)?;
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
    #[must_use]
    pub const fn uncertainty(&self) -> &RecoveredCoefficientUncertainty {
        &self.uncertainty
    }

    /// Outcome coordinate.
    #[must_use]
    pub const fn outcome(&self) -> usize {
        self.outcome
    }
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
            EvaluatedOutcome::Equilibrium { mean, covariance } => Some((mean, covariance)),
            EvaluatedOutcome::Inadmissible(_) | EvaluatedOutcome::NotCertified { .. } => None,
        }
    }

    /// Inadmissible status, when the coefficient law already refused the draw.
    #[must_use]
    pub const fn inadmissible_reason(&self) -> Option<InadmissibleCoefficientDraw> {
        match self.outcome {
            EvaluatedOutcome::Inadmissible(reason) => Some(reason),
            EvaluatedOutcome::Equilibrium { .. } | EvaluatedOutcome::NotCertified { .. } => None,
        }
    }
}

/// Result of compiling one draw. Failures stay attached to the draw.
#[derive(Clone, Debug, PartialEq)]
pub enum EvaluatedOutcome {
    /// Certified Gaussian equilibrium under the retained hard clamp.
    Equilibrium {
        /// Equilibrium mean, in node order.
        mean: Vec<f64>,
        /// Row-major equilibrium covariance.
        covariance: Vec<f64>,
    },
    /// The coefficient law marked the draw inadmissible. No matrix was completed.
    Inadmissible(InadmissibleCoefficientDraw),
    /// Compilation or evidence did not certify the draw. The weight remains.
    NotCertified {
        /// Why certification stopped.
        detail: &'static str,
    },
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

    /// Mass whose status is inadmissible or uncertified.
    #[must_use]
    pub fn withheld_mass(&self) -> f64 {
        self.draws
            .iter()
            .filter(|draw| draw.equilibrium().is_none())
            .map(|draw| positive_weight(draw.weight))
            .sum()
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
    #[must_use]
    pub const fn evaluation(&self) -> &CertifiedInterventionEvaluation {
        &self.evaluation
    }

    /// Mass of certified draws whose outcome mean is strictly above the threshold.
    #[must_use]
    pub fn above_threshold(&self) -> f64 {
        self.partition().0
    }

    /// Mass of certified draws whose outcome mean is finite and not strictly above.
    #[must_use]
    pub fn at_or_below_threshold(&self) -> f64 {
        self.partition().1
    }

    /// Mass that was inadmissible, uncertified, or had no finite outcome mean.
    #[must_use]
    pub fn withheld_mass(&self) -> f64 {
        self.partition().2
    }

    /// A single action, only when one side holds every unit of mass.
    #[must_use]
    pub fn choice(&self) -> Option<ThresholdChoice> {
        let (above, below, withheld) = self.partition();
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

    fn partition(&self) -> (f64, f64, f64) {
        let mut above = 0.0;
        let mut below = 0.0;
        let mut withheld = 0.0;
        for draw in self.evaluation.draws() {
            let weight = positive_weight(draw.weight());
            match draw.equilibrium().and_then(|(mean, _)| mean.get(self.evaluation.outcome())) {
                Some(value) if value.is_finite() && *value > self.threshold => above += weight,
                Some(value) if value.is_finite() => below += weight,
                Some(_) | None => withheld += weight,
            }
        }
        (above, below, withheld)
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
    let evaluation = if evidence_certifies(plan, evidence_id) {
        consume_retained_recovery_plan(plan)
    } else {
        withhold_for_evidence(plan)
    };
    Ok(RecoveryEvidenceDecision { evaluation, threshold })
}

/// Encode `plan` for an independent consumer.
///
/// # Errors
///
/// [`RecoveryInterventionFault::IndexOverflow`] when a length does not fit in `u32`.
pub fn export_retained_recovery_plan(
    plan: &RetainedRecoveryPlan,
) -> Result<Vec<u8>, RecoveryInterventionFault> {
    let mut buf = Vec::new();
    buf.extend_from_slice(ARTIFACT_MAGIC);
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
pub fn consume_recovery_artifact(
    bytes: &[u8],
    evidence_id: &str,
    threshold: f64,
) -> Result<RecoveryEvidenceDecision, RecoveryInterventionFault> {
    let plan = read_plan(bytes)?;
    consume_recovery_evidence(&plan, evidence_id, threshold)
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

fn evidence_certifies(plan: &RetainedRecoveryPlan, evidence_id: &str) -> bool {
    !evidence_id.is_empty()
        && evidence_id != plan.source_id
        && plan.certifying_evidence.iter().any(|cited| cited == evidence_id)
}

fn withhold_for_evidence(plan: &RetainedRecoveryPlan) -> CertifiedInterventionEvaluation {
    let draws = plan
        .uncertainty
        .draws()
        .iter()
        .map(|draw| EvaluatedRecoveryDraw {
            weight: draw.weight(),
            outcome: match draw.admissibility() {
                DrawAdmissibility::Inadmissible(reason) => EvaluatedOutcome::Inadmissible(reason),
                DrawAdmissibility::Admissible => EvaluatedOutcome::NotCertified {
                    detail: "presented evidence does not certify the component",
                },
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
    match compile_draw(plan, draw) {
        Ok((mean, covariance)) => EvaluatedRecoveryDraw {
            weight,
            outcome: EvaluatedOutcome::Equilibrium { mean, covariance },
        },
        Err(detail) => {
            EvaluatedRecoveryDraw { weight, outcome: EvaluatedOutcome::NotCertified { detail } }
        }
    }
}

fn compile_draw(
    plan: &RetainedRecoveryPlan,
    draw: &JointRecoveryDraw,
) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    let (coefficients, intercept) =
        mechanism_matrix(plan.uncertainty.nodes(), draw.coefficients(), draw.intercept())?;
    let model = LinearStructuralMap::try_new(coefficients, intercept)
        .map_err(|_| "structural map was refused")?;
    let disturbance = GaussianDisturbance::try_new(
        draw.noise_mean().to_vec(),
        draw.noise_covariance().to_vec(),
    )
    .map_err(|_| "disturbance was not a Gaussian law")?;
    let open = HardClamp::none(model.dimension()).map_err(|_| "structural map was refused")?;
    let assessment = assess(&model, &open).map_err(|_| "structural map was refused")?;
    if !assessment.family.is_admitted() {
        return Err("family certificate was not admitted");
    }
    let evidence: Vec<&str> = plan.certifying_evidence.iter().map(String::as_str).collect();
    let component = CyclicComponent::try_new(
        RECOVERED_EQUILIBRIUM,
        &plan.source_id,
        &evidence,
        &model,
        &assessment.family,
    )
    .map_err(|error| match error {
        CyclicComponentError::FamilyNotAdmitted { .. } => "family certificate was not admitted",
        _ => "component was not admitted",
    })?;
    let request =
        InterventionRequest::hard_clamp(model.dimension(), &plan.clamp_nodes, &plan.clamp_values)
            .map_err(|_| "intervention request was refused")?;
    let equilibrium = compile_under_request(
        &component,
        &model,
        &disturbance,
        &request,
        CompileBudget::unlimited(),
    )
    .map_err(|_| "intervention compile was refused")?;
    Ok((equilibrium.mean().to_vec(), equilibrium.covariance().to_vec()))
}

fn mechanism_matrix(
    nodes: &[VariableId],
    coefficients: &[(DirectCoefficient, f64)],
    intercept: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    let n = nodes.len();
    if intercept.len() != n {
        return Err("intercept does not match the nodes");
    }
    let Some(len) = n.checked_mul(n) else {
        return Err("mechanism matrix was not complete");
    };
    let mut matrix = vec![0.0; len];
    let mut filled = vec![false; len];
    for (coefficient, value) in coefficients {
        let Some(into) = nodes.iter().position(|node| *node == coefficient.into) else {
            return Err("coefficient names a node outside the law");
        };
        let Some(from) = nodes.iter().position(|node| *node == coefficient.from) else {
            return Err("coefficient names a node outside the law");
        };
        if into == from {
            return Err("coefficient lies on the structural diagonal");
        }
        let index = into * n + from;
        if filled[index] {
            return Err("coefficient was named twice");
        }
        filled[index] = true;
        matrix[index] = *value;
    }
    for into in 0..n {
        for from in 0..n {
            if into != from && !filled[into * n + from] {
                return Err("mechanism matrix was not complete");
            }
        }
    }
    Ok((matrix, intercept.to_vec()))
}

fn positive_weight(weight: f64) -> f64 {
    if weight.is_finite() && weight > 0.0 { weight } else { 0.0 }
}

fn is_zero(value: f64) -> bool {
    value.to_bits() == 0.0_f64.to_bits() || value.to_bits() == (-0.0_f64).to_bits()
}

#[allow(clippy::unnecessary_wraps, reason = "callers chain it with fallible length conversions")]
fn write_u32(buf: &mut Vec<u8>, value: u32) -> Result<(), RecoveryInterventionFault> {
    buf.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_f64(buf: &mut Vec<u8>, value: f64) {
    buf.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn write_str(buf: &mut Vec<u8>, value: &str) -> Result<(), RecoveryInterventionFault> {
    let len = u32::try_from(value.len()).map_err(|_| RecoveryInterventionFault::IndexOverflow)?;
    write_u32(buf, len)?;
    buf.extend_from_slice(value.as_bytes());
    Ok(())
}

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

fn write_slice(buf: &mut Vec<u8>, values: &[f64]) -> Result<(), RecoveryInterventionFault> {
    write_u32(buf, u32::try_from(values.len())?)?;
    for value in values {
        write_f64(buf, *value);
    }
    Ok(())
}

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

fn read_plan(bytes: &[u8]) -> Result<RetainedRecoveryPlan, RecoveryInterventionFault> {
    let mut reader = Reader { rest: bytes };
    let magic = reader.take(4)?;
    if magic != ARTIFACT_MAGIC {
        return Err(RecoveryInterventionFault::Artifact);
    }
    let node_count = reader.u32()? as usize;
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(VariableId::from_raw(reader.u32()?));
    }
    let clamp_count = reader.u32()? as usize;
    let mut clamp_nodes = Vec::with_capacity(clamp_count);
    let mut clamp_values = Vec::with_capacity(clamp_count);
    for _ in 0..clamp_count {
        clamp_nodes.push(reader.u32()? as usize);
        clamp_values.push(reader.f64()?);
    }
    let outcome = reader.u32()? as usize;
    let source_id = reader.string()?;
    let evidence_count = reader.u32()? as usize;
    let mut certifying_evidence = Vec::with_capacity(evidence_count);
    for _ in 0..evidence_count {
        certifying_evidence.push(reader.string()?);
    }
    let draw_count = reader.u32()? as usize;
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

struct Reader<'a> {
    rest: &'a [u8],
}

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

    fn string(&mut self) -> Result<String, RecoveryInterventionFault> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| RecoveryInterventionFault::Artifact)
    }

    fn draw(&mut self) -> Result<RecoveryDrawInput, RecoveryInterventionFault> {
        let weight = self.f64()?;
        let marked = self.take(1)?[0] == 1;
        let coefficient_count = self.u32()? as usize;
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
        let count = self.u32()? as usize;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.f64()?);
        }
        Ok(values)
    }

    fn directions(&mut self) -> Result<Vec<UnresolvedDirection>, RecoveryInterventionFault> {
        let count = self.u32()? as usize;
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
        let support_count = self.u32()? as usize;
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
        let count = self.u32()? as usize;
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
        EvaluatedOutcome, RecoveryPlanParts, RetainedRecoveryPlan, ThresholdChoice,
        consume_recovery_artifact, consume_recovery_evidence, consume_retained_recovery_plan,
        export_retained_recovery_plan,
    };
    use antecedent_core::VariableId;
    use antecedent_identify::experimental_recovery::{
        AssembledCoefficientRecovery, DirectCoefficient, InadmissibleCoefficientDraw,
        RecoveredCoefficientUncertainty, RecoveryDrawInput, joint_draw_from_blocks,
    };

    fn node(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn coefficient(into: u32, from: u32, value: f64) -> (DirectCoefficient, f64) {
        (DirectCoefficient { into: node(into), from: node(from) }, value)
    }

    fn noise(variance: f64) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0, 0.0], vec![1.0, 0.3, 0.3, variance])
    }

    fn draw(variance: f64, weight: f64, marked_inadmissible: bool) -> RecoveryDrawInput {
        let (noise_mean, noise_covariance) = noise(variance);
        RecoveryDrawInput {
            coefficients: vec![coefficient(0, 1, 0.4), coefficient(1, 0, 0.5)],
            intercept: vec![0.0, 0.0],
            noise_mean,
            noise_covariance,
            unresolved: Vec::new(),
            weight,
            marked_inadmissible,
        }
    }

    fn plan(draws: Vec<RecoveryDrawInput>) -> RetainedRecoveryPlan {
        let uncertainty =
            RecoveredCoefficientUncertainty::retain(&[node(0), node(1)], draws).expect("law");
        RetainedRecoveryPlan::try_new(RecoveryPlanParts {
            uncertainty,
            clamp_nodes: vec![0],
            clamp_values: vec![2.0],
            outcome: 1,
            source_id: "clamp-sample".to_owned(),
            certifying_evidence: vec!["solve-note".to_owned()],
        })
        .expect("plan")
    }

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1.0e-8
    }

    #[test]
    fn retained_plan_keeps_parameter_noise_dependence_and_inadmissible_mass() {
        let retained =
            plan(vec![draw(1.0, 1.0, false), draw(4.0, 1.0, false), draw(1.0, 0.5, true)]);
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
        assert!(close(evaluation.withheld_mass(), 0.5));
    }

    #[test]
    fn evidence_decision_keeps_inadmissible_mass_and_withholds_the_choice() {
        let retained = plan(vec![draw(1.0, 1.0, false), draw(1.0, 1.0, true)]);
        let decision = consume_recovery_evidence(&retained, "solve-note", 0.0).expect("decision");
        let (mean, _) = decision.evaluation().draws()[0].equilibrium().expect("compiled mean");
        assert!(close(mean[1], 1.0));
        assert!(close(decision.above_threshold(), 1.0));
        assert!(close(decision.at_or_below_threshold(), 0.0));
        assert!(close(decision.withheld_mass(), 1.0));
        assert!(decision.choice().is_none());
        let refused = consume_recovery_evidence(&retained, "clamp-sample", 0.0).expect("source");
        assert!(refused.evaluation().draws().iter().all(|draw| draw.equilibrium().is_none()));
        assert!(close(refused.withheld_mass(), 2.0));
        assert!(refused.choice().is_none());
        assert!(matches!(
            refused.evaluation().draws()[0].outcome(),
            EvaluatedOutcome::NotCertified { .. }
        ));
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
        let at = bytes.windows(marker.len()).position(|window| window == marker).expect("K[1, 0]");
        let mut tampered = bytes.clone();
        tampered[at + 8..at + 16].copy_from_slice(&0.0_f64.to_bits().to_le_bytes());
        let recomputed = consume_recovery_artifact(&tampered, "solve-note", 0.5).expect("tampered");
        let (mean, _) = recomputed.evaluation().draws()[0].equilibrium().expect("recompiled");
        assert!(close(mean[1], 0.0));
        assert_eq!(recomputed.choice(), Some(ThresholdChoice::AtOrBelow));
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
}
