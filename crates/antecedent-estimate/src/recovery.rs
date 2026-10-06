//! Exact evaluation of a checked observation-recovery formula (2.2B X10) and of
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
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{CatalogDistribution, ExecutionContext, Value, VariableId};
use antecedent_expr::{
    Assignment, DiscreteAxis, EvalContext, ExactDiscreteLaw, ExactDistribution,
    ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData, LawOrigin,
};
use antecedent_identify::recovery::MISSING_LEVEL;
use antecedent_identify::{
    RECOVERY_MAX_OBSERVED_CELLS, RecoveryDerivation, RecoveryDetail, RecoveryError,
};

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
    law: ExactDiscreteLaw,
    descriptor: CatalogDistribution,
    derivation: String,
}

impl RecoveredLaw {
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
#[allow(clippy::too_many_lines)] // One pass: identity, bounds, axes, levels and the proxy model.
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
        effect.arena(),
        effect.root(),
        data,
        effect.outcomes().to_vec(),
        request,
        limits,
        law.tolerance(),
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
