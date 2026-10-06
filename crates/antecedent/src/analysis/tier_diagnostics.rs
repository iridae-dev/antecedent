//! Tier-aware overlap and E-value diagnostics of an executed tiered result (2.2 E7).
//!
//! A tiered study (`.tiered_background(..)`) runs one of two licensed average-effect cells:
//!
//! * **`CoDetermined`**: same-tier nodes are bidirected, the tier closure is the
//!   generalized-adjustment set, and the AIPW estimator fits a propensity on it. The
//!   executed design is that closure set and its row identity; the overlap report is the
//!   one the estimator itself computed on that design.
//! * **`Unknown`**: within-tier orientation is not known, so the cell reports two declared
//!   orientation scenarios (treatment precedes every same-tier peer, or follows every
//!   peer) from a linear adjustment. There is no single effect, no single adjustment set
//!   and no propensity score.
//!
//! [`tier_diagnostics`] reads **only what the executed result carries** and states which
//! diagnostic is available for it. Nothing is refit, and no graph is built: in particular
//! no DAG with an assumed (or assumed absent) within-tier edge is constructed to run a
//! graph-based check on, because the tier background deliberately does not assert one.
//!
//! * **Overlap.** Available for `CoDetermined` when the estimator carried an overlap
//!   report (a propensity-based estimator). Typed unavailable for `Unknown`, and for a
//!   `CoDetermined` result whose estimator fits no propensity.
//! * **E-value, point.** The `CoDetermined` cell attaches a point E-value from the
//!   effect and the outcome SD (`RR ~ exp(0.91 d)`, VanderWeele and Ding). It is reported
//!   with its method. When the E-value on the result instead came from the E-value
//!   refuter (which reports the smaller of a point and a converted interval-limit
//!   E-value, one number), it embeds a converted endpoint and is not reported as a point
//!   E-value. Typed unavailable for `Unknown`.
//! * **E-value, interval.** Never reported: the converted effect-interval endpoint is an
//!   approximation whose own uncertainty is excluded and no coverage record measures it.
//!   It is a typed unavailable field on every result, not an omitted one.
//!
//! The E-value is a sensitivity summary of the stated no-latent-path premise, never a
//! test of it and never a tier-identification certificate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId, reason_code};
use antecedent_estimate::{Availability, OverlapReport, Unavailable};

use crate::result::StudyResult;
use crate::support::CellStatus;

/// Derivation rule of a single-treatment `CoDetermined` closure identification.
const CLOSURE_RULE: &str = "tiered.closure";

/// Derivation rule of the `Unknown` two-scenario identification.
const UNKNOWN_RULE: &str = "tiered.unknown.envelope";

/// Diagnostic code under which the `CoDetermined` cell attaches its point E-value.
const POINT_EVALUE_DIAGNOSTIC: &str = "tiered.evalue.vanderweele_approx";

/// A refusal to read tier diagnostics off a result, with its registered reason code.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct TierDiagnosticsError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `tier_diagnostics.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

/// One declared orientation scenario of an `Unknown` tiered result.
#[derive(Clone, Debug, PartialEq)]
pub struct TierScenario {
    /// Scenario estimand method (`tiered.unknown.pretreatment` or `tiered.unknown.closure`).
    pub method: Arc<str>,
    /// The adjustment set of that scenario.
    pub adjustment: Vec<VariableId>,
    /// The scenario effect as reported.
    pub effect: f64,
}

/// The design the tiered result actually executed.
#[derive(Clone, Debug, PartialEq)]
pub enum TierDesign {
    /// `CoDetermined`: one tier-closure adjustment set.
    CoDeterminedClosure {
        /// The closure adjustment set the estimand used.
        adjustment: Vec<VariableId>,
        /// Complete rows that informed the fit, when the estimator recorded them.
        rows: Option<u64>,
        /// Whether the estimator fitted a propensity score (and so has an overlap report).
        propensity_scored: bool,
    },
    /// `Unknown`: two declared orientation scenarios, never one set and never one DAG.
    UnknownScenarios {
        /// The scenarios in estimand order. They are not exhaustive completions of the
        /// unknown within-tier orientation and not bounds over it.
        scenarios: Vec<TierScenario>,
    },
}

/// A point E-value with the method that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct TierPointEvalue {
    /// The E-value: the confounding strength, on the risk-ratio scale, needed to explain
    /// away the point effect. Always at least 1.
    pub value: f64,
    /// The conversion used to put the effect on the risk-ratio scale.
    pub method: Arc<str>,
}

/// E-value diagnostics: the point value where licensed, the interval value never.
#[derive(Clone, Debug, PartialEq)]
pub struct TierEvalue {
    /// The point E-value, or the typed reason it is not carried.
    pub point: Availability<TierPointEvalue>,
    /// Why no E-value for the effect-interval limit is reported (always present).
    pub interval: Unavailable,
}

/// Which diagnostics a tiered result carries.
#[derive(Clone, Debug, PartialEq)]
pub struct TierDiagnostics {
    /// The executed design.
    pub design: TierDesign,
    /// The estimator's own overlap report, or the typed reason there is none.
    pub overlap: Availability<OverlapReport>,
    /// E-value diagnostics.
    pub evalue: TierEvalue,
}

fn refusal(code: &'static str, detail: &'static str, message: &str) -> TierDiagnosticsError {
    TierDiagnosticsError { code, detail, message: message.to_owned() }
}

fn not_tiered() -> TierDiagnosticsError {
    refusal(
        reason_code!("invalid_argument"),
        "tier_diagnostics.not_a_tiered_average_result",
        "the result is not a single-treatment tiered average-effect result: it records neither a \
         CoDetermined tier-closure identification nor the Unknown two-scenario identification \
         with scenario effects (a joint-cell or non-tiered result has no tier diagnostics)",
    )
}

/// Observe cancellation: a cancelled read reports no diagnostics, never a partial set.
fn observe(ctx: &ExecutionContext) -> Result<(), TierDiagnosticsError> {
    if ctx.cancellation.is_cancelled() {
        return Err(refusal(
            reason_code!("cancelled_no_claim"),
            "tier_diagnostics.cancelled",
            "reading the tier diagnostics was cancelled; none is reported and the stop is not a \
             verdict on the result",
        ));
    }
    Ok(())
}

fn unavailable(detail: &'static str, reason: &'static str) -> Unavailable {
    Unavailable { code: reason_code!("diagnostic_not_available"), detail, reason }
}

/// Read the tier diagnostics a licensed tiered average-effect result carries.
///
/// # Errors
/// `cell_not_licensed` (`tier_diagnostics.result_not_licensed`) when the result's support
/// status is not licensed; `invalid_argument` (`tier_diagnostics.not_a_tiered_average_result`)
/// when it is not a single-treatment `CoDetermined` closure or `Unknown` two-scenario
/// average-effect result; `cancelled_no_claim` (`tier_diagnostics.cancelled`) when `ctx` is
/// cancelled (observed on entry, before each of the two tier reads and once per scenario),
/// with no diagnostics reported.
pub fn tier_diagnostics(
    result: &StudyResult,
    ctx: &ExecutionContext,
) -> Result<TierDiagnostics, TierDiagnosticsError> {
    observe(ctx)?;
    if !matches!(result.support_status, Some(CellStatus::Licensed)) {
        return Err(refusal(
            reason_code!("cell_not_licensed"),
            "tier_diagnostics.result_not_licensed",
            "tier diagnostics are read only off a result of a licensed support cell",
        ));
    }
    let derived = |rule: &str| {
        result.identification.derivation.steps.iter().any(|step| step.rule.as_ref() == rule)
    };
    let effect = result.estimate.as_effect().ok_or_else(not_tiered)?;
    if derived(CLOSURE_RULE) && effect.scenario_effects.is_none() && effect.ate.is_finite() {
        observe(ctx)?;
        let design = TierDesign::CoDeterminedClosure {
            adjustment: result.estimand.adjustment_set.to_vec(),
            rows: effect.n_obs,
            propensity_scored: effect.overlap_report.is_some(),
        };
        let overlap = effect.overlap_report.clone().map_or_else(
            || {
                Availability::Unavailable(unavailable(
                    "tier_diagnostics.overlap_not_carried",
                    "the executed estimator fitted no propensity score, so the result carries no \
                     overlap report; none is refit here because a refit would be a design the \
                     estimate did not use",
                ))
            },
            Availability::Available,
        );
        let evalue = TierEvalue { point: point_evalue(result), interval: interval_withheld() };
        return Ok(TierDiagnostics { design, overlap, evalue });
    }
    if derived(UNKNOWN_RULE) {
        observe(ctx)?;
        let effects = effect.scenario_effects.as_deref().ok_or_else(not_tiered)?;
        let estimands = &result.identification.estimands;
        if effects.len() != estimands.len() || effects.is_empty() {
            return Err(not_tiered());
        }
        let mut scenarios = Vec::with_capacity(effects.len());
        for (estimand, &value) in estimands.iter().zip(effects) {
            observe(ctx)?;
            scenarios.push(TierScenario {
                method: Arc::clone(&estimand.method),
                adjustment: estimand.adjustment_set.to_vec(),
                effect: value,
            });
        }
        return Ok(TierDiagnostics {
            design: TierDesign::UnknownScenarios { scenarios },
            overlap: Availability::Unavailable(unavailable(
                "tier_diagnostics.unknown_scenarios_no_overlap",
                "the Unknown-tier cell is a linear adjustment under two declared orientation \
                 scenarios: it fits no propensity score, so no overlap is carried; none is refit \
                 here and no within-tier edge is assumed to build one",
            )),
            evalue: TierEvalue {
                point: Availability::Unavailable(unavailable(
                    "tier_diagnostics.unknown_scenarios_no_evalue",
                    "the Unknown-tier headline effect is NaN by design and the two scenario \
                     effects rest on different orientation assumptions over the same data; one \
                     E-value would be a statement across structural scenarios, so none is formed",
                )),
                interval: interval_withheld(),
            },
        });
    }
    Err(not_tiered())
}

/// The `CoDetermined` point E-value, only when it is the tier cell's own point value.
fn point_evalue(result: &StudyResult) -> Availability<TierPointEvalue> {
    let Some(effect) = result.estimate.as_effect() else {
        return Availability::Unavailable(unavailable(
            "tier_diagnostics.evalue_not_computed",
            "the result carries no effect estimate to put an E-value on",
        ));
    };
    let attached = result.diagnostics.iter().find(|d| d.code.as_ref() == POINT_EVALUE_DIAGNOSTIC);
    match (effect.evalue, effect.evalue_threshold, attached) {
        (Some(value), None, Some(diagnostic)) if value.is_finite() && value >= 1.0 => {
            let method =
                diagnostic.fields.iter().find(|(key, _)| key.as_ref() == "method").map_or_else(
                    || Arc::from("vanderweele_outcome_sd"),
                    |(_, method)| Arc::clone(method),
                );
            Availability::Available(TierPointEvalue { value, method })
        }
        (Some(_), Some(_), _) => Availability::Unavailable(unavailable(
            "tier_diagnostics.evalue_embeds_interval_limit",
            "the E-value on this result came from the E-value refuter, which reports the smaller \
             of a point and a converted effect-interval-limit E-value as one number; it embeds a \
             converted endpoint and is not reported as a point E-value (run without the E-value \
             refuter to have the tier cell attach its point value)",
        )),
        _ => Availability::Unavailable(unavailable(
            "tier_diagnostics.evalue_not_computed",
            "the tier cell attached no point E-value to this result (a non-finite effect, an \
             outcome with no usable spread, or a quantile functional, which carries no E-value)",
        )),
    }
}

/// The always-present reason no interval-limit E-value is reported.
fn interval_withheld() -> Unavailable {
    Unavailable {
        code: reason_code!("cell_not_licensed"),
        detail: "tier_diagnostics.evalue_interval_withheld",
        reason: "an E-value for the effect-interval limit needs that endpoint converted to a risk \
                 ratio; the conversion (outcome-SD plug-in, RR ~ exp(0.91 d)) is an approximation \
                 whose own uncertainty is excluded and no coverage record measures the converted \
                 endpoint, so only the point E-value is reported",
    }
}
