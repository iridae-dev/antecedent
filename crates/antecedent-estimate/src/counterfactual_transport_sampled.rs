//! Sampled inference for general counterfactual transport.
//!
//! The exact route ([`crate::counterfactual_transport`]) evaluates a stored
//! derivation on supplied finite laws and licenses no sampling uncertainty. This
//! route evaluates the same stored derivation, without re-identifying, on finite
//! laws estimated from sampled counts, and reports four distinct claims:
//!
//! 1. *Structural identification*: the stored derivation identified the query
//!    from the declared graph, selection diagram and regime family. Nothing about
//!    the sample enters this claim.
//! 2. *Exact finite-law evaluation*: the arithmetic of the derivation on the
//!    estimated tables is exact; it is a statement about those tables, not about
//!    the population.
//! 3. *Point estimate*: the plug-in value is an estimate of the population
//!    probability. Smoothed tables are labelled smoothed; an envelope request
//!    reports a structural range and no point.
//! 4. *Statistical inference*: withheld. No coordinate of this route has
//!    measured whole-method coverage, so no interval is published; the delta
//!    method standard error is reported only as `uncalibrated_standard_error`,
//!    and the interval request carries `cell_not_licensed`.
//!
//! # Evidence
//!
//! A count table is one population's dense joint counts of one regime
//! ([`SampledRegimeCounts`]) with a declared sampling design (independent units,
//! or strata with fixed totals). An unseen cell is refused, smoothed under a
//! declared pseudo-count, or reported through a structural envelope
//! ([`UnseenCellRule`]); it is never zero-filled silently, and structural zeros
//! are declared. The exact route's refusals (zero conditioning mass, missing
//! factor laws, support failure) are preserved on the estimated tables.
//!
//! # Compatibility of estimated tables
//!
//! A district factor read from a regime law is, in the model, a function of its
//! declared parents only. Estimated from a sample it differs across coordinates
//! it does not read by sampling noise, so the exact route's agreement check
//! cannot hold. The estimated route evaluates such a factor as its mass-weighted
//! average over those coordinates under the law's own mass given the parents;
//! the two routes coincide when the tables satisfy the graph, and the number of
//! averaged factors and any tolerance repair are reported.
//!
//! # Uncertainty method
//!
//! One method is chosen: the delta method for the multinomial cell proportions of
//! each dataset, with the gradient of the whole functional by a central
//! difference at a stated relative step ([`SAMPLED_TRANSPORT_GRADIENT_RELATIVE_STEP`]).
//! The nonparametric bootstrap of the same functional needs one replicate stream
//! per (population, regime, replicate) and re-evaluates the derivation per
//! replicate; the delta method is exact in the limit it targets, deterministic
//! (no stream), cheap enough to be measured on a grid of designs, and its
//! failure modes (an undefined perturbed evaluation) are explicit. Its weakness
//! is the boundary: at tables with unseen cells or near zero conditioning mass
//! the linearization is poor, which is exactly what whole-method calibration
//! must measure before any interval is published.
//!
//! Dependence between tables is declared ([`OverlapDeclaration`]): independent
//! tables are independent samples; shared units are modelled only for tables
//! that are the same dataset under two labels (one multinomial); an undeclared
//! or unknown relation is refused (`sampling_dependence_unknown`), and tables
//! declared independent that hold identical observations are refused as double
//! counting (`evidence_double_counted`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod counts;
mod envelope;
mod inference;
mod interval;
mod overlap;

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, ExecutionContext, RegimeId};
use antecedent_expr::{ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::CounterfactualTransportDerivation;

use crate::counterfactual_transport::{
    CounterfactualTransportPoint, EvaluationRepairs, evaluate_estimated_tables,
};
use crate::error::EstimationError;

pub use counts::{
    SAMPLED_COUNT_IDENTITY_TAG, SAMPLED_TRANSPORT_MAX_AXES, SampledAxis, SampledRegimeCounts,
    SampledRegimeCountsParts, SampledTableKey, SampledTransportLimits, SamplingDesign,
    UnseenCellRule,
};
pub use envelope::{STRUCTURAL_ENVELOPE_LABEL, StructuralEnvelope};
pub use interval::{
    SAMPLED_TRANSPORT_ESTIMATOR, SAMPLED_TRANSPORT_INTERVAL_LEVEL,
    SAMPLED_TRANSPORT_INTERVAL_METHOD, SAMPLED_TRANSPORT_MIN_CONDITIONING_MASS,
    SAMPLED_TRANSPORT_SE_KIND, SampledTransportInterval, sampled_transport_interval,
};
pub use overlap::{DatasetOverlap, OverlapDeclaration, OverlapPair};

use counts::{LawFlavor, law_of, table_probabilities};
use inference::{DeltaOutcome, delta_standard_error};
use overlap::{DatasetGroups, resolve_overlap};

/// Most count tables of one request.
pub const SAMPLED_TRANSPORT_MAX_TABLES: usize = 64;
/// Most cells of one count table.
pub const SAMPLED_TRANSPORT_MAX_TABLE_CELLS: usize = 1 << 16;
/// Largest envelope family a caller may request.
pub const SAMPLED_TRANSPORT_MAX_ENVELOPE_EVALUATIONS: u32 = 65_536;
/// Relative step of the central-difference gradient of the delta method: each
/// positive cell probability `p` is moved by `step * p`.
pub const SAMPLED_TRANSPORT_GRADIENT_RELATIVE_STEP: f64 = 1e-5;
/// Name of the uncertainty method.
pub const SAMPLED_TRANSPORT_UNCERTAINTY_METHOD: &str =
    "delta_method_multinomial_central_difference";
/// Label of a reported standard error.
pub const UNCALIBRATED_STANDARD_ERROR: &str = "uncalibrated_standard_error";
/// The contract of the sampled route.
pub const SAMPLED_TRANSPORT_CONTRACT: &str = "sampled_counterfactual_transport_v1";
/// Domain tag of the evidence identity.
pub const SAMPLED_TRANSPORT_IDENTITY_TAG: &str =
    "antecedent.estimate.counterfactual_transport_sampled.evidence.v1";
/// Domain tag of the derivation identity.
pub const SAMPLED_TRANSPORT_DERIVATION_TAG: &str =
    "antecedent.estimate.counterfactual_transport_sampled.derivation.v1";

pub(crate) const DETAIL_INVALID_COUNTS: &str = "counterfactual_transport_sampled.invalid_counts";
pub(crate) const DETAIL_SAMPLE_SIZE: &str =
    "counterfactual_transport_sampled.sample_size_below_limit";
pub(crate) const DETAIL_UNSEEN: &str = "counterfactual_transport_sampled.unseen_cell";
pub(crate) const DETAIL_SPARSE: &str = "counterfactual_transport_sampled.sparsity_above_limit";
pub(crate) const DETAIL_OVERLAP_UNKNOWN: &str = "counterfactual_transport_sampled.overlap_unknown";
pub(crate) const DETAIL_OVERLAP_DOUBLE: &str =
    "counterfactual_transport_sampled.overlap_declared_independent";
pub(crate) const DETAIL_OVERLAP_UNLINKED: &str =
    "counterfactual_transport_sampled.shared_units_unlinked";
pub(crate) const DETAIL_OVERLAP_INVALID: &str = "counterfactual_transport_sampled.invalid_overlap";
pub(crate) const DETAIL_NOT_CITED: &str = "counterfactual_transport_sampled.table_not_cited";
pub(crate) const DETAIL_MISSING_TABLE: &str = "counterfactual_transport_sampled.missing_table";
pub(crate) const DETAIL_ENVELOPE_BUDGET: &str = "counterfactual_transport_sampled.envelope_budget";
pub(crate) const DETAIL_CANCELLED: &str = "counterfactual_transport_sampled.cancelled";
pub(crate) const DETAIL_ENVELOPE_EMPTY: &str =
    "counterfactual_transport_sampled.envelope_undefined";
/// Detail of the withheld interval.
pub const DETAIL_INTERVAL_NOT_LICENSED: &str =
    "counterfactual_transport_sampled.interval_not_licensed";

pub(crate) fn refused(
    code: &'static str,
    detail: &str,
    message: impl AsRef<str>,
) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {}", message.as_ref()))
}

pub(crate) fn invalid_counts(message: impl AsRef<str>) -> EstimationError {
    refused(antecedent_core::reason_code!("invalid_argument"), DETAIL_INVALID_COUNTS, message)
}

/// The sampled evidence, its dependence declaration, the unseen-cell rule and the limits.
#[derive(Clone, Debug)]
pub struct SampledTransportRequest {
    /// Count tables, one per cited population, regime and intervention world.
    pub tables: Vec<SampledRegimeCounts>,
    /// Declared overlap of the units behind the tables.
    pub overlap: OverlapDeclaration,
    /// Handling of unseen cells.
    pub unseen_rule: UnseenCellRule,
    /// Sample-size, sparsity and evaluation limits.
    pub limits: SampledTransportLimits,
}

/// The four claim labels of a sampled result, kept distinct.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampledClaimLabels {
    /// What identification established.
    pub structural_identification: &'static str,
    /// What the evaluation of the derivation is exact about.
    pub exact_finite_law_evaluation: &'static str,
    /// What kind of number the value is.
    pub point_estimate: &'static str,
    /// What statistical inference is licensed.
    pub statistical_inference: &'static str,
}

/// The stored derivation identified the query structurally.
pub const LABEL_IDENTIFIED: &str = "identified_by_stored_derivation";
/// The arithmetic is exact on the estimated tables only.
pub const LABEL_EXACT_ON_ESTIMATED_TABLES: &str = "exact_on_estimated_tables_not_population_laws";
/// A plug-in estimate on unsmoothed count shares.
pub const LABEL_PLUG_IN: &str = "plug_in_point_estimate";
/// A plug-in estimate on smoothed tables.
pub const LABEL_SMOOTHED_PLUG_IN: &str = "smoothed_plug_in_point_estimate";
/// No point estimate: a structural envelope only.
pub const LABEL_ENVELOPE_ONLY: &str = "structural_envelope_no_point_estimate";
/// Inference withheld: no coordinate has measured coverage.
pub const LABEL_INFERENCE_WITHHELD: &str = "withheld_cell_not_licensed";

/// A plug-in value and its parts.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledPointEstimate {
    /// The requested probability on the estimated tables.
    pub probability: f64,
    /// Numerator of the (possibly reduced) query.
    pub numerator: f64,
    /// Denominator of the (possibly reduced) query.
    pub denominator: f64,
    /// Original conditioning mass when derived or checked.
    pub conditioning_mass: Option<f64>,
    /// Whether the original conditioning event's mass was checked.
    pub conditioning_mass_checked: bool,
    /// Whether the tables were smoothed.
    pub smoothed: bool,
}

/// The delta-method standard error, labelled uncalibrated.
#[derive(Clone, Debug, PartialEq)]
pub struct UncalibratedStandardError {
    /// The standard error.
    pub value: f64,
    /// Always [`UNCALIBRATED_STANDARD_ERROR`].
    pub label: &'static str,
    /// Always [`SAMPLED_TRANSPORT_UNCERTAINTY_METHOD`].
    pub method: &'static str,
    /// Relative gradient step.
    pub gradient_relative_step: f64,
    /// Functional evaluations spent on the gradient.
    pub gradient_evaluations: u64,
    /// Independent datasets (tables of one shared dataset count once).
    pub datasets: usize,
}

/// An interval route that is closed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WithheldUncertainty {
    /// Always `cell_not_licensed`.
    pub code: &'static str,
    /// Always [`DETAIL_INTERVAL_NOT_LICENSED`].
    pub detail: &'static str,
    /// Calibration reason: `estimator_grid_not_measured`.
    pub reason: &'static str,
}

/// Statistical inference of a sampled result: no interval is ever published.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledInference {
    /// Method that produced the standard error.
    pub method: &'static str,
    /// The standard error, when computed.
    pub standard_error: Option<UncalibratedStandardError>,
    /// Why no standard error is reported, when it is not.
    pub standard_error_withheld_reason: Option<&'static str>,
    /// The closed interval route.
    pub interval: WithheldUncertainty,
}

/// Per-table accounting of how its law was produced.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledTableReport {
    /// The table.
    pub key: SampledTableKey,
    /// Content identity of the counts.
    pub content_identity: String,
    /// Observations.
    pub total: u64,
    /// Cells.
    pub cells: usize,
    /// Cells with observations.
    pub observed_cells: usize,
    /// Unseen non-structural cells.
    pub unseen_cells: usize,
    /// Declared structural zeros.
    pub structural_zeros: usize,
    /// Whether the table's law was smoothed.
    pub smoothed: bool,
    /// Pseudo-count applied, when smoothed.
    pub pseudo_count: Option<f64>,
    /// Dataset index (tables of one shared dataset share it).
    pub dataset: usize,
}

/// Tolerance repairs applied while evaluating on estimated tables.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SampledRepairs {
    /// Factors read as a mass-weighted average over unread coordinates.
    pub averaged_factors: u64,
    /// Negative consistency complements set to zero.
    pub complements_clamped: u64,
    /// Whether the final probability was clamped into `[0, 1]`.
    pub probability_clamped: bool,
}

/// The result of a sampled counterfactual transport estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledCounterfactualTransportEstimate {
    /// The four claim labels.
    pub labels: SampledClaimLabels,
    /// Digest of the stored derivation's canonical text.
    pub derivation_identity: String,
    /// Digest of the counts, overlap declaration, rule and limits.
    pub evidence_identity: String,
    /// The plug-in value; absent only for an envelope request whose observed
    /// shares leave the formula undefined.
    pub point: Option<SampledPointEstimate>,
    /// Why no point is reported, when it is not.
    pub point_withheld_reason: Option<&'static str>,
    /// The structural envelope, for an unseen-mass request with unseen cells.
    pub envelope: Option<StructuralEnvelope>,
    /// How each table's law was produced.
    pub tables: Vec<SampledTableReport>,
    /// Dataset groups of the overlap resolution, as table keys.
    pub datasets: Vec<Vec<SampledTableKey>>,
    /// The declared unseen-cell rule.
    pub unseen_rule: UnseenCellRule,
    /// Tolerance repairs.
    pub repairs: SampledRepairs,
    /// Regime-law cells read by the point evaluation.
    pub cells_read: u64,
    /// Statistical inference: standard error uncalibrated, interval withheld.
    pub inference: SampledInference,
}

/// Evaluate the stored derivation on laws built from counts and return the
/// value with the point-evaluation parts.
pub(crate) fn evaluate_detailed(
    derivation: &CounterfactualTransportDerivation,
    tables: &[SampledRegimeCounts],
    probabilities: &[Vec<f64>],
    flavors: &[LawFlavor],
    limits: &SampledTransportLimits,
    ctx: &ExecutionContext,
) -> Result<(CounterfactualTransportPoint, EvaluationRepairs), EstimationError> {
    let mut laws = Vec::with_capacity(tables.len());
    for ((table, p), flavor) in tables.iter().zip(probabilities).zip(flavors) {
        laws.push(law_of(table, p, *flavor)?);
    }
    let data = ExactTransportData::try_new(laws, SAMPLED_TRANSPORT_MAX_TABLE_CELLS)
        .map_err(|e| invalid_counts(format!("the estimated laws are not a provider set: {e}")))?;
    evaluate_estimated_tables(derivation, &data, limits.evaluation, ctx)
}

/// The probability of the stored derivation on the given estimated tables.
pub(crate) fn evaluate_probability(
    derivation: &CounterfactualTransportDerivation,
    tables: &[SampledRegimeCounts],
    probabilities: &[Vec<f64>],
    flavors: &[LawFlavor],
    limits: &SampledTransportLimits,
    ctx: &ExecutionContext,
) -> Result<f64, EstimationError> {
    evaluate_detailed(derivation, tables, probabilities, flavors, limits, ctx)
        .map(|(point, _)| point.probability)
}

fn validate_request(request: &SampledTransportRequest) -> Result<(), EstimationError> {
    request.unseen_rule.validate()?;
    let limits = &request.limits;
    if limits.min_table_total == 0
        || !limits.max_unseen_fraction.is_finite()
        || !(0.0..=1.0).contains(&limits.max_unseen_fraction)
        || limits.max_envelope_evaluations == 0
        || limits.max_envelope_evaluations > SAMPLED_TRANSPORT_MAX_ENVELOPE_EVALUATIONS
    {
        return Err(invalid_counts(
            "sampled limits need a positive minimum total, a sparsity fraction in [0, 1] and an \
             envelope budget between one and the contract's maximum",
        ));
    }
    if request.tables.is_empty() || request.tables.len() > SAMPLED_TRANSPORT_MAX_TABLES {
        return Err(invalid_counts(format!(
            "a request supplies between one and {SAMPLED_TRANSPORT_MAX_TABLES} count tables"
        )));
    }
    Ok(())
}

/// Digest of everything a plug-in estimate depends on besides the derivation.
fn evidence_identity(request: &SampledTransportRequest, sorted: &[SampledRegimeCounts]) -> String {
    let mut w = CanonicalWriter::new(SAMPLED_TRANSPORT_IDENTITY_TAG);
    w.word(sorted.len() as u64);
    for table in sorted {
        table.write_identity(&mut w);
    }
    request.overlap.write_identity(&mut w);
    w.text(request.unseen_rule.name()).opt_real(request.unseen_rule.parameter());
    let limits = &request.limits;
    w.word(limits.min_table_total).real(limits.max_unseen_fraction);
    w.word(u64::from(limits.max_envelope_evaluations));
    w.word(limits.evaluation.operations as u64).word(limits.evaluation.depth as u64);
    w.finish_hex()
}

/// Estimate a stored counterfactual transport derivation from sampled counts.
///
/// The derivation is evaluated as stored; identification is never repeated.
/// Tables are processed in canonical key order, so the result does not depend
/// on the order of `request.tables`.
///
/// # Errors
///
/// `invalid_argument` (`counterfactual_transport_sampled.invalid_counts`,
/// `.table_not_cited`, `.invalid_overlap`) for malformed counts, limits or
/// declarations and for a table the derivation does not cite;
/// `transport_missing_provider` (`.missing_table`) when a cited regime has no
/// table; `transport_support_failure` (`.sample_size_below_limit`,
/// `.unseen_cell`, `.sparsity_above_limit`, `.envelope_undefined`, and the exact
/// route's `counterfactual_transport.support_failure` and
/// `.zero_conditioning_mass`) for a sparse, unseen or zero-mass table;
/// `sampling_dependence_unknown` (`.overlap_unknown`) and
/// `evidence_double_counted` (`.overlap_declared_independent`) for undeclared or
/// contradicted overlap; `route_not_supported` (`.shared_units_unlinked`) for
/// shared units that are not one dataset; `transport_budget_cancel` for a limit
/// or cancellation; `transport_numerical_failure` when the tables contradict the
/// graph beyond tolerance.
#[allow(
    clippy::too_many_lines,
    reason = "one pass whose declared-bound checks, sampling loop and refusal assembly share the same run state"
)]
pub fn estimate_sampled_counterfactual_transport(
    derivation: &CounterfactualTransportDerivation,
    request: &SampledTransportRequest,
    ctx: &ExecutionContext,
) -> Result<SampledCounterfactualTransportEstimate, EstimationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(refused(
            antecedent_core::reason_code!("transport_budget_cancel"),
            DETAIL_CANCELLED,
            "the sampled evaluation was cancelled",
        ));
    }
    validate_request(request)?;
    let mut tables: Vec<SampledRegimeCounts> = request.tables.clone();
    tables.sort_by_key(|a| (a.key(), a.content_identity()));
    if tables.windows(2).any(|w| w[0].key() == w[1].key()) {
        return Err(invalid_counts(
            "two count tables name one population, regime and intervention world",
        ));
    }
    for table in &tables {
        let cited = derivation
            .cited
            .iter()
            .any(|c| c.regime == table.regime() && c.population.as_ref() == table.population());
        if !cited {
            return Err(refused(
                antecedent_core::reason_code!("invalid_argument"),
                DETAIL_NOT_CITED,
                format!(
                    "population {} regime {} is not cited by the stored derivation; evidence the \
                     derivation does not read is refused rather than ignored",
                    table.population(),
                    table.regime().raw()
                ),
            ));
        }
    }
    for cited in &derivation.cited {
        let supplied = tables
            .iter()
            .any(|t| t.regime() == cited.regime && t.population() == cited.population.as_ref());
        if !supplied {
            return Err(refused(
                antecedent_core::reason_code!("transport_missing_provider"),
                DETAIL_MISSING_TABLE,
                format!(
                    "the derivation cites population {} regime {} and no count table was supplied",
                    cited.population,
                    cited.regime.raw()
                ),
            ));
        }
    }
    let groups: DatasetGroups = resolve_overlap(&tables, &request.overlap)?;
    let mut probabilities: Vec<Vec<f64>> = Vec::with_capacity(tables.len());
    let mut smoothed: Vec<bool> = Vec::with_capacity(tables.len());
    for table in &tables {
        let (p, was_smoothed) = table_probabilities(table, request.unseen_rule, &request.limits)?;
        probabilities.push(p);
        smoothed.push(was_smoothed);
    }
    let flavors: Vec<LawFlavor> =
        smoothed.iter().map(|s| if *s { LawFlavor::Model } else { LawFlavor::Empirical }).collect();
    let any_smoothed = smoothed.iter().any(|s| *s);
    let any_unseen = tables.iter().any(|t| !t.unseen_cells().is_empty());
    let envelope_mass = match request.unseen_rule {
        UnseenCellRule::UnseenMassEnvelope { unseen_mass } if any_unseen => Some(unseen_mass),
        _ => None,
    };

    let mut repairs = SampledRepairs::default();
    let mut point: Option<SampledPointEstimate> = None;
    let mut point_withheld_reason: Option<&'static str> = None;
    let mut cells_read = 0u64;
    let mut envelope: Option<StructuralEnvelope> = None;
    let evaluated =
        evaluate_detailed(derivation, &tables, &probabilities, &flavors, &request.limits, ctx);
    match (evaluated, envelope_mass) {
        (Ok((value, applied)), _) => {
            repairs = SampledRepairs {
                averaged_factors: applied.averaged_factors,
                complements_clamped: applied.complements_clamped,
                probability_clamped: applied.probability_clamped,
            };
            cells_read = value.cells_read;
            point = Some(SampledPointEstimate {
                probability: value.probability,
                numerator: value.numerator,
                denominator: value.denominator,
                conditioning_mass: value.conditioning_mass,
                conditioning_mass_checked: value.conditioning_mass_checked,
                smoothed: any_smoothed,
            });
        }
        (Err(EstimationError::Refused { code, .. }), Some(_))
            if code == antecedent_core::reason_code!("transport_support_failure")
                || code == antecedent_core::reason_code!("transport_missing_provider") =>
        {
            point_withheld_reason = Some("observed_shares_leave_the_formula_undefined");
        }
        (Err(error), _) => return Err(error),
    }
    if let Some(mass) = envelope_mass {
        envelope = Some(envelope::evaluate_envelope(
            derivation,
            &tables,
            &probabilities,
            mass,
            &request.limits,
            ctx,
        )?);
    }

    let datasets = groups.groups.len();
    let (standard_error, standard_error_withheld_reason) = if envelope.is_some() {
        (None, Some("partial_identification_has_no_single_standard_error"))
    } else {
        match delta_standard_error(
            derivation,
            &tables,
            &probabilities,
            &groups,
            &request.limits,
            ctx,
        )? {
            DeltaOutcome::Computed { standard_error, evaluations } => (
                Some(UncalibratedStandardError {
                    value: standard_error,
                    label: UNCALIBRATED_STANDARD_ERROR,
                    method: SAMPLED_TRANSPORT_UNCERTAINTY_METHOD,
                    gradient_relative_step: SAMPLED_TRANSPORT_GRADIENT_RELATIVE_STEP,
                    gradient_evaluations: evaluations,
                    datasets,
                }),
                None,
            ),
            DeltaOutcome::Withheld { reason } => (None, Some(reason)),
        }
    };

    let mut dataset_of = vec![0usize; tables.len()];
    for (g, members) in groups.groups.iter().enumerate() {
        for &m in members {
            dataset_of[m] = g;
        }
    }
    let table_reports: Vec<SampledTableReport> = tables
        .iter()
        .enumerate()
        .map(|(i, t)| SampledTableReport {
            key: t.key(),
            content_identity: t.content_identity(),
            total: t.total(),
            cells: t.cell_count(),
            observed_cells: t.observed_cells(),
            unseen_cells: t.unseen_cells().len(),
            structural_zeros: t.structural_zeros().len(),
            smoothed: smoothed[i],
            pseudo_count: if smoothed[i] { request.unseen_rule.parameter() } else { None },
            dataset: dataset_of[i],
        })
        .collect();
    let dataset_keys: Vec<Vec<SampledTableKey>> = groups
        .groups
        .iter()
        .map(|members| members.iter().map(|m| tables[*m].key()).collect())
        .collect();
    let point_label = if envelope.is_some() && point.is_none() {
        LABEL_ENVELOPE_ONLY
    } else if any_smoothed {
        LABEL_SMOOTHED_PLUG_IN
    } else {
        LABEL_PLUG_IN
    };
    Ok(SampledCounterfactualTransportEstimate {
        labels: SampledClaimLabels {
            structural_identification: LABEL_IDENTIFIED,
            exact_finite_law_evaluation: LABEL_EXACT_ON_ESTIMATED_TABLES,
            point_estimate: point_label,
            statistical_inference: LABEL_INFERENCE_WITHHELD,
        },
        derivation_identity: CanonicalWriter::digest_bytes_hex(
            SAMPLED_TRANSPORT_DERIVATION_TAG,
            derivation.canonical_text().as_bytes(),
        ),
        evidence_identity: evidence_identity(request, &tables),
        point,
        point_withheld_reason,
        envelope,
        tables: table_reports,
        datasets: dataset_keys,
        unseen_rule: request.unseen_rule,
        repairs,
        cells_read,
        inference: SampledInference {
            method: SAMPLED_TRANSPORT_UNCERTAINTY_METHOD,
            standard_error,
            standard_error_withheld_reason,
            interval: WithheldUncertainty {
                code: antecedent_core::reason_code!("cell_not_licensed"),
                detail: DETAIL_INTERVAL_NOT_LICENSED,
                reason: antecedent_core::reason_code!("estimator_grid_not_measured"),
            },
        },
    })
}

/// A nominal normal interval around the plug-in estimate, for the calibration
/// harness only.
#[cfg(feature = "calibration-internal")]
#[derive(Clone, Debug, PartialEq)]
pub struct SampledTransportWaldInterval {
    /// Nominal two-sided level.
    pub level: f64,
    /// Lower end, clamped to `[0, 1]`.
    pub lower: f64,
    /// Upper end, clamped to `[0, 1]`.
    pub upper: f64,
    /// `delta_wald_clamped`.
    pub method: &'static str,
    /// Always `estimator_grid_not_measured`: the interval is a candidate, not a claim.
    pub reason: &'static str,
}

/// Wald interval `estimate +/- z * se`, clamped to `[0, 1]`, so a harness can
/// measure whole-method coverage of the delta standard error. Compiled only
/// with the `calibration-internal` feature, which no released build carries;
/// the public result never holds an interval.
///
/// # Errors
///
/// The stable reason the interval is withheld: no point or no standard error, a
/// partially identified result, or a level outside `(0, 1)`.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn sampled_transport_wald_interval_internal(
    estimate: &SampledCounterfactualTransportEstimate,
    level: f64,
) -> Result<SampledTransportWaldInterval, &'static str> {
    if !(level > 0.0 && level < 1.0) {
        return Err("invalid_interval_level");
    }
    let Some(point) = &estimate.point else {
        return Err("no_point_estimate");
    };
    let Some(se) = &estimate.inference.standard_error else {
        return Err(estimate
            .inference
            .standard_error_withheld_reason
            .unwrap_or("no_standard_error"));
    };
    let z = antecedent_stats::special::normal_ppf(0.5 + level / 2.0);
    Ok(SampledTransportWaldInterval {
        level,
        lower: (point.probability - z * se.value).max(0.0),
        upper: (point.probability + z * se.value).min(1.0),
        method: "delta_wald_clamped",
        reason: antecedent_core::reason_code!("estimator_grid_not_measured"),
    })
}

/// Population and regime identity of a cited law, for artifacts and reports.
#[must_use]
pub fn cited_identities(
    derivation: &CounterfactualTransportDerivation,
) -> Vec<(Arc<str>, RegimeId)> {
    derivation.cited.iter().map(|c| (Arc::clone(&c.population), c.regime)).collect()
}

/// Default exact-evaluation limits used by a sampled request.
#[must_use]
pub fn default_evaluation_limits() -> ExactEvaluationLimits {
    ExactEvaluationLimits::default()
}
