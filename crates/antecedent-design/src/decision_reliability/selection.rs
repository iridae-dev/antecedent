//! Selection-aware evaluation of a decision chosen among many noisy candidates.
//!
//! The maximum of `N` noisy utility estimates is biased upward for the utility of the
//! candidate it picked (the winner's curse). This module provides the three remedies
//! the claim needs, and keeps the unadjusted value labelled `in_sample`:
//!
//! * a **bound** on the expected upward bias: for Gaussian Monte Carlo noise with known
//!   standard errors `s_j`, `E[e_selected] <= E[max_j e_j]`, evaluated exactly by
//!   quadrature when the noise is declared independent, and by the dependence-free
//!   union bound `s_max * sqrt(2 ln N)` otherwise. An observed nonnegative sample
//!   covariance is only a diagnostic; it does not establish nonnegative population
//!   covariance and cannot license the Sudakov-Fernique comparison. The bound is an
//!   expectation over repetitions of the whole selection, not an interval for this
//!   realization;
//! * a **held-out re-evaluation** of the selected candidate on a stream independent of
//!   every selection stream, which is unbiased for the selected candidate's value;
//! * **sample splitting**: select on one segment of the draws, validate on a disjoint
//!   one separated by a gap.
//!
//! Selection among candidates the search *considered but did not retain* is accounted
//! through the search record's count; the unseen candidates are assumed to carry the
//! largest supplied standard error, an assumption that is stated in the output.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_kernels::log_norm_cdf;
use serde::{Deserialize, Serialize};

use super::ReliabilityError;
use super::draws::{
    CandidateCoupling, CandidateDraws, CandidateOrigin, DecisionDraws, WithinStreamDependence,
};
use super::mc_error::{BatchPlan, Functional, FunctionalEstimate, McEstimate, estimate};
use super::stream::{DrawSegment, StreamId};

/// Largest number of distinct standard errors the quadrature accepts.
pub const MAX_QUADRATURE_GROUPS: usize = 4096;
const QUADRATURE_PANELS: usize = 4000;

/// Whether a reported value used the same draws that selected it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueLabel {
    /// Estimated on the draws that selected the candidate: biased upward.
    InSample,
    /// Re-estimated on draws independent of every selection draw.
    HeldOut,
}

/// How far the selected decision's value has been checked for selection bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStatus {
    /// Only the in-sample value; the winner's-curse bound was not computable.
    InSampleOnly,
    /// In-sample value with a bound on the expected upward bias.
    WinnersCurseBounded,
    /// Re-evaluated on an independent simulation stream or held-out data.
    HeldOutValidated,
    /// Selected and validated on disjoint segments of one stream.
    SampleSplitValidated,
}

/// How the validation draws relate to the selection draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationKind {
    /// A separately seeded simulation stream.
    IndependentSimulationStream,
    /// A disjoint segment of the selection stream.
    SplitOfSelectionStream,
    /// Data held out from every selection stage (a producer declaration).
    HeldOutData,
}

/// Which stream selected and which validated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationRecord {
    /// Candidate that was selected.
    pub selected_id: String,
    /// Distinct stream segments the selection used, sorted.
    pub selection_segments: Vec<DrawSegment>,
    /// Segment the selected candidate was re-evaluated on.
    pub validation_segment: DrawSegment,
    /// How the validation draws were obtained.
    pub kind: ValidationKind,
    /// Draws separating the segments when they share a stream.
    pub gap: u64,
}

/// One candidate's estimate under the selection functional.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEstimate {
    /// Candidate identity.
    pub id: String,
    /// Estimate and numerical error.
    pub estimate: McEstimate,
}

/// The in-sample winner of a finite candidate set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InSampleSelection {
    /// Functional maximized.
    pub functional: Functional,
    /// The selected candidate (ties: smallest id).
    pub selected_id: String,
    /// Every candidate's estimate, in canonical (id) order.
    pub estimates: Vec<CandidateEstimate>,
    /// Always [`ValueLabel::InSample`].
    pub label: ValueLabel,
}

/// The candidate maximizing `functional` on the supplied draws.
///
/// The returned value is the unadjusted maximum and is labelled in-sample.
///
/// # Errors
/// Anything [`DecisionDraws::validate`] or [`estimate`] refuses.
pub fn select_in_sample(
    draws: &DecisionDraws,
    functional: &Functional,
    plan: BatchPlan,
) -> Result<InSampleSelection, ReliabilityError> {
    draws.validate()?;
    select_validated(draws, functional, plan, None)
}

pub(crate) fn select_validated(
    draws: &DecisionDraws,
    functional: &Functional,
    plan: BatchPlan,
    context: Option<&antecedent_core::ExecutionContext>,
) -> Result<InSampleSelection, ReliabilityError> {
    let weights = draws.weights.as_deref();
    let mut estimates = Vec::with_capacity(draws.n_candidates());
    for &i in &draws.canonical_order() {
        let candidate = &draws.candidates[i];
        let estimated = match context {
            Some(ctx) => super::mc_error::estimate_in_context(
                &candidate.utilities,
                weights,
                draws.within_stream,
                plan,
                functional,
                ctx,
            )?,
            None => estimate(&candidate.utilities, weights, draws.within_stream, plan, functional)?,
        };
        estimates
            .push(CandidateEstimate { id: candidate.id.clone(), estimate: estimated.estimate });
    }
    let selected = estimates
        .iter()
        .fold(None::<&CandidateEstimate>, |best, c| match best {
            Some(b) if b.estimate.value >= c.estimate.value => Some(b),
            _ => Some(c),
        })
        .map(|c| c.id.clone())
        .ok_or_else(|| ReliabilityError::invalid("candidate_count", "no candidates"))?;
    Ok(InSampleSelection {
        functional: functional.clone(),
        selected_id: selected,
        estimates,
        label: ValueLabel::InSample,
    })
}

/// What is known about the dependence of the Monte Carlo errors across candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseDependence {
    /// Candidates were declared independent on separate streams.
    IndependentStreams,
    /// A shared simulation whose pairwise sample covariances of the utilities were all
    /// nonnegative (a sample-sign check, not a guarantee).
    SampleCovariancesNonNegative,
    /// A shared simulation whose covariance signs were not established.
    Unassessed,
}

/// Most candidates whose pairwise covariance signs are checked.
const MAX_COVARIANCE_CHECK: usize = 128;

fn initial_noise_dependence(draws: &DecisionDraws) -> Option<NoiseDependence> {
    if draws.candidates.len() < 2 || draws.coupling == CandidateCoupling::DeclaredIndependent {
        return Some(NoiseDependence::IndependentStreams);
    }
    if draws.candidates.len() > MAX_COVARIANCE_CHECK
        || draws.coupling != CandidateCoupling::SharedSimulation
    {
        return Some(NoiseDependence::Unassessed);
    }
    None
}

pub(crate) fn noise_dependence_in_context(
    draws: &DecisionDraws,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<NoiseDependence, ReliabilityError> {
    if let Some(known) = initial_noise_dependence(draws) {
        return Ok(known);
    }
    let _values = super::workspace::reserve::<f64>(
        ctx,
        super::workspace::product(draws.n_candidates(), draws.n_draws())?,
    )?;
    let _columns = super::workspace::reserve::<Vec<f64>>(ctx, draws.n_candidates())?;
    Ok(noise_dependence(draws))
}

/// Noise dependence of the candidates' Monte Carlo errors.
#[must_use]
pub fn noise_dependence(draws: &DecisionDraws) -> NoiseDependence {
    if let Some(known) = initial_noise_dependence(draws) {
        return known;
    }
    let n = draws.n_draws() as f64;
    let centred: Vec<Vec<f64>> = draws
        .candidates
        .iter()
        .map(|c| {
            let mean = c.utilities.iter().sum::<f64>() / n;
            c.utilities.iter().map(|u| u - mean).collect()
        })
        .collect();
    for (a, xa) in centred.iter().enumerate() {
        for xb in &centred[a + 1..] {
            let cov: f64 = xa.iter().zip(xb).map(|(x, y)| x * y).sum();
            if cov < 0.0 {
                return NoiseDependence::Unassessed;
            }
        }
    }
    NoiseDependence::SampleCovariancesNonNegative
}

/// Which bound `bound` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundKind {
    /// The exact expected maximum of independent Gaussian noise with the supplied
    /// standard errors, valid for declared independent noise.
    GaussianExpectedMaximum,
    /// `s_max * sqrt(2 ln N)`, valid for any dependence of Gaussian noise.
    UnionBound,
}

/// A bound on the expected upward bias of the maximum of `N` noisy estimates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WinnersCurseBound {
    /// Candidates the selection considered.
    pub n_considered: u64,
    /// Candidates whose standard errors were supplied.
    pub n_supplied: u64,
    /// Considered candidates without a supplied standard error, assumed to carry the
    /// largest supplied one.
    pub padded_candidates: u64,
    /// Dependence the bound relied on.
    pub noise_dependence: NoiseDependence,
    /// Exact expected maximum of independent Gaussian noise, when computed.
    pub gaussian_expected_maximum: Option<f64>,
    /// The dependence-free union bound.
    pub union_bound: f64,
    /// The bound reported.
    pub bound: f64,
    /// Which bound `bound` is.
    pub bound_kind: BoundKind,
    /// The unadjusted maximum (in-sample).
    pub in_sample_value: f64,
    /// `in_sample_value - bound`: a lower bound on the *expected* value of the selected
    /// candidate over repetitions of the selection, under the Gaussian-noise
    /// assumption. Not an interval for the realized selection.
    pub adjusted_value: f64,
}

/// Exact `E[max_j Z_j]` for independent `Z_j ~ N(0, s_j^2)`, as
/// `int_0^inf (1 - F(x)) dx - int_0^inf F(-x) dx` with `F(x) = prod_j Phi(x / s_j)`.
///
/// Composite Simpson on `[0, 9 s_max]`, graded to the distinct scales; the neglected tails are
/// below `N * 1e-18`.
/// `groups` are `(standard error, multiplicity)` pairs.
fn expected_maximum_independent(groups: &[(f64, u64)]) -> f64 {
    let s_max = groups.iter().map(|g| g.0).fold(0.0_f64, f64::max);
    if s_max <= 0.0 {
        return 0.0;
    }
    let log_cdf = |x: f64| -> f64 {
        groups
            .iter()
            .map(|&(s, count)| {
                let term = if s > 0.0 {
                    log_norm_cdf(x / s)
                } else if x >= 0.0 {
                    0.0
                } else {
                    f64::NEG_INFINITY
                };
                count as f64 * term
            })
            .sum()
    };
    // `F(-x)` for the lower half-line is the limit from `x > 0`: a candidate with no noise is
    // at zero, so `P(Z <= -x)` is 0 for every `x > 0`. Evaluated at `x = 0` itself the point
    // mass would count as 1 and put a jump of that size on the Simpson end point, an error of
    // `h / 6` in the expected maximum.
    let has_point_mass = groups.iter().any(|g| g.0 <= 0.0);
    let lower = |x: f64| -> f64 { if has_point_mass { 0.0 } else { log_cdf(-x).exp() } };
    let integrand = |x: f64| -> f64 { (1.0 - log_cdf(x).exp()) - lower(x) };
    // Composite Simpson, with the mesh graded to the distinct scales when they are far apart.
    // A single mesh sized by `s_max` has `4000 / 9` panels per `s_max`, which resolves any
    // scale within a factor of four of it; a candidate whose standard error is orders of
    // magnitude smaller is under-resolved (a relative error of order `1 / panels` in the
    // expected maximum when the scales differ by `1e3`). For those, a breakpoint sits at nine
    // times each scale below `s_max / 4` that is at least twice the previous breakpoint's
    // scale, and the panels are shared among the segments, never fewer than 400 each. With no
    // such scale the mesh is the single segment `[0, 9 s_max]`.
    let mut breaks = vec![0.0_f64];
    let mut scales: Vec<f64> =
        groups.iter().map(|g| g.0).filter(|s| *s > 0.0 && *s < 0.25 * s_max).collect();
    scales.sort_unstable_by(f64::total_cmp);
    for s in scales {
        let edge = 9.0 * s;
        let last = breaks[breaks.len() - 1];
        if edge >= 2.0 * last {
            breaks.push(edge);
        }
    }
    let top = 9.0 * s_max;
    if breaks[breaks.len() - 1] < top {
        breaks.push(top);
    }
    let segments = breaks.len() - 1;
    let panels = ((QUADRATURE_PANELS / segments).max(400) / 2) * 2;
    let mut total = 0.0;
    for window in breaks.windows(2) {
        let (lo, hi) = (window[0], window[1]);
        let h = (hi - lo) / panels as f64;
        let mut sum = integrand(lo) + integrand(hi);
        for k in 1..panels {
            let weight = if k % 2 == 1 { 4.0 } else { 2.0 };
            sum += weight * integrand(lo + k as f64 * h);
        }
        total += sum * h / 3.0;
    }
    total
}

/// Bound the expected upward bias of the maximum of noisy estimates.
///
/// `estimates` are the supplied candidates' `(value, standard error)` under the
/// selection functional; `n_considered` counts every candidate the selection looked at
/// (at least the supplied count).
///
/// # Errors
/// No supplied candidate, a negative or non-finite standard error, or fewer considered
/// than supplied candidates.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "split sizes are derived from bounded draw counts"
)]
pub fn winners_curse_bound(
    estimates: &[McEstimate],
    n_considered: u64,
    noise_dependence: NoiseDependence,
    in_sample_value: f64,
) -> Result<WinnersCurseBound, ReliabilityError> {
    let n_supplied = estimates.len() as u64;
    if estimates.is_empty()
        || n_considered < n_supplied
        || estimates.iter().any(|e| !e.standard_error.is_finite() || e.standard_error < 0.0)
        || !in_sample_value.is_finite()
    {
        return Err(ReliabilityError::invalid(
            "winners_curse_input",
            "need supplied estimates with finite non-negative standard errors and \
             n_considered >= supplied",
        ));
    }
    let s_max = estimates.iter().map(|e| e.standard_error).fold(0.0_f64, f64::max);
    let padded = n_considered - n_supplied;
    let union_bound = s_max * (2.0 * (n_considered as f64).ln()).sqrt();
    let mut ses: Vec<f64> = estimates.iter().map(|e| e.standard_error).collect();
    ses.sort_unstable_by(f64::total_cmp);
    let mut groups: Vec<(f64, u64)> = Vec::with_capacity(estimates.len());
    for s in ses {
        let repeated = groups.last().is_some_and(|g| g.0.to_bits() == s.to_bits());
        if repeated {
            if let Some(last) = groups.last_mut() {
                last.1 += 1;
            }
        } else {
            groups.push((s, 1));
        }
    }
    if padded > 0 {
        if let Some(position) = groups.iter().position(|g| g.0.to_bits() == s_max.to_bits()) {
            groups[position].1 += padded;
        } else {
            groups.push((s_max, padded));
        }
    }
    let gaussian =
        (groups.len() <= MAX_QUADRATURE_GROUPS).then(|| expected_maximum_independent(&groups));
    // The sample covariance route is descriptive only. Its sign does not establish
    // the population covariance ordering required by Sudakov-Fernique, so only an
    // independent-stream declaration licenses the exact independent Gaussian maximum.
    let gaussian_valid = noise_dependence == NoiseDependence::IndependentStreams;
    let (bound, bound_kind) = match gaussian {
        Some(g) if gaussian_valid && g <= union_bound => (g, BoundKind::GaussianExpectedMaximum),
        _ => (union_bound, BoundKind::UnionBound),
    };
    Ok(WinnersCurseBound {
        n_considered,
        n_supplied,
        padded_candidates: padded,
        noise_dependence,
        gaussian_expected_maximum: gaussian,
        union_bound,
        bound,
        bound_kind,
        in_sample_value,
        adjusted_value: in_sample_value - bound,
    })
}

/// Re-evaluation of the selected candidate on independent draws.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeldOutAssessment {
    /// Which streams selected and which validated.
    pub record: ValidationRecord,
    /// The selected candidate's estimate on the validation draws.
    pub held_out: FunctionalEstimate,
    /// The selected candidate's estimate on the selection draws (in-sample).
    pub in_sample: McEstimate,
    /// Always [`ValueLabel::HeldOut`].
    pub label: ValueLabel,
    /// `in_sample - held_out`: the realized optimism of the selection.
    pub optimism: f64,
    /// `sqrt(se_in^2 + se_held^2)`: the segments are independent, so errors add.
    pub optimism_standard_error: f64,
}

fn segments_by_stream<'a>(
    candidates: impl Iterator<Item = &'a CandidateDraws>,
) -> BTreeMap<&'a StreamId, Vec<(u64, u64)>> {
    let mut map: BTreeMap<&StreamId, Vec<(u64, u64)>> = BTreeMap::new();
    for candidate in candidates {
        map.entry(&candidate.segment.stream)
            .or_default()
            .push((candidate.segment.start, candidate.segment.end()));
    }
    map
}

/// Re-evaluate `selected_id` on `validation` and check independence from every
/// selection segment.
///
/// `gap` is the separation, in draws, demanded between ranges of one shared stream.
///
/// # Errors
/// Overlapping or insufficiently separated validation draws, a candidate missing from
/// either carrier, differing utility units, or any refusal of [`estimate`].
pub fn held_out_evaluation(
    selection: &DecisionDraws,
    selected_id: &str,
    validation: &DecisionDraws,
    functional: &Functional,
    plan: BatchPlan,
    kind: ValidationKind,
    gap: u64,
) -> Result<HeldOutAssessment, ReliabilityError> {
    validation.validate()?;
    if validation.repeated_evaluations > 0 {
        return Err(ReliabilityError::ValidationNotIndependent {
            selection: "draws that selected the decision".into(),
            validation: format!(
                "validation draws already evaluated {} time(s) before this re-evaluation",
                validation.repeated_evaluations
            ),
        });
    }
    if selection.utility_unit != validation.utility_unit {
        return Err(ReliabilityError::invalid(
            "validation_unit",
            "validation utilities are in a different unit than the selection utilities",
        ));
    }
    let chosen_selection = selection
        .candidate(selected_id)
        .ok_or_else(|| ReliabilityError::invalid("selected_id", "not in the selection draws"))?;
    let chosen_validation = validation
        .candidate(selected_id)
        .ok_or_else(|| ReliabilityError::invalid("selected_id", "not in the validation draws"))?;
    let by_stream = segments_by_stream(selection.candidates.iter());
    let validation_segment = &chosen_validation.segment;
    if let Some(ranges) = by_stream.get(&validation_segment.stream) {
        let (v0, v1) = (validation_segment.start, validation_segment.end());
        let too_close = ranges.iter().any(|&(s0, s1)| {
            let separation = if s1 <= v0 { v0 - s1 } else { s0.saturating_sub(v1) };
            let overlapping = !(s1 <= v0 || v1 <= s0);
            overlapping || separation < gap
        });
        if too_close {
            return Err(ReliabilityError::ValidationNotIndependent {
                selection: selection
                    .candidates
                    .iter()
                    .find(|c| c.segment.stream == validation_segment.stream)
                    .map_or_else(String::new, |c| c.segment.describe()),
                validation: validation_segment.describe(),
            });
        }
    }
    let held_out = estimate(
        &chosen_validation.utilities,
        validation.weights.as_deref(),
        validation.within_stream,
        plan,
        functional,
    )?;
    let in_sample = estimate(
        &chosen_selection.utilities,
        selection.weights.as_deref(),
        selection.within_stream,
        plan,
        functional,
    )?
    .estimate;
    let mut selection_segments: Vec<DrawSegment> = Vec::new();
    for candidate in &selection.candidates {
        if !selection_segments.contains(&candidate.segment) {
            selection_segments.push(candidate.segment.clone());
        }
    }
    selection_segments
        .sort_by(|a, b| (&a.stream, a.start, a.len).cmp(&(&b.stream, b.start, b.len)));
    let optimism = in_sample.value - held_out.estimate.value;
    let optimism_standard_error =
        (in_sample.standard_error.powi(2) + held_out.estimate.standard_error.powi(2)).sqrt();
    Ok(HeldOutAssessment {
        record: ValidationRecord {
            selected_id: selected_id.to_owned(),
            selection_segments,
            validation_segment: validation_segment.clone(),
            kind,
            gap,
        },
        held_out,
        in_sample,
        label: ValueLabel::HeldOut,
        optimism,
        optimism_standard_error,
    })
}

/// Split one carrier into a selection part and a validation part: contiguous blocks
/// `[0, n_sel)` and `[n_sel + gap, n)`, the same ranges for every candidate.
///
/// `selection_fraction` is the share of draws that select. An autocorrelated stream
/// needs a positive `gap`; the gap should exceed the stream's autocorrelation time,
/// which this function cannot check.
///
/// The validation part is a pure re-evaluation carrier (`Enumerated`, no search
/// record): it never selects.
///
/// # Errors
/// A fraction outside `(0, 1)`, either part with fewer than the minimum draws, or a
/// zero gap on an autocorrelated stream.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "split sizes are derived from bounded draw counts"
)]
fn split_layout(
    draws: &DecisionDraws,
    selection_fraction: f64,
    gap: u64,
) -> Result<(usize, usize), ReliabilityError> {
    draws.validate()?;
    if !(selection_fraction.is_finite() && selection_fraction > 0.0 && selection_fraction < 1.0) {
        return Err(ReliabilityError::invalid("selection_fraction", "must lie in (0, 1)"));
    }
    let n = draws.n_draws();
    let n_sel = (selection_fraction * n as f64).floor() as usize;
    let start_val = n_sel.saturating_add(gap as usize);
    let selection_part = draws.candidates[0].segment.sub(0, n_sel as u64);
    if draws.within_stream == WithinStreamDependence::Autocorrelated && gap == 0 {
        let validation_part =
            draws.candidates[0].segment.sub(start_val as u64, n.saturating_sub(start_val) as u64);
        return Err(ReliabilityError::ValidationNotIndependent {
            selection: selection_part.describe(),
            validation: validation_part.describe(),
        });
    }
    if n_sel < super::MIN_DRAWS || n.saturating_sub(start_val) < super::MIN_DRAWS {
        return Err(ReliabilityError::invalid(
            "split_size",
            "each part of a split needs at least the minimum number of draws",
        ));
    }
    Ok((n_sel, start_val))
}

pub(crate) struct ContextualSplit {
    pub selection: DecisionDraws,
    pub validation: DecisionDraws,
    _rows: antecedent_core::execution::AdmissionLease,
}

pub(crate) fn split_draws_in_context(
    draws: &DecisionDraws,
    fraction: f64,
    gap: u64,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ContextualSplit, ReliabilityError> {
    let (selection_rows, validation_start) = split_layout(draws, fraction, gap)?;
    let rows = selection_rows + draws.n_draws() - validation_start;
    let columns = draws.n_candidates() + usize::from(draws.weights.is_some());
    let lease = super::workspace::reserve::<f64>(ctx, super::workspace::product(rows, columns)?)?;
    let (selection, validation) = split_draws(draws, fraction, gap)?;
    Ok(ContextualSplit { selection, validation, _rows: lease })
}

/// Split the validated carrier into selection and held-out streams.
///
/// # Errors
/// Invalid fraction, insufficient rows, or an unseparated dependent stream.
pub fn split_draws(
    draws: &DecisionDraws,
    selection_fraction: f64,
    gap: u64,
) -> Result<(DecisionDraws, DecisionDraws), ReliabilityError> {
    let (n_sel, start_val) = split_layout(draws, selection_fraction, gap)?;
    let n = draws.n_draws();
    let n_val = n - start_val;
    let part = |from: usize, len: usize, keep_search: bool| DecisionDraws {
        utility_unit: draws.utility_unit.clone(),
        producer: draws.producer.clone(),
        source: draws.source,
        coupling: draws.coupling,
        within_stream: draws.within_stream,
        weights: draws.weights.as_ref().map(|w| w[from..from + len].to_vec()),
        repeated_evaluations: if keep_search { draws.repeated_evaluations } else { 0 },
        origin: if keep_search { draws.origin } else { CandidateOrigin::Enumerated },
        search_record: if keep_search { draws.search_record.clone() } else { None },
        candidates: draws
            .candidates
            .iter()
            .map(|c| CandidateDraws {
                id: c.id.clone(),
                utilities: c.utilities[from..from + len].to_vec(),
                segment: c.segment.sub(from as u64, len as u64),
            })
            .collect(),
    };
    Ok((part(0, n_sel, true), part(start_val, n_val, false)))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "split sizes are derived from bounded draw counts"
)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_maximum_matches_tabulated_gaussian_order_statistics() {
        // Tippett's tabulated E[max] of N standard normals.
        let table = [
            (1_u64, 0.0),
            (2, 0.564_189_583_5),
            (3, 0.846_284_375_3),
            (4, 1.029_375_373_0),
            (5, 1.162_964_473_6),
            (10, 1.538_752_731_0),
        ];
        for (n, expected) in table {
            let got = expected_maximum_independent(&[(1.0, n)]);
            assert!((got - expected).abs() < 1e-8, "N={n}: {got} vs {expected}");
        }
        // Two independent Gaussians: max = (X + Y)/2 + |X - Y|/2, so
        // E max = sqrt(s1^2 + s2^2) / sqrt(2 pi).
        let mixed = expected_maximum_independent(&[(1.0, 1), (2.0, 1)]);
        let exact = 5.0_f64.sqrt() / (2.0 * std::f64::consts::PI).sqrt();
        assert!((mixed - exact).abs() < 1e-8);
    }
}
