//! A measure over the preference set checked against the set: its points, normalized weights
//! and, for a point with unresolved parameters, the slice of the set through it.

// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(clippy::wildcard_imports, reason = "a part of its parent module")]
use super::*;

/// A measure checked against the set, with normalized weights and, for each point with
/// unresolved parameters, the slice of the set through it.
#[derive(Clone, Debug)]
pub(super) struct PreparedMeasure {
    pub(super) points: Vec<Vec<f64>>,
    pub(super) weights: Vec<f64>,
    pub(super) unresolved: Vec<usize>,
    pub(super) slices: Vec<Option<Region>>,
}

#[allow(clippy::too_many_lines, reason = "one check per declared field, in field order")]
pub(super) fn prepare_measure(
    measure: &PreferenceMeasure,
    set: &PreferenceSet,
    region: &Region,
    lp: LpOptions,
) -> Result<PreparedMeasure, PreferenceDecisionError> {
    let dimension = set.dimension();
    let count = measure.points.len();
    if count == 0 {
        return Err(E::MeasureEmpty);
    }
    let origin_blank = match &measure.origin {
        MeasureOrigin::Declared { assumption_id } => is_blank(assumption_id),
        MeasureOrigin::ChoiceEstimate { estimate_identity } => is_blank(estimate_identity),
    };
    if origin_blank {
        return Err(E::MeasureOriginBlank);
    }
    for (index, point) in measure.points.iter().enumerate() {
        if point.len() != dimension {
            return Err(E::MeasurePointDimension {
                index,
                expected: dimension,
                found: point.len(),
            });
        }
        if point.iter().any(|value| !value.is_finite()) {
            return Err(E::MeasurePointNonFinite { index });
        }
    }
    let weights = match &measure.weights {
        None => vec![1.0 / count as f64; count],
        Some(weights) => {
            if weights.len() != count {
                return Err(E::MeasureWeightCount { expected: count, found: weights.len() });
            }
            if let Some(index) = weights.iter().position(|w| !w.is_finite() || *w < 0.0) {
                return Err(E::MeasureWeightInvalid { index });
            }
            let total: f64 = weights.iter().sum();
            if total <= 0.0 || !total.is_finite() {
                return Err(E::MeasureWeightTotalZero);
            }
            weights.iter().map(|w| w / total).collect()
        }
    };
    if let Some(labels) = &measure.labels {
        if labels.len() != count {
            return Err(E::MeasureLabelCount { expected: count, found: labels.len() });
        }
        let mut seen = HashSet::new();
        for (index, label) in labels.iter().enumerate() {
            if is_blank(label) {
                return Err(E::MeasureLabelBlank { index });
            }
            if !seen.insert(label.as_str()) {
                return Err(E::MeasureLabelDuplicate { label: label.clone() });
            }
        }
    }
    let mut unresolved = Vec::with_capacity(measure.unresolved.len());
    for &index in &measure.unresolved {
        if index >= dimension {
            return Err(E::MeasureUnresolvedUnknown { index });
        }
        if unresolved.contains(&index) {
            return Err(E::MeasureUnresolvedRepeated { index });
        }
        unresolved.push(index);
    }
    unresolved.sort_unstable();
    let mut slices = Vec::with_capacity(count);
    for (index, point) in measure.points.iter().enumerate() {
        if unresolved.is_empty() {
            if !set.contains(point, point_tolerance(point, lp)) {
                return Err(E::MeasurePointOutsideSet { index });
            }
            slices.push(None);
            continue;
        }
        match measure_slice(region, point, &unresolved, lp)? {
            Some(slice) => slices.push(Some(slice)),
            None => return Err(E::MeasurePointOutsideSet { index }),
        }
    }
    Ok(PreparedMeasure { points: measure.points.clone(), weights, unresolved, slices })
}

/// How far a measure point may lie outside a range or relation: the linear programs'
/// feasibility tolerance relative to the point's largest coordinate.
pub(super) fn point_tolerance(point: &[f64], lp: LpOptions) -> f64 {
    lp.feasibility_tolerance * (1.0 + point.iter().fold(0.0_f64, |m, v| m.max(v.abs())))
}

/// The slice of `region` through `point`: every coordinate outside `unresolved` fixed at its
/// value (clamped into its declared range), the unresolved coordinates free in theirs. `None`
/// when a fixed coordinate lies outside its declared range by more than the point tolerance or
/// when the slice holds no point of the relations.
pub(super) fn measure_slice(
    region: &Region,
    point: &[f64],
    unresolved: &[usize],
    lp: LpOptions,
) -> Result<Option<Region>, E> {
    let tolerance = point_tolerance(point, lp);
    let mut lower = region.lower.clone();
    let mut upper = region.upper.clone();
    for j in (0..region.dimension()).filter(|j| !unresolved.contains(j)) {
        let value = point[j];
        if value < region.lower[j] - tolerance || value > region.upper[j] + tolerance {
            return Ok(None);
        }
        let fixed = value.clamp(region.lower[j], region.upper[j]);
        lower[j] = fixed;
        upper[j] = fixed;
    }
    let slice = Region {
        rows: region.rows.clone(),
        lower,
        upper,
        cancellation: region.cancellation.clone(),
    };
    Ok(slice.feasible(&[], lp)?.is_some().then_some(slice))
}

/// Whether reading `point` as a measure point with the parameters `unresolved` left open
/// accepts it: with nothing unresolved, the point lies in the declared ranges and linear
/// relations within the point tolerance; otherwise some value of the unresolved parameters in
/// their declared ranges completes it to such a point. This is the check a measure over the set
/// passes when it is read, so a learner that keeps only accepted points produces a measure the
/// evaluation accepts. A point of the wrong length or with a non-finite coordinate is not
/// accepted.
///
/// # Errors
/// A refusal of the slice's linear program.
pub(crate) fn measure_point_accepted(
    set: &PreferenceSet,
    point: &[f64],
    unresolved: &[usize],
    lp: LpOptions,
) -> Result<bool, PreferenceDecisionError> {
    if point.len() != set.dimension() || point.iter().any(|v| !v.is_finite()) {
        return Ok(false);
    }
    if unresolved.is_empty() {
        return Ok(set.contains(point, point_tolerance(point, lp)));
    }
    Ok(measure_slice(&Region::of(set), point, unresolved, lp)?.is_some())
}

/// A point of the declared ranges cut by the linear relations of `set`, from the linear program
/// this route reads every set with, or `None` when the ranges and relations admit no point.
/// Value relations are not read.
///
/// # Errors
/// A refusal of the linear program.
pub(crate) fn linear_set_point(
    set: &PreferenceSet,
    lp: LpOptions,
) -> Result<Option<Vec<f64>>, PreferenceDecisionError> {
    Region::of(set).feasible(&[], lp)
}
