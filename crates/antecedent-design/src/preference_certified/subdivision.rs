//! The subdivision driver of the certified route: splitting the parameter box until each kept
//! box is decided or the budget is spent, the fractions and enclosures over the kept boxes, and
//! the verdict each preference policy reads from them, on boxes or on measure points.

// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(clippy::wildcard_imports, reason = "a part of its parent module")]
use super::*;

// ---------------------------------------------------------------------------------------------
// Aggregation over kept boxes
// ---------------------------------------------------------------------------------------------

/// Fractions and enclosures over the kept boxes, scores in the larger-is-better orientation.
pub(super) struct Tally {
    pub(super) best_fraction: Vec<f64>,
    pub(super) undecided_fraction: f64,
    pub(super) value: Vec<Option<(f64, f64)>>,
    pub(super) worst: Vec<Option<(f64, f64)>>,
    pub(super) regret: Vec<Option<(f64, f64)>>,
}

/// Fractions and enclosures over the kept boxes.
///
/// A worst value over the set is at most the value at any point of the set, so the upper end
/// of a worst-value enclosure also uses the enclosure at the center of each box wholly inside
/// the relations; likewise a worst-case regret is at least the margin at such a center.
pub(super) fn tally(
    engine: &Engine<'_>,
    kept: &[Examined],
) -> Result<Tally, PreferenceCertifiedError> {
    let n = engine.contract.actions.len();
    let total: f64 = kept.iter().map(|item| item.volume).sum();
    let fraction = |volume: f64| if total > 0.0 { volume / total } else { 0.0 };
    let mut best_fraction = vec![0.0; n];
    let mut undecided = 0.0;
    for item in kept {
        match item.best {
            Some(a) => best_fraction[a] += item.volume,
            None => undecided += item.volume,
        }
    }
    let best_fraction = best_fraction.into_iter().map(fraction).collect();
    let mut centers = Vec::new();
    for item in kept.iter().filter(|item| !item.boundary) {
        let center: Vec<f64> =
            item.lower.iter().zip(&item.upper).map(|(l, u)| 0.5 * l + 0.5 * u).collect();
        let params = engine.params(&center, &center)?;
        centers.push(engine.evaluate_box(&params, &center, &center)?);
    }
    let admissible = engine.admissible_indices();
    let mut value = vec![None; n];
    let mut worst = vec![None; n];
    let mut regret = vec![None; n];
    for &a in &admissible {
        let scores: Vec<Interval> = kept.iter().filter_map(|item| item.eval.scores[a]).collect();
        if scores.is_empty() {
            continue;
        }
        let lo = scores.iter().map(|s| s.lo).fold(f64::INFINITY, f64::min);
        let hi = scores.iter().map(|s| s.hi).fold(f64::NEG_INFINITY, f64::max);
        let inside_hi = kept
            .iter()
            .filter(|item| !item.boundary)
            .filter_map(|item| item.eval.scores[a])
            .chain(centers.iter().filter_map(|center| center.scores[a]))
            .map(|s| s.hi)
            .fold(f64::INFINITY, f64::min);
        value[a] = Some((lo, hi));
        worst[a] = Some((lo, if inside_hi.is_finite() { inside_hi } else { hi }));
        if is_expectation(engine.criterion()) {
            let rivals = || admissible.iter().copied().filter(move |b| *b != a);
            let upper = kept
                .iter()
                .flat_map(|item| rivals().filter_map(move |b| item.eval.margins[b][a]))
                .fold(0.0_f64, |top, m| top.max(m.hi));
            let lower = kept
                .iter()
                .filter(|item| !item.boundary)
                .map(|item| &item.eval)
                .chain(&centers)
                .flat_map(|eval| rivals().filter_map(move |b| eval.margins[b][a]))
                .fold(0.0_f64, |top, m| top.max(m.lo));
            regret[a] = Some((lower, upper));
        }
    }
    Ok(Tally { best_fraction, undecided_fraction: fraction(undecided), value, worst, regret })
}

/// The action whose enclosure (larger is better) lies above every other's by more than
/// `tie`, or the actions not proven below another.
pub(super) fn dominant(enclosures: &[Option<(f64, f64)>], tie: f64) -> Result<usize, Vec<usize>> {
    let present: Vec<usize> = (0..enclosures.len()).filter(|a| enclosures[*a].is_some()).collect();
    let above = |a: usize, b: usize| match (enclosures[a], enclosures[b]) {
        (Some(x), Some(y)) => x.0 > y.1 + tie,
        _ => false,
    };
    if let Some(a) =
        present.iter().copied().find(|&a| present.iter().all(|&b| b == a || above(a, b)))
    {
        return Ok(a);
    }
    Err(present
        .iter()
        .copied()
        .filter(|&a| !present.iter().any(|&b| b != a && above(b, a)))
        .collect())
}

pub(super) fn ids(
    contract: &PreferenceContract,
    indices: impl IntoIterator<Item = usize>,
) -> Vec<String> {
    indices.into_iter().map(|a| contract.actions[a].id.clone()).collect()
}

/// Whether the set is proven to hold a point. The linear program that checks the set before
/// subdividing works to a feasibility tolerance, so its point is evidence, not proof. A witness
/// is a kept box wholly inside every relation, or a point (the set check's point, then the centre
/// of each kept box) at which every relation is proven satisfied with no tolerance: linear
/// relations in arithmetic whose ends move outward only on an inexact operation, a parameter
/// solved from an equality as the enclosure of its solution, value relations by their outward-
/// rounded gap enclosure.
pub(super) fn set_witnessed(engine: &Engine<'_>, kept: &[Examined]) -> bool {
    if kept.iter().any(|item| !item.boundary) {
        return true;
    }
    engine.point_witnessed(engine.interior)
        || kept.iter().any(|item| engine.point_witnessed(&engine.centre(&item.lower, &item.upper)))
}

pub(super) fn subdivision_verdict(
    engine: &Engine<'_>,
    kept: &[Examined],
    tally: &Tally,
) -> CertifiedVerdict {
    let contract = engine.contract;
    let verdict = policy_verdict(engine, kept, tally);
    let choosing = matches!(
        verdict,
        CertifiedVerdict::NecessarilyBest(_)
            | CertifiedVerdict::WorstCaseChoice(_)
            | CertifiedVerdict::MinimaxRegretChoice(_)
    );
    if choosing && !set_witnessed(engine, kept) {
        // Every kept box may lie outside the value relations: a choice over a set not shown to
        // hold a point is not made.
        let n = contract.actions.len();
        let mut possible = vec![false; n];
        for item in kept {
            for a in &item.candidates {
                possible[*a] = true;
            }
        }
        return CertifiedVerdict::Undetermined {
            candidates: ids(contract, (0..n).filter(|a| possible[*a])),
            undecided_fraction: tally.undecided_fraction,
        };
    }
    verdict
}

pub(super) fn policy_verdict(
    engine: &Engine<'_>,
    kept: &[Examined],
    tally: &Tally,
) -> CertifiedVerdict {
    let contract = engine.contract;
    let tie = engine.options.tie_tolerance;
    let undetermined = |candidates: Vec<usize>| CertifiedVerdict::Undetermined {
        candidates: ids(contract, candidates),
        undecided_fraction: tally.undecided_fraction,
    };
    match contract.preference_policy {
        PreferencePolicy::ReportOnly | PreferencePolicy::Weighted => CertifiedVerdict::ReportOnly,
        PreferencePolicy::RequireInvariantBestAction => {
            if let Some(first) = kept.first().and_then(|item| item.best)
                && kept.iter().all(|item| item.best == Some(first))
            {
                return CertifiedVerdict::NecessarilyBest(contract.actions[first].id.clone());
            }
            let n = contract.actions.len();
            let mut certified = vec![false; n];
            let mut possible = vec![false; n];
            for item in kept {
                if let (Some(a), false) = (item.best, item.boundary) {
                    certified[a] = true;
                }
                for a in &item.candidates {
                    possible[*a] = true;
                }
            }
            if certified.iter().filter(|c| **c).count() >= 2 {
                CertifiedVerdict::PreferenceDependent {
                    certified_somewhere: ids(contract, (0..n).filter(|a| certified[*a])),
                }
            } else {
                undetermined((0..n).filter(|a| possible[*a]).collect())
            }
        }
        PreferencePolicy::Maximin => match dominant(&tally.worst, tie) {
            Ok(a) => CertifiedVerdict::WorstCaseChoice(contract.actions[a].id.clone()),
            Err(candidates) => undetermined(candidates),
        },
        PreferencePolicy::MinimaxRegret => {
            let negated: Vec<Option<(f64, f64)>> =
                tally.regret.iter().map(|r| r.map(|(lo, hi)| (-hi, -lo))).collect();
            match dominant(&negated, tie) {
                Ok(a) => CertifiedVerdict::MinimaxRegretChoice(contract.actions[a].id.clone()),
                Err(candidates) => undetermined(candidates),
            }
        }
    }
}

/// Measure-weighted values and shares, scores in the larger-is-better orientation.
pub(super) struct MeasureSummary {
    pub(super) value: Vec<Option<f64>>,
    pub(super) share: Vec<Option<f64>>,
    pub(super) verdict: CertifiedVerdict,
    /// The declared tie tolerance plus the largest width of a measure-weighted margin
    /// enclosure.
    pub(super) effective_tie_tolerance: f64,
}

pub(super) fn measure_weights(measure: &PreferenceMeasure) -> Vec<f64> {
    let raw = measure.weights.clone().unwrap_or_else(|| vec![1.0; measure.points.len()]);
    let total: f64 = raw.iter().sum();
    raw.into_iter().map(|w| w / total).collect()
}

pub(super) fn weighted_policy(
    engine: &Engine<'_>,
    measure: &PreferenceMeasure,
) -> Result<MeasureSummary, PreferenceCertifiedError> {
    let n = engine.contract.actions.len();
    let tie = engine.options.tie_tolerance;
    let admissible = engine.admissible_indices();
    let mut value: Vec<Option<f64>> = (0..n).map(|a| engine.admissible[a].then_some(0.0)).collect();
    let mut share = value.clone();
    // Measure-weighted margin enclosures: `totals[a][b]` encloses the measure-weighted
    // `score_a - score_b`.
    let mut totals: Vec<Vec<Option<Interval>>> = vec![vec![None; n]; n];
    for &a in &admissible {
        for &b in admissible.iter().filter(|b| **b != a) {
            totals[a][b] = Some(Interval::point(0.0));
        }
    }
    for (point, weight) in measure.points.iter().zip(measure_weights(measure)) {
        engine.observe()?;
        let scores = engine.point_scores(point)?;
        for &a in &admissible {
            if let (Some(total), Some(score)) = (value[a].as_mut(), scores[a]) {
                *total += weight * score;
            }
        }
        // Point scores are floating evaluations, and a lead between two of them can be pure
        // rounding when a value cancels large terms; a lead counts only when the outward-rounded
        // enclosure of the margin at the point proves it.
        let eval = engine.point_eval(point)?;
        let fail =
            |failure, a: usize| box_refusal(failure, &engine.contract.actions[a].id, point, point);
        for &a in &admissible {
            for &b in admissible.iter().filter(|b| **b != a) {
                let margin =
                    eval.margins[a][b].ok_or_else(|| fail(EncloseFailure::NonFinite, a))?;
                if let Some(total) = totals[a][b].as_mut() {
                    *total = total
                        .plus(margin.scale(weight).map_err(|f| fail(f, a))?)
                        .map_err(|f| fail(f, a))?;
                }
            }
        }
        let leads = |a: usize| {
            admissible.iter().all(|&b| b == a || eval.margins[a][b].is_some_and(|m| m.lo > tie))
        };
        if let Some(leader) = admissible.iter().copied().find(|&a| leads(a))
            && let Some(slot) = share[leader].as_mut()
        {
            *slot += weight;
        }
    }
    let proven = |a: usize, b: usize| totals[a][b].is_some_and(|m| m.lo > tie);
    let resolution = totals
        .iter()
        .flatten()
        .flatten()
        .fold(0.0_f64, |widest, margin| widest.max(margin.hi - margin.lo));
    let verdict = match admissible
        .iter()
        .copied()
        .find(|&a| admissible.iter().all(|&b| b == a || proven(a, b)))
    {
        Some(a) => CertifiedVerdict::MeasureChoice {
            action: engine.contract.actions[a].id.clone(),
            share: share[a].unwrap_or(0.0),
        },
        None => CertifiedVerdict::Undetermined {
            candidates: ids(
                engine.contract,
                admissible
                    .iter()
                    .copied()
                    .filter(|&a| !admissible.iter().any(|&b| b != a && proven(b, a))),
            ),
            undecided_fraction: 0.0,
        },
    };
    Ok(MeasureSummary { value, share, verdict, effective_tie_tolerance: tie + resolution })
}

/// What a run reports besides the per-action table.
pub(super) struct Subdivided {
    pub(super) regions: Vec<CertifiedRegion>,
    pub(super) undecided_fraction: f64,
    pub(super) boxes_examined: usize,
    pub(super) verdict: CertifiedVerdict,
    pub(super) effective_tie_tolerance: f64,
}

/// Subdivide, fill the per-action enclosures and fractions, and decide.
pub(super) fn subdivided(
    engine: &Engine<'_>,
    actions: &mut [CertifiedActionOutcome],
) -> Result<Subdivided, PreferenceCertifiedError> {
    let contract = engine.contract;
    let sign = orientation(contract.criterion);
    let (kept, boxes_examined) = engine.subdivide()?;
    let tally = tally(engine, &kept)?;
    for (a, outcome) in actions.iter_mut().enumerate() {
        outcome.best_fraction = tally.best_fraction[a];
        outcome.value_bounds = tally.value[a].map(|v| oriented(sign, v));
        outcome.worst_case_bounds = tally.worst[a].map(|v| oriented(sign, v));
        outcome.worst_case_regret_bounds = tally.regret[a];
    }
    let verdict = subdivision_verdict(engine, &kept, &tally);
    let mut regions: Vec<CertifiedRegion> = kept
        .into_iter()
        .map(|item| {
            let (lower, upper) = engine.reported(&item);
            CertifiedRegion {
                status: match item.best {
                    Some(a) => RegionStatus::Best(contract.actions[a].id.clone()),
                    None => RegionStatus::Undecided { candidates: ids(contract, item.candidates) },
                },
                lower,
                upper,
                touches_relation_boundary: item.boundary,
            }
        })
        .collect();
    regions.sort_by(|x, y| {
        lexicographic(&x.lower, &y.lower).then_with(|| lexicographic(&x.upper, &y.upper))
    });
    Ok(Subdivided {
        regions,
        undecided_fraction: tally.undecided_fraction,
        boxes_examined,
        verdict,
        effective_tie_tolerance: engine.options.tie_tolerance,
    })
}
