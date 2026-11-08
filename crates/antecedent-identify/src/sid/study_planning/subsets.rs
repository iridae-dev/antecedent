use super::{
    DistributionAvailability, EvidenceCatalog, LawOrigin, RegimeKind, STUDY_PLAN_MAX_COST_UNITS,
    SamplingSelection, StudyPlanCandidate, StudyPlanRefusal, StudyPlanRoute,
};
use std::collections::BTreeSet;
use std::sync::Arc;

pub(super) fn subset_label(ids: &[Arc<str>]) -> String {
    format!("subset:[{}]", ids.iter().map(AsRef::as_ref).collect::<Vec<_>>().join(","))
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-'))
}

/// Check a candidate universe against the bounds, the base catalog and the
/// route's population semantics. Candidates are returned sorted by id.
#[allow(clippy::too_many_lines, reason = "One linear pass over every declared candidate rule")]
pub(super) fn validate_candidates(
    route: &StudyPlanRoute,
    catalog: &EvidenceCatalog,
    candidates: &[StudyPlanCandidate],
) -> Result<Vec<StudyPlanCandidate>, StudyPlanRefusal> {
    use StudyPlanRefusal as R;
    if candidates.is_empty() {
        return Err(R::invalid_candidate("a plan needs at least one candidate study"));
    }
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    let ids = sorted.iter().map(|c| Arc::clone(&c.id)).collect::<BTreeSet<_>>();
    if ids.len() != sorted.len() {
        return Err(R::invalid_candidate("candidate ids must be unique"));
    }
    let mut regimes = catalog.regimes.iter().map(|r| r.id).collect::<BTreeSet<_>>();
    // Labels are unique per catalog: a label two candidates share would make
    // their joint preview fail after the plan started, so it is refused here.
    let mut labels =
        catalog.regimes.iter().filter_map(|r| r.label.clone()).collect::<BTreeSet<_>>();
    let target = route.target();
    for candidate in &sorted {
        let who = &candidate.id;
        if !valid_id(who) {
            return Err(R::invalid_candidate(format!(
                "candidate id {who:?} must be 1-64 characters of [A-Za-z0-9_.:-]"
            )));
        }
        if candidate.cost_units == 0 || candidate.cost_units > STUDY_PLAN_MAX_COST_UNITS {
            return Err(R::invalid_candidate(format!(
                "candidate {who} must cost 1 to {STUDY_PLAN_MAX_COST_UNITS} units"
            )));
        }
        if candidate.sample_budget > STUDY_PLAN_MAX_COST_UNITS {
            return Err(R::invalid_candidate(format!(
                "candidate {who} sample budget is too large"
            )));
        }
        if candidate.delta.proposed_regimes.is_empty() {
            return Err(R::invalid_candidate(format!("candidate {who} proposes no regime")));
        }
        for other in candidate.requires.iter().chain(candidate.conflicts.iter()) {
            if !ids.contains(other) || other == who {
                return Err(R::invalid_candidate(format!(
                    "candidate {who} names unknown or self constraint {other}"
                )));
            }
        }
        if candidate.requires.iter().any(|r| candidate.conflicts.contains(r)) {
            return Err(R::invalid_candidate(format!(
                "candidate {who} both requires and conflicts with a candidate"
            )));
        }
        candidate.delta.preview_catalog(catalog).map_err(|error| {
            R::invalid_candidate(format!("candidate {who} does not preview on the base: {error}"))
        })?;
        for regime in candidate.delta.proposed_regimes.iter() {
            if !regimes.insert(regime.id) {
                return Err(R::invalid_candidate(format!(
                    "regime {} of candidate {who} repeats a base or candidate regime id",
                    regime.id.raw()
                )));
            }
            if let Some(label) = &regime.label
                && !labels.insert(Arc::clone(label))
            {
                return Err(R::invalid_candidate(format!(
                    "regime {} of candidate {who} repeats the label {label:?} of a base or \
                         candidate regime",
                    regime.id.raw()
                )));
            }
            if regime.measured.is_empty()
                || !regime.conditioned_on.is_empty()
                || regime.distribution != DistributionAvailability::Joint
                || regime.selection != SamplingSelection::Population
                || regime.origin != LawOrigin::Measured
            {
                return Err(R::invalid_candidate(format!(
                    "regime {} of candidate {who} must be a measured, unconditioned, \
                     whole-population joint law with a non-empty margin",
                    regime.id.raw()
                )));
            }
            if regime.population.as_ref() == target {
                if matches!(route, StudyPlanRoute::Mz(_)) && regime.kind == RegimeKind::Experimental
                {
                    return Err(R::invalid_candidate(format!(
                        "candidate {who} experiments in the target; the mz route's target \
                         supplies observational evidence only"
                    )));
                }
            } else {
                let Some(source) =
                    route.sources().iter().find(|s| s.population == regime.population)
                else {
                    return Err(R::invalid_candidate(format!(
                        "candidate {who} studies population {} that is neither the target nor a \
                         declared source (new populations are not planned)",
                        regime.population
                    )));
                };
                if regime.interventions.iter().any(|v| !source.controllable.contains(v)) {
                    return Err(R::invalid_candidate(format!(
                        "candidate {who} intervenes outside source {}'s declared controllable set",
                        source.population
                    )));
                }
            }
            if matches!(route, StudyPlanRoute::Mixed(_)) && !regime.intervention_values.is_empty() {
                return Err(R::invalid_candidate(format!(
                    "candidate {who} restricts its levels; the mixed route states \
                     identification for every level and never cites a value-restricted study"
                )));
            }
        }
    }
    Ok(sorted)
}

/// The next `k`-combination of `0..n` in lexicographic order; false after the last.
fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if indices[i] < n - k + i {
            indices[i] += 1;
            for j in i + 1..k {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// The largest subset size whose cumulative subset count (sizes one to that size, out of `n`
/// candidates) fits `operations`. Every evaluated subset charges at least one operation, so a
/// larger enumeration could not be paid for; at least singletons are always listed.
pub(super) fn subset_size_limit(n: usize, operations: usize) -> usize {
    let budget = operations as u128;
    let (mut total, mut choose, mut size) = (0u128, 1u128, 0usize);
    while size < n {
        let next = choose.saturating_mul((n - size) as u128) / (size as u128 + 1);
        if total.saturating_add(next) > budget {
            break;
        }
        total += next;
        choose = next;
        size += 1;
    }
    size.max(1).min(n)
}

/// Every non-empty subset of at most `max_size` candidates, as
/// sorted index lists, in ranking order.
pub(super) fn subsets(
    candidates: &[StudyPlanCandidate],
    max_size: usize,
) -> Result<Vec<(Vec<usize>, u64, u64)>, StudyPlanRefusal> {
    let n = candidates.len();
    let mut out = Vec::new();
    for size in 1..=max_size.min(n) {
        let mut indices = (0..size).collect::<Vec<_>>();
        loop {
            out.push(indices.clone());
            if !next_combination(&mut indices, n) {
                break;
            }
        }
    }
    let mut keyed = out
        .into_iter()
        .map(|members| {
            let cost = members
                .iter()
                .try_fold(0u64, |total, &i| total.checked_add(candidates[i].cost_units))
                .ok_or_else(|| {
                    StudyPlanRefusal::invalid_candidate("subset total cost exceeds u64")
                })?;
            let samples = members
                .iter()
                .try_fold(0u64, |total, &i| total.checked_add(candidates[i].sample_budget))
                .ok_or_else(|| {
                    StudyPlanRefusal::invalid_candidate("subset total sample budget exceeds u64")
                })?;
            Ok((members, cost, samples))
        })
        .collect::<Result<Vec<_>, StudyPlanRefusal>>()?;
    keyed.sort_by(|a, b| {
        a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.len().cmp(&b.0.len())).then_with(|| {
            let ids =
                |m: &[usize]| m.iter().map(|&i| Arc::clone(&candidates[i].id)).collect::<Vec<_>>();
            ids(&a.0).cmp(&ids(&b.0))
        })
    });
    Ok(keyed)
}

pub(super) fn infeasible(
    candidates: &[StudyPlanCandidate],
    members: &[usize],
) -> Option<&'static str> {
    let ids = members.iter().map(|&i| &candidates[i].id).collect::<BTreeSet<_>>();
    for &i in members {
        if candidates[i].requires.iter().any(|r| !ids.contains(r)) {
            return Some("requires");
        }
        if candidates[i].conflicts.iter().any(|c| ids.contains(c)) {
            return Some("conflicts");
        }
    }
    None
}
