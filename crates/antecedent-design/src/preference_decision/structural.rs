//! The combination of per-structure analyses across structures under the structural policy:
//! joined receipts, merged exclusions, the verdict the structures agree on, and the combined
//! table of each policy.

// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(clippy::wildcard_imports, reason = "a part of its parent module")]
use super::*;

/// One evaluated structure, with its own analysis under the overall exclusions.
pub(super) struct Evaluated<'a> {
    pub(super) id: &'a str,
    pub(super) probability: Option<f64>,
    /// Whether its exclusions apply in every structure.
    pub(super) excludes: bool,
    pub(super) facts: &'a SourceFacts,
    pub(super) analysis: Analysis,
    pub(super) verdict: PreferenceVerdict,
}

pub(super) struct StructureFacts {
    pub(super) all_evaluated: bool,
    pub(super) unidentified_mass: f64,
    pub(super) unevaluated_mass: f64,
    pub(super) evaluated_mass: f64,
}

impl StructureFacts {
    /// The evidence the structural policy needs and does not have, as the ordinary aggregation
    /// states it.
    pub(super) fn missing_evidence(&self, policy: StructuralPolicy) -> Option<&'static str> {
        missing_evidence(
            policy_kind(policy),
            self.all_evaluated,
            self.unidentified_mass,
            self.unevaluated_mass,
            self.evaluated_mass,
        )
    }
}

/// Each field lists the distinct values of the receipts, in order, joined by `+`.
pub(super) fn joined(receipts: &[&SourceReceipt]) -> SourceReceipt {
    let join = |field: fn(&SourceReceipt) -> &str| {
        let mut values: Vec<&str> = Vec::new();
        for receipt in receipts {
            let value = field(receipt);
            if !values.contains(&value) {
                values.push(value);
            }
        }
        values.join("+")
    };
    SourceReceipt {
        provider_id: join(|r| r.provider_id.as_str()),
        snapshot_id: join(|r| r.snapshot_id.as_str()),
        rng_id: join(|r| r.rng_id.as_str()),
        causal_contract_id: join(|r| r.causal_contract_id.as_str()),
    }
}

/// Exclusions of each action across the structures that exclude, first occurrence of each
/// constraint kept.
pub(super) fn merged_exclusions(
    evaluated: &[Evaluated<'_>],
    n: usize,
) -> Vec<Vec<ConstraintExclusion>> {
    let mut merged: Vec<Vec<ConstraintExclusion>> = vec![Vec::new(); n];
    for atom in evaluated.iter().filter(|atom| atom.excludes) {
        for (a, list) in atom.facts.exclusions.iter().enumerate() {
            for exclusion in list {
                if !merged[a].iter().any(|e| e.constraint_id == exclusion.constraint_id) {
                    merged[a].push(exclusion.clone());
                }
            }
        }
    }
    merged
}

/// The verdict shared by every structure, if they agree. Measure choices of one action agree
/// with the smallest share. A unique winner in some structures that is never beaten (with ties)
/// in the others is never beaten across structures, tying with every rival it ties anywhere,
/// listed in the declaration order `order`.
pub(super) fn agreed(evaluated: &[Evaluated<'_>], order: &[String]) -> Option<PreferenceVerdict> {
    let first = &evaluated.first()?.verdict;
    if evaluated.iter().all(|atom| &atom.verdict == first) {
        return Some(first.clone());
    }
    let unbeaten = |verdict: &PreferenceVerdict| match verdict {
        PreferenceVerdict::NecessarilyBest(action) => Some((action.clone(), Vec::new())),
        PreferenceVerdict::NeverBeaten { action, ties_with } => {
            Some((action.clone(), ties_with.clone()))
        }
        _ => None,
    };
    if let Some((action, _)) = unbeaten(first) {
        let mut ties: Vec<String> = Vec::new();
        for atom in evaluated {
            let (other, with) = unbeaten(&atom.verdict)?;
            if other != action {
                return None;
            }
            ties.extend(with);
        }
        let ties_with: Vec<String> =
            order.iter().filter(|id| ties.contains(*id)).cloned().collect();
        return Some(PreferenceVerdict::NeverBeaten { action, ties_with });
    }
    if let PreferenceVerdict::MeasureChoice { action, .. } = first {
        let mut smallest = f64::INFINITY;
        for atom in evaluated {
            match &atom.verdict {
                PreferenceVerdict::MeasureChoice { action: other, share } if other == action => {
                    smallest = smallest.min(*share);
                }
                _ => return None,
            }
        }
        return Some(PreferenceVerdict::MeasureChoice { action: action.clone(), share: smallest });
    }
    None
}

/// Ranges over structures: lowest of the lows and highest of the highs. The standing of each
/// action is read from the combined margins, so an action is never beaten (uniquely best,
/// dominated by a rival) across structures exactly when it is in every structure.
pub(super) fn combine_invariant(
    analyses: &[&Analysis],
    admissible: &[bool],
    declared: f64,
) -> Analysis {
    let n = admissible.len();
    let tolerance = analyses.iter().map(|a| a.tolerance).fold(declared, f64::max);
    let mut value = vec![None; n];
    let mut margins = MarginTable::new(n);
    let mut possibly = vec![Some(false); n];
    let mut regret: Vec<Option<f64>> = vec![None; n];
    for analysis in analyses {
        for a in 0..n {
            value[a] = match (value[a], analysis.value[a]) {
                (None, next) => next,
                (current, None) => current,
                (Some((lo, hi)), Some((l, h))) => Some((lo.min(l), hi.max(h))),
            };
            possibly[a] = match (possibly[a], analysis.possibly[a]) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (None, _) | (_, None) => None,
                _ => Some(false),
            };
            regret[a] = match (regret[a], analysis.regret[a]) {
                (None, next) => next,
                (current, None) => current,
                (Some(x), Some(y)) => Some(x.max(y)),
            };
            for b in (a + 1)..n {
                let Some(next) = analysis.margins.get(a, b) else { continue };
                let slot = &mut margins.entries[a * n + b];
                *slot = Some(match slot.take() {
                    None => next,
                    Some(mut current) => {
                        if next.min < current.min {
                            current.min = next.min;
                            current.min_at = next.min_at;
                        }
                        if next.max > current.max {
                            current.max = next.max;
                            current.max_at = next.max_at;
                        }
                        current
                    }
                });
            }
        }
    }
    let standing = if analyses.is_empty() {
        Standing::empty(n)
    } else {
        let active: Vec<usize> = (0..n).filter(|&a| admissible[a]).collect();
        Standing::of(&margins, &active, n, tolerance)
    };
    Analysis {
        value,
        margins,
        standing,
        possibly,
        regret,
        share: vec![None; n],
        measure_value: vec![None; n],
        unresolved_share: None,
        tolerance,
    }
}

impl Prepared<'_> {
    /// The combined table and the structural verdict.
    #[allow(clippy::too_many_lines, reason = "one branch per structural policy")]
    pub(super) fn combine(
        &self,
        policy: StructuralPolicy,
        evaluated: &[Evaluated<'_>],
        admissible: &[bool],
        state: &StructureFacts,
    ) -> Result<(Option<PreferenceDecisionResult>, StructuralPreferenceVerdict), E> {
        if policy == StructuralPolicy::ReportOnly {
            return Ok((None, StructuralPreferenceVerdict::ReportOnly));
        }
        if evaluated.is_empty() {
            let verdict = if self.contract.preference_policy == PreferencePolicy::ReportOnly {
                StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
            } else {
                StructuralPreferenceVerdict::InsufficientScience(NO_STRUCTURE_EVALUATED.into())
            };
            return Ok((None, verdict));
        }
        let n = admissible.len();
        let receipts: Vec<&SourceReceipt> = evaluated.iter().map(|e| &e.facts.receipt).collect();
        let source = joined(&receipts);
        let n_draws = evaluated.iter().map(|e| e.facts.n_draws).sum();
        let exclusions = merged_exclusions(evaluated, n);
        let report = self.contract.preference_policy == PreferencePolicy::ReportOnly;
        let combined = match policy {
            StructuralPolicy::BayesOverStructures => {
                let mut mixture = vec![vec![0.0; self.parts()]; n];
                for atom in evaluated {
                    let p = atom.probability.unwrap_or(0.0);
                    for (row, own) in mixture.iter_mut().zip(&atom.facts.coefficients) {
                        for (slot, c) in row.iter_mut().zip(own) {
                            *slot += p * c;
                        }
                    }
                }
                let pieces: Vec<Vec<Vec<f64>>> = mixture.iter().map(|c| vec![c.clone()]).collect();
                // The mixture adds one product and one sum per structure to each coefficient.
                let rounding = evaluated.iter().map(|e| e.facts.rounding).fold(0.0, f64::max)
                    + summation_rounding(
                        evaluated.len(),
                        self.coefficient_magnitude(
                            evaluated.iter().flat_map(|e| e.facts.coefficients.iter()),
                        ),
                    );
                let analysis = self.analyse(&pieces, admissible, rounding)?;
                let result = self.assemble(Assembly {
                    analysis: &analysis,
                    admissible,
                    exclusions,
                    coefficients: mixture.into_iter().map(Some).collect(),
                    pieces: Some(&pieces),
                    source,
                    n_draws,
                });
                let verdict = if report {
                    StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
                } else if let Some(why) = state.missing_evidence(policy) {
                    StructuralPreferenceVerdict::InsufficientScience(why.into())
                } else {
                    StructuralPreferenceVerdict::Combined(result.verdict.clone())
                };
                (result, verdict)
            }
            StructuralPolicy::Maximin => {
                let pieces: Vec<Vec<Vec<f64>>> = (0..n)
                    .map(|a| {
                        evaluated
                            .iter()
                            .map(|e| e.facts.coefficients[a].clone())
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let rounding = evaluated.iter().map(|e| e.facts.rounding).fold(0.0, f64::max);
                let analysis = self.analyse(&pieces, admissible, rounding)?;
                let result = self.assemble(Assembly {
                    analysis: &analysis,
                    admissible,
                    exclusions,
                    coefficients: vec![None; n],
                    pieces: Some(&pieces),
                    source,
                    n_draws,
                });
                let verdict = if report {
                    StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
                } else if let Some(why) = state.missing_evidence(policy) {
                    StructuralPreferenceVerdict::InsufficientScience(why.into())
                } else {
                    StructuralPreferenceVerdict::Combined(result.verdict.clone())
                };
                (result, verdict)
            }
            _ => self.invariant_combined(evaluated, admissible, state, exclusions, source, n_draws),
        };
        Ok((Some(combined.0), combined.1))
    }

    pub(super) fn invariant_combined(
        &self,
        evaluated: &[Evaluated<'_>],
        admissible: &[bool],
        state: &StructureFacts,
        exclusions: Vec<Vec<ConstraintExclusion>>,
        source: SourceReceipt,
        n_draws: usize,
    ) -> (PreferenceDecisionResult, StructuralPreferenceVerdict) {
        let n = admissible.len();
        let analyses: Vec<&Analysis> = evaluated.iter().map(|e| &e.analysis).collect();
        let analysis = combine_invariant(&analyses, admissible, self.options.tie_tolerance);
        let mut result = self.assemble(Assembly {
            analysis: &analysis,
            admissible,
            exclusions,
            coefficients: vec![None; n],
            pieces: None,
            source,
            n_draws,
        });
        let order: Vec<String> = self.contract.actions.iter().map(|a| a.id.clone()).collect();
        let agreement = agreed(evaluated, &order);
        result.verdict = agreement.clone().unwrap_or_else(|| {
            if admissible.iter().any(|a| *a) {
                PreferenceVerdict::PreferenceDependent {
                    possibly_best: (0..n)
                        .filter(|&a| admissible[a] && analysis.possibly[a] != Some(false))
                        .map(|a| self.contract.actions[a].id.clone())
                        .collect(),
                }
            } else {
                PreferenceVerdict::NoAdmissibleAction
            }
        });
        let verdict = if self.contract.preference_policy == PreferencePolicy::ReportOnly {
            StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
        } else if let Some(why) =
            state.missing_evidence(StructuralPolicy::RequireInvariantBestAction)
        {
            StructuralPreferenceVerdict::InsufficientScience(why.into())
        } else if let Some(verdict) = agreement {
            StructuralPreferenceVerdict::Combined(verdict)
        } else {
            StructuralPreferenceVerdict::StructuresDisagree(
                evaluated
                    .iter()
                    .map(|e| AtomChoice { atom: e.id.to_owned(), verdict: e.verdict.clone() })
                    .collect(),
            )
        };
        (result, verdict)
    }
}
