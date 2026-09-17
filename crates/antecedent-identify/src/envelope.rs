//! Identification envelopes over graph classes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::result::{IdentificationResult, IdentificationStatus, IdentifiedEstimand};

/// Whether a case with this status contributes to an envelope's identified
/// mass.
///
/// The single owner of that list. [`IdentificationEnvelope::from_cases`] splits
/// `identified_weight` from `unidentified_weight` by exactly this predicate, so
/// any other reading of "which completion mass is identified" — a class
/// mixture, a diagnostic, a published mass — must ask here rather than restate
/// the arms, or it contradicts the envelope it was built from.
///
/// It is deliberately wider than "which cases may be estimated": a completion
/// identified only under prior restrictions carries identified mass and its
/// assumptions, but no frequentist arm estimates it, so its mass lands in the
/// unevaluable bucket rather than the unidentified one.
#[must_use]
pub const fn carries_identified_mass(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
    )
}

/// Probability mass on `[0, 1]` (not necessarily normalized across fields alone).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbabilityMass(pub f64);

impl ProbabilityMass {
    /// Zero mass.
    #[must_use]
    pub const fn zero() -> Self {
        Self(0.0)
    }
}

/// Critical graph feature blocking identification or driving case splits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphFeature {
    /// Feature tag.
    pub kind: Arc<str>,
    /// Detail.
    pub detail: Arc<str>,
}

/// One case in an identification envelope.
#[derive(Clone, Debug)]
pub struct GraphIdentificationCase<G> {
    /// Graph (or completion) for this case.
    pub graph: G,
    /// Identification result on this graph.
    pub result: IdentificationResult,
    /// Weight / probability mass of this case.
    pub weight: ProbabilityMass,
}

/// Ensemble / equivalence-class identification result .
///
/// Unidentified mass is preserved explicitly .
#[derive(Clone, Debug)]
pub struct IdentificationEnvelope<G> {
    /// Estimand shared by all identified cases, if any.
    pub invariant: Option<IdentifiedEstimand>,
    /// Per-graph cases (bounded by caller / sampler).
    pub cases: Vec<GraphIdentificationCase<G>>,
    /// Total identified weight.
    pub identified_weight: ProbabilityMass,
    /// Total unidentified weight (must not be dropped).
    pub unidentified_weight: ProbabilityMass,
    /// Features that drive splits or non-ID.
    pub critical_graph_features: Vec<GraphFeature>,
    /// Aggregate status.
    pub status: IdentificationStatus,
    /// Count of cases whose search was truncated before it could determine identifiability
    /// (e.g. a MAG-completion enumeration cap exceeded, see
    /// [`crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE`]), as opposed to a case whose
    /// search completed and proved non-identifiability. Truncated cases still contribute to
    /// `unidentified_weight` like any other unidentified case (unidentified mass is never
    /// dropped) — this counter exists so a caller can tell the two apart instead of silently
    /// treating "we could not tell" the same as "we proved it's impossible".
    pub truncated_completions: usize,
}

impl<G> IdentificationEnvelope<G> {
    /// Build an envelope from weighted cases, preserving unidentified mass.
    #[must_use]
    pub fn from_cases(cases: Vec<GraphIdentificationCase<G>>) -> Self {
        let mut identified = 0.0;
        let mut unidentified = 0.0;
        let mut all_id = true;
        let mut any_id = false;
        let mut any_parametric = false;
        let mut any_prior_restricted = false;
        let mut any_partial = false;
        let mut invariant: Option<IdentifiedEstimand> = None;
        let mut invariant_conflict = false;
        for c in &cases {
            // `carries_identified_mass` is the list; this is the split it owns.
            if carries_identified_mass(c.result.status) {
                any_id = true;
                any_parametric |= matches!(
                    c.result.status,
                    IdentificationStatus::IdentifiedUnderParametricRestrictions
                );
                any_prior_restricted |= matches!(
                    c.result.status,
                    IdentificationStatus::IdentifiedUnderPriorRestrictions
                );
                any_partial |= matches!(c.result.status, IdentificationStatus::PartiallyIdentified);
                identified += c.weight.0;
                if let Some(est) = c.result.estimands.first() {
                    match &invariant {
                        None => invariant = Some(est.clone()),
                        Some(prev) if !estimands_agree(prev, est) => {
                            invariant_conflict = true;
                        }
                        _ => {}
                    }
                }
            } else {
                all_id = false;
                unidentified += c.weight.0;
            }
        }
        let status = if cases.is_empty() {
            IdentificationStatus::NotIdentified
        } else if all_id && any_id && !invariant_conflict {
            if any_partial {
                IdentificationStatus::PartiallyIdentified
            } else if any_prior_restricted {
                IdentificationStatus::IdentifiedUnderPriorRestrictions
            } else if any_parametric {
                IdentificationStatus::IdentifiedUnderParametricRestrictions
            } else {
                IdentificationStatus::NonparametricallyIdentified
            }
        } else if any_id && unidentified > 0.0 {
            IdentificationStatus::GraphDependent
        } else if any_id {
            IdentificationStatus::PartiallyIdentified
        } else {
            IdentificationStatus::NotIdentified
        };
        if invariant_conflict {
            invariant = None;
        }
        let critical_graph_features = collect_critical_features(&cases, status, unidentified);
        let truncated_completions = cases.iter().filter(|c| case_truncated(c)).count();
        Self {
            invariant,
            cases,
            identified_weight: ProbabilityMass(identified),
            unidentified_weight: ProbabilityMass(unidentified),
            critical_graph_features,
            status,
            truncated_completions,
        }
    }

    /// Weight of cases whose search was truncated before it could determine
    /// identifiability. This mass is counted in [`Self::unidentified_weight`]
    /// like any other unidentified case; a caller that must separate "could
    /// not tell" from "proved impossible" subtracts it.
    #[must_use]
    pub fn truncated_weight(&self) -> f64 {
        self.cases.iter().filter(|case| case_truncated(case)).map(|case| case.weight.0).sum()
    }

    /// Merge additional critical features (e.g. source-PAG circle marks) without duplicates.
    pub fn push_features(&mut self, extra: impl IntoIterator<Item = GraphFeature>) {
        for f in extra {
            if !self
                .critical_graph_features
                .iter()
                .any(|e| e.kind == f.kind && e.detail == f.detail)
            {
                self.critical_graph_features.push(f);
            }
        }
    }
}

/// Whether this case's search was truncated before it could decide.
fn case_truncated<G>(case: &GraphIdentificationCase<G>) -> bool {
    search_truncated(&case.result)
}

/// Whether an identification search stopped at a budget (a completion or
/// history cap) before it could decide, rather than completing.
///
/// A truncated `NotIdentified` is not a proof of non-identification.
#[must_use]
pub fn search_truncated(result: &crate::IdentificationResult) -> bool {
    result.diagnostics.iter().any(|d| {
        d.code.as_ref() == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
            || d.code.as_ref() == crate::temporal_mag::HISTORY_CAPPED
    })
}

/// Class-wide invariant estimands must agree on the functional roles, not just the method tag.
/// `functional` `ExprIds` live in per-case arenas and are not comparable.
fn estimands_agree(a: &IdentifiedEstimand, b: &IdentifiedEstimand) -> bool {
    a.method == b.method
        && a.adjustment_set == b.adjustment_set
        && a.instruments == b.instruments
        && a.mediators == b.mediators
        && a.rd_design == b.rd_design
}

fn collect_critical_features<G>(
    cases: &[GraphIdentificationCase<G>],
    status: IdentificationStatus,
    unidentified: f64,
) -> Vec<GraphFeature> {
    let mut features = Vec::new();
    let mut seen = std::collections::BTreeSet::<(Arc<str>, Arc<str>)>::new();
    let mut push = |kind: &str, detail: String| {
        let kind: Arc<str> = Arc::from(kind);
        let detail: Arc<str> = Arc::from(detail);
        if seen.insert((Arc::clone(&kind), Arc::clone(&detail))) {
            features.push(GraphFeature { kind, detail });
        }
    };

    if unidentified > 0.0 {
        push("unidentified_mass", format!("unidentified_weight={unidentified}"));
    }
    if matches!(status, IdentificationStatus::GraphDependent) {
        push(
            "graph_dependent",
            "identified and unidentified completions both have positive mass".into(),
        );
    }

    let mut unidentified_cases = 0u64;
    for c in cases {
        match c.result.status {
            IdentificationStatus::NotIdentified | IdentificationStatus::GraphDependent => {
                unidentified_cases += 1;
                for step in &c.result.derivation.steps {
                    if step.rule.as_ref().contains("not")
                        || step.detail.as_ref().contains("not a MAG")
                        || step.detail.as_ref().contains("no qualifying")
                        || step.detail.as_ref().contains("exceeds")
                    {
                        push("completion_block", step.detail.as_ref().to_string());
                    }
                }
                for d in &c.result.diagnostics {
                    push("diagnostic", format!("{}: {}", d.code, d.message));
                }
            }
            _ => {}
        }
    }
    if unidentified_cases > 0 {
        push(
            "unidentified_completions",
            format!("{unidentified_cases} completion(s) not identified"),
        );
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{DerivationTrace, IdentificationPerformanceRecord};
    use antecedent_core::{AssumptionSet, AverageEffectQuery, CausalQuery, VariableId};
    use antecedent_expr::CausalExprArena;

    fn dummy_result(status: IdentificationStatus) -> IdentificationResult {
        IdentificationResult {
            status,
            query: CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            )),
            estimands: Vec::new(),
            arena: CausalExprArena::new(),
            derivation: DerivationTrace::default(),
            required_assumptions: AssumptionSet::default(),
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord::default(),
            hedge: None,
        }
    }

    #[test]
    fn preserves_unidentified_mass() {
        let cases = vec![
            GraphIdentificationCase {
                graph: 0u32,
                result: dummy_result(IdentificationStatus::NonparametricallyIdentified),
                weight: ProbabilityMass(0.4),
            },
            GraphIdentificationCase {
                graph: 1u32,
                result: dummy_result(IdentificationStatus::NotIdentified),
                weight: ProbabilityMass(0.6),
            },
        ];
        let env = IdentificationEnvelope::from_cases(cases);
        assert!((env.identified_weight.0 - 0.4).abs() < 1e-12);
        assert!((env.unidentified_weight.0 - 0.6).abs() < 1e-12);
        assert_eq!(env.status, IdentificationStatus::GraphDependent);
        assert!(
            env.critical_graph_features.iter().any(|f| f.kind.as_ref() == "unidentified_mass"),
            "features={:?}",
            env.critical_graph_features
        );
    }

    // An envelope with no capped case truncates exactly zero weight, not
    // approximately zero: the sum runs over an empty set.
    #[allow(clippy::float_cmp)]
    #[test]
    fn truncated_weight_separates_capped_search_from_proved_non_identification() {
        let mut capped = dummy_result(IdentificationStatus::NotIdentified);
        capped.diagnostics.push(antecedent_core::Diagnostic::new(
            crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE,
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Warning,
            "completion enumeration exceeded its budget",
        ));
        let env = IdentificationEnvelope::from_cases(vec![
            GraphIdentificationCase {
                graph: 0u32,
                result: dummy_result(IdentificationStatus::NonparametricallyIdentified),
                weight: ProbabilityMass(0.25),
            },
            GraphIdentificationCase {
                graph: 1u32,
                result: dummy_result(IdentificationStatus::NotIdentified),
                weight: ProbabilityMass(0.5),
            },
            GraphIdentificationCase { graph: 2u32, result: capped, weight: ProbabilityMass(0.25) },
        ]);
        assert_eq!(env.truncated_completions, 1);
        assert!((env.truncated_weight() - 0.25).abs() < 1e-12);
        assert!((env.unidentified_weight.0 - 0.75).abs() < 1e-12, "capped mass is not dropped");
        let complete = IdentificationEnvelope::from_cases(vec![GraphIdentificationCase {
            graph: 0u32,
            result: dummy_result(IdentificationStatus::NotIdentified),
            weight: ProbabilityMass(1.0),
        }]);
        assert_eq!(complete.truncated_weight(), 0.0);
    }

    #[test]
    fn aggregate_status_never_upgrades_restricted_identification() {
        let parametric = vec![GraphIdentificationCase {
            graph: 0u32,
            result: dummy_result(IdentificationStatus::IdentifiedUnderParametricRestrictions),
            weight: ProbabilityMass(1.0),
        }];
        assert_eq!(
            IdentificationEnvelope::from_cases(parametric).status,
            IdentificationStatus::IdentifiedUnderParametricRestrictions
        );

        let prior_restricted = vec![GraphIdentificationCase {
            graph: 0u32,
            result: dummy_result(IdentificationStatus::IdentifiedUnderPriorRestrictions),
            weight: ProbabilityMass(1.0),
        }];
        assert_eq!(
            IdentificationEnvelope::from_cases(prior_restricted).status,
            IdentificationStatus::IdentifiedUnderPriorRestrictions
        );

        let partial = vec![GraphIdentificationCase {
            graph: 0u32,
            result: dummy_result(IdentificationStatus::PartiallyIdentified),
            weight: ProbabilityMass(1.0),
        }];
        assert_eq!(
            IdentificationEnvelope::from_cases(partial).status,
            IdentificationStatus::PartiallyIdentified
        );
    }
}
