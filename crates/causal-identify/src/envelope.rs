//! Identification envelopes over graph classes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::result::{IdentificationResult, IdentificationStatus, IdentifiedEstimand};

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
}

impl<G> IdentificationEnvelope<G> {
    /// Build an envelope from weighted cases, preserving unidentified mass.
    #[must_use]
    pub fn from_cases(cases: Vec<GraphIdentificationCase<G>>) -> Self {
        let mut identified = 0.0;
        let mut unidentified = 0.0;
        let mut all_id = true;
        let mut any_id = false;
        let mut invariant: Option<IdentifiedEstimand> = None;
        let mut invariant_conflict = false;
        for c in &cases {
            match c.result.status {
                IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
                | IdentificationStatus::IdentifiedUnderPriorRestrictions
                | IdentificationStatus::PartiallyIdentified => {
                    any_id = true;
                    identified += c.weight.0;
                    if let Some(est) = c.result.estimands.first() {
                        match &invariant {
                            None => invariant = Some(est.clone()),
                            Some(prev) if prev.method != est.method => {
                                invariant_conflict = true;
                            }
                            _ => {}
                        }
                    }
                }
                IdentificationStatus::NotIdentified | IdentificationStatus::GraphDependent => {
                    all_id = false;
                    unidentified += c.weight.0;
                }
            }
        }
        let status = if cases.is_empty() {
            IdentificationStatus::NotIdentified
        } else if all_id && any_id && !invariant_conflict {
            IdentificationStatus::NonparametricallyIdentified
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
        Self {
            invariant,
            cases,
            identified_weight: ProbabilityMass(identified),
            unidentified_weight: ProbabilityMass(unidentified),
            critical_graph_features,
            status,
        }
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
        push(
            "unidentified_mass",
            format!("unidentified_weight={unidentified}"),
        );
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
    use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, VariableId};
    use causal_expr::CausalExprArena;

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
            env.critical_graph_features
                .iter()
                .any(|f| f.kind.as_ref() == "unidentified_mass"),
            "features={:?}",
            env.critical_graph_features
        );
    }
}
