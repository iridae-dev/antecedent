//! Scientific permission to consume a cached decision.
//!
//! A cache hit is an identity comparison. Permission is the current
//! classification of the decision's dependencies: source withdrawal,
//! supersession, provider trust, support and calibration are read again at
//! the consuming coordinate through the composition graph and the evidence
//! ledger. Repeated observations are counted once, and only after a ledger
//! result has declared the reuse.
//!
//! This module does not score utilities. [`crate::decision_eval`] remains the
//! decision runtime.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::composition_verification::{
    CalibrationState, ClaimRevalidation, CompositionGraph, GraphError, PriorAuthority,
    Revalidation, Withdrawal, WithdrawalKind, ledger_effects, revalidate,
};
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, LedgerError, MassAccount, ResultId, ResultStatus, SourceId, SourceStatus,
};
use antecedent_core::SemanticDigest;

/// Why a cached decision could not be consumed.
#[derive(Debug)]
pub enum PermissionError {
    /// The dependency graph refused.
    Graph(GraphError),
    /// The ledger refused the source account.
    Ledger(LedgerError),
    /// `decision_key` is not a node of the current graph.
    DecisionMissing(String),
    /// A dependency names a source the ledger does not hold.
    UnknownSource(SourceId),
}

impl std::fmt::Display for PermissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "{error}"),
            Self::Ledger(error) => write!(f, "{error}"),
            Self::DecisionMissing(key) => write!(f, "decision node {key} is not in the graph"),
            Self::UnknownSource(id) => write!(f, "source {id} is not in the ledger"),
        }
    }
}

impl std::error::Error for PermissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::Ledger(error) => Some(error),
            Self::DecisionMissing(_) | Self::UnknownSource(_) => None,
        }
    }
}

impl From<GraphError> for PermissionError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

impl From<LedgerError> for PermissionError {
    fn from(error: LedgerError) -> Self {
        Self::Ledger(error)
    }
}

/// Permission to use a cached decision at the current coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct SourcePermission {
    /// Decision node that was consumed.
    pub decision_key: String,
    /// Classification after source standing is applied to the graph class.
    pub class: Revalidation,
    /// Whether the earlier fit and uncertainty still stand.
    pub authority: PriorAuthority,
    /// Licenses lost, in canonical order.
    pub withdrawals: Vec<Withdrawal>,
    /// Calibration at the decision's current coordinate.
    pub calibration: CalibrationState,
    /// Observations behind the decision's sources, each row once.
    pub repeated_use: MassAccount,
    /// Shared rows are licensed only by a valid reuse or dependence result.
    pub overlap_accounted: bool,
    /// The cached digest equals the current decision-node identity.
    pub cache_identity_matches: bool,
    /// The cache may be used as scientific permission.
    pub scientific_permission_retained: bool,
}

/// Revalidate a decision at consumption.
///
/// `ledger_bindings` are `(node key, result id)` pairs for ledger results the
/// graph names. `cached_identity` is the digest stored with the artifact.
/// Equality with the current node is recorded and does not grant permission.
///
/// # Errors
/// An invalid graph, a missing decision node, an unknown bound result, an
/// evidence id the ledger does not hold, or a ledger account that refuses.
pub fn consume_decision(
    decision_key: &str,
    previous: &CompositionGraph,
    current: &CompositionGraph,
    ledger: &EvidenceLedger,
    ledger_bindings: &[(&str, ResultId)],
    cached_identity: Option<SemanticDigest>,
) -> Result<SourcePermission, PermissionError> {
    let effects = ledger_effects(ledger, ledger_bindings)?;
    let classified = revalidate(previous, current, &effects)?;
    let claim = classified
        .claim(decision_key)
        .ok_or_else(|| PermissionError::DecisionMissing(decision_key.to_owned()))?;
    let node = current
        .node(decision_key)
        .ok_or_else(|| PermissionError::DecisionMissing(decision_key.to_owned()))?;
    let sources = evidence_along(current, decision_key);
    for id in &sources {
        if ledger.source(*id).is_none() {
            return Err(PermissionError::UnknownSource(*id));
        }
    }
    let repeated_use = ledger.mass_account(&sources)?;
    let (class, authority, withdrawals) = apply_source_standing(claim, ledger, &sources);
    let overlap_accounted = overlap_is_accounted(&repeated_use, ledger, ledger_bindings);
    let cache_identity_matches = cached_identity == Some(node.identity);
    let scientific_permission_retained = class == Revalidation::StillValid
        && authority == PriorAuthority::Retained
        && overlap_accounted;
    Ok(SourcePermission {
        decision_key: decision_key.to_owned(),
        class,
        authority,
        withdrawals,
        calibration: claim.calibration.state,
        repeated_use,
        overlap_accounted,
        cache_identity_matches,
        scientific_permission_retained,
    })
}

fn evidence_along(graph: &CompositionGraph, decision_key: &str) -> Vec<SourceId> {
    let mut ids = BTreeSet::new();
    if let Some(node) = graph.node(decision_key) {
        ids.extend(node.evidence.iter().copied());
    }
    for ancestor in graph.ancestors_of(decision_key) {
        if let Some(node) = graph.node(ancestor) {
            ids.extend(node.evidence.iter().copied());
        }
    }
    ids.into_iter().collect()
}

fn apply_source_standing(
    claim: &ClaimRevalidation,
    ledger: &EvidenceLedger,
    sources: &[SourceId],
) -> (Revalidation, PriorAuthority, Vec<Withdrawal>) {
    let mut class = claim.class;
    let mut authority = claim.authority;
    let mut withdrawals = claim.withdrawals.clone();
    for id in sources {
        let Some(source) = ledger.source(*id) else { continue };
        match &source.status {
            SourceStatus::Active => {}
            SourceStatus::Withdrawn { reason, .. } => {
                class = Revalidation::Withdrawn;
                authority = PriorAuthority::NotRetained;
                let withdrawal = Withdrawal {
                    kind: WithdrawalKind::Retired,
                    origin: claim.key.clone(),
                    via: None,
                    detail: reason.clone(),
                };
                if !withdrawals.contains(&withdrawal) {
                    withdrawals.push(withdrawal);
                }
            }
            SourceStatus::Superseded { .. }
            | SourceStatus::Replaced { .. }
            | SourceStatus::Updated { .. } => {
                if class == Revalidation::StillValid {
                    class = Revalidation::Recompute;
                }
                authority = PriorAuthority::NotRetained;
            }
        }
    }
    withdrawals.sort();
    (class, authority, withdrawals)
}

fn overlap_is_accounted(
    account: &MassAccount,
    ledger: &EvidenceLedger,
    bindings: &[(&str, ResultId)],
) -> bool {
    if !account.unknown_sources.is_empty() {
        return false;
    }
    if account.repeated_slots == 0 {
        return true;
    }
    bindings.iter().any(|(_, id)| {
        ledger.result(*id).is_some_and(|record| {
            record.status == ResultStatus::Valid
                && (record.data_reuse || record.dependence_declared)
        })
    })
}

#[cfg(test)]
mod tests {
    use antecedent_contract::composition_verification::{
        CalibrationDeclaration, CompositionGraph, CompositionNode, DependencyKind, NodeKind,
        SupportStanding, TrustStanding, WithdrawalKind,
    };
    use antecedent_contract::evidence_ledger::{
        CombinationRequest, EvidenceLedger, ObservationId, PoolingDependence, SourceDeclaration,
        SourceKind, SynthesisOperation,
    };
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_core::SemanticDigest;
    use antecedent_io_core::distribution_artifact::{
        DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
        DistributionProvenance, DistributionTrust, DrawAlignment,
    };
    use antecedent_io_core::quantity_wire::DistributionMeaningWire;
    use antecedent_io_core::{
        DesignVariableRole, DesignVariableSummary, EstimandFingerprint, PriorCatalog,
        PriorSourceMeta, PriorSourceRef, TargetDesign,
    };

    use super::consume_decision;
    use crate::composition_boundary::{
        BoundaryError, DecisionInput, EvidenceRelation, PairRelation, TrustEvidence,
        TrustRequirement, check_paired_draws,
    };
    use crate::decision_contract::{
        ActionKind, DecisionAction, DecisionContract, DecisionContractError, DecisionCriterion,
        StructuralPolicy, UtilityExpr,
    };
    use crate::decision_eval::{DecisionEvalError, MeanSource, evaluate_contract_on_means};
    use crate::preposterior::DecisionPrior;
    use crate::prior_signal::{
        CandidateSignalSource, NativeSignalFamily, PriorSignalRequest, PriorSourceInput,
        SourceToTarget, adapt_prior_to_signal,
    };

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::from_bytes([byte; 32])
    }

    fn row(key: &[u8]) -> ObservationId {
        ObservationId::from_key("rows", key)
    }

    fn observations(label: &str, keys: &[&[u8]]) -> SourceDeclaration {
        SourceDeclaration::new(label, SourceKind::ObservationSet)
            .with_observations(keys.iter().copied().map(row))
    }

    fn graph(
        nodes: Vec<CompositionNode>,
        edges: &[(&str, &str, DependencyKind)],
    ) -> CompositionGraph {
        let mut built = CompositionGraph::new();
        for node in nodes {
            built.add_node(node).expect("node");
        }
        for (down, up, kind) in edges {
            built.add_edge(down, up, *kind).expect("edge");
        }
        built
    }

    fn decision_on_result(
        evidence: Vec<antecedent_contract::evidence_ledger::SourceId>,
    ) -> CompositionGraph {
        graph(
            vec![
                CompositionNode::new("cohort", NodeKind::CohortSource, digest(1))
                    .with_evidence(evidence),
                CompositionNode::new("result", NodeKind::LedgerResult, digest(2)),
                CompositionNode::new("decision", NodeKind::Decision, digest(3)),
            ],
            &[
                ("result", "cohort", DependencyKind::Evidence),
                ("decision", "result", DependencyKind::Data),
            ],
        )
    }

    #[test]
    fn withdrawing_a_counted_source_removes_decision_permission() {
        let mut ledger = EvidenceLedger::new();
        let trial = ledger.add_source(observations("trial", &[b"r1", b"r2"])).unwrap();
        let registry = ledger.add_source(observations("registry", &[b"g1"])).unwrap();
        let result = ledger
            .combine(CombinationRequest {
                label: "pool".into(),
                operation: SynthesisOperation::StatisticalPooling {
                    sources: vec![trial, registry],
                    dependence: PoolingDependence::Independent,
                },
                replaces: None,
            })
            .unwrap();
        ledger.withdraw_source(trial, "retracted assay").unwrap();
        ledger.revalidate_downstream().unwrap();

        let scene = decision_on_result(vec![trial, registry]);
        let permission = consume_decision(
            "decision",
            &scene,
            &scene,
            &ledger,
            &[("result", result)],
            Some(digest(3)),
        )
        .unwrap();
        assert!(permission.cache_identity_matches);
        assert!(!permission.scientific_permission_retained);
        assert_eq!(permission.class, antecedent_core::recalc::Revalidation::Withdrawn);
        assert!(permission.withdrawals.iter().any(|w| w.kind == WithdrawalKind::Retired));
        assert_eq!(
            permission.authority,
            antecedent_contract::composition_verification::PriorAuthority::NotRetained
        );
    }

    #[test]
    fn cached_identity_does_not_keep_withdrawn_permission() {
        let mut ledger = EvidenceLedger::new();
        let trial = ledger.add_source(observations("trial", &[b"r1"])).unwrap();
        let registry = ledger.add_source(observations("registry", &[b"g1"])).unwrap();
        let result = ledger
            .combine(CombinationRequest {
                label: "pool".into(),
                operation: SynthesisOperation::StatisticalPooling {
                    sources: vec![trial, registry],
                    dependence: PoolingDependence::Independent,
                },
                replaces: None,
            })
            .unwrap();
        let scene = decision_on_result(vec![trial, registry]);
        let before = consume_decision(
            "decision",
            &scene,
            &scene,
            &ledger,
            &[("result", result)],
            Some(digest(3)),
        )
        .unwrap();
        assert!(before.scientific_permission_retained);
        assert!(before.cache_identity_matches);

        ledger.withdraw_source(registry, "site withdrew consent").unwrap();
        ledger.revalidate_downstream().unwrap();
        let after = consume_decision(
            "decision",
            &scene,
            &scene,
            &ledger,
            &[("result", result)],
            Some(digest(3)),
        )
        .unwrap();
        assert!(after.cache_identity_matches);
        assert!(!after.scientific_permission_retained);
        assert_eq!(after.class, antecedent_core::recalc::Revalidation::Withdrawn);
    }

    #[test]
    fn superseding_a_model_asks_for_recomputation_and_drops_authority() {
        let mut ledger = EvidenceLedger::new();
        let model = ledger
            .add_source(
                SourceDeclaration::new("fit", SourceKind::FittedModel)
                    .with_observations([row(b"m1")]),
            )
            .unwrap();
        let cohort = ledger.add_source(observations("cohort", &[b"d1"])).unwrap();
        let result = ledger
            .combine(CombinationRequest {
                label: "pool".into(),
                operation: SynthesisOperation::StatisticalPooling {
                    sources: vec![model, cohort],
                    dependence: PoolingDependence::Independent,
                },
                replaces: None,
            })
            .unwrap();
        ledger
            .supersede_source(
                model,
                SourceDeclaration::new("fit-next", SourceKind::FittedModel)
                    .with_observations([row(b"m2")]),
            )
            .unwrap();
        ledger.revalidate_downstream().unwrap();
        let scene = decision_on_result(vec![model, cohort]);
        let permission = consume_decision(
            "decision",
            &scene,
            &scene,
            &ledger,
            &[("result", result)],
            Some(digest(3)),
        )
        .unwrap();
        assert!(permission.cache_identity_matches);
        assert!(!permission.scientific_permission_retained);
        assert_eq!(permission.class, antecedent_core::recalc::Revalidation::Recompute);
        assert_eq!(
            permission.authority,
            antecedent_contract::composition_verification::PriorAuthority::NotRetained
        );
    }

    #[test]
    fn provider_trust_withdrawal_reaches_the_decision() {
        let provider = |trust| {
            CompositionNode::new("provider", NodeKind::ExternalProvider, digest(4))
                .with_trust(trust)
        };
        let previous = graph(
            vec![
                provider(TrustStanding::Held),
                CompositionNode::new("decision", NodeKind::Decision, digest(3)),
            ],
            &[("decision", "provider", DependencyKind::ProviderTrust)],
        );
        let current = graph(
            vec![
                provider(TrustStanding::Withdrawn),
                CompositionNode::new("decision", NodeKind::Decision, digest(3)),
            ],
            &[("decision", "provider", DependencyKind::ProviderTrust)],
        );
        let ledger = EvidenceLedger::new();
        let permission =
            consume_decision("decision", &previous, &current, &ledger, &[], Some(digest(3)))
                .unwrap();
        assert!(permission.cache_identity_matches);
        assert!(!permission.scientific_permission_retained);
        assert!(permission.withdrawals.iter().any(|w| w.kind == WithdrawalKind::ProviderTrust));
        assert_eq!(permission.class, antecedent_core::recalc::Revalidation::Withdrawn);
    }

    #[test]
    fn unsupported_coordinate_withdraws_support_at_consumption() {
        let target = |support: SupportStanding| {
            let node = CompositionNode::new("target", NodeKind::TargetPopulation, digest(5));
            match support {
                SupportStanding::Supported => node,
                SupportStanding::Unsupported { coordinates } => node.with_unsupported(coordinates),
            }
        };
        let edges = &[("decision", "target", DependencyKind::Support)];
        let previous = graph(
            vec![
                target(SupportStanding::Supported),
                CompositionNode::new("decision", NodeKind::Decision, digest(3)),
            ],
            edges,
        );
        let current = graph(
            vec![
                target(SupportStanding::Unsupported { coordinates: vec!["age>=80".into()] }),
                CompositionNode::new("decision", NodeKind::Decision, digest(3)),
            ],
            edges,
        );
        let permission = consume_decision(
            "decision",
            &previous,
            &current,
            &EvidenceLedger::new(),
            &[],
            Some(digest(3)),
        )
        .unwrap();
        assert!(!permission.scientific_permission_retained);
        assert!(permission.withdrawals.iter().any(|w| w.kind == WithdrawalKind::Support));
        assert_eq!(
            permission.calibration,
            antecedent_contract::composition_verification::CalibrationState::NotApplicable
        );
    }

    #[test]
    fn calibration_at_another_coordinate_is_withdrawn() {
        let record = digest(9);
        let coordinate = digest(8);
        let held = CalibrationDeclaration::Held { record, coordinate };
        let inherited =
            CalibrationDeclaration::Inherited { from: "result".into(), record, coordinate };
        let nodes = |decision_coordinate| {
            vec![
                CompositionNode::new("result", NodeKind::LedgerResult, digest(2))
                    .with_coordinate(coordinate)
                    .with_calibration(held.clone()),
                CompositionNode::new("decision", NodeKind::Decision, digest(3))
                    .with_coordinate(decision_coordinate)
                    .with_calibration(inherited.clone()),
            ]
        };
        let edges = &[
            ("decision", "result", DependencyKind::Data),
            ("decision", "result", DependencyKind::Calibration),
        ];
        let previous = graph(nodes(coordinate), edges);
        let current = graph(nodes(digest(7)), edges);
        let permission =
            consume_decision("decision", &previous, &current, &EvidenceLedger::new(), &[], None)
                .unwrap();
        assert!(!permission.cache_identity_matches);
        assert!(!permission.scientific_permission_retained);
        assert_eq!(permission.class, antecedent_core::recalc::Revalidation::Withdrawn);
        assert!(permission.withdrawals.iter().any(|w| w.kind == WithdrawalKind::Calibration));
        assert_eq!(
            permission.calibration,
            antecedent_contract::composition_verification::CalibrationState::Lost
        );
    }

    #[test]
    fn repeated_observation_use_counts_each_shared_row_once() {
        let mut ledger = EvidenceLedger::new();
        let left = ledger.add_source(observations("left", &[b"r1", b"r2"])).unwrap();
        let right = ledger.add_source(observations("right", &[b"r2", b"r3"])).unwrap();
        let result = ledger
            .combine(CombinationRequest {
                label: "again".into(),
                operation: SynthesisOperation::RepeatedObservationUse {
                    sources: vec![left, right],
                    stage: "decision".into(),
                },
                replaces: None,
            })
            .unwrap();
        let scene = decision_on_result(vec![left, right]);
        let permission = consume_decision(
            "decision",
            &scene,
            &scene,
            &ledger,
            &[("result", result)],
            Some(digest(3)),
        )
        .unwrap();
        assert!(permission.scientific_permission_retained);
        assert!(permission.overlap_accounted);
        assert_eq!(permission.repeated_use.declared_slots, 4);
        assert_eq!(permission.repeated_use.distinct_observations, 3);
        assert_eq!(permission.repeated_use.repeated_slots, 1);
        assert!(permission.repeated_use.unknown_sources.is_empty());
    }

    #[test]
    fn shared_rows_without_a_repeated_use_record_are_not_licensed() {
        let mut ledger = EvidenceLedger::new();
        let left = ledger.add_source(observations("left", &[b"r1", b"r2"])).unwrap();
        let right = ledger.add_source(observations("right", &[b"r2", b"r3"])).unwrap();
        let scene = decision_on_result(vec![left, right]);
        let permission =
            consume_decision("decision", &scene, &scene, &ledger, &[], Some(digest(3))).unwrap();
        assert_eq!(permission.class, antecedent_core::recalc::Revalidation::StillValid);
        assert_eq!(permission.repeated_use.repeated_slots, 1);
        assert!(!permission.overlap_accounted);
        assert!(!permission.scientific_permission_retained);
        assert!(permission.cache_identity_matches);
    }

    #[test]
    fn undeclared_observations_are_not_treated_as_independent() {
        let mut ledger = EvidenceLedger::new();
        let quiet =
            ledger.add_source(SourceDeclaration::new("quiet", SourceKind::ExternalClaim)).unwrap();
        let scene = decision_on_result(vec![quiet]);
        let permission = consume_decision("decision", &scene, &scene, &ledger, &[], None).unwrap();
        assert!(!permission.repeated_use.unknown_sources.is_empty());
        assert!(!permission.overlap_accounted);
        assert!(!permission.scientific_permission_retained);
        assert_eq!(permission.repeated_use.distinct_observations, 0);
    }

    #[test]
    fn conflict_shrinkage_moves_location_and_keeps_spread() {
        let catalog = PriorCatalog::from_sources(vec![
            PriorSourceRef::from_meta(
                PriorSourceMeta::new(
                    "A",
                    EstimandFingerprint::new("ate", "t", "y"),
                    "NonparametricallyIdentified",
                )
                .with_design(vec![
                    DesignVariableSummary::new("t", DesignVariableRole::Treatment),
                    DesignVariableSummary::new("y", DesignVariableRole::Outcome),
                ]),
            ),
            PriorSourceRef::from_meta(
                PriorSourceMeta::new(
                    "B",
                    EstimandFingerprint::new("ate", "t", "y"),
                    "NonparametricallyIdentified",
                )
                .with_design(vec![
                    DesignVariableSummary::new("t", DesignVariableRole::Treatment),
                    DesignVariableSummary::new("y", DesignVariableRole::Outcome),
                ]),
            ),
        ]);
        let target = TargetDesign::new(EstimandFingerprint::new("ate", "t", "y"), ["t", "y"]);
        let source = |id: &str, draws: &[f64], shrinkage: f64| PriorSourceInput {
            artifact_id: id.into(),
            source_population: "target".into(),
            draws: draws.to_vec(),
            weight: 1.0,
            prior_strength: 1.0,
            conflict_shrinkage: shrinkage,
            lineage: vec![id.into()],
            observation_ids: vec![format!("obs-{id}")],
            source_to_target: SourceToTarget::Identity,
        };
        let request = PriorSignalRequest {
            prior_id: "prior".into(),
            sources: vec![source("A", &[0.2, 0.4], 0.0), source("B", &[0.6, 0.8], 1.0)],
            signal: CandidateSignalSource::Native(NativeSignalFamily::Binomial),
            target_population: "target".into(),
            transport: None,
            candidate_observation_ids: vec!["future".into()],
            resolution: 4,
        };
        let adapted = adapt_prior_to_signal(&catalog, &target, &request).unwrap();
        let DecisionPrior::Draws(draws) = adapted.prior else {
            panic!("draws");
        };
        // Pooled mean is 0.5. B moves from 0.7 to 0.5 and keeps the gap 0.2.
        assert!((draws[2] - 0.4).abs() < 1e-12);
        assert!((draws[3] - 0.6).abs() < 1e-12);
        assert!((draws[3] - draws[2] - 0.2).abs() < 1e-12);
    }

    fn quantity(regime: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "y".into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: regime.into(),
            horizon: 0,
            functional_id: "mean".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    #[test]
    fn mean_source_refuses_duplicate_coordinates() {
        let contract = DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "wait".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![quantity("do(a=0)")],
                    utility: UtilityExpr::Input(0),
                },
                DecisionAction {
                    id: "treat".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![quantity("do(a=1)")],
                    utility: UtilityExpr::Input(0),
                },
            ],
            utility_units: "utility".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        };
        let source = MeanSource {
            coordinates: vec![quantity("do(a=0)"), quantity("do(a=0)")],
            means: vec![1.0, 2.0],
            provider_id: "lab".into(),
            snapshot_id: "snap".into(),
            causal_contract_id: "checked".into(),
            rng_id: "none:mean_grid".into(),
        };
        let error = evaluate_contract_on_means(&contract, &source).unwrap_err();
        let DecisionEvalError::Contract(DecisionContractError::InvalidDeclaration(message)) = error
        else {
            panic!("duplicate coordinate");
        };
        assert_eq!(message, "a mean source names each coordinate once");
    }

    fn one_column(snapshot: &str, n: usize) -> DistributionArtifact {
        let columns = vec![quantity("do(a=0)")];
        let identity = DistributionIdentity::new(
            DistributionMeaningWire::InterventionalPredictive,
            &columns,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: format!("study-{snapshot}"),
                provider_id: "lab".into(),
                rng_id: "deterministic_exact".into(),
                snapshot_id: snapshot.into(),
                causal_contract_id: "checked".into(),
            },
        )
        .unwrap();
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [n, 1],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::Unmeasured,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            vec![1.0; n],
        )
        .unwrap()
    }

    #[test]
    fn measured_joint_refuses_unequal_draw_counts() {
        let first = DecisionInput::from_distribution_artifact(
            "first",
            &one_column("snap-a", 4),
            &TrustEvidence::None,
            TrustRequirement::Unrestricted,
        )
        .unwrap();
        let mut short = one_column("snap-a", 4);
        let mut metadata = short.metadata().clone();
        metadata.shape[0] = 3;
        short = DistributionArtifact::new(metadata, short.draws()[..3].to_vec()).unwrap();
        let second = DecisionInput::from_distribution_artifact(
            "second",
            &short,
            &TrustEvidence::None,
            TrustRequirement::Unrestricted,
        )
        .unwrap();
        let relation = PairRelation {
            left: "first".into(),
            right: "second".into(),
            relation: EvidenceRelation::IndependentSources,
            route: None,
        };
        let error = check_paired_draws(&[first, second], &[relation]).unwrap_err();
        assert_eq!(error, BoundaryError::InvalidInput("paired draw count mismatch"));
    }
}
