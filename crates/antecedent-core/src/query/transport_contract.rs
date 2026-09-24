//! Transport theorem scopes, typed outcomes, and stage-specific support.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::VariableId;

/// Named transport theorem family. Completeness is scoped per family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TheoremFamily {
    /// Classical single-source sID (Pearl & Bareinboim general transportability).
    ///
    /// Completeness applies only to that paper's experimental-information
    /// family, not to every catalog the API can represent.
    ClassicalSid,
    /// Classical multi-source meta-transportability with full source experiments.
    MetaSid,
    /// Sound search over a finite supplied catalog. Not a completeness theorem.
    FiniteCatalogSearch,
    /// Later z- / limited-experiment contracts. Named so they cannot inherit
    /// classical sID completeness by accident.
    LimitedExperiment,
    /// Single-source z-transportability (experiments on a declared controllable set).
    ZTransportability,
}

impl TheoremFamily {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClassicalSid => "classical_sid",
            Self::MetaSid => "meta_sid",
            Self::FiniteCatalogSearch => "finite_catalog_search",
            Self::LimitedExperiment => "limited_experiment",
            Self::ZTransportability => "z_transportability",
        }
    }
}

/// Pinned citation and version for a theorem family.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TheoremReference {
    /// Bibliographic or arXiv identifier.
    pub citation: Arc<str>,
    /// Implementation-pinned version tag.
    pub version: Arc<str>,
}

/// Graph assumptions the theorem takes as given.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GraphAssumptionSet {
    /// Semi-Markovian ADMG with explicit selection targets.
    SemiMarkovianSelectionAdmg,
}

impl GraphAssumptionSet {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SemiMarkovianSelectionAdmg => "semi_markovian_selection_admg",
        }
    }
}

/// Experiments the theorem treats as available.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExperimentFamily {
    /// The theorem's own experimental-information setting.
    TheoremExperiments,
    /// Only the finite catalog the caller supplied.
    SuppliedCatalog,
}

impl ExperimentFamily {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TheoremExperiments => "theorem_experiments",
            Self::SuppliedCatalog => "supplied_catalog",
        }
    }
}

/// Distributional family the theorem is stated over.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TransportDistributionFamily {
    /// Positive discrete / density laws as in classical sID.
    PositiveLaws,
    /// Finite nonnegative tables supplied as exact laws (T4).
    FiniteExactTables,
}

impl TransportDistributionFamily {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PositiveLaws => "positive_laws",
            Self::FiniteExactTables => "finite_exact_tables",
        }
    }
}

/// Queries the theorem is stated for.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TransportQueryScope {
    /// Interventional response in a named target population.
    TargetInterventionalResponse,
}

impl TransportQueryScope {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TargetInterventionalResponse => "target_interventional_response",
        }
    }
}

/// What a successful run of the family is allowed to claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OutcomeGuarantee {
    /// Complete for the stated mathematical input family only.
    CompleteForStatedFamily,
    /// Sound and incomplete. [`TransportOutcomeKind::NotCertified`] is not impossibility.
    SoundIncomplete,
}

impl OutcomeGuarantee {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CompleteForStatedFamily => "complete_for_stated_family",
            Self::SoundIncomplete => "sound_incomplete",
        }
    }
}

/// Implemented computation limits. Exhaustion is never an impossibility claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ComputationLimits {
    /// Maximum S-admissible standardizer-candidate count before fail-closed.
    pub max_standardizer_candidates: u32,
    /// Whether multi-node c-component recursion is implemented.
    pub multi_node_c_component_recursion: bool,
}

/// Exact theorem scope for one transport family.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TheoremScope {
    /// Family this record describes.
    pub family: TheoremFamily,
    /// Pinned reference.
    pub reference: TheoremReference,
    /// Graph assumptions.
    pub graph_assumptions: GraphAssumptionSet,
    /// Variables treated as observed.
    pub observed: Arc<[VariableId]>,
    /// Experiments the family treats as given.
    pub allowed_experiments: ExperimentFamily,
    /// Distributional family.
    pub distribution_family: TransportDistributionFamily,
    /// Query scope.
    pub query_scope: TransportQueryScope,
    /// Outcome guarantees.
    pub outcome_guarantees: OutcomeGuarantee,
    /// Implemented computation limits.
    pub computation_limits: ComputationLimits,
}

impl TheoremScope {
    /// Sound subset used by the conservative identifier: no multi-node
    /// c-component recursion. Completeness is not claimed.
    #[must_use]
    pub fn classical_sid() -> Self {
        Self {
            family: TheoremFamily::ClassicalSid,
            reference: TheoremReference {
                citation: Arc::from(
                    "Bareinboim & Pearl, A General Algorithm for Deciding Transportability of Experimental Results, arXiv:1312.7485v1",
                ),
                version: Arc::from("classical-sid-t1"),
            },
            graph_assumptions: GraphAssumptionSet::SemiMarkovianSelectionAdmg,
            observed: Arc::from([]),
            allowed_experiments: ExperimentFamily::TheoremExperiments,
            distribution_family: TransportDistributionFamily::PositiveLaws,
            query_scope: TransportQueryScope::TargetInterventionalResponse,
            outcome_guarantees: OutcomeGuarantee::SoundIncomplete,
            computation_limits: ComputationLimits {
                max_standardizer_candidates: 20,
                multi_node_c_component_recursion: false,
            },
        }
    }

    /// Classical single-source family after Figure-5 recursion (T3). Completeness
    /// applies only to the paper's experimental-information setting.
    #[must_use]
    pub fn classical_sid_complete() -> Self {
        Self {
            family: TheoremFamily::ClassicalSid,
            reference: TheoremReference {
                citation: Arc::from(
                    "Bareinboim & Pearl, A General Algorithm for Deciding Transportability of Experimental Results, arXiv:1312.7485v1",
                ),
                version: Arc::from("classical-sid-t3"),
            },
            graph_assumptions: GraphAssumptionSet::SemiMarkovianSelectionAdmg,
            observed: Arc::from([]),
            allowed_experiments: ExperimentFamily::TheoremExperiments,
            distribution_family: TransportDistributionFamily::PositiveLaws,
            query_scope: TransportQueryScope::TargetInterventionalResponse,
            outcome_guarantees: OutcomeGuarantee::CompleteForStatedFamily,
            computation_limits: ComputationLimits {
                max_standardizer_candidates: 20,
                multi_node_c_component_recursion: true,
            },
        }
    }

    /// Bounded search over a supplied catalog. Missing a factor is not a proof
    /// that no alternative catalog-supported formula exists.
    #[must_use]
    pub fn finite_catalog_search() -> Self {
        Self {
            family: TheoremFamily::FiniteCatalogSearch,
            reference: TheoremReference {
                citation: Arc::from(
                    "Bareinboim & Pearl, A General Algorithm for Deciding Transportability of Experimental Results, arXiv:1312.7485v1; finite-catalog search is implemented, not a completeness theorem",
                ),
                version: Arc::from("finite-catalog-search-bounded"),
            },
            graph_assumptions: GraphAssumptionSet::SemiMarkovianSelectionAdmg,
            observed: Arc::from([]),
            allowed_experiments: ExperimentFamily::SuppliedCatalog,
            distribution_family: TransportDistributionFamily::FiniteExactTables,
            query_scope: TransportQueryScope::TargetInterventionalResponse,
            outcome_guarantees: OutcomeGuarantee::SoundIncomplete,
            computation_limits: ComputationLimits {
                max_standardizer_candidates: 20,
                multi_node_c_component_recursion: true,
            },
        }
    }

    /// `μsID` Figure 5: complete only in the unrestricted source-experiment setting.
    #[must_use]
    pub fn meta_sid_complete() -> Self {
        let mut scope = Self::classical_sid_complete();
        scope.family = TheoremFamily::MetaSid;
        scope.reference = TheoremReference {
            citation: Arc::from(
                "Bareinboim & Pearl (2013), Meta-Transportability of Causal Effects, PMLR 31:135-143, Figure 5, Theorems 3-5",
            ),
            version: Arc::from("meta-sid-pmlr31-2013-figure5-v1"),
        };
        scope
    }

    /// Bounded single-source z-transportability research scope.
    ///
    /// The current graph-specific implementation is sound but incomplete even
    /// within six observed and two controllable variables. Evidence binding
    /// is a separate execution step.
    #[must_use]
    pub fn z_transportability() -> Self {
        Self {
            family: TheoremFamily::ZTransportability,
            reference: TheoremReference {
                citation: Arc::from(
                    "Lee & Honavar, Causal Transportability of Experiments on Controllable Subsets of Variables: z-Transportability, UAI 2013, arXiv:1309.6842",
                ),
                version: Arc::from("lee-honavar-sidz-uai2013-bounded-v1"),
            },
            graph_assumptions: GraphAssumptionSet::SemiMarkovianSelectionAdmg,
            observed: Arc::from([]),
            allowed_experiments: ExperimentFamily::TheoremExperiments,
            distribution_family: TransportDistributionFamily::FiniteExactTables,
            query_scope: TransportQueryScope::TargetInterventionalResponse,
            // Do not advertise theorem completeness until the bounded sIDz
            // search and its independent proof checker are wired together.
            outcome_guarantees: OutcomeGuarantee::SoundIncomplete,
            computation_limits: ComputationLimits {
                max_standardizer_candidates: 20,
                multi_node_c_component_recursion: true,
            },
        }
    }

    /// Durable inspect token used by exact-law preparation.
    #[must_use]
    pub fn exact_law_inspect_label() -> &'static str {
        "classical_single_source_all_experiments_v1; finite_catalog_search_bounded"
    }

    /// Durable inspect token used by empirical-table preparation.
    #[must_use]
    pub fn statistical_table_inspect_label() -> &'static str {
        "classical_single_source_all_experiments_v1; empirical_table_plugin_iid"
    }
}

/// Optional factor or graph location attached to a [`TransportOutcome`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportLocation {
    /// Stable factor id when the outcome names one leaf or kernel.
    pub factor: Option<Arc<str>>,
    /// Graph / selection nodes witnessing the outcome.
    pub graph_nodes: Arc<[VariableId]>,
}

impl TransportLocation {
    /// Location that names only graph nodes.
    #[must_use]
    pub fn nodes(nodes: impl Into<Arc<[VariableId]>>) -> Self {
        Self { factor: None, graph_nodes: nodes.into() }
    }
}

/// Stable kind of a transport result. Callers match this; they must not parse prose.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TransportOutcomeKind {
    /// A sound implemented rule produced a formula.
    Identified,
    /// A checked impossibility witness for the named theorem family.
    ProvenNonTransportable,
    /// No implemented sound rule applies. Historical `NotCertified` meaning.
    NotCertified,
    /// Query, diagram, or catalog failed validation.
    InvalidInput,
    /// A required available regime or provider input is absent.
    MissingEvidence,
    /// A statistical provider required by a certified factor is absent.
    MissingProvider,
    /// The formula is identified but this evaluator does not run it.
    UnsupportedEvaluator,
    /// A stage-specific support coordinate failed.
    SupportFailure,
    /// Evaluation failed numerically after a certified formula.
    NumericalFailure,
    /// Step, memory, recursion, or cancellation budget exhausted.
    BudgetCancel,
}

impl TransportOutcomeKind {
    /// Every stable variant, in declaration order.
    pub const ALL: [Self; 10] = [
        Self::Identified,
        Self::ProvenNonTransportable,
        Self::NotCertified,
        Self::InvalidInput,
        Self::MissingEvidence,
        Self::MissingProvider,
        Self::UnsupportedEvaluator,
        Self::SupportFailure,
        Self::NumericalFailure,
        Self::BudgetCancel,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identified => "identified",
            Self::ProvenNonTransportable => "proven_non_transportable",
            Self::NotCertified => "not_certified",
            Self::InvalidInput => "invalid_input",
            Self::MissingEvidence => "missing_evidence",
            Self::MissingProvider => "missing_provider",
            Self::UnsupportedEvaluator => "unsupported_evaluator",
            Self::SupportFailure => "support_failure",
            Self::NumericalFailure => "numerical_failure",
            Self::BudgetCancel => "budget_cancel",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }
}

/// Typed transport result. Callers match [`Self::kind`]; they must not parse prose.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportOutcome {
    /// Stable variant.
    pub kind: TransportOutcomeKind,
    /// Stable reason id. Not a display string.
    pub reason: Arc<str>,
    /// Optional factor / graph location.
    pub location: Option<TransportLocation>,
}

impl TransportOutcome {
    /// Construct an outcome.
    #[must_use]
    pub fn new(
        kind: TransportOutcomeKind,
        reason: impl Into<Arc<str>>,
        location: Option<TransportLocation>,
    ) -> Self {
        Self { kind, reason: reason.into(), location }
    }

    /// Identified by a named rule.
    #[must_use]
    pub fn identified(reason: impl Into<Arc<str>>) -> Self {
        Self::new(TransportOutcomeKind::Identified, reason, None)
    }

    /// Verified impossibility witness. Distinct from [`Self::not_certified`].
    #[must_use]
    pub fn proven_non_transportable(
        reason: impl Into<Arc<str>>,
        location: impl Into<Arc<[VariableId]>>,
    ) -> Self {
        Self::new(
            TransportOutcomeKind::ProvenNonTransportable,
            reason,
            Some(TransportLocation::nodes(location)),
        )
    }

    /// No implemented sound rule. Historical `NotCertified` meaning.
    #[must_use]
    pub fn not_certified(
        reason: impl Into<Arc<str>>,
        location: impl Into<Arc<[VariableId]>>,
    ) -> Self {
        Self::new(
            TransportOutcomeKind::NotCertified,
            reason,
            Some(TransportLocation::nodes(location)),
        )
    }

    /// Required available evidence is absent.
    #[must_use]
    pub fn missing_evidence(
        reason: impl Into<Arc<str>>,
        location: impl Into<Arc<[VariableId]>>,
    ) -> Self {
        Self::new(
            TransportOutcomeKind::MissingEvidence,
            reason,
            Some(TransportLocation::nodes(location)),
        )
    }

    /// A certified leaf has no supplied law and no bound sample.
    #[must_use]
    pub fn missing_provider(reason: impl Into<Arc<str>>) -> Self {
        Self::new(TransportOutcomeKind::MissingProvider, reason, None)
    }
}

/// Identify-stage support coordinate. Not an `analyze` capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportIdentifySupport {
    /// Graph class (`admg`, …).
    pub graph_class: Arc<str>,
    /// Evidence setting (`classical_sid`, `supplied_catalog`, …).
    pub evidence_setting: Arc<str>,
    /// Target functional (`mean_curve`, …).
    pub target_functional: Arc<str>,
    /// Observation contract (`complete`, …).
    pub observation_contract: Arc<str>,
}

/// Evaluate-stage support coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportEvaluateSupport {
    /// Evaluator id (`trial_ipw`, `exact_table`, …).
    pub evaluator: Arc<str>,
    /// Evidence setting the evaluator consumes.
    pub evidence_setting: Arc<str>,
    /// Target functional.
    pub target_functional: Arc<str>,
}

/// Uncertainty-stage support coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportUncertaintySupport {
    /// Uncertainty method (`delta_method`, `none`, …).
    pub uncertainty_method: Arc<str>,
    /// Evaluator the method attaches to.
    pub evaluator: Arc<str>,
    /// Observation contract.
    pub observation_contract: Arc<str>,
}

/// Stage-specific transport support. Do not collapse these into one analyze cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransportSupportCoordinate {
    /// Identification support.
    Identify(TransportIdentifySupport),
    /// Evaluation support.
    Evaluate(TransportEvaluateSupport),
    /// Uncertainty support.
    Uncertainty(TransportUncertaintySupport),
}

impl TransportSupportCoordinate {
    /// Stage name (`identify`, `evaluate`, `uncertainty`).
    #[must_use]
    pub const fn stage(&self) -> &'static str {
        match self {
            Self::Identify(_) => "identify",
            Self::Evaluate(_) => "evaluate",
            Self::Uncertainty(_) => "uncertainty",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_outcome_kind_round_trips_by_name() {
        for kind in TransportOutcomeKind::ALL {
            let parsed = TransportOutcomeKind::from_name(kind.as_str())
                .unwrap_or_else(|| panic!("round-trip failed for {}", kind.as_str()));
            assert_eq!(parsed, kind);
        }
        assert!(TransportOutcomeKind::from_name("not_implemented").is_none());
        assert_eq!(TransportOutcomeKind::ALL.len(), 10);
    }

    #[test]
    fn missing_experiment_and_impossibility_witness_are_different_variants() {
        let missing = TransportOutcome::missing_evidence(
            "transport.source_experiment_missing",
            [VariableId::from_raw(0)],
        );
        let impossible = TransportOutcome::proven_non_transportable(
            "transport.sid.impossibility_witness",
            [VariableId::from_raw(0)],
        );
        assert_ne!(missing.kind, impossible.kind);
        assert_eq!(missing.kind, TransportOutcomeKind::MissingEvidence);
        assert_eq!(impossible.kind, TransportOutcomeKind::ProvenNonTransportable);
        assert_eq!(missing.kind.as_str(), "missing_evidence");
        assert_eq!(impossible.kind.as_str(), "proven_non_transportable");
        let provider = TransportOutcome::missing_provider("transport_missing_provider");
        assert_eq!(provider.kind, TransportOutcomeKind::MissingProvider);
        assert_eq!(provider.kind.as_str(), "missing_provider");
    }

    #[test]
    fn classical_sid_scope_does_not_claim_catalog_completeness() {
        let scope = TheoremScope::classical_sid();
        assert_eq!(scope.family, TheoremFamily::ClassicalSid);
        assert_eq!(scope.outcome_guarantees, OutcomeGuarantee::SoundIncomplete);
        assert!(!scope.computation_limits.multi_node_c_component_recursion);
        assert_ne!(scope.family.as_str(), TheoremFamily::FiniteCatalogSearch.as_str());
    }

    #[test]
    fn classical_complete_and_catalog_search_are_distinct_families() {
        let complete = TheoremScope::classical_sid_complete();
        let catalog = TheoremScope::finite_catalog_search();
        assert!(complete.computation_limits.multi_node_c_component_recursion);
        assert_eq!(complete.outcome_guarantees, OutcomeGuarantee::CompleteForStatedFamily);
        assert_eq!(complete.family, TheoremFamily::ClassicalSid);
        assert_eq!(catalog.family, TheoremFamily::FiniteCatalogSearch);
        assert_eq!(catalog.outcome_guarantees, OutcomeGuarantee::SoundIncomplete);
        assert_eq!(catalog.allowed_experiments, ExperimentFamily::SuppliedCatalog);
        assert_eq!(
            TheoremScope::exact_law_inspect_label(),
            "classical_single_source_all_experiments_v1; finite_catalog_search_bounded"
        );
    }

    #[test]
    fn support_coordinates_are_stage_specific() {
        let identify = TransportSupportCoordinate::Identify(TransportIdentifySupport {
            graph_class: Arc::from("admg"),
            evidence_setting: Arc::from("classical_sid"),
            target_functional: Arc::from("mean_curve"),
            observation_contract: Arc::from("complete"),
        });
        let evaluate = TransportSupportCoordinate::Evaluate(TransportEvaluateSupport {
            evaluator: Arc::from("trial_ipw"),
            evidence_setting: Arc::from("classical_sid"),
            target_functional: Arc::from("mean_curve"),
        });
        assert_eq!(identify.stage(), "identify");
        assert_eq!(evaluate.stage(), "evaluate");
        assert_ne!(identify.stage(), "analyze");
    }
}
