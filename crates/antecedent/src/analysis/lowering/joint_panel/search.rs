//! Lower a program search over a fitted joint panel response, or a held-out selection among
//! randomized or instrumented candidate programs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, FrozenFitReuse, Operator, OperatorDeclaration, OperatorEffect,
    OperatorGraph, OperatorPorts, OperatorRole, PhysicalEffect, PhysicalOperator, ScoreFrequentist,
    ScoreModel, SemanticDigest,
};

use super::design::{physical_read_and_fit, read_and_fit, refuse};
use crate::analysis::joint_panel::compiled::support_status;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, JointPanelLoweredParts,
    JointPanelRoute, LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget,
    ProgramIdentityScope, ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::support::CellStatus;

/// Domain tag of the question a search or selection program answers.
const QUESTION_TAG: &str = "antecedent.lowering.joint_panel.search.v1";
/// What a search program evaluates instead of one quoted design matrix.
const SEARCH_DESIGN: &str =
    "candidate programs evaluated with the frozen fit of the parent response";
/// What a selection program fits instead of one quoted design matrix.
const SELECTION_DESIGN: &str =
    "one fit per training sample and resample, and one on the held-out units";

/// The search or selection a program-search program compiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SearchOperation {
    /// Exhaustive search over a finite program domain of a fitted response.
    Finite,
    /// Bounded best-found search over continuous controls, evaluated with the frozen fit.
    Continuous,
    /// Global search with a replayable numerical certificate.
    Certified,
    /// Held-out selection among whole-program randomized candidates.
    RandomizedSelection,
    /// Held-out selection of an instrumented dose program.
    InstrumentedSelection,
}

impl SearchOperation {
    /// Stable `snake_case` name.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Finite => "finite",
            Self::Continuous => "continuous",
            Self::Certified => "certified",
            Self::RandomizedSelection => "randomized_selection",
            Self::InstrumentedSelection => "instrumented_selection",
        }
    }

    /// The route a program of this operation is lowered for.
    pub(crate) const fn route(self) -> JointPanelRoute {
        match self {
            Self::Finite | Self::Continuous | Self::Certified => JointPanelRoute::ProgramSearch,
            Self::RandomizedSelection | Self::InstrumentedSelection => {
                JointPanelRoute::ProgramSelection
            }
        }
    }

    /// Route record plan id of a program of this operation.
    const fn plan_id(self) -> &'static str {
        match self {
            Self::Finite => "joint_panel_program_search.finite",
            Self::Continuous => "joint_panel_program_search.continuous",
            Self::Certified => "joint_panel_program_search.certified",
            Self::RandomizedSelection => "joint_panel_program_selection.randomized",
            Self::InstrumentedSelection => "joint_panel_program_selection.instrumented",
        }
    }

    /// Whether the operation reads its own training and held-out samples.
    const fn selection(self) -> bool {
        matches!(self, Self::RandomizedSelection | Self::InstrumentedSelection)
    }

    /// Whether the operation proposes programs outside the parent's declared worlds.
    const fn proposes(self) -> bool {
        matches!(self, Self::Continuous | Self::Certified)
    }
}

/// Lower the program of a compiled search or selection (`operation.route()`).
///
/// A search over a fitted response is a child of that response's program: `parent` is its
/// identity, cited as the evidence source of a chained composition. A selection that reads
/// its own training and held-out samples has no parent. The six layers:
///
/// 1. the decision the operation answers, over the candidate programs;
/// 2. identification carried from the parent's scientific scope, or the selection's
///    randomized or instrumented design;
/// 3. the realization of the response representation the search reads (point means or the
///    aligned law) or of the held-out value estimate;
/// 4. the parent program and `source`, with `decision`: its carrier (`Mean` or
///    `DecisionDraws`), the candidate programs as actions and the budget and restrictions as
///    constraints;
/// 5. the read of `source_bytes` and the search or selection scan;
/// 6. the decision claim.
///
/// `source` is the identity of the evidence the operation reads: the retained decision source
/// of the parent fit for a search, the training and held-out samples for a selection.
///
/// A search evaluates the parent's retained response exactly: its program states no
/// uncertainty of its own, and its answer is conditional on the parent program. A selection
/// states the sampling law of its held-out value estimate, conditional on its declared design.
///
/// # Errors
/// A program layer the planner refuses to declare.
pub(crate) fn lower(
    operation: SearchOperation,
    parent: Option<SemanticDigest>,
    source: SemanticDigest,
    source_bytes: u64,
    decision: DecisionProgram,
) -> Result<LoweredProgram, CausalError> {
    let route = operation.route();
    let selection = operation.selection();
    let status = support_status(route);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: None,
            query: question(operation, parent, source, &decision),
        },
        plan_id: Arc::from(operation.plan_id()),
        data: if selection { DataClassification::Tabular } else { DataClassification::Panel },
        graph_class: if selection { "design_native" } else { "joint_panel_model" },
        query_variables: Arc::from([]),
        source_digest: Some(*source.as_bytes()),
        support_coordinate: None,
        supported: status == CellStatus::Supported,
        support_status: Some(status),
    };
    // A search is conditional on the parent's fitted response; a selection on its declared
    // design, which its source digest binds.
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional {
            model: *parent.unwrap_or(source).as_bytes(),
        },
    ));
    let design =
        DesignShape::not_design_based(if selection { SELECTION_DESIGN } else { SEARCH_DESIGN });
    let statistical = statistical_program(operation, &decision, design)?;
    let mut sources = Vec::with_capacity(2);
    if let Some(parent) = parent {
        sources.push(EvidenceSourceRef {
            role: EvidenceRole::ChainSource,
            digest: Some(*parent.as_bytes()),
        });
    }
    sources
        .push(EvidenceSourceRef { role: source_role(operation), digest: Some(*source.as_bytes()) });
    // A search reads the parent's response and then decides; a selection selects on its
    // training sample and then evaluates the selected program on its held-out sample.
    let evidence = EvidenceDecisionProgram {
        sources: sources.into(),
        composition: Composition::Chained { stages: 2 },
        decision: Some(decision),
        planning: None,
        chain: None,
    };
    let physical =
        PhysicalLayer { operators: physical_operators(operation, source_bytes)?.into(), design };
    let claim = ClaimDemand {
        uncertainty: if selection {
            UncertaintyReading::Sampling
        } else {
            UncertaintyReading::None
        },
        kind: Some(ClaimKind::PolicyValue),
    };
    Ok(JointPanelLoweredParts {
        route,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

/// Digest of the question the program answers: the operation, the parent program, the source
/// and the declared decision contract.
fn question(
    operation: SearchOperation,
    parent: Option<SemanticDigest>,
    source: SemanticDigest,
    decision: &DecisionProgram,
) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(QUESTION_TAG);
    writer.text(operation.as_str());
    match parent {
        Some(parent) => writer.word(1).bytes(parent.as_bytes()),
        None => writer.word(0),
    };
    writer.bytes(source.as_bytes()).opt_text(decision.contract_identity());
    writer.finish()
}

/// What the operation's source contributes: the retained source of the parent fit for a
/// search, randomized units for a randomized selection, instrumented observational units for
/// an instrumented selection.
const fn source_role(operation: SearchOperation) -> EvidenceRole {
    match operation {
        SearchOperation::Finite | SearchOperation::Continuous | SearchOperation::Certified => {
            EvidenceRole::ChainSource
        }
        SearchOperation::RandomizedSelection => EvidenceRole::Trial,
        SearchOperation::InstrumentedSelection => EvidenceRole::Observations,
    }
}

/// Layer three. A search reuses the parent's frozen fit, and a continuous or certified search
/// also names the proposed programs as intervention worlds; it evaluates them exactly. A
/// selection reads and fits its own samples, and its held-out value is a frequentist estimate.
fn statistical_program(
    operation: SearchOperation,
    decision: &DecisionProgram,
    design: DesignShape,
) -> Result<StatisticalProgram, CausalError> {
    let (realization, operators) = if operation.selection() {
        let score = ScoreModel::new(match operation {
            SearchOperation::RandomizedSelection => "heldout_whole_program_value",
            _ => "heldout_structural_dose_contrast",
        })
        .map_err(refuse)?;
        (Realization::Frequentist(ScoreFrequentist::new(score)), Vec::from(read_and_fit()?))
    } else {
        let mut operators = vec![
            Operator::new(
                OperatorDeclaration::new(
                    OperatorEffect::ReuseFrozenFit,
                    OperatorGraph::Execution,
                    OperatorPorts::new([OperatorRole::FitHandle], [OperatorRole::FitHandle]),
                )
                .with_frozen_fit_reuse(FrozenFitReuse::Allowed),
            )
            .map_err(refuse)?,
        ];
        if operation.proposes() {
            operators.push(
                Operator::new(OperatorDeclaration::new(
                    OperatorEffect::Intervene,
                    OperatorGraph::Causal,
                    OperatorPorts::new([OperatorRole::FitHandle], [OperatorRole::WorldOrSchedule]),
                ))
                .map_err(refuse)?,
            );
        }
        let evaluator = format!(
            "joint_panel_program_search_{}_{}_v1",
            operation.as_str(),
            decision.carrier().as_str()
        );
        (Realization::ExactPoint { evaluator: Arc::from(evaluator) }, operators)
    };
    Ok(StatisticalProgram {
        identifier: None,
        estimator: None,
        realization,
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: operators.into(),
    })
}

/// Layer five. A search holds the parent's retained source of `source_bytes` and, for
/// proposed programs, intervenes on each; a selection reads its samples and fits them.
fn physical_operators(
    operation: SearchOperation,
    source_bytes: u64,
) -> Result<Vec<PhysicalOperator>, CausalError> {
    if operation.selection() {
        return Ok(Vec::from(physical_read_and_fit(source_bytes)?));
    }
    let mut operators = vec![
        PhysicalOperator::declare(PhysicalEffect::ReuseFrozenFit)
            .estimates(ByteEstimates::absent().with_retained(source_bytes))
            .build()
            .map_err(refuse)?,
    ];
    if operation.proposes() {
        operators
            .push(PhysicalOperator::declare(PhysicalEffect::Intervene).build().map_err(refuse)?);
    }
    Ok(operators)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::SemanticDigest;

    use super::{SearchOperation, lower};
    use crate::planner::lowered::{CarrierDemand, Composition, DecisionProgram, EvidenceRole};

    fn decision(carrier: CarrierDemand, contract: &str) -> DecisionProgram {
        DecisionProgram {
            carrier,
            utility: DecisionProgram::utility_form(None),
            actions: 4,
            constraints: 1,
            chance_threshold: 0.0,
            utility_unit: Some(Arc::from("response")),
            contract_identity: Some(Arc::from(contract)),
            information_value: None,
            scenario_source: None,
        }
    }

    #[test]
    fn a_search_is_a_chained_child_of_its_parent_and_its_contract_enters_the_identity() {
        let parent = SemanticDigest::from_bytes([1; 32]);
        let source = SemanticDigest::from_bytes([2; 32]);
        let search = |operation, bytes, carrier, contract| {
            lower(operation, Some(parent), source, bytes, decision(carrier, contract)).unwrap()
        };
        let program = search(SearchOperation::Finite, 64, CarrierDemand::Mean, "a");
        let evidence = program.evidence();
        assert_eq!(evidence.composition(), Composition::Chained { stages: 2 });
        assert_eq!(evidence.sources()[0].role(), EvidenceRole::ChainSource);
        assert_eq!(evidence.sources()[0].digest(), Some(*parent.as_bytes()));
        assert_eq!(evidence.decision().map(DecisionProgram::actions), Some(4));
        let quoted = search(SearchOperation::Finite, 128, CarrierDemand::Mean, "a");
        assert_eq!(program.identity(), quoted.identity(), "the byte quote is physical only");
        for other in [
            search(SearchOperation::Finite, 64, CarrierDemand::Mean, "b"),
            search(SearchOperation::Finite, 64, CarrierDemand::DecisionDraws, "a"),
            search(SearchOperation::Continuous, 64, CarrierDemand::Mean, "a"),
        ] {
            assert_ne!(other.identity(), program.identity());
        }
    }

    #[test]
    fn a_selection_reads_its_own_samples_without_a_parent() {
        let source = SemanticDigest::from_bytes([3; 32]);
        for (operation, role) in [
            (SearchOperation::RandomizedSelection, EvidenceRole::Trial),
            (SearchOperation::InstrumentedSelection, EvidenceRole::Observations),
        ] {
            let program =
                lower(operation, None, source, 64, decision(CarrierDemand::Mean, "a")).unwrap();
            let sources = program.evidence().sources();
            assert_eq!(sources.len(), 1);
            assert_eq!(sources[0].role(), role);
            assert_eq!(program.cell(), operation.route().cell());
        }
    }
}
