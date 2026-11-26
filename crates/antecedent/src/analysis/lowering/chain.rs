//! Propagation-chain lowering: every compiled chain carries a lowered program.
//!
//! A chain compiles through its own entry
//! (`compile_propagation_chain`), not through `Study::compile`, because it binds
//! no study: its sources are declared draws, intervals and points. It lowers to
//! the same six layers:
//!
//! 1. the chain's output as a typed functional keyed by the digest of the
//!    declared chain, so two chains lower to different programs;
//! 2. the dependence contract as the identification of overlap: which rows,
//!    clusters, models, priors, disturbances and periods each stage inherits and
//!    what every combining stage shares. Nothing is identified from
//!    observations;
//! 3. the statistical program from the law the output carries (a sampling law, a
//!    posterior, or an exact enclosure or point) and one model slot per stage;
//! 4. the evidence: one source per chain source, the stages in sequence, and the
//!    declared chain itself (the record identity the execution seals);
//! 5. one physical operator per stage and, for a combined decision, one scoring
//!    operator, with their output bytes admitted;
//! 6. the claim demand.
//!
//! Lowering computes no number and grants no support: a chain value is a labelled
//! point or uncalibrated spread, and an interval of a composed quantity stays
//! not constructed without a measured coverage record.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::propagation_chain::{
    ChainPlan, ChainRefusal, FIX_CHAIN_DECLARATION, LawMeaning, NodeRepresentation, StageKind,
};
use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, LikelihoodModel, OrdinaryBayesianLikelihood, PhysicalEffect,
    PhysicalOperator, Prior, ScoreFrequentist, ScoreModel,
};
use antecedent_io::propagation_chain_record::{
    ChainDeferredTableWire, PROPAGATION_CHAIN_ASSUMPTION_VERSION,
    PROPAGATION_CHAIN_JOINT_LAW_VERSION, PROPAGATION_CHAIN_LAW_VERSION, PropagationChainDraftWire,
    PropagationChainRecordWire, propagation_chain_draft_identity, propagation_chain_identity,
};

use crate::planner::lowered::{
    CarrierDemand, ChainLoweredParts, ChainProgram, ClaimDemand, Composition, DecisionProgram,
    DesignShape, EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef, FunctionalKind,
    IdentificationLayer, LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget,
    ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{ModelRole, ModelSlot, Realization, StatisticalProgram};

/// Why a chain's design shape is not a quoted adjustment design.
const CHAIN_DESIGN_SHAPE: &str = "a propagation chain combines declared sources; it fits no \
                                  adjustment design";

const QUERY_TAG: &str = "antecedent.propagation_chain.lowered_query";
const SOURCE_TAG: &str = "antecedent.propagation_chain.lowered_source";
const DEFERRED_SOURCE_TAG: &str = "antecedent.propagation_chain.lowered_deferred_source.v1";

fn refuse(reason: impl std::fmt::Display) -> ChainRefusal {
    ChainRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        antecedent_core::reason_details::propagation_chain::LOWERING,
        format!("propagation chain lowering: {reason}"),
        None,
        FIX_CHAIN_DECLARATION,
    )
}

/// The identity a sealed record of `draft` carries: the declaration, the draws, the
/// future-study draws and the changes, with the label the rule yields and no results.
///
/// # Errors
///
/// A stored name that is not one of the format's names.
pub(crate) fn declaration_identity(
    draft: &PropagationChainRecordWire,
    deferred: Option<&ChainDeferredTableWire>,
    plan: &ChainPlan,
) -> Result<String, ChainRefusal> {
    let mut declared = draft.clone();
    plan.label.clone_into(&mut declared.label);
    match deferred {
        // A draft whose family table is bound at execution has its own identity domain.
        Some(table) => propagation_chain_draft_identity(&PropagationChainDraftWire {
            record: declared,
            deferred: table.clone(),
        }),
        None => propagation_chain_identity(&declared),
    }
    .map_err(|error| error.to_refusal())
}

fn uncertainty(plan: &ChainPlan) -> UncertaintyReading {
    match plan.stage(&plan.output) {
        Some(output) => match (output.representation, output.meaning) {
            (
                NodeRepresentation::Enclosure,
                Some(LawMeaning::Sampling | LawMeaning::Posterior | LawMeaning::Predictive),
            ) => UncertaintyReading::AssumptionEnvelope,
            (NodeRepresentation::Enclosure, _) => UncertaintyReading::Envelope,
            (NodeRepresentation::Draws, Some(LawMeaning::Posterior)) => {
                UncertaintyReading::Posterior
            }
            (NodeRepresentation::Draws, Some(LawMeaning::Predictive)) => {
                UncertaintyReading::Predictive
            }
            (NodeRepresentation::Draws, _) => UncertaintyReading::Sampling,
            _ => UncertaintyReading::None,
        },
        None => UncertaintyReading::None,
    }
}

fn source_digest(
    source: &antecedent_io::propagation_chain_record::ChainSourceWire,
    deferred: Option<&ChainDeferredTableWire>,
) -> [u8; 32] {
    // A slot of a deferred table has no draws yet: it is digested by its place in the table,
    // whose rows and seed are declared. A concrete source keeps its digest exactly.
    if let Some(table) = deferred
        && let Some(slot) = table.slots.iter().position(|id| *id == source.id)
    {
        let mut writer = CanonicalWriter::new(DEFERRED_SOURCE_TAG);
        writer
            .text(&source.id)
            .text(&source.representation)
            .text(&table.construction)
            .word(table.rows)
            .word(table.seed)
            .word(slot as u64);
        return writer.finish();
    }
    let mut writer = CanonicalWriter::new(SOURCE_TAG);
    writer.text(&source.id).text(&source.representation).word(source.values.len() as u64);
    for value in &source.values {
        writer.word(*value);
    }
    writer.finish()
}

/// The namespace of the names a chain gives its program, model families and plan. These are
/// program names, not refusal details.
const PROPAGATION_CHAIN_NAMESPACE: &str = "propagation_chain";

fn statistical_program(
    draft: &PropagationChainRecordWire,
    plan: &ChainPlan,
    uncertainty: UncertaintyReading,
) -> Result<StatisticalProgram, ChainRefusal> {
    let name = format!("{PROPAGATION_CHAIN_NAMESPACE}.{}", plan.reading.as_str());
    let realization = match uncertainty {
        UncertaintyReading::Sampling => {
            Realization::Frequentist(ScoreFrequentist::new(ScoreModel::new(name).map_err(refuse)?))
        }
        UncertaintyReading::Posterior => Realization::Bayesian(OrdinaryBayesianLikelihood::new(
            LikelihoodModel::new(name).map_err(refuse)?,
            Prior::new(format!("{PROPAGATION_CHAIN_NAMESPACE}.declared_posterior_sources"))
                .map_err(refuse)?,
        )),
        UncertaintyReading::Predictive => {
            Realization::DeclaredPredictive { evaluator: Arc::from(name) }
        }
        UncertaintyReading::MonteCarlo => {
            return Err(refuse("Monte Carlo decision draws require their own compiler route"));
        }
        UncertaintyReading::AssumptionEnvelope => {
            let draw_meaning = plan
                .stage(&plan.output)
                .and_then(|stage| stage.meaning)
                .ok_or_else(|| refuse("an assumption envelope needs a draw-law meaning"))?;
            Realization::DeclaredAssumptionEnvelope { evaluator: Arc::from(name), draw_meaning }
        }
        UncertaintyReading::Envelope | UncertaintyReading::Empirical | UncertaintyReading::None => {
            Realization::ExactPoint { evaluator: Arc::from(name) }
        }
    };
    let models: Vec<ModelSlot> = draft
        .stages
        .iter()
        .map(|stage| ModelSlot {
            role: if matches!(stage.kind.as_str(), "preposterior_update" | "predictive_mapping") {
                ModelRole::Outcome
            } else {
                ModelRole::Mechanism
            },
            family: Arc::from(format!("{PROPAGATION_CHAIN_NAMESPACE}.{}.{}", stage.kind, stage.id)),
            inputs: Arc::from(Vec::new()),
            output: None,
        })
        .collect();
    Ok(StatisticalProgram {
        identifier: None,
        estimator: None,
        realization,
        models: Arc::from(models),
        folds: None,
        resampling: None,
        posterior: None,
        design: DesignShape::not_design_based(CHAIN_DESIGN_SHAPE),
        operators: Arc::from(Vec::new()),
    })
}

fn operator(effect: PhysicalEffect, output: Option<u64>) -> Result<PhysicalOperator, ChainRefusal> {
    let mut builder = PhysicalOperator::declare(effect);
    if let Some(point) = effect.required_cancellation_point() {
        builder = builder.cancellation_point(point);
    }
    if let Some(bytes) = output {
        builder = builder.estimates(ByteEstimates::absent().with_output(bytes));
    }
    builder.build().map_err(refuse)
}

/// One read of the sources, then one operator per stage; the last chain stage
/// carries the table bytes. A combined decision adds its scoring operator.
fn physical_layer(
    draft: &PropagationChainRecordWire,
    deferred: Option<&ChainDeferredTableWire>,
    decision: Option<&DecisionProgram>,
) -> Result<PhysicalLayer, ChainRefusal> {
    let rows = draft
        .sources
        .iter()
        .map(|s| s.values.len() as u64)
        .chain(deferred.map(|table| table.rows))
        .max()
        .unwrap_or(1);
    let nodes = (draft.sources.len() + draft.stages.len()) as u64;
    let retained_joint_columns = draft.joint_outputs.len() as u64;
    let table_bytes = rows
        .saturating_mul(nodes.saturating_add(retained_joint_columns).saturating_add(1))
        .saturating_mul(8);
    let mut effects = vec![PhysicalEffect::Read];
    for stage in &draft.stages {
        effects.push(match StageKind::from_name(&stage.kind) {
            Some(StageKind::Transport) => PhysicalEffect::Transport,
            Some(StageKind::PreposteriorUpdate) => PhysicalEffect::Prior,
            _ => PhysicalEffect::Condition,
        });
    }
    let last = effects.len() - 1;
    let mut operators = effects
        .into_iter()
        .enumerate()
        .map(|(index, effect)| operator(effect, (index == last).then_some(table_bytes)))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(decision) = decision {
        // Utility and chance-constraint scoring is a physical step of this
        // combined program. The law route may retain one utility column per
        // action and a weight column; the envelope route retains less.
        let decision_bytes = rows.saturating_mul(decision.actions() as u64 + 1).saturating_mul(8);
        operators.push(operator(PhysicalEffect::Condition, Some(decision_bytes))?);
    }
    Ok(PhysicalLayer {
        operators: Arc::from(operators),
        design: DesignShape::not_design_based(CHAIN_DESIGN_SHAPE),
    })
}

/// Lower a checked chain.
///
/// `draft` is the declared chain with no results, `plan` what the chain rule
/// decided for it. Nothing is evaluated.
///
/// # Errors
///
/// A name the format does not know, a model name the statistical vocabulary
/// refuses, or a physical operator that does not declare its cancellation point.
pub(crate) fn lower_chain(
    draft: PropagationChainRecordWire,
    deferred: Option<ChainDeferredTableWire>,
    plan: ChainPlan,
) -> Result<LoweredProgram, ChainRefusal> {
    lower_chain_with_decision(draft, deferred, plan, None)
}

/// Lower a chain and an optional decision contract as one six-layer program.
/// The caller checks the decision's quantities and carrier before supplying it.
#[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
pub(crate) fn lower_chain_with_decision(
    draft: PropagationChainRecordWire,
    deferred: Option<ChainDeferredTableWire>,
    plan: ChainPlan,
    declared_decision: Option<DecisionProgram>,
) -> Result<LoweredProgram, ChainRefusal> {
    let identity = declaration_identity(&draft, deferred.as_ref(), &plan)?;
    let query = CanonicalWriter::digest_bytes(QUERY_TAG, identity.as_bytes());
    let uncertainty = uncertainty(&plan);
    if let Some(decision) = &declared_decision {
        let supported = matches!(
            (decision.carrier(), uncertainty),
            (CarrierDemand::ParameterPosterior, UncertaintyReading::Posterior)
                | (CarrierDemand::PredictiveLaw, UncertaintyReading::Predictive)
                | (CarrierDemand::IdentifiedSet, UncertaintyReading::AssumptionEnvelope)
        );
        if !supported {
            return Err(refuse(
                "the decision's law or identified-set carrier does not match the chain output reading",
            ));
        }
    }
    let reading = plan.reading.as_str();
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: crate::planner::lowered::ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PropagatedChain,
            outcome: None,
            query,
        },
        plan_id: Arc::from(format!("{PROPAGATION_CHAIN_NAMESPACE}.{reading}")),
        data: DataClassification::Tabular,
        graph_class: "none",
        query_variables: Arc::from(Vec::new()),
        source_digest: None,
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let statistical = statistical_program(&draft, &plan, uncertainty)?;
    let physical = physical_layer(&draft, deferred.as_ref(), declared_decision.as_ref())?;
    let sources: Vec<EvidenceSourceRef> = draft
        .sources
        .iter()
        .map(|source| EvidenceSourceRef {
            role: EvidenceRole::ChainSource,
            digest: Some(source_digest(source, deferred.as_ref())),
        })
        .collect();
    let stages = draft.stages.len();
    // Only the answer's own utility stage is a terminal decision: the generic
    // decision program reads the answer, not an unrelated intermediate stage.
    // Version 1 keeps its original lowered identity and stage-only verdict.
    let decision_reference = draft
        .stages
        .iter()
        .find(|stage| stage.id == draft.output)
        .and_then(|stage| stage.decision_reference);
    let decision = if let Some(declared) = declared_decision {
        Some(declared)
    } else if let Some(reference_bits) = decision_reference
        && matches!(
            draft.version,
            PROPAGATION_CHAIN_LAW_VERSION
                | PROPAGATION_CHAIN_ASSUMPTION_VERSION
                | PROPAGATION_CHAIN_JOINT_LAW_VERSION
        )
    {
        let reference = f64::from_bits(reference_bits);
        let assumption_envelope = draft.version == PROPAGATION_CHAIN_ASSUMPTION_VERSION;
        Some(DecisionProgram {
            // The utility stage already transforms every aligned row. Its
            // expected value is enough to compare with a fixed reference.
            carrier: if assumption_envelope {
                CarrierDemand::IdentifiedSet
            } else {
                CarrierDemand::Mean
            },
            utility: DecisionProgram::utility_form(Some(vec![(reference, 0.0), (0.0, 1.0)])),
            actions: 2,
            constraints: 0,
            chance_threshold: 0.0,
            utility_unit: draft.output_quantity.as_ref().and_then(|quantity| {
                (quantity.role == "utility").then(|| Arc::from(quantity.units.as_str()))
            }),
            contract_identity: Some(Arc::from(identity.as_str())),
            information_value: None,
            scenario_source: assumption_envelope.then(|| Arc::from(identity.as_str())),
        })
    } else {
        None
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(sources),
        composition: Composition::Chained { stages },
        decision,
        planning: None,
        chain: Some(ChainProgram::new(draft, deferred, plan, identity)),
    };
    let claim = ClaimDemand {
        uncertainty,
        kind: Some(match uncertainty {
            UncertaintyReading::Envelope | UncertaintyReading::AssumptionEnvelope => {
                ClaimKind::Bounds
            }
            UncertaintyReading::Predictive => ClaimKind::Associational,
            _ => ClaimKind::Point,
        }),
    };
    Ok(ChainLoweredParts {
        reading,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::analysis::propagation_chain::{PropagationChain, compile_propagation_chain};
    use crate::identity_laws::{
        ADVERSARIAL_TEXT, all_distinct, bit_exact, injective, ordered_laws,
    };
    use antecedent_contract::propagation_chain::{ChainReading, TableConstruction};
    use antecedent_core::ExecutionContext;
    use antecedent_io::propagation_chain_record::{
        ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire,
    };

    fn source(values: &[f64]) -> ChainSourceWire {
        ChainSourceWire::draws("estimate", LawMeaning::Sampling, "rows", values)
    }

    #[test]
    fn site_chain_source_digest_orders_rows_and_frames_its_texts() {
        ordered_laws(&[1.0_f64, 2.0, 3.0], |values| Some(source_digest(&source(values), None)));
        bit_exact(|value| Some(source_digest(&source(&[value, 1.0]), None)));
        injective(
            &ADVERSARIAL_TEXT,
            |pair| *pair,
            |(id, representation)| {
                let mut wire = source(&[1.0]);
                (*id).clone_into(&mut wire.id);
                (*representation).clone_into(&mut wire.representation);
                Some(source_digest(&wire, None))
            },
        );
    }

    #[test]
    fn site_chain_source_digest_binds_a_deferred_slot_by_its_column() {
        let table = |slots: &[&str], rows: u64, seed: u64| ChainDeferredTableWire {
            construction: "moment_matched_normal".into(),
            rows,
            seed,
            slots: slots.iter().map(|slot| (*slot).to_owned()).collect(),
            declared_cross: Vec::new(),
            within_blocks: Vec::new(),
        };
        let digest = |slots: &[&str], rows: u64, seed: u64| {
            source_digest(&source(&[]), Some(&table(slots, rows, seed)))
        };
        all_distinct(&[
            digest(&["estimate", "other"], 8, 3),
            digest(&["other", "estimate"], 8, 3),
            digest(&["estimate", "other"], 9, 3),
            digest(&["estimate", "other"], 8, 4),
            source_digest(&source(&[]), None),
        ]);
    }

    #[test]
    fn site_chain_lowered_query_binds_the_declaration() {
        let ctx = ExecutionContext::for_tests(23);
        let draft = |values: &[f64], slope: f64| {
            PropagationChain::draft(
                TableConstruction::JointDraws,
                ChainReading::Spread,
                "output",
                vec![ChainSourceWire::draws("source", LawMeaning::Sampling, "aligned", values)],
                vec![ChainStageWire::new(
                    "output",
                    StageKind::DerivedQuantity,
                    &["source"],
                    ChainOpWire::affine(
                        ChainCoefficientWire::fixed(slope),
                        ChainCoefficientWire::fixed(0.0),
                    ),
                )],
            )
        };
        let identity = |values: &[f64], slope: f64| {
            compile_propagation_chain(draft(values, slope), &ctx)
                .expect("chain compiler")
                .lowered()
                .identity()
        };
        all_distinct(&[
            identity(&[1.0, 2.0, 3.0], 2.0),
            identity(&[2.0, 1.0, 3.0], 2.0),
            identity(&[1.0, 2.0, 3.0], 3.0),
        ]);
    }
}
