//! Cyclic family lowering: every cyclic preparation carries a lowered program.
//!
//! A cyclic family prepares through its own entry (`Study::prepare_cyclic*`), not
//! through `Study::compile`, but it lowers to the same six layers:
//!
//! 1. the keyed worlds of the request as a scientific target in the feedback
//!    vocabulary (equilibrium, continuous-time or stationary frames; hard clamps,
//!    mechanism replacements, additive shocks and initial-state changes), or the
//!    functional a family without keyed worlds answers;
//! 2. identification that is not a graphical identification product: conditional
//!    on the specified model, an experimental recovery with its design digest and
//!    the directions the design leaves unresolved, or a theorem-scoped cyclic
//!    input/output SCM identification;
//! 3. the statistical program from the family's draw origin and its typed
//!    fit-dependency set;
//! 4. the evidence the family reads;
//! 5. one physical operator per compiled solve block, with the output bytes the
//!    family admits before its numerical engine runs;
//! 6. the claim demand.
//!
//! Lowering does not fit, sample, solve or encode an artifact, and grants no
//! scientific license: a specified model stays model-conditional.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CancellationPoint, ExecutionContext, LikelihoodModel,
    OrdinaryBayesianLikelihood, PhysicalEffect, PhysicalOperator, Prior, ScoreFrequentist,
    ScoreModel, VariableId,
};
use antecedent_io::contract_section::cyclic::{CyclicContractPayloads, CyclicIdentificationWire};
use antecedent_io::cyclic_family_artifact::{
    CompiledCyclicFamilyProgram, CyclicFamilyRequest, CyclicProblem, CyclicRealization,
    CyclicUncertainty, cyclic_contract_payloads, cyclic_problem, cyclic_realization,
};
use antecedent_io::cyclic_system_artifact::{CyclicArtifactError, CyclicExecutionRequest};
use antecedent_model::compile::{CompiledSystemPlan, ExecutionBlock};

use crate::planner::lowered::{
    ClaimDemand, Composition, CyclicLoweredParts, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ScientificProblem,
    UncertaintyReading,
};
use crate::planner::statistical::{
    ModelRole, ModelSlot, Realization, ResampleUnit, ResamplingMethod, ResamplingPlan,
    StatisticalProgram,
};

/// Why a cyclic family's design shape is not a quoted adjustment design.
const CYCLIC_SOLVE_BLOCKS: &str = "a cyclic family solves blocks; it fits no adjustment design";

fn refuse(reason: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(format!("cyclic lowering: {reason}"))
}

/// Lower a compiled cyclic family program.
///
/// # Errors
///
/// A keyed world the scientific vocabulary refuses, an output bound that
/// overflows, or a physical operator that does not declare its cancellation point.
pub(crate) fn lower_cyclic_family(
    program: &CompiledCyclicFamilyProgram,
) -> Result<LoweredProgram, CyclicArtifactError> {
    let request = program.request();
    let payloads = cyclic_contract_payloads(
        request,
        program.execution_plans(),
        program.fit_dependency_identity(),
    )?;
    let realization = cyclic_realization(request);
    let scientific = scientific_problem(request, &payloads)?;
    let identification = identification_layer(&payloads.identification);
    let statistical = statistical_program(request, &realization, &payloads)?;
    let evidence = evidence_program(&payloads);
    let physical =
        physical_layer(program.execution_plans(), &realization, program.output_bytes_bound()?)?;
    let claim = ClaimDemand {
        uncertainty: match realization.uncertainty {
            CyclicUncertainty::None => UncertaintyReading::None,
            CyclicUncertainty::Sampling => UncertaintyReading::Sampling,
            CyclicUncertainty::Posterior => UncertaintyReading::Posterior,
        },
        kind: Some(match &scientific.target {
            ProblemTarget::QueryFunctional { kind: FunctionalKind::IdentifiedSet, .. } => {
                ClaimKind::Bounds
            }
            _ => ClaimKind::Response,
        }),
    };
    Ok(CyclicLoweredParts {
        family: request.family_name(),
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

/// Lower a single-system request as the `system` family it is carried as.
///
/// Compiles the family schedules for the plan and does not fit.
///
/// # Errors
///
/// As [`lower_cyclic_family`], or the system request does not compile.
pub(crate) fn lower_cyclic_system(
    request: &CyclicExecutionRequest,
    ctx: &ExecutionContext,
) -> Result<LoweredProgram, CyclicArtifactError> {
    let family = CyclicFamilyRequest::System { request: request.clone() };
    lower_cyclic_family(&CompiledCyclicFamilyProgram::compile(&family, ctx)?)
}

fn functional_kind(kind: &str) -> FunctionalKind {
    match kind {
        "counterfactual" => FunctionalKind::Counterfactual,
        "identified_set" => FunctionalKind::IdentifiedSet,
        _ => FunctionalKind::InterventionalDistribution,
    }
}

fn data_classification(modality: &str) -> DataClassification {
    match modality {
        "clamp_experiments" | "stationary_regimes" => DataClassification::MultiEnvironment,
        "trajectories" => DataClassification::Temporal,
        _ => DataClassification::Tabular,
    }
}

fn reads_data(payloads: &CyclicContractPayloads) -> bool {
    payloads.observation.modality != "specified"
}

fn variables(payloads: &CyclicContractPayloads) -> Arc<[VariableId]> {
    payloads.observation.variables.iter().copied().map(VariableId::from_raw).collect()
}

fn scientific_problem(
    request: &CyclicFamilyRequest,
    payloads: &CyclicContractPayloads,
) -> Result<ScientificProblem, CyclicArtifactError> {
    let target = match cyclic_problem(request)? {
        CyclicProblem::Target(target) => ProblemTarget::Target(target),
        CyclicProblem::JointTargets(targets) => ProblemTarget::Targets(Arc::from(targets)),
        CyclicProblem::Functional { kind, outcome, query } => {
            ProblemTarget::QueryFunctional { kind: functional_kind(kind), outcome, query }
        }
    };
    Ok(ScientificProblem {
        target,
        plan_id: Arc::from(format!("cyclic_family.{}", request.family_name())),
        data: data_classification(&payloads.observation.modality),
        graph_class: "cyclic",
        query_variables: variables(payloads),
        source_digest: reads_data(payloads).then_some(payloads.data_snapshot.content),
        support_coordinate: None,
        licensed: false,
    })
}

fn identification_layer(identification: &CyclicIdentificationWire) -> IdentificationLayer {
    IdentificationLayer::NonGraphical(Arc::new(match identification {
        CyclicIdentificationWire::SpecifiedModelConditional { model } => {
            NonGraphicalIdentification::SpecifiedModelConditional { model: *model }
        }
        CyclicIdentificationWire::ExperimentalRecovery { design, unresolved } => {
            NonGraphicalIdentification::ExperimentalRecovery {
                design: *design,
                unresolved: unresolved
                    .iter()
                    .map(|direction| {
                        (
                            VariableId::from_raw(direction.target),
                            direction
                                .missing_clamps
                                .iter()
                                .copied()
                                .map(VariableId::from_raw)
                                .collect::<Arc<[VariableId]>>(),
                        )
                    })
                    .collect(),
            }
        }
        CyclicIdentificationWire::CyclicIoScm { source } => {
            NonGraphicalIdentification::CyclicIoScm { source: *source }
        }
    }))
}

fn resample_unit(unit: &str) -> ResampleUnit {
    match unit {
        "subject" => ResampleUnit::Subject,
        "cluster" => ResampleUnit::Cluster,
        "calendar_block" => ResampleUnit::CalendarBlock,
        _ => ResampleUnit::Row,
    }
}

/// The statistical program: the draw origin (how uncertainty is produced) and the
/// typed fit-dependency set (which models the fit estimates). The fit's data
/// enters the evidence layer, so a data refresh does not change the program.
fn statistical_program(
    request: &CyclicFamilyRequest,
    realization: &CyclicRealization,
    payloads: &CyclicContractPayloads,
) -> Result<StatisticalProgram, CyclicArtifactError> {
    let family = request.family_name();
    let name = format!("cyclic.{family}.{}.{}", realization.realization, realization.method);
    let realized = match realization.uncertainty {
        CyclicUncertainty::Sampling => {
            Realization::Frequentist(ScoreFrequentist::new(ScoreModel::new(name).map_err(refuse)?))
        }
        CyclicUncertainty::Posterior => {
            let prior = Prior::new(format!("cyclic.{family}.declared_prior")).map_err(refuse)?;
            Realization::Bayesian(OrdinaryBayesianLikelihood::new(
                LikelihoodModel::new(name).map_err(refuse)?,
                prior,
            ))
        }
        CyclicUncertainty::None => Realization::ExactPoint { evaluator: Arc::from(name) },
    };
    let inputs = variables(payloads);
    let mut models = vec![ModelSlot {
        role: ModelRole::Mechanism,
        family: Arc::from(format!("cyclic.{family}.mechanisms")),
        inputs: Arc::clone(&inputs),
        output: None,
    }];
    if realization.realization != "specified_evaluation"
        && realization.realization != "supplied_draws"
        && realization.realization != "graphical_reasoning"
    {
        models.push(ModelSlot {
            role: ModelRole::Nuisance,
            family: Arc::from(format!("cyclic.fit_dependencies.{family}")),
            inputs,
            output: None,
        });
    }
    let resampling =
        realization.replicates.filter(|replicates| *replicates > 0).map(|replicates| {
            let unit = resample_unit(realization.resample_unit.unwrap_or("row"));
            ResamplingPlan {
                method: if unit == ResampleUnit::CalendarBlock {
                    ResamplingMethod::BlockBootstrap
                } else {
                    ResamplingMethod::Bootstrap
                },
                replicates: u32::try_from(replicates).unwrap_or(u32::MAX),
                unit,
            }
        });
    Ok(StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: realized,
        models: Arc::from(models),
        folds: None,
        resampling,
        posterior: None,
        design: DesignShape::not_design_based(CYCLIC_SOLVE_BLOCKS),
        operators: Arc::from(Vec::new()),
    })
}

fn evidence_program(payloads: &CyclicContractPayloads) -> EvidenceDecisionProgram {
    let role = if payloads.observation.modality == "clamp_experiments" {
        EvidenceRole::Trial
    } else {
        EvidenceRole::Observations
    };
    let sources: Vec<EvidenceSourceRef> = if reads_data(payloads) {
        vec![EvidenceSourceRef { role, digest: Some(payloads.data_snapshot.content) }]
    } else {
        Vec::new()
    };
    EvidenceDecisionProgram {
        sources: Arc::from(sources),
        composition: Composition::Single,
        decision: None,
    }
}

fn operator(
    effect: PhysicalEffect,
    output: Option<u64>,
) -> Result<PhysicalOperator, CyclicArtifactError> {
    let mut builder = PhysicalOperator::declare(effect);
    if let Some(point) = effect.required_cancellation_point() {
        builder = builder.cancellation_point(point);
    }
    if let Some(bytes) = output {
        builder = builder.estimates(ByteEstimates::absent().with_output(bytes));
    }
    builder.build().map_err(refuse)
}

/// One operator per compiled block, a fit before them when the family fits, and
/// the family's admitted output bytes on the last operator.
fn physical_layer(
    plans: &[CompiledSystemPlan],
    realization: &CyclicRealization,
    output_bytes: u64,
) -> Result<PhysicalLayer, CyclicArtifactError> {
    let mut effects = Vec::new();
    if realization.realization != "specified_evaluation"
        && realization.realization != "supplied_draws"
        && realization.realization != "graphical_reasoning"
        && realization.realization != "exact_functional"
    {
        effects.push(PhysicalEffect::Fit);
    }
    for plan in plans {
        for block in &plan.blocks {
            effects.push(match block {
                ExecutionBlock::Mechanism { .. } => PhysicalEffect::Intervene,
                ExecutionBlock::SimultaneousEquations { .. } => PhysicalEffect::Solve,
                ExecutionBlock::DynamicEvolution { .. } => PhysicalEffect::Scan,
            });
        }
    }
    if effects.is_empty() {
        effects.push(PhysicalEffect::Intervene);
    }
    let last = effects.len() - 1;
    let operators = effects
        .into_iter()
        .enumerate()
        .map(|(index, effect)| operator(effect, (index == last).then_some(output_bytes)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PhysicalLayer {
        operators: Arc::from(operators),
        design: DesignShape::not_design_based(CYCLIC_SOLVE_BLOCKS),
    })
}

/// A solve block publishes its cancellation point.
const _: () = assert!(matches!(
    PhysicalEffect::Solve.required_cancellation_point(),
    Some(CancellationPoint::Solve)
));

#[cfg(test)]
#[path = "cyclic_tests.rs"]
mod tests;
