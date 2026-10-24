//! Checked-lowering sections of a program identity.
//!
//! A prepared plan binds at most one checked procedure, so the lowering a
//! contract commits to is a function of that binding. [`ProgramLowerings::bind`]
//! is that function; the per-procedure conversions it dispatches to are pure
//! readers of the checked preparation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CausalQuery;
use antecedent_io::{
    CheckedAipwLoweringWire, CheckedFrontDoorLoweringWire, CheckedFunctionalResponseGridWire,
    CheckedFunctionalResponseMemberWire, CheckedIvLoweringWire,
    CheckedLinearAdjustmentLoweringWire, CheckedNestedCounterfactualWire, FunctionalProgramWire,
    InferentialCommitmentsWire, causal_query_to_wire, functional_program_to_wire,
    identification_product_wire,
};

use crate::EstimatorId;
use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::support::StructureSource;

use super::super::execute::Study;
use super::super::prepared::CheckedProgramBinding;
use super::{data_schema, io_err, overlap_policy_tag};
use antecedent_identify::IdentificationResult;

/// The estimator and query a checked binding is only valid for.
///
/// A binding that disagrees with the selected estimator or the study's query
/// would describe a procedure no prepared plan selected.
pub(super) fn binding_agrees(
    binding: CheckedProgramBinding<'_>,
    selected: Option<EstimatorId>,
    query: &CausalQuery,
) -> bool {
    let average = matches!(query, CausalQuery::AverageEffect(_));
    match binding {
        CheckedProgramBinding::None => true,
        CheckedProgramBinding::Aipw(_) => selected == Some(EstimatorId::Aipw) && average,
        CheckedProgramBinding::FrontDoor(_) => {
            selected == Some(EstimatorId::FrontDoorTwoStage) && average
        }
        CheckedProgramBinding::Iv(_) => {
            matches!(selected, Some(EstimatorId::IvWald | EstimatorId::Iv2Sls)) && average
        }
        CheckedProgramBinding::Linear(_) => {
            selected == Some(EstimatorId::LinearAdjustmentAte) && average
        }
        CheckedProgramBinding::FunctionalEffect(_) => {
            selected == Some(EstimatorId::FunctionalEffect)
                && matches!(
                    query,
                    CausalQuery::AverageEffect(_)
                        | CausalQuery::PathSpecific(_)
                        | CausalQuery::ConditionalEffect(_)
                )
        }
        CheckedProgramBinding::FunctionalResponse(_) => {
            selected == Some(EstimatorId::FunctionalEffect)
                && matches!(query, CausalQuery::Response(_))
        }
        CheckedProgramBinding::Distribution(_) => {
            selected == Some(EstimatorId::FunctionalDistribution)
                && matches!(query, CausalQuery::Distribution(_))
        }
        CheckedProgramBinding::NestedCounterfactual(_) => {
            selected == Some(EstimatorId::StaticMediationLinear)
                && matches!(query, CausalQuery::NestedCounterfactual(_))
        }
    }
}

/// The lowering sections a program identity carries for its bound procedure.
/// At most one is present.
#[derive(Default)]
pub(super) struct ProgramLowerings {
    pub(super) aipw: Option<CheckedAipwLoweringWire>,
    pub(super) frontdoor: Option<CheckedFrontDoorLoweringWire>,
    pub(super) iv: Option<CheckedIvLoweringWire>,
    pub(super) linear_adjustment: Option<CheckedLinearAdjustmentLoweringWire>,
    pub(super) functional_response_grid: Option<CheckedFunctionalResponseGridWire>,
    pub(super) nested_counterfactual: Option<CheckedNestedCounterfactualWire>,
}

impl ProgramLowerings {
    /// Lowering of the bound procedure. A procedure that fixes the
    /// standard-error kind records it into `commitments`.
    pub(super) fn bind(
        binding: CheckedProgramBinding<'_>,
        study: &Study,
        resolved_estimator: Option<&str>,
        commitments: &mut InferentialCommitmentsWire,
    ) -> Result<Self, CausalError> {
        let mut lowerings = Self::default();
        match binding {
            CheckedProgramBinding::None
            | CheckedProgramBinding::FunctionalEffect(_)
            | CheckedProgramBinding::Distribution(_) => {}
            CheckedProgramBinding::Aipw(checked) => {
                let (wire, se_kind) = aipw_lowering(checked);
                commitments.se_kind = Some(se_kind);
                lowerings.aipw = Some(wire);
            }
            CheckedProgramBinding::FrontDoor(checked) => {
                lowerings.frontdoor = Some(frontdoor_lowering(checked, commitments)?);
            }
            CheckedProgramBinding::Iv(checked) => {
                let (wire, se_kind) = iv_lowering(checked);
                commitments.se_kind = Some(se_kind);
                lowerings.iv = Some(wire);
            }
            CheckedProgramBinding::Linear(operation) => {
                if linear_lowering_licensed(study, resolved_estimator) {
                    let (wire, se_kind) = linear_adjustment_lowering(operation, commitments)?;
                    commitments.se_kind = Some(se_kind);
                    lowerings.linear_adjustment = Some(wire);
                }
            }
            CheckedProgramBinding::FunctionalResponse(members) => {
                lowerings.functional_response_grid = functional_response_grid(study, members)?;
            }
            CheckedProgramBinding::NestedCounterfactual(operation) => {
                let query = operation.query();
                lowerings.nested_counterfactual = Some(CheckedNestedCounterfactualWire {
                    format: 1,
                    treatment: query.treatment.raw(),
                    mediator: query.mediator.raw(),
                    outcome: query.outcome.raw(),
                    control_bits: query.control_value().to_bits(),
                    active_bits: query.active_value().to_bits(),
                    model: "linear_gaussian".into(),
                    procedure: "natural_direct_shared_exogenous".into(),
                });
            }
        }
        Ok(lowerings)
    }
}

fn se_lag(se_kind: antecedent_estimate::AnalyticSeKind) -> Option<u64> {
    match se_kind {
        antecedent_estimate::AnalyticSeKind::NeweyWest { lag }
        | antecedent_estimate::AnalyticSeKind::PanelClusterHac { lag } => Some(lag as u64),
        _ => None,
    }
}

fn aipw_lowering(
    checked: &antecedent_estimate::CheckedAipwPreparation,
) -> (CheckedAipwLoweringWire, String) {
    let lowering = checked.lowering();
    let wire = CheckedAipwLoweringWire {
        format: 1,
        functional: lowering.functional.raw(),
        treatment: lowering.treatment.raw(),
        outcome: lowering.outcome.raw(),
        adjustment: lowering.adjustment.iter().map(|id| id.raw()).collect(),
        population: "all_observed".into(),
        procedure: lowering.procedure.as_str().into(),
        folds: lowering.folds as u64,
        se_kind: lowering.se_kind.as_str().into(),
        se_lag: se_lag(lowering.se_kind),
        bootstrap_replicates: lowering.bootstrap_replicates,
        trim_bits: match lowering.overlap {
            antecedent_estimate::OverlapPolicy::RequireDiagnostics { trim: Some(trim), .. } => {
                Some(trim.to_bits())
            }
            _ => None,
        },
    };
    (wire, lowering.se_kind.as_str().to_string())
}

fn frontdoor_lowering(
    checked: &antecedent_estimate::CheckedFrontDoorPreparation,
    commitments: &InferentialCommitmentsWire,
) -> Result<CheckedFrontDoorLoweringWire, CausalError> {
    use antecedent_estimate::frontdoor::CheckedFrontDoorProcedure;
    let lowering = checked.lowering();
    let procedure = match lowering.procedure {
        CheckedFrontDoorProcedure::LinearPathProduct => "linear_path_product",
        CheckedFrontDoorProcedure::Functional => "functional",
    };
    let problem = checked.problem();
    Ok(CheckedFrontDoorLoweringWire {
        format: 1,
        functional: lowering.functional.raw(),
        executable: lowering.executable.raw(),
        treatment: lowering.treatment.raw(),
        outcome: lowering.outcome.raw(),
        mediators: lowering.mediators.iter().map(|id| id.raw()).collect(),
        active_bits: lowering.active.to_bits(),
        control_bits: lowering.control.to_bits(),
        procedure: procedure.into(),
        complete_case_rows: problem.nrows as u64,
        overlap: overlap_policy_tag(problem.overlap),
        uncertainty: format!(
            "{}:{}",
            commitments.interval_method,
            commitments.se_kind.as_deref().unwrap_or("unspecified")
        ),
        arena: antecedent_io::expr_arena_to_wire(checked.program().arena())
            .map_err(|error| io_err(&error))?,
    })
}

fn iv_lowering(
    checked: &antecedent_estimate::CheckedIvPreparation,
) -> (CheckedIvLoweringWire, String) {
    use antecedent_estimate::CheckedIvProcedure;
    let lowering = checked.lowering();
    let procedure = match lowering.procedure {
        CheckedIvProcedure::Wald => "wald",
        CheckedIvProcedure::TwoStageLeastSquares => "two_stage_least_squares",
    };
    let weak_instrument_uncertainty =
        if lowering.se_kind == antecedent_estimate::AnalyticSeKind::Homoskedastic {
            "anderson_rubin_if_weak"
        } else {
            "withheld_non_homoskedastic"
        };
    let wire = CheckedIvLoweringWire {
        format: 1,
        functional: lowering.functional.raw(),
        treatment: lowering.treatment.raw(),
        outcome: lowering.outcome.raw(),
        instrument: lowering.instruments.first().map_or(u32::MAX, |id| id.raw()),
        adjustment: lowering.adjustment.iter().map(|id| id.raw()).collect(),
        active_bits: lowering.active.to_bits(),
        control_bits: lowering.control.to_bits(),
        instrument_active_bits: lowering.instrument_active.to_bits(),
        instrument_control_bits: lowering.instrument_control.to_bits(),
        procedure: procedure.into(),
        se_kind: lowering.se_kind.as_str().into(),
        weak_instrument_uncertainty: weak_instrument_uncertainty.into(),
        complete_case_rows: u64::try_from(checked.problem().nrows).unwrap_or(u64::MAX),
    };
    (wire, lowering.se_kind.as_str().to_string())
}

/// The linear adjustment lowering is a program commitment only for the
/// frequentist route over an explicit or accepted DAG.
fn linear_lowering_licensed(study: &Study, resolved_estimator: Option<&str>) -> bool {
    resolved_estimator == Some("linear.adjustment.ate")
        && study.graph.class() == GraphClass::Dag
        && matches!(study.structure_source, StructureSource::Explicit | StructureSource::Accepted)
        && matches!(study.inference, InferenceMode::Frequentist)
}

fn linear_adjustment_lowering(
    operation: &crate::analysis::prepared::CheckedLinearOperation,
    commitments: &InferentialCommitmentsWire,
) -> Result<(CheckedLinearAdjustmentLoweringWire, String), CausalError> {
    use antecedent_estimate::LinearFitKind;
    use antecedent_stats::DesignColumnRole;
    let checked = &operation.preparation;
    let lowering = checked.lowering();
    let problem = checked.problem();
    let fit_kind = match operation.fitter.fit_kind {
        LinearFitKind::Ols => "ols".to_string(),
        LinearFitKind::Ridge { lambda } => format!("ridge:{:016x}", lambda.to_bits()),
        LinearFitKind::Lasso { lambda } => format!("lasso:{:016x}", lambda.to_bits()),
        LinearFitKind::Huber { c } => format!("huber:{:016x}", c.to_bits()),
    };
    let design_columns = problem
        .design
        .columns
        .iter()
        .map(|column| match column.role {
            DesignColumnRole::Intercept => "intercept".to_string(),
            DesignColumnRole::Treatment => "treatment".to_string(),
            DesignColumnRole::Covariate(id) => format!("covariate:{}", id.raw()),
        })
        .collect();
    let se_kind = operation.fitter.se_kind;
    let wire = CheckedLinearAdjustmentLoweringWire {
        format: 1,
        functional: lowering.source.raw(),
        executable: lowering.executable.raw(),
        treatment: lowering.treatment.raw(),
        outcome: lowering.outcome.raw(),
        adjustment: lowering.adjustment.iter().map(|id| id.raw()).collect(),
        active_bits: lowering.active.to_bits(),
        control_bits: lowering.control.to_bits(),
        population: antecedent_io::TargetPopulationWire::from_domain(&lowering.population)
            .map_err(|error| io_err(&error))?,
        design_columns,
        fit_kind,
        backend: "faer".into(),
        se_kind: se_kind.as_str().into(),
        se_lag: se_lag(se_kind),
        bootstrap_replicates: operation.fitter.bootstrap_replicates,
        interval_method: commitments.interval_method.clone(),
        complete_case_rows: problem.design.nrows as u64,
        arena: antecedent_io::expr_arena_to_wire(checked.program().arena())
            .map_err(|error| io_err(&error))?,
    };
    Ok((wire, se_kind.as_str().to_string()))
}

/// Grid of checked response members, for a mean-curve target only.
fn functional_response_grid(
    study: &Study,
    members: &[crate::analysis::prepared::CheckedFunctionalEffectResponseMember],
) -> Result<Option<CheckedFunctionalResponseGridWire>, CausalError> {
    let CausalQuery::Response(query) = &study.query else {
        return Ok(None);
    };
    let antecedent_core::ResponseFunctional::MeanCurve { treatment, outcome } = &query.functional
    else {
        return Ok(None);
    };
    let members = members
        .iter()
        .map(|member| {
            let query = CausalQuery::Response(member.query().clone());
            Ok(CheckedFunctionalResponseMemberWire {
                grid_value_bits: member.grid_value().to_bits(),
                query: causal_query_to_wire(&query).map_err(|error| io_err(&error))?,
                identification: identification_product_wire(member.identification(), false)
                    .map_err(|error| io_err(&error))?,
                program: functional_program_to_wire(member.program())
                    .map_err(|error| io_err(&error))?,
            })
        })
        .collect::<Result<Vec<_>, CausalError>>()?;
    Ok(Some(CheckedFunctionalResponseGridWire {
        format: 1,
        treatment: treatment.variable.raw(),
        outcome: outcome.raw(),
        members,
    }))
}

/// Functional program a program identity commits to.
///
/// An executable functional effect or distribution carries its checked
/// program; a distribution prepared without one derives it from the cached
/// identification's selected estimand.
pub(super) fn functional_program(
    binding: CheckedProgramBinding<'_>,
    study: &Study,
    cached: Option<&IdentificationResult>,
    selected: Option<EstimatorId>,
) -> Result<Option<FunctionalProgramWire>, CausalError> {
    let wire = |program: &antecedent_expr::FunctionalProgram| {
        functional_program_to_wire(program).map_err(|error| io_err(&error))
    };
    let derived = |estimator: EstimatorId| {
        cached
            .map(|identification| {
                let estimand =
                    crate::strategy_table::select_estimand(identification, estimator)?;
                let program_schema = antecedent_expr::ProgramSchema::new(
                    data_schema(&study.data).variables().iter().map(|variable| {
                        (
                            variable.id,
                            antecedent_expr::ProgramVariable { name: Arc::clone(&variable.name) },
                        )
                    }),
                );
                let program = antecedent_expr::FunctionalProgram::new(
                    identification.arena.clone(),
                    program_schema,
                    estimand.functional,
                    estimand.functional,
                    antecedent_expr::ProgramLimits::default(),
                )?;
                wire(&program)
            })
            .transpose()
    };
    match selected {
        Some(EstimatorId::FunctionalEffect) => match binding {
            CheckedProgramBinding::FunctionalEffect(program) => wire(program).map(Some),
            // A path-restricted effect executed through the lowered route retains no
            // operation of its own; its program is the identified functional.
            CheckedProgramBinding::None
                if matches!(
                    study.query,
                    CausalQuery::PathSpecific(_) | CausalQuery::NestedCounterfactual(_)
                ) =>
            {
                derived(EstimatorId::FunctionalEffect)
            }
            _ => Ok(None),
        },
        Some(EstimatorId::FunctionalDistribution) => match binding {
            CheckedProgramBinding::Distribution(program) => wire(program).map(Some),
            _ => derived(EstimatorId::FunctionalDistribution),
        },
        _ => Ok(None),
    }
}
