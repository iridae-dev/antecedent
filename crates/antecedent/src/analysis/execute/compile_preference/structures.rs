//! The structures a decision is compiled under, and the decision across them.
//!
//! Each structure is compiled with the decision in layer four, its identification read from
//! the plan, and prepared. At execution every prepared structure is estimated once; its
//! response is retained beside its weight, and the unidentified and unevaluated mass are
//! reported, never renormalized.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, IdentificationStatus, ResponseIdentification};
use antecedent_design::preference_decision::PreferenceMeasure;
use antecedent_design::preference_evaluation::PreferenceEvaluationOptions;

use super::{
    BoundStudy, CompiledPreferenceDecision, CompiledPreferenceEvaluation,
    PreferenceDecisionExecution, PreferenceRouteError, no_response, retained_response,
    unsupported_query,
};
use crate::analysis::lowering::CompileGoal;
use crate::analysis::preference_decision::{
    decide_across_structures_with_context, structure_claims,
};
use crate::planner::plan_view::IdentificationKind;
use crate::planner::{PlanStage, PlanView};
use crate::result::{StructuralResponseAtom, StructuralResponseMixture, StructuralWeightBasis};
use crate::{CausalError, PreparedStudy, Study};

/// How identification was established for one structure.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureIdentification {
    /// Identified at compile with this status and retained with the program.
    Identified(IdentificationStatus),
    /// The program identifies on each execution; preparation at compile identified it.
    IdentifiedAtPreparation,
    /// Not identified under this structure, with the compiler's reason.
    NotIdentified(String),
}

/// One structure the decision is compiled under.
#[derive(Clone, Debug)]
pub struct CompiledStructure {
    pub(super) key: Option<u64>,
    /// Key of the equivalence-class atom a completion belongs to.
    pub(super) class: Option<u64>,
    pub(super) weight: f64,
    pub(super) source: usize,
    pub(super) identification: StructureIdentification,
    pub(super) plan: Option<PlanView>,
    pub(super) runner: Option<PreparedStudy>,
}

impl CompiledStructure {
    /// Graph key of a graph-posterior structure; `None` for a study's single structure.
    #[must_use]
    pub const fn key(&self) -> Option<u64> {
        self.key
    }

    /// Share of the posterior mass on this structure; one for a single structure. For a
    /// completion of an equivalence-class atom it is the class weight split evenly over the
    /// class's completions: an enumeration weight, not a probability.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// Graph key of the equivalence-class atom this structure completes; `None` for a
    /// structure that is not a completion.
    #[must_use]
    pub const fn class_key(&self) -> Option<u64> {
        self.class
    }

    /// Index of the study the structure belongs to.
    #[must_use]
    pub const fn source(&self) -> usize {
        self.source
    }

    /// How identification was established.
    #[must_use]
    pub const fn identification(&self) -> &StructureIdentification {
        &self.identification
    }

    /// The compiled plan of this structure, with the decision in layer four; `None` when the
    /// structure did not compile because the quantity is not identified under it.
    #[must_use]
    pub const fn plan_view(&self) -> Option<&PlanView> {
        self.plan.as_ref()
    }
}

const fn point_identified(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
    )
}

/// A structure that compiled and prepared, or why the quantity is not identified under it.
pub(super) type StructureOutcome =
    Result<(PlanView, StructureIdentification, PreparedStudy), String>;

/// Compile `study` with the decision in layer four, read its identification from the plan
/// and prepare it (or take the caller's prepared handle). An identification refusal is the
/// structure's outcome; any other refusal ends the compile.
pub(super) fn compile_structure(
    study: &Study,
    goal: &CompileGoal,
    prepared: Option<&PreparedStudy>,
    ctx: &ExecutionContext,
) -> Result<StructureOutcome, PreferenceRouteError> {
    let plan = match study.compile_for_goal(goal, ctx) {
        Ok(plan) => plan,
        Err(error) if error.is_identification_refusal() => return Ok(Err(error.to_string())),
        Err(error) => return Err(error.into()),
    };
    let view =
        PlanView::of_plan(&plan, PlanStage::Compiled, Some(study), Some(ctx.kernel_policy), None)?;
    let identification = view.identification();
    let identified = match identification.kind() {
        IdentificationKind::Retained => match identification.status() {
            Some(status) if point_identified(status) => StructureIdentification::Identified(status),
            Some(status) => {
                return Ok(Err(format!("identification status {}", status.as_str())));
            }
            None => StructureIdentification::IdentifiedAtPreparation,
        },
        IdentificationKind::Deferred => {
            return Ok(Err(identification.reason().unwrap_or("deferred").to_owned()));
        }
        IdentificationKind::PerExecution => StructureIdentification::IdentifiedAtPreparation,
        IdentificationKind::NonGraphical | IdentificationKind::NotClaimed => {
            return Err(unsupported_query("a program without a graphical identification"));
        }
    };
    let runner = match prepared {
        Some(prepared) => prepared.clone(),
        None => match study.prepare(ctx) {
            Ok(prepared) => prepared,
            Err(error) if error.is_identification_refusal() => {
                return Ok(Err(error.to_string()));
            }
            Err(error) => return Err(error.into()),
        },
    };
    Ok(Ok((view, identified, runner)))
}

/// A cancellation or resource refusal ends the run; any other estimation failure leaves a
/// structure unevaluated.
fn fatal(error: &CausalError) -> bool {
    matches!(error.peeled(), CausalError::Cancelled { .. } | CausalError::Resource { .. })
}

impl CompiledPreferenceDecision {
    pub(super) fn execute_structures(
        &self,
        studies: &[BoundStudy],
        (structures, basis): (&[CompiledStructure], StructuralWeightBasis),
        measure: Option<&PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        ctx: &ExecutionContext,
    ) -> Result<PreferenceDecisionExecution, PreferenceRouteError> {
        let study = studies.first().ok_or_else(no_response)?;
        let mut atoms = Vec::with_capacity(structures.len());
        let mut programs = Vec::with_capacity(structures.len());
        let (mut identified, mut unidentified, mut unevaluable) = (0.0, 0.0, 0.0);
        for structure in structures {
            let graph_key = structure.key.unwrap_or_default();
            let weight = structure.weight;
            let Some(runner) = &structure.runner else {
                unidentified += weight;
                atoms.push(structural_atom(graph_key, weight, None));
                continue;
            };
            programs.extend(structure.plan.as_ref().map(PlanView::identity));
            match runner.estimate_retained(ctx) {
                Ok(result) => {
                    if let Some(response) = retained_response(&result) {
                        identified += weight;
                        atoms.push(structural_atom(graph_key, weight, Some(response)));
                    } else {
                        unevaluable += weight;
                        atoms.push(unevaluated_atom(graph_key, weight));
                    }
                }
                Err(error) if fatal(&error) => return Err(error.into()),
                Err(_) => {
                    unevaluable += weight;
                    atoms.push(unevaluated_atom(graph_key, weight));
                }
            }
        }
        let mixture = StructuralResponseMixture {
            weight_basis: basis,
            atoms,
            identified_mass: identified,
            unidentified_mass: unidentified,
            unevaluable_mass: unevaluable,
            subsampled_out_mass: 0.0,
            identified_set: None,
            identified_set_interval: None,
            conditional_on_identified: None,
            full_mass_scope: true,
            truncated_atoms: 0,
            aggregation_policy: None,
        };
        let claims = structure_claims(
            &mixture,
            &[],
            &study.schema,
            &study.context(ctx),
            self.contract.structural_policy,
        )?;
        let evaluation =
            decide_across_structures_with_context(&self.contract, &claims, measure, options, ctx)?;
        Ok(PreferenceDecisionExecution {
            evaluation: CompiledPreferenceEvaluation::Structures(evaluation),
            source: None,
            inputs: Vec::new(),
            mixture: Some(mixture),
            receipt: self.receipt(programs),
        })
    }
}

fn structural_atom(
    graph_key: u64,
    weight: f64,
    response: Option<antecedent_core::CausalResponse>,
) -> StructuralResponseAtom {
    let Some(response) = response else {
        return StructuralResponseAtom {
            graph_key,
            weight,
            status: IdentificationStatus::NotIdentified,
            value: None,
            posterior: None,
            response: None,
        };
    };
    let value = match &response.estimate {
        ResponseIdentification::PointIdentified(value) => Some(value.clone()),
        _ => None,
    };
    StructuralResponseAtom {
        graph_key,
        weight,
        status: response.identification_status,
        value,
        posterior: None,
        response: Some(response),
    }
}

/// An identified structure whose estimate did not complete: it carries no response, so the
/// decision reports its mass as unevaluated.
const fn unevaluated_atom(graph_key: u64, weight: f64) -> StructuralResponseAtom {
    StructuralResponseAtom {
        graph_key,
        weight,
        status: IdentificationStatus::NonparametricallyIdentified,
        value: None,
        posterior: None,
        response: None,
    }
}
