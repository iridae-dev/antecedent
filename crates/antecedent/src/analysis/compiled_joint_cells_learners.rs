//! Compiled factorized joint-cell fit with caller-supplied learners.
//!
//! [`Study::compile_joint_cells_learners`] binds a table, the joint treatment components, the
//! outcome, the declared adjustment set, the factorization orderings, the joint-cell
//! configuration and the declared learners into a lowered program.
//! [`CompiledJointCellsLearners::execute`] fits every prefix-stratum conditional propensity with
//! the declared binary-probability factory and every per-cell outcome model with the declared
//! outcome-regression factory through
//! `antecedent_estimate::fit_factorized_joint_cells_with_factories`, the kernel the direct entry
//! is, under the caller's execution context, and returns the kernel's fit unchanged.
//!
//! The learners are the study's caller-learner declaration ([`LearnerFactories`], what
//! [`crate::StudyBuilder::learner_factories`] declares): the outcome factory fits the per-cell
//! outcome regressions and the treatment factory the binary conditionals of the factorized
//! cell propensity. They are external code the library does not verify: the fit records them as
//! unverified (`external_nuisances_unverified`, and `trust=unverified` on the score-table
//! provenance). The adjustment set is the caller's declaration, as on the direct entry: the
//! cells are point estimates under it, the family is point-only and reports calibration
//! `not_measured`, and the run carries the declaration's own calibration label, so a record
//! measured with other learners never annotates a factory fit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect, PhysicalOperator,
    SemanticDigest, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{FactorizedJointConfig, FactorizedJointFit};

use super::declared_learners::{LearnerFactories, calibration_label};
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::supplied_identity::{Canonical, Declaration, Wire};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{ModelRole, ModelSlot, Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

const PLAN_ID: &str = "joint_cells_learner_fit.v1";
const EVALUATOR: &str = "antecedent_estimate::fit_factorized_joint_cells_with_factories";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// A factorized joint-cell request whose nuisances the caller's learners fit.
///
/// `treatments` are the binary components in cell-bit order, `adjustment` the declared
/// adjustment set, `orderings` the factorization orderings (the first is the declared one; see
/// `antecedent_estimate::orderings_for`). `learners` pairs the per-cell outcome factory
/// (`outcome`, a regression) with the conditional propensity factory (`treatment`, a binary
/// probability); its declared identity enters the program. `config.learner` must be `None`:
/// the declared learners replace the built-in learner specification.
#[derive(Clone, Debug)]
pub struct JointCellsLearnersRequest {
    data: TabularData,
    treatments: Vec<VariableId>,
    outcome: VariableId,
    adjustment: Vec<VariableId>,
    orderings: Vec<Vec<usize>>,
    config: FactorizedJointConfig,
    learners: LearnerFactories,
}

impl JointCellsLearnersRequest {
    /// Declare the table, components, outcome, adjustment set, orderings, configuration and
    /// caller learners.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "the joint query and its learners are independent inputs"
    )]
    pub fn new(
        data: TabularData,
        treatments: Vec<VariableId>,
        outcome: VariableId,
        adjustment: Vec<VariableId>,
        orderings: Vec<Vec<usize>>,
        config: FactorizedJointConfig,
        learners: LearnerFactories,
    ) -> Self {
        Self { data, treatments, outcome, adjustment, orderings, config, learners }
    }
}

/// A factorized joint-cell fit with caller learners, bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledJointCellsLearners {
    request: JointCellsLearnersRequest,
    lowered: Arc<LoweredProgram>,
}

/// The cells the bound request evaluates to, bound to the program that evaluated them.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct JointCellsLearnersRun {
    /// The cells, normalization checks, ordering sensitivity and aligned scores, as the kernel
    /// returned them.
    pub fit: FactorizedJointFit,
    /// Identity of the lowered program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `point_only`: the joint cells publish no interval.
    pub claim: &'static str,
    /// Always `not_measured`.
    pub calibration: &'static str,
    /// The calibration label of the declared learners (`learners=caller_factory:<digest of the
    /// declared identity>`): a record measured with other learners, the library's among them,
    /// never binds to it.
    pub calibration_label: String,
}

/// A 32-byte digest, written as its bytes.
struct Digest32([u8; 32]);

impl Canonical for Digest32 {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.bytes(&self.0);
    }
}

/// Digest of the table the kernel reads: its typed contents, validity, mask and weights, and
/// its schema.
fn source_digest(data: &TabularData) -> [u8; 32] {
    let storage = data.storage();
    Declaration::new("antecedent.joint_cells_caller_learners.rows.v1")
        .field("content", &Digest32(storage.content_digest()))
        .field("schema", &Digest32(antecedent_data::source::schema_digest(storage.schema())))
        .finish()
}

/// Digest of the question: components, outcome, adjustment, orderings, configuration and the
/// declared learners.
fn question_digest(request: &JointCellsLearnersRequest) -> [u8; 32] {
    let config = &request.config;
    let mut declaration = Declaration::new("antecedent.joint_cells_caller_learners.question.v1");
    declaration
        .field("treatments", &request.treatments)
        .field("outcome", &request.outcome)
        .field("adjustment", &request.adjustment)
        .field("orderings", &request.orderings)
        .field("ridge_lambdas", config.tuning.lambdas())
        .field("ridge_inner_folds", &config.tuning.inner_folds())
        .field("learner", &config.learner.as_ref().map(Wire))
        .field("folds", &config.folds)
        .field("seed", &config.seed)
        .field("clip", &config.clip)
        .field("min_cell_ess", &config.min_cell_ess)
        .field("normalization_tolerance", &config.normalization_tolerance)
        .field("ordering_tolerance_sd", &config.ordering_tolerance_sd)
        .callback("caller_learner_pair", Some(request.learners.identity()));
    declaration.finish()
}

fn lower(request: &JointCellsLearnersRequest) -> Result<Arc<LoweredProgram>, CausalError> {
    let question = question_digest(request);
    let source = source_digest(&request.data);
    let storage = request.data.storage();
    let rows = storage.row_count() as u64;
    let columns = (request.treatments.len() + request.adjustment.len() + 1) as u64;
    let design = DesignShape::quoted(rows, columns, true);
    let query_variables: Arc<[VariableId]> =
        request.treatments.iter().copied().chain(std::iter::once(request.outcome)).collect();
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::InterventionalDistribution,
            outcome: Some(request.outcome),
            query: question,
        },
        plan_id: Arc::from(PLAN_ID),
        data: DataClassification::Tabular,
        graph_class: "caller_declared_adjustment_set_for_a_joint_binary_intervention",
        query_variables,
        source_digest: Some(source),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let slot = |family: &str, output: Option<VariableId>| ModelSlot {
        role: ModelRole::Nuisance,
        family: Arc::from(family),
        inputs: Arc::from([]),
        output,
    };
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(EVALUATOR) },
        models: Arc::from([
            slot("caller_cell_outcome_regression", Some(request.outcome)),
            slot("caller_conditional_propensity", None),
        ]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: Some(source),
        }]),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let operator = |effect: PhysicalEffect, bytes: Option<u64>| {
        let declared = PhysicalOperator::declare(effect);
        let declared = match bytes {
            Some(bytes) => declared.estimates(ByteEstimates::absent().with_output(bytes)),
            None => declared,
        };
        declared.build().map_err(|_| CausalError::Compile {
            message: format!("{PLAN_ID}: a physical operator cannot be declared"),
        })
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            operator(PhysicalEffect::Read, Some(rows.saturating_mul(columns).saturating_mul(8)))?,
            operator(PhysicalEffect::Fit, None)?,
        ]),
        design,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::JointCellsLearnerFit,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::None,
                kind: Some(ClaimKind::Point),
            },
        }
        .finish(),
    ))
}

impl Study {
    /// Bind a factorized joint-cell request with caller learners.
    ///
    /// Nothing is fitted here. A built-in learner specification beside the declared learners is
    /// refused; the kernel decides every other refusal (configuration, components, orderings,
    /// columns, rows, learner tasks and capabilities, cell support) at execution, as on the
    /// direct entry.
    ///
    /// # Errors
    ///
    /// `option_not_applicable` (`joint_cells_caller_learners.built_in_learner_declared`) when
    /// the configuration also declares a built-in learner, or a physical operator that cannot be
    /// declared.
    pub fn compile_joint_cells_learners(
        request: JointCellsLearnersRequest,
    ) -> Result<CompiledJointCellsLearners, CausalError> {
        if request.config.learner.is_some() {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                detail = antecedent_core::reason_details::joint_cells_caller_learners::BUILT_IN_LEARNER_DECLARED,
                "the declared caller learners replace the built-in learner specification. Next \
                 attempt: leave FactorizedJointConfig::learner unset"
            ));
        }
        let lowered = lower(&request)?;
        Ok(CompiledJointCellsLearners { request, lowered })
    }
}

impl CompiledProgramHandle for CompiledJointCellsLearners {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledJointCellsLearners {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without evaluating.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Fit the caller learners and evaluate every joint cell.
    ///
    /// # Errors
    ///
    /// Every refusal of `fit_factorized_joint_cells_with_factories` (a built-in learner
    /// specification beside the factories, a factory without the required task or capability,
    /// configuration, component, ordering, column, row, support or resource refusals,
    /// cancellation), or the support gate once it is on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<JointCellsLearnersRun, CausalError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)?;
        }
        let request = &self.request;
        let fit = antecedent_estimate::fit_factorized_joint_cells_with_factories(
            &request.data,
            &request.treatments,
            request.outcome,
            &request.adjustment,
            &request.orderings,
            &request.config,
            &scoped,
            request.learners.outcome(),
            request.learners.treatment(),
        )?;
        Ok(JointCellsLearnersRun {
            fit,
            lowered_program_identity: self.lowered.identity(),
            claim: "point_only",
            calibration: "not_measured",
            calibration_label: calibration_label(Some(&request.learners)).unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod site_tests {
    use std::sync::Arc;

    use antecedent_core::VariableId;
    use antecedent_data::TabularData;
    use antecedent_estimate::{
        FactorizedJointConfig, LearnerSpec, LinearSpec, LogisticSpec, RidgeTuning,
    };
    use antecedent_learn::{LearnerFactory, PredictionTask, resolve_for};

    use super::{JointCellsLearnersRequest, LearnerFactories, question_digest, source_digest};
    use crate::identity_laws::{all_distinct, ordered_laws, set_laws};

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn learners() -> LearnerFactories {
        let outcome: Arc<dyn LearnerFactory> = Arc::from(
            resolve_for(LearnerSpec::Linear(LinearSpec::default()), PredictionTask::Regression)
                .expect("a regression learner"),
        );
        let propensity: Arc<dyn LearnerFactory> = Arc::from(
            resolve_for(
                LearnerSpec::Logistic(LogisticSpec::default()),
                PredictionTask::BinaryProbability,
            )
            .expect("a probability learner"),
        );
        LearnerFactories::new(outcome, propensity, "pair").expect("declared learners")
    }

    fn table(shift: f64) -> TabularData {
        let t = [0.0, 1.0, 0.0, 1.0];
        let y = t.map(|value| value + shift);
        TabularData::from_f64_columns([("t0", &t[..]), ("t1", &t[..]), ("y", &y[..])])
            .expect("a table")
    }

    fn question(
        treatments: &[VariableId],
        adjustment: &[VariableId],
        orderings: Vec<Vec<usize>>,
        tuning: RidgeTuning,
    ) -> [u8; 32] {
        question_digest(&JointCellsLearnersRequest::new(
            table(0.0),
            treatments.to_vec(),
            v(2),
            adjustment.to_vec(),
            orderings,
            FactorizedJointConfig::new(tuning),
            learners(),
        ))
    }

    fn base_orderings() -> Vec<Vec<usize>> {
        vec![vec![0, 1], vec![1, 0]]
    }

    #[test]
    fn site_compiled_joint_cells_learners_question_orders_components_and_orderings() {
        ordered_laws(&[v(0), v(1), v(3)], |treatments| {
            Some(question(treatments, &[v(4)], base_orderings(), RidgeTuning::default()))
        });
        ordered_laws(&[v(4), v(5)], |adjustment| {
            Some(question(&[v(0), v(1)], adjustment, base_orderings(), RidgeTuning::default()))
        });
        ordered_laws(&base_orderings(), |orderings| {
            Some(question(&[v(0), v(1)], &[v(4)], orderings.to_vec(), RidgeTuning::default()))
        });
        // The penalty grid is canonical by construction: `RidgeTuning::new` sorts and
        // deduplicates it.
        set_laws(&[0.1, 1.0, 10.0], |lambdas| {
            let tuning = RidgeTuning::new(lambdas, 3).ok()?;
            Some(question(&[v(0), v(1)], &[v(4)], base_orderings(), tuning))
        });
    }

    #[test]
    fn site_compiled_joint_cells_learners_source_binds_the_table() {
        let data = |shift: f64| source_digest(&table(shift));
        all_distinct(&[data(0.0), data(1.0)]);
        let renamed = TabularData::from_f64_columns([
            ("t0", &[0.0, 1.0, 0.0, 1.0][..]),
            ("t1", &[0.0, 1.0, 0.0, 1.0][..]),
            ("outcome", &[0.0, 1.0, 0.0, 1.0][..]),
        ])
        .expect("a table");
        all_distinct(&[data(0.0), source_digest(&renamed)]);
    }
}
