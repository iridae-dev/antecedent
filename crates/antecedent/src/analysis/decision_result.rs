//! Apply a decision contract to a study result that was already estimated.
//!
//! The static average effect is read as one nonparametrically identified mean on the
//! observed sample: no effect modifiers and no structural response. A response uses the
//! native claim's mean adapter. Anything else is refused. Nothing here estimates.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{ExternalRefusal, ProgramBinding, QuantityRole, ScientificQuantity};
use antecedent_core::{
    CanonicalWriter, CausalResponse, CausalSchema, ExecutionContext, ResponseFunctional,
    ResponseUncertainty, SemanticDigest, VariableId,
};
use antecedent_design::decision_contract::{
    DecisionContract, DecisionContractError, SourceRepresentation, SourceRequirement,
};
use antecedent_design::decision_eval::DecisionEvalError;
use antecedent_io::distribution_artifact::DistributionCalibration;

use super::claim_decision_programs::{ContractSource, TerminalError, compile_contract_decision};
use super::execute::CompiledPreferenceEvaluation;
use super::native_claims::{NativeDecisionSource, NativeResponseClaim, NativeResponseContext};
use crate::result::{ExecutedContract, ResultPayload, StudyResult};

mod multi_arm;
mod static_effect;
pub use multi_arm::{
    MultiArmDecisionAuthority, MultiArmMeanProjection, PROVIDER_ID as MULTI_ARM_PROVIDER_ID,
    RNG_ID as MULTI_ARM_RNG_ID, project_multi_arm_means,
};
#[doc(hidden)]
pub use static_effect::{NativeEffectAuthority, StaticEffectSource, project_static_effect};

const ONE_EFFECT_DECLARATION: &str = "effect evaluation needs one declared unit/population";
const ONE_RESPONSE_DECLARATION: &str = "analysis-result evaluation needs one declared outcome/unit/population in the contract, or an explicit program= binding";
const TREATMENT_NAME: &str = "original treatment identity unavailable";
const OUTCOME_NAME: &str = "original outcome identity unavailable";
const STATIC_EXPECTED: &str = "unchanged original static mean ATE";
const STATIC_SUPPLIED: &str = "scalar analysis";
const STATIC_REMEDY: &str = "use the original checked ATE contrast";
const CHILD_IDENTITY_TAG: &str = "antecedent.decision.supplied_child.v1";

/// Why applying a contract to a result refused.
#[derive(Clone, Debug)]
pub enum DecisionProjectionError {
    /// Units or population are not the declaration the effect projection accepts.
    Declaration(&'static str),
    /// The authority check or an evaluator refused.
    Refusal(ExternalRefusal),
    /// The compiled decision refused.
    Compiler(crate::CausalError),
}

impl DecisionProjectionError {
    /// The structured refusal, when the projection refused rather than the declaration.
    #[must_use]
    pub fn refusal(&self) -> Option<&ExternalRefusal> {
        match self {
            Self::Refusal(refusal) => Some(refusal),
            Self::Declaration(_) | Self::Compiler(_) => None,
        }
    }
}

impl std::fmt::Display for DecisionProjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declaration(message) => formatter.write_str(message),
            Self::Refusal(refusal) => write!(formatter, "{}: {}", refusal.code, refusal.detail),
            Self::Compiler(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DecisionProjectionError {}

impl From<ExternalRefusal> for DecisionProjectionError {
    fn from(refusal: ExternalRefusal) -> Self {
        Self::Refusal(refusal)
    }
}

impl From<DecisionEvalError> for DecisionProjectionError {
    fn from(error: DecisionEvalError) -> Self {
        Self::Refusal(error.to_refusal())
    }
}

impl From<DecisionContractError> for DecisionProjectionError {
    fn from(error: DecisionContractError) -> Self {
        Self::Refusal(error.to_refusal())
    }
}

/// Ordered identity of a compiled child and of a result offered in its place.
///
/// The three texts are the lowered program, the data snapshot, and the execution
/// digest, each already hex (or `none` when a program digest is absent). Order is
/// the identity: swapping two texts is a different child.
#[must_use]
pub fn supplied_child_identity(program: &str, snapshot: &str, execution: &str) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(CHILD_IDENTITY_TAG);
    writer.text(program);
    writer.text(snapshot);
    writer.text(execution);
    SemanticDigest::from_bytes(writer.finish())
}

/// Apply `contract` to `result`.
///
/// A static average effect is admitted only as a nonparametrically identified mean
/// on the observed sample, with no effect modifiers and no structural response. A
/// response is read through [`NativeResponseClaim`]. Every other payload is refused.
/// The evaluators are the existing contract evaluators; this function does not estimate.
///
/// # Errors
/// The same declaration failures as the Python effect projection, the same authority
/// refusals as `effect_source_json` and `native_claim`, and the evaluators' own refusals.
#[allow(
    clippy::result_large_err,
    reason = "ExternalRefusal is the structured refusal of the external contract, returned by value on a cold once-per-decision refusal path"
)]
pub fn apply_decision(
    contract: &DecisionContract,
    result: &StudyResult,
    schema: &CausalSchema,
    execution: &ExecutionContext,
) -> Result<CompiledPreferenceEvaluation, DecisionProjectionError> {
    let source = if result.response().is_some() {
        response_source(contract, result, schema, execution)?
    } else if matches!(result.payload(), ResultPayload::Scalar)
        && result.identification.average_effect().is_some()
    {
        let (units, population) = input_declaration(contract, ONE_EFFECT_DECLARATION)?;
        let authority = NativeEffectAuthority::from_result(result);
        let projected =
            project_static_effect(authority.as_ref(), units, population, result.effect(), |id| {
                schema.get(id).ok().map(|variable| variable.name.to_string())
            })?;
        ContractSource::Means(projected.source)
    } else {
        return Err(static_refusal(
            antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND,
        )
        .into());
    };
    let compiled = compile_contract_decision(contract.clone(), source, execution)
        .map_err(DecisionProjectionError::Compiler)?;
    let provenance = result.method_provenance();
    let decided = compiled.execute_with_evidence(execution, Some(&provenance)).map_err(
        |error| match error {
            TerminalError::Refused(error) => DecisionProjectionError::Compiler(error),
            TerminalError::Kernel(error) => DecisionProjectionError::from(error),
        },
    )?;
    Ok(CompiledPreferenceEvaluation::Ordinary(Box::new(decided)))
}

pub(super) fn causal_source_identity(executed: &ExecutedContract) -> String {
    executed.identities.program.unwrap_or(executed.identities.identification).to_hex()
}

pub(super) fn mean_contrast_quantity(
    treatment: &str,
    outcome: &str,
    units: &str,
    population: &str,
    control: f64,
    active: f64,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: outcome.to_owned(),
        variable_name: outcome.to_owned(),
        role: QuantityRole::Outcome,
        units: units.to_owned(),
        population_id: population.to_owned(),
        regime_id: format!("contrast:do({treatment}={active})-do({treatment}={control})"),
        horizon: 0,
        functional_id: "mean_difference".to_owned(),
        conditioning: Vec::new(),
        transform_id: "identity".to_owned(),
    }
}

fn response_source(
    contract: &DecisionContract,
    result: &StudyResult,
    schema: &CausalSchema,
    execution: &ExecutionContext,
) -> Result<ContractSource, DecisionProjectionError> {
    let response = result.response().expect("the caller checked for a response");
    let context = response_context(contract, result, response, schema, execution)?;
    let claim = NativeResponseClaim::from_response(response, schema, &context)?;
    let requirement = contract.source_requirement()?.unwrap_or(SourceRequirement {
        any_of: vec![SourceRepresentation::Mean],
        sampled_needs_error_receipt: false,
    });
    let source = claim.decision_source(&requirement)?.source;
    Ok(match source {
        NativeDecisionSource::Mean(means) => ContractSource::Means(means),
        NativeDecisionSource::JointLaw(law) => ContractSource::Law(*law),
    })
}

fn response_context(
    contract: &DecisionContract,
    result: &StudyResult,
    response: &CausalResponse,
    schema: &CausalSchema,
    execution: &ExecutionContext,
) -> Result<NativeResponseContext, DecisionProjectionError> {
    let Some(executed) = result.executed_contract() else {
        return Err(static_refusal(
            antecedent_core::reason_details::native_claims::NATIVE_STATE_UNAVAILABLE,
        )
        .into());
    };
    let ResponseFunctional::MeanCurve { outcome, treatment } = &response.estimand else {
        return Err(static_refusal(
            antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND,
        )
        .into());
    };
    let doses = treatment.grid.values().map_err(|_| {
        DecisionProjectionError::from(bind_refusal(
            antecedent_core::reason_details::native_claims::GRID_MISMATCH,
            "a valid dose grid",
            "invalid",
        ))
    })?;
    let (units, population) = input_declaration(contract, ONE_RESPONSE_DECLARATION)?;
    let outcome_name = variable_name(schema, *outcome, OUTCOME_NAME)?;
    let treatment_name = variable_name(schema, treatment.variable, TREATMENT_NAME)?;
    let dose_units = schema
        .get(treatment.variable)
        .ok()
        .and_then(|variable| variable.unit.as_deref())
        .unwrap_or("native_numeric_scale")
        .to_owned();
    let graph_id = format!("graph:{}", executed.identities.identification);
    let contract_id = crate::io::external_binding_wire::native_contract_identity(
        &graph_id,
        response.identification_status.as_str(),
        None,
    )
    .map_err(|_| {
        DecisionProjectionError::from(static_refusal(
            antecedent_core::reason_details::native_claims::CONTRACT_MISMATCH,
        ))
    })?;
    let program = ProgramBinding {
        graph_id,
        contract_id,
        treatment_id: treatment_name,
        outcome_id: outcome_name,
        population_id: population.to_owned(),
        intervention_kind: "do".to_owned(),
        horizon: contract.horizon,
        dose_grid: doses,
        dose_units,
        outcome_units: units.to_owned(),
        functional_id: "mean".to_owned(),
        transform_id: "identity".to_owned(),
    };
    let calibration = if matches!(response.uncertainty, ResponseUncertainty::None) {
        DistributionCalibration::PointOnly
    } else {
        DistributionCalibration::NotMeasured
    };
    Ok(NativeResponseContext {
        program,
        snapshot_id: executed.identities.data_snapshot.to_hex(),
        rng_id: format!("native:seed:{}", execution.rng.master_seed()),
        calibration,
    })
}

fn input_declaration<'a>(
    contract: &'a DecisionContract,
    detail: &'static str,
) -> Result<(&'a str, &'a str), DecisionProjectionError> {
    let mut pairs = contract
        .actions
        .iter()
        .flat_map(|action| &action.inputs)
        .map(|input| (input.units.as_str(), input.population_id.as_str()));
    let pair = pairs.next().ok_or(DecisionProjectionError::Declaration(detail))?;
    if pairs.any(|other| other != pair) {
        return Err(DecisionProjectionError::Declaration(detail));
    }
    Ok(pair)
}

fn variable_name(
    schema: &CausalSchema,
    id: VariableId,
    missing: &'static str,
) -> Result<String, DecisionProjectionError> {
    schema
        .get(id)
        .map(|variable| variable.name.to_string())
        .map_err(|_| DecisionProjectionError::Declaration(missing))
}

fn static_refusal(detail: &'static str) -> ExternalRefusal {
    let code = if detail == antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND {
        antecedent_core::reason_code!("route_not_supported")
    } else {
        antecedent_core::reason_code!("invalid_argument")
    };
    ExternalRefusal::new(code, "bind", detail)
        .with_expected(STATIC_EXPECTED)
        .with_supplied(STATIC_SUPPLIED)
        .with_remedy(STATIC_REMEDY)
}

fn bind_refusal(detail: &'static str, expected: &str, supplied: &str) -> ExternalRefusal {
    ExternalRefusal::new(antecedent_core::reason_code!("invalid_argument"), "bind", detail)
        .with_expected(expected)
        .with_supplied(supplied)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::execute::CompiledPreferenceEvaluation;
    use crate::{RefuteSuite, Study};
    use antecedent_contract::{QuantityRole, ScientificQuantity, dose_label};
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, ExecutionContext, GridSpec, Intervention,
        ResponseFunctional, ResponseQuery, Value, VariableId,
    };
    use antecedent_data::{TableView, TabularData};
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy,
        UtilityExpr,
    };
    use antecedent_design::decision_eval::Verdict;
    use antecedent_graph::{Dag, DenseNodeId};

    use super::supplied_child_identity;

    #[test]
    fn supplied_child_identity_orders_program_snapshot_and_execution() {
        let forward = supplied_child_identity("program", "snapshot", "execution");
        let swapped = supplied_child_identity("snapshot", "program", "execution");
        assert_ne!(forward, swapped);
        assert_eq!(forward, supplied_child_identity("program", "snapshot", "execution"));
    }

    fn binary_table() -> TabularData {
        TabularData::from_f64_columns([
            ("a", &[0.0, 0.0, 1.0, 1.0][..]),
            ("y", &[0.0, 0.0, 2.0, 2.0][..]),
        ])
        .unwrap()
    }

    fn ate_study(data: &TabularData) -> Study {
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        Study::tabular(data.clone())
            .graph(graph)
            .query(CausalQuery::AverageEffect(AverageEffectQuery::new(
                treatment,
                outcome,
                Arc::<[VariableId]>::from([]),
                Intervention::set(treatment, Value::f64(0.0)),
                Intervention::set(treatment, Value::f64(1.0)),
                antecedent_core::TargetPopulation::AllObserved,
            )))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    }

    fn contrast(units: &str, population: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "y".into(),
            role: QuantityRole::Outcome,
            units: units.into(),
            population_id: population.into(),
            regime_id: "contrast:do(a=1)-do(a=0)".into(),
            horizon: 0,
            functional_id: "mean_difference".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn effect_contract(
        quantity: ScientificQuantity,
        criterion: DecisionCriterion,
    ) -> DecisionContract {
        let action = |id: &str, utility: UtilityExpr| DecisionAction {
            id: id.into(),
            kind: ActionKind::Intervention,
            inputs: vec![quantity.clone()],
            utility,
        };
        DecisionContract {
            actions: vec![
                action("act", UtilityExpr::Input(0)),
                action(
                    "wait",
                    UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
                ),
            ],
            utility_units: "mmHg".into(),
            criterion,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        }
    }

    #[test]
    fn a_static_average_effect_is_the_observed_mean_contrast() {
        let data = binary_table();
        let ctx = ExecutionContext::for_tests(11);
        let result = ate_study(&data).run(&ctx).unwrap();
        let contract = effect_contract(
            contrast("mmHg", "target"),
            DecisionCriterion::PosteriorExpectedUtility,
        );
        let decided = super::apply_decision(&contract, &result, data.schema(), &ctx).unwrap();
        let CompiledPreferenceEvaluation::Ordinary(decision) = decided else {
            panic!("an ordinary contract returns its own evaluation");
        };
        assert!((result.effect() - 2.0).abs() < 1e-9, "{}", result.effect());
        assert_eq!(decision.verdict, Verdict::UniquelyOptimal("act".into()));
        assert!((decision.actions[0].expected_utility - result.effect()).abs() < 1e-12);
        assert!(decision.actions[1].expected_utility.abs() < 1e-12);
        assert!(decision.evpi.is_none());
    }

    #[test]
    fn a_changed_effect_and_a_quantile_refuse_like_the_python_projection() {
        let data = binary_table();
        let ctx = ExecutionContext::for_tests(11);
        let result = ate_study(&data).run(&ctx).unwrap();
        let schema = data.schema();
        let authority = super::NativeEffectAuthority::from_result(&result).unwrap();
        let mismatched =
            super::project_static_effect(Some(&authority), "mmHg", "target", 99.0, |id| {
                schema.get(id).ok().map(|variable| variable.name.to_string())
            })
            .unwrap_err();
        let refusal = mismatched.refusal().unwrap();
        assert_eq!(refusal.code, "invalid_argument");
        assert_eq!(
            refusal.detail,
            antecedent_core::reason_details::native_claims::PROJECTION_MISMATCH
        );
        let contract =
            effect_contract(contrast("mmHg", "target"), DecisionCriterion::Quantile { p: 0.5 });
        let refused = super::apply_decision(&contract, &result, schema, &ctx).unwrap_err();
        let refusal = refused.refusal().unwrap();
        assert_eq!(refusal.code, "decision_contract_unsatisfied");
        assert_eq!(
            refusal.detail,
            antecedent_core::reason_details::decision_evaluation::MEAN_SOURCE_INSUFFICIENT
        );
        let empty =
            effect_contract(contrast("", "target"), DecisionCriterion::PosteriorExpectedUtility);
        let message = super::apply_decision(&empty, &result, schema, &ctx).unwrap_err().to_string();
        assert_eq!(message, super::static_effect::UNITS_POPULATION);
        let other =
            effect_contract(contrast("mmHg", "other"), DecisionCriterion::PosteriorExpectedUtility);
        let message = super::apply_decision(&other, &result, schema, &ctx).unwrap_err().to_string();
        assert_eq!(message, super::static_effect::UNITS_POPULATION);
    }

    fn curve_table() -> TabularData {
        let n = 40_u32;
        let t: Vec<f64> = (0..n).map(|i| 2.0 * f64::from(i) / f64::from(n - 1)).collect();
        let y: Vec<f64> = t.iter().map(|dose| 0.5 + 1.5 * dose).collect();
        TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap()
    }

    fn response_quantity(dose: f64) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "y".into(),
            role: QuantityRole::Outcome,
            units: "mmHg".into(),
            population_id: "target".into(),
            regime_id: format!("do(t={})", dose_label(dose)),
            horizon: 0,
            functional_id: "mean".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    #[test]
    fn a_response_uses_the_native_mean_adapter_or_refuses_a_law() {
        let data = curve_table();
        let ctx = ExecutionContext::for_tests(11);
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let study = Study::tabular(data.clone())
            .graph(graph)
            .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: antecedent_core::ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from([0.5, 1.5])),
                ),
            })))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let result = study.run(&ctx).unwrap();
        let action = |id: &str, dose: f64| DecisionAction {
            id: id.into(),
            kind: ActionKind::Intervention,
            inputs: vec![response_quantity(dose)],
            utility: UtilityExpr::Input(0),
        };
        let contract = DecisionContract {
            actions: vec![action("low", 0.5), action("high", 1.5)],
            utility_units: "mmHg".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        };
        let decided = super::apply_decision(&contract, &result, data.schema(), &ctx).unwrap();
        let ordinary = decided.ordinary().unwrap();
        assert_eq!(ordinary.verdict, Verdict::UniquelyOptimal("high".into()));
        let law =
            DecisionContract { criterion: DecisionCriterion::Quantile { p: 0.5 }, ..contract };
        let refused = super::apply_decision(&law, &result, data.schema(), &ctx).unwrap_err();
        let refusal = refused.refusal().unwrap();
        assert_eq!(refusal.code, "decision_contract_unsatisfied");
        assert_eq!(
            refusal.detail,
            antecedent_core::reason_details::native_claims::SOURCE_NOT_SUPPLIED
        );
    }

    #[test]
    fn a_supplied_child_matches_or_refuses_executed_program_mismatch() {
        let data = curve_table();
        let ctx = ExecutionContext::for_tests(11);
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let study = Study::tabular(data)
            .graph(graph)
            .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: antecedent_core::ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from([0.5, 1.5])),
                ),
            })))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let action = |id: &str, dose: f64| DecisionAction {
            id: id.into(),
            kind: ActionKind::Intervention,
            inputs: vec![response_quantity(dose)],
            utility: UtilityExpr::Input(0),
        };
        let contract = DecisionContract {
            actions: vec![action("low", 0.5), action("high", 1.5)],
            utility_units: "mmHg".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        };
        let options =
            antecedent_design::preference_evaluation::PreferenceEvaluationOptions::default();
        let compiled = study.compile_for_contract_decision(&contract, &ctx).unwrap();
        let child = compiled
            .structures()
            .iter()
            .find_map(|structure| structure.prepared())
            .unwrap()
            .estimate_retained(&ctx)
            .unwrap();
        let admitted =
            compiled.execute_supplied(&[child.clone()], &ctx, None, options, &ctx).unwrap();
        let estimated = compiled.execute(None, options, &ctx).unwrap();
        assert_eq!(admitted.evaluation, estimated.evaluation);

        let other = ExecutionContext::for_tests(99);
        let mismatched = compiled
            .execute_supplied(std::slice::from_ref(&child), &other, None, options, &other)
            .unwrap_err();
        let refusal = mismatched.refusal().unwrap();
        assert_eq!(refusal.code, antecedent_core::reason_code!("route_not_lowered"));
        assert_eq!(
            refusal.detail,
            antecedent_core::reason_details::preference_compile::EXECUTED_PROGRAM_MISMATCH
        );

        let plain = study.run(&ctx).unwrap();
        let foreign = compiled.execute_supplied(&[plain], &ctx, None, options, &ctx).unwrap_err();
        let refusal = foreign.refusal().unwrap();
        assert_eq!(
            refusal.detail,
            antecedent_core::reason_details::preference_compile::EXECUTED_PROGRAM_MISMATCH
        );
    }
}
