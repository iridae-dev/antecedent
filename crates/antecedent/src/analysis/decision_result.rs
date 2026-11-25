//! Apply a decision contract to a study result that was already estimated.
//!
//! The static average effect is read as one nonparametrically identified mean on the
//! observed sample: no effect modifiers and no structural response. A response uses the
//! native claim's mean adapter. Anything else is refused. Nothing here estimates.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    ExternalRefusal, ProgramBinding, QuantityRole, ScientificQuantity,
};
use antecedent_core::{
    CanonicalWriter, CausalResponse, CausalSchema, ExecutionContext, IdentificationStatus,
    Intervention, OutcomeFunctional, ResponseFunctional, ResponseUncertainty, SemanticDigest,
    TargetPopulation, VariableId,
};
use antecedent_design::decision_contract::{
    DecisionContract, DecisionContractError, SourceRepresentation, SourceRequirement,
};
use antecedent_design::decision_eval::{
    DecisionEvalError, DecisionResult, MeanEstimate, MeanSource, evaluate_contract,
    evaluate_contract_on_means,
};
use antecedent_io::distribution_artifact::DistributionCalibration;

use super::execute::CompiledPreferenceEvaluation;
use super::native_claims::{
    NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use crate::result::{ResultPayload, StudyResult};
use crate::support::CellStatus;

const UNITS_POPULATION: &str = "effect units must be declared (at most 256 bytes) and population must be target (the original observed sample)";
const ONE_EFFECT_DECLARATION: &str = "effect evaluation needs one declared unit/population";
const ONE_RESPONSE_DECLARATION: &str =
    "analysis-result evaluation needs one declared outcome/unit/population in the contract, or an explicit program= binding";
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
}

impl DecisionProjectionError {
    /// The structured refusal, when the projection refused rather than the declaration.
    #[must_use]
    pub fn refusal(&self) -> Option<&ExternalRefusal> {
        match self {
            Self::Refusal(refusal) => Some(refusal),
            Self::Declaration(_) => None,
        }
    }
}

impl std::fmt::Display for DecisionProjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declaration(message) => formatter.write_str(message),
            Self::Refusal(refusal) => write!(formatter, "{}: {}", refusal.code, refusal.detail),
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
    let decided = if result.response().is_some() {
        evaluate_response(contract, result, schema, execution)?
    } else if matches!(result.payload(), ResultPayload::Scalar) && result.identification.average_effect().is_some()
    {
        let (units, population) = effect_declaration(contract)?;
        let source = static_effect_source(result, schema, units, population, result.effect())?;
        evaluate_contract_on_means(contract, &source)?
    } else {
        return Err(static_refusal(antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND).into());
    };
    Ok(CompiledPreferenceEvaluation::Ordinary(Box::new(decided)))
}

fn effect_declaration(contract: &DecisionContract) -> Result<(&str, &str), DecisionProjectionError> {
    let mut declared: Option<(&str, &str)> = None;
    for action in &contract.actions {
        for input in &action.inputs {
            let pair = (input.units.as_str(), input.population_id.as_str());
            match declared {
                None => declared = Some(pair),
                Some(seen) if seen == pair => {}
                Some(_) => return Err(DecisionProjectionError::Declaration(ONE_EFFECT_DECLARATION)),
            }
        }
    }
    let Some((units, population)) = declared else {
        return Err(DecisionProjectionError::Declaration(ONE_EFFECT_DECLARATION));
    };
    if units.trim().is_empty() || units.len() > 256 || population != "target" {
        return Err(DecisionProjectionError::Declaration(UNITS_POPULATION));
    }
    Ok((units, population))
}

fn static_effect_source(
    result: &StudyResult,
    schema: &CausalSchema,
    units: &str,
    population: &str,
    supplied: f64,
) -> Result<MeanSource, DecisionProjectionError> {
    let Some(query) = result.identification.average_effect() else {
        return Err(static_refusal(antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND).into());
    };
    let Some(effect) = result.estimate().as_effect() else {
        return Err(static_refusal(antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND).into());
    };
    if result.identification.status != IdentificationStatus::NonparametricallyIdentified
        || query.target_population != TargetPopulation::AllObserved
        || query.outcome_functional != OutcomeFunctional::Mean
        || !query.effect_modifiers.is_empty()
        || result.structural_response.is_some()
        || result.support_status().is_some_and(|status| status != CellStatus::Supported)
    {
        return Err(static_refusal(antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND).into());
    }
    if !effect.ate.is_finite() || supplied.to_bits() != effect.ate.to_bits() {
        return Err(static_refusal(antecedent_core::reason_details::native_claims::PROJECTION_MISMATCH).into());
    }
    let Some(executed) = result.executed_contract() else {
        return Err(
            static_refusal(antecedent_core::reason_details::native_claims::NATIVE_STATE_UNAVAILABLE).into(),
        );
    };
    let level = |intervention: &Intervention| match intervention {
        Intervention::Set { variable, value } if *variable == query.treatment => value.as_f64(),
        _ => None,
    };
    let (Some(control), Some(active)) = (level(&query.control), level(&query.active)) else {
        return Err(static_refusal(antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND).into());
    };
    let treatment = variable_name(schema, query.treatment, TREATMENT_NAME)?;
    let outcome = variable_name(schema, query.outcome, OUTCOME_NAME)?;
    let coordinate = ScientificQuantity {
        variable_id: outcome.clone(),
        variable_name: outcome.clone(),
        role: QuantityRole::Outcome,
        units: units.to_owned(),
        population_id: population.to_owned(),
        regime_id: format!("contrast:do({treatment}={active})-do({treatment}={control})"),
        horizon: 0,
        functional_id: "mean_difference".to_owned(),
        conditioning: Vec::new(),
        transform_id: "identity".to_owned(),
    };
    Ok(MeanSource {
        coordinates: vec![coordinate],
        means: vec![effect.ate],
        estimate: MeanEstimate::Frequentist,
        provider_id: "native:static_ate".to_owned(),
        snapshot_id: executed.identities.data_snapshot.to_hex(),
        causal_contract_id: executed
            .identities
            .program
            .unwrap_or(executed.identities.identification)
            .to_hex(),
        rng_id: "none:mean_grid".to_owned(),
    })
}

fn evaluate_response(
    contract: &DecisionContract,
    result: &StudyResult,
    schema: &CausalSchema,
    execution: &ExecutionContext,
) -> Result<DecisionResult, DecisionProjectionError> {
    let response = result.response().expect("the caller checked for a response");
    let context = response_context(contract, result, response, schema, execution)?;
    let claim = NativeResponseClaim::from_response(response, schema, &context)?;
    let requirement = contract.source_requirement()?.unwrap_or(SourceRequirement {
        any_of: vec![SourceRepresentation::Mean],
        sampled_needs_error_receipt: false,
    });
    let source = claim.decision_source(&requirement)?.source;
    Ok(match source {
        NativeDecisionSource::Mean(means) => evaluate_contract_on_means(contract, &means)?,
        NativeDecisionSource::JointLaw(law) => evaluate_contract(contract, &law)?,
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
        return Err(
            static_refusal(antecedent_core::reason_details::native_claims::NATIVE_STATE_UNAVAILABLE).into(),
        );
    };
    let ResponseFunctional::MeanCurve { outcome, treatment } = &response.estimand else {
        return Err(static_refusal(antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND).into());
    };
    let doses = treatment.grid.values().map_err(|_| {
        DecisionProjectionError::from(bind_refusal(
            antecedent_core::reason_details::native_claims::GRID_MISMATCH,
            "a valid dose grid",
            "invalid",
        ))
    })?;
    let (units, population) = response_declaration(contract)?;
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

fn response_declaration(contract: &DecisionContract) -> Result<(&str, &str), DecisionProjectionError> {
    let mut declared: Option<(&str, &str)> = None;
    for action in &contract.actions {
        for input in &action.inputs {
            let pair = (input.units.as_str(), input.population_id.as_str());
            match declared {
                None => declared = Some(pair),
                Some(seen) if seen == pair => {}
                Some(_) => {
                    return Err(DecisionProjectionError::Declaration(ONE_RESPONSE_DECLARATION));
                }
            }
        }
    }
    declared.ok_or(DecisionProjectionError::Declaration(ONE_RESPONSE_DECLARATION))
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

    use antecedent_contract::{QuantityRole, ScientificQuantity, dose_label};
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, ExecutionContext, GridSpec, Intervention, ResponseFunctional,
        ResponseQuery, Value, VariableId,
    };
    use antecedent_data::{TableView, TabularData};
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };
    use antecedent_design::decision_eval::Verdict;
    use antecedent_graph::{Dag, DenseNodeId};
    use super::super::execute::CompiledPreferenceEvaluation;
    use crate::{RefuteSuite, Study};

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

    fn effect_contract(quantity: ScientificQuantity, criterion: DecisionCriterion) -> DecisionContract {
        let action = |id: &str, utility: UtilityExpr| DecisionAction {
            id: id.into(),
            kind: ActionKind::Intervention,
            inputs: vec![quantity.clone()],
            utility,
        };
        DecisionContract {
            actions: vec![
                action("act", UtilityExpr::Input(0)),
                action("wait", UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0))),
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
        let contract = effect_contract(contrast("mmHg", "target"), DecisionCriterion::PosteriorExpectedUtility);
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
        let mismatched =
            super::static_effect_source(&result, schema, "mmHg", "target", 99.0).unwrap_err();
        let refusal = mismatched.refusal().unwrap();
        assert_eq!(refusal.code, "invalid_argument");
        assert_eq!(
            refusal.detail,
            antecedent_core::reason_details::native_claims::PROJECTION_MISMATCH
        );
        let contract = effect_contract(
            contrast("mmHg", "target"),
            DecisionCriterion::Quantile { p: 0.5 },
        );
        let refused = super::apply_decision(&contract, &result, schema, &ctx).unwrap_err();
        let refusal = refused.refusal().unwrap();
        assert_eq!(refusal.code, "decision_contract_unsatisfied");
        assert_eq!(
            refusal.detail,
            antecedent_core::reason_details::decision_evaluation::MEAN_SOURCE_INSUFFICIENT
        );
        let empty = effect_contract(contrast("", "target"), DecisionCriterion::PosteriorExpectedUtility);
        let message = super::apply_decision(&empty, &result, schema, &ctx).unwrap_err().to_string();
        assert_eq!(message, super::UNITS_POPULATION);
        let other = effect_contract(contrast("mmHg", "other"), DecisionCriterion::PosteriorExpectedUtility);
        let message = super::apply_decision(&other, &result, schema, &ctx).unwrap_err().to_string();
        assert_eq!(message, super::UNITS_POPULATION);
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
        let law = DecisionContract {
            criterion: DecisionCriterion::Quantile { p: 0.5 },
            ..contract
        };
        let refused = super::apply_decision(&law, &result, data.schema(), &ctx).unwrap_err();
        let refusal = refused.refusal().unwrap();
        assert_eq!(refusal.code, "decision_contract_unsatisfied");
        assert_eq!(refusal.detail, antecedent_core::reason_details::native_claims::SOURCE_NOT_SUPPLIED);
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
        let options = antecedent_design::preference_evaluation::PreferenceEvaluationOptions::default();
        let compiled = study.compile_for_contract_decision(&contract, &ctx).unwrap();
        let child = compiled
            .structures()
            .iter()
            .find_map(|structure| structure.prepared())
            .unwrap()
            .estimate_retained(&ctx)
            .unwrap();
        let admitted = compiled.execute_supplied(&[child.clone()], &ctx, None, options, &ctx).unwrap();
        let estimated = compiled.execute(None, options, &ctx).unwrap();
        assert_eq!(admitted.evaluation, estimated.evaluation);

        let other = ExecutionContext::for_tests(99);
        let mismatched = compiled
            .execute_supplied(std::slice::from_ref(&child), &ctx, None, options, &other)
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
