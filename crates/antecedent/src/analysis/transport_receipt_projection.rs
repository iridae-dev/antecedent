//! Project a transport [`StudyResult`] onto a [`TransportedResponse`].
//!
//! A transport result becomes a transported response only when the receipt
//! fields are already on it. Missing fields refuse. The function computes no
//! estimate and writes no identity.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, CausalSchema, IdentityDomain, SupportStatus, TransportQuery, VariableId,
};
use antecedent_estimate::EffectEstimate;
use antecedent_identify::TransportIdentification;

use super::{
    TransportDeclaration, TransportedRefusal, TransportedResponse, TransportedRoute,
    assumption_label,
};
use crate::CausalError;
use crate::analysis::builder::TransportTrialSpec;
use crate::result::{ResultPayload, StudyResult};

/// The coordinate and proof identity issued by the trial-to-target executor.
#[derive(Clone, Debug)]
pub(crate) struct TrialTransportReceipt {
    source_population: String,
    quantity: ScientificQuantity,
    transport_identity: String,
}

impl TrialTransportReceipt {
    pub(crate) fn from_execution(
        schema: &CausalSchema,
        (treatment, outcome): (VariableId, VariableId),
        query: &TransportQuery,
        premises: (&antecedent_graph::SelectionDiagram, &TransportTrialSpec),
        proof: &TransportIdentification,
    ) -> Result<Self, CausalError> {
        let (diagram, trial) = premises;
        let treatment = &schema.get(treatment)?.name;
        let outcome = schema.get(outcome)?;
        let transport = antecedent_io::transport_identity(
            Some(diagram),
            Some((trial.trial, trial.selection_probability, trial.treatment_probability)),
        )?;
        let transport_identity = antecedent_io::digest_wire(
            IdentityDomain::TransportCertificate,
            &(transport, antecedent_io::transport_identification_to_wire(proof)),
        )?
        .to_hex();
        Ok(Self {
            source_population: query.source_population.to_string(),
            quantity: ScientificQuantity {
                variable_id: outcome.name.to_string(),
                variable_name: outcome.name.to_string(),
                role: QuantityRole::Outcome,
                units: outcome.unit.as_deref().unwrap_or("undeclared").to_owned(),
                population_id: query.target_population.to_string(),
                regime_id: format!("contrast:do({treatment}=1)-do({treatment}=0)"),
                horizon: 0,
                functional_id: "mean_difference".to_owned(),
                conditioning: Vec::new(),
                transform_id: "identity".to_owned(),
            },
            transport_identity,
        })
    }
}

fn transport_point(
    result: &StudyResult,
) -> Result<(&TransportQuery, f64, &EffectEstimate), TransportedRefusal> {
    let ResultPayload::Transport(transported) = result.payload() else {
        return Err(TransportedRefusal::NotTransported);
    };
    let CausalQuery::Transport(query) = &result.identification.query else {
        return Err(TransportedRefusal::NotTransported);
    };
    let Some(effect) = result.estimate().as_effect() else {
        return Err(TransportedRefusal::InvalidValue("transported point"));
    };
    if !transported.ipw.is_finite() || effect.ate.to_bits() != transported.ipw.to_bits() {
        return Err(TransportedRefusal::InvalidValue("transported point"));
    }
    Ok((query, transported.ipw, effect))
}

// not_scientific: this reads a receipt the result already carries and wraps it
// for evaluate_transported. The wrapper identity is the receipt's canonical identity.
/// Project a trial-to-target result carrying an execution receipt onto a
/// [`TransportedResponse`].
///
/// The engine records the coordinate and transport proof identity at execution.
/// The result supplies the executed program, contract, snapshot, estimator, and
/// uncalibrated analytic standard error. The projected point is extrapolative.
///
/// # Errors
///
/// [`TransportedRefusal::NotTransported`] when the result is not a transport.
/// [`TransportedRefusal::TransportIdentityIncomplete`] when required execution
/// receipt data is absent. Other refusals cover inconsistent points, invalid
/// uncertainty, or a receipt that fails [`TransportedResponse::declared_point`].
pub fn project_transported_response(
    result: &StudyResult,
) -> Result<TransportedResponse, TransportedRefusal> {
    let (query, ipw, effect) = transport_point(result)?;
    let (Some(receipt), Some(source_identity), Some(contract), Some(provider_id)) = (
        &result.trial_transport_receipt,
        result.lowered_program_identity(),
        result.executed_contract(),
        result.logical_plan.estimator.as_deref().filter(|id| !id.is_empty()),
    ) else {
        return Err(TransportedRefusal::TransportIdentityIncomplete);
    };
    if receipt.source_population != query.source_population.as_ref()
        || receipt.quantity.population_id != query.target_population.as_ref()
    {
        return Err(TransportedRefusal::InvalidValue("transported population"));
    }
    let assumptions: Vec<_> = effect
        .provenance
        .assumptions
        .entries
        .iter()
        .map(|record| assumption_label(&record.assumption))
        .collect();
    if assumptions.is_empty() {
        return Err(TransportedRefusal::TransportIdentityIncomplete);
    }
    let declaration = TransportDeclaration {
        route: TransportedRoute::Declared,
        source_identity: source_identity.to_hex(),
        source_populations: vec![receipt.source_population.clone()],
        target_population: receipt.quantity.population_id.clone(),
        transport_identities: vec![receipt.transport_identity.clone()],
        assumptions,
        support: SupportStatus::Extrapolative,
        uncalibrated_standard_error: Some(effect.uncertainty.analytic.se),
        provider_id: provider_id.to_owned(),
        snapshot_id: contract.identities.data_snapshot.to_hex(),
        causal_contract_id: contract
            .identities
            .program
            .unwrap_or(contract.identities.identification)
            .to_hex(),
    };
    TransportedResponse::declared_point(vec![receipt.quantity.clone()], vec![ipw], &declaration)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec,
        ResponseFunctional, ResponseQuery, SupportStatus, TransportQuery, VariableId,
    };
    use antecedent_data::TabularData;
    use antecedent_design::composition_boundary::UnsupportedActionPolicy;
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy,
        UtilityExpr,
    };
    use antecedent_graph::{Admg, Dag, DenseNodeId};

    use super::super::{
        AdmissibilityRules, AdmissibleDecisionContract, BoundDecisionContract, RefuteSuite, Study,
        SupportPolicy, TransportDeclaration, TransportTrialSpec, TransportedDemand,
        TransportedRefusal, TransportedResponse, TransportedRoute, assumption_label,
    };
    use super::project_transported_response;

    fn on_large_stack(run: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("transport-receipt-projection".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(run)
            .expect("spawn projection test")
            .join()
            .expect("projection test panicked");
    }

    fn transport_query() -> TransportQuery {
        let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        });
        TransportQuery::new(response, "trial", "target", [VariableId::from_raw(0)])
    }

    fn trial_study() -> Study {
        let data = TabularData::from_f64_columns([
            ("a", &[1.0, 0.0, 0.0, 0.0][..]),
            ("y", &[3.0, 1.0, 0.0, 0.0][..]),
            ("trial", &[1.0, 1.0, 0.0, 0.0][..]),
            ("s", &[0.5, 0.5, 0.5, 0.5][..]),
            ("e", &[0.5, 0.5, 0.5, 0.5][..]),
        ])
        .unwrap();
        let mut admg = Admg::with_variables(5);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        Study::tabular(data)
            .graph(admg)
            .query(CausalQuery::Transport(transport_query()))
            .selection_targets(Arc::<[VariableId]>::from([]))
            .transport_trial(TransportTrialSpec {
                trial: VariableId::from_raw(2),
                selection_probability: VariableId::from_raw(3),
                treatment_probability: VariableId::from_raw(4),
            })
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
    }

    fn ate_result() -> crate::StudyResult {
        let data = TabularData::from_f64_columns([
            ("t", &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0][..]),
            ("y", &[0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0][..]),
        ])
        .unwrap();
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        Study::tabular(data)
            .graph(dag)
            .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            )))
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(7))
            .unwrap()
    }

    fn quantity(target: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "y".into(),
            role: QuantityRole::Outcome,
            units: "undeclared".into(),
            population_id: target.into(),
            regime_id: "contrast:do(a=1)-do(a=0)".into(),
            horizon: 0,
            functional_id: "mean_difference".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn declaration_of(result: &crate::StudyResult, ipw_se: f64) -> TransportDeclaration {
        let effect = result.estimate().as_effect().unwrap();
        let contract = result.executed_contract().unwrap();
        let causal_contract_id = match contract.identities.program {
            Some(program) => program.to_hex(),
            None => contract.identities.identification.to_hex(),
        };
        TransportDeclaration {
            route: TransportedRoute::Declared,
            source_identity: result.lowered_program_identity().unwrap().to_hex(),
            source_populations: vec!["trial".into()],
            target_population: "target".into(),
            transport_identities: vec![
                result.trial_transport_receipt.as_ref().unwrap().transport_identity.clone(),
            ],
            assumptions: effect
                .provenance
                .assumptions
                .entries
                .iter()
                .map(|record| assumption_label(&record.assumption))
                .collect(),
            support: SupportStatus::Extrapolative,
            uncalibrated_standard_error: Some(ipw_se),
            provider_id: result.logical_plan.estimator.as_deref().unwrap().to_owned(),
            snapshot_id: contract.identities.data_snapshot.to_hex(),
            causal_contract_id,
        }
    }

    fn action(id: &str, quantity: ScientificQuantity, utility: UtilityExpr) -> DecisionAction {
        DecisionAction {
            id: id.into(),
            kind: ActionKind::Intervention,
            inputs: vec![quantity],
            utility,
        }
    }

    #[allow(
        clippy::float_cmp,
        clippy::too_many_lines,
        reason = "ipw bits and receipt refusals share one trial"
    )]
    fn project_the_trial() {
        let ate = ate_result();
        assert_eq!(
            project_transported_response(&ate).unwrap_err(),
            TransportedRefusal::NotTransported,
        );

        let study = trial_study();
        let result = study.run(&ExecutionContext::for_tests(11)).unwrap();
        let mut incomplete = result.clone();
        incomplete.trial_transport_receipt = None;
        assert_eq!(
            project_transported_response(&incomplete).unwrap_err(),
            TransportedRefusal::TransportIdentityIncomplete,
        );

        let effect = result.estimate().as_effect().unwrap();
        let ipw = match result.payload() {
            crate::ResultPayload::Transport(transported) => transported.ipw,
            other => panic!("transport payload, got {other:?}"),
        };
        let projected = project_transported_response(&result).unwrap();
        let declared = TransportedResponse::declared_point(
            vec![quantity("target")],
            vec![ipw],
            &declaration_of(&result, effect.uncertainty.analytic.se),
        )
        .unwrap();
        assert_eq!(projected, declared);
        assert_eq!(
            projected.receipt().canonical_identity(),
            declared.receipt().canonical_identity(),
        );

        let coordinate = projected.coordinates().into_iter().next().expect("coordinate");
        let contract = DecisionContract {
            actions: vec![
                action("treat", coordinate.clone(), UtilityExpr::Input(0)),
                action(
                    "wait",
                    coordinate,
                    UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
                ),
            ],
            utility_units: "utility".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        };
        let bound = BoundDecisionContract::bind(
            &study.inspect().unwrap(),
            AdmissibleDecisionContract { contract, rules: AdmissibilityRules::default() },
        )
        .unwrap();
        let decided = bound
            .evaluate_transported(
                &projected,
                TransportedDemand::ExpectedUtility,
                &SupportPolicy {
                    unsupported: UnsupportedActionPolicy::RequireAllActions,
                    weakest_support: SupportStatus::Extrapolative,
                },
            )
            .unwrap();
        assert_eq!(
            decided.receipt().canonical_identity(),
            projected.receipt().canonical_identity(),
        );
    }

    #[test]
    fn trial_execution_issues_a_receipt_for_transported_decisions() {
        on_large_stack(project_the_trial);
    }
}
