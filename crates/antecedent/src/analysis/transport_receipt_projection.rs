//! Project a transport [`StudyResult`] onto a [`TransportedResponse`].
//!
//! A transport result becomes a transported response only when the receipt
//! fields are already on it. Missing fields refuse. The function computes no
//! estimate and writes no identity.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{CausalQuery, SupportStatus, TransportQuery};
use antecedent_estimate::EffectEstimate;

use super::{
    assumption_label, TransportDeclaration, TransportedRefusal, TransportedResponse,
    TransportedRoute,
};
use crate::result::{ResultPayload, StudyResult};

const RECEIPT_KEYS: &[&str] = &[
    "assumption",
    "conditioning",
    "functional_id",
    "horizon",
    "regime_id",
    "representation",
    "role",
    "route",
    "source_identity",
    "source_population",
    "support",
    "target_population",
    "transform_id",
    "transport_identity",
    "units",
    "variable_id",
    "variable_name",
];

struct ReceiptFields<'a> {
    values: Vec<(&'a str, &'a str)>,
}

impl<'a> ReceiptFields<'a> {
    fn read(result: &'a StudyResult) -> Result<Self, TransportedRefusal> {
        let mut values = Vec::new();
        for diagnostic in &result.diagnostics {
            for (key, value) in diagnostic.fields.iter() {
                let Some(name) = key.as_ref().strip_prefix("receipt.") else {
                    continue;
                };
                if !RECEIPT_KEYS.contains(&name) {
                    return Err(TransportedRefusal::InvalidValue("receipt field"));
                }
                if value.is_empty() {
                    return Err(TransportedRefusal::TransportIdentityIncomplete);
                }
                values.push((name, value.as_ref()));
            }
        }
        Ok(Self { values })
    }

    fn unique(
        &self,
        name: &str,
        label: &'static str,
    ) -> Result<Option<&'a str>, TransportedRefusal> {
        let mut found = None;
        for (key, value) in &self.values {
            if *key != name {
                continue;
            }
            match found {
                None => found = Some(*value),
                Some(prior) if prior == *value => {}
                Some(_) => return Err(TransportedRefusal::InvalidValue(label)),
            }
        }
        Ok(found)
    }

    fn require(&self, name: &str, label: &'static str) -> Result<&'a str, TransportedRefusal> {
        self.unique(name, label)?.ok_or(TransportedRefusal::TransportIdentityIncomplete)
    }

    fn list(&self, name: &str) -> Vec<&'a str> {
        self.values.iter().filter(|(key, _)| *key == name).map(|(_, value)| *value).collect()
    }
}

struct Stamped {
    source_identity: String,
    assumptions: Vec<String>,
    provider_id: String,
    snapshot_id: String,
    causal_contract_id: String,
    standard_error: f64,
}

fn transport_point<'a>(
    result: &'a StudyResult,
) -> Result<(&'a TransportQuery, f64, &'a EffectEstimate), TransportedRefusal> {
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

fn stamped_receipt(
    result: &StudyResult,
    effect: &EffectEstimate,
) -> Result<Stamped, TransportedRefusal> {
    let Some(source_identity) = result.lowered_program_identity() else {
        return Err(TransportedRefusal::TransportIdentityIncomplete);
    };
    let Some(contract) = result.executed_contract() else {
        return Err(TransportedRefusal::TransportIdentityIncomplete);
    };
    let Some(provider_id) = result.logical_plan.estimator.as_deref().filter(|id| !id.is_empty())
    else {
        return Err(TransportedRefusal::TransportIdentityIncomplete);
    };
    let assumptions: Vec<String> = effect
        .provenance
        .assumptions
        .entries
        .iter()
        .map(|record| assumption_label(&record.assumption))
        .collect();
    if assumptions.is_empty() {
        return Err(TransportedRefusal::TransportIdentityIncomplete);
    }
    let standard_error = effect.uncertainty.analytic.se;
    if !standard_error.is_finite() || standard_error < 0.0 {
        return Err(TransportedRefusal::InvalidValue("uncalibrated_standard_error"));
    }
    let causal_contract_id = match contract.identities.program {
        Some(program) => program.to_hex(),
        None => contract.identities.identification.to_hex(),
    };
    Ok(Stamped {
        source_identity: source_identity.to_hex(),
        assumptions,
        provider_id: provider_id.to_owned(),
        snapshot_id: contract.identities.data_snapshot.to_hex(),
        causal_contract_id,
        standard_error,
    })
}

fn agree(
    stated: Option<&str>,
    actual: &str,
    label: &'static str,
) -> Result<(), TransportedRefusal> {
    if stated.is_some_and(|value| value != actual) {
        Err(TransportedRefusal::InvalidValue(label))
    } else {
        Ok(())
    }
}

fn same_members(stated: &[&str], expected: &[String]) -> bool {
    let mut left: Vec<&str> = stated.to_vec();
    left.sort_unstable();
    left.dedup();
    let mut right: Vec<&str> = expected.iter().map(String::as_str).collect();
    right.sort_unstable();
    right.dedup();
    left == right
}

fn declaration_from(
    fields: &ReceiptFields<'_>,
    query: &TransportQuery,
    stamped: &Stamped,
) -> Result<TransportDeclaration, TransportedRefusal> {
    let route_text = fields.require("route", "receipt.route")?;
    let route = parse_route(route_text).ok_or(TransportedRefusal::InvalidValue("receipt.route"))?;
    let transport_identities = fields.list("transport_identity");
    if transport_identities.is_empty() {
        return Err(TransportedRefusal::TransportIdentityIncomplete);
    }
    let support_text = fields.require("support", "receipt.support")?;
    let support = match SupportStatus::from_name(support_text) {
        Some(status) if status.severity() >= SupportStatus::Extrapolative.severity() => status,
        _ => return Err(TransportedRefusal::SupportLabelInflated(support_text.to_owned())),
    };
    match fields.unique("representation", "receipt.representation")? {
        None | Some("point_only") => {}
        Some(_) => return Err(TransportedRefusal::InvalidValue("receipt.representation")),
    }
    let stated_assumptions = fields.list("assumption");
    if !stated_assumptions.is_empty() && !same_members(&stated_assumptions, &stamped.assumptions) {
        return Err(TransportedRefusal::InvalidValue("receipt.assumption"));
    }
    agree(
        fields.unique("source_identity", "receipt.source_identity")?,
        &stamped.source_identity,
        "receipt.source_identity",
    )?;
    let stated_sources = fields.list("source_population");
    if !stated_sources.is_empty()
        && !same_members(&stated_sources, &[query.source_population.as_ref().to_owned()])
    {
        return Err(TransportedRefusal::InvalidValue("receipt.source_population"));
    }
    agree(
        fields.unique("target_population", "receipt.target_population")?,
        query.target_population.as_ref(),
        "receipt.target_population",
    )?;
    Ok(TransportDeclaration {
        route,
        source_identity: stamped.source_identity.clone(),
        source_populations: vec![query.source_population.as_ref().to_owned()],
        target_population: query.target_population.as_ref().to_owned(),
        transport_identities: transport_identities.into_iter().map(str::to_owned).collect(),
        assumptions: stamped.assumptions.clone(),
        support,
        uncalibrated_standard_error: Some(stamped.standard_error),
        provider_id: stamped.provider_id.clone(),
        snapshot_id: stamped.snapshot_id.clone(),
        causal_contract_id: stamped.causal_contract_id.clone(),
    })
}

fn coordinate_from(
    fields: &ReceiptFields<'_>,
    target_population: &str,
) -> Result<ScientificQuantity, TransportedRefusal> {
    let variable_id = fields.require("variable_id", "receipt.variable_id")?;
    if fields
        .unique("variable_name", "receipt.variable_name")?
        .is_some_and(|name| name != variable_id)
    {
        return Err(TransportedRefusal::InvalidValue("receipt.variable_name"));
    }
    if fields.unique("conditioning", "receipt.conditioning")?.is_some() {
        return Err(TransportedRefusal::InvalidValue("receipt.conditioning"));
    }
    let role = match fields.require("role", "receipt.role")? {
        "outcome" => QuantityRole::Outcome,
        _ => return Err(TransportedRefusal::InvalidValue("receipt.role")),
    };
    let horizon = fields
        .require("horizon", "receipt.horizon")?
        .parse::<u32>()
        .map_err(|_| TransportedRefusal::InvalidValue("receipt.horizon"))?;
    Ok(ScientificQuantity {
        variable_id: variable_id.to_owned(),
        variable_name: variable_id.to_owned(),
        role,
        units: fields.require("units", "receipt.units")?.to_owned(),
        population_id: target_population.to_owned(),
        regime_id: fields.require("regime_id", "receipt.regime_id")?.to_owned(),
        horizon,
        functional_id: fields.require("functional_id", "receipt.functional_id")?.to_owned(),
        conditioning: Vec::new(),
        transform_id: fields.require("transform_id", "receipt.transform_id")?.to_owned(),
    })
}

fn parse_route(text: &str) -> Option<TransportedRoute> {
    [
        TransportedRoute::CohortEffect,
        TransportedRoute::ShiftedCohort,
        TransportedRoute::Policy,
        TransportedRoute::Synthesis,
        TransportedRoute::Mechanism,
        TransportedRoute::TemporalPolicy,
        TransportedRoute::AdaptiveTrial,
        TransportedRoute::SurrogateBridge,
        TransportedRoute::Declared,
    ]
    .into_iter()
    .find(|route| route.as_str() == text)
}

// not_scientific: this reads a receipt the result already carries and wraps it
// for evaluate_transported. It computes no estimate, changes no standing, and
// writes no identity, so it is not an alias of an executing unit and it creates
// no support or coverage unit. The wrapper identity is the receipt's
// canonical identity.
/// Project a transport result onto a [`TransportedResponse`].
///
/// The payload must be a trial-to-target transport and the identification query
/// a transport query. The receipt's route, transport identities, support label,
/// and the coordinate's units, variable, regime, functional, transform, horizon,
/// and role must already be present as `receipt.*` diagnostic fields. The
/// source identity, populations, assumptions, provider, snapshot, contract, the
/// transported point, and its analytic standard error are read from the result.
/// A field that is missing refuses. A field that restates a stamped value and
/// disagrees refuses. The returned wrapper is [`TransportedResponse::declared_point`]
/// of those fields, so its identity is the receipt's canonical identity and an
/// existing refusal is unchanged.
///
/// # Errors
/// [`TransportedRefusal::NotTransported`] when the result is not a transport.
/// [`TransportedRefusal::TransportIdentityIncomplete`] when a receipt field is
/// missing. Support, route, value, and identity refusals are the ones
/// [`TransportedResponse::declared_point`] already returns.
pub fn project_transported_response(
    result: &StudyResult,
) -> Result<TransportedResponse, TransportedRefusal> {
    let (query, ipw, effect) = transport_point(result)?;
    let stamped = stamped_receipt(result, effect)?;
    let fields = ReceiptFields::read(result)?;
    let declaration = declaration_from(&fields, query, &stamped)?;
    let quantity = coordinate_from(&fields, declaration.target_population.as_str())?;
    TransportedResponse::declared_point(vec![quantity], vec![ipw], &declaration)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, ContinuousDomain, Diagnostic, DiagnosticKind,
        DiagnosticSeverity, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
        SupportStatus, TransportQuery, VariableId,
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

    fn push_receipt(
        result: &mut crate::StudyResult,
        route: &str,
        support: &str,
        extra: &[(&str, &str)],
    ) {
        let mut fields: Vec<(String, String)> = vec![
            ("receipt.route".into(), route.into()),
            ("receipt.support".into(), support.into()),
            ("receipt.transport_identity".into(), "diagram-1".into()),
            ("receipt.units".into(), "units".into()),
            ("receipt.variable_id".into(), "y".into()),
            ("receipt.regime_id".into(), "contrast:do(a=1)-do(a=0)".into()),
            ("receipt.functional_id".into(), "mean".into()),
            ("receipt.transform_id".into(), "identity".into()),
            ("receipt.horizon".into(), "0".into()),
            ("receipt.role".into(), "outcome".into()),
        ];
        for (key, value) in extra {
            fields.push(((*key).to_owned(), (*value).to_owned()));
        }
        result.diagnostics.push(
            Diagnostic::new(
                "estimate.transport.receipt",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "transport receipt carried on the result",
            )
            .with_fields(fields),
        );
    }

    fn quantity(target: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "y".into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: target.into(),
            regime_id: "contrast:do(a=1)-do(a=0)".into(),
            horizon: 0,
            functional_id: "mean".into(),
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
            transport_identities: vec!["diagram-1".into()],
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
        let diagnostics = result.diagnostics.len();
        assert_eq!(
            project_transported_response(&result).unwrap_err(),
            TransportedRefusal::TransportIdentityIncomplete,
        );

        let effect = result.estimate().as_effect().unwrap();
        let ipw = match result.payload() {
            crate::ResultPayload::Transport(transported) => transported.ipw,
            other => panic!("transport payload, got {other:?}"),
        };
        let mut carried = result.clone();
        push_receipt(&mut carried, "declared", "extrapolative", &[]);
        let projected = project_transported_response(&carried).unwrap();
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
        assert_eq!(result.diagnostics.len(), diagnostics);

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

        let mut policy = result.clone();
        push_receipt(&mut policy, "policy", "extrapolative", &[]);
        assert_eq!(
            project_transported_response(&policy).unwrap_err(),
            TransportedRefusal::RouteNeedsTypedSource(TransportedRoute::Policy),
        );

        let mut inflated = result.clone();
        push_receipt(&mut inflated, "declared", "supported", &[]);
        assert_eq!(
            project_transported_response(&inflated).unwrap_err(),
            TransportedRefusal::SupportLabelInflated("supported".into()),
        );

        let mut restated = result.clone();
        push_receipt(
            &mut restated,
            "declared",
            "extrapolative",
            &[("receipt.source_identity", "not-the-lowered-program")],
        );
        assert_eq!(
            project_transported_response(&restated).unwrap_err(),
            TransportedRefusal::InvalidValue("receipt.source_identity"),
        );
    }

    #[test]
    fn a_transport_receipt_projects_only_when_the_result_already_carries_it() {
        on_large_stack(project_the_trial);
    }
}
