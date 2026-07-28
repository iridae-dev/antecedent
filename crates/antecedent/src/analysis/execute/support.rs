// Free functions supporting Study execute paths.
// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) fn gcm_query_vars(query: &CausalQuery) -> Result<(VariableId, VariableId), CausalError> {
    match query {
        CausalQuery::Counterfactual(q) => {
            let outcome = *q.outcomes.first().ok_or_else(|| CausalError::Compile {
                message: "counterfactual missing outcome".into(),
            })?;
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).unwrap_or(outcome);
            Ok((treatment, outcome))
        }
        CausalQuery::AnomalyAttribution(q) => {
            let outcome = *q.targets.first().unwrap_or(&VariableId::from_raw(0));
            Ok((outcome, outcome))
        }
        CausalQuery::ChangeAttribution(q) => Ok((q.outcome, q.outcome)),
        CausalQuery::MechanismChange(q) => {
            let outcome = *q.targets.first().unwrap_or(&VariableId::from_raw(0));
            Ok((outcome, outcome))
        }
        CausalQuery::UnitChange(q) => Ok((q.outcome, q.outcome)),
        _ => Err(CausalError::Compile { message: "gcm_query_vars: unsupported query".into() }),
    }
}

pub(super) fn nan_effect() -> EffectEstimate {
    EffectEstimate::new(
        f64::NAN,
        f64::NAN,
        antecedent_core::AssumptionSet::default(),
        OverlapPolicy::ExplicitOverride,
    )
}

/// Interactive graph×effect: stratified subsample of Identified graphs; leftover
/// identified mass is flipped to Unidentified (never silent renormalize to 1).
pub(super) fn maybe_interactive_envelope_subsample(
    latency_mode: Option<LatencyMode>,
    graphs: WeightedGraphSamples,
    per_graph: Vec<GraphEffectDraws>,
    ctx: &ExecutionContext,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<(WeightedGraphSamples, Vec<GraphEffectDraws>), CausalError> {
    if latency_mode != Some(LatencyMode::Interactive) {
        return Ok((graphs, per_graph));
    }
    let mut rng = ctx.rng.stream(0xE11E_u64);
    let sub = graphs
        .stratified_interactive_subsample(INTERACTIVE_MAX_ENVELOPE_GRAPHS, &mut rng)
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if !sub.approximate {
        return Ok((sub.graphs, per_graph));
    }
    let keep_keys: std::collections::HashSet<u64> = sub
        .graphs
        .graph_keys
        .iter()
        .zip(sub.graphs.identified.iter())
        .filter(|(_, f)| **f == GraphIdentFlag::Identified)
        .map(|(k, _)| *k)
        .collect();
    let filtered: Vec<GraphEffectDraws> =
        per_graph.into_iter().filter(|g| keep_keys.contains(&g.graph_key)).collect();
    diagnostics.push(Diagnostic::new(
        "estimate.envelope.interactive_subsample",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "approximate=true leftover_identified_mass={} max_identified={}",
            sub.leftover_identified_mass, INTERACTIVE_MAX_ENVELOPE_GRAPHS
        ),
    ));
    Ok((sub.graphs, filtered))
}

pub(super) fn parametric_scm_identification(
    query: CausalQuery,
    _treatment: VariableId,
    _outcome: VariableId,
) -> (IdentificationResult, IdentifiedEstimand) {
    let estimand = IdentifiedEstimand::backdoor(
        "gcm.parametric",
        Arc::from([]),
        antecedent_expr::ExprId::from_raw(0),
    );
    let identification = IdentificationResult::from_parts(
        IdentificationStatus::IdentifiedUnderParametricRestrictions,
        query,
        vec![estimand.clone()],
        CausalExprArena::new(),
        DerivationTrace::default(),
        antecedent_core::AssumptionSet::default(),
        Vec::new(),
        IdentificationPerformanceRecord::default(),
        None,
    );
    (identification, estimand)
}

pub(super) fn binary_cf_interventions(
    query: &antecedent_core::CounterfactualQuery,
) -> Result<(VariableId, f64, f64), CausalError> {
    if query.interventions.len() != 1 {
        return Err(CausalError::Unsupported {
            message: "Study counterfactual path currently supports a single hard \
                 intervention for ITE (use gcm helpers for multi-world predict)",
        });
    }
    let Intervention::Set { variable, value } = &query.interventions[0] else {
        return Err(CausalError::Unsupported {
            message: "Study counterfactual path requires a hard Set intervention",
        });
    };
    let active = value.as_f64().ok_or_else(|| CausalError::Compile {
        message: "counterfactual intervention value must be f64".into(),
    })?;
    Ok((*variable, active, 0.0))
}

pub(super) fn identification_status_ok_for_case(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
    )
}

pub(super) fn envelope_to_identification_result(
    envelope: &IdentificationEnvelope<Pag>,
    query: &AverageEffectQuery,
) -> IdentificationResult {
    let mut estimands = Vec::new();
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut diagnostics = Vec::new();
    for case in &envelope.cases {
        if identification_status_ok_for_case(case.result.status) {
            estimands.extend(case.result.estimands.iter().cloned());
            assumptions = case.result.required_assumptions.clone();
            diagnostics.extend(case.result.diagnostics.iter().cloned());
        }
    }
    if let Some(inv) = &envelope.invariant {
        if estimands.is_empty() {
            estimands.push(inv.clone());
        }
    }
    IdentificationResult::from_parts(
        envelope.status,
        CausalQuery::AverageEffect(query.clone()),
        estimands,
        CausalExprArena::new(),
        DerivationTrace::default(),
        assumptions,
        diagnostics,
        IdentificationPerformanceRecord::default(),
        None,
    )
}

pub(super) fn mark_panel_classification(compiled: CompiledAnalysis) -> CompiledAnalysis {
    match compiled {
        CompiledAnalysis::Ready(mut physical) => {
            physical.logical.record.data_classification = DataClassification::Panel;
            CompiledAnalysis::Ready(physical)
        }
        other => other,
    }
}

pub(super) fn admg_has_bidirected(admg: &Admg) -> bool {
    (0..admg.node_count()).any(|i| {
        let id = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        !admg.bidirected_neighbors(id).is_empty()
    })
}

pub(super) fn admg_to_dag(admg: &Admg) -> Result<Dag, CausalError> {
    let n = u32::try_from(admg.node_count())
        .map_err(|_| CausalError::Compile { message: "ADMG too large".into() })?;
    let mut dag = Dag::with_variables(n);
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        for &to in admg.children(from) {
            dag.insert_directed(from, to)
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        }
    }
    Ok(dag)
}
