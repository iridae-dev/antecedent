//! Contract compilation: program payloads, data snapshot and identity layers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(
    clippy::wildcard_imports,
    reason = "the contract module is split by concern and these files share its imports and private items"
)]
use super::*;

pub(super) fn program_payloads_for(
    study: &Study,
    resolved_estimator: Option<&str>,
    binding: CheckedProgramBinding<'_>,
    posterior_programs: Option<&antecedent_io::GraphPosteriorProgramsWire>,
) -> Result<ProgramPayloads, CausalError> {
    let cached = contract_identification(study);
    let cached = cached.as_deref();
    program_payloads(
        study,
        cached,
        cached.is_some_and(identification_search_capped),
        resolved_estimator,
        binding,
        posterior_programs.cloned(),
    )
}

pub(super) fn merge_response_factor_snapshots(
    snapshots: &[antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot],
) -> Result<
    antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot,
    CausalError,
> {
    let Some(first) = snapshots.first() else {
        return Err(CausalError::Compile {
            message: "response curve has no checked members".into(),
        });
    };
    if snapshots.iter().any(|snapshot| snapshot.provenance != first.provenance) {
        return Err(CausalError::Compile {
            message: "response member factor snapshots have different provider provenance".into(),
        });
    }
    let mut requirements = Vec::new();
    let mut domains = Vec::new();
    let mut factors = Vec::new();
    for snapshot in snapshots {
        requirements.extend(snapshot.requirements.iter().cloned());
        for domain in snapshot.provider.domains.iter() {
            if let Some(existing) = domains.iter().find(
                |existing: &&antecedent_expr::provider::DiscreteDomainSnapshot| {
                    existing.variable == domain.variable
                },
            ) {
                if existing.values != domain.values {
                    return Err(CausalError::Compile {
                        message: "response member snapshots disagree on a finite domain".into(),
                    });
                }
            } else {
                domains.push(domain.clone());
            }
        }
        for factor in snapshot.provider.factors.iter() {
            if let Some(existing) = factors.iter().find(
                |existing: &&antecedent_expr::provider::EmpiricalFactorTableSnapshot| {
                    existing.variables == factor.variables
                        && existing.conditioned_on == factor.conditioned_on
                        && existing.intervention == factor.intervention
                        && existing.domain == factor.domain
                        && existing.population == factor.population
                        && existing.regime == factor.regime
                },
            ) {
                if existing.rows != factor.rows {
                    return Err(CausalError::Compile {
                        message: "aliased response factors have inconsistent laws".into(),
                    });
                }
            } else {
                factors.push(factor.clone());
            }
        }
    }
    Ok(antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot {
        requirements: Arc::from(requirements),
        provider: antecedent_expr::provider::EmpiricalProviderSnapshot {
            domains: Arc::from(domains),
            factors: Arc::from(factors),
        },
        provenance: first.provenance.clone(),
    })
}

pub(super) fn nested_counterfactual_fit_wire(
    data: &antecedent_data::TabularData,
    operation: &crate::gcm::NestedCounterfactualOperation,
) -> Result<antecedent_io::NestedCounterfactualFitWire, CausalError> {
    let query = operation.query();
    let x = data.float64_values(query.treatment).map_err(CausalError::rejected)?;
    let m = data.float64_values(query.mediator).map_err(CausalError::rejected)?;
    let y = data.float64_values(query.outcome).map_err(CausalError::rejected)?;
    let mut gram = [0.0_f64; 9];
    let mut outcome_cross = [0.0_f64; 3];
    let mut complete_case_rows = 0_u64;
    for ((x, m), y) in x.iter().zip(&m).zip(&y) {
        if !x.is_finite() || !m.is_finite() || !y.is_finite() {
            continue;
        }
        complete_case_rows += 1;
        let design = [1.0, *x, *m];
        for row in 0..3 {
            outcome_cross[row] += design[row] * y;
            for column in 0..3 {
                gram[row * 3 + column] += design[row] * design[column];
            }
        }
    }
    if complete_case_rows < 4 {
        return Err(CausalError::Compile {
            message:
                "nested counterfactual requires at least four complete outcome-regression rows"
                    .into(),
        });
    }
    Ok(antecedent_io::NestedCounterfactualFitWire {
        format: 1,
        complete_case_rows,
        gram,
        outcome_cross,
    })
}

pub(super) fn nested_prior_scale(inference: &InferenceMode) -> Option<f64> {
    match inference {
        InferenceMode::Bayesian(cfg) => Some(cfg.prior_scale),
        InferenceMode::Frequentist => None,
    }
}

/// Lagged `[1, treatment, mediator]` moments for a nested time-graph point.
pub(super) fn temporal_nested_fit_wire(
    data: &antecedent_data::TimeSeriesData,
    query: &antecedent_core::NestedCounterfactualQuery,
    prior_scale: Option<f64>,
) -> Result<antecedent_io::NestedCounterfactualFitWire, CausalError> {
    let moments = antecedent_estimate::nested_direct_fit_moments(
        data,
        query.treatment,
        query.mediator,
        query.outcome,
        &antecedent_core::KernelPolicy::scalar_only(),
        prior_scale,
    )?;
    Ok(antecedent_io::NestedCounterfactualFitWire {
        format: 1,
        complete_case_rows: moments.complete_case_rows,
        gram: moments.gram,
        outcome_cross: moments.outcome_cross,
    })
}

/// Retain only the moments needed to replay a point estimate and classical
/// homoskedastic standard error. Other fit/uncertainty procedures keep their
/// explicit dependency refusal in the independent consumer.
pub(super) fn linear_fit_moments_wire(
    data: &antecedent_data::TabularData,
    operation: &crate::analysis::prepared::CheckedLinearOperation,
) -> Result<Option<antecedent_io::LinearFitMomentsWire>, CausalError> {
    if !matches!(operation.fitter.fit_kind, antecedent_estimate::LinearFitKind::Ols)
        || operation.fitter.inference.se_kind != antecedent_estimate::AnalyticSeKind::Homoskedastic
        || operation.fitter.inference.bootstrap_replicates != 0
    {
        return Ok(None);
    }
    let checked =
        operation.fitter.rebind_checked(&operation.preparation, data).map_err(CausalError::from)?;
    let design = &checked.problem().design;
    if design.nrows == 0 || design.ncols < 2 || design.matrix.len() != design.nrows * design.ncols {
        return Ok(None);
    }
    let p = design.ncols;
    let mut gram = vec![0.0; p * p];
    let mut outcome_cross = vec![0.0; p];
    let mut outcome_square = 0.0;
    for row in 0..design.nrows {
        let y = design.outcome[row];
        if !y.is_finite() {
            return Ok(None);
        }
        outcome_square += y * y;
        for left in 0..p {
            let x_left = design.matrix[left * design.nrows + row];
            if !x_left.is_finite() {
                return Ok(None);
            }
            outcome_cross[left] += x_left * y;
            for right in 0..p {
                gram[left * p + right] += x_left * design.matrix[right * design.nrows + row];
            }
        }
    }
    if gram.iter().chain(&outcome_cross).any(|value| !value.is_finite())
        || !outcome_square.is_finite()
    {
        return Ok(None);
    }
    let Ok(complete_case_rows) = u64::try_from(design.nrows) else {
        return Ok(None);
    };
    let Ok(columns) = u32::try_from(p) else {
        return Ok(None);
    };
    Ok(Some(antecedent_io::LinearFitMomentsWire {
        format: 1,
        complete_case_rows,
        columns,
        gram,
        outcome_cross,
        outcome_square,
    }))
}

/// Add `study`'s data snapshot to program payloads already compiled for it.
pub(super) fn contract_payloads(
    program: Arc<ProgramPayloads>,
    study: &Study,
    prepared: Option<&PreparedStudy>,
) -> Result<ContractPayloads, CausalError> {
    let mut data_snapshot = data_snapshot_wire(
        &study.data,
        study.interference.as_ref(),
        study.declared.participants.as_ref(),
        missing_data::policy_wire(study),
        sampling_identity(study.declared.sampling_design, study.declared.score_dependence.as_ref()),
        &program.observation_digest,
    )?;
    if let (Some(prepared), DataInput::Tabular(tabular)) = (prepared, &study.data) {
        let laws = prepared
            .distribution_factor_snapshot(tabular)?
            .or(prepared.functional_effect_factor_snapshot(tabular)?);
        if let Some(laws) = laws {
            data_snapshot.distribution_factor_laws = Some(
                antecedent_io::distribution_factor_laws_to_wire(&laws)
                    .map_err(|err| io_err(&err))?,
            );
        } else if let Some(snapshots) =
            prepared.functional_effect_response_factor_snapshots(tabular)?
        {
            let merged = merge_response_factor_snapshots(&snapshots)?;
            data_snapshot.distribution_factor_laws = Some(
                antecedent_io::distribution_factor_laws_to_wire(&merged)
                    .map_err(|err| io_err(&err))?,
            );
        }
        if let Some(operation) = prepared.execution().nested_counterfactual() {
            data_snapshot.nested_counterfactual_fit =
                Some(nested_counterfactual_fit_wire(tabular, operation)?);
        }
        if let Some(operation) = prepared.execution().linear_operation() {
            data_snapshot.linear_fit_moments = linear_fit_moments_wire(tabular, operation)?;
        }
    }
    if let DataInput::Temporal(series) | DataInput::Event(series) = &study.data {
        if let CausalQuery::NestedCounterfactual(query) = &study.query {
            data_snapshot.nested_counterfactual_fit = Some(temporal_nested_fit_wire(
                series,
                query,
                nested_prior_scale(&study.settings.inference),
            )?);
        }
    }
    let snapshot_digest = data_snapshot_digest(&data_snapshot).map_err(|err| io_err(&err))?;
    Ok(ContractPayloads { identities: program.identities(snapshot_digest), program, data_snapshot })
}

pub(super) fn program_payloads(
    study: &Study,
    cached: Option<&IdentificationResult>,
    search_capped: bool,
    resolved_estimator: Option<&str>,
    binding: CheckedProgramBinding<'_>,
    graph_posterior_programs: Option<antecedent_io::GraphPosteriorProgramsWire>,
) -> Result<ProgramPayloads, CausalError> {
    let selected = resolved_estimator.and_then(|name| name.parse::<crate::EstimatorId>().ok());
    if !program_wire::binding_agrees(binding, selected, &study.query) {
        return Err(CausalError::Compile {
            message: "checked program binding disagrees with the selected estimator or query"
                .into(),
        });
    }
    let schema = data_schema(&study.data);
    let mut target = TargetIdentityWire {
        format: IDENTITY_FORMAT,
        schema: schema_to_wire(schema),
        query: causal_query_to_wire_with_registry(
            &study.query,
            study.declared.population_registry.as_ref(),
        )
        .map_err(|err| io_err(&err))?,
    };
    let target_digest = digest_wire(IdentityDomain::Target, &target).map_err(|err| io_err(&err))?;
    let question_digest = target.question_digest().map_err(|err| io_err(&err))?;
    let observation = observation_identity_wire(
        schema,
        crate::analysis::contract_identity::observation_identity_tags(
            &study.query,
            study.observation_delayed_entry,
        ),
    );
    let observation_digest =
        digest_wire(IdentityDomain::Observation, &observation).map_err(|err| io_err(&err))?;
    let graph = graph_identity(study)?;
    let identification = identification_identity_wire(
        study,
        schema,
        *question_digest.as_bytes(),
        *observation_digest.as_bytes(),
        graph,
    )?;
    let identification_digest =
        identification_digest(&identification).map_err(|err| io_err(&err))?;
    let identification_product = match cached {
        Some(result) => {
            let mut wire =
                identification_product_wire(result, search_capped).map_err(|err| io_err(&err))?;
            // Two envelopes over the same class with the same estimands are
            // still different products when they examined different
            // completions or carry different mass.
            wire.envelope = class_envelope_wire(study);
            Some(wire)
        }
        None => None,
    };
    let identification_product_digest = match &identification_product {
        Some(wire) => Some(identification_product_digest_wire(wire).map_err(|err| io_err(&err))?),
        None => None,
    };
    let mut commitments = inferential_commitments(study, resolved_estimator);
    let lowerings =
        program_wire::ProgramLowerings::bind(binding, study, resolved_estimator, &mut commitments)?;
    let functional_program = program_wire::functional_program(binding, study, cached, selected)?;
    let program = ProgramIdentityWire {
        format: IDENTITY_FORMAT,
        target: *target_digest.as_bytes(),
        identification: *identification_digest.as_bytes(),
        identification_product: identification_product_digest.map(|digest| *digest.as_bytes()),
        completion_budget: study.declared.max_completions.map(|cap| cap as u64),
        commitments: commitments.clone(),
        functional_program,
        checked_aipw_lowering: lowerings.aipw,
        checked_frontdoor_lowering: lowerings.frontdoor,
        checked_iv_lowering: lowerings.iv,
        checked_linear_adjustment_lowering: lowerings.linear_adjustment,
        checked_functional_response_grid: lowerings.functional_response_grid,
        checked_nested_counterfactual: lowerings.nested_counterfactual,
        graph_posterior_programs,
        structure_provenance: study
            .structure_provenance()
            .map(crate::accepted::StructureProvenance::digest),
    };
    let program_digest = program_digest(&program).map_err(|err| io_err(&err))?;
    let inference_binding = inference_binding_wire(study, commitments.inference.clone());
    let inference_binding_digest =
        inference_binding_digest(&inference_binding).map_err(|err| io_err(&err))?;
    Ok(ProgramPayloads {
        target,
        identification,
        identification_product,
        program,
        inference_binding,
        observation,
        target_digest,
        identification_digest,
        identification_product_digest,
        program_digest,
        inference_binding_digest,
        observation_digest,
    })
}

/// Identification identity wire: the question, graph, transport and observation bindings that fix identification.
pub(super) fn identification_identity_wire(
    study: &Study,
    schema: &CausalSchema,
    target_question: [u8; 32],
    observation: [u8; 32],
    graph: GraphIdentityWire,
) -> Result<IdentificationIdentityWire, CausalError> {
    Ok(IdentificationIdentityWire {
        format: IDENTITY_FORMAT,
        target_question,
        population_depends_on: population_depends_on(
            &study.query,
            study.declared.population_registry.as_ref(),
        ),
        rd_config: study.rd.as_ref().map(|cfg| antecedent_io::RdConfigWire {
            running_variable: cfg.running_variable.raw(),
            cutoff_bits: cfg.cutoff.to_bits(),
            bandwidth_bits: cfg.bandwidth.to_bits(),
            se_kind: Some(cfg.se_kind.as_str().into()),
        }),
        class_prior: study.declared.class_prior.as_ref().map(crate::ClassPrior::identity_wire),
        transport: transport_premises(study)?,
        graph_class: study.graph.class().as_str().into(),
        structure_source: study.structure_source.as_str().into(),
        accepted_version: study.graph.version(),
        algorithm_id: study.graph.algorithm_id().map(str::to_string),
        schema_names: Some(schema.variables().iter().map(|v| v.name.to_string()).collect()),
        graph,
        observation,
    })
}

/// Inference binding wire: the inference mode, replicates, priors, overlap, options and split the result depends on.
pub(super) fn inference_binding_wire(study: &Study, inference: String) -> InferenceBindingWire {
    InferenceBindingWire {
        format: IDENTITY_FORMAT,
        inference,
        bootstrap_replicates: study.settings.bootstrap_replicates,
        bayesian: match &study.settings.inference {
            InferenceMode::Bayesian(cfg) => {
                Some(crate::analysis::contract_identity::bayesian_binding(cfg, study.draw_controls))
            }
            InferenceMode::Frequentist => None,
        },
        validation_suite: study.settings.refute.validation_suite_id().map(str::to_string),
        overlap_policy: study.declared.overlap_policy.map(overlap_policy_tag),
        estimator_spec: study.estimator_spec_identity.clone(),
        response_options: study
            .response_options
            .as_ref()
            .map(crate::analysis::contract_identity::response_options),
        observation_options: crate::analysis::contract_identity::observation_options(
            &study.observation_options,
        ),
        split: study.declared.split.as_ref().map(crate::analysis::contract_identity::split),
        joint_horizons: study.settings.retain_joint_horizons,
    }
}

pub(super) fn transport_premises(
    study: &Study,
) -> Result<Option<antecedent_io::TransportIdentityWire>, CausalError> {
    if study.selection_diagram.is_none() && study.transport_trial.is_none() {
        return Ok(None);
    }
    antecedent_io::transport_identity(
        study.selection_diagram.as_ref(),
        study
            .transport_trial
            .as_ref()
            .map(|trial| (trial.trial, trial.selection_probability, trial.treatment_probability)),
    )
    .map(Some)
    .map_err(|err| io_err(&err))
}

pub(super) fn compile_contract(
    study: &Study,
    prepared: Option<&PreparedStudy>,
) -> Result<CausalContract, CausalError> {
    Ok(compile_with_payloads(study, prepared, None)?.0)
}

pub(super) fn compile_with_payloads(
    study: &Study,
    prepared: Option<&PreparedStudy>,
    program: Option<Arc<ProgramPayloads>>,
) -> Result<(CausalContract, ContractPayloads), CausalError> {
    // Cheap inspection is the pre-prepare record: it reads no cache, so a
    // prepared handle's `inspect` and `contract` differ by the products, not
    // by which fields happen to be populated.
    let cached = prepared.and_then(|_| contract_identification(study));
    let cached = cached.as_deref();
    let search_capped = cached.is_some_and(identification_search_capped);
    let resolved_estimator =
        prepared.and_then(|prepared| prepared.plan().logical.record.estimator.clone());
    let program = match program {
        Some(program) => program,
        None => Arc::new(program_payloads(
            study,
            cached,
            search_capped,
            resolved_estimator.as_deref(),
            prepared
                .filter(|prepared| study.query == *prepared.query())
                .map_or(CheckedProgramBinding::None, PreparedStudy::program_binding),
            None,
        )?),
    };
    let mut payloads = contract_payloads(program, study, prepared)?;
    if prepared.is_none() {
        // Cheap inspection runs no identification: the program the prepared
        // handle compiles covers products that do not exist yet.
        payloads.identities.program = None;
    }
    let reasoning = reasoning_view(study, prepared, cached, search_capped);
    let lagged = full_temporal_identification(study);
    let mut body = body_frame(
        &study.query,
        lagged.as_deref(),
        cached,
        study.declared.population_registry.as_ref(),
    )?;
    body.window_variables = window_namespace(study);
    let (evidence_obligations, evidence_actions) = evidence_needs::derive(study, cached);
    let functional = contract_functional_label(study, resolved_estimator.as_deref());
    Ok((
        CausalContract {
            identities: payloads.identities,
            reasoning,
            graph_class: study.graph.class(),
            structure_source: study.structure_source,
            accepted_version: study.graph.version(),
            discovery_algorithm: study.graph.algorithm_id().map(Arc::from),
            structure_provenance: study.structure_provenance_shared(),
            accepted_variable_names: study.graph.variable_names().map(Arc::from),
            support_status: study.support_status,
            identifier: study.declared.identifier.map(|id| Arc::from(id.as_str())),
            estimator: advertised_estimator(study, resolved_estimator.as_deref()).map(Arc::from),
            resolved_estimator,
            row_count: data_row_count(&study.data),
            modality: Arc::from(snapshot_modality(&study.data)),
            inference: Arc::from(declared_inference(study)),
            query_kind: Arc::from(
                (matches!(study.data, DataInput::Temporal(_))
                    && matches!(study.query, CausalQuery::Mediation(_))
                    && study.graph.class() == GraphClass::TemporalAdmg)
                    .then_some("TemporalMediationEffect")
                    .or_else(|| crate::support::query_axis_name(&study.query, study.graph.class()))
                    .or_else(|| compiled_off_axis_kind(&study.query))
                    .unwrap_or("Unknown"),
            ),
            score_reuse: None,
            target_weights: None,
            checked_aipw_rows: None,
            posterior: Arc::from(posterior_label(
                &study.settings.inference,
                matches!(study.query, CausalQuery::AverageEffect(_))
                    && study.graph.class() == GraphClass::Dag
                    || matches!(study.query, CausalQuery::TemporalEffect(_))
                        && study.graph.class() == GraphClass::TemporalDag,
            )),
            functional: Arc::from(functional),
            posterior_draws: match &study.settings.inference {
                InferenceMode::Bayesian(cfg) => u32::try_from(cfg.n_draws).ok(),
                InferenceMode::Frequentist => None,
            },
            evidence_obligations,
            evidence_actions,
            body: Arc::new(body),
        },
        payloads,
    ))
}

/// Calibration labels retain estimator choices in the established wire order.
pub(super) fn contract_functional_label(study: &Study, resolved_estimator: Option<&str>) -> String {
    let mut functional = match overlap_label(study, resolved_estimator) {
        Some(overlap) => format!("{}+{overlap}", functional_label(&study.query)),
        None => functional_label(&study.query),
    };
    // Calibration labels retain nuisance, stratification and matching choices in order.
    if let Some(estimator) = study.declared.estimator_spec.as_deref() {
        for label in [
            estimator.propensity_calibration_label(),
            estimator.stratification_calibration_label().map(str::to_owned),
            estimator.matching_calibration_label().map(str::to_owned),
            estimator.bootstrap_interval_calibration_label().map(str::to_owned),
        ]
        .into_iter()
        .flatten()
        {
            functional = format!("{functional}+{label}");
        }
    }
    // Sampling weights change the estimand and the variance: an interval measured on unweighted
    // rows must not bind to a weighted-population estimate.
    let weighted_table = match &study.data {
        DataInput::Tabular(data) => data.storage().weights().is_some(),
        DataInput::Temporal(data) | DataInput::Event(data) => data.storage().weights().is_some(),
        DataInput::MultiEnv(_) | DataInput::Panel(_) => false,
    };
    if weighted_table
        || study
            .declared
            .estimator_spec
            .as_deref()
            .and_then(crate::estimator_spec::EstimatorSpec::sampling_weights_calibration_label)
            .is_some()
    {
        functional = format!("{functional}+{}", crate::estimator_spec::SAMPLING_WEIGHTS_LABEL);
    }
    functional
}

pub(super) fn data_schema(data: &DataInput) -> &CausalSchema {
    match data {
        DataInput::Tabular(data) => data.schema(),
        DataInput::Temporal(data) | DataInput::Event(data) => data.schema(),
        DataInput::MultiEnv(data) => data.schema(),
        DataInput::Panel(data) => data.schema(),
    }
}

/// Data modality of an execution, in the `DataSnapshotIdentityWire::modality`
/// vocabulary.
///
/// The single owner of that vocabulary. The data snapshot and the calibration
/// match key must name the same modality for the same execution, or the claim
/// cites coverage measured on data of another shape — so both read this, from
/// the `DataInput` the study actually holds, rather than one of them
/// re-deriving it from a plan record that may have been compiled under a
/// coarser classification.
pub(super) const fn snapshot_modality(data_input: &DataInput) -> &'static str {
    match data_input {
        DataInput::Tabular(_) => "tabular",
        DataInput::Temporal(_) => "series",
        DataInput::Event(_) => "event",
        DataInput::MultiEnv(_) => "multi_env",
        DataInput::Panel(_) => "panel",
    }
}

/// The sampling declaration as the snapshot identity records it: the design, and when cluster
/// labels or a serial kernel are declared beside it, a fingerprint of them, so two results that
/// differ only in the declared dependence never share an identity.
pub(super) fn sampling_identity(
    design: Option<antecedent_core::DeclaredSamplingDesign>,
    dependence: Option<&antecedent_estimate::ScoreDependence>,
) -> Option<String> {
    let base = design.map(|design| design.as_str().to_owned());
    match dependence {
        None => base,
        Some(dependence) => Some(format!(
            "{};score_dependence={:016x}",
            base.as_deref().unwrap_or("undeclared"),
            dependence.digest()
        )),
    }
}

pub(super) fn data_snapshot_wire(
    data_input: &DataInput,
    interference: Option<&crate::analysis::builder::InterferenceSpec>,
    participants: Option<&antecedent_io::ParticipantBindingWire>,
    missing: Option<antecedent_io::MissingPolicyWire>,
    sampling_design: Option<String>,
    observation: &antecedent_core::SemanticDigest,
) -> Result<DataSnapshotIdentityWire, CausalError> {
    let modality = snapshot_modality(data_input);
    let (regularity, row_count, unit_count) = match data_input {
        DataInput::Tabular(data) => (None, u64_count(data.row_count())?, None),
        DataInput::Temporal(data) | DataInput::Event(data) => (
            Some(regularity_tag(&data.time_index().regularity)),
            u64_count(data.row_count())?,
            None,
        ),
        DataInput::MultiEnv(data) => (
            None,
            u64_count(data.environments().iter().map(TableView::row_count).sum())?,
            Some(u64_count(data.env_count())?),
        ),
        DataInput::Panel(data) => {
            (None, u64_count(data.total_rows())?, Some(u64_count(data.unit_count())?))
        }
    };
    let partitions = source_partitions(data_input)?;
    Ok(DataSnapshotIdentityWire {
        format: IDENTITY_FORMAT,
        observation: *observation.as_bytes(),
        modality: modality.into(),
        regularity,
        row_count,
        unit_count,
        partitions,
        interference: interference.map(crate::analysis::contract_identity::interference_snapshot),
        distribution_factor_laws: None,
        nested_counterfactual_fit: None,
        linear_fit_moments: None,
        participants: participants.cloned(),
        missing,
        sampling_design,
    })
}

/// The ordered partitions of a data input, read from its unified source contract.
///
/// Each table, series, panel unit and environment is snapshotted as a source whose contract
/// binds modality, unit label and time index, and the content digest is verified against the
/// storage before it is recorded. A panel or multi-environment input is also sealed as one
/// composite identity, so a part that cannot be scanned refuses the whole snapshot.
pub(super) fn source_partitions(
    data_input: &DataInput,
) -> Result<Vec<DataPartitionIdentityWire>, CausalError> {
    use antecedent_data::lazy_identity::FullContentIdentity;
    use antecedent_data::source::{OwnedSource, SegmentedSource, SourceModality};

    let partition = |source: &OwnedSource, unit_id: Option<u32>| {
        let full = FullContentIdentity::try_from_source(source)?;
        let contract = source.contract();
        Ok::<_, CausalError>(DataPartitionIdentityWire {
            content: full.digest(),
            row_count: u64_count(contract.rows)?,
            unit_id,
            regularity: contract.time_index.as_ref().map(|index| regularity_tag(&index.regularity)),
        })
    };
    let segmented = |source: &SegmentedSource| {
        FullContentIdentity::try_from_segmented(source)?;
        source
            .parts()
            .iter()
            .enumerate()
            .map(|(index, part)| partition(part, source.unit_id(index)))
            .collect::<Result<Vec<_>, CausalError>>()
    };
    match data_input {
        DataInput::Tabular(data) => Ok(vec![partition(&OwnedSource::from_tabular(data), None)?]),
        DataInput::Temporal(data) | DataInput::Event(data) => Ok(vec![partition(
            &OwnedSource::from_time_series(data, SourceModality::TimeSeries)?,
            None,
        )?]),
        DataInput::MultiEnv(data) => segmented(&SegmentedSource::from_multi_environment(data)),
        DataInput::Panel(data) => segmented(&SegmentedSource::from_panel(data)),
    }
}

pub(super) fn u64_count(value: usize) -> Result<u64, CausalError> {
    u64::try_from(value).map_err(|_| CausalError::Compile { message: "count exceeds u64".into() })
}

pub(super) fn regularity_tag(regularity: &antecedent_data::SamplingRegularity) -> String {
    match regularity {
        antecedent_data::SamplingRegularity::Regular { interval_ns } => {
            format!("regular:{interval_ns}")
        }
        antecedent_data::SamplingRegularity::Irregular => "irregular".into(),
    }
}

pub(super) fn observation_tags(query: &antecedent_core::CausalQuery) -> Vec<String> {
    match query {
        antecedent_core::CausalQuery::Response(query) => {
            let mut tags: Vec<String> =
                query.observation_assumptions.iter().map(observation_assumption_tag).collect();
            if let Some(spec) = observation_spec_tag(&query.observation) {
                tags.push(spec);
            }
            tags
        }
        // The premises of a description: stated beside every association claim, none of them a
        // causal identification premise.
        antecedent_core::CausalQuery::Association(_) => vec![
            "association.descriptive_only".into(),
            "association.no_confounding_adjustment".into(),
            "association.uncalibrated_uncertainty".into(),
        ],
        _ => Vec::new(),
    }
}

pub(super) fn observation_spec_tag(spec: &antecedent_core::ObservationSpec) -> Option<String> {
    match spec {
        antecedent_core::ObservationSpec::Complete => None,
        antecedent_core::ObservationSpec::RightCensored { .. } => Some("right_censored".into()),
        antecedent_core::ObservationSpec::LeftCensored { .. } => Some("left_censored".into()),
        antecedent_core::ObservationSpec::IntervalCensored { .. } => {
            Some("interval_censored".into())
        }
        antecedent_core::ObservationSpec::Truncated { .. } => Some("truncated".into()),
        antecedent_core::ObservationSpec::Selected { .. } => Some("selected".into()),
    }
}

pub(super) fn observation_assumption_tag(
    assumption: &antecedent_core::ObservationAssumption,
) -> String {
    match assumption {
        antecedent_core::ObservationAssumption::IndependentGiven(_) => "independent_given".into(),
        antecedent_core::ObservationAssumption::OutcomeIndependentGiven(_) => {
            "outcome_independent_given".into()
        }
        antecedent_core::ObservationAssumption::Structural(name) => format!("structural:{name}"),
    }
}

pub(crate) fn graph_identity(study: &Study) -> Result<GraphIdentityWire, CausalError> {
    if let Some(posterior) = &study.graph_posterior {
        return Ok(GraphIdentityWire::GraphPosterior {
            graph_class: study.graph.class().as_str().into(),
            n_atoms: u64::try_from(posterior.n_graphs()).unwrap_or(u64::MAX),
            atoms: graph_posterior_atom_identities(posterior).map_err(|err| io_err(&err))?,
        });
    }
    accepted_graph_identity(&study.graph)
}

pub(crate) fn accepted_graph_identity(
    graph: &AcceptedGraph,
) -> Result<GraphIdentityWire, CausalError> {
    match graph.graph_ref() {
        GraphRef::Dag(g) => dag_identity(g).map_err(|err| io_err(&err)),
        GraphRef::Admg(g) => admg_identity(g).map_err(|err| io_err(&err)),
        GraphRef::Cpdag(g) => cpdag_identity(g).map_err(|err| io_err(&err)),
        GraphRef::Pag(g) => pag_identity(g).map_err(|err| io_err(&err)),
        GraphRef::TemporalDag(g) => temporal_dag_identity(g).map_err(|err| io_err(&err)),
        GraphRef::TemporalCpdag(g) => Ok(temporal_cpdag_identity(g)),
        GraphRef::TemporalPag(g) => Ok(temporal_pag_identity(g)),
        GraphRef::TemporalAdmg(g) => temporal_admg_identity(g).map_err(|err| io_err(&err)),
        GraphRef::RandomizedTrial => Ok(GraphIdentityWire::RandomizedTrial),
    }
}

/// Estimator name the contract may advertise.
///
/// A builder placeholder ([`Study::caller_selected_estimator`] is `None`) is
/// advertised only while it agrees with the plan, or before a plan resolves
/// one: `inference()` binds `BayesianGcomp` on every static average effect,
/// including classes whose compiler keeps an identifier-native estimator
/// (`functional.effect` on an ADMG), and advertising both names makes the
/// functional-program contract fail to verify. A caller-selected estimator is
/// kept even when it disagrees; the verifier then refuses it.
pub(super) fn advertised_estimator(
    study: &Study,
    resolved_estimator: Option<&str>,
) -> Option<&'static str> {
    study
        .caller_selected_estimator()
        .or_else(|| {
            study.declared.estimator.filter(|placeholder| {
                resolved_estimator.is_none_or(|resolved| resolved == placeholder.as_str())
            })
        })
        .map(|id| id.as_str())
        .or_else(|| {
            // A temporal response (a derivative included) has one estimator per inference
            // mode and the builder binds no placeholder; the plan's choice is the one the
            // program runs, so the contract names it.
            let temporal = matches!(&study.query, CausalQuery::Response(q) if q.temporal.is_some());
            resolved_estimator
                .filter(|_| temporal)
                .and_then(|name| name.parse::<crate::EstimatorId>().ok())
                .map(|id| id.as_str())
        })
}

/// Inferential commitments of the compiled program. `resolved_estimator` is
/// the prepared plan's `logical_plan.estimator`; cheap inspection compiles no
/// plan and leaves it `None`.
pub(super) fn inferential_commitments(
    study: &Study,
    resolved_estimator: Option<&str>,
) -> InferentialCommitmentsWire {
    let (mut interval_method, mut se_kind) = compiled_interval(study);
    if let Some(antecedent_io::EstimatorSpecWire::FrontDoorTwoStage(config)) =
        study.estimator_spec_identity.as_ref()
    {
        se_kind.clone_from(&config.se_kind);
        if config.bootstrap_replicates > 0 {
            interval_method = IntervalMethod::BootstrapSe;
        }
    }
    InferentialCommitmentsWire {
        format: IDENTITY_FORMAT,
        estimator: advertised_estimator(study, resolved_estimator).map(str::to_string),
        resolved_estimator: resolved_estimator.map(str::to_string),
        identifier: study.declared.identifier.map(|id| id.as_str().to_string()),
        // The commitments bind the estimator's own inference configuration, which
        // the inference binding replays; a structure posterior's prior and score are
        // in its query and lowered program, not in a Bayesian estimator configuration.
        inference: match study.settings.inference {
            InferenceMode::Frequentist => "frequentist".into(),
            InferenceMode::Bayesian(_) => "bayesian".into(),
        },
        validation_suite: study.settings.refute.validation_suite_id().map(str::to_string),
        interval_method: interval_method.as_str().into(),
        se_kind,
        prior_required: matches!(study.settings.inference, InferenceMode::Bayesian(_)),
    }
}

/// The inference family a study's program declares, as the contract labels
/// it. A structure search declares its method's: a structure posterior is
/// Bayesian by its method, whatever the study's estimator configuration.
pub(super) fn declared_inference(study: &Study) -> &'static str {
    if let CausalQuery::Discover(query) = &study.query {
        return query.inference().as_str();
    }
    match study.settings.inference {
        InferenceMode::Frequentist => "frequentist",
        InferenceMode::Bayesian(_) => "bayesian",
    }
}

/// Query name of a compiled route kind that sits off the support-matrix axes.
pub(super) const fn compiled_off_axis_kind(query: &CausalQuery) -> Option<&'static str> {
    match query {
        CausalQuery::Discover(_) => Some("Discover"),
        CausalQuery::Attribute(_) => Some("Attribute"),
        CausalQuery::Simulate(_) => Some("Simulate"),
        CausalQuery::TemporalDesign(_) => Some("TemporalDesign"),
        CausalQuery::IvBounds(_) => Some("IvBounds"),
        CausalQuery::Association(_) => Some("Association"),
        _ => None,
    }
}

pub(super) fn compiled_interval(study: &Study) -> (IntervalMethod, Option<String>) {
    match &study.settings.inference {
        InferenceMode::Bayesian(_) => (IntervalMethod::PosteriorQuantile, None),
        InferenceMode::Frequentist if study.settings.bootstrap_replicates > 0 => {
            (IntervalMethod::BootstrapSe, None)
        }
        InferenceMode::Frequentist => (
            IntervalMethod::AnalyticSe,
            study.rd.as_ref().map(|cfg| cfg.se_kind.as_str().to_string()),
        ),
    }
}

pub(super) fn population_depends_on(
    query: &CausalQuery,
    registry: Option<&antecedent_core::PopulationRegistry>,
) -> Vec<u32> {
    match query.target_population() {
        Some(TargetPopulation::CustomDistribution(id)) => registry
            .and_then(|reg| reg.distribution_dependencies(*id))
            .unwrap_or(&[])
            .iter()
            .map(|id| id.raw())
            .collect(),
        // Row weights carry their declared parents on the target population
        // itself; every other population declares no covariate dependence.
        _ => Vec::new(),
    }
}

pub(super) fn analysis_row_count(result: &StudyResult, contract: &CausalContract) -> u64 {
    result
        .estimate
        .as_effect()
        .and_then(|estimate| {
            estimate
                .provenance
                .n_obs
                .or_else(|| estimate.scores.influence.as_ref().map(|rows| rows.len() as u64))
                .or_else(|| estimate.scores.table.as_ref().map(|table| table.n_rows as u64))
                .or_else(|| estimate.uncertainty.block.resampling.map(|block| block.rows as u64))
        })
        .unwrap_or(contract.row_count)
}

pub(super) fn data_row_count(data: &DataInput) -> u64 {
    match data {
        DataInput::Tabular(data) => data.row_count() as u64,
        DataInput::Temporal(data) | DataInput::Event(data) => data.row_count() as u64,
        DataInput::Panel(data) => data.total_rows() as u64,
        DataInput::MultiEnv(data) => {
            data.environments().iter().map(TableView::row_count).sum::<usize>() as u64
        }
    }
}
