//! Builder independent execution for the fixed network interference contract.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
enum InterferenceProcedure {
    DesignBasedYoung,
    ConjugateGaussian { draws: usize, prior_sd: f64 },
}

/// Complete fixed network, assignment, query and inference procedure for one exposure contrast.
#[derive(Clone)]
pub(crate) struct CheckedInterferenceOperation {
    graph: Dag,
    query: antecedent_core::InterferenceQuery,
    source_schema: antecedent_core::CausalSchema,
    source_rows: usize,
    graph_signature: (usize, Arc<[(u32, u32)]>),
    network_edges: Arc<[antecedent_data::NetworkEdge]>,
    assignment: Arc<[bool]>,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    procedure: InterferenceProcedure,
    estimator: EstimatorId,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedInterferenceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedInterferenceOperation")
            .field("query", &self.query)
            .field("graph_signature", &self.graph_signature)
            .field("source_rows", &self.source_rows)
            .field("network_edges", &self.network_edges.len())
            .field("procedure", &self.procedure)
            .field("estimator", &self.estimator)
            .finish_non_exhaustive()
    }
}

impl CheckedInterferenceOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Interference(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "checked interference requires an exposure contrast target".into(),
            });
        };
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        let graph = study.graph.as_dag().ok_or(CausalError::Unsupported {
            message: "checked interference requires a supplied DAG schema",
        })?;
        let design = study.interference.as_ref().ok_or(CausalError::Unsupported {
            message: "checked interference requires a fixed network and realized assignment",
        })?;
        let design = design.bound_to(data)?;
        if study.structure_source != crate::support::StructureSource::Explicit
            || study.tiered.is_some()
            || study.graph_posterior.is_some()
            || study.refute != RefuteSuite::None
            || !study.custom_validators.is_empty()
            || design.assignment.len() != data.row_count()
            || design.network.units().row_count() != data.row_count()
            || graph.node_count() != data.schema().variables().len()
        {
            return Err(CausalError::Unsupported {
                message: "checked interference is limited to an explicit fixed DAG schema and no validation suite",
            });
        }
        let (procedure, estimator) = match &study.inference {
            InferenceMode::Frequentist => {
                if !matches!(query.assignment, antecedent_core::AssignmentDesign::Bernoulli { .. })
                    || query.exposure != antecedent_core::ExposureMapping::NeighborCount
                {
                    return Err(CausalError::Unsupported {
                        message: "the licensed design based interference route requires Bernoulli assignment with NeighborCount exposure",
                    });
                }
                (InterferenceProcedure::DesignBasedYoung, EstimatorId::InterferenceHtHajek)
            }
            InferenceMode::Bayesian(config) => {
                if config.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
                    || config.likelihood != antecedent_prob::BayesLikelihood::GaussianIdentity
                    || config.prior.is_some()
                    || config.prior_artifact.is_some()
                    || config.prior_mapping.is_some()
                    || config.external_compose.is_some()
                    || !matches!(
                        query.assignment,
                        antecedent_core::AssignmentDesign::Bernoulli { .. }
                    )
                    || query.exposure != antecedent_core::ExposureMapping::NeighborCount
                {
                    return Err(CausalError::Unsupported {
                        message: "the licensed Bayesian interference route requires Bernoulli assignment, NeighborCount exposure and the conjugate Gaussian model with its declared isotropic prior",
                    });
                }
                if config.n_draws < 2
                    || !config.prior_scale.is_finite()
                    || config.prior_scale <= 0.0
                {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian interference requires at least two posterior draws and a positive finite prior scale",
                    });
                }
                (
                    InterferenceProcedure::ConjugateGaussian {
                        draws: config.n_draws,
                        prior_sd: config.prior_scale,
                    },
                    EstimatorId::InterferenceBayesianGaussian,
                )
            }
        };
        if physical.logical.query != study.query
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::InterferenceDesign.as_str())
            || physical.logical.record.estimator.as_deref() != Some(estimator.as_str())
        {
            return Err(CausalError::Compile {
                message: "interference target, fixed design or selected estimator differs from its checked plan".into(),
            });
        }
        let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
            query.functional;
        if data.schema().get(outcome).is_err() {
            return Err(CausalError::Unsupported {
                message: "interference outcome is absent from the bound unit table",
            });
        }
        let (identification, estimand) = interference_identification(
            CausalQuery::Interference(query.clone()),
            outcome,
            &procedure,
        );
        Ok(Self {
            graph: graph.clone(),
            query: query.clone(),
            source_schema: data.schema().clone(),
            source_rows: data.row_count(),
            graph_signature: graph_signature(graph),
            network_edges: Arc::from(design.network.edges()),
            assignment: Arc::clone(&design.assignment),
            identification,
            estimand,
            procedure,
            estimator,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn query(&self) -> &antecedent_core::InterferenceQuery {
        &self.query
    }

    pub(crate) fn procedure(&self) -> (IdentifierId, EstimatorId) {
        (IdentifierId::InterferenceDesign, self.estimator)
    }

    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.source_rows, self.network_edges.len())
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.source_schema || data.row_count() != self.source_rows {
            return Err(CausalError::Unsupported {
                message: "interference execution requires the prepared unit schema and row identity order",
            });
        }
        if self.result_context.query != CausalQuery::Interference(self.query.clone())
            || self.identification.query != CausalQuery::Interference(self.query.clone())
            || graph_signature(&self.graph) != self.graph_signature
        {
            return Err(CausalError::Compile {
                message: "checked interference operation lost its target or source graph binding"
                    .into(),
            });
        }
        self.query
            .validate()
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let network =
            antecedent_data::NetworkData::try_new(data.clone(), Arc::clone(&self.network_edges))?;
        let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
            self.query.functional;
        let started = Instant::now();
        let (estimated, posterior, estimate, diagnostic) = match self.procedure {
            InterferenceProcedure::DesignBasedYoung => {
                let seed =
                    ctx.rng.stream_for(antecedent_core::StreamDomain::Transport, 0x1F7E).next_u64();
                let estimated = antecedent_estimate::estimate_interference(
                    &self.query,
                    &network,
                    &self.assignment,
                    seed,
                )
                .map_err(CausalError::from)?;
                let se = estimated.contrast.conservative_variance.sqrt();
                (
                    Some(estimated.clone()),
                    None,
                    EffectEstimate::new(
                        estimated.contrast.horvitz_thompson,
                        se,
                        self.identification.required_assumptions.clone(),
                        OverlapPolicy::ExplicitOverride,
                    ),
                    Diagnostic::new(
                        "estimate.interference.young_bound",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "conservative Young variance bound; not the Aronow–Samii joint-exposure variance",
                    ),
                )
            }
            InterferenceProcedure::ConjugateGaussian { draws, prior_sd } => {
                let seed =
                    ctx.rng.stream_for(antecedent_core::StreamDomain::Bayesian, 0x1F7E).next_u64();
                let estimated = antecedent_estimate::estimate_interference_bayesian(
                    &self.query,
                    &network,
                    &self.assignment,
                    draws,
                    seed,
                    prior_sd,
                )
                .map_err(CausalError::from)?;
                let assumptions = interference_bayesian_assumptions(prior_sd);
                let schema = antecedent_prob::PosteriorSchema {
                    quantities: Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
                        name: Arc::from("finite_network_exposure_contrast"),
                    }]),
                };
                let posterior_draws = antecedent_prob::PosteriorDraws::from_column_major(
                    schema,
                    estimated.contrast_draws.len(),
                    Arc::<[f64]>::from(estimated.contrast_draws),
                )
                .map_err(|error| CausalError::Compile { message: error.to_string() })?;
                let posterior = CausalPosterior {
                    summaries: posterior_draws.summarize(),
                    draws: posterior_draws,
                    identification: self.identification.status,
                    prior_sensitivity: None,
                    conflict_summary: None,
                    diagnostics: InferenceDiagnostics::analytic("interference.bayesian_gaussian"),
                    assumptions,
                    unidentified_mass: 0.0,
                    subsampled_out_mass: 0.0,
                    unevaluable_mass: 0.0,
                    early_stopped: false,
                    treatment_contrast: None,
                };
                let estimate = effect_from_posterior(&posterior)?;
                (
                    None,
                    Some(posterior),
                    estimate,
                    Diagnostic::new(
                        "estimate.interference.bayesian_gaussian",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "finite-network posterior under an additive Gaussian potential-outcome model with shared unit disturbances; conditional on fixed network and realized assignment",
                    ),
                )
            }
        };
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(),
                estimand: self.estimand.clone(),
                estimate,
                identifier_id: IdentifierId::InterferenceDesign,
                estimator_id: self.estimator,
                treatment: outcome,
                outcome,
                identify_cached: true,
                extra_diagnostics: vec![diagnostic],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.interference = estimated;
        result.posterior = posterior;
        Ok(result)
    }
}

fn interference_identification(
    query: CausalQuery,
    outcome: VariableId,
    procedure: &InterferenceProcedure,
) -> (IdentificationResult, IdentifiedEstimand) {
    let (rule, note, assumptions) = match procedure {
        InterferenceProcedure::DesignBasedYoung => (
            "interference.design",
            "Identification is the known Bernoulli assignment and NeighborCount exposure mapping (Horvitz–Thompson/Hájek); the DAG binds the outcome schema only.",
            design_identification_assumptions(),
        ),
        InterferenceProcedure::ConjugateGaussian { prior_sd, .. } => (
            "interference.bayesian_gaussian",
            "Finite-network contrast under the declared additive Gaussian potential-outcome model and shared unit disturbance.",
            interference_bayesian_assumptions(*prior_sd),
        ),
    };
    let mut arena = CausalExprArena::new();
    let outcome_set = arena.intern_var_set([outcome]);
    let intervention_set = arena.intern_intervention_set([outcome]);
    let empty = arena.empty_var_set();
    let distribution = arena.intern_distribution(
        outcome_set,
        empty,
        intervention_set,
        antecedent_expr::DomainRef::Interventional,
    );
    let functional = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(outcome),
        distribution,
    });
    arena.set_derivation(
        functional,
        antecedent_expr::DerivationMeta::rule(rule, Some(Arc::from(note))),
    );
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut trace = DerivationTrace::default();
    trace.push(rule, note);
    let mut identification = IdentificationResult::identified(
        query,
        vec![estimand.clone()],
        arena,
        trace,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    if matches!(procedure, InterferenceProcedure::ConjugateGaussian { .. }) {
        identification.status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
    }
    (identification, estimand)
}

fn design_identification_assumptions() -> antecedent_core::AssumptionSet {
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
        AssumptionStatus,
    };
    let mut assumptions = AssumptionSet::default();
    assumptions.push(AssumptionRecord {
        assumption: Assumption::Custom {
            id: Arc::from("interference.design"),
            description: Arc::from("Identification is the known assignment mechanism and exposure mapping; the graph binds the unit outcome schema only."),
        },
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("interference.design") },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
    assumptions
}

fn graph_signature(graph: &Dag) -> (usize, Arc<[(u32, u32)]>) {
    let mut edges = graph.edges().map(|edge| (edge.a.raw(), edge.b.raw())).collect::<Vec<_>>();
    edges.sort_unstable();
    (graph.node_count(), edges.into())
}

fn interference_bayesian_assumptions(prior_sd: f64) -> antecedent_core::AssumptionSet {
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
        AssumptionStatus, ParametricAssumption,
    };
    let algorithm: Arc<str> = Arc::from("interference.bayesian_gaussian");
    let mut assumptions = AssumptionSet::default();
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("interference.fixed_network_gaussian_potential_outcomes"),
            description: Arc::from("Finite-network target over observed units under Y_i(z,g)=alpha+beta*z+gamma*g+epsilon_i, with a shared unit disturbance across exposure-specific potential outcomes; Gaussian identity likelihood with residual variance fixed to one."),
        }),
        source: AssumptionSource::AlgorithmDefault { algorithm: algorithm.clone() },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("interference.gaussian_coefficient_prior"),
            description: Arc::from(format!("Independent Normal(0, {prior_sd}²) prior on intercept, own-treatment, and treated-neighbor-count coefficients.")),
        }),
        source: AssumptionSource::AlgorithmDefault { algorithm },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions
}

#[cfg(test)]
mod checked_interference_tests {
    use super::*;
    use antecedent_core::{
        AssignmentDesign, CausalQuery, ExposureLevel, ExposureMapping, InterferenceFunctional,
        InterferenceQuery, VariableId,
    };
    use antecedent_data::{NetworkData, NetworkEdge, TabularData};

    fn context() -> ExecutionContext {
        ExecutionContext::for_tests(0x1F7E)
    }

    fn dag() -> Dag {
        Dag::with_variables(1)
    }

    fn operation(
        data: &TabularData,
        graph: Dag,
        query: InterferenceQuery,
        edges: Vec<NetworkEdge>,
        assignment: Vec<bool>,
        inference: InferenceMode,
    ) -> CheckedInterferenceOperation {
        let builder = crate::Study::tabular(data.clone())
            .graph(graph)
            .query(CausalQuery::Interference(query))
            .interference(crate::InterferenceSpec {
                network: NetworkData::try_new(data.clone(), edges).unwrap(),
                assignment: Arc::from(assignment),
            })
            .inference(inference)
            .refute(RefuteSuite::None);
        let study = builder.build().expect("study build");
        let physical = study.plan(&context()).expect("physical plan");
        let operation = CheckedInterferenceOperation::checked(&study, data, &physical)
            .expect("checked interference operation");
        drop(study);
        operation
    }

    fn contrast_query() -> InterferenceQuery {
        InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 1.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.0 },
            },
        )
    }

    #[test]
    fn design_based_operation_replays_fixed_network_truth_after_builder_drop() {
        let truth: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/response/randomized_interference/expected.json"
        ))
        .unwrap();
        let outcomes = truth["outcomes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let edges = vec![
            NetworkEdge { from: 0, to: 1, weight: 1.0 },
            NetworkEdge { from: 1, to: 0, weight: 1.0 },
        ];
        let assignment = truth["assignment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_bool().unwrap())
            .collect::<Vec<_>>();
        let operation = operation(
            &data,
            dag(),
            contrast_query(),
            edges,
            assignment,
            InferenceMode::Frequentist,
        );
        let result = operation.execute(&data, &context()).expect("sealed execution");
        let expected = truth["expected"]["horvitz_thompson_contrast"].as_f64().unwrap();
        let se = truth["expected"]["conservative_variance"].as_f64().unwrap().sqrt();
        assert!((result.estimate.ate - expected).abs() < 1e-12);
        assert!((result.estimate.se_analytic - se).abs() < 1e-12);
        assert!(result.interference.is_some());
        assert!(result.posterior.is_none());
    }

    #[test]
    fn bayesian_operation_retains_conjugate_model_and_known_finite_network_truth() {
        let truth: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/estimate/bayesian_interference/expected.json"
        ))
        .unwrap();
        let outcomes = truth["outcomes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let edges = truth["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|edge| NetworkEdge {
                from: edge[0].as_u64().unwrap() as u32,
                to: edge[1].as_u64().unwrap() as u32,
                weight: 1.0,
            })
            .collect::<Vec<_>>();
        let assignment = truth["assignment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_bool().unwrap())
            .collect::<Vec<_>>();
        let query = contrast_query();
        let operation = operation(
            &data,
            dag(),
            query,
            edges,
            assignment,
            InferenceMode::Bayesian(
                crate::BayesianConfig::conjugate().n_draws(8_192).prior_scale(100.0),
            ),
        );
        let result = operation.execute(&data, &context()).expect("sealed Bayesian execution");
        let posterior = result.posterior.as_ref().expect("finite-network posterior");
        assert_eq!(posterior.draws.n_draws, 8_192);
        let expected = truth["expected_contrast"].as_f64().unwrap();
        let tolerance = truth["tolerance"].as_f64().unwrap();
        assert!(
            (posterior.summaries.mean[0] - expected).abs() < tolerance,
            "posterior mean {}, expected {expected} ± {tolerance}",
            posterior.summaries.mean[0]
        );
        assert_eq!(operation.procedure().1, EstimatorId::InterferenceBayesianGaussian);
        assert!(result.interference.is_none());
    }

    #[test]
    fn checked_refresh_preserves_network_order_and_rejects_changed_unit_schema() {
        let data = TabularData::from_f64_columns([("y", &[1.0, 4.0][..])]).unwrap();
        let operation = operation(
            &data,
            dag(),
            contrast_query(),
            vec![
                NetworkEdge { from: 0, to: 1, weight: 1.0 },
                NetworkEdge { from: 1, to: 0, weight: 1.0 },
            ],
            vec![true, false],
            InferenceMode::Frequentist,
        );
        let refreshed = TabularData::from_f64_columns([("y", &[2.0, 8.0][..])]).unwrap();
        assert!(operation.execute(&refreshed, &context()).is_ok());
        let reordered = TabularData::from_f64_columns([("x", &[2.0, 8.0][..])]).unwrap();
        assert!(operation.execute(&reordered, &context()).is_err());
    }
}
