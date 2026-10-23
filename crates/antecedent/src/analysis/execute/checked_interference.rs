//! Builder independent execution for the fixed network interference contract.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalExprArena, CausalPosterior, CausalQuery, Dag, DerivationTrace,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext,
    IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteFinish, IdentifiedResultContext, IdentifierId,
    InferenceDiagnostics, InferenceMode, Instant, OverlapPolicy, PhysicalExecutionPlan,
    RefuteSuite, Strategy, Study, StudyResult, TableView, TabularData, VariableId,
    effect_from_posterior, elapsed_ns, finish_identified_execute_with_context,
};
use crate::analysis::prepared::Inspect;

#[derive(Clone, Copy, Debug, PartialEq)]
enum InterferenceProcedure {
    DesignBasedYoung,
    SaturationExact,
    ObservationalIpw,
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
    physical: std::sync::Arc<PhysicalExecutionPlan>,
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
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Interference(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "checked interference requires an exposure contrast target".into(),
            });
        };
        query.validate()?;
        let graph = study.graph.as_dag().ok_or(CausalError::Unsupported {
            message: "checked interference requires a supplied DAG schema",
        })?;
        let design = study.interference.as_ref().ok_or(CausalError::Unsupported {
            message: "checked interference requires a fixed network and realized assignment",
        })?;
        let design = design.bound_to(data)?;
        refuse_unsupported_interference_design(study, &design, data, graph)?;
        let (procedure, estimator) = match &study.inference {
            InferenceMode::Frequentist => frequentist_interference_procedure(query, &design)?,
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
            || !physical.binding().names(IdentifierId::InterferenceDesign, estimator)
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
            partial_interference(query),
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

    /// The identification this operation records for `query` under `inference`: the
    /// procedure [`Self::checked`] selects for the declared assignment, without the
    /// prepare-time design validation that needs the bound network. The route lowering
    /// retains it at compile.
    pub(crate) fn design_identification(
        query: &antecedent_core::InterferenceQuery,
        inference: &InferenceMode,
    ) -> (IdentificationResult, IdentifiedEstimand) {
        let procedure = match inference {
            InferenceMode::Bayesian(config) => InterferenceProcedure::ConjugateGaussian {
                draws: config.n_draws,
                prior_sd: config.prior_scale,
            },
            InferenceMode::Frequentist => match query.assignment {
                antecedent_core::AssignmentDesign::ObservedExposure { .. } => {
                    InterferenceProcedure::ObservationalIpw
                }
                antecedent_core::AssignmentDesign::TwoStageSaturation { .. } => {
                    InterferenceProcedure::SaturationExact
                }
                _ => InterferenceProcedure::DesignBasedYoung,
            },
        };
        let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
            query.functional;
        interference_identification(
            CausalQuery::Interference(query.clone()),
            outcome,
            &procedure,
            partial_interference(query),
        )
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
        self.query.validate()?;
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
        let (estimated, interference_inference, posterior, estimate, diagnostic) =
            match self.procedure {
                InterferenceProcedure::DesignBasedYoung
                | InterferenceProcedure::SaturationExact => self.design_based_arm(&network, ctx)?,
                InterferenceProcedure::ObservationalIpw => self.observational_arm(&network)?,
                InterferenceProcedure::ConjugateGaussian { draws, prior_sd } => {
                    self.bayesian_arm(&network, ctx, draws, prior_sd)?
                }
            };
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.identification.clone(),
                self.estimand.clone(),
                estimate,
                Strategy::new(IdentifierId::InterferenceDesign, self.estimator),
                outcome,
                outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(vec![diagnostic])
            .wall_time_ns(elapsed_ns(started)),
        );
        result.payload =
            crate::result::ResultPayload::from_interference(estimated, interference_inference);
        result.posterior = posterior;
        Ok(result)
    }

    /// Design-based, cluster-total and saturation arms share the Young-bound seed draw and result shape.
    fn design_based_arm(
        &self,
        network: &antecedent_data::NetworkData,
        ctx: &ExecutionContext,
    ) -> Result<InterferenceArm, CausalError> {
        let seed = ctx.rng.stream_for(antecedent_core::StreamDomain::Transport, 0x1F7E).next_u64();
        let saturation = matches!(self.procedure, InterferenceProcedure::SaturationExact);
        let (estimated, interference_inference) = if saturation {
            self.saturation_inference(network)?
        } else if matches!(
            self.query.assignment,
            antecedent_core::AssignmentDesign::ClusterRandomization { .. }
        ) {
            self.cluster_inference(network)?
        } else {
            (
                antecedent_estimate::estimate_interference(
                    &self.query,
                    network,
                    &self.assignment,
                    seed,
                )
                .map_err(CausalError::from)?,
                None,
            )
        };
        let se =
            if saturation { f64::NAN } else { estimated.contrast.conservative_variance.sqrt() };
        let cluster = matches!(
            self.query.assignment,
            antecedent_core::AssignmentDesign::ClusterRandomization { .. }
        );
        Ok((
            Some(estimated.clone()),
            interference_inference,
            None,
            EffectEstimate::new(
                estimated.contrast.horvitz_thompson,
                se,
                self.identification.required_assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            ),
            Diagnostic::new(
                if saturation {
                    "estimate.interference.saturation_exact"
                } else if cluster {
                    "estimate.interference.cluster_neyman"
                } else {
                    "estimate.interference.young_bound"
                },
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                if saturation {
                    "Exact two-stage saturation exposure probabilities; independent-cluster pointwise 95% interval is reported only when first-stage and realized-exposure support pass"
                } else if cluster {
                    "Cluster-level Neyman pointwise 95% interval for the total effect under partial interference, when independent-cluster support passes"
                } else {
                    "conservative Young variance bound; not the Aronow–Samii joint-exposure variance"
                },
            ),
        ))
    }

    /// Exact two-stage saturation estimate with its cluster-level Welch interval.
    fn saturation_inference(
        &self,
        network: &antecedent_data::NetworkData,
    ) -> Result<
        (antecedent_estimate::InterferenceEstimate, Option<crate::result::InterferenceInference>),
        CausalError,
    > {
        let value = antecedent_estimate::estimate_saturation_interference(
            &self.query,
            network,
            &self.assignment,
        )
        .map_err(CausalError::from)?;
        let interval =
            value.pointwise_interval.map(|interval| crate::result::InterferencePointwiseInterval {
                lower: interval.bounds[0],
                upper: interval.bounds[1],
                standard_error: interval.standard_error,
                degrees_of_freedom: interval.degrees_of_freedom,
                first_stage_arm_clusters: [interval.low_clusters, interval.high_clusters],
            });
        let graphless_support_status = interval.as_ref().and_then(|interval| {
            interference_graphless_license(
                "two_stage_saturation",
                "saturation_cluster_neyman_welch",
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "cluster",
                    // first_stage_arm_clusters is [low, high] saturation arms.
                    control: interval.first_stage_arm_clusters[0],
                    treated: interval.first_stage_arm_clusters[1],
                    blocks: 2,
                    min_block_arm: value.from_exposed_clusters.min(value.to_exposed_clusters),
                    interval_95_published: true,
                    reported_intervals: 1,
                    ..Default::default()
                },
            )
        });
        let inference = crate::result::InterferenceInference {
            method: "saturation_cluster_neyman_welch",
            graphless_support_status,
            interval,
            interval_unavailable_reason: value.interval_unavailable_reason,
            from_exposed_units: value.from_exposed_units,
            to_exposed_units: value.to_exposed_units,
            from_exposed_clusters: value.from_exposed_clusters,
            to_exposed_clusters: value.to_exposed_clusters,
        };
        Ok((value.estimate, Some(inference)))
    }

    /// Cluster-total Neyman estimate with its between-cluster interval and arm-size reasons.
    fn cluster_inference(
        &self,
        network: &antecedent_data::NetworkData,
    ) -> Result<
        (antecedent_estimate::InterferenceEstimate, Option<crate::result::InterferenceInference>),
        CausalError,
    > {
        let (value, interval) =
            antecedent_estimate::estimate_cluster_interference_total_with_inference(
                &self.query,
                network,
                &self.assignment,
            )
            .map_err(CausalError::from)?;
        let (control_clusters, treated_clusters) = match &self.query.assignment {
            antecedent_core::AssignmentDesign::ClusterRandomization {
                clusters,
                treated_clusters,
            } => {
                let independent =
                    clusters.iter().copied().collect::<std::collections::BTreeSet<_>>().len();
                (independent - *treated_clusters, *treated_clusters)
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "cluster-total inference requires a cluster-randomization assignment"
                        .into(),
                });
            }
        };
        let interval_available = interval.is_some();
        let to_exposed_units = self.assignment.iter().filter(|&&assigned| assigned).count();
        let graphless_support_status = interval.as_ref().and_then(|_| {
            interference_graphless_license(
                "cluster_randomization_total",
                "cluster_total_neyman_welch",
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "cluster",
                    treated: treated_clusters,
                    control: control_clusters,
                    interval_95_published: true,
                    reported_intervals: 1,
                    ..Default::default()
                },
            )
        });
        let inference = crate::result::InterferenceInference {
            method: "cluster_total_neyman_welch",
            graphless_support_status,
            interval: interval.map(|interval| crate::result::InterferencePointwiseInterval {
                lower: interval.bounds[0],
                upper: interval.bounds[1],
                standard_error: interval.standard_error,
                degrees_of_freedom: interval.degrees_of_freedom,
                first_stage_arm_clusters: [interval.control_clusters, interval.treated_clusters],
            }),
            interval_unavailable_reason: if control_clusters < 8 || treated_clusters < 8 {
                Some(
                    "pointwise cluster-total inference requires eight independent clusters in each assignment arm",
                )
            } else if !interval_available {
                Some(
                    "pointwise cluster-total inference requires positive finite between-cluster variation",
                )
            } else {
                None
            },
            from_exposed_units: self.assignment.len() - to_exposed_units,
            to_exposed_units,
            from_exposed_clusters: control_clusters,
            to_exposed_clusters: treated_clusters,
        };
        Ok((value, Some(inference)))
    }

    /// Observational exposure contrast with supplied probabilities and its cluster-t interval.
    fn observational_arm(
        &self,
        network: &antecedent_data::NetworkData,
    ) -> Result<InterferenceArm, CausalError> {
        let summary = estimate_observed_exposure(&self.query, network, &self.assignment)?;
        let interval = summary.pointwise_interval.as_ref().map(|interval| {
            crate::result::InterferencePointwiseInterval {
                lower: interval.bounds[0],
                upper: interval.bounds[1],
                standard_error: interval.standard_error,
                degrees_of_freedom: interval.degrees_of_freedom,
                first_stage_arm_clusters: [
                    summary.from_exposed_clusters,
                    summary.to_exposed_clusters,
                ],
            }
        });
        let graphless_support_status = interval.as_ref().and_then(|_| {
            interference_graphless_license(
                "observational_known_exposure",
                "observational_known_exposure_cluster_t",
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "cluster",
                    treated: summary.to_exposed_clusters,
                    control: summary.from_exposed_clusters,
                    // Independent partial-interference clusters are the units of inference.
                    blocks: summary.clusters,
                    interval_95_published: true,
                    reported_intervals: 1,
                    known_density: summary.propensity_provenance
                        == antecedent_core::ExposurePropensityProvenance::Known,
                    ..Default::default()
                },
            )
        });
        let interference_inference = crate::result::InterferenceInference {
            method: "observational_known_exposure_cluster_t",
            graphless_support_status,
            interval,
            interval_unavailable_reason: summary.interval_unavailable_reason,
            from_exposed_units: summary.from_exposed_units,
            to_exposed_units: summary.to_exposed_units,
            from_exposed_clusters: summary.from_exposed_clusters,
            to_exposed_clusters: summary.to_exposed_clusters,
        };
        let method = match summary.propensity_provenance {
            antecedent_core::ExposurePropensityProvenance::Known => {
                antecedent_stats::ExposureProbabilityMethod::SuppliedKnown
            }
            antecedent_core::ExposurePropensityProvenance::ExternallyEstimated => {
                antecedent_stats::ExposureProbabilityMethod::SuppliedExternallyEstimated
            }
        };
        let estimated = antecedent_estimate::InterferenceEstimate {
            contrast: antecedent_stats::RandomizationContrast {
                horvitz_thompson: summary.horvitz_thompson,
                hajek: summary.hajek,
                conservative_variance: summary.cluster_robust_variance,
            },
            from_probability_method: method,
            to_probability_method: method,
            minimum_exposure_probability: summary.minimum_exposure_probability,
        };
        let diagnostic = Diagnostic::new(
                    "estimate.interference.observational_ipw",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "Observational network exposure contrast with supplied probabilities; a pointwise cluster interval is reported only for known fixed probabilities and sufficient independent-cluster exposure support",
                ).with_fields([
                    ("from_exposed_units", summary.from_exposed_units.to_string()),
                    ("to_exposed_units", summary.to_exposed_units.to_string()),
                    ("from_exposed_clusters", summary.from_exposed_clusters.to_string()),
                    ("to_exposed_clusters", summary.to_exposed_clusters.to_string()),
                    ("clusters", summary.clusters.to_string()),
                    ("minimum_exposure_probability", summary.minimum_exposure_probability.to_string()),
                    ("maximum_exposure_probability", summary.maximum_exposure_probability.to_string()),
                ]);
        Ok((
            Some(estimated),
            Some(interference_inference),
            None,
            EffectEstimate::new(
                summary.horvitz_thompson,
                summary
                    .pointwise_interval
                    .as_ref()
                    .map_or(f64::NAN, |interval| interval.standard_error),
                self.identification.required_assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            ),
            diagnostic,
        ))
    }

    /// Conjugate Gaussian finite-network posterior for the Bernoulli/NeighborCount contrast.
    fn bayesian_arm(
        &self,
        network: &antecedent_data::NetworkData,
        ctx: &ExecutionContext,
        draws: usize,
        prior_sd: f64,
    ) -> Result<InterferenceArm, CausalError> {
        let seed = ctx.rng.stream_for(antecedent_core::StreamDomain::Bayesian, 0x1F7E).next_u64();
        let estimated = antecedent_estimate::estimate_interference_bayesian(
            &self.query,
            network,
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
        )?;
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
            preparation: None,
        };
        let estimate = effect_from_posterior(&posterior)?;
        Ok((
            None,
            None,
            Some(posterior),
            estimate,
            Diagnostic::new(
                "estimate.interference.bayesian_gaussian",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "finite-network posterior under an additive Gaussian potential-outcome model with shared unit disturbances; conditional on fixed network and realized assignment",
            ),
        ))
    }
}

/// Estimate, inference block, posterior, effect and diagnostic one procedure arm produces.
type InterferenceArm = (
    Option<antecedent_estimate::InterferenceEstimate>,
    Option<crate::result::InterferenceInference>,
    Option<CausalPosterior>,
    EffectEstimate,
    Diagnostic,
);

/// Refuses studies outside the explicit fixed-DAG, bound-network, no-validation-suite contract.
fn refuse_unsupported_interference_design(
    study: &Study,
    design: &crate::InterferenceSpec,
    data: &TabularData,
    graph: &Dag,
) -> Result<(), CausalError> {
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
    Ok(())
}

/// Selects the design-based or observational procedure for the query and validates its design during prepare.
fn frequentist_interference_procedure(
    query: &antecedent_core::InterferenceQuery,
    design: &crate::InterferenceSpec,
) -> Result<(InterferenceProcedure, EstimatorId), CausalError> {
    let bernoulli = matches!(query.assignment, antecedent_core::AssignmentDesign::Bernoulli { .. })
        && query.exposure == antecedent_core::ExposureMapping::NeighborCount;
    let cluster =
        matches!(query.assignment, antecedent_core::AssignmentDesign::ClusterRandomization { .. })
            && query.exposure == antecedent_core::ExposureMapping::NeighborFraction;
    let saturation =
        matches!(query.assignment, antecedent_core::AssignmentDesign::TwoStageSaturation { .. })
            && matches!(
                query.exposure,
                antecedent_core::ExposureMapping::NeighborCount
                    | antecedent_core::ExposureMapping::NeighborFraction
                    | antecedent_core::ExposureMapping::WeightedNeighborExposure
            );
    let observed =
        matches!(query.assignment, antecedent_core::AssignmentDesign::ObservedExposure { .. })
            && matches!(
                query.exposure,
                antecedent_core::ExposureMapping::NeighborCount
                    | antecedent_core::ExposureMapping::NeighborFraction
                    | antecedent_core::ExposureMapping::WeightedNeighborExposure
            );
    if !bernoulli && !cluster && !saturation && !observed {
        return Err(CausalError::Unsupported {
            message: "interference requires Bernoulli/NeighborCount, cluster total, exact two-stage saturation, or supplied-propensity observational exposure",
        });
    }
    if cluster {
        // Reject unsupported neighborhood, assignment, and arm sizes during prepare,
        // before the checked operation is retained or an interval can be requested.
        antecedent_estimate::estimate_cluster_interference_total(
            query,
            &design.network,
            &design.assignment,
        )
        .map_err(CausalError::from)?;
    }
    if saturation {
        antecedent_estimate::estimate_saturation_interference(
            query,
            &design.network,
            &design.assignment,
        )
        .map_err(CausalError::from)?;
    }
    if observed {
        estimate_observed_exposure(query, &design.network, &design.assignment)?;
    }
    Ok((
        if observed {
            InterferenceProcedure::ObservationalIpw
        } else if saturation {
            InterferenceProcedure::SaturationExact
        } else {
            InterferenceProcedure::DesignBasedYoung
        },
        if observed {
            EstimatorId::InterferenceObservationalIpw
        } else if saturation {
            EstimatorId::InterferenceSaturationExact
        } else if cluster {
            EstimatorId::InterferenceClusterNeyman
        } else {
            EstimatorId::InterferenceHtHajek
        },
    ))
}

/// Whether the declared assignment confines interference to disjoint clusters.
fn partial_interference(query: &antecedent_core::InterferenceQuery) -> bool {
    matches!(
        query.assignment,
        antecedent_core::AssignmentDesign::ClusterRandomization { .. }
            | antecedent_core::AssignmentDesign::TwoStageSaturation { .. }
            | antecedent_core::AssignmentDesign::ObservedExposure { .. }
    )
}

fn interference_identification(
    query: CausalQuery,
    outcome: VariableId,
    procedure: &InterferenceProcedure,
    cluster: bool,
) -> (IdentificationResult, IdentifiedEstimand) {
    let (rule, note, assumptions) = match procedure {
        InterferenceProcedure::DesignBasedYoung => (
            "interference.design",
            if cluster {
                "Identification is complete randomization of clusters and partial interference within disjoint clusters; the (0,0) to (1,1) contrast is the total effect of assigning an entire cluster."
            } else {
                "Identification is the known Bernoulli assignment and NeighborCount exposure mapping (Horvitz–Thompson/Hájek); the DAG binds the outcome schema only."
            },
            design_identification_assumptions(cluster),
        ),
        InterferenceProcedure::SaturationExact => (
            "interference.design",
            "Identification is the declared complete allocation of clusters to low/high saturation followed by within-cluster Bernoulli treatment and the fixed partial-interference exposure mapping.",
            saturation_identification_assumptions(),
        ),
        InterferenceProcedure::ObservationalIpw => (
            "interference.observational_exchangeability",
            "Identification assumes no unmeasured confounding of network exposure and potential outcomes conditional on the variables used to supply exposure probabilities, plus partial interference, consistency, and positivity.",
            observational_identification_assumptions(),
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

fn design_identification_assumptions(cluster: bool) -> antecedent_core::AssumptionSet {
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
    if cluster {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("interference.partial_interference"),
                description: Arc::from("Potential outcomes may depend on assignments within the unit's cluster, but not on assignments in other clusters; the fixed exposure mapping summarizes within-cluster assignments."),
            },
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("interference.cluster_neyman") },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    assumptions
}

fn saturation_identification_assumptions() -> antecedent_core::AssumptionSet {
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    };
    let mut assumptions = design_identification_assumptions(true);
    assumptions.push(AssumptionRecord {
        assumption: Assumption::Custom {
            id: Arc::from("interference.two_stage_saturation"),
            description: Arc::from("Clusters are completely randomized to low/high saturation counts; units are independently Bernoulli assigned at their cluster's declared probability; the fixed network has no cross-cluster edges."),
        },
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("interference.saturation_exact") },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
    assumptions
}

fn observational_identification_assumptions() -> antecedent_core::AssumptionSet {
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
        AssumptionStatus,
    };
    let mut assumptions = AssumptionSet::default();
    for (id, description) in [
        (
            "interference.network_exchangeability",
            "No unmeasured network-exposure confounding conditional on the covariates used to supply exposure probabilities; this is declared by the caller, not checked from the network.",
        ),
        (
            "interference.partial_interference",
            "Potential outcomes may depend on assignments within the unit's cluster, but not on assignments in other clusters.",
        ),
        (
            "interference.consistency",
            "The fixed network exposure mapping agrees with the potential-outcome exposure definition and observed outcomes.",
        ),
        (
            "interference.exposure_positivity",
            "Every requested exposure has positive supplied probability for every unit.",
        ),
        (
            "interference.supplied_propensity",
            "Exposure probabilities are known or externally estimated and supplied in unit-row order; their fitting uncertainty is not included in the cluster variance.",
        ),
    ] {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("interference.observational_ipw"),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    assumptions
}

fn estimate_observed_exposure(
    query: &antecedent_core::InterferenceQuery,
    network: &antecedent_data::NetworkData,
    assignment: &[bool],
) -> Result<antecedent_estimate::ObservationalExposureEstimate, CausalError> {
    let antecedent_core::AssignmentDesign::ObservedExposure {
        clusters,
        propensity_from,
        propensity_to,
        provenance,
        ..
    } = &query.assignment
    else {
        return Err(CausalError::Compile {
            message: "observational interference requires supplied exposure probabilities".into(),
        });
    };
    let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, from, to } =
        query.functional;
    antecedent_estimate::estimate_observational_exposure(
        network,
        outcome,
        &antecedent_estimate::ObservationalExposureSpec {
            assignment,
            clusters,
            exposure: &query.exposure,
            from,
            to,
            propensity_from,
            propensity_to,
            propensity_provenance: *provenance,
        },
    )
    .map_err(CausalError::from)
}

/// License one interference exposure-contrast interval against the exact
/// graphless support table. The interval must already be published; a withheld
/// interval or too little independent-cluster support classifies as refused and
/// yields no license.
fn interference_graphless_license(
    design: &'static str,
    method: &'static str,
    support: crate::support::GraphlessAssignmentSupport,
) -> Option<crate::support::CellStatus> {
    crate::support::license_if_graphless(
        crate::support::GraphlessSupportKey {
            family: "interference",
            design,
            method,
            inference_claim: "exposure_contrast_pointwise_95_student_interval",
        },
        support,
    )
}

fn graph_signature(graph: &Dag) -> (usize, Arc<[(u32, u32)]>) {
    (graph.node_count(), graph.directed_edge_pairs_sorted().into())
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

/// Read-only target and fixed design for checked interference execution.
#[derive(Clone, Debug)]
pub struct CheckedInterferenceInfo {
    /// Exposure contrast, assignment mechanism, and outcome.
    pub query: antecedent_core::InterferenceQuery,
    /// Identifier selected during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator selected during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Number of units in the fixed network.
    pub unit_count: usize,
    /// Number of directed network edges.
    pub network_edge_count: usize,
}

impl Inspect for CheckedInterferenceOperation {
    type Info = CheckedInterferenceInfo;

    fn info(&self) -> CheckedInterferenceInfo {
        let (identifier, estimator) = self.procedure();
        let (unit_count, network_edge_count) = self.counts();
        CheckedInterferenceInfo {
            query: self.query().clone(),
            identifier,
            estimator,
            unit_count,
            network_edge_count,
        }
    }
}

#[cfg(test)]
mod checked_interference_tests {
    #![cfg_attr(
        test,
        allow(
            clippy::float_cmp,
            reason = "tests assert exact deterministic estimates and probabilities"
        )
    )]
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
        let operation = CheckedInterferenceOperation::checked(&study, data, &Arc::new(physical))
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

    fn cluster_total_query() -> InterferenceQuery {
        InterferenceQuery::new(
            AssignmentDesign::ClusterRandomization {
                clusters: Arc::from([0, 0, 1, 1, 2, 2, 3, 3]),
                treated_clusters: 2,
            },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            },
        )
    }

    fn cluster_edges() -> Vec<NetworkEdge> {
        (0..4u64)
            .flat_map(|cluster| {
                let first = cluster * 2;
                [NetworkEdge::new(first, first + 1, 1.0), NetworkEdge::new(first + 1, first, 1.0)]
            })
            .collect()
    }

    #[test]
    fn cluster_total_effect_retains_known_truth_variance_and_artifact() {
        let data =
            TabularData::from_f64_columns([("y", &[4.0, 4.0, 6.0, 6.0, 1.0, 1.0, 3.0, 3.0][..])])
                .unwrap();
        let operation = operation(
            &data,
            dag(),
            cluster_total_query(),
            cluster_edges(),
            vec![true, true, true, true, false, false, false, false],
            InferenceMode::Frequentist,
        );
        let result = operation.execute(&data, &context()).unwrap();
        assert!((result.estimate.as_effect().unwrap().ate - 3.0).abs() < 1e-12);
        assert!(
            (result.estimate.as_effect().unwrap().se_analytic() - 2.0_f64.sqrt()).abs() < 1e-12
        );
        let interference = result.interference().unwrap();
        assert_eq!(interference.minimum_exposure_probability, 0.5);
        let wire = antecedent_io::interference_estimate_to_wire(interference);
        assert_eq!(antecedent_io::interference_estimate_from_wire(&wire), *interference);
        assert!(result.estimate.as_effect().unwrap().provenance.assumptions.entries.iter().any(|entry| matches!(
            &entry.assumption,
            antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == "interference.partial_interference"
        )));
    }

    #[test]
    fn cluster_partial_interference_refuses_cross_cluster_edge_during_prepare() {
        let data =
            TabularData::from_f64_columns([("y", &[4.0, 4.0, 6.0, 6.0, 1.0, 1.0, 3.0, 3.0][..])])
                .unwrap();
        let mut edges = cluster_edges();
        edges.push(NetworkEdge::new(0, 2, 1.0));
        let study = crate::Study::tabular(data.clone())
            .graph(dag())
            .query(CausalQuery::Interference(cluster_total_query()))
            .interference(crate::InterferenceSpec {
                network: NetworkData::try_new(data.clone(), edges).unwrap(),
                assignment: Arc::from([true, true, true, true, false, false, false, false]),
            })
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let physical = study.plan(&context()).unwrap();
        assert!(CheckedInterferenceOperation::checked(&study, &data, &Arc::new(physical)).is_err());
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
        let edges = vec![NetworkEdge::new(0, 1, 1.0), NetworkEdge::new(1, 0, 1.0)];
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
        assert!((result.estimate.as_effect().unwrap().ate - expected).abs() < 1e-12);
        assert!((result.estimate.as_effect().unwrap().se_analytic() - se).abs() < 1e-12);
        assert!(result.interference().is_some());
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
            .map(|edge| {
                NetworkEdge::new(
                    edge[0].as_u64().expect("edge endpoint"),
                    edge[1].as_u64().expect("edge endpoint"),
                    1.0,
                )
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
        assert!(result.interference().is_none());
    }

    #[test]
    fn checked_refresh_preserves_network_order_and_rejects_changed_unit_schema() {
        let data = TabularData::from_f64_columns([("y", &[1.0, 4.0][..])]).unwrap();
        let operation = operation(
            &data,
            dag(),
            contrast_query(),
            vec![NetworkEdge::new(0, 1, 1.0), NetworkEdge::new(1, 0, 1.0)],
            vec![true, false],
            InferenceMode::Frequentist,
        );
        let refreshed = TabularData::from_f64_columns([("y", &[2.0, 8.0][..])]).unwrap();
        assert!(operation.execute(&refreshed, &context()).is_ok());
        let reordered = TabularData::from_f64_columns([("x", &[2.0, 8.0][..])]).unwrap();
        assert!(operation.execute(&reordered, &context()).is_err());
    }
}
