// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use antecedent_core::{
    CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext, StreamDomain,
};
use antecedent_data::TableView;
use antecedent_estimate::transport_design::{RowDependence, TrialToTargetRows};
use antecedent_estimate::{
    EffectEstimate, OverlapPolicy, TransportEffectEstimate, estimate_interference,
    estimate_interference_bayesian, trial_to_target_effect, trial_to_target_ipw_se,
};
use antecedent_identify::{
    TransportIdentification, TransportIdentifier, bind_transport_derivation, lower_transport_mean,
};

use super::{
    Arc, BayesianConfig, CausalExprArena, CausalPosterior, DerivationMeta, DerivationTrace,
    DomainRef, ExprNode, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteFinish, InferenceDiagnostics,
    InferenceMode, OutcomeExprId, PhysicalExecutionPlan, Strategy, StudyResult, TabularData,
    VariableId, effect_from_posterior, elapsed_ns, report_identify_compute,
};
use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::strategy_table::{EstimatorId, IdentifierId};

/// Everything a trial-to-target execution reads besides the data: the
/// certified sID proof, the design columns, and the inference procedure. The
/// checked route supplies these from its retained plan; the ordinary route
/// derives them from the study.
#[derive(Clone, Copy)]
pub(super) struct TransportTrialInputs<'a> {
    pub(super) trial: &'a super::super::builder::TransportTrialSpec,
    pub(super) transport: &'a TransportIdentification,
    pub(super) identification: &'a IdentificationResult,
    pub(super) estimand: &'a IdentifiedEstimand,
    pub(super) inference: &'a InferenceMode,
    pub(super) identify_cached: bool,
}

/// Refuse a missing value in the trial indicator or in the treatment of a trial row.
///
/// Both are read as booleans (`!= 0.0`), under which a missing value would silently become a
/// trial row or a treated row and move the estimate. The treatment of a target row is never
/// read, so a missing value there is allowed.
///
/// # Errors
///
/// `missing_policy_required` naming the first missing cell.
pub(super) fn refuse_missing_trial_design(
    treatment: &[f64],
    trial: &[f64],
) -> Result<(), CausalError> {
    let missing = |what: &str, row: usize| {
        crate::compile_reason!(
            "missing_policy_required",
            "the transport route reads {what} as a yes/no value and found a missing value at \
             row {row}; it would be read as a trial row or a treated row. Next attempt: drop \
             or impute the row before the study, or recode the column"
        )
    };
    if let Some(row) = trial.iter().position(|value| !value.is_finite()) {
        return Err(missing("the trial indicator", row));
    }
    if let Some(row) =
        treatment.iter().zip(trial).position(|(value, member)| *member != 0.0 && !value.is_finite())
    {
        return Err(missing("the treatment of a trial row", row));
    }
    Ok(())
}

/// Owned probability and outcome columns detached from the transport design declaration.
struct TrialSample {
    outcomes: Vec<f64>,
    treatment_bool: Vec<bool>,
    trial_bool: Vec<bool>,
    selection: Vec<f64>,
    propensity: Vec<f64>,
    /// Observation weights carried by the table, aligned to its rows.
    weights: Option<Vec<f64>>,
}

impl TrialSample {
    fn read(
        data: &TabularData,
        trial_spec: &super::super::builder::TransportTrialSpec,
        treatment: VariableId,
        outcome: VariableId,
    ) -> Result<Self, CausalError> {
        let outcomes = data.float64_values(outcome).map_err(CausalError::rejected)?;
        let treatment_col = data.float64_values(treatment).map_err(CausalError::rejected)?;
        let trial_col = data.float64_values(trial_spec.trial).map_err(CausalError::rejected)?;
        let selection =
            data.float64_values(trial_spec.selection_probability).map_err(CausalError::rejected)?;
        let propensity =
            data.float64_values(trial_spec.treatment_probability).map_err(CausalError::rejected)?;
        refuse_missing_trial_design(&treatment_col, &trial_col)?;
        let treatment_bool: Vec<bool> = treatment_col.iter().map(|v| *v != 0.0).collect();
        let trial_bool: Vec<bool> = trial_col.iter().map(|v| *v != 0.0).collect();
        let weights = data.storage().weights().map(<[f64]>::to_vec);
        Ok(Self { outcomes, treatment_bool, trial_bool, selection, propensity, weights })
    }
}

impl super::Study {
    pub(super) fn execute_transport(
        &self,
        data: &TabularData,
        query: &antecedent_core::TransportQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        query.validate()?;
        // A Cpdag or Pag names no single selection diagram: sID decides each member.
        if matches!(
            self.graph.class(),
            crate::accepted::GraphClass::Cpdag | crate::accepted::GraphClass::Pag
        ) {
            return self.execute_transport_class(data, query, physical, ctx);
        }
        let diagram = self.selection_diagram.as_ref().ok_or(CausalError::Unsupported {
            message: "TransportQuery execute requires a selection diagram",
        })?;
        let trial_spec = self.transport_trial.as_ref().ok_or(CausalError::Unsupported {
            message: "TransportQuery execute requires transport_trial columns",
        })?;
        // Refuse a query without a treatment/outcome pair before identifying.
        transport_primary_pair(query)?;
        let (transport_id, identify_cached) =
            if let Some((cached, prepared)) = self.reuse_transport_identification(physical) {
                ((*cached).clone(), prepared)
            } else {
                report_identify_compute(ctx);
                (live_transport_identification(diagram, query, ctx)?, false)
            };
        let (identification, estimand) =
            Self::transport_design_identification(query, &transport_id)?;
        self.execute_transport_identified(
            data,
            query,
            physical,
            TransportTrialInputs {
                trial: trial_spec,
                transport: &transport_id,
                identification: &identification,
                estimand: &estimand,
                inference: &self.settings.inference,
                identify_cached,
            },
            ctx,
        )
    }

    /// The sID identification of a certified transport formula for `query`. Execution and
    /// the route lowering share this one builder.
    ///
    /// # Errors
    ///
    /// The inner response names no treatment/outcome pair, or the formula is not certified
    /// or not estimable on the trial-to-target IPW path.
    pub(crate) fn transport_design_identification(
        query: &antecedent_core::TransportQuery,
        identified: &TransportIdentification,
    ) -> Result<(IdentificationResult, IdentifiedEstimand), CausalError> {
        let (treatment, outcome) = transport_primary_pair(query)?;
        refuse_unestimable_transport(identified)?;
        Ok(transport_sid_identification(
            CausalQuery::Transport(query.clone()),
            treatment,
            outcome,
            identified,
        ))
    }

    /// The identification the one-shot interference route records: the fixed-network
    /// Gaussian model for a Bayesian study, the known assignment mechanism otherwise.
    /// Execution and the route lowering share this one builder.
    pub(crate) fn interference_route_identification(
        query: &antecedent_core::InterferenceQuery,
        bayesian: bool,
    ) -> (IdentificationResult, IdentifiedEstimand) {
        let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
            query.functional;
        if bayesian {
            interference_bayesian_identification(CausalQuery::Interference(query.clone()), outcome)
        } else {
            interference_design_identification(CausalQuery::Interference(query.clone()), outcome)
        }
    }

    /// Binary trial-to-target IPW (or its Bayesian bootstrap) from an
    /// already-certified transport formula and frozen design columns.
    pub(super) fn execute_transport_identified(
        &self,
        data: &TabularData,
        query: &antecedent_core::TransportQuery,
        physical: &PhysicalExecutionPlan,
        inputs: TransportTrialInputs<'_>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let TransportTrialInputs {
            trial: trial_spec,
            transport: transport_id,
            identification,
            estimand,
            inference,
            identify_cached,
        } = inputs;
        let (treatment, outcome) = transport_primary_pair(query)?;
        let TrialSample { outcomes, treatment_bool, trial_bool, selection, propensity, weights } =
            TrialSample::read(data, trial_spec, treatment, outcome)?;
        let columns = TrialColumns {
            outcomes: &outcomes,
            treatment: &treatment_bool,
            trial: &trial_bool,
            selection: &selection,
            propensity: &propensity,
            weights: weights.as_deref(),
        };
        let dependence = self.transport_dependence(outcomes.len())?;
        self.refuse_unread_transport_validation(&columns, &dependence)?;
        let mut transported = transport_point(transport_id, &columns)?;
        let mut assumptions = identification.required_assumptions.clone();
        assumptions.push(known_probability_assumption());
        if weights.is_some() {
            assumptions.push(sampling_weights_assumption());
        }
        let (estimate, posterior, estimator_id, diagnostic) = match inference {
            InferenceMode::Bayesian(cfg) => {
                let (estimate, posterior, diagnostic) = bayesian_transport_estimate(
                    cfg,
                    (transport_id, &dependence),
                    &columns,
                    assumptions,
                    identification.status,
                    ctx,
                )?;
                transported.ipw = estimate.ate;
                (
                    estimate,
                    Some(posterior),
                    EstimatorId::TransportTrialBayesianBootstrap,
                    diagnostic,
                )
            }
            InferenceMode::Frequentist => {
                let (estimate, diagnostic) = frequentist_transport_estimate(
                    &columns,
                    (transported.ipw, &dependence),
                    assumptions,
                )?;
                (estimate, None, EstimatorId::TransportTrialIpw, diagnostic)
            }
        };
        // The calibration and overlap checks of the supplied probabilities and, under full
        // validation, a whole-procedure bootstrap of the transported contrast.
        let refutations = super::transport_refutation::refute_transport(
            &super::transport_refutation::TransportRefutationInput {
                transport: transport_id,
                outcomes: &outcomes,
                treatment: &treatment_bool,
                trial: &trial_bool,
                selection: &selection,
                propensity: &propensity,
                overlap: transported.overlap,
                published: estimate.ate,
                suite: self.settings.refute,
                dependence: dependence.as_grouping(),
                replicates: self.settings.bootstrap_replicates,
            },
            ctx,
        )?;
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification.clone(),
                estimand.clone(),
                estimate,
                Strategy::new(IdentifierId::TransportSid, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .extra_diagnostics(vec![diagnostic])
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled()),
        );
        result.payload = crate::result::ResultPayload::Transport(transported);
        result.posterior = posterior;
        Ok(result)
    }

    pub(super) fn execute_interference(
        &self,
        data: &TabularData,
        query: &antecedent_core::InterferenceQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate()?;
        // The executed outcomes are `data`, under the frozen network and assignment.
        let spec = self
            .interference
            .as_ref()
            .ok_or(CausalError::Unsupported {
                message: "InterferenceQuery execute requires StudyBuilder::interference",
            })?
            .bound_to(data)?;
        let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
            query.functional;
        let bayesian_model = self.settings.inference.is_bayesian();
        super::interference_refutation::admit_interference_procedure(
            query,
            spec.assignment.len(),
            spec.network.edges().len(),
            !bayesian_model,
            !bayesian_model,
            self.settings.refute,
            self.settings.bootstrap_replicates,
            ctx,
        )?;
        let estimated = if bayesian_model {
            None
        } else {
            let seed = ctx.rng.stream_for(StreamDomain::Transport, 0x1F7E).next_u64();
            Some(
                antecedent_core::numerical_admission::with_context(ctx, || {
                    estimate_interference(query, &spec.network, &spec.assignment, seed)
                })
                .map_err(super::checked_interference::interference_estimation_error)?,
            )
        };
        let (identification, estimand) =
            Self::interference_route_identification(query, bayesian_model);
        let (estimate, posterior, estimator_id, diagnostic) = match &self.settings.inference {
            InferenceMode::Frequentist => {
                let estimated = estimated.as_ref().expect("frequentist interference estimate");
                let se = estimated.contrast.conservative_variance.sqrt();
                (
                    EffectEstimate::new(
                        estimated.contrast.horvitz_thompson,
                        se,
                        identification.required_assumptions.clone(),
                        OverlapPolicy::ExplicitOverride,
                    ),
                    None,
                    EstimatorId::InterferenceHtHajek,
                    Diagnostic::new(
                        "estimate.interference.young_bound",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "conservative Young variance bound; not the Aronow–Samii joint-exposure variance",
                    ),
                )
            }
            InferenceMode::Bayesian(cfg) => {
                let (estimate, posterior, diagnostic) =
                    interference_bayesian_estimate(cfg, query, &spec, identification.status, ctx)?;
                (estimate, Some(posterior), EstimatorId::InterferenceBayesianGaussian, diagnostic)
            }
        };
        let refutations = super::interference_refutation::refute_interference(
            &super::interference_refutation::InterferenceRefutationInput {
                query,
                network: &spec.network,
                assignment: &spec.assignment,
                estimated: estimated.as_ref(),
                suite: self.settings.refute,
                replicates: self.settings.bootstrap_replicates,
            },
            ctx,
        )?;
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::InterferenceDesign, estimator_id),
                outcome,
                outcome,
            )
            .refutations(refutations)
            .extra_diagnostics(vec![diagnostic])
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled()),
        );
        result.payload = crate::result::ResultPayload::from_interference(estimated, None);
        result.posterior = posterior;
        Ok(result)
    }
}

/// Conjugate-Gaussian finite-network posterior for the exposure contrast under the declared isotropic prior.
fn interference_bayesian_estimate(
    cfg: &BayesianConfig,
    query: &antecedent_core::InterferenceQuery,
    spec: &crate::InterferenceSpec,
    status: IdentificationStatus,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, CausalPosterior, Diagnostic), CausalError> {
    if cfg.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian {
        return Err(CausalError::Unsupported {
            message: "Bayesian interference currently requires the analytic conjugate Gaussian backend",
        });
    }
    if cfg.has_transfer() {
        return Err(CausalError::Unsupported {
            message: "Bayesian interference uses its declared isotropic Gaussian coefficient prior; transferred and composed priors are unsupported",
        });
    }
    let model = estimate_interference_bayesian(
        query,
        &spec.network,
        &spec.assignment,
        cfg.n_draws,
        ctx.rng.stream_for(StreamDomain::Bayesian, 0x1F7E).next_u64(),
        cfg.prior_scale,
    )
    .map_err(CausalError::from)?;
    let assumptions = interference_bayesian_assumptions(cfg.prior_scale);
    let schema = antecedent_prob::PosteriorSchema {
        quantities: Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
            name: Arc::from("finite_network_exposure_contrast"),
        }]),
    };
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        model.contrast_draws.len(),
        Arc::<[f64]>::from(model.contrast_draws),
    )?;
    let posterior = CausalPosterior {
        summaries: draws.summarize(),
        draws,
        identification: status,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: InferenceDiagnostics::analytic("interference.bayesian_gaussian"),
        assumptions: assumptions.clone(),
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
        preparation: None,
    };
    let estimate = effect_from_posterior(&posterior)?;
    Ok((
        estimate,
        posterior,
        Diagnostic::new(
            "estimate.interference.bayesian_gaussian",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "finite-network posterior under an additive Gaussian potential-outcome model with shared unit disturbances; conditional on fixed network and realized assignment",
        ),
    ))
}

/// Estimation-scope assumption that the supplied selection and treatment probabilities are known.
pub(super) fn known_probability_assumption() -> antecedent_core::AssumptionRecord {
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("transport.known_selection_probabilities"),
                description: Arc::from(
                    "selection and treatment probabilities are treated as known: uncertainty conditions on the supplied probability columns and excludes the estimation error of any fitted participation or propensity model",
                ),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.transport.trial_ipw"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// How the rows of a trial-to-target study depend on each other, as the study declared it.
#[derive(Clone, Debug)]
pub(super) enum TrialDependence {
    /// Independent rows.
    Independent,
    /// One series: the Newey-West lag of the error and the block length of the bootstrap.
    Series(usize),
    /// Whole clusters, one label per table row.
    Clusters(Arc<[u32]>),
}

impl TrialDependence {
    /// The estimator's view of the dependence.
    pub(super) fn as_row(&self) -> RowDependence<'_> {
        match self {
            Self::Independent => RowDependence::Independent,
            Self::Series(lag) => RowDependence::SeriesBlocks(*lag),
            Self::Clusters(labels) => RowDependence::Clusters(labels),
        }
    }

    /// The refuter state's view of the dependence.
    pub(super) fn as_grouping(&self) -> super::transport_refutation::Grouping<'_> {
        use super::transport_refutation::Grouping;
        match self {
            Self::Independent => Grouping::Independent,
            Self::Series(block) => Grouping::Blocks(*block),
            Self::Clusters(labels) => Grouping::Clusters(labels),
        }
    }

    /// How the dependence enters the assumption and diagnostic text.
    fn describe(&self) -> String {
        match self {
            Self::Independent => "independent rows".to_owned(),
            Self::Series(lag) => format!("one dependent series (lag {lag})"),
            Self::Clusters(labels) => {
                let distinct: std::collections::BTreeSet<u32> = labels.iter().copied().collect();
                format!("{} whole clusters", distinct.len())
            }
        }
    }
}

/// The record that the rows carry sampling weights the contrast and its uncertainty read.
pub(super) fn sampling_weights_assumption() -> antecedent_core::AssumptionRecord {
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("transport.observation_weights"),
                description: Arc::from(
                    "the table's observation weights are the number of population units each \
                     row stands for: the transported contrast is the weighted ratio of the trial \
                     and target sums, the selection and treatment probabilities are the \
                     probabilities in that weighted population, and the error is the weighted \
                     linearization with the declared rows, clusters or series as the sampling \
                     units",
                ),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.transport.trial_ipw"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// The trial-to-target point estimate of `columns` under `transport`: the unweighted contrast
/// when the rows carry no weights, the weighted ratio when they do.
///
/// # Errors
///
/// An uncertified formula, or invalid rows, probabilities or weights.
pub(super) fn transport_point(
    transport: &TransportIdentification,
    columns: &TrialColumns<'_>,
) -> Result<TransportEffectEstimate, CausalError> {
    let TrialColumns { outcomes, treatment, trial, selection, propensity, weights } = *columns;
    if weights.is_none() {
        return trial_to_target_effect(
            transport, outcomes, treatment, trial, selection, propensity, None,
        )
        .map_err(CausalError::from);
    }
    let rows = columns.rows();
    let ipw = rows.ipw(transport).map_err(CausalError::from)?;
    let overlap = rows.overlap().map_err(CausalError::from)?;
    Ok(TransportEffectEstimate { ipw, aipw: None, overlap })
}

/// Delta-method SE of the ratio-of-means IPW contrast under known probabilities.
///
/// The independent-row error is the root of the summed squared influence; on one series it is
/// the Newey-West long-run variance of the influence at the block length, and on declared
/// clusters the cluster sandwich (sums of the influence inside each cluster, with the `G / (G
/// - 1)` factor); with sampling weights the influence carries each row's weight.
pub(super) fn frequentist_transport_estimate(
    columns: &TrialColumns<'_>,
    (ipw, dependence): (f64, &TrialDependence),
    assumptions: antecedent_core::AssumptionSet,
) -> Result<(EffectEstimate, Diagnostic), CausalError> {
    let TrialColumns { outcomes, treatment, trial, selection, propensity, weights } = *columns;
    let weighted = if weights.is_some() { " weighted" } else { "" };
    let (se, se_kind) = match (dependence, weights) {
        (TrialDependence::Independent, None) => (
            trial_to_target_ipw_se(outcomes, treatment, trial, selection, propensity, ipw)
                .map_err(CausalError::from)?,
            antecedent_estimate::AnalyticSeKind::default(),
        ),
        (TrialDependence::Series(lag), None) => (
            antecedent_estimate::trial_to_target_ipw_hac_se(
                outcomes,
                treatment,
                trial,
                selection,
                propensity,
                (ipw, *lag),
            )
            .map_err(CausalError::from)?,
            antecedent_estimate::AnalyticSeKind::NeweyWest { lag: *lag },
        ),
        (dependence, _) => (
            columns.rows().ipw_se(ipw, dependence.as_row()).map_err(CausalError::from)?,
            match dependence {
                TrialDependence::Independent => antecedent_estimate::AnalyticSeKind::default(),
                TrialDependence::Series(lag) => {
                    antecedent_estimate::AnalyticSeKind::NeweyWest { lag: *lag }
                }
                TrialDependence::Clusters(_) => antecedent_estimate::AnalyticSeKind::Cluster,
            },
        ),
    };
    let message = match dependence {
        TrialDependence::Independent => {
            format!(
                "binary trial-to-target{weighted} IPW; inner ResponseCurve names treatment/outcome only"
            )
        }
        TrialDependence::Series(lag) => format!(
            "binary trial-to-target{weighted} IPW on the consecutive times of one series; the SE \
             is the Newey-West long-run variance of the per-row influence at lag {lag}, \
             unlicensed and unmeasured; inner ResponseCurve names treatment/outcome only"
        ),
        TrialDependence::Clusters(_) => format!(
            "binary trial-to-target{weighted} IPW on {}; the SE is the cluster sandwich of the \
             per-row influence (whole clusters are the sampling units, G/(G-1) factor), \
             unlicensed and unmeasured; inner ResponseCurve names treatment/outcome only",
            dependence.describe()
        ),
    };
    let mut estimate = EffectEstimate::new(ipw, se, assumptions, OverlapPolicy::ExplicitOverride)
        .with_se_kind(se_kind);
    if let TrialDependence::Clusters(labels) = dependence {
        // The cluster sandwich is referred to a Student-t law with one fewer degree of freedom
        // than the clusters it sums over, the small-cluster reference of a cluster-robust error.
        let read = |row: usize| columns.weights.is_none_or(|weights| weights[row] > 0.0);
        let clusters: std::collections::BTreeSet<u32> = labels
            .iter()
            .enumerate()
            .filter(|(row, _)| read(*row))
            .map(|(_, label)| *label)
            .collect();
        estimate.uncertainty.analytic.reference_df = Some(clusters.len().saturating_sub(1) as f64);
    }
    Ok((
        estimate,
        Diagnostic::new(
            "estimate.transport.trial_ipw",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            message,
        ),
    ))
}

impl super::Study {
    /// How the trial-to-target rows depend on each other on this study.
    ///
    /// * No declaration: one series when the graph is a temporal projection, independent rows
    ///   otherwise.
    /// * `DependentTimeSeries`: one series on any graph. The Newey-West lag and block length is
    ///   the declared `score_dependence` lag, or the default rule of the series bootstrap.
    /// * `ClusteredPanel`: whole clusters, labelled by the declared `score_dependence` clusters
    ///   (one label per table row). Without labels the design is refused: a cluster draw cannot
    ///   be formed from a table that names none.
    /// * `IndependentSubjectTrajectories`: independent rows.
    ///
    /// # Errors
    ///
    /// `option_not_applicable` when the declared design needs labels or a lag that were not
    /// declared, when a declared dependence does not pair with the declared design, or when
    /// the labels do not cover the `n` table rows.
    pub(super) fn transport_dependence(&self, n: usize) -> Result<TrialDependence, CausalError> {
        use antecedent_core::DeclaredSamplingDesign as Design;
        use antecedent_estimate::ScoreDependence;
        let temporal = matches!(
            self.graph.class(),
            GraphClass::TemporalDag
                | GraphClass::TemporalAdmg
                | GraphClass::TemporalCpdag
                | GraphClass::TemporalPag
        );
        let series_default = || {
            antecedent_estimate::RowResampling::SeriesBlocks
                .block_length(n)
                .map_or(TrialDependence::Independent, TrialDependence::Series)
        };
        match (self.declared.sampling_design, self.declared.score_dependence.as_ref()) {
            (Some(Design::ClusteredPanel), Some(ScoreDependence::Clusters(labels))) => {
                if labels.len() != n {
                    return Err(crate::compile_reason!(
                        "option_not_applicable",
                        "the declared cluster labels name {} rows and the table has {n}; a \
                         cluster label is needed for every row. Next attempt: declare one \
                         label per table row",
                        labels.len()
                    ));
                }
                Ok(TrialDependence::Clusters(Arc::clone(labels)))
            }
            (Some(Design::ClusteredPanel), _) => Err(crate::compile_reason!(
                "option_not_applicable",
                "the declared `clustered_panel` design makes the rows dependent within \
                 clusters, but the table carries no cluster labels, so no whole cluster can be \
                 summed or drawn for the transport error. Next attempt: declare one cluster \
                 label per row with score_dependence(ScoreDependence::clusters(..)), or declare \
                 independent subject trajectories if the rows are independent"
            )),
            (Some(Design::DependentTimeSeries), Some(ScoreDependence::NeweyWest { lag })) => {
                Ok(TrialDependence::Series(*lag))
            }
            (Some(Design::DependentTimeSeries) | None, None) if temporal => Ok(series_default()),
            (Some(Design::DependentTimeSeries), None) => Ok(series_default()),
            (None, None) | (Some(Design::IndependentSubjectTrajectories), None) => {
                Ok(TrialDependence::Independent)
            }
            (_, Some(_)) => Err(crate::compile_reason!(
                "option_not_applicable",
                "the declared score dependence pairs cluster labels with a clustered-panel \
                 design and a Newey-West lag with a dependent-time-series design; the declared \
                 sampling design does not match it, so the transport error would read the wrong \
                 dependence. Next attempt: declare the matching sampling_design"
            )),
        }
    }

    /// Refuse the declarations this route would read nothing from, and the transport-native
    /// validation suite when the rows carry sampling weights.
    ///
    /// * An estimator other than the route's own (`transport.trial_ipw`, or its Bayesian
    ///   bootstrap): the route runs inverse-probability weighting, so a named AIPW or matching
    ///   estimator would be answered with a different number than the one asked for.
    /// * An overlap policy: the supplied probabilities are the design and are never clipped or
    ///   trimmed here, so a declared clip or trim would change nothing.
    /// * A population registry: the target is the target rows of the table, not a named or
    ///   weighted population.
    /// * The validation suite beside sampling weights: its calibration tests count the
    ///   unweighted rows and its bootstrap resamples unweighted rows, so its verdicts would
    ///   describe a population the published number is not about.
    ///
    /// # Errors
    ///
    /// `option_not_applicable` naming the declaration and the next attempt.
    pub(super) fn refuse_unread_transport_validation(
        &self,
        columns: &TrialColumns<'_>,
        _dependence: &TrialDependence,
    ) -> Result<(), CausalError> {
        let own = if self.settings.inference.is_bayesian() {
            EstimatorId::TransportTrialBayesianBootstrap
        } else {
            EstimatorId::TransportTrialIpw
        };
        if let Some(declared) = self.declared.estimator.filter(|declared| *declared != own) {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "the estimator `{}` was declared, but a trial-to-target TransportQuery runs \
                 inverse-probability weighting (`{}`) and would answer with a different \
                 estimate than the one asked for. Next attempt: drop the estimator declaration, \
                 or ask for the estimand on a route that runs it",
                declared.as_str(),
                own.as_str()
            ));
        }
        if self.declared.overlap_policy.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "an overlap policy (clip or trim) was declared, but the trial-to-target \
                 contrast takes the supplied selection and treatment probabilities as the \
                 design and never clips or trims them, so the declaration would change nothing. \
                 Next attempt: clip or trim the probability columns before the study, or drop \
                 the overlap policy"
            ));
        }
        if self.declared.population_registry.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a population registry was declared, but the trial-to-target contrast is the \
                 effect in the target rows of the table and reads no named or custom \
                 population, so the declaration would change nothing. Next attempt: put the \
                 target population in the table (its target rows and, if they are a sample, \
                 their observation weights), or drop the registry"
            ));
        }
        if columns.weights.is_some() && self.settings.refute != super::RefuteSuite::None {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "the transport-native validation suite counts and resamples the unweighted \
                 rows, and the table carries observation weights it would not read, so its \
                 verdicts would describe a different population than the weighted estimate. \
                 Next attempt: run the weighted contrast with RefuteSuite::None, or drop the \
                 weights to validate the unweighted rows"
            ));
        }
        Ok(())
    }
}

/// The binary trial-to-target columns the IPW estimator and its bootstrap read.
#[derive(Clone, Copy)]
pub(super) struct TrialColumns<'a> {
    pub(super) outcomes: &'a [f64],
    pub(super) treatment: &'a [bool],
    pub(super) trial: &'a [bool],
    pub(super) selection: &'a [f64],
    pub(super) propensity: &'a [f64],
    /// Observation weights of the rows; `None` is unweighted.
    pub(super) weights: Option<&'a [f64]>,
}

impl<'a> TrialColumns<'a> {
    /// The columns as the weighted, design-aware estimator reads them.
    pub(super) fn rows(&self) -> TrialToTargetRows<'a> {
        TrialToTargetRows {
            outcome: self.outcomes,
            treatment: self.treatment,
            trial: self.trial,
            selection_probability: self.selection,
            treatment_probability: self.propensity,
            weights: self.weights,
        }
    }
}

/// Dirichlet Bayesian bootstrap of the trial-to-target IPW contrast over the observed trial rows
/// (or whole clusters, or blocks of one series).
pub(super) fn bayesian_transport_estimate(
    cfg: &BayesianConfig,
    (transport_id, dependence): (&TransportIdentification, &TrialDependence),
    columns: &TrialColumns<'_>,
    mut assumptions: antecedent_core::AssumptionSet,
    status: IdentificationStatus,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, CausalPosterior, Diagnostic), CausalError> {
    let TrialColumns { outcomes, treatment, trial, selection, propensity, weights } = *columns;
    refuse_transport_bayesian_priors(cfg)?;
    let spec = antecedent_estimate::DrawSpec::new(
        cfg.n_draws,
        ctx.rng.stream_for(StreamDomain::Bayesian, 0x7A17_0001).next_u64(),
    );
    let draws = if weights.is_none() && !matches!(dependence, TrialDependence::Clusters(_)) {
        let block = match dependence {
            TrialDependence::Series(block) => Some(*block),
            _ => None,
        };
        antecedent_estimate::trial_to_target_bayesian_bootstrap_blocks(
            transport_id,
            outcomes,
            treatment,
            trial,
            selection,
            propensity,
            (spec, block),
        )
    } else {
        columns.rows().bayesian_bootstrap(transport_id, dependence.as_row(), spec)
    }
    .map_err(CausalError::from)?;
    let unit = match dependence {
        TrialDependence::Independent => "observed trial rows",
        TrialDependence::Series(_) => "contiguous blocks of the observed trial rows",
        TrialDependence::Clusters(_) => "whole clusters of the observed trial rows",
    };
    let weighted = if weights.is_some() {
        ", each shared draw multiplied by the row's observation weight"
    } else {
        ""
    };
    assumptions.push(antecedent_core::AssumptionRecord {
                    assumption: antecedent_core::Assumption::ParametricRestriction(
                        antecedent_core::ParametricAssumption {
                            id: Arc::from("transport.trial_empirical_support_prior"),
                            description: if matches!(dependence, TrialDependence::Independent) && weights.is_none() {
                                Arc::from("Dirichlet(1,...,1) Bayesian bootstrap over the observed trial rows, conditional on the selected trial sample size, target sample size, and supplied selection/treatment probabilities; the target row law and fitted probability-model uncertainty are held fixed")
                            } else {
                                Arc::from(format!("Dirichlet(1,...,1) Bayesian bootstrap over the {unit}{weighted}, conditional on the trial and target weight totals and the supplied selection/treatment probabilities; the target row law and fitted probability-model uncertainty are held fixed"))
                            },
                        },
                    ),
                    source: antecedent_core::AssumptionSource::AlgorithmDefault {
                        algorithm: Arc::from("transport.trial_bayesian_bootstrap"),
                    },
                    scope: antecedent_core::AssumptionScope::Estimation,
                    status: antecedent_core::AssumptionStatus::Declared,
                });
    let schema = antecedent_prob::PosteriorSchema {
        quantities: Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
            name: Arc::from("transported_ate"),
        }]),
    };
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        draws.len(),
        Arc::<[f64]>::from(draws),
    )?;
    let posterior = CausalPosterior {
        summaries: draws.summarize(),
        draws,
        identification: status,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: InferenceDiagnostics::analytic("transport.trial_bayesian_bootstrap"),
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
        estimate,
        posterior,
        Diagnostic::new(
            "estimate.transport.trial_bayesian_bootstrap",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "Dirichlet trial-row-law posterior of binary trial-to-target IPW; supplied probabilities and target sample size held fixed",
        ),
    ))
}

fn interference_bayesian_identification(
    query: CausalQuery,
    outcome: VariableId,
) -> (IdentificationResult, IdentifiedEstimand) {
    let (arena, estimand) = inspectable_do_expectation(
        outcome,
        outcome,
        "interference.bayesian_gaussian",
        "Finite-network mean potential-outcome contrast identified under the declared additive Gaussian exposure-response model and shared unit disturbance.",
    );
    let mut assumptions = antecedent_core::AssumptionSet::default();
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom { id: Arc::from("interference.fixed_network_model_identification"), description: Arc::from("The finite-network exposure contrast follows from the declared additive potential-outcome model on the fixed network; it is not design based and does not imply superpopulation identification.") },
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("interference.bayesian_gaussian") },
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let mut derivation = DerivationTrace::default();
    derivation.push(
        "interference.bayesian_gaussian",
        "fixed-network model-based identification; target is the supplied finite set of units",
    );
    let mut identification = IdentificationResult::identified(
        query,
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    identification.status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
    (identification, estimand)
}

fn interference_bayesian_assumptions(prior_sd: f64) -> antecedent_core::AssumptionSet {
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
        AssumptionStatus, ParametricAssumption,
    };
    let mut assumptions = AssumptionSet::default();
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("interference.fixed_network_gaussian_potential_outcomes"),
            description: Arc::from("Finite-network target over the observed units under Y_i(z,g)=alpha+beta*z+gamma*g+epsilon_i, with the same unit disturbance shared across that unit's exposure-specific potential outcomes; Gaussian identity likelihood with residual variance fixed to one."),
        }),
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("interference.bayesian_gaussian") }, scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared,
    });
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("interference.gaussian_coefficient_prior"),
            description: Arc::from(format!("Independent Normal(0, {prior_sd}^2) prior on intercept, own-treatment, and treated-neighbor-count coefficients.")),
        }),
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("interference.bayesian_gaussian") }, scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared,
    });
    assumptions
}

/// Treatment and outcome named by the inner transport response.
pub(super) fn transport_primary_pair(
    query: &antecedent_core::TransportQuery,
) -> Result<(VariableId, VariableId), CausalError> {
    crate::analysis::proof_support::require_primary_pair(&query.response.functional)
}

/// The Bayesian trial route owns its row-law prior; coefficient and transferred
/// priors have no meaning for it.
pub(super) fn refuse_transport_bayesian_priors(cfg: &BayesianConfig) -> Result<(), CausalError> {
    if cfg.has_transfer() {
        return Err(CausalError::Unsupported {
            message: "Bayesian trial transport uses an empirical-support row-law prior and does not accept coefficient or transferred priors",
        });
    }
    Ok(())
}

pub(crate) fn live_transport_identification(
    diagram: &antecedent_graph::SelectionDiagram,
    query: &antecedent_core::TransportQuery,
    ctx: &ExecutionContext,
) -> Result<TransportIdentification, CausalError> {
    let identified = TransportIdentifier::new()
        .identify_with_context(diagram, query, ctx)
        .map_err(CausalError::from)?;
    refuse_unestimable_transport(&identified)?;
    Ok(identified)
}

/// Inspectable do-expectation leaf. Same shape as `parametric_scm_identification`,
/// but the rule and assumptions name the design-based identifier, not GCM.
fn inspectable_do_expectation(
    treatment: VariableId,
    outcome: VariableId,
    rule: &str,
    note: &str,
) -> (CausalExprArena, IdentifiedEstimand) {
    let mut arena = CausalExprArena::new();
    let y = arena.intern_var_set([outcome]);
    let do_t = arena.intern_intervention_set([treatment]);
    let empty = arena.empty_var_set();
    let distribution = arena.intern_distribution(y, empty, do_t, DomainRef::Interventional);
    let functional = arena
        .intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution });
    arena.set_derivation(functional, DerivationMeta::rule(rule, Some(Arc::from(note))));
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    (arena, estimand)
}

pub(super) fn transport_sid_identification(
    query: CausalQuery,
    treatment: VariableId,
    outcome: VariableId,
    identified: &TransportIdentification,
) -> (IdentificationResult, IdentifiedEstimand) {
    let TransportIdentification::Transportable { certificate, formula } = identified else {
        unreachable!("execute already refused an uncertified transport formula");
    };
    let premises = certificate.premises.iter().map(AsRef::as_ref).collect::<Vec<_>>().join("; ");
    let mut arena = CausalExprArena::new();
    let functional = lower_transport_mean(&mut arena, formula, outcome);
    bind_transport_derivation(
        &mut arena,
        functional,
        certificate,
        format!("sID: treatment={treatment:?} outcome={outcome:?}; {premises}"),
    );
    let estimand = IdentifiedEstimand::new(
        certificate.rule.as_ref(),
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut assumptions = antecedent_core::AssumptionSet::default();
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom {
            id: Arc::from(certificate.rule.as_ref()),
            description: Arc::from(if premises.is_empty() {
                "Structural transportability under the implemented sID subset (Direct or S-admissible standardize)."
                    .to_string()
            } else {
                premises.clone()
            }),
        },
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("transport.sid"),
        },
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let mut derivation = DerivationTrace::default();
    derivation.push(certificate.rule.as_ref(), premises);
    let identification = IdentificationResult::identified(
        query,
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}

fn interference_design_identification(
    query: CausalQuery,
    outcome: VariableId,
) -> (IdentificationResult, IdentifiedEstimand) {
    let (arena, estimand) = inspectable_do_expectation(
        outcome,
        outcome,
        "interference.design",
        "The Dag binds schema and outcome; it does not identify the exposure contrast. \
         Identification is the known assignment mechanism and exposure mapping \
         (Horvitz–Thompson / Hájek).",
    );
    let mut assumptions = antecedent_core::AssumptionSet::default();
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom {
            id: Arc::from("interference.design"),
            description: Arc::from(
                "The Dag binds schema and outcome; it does not identify the exposure contrast. \
                 Identification is the known assignment mechanism and exposure mapping.",
            ),
        },
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("interference.design"),
        },
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let mut derivation = DerivationTrace::default();
    derivation.push("interference.design", "known assignment mechanism; graph is schema-only");
    let identification = IdentificationResult::identified(
        query,
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}

pub(super) fn refuse_unestimable_transport(
    identified: &TransportIdentification,
) -> Result<(), CausalError> {
    match identified {
        TransportIdentification::NotCertified(certificate) => Err(CausalError::Compile {
            message: format!(
                "transport not certified: {} ({})",
                certificate.reason, certificate.message
            ),
        }),
        TransportIdentification::MissingEvidence(certificate) => Err(CausalError::Compile {
            message: format!(
                "transport missing evidence: {} ({})",
                certificate.reason, certificate.message
            ),
        }),
        TransportIdentification::Transportable {
            formula: antecedent_identify::TransportFormula::RecursiveFactorization { .. },
            ..
        } => Err(crate::support_reason!(
            "construction_not_licensed",
            "RecursiveFactorization is identified but not estimable on the licensed \
             trial-to-target IPW path"
        )),
        TransportIdentification::Transportable { .. } => Ok(()),
    }
}
