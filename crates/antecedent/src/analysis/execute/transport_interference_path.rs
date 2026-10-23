// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use antecedent_core::{
    CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext, StreamDomain,
};
use antecedent_data::TableView;
use antecedent_estimate::{
    EffectEstimate, OverlapPolicy, estimate_interference, estimate_interference_bayesian,
    trial_to_target_effect, trial_to_target_ipw_se,
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

impl super::Study {
    pub(super) fn execute_transport(
        &self,
        data: &TabularData,
        query: &antecedent_core::TransportQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        query.validate()?;
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
                (live_transport_identification(diagram, query)?, false)
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
                inference: &self.inference,
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
        let outcomes = data.float64_values(outcome).map_err(CausalError::rejected)?;
        let treatment_col = data.float64_values(treatment).map_err(CausalError::rejected)?;
        let trial_col = data.float64_values(trial_spec.trial).map_err(CausalError::rejected)?;
        let selection =
            data.float64_values(trial_spec.selection_probability).map_err(CausalError::rejected)?;
        let propensity =
            data.float64_values(trial_spec.treatment_probability).map_err(CausalError::rejected)?;
        let treatment_bool: Vec<bool> = treatment_col.iter().map(|v| *v != 0.0).collect();
        let trial_bool: Vec<bool> = trial_col.iter().map(|v| *v != 0.0).collect();
        let mut transported = trial_to_target_effect(
            transport_id,
            &outcomes,
            &treatment_bool,
            &trial_bool,
            &selection,
            &propensity,
            None,
        )
        .map_err(CausalError::from)?;
        let mut assumptions = identification.required_assumptions.clone();
        assumptions.push(known_probability_assumption());
        let columns = TrialColumns {
            outcomes: &outcomes,
            treatment: &treatment_bool,
            trial: &trial_bool,
            selection: &selection,
            propensity: &propensity,
        };
        let series_block = self.transport_series_block(outcomes.len())?;
        let (estimate, posterior, estimator_id, diagnostic) = match inference {
            InferenceMode::Bayesian(cfg) => {
                let (estimate, posterior, diagnostic) = bayesian_transport_estimate(
                    cfg,
                    (transport_id, series_block),
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
                    (transported.ipw, series_block),
                    assumptions,
                )?;
                (estimate, None, EstimatorId::TransportTrialIpw, diagnostic)
            }
        };
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
        let bayesian_model = self.inference.is_bayesian();
        let estimated = if bayesian_model {
            None
        } else {
            let seed = ctx.rng.stream_for(StreamDomain::Transport, 0x1F7E).next_u64();
            Some(
                estimate_interference(query, &spec.network, &spec.assignment, seed)
                    .map_err(CausalError::from)?,
            )
        };
        let (identification, estimand) =
            Self::interference_route_identification(query, bayesian_model);
        let (estimate, posterior, estimator_id, diagnostic) = match &self.inference {
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
fn known_probability_assumption() -> antecedent_core::AssumptionRecord {
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

/// Delta-method SE of the ratio-of-means IPW contrast under known probabilities.
///
/// When the rows are the consecutive times of one series (`series_block`), the SE is the
/// Newey–West long-run variance of the per-row influence at the block length.
fn frequentist_transport_estimate(
    columns: &TrialColumns<'_>,
    (ipw, series_block): (f64, Option<usize>),
    assumptions: antecedent_core::AssumptionSet,
) -> Result<(EffectEstimate, Diagnostic), CausalError> {
    let TrialColumns { outcomes, treatment, trial, selection, propensity } = *columns;
    let Some(lag) = series_block else {
        let se = trial_to_target_ipw_se(outcomes, treatment, trial, selection, propensity, ipw)
            .map_err(CausalError::from)?;
        return Ok((
            EffectEstimate::new(ipw, se, assumptions, OverlapPolicy::ExplicitOverride),
            Diagnostic::new(
                "estimate.transport.trial_ipw",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "binary trial-to-target IPW; inner ResponseCurve names treatment/outcome only",
            ),
        ));
    };
    let se = antecedent_estimate::trial_to_target_ipw_hac_se(
        outcomes,
        treatment,
        trial,
        selection,
        propensity,
        (ipw, lag),
    )
    .map_err(CausalError::from)?;
    Ok((
        EffectEstimate::new(ipw, se, assumptions, OverlapPolicy::ExplicitOverride)
            .with_se_kind(antecedent_estimate::AnalyticSeKind::NeweyWest { lag }),
        Diagnostic::new(
            "estimate.transport.trial_ipw",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "binary trial-to-target IPW on the consecutive times of one series; the SE is \
                 the Newey-West long-run variance of the per-row influence at lag {lag} (the \
                 contemporaneous projection's block length), unlicensed and unmeasured; inner \
                 ResponseCurve names treatment/outcome only"
            ),
        ),
    ))
}

impl super::Study {
    /// The block length of a trial-transport uncertainty over `n` rows: `Some` when the
    /// rows are one dependent series (a `TemporalAdmg` projection), `None` for independent
    /// rows. A declared independent-subject or clustered design on the projection refuses
    /// ([`Self::functional_row_resampling`]).
    fn transport_series_block(&self, n: usize) -> Result<Option<usize>, CausalError> {
        let (resampling, _) = self.functional_row_resampling()?;
        Ok(resampling.block_length(n))
    }
}

/// The binary trial-to-target columns the IPW estimator and its bootstrap read.
struct TrialColumns<'a> {
    outcomes: &'a [f64],
    treatment: &'a [bool],
    trial: &'a [bool],
    selection: &'a [f64],
    propensity: &'a [f64],
}

/// Dirichlet Bayesian bootstrap of the trial-to-target IPW contrast over the observed trial rows.
fn bayesian_transport_estimate(
    cfg: &BayesianConfig,
    (transport_id, series_block): (&TransportIdentification, Option<usize>),
    columns: &TrialColumns<'_>,
    mut assumptions: antecedent_core::AssumptionSet,
    status: IdentificationStatus,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, CausalPosterior, Diagnostic), CausalError> {
    let TrialColumns { outcomes, treatment, trial, selection, propensity } = *columns;
    refuse_transport_bayesian_priors(cfg)?;
    let draws = antecedent_estimate::trial_to_target_bayesian_bootstrap_blocks(
        transport_id,
        outcomes,
        treatment,
        trial,
        selection,
        propensity,
        (
            antecedent_estimate::DrawSpec::new(
                cfg.n_draws,
                ctx.rng.stream_for(StreamDomain::Bayesian, 0x7A17_0001).next_u64(),
            ),
            series_block,
        ),
    )
    .map_err(CausalError::from)?;
    assumptions.push(antecedent_core::AssumptionRecord {
                    assumption: antecedent_core::Assumption::ParametricRestriction(
                        antecedent_core::ParametricAssumption {
                            id: Arc::from("transport.trial_empirical_support_prior"),
                            description: Arc::from("Dirichlet(1,...,1) Bayesian bootstrap over the observed trial rows, conditional on the selected trial sample size, target sample size, and supplied selection/treatment probabilities; the target row law and fitted probability-model uncertainty are held fixed"),
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
) -> Result<TransportIdentification, CausalError> {
    let identified =
        TransportIdentifier::new().identify(diagram, query).map_err(CausalError::rejected)?;
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
