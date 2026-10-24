//! Retained observational two-decision contrast of two static regimes.
//!
//! One row per subject: the first binary action, covariates measured after it, the second binary
//! action and the terminal outcome are table columns named by the query. The cross-fit augmented
//! sequential score of [`antecedent_estimate::sequential_crossfit`] gives one score per subject
//! and regime. Frequentist inference reports the normal interval of the mean contrast score;
//! Bayesian inference reports the modular Bayesian-bootstrap posterior of the same scores, with
//! the fitted propensities and conditional means held at their cross-fit values. Sequential
//! conditional exchangeability given the measured history, positivity, consistency and no
//! interference are declared, not derived. The route is unlicensed and its calibration is
//! unmeasured.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::bayesian_quasi_common::{
    assemble_posterior, effect_quantity, model_assumption, require_built_in_model, scalar_quantity,
};
use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationMeta, DerivationTrace, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate, EstimatorId, ExecutionContext,
    ExprNode, GraphClass, IdentificationPerformanceRecord, IdentificationResult,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId,
    InferenceDiagnostics, InferenceMode, OutcomeExprId, OverlapPolicy, PhysicalExecutionPlan,
    RandomizedTrialOperation, RandomizedTrialQuery, RefuteSuite, Strategy, Study, StudyResult,
    TableView, TabularData, finish_identified_execute_with_context,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    NumericalPolicy, SemanticDigest, SourceDependencyId, reason_code,
};
use antecedent_estimate::EstimationError;
use antecedent_estimate::nuisance_cache::{
    CertifiedSource, FoldAssignment, FoldRow, NuisanceOptions, NumericalDependencies,
    NumericalDependency, ProviderDependency, TargetDependency,
};
use antecedent_estimate::score_posterior::ScorePosteriorOptions;
use antecedent_estimate::sequential_aipw_inference::{
    bayesian_sequential_contrast, sequential_contrast_wald_interval,
};
use antecedent_estimate::sequential_crossfit::{
    ActionLevel, MIN_PROBABILITY, SequentialContrast, SequentialCrossfitError, SequentialIdentity,
    SequentialNuisanceStore, SequentialPanel, StaticRegime, estimate_sequential_contrast,
};

/// Posterior backend of the sequential contrast posterior.
const BACKEND: &str = "bayesian_bootstrap.sequential_crossfit_contrast";

/// Uncertainty label of the normal interval.
const INTERVAL_UNCERTAINTY: &str = "normal_interval_calibration_unmeasured";

/// Uncertainty label of the modular posterior.
const POSTERIOR_UNCERTAINTY: &str = "modular_bayesian_bootstrap_posterior_no_interval";

/// Central coverage of the interval and of the posterior's equal-tailed interval.
const COVERAGE: f64 = 0.95;

/// Frozen sequential contrast; the extra is the Bayesian configuration of a posterior study.
pub(crate) type CheckedSequentialContrastOperation = RandomizedTrialOperation<
    antecedent_core::SequentialContrastQuery,
    Option<super::BayesianConfig>,
>;

impl RandomizedTrialQuery for antecedent_core::SequentialContrastQuery {
    const OPERATION: &'static str = "CheckedSequentialContrastOperation";
    const REFRESH_MISMATCH: &'static str =
        "sequential contrast refresh requires the prepared schema and row order";
}

fn reasoned(code: &'static str, message: String) -> CausalError {
    CausalError::Estimate(EstimationError::refused(code, message))
}

fn crossfit(error: SequentialCrossfitError) -> CausalError {
    match error {
        SequentialCrossfitError::Positivity { .. } => {
            reasoned(reason_code!("positivity_violated"), error.to_string())
        }
        other => CausalError::Estimate(EstimationError::data_msg(other.to_string())),
    }
}

fn level(code: u8) -> ActionLevel {
    if code == 0 { ActionLevel::Zero } else { ActionLevel::One }
}

fn regime(actions: [u8; 2]) -> StaticRegime {
    StaticRegime::new(level(actions[0]), level(actions[1]))
}

impl CheckedSequentialContrastOperation {
    /// The design identification this operation records; the route lowering retains it.
    pub(crate) fn design_identification(
        query: &antecedent_core::SequentialContrastQuery,
    ) -> (IdentificationResult, IdentifiedEstimand) {
        contrast_identification(query)
    }

    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::SequentialContrast(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "sequential contrast operation requires SequentialContrastQuery".into(),
            });
        };
        query.validate()?;
        if query.fold_of_row.len() != data.row_count()
            || study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "the sequential contrast reads one row per subject, with no graph, refuter or bootstrap",
            });
        }
        let bayesian = match &study.inference {
            InferenceMode::Frequentist => None,
            InferenceMode::Bayesian(config) => {
                require_built_in_model(config, "the Bayesian sequential contrast")?;
                Some(config.clone())
            }
        };
        for id in contrast_columns(query) {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let estimator_id = if bayesian.is_some() {
            EstimatorId::BayesianSequentialCrossfitContrast
        } else {
            EstimatorId::SequentialCrossfitContrastWald
        };
        if physical.logical.query != study.query
            || !physical.binding().names(IdentifierId::RandomizedDesign, estimator_id)
        {
            return Err(CausalError::Compile {
                message: "sequential contrast query differs from compiled plan".into(),
            });
        }
        let (identification, estimand) = contrast_identification(query);
        Ok(Self::freeze(study, query, data, physical, identification, estimand, bayesian))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        let columns = ContrastColumns::read(&self.query, data)?;
        let contrast = self.cross_fit(&columns)?;
        let (estimator_id, uncertainty, standard_error, interval, posterior) =
            match self.extra.as_ref() {
                None => {
                    let wald = sequential_contrast_wald_interval(&contrast, COVERAGE)
                        .map_err(CausalError::from)?;
                    (
                        EstimatorId::SequentialCrossfitContrastWald,
                        INTERVAL_UNCERTAINTY,
                        Some(wald.standard_error),
                        Some([wald.interval.0, wald.interval.1]),
                        None,
                    )
                }
                Some(config) => (
                    EstimatorId::BayesianSequentialCrossfitContrast,
                    POSTERIOR_UNCERTAINTY,
                    None,
                    None,
                    Some(self.posterior(&contrast, config, ctx)?),
                ),
            };
        let n_draws = posterior
            .as_ref()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.identification.clone(),
                self.estimand.clone(),
                EffectEstimate::new(
                    0.0,
                    f64::NAN,
                    self.identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                Strategy::new(IdentifierId::RandomizedDesign, estimator_id),
                self.query.outcome,
                self.query.outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(vec![Diagnostic::new(
                "estimate.sequential_contrast.unlicensed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "unlicensed and uncalibrated route: sequential exchangeability given the \
                 measured history is declared and not checked, the nuisance fits are held fixed \
                 in each score, and no repeated-sampling calibration of the reported uncertainty \
                 has been measured",
            )])
            .extras(IdentifiedExecuteExtras {
                posterior,
                n_draws,
                ..IdentifiedExecuteExtras::default()
            }),
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.treatment = None;
        result.support_status = None;
        result.payload =
            crate::result::ResultPayload::SequentialContrast(crate::SequentialContrastEstimate {
                active_regime: self.query.active,
                control_regime: self.query.control,
                contrast: contrast.contrast,
                active_mean: contrast.active_mean,
                control_mean: contrast.control_mean,
                evaluated_rows: contrast.evaluated_rows,
                minimum_action_probability: contrast.minimum_action_probability,
                min_probability: self.query.min_probability,
                standard_error,
                interval,
                coverage: COVERAGE,
                active_scores: Arc::from(contrast.active_scores.as_slice()),
                control_scores: Arc::from(contrast.control_scores.as_slice()),
                contrast_scores: Arc::from(contrast.contrast_scores.as_slice()),
                uncertainty: Arc::from(uncertainty),
                fits_computed: contrast.fits_computed,
                fits_reused: contrast.fits_reused,
            });
        Ok(result)
    }

    /// The cross-fit scores of both regimes.
    fn cross_fit(&self, columns: &ContrastColumns) -> Result<SequentialContrast, CausalError> {
        let baseline: Vec<&[f64]> = columns.baseline.iter().map(Vec::as_slice).collect();
        let intermediate: Vec<&[f64]> = columns.intermediate.iter().map(Vec::as_slice).collect();
        let panel = SequentialPanel {
            baseline: &baseline,
            first_treatment: &columns.first,
            intermediate: &intermediate,
            second_treatment: &columns.second,
            outcome: &columns.outcome,
        };
        let identity = self.crossfit_identity(panel, columns)?;
        let mut store = SequentialNuisanceStore::new();
        estimate_sequential_contrast(
            panel,
            &identity,
            regime(self.query.active),
            regime(self.query.control),
            &mut store,
        )
        .map_err(crossfit)
    }

    /// The reuse key of the nuisance fits: derived from the panel's own values, the regimes,
    /// the folds and the floor.
    fn crossfit_identity(
        &self,
        panel: SequentialPanel<'_>,
        columns: &ContrastColumns,
    ) -> Result<SequentialIdentity, CausalError> {
        let query = &self.query;
        let invalid = |message: String| reasoned(reason_code!("invalid_argument"), message);
        let rows: Vec<FoldRow> = query
            .fold_of_row
            .iter()
            .enumerate()
            .map(|(row, fold)| FoldRow::new(row as u64, *fold))
            .collect();
        let mut hashed: Vec<&[f64]> = vec![&columns.first, &columns.second, &columns.outcome];
        hashed.extend(columns.baseline.iter().map(Vec::as_slice));
        hashed.extend(columns.intermediate.iter().map(Vec::as_slice));
        let regimes = [
            f64::from(query.active[0]),
            f64::from(query.active[1]),
            f64::from(query.control[0]),
            f64::from(query.control[1]),
        ];
        let floor = NumericalDependency::try_new(MIN_PROBABILITY, query.min_probability)
            .map_err(|e| invalid(e.to_string()))?;
        SequentialIdentity::from_panel(
            panel,
            CertifiedSource::new(SourceDependencyId::from_raw(1), digest("source", &hashed)),
            TargetDependency::of_functional("static_regimes", &regimes),
            FoldAssignment::try_new(query.fold_count, &rows, &[])
                .map_err(|e| invalid(e.to_string()))?,
            ProviderDependency::named("sequential_crossfit", env!("CARGO_PKG_VERSION")),
            NumericalDependencies::try_new(NumericalPolicy::strict(), &[floor])
                .map_err(|e| invalid(e.to_string()))?,
            &NuisanceOptions::new("sequential_contrast"),
        )
        .map_err(crossfit)
    }

    /// Modular Bayesian-bootstrap posterior of the cross-fit contrast scores.
    fn posterior(
        &self,
        contrast: &SequentialContrast,
        config: &super::BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_estimate::CausalPosterior, CausalError> {
        let options = ScorePosteriorOptions { draws: config.n_draws, coverage: COVERAGE };
        let fit =
            bayesian_sequential_contrast(contrast, options, ctx).map_err(CausalError::from)?;
        let columns = vec![
            (effect_quantity("sequential_contrast"), fit.contrast.draws.clone()),
            (scalar_quantity("active_regime_value"), fit.active.draws.clone()),
            (scalar_quantity("control_regime_value"), fit.control.draws.clone()),
        ];
        let mut assumptions = self.identification.required_assumptions.clone();
        assumptions.push(model_assumption(
            BACKEND,
            "modular_bayesian_bootstrap_sequential_scores",
            "the subject law has a vanishing-concentration Dirichlet-process prior and each draw \
             reweights the cross-fit orthogonal scores; the fitted propensities and conditional \
             means are held at their cross-fit values, so nuisance-fit uncertainty is omitted",
        ));
        let mut inference = InferenceDiagnostics::analytic(BACKEND);
        let note = format!("draws={} interval=equal_tailed_95", config.n_draws);
        inference.notes = [note].iter().map(|text| Arc::<str>::from(text.as_str())).collect();
        assemble_posterior(&columns, &self.identification, assumptions, inference)
    }
}

impl Study {
    pub(super) fn execute_sequential_contrast(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedSequentialContrastOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

fn contrast_columns(
    query: &antecedent_core::SequentialContrastQuery,
) -> impl Iterator<Item = antecedent_core::VariableId> + '_ {
    [query.outcome, query.first_treatment, query.second_treatment]
        .into_iter()
        .chain(query.baseline.iter().copied())
        .chain(query.intermediate.iter().copied())
}

/// The complete numeric columns the contrast reads.
struct ContrastColumns {
    first: Vec<f64>,
    second: Vec<f64>,
    outcome: Vec<f64>,
    baseline: Vec<Vec<f64>>,
    intermediate: Vec<Vec<f64>>,
}

impl ContrastColumns {
    fn read(
        query: &antecedent_core::SequentialContrastQuery,
        data: &TabularData,
    ) -> Result<Self, CausalError> {
        let read = |id| -> Result<Vec<f64>, CausalError> {
            match data.column(id).map_err(CausalError::from)? {
                antecedent_data::ColumnView::Float64(column) if column.validity.is_all_valid() => {
                    Ok(column.values.as_slice().to_vec())
                }
                _ => Err(CausalError::Unsupported {
                    message: "sequential contrast columns must be complete continuous data, with both actions coded 0 or 1",
                }),
            }
        };
        Ok(Self {
            first: read(query.first_treatment)?,
            second: read(query.second_treatment)?,
            outcome: read(query.outcome)?,
            baseline: query.baseline.iter().map(|id| read(*id)).collect::<Result<_, _>>()?,
            intermediate: query
                .intermediate
                .iter()
                .map(|id| read(*id))
                .collect::<Result<_, _>>()?,
        })
    }
}

fn digest(label: &str, columns: &[&[f64]]) -> SemanticDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(label.as_bytes());
    for column in columns {
        hasher.update(&(column.len() as u64).to_le_bytes());
        for value in *column {
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }
    SemanticDigest::from_bytes(*hasher.finalize().as_bytes())
}

/// Declared assumptions of the observational sequential contrast.
fn contrast_assumptions() -> antecedent_core::AssumptionSet {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    for assumption in [Assumption::Consistency, Assumption::NoInterference, Assumption::Positivity]
    {
        assumptions.push(AssumptionRecord {
            assumption,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    for (id, description) in [
        (
            "sequential_conditional_exchangeability",
            "each action is independent of the potential outcomes given the measured history before it (declared, not checked)",
        ),
        (
            "no_unmeasured_second_decision_confounder",
            "no unmeasured confounder of the second decision is affected by the first action and omitted from the measured history",
        ),
        (
            "nuisance_product_rate",
            "the product of the propensity and conditional-mean errors is o(n^-1/2)",
        ),
    ] {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    assumptions
}

fn contrast_identification(
    query: &antecedent_core::SequentialContrastQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let assumptions = contrast_assumptions();
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let empty = arena.empty_var_set();
    let intervention = arena.intern_intervention_set([]);
    let distribution =
        arena.intern_distribution(outcomes, empty, intervention, DomainRef::Observational);
    let functional = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(query.outcome),
        distribution,
    });
    let rule = "sequential.crossfit_contrast";
    arena.set_derivation(
        functional,
        DerivationMeta::rule(
            rule,
            Some(Arc::from("sequential exchangeability given the measured history identifies the mean outcome under each static two-decision regime")),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push(
        rule,
        "the contrast of two static two-decision regimes is identified under declared sequential exchangeability, positivity and consistency",
    );
    let identification = IdentificationResult::identified(
        CausalQuery::SequentialContrast(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
