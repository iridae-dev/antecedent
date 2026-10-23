//! Retained sharp bounds on the average effect of a binary treatment under a binary instrument.
//!
//! One table row per unit; the instrument, the treatment and the outcome are 0/1 columns and an
//! optional sampling-weight column weights the rows. The observed law of the outcome and the
//! treatment in each instrument arm is read from the rows and the Balke-Pearl program is solved
//! on it ([`antecedent_estimate::IvBoundsEstimator`]). The instrument is declared independent of
//! the potential outcomes and excluded from the outcome; neither is checked. Those premises
//! partially identify the effect: the answer is the interval of effects the observed law leaves
//! open, never a point, and the identification status is `partially_identified`.
//!
//! Frequentist inference adds an Imbens-Manski interval from row-bootstrap endpoint replicates
//! when replicates are requested; Bayesian inference reports the Bayesian-bootstrap posterior of
//! the endpoints and the same tail rule. Rows are independent subjects, so a declared dependent
//! series, independent-trajectory or clustered design refuses the row-level draw. The route is
//! unlicensed and its calibration is unmeasured.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::bayesian_quasi_common::require_built_in_model;
use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationMeta, DerivationTrace, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate, EstimatorId, ExecutionContext,
    ExprNode, GraphClass, IdentificationPerformanceRecord, IdentificationResult,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId,
    InferenceMode, OutcomeExprId, OverlapPolicy, PhysicalExecutionPlan, RandomizedTrialOperation,
    RandomizedTrialQuery, RefuteSuite, Strategy, Study, StudyResult, TableView, TabularData,
    finish_identified_execute_with_context,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    IdentificationStatus,
};
use antecedent_estimate::IvBoundsEstimator;

/// Stable code of the disclosure every bounds result carries.
const UNLICENSED_DISCLOSURE: &str = "estimate.iv_bounds.unlicensed";

/// What a prepared bounds study keeps beside its query: the Bayesian configuration of a
/// posterior study and the row-bootstrap replicates of a frequentist one.
#[derive(Clone, Debug)]
pub(crate) struct IvBoundsSettings {
    bayesian: Option<super::BayesianConfig>,
    replicates: u32,
    sampling_design: antecedent_estimate::util::SamplingDeclaration,
}

/// Frozen bounds study.
pub(crate) type CheckedIvBoundsOperation =
    RandomizedTrialOperation<antecedent_core::IvBoundsQuery, IvBoundsSettings>;

impl RandomizedTrialQuery for antecedent_core::IvBoundsQuery {
    const OPERATION: &'static str = "CheckedIvBoundsOperation";
    const REFRESH_MISMATCH: &'static str =
        "IV bounds refresh requires the prepared schema and row order";
}

impl CheckedIvBoundsOperation {
    /// The design identification this operation records; the route lowering retains it.
    pub(crate) fn design_identification(
        query: &antecedent_core::IvBoundsQuery,
    ) -> (IdentificationResult, IdentifiedEstimand) {
        bounds_identification(query)
    }

    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::IvBounds(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "IV bounds operation requires IvBoundsQuery".into(),
            });
        };
        query.validate()?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || study.refute != RefuteSuite::None
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "IV bounds read one row per unit, with no graph, refuter or custom validator",
            });
        }
        let bayesian = match &study.inference {
            InferenceMode::Frequentist => None,
            InferenceMode::Bayesian(config) => {
                require_built_in_model(config, "the Bayesian IV bounds")?;
                Some(config.clone())
            }
        };
        let replicates = study.bootstrap_replicates;
        if replicates != 0 && bayesian.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "the Bayesian IV bounds draw the endpoints' posterior with n_draws and read no \
                 bootstrap replicates. Next attempt: omit bootstrap_replicates, or use \
                 frequentist inference for the Imbens-Manski row-bootstrap interval"
            ));
        }
        for id in query_columns(query) {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let estimator_id = estimator_for(bayesian.is_some());
        if physical.logical.query != study.query
            || !physical.binding().names(IdentifierId::RandomizedDesign, estimator_id)
        {
            return Err(CausalError::Compile {
                message: "IV bounds query differs from compiled plan".into(),
            });
        }
        let (identification, estimand) = bounds_identification(query);
        Ok(Self::freeze(
            study,
            query,
            data,
            physical,
            identification,
            estimand,
            IvBoundsSettings {
                bayesian,
                replicates,
                sampling_design: crate::estimator_spec::declared_sampling(study.sampling_design),
            },
        ))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        let started = std::time::Instant::now();
        let estimator = IvBoundsEstimator::new()
            .with_bootstrap_replicates(self.extra.replicates)
            .with_sampling_design(self.extra.sampling_design);
        let prepared = estimator.prepare(data, &self.query).map_err(CausalError::from)?;
        let estimate = match &self.extra.bayesian {
            Some(config) => estimator
                .estimate_bayesian(&prepared, config.n_draws, ctx)
                .map_err(CausalError::from)?,
            None => estimator.estimate(&prepared, ctx).map_err(CausalError::from)?,
        };
        let estimator_id = estimator_for(self.extra.bayesian.is_some());
        let n_draws = estimate
            .posterior
            .as_ref()
            .map(|posterior| u32::try_from(posterior.n_draws).unwrap_or(u32::MAX));
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
                self.query.treatment,
                self.query.outcome,
            )
            .identify_cached(true)
            .wall_time_ns(super::elapsed_ns(started))
            .extra_diagnostics(vec![disclosure(&estimate)])
            .extras(IdentifiedExecuteExtras { n_draws, ..Default::default() }),
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.treatment = None;
        result.support_status = None;
        result.payload = crate::result::ResultPayload::IvBounds(Box::new(estimate));
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_iv_bounds(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedIvBoundsOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

/// The columns a bounds query reads.
pub(crate) fn query_columns(
    query: &antecedent_core::IvBoundsQuery,
) -> Vec<antecedent_core::VariableId> {
    let mut columns = vec![query.outcome, query.treatment, query.instrument];
    columns.extend(query.weight);
    columns
}

/// The estimator of a bounds study under an inference mode.
fn estimator_for(bayesian: bool) -> EstimatorId {
    if bayesian { EstimatorId::BayesianIvBounds } else { EstimatorId::IvBoundsSharp }
}

fn disclosure(estimate: &antecedent_estimate::IvBoundsEstimate) -> Diagnostic {
    let interval = match &estimate.interval {
        Some(interval) => format!(
            "a {:.0}% Imbens-Manski style interval about the identified set ({})",
            interval.level * 100.0,
            interval.method.as_str()
        ),
        None => "the identified set without a sampling interval".to_owned(),
    };
    Diagnostic::new(
        UNLICENSED_DISCLOSURE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "unlicensed and uncalibrated route: the instrument is declared independent of the \
             potential outcomes and excluded from the outcome (neither is checked), the answer \
             is the identified set [{:.4}, {:.4}] of average effects the observed law leaves \
             open and not a point, and the result reports {interval}; no coverage record names it",
            estimate.lower, estimate.upper,
        ),
    )
    .with_fields([
        ("lower", estimate.lower.to_string()),
        ("upper", estimate.upper.to_string()),
        ("rows", estimate.rows.to_string()),
        ("rows_dropped", estimate.rows_dropped.to_string()),
    ])
}

fn bounds_assumptions(query: &antecedent_core::IvBoundsQuery) -> antecedent_core::AssumptionSet {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let declared = |assumption| AssumptionRecord {
        assumption,
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    };
    for assumption in [Assumption::Consistency, Assumption::NoInterference] {
        assumptions.push(declared(assumption));
    }
    assumptions.push(declared(Assumption::ExclusionRestriction { instrument: query.instrument }));
    for (id, description) in [
        (
            "instrument_independence",
            "the instrument is independent of the potential outcomes and potential treatments, \
             as when it is randomized; this is declared and not checked",
        ),
        (
            "instrument_relevance_not_required",
            "the bounds are valid for any instrument strength; a weak or irrelevant instrument \
             yields the no-assumption bounds",
        ),
    ] {
        assumptions.push(declared(Assumption::Custom {
            id: Arc::from(id),
            description: Arc::from(description),
        }));
    }
    assumptions
}

fn bounds_identification(
    query: &antecedent_core::IvBoundsQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let assumptions = bounds_assumptions(query);
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
    let rule = "iv_bounds.balke_pearl";
    let note = "a binary instrument independent of the potential outcomes and excluded from the \
                outcome restricts the 16 response types of a binary treatment and outcome; the \
                observed law bounds the average effect to a sharp interval, not a point";
    arena.set_derivation(functional, DerivationMeta::rule(rule, Some(Arc::from(note))));
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push(rule, note);
    let mut identification = IdentificationResult::identified(
        CausalQuery::IvBounds(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    identification.status = IdentificationStatus::PartiallyIdentified;
    (identification, estimand)
}
