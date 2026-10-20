//! Retained linear/Gaussian intervention route.
//!
//! Lowers the existing static `linear.adjustment.ate` contrast through the six
//! compiler layers and executes the checked program that lowering retains.
//! [`super::super::Study::compile`] returns this route's physical record when
//! the study is the intervention this module admits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(not(test), allow(dead_code))]

use std::cell::RefCell;
use std::sync::Arc;

use antecedent_contract::ExecutionPerformanceRecord;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionSet, AssumptionStatus, AverageEffectQuery, CausalQuery,
    ExecutionContext, Operator, ScientificTarget,
};
use antecedent_data::TabularData;
use antecedent_estimate::{
    CheckedLinearAdjustmentAte, EffectEstimate, EstimationWorkspace, LinearAdjustmentAte,
    LinearFitKind,
};
use antecedent_graph::Dag;
use antecedent_identify::{
    AssembledFunctionalObligation, CompiledIdentificationProduct, CompiledProductParts,
    IdentificationClaimClass, IdentificationOutcome, IdentificationResult, RequiredFactor,
    UnresolvedObligation, WorldObligation,
};

use crate::analysis::prepared::recompile::{LifecycleInput, LifecycleProduct, PreparedRecompile};
use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;
use crate::planner::{
    PhysicalExecutionPlan, linear_gaussian_physical_operators, linear_gaussian_program_operators,
    linear_gaussian_scientific_problem,
};
use crate::strategy_table::{
    EstimatorId, IdentifierId, identify_static_query_with_rd, select_claim,
};
use crate::support::CellStatus;

use super::super::Study;

const CONTROL_WORLD: u32 = 0;
const ACTIVE_WORLD: u32 = 1;

/// Decision request compiled for this route.
///
/// The route asks for the contrast mean. It does not ask for a utility, a
/// constraint, or a value of information.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinearGaussianDecision {
    /// Mean of the active-minus-control contrast.
    ContrastMean,
}

/// Evidence program: support status, provenance/execution/causal edges, and the
/// decision request. Assumptions stay on the identification product.
#[derive(Clone, Debug)]
pub(crate) struct LinearGaussianEvidenceProgram {
    support_status: Option<CellStatus>,
    operators: Vec<Operator>,
    decision: LinearGaussianDecision,
}

impl LinearGaussianEvidenceProgram {
    /// Support status copied from the study that was lowered.
    #[must_use]
    pub(crate) const fn support_status(&self) -> Option<CellStatus> {
        self.support_status
    }

    /// Read, fit, and intervene edges.
    #[must_use]
    pub(crate) fn operators(&self) -> &[Operator] {
        &self.operators
    }

    /// The decision request. This route has one.
    #[must_use]
    pub(crate) const fn decision(&self) -> LinearGaussianDecision {
        self.decision
    }
}

/// Claim produced by executing the retained program.
#[derive(Clone, Debug)]
pub(crate) struct LinearGaussianClaim {
    /// Point and uncertainty from the retained fit.
    pub(crate) estimate: EffectEstimate,
    /// Support status the study carried when it was lowered.
    pub(crate) support_status: Option<CellStatus>,
    /// Execution facts. Byte estimates stay on the physical plan.
    pub(crate) receipt: ExecutionPerformanceRecord,
}

/// Static linear/Gaussian intervention route, retained through all six layers.
///
/// Execute fits this program. It does not build another study.
#[derive(Clone, Debug)]
#[must_use = "execute the retained linear/Gaussian program"]
pub(crate) struct LoweredLinearGaussianRoute {
    scientific: ScientificTarget,
    identification: CompiledIdentificationProduct,
    program: CheckedLinearAdjustmentAte,
    fitter: LinearAdjustmentAte,
    /// The id-selected route fits the point, then bootstraps. A configured OLS
    /// fitter already bootstraps inside `fit_checked`.
    point_then_bootstrap: bool,
    evidence: LinearGaussianEvidenceProgram,
    physical: PhysicalExecutionPlan,
    physical_operators: Vec<antecedent_core::PhysicalOperator>,
    /// Products whose validity decides whether the next call fits.
    recompile: RefCell<PreparedRecompile>,
    /// Claim from the last causal fit. Absent until that fit succeeds.
    cached_claim: RefCell<Option<LinearGaussianClaim>>,
}

impl LoweredLinearGaussianRoute {
    /// Layer 1. The contrast of the two intervention worlds.
    #[must_use]
    pub(crate) const fn scientific(&self) -> &ScientificTarget {
        &self.scientific
    }

    /// Layer 2. Functional, derivation, assumptions, factors, and obligations.
    pub(crate) const fn identification(&self) -> &CompiledIdentificationProduct {
        &self.identification
    }

    /// Layer 3. Checked linear program. Execution uses this value.
    #[must_use]
    pub(crate) const fn program(&self) -> &CheckedLinearAdjustmentAte {
        &self.program
    }

    /// Layer 4.
    #[must_use]
    pub(crate) const fn evidence(&self) -> &LinearGaussianEvidenceProgram {
        &self.evidence
    }

    /// Layer 5 record. The same record [`Study::compile`] returns for this route.
    #[must_use]
    pub(crate) const fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }

    /// Physical record [`Study::compile`] returns after building this program.
    #[must_use = "the physical plan is the compile result"]
    pub(crate) fn into_physical(self) -> PhysicalExecutionPlan {
        self.physical
    }

    /// Layer 5 operators.
    #[must_use]
    pub(crate) fn physical_operators(&self) -> &[antecedent_core::PhysicalOperator] {
        &self.physical_operators
    }

    /// Fit the retained program.
    ///
    /// A later call reuses the retained causal fit while that product is still
    /// valid. Utility and action changes leave it valid, so they perform no
    /// new causal fit. The numeric fit, when it does run, is the ordinary
    /// least-squares program this route already retained.
    ///
    /// # Errors
    ///
    /// The context is already cancelled, or the retained fit fails.
    pub(crate) fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<LinearGaussianClaim, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        if self.recompile.borrow().reuses_causal_fit() {
            if let Some(claim) = self.cached_claim.borrow().clone() {
                self.recompile.borrow_mut().retain_decision();
                return Ok(claim);
            }
        }
        let mut workspace = EstimationWorkspace::default();
        let estimate = if self.point_then_bootstrap {
            let point = self
                .fitter
                .fit_point(
                    self.program.problem(),
                    &mut workspace,
                    self.program.required_assumptions().clone(),
                )
                .map_err(CausalError::from)?;
            if self.fitter.bootstrap_replicates == 0 || ctx.cancellation.is_cancelled() {
                point
            } else {
                self.fitter
                    .attach_bootstrap(self.program.problem(), &mut workspace, ctx, point)
                    .map_err(CausalError::from)?
            }
        } else {
            self.fitter
                .fit_checked(&self.program, &mut workspace, ctx)
                .map_err(CausalError::from)?
        };
        let claim = LinearGaussianClaim {
            receipt: receipt(&estimate, self.fitter.bootstrap_replicates),
            support_status: self.evidence.support_status,
            estimate,
        };
        *self.cached_claim.borrow_mut() = Some(claim.clone());
        self.recompile.borrow_mut().retain_causal_fit();
        Ok(claim)
    }

    /// Apply one lifecycle change and drop a cached claim whose fit it invalidates.
    pub(crate) fn apply_input(&self, input: LifecycleInput) {
        let keep_fit = {
            let mut state = self.recompile.borrow_mut();
            state.apply(input);
            state.reuses_causal_fit()
        };
        if !keep_fit {
            self.cached_claim.borrow_mut().take();
        }
    }

    /// Whether `product` is still valid on this route.
    #[must_use = "an invalid product has to be rebuilt"]
    pub(crate) fn product_is_valid(&self, product: LifecycleProduct) -> bool {
        self.recompile.borrow().is_valid(product)
    }

    /// Causal fits this route has run.
    #[must_use = "compare this with the count from before the change"]
    pub(crate) fn causal_fit_count(&self) -> u64 {
        self.recompile.borrow().causal_fits()
    }
}

fn receipt(estimate: &EffectEstimate, requested: u32) -> ExecutionPerformanceRecord {
    let bootstrap = &estimate.uncertainty.bootstrap;
    ExecutionPerformanceRecord {
        bootstrap_replicates_requested: Some(requested),
        bootstrap_replicates_ok: bootstrap.replicates_ok,
        cancelled: bootstrap.cancelled,
        early_stopped: bootstrap.early_stopped,
        copy_count: 1,
        ..ExecutionPerformanceRecord::default()
    }
}

struct Admitted<'a> {
    data: &'a TabularData,
    query: &'a AverageEffectQuery,
    graph: &'a Dag,
    identifier: IdentifierId,
}

impl Study {
    /// Lower the static ordinary-least-squares intervention route.
    ///
    /// The study must resolve to `linear.adjustment.ate` on a static DAG average
    /// contrast with an empty modifier list. Population, pair, and identification
    /// refusals are the ones [`Self::compile`] and the checked estimator already
    /// return. The retained program is what [`LoweredLinearGaussianRoute::execute`]
    /// fits.
    ///
    /// # Errors
    ///
    /// The study is a different route, the fit is not ordinary least squares, or
    /// identification or preparation refuses.
    pub(crate) fn lower_linear_gaussian_intervention(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<LoweredLinearGaussianRoute, CausalError> {
        let admitted = self.admit_linear_gaussian()?;
        let logical =
            self.compile_logical_ate_dag(admitted.data, admitted.query, admitted.graph)?;
        let physical = logical.compile_physical(ctx)?;
        let identification = self.identify_linear_gaussian(&admitted, ctx)?;
        let (program, fitter, point_then_bootstrap) =
            prepare_linear_gaussian(self, admitted.data, &identification)?;
        let product = retain_linear_gaussian(&identification, &program)?;
        let resample = fitter.bootstrap_replicates > 0;
        Ok(LoweredLinearGaussianRoute {
            scientific: linear_gaussian_scientific_problem(
                admitted.query,
                self.inference.is_bayesian(),
            )?,
            identification: product,
            program,
            fitter,
            point_then_bootstrap,
            evidence: LinearGaussianEvidenceProgram {
                support_status: self.support_status,
                operators: linear_gaussian_program_operators()?,
                decision: LinearGaussianDecision::ContrastMean,
            },
            physical_operators: linear_gaussian_physical_operators(&physical, resample)?,
            physical,
            recompile: RefCell::new(PreparedRecompile::after_lower()),
            cached_claim: RefCell::new(None),
        })
    }

    /// Whether this study is the static ordinary-least-squares intervention.
    ///
    /// A refusal from identification or the fit is not a mismatch: [`Self::compile`]
    /// returns that error instead of another route.
    pub(super) fn admits_linear_gaussian(&self) -> bool {
        self.admit_linear_gaussian().is_ok()
    }

    fn admit_linear_gaussian(&self) -> Result<Admitted<'_>, CausalError> {
        let (identifier, estimator) = self.resolve_static_ids();
        if estimator != EstimatorId::LinearAdjustmentAte {
            return Err(CausalError::Compile {
                message: format!(
                    "linear/Gaussian lowering admits estimator \"{}\"; got \"{}\"",
                    EstimatorId::LinearAdjustmentAte.as_str(),
                    estimator.as_str()
                ),
            });
        }
        let super::super::RouteInput::StaticAteDag { data, query, graph } = self.route_input()?
        else {
            return Err(CausalError::Compile {
                message: "linear/Gaussian lowering admits a static DAG average contrast".into(),
            });
        };
        if !query.effect_modifiers.is_empty() {
            return Err(CausalError::Compile {
                message: "linear/Gaussian lowering admits an unconditional average contrast".into(),
            });
        }
        Ok(Admitted { data, query, graph, identifier })
    }

    fn identify_linear_gaussian(
        &self,
        admitted: &Admitted<'_>,
        ctx: &ExecutionContext,
    ) -> Result<IdentificationResult, CausalError> {
        let rd = self.rd.map(|config| {
            antecedent_identify::SharpRdConfig::new(
                config.running_variable,
                config.cutoff,
                config.bandwidth,
            )
        });
        let (identification, _, _) = super::super::identification_from_cache_or(
            ctx,
            self.identification_cache.as_deref(),
            || {
                let identification = identify_static_query_with_rd(
                    admitted.identifier,
                    admitted.graph,
                    &CausalQuery::AverageEffect(admitted.query.clone()),
                    rd,
                )?;
                select_claim(identification, EstimatorId::LinearAdjustmentAte)
            },
        )?;
        Ok(identification)
    }
}

fn prepare_linear_gaussian(
    study: &Study,
    data: &TabularData,
    identification: &IdentificationResult,
) -> Result<(CheckedLinearAdjustmentAte, LinearAdjustmentAte, bool), CausalError> {
    let (fitter, point_then_bootstrap) = gaussian_fitter(study)?;
    let program = fitter.prepare_checked(data, identification, 0).map_err(CausalError::from)?;
    Ok((program, fitter, point_then_bootstrap))
}

fn gaussian_fitter(study: &Study) -> Result<(LinearAdjustmentAte, bool), CausalError> {
    match study.estimator_spec.as_deref() {
        None | Some(EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte)) => {
            let mut fitter = LinearAdjustmentAte::new();
            fitter.bootstrap_replicates = study.bootstrap_replicates;
            Ok((fitter, true))
        }
        Some(EstimatorSpec::LinearAdjustmentAte(cfg)) => {
            if !matches!(cfg.fit_kind, LinearFitKind::Ols) {
                return Err(CausalError::Compile {
                    message: "linear/Gaussian lowering admits ordinary least squares".into(),
                });
            }
            Ok(((**cfg).clone(), false))
        }
        Some(other) => Err(CausalError::Compile {
            message: format!(
                "linear/Gaussian lowering admits estimator \"{}\"; got \"{}\"",
                EstimatorId::LinearAdjustmentAte.as_str(),
                other.id().as_str()
            ),
        }),
    }
}

fn retain_linear_gaussian(
    identification: &IdentificationResult,
    program: &CheckedLinearAdjustmentAte,
) -> Result<CompiledIdentificationProduct, CausalError> {
    if program.required_assumptions() != &identification.required_assumptions {
        return Err(CausalError::Compile {
            message: "checked linear program dropped an identification assumption".into(),
        });
    }
    let unresolved = unresolved_obligations(&identification.required_assumptions);
    let factors = required_factors(program);
    let (outcome, claim_class) = identification_claim(identification.status);
    let worlds = world_obligations(outcome, claim_class, &unresolved);
    let assembled =
        AssembledFunctionalObligation::new(outcome, claim_class, unresolved.clone(), None);
    let mut parts = CompiledProductParts::new(outcome, claim_class);
    parts.required_factors = factors;
    parts.unresolved_obligations = unresolved;
    parts.world_obligations = worlds;
    parts.assembled = Some(assembled);
    CompiledIdentificationProduct::retain(identification, parts)
        .map_err(|refusal| CausalError::Compile { message: refusal.to_string() })
}

fn required_factors(program: &CheckedLinearAdjustmentAte) -> Vec<RequiredFactor> {
    program
        .factor_requirements()
        .iter()
        .enumerate()
        .map(|(index, factor)| {
            RequiredFactor::new(format!("factor.{index}"), Arc::clone(&factor.variables))
        })
        .collect()
}

fn world_obligations(
    outcome: IdentificationOutcome,
    claim_class: IdentificationClaimClass,
    unresolved: &[UnresolvedObligation],
) -> Vec<WorldObligation> {
    [CONTROL_WORLD, ACTIVE_WORLD]
        .into_iter()
        .map(|raw| {
            WorldObligation::new(
                antecedent_core::ScientificWorldId::from_raw(raw),
                outcome,
                claim_class,
                unresolved.to_vec(),
                None,
            )
        })
        .collect()
}

fn identification_claim(
    status: antecedent_core::IdentificationStatus,
) -> (IdentificationOutcome, IdentificationClaimClass) {
    match status {
        antecedent_core::IdentificationStatus::NonparametricallyIdentified
        | antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions
        | antecedent_core::IdentificationStatus::IdentifiedUnderPriorRestrictions => {
            (IdentificationOutcome::Identified, IdentificationClaimClass::IdentifiedClaim)
        }
        antecedent_core::IdentificationStatus::PartiallyIdentified
        | antecedent_core::IdentificationStatus::GraphDependent
        | antecedent_core::IdentificationStatus::NotIdentified => {
            (IdentificationOutcome::Unidentified, IdentificationClaimClass::IdentifiedClaim)
        }
    }
}

fn unresolved_obligations(assumptions: &AssumptionSet) -> Vec<UnresolvedObligation> {
    assumptions
        .entries
        .iter()
        .enumerate()
        .filter(|(_, record)| {
            matches!(record.status, AssumptionStatus::Declared | AssumptionStatus::Contradicted)
        })
        .map(|(index, record)| {
            UnresolvedObligation::new(
                obligation_code(&record.assumption, index),
                obligation_detail(record),
            )
        })
        .collect()
}

fn obligation_code(assumption: &Assumption, index: usize) -> String {
    let stem = match assumption {
        Assumption::CausalMarkov => "causal_markov",
        Assumption::Faithfulness => "faithfulness",
        Assumption::CausalSufficiency => "causal_sufficiency",
        Assumption::Consistency => "consistency",
        Assumption::Positivity => "positivity",
        Assumption::NoInterference => "no_interference",
        Assumption::Stationarity => "stationarity",
        Assumption::PiecewiseStationarity => "piecewise_stationarity",
        Assumption::NoSelectionBias => "no_selection_bias",
        Assumption::ExclusionRestriction { .. } => "exclusion_restriction",
        Assumption::Monotonicity => "monotonicity",
        Assumption::ParametricRestriction(_) => "parametric_restriction",
        Assumption::PriorRestriction(_) => "prior_restriction",
        Assumption::Custom { .. } => "custom",
    };
    format!("{stem}.{index}")
}

fn obligation_detail(record: &AssumptionRecord) -> String {
    match &record.assumption {
        Assumption::ParametricRestriction(restriction) => {
            format!("{}: {}", record.status.as_str(), restriction.id)
        }
        Assumption::PriorRestriction(restriction) => {
            format!("{}: {}", record.status.as_str(), restriction.id)
        }
        Assumption::Custom { id, .. } => format!("{}: {id}", record.status.as_str()),
        _ => record.status.as_str().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, TargetForm, TargetPopulation,
    };
    use antecedent_data::TabularData;
    use antecedent_estimate::{LinearAdjustmentAte, LinearFitKind};
    use antecedent_graph::Dag;
    use antecedent_identify::IdentificationOutcome;

    use antecedent_stats::fit_counts::count_least_squares_solves;

    use crate::RefuteSuite;
    use crate::analysis::prepared::recompile::{LifecycleInput, LifecycleProduct};
    use crate::strategy_table::EstimatorId;

    use super::super::super::Study;

    fn contrast_study() -> Study {
        let schema = CausalSchemaBuilder::new()
            .continuous("t")
            .treatment()
            .continuous("y")
            .outcome()
            .build()
            .unwrap();
        let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let outcome = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let data = TabularData::try_from_schema_f64(
            schema.clone(),
            [("t", treatment.as_slice()), ("y", outcome.as_slice())],
        )
        .unwrap();
        let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
        let query =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        Study::tabular(data)
            .graph(graph)
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "the retained fit must match Study::run bit for bit")]
    fn linear_gaussian_route_executes_without_rebuilding_a_builder() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        let expected = study.run(&ctx).unwrap().effect();
        drop(study);
        let claim = route.execute(&ctx).unwrap();
        assert_eq!(claim.estimate.ate, expected);
        assert_eq!(claim.receipt.bootstrap_replicates_requested, Some(0));
        assert!(matches!(route.scientific().form(), TargetForm::Contrast(_)));
        let again = route.execute(&ctx).unwrap();
        assert_eq!(claim.estimate.se_analytic(), again.estimate.se_analytic());
        assert_ne!(route.program().executable_functional(), route.program().source_functional());
        assert_eq!(route.evidence().decision(), super::LinearGaussianDecision::ContrastMean);
    }

    #[test]
    fn linear_gaussian_route_keeps_support_obligations() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        assert_eq!(route.evidence().support_status(), study.support_status());
        assert_eq!(route.identification().assumptions(), route.program().required_assumptions());
        assert!(route.identification().is_identified_claim());
        assert_eq!(route.identification().outcome(), IdentificationOutcome::Identified);
        assert_eq!(route.identification().world_obligations().len(), 2);
        assert!(
            route
                .identification()
                .assembled_obligation()
                .is_some_and(|obligation| { obligation.is_identified_claim() })
        );
        let claim = route.execute(&ctx).unwrap();
        assert_eq!(&claim.estimate.provenance.assumptions, route.identification().assumptions());
        assert_eq!(claim.support_status, study.support_status());
    }

    #[test]
    fn linear_gaussian_route_preserves_physical_plan() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        let compiled = study.compile(&ctx).unwrap();
        assert_eq!(route.physical().record, compiled.record);
        let effects: Vec<_> =
            route.physical_operators().iter().map(|op| op.effect().as_str()).collect();
        assert_eq!(effects, ["read", "fit", "intervene"]);
        let graphs: Vec<_> =
            route.evidence().operators().iter().map(|op| op.graph().as_str()).collect();
        assert_eq!(graphs, ["provenance", "execution", "causal"]);
    }

    #[test]
    fn linear_gaussian_route_refuses_a_different_estimator() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        assert!(study.compile(&ctx).is_ok());
        let refused = rebuild_like(&study, true, |builder| {
            builder.estimator(EstimatorId::PropensityWeighting)
        });
        let err = refused.lower_linear_gaussian_intervention(&ctx).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("linear.adjustment.ate"), "{message}");
    }

    #[test]
    fn linear_gaussian_route_refuses_another_target_population() {
        let base = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let study = rebuild_like(&base, true, |builder| {
            let query = match base.query.clone() {
                antecedent_core::CausalQuery::AverageEffect(query) => {
                    query.with_target_population(TargetPopulation::Treated)
                }
                other => panic!("expected an average effect, got {other:?}"),
            };
            builder.query(query)
        });
        let err = study.lower_linear_gaussian_intervention(&ctx).unwrap_err();
        let message = err.to_string();
        assert_eq!(
            antecedent_core::reason_code::split_prefix(&message).map(|(code, _)| code),
            Some("population_not_estimable"),
            "{message}"
        );
    }

    #[test]
    fn linear_gaussian_route_refuses_non_ols_fit() {
        let base = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let ridge = LinearAdjustmentAte::new().with_fit_kind(LinearFitKind::Ridge { lambda: 1.0 });
        let study = rebuild_like(&base, false, |builder| builder.estimator(ridge));
        let err = study.lower_linear_gaussian_intervention(&ctx).unwrap_err();
        assert!(err.to_string().contains("ordinary least squares"), "{err}");
    }

    #[test]
    fn prepared_linear_gaussian_study_executes_the_retained_program() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        let compiled = study.compile(&ctx).unwrap();
        assert_eq!(compiled.record, route.physical().record);
        assert_eq!(compiled.logical.record, route.physical().logical.record);
        let prepared = study.prepare(&ctx).unwrap();
        assert_eq!(prepared.plan().record, route.physical().record);
        let sealed = prepared.checked_linear_adjustment().unwrap();
        assert_eq!(sealed.executable_functional(), route.program().executable_functional());
        assert_eq!(sealed.source_functional(), route.program().source_functional());
        assert_eq!(sealed.required_assumptions(), route.program().required_assumptions());
        let expected = study.run(&ctx).unwrap().effect();
        let claim = route.execute(&ctx).unwrap();
        assert_eq!(claim.estimate.ate.to_bits(), expected.to_bits());
        assert_eq!(&claim.estimate.provenance.assumptions, route.program().required_assumptions(),);
        let retained = prepared.estimate_retained(&ctx).unwrap();
        assert_eq!(retained.effect().to_bits(), claim.estimate.ate.to_bits());
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "a reused fit must return the same contrast bits")]
    fn linear_gaussian_utility_and_action_changes_perform_zero_causal_fits() {
        let (route, ctx, ate, fits_before) = fitted_route();
        for input in [LifecycleInput::Utility, LifecycleInput::Action] {
            route.apply_input(input);
            assert!(route.product_is_valid(LifecycleProduct::CausalFit), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Structural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Unrelated), "{input:?}");
            assert!(!route.product_is_valid(LifecycleProduct::Decision), "{input:?}");
            let (claim, solves) = count_least_squares_solves(|| route.execute(&ctx).unwrap());
            assert_eq!(solves, 0, "{input:?}");
            assert_eq!(route.causal_fit_count(), fits_before, "{input:?}");
            assert_eq!(claim.estimate.ate, ate, "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Decision), "{input:?}");
        }
    }

    #[test]
    fn linear_gaussian_prior_change_preserves_only_valid_structure() {
        let (route, ctx, _, fits_before) = fitted_route();
        route.apply_input(LifecycleInput::Prior);
        assert!(route.product_is_valid(LifecycleProduct::Structural));
        assert!(route.product_is_valid(LifecycleProduct::Unrelated));
        assert!(!route.product_is_valid(LifecycleProduct::PriorStructural));
        assert!(!route.product_is_valid(LifecycleProduct::CausalFit));
        assert!(!route.product_is_valid(LifecycleProduct::Decision));
        let (_, solves) = count_least_squares_solves(|| route.execute(&ctx).unwrap());
        assert!(solves > 0);
        assert_eq!(route.causal_fit_count(), fits_before + 1);
    }

    #[test]
    fn linear_gaussian_source_period_and_history_changes_keep_unrelated_products() {
        for input in [LifecycleInput::Source, LifecycleInput::Period, LifecycleInput::History] {
            let (route, ctx, _, _) = fitted_route();
            route.apply_input(input);
            assert!(!route.product_is_valid(LifecycleProduct::CausalFit), "{input:?}");
            assert!(!route.product_is_valid(LifecycleProduct::Decision), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Structural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::PriorStructural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Unrelated), "{input:?}");
            let (_, solves) = count_least_squares_solves(|| route.execute(&ctx).unwrap());
            assert!(solves > 0, "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Structural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::PriorStructural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Unrelated), "{input:?}");
        }
    }

    fn fitted_route() -> (super::LoweredLinearGaussianRoute, ExecutionContext, f64, u64) {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        let compiled = study.compile(&ctx).unwrap();
        assert_eq!(compiled.record, route.physical().record);
        assert_eq!(compiled.logical.record, route.physical().logical.record);
        let (claim, solves) = count_least_squares_solves(|| route.execute(&ctx).unwrap());
        assert!(solves > 0);
        let fits = route.causal_fit_count();
        assert_eq!(fits, 1);
        (route, ctx, claim.estimate.ate, fits)
    }

    fn rebuild_like(
        study: &Study,
        zero_bootstrap: bool,
        edit: impl FnOnce(crate::analysis::StudyBuilder) -> crate::analysis::StudyBuilder,
    ) -> Study {
        let antecedent_core::CausalQuery::AverageEffect(query) = study.query.clone() else {
            panic!("expected an average effect");
        };
        let data = match &study.data {
            crate::analysis::builder::DataInput::Tabular(data) => data.clone(),
            _ => panic!("expected tabular data"),
        };
        let graph = study.graph.as_dag().unwrap().clone();
        let mut builder = Study::tabular(data).graph(graph).query(query).refute(RefuteSuite::None);
        if zero_bootstrap {
            builder = builder.bootstrap_replicates(0);
        }
        edit(builder).build().unwrap()
    }
}
