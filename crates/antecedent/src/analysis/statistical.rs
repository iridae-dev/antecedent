//! Empirical-table modality of the retained common prepared-study lifecycle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::exact::{ExactFactorRequirement, ExactStudyIdentities};
use super::{PreparedStudy, StudyBuilder};
use antecedent_core::{
    AssumptionSlot, AssumptionSource, AssumptionStatus, ExecutionContext, IdentificationSlot,
    IdentificationStatus, IdentityDomain, NodeRef, ObligationKind, ObligationRecord,
    ObligationScope, ReasoningView, SlotAvailability, SupportSlot, TheoremScope,
    UncertaintyComponent, UncertaintySlot, UncertaintySource, VariableId,
};
use antecedent_estimate::{
    EmpiricalTableOptions, StatisticalTransportEstimate, StatisticalTransportInput,
    TransportUncertaintyRow, assemble_point_laws,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{BoundTransportFunctional, ClassicalTransportQuery, SidLimits};
use antecedent_io::transport_grid_wire::{
    BayesianTransportPosteriorWire, SampleSummary, StatisticalOptionsWire,
};
use antecedent_io::{
    IoError, exact_law_wire::ExactLawWire, query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire, transport_proof::TransportProofWire,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::result::PERCENTILE_95_MIN_REPLICATES;

use super::transport_common::{GraphFields, digest, err, rebind_snapshots, rebuild_checked_proof};

fn is_bayesian_provider(estimator: antecedent_estimate::EmpiricalTableEstimator) -> bool {
    matches!(
        estimator,
        antecedent_estimate::EmpiricalTableEstimator::EmpiricalSupportBayesianBootstrap
            | antecedent_estimate::EmpiricalTableEstimator::StateSpaceDirichlet
    )
}

fn bayesian_provider(
    estimator: antecedent_estimate::EmpiricalTableEstimator,
) -> Option<antecedent_estimate::BayesianTransportLawProvider> {
    match estimator {
        antecedent_estimate::EmpiricalTableEstimator::EmpiricalSupportBayesianBootstrap => {
            Some(antecedent_estimate::BayesianTransportLawProvider::EmpiricalSupport)
        }
        antecedent_estimate::EmpiricalTableEstimator::StateSpaceDirichlet => {
            Some(antecedent_estimate::BayesianTransportLawProvider::DeclaredStateSpaceDirichlet)
        }
        _ => None,
    }
}

fn apply_posterior_draw_floor(
    estimate: &mut antecedent_estimate::BayesianStatisticalTransportEstimate,
) {
    if estimate.interval_reason.as_deref()
        != Some(antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED)
    {
        return;
    }
    if estimate.draws_requested < PERCENTILE_95_MIN_REPLICATES
        || estimate.draws_ok < PERCENTILE_95_MIN_REPLICATES
    {
        estimate.atom_intervals = Arc::from([]);
        estimate.mean_intervals = Arc::from([]);
        estimate.interval_reason = Some(Arc::from("insufficient_bootstrap_replicates"));
    }
}

fn bayesian_uncertainty_slot(
    requested: u32,
    dependence: Option<&str>,
    posterior: Option<&antecedent_estimate::BayesianStatisticalTransportEstimate>,
) -> SlotAvailability<UncertaintySlot> {
    if let Some(posterior) = posterior {
        return match posterior.interval_reason.as_deref() {
            Some(antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED) => {
                SlotAvailability::Available(UncertaintySlot::new([UncertaintyComponent::new(
                    UncertaintySource::Sampling,
                    antecedent_estimate::POSTERIOR_EQUAL_TAIL,
                    false,
                )]))
            }
            Some(reason) => SlotAvailability::unavailable(reason),
            None => SlotAvailability::unavailable("uncertainty_unavailable"),
        };
    }
    if let Some(reason) = dependence {
        return SlotAvailability::unavailable(reason);
    }
    if requested < PERCENTILE_95_MIN_REPLICATES {
        SlotAvailability::unavailable("insufficient_bootstrap_replicates")
    } else {
        SlotAvailability::Available(UncertaintySlot::new([UncertaintyComponent::new(
            UncertaintySource::Sampling,
            antecedent_estimate::POSTERIOR_EQUAL_TAIL,
            false,
        )]))
    }
}

fn point_options(options: EmpiricalTableOptions) -> EmpiricalTableOptions {
    if is_bayesian_provider(options.estimator) {
        EmpiricalTableOptions {
            estimator: antecedent_estimate::EmpiricalTableEstimator::Plugin,
            bootstrap_replicates: 0,
            ..options
        }
    } else {
        options
    }
}

/// Native retained statistical-transport state.
#[derive(Clone, Debug)]
pub struct StatisticalPreparedState {
    diagram: SelectionDiagram,
    functional: BoundTransportFunctional,
    input: StatisticalTransportInput,
    data: ExactTransportData,
    request: Assignment,
    limits: ExactEvaluationLimits,
    options: EmpiricalTableOptions,
    identities: ExactStudyIdentities,
    shape: String,
    seed: u64,
    samples: Vec<SampleSummary>,
    loaded_only: bool,
    grid: Vec<Vec<(u32, ValueWire)>>,
}

/// Statistical execution result: plug-in law plus licensed or withheld uncertainty.
#[derive(Clone, Debug)]
pub struct StatisticalStudyResult {
    estimate: StatisticalTransportEstimate,
    bayesian_estimate: Option<antecedent_estimate::BayesianStatisticalTransportEstimate>,
    identities: ExactStudyIdentities,
    reasoning: ReasoningView,
}

impl StatisticalStudyResult {
    /// Plug-in target law.
    #[must_use]
    pub const fn distribution(&self) -> &antecedent_expr::ExactDistribution {
        &self.estimate.distribution
    }
    /// Interval row and joint replicates.
    #[must_use]
    pub const fn estimate(&self) -> &StatisticalTransportEstimate {
        &self.estimate
    }
    /// Optional Bayesian posterior law summaries for Bayesian transport providers.
    #[must_use]
    pub const fn bayesian_estimate(
        &self,
    ) -> Option<&antecedent_estimate::BayesianStatisticalTransportEstimate> {
        self.bayesian_estimate.as_ref()
    }
    /// Paired difference of means, preserving shared-sample covariance.
    /// # Errors
    /// Different evidence, snapshots, inference settings or unavailable outcome.
    pub fn contrast(
        &self,
        reference: &Self,
        outcome: VariableId,
    ) -> Result<StatisticalContrast, IoError> {
        if self.identities.identification != reference.identities.identification
            || self.identities.snapshot != reference.identities.snapshot
            || self.identities.inference_binding != reference.identities.inference_binding
        {
            return Err(err(
                "contrast requires the same evidence, snapshots and bootstrap settings",
            ));
        }
        let point =
            self.distribution().mean_difference(reference.distribution(), outcome).map_err(err)?;
        let mut contrast = StatisticalContrast {
            estimate: point,
            interval: None,
            coverage_target: None,
            replicates_ok: 0,
            replicates_failed: 0,
            reason: Some("uncertainty_unavailable"),
        };
        // A point's own `uncertainty_reason` also covers the facade's licensing floor
        // (`insufficient_bootstrap_replicates`), which withholds only that point's
        // *pointwise* percentile edges while still retaining its replicate draws. The
        // contrast draws its own paired sample from those retained replicates and
        // applies its own floor below, so only a hard absence of replicate data (no
        // bootstrap ran at all, or the law came from an exact supplied source) should
        // withhold the contrast; that absence is what the destructure below detects.
        let (Some(row), Some(ids), Some(reference_ids), Some(left), Some(right)) = (
            &self.estimate.uncertainty,
            &self.estimate.replicate_ids,
            &reference.estimate.replicate_ids,
            &self.estimate.mean_replicates,
            &reference.estimate.mean_replicates,
        ) else {
            return Ok(contrast);
        };
        let left = left
            .iter()
            .find(|(v, _)| *v == outcome)
            .ok_or_else(|| err("unknown contrast outcome"))?;
        let right = right
            .iter()
            .find(|(v, _)| *v == outcome)
            .ok_or_else(|| err("unknown contrast outcome"))?;
        let pairs: std::collections::BTreeMap<_, _> =
            reference_ids.iter().zip(right.1.iter()).collect();
        let draws: Vec<_> = ids
            .iter()
            .zip(left.1.iter())
            .filter_map(|(id, value)| pairs.get(id).map(|other| value - **other))
            .collect();
        contrast.replicates_ok = u32::try_from(draws.len()).map_err(err)?;
        contrast.replicates_failed = row
            .replicates_requested
            .checked_sub(contrast.replicates_ok)
            .ok_or_else(|| err("invalid replicate accounting"))?;
        contrast.coverage_target = Some(row.coverage_target);
        if draws.len() < PERCENTILE_95_MIN_REPLICATES as usize
            || f64::from(contrast.replicates_failed) / f64::from(row.replicates_requested) > 0.5
        {
            contrast.reason = Some("bootstrap_failure_fraction");
        } else {
            contrast.interval =
                Some(antecedent_estimate::percentile_interval(&draws, row.coverage_target));
            contrast.reason = None;
        }
        Ok(contrast)
    }

    /// Executed identity layers.
    #[must_use]
    pub const fn identities(&self) -> &ExactStudyIdentities {
        &self.identities
    }
    /// Four typed reasoning slots.
    #[must_use]
    pub const fn reasoning(&self) -> &ReasoningView {
        &self.reasoning
    }
}

/// A paired mean contrast on the same frozen empirical evidence.
#[derive(Clone, Debug)]
pub struct StatisticalContrast {
    /// Difference of plug-in means.
    pub estimate: f64,
    /// Pointwise percentile interval, when available.
    pub interval: Option<(f64, f64)>,
    /// Requested nominal coverage, not an execution-specific calibration claim.
    pub coverage_target: Option<f64>,
    /// Shared successful replicate count.
    pub replicates_ok: u32,
    /// Union of failed replicate counts.
    pub replicates_failed: u32,
    /// Why the interval was withheld.
    pub reason: Option<&'static str>,
}

/// Metadata-only inspection of statistical prepared authority.
#[derive(Clone, Debug)]
pub struct StatisticalStudyInspection {
    /// Identity layers; inspect does not fit or fetch.
    pub identities: ExactStudyIdentities,
    /// Provider-bound formula.
    pub formula: String,
    /// Required population/regime leaves.
    pub factors: Vec<ExactFactorRequirement>,
    /// Supplied or estimated snapshot bindings.
    pub bindings: Vec<StatisticalBindingView>,
    /// Four typed reasoning slots.
    pub reasoning: ReasoningView,
    /// Inspect token.
    pub theorem_scope: &'static str,
    /// Classical family.
    pub classical_scope: TheoremScope,
    /// Bounded catalog search.
    pub catalog_scope: TheoremScope,
}

/// One inspectable provider binding.
#[derive(Clone, Debug)]
pub struct StatisticalBindingView {
    /// Population.
    pub population: String,
    /// Regime raw id.
    pub regime: u32,
    /// Snapshot identity.
    pub snapshot: String,
    /// `supplied_exact` or `empirical_plugin`.
    pub origin: &'static str,
}

// Canonicalize only new multi-source/grid provider collections; historical scalar
// statistical-v2 identities retain their original ordering.
pub(super) fn canonicalize_input(input: &mut StatisticalTransportInput) -> Result<(), IoError> {
    input.samples.sort_by_key(|s| {
        (
            s.population.clone(),
            s.regime.raw(),
            s.snapshot_identity.clone(),
            antecedent_estimate::empirical_table::sample_key(s),
        )
    });
    let mut supplied = input
        .supplied
        .drain(..)
        .map(|law| Ok((antecedent_io::to_cbor(&ExactLawWire::from_law(&law))?, law)))
        .collect::<Result<Vec<_>, IoError>>()?;
    supplied.sort_by(|a, b| a.0.cmp(&b.0));
    input.supplied = supplied.into_iter().map(|(_, law)| law).collect();
    Ok(())
}

impl StudyBuilder {
    /// Prepare mixed supplied-law / empirical-table transport on the common handle.
    ///
    /// # Errors
    /// Missing providers, unlicensed estimators, incompatible catalogs, or budgets.
    pub fn statistical_transport(
        diagram: SelectionDiagram,
        functional: BoundTransportFunctional,
        input: StatisticalTransportInput,
        request: Assignment,
        limits: ExactEvaluationLimits,
        options: EmpiricalTableOptions,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<StatisticalPreparedState>, IoError> {
        antecedent_identify::verify_classical_transport(
            &diagram,
            functional.derivation().query(),
            functional.derivation(),
            SidLimits { steps: limits.operations, depth: limits.depth },
            ctx,
        )
        .map_err(err)?;
        PreparedStudy::<StatisticalPreparedState>::build(
            diagram, functional, input, request, limits, options, ctx,
        )
    }
}

impl PreparedStudy<StatisticalPreparedState> {
    fn build(
        diagram: SelectionDiagram,
        functional: BoundTransportFunctional,
        input: StatisticalTransportInput,
        request: Assignment,
        limits: ExactEvaluationLimits,
        options: EmpiricalTableOptions,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let mut input = input;
        if !functional.derivation().sources().is_empty() {
            canonicalize_input(&mut input)?;
        }
        antecedent_estimate::statistical_transport::check_statistical_resources(
            &input,
            &functional,
            &options,
            ctx,
        )
        .map_err(err)?;
        let point_options = point_options(options);
        let data = assemble_point_laws(&input, &functional, &point_options, ctx).map_err(err)?;
        let samples =
            input.samples.iter().map(SampleSummary::from_sample).collect::<Result<Vec<_>, _>>()?;
        Self::build_fitted(
            diagram, functional, input, data, samples, request, limits, options, ctx, false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_fitted(
        diagram: SelectionDiagram,
        functional: BoundTransportFunctional,
        input: StatisticalTransportInput,
        data: ExactTransportData,
        samples: Vec<SampleSummary>,
        request: Assignment,
        limits: ExactEvaluationLimits,
        options: EmpiricalTableOptions,
        ctx: &ExecutionContext,
        loaded_only: bool,
    ) -> Result<Self, IoError> {
        antecedent_estimate::empirical_table::validate_options(&options).map_err(err)?;
        // Coverage, assignments, binding snapshots and allocation limits are checked at preparation.
        antecedent_estimate::prepare_exact_transport(
            &functional,
            data.clone(),
            request.clone(),
            limits,
            ctx,
        )
        .map_err(err)?;
        let proof = TransportProofWire::from_checked(functional.derivation())?;
        let catalog = functional.catalog().canonicalized().map_err(err)?;
        let mut contract = EvidenceCatalogWire::from_catalog(&catalog);
        contract.bindings.clear();
        let shape = statistical_shape(&data)?;
        let target = digest(
            IdentityDomain::Target,
            &(
                &proof.proof.outcomes,
                &proof.proof.treatments,
                &proof.proof.target,
                statistical_assignments(&request),
            ),
        )?;
        let observation = digest(IdentityDomain::Observation, &(&contract, &shape))?;
        let dependence: Vec<_> = catalog
            .bindings
            .iter()
            .map(|b| {
                (
                    b.regime.raw(),
                    b.sampling.as_str(),
                    b.dependence.as_str(),
                    b.weights.as_ref().map(|v| v.snapshot_identity.as_ref()),
                )
            })
            .collect();
        let inference_binding = digest(
            IdentityDomain::InferenceBinding,
            &(
                options.estimator.identity(),
                options.bootstrap_replicates,
                options.coverage_level.to_bits(),
                options.max_joint_cells,
                "percentile_bootstrap",
                "pointwise",
                &dependence,
                ctx.rng.master_seed(),
                limits.operations,
                limits.depth,
            ),
        )?;
        let aliases: Vec<_> = catalog
            .bindings
            .iter()
            .filter_map(|b| b.dataset_identity.as_ref().map(|id| (b.regime.raw(), id.as_ref())))
            .collect();
        let inference_binding = if aliases.is_empty() {
            inference_binding
        } else {
            digest(
                IdentityDomain::InferenceBinding,
                &(&inference_binding, "shared_dataset_aliases_v1", aliases),
            )?
        };
        let identification = digest(
            IdentityDomain::Identification,
            &(
                &proof.proof.graph_signature,
                &proof.proof.outcomes,
                &proof.proof.treatments,
                &proof.proof.source,
                &proof.proof.target,
                &proof.proof.evidence_setting,
                &observation,
            ),
        )?;
        let identification = if proof.proof.sources.is_empty() {
            identification
        } else {
            digest(IdentityDomain::Identification, &(&identification, &proof.proof.sources))?
        };
        let identification_product = digest(IdentityDomain::IdentificationProduct, &proof)?;
        let program = digest(
            IdentityDomain::Program,
            &(
                &identification,
                &identification_product,
                &target,
                antecedent_io::expr_wire::expr_arena_to_wire(functional.arena())?,
                functional.root().raw(),
            ),
        )?;
        let snapshot =
            digest(IdentityDomain::DataSnapshot, &(statistical_law_wires(&data)?, &samples))?;
        let execution =
            digest(IdentityDomain::Execution, &(&program, &snapshot, &inference_binding))?;
        Ok(Self {
            state: StatisticalPreparedState {
                diagram,
                functional,
                input,
                data,
                request,
                limits,
                options,
                identities: ExactStudyIdentities {
                    target,
                    observation,
                    inference_binding,
                    identification,
                    identification_product,
                    program,
                    snapshot,
                    execution,
                },
                shape,
                seed: ctx.rng.master_seed(),
                samples,
                loaded_only,
                grid: Vec::new(),
            },
        })
    }

    fn with_grid(mut self, grid: Vec<Vec<(u32, ValueWire)>>) -> Result<Self, IoError> {
        if !grid.is_empty() {
            if !grid.contains(&statistical_assignments(&self.state.request)) {
                return Err(err("grid omits the retained target"));
            }
            self.state.identities.inference_binding = digest(
                IdentityDomain::InferenceBinding,
                &(&self.state.identities.inference_binding, &grid),
            )?;
            self.state.identities.execution = digest(
                IdentityDomain::Execution,
                &(
                    &self.state.identities.program,
                    &self.state.identities.snapshot,
                    &self.state.identities.inference_binding,
                ),
            )?;
        }
        self.state.grid = grid;
        Ok(self)
    }

    /// Frozen evidence contract.
    #[must_use]
    pub fn evidence_catalog(&self) -> &antecedent_core::EvidenceCatalog {
        self.state.functional.catalog()
    }
    /// Frozen selection diagram.
    #[must_use]
    pub fn diagram(&self) -> &SelectionDiagram {
        &self.state.diagram
    }
    /// Consumed derivation rules.
    #[must_use]
    pub fn rules(&self) -> Vec<&'static str> {
        self.state.functional.derivation().rules()
    }
    /// Frozen query.
    #[must_use]
    pub fn query(&self) -> &ClassicalTransportQuery {
        self.state.functional.derivation().query()
    }
    /// Frozen inference options.
    #[must_use]
    pub const fn options(&self) -> EmpiricalTableOptions {
        self.state.options
    }

    /// The estimator's own license rule: an earned percentile floor plus iid,
    /// unweighted, independent-study dependence for every estimated regime.
    fn licensed_interval(&self) -> bool {
        if is_bayesian_provider(self.state.options.estimator) {
            return false;
        }
        self.state.options.bootstrap_replicates >= PERCENTILE_95_MIN_REPLICATES
            && !self.state.input.samples.is_empty()
            && antecedent_estimate::licensed_iid_dependence(
                self.state.functional.catalog(),
                &self.state.input.samples,
            )
    }

    /// Drop percentile edges that cannot earn the options' nominal coverage label.
    fn withhold_unearned_percentile(
        estimate: &mut StatisticalTransportEstimate,
        options: &EmpiricalTableOptions,
    ) {
        let ok = estimate.uncertainty.as_ref().map_or(0, |row| row.replicates_ok);
        let requested = options.bootstrap_replicates;
        if requested == 0
            || (ok >= PERCENTILE_95_MIN_REPLICATES && requested >= PERCENTILE_95_MIN_REPLICATES)
        {
            return;
        }
        if estimate.atom_intervals.is_none()
            && estimate.mean_intervals.is_none()
            && estimate.uncertainty_reason.is_some()
        {
            return;
        }
        estimate.atom_intervals = None;
        estimate.mean_intervals = None;
        if estimate.uncertainty_reason.is_none() {
            estimate.uncertainty_reason = Some(Arc::from("insufficient_bootstrap_replicates"));
        }
    }

    /// Frozen bootstrap seed, independent of later caller contexts.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.state.seed
    }

    /// Metadata-only inspection: no refetch or refit.
    #[must_use]
    pub fn inspect(&self) -> StatisticalStudyInspection {
        StatisticalStudyInspection {
            identities: self.state.identities.clone(),
            formula: self.state.functional.arena().pretty(self.state.functional.root()),
            factors: statistical_requirements(&self.state.functional),
            bindings: statistical_bindings(&self.state.input, &self.state.data),
            reasoning: self.reasoning(false, None, None),
            theorem_scope: if matches!(
                self.state.options.estimator,
                antecedent_estimate::EmpiricalTableEstimator::Learned(_)
            ) {
                "checked_transport; learned_categorical_plugin; uncalibrated"
            } else if is_bayesian_provider(self.state.options.estimator) {
                "checked_transport; finite_discrete_bayesian_posterior; estimator_grid_not_measured"
            } else if self.state.functional.derivation().sources().is_empty() {
                TheoremScope::statistical_table_inspect_label()
            } else {
                "classical_meta_all_source_experiments_v1; empirical_table_plugin_iid"
            },
            classical_scope: self.state.functional.derivation().theorem_scope(),
            catalog_scope: TheoremScope::finite_catalog_search(),
        }
    }

    fn reasoning(
        &self,
        evaluated: bool,
        estimate: Option<&StatisticalTransportEstimate>,
        posterior: Option<&antecedent_estimate::BayesianStatisticalTransportEstimate>,
    ) -> ReasoningView {
        let uncertainty = if is_bayesian_provider(self.state.options.estimator) {
            bayesian_uncertainty_slot(
                self.state.options.posterior_draws,
                antecedent_estimate::dependence_refusal(
                    self.state.functional.catalog(),
                    &self.state.input.samples,
                ),
                posterior,
            )
        } else {
            match estimate {
                Some(est) if est.uncertainty.is_some() && est.uncertainty_reason.is_none() => {
                    SlotAvailability::Available(UncertaintySlot::new([UncertaintyComponent::new(
                        UncertaintySource::Sampling,
                        "percentile_bootstrap",
                        false,
                    )]))
                }
                Some(est) => SlotAvailability::unavailable(
                    est.uncertainty_reason.as_deref().unwrap_or("uncertainty_unavailable"),
                ),
                None if self.licensed_interval() => {
                    SlotAvailability::Available(UncertaintySlot::new([UncertaintyComponent::new(
                        UncertaintySource::Sampling,
                        "percentile_bootstrap",
                        false,
                    )]))
                }
                None if self.state.input.samples.is_empty() => {
                    SlotAvailability::unavailable("exact_supplied_law_no_sampling_uncertainty")
                }
                None if self.state.options.bootstrap_replicates == 0 => {
                    SlotAvailability::unavailable("bootstrap_not_requested")
                }
                None if self.state.options.bootstrap_replicates < PERCENTILE_95_MIN_REPLICATES => {
                    SlotAvailability::unavailable("insufficient_bootstrap_replicates")
                }
                None => SlotAvailability::unavailable(
                    antecedent_estimate::dependence_refusal(
                        self.state.functional.catalog(),
                        &self.state.input.samples,
                    )
                    .unwrap_or("transport.unsupported_dependence"),
                ),
            }
        };
        ReasoningView::new(
            SlotAvailability::Available(IdentificationSlot::identified_singleton(
                IdentificationStatus::NonparametricallyIdentified,
            )),
            SlotAvailability::Available(SupportSlot::new(
                "stage_contract",
                Some(Arc::from(if is_bayesian_provider(self.state.options.estimator) {
                    self.state.options.estimator.as_str()
                } else if matches!(
                    self.state.options.estimator,
                    antecedent_estimate::EmpiricalTableEstimator::Learned(_)
                ) {
                    "transport.learned_categorical"
                } else {
                    "transport.empirical_table"
                })),
                if evaluated {
                    SlotAvailability::Available(Arc::from("empirical_factor_support_checked"))
                } else {
                    SlotAvailability::unavailable("execution_specific")
                },
            )),
            uncertainty,
            SlotAvailability::Available(AssumptionSlot::new(vec![ObligationRecord::new(
                "transport.selection_diagram",
                ObligationScope::Program,
                AssumptionSource::UserDeclared,
                ObligationKind::Uncheckable,
                AssumptionStatus::Untestable,
                if is_bayesian_provider(self.state.options.estimator) {
                    match self.state.options.estimator {
                        antecedent_estimate::EmpiricalTableEstimator::EmpiricalSupportBayesianBootstrap =>
                            "Declared finite joint; Rubin Exp(1) row weights define a posterior on observed support only. The equal-tail interval is pointwise and its coverage reason is estimator_grid_not_measured.",
                        _ =>
                            "Declared full categorical state space; each cell has a symmetric Dirichlet(1) prior. The equal-tail interval is pointwise and its coverage reason is estimator_grid_not_measured.",
                    }
                } else if matches!(
                    self.state.options.estimator,
                    antecedent_estimate::EmpiricalTableEstimator::Learned(_)
                ) {
                    "Accepted graph and selections; coherent categorical chain model with declared ordering. Model-based cell predictions can extrapolate beyond observed cells; conditioning support is checked against empirical counts. Pointwise bootstrap is nominal and uncalibrated; no double-robustness claim."
                } else {
                    "Accepted causal graph, mechanism selections, and empirical-table regularity describe the declared populations."
                },
            )])),
        )
    }

    /// Execute the frozen request identity.
    ///
    /// # Errors
    /// Stale execution identity, support failure, cancellation, or resource exhaustion.
    pub fn estimate_checked(
        &self,
        execution: &str,
        ctx: &ExecutionContext,
    ) -> Result<StatisticalStudyResult, IoError> {
        if execution != self.state.identities.execution {
            return Err(err("transport.stale_request"));
        }
        self.estimate_retained(ctx)
    }

    fn bayesian_estimates(
        &self,
        ctx: &ExecutionContext,
        requests: &[Assignment],
    ) -> Result<Option<Vec<antecedent_estimate::BayesianStatisticalTransportEstimate>>, IoError>
    {
        let Some(provider) = bayesian_provider(self.state.options.estimator) else {
            return Ok(None);
        };
        let mut estimates = antecedent_estimate::evaluate_bayesian_statistical_transport_grid(
            &self.state.functional,
            &self.state.input,
            requests,
            self.state.limits,
            antecedent_estimate::statistical_transport::BayesianStatisticalTransportOptions {
                provider,
                draws: self.state.options.posterior_draws,
                coverage_level: self.state.options.coverage_level,
                max_joint_cells: self.state.options.max_joint_cells,
            },
            ctx,
        )
        .map_err(err)?;
        for estimate in &mut estimates {
            apply_posterior_draw_floor(estimate);
        }
        Ok(Some(estimates))
    }

    /// Estimate using retained providers.
    ///
    /// # Errors
    /// Located support or numerical failure, cancellation, or exhausted budget.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<StatisticalStudyResult, IoError> {
        self.estimate_retained(ctx)
    }

    /// Execute without re-identification.
    ///
    /// # Errors
    /// Support failure, cancellation, or resource exhaustion.
    pub fn estimate_retained(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<StatisticalStudyResult, IoError> {
        if self.state.loaded_only {
            return Err(err(
                "transport.samples_not_embedded: refresh with source samples before re-estimating",
            ));
        }
        let mut frozen_ctx = ctx.clone();
        frozen_ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let point_options = point_options(self.state.options);
        let requests = if self.state.grid.is_empty() {
            vec![self.state.request.clone()]
        } else {
            self.state
                .grid
                .iter()
                .map(|request| {
                    Assignment::from_pairs(
                        request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
                    )
                })
                .collect()
        };
        let index = if self.state.grid.is_empty() {
            0
        } else {
            self.state
                .grid
                .iter()
                .position(|request| *request == statistical_assignments(&self.state.request))
                .ok_or_else(|| err("retained target missing from grid"))?
        };
        let mut estimate = antecedent_estimate::statistical_transport::evaluate_statistical_transport_grid_with_point_laws(
            &self.state.functional,
            &self.state.input,
            &requests,
            self.state.limits,
            &point_options,
            &frozen_ctx,
            &self.state.data,
        )
        .map_err(err)?
        .remove(index);
        Self::withhold_unearned_percentile(&mut estimate, &self.state.options);
        let bayesian_estimate =
            self.bayesian_estimates(&frozen_ctx, &requests)?.map(|mut rows| rows.remove(index));
        Ok(StatisticalStudyResult {
            reasoning: self.reasoning(true, Some(&estimate), bayesian_estimate.as_ref()),
            estimate,
            bayesian_estimate,
            identities: self.state.identities.clone(),
        })
    }

    /// Evaluate a treatment grid with shared outer replicates and retained authority per point.
    /// # Errors
    /// Invalid request, absent source samples, or a failed joint execution.
    pub fn estimate_grid(
        &self,
        requests: &[Assignment],
        ctx: &ExecutionContext,
    ) -> Result<Vec<(Self, StatisticalStudyResult)>, IoError> {
        self.estimate_grid_with(requests, None, ctx)
    }

    /// [`Self::estimate_grid`] for a caller that already evaluated each request's plan on this
    /// study's retained point laws: `evaluated[i]` is the exact evaluation of `requests[i]`,
    /// used as that point's estimate instead of evaluating it again.
    /// # Errors
    /// Misaligned evaluations, absent source samples, or a failed joint execution.
    pub(super) fn estimate_grid_evaluated(
        &self,
        requests: &[Assignment],
        evaluated: Vec<antecedent_expr::ExactDistribution>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(Self, StatisticalStudyResult)>, IoError> {
        self.estimate_grid_with(requests, Some(evaluated), ctx)
    }

    fn estimate_grid_with(
        &self,
        requests: &[Assignment],
        evaluated: Option<Vec<antecedent_expr::ExactDistribution>>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(Self, StatisticalStudyResult)>, IoError> {
        if self.state.loaded_only {
            return Err(err("transport.samples_not_embedded"));
        }
        let mut frozen_ctx = ctx.clone();
        frozen_ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let point_options = point_options(self.state.options);
        let estimates = match evaluated {
            Some(distributions) => antecedent_estimate::statistical_transport::evaluate_statistical_transport_grid_with_evaluated_point_laws(
                &self.state.functional,
                &self.state.input,
                requests,
                self.state.limits,
                &point_options,
                &frozen_ctx,
                &self.state.data,
                distributions,
            ),
            None => antecedent_estimate::statistical_transport::evaluate_statistical_transport_grid_with_point_laws(
                &self.state.functional,
                &self.state.input,
                requests,
                self.state.limits,
                &point_options,
                &frozen_ctx,
                &self.state.data,
            ),
        }
        .map_err(err)?;
        let posteriors = self.bayesian_estimates(&frozen_ctx, requests)?;
        requests
            .iter()
            .zip(estimates)
            .enumerate()
            .map(|(index, (request, mut estimate))| {
                let prepared = Self::build_fitted(
                    self.state.diagram.clone(),
                    self.state.functional.clone(),
                    self.state.input.clone(),
                    self.state.data.clone(),
                    self.state.samples.clone(),
                    request.clone(),
                    self.state.limits,
                    self.state.options,
                    &frozen_ctx,
                    false,
                )?
                .with_grid(requests.iter().map(statistical_assignments).collect())?;
                Self::withhold_unearned_percentile(&mut estimate, &prepared.state.options);
                let bayesian_estimate = posteriors.as_ref().map(|rows| rows[index].clone());
                let result = StatisticalStudyResult {
                    reasoning: prepared.reasoning(
                        true,
                        Some(&estimate),
                        bayesian_estimate.as_ref(),
                    ),
                    identities: prepared.state.identities.clone(),
                    estimate,
                    bayesian_estimate,
                };
                Ok((prepared, result))
            })
            .collect()
    }

    /// Preview using the common transformation/invalidation contract.
    ///
    /// # Errors
    /// Invalid retained identity.
    pub fn preview_transform(
        &self,
        intent: antecedent_core::TransformIntent,
    ) -> Result<antecedent_core::TransformationReport, IoError> {
        use antecedent_core::{IdentityRef, SemanticDigest, TransformIntent, TransformationReport};
        let ids = &self.state.identities;
        let inputs = [
            (IdentityDomain::Target, &ids.target),
            (IdentityDomain::Identification, &ids.identification),
            (IdentityDomain::IdentificationProduct, &ids.identification_product),
            (IdentityDomain::Observation, &ids.observation),
            (IdentityDomain::Program, &ids.program),
            (IdentityDomain::InferenceBinding, &ids.inference_binding),
            (IdentityDomain::DataSnapshot, &ids.snapshot),
            (IdentityDomain::Execution, &ids.execution),
        ]
        .into_iter()
        .map(|(domain, value)| {
            Ok(IdentityRef::new(
                domain,
                SemanticDigest::from_bytes(antecedent_io::external_estimate::parse_digest_hex(
                    value,
                )?),
            ))
        })
        .collect::<Result<Vec<_>, IoError>>()?;
        let report = TransformationReport::new(
            intent,
            inputs,
            antecedent_core::intent_effects(intent).iter().cloned(),
            Vec::new(),
        );
        Ok(
            if matches!(
                intent,
                TransformIntent::DisplayPrecision
                    | TransformIntent::FilterDisplay
                    | TransformIntent::CompatibleDataReplace
            ) {
                report
            } else {
                report.refused_on_handle(
                    "transport.reprepare_required: statistical structural or execution contract changed",
                )
            },
        )
    }

    /// Whether a replacement keeps the observation shape.
    ///
    /// # Errors
    /// Identity encoding failure.
    pub fn preview_snapshot(&self, input: &StatisticalTransportInput) -> Result<bool, IoError> {
        let mut laws: Vec<_> = input.supplied.iter().map(ExactLawWire::metadata).collect();
        for sample in &input.samples {
            laws.push(ExactLawWire {
                empirical_counts: None,
                population: sample.population.to_string(),
                regime: sample.regime.raw(),
                interventions: sample
                    .interventions
                    .iter()
                    .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                    .collect(),
                axes: antecedent_estimate::catalog_axes(self.evidence_catalog(), sample)
                    .map_err(err)?
                    .iter()
                    .map(|axis| {
                        (
                            axis.variable.raw(),
                            axis.values.iter().map(ValueWire::from_value).collect(),
                        )
                    })
                    .collect(),
                probabilities: Vec::new(),
                snapshot: String::new(),
                origin: if matches!(
                    self.state.options.estimator,
                    antecedent_estimate::EmpiricalTableEstimator::Learned(_)
                ) {
                    "learned_plugin"
                } else {
                    "empirical_plugin"
                }
                .into(),
                absolute_tolerance: 0.0,
                relative_tolerance: 0.0,
            });
        }
        Ok(statistical_shape_wires(laws)? == self.state.shape)
    }

    fn replacement(
        &self,
        input: StatisticalTransportInput,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if !self.preview_snapshot(&input)? {
            return Err(err("transport.reprepare_required"));
        }
        let data = assemble_point_laws(
            &input,
            &self.state.functional,
            &point_options(self.state.options),
            ctx,
        )
        .map_err(err)?;
        let catalog = rebind_snapshots(self.state.functional.catalog(), |regime| {
            data.laws()
                .iter()
                .filter(|law| law.regime() == regime)
                .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity)
                .collect()
        })?;
        let functional = self.state.functional.derivation().bind_catalog(&catalog).map_err(err)?;
        let mut frozen_ctx = ctx.clone();
        frozen_ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        Self::build(
            self.state.diagram.clone(),
            functional,
            input,
            self.state.request.clone(),
            self.state.limits,
            self.state.options,
            &frozen_ctx,
        )?
        .with_grid(self.state.grid.clone())
    }

    /// Replace compatible snapshots and invalidate execution claims.
    ///
    /// # Errors
    /// Changed evidence shape, missing providers, or exceeded budget.
    pub fn replace_snapshot(
        &mut self,
        input: StatisticalTransportInput,
        ctx: &ExecutionContext,
    ) -> Result<(), IoError> {
        *self = self.replacement(input, ctx)?;
        Ok(())
    }

    /// Atomic refresh: execute a candidate before publishing it.
    ///
    /// # Errors
    /// Failed preparation/evaluation leaves the previous valid state intact.
    pub fn refresh(
        &mut self,
        input: StatisticalTransportInput,
        ctx: &ExecutionContext,
    ) -> Result<StatisticalStudyResult, IoError> {
        let candidate = self.replacement(input, ctx)?;
        let result = candidate.estimate_retained(ctx)?;
        *self = candidate;
        Ok(result)
    }

    /// Export checked proof, fitted joints, identities, uncertainty and reasoning.
    ///
    /// # Errors
    /// Stale result or encoding failure.
    pub fn export(&self, result: &StatisticalStudyResult) -> Result<Vec<u8>, IoError> {
        if result.identities != self.state.identities {
            return Err(err("transport.stale_result"));
        }
        let graph = self.state.diagram.causal_graph();
        let edges = antecedent_io::admg_to_wire(graph)?;
        let nodes = graph
            .nodes()
            .iter()
            .map(|node| match node {
                NodeRef::Static(v) => Ok(v.raw()),
                _ => Err(err("transport requires static coordinates")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        antecedent_io::to_cbor(&StatisticalExecutionWire {
            format: "antecedent.statistical_transport.v2".into(),
            seed: self.state.seed,
            grid: self.state.grid.clone(),
            samples: self.state.samples.clone(),
            nodes,
            directed: edges.directed,
            bidirected: edges.bidirected,
            selections: self.state.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            proof: TransportProofWire::from_checked(self.state.functional.derivation())?,
            catalog: EvidenceCatalogWire::from_catalog(self.state.functional.catalog()),
            laws: statistical_law_wires(&self.state.data)?,
            request: statistical_assignments(&self.state.request),
            operations: self.state.limits.operations,
            depth: self.state.limits.depth,
            options: StatisticalOptionsWire::from_options(&self.state.options),
            identities: self.state.identities.clone(),
            atoms: result
                .distribution()
                .atoms
                .iter()
                .map(|row| row.iter().map(ValueWire::from_value).collect())
                .collect(),
            probabilities: result.distribution().probabilities.to_vec(),
            uncertainty: result.estimate.uncertainty.as_ref().map(UncertaintyRowWire::from_row),
            bayesian_posterior: result.bayesian_estimate.as_ref().map(|posterior| {
                BayesianTransportPosteriorWire {
                    estimator: posterior.estimator.to_string(),
                    interval_method: posterior.interval_method.to_string(),
                    draws_requested: posterior.draws_requested,
                    draws_ok: posterior.draws_ok,
                    draws_failed: posterior.draws_failed,
                    interval_reason: posterior.interval_reason.as_ref().map(ToString::to_string),
                    probabilities: posterior
                        .distributions
                        .iter()
                        .map(|distribution| distribution.probabilities.to_vec())
                        .collect(),
                    atom_intervals: posterior.atom_intervals.to_vec(),
                    mean_intervals: posterior
                        .mean_intervals
                        .iter()
                        .map(|(variable, lower, upper)| (variable.raw(), *lower, *upper))
                        .collect(),
                }
            }),
            atom_intervals: result.estimate.atom_intervals.as_ref().map(|rows| rows.to_vec()),
            mean_intervals: result
                .estimate
                .mean_intervals
                .as_ref()
                .map(|rows| rows.iter().map(|(v, lo, hi)| (v.raw(), *lo, *hi)).collect()),
            replicate_ids: result.estimate.replicate_ids.as_ref().map(|ids| ids.to_vec()),
            uncertainty_reason: result
                .estimate
                .uncertainty_reason
                .as_ref()
                .map(ToString::to_string),
            atom_replicates: result
                .estimate
                .atom_replicates
                .as_ref()
                .map(|rows| rows.iter().map(|row| row.to_vec()).collect()),
            reasoning: super::contract::reasoning_section(&result.reasoning),
        })
    }

    /// Independently consume a statistical artifact. Recomputes the point from
    /// embedded fitted joints and checks identities; does not re-bootstrap.
    ///
    /// # Errors
    /// Invalid proof, substituted binding, changed claims, or resource exhaustion.
    pub fn consume(
        bytes: &[u8],
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, StatisticalStudyResult), IoError> {
        if ctx.cancellation.is_cancelled()
            || ctx.memory.hard_limit_bytes.is_some_and(|limit| bytes.len() as u64 > limit)
        {
            return Err(err("transport artifact budget/cancellation"));
        }
        let wire: StatisticalExecutionWire = antecedent_io::from_cbor(bytes)?;
        if wire.format != "antecedent.statistical_transport.v2"
            || wire.operations > limits.operations
            || wire.depth > limits.depth
        {
            return Err(err("transport artifact format/limits"));
        }
        let (diagram, proof) = rebuild_checked_proof(
            &GraphFields {
                nodes: &wire.nodes,
                directed: &wire.directed,
                bidirected: &wire.bidirected,
                selections: &wire.selections,
            },
            &wire.proof,
            limits,
            ctx,
        )?;
        let functional = proof.bind_catalog(&wire.catalog.to_catalog()?).map_err(err)?;
        let laws = wire.laws.iter().map(ExactLawWire::to_law).collect::<Result<Vec<_>, _>>()?;
        let input = StatisticalTransportInput { supplied: laws, samples: Vec::new() };
        let request = Assignment::from_pairs(
            wire.request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
        );
        if request.entries().len() != wire.request.len() {
            return Err(err("duplicate request coordinate"));
        }
        let options = wire.options.to_options()?;
        let data = ExactTransportData::try_new(input.supplied.clone(), options.max_joint_cells)
            .map_err(err)?;
        validate_sample_summaries(&wire.samples, &data, functional.catalog())?;
        let mut frozen_ctx = ctx.clone();
        frozen_ctx.rng = antecedent_core::RngFactory::from_seed(wire.seed);
        let prepared = Self::build_fitted(
            diagram,
            functional,
            input,
            data,
            wire.samples.clone(),
            request,
            ExactEvaluationLimits { operations: wire.operations, depth: wire.depth },
            options,
            &frozen_ctx,
            true,
        )?
        .with_grid(wire.grid.clone())?;
        if prepared.state.identities != wire.identities {
            return Err(err("transport artifact identity mismatch"));
        }
        let point = antecedent_estimate::evaluate_exact_transport(
            &prepared.state.functional,
            prepared.state.data.clone(),
            prepared.state.request.clone(),
            prepared.state.limits,
            ctx,
        )
        .map_err(err)?;
        let atoms: Vec<Vec<_>> =
            point.atoms.iter().map(|row| row.iter().map(ValueWire::from_value).collect()).collect();
        if atoms != wire.atoms || point.probabilities.as_ref() != wire.probabilities.as_slice() {
            return Err(err("transport artifact execution claim mismatch"));
        }
        let estimate = validate_uncertainty(&wire, point.clone(), &prepared)?;
        let bayesian_estimate = validate_bayesian_posterior(
            wire.bayesian_posterior.as_ref(),
            point,
            wire.options.estimator.as_str(),
            wire.options.posterior_draws,
            wire.options.coverage_level,
        )?;
        let reasoning = prepared.reasoning(true, Some(&estimate), bayesian_estimate.as_ref());
        if super::contract::reasoning_section(&reasoning) != wire.reasoning {
            return Err(err("transport artifact reasoning mismatch"));
        }
        let result = StatisticalStudyResult {
            reasoning,
            estimate,
            identities: wire.identities,
            bayesian_estimate,
        };
        Ok((prepared, result))
    }
}

fn statistical_requirements(functional: &BoundTransportFunctional) -> Vec<ExactFactorRequirement> {
    use antecedent_expr::ExprNode;
    let arena = functional.arena();
    let mut pending = vec![functional.root()];
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                population,
                regime,
                ..
            } => out.push(ExactFactorRequirement {
                expression: id,
                binding: antecedent_expr::LeafBinding {
                    population: Arc::from(arena.population(*population)),
                    regime: *regime,
                },
                variables: arena.var_set(*variables).to_vec(),
                conditioned_on: arena.var_set(*conditioned_on).to_vec(),
                interventions: arena.intervention_set(*intervention),
            }),
            ExprNode::Kernel { body, .. } => pending.push(*body),
            ExprNode::Product(list) => pending.extend(arena.list(*list)),
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                pending.push(*expr)
            }
            ExprNode::Ratio { numerator, denominator } => {
                pending.extend([*numerator, *denominator]);
            }
            ExprNode::Expectation { distribution, .. } => pending.push(*distribution),
            ExprNode::Contrast { left, right, .. } => pending.extend([*left, *right]),
        }
    }
    out.sort_by_key(|factor| factor.expression.raw());
    out
}

fn statistical_bindings(
    _input: &StatisticalTransportInput,
    data: &ExactTransportData,
) -> Vec<StatisticalBindingView> {
    data.laws()
        .iter()
        .map(|law| StatisticalBindingView {
            population: law.population().into(),
            regime: law.regime().raw(),
            snapshot: law.snapshot_identity().into(),
            origin: law.origin().as_str(),
        })
        .collect()
}

fn statistical_law_wires(data: &ExactTransportData) -> Result<Vec<ExactLawWire>, IoError> {
    let mut laws = data
        .laws()
        .iter()
        .map(|law| {
            let wire = ExactLawWire::from_law(law);
            Ok((
                antecedent_io::to_cbor(&(
                    wire.population.clone(),
                    wire.regime,
                    &wire.interventions,
                ))?,
                wire,
            ))
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    laws.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(laws.into_iter().map(|(_, law)| law).collect())
}

fn statistical_shape(data: &ExactTransportData) -> Result<String, IoError> {
    statistical_shape_wires(data.laws().iter().map(ExactLawWire::metadata).collect())
}

fn statistical_shape_wires(mut laws: Vec<ExactLawWire>) -> Result<String, IoError> {
    for law in &mut laws {
        law.probabilities.clear();
        law.snapshot.clear();
        law.axes.sort_by_key(|axis| axis.0);
        for (_, values) in &mut law.axes {
            values.sort_by(|a, b| {
                a.to_value()
                    .as_f64()
                    .partial_cmp(&b.to_value().as_f64())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        law.interventions.sort_by_key(|a| a.0);
        law.absolute_tolerance = 0.0;
        law.relative_tolerance = 0.0;
    }
    let mut canonical = laws.iter().map(antecedent_io::to_cbor).collect::<Result<Vec<_>, _>>()?;
    canonical.sort();
    digest(IdentityDomain::Observation, &canonical)
}

fn statistical_assignments(request: &Assignment) -> Vec<(u32, ValueWire)> {
    let mut values: Vec<_> =
        request.entries().iter().map(|(v, x)| (v.raw(), ValueWire::from_value(x))).collect();
    values.sort_by_key(|(v, _)| *v);
    values
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct UncertaintyRowWire {
    estimator: String,
    method: String,
    coverage_target: f64,
    interval_scope: String,
    sample_sizes: Vec<(String, u32)>,
    support_regime: String,
    replicates_requested: u32,
    replicates_ok: u32,
    replicates_failed: u32,
    seed: u64,
    calibration_binding: Option<String>,
}

impl UncertaintyRowWire {
    fn from_row(row: &TransportUncertaintyRow) -> Self {
        Self {
            estimator: row.estimator.to_string(),
            method: row.method.to_string(),
            coverage_target: row.coverage_target,
            interval_scope: row.interval_scope.to_string(),
            sample_sizes: row.sample_sizes.iter().map(|(name, n)| (name.to_string(), *n)).collect(),
            support_regime: row.support_regime.to_string(),
            replicates_requested: row.replicates_requested,
            replicates_ok: row.replicates_ok,
            replicates_failed: row.replicates_failed,
            seed: row.seed,
            calibration_binding: row.calibration_binding.as_ref().map(ToString::to_string),
        }
    }
    fn to_row(&self) -> TransportUncertaintyRow {
        TransportUncertaintyRow {
            estimator: Arc::from(self.estimator.as_str()),
            method: Arc::from(self.method.as_str()),
            coverage_target: self.coverage_target,
            interval_scope: Arc::from(self.interval_scope.as_str()),
            sample_sizes: self
                .sample_sizes
                .iter()
                .map(|(name, n)| (Arc::from(name.as_str()), *n))
                .collect(),
            support_regime: Arc::from(self.support_regime.as_str()),
            replicates_requested: self.replicates_requested,
            replicates_ok: self.replicates_ok,
            replicates_failed: self.replicates_failed,
            seed: self.seed,
            calibration_binding: self.calibration_binding.as_deref().map(Arc::from),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StatisticalExecutionWire {
    format: String,
    seed: u64,
    grid: Vec<Vec<(u32, ValueWire)>>,
    samples: Vec<SampleSummary>,
    nodes: Vec<u32>,
    directed: Vec<(u32, u32)>,
    bidirected: Vec<(u32, u32)>,
    selections: Vec<u32>,
    proof: TransportProofWire,
    catalog: EvidenceCatalogWire,
    laws: Vec<ExactLawWire>,
    request: Vec<(u32, ValueWire)>,
    operations: usize,
    depth: usize,
    options: StatisticalOptionsWire,
    identities: ExactStudyIdentities,
    atoms: Vec<Vec<ValueWire>>,
    probabilities: Vec<f64>,
    uncertainty: Option<UncertaintyRowWire>,
    #[serde(default)]
    bayesian_posterior: Option<BayesianTransportPosteriorWire>,
    atom_intervals: Option<Vec<(f64, f64)>>,
    mean_intervals: Option<Vec<(u32, f64, f64)>>,
    replicate_ids: Option<Vec<u32>>,
    uncertainty_reason: Option<String>,
    atom_replicates: Option<Vec<Vec<f64>>>,
    reasoning: antecedent_io::ReasoningSectionWire,
}

pub(super) fn validate_sample_summaries(
    samples: &[SampleSummary],
    data: &ExactTransportData,
    catalog: &antecedent_core::EvidenceCatalog,
) -> Result<(), IoError> {
    let mut seen = std::collections::BTreeSet::new();
    for sample in samples {
        antecedent_io::external_estimate::parse_digest_hex(&sample.content_digest)?;
        let key =
            antecedent_io::to_cbor(&(&sample.population, sample.regime, &sample.interventions))?;
        if sample.n == 0 || !seen.insert(key) {
            return Err(err("invalid empirical sample summary"));
        }
        let law = data
            .laws()
            .iter()
            .find(|law| {
                let wire = ExactLawWire::metadata(law);
                wire.population == sample.population
                    && wire.regime == sample.regime
                    && wire.interventions == sample.interventions
            })
            .ok_or_else(|| err("sample summary has no fitted joint"))?;
        if law.origin() == antecedent_expr::LawOrigin::SuppliedExact
            || law.snapshot_identity() != sample.snapshot
        {
            return Err(err("sample summary origin/snapshot mismatch"));
        }
        if catalog.bindings.iter().any(|b| {
            b.regime.raw() == sample.regime && b.snapshot_identity.as_ref() != sample.snapshot
        }) {
            return Err(err("sample summary binding mismatch"));
        }
        if let Some(counts) = law.empirical_counts() {
            if counts.iter().sum::<u64>() != u64::from(sample.n) {
                return Err(err("learned support/sample size mismatch"));
            }
            continue;
        }
        for p in law.probabilities() {
            let count = p * f64::from(sample.n);
            if (count - count.round()).abs() > (8.0 * f64::EPSILON * f64::from(sample.n)).max(1e-7)
            {
                return Err(err("fitted joint is inconsistent with sample size"));
            }
        }
    }
    let mut aliases = std::collections::BTreeMap::new();
    for sample in samples {
        if let Some(identity) = catalog
            .bindings
            .iter()
            .find(|binding| binding.regime.raw() == sample.regime)
            .and_then(|binding| binding.dataset_identity.as_ref())
        {
            let key = antecedent_io::to_cbor(&(identity.as_ref(), &sample.interventions))?;
            if let Some(previous) = aliases.insert(key, sample) {
                if previous.n != sample.n || previous.content_digest != sample.content_digest {
                    return Err(err("conflicting forwarded dataset provenance"));
                }
            }
        }
    }
    let empirical = data
        .laws()
        .iter()
        .filter(|law| law.origin() != antecedent_expr::LawOrigin::SuppliedExact)
        .count();
    if empirical != samples.len() {
        return Err(err("fitted joints lack sample provenance"));
    }
    Ok(())
}

fn validate_uncertainty(
    wire: &StatisticalExecutionWire,
    point: antecedent_expr::ExactDistribution,
    prepared: &PreparedStudy<StatisticalPreparedState>,
) -> Result<StatisticalTransportEstimate, IoError> {
    use antecedent_estimate::percentile_interval;
    let bad = || err("transport artifact uncertainty mismatch");
    let mut estimate = StatisticalTransportEstimate {
        distribution: point,
        uncertainty: None,
        atom_intervals: None,
        mean_intervals: None,
        atom_replicates: None,
        mean_replicates: None,
        replicate_ids: None,
        uncertainty_reason: wire.uncertainty_reason.as_deref().map(Arc::from),
    };
    let Some(row) = &wire.uncertainty else {
        let expected = if wire.samples.is_empty() {
            "exact_supplied_law_no_sampling_uncertainty"
        } else if is_bayesian_provider(wire.options.to_options()?.estimator) {
            "bootstrap_not_requested"
        } else if wire.options.bootstrap_replicates == 0 {
            "bootstrap_not_requested"
        } else {
            "transport.unsupported_dependence"
        };
        if wire.atom_intervals.is_some()
            || wire.mean_intervals.is_some()
            || wire.atom_replicates.is_some()
            || wire.replicate_ids.is_some()
            || wire.uncertainty_reason.as_deref() != Some(expected)
            || prepared.licensed_interval()
        {
            return Err(bad());
        }
        return Ok(estimate);
    };
    let sizes: Vec<_> = wire.samples.iter().map(|s| (s.snapshot.clone(), s.n)).collect();
    if row.estimator != wire.options.estimator
        || row.method != "percentile_bootstrap"
        || row.interval_scope != "pointwise"
        || row.coverage_target.to_bits() != wire.options.coverage_level.to_bits()
        || row.replicates_requested != wire.options.bootstrap_replicates
        || row.seed != wire.seed
        || row.sample_sizes != sizes
        || row.calibration_binding.is_some()
        || row.support_regime != "empirical_complete_case_no_smoothing"
        || row.replicates_ok.checked_add(row.replicates_failed) != Some(row.replicates_requested)
        || wire.samples.is_empty()
        || row.replicates_requested == 0
    {
        return Err(bad());
    }
    if !antecedent_estimate::licensed_iid_regimes(
        prepared.evidence_catalog(),
        wire.samples.iter().map(|sample| antecedent_core::RegimeId::from_raw(sample.regime)),
    ) {
        return Err(bad());
    }
    let rows = wire.atom_replicates.as_ref().ok_or_else(bad)?;
    let ids = wire.replicate_ids.as_ref().ok_or_else(bad)?;
    if rows.len() != row.replicates_ok as usize
        || ids.len() != rows.len()
        || ids.windows(2).any(|pair| pair[0] >= pair[1])
        || ids.iter().any(|id| *id >= row.replicates_requested)
    {
        return Err(bad());
    }
    let width = estimate.distribution.atoms.len();
    if rows.len().checked_mul(width).is_none_or(|n| n > wire.operations) {
        return Err(err("transport artifact operation budget"));
    }
    let mut means: Vec<(VariableId, Vec<f64>)> =
        estimate.distribution.outcomes.iter().map(|v| (*v, Vec::new())).collect();
    for row in rows {
        if row.len() != width
            || row.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || (row.iter().sum::<f64>() - 1.0).abs() > 1e-9
        {
            return Err(bad());
        }
        let mut distribution = estimate.distribution.clone();
        distribution.probabilities = row.clone().into();
        for (v, values) in &mut means {
            values.push(distribution.mean(*v).map_err(err)?);
        }
    }
    // Two distinct reasons withhold the percentile edges: the estimator's own
    // failure-fraction refusal (too few successful replicates or too many failed
    // ones to trust the resample), and the facade's licensing floor (enough
    // replicates ran clean, but fewer than the nominal-0.95 percentile requires).
    // `withhold_unearned_percentile` never overwrites an existing failure-fraction
    // reason, so the two are mutually exclusive and must be told apart here too.
    let bootstrap_failed = row.replicates_ok < 2
        || f64::from(row.replicates_failed) / f64::from(row.replicates_requested) > 0.5;
    let unlicensed = !bootstrap_failed
        && (row.replicates_ok < PERCENTILE_95_MIN_REPLICATES
            || row.replicates_requested < PERCENTILE_95_MIN_REPLICATES);
    if bootstrap_failed {
        if wire.uncertainty_reason.as_deref() != Some("bootstrap_failure_fraction")
            || wire.atom_intervals.is_some()
            || wire.mean_intervals.is_some()
        {
            return Err(bad());
        }
    } else if unlicensed {
        if wire.uncertainty_reason.as_deref() != Some("insufficient_bootstrap_replicates")
            || wire.atom_intervals.is_some()
            || wire.mean_intervals.is_some()
        {
            return Err(bad());
        }
    } else {
        let atoms: Vec<_> = (0..width)
            .map(|i| {
                percentile_interval(
                    &rows.iter().map(|r| r[i]).collect::<Vec<_>>(),
                    row.coverage_target,
                )
            })
            .collect();
        let intervals: Vec<_> = means
            .iter()
            .map(|(v, reps)| {
                let (lo, hi) = percentile_interval(reps, row.coverage_target);
                (v.raw(), lo, hi)
            })
            .collect();
        if wire.uncertainty_reason.is_some()
            || wire.atom_intervals.as_ref() != Some(&atoms)
            || wire.mean_intervals.as_ref() != Some(&intervals)
        {
            return Err(bad());
        }
        estimate.atom_intervals = Some(atoms.into());
        estimate.mean_intervals = Some(
            intervals.into_iter().map(|(v, lo, hi)| (VariableId::from_raw(v), lo, hi)).collect(),
        );
    }
    estimate.uncertainty = Some(row.to_row());
    estimate.atom_replicates = Some(rows.iter().map(|row| Arc::from(row.clone())).collect());
    estimate.mean_replicates = Some(means.into_iter().map(|(v, row)| (v, row.into())).collect());
    estimate.replicate_ids = Some(ids.clone().into());
    Ok(estimate)
}

fn validate_bayesian_posterior(
    wire: Option<&BayesianTransportPosteriorWire>,
    point: antecedent_expr::ExactDistribution,
    estimator: &str,
    requested: u32,
    coverage: f64,
) -> Result<Option<antecedent_estimate::BayesianStatisticalTransportEstimate>, IoError> {
    let expected = matches!(
        estimator,
        antecedent_estimate::EMPIRICAL_SUPPORT_BAYESIAN_BOOTSTRAP
            | antecedent_estimate::STATE_SPACE_DIRICHLET
    );
    let Some(wire) = wire else {
        return if expected {
            Err(err("Bayesian transport artifact lacks posterior draws"))
        } else {
            Ok(None)
        };
    };
    if !expected
        || wire.estimator != estimator
        || wire.interval_method != antecedent_estimate::POSTERIOR_EQUAL_TAIL
        || wire.draws_requested != requested
        || wire.probabilities.len() != wire.draws_ok as usize
        || wire.probabilities.len().checked_mul(point.atoms.len()).is_none_or(|n| n > 10_000_000)
    {
        return Err(err("Bayesian transport artifact draw metadata mismatch"));
    }
    let reason = wire.interval_reason.as_deref();
    let dependence =
        matches!(reason, Some("transport.unsupported_dependence" | "no_estimated_regime"));
    let withheld = dependence
        || matches!(
            reason,
            Some("bootstrap_failure_fraction" | "insufficient_bootstrap_replicates")
        );
    if dependence {
        if wire.draws_ok != 0 || wire.draws_failed != 0 || !wire.probabilities.is_empty() {
            return Err(err("Bayesian transport artifact draw metadata mismatch"));
        }
    } else if wire.draws_ok.checked_add(wire.draws_failed) != Some(wire.draws_requested) {
        return Err(err("Bayesian transport artifact draw metadata mismatch"));
    }
    if withheld && (!wire.atom_intervals.is_empty() || !wire.mean_intervals.is_empty()) {
        return Err(err("Bayesian transport artifact posterior summary mismatch"));
    }
    let mut distributions = Vec::with_capacity(wire.probabilities.len());
    for probabilities in &wire.probabilities {
        let mass: f64 = probabilities.iter().sum();
        if probabilities.len() != point.atoms.len()
            || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
            || !mass.is_finite()
            || (mass - 1.0).abs() > 1e-10
        {
            return Err(err("Bayesian transport artifact has invalid draw law"));
        }
        distributions.push(antecedent_expr::ExactDistribution {
            outcomes: point.outcomes.clone(),
            atoms: point.atoms.clone(),
            probabilities: probabilities.clone().into(),
            support: point.support.clone(),
        });
    }
    if withheld {
        return Ok(Some(antecedent_estimate::BayesianStatisticalTransportEstimate {
            estimator: Arc::from(wire.estimator.as_str()),
            interval_method: Arc::from(wire.interval_method.as_str()),
            distributions: distributions.into(),
            atom_intervals: Arc::from([]),
            mean_intervals: Arc::from([]),
            interval_reason: reason.map(Arc::from),
            draws_requested: wire.draws_requested,
            draws_ok: wire.draws_ok,
            draws_failed: wire.draws_failed,
        }));
    }
    if wire.draws_ok < 2
        || (reason.is_some()
            && reason != Some(antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED))
    {
        return Err(err("Bayesian transport artifact draw metadata mismatch"));
    }
    let mut atom_intervals = Vec::with_capacity(point.atoms.len());
    for atom in 0..point.atoms.len() {
        let mut values =
            distributions.iter().map(|draw| draw.probabilities[atom]).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        atom_intervals.push(antecedent_stats::equal_tail_interval_sorted(
            &values,
            coverage,
            antecedent_stats::QuantileRule::Interpolated,
        ));
    }
    let mut mean_intervals = Vec::with_capacity(point.outcomes.len());
    for outcome in point.outcomes.iter().copied() {
        let mut values = distributions
            .iter()
            .map(|draw| draw.mean(outcome).map_err(err))
            .collect::<Result<Vec<_>, _>>()?;
        values.sort_by(f64::total_cmp);
        let (lower, upper) = antecedent_stats::equal_tail_interval_sorted(
            &values,
            coverage,
            antecedent_stats::QuantileRule::Interpolated,
        );
        mean_intervals.push((outcome, lower, upper));
    }
    if !same_intervals(&atom_intervals, &wire.atom_intervals)
        || mean_intervals.len() != wire.mean_intervals.len()
        || mean_intervals.iter().zip(&wire.mean_intervals).any(|((v, lo, hi), (wv, wlo, whi))| {
            v.raw() != *wv || lo.to_bits() != wlo.to_bits() || hi.to_bits() != whi.to_bits()
        })
    {
        return Err(err("Bayesian transport artifact posterior summary mismatch"));
    }
    Ok(Some(antecedent_estimate::BayesianStatisticalTransportEstimate {
        estimator: Arc::from(wire.estimator.as_str()),
        interval_method: Arc::from(wire.interval_method.as_str()),
        distributions: distributions.into(),
        atom_intervals: atom_intervals.into(),
        mean_intervals: mean_intervals.into(),
        draws_requested: wire.draws_requested,
        draws_ok: wire.draws_ok,
        draws_failed: wire.draws_failed,
        interval_reason: Some(Arc::from(
            reason.unwrap_or(antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED),
        )),
    }))
}

fn same_intervals(left: &[(f64, f64)], right: &[(f64, f64)]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_graph::{Admg, DenseNodeId};
    use std::collections::BTreeMap;

    use antecedent_core::{
        DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
        EvidenceRegime, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, TargetSampling, Value,
        VariableCoordinate, VariableDomain,
    };
    use antecedent_estimate::RegimeSample;

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    fn sample(snapshot: &str, counts: [usize; 4]) -> RegimeSample {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (xv, yv, n) in [
            (0.0, 0.0, counts[0]),
            (0.0, 1.0, counts[1]),
            (1.0, 0.0, counts[2]),
            (1.0, 1.0, counts[3]),
        ] {
            x.extend(std::iter::repeat_n(Some(xv), n));
            y.extend(std::iter::repeat_n(Some(yv), n));
        }
        RegimeSample {
            population: Arc::from("target"),
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from(snapshot),
            interventions: Arc::from([]),
            columns: BTreeMap::from([(v(0), x), (v(1), y)]),
        }
    }

    fn prepared(snapshot: &str, counts: [usize; 4]) -> PreparedStudy<StatisticalPreparedState> {
        prepared_with_replicates(snapshot, counts, PERCENTILE_95_MIN_REPLICATES)
    }

    fn prepared_with_replicates(
        snapshot: &str,
        counts: [usize; 4],
        bootstrap_replicates: u32,
    ) -> PreparedStudy<StatisticalPreparedState> {
        prepared_with_dependence(
            snapshot,
            counts,
            bootstrap_replicates,
            DependenceGroup::IndependentStudies,
        )
    }

    fn prepared_with_dependence(
        snapshot: &str,
        counts: [usize; 4],
        bootstrap_replicates: u32,
        dependence: DependenceGroup,
    ) -> PreparedStudy<StatisticalPreparedState> {
        prepared_with_options(
            snapshot,
            counts,
            dependence,
            EmpiricalTableOptions { bootstrap_replicates, ..EmpiricalTableOptions::default() },
        )
    }

    fn prepared_with_options(
        snapshot: &str,
        counts: [usize; 4],
        dependence: DependenceGroup,
        options: EmpiricalTableOptions,
    ) -> PreparedStudy<StatisticalPreparedState> {
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [v(1)]).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([v(1)]),
            treatments: Arc::from([v(0)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ctx = ExecutionContext::for_tests(7);
        let antecedent_identify::ClassicalTransportResult::Identified(proof) =
            antecedent_identify::identify_classical_transport(
                &diagram,
                &query,
                SidLimits::default(),
                &ctx,
            )
            .unwrap()
        else {
            panic!("identified");
        };
        let catalog = EvidenceCatalog::try_new(
            [Environment::try_new(
                "target",
                [
                    VariableCoordinate {
                        variable: v(0),
                        domain: VariableDomain::Binary,
                        unit: None,
                    },
                    VariableCoordinate {
                        variable: v(1),
                        domain: VariableDomain::Binary,
                        unit: None,
                    },
                ],
                [],
            )
            .unwrap()],
            [EvidenceRegime::try_new(
                RegimeId::from_raw(0),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                [v(0), v(1)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap()],
            [RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(0),
                snapshot_identity: Arc::from(snapshot),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence,
            }],
            Some(TargetSampling::RepresentativeSample),
        )
        .unwrap();
        StudyBuilder::statistical_transport(
            diagram,
            proof.bind_catalog(&catalog).unwrap(),
            StatisticalTransportInput {
                supplied: Vec::new(),
                samples: vec![sample(snapshot, counts)],
            },
            Assignment::from_pairs([(v(0), Value::Int64(1))]),
            ExactEvaluationLimits::default(),
            options,
            &ctx,
        )
        .unwrap()
    }

    #[test]
    fn licensed_interval_requires_earned_percentile_minimum() {
        for replicates in [0, 1, 2] {
            let study = prepared_with_replicates("one", [20, 5, 10, 15], replicates);
            assert!(
                !study.licensed_interval(),
                "B={replicates} cannot earn a nominal 0.95 percentile label"
            );
            let inspect = study.inspect();
            assert!(!inspect.reasoning.uncertainty.is_available());
            match &inspect.reasoning.uncertainty {
                SlotAvailability::Unavailable { reason } if replicates == 0 => {
                    assert_eq!(reason.as_ref(), "bootstrap_not_requested");
                }
                SlotAvailability::Unavailable { reason } => {
                    assert_eq!(reason.as_ref(), "insufficient_bootstrap_replicates");
                }
                SlotAvailability::Available(_) => panic!("uncertainty must stay withheld"),
                _ => panic!("unexpected uncertainty slot"),
            }
        }
        let below =
            prepared_with_replicates("one", [20, 5, 10, 15], PERCENTILE_95_MIN_REPLICATES - 1);
        assert!(!below.licensed_interval());
        let earned = prepared_with_replicates("one", [20, 5, 10, 15], PERCENTILE_95_MIN_REPLICATES);
        assert!(earned.licensed_interval());
        assert!(earned.inspect().reasoning.uncertainty.is_available());
    }

    #[test]
    fn bayesian_transport_posterior_survives_export_and_independent_consume() {
        let study = prepared_with_options(
            "bayesian",
            [20, 5, 10, 15],
            DependenceGroup::IndependentStudies,
            EmpiricalTableOptions {
                estimator: antecedent_estimate::EmpiricalTableEstimator::StateSpaceDirichlet,
                posterior_draws: 16,
                ..EmpiricalTableOptions::default()
            },
        );
        let ctx = ExecutionContext::for_tests(7);
        let result = study.estimate_retained(&ctx).unwrap();
        let posterior = result.bayesian_estimate().expect("Bayesian posterior attached");
        assert_eq!(posterior.draws_requested, 16);
        assert_eq!(posterior.draws_ok, 16);
        assert_eq!(posterior.estimator.as_ref(), antecedent_estimate::STATE_SPACE_DIRICHLET);
        assert!(!result.reasoning().uncertainty.is_available());
        assert!(matches!(
            &result.reasoning().uncertainty,
            SlotAvailability::Unavailable { reason } if reason.as_ref() == "insufficient_bootstrap_replicates"
        ));

        let artifact = study.export(&result).unwrap();
        let (_consumed, replayed) = PreparedStudy::<StatisticalPreparedState>::consume(
            &artifact,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            replayed.bayesian_estimate().unwrap().distributions.len(),
            posterior.distributions.len()
        );
        assert_eq!(replayed.bayesian_estimate().unwrap().mean_intervals, posterior.mean_intervals);
    }

    #[test]
    fn bayesian_transport_providers_match_independent_dirichlet_moments() {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../conformance/estimate/bayesian_structural_transport/expected.json"
        ))
        .unwrap();
        let counts: [usize; 4] = oracle["counts_x0y0_x0y1_x1y0_x1y1"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let draws = oracle["draws"].as_u64().unwrap() as u32;
        let tolerance = oracle["monte_carlo_tolerance"].as_f64().unwrap();
        let ctx = ExecutionContext::for_tests(251);
        for (estimator, expected_key) in [
            (
                antecedent_estimate::EmpiricalTableEstimator::EmpiricalSupportBayesianBootstrap,
                "empirical_support_p_y1_given_do_x1",
            ),
            (
                antecedent_estimate::EmpiricalTableEstimator::StateSpaceDirichlet,
                "state_space_p_y1_given_do_x1",
            ),
        ] {
            let study = prepared_with_options(
                "dirichlet-oracle",
                counts,
                DependenceGroup::IndependentStudies,
                EmpiricalTableOptions {
                    estimator,
                    posterior_draws: draws,
                    ..EmpiricalTableOptions::default()
                },
            );
            let result = study.estimate_retained(&ctx).unwrap();
            let posterior = result.bayesian_estimate().unwrap();
            assert_eq!(posterior.draws_ok, draws);
            assert_eq!(posterior.draws_failed, 0);
            let mean = posterior
                .distributions
                .iter()
                .map(|distribution| distribution.mean(v(1)).unwrap())
                .sum::<f64>()
                / f64::from(draws);
            let expected = oracle[expected_key].as_f64().unwrap();
            assert!((mean - expected).abs() < tolerance, "{estimator:?}: {mean} != {expected}");
            let artifact = study.export(&result).unwrap();
            let (_, consumed) = PreparedStudy::<StatisticalPreparedState>::consume(
                &artifact,
                ExactEvaluationLimits::default(),
                &ctx,
            )
            .unwrap();
            assert_eq!(consumed.bayesian_estimate().unwrap().draws_ok, posterior.draws_ok,);
            assert_eq!(
                consumed.bayesian_estimate().unwrap().mean_intervals,
                posterior.mean_intervals,
            );
        }
    }

    #[test]
    fn bayesian_draws_that_miss_the_conditioner_withhold_the_interval() {
        let study = prepared_with_options(
            "empty-arm",
            [20, 5, 0, 0],
            DependenceGroup::IndependentStudies,
            EmpiricalTableOptions {
                estimator:
                    antecedent_estimate::EmpiricalTableEstimator::EmpiricalSupportBayesianBootstrap,
                posterior_draws: 8,
                ..EmpiricalTableOptions::default()
            },
        );
        let ctx = ExecutionContext::for_tests(3);
        let withheld = antecedent_estimate::evaluate_bayesian_statistical_transport(
            &study.state.functional,
            &study.state.input,
            &study.state.request,
            study.state.limits,
            antecedent_estimate::statistical_transport::BayesianStatisticalTransportOptions {
                provider: antecedent_estimate::BayesianTransportLawProvider::EmpiricalSupport,
                draws: 8,
                coverage_level: 0.95,
                max_joint_cells: study.state.options.max_joint_cells,
            },
            &ctx,
        )
        .unwrap();
        assert_eq!(withheld.draws_failed, 8);
        assert!(withheld.mean_intervals.is_empty());
        assert_eq!(withheld.interval_reason.as_deref(), Some("bootstrap_failure_fraction"));
    }

    #[test]
    fn dependent_binding_withholds_interval_with_the_estimators_reason() {
        for dependence in [DependenceGroup::LinkedUnits, DependenceGroup::UnknownDependence] {
            let study = prepared_with_dependence(
                "one",
                [20, 5, 10, 15],
                PERCENTILE_95_MIN_REPLICATES,
                dependence,
            );
            assert!(!study.licensed_interval(), "{dependence:?} cannot license an iid bootstrap");
            let inspect = study.inspect();
            match &inspect.reasoning.uncertainty {
                SlotAvailability::Unavailable { reason } => {
                    assert_eq!(reason.as_ref(), "transport.unsupported_dependence");
                }
                _ => panic!("dependent evidence must withhold the interval"),
            }
        }
    }

    #[test]
    fn statistical_lifecycle_publishes_pointwise_bootstrap() {
        let ctx = ExecutionContext::for_tests(7);
        let mut study = prepared("one", [20, 5, 10, 15]);
        let before = study.inspect();
        assert!(before.reasoning.uncertainty.is_available());
        assert_eq!(before.bindings.len(), 1);
        assert_eq!(before.bindings[0].origin, "empirical_plugin");
        let result = study.estimate_checked(&before.identities.execution, &ctx).unwrap();
        assert!(result.reasoning().uncertainty.is_available());
        assert!(result.estimate().atom_intervals.is_some());
        assert!(result.estimate().atom_replicates.is_some());
        let mean = result.distribution().mean(v(1)).unwrap();
        assert!((mean - 0.6).abs() < 1e-12);
        let bytes = study.export(&result).unwrap();
        let (_, consumed) = PreparedStudy::<StatisticalPreparedState>::consume(
            &bytes,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            consumed.distribution().probabilities.as_ref(),
            result.distribution().probabilities.as_ref()
        );
        assert!(consumed.estimate().uncertainty.is_some());
        let replacement = StatisticalTransportInput {
            supplied: Vec::new(),
            samples: vec![sample("two", [10, 10, 10, 10])],
        };
        assert!(study.preview_snapshot(&replacement).unwrap());
        let refreshed = study.refresh(replacement, &ctx).unwrap();
        assert!((refreshed.distribution().mean(v(1)).unwrap() - 0.5).abs() < 1e-12);
        assert_ne!(refreshed.identities().snapshot, result.identities().snapshot);
        assert_eq!(refreshed.identities().identification, result.identities().identification);
    }

    #[test]
    fn statistical_artifact_rejects_claim_and_identity_mutations() {
        let ctx = ExecutionContext::for_tests(7);
        let study = prepared("one", [20, 5, 10, 15]);
        let result = study.estimate(&ctx).unwrap();
        let bytes = study.export(&result).unwrap();
        let original: StatisticalExecutionWire = antecedent_io::from_cbor(&bytes).unwrap();
        let mut variants = Vec::new();
        let mut wire = original.clone();
        wire.identities.snapshot = "00".repeat(32);
        variants.push(wire);
        let mut wire = original.clone();
        wire.identities.inference_binding = "00".repeat(32);
        variants.push(wire);
        let mut wire = original.clone();
        wire.samples[0].n *= 2;
        variants.push(wire);
        let mut wire = original.clone();
        wire.seed += 1;
        variants.push(wire);
        let mut wire = original.clone();
        wire.atom_intervals.as_mut().unwrap()[0].0 = 0.0;
        variants.push(wire);
        let mut wire = original.clone();
        wire.mean_intervals.as_mut().unwrap()[0].2 = 1.0;
        variants.push(wire);
        let mut wire = original.clone();
        wire.uncertainty.as_mut().unwrap().replicates_failed = 1;
        variants.push(wire);
        let mut wire = original.clone();
        wire.uncertainty.as_mut().unwrap().method = "calibrated_magic".into();
        variants.push(wire);
        let mut wire = original.clone();
        wire.atom_replicates.as_mut().unwrap()[0][0] = -1.0;
        variants.push(wire);
        let mut wire = original.clone();
        wire.replicate_ids.as_mut().unwrap()[0] = 999;
        variants.push(wire);
        let mut wire = original.clone();
        wire.options.estimator = "unknown".into();
        variants.push(wire);
        for wire in variants {
            assert!(
                PreparedStudy::<StatisticalPreparedState>::consume(
                    &antecedent_io::to_cbor(&wire).unwrap(),
                    ExactEvaluationLimits::default(),
                    &ctx
                )
                .is_err()
            );
        }
        let (loaded, result) = PreparedStudy::<StatisticalPreparedState>::consume(
            &bytes,
            ExactEvaluationLimits::default(),
            &ExecutionContext::for_tests(999),
        )
        .unwrap();
        assert_eq!(loaded.export(&result).unwrap(), bytes);
        assert_eq!(loaded.seed(), 7);
        assert!(loaded.estimate(&ctx).unwrap_err().to_string().contains("samples_not_embedded"));
    }

    #[test]
    fn cancellation_after_a_completed_replicate_publishes_no_partial_bootstrap() {
        struct CancelAfterFirst(antecedent_core::CancellationToken);
        impl antecedent_core::ProgressSink for CancelAfterFirst {
            fn report(&self, _: f64, stage: &str) {
                if stage == "transport.bootstrap" {
                    self.0.cancel();
                }
            }
        }
        let study = prepared("one", [20, 5, 10, 15]);
        let mut ctx = ExecutionContext::for_tests(7);
        ctx.progress = Some(Arc::new(CancelAfterFirst(ctx.cancellation.clone())));
        assert!(study.estimate(&ctx).unwrap_err().to_string().contains("cancel"));
    }

    #[test]
    fn retained_seed_and_sample_size_are_execution_authority() {
        let a = prepared("one", [20, 5, 10, 15]);
        let b = prepared("one", [40, 10, 20, 30]);
        assert_ne!(a.inspect().identities.snapshot, b.inspect().identities.snapshot);
        assert_eq!(a.inspect().identities.identification, b.inspect().identities.identification);
        let first = a.estimate(&ExecutionContext::for_tests(7)).unwrap();
        let second = a.estimate(&ExecutionContext::for_tests(999)).unwrap();
        assert_eq!(first.estimate().atom_replicates, second.estimate().atom_replicates);
        assert_eq!(first.estimate().uncertainty.as_ref().unwrap().seed, 7);
    }

    #[test]
    fn unknown_dependence_keeps_identification_and_withholds_interval() {
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [v(1)]).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([v(1)]),
            treatments: Arc::from([v(0)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ctx = ExecutionContext::for_tests(1);
        let antecedent_identify::ClassicalTransportResult::Identified(proof) =
            antecedent_identify::identify_classical_transport(
                &diagram,
                &query,
                SidLimits::default(),
                &ctx,
            )
            .unwrap()
        else {
            panic!("identified");
        };
        let catalog = EvidenceCatalog::try_new(
            [Environment::try_new(
                "target",
                [
                    VariableCoordinate {
                        variable: v(0),
                        domain: VariableDomain::Binary,
                        unit: None,
                    },
                    VariableCoordinate {
                        variable: v(1),
                        domain: VariableDomain::Binary,
                        unit: None,
                    },
                ],
                [],
            )
            .unwrap()],
            [EvidenceRegime::try_new(
                RegimeId::from_raw(0),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                [v(0), v(1)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap()],
            [RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(0),
                snapshot_identity: Arc::from("one"),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Unknown,
                weights: None,
                dependence: DependenceGroup::UnknownDependence,
            }],
            Some(TargetSampling::RepresentativeSample),
        )
        .unwrap();
        let study = StudyBuilder::statistical_transport(
            diagram,
            proof.bind_catalog(&catalog).unwrap(),
            StatisticalTransportInput {
                supplied: Vec::new(),
                samples: vec![sample("one", [8, 8, 8, 8])],
            },
            Assignment::from_pairs([(v(0), Value::Int64(1))]),
            ExactEvaluationLimits::default(),
            EmpiricalTableOptions::default(),
            &ctx,
        )
        .unwrap();
        let result = study.estimate(&ctx).unwrap();
        assert!(!result.reasoning().uncertainty.is_available());
        assert_eq!(
            result.estimate().uncertainty_reason.as_deref(),
            Some("transport.unsupported_dependence")
        );
        assert!((result.distribution().mean(v(1)).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn clustered_dependence_keeps_the_point_and_withholds_the_interval() {
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [v(1)]).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([v(1)]),
            treatments: Arc::from([v(0)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ctx = ExecutionContext::for_tests(3);
        let antecedent_identify::ClassicalTransportResult::Identified(proof) =
            antecedent_identify::identify_classical_transport(
                &diagram,
                &query,
                SidLimits::default(),
                &ctx,
            )
            .unwrap()
        else {
            panic!("identified");
        };
        let catalog = EvidenceCatalog::try_new(
            [Environment::try_new(
                "target",
                [
                    VariableCoordinate {
                        variable: v(0),
                        domain: VariableDomain::Binary,
                        unit: None,
                    },
                    VariableCoordinate {
                        variable: v(1),
                        domain: VariableDomain::Binary,
                        unit: None,
                    },
                ],
                [],
            )
            .unwrap()],
            [EvidenceRegime::try_new(
                RegimeId::from_raw(0),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                [v(0), v(1)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap()],
            [RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(0),
                snapshot_identity: Arc::from("one"),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Clustered,
                weights: None,
                dependence: DependenceGroup::LinkedUnits,
            }],
            Some(TargetSampling::RepresentativeSample),
        )
        .unwrap();
        let study = StudyBuilder::statistical_transport(
            diagram,
            proof.bind_catalog(&catalog).unwrap(),
            StatisticalTransportInput {
                supplied: Vec::new(),
                samples: vec![sample("one", [8, 8, 8, 8])],
            },
            Assignment::from_pairs([(v(0), Value::Int64(1))]),
            ExactEvaluationLimits::default(),
            EmpiricalTableOptions::default(),
            &ctx,
        )
        .unwrap();
        assert!(!study.inspect().reasoning.uncertainty.is_available());
        let result = study.estimate(&ctx).unwrap();
        assert_eq!(
            result.estimate().uncertainty_reason.as_deref(),
            Some("transport.unsupported_dependence")
        );
        assert!((result.distribution().mean(v(1)).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn missing_available_regime_is_a_missing_provider() {
        let study = prepared("one", [8, 8, 8, 8]);
        let err = StudyBuilder::statistical_transport(
            study.diagram().clone(),
            study.state.functional.clone(),
            StatisticalTransportInput::default(),
            Assignment::from_pairs([(v(0), Value::Int64(1))]),
            ExactEvaluationLimits::default(),
            EmpiricalTableOptions::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("transport_missing_provider")
                || err.to_string().contains("neither a supplied law")
        );
    }
}
