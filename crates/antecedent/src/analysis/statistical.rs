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
    TransportUncertaintyRow, assemble_point_laws, evaluate_statistical_transport,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{BoundTransportFunctional, ClassicalTransportQuery, SidLimits};
use antecedent_io::{
    IoError, exact_law_wire::ExactLawWire, query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire, transport_proof::TransportProofWire,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

fn err(error: impl std::fmt::Display) -> IoError {
    IoError::Convert(error.to_string())
}
fn digest(domain: IdentityDomain, value: &impl Serialize) -> Result<String, IoError> {
    Ok(antecedent_io::identity::digest_wire(domain, value)?.to_hex())
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
        if self.estimate.uncertainty_reason.is_some()
            || reference.estimate.uncertainty_reason.is_some()
        {
            return Ok(contrast);
        }
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
        if draws.len() < 2
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
        let data = assemble_point_laws(&input, &functional, &options).map_err(err)?;
        let samples =
            input.samples.iter().map(SampleSummary::from_sample).collect::<Result<Vec<_>, _>>()?;
        Self::build_fitted(
            diagram, functional, input, data, samples, request, limits, options, ctx, false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_fitted(
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
                options.estimator.as_str(),
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

    fn licensed_interval(&self) -> bool {
        self.state.options.bootstrap_replicates >= 2
            && !self.state.samples.is_empty()
            && self.state.samples.iter().all(|sample| {
                self.evidence_catalog().bindings.iter().any(|b| {
                    b.regime.raw() == sample.regime
                        && b.sampling == antecedent_core::SamplingDesign::Independent
                        && b.dependence == antecedent_core::DependenceGroup::IndependentStudies
                        && b.weights.is_none()
                })
            })
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
            reasoning: self.reasoning(false, None),
            theorem_scope: if self.state.functional.derivation().sources().is_empty() {
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
    ) -> ReasoningView {
        let uncertainty = match estimate {
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
            None if self.state.options.bootstrap_replicates == 0 => {
                SlotAvailability::unavailable("bootstrap_not_requested")
            }
            None if self.state.options.bootstrap_replicates == 1 => {
                SlotAvailability::unavailable("insufficient_bootstrap_replicates")
            }
            None => SlotAvailability::unavailable(
                antecedent_estimate::dependence_refusal(
                    self.state.functional.catalog(),
                    &self.state.input.samples,
                )
                .unwrap_or("exact_supplied_law_no_sampling_uncertainty"),
            ),
        };
        ReasoningView::new(
            SlotAvailability::Available(IdentificationSlot::identified_singleton(
                IdentificationStatus::NonparametricallyIdentified,
            )),
            SlotAvailability::Available(SupportSlot::new(
                "stage_contract",
                Some(Arc::from("transport.empirical_table")),
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
                "Accepted causal graph, mechanism selections, and empirical-table regularity describe the declared populations.",
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
        let estimate = if self.state.grid.is_empty() {
            evaluate_statistical_transport(
                &self.state.functional,
                &self.state.input,
                self.state.request.clone(),
                self.state.limits,
                &self.state.options,
                &frozen_ctx,
            )
            .map_err(err)?
        } else {
            let requests: Vec<_> = self
                .state
                .grid
                .iter()
                .map(|request| {
                    Assignment::from_pairs(
                        request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
                    )
                })
                .collect();
            let index = self
                .state
                .grid
                .iter()
                .position(|request| *request == statistical_assignments(&self.state.request))
                .ok_or_else(|| err("retained target missing from grid"))?;
            antecedent_estimate::statistical_transport::evaluate_statistical_transport_grid(
                &self.state.functional,
                &self.state.input,
                &requests,
                self.state.limits,
                &self.state.options,
                &frozen_ctx,
            )
            .map_err(err)?
            .remove(index)
        };
        Ok(StatisticalStudyResult {
            reasoning: self.reasoning(true, Some(&estimate)),
            estimate,
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
        if self.state.loaded_only {
            return Err(err("transport.samples_not_embedded"));
        }
        let mut frozen_ctx = ctx.clone();
        frozen_ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let estimates =
            antecedent_estimate::statistical_transport::evaluate_statistical_transport_grid(
                &self.state.functional,
                &self.state.input,
                requests,
                self.state.limits,
                &self.state.options,
                &frozen_ctx,
            )
            .map_err(err)?;
        requests
            .iter()
            .zip(estimates)
            .map(|(request, estimate)| {
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
                let result = StatisticalStudyResult {
                    reasoning: prepared.reasoning(true, Some(&estimate)),
                    identities: prepared.state.identities.clone(),
                    estimate,
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
                origin: "empirical_plugin".into(),
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
        let data = assemble_point_laws(&input, &self.state.functional, &self.state.options)
            .map_err(err)?;
        let mut catalog = self.state.functional.catalog().clone();
        let mut bindings = catalog.bindings.to_vec();
        for binding in &mut bindings {
            let mut snapshots = data
                .laws()
                .iter()
                .filter(|law| law.regime() == binding.regime)
                .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity);
            let snapshot = snapshots.next().ok_or_else(|| err("missing bound snapshot"))?;
            if snapshots.any(|other| other != snapshot) {
                return Err(err("regime has inconsistent snapshot identities"));
            }
            binding.snapshot_identity = Arc::from(snapshot);
        }
        catalog.bindings = bindings.into();
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
        let mut graph = Admg::empty();
        for node in &wire.nodes {
            graph.add_node(NodeRef::Static(VariableId::from_raw(*node))).map_err(err)?;
        }
        for &(a, b) in &wire.directed {
            graph
                .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .map_err(err)?;
        }
        for &(a, b) in &wire.bidirected {
            graph
                .insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .map_err(err)?;
        }
        graph.validate().map_err(err)?;
        let diagram = SelectionDiagram::try_new(
            graph,
            wire.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(err)?;
        let query = ClassicalTransportQuery {
            outcomes: wire.proof.proof.outcomes.iter().copied().map(VariableId::from_raw).collect(),
            treatments: wire
                .proof
                .proof
                .treatments
                .iter()
                .copied()
                .map(VariableId::from_raw)
                .collect(),
            source: Arc::from(wire.proof.proof.source.as_str()),
            target: Arc::from(wire.proof.proof.target.as_str()),
        };
        let proof = wire.proof.check(
            &diagram,
            &query,
            SidLimits { steps: limits.operations, depth: limits.depth },
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
        let estimate = validate_uncertainty(&wire, point, &prepared)?;
        let reasoning = prepared.reasoning(true, Some(&estimate));
        if super::contract::reasoning_section(&reasoning) != wire.reasoning {
            return Err(err("transport artifact reasoning mismatch"));
        }
        let result = StatisticalStudyResult { reasoning, estimate, identities: wire.identities };
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
pub(super) struct StatisticalOptionsWire {
    estimator: String,
    bootstrap_replicates: u32,
    coverage_level: f64,
    max_joint_cells: usize,
}

impl StatisticalOptionsWire {
    pub(super) fn from_options(options: &EmpiricalTableOptions) -> Self {
        Self {
            estimator: options.estimator.as_str().into(),
            bootstrap_replicates: options.bootstrap_replicates,
            coverage_level: options.coverage_level,
            max_joint_cells: options.max_joint_cells,
        }
    }
    pub(super) fn to_options(&self) -> Result<EmpiricalTableOptions, IoError> {
        if self.estimator != antecedent_estimate::EMPIRICAL_TABLE_PLUGIN {
            return Err(err("unknown or unlicensed empirical estimator"));
        }
        Ok(EmpiricalTableOptions {
            estimator: if self.estimator == antecedent_estimate::EMPIRICAL_TABLE_DIRICHLET {
                antecedent_estimate::EmpiricalTableEstimator::Dirichlet
            } else {
                antecedent_estimate::EmpiricalTableEstimator::Plugin
            },
            bootstrap_replicates: self.bootstrap_replicates,
            coverage_level: self.coverage_level,
            max_joint_cells: self.max_joint_cells,
        })
    }
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
    atom_intervals: Option<Vec<(f64, f64)>>,
    mean_intervals: Option<Vec<(u32, f64, f64)>>,
    replicate_ids: Option<Vec<u32>>,
    uncertainty_reason: Option<String>,
    atom_replicates: Option<Vec<Vec<f64>>>,
    reasoning: antecedent_io::ReasoningSectionWire,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct SampleSummary {
    population: String,
    regime: u32,
    snapshot: String,
    interventions: Vec<(u32, ValueWire)>,
    n: u32,
    content_digest: String,
}
impl SampleSummary {
    pub(super) fn from_sample(sample: &antecedent_estimate::RegimeSample) -> Result<Self, IoError> {
        let columns: Vec<_> = sample.columns.iter().map(|(v, xs)| (v.raw(), xs)).collect();
        let mut interventions: Vec<_> = sample
            .interventions
            .iter()
            .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
            .collect();
        interventions.sort_by_key(|a| a.0);
        Ok(Self {
            population: sample.population.to_string(),
            regime: sample.regime.raw(),
            snapshot: sample.snapshot_identity.to_string(),
            interventions,
            n: u32::try_from(sample.n()).map_err(err)?,
            content_digest: digest(IdentityDomain::DataSnapshot, &columns)?,
        })
    }
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
        if law.origin() != antecedent_expr::LawOrigin::EmpiricalPlugin
            || law.snapshot_identity() != sample.snapshot
        {
            return Err(err("sample summary origin/snapshot mismatch"));
        }
        if catalog.bindings.iter().any(|b| {
            b.regime.raw() == sample.regime && b.snapshot_identity.as_ref() != sample.snapshot
        }) {
            return Err(err("sample summary binding mismatch"));
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
        .filter(|law| law.origin() == antecedent_expr::LawOrigin::EmpiricalPlugin)
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
    if !wire.samples.iter().all(|sample| {
        prepared.evidence_catalog().bindings.iter().any(|b| {
            b.regime.raw() == sample.regime
                && b.sampling == antecedent_core::SamplingDesign::Independent
                && b.dependence == antecedent_core::DependenceGroup::IndependentStudies
                && b.weights.is_none()
        })
    }) {
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
    let unavailable = row.replicates_ok < 2
        || f64::from(row.replicates_failed) / f64::from(row.replicates_requested) > 0.5;
    if unavailable {
        if wire.uncertainty_reason.as_deref() != Some("bootstrap_failure_fraction")
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

#[cfg(test)]
mod tests {
    use super::*;
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
                dependence: DependenceGroup::IndependentStudies,
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
            EmpiricalTableOptions { bootstrap_replicates: 39, ..EmpiricalTableOptions::default() },
            &ctx,
        )
        .unwrap()
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
