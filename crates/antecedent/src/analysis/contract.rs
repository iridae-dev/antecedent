//! Causal-contract companion for [`Study`] / [`PreparedStudy`].
//!
//! ADR 0022: the practitioner handle stays [`PreparedStudy`]. This module
//! builds an inspectable contract from existing products. `inspect` is cheap
//! structural classification; `contract` includes cached identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionSet, AssumptionSlot, AssumptionSource, AssumptionStatus, CausalSchema,
    ClaimDomains, ClaimEnvelope, ClaimKind, ContractIdentities, DomainStatus, ExecutionContext,
    IDENTITY_FORMAT, IdentificationSlot, IdentificationStatus, IdentityDomain, ObligationKind,
    ObligationRecord, ObligationScope, ReasoningView, SlotAvailability, SupportSlot,
    TransformIntent, TransformationReport, UncertaintyComponent, UncertaintySlot,
    UncertaintySource, intent_effects,
};
use antecedent_data::TableView;
use antecedent_identify::{CAPPED_COMPLETION_DIAGNOSTIC_CODE, IdentificationResult};
use antecedent_io::{
    AnalysisResultContractWire, AnalysisResultWire, AssumptionSlotWire, ClaimIdentityWire,
    ClaimSectionWire, ContractIdentitiesWire, DataPartitionIdentityWire, DataSnapshotIdentityWire,
    ExecutionIdentityWire, GraphIdentityWire, IdentificationIdentityWire, IdentificationProductWire,
    IdentificationSlotWire, InferenceBindingWire, InferentialCommitmentsWire, ObligationSectionWire,
    ObservationIdentityWire, ProgramIdentityWire, ReasoningSectionWire, SlotSectionWire,
    SupportSlotWire, TargetIdentityWire, UncertaintyComponentWire, UncertaintySlotWire,
    admg_identity, causal_query_to_wire, claim_digest, cpdag_identity, dag_identity,
    HorizonAdjustmentNodeWire, TemporalIdentificationWire, data_snapshot_digest, digest_wire,
    encode_analysis_result_artifact_with_contract, execution_digest,
    execution_identity_from_context, identification_digest, identification_product_digest_wire,
    identification_product_wire, identification_to_wire, inference_binding_digest,
    observation_identity_wire, pag_identity, program_digest, schema_to_wire,
    temporal_cpdag_identity, temporal_dag_identity, temporal_pag_identity,
};

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::result::StudyResult;
use crate::support::{CellStatus, StructureSource};

use super::builder::DataInput;
use super::execute::Study;
use super::prepared::{CachedTemporalIdentification, PreparedStudy};

/// Immutable causal-contract companion. Not a second builder.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CausalContract {
    /// Domain-separated identities.
    pub identities: ContractIdentities,
    /// Four reasoning slots.
    pub reasoning: ReasoningView,
    /// Graph class of the accepted structure (stub class for graph-posterior).
    pub graph_class: GraphClass,
    /// How structure was supplied.
    pub structure_source: StructureSource,
    /// Acceptance version for audit; does not alter graph semantics.
    pub accepted_version: u32,
    /// Discovery algorithm for audit; distinct from the selected identifier.
    pub discovery_algorithm: Option<Arc<str>>,
    /// Original optional portable binding on the accepted structure.
    pub accepted_variable_names: Option<Arc<[Arc<str>]>>,
    /// Support-matrix status, when the query is on-axis.
    pub support_status: Option<CellStatus>,
    /// Identifier selected or defaulted.
    pub identifier: Option<Arc<str>>,
    /// Estimator selected or defaulted.
    pub estimator: Option<Arc<str>>,
}

impl CausalContract {
    /// Preview a transformation against this contract's identities.
    #[must_use]
    pub fn preview_transform(&self, intent: TransformIntent) -> TransformationReport {
        let mut obligations: Vec<ObligationRecord> = self
            .reasoning
            .assumptions
            .as_ref()
            .map(|slot| slot.obligations.iter().cloned().collect())
            .unwrap_or_default();
        if matches!(
            intent,
            TransformIntent::NewConditionalQuery | TransformIntent::FilterPopulation
        ) {
            obligations.push(preview_obligation(
                "transform.population_or_query",
                ObligationKind::CheckNotRun,
                "reidentify",
                "new target requires identification; a dataframe filter is not a certificate",
            ));
        }
        if intent == TransformIntent::AverageUnweightedClass {
            obligations.push(preview_obligation(
                "transform.unweighted_class_prior",
                ObligationKind::Uncheckable,
                "declared_class_prior",
                "enumeration weights are not a class prior; refuse probabilistic aggregation",
            ));
        }
        if intent == TransformIntent::Retarget {
            obligations.push(preview_obligation(
                "transform.retarget_score_table",
                ObligationKind::CheckNotRun,
                "retarget_overlap",
                "retarget reuses the prepared score table only within declared weight dependencies",
            ));
        }
        TransformationReport::new(
            intent,
            self.identities.refs(),
            intent_effects(intent).iter().cloned(),
            obligations,
        )
    }

    /// Durable contract section bound to this study's target payload.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn artifact_section(
        &self,
        study: &Study,
        claim: Option<&ClaimEnvelope>,
        execution: Option<&ExecutionIdentityWire>,
    ) -> Result<AnalysisResultContractWire, CausalError> {
        Ok(self.section_from_payloads(&contract_payloads_for(study)?, claim, execution))
    }

    fn section_from_payloads(
        &self,
        payloads: &ContractPayloads,
        claim: Option<&ClaimEnvelope>,
        execution: Option<&ExecutionIdentityWire>,
    ) -> AnalysisResultContractWire {
        AnalysisResultContractWire {
            format: antecedent_io::CONTRACT_SECTION_FORMAT,
            identities: ContractIdentitiesWire::from(&self.identities),
            target: payloads.target.clone(),
            reasoning: reasoning_section(&self.reasoning),
            graph_class: self.graph_class.as_str().into(),
            structure_source: self.structure_source.as_str().into(),
            identifier: self.identifier.as_ref().map(std::string::ToString::to_string),
            estimator: self.estimator.as_ref().map(std::string::ToString::to_string),
            claim: claim.map(claim_section),
            identification: Some(payloads.identification.clone()),
            identification_product: payloads.identification_product.clone(),
            program: Some(payloads.program.clone()),
            inference_binding: Some(payloads.inference_binding.clone()),
            observation: Some(payloads.observation.clone()),
            data_snapshot: Some(payloads.data_snapshot.clone()),
            execution: execution.cloned(),
        }
    }
}

impl Study {
    /// Cheap structural inspection. Does not identify, fit, or execute.
    ///
    /// Identification-product and uncertainty slots are explicitly unavailable.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn inspect(&self) -> Result<CausalContract, CausalError> {
        compile_contract(self, None)
    }
}

impl PreparedStudy {
    /// Crate-visible study for contract construction.
    pub(crate) fn study(&self) -> &Study {
        &self.analysis
    }

    /// Contract built from cached identification products.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn contract(&self) -> Result<CausalContract, CausalError> {
        compile_contract(self.study(), Some(self))
    }

    /// Pure transformation preview bound to this handle's program identity.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures while reading the contract.
    pub fn preview_transform(
        &self,
        intent: TransformIntent,
    ) -> Result<TransformationReport, CausalError> {
        Ok(self.contract()?.preview_transform(intent))
    }

    /// Encode an executed result with a verified contract section.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures or mass totals that do not conserve.
    pub fn encode_contracted_result(
        &self,
        result: &StudyResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CausalError> {
        let (contract, payloads) = compile_with_payloads(self.study(), Some(self))?;
        let claim = result.claim(&contract, ctx)?;
        let execution = execution_identity_from_context(ctx);
        let section = contract.section_from_payloads(&payloads, Some(&claim), Some(&execution));
        let body = analysis_result_wire(
            self.query(),
            result,
            self.temporal_identification(),
        )?;
        let names: Vec<String> =
            self.schema().variables().iter().map(|variable| variable.name.to_string()).collect();
        let artifact = encode_analysis_result_artifact_with_contract(
            &body,
            names,
            artifact_id,
            Some(&section),
        )
        .map_err(|err| io_err(&err))?;
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).map_err(|err| io_err(&err))?;
        Ok(bytes)
    }

    /// Apply a compatible-data refresh only if `preview` still binds this handle.
    ///
    /// Failed refresh leaves the original handle usable.
    ///
    /// # Errors
    ///
    /// Stale preview, schema mismatch, or estimation failure.
    pub fn apply_refresh(
        &mut self,
        preview: &TransformationReport,
        data: antecedent_data::TabularData,
        ctx: &ExecutionContext,
    ) -> Result<crate::result::StudyResult, CausalError> {
        self.require_bound_preview(preview, TransformIntent::CompatibleDataReplace, "refresh")?;
        self.refresh(data, ctx)
    }

    /// Apply retarget only if `preview` still binds this handle.
    ///
    /// # Errors
    ///
    /// Stale preview, missing score table, or retarget refusal.
    pub fn apply_retarget(
        &self,
        preview: &TransformationReport,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
        ctx: &ExecutionContext,
    ) -> Result<crate::result::StudyResult, CausalError> {
        self.require_bound_preview(preview, TransformIntent::Retarget, "retarget")?;
        self.retarget(weights, depends_on, ctx)
    }

    /// Compose the existing design ranker onto this prepared handle.
    ///
    /// # Errors
    ///
    /// Unlicensed support, missing identification product, or ranker failure.
    pub fn rank_designs<A, O>(
        &self,
        ranker: &crate::design::DesignRanker,
        objective: &crate::design::DesignObjective,
        candidates: &[crate::design::CandidateDesign],
        eval: &crate::design::DesignEvaluationContext<'_, A, O>,
        ctx: &ExecutionContext,
    ) -> Result<crate::design::DesignRanking, CausalError>
    where
        A: Clone,
        O: Clone,
    {
        let contract = self.contract()?;
        if !matches!(contract.support_status, Some(CellStatus::Licensed)) {
            return Err(CausalError::Unsupported {
                message: "design ranking requires a licensed prepared contract",
            });
        }
        if contract.identities.identification_product.is_none() {
            return Err(CausalError::Unsupported {
                message: "design ranking requires cached identification products",
            });
        }
        crate::design::rank_designs(ranker, objective, candidates, eval, ctx)
    }

    fn require_bound_preview(
        &self,
        preview: &TransformationReport,
        intent: TransformIntent,
        action: &'static str,
    ) -> Result<(), CausalError> {
        let contract = self.contract()?;
        if preview.intent != intent || !preview.binds_contract(&contract.identities) {
            return Err(CausalError::Compile {
                message: format!("{action} preview does not bind the current prepared contract"),
            });
        }
        Ok(())
    }
}

impl StudyResult {
    /// Portable claim over `contract` and this execution.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures, or mass totals that do not conserve.
    pub fn claim(
        &self,
        contract: &CausalContract,
        ctx: &ExecutionContext,
    ) -> Result<ClaimEnvelope, CausalError> {
        let reasoning = result_reasoning(self, &contract.reasoning)?;
        let kind = claim_kind(self, &reasoning);
        let value = match kind {
            ClaimKind::Point => Some(self.effect()),
            _ => None,
        };
        let execution = execution_digest(&execution_identity_from_context(ctx))
            .map_err(|err| io_err(&err))?;
        let claim_id = claim_digest(&ClaimIdentityWire::new(
            *contract.identities.program.as_bytes(),
            *contract.identities.target.as_bytes(),
            kind.as_str(),
            value.filter(|v| v.is_finite()).map(f64::to_bits),
            Some(*execution.as_bytes()),
        ))
        .map_err(|err| io_err(&err))?;
        let identified =
            reasoning.identification.as_ref().is_some_and(|slot| slot.identified_mass > 0.0);
        let support_domain = match (
            contract.support_status,
            reasoning.support.as_ref().and_then(|slot| slot.empirical.as_ref()),
        ) {
            (Some(CellStatus::Licensed), Some(_)) => DomainStatus::Supported,
            (Some(CellStatus::Refused | CellStatus::NotApplicable { .. }), _) => {
                DomainStatus::OutsideScope
            }
            _ => DomainStatus::Unknown,
        };
        Ok(ClaimEnvelope::new(
            claim_id,
            contract.identities,
            kind,
            value,
            None,
            reasoning,
            ClaimDomains::new(
                if identified { DomainStatus::Identified } else { DomainStatus::Unknown },
                support_domain,
                DomainStatus::Evaluated,
            ),
            Some(execution),
            [],
        ))
    }

    /// Scalar / identification body for the composite `analysis_result` container.
    ///
    /// Heavy axes (posterior draws, mediation grids, structural mixtures) stay
    /// on existing specialized encoders; this path records the query,
    /// identification certificate, and scalar estimate used by the contract
    /// section.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn analysis_result_wire(
        &self,
        query: &antecedent_core::CausalQuery,
    ) -> Result<AnalysisResultWire, CausalError> {
        analysis_result_wire(query, self, None)
    }
}

fn io_err(err: &antecedent_io::IoError) -> CausalError {
    CausalError::Compile { message: err.to_string() }
}

struct ContractPayloads {
    target: TargetIdentityWire,
    identification: IdentificationIdentityWire,
    identification_product: Option<IdentificationProductWire>,
    program: ProgramIdentityWire,
    inference_binding: InferenceBindingWire,
    observation: ObservationIdentityWire,
    data_snapshot: DataSnapshotIdentityWire,
    identities: ContractIdentities,
}

fn contract_payloads_for(study: &Study) -> Result<ContractPayloads, CausalError> {
    let cached = cached_identification(study);
    contract_payloads(study, cached, cached.is_some_and(identification_search_capped))
}

fn contract_payloads(
    study: &Study,
    cached: Option<&IdentificationResult>,
    search_capped: bool,
) -> Result<ContractPayloads, CausalError> {
    let schema = data_schema(&study.data);
    let target = TargetIdentityWire {
        format: IDENTITY_FORMAT,
        schema: schema_to_wire(schema),
        query: causal_query_to_wire(&study.query).map_err(|err| io_err(&err))?,
    };
    let target_digest = digest_wire(IdentityDomain::Target, &target).map_err(|err| io_err(&err))?;
    let observation = observation_identity_wire(schema, observation_tags(&study.query));
    let observation_digest =
        digest_wire(IdentityDomain::Observation, &observation).map_err(|err| io_err(&err))?;
    let graph = graph_identity(study)?;
    let identification = IdentificationIdentityWire {
        format: IDENTITY_FORMAT,
        target: *target_digest.as_bytes(),
        graph_class: study.graph.class().as_str().into(),
        structure_source: study.structure_source.as_str().into(),
        accepted_version: study.graph.version(),
        algorithm_id: study.graph.algorithm_id().map(str::to_string),
        schema_names: Some(schema.variables().iter().map(|v| v.name.to_string()).collect()),
        graph,
        observation: *observation_digest.as_bytes(),
    };
    let identification_digest =
        identification_digest(&identification).map_err(|err| io_err(&err))?;
    let identification_product = match cached {
        Some(result) => {
            Some(identification_product_wire(result, search_capped).map_err(|err| io_err(&err))?)
        }
        None => None,
    };
    let identification_product_digest = match &identification_product {
        Some(wire) => {
            Some(identification_product_digest_wire(wire).map_err(|err| io_err(&err))?)
        }
        None => None,
    };
    let commitments = inferential_commitments(study);
    let program = ProgramIdentityWire {
        format: IDENTITY_FORMAT,
        identification: *identification_digest.as_bytes(),
        identification_product: identification_product_digest.map(|digest| *digest.as_bytes()),
        commitments: commitments.clone(),
    };
    let program_digest = program_digest(&program).map_err(|err| io_err(&err))?;
    let inference_binding = InferenceBindingWire {
        format: IDENTITY_FORMAT,
        program: *program_digest.as_bytes(),
        inference: commitments.inference.clone(),
        bootstrap_replicates: study.bootstrap_replicates,
        n_draws: match &study.inference {
            InferenceMode::Bayesian(cfg) => Some(u64::try_from(cfg.n_draws).unwrap_or(u64::MAX)),
            InferenceMode::Frequentist => None,
        },
        prior_scale: match &study.inference {
            InferenceMode::Bayesian(cfg) if cfg.prior.is_none() && cfg.prior_artifact.is_none() => {
                Some(cfg.prior_scale)
            }
            _ => None,
        },
        prior_mapping: match &study.inference {
            InferenceMode::Bayesian(cfg) => cfg.prior_mapping.as_ref().map(prior_mapping_tag),
            InferenceMode::Frequentist => None,
        },
        validation_suite: study.refute.validation_suite_id().map(str::to_string),
        overlap_policy: study.overlap_policy.map(overlap_policy_tag),
    };
    let inference_binding_digest =
        inference_binding_digest(&inference_binding).map_err(|err| io_err(&err))?;
    let data_snapshot = data_snapshot_wire(study, &observation_digest)?;
    let snapshot_digest = data_snapshot_digest(&data_snapshot).map_err(|err| io_err(&err))?;
    Ok(ContractPayloads {
        target,
        identification,
        identification_product,
        program,
        inference_binding,
        observation,
        data_snapshot,
        identities: ContractIdentities::new(
            target_digest,
            identification_digest,
            identification_product_digest,
            program_digest,
            inference_binding_digest,
            observation_digest,
            snapshot_digest,
        ),
    })
}

fn compile_contract(
    study: &Study,
    prepared: Option<&PreparedStudy>,
) -> Result<CausalContract, CausalError> {
    Ok(compile_with_payloads(study, prepared)?.0)
}

fn compile_with_payloads(
    study: &Study,
    prepared: Option<&PreparedStudy>,
) -> Result<(CausalContract, ContractPayloads), CausalError> {
    let cached = cached_identification(study);
    let search_capped = cached.is_some_and(identification_search_capped);
    let payloads = contract_payloads(study, cached, search_capped)?;
    let reasoning = reasoning_view(study, prepared, cached, search_capped);
    Ok((
        CausalContract {
            identities: payloads.identities,
            reasoning,
            graph_class: study.graph.class(),
            structure_source: study.structure_source,
            accepted_version: study.graph.version(),
            discovery_algorithm: study.graph.algorithm_id().map(Arc::from),
            accepted_variable_names: study.graph.variable_names().map(Arc::from),
            support_status: study.support_status,
            identifier: study.identifier.map(|id| Arc::from(id.as_str())),
            estimator: study.estimator.map(|id| Arc::from(id.as_str())),
        },
        payloads,
    ))
}

fn data_schema(data: &DataInput) -> &CausalSchema {
    match data {
        DataInput::Tabular(data) => data.schema(),
        DataInput::Temporal(data) | DataInput::Event(data) => data.schema(),
        DataInput::MultiEnv(data) => data.schema(),
        DataInput::Panel(data) => data.schema(),
    }
}

fn data_snapshot_wire(
    study: &Study,
    observation: &antecedent_core::SemanticDigest,
) -> Result<DataSnapshotIdentityWire, CausalError> {
    let (modality, regularity, row_count, unit_count) = match &study.data {
        DataInput::Tabular(data) => ("tabular", None, u64_count(data.row_count())?, None),
        DataInput::Temporal(data) | DataInput::Event(data) => (
            if matches!(study.data, DataInput::Event(_)) { "event" } else { "series" },
            Some(regularity_tag(&data.time_index().regularity)),
            u64_count(data.row_count())?,
            None,
        ),
        DataInput::MultiEnv(data) => (
            "multi_env",
            None,
            u64_count(data.environments().iter().map(TableView::row_count).sum())?,
            Some(u64_count(data.env_count())?),
        ),
        DataInput::Panel(data) => {
            ("panel", None, u64_count(data.total_rows())?, Some(u64_count(data.unit_count())?))
        }
    };
    let partitions = match &study.data {
        DataInput::Tabular(data) => vec![data_partition(data.storage(), None, None)?],
        DataInput::Temporal(data) | DataInput::Event(data) => vec![series_partition(data, None)?],
        DataInput::MultiEnv(data) => data
            .environments()
            .iter()
            .map(|series| series_partition(series, None))
            .collect::<Result<_, _>>()?,
        DataInput::Panel(data) => data
            .units()
            .iter()
            .map(|unit| series_partition(&unit.series, Some(unit.unit_id)))
            .collect::<Result<_, _>>()?,
    };
    Ok(DataSnapshotIdentityWire {
        format: IDENTITY_FORMAT,
        observation: *observation.as_bytes(),
        modality: modality.into(),
        regularity,
        row_count,
        unit_count,
        partitions,
    })
}

fn data_partition(
    storage: &antecedent_data::OwnedColumnarStorage,
    unit_id: Option<u32>,
    regularity: Option<String>,
) -> Result<DataPartitionIdentityWire, CausalError> {
    Ok(DataPartitionIdentityWire {
        content: storage.content_digest(),
        row_count: u64_count(storage.row_count())?,
        unit_id,
        regularity,
    })
}

fn series_partition(
    series: &antecedent_data::TimeSeriesData,
    unit_id: Option<u32>,
) -> Result<DataPartitionIdentityWire, CausalError> {
    data_partition(series.storage(), unit_id, Some(regularity_tag(&series.time_index().regularity)))
}

fn u64_count(value: usize) -> Result<u64, CausalError> {
    u64::try_from(value).map_err(|_| CausalError::Compile { message: "count exceeds u64".into() })
}

fn regularity_tag(regularity: &antecedent_data::SamplingRegularity) -> String {
    match regularity {
        antecedent_data::SamplingRegularity::Regular { interval_ns } => {
            format!("regular:{interval_ns}")
        }
        antecedent_data::SamplingRegularity::Irregular => "irregular".into(),
    }
}

fn observation_tags(query: &antecedent_core::CausalQuery) -> Vec<String> {
    match query {
        antecedent_core::CausalQuery::Response(query) => {
            query.observation_assumptions.iter().map(observation_assumption_tag).collect()
        }
        _ => Vec::new(),
    }
}

fn observation_assumption_tag(assumption: &antecedent_core::ObservationAssumption) -> String {
    match assumption {
        antecedent_core::ObservationAssumption::IndependentGiven(_) => "independent_given".into(),
        antecedent_core::ObservationAssumption::OutcomeIndependentGiven(_) => {
            "outcome_independent_given".into()
        }
        antecedent_core::ObservationAssumption::Structural(name) => format!("structural:{name}"),
    }
}

fn graph_identity(study: &Study) -> Result<GraphIdentityWire, CausalError> {
    if let Some(posterior) = &study.graph_posterior {
        return Ok(GraphIdentityWire::GraphPosterior {
            graph_class: study.graph.class().as_str().into(),
            n_atoms: u64::try_from(posterior.n_graphs).unwrap_or(u64::MAX),
        });
    }
    accepted_graph_identity(&study.graph)
}

fn accepted_graph_identity(graph: &AcceptedGraph) -> Result<GraphIdentityWire, CausalError> {
    match graph.class() {
        GraphClass::Dag => {
            dag_identity(graph.as_dag().expect("Dag class")).map_err(|err| io_err(&err))
        }
        GraphClass::Admg => {
            admg_identity(graph.as_admg().expect("Admg class")).map_err(|err| io_err(&err))
        }
        GraphClass::Cpdag => {
            cpdag_identity(graph.as_cpdag().expect("Cpdag class")).map_err(|err| io_err(&err))
        }
        GraphClass::Pag => {
            pag_identity(graph.as_pag().expect("Pag class")).map_err(|err| io_err(&err))
        }
        GraphClass::TemporalDag => {
            temporal_dag_identity(graph.as_temporal_dag().expect("TemporalDag class"))
                .map_err(|err| io_err(&err))
        }
        GraphClass::TemporalCpdag => {
            Ok(temporal_cpdag_identity(graph.as_temporal_cpdag().expect("TemporalCpdag class")))
        }
        GraphClass::TemporalPag => {
            Ok(temporal_pag_identity(graph.as_temporal_pag().expect("TemporalPag class")))
        }
    }
}

fn cached_identification(study: &Study) -> Option<&IdentificationResult> {
    if let Some(cache) = study.identification_cache.as_ref() {
        return Some(&cache.identification);
    }
    if let Some(cache) = study.pag_identification_cache.as_ref() {
        return Some(&cache.identification);
    }
    if let Some(cache) = study.cpdag_identification_cache.as_ref() {
        return Some(&cache.identification);
    }
    if let Some(cache) = study.temporal_identification_cache.as_ref() {
        return cache.by_horizon.first().map(|horizon| &horizon.identification);
    }
    None
}

fn identification_search_capped(result: &IdentificationResult) -> bool {
    result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code.as_ref() == CAPPED_COMPLETION_DIAGNOSTIC_CODE)
}

fn inferential_commitments(study: &Study) -> InferentialCommitmentsWire {
    InferentialCommitmentsWire {
        format: IDENTITY_FORMAT,
        estimator: study.estimator.map(|id| id.as_str().to_string()),
        identifier: study.identifier.map(|id| id.as_str().to_string()),
        inference: match study.inference {
            InferenceMode::Frequentist => "frequentist".into(),
            InferenceMode::Bayesian(_) => "bayesian".into(),
        },
        validation_suite: study.refute.validation_suite_id().map(str::to_string),
        interval_target: match study.inference {
            InferenceMode::Frequentist => "sampling_se".into(),
            InferenceMode::Bayesian(_) => "posterior_quantile".into(),
        },
        prior_required: matches!(study.inference, InferenceMode::Bayesian(_)),
    }
}

fn prior_mapping_tag(mapping: &antecedent_io::PriorMapping) -> String {
    match mapping {
        antecedent_io::PriorMapping::IdenticalCoefficientSubspace => {
            "identical_coefficient_subspace".into()
        }
        antecedent_io::PriorMapping::EffectFunctional { source_quantity } => {
            format!("effect_functional:{source_quantity}")
        }
        antecedent_io::PriorMapping::NamedParameters { pairs } => {
            let mut pairs = pairs.clone();
            pairs.sort();
            format!(
                "named_parameters:{}",
                pairs
                    .iter()
                    .map(|(src, dst)| format!("{src}->{dst}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

fn overlap_policy_tag(policy: antecedent_estimate::OverlapPolicy) -> String {
    match policy {
        antecedent_estimate::OverlapPolicy::ExplicitOverride => "explicit_override".into(),
        antecedent_estimate::OverlapPolicy::RequireDiagnostics { clip, trim } => {
            format!(
                "require_diagnostics:clip={}:trim={}",
                clip.map_or_else(|| "none".into(), ordered_float_bits),
                trim.map_or_else(|| "none".into(), ordered_float_bits),
            )
        }
    }
}

fn ordered_float_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn reasoning_view(
    study: &Study,
    prepared: Option<&PreparedStudy>,
    cached: Option<&IdentificationResult>,
    search_capped: bool,
) -> ReasoningView {
    let support = SupportSlot::new(
        study.support_status.map_or("off_axis", CellStatus::as_str),
        matrix_coordinate(study).map(Arc::from),
        SlotAvailability::unavailable("not_evaluated"),
    );
    let mut obligations =
        obligations_from_set(cached.map(|result| &result.required_assumptions));
    obligations.extend(observation_obligations(&study.query));
    let assumptions = AssumptionSlot::new(obligations);
    if prepared.is_none() || cached.is_none() {
        return ReasoningView::structural(support, assumptions);
    }
    let result = cached.expect("checked");
    let mut slot = IdentificationSlot::identified_singleton(result.status);
    slot.search_capped = search_capped;
    if search_capped {
        slot.full_mass_scope = false;
    }
    ReasoningView::new(
        SlotAvailability::Available(slot),
        SlotAvailability::Available(support),
        SlotAvailability::unavailable("execution_specific"),
        SlotAvailability::Available(assumptions),
    )
}

fn matrix_coordinate(study: &Study) -> Option<String> {
    let cell = crate::support::support_cell(
        &study.query,
        study.graph.class(),
        study.structure_source,
        &study.inference,
        study.refute,
    )?;
    Some(format!(
        "{}:{}:{}:{}:{}",
        cell.query, cell.graph_class, cell.structure, cell.inference, cell.validation
    ))
}

fn observation_obligations(query: &antecedent_core::CausalQuery) -> Vec<ObligationRecord> {
    observation_tags(query)
        .into_iter()
        .enumerate()
        .map(|(index, tag)| {
            ObligationRecord::new(
                format!("observation.{index}"),
                ObligationScope::Program,
                AssumptionSource::UserDeclared,
                ObligationKind::UserAssertion,
                AssumptionStatus::Declared,
                tag,
            )
        })
        .collect()
}

fn obligations_from_set(set: Option<&AssumptionSet>) -> Vec<ObligationRecord> {
    let Some(set) = set else {
        return Vec::new();
    };
    set.entries
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let kind = match record.status {
                AssumptionStatus::Declared => ObligationKind::CheckNotRun,
                AssumptionStatus::Supported => ObligationKind::EmpiricalDiagnostic,
                AssumptionStatus::Contradicted => ObligationKind::FailedCheck,
                AssumptionStatus::Untestable => ObligationKind::Uncheckable,
            };
            let kind = match (&record.source, kind) {
                (AssumptionSource::UserDeclared, ObligationKind::CheckNotRun) => {
                    ObligationKind::UserAssertion
                }
                (AssumptionSource::AlgorithmDefault { .. }, ObligationKind::CheckNotRun) => {
                    ObligationKind::GraphicalImplication
                }
                _ => kind,
            };
            ObligationRecord::new(
                format!("assumption.{index}"),
                ObligationScope::Program,
                record.source.clone(),
                kind,
                record.status,
                assumption_label(&record.assumption),
            )
            .with_assumption(record.assumption.clone())
        })
        .collect()
}

fn assumption_label(assumption: &Assumption) -> String {
    match assumption {
        Assumption::CausalMarkov => "causal_markov".into(),
        Assumption::Faithfulness => "faithfulness".into(),
        Assumption::CausalSufficiency => "causal_sufficiency".into(),
        Assumption::Consistency => "consistency".into(),
        Assumption::Positivity => "positivity".into(),
        Assumption::NoInterference => "no_interference".into(),
        Assumption::Stationarity => "stationarity".into(),
        Assumption::PiecewiseStationarity => "piecewise_stationarity".into(),
        Assumption::NoSelectionBias => "no_selection_bias".into(),
        Assumption::ExclusionRestriction { .. } => "exclusion_restriction".into(),
        Assumption::Monotonicity => "monotonicity".into(),
        Assumption::ParametricRestriction(restriction) => {
            format!("parametric:{}", restriction.id)
        }
        Assumption::PriorRestriction(restriction) => format!("prior:{}", restriction.id),
        Assumption::Custom { id, .. } => format!("custom:{id}"),
    }
}

fn result_reasoning(
    result: &StudyResult,
    prepared: &ReasoningView,
) -> Result<ReasoningView, CausalError> {
    let identification = if let Some(mixture) = &result.structural_response {
        antecedent_io::validate_mixture_masses(
            mixture.identified_mass,
            mixture.unidentified_mass,
            mixture.unevaluable_mass,
            mixture.subsampled_out_mass,
        )
        .map_err(|err| io_err(&err))?;
        IdentificationSlot::new(
            result.identification.status,
            mixture.identified_mass,
            mixture.unidentified_mass,
            mixture.unevaluable_mass,
            mixture.subsampled_out_mass,
            mixture.full_mass_scope,
            Some(Arc::from(mixture.weight_basis.as_str())),
            mixture.truncated_atoms > 0,
        )
    } else {
        IdentificationSlot::identified_singleton(result.identification.status)
    };
    let mut components = Vec::new();
    if result.estimate.se_analytic.is_finite() && result.estimate.se_analytic > 0.0 {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Sampling,
            "analytic_se",
            false,
        ));
    }
    if result.estimate.se_bootstrap.is_some() {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Sampling,
            "bootstrap_se",
            false,
        ));
    }
    if result.posterior.is_some() {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Parameter,
            "posterior",
            false,
        ));
    }
    if result.structural_response.is_some() {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Structural,
            "structural_mixture",
            false,
        ));
    }
    let uncertainty = if components.is_empty() {
        SlotAvailability::unavailable("omitted")
    } else {
        SlotAvailability::Available(UncertaintySlot::new(components))
    };
    let mut support = prepared.support.clone();
    if let SlotAvailability::Available(slot) = &mut support {
        slot.empirical = if result.refutations.is_empty() {
            SlotAvailability::unavailable("not_evaluated")
        } else {
            SlotAvailability::Available(Arc::from("evaluated"))
        };
    }
    Ok(ReasoningView::new(
        SlotAvailability::Available(identification),
        support,
        uncertainty,
        prepared.assumptions.clone(),
    ))
}

fn analysis_result_wire(
    query: &antecedent_core::CausalQuery,
    result: &StudyResult,
    temporal: Option<&CachedTemporalIdentification>,
) -> Result<AnalysisResultWire, CausalError> {
    let identification = identification_to_wire(&result.identification).map_err(|err| io_err(&err))?;
    let temporal_identification = temporal_identification_wires(temporal)?;
    let identification_variables = temporal_identification
        .iter()
        .find(|entry| entry.identification.query == identification.query)
        .map(|entry| entry.variables.clone());
    Ok(AnalysisResultWire {
        query: causal_query_to_wire(query).map_err(|err| io_err(&err))?,
        identification,
        identification_variables,
        temporal_identification,
        estimate: result.estimate.ate.is_finite().then_some(result.estimate.ate),
        standard_error: result.estimate.se_bootstrap.or_else(|| {
            result.estimate.se_analytic.is_finite().then_some(result.estimate.se_analytic)
        }),
        assumptions: antecedent_io::assumptions_to_wire(&result.estimate.assumptions),
        diagnostics: result.diagnostics.iter().map(antecedent_io::diagnostic_to_wire).collect(),
        refutations: result.refutations.iter().map(antecedent_io::refutation_to_wire).collect(),
        response: None,
        posterior_artifact: None,
        mediation_grid: None,
        structural_response: None,
    })
}

fn temporal_identification_wires(
    temporal: Option<&CachedTemporalIdentification>,
) -> Result<Vec<TemporalIdentificationWire>, CausalError> {
    let Some(temporal) = temporal else {
        return Ok(Vec::new());
    };
    temporal
        .by_horizon
        .iter()
        .map(|entry| {
            let variables = (0..entry.indexer.dense_len())
                .map(|dense| -> Result<HorizonAdjustmentNodeWire, CausalError> {
                    let key = entry.indexer.key_of(
                        u32::try_from(dense).map_err(|_| CausalError::Compile {
                            message: "temporal dense id exceeds u32".into(),
                        })?,
                    )
                    .map_err(|err| CausalError::Compile { message: err.to_string() })?;
                    Ok(HorizonAdjustmentNodeWire {
                        variable: key.variable.raw(),
                        offset: key.offset,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TemporalIdentificationWire {
                horizon: entry.horizon,
                variables,
                identification: identification_to_wire(&entry.identification)
                    .map_err(|err| io_err(&err))?,
            })
        })
        .collect()
}

fn reasoning_section(view: &ReasoningView) -> ReasoningSectionWire {
    ReasoningSectionWire {
        identification: slot_section_from_availability(&view.identification, |slot| {
            IdentificationSlotWire {
                status: slot.status.as_str().into(),
                identified_mass: slot.identified_mass,
                unidentified_mass: slot.unidentified_mass,
                unevaluable_mass: slot.unevaluable_mass,
                incomplete_search_mass: slot.incomplete_search_mass,
                full_mass_scope: slot.full_mass_scope,
                search_capped: slot.search_capped,
            }
        }),
        support: slot_section_from_availability(&view.support, |slot| SupportSlotWire {
            matrix_status: slot.matrix_status.to_string(),
            matrix_coordinate: slot
                .matrix_coordinate
                .as_ref()
                .map(std::string::ToString::to_string),
            empirical: slot.empirical.label(std::string::ToString::to_string),
        }),
        uncertainty: slot_section_from_availability(&view.uncertainty, |slot| {
            UncertaintySlotWire {
                components: slot
                    .components
                    .iter()
                    .map(|component| UncertaintyComponentWire {
                        source: component.source.as_str().into(),
                        target: component.target.to_string(),
                        omitted: component.omitted,
                    })
                    .collect(),
            }
        }),
        assumptions: slot_section_from_availability(&view.assumptions, |slot| AssumptionSlotWire {
            obligations: slot
                .obligations
                .iter()
                .map(|obligation| ObligationSectionWire {
                    id: obligation.id.to_string(),
                    scope: obligation.scope.as_str().into(),
                    kind: obligation.kind.as_str().into(),
                    status: obligation.status.as_str().into(),
                })
                .collect(),
        }),
    }
}

fn slot_section_from_availability<T, U>(
    slot: &SlotAvailability<T>,
    map: impl FnOnce(&T) -> U,
) -> SlotSectionWire<U> {
    match slot {
        SlotAvailability::Available(value) => {
            SlotSectionWire { value: Some(map(value)), unavailable: None }
        }
        SlotAvailability::Unavailable { reason } => {
            SlotSectionWire { value: None, unavailable: Some(reason.to_string()) }
        }
        _ => SlotSectionWire { value: None, unavailable: Some("unknown".into()) },
    }
}

fn claim_section(claim: &ClaimEnvelope) -> ClaimSectionWire {
    ClaimSectionWire {
        claim_id: *claim.claim_id.as_bytes(),
        kind: claim.kind.as_str().into(),
        value_bits: claim.value.filter(|value| value.is_finite()).map(f64::to_bits),
        execution: claim.execution.map(|digest| *digest.as_bytes()),
        identification_domain: claim.domains.identification.as_str().into(),
        support_domain: claim.domains.support.as_str().into(),
        evaluated_domain: claim.domains.evaluated.as_str().into(),
    }
}

fn preview_obligation(
    id: &'static str,
    kind: ObligationKind,
    check: &'static str,
    message: &'static str,
) -> ObligationRecord {
    ObligationRecord::new(
        id,
        ObligationScope::Program,
        AssumptionSource::UserDeclared,
        kind,
        AssumptionStatus::Declared,
        message,
    )
    .with_required_check(check)
}

fn claim_kind(result: &StudyResult, reasoning: &ReasoningView) -> ClaimKind {
    if result.response.is_some() {
        return ClaimKind::Response;
    }
    if let Some(mixture) = &result.structural_response {
        if mixture.unidentified_mass > 0.0 {
            return ClaimKind::Mixture;
        }
        if mixture.identified_set.is_some() {
            return ClaimKind::Bounds;
        }
        return ClaimKind::Mixture;
    }
    if matches!(
        reasoning.identification.as_ref().map(|slot| slot.status),
        Some(IdentificationStatus::NotIdentified)
    ) {
        return ClaimKind::Incomplete;
    }
    ClaimKind::Point
}
