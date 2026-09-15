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
    IDENTITY_FORMAT, IdentificationSlot, IdentificationStatus, IdentityDomain, NextAction,
    ObligationKind, ObligationRecord, ObligationScope, OperationKind, OperationReadiness,
    OperationReport, ReasoningView, SemanticApplicability, SemanticLayer, SlotAvailability,
    SupportSlot, TransformIntent, TransformationReport, UncertaintyComponent, UncertaintySlot,
    UncertaintySource, intent_effects,
};
use antecedent_data::TableView;
use antecedent_identify::{CAPPED_COMPLETION_DIAGNOSTIC_CODE, IdentificationResult};
use antecedent_io::{
    AnalysisResultContractWire, AnalysisResultWire, AssumptionSlotWire, ClaimIdentityWire,
    ClaimSectionWire, ContractIdentitiesWire, DataPartitionIdentityWire, DataSnapshotIdentityWire,
    ExecutionIdentityWire, GraphIdentityWire, HorizonAdjustmentNodeWire,
    IdentificationIdentityWire, IdentificationProductWire, IdentificationSlotWire,
    InferenceBindingWire, InferentialCommitmentsWire, ObligationSectionWire,
    ObservationIdentityWire, ProgramIdentityWire, ReasoningSectionWire, ScoreReuseIdentityWire,
    SlotSectionWire, SupportSlotWire, TargetIdentityWire, TemporalIdentificationWire,
    UncertaintyComponentWire, UncertaintySlotWire, admg_identity, causal_query_to_wire,
    claim_digest, cpdag_identity, dag_identity, data_snapshot_digest, dbn_atom_identities,
    digest_wire, encode_analysis_result_artifact_with_contract, execution_digest,
    execution_identity_from_context, identification_digest, identification_product_digest_wire,
    identification_product_wire, identification_to_wire, inference_binding_digest,
    observation_identity_wire, pag_identity, program_digest, schema_to_wire, score_reuse_digest,
    temporal_cpdag_identity, temporal_dag_identity, temporal_pag_identity,
};

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::result::StudyResult;
use crate::support::{CellStatus, StructureSource};

use super::batch::PreparedBatch;
use super::builder::DataInput;
use super::execute::Study;
use super::prepared::{
    CachedTemporalHorizonIdentification, CachedTemporalIdentification, PreparedStudy,
};

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

    /// Capability report for cheap inspect / classify of this contract.
    #[must_use]
    pub fn capability(&self) -> OperationReport {
        let operation = if self.identities.identification_product.is_some() {
            OperationKind::Execute
        } else {
            OperationKind::Inspect
        };
        self.capability_for(operation)
    }

    /// Capability report for `operation`. Scoped ops do not invent matrix rows.
    #[must_use]
    pub fn capability_for(&self, operation: OperationKind) -> OperationReport {
        operation_report(self, operation)
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
            reasoning: reasoning_section(claim.map_or(&self.reasoning, |claim| &claim.reasoning)),
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

    /// Capability report from cheap inspect. Does not identify.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn capability(&self) -> Result<OperationReport, CausalError> {
        Ok(self.inspect()?.capability())
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

    /// Score-table reuse key. Stricter than identification: folds, rows,
    /// nuisance provenance, and the snapshot are part of the digest.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn score_reuse_identity(
        &self,
    ) -> Result<Option<antecedent_core::SemanticDigest>, CausalError> {
        let Some(table) = self.score_table() else {
            return Ok(None);
        };
        let identities = self.contract()?.identities;
        let wire = ScoreReuseIdentityWire::score_table(
            identities.identification,
            identities.data_snapshot,
            &table.row_index,
            &table.fold_ids,
            table.n_folds,
            &table.adjustment_set,
            table.nuisance_provenance.as_ref(),
            table.treatment,
            &table.intervened,
            self.study().bootstrap_replicates,
        );
        Ok(Some(score_reuse_digest(&wire).map_err(|err| io_err(&err))?))
    }

    /// Shared-design key: folds and covariates, not per-query nuisance fits.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn batch_share_identity(
        &self,
    ) -> Result<Option<antecedent_core::SemanticDigest>, CausalError> {
        let Some(shared) = self.shared_design() else {
            return Ok(None);
        };
        let identities = self.contract()?.identities;
        let adjustment: &[antecedent_core::VariableId] =
            shared.covariate.as_ref().map_or(&[], |cov| cov.adjustment_set.as_ref());
        let wire = ScoreReuseIdentityWire::batch_share(
            identities.data_snapshot,
            &shared.fold_ids,
            shared.n_folds,
            adjustment,
        );
        Ok(Some(score_reuse_digest(&wire).map_err(|err| io_err(&err))?))
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
        let lagged = self
            .temporal_identification()
            .cloned()
            .or_else(|| dbn_projected_temporal_identification(self.study()))
            .or_else(|| class_projected_temporal_identification(self.study()));
        let mut body = analysis_result_wire(
            self.query(),
            result,
            lagged.as_ref(),
            cached_identification(self.study()),
        )?;
        if let Some(product) = &section.identification_product {
            body.identification.status.clone_from(&product.status);
            body.identification.estimands.clone_from(&product.estimands);
            body.identification.arena.clone_from(&product.arena);
            body.identification.derivation.clone_from(&product.derivation);
            body.identification.required_assumptions.clone_from(&product.required_assumptions);
        }
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

    /// Capability report projected from the prepared contract.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures while reading the contract.
    pub fn capability(&self) -> Result<OperationReport, CausalError> {
        Ok(self.contract()?.capability_for(OperationKind::Execute))
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
                message: crate::error::RANK_DESIGNS_REQUIRES_LICENSE,
            });
        }
        if contract.identities.identification_product.is_none() {
            return Err(CausalError::Unsupported {
                message: crate::error::RANK_DESIGNS_REQUIRES_PRODUCT,
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

impl PreparedBatch {
    /// Shared-design key for the batch. Equal across plans; not a nuisance key.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn batch_share_identity(
        &self,
    ) -> Result<Option<antecedent_core::SemanticDigest>, CausalError> {
        match self.plans().first() {
            Some(plan) => plan.batch_share_identity(),
            None => Ok(None),
        }
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
            ClaimKind::Point => executed_scalar(self),
            _ => None,
        };
        let execution =
            execution_digest(&execution_identity_from_context(ctx)).map_err(|err| io_err(&err))?;
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
            [
                Arc::from(format!("snapshot:{}", contract.identities.data_snapshot.to_hex())),
                Arc::from(format!(
                    "identification:{}",
                    contract.identities.identification.to_hex()
                )),
                Arc::from(format!("program:{}", contract.identities.program.to_hex())),
            ],
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
        analysis_result_wire(query, self, None, None)
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
        Some(wire) => Some(identification_product_digest_wire(wire).map_err(|err| io_err(&err))?),
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
            InferenceMode::Bayesian(cfg) => cfg
                .prior_mapping
                .as_ref()
                .map(prior_mapping_tag)
                .or_else(|| cfg.prior_artifact.as_ref().map(|_| "prior_artifact".into())),
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
            let mut tags: Vec<String> =
                query.observation_assumptions.iter().map(observation_assumption_tag).collect();
            if let Some(spec) = observation_spec_tag(&query.observation) {
                tags.push(spec);
            }
            tags
        }
        _ => Vec::new(),
    }
}

fn observation_spec_tag(spec: &antecedent_core::ObservationSpec) -> Option<String> {
    match spec {
        antecedent_core::ObservationSpec::Complete => None,
        antecedent_core::ObservationSpec::RightCensored { .. } => Some("right_censored".into()),
        antecedent_core::ObservationSpec::LeftCensored { .. } => Some("left_censored".into()),
        antecedent_core::ObservationSpec::IntervalCensored { .. } => {
            Some("interval_censored".into())
        }
        antecedent_core::ObservationSpec::Truncated { .. } => Some("truncated".into()),
        antecedent_core::ObservationSpec::Selected { .. } => Some("selected".into()),
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
        let names: Vec<String> = data_schema(&study.data)
            .variables()
            .iter()
            .map(|variable| variable.name.to_string())
            .collect();
        return Ok(GraphIdentityWire::GraphPosterior {
            graph_class: study.graph.class().as_str().into(),
            n_atoms: u64::try_from(posterior.n_graphs).unwrap_or(u64::MAX),
            atoms: dbn_atom_identities(posterior, &names).map_err(|err| io_err(&err))?,
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
    if let Some(cache) = study.dbn_posterior_identification_cache.as_ref() {
        return cache.atoms.first().map(|atom| &atom.identification);
    }
    None
}

/// Project a DBN posterior atom onto the existing temporal-namespace owner.
///
/// Graph-posterior Pulse stores lagged-node identification on
/// `dbn_posterior_identification_cache`, not `temporal_identification_cache`.
/// Encode reuses [`HorizonAdjustmentNodeWire`] so unfolded IDs stay inside a
/// declared variable+offset namespace.
fn dbn_projected_temporal_identification(study: &Study) -> Option<CachedTemporalIdentification> {
    let cache = study.dbn_posterior_identification_cache.as_ref()?;
    let atom = cache.atoms.first()?;
    let horizon = query_horizon_steps(&study.query);
    Some(CachedTemporalIdentification {
        by_horizon: Arc::from([CachedTemporalHorizonIdentification {
            horizon,
            identification: atom.identification.clone(),
            estimand: atom.estimand.clone(),
            indexer: atom.indexer.clone(),
        }]),
    })
}

/// Project a TemporalCpdag/Pag envelope indexer onto the same namespace owner.
fn class_projected_temporal_identification(study: &Study) -> Option<CachedTemporalIdentification> {
    let cache = study.temporal_class_identification_cache.as_ref()?;
    let (horizon, envelope) = cache.by_horizon.first().map_or_else(
        || (query_horizon_steps(&study.query), &cache.envelope),
        |(horizon, envelope)| (*horizon, envelope),
    );
    let indexer = envelope.indexers.first()?.clone();
    let case = envelope.envelope.cases.iter().find(|case| !case.result.estimands.is_empty())?;
    Some(CachedTemporalIdentification {
        by_horizon: Arc::from([CachedTemporalHorizonIdentification {
            horizon,
            identification: case.result.clone(),
            estimand: case.result.estimands[0].clone(),
            indexer,
        }]),
    })
}

fn query_horizon_steps(query: &antecedent_core::CausalQuery) -> u32 {
    match query {
        antecedent_core::CausalQuery::TemporalEffect(query) => query.horizon_steps,
        antecedent_core::CausalQuery::Response(query) => {
            query.temporal.as_ref().and_then(|spec| spec.horizons.first()).copied().unwrap_or(1)
        }
        _ => 1,
    }
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
    let mut obligations = obligations_from_set(cached.map(|result| &result.required_assumptions));
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
    Some(crate::support::cell_coordinate(cell))
}

fn operation_report(contract: &CausalContract, operation: OperationKind) -> OperationReport {
    let coordinate =
        contract.reasoning.support.as_ref().and_then(|slot| slot.matrix_coordinate.clone());
    let cell = coordinate.as_deref().and_then(crate::support::support_cell_from_coordinate);
    let unmet: Vec<ObligationRecord> = contract
        .reasoning
        .assumptions
        .as_ref()
        .map(|slot| slot.obligations.iter().filter(|o| o.is_unresolved()).cloned().collect())
        .unwrap_or_default();
    let (applicability, mut blockers) = match contract.support_status {
        Some(CellStatus::Licensed) => (SemanticApplicability::Licensed, Vec::new()),
        Some(CellStatus::Refused) => {
            let reason = cell.map_or(
                "cell is not licensed (parity/support_licensed.toml) and is not n/a; it is refused.",
                crate::support::refused_message,
            );
            (
                SemanticApplicability::Unlicensed,
                vec![antecedent_core::BlockedOperation::refused(reason)],
            )
        }
        Some(CellStatus::NotApplicable { reason }) => (
            SemanticApplicability::Impossible,
            vec![antecedent_core::BlockedOperation::not_applicable(reason)],
        ),
        Some(CellStatus::Allowlisted { reason, .. }) => (
            SemanticApplicability::Unlicensed,
            vec![antecedent_core::BlockedOperation::refused(reason)],
        ),
        None => {
            (SemanticApplicability::Unknown, vec![antecedent_core::BlockedOperation::off_axis()])
        }
    };
    let neighbors = match contract.support_status {
        Some(CellStatus::Refused | CellStatus::NotApplicable { .. }) => {
            cell.map(crate::support::licensed_neighbors).unwrap_or_default()
        }
        _ => Vec::new(),
    };

    if !operation.uses_matrix_row() && applicability != SemanticApplicability::Licensed {
        let reason = match operation {
            OperationKind::RankDesigns => crate::error::RANK_DESIGNS_REQUIRES_LICENSE,
            OperationKind::Retarget => "retarget requires a licensed prepared contract",
            OperationKind::Export => "export requires a licensed prepared contract",
            _ => "operation requires a licensed prepared contract",
        };
        blockers = vec![antecedent_core::BlockedOperation::operation_unlicensed(reason)];
    } else if operation == OperationKind::RankDesigns
        && contract.identities.identification_product.is_none()
    {
        blockers.push(antecedent_core::BlockedOperation::binding("identification_product"));
    }

    let readiness = if !blockers.is_empty() {
        if blockers.iter().any(|blocker| blocker.id.as_ref().starts_with("binding.")) {
            Some(OperationReadiness::BindingMissing)
        } else if applicability == SemanticApplicability::Unknown {
            Some(OperationReadiness::UnknownSupport)
        } else {
            None
        }
    } else if unmet.iter().any(|o| o.kind == ObligationKind::FailedCheck) {
        Some(OperationReadiness::EmpiricalSupportFailed)
    } else if unmet.iter().any(|o| o.kind == ObligationKind::CheckNotRun) {
        Some(OperationReadiness::EmpiricalCheckPending)
    } else {
        Some(OperationReadiness::Executable)
    };

    let next_actions: Vec<NextAction> =
        if applicability == SemanticApplicability::Licensed && blockers.is_empty() {
            Vec::new()
        } else if neighbors.is_empty() {
            vec![NextAction::new(
                "no_licensed_neighbor",
                "no justified licensed neighbor; do not relabel the graph class",
            )]
        } else {
            vec![NextAction::new(
                "consider_licensed_neighbor",
                "choose an explicit licensed neighbor; never an automatic fallback",
            )]
        };

    let (preserved, invalidated) =
        if applicability == SemanticApplicability::Licensed && blockers.is_empty() {
            (
                vec![
                    SemanticLayer::Target,
                    SemanticLayer::Identification,
                    SemanticLayer::Program,
                    SemanticLayer::Support,
                ],
                Vec::new(),
            )
        } else {
            (
                vec![SemanticLayer::Target],
                vec![SemanticLayer::Support, SemanticLayer::Identification, SemanticLayer::Results],
            )
        };

    OperationReport::new(
        operation,
        applicability,
        readiness,
        coordinate,
        blockers,
        unmet,
        preserved,
        invalidated,
        next_actions,
        neighbors,
    )
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
    let identification = identification_slot_from_result(result)?;
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
    if result.structural_response.is_some()
        || identification.weight_basis.is_some()
        || matches!(
            identification.status,
            IdentificationStatus::GraphDependent | IdentificationStatus::PartiallyIdentified
        )
    {
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

fn executed_scalar(result: &StudyResult) -> Option<f64> {
    if result.response.is_some() {
        return None;
    }
    if let Some(distribution) = &result.distribution {
        return distribution.mean.is_finite().then_some(distribution.mean);
    }
    if let Some(counterfactual) = &result.counterfactual {
        return counterfactual.mean_ite.is_finite().then_some(counterfactual.mean_ite);
    }
    result.estimate.ate.is_finite().then_some(result.estimate.ate)
}

fn analysis_result_wire(
    query: &antecedent_core::CausalQuery,
    result: &StudyResult,
    temporal: Option<&CachedTemporalIdentification>,
    cached: Option<&IdentificationResult>,
) -> Result<AnalysisResultWire, CausalError> {
    let identification = identification_to_wire(cached.unwrap_or(&result.identification))
        .map_err(|err| io_err(&err))?;
    let temporal_identification = temporal_identification_wires(temporal)?;
    let identification_variables = temporal_identification
        .iter()
        .find(|entry| entry.identification.query == identification.query)
        .or_else(|| temporal_identification.first())
        .map(|entry| entry.variables.clone());
    Ok(AnalysisResultWire {
        query: causal_query_to_wire(query).map_err(|err| io_err(&err))?,
        identification,
        identification_variables,
        temporal_identification,
        estimate: executed_scalar(result),
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
                    let key = entry
                        .indexer
                        .key_of(u32::try_from(dense).map_err(|_| CausalError::Compile {
                            message: "temporal dense id exceeds u32".into(),
                        })?)
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

fn identification_slot_from_result(
    result: &StudyResult,
) -> Result<IdentificationSlot, CausalError> {
    if let Some(mixture) = &result.structural_response {
        antecedent_io::validate_mixture_masses(
            mixture.identified_mass,
            mixture.unidentified_mass,
            mixture.unevaluable_mass,
            mixture.subsampled_out_mass,
        )
        .map_err(|err| io_err(&err))?;
        return Ok(IdentificationSlot::new(
            result.identification.status,
            mixture.identified_mass,
            mixture.unidentified_mass,
            mixture.unevaluable_mass,
            mixture.subsampled_out_mass,
            mixture.full_mass_scope,
            Some(Arc::from(mixture.weight_basis.as_str())),
            mixture.truncated_atoms > 0,
        ));
    }
    if let Some((unidentified, incomplete, basis)) = projected_mixture_masses(result) {
        let identified = (1.0 - unidentified - incomplete).max(0.0);
        antecedent_io::validate_mixture_masses(identified, unidentified, 0.0, incomplete)
            .map_err(|err| io_err(&err))?;
        return Ok(IdentificationSlot::new(
            result.identification.status,
            identified,
            unidentified,
            0.0,
            incomplete,
            incomplete == 0.0,
            Some(Arc::from(basis)),
            incomplete > 0.0,
        ));
    }
    Ok(IdentificationSlot::identified_singleton(result.identification.status))
}

fn projected_mixture_masses(result: &StudyResult) -> Option<(f64, f64, &'static str)> {
    let basis = match result.structure_source {
        StructureSource::GraphPosterior => "posterior_probability",
        _ => "completion_enumeration",
    };
    if let Some(posterior) = &result.posterior {
        if posterior.unidentified_mass > 0.0
            || posterior.subsampled_out_mass > 0.0
            || matches!(
                result.identification.status,
                IdentificationStatus::GraphDependent | IdentificationStatus::PartiallyIdentified
            )
        {
            return Some((posterior.unidentified_mass, posterior.subsampled_out_mass, basis));
        }
    }
    if matches!(
        result.identification.status,
        IdentificationStatus::GraphDependent | IdentificationStatus::PartiallyIdentified
    ) {
        let unidentified = unidentified_mass_from_diagnostics(&result.diagnostics).unwrap_or(0.0);
        return Some((unidentified, 0.0, basis));
    }
    None
}

fn unidentified_mass_from_diagnostics(diagnostics: &[antecedent_core::Diagnostic]) -> Option<f64> {
    diagnostics.iter().find_map(|diagnostic| {
        let unidentified = parse_mass_field(&diagnostic.message, "unidentified_mass=")?;
        if let Some(identified) = parse_mass_field(&diagnostic.message, "identified_mass=") {
            let total = identified + unidentified;
            if total > 0.0 {
                return Some(unidentified / total);
            }
        }
        Some(unidentified)
    })
}

fn parse_mass_field(message: &str, field: &str) -> Option<f64> {
    message.split(field).nth(1).and_then(|rest| {
        rest.split(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E')
            .next()
            .and_then(|token| token.parse().ok())
    })
}

fn claim_kind(result: &StudyResult, reasoning: &ReasoningView) -> ClaimKind {
    if result.response.is_some() {
        return ClaimKind::Response;
    }
    if let Some(slot) = reasoning.identification.as_ref() {
        if slot.status == IdentificationStatus::NotIdentified {
            return ClaimKind::Incomplete;
        }
        if slot.unidentified_mass > 0.0 || slot.weight_basis.is_some() {
            if result
                .structural_response
                .as_ref()
                .is_some_and(|mixture| mixture.identified_set.is_some())
            {
                return ClaimKind::Bounds;
            }
            return ClaimKind::Mixture;
        }
    }
    ClaimKind::Point
}
