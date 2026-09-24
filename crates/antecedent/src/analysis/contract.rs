//! Causal-contract companion for [`Study`] / [`PreparedStudy`].
//!
//! ADR 0022: the practitioner handle stays [`PreparedStudy`]. This module
//! builds an inspectable contract from existing products. `inspect` is cheap
//! structural classification; `contract` includes cached identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;
use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionSet, AssumptionSlot, AssumptionSource, AssumptionStatus,
    AttestedEvidence, CalibrationView, CausalQuery, CausalSchema, ClaimDomains, ClaimEnvelope,
    ClaimKind, ContractIdentities, DomainStatus, ExecutionContext, IDENTITY_FORMAT,
    IdentificationSlot, IdentificationStatus, IdentityDomain, IntervalInterpretation,
    IntervalMethod, NextAction, ObligationKind, ObligationRecord, ObligationScope, OperationKind,
    OperationReadiness, OperationReport, ReasoningView, ResponseUncertainty, SemanticApplicability,
    SemanticLayer, SlotAvailability, SupportSlot, TargetPopulation, TransformIntent,
    TransformationReport, UncertaintyComponent, UncertaintySlot, UncertaintySource, intent_effects,
};
use antecedent_data::TableView;
use antecedent_identify::{
    CAPPED_COMPLETION_DIAGNOSTIC_CODE, IdentificationEnvelope, IdentificationResult,
};
use antecedent_io::{
    AnalysisResultContractWire, AnalysisResultWire, AssumptionSlotWire, AttestedEvidenceWire,
    CalibrationSlotWire, CausalQueryWire, ClaimIdentityWire, ClaimSectionWire,
    ContractIdentitiesWire, DataPartitionIdentityWire, DataSnapshotIdentityWire,
    ExecutionIdentityWire, GraphIdentityWire, HorizonAdjustmentNodeWire,
    IdentificationEnvelopeWire, IdentificationIdentityWire, IdentificationProductWire,
    IdentificationResultWire, IdentificationSlotWire, InferenceBindingWire,
    InferentialCommitmentsWire, ObligationSectionWire, ObservationIdentityWire,
    ProgramIdentityWire, ReasoningSectionWire, ScoreReuseIdentityWire, SlotSectionWire,
    SupportSlotWire, TargetIdentityWire, TargetWeightsIdentityWire, TargetWeightsSectionWire,
    TemporalIdentificationWire, UncertaintyComponentWire, UncertaintySlotWire, admg_identity,
    causal_query_to_wire_with_registry, claim_digest, cpdag_identity, dag_identity,
    data_snapshot_digest, digest_wire, encode_analysis_result_artifact_with_contract,
    execution_digest, execution_identity_from_context, graph_posterior_atom_identities,
    identification_digest, identification_product_digest_wire, identification_product_wire,
    identification_to_wire_with_registry, inference_binding_digest, observation_identity_wire,
    pag_identity, program_digest, schema_to_wire, score_reuse_digest, target_weights_digest,
    temporal_cpdag_identity, temporal_dag_identity, temporal_pag_identity,
};

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::result::{RowWeightsBinding, StudyResult};
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
    /// Estimator the prepared plan resolved (`logical_plan.estimator`);
    /// `None` for cheap inspection, which compiles no plan.
    pub resolved_estimator: Option<Arc<str>>,
    /// Data snapshot row count used for calibration scope.
    pub row_count: u64,
    /// Data modality of the execution, the same string the data snapshot
    /// carries (`tabular`, `series`, `event`, `panel`, `multi_env`). Read by
    /// the calibration match key, so a record binds only to the modality it
    /// was measured on.
    pub modality: Arc<str>,
    /// Inference family used to compile the program (`frequentist` / `bayesian`).
    pub inference: Arc<str>,
    /// Licensed query name used for calibration matching.
    pub query_kind: Arc<str>,
    /// Score-reuse identity this handle exports, when it holds a score table.
    pub score_reuse: Option<antecedent_core::SemanticDigest>,
    /// Target-weights identity of a row-weight retarget, when the result is one.
    pub target_weights: Option<antecedent_core::SemanticDigest>,
    /// Digest of the executed checked AIPW complete-case rows, when exported.
    checked_aipw_rows: Option<[u8; 32]>,
    /// Posterior construction label used for calibration matching
    /// (`<backend>.<likelihood>.<prior>`); empty for Frequentist programs.
    pub posterior: Arc<str>,
    /// Functional label used for calibration matching: the target population,
    /// outcome functional, contrast, horizon and policy that the support axis
    /// query name does not distinguish.
    pub functional: Arc<str>,
    /// Posterior draws the Bayesian program asked for. Used as the calibration
    /// scope's draw count when the result keeps no posterior artifact (a
    /// Bayesian response's band summarizes draws it does not retain).
    pub posterior_draws: Option<u32>,
    /// What an executed body reports about this program: the target query and
    /// the identification certificate the program was compiled with.
    pub(crate) body: Arc<BodyFrame>,
}

/// Program-side part of an executed `analysis_result` body.
///
/// One owner for the body a claim digests and the body an export writes, so
/// the two can never differ.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BodyFrame {
    /// Target query.
    query: CausalQueryWire,
    /// Identification certificate the program was compiled with (cached
    /// products, mixture status aggregated), in its original query namespace.
    identification: Option<IdentificationResultWire>,
    /// Status of [`Self::identification`].
    identification_status: Option<IdentificationStatus>,
    /// Horizon-specific identification and unfolded variable namespaces.
    temporal_identification: Vec<TemporalIdentificationWire>,
    /// Bindings for named or custom populations the query references.
    registry: Option<antecedent_core::PopulationRegistry>,
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

    /// Advertised identities, including the reuse layers this export carries.
    fn identities_wire(
        &self,
        execution: Option<[u8; 32]>,
    ) -> Result<ContractIdentitiesWire, CausalError> {
        let mut identities =
            ContractIdentitiesWire::try_from(&self.identities).map_err(|err| io_err(&err))?;
        identities.execution = execution;
        identities.score_reuse = self.score_reuse.map(|digest| *digest.as_bytes());
        identities.target_weights = self.target_weights.map(|digest| *digest.as_bytes());
        identities.checked_aipw_rows = self.checked_aipw_rows;
        Ok(identities)
    }

    /// Seal over this contract's identities, `reasoning`, and audit fields.
    ///
    /// `execution` is the digest the claim being sealed reports, so a claim id
    /// and the section it is exported in always seal the same identities.
    fn seal(
        &self,
        reasoning: &ReasoningSectionWire,
        execution: Option<[u8; 32]>,
    ) -> Result<[u8; 32], CausalError> {
        antecedent_io::contract_seal(
            &self.identities_wire(execution)?,
            reasoning,
            self.graph_class.as_str(),
            self.structure_source.as_str(),
            self.identifier.as_deref(),
            self.estimator.as_deref(),
        )
        .map_err(|err| io_err(&err))
    }

    fn section_from_payloads(
        &self,
        payloads: &ContractPayloads,
        claim: Option<&ClaimEnvelope>,
        execution: Option<&ExecutionIdentityWire>,
        reuse: &ReuseSection,
    ) -> Result<AnalysisResultContractWire, CausalError> {
        let reasoning = reasoning_section(claim.map_or(&self.reasoning, |claim| &claim.reasoning));
        let program = &payloads.program;
        let executed = claim.and_then(|claim| claim.execution).map(|digest| *digest.as_bytes());
        // The seal covers the identities, so the execution and reuse layers it
        // advertises are sealed and the claim id covers them like every other
        // identity.
        Ok(AnalysisResultContractWire {
            format: antecedent_io::CONTRACT_SECTION_FORMAT,
            identities: self.identities_wire(executed)?,
            seal: self.seal(&reasoning, executed)?,
            target: program.target.clone(),
            reasoning,
            graph_class: self.graph_class.as_str().into(),
            structure_source: self.structure_source.as_str().into(),
            identifier: self.identifier.as_ref().map(std::string::ToString::to_string),
            estimator: self.estimator.as_ref().map(std::string::ToString::to_string),
            claim: claim.map(claim_section),
            identification: Some(program.identification.clone()),
            identification_product: program.identification_product.clone(),
            program: Some(program.program.clone()),
            inference_binding: Some(program.inference_binding.clone()),
            observation: Some(program.observation.clone()),
            data_snapshot: Some(payloads.data_snapshot.clone()),
            execution: execution.cloned(),
            score_reuse: reuse.score_reuse.clone(),
            target_weights: reuse.target_weights.clone(),
            checked_aipw_rows: reuse.checked_aipw_rows.clone(),
        })
    }
}

/// Score-reuse and row-weight layers an export carries beside the contract.
///
/// Both travel with their rehashable payload so an independent consumer can
/// re-derive them; both are advertised in the identities, hence under the seal
/// and the claim id.
#[derive(Default)]
struct ReuseSection {
    score_reuse: Option<ScoreReuseIdentityWire>,
    target_weights: Option<TargetWeightsSectionWire>,
    checked_aipw_rows: Option<antecedent_io::CheckedAipwRowsWire>,
}

/// Refusal when row weights meet a snapshot or score table they do not index.
///
/// Built by [`crate::unsupported_reason!`] so the code is checked against
/// `parity/reason_codes.toml` at compile time instead of being spelled as a
/// raw `reason=…` prefix.
macro_rules! row_weights_bound_to_snapshot {
    () => {
        crate::unsupported_reason!(
            "row_weights_bound_to_snapshot",
            "row weights index the rows of the data snapshot and score table they were bound \
             to; this handle holds a different one"
        )
    };
}

impl Study {
    /// Cheap structural inspection. Does not identify, fit, or execute.
    ///
    /// Identification-product and uncertainty slots are explicitly
    /// unavailable, whatever a prepared handle has cached on this study.
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
    /// Cheap inspect of the bound study.
    ///
    /// Identification products are unavailable even when this handle has
    /// cached them (ADR 0022, "Cheap inspection versus identification"); the
    /// record is the one [`Study::inspect`] returns before prepare. Use
    /// [`Self::contract`] for the prepared products.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn inspect(&self) -> Result<CausalContract, CausalError> {
        self.study().inspect()
    }

    /// Contract built from cached identification products.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn contract(&self) -> Result<CausalContract, CausalError> {
        Ok(compile_with_payloads(self.study(), Some(self), Some(self.program_payloads()?))?.0)
    }

    /// Program-side payloads of the bound study, compiled once per handle
    /// state: data-independent, so every estimate click reuses them.
    fn program_payloads(&self) -> Result<Arc<ProgramPayloads>, CausalError> {
        if let Some(cached) = self.program_cache().get() {
            return Ok(Arc::clone(cached));
        }
        let compiled = Arc::new(program_payloads_for(
            self.study(),
            self.plan().logical.record.estimator.as_deref(),
            self.checked_aipw_ate(),
            self.checked_frontdoor_linear(),
            self.checked_iv(),
            self.checked_linear_operation(),
            self.checked_functional_effect_program(),
        )?);
        Ok(Arc::clone(self.program_cache().get_or_init(|| compiled)))
    }

    /// Contract a result from this handle is bound to: the bound study with
    /// the execution's validation suite and target population (a retarget),
    /// on the handle's current data.
    fn study_for_execution(
        &self,
        refute: crate::RefuteSuite,
        population: Option<&TargetPopulation>,
    ) -> Option<Study> {
        if refute == self.study().refute && population.is_none() {
            return None;
        }
        let mut study = self.study().clone();
        study.refute = refute;
        if let (Some(population), Some(target)) = (population, study.query.target_population_mut())
        {
            *target = population.clone();
        }
        Some(study)
    }

    /// Stamp the contract a result was executed under, on `data`.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub(crate) fn executed_contract(
        &self,
        data: &DataInput,
        refute: crate::RefuteSuite,
        population: Option<&TargetPopulation>,
    ) -> Result<crate::result::ExecutedContract, CausalError> {
        let program = match self.study_for_execution(refute, population) {
            None => self.program_payloads()?,
            Some(study) => Arc::new(program_payloads_for(
                &study,
                self.plan().logical.record.estimator.as_deref(),
                if population.is_none() { self.checked_aipw_ate() } else { None },
                if population.is_none() { self.checked_frontdoor_linear() } else { None },
                if population.is_none() { self.checked_iv() } else { None },
                if population.is_none() { self.checked_linear_operation() } else { None },
                if population.is_none() { self.checked_functional_effect_program() } else { None },
            )?),
        };
        let snapshot = data_snapshot_wire(
            data,
            self.study().interference.as_ref(),
            &program.observation_digest,
        )?;
        let mut snapshot = snapshot;
        if let DataInput::Tabular(tabular) = data {
            let laws = self
                .distribution_factor_snapshot(tabular)?
                .or(self.functional_effect_factor_snapshot(tabular)?);
            if let Some(laws) = laws {
                snapshot.distribution_factor_laws = Some(
                    antecedent_io::distribution_factor_laws_to_wire(&laws)
                        .map_err(|err| io_err(&err))?,
                );
            }
        }
        let snapshot = data_snapshot_digest(&snapshot).map_err(|err| io_err(&err))?;
        Ok(crate::result::ExecutedContract { identities: program.identities(snapshot), refute })
    }

    /// Contract and claim of an execution of this handle, exactly as
    /// [`Self::encode_contracted_result`] exports them.
    ///
    /// A row-weight retarget reports its `RowWeights` target, never this
    /// handle's original population.
    ///
    /// # Errors
    ///
    /// See [`Self::contract_for_result`]; canonical-encoding failures.
    pub fn execution_contract(
        &self,
        result: &StudyResult,
        ctx: &ExecutionContext,
    ) -> Result<(CausalContract, ClaimEnvelope), CausalError> {
        let (mut contract, _payloads) = self.contract_and_payloads_for_result(result)?;
        self.bind_reuse(&mut contract, result)?;
        let claim = result.claim(&contract, ctx)?;
        Ok((contract, claim))
    }

    /// The contract `result` was executed under, recompiled on this handle.
    ///
    /// Refuses a result that no prepared handle executed, a result from
    /// another program, and a result computed on a different data snapshot
    /// than the one this handle now binds.
    ///
    /// # Errors
    ///
    /// [`CausalError::Conflict`] on any identity mismatch; canonical-encoding
    /// failures.
    pub fn contract_for_result(&self, result: &StudyResult) -> Result<CausalContract, CausalError> {
        Ok(self.contract_and_payloads_for_result(result)?.0)
    }

    fn contract_and_payloads_for_result(
        &self,
        result: &StudyResult,
    ) -> Result<(CausalContract, ContractPayloads), CausalError> {
        let executed = result.executed_contract.as_ref().ok_or(CausalError::Conflict {
            what: "result",
            detail: "result was not executed by a prepared handle",
        })?;
        let retargeted = result.retarget_population();
        let compiled = match self.study_for_execution(executed.refute, retargeted.as_ref()) {
            None => {
                compile_with_payloads(self.study(), Some(self), Some(self.program_payloads()?))?
            }
            Some(study) => compile_with_payloads(&study, Some(self), None)?,
        };
        if let Some(binding) = &result.row_weights {
            // Row weights index the rows of one snapshot and one score table.
            // Re-derive the binding here, before the identity comparison, so
            // the refusal names the weights rather than the snapshot layer.
            let rebound = self.row_weights_binding(&binding.weights, &binding.depends_on)?;
            if rebound.identity.data_snapshot != binding.identity.data_snapshot
                || rebound.identity.score_reuse != binding.identity.score_reuse
            {
                return Err(row_weights_bound_to_snapshot!());
            }
            if rebound != *binding {
                return Err(CausalError::Conflict {
                    what: "target_weights",
                    detail: "row weights do not match their recorded target-weights identity",
                });
            }
        }
        let ours = compiled.0.identities;
        let theirs = executed.identities;
        if ours.data_snapshot != theirs.data_snapshot {
            return Err(CausalError::Conflict {
                what: "data_snapshot",
                detail: "result was computed on a different data snapshot than this handle binds",
            });
        }
        if ours != theirs {
            return Err(CausalError::Conflict {
                what: "program",
                detail: "result was not executed under this prepared contract",
            });
        }
        Ok(compiled)
    }

    /// Score-table reuse key. Stricter than identification: folds, rows,
    /// nuisance provenance, the inference binding (estimator options, overlap, backend)
    /// and the snapshot are part of the digest.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn score_reuse_identity(
        &self,
    ) -> Result<Option<antecedent_core::SemanticDigest>, CausalError> {
        if self.score_table().is_none() {
            return Ok(None);
        }
        let identities = self.contract()?.identities;
        self.score_reuse_wire(&identities)
            .map(|wire| score_reuse_digest(&wire).map_err(|err| io_err(&err)))
            .transpose()
    }

    /// Score-reuse payload for this handle's score table under `identities`.
    fn score_reuse_wire(&self, identities: &ContractIdentities) -> Option<ScoreReuseIdentityWire> {
        let table = self.score_table()?;
        Some(ScoreReuseIdentityWire::score_table(
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
            identities.inference_binding,
        ))
    }

    /// The reuse layers this handle exports beside a result, advertised on the
    /// contract so the seal and the claim id cover them.
    fn bind_reuse(
        &self,
        contract: &mut CausalContract,
        result: &StudyResult,
    ) -> Result<ReuseSection, CausalError> {
        let mut reuse = ReuseSection::default();
        if let Some(score) = self.score_reuse_wire(&contract.identities) {
            contract.score_reuse = Some(score_reuse_digest(&score).map_err(|err| io_err(&err))?);
            reuse.score_reuse = Some(score);
        }
        if let Some(binding) = &result.row_weights {
            contract.target_weights = Some(binding.target_weights);
            reuse.target_weights = Some(TargetWeightsSectionWire {
                identity: binding.identity.clone(),
                values: binding.weights.to_vec(),
            });
        }
        if self.checked_aipw_ate().is_some() {
            let table =
                result.estimate.score_table.as_ref().ok_or_else(|| CausalError::Compile {
                    message: "checked AIPW execution did not retain its complete-case score rows"
                        .into(),
                })?;
            let rows = antecedent_io::CheckedAipwRowsWire {
                format: 1,
                rows: table.row_index.to_vec(),
                data_snapshot: *contract.identities.data_snapshot.as_bytes(),
            };
            contract.checked_aipw_rows =
                Some(antecedent_io::checked_aipw_rows_digest(&rows).map_err(|err| io_err(&err))?);
            reuse.checked_aipw_rows = Some(rows);
        }
        Ok(reuse)
    }

    /// Bind row weights to this handle's data snapshot and score table.
    ///
    /// # Errors
    ///
    /// No score table, or canonical-encoding failures.
    pub(crate) fn row_weights_binding(
        &self,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
    ) -> Result<RowWeightsBinding, CausalError> {
        let identities = self.contract()?.identities;
        let score = self.score_reuse_wire(&identities).ok_or(CausalError::Unsupported {
            message: "row-weight target requires a prepared score table",
        })?;
        let score = score_reuse_digest(&score).map_err(|err| io_err(&err))?;
        let identity =
            TargetWeightsIdentityWire::new(weights, identities.data_snapshot, score, depends_on);
        let target_weights = target_weights_digest(&identity).map_err(|err| io_err(&err))?;
        Ok(RowWeightsBinding {
            weights: Arc::from(weights),
            depends_on: Arc::from(depends_on),
            identity,
            target_weights,
        })
    }

    /// Re-execute the row-weight retarget an exported contract carries.
    ///
    /// Row weights index the rows of one data snapshot and one score table:
    /// this handle must hold that snapshot and that table.
    ///
    /// # Errors
    ///
    /// `reason=row_weights_bound_to_snapshot` when this handle holds a
    /// different snapshot or score table; otherwise the refusals of
    /// [`Self::retarget`].
    pub fn reexecute_retarget(
        &self,
        section: &TargetWeightsSectionWire,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let depends_on: Vec<antecedent_core::VariableId> = section
            .identity
            .depends_on
            .iter()
            .copied()
            .map(antecedent_core::VariableId::from_raw)
            .collect();
        let bound = self.row_weights_binding(&section.values, &depends_on)?;
        if bound.identity != section.identity {
            return Err(row_weights_bound_to_snapshot!());
        }
        self.retarget(&section.values, &depends_on, ctx)
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
    /// The preview reports the refusal the apply raises when this handle
    /// cannot perform the transformation at all (a retarget on a study with no
    /// prepared score table), from the same check the apply runs
    /// ([`Self::transform_capability`]). Data-dependent checks — schema
    /// compatibility of refreshed data, retarget weights, dependencies and
    /// weighted overlap — can only run at apply and stay as obligations.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures while reading the contract.
    pub fn preview_transform(
        &self,
        intent: TransformIntent,
    ) -> Result<TransformationReport, CausalError> {
        let report = self.contract()?.preview_transform(intent);
        Ok(match self.transform_capability(intent) {
            Ok(()) => report,
            Err(refusal) => report.refused_on_handle(refusal.to_string()),
        })
    }

    /// Encode an executed result with a verified contract section.
    ///
    /// The result must have been executed by this handle, under its current
    /// program and on its current data snapshot; see
    /// [`Self::contract_for_result`].
    ///
    /// # Errors
    ///
    /// [`CausalError::Conflict`] for a result from another program or data
    /// snapshot, canonical-encoding failures, or mass totals that do not
    /// conserve.
    pub fn encode_contracted_result(
        &self,
        result: &StudyResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CausalError> {
        let (mut contract, payloads) = self.contract_and_payloads_for_result(result)?;
        // Bind the reuse layers before the claim: the claim id covers the seal,
        // and the seal covers every advertised identity.
        let reuse = self.bind_reuse(&mut contract, result)?;
        let (claim, body) = result.claim_with_body(&contract, ctx)?;
        let execution = execution_identity_from_context(ctx);
        let section =
            contract.section_from_payloads(&payloads, Some(&claim), Some(&execution), &reuse)?;
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

    /// Inspect ranking obligations without scoring.
    ///
    /// # Errors
    ///
    /// Unlicensed support, missing identification product, or a width ranking
    /// across incomparable estimands.
    pub fn preview_rank_designs<A, O>(
        &self,
        objective: &crate::design::DesignObjective,
        candidates: &[crate::design::CandidateDesign],
        eval: &crate::design::DesignEvaluationContext<'_, A, O>,
        decision_target: Option<antecedent_core::SemanticDigest>,
    ) -> Result<crate::design::DesignRankPreview, CausalError>
    where
        A: Clone,
        O: Clone,
    {
        crate::design::preview_design_rank(
            &self.contract()?,
            objective,
            candidates,
            eval,
            decision_target,
        )
    }

    /// Compose the existing design ranker onto this prepared handle.
    ///
    /// # Errors
    ///
    /// Unlicensed support, missing identification product, incomparable width
    /// targets, or ranker failure.
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
        self.rank_designs_bound(ranker, objective, candidates, eval, ctx, None)
    }

    /// Rank after binding an optional decision target.
    ///
    /// A width ranking across a different target is refused unless the caller
    /// uses a common decision utility instead of width.
    ///
    /// # Errors
    ///
    /// Unlicensed support, missing identification product, incomparable width
    /// targets, or ranker failure.
    pub fn rank_designs_bound<A, O>(
        &self,
        ranker: &crate::design::DesignRanker,
        objective: &crate::design::DesignObjective,
        candidates: &[crate::design::CandidateDesign],
        eval: &crate::design::DesignEvaluationContext<'_, A, O>,
        ctx: &ExecutionContext,
        decision_target: Option<antecedent_core::SemanticDigest>,
    ) -> Result<crate::design::DesignRanking, CausalError>
    where
        A: Clone,
        O: Clone,
    {
        crate::design::rank_designs_bound(
            &self.contract()?,
            ranker,
            objective,
            candidates,
            eval,
            ctx,
            decision_target,
        )
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
    /// [`CausalError::Conflict`] when the result carries no execution stamp or was
    /// executed under other identities; canonical-encoding failures, or mass totals
    /// that do not conserve.
    pub fn claim(
        &self,
        contract: &CausalContract,
        ctx: &ExecutionContext,
    ) -> Result<ClaimEnvelope, CausalError> {
        Ok(self.claim_with_body(contract, ctx)?.0)
    }

    /// Claim plus the exact result body its id digests.
    ///
    /// The claim id binds the contract seal (every identity, the four slots,
    /// graph class, structure source, identifier, estimator), every claim
    /// field (kind, value, domains, calibration, attested evidence,
    /// execution), and [`antecedent_io::result_digest`] of the body. Kind and
    /// domains come from the rules the independent consumer re-derives.
    fn claim_with_body(
        &self,
        contract: &CausalContract,
        ctx: &ExecutionContext,
    ) -> Result<(ClaimEnvelope, AnalysisResultWire), CausalError> {
        // A claim seals the result under the contract's identities, so the result must
        // prove which contract it ran under. An unstamped result (a plain `Study::run`, a
        // mixed-execution refutation) would be sealed under identities it never carried.
        let Some(executed) = &self.executed_contract else {
            return Err(CausalError::Conflict {
                what: "result",
                detail: "result carries no execution stamp; only a prepared handle's execution \
                         can be sealed into a claim",
            });
        };
        if executed.identities != contract.identities {
            return Err(CausalError::Conflict {
                what: "program",
                detail: "result was not executed under this contract",
            });
        }
        let body = body_for(&contract.body, self)?;
        let mut reasoning = result_reasoning(self, &contract.reasoning, &body)?;
        if let (SlotAvailability::Available(slot), Some(status)) =
            (&mut reasoning.identification, contract.body.identification_status)
        {
            // The body reports the program's identification certificate; the
            // slot states the same status the body carries.
            slot.status = status;
        }
        let reasoning_wire = reasoning_section(&reasoning);
        let kind_name =
            antecedent_io::claim_kind_name(&body, reasoning_wire.identification.value.as_ref());
        let kind = ClaimKind::from_name(kind_name).ok_or_else(|| CausalError::Compile {
            message: format!("unknown claim kind {kind_name}"),
        })?;
        let value = match kind {
            ClaimKind::Point => executed_scalar(self),
            _ => None,
        };
        let domains = antecedent_io::claim_domains(
            reasoning_wire.support.value.as_ref(),
            reasoning_wire.identification.value.as_ref(),
        );
        let domain = |name: &str| {
            DomainStatus::from_name(name).ok_or_else(|| CausalError::Compile {
                message: format!("unknown domain status {name}"),
            })
        };
        let execution =
            execution_digest(&execution_identity_from_context(ctx)).map_err(|err| io_err(&err))?;
        let calibration = antecedent_io::calibration::calibration_slots(
            &self.calibration_bases_with(contract, reasoning.identification.as_ref()),
        );
        let mut envelope = ClaimEnvelope::new(
            antecedent_core::SemanticDigest::from_bytes([0; 32]),
            contract.identities,
            kind,
            value,
            None,
            reasoning,
            ClaimDomains::new(
                domain(&domains.identification)?,
                domain(&domains.support)?,
                domain(&domains.evaluated)?,
            ),
            Some(execution),
            [
                Arc::from(format!("snapshot:{}", contract.identities.data_snapshot.to_hex())),
                Arc::from(format!(
                    "identification:{}",
                    contract.identities.identification.to_hex()
                )),
                Arc::from(format!(
                    "program:{}",
                    contract.identities.program.map_or_else(String::new, |p| p.to_hex())
                )),
            ],
        );
        envelope.calibration = calibration_view(&calibration);
        envelope.attested = attested_evidence(self);
        let seal = contract.seal(&reasoning_wire, Some(*execution.as_bytes()))?;
        let result = antecedent_io::result_digest(&body).map_err(|err| io_err(&err))?;
        envelope.claim_id =
            claim_digest(&ClaimIdentityWire::new(seal, &claim_section(&envelope), result))
                .map_err(|err| io_err(&err))?;
        Ok((envelope, body))
    }

    /// Retain the full execution payload alongside a canonical result contract.
    ///
    /// # Errors
    ///
    /// Invalid response, posterior, interval or structural weights.
    pub fn fill_analysis_result_payloads(
        &self,
        wire: &mut AnalysisResultWire,
        artifact_id: &str,
    ) -> Result<(), CausalError> {
        wire.response = self
            .response
            .as_ref()
            .map(antecedent_io::causal_response_to_wire)
            .transpose()
            .map_err(|err| io_err(&err))?;
        wire.posterior_artifact = self
            .posterior
            .as_ref()
            .map(|posterior| antecedent_io::encode_causal_posterior_bytes(posterior, artifact_id))
            .transpose()
            .map_err(|err| io_err(&err))?;
        wire.mediation_grid = self.mediation_grid.as_ref().map(mediation_grid_wire);
        wire.structural_response =
            self.structural_response.as_ref().map(structural_response_wire).transpose()?;
        wire.unit_effects =
            self.counterfactual.as_ref().map(|ite| antecedent_io::UnitEffectsWire {
                effects: ite.unit_effects.to_vec(),
                homogeneous: self.estimate.unit_effects_homogeneous,
                intervals: ite.unit_effect_intervals.as_ref().map(|intervals| {
                    antecedent_io::UnitEffectIntervalsWire {
                        lower: intervals.lower.to_vec(),
                        upper: intervals.upper.to_vec(),
                        level: intervals.level,
                        method: intervals.method.to_string(),
                    }
                }),
                extrapolative: ite.unit_extrapolative.as_ref().map(|flags| flags.to_vec()),
            });
        Ok(())
    }

    /// Full execution body for the composite `analysis_result` container.
    ///
    /// Response values, posterior draws, mediation grids and structural atoms
    /// accompany the query, identification certificate and scalar summary.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn analysis_result_wire(
        &self,
        query: &antecedent_core::CausalQuery,
    ) -> Result<AnalysisResultWire, CausalError> {
        self.analysis_result_wire_with_context(query, None, None)
    }

    /// [`Self::analysis_result_wire`], threading the originating study's population
    /// registry and cached temporal identification. Callers that hold both the
    /// prepared study and its executed result (the composite artifact path) use
    /// this instead of re-deriving the body from the result alone, so the scalar,
    /// identification and every other field come from the one place that builds
    /// the contracted artifact.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn analysis_result_wire_with_context(
        &self,
        query: &antecedent_core::CausalQuery,
        registry: Option<&antecedent_core::PopulationRegistry>,
        temporal: Option<&CachedTemporalIdentification>,
    ) -> Result<AnalysisResultWire, CausalError> {
        body_for(&body_frame(query, temporal, None, registry)?, self)
    }
}

/// Uncertainty component of a response interval by what its level means: a credible
/// interval is posterior parameter uncertainty, a confidence interval is sampling
/// uncertainty.
fn response_interval_component(
    interpretation: IntervalInterpretation,
    credible: &'static str,
    confidence: &'static str,
) -> (UncertaintySource, &'static str) {
    match interpretation {
        IntervalInterpretation::Credible => (UncertaintySource::Parameter, credible),
        IntervalInterpretation::Confidence => (UncertaintySource::Sampling, confidence),
    }
}

fn io_err(err: &antecedent_io::IoError) -> CausalError {
    CausalError::Compile { message: err.to_string() }
}

/// Data-independent layers of a contract: target through observation.
///
/// A prepared handle compiles these once per handle state; every estimate
/// click only adds its data snapshot.
#[derive(Clone, Debug)]
pub(crate) struct ProgramPayloads {
    target: TargetIdentityWire,
    identification: IdentificationIdentityWire,
    identification_product: Option<IdentificationProductWire>,
    program: ProgramIdentityWire,
    inference_binding: InferenceBindingWire,
    observation: ObservationIdentityWire,
    target_digest: antecedent_core::SemanticDigest,
    identification_digest: antecedent_core::SemanticDigest,
    identification_product_digest: Option<antecedent_core::SemanticDigest>,
    program_digest: antecedent_core::SemanticDigest,
    inference_binding_digest: antecedent_core::SemanticDigest,
    observation_digest: antecedent_core::SemanticDigest,
}

impl ProgramPayloads {
    fn identities(&self, data_snapshot: antecedent_core::SemanticDigest) -> ContractIdentities {
        ContractIdentities::new(
            self.target_digest,
            self.identification_digest,
            self.identification_product_digest,
            Some(self.program_digest),
            self.inference_binding_digest,
            self.observation_digest,
            data_snapshot,
        )
    }
}

struct ContractPayloads {
    program: Arc<ProgramPayloads>,
    data_snapshot: DataSnapshotIdentityWire,
    identities: ContractIdentities,
}

fn program_payloads_for(
    study: &Study,
    resolved_estimator: Option<&str>,
    checked_aipw: Option<&antecedent_estimate::CheckedAipwPreparation>,
    checked_frontdoor: Option<&antecedent_estimate::CheckedFrontDoorPreparation>,
    checked_iv: Option<&antecedent_estimate::CheckedIvPreparation>,
    checked_linear: Option<&super::prepared::CheckedLinearOperation>,
    checked_functional_effect: Option<&antecedent_expr::FunctionalProgram>,
) -> Result<ProgramPayloads, CausalError> {
    let cached = contract_identification(study);
    let cached = cached.as_deref();
    program_payloads(
        study,
        cached,
        cached.is_some_and(identification_search_capped),
        resolved_estimator,
        checked_aipw,
        checked_frontdoor,
        checked_iv,
        checked_linear,
        checked_functional_effect,
    )
}

/// Add `study`'s data snapshot to program payloads already compiled for it.
fn contract_payloads(
    program: Arc<ProgramPayloads>,
    study: &Study,
    prepared: Option<&PreparedStudy>,
) -> Result<ContractPayloads, CausalError> {
    let mut data_snapshot =
        data_snapshot_wire(&study.data, study.interference.as_ref(), &program.observation_digest)?;
    if let (Some(prepared), DataInput::Tabular(tabular)) = (prepared, &study.data) {
        let laws = prepared
            .distribution_factor_snapshot(tabular)?
            .or(prepared.functional_effect_factor_snapshot(tabular)?);
        if let Some(laws) = laws {
            data_snapshot.distribution_factor_laws = Some(
                antecedent_io::distribution_factor_laws_to_wire(&laws)
                    .map_err(|err| io_err(&err))?,
            );
        }
    }
    let snapshot_digest = data_snapshot_digest(&data_snapshot).map_err(|err| io_err(&err))?;
    Ok(ContractPayloads { identities: program.identities(snapshot_digest), program, data_snapshot })
}

fn program_payloads(
    study: &Study,
    cached: Option<&IdentificationResult>,
    search_capped: bool,
    resolved_estimator: Option<&str>,
    checked_aipw: Option<&antecedent_estimate::CheckedAipwPreparation>,
    checked_frontdoor: Option<&antecedent_estimate::CheckedFrontDoorPreparation>,
    checked_iv: Option<&antecedent_estimate::CheckedIvPreparation>,
    checked_linear: Option<&super::prepared::CheckedLinearOperation>,
    checked_functional_effect: Option<&antecedent_expr::FunctionalProgram>,
) -> Result<ProgramPayloads, CausalError> {
    let schema = data_schema(&study.data);
    let target = TargetIdentityWire {
        format: IDENTITY_FORMAT,
        schema: schema_to_wire(schema),
        query: causal_query_to_wire_with_registry(&study.query, study.population_registry.as_ref())
            .map_err(|err| io_err(&err))?,
    };
    let target_digest = digest_wire(IdentityDomain::Target, &target).map_err(|err| io_err(&err))?;
    let question_digest =
        digest_wire(IdentityDomain::Target, &target.question()).map_err(|err| io_err(&err))?;
    let observation = observation_identity_wire(
        schema,
        super::contract_identity::observation_identity_tags(
            &study.query,
            study.observation_delayed_entry,
        ),
    );
    let observation_digest =
        digest_wire(IdentityDomain::Observation, &observation).map_err(|err| io_err(&err))?;
    let graph = graph_identity(study)?;
    let identification = IdentificationIdentityWire {
        format: IDENTITY_FORMAT,
        target_question: *question_digest.as_bytes(),
        population_depends_on: population_depends_on(
            &study.query,
            study.population_registry.as_ref(),
        ),
        rd_config: study.rd.as_ref().map(|cfg| antecedent_io::RdConfigWire {
            running_variable: cfg.running_variable.raw(),
            cutoff_bits: cfg.cutoff.to_bits(),
            bandwidth_bits: cfg.bandwidth.to_bits(),
            se_kind: Some(cfg.se_kind.as_str().into()),
        }),
        class_prior: study.class_prior.as_ref().map(crate::ClassPrior::identity_wire),
        transport: transport_premises(study)?,
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
            let mut wire =
                identification_product_wire(result, search_capped).map_err(|err| io_err(&err))?;
            // Two envelopes over the same class with the same estimands are
            // still different products when they examined different
            // completions or carry different mass.
            wire.envelope = class_envelope_wire(study);
            Some(wire)
        }
        None => None,
    };
    let identification_product_digest = match &identification_product {
        Some(wire) => Some(identification_product_digest_wire(wire).map_err(|err| io_err(&err))?),
        None => None,
    };
    let mut commitments = inferential_commitments(study, resolved_estimator);
    let checked_aipw_lowering = checked_aipw.map(|checked| {
        let lowering = checked.lowering();
        commitments.se_kind = Some(lowering.se_kind.as_str().to_string());
        let se_lag = match lowering.se_kind {
            antecedent_estimate::AnalyticSeKind::NeweyWest { lag }
            | antecedent_estimate::AnalyticSeKind::PanelClusterHac { lag } => Some(lag as u64),
            _ => None,
        };
        antecedent_io::CheckedAipwLoweringWire {
            format: 1,
            functional: lowering.functional.raw(),
            treatment: lowering.treatment.raw(),
            outcome: lowering.outcome.raw(),
            adjustment: lowering.adjustment.iter().map(|id| id.raw()).collect(),
            population: "all_observed".into(),
            procedure: "cross_fitted_logistic_ols".into(),
            folds: lowering.folds as u64,
            se_kind: lowering.se_kind.as_str().into(),
            se_lag,
            bootstrap_replicates: lowering.bootstrap_replicates,
        }
    });
    let checked_frontdoor_lowering =
        checked_frontdoor
            .map(|checked| {
                let lowering = checked.lowering();
                let procedure = match lowering.procedure {
            antecedent_estimate::frontdoor::CheckedFrontDoorProcedure::LinearPathProduct => {
                "linear_path_product"
            }
            antecedent_estimate::frontdoor::CheckedFrontDoorProcedure::Functional => "functional",
        };
                let problem = checked.problem();
                Ok::<_, CausalError>(antecedent_io::CheckedFrontDoorLoweringWire {
                    format: 1,
                    functional: lowering.functional.raw(),
                    executable: lowering.executable.raw(),
                    treatment: lowering.treatment.raw(),
                    outcome: lowering.outcome.raw(),
                    mediators: lowering.mediators.iter().map(|id| id.raw()).collect(),
                    active_bits: lowering.active.to_bits(),
                    control_bits: lowering.control.to_bits(),
                    procedure: procedure.into(),
                    complete_case_rows: problem.nrows as u64,
                    overlap: overlap_policy_tag(problem.overlap),
                    uncertainty: format!(
                        "{}:{}",
                        commitments.interval_method,
                        commitments.se_kind.as_deref().unwrap_or("unspecified")
                    ),
                    arena: antecedent_io::expr_arena_to_wire(checked.program().arena())
                        .map_err(|error| io_err(&error))?,
                })
            })
            .transpose()?;
    let checked_iv_lowering = checked_iv.map(|checked| {
        let lowering = checked.lowering();
        commitments.se_kind = Some(lowering.se_kind.as_str().to_string());
        let procedure = match lowering.procedure {
            antecedent_estimate::CheckedIvProcedure::Wald => "wald",
            antecedent_estimate::CheckedIvProcedure::TwoStageLeastSquares => {
                "two_stage_least_squares"
            }
        };
        let weak_instrument_uncertainty =
            if lowering.se_kind == antecedent_estimate::AnalyticSeKind::Homoskedastic {
                "anderson_rubin_if_weak"
            } else {
                "withheld_non_homoskedastic"
            };
        antecedent_io::CheckedIvLoweringWire {
            format: 1,
            functional: lowering.functional.raw(),
            treatment: lowering.treatment.raw(),
            outcome: lowering.outcome.raw(),
            instrument: lowering.instruments.first().map_or(u32::MAX, |id| id.raw()),
            adjustment: lowering.adjustment.iter().map(|id| id.raw()).collect(),
            active_bits: lowering.active.to_bits(),
            control_bits: lowering.control.to_bits(),
            instrument_active_bits: lowering.instrument_active.to_bits(),
            instrument_control_bits: lowering.instrument_control.to_bits(),
            procedure: procedure.into(),
            se_kind: lowering.se_kind.as_str().into(),
            weak_instrument_uncertainty: weak_instrument_uncertainty.into(),
            complete_case_rows: u64::try_from(checked.problem().nrows).unwrap_or(u64::MAX),
        }
    });
    let checked_linear_adjustment_lowering = checked_linear
        .filter(|_| resolved_estimator == Some("linear.adjustment.ate"))
        .filter(|_| {
            study.graph.class() == GraphClass::Dag
                && matches!(
                    study.structure_source,
                    StructureSource::Explicit | StructureSource::Accepted
                )
                && matches!(study.inference, InferenceMode::Frequentist)
        })
        .map(|operation| {
            let checked = &operation.preparation;
            let lowering = checked.lowering();
            let problem = checked.problem();
            let fit_kind = match operation.fitter.fit_kind {
                antecedent_estimate::LinearFitKind::Ols => "ols".to_string(),
                antecedent_estimate::LinearFitKind::Ridge { lambda } => {
                    format!("ridge:{:016x}", lambda.to_bits())
                }
                antecedent_estimate::LinearFitKind::Lasso { lambda } => {
                    format!("lasso:{:016x}", lambda.to_bits())
                }
                antecedent_estimate::LinearFitKind::Huber { c } => {
                    format!("huber:{:016x}", c.to_bits())
                }
            };
            let design_columns = problem
                .design
                .columns
                .iter()
                .map(|column| match column.role {
                    antecedent_stats::DesignColumnRole::Intercept => "intercept".to_string(),
                    antecedent_stats::DesignColumnRole::Treatment => "treatment".to_string(),
                    antecedent_stats::DesignColumnRole::Covariate(id) => {
                        format!("covariate:{}", id.raw())
                    }
                })
                .collect();
            let se_kind = operation.fitter.se_kind;
            commitments.se_kind = Some(se_kind.as_str().to_string());
            Ok::<_, CausalError>(antecedent_io::CheckedLinearAdjustmentLoweringWire {
                format: 1,
                functional: lowering.source.raw(),
                executable: lowering.executable.raw(),
                treatment: lowering.treatment.raw(),
                outcome: lowering.outcome.raw(),
                adjustment: lowering.adjustment.iter().map(|id| id.raw()).collect(),
                active_bits: lowering.active.to_bits(),
                control_bits: lowering.control.to_bits(),
                population: antecedent_io::TargetPopulationWire::from_domain(&lowering.population)
                    .map_err(|error| io_err(&error))?,
                design_columns,
                fit_kind,
                backend: "faer".into(),
                se_kind: se_kind.as_str().into(),
                se_lag: match se_kind {
                    antecedent_estimate::AnalyticSeKind::NeweyWest { lag }
                    | antecedent_estimate::AnalyticSeKind::PanelClusterHac { lag } => {
                        Some(lag as u64)
                    }
                    _ => None,
                },
                bootstrap_replicates: operation.fitter.bootstrap_replicates,
                interval_method: commitments.interval_method.clone(),
                complete_case_rows: problem.design.nrows as u64,
                arena: antecedent_io::expr_arena_to_wire(checked.program().arena())
                    .map_err(|error| io_err(&error))?,
            })
        })
        .transpose()?;
    let functional_program = if resolved_estimator
        .and_then(|name| name.parse::<crate::EstimatorId>().ok())
        == Some(crate::EstimatorId::FunctionalEffect)
    {
        checked_functional_effect
            .map(antecedent_io::functional_program_to_wire)
            .transpose()
            .map_err(|err| io_err(&err))?
    } else if resolved_estimator.and_then(|name| name.parse::<crate::EstimatorId>().ok())
        == Some(crate::EstimatorId::FunctionalDistribution)
    {
        cached
            .map(|identification| {
                let estimand = crate::strategy_table::select_estimand(
                    identification,
                    crate::EstimatorId::FunctionalDistribution,
                )?;
                let schema = data_schema(&study.data);
                let program_schema = antecedent_expr::ProgramSchema::new(
                    schema.variables().iter().map(|variable| {
                        (
                            variable.id,
                            antecedent_expr::ProgramVariable { name: Arc::clone(&variable.name) },
                        )
                    }),
                );
                let root = estimand.functional;
                let program = antecedent_expr::FunctionalProgram::new(
                    identification.arena.clone(),
                    program_schema,
                    root,
                    root,
                    antecedent_expr::ProgramLimits::default(),
                )
                .map_err(|error| CausalError::Compile { message: error.to_string() })?;
                antecedent_io::functional_program_to_wire(&program).map_err(|error| io_err(&error))
            })
            .transpose()?
    } else {
        None
    };
    let program = ProgramIdentityWire {
        format: IDENTITY_FORMAT,
        target: *target_digest.as_bytes(),
        identification: *identification_digest.as_bytes(),
        identification_product: identification_product_digest.map(|digest| *digest.as_bytes()),
        completion_budget: study.max_completions.map(|cap| cap as u64),
        commitments: commitments.clone(),
        functional_program,
        checked_aipw_lowering,
        checked_frontdoor_lowering,
        checked_iv_lowering,
        checked_linear_adjustment_lowering,
    };
    let program_digest = program_digest(&program).map_err(|err| io_err(&err))?;
    let inference_binding = InferenceBindingWire {
        format: IDENTITY_FORMAT,
        inference: commitments.inference.clone(),
        bootstrap_replicates: study.bootstrap_replicates,
        bayesian: match &study.inference {
            InferenceMode::Bayesian(cfg) => Some(super::contract_identity::bayesian_binding(cfg)),
            InferenceMode::Frequentist => None,
        },
        validation_suite: study.refute.validation_suite_id().map(str::to_string),
        overlap_policy: study.overlap_policy.map(overlap_policy_tag),
        estimator_spec: study.estimator_spec_identity.clone(),
        response_options: study
            .response_options
            .as_ref()
            .map(super::contract_identity::response_options),
        observation_options: super::contract_identity::observation_options(
            &study.observation_options,
        ),
        split: study.split.as_ref().map(super::contract_identity::split),
    };
    let inference_binding_digest =
        inference_binding_digest(&inference_binding).map_err(|err| io_err(&err))?;
    Ok(ProgramPayloads {
        target,
        identification,
        identification_product,
        program,
        inference_binding,
        observation,
        target_digest,
        identification_digest,
        identification_product_digest,
        program_digest,
        inference_binding_digest,
        observation_digest,
    })
}

fn transport_premises(
    study: &Study,
) -> Result<Option<antecedent_io::TransportIdentityWire>, CausalError> {
    if study.selection_diagram.is_none() && study.transport_trial.is_none() {
        return Ok(None);
    }
    antecedent_io::transport_identity(
        study.selection_diagram.as_ref(),
        study
            .transport_trial
            .as_ref()
            .map(|trial| (trial.trial, trial.selection_probability, trial.treatment_probability)),
    )
    .map(Some)
    .map_err(|err| io_err(&err))
}

fn compile_contract(
    study: &Study,
    prepared: Option<&PreparedStudy>,
) -> Result<CausalContract, CausalError> {
    Ok(compile_with_payloads(study, prepared, None)?.0)
}

fn compile_with_payloads(
    study: &Study,
    prepared: Option<&PreparedStudy>,
    program: Option<Arc<ProgramPayloads>>,
) -> Result<(CausalContract, ContractPayloads), CausalError> {
    // Cheap inspection is the pre-prepare record: it reads no cache, so a
    // prepared handle's `inspect` and `contract` differ by the products, not
    // by which fields happen to be populated.
    let cached = prepared.and_then(|_| contract_identification(study));
    let cached = cached.as_deref();
    let search_capped = cached.is_some_and(identification_search_capped);
    let resolved_estimator =
        prepared.and_then(|prepared| prepared.plan().logical.record.estimator.clone());
    let program = match program {
        Some(program) => program,
        None => Arc::new(program_payloads(
            study,
            cached,
            search_capped,
            resolved_estimator.as_deref(),
            prepared
                .filter(|prepared| study.query == *prepared.query())
                .and_then(PreparedStudy::checked_aipw_ate),
            prepared
                .filter(|prepared| study.query == *prepared.query())
                .and_then(PreparedStudy::checked_frontdoor_linear),
            prepared
                .filter(|prepared| study.query == *prepared.query())
                .and_then(PreparedStudy::checked_iv),
            prepared
                .filter(|prepared| study.query == *prepared.query())
                .and_then(PreparedStudy::checked_linear_operation),
            prepared
                .filter(|prepared| study.query == *prepared.query())
                .and_then(PreparedStudy::checked_functional_effect_program),
        )?),
    };
    let mut payloads = contract_payloads(program, study, prepared)?;
    if prepared.is_none() {
        // Cheap inspection runs no identification: the program the prepared
        // handle compiles covers products that do not exist yet.
        payloads.identities.program = None;
    }
    let reasoning = reasoning_view(study, prepared, cached, search_capped);
    let lagged = full_temporal_identification(study);
    let body =
        body_frame(&study.query, lagged.as_deref(), cached, study.population_registry.as_ref())?;
    let functional = match overlap_label(study, resolved_estimator.as_deref()) {
        Some(overlap) => format!("{}+{overlap}", functional_label(&study.query)),
        None => functional_label(&study.query),
    };
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
            resolved_estimator,
            row_count: data_row_count(&study.data),
            modality: Arc::from(snapshot_modality(&study.data)),
            inference: Arc::from(match study.inference {
                InferenceMode::Frequentist => "frequentist",
                InferenceMode::Bayesian(_) => "bayesian",
            }),
            query_kind: Arc::from(
                crate::support::query_axis_name(&study.query, study.graph.class())
                    .unwrap_or("Unknown"),
            ),
            score_reuse: None,
            target_weights: None,
            checked_aipw_rows: None,
            posterior: Arc::from(posterior_label(&study.inference)),
            functional: Arc::from(functional),
            posterior_draws: match &study.inference {
                InferenceMode::Bayesian(cfg) => u32::try_from(cfg.n_draws).ok(),
                InferenceMode::Frequentist => None,
            },
            body: Arc::new(body),
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

/// Data modality of an execution, in the `DataSnapshotIdentityWire::modality`
/// vocabulary.
///
/// The single owner of that vocabulary. The data snapshot and the calibration
/// match key must name the same modality for the same execution, or the claim
/// cites coverage measured on data of another shape — so both read this, from
/// the `DataInput` the study actually holds, rather than one of them
/// re-deriving it from a plan record that may have been compiled under a
/// coarser classification.
const fn snapshot_modality(data_input: &DataInput) -> &'static str {
    match data_input {
        DataInput::Tabular(_) => "tabular",
        DataInput::Temporal(_) => "series",
        DataInput::Event(_) => "event",
        DataInput::MultiEnv(_) => "multi_env",
        DataInput::Panel(_) => "panel",
    }
}

fn data_snapshot_wire(
    data_input: &DataInput,
    interference: Option<&super::builder::InterferenceSpec>,
    observation: &antecedent_core::SemanticDigest,
) -> Result<DataSnapshotIdentityWire, CausalError> {
    let modality = snapshot_modality(data_input);
    let (regularity, row_count, unit_count) = match data_input {
        DataInput::Tabular(data) => (None, u64_count(data.row_count())?, None),
        DataInput::Temporal(data) | DataInput::Event(data) => (
            Some(regularity_tag(&data.time_index().regularity)),
            u64_count(data.row_count())?,
            None,
        ),
        DataInput::MultiEnv(data) => (
            None,
            u64_count(data.environments().iter().map(TableView::row_count).sum())?,
            Some(u64_count(data.env_count())?),
        ),
        DataInput::Panel(data) => {
            (None, u64_count(data.total_rows())?, Some(u64_count(data.unit_count())?))
        }
    };
    let partitions = match data_input {
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
        interference: interference.map(super::contract_identity::interference_snapshot),
        distribution_factor_laws: None,
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
        return Ok(GraphIdentityWire::GraphPosterior {
            graph_class: study.graph.class().as_str().into(),
            n_atoms: u64::try_from(posterior.n_graphs).unwrap_or(u64::MAX),
            atoms: graph_posterior_atom_identities(posterior).map_err(|err| io_err(&err))?,
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

// A first identified posterior atom is not the identification status of the
// entire mixture. Preserve unidentified mass and disagreeing estimands at
// compilation, as the executor does, so the program and execution agree.
fn contract_identification(study: &Study) -> Option<Cow<'_, IdentificationResult>> {
    let Some(cached) = cached_identification(study) else {
        // TemporalCpdag / TemporalPag prepare caches a class envelope rather
        // than a per-graph result. Aggregate it the way the executor does, so
        // the prepared program binds a product instead of none at all.
        let cache = study.temporal_class_identification_cache.as_ref()?;
        return Some(Cow::Owned(super::execute::envelope_to_identification_result_for(
            &cache.envelope.envelope,
            study.query.clone(),
        )));
    };
    let (unidentified, contributing) =
        if let Some(cache) = study.graph_posterior_identification_cache.as_ref() {
            (
                cache.graphs.unidentified_mass(),
                cache.atoms.iter().map(|atom| &atom.estimand).collect::<Vec<_>>(),
            )
        } else if let Some(cache) = study.dbn_posterior_identification_cache.as_ref() {
            (
                cache.graphs.unidentified_mass(),
                cache.atoms.iter().map(|atom| &atom.estimand).collect::<Vec<_>>(),
            )
        } else {
            return Some(Cow::Borrowed(cached));
        };
    let status =
        super::execute::graph_posterior_mixture_status(unidentified, &contributing, cached.status);
    if status == cached.status {
        Some(Cow::Borrowed(cached))
    } else {
        let mut aggregate = cached.clone();
        aggregate.status = status;
        Some(Cow::Owned(aggregate))
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
    if let Some(cache) = study.graph_posterior_identification_cache.as_ref() {
        return cache.atoms.first().map(|atom| &atom.identification);
    }
    None
}

/// The study's temporal identification product and its exact unfolded variable namespace,
/// however it was produced.
///
/// A `TemporalDag` prepare caches this directly. A DBN posterior or a TemporalCpdag/Pag
/// envelope instead cache a per-atom or per-completion result with its own unfold indexer;
/// [`dbn_projected_temporal_identification`] and [`class_projected_temporal_identification`]
/// project those onto the same [`CachedTemporalIdentification`] shape so every consumer —
/// the compiled contract and an exported `analysis_result` artifact alike — validates and
/// reports the same namespace for the same study.
pub(crate) fn full_temporal_identification(
    study: &Study,
) -> Option<Cow<'_, CachedTemporalIdentification>> {
    if let Some(cache) = study.temporal_identification_cache.as_deref() {
        return Some(Cow::Borrowed(cache));
    }
    dbn_projected_temporal_identification(study)
        .or_else(|| class_projected_temporal_identification(study))
        .map(Cow::Owned)
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
    if let Some(cache) = study.temporal_class_identification_cache.as_ref() {
        let (horizon, envelope) = cache.by_horizon.first().map_or_else(
            || (query_horizon_steps(&study.query), &cache.envelope),
            |(horizon, envelope)| (*horizon, envelope),
        );
        if let Some(projected) = project_class_envelope(horizon, envelope) {
            return Some(projected);
        }
    }
    let cache = study.temporal_class_posterior_identification_cache.as_ref()?;
    let atom = cache.class_atoms.first()?;
    project_class_envelope(query_horizon_steps(&study.query), &atom.envelope)
}

fn project_class_envelope(
    horizon: u32,
    envelope: &antecedent_identify::TemporalClassEnvelope,
) -> Option<CachedTemporalIdentification> {
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

/// Shape of the enumerated class a prepared product covers, when there is one.
fn class_envelope_wire(study: &Study) -> Option<IdentificationEnvelopeWire> {
    if let Some(cache) = study.temporal_class_identification_cache.as_ref() {
        return Some(envelope_shape(&cache.envelope.envelope));
    }
    if let Some(cache) = study.cpdag_identification_cache.as_ref() {
        return Some(envelope_shape(&cache.envelope));
    }
    if let Some(cache) = study.pag_identification_cache.as_ref() {
        return Some(envelope_shape(&cache.envelope));
    }
    None
}

fn envelope_shape<G>(envelope: &IdentificationEnvelope<G>) -> IdentificationEnvelopeWire {
    IdentificationEnvelopeWire {
        cases: envelope.cases.len() as u64,
        identified_weight_bits: envelope.identified_weight.0.to_bits(),
        unidentified_weight_bits: envelope.unidentified_weight.0.to_bits(),
        truncated_completions: envelope.truncated_completions as u64,
    }
}

fn identification_search_capped(result: &IdentificationResult) -> bool {
    result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code.as_ref() == CAPPED_COMPLETION_DIAGNOSTIC_CODE)
}

/// Inferential commitments of the compiled program. `resolved_estimator` is
/// the prepared plan's `logical_plan.estimator`; cheap inspection compiles no
/// plan and leaves it `None`.
fn inferential_commitments(
    study: &Study,
    resolved_estimator: Option<&str>,
) -> InferentialCommitmentsWire {
    let (mut interval_method, mut se_kind) = compiled_interval(study);
    if let Some(antecedent_io::EstimatorSpecWire::FrontDoorTwoStage(config)) =
        study.estimator_spec_identity.as_ref()
    {
        se_kind = config.se_kind.clone();
        if config.bootstrap_replicates > 0 {
            interval_method = IntervalMethod::BootstrapSe;
        }
    }
    InferentialCommitmentsWire {
        format: IDENTITY_FORMAT,
        estimator: study.estimator.map(|id| id.as_str().to_string()),
        resolved_estimator: resolved_estimator.map(str::to_string),
        identifier: study.identifier.map(|id| id.as_str().to_string()),
        inference: match study.inference {
            InferenceMode::Frequentist => "frequentist".into(),
            InferenceMode::Bayesian(_) => "bayesian".into(),
        },
        validation_suite: study.refute.validation_suite_id().map(str::to_string),
        interval_method: interval_method.as_str().into(),
        se_kind,
        prior_required: matches!(study.inference, InferenceMode::Bayesian(_)),
    }
}

fn compiled_interval(study: &Study) -> (IntervalMethod, Option<String>) {
    match &study.inference {
        InferenceMode::Bayesian(_) => (IntervalMethod::PosteriorQuantile, None),
        InferenceMode::Frequentist if study.bootstrap_replicates > 0 => {
            (IntervalMethod::BootstrapSe, None)
        }
        InferenceMode::Frequentist => (
            IntervalMethod::AnalyticSe,
            study.rd.as_ref().map(|cfg| cfg.se_kind.as_str().to_string()),
        ),
    }
}

fn population_depends_on(
    query: &CausalQuery,
    registry: Option<&antecedent_core::PopulationRegistry>,
) -> Vec<u32> {
    match query.target_population() {
        Some(TargetPopulation::CustomDistribution(id)) => registry
            .and_then(|reg| reg.distribution_dependencies(*id))
            .unwrap_or(&[])
            .iter()
            .map(|id| id.raw())
            .collect(),
        // Row weights carry their declared parents on the target population
        // itself; every other population declares no covariate dependence.
        _ => Vec::new(),
    }
}

fn analysis_row_count(result: &StudyResult, contract: &CausalContract) -> u64 {
    result
        .estimate
        .n_obs
        .or_else(|| result.estimate.influence.as_ref().map(|rows| rows.len() as u64))
        .or_else(|| result.estimate.score_table.as_ref().map(|table| table.n_rows as u64))
        .or_else(|| result.estimate.block_resampling.map(|block| block.rows as u64))
        .unwrap_or(contract.row_count)
}

fn data_row_count(data: &DataInput) -> u64 {
    match data {
        DataInput::Tabular(data) => data.row_count() as u64,
        DataInput::Temporal(data) | DataInput::Event(data) => data.row_count() as u64,
        DataInput::Panel(data) => data.total_rows() as u64,
        DataInput::MultiEnv(data) => {
            data.environments().iter().map(TableView::row_count).sum::<usize>() as u64
        }
    }
}

impl StudyResult {
    /// Calibration match bases of every interval this execution reported
    /// ([`StudyResult::reported_interval_bindings`]), primary first: the keys
    /// [`StudyResult::claim`] matches against the coverage records, and the
    /// keys a coverage test records.
    ///
    /// # Errors
    ///
    /// Structural mass totals that do not conserve.
    pub fn calibration_bases(
        &self,
        contract: &CausalContract,
    ) -> Result<Vec<antecedent_io::calibration::CalibrationBasisWire>, CausalError> {
        let identification = identification_slot_from_result(self)?;
        Ok(self.calibration_bases_with(contract, Some(&identification)))
    }

    fn calibration_bases_with(
        &self,
        contract: &CausalContract,
        identification: Option<&IdentificationSlot>,
    ) -> Vec<antecedent_io::calibration::CalibrationBasisWire> {
        use antecedent_io::calibration::{
            CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire,
        };
        let coordinate = contract
            .reasoning
            .support
            .as_ref()
            .and_then(|slot| slot.matrix_coordinate.as_deref())
            .and_then(crate::support::support_cell_from_coordinate);
        let (query, graph_class, structure, inference) = match coordinate {
            Some(cell) => (
                cell.query.to_string(),
                cell.graph_class.to_string(),
                if cell.structure == "graph_posterior" { "graph_posterior" } else { "fixed" },
                cell.inference.to_string(),
            ),
            None => (
                contract.query_kind.to_string(),
                contract.graph_class.as_str().to_string(),
                if contract.structure_source == StructureSource::GraphPosterior {
                    "graph_posterior"
                } else {
                    "fixed"
                },
                if contract.inference.eq_ignore_ascii_case("bayesian") {
                    "Bayesian".to_string()
                } else {
                    "Frequentist".to_string()
                },
            ),
        };
        // The snapshot's modality, not one re-derived from the compiled plan's
        // data classification: a temporal plan is compiled as `Temporal` for
        // event data too, so re-deriving stamped `series` on an event
        // execution whose snapshot said `event`, and the claim then failed to
        // verify against its own body.
        let modality = contract.modality.as_ref();
        let (label, unidentified_mass) = identification_label(identification);
        let label = antecedent_io::calibration::identification_key(
            label,
            self.identification.derivation.steps.iter().map(|step| step.rule.as_ref()),
        );
        let bayesian = contract.inference.eq_ignore_ascii_case("bayesian");
        self.reported_interval_bindings(bayesian)
            .into_iter()
            .map(|binding| CalibrationBasisWire {
                key: CalibrationKeyWire {
                    query: query.clone(),
                    graph_class: graph_class.clone(),
                    structure: structure.to_string(),
                    modality: modality.to_string(),
                    inference: inference.clone(),
                    estimator: self.logical_plan.estimator.as_deref().unwrap_or("").to_string(),
                    interval_method: binding.method.as_str().to_string(),
                    se_kind: binding.se_kind.map_or("", |kind| kind.as_str()).to_string(),
                    dependence: binding.dependence.to_string(),
                    posterior: contract.posterior.to_string(),
                    functional: contract.functional.to_string(),
                    level: if binding.level.is_finite() { binding.level } else { 0.0 },
                    identification: label.to_string(),
                },
                scope: CalibrationScopeWire {
                    row_count: analysis_row_count(self, contract),
                    replicates_ok: binding.replicates_ok,
                    posterior_draws: binding.posterior_draws.or_else(|| {
                        (binding.method == IntervalMethod::PosteriorQuantile)
                            .then_some(contract.posterior_draws)
                            .flatten()
                    }),
                    unidentified_mass,
                },
            })
            .collect()
    }
}

/// `point` when every structural mass is identified under an identified
/// status; `partial` otherwise. Returns the non-identified mass beside it.
fn identification_label(slot: Option<&IdentificationSlot>) -> (&'static str, f64) {
    let Some(slot) = slot else {
        return ("partial", 1.0);
    };
    let unidentified = slot.unidentified_mass + slot.unevaluable_mass + slot.incomplete_search_mass;
    let identified_status = matches!(
        slot.status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
    );
    if identified_status && unidentified <= 1e-12 {
        ("point", unidentified)
    } else {
        ("partial", unidentified)
    }
}

/// Functional label of `query` for the calibration key: what the support
/// axis query name leaves open and a coverage measurement depends on.
fn functional_label(query: &CausalQuery) -> String {
    fn population(population: &TargetPopulation) -> &'static str {
        match population {
            TargetPopulation::AllObserved => "all_observed",
            TargetPopulation::Treated => "treated",
            TargetPopulation::Untreated => "untreated",
            TargetPopulation::Environment(_) => "environment",
            TargetPopulation::Predicate(_) => "predicate",
            TargetPopulation::CustomDistribution(_) => "custom_distribution",
            TargetPopulation::RowWeights { .. } => "row_weights",
            TargetPopulation::LocalAtCutoff { .. } => "local_at_cutoff",
            _ => "other_population",
        }
    }
    fn outcome(functional: &antecedent_core::OutcomeFunctional) -> String {
        match functional {
            antecedent_core::OutcomeFunctional::Mean => "mean".into(),
            antecedent_core::OutcomeFunctional::Exceedance(_) => "exceedance".into(),
            antecedent_core::OutcomeFunctional::ExceedanceGrid(_) => "exceedance_grid".into(),
            antecedent_core::OutcomeFunctional::Quantile(tau) => {
                format!("quantile:{}", tau.to_f64())
            }
            _ => "other_functional".into(),
        }
    }
    fn policy(policy: &antecedent_core::TemporalPolicy) -> String {
        match policy {
            antecedent_core::TemporalPolicy::Pulse { .. } => "pulse".into(),
            antecedent_core::TemporalPolicy::Sustained { from, until } => {
                format!("sustained:{}", i64::from(*until) - i64::from(*from) + 1)
            }
            antecedent_core::TemporalPolicy::Dynamic { active_at, .. } => {
                format!("dynamic:{}", active_at.len())
            }
            _ => "other_policy".into(),
        }
    }
    match query {
        CausalQuery::AverageEffect(q) => {
            format!("{}.{}", population(&q.target_population), outcome(&q.outcome_functional))
        }
        CausalQuery::ConditionalEffect(q) => format!(
            "{}.{}",
            population(&q.inner.target_population),
            outcome(&q.inner.outcome_functional)
        ),
        CausalQuery::Mediation(q) => {
            let contrast = match q.contrast {
                antecedent_core::MediationContrast::Total => "total",
                antecedent_core::MediationContrast::Direct => "direct",
                antecedent_core::MediationContrast::Mediated => "mediated",
                antecedent_core::MediationContrast::NaturalDirect => "natural_direct",
                antecedent_core::MediationContrast::NaturalIndirect => "natural_indirect",
            };
            format!("{contrast}.{}", population(&q.target_population))
        }
        CausalQuery::NestedCounterfactual(_) => "natural_direct_shared_exogenous".into(),
        CausalQuery::TemporalEffect(q) => format!(
            "{}.h{}.{}",
            policy(&q.policy),
            q.horizon_steps,
            population(&q.target_population)
        ),
        CausalQuery::Response(q) => {
            let observation = match q.observation {
                antecedent_core::ObservationSpec::Complete => "complete",
                _ => "observation_adjusted",
            };
            let temporal = q.temporal.as_ref().map_or_else(String::new, |spec| {
                let horizons: Vec<String> = spec.horizons.iter().map(u32::to_string).collect();
                format!(".{}.h{}", policy(&spec.policy), horizons.join(","))
            });
            format!(
                "{}.{}.{observation}{temporal}",
                population(&q.target_population),
                outcome(&q.outcome_functional)
            )
        }
        CausalQuery::Distribution(q) => population(&q.target_population).into(),
        CausalQuery::PathSpecific(q) => population(&q.target_population).into(),
        _ => String::new(),
    }
}

/// Calibration label of a non-default propensity overlap policy.
///
/// A propensity-score estimator (weighting, matching, stratification,
/// distance matching, AIPW) clips propensities into `[0.01, 0.99]` and trims
/// nothing by default, and every coverage record for those estimators was
/// measured there. Any other clip or trim is a different construction (a trim
/// also changes the population the interval covers), so it is appended to the
/// functional label and binds only to records measured under it. `None` at the
/// default policy and for estimators that fit no propensity model.
fn overlap_label(study: &Study, resolved_estimator: Option<&str>) -> Option<String> {
    use antecedent_estimate::OverlapPolicy;
    let configured = study
        .estimator_spec
        .as_ref()
        .and_then(crate::estimator_spec::EstimatorSpec::propensity_overlap);
    let policy = if let Some(policy) = configured {
        policy
    } else {
        let estimator = study.estimator.map(|id| id.as_str()).or(resolved_estimator)?;
        if !matches!(
            estimator,
            "propensity.weighting"
                | "propensity.matching"
                | "propensity.stratification"
                | "distance.matching"
                | "aipw"
        ) {
            return None;
        }
        study.overlap_policy?
    };
    if policy == antecedent_estimate::default_propensity_overlap() {
        return None;
    }
    Some(match policy {
        OverlapPolicy::ExplicitOverride => "overlap=explicit_override".into(),
        OverlapPolicy::RequireDiagnostics { clip, trim } => {
            let bound = |value: Option<f64>| value.map_or_else(|| "none".into(), |v| v.to_string());
            format!("overlap=clip:{},trim:{}", bound(clip), bound(trim))
        }
    })
}

/// Posterior construction label: backend, likelihood and prior.
fn posterior_label(inference: &InferenceMode) -> String {
    let InferenceMode::Bayesian(cfg) = inference else {
        return String::new();
    };
    let backend = match cfg.backend {
        antecedent_estimate::BayesianBackendKind::ConjugateGaussian => "conjugate_gaussian",
        antecedent_estimate::BayesianBackendKind::Laplace => "laplace",
        antecedent_estimate::BayesianBackendKind::Hmc => "hmc",
    };
    let likelihood = match cfg.likelihood {
        antecedent_prob::BayesLikelihood::GaussianIdentity => "gaussian_identity",
        antecedent_prob::BayesLikelihood::BernoulliLogit => "bernoulli_logit",
        antecedent_prob::BayesLikelihood::BernoulliProbit => "bernoulli_probit",
        antecedent_prob::BayesLikelihood::PoissonLog => "poisson_log",
    };
    let prior =
        if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
            "prior=supplied".to_string()
        } else {
            format!("prior_scale={}", cfg.prior_scale)
        };
    format!("{backend}.{likelihood}.{prior}")
}

fn attested_evidence(result: &StudyResult) -> Arc<[AttestedEvidence]> {
    let names: std::collections::HashSet<&str> =
        result.custom_validator_names.iter().map(Arc::as_ref).collect();
    result
        .refutations
        .iter()
        .filter(|report| names.contains(report.refuter.as_ref()))
        .map(|report| AttestedEvidence {
            name: Arc::clone(&report.refuter),
            kind: Arc::from("custom_validator"),
            passed: report.passed,
            refuted_ate: Some(report.refuted_ate),
            comparison: Some(report.comparison),
            informative: report.informative,
            failure_condition: report.failure_condition.clone(),
            reverifiable: false,
            config_digest: None,
            payload_digest: None,
        })
        .collect()
}

/// Project a computed slot onto the claim envelope's view.
fn calibration_view(slot: &CalibrationSlotWire) -> CalibrationView {
    CalibrationView {
        status: Arc::from(slot.status.as_str()),
        record_id: slot.record_id.as_deref().map(Arc::from),
        reason: slot.reason.as_deref().map(Arc::from),
        scope_n: slot.scope_n,
        scope_dependence: slot.scope_dependence.as_deref().map(Arc::from),
        calibration_sha: slot.calibration_sha.as_deref().map(Arc::from),
        scope_n_max: slot.scope_n_max,
        nominal: slot.nominal,
        observed: slot.observed,
        basis: slot.basis.as_ref().map(|basis| {
            let key = &basis.key;
            antecedent_core::CalibrationBasis::new(
                [
                    Arc::from(key.query.as_str()),
                    Arc::from(key.graph_class.as_str()),
                    Arc::from(key.structure.as_str()),
                    Arc::from(key.modality.as_str()),
                    Arc::from(key.inference.as_str()),
                    Arc::from(key.estimator.as_str()),
                    Arc::from(key.interval_method.as_str()),
                    Arc::from(key.se_kind.as_str()),
                    Arc::from(key.dependence.as_str()),
                    Arc::from(key.posterior.as_str()),
                    Arc::from(key.functional.as_str()),
                ],
                key.level,
                Arc::from(key.identification.as_str()),
                basis.scope.row_count,
                basis.scope.replicates_ok,
                basis.scope.posterior_draws,
                basis.scope.unidentified_mass,
            )
        }),
        secondary: slot.secondary.iter().map(calibration_view).collect(),
    }
}

/// The inverse of [`calibration_view`]: the claim section stores the slot the
/// claim computed, basis included, so the consumer re-derives it.
fn calibration_wire(view: &CalibrationView) -> CalibrationSlotWire {
    use antecedent_io::calibration::{
        CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire,
    };
    CalibrationSlotWire {
        status: view.status.to_string(),
        record_id: view.record_id.as_ref().map(ToString::to_string),
        reason: view.reason.as_ref().map(ToString::to_string),
        scope_n: view.scope_n,
        scope_dependence: view.scope_dependence.as_ref().map(ToString::to_string),
        calibration_sha: view.calibration_sha.as_ref().map(ToString::to_string),
        scope_n_max: view.scope_n_max,
        nominal: view.nominal,
        observed: view.observed,
        basis: view.basis.as_ref().map(|basis| CalibrationBasisWire {
            key: CalibrationKeyWire {
                query: basis.query.to_string(),
                graph_class: basis.graph_class.to_string(),
                structure: basis.structure.to_string(),
                modality: basis.modality.to_string(),
                inference: basis.inference.to_string(),
                estimator: basis.estimator.to_string(),
                interval_method: basis.interval_method.to_string(),
                se_kind: basis.se_kind.to_string(),
                dependence: basis.dependence.to_string(),
                posterior: basis.posterior.to_string(),
                functional: basis.functional.to_string(),
                level: basis.level,
                identification: basis.identification.to_string(),
            },
            scope: CalibrationScopeWire {
                row_count: basis.row_count,
                replicates_ok: basis.replicates_ok,
                posterior_draws: basis.posterior_draws,
                unidentified_mass: basis.unidentified_mass,
            },
        }),
        secondary: view.secondary.iter().map(calibration_wire).collect(),
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
    if prepared.is_some() && cached.is_none() && identifies_per_execution(study) {
        // Sharp RD identifies from the running-variable design on every
        // estimate click (ADR 0020 / 0022), so a prepared program has no
        // identification product to report; each executed claim carries it.
        let mut view = ReasoningView::structural(support, assumptions);
        view.identification = SlotAvailability::unavailable(IDENTIFIED_PER_EXECUTION);
        return view;
    }
    if prepared.is_none() || cached.is_none() {
        return ReasoningView::structural(support, assumptions);
    }
    let result = cached.expect("checked");
    let slot = identification_slot_from_prepared(study, result, search_capped);
    ReasoningView::new(
        SlotAvailability::Available(slot),
        SlotAvailability::Available(support),
        SlotAvailability::unavailable("execution_specific"),
        SlotAvailability::Available(assumptions),
    )
}

/// Identification-slot reason on a prepared program that re-identifies on
/// every execution instead of caching an identification product.
pub const IDENTIFIED_PER_EXECUTION: &str = "identified_per_execution";

/// Whether this study is the identify-per-click exception (sharp RD).
fn identifies_per_execution(study: &Study) -> bool {
    study.estimator == Some(crate::EstimatorId::RdSharp)
        || study.identifier == Some(crate::IdentifierId::RdSharp)
}

fn matrix_coordinate(study: &Study) -> Option<String> {
    let cell = crate::support::support_cell_named(
        &study.query,
        if study.graph_posterior.is_some() {
            study.graph.class().as_str()
        } else {
            crate::support::matrix_graph_class(&study.graph, &study.query, study.tiered.as_ref())
        },
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
            OperationKind::Retarget => crate::error::RETARGET_REQUIRES_LICENSE,
            OperationKind::Export => crate::error::EXPORT_REQUIRES_LICENSE,
            _ => crate::error::OPERATION_REQUIRES_LICENSE,
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
    body: &AnalysisResultWire,
) -> Result<ReasoningView, CausalError> {
    let identification = identification_slot_from_result(result)?;
    let mut components = Vec::new();
    let published = crate::PublishedScalarUncertainty::select(&result.estimate);
    match published.method {
        antecedent_core::IntervalMethod::AnalyticSe => {
            components.push(UncertaintyComponent::new(
                UncertaintySource::Sampling,
                "analytic_se",
                false,
            ));
        }
        antecedent_core::IntervalMethod::BootstrapSe => {
            components.push(UncertaintyComponent::new(
                UncertaintySource::Sampling,
                "bootstrap_se",
                false,
            ));
        }
        antecedent_core::IntervalMethod::AndersonRubin => {
            components.push(UncertaintyComponent::new(
                UncertaintySource::Sampling,
                "anderson_rubin",
                false,
            ));
        }
        _ => {}
    }
    if result.posterior.is_some() {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Parameter,
            "posterior",
            false,
        ));
    }
    // A function-valued posterior can carry its band directly on the response,
    // without a scalar posterior or SE on StudyResult. Its portable reasoning
    // must not call that published parameter uncertainty "omitted". This runs for
    // every inference kind: the interval's own tag decides the uncertainty source
    // below, so a Frequentist response confidence band is exactly as available
    // here as a Bayesian credible band — there is nothing left for an
    // `inference == "bayesian"` gate to guard.
    {
        if let Some(response) = &result.response {
            // The interval's own tag decides the source: a credible interval is posterior
            // parameter uncertainty; a confidence interval (a Wald interval on influence
            // scores, say) is sampling uncertainty even when the fit was Bayesian.
            let target = match &response.uncertainty {
                ResponseUncertainty::None => None,
                ResponseUncertainty::Scalar { interpretation, .. } => {
                    Some(response_interval_component(
                        *interpretation,
                        "posterior_interval",
                        "response_confidence_interval",
                    ))
                }
                ResponseUncertainty::PointwiseBand { interpretation, .. } => {
                    Some(response_interval_component(
                        *interpretation,
                        "posterior_pointwise_band",
                        "response_pointwise_confidence_band",
                    ))
                }
                ResponseUncertainty::SimultaneousBand { interpretation, .. } => {
                    Some(response_interval_component(
                        *interpretation,
                        "posterior_simultaneous_band",
                        "response_simultaneous_confidence_band",
                    ))
                }
                ResponseUncertainty::IdentifiedEnvelopeBand { interpretation, .. } => {
                    Some(response_interval_component(
                        *interpretation,
                        "posterior_envelope_band",
                        "response_envelope_confidence_band",
                    ))
                }
                ResponseUncertainty::Posterior { .. } => {
                    Some((UncertaintySource::Parameter, "posterior_artifact"))
                }
            };
            if let Some((source, target)) = target {
                components.push(UncertaintyComponent::new(source, target, false));
            }
        }
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
        // Only overlap / positivity checks speak to support, and only a pass
        // supports it: a failed check is recorded as `failed:<refuter>`.
        slot.empirical = antecedent_io::support_empirical(&body.refutations).map_or_else(
            || SlotAvailability::unavailable("not_evaluated"),
            |label| SlotAvailability::Available(Arc::from(label)),
        );
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

fn body_frame(
    query: &antecedent_core::CausalQuery,
    temporal: Option<&CachedTemporalIdentification>,
    cached: Option<&IdentificationResult>,
    registry: Option<&antecedent_core::PopulationRegistry>,
) -> Result<BodyFrame, CausalError> {
    Ok(BodyFrame {
        query: causal_query_to_wire_with_registry(query, registry).map_err(|err| io_err(&err))?,
        identification: cached
            .map(|result| identification_to_wire_with_registry(result, registry))
            .transpose()
            .map_err(|err| io_err(&err))?,
        identification_status: cached.map(|result| result.status),
        temporal_identification: temporal_identification_wires(temporal, registry)?,
        registry: registry.cloned(),
    })
}

/// Executed body: the program's frame plus everything the result reports.
fn body_for(frame: &BodyFrame, result: &StudyResult) -> Result<AnalysisResultWire, CausalError> {
    let mut identification = match &frame.identification {
        Some(identification) => identification.clone(),
        None => {
            identification_to_wire_with_registry(&result.identification, frame.registry.as_ref())
                .map_err(|err| io_err(&err))?
        }
    };
    let temporal_identification = frame.temporal_identification.clone();
    let identification_variables = temporal_identification
        .iter()
        .find(|entry| entry.identification.query == identification.query)
        .or_else(|| temporal_identification.first())
        .map(|entry| entry.variables.clone());
    identification.query = frame.query.clone();
    let published = crate::PublishedScalarUncertainty::select(&result.estimate);
    let mut wire = AnalysisResultWire {
        query: frame.query.clone(),
        identification,
        identification_variables,
        temporal_identification,
        estimate: executed_scalar(result),
        interventional_distribution: result.distribution.as_ref().map(|distribution| {
            antecedent_io::InterventionalDistributionWire {
                atoms: distribution
                    .atoms
                    .iter()
                    .map(|atom| antecedent_io::DistributionAtomWire {
                        outcomes: atom
                            .outcomes
                            .iter()
                            .map(|(variable, value)| {
                                (variable.raw(), antecedent_io::ValueWire::from_value(value))
                            })
                            .collect(),
                        conditioning: atom
                            .conditioning
                            .iter()
                            .map(|(variable, value)| {
                                (variable.raw(), antecedent_io::ValueWire::from_value(value))
                            })
                            .collect(),
                        probability: atom.probability,
                    })
                    .collect(),
            }
        }),
        standard_error: published.standard_error,
        interval_lower: published.lower,
        interval_upper: published.upper,
        assumptions: antecedent_io::assumptions_to_wire(&result.estimate.assumptions),
        diagnostics: result.diagnostics.iter().map(antecedent_io::diagnostic_to_wire).collect(),
        refutations: result.refutations.iter().map(antecedent_io::refutation_to_wire).collect(),
        response: None,
        posterior_artifact: None,
        mediation_grid: None,
        structural_response: None,
        unit_effects: None,
        cate: result.estimate.cate.as_ref().map(|v| v.to_vec()),
        fitted_effect: result.estimate.fitted_effect.as_deref().cloned(),
        cate_se: result.estimate.cate_se.as_ref().map(|v| v.to_vec()),
        cate_leaf_dispersion: result.estimate.cate_leaf_dispersion.as_ref().map(|v| v.to_vec()),
        outcome_oof_r2: result.estimate.outcome_oof_r2,
        treatment_oof_logloss: result.estimate.treatment_oof_logloss,
        crossfit_folds: result.estimate.crossfit_folds,
        crossfit_seed: result.estimate.crossfit_seed,
        learner_provenance: result
            .estimate
            .learner_provenance
            .iter()
            .map(|p| (p.spec.clone(), p.implementation.clone(), p.version.clone()))
            .collect(),
    };
    result.fill_analysis_result_payloads(&mut wire, "execution-posterior")?;
    Ok(wire)
}

fn temporal_identification_wires(
    temporal: Option<&CachedTemporalIdentification>,
    registry: Option<&antecedent_core::PopulationRegistry>,
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
                identification: identification_to_wire_with_registry(
                    &entry.identification,
                    registry,
                )
                .map_err(|err| io_err(&err))?,
            })
        })
        .collect()
}

pub(super) fn reasoning_section(view: &ReasoningView) -> ReasoningSectionWire {
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
                weight_basis: slot.weight_basis.as_ref().map(std::string::ToString::to_string),
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
        calibration: calibration_wire(&claim.calibration),
        attested: claim
            .attested
            .iter()
            .map(|item| AttestedEvidenceWire {
                name: item.name.to_string(),
                kind: item.kind.to_string(),
                passed: item.passed,
                refuted_ate: item.refuted_ate,
                comparison: item.comparison,
                informative: item.informative,
                failure_condition: item
                    .failure_condition
                    .as_ref()
                    .map(std::string::ToString::to_string),
                reverifiable: item.reverifiable,
                config_digest: item.config_digest.as_ref().map(std::string::ToString::to_string),
                payload_digest: item.payload_digest.as_ref().map(std::string::ToString::to_string),
            })
            .collect(),
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

fn identification_slot_from_prepared(
    study: &Study,
    result: &IdentificationResult,
    search_capped: bool,
) -> IdentificationSlot {
    if let Some(cache) = study.temporal_class_identification_cache.as_ref() {
        return slot_from_envelope(
            result.status,
            &cache.envelope.envelope,
            "completion_enumeration",
            search_capped,
        );
    }
    if let Some(cache) = study.cpdag_identification_cache.as_ref() {
        return slot_from_envelope(
            cache.identification.status,
            &cache.envelope,
            "completion_enumeration",
            search_capped,
        );
    }
    if let Some(cache) = study.pag_identification_cache.as_ref() {
        return slot_from_envelope(
            cache.identification.status,
            &cache.envelope,
            "completion_enumeration",
            search_capped,
        );
    }
    if let Some(cache) = study.graph_posterior_identification_cache.as_ref() {
        return slot_from_graph_samples(result.status, &cache.graphs, search_capped);
    }
    if let Some(cache) = study.dbn_posterior_identification_cache.as_ref() {
        return slot_from_graph_samples(result.status, &cache.graphs, search_capped);
    }
    let mut slot = IdentificationSlot::identified_singleton(result.status);
    slot.search_capped = search_capped;
    if search_capped {
        slot.full_mass_scope = false;
    }
    slot
}

/// Project an enumerated class envelope onto the identification slot.
///
/// Weight whose search was capped before it could decide is incomplete search,
/// not proof of non-identification (ADR 0022). Zero examined mass is the same
/// kind of ignorance: it is never reported as fully unidentified.
fn slot_from_envelope<G>(
    status: IdentificationStatus,
    envelope: &IdentificationEnvelope<G>,
    basis: &'static str,
    search_capped: bool,
) -> IdentificationSlot {
    let identified = envelope.identified_weight.0;
    let unidentified = envelope.unidentified_weight.0;
    let truncated = envelope.truncated_weight().clamp(0.0, unidentified.max(0.0));
    let total = identified + unidentified;
    let (identified_mass, unidentified_mass, incomplete_mass) = if total > 0.0 {
        (identified / total, (unidentified - truncated) / total, truncated / total)
    } else {
        (0.0, 0.0, 1.0)
    };
    let capped = search_capped || envelope.truncated_completions > 0;
    IdentificationSlot::new(
        status,
        identified_mass,
        unidentified_mass,
        0.0,
        incomplete_mass,
        total > 0.0 && !capped,
        Some(Arc::from(basis)),
        capped,
    )
}

fn slot_from_graph_samples(
    status: IdentificationStatus,
    graphs: &antecedent_prob::WeightedGraphSamples,
    search_capped: bool,
) -> IdentificationSlot {
    let identified = graphs.identified_mass();
    let unidentified = graphs.unidentified_mass();
    let total = identified + unidentified;
    // No examined mass tells us nothing; it is not a fully unidentified class.
    let (identified_mass, unidentified_mass, incomplete_mass) =
        if total > 0.0 { (identified / total, unidentified / total, 0.0) } else { (0.0, 0.0, 1.0) };
    IdentificationSlot::new(
        status,
        identified_mass,
        unidentified_mass,
        0.0,
        incomplete_mass,
        total > 0.0 && !search_capped,
        Some(Arc::from("posterior_probability")),
        search_capped,
    )
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
        let unidentified = unidentified_mass_from_diagnostics(&result.diagnostics)?;
        return Some((unidentified, 0.0, basis));
    }
    None
}

/// Identification mass carried on a diagnostic's structured fields.
///
/// Prose is never parsed: a message that mentions only `unidentified_mass`
/// once read as 50% identified.
fn unidentified_mass_from_diagnostics(diagnostics: &[antecedent_core::Diagnostic]) -> Option<f64> {
    diagnostics.iter().find_map(|diagnostic| {
        let unidentified = mass_field(diagnostic, "unidentified_mass")?;
        if let Some(identified) = mass_field(diagnostic, "identified_mass") {
            let total = identified + unidentified;
            if total > 0.0 {
                return Some(unidentified / total);
            }
        }
        Some(unidentified)
    })
}

fn mass_field(diagnostic: &antecedent_core::Diagnostic, field: &str) -> Option<f64> {
    diagnostic
        .fields
        .iter()
        .find(|(key, _)| key.as_ref() == field)
        .and_then(|(_, value)| value.parse().ok())
}

fn identification_status_wire(
    status: antecedent_core::IdentificationStatus,
) -> antecedent_io::IdentificationStatusWire {
    match status {
        antecedent_core::IdentificationStatus::NonparametricallyIdentified => {
            antecedent_io::IdentificationStatusWire::NonparametricallyIdentified
        }
        antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions => {
            antecedent_io::IdentificationStatusWire::IdentifiedUnderParametricRestrictions
        }
        antecedent_core::IdentificationStatus::IdentifiedUnderPriorRestrictions => {
            antecedent_io::IdentificationStatusWire::IdentifiedUnderPriorRestrictions
        }
        antecedent_core::IdentificationStatus::PartiallyIdentified => {
            antecedent_io::IdentificationStatusWire::PartiallyIdentified
        }
        antecedent_core::IdentificationStatus::GraphDependent => {
            antecedent_io::IdentificationStatusWire::GraphDependent
        }
        antecedent_core::IdentificationStatus::NotIdentified => {
            antecedent_io::IdentificationStatusWire::NotIdentified
        }
    }
}

fn mediation_grid_wire(
    grid: &crate::estimate::TemporalMediationGrid,
) -> antecedent_io::TemporalMediationGridWire {
    let interval = |summary: antecedent_estimate::MediationPosteriorSummary| {
        antecedent_io::MediationPosteriorSummaryWire {
            mean: summary.mean,
            standard_deviation: summary.standard_deviation,
            q025: summary.q025,
            q975: summary.q975,
        }
    };
    antecedent_io::TemporalMediationGridWire {
        slices: grid
            .slices
            .iter()
            .map(|slice| {
                let uncertainty = match &slice.uncertainty {
                    antecedent_estimate::TemporalMediationUncertainty::FrequentistPointwise {
                        standard_error,
                    }
                    | antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                        requested: standard_error,
                        ..
                    } => antecedent_io::TemporalMediationUncertaintyWire::FrequentistPointwise {
                        standard_error: *standard_error,
                    },
                    antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
                        requested,
                        total,
                        direct,
                        mediated,
                        n_draws,
                        backend,
                    } => antecedent_io::TemporalMediationUncertaintyWire::BayesianPointwise {
                        requested: interval(*requested),
                        total: interval(*total),
                        direct: interval(*direct),
                        mediated: interval(*mediated),
                        n_draws: u64::try_from(*n_draws).unwrap_or(u64::MAX),
                        backend: backend.to_string(),
                    },
                    _ => antecedent_io::TemporalMediationUncertaintyWire::Unavailable,
                };
                antecedent_io::TemporalMediationSliceWire {
                    horizon: slice.horizon,
                    identification_status: identification_status_wire(slice.identification_status),
                    method: slice.method.to_string(),
                    adjustment: slice
                        .adjustment
                        .iter()
                        .map(|key| antecedent_io::HorizonAdjustmentNodeWire {
                            variable: key.variable.raw(),
                            offset: key.offset,
                        })
                        .collect(),
                    effect: slice.estimate.effect.ate,
                    total: slice.estimate.total,
                    direct: slice.estimate.direct,
                    mediated: slice.estimate.mediated,
                    uncertainty,
                    identified_set: slice
                        .identified_set
                        .map(|identified| [identified.lower, identified.upper]),
                    diagnostics: slice
                        .diagnostics
                        .iter()
                        .map(antecedent_io::diagnostic_to_wire)
                        .collect(),
                }
            })
            .collect(),
        joint_posterior: grid.joint_posterior,
    }
}

fn structural_response_wire(
    mixture: &crate::result::StructuralResponseMixture,
) -> Result<antecedent_io::StructuralResponseMixtureWire, CausalError> {
    let weight_basis = antecedent_io::StructuralWeightBasisWire::from(mixture.weight_basis);
    // Core completion weights may be counts, or be normalized separately at
    // each horizon. The portable container requires unit total while retaining
    // the weight basis and every relative weight; no weights are inferred.
    let total: f64 = mixture.atoms.iter().map(|atom| atom.weight).sum();
    if !mixture.atoms.is_empty() && (!total.is_finite() || total <= 0.0) {
        return Err(CausalError::Compile {
            message: "cannot export structural atoms with invalid total weight".into(),
        });
    }
    Ok(antecedent_io::StructuralResponseMixtureWire {
        weight_basis,
        atoms: mixture
            .atoms
            .iter()
            .map(|atom| {
                Ok(antecedent_io::StructuralResponseAtomWire {
                    graph_key: atom.graph_key,
                    weight: atom.weight / total,
                    identification_status: identification_status_wire(atom.status),
                    value: atom.value.as_ref().map(antecedent_io::response_value_to_wire),
                    posterior_artifact: atom
                        .posterior
                        .as_ref()
                        .map(|posterior| {
                            antecedent_io::encode_causal_posterior_bytes(
                                posterior,
                                "structural_atom",
                            )
                        })
                        .transpose()
                        .map_err(|err| io_err(&err))?,
                    response: atom
                        .response
                        .as_ref()
                        .map(antecedent_io::causal_response_to_wire)
                        .transpose()
                        .map_err(|err| io_err(&err))?,
                })
            })
            .collect::<Result<Vec<_>, CausalError>>()?,
        identified_mass: mixture.identified_mass,
        unidentified_mass: mixture.unidentified_mass,
        unevaluable_mass: mixture.unevaluable_mass,
        subsampled_out_mass: mixture.subsampled_out_mass,
        identified_set: mixture.identified_set.as_ref().map(|envelope| {
            antecedent_io::ResponseEnvelopeWire {
                grid: envelope.grid.to_vec(),
                dimension: u64::try_from(envelope.dimension).unwrap_or(u64::MAX),
                lower: envelope.lower.to_vec(),
                upper: envelope.upper.to_vec(),
            }
        }),
        identified_set_interval: mixture
            .identified_set_interval
            .as_ref()
            .map(antecedent_io::identified_set_interval_to_wire)
            .transpose()
            .map_err(|err| io_err(&err))?,
        conditional_on_identified: mixture
            .conditional_on_identified
            .as_ref()
            .map(antecedent_io::response_value_to_wire),
        full_mass_scope: mixture.full_mass_scope,
        truncated_atoms: u64::try_from(mixture.truncated_atoms).unwrap_or(u64::MAX),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{AverageEffectQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity};

    #[test]
    fn response_interval_component_follows_the_interval_tag_not_the_inference_mode() {
        assert_eq!(
            response_interval_component(IntervalInterpretation::Credible, "posterior", "sampling"),
            (UncertaintySource::Parameter, "posterior")
        );
        assert_eq!(
            response_interval_component(
                IntervalInterpretation::Confidence,
                "posterior",
                "sampling"
            ),
            (UncertaintySource::Sampling, "sampling")
        );
    }
    use antecedent_identify::{
        CAPPED_COMPLETION_DIAGNOSTIC_CODE, DerivationTrace, GraphIdentificationCase,
        IdentificationPerformanceRecord, ProbabilityMass,
    };

    fn case(
        status: IdentificationStatus,
        weight: f64,
        capped: bool,
    ) -> GraphIdentificationCase<u32> {
        let diagnostics = if capped {
            vec![Diagnostic::new(
                CAPPED_COMPLETION_DIAGNOSTIC_CODE,
                DiagnosticKind::Execution,
                DiagnosticSeverity::Warning,
                "completion enumeration exceeded its budget",
            )]
        } else {
            Vec::new()
        };
        GraphIdentificationCase {
            graph: 0,
            result: IdentificationResult::from_parts(
                status,
                CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
                    antecedent_core::VariableId::from_raw(0),
                    antecedent_core::VariableId::from_raw(1),
                )),
                Vec::new(),
                antecedent_expr::CausalExprArena::new(),
                DerivationTrace::default(),
                AssumptionSet::default(),
                diagnostics,
                IdentificationPerformanceRecord::default(),
                None,
            ),
            weight: ProbabilityMass(weight),
        }
    }

    #[test]
    fn capped_completions_are_incomplete_search_not_unidentified_mass() {
        let envelope = IdentificationEnvelope::from_cases(vec![
            case(IdentificationStatus::NonparametricallyIdentified, 0.5, false),
            case(IdentificationStatus::NotIdentified, 0.25, false),
            case(IdentificationStatus::NotIdentified, 0.25, true),
        ]);
        let slot = slot_from_envelope(envelope.status, &envelope, "completion_enumeration", false);
        assert!((slot.identified_mass - 0.5).abs() < 1e-12, "{slot:?}");
        assert!((slot.unidentified_mass - 0.25).abs() < 1e-12, "{slot:?}");
        assert!((slot.incomplete_search_mass - 0.25).abs() < 1e-12, "{slot:?}");
        assert!(slot.search_capped);
        assert!(!slot.full_mass_scope);
        antecedent_io::validate_mixture_masses(
            slot.identified_mass,
            slot.unidentified_mass,
            slot.unevaluable_mass,
            slot.incomplete_search_mass,
        )
        .expect("masses conserve");
    }

    // Exact 0.0 / 1.0 are the facts under test: no capped case contributes
    // exactly no mass, and an unexamined envelope is entirely incomplete.
    #[allow(
        clippy::float_cmp,
        reason = "an empty search leaves the mass exactly 0.0, which is the claim under test"
    )]
    #[test]
    fn a_complete_envelope_reports_no_incomplete_search_mass() {
        let envelope = IdentificationEnvelope::from_cases(vec![
            case(IdentificationStatus::NonparametricallyIdentified, 0.5, false),
            case(IdentificationStatus::NotIdentified, 0.5, false),
        ]);
        let slot = slot_from_envelope(envelope.status, &envelope, "completion_enumeration", false);
        assert_eq!(slot.incomplete_search_mass, 0.0);
        assert!((slot.unidentified_mass - 0.5).abs() < 1e-12);
        assert!(slot.full_mass_scope);
        assert!(!slot.search_capped);
    }

    #[test]
    fn identification_mass_comes_from_structured_fields_not_prose() {
        let prose = Diagnostic::new(
            "estimate.pag.nonidentified_prior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "Pag not identified; returning prior-predictive draws (unidentified_mass=0.8)",
        );
        assert_eq!(
            unidentified_mass_from_diagnostics(std::slice::from_ref(&prose)),
            None,
            "prose is not a mass field"
        );
        let unidentified_only = prose.clone().with_fields([("unidentified_mass", "0.8")]);
        assert_eq!(unidentified_mass_from_diagnostics(&[unidentified_only]), Some(0.8));
        let mixture =
            prose.with_fields([("identified_mass", "0.25"), ("unidentified_mass", "0.75")]);
        assert_eq!(unidentified_mass_from_diagnostics(&[mixture]), Some(0.75));
    }

    // Exact 0.0 / 1.0 are the facts under test: no capped case contributes
    // exactly no mass, and an unexamined envelope is entirely incomplete.
    #[allow(
        clippy::float_cmp,
        reason = "an unexamined envelope has exactly 0.0 unidentified and 1.0 incomplete mass, which is the claim under test"
    )]
    #[test]
    fn nothing_examined_is_not_proof_of_non_identification() {
        let envelope: IdentificationEnvelope<u32> = IdentificationEnvelope::from_cases(Vec::new());
        let slot = slot_from_envelope(envelope.status, &envelope, "completion_enumeration", true);
        assert_eq!(slot.unidentified_mass, 0.0);
        assert_eq!(slot.incomplete_search_mass, 1.0);
        assert!(!slot.full_mass_scope);
        assert!(slot.search_capped);
    }
}

#[cfg(test)]
mod frontdoor_artifact_tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::{AnalyticSeKind, FrontDoorTwoStage};
    use antecedent_graph::Dag;

    use crate::{Study, analysis::builder::RefuteSuite, strategy_table::IdentifierId};

    fn fixture() -> TabularData {
        let mut builder = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("m", RoleHint::Context),
        ] {
            builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(hint),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = builder.build().unwrap();
        let ids =
            [schema.id_of("t").unwrap(), schema.id_of("y").unwrap(), schema.id_of("m").unwrap()];
        let n = 160;
        let mut treatment = Vec::with_capacity(n);
        let mut mediator = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        for i in 0..(n / 2) {
            let mediator_noise = ((i * 17 % 101) as f64 - 50.0) / 65.0;
            let outcome_noise = ((i * 31 % 97) as f64 - 48.0) / 42.0;
            for t in [0.0, 1.0] {
                treatment.push(t);
                mediator.push(0.8 * t + mediator_noise);
                outcome.push(2.0 * (0.8 * t + mediator_noise) + outcome_noise);
            }
        }
        let columns = [treatment, outcome, mediator]
            .into_iter()
            .zip(ids)
            .map(|(values, id)| {
                OwnedColumn::Float64(
                    Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n))
                        .unwrap(),
                )
            })
            .collect();
        TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
    }

    #[test]
    fn rehashed_frontdoor_procedure_tampering_is_refused_semantically() {
        let data = fixture();
        let graph = Dag::from_named_edges(data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
        let query = AverageEffectQuery::with_levels(
            data.schema().id_of("t").unwrap(),
            data.schema().id_of("y").unwrap(),
            0.0,
            1.0,
        );
        let study = Study::tabular(data.clone())
            .graph(graph)
            .query(query)
            .identifier(IdentifierId::Frontdoor)
            .estimator(
                FrontDoorTwoStage::new()
                    .with_bootstrap_replicates(0)
                    .with_se_kind(AnalyticSeKind::Hc1),
            )
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let context = ExecutionContext::for_tests(901);
        let prepared = study.prepare(&context).unwrap();
        let result = prepared.estimate(&data, &context).unwrap();
        let bytes = prepared.encode_contracted_result(&result, "frontdoor", &context).unwrap();
        let intact = antecedent_io::consume_analysis_result(&bytes).unwrap();
        assert!(
            intact.acceptance.accepts_as_verified_program(),
            "{:?}",
            intact.acceptance.unresolved
        );

        let (artifact, _header, body) =
            antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
        let mut contract =
            antecedent_io::decode_analysis_result_contract(&artifact).unwrap().unwrap();
        let lowering =
            contract.program.as_mut().unwrap().checked_frontdoor_lowering.as_mut().unwrap();
        lowering.procedure = "frontdoor_functional".into();
        let program_digest =
            antecedent_io::program_digest(contract.program.as_ref().unwrap()).unwrap();
        contract.identities.program = *program_digest.as_bytes();
        contract.seal = antecedent_io::contract_seal(
            &contract.identities,
            &contract.reasoning,
            &contract.graph_class,
            &contract.structure_source,
            contract.identifier.as_deref(),
            contract.estimator.as_deref(),
        )
        .unwrap();
        let claim = contract.claim.as_mut().unwrap();
        let result_digest = antecedent_io::result_digest(&body).unwrap();
        claim.claim_id = *antecedent_io::claim_digest(&antecedent_io::ClaimIdentityWire::new(
            contract.seal,
            claim,
            result_digest,
        ))
        .unwrap()
        .as_bytes();
        let payload = antecedent_io::to_cbor(&contract).unwrap();
        let (descriptor, section) = antecedent_io::pack_section(
            antecedent_io::CONTRACT_SECTION,
            "application/cbor",
            payload,
            antecedent_io::CompressPolicy::Auto,
        );
        let mut tampered = artifact;
        let section_index = tampered
            .manifest
            .sections
            .iter()
            .position(|descriptor| descriptor.id == antecedent_io::CONTRACT_SECTION)
            .unwrap();
        tampered.manifest.sections[section_index] = descriptor;
        tampered.sections[section_index] = section;
        let mut tampered_bytes = Vec::new();
        tampered.write_to(&mut tampered_bytes).unwrap();
        let consumed = antecedent_io::consume_analysis_result(&tampered_bytes).unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|item| item.as_ref() == "program.frontdoor_binding"),
            "{:?}",
            consumed.acceptance.unresolved
        );
    }
}

#[cfg(test)]
mod checked_iv_artifact_tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::{TwoStageLeastSquares, WaldIv};
    use antecedent_graph::Dag;

    use crate::{Study, analysis::builder::RefuteSuite, strategy_table::IdentifierId};

    fn fixture() -> TabularData {
        let mut builder = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("z", RoleHint::InstrumentCandidate),
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
        ] {
            builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(hint),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = builder.build().unwrap();
        let ids =
            [schema.id_of("z").unwrap(), schema.id_of("t").unwrap(), schema.id_of("y").unwrap()];
        let n = 160;
        let z = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
        let t = z.clone();
        let y =
            (0..n).map(|i| 2.0 * t[i] + ((i * 17 % 31) as f64 - 15.0) / 20.0).collect::<Vec<_>>();
        let columns = [z, t, y]
            .into_iter()
            .zip(ids)
            .map(|(values, id)| {
                OwnedColumn::Float64(
                    Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n))
                        .unwrap(),
                )
            })
            .collect();
        TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
    }

    fn tamper_procedure(bytes: &[u8]) -> Vec<u8> {
        let (artifact, _header, body) =
            antecedent_io::decode_analysis_result_artifact(bytes).unwrap();
        let mut contract =
            antecedent_io::decode_analysis_result_contract(&artifact).unwrap().unwrap();
        let lowering = contract.program.as_mut().unwrap().checked_iv_lowering.as_mut().unwrap();
        lowering.procedure = if lowering.procedure == "wald" {
            "two_stage_least_squares".into()
        } else {
            "wald".into()
        };
        let digest = antecedent_io::program_digest(contract.program.as_ref().unwrap()).unwrap();
        contract.identities.program = *digest.as_bytes();
        contract.seal = antecedent_io::contract_seal(
            &contract.identities,
            &contract.reasoning,
            &contract.graph_class,
            &contract.structure_source,
            contract.identifier.as_deref(),
            contract.estimator.as_deref(),
        )
        .unwrap();
        let claim = contract.claim.as_mut().unwrap();
        let result_digest = antecedent_io::result_digest(&body).unwrap();
        claim.claim_id = *antecedent_io::claim_digest(&antecedent_io::ClaimIdentityWire::new(
            contract.seal,
            claim,
            result_digest,
        ))
        .unwrap()
        .as_bytes();
        let payload = antecedent_io::to_cbor(&contract).unwrap();
        let (descriptor, section) = antecedent_io::pack_section(
            antecedent_io::CONTRACT_SECTION,
            "application/cbor",
            payload,
            antecedent_io::CompressPolicy::Auto,
        );
        let mut artifact = artifact;
        let index = artifact
            .manifest
            .sections
            .iter()
            .position(|item| item.id == antecedent_io::CONTRACT_SECTION)
            .unwrap();
        artifact.manifest.sections[index] = descriptor;
        artifact.sections[index] = section;
        let mut out = Vec::new();
        artifact.write_to(&mut out).unwrap();
        out
    }

    #[test]
    fn checked_wald_and_binary_2sls_artifacts_verify_and_rehashed_procedure_tampering_fails() {
        let data = fixture();
        let graph = Dag::from_named_edges(data.schema(), &[("z", "t"), ("t", "y")]).unwrap();
        let query = AverageEffectQuery::binary_ate(
            data.schema().id_of("t").unwrap(),
            data.schema().id_of("y").unwrap(),
        );
        for (estimator, expected) in [(0, "wald"), (1, "two_stage_least_squares")] {
            let study = Study::tabular(data.clone())
                .graph(graph.clone())
                .query(query.clone())
                .identifier(IdentifierId::Iv)
                .estimator(if estimator == 0 {
                    crate::EstimatorSpec::from(WaldIv::new().with_bootstrap_replicates(0))
                } else {
                    crate::EstimatorSpec::from(
                        TwoStageLeastSquares::new().with_bootstrap_replicates(0),
                    )
                })
                .refute(RefuteSuite::None)
                .build()
                .unwrap();
            let context = ExecutionContext::for_tests(903 + estimator as u64);
            let prepared = study.prepare(&context).unwrap();
            let result = prepared.estimate(&data, &context).unwrap();
            let bytes = prepared.encode_contracted_result(&result, "iv", &context).unwrap();
            let intact = antecedent_io::consume_analysis_result(&bytes).unwrap();
            assert!(
                intact.acceptance.accepts_as_verified_program(),
                "{expected}: {:?}",
                intact.acceptance.unresolved
            );
            let (artifact, _, _) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
            let contract =
                antecedent_io::decode_analysis_result_contract(&artifact).unwrap().unwrap();
            assert_eq!(contract.program.unwrap().checked_iv_lowering.unwrap().procedure, expected);
            let tampered = tamper_procedure(&bytes);
            let consumed = antecedent_io::consume_analysis_result(&tampered).unwrap();
            assert!(
                !consumed.acceptance.accepts_as_verified_program(),
                "{expected}: {:?}",
                consumed.acceptance.unresolved
            );
            assert!(
                consumed
                    .acceptance
                    .unresolved
                    .iter()
                    .any(|item| item.as_ref() == "program.checked_iv_binding")
            );
        }
    }
}
