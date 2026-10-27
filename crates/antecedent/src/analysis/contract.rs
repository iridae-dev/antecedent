//! Causal-contract companion for [`Study`] / [`PreparedStudy`].
//!
//! ADR 0022: the practitioner handle stays [`PreparedStudy`]. This module
//! builds an inspectable contract from existing products. `inspect` is cheap
//! structural classification; `contract` includes cached identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;
use std::sync::Arc;

use antecedent_contract::{
    AssumptionSlot, AttestedEvidence, CalibrationView, ClaimEnvelope, ClaimKind,
    IDENTIFICATION_NOT_CLAIMED, IdentificationSlot, NextAction, ObligationKind, ObligationRecord,
    ObligationScope, OperationKind, OperationReadiness, OperationReport, ReasoningView,
    SemanticApplicability, SemanticLayer, SlotAvailability, SupportSlot, TransformIntent,
    TransformationReport, UncertaintyComponent, UncertaintySlot, UncertaintySource, intent_effects,
};
use antecedent_core::{
    Assumption, AssumptionSet, AssumptionSource, AssumptionStatus, CausalQuery, CausalSchema,
    ContractIdentities, ExecutionContext, IDENTITY_FORMAT, IdentificationStatus, IdentityDomain,
    IntervalInterpretation, IntervalMethod, ResponseUncertainty, TargetPopulation,
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
    temporal_admg_identity, temporal_cpdag_identity, temporal_dag_identity, temporal_pag_identity,
};

use crate::accepted::{AcceptedGraph, GraphClass, GraphRef};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::result::{IntervalBinding, ResultPayload, RowWeightsBinding, StudyResult};
use crate::support::{CellStatus, StructureSource};

use super::batch::PreparedBatch;
use super::builder::DataInput;
use super::execute::Study;
use super::missing_data;
use super::prepared::{
    CachedTemporalHorizonIdentification, CachedTemporalIdentification, CheckedProgramBinding,
    PreparedStudy,
};

mod evidence_needs;
mod failure_inputs;
mod posterior_programs;
pub(crate) use posterior_programs::graph_posterior_programs;
mod calibration;
mod claim;
mod compile;
mod identification_projection;
mod prepared_inspection;
pub use prepared_inspection::PreparedInspection;
mod program_wire;
mod reasoning;
mod result_wire;
mod wire;

#[allow(
    clippy::wildcard_imports,
    reason = "the split modules re-export their items under the contract module's one path"
)]
pub(crate) use self::calibration::*;
#[allow(
    clippy::wildcard_imports,
    reason = "the split modules re-export their items under the contract module's one path"
)]
pub(crate) use self::claim::*;
#[allow(
    clippy::wildcard_imports,
    reason = "the split modules re-export their items under the contract module's one path"
)]
pub(crate) use self::compile::*;
#[allow(
    clippy::wildcard_imports,
    reason = "the split modules re-export their items under the contract module's one path"
)]
pub(crate) use self::identification_projection::*;
#[allow(
    clippy::wildcard_imports,
    reason = "the split modules re-export their items under the contract module's one path"
)]
pub(crate) use self::reasoning::*;
#[allow(
    clippy::wildcard_imports,
    reason = "the split modules re-export their items under the contract module's one path"
)]
pub(crate) use self::wire::*;

pub(crate) use evidence_needs::evidence_next_actions;
pub(crate) use failure_inputs::{FailureInputs, failure_inputs, failure_inputs_of};

/// The verified obligations and evidence-gathering next actions of the not-identified
/// product `result` that `study` retained at compile: the same derivation a prepared
/// contract performs, for a study whose preparation refuses.
pub(crate) fn failure_evidence_needs(
    study: &Study,
    result: &IdentificationResult,
) -> (Arc<[antecedent_contract::EvidenceObligation]>, Arc<[antecedent_contract::NextAction]>) {
    evidence_needs::derive(study, Some(result))
}
use result_wire::{LicensedWire, distribution_wire};

/// Bind an executed graphless license onto the contract's support reasoning.
///
/// Every graphless family sets the same fields once the realized support
/// status is known: the contract-level `support_status`, then the support
/// slot's `matrix_status` and `matrix_coordinate`. A result carrying none of
/// those families leaves the contract untouched. The coordinate is derived
/// only when the row is actually licensed.
///
/// # Errors
///
/// [`CausalError::Conflict`] when a licensed result's payload does not match
/// the handle's query, which the contract identity comparison upstream rules
/// out for any result this handle executed.
fn bind_graphless_reasoning(
    contract: &mut CausalContract,
    result: &StudyResult,
    query: &CausalQuery,
) -> Result<(), CausalError> {
    let graphless = result.continuous_dose_response().is_some()
        || result.policy_value().is_some()
        || result.panel_did().is_some()
        || result.survival().is_some()
        || result.randomized_effect().is_some()
        || result.longitudinal_regime().is_some()
        || result.sequential_contrast().is_some()
        || result.history_match().is_some();
    if !graphless {
        return Ok(());
    }
    let support_status = result.support_status;
    let coordinate = if support_status == Some(CellStatus::Licensed) {
        Some(graphless_coordinate(result, query)?)
    } else {
        None
    };
    contract.support_status = support_status;
    if let SlotAvailability::Available(support) = &mut contract.reasoning.support {
        support.matrix_status = Arc::from(support_status.map_or("off_axis", CellStatus::as_str));
        support.matrix_coordinate = coordinate;
    }
    Ok(())
}

/// Matrix coordinate of a licensed graphless result.
///
/// The coordinate is a function of the query's facets, shared with the
/// execute-side licence so the published coordinate is the key the design was
/// licensed under. A payload whose query facet is absent, or a calibrated
/// route that cannot be licensed, is a typed conflict rather than a panic.
///
/// Interference and `local_polynomial_ratio` are deliberately absent. They
/// carry their licence on the per-fit `graphless_support_status`, rechecked
/// by `validate_result` against the published interval, and stay off the
/// matrix axis so calibration verification is not broken by a coordinate no
/// route expects.
fn graphless_coordinate(
    result: &StudyResult,
    query: &CausalQuery,
) -> Result<Arc<str>, CausalError> {
    const MISMATCH: CausalError = CausalError::Conflict {
        what: "result",
        detail: "licensed graphless result does not match this handle's query",
    };
    let mut coordinate = None;
    if result.continuous_dose_response().is_some() {
        coordinate = Some(
            "graphless:continuous_dose_policy/fixed_group_kernel/inverse_density_kernel_paired_scores/policy_reference_incremental_pointwise_95_normal_intervals"
                .to_string(),
        );
    }
    if result.policy_value().is_some() {
        let CausalQuery::PolicyValue(query) = query else { return Err(MISMATCH) };
        let multi = query.multi_action.is_some();
        let uplift = query.uplift_bin_count > 0;
        let cate = query.multi_action.as_ref().is_some_and(|policy| !policy.cate_groups.is_empty());
        let ipw = query.mu0.is_empty();
        let (design, method, claim) = antecedent_core::policy_graphless_coordinate(
            multi,
            ipw,
            uplift,
            cate,
            query.regret.is_some(),
            !ipw && query.crossfit_fold_ownership_valid && !query.disjoint_training_subjects,
        );
        coordinate = Some(format!("graphless:policy_value/{design}/{method}/{claim}"));
    }
    if result.panel_did().is_some() {
        let CausalQuery::PanelDid(query) = query else { return Err(MISMATCH) };
        let key = super::execute::panel_did_graphless_key(query.design);
        coordinate = Some(format!(
            "graphless:{}/{}/{}/{}",
            key.family, key.design, key.method, key.inference_claim
        ));
    }
    if result.survival().is_some() {
        // The licensed scalar route is fixed by the query's entry and
        // censoring facets. Cumulative incidence never reaches this arm; only
        // the paired RMST + horizon survival scalar intervals carry a
        // graphless matrix coordinate. The simultaneous band is a separate
        // claim carried on the band's own `graphless_support_status`.
        let CausalQuery::Survival(query) = query else { return Err(MISMATCH) };
        let design = if query.delayed_entry.is_some() {
            "delayed_entry_two_arm_individual_randomized"
        } else {
            "two_arm_individual_randomized"
        };
        let method = if query.known_censoring.is_some() {
            "arm_stratified_subject_bootstrap_ipcw_product_limit"
        } else {
            "arm_stratified_subject_bootstrap_product_limit"
        };
        coordinate = Some(format!(
            "graphless:survival/{design}/{method}/rmst_and_horizon_survival_pointwise_95_percentile_intervals"
        ));
    }
    if result.randomized_effect().is_some() {
        // Preparation cannot know the realized outcome variance, so the
        // executed graphless license binds in the result-specific reasoning
        // and claim only after estimation. The Wald complier effect is its
        // own graphless family; a Bernoulli assignment design does not fold
        // it into ITT.
        let CausalQuery::RandomizedEffect(query) = query else { return Err(MISMATCH) };
        let (family, design, method, _assignment_unit, claim) =
            antecedent_core::randomized_graphless_coordinate(
                &query.design,
                query.estimand,
                query.received_treatment.is_some(),
                query.fixed_cuped.is_some(),
                !query.ancova_covariates.is_empty(),
            );
        coordinate = Some(format!("graphless:{family}/{design}/{method}/{claim}"));
    }
    if result.longitudinal_regime().is_some() {
        use antecedent_core::LongitudinalRegimeMethod as Method;
        let CausalQuery::LongitudinalRegime(query) = query else { return Err(MISMATCH) };
        let (design, method, claim) = match query.method {
            Method::SequentialDoublyRobust => (
                if query.periods == 2 {
                    "known_sequential_randomized_two_period"
                } else {
                    "known_sequential_randomized_three_period"
                },
                "subject_excluded_q_sequential_dr_scores",
                "conditional_q_pointwise_95_normal_interval",
            ),
            Method::GFormula => (
                "known_sequential_randomized_two_period",
                "fixed_known_q_g_formula_scores",
                "conditional_q_pointwise_95_normal_interval",
            ),
            Method::MarginalStructuralModel => (
                "known_sequential_randomized_additive_msm",
                "stabilized_ipw_cr1_scores",
                "intercept_and_period_effects_pointwise_95_normal_intervals",
            ),
            // Only calibrated graphless rows carry a licence.
            Method::Ipw => return Err(MISMATCH),
        };
        coordinate = Some(format!("graphless:longitudinal_regime/{design}/{method}/{claim}"));
    }
    coordinate.map(Arc::from).ok_or(MISMATCH)
}

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
    /// The compiled search and review the accepted structure came from, when it
    /// was accepted from a compiled discovery (see
    /// [`crate::accept_discovered`]). Identification excludes it, as it
    /// excludes the discovery algorithm; the program identity binds its digest.
    pub structure_provenance: Option<Arc<crate::accepted::StructureProvenance>>,
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
    /// Evidence that would resolve a not-identified product, one verified
    /// obligation per route with its proof step. Empty unless the contract holds
    /// a not-identified product (cheap inspection holds none).
    pub evidence_obligations: Arc<[antecedent_contract::EvidenceObligation]>,
    /// The evidence-gathering next actions those obligations imply
    /// (`measure_variable:<name>`, `run_experiment:<target>`, `find_instrument`,
    /// `collect_panel`), appended to the capability report's next actions.
    pub(crate) evidence_actions: Arc<[NextAction]>,
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
    /// Window namespace of a mediation under a temporal ADMG or PAG, whose identification names
    /// its nodes by window coordinate.
    window_variables: Option<Vec<HorizonAdjustmentNodeWire>>,
    /// Bindings for named or custom populations the query references.
    registry: Option<antecedent_core::PopulationRegistry>,
}

impl CausalContract {
    /// The cached identification wire and temporal products, without numerical execution.
    ///
    /// # Errors
    /// A certificate whose canonical wire cannot be serialized.
    pub fn identification_details(&self) -> Result<serde_json::Value, CausalError> {
        Ok(serde_json::json!({
            "status": self.body.identification.as_ref().map(|r| r.status.as_str()),
            "method": self.identifier.as_deref(),
            "graph_class": self.graph_class.as_str(),
            "cases": self.body.identification.as_ref().map(|r| vec![serde_json::json!({"identification": r, "weight": 1.0})]).unwrap_or_default(),
            "temporal_identification": self.body.temporal_identification,
            "reasoning": reasoning_section(&self.reasoning),
        }))
    }
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

/// One canonical derivation shared by native inspection records and artifacts.
struct ExecutedRecordParts {
    contract: CausalContract,
    compiled: Arc<CompiledContract>,
    claim: ClaimEnvelope,
    body: AnalysisResultWire,
    reuse: ReuseSection,
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
        Ok(self.compiled_contract()?.contract.clone())
    }

    fn compiled_contract(&self) -> Result<Arc<CompiledContract>, CausalError> {
        if let Some(cached) = self.contract_cache().get() {
            return Ok(Arc::clone(cached));
        }
        let (contract, payloads) =
            compile_with_payloads(self.study(), Some(self), Some(self.program_payloads()?))?;
        let compiled = Arc::new(CompiledContract::new(contract, payloads));
        // Keep compiled metadata, not potentially row-sized fitted laws or
        // participant/network exports. Those retain the ordinary on-demand
        // compilation path.
        let snapshot = &compiled.payloads.data_snapshot;
        if snapshot.distribution_factor_laws.is_some()
            || snapshot.nested_counterfactual_fit.is_some()
            || snapshot.interference.is_some()
            || snapshot.participants.is_some()
        {
            return Ok(compiled);
        }
        Ok(Arc::clone(self.contract_cache().get_or_init(|| compiled)))
    }

    /// Compiled contract of an execution under `refute` and `population`.
    ///
    /// The handle's own contract when both match the bound study. A retarget
    /// or different validation suite is compiled once and memoised beside the
    /// base contract, so it shares the base contract's lifetime and is dropped
    /// with it whenever the handle's state changes.
    fn compiled_contract_for(
        &self,
        refute: crate::RefuteSuite,
        population: Option<&TargetPopulation>,
    ) -> Result<Arc<CompiledContract>, CausalError> {
        if refute == self.study().refute && population.is_none() {
            return self.compiled_contract();
        }
        let key = (refute, population.cloned());
        let base = self.contract_cache().get();
        if let Some(found) = base.and_then(|base| base.variants.get(&key)) {
            return Ok(found);
        }
        // Only a miss pays for the retargeted study and its compilation.
        let study =
            self.study_for_execution(refute, population).ok_or_else(|| CausalError::Compile {
                message: "retargeted study requested for the handle's own contract".into(),
            })?;
        let (contract, payloads) = compile_with_payloads(&study, Some(self), None)?;
        let compiled = Arc::new(CompiledContract::new(contract, payloads));
        Ok(match base {
            Some(base) => base.variants.insert(key, compiled),
            None => compiled,
        })
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
            self.program_binding(),
            self.posterior_programs(&self.study().query)?,
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
                if population.is_none() {
                    self.program_binding()
                } else {
                    CheckedProgramBinding::None
                },
                self.posterior_programs(&study.query)?,
            )?),
        };
        let snapshot = data_snapshot_wire(
            data,
            self.study().interference.as_ref(),
            self.study().participants.as_ref(),
            missing_data::policy_wire(self.study()),
            self.study().sampling_design,
            &program.observation_digest,
        )?;
        let mut snapshot = snapshot;
        if let DataInput::Tabular(tabular) = data {
            let laws = self
                .distribution_factor_snapshot(tabular)?
                .or(self.functional_effect_factor_snapshot(tabular)?);
            let laws = match laws {
                Some(laws) => Some(laws),
                None => self
                    .functional_effect_response_factor_snapshots(tabular)?
                    .map(|snapshots| merge_response_factor_snapshots(&snapshots))
                    .transpose()?,
            };
            if let Some(laws) = laws {
                snapshot.distribution_factor_laws = Some(
                    antecedent_io::distribution_factor_laws_to_wire(&laws)
                        .map_err(|err| io_err(&err))?,
                );
            }
            if let Some(operation) = self.execution().nested_counterfactual() {
                snapshot.nested_counterfactual_fit =
                    Some(nested_counterfactual_fit_wire(tabular, operation)?);
            }
            if let Some(operation) = self.execution().linear_operation() {
                snapshot.linear_fit_moments = linear_fit_moments_wire(tabular, operation)?;
            }
        }
        if let DataInput::Temporal(series) | DataInput::Event(series) = data {
            if let CausalQuery::NestedCounterfactual(query) = &self.study().query {
                snapshot.nested_counterfactual_fit = Some(temporal_nested_fit_wire(
                    series,
                    query,
                    nested_prior_scale(&self.study().inference),
                )?);
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
        self.execution_contract_with_quantity_labels(result, ctx, None)
    }

    /// Inspect the executed claim under explicit response measurement labels.
    ///
    /// Uses the same canonical claim derivation as export, without encoding an
    /// artifact or materializing its payload in a host language.
    ///
    /// # Errors
    /// Contract mismatches, unsupported response coordinates or canonical encoding failures.
    pub fn execution_contract_with_quantity_labels(
        &self,
        result: &StudyResult,
        ctx: &ExecutionContext,
        labels: Option<&antecedent_contract::ResponseCoordinateLabels<'_>>,
    ) -> Result<(CausalContract, ClaimEnvelope), CausalError> {
        let parts = self.execution_record_parts(result, ctx, labels)?;
        Ok((parts.contract, parts.claim))
    }

    /// Derive the executed native contract section and scientific body without
    /// encoding an artifact. Execution adapters can retain presentation metadata
    /// now so later inspection need not serialize or revisit native execution.
    ///
    /// # Errors
    /// The same identity, coordinate and claim checks as contracted export.
    pub fn execution_record_with_quantity_labels(
        &self,
        result: &StudyResult,
        ctx: &ExecutionContext,
        labels: Option<&antecedent_contract::ResponseCoordinateLabels<'_>>,
    ) -> Result<(AnalysisResultContractWire, AnalysisResultWire), CausalError> {
        let parts = self.execution_record_parts(result, ctx, labels)?;
        let execution = execution_identity_from_context(ctx);
        let section = parts.contract.section_from_payloads(
            &parts.compiled.payloads,
            Some(&parts.claim),
            Some(&execution),
            &parts.reuse,
        )?;
        Ok((section, parts.body))
    }

    fn execution_record_parts(
        &self,
        result: &StudyResult,
        ctx: &ExecutionContext,
        labels: Option<&antecedent_contract::ResponseCoordinateLabels<'_>>,
    ) -> Result<ExecutedRecordParts, CausalError> {
        let quantities = labels
            .map(|labels| {
                let response = result.response().ok_or_else(|| CausalError::Compile {
                    message: "scientific coordinates require a native response".into(),
                })?;
                antecedent_contract::response_coordinates(
                    response,
                    &|id| {
                        self.schema()
                            .variables()
                            .get(id.as_usize())
                            .map(|variable| variable.name.to_string())
                    },
                    labels,
                )
                .map_err(|refusal| CausalError::Compile {
                    message: format!("{}: {}", refusal.code, refusal.detail),
                })
            })
            .transpose()?;
        let (mut contract, compiled) = self.contract_and_payloads_for_result(result)?;
        let reuse = self.bind_reuse(&mut contract, result)?;
        let (claim, body) = result.claim_with_body(&contract, ctx, quantities.as_deref())?;
        Ok(ExecutedRecordParts { contract, compiled, claim, body, reuse })
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
    ) -> Result<(CausalContract, Arc<CompiledContract>), CausalError> {
        let executed = result.executed_contract.as_ref().ok_or(CausalError::Conflict {
            what: "result",
            detail: "result was not executed by a prepared handle",
        })?;
        let retargeted = result.retarget_population();
        let compiled = self.compiled_contract_for(executed.refute, retargeted.as_ref())?;
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
        let ours = compiled.contract.identities;
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
        let mut contract = compiled.contract.clone();
        bind_graphless_reasoning(&mut contract, result, self.query())?;
        // Interference is on the geometric support axis: its contract keeps the
        // geometric `InterferenceQuery` coordinate and calibration basis. The
        // off-axis exposure-contrast interval license is carried instead in
        // `interference_inference.graphless_support_status`, wired to the
        // artifact and revalidated by `validate_result`, so it is never a
        // separate matrix coordinate that would break calibration verification.
        //
        // `local_polynomial_ratio` is deliberately treated the same way and has
        // no arm here. Its execute license surfaces on the per-fit
        // `LocalPolynomialRatioWire.graphless_support_status` field, which
        // `validate_result` independently rechecks against the published
        // interval, mirroring the interference precedent. It carries no
        // `graphless:local_polynomial_ratio/...` matrix coordinate (no route or
        // Python assertion expects one), so it stays off the matrix axis.
        Ok((contract, compiled))
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
        let row_ids: Vec<u64> = table.row_index.iter().map(|id| id.get()).collect();
        Some(ScoreReuseIdentityWire::score_table(&antecedent_io::ScoreTableKey {
            identification: identities.identification,
            data_snapshot: identities.data_snapshot,
            row_index: &row_ids,
            fold_ids: &table.fold_ids,
            n_folds: table.n_folds,
            adjustment_set: &table.adjustment_set,
            nuisance_provenance: table.nuisance_provenance.as_ref(),
            treatment: table.treatment,
            intervened: &table.intervened,
            bootstrap_replicates: self.study().bootstrap_replicates,
            inference_binding: identities.inference_binding,
        }))
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
        if let Some(checked) = self.execution().checked_aipw() {
            // The cross-fitted procedure publishes its complete-case rows on the
            // score table; the trimmed common-support procedure has no table and
            // binds the rows its preparation retained.
            let rows = match (
                result.estimate.as_effect().and_then(|effect| effect.scores.table.as_ref()),
                checked.lowering().procedure,
            ) {
                (Some(table), _) => table.row_index.to_vec(),
                (None, antecedent_estimate::CheckedAipwProcedure::TrimmedLogisticOls) => {
                    checked.lowering().rows.to_vec()
                }
                (None, antecedent_estimate::CheckedAipwProcedure::CrossFittedLogisticOls) => {
                    return Err(CausalError::Compile {
                        message:
                            "checked AIPW execution did not retain its complete-case score rows"
                                .into(),
                    });
                }
            };
            let rows = antecedent_io::CheckedAipwRowsWire {
                format: 1,
                rows: rows.iter().map(|id| id.get()).collect(),
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
        let identity = self.row_weights_identity(weights, depends_on)?;
        let target_weights = target_weights_digest(&identity).map_err(|err| io_err(&err))?;
        Ok(RowWeightsBinding {
            weights: Arc::from(weights),
            depends_on: Arc::from(depends_on),
            identity,
            target_weights,
        })
    }

    fn row_weights_identity(
        &self,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
    ) -> Result<TargetWeightsIdentityWire, CausalError> {
        let identities = self.contract()?.identities;
        let score = self.score_reuse_wire(&identities).ok_or(CausalError::Unsupported {
            message: "row-weight target requires a prepared score table",
        })?;
        let score = score_reuse_digest(&score).map_err(|err| io_err(&err))?;
        Ok(TargetWeightsIdentityWire::new(weights, identities.data_snapshot, score, depends_on))
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
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "reexecute_retarget" });
        }
        // Both representations are live while the expected identity is compared:
        // native VariableIds and the raw-u32 dependencies in the identity payload.
        // The artifact's weights are borrowed rather than copied for this check.
        let _dependencies = ctx
            .acquire_admission_product(
                0,
                u64::try_from(section.identity.depends_on.len()).map_err(|_| {
                    CausalError::Resource { message: "retarget dependency count overflow".into() }
                })?,
                u64::try_from(size_of::<antecedent_core::VariableId>() + size_of::<u32>())
                    .map_err(|_| CausalError::Resource {
                        message: "retarget dependency width overflow".into(),
                    })?,
            )
            .map_err(|error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "reexecute_retarget" }
                }
                error => CausalError::Resource { message: error.to_string() },
            })?;
        let depends_on: Vec<antecedent_core::VariableId> = section
            .identity
            .depends_on
            .iter()
            .copied()
            .map(antecedent_core::VariableId::from_raw)
            .collect();
        let identity = self.row_weights_identity(&section.values, &depends_on)?;
        if identity != section.identity {
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
        self.encode_contracted_result_with_quantity_labels(result, artifact_id, ctx, None)
    }

    /// Encode an executed response with explicitly declared measurement semantics.
    /// Native response structure determines variable, regime, horizon and functional;
    /// the caller declares units, population and transform. Coordinates are included
    /// in the checked result digest before sealing its claim.
    ///
    /// # Errors
    /// The ordinary contracted-result errors, absent response or undescribed coordinates.
    pub fn encode_contracted_result_with_quantity_labels(
        &self,
        result: &StudyResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
        labels: Option<&antecedent_contract::ResponseCoordinateLabels<'_>>,
    ) -> Result<Vec<u8>, CausalError> {
        let (section, body) = self.execution_record_with_quantity_labels(result, ctx, labels)?;
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
        self.refresh_in_place(data, ctx)
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
    ) -> Result<crate::design::ScoredDesignRanking, CausalError>
    where
        A: Clone + Sync,
        O: Clone + Sync,
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
    ) -> Result<crate::design::ScoredDesignRanking, CausalError>
    where
        A: Clone + Sync,
        O: Clone + Sync,
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

fn io_err(err: &antecedent_io::IoError) -> CausalError {
    CausalError::rejected(err.clone())
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

/// Base prepared contract. Never stores an execution's claim or result reasoning.
#[derive(Debug)]
pub(crate) struct CompiledContract {
    contract: CausalContract,
    payloads: ContractPayloads,
    /// Contracts of executions under another validation suite or target
    /// population, compiled on first use.
    variants: VariantContracts,
}

impl CompiledContract {
    fn new(contract: CausalContract, payloads: ContractPayloads) -> Self {
        Self { contract, payloads, variants: VariantContracts::default() }
    }
}

type VariantKey = (crate::RefuteSuite, Option<TargetPopulation>);

/// Memo of retargeted compiled contracts, keyed by validation suite and population.
#[derive(Debug, Default)]
struct VariantContracts(
    std::sync::Mutex<std::collections::HashMap<VariantKey, Arc<CompiledContract>>>,
);

impl VariantContracts {
    fn get(&self, key: &VariantKey) -> Option<Arc<CompiledContract>> {
        let map = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        map.get(key).map(Arc::clone)
    }

    /// Store `compiled` unless a concurrent caller stored one first; return the retained one.
    fn insert(&self, key: VariantKey, compiled: Arc<CompiledContract>) -> Arc<CompiledContract> {
        let mut map = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(map.entry(key).or_insert(compiled))
    }
}

#[derive(Clone, Debug)]
struct ContractPayloads {
    program: Arc<ProgramPayloads>,
    data_snapshot: DataSnapshotIdentityWire,
    identities: ContractIdentities,
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
                antecedent_identify::IdentificationResultParts {
                    status,
                    query: CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
                        antecedent_core::VariableId::from_raw(0),
                        antecedent_core::VariableId::from_raw(1),
                    )),
                    estimands: Vec::new(),
                    arena: antecedent_expr::CausalExprArena::new(),
                    derivation: DerivationTrace::default(),
                    required_assumptions: AssumptionSet::default(),
                    diagnostics,
                    performance: IdentificationPerformanceRecord::default(),
                    hedge: None,
                },
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

    use super::program_payloads_for;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::{AnalyticSeKind, FrontDoorTwoStage};
    use antecedent_graph::Dag;

    use crate::{
        CausalError, Study, analysis::builder::RefuteSuite,
        analysis::prepared::CheckedProgramBinding, strategy_table::IdentifierId,
    };

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
        let wrong_binding = program_payloads_for(
            prepared.study(),
            Some(crate::EstimatorId::Aipw.as_str()),
            CheckedProgramBinding::FrontDoor(
                &prepared.execution().frontdoor_linear().unwrap().preparation,
            ),
            None,
        );
        assert!(matches!(wrong_binding, Err(CausalError::Compile { .. })));
        let result = prepared.estimate(&data, &context).unwrap();
        let (record_section, record_body) =
            prepared.execution_record_with_quantity_labels(&result, &context, None).unwrap();
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
        assert_eq!(record_section, contract);
        assert_eq!(record_body, body);
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
        for (estimator, expected) in [(0_u64, "wald"), (1, "two_stage_least_squares")] {
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
            let context = ExecutionContext::for_tests(903 + estimator);
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
