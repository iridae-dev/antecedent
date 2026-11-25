//! Compiled decisions, rankings and plans over declared inputs.
//!
//! Each entry compiles a computation a caller states over evidence it supplies into a program
//! of its own family, and executes the same kernel the direct entry calls, on the same inputs:
//!
//! | entry | family | kernel |
//! |---|---|---|
//! | [`compile_supported_input_decision`] | `SupportedInputDecision` | `composition_boundary::evaluate_with_support` |
//! | [`compile_input_functional`] | `SupportedInputDecision` | `composition_boundary::evaluate_functional_on_input` |
//! | [`compile_sensitivity_decision`] | `SensitivityDecision` | `sensitivity_decision::evaluate_sensitivity_decision` |
//! | [`compile_inverse_query`] | `InverseQueryDecision` | `inverse_query::evaluate_inverse_query` |
//! | [`compile_finite_enumeration_baseline`] | `InverseQueryDecision` | `inverse_query::finite_enumeration_baseline` |
//! | [`compile_decision_draws_ranking`] | `DecisionDrawsRanking` | `decision_reliability::rank` |
//! | [`compile_producer_carrier_ranking`] | `DecisionDrawsRanking` | `decision_reliability::ProducerCarrier::rank` |
//! | [`compile_prior_signal_adaptation`] | `PriorSignalAdaptation` | `prior_signal::adapt_prior_to_signal` |
//! | [`compile_evidence_obligations`] | `EvidenceObligations` | `obligation_adapters::identification_failure_obligations` |
//! | [`compile_finite_action_decision`] | `FiniteActionDecision` | `decision::evaluate_decision` |
//! | [`compile_adaptive_policy_search`] | `AdaptivePolicySearch` | `PreparedAdaptivePolicySearch::execute` |
//!
//! A refusal of the kernel is returned unchanged as [`TerminalError::Kernel`]; the compiler's
//! own refusals (re-entry, a release build's support gate) are
//! [`TerminalError::Refused`].
//!
//! A declared input that is a live callback (a utility or constraint of a finite decision
//! problem, an external signal provider, the response source of a policy search) is a runtime
//! binding: the program identity records that one is bound, not what it computes.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_contract::evidence_obligation::EvidenceObligation;
use antecedent_contract::external::ExternalCapability;
use antecedent_core::{
    CanonicalEncode, CanonicalWriter, ExecutionContext, SemanticDigest, SupportStatus, VariableId,
};
use antecedent_design::DecisionFunctional;
use antecedent_design::adaptive_policy_search::{
    AdaptivePolicySearchError, AdaptivePolicySearchResult, PolicyRolloutSource,
    PreparedAdaptivePolicySearch,
};
use antecedent_design::composition_boundary::{
    BoundaryError, CalibrationStatus, DecisionInput, InputSource, ProviderKind, ReceiptRef,
    ScalarClaim, SupportPolicy, SupportedDecision, UnsupportedActionPolicy,
    evaluate_functional_on_input, evaluate_with_support,
};
use antecedent_design::decision_contract::DecisionContract;
use antecedent_design::decision_eval::{FunctionalValue, MeanEstimate, MeanSource};
use antecedent_design::decision_reliability::{
    DecisionDraws, JointLawStanding, ProducerCarrier, RankingReport, RankingRequest,
    ReliabilityError, rank,
};
use antecedent_design::decision_structural::StructuralAtom;
use antecedent_design::gaussian_forward_law::GaussianForwardLaw;
use antecedent_design::inverse_query::{
    Comparison, EnumeratedForwardValue, FeasibilityStatus, ForwardClaim, ForwardEvidence,
    GridScope, IdentifiedSet as MemberSet, IntervalRegion, InverseConstraint, InverseQuery,
    InverseQueryError, InverseResult, SelectionRule, evaluate_inverse_query,
    finite_enumeration_baseline,
};
use antecedent_design::obligation_adapters::identification_failure_obligations;
use antecedent_design::prior_signal::{
    CandidateSignalPrior, CandidateSignalSource, NativeSignalFamily, PriorSignalError,
    PriorSignalRequest, PriorSourceInput, SourceToTarget, adapt_prior_to_signal,
};
use antecedent_design::sensitivity_decision::{
    SensitivityArtifact, SensitivityDecisionError, SensitivityDecisionResult,
    SensitivityDecisionSpec, evaluate_sensitivity_decision,
};
use antecedent_design::{
    DecisionEvaluation, DecisionProblem, DesignError, RepairError, SensitivityFinding,
    evaluate_decision,
};
use antecedent_graph::Admg;
use antecedent_identify::IdentificationResult;
use antecedent_io_core::prior_bank::{PriorArtifactBody, PriorSourceRef};
use antecedent_io_core::{PriorCatalog, TargetDesign};

pub use crate::analysis::supplied_terminal::TerminalError;

use crate::analysis::program_handle::CompiledProgramHandle;
use crate::analysis::supplied_identity::{Canonical, Declaration, SetOf, Wire, encoded, variant};
use crate::analysis::supplied_terminal::{
    Reading, TerminalSpec, admit_compile, compiled_terminal, lower,
};
use crate::error::CausalError;
use crate::planner::lowered::{
    CarrierDemand, Composition, DecisionProgram, EvidenceRole, FunctionalKind, LoweredProgram,
    ProgramCell,
};

/// The decision layer of a contract's program: its actions, constraints, strictest chance
/// threshold, utility unit and identity (absent when the contract does not validate; the
/// kernel then refuses it at execution with its own error).
fn contract_program(
    contract: &DecisionContract,
    carrier: CarrierDemand,
    source: Option<&str>,
) -> DecisionProgram {
    DecisionProgram {
        carrier,
        utility: DecisionProgram::utility_form(None),
        actions: contract.actions.len(),
        constraints: contract.constraints.len(),
        chance_threshold: contract
            .constraints
            .iter()
            .map(|constraint| constraint.min_probability)
            .fold(0.0_f64, f64::max),
        utility_unit: Some(Arc::from(contract.utility_units.as_str())),
        contract_identity: contract.identity().ok().map(Arc::from),
        information_value: None,
        scenario_source: source.map(Arc::from),
    }
}

// --- canonical encoding of the declared inputs ----------------------------------------------

/// A declared input of this module's families, encoded canonically (see
/// [`crate::analysis::supplied_identity`]). A local wrapper, so these encodings never collide
/// with another module's encoding of the same kernel type.
struct In<'a, T: ?Sized>(&'a T);

impl Canonical for In<'_, MeanEstimate> {
    fn encode(&self, out: &mut CanonicalWriter) {
        Wire(self.0).encode(out);
    }
}

impl Canonical for In<'_, MeanSource> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let source = self.0;
        // Ordered: means align with coordinates by position.
        source.coordinates.encode(out);
        source.means.encode(out);
        In(&source.estimate).encode(out);
        out.text(&source.provider_id)
            .text(&source.snapshot_id)
            .text(&source.causal_contract_id)
            .text(&source.rng_id);
    }
}

impl Canonical for In<'_, DecisionFunctional> {
    fn encode(&self, out: &mut CanonicalWriter) {
        // Serde wire of scalar fields and the `Tail` unit enum; no maps.
        Wire(self.0).encode(out);
    }
}

impl Canonical for In<'_, ScalarClaim> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let claim = self.0;
        claim.coordinate.encode(out);
        out.real(claim.value)
            .text(&claim.provider_id)
            .text(&claim.snapshot_id)
            .text(&claim.causal_contract_id);
    }
}

impl Canonical for In<'_, ProviderKind> {
    fn encode(&self, out: &mut CanonicalWriter) {
        variant(
            out,
            match self.0 {
                ProviderKind::Native => "native",
                ProviderKind::ExternalAttested => "external_attested",
                ProviderKind::ExternalExactRequestVerified => "external_exact_request_verified",
            },
        );
    }
}

impl Canonical for In<'_, CalibrationStatus> {
    fn encode(&self, out: &mut CanonicalWriter) {
        variant(
            out,
            match self.0 {
                CalibrationStatus::Exact => "exact",
                CalibrationStatus::PointOnly => "point_only",
                CalibrationStatus::Measured => "measured",
                CalibrationStatus::NotMeasured => "not_measured",
                CalibrationStatus::ExactClaimUnverified => "exact_claim_unverified",
            },
        );
    }
}

impl Canonical for In<'_, ReceiptRef> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0 {
            ReceiptRef::NativeExecution { execution_id } => {
                variant(out, "native_execution").text(execution_id);
            }
            ReceiptRef::Attestation { attestor } => {
                variant(out, "attestation").text(attestor);
            }
            ReceiptRef::Verification { object_id, request_id } => {
                variant(out, "verification").text(object_id).text(request_id);
            }
        }
    }
}

impl Canonical for In<'_, ExternalCapability> {
    fn encode(&self, out: &mut CanonicalWriter) {
        variant(
            out,
            match self.0 {
                ExternalCapability::Sample => "sample",
                ExternalCapability::Cdf => "cdf",
                ExternalCapability::Quantile => "quantile",
                ExternalCapability::LogProbability => "log_probability",
                ExternalCapability::Mean => "mean",
                ExternalCapability::Covariance => "covariance",
                ExternalCapability::Conditional => "conditional",
                ExternalCapability::Intervention => "intervention",
                ExternalCapability::PosteriorPredictive => "posterior_predictive",
                ExternalCapability::Update => "update",
                ExternalCapability::Factor => "factor",
                ExternalCapability::EvaluateUtility => "evaluate_utility",
            },
        );
    }
}

impl Canonical for In<'_, DecisionInput> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let input = self.0;
        out.text(input.id());
        match input.source() {
            InputSource::Mean(means) => {
                variant(out, "mean");
                In(means).encode(out);
            }
            InputSource::JointLaw(law) => {
                variant(out, "joint_law");
                law.encode(out);
            }
            InputSource::Scalar(claim) => {
                variant(out, "scalar");
                In(claim).encode(out);
            }
        }
        let provenance = input.provenance();
        In(&provenance.provider_kind).encode(out);
        // Unit enum with a serde wire; no maps.
        Wire(&provenance.trust).encode(out);
        match &provenance.receipt {
            Some(receipt) => {
                out.word(1);
                In(receipt).encode(out);
            }
            None => {
                out.word(0);
            }
        }
        In(&provenance.calibration).encode(out);
        // A set: the boundary only asks whether a capability is declared, so order and
        // repetition are dropped.
        let capabilities: Vec<In<'_, ExternalCapability>> =
            provenance.capabilities.iter().map(In).collect();
        SetOf(&capabilities).encode(out);
        out.text(&provenance.provider_id)
            .text(&provenance.snapshot_id)
            .text(&provenance.lineage_digest);
        // Ordered: a coordinate's status is the first matching entry.
        input.support().entries().encode(out);
    }
}

impl Canonical for In<'_, SupportPolicy> {
    fn encode(&self, out: &mut CanonicalWriter) {
        variant(
            out,
            match self.0.unsupported {
                UnsupportedActionPolicy::RequireAllActions => "require_all_actions",
                UnsupportedActionPolicy::CompareSupported => "compare_supported",
            },
        );
        self.0.weakest_support.encode(out);
    }
}

impl Canonical for In<'_, Comparison> {
    fn encode(&self, out: &mut CanonicalWriter) {
        variant(out, self.0.as_str());
    }
}

impl Canonical for In<'_, InverseConstraint> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match *self.0 {
            InverseConstraint::TargetMean { target, comparison } => {
                variant(out, "target_mean").real(target);
                In(&comparison).encode(out);
            }
            InverseConstraint::TargetQuantile { p, target, comparison } => {
                variant(out, "target_quantile").real(p).real(target);
                In(&comparison).encode(out);
            }
            InverseConstraint::ProbabilityThreshold {
                outcome_threshold,
                tail,
                probability,
                comparison,
            } => {
                variant(out, "probability_threshold").real(outcome_threshold);
                Wire(&tail).encode(out);
                out.real(probability);
                In(&comparison).encode(out);
            }
        }
    }
}

impl Canonical for In<'_, InverseQuery> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let query = self.0;
        query.contract.encode(out);
        // Ordered: the declared grid is the tie-rule order.
        query.grid_order.encode(out);
        variant(
            out,
            match query.grid_scope {
                GridScope::FiniteEnumeration => "finite_enumeration",
                GridScope::ContinuousSample => "continuous_sample",
            },
        );
        // Ordered: values are reported per constraint in declaration order.
        let constraints: Vec<In<'_, InverseConstraint>> =
            query.constraints.iter().map(In).collect();
        constraints.encode(out);
        variant(
            out,
            match query.selection {
                SelectionRule::FirstInGridOrder => "first_in_grid_order",
                SelectionRule::LastInGridOrder => "last_in_grid_order",
                SelectionRule::RequireUnique => "require_unique",
            },
        );
        out.real(query.tolerance);
        query.max_evaluations.encode(out);
    }
}

impl Canonical for In<'_, ForwardClaim> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0 {
            ForwardClaim::Law(law) => {
                variant(out, "law");
                law.encode(out);
            }
            ForwardClaim::Gaussian(law) => {
                variant(out, "gaussian");
                // Serde wire of scalar fields and a quantity; no maps.
                Wire::<GaussianForwardLaw>(law).encode(out);
            }
            ForwardClaim::Means(means) => {
                variant(out, "means");
                In(means).encode(out);
            }
        }
    }
}

impl Canonical for In<'_, IntervalRegion> {
    fn encode(&self, out: &mut CanonicalWriter) {
        In(&self.0.lower).encode(out);
        In(&self.0.upper).encode(out);
        self.0.endpoints_bound_functional.encode(out);
    }
}

impl Canonical for In<'_, StructuralAtom> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.0.id);
        self.0.probability.encode(out);
        self.0.evidence.encode(out);
    }
}

fn atoms(items: &[StructuralAtom]) -> Vec<In<'_, StructuralAtom>> {
    items.iter().map(In).collect()
}

impl Canonical for In<'_, MemberSet> {
    fn encode(&self, out: &mut CanonicalWriter) {
        // Ordered: members are reported in declaration order.
        atoms(&self.0.members).encode(out);
        self.0.exhaustive.encode(out);
    }
}

impl Canonical for In<'_, ForwardEvidence> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let evidence = self.0;
        evidence.point.as_ref().map(In).encode(out);
        evidence.interval_region.as_ref().map(In).encode(out);
        evidence.identified_set.as_ref().map(In).encode(out);
        evidence.scenarios.as_deref().map(atoms).encode(out);
    }
}

impl Canonical for In<'_, EnumeratedForwardValue> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.0.id);
        self.0.value.encode(out);
        self.0.supported.encode(out);
    }
}

impl Canonical for In<'_, SourceToTarget> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match *self.0 {
            SourceToTarget::Identity => {
                variant(out, "identity");
            }
            SourceToTarget::Affine { intercept, slope } => {
                variant(out, "affine").real(intercept).real(slope);
            }
        }
    }
}

impl Canonical for In<'_, PriorSourceInput> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let source = self.0;
        out.text(&source.artifact_id).text(&source.source_population);
        source.draws.encode(out);
        out.real(source.weight).real(source.prior_strength).real(source.conflict_shrinkage);
        // Sets: lineage is unioned and observation ids are checked for overlap as sets, so
        // order and repetition are dropped.
        SetOf(&source.lineage).encode(out);
        SetOf(&source.observation_ids).encode(out);
        In(&source.source_to_target).encode(out);
    }
}

impl Canonical for In<'_, PriorSourceRef> {
    fn encode(&self, out: &mut CanonicalWriter) {
        // Serde wire of strings, ordered lists and BTreeMaps (sorted keys); no hash maps.
        Wire(&self.0.meta).encode(out);
        match &self.0.artifact {
            None => {
                variant(out, "no_artifact");
            }
            Some(PriorArtifactBody::Bytes(bytes)) => {
                variant(out, "bytes").bytes(bytes);
            }
            Some(PriorArtifactBody::Path(path)) => {
                // A path names its bytes; the program reads them when it executes.
                variant(out, "path").text(&path.to_string_lossy());
            }
        }
    }
}

impl Canonical for In<'_, TargetDesign> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let target = self.0;
        Wire(&target.estimand).encode(out);
        // A BTreeSet: already sorted and distinct.
        let variables: Vec<&str> = target.variables.iter().map(String::as_str).collect();
        out.texts(&variables);
        // A BTreeMap: sorted by key.
        out.word(target.tags.len() as u64);
        for (key, value) in &target.tags {
            out.text(key).text(value);
        }
        target.allow_unidentified.encode(out);
    }
}

// --- decisions over several supported inputs ------------------------------------------------

compiled_terminal! {
    /// A compiled decision over several inputs, with support decided per action.
    #[derive(Clone, Debug)]
    CompiledSupportedInputDecision {
        contract: DecisionContract,
        inputs: Vec<DecisionInput>,
        policy: SupportPolicy,
    }
}

/// Compile the decision of `contract` over `inputs` under the support `policy`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_supported_input_decision(
    contract: DecisionContract,
    inputs: Vec<DecisionInput>,
    policy: SupportPolicy,
    ctx: &ExecutionContext,
) -> Result<CompiledSupportedInputDecision, CausalError> {
    admit_compile(ctx)?;
    // Ordered: values are reported per input, and actions are assigned to inputs in order.
    let ordered: Vec<In<'_, DecisionInput>> = inputs.iter().map(In).collect();
    let question = Declaration::new("antecedent.supported_input_decision.v2")
        .field("contract", &contract)
        .field("inputs", &ordered)
        .field("policy", &In(&policy))
        .finish();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::SupportedInputDecision,
        plan_id: "supplied.supported_input_decision.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_inputs",
        question,
        evaluator: "composition_boundary::evaluate_with_support",
        reading: Reading::Predictive,
        role: EvidenceRole::ChainSource,
        composition: Composition::Pooled { sources: inputs.len() },
        decision: Some(contract_program(&contract, CarrierDemand::PredictiveLaw, None)),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledSupportedInputDecision { lowered, contract, inputs, policy })
}

impl CompiledSupportedInputDecision {
    /// Decide over the inputs.
    ///
    /// # Errors
    /// The composition boundary's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SupportedDecision, TerminalError<BoundaryError>> {
        let _scoped = self.enter(ctx)?;
        evaluate_with_support(&self.contract, &self.inputs, &self.policy)
            .map_err(TerminalError::Kernel)
    }
}

compiled_terminal! {
    /// A compiled functional of one action's utility from one supported input.
    #[derive(Clone, Debug)]
    CompiledInputFunctional {
        contract: DecisionContract,
        action_id: String,
        functional: DecisionFunctional,
        input: DecisionInput,
        weakest: SupportStatus,
    }
}

/// Compile `functional` of `action_id`'s utility from `input`, refusing a coordinate below
/// `weakest`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_input_functional(
    contract: DecisionContract,
    action_id: &str,
    functional: DecisionFunctional,
    input: DecisionInput,
    weakest: SupportStatus,
    ctx: &ExecutionContext,
) -> Result<CompiledInputFunctional, CausalError> {
    admit_compile(ctx)?;
    let question = Declaration::new("antecedent.input_functional.v2")
        .field("contract", &contract)
        .field("action_id", action_id)
        .field("functional", &In(&functional))
        .field("input", &In(&input))
        .field("weakest", &weakest)
        .finish();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::SupportedInputDecision,
        plan_id: "supplied.input_functional.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_inputs",
        question,
        evaluator: "composition_boundary::evaluate_functional_on_input",
        reading: Reading::Predictive,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(contract_program(&contract, CarrierDemand::PredictiveLaw, None)),
        claim: Some(ClaimKind::Point),
    })?;
    Ok(CompiledInputFunctional {
        lowered,
        contract,
        action_id: action_id.to_owned(),
        functional,
        input,
        weakest,
    })
}

impl CompiledInputFunctional {
    /// Compute the functional.
    ///
    /// # Errors
    /// The composition boundary's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<FunctionalValue, TerminalError<BoundaryError>> {
        let _scoped = self.enter(ctx)?;
        evaluate_functional_on_input(
            &self.contract,
            &self.action_id,
            self.functional,
            &self.input,
            self.weakest,
        )
        .map_err(TerminalError::Kernel)
    }
}

// --- decisions over an assumption-sensitivity surface --------------------------------------

compiled_terminal! {
    /// A compiled decision over the surface of an assumption-sensitivity artifact.
    #[derive(Clone, Debug)]
    CompiledSensitivityDecision {
        contract: DecisionContract,
        artifact: SensitivityArtifact,
        spec: SensitivityDecisionSpec,
    }
}

/// Compile the decision of `contract` over the surface of `artifact`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_sensitivity_decision(
    contract: DecisionContract,
    artifact: SensitivityArtifact,
    spec: SensitivityDecisionSpec,
    ctx: &ExecutionContext,
) -> Result<CompiledSensitivityDecision, CausalError> {
    admit_compile(ctx)?;
    // The artifact by its own canonical identity (premises and numbers).
    let question = Declaration::new("antecedent.sensitivity_decision.v2")
        .field("contract", &contract)
        .field("artifact", &artifact.identity().digest)
        .field("range", &spec.range)
        .field("weights", &spec.weights)
        .field("sampling_composition", &spec.sampling_composition)
        .finish();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::SensitivityDecision,
        plan_id: "supplied.sensitivity_decision.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_assumption_surface",
        question,
        evaluator: "sensitivity_decision::evaluate_sensitivity_decision",
        reading: Reading::Envelope,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(contract_program(
            &contract,
            CarrierDemand::IdentifiedSet,
            Some("assumption_surface"),
        )),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledSensitivityDecision { lowered, contract, artifact, spec })
}

impl CompiledSensitivityDecision {
    /// Decide over the surface.
    ///
    /// # Errors
    /// The sensitivity decision's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SensitivityDecisionResult, TerminalError<SensitivityDecisionError>> {
        let _scoped = self.enter(ctx)?;
        evaluate_sensitivity_decision(&self.contract, &self.artifact, &self.spec)
            .map_err(TerminalError::Kernel)
    }
}

// --- inverse decision queries ---------------------------------------------------------------

compiled_terminal! {
    /// A compiled inverse decision query over forward evidence.
    #[derive(Clone, Debug)]
    CompiledInverseQuery {
        query: InverseQuery,
        evidence: ForwardEvidence,
    }
}

/// Compile the inverse `query` over `evidence`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_inverse_query(
    query: InverseQuery,
    evidence: ForwardEvidence,
    ctx: &ExecutionContext,
) -> Result<CompiledInverseQuery, CausalError> {
    admit_compile(ctx)?;
    let question = Declaration::new("antecedent.inverse_query.v2")
        .field("query", &In(&query))
        .field("evidence", &In(&evidence))
        .finish();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::InverseQueryDecision,
        plan_id: "supplied.inverse_query.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_forward_evidence",
        question,
        evaluator: "inverse_query::evaluate_inverse_query",
        reading: Reading::Predictive,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(contract_program(&query.contract, CarrierDemand::PredictiveLaw, None)),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledInverseQuery { lowered, query, evidence })
}

impl CompiledInverseQuery {
    /// Answer the inverse query.
    ///
    /// # Errors
    /// The inverse query's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<InverseResult, TerminalError<InverseQueryError>> {
        let _scoped = self.enter(ctx)?;
        evaluate_inverse_query(&self.query, &self.evidence).map_err(TerminalError::Kernel)
    }
}

compiled_terminal! {
    /// The compiled finite-enumeration baseline of an inverse mean query.
    #[derive(Clone, Debug)]
    CompiledFiniteEnumerationBaseline {
        points: Vec<EnumeratedForwardValue>,
        target: f64,
        comparison: Comparison,
        tolerance: f64,
    }
}

/// Compile the classification of each enumerated action by its forward mean against `target`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_finite_enumeration_baseline(
    points: Vec<EnumeratedForwardValue>,
    target: f64,
    comparison: Comparison,
    tolerance: f64,
    ctx: &ExecutionContext,
) -> Result<CompiledFiniteEnumerationBaseline, CausalError> {
    admit_compile(ctx)?;
    // Ordered: each point is classified and reported in declaration order.
    let enumerated: Vec<In<'_, EnumeratedForwardValue>> = points.iter().map(In).collect();
    let question = Declaration::new("antecedent.finite_enumeration_baseline.v2")
        .field("points", &enumerated)
        .field("target", &target)
        .field("comparison", &In(&comparison))
        .field("tolerance", &tolerance)
        .finish();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::InverseQueryDecision,
        plan_id: "supplied.finite_enumeration_baseline.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_forward_means",
        question,
        evaluator: "inverse_query::finite_enumeration_baseline",
        reading: Reading::Exact,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: None,
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledFiniteEnumerationBaseline { lowered, points, target, comparison, tolerance })
}

impl CompiledFiniteEnumerationBaseline {
    /// Classify each enumerated action.
    ///
    /// # Errors
    /// The baseline's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, FeasibilityStatus)>, TerminalError<InverseQueryError>> {
        let _scoped = self.enter(ctx)?;
        finite_enumeration_baseline(&self.points, self.target, self.comparison, self.tolerance)
            .map_err(TerminalError::Kernel)
    }
}

// --- ranking statements over decision draws ------------------------------------------------

compiled_terminal! {
    /// Compiled ranking statements over aligned decision draws.
    #[derive(Clone, Debug)]
    CompiledDecisionDrawsRanking {
        draws: DecisionDraws,
        request: RankingRequest,
    }
}

/// Compile the ranking statements of `draws` under `request`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_decision_draws_ranking(
    draws: DecisionDraws,
    request: RankingRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledDecisionDrawsRanking, CausalError> {
    admit_compile(ctx)?;
    // The draws by the carrier's own canonical digest, which writes the candidates in their
    // canonical (identity) order: the ranking reads the candidates as a set.
    let question = Declaration::new("antecedent.decision_draws_ranking.v2")
        .field("draws", &draws.digest_hex())
        // Serde wire: an optional baseline, the ordered top-k list and the batch plan; no maps.
        .field("request", &Wire(&request))
        .finish();
    let decision = DecisionProgram {
        carrier: CarrierDemand::DecisionDraws,
        utility: DecisionProgram::utility_form(None),
        actions: draws.candidates.len(),
        constraints: 0,
        chance_threshold: 0.0,
        utility_unit: Some(Arc::from(draws.utility_unit.as_str())),
        contract_identity: None,
        information_value: None,
        scenario_source: Some(Arc::from("decision_draws")),
    };
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DecisionDrawsRanking,
        plan_id: "supplied.decision_draws_ranking.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_draws",
        question,
        evaluator: "decision_reliability::rank",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledDecisionDrawsRanking { lowered, draws, request })
}

impl CompiledDecisionDrawsRanking {
    /// State the ranking.
    ///
    /// # Errors
    /// The ranking's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<RankingReport, TerminalError<ReliabilityError>> {
        let scoped = self.enter(ctx)?;
        rank(&self.draws, &self.request, &scoped).map_err(TerminalError::Kernel)
    }
}

/// Joint-law standing of a producer carrier. A marginal-only carrier is a different program
/// from one that ranks under one shared law, even when the draws digest matches.
struct CarrierJointLaw(JointLawStanding);

impl Canonical for CarrierJointLaw {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0 {
            JointLawStanding::OneSharedLaw => {
                variant(out, "one_shared_law");
            }
            JointLawStanding::MarginalOnly { reason } => {
                variant(out, "marginal_only").text(reason);
            }
        }
    }
}

compiled_terminal! {
    /// A compiled ranking of a producer carrier, after its joint-law standing is checked.
    #[derive(Clone, Debug)]
    CompiledProducerCarrierRanking {
        carrier: ProducerCarrier,
        request: RankingRequest,
    }
}

/// Compile the ranking statements of `carrier` under `request`.
///
/// The declaration includes the carrier's joint-law standing, so a marginal-only carrier is a
/// different program from one that ranks under one shared law. The draws enter by
/// [`DecisionDraws::digest_hex`]: the ranking reads the candidates as a set. `request.top_k`
/// stays in declaration order.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_producer_carrier_ranking(
    carrier: ProducerCarrier,
    request: RankingRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledProducerCarrierRanking, CausalError> {
    admit_compile(ctx)?;
    let standing = carrier.joint_law();
    let draws = carrier.draws();
    let question = Declaration::new("antecedent.producer_carrier_ranking.v1")
        .field("draws", &draws.digest_hex())
        .field("request", &Wire(&request))
        .field("joint_law", &CarrierJointLaw(standing))
        .finish();
    let decision = DecisionProgram {
        carrier: CarrierDemand::DecisionDraws,
        utility: DecisionProgram::utility_form(None),
        actions: draws.candidates.len(),
        constraints: 0,
        chance_threshold: 0.0,
        utility_unit: Some(Arc::from(draws.utility_unit.as_str())),
        contract_identity: None,
        information_value: None,
        scenario_source: Some(Arc::from("decision_draws")),
    };
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DecisionDrawsRanking,
        plan_id: "supplied.producer_carrier_ranking.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_draws",
        question,
        evaluator: "decision_reliability::ProducerCarrier::rank",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledProducerCarrierRanking { lowered, carrier, request })
}

impl CompiledProducerCarrierRanking {
    /// Rank the carrier: its joint-law check, then the ranking kernel on its draws.
    ///
    /// # Errors
    /// A marginal-only carrier, the ranking's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<RankingReport, TerminalError<ReliabilityError>> {
        let scoped = self.enter(ctx)?;
        self.carrier.rank(&self.request, &scoped).map_err(TerminalError::Kernel)
    }
}

// --- a prior adapted to a future signal ----------------------------------------------------

compiled_terminal! {
    /// A compiled adaptation of catalogued posterior sources to a future candidate signal.
    #[derive(Clone)]
    CompiledPriorSignalAdaptation {
        catalog: PriorCatalog,
        target: TargetDesign,
        request: PriorSignalRequest,
    }
}

impl std::fmt::Debug for CompiledPriorSignalAdaptation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledPriorSignalAdaptation")
            .field("identity", &self.lowered.identity())
            .field("catalog", &self.catalog)
            .field("target", &self.target)
            .field("prior_id", &self.request.prior_id)
            .finish_non_exhaustive()
    }
}

/// The identity of a prior-signal declaration. A signal provider is a runtime binding: it
/// declares no identity of its own, so the declaration records that one is bound.
fn prior_signal_digest(
    catalog: &PriorCatalog,
    target: &TargetDesign,
    request: &PriorSignalRequest,
) -> [u8; 32] {
    // Ordered: sources are pooled in declaration order.
    let catalogued: Vec<In<'_, PriorSourceRef>> = catalog.sources.iter().map(In).collect();
    let sources: Vec<In<'_, PriorSourceInput>> = request.sources.iter().map(In).collect();
    let mut declaration = Declaration::new("antecedent.prior_signal_adaptation.v2");
    declaration
        .field("catalog", &catalogued)
        .field("target", &In(target))
        .field("prior_id", &request.prior_id)
        .field("sources", &sources)
        .field("target_population", &request.target_population)
        .field("transport", &request.transport.as_ref().map(|t| t.policy_id.as_str()))
        // A set: the candidate's observation ids are checked for overlap as a set, so order
        // and repetition are dropped.
        .field("candidate_observation_ids", &SetOf(&request.candidate_observation_ids))
        .field("resolution", &request.resolution);
    match &request.signal {
        CandidateSignalSource::Native(NativeSignalFamily::GaussianMean { noise_variance }) => {
            declaration
                .field("signal", "native_gaussian_mean")
                .field("noise_variance", noise_variance);
        }
        CandidateSignalSource::Native(NativeSignalFamily::Binomial) => {
            declaration.field("signal", "native_binomial");
        }
        CandidateSignalSource::Provider(_) => {
            declaration.field("signal", "provider").callback("signal_provider", None);
        }
        CandidateSignalSource::CoefficientPriorOnly => {
            declaration.field("signal", "coefficient_prior_only");
        }
        CandidateSignalSource::UnlockListOnly(unlocks) => {
            declaration.field("signal", "unlock_list_only").field("unlocks", unlocks);
        }
    }
    declaration.finish()
}

/// Compile the adaptation of `catalog`'s sources named in `request` to its future signal.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_prior_signal_adaptation(
    catalog: PriorCatalog,
    target: TargetDesign,
    request: PriorSignalRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledPriorSignalAdaptation, CausalError> {
    admit_compile(ctx)?;
    let question = prior_signal_digest(&catalog, &target, &request);
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::PriorSignalAdaptation,
        plan_id: "supplied.prior_signal_adaptation.v1",
        functional: FunctionalKind::PropagatedChain,
        graph_class: "catalogued_posterior_sources",
        question,
        evaluator: "prior_signal::adapt_prior_to_signal",
        reading: Reading::Posterior,
        role: EvidenceRole::PriorArtifact,
        composition: Composition::Pooled { sources: request.sources.len() },
        decision: None,
        claim: None,
    })?;
    Ok(CompiledPriorSignalAdaptation { lowered, catalog, target, request })
}

impl CompiledPriorSignalAdaptation {
    /// Adapt the sources to the signal.
    ///
    /// # Errors
    /// The adapter's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CandidateSignalPrior, TerminalError<PriorSignalError>> {
        let _scoped = self.enter(ctx)?;
        adapt_prior_to_signal(&self.catalog, &self.target, &self.request)
            .map_err(TerminalError::Kernel)
    }
}

// --- evidence a failed identification owes -------------------------------------------------

compiled_terminal! {
    /// The compiled evidence obligations of a failed identification.
    #[derive(Clone, Debug)]
    CompiledEvidenceObligations {
        result: IdentificationResult,
        graph: Admg,
        observed: Vec<VariableId>,
        population: String,
        sensitivity: Option<SensitivityFinding>,
    }
}

/// Compile the evidence obligations a failed identification `result` owes on `graph`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_evidence_obligations(
    result: IdentificationResult,
    graph: Admg,
    observed: Vec<VariableId>,
    population: &str,
    sensitivity: Option<SensitivityFinding>,
    ctx: &ExecutionContext,
) -> Result<CompiledEvidenceObligations, CausalError> {
    admit_compile(ctx)?;
    // The kernel reads the result's status and, for an average effect, its treatment and
    // outcome; the query is written by its own canonical encoding.
    let mut query = CanonicalWriter::new("antecedent.evidence_obligations.query.v1");
    result.query.encode(&mut query);
    // A graph without a canonical identity refuses the compile (`encoded`).
    let question = encoded("supplied.evidence_obligations.v1", || {
        Declaration::new("antecedent.evidence_obligations.v2")
            .field("status", result.status.as_str())
            .field("query", &SemanticDigest::from_bytes(query.finish()).to_hex())
            .field("graph", &graph)
            // A set: the observed variables are read by membership, so order and repetition
            // are dropped.
            .field("observed", &SetOf(&observed))
            .field("population", population)
            .field(
                "sensitivity",
                &sensitivity
                    .as_ref()
                    .map(|finding| (finding.method.as_str(), (finding.strength, finding.robust))),
            )
            .finish()
    })?;
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::EvidenceObligations,
        plan_id: "supplied.evidence_obligations.v1",
        functional: FunctionalKind::InterventionalDistribution,
        graph_class: "admg_identification_failure",
        question,
        evaluator: "obligation_adapters::identification_failure_obligations",
        reading: Reading::Exact,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: None,
        claim: None,
    })?;
    Ok(CompiledEvidenceObligations {
        lowered,
        result,
        graph,
        observed,
        population: population.to_owned(),
        sensitivity,
    })
}

impl CompiledEvidenceObligations {
    /// State the obligations.
    ///
    /// # Errors
    /// The adapter's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<EvidenceObligation>, TerminalError<RepairError>> {
        let _scoped = self.enter(ctx)?;
        identification_failure_obligations(
            &self.result,
            &self.graph,
            &self.observed,
            &self.population,
            self.sensitivity.as_ref(),
        )
        .map_err(TerminalError::Kernel)
    }
}

// --- a finite decision problem under draws of the state ------------------------------------

/// A compiled finite decision problem under equally weighted draws of the decision state.
///
/// The utility and the constraints are live callbacks: the program identity records the
/// problem's shape (actions, draws, constraints, chance threshold) and that callbacks are
/// bound, not what they compute.
pub struct CompiledFiniteActionDecision<A, O> {
    lowered: Arc<LoweredProgram>,
    problem: DecisionProblem<A, O>,
    outcomes: Vec<O>,
}

impl<A, O> CompiledProgramHandle for CompiledFiniteActionDecision<A, O> {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl<A, O> std::fmt::Debug for CompiledFiniteActionDecision<A, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledFiniteActionDecision")
            .field("identity", &self.lowered.identity())
            .field("actions", &self.problem.actions.len())
            .field("outcomes", &self.outcomes.len())
            .finish_non_exhaustive()
    }
}

/// Compile the Bayes action and expected regret of `problem` under the equally weighted
/// draws `outcomes`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_finite_action_decision<A, O>(
    problem: DecisionProblem<A, O>,
    outcomes: Vec<O>,
    ctx: &ExecutionContext,
) -> Result<CompiledFiniteActionDecision<A, O>, CausalError> {
    admit_compile(ctx)?;
    // The actions and outcome draws are opaque caller types: their counts are declared, their
    // values are read only through the callbacks. The utility and every constraint are runtime
    // bindings; a constraint declares its name and the utility may declare its affine form.
    let mut declaration = Declaration::new("antecedent.finite_action_decision.v2");
    declaration
        .field("actions", &problem.actions.len())
        .field("outcomes", &outcomes.len())
        .field("chance_threshold", &problem.chance_threshold)
        .callback("utility", None)
        .field("utility_affine_form", &problem.utility.affine_in_state(&problem.actions));
    for constraint in &problem.constraints {
        declaration.callback("constraint", Some(constraint.name()));
    }
    let decision = DecisionProgram {
        carrier: CarrierDemand::ParameterPosterior,
        utility: DecisionProgram::utility_form(None),
        actions: problem.actions.len(),
        constraints: problem.constraints.len(),
        chance_threshold: problem.chance_threshold,
        utility_unit: None,
        contract_identity: None,
        information_value: None,
        scenario_source: Some(Arc::from("state_draws")),
    };
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::FiniteActionDecision,
        plan_id: "supplied.finite_action_decision.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_state_draws",
        question: declaration.finish(),
        evaluator: "decision::evaluate_decision",
        reading: Reading::Posterior,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledFiniteActionDecision { lowered, problem, outcomes })
}

impl<A, O> CompiledFiniteActionDecision<A, O> {
    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// The plan view of the compiled program. Reading it runs nothing.
    #[must_use]
    pub fn plan_view(&self) -> crate::planner::PlanView {
        crate::planner::PlanView::of_program(
            Arc::clone(&self.lowered),
            crate::planner::PlanStage::Compiled,
        )
    }

    /// Identity of the compiled program.
    #[must_use]
    pub fn identity(&self) -> antecedent_core::SemanticDigest {
        self.lowered.identity()
    }

    /// Evaluate the Bayes action and its expected regret.
    ///
    /// # Errors
    /// The evaluation's own refusals (a callback failure, no actions or draws, no admissible
    /// action), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DecisionEvaluation, TerminalError<DesignError>> {
        let _scoped = crate::analysis::supplied_terminal::enter(ctx, self)?;
        evaluate_decision(&self.problem, &self.outcomes).map_err(TerminalError::Kernel)
    }
}

// --- a search over adaptive policy families -------------------------------------------------

/// Whether a release build refuses an adaptive policy search whose support unit is not
/// supported.
///
/// The unit is the capability record bound to the `AdaptivePolicySearch` route in
/// `parity/compiled_units.toml`. The gate stays closed until that record's evidence is earned,
/// so no valid use refuses before then; flipping this constant is the only change that turns
/// it on.
pub(crate) const ADAPTIVE_POLICY_SEARCH_GATE_LIVE: bool = false;

/// A compiled bounded search over adaptive policy families.
#[derive(Clone, Debug)]
pub struct CompiledAdaptivePolicySearch {
    lowered: Arc<LoweredProgram>,
    search: PreparedAdaptivePolicySearch,
}

impl CompiledProgramHandle for CompiledAdaptivePolicySearch {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

/// Compile a prepared adaptive policy search. The response source is bound at execution: the
/// program identity covers the families, constraints, objective and options, and the result
/// carries the source's identity.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_adaptive_policy_search(
    search: PreparedAdaptivePolicySearch,
    ctx: &ExecutionContext,
) -> Result<CompiledAdaptivePolicySearch, CausalError> {
    admit_compile(ctx)?;
    // The prepared search by its own canonical identity (families, constraints, objective,
    // options and baseline). The response source is bound at execution; the result carries
    // its identity.
    let mut declaration = Declaration::new("antecedent.adaptive_policy_search.v2");
    declaration
        .field("search", &SemanticDigest::from_bytes(search.identity()).to_hex())
        .callback("response_source", None);
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::AdaptivePolicySearch,
        plan_id: "supplied.adaptive_policy_search.v1",
        functional: FunctionalKind::RegimeValue,
        graph_class: "declared_policy_families",
        question: declaration.finish(),
        evaluator: "adaptive_policy_search::execute",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: None,
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledAdaptivePolicySearch { lowered, search })
}

impl CompiledAdaptivePolicySearch {
    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// The plan view of the compiled program. Reading it runs nothing.
    #[must_use]
    pub fn plan_view(&self) -> crate::planner::PlanView {
        crate::planner::PlanView::of_program(
            Arc::clone(&self.lowered),
            crate::planner::PlanStage::Compiled,
        )
    }

    /// Identity of the compiled program.
    #[must_use]
    pub fn identity(&self) -> antecedent_core::SemanticDigest {
        self.lowered.identity()
    }

    /// The prepared search.
    #[must_use]
    pub const fn search(&self) -> &PreparedAdaptivePolicySearch {
        &self.search
    }

    /// Run the search against `source`. Cancellation is the search's own refusal, as on the
    /// direct path.
    ///
    /// # Errors
    /// The search's own refusals, or the compiler's support gate once it is live.
    pub fn execute(
        &self,
        source: &dyn PolicyRolloutSource,
        ctx: &ExecutionContext,
    ) -> Result<AdaptivePolicySearchResult, TerminalError<AdaptivePolicySearchError>> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if ADAPTIVE_POLICY_SEARCH_GATE_LIVE {
            crate::analysis::program_handle::gate_program(&scoped, self)
                .map_err(CausalError::from)?;
        }
        self.search.execute(source, &scoped).map_err(TerminalError::Kernel)
    }
}

#[cfg(test)]
#[path = "input_decision_programs_site_tests.rs"]
mod site_tests;
