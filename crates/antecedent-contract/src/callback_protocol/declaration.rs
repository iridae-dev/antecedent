//! The callback declaration: operation, schemas, semantics, provider identity,
//! dependencies, replay contract, support and trust, as data.
//!
//! A declaration runs nothing. It is an instance of the one interface-contract
//! schema ([`CallbackDeclaration::interface_contract`]) plus the execution
//! facts an interface contract has no slot for (the provider object, the
//! dependency set, the replay contract, the trust the provider claims). Its
//! canonical identity binds both.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::identity::{CanonicalWriter, SemanticDigest};

use super::refusal::{
    CallbackRefusal, blank_field, duplicate_name, invalid_quantity, invalid_support,
    meaning_not_for_kind, nested_inputs_mismatch, no_operations, operation_not_for_kind,
    replay_contract_incomplete, too_many_items, trust_incomplete, update_declaration_mismatch,
};
use crate::composition_provenance::CompositionStage;
use crate::evidence_ledger::SourceId;
use crate::interface_contract::{
    CalibrationStatus, ConsumedPort, ContractOperation, DependenceRequirement, EvidenceBasis,
    InterfaceContract, MAX_INTERFACE_TEXT_BYTES, ProducedPort, SupportDomain, UncertaintyMeaning,
    UncertaintyParadigm,
};
use crate::{
    DistributionMeaning, ExternalCapability, ExternalPosteriorKind, ExternalScientificObject,
    LawProviderContract, PosteriorProviderContract, ProviderObjectIdentity,
    ProviderVerificationReceipt, ScientificQuantity, SignalProviderContract,
};

/// Version word written into every callback declaration identity. Bump only
/// when the canonical encoding below changes.
pub const CALLBACK_PROTOCOL_VERSION: u64 = 1;
/// Most dependencies one declaration may list.
pub const MAX_CALLBACK_DEPENDENCIES: usize = 1024;
/// Most inputs one declaration may list.
pub const MAX_CALLBACK_INPUTS: usize = 256;
/// Longest identity or name, in bytes.
pub const MAX_CALLBACK_TEXT_BYTES: usize = MAX_INTERFACE_TEXT_BYTES;
/// The component name's single produced port.
pub const CALLBACK_OUTPUT_PORT: &str = "output";

const TAG_DECLARATION: &str = "antecedent.callback_protocol";

/// What the callback is.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CallbackKind {
    /// A probability law over one scalar quantity.
    Law,
    /// A posterior over a parameter, functional or causal functional.
    Posterior,
    /// A callback that itself consumes the outputs of other callbacks.
    Nested,
}

impl CallbackKind {
    /// Every kind.
    pub const ALL: [Self; 3] = [Self::Law, Self::Posterior, Self::Nested];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Law => "law",
            Self::Posterior => "posterior",
            Self::Nested => "nested",
        }
    }

    /// Parse the name written by [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Law => 1,
            Self::Posterior => 2,
            Self::Nested => 3,
        }
    }
}

/// One operation a callback may answer. Each is negotiated on its own: a mean
/// does not determine a law, draws do not determine a distribution function,
/// and none determines a posterior update.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CallbackOperation {
    /// The mean of the quantity.
    Mean,
    /// Marginal draws of the quantity.
    Sample,
    /// The distribution function at named thresholds.
    Cdf,
    /// The quantile function at named levels.
    Quantile,
    /// The posterior after a possible observation.
    PosteriorUpdate,
}

impl CallbackOperation {
    /// Every operation, in canonical order.
    pub const ALL: [Self; 5] =
        [Self::Mean, Self::Sample, Self::Cdf, Self::Quantile, Self::PosteriorUpdate];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Sample => "sample",
            Self::Cdf => "cdf",
            Self::Quantile => "quantile",
            Self::PosteriorUpdate => "posterior_update",
        }
    }

    /// Parse the name written by [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|operation| operation.as_str() == name)
    }

    /// The provider capability the operation is.
    #[must_use]
    pub const fn capability(self) -> ExternalCapability {
        match self {
            Self::Mean => ExternalCapability::Mean,
            Self::Sample => ExternalCapability::Sample,
            Self::Cdf => ExternalCapability::Cdf,
            Self::Quantile => ExternalCapability::Quantile,
            Self::PosteriorUpdate => ExternalCapability::Update,
        }
    }

    /// The interface-contract operation the operation discloses. A distribution
    /// function and a quantile function both disclose a law there; the protocol
    /// keeps them apart.
    #[must_use]
    pub const fn contract_operation(self) -> ContractOperation {
        match self {
            Self::Mean => ContractOperation::Mean,
            Self::Sample => ContractOperation::Draws,
            Self::Cdf | Self::Quantile => ContractOperation::Law,
            Self::PosteriorUpdate => ContractOperation::Posterior,
        }
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Mean => 1,
            Self::Sample => 2,
            Self::Cdf => 3,
            Self::Quantile => 4,
            Self::PosteriorUpdate => 5,
        }
    }
}

/// What a declared dependency is.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DependencyKind {
    /// A data snapshot.
    Data,
    /// A model, its weights or its parameters.
    Model,
    /// The runtime environment (interpreter, libraries, hardware class).
    Environment,
    /// A file.
    File,
    /// A remote service version.
    RemoteService,
    /// State captured by the callback's closure.
    ClosureState,
}

impl DependencyKind {
    /// Every kind.
    pub const ALL: [Self; 6] = [
        Self::Data,
        Self::Model,
        Self::Environment,
        Self::File,
        Self::RemoteService,
        Self::ClosureState,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Data => "data",
            Self::Model => "model",
            Self::Environment => "environment",
            Self::File => "file",
            Self::RemoteService => "remote_service",
            Self::ClosureState => "closure_state",
        }
    }

    /// Parse the name written by [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Data => 1,
            Self::Model => 2,
            Self::Environment => 3,
            Self::File => 4,
            Self::RemoteService => 5,
            Self::ClosureState => 6,
        }
    }
}

/// One declared dependency with the version or digest it was used at. A change
/// of the version is a different callback: dependents are invalidated.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CallbackDependency {
    /// What kind of dependency.
    pub kind: DependencyKind,
    /// Stable name, unique per kind.
    pub name: String,
    /// Version or content digest.
    pub version: String,
}

impl CallbackDependency {
    /// A dependency.
    #[must_use]
    pub fn new(kind: DependencyKind, name: impl Into<String>, version: impl Into<String>) -> Self {
        Self { kind, name: name.into(), version: version.into() }
    }
}

/// Whether the callback's dependencies are fully declared. Anything but a
/// complete declaration makes a stored result ineligible for reuse.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DependencyDeclaration {
    /// Every dependency is listed, at a version. An empty list says the
    /// callback has none beyond the provider object's own version.
    Complete(Vec<CallbackDependency>),
    /// The provider says it has dependencies it will not describe.
    Opaque,
    /// The producer did not say.
    Undeclared,
}

impl DependencyDeclaration {
    /// The listed dependencies, when complete.
    #[must_use]
    pub fn complete(&self) -> Option<&[CallbackDependency]> {
        match self {
            Self::Complete(dependencies) => Some(dependencies),
            _ => None,
        }
    }
}

/// How a failed side-effecting call may be repeated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryContract {
    /// Never repeated: effects of a repeated call are unknown.
    NoRetry,
    /// May be repeated with the same idempotency key, at most `max_attempts`
    /// invocations in all.
    IdempotentKey {
        /// Most invocations of one request, the first included.
        max_attempts: u32,
    },
}

/// The replay contract the callback declares. Reuse of a stored result needs
/// the identity this contract names to match.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayContract {
    /// A pure function of the declared dependencies and the request.
    DeterministicPure,
    /// A function of the declared dependencies, the request and a seed, drawn
    /// from the stream with this semantic identity.
    SeededStochastic {
        /// What the stream is for, independent of call order and count.
        stream_id: String,
    },
    /// Depends on state that changes between calls.
    Stateful {
        /// Whether the provider reports a state identity through the protocol.
        exposes_state_identity: bool,
    },
    /// Has effects outside the process.
    SideEffecting {
        /// How a repeated call is governed.
        retry: RetryContract,
    },
}

impl ReplayContract {
    /// Stable `snake_case` label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::DeterministicPure => "deterministic_pure",
            Self::SeededStochastic { .. } => "seeded_stochastic",
            Self::Stateful { .. } => "stateful",
            Self::SideEffecting { .. } => "side_effecting",
        }
    }

    /// Whether the callback may have effects outside the process.
    #[must_use]
    pub const fn has_side_effects(&self) -> bool {
        matches!(self, Self::SideEffecting { .. })
    }
}

/// The trust the provider claims. A claim is believed only where it is
/// confirmed; there is no native level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CallbackTrust {
    /// No claim.
    Unverified,
    /// The supplier asserts it. No numerical check was performed here.
    ExternallyAttested {
        /// Party making the assertion.
        attestor: String,
    },
    /// The exact provider contract passed object-level verification, in the
    /// receipt with this digest.
    ExactRequestVerified {
        /// Digest of the verification receipt.
        receipt_digest: String,
    },
}

impl CallbackTrust {
    /// Stable `snake_case` label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Unverified => "unverified",
            Self::ExternallyAttested { .. } => "externally_attested",
            Self::ExactRequestVerified { .. } => "verified_extension",
        }
    }
}

/// The posterior-update declaration: the candidate study, the prior or state
/// the update consumes and the possible observation.
#[derive(Clone, Debug, PartialEq)]
pub struct UpdateDeclaration {
    /// Stable candidate identity.
    pub candidate_id: String,
    /// Prior or current-state identity.
    pub prior_id: String,
    /// The possible observation.
    pub observation: ScientificQuantity,
}

/// Where an input comes from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InputSupplier {
    /// A claim from outside the callback graph, connected through the interface
    /// contract (a transported claim, an observed summary).
    External,
    /// The output of another callback, by its declaration identity (hex), with
    /// the evidence that callback declares.
    Callback {
        /// Canonical identity of the inner declaration (hex).
        identity: String,
        /// The inner callback's evidence basis.
        evidence: EvidenceBasis,
    },
}

/// One input of the callback.
#[derive(Clone, Debug, PartialEq)]
pub struct CallbackInput {
    /// Name, unique among the inputs.
    pub name: String,
    /// The scientific quantity the input must be.
    pub quantity: ScientificQuantity,
    /// The estimand the input must answer.
    pub semantic_identity: String,
    /// The values the input takes.
    pub support: SupportDomain,
    /// The operation of the supplier the callback consumes.
    pub operation: CallbackOperation,
    /// What the supplier's randomness means; present exactly when the
    /// operation carries uncertainty (sample, distribution function, quantile
    /// function, posterior update).
    pub uncertainty: Option<DistributionMeaning>,
    /// Where the input comes from.
    pub supplier: InputSupplier,
}

/// A callback as the protocol declares it.
#[derive(Clone, Debug, PartialEq)]
pub struct CallbackDeclaration {
    /// Component name in the interface contract.
    pub component: String,
    /// What the callback is.
    pub kind: CallbackKind,
    /// Provider, object, version, snapshot and request identity.
    pub identity: ProviderObjectIdentity,
    /// The coordinate of the quantity the callback answers about.
    pub quantity: ScientificQuantity,
    /// The estimand or semantic identity the value answers.
    pub semantic_identity: String,
    /// What one draw, and any distribution function, means.
    pub meaning: DistributionMeaning,
    /// The values the quantity takes.
    pub support: SupportDomain,
    /// The operations answered, each declared on its own.
    pub operations: BTreeSet<CallbackOperation>,
    /// The posterior-update declaration, exactly when the update is answered.
    pub update: Option<UpdateDeclaration>,
    /// The input schema.
    pub inputs: Vec<CallbackInput>,
    /// Data, model, file, remote and environment dependencies.
    pub dependencies: DependencyDeclaration,
    /// Randomness, state and side-effect contract.
    pub replay: ReplayContract,
    /// The trust the provider claims.
    pub trust: CallbackTrust,
    /// The evidence sources (ledger identities) the callback itself rests on;
    /// inputs add theirs. `Undeclared` leaves overlap unknown.
    pub evidence: EvidenceBasis,
}

fn check_text(field: &str, text: &str) -> Result<(), CallbackRefusal> {
    if text.trim().is_empty() || text.len() > MAX_CALLBACK_TEXT_BYTES {
        Err(blank_field(field))
    } else {
        Ok(())
    }
}

impl CallbackDeclaration {
    /// Check the declaration is well formed.
    ///
    /// # Errors
    /// The first defect, as a `callback_protocol_invalid` refusal that names
    /// the declaration that would make the callback well formed.
    pub fn validate(&self) -> Result<(), CallbackRefusal> {
        self.validate_identity()?;
        self.validate_operations()?;
        self.validate_inputs()?;
        self.validate_dependencies_and_replay()?;
        match &self.trust {
            CallbackTrust::Unverified => {}
            CallbackTrust::ExternallyAttested { attestor } => {
                if attestor.trim().is_empty() || attestor.len() > MAX_CALLBACK_TEXT_BYTES {
                    return Err(trust_incomplete());
                }
            }
            CallbackTrust::ExactRequestVerified { receipt_digest } => {
                if receipt_digest.trim().is_empty()
                    || receipt_digest.len() > MAX_CALLBACK_TEXT_BYTES
                {
                    return Err(trust_incomplete());
                }
            }
        }
        if let EvidenceBasis::Declared(sources) = &self.evidence {
            if sources.len() > MAX_CALLBACK_DEPENDENCIES {
                return Err(too_many_items("evidence"));
            }
        }
        // The projection is part of the declaration: it must be a valid
        // interface contract of the one schema.
        self.interface_contract().validate()?;
        Ok(())
    }

    fn validate_identity(&self) -> Result<(), CallbackRefusal> {
        check_text("component", &self.component)?;
        for (field, value) in [
            ("identity.provider_id", &self.identity.provider_id),
            ("identity.object_id", &self.identity.object_id),
            ("identity.version_id", &self.identity.version_id),
            ("identity.snapshot_id", &self.identity.snapshot_id),
            ("identity.request_id", &self.identity.request_id),
        ] {
            check_text(field, value)?;
        }
        check_text("semantic_identity", &self.semantic_identity)?;
        self.quantity.validate().map_err(|_| invalid_quantity("output"))?;
        if !self.support.is_well_formed()
            || matches!(self.support, SupportDomain::Categorical { .. })
        {
            return Err(invalid_support("output"));
        }
        Ok(())
    }

    fn validate_operations(&self) -> Result<(), CallbackRefusal> {
        if self.operations.is_empty() {
            return Err(no_operations());
        }
        let kind = self.kind.as_str();
        for operation in &self.operations {
            let allowed = match self.kind {
                CallbackKind::Law => !matches!(operation, CallbackOperation::PosteriorUpdate),
                CallbackKind::Posterior => {
                    !matches!(operation, CallbackOperation::Cdf | CallbackOperation::Quantile)
                }
                CallbackKind::Nested => true,
            };
            if !allowed {
                return Err(operation_not_for_kind(operation.as_str(), kind));
            }
        }
        let posterior_meaning =
            UncertaintyMeaning::new(self.meaning).paradigm() == UncertaintyParadigm::Posterior;
        let meaning_ok = match self.kind {
            CallbackKind::Law | CallbackKind::Nested => !matches!(
                self.meaning,
                DistributionMeaning::ParameterPosterior
                    | DistributionMeaning::CausalFunctionalPosterior
            ),
            CallbackKind::Posterior => posterior_meaning,
        };
        let updates = self.operations.contains(&CallbackOperation::PosteriorUpdate);
        if !meaning_ok || (updates && !posterior_meaning) {
            return Err(meaning_not_for_kind(UncertaintyMeaning::new(self.meaning).name(), kind));
        }
        match (&self.update, updates) {
            (Some(update), true) => {
                // As in the provider contract, an update comes with the
                // observation law it updates on.
                if !self.operations.contains(&CallbackOperation::Sample) {
                    return Err(update_declaration_mismatch());
                }
                check_text("update.candidate_id", &update.candidate_id)?;
                check_text("update.prior_id", &update.prior_id)?;
                update
                    .observation
                    .validate()
                    .map_err(|_| invalid_quantity("update.observation"))?;
                Ok(())
            }
            (None, false) => Ok(()),
            _ => Err(update_declaration_mismatch()),
        }
    }

    fn validate_inputs(&self) -> Result<(), CallbackRefusal> {
        if self.inputs.len() > MAX_CALLBACK_INPUTS {
            return Err(too_many_items("inputs"));
        }
        let mut names = BTreeSet::new();
        let mut from_callbacks = 0usize;
        for input in &self.inputs {
            check_text("input.name", &input.name)?;
            if !names.insert(input.name.as_str()) {
                return Err(duplicate_name(&input.name));
            }
            input.quantity.validate().map_err(|_| invalid_quantity(&input.name))?;
            check_text("input.semantic_identity", &input.semantic_identity)?;
            if !input.support.is_well_formed() {
                return Err(invalid_support(&input.name));
            }
            if input.operation.contract_operation().carries_uncertainty()
                != input.uncertainty.is_some()
            {
                return Err(meaning_not_for_kind(&input.name, "input"));
            }
            if let InputSupplier::Callback { identity, evidence } = &input.supplier {
                from_callbacks += 1;
                check_text("input.callback_identity", identity)?;
                if let EvidenceBasis::Declared(sources) = evidence {
                    if sources.len() > MAX_CALLBACK_DEPENDENCIES {
                        return Err(too_many_items("input.evidence"));
                    }
                }
            }
        }
        match (self.kind == CallbackKind::Nested, from_callbacks > 0) {
            (true, false) | (false, true) => Err(nested_inputs_mismatch()),
            _ => Ok(()),
        }
    }

    fn validate_dependencies_and_replay(&self) -> Result<(), CallbackRefusal> {
        if let DependencyDeclaration::Complete(dependencies) = &self.dependencies {
            if dependencies.len() > MAX_CALLBACK_DEPENDENCIES {
                return Err(too_many_items("dependencies"));
            }
            let mut seen = BTreeSet::new();
            for dependency in dependencies {
                check_text("dependency.name", &dependency.name)?;
                check_text("dependency.version", &dependency.version)?;
                if !seen.insert((dependency.kind, dependency.name.as_str())) {
                    return Err(duplicate_name(&dependency.name));
                }
            }
        }
        match &self.replay {
            ReplayContract::SeededStochastic { stream_id } => {
                if stream_id.trim().is_empty() || stream_id.len() > MAX_CALLBACK_TEXT_BYTES {
                    return Err(replay_contract_incomplete("stream_id"));
                }
            }
            ReplayContract::SideEffecting {
                retry: RetryContract::IdempotentKey { max_attempts },
            } => {
                if *max_attempts == 0 {
                    return Err(replay_contract_incomplete("max_attempts"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// The evidence the output rests on: the callback's own sources and those
    /// of every callback input. Unknown if any part is undeclared; never
    /// widened.
    #[must_use]
    pub fn effective_evidence(&self) -> EvidenceBasis {
        let EvidenceBasis::Declared(own) = &self.evidence else {
            return EvidenceBasis::Undeclared;
        };
        let mut sources: BTreeSet<SourceId> = own.clone();
        for input in &self.inputs {
            if let InputSupplier::Callback { evidence, .. } = &input.supplier {
                match evidence {
                    EvidenceBasis::Declared(inner) => sources.extend(inner.iter().copied()),
                    EvidenceBasis::Undeclared => return EvidenceBasis::Undeclared,
                }
            }
        }
        EvidenceBasis::Declared(sources)
    }

    /// Whether the declaration lists every dependency.
    #[must_use]
    pub const fn dependencies_complete(&self) -> bool {
        matches!(self.dependencies, DependencyDeclaration::Complete(_))
    }

    /// The component as one instance of the interface-contract schema: the
    /// quantity the callback produces with the operations it declares, and one
    /// consumed port per input.
    ///
    /// The produced port is point-only when only a mean is offered and
    /// unmeasured otherwise; it is never measured, so no interval or posterior
    /// guarantee is carried.
    #[must_use]
    pub fn interface_contract(&self) -> InterfaceContract {
        let operations: BTreeSet<ContractOperation> =
            self.operations.iter().copied().map(CallbackOperation::contract_operation).collect();
        let carries = operations.iter().copied().any(ContractOperation::carries_uncertainty);
        let mut produced = ProducedPort::new(
            CALLBACK_OUTPUT_PORT,
            self.quantity.clone(),
            self.semantic_identity.clone(),
            self.support.clone(),
            operations,
        )
        .with_provenance([CompositionStage::ExternalProvider.as_str()]);
        if carries {
            produced = produced
                .with_uncertainty(self.meaning)
                .with_calibration(CalibrationStatus::Unmeasured);
        } else {
            produced = produced.with_calibration(CalibrationStatus::PointOnly);
        }
        if let EvidenceBasis::Declared(sources) = self.effective_evidence() {
            produced = produced.with_evidence(sources);
        }
        let consumes = self
            .inputs
            .iter()
            .map(|input| {
                let operation = input.operation.contract_operation();
                let mut port = ConsumedPort::new(
                    input.name.clone(),
                    input.quantity.clone(),
                    input.semantic_identity.clone(),
                    input.support.clone(),
                    [operation],
                )
                .with_dependence(DependenceRequirement::DeclaredEvidence);
                if let Some(meaning) = input.uncertainty {
                    port = port.with_uncertainty(meaning);
                }
                port
            })
            .collect();
        InterfaceContract { component: self.component.clone(), produces: vec![produced], consumes }
    }

    /// The provider object the declaration bridges to in the one provider
    /// contract, so negotiation and exact-request verification apply to the
    /// callback unchanged.
    ///
    /// A law and a nested callback are a law; a posterior callback a posterior.
    /// A posterior update is a separate signal object
    /// ([`Self::update_object`]).
    #[must_use]
    pub fn provider_object(&self) -> ExternalScientificObject {
        let capabilities: Vec<ExternalCapability> = self
            .operations
            .iter()
            .copied()
            .filter(|op| !matches!(op, CallbackOperation::PosteriorUpdate))
            .map(CallbackOperation::capability)
            .collect();
        match self.kind {
            CallbackKind::Law | CallbackKind::Nested => {
                ExternalScientificObject::Law(LawProviderContract {
                    identity: self.identity.clone(),
                    quantities: vec![self.quantity.clone()],
                    meaning: self.meaning,
                    capabilities,
                })
            }
            CallbackKind::Posterior => {
                let kind = match self.meaning {
                    DistributionMeaning::ParameterPosterior => ExternalPosteriorKind::Parameter,
                    DistributionMeaning::CausalFunctionalPosterior => {
                        ExternalPosteriorKind::CausalFunctional
                    }
                    _ => ExternalPosteriorKind::Functional,
                };
                ExternalScientificObject::Posterior(PosteriorProviderContract {
                    identity: self.identity.clone(),
                    kind,
                    quantities: vec![self.quantity.clone()],
                    capabilities,
                })
            }
        }
    }

    /// The posterior-update signal object, when the update is declared. The
    /// provider contract requires an update to come with an observation law,
    /// so it exists only when sampling is declared too.
    #[must_use]
    pub fn update_object(&self) -> Option<ExternalScientificObject> {
        let update = self.update.as_ref()?;
        if !self.operations.contains(&CallbackOperation::Sample) {
            return None;
        }
        Some(ExternalScientificObject::Signal(SignalProviderContract {
            identity: self.identity.clone(),
            candidate_id: update.candidate_id.clone(),
            prior_id: update.prior_id.clone(),
            observation: update.observation.clone(),
            capabilities: vec![ExternalCapability::Sample, ExternalCapability::Update],
        }))
    }

    /// Whether `receipt` is the verification this declaration names and covers
    /// the exact provider object. Only then does a declared
    /// `ExactRequestVerified` trust stand.
    #[must_use]
    pub fn confirms_verification(&self, receipt: &ProviderVerificationReceipt) -> bool {
        match &self.trust {
            CallbackTrust::ExactRequestVerified { receipt_digest } => {
                *receipt_digest == receipt.digest() && receipt.covers(&self.provider_object())
            }
            _ => false,
        }
    }

    /// The canonical identity of the declaration: a [`CanonicalWriter`] stream
    /// under the tag `antecedent.callback_protocol`. Operation, input,
    /// dependency and evidence order do not affect it; a version of a
    /// dependency, the replay contract or the provider request do.
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "one canonical-encoding pass over every declared field, in declaration order"
    )]
    pub fn canonical_identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(TAG_DECLARATION);
        writer.word(CALLBACK_PROTOCOL_VERSION).text(&self.component).word(self.kind.tag());
        for field in [
            &self.identity.provider_id,
            &self.identity.object_id,
            &self.identity.version_id,
            &self.identity.snapshot_id,
            &self.identity.request_id,
        ] {
            writer.text(field);
        }
        self.quantity.write_canonical(&mut writer);
        writer.text(&self.semantic_identity).text(crate::quantity::meaning_tag(self.meaning));
        write_support(&mut writer, &self.support);
        writer.word(self.operations.len() as u64);
        for operation in &self.operations {
            writer.word(operation.tag());
        }
        match &self.update {
            Some(update) => {
                writer.word(1).text(&update.candidate_id).text(&update.prior_id);
                update.observation.write_canonical(&mut writer);
            }
            None => {
                writer.word(0);
            }
        }
        let mut inputs: Vec<&CallbackInput> = self.inputs.iter().collect();
        inputs.sort_by(|a, b| a.name.cmp(&b.name));
        writer.word(inputs.len() as u64);
        for input in inputs {
            writer.text(&input.name);
            input.quantity.write_canonical(&mut writer);
            writer.text(&input.semantic_identity).word(input.operation.tag());
            writer.opt_text(input.uncertainty.map(crate::quantity::meaning_tag));
            write_support(&mut writer, &input.support);
            match &input.supplier {
                InputSupplier::External => {
                    writer.word(0);
                }
                InputSupplier::Callback { identity, evidence } => {
                    writer.word(1).text(identity);
                    write_evidence(&mut writer, evidence);
                }
            }
        }
        match &self.dependencies {
            DependencyDeclaration::Complete(dependencies) => {
                let mut sorted: Vec<&CallbackDependency> = dependencies.iter().collect();
                sorted.sort();
                writer.word(1).word(sorted.len() as u64);
                for dependency in sorted {
                    writer
                        .word(dependency.kind.tag())
                        .text(&dependency.name)
                        .text(&dependency.version);
                }
            }
            DependencyDeclaration::Opaque => {
                writer.word(2);
            }
            DependencyDeclaration::Undeclared => {
                writer.word(0);
            }
        }
        match &self.replay {
            ReplayContract::DeterministicPure => {
                writer.word(1);
            }
            ReplayContract::SeededStochastic { stream_id } => {
                writer.word(2).text(stream_id);
            }
            ReplayContract::Stateful { exposes_state_identity } => {
                writer.word(3).word(u64::from(*exposes_state_identity));
            }
            ReplayContract::SideEffecting { retry } => {
                writer.word(4);
                match retry {
                    RetryContract::NoRetry => {
                        writer.word(0);
                    }
                    RetryContract::IdempotentKey { max_attempts } => {
                        writer.word(1).word(u64::from(*max_attempts));
                    }
                }
            }
        }
        match &self.trust {
            CallbackTrust::Unverified => {
                writer.word(0);
            }
            CallbackTrust::ExternallyAttested { attestor } => {
                writer.word(1).text(attestor);
            }
            CallbackTrust::ExactRequestVerified { receipt_digest } => {
                writer.word(2).text(receipt_digest);
            }
        }
        write_evidence(&mut writer, &self.evidence);
        SemanticDigest::from_bytes(writer.finish())
    }
}

fn write_evidence(writer: &mut CanonicalWriter, evidence: &EvidenceBasis) {
    match evidence {
        EvidenceBasis::Undeclared => {
            writer.word(0);
        }
        EvidenceBasis::Declared(sources) => {
            writer.word(1).word(sources.len() as u64);
            for source in sources {
                writer.text(&source.to_hex());
            }
        }
    }
}

fn write_support(writer: &mut CanonicalWriter, support: &SupportDomain) {
    match support {
        SupportDomain::Real => {
            writer.word(1);
        }
        SupportDomain::NonNegative => {
            writer.word(2);
        }
        SupportDomain::Binary => {
            writer.word(3);
        }
        SupportDomain::Interval { lower_bits, upper_bits } => {
            writer.word(4).word(*lower_bits).word(*upper_bits);
        }
        SupportDomain::Categorical { levels } => {
            writer.word(5).word(levels.len() as u64);
            for level in levels {
                writer.text(level);
            }
        }
    }
}
