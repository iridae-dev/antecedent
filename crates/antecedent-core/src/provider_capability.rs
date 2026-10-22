//! Declared capabilities of an inference provider.
//!
//! A provider can predict, evaluate a density, sample, differentiate a log
//! density, form estimating equations, contribute an influence function, update,
//! integrate exactly, or step a state only when a [`ProviderDeclaration`] lists
//! that capability. The provider name is a label. It is not a capability, and
//! compilation does not read one out of it.
//!
//! [`ProviderDeclaration::compile`] checks one [`CapabilityRequest`] against the
//! declared set and stops there. A declared capability becomes a
//! [`CompiledCapabilityRequest`]. No sampling, density evaluation, integration,
//! update, or fit runs. A missing capability is a [`FailedCapabilityAttempt`]:
//! the attempt at that operation failed before any of that work. The diagnostic
//! names the provider and the capability that would be a legal next request,
//! taken from the declared set. An empty declaration supports nothing, so the
//! diagnostic names no next capability.
//!
//! [`ProviderDeclaration::probe`] asks the body to answer that same request.
//! The declaration is not consulted. A capability the declaration lists and
//! the body does not supply is an [`UnansweredCapability`]. The probe does not
//! evaluate a density, update a posterior, or fit a model.
//!
//! Two declarations with the same name and different sets are different values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use crate::diagnostic::{Diagnostic, DiagnosticKind, DiagnosticSeverity};
use crate::named_variants;
use crate::text::is_blank;

/// Capability an inference provider may be asked to perform.
///
/// Each variant is independent. Declaring density does not declare sampling,
/// differentiable log density, or prediction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ProviderCapability {
    /// Predictive simulation from the declared model.
    Prediction,
    /// Conditional density.
    Density,
    /// Draws from the declared model.
    Sampling,
    /// Differentiable log density. Density alone does not supply this.
    DifferentiableLogDensity,
    /// Estimating equations for a declared estimator.
    EstimatingEquations,
    /// Influence-function contributions.
    InfluenceContributions,
    /// Posterior or generalized-posterior updating.
    Updating,
    /// Exact integration, as opposed to a Monte Carlo estimate.
    ExactIntegration,
    /// One declared state transition.
    StateTransition,
}

named_variants! {
    ProviderCapability {
        Prediction => "prediction",
        Density => "density",
        Sampling => "sampling",
        DifferentiableLogDensity => "differentiable_log_density",
        EstimatingEquations => "estimating_equations",
        InfluenceContributions => "influence_contributions",
        Updating => "updating",
        ExactIntegration => "exact_integration",
        StateTransition => "state_transition",
    }
}

impl ProviderCapability {
    const fn mask(self) -> u16 {
        match self {
            Self::Prediction => 1u16,
            Self::Density => 1u16 << 1,
            Self::Sampling => 1u16 << 2,
            Self::DifferentiableLogDensity => 1u16 << 3,
            Self::EstimatingEquations => 1u16 << 4,
            Self::InfluenceContributions => 1u16 << 5,
            Self::Updating => 1u16 << 6,
            Self::ExactIntegration => 1u16 << 7,
            Self::StateTransition => 1u16 << 8,
        }
    }
}

impl fmt::Display for ProviderCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Caller-chosen id of one provider.
///
/// The id is not a capability and it is not parsed from the provider name.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProviderId(u64);

impl ProviderId {
    /// Id from its raw value.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw id.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "P{}", self.0)
    }
}

/// Explicit set of [`ProviderCapability`] values a provider declared.
///
/// Membership is the bits set by [`Self::from_capabilities`]. The provider name
/// is not an input.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct CapabilitySet {
    bits: u16,
}

impl CapabilitySet {
    /// No capabilities.
    #[must_use]
    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    /// Set containing each capability in `capabilities`, once.
    ///
    /// Order and duplicates do not change the set.
    #[must_use]
    pub fn from_capabilities(capabilities: impl IntoIterator<Item = ProviderCapability>) -> Self {
        let mut bits = 0u16;
        for capability in capabilities {
            bits |= capability.mask();
        }
        Self { bits }
    }

    /// Whether `capability` was declared.
    #[must_use]
    pub const fn contains(self, capability: ProviderCapability) -> bool {
        self.bits & capability.mask() != 0
    }

    /// Whether nothing was declared.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// How many capabilities were declared.
    #[must_use]
    pub const fn len(self) -> usize {
        let mut count = 0usize;
        let mut index = 0u16;
        while index < 9 {
            if self.bits & (1u16 << index) != 0 {
                count += 1;
            }
            index += 1;
        }
        count
    }

    /// This set plus `capability`.
    #[must_use]
    const fn insert(self, capability: ProviderCapability) -> Self {
        Self { bits: self.bits | capability.mask() }
    }

    /// Earliest declared capability in [`ProviderCapability::ALL`].
    ///
    /// This is a legal next request when the set is non-empty. It is `None`
    /// when the set is empty. Bit 0 is the first entry of that list. The choice
    /// uses the declared bits only.
    #[must_use = "none means the declaration supports no capability"]
    pub const fn legal_next(self) -> Option<ProviderCapability> {
        match self.bits.trailing_zeros() {
            0 => Some(ProviderCapability::Prediction),
            1 => Some(ProviderCapability::Density),
            2 => Some(ProviderCapability::Sampling),
            3 => Some(ProviderCapability::DifferentiableLogDensity),
            4 => Some(ProviderCapability::EstimatingEquations),
            5 => Some(ProviderCapability::InfluenceContributions),
            6 => Some(ProviderCapability::Updating),
            7 => Some(ProviderCapability::ExactIntegration),
            8 => Some(ProviderCapability::StateTransition),
            _ => None,
        }
    }
}

/// Calls a provider body answers.
///
/// Distinct from the declared set. Listing a capability does not supply the call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
struct ProviderBody {
    supplied: CapabilitySet,
}

impl ProviderBody {
    #[must_use]
    const fn supply(self, capability: ProviderCapability) -> Self {
        Self { supplied: self.supplied.insert(capability) }
    }

    #[must_use]
    const fn supplies(self, capability: ProviderCapability) -> bool {
        self.supplied.contains(capability)
    }
}

/// One provider, the capabilities it declared, and the calls its body answers.
///
/// Equality uses the id, the name, the capability set, and the body. The same
/// name with a different set or a different body is a different declaration.
/// The body starts empty. [`Self::answering`] supplies calls.
/// [`Self::compile`] reads the declaration only. [`Self::probe`] reads the body
/// only.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ProviderDeclaration {
    id: ProviderId,
    name: Arc<str>,
    capabilities: CapabilitySet,
    body: ProviderBody,
}

impl ProviderDeclaration {
    /// Declaration of provider `id` named `name` with exactly `capabilities`.
    ///
    /// The body answers no call. [`Self::answering`] supplies calls. Listing a
    /// capability here does not supply one.
    ///
    /// # Errors
    ///
    /// [`ProviderDeclarationError::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank provider name is refused"]
    pub fn new(
        id: ProviderId,
        name: impl Into<Arc<str>>,
        capabilities: impl IntoIterator<Item = ProviderCapability>,
    ) -> Result<Self, ProviderDeclarationError> {
        let name = name.into();
        if is_blank(&name) {
            return Err(ProviderDeclarationError::BlankName);
        }
        Ok(Self {
            id,
            name,
            capabilities: CapabilitySet::from_capabilities(capabilities),
            body: ProviderBody::default(),
        })
    }

    /// Body that answers `calls`.
    ///
    /// The declaration is unchanged. A call absent from `calls` stays unanswered.
    /// Calls already supplied stay supplied.
    #[must_use = "the body answers these calls; the declaration is unchanged"]
    pub fn answering(mut self, calls: impl IntoIterator<Item = ProviderCapability>) -> Self {
        for call in calls {
            self.body = self.body.supply(call);
        }
        self
    }

    /// Provider id.
    #[must_use]
    pub const fn id(&self) -> ProviderId {
        self.id
    }

    /// Provider name. Compilation does not read capabilities from this string.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Capabilities this provider declared.
    #[must_use]
    pub const fn capabilities(&self) -> CapabilitySet {
        self.capabilities
    }

    /// Whether `capability` is in [`Self::capabilities`].
    #[must_use]
    pub const fn supports(&self, capability: ProviderCapability) -> bool {
        self.capabilities.contains(capability)
    }

    /// Check `request` against this declaration.
    ///
    /// The check reads the declared set only. The body is not an input. It does
    /// not sample, evaluate a density, integrate, update, or fit.
    ///
    /// # Errors
    ///
    /// [`FailedCapabilityAttempt`] when `request` names a capability this
    /// declaration does not contain. The error is that failed attempt.
    #[must_use = "an unsupported capability is a failed attempt and must be handled"]
    pub fn compile(
        &self,
        request: CapabilityRequest,
    ) -> Result<CompiledCapabilityRequest, FailedCapabilityAttempt> {
        let capability = request.capability;
        if self.capabilities.contains(capability) {
            return Ok(CompiledCapabilityRequest {
                id: self.id,
                name: Arc::clone(&self.name),
                capability,
            });
        }
        Err(FailedCapabilityAttempt {
            id: self.id,
            name: Arc::clone(&self.name),
            operation: capability,
            declared: self.capabilities,
        })
    }

    /// Check `request` against the declaration and against the body.
    ///
    /// A declared capability is trusted only when the body also answers it. This
    /// is the check to use where the body is available: a listed capability the
    /// body does not supply is refused, not compiled. No density is evaluated and
    /// no model is fit.
    ///
    /// # Errors
    ///
    /// [`ProviderCapabilityRefusal::Undeclared`] when the declaration lacks `request`.
    /// [`ProviderCapabilityRefusal::Unanswered`] when it is declared and the body does not
    /// supply it.
    #[must_use = "an unverified capability is a refusal and must be handled"]
    pub fn compile_verified(
        &self,
        request: CapabilityRequest,
    ) -> Result<CompiledCapabilityRequest, ProviderCapabilityRefusal> {
        let compiled = self.compile(request).map_err(ProviderCapabilityRefusal::Undeclared)?;
        self.probe(request).map_err(ProviderCapabilityRefusal::Unanswered)?;
        Ok(compiled)
    }

    /// Capabilities the declaration lists and the body does not supply.
    ///
    /// Empty when every declared capability is answered.
    #[must_use]
    pub const fn declared_but_unanswered(&self) -> CapabilitySet {
        CapabilitySet { bits: self.capabilities.bits & !self.body.supplied.bits }
    }

    /// Capabilities the body supplies and the declaration does not list.
    ///
    /// The declaration is the contract. A supplied call outside it is never
    /// compiled.
    #[must_use]
    pub const fn answered_but_undeclared(&self) -> CapabilitySet {
        CapabilitySet { bits: self.body.supplied.bits & !self.capabilities.bits }
    }

    /// Ask the body to answer `request`.
    ///
    /// The declaration is not this answer. A capability the declaration lists
    /// and the body does not supply is unanswered. No density is evaluated, no
    /// posterior is updated, and no model is fit.
    ///
    /// # Errors
    ///
    /// [`UnansweredCapability`] when the body does not supply `request`.
    #[must_use = "an unanswered call is a failed probe and must be handled"]
    pub fn probe(
        &self,
        request: CapabilityRequest,
    ) -> Result<CapabilityAnswer, UnansweredCapability> {
        let capability = request.capability();
        if self.body.supplies(capability) {
            return Ok(CapabilityAnswer { id: self.id, name: Arc::clone(&self.name), capability });
        }
        Err(UnansweredCapability { id: self.id, name: Arc::clone(&self.name), capability })
    }
}

/// Refusal from [`ProviderDeclaration::new`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ProviderDeclarationError {
    /// The provider name was empty or whitespace.
    #[error("provider name must be non-blank")]
    BlankName,
}

/// Request to compile one capability for a provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CapabilityRequest {
    capability: ProviderCapability,
}

impl CapabilityRequest {
    /// Request for `capability`.
    #[must_use]
    pub const fn new(capability: ProviderCapability) -> Self {
        Self { capability }
    }

    /// Capability the request asks for.
    #[must_use]
    pub const fn capability(self) -> ProviderCapability {
        self.capability
    }
}

/// A request whose capability the provider declared.
///
/// Compilation produced this value and did not run the provider.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CompiledCapabilityRequest {
    id: ProviderId,
    name: Arc<str>,
    capability: ProviderCapability,
}

impl CompiledCapabilityRequest {
    /// Provider the request was compiled for.
    #[must_use]
    pub const fn provider(&self) -> ProviderId {
        self.id
    }

    /// Provider name. This is a label, not evidence of a capability.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        &self.name
    }

    /// Capability the declaration contains.
    #[must_use]
    pub const fn capability(&self) -> ProviderCapability {
        self.capability
    }
}

/// Failed attempt to compile a capability the provider did not declare.
///
/// The attempt is this value. No provider operation ran. [`Self::legal_next`]
/// is a capability the declaration does contain, when it contains any.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FailedCapabilityAttempt {
    id: ProviderId,
    name: Arc<str>,
    operation: ProviderCapability,
    declared: CapabilitySet,
}

impl FailedCapabilityAttempt {
    /// Provider that was asked.
    #[must_use]
    pub const fn provider(&self) -> ProviderId {
        self.id
    }

    /// Provider name. The legal next request is not parsed from this string.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        &self.name
    }

    /// Capability that was requested and is not declared.
    #[must_use]
    pub const fn operation(&self) -> ProviderCapability {
        self.operation
    }

    /// Capabilities the provider did declare.
    #[must_use]
    pub const fn declared(&self) -> CapabilitySet {
        self.declared
    }

    /// Capability that would be a legal next request, if the provider declared one.
    ///
    /// `None` when the declaration is empty. A provider name that happens to
    /// match a capability word is not used.
    #[must_use = "none means the declaration supports no capability"]
    pub const fn legal_next(&self) -> Option<ProviderCapability> {
        self.declared.legal_next()
    }

    /// Diagnostic naming the provider, the failed operation, and the legal next request.
    #[must_use = "the diagnostic names the provider and the legal next capability"]
    pub fn diagnostic(&self) -> Diagnostic {
        let mut fields = Vec::with_capacity(4);
        fields.push((Arc::<str>::from("provider"), Arc::<str>::from(self.id.to_string())));
        fields.push((Arc::<str>::from("provider_name"), Arc::clone(&self.name)));
        fields.push((Arc::<str>::from("operation"), Arc::<str>::from(self.operation.as_str())));
        if let Some(capability) = self.legal_next() {
            fields.push((Arc::<str>::from("legal_next"), Arc::<str>::from(capability.as_str())));
        }
        Diagnostic::new(
            "provider.capability_unsupported",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Error,
            self.to_string(),
        )
        .with_fields(fields)
    }
}

impl fmt::Display for FailedCapabilityAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "provider {} ({}) does not declare {}",
            self.id, self.name, self.operation
        )?;
        match self.legal_next() {
            Some(capability) => write!(formatter, "; a legal next request is {capability}"),
            None => formatter.write_str("; the declaration supports nothing"),
        }
    }
}

impl std::error::Error for FailedCapabilityAttempt {}

/// Refusal from [`ProviderDeclaration::compile_verified`].
#[derive(Clone, Debug, Eq, PartialEq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum ProviderCapabilityRefusal {
    /// The declaration does not list the capability.
    #[error("{0}")]
    Undeclared(FailedCapabilityAttempt),
    /// The declaration lists the capability and the body does not supply it.
    #[error("{0}")]
    Unanswered(UnansweredCapability),
}

/// Answer the body returned for one capability call.
///
/// The probe produced this value. No density was evaluated and no model was fit.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CapabilityAnswer {
    id: ProviderId,
    name: Arc<str>,
    capability: ProviderCapability,
}

impl CapabilityAnswer {
    /// Provider whose body answered.
    #[must_use]
    pub const fn provider(&self) -> ProviderId {
        self.id
    }

    /// Provider name. This is a label, not the answer.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        &self.name
    }

    /// Capability the body supplied.
    #[must_use]
    pub const fn capability(&self) -> ProviderCapability {
        self.capability
    }
}

/// Probe whose call the body did not answer.
///
/// The declaration may list the capability. That list is not this call. No
/// density was evaluated and no model was fit.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct UnansweredCapability {
    id: ProviderId,
    name: Arc<str>,
    capability: ProviderCapability,
}

impl UnansweredCapability {
    /// Provider that was asked.
    #[must_use]
    pub const fn provider(&self) -> ProviderId {
        self.id
    }

    /// Provider name. The answer is not parsed from this string.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        &self.name
    }

    /// Capability the probe asked for and the body did not supply.
    #[must_use]
    pub const fn capability(&self) -> ProviderCapability {
        self.capability
    }
}

impl fmt::Display for UnansweredCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "provider {} ({}) does not answer {}",
            self.id, self.name, self.capability
        )
    }
}

impl std::error::Error for UnansweredCapability {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        CapabilityRequest, CapabilitySet, FailedCapabilityAttempt, ProviderCapability,
        ProviderCapabilityRefusal, ProviderDeclaration, ProviderDeclarationError, ProviderId,
    };
    use crate::diagnostic::{DiagnosticKind, DiagnosticSeverity};

    fn declaration(
        id: u64,
        name: &str,
        capabilities: impl IntoIterator<Item = ProviderCapability>,
    ) -> ProviderDeclaration {
        ProviderDeclaration::new(ProviderId::from_raw(id), name, capabilities).unwrap()
    }

    #[test]
    fn empty_declaration_supports_nothing() {
        let declaration = declaration(3, "sampling", []);
        assert!(declaration.capabilities().is_empty());
        assert_eq!(declaration.capabilities().len(), 0);
        assert_eq!(declaration.capabilities(), CapabilitySet::empty());
        for capability in ProviderCapability::ALL {
            assert!(!declaration.supports(capability));
            let failed = declaration.compile(CapabilityRequest::new(capability)).unwrap_err();
            assert_eq!(failed.operation(), capability);
            assert_eq!(failed.legal_next(), None);
            assert!(failed.declared().is_empty());
        }
    }

    #[test]
    fn declaration_does_not_answer_the_probe() {
        let declared = declaration(
            8,
            "density",
            [
                ProviderCapability::Density,
                ProviderCapability::DifferentiableLogDensity,
                ProviderCapability::Updating,
            ],
        );
        let log_density = CapabilityRequest::new(ProviderCapability::DifferentiableLogDensity);
        assert_eq!(
            declared.compile(log_density).unwrap().capability(),
            ProviderCapability::DifferentiableLogDensity
        );
        let unanswered = declared.probe(log_density).unwrap_err();
        assert_eq!(unanswered.provider(), ProviderId::from_raw(8));
        assert_eq!(unanswered.provider_name(), "density");
        assert_eq!(unanswered.capability(), ProviderCapability::DifferentiableLogDensity);
        assert_eq!(
            unanswered.to_string(),
            "provider P8 (density) does not answer differentiable_log_density"
        );
        let update = CapabilityRequest::new(ProviderCapability::Updating);
        assert_eq!(declared.probe(update).unwrap_err().capability(), ProviderCapability::Updating);

        let answering = declared.clone().answering([
            ProviderCapability::DifferentiableLogDensity,
            ProviderCapability::Updating,
        ]);
        let answer = answering.probe(log_density).unwrap();
        assert_eq!(answer.provider(), ProviderId::from_raw(8));
        assert_eq!(answer.provider_name(), "density");
        assert_eq!(answer.capability(), ProviderCapability::DifferentiableLogDensity);
        assert_eq!(answering.probe(update).unwrap().capability(), ProviderCapability::Updating);
        let density = CapabilityRequest::new(ProviderCapability::Density);
        assert!(answering.probe(density).is_err());
        assert!(answering.supports(ProviderCapability::Density));
        assert_ne!(declared, answering);
    }

    #[test]
    fn verified_compile_requires_the_body_to_answer_the_declaration() {
        let declared =
            declaration(9, "density", [ProviderCapability::Density, ProviderCapability::Sampling]);
        let density = CapabilityRequest::new(ProviderCapability::Density);
        let sampling = CapabilityRequest::new(ProviderCapability::Sampling);
        let updating = CapabilityRequest::new(ProviderCapability::Updating);
        // Declared only: the unverified path compiles, the verified one refuses.
        assert!(declared.compile(density).is_ok());
        match declared.compile_verified(density).unwrap_err() {
            ProviderCapabilityRefusal::Unanswered(unanswered) => {
                assert_eq!(unanswered.capability(), ProviderCapability::Density);
            }
            other => panic!("expected an unanswered capability, got {other:?}"),
        }
        assert_eq!(
            declared.declared_but_unanswered(),
            CapabilitySet::from_capabilities([
                ProviderCapability::Density,
                ProviderCapability::Sampling,
            ])
        );

        // Body supplies density and an undeclared update.
        let partial =
            declared.clone().answering([ProviderCapability::Density, ProviderCapability::Updating]);
        assert_eq!(
            partial.compile_verified(density).unwrap().capability(),
            ProviderCapability::Density
        );
        assert!(matches!(
            partial.compile_verified(sampling),
            Err(ProviderCapabilityRefusal::Unanswered(_))
        ));
        // Supplying an undeclared call does not declare it.
        assert!(matches!(
            partial.compile_verified(updating),
            Err(ProviderCapabilityRefusal::Undeclared(_))
        ));
        assert_eq!(
            partial.answered_but_undeclared(),
            CapabilitySet::from_capabilities([ProviderCapability::Updating])
        );
        assert_eq!(
            partial.declared_but_unanswered(),
            CapabilitySet::from_capabilities([ProviderCapability::Sampling])
        );

        let full = declared.answering([ProviderCapability::Density, ProviderCapability::Sampling]);
        assert!(full.declared_but_unanswered().is_empty());
        assert!(full.answered_but_undeclared().is_empty());
    }

    #[test]
    fn only_declared_capabilities_compile() {
        let declaration = declaration(4, "density", [ProviderCapability::Density]);
        for capability in ProviderCapability::ALL {
            let result = declaration.compile(CapabilityRequest::new(capability));
            if capability == ProviderCapability::Density {
                let compiled = result.unwrap();
                assert_eq!(compiled.provider(), ProviderId::from_raw(4));
                assert_eq!(compiled.provider_name(), "density");
                assert_eq!(compiled.capability(), ProviderCapability::Density);
            } else {
                let failed = result.unwrap_err();
                assert_eq!(failed.operation(), capability);
                assert_eq!(failed.legal_next(), Some(ProviderCapability::Density));
            }
        }
    }

    #[test]
    fn missing_capability_is_a_failed_attempt() {
        let declaration = declaration(
            1,
            "gaussian",
            [ProviderCapability::Density, ProviderCapability::StateTransition],
        );
        let failed = declaration
            .compile(CapabilityRequest::new(ProviderCapability::EstimatingEquations))
            .unwrap_err();
        assert_eq!(failed.provider(), ProviderId::from_raw(1));
        assert_eq!(failed.provider_name(), "gaussian");
        assert_eq!(failed.operation(), ProviderCapability::EstimatingEquations);
        assert_eq!(failed.legal_next(), Some(ProviderCapability::Density));
        assert_eq!(
            failed.declared(),
            CapabilitySet::from_capabilities([
                ProviderCapability::Density,
                ProviderCapability::StateTransition,
            ])
        );
        let next = failed.legal_next().unwrap();
        let compiled = declaration.compile(CapabilityRequest::new(next)).unwrap();
        assert_eq!(compiled.capability(), ProviderCapability::Density);
        assert_eq!(compiled.provider(), failed.provider());
    }

    #[test]
    fn diagnostic_names_the_provider_and_the_legal_next_request() {
        let declaration = declaration(1, "gaussian", [ProviderCapability::Density]);
        let failed: FailedCapabilityAttempt = declaration
            .compile(CapabilityRequest::new(ProviderCapability::EstimatingEquations))
            .unwrap_err();
        assert_eq!(
            failed.to_string(),
            "provider P1 (gaussian) does not declare estimating_equations; \
             a legal next request is density"
        );
        let diagnostic = failed.diagnostic();
        assert_eq!(diagnostic.code.as_ref(), "provider.capability_unsupported");
        assert_eq!(diagnostic.kind, DiagnosticKind::Scientific);
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
        assert_eq!(diagnostic.message.as_ref(), failed.to_string());
        assert_eq!(
            diagnostic.fields.as_ref(),
            &[
                (Arc::from("provider"), Arc::from("P1")),
                (Arc::from("provider_name"), Arc::from("gaussian")),
                (Arc::from("operation"), Arc::from("estimating_equations")),
                (Arc::from("legal_next"), Arc::from("density")),
            ]
        );
    }

    #[test]
    fn legal_next_request_is_not_guessed_from_the_name() {
        let named_sampling = declaration(5, "sampling", [ProviderCapability::Prediction]);
        let failed = named_sampling
            .compile(CapabilityRequest::new(ProviderCapability::Sampling))
            .unwrap_err();
        assert_eq!(
            ProviderCapability::from_name(named_sampling.name()),
            Some(ProviderCapability::Sampling)
        );
        assert_eq!(failed.legal_next(), Some(ProviderCapability::Prediction));
        assert_ne!(failed.legal_next(), ProviderCapability::from_name(named_sampling.name()));

        let empty = declaration(9, "sampling", []);
        let failed =
            empty.compile(CapabilityRequest::new(ProviderCapability::Sampling)).unwrap_err();
        assert_eq!(
            failed.to_string(),
            "provider P9 (sampling) does not declare sampling; the declaration supports nothing"
        );
        assert_eq!(failed.legal_next(), None);
        let diagnostic = failed.diagnostic();
        assert!(diagnostic.fields.iter().all(|(key, _)| key.as_ref() != "legal_next"));
        assert_eq!(
            diagnostic.fields.as_ref(),
            &[
                (Arc::from("provider"), Arc::from("P9")),
                (Arc::from("provider_name"), Arc::from("sampling")),
                (Arc::from("operation"), Arc::from("sampling")),
            ]
        );
    }

    #[test]
    fn same_name_and_different_sets_stay_different() {
        let sampling = declaration(1, "gaussian", [ProviderCapability::Sampling]);
        let density = declaration(2, "gaussian", [ProviderCapability::Density]);
        assert_eq!(sampling.name(), density.name());
        assert_ne!(sampling.id(), density.id());
        assert_ne!(sampling.capabilities(), density.capabilities());
        assert_ne!(sampling, density);
        assert!(sampling.supports(ProviderCapability::Sampling));
        assert!(!sampling.supports(ProviderCapability::Density));
        assert!(density.supports(ProviderCapability::Density));
        assert!(!density.supports(ProviderCapability::Sampling));

        let same_id_different_set = declaration(1, "gaussian", [ProviderCapability::Density]);
        assert_eq!(sampling.id(), same_id_different_set.id());
        assert_eq!(sampling.name(), same_id_different_set.name());
        assert_ne!(sampling, same_id_different_set);
    }

    #[test]
    fn capability_names_match_exactly() {
        for capability in ProviderCapability::ALL {
            assert_eq!(ProviderCapability::from_name(capability.as_str()), Some(capability));
            assert_eq!(capability.as_str(), capability.to_string());
        }
        for name in [
            "sample",
            "Sample",
            "SAMPLING",
            "log_density",
            "predict",
            "state_transitions",
            "gaussian",
            "density ",
            "",
        ] {
            assert_eq!(ProviderCapability::from_name(name), None);
        }
    }

    #[test]
    fn capability_set_is_unordered_and_deduplicated() {
        assert_eq!(
            CapabilitySet::from_capabilities([
                ProviderCapability::Sampling,
                ProviderCapability::Sampling,
            ]),
            CapabilitySet::from_capabilities([ProviderCapability::Sampling])
        );
        assert_eq!(CapabilitySet::from_capabilities([ProviderCapability::Sampling]).len(), 1);
        let forward = CapabilitySet::from_capabilities([
            ProviderCapability::StateTransition,
            ProviderCapability::Prediction,
        ]);
        let reverse = CapabilitySet::from_capabilities([
            ProviderCapability::Prediction,
            ProviderCapability::StateTransition,
        ]);
        assert_eq!(forward, reverse);
        assert_eq!(forward.legal_next(), Some(ProviderCapability::Prediction));
        assert_eq!(CapabilitySet::empty().legal_next(), None);
        assert_eq!(CapabilitySet::from_capabilities([]), CapabilitySet::empty());
    }

    #[test]
    fn blank_provider_name_is_refused() {
        let err = ProviderDeclaration::new(ProviderId::from_raw(1), " \t", []).unwrap_err();
        assert_eq!(err, ProviderDeclarationError::BlankName);
        assert_eq!(err.to_string(), "provider name must be non-blank");
    }
}
