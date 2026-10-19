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
//! Two declarations with the same name and different sets are different values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use crate::diagnostic::{Diagnostic, DiagnosticKind, DiagnosticSeverity};
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

impl ProviderCapability {
    /// Every capability, in the order a legal next request is chosen.
    pub const ALL: [Self; 9] = [
        Self::Prediction,
        Self::Density,
        Self::Sampling,
        Self::DifferentiableLogDensity,
        Self::EstimatingEquations,
        Self::InfluenceContributions,
        Self::Updating,
        Self::ExactIntegration,
        Self::StateTransition,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prediction => "prediction",
            Self::Density => "density",
            Self::Sampling => "sampling",
            Self::DifferentiableLogDensity => "differentiable_log_density",
            Self::EstimatingEquations => "estimating_equations",
            Self::InfluenceContributions => "influence_contributions",
            Self::Updating => "updating",
            Self::ExactIntegration => "exact_integration",
            Self::StateTransition => "state_transition",
        }
    }

    /// Capability with this exact `snake_case` name.
    ///
    /// Comparison is exact. A nearby word, a different case, or a provider name
    /// is not a capability.
    #[must_use = "an unrecognized name is not a capability"]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "prediction" => Self::Prediction,
            "density" => Self::Density,
            "sampling" => Self::Sampling,
            "differentiable_log_density" => Self::DifferentiableLogDensity,
            "estimating_equations" => Self::EstimatingEquations,
            "influence_contributions" => Self::InfluenceContributions,
            "updating" => Self::Updating,
            "exact_integration" => Self::ExactIntegration,
            "state_transition" => Self::StateTransition,
            _ => return None,
        })
    }

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

/// One provider and the capabilities it actually has.
///
/// Equality uses the id, the name, and the capability set. The same name with
/// a different set is a different declaration.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ProviderDeclaration {
    id: ProviderId,
    name: Arc<str>,
    capabilities: CapabilitySet,
}

impl ProviderDeclaration {
    /// Declaration of provider `id` named `name` with exactly `capabilities`.
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
        Ok(Self { id, name, capabilities: CapabilitySet::from_capabilities(capabilities) })
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
    /// The check is the whole function. It does not sample, evaluate a density,
    /// integrate, update, or fit.
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        CapabilityRequest, CapabilitySet, FailedCapabilityAttempt, ProviderCapability,
        ProviderDeclaration, ProviderDeclarationError, ProviderId,
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
