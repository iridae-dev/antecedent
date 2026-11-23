//! Registered mechanism specifications, trust and resolution.

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{CanonicalWriter, blake3_hex};
use serde::{Deserialize, Serialize};

use crate::nonlinear_equilibrium::EquilibriumError;

/// Maximum declared dependencies of one implementation.
pub const MAX_EXTERNAL_DEPENDENCIES: usize = 256;
/// Maximum inputs or parameters of one external mechanism.
pub const MAX_EXTERNAL_ARITY: usize = 1024;
const MAX_TEXT: usize = 256;

/// Standing of a mechanism or system in evidence and claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MechanismTrust {
    /// Every mechanism is a built-in, serializable library mechanism.
    Native,
    /// At least one mechanism is an external implementation verified against its recorded
    /// identity. This is never upgraded to [`Self::Native`].
    VerifiedExternal,
}
impl MechanismTrust {
    /// Stable wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::VerifiedExternal => "verified_external",
        }
    }
}

/// Typed kind of an external-mechanism refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalRefusalKind {
    /// A declaration, signature or parameter schema is malformed or incomplete.
    DeclarationInvalid,
    /// The record names an external mechanism and the caller supplied no resolver.
    ResolverMissing,
    /// The resolver has no implementation for the recorded identity.
    Unavailable,
    /// The resolved implementation differs from the recorded identity.
    Changed,
    /// The caller's resolver itself failed (raised) while resolving.
    ResolverFailed,
}

/// A typed refusal carrying a registered reason code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalMechanismRefusal {
    /// Which refusal.
    pub kind: ExternalRefusalKind,
    /// Human-readable detail naming the object and the field that decides.
    pub detail: String,
}
impl ExternalMechanismRefusal {
    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self.kind {
            ExternalRefusalKind::DeclarationInvalid => {
                antecedent_core::reason_code!("mechanism_declaration_invalid")
            }
            ExternalRefusalKind::ResolverMissing => {
                antecedent_core::reason_code!("mechanism_resolver_missing")
            }
            ExternalRefusalKind::Unavailable => {
                antecedent_core::reason_code!("mechanism_implementation_unavailable")
            }
            ExternalRefusalKind::Changed => {
                antecedent_core::reason_code!("mechanism_implementation_changed")
            }
            ExternalRefusalKind::ResolverFailed => {
                antecedent_core::reason_code!("mechanism_resolver_failed")
            }
        }
    }
    /// Stable detail identifier of the refusal, as recorded in the capability record.
    #[must_use]
    pub const fn detail_id(&self) -> &'static str {
        match self.kind {
            ExternalRefusalKind::DeclarationInvalid => {
                antecedent_core::reason_details::external_mechanism::DECLARATION_INVALID
            }
            ExternalRefusalKind::ResolverMissing => {
                antecedent_core::reason_details::external_mechanism::RESOLVER_MISSING
            }
            ExternalRefusalKind::Unavailable => {
                antecedent_core::reason_details::external_mechanism::IMPLEMENTATION_UNAVAILABLE
            }
            ExternalRefusalKind::Changed => {
                antecedent_core::reason_details::external_mechanism::IMPLEMENTATION_CHANGED
            }
            ExternalRefusalKind::ResolverFailed => {
                antecedent_core::reason_details::external_mechanism::RESOLVER_FAILED
            }
        }
    }
    pub(super) fn declaration(detail: impl Into<String>) -> Self {
        Self { kind: ExternalRefusalKind::DeclarationInvalid, detail: detail.into() }
    }
    pub(super) fn missing(spec: &ExternalMechanismSpec) -> Self {
        Self {
            kind: ExternalRefusalKind::ResolverMissing,
            detail: format!(
                "the record names external mechanism `{}` version `{}` (content digest {}); supply a trusted resolver that maps it to an implementation",
                spec.implementation_id, spec.version, spec.content_digest
            ),
        }
    }
    pub(super) fn failed(spec: &ExternalMechanismSpec, failure: &str) -> Self {
        Self {
            kind: ExternalRefusalKind::ResolverFailed,
            detail: format!(
                "the trusted resolver failed while resolving `{}` version `{}`: {failure}; this is a resolver fault, not an absent implementation",
                spec.implementation_id, spec.version
            ),
        }
    }
    pub(super) fn unavailable(spec: &ExternalMechanismSpec) -> Self {
        Self {
            kind: ExternalRefusalKind::Unavailable,
            detail: format!(
                "the resolver has no implementation of `{}` version `{}` (content digest {}); register exactly that implementation",
                spec.implementation_id, spec.version, spec.content_digest
            ),
        }
    }
    fn changed(spec: &ExternalMechanismSpec, field: &str, recorded: &str, resolved: &str) -> Self {
        Self {
            kind: ExternalRefusalKind::Changed,
            detail: format!(
                "the resolved implementation of `{}` differs from the record in {field} (recorded {recorded}, resolved {resolved}); independent replay is refused",
                spec.implementation_id
            ),
        }
    }
}
impl std::fmt::Display for ExternalMechanismRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}: {}", antecedent_core::reason_code::PREFIX, self.code(), self.detail)
    }
}
impl std::error::Error for ExternalMechanismRefusal {}
impl From<ExternalMechanismRefusal> for EquilibriumError {
    fn from(refusal: ExternalMechanismRefusal) -> Self {
        Self::Invalid(refusal.to_string())
    }
}

/// A declared dependency of an implementation, bound by version and digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalDependency {
    /// Dependency name.
    pub name: String,
    /// Dependency version.
    pub version: String,
    /// Dependency content digest.
    pub digest: String,
}

/// Finite-difference scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferenceScheme {
    /// `(f(x+h)-f(x))/h`; first-order truncation error.
    Forward,
    /// `(f(x+h)-f(x-h))/(2h)`; second-order truncation error.
    Central,
}

/// How the solver obtains input derivatives of an external mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DerivativeDeclaration {
    /// The implementation supplies an analytic input gradient (and optionally a
    /// parameter gradient); a missing gradient is a refusal, not a silent difference.
    Analytic,
    /// Derivatives are finite differences with the declared relative step
    /// `h_j = step * max(1, |x_j|)`.
    FiniteDifference {
        /// Relative step in `(0, 1]`.
        step: f64,
        /// Difference scheme.
        scheme: DifferenceScheme,
    },
}

/// The portable identity and declarations of an external mechanism. Contains no code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalMechanismSpec {
    /// Versioned implementation identifier (namespace-qualified name).
    pub implementation_id: String,
    /// Implementation version.
    pub version: String,
    /// BLAKE3 digest (64 lowercase hex characters) of the implementation's content.
    pub content_digest: String,
    /// Declared dependencies.
    pub dependencies: Vec<ExternalDependency>,
    /// Endogenous coordinates read, in the implementation's input order. The output is one
    /// finite scalar.
    pub reads: Vec<usize>,
    /// Parameter schema: ordered parameter names. Values live in the fitted system.
    pub parameter_names: Vec<String>,
    /// Derivative declaration.
    pub derivative: DerivativeDeclaration,
    /// Optional global bounds on `|d output / d input_k|` over all of R^k, in input order.
    /// Present bounds are caller declarations; replay verifies they equal the record, not that
    /// they are true. Absent bounds support no contraction or density.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_lipschitz: Option<Vec<f64>>,
}

pub(super) fn text_ok(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= MAX_TEXT && !value.chars().any(char::is_control)
}
fn digest_ok(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

impl ExternalMechanismSpec {
    /// Build a specification for `content`, computing its content digest.
    #[must_use]
    pub fn for_content(
        implementation_id: &str,
        version: &str,
        content: &[u8],
        reads: Vec<usize>,
        parameter_names: Vec<String>,
        derivative: DerivativeDeclaration,
    ) -> Self {
        Self {
            implementation_id: implementation_id.into(),
            version: version.into(),
            content_digest: blake3_hex(content),
            dependencies: Vec::new(),
            reads,
            parameter_names,
            derivative,
            input_lipschitz: None,
        }
    }

    /// Check the declarations without resolving anything.
    pub fn validate(&self) -> Result<(), ExternalMechanismRefusal> {
        let bad = |what: &str| Err(ExternalMechanismRefusal::declaration(what));
        if !text_ok(&self.implementation_id) || !text_ok(&self.version) {
            return bad("implementation id and version must be nonblank text of at most 256 bytes");
        }
        if !digest_ok(&self.content_digest) {
            return bad("content digest must be 64 lowercase hex characters");
        }
        if self.dependencies.len() > MAX_EXTERNAL_DEPENDENCIES {
            return bad("too many declared dependencies");
        }
        let mut names = std::collections::BTreeSet::new();
        for d in &self.dependencies {
            if !text_ok(&d.name) || !text_ok(&d.version) || !text_ok(&d.digest) {
                return bad("a dependency needs a name, version and digest");
            }
            if !names.insert(d.name.as_str()) {
                return bad("duplicate dependency name");
            }
        }
        if self.reads.len() > MAX_EXTERNAL_ARITY || self.parameter_names.len() > MAX_EXTERNAL_ARITY
        {
            return bad("input or parameter arity limit");
        }
        if self.reads.iter().enumerate().any(|(i, r)| self.reads[..i].contains(r)) {
            return bad("duplicate input coordinate");
        }
        let mut params = std::collections::BTreeSet::new();
        if self.parameter_names.iter().any(|p| !text_ok(p) || !params.insert(p.as_str())) {
            return bad("parameter names must be distinct nonblank text");
        }
        if let DerivativeDeclaration::FiniteDifference { step, .. } = self.derivative
            && !(step.is_finite() && step > 0.0 && step <= 1.0)
        {
            return bad("finite-difference step must lie in (0, 1]");
        }
        if let Some(bounds) = &self.input_lipschitz
            && (bounds.len() != self.reads.len()
                || bounds.iter().any(|b| !b.is_finite() || *b < 0.0))
        {
            return bad("input Lipschitz bounds need one finite nonnegative bound per input");
        }
        Ok(())
    }

    /// Stable identity over every declared field. Independent of dependency listing order.
    #[must_use]
    pub fn identity(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.external_cyclic_mechanism.identity.v1");
        w.text(&self.implementation_id).text(&self.version).text(&self.content_digest);
        let mut deps: Vec<_> = self.dependencies.iter().collect();
        deps.sort_by(|a, b| a.name.cmp(&b.name));
        w.word(deps.len() as u64);
        for d in deps {
            w.text(&d.name).text(&d.version).text(&d.digest);
        }
        w.word(self.reads.len() as u64);
        for &r in &self.reads {
            w.word(r as u64);
        }
        w.texts(&self.parameter_names);
        match self.derivative {
            DerivativeDeclaration::Analytic => {
                w.word(0);
            }
            DerivativeDeclaration::FiniteDifference { step, scheme } => {
                w.word(1).real(step).word(u64::from(scheme == DifferenceScheme::Central));
            }
        }
        w.opt_word(self.input_lipschitz.as_ref().map(|b| b.len() as u64));
        if let Some(bounds) = &self.input_lipschitz {
            w.reals(bounds);
        }
        w.finish_hex()
    }
}

/// An implementation of a registered mechanism. Process-local: its code is never serialized.
pub trait ExternalMechanismImplementation: Send + Sync {
    /// The specification this implementation realizes.
    fn spec(&self) -> ExternalMechanismSpec;
    /// The implementation's content (source, bytecode or any canonical byte description);
    /// its BLAKE3 digest must equal `spec().content_digest`.
    fn content(&self) -> &[u8];
    /// Scalar value at `inputs` (in `spec().reads` order) and `parameters`.
    fn value(&self, inputs: &[f64], parameters: &[f64]) -> Result<f64, EquilibriumError>;
    /// Analytic derivatives with respect to the inputs; `None` when not supplied.
    fn input_gradient(
        &self,
        _inputs: &[f64],
        _parameters: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        Ok(None)
    }
    /// Analytic derivatives of the input gradient with respect to the parameters, indexed
    /// `[parameter][input]` (the mixed second derivatives); `None` when not supplied. With it
    /// the score of the log-determinant term is analytic.
    fn input_gradient_parameter_derivative(
        &self,
        _inputs: &[f64],
        _parameters: &[f64],
    ) -> Result<Option<Vec<Vec<f64>>>, EquilibriumError> {
        Ok(None)
    }
    /// Analytic derivatives with respect to the parameters; `None` when not supplied.
    fn parameter_gradient(
        &self,
        _inputs: &[f64],
        _parameters: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        Ok(None)
    }
}

/// Caller-supplied map from a recorded identity to an implementation. The caller's act of
/// supplying a resolver is the act of trust; the library verifies the result against the
/// record before running it and never loads anything the record names.
pub trait TrustedMechanismResolver: Send + Sync {
    /// The implementation for `recorded`, or `None` when the caller has none.
    fn resolve(
        &self,
        recorded: &ExternalMechanismSpec,
    ) -> Option<Arc<dyn ExternalMechanismImplementation>>;
    /// Like [`Self::resolve`], but a resolver that itself fails reports the failure instead of
    /// answering `None`, so a fault is never mistaken for an absent implementation.
    fn resolve_checked(
        &self,
        recorded: &ExternalMechanismSpec,
    ) -> Result<Option<Arc<dyn ExternalMechanismImplementation>>, String> {
        Ok(self.resolve(recorded))
    }
}

/// An in-memory resolver keyed by implementation id and version.
#[derive(Default)]
pub struct MechanismRegistry {
    entries: BTreeMap<(String, String), Arc<dyn ExternalMechanismImplementation>>,
}
impl MechanismRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Register an implementation after checking its declarations and that its content
    /// digest equals its specification.
    pub fn register(
        &mut self,
        implementation: Arc<dyn ExternalMechanismImplementation>,
    ) -> Result<(), ExternalMechanismRefusal> {
        let spec = implementation.spec();
        spec.validate()?;
        if blake3_hex(implementation.content()) != spec.content_digest {
            return Err(ExternalMechanismRefusal::declaration(
                "the implementation's content does not hash to its declared content digest",
            ));
        }
        let key = (spec.implementation_id.clone(), spec.version.clone());
        if self.entries.contains_key(&key) {
            return Err(ExternalMechanismRefusal::declaration(
                "an implementation with this id and version is already registered",
            ));
        }
        self.entries.insert(key, implementation);
        Ok(())
    }
    /// Number of registered implementations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether nothing is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
impl TrustedMechanismResolver for MechanismRegistry {
    fn resolve(
        &self,
        recorded: &ExternalMechanismSpec,
    ) -> Option<Arc<dyn ExternalMechanismImplementation>> {
        self.entries.get(&(recorded.implementation_id.clone(), recorded.version.clone())).cloned()
    }
}

/// Verify that a resolved implementation equals the recorded identity.
pub fn verify_resolution(
    recorded: &ExternalMechanismSpec,
    implementation: &dyn ExternalMechanismImplementation,
) -> Result<(), ExternalMechanismRefusal> {
    let resolved = implementation.spec();
    if resolved.implementation_id != recorded.implementation_id {
        return Err(ExternalMechanismRefusal::changed(
            recorded,
            "implementation id",
            &recorded.implementation_id,
            &resolved.implementation_id,
        ));
    }
    if resolved.version != recorded.version {
        return Err(ExternalMechanismRefusal::changed(
            recorded,
            "version",
            &recorded.version,
            &resolved.version,
        ));
    }
    let actual = blake3_hex(implementation.content());
    if actual != recorded.content_digest || resolved.content_digest != recorded.content_digest {
        return Err(ExternalMechanismRefusal::changed(
            recorded,
            "content digest",
            &recorded.content_digest,
            &actual,
        ));
    }
    let sorted = |d: &[ExternalDependency]| {
        let mut v = d.to_vec();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    };
    if sorted(&resolved.dependencies) != sorted(&recorded.dependencies) {
        return Err(ExternalMechanismRefusal::changed(
            recorded,
            "declared dependencies",
            &format!("{} dependencies", recorded.dependencies.len()),
            &format!("{} dependencies", resolved.dependencies.len()),
        ));
    }
    if resolved.identity() != recorded.identity() {
        return Err(ExternalMechanismRefusal::changed(
            recorded,
            "declared signature, parameter schema, derivative or Lipschitz declaration",
            &recorded.identity(),
            &resolved.identity(),
        ));
    }
    Ok(())
}
