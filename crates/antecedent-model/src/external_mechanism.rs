// SPDX-License-Identifier: MIT OR Apache-2.0
//! Registered external cyclic mechanisms that can take part in portable scientific artifacts.
//!
//! A portable record never embeds executable code. It names a mechanism by a versioned
//! implementation identifier, the BLAKE3 digest of the implementation's content, the
//! declared dependencies, and the declarations a solver needs: the input signature, the
//! parameter schema, how derivatives are obtained and any global Lipschitz bounds.
//!
//! Replaying a record needs a [`TrustedMechanismResolver`] supplied by the caller. The
//! resolver maps a recorded specification to an implementation; the implementation is run
//! only after its identifier, version, content digest, dependencies and declarations equal
//! the recorded ones. No resolver, no implementation, or any difference refuses with a typed
//! reason code. A system that contains such a mechanism is `verified_external`, never native.

use crate::continuous_cyclic::{OdePolicy, OdeTrajectory, TimedClamp};
use crate::local_cyclic::{LocalCyclicMechanism, LocalCyclicModel};
use crate::nonlinear_equilibrium::{
    check_cancel, cholesky, local_positions, Clamp, Equation, EquilibriumError, EquilibriumResult,
    RegimeRow, SolvePolicy, MAX_ORDER,
};
use crate::sparse_equilibrium::{
    infinity_condition_estimate, nonnegative_spectral_upper_bound, CsrMatrix, FactorError,
    SolveEvidence, SolverBudgetCode, SparseLu,
};
use antecedent_core::{blake3_hex, CanonicalWriter, ExecutionContext};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

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
    fn declaration(detail: impl Into<String>) -> Self {
        Self { kind: ExternalRefusalKind::DeclarationInvalid, detail: detail.into() }
    }
    fn missing(spec: &ExternalMechanismSpec) -> Self {
        Self {
            kind: ExternalRefusalKind::ResolverMissing,
            detail: format!(
                "the record names external mechanism `{}` version `{}` (content digest {}); supply a trusted resolver that maps it to an implementation",
                spec.implementation_id, spec.version, spec.content_digest
            ),
        }
    }
    fn failed(spec: &ExternalMechanismSpec, failure: &str) -> Self {
        Self {
            kind: ExternalRefusalKind::ResolverFailed,
            detail: format!(
                "the trusted resolver failed while resolving `{}` version `{}`: {failure}; this is a resolver fault, not an absent implementation",
                spec.implementation_id, spec.version
            ),
        }
    }
    fn unavailable(spec: &ExternalMechanismSpec) -> Self {
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
    /// they are true. Absent bounds license no contraction or density.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_lipschitz: Option<Vec<f64>>,
}

fn text_ok(value: &str) -> bool {
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
        if let DerivativeDeclaration::FiniteDifference { step, .. } = self.derivative {
            if !(step.is_finite() && step > 0.0 && step <= 1.0) {
                return bad("finite-difference step must lie in (0, 1]");
            }
        }
        if let Some(bounds) = &self.input_lipschitz {
            if bounds.len() != self.reads.len() || bounds.iter().any(|b| !b.is_finite() || *b < 0.0)
            {
                return bad("input Lipschitz bounds need one finite nonnegative bound per input");
            }
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

/// Identifier of one fitted scalar of an external-capable system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExternalFitParameter {
    /// An equation intercept.
    Intercept {
        /// Dense equation index.
        equation: usize,
    },
    /// A parameter of an external mechanism, by schema position.
    Parameter {
        /// Dense equation index.
        equation: usize,
        /// Position in the mechanism's parameter schema.
        parameter: usize,
    },
}

impl ExternalFitParameter {
    /// Dense equation index of the target.
    #[must_use]
    pub const fn equation(self) -> usize {
        match self {
            Self::Intercept { equation } | Self::Parameter { equation, .. } => equation,
        }
    }
}

/// One equation of an external-capable system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "origin", rename_all = "snake_case", deny_unknown_fields)]
pub enum SystemEquation {
    /// A built-in, serializable library mechanism.
    Builtin {
        /// The equation.
        equation: Equation,
    },
    /// A registered external mechanism, recorded by identity only.
    External {
        /// Stable mechanism identity, unique within the system.
        id: String,
        /// Additive intercept outside the external value.
        intercept: f64,
        /// Fitted parameter values, in schema order.
        parameters: Vec<f64>,
        /// Recorded identity and declarations.
        mechanism: ExternalMechanismSpec,
    },
}
impl SystemEquation {
    fn id(&self) -> &str {
        match self {
            Self::Builtin { equation } => &equation.id,
            Self::External { id, .. } => id,
        }
    }
}

fn invalid(message: &str) -> EquilibriumError {
    EquilibriumError::Invalid(message.into())
}

fn validate_equations(equations: &[SystemEquation]) -> Result<(), EquilibriumError> {
    let n = equations.len();
    if n == 0 {
        return Err(invalid("external system needs at least one equation"));
    }
    if n > MAX_ORDER {
        return Err(crate::sparse_equilibrium::SolverBudgetCode::ModelOrder
            .refusal(n as u64, MAX_ORDER as u64));
    }
    let mut ids = std::collections::BTreeSet::new();
    for equation in equations {
        if !ids.insert(equation.id()) {
            return Err(invalid("duplicate mechanism identity"));
        }
        match equation {
            SystemEquation::Builtin { equation } => equation.validate(n)?,
            SystemEquation::External { id, intercept, parameters, mechanism } => {
                if !text_ok(id) || !intercept.is_finite() {
                    return Err(invalid("external mechanism identity or intercept"));
                }
                mechanism.validate()?;
                if mechanism.reads.iter().any(|&r| r >= n) {
                    return Err(invalid(
                        "external mechanism reads a coordinate outside the system",
                    ));
                }
                if parameters.len() != mechanism.parameter_names.len()
                    || parameters.iter().any(|p| !p.is_finite())
                {
                    return Err(ExternalMechanismRefusal::declaration(
                        "parameter values must match the declared parameter schema and be finite",
                    )
                    .into());
                }
            }
        }
    }
    Ok(())
}

fn trust_of(equations: &[SystemEquation]) -> MechanismTrust {
    if equations.iter().any(|e| matches!(e, SystemEquation::External { .. })) {
        MechanismTrust::VerifiedExternal
    } else {
        MechanismTrust::Native
    }
}

type Implementations = Vec<Option<Arc<dyn ExternalMechanismImplementation>>>;

fn resolve_implementations(
    equations: &[SystemEquation],
    resolver: Option<&dyn TrustedMechanismResolver>,
) -> Result<Implementations, ExternalMechanismRefusal> {
    equations
        .iter()
        .map(|equation| match equation {
            SystemEquation::Builtin { .. } => Ok(None),
            SystemEquation::External { mechanism, .. } => {
                let resolver =
                    resolver.ok_or_else(|| ExternalMechanismRefusal::missing(mechanism))?;
                let implementation = resolver
                    .resolve_checked(mechanism)
                    .map_err(|failure| ExternalMechanismRefusal::failed(mechanism, &failure))?
                    .ok_or_else(|| ExternalMechanismRefusal::unavailable(mechanism))?;
                verify_resolution(mechanism, implementation.as_ref())?;
                Ok(Some(implementation))
            }
        })
        .collect()
}

/// A verified evidence record for one external mechanism of a resolved system.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalMechanismEvidence {
    /// Equation identity in the system.
    pub equation: String,
    /// Versioned implementation id.
    pub implementation_id: String,
    /// Implementation version.
    pub version: String,
    /// Content digest the resolved implementation was verified against.
    pub content_digest: String,
    /// Identity over all declarations.
    pub identity: String,
    /// Always `verified_external`.
    pub trust: MechanismTrust,
    /// `analytic`, `finite_difference_forward` or `finite_difference_central`.
    pub derivative_basis: String,
    /// `declared` when global Lipschitz bounds were declared, else `absent`.
    pub lipschitz_basis: String,
}

struct ResolvedEquation {
    id: String,
    reads: Vec<usize>,
    n: usize,
    kind: ResolvedKind,
}
enum ResolvedKind {
    Builtin(Equation),
    External {
        intercept: f64,
        parameters: Vec<f64>,
        spec: ExternalMechanismSpec,
        implementation: Arc<dyn ExternalMechanismImplementation>,
    },
}
impl ResolvedEquation {
    fn build(
        equation: &SystemEquation,
        implementation: Option<Arc<dyn ExternalMechanismImplementation>>,
        n: usize,
    ) -> Result<Self, EquilibriumError> {
        Ok(match (equation, implementation) {
            (SystemEquation::Builtin { equation }, _) => Self {
                id: equation.id.clone(),
                reads: equation.reads()?,
                n,
                kind: ResolvedKind::Builtin(equation.clone()),
            },
            (
                SystemEquation::External { id, intercept, parameters, mechanism },
                Some(implementation),
            ) => Self {
                id: id.clone(),
                reads: mechanism.reads.clone(),
                n,
                kind: ResolvedKind::External {
                    intercept: *intercept,
                    parameters: parameters.clone(),
                    spec: mechanism.clone(),
                    implementation,
                },
            },
            (SystemEquation::External { .. }, None) => {
                return Err(invalid("external mechanism without a resolved implementation"));
            }
        })
    }

    fn external_value(
        spec_inputs: &[f64],
        parameters: &[f64],
        implementation: &dyn ExternalMechanismImplementation,
    ) -> Result<f64, EquilibriumError> {
        let value = implementation.value(spec_inputs, parameters)?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(invalid("external mechanism returned a nonfinite value"))
        }
    }

    /// Value at inputs aligned to `reads`.
    fn value(&self, inputs: &[f64]) -> Result<f64, EquilibriumError> {
        match &self.kind {
            ResolvedKind::Builtin(equation) => {
                let mut state = vec![0.0; self.n];
                for (&r, &x) in self.reads.iter().zip(inputs) {
                    state[r] = x;
                }
                equation.structural_value(&state)
            }
            ResolvedKind::External { intercept, parameters, implementation, .. } => {
                Ok(intercept + Self::external_value(inputs, parameters, implementation.as_ref())?)
            }
        }
    }

    /// Value and derivatives with respect to the inputs, aligned to `reads`.
    fn evaluate(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), EquilibriumError> {
        if inputs.len() != self.reads.len() || inputs.iter().any(|v| !v.is_finite()) {
            return Err(invalid("external mechanism inputs"));
        }
        match &self.kind {
            ResolvedKind::Builtin(equation) => {
                let mut state = vec![0.0; self.n];
                for (&r, &x) in self.reads.iter().zip(inputs) {
                    state[r] = x;
                }
                let value = equation.structural_value(&state)?;
                let full = equation.input_derivatives(&state)?;
                Ok((value, self.reads.iter().map(|&r| full[r]).collect()))
            }
            ResolvedKind::External { intercept, parameters, spec, implementation } => {
                let value = Self::external_value(inputs, parameters, implementation.as_ref())?;
                let gradient = match spec.derivative {
                    DerivativeDeclaration::Analytic => {
                        let g = implementation.input_gradient(inputs, parameters)?.ok_or_else(
                            || {
                                EquilibriumError::from(ExternalMechanismRefusal::declaration(
                                    "the mechanism declares analytic derivatives but its implementation supplies no input gradient",
                                ))
                            },
                        )?;
                        if g.len() != inputs.len() || g.iter().any(|v| !v.is_finite()) {
                            return Err(invalid("external analytic gradient shape or nonfinite"));
                        }
                        g
                    }
                    DerivativeDeclaration::FiniteDifference { step, scheme } => {
                        let mut g = Vec::with_capacity(inputs.len());
                        for j in 0..inputs.len() {
                            let h = step * inputs[j].abs().max(1.0);
                            let mut plus = inputs.to_vec();
                            plus[j] += h;
                            let f_plus =
                                Self::external_value(&plus, parameters, implementation.as_ref())?;
                            g.push(match scheme {
                                DifferenceScheme::Forward => (f_plus - value) / h,
                                DifferenceScheme::Central => {
                                    let mut minus = inputs.to_vec();
                                    minus[j] -= h;
                                    let f_minus = Self::external_value(
                                        &minus,
                                        parameters,
                                        implementation.as_ref(),
                                    )?;
                                    (f_plus - f_minus) / (2.0 * h)
                                }
                            });
                        }
                        g
                    }
                };
                Ok((intercept + value, gradient))
            }
        }
    }

    /// Global derivative bounds over all `n` coordinates; infinity where undeclared.
    fn bounds(&self) -> Result<Vec<f64>, EquilibriumError> {
        match &self.kind {
            ResolvedKind::Builtin(equation) => equation.derivative_bounds(),
            ResolvedKind::External { spec, .. } => {
                let mut out = vec![0.0; self.n];
                for (k, &r) in self.reads.iter().enumerate() {
                    out[r] = spec.input_lipschitz.as_ref().map_or(f64::INFINITY, |b| b[k]);
                }
                Ok(out)
            }
        }
    }
}

struct Adapter {
    equation: Arc<ResolvedEquation>,
}
impl LocalCyclicMechanism for Adapter {
    fn id(&self) -> &str {
        &self.equation.id
    }
    fn reads(&self) -> &[usize] {
        &self.equation.reads
    }
    fn evaluate(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), EquilibriumError> {
        self.equation.evaluate(inputs)
    }
}

fn evidence_of(equations: &[SystemEquation]) -> Vec<ExternalMechanismEvidence> {
    equations
        .iter()
        .filter_map(|e| match e {
            SystemEquation::External { id, mechanism, .. } => Some(ExternalMechanismEvidence {
                equation: id.clone(),
                implementation_id: mechanism.implementation_id.clone(),
                version: mechanism.version.clone(),
                content_digest: mechanism.content_digest.clone(),
                identity: mechanism.identity(),
                trust: MechanismTrust::VerifiedExternal,
                derivative_basis: match mechanism.derivative {
                    DerivativeDeclaration::Analytic => "analytic".into(),
                    DerivativeDeclaration::FiniteDifference { scheme, .. } => match scheme {
                        DifferenceScheme::Forward => "finite_difference_forward".into(),
                        DifferenceScheme::Central => "finite_difference_central".into(),
                    },
                },
                lipschitz_basis: if mechanism.input_lipschitz.is_some() {
                    "declared".into()
                } else {
                    "absent".into()
                },
            }),
            SystemEquation::Builtin { .. } => None,
        })
        .collect()
}

/// Likelihood of regime rows split into the data-fit and Jacobian terms.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodTerms {
    /// Gaussian disturbance term `-0.5 (z'z + log|S| + k log 2 pi)`.
    pub data: f64,
    /// `log |det(I - J_FF)|` summed over rows.
    pub log_determinant: f64,
}
impl LikelihoodTerms {
    /// Joint equilibrium log density.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.data + self.log_determinant
    }
}

/// An additive-noise feedback system whose mechanisms are built-in or registered external.
/// Serializable and portable: it records identities, never code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalFeedbackSystem {
    /// One equation per variable.
    pub equations: Vec<SystemEquation>,
    /// Joint zero-mean Gaussian disturbance covariance, row-major.
    pub noise_covariance: Vec<f64>,
}

impl ExternalFeedbackSystem {
    /// Number of endogenous variables.
    #[must_use]
    pub fn order(&self) -> usize {
        self.equations.len()
    }
    /// Validate shapes, identities, declarations and the disturbance covariance.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        validate_equations(&self.equations)?;
        let n = self.order();
        if self.noise_covariance.len() != n * n {
            return Err(invalid("model order/covariance shape"));
        }
        cholesky(&self.noise_covariance, n)?;
        Ok(())
    }
    /// `verified_external` when any mechanism is external; native only when none is.
    #[must_use]
    pub fn trust(&self) -> MechanismTrust {
        trust_of(&self.equations)
    }
    /// The recorded identities of every external mechanism.
    #[must_use]
    pub fn external_specs(&self) -> Vec<(&str, &ExternalMechanismSpec)> {
        self.equations
            .iter()
            .filter_map(|e| match e {
                SystemEquation::External { id, mechanism, .. } => Some((id.as_str(), mechanism)),
                SystemEquation::Builtin { .. } => None,
            })
            .collect()
    }
    /// Current values of fitted scalars.
    pub fn parameter_values(
        &self,
        targets: &[ExternalFitParameter],
    ) -> Result<Vec<f64>, EquilibriumError> {
        if targets.is_empty() {
            return Err(invalid("parameter count"));
        }
        let mut out = Vec::with_capacity(targets.len());
        for (k, &target) in targets.iter().enumerate() {
            if targets[..k].contains(&target) {
                return Err(invalid("duplicate parameter"));
            }
            let equation =
                self.equations.get(target.equation()).ok_or_else(|| invalid("equation index"))?;
            out.push(match (target, equation) {
                (
                    ExternalFitParameter::Intercept { .. },
                    SystemEquation::External { intercept, .. },
                ) => *intercept,
                (ExternalFitParameter::Intercept { .. }, SystemEquation::Builtin { equation }) => {
                    equation.intercept
                }
                (
                    ExternalFitParameter::Parameter { parameter, .. },
                    SystemEquation::External { parameters, .. },
                ) => *parameters.get(parameter).ok_or_else(|| invalid("parameter index"))?,
                _ => return Err(invalid("parameter target is not a fitted scalar of this system")),
            });
        }
        Ok(out)
    }
    /// A copy with the given fitted scalars replaced.
    pub fn with_parameters(
        &self,
        targets: &[ExternalFitParameter],
        values: &[f64],
    ) -> Result<Self, EquilibriumError> {
        self.parameter_values(targets)?;
        if values.len() != targets.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("parameter vector"));
        }
        let mut out = self.clone();
        for (&target, &value) in targets.iter().zip(values) {
            match (target, &mut out.equations[target.equation()]) {
                (
                    ExternalFitParameter::Intercept { .. },
                    SystemEquation::External { intercept, .. },
                ) => *intercept = value,
                (ExternalFitParameter::Intercept { .. }, SystemEquation::Builtin { equation }) => {
                    equation.intercept = value
                }
                (
                    ExternalFitParameter::Parameter { parameter, .. },
                    SystemEquation::External { parameters, .. },
                ) => parameters[parameter] = value,
                _ => return Err(invalid("parameter target")),
            }
        }
        Ok(out)
    }

    /// Resolve and verify every external mechanism. Without an external mechanism no
    /// resolver is needed; with one, a missing resolver, a missing implementation or any
    /// difference from the record is a typed refusal and nothing runs.
    pub fn resolve(
        &self,
        resolver: Option<&dyn TrustedMechanismResolver>,
    ) -> Result<ResolvedFeedbackSystem, EquilibriumError> {
        self.validate()?;
        let implementations = resolve_implementations(&self.equations, resolver)?;
        ResolvedFeedbackSystem::from_parts(self.clone(), implementations)
    }
}

/// A feedback system whose external mechanisms were resolved and verified.
pub struct ResolvedFeedbackSystem {
    system: ExternalFeedbackSystem,
    implementations: Implementations,
    equations: Vec<Arc<ResolvedEquation>>,
    local: LocalCyclicModel,
}

/// Per-row quantities shared by the likelihood and its scores.
struct RowTerms {
    free: Vec<usize>,
    blocks: Vec<Vec<usize>>,
    gradients: Vec<Vec<f64>>,
    whitened_residual_weights: Vec<f64>,
    data: f64,
    log_determinant: f64,
}

fn log_abs_determinant(mut a: Vec<f64>, k: usize) -> Result<f64, EquilibriumError> {
    let mut sum = 0.0;
    for col in 0..k {
        let mut pivot = col;
        for r in col + 1..k {
            if a[r * k + col].abs() > a[pivot * k + col].abs() {
                pivot = r;
            }
        }
        let p = a[pivot * k + col];
        if p == 0.0 || !p.is_finite() {
            return Err(EquilibriumError::Singular);
        }
        if pivot != col {
            for c in 0..k {
                a.swap(pivot * k + c, col * k + c);
            }
        }
        sum += p.abs().ln();
        for r in col + 1..k {
            let factor = a[r * k + col] / p;
            if factor != 0.0 {
                for c in col..k {
                    a[r * k + c] -= factor * a[col * k + c];
                }
            }
        }
    }
    Ok(sum)
}

impl ResolvedFeedbackSystem {
    fn from_parts(
        system: ExternalFeedbackSystem,
        implementations: Implementations,
    ) -> Result<Self, EquilibriumError> {
        let n = system.order();
        let equations = system
            .equations
            .iter()
            .zip(&implementations)
            .map(|(e, i)| ResolvedEquation::build(e, i.clone(), n).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let local = LocalCyclicModel {
            mechanisms: equations
                .iter()
                .map(|e| {
                    Box::new(Adapter { equation: Arc::clone(e) }) as Box<dyn LocalCyclicMechanism>
                })
                .collect(),
        };
        Ok(Self { system, implementations, equations, local })
    }

    /// The portable system this resolution verified.
    #[must_use]
    pub fn system(&self) -> &ExternalFeedbackSystem {
        &self.system
    }
    /// `verified_external` when any mechanism is external; there is no route to native.
    #[must_use]
    pub fn trust(&self) -> MechanismTrust {
        self.system.trust()
    }
    /// Verified evidence of each external mechanism.
    #[must_use]
    pub fn evidence(&self) -> Vec<ExternalMechanismEvidence> {
        evidence_of(&self.system.equations)
    }
    /// The same verified implementations with other fitted scalars.
    pub fn with_parameters(
        &self,
        targets: &[ExternalFitParameter],
        values: &[f64],
    ) -> Result<Self, EquilibriumError> {
        let system = self.system.with_parameters(targets, values)?;
        Self::from_parts(system, self.implementations.clone())
    }

    /// Structural means at one state; joint noise is added by the execution.
    pub fn mechanism_values(&self, state: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        if state.len() != self.system.order() || state.iter().any(|v| !v.is_finite()) {
            return Err(invalid("state dimension or nonfinite"));
        }
        self.equations
            .iter()
            .map(|e| {
                let inputs: Vec<f64> = e.reads.iter().map(|&r| state[r]).collect();
                e.value(&inputs)
            })
            .collect()
    }

    /// Post-clamp SCC schedule.
    pub fn blocks(&self, clamps: &[Clamp]) -> Result<Vec<Vec<usize>>, EquilibriumError> {
        Ok(self.local.inspect(clamps)?.blocks)
    }

    /// Largest block infinity-norm bound of the free system from declared global
    /// derivative bounds; infinite when a bound is undeclared. Below one proves a
    /// contraction on R^k for every disturbance, conditional on the declarations.
    pub fn contraction_bound(&self, clamps: &[Clamp]) -> Result<f64, EquilibriumError> {
        let blocks = self.blocks(clamps)?;
        let bounds: Vec<Vec<f64>> =
            self.equations.iter().map(|e| e.bounds()).collect::<Result<_, _>>()?;
        let mut worst = 0.0_f64;
        for block in &blocks {
            for &i in block {
                let sum: f64 = block.iter().map(|&j| bounds[i][j]).sum();
                worst = worst.max(sum);
            }
        }
        Ok(worst)
    }

    fn reads_of(&self) -> Vec<&[usize]> {
        self.equations.iter().map(|e| e.reads.as_slice()).collect()
    }

    fn residual(
        &self,
        state: &[f64],
        noise: &[f64],
        rows: &[usize],
    ) -> Result<Vec<f64>, EquilibriumError> {
        rows.iter()
            .map(|&i| {
                let e = &self.equations[i];
                let inputs: Vec<f64> = e.reads.iter().map(|&r| state[r]).collect();
                Ok(state[i] - e.value(&inputs)? - noise[i])
            })
            .collect()
    }

    /// `I - df/dx` over `rows_of`, local to that set, from the verified mechanisms' derivatives.
    fn sparse_jacobian(
        &self,
        state: &[f64],
        rows_of: &[usize],
    ) -> Result<CsrMatrix, EquilibriumError> {
        let local = local_positions(rows_of);
        let mut rows = Vec::with_capacity(rows_of.len());
        for (a, &i) in rows_of.iter().enumerate() {
            let e = &self.equations[i];
            let inputs: Vec<f64> = e.reads.iter().map(|&r| state[r]).collect();
            let (_, gradient) = e.evaluate(&inputs)?;
            let mut row = vec![(a, 1.0)];
            for (g, &read) in e.reads.iter().enumerate() {
                if let Some(&b) = local.get(&read) {
                    row.push((b, -gradient[g]));
                }
            }
            rows.push(row);
        }
        CsrMatrix::from_rows(rows_of.len(), rows)
    }

    /// Declared global derivative-bound matrix of a block.
    fn block_bound_matrix(&self, block: &[usize]) -> Result<CsrMatrix, EquilibriumError> {
        let local = local_positions(block);
        let mut rows = Vec::with_capacity(block.len());
        for &i in block {
            let bounds = self.equations[i].bounds()?;
            rows.push(
                self.equations[i]
                    .reads
                    .iter()
                    .filter_map(|j| local.get(j).map(|&b| (b, bounds[*j])))
                    .collect(),
            );
        }
        CsrMatrix::from_rows(block.len(), rows)
    }

    /// Numerical SCC solve under the same budgets, admission, cancellation and evidence as the
    /// built-in sparse solver: Jacobian nonzeros, block size, memory, factor fill, iteration
    /// count and wall clock. Existence and uniqueness are marked only when the declared global
    /// bounds give a block contraction (row-sum or spectral bound) below one; otherwise the
    /// root is numerical and, with `require_contraction`, refused.
    #[allow(
        clippy::too_many_lines,
        reason = "one budgeted block solve retains admission, certificates and evidence together"
    )]
    pub fn solve(
        &self,
        noise: &[f64],
        clamps: &[Clamp],
        initial: &[f64],
        policy: SolvePolicy,
        ctx: &ExecutionContext,
    ) -> Result<EquilibriumResult, EquilibriumError> {
        policy.validate()?;
        let n = self.system.order();
        for v in [noise, initial] {
            if v.len() != n || v.iter().any(|x| !x.is_finite()) {
                return Err(invalid("state/noise dimension or nonfinite"));
            }
        }
        let blocks = self.blocks(clamps)?;
        let mut is_free = vec![true; n];
        for c in clamps {
            if !c.value.is_finite() {
                return Err(invalid("clamp value"));
            }
            is_free[c.variable] = false;
        }
        let free: Vec<usize> = (0..n).filter(|&i| is_free[i]).collect();
        let reads = self.reads_of();
        let jacobian_nnz: usize = free
            .iter()
            .map(|&i| reads[i].iter().filter(|&&j| j != i && is_free[j]).count() + 1)
            .sum();
        if jacobian_nnz > policy.max_nnz {
            return Err(
                SolverBudgetCode::JacobianNnz.refusal(jacobian_nnz as u64, policy.max_nnz as u64)
            );
        }
        let largest_block = blocks.iter().map(Vec::len).max().unwrap_or(0);
        if largest_block > policy.max_block_size {
            return Err(SolverBudgetCode::BlockSize
                .refusal(largest_block as u64, policy.max_block_size as u64));
        }
        let fill = policy
            .max_factor_fill
            .min(jacobian_nnz.saturating_mul(8).saturating_add(64 * largest_block));
        let bytes = u64::try_from(
            16usize
                .saturating_mul(n)
                .saturating_mul(n)
                .saturating_add(128 * n)
                .saturating_add(jacobian_nnz.saturating_mul(32))
                .saturating_add((free.len() + 1).saturating_mul(16))
                .saturating_add(fill.saturating_mul(24)),
        )
        .map_err(|_| invalid("resource size"))?;
        if bytes > policy.max_bytes {
            return Err(SolverBudgetCode::Memory.refusal(bytes, policy.max_bytes));
        }
        let _lease = ctx
            .acquire_admission(1, bytes)
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let started = Instant::now();
        let mut checks = 0usize;
        let tick = |checks: &mut usize| -> Result<(), EquilibriumError> {
            *checks += 1;
            check_cancel(ctx)?;
            if let Some(limit) = policy.max_wall_clock_ms {
                let elapsed = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                if elapsed >= limit {
                    return Err(SolverBudgetCode::WallClock.refusal(elapsed, limit));
                }
            }
            Ok(())
        };
        let mut evidence = SolveEvidence {
            free_variables: free.len(),
            block_count: blocks.len(),
            largest_block,
            jacobian_nnz,
            estimated_bytes: bytes,
            admitted_bytes: bytes,
            ..SolveEvidence::default()
        };
        let mut values = initial.to_vec();
        for c in clamps {
            values[c.variable] = c.value;
        }
        let max_abs = |v: &[f64]| {
            v.iter()
                .map(|x| if x.is_finite() { x.abs() } else { f64::INFINITY })
                .fold(0.0, f64::max)
        };
        let factor_failure = |error: FactorError, residual: f64| match error {
            FactorError::Singular => EquilibriumError::Singular,
            FactorError::NonFinite => EquilibriumError::Nonconvergence { residual },
            FactorError::FillBudget(required) => {
                SolverBudgetCode::FactorFill.refusal(required as u64, policy.max_factor_fill as u64)
            }
            FactorError::Interrupted(error) => error,
        };
        let mut total = 0usize;
        let mut all_contractive = true;
        for block in &blocks {
            tick(&mut checks)?;
            // Positive summation error is bounded conservatively before comparing to one.
            let inflation = 1.0 + (block.len() + 1) as f64 * f64::EPSILON;
            let matrix = self.block_bound_matrix(block)?;
            let row_sum = matrix.infinity_norm() * inflation;
            let (spectral, spectral_iterations) = if row_sum < 1.0 || block.len() == 1 {
                (row_sum, 0)
            } else {
                let found = nonnegative_spectral_upper_bound(&matrix, policy.spectral_iterations);
                (found.upper.min(row_sum), found.iterations)
            };
            let bound = row_sum.min(spectral);
            let retained = |x: f64| if x.is_finite() { x } else { f64::MAX };
            evidence.row_sum_bound = evidence.row_sum_bound.max(retained(row_sum));
            evidence.spectral_bound = evidence.spectral_bound.max(retained(spectral));
            evidence.spectral_iterations += spectral_iterations;
            evidence.contraction_bound = evidence.contraction_bound.max(retained(bound));
            if block.len() > 1 || reads[block[0]].contains(&block[0]) {
                evidence.cyclic_block_count += 1;
            }
            let contraction = bound < 1.0;
            all_contractive &= contraction;
            if policy.require_contraction && !contraction {
                return Err(EquilibriumError::Uncertified { bound });
            }
            let mut converged = false;
            for _ in 0..policy.max_iterations {
                tick(&mut checks)?;
                let residual = self.residual(&values, noise, block)?;
                let norm = max_abs(&residual);
                if !norm.is_finite() {
                    return Err(EquilibriumError::Nonconvergence { residual: norm });
                }
                if norm <= policy.tolerance {
                    converged = true;
                    break;
                }
                total += 1;
                if total > policy.max_total_iterations {
                    return Err(SolverBudgetCode::IterationBudget
                        .refusal(total as u64, policy.max_total_iterations as u64));
                }
                if contraction {
                    evidence.contraction_iterations += 1;
                    for (&i, &r) in block.iter().zip(&residual) {
                        values[i] -= r;
                    }
                } else {
                    evidence.newton_steps += 1;
                    let jac = self.sparse_jacobian(&values, block)?;
                    let lu =
                        SparseLu::factor(&jac, policy.max_factor_fill, &mut || tick(&mut checks))
                            .map_err(|e| factor_failure(e, norm))?;
                    evidence.factorizations += 1;
                    evidence.peak_factor_fill = evidence.peak_factor_fill.max(lu.fill());
                    let step = lu.solve(&residual);
                    if step.iter().any(|v| !v.is_finite()) {
                        return Err(EquilibriumError::Nonconvergence { residual: norm });
                    }
                    let old = values.clone();
                    let mut accepted = false;
                    let mut scale = 1.0;
                    for _ in 0..40 {
                        for (&i, &delta) in block.iter().zip(&step) {
                            values[i] = old[i] - scale * delta;
                        }
                        let next = max_abs(&self.residual(&values, noise, block)?);
                        if next.is_finite() && next < norm {
                            accepted = true;
                            break;
                        }
                        scale *= 0.5;
                    }
                    if !accepted {
                        return Err(EquilibriumError::Nonconvergence { residual: norm });
                    }
                }
            }
            let residual = max_abs(&self.residual(&values, noise, block)?);
            if !converged && residual > policy.tolerance {
                return Err(EquilibriumError::Nonconvergence { residual });
            }
        }
        let residual = max_abs(&self.residual(&values, noise, &free)?);
        if values.iter().any(|v| !v.is_finite())
            || !residual.is_finite()
            || residual > policy.tolerance
        {
            return Err(EquilibriumError::Nonconvergence { residual });
        }
        // Free-Jacobian conditioning at the returned state.
        let mut jacobian_condition_estimate = None;
        let mut factored = "not_applicable";
        if !free.is_empty() {
            let jacobian = self.sparse_jacobian(&values, &free)?;
            evidence.jacobian_diagonal_dominance_margin =
                jacobian.diagonal_dominance_margin().filter(|m| m.is_finite());
            let factors = match SparseLu::factor(&jacobian, policy.max_factor_fill, &mut || {
                tick(&mut checks)
            }) {
                Ok(lu) => {
                    SparseLu::factor(&jacobian.transpose(), policy.max_factor_fill, &mut || {
                        tick(&mut checks)
                    })
                    .map(|lut| (lu, lut))
                }
                Err(error) => Err(error),
            };
            match factors {
                Ok((lu, lut)) => {
                    evidence.factorizations += 2;
                    evidence.peak_factor_fill =
                        evidence.peak_factor_fill.max(lu.fill()).max(lut.fill());
                    jacobian_condition_estimate = infinity_condition_estimate(&jacobian, &lu, &lut);
                    factored = "factorized";
                }
                Err(FactorError::Singular) => factored = "singular",
                Err(FactorError::NonFinite) => factored = "nonfinite",
                Err(FactorError::FillBudget(_)) => factored = "factor_fill_budget",
                Err(FactorError::Interrupted(error)) => return Err(error),
            }
        }
        evidence.jacobian_nonsingular_basis = if free.is_empty() {
            "not_applicable".into()
        } else if factored != "factorized" {
            factored.into()
        } else if all_contractive {
            "contraction".into()
        } else if evidence.jacobian_diagonal_dominance_margin.is_some_and(|m| m > 0.0) {
            "strict_diagonal_dominance".into()
        } else {
            "factorization".into()
        };
        evidence.unique_root_basis = if all_contractive { "contraction" } else { "none" }.into();
        evidence.ill_conditioned =
            jacobian_condition_estimate.is_some_and(|k| k > policy.condition_warning);
        evidence.budget_checks = checks;
        Ok(EquilibriumResult {
            values,
            residual,
            iterations: total,
            globally_unique: all_contractive,
            existence_certified: all_contractive,
            numerical_converged: true,
            dynamic_stability: None,
            jacobian_condition_estimate,
            blocks,
            evidence,
        })
    }

    /// Exact additive abduction at a fully observed state.
    pub fn abduce(&self, factual: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        let means = self.mechanism_values(factual)?;
        Ok(factual.iter().zip(means).map(|(x, m)| x - m).collect())
    }

    /// Abduce once and reuse the disturbances in the intervened world.
    pub fn counterfactual(
        &self,
        factual: &[f64],
        clamps: &[Clamp],
        policy: SolvePolicy,
        ctx: &ExecutionContext,
    ) -> Result<EquilibriumResult, EquilibriumError> {
        let noise = self.abduce(factual)?;
        self.solve(&noise, clamps, factual, policy, ctx)
    }

    #[allow(
        clippy::float_cmp,
        reason = "observed hard clamps must equal the declared structural value exactly"
    )]
    fn check_row(&self, row: &RegimeRow) -> Result<Vec<usize>, EquilibriumError> {
        let n = self.system.order();
        if row.values.len() != n || row.values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("state/noise dimension or nonfinite"));
        }
        let mut seen = vec![false; n];
        for c in &row.clamps {
            if c.variable >= n || !c.value.is_finite() || seen[c.variable] {
                return Err(invalid("clamp target, value, or duplicate"));
            }
            seen[c.variable] = true;
            if row.values[c.variable] != c.value {
                return Err(invalid("observed value differs from known clamp"));
            }
        }
        Ok((0..n).filter(|&i| !seen[i]).collect())
    }

    fn row_terms(&self, row: &RegimeRow) -> Result<RowTerms, EquilibriumError> {
        let n = self.system.order();
        let free = self.check_row(row)?;
        if free.is_empty() {
            return Ok(RowTerms {
                free,
                blocks: Vec::new(),
                gradients: Vec::new(),
                whitened_residual_weights: Vec::new(),
                data: 0.0,
                log_determinant: 0.0,
            });
        }
        let bound = self.contraction_bound(&row.clamps)?;
        if bound.is_nan() || bound >= 1.0 {
            return Err(EquilibriumError::Uncertified { bound });
        }
        let k = free.len();
        let mut values = vec![0.0; n];
        let mut gradients: Vec<Vec<f64>> = vec![Vec::new(); n];
        for &i in &free {
            let e = &self.equations[i];
            let inputs: Vec<f64> = e.reads.iter().map(|&r| row.values[r]).collect();
            let (v, g) = e.evaluate(&inputs)?;
            values[i] = v;
            gradients[i] = g;
        }
        let residual: Vec<f64> = free.iter().map(|&i| row.values[i] - values[i]).collect();
        let cov: Vec<f64> = free
            .iter()
            .flat_map(|&i| free.iter().map(move |&j| self.system.noise_covariance[i * n + j]))
            .collect();
        let l = cholesky(&cov, k)?;
        let mut z = residual;
        for i in 0..k {
            for j in 0..i {
                z[i] -= l[i * k + j] * z[j];
            }
            z[i] /= l[i * k + i];
        }
        // w = Sigma^-1 r = L^-T z.
        let mut w = z.clone();
        for i in (0..k).rev() {
            for j in i + 1..k {
                w[i] -= l[j * k + i] * w[j];
            }
            w[i] /= l[i * k + i];
        }
        let noise_logdet = 2.0 * (0..k).map(|i| l[i * k + i].ln()).sum::<f64>();
        let data = -0.5
            * (z.iter().map(|x| x * x).sum::<f64>()
                + noise_logdet
                + k as f64 * (2.0 * std::f64::consts::PI).ln());
        let mut log_determinant = 0.0;
        let blocks = self.blocks(&row.clamps)?;
        for block in &blocks {
            let m = block.len();
            let mut matrix = vec![0.0; m * m];
            for (a, &i) in block.iter().enumerate() {
                matrix[a * m + a] = 1.0;
                for (g, &read) in self.equations[i].reads.iter().enumerate() {
                    if let Some(b) = block.iter().position(|&j| j == read) {
                        matrix[a * m + b] -= gradients[i][g];
                    }
                }
            }
            log_determinant += log_abs_determinant(matrix, m)?;
        }
        Ok(RowTerms {
            free,
            blocks,
            gradients,
            whitened_residual_weights: w,
            data,
            log_determinant,
        })
    }

    /// Joint equilibrium log-likelihood terms of known-regime rows. Requires a declared
    /// global contraction for every free block, so one inverse branch is never normalized as
    /// the density of a multiple-equilibrium system.
    pub fn likelihood_terms(
        &self,
        rows: &[RegimeRow],
        ctx: &ExecutionContext,
    ) -> Result<LikelihoodTerms, EquilibriumError> {
        let n = self.system.order();
        let _lease = ctx
            .acquire_admission(
                1,
                u64::try_from(64 * n * n + 256 * n).map_err(|_| invalid("resource size"))?,
            )
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let mut total = LikelihoodTerms { data: 0.0, log_determinant: 0.0 };
        for row in rows {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            let terms = self.row_terms(row)?;
            total.data += terms.data;
            total.log_determinant += terms.log_determinant;
        }
        if !total.total().is_finite() {
            return Err(invalid("nonfinite likelihood"));
        }
        Ok(total)
    }

    /// Analytic derivative of the log-determinant term `log |det(I - J)|` with respect to
    /// fitted scalars: `d/dtheta = -sum_j (M^-1)_{j,e} dJ_{e,j}/dtheta` with `M = I - J` of the
    /// equation's block, from the implementation's declared mixed derivatives. `None` when a
    /// parameter target's mechanism is not declared analytic or supplies no mixed derivative;
    /// the caller then falls back to a central difference. An intercept does not move `J`.
    pub fn determinant_term_score(
        &self,
        rows: &[RegimeRow],
        targets: &[ExternalFitParameter],
        ctx: &ExecutionContext,
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        self.system.parameter_values(targets)?;
        let mut score = vec![0.0; targets.len()];
        for row in rows {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            let terms = self.row_terms(row)?;
            for (t, target) in targets.iter().enumerate() {
                let ExternalFitParameter::Parameter { equation, parameter } = *target else {
                    continue;
                };
                if !terms.free.contains(&equation) {
                    continue;
                }
                let e = &self.equations[equation];
                let ResolvedKind::External { parameters, implementation, spec, .. } = &e.kind
                else {
                    return Err(invalid("parameter target is not external"));
                };
                if spec.derivative != DerivativeDeclaration::Analytic {
                    return Ok(None);
                }
                let inputs: Vec<f64> = e.reads.iter().map(|&r| row.values[r]).collect();
                let Some(mixed) =
                    implementation.input_gradient_parameter_derivative(&inputs, parameters)?
                else {
                    return Ok(None);
                };
                if mixed.len() != parameters.len()
                    || mixed[parameter].len() != inputs.len()
                    || mixed[parameter].iter().any(|v| !v.is_finite())
                {
                    return Err(invalid("external mixed derivative shape or nonfinite"));
                }
                let block = terms
                    .blocks
                    .iter()
                    .find(|b| b.contains(&equation))
                    .ok_or_else(|| invalid("equation outside every block"))?;
                let m = block.len();
                let mut matrix = vec![0.0; m * m];
                for (a, &i) in block.iter().enumerate() {
                    matrix[a * m + a] = 1.0;
                    for (g, &read) in self.equations[i].reads.iter().enumerate() {
                        if let Some(b) = block.iter().position(|&j| j == read) {
                            matrix[a * m + b] -= terms.gradients[i][g];
                        }
                    }
                }
                let position = block.iter().position(|&j| j == equation).unwrap_or(0);
                let mut unit = vec![0.0; m];
                unit[position] = 1.0;
                if !antecedent_kernels::gauss_jordan_solve(&mut matrix, &mut unit, m, 1) {
                    return Err(EquilibriumError::Singular);
                }
                for (g, &read) in e.reads.iter().enumerate() {
                    if let Some(b) = block.iter().position(|&j| j == read) {
                        score[t] -= unit[b] * mixed[parameter][g];
                    }
                }
            }
        }
        Ok(Some(score))
    }

    /// Analytic derivative of the Gaussian data term with respect to fitted scalars, or
    /// `None` when a target is a parameter whose implementation supplies no parameter
    /// gradient. The log-determinant term is not covered here.
    pub fn data_term_score(
        &self,
        rows: &[RegimeRow],
        targets: &[ExternalFitParameter],
        ctx: &ExecutionContext,
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        self.system.parameter_values(targets)?;
        let mut score = vec![0.0; targets.len()];
        for row in rows {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            let terms = self.row_terms(row)?;
            for (t, target) in targets.iter().enumerate() {
                match *target {
                    ExternalFitParameter::Intercept { equation } => {
                        if let Some(pos) = terms.free.iter().position(|&i| i == equation) {
                            score[t] += terms.whitened_residual_weights[pos];
                        }
                    }
                    ExternalFitParameter::Parameter { equation, parameter } => {
                        let Some(pos) = terms.free.iter().position(|&i| i == equation) else {
                            continue;
                        };
                        let e = &self.equations[equation];
                        let ResolvedKind::External { parameters, implementation, .. } = &e.kind
                        else {
                            return Err(invalid("parameter target is not external"));
                        };
                        let inputs: Vec<f64> = e.reads.iter().map(|&r| row.values[r]).collect();
                        let Some(g) = implementation.parameter_gradient(&inputs, parameters)?
                        else {
                            return Ok(None);
                        };
                        if g.len() != parameters.len() || g.iter().any(|v| !v.is_finite()) {
                            return Err(invalid("external parameter gradient shape or nonfinite"));
                        }
                        score[t] += terms.whitened_residual_weights[pos] * g[parameter];
                    }
                }
            }
        }
        Ok(Some(score))
    }
}

/// Continuous-time drift `dx_i/dt = f_i(x) - decay_i x_i` with built-in or registered
/// external mechanisms. Self-dependence is allowed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalOdeSystem {
    /// One drift mechanism per variable.
    pub equations: Vec<SystemEquation>,
    /// Additional linear decay per variable.
    pub decay: Vec<f64>,
}

/// A finite-horizon external-mechanism trajectory and its standing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalOdeTrajectory {
    /// States and step counts.
    pub trajectory: OdeTrajectory,
    /// Global Lipschitz upper bound of the drift (infinity-norm) from declared bounds; absent
    /// when any bound is undeclared. A declaration, not a measurement.
    pub drift_lipschitz: Option<f64>,
    /// `verified_external` when any mechanism is external.
    pub trust: MechanismTrust,
    /// Verified evidence of each external mechanism.
    pub mechanisms: Vec<ExternalMechanismEvidence>,
}

/// An ODE system with verified implementations.
pub struct ResolvedOdeSystem {
    system: ExternalOdeSystem,
    equations: Vec<Arc<ResolvedEquation>>,
}

impl ExternalOdeSystem {
    /// Validate shapes, identities and declarations.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        let n = self.equations.len();
        if n == 0 || n > 128 || self.decay.len() != n || self.decay.iter().any(|x| !x.is_finite()) {
            return Err(invalid("ODE shape/decay"));
        }
        validate_equations(&self.equations)
    }
    /// Resolve and verify every external mechanism, as for a static system.
    pub fn resolve(
        &self,
        resolver: Option<&dyn TrustedMechanismResolver>,
    ) -> Result<ResolvedOdeSystem, EquilibriumError> {
        self.validate()?;
        let implementations = resolve_implementations(&self.equations, resolver)?;
        let n = self.equations.len();
        let equations = self
            .equations
            .iter()
            .zip(implementations)
            .map(|(e, i)| ResolvedEquation::build(e, i, n).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ResolvedOdeSystem { system: self.clone(), equations })
    }
}

impl ResolvedOdeSystem {
    /// Declared drift at `state`.
    pub fn drift(&self, state: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        if state.len() != self.equations.len() || state.iter().any(|v| !v.is_finite()) {
            return Err(invalid("ODE state"));
        }
        self.equations
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let inputs: Vec<f64> = e.reads.iter().map(|&r| state[r]).collect();
                e.value(&inputs).map(|v| v - self.system.decay[i] * state[i])
            })
            .collect()
    }

    /// Infinity-norm global Lipschitz bound of the drift from declared bounds.
    pub fn drift_lipschitz(&self) -> Result<Option<f64>, EquilibriumError> {
        let mut worst = 0.0_f64;
        for (i, e) in self.equations.iter().enumerate() {
            let mut bounds = e.bounds()?;
            bounds[i] += self.system.decay[i].abs();
            let sum: f64 = bounds.iter().sum();
            if !sum.is_finite() {
                return Ok(None);
            }
            worst = worst.max(sum);
        }
        Ok(Some(worst))
    }

    /// Dormand-Prince trajectory with timed hard clamps. A declared Lipschitz bound caps the
    /// first proposed step at its reciprocal; error control then governs every step.
    pub fn trajectory(
        &self,
        initial: &[f64],
        times: &[f64],
        schedule: &[TimedClamp],
        policy: OdePolicy,
        ctx: &ExecutionContext,
    ) -> Result<ExternalOdeTrajectory, EquilibriumError> {
        let lipschitz = self.drift_lipschitz()?;
        let mut policy = policy;
        if let Some(l) = lipschitz.filter(|l| *l > 0.0) {
            policy.initial_step = policy.initial_step.min(1.0 / l);
        }
        let trajectory =
            crate::continuous_cyclic::integrate(initial, times, schedule, policy, ctx, |state| {
                self.drift(state)
            })?;
        Ok(ExternalOdeTrajectory {
            trajectory,
            drift_lipschitz: lipschitz,
            trust: trust_of(&self.system.equations),
            mechanisms: evidence_of(&self.system.equations),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear_equilibrium::Activation;

    /// `a * tanh(b * x)`-free affine mechanism: `p0 * x0 (+ p1)` with exact gradients.
    struct Affine {
        spec: ExternalMechanismSpec,
        content: Vec<u8>,
    }
    impl Affine {
        fn new(version: &str, content: &str, derivative: DerivativeDeclaration) -> Arc<Self> {
            let content = content.as_bytes().to_vec();
            let mut spec = ExternalMechanismSpec::for_content(
                "lab/affine",
                version,
                &content,
                vec![1],
                vec!["slope".into()],
                derivative,
            );
            spec.input_lipschitz = Some(vec![0.9]);
            Arc::new(Self { spec, content })
        }
    }
    impl ExternalMechanismImplementation for Affine {
        fn spec(&self) -> ExternalMechanismSpec {
            self.spec.clone()
        }
        fn content(&self) -> &[u8] {
            &self.content
        }
        fn value(&self, inputs: &[f64], parameters: &[f64]) -> Result<f64, EquilibriumError> {
            Ok(parameters[0] * inputs[0])
        }
        fn input_gradient(
            &self,
            _inputs: &[f64],
            parameters: &[f64],
        ) -> Result<Option<Vec<f64>>, EquilibriumError> {
            Ok(Some(vec![parameters[0]]))
        }
        fn parameter_gradient(
            &self,
            inputs: &[f64],
            _parameters: &[f64],
        ) -> Result<Option<Vec<f64>>, EquilibriumError> {
            Ok(Some(vec![inputs[0]]))
        }
    }

    fn system(spec: &ExternalMechanismSpec, slope: f64) -> ExternalFeedbackSystem {
        ExternalFeedbackSystem {
            equations: vec![
                SystemEquation::External {
                    id: "a".into(),
                    intercept: 0.0,
                    parameters: vec![slope],
                    mechanism: spec.clone(),
                },
                SystemEquation::Builtin {
                    equation: Equation {
                        id: "b".into(),
                        intercept: 0.0,
                        weights: vec![0.5, 0.0],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: vec![],
                    },
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn registered_mechanism_solves_the_closed_form_and_stays_verified_external() {
        let implementation = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic);
        let mut registry = MechanismRegistry::new();
        registry.register(implementation.clone()).unwrap();
        let system = system(&implementation.spec, 0.6);
        let resolved = system.resolve(Some(&registry)).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let (u0, u1) = (0.4, -0.3);
        let result =
            resolved.solve(&[u0, u1], &[], &[0.0, 0.0], SolvePolicy::default(), &ctx).unwrap();
        // x0 = 0.6 x1 + u0 and x1 = 0.5 x0 + u1.
        let x0 = (u0 + 0.6 * u1) / (1.0 - 0.3);
        assert!((result.values[0] - x0).abs() < 1e-9);
        assert!(result.globally_unique && result.existence_certified);
        assert_eq!(resolved.trust(), MechanismTrust::VerifiedExternal);
        assert_eq!(resolved.evidence().len(), 1);
    }

    #[test]
    fn replay_is_refused_without_a_resolver_an_implementation_or_with_a_changed_one() {
        let original = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic);
        let system = system(&original.spec, 0.6);
        let code = |r: Result<ResolvedFeedbackSystem, EquilibriumError>| match r {
            Err(EquilibriumError::Invalid(m)) => m,
            Err(other) => panic!("unexpected {other:?}"),
            Ok(_) => panic!("replay must be refused"),
        };
        assert!(code(system.resolve(None)).contains("mechanism_resolver_missing"));
        let empty = MechanismRegistry::new();
        assert!(code(system.resolve(Some(&empty))).contains("mechanism_implementation_unavailable"));
        let mut other_content = MechanismRegistry::new();
        other_content
            .register(Affine::new("1", "affine v1 edited", DerivativeDeclaration::Analytic))
            .unwrap();
        assert!(
            code(system.resolve(Some(&other_content))).contains("mechanism_implementation_changed")
        );
        let mut other_version = MechanismRegistry::new();
        other_version
            .register(Affine::new("2", "affine v1", DerivativeDeclaration::Analytic))
            .unwrap();
        assert!(code(system.resolve(Some(&other_version)))
            .contains("mechanism_implementation_unavailable"));
    }

    #[test]
    fn finite_difference_gradient_matches_the_analytic_gradient() {
        let analytic = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic);
        let difference = Affine::new(
            "1",
            "affine v1",
            DerivativeDeclaration::FiniteDifference {
                step: 1e-6,
                scheme: DifferenceScheme::Central,
            },
        );
        let ctx = ExecutionContext::for_tests(1);
        let rows: Vec<RegimeRow> = [-1.0_f64, 0.5, 1.5]
            .into_iter()
            .map(|x| RegimeRow {
                values: vec![0.6 * x + 0.1, x],
                clamps: vec![Clamp { variable: 1, value: x }],
            })
            .collect();
        let mut terms = Vec::new();
        for implementation in [analytic, difference] {
            let mut registry = MechanismRegistry::new();
            registry.register(implementation.clone()).unwrap();
            let resolved = system(&implementation.spec, 0.6).resolve(Some(&registry)).unwrap();
            terms.push(resolved.likelihood_terms(&rows, &ctx).unwrap().total());
        }
        assert!((terms[0] - terms[1]).abs() < 1e-8);
    }

    #[test]
    fn spec_identity_ignores_dependency_order_and_sees_every_declaration() {
        let base = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic).spec.clone();
        let mut a = base.clone();
        a.dependencies = vec![
            ExternalDependency { name: "x".into(), version: "1".into(), digest: "d".into() },
            ExternalDependency { name: "y".into(), version: "2".into(), digest: "e".into() },
        ];
        let mut b = a.clone();
        b.dependencies.reverse();
        assert_eq!(a.identity(), b.identity());
        let mut c = a.clone();
        c.input_lipschitz = Some(vec![0.8]);
        assert_ne!(a.identity(), c.identity());
    }

    #[test]
    fn external_ode_matches_the_closed_form_for_a_linear_drift() {
        let mut implementation = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic);
        Arc::get_mut(&mut implementation).unwrap().spec.input_lipschitz = Some(vec![0.5]);
        let mut registry = MechanismRegistry::new();
        registry.register(implementation.clone()).unwrap();
        // dx0/dt = 0.5 x1 - x0 with x1 constant: x0(t) = 1 + (x0(0) - 1) e^{-t}.
        let ode = ExternalOdeSystem {
            equations: vec![
                SystemEquation::External {
                    id: "a".into(),
                    intercept: 0.0,
                    parameters: vec![0.5],
                    mechanism: implementation.spec.clone(),
                },
                SystemEquation::Builtin {
                    equation: Equation {
                        id: "b".into(),
                        intercept: 0.0,
                        weights: vec![0.0, 0.0],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: vec![],
                    },
                },
            ],
            decay: vec![1.0, 0.0],
        };
        let ctx = ExecutionContext::for_tests(1);
        let out = ode
            .resolve(Some(&registry))
            .unwrap()
            .trajectory(&[3.0, 2.0], &[0.0, 1.0, 2.0], &[], OdePolicy::default(), &ctx)
            .unwrap();
        for (k, t) in [0.0_f64, 1.0, 2.0].into_iter().enumerate() {
            assert!((out.trajectory.states[k][0] - (1.0 + 2.0 * (-t).exp())).abs() < 1e-6);
        }
        assert_eq!(out.trust, MechanismTrust::VerifiedExternal);
        assert!((out.drift_lipschitz.unwrap() - 1.5).abs() < 1e-12);
    }
}
