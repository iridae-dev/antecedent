// SPDX-License-Identifier: MIT OR Apache-2.0
//! Portable fitted feedback systems with registered external mechanisms.
//!
//! The artifact records, for every external mechanism, a versioned implementation id, the
//! content digest of the implementation, its declared dependencies, signature, parameter
//! schema, derivative declaration and Lipschitz bounds. It never embeds code: no field can
//! carry an implementation, and the encoder has no input that could supply one. Independent
//! replay takes a caller-supplied [`TrustedMechanismResolver`], verifies each resolved
//! implementation against the record, then refits and recomputes every answer. With no
//! resolver, a missing implementation or any difference the replay is refused with a typed
//! reason code and nothing is run. A system with an external mechanism stays
//! `verified_external`; replay never upgrades it to native.

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::external_cyclic_fit::{
    fit_external_feedback, ExternalFeedbackFit, ExternalFitPolicy,
};
use antecedent_model::external_mechanism::{
    ExternalFeedbackSystem, ExternalFitParameter, ExternalMechanismEvidence,
    ExternalMechanismRefusal, ExternalMechanismSpec, MechanismTrust, TrustedMechanismResolver,
};
use antecedent_model::nonlinear_equilibrium::{
    Clamp, EquilibriumError, EquilibriumResult, RegimeRow, SolvePolicy,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const KIND: &str = "external_cyclic_equilibrium";
const SECTION: &str = "external_cyclic_equilibrium_v1";
/// Maximum logical/container bytes accepted by this family.
pub const MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;

/// Optional model-fitting stage; rows are independent equilibrium replicates under known
/// hard-clamp regimes. Not a time-series licence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalFitRequest {
    /// Scalars to estimate jointly.
    pub parameters: Vec<ExternalFitParameter>,
    /// Observations and their known hard-clamp regimes.
    pub rows: Vec<RegimeRow>,
    /// Deterministic numerical policy.
    pub policy: ExternalFitPolicy,
}

/// Complete retained target, specified system and optional statistical program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalCyclicRequest {
    /// Specified system: built-in equations and external mechanism identities.
    pub system: ExternalFeedbackSystem,
    /// Optional joint fit, run before evaluation.
    pub fit: Option<ExternalFitRequest>,
    /// Prespecified hard-clamp worlds, in requested order.
    pub worlds: Vec<Vec<Clamp>>,
    /// Aligned disturbance realizations shared across worlds.
    pub noise: Vec<Vec<f64>>,
    /// Numerical starting state; not an equilibrium selection law.
    pub initial: Vec<f64>,
    /// Numerical solve policy; part of retained identity.
    pub policy: SolvePolicy,
    /// Optional fully observed factual state for shared-noise counterfactuals.
    pub factual: Option<Vec<f64>>,
}

/// Recomputed answers and the verified standing of the mechanisms behind them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalCyclicReport {
    /// Request identity; retain independently before consumption.
    pub identity: String,
    /// The specified or fitted system the answers come from.
    pub system: ExternalFeedbackSystem,
    /// Joint fit and its numerical facts, when requested.
    pub fit: Option<ExternalFeedbackFit>,
    /// Verified evidence for each external mechanism.
    pub mechanisms: Vec<ExternalMechanismEvidence>,
    /// `verified_external` when any mechanism is external; never `native` then.
    pub trust: MechanismTrust,
    /// Results indexed by world, then disturbance row.
    pub worlds: Vec<Vec<EquilibriumResult>>,
    /// Shared-noise counterfactual for each requested world.
    pub counterfactuals: Option<Vec<EquilibriumResult>>,
    /// `verified_external_model_conditional` or `specified_or_fitted_model_conditional`.
    pub claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
}

/// Refusal, integrity, scientific or execution failure.
#[derive(Debug, Error)]
pub enum ExternalCyclicArtifactError {
    /// A typed external-mechanism refusal: no resolver, no implementation, or a changed one.
    #[error(transparent)]
    Refusal(#[from] ExternalMechanismRefusal),
    /// Size, shape or layout failure.
    #[error("external cyclic artifact invalid: {0}")]
    Invalid(String),
    /// Admission rejected the reservation.
    #[error(transparent)]
    Admission(#[from] antecedent_core::execution::AdmissionError),
    /// Cancelled by the execution policy.
    #[error("external cyclic execution cancelled")]
    Cancelled,
    /// Model or statistical computation failed.
    #[error(transparent)]
    Execution(EquilibriumError),
    /// The expected identity does not match, or replay differs from the stored results.
    #[error(
        "external cyclic artifact identity/replay mismatch; next attempt: consume it against the identity its producer retained"
    )]
    Mismatch,
}

impl From<EquilibriumError> for ExternalCyclicArtifactError {
    fn from(error: EquilibriumError) -> Self {
        // A refusal raised inside the model keeps its typed code across this boundary.
        if let EquilibriumError::Invalid(message) = &error {
            if let Some((code, detail)) = antecedent_core::reason_code::split_prefix(message) {
                if let Some(refusal) = refusal_from_code(code, detail) {
                    return Self::Refusal(refusal);
                }
            }
        }
        Self::Execution(error)
    }
}

fn refusal_from_code(code: &str, detail: &str) -> Option<ExternalMechanismRefusal> {
    use antecedent_model::external_mechanism::ExternalRefusalKind as Kind;
    let kind = match code {
        "mechanism_declaration_invalid" => Kind::DeclarationInvalid,
        "mechanism_resolver_missing" => Kind::ResolverMissing,
        "mechanism_implementation_unavailable" => Kind::Unavailable,
        "mechanism_implementation_changed" => Kind::Changed,
        "mechanism_resolver_failed" => Kind::ResolverFailed,
        _ => return None,
    };
    Some(ExternalMechanismRefusal { kind, detail: detail.to_owned() })
}

fn invalid(e: impl std::fmt::Display) -> ExternalCyclicArtifactError {
    ExternalCyclicArtifactError::Invalid(e.to_string())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u32,
    request: ExternalCyclicRequest,
    report: ExternalCyclicReport,
}

impl ExternalCyclicRequest {
    /// Identity of the complete retained scientific and numerical request. The external
    /// mechanisms enter through their recorded identities, never through code.
    pub fn identity(&self) -> Result<String, ExternalCyclicArtifactError> {
        self.check_limits()?;
        let bytes = serde_json::to_vec(self).map_err(invalid)?;
        if bytes.len() > MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES {
            return Err(invalid("request byte limit"));
        }
        Ok(CanonicalWriter::digest_bytes_hex(
            "antecedent.external_cyclic_equilibrium.request.v1",
            &bytes,
        ))
    }

    fn check_limits(&self) -> Result<(), ExternalCyclicArtifactError> {
        self.system.validate()?;
        let n = self.system.order();
        if self.worlds.is_empty()
            || self.worlds.len() > 256
            || self.noise.is_empty()
            || self.noise.len() > 10_000
            || self.worlds.len().saturating_mul(self.noise.len()).saturating_mul(n) > 100_000
            || self.initial.len() != n
            || self.initial.iter().any(|v| !v.is_finite())
            || self.noise.iter().any(|row| row.len() != n || row.iter().any(|v| !v.is_finite()))
            || self
                .factual
                .as_ref()
                .is_some_and(|v| v.len() != n || v.iter().any(|v| !v.is_finite()))
            || self.fit.as_ref().is_some_and(|f| f.rows.len() > 10_000 || f.parameters.len() > 1024)
        {
            return Err(invalid("retained request shape or size limit"));
        }
        Ok(())
    }

    /// Resolve and verify the external mechanisms, run the retained fit and the aligned
    /// intervention program. Refuses before any computation when the resolver is absent or
    /// its implementations differ from the record.
    pub fn execute(
        &self,
        resolver: Option<&dyn TrustedMechanismResolver>,
        ctx: &ExecutionContext,
    ) -> Result<ExternalCyclicReport, ExternalCyclicArtifactError> {
        let identity = self.identity()?;
        // Verification precedes everything, including fitting.
        self.system.resolve(resolver)?;
        let n = self.system.order();
        let bytes = (n as u64)
            .saturating_mul(n as u64)
            .saturating_mul(64)
            .saturating_add(self.worlds.len() as u64 * self.noise.len() as u64 * n as u64 * 128)
            .saturating_add(65_536);
        let _lease = ctx.acquire_admission(0, bytes)?;
        let fit = match &self.fit {
            Some(request) => {
                let fit = fit_external_feedback(
                    &self.system,
                    resolver,
                    &request.parameters,
                    &request.rows,
                    request.policy,
                    ctx,
                )?;
                if !fit.converged {
                    return Err(invalid("fit did not converge"));
                }
                Some(fit)
            }
            None => None,
        };
        let system = fit.as_ref().map_or_else(|| self.system.clone(), |f| f.system.clone());
        let resolved = system.resolve(resolver)?;
        let mut worlds = Vec::with_capacity(self.worlds.len());
        for clamps in &self.worlds {
            let mut rows = Vec::with_capacity(self.noise.len());
            for noise in &self.noise {
                rows.push(resolved.solve(noise, clamps, &self.initial, self.policy, ctx)?);
            }
            worlds.push(rows);
        }
        let counterfactuals = self
            .factual
            .as_ref()
            .map(|factual| {
                self.worlds
                    .iter()
                    .map(|clamps| resolved.counterfactual(factual, clamps, self.policy, ctx))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        if ctx.cancellation.is_cancelled() {
            return Err(ExternalCyclicArtifactError::Cancelled);
        }
        let trust = resolved.trust();
        Ok(ExternalCyclicReport {
            identity,
            mechanisms: resolved.evidence(),
            system,
            fit,
            trust,
            worlds,
            counterfactuals,
            claim: match trust {
                MechanismTrust::VerifiedExternal => "verified_external_model_conditional".into(),
                MechanismTrust::Native => "specified_or_fitted_model_conditional".into(),
            },
            calibration: "unmeasured".into(),
        })
    }

    /// Execute and encode in the bounded, checksummed section container.
    pub fn export(
        &self,
        artifact_id: &str,
        resolver: Option<&dyn TrustedMechanismResolver>,
        ctx: &ExecutionContext,
    ) -> Result<(ExternalCyclicReport, Vec<u8>), ExternalCyclicArtifactError> {
        if artifact_id.trim().is_empty() || artifact_id.len() > 256 {
            return Err(invalid("artifact id"));
        }
        let report = self.execute(resolver, ctx)?;
        let record = Record { version: 1, request: self.clone(), report: report.clone() };
        let bytes = serde_json::to_vec(&record).map_err(invalid)?;
        if bytes.len() > MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES {
            return Err(invalid("logical section byte limit"));
        }
        let container = crate::encode_container(
            &crate::ArtifactContainer {
                kind: KIND,
                note: "verified_external_model_conditional_calibration_unmeasured",
                max_bytes: MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES,
                artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[crate::OutSection { id: SECTION, content_type: crate::CONTENT_BYTES, bytes: &bytes }],
        )
        .map_err(|e| invalid(e.into_io("external cyclic artifact layout")))?;
        Ok((report, container))
    }
}

fn decode_record(bytes: &[u8]) -> Result<Record, ExternalCyclicArtifactError> {
    if bytes.len() > MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES {
        return Err(invalid("byte limit"));
    }
    let parts =
        crate::decode_container(KIND, MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES, &[SECTION], bytes)
            .map_err(|e| invalid(e.into_io("external cyclic artifact layout")))?;
    let payload = parts.first().ok_or_else(|| invalid("missing section"))?;
    let record: Record = serde_json::from_slice(payload).map_err(invalid)?;
    if record.version != 1 {
        return Err(invalid("unsupported family version"));
    }
    Ok(record)
}

/// What an artifact records about its mechanisms, read without any resolver or execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalCyclicInspection {
    /// Request identity recomputed from the stored request.
    pub identity: String,
    /// Recorded identities (never code) of every external mechanism.
    pub mechanisms: Vec<RecordedMechanism>,
    /// Standing recorded for the system.
    pub trust: MechanismTrust,
}

/// A recorded external mechanism identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedMechanism {
    /// Equation identity.
    pub equation: String,
    /// Identity over every declaration.
    pub identity: String,
    /// The recorded declarations.
    pub spec: ExternalMechanismSpec,
}

/// Inspect the recorded mechanism identities of an artifact. Runs nothing and needs no
/// resolver; the identities are the producer's claims until a replay verifies them.
pub fn inspect_external_cyclic_artifact(
    bytes: &[u8],
) -> Result<ExternalCyclicInspection, ExternalCyclicArtifactError> {
    let record = decode_record(bytes)?;
    Ok(ExternalCyclicInspection {
        identity: record.request.identity()?,
        mechanisms: record
            .request
            .system
            .external_specs()
            .into_iter()
            .map(|(equation, spec)| RecordedMechanism {
                equation: equation.to_owned(),
                identity: spec.identity(),
                spec: spec.clone(),
            })
            .collect(),
        trust: record.request.system.trust(),
    })
}

/// Independently replay an artifact: check the retained identity, verify every external
/// implementation the caller's resolver supplies against the record, then refit and
/// recompute all answers and compare them with the stored report.
///
/// # Errors
/// A typed refusal when the resolver is absent, lacks an implementation or supplies one that
/// differs from the record; framing, identity, replay and resource failures.
pub fn consume_external_cyclic_artifact(
    bytes: &[u8],
    expected_identity: &str,
    resolver: Option<&dyn TrustedMechanismResolver>,
    ctx: &ExecutionContext,
) -> Result<ExternalCyclicReport, ExternalCyclicArtifactError> {
    if expected_identity.len() != 64 {
        return Err(invalid("expected identity"));
    }
    let record = decode_record(bytes)?;
    if record.request.identity()? != expected_identity
        || record.report.identity != expected_identity
    {
        return Err(ExternalCyclicArtifactError::Mismatch);
    }
    let report = record.request.execute(resolver, ctx)?;
    if report != record.report {
        return Err(ExternalCyclicArtifactError::Mismatch);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_model::external_mechanism::{
        DerivativeDeclaration, ExternalMechanismImplementation, MechanismRegistry, SystemEquation,
    };
    use antecedent_model::nonlinear_equilibrium::{Activation, Equation};
    use std::sync::Arc;

    const SOURCE: &str = "SECRET_IMPLEMENTATION_MARKER gain * tanh(x)";
    struct Gain {
        spec: ExternalMechanismSpec,
        content: Vec<u8>,
    }
    impl Gain {
        fn new(source: &str) -> Arc<Self> {
            let content = source.as_bytes().to_vec();
            let mut spec = ExternalMechanismSpec::for_content(
                "lab/gain",
                "1.0",
                &content,
                vec![1],
                vec!["gain".into()],
                DerivativeDeclaration::Analytic,
            );
            spec.input_lipschitz = Some(vec![0.9]);
            Arc::new(Self { spec, content })
        }
    }
    impl ExternalMechanismImplementation for Gain {
        fn spec(&self) -> ExternalMechanismSpec {
            self.spec.clone()
        }
        fn content(&self) -> &[u8] {
            &self.content
        }
        fn value(&self, inputs: &[f64], p: &[f64]) -> Result<f64, EquilibriumError> {
            Ok(p[0] * inputs[0].tanh())
        }
        fn input_gradient(
            &self,
            inputs: &[f64],
            p: &[f64],
        ) -> Result<Option<Vec<f64>>, EquilibriumError> {
            Ok(Some(vec![p[0] * (1.0 - inputs[0].tanh().powi(2))]))
        }
        fn parameter_gradient(
            &self,
            inputs: &[f64],
            _p: &[f64],
        ) -> Result<Option<Vec<f64>>, EquilibriumError> {
            Ok(Some(vec![inputs[0].tanh()]))
        }
    }

    fn request(spec: &ExternalMechanismSpec) -> ExternalCyclicRequest {
        let system = ExternalFeedbackSystem {
            equations: vec![
                SystemEquation::External {
                    id: "response".into(),
                    intercept: 0.0,
                    parameters: vec![0.0],
                    mechanism: spec.clone(),
                },
                SystemEquation::Builtin {
                    equation: Equation {
                        id: "feedback".into(),
                        intercept: 0.0,
                        weights: vec![0.2, 0.0],
                        activation: Activation::Tanh,
                        expression: None,
                        parameters: vec![],
                    },
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        };
        let rows = [-2.0_f64, -1.0, 0.5, 1.0, 2.0]
            .into_iter()
            .flat_map(|x| {
                [-0.1, 0.1].map(|noise| RegimeRow {
                    values: vec![0.7 * x.tanh() + noise, x],
                    clamps: vec![Clamp { variable: 1, value: x }],
                })
            })
            .collect();
        ExternalCyclicRequest {
            system,
            fit: Some(ExternalFitRequest {
                parameters: vec![ExternalFitParameter::Parameter { equation: 0, parameter: 0 }],
                rows,
                policy: ExternalFitPolicy::default(),
            }),
            worlds: vec![vec![], vec![Clamp { variable: 1, value: 1.0 }]],
            noise: vec![vec![0.1, 0.2], vec![-0.1, -0.2]],
            initial: vec![0.0; 2],
            policy: SolvePolicy::default(),
            factual: Some(vec![0.3, 0.7]),
        }
    }

    fn registry(source: &str) -> MechanismRegistry {
        let mut registry = MechanismRegistry::new();
        registry.register(Gain::new(source)).unwrap();
        registry
    }

    #[test]
    fn fit_export_and_independent_replay_with_a_fresh_resolver_reproduce_the_results() {
        let ctx = ExecutionContext::for_tests(11);
        let request = request(&Gain::new(SOURCE).spec);
        let (report, bytes) =
            request.export("external_fit", Some(&registry(SOURCE)), &ctx).unwrap();
        assert_eq!(report.trust, MechanismTrust::VerifiedExternal);
        assert_ne!(report.trust, MechanismTrust::Native);
        let SystemEquation::External { parameters, .. } = &report.system.equations[0] else {
            panic!("external equation");
        };
        assert!((parameters[0] - 0.7).abs() < 0.05, "{parameters:?}");
        // A new registry stands in for a fresh process.
        let identity = request.identity().unwrap();
        let replay =
            consume_external_cyclic_artifact(&bytes, &identity, Some(&registry(SOURCE)), &ctx)
                .unwrap();
        assert_eq!(replay, report);
        assert_eq!(replay.trust, MechanismTrust::VerifiedExternal);
        let inspected = inspect_external_cyclic_artifact(&bytes).unwrap();
        assert_eq!(inspected.mechanisms.len(), 1);
        assert_eq!(inspected.trust, MechanismTrust::VerifiedExternal);
    }

    #[test]
    fn replay_without_a_resolver_or_with_a_changed_implementation_is_refused() {
        let ctx = ExecutionContext::for_tests(11);
        let request = request(&Gain::new(SOURCE).spec);
        let (_, bytes) = request.export("external_fit", Some(&registry(SOURCE)), &ctx).unwrap();
        let identity = request.identity().unwrap();
        let kind = |r: Result<ExternalCyclicReport, ExternalCyclicArtifactError>| match r {
            Err(ExternalCyclicArtifactError::Refusal(refusal)) => refusal.code(),
            other => panic!("expected a typed refusal, got {other:?}"),
        };
        assert_eq!(
            kind(consume_external_cyclic_artifact(&bytes, &identity, None, &ctx)),
            "mechanism_resolver_missing"
        );
        assert_eq!(
            kind(consume_external_cyclic_artifact(
                &bytes,
                &identity,
                Some(&MechanismRegistry::new()),
                &ctx
            )),
            "mechanism_implementation_unavailable"
        );
        let edited = registry("SECRET_IMPLEMENTATION_MARKER gain * tanh(1.01 * x)");
        assert_eq!(
            kind(consume_external_cyclic_artifact(&bytes, &identity, Some(&edited), &ctx)),
            "mechanism_implementation_changed"
        );
    }

    #[test]
    fn artifact_bytes_contain_no_executable_code() {
        let ctx = ExecutionContext::for_tests(11);
        let request = request(&Gain::new(SOURCE).spec);
        let (_, bytes) = request.export("external_fit", Some(&registry(SOURCE)), &ctx).unwrap();
        let marker = b"SECRET_IMPLEMENTATION_MARKER";
        assert!(!bytes.windows(marker.len()).any(|w| w == marker));
        let parts =
            crate::decode_container(KIND, MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES, &[SECTION], &bytes)
                .unwrap();
        let section = &parts[0];
        assert!(!section.windows(marker.len()).any(|w| w == marker));
        let text = std::str::from_utf8(section).unwrap();
        for forbidden in
            ["\"content\"", "\"source\"", "\"code\"", "\"callable\"", "\"bytecode\"", "tanh("]
        {
            assert!(!text.contains(forbidden), "artifact carries `{forbidden}`");
        }
        assert!(text.contains(&Gain::new(SOURCE).spec.content_digest));
    }

    #[test]
    fn a_resealed_record_is_refused_by_the_retained_identity() {
        let ctx = ExecutionContext::for_tests(11);
        let request = request(&Gain::new(SOURCE).spec);
        let (_, bytes) = request.export("external_fit", Some(&registry(SOURCE)), &ctx).unwrap();
        let mut other = request.clone();
        other.noise[0][0] += 0.5;
        let identity = other.identity().unwrap();
        assert!(matches!(
            consume_external_cyclic_artifact(&bytes, &identity, Some(&registry(SOURCE)), &ctx),
            Err(ExternalCyclicArtifactError::Mismatch)
        ));
    }
}
