//! One verification receipt for every external provider, typed objects and
//! executable providers alike.
//!
//! A receipt records which exact requests passed which independent probes, and
//! who held the independent reference. It is plain data: it keeps each request's
//! identity and a digest of the contract that was checked, never the contract
//! itself, so it can leave the process and come back byte-for-byte equal. It
//! covers only those exact requests and certifies no scientific truth beyond
//! the probes, and holding one never licenses native behavior.

use std::collections::HashSet;

use antecedent_core::identity::CanonicalWriter;

use crate::{
    ExternalCapability, ExternalContractError, ExternalPosteriorKind, ExternalScientificObject,
    ExternalVerificationError, ProviderObjectIdentity, UtilityMonotonicity, VerificationProbe,
    VerificationProbeKind, capability_name, probe_name,
};

const RECEIPT_TAG: &str = "antecedent.provider_verification.receipt.v1";
const CONTRACT_TAG: &str = "antecedent.provider_verification.contract.v1";

/// What a verified request was a request *of*.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VerifiedContractKind {
    /// A declared probability law.
    Law,
    /// A declared posterior.
    Posterior,
    /// A candidate signal and update.
    Signal,
    /// A proof evidence factor.
    Evidence,
    /// An action utility.
    Utility,
    /// An executable provider run at one exact request.
    Execution,
}

impl VerifiedContractKind {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Law => "law",
            Self::Posterior => "posterior",
            Self::Signal => "signal",
            Self::Evidence => "evidence",
            Self::Utility => "utility",
            Self::Execution => "execution",
        }
    }

    /// Parse the name written by [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Law, Self::Posterior, Self::Signal, Self::Evidence, Self::Utility, Self::Execution]
            .into_iter()
            .find(|kind| kind.as_str() == name)
    }
}

/// An executable provider exercised at one exact request.
///
/// Unlike a scientific object it declares no coordinates: it is whatever the
/// provider's declarative spec says it is, identified by the digest of that
/// spec (carried as `identity.version_id`) and by the request fingerprint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionContract {
    /// Provider, query family (object), spec digest (version), data snapshot and request.
    pub identity: ProviderObjectIdentity,
    /// Whether repeated runs at this request must agree exactly.
    pub deterministic: bool,
    /// Whether the run reports an uncertainty that the reference also constrains.
    pub reports_uncertainty: bool,
    /// BLAKE3 hex digest of the opaque artifact the run emitted, when it emitted one.
    pub artifact_blake3: Option<String>,
}

fn hex64(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl ExecutionContract {
    fn validate(&self) -> Result<(), ExternalContractError> {
        if !self.identity.is_complete()
            || self.artifact_blake3.as_deref().is_some_and(|d| !hex64(d))
        {
            return Err(ExternalContractError::InvalidIdentity);
        }
        Ok(())
    }

    /// Canonical digest of every field of this contract.
    #[must_use]
    pub fn contract_digest(&self) -> String {
        let mut stream = CanonicalWriter::new(CONTRACT_TAG);
        stream.text(VerifiedContractKind::Execution.as_str());
        write_identity(&mut stream, &self.identity);
        stream
            .word(u64::from(self.deterministic))
            .word(u64::from(self.reports_uncertainty))
            .opt_text(self.artifact_blake3.as_deref());
        stream.finish_hex()
    }
}

/// The contract a set of probes is checked against.
#[derive(Clone, Copy, Debug)]
pub enum VerificationSubject<'a> {
    /// A typed scientific object, including its exact request identity.
    Object(&'a ExternalScientificObject),
    /// An executable provider at one exact request.
    Execution(&'a ExecutionContract),
}

fn write_identity(stream: &mut CanonicalWriter, identity: &ProviderObjectIdentity) {
    stream
        .text(&identity.provider_id)
        .text(&identity.object_id)
        .text(&identity.version_id)
        .text(&identity.snapshot_id)
        .text(&identity.request_id);
}

fn write_capabilities(stream: &mut CanonicalWriter, capabilities: &[ExternalCapability]) {
    let mut names: Vec<&str> = capabilities.iter().map(|c| capability_name(*c)).collect();
    names.sort_unstable();
    stream.texts(&names);
}

fn write_quantities(stream: &mut CanonicalWriter, quantities: &[crate::ScientificQuantity]) {
    stream.word(quantities.len() as u64);
    for quantity in quantities {
        quantity.write_canonical(stream);
    }
}

impl ExternalScientificObject {
    /// Which responsibility this declaration is.
    #[must_use]
    pub const fn verified_kind(&self) -> VerifiedContractKind {
        match self {
            Self::Law(_) => VerifiedContractKind::Law,
            Self::Posterior(_) => VerifiedContractKind::Posterior,
            Self::Signal(_) => VerifiedContractKind::Signal,
            Self::Evidence(_) => VerifiedContractKind::Evidence,
            Self::Utility(_) => VerifiedContractKind::Utility,
        }
    }

    /// Canonical digest of every declared field, including the exact request.
    ///
    /// Capabilities enter sorted, so declaration order does not move the digest;
    /// coordinate order does, because it is meaningful for joint samples.
    #[must_use]
    pub fn contract_digest(&self) -> String {
        let mut stream = CanonicalWriter::new(CONTRACT_TAG);
        stream.text(self.verified_kind().as_str());
        write_identity(&mut stream, self.identity());
        match self {
            Self::Law(law) => {
                write_quantities(&mut stream, &law.quantities);
                stream.text(crate::quantity::meaning_tag(law.meaning));
                write_capabilities(&mut stream, &law.capabilities);
            }
            Self::Posterior(post) => {
                write_quantities(&mut stream, &post.quantities);
                stream.word(match post.kind {
                    ExternalPosteriorKind::Parameter => 1,
                    ExternalPosteriorKind::Functional => 2,
                    ExternalPosteriorKind::CausalFunctional => 3,
                    ExternalPosteriorKind::StructuralModel => 4,
                });
                write_capabilities(&mut stream, &post.capabilities);
            }
            Self::Signal(signal) => {
                stream.text(&signal.candidate_id).text(&signal.prior_id);
                write_quantities(&mut stream, std::slice::from_ref(&signal.observation));
                write_capabilities(&mut stream, &signal.capabilities);
            }
            Self::Evidence(evidence) => {
                stream.text(&evidence.factor_id).word(match evidence.kind {
                    crate::ExternalEvidenceKind::Exact => 1,
                    crate::ExternalEvidenceKind::Empirical => 2,
                });
                write_quantities(&mut stream, &evidence.quantities);
                write_capabilities(&mut stream, &evidence.capabilities);
            }
            Self::Utility(utility) => {
                write_quantities(&mut stream, &utility.inputs);
                write_quantities(&mut stream, std::slice::from_ref(&utility.output));
                stream.texts(&utility.action_ids).word(u64::from(utility.stochastic));
                match utility.bounds {
                    Some((lo, hi)) => stream.word(1).real(lo).real(hi),
                    None => stream.word(0),
                };
                stream.word(utility.monotonicity.len() as u64);
                for monotonicity in &utility.monotonicity {
                    stream.word(match monotonicity {
                        UtilityMonotonicity::Unspecified => 0,
                        UtilityMonotonicity::Nondecreasing => 1,
                        UtilityMonotonicity::Nonincreasing => 2,
                    });
                }
                write_capabilities(&mut stream, &utility.capabilities);
            }
        }
        stream.finish_hex()
    }
}

impl VerificationSubject<'_> {
    fn identity(&self) -> &ProviderObjectIdentity {
        match self {
            Self::Object(object) => object.identity(),
            Self::Execution(execution) => &execution.identity,
        }
    }

    fn kind(&self) -> VerifiedContractKind {
        match self {
            Self::Object(object) => object.verified_kind(),
            Self::Execution(_) => VerifiedContractKind::Execution,
        }
    }

    fn contract_digest(&self) -> String {
        match self {
            Self::Object(object) => object.contract_digest(),
            Self::Execution(execution) => execution.contract_digest(),
        }
    }

    fn validate(&self) -> Result<(), ExternalContractError> {
        match self {
            Self::Object(object) => object.validate(),
            Self::Execution(execution) => execution.validate(),
        }
    }

    fn required_probes(&self) -> Vec<VerificationProbeKind> {
        use VerificationProbeKind as P;
        match self {
            Self::Object(object) => required_object_probes(object),
            Self::Execution(execution) => {
                let mut required = vec![P::KnownTruth, P::Shape];
                if execution.reports_uncertainty {
                    required.push(P::Uncertainty);
                }
                if execution.deterministic {
                    required.push(P::SeededBehavior);
                }
                required
            }
        }
    }
}

fn required_object_probes(object: &ExternalScientificObject) -> Vec<VerificationProbeKind> {
    use VerificationProbeKind as P;
    let has = |op| object.capabilities().contains(&op);
    let mut required = vec![P::KnownTruth];
    match object {
        ExternalScientificObject::Law(_) => {
            required.extend([P::Shape, P::Support]);
            if has(ExternalCapability::Cdf)
                || has(ExternalCapability::Quantile)
                || has(ExternalCapability::LogProbability)
            {
                required.push(P::Normalization);
            }
            if has(ExternalCapability::Mean) || has(ExternalCapability::Covariance) {
                required.push(P::Moments);
            }
            if has(ExternalCapability::Sample) {
                required.push(P::SeededBehavior);
            }
        }
        ExternalScientificObject::Posterior(_) => {
            required.extend([P::Shape, P::Support]);
            if has(ExternalCapability::Mean) || has(ExternalCapability::Covariance) {
                required.push(P::Moments);
            }
            if has(ExternalCapability::Sample) {
                required.push(P::SeededBehavior);
            }
        }
        ExternalScientificObject::Signal(_) => {
            required.push(P::UpdateCoherence);
            if has(ExternalCapability::Factor) {
                required.push(P::Normalization);
            }
            if has(ExternalCapability::Sample) {
                required.push(P::SeededBehavior);
            }
        }
        ExternalScientificObject::Evidence(_) => {
            required.extend([P::Normalization, P::Support]);
        }
        ExternalScientificObject::Utility(value) => {
            if value.bounds.is_some() {
                required.push(P::Support);
            }
            if value.monotonicity.iter().any(|m| *m != UtilityMonotonicity::Unspecified) {
                required.push(P::Monotonicity);
            }
            if value.stochastic {
                required.push(P::SeededBehavior);
            }
        }
    }
    required
}

fn check_probes(
    probes: &[VerificationProbe],
) -> Result<HashSet<VerificationProbeKind>, ExternalVerificationError> {
    let mut seen = HashSet::new();
    for probe in probes {
        if !probe.observed.is_finite()
            || !probe.expected.is_finite()
            || !probe.tolerance.is_finite()
            || probe.tolerance < 0.0
        {
            return Err(ExternalVerificationError::InvalidProbe(probe.kind));
        }
        if !seen.insert(probe.kind) {
            return Err(ExternalVerificationError::DuplicateProbe(probe.kind));
        }
    }
    Ok(seen)
}

fn require_passing(probes: &[VerificationProbe]) -> Result<(), ExternalVerificationError> {
    match probes.iter().find(|p| (p.observed - p.expected).abs() > p.tolerance) {
        Some(bad) => Err(ExternalVerificationError::FailedProbe(bad.kind)),
        None => Ok(()),
    }
}

/// One exact request that passed its probes.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedRequest {
    identity: ProviderObjectIdentity,
    kind: VerifiedContractKind,
    contract_digest: String,
    probes: Vec<VerificationProbe>,
}

impl VerifiedRequest {
    /// Rebuild a request from retained parts, for example from a stored receipt.
    ///
    /// The parts are re-checked: identities complete, the digest well formed,
    /// every probe finite, distinct and within its tolerance. The required
    /// probe set depends on the verified contract, which is not retained, so a
    /// reader relies on the consumer-held receipt digest for that.
    ///
    /// # Errors
    /// A blank identity, malformed digest, ill-formed or duplicate probe, or a
    /// probe outside its tolerance refuses.
    pub fn from_parts(
        identity: ProviderObjectIdentity,
        kind: VerifiedContractKind,
        contract_digest: String,
        probes: Vec<VerificationProbe>,
    ) -> Result<Self, ExternalVerificationError> {
        if !identity.is_complete() || !hex64(&contract_digest) {
            return Err(ExternalVerificationError::Contract(
                ExternalContractError::InvalidIdentity,
            ));
        }
        check_probes(&probes)?;
        require_passing(&probes)?;
        Ok(Self { identity, kind, contract_digest, probes })
    }

    /// Exact provider object, version, snapshot and request that was verified.
    #[must_use]
    pub fn identity(&self) -> &ProviderObjectIdentity {
        &self.identity
    }

    /// Which responsibility was verified.
    #[must_use]
    pub const fn kind(&self) -> VerifiedContractKind {
        self.kind
    }

    /// Canonical digest of the contract that was checked.
    #[must_use]
    pub fn contract_digest(&self) -> &str {
        &self.contract_digest
    }

    /// Probes that passed.
    #[must_use]
    pub fn probes(&self) -> &[VerificationProbe] {
        &self.probes
    }
}

/// Evidence that complete provider contracts passed their required probes.
///
/// It covers only requests equal in every declared field, including the exact
/// request; any other request needs its own verification.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderVerificationReceipt {
    evidence_origin: Option<String>,
    requests: Vec<VerifiedRequest>,
}

impl ProviderVerificationReceipt {
    /// Assemble a receipt from verified requests.
    ///
    /// # Errors
    /// No requests, a blank origin, or two requests with one identity refuse.
    pub fn from_requests(
        evidence_origin: Option<String>,
        requests: Vec<VerifiedRequest>,
    ) -> Result<Self, ExternalVerificationError> {
        use ExternalContractError as C;
        let contract = |error| ExternalVerificationError::Contract(error);
        if requests.is_empty()
            || evidence_origin.as_deref().is_some_and(|origin| origin.trim().is_empty())
        {
            return Err(contract(C::InvalidIdentity));
        }
        let mut identities = HashSet::new();
        if requests.iter().any(|request| !identities.insert(&request.identity)) {
            return Err(contract(C::DuplicateDeclaration));
        }
        Ok(Self { evidence_origin, requests })
    }

    /// Who held the independent reference, when it was named.
    #[must_use]
    pub fn evidence_origin(&self) -> Option<&str> {
        self.evidence_origin.as_deref()
    }

    /// Every verified request, in the order they were supplied.
    #[must_use]
    pub fn requests(&self) -> &[VerifiedRequest] {
        &self.requests
    }

    /// The verified request with exactly this identity, if any.
    #[must_use]
    pub fn request_for(&self, identity: &ProviderObjectIdentity) -> Option<&VerifiedRequest> {
        self.requests.iter().find(|request| &request.identity == identity)
    }

    /// Whether this receipt covers exactly the supplied typed object.
    #[must_use]
    pub fn covers(&self, object: &ExternalScientificObject) -> bool {
        self.covers_subject(&VerificationSubject::Object(object))
    }

    /// Whether this receipt covers exactly the supplied executable request.
    #[must_use]
    pub fn covers_execution(&self, execution: &ExecutionContract) -> bool {
        self.covers_subject(&VerificationSubject::Execution(execution))
    }

    fn covers_subject(&self, subject: &VerificationSubject<'_>) -> bool {
        self.request_for(subject.identity()).is_some_and(|request| {
            request.kind == subject.kind() && request.contract_digest == subject.contract_digest()
        })
    }

    /// Probes that passed for exactly this typed object, if it is covered.
    #[must_use]
    pub fn probes_for(&self, object: &ExternalScientificObject) -> Option<&[VerificationProbe]> {
        self.covers(object)
            .then(|| self.request_for(object.identity()).map(VerifiedRequest::probes))
            .flatten()
    }

    /// Canonical BLAKE3 digest of the whole receipt: origin, then every request's
    /// identity, kind, contract digest and probe values in order.
    ///
    /// A consumer retains this independently and refuses a receipt that differs.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut stream = CanonicalWriter::new(RECEIPT_TAG);
        stream.opt_text(self.evidence_origin.as_deref()).word(self.requests.len() as u64);
        for request in &self.requests {
            write_identity(&mut stream, &request.identity);
            stream
                .text(request.kind.as_str())
                .text(&request.contract_digest)
                .word(request.probes.len() as u64);
            for probe in &request.probes {
                stream
                    .text(probe_name(probe.kind))
                    .real(probe.observed)
                    .real(probe.expected)
                    .real(probe.tolerance);
            }
        }
        stream.finish_hex()
    }
}

fn verify_one(
    subject: &VerificationSubject<'_>,
    probes: &[VerificationProbe],
) -> Result<VerifiedRequest, ExternalVerificationError> {
    subject.validate().map_err(ExternalVerificationError::Contract)?;
    let seen = check_probes(probes)?;
    for kind in subject.required_probes() {
        if !seen.contains(&kind) {
            return Err(ExternalVerificationError::MissingProbe(kind));
        }
    }
    require_passing(probes)?;
    Ok(VerifiedRequest {
        identity: subject.identity().clone(),
        kind: subject.kind(),
        contract_digest: subject.contract_digest(),
        probes: probes.to_vec(),
    })
}

/// Verify one complete provider contract against independent probe values.
///
/// Every property required by the object kind and declared capabilities must
/// be present once and within its predeclared tolerance. Extra probes of
/// other kinds are accepted only if they also pass. The receipt names no
/// evidence origin; use [`verify_provider_requests`] to record one.
///
/// # Errors
/// An invalid contract, ill-formed or duplicate probe, a missing required
/// property or any out-of-tolerance probe refuses.
pub fn verify_external_object(
    object: &ExternalScientificObject,
    probes: &[VerificationProbe],
) -> Result<ProviderVerificationReceipt, ExternalVerificationError> {
    let request = verify_one(&VerificationSubject::Object(object), probes)?;
    ProviderVerificationReceipt::from_requests(None, vec![request])
}

/// Verify several exact requests of one provider under one named evidence origin.
///
/// Each request is checked as [`verify_external_object`] checks one object; the
/// first refusal is returned wrapped with its position.
///
/// # Errors
/// As [`verify_external_object`], per request, plus a blank origin, no requests
/// or two requests with one identity.
pub fn verify_provider_requests(
    evidence_origin: &str,
    requests: &[(VerificationSubject<'_>, &[VerificationProbe])],
) -> Result<ProviderVerificationReceipt, ExternalVerificationError> {
    let verified = requests
        .iter()
        .enumerate()
        .map(|(index, (subject, probes))| {
            verify_one(subject, probes).map_err(|cause| ExternalVerificationError::Request {
                index,
                cause: Box::new(cause),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    ProviderVerificationReceipt::from_requests(Some(evidence_origin.to_owned()), verified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DistributionMeaning, ExternalTrustState, LawProviderContract, QuantityRole};

    fn identity(request: &str) -> ProviderObjectIdentity {
        ProviderObjectIdentity {
            provider_id: "registry-name".into(),
            object_id: "effect".into(),
            version_id: "a".repeat(64),
            snapshot_id: "rows-1".into(),
            request_id: request.into(),
        }
    }

    fn execution(request: &str) -> ExecutionContract {
        ExecutionContract {
            identity: identity(request),
            deterministic: true,
            reports_uncertainty: false,
            artifact_blake3: Some("b".repeat(64)),
        }
    }

    fn probe(kind: VerificationProbeKind) -> VerificationProbe {
        VerificationProbe { kind, observed: 0.0, expected: 0.0, tolerance: 1e-9 }
    }

    fn passing() -> Vec<VerificationProbe> {
        use VerificationProbeKind as P;
        [P::KnownTruth, P::Shape, P::SeededBehavior].into_iter().map(probe).collect()
    }

    fn law() -> ExternalScientificObject {
        ExternalScientificObject::Law(LawProviderContract {
            identity: identity("req-law"),
            quantities: vec![crate::ScientificQuantity {
                variable_id: "schema:y".into(),
                variable_name: "Y".into(),
                role: QuantityRole::Outcome,
                units: "count".into(),
                population_id: "target".into(),
                regime_id: "do(a=1)".into(),
                horizon: 0,
                functional_id: "mean".into(),
                conditioning: vec![],
                transform_id: "identity".into(),
            }],
            meaning: DistributionMeaning::InterventionalPredictive,
            capabilities: vec![ExternalCapability::Mean],
        })
    }

    fn law_probes() -> Vec<VerificationProbe> {
        use VerificationProbeKind as P;
        [P::KnownTruth, P::Shape, P::Support, P::Moments].into_iter().map(probe).collect()
    }

    #[test]
    fn one_receipt_covers_executions_and_objects_at_their_exact_requests() {
        let (first, second, typed) = (execution("r1"), execution("r2"), law());
        let probes = passing();
        let law_probes = law_probes();
        let receipt = verify_provider_requests(
            "independent-fixtures/v1",
            &[
                (VerificationSubject::Execution(&first), &probes),
                (VerificationSubject::Execution(&second), &probes),
                (VerificationSubject::Object(&typed), &law_probes),
            ],
        )
        .unwrap();
        assert_eq!(receipt.evidence_origin(), Some("independent-fixtures/v1"));
        assert!(receipt.covers_execution(&first) && receipt.covers_execution(&second));
        assert!(receipt.covers(&typed));
        assert!(!receipt.covers_execution(&execution("r3")));
        let mut tampered = first.clone();
        tampered.artifact_blake3 = None;
        assert!(!receipt.covers_execution(&tampered), "an artifact change is another contract");
        let state = ExternalTrustState::ExactRequestVerified(Box::new(receipt));
        assert!(state.verifies(&typed) && state.verifies_execution(&first));
        assert!(!state.is_native_licensed());
    }

    #[test]
    fn execution_requires_its_declared_probes_and_names_the_failing_request() {
        use VerificationProbeKind as P;
        let (first, second) = (execution("r1"), execution("r2"));
        let good = passing();
        let mut no_replay = passing();
        no_replay.retain(|p| p.kind != P::SeededBehavior);
        assert_eq!(
            verify_provider_requests(
                "origin",
                &[
                    (VerificationSubject::Execution(&first), &good),
                    (VerificationSubject::Execution(&second), &no_replay),
                ],
            ),
            Err(ExternalVerificationError::Request {
                index: 1,
                cause: Box::new(ExternalVerificationError::MissingProbe(P::SeededBehavior)),
            })
        );
        let mut off = passing();
        off[0].observed = 0.5;
        let error =
            verify_provider_requests("origin", &[(VerificationSubject::Execution(&first), &off)])
                .unwrap_err();
        let refusal = error.to_refusal();
        assert_eq!(refusal.detail, "external_object_verification.failed_probe");
        assert_eq!(refusal.offending.as_deref(), Some("request[0].known_truth"));
        let uncertain = ExecutionContract { reports_uncertainty: true, ..execution("r1") };
        assert_eq!(
            verify_external_object_subject(&uncertain, &good),
            Err(ExternalVerificationError::MissingProbe(P::Uncertainty))
        );
    }

    fn verify_external_object_subject(
        execution: &ExecutionContract,
        probes: &[VerificationProbe],
    ) -> Result<VerifiedRequest, ExternalVerificationError> {
        verify_one(&VerificationSubject::Execution(execution), probes)
    }

    #[test]
    fn receipt_refuses_blank_origin_empty_and_duplicate_requests() {
        let first = execution("r1");
        let probes = passing();
        let subject = VerificationSubject::Execution(&first);
        assert_eq!(
            verify_provider_requests("  ", &[(subject, &probes)]),
            Err(ExternalVerificationError::Contract(ExternalContractError::InvalidIdentity))
        );
        assert_eq!(
            verify_provider_requests("origin", &[]),
            Err(ExternalVerificationError::Contract(ExternalContractError::InvalidIdentity))
        );
        assert_eq!(
            verify_provider_requests("origin", &[(subject, &probes), (subject, &probes)]),
            Err(ExternalVerificationError::Contract(ExternalContractError::DuplicateDeclaration))
        );
    }

    #[test]
    fn digest_is_stable_and_moves_with_every_retained_field() {
        let first = execution("r1");
        let probes = passing();
        let build = |origin: &str, probes: &[VerificationProbe], request: &str| {
            let contract = ExecutionContract { identity: identity(request), ..first.clone() };
            verify_provider_requests(origin, &[(VerificationSubject::Execution(&contract), probes)])
                .unwrap()
                .digest()
        };
        let base = build("origin", &probes, "r1");
        assert_eq!(base.len(), 64);
        assert_eq!(base, build("origin", &probes, "r1"));
        assert_ne!(base, build("other-origin", &probes, "r1"));
        assert_ne!(base, build("origin", &probes, "r2"));
        let mut looser = passing();
        looser[0].tolerance = 1e-3;
        assert_ne!(base, build("origin", &looser, "r1"));
    }

    #[test]
    fn retained_parts_round_trip_and_refuse_a_failed_or_malformed_probe() {
        let (first, typed) = (execution("r1"), law());
        let (probes, law_probes) = (passing(), law_probes());
        let receipt = verify_provider_requests(
            "origin",
            &[
                (VerificationSubject::Execution(&first), &probes),
                (VerificationSubject::Object(&typed), &law_probes),
            ],
        )
        .unwrap();
        let rebuilt = ProviderVerificationReceipt::from_requests(
            receipt.evidence_origin().map(str::to_owned),
            receipt
                .requests()
                .iter()
                .map(|r| {
                    VerifiedRequest::from_parts(
                        r.identity().clone(),
                        r.kind(),
                        r.contract_digest().to_owned(),
                        r.probes().to_vec(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(rebuilt, receipt);
        assert_eq!(rebuilt.digest(), receipt.digest());
        let mut failed = probes.clone();
        failed[0].observed = 1.0;
        assert!(matches!(
            VerifiedRequest::from_parts(
                identity("r1"),
                VerifiedContractKind::Execution,
                "c".repeat(64),
                failed,
            ),
            Err(ExternalVerificationError::FailedProbe(VerificationProbeKind::KnownTruth))
        ));
        assert!(
            VerifiedRequest::from_parts(
                identity("r1"),
                VerifiedContractKind::Execution,
                "short".into(),
                probes,
            )
            .is_err()
        );
    }
}
