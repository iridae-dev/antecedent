//! Explicit joint scenarios for the compiler's robust terminal; no Cartesian
//! product of marginal envelopes or invented structure probabilities.
use super::{
    AdmissibleDecisionContract, AtomEvidence, AtomWire, CanonicalWriter, Deserialize,
    ExternalRefusal, IoError, ProfileWire, ResultWire, RobustVerdict, Serialize, atom_wires,
    check_atom_wires, debug_error, profile_from_wire, profile_to_wire, result_wire, source_digest,
    to_cbor,
};
use crate::decision_adapters::{AdaptedClaims, AdaptedDecision, ClaimKind, evaluate_adapted};
use crate::decision_contract::StructuralPolicy;
use crate::decision_structural::StructuralVerdict;
use antecedent_core::execution::AdmissionLease;
use antecedent_core::{ExecutionContext, SearchStop};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionMetadata, DrawAlignment,
};
use std::{io::Write, sync::Arc};

const MAGIC: &[u8] = b"ANTE-JOINT-SCENARIOS-1";
const RESULT_MAGIC: &[u8] = b"ANTE-COMPILED-ROBUST-1";
/// Maximum encoded bytes of the narrow scenario carrier.
pub const MAX_SCENARIO_CARRIER_BYTES: usize = 64 * 1024 * 1024;
/// Maximum scenarios, actions or constraints admitted by this compiler terminal.
pub const MAX_COMPILED_ROBUST_ITEMS: usize = 256;
const DOMAIN: &str = "antecedent.compiled_robust.scenarios.v1";

fn carrier_refused(message: impl Into<String>) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
        message: format!(
            "{}: {}",
            antecedent_core::reason_details::decision_joint_scenarios::CARRIER,
            message.into()
        ),
    }
}
fn refusal_error(error: &impl std::fmt::Debug, refusal: &ExternalRefusal) -> IoError {
    IoError::Refused { code: refusal.code, message: format!("{}: {error:?}", refusal.detail) }
}
fn cancelled(ctx: &ExecutionContext) -> Result<(), IoError> {
    if ctx.cancellation.is_cancelled() {
        return Err(antecedent_core::execution::AdmissionError::Cancelled.into());
    }
    Ok(())
}
struct ByteCounter(u64);
impl Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(buf.len() as u64)
            .ok_or_else(|| std::io::Error::other("encoded size overflow"))?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn cbor_len(value: &impl Serialize) -> Result<u64, IoError> {
    let mut count = ByteCounter(0);
    ciborium::into_writer(value, &mut count).map_err(debug_error)?;
    Ok(count.0)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioLawWire {
    atom_id: String,
    metadata: DistributionMetadata,
    draws: Vec<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioCarrierWire {
    version: u32,
    causal_target: String,
    causal_identification: String,
    joint_structure: String,
    exhaustive: bool,
    stop: Option<String>,
    atoms: Vec<AtomWire>,
    profile: ProfileWire,
    laws: Vec<ScenarioLawWire>,
    completion_counts: Vec<(String, u64)>,
}

/// Validated finite joint scenario set with causal binding and explicit dependence.
/// Scientific standing remains conditional on these supplied scenarios. Loading and
/// numerical replay do not prove that the scenarios exhaust the scientific model.
#[derive(Clone, Debug)]
pub struct JointScenarioCarrier {
    wire: Arc<ScenarioCarrierWire>,
    digest: String,
    retention: Arc<AdmissionLease>,
}

/// A completed robust-engine readout whose retained output remains admitted.
/// It carries conditional scenario standing; it is not a posterior.
#[derive(Clone, Debug)]
pub struct ScenarioDecision {
    decision: Arc<AdaptedDecision>,
    source_identity: Arc<str>,
    joint_structure: Arc<str>,
    complete: bool,
    stop: Option<Arc<str>>,
    _retention: Arc<AdmissionLease>,
}
impl ScenarioDecision {
    /// Per-scenario values, robustness and unresolved probability masses.
    #[must_use]
    pub fn decision(&self) -> &AdaptedDecision {
        &self.decision
    }
    /// Exact source identity the terminal read.
    #[must_use]
    pub fn source_identity(&self) -> &str {
        &self.source_identity
    }
    /// Declared cross-action structure, not an inferred independence assumption.
    #[must_use]
    pub fn joint_structure(&self) -> &str {
        &self.joint_structure
    }
    /// Whether the declared finite scenario set is exhaustive and fully identified/evaluated.
    #[must_use]
    pub const fn complete(&self) -> bool {
        self.complete
    }
    /// Retained historical search stop.
    #[must_use]
    pub fn search_stop(&self) -> Option<&str> {
        self.stop.as_deref()
    }
}

impl JointScenarioCarrier {
    /// Own already adapted scenarios. `joint_structure` must explain their common
    /// cross-action state; every evaluated artifact must explicitly carry joint rows.
    /// The exhaustive declaration and any search stop are retained, not inferred.
    ///
    /// # Errors
    /// Non-finite-scenario claims, absent joint structure, marginal alignment, excess
    /// size, malformed atoms/support or admission/cancellation failure.
    pub fn checked(
        causal_target: &str,
        causal_identification: &str,
        joint_structure: &str,
        exhaustive: bool,
        stop: Option<SearchStop>,
        adapted: AdaptedClaims,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        cancelled(ctx)?;
        if adapted.kind != ClaimKind::FiniteScenarios
            || joint_structure.trim().is_empty()
            || joint_structure.len() > 4096
            || adapted.atoms.is_empty()
            || adapted.atoms.len() > MAX_COMPILED_ROBUST_ITEMS
        {
            return Err(carrier_refused(
                "supply at most 256 explicit finite joint scenarios and a named cross-action dependence declaration",
            ));
        }
        let mut quote = 4096u64;
        for atom in &adapted.atoms {
            quote = quote.checked_add(atom.id.len() as u64).ok_or(IoError::TooLarge)?;
            if let AtomEvidence::Evaluated(source) = &atom.evidence {
                if source.metadata().identity.alignment != DrawAlignment::Joint {
                    return Err(carrier_refused(
                        "independent marginals cannot establish joint cross-action scenarios",
                    ));
                }
                let bytes = cbor_len(source.metadata())?
                    .checked_add(
                        (source.draws().len() as u64).checked_mul(8).ok_or(IoError::TooLarge)?,
                    )
                    .and_then(|n| n.checked_add(4096))
                    .ok_or(IoError::TooLarge)?;
                quote = quote.checked_add(bytes).ok_or(IoError::TooLarge)?;
            }
        }
        // Include every support string/entry without allocating a cloned profile first.
        for (id, support) in &adapted.profile.support {
            quote = quote.checked_add(id.len() as u64 + 256).ok_or(IoError::TooLarge)?;
            for input in &support.per_input {
                quote = quote
                    .checked_add(input.action_id.len() as u64 + 256)
                    .ok_or(IoError::TooLarge)?;
            }
        }
        if quote > MAX_SCENARIO_CARRIER_BYTES as u64 {
            return Err(IoError::TooLarge);
        }
        let reservation = ctx.acquire_admission_product(0, quote, 6)?;
        let mut laws = Vec::new();
        for atom in &adapted.atoms {
            cancelled(ctx)?;
            if let AtomEvidence::Evaluated(source) = &atom.evidence {
                laws.push(ScenarioLawWire {
                    atom_id: atom.id.clone(),
                    metadata: source.metadata().clone(),
                    draws: source.draws().to_vec(),
                });
            }
        }
        let wire = ScenarioCarrierWire {
            version: 1,
            causal_target: causal_target.into(),
            causal_identification: causal_identification.into(),
            joint_structure: joint_structure.into(),
            exhaustive,
            stop: stop.map(|s| s.code().into()),
            atoms: atom_wires(&adapted.atoms),
            profile: profile_to_wire(&adapted.profile),
            laws,
            completion_counts: adapted.completion_counts,
        };
        let carrier = Self::from_wire(wire, &reservation)?;
        drop(reservation);
        let _workspace = ctx.acquire_admission(0, carrier.source_workspace_bytes(1)?)?;
        carrier.claims(StructuralPolicy::ReportOnly)?;
        Ok(carrier)
    }
    fn from_wire(wire: ScenarioCarrierWire, reservation: &AdmissionLease) -> Result<Self, IoError> {
        if wire.version != 1 {
            return Err(IoError::UnsupportedVersion { version: wire.version });
        }
        if wire.atoms.is_empty()
            || wire.atoms.len() > MAX_COMPILED_ROBUST_ITEMS
            || wire.laws.len() > wire.atoms.len()
            || wire.joint_structure.trim().is_empty()
            || wire.joint_structure.len() > 4096
            || wire.stop.as_deref().is_some_and(|s| {
                !matches!(
                    s,
                    "search.operations" | "search.depth" | "search.memory" | "search.cancelled"
                )
            })
            || [&wire.causal_target, &wire.causal_identification].iter().any(|s| {
                s.len() != 64
                    || !s.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            })
        {
            return Err(carrier_refused("unsupported or malformed joint scenario carrier"));
        }
        check_atom_wires(&wire.atoms)?;
        let encoded = to_cbor(&wire)?;
        if encoded.len().checked_add(MAGIC.len()).is_none_or(|n| n > MAX_SCENARIO_CARRIER_BYTES) {
            return Err(IoError::TooLarge);
        }
        let digest = CanonicalWriter::digest_bytes_hex(DOMAIN, &encoded);
        let retention = Arc::new(reservation.acquire_child_product(0, encoded.len() as u64, 4)?);
        Ok(Self { wire: Arc::new(wire), digest, retention })
    }
    /// Content identity of the source laws, causal binding, support and search declaration.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.digest
    }
    /// Source causal target identity, checked by the facade compiler.
    #[must_use]
    pub fn causal_target(&self) -> &str {
        &self.wire.causal_target
    }
    /// Source identification identity, checked by the facade compiler.
    #[must_use]
    pub fn causal_identification(&self) -> &str {
        &self.wire.causal_identification
    }
    /// Explicitly declared cross-action dependence, retained verbatim.
    #[must_use]
    pub fn joint_structure(&self) -> &str {
        &self.wire.joint_structure
    }
    /// Whether the declared finite scenario set is exhaustive and fully evaluated.
    #[must_use]
    pub fn complete(&self) -> bool {
        let probabilities_complete = self.wire.atoms.iter().all(|a| a.probability.is_none())
            || (self.wire.atoms.iter().all(|a| a.probability.is_some())
                && (self.wire.atoms.iter().filter_map(|a| a.probability).sum::<f64>() - 1.0).abs()
                    <= 1e-12);
        self.wire.exhaustive
            && self.wire.stop.is_none()
            && self.wire.atoms.iter().all(|a| a.status == "evaluated")
            && probabilities_complete
    }
    /// Search stop, when discovery was incomplete.
    #[must_use]
    pub fn search_stop(&self) -> Option<&str> {
        self.wire.stop.as_deref()
    }
    /// Number of retained scenarios, including unidentified and unevaluated entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.wire.atoms.len()
    }
    /// A validated carrier is nonempty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.wire.atoms.is_empty()
    }
    fn source_workspace_bytes(&self, actions: usize) -> Result<u64, IoError> {
        let mut bytes = 4096u64;
        for law in &self.wire.laws {
            bytes = bytes
                .checked_add(cbor_len(&law.metadata)?)
                .and_then(|n| n.checked_add((law.draws.len() as u64).checked_mul(8)?))
                .ok_or(IoError::TooLarge)?;
        }
        bytes
            .checked_mul((actions as u64).checked_add(8).ok_or(IoError::TooLarge)?)
            .and_then(|n| n.checked_mul(4))
            .ok_or(IoError::TooLarge)
    }
    /// Move this retained source into the requested execution budget without copying it.
    /// # Errors
    /// Cancellation or insufficient retained-memory admission.
    #[doc(hidden)]
    pub fn admit(mut self, ctx: &ExecutionContext) -> Result<Self, IoError> {
        cancelled(ctx)?;
        if self.retention.belongs_to(&ctx.admission()) {
            self.retention.observe_current();
        } else {
            self.retention =
                Arc::new(ctx.acquire_admission_product(0, cbor_len(&*self.wire)?, 4)?);
        }
        Ok(self)
    }
    fn output_bytes(&self, contract: &AdmissibleDecisionContract) -> Result<u64, IoError> {
        let mut metadata = ByteCounter(0);
        write!(&mut metadata, "{contract:?}").map_err(debug_error)?;
        let atoms = self.len() as u64;
        let actions = contract.contract.actions.len() as u64;
        let constraints = contract.contract.constraints.len() as u64;
        let copies = atoms
            .checked_mul(actions.checked_add(1).ok_or(IoError::TooLarge)?)
            .ok_or(IoError::TooLarge)?;
        let cells = copies
            .checked_mul(constraints.checked_add(16).ok_or(IoError::TooLarge)?)
            .ok_or(IoError::TooLarge)?;
        cells
            .checked_mul(256)
            .and_then(|n| n.checked_add(metadata.0.checked_mul(copies)?.checked_mul(4)?))
            .and_then(|n| n.checked_add(32_768))
            .ok_or(IoError::TooLarge)
    }
    /// JSON read model only; accepting a result still requires its checked consumer.
    /// # Errors
    /// Source mismatch or serialization failure.
    #[doc(hidden)]
    pub fn result_json(&self, result: &ScenarioDecision) -> Result<String, IoError> {
        if result.source_identity() != self.identity() {
            return Err(carrier_refused("readout belongs to a different joint source"));
        }
        serde_json::to_string(&serde_json::json!({"robust":result_wire(&result.decision().robust),
            "source_identity":self.identity(),"causal_target":self.causal_target(),"causal_identification":self.causal_identification(),
            "joint_structure":self.joint_structure(),"complete":result.complete(),"search_stop":result.search_stop(),
            "standing":"conditional_on_declared_joint_scenarios","native_verified":false})).map_err(debug_error)
    }
    /// Export the full joint source with its provenance and search declaration.
    /// # Errors
    /// Serialization, limits or context admission failure.
    pub fn export(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, IoError> {
        cancelled(ctx)?;
        let size = cbor_len(&*self.wire)?;
        let _import = if self.retention.belongs_to(&ctx.admission()) {
            self.retention.observe_current();
            None
        } else {
            Some(ctx.acquire_admission_product(0, size, 4)?)
        };
        let _workspace = ctx.acquire_admission_product(0, size, 2)?;
        let body = to_cbor(&*self.wire)?;
        let mut bytes = Vec::with_capacity(MAGIC.len() + body.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&body);
        Ok(bytes)
    }
    /// Load only against a separately retained source identity. An edited source
    /// cannot be substituted for a carrier a compiler already bound.
    /// # Errors
    /// Malformed/unsupported framing, source mismatch, limits or invalid joint claims.
    pub fn consume(
        bytes: &[u8],
        expected_identity: &str,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        cancelled(ctx)?;
        if bytes.len() > MAX_SCENARIO_CARRIER_BYTES {
            return Err(IoError::TooLarge);
        }
        // Invalid inputs can contain densely encoded empty-string/count tuples
        // before the typed profile/count checks reject them. Include decoded
        // container capacity overhead, as for the result frame below.
        let reservation = ctx.acquire_admission_product(0, bytes.len() as u64, 64)?;
        let body = bytes.strip_prefix(MAGIC).ok_or(IoError::BadMagic)?;
        let wire = antecedent_io_core::from_cbor(body)?;
        let carrier = Self::from_wire(wire, &reservation)?;
        drop(reservation);
        if carrier.identity() != expected_identity {
            return Err(carrier_refused(
                "joint scenario source differs from the retained compiler input",
            ));
        }
        let _workspace = ctx.acquire_admission(0, carrier.source_workspace_bytes(1)?)?;
        carrier.claims(StructuralPolicy::ReportOnly)?;
        Ok(carrier)
    }
    fn claims(&self, policy: StructuralPolicy) -> Result<AdaptedClaims, IoError> {
        use crate::decision_adapters::{ClaimProbability, SuppliedClaim, adapt_finite_scenarios};
        let profile = profile_from_wire(self.wire.profile.clone())?;
        let mut used = std::collections::HashSet::new();
        let mut claims = Vec::with_capacity(self.wire.atoms.len());
        for atom in &self.wire.atoms {
            let evidence = match atom.status.as_str() {
                "evaluated" => {
                    let mut laws = self.wire.laws.iter().filter(|law| law.atom_id == atom.id);
                    let law = laws.next().ok_or_else(|| {
                        carrier_refused("evaluated scenario lacks its original joint law")
                    })?;
                    if laws.next().is_some()
                        || law.metadata.identity.alignment != DrawAlignment::Joint
                    {
                        return Err(carrier_refused("scenario joint law is ambiguous or marginal"));
                    }
                    if law.metadata.identity.alignment != DrawAlignment::Joint {
                        return Err(carrier_refused(
                            "scenario law must retain explicit joint alignment",
                        ));
                    }
                    let source =
                        DistributionArtifact::new(law.metadata.clone(), law.draws.clone())?;
                    if atom.source_digest.as_deref() != Some(source_digest(&source).as_str()) {
                        return Err(carrier_refused("scenario source digest differs"));
                    }
                    used.insert(law.atom_id.as_str());
                    AtomEvidence::Evaluated(Box::new(source))
                }
                "unidentified" => AtomEvidence::Unidentified,
                "unevaluated" => AtomEvidence::Unevaluated(
                    atom.reason
                        .clone()
                        .ok_or_else(|| carrier_refused("unevaluated scenario lacks its reason"))?,
                ),
                _ => return Err(carrier_refused("unsupported scenario evidence state")),
            };
            let count =
                self.wire.completion_counts.iter().find(|(id, _)| *id == atom.id).map(|(_, n)| *n);
            if count.is_some() && atom.probability.is_some() {
                return Err(carrier_refused(
                    "completion counts cannot also be structure probabilities",
                ));
            }
            let probability = atom.probability.map_or_else(
                || count.map_or(ClaimProbability::Unspecified, ClaimProbability::CompletionCount),
                ClaimProbability::Genuine,
            );
            let support = profile
                .support
                .iter()
                .find(|(id, _)| *id == atom.id)
                .map(|(_, s)| s.clone())
                .ok_or_else(|| carrier_refused("scenario lacks its support declaration"))?;
            claims.push(SuppliedClaim { id: atom.id.clone(), probability, evidence, support });
        }
        if used.len() != self.wire.laws.len() {
            return Err(carrier_refused("joint source names an absent or unevaluated scenario"));
        }
        let adapted = adapt_finite_scenarios(claims, policy)
            .map_err(|e| refusal_error(&e, &e.to_refusal()))?;
        if to_cbor(&profile_to_wire(&adapted.profile))? != to_cbor(&self.wire.profile)?
            || adapted.completion_counts != self.wire.completion_counts
        {
            return Err(carrier_refused(
                "scenario profile/counts do not follow from the original claims",
            ));
        }
        Ok(adapted)
    }
    /// Evaluate only invariant, maximin or report-only policies. An unfinished
    /// scenario search or missing identification forces an incomplete verdict,
    /// retaining per-scenario values and all probability masses as reported.
    /// # Errors
    /// Unsupported policy, incompatible quantities, invalid claims or admission failure.
    pub fn evaluate(
        &self,
        contract: &AdmissibleDecisionContract,
        ctx: &ExecutionContext,
    ) -> Result<ScenarioDecision, IoError> {
        cancelled(ctx)?;
        contract.identity().map_err(|e| refusal_error(&e, &e.to_refusal()))?;
        if contract.contract.structural_policy == StructuralPolicy::BayesOverStructures
            || contract.contract.actions.len() > MAX_COMPILED_ROBUST_ITEMS
            || contract.contract.constraints.len() > MAX_COMPILED_ROBUST_ITEMS
        {
            return Err(carrier_refused(
                "compiled robust decisions require invariant, maximin or report-only policy and at most 256 actions and constraints",
            ));
        }
        let bytes = cbor_len(&*self.wire)?;
        let _import = if self.retention.belongs_to(&ctx.admission()) {
            self.retention.observe_current();
            None
        } else {
            Some(ctx.acquire_admission_product(0, bytes, 4)?)
        };
        let _workspace = ctx
            .acquire_admission(0, self.source_workspace_bytes(contract.contract.actions.len())?)?;
        let output = Arc::new(ctx.acquire_admission(0, self.output_bytes(contract)?)?);
        let adapted = self.claims(contract.contract.structural_policy)?;
        let mut result =
            evaluate_adapted(contract, &adapted).map_err(|e| refusal_error(&e, &e.to_refusal()))?;
        if result.robust.actions.iter().any(|a| {
            [a.range, a.supported_range]
                .into_iter()
                .flatten()
                .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite())
        }) || result.structural.actions.iter().any(|a| {
            a.per_atom.iter().flatten().any(|v| !v.is_finite())
                || a.weighted_value.is_some_and(|v| !v.is_finite())
        }) {
            return Err(carrier_refused(
                "scenario utility arithmetic is non-finite; no robust verdict is available",
            ));
        }
        let complete = self.complete()
            && result.structural.atoms.iter().all(|atom| {
                matches!(atom.status, crate::decision_structural::AtomStatus::Evaluated(_))
            });
        if !complete {
            let why = format!(
                "incomplete declared scenario set; search stop {}",
                self.search_stop().unwrap_or(
                    "none; unidentified, unevaluated, failed or nonexhaustive scenarios remain"
                )
            );
            result.structural.verdict = StructuralVerdict::InsufficientScience(why.clone());
            result.robust.verdict = RobustVerdict::InsufficientClaims(why);
        }
        result
            .robust
            .assumptions
            .push(format!("conditional on declared joint scenarios: {}", self.joint_structure()));
        cancelled(ctx)?;
        Ok(ScenarioDecision {
            decision: Arc::new(result),
            source_identity: Arc::from(self.identity()),
            joint_structure: Arc::from(self.joint_structure()),
            complete,
            stop: self.search_stop().map(Arc::from),
            _retention: output,
        })
    }
    /// Seal a terminal result with the exact compiler/source/contract identities.
    /// # Errors
    /// Evaluation or serialization failure.
    pub fn seal_decision(
        &self,
        contract: &AdmissibleDecisionContract,
        program_identity: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, IoError> {
        let result = self.evaluate(contract, ctx)?;
        let wire = CompiledScenarioResultWire {
            version: 1,
            program_identity: program_identity.into(),
            contract_identity: contract
                .identity()
                .map_err(|e| refusal_error(&e, &e.to_refusal()))?,
            source_identity: self.identity().into(),
            source: self.export(ctx)?,
            result: result_wire(&result.decision().robust),
        };
        let size = cbor_len(&wire)?;
        if size
            .checked_add(RESULT_MAGIC.len() as u64)
            .is_none_or(|n| n > 2 * MAX_SCENARIO_CARRIER_BYTES as u64)
        {
            return Err(IoError::TooLarge);
        }
        let _workspace = ctx.acquire_admission_product(0, size, 2)?;
        let body = to_cbor(&wire)?;
        let mut bytes = Vec::with_capacity(RESULT_MAGIC.len() + body.len());
        bytes.extend_from_slice(RESULT_MAGIC);
        bytes.extend_from_slice(&body);
        Ok(bytes)
    }
    /// Independently reconstruct the stored source and rerun the robust terminal
    /// against this separately retained compiler input. Bytes cannot resume a fit
    /// or manufacture a posterior/expected-utility interpretation.
    /// # Errors
    /// Any compiler, source, contract or numerical mismatch, or admission failure.
    pub fn replay_decision(
        &self,
        contract: &AdmissibleDecisionContract,
        program_identity: &str,
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<ScenarioDecision, IoError> {
        cancelled(ctx)?;
        if bytes.len() > 2 * MAX_SCENARIO_CARRIER_BYTES {
            return Err(IoError::TooLarge);
        }
        // Hostile CBOR may encode vectors of empty strings in one byte per
        // element. Charge decoded object/capacity overhead before allocation.
        let _workspace = ctx.acquire_admission_product(0, bytes.len() as u64, 64)?;
        let body = bytes.strip_prefix(RESULT_MAGIC).ok_or(IoError::BadMagic)?;
        let wire: CompiledScenarioResultWire = antecedent_io_core::from_cbor(body)?;
        if wire.version != 1
            || wire.program_identity != program_identity
            || wire.source_identity != self.identity()
            || wire.contract_identity
                != contract.identity().map_err(|e| refusal_error(&e, &e.to_refusal()))?
        {
            return Err(carrier_refused(
                "result belongs to a different compiled source or decision",
            ));
        }
        let source = Self::consume(&wire.source, self.identity(), ctx)?;
        let result = source.evaluate(contract, ctx)?;
        if to_cbor(&result_wire(&result.decision().robust))? != to_cbor(&wire.result)? {
            return Err(carrier_refused(
                "recomputed robust result differs from stored numbers or standing",
            ));
        }
        Ok(result)
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompiledScenarioResultWire {
    version: u32,
    program_identity: String,
    contract_identity: String,
    source_identity: String,
    source: Vec<u8>,
    result: ResultWire,
}
