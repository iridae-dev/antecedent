//! Native system-to-family source bridge.

use super::{
    A, CanonicalWriter, CyclicArtifactError, CyclicEvidence, CyclicEvidenceSchema,
    CyclicEvidenceSelector, CyclicEvidenceSource, CyclicFamilyReport, ExecutionContext, R,
    cyclic_decode_admission, invalid,
};
use serde::Serialize;
use std::sync::Arc;

/// Explicit provenance of the native system-to-family source bridge.
/// Both identities are canonical native request identities, not replacement labels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CyclicSystemBridgeProvenance {
    /// Identity of the original executed/independently verified system.
    pub source_identity: String,
    /// Identity of the corresponding compiled family request and contract.
    pub family_identity: String,
    /// Domain-separated binding of those two source identities.
    pub identity: String,
}

pub(super) fn system_bridge_provenance(
    source_identity: &str,
    family_identity: &str,
) -> CyclicSystemBridgeProvenance {
    let mut bytes = Vec::with_capacity(source_identity.len() + family_identity.len());
    bytes.extend_from_slice(source_identity.as_bytes());
    bytes.extend_from_slice(family_identity.as_bytes());
    CyclicSystemBridgeProvenance {
        source_identity: source_identity.into(),
        family_identity: family_identity.into(),
        identity: CanonicalWriter::digest_bytes_hex(
            "antecedent.cyclic_system_family_bridge.v1",
            &bytes,
        ),
    }
}

/// A compiled family view of the exact original system output, with no new execution.
/// It retains metadata and the shared report; it contains no reconstructed study.
#[derive(Clone, Debug)]
pub struct CyclicSystemFamilyBridge {
    program: Arc<antecedent_io::cyclic_family_artifact::CompiledCyclicFamilyProgram>,
    report: Arc<CyclicFamilyReport>,
    contract: Arc<antecedent_io::contract_section::cyclic::CyclicContractWire>,
    provenance: CyclicSystemBridgeProvenance,
}

impl CyclicSystemFamilyBridge {
    fn from_system(
        request: &antecedent_io::cyclic_system_artifact::CyclicExecutionRequest,
        report: Arc<antecedent_io::cyclic_system_artifact::CyclicExecutionReport>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        let program =
            Arc::new(antecedent_io::cyclic_family_artifact::CompiledCyclicFamilyProgram::compile(
                &R::System { request: request.clone() },
                ctx,
            )?);
        let report = Arc::new(program.bind_system_execution(report, ctx)?);
        let A::System(system) = &report.answer else {
            return Err(invalid("system bridge answer"));
        };
        let provenance = system_bridge_provenance(&system.identity, &report.identity);
        let contract = Arc::new(program.contract()?);
        Ok(Self { program, report, contract, provenance })
    }

    /// Exact family report; its System answer shares the original report Arc.
    #[must_use]
    pub fn report(&self) -> &CyclicFamilyReport {
        &self.report
    }

    /// Native compiled contract of the corresponding family source.
    #[must_use]
    pub fn contract_metadata(
        &self,
    ) -> &antecedent_io::contract_section::cyclic::CyclicContractWire {
        &self.contract
    }

    /// The original source ID, corresponding family ID, and explicit bridge ID.
    #[must_use]
    pub fn provenance(&self) -> &CyclicSystemBridgeProvenance {
        &self.provenance
    }

    /// Encode this exact source bridge through the existing family codec.
    /// # Errors
    /// Cancellation or encoding/resource failure.
    pub fn export(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, CyclicArtifactError> {
        self.program.encode_execution(&self.report, ctx)
    }

    /// Project compatible native outputs without executing or exporting the source.
    /// # Errors
    /// Invalid schema/selectors, cancellation or resource refusal.
    pub fn evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        CyclicEvidence::build(
            self.program.request(),
            &self.report,
            CyclicEvidenceSource::SystemBridge(Arc::new(self.clone())),
            schema,
            selectors,
            ctx,
        )
    }
}

impl super::super::cyclic::ExecutedCyclicSystem {
    /// Retain compatible family metadata around this exact system execution.
    /// # Errors
    /// Invalid family metadata, cancellation or resource refusal.
    pub fn family_bridge(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<CyclicSystemFamilyBridge>, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CyclicArtifactError::Cancelled);
        }
        let mut cache =
            self.family_bridge.lock().map_err(|_| invalid("system bridge cache lock poisoned"))?;
        if let Some(bridge) = cache.as_ref() {
            return Ok(Arc::clone(bridge));
        }
        let bridge = Arc::new(CyclicSystemFamilyBridge::from_system(
            self.prepared().request(),
            Arc::clone(self.shared_report()),
            ctx,
        )?);
        *cache = Some(Arc::clone(&bridge));
        Ok(bridge)
    }

    /// Compose the actual retained system output through its explicit native bridge.
    /// # Errors
    /// Invalid schema/selectors, cancellation or resource refusal.
    pub fn evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        self.family_bridge(ctx)?.evidence(schema, selectors, ctx)
    }
}

/// Independently verified system source with no reconstructed `PreparedStudy`.
#[derive(Clone, Debug)]
pub struct ConsumedCyclicSystem {
    request: Arc<antecedent_io::cyclic_system_artifact::CyclicExecutionRequest>,
    pub(super) report: Arc<antecedent_io::cyclic_system_artifact::CyclicExecutionReport>,
    bytes: Arc<[u8]>,
    _retention: Arc<antecedent_core::execution::AdmissionLease>,
    family_bridge: Arc<std::sync::Mutex<Option<Arc<CyclicSystemFamilyBridge>>>>,
}

impl ConsumedCyclicSystem {
    /// Invoke the independent system consumer exactly once and retain its source.
    /// # Errors
    /// Framing, identity, replay, cancellation or resource failure.
    pub fn consume(
        bytes: &[u8],
        expected_identity: &str,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        let retention = Arc::new(cyclic_decode_admission(bytes, ctx)?);
        let (request, report) =
            antecedent_io::cyclic_system_artifact::consume_cyclic_artifact_with_request(
                bytes,
                expected_identity,
                ctx,
            )?;
        Ok(Self {
            request: Arc::new(request),
            report: Arc::new(report),
            bytes: Arc::from(bytes),
            _retention: retention,
            family_bridge: Arc::default(),
        })
    }

    /// Exact independently replayed output, borrowing native immutable storage.
    #[must_use]
    pub fn report(&self) -> &antecedent_io::cyclic_system_artifact::CyclicExecutionReport {
        &self.report
    }

    /// Independently verified original declaration, without a live study.
    #[must_use]
    pub fn request(&self) -> &antecedent_io::cyclic_system_artifact::CyclicExecutionRequest {
        &self.request
    }

    /// Original verified source identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.report.identity
    }

    /// Original verified bytes; exporting them never reconstructs or executes a study.
    #[must_use]
    pub fn artifact(&self) -> &[u8] {
        &self.bytes
    }

    /// Compile corresponding family metadata around the verified report once.
    /// # Errors
    /// Invalid metadata, cancellation or resource failure.
    pub fn family_bridge(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<CyclicSystemFamilyBridge>, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CyclicArtifactError::Cancelled);
        }
        let mut cache = self
            .family_bridge
            .lock()
            .map_err(|_| invalid("consumed system bridge cache lock poisoned"))?;
        if let Some(bridge) = cache.as_ref() {
            return Ok(Arc::clone(bridge));
        }
        let bridge = Arc::new(CyclicSystemFamilyBridge::from_system(
            &self.request,
            Arc::clone(&self.report),
            ctx,
        )?);
        *cache = Some(Arc::clone(&bridge));
        Ok(bridge)
    }

    /// Project verified output directly without another replay, fit or solve.
    /// # Errors
    /// Invalid schema/selectors, cancellation or resource failure.
    pub fn evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        self.family_bridge(ctx)?.evidence(schema, selectors, ctx)
    }
}
