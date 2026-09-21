//! Durable structural transport claims, including independently checked negatives.
use crate::error::convert_err as err;
use crate::{
    IoError,
    contract_section::{
        AssumptionSlotWire, IdentificationSlotWire, ObligationSectionWire, ReasoningSectionWire,
        SlotSectionWire,
    },
    transport_proof::TransportProofWire,
};
use antecedent_core::{ExecutionContext, IdentityDomain, NodeRef, VariableId};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    ClassicalTransportQuery, ClassicalTransportResult, MetaSource, SidLimits,
    sid::{SHedgeCertificate, SHedgeRecord},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
/// Versioned structural outcome; decoding does not convey scientific authority.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CertificateOutcome {
    /// Positive expression and all local premises.
    Identified(TransportProofWire),
    /// An obstruction, requiring verification against every declared source.
    ProvenNonTransportable(SHedgeRecord),
    /// A conservative or computationally incomplete outcome, never a negative proof.
    NotCertified {
        /// Stable explanation, including exhaustion when applicable.
        reason: String,
    },
}
/// Portable query, graph, source collection and four-slot structural claim.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportCertificateWire {
    /// Independent artifact schema version.
    pub version: u32,
    /// Required semantics; unknown entries prevent scientific acceptance.
    pub required_features: Vec<String>,
    /// Original graph coordinates.
    pub nodes: Vec<u32>,
    /// Dense graph edges.
    pub directed: Vec<(u32, u32)>,
    /// Dense confounding edges.
    pub bidirected: Vec<(u32, u32)>,
    /// Primary diagram selections (all meta diagrams are in sources).
    pub selections: Vec<u32>,
    /// Joint outcome coordinates.
    pub outcomes: Vec<u32>,
    /// Joint intervention coordinates.
    pub treatments: Vec<u32>,
    /// Primary source for classical compatibility.
    pub source: String,
    /// Target population.
    pub target: String,
    /// Complete meta source collection, or empty for classical sID.
    pub sources: Vec<MetaSource>,
    /// Untrusted outcome to check.
    pub outcome: CertificateOutcome,
    /// Independently reconstructed four reasoning slots.
    pub reasoning: ReasoningSectionWire,
}
/// Reconstruct the four structural reasoning slots from the verified outcome kind.
#[must_use]
pub fn structural_reasoning(outcome: &CertificateOutcome) -> ReasoningSectionWire {
    let (status, identified, unidentified, incomplete) = match outcome {
        CertificateOutcome::Identified(_) => ("nonparametrically_identified", 1., 0., 0.),
        CertificateOutcome::ProvenNonTransportable(_) => ("proven_non_transportable", 0., 1., 0.),
        CertificateOutcome::NotCertified { .. } => ("not_certified", 0., 0., 1.),
    };
    ReasoningSectionWire {
        identification: SlotSectionWire {
            value: Some(IdentificationSlotWire {
                status: status.into(),
                identified_mass: identified,
                unidentified_mass: unidentified,
                unevaluable_mass: 0.,
                incomplete_search_mass: incomplete,
                full_mass_scope: true,
                search_capped: matches!(outcome, CertificateOutcome::NotCertified { .. }),
                weight_basis: None,
            }),
            unavailable: None,
        },
        support: SlotSectionWire { value: None, unavailable: Some("providers_not_bound".into()) },
        uncertainty: SlotSectionWire { value: None, unavailable: Some("not_estimated".into()) },
        assumptions: SlotSectionWire {
            value: Some(AssumptionSlotWire {
                obligations: vec![ObligationSectionWire {
                    id: "transport.population_selection_graph".into(),
                    scope: "identification".into(),
                    kind: "graph_assumption".into(),
                    status: "declared_not_empirically_verified".into(),
                }],
            }),
            unavailable: None,
        },
    }
}
impl TransportCertificateWire {
    /// Encode a native structural result under explicit immutable scientific inputs.
    /// # Errors
    /// Encoding failure or a result that does not verify against its enclosing inputs.
    pub fn from_result(
        diagram: &SelectionDiagram,
        query: &ClassicalTransportQuery,
        sources: Vec<MetaSource>,
        result: &ClassicalTransportResult,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let outcome = match result {
            ClassicalTransportResult::Identified(proof) => {
                CertificateOutcome::Identified(TransportProofWire::from_checked(proof)?)
            }
            ClassicalTransportResult::ProvenNonTransportable(w) => {
                CertificateOutcome::ProvenNonTransportable(w.to_record())
            }
            ClassicalTransportResult::NotCertified => {
                CertificateOutcome::NotCertified { reason: "bounded_search_not_certified".into() }
            }
        };
        let edges = crate::admg_to_wire(diagram.causal_graph())?;
        let wire = Self {
            version: 1,
            required_features: vec!["checked_transport_certificate_v1".into()],
            nodes: diagram
                .causal_graph()
                .nodes()
                .iter()
                .map(|n| match n {
                    NodeRef::Static(v) => Ok(v.raw()),
                    _ => Err(err("static graph required")),
                })
                .collect::<Result<_, _>>()?,
            directed: edges.directed,
            bidirected: edges.bidirected,
            selections: diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            outcomes: query.outcomes.iter().map(|v| v.raw()).collect(),
            treatments: query.treatments.iter().map(|v| v.raw()).collect(),
            source: query.source.to_string(),
            target: query.target.to_string(),
            sources,
            reasoning: structural_reasoning(&outcome),
            outcome,
        };
        wire.check(limits, ctx)?;
        Ok(wire)
    }
    /// Independently consume embedded premises. Never identify, fit, or fetch providers.
    /// # Errors
    /// Unknown semantics, malformed graph/query, or changed proof/witness/slots.
    pub fn check(
        &self,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<(SelectionDiagram, ClassicalTransportQuery, ClassicalTransportResult), IoError>
    {
        if self.version != 1
            || self.required_features != ["checked_transport_certificate_v1"]
            || self.reasoning != structural_reasoning(&self.outcome)
            || ctx.cancellation.is_cancelled()
            || self.nodes.len() > limits.steps
        {
            return Err(err("unsupported or invalid transport certificate"));
        }
        let mut graph = Admg::empty();
        for v in &self.nodes {
            graph.add_node(NodeRef::Static(VariableId::from_raw(*v))).map_err(err)?;
        }
        for (a, b) in &self.directed {
            graph
                .insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b))
                .map_err(err)?;
        }
        for (a, b) in &self.bidirected {
            graph
                .insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b))
                .map_err(err)?;
        }
        graph.validate().map_err(err)?;
        let diagram = SelectionDiagram::try_new(
            graph,
            self.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(err)?;
        let query = ClassicalTransportQuery {
            outcomes: self.outcomes.iter().copied().map(VariableId::from_raw).collect(),
            treatments: self.treatments.iter().copied().map(VariableId::from_raw).collect(),
            source: Arc::from(self.source.as_str()),
            target: Arc::from(self.target.as_str()),
        };
        if query.outcomes.is_empty()
            || query.source == query.target
            || query.source.is_empty()
            || query.target.is_empty()
            || self.outcomes.iter().any(|v| !self.nodes.contains(v) || self.treatments.contains(v))
            || self.treatments.iter().any(|v| !self.nodes.contains(v))
        {
            return Err(err("invalid certificate query"));
        }
        let unique =
            |xs: &[u32]| xs.iter().collect::<std::collections::BTreeSet<_>>().len() == xs.len();
        if !unique(&self.nodes)
            || !unique(&self.outcomes)
            || !unique(&self.treatments)
            || !unique(&self.selections)
            || self.sources.windows(2).any(|pair| pair[0].population >= pair[1].population)
            || self
                .sources
                .first()
                .is_some_and(|s| s.population != self.source || s.selections != self.selections)
            || self.sources.iter().any(|s| {
                s.population.trim().is_empty()
                    || s.population == self.target
                    || !unique(&s.selections)
                    || s.selections.windows(2).any(|w| w[0] >= w[1])
                    || s.selections.iter().any(|v| !self.nodes.contains(v))
            })
        {
            return Err(err("invalid certificate coordinates or source collection"));
        }
        let result = match &self.outcome {
            CertificateOutcome::Identified(proof) => {
                if proof.proof.sources != self.sources {
                    return Err(err("certificate source mismatch"));
                }
                ClassicalTransportResult::Identified(Box::new(
                    proof.check(&diagram, &query, limits, ctx)?,
                ))
            }
            CertificateOutcome::ProvenNonTransportable(record) => {
                if record.sources != self.sources {
                    return Err(err("witness source mismatch"));
                }
                ClassicalTransportResult::ProvenNonTransportable(
                    SHedgeCertificate::from_record_checked(record.clone(), &diagram, &query, ctx)
                        .map_err(err)?,
                )
            }
            CertificateOutcome::NotCertified { .. } => ClassicalTransportResult::NotCertified,
        };
        Ok((diagram, query, result))
    }
    /// Sectioned, checksummed storage of this structural claim.
    /// # Errors
    /// Container or serialization failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        use crate::container::{ArtifactManifest, CompressPolicy, EncodedArtifact, pack_section};
        use crate::wire::{ArtifactKind, FormatVersion, ProvenanceWire, SemanticVersion};
        let (desc, section) = pack_section(
            "transport_certificate_v1",
            "application/cbor",
            crate::to_cbor(self)?,
            CompressPolicy::Never,
        );
        let artifact=EncodedArtifact {manifest:ArtifactManifest{format_version:FormatVersion{major:1,minor:0},minimum_reader_version:FormatVersion{major:1,minor:0},artifact_kind:ArtifactKind::Other("transport_certificate".into()),library_version:SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION")).map_err(err)?,artifact_id:crate::identity::digest_wire(IdentityDomain::TransportCertificate,self)?.to_hex(),sections:vec![desc],provenance:ProvenanceWire{note:"checked structural transport claim; no empirical support or sampling coverage".into()}},sections:vec![section]};
        let mut bytes = vec![];
        artifact.write_to(&mut bytes)?;
        Ok(bytes)
    }
    /// Read and verify a structural artifact without data access.
    /// # Errors
    /// Changed identities, unknown features, corrupt premises, or exhausted resources.
    pub fn consume(
        bytes: &[u8],
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if ctx.memory.hard_limit_bytes.is_some_and(|n| bytes.len() as u64 > n)
            || ctx.cancellation.is_cancelled()
        {
            return Err(err("certificate budget/cancellation"));
        }
        let artifact = read_bounded_transport_artifact(bytes, ctx)?;
        if artifact.manifest.artifact_kind
            != crate::wire::ArtifactKind::Other("transport_certificate".into())
            || artifact.sections.len() != 1
            || artifact.sections[0].id != "transport_certificate_v1"
        {
            return Err(err("unsupported transport certificate container"));
        }
        let wire: Self = crate::from_cbor(&artifact.sections[0].data)?;
        if artifact.manifest.artifact_id
            != crate::identity::digest_wire(IdentityDomain::TransportCertificate, &wire)?.to_hex()
        {
            return Err(err("certificate identity mismatch"));
        }
        wire.check(limits, ctx)?;
        Ok(wire)
    }
}

/// Read a transport container after checking declared decompression and decode budgets.
/// # Errors
/// Unknown container/artifact versions, cancellation, or memory-bound violation.
pub fn read_bounded_transport_artifact(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<crate::container::EncodedArtifact, IoError> {
    if bytes.len() < 16 || &bytes[..8] != crate::container::MAGIC || ctx.cancellation.is_cancelled()
    {
        return Err(err("invalid transport container/cancellation"));
    }
    if ctx.memory.hard_limit_bytes.is_some_and(|limit| bytes.len() as u64 > limit) {
        return Err(err("transport artifact decode memory budget"));
    }
    let length = u32::from_le_bytes(bytes[12..16].try_into().map_err(err)?) as usize;
    let end = 16usize.checked_add(length).ok_or_else(|| err("transport manifest overflow"))?;
    if length > 16 * 1024 * 1024 || end > bytes.len() {
        return Err(err("transport manifest size"));
    }
    let manifest: crate::container::ArtifactManifest = crate::from_cbor(&bytes[16..end])?;
    if manifest.format_version.major != 1
        || manifest.minimum_reader_version.major > 1
        || manifest.minimum_reader_version.minor > 0
    {
        return Err(err("unsupported transport artifact version"));
    }
    let logical = manifest
        .sections
        .iter()
        .try_fold(bytes.len() as u64, |n, s| n.checked_add(s.uncompressed_size.checked_mul(4)?))
        .ok_or_else(|| err("transport artifact memory overflow"))?;
    if ctx.memory.hard_limit_bytes.is_some_and(|limit| logical > limit) {
        return Err(err("transport artifact decode memory budget"));
    }
    crate::container::EncodedArtifact::read_from(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn negative_meta_certificate_rejects_altered_scope_and_witness() {
        let v = VariableId::from_raw;
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let sources = vec![
            MetaSource { population: "a".into(), selections: vec![1] },
            MetaSource { population: "b".into(), selections: vec![1] },
        ];
        let meta = antecedent_identify::MetaTransportQuery {
            outcomes: Arc::from([v(1)]),
            treatments: Arc::from([v(0)]),
            target: Arc::from("target"),
            sources: sources.clone(),
        };
        let ctx = ExecutionContext::for_tests(0);
        let limits = SidLimits::default();
        let result =
            antecedent_identify::identify_meta_transport(&graph, &meta, limits, &ctx).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [v(1)]).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: meta.outcomes,
            treatments: meta.treatments,
            source: Arc::from("a"),
            target: meta.target,
        };
        let wire =
            TransportCertificateWire::from_result(&diagram, &query, sources, &result, limits, &ctx)
                .unwrap();
        TransportCertificateWire::consume(&wire.export().unwrap(), limits, &ctx).unwrap();
        let mut altered = wire.clone();
        altered.target = "other".into();
        assert!(altered.check(limits, &ctx).is_err());
        let mut altered = wire.clone();
        if let CertificateOutcome::ProvenNonTransportable(w) = &mut altered.outcome {
            w.smaller.nodes = vec![0];
        }
        assert!(altered.check(limits, &ctx).is_err());
        let mut altered = wire.clone();
        if let CertificateOutcome::ProvenNonTransportable(w) = &mut altered.outcome {
            w.evidence_setting = "limited_experiments".into();
        }
        assert!(altered.check(limits, &ctx).is_err());
        let mut altered = wire.clone();
        altered.sources[1].selections.clear();
        assert!(altered.check(limits, &ctx).is_err());
        let mut incomplete = wire;
        incomplete.outcome =
            CertificateOutcome::NotCertified { reason: "identification_budget".into() };
        incomplete.reasoning = structural_reasoning(&incomplete.outcome);
        if let Ok(path) = std::env::var("ANTECEDENT_WRITE_NOTCERT_FIXTURE") {
            let mut framed = b"ANTECEDENT-TRANSPORT-CERTIFICATE\x01".to_vec();
            framed.extend(crate::to_cbor(&(vec!["x", "y"], incomplete.export().unwrap())).unwrap());
            std::fs::write(path, framed).unwrap();
        }
        let stored =
            TransportCertificateWire::consume(&incomplete.export().unwrap(), limits, &ctx).unwrap();
        assert!(matches!(
            stored.check(limits, &ctx).unwrap().2,
            ClassicalTransportResult::NotCertified
        ));
    }
}
