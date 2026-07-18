//! Model bundle artifact encode/decode (DESIGN.md §24).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{CausalSchema, VERSION};
use causal_graph::Dag;
use causal_model::CompiledMechanismStore;
use serde::{Deserialize, Serialize};

use crate::analysis_wire::{
    DiagnosticWire, EffectEstimateWire, IdentificationResultWire, RefutationReportWire,
};
use crate::container::{
    ArtifactManifest, CompressPolicy, EncodedArtifact, SectionBytes, pack_section,
};
use crate::contrast_wire::ContrastBundleWire;
use crate::convert::{dag_from_wire, dag_to_wire, from_cbor, schema_from_wire, schema_to_wire, to_cbor};
use crate::discovery_wire::{
    DiscoveryHeaderWire, EdgeEvidenceWire, TemporalGraphWire,
};
use crate::error::IoError;
use crate::mechanism_wire::{
    MechanismStoreWire, ModelKindWire, mechanisms_from_wire, mechanisms_to_wire,
};
use crate::migrate::STABLE_FORMAT;
use crate::plan_wire::{
    ExecutionPerformanceWire, LogicalAnalysisPlanWire, PhysicalExecutionPlanWire,
};
use crate::posterior::CausalPosteriorWire;
use crate::provenance_wire::ProvenanceGraphWire;
use crate::query_wire::CausalQueryWire;
use crate::trace::AnalysisTraceWire;
use crate::wire::{ArtifactKind, DagWire, ProvenanceWire, SchemaWire, SemanticVersion};

/// Bundle header section.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelBundleHeaderWire {
    /// Model kind.
    pub model_kind: ModelKindWire,
    /// Optional human label.
    pub label: Option<String>,
}

/// Decoded model bundle (required sections always present).
#[derive(Clone, Debug)]
pub struct ModelBundle {
    /// Header.
    pub header: ModelBundleHeaderWire,
    /// Schema.
    pub schema: CausalSchema,
    /// Static DAG.
    pub dag: Dag,
    /// Mechanisms.
    pub mechanisms: CompiledMechanismStore,
    /// Optional contrasts.
    pub contrast: Option<ContrastBundleWire>,
    /// Optional query.
    pub query: Option<CausalQueryWire>,
    /// Optional analysis trace.
    pub analysis_trace: Option<AnalysisTraceWire>,
    /// Optional identification.
    pub identification: Option<IdentificationResultWire>,
    /// Optional estimate.
    pub estimate: Option<EffectEstimateWire>,
    /// Optional refutations.
    pub refutations: Option<Vec<RefutationReportWire>>,
    /// Optional logical plan.
    pub logical_plan: Option<LogicalAnalysisPlanWire>,
    /// Optional physical plan.
    pub physical_plan: Option<PhysicalExecutionPlanWire>,
    /// Optional performance.
    pub performance: Option<ExecutionPerformanceWire>,
    /// Optional diagnostics.
    pub diagnostics: Option<Vec<DiagnosticWire>>,
    /// Optional provenance graph.
    pub provenance: Option<ProvenanceGraphWire>,
    /// Optional posterior meta (draws live in sibling section when present).
    pub posterior_meta: Option<CausalPosteriorWire>,
    /// Optional posterior draws (f64 LE col-major).
    pub posterior_draws: Option<Vec<f64>>,
    /// Optional discovery header.
    pub discovery_header: Option<DiscoveryHeaderWire>,
    /// Optional discovery graph.
    pub discovery_graph: Option<TemporalGraphWire>,
    /// Optional discovery evidence.
    pub discovery_evidence: Option<Vec<EdgeEvidenceWire>>,
}

/// Builder inputs for encoding a model bundle.
#[derive(Clone, Debug)]
pub struct ModelBundleEncode<'a> {
    /// Header.
    pub header: ModelBundleHeaderWire,
    /// Schema.
    pub schema: &'a CausalSchema,
    /// DAG.
    pub dag: &'a Dag,
    /// Mechanisms.
    pub mechanisms: &'a CompiledMechanismStore,
    /// Artifact id.
    pub artifact_id: &'a str,
    /// Optional contrast.
    pub contrast: Option<&'a ContrastBundleWire>,
    /// Optional query.
    pub query: Option<&'a CausalQueryWire>,
    /// Optional analysis trace.
    pub analysis_trace: Option<&'a AnalysisTraceWire>,
    /// Optional identification.
    pub identification: Option<&'a IdentificationResultWire>,
    /// Optional estimate.
    pub estimate: Option<&'a EffectEstimateWire>,
    /// Optional refutations.
    pub refutations: Option<&'a [RefutationReportWire]>,
    /// Optional logical plan.
    pub logical_plan: Option<&'a LogicalAnalysisPlanWire>,
    /// Optional physical plan.
    pub physical_plan: Option<&'a PhysicalExecutionPlanWire>,
    /// Optional performance.
    pub performance: Option<&'a ExecutionPerformanceWire>,
    /// Optional diagnostics.
    pub diagnostics: Option<&'a [DiagnosticWire]>,
    /// Optional provenance.
    pub provenance: Option<&'a ProvenanceGraphWire>,
    /// Optional posterior meta + draws.
    pub posterior: Option<(&'a CausalPosteriorWire, &'a [f64])>,
    /// Optional discovery triple.
    pub discovery: Option<(&'a DiscoveryHeaderWire, &'a TemporalGraphWire, &'a [EdgeEvidenceWire])>,
}

/// Encode a model bundle artifact.
///
/// # Errors
///
/// CBOR / DAG conversion failures.
pub fn encode_model_bundle(input: ModelBundleEncode<'_>) -> Result<EncodedArtifact, IoError> {
    let mut descs = Vec::new();
    let mut sections = Vec::new();

    fn push_cbor(
        id: &str,
        logical_schema: &str,
        bytes: Vec<u8>,
        descs: &mut Vec<crate::wire::SectionDescriptor>,
        sections: &mut Vec<SectionBytes>,
    ) {
        let (mut desc, sec) = pack_section(id, "application/cbor", bytes, CompressPolicy::Auto);
        desc.logical_schema = logical_schema.into();
        descs.push(desc);
        sections.push(sec);
    }

    push_cbor(
        "bundle.header",
        "model_bundle.header.v1",
        to_cbor(&input.header)?,
        &mut descs,
        &mut sections,
    );
    push_cbor(
        "schema",
        "schema.v2",
        to_cbor(&schema_to_wire(input.schema))?,
        &mut descs,
        &mut sections,
    );
    push_cbor("dag", "dag.v1", to_cbor(&dag_to_wire(input.dag)?)?, &mut descs, &mut sections);
    push_cbor(
        "mechanisms",
        "mechanisms.v1",
        to_cbor(&mechanisms_to_wire(input.mechanisms)?)?,
        &mut descs,
        &mut sections,
    );

    if let Some(c) = input.contrast {
        push_cbor("contrast", "contrast.v1", to_cbor(c)?, &mut descs, &mut sections);
    }
    if let Some(q) = input.query {
        push_cbor("query", "query.v1", to_cbor(q)?, &mut descs, &mut sections);
    }
    if let Some(t) = input.analysis_trace {
        push_cbor("analysis.trace", "analysis.trace.v1", to_cbor(t)?, &mut descs, &mut sections);
    }
    if let Some(i) = input.identification {
        push_cbor("identification", "identification.v1", to_cbor(i)?, &mut descs, &mut sections);
    }
    if let Some(e) = input.estimate {
        push_cbor("estimate", "estimate.v1", to_cbor(e)?, &mut descs, &mut sections);
    }
    if let Some(r) = input.refutations {
        push_cbor(
            "refutations",
            "refutations.v1",
            to_cbor(&r.to_vec())?,
            &mut descs,
            &mut sections,
        );
    }
    if let Some(p) = input.logical_plan {
        push_cbor("logical_plan", "logical_plan.v1", to_cbor(p)?, &mut descs, &mut sections);
    }
    if let Some(p) = input.physical_plan {
        push_cbor("physical_plan", "physical_plan.v1", to_cbor(p)?, &mut descs, &mut sections);
    }
    if let Some(p) = input.performance {
        push_cbor("performance", "performance.v1", to_cbor(p)?, &mut descs, &mut sections);
    }
    if let Some(d) = input.diagnostics {
        push_cbor(
            "diagnostics",
            "diagnostics.v1",
            to_cbor(&d.to_vec())?,
            &mut descs,
            &mut sections,
        );
    }
    if let Some(p) = input.provenance {
        push_cbor("provenance", "provenance.v1", to_cbor(p)?, &mut descs, &mut sections);
    }
    if let Some((meta, draws)) = input.posterior {
        push_cbor("posterior.meta", "posterior.meta.v1", to_cbor(meta)?, &mut descs, &mut sections);
        let mut draw_bytes = Vec::with_capacity(draws.len() * 8);
        for &v in draws {
            draw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let (desc, sec) = pack_section(
            "posterior.draws",
            "application/octet-stream",
            draw_bytes,
            CompressPolicy::Auto,
        );
        descs.push(desc);
        sections.push(sec);
    }
    if let Some((h, g, e)) = input.discovery {
        push_cbor(
            "discovery.header",
            "discovery.header.v1",
            to_cbor(h)?,
            &mut descs,
            &mut sections,
        );
        push_cbor("discovery.graph", "discovery.graph.v1", to_cbor(g)?, &mut descs, &mut sections);
        push_cbor(
            "discovery.evidence",
            "discovery.evidence.v1",
            to_cbor(&e.to_vec())?,
            &mut descs,
            &mut sections,
        );
    }

    Ok(EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: STABLE_FORMAT,
            minimum_reader_version: STABLE_FORMAT,
            artifact_kind: ArtifactKind::ModelBundle,
            library_version: SemanticVersion::from_crate_version(VERSION)?,
            artifact_id: input.artifact_id.into(),
            sections: descs,
            provenance: ProvenanceWire { note: "model_bundle".into() },
        },
        sections,
    })
}

/// Decode a model bundle artifact.
///
/// # Errors
///
/// Missing required sections or CBOR/DAG failures.
pub fn decode_model_bundle(artifact: &EncodedArtifact) -> Result<ModelBundle, IoError> {
    if artifact.manifest.artifact_kind != ArtifactKind::ModelBundle {
        return Err(IoError::Convert(format!(
            "expected ModelBundle, got {:?}",
            artifact.manifest.artifact_kind
        )));
    }
    let find = |id: &str| -> Result<&SectionBytes, IoError> {
        artifact
            .sections
            .iter()
            .find(|s| s.id == id)
            .ok_or_else(|| IoError::Convert(format!("missing section `{id}`")))
    };
    let opt = |id: &str| artifact.sections.iter().find(|s| s.id == id);

    let header: ModelBundleHeaderWire = from_cbor(&find("bundle.header")?.data)?;
    let schema_wire: SchemaWire = from_cbor(&find("schema")?.data)?;
    let schema = schema_from_wire(&schema_wire)?;
    let dag_wire: DagWire = from_cbor(&find("dag")?.data)?;
    let dag = dag_from_wire(&dag_wire)?;
    let mech_wire: MechanismStoreWire = from_cbor(&find("mechanisms")?.data)?;
    let mechanisms = mechanisms_from_wire(&mech_wire)?;

    let contrast = opt("contrast").map(|s| from_cbor(&s.data)).transpose()?;
    let query = opt("query").map(|s| from_cbor(&s.data)).transpose()?;
    let analysis_trace = opt("analysis.trace").map(|s| from_cbor(&s.data)).transpose()?;
    let identification = opt("identification").map(|s| from_cbor(&s.data)).transpose()?;
    let estimate = opt("estimate").map(|s| from_cbor(&s.data)).transpose()?;
    let refutations = opt("refutations").map(|s| from_cbor(&s.data)).transpose()?;
    let logical_plan = opt("logical_plan").map(|s| from_cbor(&s.data)).transpose()?;
    let physical_plan = opt("physical_plan").map(|s| from_cbor(&s.data)).transpose()?;
    let performance = opt("performance").map(|s| from_cbor(&s.data)).transpose()?;
    let diagnostics = opt("diagnostics").map(|s| from_cbor(&s.data)).transpose()?;
    let provenance = opt("provenance").map(|s| from_cbor(&s.data)).transpose()?;

    let (posterior_meta, posterior_draws) =
        if let (Some(meta_sec), Some(draw_sec)) = (opt("posterior.meta"), opt("posterior.draws")) {
            let meta: CausalPosteriorWire = from_cbor(&meta_sec.data)?;
            let draws = decode_f64_le(draw_sec)?;
            (Some(meta), Some(draws))
        } else {
            (None, None)
        };

    let discovery_header = opt("discovery.header").map(|s| from_cbor(&s.data)).transpose()?;
    let discovery_graph = opt("discovery.graph").map(|s| from_cbor(&s.data)).transpose()?;
    let discovery_evidence = opt("discovery.evidence").map(|s| from_cbor(&s.data)).transpose()?;

    Ok(ModelBundle {
        header,
        schema,
        dag,
        mechanisms,
        contrast,
        query,
        analysis_trace,
        identification,
        estimate,
        refutations,
        logical_plan,
        physical_plan,
        performance,
        diagnostics,
        provenance,
        posterior_meta,
        posterior_draws,
        discovery_header,
        discovery_graph,
        discovery_evidence,
    })
}

fn decode_f64_le(sec: &SectionBytes) -> Result<Vec<f64>, IoError> {
    if sec.data.len() % 8 != 0 {
        return Err(IoError::Convert("draws not multiple of 8".into()));
    }
    let mut out = Vec::with_capacity(sec.data.len() / 8);
    for chunk in sec.data.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        out.push(f64::from_le_bytes(buf));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_graph::DenseNodeId;
    use causal_model::{CompiledMechanismStore, MechanismSlot};

    use super::*;

    #[test]
    fn model_bundle_round_trip() {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let mechanisms = CompiledMechanismStore {
            slots: vec![
                MechanismSlot::Constant { value: 0.0 },
                MechanismSlot::LinearGaussian {
                    intercept: 0.1,
                    coeffs: std::sync::Arc::from([0.5f64]),
                    sigma: 1.0,
                },
            ]
            .into(),
        };
        let query = CausalQueryWire::AverageEffect {
            treatment: 0,
            outcome: 1,
            effect_modifiers: vec![],
            control: crate::query_wire::InterventionWire::Set {
                variable: 0,
                value: crate::query_wire::ValueWire::Float64(0.0),
            },
            active: crate::query_wire::InterventionWire::Set {
                variable: 0,
                value: crate::query_wire::ValueWire::Float64(1.0),
            },
            target_population: crate::query_wire::TargetPopulationWire::AllObserved,
        };
        let art = encode_model_bundle(ModelBundleEncode {
            header: ModelBundleHeaderWire {
                model_kind: ModelKindWire::Scm,
                label: Some("test".into()),
            },
            schema: &schema,
            dag: &dag,
            mechanisms: &mechanisms,
            artifact_id: "bundle-test",
            contrast: None,
            query: Some(&query),
            analysis_trace: None,
            identification: None,
            estimate: None,
            refutations: None,
            logical_plan: None,
            physical_plan: None,
            performance: None,
            diagnostics: None,
            provenance: None,
            posterior: None,
            discovery: None,
        })
        .unwrap();
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let decoded = EncodedArtifact::read_from(buf.as_slice()).unwrap();
        let bundle = decode_model_bundle(&decoded).unwrap();
        assert_eq!(bundle.schema.len(), 2);
        assert_eq!(bundle.dag.node_count(), 2);
        assert!(matches!(bundle.mechanisms.slots[1], MechanismSlot::LinearGaussian { .. }));
        assert!(bundle.query.is_some());
        let _ = VariableId::from_raw(0);
    }
}
