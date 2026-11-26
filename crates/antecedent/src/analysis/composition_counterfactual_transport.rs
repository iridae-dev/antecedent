//! Replayed exact and sampled counterfactual transport at the bundle boundary.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::{QuantityCondition, QuantityRole, ScientificQuantity};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_design::composition_verifiers::{
    ArtifactDescription, FACT_COORDINATES, FACT_MEAN_SOURCE, TRUST_NATIVE, coordinate_id,
};
use antecedent_design::decision_artifact::bound_mean_result_to_bytes;
use antecedent_design::decision_contract::DecisionContract;
use antecedent_design::decision_eval::{MeanEstimate, MeanSource, evaluate_contract_on_means};
use antecedent_io::counterfactual_transport_artifact::{
    ConsumedCounterfactualTransportOutcome, CounterfactualTransportConsumeLimits,
    consume_counterfactual_transport_with_limits,
};
use antecedent_io::sampled_counterfactual_transport_artifact::{
    SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE, SampledCounterfactualTransportConsumeLimits,
    SampledCounterfactualTransportExpectedIdentity,
    consume_sampled_counterfactual_transport_with_limits,
};
use antecedent_io_core::counterfactual_transport_wire::{
    COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE, CounterfactualTransportExpectedIdentity,
    CounterfactualTransportQueryWire,
};
use serde::{Deserialize, Serialize};

use super::composition::{
    BundleStage, FACT_EVIDENCE_DIGEST, FACT_LAW, FACT_REQUEST, FACT_SNAPSHOT, FACT_TRUST,
    LAW_MEAN_ONLY, NodeFailure, NodeKind, NodeVerifier, UpstreamNode, VerifiedNode,
};

struct TransportMean {
    query: CounterfactualTransportQueryWire,
    variable_names: Vec<String>,
    premises: String,
    data: String,
    probability: f64,
    conditioning_mass_checked: bool,
    sampling: Option<String>,
}

fn failure(stage: BundleStage, error: impl ToString) -> NodeFailure {
    NodeFailure { stage, reason: error.to_string() }
}

fn digest<T: Serialize>(domain: &str, value: &T) -> Result<String, NodeFailure> {
    let bytes = antecedent_io::to_cbor(value)
        .map_err(|error| failure(BundleStage::TamperedQuantity, error))?;
    Ok(CanonicalWriter::digest_bytes_hex(domain, &bytes))
}

#[derive(Deserialize)]
struct TransportFeature {
    required_features: [String; 1],
}

fn feature(bytes: &[u8]) -> Option<TransportFeature> {
    if bytes.len() > 64 * 1024 * 1024 {
        return None;
    }
    let feature = antecedent_io::from_cbor::<TransportFeature>(bytes).ok()?;
    matches!(
        feature.required_features[0].as_str(),
        COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE
            | SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE
    )
    .then_some(feature)
}

pub(super) fn detects(bytes: &[u8]) -> bool {
    feature(bytes).is_some()
}

fn read(bytes: &[u8], ctx: &ExecutionContext) -> Result<TransportMean, NodeFailure> {
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(failure(
            BundleStage::Oversized,
            "counterfactual transport artifact exceeds its bound",
        ));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(failure(
            BundleStage::ExecutionCancelled,
            "counterfactual transport replay was cancelled",
        ));
    }
    let feature = feature(bytes).ok_or_else(|| {
        failure(BundleStage::SwappedEvidence, "expected a counterfactual transport artifact")
    })?;
    match feature.required_features[0].as_str() {
        COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE => {
            let consumed = consume_counterfactual_transport_with_limits(
                bytes,
                &CounterfactualTransportExpectedIdentity::default(),
                CounterfactualTransportConsumeLimits::default(),
                ctx,
            )
            .map_err(|error| failure(BundleStage::TamperedQuantity, error))?;
            let ConsumedCounterfactualTransportOutcome::Identified(result) = consumed.outcome
            else {
                return Err(failure(
                    BundleStage::UnsupportedLaw,
                    "a transport obstruction does not supply a point",
                ));
            };
            Ok(TransportMean {
                query: consumed.wire.query,
                variable_names: consumed.wire.variable_names,
                premises: consumed.wire.premises_digest,
                data: consumed.wire.data_digest,
                probability: result.point.probability,
                conditioning_mass_checked: result.point.conditioning_mass_checked,
                sampling: None,
            })
        }
        SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE => {
            let consumed = consume_sampled_counterfactual_transport_with_limits(
                bytes,
                &SampledCounterfactualTransportExpectedIdentity::default(),
                SampledCounterfactualTransportConsumeLimits::default(),
                ctx,
            )
            .map_err(|error| failure(BundleStage::TamperedQuantity, error))?;
            let Some(point) = consumed.estimate.point else {
                return Err(failure(
                    BundleStage::UnsupportedLaw,
                    "the sampled envelope does not supply a plug-in point",
                ));
            };
            let sampling = serde_json::to_string(&consumed.wire.estimate)
                .map_err(|error| failure(BundleStage::TamperedQuantity, error))?;
            Ok(TransportMean {
                query: consumed.wire.query,
                variable_names: consumed.wire.variable_names,
                premises: consumed.wire.premises_digest,
                data: consumed.wire.data_digest,
                probability: point.probability,
                conditioning_mass_checked: point.conditioning_mass_checked,
                sampling: Some(sampling),
            })
        }
        _ => Err(failure(
            BundleStage::SwappedEvidence,
            "expected an exact or sampled counterfactual transport artifact",
        )),
    }
}

impl TransportMean {
    fn identity(&self) -> String {
        let mut writer = CanonicalWriter::new("antecedent.composition.counterfactual_transport.v1");
        writer.text(&self.premises).text(&self.data);
        writer.finish_hex()
    }

    fn source(&self) -> Result<MeanSource, NodeFailure> {
        let event = &self.query.event;
        let variable_id = digest(
            "antecedent.counterfactual_transport.event.v1",
            &(&event.event, &self.variable_names),
        )?;
        let regime_id = digest(
            "antecedent.counterfactual_transport.regime.v1",
            &(&event.worlds, &event.coupling),
        )?;
        let mut conditioning = event
            .given
            .iter()
            .map(|(world, variable, value)| QuantityCondition {
                variable_id: format!("world:{world}:variable:{variable}"),
                value_id: format!("{value:016x}"),
            })
            .collect::<Vec<_>>();
        conditioning.sort_by(|left, right| left.variable_id.cmp(&right.variable_id));
        let quantity = ScientificQuantity {
            variable_id,
            variable_name: "counterfactual target event".into(),
            role: QuantityRole::Outcome,
            units: "probability".into(),
            population_id: self.query.target.clone(),
            regime_id,
            horizon: 0,
            functional_id: "counterfactual_event_probability".into(),
            conditioning,
            transform_id: "identity".into(),
        };
        quantity.validate().map_err(|_| {
            failure(BundleStage::TamperedQuantity, "invalid transport quantity coordinate")
        })?;
        Ok(MeanSource {
            coordinates: vec![quantity],
            means: vec![self.probability],
            estimate: MeanEstimate::Frequentist,
            provider_id: self.identity(),
            snapshot_id: self.data.clone(),
            causal_contract_id: self.premises.clone(),
            rng_id: "none:deterministic_transport".into(),
        })
    }
}

pub(super) fn describe(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<ArtifactDescription, NodeFailure> {
    let transport = read(bytes, ctx)?;
    let source = transport.source()?;
    let quantity = &source.coordinates[0];
    let quantity_wire = antecedent_io::quantity_wire::ScientificQuantityWire::from(quantity);
    let coordinate = coordinate_id(&quantity_wire);
    let quantity_json = serde_json::to_string(&quantity_wire)
        .map_err(|error| failure(BundleStage::TamperedQuantity, error))?;
    let mut facts = BTreeMap::from([
        (FACT_COORDINATES.into(), coordinate),
        (FACT_EVIDENCE_DIGEST.into(), transport.premises.clone()),
        (FACT_LAW.into(), LAW_MEAN_ONLY.into()),
        (FACT_MEAN_SOURCE.into(), source.provider_id.clone()),
        (FACT_REQUEST.into(), source.causal_contract_id),
        (FACT_SNAPSHOT.into(), format!("{}|{}", source.snapshot_id, transport.premises)),
        (FACT_TRUST.into(), TRUST_NATIVE.into()),
        ("quantity".into(), quantity_json),
        (
            "counterfactual_probability_bits".into(),
            format!("{:016x}", transport.probability.to_bits()),
        ),
        ("counterfactual_data_digest".into(), source.snapshot_id),
        ("target_population".into(), transport.query.target),
        ("conditioning_mass_checked".into(), transport.conditioning_mass_checked.to_string()),
    ]);
    if let Some(sampling) = transport.sampling {
        facts.insert("sampling".into(), sampling);
        facts.insert("construction".into(), "sampled_plugin".into());
    } else {
        facts.insert("construction".into(), "exact_finite_law".into());
    }
    Ok(ArtifactDescription {
        kind: NodeKind::CounterfactualTransportResult,
        identity: source.provider_id,
        facts,
        values: vec![("probability".into(), transport.probability)],
    })
}

pub(super) fn decision_bytes(
    contract: &DecisionContract,
    bytes: &[u8],
    artifact_id: &str,
    ctx: &ExecutionContext,
) -> Result<Vec<u8>, NodeFailure> {
    let transport = read(bytes, ctx)?;
    let source = transport.source()?;
    let result = evaluate_contract_on_means(contract, &source)
        .map_err(|error| failure(BundleStage::UnsupportedLaw, error))?;
    bound_mean_result_to_bytes(&result, &source.provider_id, artifact_id)
        .map_err(|error| failure(BundleStage::TamperedQuantity, error))
}

pub(super) struct CounterfactualTransportVerifier(pub(super) ExecutionContext);

impl NodeVerifier for CounterfactualTransportVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::CounterfactualTransportResult
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let description = describe(bytes, &self.0)?;
        Ok(VerifiedNode {
            identity: description.identity,
            facts: description.facts,
            values: description.values,
        })
    }
}
