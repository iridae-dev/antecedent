//! Versioned, independently consumable learner-backed trial transport claims.
use crate::{IoError, query_wire::TransportQueryWire, wire::AdmgWire};
use antecedent_estimate::{TrialAipwEstimate, TrialAipwInput, TrialAipwOptions};
use serde::{Deserialize, Serialize};

/// Scientific inputs and retained score evidence for a trial transport execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedTrialWire {
    /// Schema version.
    pub version: u32,
    /// Original static graph coordinates.
    pub graph: AdmgWire,
    /// Variables whose mechanisms differ.
    pub selections: Vec<u32>,
    /// Target, populations, and evidence catalog.
    pub query: TransportQueryWire,
    /// Raw source/target snapshot and sampling design.
    pub input: TrialAipwInput,
    /// Frozen fit/inference settings.
    pub options: TrialAipwOptions,
    /// Seed used for every retained fit and draw.
    pub seed: u64,
    /// Executed OOF score and bootstrap evidence.
    pub result: TrialAipwEstimate,
}
use crate::error::convert_err as err;
impl LearnedTrialWire {
    /// Verify graph identification, exact score replay, and inference bookkeeping.
    /// No fitting or resampling is performed.
    /// # Errors
    /// Invalid graph, unsupported formula, inconsistent score, or invalid intervals.
    pub fn verify(&self) -> Result<(), IoError> {
        if self.version != 1 {
            return Err(err("unsupported learned trial version"));
        }
        let graph = crate::admg_from_wire(&self.graph)?;
        let diagram = antecedent_graph::SelectionDiagram::try_new(
            graph,
            self.selections
                .iter()
                .map(|v| antecedent_core::VariableId::from_raw(*v))
                .collect::<Vec<_>>(),
        )
        .map_err(err)?;
        let query = crate::query_wire::transport_query_from_wire(&self.query)?;
        antecedent_estimate::validate_trial_query(&query).map_err(err)?;
        let id = antecedent_identify::TransportIdentifier::new()
            .identify(&diagram, &query)
            .map_err(err)?;
        antecedent_estimate::validate_trial_aipw(&id, &self.input, &self.options).map_err(err)?;
        let replay = antecedent_estimate::trial_to_target_effect(
            &id,
            &self.input.outcome,
            &self.input.treatment,
            &self.input.source,
            &self.result.membership,
            &self.input.randomization,
            Some((&self.result.mu0, &self.result.mu1)),
        )
        .map_err(err)?;
        if antecedent_estimate::learned_trial::trial_nuisance_diagnostics(
            &self.input,
            &self.result.membership,
            &self.result.mu0,
            &self.result.mu1,
        )
        .map_err(err)?
            != self.result.diagnostics
        {
            return Err(err("learned trial nuisance diagnostics mismatch"));
        }
        if replay.overlap != self.result.overlap
            || replay.aipw != Some(self.result.estimate)
            || !self.result.estimate.is_finite()
            || self.result.replicates.len() as u64 + u64::from(self.result.failures)
                != u64::from(self.options.bootstrap)
            || self
                .result
                .replicates
                .iter()
                .any(|(i, v)| *i >= self.options.bootstrap || !v.is_finite())
            || self.result.replicates.windows(2).any(|p| p[0].0 >= p[1].0)
        {
            return Err(err("learned trial score/replicate evidence mismatch"));
        }
        let expected = if self.options.bootstrap >= 2 && self.result.failures == 0 {
            let values: Vec<_> = self.result.replicates.iter().map(|(_, v)| *v).collect();
            Some(antecedent_estimate::statistical_transport::percentile_interval(
                &values,
                self.options.coverage_level,
            ))
        } else {
            None
        };
        let reason = if expected.is_some() {
            None
        } else {
            Some(
                if self.result.failures > 0 {
                    "bootstrap_replicate_failure"
                } else if self.options.bootstrap == 0 {
                    "bootstrap_not_requested"
                } else {
                    "insufficient_bootstrap_replicates"
                }
                .to_owned(),
            )
        };
        if expected != self.result.interval || reason != self.result.uncertainty_reason {
            return Err(err("learned trial uncertainty claim mismatch"));
        }
        Ok(())
    }
    /// Scientific execution identity, including retained evidence.
    /// # Errors
    /// Canonical encoding failure.
    pub fn identity(&self) -> Result<String, IoError> {
        Ok(crate::identity::digest_wire(antecedent_core::IdentityDomain::LearnedTrial, self)?
            .to_hex())
    }
    /// Encode a validated claim in the common artifact container.
    /// # Errors
    /// Verification or encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        self.verify()?;
        crate::transport_grid_wire::encode_transport_section(
            "learned_trial_v1",
            "learned_trial",
            &self.identity()?,
            self,
        )
    }
    /// Consume and verify without fitting.
    /// # Errors
    /// Unknown artifact format, invalid payload, or changed identity.
    pub fn consume(bytes: &[u8]) -> Result<Self, IoError> {
        let artifact = crate::transport_certificate::read_bounded_transport_artifact(
            bytes,
            &antecedent_core::ExecutionContext::production_default(0),
        )?;
        if artifact.sections.len() != 1
            || artifact.manifest.artifact_kind
                != crate::wire::ArtifactKind::Other("learned_trial".into())
        {
            return Err(err("expected learned trial artifact"));
        }
        let section = artifact
            .sections
            .iter()
            .find(|s| s.id == "learned_trial_v1")
            .ok_or_else(|| err("missing learned trial section"))?;
        let result: Self = crate::from_cbor(&section.data)?;
        result.verify()?;
        if result.identity()? != artifact.manifest.artifact_id {
            return Err(err("learned trial identity mismatch"));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The foreign option struct is serialized directly into the learned-trial artifact and
    /// its identity digest (CBOR map order is declaration order). A rename or reorder in the
    /// owning crate would silently change both, so the field names and order are pinned here.
    #[test]
    fn embedded_trial_options_keep_their_wire_field_names_and_order() {
        let bytes = crate::to_cbor(&TrialAipwOptions::default()).unwrap();
        let value: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        let entries = value.as_map().expect("options serialize as a CBOR map");
        let keys: Vec<&str> = entries.iter().map(|(key, _)| key.as_text().unwrap()).collect();
        assert_eq!(keys, ["outcome", "membership", "folds", "bootstrap", "coverage_level"]);
        let scalar = |name: &str| {
            entries.iter().find(|(key, _)| key.as_text() == Some(name)).map(|(_, v)| v.clone())
        };
        assert_eq!(scalar("folds"), Some(ciborium::Value::Integer(5.into())));
        assert_eq!(scalar("bootstrap"), Some(ciborium::Value::Integer(199.into())));
        assert_eq!(scalar("coverage_level"), Some(ciborium::Value::Float(0.95)));
    }
}
