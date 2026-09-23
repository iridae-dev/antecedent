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

    fn cbor_keys(value: &ciborium::Value) -> Vec<String> {
        value
            .as_map()
            .expect("serializes as a CBOR map")
            .iter()
            .map(|(key, _)| key.as_text().unwrap().to_owned())
            .collect()
    }

    fn cbor_field<'a>(value: &'a ciborium::Value, name: &str) -> &'a ciborium::Value {
        &value
            .as_map()
            .expect("serializes as a CBOR map")
            .iter()
            .find(|(key, _)| key.as_text() == Some(name))
            .unwrap_or_else(|| panic!("missing field {name}"))
            .1
    }

    fn cbor_of<T: Serialize>(value: &T) -> ciborium::Value {
        let bytes = crate::to_cbor(value).unwrap();
        ciborium::from_reader(bytes.as_slice()).unwrap()
    }

    /// Every other foreign type embedded in the learned-trial artifact, the fitted-effect
    /// receipt and the transport grid is pinned the same way: a JSON document in the
    /// current spelling must still decode, and re-encoding must keep the declaration order the
    /// identity digest hashes.
    #[test]
    fn embedded_foreign_structs_keep_their_wire_spelling_and_order() {
        let input: TrialAipwInput = serde_json::from_str(
            r#"{"features":[0],"covariates":[[0.5]],"outcome":[1.0],"treatment":[true],
                "source":[true],"randomization":[0.5],"sampling":"independent_samples"}"#,
        )
        .unwrap();
        let value = cbor_of(&input);
        assert_eq!(
            cbor_keys(&value),
            [
                "features",
                "covariates",
                "outcome",
                "treatment",
                "source",
                "randomization",
                "sampling"
            ]
        );
        assert_eq!(
            cbor_field(&value, "sampling"),
            &ciborium::Value::Text("independent_samples".into())
        );
        assert_eq!(
            cbor_of(&antecedent_estimate::TrialSampling::NestedCohort),
            ciborium::Value::Text("nested_cohort".into())
        );

        let options = cbor_of(&TrialAipwOptions::default());
        let outcome = cbor_field(&options, "outcome");
        assert_eq!(cbor_keys(outcome), ["kind", "lambda"]);
        assert_eq!(cbor_field(outcome, "kind"), &ciborium::Value::Text("ridge".into()));
        let membership = cbor_field(&options, "membership");
        assert_eq!(cbor_keys(membership), ["kind", "ridge_lambda"]);
        assert_eq!(cbor_field(membership, "kind"), &ciborium::Value::Text("logistic".into()));
        for kind in ["auto", "linear", "elastic_net", "gradient_boosted_trees", "random_forest"] {
            let spec: antecedent_estimate::LearnerSpec =
                serde_json::from_str(&format!(r#"{{"kind":"{kind}"}}"#)).unwrap();
            assert_eq!(
                cbor_field(&cbor_of(&spec), "kind"),
                &ciborium::Value::Text(kind.into()),
                "{kind}"
            );
        }

        let fitted: antecedent_estimate::FittedEffect = serde_json::from_str(
            r#"{"version":1,"features":[0],"intercept":true,"predictor":{"version":1,
                "columns":2,"provenance":{"spec":"linear","implementation":"faer","version":"1"},
                "model":{"kind":"linear","coefficients":[0.5,1.0],"logistic":false}}}"#,
        )
        .unwrap();
        let value = cbor_of(&fitted);
        assert_eq!(cbor_keys(&value), ["version", "features", "intercept", "predictor"]);
        let predictor = cbor_field(&value, "predictor");
        assert_eq!(cbor_keys(predictor), ["version", "columns", "provenance", "model"]);
        assert_eq!(
            cbor_keys(cbor_field(predictor, "provenance")),
            ["spec", "implementation", "version"]
        );
        assert_eq!(cbor_keys(cbor_field(predictor, "model")), ["kind", "coefficients", "logistic"]);

        let estimate: antecedent_estimate::TrialAipwEstimate = serde_json::from_str(
            r#"{"estimate":0.1,"interval":null,"uncertainty_reason":null,"replicates":[],
                "failures":0,"membership":[0.5],"mu0":[0.0],"mu1":[1.0],"provenance":[],
                "diagnostics":{"membership_logloss":0.7,"outcome_rmse":[0.1,0.2]},
                "overlap":{"selection":{"probability_min":0.1,"probability_max":0.9,
                "effective_sample_size":10.0,"extreme_weight_count":0},
                "treatment":{"probability_min":0.5,"probability_max":0.5,
                "effective_sample_size":10.0,"extreme_weight_count":0}}}"#,
        )
        .unwrap();
        let value = cbor_of(&estimate);
        assert_eq!(
            cbor_keys(&value),
            [
                "estimate",
                "interval",
                "uncertainty_reason",
                "replicates",
                "failures",
                "membership",
                "mu0",
                "mu1",
                "provenance",
                "diagnostics",
                "overlap"
            ]
        );
        assert_eq!(
            cbor_keys(cbor_field(&value, "diagnostics")),
            ["membership_logloss", "outcome_rmse"]
        );
        let overlap = cbor_field(&value, "overlap");
        assert_eq!(cbor_keys(overlap), ["selection", "treatment"]);
        assert_eq!(
            cbor_keys(cbor_field(overlap, "selection")),
            ["probability_min", "probability_max", "effective_sample_size", "extreme_weight_count"]
        );
    }
}
