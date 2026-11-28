//! Deterministic scientific outputs for evidence carries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use ciborium::value::{CanonicalValue, Value};
use serde::Serialize;

use super::{IntervalBinding, ResultPayload, StudyResult};

impl StudyResult {
    pub(super) fn evidence_output(&self, route: &str) -> Option<Vec<u8>> {
        // Fail closed for result families whose full typed answer has no audited wire section.
        // An exhaustive match makes a new family state its policy explicitly.
        let complete: fn(&antecedent_io::AnalysisResultWire) -> bool = match &self.payload {
            ResultPayload::Scalar => |body| body.estimate.is_some(),
            ResultPayload::PolicyValue(_) => |body| body.policy_value.is_some(),
            ResultPayload::ContinuousDoseResponse(_) => {
                |body| body.continuous_dose_response.is_some()
            }
            ResultPayload::Response(_)
            | ResultPayload::Distribution(_)
            | ResultPayload::Mediation { .. }
            | ResultPayload::Counterfactual(_)
            | ResultPayload::CounterfactualContrasts(_)
            | ResultPayload::Anomaly(_)
            | ResultPayload::ChangeAttribution(_)
            | ResultPayload::MechanismChange(_)
            | ResultPayload::UnitChange(_)
            | ResultPayload::Transport(_)
            | ResultPayload::Interference { .. }
            | ResultPayload::RandomizedEffect(_)
            | ResultPayload::PanelDid(_)
            | ResultPayload::SyntheticControl(_)
            | ResultPayload::SyntheticDid(_)
            | ResultPayload::LocalPolynomialRatio(_)
            | ResultPayload::Survival(_)
            | ResultPayload::PosteriorRow(_)
            | ResultPayload::LongitudinalRegime(_)
            | ResultPayload::SequentialContrast(_)
            | ResultPayload::HistoryMatch(_)
            | ResultPayload::TemporalDesign(_)
            | ResultPayload::IvBounds(_)
            | ResultPayload::Association(_)
            | ResultPayload::QuasiRollout(_)
            | ResultPayload::Discovery(_)
            | ResultPayload::Attribution(_)
            | ResultPayload::Simulation(_) => return None,
        };
        if self.posterior.is_some()
            || self.structural_response.is_some()
            || self.predictive_replicates.is_some()
            || self.full_target_diagnostics.is_some()
            || self.trial_transport_receipt.is_some()
            || self.temporal_response_witnesses.is_some()
            || self.temporal_posterior_witnesses.is_some()
            || self.static_class_witnesses.is_some()
            || self.static_posterior_source.is_some()
            || !self.refutations.is_empty()
            || !self.predictive_checks.is_empty()
            || self.extras.candidate_selection.is_some()
            || self.extras.row_weights.is_some()
            || self.extras.missing_data.is_some()
        {
            return None;
        }
        let mut body = self
            .analysis_result_wire_with_context(
                &self.identification.query,
                self.population_registry.as_ref(),
                None,
            )
            .ok()?;
        if !complete(&body) {
            return None;
        }
        body.native_execution = None;
        let interval = self.extras.interval.map(|interval| {
            let IntervalBinding {
                method,
                se_kind,
                level,
                dependence,
                replicates_ok,
                replicates_failed,
                posterior_draws,
            } = interval;
            (
                method.as_str(),
                format!("{se_kind:?}"),
                level,
                dependence,
                replicates_ok,
                replicates_failed,
                posterior_draws,
            )
        });
        let unit = self.extras.unit.as_ref().map(|unit| (unit.kind(), unit.record_id()));
        // The Rust scalar estimate also exposes native diagnostics and score tables beyond the
        // portable body. Keep the compiler-derived representation of the entire estimate as an
        // additional guard and preserve the scalar and raw analytic SE bits in CBOR.
        let native_estimate = if let Some(estimate) = self.estimate.as_effect() {
            let analytic_se = estimate.uncertainty.analytic.se;
            let debug = format!("{estimate:?}");
            // Debug cannot distinguish NaN payloads. The primary and analytic SE are also
            // encoded directly below. Decline an additional NaN in a native-only field.
            if debug.matches("NaN").count()
                > usize::from(estimate.ate.is_nan()) + usize::from(analytic_se.is_nan())
            {
                return None;
            }
            Some((estimate.ate, analytic_se, debug))
        } else {
            None
        };
        let replication = (
            self.performance.bootstrap_replicates_requested,
            self.performance.bootstrap_replicates_ok,
            self.performance.n_draws,
            self.performance.cancelled,
            self.performance.early_stopped,
        );
        let variable_ids =
            |ids: &[antecedent_core::VariableId]| ids.iter().map(|id| id.raw()).collect::<Vec<_>>();
        let evidence_context = (
            self.analysis_trace_wire(),
            antecedent_io::provenance_to_wire(&self.provenance),
            self.estimand.functional.raw(),
            variable_ids(&self.estimand.instruments),
            variable_ids(&self.estimand.mediators),
            self.estimand.rd_design.map(|d| (d.running_variable.raw(), d.cutoff, d.bandwidth)),
            identification_claims(&self.identification),
            &self.extras.custom_validator_names,
        );
        canonical_bytes(&(
            route,
            self.lowered_identity.as_ref().map(antecedent_core::SemanticDigest::as_bytes),
            unit,
            self.support_status().map(crate::support::CellStatus::as_str),
            self.calibration_state(),
            body,
            point_certificate(self)?,
            native_estimate,
            interval,
            replication,
            evidence_context,
        ))
    }
}

fn point_certificate(result: &StudyResult) -> Option<Option<impl Serialize>> {
    let Some(certificate) = &result.certificate else { return Some(None) };
    let crate::Identification::Point {
        result: identification,
        temporal_indexer: None,
        strategy,
        structure_version,
    } = &certificate.identification
    else {
        return None;
    };
    let registry = result.population_registry.as_ref();
    Some(Some((
        antecedent_io::identification_to_wire_with_registry(identification, registry).ok()?,
        identification_claims(identification),
        antecedent_io::causal_query_to_wire_with_registry(&certificate.query, registry).ok()?,
        certificate.graph_class.as_str(),
        strategy.as_str(),
        *structure_version,
    )))
}

fn identification_claims(
    identification: &antecedent_identify::IdentificationResult,
) -> impl Serialize {
    identification
        .estimand_claims
        .iter()
        .map(|claim| {
            (
                format!("{:?}", claim.status),
                antecedent_io::assumptions_to_wire(&claim.required_assumptions),
            )
        })
        .collect::<Vec<_>>()
}

fn order_maps(value: &mut Value) {
    match value {
        Value::Array(items) => items.iter_mut().for_each(order_maps),
        Value::Map(entries) => {
            for (key, item) in entries.iter_mut() {
                order_maps(key);
                order_maps(item);
            }
            entries.sort_by_cached_key(|(key, _)| CanonicalValue::from(key.clone()));
        }
        Value::Tag(_, item) => order_maps(item),
        _ => {}
    }
}

fn canonical_bytes(value: &impl Serialize) -> Option<Vec<u8>> {
    // CBOR keeps signed zero, infinities and NaN payloads, which serde_json would collapse.
    let mut value = Value::serialized(value).ok()?;
    order_maps(&mut value);
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(&value, &mut bytes).ok()?;
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::{Value, canonical_bytes};
    use antecedent_io::PolicyValueWire;
    use antecedent_io::analysis_result_artifact::ContinuousDosePointWire;

    #[test]
    fn canonical_scientific_outputs_keep_scalar_bits_and_map_order_does_not_matter() {
        let first_nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let second_nan = f64::from_bits(0x7ff8_0000_0000_0002);
        assert_ne!(canonical_bytes(&first_nan), canonical_bytes(&second_nan));
        assert_ne!(canonical_bytes(&0.0), canonical_bytes(&-0.0));
        let a = (Value::Text("a".into()), Value::Float(-0.0));
        let b = (Value::Text("b".into()), Value::Float(0.0));
        assert_eq!(
            canonical_bytes(&Value::Map(vec![a.clone(), b.clone()])),
            canonical_bytes(&Value::Map(vec![b, a]))
        );
    }

    #[test]
    fn canonical_scientific_outputs_observe_typed_policy_and_vector_values() {
        let mut policy = PolicyValueWire {
            policy_value: 1.5,
            reference_value: 1.0,
            incremental_value: 0.5,
            relative_value_gap: -0.5,
            treatment_rate: 0.25,
            total_cost: 1.0,
            policy_standard_error: 0.2,
            reference_standard_error: 0.4,
            incremental_standard_error: 0.2,
            policy_interval_95: Some([1.1, 1.9]),
            incremental_interval_95: None,
            prediction_ownership: "known".into(),
            propensity_min: 0.5,
            propensity_max: 0.5,
            uncertainty: "row-score".into(),
            graphless_support_status: None,
            uplift_bins: Vec::new(),
            multi_action_cate: Vec::new(),
            regret: None,
        };
        let initial = canonical_bytes(&policy).unwrap();
        policy.reference_value += 1.0;
        assert_ne!(canonical_bytes(&policy).unwrap(), initial);
        let point = ContinuousDosePointWire {
            baseline_group: "all".into(),
            target_dose: 2.5,
            response: 1.5,
            local_rows: 10,
            effective_sample_size: 8.5,
            minimum_dose_density: 0.3,
            maximum_normalized_weight: 0.2,
            local_outcome_sd: 0.5,
        };
        let mut points = vec![point.clone(), point];
        let initial = canonical_bytes(&points).unwrap();
        points[1].response += 1.0;
        assert_ne!(canonical_bytes(&points).unwrap(), initial);
    }

    #[cfg(feature = "development")]
    #[test]
    fn compiled_scalar_study_exposes_its_value_to_the_evidence_digest() {
        use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
        use antecedent_data::TabularData;
        use antecedent_graph::{Dag, DenseNodeId};

        let data = TabularData::from_f64_columns([
            ("t", &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0][..]),
            ("y", &[0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0][..]),
        ])
        .unwrap();
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let mut result = crate::Study::tabular(data)
            .graph(dag)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .bootstrap_replicates(0)
            .refute(crate::RefuteSuite::None)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(7))
            .unwrap();
        let initial = result.evidence_output("AverageEffect").expect("scalar answer");
        result.estimate.as_effect_mut().unwrap().ate += 0.25;
        let changed_value = result.evidence_output("AverageEffect").expect("scalar answer");
        assert_ne!(changed_value, initial);
        result.estimate.as_effect_mut().unwrap().heterogeneity.cate = Some([0.5, 1.0].into());
        let changed_heterogeneity = result.evidence_output("AverageEffect").expect("scalar answer");
        assert_ne!(changed_heterogeneity, changed_value);
        result.view.provenance.nodes[0].operation = "revised operation".into();
        assert_ne!(
            result.evidence_output("AverageEffect").expect("scalar answer"),
            changed_heterogeneity
        );
    }

    #[cfg(feature = "development")]
    #[test]
    fn compiled_policy_exposes_its_typed_answer_to_the_evidence_digest() {
        use antecedent_core::{CausalQuery, ExecutionContext, PolicyValueQuery, VariableId};
        use antecedent_data::TabularData;
        use std::sync::Arc;

        let data = TabularData::from_f64_columns([("outcome", &[1.0, 2.0, 3.0, 4.0][..])]).unwrap();
        let query = PolicyValueQuery {
            outcome: VariableId::from_raw(0),
            assignment: [false, true, false, true].into(),
            propensity: [0.5].into(),
            actions: [true; 4].into(),
            reference: [false; 4].into(),
            mu0: [].into(),
            mu1: [].into(),
            costs: [0.0].into(),
            reference_costs: [0.0].into(),
            evaluation_subject_ids: ["a", "b", "c", "d"].map(Arc::from).into(),
            disjoint_training_subjects: false,
            crossfit_fold_ownership_valid: false,
            global_constraints_present: false,
            multi_action: None,
            uplift_bins: [].into(),
            uplift_bin_count: 0,
            uplift_training_subject_ids: [].into(),
            regret: None,
        };
        let mut result = crate::Study::tabular(data)
            .query(CausalQuery::PolicyValue(query))
            .estimator(crate::EstimatorId::RandomizedIpwPolicyNet)
            .refute(crate::RefuteSuite::None)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(7))
            .unwrap();
        let initial = result.evidence_output("PolicyValue").expect("typed policy answer");
        let super::ResultPayload::PolicyValue(value) = &mut result.payload else {
            panic!("the compiled policy route returned another payload")
        };
        value.reference_value += 1.0;
        assert_ne!(result.evidence_output("PolicyValue").expect("typed policy answer"), initial);
    }
}
