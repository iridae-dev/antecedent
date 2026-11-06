//! Decision and joint-law composition of cyclic evidence.

use super::{
    CyclicArtifactError, CyclicEvidence, CyclicEvidenceColumn, Interval, ScientificQuantity,
    invalid,
};
use antecedent_io::{
    distribution_artifact::{
        DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
        DistributionProvenance, DistributionTrust, DrawAlignment,
    },
    quantity_wire::DistributionMeaningWire,
};

impl CyclicEvidence {
    /// Existing decision engine's affine-mean route, conditional on the reported source.
    /// A posterior/ bootstrap column is not silently averaged into an outcome law.
    /// # Errors
    /// Wrong evidence port, unresolved slots, incompatible decision contract or numeric failure.
    pub fn decide_means(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
    ) -> Result<serde_json::Value, CyclicArtifactError> {
        let mut coordinates = Vec::new();
        let mut means = Vec::new();
        for column in &self.report.columns {
            if column.port != "mean" || column.samples.len() != 1 || column.samples[0].is_none() {
                return Err(invalid(
                    "decision needs point/mean evidence; unresolved states and other uncertainty ports are retained, not coerced",
                ));
            }
            coordinates
                .push(ScientificQuantity::try_from(column.quantity.clone()).map_err(invalid)?);
            means.push(column.samples[0].ok_or_else(|| invalid("missing mean"))?);
        }
        let source = antecedent_design::decision_eval::MeanSource {
            coordinates,
            means,
            estimate: antecedent_design::decision_eval::MeanEstimate::Frequentist,
            provider_id: "native:cyclic_model_conditional".into(),
            snapshot_id: self.report.source_identity.clone(),
            causal_contract_id: self.report.identity.clone(),
            rng_id: "deterministic_native_projection".into(),
        };
        let mut result =
            antecedent_design::decision_eval::evaluate_contract_on_means(contract, &source)
                .map_err(|e| invalid(format!("{e:?}")))?;
        result.assumptions.push(self.report.scope.clone());
        let result = antecedent_design::decision_artifact::mean_result_to_json(&result, &source)
            .map_err(invalid)?;
        Ok(
            serde_json::json!({"evidence_identity":self.report.identity,"source_identity":self.report.source_identity,"scope":self.report.scope,"result":serde_json::from_str::<serde_json::Value>(&result).map_err(invalid)?}),
        )
    }
    /// Evaluate conservative solution/scenario utility enclosures with the existing
    /// robust set decision engine. No equilibrium probabilities or nonparametric
    /// identification claim are inferred. Unresolved boxes remain inside the set.
    /// # Errors
    /// Missing/undefined states, incompatible quantities, probability criteria,
    /// hard chance constraints or an invalid existing decision contract.
    pub fn decide_enclosures(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
    ) -> Result<serde_json::Value, CyclicArtifactError> {
        use antecedent_design::{
            decision_adapters::{IdentifiedUtility, evaluate_identified_sets},
            decision_contract::{AdmissibilityRules, AdmissibleDecisionContract},
            decision_robustness::AtomSupport,
        };
        if !contract.constraints.is_empty() {
            return Err(invalid(
                "a solution enclosure does not supply a chance-constraint probability law",
            ));
        }
        let coordinates = self
            .report
            .columns
            .iter()
            .map(|c| ScientificQuantity::try_from(c.quantity.clone()).map_err(invalid))
            .collect::<Result<Vec<_>, _>>()?;
        let mut utilities = Vec::new();
        for action in &contract.actions {
            let mut inputs = Vec::new();
            for quantity in &action.inputs {
                let i = coordinates
                    .iter()
                    .position(|c| c.require_same_coordinate(quantity).is_ok())
                    .ok_or_else(|| {
                        invalid("solution decision quantity does not match original evidence")
                    })?;
                let column = &self.report.columns[i];
                if column.port != "scenario" || column.numerical_enclosures.is_empty() {
                    return Err(invalid(
                        "solution decision requires a declared numerical solution/scenario enclosure",
                    ));
                }
                if column.mass_bounds.is_some()
                    && (column
                        .diagnostics
                        .get("positive_evidence_certified")
                        .and_then(serde_json::Value::as_bool)
                        != Some(true)
                        || column
                            .mass_bounds
                            .as_ref()
                            .is_some_and(|m| m.len() != column.numerical_enclosures.len()))
                {
                    return Err(invalid("posterior selection evidence is not certified positive"));
                }
                let mut lower = f64::INFINITY;
                let mut upper = f64::NEG_INFINITY;
                for (k, bound) in column.numerical_enclosures.iter().enumerate() {
                    if column
                        .mass_bounds
                        .as_ref()
                        .is_some_and(|m| m.get(k).is_some_and(|p| p.upper == 0.0))
                    {
                        continue;
                    }
                    let value=bound.ok_or_else(||invalid("undefined or unresolved selected-state mass has no decision utility enclosure"))?;
                    lower = lower.min(value.lower);
                    upper = upper.max(value.upper);
                }
                if !lower.is_finite() || !upper.is_finite() {
                    return Err(invalid("empty/undefined solution enclosure"));
                }
                inputs.push(Interval { lower, upper });
            }
            let enclosure = enclose_utility(&action.utility, &inputs)?;
            utilities.push(IdentifiedUtility {
                action_id: action.id.clone(),
                utility: antecedent_core::IdentifiedSet::try_new(enclosure.lower, enclosure.upper)
                    .map_err(invalid)?,
                hard_exclusions: vec![],
                support: AtomSupport::unassessed(),
            });
        }
        let result = evaluate_identified_sets(
            &AdmissibleDecisionContract {
                contract: contract.clone(),
                rules: AdmissibilityRules::default(),
            },
            &utilities,
        )
        .map_err(|e| invalid(format!("{e:?}")))?;
        let actions=result.actions.iter().map(|a|serde_json::json!({"id":a.id,"utility":a.utility,"eligible":a.eligible,"dominated_by":a.dominated_by,"possibly_optimal":a.possibly_optimal,"necessarily_optimal":a.necessarily_optimal,"max_regret":a.max_regret})).collect::<Vec<_>>();
        Ok(
            serde_json::json!({"evidence_identity":self.report.identity,"source_identity":self.report.source_identity,"scope":"conditional_model_solution_outer_enclosure","actions":actions,"verdict":format!("{:?}",result.verdict),"lower_leader":result.lower_leader,"upper_leader":result.upper_leader,"conflicting_leaders":result.conflicting_leaders,"selection_probabilities_inferred":false}),
        )
    }
    /// Build the existing aligned joint-law artifact for a complete posterior column set.
    /// Bootstrap, root sets, finite mass bounds and failed slots cannot be coerced to a belief law.
    /// # Errors
    /// Different axes/alignments, missing mass, wrong port or invalid artifact coordinates.
    pub fn joint_law(&self) -> Result<DistributionArtifact, CyclicArtifactError> {
        let columns = self.report.columns.iter().collect::<Vec<_>>();
        let provenance = if let Some(origin) = &self.report.recovery_origins {
            recovery_distribution_provenance(origin)
        } else {
            DistributionProvenance {
                source_id: self.report.source_identity.clone(),
                provider_id: "native:cyclic_model_conditional".into(),
                rng_id: format!("retained:{}", self.report.source_identity),
                snapshot_id: self.report.source_identity.clone(),
                causal_contract_id: self.report.identity.clone(),
            }
        };
        cyclic_joint_columns(&columns, provenance, self.version == 3)
    }

    /// Compose explicit native evidence objects only under an established draw pairing.
    /// Source evidence, statistical fit and row-alignment identities are separate.
    /// # Errors
    /// Shared evidence declared independent, unknown/mismatched pairing, failed slots,
    /// incompatible quantity declarations or an uncertainty port without an outcome law.
    pub fn joint_law_with(
        &self,
        others: &[&Self],
        dependence: Option<&str>,
    ) -> Result<DistributionArtifact, CyclicArtifactError> {
        if others.len() > 256 {
            return Err(invalid("at most 256 additional evidence products"));
        }
        if others.is_empty() && dependence.is_none() {
            return self.joint_law();
        }
        let sources = std::iter::once(self).chain(others.iter().copied()).collect::<Vec<_>>();
        if dependence == Some("independent") {
            for (i, first) in sources.iter().enumerate() {
                for second in &sources[i + 1..] {
                    if let (Some(a), Some(b)) =
                        (&first.report.recovery_origins, &second.report.recovery_origins)
                    {
                        if a.evidence == b.evidence {
                            return Err(joint_refusal(
                                antecedent_core::reason_code!("evidence_double_counted"),
                                "cyclic_evidence.shared_training_declared_independent: these products read the same experiment evidence; supply their verified aligned joint law",
                            ));
                        }
                    }
                }
            }
            return Err(joint_refusal(
                antecedent_core::reason_code!("evidence_overlap_unknown"),
                "cyclic_evidence.independence_not_established: different labels do not prove disjoint observations or independent laws; supply an explicit joint law",
            ));
        }
        if dependence != Some("aligned") {
            return Err(joint_refusal(
                antecedent_core::reason_code!("joint_law_required"),
                "cyclic_evidence.alignment_declaration_required: select aligned only for native products carrying the same ordered-law pairing proof",
            ));
        }
        let origin = self.report.recovery_origins.as_ref().ok_or_else(|| joint_refusal(
            antecedent_core::reason_code!("joint_law_required"),
            "cyclic_evidence.legacy_alignment_unavailable: this historical envelope does not attest cross-product recovery origins; independently replay its source and explicitly project new evidence"))?;
        let mut columns = Vec::new();
        for source in sources {
            let found = source.report.recovery_origins.as_ref().ok_or_else(|| joint_refusal(
                antecedent_core::reason_code!("joint_law_required"), "cyclic_evidence.alignment_unavailable: independently project version-three evidence from its verified source"))?;
            if found.alignment != origin.alignment
                || found.fit != origin.fit
                || found.law != origin.law
            {
                return Err(joint_refusal(
                    antecedent_core::reason_code!("joint_law_required"),
                    "cyclic_evidence.draw_alignment_mismatch: equal draw counts are insufficient; provide the actual shared ordered law or an explicit coupling",
                ));
            }
            columns.extend(source.report.columns.iter());
        }
        cyclic_joint_columns(&columns, recovery_distribution_provenance(origin), true)
    }
}

fn joint_refusal(code: &'static str, message: &str) -> CyclicArtifactError {
    CyclicArtifactError::ScientificRefusal {
        code,detail:"cyclic_evidence.joint_law",message:message.into(),
        next_attempt:"declare an eligible shared joint law with compatible quantities, units, populations and uncertainty ports".into(),
    }
}
fn recovery_distribution_provenance(
    origin: &antecedent_io::cyclic_family_artifact::CyclicRecoveryOrigins,
) -> DistributionProvenance {
    DistributionProvenance {
        source_id: origin.evidence.clone(),
        provider_id: "native:cyclic_model_conditional".into(),
        rng_id: origin.alignment.clone(),
        snapshot_id: origin.law.clone(),
        // The fitting contract is shared; each target remains in its quantity/regime identity.
        causal_contract_id: origin.fit.clone(),
    }
}
#[allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    reason = "joint-law checks preserve exact mass and provenance semantics"
)]
fn cyclic_joint_columns(
    columns: &[&CyclicEvidenceColumn],
    provenance: DistributionProvenance,
    exact_recovery_weights: bool,
) -> Result<DistributionArtifact, CyclicArtifactError> {
    use antecedent_io::distribution_artifact::{
        MAX_DISTRIBUTION_ARTIFACT_BYTES, MAX_DISTRIBUTION_COORDINATES, MAX_DISTRIBUTION_DRAWS,
    };
    let first = columns.first().ok_or_else(|| invalid("empty evidence"))?;
    if !matches!(first.port.as_str(), "parameter_posterior" | "interventional_law") {
        return Err(joint_refusal(
            antecedent_core::reason_code!("joint_law_required"),
            "cyclic_evidence.uncertainty_port: sampling laws and means cannot become posterior or outcome distributions",
        ));
    }
    let count = first.samples.len();
    if count == 0 {
        return Err(invalid("empty law"));
    }
    if columns.len() > MAX_DISTRIBUTION_COORDINATES
        || count > MAX_DISTRIBUTION_DRAWS
        || count
            .checked_mul(columns.len())
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
            .is_none_or(|bytes| bytes > MAX_DISTRIBUTION_ARTIFACT_BYTES)
    {
        return Err(invalid("joint law output exceeds native distribution limits"));
    }
    let weights_of =
        |column: &CyclicEvidenceColumn| -> Result<Option<Vec<f64>>, CyclicArtifactError> {
            let Some(bounds) = &column.mass_bounds else {
                return Ok(None);
            };
            // Only an actual posterior of recovered functional draws is supported
            // here. A Gaussian mean with variance in diagnostics is not an outcome draw.
            if !exact_recovery_weights
                || column.port != "parameter_posterior"
                || bounds.len() != count
            {
                return Err(joint_refusal(
                    antecedent_core::reason_code!("joint_law_required"),
                    "cyclic_evidence.mass_or_mean_not_a_law: numerical sets, uncertain mass and projected Gaussian means require their scientific law adapter",
                ));
            }
            let mut weights = Vec::with_capacity(count);
            for mass in bounds {
                if mass.lower != mass.upper || !mass.lower.is_finite() || mass.lower < 0.0 {
                    return Err(joint_refusal(
                        antecedent_core::reason_code!("joint_law_required"),
                        "cyclic_evidence.nonexact_mass: preserve mass bounds and provide a complete weighted law",
                    ));
                }
                weights.push(mass.lower);
            }
            if (weights.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
                return Err(joint_refusal(
                    antecedent_core::reason_code!("joint_law_required"),
                    "cyclic_evidence.incomplete_mass: failed or withheld mass cannot be discarded",
                ));
            }
            Ok(Some(weights))
        };
    let weights = weights_of(first)?;
    let population = &first.quantity.population_id;
    let mut quantities = Vec::with_capacity(columns.len());
    for column in columns {
        if column.port != first.port
            || column.uncertainty_axes != first.uncertainty_axes
            || column.samples.len() != count
            || column.samples.iter().any(Option::is_none)
            || weights_of(column)? != weights
        {
            return Err(joint_refusal(
                antecedent_core::reason_code!("joint_law_required"),
                "cyclic_evidence.incomplete_or_misaligned_law: retain every original draw/weight, failed slot and uncertainty meaning",
            ));
        }
        if &column.quantity.population_id != population {
            return Err(joint_refusal(
                antecedent_core::reason_code!("joint_law_required"),
                "cyclic_evidence.population_bridge_required: declare a scientific population bridge before composing these quantities",
            ));
        }
        for previous in &quantities {
            let previous: &ScientificQuantity = previous;
            if previous.variable_name == column.quantity.variable_name
                && previous.units != column.quantity.units
            {
                return Err(joint_refusal(
                    antecedent_core::reason_code!("joint_law_required"),
                    "cyclic_evidence.units_bridge_required: declare a unit conversion before composing the same variable under incompatible units",
                ));
            }
        }
        quantities.push(ScientificQuantity::try_from(column.quantity.clone()).map_err(invalid)?);
    }
    let mut rows = Vec::with_capacity(
        count.checked_mul(columns.len()).ok_or_else(|| invalid("joint law shape overflow"))?,
    );
    for row in 0..count {
        for column in columns {
            rows.push(column.samples[row].ok_or_else(|| invalid("missing draw"))?);
        }
    }
    let meaning = if first.port == "parameter_posterior" {
        DistributionMeaningWire::CausalFunctionalPosterior
    } else {
        DistributionMeaningWire::InterventionalPredictive
    };
    let identity =
        DistributionIdentity::new(meaning, &quantities, DrawAlignment::Joint, provenance)
            .map_err(invalid)?;
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [count, quantities.len()],
            weights,
            supported: None,
            calibration: DistributionCalibration::NotMeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        rows,
    )
    .map_err(invalid)
}

fn enclose_utility(
    expression: &antecedent_design::decision_contract::UtilityExpr,
    inputs: &[Interval],
) -> Result<Interval, CyclicArtifactError> {
    use antecedent_design::decision_contract::UtilityExpr as U;
    let output = match expression {
        U::Const(value) => Interval { lower: *value, upper: *value },
        U::Input(i) => *inputs.get(*i).ok_or_else(|| invalid("utility input index"))?,
        U::Neg(a) => enclose_utility(a, inputs)?.scale_outward(-1.0),
        U::Add(a, b) => enclose_utility(a, inputs)?.add_outward(enclose_utility(b, inputs)?),
        U::Sub(a, b) => {
            enclose_utility(a, inputs)?.add_outward(enclose_utility(b, inputs)?.scale_outward(-1.0))
        }
        U::Mul(a, b) => enclose_utility(a, inputs)?.multiply_outward(enclose_utility(b, inputs)?),
        U::Min(a, b) => {
            let a = enclose_utility(a, inputs)?;
            let b = enclose_utility(b, inputs)?;
            Interval { lower: a.lower.min(b.lower), upper: a.upper.min(b.upper) }
        }
        U::Max(a, b) => {
            let a = enclose_utility(a, inputs)?;
            let b = enclose_utility(b, inputs)?;
            Interval { lower: a.lower.max(b.lower), upper: a.upper.max(b.upper) }
        }
    };
    if !output.lower.is_finite() || !output.upper.is_finite() {
        return Err(invalid("nonfinite utility enclosure"));
    }
    Ok(output)
}
