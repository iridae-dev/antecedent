//! Claim assembly: the portable claim over a contract and an execution, and its wire section.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(
    clippy::wildcard_imports,
    reason = "the contract module is split by concern and these files share its imports and private items"
)]
use super::*;

/// The evidence references of a claim: the three identities, plus, only when the result's
/// missing-data policy differs from its route default, the policy, the assumption it needs and
/// the digest of the dropped rows. Default claims keep exactly the references they always had.
pub(super) fn claim_evidence(result: &StudyResult, contract: &CausalContract) -> Vec<Arc<str>> {
    let mut evidence: Vec<Arc<str>> = vec![
        Arc::from(format!("snapshot:{}", contract.identities.data_snapshot.to_hex())),
        Arc::from(format!("identification:{}", contract.identities.identification.to_hex())),
        Arc::from(format!(
            "program:{}",
            contract.identities.program.map_or_else(String::new, |p| p.to_hex())
        )),
    ];
    if let Some(report) = result.extras.missing_data.as_ref().filter(|report| report.non_default) {
        evidence.push(Arc::from(format!("missing_policy:{}", report.policy().name())));
        evidence.push(Arc::from(format!("missing_assumption:{}", report.assumption)));
        evidence.push(Arc::from(format!("missing_dropped:{}", report.accounting.dropped_digest)));
        if let Some(imputation) = &report.imputation {
            for entry in crate::analysis::missing_imputation::evidence_entries(imputation) {
                evidence.push(Arc::from(entry));
            }
        }
    }
    evidence
}

impl StudyResult {
    /// Portable claim over `contract` and this execution.
    ///
    /// # Errors
    ///
    /// [`CausalError::Conflict`] when the result carries no execution stamp or was
    /// executed under other identities; canonical-encoding failures, or mass totals
    /// that do not conserve.
    pub fn claim(
        &self,
        contract: &CausalContract,
        ctx: &ExecutionContext,
    ) -> Result<ClaimEnvelope, CausalError> {
        Ok(self.claim_with_body(contract, ctx, None)?.0)
    }

    /// Claim plus the exact result body its id digests.
    ///
    /// The claim id binds the contract seal (every identity, the four slots,
    /// graph class, structure source, identifier, estimator), every claim
    /// field (kind, value, domains, calibration, attested evidence,
    /// execution), and [`antecedent_io::result_digest`] of the body. Kind and
    /// domains come from the rules the independent consumer re-derives.
    pub(super) fn claim_with_body(
        &self,
        contract: &CausalContract,
        ctx: &ExecutionContext,
        quantities: Option<&[antecedent_contract::ScientificQuantity]>,
    ) -> Result<(ClaimEnvelope, AnalysisResultWire), CausalError> {
        // A claim seals the result under the contract's identities, so the result must
        // prove which contract it ran under. An unstamped result (a plain `Study::run`, a
        // mixed-execution refutation) would be sealed under identities it never carried.
        let Some(executed) = &self.executed_contract else {
            return Err(CausalError::Conflict {
                what: "result",
                detail: "result carries no execution stamp; only a prepared handle's execution \
                         can be sealed into a claim",
            });
        };
        if executed.identities != contract.identities {
            return Err(CausalError::Conflict {
                what: "program",
                detail: "result was not executed under this contract",
            });
        }
        // A refutation this result's own program did not produce can only be a scalar
        // average-effect refutation, and it does not certify another target: a claim on a
        // conditional, derivative, temporal-vector, mediation, world-valued or joint target
        // refuses one, naming the target-native suite as the next attempt. Route-native
        // reports carry the producing program's stamp and pass.
        crate::analysis::target_validation::refuse_scalar_refutations_for_result(self)
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        let mut body = body_for(&contract.body, self)?;
        if let Some(quantities) = quantities {
            let response = body.response.as_mut().ok_or_else(|| CausalError::Compile {
                message: "scientific coordinates require an exported response".into(),
            })?;
            response.coordinates = Some(
                quantities
                    .iter()
                    .map(antecedent_io::quantity_wire::ScientificQuantityWire::from)
                    .collect(),
            );
        }
        let mut reasoning = result_reasoning(self, &contract.reasoning, &body)?;
        if let (SlotAvailability::Available(slot), Some(status)) =
            (&mut reasoning.identification, contract.body.identification_status)
        {
            // The body reports the program's identification certificate; the
            // slot states the same status the body carries.
            slot.status = status;
        }
        let reasoning_wire = reasoning_section(&reasoning);
        let kind = antecedent_io::claim_kind_of_section(&body, &reasoning_wire.identification);
        let value = match kind {
            ClaimKind::Point => executed_scalar(self),
            _ => None,
        };
        let domains = antecedent_io::claim_domain_statuses_of_section(
            reasoning_wire.support.value.as_ref(),
            &reasoning_wire.identification,
        );
        let execution =
            execution_digest(&execution_identity_from_context(ctx)).map_err(|err| io_err(&err))?;
        let calibration = if !reports_calibrated_effect_interval(&self.payload) {
            // These non-effect result families report no calibrated scalar
            // effect interval. Their uncertainty semantics live in their
            // dedicated result sections.
            antecedent_io::calibration::calibration_slots(&[])
        } else {
            antecedent_io::calibration::calibration_slots(
                &self.calibration_bases_with(contract, reasoning.identification.as_ref()),
            )
        };
        let mut envelope = ClaimEnvelope::new(
            antecedent_core::SemanticDigest::from_bytes([0; 32]),
            contract.identities,
            kind,
            value,
            None,
            reasoning,
            domains,
            Some(execution),
            claim_evidence(self, contract),
        );
        envelope.calibration = calibration_view(&calibration);
        envelope.attested = attested_evidence(self);
        let seal = contract.seal(&reasoning_wire, Some(*execution.as_bytes()))?;
        let result = antecedent_io::result_digest(&body).map_err(|err| io_err(&err))?;
        envelope.claim_id =
            claim_digest(&ClaimIdentityWire::new(seal, &claim_section(&envelope), result))
                .map_err(|err| io_err(&err))?;
        Ok((envelope, body))
    }

    /// Retain the full execution payload alongside a canonical result contract.
    ///
    /// # Errors
    ///
    /// Invalid response, posterior, interval or structural weights.
    pub fn fill_analysis_result_payloads(
        &self,
        wire: &mut AnalysisResultWire,
        artifact_id: &str,
    ) -> Result<(), CausalError> {
        wire.response = self
            .response()
            .map(antecedent_io::causal_response_to_wire)
            .transpose()
            .map_err(|err| io_err(&err))?;
        wire.posterior_artifact = self
            .posterior
            .as_ref()
            .map(|posterior| antecedent_io::encode_causal_posterior_bytes(posterior, artifact_id))
            .transpose()
            .map_err(|err| io_err(&err))?;
        wire.mediation_grid = self.mediation_grid().map(mediation_grid_wire);
        wire.structural_response =
            self.structural_response.as_ref().map(structural_response_wire).transpose()?;
        wire.unit_effects = self.counterfactual().map(|ite| antecedent_io::UnitEffectsWire {
            effects: ite.unit_effects.to_vec(),
            homogeneous: self
                .estimate
                .as_effect()
                .is_some_and(|effect| effect.heterogeneity.unit_effects_homogeneous),
            intervals: ite.unit_effect_intervals.as_ref().map(|intervals| {
                antecedent_io::UnitEffectIntervalsWire {
                    lower: intervals.lower.to_vec(),
                    upper: intervals.upper.to_vec(),
                    level: intervals.level,
                    method: intervals.method.to_string(),
                }
            }),
            extrapolative: ite.unit_extrapolative.as_ref().map(|flags| flags.to_vec()),
        });
        wire.counterfactual_contrasts =
            self.counterfactual_contrasts().map(counterfactual_contrasts_wire);
        Ok(())
    }

    /// Full execution body for the composite `analysis_result` container.
    ///
    /// Response values, posterior draws, mediation grids and structural atoms
    /// accompany the query, identification certificate and scalar summary.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn analysis_result_wire(
        &self,
        query: &antecedent_core::CausalQuery,
    ) -> Result<AnalysisResultWire, CausalError> {
        self.analysis_result_wire_with_context(query, None, None)
    }

    /// [`Self::analysis_result_wire`], threading the originating study's population
    /// registry and cached temporal identification. Callers that hold both the
    /// prepared study and its executed result (the composite artifact path) use
    /// this instead of re-deriving the body from the result alone, so the scalar,
    /// identification and every other field come from the one place that builds
    /// the contracted artifact.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn analysis_result_wire_with_context(
        &self,
        query: &antecedent_core::CausalQuery,
        registry: Option<&antecedent_core::PopulationRegistry>,
        temporal: Option<&CachedTemporalIdentification>,
    ) -> Result<AnalysisResultWire, CausalError> {
        body_for(&body_frame(query, temporal, None, registry)?, self)
    }

    /// The portable static-result body of an executed mediation or
    /// counterfactual result: the one producer of
    /// [`antecedent_io::StaticResultWire`]. `query` is the query that produced
    /// the result and fixes the control and active levels (the first
    /// intervention's, for a joint baseline); `registry` is the originating
    /// study's population registry.
    ///
    /// A counterfactual over several outcomes or a joint baseline carries every
    /// outcome's contrast in [`antecedent_io::StaticResultWire::contrasts`]; the
    /// scalar, the per-unit effects and the flags are the first outcome's.
    ///
    /// # Errors
    ///
    /// A result without a mediation or counterfactual payload, a mediation
    /// payload missing a component, a query without hard control and active
    /// levels ([`CausalError::Compile`]), or an identification encoding failure.
    pub fn static_result_wire(
        &self,
        query: &antecedent_core::CausalQuery,
        registry: Option<&antecedent_core::PopulationRegistry>,
    ) -> Result<antecedent_io::StaticResultWire, CausalError> {
        use antecedent_core::{CausalQuery, Intervention};
        let compile = |message: &str| CausalError::Compile { message: message.into() };
        let hard = |intervention: &Intervention| match intervention {
            Intervention::Set { value, .. } => value.as_f64(),
            _ => None,
        };
        let (control_level, active_level) = match query {
            CausalQuery::Mediation(q) => (hard(&q.control), hard(&q.active)),
            CausalQuery::Counterfactual(q) => {
                (hard(&q.control), q.interventions.first().and_then(hard))
            }
            CausalQuery::NestedCounterfactual(q) => {
                (Some(q.control_value()), Some(q.active_value()))
            }
            _ => (None, None),
        };
        let set = self.counterfactual_contrasts();
        if self.mediation().is_none() && self.counterfactual().is_none() && set.is_none() {
            return Err(compile("static result requires a mediation or counterfactual payload"));
        }
        if set.is_some() && !matches!(query, CausalQuery::Counterfactual(_)) {
            return Err(compile("a counterfactual contrast set answers a counterfactual query"));
        }
        // The first contrast's per-unit effects stand for the result's single form; a
        // population-level contrast has none.
        let primary_units =
            set.map(|set| set.primary()).filter(|primary| !primary.unit_effects.is_empty());
        let effect = self
            .estimate
            .as_effect()
            .ok_or_else(|| compile("static result requires an effect estimate"))?;
        let published = crate::PublishedScalarUncertainty::select(effect);
        Ok(antecedent_io::StaticResultWire {
            identification: antecedent_io::identification_to_wire_with_registry(
                &self.identification,
                registry,
            )?,
            estimate: effect.ate,
            standard_error: published.standard_error,
            interval_lower: published.lower,
            interval_upper: published.upper,
            assumptions: antecedent_io::assumptions_to_wire(&effect.provenance.assumptions),
            support: self.support_diagnostics().map(antecedent_io::diagnostic_to_wire).collect(),
            diagnostics: self.diagnostics.iter().map(antecedent_io::diagnostic_to_wire).collect(),
            refutations: self.refutations.iter().map(antecedent_io::refutation_to_wire).collect(),
            unit_effects: self
                .counterfactual()
                .map(|c| c.unit_effects.to_vec())
                .or_else(|| primary_units.map(|primary| primary.unit_effects.to_vec())),
            unit_extrapolative: self
                .counterfactual()
                .and_then(|c| c.unit_extrapolative.as_ref())
                .or_else(|| primary_units.and_then(|primary| primary.unit_extrapolative.as_ref()))
                .map(|flags| flags.to_vec()),
            mediation: self
                .mediation()
                .map(|m| {
                    Ok::<_, CausalError>([
                        m.total.ok_or_else(|| compile("mediation total is unavailable"))?,
                        m.direct.ok_or_else(|| compile("mediation direct is unavailable"))?,
                        m.mediated.ok_or_else(|| compile("mediation mediated is unavailable"))?,
                    ])
                })
                .transpose()?,
            control_level: control_level.ok_or_else(|| compile("missing control"))?,
            active_level: active_level.ok_or_else(|| compile("missing active"))?,
            contrasts: set
                .map(|set| counterfactual_contrasts_wire(set).contrasts)
                .unwrap_or_default(),
        })
    }
}

pub(super) fn attested_evidence(result: &StudyResult) -> Arc<[AttestedEvidence]> {
    let names: std::collections::HashSet<&str> =
        result.extras.custom_validator_names.iter().map(Arc::as_ref).collect();
    result
        .refutations
        .iter()
        .filter(|report| names.contains(report.refuter.as_ref()))
        .map(|report| AttestedEvidence {
            name: Arc::clone(&report.refuter),
            kind: Arc::from("custom_validator"),
            passed: report.passed,
            refuted_ate: Some(report.refuted_ate),
            comparison: Some(report.comparison),
            informative: report.informative,
            failure_condition: report.failure_condition.clone(),
            reverifiable: false,
            config_digest: None,
            payload_digest: None,
        })
        .collect()
}

pub(super) fn claim_section(claim: &ClaimEnvelope) -> ClaimSectionWire {
    ClaimSectionWire {
        claim_id: *claim.claim_id.as_bytes(),
        kind: claim.kind.as_str().into(),
        value_bits: claim.value.filter(|value| value.is_finite()).map(f64::to_bits),
        execution: claim.execution.map(|digest| *digest.as_bytes()),
        identification_domain: claim.domains.identification.as_str().into(),
        support_domain: claim.domains.support.as_str().into(),
        evaluated_domain: claim.domains.evaluated.as_str().into(),
        calibration: calibration_wire(&claim.calibration),
        attested: claim
            .attested
            .iter()
            .map(|item| AttestedEvidenceWire {
                name: item.name.to_string(),
                kind: item.kind.to_string(),
                passed: item.passed,
                refuted_ate: item.refuted_ate,
                comparison: item.comparison,
                informative: item.informative,
                failure_condition: item
                    .failure_condition
                    .as_ref()
                    .map(std::string::ToString::to_string),
                reverifiable: item.reverifiable,
                config_digest: item.config_digest.as_ref().map(std::string::ToString::to_string),
                payload_digest: item.payload_digest.as_ref().map(std::string::ToString::to_string),
            })
            .collect(),
    }
}

/// The wire section of a counterfactual contrast set: one entry per outcome, in the query's
/// order, each with its per-unit effects, intervals and extrapolation flags.
fn counterfactual_contrasts_wire(
    set: &crate::gcm::CounterfactualContrasts,
) -> antecedent_io::CounterfactualContrastsWire {
    antecedent_io::CounterfactualContrastsWire {
        noise_inference: format!("{:?}", set.noise_inference),
        contrasts: set
            .contrasts
            .iter()
            .map(|contrast| antecedent_io::OutcomeContrastWire {
                outcome: contrast.outcome.raw(),
                mean: contrast.mean_ite,
                standard_error: contrast.standard_error,
                unit_effects: antecedent_io::UnitEffectsWire {
                    effects: contrast.unit_effects.to_vec(),
                    homogeneous: contrast.unit_effects_homogeneous,
                    intervals: contrast.unit_effect_intervals.as_ref().map(|intervals| {
                        antecedent_io::UnitEffectIntervalsWire {
                            lower: intervals.lower.to_vec(),
                            upper: intervals.upper.to_vec(),
                            level: intervals.level,
                            method: intervals.method.to_string(),
                        }
                    }),
                    extrapolative: contrast.unit_extrapolative.as_ref().map(|flags| flags.to_vec()),
                },
            })
            .collect(),
    }
}
