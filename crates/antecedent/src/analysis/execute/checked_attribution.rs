//! Builder-independent execution for frequentist DAG attribution operations.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[derive(Clone, Debug)]
enum AttributionTarget {
    Anomaly(antecedent_core::AnomalyAttributionQuery),
    Change(antecedent_core::ChangeAttributionQuery),
}

/// A frozen GCM attribution target and its prepare-time identification contract.
#[derive(Clone)]
pub(crate) struct CheckedAttributionOperation {
    graph: Dag,
    target: AttributionTarget,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedAttributionOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedAttributionOperation")
            .field("target", &self.target)
            .field("estimand", &self.estimand)
            .field("identification_status", &self.identification.status)
            .field("identifier", &IdentifierId::GcmParametric)
            .field("estimator", &EstimatorId::GcmFit)
            .finish_non_exhaustive()
    }
}

impl CheckedAttributionOperation {
    pub(crate) fn checked(
        study: &Study,
        physical: &PhysicalExecutionPlan,
        cache: &crate::analysis::prepared::CachedStaticIdentification,
    ) -> Result<Self, CausalError> {
        let graph = study.graph.as_dag().ok_or(CausalError::Unsupported {
            message: "checked attribution requires a supplied DAG",
        })?;
        let target = match (&study.query, &study.inference) {
            (CausalQuery::AnomalyAttribution(query), InferenceMode::Frequentist) => {
                query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                AttributionTarget::Anomaly(query.clone())
            }
            (CausalQuery::ChangeAttribution(query), InferenceMode::Frequentist) => {
                query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                AttributionTarget::Change(query.clone())
            }
            _ => {
                return Err(CausalError::Compile {
                    message:
                        "checked attribution supports only frequentist anomaly or change targets"
                            .into(),
                });
            }
        };
        let target_query = target.query();
        if !study.custom_validators.is_empty() {
            return Err(CausalError::Unsupported {
                message: "checked attribution does not support custom validators",
            });
        }
        let (identifier, estimator) = match &target {
            AttributionTarget::Anomaly(_) => ("gcm.parametric", "gcm.fit"),
            AttributionTarget::Change(_) => ("gcm.parametric", "gcm.fit"),
        };
        if study.structure_source != crate::support::StructureSource::Explicit
            || cache.identification.query != target_query
            || physical.logical.query != target_query
            || physical.logical.record.identifier.as_deref() != Some(identifier)
            || physical.logical.record.estimator.as_deref() != Some(estimator)
        {
            return Err(CausalError::Compile {
                message: "attribution target, source graph, identification, or estimator does not match the checked operation".into(),
            });
        }
        let estimand = cache.estimand.clone();
        if !cache.identification.estimands.iter().any(|candidate| {
            candidate.method == estimand.method && candidate.functional == estimand.functional
        }) {
            return Err(CausalError::Compile {
                message: "attribution estimand is absent from its identification result".into(),
            });
        }
        Ok(Self {
            graph: graph.clone(),
            target,
            identification: cache.identification.clone(),
            estimand,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn query(&self) -> CausalQuery {
        self.target.query()
    }

    pub(crate) fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn identification(&self) -> &IdentificationResult {
        &self.identification
    }

    pub(crate) fn procedure(&self) -> (&'static str, &'static str) {
        (IdentifierId::GcmParametric.as_str(), EstimatorId::GcmFit.as_str())
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.result_context.query != self.query()
            || self.identification.query != self.query()
            || !self.identification.estimands.iter().any(|candidate| {
                candidate.method == self.estimand.method
                    && candidate.functional == self.estimand.functional
            })
        {
            return Err(CausalError::Compile {
                message: "attribution operation lost its checked target or identification binding"
                    .into(),
            });
        }
        match &self.target {
            AttributionTarget::Anomaly(query) => {
                let started = Instant::now();
                let fitted = fit_gcm(self.graph.clone(), data)?;
                let scores = anomaly_attribution_with(
                    &fitted.model,
                    data,
                    query.targets.iter().copied(),
                    query.max_units,
                    ctx,
                )?;
                let outcome = *query.targets.first().ok_or_else(|| CausalError::Compile {
                    message: "checked anomaly target has no outcome variable".into(),
                })?;
                Ok(finish_identified_execute_with_context(
                    &self.result_context,
                    Some(data),
                    IdentifiedExecuteFinish {
                        physical: &self.physical,
                        identification: self.identification.clone(),
                        estimand: self.estimand.clone(),
                        estimate: nan_effect(),
                        identifier_id: IdentifierId::GcmParametric,
                        estimator_id: EstimatorId::GcmFit,
                        treatment: outcome,
                        outcome,
                        identify_cached: true,
                        extra_diagnostics: Vec::new(),
                        refutations: Vec::new(),
                        distribution: None,
                        mediation: None,
                        wall_time_ns: u64::try_from(started.elapsed().as_nanos())
                            .unwrap_or(u64::MAX),
                        bootstrap_replicates_ok: None,
                        cancelled: false,
                        early_stopped: false,
                        extras: IdentifiedExecuteExtras {
                            gcm: Some(GcmSlot::Anomaly(scores)),
                            empty_provenance: true,
                            ..Default::default()
                        },
                    },
                ))
            }
            AttributionTarget::Change(query) => {
                let started = Instant::now();
                let fitted = fit_gcm(self.graph.clone(), data)?;
                let change = attribute_distribution_change(
                    &fitted.model,
                    data,
                    query,
                    &antecedent_attribution::DistributionChangeOptions::default(),
                    ctx,
                )?;
                let estimate = EffectEstimate::new(
                    change.total_change,
                    f64::NAN,
                    antecedent_core::AssumptionSet::default(),
                    OverlapPolicy::ExplicitOverride,
                );
                Ok(finish_identified_execute_with_context(
                    &self.result_context,
                    Some(data),
                    IdentifiedExecuteFinish {
                        physical: &self.physical,
                        identification: self.identification.clone(),
                        estimand: self.estimand.clone(),
                        estimate,
                        identifier_id: IdentifierId::GcmParametric,
                        estimator_id: EstimatorId::GcmFit,
                        treatment: query.outcome,
                        outcome: query.outcome,
                        identify_cached: true,
                        extra_diagnostics: Vec::new(),
                        refutations: Vec::new(),
                        distribution: None,
                        mediation: None,
                        wall_time_ns: u64::try_from(started.elapsed().as_nanos())
                            .unwrap_or(u64::MAX),
                        bootstrap_replicates_ok: None,
                        cancelled: false,
                        early_stopped: false,
                        extras: IdentifiedExecuteExtras {
                            gcm: Some(GcmSlot::Change(change)),
                            empty_provenance: true,
                            ..Default::default()
                        },
                    },
                ))
            }
        }
    }
}

impl AttributionTarget {
    fn query(&self) -> CausalQuery {
        match self {
            Self::Anomaly(query) => CausalQuery::AnomalyAttribution(query.clone()),
            Self::Change(query) => CausalQuery::ChangeAttribution(query.clone()),
        }
    }
}
