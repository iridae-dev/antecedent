//! Checked TemporalCpdag class-envelope and graph-posterior temporal mediation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::prepared::{
    CachedDbnPosteriorIdentification, CachedTemporalClassIdentification,
    CachedTemporalClassPosteriorIdentification,
};
use antecedent_discovery::GraphPosteriorAtomKind;

/// Structural proof retained for a checked class or posterior mediation route.
///
/// Each variant holds the frozen structure and the prepare-time identification
/// products that the execution composes over. Refresh re-checks the structure
/// and the plan against these before any estimation runs.
#[derive(Clone)]
pub(crate) enum CheckedTemporalMediationProof {
    /// Fixed TemporalCpdag: one completion envelope per requested horizon.
    ClassEnvelope {
        graph: AcceptedGraph,
        cache: CachedTemporalClassIdentification,
        max_completions: Option<usize>,
        class_prior: Option<crate::ClassPrior>,
    },
    /// TemporalCpdag posterior atoms, each carrying its completion envelope.
    ClassPosterior {
        posterior: GraphPosterior,
        cache: CachedTemporalClassPosteriorIdentification,
        max_completions: Option<usize>,
    },
    /// TemporalDag (DBN) posterior atoms with per-horizon mediation sets.
    DbnPosterior { posterior: GraphPosterior, cache: CachedDbnPosteriorIdentification },
}

impl CheckedTemporalMediationProof {
    fn name(&self) -> &'static str {
        match self {
            Self::ClassEnvelope { .. } => "class_envelope",
            Self::ClassPosterior { .. } => "class_posterior",
            Self::DbnPosterior { .. } => "dbn_posterior",
        }
    }
}

/// Frozen mediation target, structural proof, inference procedure, and
/// validation contract for a TemporalCpdag envelope or a temporal graph
/// posterior. Execution composes the retained proof over the existing class
/// and posterior mediation executors without consulting the builder.
#[derive(Clone)]
pub(crate) struct CheckedTemporalClassMediationOperation {
    proof: CheckedTemporalMediationProof,
    query: antecedent_core::MediationQuery,
    inference: InferenceMode,
    bootstrap_replicates: u32,
    validation: RefuteSuite,
    physical: PhysicalExecutionPlan,
    schema: antecedent_core::CausalSchema,
    structure_source: crate::support::StructureSource,
    graph_class: GraphClass,
    graph_version: u32,
}

impl std::fmt::Debug for CheckedTemporalClassMediationOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedTemporalClassMediationOperation")
            .field("proof", &self.proof.name())
            .field("query", &self.query)
            .field("graph_class", &self.graph_class)
            .field("structure_source", &self.structure_source)
            .field("estimator", &self.estimator())
            .field("validation", &self.validation)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .finish_non_exhaustive()
    }
}

impl CheckedTemporalClassMediationOperation {
    /// Whether `study` is a coordinate this operation seals: a series mediation
    /// target on a fixed TemporalCpdag, a TemporalCpdag posterior, or a
    /// TemporalDag posterior, with the default point-estimate settings.
    ///
    /// The one-shot facade and the prepared builder share this predicate so a
    /// study that prepares into this operation is exactly the study the facade
    /// expects to retain it.
    #[must_use]
    pub(crate) fn admits(study: &Study) -> bool {
        let CausalQuery::Mediation(query) = &study.query else {
            return false;
        };
        if !matches!(study.data, DataInput::Temporal(_) | DataInput::Event(_))
            || study.tiered.is_some()
            || study.split.is_some()
            || !study.custom_validators.is_empty()
            || !matches!(study.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
        {
            return false;
        }
        if let InferenceMode::Bayesian(config) = &study.inference {
            if config.prior.is_some()
                || config.prior_artifact.is_some()
                || config.external_compose.is_some()
            {
                return false;
            }
        }
        match (study.graph_posterior.as_ref(), study.graph.class(), study.structure_source) {
            (
                None,
                GraphClass::TemporalCpdag,
                crate::support::StructureSource::Explicit
                | crate::support::StructureSource::Accepted,
            ) => study.identifier.is_none_or(|id| id == IdentifierId::GeneralizedAdjustment),
            (
                Some(posterior),
                GraphClass::TemporalCpdag,
                crate::support::StructureSource::GraphPosterior,
            ) => posterior.atom_kind == GraphPosteriorAtomKind::Cpdag && query.horizons.len() == 1,
            (
                Some(posterior),
                GraphClass::TemporalDag,
                crate::support::StructureSource::GraphPosterior,
            ) => {
                posterior.atom_kind == GraphPosteriorAtomKind::Dag
                    && (matches!(study.inference, InferenceMode::Bayesian(_))
                        || query.horizons.len() == 1)
            }
            _ => false,
        }
    }

    /// Check the prepared caches against a fresh identification of the same
    /// structure and freeze them with the target, procedure, and plan.
    pub(crate) fn checked(
        study: &Study,
        data: &TimeSeriesData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Mediation(query) = &study.query else {
            return Err(CausalError::Compile {
                message:
                    "checked temporal class mediation requires a TemporalMediationEffect target"
                        .into(),
            });
        };
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if !Self::admits(study) || physical.logical.query != study.query {
            return Err(CausalError::Compile {
                message: "temporal class mediation structure, target, validation, or plan changed after preparation".into(),
            });
        }
        let expected_estimator = match &study.inference {
            InferenceMode::Frequentist => EstimatorId::TemporalMediation,
            InferenceMode::Bayesian(_) => EstimatorId::BayesianTemporalMediation,
        };
        if physical.logical.record.estimator.as_deref() != Some(expected_estimator.as_str()) {
            return Err(CausalError::Compile {
                message:
                    "temporal class mediation estimator differs from its checked physical plan"
                        .into(),
            });
        }
        let variables: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        let proof = match study.graph_posterior.as_ref() {
            None => {
                Self::check_plan_record(
                    physical,
                    "temporal_mediation_class",
                    "generalized.adjustment",
                )?;
                let cache =
                    study.temporal_class_identification_cache.as_deref().ok_or_else(|| {
                        CausalError::Compile {
                            message: "temporal class mediation proof was not prepared".into(),
                        }
                    })?;
                Self::check_class_envelope(study, query, cache)?;
                CheckedTemporalMediationProof::ClassEnvelope {
                    graph: study.graph.clone(),
                    cache: cache.clone(),
                    max_completions: study.max_completions,
                    class_prior: study.class_prior.clone(),
                }
            }
            Some(posterior) if posterior.atom_kind == GraphPosteriorAtomKind::Cpdag => {
                Self::check_plan_record(physical, "temporal_mediation", "temporal.mediation")?;
                let cache = study
                    .temporal_class_posterior_identification_cache
                    .as_deref()
                    .ok_or_else(|| CausalError::Compile {
                        message: "temporal class posterior mediation proof was not prepared".into(),
                    })?;
                Self::check_class_posterior(
                    posterior,
                    &variables,
                    query,
                    study.max_completions,
                    cache,
                    ctx,
                )?;
                CheckedTemporalMediationProof::ClassPosterior {
                    posterior: posterior.clone(),
                    cache: cache.clone(),
                    max_completions: study.max_completions,
                }
            }
            Some(posterior) => {
                Self::check_plan_record(physical, "temporal_mediation", "temporal.mediation")?;
                let cache =
                    study.dbn_posterior_identification_cache.as_deref().ok_or_else(|| {
                        CausalError::Compile {
                            message: "DBN posterior mediation proof was not prepared".into(),
                        }
                    })?;
                Self::check_dbn_posterior(posterior, &variables, query, cache, ctx)?;
                CheckedTemporalMediationProof::DbnPosterior {
                    posterior: posterior.clone(),
                    cache: cache.clone(),
                }
            }
        };
        Ok(Self {
            proof,
            query: query.clone(),
            inference: study.inference.clone(),
            bootstrap_replicates: study.bootstrap_replicates,
            validation: study.refute,
            physical: physical.clone(),
            schema: data.schema().clone(),
            structure_source: study.structure_source,
            graph_class: study.graph.class(),
            graph_version: study.graph.version(),
        })
    }

    fn check_plan_record(
        physical: &PhysicalExecutionPlan,
        plan_id: &str,
        identifier: &str,
    ) -> Result<(), CausalError> {
        if physical.logical.record.plan_id.as_ref() != plan_id
            || physical.logical.record.identifier.as_deref() != Some(identifier)
        {
            return Err(CausalError::Compile {
                message: "temporal class mediation plan record differs from its retained proof"
                    .into(),
            });
        }
        Ok(())
    }

    /// Replay the per-horizon completion envelopes from the source graph.
    fn check_class_envelope(
        study: &Study,
        query: &antecedent_core::MediationQuery,
        cache: &CachedTemporalClassIdentification,
    ) -> Result<(), CausalError> {
        let mut fresh = study.clone();
        fresh.temporal_class_identification_cache = None;
        for &horizon in query.horizons.iter() {
            let (_, cached) = cache
                .by_horizon
                .iter()
                .find(|(member, _)| *member == horizon)
                .ok_or_else(|| CausalError::Compile {
                    message: format!("temporal class mediation proof missing horizon {horizon}"),
                })?;
            let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
            witness.horizon_steps = horizon;
            let expected =
                fresh.identify_temporal_class(IdentifierId::GeneralizedAdjustment, &witness)?;
            if cached.envelope.cases.is_empty() || !same_class_envelope(&expected.envelope, cached)
            {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal class mediation proof failed replay at horizon {horizon}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Replay every TemporalCpdag posterior atom's completion envelope.
    fn check_class_posterior(
        posterior: &GraphPosterior,
        variables: &[antecedent_core::VariableId],
        query: &antecedent_core::MediationQuery,
        max_completions: Option<usize>,
        cache: &CachedTemporalClassPosteriorIdentification,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let horizon = query.horizons.first().copied().ok_or_else(|| CausalError::Compile {
            message: "temporal class posterior mediation requires a horizon".into(),
        })?;
        let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
        witness.horizon_steps = horizon;
        let expected =
            crate::analysis::prepared::build_temporal_class_posterior_identification_cache(
                posterior,
                variables,
                &witness,
                max_completions,
                ctx,
            )?;
        let atoms_agree = expected.class_atoms.len() == cache.class_atoms.len()
            && expected.class_atoms.iter().zip(cache.class_atoms.iter()).all(|(a, b)| {
                a.key == b.key
                    && a.identified_weight.to_bits() == b.identified_weight.to_bits()
                    && a.truncated_completions == b.truncated_completions
                    && a.identification.status == b.identification.status
                    && a.invariant.as_ref().map(|e| &e.adjustment_set)
                        == b.invariant.as_ref().map(|e| &e.adjustment_set)
                    && same_class_envelope(&a.envelope, &b.envelope)
            });
        if expected.graphs != cache.graphs
            || cache.graphs.weights.as_ref() != posterior.weights.as_ref()
            || cache.graphs.n_samples != posterior.n_graphs
            || !atoms_agree
        {
            return Err(CausalError::Compile {
                message: "temporal class posterior mediation proof failed replay".into(),
            });
        }
        Ok(())
    }

    /// Replay every DBN posterior atom's per-horizon mediation identification.
    fn check_dbn_posterior(
        posterior: &GraphPosterior,
        variables: &[antecedent_core::VariableId],
        query: &antecedent_core::MediationQuery,
        cache: &CachedDbnPosteriorIdentification,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let expected =
            crate::analysis::prepared::build_dbn_posterior_mediation_identification_cache(
                posterior, variables, query, ctx,
            )?;
        let atoms_agree = expected.atoms.len() == cache.atoms.len()
            && expected.atoms.iter().zip(cache.atoms.iter()).all(|(a, b)| {
                a.key == b.key
                    && same_estimand(&a.estimand, &b.estimand)
                    && a.identification.status == b.identification.status
                    && a.identification.query == b.identification.query
                    && a.indexer == b.indexer
                    && match (a.horizons.as_ref(), b.horizons.as_ref()) {
                        (None, None) => true,
                        (Some(x), Some(y)) => {
                            x.by_horizon.len() == y.by_horizon.len()
                                && x.by_horizon.iter().zip(y.by_horizon.iter()).all(|(p, q)| {
                                    p.horizon == q.horizon
                                        && same_estimand(&p.estimand, &q.estimand)
                                        && p.identification.status == q.identification.status
                                        && p.indexer == q.indexer
                                })
                        }
                        _ => false,
                    }
            });
        if expected.graphs != cache.graphs
            || cache.graphs.weights.as_ref() != posterior.weights.as_ref()
            || cache.graphs.n_samples != posterior.n_graphs
            || expected.identify_demotion != cache.identify_demotion
            || expected.horizon_demotions.as_ref() != cache.horizon_demotions.as_ref()
            || !atoms_agree
        {
            return Err(CausalError::Compile {
                message: "DBN posterior mediation proof failed replay".into(),
            });
        }
        Ok(())
    }

    pub(crate) fn estimator(&self) -> EstimatorId {
        match &self.inference {
            InferenceMode::Frequentist => EstimatorId::TemporalMediation,
            InferenceMode::Bayesian(_) => EstimatorId::BayesianTemporalMediation,
        }
    }

    pub(crate) fn query(&self) -> &antecedent_core::MediationQuery {
        &self.query
    }

    pub(crate) fn validation(&self) -> RefuteSuite {
        self.validation
    }

    pub(crate) fn bootstrap_replicates(&self) -> u32 {
        self.bootstrap_replicates
    }

    pub(crate) fn graph_class(&self) -> GraphClass {
        self.graph_class
    }

    pub(crate) fn structure_source(&self) -> crate::support::StructureSource {
        self.structure_source
    }

    pub(crate) fn identifier(&self) -> Arc<str> {
        self.physical
            .logical
            .record
            .identifier
            .clone()
            .unwrap_or_else(|| Arc::from("temporal.mediation"))
    }

    /// Structural atoms the proof retains and the enumeration or posterior mass
    /// that identified: `(atom_count, identified_mass, unidentified_mass)`.
    pub(crate) fn proof_mass(&self) -> (usize, f64, f64) {
        match &self.proof {
            CheckedTemporalMediationProof::ClassEnvelope { cache, .. } => {
                let envelope =
                    cache.by_horizon.first().map_or(&cache.envelope, |(_, envelope)| envelope);
                (
                    envelope.envelope.cases.len(),
                    envelope.envelope.identified_weight.0,
                    envelope.envelope.unidentified_weight.0,
                )
            }
            CheckedTemporalMediationProof::ClassPosterior { cache, .. } => {
                let unidentified = cache.graphs.unidentified_mass();
                (cache.graphs.n_samples, 1.0 - unidentified, unidentified)
            }
            CheckedTemporalMediationProof::DbnPosterior { cache, .. } => {
                let unidentified = cache.graphs.unidentified_mass();
                (cache.graphs.n_samples, 1.0 - unidentified, unidentified)
            }
        }
    }

    /// Stable identity of the retained structure: the graph or the posterior's
    /// frozen weights, adjacency, and lag masks.
    pub(crate) fn proof_signature(&self) -> Arc<str> {
        match &self.proof {
            CheckedTemporalMediationProof::ClassEnvelope { graph, .. } => {
                Arc::from(format!("{graph:?}"))
            }
            CheckedTemporalMediationProof::ClassPosterior { posterior, .. }
            | CheckedTemporalMediationProof::DbnPosterior { posterior, .. } => Arc::from(format!(
                "{}:{:?}:{:?}:{:?}:{:?}:{:?}",
                self.proof.name(),
                posterior.weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
                posterior.adjacency,
                posterior.graph_keys,
                posterior.lag_masks,
                posterior.max_lag,
            )),
        }
    }

    /// Re-check the structure a study carries against the retained proof and
    /// the frozen plan before the proof is composed over it.
    pub(crate) fn verify_structure(&self, study: &Study) -> Result<(), CausalError> {
        if study.query != self.physical.logical.query
            || study.graph.class() != self.graph_class
            || study.graph.version() != self.graph_version
            || study.structure_source != self.structure_source
        {
            return Err(CausalError::Compile {
                message:
                    "checked temporal class mediation target or structure changed after preparation"
                        .into(),
            });
        }
        let agrees = match (&self.proof, study.graph_posterior.as_ref()) {
            (CheckedTemporalMediationProof::ClassEnvelope { graph, .. }, None) => {
                format!("{graph:?}") == format!("{:?}", study.graph)
            }
            (
                CheckedTemporalMediationProof::ClassPosterior { posterior, .. }
                | CheckedTemporalMediationProof::DbnPosterior { posterior, .. },
                Some(current),
            ) => same_posterior(posterior, current),
            _ => false,
        };
        if !agrees {
            return Err(CausalError::Compile {
                message:
                    "checked temporal class mediation structure differs from its retained proof"
                        .into(),
            });
        }
        Ok(())
    }

    pub(crate) fn rebind(&self, data: &TimeSeriesData) -> Result<(), CausalError> {
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "temporal class mediation refresh changed the semantic variable schema"
                    .into(),
            });
        }
        Ok(())
    }

    /// Compose the retained proof over the class or posterior mediation
    /// executor. `base` supplies only execution-context settings (latency,
    /// stage sink, population registry); the target, structure, proof,
    /// procedure, and validation all come from this operation.
    pub(crate) fn execute(
        &self,
        base: &Study,
        input: &DataInput,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let (DataInput::Temporal(series) | DataInput::Event(series)) = input else {
            return Err(CausalError::Compile {
                message: "checked temporal class mediation requires series data".into(),
            });
        };
        self.rebind(series)?;
        self.verify_structure(base)?;
        let mut click = base.clone();
        click.data = input.clone();
        click.query = CausalQuery::Mediation(self.query.clone());
        click.inference = self.inference.clone();
        click.refute = self.validation;
        click.bootstrap_replicates = self.bootstrap_replicates;
        click.custom_validators = Vec::new();
        click.split = None;
        click.tiered = None;
        click.temporal_identification_cache = None;
        click.temporal_class_identification_cache = None;
        click.dbn_posterior_identification_cache = None;
        click.temporal_class_posterior_identification_cache = None;
        match &self.proof {
            CheckedTemporalMediationProof::ClassEnvelope {
                graph,
                cache,
                max_completions,
                class_prior,
            } => {
                click.graph = graph.clone();
                click.graph_posterior = None;
                click.identifier = Some(IdentifierId::GeneralizedAdjustment);
                click.max_completions = *max_completions;
                click.class_prior.clone_from(class_prior);
                click.temporal_class_identification_cache = Some(Arc::new(cache.clone()));
                click.execute_temporal_cpdag_mediation(series, &self.query, &self.physical, ctx)
            }
            CheckedTemporalMediationProof::ClassPosterior { posterior, cache, max_completions } => {
                click.graph_posterior = Some(posterior.clone());
                click.max_completions = *max_completions;
                click.temporal_class_posterior_identification_cache = Some(Arc::new(cache.clone()));
                click.execute_temporal_class_graph_posterior_mediation(
                    series,
                    posterior,
                    &self.query,
                    &self.physical,
                    ctx,
                )
            }
            CheckedTemporalMediationProof::DbnPosterior { posterior, cache } => {
                click.graph_posterior = Some(posterior.clone());
                click.dbn_posterior_identification_cache = Some(Arc::new(cache.clone()));
                click.execute_dbn_posterior_mediation(
                    series,
                    posterior,
                    &self.query,
                    &self.physical,
                    ctx,
                )
            }
        }
    }
}

fn same_estimand(a: &IdentifiedEstimand, b: &IdentifiedEstimand) -> bool {
    a.method == b.method
        && a.adjustment_set == b.adjustment_set
        && a.mediators == b.mediators
        && a.instruments == b.instruments
}

fn same_class_envelope(
    expected: &antecedent_identify::TemporalClassEnvelope,
    supplied: &antecedent_identify::TemporalClassEnvelope,
) -> bool {
    let left = &expected.envelope;
    let right = &supplied.envelope;
    expected.indexers == supplied.indexers
        && left.cases.len() == right.cases.len()
        && left.status == right.status
        && left.identified_weight.0.to_bits() == right.identified_weight.0.to_bits()
        && left.unidentified_weight.0.to_bits() == right.unidentified_weight.0.to_bits()
        && left.truncated_completions == right.truncated_completions
        && left.cases.iter().zip(&right.cases).all(|(a, b)| {
            a.graph.fingerprint() == b.graph.fingerprint()
                && a.weight.0.to_bits() == b.weight.0.to_bits()
                && a.result.query == b.result.query
                && a.result.status == b.result.status
                && a.result.estimands.len() == b.result.estimands.len()
                && a.result
                    .estimands
                    .iter()
                    .zip(&b.result.estimands)
                    .all(|(x, y)| same_estimand(x, y))
        })
}

fn same_posterior(retained: &GraphPosterior, current: &GraphPosterior) -> bool {
    retained.n_vars == current.n_vars
        && retained.n_graphs == current.n_graphs
        && retained.atom_kind == current.atom_kind
        && retained.max_lag == current.max_lag
        && retained.adjacency.as_ref() == current.adjacency.as_ref()
        && retained.graph_keys.as_ref() == current.graph_keys.as_ref()
        && retained.lag_masks.as_deref() == current.lag_masks.as_deref()
        && retained.weights.len() == current.weights.len()
        && retained
            .weights
            .iter()
            .zip(current.weights.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}
