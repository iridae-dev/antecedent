//! Checked `TemporalCpdag` class-envelope and graph-posterior temporal mediation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AcceptedGraph, Arc, CausalError, CausalQuery, DataInput, EstimatorId, ExecutionContext,
    GraphClass, GraphPosterior, IdentifierId, InferenceMode, PhysicalExecutionPlan, RefuteSuite,
    Study, StudyResult, TableView, TimeSeriesData,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::prepared::{
    CachedDbnPosteriorIdentification, CachedTemporalClassIdentification,
    CachedTemporalClassPosteriorIdentification,
};
use crate::analysis::route_guards::same_accepted_graph;
use antecedent_discovery::GraphPosteriorAtomKind;

/// Structural proof retained for a checked class or posterior mediation route.
///
/// Each variant holds the frozen structure and the prepare-time identification
/// products that the execution composes over. Refresh re-checks the structure
/// and the plan against these before any estimation runs.
#[derive(Clone)]
pub(crate) enum CheckedTemporalMediationProof {
    /// Fixed `TemporalCpdag`: one completion envelope per requested horizon.
    ClassEnvelope {
        graph: AcceptedGraph,
        cache: Arc<CachedTemporalClassIdentification>,
        max_completions: Option<usize>,
        class_prior: Option<crate::ClassPrior>,
    },
    /// `TemporalCpdag` posterior atoms, each carrying its completion envelope.
    ClassPosterior {
        posterior: GraphPosterior,
        cache: Arc<CachedTemporalClassPosteriorIdentification>,
        max_completions: Option<usize>,
    },
    /// `TemporalDag` (DBN) posterior atoms with per-horizon mediation sets.
    DbnPosterior { posterior: GraphPosterior, cache: Arc<CachedDbnPosteriorIdentification> },
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
/// validation contract for a `TemporalCpdag` envelope or a temporal graph
/// posterior. Execution composes the retained proof over the existing class
/// and posterior mediation executors without consulting the builder.
#[derive(Clone)]
pub(crate) struct CheckedTemporalClassMediationOperation {
    proof: CheckedTemporalMediationProof,
    query: antecedent_core::MediationQuery,
    target_bindings: Arc<[CheckedTemporalMediationTargetInfo]>,
    inference: InferenceMode,
    bootstrap_replicates: u32,
    validation: RefuteSuite,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
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
    /// target on a fixed `TemporalCpdag`, a `TemporalCpdag` posterior, or a
    /// `TemporalDag` posterior, with the default point-estimate settings.
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
            || !study.refute.is_point()
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
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Mediation(query) = &study.query else {
            return Err(CausalError::Compile {
                message:
                    "checked temporal class mediation requires a TemporalMediationEffect target"
                        .into(),
            });
        };
        query.validate()?;
        if !Self::admits(study) || physical.logical.query != study.query {
            return Err(CausalError::Compile {
                message: "temporal class mediation structure, target, validation, or plan changed after preparation".into(),
            });
        }
        let expected_estimator = EstimatorId::temporal_mediation_for(&study.inference);
        if !physical.binding().estimator.is(expected_estimator) {
            return Err(CausalError::Compile {
                message:
                    "temporal class mediation estimator differs from its checked physical plan"
                        .into(),
            });
        }
        let proof = match study.graph_posterior.as_ref() {
            None => Self::class_envelope_proof(study, physical, query)?,
            Some(posterior) if posterior.atom_kind == GraphPosteriorAtomKind::Cpdag => {
                Self::class_posterior_proof(study, physical, posterior)?
            }
            Some(posterior) => Self::dbn_posterior_proof(study, physical, posterior)?,
        };
        let target_bindings =
            retained_mediation_targets(&proof, query, data.schema(), &study.inference)?;
        Ok(Self {
            proof,
            query: query.clone(),
            target_bindings: target_bindings.into(),
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

    /// The proof of an explicit `TemporalCpdag` mediation: the prepared class
    /// envelope must cover every requested horizon with completions.
    fn class_envelope_proof(
        study: &Study,
        physical: &PhysicalExecutionPlan,
        query: &antecedent_core::MediationQuery,
    ) -> Result<CheckedTemporalMediationProof, CausalError> {
        Self::check_plan_record(physical, "temporal_mediation_class", "generalized.adjustment")?;
        let cache = study.temporal_class_cache().ok_or_else(|| {
            CausalError::Compile {
                message: "temporal class mediation proof was not prepared".into(),
            }
        })?;
        for &horizon in query.horizons.iter() {
            let cached = cache
                .by_horizon
                .iter()
                .find(|(member, _)| *member == horizon)
                .map(|(_, cached)| cached)
                .ok_or_else(|| CausalError::Compile {
                    message: format!("temporal class mediation proof missing horizon {horizon}"),
                })?;
            if cached.envelope.cases.is_empty() {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal class mediation proof has no completions at horizon {horizon}"
                    ),
                });
            }
        }

        Ok(CheckedTemporalMediationProof::ClassEnvelope {
            graph: (*study.graph).clone(),
            cache: Arc::clone(cache),
            max_completions: study.max_completions,
            class_prior: study.class_prior.clone(),
        })
    }

    /// The proof of a `TemporalCpdag` graph posterior mediation: the prepared
    /// class posterior cache must bind the frozen posterior weights.
    fn class_posterior_proof(
        study: &Study,
        physical: &PhysicalExecutionPlan,
        posterior: &antecedent_discovery::GraphPosterior,
    ) -> Result<CheckedTemporalMediationProof, CausalError> {
        Self::check_plan_record(physical, "temporal_mediation", "temporal.mediation")?;
        let cache =
            study.temporal_class_posterior_cache().ok_or_else(|| {
                CausalError::Compile {
                    message: "temporal class posterior mediation proof was not prepared".into(),
                }
            })?;
        if cache.graphs.weights.as_ref() != posterior.weights.as_ref()
            || cache.graphs.n_samples != posterior.n_graphs
        {
            return Err(CausalError::Compile {
                message:
                    "temporal class posterior mediation proof does not bind the frozen posterior"
                        .into(),
            });
        }

        let CausalQuery::Mediation(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "class posterior requires a mediation target".into(),
            });
        };
        let cache = retain_class_mediation_proofs(cache, query)?;
        Ok(CheckedTemporalMediationProof::ClassPosterior {
            posterior: posterior.clone(),
            cache: Arc::new(cache),
            max_completions: study.max_completions,
        })
    }

    /// The proof of a `TemporalDag` graph posterior mediation: the prepared DBN
    /// posterior cache must hold atoms and bind the frozen posterior weights.
    fn dbn_posterior_proof(
        study: &Study,
        physical: &PhysicalExecutionPlan,
        posterior: &antecedent_discovery::GraphPosterior,
    ) -> Result<CheckedTemporalMediationProof, CausalError> {
        Self::check_plan_record(physical, "temporal_mediation", "temporal.mediation")?;
        let cache = study.dbn_posterior_cache().ok_or_else(|| {
            CausalError::Compile {
                message: "DBN posterior mediation proof was not prepared".into(),
            }
        })?;
        if cache.atoms.is_empty() {
            return Err(CausalError::Compile {
                message: "DBN posterior mediation proof retained no atoms".into(),
            });
        }
        if cache.graphs.weights.as_ref() != posterior.weights.as_ref()
            || cache.graphs.n_samples != posterior.n_graphs
        {
            return Err(CausalError::Compile {
                message: "DBN posterior mediation proof does not bind the frozen posterior".into(),
            });
        }

        Ok(CheckedTemporalMediationProof::DbnPosterior {
            posterior: posterior.clone(),
            cache: Arc::clone(cache),
        })
    }

    fn check_plan_record(
        physical: &PhysicalExecutionPlan,
        plan_id: &str,
        identifier: &str,
    ) -> Result<(), CausalError> {
        if physical.logical.record.plan_id.as_ref() != plan_id
            || !physical.binding().identifier.is_wire(identifier)
        {
            return Err(CausalError::Compile {
                message: "temporal class mediation plan record differs from its retained proof"
                    .into(),
            });
        }
        Ok(())
    }

    pub(crate) fn estimator(&self) -> EstimatorId {
        EstimatorId::temporal_mediation_for(&self.inference)
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
                Arc::from(accepted_graph_signature(graph))
            }
            CheckedTemporalMediationProof::ClassPosterior { posterior, .. }
            | CheckedTemporalMediationProof::DbnPosterior { posterior, .. } => Arc::from(format!(
                "{}:{}:{}:{}:{}:{}",
                self.proof.name(),
                join_bits(&posterior.weights),
                join_masks(&posterior.atoms),
                join_u64(&posterior.graph_keys),
                posterior.lag_masks.as_deref().map_or_else(|| "none".to_string(), join_u64),
                posterior.max_lag.map_or_else(|| "none".to_string(), |lag| lag.to_string()),
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
                same_accepted_graph(graph, &study.graph)
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
        if data.schema() != &self.schema
            || CausalQuery::Mediation(self.query.clone()) != self.physical.logical.query
            || self.target_bindings.as_ref()
                != retained_mediation_targets(
                    &self.proof,
                    &self.query,
                    &self.schema,
                    &self.inference,
                )?
                .as_slice()
        {
            return Err(CausalError::Compile {
                message:
                    "temporal class mediation refresh changed its schema or theorem target bindings"
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
        click.clear_identification();
        match &self.proof {
            CheckedTemporalMediationProof::ClassEnvelope {
                graph,
                cache,
                max_completions,
                class_prior,
            } => {
                click.graph = Arc::new(graph.clone());
                click.graph_posterior = None;
                click.identifier = Some(IdentifierId::GeneralizedAdjustment);
                click.max_completions = *max_completions;
                click.class_prior.clone_from(class_prior);
                click.install_identification(crate::analysis::IdentificationCache::TemporalClass(Arc::clone(cache)));
                click.execute_temporal_cpdag_mediation(series, &self.query, &self.physical, ctx)
            }
            CheckedTemporalMediationProof::ClassPosterior { posterior, cache, max_completions } => {
                click.graph_posterior = Some(posterior.clone());
                click.max_completions = *max_completions;
                click.install_identification(crate::analysis::IdentificationCache::TemporalClassPosterior(Arc::clone(cache)));
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
                click.install_identification(crate::analysis::IdentificationCache::DbnPosterior(Arc::clone(cache)));
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

fn same_posterior(retained: &GraphPosterior, current: &GraphPosterior) -> bool {
    retained.n_vars == current.n_vars
        && retained.n_graphs == current.n_graphs
        && retained.atom_kind == current.atom_kind
        && retained.max_lag == current.max_lag
        && retained.atoms.as_ref() == current.atoms.as_ref()
        && retained.graph_keys.as_ref() == current.graph_keys.as_ref()
        && retained.lag_masks.as_deref() == current.lag_masks.as_deref()
        && retained.weights.len() == current.weights.len()
        && retained
            .weights
            .iter()
            .zip(current.weights.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Adjacency sets as comma-separated words, each set's words colon-separated with trailing zero
/// words dropped, so a single-word set reads as the integer it always did.
fn join_masks(masks: &[antecedent_discovery::AdjacencyMask]) -> String {
    let words = |mask: &antecedent_discovery::AdjacencyMask| {
        let words = mask.words();
        let used = words.iter().rposition(|&word| word != 0).map_or(1, |last| last + 1);
        words[..used].iter().map(u64::to_string).collect::<Vec<_>>().join(":")
    };
    masks.iter().map(words).collect::<Vec<_>>().join(",")
}

fn join_u64(values: &[u64]) -> String {
    let mut out = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out
}

fn join_bits(values: &[f64]) -> String {
    let mut out = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&value.to_bits().to_string());
    }
    out
}

fn accepted_graph_signature(graph: &crate::AcceptedGraph) -> String {
    let mut out = format!("class={};version={}", graph.class().as_str(), graph.version());
    if let Some(names) = graph.variable_names() {
        out.push_str(";names=");
        for (index, name) in names.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push_str(name);
        }
    }
    if let Some(dag) = graph.as_temporal_dag() {
        out.push(';');
        out.push_str(&crate::analysis::checked_temporal_effect::temporal_dag_signature(dag));
    } else if let Some(cpdag) = graph.as_temporal_cpdag() {
        out.push(';');
        out.push_str(&marked_graph_signature(cpdag.nodes(), &cpdag.edges()));
    }
    out
}

fn marked_graph_signature(
    nodes: &[antecedent_graph::NodeRef],
    edges: &[antecedent_graph::MarkedEdge],
) -> String {
    let mut listed = edges.to_vec();
    listed.sort_unstable_by_key(|edge| {
        (
            edge.a.raw(),
            edge.b.raw(),
            endpoint_tag(edge.at_a),
            endpoint_tag(edge.at_b),
            middle_tag(edge.middle),
        )
    });
    let mut out = String::from("nodes=");
    for (index, node) in nodes.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        match node {
            antecedent_graph::NodeRef::Lagged { variable, lag } => {
                out.push('l');
                out.push_str(&variable.raw().to_string());
                out.push('@');
                out.push_str(&lag.raw().to_string());
            }
            antecedent_graph::NodeRef::Context { variable, environment } => {
                out.push('c');
                out.push_str(
                    &environment.map_or(u32::MAX, antecedent_core::EnvironmentId::raw).to_string(),
                );
                out.push(':');
                out.push_str(&variable.raw().to_string());
            }
            antecedent_graph::NodeRef::Static(variable) => {
                out.push('s');
                out.push_str(&variable.raw().to_string());
            }
            antecedent_graph::NodeRef::Unfolded { variable, offset } => {
                out.push('u');
                out.push_str(&variable.raw().to_string());
                out.push('@');
                out.push_str(&offset.to_string());
            }
        }
    }
    out.push_str(";edges=");
    for (index, edge) in listed.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&edge.a.raw().to_string());
        out.push(endpoint_tag(edge.at_a));
        out.push(middle_tag(edge.middle));
        out.push(endpoint_tag(edge.at_b));
        out.push_str(&edge.b.raw().to_string());
    }
    out
}

fn endpoint_tag(endpoint: antecedent_graph::Endpoint) -> char {
    match endpoint {
        antecedent_graph::Endpoint::Tail => 't',
        antecedent_graph::Endpoint::Arrow => 'a',
        antecedent_graph::Endpoint::Circle => 'o',
        antecedent_graph::Endpoint::Conflict => 'x',
    }
}

fn middle_tag(middle: antecedent_graph::MiddleMark) -> char {
    match middle {
        antecedent_graph::MiddleMark::Unknown => '?',
        antecedent_graph::MiddleMark::Left => 'L',
        antecedent_graph::MiddleMark::Right => 'R',
        antecedent_graph::MiddleMark::Both => '!',
        antecedent_graph::MiddleMark::Empty => '-',
    }
}

/// Bind the requested mediation theorem once, then execute from that same retained cache.
fn retain_class_mediation_proofs(
    cache: &CachedTemporalClassPosteriorIdentification,
    query: &antecedent_core::MediationQuery,
) -> Result<CachedTemporalClassPosteriorIdentification, CausalError> {
    let [horizon] = query.horizons.as_ref() else {
        return Err(CausalError::Compile {
            message: "class posterior requires one mediation horizon".into(),
        });
    };
    let mut atoms = Vec::with_capacity(cache.class_atoms.len());
    for original in cache.class_atoms.iter() {
        let mut atom = original.clone();
        let mut envelope = (*original.envelope).clone();
        crate::identify_api::refine_temporal_class_identification(
            &mut envelope,
            &CausalQuery::Mediation(query.clone()),
            *horizon,
        )?;
        atom.identification = super::envelope_to_identification_result_for(
            &envelope.envelope,
            CausalQuery::Mediation(query.clone()),
        );
        atom.invariant.clone_from(&envelope.envelope.invariant);
        atom.identified_weight = envelope.envelope.identified_weight.0;
        atom.truncated_completions = envelope.envelope.truncated_completions;
        atom.envelope = Arc::new(envelope);
        // Keep a failed refinement's proof and completion weights for inspection.
        atoms.push(atom);
    }
    let eligible = atoms
        .iter()
        .filter(|atom| atom.identified_weight > 0.0)
        .map(|atom| atom.key)
        .collect::<std::collections::BTreeSet<_>>();
    let flags = cache
        .graphs
        .graph_keys
        .iter()
        .map(|key| {
            if eligible.contains(key) {
                antecedent_prob::GraphIdentFlag::Identified
            } else {
                antecedent_prob::GraphIdentFlag::Unidentified
            }
        })
        .collect::<Vec<_>>();
    let graphs = antecedent_prob::WeightedGraphSamples::new(
        Arc::clone(&cache.graphs.weights),
        flags,
        Arc::clone(&cache.graphs.graph_keys),
    )
    .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedTemporalClassPosteriorIdentification {
        graphs,
        class_atoms: atoms.into(),
        first_atom_class: None,
    })
}

fn retained_mediation_targets(
    proof: &CheckedTemporalMediationProof,
    query: &antecedent_core::MediationQuery,
    schema: &antecedent_core::CausalSchema,
    inference: &InferenceMode,
) -> Result<Vec<CheckedTemporalMediationTargetInfo>, CausalError> {
    let selection = match inference {
        InferenceMode::Frequentist => antecedent_core::InferenceSelection::Frequentist,
        InferenceMode::Bayesian(_) => antecedent_core::InferenceSelection::Bayesian,
    };
    let mut targets = Vec::new();
    match proof {
        CheckedTemporalMediationProof::ClassEnvelope { cache, .. } => {
            for &horizon in query.horizons.iter() {
                let envelope = cache
                    .by_horizon
                    .iter()
                    .find(|(h, _)| *h == horizon)
                    .map(|(_, envelope)| envelope)
                    .ok_or_else(|| CausalError::Compile {
                        message: format!("temporal mediation target missing horizon {horizon}"),
                    })?;
                append_completion_targets(
                    &mut targets,
                    envelope,
                    query,
                    schema,
                    selection,
                    horizon,
                    None,
                )?;
            }
        }
        CheckedTemporalMediationProof::ClassPosterior { cache, .. } => {
            let [horizon] = query.horizons.as_ref() else {
                return Err(CausalError::Compile {
                    message: "class-posterior mediation target requires its one retained horizon"
                        .into(),
                });
            };
            let atoms = cache
                .class_atoms
                .iter()
                .map(|atom| (atom.key, atom))
                .collect::<std::collections::BTreeMap<_, _>>();
            for index in 0..cache.graphs.n_samples {
                let (key, weight, identified) = posterior_slot(&cache.graphs, index)?;
                if let Some(atom) = atoms.get(&key) {
                    append_completion_targets(
                        &mut targets,
                        &atom.envelope,
                        query,
                        schema,
                        selection,
                        *horizon,
                        Some((key, weight, identified)),
                    )?;
                } else {
                    targets.push(unresolved_target(*horizon, key, weight, identified)?);
                }
            }
        }
        CheckedTemporalMediationProof::DbnPosterior { cache, .. } => {
            for &horizon in query.horizons.iter() {
                // The existing projection owns per-horizon eligibility and preserves failed mass.
                let projected = cache.mediation_horizon(horizon)?;
                let atoms = projected
                    .atoms
                    .iter()
                    .map(|atom| (atom.key, atom))
                    .collect::<std::collections::BTreeMap<_, _>>();
                for index in 0..projected.graphs.n_samples {
                    let (key, weight, identified) = posterior_slot(&projected.graphs, index)?;
                    if let Some(atom) = atoms.get(&key) {
                        if !mediation_query_matches(&atom.identification.query, query, horizon)
                            || atom.indexer.horizon() != horizon
                        {
                            return Err(CausalError::Compile {
                                message: "DBN mediation target belongs to another query or horizon"
                                    .into(),
                            });
                        }
                        let target = super::checked_static_mediation::theorem_mediation_target(
                            query,
                            schema,
                            &atom.identification.required_assumptions,
                            selection,
                            Some((horizon, atom.indexer.history())),
                        )?;
                        targets.push(CheckedTemporalMediationTargetInfo {
                            horizon,
                            atom_key: Some(key),
                            completion_index: None,
                            atom_weight: weight,
                            completion_weight: None,
                            atom_identified: Some(identified),
                            identification_status: Some(atom.identification.status),
                            scientific_target: Some(target),
                        });
                    } else {
                        targets.push(unresolved_target(horizon, key, weight, identified)?);
                    }
                }
            }
        }
    }
    Ok(targets)
}

fn mediation_query_matches(
    actual: &CausalQuery,
    requested: &antecedent_core::MediationQuery,
    horizon: u32,
) -> bool {
    let CausalQuery::Mediation(actual) = actual else { return false };
    if !actual.horizons.contains(&horizon) {
        return false;
    }
    let mut actual = actual.clone();
    let mut requested = requested.clone();
    actual.horizons = Arc::from([horizon]);
    requested.horizons = Arc::from([horizon]);
    actual == requested
}

fn posterior_slot(
    graphs: &antecedent_prob::WeightedGraphSamples,
    index: usize,
) -> Result<(u64, f64, bool), CausalError> {
    let missing = || CausalError::Compile {
        message: "temporal mediation posterior target arrays are misaligned".into(),
    };
    let key = *graphs.graph_keys.get(index).ok_or_else(missing)?;
    let weight = *graphs.weights.get(index).ok_or_else(missing)?;
    let flag = *graphs.identified.get(index).ok_or_else(missing)?;
    Ok((key, weight, flag == antecedent_prob::GraphIdentFlag::Identified))
}

fn unresolved_target(
    horizon: u32,
    key: u64,
    weight: f64,
    identified: bool,
) -> Result<CheckedTemporalMediationTargetInfo, CausalError> {
    if identified {
        return Err(CausalError::Compile {
            message: "identified temporal mediation atom has no retained theorem proof".into(),
        });
    }
    Ok(CheckedTemporalMediationTargetInfo {
        horizon,
        atom_key: Some(key),
        completion_index: None,
        atom_weight: weight,
        completion_weight: None,
        atom_identified: Some(false),
        identification_status: None,
        scientific_target: None,
    })
}

fn append_completion_targets(
    targets: &mut Vec<CheckedTemporalMediationTargetInfo>,
    envelope: &antecedent_identify::TemporalClassEnvelope,
    query: &antecedent_core::MediationQuery,
    schema: &antecedent_core::CausalSchema,
    inference: antecedent_core::InferenceSelection,
    horizon: u32,
    atom: Option<(u64, f64, bool)>,
) -> Result<(), CausalError> {
    if envelope.envelope.cases.is_empty()
        || envelope.envelope.cases.len() != envelope.indexers.len()
    {
        return Err(CausalError::Compile {
            message: "temporal mediation completion targets lack aligned proof indexers".into(),
        });
    }
    for (index, (case, indexer)) in
        envelope.envelope.cases.iter().zip(&envelope.indexers).enumerate()
    {
        let is_mediation = mediation_query_matches(&case.result.query, query, horizon)
            && indexer.horizon() == horizon;
        if !is_mediation && antecedent_identify::carries_identified_mass(case.result.status) {
            return Err(CausalError::Compile {
                message: "temporal mediation target cannot inherit another functional's proof"
                    .into(),
            });
        }
        let target = if is_mediation {
            Some(super::checked_static_mediation::theorem_mediation_target(
                query,
                schema,
                &case.result.required_assumptions,
                inference,
                Some((horizon, indexer.history())),
            )?)
        } else {
            None
        };
        targets.push(CheckedTemporalMediationTargetInfo {
            horizon,
            atom_key: atom.map(|(key, _, _)| key),
            completion_index: Some(index),
            atom_weight: atom.map_or(1.0, |(_, weight, _)| weight),
            completion_weight: Some(case.weight.0),
            atom_identified: atom.map(|(_, _, identified)| identified),
            identification_status: is_mediation.then_some(case.result.status),
            scientific_target: target,
        });
    }
    Ok(())
}

/// One theorem-scoped target retained for a temporal mediation proof slot.
///
/// Posterior atom weights and completion enumeration weights remain separate. A slot
/// without a retained proof keeps its original atom weight but has no fabricated target,
/// history bound or scientific identification verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedTemporalMediationTargetInfo {
    /// Requested outcome horizon.
    pub horizon: u32,
    /// Position-derived key in the frozen posterior, absent for a fixed class.
    pub atom_key: Option<u64>,
    /// Completion position within a class envelope, absent for a DAG or missing proof.
    pub completion_index: Option<usize>,
    /// Original posterior atom weight; one for a fixed source class.
    pub atom_weight: f64,
    /// Original completion enumeration weight, never a posterior probability.
    pub completion_weight: Option<f64>,
    /// Per-horizon atom eligibility, absent for a fixed class.
    pub atom_identified: Option<bool>,
    /// Actual retained proof status; absent when no proof was retained.
    pub identification_status: Option<antecedent_core::IdentificationStatus>,
    /// Exact conditional target using this proof's history and assumptions.
    pub scientific_target: Option<antecedent_core::ScientificTarget>,
}

/// Read-only target, structural proof, and procedure for a checked
/// `TemporalCpdag` class-envelope or temporal graph-posterior mediation family.
#[derive(Clone, Debug)]
pub struct CheckedTemporalClassMediationInfo {
    /// Exact mediation target and requested horizon family.
    pub query: antecedent_core::MediationQuery,
    /// Per-horizon, per-atom and per-completion theorem targets, including unresolved slots.
    pub scientific_targets: Arc<[CheckedTemporalMediationTargetInfo]>,
    /// Class of the fixed source graph or of the posterior's atoms.
    pub graph_class: crate::accepted::GraphClass,
    /// How the structure was supplied: explicit, accepted, or a graph posterior.
    pub structure_source: crate::support::StructureSource,
    /// Identifier recorded by the checked physical plan.
    pub identifier: std::sync::Arc<str>,
    /// Estimator family fixed at preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite selected at preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Circular-block bootstrap budget fixed at preparation.
    pub bootstrap_replicates: u32,
    /// Horizon members retained in order.
    pub horizons: std::sync::Arc<[u32]>,
    /// Completion atoms (fixed class) or posterior atoms (graph posterior)
    /// retained by the proof.
    pub atom_count: usize,
    /// Enumeration or posterior mass that identified.
    pub identified_mass: f64,
    /// Enumeration or posterior mass that stays unidentified.
    pub unidentified_mass: f64,
    /// Signature of the retained source graph or frozen posterior.
    pub proof_signature: std::sync::Arc<str>,
}

impl Inspect for CheckedTemporalClassMediationOperation {
    type Info = CheckedTemporalClassMediationInfo;

    fn info(&self) -> CheckedTemporalClassMediationInfo {
        let (atom_count, identified_mass, unidentified_mass) = self.proof_mass();
        CheckedTemporalClassMediationInfo {
            query: self.query().clone(),
            scientific_targets: Arc::clone(&self.target_bindings),
            graph_class: self.graph_class(),
            structure_source: self.structure_source(),
            identifier: self.identifier(),
            estimator: self.estimator(),
            validation: self.validation(),
            bootstrap_replicates: self.bootstrap_replicates(),
            horizons: Arc::clone(&self.query().horizons),
            atom_count,
            identified_mass,
            unidentified_mass,
            proof_signature: self.proof_signature(),
        }
    }
}
