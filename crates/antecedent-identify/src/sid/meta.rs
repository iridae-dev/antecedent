//! Classical meta-transportability (Bareinboim & Pearl 2013, Figure 5).
use super::{
    Admg, Arc, CatalogTransportResult, ClassicalTransportDerivation, ClassicalTransportQuery,
    ClassicalTransportResult, Engine, ExecutionContext, IdentificationBudget, IdentificationError,
    SHedgeCertificate, SelectionDiagram, SidLimits, VariableId, identify_catalog_transport,
    verify_classical_transport, verify_s_hedge,
};

pub(super) const CLASSICAL_SETTING: &str = "classical_single_source_all_experiments_v1";
pub(super) const META_SETTING: &str = "classical_meta_all_source_experiments_v1";

/// One source's mechanism discrepancies relative to the common target.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetaSource {
    /// Stable population identity, never a positional source index.
    pub population: String,
    /// Selection targets in original variable coordinates, canonically sorted.
    pub selections: Vec<u32>,
}
/// Joint target query under the full experimental family in every source.
#[derive(Clone, Debug)]
pub struct MetaTransportQuery {
    /// Joint outcome coordinates.
    pub outcomes: Arc<[VariableId]>,
    /// Hard intervention coordinates.
    pub treatments: Arc<[VariableId]>,
    /// Target population with observational information.
    pub target: Arc<str>,
    /// Source-specific selection diagrams on the shared causal graph.
    pub sources: Vec<MetaSource>,
}
/// Checked representation shared by classical and meta-transport derivations.
/// The scientific scope is explicit in `evidence_setting` and `sources`.
pub type CheckedTransportDerivation = ClassicalTransportDerivation;

impl ClassicalTransportDerivation {
    /// Immutable theorem evidence setting (separate from catalog availability).
    #[must_use]
    pub fn evidence_setting(&self) -> &'static str {
        if self.sources.is_empty() { CLASSICAL_SETTING } else { META_SETTING }
    }
    /// All source-selection dependencies of a meta proof; empty for historical sID.
    #[must_use]
    pub fn sources(&self) -> &[MetaSource] {
        &self.sources
    }
    /// The theorem scope of this checked proof.
    #[must_use]
    pub fn theorem_scope(&self) -> antecedent_core::TheoremScope {
        if self.sources.is_empty() {
            antecedent_core::TheoremScope::classical_sid_complete()
        } else {
            antecedent_core::TheoremScope::meta_sid_complete()
        }
    }
    /// Bind after conservative operation/memory preflight, checking cancellation before publication.
    /// # Errors
    /// Invalid evidence, missing factors or exhausted resource limits.
    pub fn bind_catalog_with_context(
        &self,
        catalog: &antecedent_core::EvidenceCatalog,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<super::BoundTransportFunctional, IdentificationError> {
        self.check_catalog_binding_resources(catalog, limits, ctx)?;
        let bound = self.bind_catalog(catalog)?;
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        Ok(bound)
    }

    fn check_catalog_binding_resources(
        &self,
        catalog: &antecedent_core::EvidenceCatalog,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<(), IdentificationError> {
        let width = catalog
            .regimes
            .iter()
            .try_fold(1usize, |n, r| {
                n.checked_add(r.measured.len())?
                    .checked_add(r.interventions.len())?
                    .checked_add(r.conditioned_on.len())?
                    .checked_add(1)
            })
            .ok_or(IdentificationError::budget(IdentificationBudget::Binding))?;
        let work = width
            .checked_mul(self.arena.len().max(1))
            .ok_or(IdentificationError::budget(IdentificationBudget::Binding))?;
        let bytes = width
            .checked_mul(128)
            .and_then(|n| n.checked_add(catalog.bindings.len().checked_mul(256)?))
            .ok_or(IdentificationError::budget(IdentificationBudget::BindingMemory))?;
        if work > limits.steps {
            return Err(IdentificationError::budget(IdentificationBudget::Binding));
        }
        super::refuse_over_budget(bytes, ctx, IdentificationBudget::BindingMemory)
    }

    /// Search finite-catalog alternatives under this proof's original scientific inputs.
    /// # Errors
    /// Invalid source contracts or exhausted computation resources.
    pub fn search_catalog(
        &self,
        diagram: &SelectionDiagram,
        catalog: &antecedent_core::EvidenceCatalog,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<CatalogTransportResult, IdentificationError> {
        if self.sources.is_empty() {
            identify_catalog_transport(diagram, &self.query, catalog, limits, ctx)
        } else {
            identify_meta_catalog(
                diagram.causal_graph(),
                &MetaTransportQuery {
                    outcomes: self.query.outcomes.clone(),
                    treatments: self.query.treatments.clone(),
                    target: self.query.target.clone(),
                    sources: self.sources.clone(),
                },
                catalog,
                limits,
                ctx,
            )
        }
    }
}
pub(super) fn check_meta_resources(
    graph: &Admg,
    sources: &[MetaSource],
    steps: usize,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    if sources.is_empty() {
        return Ok(());
    }
    let work = sources
        .iter()
        .try_fold(0usize, |n, s| n.checked_add(s.selections.len().checked_add(1)?))
        .ok_or(IdentificationError::budget(IdentificationBudget::Steps))?;
    let bytes = sources
        .iter()
        .try_fold(0usize, |n, s| {
            n.checked_add(s.population.len())?
                .checked_add(s.selections.len().checked_mul(4)?)?
                .checked_add(128)
        })
        .and_then(|n| n.checked_add(graph.node_count().checked_mul(64)?))
        .and_then(|n| n.checked_mul(4))
        .ok_or(IdentificationError::budget(IdentificationBudget::Steps))?;
    if work > steps {
        return Err(IdentificationError::budget(IdentificationBudget::Steps));
    }
    super::refuse_over_budget(bytes, ctx, IdentificationBudget::BindingMemory)
}
pub(super) fn validate_meta_sources(
    graph: &Admg,
    query: &ClassicalTransportQuery,
    sources: &[MetaSource],
) -> Result<(), IdentificationError> {
    if sources.is_empty() {
        return Ok(());
    }
    if sources[0].population != query.source.as_ref()
        || sources.windows(2).any(|s| s[0].population >= s[1].population)
    {
        return Err(IdentificationError::invalid_input(
            "transport.invalid_query: invalid canonical meta source identities",
        ));
    }
    for source in sources {
        if source.population.trim().is_empty()
            || source.population == query.target.as_ref()
            || source.selections.windows(2).any(|v| v[0] >= v[1])
        {
            return Err(IdentificationError::invalid_input(
                "transport.invalid_query: invalid meta source/selection contract",
            ));
        }
        SelectionDiagram::try_new(
            graph.clone(),
            source.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}
impl MetaTransportQuery {
    /// Read source selection assumptions from the supplied catalog, without providers.
    /// # Errors
    /// Invalid catalog or missing target/source identities.
    pub fn from_catalog(
        outcomes: Arc<[VariableId]>,
        treatments: Arc<[VariableId]>,
        target: Arc<str>,
        catalog: &antecedent_core::EvidenceCatalog,
    ) -> Result<Self, IdentificationError> {
        catalog.validate().map_err(|e| IdentificationError::invalid_catalog(e.to_string()))?;
        let target_environment =
            catalog.environments.iter().find(|e| e.identity == target).ok_or_else(|| {
                IdentificationError::invalid_catalog(
                    "transport.invalid_input: missing target environment",
                )
            })?;
        if !target_environment.selection_targets.is_empty() {
            return Err(IdentificationError::invalid_catalog(
                "transport.invalid_input: target environment cannot declare selection differences from itself",
            ));
        }
        let sources = catalog
            .environments
            .iter()
            .filter(|e| e.identity != target)
            .map(|e| MetaSource {
                population: e.identity.to_string(),
                selections: e.selection_targets.iter().map(|v| v.raw()).collect(),
            })
            .collect();
        Ok(Self { outcomes, treatments, target, sources })
    }
    fn canonical(
        &self,
        graph: &Admg,
    ) -> Result<(ClassicalTransportQuery, SelectionDiagram, Vec<MetaSource>), IdentificationError>
    {
        let mut sources = self.sources.clone();
        sources.sort_by(|a, b| a.population.cmp(&b.population));
        for source in &mut sources {
            source.selections.sort_unstable();
        }
        let first = sources.first().ok_or_else(|| {
            IdentificationError::invalid_input(
                "transport.invalid_query: meta transport requires a source",
            )
        })?;
        let query = ClassicalTransportQuery {
            outcomes: self.outcomes.clone(),
            treatments: self.treatments.clone(),
            source: Arc::from(first.population.as_str()),
            target: self.target.clone(),
        };
        validate_meta_sources(graph, &query, &sources)?;
        let diagram = SelectionDiagram::try_new(
            graph.clone(),
            first.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        )?;
        Ok((query, diagram, sources))
    }
}
/// Identify with `μsID` Figure 5. Target-only ID has priority; scientific negatives
/// require the same independently checked forest obstruction in every source.
/// # Errors
/// Invalid coordinates, cancellation, resource exhaustion, or invalid proof.
pub fn identify_meta_transport(
    graph: &Admg,
    query: &MetaTransportQuery,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<ClassicalTransportResult, IdentificationError> {
    check_meta_resources(graph, &query.sources, limits.steps, ctx)?;
    let (query, diagram, sources) = query.canonical(graph)?;
    let mut engine = Engine::new(&diagram, &query, limits, ctx)?;
    engine.sources.clone_from(&sources);
    let state = engine.initial()?;
    let mut result = engine.solve(state.clone(), false, 0)?;
    if result.is_none() {
        result = engine.solve(state, true, 0)?;
    }
    if let Some(root_step) = result {
        let proof = engine.solved_derivation(root_step, sources)?;
        return Ok(ClassicalTransportResult::Identified(Box::new(proof)));
    }
    if let Some(state) = engine.obstruction.clone() {
        if let Some(mut witness) = engine.negative_witness(&state)? {
            witness.sources = sources;
            // Only a witness that is not a common s-hedge is inconclusive; a
            // budget, cancellation or graph failure while checking it is an error.
            match verify_s_hedge(&diagram, &query, &witness, ctx) {
                Ok(()) => return Ok(ClassicalTransportResult::ProvenNonTransportable(witness)),
                Err(IdentificationError::InvalidDerivation {
                    code: "transport.invalid_s_hedge",
                }) => {}
                Err(error) => return Err(error),
            }
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    Ok(ClassicalTransportResult::NotCertified)
}
/// Deterministic sound finite-catalog search; no arbitrary-catalog completeness claim.
/// # Errors
/// Invalid declarations or exhausted search resources.
pub fn identify_meta_catalog(
    graph: &Admg,
    query: &MetaTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<CatalogTransportResult, IdentificationError> {
    check_meta_resources(graph, &query.sources, limits.steps, ctx)?;
    let (query, diagram, sources) = query.canonical(graph)?;
    catalog.validate().map_err(|e| IdentificationError::invalid_catalog(e.to_string()))?;
    let mut engine = Engine::new(&diagram, &query, limits, ctx)?;
    engine.sources.clone_from(&sources);
    let state = engine.initial()?;
    let mut obligations = Vec::new();
    let mut had_derivation = false;
    for strategy in 0..3 {
        if strategy == 2 {
            engine.source_catalog = Some(catalog);
            engine.memo.clear();
        }
        if let Some(root_step) = engine.solve(state.clone(), strategy != 0, 0)? {
            had_derivation = true;
            let proof = engine.solved_derivation(root_step, sources.clone())?;
            proof.check_catalog_binding_resources(catalog, limits, ctx)?;
            match proof.bind_catalog_validated(catalog) {
                Ok(mut bound) => {
                    bound.searched =
                        ["target_only_id", "canonical_meta_sid", "catalog_available_meta_sid"]
                            [..=strategy]
                            .iter()
                            .map(|s| Arc::from(*s))
                            .collect();
                    return Ok(CatalogTransportResult::Identified(Box::new(bound)));
                }
                Err(error) => obligations.push(Arc::from(error.to_string())),
            }
        }
    }
    let searched = ["target_only_id", "canonical_meta_sid", "catalog_available_meta_sid"]
        .map(Arc::from)
        .into();
    if had_derivation {
        Ok(CatalogTransportResult::MissingEvidence { searched, obligations: obligations.into() })
    } else {
        Ok(CatalogTransportResult::NotCertified { searched, obligations: obligations.into() })
    }
}

/// Check a meta proof against an independently supplied complete source collection.
/// # Errors
/// Any changed source, query, graph, selection premise, or resource failure.
pub fn verify_meta_transport(
    graph: &Admg,
    query: &MetaTransportQuery,
    proof: &CheckedTransportDerivation,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    check_meta_resources(graph, &query.sources, limits.steps, ctx)?;
    let (query, diagram, sources) = query.canonical(graph)?;
    if proof.sources != sources {
        return Err(IdentificationError::invalid_derivation(
            "transport.invalid_derivation: meta proof source collection mismatch",
        ));
    }
    verify_classical_transport(&diagram, &query, proof, limits, ctx)
}
/// Check the same obstruction forests against every externally declared source.
/// # Errors
/// Altered source collection or invalid witness premises.
pub fn verify_meta_s_hedge(
    graph: &Admg,
    query: &MetaTransportQuery,
    witness: &SHedgeCertificate,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    check_meta_resources(graph, &query.sources, usize::MAX, ctx)?;
    let (query, diagram, sources) = query.canonical(graph)?;
    if witness.sources != sources {
        return Err(IdentificationError::invalid_derivation(
            "transport.invalid_s_hedge: meta witness source collection mismatch",
        ));
    }
    verify_s_hedge(&diagram, &query, witness, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_graph::DenseNodeId;
    fn v(n: u32) -> VariableId {
        VariableId::from_raw(n)
    }
    fn query() -> MetaTransportQuery {
        MetaTransportQuery {
            outcomes: Arc::from([v(2)]),
            treatments: Arc::from([v(0)]),
            target: Arc::from("target"),
            sources: vec![
                MetaSource { population: "a".into(), selections: vec![2] },
                MetaSource { population: "b".into(), selections: vec![1] },
            ],
        }
    }
    fn graph() -> Admg {
        let mut graph = Admg::with_variables(3);
        for (a, b) in [(0, 1), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 1), (0, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        graph
    }
    #[test]
    fn complementary_sources_require_both_and_proof_is_checked() {
        let ctx = ExecutionContext::for_tests(0);
        let q = query();
        let g = graph();
        let ClassicalTransportResult::Identified(proof) =
            identify_meta_transport(&g, &q, SidLimits::default(), &ctx).unwrap()
        else {
            panic!("combined evidence identifies")
        };
        let formula = proof.arena().pretty(proof.root());
        assert!(formula.contains('a') && formula.contains('b'), "{formula}");
        for source in &q.sources {
            let single = MetaTransportQuery { sources: vec![source.clone()], ..q.clone() };
            assert!(matches!(
                identify_meta_transport(&g, &single, SidLimits::default(), &ctx).unwrap(),
                ClassicalTransportResult::ProvenNonTransportable(_)
            ));
        }
        let (classical, diagram, _) = q.canonical(&g).unwrap();
        let record = proof.to_record();
        ClassicalTransportDerivation::from_record_checked(
            record.clone(),
            proof.arena().clone(),
            &diagram,
            &classical,
            SidLimits::default(),
            &ctx,
        )
        .unwrap();
        let mut changed = record;
        changed.sources[1].selections.push(2);
        assert!(
            ClassicalTransportDerivation::from_record_checked(
                changed,
                proof.arena().clone(),
                &diagram,
                &classical,
                SidLimits::default(),
                &ctx
            )
            .is_err()
        );
        let mut permuted = q.clone();
        permuted.sources.reverse();
        let ClassicalTransportResult::Identified(other) =
            identify_meta_transport(&g, &permuted, SidLimits::default(), &ctx).unwrap()
        else {
            panic!()
        };
        assert_eq!(formula, other.arena().pretty(other.root()));
    }
    #[test]
    fn common_negative_witness_cannot_drop_an_admissible_source() {
        let ctx = ExecutionContext::for_tests(0);
        let mut q = query();
        q.sources[1].selections = vec![2];
        let ClassicalTransportResult::ProvenNonTransportable(witness) =
            identify_meta_transport(&graph(), &q, SidLimits::default(), &ctx).unwrap()
        else {
            panic!("common obstruction")
        };
        let (query, diagram, _) = q.canonical(&graph()).unwrap();
        let mut record = witness.to_record();
        record.sources[1].selections.clear();
        assert!(SHedgeCertificate::from_record_checked(record, &diagram, &query, &ctx).is_err());
        // The first source's selections are the diagram's; a record that
        // relabels them describes another diagram and is refused.
        let mut relabelled = witness.to_record();
        relabelled.sources[0].selections = vec![1];
        assert!(matches!(
            SHedgeCertificate::from_record_checked(relabelled, &diagram, &query, &ctx),
            Err(IdentificationError::InvalidDerivation { code: "transport.invalid_s_hedge" })
        ));
        assert!(matches!(
            identify_meta_transport(&graph(), &q, SidLimits { steps: 1, depth: 1 }, &ctx),
            Err(IdentificationError::Budget { .. })
        ));
    }

    #[test]
    fn a_cancelled_witness_check_is_a_cancellation_not_an_inconclusive_result() {
        let mut q = query();
        q.sources[1].selections = vec![2];
        let cancelled = ExecutionContext::for_tests(0);
        cancelled.cancellation.cancel();
        assert!(matches!(
            identify_meta_transport(&graph(), &q, SidLimits::default(), &cancelled),
            Err(IdentificationError::Cancelled)
        ));
    }
}
