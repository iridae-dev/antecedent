use super::{
    Admg, Arc, BTreeSet, CatalogTransportResult, CausalExprArena, ClassicalTransportQuery,
    EvidenceCatalog, EvidenceCatalogDelta, ExecutionContext, ExprId, ExprNode, IdentificationError,
    MetaSource, MetaTransportQuery, MixedSourceDecision, MixedSourceQuery, MzTransportDecision,
    MzTransportQuery, MzTransportRoute, RegimeId, SearchReceipt, SearchStop, SelectionDiagram,
    SharedSearch, StudyFactor, StudyPlanCandidate, StudyPlanRoute, StudyProposal,
    StudyProposalDerivation, StudyRepair, VariableId, ZTransportDecision, ZTransportQuery,
    bind_mixed_source_catalog, bind_mz_transport_catalog, bind_z_transport_catalog,
    decide_mixed_source_shared, decide_mz_transport_shared, decide_z_transport_reporting,
    identify_catalog_transport_metered, identify_meta_catalog_metered, reason_code,
    recheck_mz_citations,
};

/// A re-run decision on a preview catalog that identified and re-checked.
pub(super) struct Found {
    stage: String,
    arena: CausalExprArena,
    root: ExprId,
    pub(super) cited: Vec<RegimeId>,
    /// `(proof step, regime)` of each mixed-source input step.
    steps: Vec<(usize, RegimeId)>,
    derivation: StudyProposalDerivation,
    pub(super) operations: usize,
}

pub(super) enum Preview {
    Identified(Box<Found>),
    NotIdentified {
        code: &'static str,
        detail: &'static str,
    },
    /// The decision was cut off at its per-decision cap after charging `operations`.
    OverCap {
        operations: usize,
    },
    Exhausted(SearchReceipt),
}

/// Re-run the route's decision on a preview catalog and re-check a derivation
/// against it. The decision itself charges at most `decision_cap` operations: the charge that
/// crosses the cap ends it as [`Preview::OverCap`], so one subset cannot spend the plan's
/// budget on a decision that could never be sufficient. A derivation is re-checked only for a
/// decision that finished within its cap, so the unmetered re-check is bounded by the cap.
pub(super) fn decide_preview(
    graph: &Admg,
    route: &StudyPlanRoute,
    preview: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
    decision_cap: usize,
) -> Result<Preview, IdentificationError> {
    let before = search.operations();
    search.cap_decision(decision_cap);
    match route {
        StudyPlanRoute::Mz(query) => {
            let decision = decide_mz_transport_shared(graph, query, preview, search, ctx);
            let operations = search.operations() - before;
            if search.release_decision() {
                return Ok(Preview::OverCap { operations });
            }
            let decision = decision?;
            match decision {
                MzTransportDecision::Identified { derivation, cited } => {
                    let bound = bind_mz_transport_catalog(graph, &derivation, preview)?;
                    recheck_mz_citations(bound.cited_regimes(), &cited)?;
                    let stage = match derivation.route() {
                        MzTransportRoute::TargetOnly => "mz_transport:target_only".to_owned(),
                        MzTransportRoute::SingleSource { population } => {
                            format!("mz_transport:single_source:{population}")
                        }
                        MzTransportRoute::Combined { .. } => "mz_transport:combined".to_owned(),
                    };
                    Ok(Preview::Identified(Box::new(Found {
                        stage,
                        arena: bound.arena().clone(),
                        root: bound.root(),
                        cited: cited.to_vec(),
                        steps: Vec::new(),
                        derivation: StudyProposalDerivation::Mz(Box::new(derivation.to_record())),
                        operations,
                    })))
                }
                MzTransportDecision::Exhausted(receipt) => Ok(Preview::Exhausted(receipt)),
                other => Ok(Preview::NotIdentified {
                    code: other.reason_code().unwrap_or_default(),
                    detail: other.detail_code().unwrap_or_default(),
                }),
            }
        }
        StudyPlanRoute::Mixed(query) => {
            let decision = decide_mixed_source_shared(graph, query, preview, search, ctx);
            let operations = search.operations() - before;
            if search.release_decision() {
                return Ok(Preview::OverCap { operations });
            }
            let decision = decision?;
            match decision {
                MixedSourceDecision::Identified { derivation, cited, .. } => {
                    // Binding replays every step through the independent checker.
                    let bound = bind_mixed_source_catalog(graph, &derivation, preview)?;
                    Ok(Preview::Identified(Box::new(Found {
                        stage: "mixed_source:rule_search".to_owned(),
                        arena: bound.arena().clone(),
                        root: bound.root(),
                        cited: cited.to_vec(),
                        steps: derivation.leaves().iter().map(|(i, l)| (*i, l.regime)).collect(),
                        derivation: StudyProposalDerivation::Mixed(Box::new(
                            derivation.to_record(),
                        )),
                        operations,
                    })))
                }
                MixedSourceDecision::NamedRoute { route: named, .. } => {
                    match rerun_named(graph, query, named, preview, search, ctx)? {
                        Some((arena, root)) => {
                            let cited = leaf_regimes(&arena, root);
                            Ok(Preview::Identified(Box::new(Found {
                                stage: format!("mixed_source:named:{named}"),
                                arena,
                                root,
                                cited,
                                steps: Vec::new(),
                                derivation: StudyProposalDerivation::Named { route: named },
                                operations,
                            })))
                        }
                        None => Ok(Preview::NotIdentified {
                            code: reason_code!("transport_not_certified"),
                            detail:
                                antecedent_core::reason_details::mixed_search::INVALID_DERIVATION,
                        }),
                    }
                }
                MixedSourceDecision::Exhausted(receipt) => Ok(Preview::Exhausted(receipt)),
                other => Ok(Preview::NotIdentified {
                    code: other.reason_code().unwrap_or_default(),
                    detail: other.detail_code().unwrap_or_default(),
                }),
            }
        }
    }
}

/// Re-run the mixed-source named route that identified the preview on the shared budget
/// and bind its derivation to the preview (`NamedRoute` carries no derivation).
/// `None` when the re-run does not identify.
fn rerun_named(
    graph: &Admg,
    query: &MixedSourceQuery,
    named: &'static str,
    preview: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<Option<(CausalExprArena, ExprId)>, IdentificationError> {
    // `query` is the plan's canonical route (sources by population), the form
    // the mixed decision ran its named stage on.
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
    let bound = |result: CatalogTransportResult| match result {
        CatalogTransportResult::Identified(bound) => Some((bound.arena().clone(), bound.root())),
        _ => None,
    };
    Ok(match named {
        "target_first_sid" => {
            let classical = ClassicalTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                source: Arc::from("<no source>"),
                target: Arc::clone(&query.target),
            };
            bound(identify_catalog_transport_metered(
                &shared,
                &classical,
                preview,
                search.meter(),
                ctx,
            )?)
        }
        "z_transport" => {
            let Some(source) = query.sources.first() else { return Ok(None) };
            let diagram =
                SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                    .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
            let single = ZTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                controllable: Arc::clone(&source.controllable),
                experiment_assignment: Arc::clone(&source.experiment_assignment),
                source: Arc::clone(&source.population),
                target: Arc::clone(&query.target),
            };
            match decide_z_transport_reporting(
                &diagram,
                &single,
                preview,
                search.meter(),
                ctx,
                &mut None,
            )? {
                ZTransportDecision::Identified(z) => {
                    let bound = bind_z_transport_catalog(&diagram, &single, &z, preview)?;
                    Some((bound.arena().clone(), bound.root()))
                }
                _ => None,
            }
        }
        "mz_transport" => {
            let mz = MzTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                target: Arc::clone(&query.target),
                sources: Arc::clone(&query.sources),
            };
            match decide_mz_transport_shared(graph, &mz, preview, search, ctx)? {
                MzTransportDecision::Identified { derivation, .. } => {
                    let bound = bind_mz_transport_catalog(graph, &derivation, preview)?;
                    Some((bound.arena().clone(), bound.root()))
                }
                MzTransportDecision::Exhausted(receipt) => {
                    return Err(stop_error(receipt.stop));
                }
                _ => None,
            }
        }
        "meta_transport" => {
            let meta = MetaTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                target: Arc::clone(&query.target),
                sources: query
                    .sources
                    .iter()
                    .map(|source| MetaSource {
                        population: source.population.to_string(),
                        selections: source.selection_targets.iter().map(|v| v.raw()).collect(),
                    })
                    .collect(),
            };
            bound(identify_meta_catalog_metered(graph, &meta, preview, search.meter(), ctx)?)
        }
        _ => None,
    })
}

const fn stop_error(stop: SearchStop) -> IdentificationError {
    match stop {
        SearchStop::Cancelled => IdentificationError::Cancelled,
        SearchStop::Memory => IdentificationError::budget(crate::IdentificationBudget::Memory),
        SearchStop::Operations | SearchStop::Depth => {
            IdentificationError::budget(crate::IdentificationBudget::Steps)
        }
    }
}

/// One distribution leaf of a formula: its regime when the binding tagged it
/// (a leaf bound to several per-level regimes stays untagged), population and
/// intervention set.
struct BoundLeaf {
    regime: Option<RegimeId>,
    population: Arc<str>,
    intervened: Vec<VariableId>,
    variables: Vec<VariableId>,
    conditioned: Vec<VariableId>,
}

impl BoundLeaf {
    /// Whether this leaf reads `regime`: tagged with it, or untagged with the
    /// regime's population and intervention set while the formula cites it.
    fn reads(&self, regime: &antecedent_core::EvidenceRegime, cited: &[RegimeId]) -> bool {
        if let Some(id) = self.regime {
            return id == regime.id;
        }
        let mut intervened = regime.interventions.to_vec();
        intervened.sort_unstable();
        cited.contains(&regime.id)
            && self.population == regime.population
            && self.intervened == intervened
    }
}

/// Every distribution leaf reachable from `root`.
fn bound_leaves(arena: &CausalExprArena, root: ExprId) -> Vec<BoundLeaf> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut pending = vec![root];
    while let Some(id) = pending.pop() {
        if !seen.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                population,
                regime,
                ..
            } => {
                let sorted = |v: &[VariableId]| {
                    let mut v = v.to_vec();
                    v.sort_unstable();
                    v
                };
                out.push(BoundLeaf {
                    regime: *regime,
                    population: Arc::from(arena.population(*population)),
                    intervened: sorted(
                        &arena
                            .intervention_assignments(*intervention)
                            .iter()
                            .map(|a| a.variable)
                            .collect::<Vec<_>>(),
                    ),
                    variables: sorted(arena.var_set(*variables)),
                    conditioned: sorted(arena.var_set(*conditioned_on)),
                });
            }
            ExprNode::Kernel { body, .. } => pending.push(*body),
            ExprNode::Product(list) => pending.extend(arena.list(*list).iter().copied()),
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                pending.push(*expr);
            }
            ExprNode::Ratio { numerator, denominator } => {
                pending.extend([*numerator, *denominator]);
            }
            ExprNode::Expectation { distribution, .. } => pending.push(*distribution),
            ExprNode::Contrast { left, right, .. } => pending.extend([*left, *right]),
        }
    }
    out
}

fn leaf_regimes(arena: &CausalExprArena, root: ExprId) -> Vec<RegimeId> {
    let mut cited =
        bound_leaves(arena, root).into_iter().filter_map(|leaf| leaf.regime).collect::<Vec<_>>();
    cited.sort_unstable();
    cited.dedup();
    cited
}

/// Package a sufficient subset: every cited proposed regime with the factors
/// that read it and the margin the proof actually needs from it.
pub(super) fn proposal(
    candidates: &[StudyPlanCandidate],
    members: &[usize],
    delta: &EvidenceCatalogDelta,
    cost_units: u64,
    sample_budget: u64,
    found: Box<Found>,
) -> StudyProposal {
    let leaves = bound_leaves(&found.arena, found.root);
    let mut cited = found.cited.clone();
    cited.sort_unstable();
    cited.dedup();
    let mut repairs = Vec::new();
    let mut used = BTreeSet::new();
    for regime in delta.proposed_regimes.iter() {
        let cites_leaf = leaves.iter().any(|leaf| leaf.regime == Some(regime.id));
        if !cited.contains(&regime.id) && !cites_leaf {
            continue;
        }
        let Some(&owner) = members
            .iter()
            .find(|&&i| candidates[i].delta.proposed_regimes.iter().any(|r| r.id == regime.id))
        else {
            continue;
        };
        used.insert(owner);
        let mut factors = Vec::new();
        let mut required = BTreeSet::new();
        for leaf in leaves.iter().filter(|leaf| leaf.reads(regime, &cited)) {
            required.extend(leaf.variables.iter().copied());
            required.extend(leaf.conditioned.iter().copied());
            let factor = StudyFactor {
                variables: leaf.variables.clone(),
                conditioned_on: leaf.conditioned.clone(),
            };
            if !factors.contains(&factor) {
                factors.push(factor);
            }
        }
        let proof_steps = found
            .steps
            .iter()
            .filter(|(_, r)| *r == regime.id)
            .map(|(step, _)| *step)
            .collect::<Vec<_>>();
        if factors.is_empty() {
            // A mixed-source input step reads the whole declared joint.
            required.extend(regime.measured.iter().copied());
        }
        let mut intervened = regime.interventions.to_vec();
        intervened.sort_unstable();
        let mut declared = regime.measured.to_vec();
        declared.sort_unstable();
        repairs.push(StudyRepair {
            candidate: Arc::clone(&candidates[owner].id),
            regime: regime.id,
            population: Arc::clone(&regime.population),
            intervened,
            factors,
            required_margin: required.into_iter().collect(),
            declared_margin: declared,
            proof_steps,
        });
    }
    StudyProposal {
        candidates: members.iter().map(|&i| Arc::clone(&candidates[i].id)).collect(),
        cost_units,
        sample_budget,
        stage: found.stage,
        cited_regimes: cited,
        repairs,
        uncited_candidates: members
            .iter()
            .filter(|i| !used.contains(i))
            .map(|&i| Arc::clone(&candidates[i].id))
            .collect(),
        derivation: found.derivation,
        decision_operations: found.operations,
    }
}
