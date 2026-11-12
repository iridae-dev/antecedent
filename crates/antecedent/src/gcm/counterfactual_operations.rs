use super::{
    AbductionMissingPolicy, Arc, CausalError, CompiledCausalModel, CompiledMechanismStore,
    ContrastRequest, ContrastSetResult, CounterfactualEngine, CounterfactualQuery, Dag,
    ExecutionContext, FittedGcm, Intervention, IteResult, MechanismFamily, MechanismRegistry,
    MechanismWorkspace, ModelError, NestedCounterfactualQuery, SelectionPolicy, TableView,
    TabularData, Value, VariableId, contrast_outcome_nodes, contrasts_from_abduction,
    fit_gcm_counterfactual, fit_gcm_with, ite_from_abduction, map_mechanism_fit,
    nested_counterfactual_with_exo,
};

/// The mechanism families selected on one table, by graph, kept so that refits on resamples
/// of that table condition on the selection instead of repeating it. A resample repeats rows,
/// and a repeated row on both sides of a validation fold favours the flexible families.
#[derive(Debug, Default)]
pub(crate) struct SelectedFamilies {
    /// Whether the selections are final: a graph met afterwards selects for itself and is
    /// not recorded.
    sealed: std::sync::atomic::AtomicBool,
    by_graph: std::sync::Mutex<SelectionsByGraph>,
}

type SelectionsByGraph =
    std::collections::HashMap<Vec<(u32, u32)>, Arc<[(VariableId, MechanismFamily)]>>;

impl SelectedFamilies {
    /// Stop recording: every later fit either reuses a recorded selection or selects anew.
    pub(crate) fn seal(&self) {
        self.sealed.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Fit `graph` to `data` with the families recorded for the graph, or select them (and
    /// record the selection while unsealed).
    fn fit(&self, graph: &Arc<Dag>, data: &TabularData) -> Result<FittedGcm, CausalError> {
        let key = graph.directed_edge_pairs_sorted();
        let known = self
            .by_graph
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .cloned();
        if let Some(families) = known {
            let constraints = families
                .iter()
                .map(|&(variable, family)| {
                    let only = super::NodeFamilyConstraint {
                        allowed: Some(Arc::from([family])),
                        excluded: Arc::from([]),
                        required: Some(family),
                    };
                    (variable, only)
                })
                .collect();
            let options = super::SelectionOptions { constraints, ..Default::default() };
            return fit_gcm_with(
                (**graph).clone(),
                data,
                &MechanismRegistry::with_heterogeneity_families(),
                &options,
            );
        }
        let fitted = fit_gcm_counterfactual(graph, data)?;
        if !self.sealed.load(std::sync::atomic::Ordering::Acquire) {
            let families = fitted
                .assignments
                .iter()
                .map(|chosen| (chosen.variable, chosen.selected))
                .collect();
            self.by_graph
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(key, families);
        }
        Ok(fitted)
    }
}

/// One intervened variable of a contrast set with the level it takes in each world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TreatmentLevels {
    /// The intervened variable.
    pub variable: VariableId,
    /// Level in the active world.
    pub active: f64,
    /// Level in the control world.
    pub control: f64,
}

/// The worlds of a counterfactual the single-ITE form does not state: several outcomes, several
/// simultaneous interventions or a joint baseline, or nested interventions.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ContrastTarget {
    /// Outcomes read from both worlds, in the query's order.
    pub outcomes: Arc<[VariableId]>,
    /// Interventions of the active world, as the query states them.
    pub active: Arc<[Intervention]>,
    /// Interventions of the control world, as the query states them.
    pub control: Arc<[Intervention]>,
    /// Whether the query admits nested interventions.
    pub allow_nested: bool,
    /// Each intervened variable with its two levels, ordered by variable.
    pub levels: Arc<[TreatmentLevels]>,
}

impl ContrastTarget {
    /// The engine request that replays this target.
    pub(crate) fn request(&self) -> ContrastRequest<'_> {
        ContrastRequest {
            outcomes: &self.outcomes,
            active: &self.active,
            control: &self.control,
            allow_nested: self.allow_nested,
        }
    }

    /// The intervened variables, ordered.
    pub(crate) fn treatments(&self) -> Vec<VariableId> {
        self.levels.iter().map(|level| level.variable).collect()
    }
}

/// The hard levels a world sets, flattened over nested sequences, ordered by variable.
pub(crate) fn world_levels(
    world: &[Intervention],
    allow_nested: bool,
) -> Result<Vec<(VariableId, f64)>, CausalError> {
    fn hard(
        intervention: &Intervention,
        allow_nested: bool,
        out: &mut Vec<(VariableId, f64)>,
    ) -> Result<(), CausalError> {
        match intervention {
            Intervention::Set { variable, value } => {
                let level = value.as_f64().ok_or_else(|| CausalError::Compile {
                    message: "counterfactual intervention value must be f64".into(),
                })?;
                if !level.is_finite() {
                    return Err(CausalError::Unsupported {
                        message: "counterfactual treatment levels must be finite",
                    });
                }
                match out.iter().find(|(seen, _)| seen == variable) {
                    Some((_, previous)) if previous.to_bits() != level.to_bits() => {
                        Err(CausalError::Compile {
                            message: format!(
                                "counterfactual world sets {variable:?} to two different levels"
                            ),
                        })
                    }
                    Some(_) => Ok(()),
                    None => {
                        out.push((*variable, level));
                        Ok(())
                    }
                }
            }
            Intervention::Sequence(sequence) if allow_nested => {
                for step in sequence.steps.iter() {
                    if !step.temporal.is_active_at(0) {
                        return Err(CausalError::Unsupported {
                            message: "a nested counterfactual step is not active at the static time",
                        });
                    }
                    hard(&step.intervention, false, out)?;
                }
                Ok(())
            }
            Intervention::Sequence(_) => Err(CausalError::Unsupported {
                message: "a nested intervention needs allow_nested on the counterfactual query",
            }),
            _ => Err(CausalError::Unsupported {
                message: "Study counterfactual path requires hard Set interventions",
            }),
        }
    }
    let mut out = Vec::new();
    for intervention in world {
        hard(intervention, allow_nested, &mut out)?;
    }
    out.sort_by_key(|(variable, _)| *variable);
    Ok(out)
}

/// Frozen ITE target and mechanism-fitting procedure.
/// The supported high-level counterfactual route uses heterogeneity-capable
/// mechanism selection and shared abducted disturbances for both worlds.
///
/// One outcome under one hard `Set` is the single form; several outcomes, several simultaneous
/// interventions with a joint baseline, or nested interventions freeze a [`ContrastTarget`].
#[derive(Clone, Debug)]
pub struct CheckedCounterfactualOperation {
    graph: Arc<Dag>,
    edges: crate::graph_identity::FrozenEdges,
    query: CounterfactualQuery,
    treatment: VariableId,
    outcome: VariableId,
    active: f64,
    control: f64,
    contrast: Option<Arc<ContrastTarget>>,
    /// Selections the mechanism fit conditions on, when the operation refits resamples.
    selected: Option<Arc<SelectedFamilies>>,
}

impl CheckedCounterfactualOperation {
    /// Condition the mechanism fit on `selected`: a graph it holds is fitted with its
    /// recorded families.
    pub(crate) fn conditional_on(mut self, selected: Arc<SelectedFamilies>) -> Self {
        self.selected = Some(selected);
        self
    }

    fn fit(&self, data: &TabularData) -> Result<FittedGcm, CausalError> {
        match &self.selected {
            Some(selected) => selected.fit(&self.graph, data),
            None => fit_gcm_counterfactual(&self.graph, data),
        }
    }

    /// Check and freeze the counterfactual: the one-outcome, one-hard-set ITE, or a contrast
    /// set over several outcomes, simultaneous interventions or nested interventions.
    pub(crate) fn compile(graph: Dag, query: CounterfactualQuery) -> Result<Self, CausalError> {
        query.validate()?;
        if query.allow_nested
            || query.outcomes.len() != 1
            || query.interventions.len() != 1
            || !query.joint_control.is_empty()
        {
            return Self::compile_contrasts(graph, query);
        }
        let Intervention::Set { variable: treatment, value } = &query.interventions[0] else {
            return Err(CausalError::Unsupported {
                message: "Study counterfactual path requires a hard Set intervention",
            });
        };
        let Intervention::Set { variable: control_variable, value: control_value } = &query.control
        else {
            return Err(CausalError::Unsupported {
                message: "Study counterfactual path requires a hard Set control intervention",
            });
        };
        if treatment != control_variable {
            return Err(CausalError::Compile {
                message: "counterfactual control targets a different treatment".into(),
            });
        }
        let active = value.as_f64().ok_or_else(|| CausalError::Compile {
            message: "counterfactual intervention value must be f64".into(),
        })?;
        let control = control_value.as_f64().ok_or_else(|| CausalError::Compile {
            message: "counterfactual control value must be f64".into(),
        })?;
        if !active.is_finite() || !control.is_finite() {
            return Err(CausalError::Unsupported {
                message: "counterfactual treatment levels must be finite",
            });
        }
        let treatment = *treatment;
        let outcome = query.outcomes[0];
        if usize::try_from(treatment.raw()).map_or(true, |id| id >= graph.node_count())
            || usize::try_from(outcome.raw()).map_or(true, |id| id >= graph.node_count())
        {
            return Err(CausalError::Compile {
                message: "counterfactual target variable is absent from the graph".into(),
            });
        }
        let edges = crate::graph_identity::FrozenEdges::of(&graph);
        Ok(Self {
            graph: Arc::new(graph),
            edges,
            query,
            treatment,
            outcome,
            active,
            control,
            contrast: None,
            selected: None,
        })
    }

    /// Freeze a contrast set: every outcome is read from the same two replays, each world a set
    /// of simultaneous interventions on the same variables.
    fn compile_contrasts(graph: Dag, query: CounterfactualQuery) -> Result<Self, CausalError> {
        let active_world = world_levels(&query.interventions, query.allow_nested)?;
        let control_world = world_levels(&query.control_world(), query.allow_nested)?;
        if active_world.is_empty()
            || active_world
                .iter()
                .map(|(variable, _)| variable)
                .ne(control_world.iter().map(|(variable, _)| variable))
        {
            return Err(CausalError::Compile {
                message: "counterfactual control world must intervene on the same variables as \
                          the active world"
                    .into(),
            });
        }
        let in_graph = |variable: VariableId| {
            usize::try_from(variable.raw()).is_ok_and(|id| id < graph.node_count())
        };
        if !active_world.iter().all(|(variable, _)| in_graph(*variable))
            || !query.outcomes.iter().copied().all(in_graph)
        {
            return Err(CausalError::Compile {
                message: "counterfactual target variable is absent from the graph".into(),
            });
        }
        let levels: Vec<TreatmentLevels> = active_world
            .iter()
            .zip(&control_world)
            .map(|(&(variable, active), &(_, control))| TreatmentLevels {
                variable,
                active,
                control,
            })
            .collect();
        let target = ContrastTarget {
            outcomes: Arc::clone(&query.outcomes),
            active: Arc::clone(&query.interventions),
            control: Arc::from(query.control_world()),
            allow_nested: query.allow_nested,
            levels: Arc::from(levels.clone()),
        };
        let edges = crate::graph_identity::FrozenEdges::of(&graph);
        Ok(Self {
            graph: Arc::new(graph),
            edges,
            treatment: levels[0].variable,
            outcome: query.outcomes[0],
            active: levels[0].active,
            control: levels[0].control,
            query,
            contrast: Some(Arc::new(target)),
            selected: None,
        })
    }

    /// The frozen contrast set, when the query is not the single one-outcome form.
    pub(crate) fn contrast_target(&self) -> Option<&ContrastTarget> {
        self.contrast.as_deref()
    }

    /// The frozen semantic target.
    #[must_use]
    pub const fn query(&self) -> &CounterfactualQuery {
        &self.query
    }

    /// The frozen graph and mechanism assignment scope.
    #[must_use]
    pub fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn matches(&self, graph: &Dag, query: &CounterfactualQuery) -> bool {
        self.query == *query && self.edges.matches(&*self.graph, graph)
    }

    /// Fit mechanisms under the frozen graph, then abduct once for both worlds.
    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<(FittedGcm, IteResult), CausalError> {
        if self.contrast.is_some() {
            return Err(CausalError::Compile {
                message: "a counterfactual contrast set is executed with execute_contrasts".into(),
            });
        }
        let FittedGcm { model, assignments } = self.fit(data)?;
        // The engine shares the model through an `Arc`; once it is dropped the fitted model
        // is moved back out without a copy.
        let model = Arc::new(model);
        let engine = CounterfactualEngine::shared(Arc::clone(&model));
        let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx)?;
        let ite = ite_from_abduction(
            &engine,
            exo,
            self.treatment,
            self.outcome,
            (self.active, self.control),
            ctx,
        )?;
        drop(engine);
        let model = Arc::try_unwrap(model).unwrap_or_else(|shared| (*shared).clone());
        Ok((FittedGcm { model, assignments }, ite))
    }

    /// Fit mechanisms under the frozen graph, then abduct once and read every outcome from the
    /// two worlds of a contrast set.
    pub(crate) fn execute_contrasts(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<(FittedGcm, ContrastSetResult), CausalError> {
        let target = self.contrast.as_deref().ok_or_else(|| CausalError::Compile {
            message: "the single-outcome counterfactual is executed with execute".into(),
        })?;
        let FittedGcm { model, assignments } = self.fit(data)?;
        let nodes = contrast_outcome_nodes(&model, &target.outcomes)?;
        let model = Arc::new(model);
        let engine = CounterfactualEngine::shared(Arc::clone(&model));
        let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx)?;
        let set = contrasts_from_abduction(&engine, exo, &target.request(), &nodes, ctx)?;
        drop(engine);
        let model = Arc::try_unwrap(model).unwrap_or_else(|shared| (*shared).clone());
        Ok((FittedGcm { model, assignments }, set))
    }
}

/// Outcome mechanism the nested natural-direct-effect route fits before
/// abducting and replaying both worlds.
///
/// The choice is what makes the shared abduced exogenous draw either invisible
/// or observable. Under a separable outcome the frozen mediator's disturbance
/// cancels out of the direct effect; under a non-separable outcome it does not.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NestedOutcomeMechanism {
    /// Separable linear-Gaussian additive-noise outcome (the supported
    /// point-only cell). The natural direct effect is `a·(active − control)`,
    /// which does not read the abducted disturbance, so a shared and an
    /// independent exogenous draw yield the identical value.
    LinearGaussian,
    /// Non-separable outcome basis (cross-parent products and per-parent smooth
    /// terms, [`MechanismFamily::LinearSpline`]). The outcome is a non-additive
    /// function of its parents, so a treatment-interacted mediator term carries
    /// the frozen mediator's abducted disturbance into the direct effect. The
    /// disturbance is additive, so abduction still inverts it exactly and the
    /// counterfactual replay round-trips.
    NonSeparableBasis,
}

impl NestedOutcomeMechanism {
    /// Tag a persisted artifact stores for this mechanism family.
    pub(crate) const fn wire_tag(self) -> &'static str {
        match self {
            Self::LinearGaussian => "linear_gaussian",
            Self::NonSeparableBasis => "non_separable_basis",
        }
    }

    /// Inverse of [`Self::wire_tag`]; `None` for a tag no family carries.
    pub(crate) fn from_wire_tag(tag: &str) -> Option<Self> {
        match tag {
            "linear_gaussian" => Some(Self::LinearGaussian),
            "non_separable_basis" => Some(Self::NonSeparableBasis),
            _ => None,
        }
    }
}

/// Complete checked operation for the supported natural direct effect route.
///
/// The operation fixes the graph, cross-world target roles, treatment levels,
/// and shared-exogenous execution procedure at preparation time.
#[derive(Clone, Debug)]
pub struct NestedCounterfactualOperation {
    graph: Arc<Dag>,
    query: NestedCounterfactualQuery,
    mediation_query: antecedent_core::MediationQuery,
    outer_world: Intervention,
    active_inner_world: Intervention,
    control_inner_world: Intervention,
    frozen: Arc<[VariableId]>,
    outcome_mechanism: NestedOutcomeMechanism,
}

impl NestedCounterfactualOperation {
    /// The frozen cross-world query and treatment levels.
    #[must_use]
    pub const fn query(&self) -> &NestedCounterfactualQuery {
        &self.query
    }

    /// The three-node Markovian mediation graph required by this operation.
    #[must_use]
    pub fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn compile(
        graph: Dag,
        query: NestedCounterfactualQuery,
    ) -> Result<Self, CausalError> {
        query.validate()?;
        let observed = graph.directed_edge_pairs_sorted();
        let mut expected = vec![
            (query.treatment.raw(), query.mediator.raw()),
            (query.treatment.raw(), query.outcome.raw()),
            (query.mediator.raw(), query.outcome.raw()),
        ];
        expected.sort_unstable();
        if graph.node_count() != 3 || observed != expected {
            return Err(crate::unsupported_reason!(
                "cross_world_not_identified",
                "query requires exactly X -> M, X -> Y, M -> Y with no other nodes or edges"
            ));
        }
        Ok(Self {
            graph: Arc::new(graph),
            query,
            mediation_query: query.as_mediation_query(),
            outer_world: Intervention::set(query.treatment, Value::f64(query.control_value())),
            active_inner_world: Intervention::set(
                query.treatment,
                Value::f64(query.active_value()),
            ),
            control_inner_world: Intervention::set(
                query.treatment,
                Value::f64(query.control_value()),
            ),
            frozen: Arc::from([query.mediator]),
            outcome_mechanism: NestedOutcomeMechanism::LinearGaussian,
        })
    }

    /// Re-target this operation to fit a non-separable outcome basis instead of
    /// the separable linear-Gaussian outcome.
    ///
    /// The graph, cross-world roles, and treatment levels are unchanged; only
    /// the fitted outcome mechanism differs. This is the variant under which the
    /// shared abduced exogenous draw is observable: the direct effect reads the
    /// frozen mediator's disturbance because the outcome is non-additive in its
    /// parents. The supported linear-Gaussian path (used by the staged Study
    /// route) is untouched.
    #[must_use]
    pub fn with_non_separable_outcome(mut self) -> Self {
        self.outcome_mechanism = NestedOutcomeMechanism::NonSeparableBasis;
        self
    }

    /// The outcome mechanism this operation fits.
    #[must_use]
    pub const fn outcome_mechanism(&self) -> NestedOutcomeMechanism {
        self.outcome_mechanism
    }

    pub(crate) fn matches(&self, graph: &Dag, query: &NestedCounterfactualQuery) -> bool {
        self.query == *query && crate::graph_identity::same_dag_structure(graph, &self.graph)
    }

    pub(crate) fn mediation_query(&self) -> &antecedent_core::MediationQuery {
        &self.mediation_query
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<f64, CausalError> {
        let compiled = CompiledCausalModel::compile_shared(Arc::clone(&self.graph))?;
        let store =
            fit_nested_mechanisms(&compiled, data, self.outcome_mechanism, self.query.outcome)?;
        let engine = CounterfactualEngine::new(compiled.with_mechanisms(store));
        let exo = engine
            .abduct(data, AbductionMissingPolicy::Error, ctx)
            .map_err(CausalError::rejected)?;
        let control_outer = [self.outer_world.clone()];
        let active_inner = [self.active_inner_world.clone()];
        let control_inner = [self.control_inner_world.clone()];
        let mut workspace = MechanismWorkspace::default();
        let active = nested_counterfactual_with_exo(
            &engine,
            &exo,
            &antecedent_counterfactual::NestedSpec {
                outer: &control_outer,
                inner: &active_inner,
                frozen: &self.frozen,
                outcome: self.query.outcome,
            },
            &mut workspace,
            ctx,
        )
        .map_err(CausalError::rejected)?;
        let control = nested_counterfactual_with_exo(
            &engine,
            &exo,
            &antecedent_counterfactual::NestedSpec {
                outer: &control_outer,
                inner: &control_inner,
                frozen: &self.frozen,
                outcome: self.query.outcome,
            },
            &mut workspace,
            ctx,
        )
        .map_err(CausalError::rejected)?;
        Ok(active - control)
    }
}

/// Fit every mechanism of the nested route's model under `mechanism`.
///
/// Shared by the natural-direct-effect operation and the cross-world edge
/// contrast, so both read the same fitted structural model. The natural direct
/// effect route calls this unpolled form, so its behavior is unchanged.
pub(crate) fn fit_nested_mechanisms(
    compiled: &CompiledCausalModel,
    data: &TabularData,
    mechanism: NestedOutcomeMechanism,
    outcome: VariableId,
) -> Result<CompiledMechanismStore, CausalError> {
    fit_nested_mechanisms_polled(compiled, data, mechanism, outcome, "", &|| false)
}

/// [`fit_nested_mechanisms`] that asks `cancelled` before every node, every
/// candidate family and every cross-validation fold fit of the model crate's
/// fitting loops, and stops with [`CausalError::Cancelled`] (`stage`, no partial
/// store) when it answers `true`. The fitted mechanisms are identical to the
/// unpolled form's: polling only reads the token.
pub(crate) fn fit_nested_mechanisms_polled(
    compiled: &CompiledCausalModel,
    data: &TabularData,
    mechanism: NestedOutcomeMechanism,
    outcome: VariableId,
    stage: &'static str,
    cancelled: &dyn Fn() -> bool,
) -> Result<CompiledMechanismStore, CausalError> {
    let map = |error: ModelError| match error {
        ModelError::Cancelled => CausalError::Cancelled { stage },
        other => CausalError::from(other),
    };
    match mechanism {
        NestedOutcomeMechanism::LinearGaussian => Ok(MechanismRegistry::standard()
            .assign_and_fit_polled(
                compiled,
                data,
                SelectionPolicy::RequireFamily(MechanismFamily::LinearGaussian),
                cancelled,
            )
            .map_err(map)?
            .0),
        NestedOutcomeMechanism::NonSeparableBasis => {
            fit_non_separable_nested_outcome_polled(compiled, data, outcome, stage, cancelled)
        }
    }
}

/// Fit the nested route's mechanisms with a non-separable outcome basis.
///
/// Every parent-free and single-parent node keeps the linear-Gaussian family
/// (a cross-parent product needs two parents), so the mediator equation stays
/// separable and its abducted disturbance is exact. The two-parent outcome is
/// refit as a [`MechanismFamily::LinearSpline`] basis, whose smooth per-parent
/// expansion and cross-parent products make the outcome a non-additive function
/// of treatment and mediator. The disturbance stays additive, so abduction
/// inverts it and the counterfactual replay round-trips.
#[cfg(test)]
pub(super) fn fit_non_separable_nested_outcome(
    compiled: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
) -> Result<CompiledMechanismStore, CausalError> {
    fit_non_separable_nested_outcome_polled(compiled, data, outcome, "", &|| false)
}

/// [`fit_non_separable_nested_outcome`] that polls `cancelled` (see
/// [`fit_nested_mechanisms_polled`]).
fn fit_non_separable_nested_outcome_polled(
    compiled: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
    stage: &'static str,
    cancelled: &dyn Fn() -> bool,
) -> Result<CompiledMechanismStore, CausalError> {
    let registry = MechanismRegistry::standard();
    let (_, mut assignments) = registry
        .assign_and_fit_polled(
            compiled,
            data,
            SelectionPolicy::RequireFamily(MechanismFamily::LinearGaussian),
            cancelled,
        )
        .map_err(|e| match e {
            ModelError::Cancelled => CausalError::Cancelled { stage },
            other => CausalError::from(other),
        })?;
    let mut retargeted = false;
    for assignment in &mut assignments {
        if assignment.variable == outcome {
            assignment.selected = MechanismFamily::LinearSpline;
            retargeted = true;
        }
    }
    if !retargeted {
        return Err(CausalError::Compile {
            message: "nested outcome variable is absent from the fitted mechanism assignments"
                .into(),
        });
    }
    let weights = vec![1.0_f64; data.row_count()];
    registry.refit_weighted_polled(compiled, data, &assignments, &weights, cancelled).map_err(|e| {
        match e {
            ModelError::Cancelled => CausalError::Cancelled { stage },
            other => map_mechanism_fit(other),
        }
    })
}
