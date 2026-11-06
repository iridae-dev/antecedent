//! Refits with a simulated common cause.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalRng, ClassSample, ColumnReplacements, ExecutionContext, Lag, Modal,
    PanelData, PanelUnit, PreparedRefit, PreparedTargetData, TableView, TabularData,
    TargetRefitError, TimeSeriesData, ValidationError, VariableId, WithinClassDistribution,
    data_failed, ensure_lagged, failed, lagged_nodes, repack_lag_mask, series_with, static_node,
    unsupported, within_class_member_cap,
};

impl PreparedRefit<'_> {
    /// The graph posterior with a simulated common cause `u` declared in every atom.
    ///
    /// A DAG atom, static or a lagged template, is repacked onto one more node, `u`, which is
    /// made a contemporaneous parent of the treatment and of the outcome; the atom weights are
    /// unchanged, so the posterior is the same mixture over structures with one root cause
    /// added to each.
    ///
    /// A CPDAG atom is a class, and the class of the DAG with `u` added depends on which member
    /// of the class is meant: an added parent can create a collider with a parent the treatment
    /// already has. The posterior does not say how a class's mass divides among its members, so
    /// this needs the declared assumption `WithinClassDistribution::UniformOverMembers`: every
    /// atom's DAG members are enumerated, each member gets an equal share of the atom's mass, the
    /// common cause is added to each, and the shares are collapsed to the classes of the
    /// augmented DAGs. A class with more members than [`within_class_member_cap`] is reduced to
    /// a deterministic uniform subsample of that many (see [`ClassSample`]); a class the search
    /// budget cannot finish enumerating is refused. The assumption, and the number of members a
    /// subsample dropped, are written into the posterior's diagnostics. A PAG atom or a lagged class posterior is
    /// refused.
    #[allow(
        clippy::too_many_lines,
        reason = "one repacking of the atoms in three shapes: DAG, lagged template and class"
    )]
    fn posterior_with_common_cause(
        &self,
        posterior: &antecedent_discovery::GraphPosterior,
        u: VariableId,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_discovery::GraphPosterior, TargetRefitError> {
        use antecedent_discovery::GraphPosteriorAtomKind as Kind;
        use antecedent_discovery::adjacency_mask::set_edge;
        let n = posterior.n_vars();
        let m = n + 1;
        let refuse = |reason: &'static str| {
            unsupported(reason, "refute one accepted structure of the posterior")
        };
        if u.as_usize() != n || !antecedent_discovery::edge_bits_fit(m) {
            return Err(refuse(
                "the posterior's nodes are not the study's columns, or the added node does not fit \
                 the edge bit index",
            ));
        }
        let (t, y) = (self.treatment.as_usize(), self.outcome.as_usize());
        if t >= n || y >= n {
            return Err(unsupported(
                "the treatment or outcome is not a node of the posterior",
                "a posterior that names both",
            ));
        }
        let lags = posterior.max_lag().map(|lag| lag as usize);
        // Repack a contemporaneous atom onto `m` nodes and add the common cause.
        let repack = |atom: &antecedent_discovery::AdjacencyMask| {
            let mut packed = antecedent_discovery::AdjacencyMask::EMPTY;
            for i in 0..n {
                for j in 0..n {
                    if i != j && atom.has_edge(n, i, j) {
                        packed = set_edge(packed, m, i, j, true);
                    }
                }
            }
            packed = set_edge(packed, m, n, t, true);
            set_edge(packed, m, n, y, true)
        };
        let mut atoms: Vec<antecedent_discovery::AdjacencyMask> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut note: Option<String> = None;
        match posterior.atom_kind() {
            Kind::Dag => {
                atoms = posterior.atoms().iter().map(repack).collect();
                weights = posterior.weights().to_vec();
            }
            Kind::Cpdag if lags.is_none() => {
                if self.options.within_class != Some(WithinClassDistribution::UniformOverMembers) {
                    return Err(refuse(
                        "a class atom does not say how its mass divides among its DAG members, \
                         and the class of an added common cause depends on the member; declare \
                         WithinClassDistribution::UniformOverMembers to assume an even division",
                    ));
                }
                let member_cap = within_class_member_cap(ctx, n);
                let (mut seen_total, mut kept_total) = (0_usize, 0_usize);
                let mut sampled_classes = 0_usize;
                for (atom, &weight) in posterior.atoms().iter().zip(posterior.weights().iter()) {
                    let cpdag =
                        antecedent_discovery::adjacency_mask::cpdag_from_adjacency_mask(atom, n)
                            .map_err(|error| {
                                TargetRefitError::Failed(ValidationError::from(error))
                            })?;
                    // Members are streamed and reduced to their augmented edge sets at once; the
                    // class keeps the `member_cap` members of smallest priority, so a class
                    // above the cap is subsampled deterministically rather than refused.
                    let mut members = antecedent_graph::CpdagCompletionSampler::with_budget(
                        cpdag,
                        usize::MAX,
                        antecedent_core::SearchLimits {
                            operations: antecedent_graph::DEFAULT_COMPLETION_OPERATIONS,
                            depth: usize::MAX,
                        },
                        ctx,
                    )
                    .map_err(|error| {
                        TargetRefitError::Failed(ValidationError::estimation_msg(error.to_string()))
                    })?;
                    let mut kept = ClassSample::new(member_cap);
                    for completion in members.by_ref() {
                        let mut mask = antecedent_discovery::AdjacencyMask::EMPTY;
                        for edge in completion.graph.edges() {
                            if let Some((from, to)) = edge.parent_child() {
                                mask = set_edge(mask, n, from.as_usize(), to.as_usize(), true);
                            }
                        }
                        kept.offer(completion.index, repack(&mask));
                    }
                    crate::strategy_table::check_class_search_stop(members.search_stop())
                        .map_err(|error| failed(&error))?;
                    if members.search_stop().is_some() {
                        return Err(refuse(
                            "the search budget stopped the enumeration of a class's DAG members \
                             before it finished, so an even division of its mass is not defined",
                        ));
                    }
                    if kept.seen == 0 {
                        return Err(refuse(
                            "a class has no enumerated DAG members, so an even division of its \
                             mass is not defined",
                        ));
                    }
                    seen_total += kept.seen;
                    kept_total += kept.members.len();
                    sampled_classes += usize::from(kept.seen > kept.members.len());
                    let share = weight / kept.members.len() as f64;
                    for packed in kept.into_members() {
                        atoms.push(packed);
                        weights.push(share);
                    }
                }
                let mut text = String::from(
                    "declared assumption: the mass of each class divides evenly among its DAG \
                     members before the common cause is added",
                );
                if sampled_classes > 0 {
                    text = format!(
                        "{text}; {sampled_classes} class(es) above the per-class limit of \
                         {member_cap} members were reduced to a deterministic subsample, \
                         dropping {} of {seen_total} enumerated members",
                        seen_total - kept_total,
                    );
                }
                note = Some(text);
            }
            _ => {
                return Err(refuse(
                    "a PAG posterior or a lagged class posterior has no DAG members to add the \
                     common cause to",
                ));
            }
        }
        let total: f64 = weights.iter().sum();
        let mut edge = vec![0.0; m * m];
        for (atom, &weight) in atoms.iter().zip(&weights) {
            for i in 0..m {
                for j in 0..m {
                    if i != j && atom.has_edge(m, i, j) {
                        edge[i * m + j] += weight / total;
                    }
                }
            }
        }
        let weights: Vec<f64> = weights.iter().map(|w| w / total).collect();
        let mut diagnostics = posterior.diagnostics().clone();
        if let Some(note) = note {
            diagnostics.notes.push(Arc::from(note.as_str()));
        }
        let mut rebuilt = antecedent_discovery::GraphPosterior::from_atoms(
            m,
            weights.clone(),
            atoms,
            edge.clone(),
            edge,
            antecedent_discovery::SamplingSummary {
                ess: posterior.ess(),
                diagnostics,
                rejected_invalid: posterior.rejected_invalid(),
            },
        )
        .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
        if let (Some(max_lag), Some(lag_masks)) = (lags, posterior.lag_masks()) {
            // Lagged edges keep their sources and targets; the added node has none, and the bit
            // of an edge moves with the node count.
            if u32::try_from(max_lag * m * m).is_err() {
                return Err(refuse(
                    "the lagged template has more edge bits than a bit index addresses with one \
                     more node",
                ));
            }
            let packed: Vec<antecedent_discovery::LagMask> =
                lag_masks.iter().map(|old| repack_lag_mask(old, n, m, max_lag)).collect();
            let mut lagged = vec![0.0; max_lag * m * m];
            for (mask, &weight) in packed.iter().zip(&weights) {
                for bit in mask.set_bits() {
                    lagged[bit as usize] += weight;
                }
            }
            rebuilt = rebuilt
                .with_lagged_marginals(u32::try_from(max_lag).unwrap_or(u32::MAX), lagged)
                .and_then(|post| post.with_lag_masks(packed))
                .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
        }
        if posterior.atom_kind() == Kind::Cpdag {
            rebuilt = rebuilt
                .to_equivalence_classes()
                .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
        }
        Ok(match posterior.algorithm() {
            Some(algorithm) => rebuilt.with_algorithm(Arc::clone(algorithm)),
            None => rebuilt,
        })
    }

    /// The graph with a simulated common cause `u` of treatment and outcome declared.
    fn graph_with_common_cause(
        &self,
        u: VariableId,
    ) -> Result<crate::accepted::AcceptedGraph, TargetRefitError> {
        let graph = &self.study().graph;
        let declare = |error: antecedent_graph::GraphError| {
            TargetRefitError::Failed(ValidationError::estimation_msg(format!(
                "declaring the simulated common cause: {error}"
            )))
        };
        if self.study().graph_posterior.is_some() {
            return Err(unsupported(
                "a graph posterior is a mixture of structures; declaring one common cause \
                 in every atom changes the posterior",
                "refute one accepted structure of the posterior",
            ));
        }
        if let Some(dag) = graph.as_dag() {
            let mut dag = dag.clone();
            let node = dag.add_node(antecedent_core::NodeRef::Static(u)).map_err(declare)?;
            let (Some(t), Some(y)) =
                (static_node(&dag, self.treatment), static_node(&dag, self.outcome))
            else {
                return Err(unsupported(
                    "the treatment or outcome is not a node of the graph",
                    "a graph that names both",
                ));
            };
            dag.insert_directed(node, t).map_err(declare)?;
            dag.insert_directed(node, y).map_err(declare)?;
            return Ok(crate::accepted::AcceptedGraph::from(dag));
        }
        if let Some(tdag) = graph.as_temporal_dag() {
            let mut tdag = tdag.clone();
            let y0 =
                ensure_lagged(&mut tdag, self.outcome, Lag::CONTEMPORANEOUS).map_err(declare)?;
            let treatment_nodes = lagged_nodes(&tdag, self.treatment);
            if treatment_nodes.is_empty() {
                return Err(unsupported(
                    "the treatment is not a node of the temporal graph",
                    "a graph that names it",
                ));
            }
            for (t, lag) in treatment_nodes {
                let raw = u32::try_from(lag).unwrap_or(u32::MAX);
                let u_at = ensure_lagged(&mut tdag, u, Lag::from_raw(raw)).map_err(declare)?;
                tdag.insert_directed(u_at, t).map_err(declare)?;
                if u_at != y0 && !tdag.children(u_at).contains(&y0) {
                    tdag.insert_directed(u_at, y0).map_err(declare)?;
                }
            }
            return Ok(crate::accepted::AcceptedGraph::temporal_dag(tdag));
        }
        Err(unsupported(
            "a simulated common cause is declared on a DAG or a temporal DAG; this graph class \
             carries marks a declared parent does not resolve",
            "accept a DAG or temporal DAG of the structure",
        ))
    }

    /// Perturb the outcome and a continuous treatment by the same simulated cause.
    fn plant_common_cause(
        &self,
        storage: &antecedent_data::OwnedColumnarStorage,
        u: &[f64],
    ) -> Result<ColumnReplacements, TargetRefitError> {
        let strength = self.options.common_cause_strength;
        if strength == 0.0 {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let y = Self::column_of(storage, self.outcome)?;
        out.push((self.outcome, y.iter().zip(u).map(|(y, u)| y + strength * u).collect()));
        let t = Self::column_of(storage, self.treatment)?;
        #[allow(
            clippy::float_cmp,
            reason = "a binary treatment is coded exactly 0.0 or 1.0, so exact equality is the test"
        )]
        let binary = t.iter().all(|v| *v == 0.0 || *v == 1.0);
        if !binary {
            out.push((self.treatment, t.iter().zip(u).map(|(t, u)| t + strength * u).collect()));
        }
        Ok(out)
    }

    /// Random common cause through the compiler: append a simulated cause, declare it as a
    /// parent of treatment and outcome, prepare the study on the augmented data and run it.
    pub(super) fn random_common_cause(
        &self,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let draw = |n: usize, rng: &mut CausalRng| -> Arc<[f64]> {
            (0..n).map(|_| antecedent_kernels::standard_normal(rng)).collect()
        };
        let name = |storage: &antecedent_data::OwnedColumnarStorage| {
            let names: Vec<&str> =
                storage.schema().variables().iter().map(|v| v.name.as_ref()).collect();
            // Among `names.len() + 1` candidates at least one is free.
            (0..=names.len())
                .map(|i| format!("simulated_common_cause_{i}"))
                .find(|candidate| !names.contains(&candidate.as_str()))
                .unwrap_or_default()
        };
        let append = |table: TabularData, u: Arc<[f64]>, label: &str| {
            table.with_appended_float(label, u).map_err(data_failed)
        };
        let (data, u_id) = match self.data {
            PreparedTargetData::Tabular(data) => {
                let u = draw(data.row_count(), rng);
                let planted = self.plant_common_cause(data.storage(), &u)?;
                let base = if planted.is_empty() {
                    data.clone()
                } else {
                    data.with_replaced_floats(&planted).map_err(data_failed)?
                };
                let (table, id) = append(base, u, &name(data.storage()))?;
                (Modal::Tabular(table), id)
            }
            PreparedTargetData::Series(series) => {
                let u = draw(series.row_count(), rng);
                let planted = self.plant_common_cause(series.storage(), &u)?;
                let base = series_with(series, &planted, None, None)?;
                let (table, id) =
                    append(TabularData::new(base.storage().clone()), u, &name(series.storage()))?;
                let out =
                    TimeSeriesData::try_new(table.storage().clone(), base.time_index().clone())
                        .map_err(data_failed)?;
                (Modal::Series(out), id)
            }
            PreparedTargetData::Panel(panel) => {
                let label = name(panel.units()[0].series.storage());
                let mut units = Vec::with_capacity(panel.unit_count());
                let mut id = None;
                for unit in panel.units() {
                    let u = draw(unit.series.row_count(), rng);
                    let planted = self.plant_common_cause(unit.series.storage(), &u)?;
                    let base = series_with(&unit.series, &planted, None, None)?;
                    let (table, uid) = append(TabularData::new(base.storage().clone()), u, &label)?;
                    id = Some(uid);
                    units.push(PanelUnit {
                        unit_id: unit.unit_id,
                        series: TimeSeriesData::try_new(
                            table.storage().clone(),
                            base.time_index().clone(),
                        )
                        .map_err(data_failed)?,
                    });
                }
                let panel = PanelData::try_new(units).map_err(data_failed)?;
                (Modal::Panel(panel), id.ok_or_else(|| unsupported("an empty panel", "data"))?)
            }
            PreparedTargetData::MultiEnv(environments) => {
                let label = name(environments.environments()[0].storage());
                let mut members = Vec::with_capacity(environments.env_count());
                let mut id = None;
                for environment in environments.environments() {
                    let u = draw(environment.row_count(), rng);
                    let planted = self.plant_common_cause(environment.storage(), &u)?;
                    let base = series_with(environment, &planted, None, None)?;
                    let (table, uid) = append(TabularData::new(base.storage().clone()), u, &label)?;
                    id = Some(uid);
                    members.push(
                        TimeSeriesData::try_new(table.storage().clone(), base.time_index().clone())
                            .map_err(data_failed)?,
                    );
                }
                let rebuilt =
                    antecedent_data::MultiEnvironmentData::try_new(members).map_err(data_failed)?;
                (Modal::MultiEnv(rebuilt), id.ok_or_else(|| unsupported("no environment", "data"))?)
            }
        };
        let mut study = self.study().clone();
        if let Some(posterior) = &self.study().graph_posterior {
            study.graph_posterior = Some(self.posterior_with_common_cause(posterior, u_id, ctx)?);
        } else {
            study.graph = Arc::new(self.graph_with_common_cause(u_id)?);
        }
        let refitted = Self::compile_and_run(study, &data, ctx).map_err(|error| {
            if matches!(error, CausalError::Cancelled { .. } | CausalError::Resource { .. }) {
                return failed(&error);
            }
            unsupported(
                format!("the study with the simulated common cause declared did not run: {error}"),
                "a route that admits an extra declared covariate",
            )
        })?;
        self.read_back(&refitted, false, ctx)
    }
}
