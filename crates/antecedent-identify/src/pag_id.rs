//! Direct marginal and conditional identification under PAG Markov equivalence.
//!
//! The recursion uses the region decomposition of Jaber, Zhang and Bareinboim
//! (ICML 2019, Corollary 1), with the simpler bucket elimination and the
//! conditional calculus of Jaber et al. (NeurIPS 2022, Algorithms 1–2).
//! This assumes a valid, fully oriented oracle PAG of a Markov equivalence
//! class without selection variables, and a strictly positive observational law.

use std::collections::BTreeSet;

use antecedent_core::{ExecutionContext, RegimeId, VariableId};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::{DenseNodeId, Endpoint, MiddleMark, NodeRef, Pag};

type Nodes = BTreeSet<DenseNodeId>;

/// A stop distinct from a proved failure of identification.
#[derive(Clone, Debug, thiserror::Error)]
pub enum PagIdError {
    /// The query or supplied graph does not satisfy the input contract.
    #[error("invalid direct PAG identification input: {0}")]
    Invalid(&'static str),
    /// Metered graph-operation or path-search budget exhausted.
    #[error("direct PAG identification work exhausted after {0} steps")]
    Budget(u64),
    /// The deterministic recursion or simple-path depth ceiling was reached.
    #[error("direct PAG identification depth limit {limit} reached after {work} steps")]
    Depth {
        /// Work done before encountering the ceiling.
        work: u64,
        /// The maximum depth accepted by the expression program.
        limit: usize,
    },
    /// Caller cancelled the symbolic search.
    #[error("direct PAG identification cancelled")]
    Cancelled,
    /// The caller's memory budget could not admit the numerical work.
    #[error("direct PAG identification admission: {0}")]
    Admission(#[from] antecedent_core::execution::AdmissionError),
}

/// The terminal obstruction returned by the published identification algorithm.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PagObstruction {
    /// IDP reached a subproblem with neither an eliminable bucket nor a proper region.
    Marginal {
        /// Variables of the remaining kernel.
        available: Vec<VariableId>,
        /// Variables required in the current subproblem.
        required: Vec<VariableId>,
    },
    /// CIDP could not exchange a bucket's interventions for observations.
    Conditional {
        /// The partially retained bucket.
        bucket: Vec<VariableId>,
        /// Its intervention coordinates.
        interventions: Vec<VariableId>,
        /// A definite m-connecting path in the manipulated PAG.
        path: Vec<VariableId>,
    },
}

/// An original symbolic expression of a PAG interventional law.
#[derive(Clone, Debug)]
pub struct PagFunctional {
    /// Interned expression arena, with a single observational joint leaf.
    pub arena: CausalExprArena,
    /// Root of the observational functional.
    pub root: ExprId,
    /// Outcome coordinates of the original query.
    pub outcomes: Vec<VariableId>,
}

/// A completed direct PAG result and its symbolic work count.
#[derive(Clone, Debug)]
pub struct PagIdentification {
    /// Derived observational functional or graphical nonidentification evidence.
    pub result: Result<PagFunctional, PagObstruction>,
    /// Metered graph visits, path work and recursive calls.
    pub work: u64,
}

struct Meter<'a> {
    ctx: &'a ExecutionContext,
    spent: u64,
}

impl Meter<'_> {
    fn step(&mut self) -> Result<(), PagIdError> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(PagIdError::Cancelled);
        }
        if self.spent >= self.ctx.id_work_limit() {
            return Err(PagIdError::Budget(self.spent));
        }
        self.spent += 1;
        Ok(())
    }

    fn depth(&self, depth: usize) -> Result<(), PagIdError> {
        let limit = antecedent_expr::ProgramLimits::default().max_depth;
        if depth >= limit {
            return Err(PagIdError::Depth { work: self.spent, limit });
        }
        Ok(())
    }
}

struct Graph<'a> {
    marks: &'a Pag,
    visible: BTreeSet<(DenseNodeId, DenseNodeId)>,
    variables: Vec<VariableId>,
}

impl<'a> Graph<'a> {
    fn new(pag: &'a Pag, meter: &mut Meter<'_>) -> Result<Self, PagIdError> {
        let mut variables = Vec::with_capacity(pag.node_count());
        let mut visible = BTreeSet::new();
        for (index, node) in pag.nodes().iter().enumerate() {
            meter.step()?;
            let NodeRef::Static(variable) = *node else {
                return Err(PagIdError::Invalid("a static oracle PAG is required"));
            };
            variables.push(variable);
            let index = u32::try_from(index).map_err(|_| PagIdError::Invalid("graph size"))?;
            let from = DenseNodeId::from_raw(index);
            for (to, at_from, at_to) in pag.neighbors(from) {
                meter.step()?;
                let edge = pag.edge_between(from, to).expect("adjacent by construction");
                if at_from == Endpoint::Conflict
                    || at_to == Endpoint::Conflict
                    || edge.middle != MiddleMark::Empty
                    || (at_from == Endpoint::Tail && at_to == Endpoint::Tail)
                {
                    return Err(PagIdError::Invalid(
                        "conflict, search-state and selection edges are outside the oracle PAG domain",
                    ));
                }
                if at_from == Endpoint::Tail
                    && at_to == Endpoint::Arrow
                    && crate::joint_response::visible_metered(pag, from, to, || meter.step())?
                {
                    visible.insert((from, to));
                }
            }
        }
        Ok(Self { marks: pag, visible, variables })
    }

    fn locate(&self, variables: &[VariableId]) -> Result<Nodes, PagIdError> {
        let nodes: Nodes = variables
            .iter()
            .map(|v| {
                self.variables
                    .iter()
                    .position(|have| have == v)
                    .and_then(|i| u32::try_from(i).ok())
                    .map(DenseNodeId::from_raw)
                    .ok_or(PagIdError::Invalid("query variable absent from PAG"))
            })
            .collect::<Result<_, _>>()?;
        if nodes.len() != variables.len() {
            return Err(PagIdError::Invalid("duplicate query variable"));
        }
        Ok(nodes)
    }

    fn labels(&self, nodes: &Nodes) -> Vec<VariableId> {
        nodes.iter().map(|node| self.variables[node.as_usize()]).collect()
    }

    fn connected(
        pag: &Pag,
        seeds: &Nodes,
        within: &Nodes,
        mut pass: impl FnMut(Endpoint, Endpoint) -> bool,
        meter: &mut Meter<'_>,
    ) -> Result<Nodes, PagIdError> {
        let mut found = seeds.clone();
        let mut todo: Vec<_> = seeds.iter().copied().collect();
        while let Some(node) = todo.pop() {
            meter.step()?;
            for (next, from, to) in pag.neighbors(node) {
                meter.step()?;
                if within.contains(&next) && pass(from, to) && found.insert(next) {
                    todo.push(next);
                }
            }
        }
        Ok(found)
    }

    fn ancestors(
        &self,
        seeds: &Nodes,
        within: &Nodes,
        meter: &mut Meter<'_>,
    ) -> Result<Nodes, PagIdError> {
        Self::connected(self.marks, seeds, within, |_, to| to != Endpoint::Arrow, meter)
    }

    fn descendants(
        &self,
        seeds: &Nodes,
        within: &Nodes,
        meter: &mut Meter<'_>,
    ) -> Result<Nodes, PagIdError> {
        Self::connected(self.marks, seeds, within, |from, _| from != Endpoint::Arrow, meter)
    }

    fn buckets(&self, within: &Nodes, meter: &mut Meter<'_>) -> Result<Vec<Nodes>, PagIdError> {
        let mut left = within.clone();
        let mut result = Vec::new();
        while let Some(first) = left.first().copied() {
            let bucket = Self::connected(
                self.marks,
                &Nodes::from([first]),
                within,
                |from, to| from == Endpoint::Circle && to == Endpoint::Circle,
                meter,
            )?;
            left = difference(&left, &bucket);
            result.push(bucket);
        }
        Ok(result)
    }

    fn visible_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
        self.visible.contains(&(a, b)) || self.visible.contains(&(b, a))
    }

    // A pc-component is a one-step relation over entire collider paths. Its
    // connectivity is not transitive. At an interior node a path can continue
    // only through two arrowheads, so all interior-to-interior edges are bidirected.
    fn pc_component(
        &self,
        seeds: &Nodes,
        within: &Nodes,
        meter: &mut Meter<'_>,
    ) -> Result<Nodes, PagIdError> {
        let mut found = seeds.clone();
        let mut expanded = seeds.clone();
        let mut todo: Vec<_> = seeds.iter().map(|&node| (node, true)).collect();
        while let Some((node, endpoint)) = todo.pop() {
            meter.step()?;
            for (next, from, to) in self.marks.neighbors(node) {
                meter.step()?;
                if !within.contains(&next)
                    || self.visible_edge(node, next)
                    || (!endpoint && from != Endpoint::Arrow)
                {
                    continue;
                }
                found.insert(next);
                if to == Endpoint::Arrow && expanded.insert(next) {
                    todo.push((next, false));
                }
            }
        }
        Ok(found)
    }

    fn region(
        &self,
        seeds: &Nodes,
        within: &Nodes,
        buckets: &[Nodes],
        meter: &mut Meter<'_>,
    ) -> Result<Nodes, PagIdError> {
        let with_colliders = self.pc_component(seeds, within, meter)?;
        Ok(buckets.iter().filter(|b| !b.is_disjoint(&with_colliders)).flatten().copied().collect())
    }

    fn manipulated(
        &self,
        upper: &Nodes,
        lower: &Nodes,
        meter: &mut Meter<'_>,
    ) -> Result<Pag, PagIdError> {
        let mut graph = Pag::with_variables(
            u32::try_from(self.variables.len()).map_err(|_| PagIdError::Invalid("graph size"))?,
        );
        for from in all_nodes(self.variables.len())? {
            for (to, at_from, at_to) in self.marks.neighbors(from) {
                meter.step()?;
                if from > to
                    || (upper.contains(&from) && at_from == Endpoint::Arrow)
                    || (upper.contains(&to) && at_to == Endpoint::Arrow)
                {
                    continue;
                }
                let outgoing = (lower.contains(&from) && at_from == Endpoint::Tail)
                    || (lower.contains(&to) && at_to == Endpoint::Tail);
                if outgoing && self.visible_edge(from, to) {
                    continue;
                }
                let edge = if outgoing {
                    antecedent_graph::MarkedEdge::bidirected(from, to)
                } else {
                    self.marks.edge_between(from, to).expect("adjacent by construction")
                };
                graph.insert_marked(edge).map_err(|_| PagIdError::Invalid("graph manipulation"))?;
            }
        }
        Ok(graph)
    }
}

fn all_nodes(n: usize) -> Result<Nodes, PagIdError> {
    let n = u32::try_from(n).map_err(|_| PagIdError::Invalid("graph size"))?;
    Ok((0..n).map(DenseNodeId::from_raw).collect())
}

fn difference(a: &Nodes, b: &Nodes) -> Nodes {
    a.difference(b).copied().collect()
}

fn union(a: &Nodes, b: &Nodes) -> Nodes {
    a.union(b).copied().collect()
}

struct Separation<'a> {
    pag: &'a Pag,
    targets: &'a Nodes,
    given: &'a Nodes,
    open_colliders: Nodes,
}

impl Separation<'_> {
    fn search(
        &self,
        path: &mut Vec<DenseNodeId>,
        meter: &mut Meter<'_>,
    ) -> Result<Option<Vec<DenseNodeId>>, PagIdError> {
        meter.step()?;
        meter.depth(path.len())?;
        let node = *path.last().expect("the initialized path is nonempty");
        if self.targets.contains(&node) {
            return Ok(Some(path.clone()));
        }
        for (next, outgoing, _) in self.pag.neighbors(node) {
            meter.step()?;
            if path.contains(&next) {
                continue;
            }
            if let [.., previous, _] = path.as_slice() {
                let incoming = self
                    .pag
                    .neighbors(node)
                    .find_map(|(id, at_node, _)| (id == *previous).then_some(at_node))
                    .expect("the path retains its incoming edge");
                let definite_collider = incoming == Endpoint::Arrow && outgoing == Endpoint::Arrow;
                let definite_noncollider = incoming == Endpoint::Tail
                    || outgoing == Endpoint::Tail
                    || (incoming == Endpoint::Circle
                        && outgoing == Endpoint::Circle
                        && self.pag.edge_between(*previous, next).is_none());
                if !(definite_collider && self.open_colliders.contains(&node)
                    || definite_noncollider && !self.given.contains(&node))
                {
                    continue;
                }
            }
            path.push(next);
            let witness = self.search(path, meter)?;
            path.pop();
            if witness.is_some() {
                return Ok(witness);
            }
        }
        Ok(None)
    }
}

fn exchange_witness(
    graph: &Graph<'_>,
    all: &Nodes,
    other_actions: &Nodes,
    changing: &Nodes,
    outcomes: &Nodes,
    given: &Nodes,
    meter: &mut Meter<'_>,
) -> Result<Option<Vec<VariableId>>, PagIdError> {
    let manipulated = graph.manipulated(other_actions, changing, meter)?;
    let given = union(other_actions, given);
    let open_colliders = Graph::connected(
        &manipulated,
        &given,
        all,
        |from, to| from == Endpoint::Arrow && to == Endpoint::Tail,
        meter,
    )?;
    let search = Separation { pag: &manipulated, targets: outcomes, given: &given, open_colliders };
    for &source in changing {
        if let Some(path) = search.search(&mut vec![source], meter)? {
            return Ok(Some(path.iter().map(|v| graph.variables[v.as_usize()]).collect()));
        }
    }
    Ok(None)
}

fn exchange_buckets(
    graph: &Graph<'_>,
    all: &Nodes,
    x: &mut Nodes,
    y: &Nodes,
    z: &mut Nodes,
    meter: &mut Meter<'_>,
) -> Result<Option<PagObstruction>, PagIdError> {
    let buckets = graph.buckets(all, meter)?;
    loop {
        let relevant = graph.ancestors(&union(y, z), &difference(all, x), meter)?;
        let Some(split) =
            buckets.iter().find(|b| !b.is_disjoint(&relevant) && !b.is_subset(&relevant))
        else {
            break;
        };
        let moving: Nodes = split.intersection(x).copied().collect();
        if moving.is_empty() {
            return Err(PagIdError::Invalid("a split bucket has no intervened member"));
        }
        let rest = difference(x, &moving);
        if let Some(path) = exchange_witness(graph, all, &rest, &moving, y, z, meter)? {
            return Ok(Some(PagObstruction::Conditional {
                bucket: graph.labels(split),
                interventions: graph.labels(&moving),
                path,
            }));
        }
        *x = rest;
        z.extend(moving);
    }

    loop {
        let mut exchange = None;
        for b in &buckets {
            let moving: Nodes = b.intersection(z).copied().collect();
            if !moving.is_empty()
                && exchange_witness(graph, all, x, &moving, y, &difference(z, &moving), meter)?
                    .is_none()
            {
                exchange = Some(moving);
                break;
            }
        }
        let Some(moving) = exchange else { break };
        *z = difference(z, &moving);
        x.extend(moving);
    }
    Ok(None)
}

struct Identify<'a> {
    graph: &'a Graph<'a>,
    arena: CausalExprArena,
}

impl Identify<'_> {
    fn sum(&mut self, vars: &Nodes, expr: ExprId) -> ExprId {
        if vars.is_empty() {
            return expr;
        }
        let variables = self.arena.intern_var_set(self.graph.labels(vars));
        self.arena.intern(ExprNode::SumOut { variables, expr })
    }

    fn ratio(&mut self, numerator: ExprId, denominator: ExprId) -> ExprId {
        self.arena.intern(ExprNode::Ratio { numerator, denominator })
    }

    fn recurse(
        &mut self,
        required: &Nodes,
        available: &Nodes,
        kernel: ExprId,
        depth: usize,
        meter: &mut Meter<'_>,
    ) -> Result<Result<ExprId, PagObstruction>, PagIdError> {
        meter.step()?;
        meter.depth(depth)?;
        if required.is_empty() {
            let empty = self.arena.intern_list([]);
            return Ok(Ok(self.arena.intern(ExprNode::Product(empty))));
        }
        if required == available {
            return Ok(Ok(kernel));
        }
        for bucket in self.graph.buckets(available, meter)? {
            if !bucket.is_disjoint(required) {
                continue;
            }
            let descendants = self.graph.descendants(&bucket, available, meter)?;
            let pc = self.graph.pc_component(&bucket, available, meter)?;
            if pc.intersection(&descendants).any(|node| !bucket.contains(node)) {
                continue;
            }
            // Proposition 2 (2022): divide by Q(B | T \ PossDe(B)).
            let numerator = self.sum(&difference(&descendants, &bucket), kernel);
            let denominator = self.sum(&descendants, kernel);
            let conditional = self.ratio(numerator, denominator);
            let reduced = self.ratio(kernel, conditional);
            return self.recurse(
                required,
                &difference(available, &bucket),
                reduced,
                depth + 1,
                meter,
            );
        }
        let buckets = self.graph.buckets(required, meter)?;
        for bucket in &buckets {
            let region = self.graph.region(bucket, required, &buckets, meter)?;
            if &region == required {
                continue;
            }
            let remainder =
                self.graph.region(&difference(required, &region), required, &buckets, meter)?;
            let shared = region.intersection(&remainder).copied().collect();
            let first = match self.recurse(&region, available, kernel, depth + 1, meter)? {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            };
            let second = match self.recurse(&remainder, available, kernel, depth + 1, meter)? {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            };
            let common = match self.recurse(&shared, available, kernel, depth + 1, meter)? {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            };
            let factors = self.arena.intern_list([first, second]);
            let product = self.arena.intern(ExprNode::Product(factors));
            return Ok(Ok(self.ratio(product, common)));
        }
        Ok(Err(PagObstruction::Marginal {
            available: self.graph.labels(available),
            required: self.graph.labels(required),
        }))
    }
}

/// Identify a marginal or conditional interventional law straight from an oracle PAG.
///
/// The caller declares the full, valid Markov-equivalence PAG of a recursive model
/// allowing latent confounding and no selection variables. Empty `given` invokes
/// marginal IDP. The arbitrary disjoint variable sets bind `P(y | do(x), z)` otherwise.
///
/// # Errors
/// Invalid graph/query declarations, caller cancellation, or an exhausted budget.
/// A completed nonidentification proof is part of the successful result.
pub fn identify_pag_direct(
    pag: &Pag,
    actions: &[VariableId],
    outcomes: &[VariableId],
    given: &[VariableId],
    population: &str,
    ctx: &ExecutionContext,
) -> Result<PagIdentification, PagIdError> {
    if ctx.cancellation.is_cancelled() {
        return Err(PagIdError::Cancelled);
    }
    if population.trim().is_empty() {
        return Err(PagIdError::Invalid("the observational population is required"));
    }
    let nodes = pag.node_count() as u64;
    let estimate = nodes
        .saturating_mul(256)
        .saturating_add(nodes.saturating_mul(nodes).saturating_mul(48))
        .saturating_add(ctx.id_work_limit().min(100_000).saturating_mul(256));
    let _lease = ctx.acquire_serial_numerical_admission(estimate)?;
    let mut meter = Meter { ctx, spent: 0 };
    let graph = Graph::new(pag, &mut meter)?;
    let all = all_nodes(pag.node_count())?;
    let mut x = graph.locate(actions)?;
    let y = graph.locate(outcomes)?;
    let mut z = graph.locate(given)?;
    if y.is_empty() || !x.is_disjoint(&y) || !x.is_disjoint(&z) || !y.is_disjoint(&z) {
        return Err(PagIdError::Invalid("query sets must be disjoint and outcomes nonempty"));
    }
    if !z.is_empty()
        && let Some(stop) = exchange_buckets(&graph, &all, &mut x, &y, &mut z, &mut meter)?
    {
        return Ok(PagIdentification { result: Err(stop), work: meter.spent });
    }
    let target = union(&y, &z);
    let required = graph.ancestors(&target, &difference(&all, &x), &mut meter)?;
    let mut identify = Identify { graph: &graph, arena: CausalExprArena::new() };
    let variables = identify.arena.intern_var_set(graph.variables.iter().copied());
    let empty = identify.arena.empty_var_set();
    let observational = identify.arena.empty_intervention_set();
    let joint = identify
        .arena
        .intern_distribution_tagged(
            antecedent_expr::DistributionLeaf {
                variables,
                conditioned_on: empty,
                intervention: observational,
                domain: DomainRef::Observational,
            },
            population,
            Some(RegimeId::from_raw(0)),
            Some(DomainRef::Observational),
        )
        .map_err(|_| PagIdError::Invalid("observational factor"))?;
    let result = identify.recurse(&required, &all, joint, 0, &mut meter)?.map(|identified| {
        let marginal = identify.sum(&difference(&required, &target), identified);
        let root = if z.is_empty() {
            marginal
        } else {
            let normalizer = identify.sum(&y, marginal);
            identify.ratio(marginal, normalizer)
        };
        PagFunctional { arena: identify.arena, root, outcomes: graph.labels(&y) }
    });
    Ok(PagIdentification { result, work: meter.spent })
}
