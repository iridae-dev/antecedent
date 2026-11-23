//! Route free support through DAG or SCC compilation.

use std::collections::BTreeMap;

use super::{
    CompileBudget, CompiledEquilibrium, CyclicComponent, DAG_EQUILIBRIUM_ATTEMPT,
    GaussianDisturbance, HardClamp, LinearStructuralMap, ModelBinding, SCC_COMPILE_ATTEMPT,
    SccCompileError, finish_compile, free_indices, plan_bytes, reject_compile_inputs, route_work,
    structural_coefficient, support_blocks,
};

/// How the free support of a linear map was scheduled.
///
/// Specified models remain model-conditional. Choosing this schedule grants no
/// scientific support.
#[derive(Clone, Debug, PartialEq)]
pub enum LinearSupportRoute {
    /// Topological order of the free nodes, sources before sinks.
    ///
    /// Each index is one mechanism. None of them is a cyclic block.
    Acyclic {
        /// Free node indices in topological order.
        order: Vec<usize>,
    },
    /// The DAG algorithm refused a directed cycle.
    ///
    /// `next_attempt` is that refusal. `blocks` is the condensation schedule
    /// the cyclic compiler may then run. It is not applied by the DAG algorithm.
    Cyclic {
        /// Existing next attempt from the DAG refusal.
        next_attempt: String,
        /// Strongly connected free blocks, sources before sinks.
        blocks: Vec<Vec<usize>>,
    },
}

/// Equilibrium compiled by [`compile_routed_gaussian_equilibrium`].
#[derive(Clone, Debug, PartialEq)]
pub struct RoutedEquilibrium {
    route: LinearSupportRoute,
    equilibrium: CompiledEquilibrium,
}

impl RoutedEquilibrium {
    /// Schedule the DAG algorithm selected, including a cycle refusal when one was required.
    #[must_use]
    pub fn route(&self) -> &LinearSupportRoute {
        &self.route
    }

    /// Compiled equilibrium. An acyclic route is not a cyclic-block solve.
    pub fn equilibrium(&self) -> &CompiledEquilibrium {
        &self.equilibrium
    }
}

/// Support routes of one linear map, by clamped node set, each computed once.
///
/// A route reads only which coefficients are nonzero and which nodes are clamped. The set is
/// bound to the exact coefficients it was built from and answers only for a map with those
/// coefficients, so a route is never applied to another support, however it was carried.
/// Computing a route is the one place its routing work is charged; a compile that is handed a
/// held route spends only its factorization work.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearRouteSet {
    coefficients: Vec<f64>,
    routes: BTreeMap<Vec<usize>, LinearSupportRoute>,
}

impl LinearRouteSet {
    /// An empty set for the coefficients of `model`.
    #[must_use]
    pub fn new(model: &LinearStructuralMap) -> Self {
        Self { coefficients: model.coefficients().to_vec(), routes: BTreeMap::new() }
    }

    /// Whether `model` has exactly the coefficients this set was built for.
    #[must_use]
    pub fn binds(&self, model: &LinearStructuralMap) -> bool {
        self.coefficients.as_slice() == model.coefficients()
    }

    /// The held route of `model` under `clamp`, when `model` is the map this set is bound to.
    #[must_use]
    pub fn get(
        &self,
        model: &LinearStructuralMap,
        clamp: &HardClamp,
    ) -> Option<&LinearSupportRoute> {
        if clamp.dimension() != model.dimension() || !self.binds(model) {
            return None;
        }
        self.routes.get(clamp.indices())
    }

    /// The route of `model` under `clamp`, and whether it was computed by this call.
    ///
    /// A held route is returned without any work. Otherwise the route is computed and, when
    /// `model` is the map this set is bound to, held for the next caller.
    ///
    /// # Errors
    ///
    /// The errors of [`route_linear_support`].
    pub fn route(
        &mut self,
        model: &LinearStructuralMap,
        clamp: &HardClamp,
    ) -> Result<(LinearSupportRoute, bool), SccCompileError> {
        if let Some(route) = self.get(model, clamp) {
            return Ok((route.clone(), false));
        }
        let route = route_linear_support(model, clamp)?;
        if self.binds(model) {
            self.routes.insert(clamp.indices().to_vec(), route.clone());
        }
        Ok((route, true))
    }

    /// Number of held routes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Whether no route is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }
}

/// Ask the DAG algorithm which schedule the free support uses.
///
/// An acyclic support returns its topological order and does not build a
/// condensation. A directed cycle is refused by [`antecedent_graph::Dag::insert_directed`]; the
/// error text is the existing next attempt, and the condensation is returned
/// only as the later cyclic schedule.
///
/// # Errors
///
/// A clamp dimension mismatch, or a support edge the graph rejected for a
/// reason other than a directed cycle.
pub fn route_linear_support(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
) -> Result<LinearSupportRoute, SccCompileError> {
    let n = model.dimension();
    if clamp.dimension() != n {
        return Err(SccCompileError::ClampDimension { clamp: clamp.dimension(), model: n });
    }
    let free = free_indices(n, clamp.indices());
    u32::try_from(free.len())
        .map_err(|_| SccCompileError::Shape { message: "free block does not fit in a dense id" })?;
    let m = free.len();
    let mut indegree = vec![0usize; m];
    for (into, &into_node) in free.iter().enumerate() {
        for (from, &from_node) in free.iter().enumerate() {
            if into != from && structural_coefficient(model, into_node, from_node)? != 0.0 {
                indegree[into] += 1;
            }
        }
    }
    // Match the graph DAG's deterministic LIFO Kahn tie policy and its
    // ascending target insertion order, while scanning each matrix row twice.
    let mut queue: Vec<usize> = (0..m).filter(|&i| indegree[i] == 0).collect();
    let mut order = Vec::with_capacity(m);
    while let Some(from) = queue.pop() {
        order.push(free[from]);
        for into in 0..m {
            if into != from && structural_coefficient(model, free[into], free[from])? != 0.0 {
                indegree[into] -= 1;
                if indegree[into] == 0 {
                    queue.push(into);
                }
            }
        }
    }
    if order.len() == m {
        return Ok(LinearSupportRoute::Acyclic { order });
    }
    let blocks = support_blocks(model, &free)?;
    // Pick a deterministic edge inside a nontrivial strongly connected block.
    let mut component = vec![usize::MAX; m];
    for (id, block) in blocks.iter().enumerate().filter(|(_, block)| block.len() > 1) {
        for node in block {
            let local = free.binary_search(node).map_err(|_| SccCompileError::Shape {
                message: "condensation member is outside the free set",
            })?;
            component[local] = id;
        }
    }
    for from in 0..m {
        for into in 0..m {
            if from != into
                && indegree[from] != 0
                && indegree[into] != 0
                && structural_coefficient(model, free[into], free[from])? != 0.0
                && component[from] != usize::MAX
                && component[from] == component[into]
            {
                let error = antecedent_graph::error::GraphError::Cycle {
                    from: u32::try_from(from).map_err(|_| SccCompileError::Shape {
                        message: "free block does not fit in a dense id",
                    })?,
                    to: u32::try_from(into).map_err(|_| SccCompileError::Shape {
                        message: "free block does not fit in a dense id",
                    })?,
                };
                return Ok(LinearSupportRoute::Cyclic { next_attempt: error.to_string(), blocks });
            }
        }
    }
    Ok(LinearSupportRoute::Cyclic { next_attempt: dag_support_refusal(), blocks })
}

pub(crate) fn dag_support_refusal() -> String {
    antecedent_graph::error::CyclicDagRefusal::unsupported(
        antecedent_graph::error::DagOperation::Simulation,
        "CompiledCausalModel",
        "a directed cycle belongs in a CyclicDigraph; condensation describes the feedback and \
         does not sample or abduct. This compiler has no cyclic support",
    )
    .to_string()
}

/// Compile a Gaussian equilibrium, using the DAG algorithm when the free support is acyclic.
///
/// A directed cycle is refused by that algorithm. The cyclic support then runs
/// the existing strongly connected compiler. Specified models remain
/// model-conditional; compilation grants no scientific support.
///
/// # Errors
///
/// The errors of [`compile_gaussian_equilibrium`]. A directed cycle is compiled
/// by the cyclic support after the DAG refusal is recorded on the route.
pub fn compile_routed_gaussian_equilibrium(
    component: &CyclicComponent,
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    clamp: &HardClamp,
    clamp_values: &[f64],
    budget: CompileBudget,
) -> Result<RoutedEquilibrium, SccCompileError> {
    compile_routed_with(component, model, disturbance, (clamp, clamp_values), budget, None)
}

/// [`compile_routed_gaussian_equilibrium`] that takes the support route from `routes` when it
/// holds one for exactly this map and clamp set.
///
/// A route taken from `routes` was charged when it was computed, so the compile records and
/// admits only the factorization work. A set built for other coefficients is never consulted:
/// the route is computed and charged here.
pub(crate) fn compile_routed_with(
    component: &CyclicComponent,
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    (clamp, clamp_values): (&HardClamp, &[f64]),
    budget: CompileBudget,
    routes: Option<&LinearRouteSet>,
) -> Result<RoutedEquilibrium, SccCompileError> {
    let n = model.dimension();
    let binding = component.model_binding();
    if binding.dimension() != n {
        return Err(SccCompileError::ComponentDimension {
            component: binding.dimension(),
            model: n,
        });
    }
    if binding != ModelBinding::of(model) {
        return Err(SccCompileError::ComponentModelMismatch {
            component: component.id().to_owned(),
        });
    }
    reject_compile_inputs(model, disturbance, clamp, clamp_values)?;
    // Refuse an impossible budget before allocating or scanning the support graph.
    let n_u64 = u64::try_from(n)
        .map_err(|_| SccCompileError::Shape { message: "model dimension does not fit in u64" })?;
    let shared = routes.and_then(|set| set.get(model, clamp));
    let route_work = if shared.is_some() { 0 } else { route_work(n_u64)? };
    let bytes = plan_bytes(n_u64)?;
    if route_work > budget.max_work {
        return Err(SccCompileError::Budget {
            resource: "work",
            requested: route_work,
            limit: budget.max_work,
        });
    }
    if bytes > budget.max_bytes {
        return Err(SccCompileError::Budget {
            resource: "bytes",
            requested: bytes,
            limit: budget.max_bytes,
        });
    }
    let route = match shared {
        Some(route) => route.clone(),
        None => route_linear_support(model, clamp)?,
    };
    let equilibrium = match &route {
        LinearSupportRoute::Acyclic { order } => {
            let blocks: Vec<Vec<usize>> = order.iter().copied().map(|node| vec![node]).collect();
            finish_compile(
                component.id(),
                model,
                disturbance,
                (clamp, clamp_values),
                budget,
                (&blocks, route_work),
                DAG_EQUILIBRIUM_ATTEMPT,
            )?
        }
        LinearSupportRoute::Cyclic { blocks, .. } => finish_compile(
            component.id(),
            model,
            disturbance,
            (clamp, clamp_values),
            budget,
            (blocks, route_work),
            SCC_COMPILE_ATTEMPT,
        )?,
    };
    Ok(RoutedEquilibrium { route, equilibrium })
}
