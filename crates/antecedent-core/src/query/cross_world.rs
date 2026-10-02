//! A graph-independent cross-world query: worlds, per-edge inputs, coupling.
//!
//! A cross-world query is a typed value, not parameters of one estimator. It
//! names the worlds (each with its hard interventions and its edge routes), the
//! variables observed in each world, and how the exogenous terms of the worlds
//! are coupled. It carries no graph: a checker decides whether a graph licenses
//! it, and an executor evaluates it on a fitted structural model.
//!
//! An *edge route* is the per-edge input assignment of an edge intervention: in
//! a world, the child's mechanism reads the parent's value from the named source
//! world instead of from its own world. Every edge without a route reads its own
//! world. A path-specific effect, the natural direct effect and the natural
//! indirect effect are all special cases; none needs a separate query type.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::error::QueryError;
use crate::ids::VariableId;

/// Most worlds one query may declare.
pub const MAX_CROSS_WORLDS: usize = 8;

/// Index of a world within one query.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WorldId(u8);

impl WorldId {
    /// World from its index.
    #[must_use]
    pub const fn new(index: u8) -> Self {
        Self(index)
    }

    /// The world's index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// One edge whose child reads its parent from another world.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeRoute {
    /// Parent whose value is read.
    pub parent: VariableId,
    /// Child whose mechanism reads it.
    pub child: VariableId,
    /// World the parent's value is taken from.
    pub source: WorldId,
}

/// One world: hard interventions and the edges that read other worlds.
///
/// Both lists are canonical (sorted, no duplicates), so equal worlds compare and
/// digest equal however they were built.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct WorldSpec {
    interventions: Vec<(VariableId, u64)>,
    routes: Vec<EdgeRoute>,
}

impl WorldSpec {
    /// Build a world from hard `do(variable = value)` settings and edge routes.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidIntervention`] for a non-finite value, a variable
    /// intervened twice, a self-edge, or an edge routed twice.
    pub fn new(
        interventions: impl IntoIterator<Item = (VariableId, f64)>,
        routes: impl IntoIterator<Item = EdgeRoute>,
    ) -> Result<Self, QueryError> {
        let mut interventions: Vec<(VariableId, u64)> = interventions
            .into_iter()
            .map(|(variable, value)| {
                if value.is_finite() {
                    // One level, one spelling: -0.0 and 0.0 are the same value.
                    Ok((variable, (value + 0.0).to_bits()))
                } else {
                    Err(QueryError::InvalidIntervention(
                        "cross-world intervention values must be finite".into(),
                    ))
                }
            })
            .collect::<Result<_, _>>()?;
        interventions.sort_unstable();
        if interventions.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err(QueryError::InvalidIntervention(
                "a variable is intervened twice in one world".into(),
            ));
        }
        let mut routes: Vec<EdgeRoute> = routes.into_iter().collect();
        routes.sort_unstable();
        if routes.iter().any(|r| r.parent == r.child) {
            return Err(QueryError::InvalidIntervention("an edge route is a self-edge".into()));
        }
        if routes.windows(2).any(|w| (w[0].parent, w[0].child) == (w[1].parent, w[1].child)) {
            return Err(QueryError::InvalidIntervention("an edge is routed twice".into()));
        }
        Ok(Self { interventions, routes })
    }

    /// Hard interventions as `(variable, value)`, in canonical order.
    pub fn interventions(&self) -> impl Iterator<Item = (VariableId, f64)> + '_ {
        self.interventions.iter().map(|&(v, bits)| (v, f64::from_bits(bits)))
    }

    /// Edge routes in canonical order.
    #[must_use]
    pub fn routes(&self) -> &[EdgeRoute] {
        &self.routes
    }

    /// The hard value set for `variable`, if any.
    #[must_use]
    pub fn intervention_of(&self, variable: VariableId) -> Option<f64> {
        self.interventions.iter().find(|(v, _)| *v == variable).map(|&(_, b)| f64::from_bits(b))
    }
}

/// How the exogenous terms of the worlds are coupled.
///
/// The path-specific edge contrast (2.2A) uses [`Self::SharedAbducedExogenous`];
/// counterfactual identification from an observational law on a latent-variable
/// graph (2.2B) uses [`Self::SharedLatentExogenous`]. A route refuses a coupling
/// its contract does not name.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ExogenousCoupling {
    /// One abduced exogenous term per variable and unit, shared by every world.
    SharedAbducedExogenous,
    /// The exogenous terms of the semi-Markovian model (one per variable and one
    /// per bidirected edge) are shared by every world and marginalized over their
    /// unknown law, never abducted per unit.
    SharedLatentExogenous,
}

impl ExogenousCoupling {
    /// Stable tag for digests and artifacts.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::SharedAbducedExogenous => "shared_abduced_exogenous",
            Self::SharedLatentExogenous => "shared_latent_exogenous",
        }
    }
}

/// A variable read in one world.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WorldObservation {
    /// World.
    pub world: WorldId,
    /// Variable.
    pub variable: VariableId,
}

/// A cross-world contrast `E[observed(plus)] - E[observed(minus)]` over worlds
/// that share exogenous terms.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CrossWorldQuery {
    worlds: Vec<WorldSpec>,
    coupling: ExogenousCoupling,
    plus: WorldObservation,
    minus: WorldObservation,
}

impl CrossWorldQuery {
    /// Build a query from explicit worlds.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidIntervention`] when fewer than two or more than
    /// [`MAX_CROSS_WORLDS`] worlds are given, a route or observation names an
    /// unknown world, a route reads its own world, or the two observations are
    /// the same.
    pub fn new(
        worlds: Vec<WorldSpec>,
        coupling: ExogenousCoupling,
        plus: WorldObservation,
        minus: WorldObservation,
    ) -> Result<Self, QueryError> {
        let query = Self { worlds, coupling, plus, minus };
        query.validate()?;
        Ok(query)
    }

    /// Validate structure that does not depend on any graph.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn validate(&self) -> Result<(), QueryError> {
        let n = self.worlds.len();
        if !(2..=MAX_CROSS_WORLDS).contains(&n) {
            return Err(QueryError::InvalidIntervention(
                "a cross-world query names between two and eight worlds".into(),
            ));
        }
        for (index, world) in self.worlds.iter().enumerate() {
            for route in &world.routes {
                if route.source.index() >= n || route.source.index() == index {
                    return Err(QueryError::InvalidIntervention(
                        "an edge route reads an unknown world or its own world".into(),
                    ));
                }
            }
        }
        for observation in [self.plus, self.minus] {
            if observation.world.index() >= n {
                return Err(QueryError::InvalidIntervention(
                    "an observation names an unknown world".into(),
                ));
            }
        }
        if self.plus == self.minus {
            return Err(QueryError::InvalidIntervention(
                "a cross-world contrast needs two distinct observations".into(),
            ));
        }
        Ok(())
    }

    /// Worlds in declaration order.
    #[must_use]
    pub fn worlds(&self) -> &[WorldSpec] {
        &self.worlds
    }

    /// Exogenous coupling.
    #[must_use]
    pub const fn coupling(&self) -> ExogenousCoupling {
        self.coupling
    }

    /// The added observation.
    #[must_use]
    pub const fn plus(&self) -> WorldObservation {
        self.plus
    }

    /// The subtracted observation.
    #[must_use]
    pub const fn minus(&self) -> WorldObservation {
        self.minus
    }

    /// The path-specific edge-intervention contrast on a treatment `x` and an
    /// outcome `y`.
    ///
    /// World 0 is the baseline `do(x = control)`, every edge reading its own
    /// world. World 1 is `do(x = active)`; each graph edge that is *not* in
    /// `intervened_edges` reads its parent from world 0 (it "sees the baseline
    /// value"), and each edge in it reads world 1 (it "sees the intervened
    /// value"). The contrast is `y` in world 1 minus `y` in world 0, both under
    /// the same exogenous terms.
    ///
    /// `graph_edges` is the graph's directed edge list; the graph does not
    /// otherwise enter the query.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidIntervention`] for non-finite or equal levels, `x`
    /// equal to `y`, or an intervened edge that is not a graph edge.
    pub fn path_specific(
        treatment: VariableId,
        outcome: VariableId,
        control: f64,
        active: f64,
        graph_edges: &[(VariableId, VariableId)],
        intervened_edges: &[(VariableId, VariableId)],
    ) -> Result<Self, QueryError> {
        if treatment == outcome {
            return Err(QueryError::InvalidIntervention(
                "treatment and outcome must be distinct".into(),
            ));
        }
        if !control.is_finite()
            || !active.is_finite()
            || (control + 0.0).to_bits() == (active + 0.0).to_bits()
        {
            return Err(QueryError::InvalidIntervention(
                "treatment levels must be finite and distinct".into(),
            ));
        }
        if let Some(edge) = intervened_edges.iter().find(|e| !graph_edges.contains(e)) {
            return Err(QueryError::InvalidIntervention(format!(
                "intervened edge {} -> {} is not a graph edge",
                edge.0, edge.1
            )));
        }
        let baseline = WorldId::new(0);
        let routes = graph_edges
            .iter()
            .filter(|edge| !intervened_edges.contains(edge))
            .map(|&(parent, child)| EdgeRoute { parent, child, source: baseline });
        Self::new(
            vec![
                WorldSpec::new([(treatment, control)], [])?,
                WorldSpec::new([(treatment, active)], routes)?,
            ],
            ExogenousCoupling::SharedAbducedExogenous,
            WorldObservation { world: WorldId::new(1), variable: outcome },
            WorldObservation { world: baseline, variable: outcome },
        )
    }

    /// The natural direct effect `Y(active, M(control)) - Y(control, M(control))`
    /// as the edge contrast whose only intervened edge is `treatment -> outcome`.
    ///
    /// # Errors
    ///
    /// See [`Self::path_specific`].
    pub fn natural_direct(
        treatment: VariableId,
        mediator: VariableId,
        outcome: VariableId,
        control: f64,
        active: f64,
    ) -> Result<Self, QueryError> {
        Self::path_specific(
            treatment,
            outcome,
            control,
            active,
            &[(treatment, mediator), (treatment, outcome), (mediator, outcome)],
            &[(treatment, outcome)],
        )
    }

    /// The natural indirect effect `Y(control, M(active)) - Y(control, M(control))`
    /// as the edge contrast intervening on `treatment -> mediator -> outcome`.
    ///
    /// # Errors
    ///
    /// See [`Self::path_specific`].
    pub fn natural_indirect(
        treatment: VariableId,
        mediator: VariableId,
        outcome: VariableId,
        control: f64,
        active: f64,
    ) -> Result<Self, QueryError> {
        Self::path_specific(
            treatment,
            outcome,
            control,
            active,
            &[(treatment, mediator), (treatment, outcome), (mediator, outcome)],
            &[(treatment, mediator), (mediator, outcome)],
        )
    }

    /// Canonical text: equal queries render equal, and it names every world,
    /// intervention level (as IEEE bits), edge route, the coupling and both
    /// observations. This is the digest input, not a display format.
    #[must_use]
    pub fn canonical_text(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::from("cross_world_query_v1;coupling=");
        out.push_str(self.coupling.tag());
        for (index, world) in self.worlds.iter().enumerate() {
            let _ = write!(out, ";world{index}{{do=[");
            for (variable, bits) in &world.interventions {
                let _ = write!(out, "{}:{bits:016x},", variable.raw());
            }
            out.push_str("];reads=[");
            for route in &world.routes {
                let _ = write!(
                    out,
                    "{}>{}@{},",
                    route.parent.raw(),
                    route.child.raw(),
                    route.source.index()
                );
            }
            out.push_str("]}");
        }
        let _ = write!(
            out,
            ";contrast=world{}.{}-world{}.{}",
            self.plus.world.index(),
            self.plus.variable.raw(),
            self.minus.world.index(),
            self.minus.variable.raw(),
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    #[test]
    fn edge_sets_are_canonical_and_order_free() {
        let edges = [(v(0), v(1)), (v(0), v(2)), (v(1), v(2))];
        let a = CrossWorldQuery::path_specific(
            v(0),
            v(2),
            0.0,
            1.0,
            &edges,
            &[(v(0), v(1)), (v(1), v(2))],
        )
        .unwrap();
        let b = CrossWorldQuery::path_specific(
            v(0),
            v(2),
            0.0,
            1.0,
            &[edges[2], edges[0], edges[1]],
            &[(v(1), v(2)), (v(0), v(1))],
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.canonical_text(), b.canonical_text());
        let direct = CrossWorldQuery::natural_direct(v(0), v(1), v(2), 0.0, 1.0).unwrap();
        assert_ne!(a.canonical_text(), direct.canonical_text());
    }

    #[test]
    fn malformed_queries_are_rejected() {
        let baseline = WorldSpec::new([(v(0), 0.0)], []).unwrap();
        let own = WorldSpec::new(
            [(v(0), 1.0)],
            [EdgeRoute { parent: v(0), child: v(1), source: WorldId::new(1) }],
        )
        .unwrap();
        let plus = WorldObservation { world: WorldId::new(1), variable: v(2) };
        let minus = WorldObservation { world: WorldId::new(0), variable: v(2) };
        let coupling = ExogenousCoupling::SharedAbducedExogenous;
        // A route reading its own world.
        assert!(CrossWorldQuery::new(vec![baseline.clone(), own], coupling, plus, minus).is_err());
        // Fewer than two worlds, an unknown observed world, identical observations.
        assert!(CrossWorldQuery::new(vec![baseline.clone()], coupling, plus, minus).is_err());
        assert!(
            CrossWorldQuery::new(vec![baseline.clone(), baseline.clone()], coupling, plus, plus)
                .is_err()
        );
        let far = WorldObservation { world: WorldId::new(5), variable: v(2) };
        assert!(
            CrossWorldQuery::new(vec![baseline.clone(), baseline], coupling, far, minus).is_err()
        );
        // A non-edge cannot be intervened.
        assert!(
            CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &[(v(0), v(2))], &[(v(1), v(2))])
                .is_err()
        );
        assert!(WorldSpec::new([(v(0), f64::NAN)], []).is_err());
        // -0.0 and 0.0 are one level: not two distinct treatment levels, and one
        // canonical spelling in the world.
        assert!(
            CrossWorldQuery::path_specific(v(0), v(2), -0.0, 0.0, &[(v(0), v(2))], &[]).is_err()
        );
        assert_eq!(
            WorldSpec::new([(v(0), -0.0)], []).unwrap(),
            WorldSpec::new([(v(0), 0.0)], []).unwrap()
        );
    }
}
