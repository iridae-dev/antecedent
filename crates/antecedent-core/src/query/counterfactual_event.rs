//! A graph-independent counterfactual event query: a conjunction of events,
//! each a variable at a level in one world, conditioned on another conjunction.
//!
//! The worlds are the cross-world query's [`WorldSpec`]s: world `w` sets its hard
//! interventions, and an event `(w, V, v)` reads "`V` would be `v` in world `w`",
//! i.e. `V_{x_w} = v`. The conditioning events are read the same way; an event in
//! a world without interventions is an observation. The coupling names how the
//! worlds share exogenous terms: for identification from an observational law it
//! is [`ExogenousCoupling::SharedLatentExogenous`] (the terms are shared and
//! marginalized, never abducted).
//!
//! The query carries no graph and no law; the identification route decides which
//! shapes it answers. The effect of treatment on the treated,
//! `P(Y_x = y | X = x')`, is [`CounterfactualEventQuery::effect_on_treated`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::cross_world::{ExogenousCoupling, MAX_CROSS_WORLDS, WorldId, WorldSpec};
use super::error::QueryError;
use crate::ids::VariableId;

/// `variable` takes `level` in `world`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CounterfactualEvent {
    world: WorldId,
    variable: VariableId,
    level_bits: u64,
}

impl CounterfactualEvent {
    /// The event `variable = level` in `world`.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidIntervention`] for a non-finite level.
    pub fn new(world: WorldId, variable: VariableId, level: f64) -> Result<Self, QueryError> {
        if !level.is_finite() {
            return Err(QueryError::InvalidIntervention(
                "a counterfactual event level must be finite".into(),
            ));
        }
        // One level, one spelling: -0.0 and 0.0 are the same level.
        let level = level + 0.0;
        Ok(Self { world, variable, level_bits: level.to_bits() })
    }

    /// World the event is read in.
    #[must_use]
    pub const fn world(self) -> WorldId {
        self.world
    }

    /// Variable.
    #[must_use]
    pub const fn variable(self) -> VariableId {
        self.variable
    }

    /// Level.
    #[must_use]
    pub const fn level(self) -> f64 {
        f64::from_bits(self.level_bits)
    }

    /// IEEE bits of the level.
    #[must_use]
    pub const fn level_bits(self) -> u64 {
        self.level_bits
    }
}

/// `P(event | given)` over worlds that share exogenous terms.
///
/// Both conjunctions are canonical (sorted, exact duplicates removed), so equal
/// queries compare, render and digest equal however they were built.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CounterfactualEventQuery {
    worlds: Vec<WorldSpec>,
    coupling: ExogenousCoupling,
    event: Vec<CounterfactualEvent>,
    given: Vec<CounterfactualEvent>,
}

impl CounterfactualEventQuery {
    /// Build a query from explicit worlds and conjunctions.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidIntervention`] when no or more than
    /// [`MAX_CROSS_WORLDS`] worlds are given, an event names an unknown world, or
    /// the event conjunction is empty.
    pub fn new(
        worlds: Vec<WorldSpec>,
        coupling: ExogenousCoupling,
        event: impl IntoIterator<Item = CounterfactualEvent>,
        given: impl IntoIterator<Item = CounterfactualEvent>,
    ) -> Result<Self, QueryError> {
        let canonical = |atoms: &mut Vec<CounterfactualEvent>| {
            atoms.sort_unstable();
            atoms.dedup();
        };
        let mut event: Vec<_> = event.into_iter().collect();
        let mut given: Vec<_> = given.into_iter().collect();
        canonical(&mut event);
        canonical(&mut given);
        let query = Self { worlds, coupling, event, given };
        query.validate()?;
        Ok(query)
    }

    /// Validate structure that does not depend on any graph.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn validate(&self) -> Result<(), QueryError> {
        if !(1..=MAX_CROSS_WORLDS).contains(&self.worlds.len()) {
            return Err(QueryError::InvalidIntervention(
                "a counterfactual event query names between one and eight worlds".into(),
            ));
        }
        if self.event.is_empty() {
            return Err(QueryError::InvalidIntervention(
                "a counterfactual event query needs at least one event".into(),
            ));
        }
        if self.event.iter().chain(&self.given).any(|a| a.world.index() >= self.worlds.len()) {
            return Err(QueryError::InvalidIntervention(
                "a counterfactual event names an unknown world".into(),
            ));
        }
        Ok(())
    }

    /// The effect of treatment on the treated, `P(Y_x = y | X = x')`: world 0
    /// is the natural world (no intervention) where `treatment = observed` is
    /// conditioned on, world 1 is `do(treatment = active)` where
    /// `outcome = outcome_level` is the event.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidIntervention`] for non-finite levels, `treatment`
    /// equal to `outcome`, or `active` equal to `observed` (that is the observed
    /// conditional, not a counterfactual).
    pub fn effect_on_treated(
        treatment: VariableId,
        active: f64,
        observed: f64,
        outcome: VariableId,
        outcome_level: f64,
    ) -> Result<Self, QueryError> {
        if treatment == outcome {
            return Err(QueryError::InvalidIntervention(
                "treatment and outcome must be distinct".into(),
            ));
        }
        if !active.is_finite()
            || !observed.is_finite()
            || (active + 0.0).to_bits() == (observed + 0.0).to_bits()
        {
            return Err(QueryError::InvalidIntervention(
                "the counterfactual and the observed treatment levels must be finite and distinct"
                    .into(),
            ));
        }
        let natural = WorldId::new(0);
        let treated = WorldId::new(1);
        Self::new(
            vec![WorldSpec::new([], [])?, WorldSpec::new([(treatment, active)], [])?],
            ExogenousCoupling::SharedLatentExogenous,
            [CounterfactualEvent::new(treated, outcome, outcome_level)?],
            [CounterfactualEvent::new(natural, treatment, observed)?],
        )
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

    /// The event conjunction, canonical.
    #[must_use]
    pub fn event(&self) -> &[CounterfactualEvent] {
        &self.event
    }

    /// The conditioning conjunction, canonical (empty for an unconditional query).
    #[must_use]
    pub fn given(&self) -> &[CounterfactualEvent] {
        &self.given
    }

    /// Canonical text: equal queries render equal. It names every world with its
    /// interventions and routes (as IEEE bits), the coupling and both
    /// conjunctions. This is the digest input, not a display format.
    #[must_use]
    pub fn canonical_text(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::from("counterfactual_event_query_v1;coupling=");
        out.push_str(self.coupling.tag());
        for (index, world) in self.worlds.iter().enumerate() {
            let _ = write!(out, ";world{index}{{do=[");
            for (variable, value) in world.interventions() {
                let _ = write!(out, "{}:{:016x},", variable.raw(), value.to_bits());
            }
            out.push_str("];reads=[");
            for route in world.routes() {
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
        for (label, atoms) in [("event", &self.event), ("given", &self.given)] {
            let _ = write!(out, ";{label}=[");
            for atom in atoms {
                let _ = write!(
                    out,
                    "w{}.{}:{:016x},",
                    atom.world.index(),
                    atom.variable.raw(),
                    atom.level_bits
                );
            }
            out.push(']');
        }
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
    fn conjunctions_are_canonical_and_order_free() {
        let worlds =
            || vec![WorldSpec::new([], []).unwrap(), WorldSpec::new([(v(0), 1.0)], []).unwrap()];
        let a = CounterfactualEvent::new(WorldId::new(1), v(2), 1.0).unwrap();
        let b = CounterfactualEvent::new(WorldId::new(0), v(1), 0.0).unwrap();
        let x = CounterfactualEvent::new(WorldId::new(0), v(0), -0.0).unwrap();
        let one = CounterfactualEventQuery::new(
            worlds(),
            ExogenousCoupling::SharedLatentExogenous,
            [a, b, a],
            [x],
        )
        .unwrap();
        let two = CounterfactualEventQuery::new(
            worlds(),
            ExogenousCoupling::SharedLatentExogenous,
            [b, a],
            [x],
        )
        .unwrap();
        assert_eq!(one, two);
        assert_eq!(one.canonical_text(), two.canonical_text());
        assert_eq!(one.event().len(), 2);
        // -0.0 and 0.0 are one level.
        assert_eq!(x.level_bits(), 0.0f64.to_bits());
        let ett = CounterfactualEventQuery::effect_on_treated(v(0), 1.0, 0.0, v(2), 1.0).unwrap();
        assert_eq!(ett.coupling(), ExogenousCoupling::SharedLatentExogenous);
        assert!(ett.canonical_text().contains("coupling=shared_latent_exogenous"));
        assert_ne!(ett.canonical_text(), one.canonical_text());
    }

    #[test]
    fn malformed_queries_are_rejected() {
        assert!(CounterfactualEventQuery::effect_on_treated(v(0), 1.0, 1.0, v(2), 1.0).is_err());
        assert!(CounterfactualEventQuery::effect_on_treated(v(0), 1.0, 0.0, v(0), 1.0).is_err());
        assert!(CounterfactualEvent::new(WorldId::new(0), v(0), f64::NAN).is_err());
        let far = CounterfactualEvent::new(WorldId::new(3), v(0), 0.0).unwrap();
        let worlds = vec![WorldSpec::new([], []).unwrap()];
        assert!(
            CounterfactualEventQuery::new(
                worlds.clone(),
                ExogenousCoupling::SharedLatentExogenous,
                [far],
                []
            )
            .is_err()
        );
        assert!(
            CounterfactualEventQuery::new(worlds, ExogenousCoupling::SharedLatentExogenous, [], [])
                .is_err()
        );
        assert!(
            CounterfactualEventQuery::new(
                Vec::new(),
                ExogenousCoupling::SharedLatentExogenous,
                [CounterfactualEvent::new(WorldId::new(0), v(0), 0.0).unwrap()],
                []
            )
            .is_err()
        );
    }
}
