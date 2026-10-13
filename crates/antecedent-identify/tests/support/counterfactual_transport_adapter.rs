//! Adapter from the finite-SCM transport fixtures to the typed inputs of the
//! counterfactual transport route.
//!
//! The fixtures (`counterfactual_transport_cases`) name variables and
//! populations by string and state queries as nested counterfactual terms of the
//! independent oracle. This module turns one case into a
//! [`CounterfactualTransportProblem`], a population-labelled
//! [`EvidenceCatalog`], source declarations, a typed
//! [`CounterfactualTransportQuery`] (nested interventions become edge routes)
//! and the exact finite laws the catalog describes. It contains no
//! identification logic.
//!
//! Included with `#[path]` next to the oracle and the cases at a test crate
//! root; it reaches both as `crate::finite_scm_oracle` and
//! `crate::counterfactual_transport_cases`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    dead_code,
    missing_docs,
    clippy::too_many_lines,
    reason = "test adapter shared by several test binaries; each uses a subset"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    CounterfactualEvent, CounterfactualEventQuery, DistributionAvailability, EdgeRoute,
    Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExogenousCoupling,
    InterventionAssignment, RegimeBinding, RegimeId, RegimeKind, Value, VariableId, WorldId,
    WorldSpec,
};
use antecedent_expr::{
    DiscreteAxis, ExactDiscreteLaw, ExactTransportData, InterventionAssignment as LawIntervention,
    LawTolerance,
};
use antecedent_identify::{
    CounterfactualTransportProblem, CounterfactualTransportQuery, CounterfactualTransportSource,
};

use crate::counterfactual_transport_cases::Case;
use crate::finite_scm_oracle::{Atom, CfVar, EvidenceSpec, Setting, TargetQuery};

/// Variable id of a case variable: its position in the case's observed list.
pub fn variable_id(case: &Case, name: &str) -> VariableId {
    let index = case
        .graph
        .observed
        .iter()
        .position(|n| *n == name)
        .unwrap_or_else(|| panic!("case {} has no variable {name}", case.name));
    VariableId::from_raw(u32::try_from(index).expect("small index"))
}

/// Declared levels of every variable, from the target population's model.
pub fn declared_levels(case: &Case) -> Vec<(VariableId, Vec<f64>)> {
    let scm = &crate::finite_scm_oracle::population(&case.populations, case.target_population);
    case.graph
        .observed
        .iter()
        .map(|name| {
            let levels = scm.levels[scm.endo_index(name)];
            (variable_id(case, name), (0..levels).map(|l| l as f64).collect())
        })
        .collect()
}

pub fn problem(case: &Case) -> CounterfactualTransportProblem {
    let variables = declared_levels(case);
    let directed: Vec<(VariableId, VariableId)> = case
        .graph
        .directed
        .iter()
        .map(|(a, b)| (variable_id(case, a), variable_id(case, b)))
        .collect();
    let bidirected: Vec<(VariableId, VariableId)> = case
        .graph
        .bidirected
        .iter()
        .map(|(a, b)| (variable_id(case, a), variable_id(case, b)))
        .collect();
    CounterfactualTransportProblem::new(&variables, &directed, &bidirected)
        .expect("a case graph is inside the contract")
}

/// Source populations: every population but the target. Each flags the case's
/// selection targets and may experiment on `controllable`.
pub fn sources(case: &Case, controllable: &[&str]) -> Vec<CounterfactualTransportSource> {
    let selection: Vec<VariableId> =
        case.graph.selection.iter().map(|(_, v)| variable_id(case, v)).collect();
    let control: Vec<VariableId> = controllable.iter().map(|v| variable_id(case, v)).collect();
    case.populations
        .iter()
        .filter(|p| p.name != case.target_population)
        .map(|p| CounterfactualTransportSource {
            population: Arc::from(p.name.as_str()),
            selection_targets: selection.clone().into(),
            controllable: control.clone().into(),
        })
        .collect()
}

/// The variables a source experiments on in the supplied evidence, for the
/// default declaration of a case.
pub fn evidence_controllable(case: &Case) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for spec in &case.evidence {
        if spec.population == case.target_population {
            continue;
        }
        for (name, _) in &spec.regime {
            if let Some(known) = case.graph.observed.iter().find(|n| **n == name.as_str()) {
                if !out.contains(known) {
                    out.push(*known);
                }
            }
        }
    }
    out
}

/// One catalog regime of a case: a population and an intervention variable set.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegimeKey {
    pub population: String,
    pub intervened: Vec<String>,
}

/// The catalog of a case: one regime per `(population, intervened variables)`.
pub struct AdaptedCatalog {
    pub catalog: EvidenceCatalog,
    pub regime_of: BTreeMap<RegimeKey, RegimeId>,
}

fn regime_key(spec: &EvidenceSpec) -> RegimeKey {
    let mut intervened: Vec<String> = spec.regime.iter().map(|(n, _)| n.clone()).collect();
    intervened.sort();
    RegimeKey { population: spec.population.clone(), intervened }
}

pub fn catalog(case: &Case) -> AdaptedCatalog {
    catalog_of(case, &case.evidence)
}

pub fn catalog_of(case: &Case, specs: &[EvidenceSpec]) -> AdaptedCatalog {
    let mut regime_of: BTreeMap<RegimeKey, RegimeId> = BTreeMap::new();
    let mut regimes: Vec<EvidenceRegime> = Vec::new();
    for spec in specs {
        let key = regime_key(spec);
        if regime_of.contains_key(&key) {
            continue;
        }
        let id = RegimeId::from_raw(u32::try_from(regimes.len()).expect("small index"));
        regime_of.insert(key.clone(), id);
        let interventions: Vec<VariableId> =
            key.intervened.iter().map(|n| variable_id(case, n)).collect();
        let measured: Vec<VariableId> = spec
            .variables
            .iter()
            .filter(|n| !key.intervened.contains(n))
            .map(|n| variable_id(case, n))
            .collect();
        let kind = if interventions.is_empty() {
            RegimeKind::Observational
        } else {
            RegimeKind::Experimental
        };
        regimes.push(
            EvidenceRegime::try_new(
                id,
                kind,
                EvidenceKind::Available,
                interventions,
                Vec::<InterventionAssignment>::new(),
                measured,
                key.population.as_str(),
                DistributionAvailability::Joint,
            )
            .expect("a case regime is well formed"),
        );
    }
    let catalog = EvidenceCatalog::try_new(
        Vec::<Environment>::new(),
        regimes,
        Vec::<RegimeBinding>::new(),
        None,
    )
    .expect("a case catalog is well formed");
    AdaptedCatalog { catalog, regime_of }
}

struct Worlds<'a> {
    case: &'a Case,
    /// `(fixed levels, routes (parent, child) -> world)` per world.
    worlds: Vec<(BTreeMap<String, usize>, BTreeMap<(String, String), usize>)>,
}

impl Worlds<'_> {
    fn world_for(&mut self, cf: &CfVar) -> usize {
        let mut fixed: BTreeMap<String, usize> = BTreeMap::new();
        let mut routes: BTreeMap<(String, String), usize> = BTreeMap::new();
        for (name, setting) in &cf.interventions {
            match setting {
                Setting::Level(level) => {
                    fixed.insert(name.clone(), *level);
                }
                Setting::Nested(inner) => {
                    // `V` is set to its value in the inner world: every child of `V`
                    // reads it from there.
                    let source = self.world_for(inner);
                    for (parent, child) in &self.case.graph.directed {
                        if *parent == name.as_str() {
                            routes.insert(((*parent).to_owned(), (*child).to_owned()), source);
                        }
                    }
                }
            }
        }
        let entry = (fixed, routes);
        if let Some(position) = self.worlds.iter().position(|w| *w == entry) {
            position
        } else {
            self.worlds.push(entry);
            self.worlds.len() - 1
        }
    }
}

/// The typed query of an oracle target in the case's target population.
pub fn query(case: &Case, target: &TargetQuery, sources: Vec<CounterfactualTransportSource>)
    -> CounterfactualTransportQuery
{
    let mut builder = Worlds { case, worlds: Vec::new() };
    let mut event: Vec<(usize, &Atom)> = Vec::new();
    let mut given: Vec<(usize, &Atom)> = Vec::new();
    for atom in &target.event {
        let world = builder.world_for(&atom.var);
        event.push((world, atom));
    }
    for atom in &target.evidence {
        let world = builder.world_for(&atom.var);
        given.push((world, atom));
    }
    let worlds: Vec<WorldSpec> = builder
        .worlds
        .iter()
        .map(|(fixed, routes)| {
            WorldSpec::new(
                fixed.iter().map(|(n, l)| (variable_id(case, n), *l as f64)),
                routes.iter().map(|((p, c), w)| EdgeRoute {
                    parent: variable_id(case, p),
                    child: variable_id(case, c),
                    source: WorldId::new(u8::try_from(*w).expect("small world index")),
                }),
            )
            .expect("a case world is well formed")
        })
        .collect();
    let atom = |(world, atom): &(usize, &Atom)| {
        CounterfactualEvent::new(
            WorldId::new(u8::try_from(*world).expect("small world index")),
            variable_id(case, &atom.var.var),
            atom.level as f64,
        )
        .expect("a case atom is finite")
    };
    let event_query = CounterfactualEventQuery::new(
        worlds,
        ExogenousCoupling::SharedLatentExogenous,
        event.iter().map(atom),
        given.iter().map(atom),
    )
    .expect("a case query is well formed");
    CounterfactualTransportQuery {
        event: event_query,
        target: Arc::from(case.target_population),
        sources: sources.into(),
    }
}

/// The exact finite laws the catalog describes, one per evidence spec, in the
/// case's own populations.
pub fn laws(case: &Case, adapted: &AdaptedCatalog) -> ExactTransportData {
    laws_of(case, adapted, &case.evidence)
}

pub fn laws_of(case: &Case, adapted: &AdaptedCatalog, specs: &[EvidenceSpec]) -> ExactTransportData {
    let tables = crate::finite_scm_oracle::build_evidence(&case.populations, specs);
    let mut out: Vec<ExactDiscreteLaw> = Vec::with_capacity(tables.len());
    for (spec, table) in specs.iter().zip(&tables) {
        let key = regime_key(spec);
        let regime = adapted.regime_of[&key];
        let intervened: Vec<&str> = key.intervened.iter().map(String::as_str).collect();
        let keep: Vec<&str> = spec
            .variables
            .iter()
            .map(String::as_str)
            .filter(|n| !intervened.contains(n))
            .collect();
        let projected = table.table.marginal(&keep);
        let axes: Vec<DiscreteAxis> = projected
            .variables
            .iter()
            .zip(&projected.levels)
            .map(|(name, levels)| DiscreteAxis {
                variable: variable_id(case, name),
                values: (0..*levels).map(|l| Value::f64(l as f64)).collect::<Vec<_>>().into(),
            })
            .collect();
        let interventions: Vec<LawIntervention> = spec
            .regime
            .iter()
            .map(|(n, l)| LawIntervention {
                variable: variable_id(case, n),
                value: Value::f64(*l as f64),
            })
            .collect();
        let world: Vec<String> = spec.regime.iter().map(|(n, l)| format!("{n}{l}")).collect();
        let snapshot = format!("{}:{}:{}", spec.population, regime.raw(), world.join("."));
        out.push(
            ExactDiscreteLaw::try_new(
                spec.population.as_str(),
                regime,
                interventions,
                axes,
                projected.probs.clone(),
                snapshot,
                LawTolerance::default(),
            )
            .expect("an enumerated law is a valid exact law"),
        );
    }
    ExactTransportData::try_new(out, 1 << 16).expect("a case's laws form a provider set")
}
