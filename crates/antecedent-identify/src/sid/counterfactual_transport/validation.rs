use super::{
    Arc, BTreeMap, Built, COUPLING_UNSUPPORTED, CounterfactualTransportProblem,
    CounterfactualTransportQuery, EvidenceCatalog, ExogenousCoupling, IdentificationError,
    RegimeId, SELECTED_SAMPLE, VariableId, WorldIdx, build_network, invalid_catalog, invalid_query,
    regime_measured_is_factor,
};

/// A usable catalog regime in internal indices.
#[derive(Clone, Copy, Debug)]
pub(super) struct UsableRegime {
    pub(super) id: RegimeId,
    pub(super) population: usize,
    pub(super) intervened: u32,
    pub(super) measured: u32,
}

/// A query that passed the contract.
pub(super) struct Validated {
    pub(super) query: CounterfactualTransportQuery,
    pub(super) populations: Vec<Arc<str>>,
    pub(super) selected: Vec<u32>,
    pub(super) controllable: Vec<u32>,
    pub(super) worlds: Vec<WorldIdx>,
    pub(super) event: Vec<(usize, usize, usize)>,
    pub(super) given: Vec<(usize, usize, usize)>,
    pub(super) usable: Vec<UsableRegime>,
    pub(super) ignored: Vec<(RegimeId, String)>,
    /// `(population, intervened set)` of supplied population laws the closure
    /// cannot use (a witness must still agree on them).
    pub(super) unused_family: Vec<(usize, u32)>,
}

/// Why a request does not reach the search.
pub(super) enum Rejection {
    Invalid(IdentificationError),
    Unsupported(&'static str, String),
}

impl From<IdentificationError> for Rejection {
    fn from(error: IdentificationError) -> Self {
        Self::Invalid(error)
    }
}

pub(super) fn atoms_map(
    atoms: &[(usize, usize, usize)],
) -> Option<BTreeMap<(usize, usize), usize>> {
    let mut map = BTreeMap::new();
    for &(world, variable, level) in atoms {
        match map.insert((world, variable), level) {
            Some(existing) if existing != level => return None,
            _ => {}
        }
    }
    Some(map)
}

#[allow(
    clippy::too_many_lines,
    reason = "a flat sequence of independent request, catalog and bound validity checks, each with its own typed rejection"
)]
pub(super) fn validate(
    problem: &CounterfactualTransportProblem,
    query: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<Validated, Rejection> {
    let query = query.canonical();
    let event = &query.event;
    if event.coupling() != ExogenousCoupling::SharedLatentExogenous {
        return Err(Rejection::Unsupported(
            COUPLING_UNSUPPORTED,
            "counterfactual transport shares the exogenous terms of the semi-Markovian model \
             across worlds and marginalizes them; an abduced per-unit coupling is a different \
             route"
                .to_owned(),
        ));
    }
    event.validate().map_err(invalid_query)?;
    if query.target.trim().is_empty() {
        return Err(invalid_query("the target population must be named").into());
    }
    let mut populations: Vec<Arc<str>> = vec![Arc::clone(&query.target)];
    for source in query.sources.iter() {
        if populations.iter().any(|p| p.as_ref() == source.population.as_ref()) {
            return Err(invalid_query(
                "source populations must be distinct and none may be the target",
            )
            .into());
        }
        populations.push(Arc::clone(&source.population));
    }
    let mask_of = |variables: &[VariableId], what: &str| -> Result<u32, Rejection> {
        let mut mask = 0u32;
        for v in variables {
            let Some(i) = problem.index_of(*v) else {
                return Err(
                    invalid_query(format!("{what} names unknown variable {}", v.raw())).into()
                );
            };
            mask |= 1 << i;
        }
        Ok(mask)
    };
    let mut selected = vec![0u32];
    let mut controllable = vec![0u32];
    for source in query.sources.iter() {
        selected.push(mask_of(&source.selection_targets, "a selection diagram")?);
        let control = mask_of(&source.controllable, "a controllable set")?;
        controllable.push(control);
    }
    // Worlds.
    let mut worlds: Vec<WorldIdx> = Vec::with_capacity(event.worlds().len());
    for (index, spec) in event.worlds().iter().enumerate() {
        let mut world = WorldIdx::default();
        for (variable, value) in spec.interventions() {
            let Some(i) = problem.index_of(variable) else {
                return Err(invalid_query(format!(
                    "world {index} intervenes on unknown variable {}",
                    variable.raw()
                ))
                .into());
            };
            let Some(level) = problem.level_index(i, value) else {
                return Err(invalid_query(format!(
                    "world {index} sets variable {} to {value}, which is not a declared level",
                    variable.raw()
                ))
                .into());
            };
            world.fixed.insert(i, level);
        }
        for route in spec.routes() {
            let (Some(parent), Some(child)) =
                (problem.index_of(route.parent), problem.index_of(route.child))
            else {
                return Err(
                    invalid_query(format!("world {index} routes an unknown variable")).into()
                );
            };
            if !problem.parents[child].contains(&parent) {
                return Err(invalid_query(format!(
                    "world {index} routes {} -> {}, which is not an edge of the graph",
                    route.parent.raw(),
                    route.child.raw()
                ))
                .into());
            }
            let source = route.source.index();
            if source >= event.worlds().len() || source == index {
                return Err(invalid_query(format!(
                    "world {index} reads a parent from an unknown world or from itself"
                ))
                .into());
            }
            world.routes.insert((parent, child), source);
        }
        worlds.push(world);
    }
    // Atoms.
    let to_atoms = |atoms: &[antecedent_core::CounterfactualEvent]| -> Result<Vec<(usize, usize, usize)>, Rejection> {
        atoms
            .iter()
            .map(|a| -> Result<(usize, usize, usize), Rejection> {
                let Some(i) = problem.index_of(a.variable()) else {
                    return Err(invalid_query(format!(
                        "an event names unknown variable {}",
                        a.variable().raw()
                    ))
                    .into());
                };
                let Some(level) = problem.level_index(i, a.level()) else {
                    return Err(invalid_query(format!(
                        "variable {} is read at {}, which is not a declared level",
                        a.variable().raw(),
                        a.level()
                    ))
                    .into());
                };
                Ok((a.world().index(), i, level))
            })
            .collect()
    };
    let event_atoms = to_atoms(event.event())?;
    let given_atoms = to_atoms(event.given())?;
    if atoms_map(&given_atoms).is_none() {
        return Err(invalid_query(
            "the conditioning event reads one variable of one world at two levels: it has \
             probability zero in every model",
        )
        .into());
    }
    if let Some(given_map) = atoms_map(&given_atoms) {
        if !given_map.is_empty() {
            let impossible = matches!(
                build_network(problem, &worlds, &given_map).map_err(Rejection::Invalid)?,
                Built::Zero
            );
            if impossible {
                return Err(invalid_query(
                    "the conditioning event contradicts the query's own interventions: it has \
                     probability zero in every model",
                )
                .into());
            }
        }
    }
    let (usable, ignored, unused_family) = validate_catalog(problem, catalog, &populations)?;
    Ok(Validated {
        query,
        populations,
        selected,
        controllable,
        worlds,
        event: event_atoms,
        given: given_atoms,
        usable,
        ignored,
        unused_family,
    })
}

type ClassifiedCatalog = (Vec<UsableRegime>, Vec<(RegimeId, String)>, Vec<(usize, u32)>);

/// Classify catalog regimes after the query and conditioning events have been admitted.
fn validate_catalog(
    problem: &CounterfactualTransportProblem,
    catalog: &EvidenceCatalog,
    populations: &[Arc<str>],
) -> Result<ClassifiedCatalog, Rejection> {
    // Catalog.
    catalog.validate().map_err(invalid_catalog)?;
    let mut usable: Vec<UsableRegime> = Vec::new();
    let mut ignored: Vec<(RegimeId, String)> = Vec::new();
    let mut unused_family: Vec<(usize, u32)> = Vec::new();
    for regime in catalog.regimes.iter() {
        if matches!(regime.selection, antecedent_core::SamplingSelection::SelectedOn { .. }) {
            return Err(Rejection::Unsupported(
                SELECTED_SAMPLE,
                format!(
                    "regime {} is the law of a selected subpopulation; this contract has no \
                     model of sampling selection, so it cannot be used or ignored soundly",
                    regime.id.raw()
                ),
            ));
        }
        let Some(population) =
            populations.iter().position(|p| p.as_ref() == regime.population.as_ref())
        else {
            return Err(invalid_catalog(format!(
                "regime {} belongs to population {}, which is neither the target nor a declared \
                 source",
                regime.id.raw(),
                regime.population
            ))
            .into());
        };
        let mut intervened = 0u32;
        let mut measured = 0u32;
        for v in regime.interventions.iter() {
            let Some(i) = problem.index_of(*v) else {
                return Err(invalid_catalog(format!(
                    "regime {} intervenes on unknown variable {}",
                    regime.id.raw(),
                    v.raw()
                ))
                .into());
            };
            intervened |= 1 << i;
        }
        for v in regime.measured.iter() {
            let Some(i) = problem.index_of(*v) else {
                return Err(invalid_catalog(format!(
                    "regime {} measures unknown variable {}",
                    regime.id.raw(),
                    v.raw()
                ))
                .into());
            };
            if (intervened >> i) & 1 == 0 {
                measured |= 1 << i;
            }
        }
        if !regime.supplies_population_law() {
            ignored.push((regime.id, "not a measured population law (no results, a model artifact, or a selected sample)".to_owned()));
            continue;
        }
        let needed = problem.mask_variables(measured);
        let interventions = problem.mask_variables(intervened);
        let need = antecedent_core::FactorNeed {
            population: regime.population.as_ref(),
            variables: &needed,
            conditioned_on: &[],
            interventions: &interventions,
        };
        if measured == 0
            || !regime.satisfies(&need)
            || !regime_measured_is_factor(problem, intervened, measured)
        {
            ignored.push((
                regime.id,
                "not a joint law over a variable set closed under ancestors of the \
                 non-intervened variables with an unrestricted intervention domain"
                    .to_owned(),
            ));
            unused_family.push((population, intervened));
            continue;
        }
        usable.push(UsableRegime { id: regime.id, population, intervened, measured });
    }
    usable.sort_by_key(|u| (u.population, u.intervened, u.measured, u.id));
    ignored.sort_by_key(|(id, _)| *id);
    unused_family.sort_unstable();
    unused_family.dedup();
    Ok((usable, ignored, unused_family))
}
