//! Acceptance and provenance for discovered structures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AcceptedGraph, Arc, CanonicalWriter, CausalError, CausalQuery, ClaimEnvelope, DiscoverQuery,
    DiscoveredStructure, DiscoverySpec, GraphPosterior, MechanismProgram, StructureProvenance,
    StudyResult,
};

fn bounds_label(query: &DiscoverQuery) -> String {
    let bounds = &query.bounds;
    let schedule = bounds.schedule.map_or_else(
        || "none".to_owned(),
        |s| format!("{}x{}+{}/{}", s.chains, s.draws, s.warmup, s.thin),
    );
    format!(
        "alpha={};max_cond_size={};max_lag={};schedule={schedule}",
        bounds.alpha, bounds.max_cond_size, bounds.max_lag
    )
}

/// Canonical digest of a discovery query.
fn query_digest(query: &DiscoverQuery) -> [u8; 32] {
    use antecedent_core::CanonicalEncode;
    let mut writer = CanonicalWriter::new("antecedent.discover_query");
    query.encode(&mut writer);
    writer.finish()
}

/// Accept the structure an executed discovery returned through its review
/// gate, and record where it came from.
///
/// `result` must be the stamped result of a prepared discovery handle and
/// `claim` the claim sealed over it. The accepted graph carries a
/// [`StructureProvenance`]: the method, its declared inference and returned
/// class, the query digest and search bounds, the data snapshot, the lowered
/// program and the claim, and how the review accepted it. A study prepared on
/// the graph carries that record in its contract.
///
/// # Errors
///
/// A result that is not an executed discovery, that no prepared handle
/// stamped, or whose claim seals another execution; and the review-gate
/// refusals of [`DiscoveredStructure::accept`].
pub fn accept_discovered(
    result: &StudyResult,
    claim: &ClaimEnvelope,
    accept_discovered: bool,
) -> Result<AcceptedGraph, CausalError> {
    let payload = result.discovery().ok_or(CausalError::Conflict {
        what: "result",
        detail: "the result is not an executed structure search",
    })?;
    let executed = result.executed_contract.as_ref().ok_or(CausalError::Conflict {
        what: "result",
        detail: "the discovery result carries no execution stamp; accept the result of a \
                 prepared discovery handle",
    })?;
    if executed.identities != claim.identities {
        return Err(CausalError::Conflict {
            what: "claim",
            detail: "the claim seals another execution than this discovery result",
        });
    }
    let program = result.lowered_identity.ok_or(CausalError::Conflict {
        what: "result",
        detail: "the discovery result carries no lowered program identity",
    })?;
    let pending = payload.structure.pending_marks();
    let accepted = payload.structure.accept(accept_discovered)?;
    let provenance = StructureProvenance {
        method: Arc::from(payload.query.method.id()),
        inference: Arc::from(payload.query.inference().as_str()),
        discovered_class: Arc::from(payload.query.output().as_str()),
        query: query_digest(&payload.query),
        search_bounds: Arc::from(bounds_label(&payload.query)),
        data_snapshot: *executed.identities.data_snapshot.as_bytes(),
        program: *program.as_bytes(),
        claim: *claim.claim_id.as_bytes(),
        review: Arc::from(if pending == 0 { "accepted" } else { "accepted_after_review" }),
        resolved_marks: pending,
    };
    Ok(accepted.with_provenance(Arc::new(provenance)))
}

/// A structure posterior accepted from a compiled discovery, with the search's
/// provenance.
///
/// A study takes it through [`crate::StudyBuilder::accepted_graph_posterior`]:
/// the posterior's atoms and weights become the study's graph posterior and
/// the provenance enters its contract and program identity.
/// [`Self::map_dag`] instead keeps the single most probable DAG as an
/// accepted graph.
#[derive(Clone, Debug)]
pub struct AcceptedPosterior {
    posterior: GraphPosterior,
    provenance: Arc<StructureProvenance>,
    temporal: bool,
}

impl AcceptedPosterior {
    /// The posterior over structures.
    #[must_use]
    pub const fn posterior(&self) -> &GraphPosterior {
        &self.posterior
    }

    /// The search and acceptance this posterior came from.
    #[must_use]
    pub fn provenance(&self) -> &StructureProvenance {
        &self.provenance
    }

    /// The posterior and its provenance.
    pub(crate) fn into_parts(self) -> (GraphPosterior, Arc<StructureProvenance>) {
        (self.posterior, self.provenance)
    }

    /// The most probable DAG atom as an accepted static graph, carrying the
    /// posterior's provenance with the review recorded as `posterior_map`.
    /// Ties go to the first atom in the posterior's order.
    ///
    /// # Errors
    ///
    /// `route_not_supported` for a temporal posterior, for atoms that are not
    /// DAGs, or for a search over a variable list that is not the schema's
    /// columns in order (the atoms index the searched variables by position);
    /// the posterior's own refusals for an atom that is not acyclic.
    pub fn map_dag(&self) -> Result<AcceptedGraph, CausalError> {
        if self.temporal {
            return Err(crate::compile_reason!(
                "route_not_supported",
                "a temporal structure posterior has no single static DAG; supply it as a graph \
                 posterior instead"
            ));
        }
        if self.posterior.atom_kind() != antecedent_discovery::GraphPosteriorAtomKind::Dag {
            return Err(crate::compile_reason!(
                "route_not_supported",
                "the posterior's atoms are not DAGs; supply it as a graph posterior instead"
            ));
        }
        let best = self
            .posterior
            .weights()
            .iter()
            .enumerate()
            .fold(None::<(usize, f64)>, |best, (index, &weight)| match best {
                Some((_, top)) if top >= weight => best,
                _ => Some((index, weight)),
            })
            .map(|(index, _)| index)
            .ok_or_else(|| {
                crate::compile_reason!("invalid_argument", "the structure posterior has no atoms")
            })?;
        let mask = self.posterior.atoms()[best].clone();
        let dag = antecedent_discovery::adjacency_mask::dag_from_adjacency_mask(
            mask,
            self.posterior.n_vars(),
        )?;
        let mut provenance = (*self.provenance).clone();
        provenance.review = Arc::from("posterior_map");
        let mut accepted = AcceptedGraph::dag(dag);
        if let Some(algorithm) = self.posterior.algorithm().map(|algorithm| &**algorithm) {
            accepted = accepted.with_discovery_algorithm(algorithm);
        }
        Ok(accepted.with_provenance(Arc::new(provenance)))
    }
}

/// Accept the structure posterior an executed discovery returned, recording
/// where it came from.
///
/// `result` must be the stamped result of a prepared discovery handle whose
/// method samples or enumerates a posterior, and `claim` the claim sealed over
/// it. A posterior has no pending marks, so the review is recorded as
/// `posterior`.
///
/// # Errors
///
/// A result that is not an executed structure posterior, that no prepared
/// handle stamped, or whose claim seals another execution; and a searched
/// variable list that is not the schema's columns in order.
pub fn accept_discovered_posterior(
    result: &StudyResult,
    claim: &ClaimEnvelope,
) -> Result<AcceptedPosterior, CausalError> {
    let payload = result.discovery().ok_or(CausalError::Conflict {
        what: "result",
        detail: "the result is not an executed structure search",
    })?;
    let DiscoveredStructure::Posterior(posterior) = &payload.structure else {
        return Err(crate::compile_reason!(
            "route_not_supported",
            "the {} search returned no structure posterior; accept it with accept_discovered",
            payload.query.method.id()
        ));
    };
    let executed = result.executed_contract.as_ref().ok_or(CausalError::Conflict {
        what: "result",
        detail: "the discovery result carries no execution stamp; accept the result of a \
                 prepared discovery handle",
    })?;
    if executed.identities != claim.identities {
        return Err(CausalError::Conflict {
            what: "claim",
            detail: "the claim seals another execution than this discovery result",
        });
    }
    let program = result.lowered_identity.ok_or(CausalError::Conflict {
        what: "result",
        detail: "the discovery result carries no lowered program identity",
    })?;
    let positional = payload
        .query
        .variables
        .iter()
        .enumerate()
        .all(|(index, variable)| variable.raw() as usize == index);
    if !positional {
        return Err(crate::compile_reason!(
            "route_not_supported",
            "the posterior's atoms index the searched variables by position; search the \
             schema's columns in order to accept the posterior as the study's structure"
        ));
    }
    let provenance = StructureProvenance {
        method: Arc::from(payload.query.method.id()),
        inference: Arc::from(payload.query.inference().as_str()),
        discovered_class: Arc::from(payload.query.output().as_str()),
        query: query_digest(&payload.query),
        search_bounds: Arc::from(bounds_label(&payload.query)),
        data_snapshot: *executed.identities.data_snapshot.as_bytes(),
        program: *program.as_bytes(),
        claim: *claim.claim_id.as_bytes(),
        review: Arc::from("posterior"),
        resolved_marks: 0,
    };
    Ok(AcceptedPosterior {
        posterior: posterior.clone(),
        provenance: Arc::new(provenance),
        temporal: payload.query.method.is_temporal(),
    })
}

impl crate::Study {
    /// Provenance of the structure this study reads, when it was accepted
    /// from a compiled discovery: on its accepted graph, or on its accepted
    /// graph posterior.
    pub(crate) fn structure_provenance(&self) -> Option<&StructureProvenance> {
        self.graph.provenance().or(self.config.posterior_provenance.as_deref())
    }

    /// Shared handle of [`Self::structure_provenance`].
    pub(crate) fn structure_provenance_shared(&self) -> Option<Arc<StructureProvenance>> {
        self.graph.provenance_shared().or_else(|| self.config.posterior_provenance.clone())
    }
}

/// Attach the executable specifications a builder carried to the study it built.
///
/// # Errors
///
/// A specification beside a query it does not execute, a discovery query
/// without its specification (unless only inspecting), or a specification
/// that does not state the study's query.
pub(crate) fn attach_specifications(
    study: &mut crate::Study,
    discovery: Option<Arc<DiscoverySpec>>,
    mechanisms: Option<Arc<MechanismProgram>>,
    inspect_only: bool,
) -> Result<(), CausalError> {
    match (&study.query, &discovery) {
        (CausalQuery::Discover(query), Some(spec)) => spec.check_query(query)?,
        (CausalQuery::Discover(_), None) if !inspect_only => {
            return Err(crate::compile_reason!(
                "required_option_missing",
                "a structure-discovery study needs its executable search; build it with \
                 StudyBuilder::discovery"
            ));
        }
        (_, Some(_)) => {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a discovery specification applies to a structure-discovery query only"
            ));
        }
        (_, None) => {}
    }
    if mechanisms.is_some()
        && !matches!(study.query, CausalQuery::Attribute(_) | CausalQuery::Simulate(_))
    {
        return Err(crate::compile_reason!(
            "option_not_applicable",
            "a mechanism program applies to an attribution or simulation query only"
        ));
    }
    if discovery.is_some() || mechanisms.is_some() {
        let config = Arc::make_mut(&mut study.config);
        config.discovery = discovery;
        config.mechanisms = mechanisms;
    }
    Ok(())
}
