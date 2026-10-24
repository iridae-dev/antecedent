//! The theorem routes of a failed identification, as repair families.
//!
//! [`crate::obligation_adapters::identification_failure_obligations`] reads the
//! evidence a failed identification owes, one verified obligation per route. This
//! module keeps the *routes themselves*: the back-door, front-door, instrument and
//! hedge families (and, when a non-robust sensitivity finding is supplied, the
//! sensitivity family) that decide whether a proposed study would repair the
//! failure, so an [`crate::EvidencePlan`] can ask each family's own theorem
//! checker. The repeated-measurement (panel) alternative owes obligations but has
//! no theorem checker; it is reported as an [`UnverifiedRoute`] and never scored.
//!
//! A route whose observed law already identifies the effect is absent, exactly as in
//! the obligation adapter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{EvidenceObligation, EvidenceObligationKind};
use antecedent_core::{CausalQuery, IdentificationStatus, VariableId};
use antecedent_graph::Admg;
use antecedent_identify::IdentificationResult;

use crate::evidence_plan::UnverifiedRoute;
use crate::identification_repair::{
    FrontDoorRepairFamily, HedgeRepairFamily, InstrumentRepairFamily, SensitivityFinding,
    SensitivityRepairFamily, latent_expansion, panel_obligations,
};
use crate::repair::{BackdoorRepairFamily, RepairError, RepairFamily};
use crate::repair_artifact::RepairFamilyRef;

/// One theorem route of a failed identification.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FailureRoute {
    /// Back-door adjustment over a set measured jointly with treatment and outcome.
    Backdoor(Box<BackdoorRepairFamily>),
    /// Front-door adjustment through a mediator set.
    FrontDoor(Box<FrontDoorRepairFamily>),
    /// An instrumental variable.
    Instrument(Box<InstrumentRepairFamily>),
    /// Experiments on a hedge.
    Hedge(Box<HedgeRepairFamily>),
    /// Measuring the confounder a sensitivity finding names.
    Sensitivity(Box<SensitivityRepairFamily>),
}

impl FailureRoute {
    /// The route's repair family.
    #[must_use]
    pub fn family(&self) -> &dyn RepairFamily {
        match self {
            Self::Backdoor(f) => &**f,
            Self::FrontDoor(f) => &**f,
            Self::Instrument(f) => &**f,
            Self::Hedge(f) => &**f,
            Self::Sensitivity(f) => &**f,
        }
    }

    /// The family as an artifact reference.
    #[must_use]
    pub fn reference(&self) -> RepairFamilyRef<'_> {
        match self {
            Self::Backdoor(f) => RepairFamilyRef::Backdoor(f),
            Self::FrontDoor(f) => RepairFamilyRef::FrontDoor(f),
            Self::Instrument(f) => RepairFamilyRef::Instrument(f),
            Self::Hedge(f) => RepairFamilyRef::Hedge(f),
            Self::Sensitivity(f) => RepairFamilyRef::Sensitivity(f),
        }
    }
}

/// Every route of a failed identification.
#[derive(Clone, Debug)]
pub struct FailureRoutes {
    /// The routes a theorem checker decides, in a fixed order.
    pub routes: Vec<FailureRoute>,
    /// The obligations of the panel alternative, which no checker decides.
    pub panel: Vec<EvidenceObligation>,
    /// The graph variable standing for the confounder a sensitivity route names
    /// (past every graph variable), when there is one.
    pub confounder: Option<VariableId>,
}

impl FailureRoutes {
    /// Every route's unresolved obligations, plus the panel's.
    #[must_use]
    pub fn obligations(&self) -> Vec<EvidenceObligation> {
        let mut out: Vec<EvidenceObligation> =
            self.routes.iter().flat_map(|r| r.family().unresolved_obligations()).collect();
        out.extend(self.panel.iter().cloned());
        out
    }

    /// The routes no theorem checker decides.
    #[must_use]
    pub fn unverified(&self) -> Vec<UnverifiedRoute> {
        if self.panel.is_empty() {
            return Vec::new();
        }
        vec![UnverifiedRoute {
            route: Arc::from("panel"),
            obligations: self.panel.iter().map(|o| Arc::clone(&o.id)).collect(),
            reason: Arc::from(
                "no theorem checker decides a within-unit comparison: it removes confounding by \
                 causes that do not change over the window only if they are constant, an \
                 assumption no study establishes",
            ),
        }]
    }
}

fn absent_when_identified<T>(attempt: Result<T, RepairError>) -> Result<Option<T>, RepairError> {
    match attempt {
        Ok(family) => Ok(Some(family)),
        Err(error) if error.detail == "identification_repair.not_a_failure" => Ok(None),
        Err(error) => Err(error),
    }
}

/// The routes of a failed identification of an average effect.
///
/// `graph` is the structure the identification ran on and `observed` the
/// variables measured jointly with the treatment and the outcome in `population`.
/// A non-robust `sensitivity` finding adds the route that measures the confounder
/// it names. Queries other than an average effect owe no route here.
///
/// # Errors
/// `identification_repair.not_a_failure` when `result` is identified, and
/// `identification_repair.invalid_request` for a graph or variables the
/// identifiers cannot read.
pub fn failure_routes(
    result: &IdentificationResult,
    graph: &Admg,
    observed: &[VariableId],
    population: &str,
    sensitivity: Option<&SensitivityFinding>,
) -> Result<FailureRoutes, RepairError> {
    if matches!(
        result.status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
    ) {
        return Err(RepairError::not_a_failure("the identification already succeeded"));
    }
    let CausalQuery::AverageEffect(query) = &result.query else {
        return Ok(FailureRoutes { routes: Vec::new(), panel: Vec::new(), confounder: None });
    };
    let (treatment, outcome) = (query.treatment, query.outcome);
    let (dag, _latents) = latent_expansion(graph)?;
    let population: Arc<str> = Arc::from(population);
    let mut routes = Vec::new();
    let mut panel = Vec::new();
    let mut confounder = None;

    if let Some(family) = absent_when_identified(BackdoorRepairFamily::try_new(
        dag.clone(),
        treatment,
        outcome,
        Arc::clone(&population),
        observed.iter().copied(),
        &[],
    ))? {
        let mut unmeasured: Vec<VariableId> = Vec::new();
        for obligation in family.unresolved_obligations() {
            if obligation.kind == EvidenceObligationKind::ProvideJointLaw {
                for variable in obligation.variables.iter() {
                    if !observed.contains(variable) && !unmeasured.contains(variable) {
                        unmeasured.push(*variable);
                    }
                }
            }
        }
        if !unmeasured.is_empty() {
            unmeasured.sort_unstable_by_key(|v| v.raw());
            panel = panel_obligations(treatment, outcome, &population, &unmeasured)?;
        }
        routes.push(FailureRoute::Backdoor(Box::new(family)));
    }
    if let Some(family) = absent_when_identified(FrontDoorRepairFamily::try_new(
        dag.clone(),
        treatment,
        outcome,
        Arc::clone(&population),
        observed.iter().copied(),
    ))? {
        routes.push(FailureRoute::FrontDoor(Box::new(family)));
    }
    if let Some(family) = absent_when_identified(InstrumentRepairFamily::try_new(
        dag.clone(),
        treatment,
        outcome,
        Arc::clone(&population),
        observed.iter().copied(),
    ))? {
        routes.push(FailureRoute::Instrument(Box::new(family)));
    }
    if let Some(family) = absent_when_identified(HedgeRepairFamily::try_new(
        graph.clone(),
        treatment,
        outcome,
        Arc::clone(&population),
        observed.iter().copied(),
    ))? {
        routes.push(FailureRoute::Hedge(Box::new(family)));
    }
    if let Some(finding) = sensitivity {
        if let Some(family) = absent_when_identified(SensitivityRepairFamily::try_new(
            dag,
            treatment,
            outcome,
            Arc::clone(&population),
            observed.iter().copied(),
            *finding,
        ))? {
            confounder = Some(family.confounder());
            routes.push(FailureRoute::Sensitivity(Box::new(family)));
        }
    }
    Ok(FailureRoutes { routes, panel, confounder })
}

#[cfg(test)]
mod tests {
    use antecedent_core::{AssumptionSet, AverageEffectQuery};
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord};

    use super::*;
    use crate::identification_repair::SensitivityMethod;
    use crate::obligation_adapters::identification_failure_obligations;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn failed(treatment: u32, outcome: u32) -> IdentificationResult {
        IdentificationResult::not_identified(
            CausalQuery::average_effect(AverageEffectQuery::binary_ate(v(treatment), v(outcome))),
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        )
    }

    fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
        let mut dag = Dag::with_variables(nodes);
        for (a, b) in edges {
            dag.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        dag
    }

    #[test]
    fn the_routes_cover_exactly_the_obligations_of_the_adapter() {
        // T=0 -> W=1 -> Y=2 with T <-> Y and W <-> Y.
        let mut graph = Admg::with_variables(3);
        for (a, b) in [(0, 1), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 2), (1, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let observed = [v(0), v(1), v(2)];
        let routes = failure_routes(&failed(0, 2), &graph, &observed, "pop", None).unwrap();
        let adapter =
            identification_failure_obligations(&failed(0, 2), &graph, &observed, "pop", None)
                .unwrap();
        let mut mine: Vec<_> = routes.obligations().iter().map(|o| o.id.to_string()).collect();
        let mut theirs: Vec<_> = adapter.iter().map(|o| o.id.to_string()).collect();
        mine.sort();
        theirs.sort();
        assert_eq!(mine, theirs);
        let families: Vec<_> = routes.routes.iter().map(|r| r.family().family_id()).collect();
        assert!(families.contains(&"hedge") && families.contains(&"backdoor"));
        assert_eq!(routes.unverified().len(), 1);
        assert_eq!(routes.unverified()[0].route.as_ref(), "panel");
    }

    #[test]
    fn a_sensitivity_finding_adds_the_confounder_route_and_a_robust_one_is_absent_by_refusal() {
        let graph = Admg::from_dag(&dag(3, &[(0, 1), (2, 0), (2, 1)])).unwrap();
        let observed = [v(0), v(1), v(2)];
        let finding =
            SensitivityFinding { method: SensitivityMethod::EValue, strength: 1.4, robust: false };
        let routes =
            failure_routes(&failed(0, 1), &graph, &observed, "pop", Some(&finding)).unwrap();
        assert_eq!(routes.confounder, Some(v(3)));
        assert!(routes.routes.iter().any(|r| r.family().family_id() == "sensitivity"));
        let robust = SensitivityFinding { robust: true, ..finding };
        // The sensitivity family refuses a robust finding, so that route is absent; the
        // other routes of the failed identification remain and no confounder is named.
        let without =
            failure_routes(&failed(0, 1), &graph, &observed, "pop", Some(&robust)).unwrap();
        assert_eq!(without.confounder, None);
        assert!(!without.routes.iter().any(|r| r.family().family_id() == "sensitivity"));
    }

    #[test]
    fn an_identified_result_owes_no_route() {
        let graph = Admg::from_dag(&dag(2, &[(0, 1)])).unwrap();
        let mut identified = failed(0, 1);
        identified.status = IdentificationStatus::NonparametricallyIdentified;
        let refused = failure_routes(&identified, &graph, &[v(0), v(1)], "pop", None).unwrap_err();
        assert_eq!(refused.detail, "identification_repair.not_a_failure");
    }
}
