//! Adapters from existing failure vocabularies to [`EvidenceObligation`]s.
//!
//! Every adapter reads a failure that a theorem-specific route already
//! reported (an unmet transport factor, a z-transport proof factor that no
//! catalog regime supplies, a support coordinate without evidence, a planner's
//! hypothetical catalog delta, an unresolved assumption record) and restates it
//! as a machine-readable obligation that retains the source proof step. They
//! never change the theorem-specific identifiers: factor ids, expression
//! leaves, rule names and regime ids are carried as provenance, not rewritten.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{
    EvidenceObligation, EvidenceObligationError, EvidenceObligationKind, EvidenceObligationSpec,
    ObligationProvenance, ObligationRecord, ObligationRegime, ObligationScope,
    unresolved_assumption_obligations,
};
use antecedent_core::{
    CausalQuery, DistributionAvailability, EvidenceCatalog, EvidenceCatalogDelta, FactorNeed,
    IdentificationStatus, RegimeKind, SupportReport, SupportStatus, VariableId,
};
use antecedent_graph::Admg;
use antecedent_identify::{CatalogTransportResult, ClassicalTransportQuery, IdentificationResult};

use crate::identification_repair::{
    FrontDoorRepairFamily, HedgeRepairFamily, InstrumentRepairFamily, SensitivityFinding,
    SensitivityRepairFamily, latent_expansion, panel_obligations,
};
use crate::repair::{BackdoorRepairFamily, RepairError, RepairFamily};
use crate::z_transport_planner::{ZTransportFailureSnapshot, ZTransportFailureStatus};

fn provenance(family: &str, source: &str, step: Option<String>) -> ObligationProvenance {
    ObligationProvenance {
        family: Arc::from(family),
        source: Arc::from(source),
        proof_step: step.map(Arc::from),
    }
}

/// Rebuild `obligation` with a more specific reason, keeping every other field.
fn with_reason(
    obligation: EvidenceObligation,
    reason: String,
) -> Result<EvidenceObligation, EvidenceObligationError> {
    EvidenceObligation::try_new(EvidenceObligationSpec {
        quantities: obligation.quantities,
        kind: obligation.kind,
        scope: obligation.scope,
        variables: obligation.variables,
        population: obligation.population,
        regime: obligation.regime,
        reason: Arc::from(reason),
        required_slots: obligation.required_slots,
        min_additional_samples: obligation.min_additional_samples,
        provenance: obligation.provenance,
    })
}

/// One obligation per executable factor that no available catalog regime
/// satisfies, in the catalog's own stable report order (factor id, then reason).
/// The catalog's [`EvidenceCatalog::unmet_factor_dependencies`] decides which
/// factors are unmet; the factor id is retained as the proof step.
///
/// # Errors
/// A need that reads no variable.
pub fn factor_obligations(
    catalog: &EvidenceCatalog,
    needs: &[(Arc<str>, FactorNeed<'_>)],
    family: &str,
    source: &str,
) -> Result<Vec<EvidenceObligation>, EvidenceObligationError> {
    let mut out = Vec::new();
    for unmet in catalog.unmet_factor_dependencies(needs).iter() {
        let Some((_, need)) = needs.iter().find(|(id, _)| id == &unmet.factor_id) else {
            continue;
        };
        out.push(EvidenceObligation::from_unmet_factor(
            catalog,
            &unmet.factor_id,
            need,
            provenance(family, source, Some(format!("factor:{}", unmet.factor_id))),
        )?);
    }
    Ok(out)
}

/// The primary source factor `P^source(outcomes | do(treatments))` that a
/// catalog-aware classical-transport search could not bind, with the solver's
/// own per-strategy obligations retained verbatim in the reason and the
/// searched stages as the proof step. Empty unless the search reported missing
/// evidence, and empty when the catalog already supplies that factor (the gap is
/// then elsewhere in the formula and is not guessed). A not-certified or
/// identified result yields no evidence request.
///
/// # Errors
/// A query that names no outcome.
pub(crate) fn transport_result_obligations(
    result: &CatalogTransportResult,
    query: &ClassicalTransportQuery,
    catalog: &EvidenceCatalog,
    source: &str,
) -> Result<Vec<EvidenceObligation>, EvidenceObligationError> {
    let CatalogTransportResult::MissingEvidence { searched, obligations } = result else {
        return Ok(Vec::new());
    };
    let need = FactorNeed {
        population: &query.source,
        variables: &query.outcomes,
        conditioned_on: &[],
        interventions: &query.treatments,
    };
    if catalog.satisfying_regime(&need).is_some() {
        return Ok(Vec::new());
    }
    let stages = searched.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join(">");
    let base = EvidenceObligation::from_unmet_factor(
        catalog,
        "source_interventional_law",
        &need,
        provenance("transport", source, Some(format!("searched:{stages}"))),
    )?;
    let solver = obligations.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join("; ");
    Ok(vec![with_reason(
        base,
        format!("transport.missing_evidence: primary source factor unbound; solver: {solver}"),
    )?])
}

/// One obligation per required z-transport factor that no catalog regime
/// supplies, read from the checked formula frozen in a missing-evidence failure
/// snapshot. The expression leaf is the proof step (`leaf:<id>`) and the
/// solver's own binding failure is kept in the reason. Empty for any other
/// status, which owes no checked factor.
///
/// # Errors
/// A factor that reads no variable.
pub(crate) fn z_transport_obligations(
    snapshot: &ZTransportFailureSnapshot,
    catalog: &EvidenceCatalog,
) -> Result<Vec<EvidenceObligation>, EvidenceObligationError> {
    if snapshot.status() != &ZTransportFailureStatus::MissingEvidence {
        return Ok(Vec::new());
    }
    let Some(derivation) = snapshot.derivation() else {
        return Ok(Vec::new());
    };
    let inspection = derivation.inspect_proof(catalog);
    let mut out = Vec::new();
    for factor in inspection.factors.iter().filter(|f| f.failure.is_some()) {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        let variables = ids(&factor.variables);
        let conditioned_on = ids(&factor.conditioned_on);
        let interventions =
            factor.intervention.iter().map(|(v, _)| VariableId::from_raw(*v)).collect::<Vec<_>>();
        let need = FactorNeed {
            population: &factor.population,
            variables: &variables,
            conditioned_on: &conditioned_on,
            interventions: &interventions,
        };
        let name = format!("leaf:{}", factor.leaf);
        let base = EvidenceObligation::from_unmet_factor(
            catalog,
            &name,
            &need,
            provenance("z_transport", snapshot.catalog_digest(), Some(name.clone())),
        )?;
        let failure = factor.failure.as_deref().unwrap_or("unbound");
        out.push(with_reason(base, format!("z_transport factor {name} unbound: {failure}"))?);
    }
    Ok(out)
}

/// Obligations of a support report's coordinates that lack evidence:
/// `MissingEvidence`, `OutsideEmpiricalSupport` and `Extrapolative` coordinates
/// become one `EstablishSupport` obligation (a slot per coordinate), and
/// `WeakOverlap` coordinates become one `IncreaseSample` obligation. A report
/// with no per-coordinate labels and a `MissingEvidence` summary yields one
/// `EstablishSupport` obligation for the whole surface. Supported coordinates
/// owe nothing.
///
/// # Errors
/// No treatment variable to name.
pub fn support_obligations(
    report: &SupportReport,
    population: &str,
    treatments: &[VariableId],
    source: &str,
) -> Result<Vec<EvidenceObligation>, EvidenceObligationError> {
    let labels: Vec<(usize, SupportStatus)> = match &report.point_status {
        Some(points) => points.iter().copied().enumerate().collect(),
        None if report.status == SupportStatus::MissingEvidence => {
            vec![(0, SupportStatus::MissingEvidence)]
        }
        None => Vec::new(),
    };
    let slots = |wanted: &dyn Fn(SupportStatus) -> bool| -> Vec<(usize, SupportStatus)> {
        labels.iter().copied().filter(|(_, status)| wanted(*status)).collect()
    };
    let establish = slots(&|s| {
        matches!(
            s,
            SupportStatus::MissingEvidence
                | SupportStatus::OutsideEmpiricalSupport
                | SupportStatus::Extrapolative
        )
    });
    let weak = slots(&|s| s == SupportStatus::WeakOverlap);
    let mut out = Vec::new();
    for (kind, group) in [
        (EvidenceObligationKind::EstablishSupport, establish),
        (EvidenceObligationKind::IncreaseSample, weak),
    ] {
        if group.is_empty() {
            continue;
        }
        let required_slots: Vec<Arc<str>> =
            group.iter().map(|(i, _)| Arc::from(format!("coordinate:{i}"))).collect();
        let worst = group
            .iter()
            .map(|(_, status)| *status)
            .max_by_key(|status| status.severity())
            .map_or("missing_evidence", SupportStatus::as_str);
        out.push(EvidenceObligation::try_new(EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::new(),
            kind,
            scope: ObligationScope::Factor,
            variables: Arc::from(treatments),
            population: Some(Arc::from(population)),
            regime: ObligationRegime::observational(false),
            reason: Arc::from(format!(
                "support: {} coordinate(s) lack support evidence (worst label {worst})",
                group.len()
            )),
            required_slots: Arc::from(required_slots),
            min_additional_samples: (kind == EvidenceObligationKind::IncreaseSample).then_some(1),
            provenance: provenance(
                "support",
                source,
                group.first().map(|(i, _)| format!("coordinate:{i}")),
            ),
        })?);
    }
    Ok(out)
}

/// What each proposed regime of a planner's catalog delta is meant to supply,
/// as an obligation: an experimental regime is `Intervene`, a conditional
/// observational regime `ProvideConditionalLaw`, a joint multi-variable
/// observational regime `ProvideJointLaw` and any other `Measure`. The regime id
/// is the proof step; the delta itself stays a hypothesis, never evidence.
///
/// # Errors
/// A proposed regime that measures nothing.
pub fn delta_obligations(
    delta: &EvidenceCatalogDelta,
    source: &str,
) -> Result<Vec<EvidenceObligation>, EvidenceObligationError> {
    delta
        .proposed_regimes
        .iter()
        .map(|regime| {
            let joint = matches!(regime.distribution, DistributionAvailability::Joint)
                && regime.measured.len() > 1;
            let kind = match regime.kind {
                RegimeKind::Experimental => EvidenceObligationKind::Intervene,
                RegimeKind::Observational if !regime.conditioned_on.is_empty() => {
                    EvidenceObligationKind::ProvideConditionalLaw
                }
                RegimeKind::Observational if joint => EvidenceObligationKind::ProvideJointLaw,
                RegimeKind::Observational => EvidenceObligationKind::Measure,
            };
            EvidenceObligation::try_new(EvidenceObligationSpec {
                quantities: std::collections::BTreeMap::new(),
                kind,
                scope: ObligationScope::Factor,
                variables: Arc::clone(&regime.measured),
                population: Some(Arc::clone(&regime.population)),
                regime: ObligationRegime {
                    interventions: Arc::clone(&regime.interventions),
                    conditioned_on: Arc::clone(&regime.conditioned_on),
                    joint,
                },
                reason: Arc::from(format!(
                    "planner delta proposes regime {} in population {}",
                    regime.id.raw(),
                    regime.population
                )),
                required_slots: Arc::from([Arc::from(format!("regime:{}", regime.id.raw()))]),
                min_additional_samples: None,
                provenance: provenance(
                    "planner_delta",
                    source,
                    Some(format!("regime:{}", regime.id.raw())),
                ),
            })
        })
        .collect()
}

/// The `EstablishAssumption` obligations of unresolved assumption records. A
/// proposed study never satisfies them.
#[must_use]
pub fn assumption_obligations(records: &[ObligationRecord]) -> Vec<EvidenceObligation> {
    unresolved_assumption_obligations(records)
}

/// A route that already identifies from the observed law is not a failure; its
/// obligations are simply absent from the failure's evidence.
fn absent_when_identified<T>(attempt: Result<T, RepairError>) -> Result<Option<T>, RepairError> {
    match attempt {
        Ok(family) => Ok(Some(family)),
        Err(error) if error.detail == "identification_repair.not_a_failure" => Ok(None),
        Err(error) => Err(error),
    }
}

/// The evidence that would resolve a failed identification, one verified
/// obligation per route, each retaining the proof step it was derived from.
///
/// `graph` is the structure the identification ran on (bidirected edges become
/// unmeasured common causes, one past the largest graph variable each), and
/// `observed` the variables measured jointly with the treatment and the outcome
/// in `population`. The routes are:
///
/// * back-door: one joint observational law over the treatment, the outcome and
///   the first admissible adjustment set ([`BackdoorRepairFamily`]);
/// * front-door: one joint law over the treatment, the outcome and a mediator
///   set the criterion accepts ([`FrontDoorRepairFamily`]);
/// * instrument: a joint law with an instrument, with its relevance, exclusion
///   and effect restriction as `EstablishAssumption` obligations that no study
///   satisfies, or the search for an instrument when the graph has none
///   ([`InstrumentRepairFamily`]);
/// * hedge: experiments on the treatment and on the hedge nodes that identify the
///   effect through a surrogate ([`HedgeRepairFamily`]);
/// * sensitivity: the confounder to measure when `sensitivity` reports a
///   conclusion an unmeasured confounder would overturn
///   ([`SensitivityRepairFamily`]);
/// * panel: repeated measurement, when back-door adjustment needs variables that
///   are not observed.
///
/// A route whose observed law already identifies the effect contributes nothing.
/// The query must be an average effect; other queries owe nothing here.
///
/// # Errors
/// `identification_repair.not_a_failure` when `result` is identified, and
/// `identification_repair.invalid_request` for a graph or variables the
/// identifiers cannot read.
pub fn identification_failure_obligations(
    result: &IdentificationResult,
    graph: &Admg,
    observed: &[VariableId],
    population: &str,
    sensitivity: Option<&SensitivityFinding>,
) -> Result<Vec<EvidenceObligation>, RepairError> {
    if matches!(
        result.status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
    ) {
        return Err(RepairError::not_a_failure("the identification already succeeded"));
    }
    let CausalQuery::AverageEffect(query) = &result.query else {
        return Ok(Vec::new());
    };
    let (treatment, outcome) = (query.treatment, query.outcome);
    let (dag, _latents) = latent_expansion(graph)?;
    let population: Arc<str> = Arc::from(population);
    let mut out = Vec::new();

    let backdoor = absent_when_identified(BackdoorRepairFamily::try_new(
        dag.clone(),
        treatment,
        outcome,
        Arc::clone(&population),
        observed.iter().copied(),
        &[],
    ))?;
    let mut unmeasured: Vec<VariableId> = Vec::new();
    if let Some(family) = backdoor {
        for obligation in family.unresolved_obligations() {
            if obligation.kind == EvidenceObligationKind::ProvideJointLaw {
                for variable in obligation.variables.iter() {
                    if !observed.contains(variable) && !unmeasured.contains(variable) {
                        unmeasured.push(*variable);
                    }
                }
            }
            out.push(obligation);
        }
    }
    if let Some(family) = absent_when_identified(FrontDoorRepairFamily::try_new(
        dag.clone(),
        treatment,
        outcome,
        Arc::clone(&population),
        observed.iter().copied(),
    ))? {
        out.extend(family.unresolved_obligations());
    }
    if let Some(family) = absent_when_identified(InstrumentRepairFamily::try_new(
        dag.clone(),
        treatment,
        outcome,
        Arc::clone(&population),
        observed.iter().copied(),
    ))? {
        out.extend(family.unresolved_obligations());
    }
    if let Some(family) = absent_when_identified(HedgeRepairFamily::try_new(
        graph.clone(),
        treatment,
        outcome,
        Arc::clone(&population),
        observed.iter().copied(),
    ))? {
        out.extend(family.unresolved_obligations());
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
            out.extend(family.unresolved_obligations());
        }
    }
    if !unmeasured.is_empty() {
        unmeasured.sort_unstable_by_key(|v| v.raw());
        out.extend(panel_obligations(treatment, outcome, &population, &unmeasured)?);
    }
    Ok(out)
}
