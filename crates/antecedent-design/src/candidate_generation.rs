//! Candidate generation: from evidence obligations to unfinished study candidates.
//!
//! A failed identification owes evidence ([`EvidenceObligation`]s, each retaining
//! its proof step). [`generate_candidates`] restates that debt as
//! [`CandidateTemplate`]s: the study kind, population, hard-intervention set,
//! measured variables, joint measurement and the laws the study would deliver are
//! filled in from the obligations, and nothing else is. Cost, timing, recruitment,
//! unit rules, feasibility and sample size are facts about a real study that only
//! the caller knows, so a template is a different type from a
//! [`DurableStudyCandidate`], refuses [`CandidateTemplate::validate`] and becomes a
//! candidate only through [`CandidateTemplate::complete`] with a full
//! [`StudyDeclarations`]. No default is ever invented.
//!
//! An assumption obligation (relevance and exclusion of an instrument, the time
//! invariance a panel relies on, the existence of an instrument) is never
//! addressed by a study, so it yields no template and is reported as
//! [`UnmatchedObligation`]. A template is a proposal of what to collect, not a
//! verdict: [`crate::repair()`] decides, with the family's own theorem checker,
//! whether the completed candidate would repair the contract.
//!
//! [`verified_unlocks`] reads a finished [`RepairReport`] back into the ranker's
//! vocabulary. Only a candidate that a repair search verified sufficient on its own
//! unlocks identification, and the unlock lists name the variables of those
//! candidates alone, so a variable-name match on an unverified study never marks it
//! licensed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_contract::{EvidenceObligation, EvidenceObligationKind};
use antecedent_core::{DistributionAvailability, QueryId, VariableId, reason_code};

use crate::CandidateDesign;
use crate::repair::{RepairClassification, RepairReport};
use crate::study_candidate::{
    DurableStudyCandidate, ExpectedEvidence, StudyCandidateError, StudyCostDeclaration, StudyKind,
    UnitRules,
};

fn refusal(detail: &'static str, message: impl Into<String>) -> StudyCandidateError {
    StudyCandidateError {
        code: reason_code!("design_signal_invalid"),
        detail,
        message: message.into(),
    }
}

fn sorted(variables: &[VariableId]) -> Vec<VariableId> {
    let mut out = variables.to_vec();
    out.sort_unstable_by_key(|v| v.raw());
    out.dedup();
    out
}

fn id_list(variables: &[VariableId]) -> String {
    sorted(variables).iter().map(|v| v.raw().to_string()).collect::<Vec<_>>().join(",")
}

/// What a caller must declare about a real study before a template becomes a
/// candidate. Every field is required: nothing here is ever defaulted.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyDeclarations {
    /// Human label; the template's suggestion when `None`.
    pub label: Option<Arc<str>>,
    /// Planned sample size, at least one (and at least the template's minimum).
    pub sample_size: u64,
    /// Recruitment and sampling declaration.
    pub recruitment: Arc<str>,
    /// Timing declaration (horizon, follow-up, delay to delivery).
    pub timing: Arc<str>,
    /// Unit and cluster rules.
    pub unit_rules: UnitRules,
    /// Declared cost with its units.
    pub cost: StudyCostDeclaration,
    /// Whether the study is feasible as declared.
    pub feasible: bool,
    /// Feasibility notes.
    pub feasibility_notes: Arc<[Arc<str>]>,
    /// External provider that would run or supply the study, if any.
    pub external_provider: Option<Arc<str>>,
}

/// An unfinished study candidate: everything an obligation determines, and none
/// of what only the caller can declare.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateTemplate {
    /// What the study does.
    pub kind: StudyKind,
    /// Population the study is collected in.
    pub population: Arc<str>,
    /// Hard-intervention set; empty unless an experiment.
    pub interventions: Arc<[VariableId]>,
    /// Variables the study measures.
    pub measured: Arc<[VariableId]>,
    /// Whether `measured` is observed jointly on each unit.
    pub joint_measurement: bool,
    /// The laws the study would deliver.
    pub expected_evidence: Arc<[ExpectedEvidence]>,
    /// Ids of the obligations this study could address (a necessary screen, not a
    /// verdict), sorted.
    pub addresses: Vec<Arc<str>>,
    /// The theorem routes (obligation provenance families) those obligations came
    /// from, sorted.
    pub routes: Vec<Arc<str>>,
    /// The fewest rows the addressed obligations require, when any states one.
    pub minimum_sample_size: Option<u64>,
    /// A label to start from; the caller may replace it.
    pub suggested_label: Arc<str>,
}

impl CandidateTemplate {
    /// A template is not a candidate: the declarations a real study needs are
    /// missing, so this always refuses.
    ///
    /// # Errors
    /// `study_candidate.declarations_required` (reason `design_signal_invalid`),
    /// naming what [`Self::complete`] needs.
    pub fn validate(&self) -> Result<(), StudyCandidateError> {
        Err(refusal(
            "study_candidate.declarations_required",
            format!(
                "template `{}` has no sample size, recruitment, timing, unit rules, cost or \
                 feasibility; those are caller declarations and are never defaulted",
                self.suggested_label
            ),
        ))
    }

    /// Complete the template with the caller's declarations and validate the
    /// result.
    ///
    /// # Errors
    /// A sample size below the template's minimum, or any
    /// `study_candidate.wrong_contract` refusal of
    /// [`DurableStudyCandidate::validate`].
    pub fn complete(
        &self,
        declarations: StudyDeclarations,
    ) -> Result<DurableStudyCandidate, StudyCandidateError> {
        if let Some(minimum) = self.minimum_sample_size {
            if declarations.sample_size < minimum {
                return Err(refusal(
                    "study_candidate.sample_below_obligation",
                    format!(
                        "template `{}` addresses an obligation that needs at least {minimum} \
                         additional rows; the declared sample size is {}",
                        self.suggested_label, declarations.sample_size
                    ),
                ));
            }
        }
        let candidate = DurableStudyCandidate {
            label: declarations.label.unwrap_or_else(|| Arc::clone(&self.suggested_label)),
            kind: self.kind,
            population: Arc::clone(&self.population),
            interventions: Arc::clone(&self.interventions),
            measured: Arc::clone(&self.measured),
            joint_measurement: self.joint_measurement,
            sample_size: declarations.sample_size,
            recruitment: declarations.recruitment,
            timing: declarations.timing,
            unit_rules: declarations.unit_rules,
            cost: declarations.cost,
            feasible: declarations.feasible,
            feasibility_notes: declarations.feasibility_notes,
            expected_evidence: Arc::clone(&self.expected_evidence),
            external_provider: declarations.external_provider,
        };
        candidate.validate()?;
        Ok(candidate)
    }
}

/// An obligation no template addresses, with the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnmatchedObligation {
    /// The obligation's id.
    pub id: Arc<str>,
    /// The obligation's kind.
    pub kind: EvidenceObligationKind,
    /// The theorem route it came from.
    pub route: Arc<str>,
    /// Why no study is proposed for it.
    pub reason: &'static str,
}

/// The templates and the obligations they could not be generated for.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateGeneration {
    /// Templates in a deterministic order (kind, population, regime, variables).
    pub templates: Vec<CandidateTemplate>,
    /// Obligations a study cannot address, in input order.
    pub unmatched: Vec<UnmatchedObligation>,
}

/// The study kinds an obligation of route `family` could be answered by.
fn kinds_for(obligation: &EvidenceObligation, measured: usize) -> Vec<StudyKind> {
    let family = obligation.provenance.family.as_ref();
    match obligation.kind {
        EvidenceObligationKind::EstablishAssumption => Vec::new(),
        EvidenceObligationKind::Intervene => vec![StudyKind::Experiment],
        EvidenceObligationKind::IncreaseSample => vec![StudyKind::SampleIncrease],
        _ if family == "instrument" => {
            let mut kinds = vec![StudyKind::NaturalExperiment];
            if measured >= 3 {
                kinds.insert(0, StudyKind::Instrument);
            }
            kinds
        }
        _ if family == "panel" => vec![StudyKind::Panel],
        _ => vec![StudyKind::Observation],
    }
}

type TemplateKey = (&'static str, Arc<str>, Vec<u32>, Vec<u32>, Vec<u32>);

fn raw(variables: &[VariableId]) -> Vec<u32> {
    sorted(variables).iter().map(|v| v.raw()).collect()
}

/// Restate evidence obligations as unfinished study candidates.
///
/// Per obligation: an `Intervene` obligation yields an experiment setting its
/// intervention set and measuring its variables; an `IncreaseSample` obligation a
/// sample increase; a joint, conditional or marginal law an observation (an
/// instrument route additionally yields an instrument study and a natural
/// experiment, a panel route a panel); an assumption yields nothing. Templates
/// that would deliver the same law share one entry listing every obligation they
/// address.
#[must_use]
pub fn generate_candidates(obligations: &[EvidenceObligation]) -> CandidateGeneration {
    let mut by_key: BTreeMap<TemplateKey, CandidateTemplate> = BTreeMap::new();
    let mut unmatched = Vec::new();
    let unmatched_for =
        |obligation: &EvidenceObligation, reason: &'static str| UnmatchedObligation {
            id: Arc::clone(&obligation.id),
            kind: obligation.kind,
            route: Arc::clone(&obligation.provenance.family),
            reason,
        };
    for obligation in obligations {
        if !obligation.satisfiable_by_study() {
            unmatched.push(unmatched_for(
                obligation,
                "an assumption is established outside a study; no data collection proves it",
            ));
            continue;
        }
        let Some(population) = obligation.population.clone() else {
            unmatched.push(unmatched_for(obligation, "the obligation names no population"));
            continue;
        };
        let interventions = sorted(&obligation.regime.interventions);
        let conditioned_on = sorted(&obligation.regime.conditioned_on);
        let mut measured: Vec<VariableId> = obligation
            .variables
            .iter()
            .chain(conditioned_on.iter())
            .copied()
            .filter(|v| !interventions.contains(v))
            .collect();
        measured = sorted(&measured);
        let kinds = kinds_for(obligation, measured.len());
        if kinds.is_empty() {
            unmatched.push(unmatched_for(obligation, "no study kind addresses this obligation"));
            continue;
        }
        if measured.is_empty() && kinds != [StudyKind::SampleIncrease] {
            unmatched.push(unmatched_for(obligation, "the obligation names nothing to measure"));
            continue;
        }
        for kind in kinds {
            let key: TemplateKey = (
                kind.as_str(),
                Arc::clone(&population),
                raw(&interventions),
                raw(&measured),
                raw(&conditioned_on),
            );
            let entry = by_key.entry(key).or_insert_with(|| {
                let measured_arc: Arc<[VariableId]> = Arc::from(measured.as_slice());
                let evidence: Arc<[ExpectedEvidence]> = if kind == StudyKind::SampleIncrease {
                    Arc::from([])
                } else {
                    Arc::from([ExpectedEvidence {
                        population: Arc::clone(&population),
                        interventions: Arc::from(interventions.as_slice()),
                        intervention_values: Arc::from([]),
                        conditioned_on: Arc::from(conditioned_on.as_slice()),
                        measured: Arc::clone(&measured_arc),
                        distribution: DistributionAvailability::Joint,
                    }])
                };
                CandidateTemplate {
                    kind,
                    population: Arc::clone(&population),
                    interventions: Arc::from(interventions.as_slice()),
                    measured: measured_arc,
                    joint_measurement: true,
                    expected_evidence: evidence,
                    addresses: Vec::new(),
                    routes: Vec::new(),
                    minimum_sample_size: None,
                    suggested_label: Arc::from(label_of(
                        kind,
                        &population,
                        &interventions,
                        &measured,
                    )),
                }
            });
            if !entry.addresses.contains(&obligation.id) {
                entry.addresses.push(Arc::clone(&obligation.id));
                entry.addresses.sort();
            }
            if !entry.routes.contains(&obligation.provenance.family) {
                entry.routes.push(Arc::clone(&obligation.provenance.family));
                entry.routes.sort();
            }
            entry.minimum_sample_size =
                match (entry.minimum_sample_size, obligation.min_additional_samples) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                };
        }
    }
    CandidateGeneration { templates: by_key.into_values().collect(), unmatched }
}

fn label_of(
    kind: StudyKind,
    population: &str,
    interventions: &[VariableId],
    measured: &[VariableId],
) -> String {
    let mut label = format!("{} in {population}", kind.as_str());
    if !interventions.is_empty() {
        label.push_str(&format!(" setting [{}]", id_list(interventions)));
    }
    if !measured.is_empty() {
        label.push_str(&format!(" measuring [{}]", id_list(measured)));
    }
    label
}

/// A declared study a repair search did not verify as sufficient on its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnverifiedStudy {
    /// Index in the candidate slice the report was computed for.
    pub index: usize,
    /// The study's semantic id.
    pub semantic_id: Arc<str>,
    /// What the search found.
    pub classification: RepairClassification,
    /// The checker's reasons, verbatim.
    pub reasons: Vec<String>,
}

/// What a finished repair search licenses in the ranker's vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedUnlocks {
    /// Variables a verified-sufficient study measures (outside the base law) or
    /// sets, sorted.
    pub variables: Vec<VariableId>,
    /// Indices of candidates verified sufficient on their own.
    pub verified: Vec<usize>,
    /// Every other declared candidate, with why.
    pub unverified: Vec<UnverifiedStudy>,
}

impl VerifiedUnlocks {
    /// The ranker's unlock entry for `query`; `None` when nothing is verified.
    #[must_use]
    pub fn unlock_entry(&self, query: QueryId) -> Option<(QueryId, Arc<[VariableId]>)> {
        (!self.variables.is_empty()).then(|| (query, Arc::from(self.variables.as_slice())))
    }

    /// The planner actions of the verified candidates, with the index in
    /// `candidates` each came from. A candidate that was not verified has no
    /// action here and is reported in [`Self::unverified`].
    #[must_use]
    pub fn designs(&self, candidates: &[DurableStudyCandidate]) -> Vec<(usize, CandidateDesign)> {
        self.verified
            .iter()
            .filter_map(|&index| candidates.get(index).map(|c| (index, c.design_action())))
            .collect()
    }
}

/// Read a finished repair search back into unlock lists.
///
/// A candidate is verified when the report classifies the subset made of that
/// candidate alone as verified sufficient; a candidate that is sufficient only
/// together with others does not identify by itself and is unverified, with the
/// subset named in its reasons. `base_law` is the law observed today: measuring
/// what is already measured unlocks nothing, while setting a variable is always
/// new evidence.
#[must_use]
pub fn verified_unlocks(
    report: &RepairReport,
    candidates: &[DurableStudyCandidate],
    base_law: &[VariableId],
) -> VerifiedUnlocks {
    let mut variables: Vec<VariableId> = Vec::new();
    let mut verified = Vec::new();
    let mut unverified = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let id = candidate.semantic_id();
        let alone =
            report.outcomes.iter().find(|o| o.candidates.len() == 1 && o.candidates[0] == id);
        if let Some(outcome) = alone {
            if outcome.classification == RepairClassification::VerifiedSufficient {
                verified.push(index);
                variables
                    .extend(candidate.measured.iter().copied().filter(|v| !base_law.contains(v)));
                variables.extend(candidate.interventions.iter().copied());
                continue;
            }
        }
        let mut reasons = alone.map(|o| o.reasons.clone()).unwrap_or_default();
        let classification = alone.map_or(RepairClassification::Unevaluated, |o| o.classification);
        if let Some(joint) = report.outcomes.iter().find(|o| {
            o.candidates.len() > 1
                && o.classification == RepairClassification::VerifiedSufficient
                && o.candidates.contains(&id)
        }) {
            let members =
                joint.candidates.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join(", ");
            reasons.push(format!("sufficient only together with the subset [{members}]"));
        }
        unverified.push(UnverifiedStudy { index, semantic_id: id, classification, reasons });
    }
    VerifiedUnlocks { variables: sorted(&variables), verified, unverified }
}

#[cfg(test)]
mod tests {
    use antecedent_contract::EvidenceObligationKind;
    use antecedent_core::{ExecutionContext, SearchLimits};
    use antecedent_graph::{Admg, Dag, DenseNodeId};

    use super::*;
    use crate::identification_repair::{
        FrontDoorRepairFamily, HedgeRepairFamily, InstrumentRepairFamily, SensitivityFinding,
        SensitivityMethod, SensitivityRepairFamily,
    };
    use crate::repair::{
        BackdoorRepairFamily, RepairFamily, RepairLimits, RepairObjective, repair,
    };

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
        let mut dag = Dag::with_variables(nodes);
        for (a, b) in edges {
            dag.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        dag
    }

    fn admg(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
        let mut graph = Admg::with_variables(nodes);
        for (a, b) in directed {
            graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        for (a, b) in bidirected {
            graph.insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        graph
    }

    fn declarations(cost: u64) -> StudyDeclarations {
        StudyDeclarations {
            label: None,
            sample_size: 200,
            recruitment: Arc::from("census"),
            timing: Arc::from("one wave"),
            unit_rules: UnitRules {
                unit: Arc::from("patient"),
                cluster: None,
                whole_cluster_sampling: false,
            },
            cost: StudyCostDeclaration {
                units: cost,
                unit_label: Arc::from("usd"),
                sample_budget: 200,
            },
            feasible: true,
            feasibility_notes: Arc::from([]),
            external_provider: None,
        }
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(1)
    }

    fn limits() -> RepairLimits {
        RepairLimits {
            search: SearchLimits { operations: 1000, depth: 2 },
            memory_limit_bytes: 16 * 1024 * 1024,
        }
    }

    /// Complete every template, repair, and return the verified outcomes' labels.
    fn verified_labels(
        family: &dyn RepairFamily,
        templates: &[&CandidateTemplate],
    ) -> (Vec<DurableStudyCandidate>, crate::repair::RepairReport) {
        let candidates: Vec<DurableStudyCandidate> =
            templates.iter().map(|t| t.complete(declarations(5)).unwrap()).collect();
        let report =
            repair(family, &candidates, RepairObjective::MinimizeCost, limits(), &ctx()).unwrap();
        (candidates, report)
    }

    fn singleton_classification(
        report: &crate::repair::RepairReport,
        candidate: &DurableStudyCandidate,
    ) -> RepairClassification {
        report
            .outcomes
            .iter()
            .find(|o| o.candidates.len() == 1 && o.candidates[0] == candidate.semantic_id())
            .unwrap()
            .classification
    }

    #[test]
    fn a_template_refuses_to_validate_until_the_caller_declares_the_study() {
        let family = BackdoorRepairFamily::try_new(
            dag(3, &[(0, 1), (2, 0), (2, 1)]),
            v(0),
            v(1),
            "pop",
            [v(0), v(1)],
            &[],
        )
        .unwrap();
        let generation = generate_candidates(&family.unresolved_obligations());
        let template = &generation.templates[0];
        let refused = template.validate().unwrap_err();
        assert_eq!(refused.detail, "study_candidate.declarations_required");
        // A blank declaration is refused by the candidate's own validation.
        let mut blank = declarations(5);
        blank.recruitment = Arc::from(" ");
        assert_eq!(template.complete(blank).unwrap_err().detail, "study_candidate.wrong_contract");
        let mut free = declarations(5);
        free.cost.units = 0;
        assert!(template.complete(free).is_err());
    }

    #[test]
    fn a_back_door_obligation_yields_an_observation_that_repair_verifies() {
        let family = BackdoorRepairFamily::try_new(
            dag(3, &[(0, 1), (2, 0), (2, 1)]),
            v(0),
            v(1),
            "pop",
            [v(0), v(1)],
            &[],
        )
        .unwrap();
        let obligations = family.unresolved_obligations();
        let generation = generate_candidates(&obligations);
        assert!(generation.unmatched.is_empty());
        assert_eq!(generation.templates.len(), 1);
        let template = &generation.templates[0];
        assert_eq!(template.kind, StudyKind::Observation);
        assert_eq!(template.measured.iter().map(|x| x.raw()).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!(template.joint_measurement);
        assert_eq!(template.addresses.len(), obligations.len());
        let (candidates, report) = verified_labels(&family, &[template]);
        assert_eq!(
            singleton_classification(&report, &candidates[0]),
            RepairClassification::VerifiedSufficient
        );
        // The obligation screen agrees: the completed candidate addresses it.
        assert!(report.outcomes.iter().any(|o| !o.addressed.is_empty()));
    }

    #[test]
    fn a_front_door_obligation_yields_a_study_of_the_mediator_set() {
        let family = FrontDoorRepairFamily::try_new(
            dag(4, &[(0, 1), (1, 2), (3, 0), (3, 2)]),
            v(0),
            v(2),
            "pop",
            [v(0), v(2)],
        )
        .unwrap();
        let generation = generate_candidates(&family.unresolved_obligations());
        let template = generation
            .templates
            .iter()
            .find(|t| t.measured.iter().map(|x| x.raw()).collect::<Vec<_>>() == vec![0, 1, 2])
            .unwrap();
        let (candidates, report) = verified_labels(&family, &[template]);
        assert_eq!(
            singleton_classification(&report, &candidates[0]),
            RepairClassification::VerifiedSufficient
        );
    }

    #[test]
    fn an_instrument_obligation_yields_instrument_and_natural_experiment_but_never_certifies() {
        let family = InstrumentRepairFamily::try_new(
            dag(4, &[(0, 1), (1, 2), (3, 1), (3, 2)]),
            v(1),
            v(2),
            "pop",
            [v(1), v(2)],
        )
        .unwrap();
        let generation = generate_candidates(&family.unresolved_obligations());
        let kinds: Vec<_> = generation.templates.iter().map(|t| t.kind).collect();
        assert!(kinds.contains(&StudyKind::Instrument));
        assert!(kinds.contains(&StudyKind::NaturalExperiment));
        // Relevance, exclusion and the effect restriction are never proposed a study.
        assert_eq!(generation.unmatched.len(), 3);
        assert!(
            generation
                .unmatched
                .iter()
                .all(|u| u.kind == EvidenceObligationKind::EstablishAssumption)
        );
        let instrument =
            generation.templates.iter().find(|t| t.kind == StudyKind::Instrument).unwrap();
        let (candidates, report) = verified_labels(&family, &[instrument]);
        assert_eq!(
            singleton_classification(&report, &candidates[0]),
            RepairClassification::NotCertified
        );
        assert!(report.best().is_none());
    }

    #[test]
    fn a_hedge_obligation_yields_experiments_that_repair_verifies() {
        let family = HedgeRepairFamily::try_new(
            admg(3, &[(0, 1), (1, 2)], &[(0, 2), (1, 2)]),
            v(0),
            v(2),
            "pop",
            [v(0), v(1), v(2)],
        )
        .unwrap();
        let generation = generate_candidates(&family.unresolved_obligations());
        assert_eq!(generation.templates.len(), 2);
        assert!(generation.templates.iter().all(|t| t.kind == StudyKind::Experiment));
        let refs: Vec<&CandidateTemplate> = generation.templates.iter().collect();
        let (candidates, report) = verified_labels(&family, &refs);
        for candidate in &candidates {
            assert_eq!(
                singleton_classification(&report, candidate),
                RepairClassification::VerifiedSufficient
            );
        }
    }

    #[test]
    fn a_sensitivity_obligation_yields_a_study_that_measures_the_named_confounder() {
        let finding =
            SensitivityFinding { method: SensitivityMethod::EValue, strength: 1.4, robust: false };
        let family = SensitivityRepairFamily::try_new(
            dag(3, &[(0, 1), (2, 0), (2, 1)]),
            v(0),
            v(1),
            "pop",
            [v(0), v(1), v(2)],
            finding,
        )
        .unwrap();
        let generation = generate_candidates(&family.unresolved_obligations());
        let template = &generation.templates[0];
        assert!(template.measured.contains(&family.confounder()));
        let (candidates, report) = verified_labels(&family, &[template]);
        assert_eq!(
            singleton_classification(&report, &candidates[0]),
            RepairClassification::VerifiedSufficient
        );
    }

    #[test]
    fn panel_obligations_yield_a_panel_whose_time_invariance_is_unmatched() {
        // T=0 -> Y=1 confounded by the unmeasured U=2 (a bidirected edge).
        let graph = Admg::from_dag(&dag(3, &[(0, 1), (2, 0), (2, 1)])).unwrap();
        let failed = antecedent_identify::IdentificationResult::not_identified(
            antecedent_core::CausalQuery::average_effect(
                antecedent_core::AverageEffectQuery::binary_ate(v(0), v(1)),
            ),
            antecedent_identify::DerivationTrace::default(),
            antecedent_core::AssumptionSet::new(),
            antecedent_identify::IdentificationPerformanceRecord::default(),
        );
        let obligations = crate::obligation_adapters::identification_failure_obligations(
            &failed,
            &graph,
            &[v(0), v(1)],
            "pop",
            None,
        )
        .unwrap();
        let generation = generate_candidates(&obligations);
        let panel = generation.templates.iter().find(|t| t.kind == StudyKind::Panel).unwrap();
        assert_eq!(panel.measured.iter().map(|x| x.raw()).collect::<Vec<_>>(), vec![0, 1]);
        assert!(generation.unmatched.iter().any(|u| u.route.as_ref() == "panel"
            && u.kind == EvidenceObligationKind::EstablishAssumption));
        assert!(panel.complete(declarations(7)).is_ok());
    }

    #[test]
    fn a_sample_increase_template_carries_the_obligations_minimum() {
        use antecedent_contract::{
            EvidenceObligationSpec, ObligationProvenance, ObligationRegime, ObligationScope,
        };
        let obligation = EvidenceObligation::try_new(EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::new(),
            kind: EvidenceObligationKind::IncreaseSample,
            scope: ObligationScope::Factor,
            variables: Arc::from([v(0)]),
            population: Some(Arc::from("pop")),
            regime: ObligationRegime::observational(false),
            reason: Arc::from("weak overlap"),
            required_slots: Arc::from([Arc::from("coordinate:0")]),
            min_additional_samples: Some(500),
            provenance: ObligationProvenance {
                family: Arc::from("support"),
                source: Arc::from("test"),
                proof_step: None,
            },
        })
        .unwrap();
        let generation = generate_candidates(&[obligation]);
        let template = &generation.templates[0];
        assert_eq!(template.kind, StudyKind::SampleIncrease);
        assert_eq!(template.minimum_sample_size, Some(500));
        let refused = template.complete(declarations(5)).unwrap_err();
        assert_eq!(refused.detail, "study_candidate.sample_below_obligation");
        let mut enough = declarations(5);
        enough.sample_size = 500;
        assert!(template.complete(enough).is_ok());
    }

    #[test]
    fn only_a_candidate_verified_on_its_own_unlocks_identification() {
        let family = BackdoorRepairFamily::try_new(
            dag(3, &[(0, 1), (2, 0), (2, 1)]),
            v(0),
            v(1),
            "pop",
            [v(0), v(1)],
            &[],
        )
        .unwrap();
        let generation = generate_candidates(&family.unresolved_obligations());
        let good = generation.templates[0].complete(declarations(5)).unwrap();
        // The same variable measured in another population: a name match, not a repair.
        let mut elsewhere = good.clone();
        elsewhere.label = Arc::from("other clinic");
        elsewhere.population = Arc::from("other");
        elsewhere.expected_evidence = elsewhere
            .expected_evidence
            .iter()
            .cloned()
            .map(|mut e| {
                e.population = Arc::from("other");
                e
            })
            .collect();
        let more_rows = DurableStudyCandidate {
            label: Arc::from("more rows"),
            kind: StudyKind::SampleIncrease,
            expected_evidence: Arc::from([]),
            ..good.clone()
        };
        let candidates = vec![good, elsewhere, more_rows];
        let report =
            repair(&family, &candidates, RepairObjective::MinimizeCost, limits(), &ctx()).unwrap();
        let unlocks = verified_unlocks(&report, &candidates, &[v(0), v(1)]);
        assert_eq!(unlocks.verified, vec![0]);
        assert_eq!(unlocks.variables, vec![v(2)]);
        assert_eq!(unlocks.unverified.len(), 2);
        assert!(unlocks.unlock_entry(QueryId::from_raw(0)).is_some());
        assert_eq!(unlocks.designs(&candidates).len(), 1);
    }
}
