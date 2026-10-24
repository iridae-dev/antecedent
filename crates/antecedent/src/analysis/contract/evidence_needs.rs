//! Evidence that would resolve a failed identification, as next actions.
//!
//! A contract that holds a not-identified product knows more than the refusal:
//! the graph says which variable to measure, which experiment to run, whether an
//! instrument or repeated measurements would help. This module derives those
//! verified [`EvidenceObligation`]s once, when the contract is compiled, and
//! turns them into the evidence-gathering [`NextAction`]s of the capability
//! report. Cheap inspection holds no identification product and so derives
//! nothing. An action is advice, never executed, and an assumption obligation
//! (an instrument's validity, the time-invariance a panel relies on) yields no
//! action that could satisfy it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_contract::{EvidenceObligation, EvidenceObligationKind, NextAction};
use antecedent_core::{IdentificationStatus, VariableId};
use antecedent_design::obligation_adapters::identification_failure_obligations;
use antecedent_design::{LatentCause, latent_expansion};
use antecedent_graph::Admg;
use antecedent_identify::IdentificationResult;

use super::{Study, data_schema};

/// The population label obligations of a compiled study carry: the population
/// the study's data were collected in.
const STUDY_POPULATION: &str = "study";

type Needs = (Arc<[EvidenceObligation]>, Arc<[NextAction]>);

fn none() -> Needs {
    (Arc::from([]), Arc::from([]))
}

/// The graph the identification ran on, when it is a directed or bidirected one.
fn failure_graph(study: &Study) -> Option<Admg> {
    study
        .graph
        .as_admg()
        .cloned()
        .or_else(|| study.graph.as_dag().and_then(|dag| Admg::from_dag(dag).ok()))
}

fn variable_name(names: &[Arc<str>], latents: &[LatentCause], variable: VariableId) -> String {
    if let Some(name) = names.get(variable.as_usize()) {
        return name.to_string();
    }
    let of = |v: VariableId| {
        names.get(v.as_usize()).map_or_else(|| format!("v{}", v.raw()), ToString::to_string)
    };
    latents.iter().find(|latent| latent.variable == variable).map_or_else(
        || format!("v{}", variable.raw()),
        |latent| format!("common_cause_of_{}_{}", of(latent.between.0), of(latent.between.1)),
    )
}

/// The obligations of a not-identified product and the actions they imply.
/// Anything else (an identified product, no product, a graph class the
/// identifiers here do not read, an unreadable query) derives nothing.
pub(super) fn derive(study: &Study, cached: Option<&IdentificationResult>) -> Needs {
    let Some(result) = cached.filter(|r| r.status == IdentificationStatus::NotIdentified) else {
        return none();
    };
    let Some(graph) = failure_graph(study) else {
        return none();
    };
    let names: Vec<Arc<str>> =
        data_schema(&study.data).variables().iter().map(|v| Arc::clone(&v.name)).collect();
    let observed: Vec<VariableId> = graph
        .nodes()
        .iter()
        .filter_map(|node| match node {
            antecedent_core::NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .filter(|variable| variable.as_usize() < names.len())
        .collect();
    let Ok(obligations) =
        identification_failure_obligations(result, &graph, &observed, STUDY_POPULATION, None)
    else {
        return none();
    };
    let latents = latent_expansion(&graph).map(|(_, latents)| latents).unwrap_or_default();
    let actions = evidence_next_actions(&obligations, &observed, &|variable| {
        variable_name(&names, &latents, variable)
    });
    (Arc::from(obligations), Arc::from(actions))
}

/// The evidence-gathering next actions of `obligations`, deterministic and
/// de-duplicated:
///
/// * `measure_variable:<name>` for every variable a joint-law or measurement
///   obligation names that `observed` does not hold;
/// * `run_experiment:<name>` for every variable an intervene obligation sets;
/// * `find_instrument` for any instrument obligation, including the one that
///   records that the graph has no instrument;
/// * `collect_panel` for the repeated-measurement alternative.
///
/// An `EstablishAssumption` obligation produces no action of its own: no study
/// satisfies it.
pub(crate) fn evidence_next_actions(
    obligations: &[EvidenceObligation],
    observed: &[VariableId],
    name_of: &dyn Fn(VariableId) -> String,
) -> Vec<NextAction> {
    let mut actions: BTreeMap<String, String> = BTreeMap::new();
    for obligation in obligations {
        let family = obligation.provenance.family.as_ref();
        let step = obligation.provenance.proof_step.as_deref().unwrap_or("unrecorded");
        if family == "instrument" {
            actions.entry("find_instrument".to_owned()).or_insert_with(|| {
                format!(
                    "find an instrument for the treatment: a variable that moves it without \
                     reaching the outcome except through it (proof step {step}); its validity \
                     stays an assumption"
                )
            });
            continue;
        }
        match obligation.kind {
            EvidenceObligationKind::EstablishAssumption => {}
            EvidenceObligationKind::Intervene => {
                for target in obligation.regime.interventions.iter() {
                    let name = name_of(*target);
                    actions.entry(format!("run_experiment:{name}")).or_insert_with(|| {
                        format!(
                            "run an experiment that sets {name} and measures the outcome \
                             (obligation {}, proof step {step})",
                            obligation.id
                        )
                    });
                }
            }
            _ if family == "panel" => {
                actions.entry("collect_panel".to_owned()).or_insert_with(|| {
                    format!(
                        "collect repeated measurements of the same units (obligation {}, proof \
                         step {step}); the unmeasured causes must be constant over the window",
                        obligation.id
                    )
                });
            }
            _ => {
                for variable in obligation.variables.iter().filter(|v| !observed.contains(v)) {
                    let name = name_of(*variable);
                    actions.entry(format!("measure_variable:{name}")).or_insert_with(|| {
                        format!(
                            "measure {name} jointly with the treatment and the outcome so the \
                             effect can be identified (obligation {}, proof step {step})",
                            obligation.id
                        )
                    });
                }
            }
        }
    }
    actions.into_iter().map(|(id, description)| NextAction::new(id, description)).collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_contract::{
        EvidenceObligationSpec, ObligationProvenance, ObligationRegime, ObligationScope,
    };

    use super::*;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn obligation(
        kind: EvidenceObligationKind,
        family: &str,
        variables: &[u32],
        interventions: &[u32],
        step: &str,
    ) -> EvidenceObligation {
        let assumption = kind == EvidenceObligationKind::EstablishAssumption;
        EvidenceObligation::try_new(EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::new(),
            kind,
            scope: if assumption { ObligationScope::Program } else { ObligationScope::Factor },
            variables: variables.iter().copied().map(v).collect::<Vec<_>>().into(),
            population: (!assumption).then(|| Arc::from("study")),
            regime: ObligationRegime {
                interventions: interventions.iter().copied().map(v).collect::<Vec<_>>().into(),
                conditioned_on: Arc::from([]),
                joint: variables.len() > 1,
            },
            reason: Arc::from("owed"),
            required_slots: Arc::from([Arc::from(format!("{family}:{step}"))]),
            min_additional_samples: None,
            provenance: ObligationProvenance {
                family: Arc::from(family),
                source: Arc::from("contract"),
                proof_step: Some(Arc::from(step)),
            },
        })
        .unwrap()
    }

    fn ids(actions: &[NextAction]) -> Vec<&str> {
        actions.iter().map(|a| a.id.as_ref()).collect()
    }

    fn names(variable: VariableId) -> String {
        ["t", "y", "u"].get(variable.as_usize()).map_or("latent", |name| *name).to_owned()
    }

    #[test]
    fn each_obligation_family_maps_to_its_evidence_gathering_action() {
        let obligations = [
            obligation(
                EvidenceObligationKind::ProvideJointLaw,
                "backdoor",
                &[0, 1, 2],
                &[],
                "backdoor.adjustment_set:0,1,2",
            ),
            obligation(EvidenceObligationKind::Intervene, "hedge", &[1], &[0], "hedge:f=[0,1]"),
            obligation(
                EvidenceObligationKind::EstablishAssumption,
                "instrument",
                &[],
                &[],
                "iv.existence",
            ),
            obligation(
                EvidenceObligationKind::ProvideJointLaw,
                "panel",
                &[0, 1],
                &[],
                "backdoor.unmeasured:2",
            ),
            obligation(
                EvidenceObligationKind::EstablishAssumption,
                "panel",
                &[],
                &[],
                "panel.time_invariance:2",
            ),
        ];
        let actions = evidence_next_actions(&obligations, &[v(0), v(1)], &names);
        assert_eq!(
            ids(&actions),
            vec!["collect_panel", "find_instrument", "measure_variable:u", "run_experiment:t"]
        );
    }

    /// T=0 -> W=1 -> Y=2 with T <-> Y and W <-> Y: observation cannot identify the
    /// effect, so the unmeasured causes (the back-door set, or the front-door set that
    /// adds the W-Y cause), an instrument, repeated measurement and experiments are
    /// all candidates.
    #[test]
    fn a_failed_identification_yields_every_evidence_gathering_action() {
        use antecedent_core::{AssumptionSet, AverageEffectQuery};
        use antecedent_graph::DenseNodeId;
        use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord};

        let mut graph = Admg::with_variables(3);
        for (from, to) in [(0, 1), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        for (a, b) in [(0, 2), (1, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let result = IdentificationResult::not_identified(
            antecedent_core::CausalQuery::average_effect(AverageEffectQuery::binary_ate(
                v(0),
                v(2),
            )),
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        );
        let observed = [v(0), v(1), v(2)];
        let obligations =
            identification_failure_obligations(&result, &graph, &observed, "study", None).unwrap();
        let (_, latents) = latent_expansion(&graph).unwrap();
        let by_name = |variable: VariableId| {
            let name = ["t", "w", "y"].get(variable.as_usize()).copied();
            match (name, latents.iter().find(|l| l.variable == variable)) {
                (Some(name), _) => name.to_owned(),
                (None, Some(l)) => format!(
                    "common_cause_of_{}_{}",
                    ["t", "w", "y"][l.between.0.as_usize()],
                    ["t", "w", "y"][l.between.1.as_usize()]
                ),
                (None, None) => "unknown".to_owned(),
            }
        };
        let actions = evidence_next_actions(&obligations, &observed, &by_name);
        assert_eq!(
            ids(&actions),
            vec![
                "collect_panel",
                "find_instrument",
                "measure_variable:common_cause_of_t_y",
                "measure_variable:common_cause_of_w_y",
                "run_experiment:t",
                "run_experiment:w",
            ]
        );
        // Every action traces back to an obligation that keeps its proof step.
        assert!(obligations.iter().all(|o| o.provenance.proof_step.is_some()));
    }

    #[test]
    fn a_pure_assumption_obligation_yields_no_action_and_observed_variables_are_not_remeasured() {
        let obligations = [
            obligation(
                EvidenceObligationKind::EstablishAssumption,
                "backdoor",
                &[],
                &[],
                "backdoor.existence",
            ),
            obligation(
                EvidenceObligationKind::ProvideJointLaw,
                "frontdoor",
                &[0, 1],
                &[],
                "frontdoor.mediator_set:1",
            ),
        ];
        assert!(evidence_next_actions(&obligations[..1], &[v(0), v(1)], &names).is_empty());
        assert!(evidence_next_actions(&obligations[1..], &[v(0), v(1)], &names).is_empty());
    }
}
