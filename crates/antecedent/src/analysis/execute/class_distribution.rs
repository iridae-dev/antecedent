// SPDX-License-Identifier: MIT OR Apache-2.0
//! Interventional distributions over the completions of a CPDAG or the ADMGs a PAG leaves open.
//!
//! Each member of the class is decided on its own by the complete ID algorithm (IDC when the
//! query conditions) and every identified member is estimated by the same finite-discrete
//! plug-in (bootstrap, or the Dirichlet row posterior) as the supplied-graph cell. Members that
//! identify the same distribution keep that estimate and its uncertainty. Members that
//! disagree leave the distribution partially identified: the result publishes, atom by atom,
//! the lowest and the highest probability over the identified members
//! (`estimate.distribution.class_envelope`), with the enumeration-weighted mean beside it,
//! never a probability over members. A member the algorithm refuses (a hedge) is unidentified
//! mass; over a PAG one refused candidate refuses the whole distribution, since no functional
//! is then correct for every model the PAG represents.

use std::sync::Arc;

use antecedent_core::{Intervention, VariableId};
use antecedent_estimate::{
    DistributionAtom, FunctionalDistribution, InterventionalDistributionEstimate,
};

use super::{
    BootstrapAccounting, CausalError, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, ExecutionContext, IdentificationEnvelope, IdentificationStatus,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, Instant, PhysicalExecutionPlan, Strategy,
    StudyResult, TabularData, elapsed_ns, envelope_to_identification_result_for,
    identification_status_ok_for_case, report_identify_compute, select_estimand,
};
use crate::planner::RouteDefaults;

/// Stable code of the per-atom envelope disclosure.
const ENVELOPE_CODE: &str = "estimate.distribution.class_envelope";
/// Probabilities closer than this are one distribution.
const AGREEMENT: f64 = 1e-9;

/// The class structure a distribution is read over.
#[derive(Clone, Copy)]
pub(super) enum DistributionClass<'a> {
    Cpdag(&'a antecedent_graph::Cpdag),
    Pag(&'a antecedent_graph::Pag),
}

/// One atom across the identified members: its mean, lowest and highest probability.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct AtomEnvelope {
    pub(super) outcomes: Arc<[(VariableId, antecedent_core::Value)]>,
    pub(super) conditioning: Arc<[(VariableId, antecedent_core::Value)]>,
    pub(super) mean: f64,
    pub(super) lower: f64,
    pub(super) upper: f64,
}

/// Combine the probabilities of the identified members, weighted by `weights`, atom by atom.
/// An atom a member lacks has probability zero there.
pub(super) fn envelope_of_members(
    members: &[&[DistributionAtom]],
    weights: &[f64],
) -> Vec<AtomEnvelope> {
    let mut keys = Vec::new();
    for atoms in members {
        for atom in *atoms {
            let key = (Arc::clone(&atom.outcomes), Arc::clone(&atom.conditioning));
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    let total: f64 = weights.iter().sum();
    keys.into_iter()
        .map(|(outcomes, conditioning)| {
            let probabilities: Vec<f64> = members
                .iter()
                .map(|atoms| {
                    atoms
                        .iter()
                        .find(|a| a.outcomes == outcomes && a.conditioning == conditioning)
                        .map_or(0.0, |a| a.probability)
                })
                .collect();
            let mean = probabilities.iter().zip(weights).map(|(p, w)| p * w).sum::<f64>() / total;
            let lower = probabilities.iter().copied().fold(f64::INFINITY, f64::min);
            let upper = probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            AtomEnvelope { outcomes, conditioning, mean, lower, upper }
        })
        .collect()
}

/// Whether every atom of the envelope is one probability across the members.
pub(super) fn members_agree(envelope: &[AtomEnvelope]) -> bool {
    envelope.iter().all(|atom| atom.upper - atom.lower <= AGREEMENT)
}

fn envelope_diagnostic(
    envelope: &[AtomEnvelope],
    members: usize,
    class: &str,
    unidentified: f64,
) -> Diagnostic {
    let mut fields: Vec<(String, String)> = vec![
        ("members".to_owned(), members.to_string()),
        ("atoms".to_owned(), envelope.len().to_string()),
        ("unidentified_mass".to_owned(), unidentified.to_string()),
    ];
    for (index, atom) in envelope.iter().enumerate() {
        fields.push((format!("atom_{index}_lower"), atom.lower.to_string()));
        fields.push((format!("atom_{index}_upper"), atom.upper.to_string()));
        fields.push((format!("atom_{index}_mean"), atom.mean.to_string()));
    }
    Diagnostic::new(
        ENVELOPE_CODE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "the {class} leaves the distribution partially identified: {members} identified \
             member(s) disagree on {} atom(s). The published atoms are enumeration-weighted \
             means, not probabilities over members; the identified set of each atom is its \
             lowest and highest probability over the members",
            envelope.iter().filter(|a| a.upper - a.lower > AGREEMENT).count(),
        ),
    )
    .with_fields(fields)
}

impl super::Study {
    /// Identify the distribution in every member of the class, estimate the identified members
    /// and publish the class-level result.
    ///
    /// # Errors
    ///
    /// A validation suite other than `none`, a distribution no member identifies (or, over a
    /// PAG, one candidate leaves unidentified), or a plug-in that cannot evaluate a cell.
    #[allow(
        clippy::too_many_lines,
        reason = "one class distribution: identification, per-member estimates, envelope and finish run in sequence"
    )]
    pub(super) fn execute_class_distribution(
        &self,
        data: &TabularData,
        class: DistributionClass<'_>,
        query: &antecedent_core::InterventionalDistributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::static_path::ensure_admg_distribution_licensed(query, self.refute)?;
        let started = Instant::now();
        let (identifier_id, estimator_id) =
            physical.binding().resolve(RouteDefaults::DISTRIBUTION)?;
        if !matches!(estimator_id, EstimatorId::FunctionalDistribution) {
            return Err(CausalError::Compile {
                message: format!(
                    "Distribution execute requires estimator functional.distribution; got \
                     {estimator_id}"
                ),
            });
        }
        report_identify_compute(ctx);
        let causal_query = CausalQuery::Distribution(query.clone());
        // The envelope types differ by class; both reduce to the same case list.
        let (
            cases,
            status,
            identified_weight,
            unidentified_weight,
            capped,
            certificate,
            label,
            identification,
        ) = match class {
            DistributionClass::Cpdag(cpdag) => {
                let envelope = crate::strategy_table::identify_cpdag_distribution_in_context(
                    identifier_id,
                    cpdag,
                    &causal_query,
                    Some(ctx),
                )?;
                let certificate = crate::Identification::CpdagEnvelope {
                    envelope: envelope.clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                };
                let identification =
                    envelope_to_identification_result_for(&envelope, self.query.clone());
                (
                    case_results(&envelope),
                    envelope.status,
                    envelope.identified_weight.0,
                    envelope.unidentified_weight.0,
                    envelope.truncated_completions > 0,
                    certificate,
                    "CPDAG",
                    identification,
                )
            }
            DistributionClass::Pag(pag) => {
                let envelope = crate::strategy_table::identify_pag_distribution_in_context(
                    identifier_id,
                    pag,
                    &causal_query,
                    Some(ctx),
                )?;
                if envelope.unidentified_weight.0 > 0.0 {
                    return Err(CausalError::not_identified(
                        envelope.status,
                        envelope.truncated_completions > 0,
                        "PAG distribution not identified: an ADMG sharing the m-separation \
                             model of a MAG completion leaves it unidentified, so no functional \
                             is correct for every model the PAG represents",
                    ));
                }
                let certificate = crate::Identification::Envelope {
                    envelope: envelope.clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                };
                let identification =
                    envelope_to_identification_result_for(&envelope, self.query.clone());
                (
                    case_results(&envelope),
                    envelope.status,
                    envelope.identified_weight.0,
                    envelope.unidentified_weight.0,
                    envelope.truncated_completions > 0,
                    certificate,
                    "PAG",
                    identification,
                )
            }
        };
        if matches!(status, IdentificationStatus::NotIdentified) || identified_weight <= 0.0 {
            return Err(CausalError::not_identified(
                status,
                capped,
                "the class distribution is not identified (no identified mass in the envelope)",
            ));
        }

        let (row_resampling, sampling_design) = self.functional_row_resampling()?;
        let fitter = FunctionalDistribution {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalDistribution::new()
        }
        .with_row_resampling(row_resampling, sampling_design);
        let mut members: Vec<(
            f64,
            InterventionalDistributionEstimate,
            Option<antecedent_estimate::CausalPosterior>,
        )> = Vec::new();
        for (weight, result) in &cases {
            if !identification_status_ok_for_case(result.status) || result.estimands.is_empty() {
                continue;
            }
            let estimand = select_estimand(result, estimator_id)?;
            let prepared = fitter
                .prepare(data, query, &estimand, &result.arena, result.required_assumptions.clone())
                .map_err(CausalError::from)?;
            let (dist, posterior) =
                self.estimate_distribution(&fitter, &prepared, result.status, ctx)?;
            members.push((*weight, dist, posterior));
        }
        if members.is_empty() {
            return Err(CausalError::Compile {
                message: "the class distribution had no estimable identified member".into(),
            });
        }

        let atoms: Vec<&[DistributionAtom]> =
            members.iter().map(|(_, dist, _)| &dist.atoms[..]).collect();
        let weights: Vec<f64> = members.iter().map(|(weight, _, _)| *weight).collect();
        let envelope = envelope_of_members(&atoms, &weights);
        let agree = members_agree(&envelope);
        let mut diagnostics = Vec::new();
        let (dist, posterior) = if agree {
            // One distribution, with the uncertainty of its own fit.
            let (_, dist, posterior) = members.remove(0);
            (dist, posterior)
        } else {
            diagnostics.push(envelope_diagnostic(
                &envelope,
                members.len(),
                label,
                unidentified_weight,
            ));
            let total: f64 = weights.iter().sum();
            let mean = members.iter().map(|(w, dist, _)| w * dist.mean).sum::<f64>() / total;
            let first = &members[0].1;
            let mut mixed = first.clone();
            mixed.atoms = envelope
                .iter()
                .map(|atom| DistributionAtom {
                    outcomes: Arc::clone(&atom.outcomes),
                    conditioning: Arc::clone(&atom.conditioning),
                    probability: atom.mean,
                })
                .collect();
            mixed.mean = mean;
            mixed.se_analytic = f64::NAN;
            mixed.se_bootstrap = None;
            mixed.atom_uncertainty = Arc::from([]);
            mixed.mean_interval = None;
            (mixed, None)
        };
        let estimate = EffectEstimate::from_parts(
            dist.mean,
            dist.se_analytic,
            BootstrapAccounting {
                se: dist.se_bootstrap,
                replicates_ok: dist.bootstrap_replicates_ok,
                replicates_failed: dist.bootstrap_replicates_failed,
                cancelled: dist.bootstrap_cancelled,
                early_stopped: dist.bootstrap_early_stopped,
            },
            dist.assumptions.clone(),
            dist.overlap,
            None,
            dist.retained_memory_bytes,
        );
        diagnostics.extend(super::static_path::distribution_interval_diagnostics(&dist));
        let treatment =
            query.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
                || CausalError::Compile {
                    message: "distribution query missing intervention target".into(),
                },
            )?;
        let outcome = *query.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "distribution query missing outcome".into(),
        })?;
        // The combined identification lists the estimand of every identified member. The
        // published atoms are the members' own estimates (and, where they disagree, their
        // envelope), so the result is labelled by the first identified member's estimand.
        let estimand = match identification.estimands.as_slice() {
            [first, ..] => first.clone(),
            [] => select_estimand(&identification, estimator_id)?,
        };
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let cancelled = estimate.uncertainty.bootstrap.cancelled;
        let early_stopped = estimate.uncertainty.bootstrap.early_stopped;
        let result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(false)
            .extra_diagnostics(diagnostics)
            .distribution(Some(dist))
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_ok)
            .cancelled(cancelled)
            .early_stopped(early_stopped)
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        );
        Ok(self.attach_certificate(result, certificate))
    }
}

/// The weight and identification result of every case of an envelope.
fn case_results<G>(
    envelope: &IdentificationEnvelope<G>,
) -> Vec<(f64, antecedent_identify::IdentificationResult)> {
    envelope.cases.iter().map(|case| (case.weight.0, case.result.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::Value;

    fn atom(level: f64, probability: f64) -> DistributionAtom {
        DistributionAtom {
            outcomes: Arc::from([(VariableId::from_raw(1), Value::f64(level))]),
            conditioning: Arc::from([]),
            probability,
        }
    }

    #[test]
    fn members_that_disagree_publish_the_weighted_mean_and_the_identified_set_of_each_atom() {
        // Two members over a binary outcome: P(Y=1) is 0.2 and 0.6, weighted 1 : 3.
        let first = [atom(0.0, 0.8), atom(1.0, 0.2)];
        let second = [atom(0.0, 0.4), atom(1.0, 0.6)];
        let envelope = envelope_of_members(&[&first, &second], &[1.0, 3.0]);
        assert_eq!(envelope.len(), 2);
        let one = envelope.iter().find(|a| a.outcomes[0].1 == Value::f64(1.0)).unwrap();
        assert!((one.mean - (0.2 + 3.0 * 0.6) / 4.0).abs() < 1e-12);
        assert!((one.lower - 0.2).abs() < 1e-12 && (one.upper - 0.6).abs() < 1e-12);
        assert!(!members_agree(&envelope));
        // The means still sum to one: a mixture of distributions is a distribution.
        assert!((envelope.iter().map(|a| a.mean).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_atom_a_member_lacks_has_probability_zero_there() {
        let first = [atom(0.0, 1.0)];
        let second = [atom(0.0, 0.5), atom(1.0, 0.5)];
        let envelope = envelope_of_members(&[&first, &second], &[1.0, 1.0]);
        let one = envelope.iter().find(|a| a.outcomes[0].1 == Value::f64(1.0)).unwrap();
        assert!((one.lower - 0.0).abs() < 1e-12 && (one.upper - 0.5).abs() < 1e-12);
    }

    #[test]
    fn identical_members_agree() {
        let member = [atom(0.0, 0.3), atom(1.0, 0.7)];
        let envelope = envelope_of_members(&[&member, &member, &member], &[1.0, 1.0, 1.0]);
        assert!(members_agree(&envelope));
        assert!((envelope[1].mean - 0.7).abs() < 1e-12);
    }
}
