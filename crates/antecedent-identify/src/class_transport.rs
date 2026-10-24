//! sID transport identification over the members of an equivalence class.
//!
//! A `TransportQuery` reads a selection diagram, which is an ADMG with the declared selection
//! targets. A CPDAG assumes causal sufficiency, so each member of its equivalence class is a
//! DAG and its selection diagram is the DAG's ADMG embedding; a PAG allows latent confounding,
//! and a MAG completion fixes the latent projection only up to its m-separation model, so each
//! completion is read through every ADMG that shares that model (see `pag_path_specific`).
//! sID decides every member on its own graph.
//!
//! The class is read as follows.
//!
//! * Members that all certify the same transport formula identify the transported quantity by
//!   it: the class determines the quantity even though it may not determine the graph.
//! * Members that certify different formulas leave the quantity an identified set: one case
//!   per distinct formula.
//! * A member sID does not certify makes its completion unidentified: no formula is then
//!   correct for every model the class represents, and the completion carries unidentified
//!   mass.
//! * A candidate search that stops at its bound has not decided the members; the completion
//!   is refused with a search-bounded diagnostic, never read as an identified formula.
//!
//! Every envelope case is returned with the sID outcome that produced it, index-aligned, so a
//! caller can evaluate the certified formula of each case.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, TransportQuery,
    VariableId,
};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{Admg, Cpdag, Dag, Pag, SelectionDiagram};

use crate::envelope::{GraphFeature, IdentificationEnvelope};
use crate::error::IdentificationError;
use crate::generalized::GeneralizedAdjustmentIdentifier;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};
use crate::transport::{NotCertifiedCertificate, TransportIdentification, TransportIdentifier};
use crate::transport_lower::{bind_transport_derivation, lower_transport_mean};

/// Diagnostic code of a completion whose ADMG candidates could not all be enumerated within
/// the search bound. The refusal it accompanies is a bound, not a statement about the class.
pub const PAG_TRANSPORT_SEARCH_BOUNDED_DIAGNOSTIC_CODE: &str =
    "identify.transport_pag.search_bounded";

/// Method tag of a class case's estimand: the transported quantity, whatever rule certified
/// the formula, so cases that certify the same formula agree on the estimand and the formula
/// text alone tells different formulas apart.
pub const CLASS_TRANSPORT_METHOD: &str = "transport.sid";

/// Reason of the proof recorded for a completion whose ADMG candidates exceeded the bound.
pub const CLASS_TRANSPORT_CANDIDATES_BOUNDED_REASON: &str =
    "transport.sid.class_candidates_bounded";

/// A class envelope of transport identifications with the sID outcome of every case.
#[derive(Clone, Debug)]
pub struct ClassTransportEnvelope<G> {
    /// The class envelope: one case per completion (CPDAG) or per distinct certified formula
    /// of a completion (PAG), with the unidentified mass preserved.
    pub envelope: IdentificationEnvelope<G>,
    /// The sID outcome of each case of [`Self::envelope`], in case order.
    pub proofs: Vec<TransportIdentification>,
}

/// sID on one member graph with the declared selection targets.
fn identify_member(
    admg: &Admg,
    selection_targets: &[VariableId],
    query: &TransportQuery,
) -> Result<TransportIdentification, IdentificationError> {
    let diagram = SelectionDiagram::try_new(admg.clone(), selection_targets.to_vec())
        .map_err(|error| IdentificationError::msg(error.to_string()))?;
    TransportIdentifier::new().identify(&diagram, query)
}

/// The identification a case states from its sID outcome: the certified formula lowered to an
/// expression, or a refusal carrying the certificate's reason.
fn case_result(query: &TransportQuery, proof: &TransportIdentification) -> IdentificationResult {
    let causal_query = CausalQuery::Transport(query.clone());
    let pair = query.response.functional.primary_pair();
    match (proof, pair) {
        (
            TransportIdentification::Transportable { formula, certificate },
            Some((treatment, outcome)),
        ) => {
            let premises =
                certificate.premises.iter().map(AsRef::as_ref).collect::<Vec<_>>().join("; ");
            let mut arena = CausalExprArena::new();
            let functional = lower_transport_mean(&mut arena, formula, outcome);
            bind_transport_derivation(
                &mut arena,
                functional,
                certificate,
                format!("sID: treatment={treatment:?} outcome={outcome:?}; {premises}"),
            );
            let estimand = IdentifiedEstimand::new(
                CLASS_TRANSPORT_METHOD,
                Arc::from([]),
                Arc::from([]),
                Arc::from([]),
                functional,
                None,
            );
            let mut assumptions = AssumptionSet::default();
            assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom {
                    id: Arc::from(certificate.rule.as_ref()),
                    description: Arc::from(if premises.is_empty() {
                        "Structural transportability under the implemented sID subset (Direct or S-admissible standardize)."
                            .to_owned()
                    } else {
                        premises.clone()
                    }),
                },
                source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("transport.sid") },
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
            let mut derivation = DerivationTrace::default();
            derivation.push(certificate.rule.as_ref(), premises);
            IdentificationResult::identified(
                causal_query,
                vec![estimand],
                arena,
                derivation,
                assumptions,
                IdentificationPerformanceRecord::default(),
            )
        }
        (proof, _) => {
            let (reason, message): (Arc<str>, Arc<str>) = match proof {
                TransportIdentification::NotCertified(certificate) => {
                    (Arc::clone(&certificate.reason), Arc::clone(&certificate.message))
                }
                TransportIdentification::MissingEvidence(certificate) => {
                    (Arc::clone(&certificate.reason), Arc::clone(&certificate.message))
                }
                TransportIdentification::Transportable { .. } => (
                    Arc::from("transport.sid.no_treatment_outcome_pair"),
                    Arc::from("the inner response names no treatment/outcome pair"),
                ),
            };
            let mut derivation = DerivationTrace::default();
            derivation.push(reason.as_ref(), message.to_string());
            let mut result = IdentificationResult::not_identified(
                causal_query,
                derivation,
                AssumptionSet::default(),
                IdentificationPerformanceRecord::default(),
            );
            result.diagnostics.push(Diagnostic::new(
                "identify.class_transport.not_certified",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!("{reason}: {message}"),
            ));
            result
        }
    }
}

/// Canonical text of a case's functional, the only form comparable across cases.
fn functional_form(result: &IdentificationResult) -> Option<String> {
    let estimand = result.estimands.first()?;
    ((estimand.functional.raw() as usize) < result.arena.len())
        .then(|| result.arena.pretty(estimand.functional))
}

impl GeneralizedAdjustmentIdentifier {
    /// Identify a transport query in every DAG completion of a CPDAG.
    ///
    /// A CPDAG assumes causal sufficiency, so each completion is a DAG read as the ADMG with no
    /// bidirected edge; sID decides it on its own, with `selection_targets` as the variables
    /// whose mechanism may differ between populations. A completion sID does not certify
    /// carries unidentified mass, and completions that certify different formulas leave the
    /// quantity partially identified over the class.
    ///
    /// # Errors
    ///
    /// A selection target outside the graph, or a query sID does not read.
    pub fn identify_cpdag_transport_envelope(
        &self,
        cpdag: &Cpdag,
        selection_targets: &[VariableId],
        query: &TransportQuery,
    ) -> Result<ClassTransportEnvelope<Dag>, IdentificationError> {
        let mut proofs = Vec::new();
        let envelope = self.cpdag_envelope_with(cpdag, |dag| {
            let admg =
                Admg::from_dag(dag).map_err(|error| IdentificationError::msg(error.to_string()))?;
            let proof = identify_member(&admg, selection_targets, query)?;
            let result = case_result(query, &proof);
            proofs.push(proof);
            Ok(result)
        })?;
        Ok(ClassTransportEnvelope { envelope, proofs })
    }

    /// Identify a transport query over the ADMGs a PAG leaves open.
    ///
    /// Each MAG completion is read through every ADMG that shares its m-separation model:
    /// completions whose candidates all certify one formula identify the quantity by it,
    /// candidates that certify different formulas yield one case per distinct formula, and a
    /// candidate sID does not certify leaves the completion unidentified.
    ///
    /// # Errors
    ///
    /// A selection target outside the graph, a query sID does not read, or a completion that
    /// fails to prepare.
    pub fn identify_pag_transport_envelope(
        &self,
        pag: &Pag,
        selection_targets: &[VariableId],
        query: &TransportQuery,
    ) -> Result<ClassTransportEnvelope<Pag>, IdentificationError> {
        let mut proofs = Vec::new();
        let mut completions = 0usize;
        let mut candidates = 0usize;
        let mut distinct = 0usize;
        let mut envelope = self.pag_envelope_with_candidates(pag, |mag| {
            completions += 1;
            let found = self.admg_candidates_of_mag(mag)?;
            candidates += found.members.len();
            if found.bounded {
                let proof = TransportIdentification::NotCertified(NotCertifiedCertificate {
                    reason: Arc::from(CLASS_TRANSPORT_CANDIDATES_BOUNDED_REASON),
                    witness: Arc::from([]),
                    message: Arc::from(
                        "the ADMGs that share this MAG's m-separation model could not all be \
                         enumerated within the search bound; no non-transportability claim is made",
                    ),
                });
                let mut result = case_result(query, &proof);
                result.diagnostics.push(Diagnostic::new(
                    PAG_TRANSPORT_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
                    DiagnosticKind::Execution,
                    DiagnosticSeverity::Warning,
                    "the ADMGs that share this MAG's m-separation model could not all be \
                     enumerated within the search bound, so the transport formula was not decided \
                     on every one of them; the completion is refused, not read as identified",
                ));
                proofs.push(proof);
                return Ok(vec![result]);
            }
            let mut forms: Vec<Option<String>> = Vec::new();
            let mut results = Vec::new();
            let mut completion_proofs = Vec::new();
            for admg in &found.members {
                let proof = identify_member(admg, selection_targets, query)?;
                let mut result = case_result(query, &proof);
                if !matches!(proof, TransportIdentification::Transportable { .. }) {
                    result.diagnostics.push(Diagnostic::new(
                        "identify.transport_pag.unidentified_candidate",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        format!(
                            "one of the {} ADMG(s) with this MAG leaves the transport formula \
                             uncertified; no formula is correct for every model the MAG represents",
                            found.members.len()
                        ),
                    ));
                    proofs.push(proof);
                    return Ok(vec![result]);
                }
                let form = functional_form(&result);
                if forms.contains(&form) {
                    continue;
                }
                forms.push(form);
                results.push(result);
                completion_proofs.push(proof);
            }
            distinct += results.len();
            proofs.extend(completion_proofs);
            Ok(results)
        })?;
        envelope.push_features([GraphFeature {
            kind: Arc::from("pag_admg_candidates"),
            detail: Arc::from(format!(
                "{completions} MAG completion(s) read through {candidates} ADMG candidate(s) \
                 sharing their m-separation models; {distinct} distinct transport formula(s)"
            )),
        }]);
        Ok(ClassTransportEnvelope { envelope, proofs })
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery};
    use antecedent_graph::DenseNodeId;

    use super::*;
    use crate::result::IdentificationStatus;
    use crate::test_support::v;

    fn query() -> TransportQuery {
        let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(1),
            treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
        });
        TransportQuery::new(response, "source", "target", [v(0)])
    }

    /// The undirected chain `2 - 0 - 1`: three DAG completions (`2 <- 0 -> 1`, `2 -> 0 -> 1` and
    /// `2 <- 0 <- 1`), none with a collider.
    fn open_chain() -> Cpdag {
        let mut cpdag = Cpdag::with_variables(3);
        cpdag.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        cpdag.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        cpdag
    }

    /// The selection target `2` is not an ancestor of the outcome `1` once the treatment `0` is
    /// held fixed, in any completion: sID certifies the same direct formula in every member, so
    /// the class agrees and identifies the quantity by it.
    #[test]
    fn a_cpdag_whose_members_certify_one_formula_agrees() {
        let identifier = GeneralizedAdjustmentIdentifier::new();
        let class =
            identifier.identify_cpdag_transport_envelope(&open_chain(), &[v(2)], &query()).unwrap();
        assert_eq!(class.envelope.cases.len(), class.proofs.len());
        assert_eq!(class.envelope.cases.len(), 3, "the chain has three completions");
        assert!(class.envelope.unidentified_weight.0 <= f64::EPSILON);
        assert!(
            class
                .proofs
                .iter()
                .all(|proof| matches!(proof, TransportIdentification::Transportable { .. }))
        );
        assert_eq!(class.envelope.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(class.envelope.invariant.is_some());
    }
}
