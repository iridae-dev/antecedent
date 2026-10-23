//! Contract section of a cyclic family artifact and its independent check.
//!
//! A cyclic family answers a specified, fitted or experimentally recovered
//! feedback model. Its contract carries the same layered identities as an
//! analysis-result contract (target, identification, identification product,
//! program, inference binding, observation, data snapshot), each next to the
//! payload it digests, plus two cyclic layers: the `cyclic.solution.v1` tag of
//! the solve semantics, and, for an experimental recovery, the recovery digest
//! over the experiment data snapshot, the regime contract and the recovery
//! derivation.
//!
//! [`verify_cyclic_contract_against_body`] re-derives every digest from its
//! payload, recomputes the seal, binds the program slot through
//! [`crate::bound_identity::bind`] with the cyclic-solution tag, and binds the
//! contract to the request bytes it travels with. It does not execute the model
//! and grants no identification: a specified model stays model-conditional.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, IdentityDomain, SemanticDigest};
use serde::{Deserialize, Serialize};

use crate::bound_identity::{
    BoundIdentity, BoundRefusal, CYCLIC_SOLUTION_CONTRACT, CyclicSolution, ExpectedIdentity,
    IdentitySlot, SelfDescribedDigest, bind, facet_digest,
};
use crate::identity::{digest_wire, payload_digest};
use crate::scientific_target_wire::ScientificTargetWire;
use crate::{IoError, to_cbor};

use super::ContractIdentitiesWire;

/// Section id of the contract on a cyclic family artifact.
pub const CYCLIC_CONTRACT_SECTION: &str = "cyclic_family.contract";

/// Cyclic contract payload format. Bump only when the wire shape changes.
pub const CYCLIC_CONTRACT_FORMAT: u16 = 1;

/// Domain tag of a cyclic family request identity.
pub const CYCLIC_FAMILY_REQUEST_TAG: &str = "antecedent.cyclic_family.request.v1";

/// Layer one: what the family answers.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "form", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicTargetWire {
    /// World-indexed target built from the request's keyed worlds and its declared outcome.
    Worlds {
        /// Rehashable scientific target.
        target: Box<ScientificTargetWire>,
    },
    /// Keyed worlds answered for every model variable: one target per variable,
    /// in variable order.
    JointWorlds {
        /// Rehashable scientific targets.
        targets: Vec<ScientificTargetWire>,
    },
    /// A functional that is not indexed by keyed worlds.
    Functional {
        /// Functional kind.
        kind: String,
        /// Outcome variable, when the functional reads one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outcome: Option<u32>,
        /// Digest of the query that defines the functional.
        query: [u8; 32],
    },
}

/// One coefficient row a clamp design leaves unresolved, known before any fit.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CyclicUnresolvedWire {
    /// Equation whose incoming coefficients are not all covered.
    pub target: u32,
    /// Sources never clamped while `target` was passive.
    pub missing_clamps: Vec<u32>,
}

/// Layer two: identification of a feedback model. None of these is a
/// graphical identification product of an acyclic model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicIdentificationWire {
    /// The answer is conditional on the specified or fitted model; nothing is
    /// identified from observations.
    SpecifiedModelConditional {
        /// Digest of the model the answer is conditional on.
        model: [u8; 32],
    },
    /// Linear law recovered from declared clamp experiments.
    ExperimentalRecovery {
        /// Digest of the declared clamp design (regimes and unit mask).
        design: [u8; 32],
        /// Directions the design leaves unresolved. Rank deficiencies found by
        /// the fit are reported with the execution, never completed.
        unresolved: Vec<CyclicUnresolvedWire>,
    },
    /// Theorem-scoped identification of an input/output structural causal model
    /// on a cyclic graph, under its own premises.
    CyclicIoScm {
        /// Digest of the graphical source and its premises.
        source: [u8; 32],
    },
}

impl CyclicIdentificationWire {
    /// Stable name of the identification kind.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::SpecifiedModelConditional { .. } => "specified_model_conditional",
            Self::ExperimentalRecovery { .. } => "experimental_recovery",
            Self::CyclicIoScm { .. } => "cyclic_ioscm",
        }
    }
}

/// Program payload: the inferential commitments of the family.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CyclicProgramWire {
    /// Stable family name.
    pub family: String,
    /// How the answer is realized (`specified_evaluation`, `sampling_law`,
    /// `parameter_posterior`, ...).
    pub realization: String,
    /// Digest of the typed fit-dependency set.
    pub fit_dependencies: [u8; 32],
}

/// Inference-binding payload: the uncertainty method and its numeric settings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CyclicInferenceWire {
    /// Uncertainty method name; `none` for a specified evaluation.
    pub method: String,
    /// Digest of the method's settings (draws, seeds, priors, sampling policy).
    pub settings: [u8; 32],
}

/// Observation payload: the variables and experimental units the family reads.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CyclicObservationWire {
    /// Data modality (`specified`, `equilibrium_rows`, `clamp_experiments`,
    /// `trajectories`, `finite_law`, `supplied_draws`).
    pub modality: String,
    /// Model variable ids, in model order.
    pub variables: Vec<u32>,
    /// Retained experimental units, when the family reads clamp experiments.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub units: Vec<u32>,
}

/// Recovery payload of an experimental recovery.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CyclicRecoveryWire {
    /// Digest of the experiment samples (sources, columns, sampling units).
    pub experiment_data: [u8; 32],
    /// Digest of the declared regime contract and unit mask.
    pub regime_contract: [u8; 32],
    /// Digest of the recovery derivation: recipe, fit options, uncertainty
    /// method and the design-determined unresolved directions.
    pub derivation: [u8; 32],
}

/// Data-snapshot payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CyclicDataSnapshotWire {
    /// Digest of the data the family reads (rows, observations, laws or supplied draws).
    pub content: [u8; 32],
    /// Recovery payload, for an experimental recovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery: Option<CyclicRecoveryWire>,
}

/// Solve-semantics payload named by [`CYCLIC_SOLUTION_CONTRACT`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CyclicSolutionWire {
    /// Always [`CYCLIC_SOLUTION_CONTRACT`].
    pub contract: String,
    /// Stable family name.
    pub family: String,
    /// Digest of the compiled solve schedules (semantics, blocks, per world).
    pub schedules: [u8; 32],
    /// Certificate demand, selection or dynamic-law policy.
    pub policy: String,
}

/// Every payload a cyclic contract digests. Built by the family's producer.
#[derive(Clone, Debug, PartialEq)]
pub struct CyclicContractPayloads {
    /// Stable family name.
    pub family: String,
    /// Layer one.
    pub target: CyclicTargetWire,
    /// Layer two.
    pub identification: CyclicIdentificationWire,
    /// Program payload.
    pub program: CyclicProgramWire,
    /// Inference binding payload.
    pub inference_binding: CyclicInferenceWire,
    /// Observation payload.
    pub observation: CyclicObservationWire,
    /// Data-snapshot payload.
    pub data_snapshot: CyclicDataSnapshotWire,
    /// Solve-semantics payload.
    pub solution: CyclicSolutionWire,
}

/// Contract section of a cyclic family artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CyclicContractWire {
    /// Section format.
    pub format: u16,
    /// Stable family name.
    pub family: String,
    /// Identity of the request the contract is bound to.
    pub request_identity: String,
    /// Advertised layered identities.
    pub identities: ContractIdentitiesWire,
    /// Advertised [`IdentityDomain::CyclicSolution`] digest.
    pub cyclic_solution: [u8; 32],
    /// Advertised [`IdentityDomain::CyclicRecovery`] digest, for an experimental recovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery: Option<[u8; 32]>,
    /// The producer's description of the bound program facets
    /// ([`facet_digest`]); recomputed and never trusted.
    pub facets: [u8; 32],
    /// Seal over the format, family, request identity, identities, cyclic
    /// layers and facets.
    pub seal: [u8; 32],
    /// Layer-one payload.
    pub target: CyclicTargetWire,
    /// Layer-two payload.
    pub identification: CyclicIdentificationWire,
    /// Program payload.
    pub program: CyclicProgramWire,
    /// Inference-binding payload.
    pub inference_binding: CyclicInferenceWire,
    /// Observation payload.
    pub observation: CyclicObservationWire,
    /// Data-snapshot payload.
    pub data_snapshot: CyclicDataSnapshotWire,
    /// Solve-semantics payload.
    pub solution: CyclicSolutionWire,
}

/// Digests every layer of `payloads`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CyclicLayerDigests {
    /// Target.
    pub target: SemanticDigest,
    /// Identification premises.
    pub identification: SemanticDigest,
    /// Identification product.
    pub identification_product: SemanticDigest,
    /// Program.
    pub program: SemanticDigest,
    /// Inference binding.
    pub inference_binding: SemanticDigest,
    /// Observation contract.
    pub observation: SemanticDigest,
    /// Data snapshot.
    pub data_snapshot: SemanticDigest,
    /// `cyclic.solution.v1` tag.
    pub cyclic_solution: SemanticDigest,
    /// Recovery, for an experimental recovery.
    pub recovery: Option<SemanticDigest>,
}

impl CyclicLayerDigests {
    /// The seven contract layers as a core [`antecedent_core::ContractIdentities`].
    #[must_use]
    pub const fn contract_identities(&self) -> antecedent_core::ContractIdentities {
        antecedent_core::ContractIdentities::new(
            self.target,
            self.identification,
            Some(self.identification_product),
            Some(self.program),
            self.inference_binding,
            self.observation,
            self.data_snapshot,
        )
    }

    fn identities_wire(&self) -> ContractIdentitiesWire {
        ContractIdentitiesWire {
            target: *self.target.as_bytes(),
            identification: *self.identification.as_bytes(),
            identification_product: Some(*self.identification_product.as_bytes()),
            program: *self.program.as_bytes(),
            inference_binding: *self.inference_binding.as_bytes(),
            observation: *self.observation.as_bytes(),
            data_snapshot: *self.data_snapshot.as_bytes(),
            execution: None,
            score_reuse: None,
            target_weights: None,
            checked_aipw_rows: None,
        }
    }

    /// Program-slot identity the consumer expects, with the cyclic-solution tag.
    pub const fn expected_program(&self) -> ExpectedIdentity {
        ExpectedIdentity {
            payload: self.program,
            coordinates: self.target,
            prior: self.inference_binding,
            alignment: self.data_snapshot,
            cyclic_solution: CyclicSolution::Tag(self.cyclic_solution),
        }
    }
}

/// Digest every layer of `payloads`.
///
/// # Errors
///
/// A payload that does not encode, or a scientific target that does not rebuild
/// to the identity it advertises.
pub fn cyclic_layer_digests(
    payloads: &CyclicContractPayloads,
) -> Result<CyclicLayerDigests, IoError> {
    check_targets(&payloads.target)?;
    let target = digest_wire(IdentityDomain::Target, &payloads.target)?;
    let observation = digest_wire(IdentityDomain::Observation, &payloads.observation)?;
    let identification = digest_wire(
        IdentityDomain::Identification,
        &(&payloads.family, target.as_bytes(), observation.as_bytes(), &payloads.identification),
    )?;
    let identification_product =
        digest_wire(IdentityDomain::IdentificationProduct, &payloads.identification)?;
    let recovery = payloads
        .data_snapshot
        .recovery
        .as_ref()
        .map(|recovery| digest_wire(IdentityDomain::CyclicRecovery, recovery))
        .transpose()?;
    let data_snapshot = digest_wire(
        IdentityDomain::DataSnapshot,
        &(observation.as_bytes(), &payloads.data_snapshot),
    )?;
    let inference_binding =
        digest_wire(IdentityDomain::InferenceBinding, &payloads.inference_binding)?;
    let cyclic_solution = digest_wire(IdentityDomain::CyclicSolution, &payloads.solution)?;
    let program = digest_wire(
        IdentityDomain::Program,
        &(
            target.as_bytes(),
            identification.as_bytes(),
            identification_product.as_bytes(),
            cyclic_solution.as_bytes(),
            &payloads.program,
        ),
    )?;
    Ok(CyclicLayerDigests {
        target,
        identification,
        identification_product,
        program,
        inference_binding,
        observation,
        data_snapshot,
        cyclic_solution,
        recovery,
    })
}

fn check_targets(target: &CyclicTargetWire) -> Result<(), IoError> {
    match target {
        CyclicTargetWire::Worlds { target } => target.to_domain().map(drop),
        CyclicTargetWire::JointWorlds { targets } => {
            targets.iter().try_for_each(|target| target.to_domain().map(drop))
        }
        CyclicTargetWire::Functional { .. } => Ok(()),
    }
}

/// Program identity the carrier presents for `identities`, `cyclic_solution` and `facets`.
fn presented_program(
    identities: &ContractIdentitiesWire,
    cyclic_solution: [u8; 32],
    facets: Option<[u8; 32]>,
) -> BoundIdentity {
    BoundIdentity {
        slot: IdentitySlot::Program,
        payload: SemanticDigest::from_bytes(identities.program),
        coordinates: SemanticDigest::from_bytes(identities.target),
        prior: SemanticDigest::from_bytes(identities.inference_binding),
        alignment: SemanticDigest::from_bytes(identities.data_snapshot),
        cyclic_solution: CyclicSolution::Tag(SemanticDigest::from_bytes(cyclic_solution)),
        self_description: facets.map(|advertised| SelfDescribedDigest {
            advertised: SemanticDigest::from_bytes(advertised),
        }),
    }
}

#[derive(Serialize)]
struct SealView<'a> {
    format: u16,
    family: &'a str,
    request_identity: &'a str,
    identities: &'a ContractIdentitiesWire,
    cyclic_solution: &'a [u8; 32],
    recovery: Option<&'a [u8; 32]>,
    facets: &'a [u8; 32],
}

fn seal_of(contract: &CyclicContractWire) -> Result<[u8; 32], IoError> {
    let bytes = to_cbor(&SealView {
        format: contract.format,
        family: &contract.family,
        request_identity: &contract.request_identity,
        identities: &contract.identities,
        cyclic_solution: &contract.cyclic_solution,
        recovery: contract.recovery.as_ref(),
        facets: &contract.facets,
    })?;
    Ok(payload_digest("cyclic_family.contract.seal", &bytes))
}

/// Identity of a cyclic family request from its encoded bytes.
#[must_use]
pub fn cyclic_request_identity(request_bytes: &[u8]) -> String {
    CanonicalWriter::digest_bytes_hex(CYCLIC_FAMILY_REQUEST_TAG, request_bytes)
}

/// Seal `payloads` into a contract bound to the request identified by `request_identity`.
///
/// # Errors
///
/// A payload that does not encode, or a scientific target that does not rebuild.
pub fn seal_cyclic_contract(
    payloads: CyclicContractPayloads,
    request_identity: &str,
) -> Result<CyclicContractWire, IoError> {
    if payloads.solution.contract != CYCLIC_SOLUTION_CONTRACT {
        return Err(IoError::Convert(format!(
            "a cyclic solution payload names `{CYCLIC_SOLUTION_CONTRACT}`"
        )));
    }
    let digests = cyclic_layer_digests(&payloads)?;
    let identities = digests.identities_wire();
    let cyclic_solution = *digests.cyclic_solution.as_bytes();
    let facets = *facet_digest(&presented_program(&identities, cyclic_solution, None)).as_bytes();
    let CyclicContractPayloads {
        family,
        target,
        identification,
        program,
        inference_binding,
        observation,
        data_snapshot,
        solution,
    } = payloads;
    let mut contract = CyclicContractWire {
        format: CYCLIC_CONTRACT_FORMAT,
        family,
        request_identity: request_identity.to_owned(),
        identities,
        cyclic_solution,
        recovery: digests.recovery.map(|digest| *digest.as_bytes()),
        facets,
        seal: [0; 32],
        target,
        identification,
        program,
        inference_binding,
        observation,
        data_snapshot,
        solution,
    };
    contract.seal = seal_of(&contract)?;
    Ok(contract)
}

impl CyclicContractWire {
    /// The payloads this contract carries.
    #[must_use]
    pub fn payloads(&self) -> CyclicContractPayloads {
        CyclicContractPayloads {
            family: self.family.clone(),
            target: self.target.clone(),
            identification: self.identification.clone(),
            program: self.program.clone(),
            inference_binding: self.inference_binding.clone(),
            observation: self.observation.clone(),
            data_snapshot: self.data_snapshot.clone(),
            solution: self.solution.clone(),
        }
    }

    /// Advertised identities as core [`antecedent_core::ContractIdentities`].
    #[must_use]
    pub fn contract_identities(&self) -> antecedent_core::ContractIdentities {
        let digest = SemanticDigest::from_bytes;
        antecedent_core::ContractIdentities::new(
            digest(self.identities.target),
            digest(self.identities.identification),
            self.identities.identification_product.map(digest),
            Some(digest(self.identities.program)),
            digest(self.identities.inference_binding),
            digest(self.identities.observation),
            digest(self.identities.data_snapshot),
        )
    }
}

fn bound_reason(refusal: BoundRefusal) -> &'static str {
    match refusal {
        BoundRefusal::SlotMismatch { .. } => "bound_identity.slot",
        BoundRefusal::PayloadMismatch { .. } => "bound_identity.program",
        BoundRefusal::CoordinatesMismatch { .. } => "bound_identity.target",
        BoundRefusal::PriorMismatch { .. } => "bound_identity.inference_binding",
        BoundRefusal::AlignmentMismatch { .. } => "bound_identity.data_snapshot",
        BoundRefusal::CyclicSolutionMismatch { .. } => "bound_identity.cyclic_solution",
        BoundRefusal::FalseSelfDescription { .. } => "bound_identity.false_self_description",
    }
}

/// Check `contract` against the request bytes it travels with, from bytes alone.
///
/// Every advertised identity is re-derived from its stored payload, the seal is
/// recomputed, the program slot is bound through [`bind`] with the recomputed
/// cyclic-solution tag as the retained expectation, and the request identity and
/// family are bound to `request_bytes`. An empty list means every check passed.
/// It does not execute the model; a consumer that holds the request type also
/// re-derives the payloads from the request and replays the execution.
#[must_use]
pub fn verify_cyclic_contract_against_body(
    contract: &CyclicContractWire,
    request_bytes: &[u8],
) -> Vec<Arc<str>> {
    let mut unresolved: Vec<&'static str> = Vec::new();
    if contract.format != CYCLIC_CONTRACT_FORMAT {
        unresolved.push("format");
    }
    if contract.solution.contract != CYCLIC_SOLUTION_CONTRACT {
        unresolved.push("solution.contract");
    }
    if contract.family != contract.program.family || contract.family != contract.solution.family {
        unresolved.push("family");
    }
    match cyclic_layer_digests(&contract.payloads()) {
        Ok(digests) => {
            let advertised = &contract.identities;
            let checks = [
                ("identities.target", advertised.target, digests.target),
                ("identities.identification", advertised.identification, digests.identification),
                ("identities.program", advertised.program, digests.program),
                (
                    "identities.inference_binding",
                    advertised.inference_binding,
                    digests.inference_binding,
                ),
                ("identities.observation", advertised.observation, digests.observation),
                ("identities.data_snapshot", advertised.data_snapshot, digests.data_snapshot),
                ("cyclic_solution", contract.cyclic_solution, digests.cyclic_solution),
            ];
            for (name, stored, derived) in checks {
                if stored != *derived.as_bytes() {
                    unresolved.push(name);
                }
            }
            if advertised.identification_product != Some(*digests.identification_product.as_bytes())
            {
                unresolved.push("identities.identification_product");
            }
            if contract.recovery != digests.recovery.map(|digest| *digest.as_bytes()) {
                unresolved.push("recovery");
            }
            if advertised.execution.is_some()
                || advertised.score_reuse.is_some()
                || advertised.target_weights.is_some()
                || advertised.checked_aipw_rows.is_some()
            {
                unresolved.push("identities.foreign_layer");
            }
            let presented =
                presented_program(advertised, contract.cyclic_solution, Some(contract.facets));
            if let Err(refusal) =
                bind(IdentitySlot::Program, &presented, Some(&digests.expected_program()))
            {
                unresolved.push(bound_reason(refusal));
            }
        }
        Err(_) => unresolved.push("payloads"),
    }
    match seal_of(contract) {
        Ok(seal) if seal == contract.seal => {}
        _ => unresolved.push("seal"),
    }
    if cyclic_request_identity(request_bytes) != contract.request_identity {
        unresolved.push("body.request_identity");
    }
    let family =
        serde_json::from_slice::<serde_json::Value>(request_bytes).ok().and_then(|value| {
            value.get("family").and_then(|family| family.as_str()).map(str::to_owned)
        });
    if family.as_deref() != Some(contract.family.as_str()) {
        unresolved.push("body.family");
    }
    unresolved.sort_unstable();
    unresolved.dedup();
    unresolved.into_iter().map(Arc::from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payloads(recovered: bool) -> CyclicContractPayloads {
        CyclicContractPayloads {
            family: "experimental_recovery".into(),
            target: CyclicTargetWire::Functional {
                kind: "interventional_distribution".into(),
                outcome: Some(1),
                query: [3; 32],
            },
            identification: if recovered {
                CyclicIdentificationWire::ExperimentalRecovery {
                    design: [4; 32],
                    unresolved: vec![CyclicUnresolvedWire { target: 1, missing_clamps: vec![0] }],
                }
            } else {
                CyclicIdentificationWire::SpecifiedModelConditional { model: [4; 32] }
            },
            program: CyclicProgramWire {
                family: "experimental_recovery".into(),
                realization: "sampling_law".into(),
                fit_dependencies: [5; 32],
            },
            inference_binding: CyclicInferenceWire {
                method: "bootstrap".into(),
                settings: [6; 32],
            },
            observation: CyclicObservationWire {
                modality: "clamp_experiments".into(),
                variables: vec![0, 1],
                units: vec![0, 1],
            },
            data_snapshot: CyclicDataSnapshotWire {
                content: [7; 32],
                recovery: recovered.then_some(CyclicRecoveryWire {
                    experiment_data: [8; 32],
                    regime_contract: [9; 32],
                    derivation: [10; 32],
                }),
            },
            solution: CyclicSolutionWire {
                contract: CYCLIC_SOLUTION_CONTRACT.into(),
                family: "experimental_recovery".into(),
                schedules: [11; 32],
                policy: "hard_clamp_worlds".into(),
            },
        }
    }

    fn request() -> Vec<u8> {
        br#"{"family":"experimental_recovery","outcome":1}"#.to_vec()
    }

    #[test]
    fn a_sealed_contract_verifies_against_its_request() {
        let body = request();
        let contract =
            seal_cyclic_contract(payloads(true), &cyclic_request_identity(&body)).unwrap();
        assert!(verify_cyclic_contract_against_body(&contract, &body).is_empty());
        assert!(contract.recovery.is_some());
        let bytes = to_cbor(&contract).unwrap();
        let decoded: CyclicContractWire = crate::from_cbor(&bytes).unwrap();
        assert_eq!(decoded, contract);
    }

    #[test]
    fn the_recovery_layer_binds_the_regime_contract_and_the_data_snapshot() {
        let body = request();
        let identity = cyclic_request_identity(&body);
        let base = seal_cyclic_contract(payloads(true), &identity).unwrap();
        let mut changed = payloads(true);
        if let Some(recovery) = &mut changed.data_snapshot.recovery {
            recovery.regime_contract = [99; 32];
        }
        let moved = seal_cyclic_contract(changed, &identity).unwrap();
        assert_ne!(moved.recovery, base.recovery);
        assert_ne!(moved.identities.data_snapshot, base.identities.data_snapshot);
        // The regime contract belongs to the data-snapshot layer; the program layer describes
        // the solver and is unaffected by it.
        assert_eq!(moved.identities.program, base.identities.program);
        let specified = seal_cyclic_contract(payloads(false), &identity).unwrap();
        assert_eq!(specified.recovery, None);
        assert_ne!(specified.identities.identification, base.identities.identification);
    }

    #[test]
    fn a_tampered_contract_is_refused() {
        let body = request();
        let contract =
            seal_cyclic_contract(payloads(true), &cyclic_request_identity(&body)).unwrap();

        let mut payload = contract.clone();
        payload.inference_binding.method = "posterior".into();
        let reasons = verify_cyclic_contract_against_body(&payload, &body);
        assert!(
            reasons.iter().any(|reason| &**reason == "identities.inference_binding"),
            "{reasons:?}"
        );

        let mut tag = contract.clone();
        tag.cyclic_solution = [0; 32];
        let reasons = verify_cyclic_contract_against_body(&tag, &body);
        assert!(reasons.iter().any(|reason| &**reason == "cyclic_solution"), "{reasons:?}");
        assert!(reasons.iter().any(|reason| &**reason == "seal"), "{reasons:?}");

        let mut resealed = contract.clone();
        resealed.solution.policy = "selected_root".into();
        resealed.cyclic_solution =
            *digest_wire(IdentityDomain::CyclicSolution, &resealed.solution).unwrap().as_bytes();
        resealed.seal = seal_of(&resealed).unwrap();
        let reasons = verify_cyclic_contract_against_body(&resealed, &body);
        assert!(
            reasons.iter().any(|reason| reason.starts_with("bound_identity.")),
            "a consistent re-seal of the tag is still refused by the program binding: {reasons:?}"
        );

        let other = br#"{"family":"experimental_recovery","outcome":0}"#;
        let reasons = verify_cyclic_contract_against_body(&contract, other);
        assert!(reasons.iter().any(|reason| &**reason == "body.request_identity"), "{reasons:?}");
    }
}
