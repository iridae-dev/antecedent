// SPDX-License-Identifier: MIT OR Apache-2.0
//! Verify graph-posterior programs and published mixtures.

use super::{
    ATOM_WEIGHT_RELATIVE_TOLERANCE, AnalysisResultContractWire, AnalysisResultWire, ContractReason,
    SUMMED_MASS_TOLERANCE, close_replay_value, graph_posterior_cross_world,
    identification_product_digest_wire, summed_masses_agree,
};

/// Identification atoms (posterior samples) of a graph-posterior contract.
pub(super) fn posterior_identification_atoms(
    contract: &AnalysisResultContractWire,
) -> Option<&[crate::PosteriorAtomIdentityWire]> {
    match &contract.identification.as_ref()?.graph {
        crate::GraphIdentityWire::GraphPosterior { atoms, .. } => Some(atoms),
        _ => None,
    }
}

/// Check the per-atom programs of a static graph-posterior functional result against what the
/// contract itself attests: the digest chain over the mixing rule, weights and programs; the
/// atoms' samples, weights and structures against the identification payload; and each
/// program against the target schema. A tampered atom program, weight or mixing rule fails
/// here even when the payload digests and the section seal are recomputed.
pub(in crate::contract_section) fn verify_graph_posterior_programs(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    if !graph_posterior_cross_world(contract) {
        return;
    }
    let Some(programs) =
        contract.program.as_ref().and_then(|program| program.graph_posterior_programs.as_ref())
    else {
        unresolved.push(ContractReason::ProgramGraphPosteriorPrograms);
        return;
    };
    let chain_ok = programs.format == antecedent_core::IDENTITY_FORMAT
        && programs.mixing == crate::GRAPH_POSTERIOR_MIXING_RULE
        && crate::GraphPosteriorProgramsWire::compute_chain(&programs.mixing, &programs.atoms)
            .is_ok_and(|chain| chain == programs.chain);
    if !chain_ok {
        unresolved.push(ContractReason::ProgramGraphPosteriorPrograms);
    }
    let Some(samples) = posterior_identification_atoms(contract) else {
        unresolved.push(ContractReason::ProgramGraphPosteriorProgramsBinding);
        return;
    };
    let expected_variables: Vec<_> = contract
        .target
        .schema
        .variables
        .iter()
        .map(|variable| (variable.id, variable.name.clone()))
        .collect();
    let mut claimed = vec![false; samples.len()];
    let mut keys = std::collections::BTreeSet::new();
    let mut bound = true;
    for atom in &programs.atoms {
        bound &= keys.insert(atom.graph_key) && !atom.samples.is_empty();
        let mut weight = 0.0_f64;
        let mut structure = None;
        for &index in &atom.samples {
            let slot = usize::try_from(index).ok().filter(|slot| *slot < samples.len());
            let Some(slot) = slot else {
                bound = false;
                continue;
            };
            bound &= !std::mem::replace(&mut claimed[slot], true);
            weight += f64::from_bits(samples[slot].weight_bits);
            bound &= structure.is_none_or(|graph| graph == &samples[slot].graph);
            structure = Some(&samples[slot].graph);
        }
        let reported = f64::from_bits(atom.weight_bits);
        bound &= reported.is_finite()
            && (reported - weight).abs() <= ATOM_WEIGHT_RELATIVE_TOLERANCE * weight.abs().max(1.0);
        let class_atom = matches!(
            structure,
            Some(crate::PosteriorAtomGraphWire::Cpdag(_) | crate::PosteriorAtomGraphWire::Pag(_))
        );
        let single_atom = matches!(
            structure,
            Some(
                crate::PosteriorAtomGraphWire::Static(_)
                    | crate::PosteriorAtomGraphWire::Admg(_)
                    | crate::PosteriorAtomGraphWire::TemporalAdmg(_)
            )
        );
        // A single structure is answered by one program, a class by its identification
        // product; a refused or non-member atom carries neither.
        bound &= match (&atom.functional_program, &atom.identification_product) {
            (None, None) => true,
            (Some(program), None) => {
                single_atom
                    && program.source == program.executable
                    && program.variables == expected_variables
                    && crate::functional_program_from_wire(
                        program,
                        antecedent_expr::ProgramLimits::default(),
                    )
                    .is_ok()
            }
            (None, Some(product)) => {
                class_atom
                    && product.envelope.is_some()
                    && identification_product_digest_wire(product).is_ok()
            }
            (Some(_), Some(_)) => false,
        };
    }
    bound &= claimed.iter().all(|claimed| *claimed);
    if !bound {
        unresolved.push(ContractReason::ProgramGraphPosteriorProgramsBinding);
    }
}

/// Check the per-atom programs against the result body's structural mixture: the same atoms
/// with the same weights, an evaluated atom carried by a program or class product, and the
/// unidentified mass covering every atom the contract holds no program for.
pub(in crate::contract_section) fn verify_graph_posterior_mixture(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
    unresolved: &mut Vec<ContractReason>,
) {
    if !graph_posterior_cross_world(contract) {
        return;
    }
    let Some(programs) =
        contract.program.as_ref().and_then(|program| program.graph_posterior_programs.as_ref())
    else {
        return;
    };
    let Some(mixture) = body.structural_response.as_ref() else {
        unresolved.push(ContractReason::BodyResponse);
        return;
    };
    let mut consistent = mixture.atoms.len() == programs.atoms.len();
    let published_by_key = first_published_atoms(&mixture.atoms);
    let mut uncovered = 0.0_f64;
    for atom in &programs.atoms {
        let Some(published) = published_by_key.get(&atom.graph_key) else {
            consistent = false;
            continue;
        };
        let weight = f64::from_bits(atom.weight_bits);
        let carried = atom.functional_program.is_some() || atom.identification_product.is_some();
        consistent &= summed_masses_agree(published.weight, weight);
        consistent &= published.value.is_none() || carried;
        consistent &= carried
            || published.identification_status == crate::IdentificationStatusWire::NotIdentified;
        if !carried {
            uncovered += weight;
        }
    }
    consistent &= mixture.unidentified_mass + SUMMED_MASS_TOLERANCE >= uncovered;
    consistent &= mixture_recomputes(mixture);
    if !consistent {
        unresolved.push(ContractReason::ProgramGraphPosteriorProgramsBinding);
    }
}

/// Index atoms without changing which entry a duplicate key resolves to.
pub(super) fn first_published_atoms(
    atoms: &[crate::StructuralResponseAtomWire],
) -> std::collections::BTreeMap<u64, &crate::StructuralResponseAtomWire> {
    let mut by_key = std::collections::BTreeMap::new();
    for atom in atoms {
        by_key.entry(atom.graph_key).or_insert(atom);
    }
    by_key
}

/// Recompute the identified-mass-conditional scalar from the published atoms: the
/// weight-averaged value over the finite scalar atoms. Mixtures that carry no scalar
/// conditional, or whose atoms are not all scalar, are left to the structural checks.
fn mixture_recomputes(mixture: &crate::StructuralResponseMixtureWire) -> bool {
    let Some(crate::ResponseValueWire::Scalar(stored)) = mixture.conditional_on_identified.as_ref()
    else {
        return true;
    };
    let (mut weight, mut weighted) = (0.0_f64, 0.0_f64);
    for atom in &mixture.atoms {
        match atom.value.as_ref() {
            Some(crate::ResponseValueWire::Scalar(value)) if value.is_finite() => {
                weight += atom.weight;
                weighted += atom.weight * value;
            }
            Some(crate::ResponseValueWire::Scalar(_)) | None => {}
            Some(_) => return true,
        }
    }
    weight > 0.0 && close_replay_value(*stored, weighted / weight)
}
