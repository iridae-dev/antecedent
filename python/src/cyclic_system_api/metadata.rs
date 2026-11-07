//! Projections of cyclic execution and contract metadata.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CyclicExecutionOwner, PyResult};

pub(super) fn origin_metadata(
    origin: &antecedent::analysis::RecoveredDrawOriginWire,
) -> serde_json::Value {
    use antecedent::analysis::RecoveredDrawOriginWire as O;
    match origin {
        O::Supplied => serde_json::json!({"origin":"supplied"}),
        O::AsymptoticNormal { draws, seed } => {
            serde_json::json!({"origin":"asymptotic_normal","draws":draws,"seed":seed})
        }
        O::Bootstrap { replicates, seed, design, rule } => {
            serde_json::json!({"origin":"bootstrap","replicates":replicates,"seed":seed,"design":design,"rule":rule})
        }
        O::Posterior { draws, burn_in, seed, .. } => {
            serde_json::json!({"origin":"posterior","draws":draws,"burn_in":burn_in,"seed":seed,"prior":"retained_in_source_declaration"})
        }
    }
}

impl CyclicExecutionOwner {
    pub(super) fn slot_metadata(
        &self,
        world: u32,
        row: usize,
        draw: usize,
        counterfactual: bool,
    ) -> PyResult<serde_json::Value> {
        use antecedent::analysis::CyclicFamilyAnswer as A;
        use antecedent::io::cyclic_family_artifact::CyclicWorldOutcomeAnswer as W;
        let position = self.position(world)?;
        let invalid = || crate::value_err("cyclic slot coordinate out of range");
        if let Some(system) = self.system() {
            if draw != 0 {
                return Err(invalid());
            }
            let value = if counterfactual {
                if row != 0 {
                    return Err(invalid());
                }
                system.counterfactuals.as_ref().and_then(|rows| rows.get(position))
            } else {
                system.worlds[position].get(row)
            }
            .ok_or_else(invalid)?;
            return Ok(solution_metadata(value));
        }
        let answer = &self.family().ok_or_else(invalid)?.answer;
        if counterfactual && !matches!(answer, A::EquilibriumActions(_)) {
            return Err(invalid());
        }
        match answer {
            A::Worlds(worlds) if draw == 0 => match &worlds[position].outcome {
                W::EquilibriumDraws { outcomes } => {
                    outcomes.get(row).map(equilibrium_metadata).ok_or_else(invalid)
                }
                W::GaussianLaw { residual_inf, block_diagnostics, .. } if row == 0 => Ok(
                    serde_json::json!({"status":"gaussian_law","residual_inf":residual_inf,"block_diagnostics":block_diagnostics}),
                ),
                W::Trajectory { trajectory } => trajectory
                    .times
                    .get(row)
                    .map(|time| serde_json::json!({"status":"trajectory","time":time}))
                    .ok_or_else(invalid),
                W::NoSolution { cause, next_attempt } if row == 0 => Ok(
                    serde_json::json!({"status":"no_solution","cause":cause,"next_attempt":next_attempt}),
                ),
                W::Withheld if row == 0 => Ok(serde_json::json!({"status":"withheld"})),
                _ => Err(invalid()),
            },
            A::EquilibriumActions(worlds) if draw == 0 => {
                let value = if counterfactual {
                    if row != 0 {
                        return Err(invalid());
                    }
                    worlds[position].counterfactual.as_ref()
                } else {
                    worlds[position].rows.get(row)
                }
                .ok_or_else(invalid)?;
                Ok(equilibrium_metadata(value))
            }
            A::ParameterPosterior { worlds, .. } => worlds[position]
                .get(draw)
                .and_then(|rows| rows.get(row))
                .map(equilibrium_metadata)
                .ok_or_else(invalid),
            A::FrequentistBootstrap { worlds, .. } => {
                match worlds[position].get(draw).ok_or_else(invalid)? {
                    Some(rows) => rows.get(row).map(equilibrium_metadata).ok_or_else(invalid),
                    None => Ok(serde_json::json!({"status":"failed_fit"})),
                }
            }
            A::JointCounterfactual { worlds, .. } | A::PartialCounterfactual { worlds, .. }
                if row == 0 =>
            {
                worlds[position].get(draw).map(equilibrium_metadata).ok_or_else(invalid)
            }
            A::RecoveredLinear(report) if row == 0 => {
                recovered_slot_metadata(&report.worlds[position], draw)
            }
            A::ExperimentalRecovery(report) if row == 0 => {
                recovered_slot_metadata(&report.worlds[position].answer, draw)
            }
            A::SdeAnalysis(report) if draw == 0 => report
                .simulation
                .times
                .get(row)
                .map(|time| serde_json::json!({"status":"ensemble_moments","time":time}))
                .ok_or_else(invalid),
            A::Ode(trajectory) if draw == 0 => trajectory
                .times
                .get(row)
                .map(|time| serde_json::json!({"status":"trajectory","time":time}))
                .ok_or_else(invalid),
            A::OuTransient(moments) if draw == 0 => moments
                .times
                .get(row)
                .map(|time| serde_json::json!({"status":"gaussian_law","time":time}))
                .ok_or_else(invalid),
            A::OuStationary(_) if draw == 0 && row == 0 => {
                Ok(serde_json::json!({"status":"stationary_covariance"}))
            }
            A::OuMechanisms { laws, .. } if draw == 0 && row == 0 => match &laws[position] {
                antecedent::estimate::ou_mechanisms::OuMechanismOutcome::Stationary(_) => {
                    Ok(serde_json::json!({"status":"gaussian_law"}))
                }
                antecedent::estimate::ou_mechanisms::OuMechanismOutcome::Failed {
                    failure,
                    detail,
                } => Ok(serde_json::json!({"status":"failed","failure":failure,"detail":detail})),
            },
            _ => Err(invalid()),
        }
    }
}

fn solution_metadata(
    value: &antecedent::gcm::nonlinear_equilibrium::EquilibriumResult,
) -> serde_json::Value {
    serde_json::json!({"status":"solved","dimension":value.values.len(),"residual":value.residual,"iterations":value.iterations,"globally_unique":value.globally_unique,"existence_certified":value.existence_certified,"numerical_converged":value.numerical_converged,"dynamic_stability":value.dynamic_stability,"jacobian_condition_estimate":value.jacobian_condition_estimate,"blocks":value.blocks,"evidence":value.evidence})
}

fn equilibrium_metadata(
    value: &antecedent::gcm::nonlinear_equilibrium::EquilibriumOutcome,
) -> serde_json::Value {
    match value {
        antecedent::gcm::nonlinear_equilibrium::EquilibriumOutcome::Solved(value) => {
            solution_metadata(value)
        }
        antecedent::gcm::nonlinear_equilibrium::EquilibriumOutcome::Failed(value) => {
            serde_json::json!({"status":"failed","failure":value.failure,"detail":value.detail})
        }
    }
}

fn recovered_slot_metadata(
    world: &antecedent::io::cyclic_family_artifact::RecoveredWorldAnswer,
    draw: usize,
) -> PyResult<serde_json::Value> {
    use antecedent::io::cyclic_family_artifact::RecoveredDrawAnswer as D;
    let value = world
        .draws
        .get(draw)
        .ok_or_else(|| crate::value_err("cyclic recovered draw out of range"))?;
    let outcome = match &value.outcome {
        D::Equilibrium { mean, solved_by, .. } => {
            serde_json::json!({"status":"equilibrium","dimension":mean.len(),"solved_by":solved_by})
        }
        D::Inadmissible { reason } => serde_json::json!({"status":"inadmissible","reason":reason}),
        D::NotCertified { fault } => serde_json::json!({"status":"not_certified","fault":fault}),
    };
    Ok(serde_json::json!({"weight":value.weight,"outcome":outcome}))
}

impl CyclicExecutionOwner {
    pub(super) fn contract_metadata(
        &self,
    ) -> Option<&antecedent::io::contract_section::cyclic::CyclicContractWire> {
        match self {
            Self::System(owner) => Some(&owner.contract_metadata().section),
            Self::Family(owner) => Some(&owner.contract_metadata().section),
            Self::ConsumedFamily(owner) => owner.contract_metadata(),
            Self::ConsumedSystem(_) => None,
        }
    }

    pub(super) fn contract_summary(&self) -> serde_json::Value {
        use antecedent::io::contract_section::cyclic::{
            CyclicIdentificationWire as I, CyclicTargetWire as T,
        };
        let Some(contract) = self.contract_metadata() else {
            return serde_json::Value::Null;
        };
        let hex =
            |bytes: [u8; 32]| antecedent::scientific::SemanticDigest::from_bytes(bytes).to_hex();
        let ids = &contract.identities;
        let identification = match &contract.identification {
            I::SpecifiedModelConditional { model } => {
                serde_json::json!({"kind":contract.identification.kind(),"model":hex(*model)})
            }
            I::ExperimentalRecovery { design, unresolved } => {
                serde_json::json!({"kind":contract.identification.kind(),"design":hex(*design),"unresolved_count":unresolved.len()})
            }
            I::CyclicIoScm { source } => {
                serde_json::json!({"kind":contract.identification.kind(),"source":hex(*source)})
            }
        };
        let count = assumption_count(&contract.target);
        let assumptions =
            serde_json::json!({"count":count,"operation":"assumptions_json","paged":true});
        let target = match &contract.target {
            T::Worlds { .. } => {
                serde_json::json!({"kind":"worlds","target_count":1,"assumptions":assumptions})
            }
            T::JointWorlds { targets } => {
                serde_json::json!({"kind":"joint_worlds","target_count":targets.len(),"assumptions":assumptions})
            }
            T::Functional { kind, outcome, query } => {
                serde_json::json!({"kind":kind,"outcome":outcome,"query":hex(*query)})
            }
        };
        let lowered = match self {
            Self::System(owner) => {
                Some(owner.contract_metadata().lowered_program_identity.to_hex())
            }
            Self::Family(owner) => {
                Some(owner.contract_metadata().lowered_program_identity.to_hex())
            }
            _ => None,
        };
        serde_json::json!({
            "family":contract.family,"request_identity":contract.request_identity,
            "identification_kind":contract.identification.kind(),
            "identities":{"target":hex(ids.target),"identification":hex(ids.identification),"identification_product":ids.identification_product.map(hex),"program":hex(ids.program),"inference_binding":hex(ids.inference_binding),"observation":hex(ids.observation),"data_snapshot":hex(ids.data_snapshot)},
            "cyclic_solution":hex(contract.cyclic_solution),"recovery":contract.recovery.map(hex),
            "lowered_program_identity":lowered,
            "layers":{"identification":identification,"target":target,"inference_binding":{"method":contract.inference_binding.method,"settings":hex(contract.inference_binding.settings)},"program":{"family":contract.program.family,"realization":contract.program.realization,"fit_dependencies":hex(contract.program.fit_dependencies)},"observation":{"modality":contract.observation.modality,"variable_count":contract.observation.variables.len(),"unit_count":contract.observation.units.len()}},
        })
    }
}

pub(super) fn assumption_count(
    target: &antecedent::io::contract_section::cyclic::CyclicTargetWire,
) -> usize {
    use antecedent::io::contract_section::cyclic::CyclicTargetWire as T;
    match target {
        T::Worlds { target } => target.assumptions().len(),
        T::JointWorlds { targets } => targets.iter().map(|target| target.assumptions().len()).sum(),
        T::Functional { .. } => 0,
    }
}
