//! The portable finite-discrete wire of a prepared empirical provider snapshot.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::RegimeId;
use antecedent_expr::{DomainRef, FactorRequirement};
use antecedent_io_core::distribution_replay::FACTOR_LAWS_FORMAT;
use antecedent_io_core::{
    DistributionFactorDomainWire, DistributionFactorKeyWire, DistributionFactorLawsWire,
    DistributionFactorRowWire, DistributionFactorTableWire, IoError, ValueWire,
};

/// Convert a prepared empirical provider snapshot into its portable finite-discrete wire.
///
/// # Errors
///
/// Returns an error when a semantic identifier cannot be represented on the wire.
pub fn distribution_factor_laws_to_wire(
    snapshot: &antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot,
) -> Result<DistributionFactorLawsWire, IoError> {
    let key = |requirement: &FactorRequirement| -> Result<DistributionFactorKeyWire, IoError> {
        Ok(DistributionFactorKeyWire {
            variables: requirement.variables.iter().map(|id| id.raw()).collect(),
            conditioned_on: requirement.conditioned_on.iter().map(|id| id.raw()).collect(),
            intervention: requirement
                .intervention
                .iter()
                .map(|assignment| {
                    Ok(crate::expr_wire::InterventionAssignmentWire {
                        symbolic: assignment.is_symbolic(),
                        variable: assignment.variable.raw(),
                        value: ValueWire::from_value(&assignment.value),
                    })
                })
                .collect::<Result<_, IoError>>()?,
            domain: match requirement.domain {
                DomainRef::Observational => DistributionFactorDomainWire::Observational,
                DomainRef::Interventional => DistributionFactorDomainWire::Interventional,
            },
            population: requirement.population.to_string(),
            regime: requirement.regime.map(RegimeId::raw),
        })
    };
    let requirements = snapshot.requirements.iter().map(key).collect::<Result<_, _>>()?;
    let domains = snapshot
        .provider
        .domains
        .iter()
        .map(|domain| {
            (domain.variable.raw(), domain.values.iter().map(ValueWire::from_value).collect())
        })
        .collect();
    let factors = snapshot
        .provider
        .factors
        .iter()
        .map(|factor| {
            let factor_key = DistributionFactorKeyWire {
                variables: factor.variables.iter().map(|id| id.raw()).collect(),
                conditioned_on: factor.conditioned_on.iter().map(|id| id.raw()).collect(),
                intervention: factor
                    .intervention
                    .iter()
                    .map(|assignment| {
                        Ok(crate::expr_wire::InterventionAssignmentWire {
                            symbolic: assignment.is_symbolic(),
                            variable: assignment.variable.raw(),
                            value: ValueWire::from_value(&assignment.value),
                        })
                    })
                    .collect::<Result<_, IoError>>()?,
                domain: match factor.domain {
                    DomainRef::Observational => DistributionFactorDomainWire::Observational,
                    DomainRef::Interventional => DistributionFactorDomainWire::Interventional,
                },
                population: factor.population.to_string(),
                regime: factor.regime.map(RegimeId::raw),
            };
            Ok(DistributionFactorTableWire {
                key: factor_key,
                rows: factor
                    .rows
                    .iter()
                    .map(|row| DistributionFactorRowWire {
                        values: row.values.iter().map(ValueWire::from_value).collect(),
                        probability: row.probability,
                    })
                    .collect(),
            })
        })
        .collect::<Result<_, IoError>>()?;
    Ok(DistributionFactorLawsWire {
        format: FACTOR_LAWS_FORMAT,
        provider: snapshot.provenance.provider.into(),
        source_rows: u64::try_from(snapshot.provenance.source_rows)
            .map_err(|_| IoError::TooLarge)?,
        complete_case_rows: u64::try_from(snapshot.provenance.complete_case_rows)
            .map_err(|_| IoError::TooLarge)?,
        missing_row_policy: snapshot.provenance.missing_row_policy.into(),
        domains,
        requirements,
        factors,
    })
}
