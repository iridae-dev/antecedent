//! Resource bounds of the adaptive-procedure routes, one class per kind of limit.
//!
//! A request to derive the uncertainty of a procedure can run into five
//! different kinds of limit, and each one says something different about what
//! to change, so each has its own [`CapacityError`] variant, refusal detail and
//! remedy:
//!
//! * **declaration**: the units and looks a procedure may declare. Observed
//!   counts are stored in 16 bits, so a declaration enrols at most 65,535
//!   units; the number of looks is bounded separately of the number of units;
//! * **exact law**: the live observed states and the unit expansions of the
//!   dynamic program ([`super::procedure_law`]). Exact enumeration is the route
//!   whose cost grows with the number of distinct histories, so it is bounded
//!   by state and work, never by participants alone;
//! * **simulation replications**: how many replications one request may run;
//! * **simulation work**: replications times units times measurements per unit,
//!   the cost of the whole-procedure simulation that large procedures use in
//!   place of the enumeration;
//! * **simulation memory**: the estimated peak bytes of the replicate records
//!   and the retained history of one replicate, against the memory budget of
//!   the execution context;
//! * **retained record**: the looks and participants one exported record holds.
//!
//! Each simulation also returns a [`ResourceReceipt`] of the work it did.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

/// Most simulation work one request may do: replications times units times
/// measurements per unit.
pub const MAX_SIMULATION_WORK: u64 = 4_000_000_000;
/// Most estimated peak bytes one simulation may hold when the execution
/// context sets no memory budget.
pub const MAX_SIMULATION_MEMORY_BYTES: u64 = 4 << 30;
/// Most looks one exported procedure record holds.
pub const MAX_RETAINED_LOOKS: usize = 64;

/// The class of resource bound a request exceeded, with the bound and the
/// requested size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CapacityError {
    /// More units (participants) than a procedure declaration may enrol.
    DeclaredUnits {
        /// The bound.
        limit: u64,
        /// The declared size.
        requested: u64,
    },
    /// More looks (batches) than a procedure declaration may schedule.
    DeclaredLooks {
        /// The bound.
        limit: u64,
        /// The declared size.
        requested: u64,
    },
    /// More live observed states than the exact law may hold at one look.
    LawStates {
        /// The bound.
        limit: u64,
        /// The size reached.
        requested: u64,
    },
    /// More unit expansions than the exact law may perform.
    LawWork {
        /// The bound.
        limit: u64,
        /// The work the enumeration reached.
        requested: u64,
    },
    /// More replications than one simulation may run.
    SimulationReplications {
        /// The bound.
        limit: u64,
        /// The requested replications.
        requested: u64,
    },
    /// More simulation work than the work budget admits.
    SimulationWork {
        /// The bound.
        limit: u64,
        /// Replications times units times measurements.
        requested: u64,
    },
    /// More estimated memory than the memory budget admits.
    SimulationMemory {
        /// The bound, in bytes.
        limit: u64,
        /// The estimated peak, in bytes.
        requested: u64,
    },
    /// More looks or participants than one exported record holds.
    RetainedRecord {
        /// The bound.
        limit: u64,
        /// The requested size.
        requested: u64,
    },
}

impl CapacityError {
    /// Stable `snake_case` name of the class of limit.
    #[must_use]
    pub const fn class(self) -> &'static str {
        match self {
            Self::DeclaredUnits { .. } => "declared_units",
            Self::DeclaredLooks { .. } => "declared_looks",
            Self::LawStates { .. } => "exact_law_states",
            Self::LawWork { .. } => "exact_law_work",
            Self::SimulationReplications { .. } => "simulation_replications",
            Self::SimulationWork { .. } => "simulation_work",
            Self::SimulationMemory { .. } => "simulation_memory",
            Self::RetainedRecord { .. } => "retained_record",
        }
    }

    /// Namespaced detail of the refusal.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::DeclaredUnits { .. } => "adaptive_inference.declared_units_exceeded",
            Self::DeclaredLooks { .. } => "adaptive_inference.declared_looks_exceeded",
            Self::LawStates { .. } => "adaptive_inference.exact_law_states_exceeded",
            Self::LawWork { .. } => "adaptive_inference.exact_law_work_exceeded",
            Self::SimulationReplications { .. } => {
                "adaptive_inference.simulation_replications_exceeded"
            }
            Self::SimulationWork { .. } => "adaptive_inference.simulation_work_exceeded",
            Self::SimulationMemory { .. } => "adaptive_inference.simulation_memory_exceeded",
            Self::RetainedRecord { .. } => "adaptive_inference.retained_record_exceeded",
        }
    }

    /// Registered reason code of every capacity refusal.
    #[must_use]
    pub fn reason_code(self) -> &'static str {
        antecedent_core::reason_code!("invalid_argument")
    }

    /// What would bring the request inside the bound.
    #[must_use]
    pub const fn remedy(self) -> &'static str {
        match self {
            Self::DeclaredUnits { .. } => {
                "declare fewer units; split a larger study into separately declared stages"
            }
            Self::DeclaredLooks { .. } => {
                "declare fewer looks: enrol larger batches between consecutive looks"
            }
            Self::LawStates { .. } => {
                "the exact law holds too many distinct observed histories at one look: simulate the whole procedure instead of enumerating it, or reduce the units, arms, contexts or delay"
            }
            Self::LawWork { .. } => {
                "the exact law would expand too many units: simulate the whole procedure under a work budget instead of enumerating it, or reduce the units, arms, contexts or delay"
            }
            Self::SimulationReplications { .. } => {
                "request between 1 and the maximum replications, or run several seeded requests"
            }
            Self::SimulationWork { .. } => {
                "reduce the replications, the units or the measurements per unit so that their product is within the work budget"
            }
            Self::SimulationMemory { .. } => {
                "reduce the replications (each keeps its record) or the units, or raise the memory budget of the execution context"
            }
            Self::RetainedRecord { .. } => {
                "retain fewer looks or participants in the exported record"
            }
        }
    }

    /// The bound that was exceeded.
    #[must_use]
    pub const fn limit(self) -> u64 {
        match self {
            Self::DeclaredUnits { limit, .. }
            | Self::DeclaredLooks { limit, .. }
            | Self::LawStates { limit, .. }
            | Self::LawWork { limit, .. }
            | Self::SimulationReplications { limit, .. }
            | Self::SimulationWork { limit, .. }
            | Self::SimulationMemory { limit, .. }
            | Self::RetainedRecord { limit, .. } => limit,
        }
    }

    /// The size that was asked for or reached.
    #[must_use]
    pub const fn requested(self) -> u64 {
        match self {
            Self::DeclaredUnits { requested, .. }
            | Self::DeclaredLooks { requested, .. }
            | Self::LawStates { requested, .. }
            | Self::LawWork { requested, .. }
            | Self::SimulationReplications { requested, .. }
            | Self::SimulationWork { requested, .. }
            | Self::SimulationMemory { requested, .. }
            | Self::RetainedRecord { requested, .. } => requested,
        }
    }
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} exceeds the bound {}",
            self.class().replace('_', " "),
            self.requested(),
            self.limit()
        )
    }
}

impl std::error::Error for CapacityError {}

/// The work and memory a simulation may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBudget {
    /// Most replications times units times measurements.
    pub max_work: u64,
    /// Most estimated peak bytes.
    pub max_memory_bytes: u64,
}

impl ResourceBudget {
    /// The library bounds.
    pub const DEFAULT: Self =
        Self { max_work: MAX_SIMULATION_WORK, max_memory_bytes: MAX_SIMULATION_MEMORY_BYTES };

    /// The library work bound and the memory budget of an execution context
    /// (the smaller of its soft and hard limits; the library bound when the
    /// context sets none).
    #[must_use]
    pub fn from_context(ctx: &ExecutionContext) -> Self {
        let configured =
            [ctx.memory.soft_limit_bytes, ctx.memory.hard_limit_bytes].into_iter().flatten().min();
        Self {
            max_work: MAX_SIMULATION_WORK,
            max_memory_bytes: configured.unwrap_or(MAX_SIMULATION_MEMORY_BYTES),
        }
    }
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// What a simulation did, returned with its result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceReceipt {
    /// Replications run.
    pub replications: u32,
    /// Looks performed, summed over the replications.
    pub looks: u64,
    /// Cells of the observed-state tally one replication holds (the
    /// sufficient statistic its allocation and decisions are computed from),
    /// at most over the replications.
    pub states: u64,
    /// Work the request was admitted for: replications times units times
    /// measurements per unit.
    pub work_planned: u64,
    /// Work done: the units enrolled in each replication, times measurements
    /// per unit, summed (less than planned when a stopping rule ends runs
    /// early).
    pub work_done: u64,
    /// Most units one replication held at its end (the peak retained state).
    pub peak_units: u32,
    /// Estimated peak bytes the request was admitted for.
    pub estimated_peak_bytes: u64,
}

impl ResourceReceipt {
    /// Count one finished replication: the `looks` it performed, the `units`
    /// it enrolled with `measurements` per unit, and the `states` of its
    /// tally. Every counter is added with checked arithmetic; one that would
    /// wrap is refused as the largest representable work, never wrapped.
    pub(crate) fn record_replicate(
        &mut self,
        looks: usize,
        units: u32,
        measurements: u32,
        states: u64,
    ) -> Result<(), CapacityError> {
        let overflow =
            CapacityError::SimulationWork { limit: self.work_planned, requested: u64::MAX };
        let work = u64::from(units).checked_mul(u64::from(measurements)).ok_or(overflow)?;
        let looks = u64::try_from(looks).map_err(|_| overflow)?;
        let work_done = self.work_done.checked_add(work).ok_or(overflow)?;
        let looks = self.looks.checked_add(looks).ok_or(overflow)?;
        let replications = self.replications.checked_add(1).ok_or(overflow)?;
        self.work_done = work_done;
        self.looks = looks;
        self.replications = replications;
        self.peak_units = self.peak_units.max(units);
        self.states = self.states.max(states);
        Ok(())
    }
}

/// The size of a simulation, for admission against a [`ResourceBudget`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct SimulationShape {
    pub replications: u32,
    pub units: u32,
    pub measurements: u32,
    /// Bytes each replicate record keeps after the replication ends.
    pub record_bytes: u64,
    /// Bytes each unit of one replication holds while it runs.
    pub unit_bytes: u64,
    /// Bytes of caches and fixed tables.
    pub fixed_bytes: u64,
}

/// A simulation the budget admitted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Admitted {
    pub work_planned: u64,
    pub estimated_peak_bytes: u64,
}

impl ResourceBudget {
    /// Admit a simulation of `shape`.
    ///
    /// Sizes are multiplied with checked arithmetic; an overflow is refused
    /// as the largest representable size.
    pub(crate) fn admit(&self, shape: &SimulationShape) -> Result<Admitted, CapacityError> {
        let work = u64::from(shape.replications)
            .checked_mul(u64::from(shape.units))
            .and_then(|w| w.checked_mul(u64::from(shape.measurements)));
        let Some(work) = work.filter(|&w| w <= self.max_work) else {
            return Err(CapacityError::SimulationWork {
                limit: self.max_work,
                requested: work.unwrap_or(u64::MAX),
            });
        };
        let bytes = u64::from(shape.replications)
            .checked_mul(shape.record_bytes)
            .and_then(|b| b.checked_add(u64::from(shape.units).checked_mul(shape.unit_bytes)?))
            .and_then(|b| b.checked_add(shape.fixed_bytes));
        let Some(bytes) = bytes.filter(|&b| b <= self.max_memory_bytes) else {
            return Err(CapacityError::SimulationMemory {
                limit: self.max_memory_bytes,
                requested: bytes.unwrap_or(u64::MAX),
            });
        };
        Ok(Admitted { work_planned: work, estimated_peak_bytes: bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_inference::{
        AnytimeSettings, BetaPrior, BorrowingPlan, ContractGap, DecisionRules, MAX_PROCEDURE_UNITS,
        ParticipantDeclaration, ParticipantSpec,
    };

    const fn shape(replications: u32, units: u32, measurements: u32) -> SimulationShape {
        SimulationShape {
            replications,
            units,
            measurements,
            record_bytes: 1,
            unit_bytes: 1,
            fixed_bytes: 0,
        }
    }

    #[test]
    fn adaptive_counters_checked_arithmetic_refuses_overflow() {
        // A product that wraps a 64-bit counter is refused as the largest representable size,
        // even under a budget that admits every representable work, and the wrapped value is
        // never compared with the budget.
        let widest = ResourceBudget { max_work: u64::MAX, max_memory_bytes: u64::MAX };
        assert_eq!(
            widest.admit(&shape(u32::MAX, u32::MAX, u32::MAX)).unwrap_err(),
            CapacityError::SimulationWork { limit: u64::MAX, requested: u64::MAX }
        );
        let mut heavy = shape(2, 1, 1);
        heavy.record_bytes = u64::MAX;
        assert_eq!(
            widest.admit(&heavy).unwrap_err(),
            CapacityError::SimulationMemory { limit: u64::MAX, requested: u64::MAX }
        );
        // A product just inside the counter is admitted exactly.
        assert_eq!(
            widest.admit(&shape(1_000_000, 65_535, 32)).map(|a| a.work_planned),
            Ok(1_000_000 * 65_535 * 32)
        );

        // The receipt adds every counter with checked arithmetic and changes nothing on refusal.
        let mut receipt = ResourceReceipt { work_planned: 7, ..ResourceReceipt::default() };
        receipt.record_replicate(3, 2, 2, 8).unwrap();
        assert_eq!((receipt.replications, receipt.looks, receipt.work_done), (1, 3, 4));
        assert_eq!((receipt.peak_units, receipt.states), (2, 8));
        receipt.work_done = u64::MAX - 3;
        let before = receipt;
        assert_eq!(
            receipt.record_replicate(1, 2, 2, 8).unwrap_err(),
            CapacityError::SimulationWork { limit: 7, requested: u64::MAX }
        );
        assert_eq!(receipt, before, "a refused count leaves the receipt as it was");
        receipt.work_done = 0;
        receipt.looks = u64::MAX;
        assert!(receipt.record_replicate(1, 1, 1, 1).is_err());
        receipt.looks = 0;
        receipt.replications = u32::MAX;
        assert!(receipt.record_replicate(1, 1, 1, 1).is_err());

        // A declaration whose enrolments overflow a 32-bit count is refused with the exact
        // 64-bit total, not a wrapped one.
        let declaration = ParticipantDeclaration {
            arms: 3,
            responses: Some(3),
            correlation: Some(0.2),
            correlation_prior: None,
            borrowing: Some(BorrowingPlan::uniform(3, BetaPrior::UNIFORM)),
            enrolment: Some(vec![u32::MAX, u32::MAX, u32::MAX]),
            lag_looks: 0,
            min_exploration: Some(0.1),
            decisions: Some(DecisionRules {
                loss_threshold: 0.01,
                futility: None,
                participant_cap: None,
            }),
            anytime: Some(AnytimeSettings { alpha: 0.05, tuning_samples: 30.0 }),
        };
        let gaps = ParticipantSpec::new(declaration).unwrap_err().gaps;
        assert_eq!(
            gaps,
            vec![ContractGap::Capacity(CapacityError::DeclaredUnits {
                limit: u64::from(MAX_PROCEDURE_UNITS),
                requested: 3 * u64::from(u32::MAX),
            })]
        );
    }
}
