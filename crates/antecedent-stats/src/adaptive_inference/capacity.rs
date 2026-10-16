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
            .and_then(|w| w.checked_mul(u64::from(shape.measurements)))
            .unwrap_or(u64::MAX);
        if work > self.max_work {
            return Err(CapacityError::SimulationWork { limit: self.max_work, requested: work });
        }
        let bytes = u64::from(shape.replications)
            .checked_mul(shape.record_bytes)
            .and_then(|b| b.checked_add(u64::from(shape.units).checked_mul(shape.unit_bytes)?))
            .and_then(|b| b.checked_add(shape.fixed_bytes))
            .unwrap_or(u64::MAX);
        if bytes > self.max_memory_bytes {
            return Err(CapacityError::SimulationMemory {
                limit: self.max_memory_bytes,
                requested: bytes,
            });
        }
        Ok(Admitted { work_planned: work, estimated_peak_bytes: bytes })
    }
}
