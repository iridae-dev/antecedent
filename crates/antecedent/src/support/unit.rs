//! One support model over the three places support is decided.
//!
//! A scientific combination is supported by exactly one of: a support-matrix cell
//! ([`classify`]), a graphless design row ([`classify_graphless`]), or a capability unit whose
//! derived status is embedded in the build (`capability_supported`). [`decide`] takes a
//! [`SupportKey`] naming one of the three and answers with the same [`UnitDecision`], so a
//! caller needs no knowledge of which source owns the unit.
//!
//! [`require_program`] is the gate a compiled route passes through before it runs: a release
//! build refuses an unsupported unit with [`NotSupported`], and a development build records
//! what a release build would refuse and runs the route. A support decision never enters a
//! program's identity.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, NotSupported};

use super::cell::{CellStatus, SupportCell};
use super::gate::capability_status;
use super::matrix::classify;
use super::{
    GraphlessAssignmentSupport, GraphlessSupportKey, GraphlessSupportStatus, classify_graphless,
};

/// The id of a capability unit (`parity/capability_records.toml`).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UnitId(&'static str);

impl UnitId {
    /// The unit with this record id.
    #[must_use]
    pub(crate) const fn new(id: &'static str) -> Self {
        Self(id)
    }

    /// The record id.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// What a support decision is about, in the vocabulary of the three sources.
#[derive(Clone, Copy, Debug)]
pub enum SupportKey<'a> {
    /// A support-matrix cell.
    Matrix(SupportCell),
    /// A graphless design row, with the assignment support observed for it.
    Graphless(GraphlessSupportKey<'a>, GraphlessAssignmentSupport),
    /// A capability unit by its record id.
    Unit(UnitId),
}

/// The unit a decision was made for, owned so that it outlives the borrowed key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnitRef {
    /// A support-matrix cell.
    Matrix(SupportCell),
    /// A graphless design row.
    Graphless {
        /// Query family.
        family: String,
        /// Assignment design.
        design: String,
        /// Estimator method.
        method: String,
        /// Exact inference claim.
        inference_claim: String,
    },
    /// A capability unit.
    Capability(UnitId),
}

impl UnitRef {
    /// A readable name of the unit, for refusal messages.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Matrix(cell) => format!("support cell {cell:?}"),
            Self::Graphless { family, design, method, inference_claim } => {
                format!("{family}/{design}/{method}/{inference_claim}")
            }
            Self::Capability(id) => id.as_str().to_owned(),
        }
    }
}

/// The support status of one unit and the evidence it still lacks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnitDecision {
    /// [`CellStatus::Supported`], or why not.
    pub status: CellStatus,
    /// The unit the status is about.
    pub unit: UnitRef,
    /// The evidence a capability unit lists as missing. Empty for a matrix cell or a graphless
    /// row, which carry no listing; a capability unit that is not supported always lists at
    /// least itself.
    pub missing_evidence: Vec<&'static str>,
}

/// The support status of the unit `key` names, by the decider that owns that kind of unit.
///
/// A capability unit that the embedded status does not list, or a status that does not decode,
/// is not supported, so a gated route refuses rather than runs on a misread status.
#[must_use]
pub fn decide(key: &SupportKey<'_>) -> UnitDecision {
    match key {
        SupportKey::Matrix(cell) => UnitDecision {
            status: classify(*cell),
            unit: UnitRef::Matrix(*cell),
            missing_evidence: Vec::new(),
        },
        SupportKey::Graphless(row, observed) => {
            let supported = matches!(
                classify_graphless(*row, *observed),
                GraphlessSupportStatus::Supported { .. }
            );
            UnitDecision {
                status: if supported { CellStatus::Supported } else { CellStatus::NotSupported },
                unit: UnitRef::Graphless {
                    family: row.family.to_owned(),
                    design: row.design.to_owned(),
                    method: row.method.to_owned(),
                    inference_claim: row.inference_claim.to_owned(),
                },
                missing_evidence: Vec::new(),
            }
        }
        SupportKey::Unit(id) => {
            let (supported, listed) = capability_status(id.as_str()).unwrap_or((false, Vec::new()));
            let missing_evidence = if supported {
                Vec::new()
            } else if listed.is_empty() {
                vec![id.as_str()]
            } else {
                listed
            };
            UnitDecision {
                status: if supported { CellStatus::Supported } else { CellStatus::NotSupported },
                unit: UnitRef::Capability(*id),
                missing_evidence,
            }
        }
    }
}

/// Pass the support gate of the program entry point `entry`, which runs under the unit `key`
/// names, with the dispatch-scoped context.
///
/// Release: [`NotSupported`] naming `entry`, with the unit's missing evidence (or the unit
/// itself when it lists none). Development: records what a release build refuses in the log
/// every clone of `ctx` shares, and the entry point runs.
///
/// # Errors
///
/// [`NotSupported`] in a release build when the unit is not supported.
pub(crate) fn require_program(
    ctx: &ExecutionContext,
    key: &SupportKey<'_>,
    entry: &'static str,
) -> Result<(), NotSupported> {
    program_gate(key, |missing| {
        antecedent_core::support_gate::require_supported(ctx, entry, missing)
    })
}

/// [`require_program`] for an entry point that compiles without an [`ExecutionContext`]. A
/// development build records into the calling thread's log
/// ([`antecedent_core::support_gate::drain_unscoped`]).
///
/// # Errors
///
/// [`NotSupported`] in a release build when the unit is not supported.
pub(crate) fn require_program_unscoped(
    key: &SupportKey<'_>,
    entry: &'static str,
) -> Result<(), NotSupported> {
    program_gate(key, |missing| {
        antecedent_core::support_gate::require_supported_unscoped(entry, missing)
    })
}

fn program_gate(
    key: &SupportKey<'_>,
    gate: impl FnOnce(&[&str]) -> Result<(), NotSupported>,
) -> Result<(), NotSupported> {
    let decision = decide(key);
    if decision.status == CellStatus::Supported {
        return Ok(());
    }
    if decision.missing_evidence.is_empty() {
        let label = decision.unit.label();
        gate(&[label.as_str()])
    } else {
        gate(&decision.missing_evidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::supported_cells;
    use crate::support::units_generated::{self, unit_of};

    /// The status the build embeds for `id`, read from the snapshot directly.
    fn embedded_status(id: &str) -> Option<String> {
        let snapshot: serde_json::Value =
            serde_json::from_str(antecedent_io_core::graphless_support::UNIT_STATUS)
                .expect("the embedded unit status decodes");
        snapshot["capabilities"]
            .as_array()
            .expect("capabilities list")
            .iter()
            .find(|unit| unit["id"] == id)
            .map(|unit| unit["status"].as_str().expect("status text").to_owned())
    }

    fn bound_units() -> [UnitId; 6] {
        [
            units_generated::COMPOSITION_BUNDLE,
            units_generated::ESTIMATE_LATENT_SOFT_INTERVENTION,
            units_generated::ESTIMATE_OBSERVATIONAL_POLICY_VALUE,
            units_generated::FORECAST_EVENT_PROCESS,
            units_generated::JOINT_RESPONSE_INTERVAL_READINGS,
            units_generated::MNAR_FIXED_SHIFT_COMPILED_CANDIDATE,
        ]
    }

    #[test]
    fn a_capability_unit_decides_as_the_embedded_status_records() {
        for unit in bound_units() {
            let recorded = embedded_status(unit.as_str())
                .unwrap_or_else(|| panic!("{} is not in the embedded status", unit.as_str()));
            let decision = decide(&SupportKey::Unit(unit));
            assert_eq!(decision.unit, UnitRef::Capability(unit));
            assert_eq!(
                decision.status == CellStatus::Supported,
                recorded == "supported",
                "{}",
                unit.as_str()
            );
            assert_eq!(decision.missing_evidence.is_empty(), recorded == "supported");
        }
    }

    #[test]
    fn a_fixed_shift_unit_decides_as_the_embedded_status_records() {
        let unit = unit_of("MnarFixedShift").expect("the fixed-shift route is bound");
        assert_eq!(unit.as_str(), "mnar_fixed_shift.compiled_candidate");
        let recorded = embedded_status(unit.as_str()).expect("the unit has an embedded status");
        let decision = decide(&SupportKey::Unit(unit));
        assert_eq!(decision.status == CellStatus::Supported, recorded == "supported");
        assert_eq!(crate::support::capability_supported(unit.as_str()), recorded == "supported");
    }

    #[test]
    fn an_unlisted_unit_is_not_supported_and_names_itself() {
        let decision = decide(&SupportKey::Unit(UnitId::new("no.such.unit")));
        assert_eq!(decision.status, CellStatus::NotSupported);
        assert_eq!(decision.missing_evidence, vec!["no.such.unit"]);
    }

    #[test]
    fn a_matrix_cell_decides_as_classify_does() {
        let cell = supported_cells().next().expect("the matrix lists a supported cell");
        let decision = decide(&SupportKey::Matrix(cell));
        assert_eq!(decision.status, classify(cell));
        assert_eq!(decision.unit, UnitRef::Matrix(cell));
    }

    #[test]
    fn a_graphless_row_decides_as_classify_graphless_does() {
        let key = GraphlessSupportKey {
            family: "no_such_family",
            design: "no_such_design",
            method: "no_such_method",
            inference_claim: "no_such_claim",
        };
        let observed = GraphlessAssignmentSupport::default();
        let decision = decide(&SupportKey::Graphless(key, observed));
        assert_eq!(decision.status, CellStatus::NotSupported);
        assert!(matches!(classify_graphless(key, observed), GraphlessSupportStatus::Refused));
        assert_eq!(
            decision.unit.label(),
            "no_such_family/no_such_design/no_such_method/no_such_claim"
        );
    }

    #[test]
    fn the_generated_table_binds_routes_by_name() {
        assert_eq!(
            unit_of("LatentSoftIntervention"),
            Some(units_generated::ESTIMATE_LATENT_SOFT_INTERVENTION)
        );
        assert_eq!(unit_of("EventProcess"), Some(units_generated::FORECAST_EVENT_PROCESS));
        assert_eq!(unit_of("NoSuchRoute"), None);
    }

    #[test]
    fn the_program_gate_passes_a_supported_unit_and_gates_an_unsupported_one() {
        let supported = SupportKey::Unit(units_generated::COMPOSITION_BUNDLE);
        if decide(&supported).status == CellStatus::Supported {
            assert!(require_program_unscoped(&supported, "test.entry").is_ok());
        }
        let unlisted = SupportKey::Unit(UnitId::new("no.such.unit"));
        let _ = antecedent_core::support_gate::drain_unscoped();
        let outcome = require_program_unscoped(&unlisted, "test.entry");
        #[cfg(not(feature = "development"))]
        assert!(outcome.is_err(), "a release build refuses an unsupported unit");
        #[cfg(feature = "development")]
        {
            assert!(outcome.is_ok(), "a development build runs the route");
            let recorded = antecedent_core::support_gate::drain_unscoped();
            assert_eq!(recorded.len(), 1);
            assert_eq!(recorded[0].detail, "test.entry");
            assert_eq!(recorded[0].missing, vec!["no.such.unit".to_owned()]);
        }
    }
}
