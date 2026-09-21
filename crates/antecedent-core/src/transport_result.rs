//! Located scientific support failures for transport execution.
/// Located point-local missing evidence or support outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportGridFailure {
    /// Stable kind: `missing_evidence` or `support_failure`.
    pub kind: String,
    /// Located provider/denominator explanation.
    pub detail: String,
    /// Stable provider/support failure code.
    pub code: String,
    /// Original variables whose provider support is required.
    pub variables: Vec<u32>,
    /// Original expression coordinate for a failed ratio.
    pub expression: Option<u32>,
    /// Located original variable assignments.
    pub assignment: Vec<(u32, crate::Value)>,
    /// Population/regime dependencies of the failing factor.
    pub bindings: Vec<(String, Option<u32>)>,
    /// Concrete intervention world required by this factor.
    pub interventions: Vec<(u32, crate::Value)>,
}
