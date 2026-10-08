//! Route-level refusal of the closed learned joint source-target transport row (2.3B B1
//! row `learned_joint_transport`, `antecedent.transport.learned_joint`).
//!
//! The Rust core ([`antecedent_estimate::learned_joint_transport`]) fits the outcome
//! mechanism through `antecedent-learn` and replays through an independent io artifact, but
//! its posterior claims a *calibrated* interval whose coverage is measured only at the
//! release cut. Until then the public producer is closed: it validates its request against
//! the row's scope with the core's own bounds and details and otherwise refuses with the live
//! typed `cell_not_licensed` refusal `learned_joint_transport.route_frozen`. A request outside
//! the row refuses first with the row's own scope refusal, so a caller learns the real
//! obstruction before the closed route.
//!
//! The function returns the refusal as plain data; a binding raises it. It never runs a fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::joint_bayesian_transport::{
    JOINT_TRANSPORT_MAX_DRAWS, JOINT_TRANSPORT_MAX_PARAMETERS, JointTransportRefusal,
};
use antecedent_estimate::learned_joint_transport as learned;
use antecedent_learn::PolynomialBasis;

/// A request to the closed learned joint transport route.
#[derive(Clone, Copy, Debug)]
pub struct LearnedJointRequest<'a> {
    /// `fixed_dag`, `admg` or `graph_posterior`.
    pub graph_class: &'a str,
    /// `independent_samples`, `overlapping_units` or `unknown`.
    pub dependence: &'a str,
    /// `intercept` or `intercept_and_covariates`.
    pub varying: &'a str,
    /// `independent_varying_blocks` or `shared_varying_block`.
    pub sharing: &'a str,
    /// Number of covariates.
    pub features: usize,
    /// Polynomial degree of the basis.
    pub basis_degree: usize,
    /// Number of source trials.
    pub sources: usize,
    /// Whether a target covariate sample was supplied.
    pub has_target: bool,
    /// Requested posterior draws.
    pub draws: usize,
}

/// A closed-route refusal as plain data: registered reason code and namespaced detail.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LearnedJointRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail (`learned_joint_transport.*`).
    pub detail: &'static str,
    /// What was refused.
    pub message: String,
}

impl LearnedJointRefusal {
    fn new(code: &'static str, detail: &'static str, message: impl Into<String>) -> Self {
        Self { code, detail, message: message.into() }
    }

    fn invalid(detail: &'static str, message: impl Into<String>) -> Self {
        Self::new(antecedent_core::reason_code!("invalid_argument"), detail, message)
    }

    /// `<detail>: <message>`, the text a binding raises after the `reason=<code>:` prefix.
    #[must_use]
    pub fn text(&self) -> String {
        format!("{}: {}", self.detail, self.message)
    }
}

fn from_core(refusal: &JointTransportRefusal) -> LearnedJointRefusal {
    LearnedJointRefusal::new(refusal.code, refusal.detail, refusal.message.clone())
}

/// Number of model parameters the fit would declare, or `None` for an unknown spelling.
fn parameters(request: &LearnedJointRequest<'_>, basis_terms: usize) -> Option<usize> {
    let (q, r) = match request.varying {
        "intercept" => (basis_terms.saturating_mul(2).saturating_add(1), 1),
        "intercept_and_covariates" => {
            (basis_terms.saturating_add(1), basis_terms.saturating_add(1))
        }
        _ => return None,
    };
    let blocks = match request.sharing {
        "independent_varying_blocks" => request.sources,
        "shared_varying_block" => 1,
        _ => return None,
    };
    Some(q.saturating_add(blocks.saturating_mul(r)))
}

fn scope_refusal(request: &LearnedJointRequest<'_>) -> Option<LearnedJointRefusal> {
    match request.graph_class {
        "fixed_dag" => {}
        "admg" | "graph_posterior" => {
            return Some(from_core(&learned::unsupported_graph_refusal()));
        }
        _ => {
            return Some(LearnedJointRefusal::invalid(
                learned::DETAIL_INVALID_INPUT,
                "graph_class must be fixed_dag, admg or graph_posterior",
            ));
        }
    }
    let Ok(basis) = PolynomialBasis::new(request.basis_degree) else {
        return Some(LearnedJointRefusal::invalid(
            learned::DETAIL_INVALID_BASIS,
            "the polynomial basis degree must be in 1..=6",
        ));
    };
    if request.draws == 0 || request.draws > JOINT_TRANSPORT_MAX_DRAWS {
        return Some(LearnedJointRefusal::invalid(
            learned::DETAIL_TOO_MANY_DRAWS,
            format!("draw count must be in 1..={JOINT_TRANSPORT_MAX_DRAWS}"),
        ));
    }
    if !matches!(request.dependence, "independent_samples" | "overlapping_units" | "unknown") {
        return Some(LearnedJointRefusal::invalid(
            learned::DETAIL_INVALID_INPUT,
            "dependence must be independent_samples, overlapping_units or unknown",
        ));
    }
    if request.dependence != "independent_samples" {
        return Some(LearnedJointRefusal::new(
            antecedent_core::reason_code!("sampling_dependence_unknown"),
            learned::DETAIL_SOURCE_DEPENDENCE,
            "only independent source samples with disjoint units are supported",
        ));
    }
    if request.sources == 0 {
        return Some(LearnedJointRefusal::new(
            antecedent_core::reason_code!("transport_missing_evidence"),
            learned::DETAIL_MISSING_SOURCE,
            "at least one source trial is required",
        ));
    }
    if !request.has_target {
        return Some(LearnedJointRefusal::new(
            antecedent_core::reason_code!("joint_law_required"),
            learned::DETAIL_MISSING_LAW,
            "the target covariate law (a nonempty target sample) is required",
        ));
    }
    let Some(count) = parameters(request, basis.width(request.features)) else {
        return Some(LearnedJointRefusal::invalid(
            learned::DETAIL_INVALID_INPUT,
            "varying must be intercept or intercept_and_covariates and sharing \
             independent_varying_blocks or shared_varying_block",
        ));
    };
    if count > JOINT_TRANSPORT_MAX_PARAMETERS {
        return Some(LearnedJointRefusal::invalid(
            learned::DETAIL_TOO_MANY_PARAMETERS,
            format!(
                "the model has {count} parameters; the bound is {JOINT_TRANSPORT_MAX_PARAMETERS}"
            ),
        ));
    }
    None
}

/// The refusal of a request to the closed learned joint transport route: the row's scope
/// refusal when the request is outside it, otherwise
/// `cell_not_licensed` / `learned_joint_transport.route_frozen`.
#[must_use]
pub fn learned_joint_transport_refusal(request: &LearnedJointRequest<'_>) -> LearnedJointRefusal {
    scope_refusal(request).unwrap_or_else(|| from_core(&learned::route_frozen_refusal()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> LearnedJointRequest<'static> {
        LearnedJointRequest {
            graph_class: "fixed_dag",
            dependence: "independent_samples",
            varying: "intercept",
            sharing: "independent_varying_blocks",
            features: 1,
            basis_degree: 2,
            sources: 2,
            has_target: true,
            draws: 1000,
        }
    }

    #[test]
    fn a2_learned_route_is_frozen_and_scope_refusals_come_first() {
        let frozen = learned_joint_transport_refusal(&request());
        assert_eq!(frozen.code, "cell_not_licensed");
        assert_eq!(frozen.detail, learned::DETAIL_ROUTE_FROZEN);
        let admg = learned_joint_transport_refusal(&LearnedJointRequest {
            graph_class: "admg",
            ..request()
        });
        assert_eq!(admg.code, "route_not_supported");
        assert_eq!(admg.detail, learned::DETAIL_UNSUPPORTED_GRAPH);
        let basis =
            learned_joint_transport_refusal(&LearnedJointRequest { basis_degree: 9, ..request() });
        assert_eq!(basis.detail, learned::DETAIL_INVALID_BASIS);
    }
}
