//! Route-level refusals of the three 2.3A closed calibrated-interval pilots.
//!
//! The Rust cores of joint Bayesian transport (A2, `antecedent.transport.joint_bayesian`),
//! the binary nested-Markov pilot (A3, `antecedent.transport.binary_nested_markov`) and
//! sampled observation recovery (A6, `antecedent.transport.sampled_observation_recovery`)
//! exist and replay through independent io artifacts, but each claims a *calibrated*
//! interval whose coverage is measured only at the release cut. Until then the public
//! producer of each route is closed: it validates its request against the pilot's scope
//! and otherwise refuses with the live typed `cell_not_licensed` refusal
//! (`bayesian_transport.route_frozen`, `nested_markov.route_frozen`,
//! `sampled_recovery.route_frozen`). A request outside the pilot refuses first with the
//! pilot's own scope refusal (`route_not_supported` or an invalid-argument detail), so a
//! caller learns the real obstruction before the closed route.
//!
//! These functions return the refusal as plain data; a binding raises it. They read the
//! engines' own bounds, details and refusals, and never run a fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::joint_bayesian_transport as joint;
use antecedent_estimate::nested_markov_binary as nested;
use antecedent_estimate::{
    EstimationError, SAMPLED_RECOVERY_MAX_BINARY_VARIABLES, SAMPLED_RECOVERY_MAX_REPLICATES,
    SAMPLED_RECOVERY_MAX_ROWS, SAMPLED_RECOVERY_MIN_REPLICATES, SampledRecoveryDetail,
    sampled_recovery_route_frozen,
};
use std::collections::BTreeSet;

/// A closed-route refusal as plain data: registered reason code and namespaced detail.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosedPilotRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail (`bayesian_transport.*`, `nested_markov.*`, `sampled_recovery.*`).
    pub detail: String,
    /// What was refused.
    pub message: String,
}

impl ClosedPilotRefusal {
    fn new(code: &'static str, detail: impl Into<String>, message: impl Into<String>) -> Self {
        Self { code, detail: detail.into(), message: message.into() }
    }

    /// `<detail>: <message>`, the text a binding raises after the `reason=<code>:` prefix.
    #[must_use]
    pub fn text(&self) -> String {
        format!("{}: {}", self.detail, self.message)
    }
}

fn from_joint(refusal: &joint::JointTransportRefusal) -> ClosedPilotRefusal {
    ClosedPilotRefusal::new(refusal.code, refusal.detail, refusal.message.clone())
}

/// A request to the closed joint Bayesian transport route.
#[derive(Clone, Copy, Debug)]
pub struct JointBayesianRequest<'a> {
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
    /// Number of source trials.
    pub sources: usize,
    /// Whether a target covariate sample was supplied.
    pub has_target: bool,
    /// Requested posterior draws.
    pub draws: usize,
}

fn invalid_joint(message: impl Into<String>) -> ClosedPilotRefusal {
    ClosedPilotRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        joint::DETAIL_INVALID_INPUT,
        message,
    )
}

/// Number of model parameters the engine would declare, or `None` for an unknown spelling.
fn joint_parameters(request: &JointBayesianRequest<'_>) -> Option<usize> {
    let p = request.features;
    let (q, r) = match request.varying {
        "intercept" => (p.saturating_mul(2).saturating_add(1), 1),
        "intercept_and_covariates" => (p.saturating_add(1), p.saturating_add(1)),
        _ => return None,
    };
    let blocks = match request.sharing {
        "independent_varying_blocks" => request.sources,
        "shared_varying_block" => 1,
        _ => return None,
    };
    Some(q.saturating_add(blocks.saturating_mul(r)))
}

fn joint_scope_refusal(request: &JointBayesianRequest<'_>) -> Option<ClosedPilotRefusal> {
    match request.graph_class {
        "fixed_dag" => {}
        "admg" | "graph_posterior" => return Some(from_joint(&joint::unsupported_graph_refusal())),
        _ => {
            return Some(invalid_joint("graph_class must be fixed_dag, admg or graph_posterior"));
        }
    }
    if request.draws == 0 || request.draws > joint::JOINT_TRANSPORT_MAX_DRAWS {
        return Some(ClosedPilotRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            joint::DETAIL_TOO_MANY_DRAWS,
            format!("draw count must be in 1..={}", joint::JOINT_TRANSPORT_MAX_DRAWS),
        ));
    }
    if !matches!(request.dependence, "independent_samples" | "overlapping_units" | "unknown") {
        return Some(invalid_joint(
            "dependence must be independent_samples, overlapping_units or unknown",
        ));
    }
    if request.dependence != "independent_samples" {
        return Some(ClosedPilotRefusal::new(
            antecedent_core::reason_code!("sampling_dependence_unknown"),
            joint::DETAIL_SOURCE_DEPENDENCE,
            "only independent source samples with disjoint units are supported",
        ));
    }
    if request.sources == 0 {
        return Some(ClosedPilotRefusal::new(
            antecedent_core::reason_code!("transport_missing_evidence"),
            joint::DETAIL_MISSING_SOURCE,
            "at least one source trial is required",
        ));
    }
    if !request.has_target {
        return Some(ClosedPilotRefusal::new(
            antecedent_core::reason_code!("joint_law_required"),
            joint::DETAIL_MISSING_LAW,
            "the target covariate law (a nonempty target sample) is required",
        ));
    }
    let Some(parameters) = joint_parameters(request) else {
        return Some(invalid_joint(
            "varying must be intercept or intercept_and_covariates and sharing independent_varying_blocks or shared_varying_block",
        ));
    };
    if parameters > joint::JOINT_TRANSPORT_MAX_PARAMETERS {
        return Some(ClosedPilotRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            joint::DETAIL_TOO_MANY_PARAMETERS,
            format!(
                "the model has {parameters} parameters; the bound is {}",
                joint::JOINT_TRANSPORT_MAX_PARAMETERS
            ),
        ));
    }
    None
}

/// The refusal of a request to the closed joint Bayesian transport route: the pilot's
/// scope refusal when the request is outside it, otherwise
/// `cell_not_licensed` / `bayesian_transport.route_frozen`.
#[must_use]
pub fn joint_bayesian_transport_refusal(request: &JointBayesianRequest<'_>) -> ClosedPilotRefusal {
    joint_scope_refusal(request).unwrap_or_else(|| from_joint(&joint::route_frozen_refusal()))
}

/// One regime-specific count table of a nested-Markov request.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedRegimeCounts {
    /// Variables an interventional regime fixes; `None` for the observational regime.
    pub fixed: Option<Vec<usize>>,
    /// Declared levels per variable.
    pub levels: Vec<usize>,
    /// Cell counts, first variable most significant.
    pub cells: Vec<f64>,
}

/// A request to the closed binary nested-Markov route.
#[derive(Clone, Debug, PartialEq)]
pub struct NestedMarkovRequest {
    /// Variable names in coordinate order.
    pub variables: Vec<String>,
    /// Directed edges `(from, to)`.
    pub directed: Vec<(usize, usize)>,
    /// Bidirected edges.
    pub bidirected: Vec<(usize, usize)>,
    /// Regime-specific counts.
    pub regimes: Vec<ClosedRegimeCounts>,
}

fn from_estimation(error: &EstimationError) -> ClosedPilotRefusal {
    match error {
        EstimationError::Refused { code, message }
        | EstimationError::RefusedWithFields { code, message, .. } => {
            let (detail, rest) = message
                .split_once(": ")
                .unwrap_or(("nested_markov.invalid_input", message.as_str()));
            ClosedPilotRefusal::new(code, detail, rest)
        }
        other => ClosedPilotRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            "nested_markov.invalid_input",
            other.to_string(),
        ),
    }
}

/// The refusal of a request to the closed binary nested-Markov route: the pilot's scope
/// refusal (`route_not_supported` / `nested_markov.outside_binary_pilot`, or a count
/// refusal) when the graph, regime or domain is outside the one selected pilot, otherwise
/// `cell_not_licensed` / `nested_markov.route_frozen`. Never a nonidentification claim.
#[must_use]
pub fn nested_markov_refusal(request: &NestedMarkovRequest) -> ClosedPilotRefusal {
    let input = nested::NestedMarkovInput {
        graph: nested::AdmgDeclaration {
            variables: request.variables.clone(),
            directed: request.directed.clone(),
            bidirected: request.bidirected.clone(),
        },
        regimes: request
            .regimes
            .iter()
            .map(|regime| nested::RegimeCounts {
                regime: regime
                    .fixed
                    .clone()
                    .map_or(nested::Regime::Observational, nested::Regime::Interventional),
                levels: regime.levels.clone(),
                cells: regime.cells.clone(),
            })
            .collect(),
    };
    match nested::binary_cells(&input) {
        Err(error) => from_estimation(&error),
        Ok(_) => from_estimation(&nested::refuse_public_interval_route()),
    }
}

/// A request to the closed sampled observation-recovery route.
#[derive(Clone, Copy, Debug)]
pub struct SampledRecoveryRequest<'a> {
    /// Whether the m-graph's checked derivation recovered (`false`: a verified
    /// nonrecoverability witness).
    pub graph_recoverable: bool,
    /// Number of partially observed variables.
    pub partially_observed: usize,
    /// Number of fully observed variables.
    pub fully_observed: usize,
    /// Bootstrap replicates.
    pub replicates: usize,
    /// Rows as `(id, responses, proxies, fully)`.
    pub rows: &'a [(u64, u8, u8, u8)],
}

fn sampled(detail: SampledRecoveryDetail, message: impl Into<String>) -> ClosedPilotRefusal {
    ClosedPilotRefusal::new(detail.reason_code(), detail.detail(), message)
}

fn low_bits(n: usize) -> u8 {
    (0..n.min(8)).fold(0_u8, |acc, i| acc | (1_u8 << i))
}

fn sampled_row_refusal(request: &SampledRecoveryRequest<'_>) -> Option<ClosedPilotRefusal> {
    let k = request.partially_observed;
    let m = request.fully_observed;
    let invalid = |message: &str| Some(sampled(SampledRecoveryDetail::InvalidInput, message));
    if request.rows.is_empty() {
        return invalid("no observation rows");
    }
    if request.rows.len() > SAMPLED_RECOVERY_MAX_ROWS {
        return Some(sampled(
            SampledRecoveryDetail::BoundsExceeded,
            format!("{} rows; at most {SAMPLED_RECOVERY_MAX_ROWS}", request.rows.len()),
        ));
    }
    let (response_mask, full_mask) = (low_bits(k), low_bits(m));
    if request.rows.iter().any(|&(_, responses, proxies, fully)| {
        responses & !response_mask != 0 || proxies & !responses != 0 || fully & !full_mask != 0
    }) {
        return invalid("a row has a pattern the proxy model excludes");
    }
    let ids: BTreeSet<u64> = request.rows.iter().map(|row| row.0).collect();
    if ids.len() != request.rows.len() {
        return invalid("row ids are not unique");
    }
    let seen: BTreeSet<(u8, u8, u8)> = request
        .rows
        .iter()
        .map(|&(_, responses, proxies, fully)| (responses, proxies, fully))
        .collect();
    let mut zero = Vec::new();
    for proxies in 0..(1_u16 << k) {
        for fully in 0..(1_u16 << m) {
            let pattern = (
                response_mask,
                u8::try_from(proxies).unwrap_or(u8::MAX),
                u8::try_from(fully).unwrap_or(u8::MAX),
            );
            if !seen.contains(&pattern) {
                zero.push(pattern);
            }
        }
    }
    if zero.is_empty() {
        return None;
    }
    Some(sampled(
        SampledRecoveryDetail::UnrecoverablePattern,
        format!(
            "{} complete-case pattern(s) have zero count (first: responses={}, proxies={}, \
             fully={}): the recovery formula divides by a margin that contains them",
            zero.len(),
            zero[0].0,
            zero[0].1,
            zero[0].2
        ),
    ))
}

fn sampled_scope_refusal(request: &SampledRecoveryRequest<'_>) -> Option<ClosedPilotRefusal> {
    if !request.graph_recoverable {
        return Some(sampled(
            SampledRecoveryDetail::UnrecoverablePattern,
            "the m-graph has a verified nonrecoverability witness; there is no recovered law \
             to resample",
        ));
    }
    if request.partially_observed + request.fully_observed > SAMPLED_RECOVERY_MAX_BINARY_VARIABLES {
        return Some(sampled(
            SampledRecoveryDetail::BoundsExceeded,
            format!(
                "{} binary variables; at most {SAMPLED_RECOVERY_MAX_BINARY_VARIABLES}",
                request.partially_observed + request.fully_observed
            ),
        ));
    }
    if request.replicates > SAMPLED_RECOVERY_MAX_REPLICATES {
        return Some(sampled(
            SampledRecoveryDetail::BoundsExceeded,
            format!("{} replicates; at most {SAMPLED_RECOVERY_MAX_REPLICATES}", request.replicates),
        ));
    }
    if request.replicates < SAMPLED_RECOVERY_MIN_REPLICATES {
        return Some(sampled(
            SampledRecoveryDetail::InvalidInput,
            "too few replicates to form a 95% percentile interval",
        ));
    }
    sampled_row_refusal(request)
}

/// The refusal of a request to the closed sampled observation-recovery route: the
/// pilot's scope refusal (`route_not_supported` /
/// `sampled_recovery.unrecoverable_pattern` or `bounds_exceeded`, or an invalid-input
/// detail) when the m-graph, bounds or observation patterns violate the exact recovery
/// formula, otherwise `cell_not_licensed` / `sampled_recovery.route_frozen`.
#[must_use]
pub fn sampled_recovery_refusal(request: &SampledRecoveryRequest<'_>) -> ClosedPilotRefusal {
    sampled_scope_refusal(request).unwrap_or_else(|| {
        let frozen = sampled_recovery_route_frozen();
        ClosedPilotRefusal::new(frozen.reason_code(), frozen.detail.detail(), frozen.message)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint_request() -> JointBayesianRequest<'static> {
        JointBayesianRequest {
            graph_class: "fixed_dag",
            dependence: "independent_samples",
            varying: "intercept",
            sharing: "independent_varying_blocks",
            features: 1,
            sources: 2,
            has_target: true,
            draws: 1000,
        }
    }

    #[test]
    fn x4_joint_route_is_frozen_and_scope_refusals_come_first() {
        let frozen = joint_bayesian_transport_refusal(&joint_request());
        assert_eq!(frozen.code, "cell_not_licensed");
        assert_eq!(frozen.detail, joint::DETAIL_ROUTE_FROZEN);
        let admg = joint_bayesian_transport_refusal(&JointBayesianRequest {
            graph_class: "admg",
            ..joint_request()
        });
        assert_eq!(admg.code, "route_not_supported");
        assert_eq!(admg.detail, joint::DETAIL_UNSUPPORTED_GRAPH);
    }

    #[test]
    fn x10_sampled_route_is_frozen_and_a_zero_cell_is_unrecoverable() {
        let rows: Vec<(u64, u8, u8, u8)> = (0..4_u8).map(|i| (u64::from(i), 1, i & 1, 0)).collect();
        let request = SampledRecoveryRequest {
            graph_recoverable: true,
            partially_observed: 1,
            fully_observed: 0,
            replicates: 100,
            rows: &rows,
        };
        let frozen = sampled_recovery_refusal(&request);
        assert_eq!(frozen.code, "cell_not_licensed");
        assert_eq!(frozen.detail, "sampled_recovery.route_frozen");
        let missing = [(0_u64, 1_u8, 0_u8, 0_u8), (1, 1, 0, 0)];
        let zero = sampled_recovery_refusal(&SampledRecoveryRequest { rows: &missing, ..request });
        assert_eq!(zero.code, "route_not_supported");
        assert_eq!(zero.detail, "sampled_recovery.unrecoverable_pattern");
    }
}
