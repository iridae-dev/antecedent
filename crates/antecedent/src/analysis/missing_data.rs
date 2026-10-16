//! Declared missing-data handling at the facade: which policies a route can execute, the
//! assumption each one needs, and the report a result carries.
//!
//! The route table is the single statement of what is supported. A declaration the table
//! does not list for the query and data modality is refused when the study is built, so a
//! policy is never accepted and then ignored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CausalQuery;
use antecedent_data::{MissingDeclaration, MissingPolicy, TableView};
use antecedent_estimate::missing_policy::policy_not_applicable;

use super::builder::DataInput;
use crate::error::CausalError;

/// Policies a static adjustment route can execute; its own default is complete-case.
const STATIC: &[MissingPolicy] = &[MissingPolicy::ErrorOnMissing, MissingPolicy::CompleteCase];

/// Policies a randomized two-arm outcome can execute; it refuses by default.
const RANDOMIZED: &[MissingPolicy] = &[
    MissingPolicy::ErrorOnMissing,
    MissingPolicy::CompleteCase,
    MissingPolicy::InverseProbabilityOfObservation,
];

/// Policies a panel difference-in-differences can execute; it refuses by default.
const PANEL: &[MissingPolicy] =
    &[MissingPolicy::ErrorOnMissing, MissingPolicy::BalancedPanel, MissingPolicy::AvailableCase];

/// Policies a lagged temporal effect can execute; it refuses by default.
const TEMPORAL: &[MissingPolicy] = &[MissingPolicy::ErrorOnMissing, MissingPolicy::CompleteCase];

/// A route family that executes a missing-data policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MissingRoute {
    /// Tabular adjustment routes that select complete cases.
    Static,
    /// Randomized two-arm ITT.
    Randomized,
    /// Panel difference-in-differences.
    Panel,
    /// Lagged temporal effect on a single series.
    Temporal,
}

impl MissingRoute {
    /// The executing route family for `query` on `data`, if any.
    pub(crate) fn of(query: &CausalQuery, data: &DataInput) -> Option<Self> {
        match (query, data) {
            (
                CausalQuery::AverageEffect(_)
                | CausalQuery::ConditionalEffect(_)
                | CausalQuery::Response(_)
                | CausalQuery::Distribution(_)
                | CausalQuery::Mediation(_),
                DataInput::Tabular(_),
            ) => Some(Self::Static),
            (CausalQuery::RandomizedEffect(_), DataInput::Tabular(_)) => Some(Self::Randomized),
            (CausalQuery::PanelDid(_), DataInput::Tabular(_)) => Some(Self::Panel),
            (CausalQuery::TemporalEffect(_), DataInput::Temporal(_)) => Some(Self::Temporal),
            _ => None,
        }
    }

    /// Policies the route can execute, in documentation order.
    pub(crate) fn supported(self) -> &'static [MissingPolicy] {
        match self {
            Self::Static => STATIC,
            Self::Randomized => RANDOMIZED,
            Self::Panel => PANEL,
            Self::Temporal => TEMPORAL,
        }
    }

    /// The policy applied when none is declared.
    pub(crate) fn default_policy(self) -> MissingPolicy {
        match self {
            Self::Static => MissingPolicy::CompleteCase,
            Self::Randomized | Self::Panel | Self::Temporal => MissingPolicy::ErrorOnMissing,
        }
    }

    /// Name used in refusals.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Static => "this adjustment route",
            Self::Randomized => "randomized intention-to-treat estimation",
            Self::Panel => "panel difference-in-differences",
            Self::Temporal => "lagged temporal estimation",
        }
    }

    /// The missingness assumption `policy` needs on this route, as published in the
    /// missing-data guide.
    pub(crate) fn assumption(self, policy: MissingPolicy) -> &'static str {
        match (self, policy) {
            (Self::Static, MissingPolicy::CompleteCase) => {
                "complete-case selection: rows with a missing treatment, outcome or adjustment value are dropped and the estimate speaks for the retained rows; it is valid for the full population when missingness is independent of the outcome given the treatment and adjustment covariates (missing completely at random, or at random given the adjustment set)"
            }
            (Self::Randomized, MissingPolicy::CompleteCase) => {
                "outcomes missing completely at random within arm: the probability that an outcome is observed does not depend on the potential outcomes within an arm; the assignment of every participant is retained and only the responders' outcomes are averaged"
            }
            (Self::Randomized, MissingPolicy::InverseProbabilityOfObservation) => {
                "outcomes missing at random given arm and the declared baseline covariates: within an arm the probability of observing the outcome depends only on those covariates and not on the outcome itself, it is bounded away from zero, and it follows the declared logistic model; every assigned participant stays in its arm's denominator"
            }
            (Self::Panel, MissingPolicy::BalancedPanel) => {
                "units with a missing outcome in any period are dropped whole; the retained units represent all units when a unit's chance of being complete is independent of its outcome changes within cohort"
            }
            (Self::Panel, MissingPolicy::AvailableCase) => {
                "each contrast uses the units observed in its baseline and target period; valid when a unit's chance of being observed in both periods is independent of its outcome change within cohort and cluster"
            }
            (Self::Temporal, MissingPolicy::CompleteCase) => {
                "windows containing a missing value are dropped and time keeps its original grid, so no lag spans a gap; the retained windows represent the process when missingness is independent of the outcome innovation given the lagged regressors"
            }
            (_, MissingPolicy::ErrorOnMissing) => {
                "no missing value is permitted; the route refuses the table at the first missing cell"
            }
            _ => "no assumption: this policy is not executed on this route",
        }
    }
}

/// Refuse a declaration the query and data modality cannot execute.
///
/// # Errors
///
/// `missing_policy_not_applicable` naming the policies the route supports, or the absence
/// of any route-level policy for a query that reads rows in its own way.
pub(crate) fn validate_declaration(
    query: Option<&CausalQuery>,
    data: &DataInput,
    declaration: Option<&MissingDeclaration>,
) -> Result<(), CausalError> {
    let (Some(query), Some(declaration)) = (query, declaration) else {
        return Ok(());
    };
    let Some(route) = MissingRoute::of(query, data) else {
        return Err(CausalError::Estimate(policy_not_applicable(
            declaration.policy,
            "this query on this data",
            "the route reads its rows in its own way and does not execute a missing-data policy",
            &[],
        )));
    };
    if !route.supported().contains(&declaration.policy) {
        return Err(CausalError::Estimate(policy_not_applicable(
            declaration.policy,
            route.label(),
            match declaration.policy {
                MissingPolicy::CompleteCase if route == MissingRoute::Panel => {
                    "deleting single unit-periods would unbalance the panel and relabel its units"
                }
                MissingPolicy::InverseProbabilityOfObservation => {
                    "only randomized outcomes have a responder-weighted estimator"
                }
                MissingPolicy::BalancedPanel | MissingPolicy::AvailableCase => {
                    "only panel designs have units to restrict"
                }
                _ => "the route has no estimator for it",
            },
            route.supported(),
        )));
    }
    if !declaration.covariates.is_empty()
        && declaration.policy != MissingPolicy::InverseProbabilityOfObservation
    {
        return Err(CausalError::Estimate(policy_not_applicable(
            declaration.policy,
            route.label(),
            "observation-model covariates are read only by inverse-probability weighting",
            route.supported(),
        )));
    }
    Ok(())
}

/// The policy in force and whether the caller declared it.
pub(crate) fn effective_policy(
    route: MissingRoute,
    declaration: Option<&MissingDeclaration>,
) -> (MissingPolicy, bool) {
    declaration.map_or((route.default_policy(), false), |d| (d.policy, true))
}

/// The declaration, when it differs from the route's default and so changes the analysis.
pub(crate) fn non_default(
    route: Option<MissingRoute>,
    declaration: Option<&MissingDeclaration>,
) -> Option<&MissingDeclaration> {
    let declaration = declaration?;
    let default = route.map(MissingRoute::default_policy);
    (Some(declaration.policy) != default || !declaration.covariates.is_empty())
        .then_some(declaration)
}

/// Build the report a result carries.
pub(crate) fn report(
    route: MissingRoute,
    declared: bool,
    covariates: Arc<[Arc<str>]>,
    accounting: antecedent_data::RowAccounting,
) -> crate::result::MissingDataReport {
    let assumption = Arc::from(route.assumption(accounting.policy));
    let non_default =
        declared && (accounting.policy != route.default_policy() || !covariates.is_empty());
    crate::result::MissingDataReport { accounting, declared, covariates, assumption, non_default }
}

/// The identity entry of a study's missing-data declaration: present only when the policy
/// differs from its route's default, so every study that keeps the default has the identity
/// it always had.
pub(crate) fn policy_wire(
    study: &super::execute::Study,
) -> Option<antecedent_io::MissingPolicyWire> {
    let route = MissingRoute::of(&study.query, &study.data);
    let declaration = non_default(route, study.missing.as_ref())?;
    let names = match &study.data {
        DataInput::Tabular(data) => declaration
            .covariates
            .iter()
            .filter_map(|&id| data.schema().get(id).ok().map(|v| v.name.to_string()))
            .collect(),
        _ => Vec::new(),
    };
    Some(antecedent_io::MissingPolicyWire {
        policy: declaration.policy.name().to_owned(),
        covariates: names,
    })
}
