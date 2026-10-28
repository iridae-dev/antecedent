//! Declared missing-data handling at the facade: which policies a route can execute, the
//! assumption each one needs, and the report a result carries.
//!
//! The route table is the single statement of what is supported. A declaration the table
//! does not list for the query and data modality is refused when the study is built, so a
//! policy is never accepted and then ignored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalQuery, DeclaredSamplingDesign};
use antecedent_data::{MissingDeclaration, MissingPolicy, TableView};
use antecedent_estimate::missing_policy::{policy_not_applicable, policy_not_applicable_next};
use antecedent_estimate::multiple_imputation;

use super::builder::DataInput;
use super::missing_support::{
    MissingInference, MissingModality, MissingRouteCell, route_missing_support,
};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::strategy_table::EstimatorId;

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
            (
                CausalQuery::TemporalEffect(_) | CausalQuery::Mediation(_),
                DataInput::Temporal(_),
            ) => Some(Self::Temporal),
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
            (Self::Static, MissingPolicy::MultipleImputation(_)) => {
                "multiple imputation: every row is kept and its missing treatment, outcome, adjustment or auxiliary values are drawn from a proper imputation model; valid when missingness is at random given the variables of the imputation model and that model is correctly specified; the completed-table fits are pooled by Rubin's rules (frequentist) or stacked into one posterior (Bayesian)"
            }
            (Self::Randomized, MissingPolicy::MultipleImputation(_)) => {
                "multiple imputation of missing outcomes in a randomized design: every assigned participant is kept and a missing outcome (or covariate) is drawn from a proper imputation model that holds the arm each participant was assigned beside the other numeric columns; valid when outcomes are missing at random given the arm and those columns and the model is correctly specified (it enters the arm as a main effect, so an arm-specific outcome relation it omits biases the imputed effect toward zero); the design's own estimator and its declared assignment probabilities run unchanged on every completed table, and the fits are pooled by Rubin's rules (frequentist) or stacked into one posterior (Bayesian)"
            }
            (Self::Temporal, MissingPolicy::MultipleImputation(_)) => {
                "multiple imputation of a series with missing time points: every time step is kept and a missing value is drawn from a lagged imputation model (each variable given the others at the same step and every variable at the neighbouring steps, parameters drawn by a block Bayesian bootstrap); valid when missingness is at random given the neighbouring steps' values and the series is stationary with dependence no longer than the conditioned neighbourhood; the temporal estimator runs on every completed series and the fits are pooled by Rubin's rules (frequentist) or stacked into one posterior (Bayesian)"
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

/// The route cell of a study: its query kind, data modality and inference mode.
pub(crate) fn route_cell(
    query: &CausalQuery,
    data: &DataInput,
    inference: &InferenceMode,
) -> MissingRouteCell {
    MissingRouteCell {
        query: query.kind(),
        modality: match data {
            DataInput::Tabular(_) => MissingModality::Tabular,
            DataInput::Temporal(_) => MissingModality::Series,
            DataInput::Event(_) => MissingModality::Event,
            DataInput::MultiEnv(_) => MissingModality::MultiEnv,
            DataInput::Panel(_) => MissingModality::Panel,
        },
        inference: if inference.is_bayesian() {
            MissingInference::Bayesian
        } else {
            MissingInference::Frequentist
        },
    }
}

/// What the builder knows about a study when it checks a missing-data declaration.
pub(crate) struct DeclarationContext<'a> {
    /// The query, when one is set.
    pub(crate) query: Option<&'a CausalQuery>,
    /// The data.
    pub(crate) data: &'a DataInput,
    /// Inference mode.
    pub(crate) inference: &'a InferenceMode,
    /// The estimator the caller chose, if any.
    pub(crate) estimator: Option<EstimatorId>,
    /// Whether the study averages over a graph posterior.
    pub(crate) graph_posterior: bool,
    /// Declared sampling design.
    pub(crate) sampling_design: Option<DeclaredSamplingDesign>,
}

/// Refuse a declaration the study's route cell cannot execute.
///
/// The route cell's row of `parity/missing_policy_support.toml` decides: a policy it does
/// not list is refused naming the supported policies and the next attempt. A multiple
/// imputation declaration is further refused when its specification is out of range or
/// declares missingness not at random (next attempt: a sensitivity analysis), when the
/// chosen estimator is not one the cell pools, when the study averages over a graph
/// posterior, or when the sampling design has no exchangeable unit for the imputation
/// model's parameter draw.
///
/// # Errors
///
/// `missing_policy_not_applicable` naming the policies the route supports, or the absence
/// of any route-level policy for a query that reads rows in its own way.
pub(crate) fn validate_declaration(
    context: &DeclarationContext<'_>,
    declaration: Option<&MissingDeclaration>,
) -> Result<(), CausalError> {
    let (Some(query), Some(declaration)) = (context.query, declaration) else {
        return Ok(());
    };
    let cell = route_cell(query, context.data, context.inference);
    let Some(row) = route_missing_support(cell) else {
        return Err(CausalError::Estimate(policy_not_applicable(
            declaration.policy,
            "this query on this data",
            "the route reads its rows in its own way and does not execute a missing-data policy",
            &[],
        )));
    };
    let label = MissingRoute::of(query, context.data).map_or("this route", MissingRoute::label);
    if !row.supports(declaration.policy) {
        return Err(CausalError::Estimate(policy_not_applicable_next(
            declaration.policy,
            label,
            match declaration.policy {
                MissingPolicy::CompleteCase if row.family == "panel" => {
                    "deleting single unit-periods would unbalance the panel and relabel its units"
                }
                MissingPolicy::InverseProbabilityOfObservation => {
                    "only randomized outcomes have a responder-weighted estimator"
                }
                MissingPolicy::BalancedPanel | MissingPolicy::AvailableCase => {
                    "only panel designs have units to restrict"
                }
                MissingPolicy::MultipleImputation(_) => {
                    "multiple imputation pools estimators that report a point and a variance or posterior draws, and this route cell has no pooled estimator"
                }
                _ => "the route has no estimator for it",
            },
            &row.supported,
            row.next_attempt,
        )));
    }
    let imputation = declaration.policy.imputation();
    if !declaration.covariates.is_empty()
        && declaration.policy != MissingPolicy::InverseProbabilityOfObservation
        && imputation.is_none()
    {
        return Err(CausalError::Estimate(policy_not_applicable(
            declaration.policy,
            label,
            "observation-model covariates are read only by inverse-probability weighting, and auxiliary variables only by multiple imputation",
            &row.supported,
        )));
    }
    if let Some(spec) = imputation {
        multiple_imputation::check_spec(&spec)
            .map_err(|error| CausalError::Estimate(error.into()))?;
        if let Some(estimator) = context.estimator {
            if !row.imputes_with(estimator.as_str()) {
                return Err(CausalError::Estimate(policy_not_applicable_next(
                    declaration.policy,
                    label,
                    &format!(
                        "multiple imputation pools the estimators {:?} on this route cell, and `{}` was chosen",
                        row.imputation_estimators,
                        estimator.as_str()
                    ),
                    &row.supported,
                    MissingPolicy::CompleteCase,
                )));
            }
        }
        if context.graph_posterior {
            graph_posterior_pooling(context, query, label)?;
        }
        match MissingRoute::of(query, context.data) {
            // A single series is imputed by a lagged model whose parameter draw resamples
            // contiguous blocks; it is the dependent time-series design that declares them.
            Some(MissingRoute::Temporal) => {
                multiple_imputation::bind_series_units(context.sampling_design, 0, 1)
                    .map_err(|error| CausalError::Estimate(error.into()))?;
            }
            // A clustered randomized design takes its clusters from the assignment units.
            Some(MissingRoute::Randomized)
                if context.sampling_design == Some(DeclaredSamplingDesign::ClusteredPanel) => {}
            _ => {
                multiple_imputation::bind_units(context.sampling_design, None, 0)
                    .map_err(|error| CausalError::Estimate(error.into()))?;
            }
        }
    }
    Ok(())
}

/// Whether a graph-posterior mixture can be pooled over completed tables, and the typed
/// refusal naming why not when it cannot.
///
/// A mixture over structures is not a point and a variance, so Rubin's rules do not apply: the
/// completed tables' effect posteriors are stacked with equal weight, each keeping its own
/// graph weights, which is the posterior of the effect given the declared structure
/// distribution and the imputation model. That is defined for a Bayesian study of a static
/// effect on tabular rows whose per-table answer is one effect posterior. Every other
/// construction has no justified pooling rule and is refused with a detail naming the reason.
///
/// # Errors
///
/// `missing_policy_not_applicable` with detail `multiple_imputation.graph_posterior_*`.
fn graph_posterior_pooling(
    context: &DeclarationContext<'_>,
    query: &CausalQuery,
    label: &str,
) -> Result<(), CausalError> {
    let refuse = |detail: &str, why: &str| {
        Err(CausalError::Estimate(antecedent_estimate::EstimationError::refused_detail(
            antecedent_core::reason_code!("missing_policy_not_applicable"),
            detail,
            format!(
                "{label} cannot pool a graph-posterior mixture over completed tables: {why}; \
                 next attempt: declare `complete_case`"
            ),
        )))
    };
    if !matches!(context.data, DataInput::Tabular(_)) {
        return refuse(
            "multiple_imputation.graph_posterior_series",
            "a series graph posterior is answered by dynamic-network and class executors whose \
             per-table answers are per-horizon and per-atom structures with no single effect \
             posterior to stack, and the dependent series has no exchangeable units for a \
             Rubin-style variance",
        );
    }
    if !context.inference.is_bayesian() {
        return refuse(
            "multiple_imputation.graph_posterior_frequentist",
            "a mixture over structures has no frequentist point and variance; only effect \
             posteriors can be mixed",
        );
    }
    if !matches!(query, CausalQuery::AverageEffect(_) | CausalQuery::ConditionalEffect(_)) {
        return refuse(
            "multiple_imputation.graph_posterior_query",
            "this query is answered atom by atom (responses, mediation, counterfactuals, \
             transport, attribution), and its per-table answer is not one equal-weight \
             mixture of effect draws",
        );
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
    crate::result::MissingDataReport {
        accounting,
        declared,
        covariates,
        assumption,
        non_default,
        imputation: None,
    }
}

/// The identity entry of a study's missing-data declaration: present only when the policy
/// differs from its route's default, so every study that keeps the default has the identity
/// it always had.
pub(crate) fn policy_wire(
    study: &super::execute::Study,
) -> Option<antecedent_io::MissingPolicyWire> {
    let route = MissingRoute::of(&study.query, &study.data);
    let declaration = non_default(route, study.declared.missing.as_ref())?;
    let names = match &study.data {
        DataInput::Tabular(data) => declaration
            .covariates
            .iter()
            .filter_map(|&id| data.schema().get(id).ok().map(|v| v.name.to_string()))
            .collect(),
        _ => Vec::new(),
    };
    let series = matches!(study.data, DataInput::Temporal(_));
    let imputation =
        declaration.policy.imputation().map(|spec| antecedent_io::ImputationSpecWire {
            model: spec.model.name().to_owned(),
            imputations: spec.imputations,
            iterations: spec.iterations,
            seed: spec.seed,
            assumption: spec.assumption.name().to_owned(),
            unit: if series {
                "group"
            } else {
                imputation_unit_name(study.declared.sampling_design)
            }
            .to_owned(),
            assumptions_digest: super::missing_imputation::declared_assumptions_digest(study),
        });
    Some(antecedent_io::MissingPolicyWire {
        policy: declaration.policy.name().to_owned(),
        covariates: names,
        imputation,
    })
}

/// The unit an imputation's parameter draw resamples under `design` on a static route.
pub(crate) const fn imputation_unit_name(design: Option<DeclaredSamplingDesign>) -> &'static str {
    match design {
        Some(DeclaredSamplingDesign::ClusteredPanel) => "group",
        _ => "row",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::missing_support::missing_support_table;

    fn family(route: MissingRoute) -> &'static str {
        match route {
            MissingRoute::Static => "static",
            MissingRoute::Randomized => "randomized",
            MissingRoute::Panel => "panel",
            MissingRoute::Temporal => "temporal",
        }
    }

    /// The executors' own policy lists agree with the registry: a row executes exactly its
    /// family's policies, plus multiple imputation where the row pools an estimator.
    #[test]
    fn route_families_agree_with_the_registry() {
        for row in missing_support_table() {
            let route = [
                MissingRoute::Static,
                MissingRoute::Randomized,
                MissingRoute::Panel,
                MissingRoute::Temporal,
            ]
            .into_iter()
            .find(|route| family(*route) == row.family)
            .expect("known family");
            assert_eq!(row.default, route.default_policy(), "{:?}", row.query);
            let without_imputation: Vec<&str> = row
                .supported
                .iter()
                .filter(|policy| policy.imputation().is_none())
                .map(|policy| policy.name())
                .collect();
            let executed: Vec<&str> = route.supported().iter().map(|p| p.name()).collect();
            assert_eq!(without_imputation, executed, "{:?}", row.query);
        }
    }
}
