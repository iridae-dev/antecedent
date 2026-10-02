//! Estimator menu for a binary trial-to-target contrast (2.2A cell X4).
//!
//! The menu is inspection: for one certified graph, one query and one learner
//! configuration it lists every estimator this release names, whether it is eligible, and
//! the laws, graph conditions, nuisance tasks, support and sampling design it requires,
//! its uncertainty status, and why an ineligible estimator is refused. It fits nothing.
//!
//! Which fields are computed and which are fixed descriptions is explicit. For the two
//! trial estimators the required laws, graph conditions, nuisance tasks, support
//! thresholds, sampling design and uncertainty status are computed from the certificate,
//! the learner specs and the request options in force ([`MenuContext`]); every entry
//! lists in `static_fields` the fields that are fixed descriptions of a named-but-refused
//! estimator or of an estimator this cell does not calibrate. Variables appear as
//! `v<coordinate>` because names live with the caller.
//!
//! Selection stays manual. The menu never ranks or labels an entry "recommended":
//! no licensed comparison criterion exists between these estimators.
use crate::learned_continuous::{LearnedContinuousOptions, LearnedContinuousUncertainty};
use crate::learned_trial::TrialSampling;
use antecedent_core::TransportQuery;
use antecedent_identify::{TransportFormula, TransportIdentification};
use antecedent_learn::LearnerSpec;
use serde::{Deserialize, Serialize};

/// Why an estimator is refused for this graph, query and provider.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MenuRefusal {
    /// Registered runtime reason code.
    pub code: String,
    /// Namespaced detail when the refusal has one.
    pub detail: Option<String>,
    /// The graph, query or provider fact that refuses it.
    pub reason: String,
}

/// One estimator of the menu.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EstimatorMenuEntry {
    /// Stable estimator name.
    pub estimator: String,
    /// Whether the estimator may run on this graph, query and provider.
    pub eligible: bool,
    /// Laws the certificate and the evidence must supply.
    pub required_laws: Vec<String>,
    /// Graph conditions the certificate must establish.
    pub required_graph_conditions: Vec<String>,
    /// Nuisance tasks the estimator learns or is given.
    pub nuisance_tasks: Vec<String>,
    /// Support (overlap) the estimator requires.
    pub support_requirements: Vec<String>,
    /// Sampling designs it is licensed for.
    pub sampling_design: Vec<String>,
    /// Uncertainty status of its interval.
    pub uncertainty_status: String,
    /// Fields (by name) that are fixed descriptions, not computed from the graph,
    /// query, learners or options in force.
    pub static_fields: Vec<String>,
    /// Why it is refused; present exactly when `eligible` is false.
    pub refusal: Option<MenuRefusal>,
}

/// The request facts a menu is computed from. Anything left `None` lists the entry under
/// the release defaults and says so.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MenuContext {
    /// The full request options in force (learners, folds, thresholds, bootstrap).
    pub options: Option<LearnedContinuousOptions>,
    /// `(outcome, membership)` learners when no full options are supplied.
    pub learners: Option<(LearnerSpec, LearnerSpec)>,
    /// The declared sampling design; `None` lists both licensed designs.
    pub sampling: Option<TrialSampling>,
}

/// The menu for one graph, query and provider.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EstimatorMenu {
    /// Always `manual`: the caller selects an estimator; nothing is recommended.
    pub selection: String,
    /// Every estimator this release names, eligible or not.
    pub entries: Vec<EstimatorMenuEntry>,
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_owned()).collect()
}

fn refusal(code: &'static str, detail: Option<&str>, reason: impl Into<String>) -> MenuRefusal {
    MenuRefusal { code: code.into(), detail: detail.map(Into::into), reason: reason.into() }
}

/// The refusal that graph and query put on every trial-to-target estimator, if any.
fn structural_refusal(id: &TransportIdentification, query: &TransportQuery) -> Option<MenuRefusal> {
    if let Err(error) = crate::validate_trial_query(query) {
        return Some(refusal(
            antecedent_core::reason_code!("route_not_supported"),
            None,
            error.to_string(),
        ));
    }
    match id {
        TransportIdentification::Transportable {
            formula: TransportFormula::Direct(_) | TransportFormula::Standardize { .. },
            ..
        } => None,
        TransportIdentification::Transportable {
            formula: TransportFormula::RecursiveFactorization { .. },
            certificate,
        } => Some(refusal(
            antecedent_core::reason_code!("transport_not_certified"),
            None,
            format!(
                "certificate rule '{}' yields a recursive factorization, which is identify-only",
                certificate.rule
            ),
        )),
        TransportIdentification::NotCertified(certificate) => Some(refusal(
            antecedent_core::reason_code!("transport_not_certified"),
            None,
            format!("{}: {}", certificate.reason, certificate.message),
        )),
        TransportIdentification::MissingEvidence(certificate) => Some(refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            None,
            format!("{}: {}", certificate.reason, certificate.message),
        )),
    }
}

fn provider_refusal(learners: Option<(LearnerSpec, LearnerSpec)>) -> Option<MenuRefusal> {
    let (outcome, membership) = learners?;
    outcome
        .validate()
        .and_then(|()| membership.validate())
        .err()
        .map(|e| refusal(antecedent_core::reason_code!("invalid_argument"), None, e.to_string()))
}

fn ids(vars: &[antecedent_core::VariableId]) -> String {
    if vars.is_empty() {
        return "none".into();
    }
    vars.iter().map(|v| format!("v{}", v.raw())).collect::<Vec<_>>().join(", ")
}

/// The certificate-derived requirement lines of the two trial estimators.
struct CertificateRequirements {
    /// Laws the learned augmented estimator needs: outcome mean per arm, membership
    /// probability and the known randomization.
    learned_laws: Vec<String>,
    /// Laws the supplied-probability IPW estimator needs: the supplied membership
    /// probability and the known randomization (no outcome regression).
    ipw_laws: Vec<String>,
    /// Graph conditions the certificate establishes.
    conditions: Vec<String>,
}

/// What the certificate requires. A standardization certificate conditions every law
/// on its standardizers, which must be exactly the supplied baseline covariates; a
/// direct certificate has no standardizer and admits no covariates
/// ([`crate::validate_trial_aipw`] refuses any), so its membership and randomization
/// laws are marginal.
fn certificate_requirements(id: &TransportIdentification) -> CertificateRequirements {
    match id {
        TransportIdentification::Transportable {
            formula: TransportFormula::RecursiveFactorization { .. },
            certificate,
        } => CertificateRequirements {
            learned_laws: strings(&[
                "not derivable by a trial estimator: the certificate is a recursive \
                 factorization (identify-only), not a direct or standardization formula",
            ]),
            ipw_laws: strings(&[
                "not derivable by a trial estimator: the certificate is a recursive \
                 factorization (identify-only), not a direct or standardization formula",
            ]),
            conditions: vec![format!(
                "certificate rule '{}' (recursive factorization): no trial estimator evaluates it",
                certificate.rule
            )],
        },
        TransportIdentification::Transportable { formula, certificate } => {
            let (kind, outcome_law, membership_law, randomization_law, covariate_condition) =
                match formula {
                    TransportFormula::Standardize { over, .. } => {
                        let x = ids(over);
                        (
                            "baseline standardization",
                            format!(
                                "source outcome mean E(Y | X=[{x}], A, S=1) per randomized arm"
                            ),
                            format!("source-membership probability P(S=1 | X=[{x}])"),
                            format!("known randomization probabilities P(A=1 | X=[{x}], S=1)"),
                            format!(
                                "the certified standardizers [{x}] equal the supplied baseline \
                                 covariates"
                            ),
                        )
                    }
                    _ => (
                        "direct transport",
                        "source outcome mean E(Y | A, S=1) per randomized arm (no standardizer)"
                            .to_owned(),
                        "marginal source-membership probability P(S=1) (no standardizer)"
                            .to_owned(),
                        "known randomization probabilities P(A=1 | S=1)".to_owned(),
                        "no baseline covariates are supplied (a direct certificate admits none)"
                            .to_owned(),
                    ),
                };
            let mut conditions = vec![
                format!("certificate rule '{}' ({kind})", certificate.rule),
                format!("selection acts on [{}] only", ids(&certificate.selection_targets)),
            ];
            conditions.extend(certificate.premises.iter().map(|p| format!("premise: {p}")));
            conditions.push(covariate_condition);
            CertificateRequirements {
                ipw_laws: vec![format!("supplied {membership_law}"), randomization_law.clone()],
                learned_laws: vec![outcome_law, membership_law, randomization_law],
                conditions,
            }
        }
        _ => CertificateRequirements {
            learned_laws: strings(&[
                "not derivable: no certificate (source outcome mean per arm, \
                 membership probability and known randomization would be required)",
            ]),
            ipw_laws: strings(&["not derivable: no certificate (supplied membership probability \
                 and known randomization would be required)"]),
            conditions: strings(&["not derivable: the graph and query carry no certificate"]),
        },
    }
}

const fn sampling_name(sampling: TrialSampling) -> &'static str {
    match sampling {
        TrialSampling::NestedCohort => "nested_cohort",
        TrialSampling::IndependentSamples => "independent_samples",
    }
}

fn learner_line(role: &str, spec: LearnerSpec, folds: usize, defaulted: bool) -> String {
    let tag = if defaulted { " (default: none requested)" } else { "" };
    format!(
        "{role}, learner {}, {folds}-fold cross-fit on one shared fold assignment{tag}",
        spec.name()
    )
}

/// List the estimators for a binary trial-to-target contrast on this graph and query.
///
/// `learners` are the outcome and membership `LearnerSpec`s a learned estimator would
/// use; `None` lists the learned entry under the default learners. See
/// [`transport_estimator_menu_with`] to compute the menu from full request options.
#[must_use]
pub fn transport_estimator_menu(
    id: &TransportIdentification,
    query: &TransportQuery,
    learners: Option<(LearnerSpec, LearnerSpec)>,
) -> EstimatorMenu {
    transport_estimator_menu_with(id, query, &MenuContext { learners, ..MenuContext::default() })
}

/// The menu computed from the certificate, the query and the request facts in `context`.
#[must_use]
#[allow(clippy::too_many_lines, reason = "one literal entry per named estimator")]
#[doc(hidden)]
pub fn transport_estimator_menu_with(
    id: &TransportIdentification,
    query: &TransportQuery,
    context: &MenuContext,
) -> EstimatorMenu {
    let structural = structural_refusal(id, query);
    let defaults = LearnedContinuousOptions::default();
    let defaulted = context.options.is_none() && context.learners.is_none();
    let mut options = context.options.unwrap_or(defaults);
    if context.options.is_none() {
        if let Some((outcome, membership)) = context.learners {
            options.outcome = outcome;
            options.membership = membership;
        }
    }
    let learners = context.options.map_or(context.learners, |o| Some((o.outcome, o.membership)));
    let designs = context.sampling.map_or_else(
        || strings(&["nested_cohort", "independent_samples"]),
        |sampling| vec![sampling_name(sampling).to_owned()],
    );
    let CertificateRequirements { learned_laws: laws, ipw_laws, conditions: graph_conditions } =
        certificate_requirements(id);
    let learned_refusal = structural.clone().or_else(|| provider_refusal(learners));
    let uncertainty = LearnedContinuousUncertainty::for_request(options.bootstrap);
    let detail = uncertainty.detail.as_deref().map_or_else(String::new, |d| format!(" ({d})"));
    let learned_uncertainty = format!(
        "{}: {}{detail}; {} bootstrap replicates requested; the joint outer bootstrap interval \
         is uncalibrated and its route is closed (cell_not_licensed)",
        uncertainty.status, uncertainty.reason, options.bootstrap
    );
    let floor = options.min_treatment_probability;
    let all_requirements = [
        "required_laws",
        "required_graph_conditions",
        "nuisance_tasks",
        "support_requirements",
        "sampling_design",
        "uncertainty_status",
    ];
    let entries = vec![
        EstimatorMenuEntry {
            estimator: "learned_trial_aipw".into(),
            eligible: learned_refusal.is_none(),
            required_laws: laws,
            required_graph_conditions: graph_conditions.clone(),
            nuisance_tasks: vec![
                learner_line(
                    "outcome regression, one fit per randomized arm",
                    options.outcome,
                    options.folds,
                    defaulted,
                ),
                learner_line(
                    "source-membership classification",
                    options.membership,
                    options.folds,
                    defaulted,
                ),
            ],
            support_requirements: vec![
                format!(
                    "out-of-fold membership probability at least {} on every row",
                    options.min_membership_probability
                ),
                format!(
                    "randomization probability within [{floor}, {}] on every source row",
                    1.0 - floor
                ),
            ],
            sampling_design: designs.clone(),
            uncertainty_status: learned_uncertainty,
            static_fields: vec![],
            refusal: learned_refusal,
        },
        EstimatorMenuEntry {
            estimator: "trial_ipw_supplied_probabilities".into(),
            eligible: structural.is_none(),
            required_laws: ipw_laws,
            required_graph_conditions: graph_conditions,
            nuisance_tasks: strings(&["none: membership probabilities are supplied by the caller"]),
            support_requirements: strings(&[
                "supplied membership and randomization probabilities inside (0, 1) on every row",
            ]),
            sampling_design: designs,
            uncertainty_status:
                "delta_method_se: the 2.1 analyze cell transport.trial_ipw; not calibrated by this cell"
                    .into(),
            static_fields: ["nuisance_tasks", "support_requirements", "uncertainty_status"]
                .map(String::from)
                .to_vec(),
            refusal: structural,
        },
        EstimatorMenuEntry {
            estimator: "dr_learner_cate".into(),
            eligible: false,
            required_laws: strings(&["conditional outcome mean per arm, evaluated pointwise"]),
            required_graph_conditions: strings(&["conditional (heterogeneous) transport identification"]),
            nuisance_tasks: strings(&["cross-fitted pseudo-outcome regression"]),
            support_requirements: strings(&["pointwise overlap at every evaluation point"]),
            sampling_design: strings(&["independent_samples"]),
            uncertainty_status: "uncalibrated: pointwise profiles are documented as uncalibrated".into(),
            static_fields: all_requirements.map(String::from).to_vec(),
            refusal: Some(refusal(
                antecedent_core::reason_code!("route_not_supported"),
                Some("learned_transport.cate_requested"),
                "heterogeneous and simultaneous targets are not licensed in this cell",
            )),
        },
        EstimatorMenuEntry {
            estimator: "exact_finite_law_evaluator".into(),
            eligible: false,
            required_laws: strings(&["finite discrete joint laws over every coordinate"]),
            required_graph_conditions: strings(&["any checked transport formula"]),
            nuisance_tasks: strings(&["none: laws are supplied"]),
            support_requirements: strings(&["finite support over every coordinate"]),
            sampling_design: strings(&["declared by the supplied laws"]),
            uncertainty_status: "point_only".into(),
            static_fields: all_requirements.map(String::from).to_vec(),
            refusal: Some(refusal(
                antecedent_core::reason_code!("transport_unsupported_evaluator"),
                None,
                "the outcome is continuous; a finite law cannot represent its mean contrast",
            )),
        },
        // 2.2B X4: the smoothed dose-response grid has its own query and menu
        // (`smoothed_dose::smoothed_dose_estimator_menu`); a binary contrast never runs it.
        EstimatorMenuEntry {
            estimator: "smoothed_dose_transport_aipw".into(),
            eligible: false,
            required_laws: strings(&[
                "source outcome regression over a randomized continuous dose",
                "known conditional dose density on a declared support",
            ]),
            required_graph_conditions: strings(&["the continuous dose as the single source experiment"]),
            nuisance_tasks: strings(&["cross-fitted outcome regression over the dose basis and membership"]),
            support_requirements: strings(&["every kernel window inside the declared dose support"]),
            sampling_design: strings(&["nested_cohort", "independent_samples"]),
            uncertainty_status: "withheld: cell_not_licensed (joint outer bootstrap route closed)".into(),
            static_fields: all_requirements.map(String::from).to_vec(),
            refusal: Some(refusal(
                antecedent_core::reason_code!("route_not_supported"),
                None,
                "the query is a binary [0,1] contrast; the smoothed dose-response grid needs a \
                 smoothed dose transport query",
            )),
        },
    ];
    EstimatorMenu { selection: "manual".into(), entries }
}
