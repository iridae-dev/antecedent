//! Which graph classes a query kind is licensed on, per data modality.
//!
//! One declarative table, [`kind_license`], gives every [`CausalQueryKind`] its
//! rule for each modality: the graph classes the query may run on, whether it
//! is a graphless design, and the refusal an unlicensed class gets. The three
//! questions the study builder and the prepared handle ask are answered from
//! it, so none of them keeps its own list of query variants:
//!
//! - `licensed_graph_classes`: the classes a (query, modality) pair is
//!   licensed on (empty when the pair is not licensed at all);
//! - [`graphless_design`] / [`is_graphless_design`]: whether the query is a
//!   design-based study whose structure is a graphless randomized trial rather
//!   than a supplied graph, and what preparation does for it;
//! - [`prepared_rule`]: the full rule `Study::prepare` enforces, with its
//!   refusal.
//!
//! The match in [`kind_license`] has no wildcard and is keyed by
//! [`CausalQueryKind`], which `antecedent-core` keeps exhaustive, so a query
//! variant added there does not compile here until its row is written.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, CausalQueryKind};

use super::execute::DataModality;
use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::support::StructureSource;

/// What `Study::prepare` enforces for one (query kind, modality) pair.
#[derive(Clone, Copy)]
pub(crate) struct PreparedRule {
    classes: &'static [GraphClass],
    graphless: Option<GraphlessDesign>,
    refusal: fn() -> CausalError,
}

/// The preparation facts of a design-based query: it has no graph to inspect
/// and the structure must be the graphless randomized-trial source.
#[derive(Clone, Copy)]
pub(crate) struct GraphlessDesign {
    /// Whether prepare identifies the design once and caches it. The designs
    /// that identify per click retain no identification at prepare.
    pub(crate) identified_at_prepare: bool,
}

impl PreparedRule {
    /// Whether `class` under `structure` is licensed.
    pub(crate) fn admits(&self, class: GraphClass, structure: StructureSource) -> bool {
        self.classes.contains(&class)
            && (self.graphless.is_none() || structure == StructureSource::RandomizedTrial)
    }

    /// The refusal for an unlicensed class or structure.
    pub(crate) fn refusal(&self) -> CausalError {
        (self.refusal)()
    }

    /// The design facts, for a graphless design.
    pub(crate) fn graphless(&self) -> Option<GraphlessDesign> {
        self.graphless
    }
}

/// The rules of one query kind, one slot per modality.
#[derive(Clone, Copy)]
struct KindLicense {
    tabular: Option<PreparedRule>,
    /// Temporal or event series.
    series: Option<PreparedRule>,
    panel: Option<PreparedRule>,
    multi_env: Option<PreparedRule>,
}

const UNLICENSED: KindLicense =
    KindLicense { tabular: None, series: None, panel: None, multi_env: None };

const STATIC_SUPPLIED: &[GraphClass] = &[
    GraphClass::Dag,
    GraphClass::Cpdag,
    GraphClass::Pag,
    GraphClass::Admg,
    GraphClass::TemporalAdmg,
];
const TEMPORAL_ENVELOPES: &[GraphClass] =
    &[GraphClass::TemporalDag, GraphClass::TemporalCpdag, GraphClass::TemporalPag];
/// The classes a pulse, sustained or conditional contrast runs on over one series: the temporal
/// envelopes, and a temporal ADMG identified on its unrolled window.
const TEMPORAL_EFFECT_SERIES: &[GraphClass] = &[
    GraphClass::TemporalDag,
    GraphClass::TemporalCpdag,
    GraphClass::TemporalPag,
    GraphClass::TemporalAdmg,
];
const TEMPORAL_MEDIATION: &[GraphClass] = &[GraphClass::TemporalDag, GraphClass::TemporalCpdag];
const DAG: &[GraphClass] = &[GraphClass::Dag];
const TEMPORAL_DAG: &[GraphClass] = &[GraphClass::TemporalDag];
const ADMG_ONLY: &[GraphClass] = &[GraphClass::Admg, GraphClass::TemporalAdmg];
const STATIC_CONDITIONAL: &[GraphClass] =
    &[GraphClass::Dag, GraphClass::Cpdag, GraphClass::Pag, GraphClass::Admg];
const STATIC_DISTRIBUTION: &[GraphClass] =
    &[GraphClass::Dag, GraphClass::Admg, GraphClass::TemporalAdmg];
const RANDOMIZED: &[GraphClass] = &[GraphClass::RandomizedTrial];
/// The classes a static structure search answers with; CPDAG and PAG stay classes.
const STATIC_DISCOVERED: &[GraphClass] = &[GraphClass::Dag, GraphClass::Cpdag, GraphClass::Pag];
/// The classes a temporal structure search answers with.
const TEMPORAL_DISCOVERED: &[GraphClass] =
    &[GraphClass::TemporalDag, GraphClass::TemporalCpdag, GraphClass::TemporalPag];

fn discovery_refusal() -> CausalError {
    crate::unsupported_reason!(
        "data_modality_not_licensed",
        "a static structure search runs on tabular data and a temporal one on series, panel or \
         multi-environment data"
    )
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "the table slots are options; the helper names a licensed slot"
)]
const fn rule(
    classes: &'static [GraphClass],
    refusal: fn() -> CausalError,
) -> Option<PreparedRule> {
    Some(PreparedRule { classes, graphless: None, refusal })
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "the table slots are options; the helper names a licensed slot"
)]
const fn design(identified_at_prepare: bool, refusal: fn() -> CausalError) -> Option<PreparedRule> {
    Some(PreparedRule {
        classes: RANDOMIZED,
        graphless: Some(GraphlessDesign { identified_at_prepare }),
        refusal,
    })
}

/// The refusal for a (query, modality) pair the prepared handle does not
/// license at all.
pub(crate) fn unlicensed_combination() -> CausalError {
    crate::unsupported_reason!(
        "data_modality_not_licensed",
        "PreparedStudy supports AverageEffect, ResponseCurve, ConditionalEffect, \
         PathSpecific, Distribution, temporal ResponseCurve, TemporalEffect (Pulse / \
         single-step Sustained), TemporalMediationEffect, panel Pulse/Sustained, \
         Counterfactual, AnomalyAttribution, ChangeAttribution, TransportQuery, or \
         InterferenceQuery, or graphless randomized ITT"
    )
}

/// Every panel query that is not one of the licensed pairs is refused with the
/// panel wording, whatever its kind.
const PANEL_REFUSED: PreparedRule =
    PreparedRule { classes: &[], graphless: None, refusal: panel_refusal };

fn panel_refusal() -> CausalError {
    crate::unsupported_reason!(
        "data_modality_not_licensed",
        "PreparedStudy supports panel Pulse/Sustained and temporal response on \
         TemporalDag, TemporalCpdag, or TemporalPag"
    )
}

fn temporal_cross_world_refusal() -> CausalError {
    crate::unsupported_reason!(
        "data_modality_not_licensed",
        "temporal cross-world effects require a TemporalDag, TemporalCpdag, or TemporalPag"
    )
}

/// The one table. Exhaustive over [`CausalQueryKind`]: no wildcard arm.
#[allow(
    clippy::too_many_lines,
    reason = "the single exhaustive CausalQueryKind-to-KindLicense table the prepare-time license is read from: each arm owns its modality slots and refusal text, and the missing wildcard makes a new query kind a compile error here"
)]
fn kind_license(kind: CausalQueryKind) -> KindLicense {
    use CausalQueryKind as K;
    match kind {
        K::AverageEffect => KindLicense {
            tabular: rule(STATIC_SUPPLIED, || CausalError::Unsupported {
                message: "PreparedStudy requires a static Dag/Cpdag/Pag/Admg structure \
                          (temporal classes are not session-refreshable here)",
            }),
            ..UNLICENSED
        },
        K::TemporalEffect => KindLicense {
            series: rule(TEMPORAL_EFFECT_SERIES, || CausalError::Unsupported {
                message: "PreparedStudy supports TemporalEffect on TemporalDag, \
                          TemporalCpdag, TemporalPag, or TemporalAdmg (a contrast on a \
                          TemporalAdmg is identified on the unrolled window of its template \
                          and executes unlicensed)",
            }),
            panel: rule(TEMPORAL_ENVELOPES, panel_refusal),
            multi_env: rule(TEMPORAL_DAG, || {
                crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "PreparedStudy supports multi-environment TemporalEffect on a TemporalDag"
                )
            }),
            ..UNLICENSED
        },
        K::Counterfactual => KindLicense {
            tabular: rule(DAG, || CausalError::Unsupported {
                message: "counterfactual requires Dag",
            }),
            series: rule(TEMPORAL_ENVELOPES, temporal_cross_world_refusal),
            ..UNLICENSED
        },
        K::NestedCounterfactual => KindLicense {
            tabular: rule(DAG, || {
                crate::unsupported_reason!(
                    "cross_world_not_identified",
                    "nested effect requires a supplied DAG"
                )
            }),
            series: rule(TEMPORAL_ENVELOPES, temporal_cross_world_refusal),
            ..UNLICENSED
        },
        K::AnomalyAttribution | K::ChangeAttribution => KindLicense {
            tabular: rule(DAG, || CausalError::Unsupported {
                message: "AnomalyAttribution and ChangeAttribution require a supplied Dag",
            }),
            series: rule(TEMPORAL_DAG, || CausalError::Unsupported {
                message: "AnomalyAttribution and ChangeAttribution on a time series require a \
                          TemporalDag",
            }),
            ..UNLICENSED
        },
        // Off the support matrix and without a coverage license; prepared on a
        // supplied or accepted Dag so the retained compiler path executes them.
        // On a time series the null law is a block permutation of the residuals, and under
        // Bayesian inference the answer is the posterior probability of a change under the
        // prior the query declares.
        K::MechanismChange => KindLicense {
            tabular: rule(DAG, || CausalError::Unsupported {
                message: "MechanismChange requires a supplied or accepted Dag, or a supplied \
                          TemporalDag over a time series",
            }),
            series: rule(TEMPORAL_DAG, || CausalError::Unsupported {
                message: "MechanismChange on a time series requires a supplied TemporalDag",
            }),
            ..UNLICENSED
        },
        // A unit change also runs on the lag-aligned unfolding of a supplied TemporalDag (the
        // fitted-model point answer; its uncertainty is the Attribute unit-change target's).
        K::UnitChange => KindLicense {
            tabular: rule(DAG, || CausalError::Unsupported {
                message: "UnitChange requires a supplied or accepted Dag, or a supplied \
                          TemporalDag over a time series",
            }),
            series: rule(TEMPORAL_DAG, || CausalError::Unsupported {
                message: "UnitChange on a time series requires a supplied TemporalDag",
            }),
            ..UNLICENSED
        },
        K::Discover => KindLicense {
            tabular: rule(STATIC_DISCOVERED, discovery_refusal),
            series: rule(TEMPORAL_DISCOVERED, discovery_refusal),
            panel: rule(TEMPORAL_DISCOVERED, discovery_refusal),
            multi_env: rule(TEMPORAL_DISCOVERED, discovery_refusal),
        },
        // The fitted-model attribution also runs on the lag-aligned unfolding of a supplied
        // TemporalDag, with block weights or block counts for its refits.
        K::Attribute => KindLicense {
            tabular: rule(DAG, || CausalError::Unsupported {
                message: "Attribute fits mechanisms on a supplied or accepted Dag, or on the \
                          lag-aligned unfolding of a supplied TemporalDag",
            }),
            series: rule(TEMPORAL_DAG, || CausalError::Unsupported {
                message: "Attribute on a time series requires a supplied TemporalDag",
            }),
            ..UNLICENSED
        },
        // On a time series the rollout of a fitted TemporalDag steps the dynamics forward from the
        // last rows of the series.
        K::Simulate => KindLicense {
            tabular: rule(DAG, || CausalError::Unsupported {
                message: "Simulate fits mechanisms on a supplied or accepted Dag, or rolls out \
                          the dynamics of a supplied TemporalDag",
            }),
            series: rule(TEMPORAL_DAG, || CausalError::Unsupported {
                message: "Simulate on a time series is a temporal rollout on a supplied \
                          TemporalDag",
            }),
            ..UNLICENSED
        },
        K::Mediation => KindLicense {
            tabular: rule(DAG, || CausalError::Unsupported {
                message: "static mediation requires Dag",
            }),
            series: rule(TEMPORAL_MEDIATION, || CausalError::Unsupported {
                message: "PreparedStudy supports TemporalMediationEffect on TemporalDag or \
                          TemporalCpdag",
            }),
            ..UNLICENSED
        },
        K::ConditionalEffect => KindLicense {
            tabular: rule(STATIC_CONDITIONAL, || CausalError::Unsupported {
                message: "PreparedStudy supports ConditionalEffect on a supplied Dag, Cpdag, Pag, \
                          or Admg. A TemporalAdmg contrast is not identified; use \
                          TemporalConditionalPulse or TemporalConditionalSustained on a \
                          TemporalDag, or supply the static Admg for the contemporaneous \
                          conditional effect",
            }),
            ..UNLICENSED
        },
        K::Distribution => KindLicense {
            tabular: rule(STATIC_DISTRIBUTION, || CausalError::Unsupported {
                message: "PreparedStudy supports Distribution on a supplied Dag or Admg",
            }),
            series: rule(TEMPORAL_ENVELOPES, temporal_cross_world_refusal),
            ..UNLICENSED
        },
        K::PathSpecific => KindLicense {
            tabular: rule(DAG, || CausalError::Unsupported {
                message: "PreparedStudy supports PathSpecific only on a supplied Dag",
            }),
            series: rule(TEMPORAL_ENVELOPES, temporal_cross_world_refusal),
            ..UNLICENSED
        },
        K::Response => KindLicense {
            // A static response runs on tabular data and a temporal one on
            // series data; `prepared_rule` selects the slot by the query's own
            // attachment.
            tabular: rule(STATIC_SUPPLIED, || CausalError::Unsupported {
                message: "PreparedStudy supports ResponseCurve on a supplied Dag, Cpdag, Pag, \
                          or Admg (or CoDetermined joint cells)",
            }),
            series: rule(TEMPORAL_ENVELOPES, || CausalError::Unsupported {
                message: "PreparedStudy supports temporal ResponseCurve on TemporalDag, \
                          TemporalCpdag, or TemporalPag",
            }),
            panel: rule(TEMPORAL_ENVELOPES, panel_refusal),
            ..UNLICENSED
        },
        K::Transport => KindLicense {
            tabular: rule(ADMG_ONLY, || CausalError::Unsupported {
                message: "TransportQuery requires Admg or TemporalAdmg",
            }),
            ..UNLICENSED
        },
        K::Interference => KindLicense {
            tabular: rule(DAG, || CausalError::Unsupported {
                message: "InterferenceQuery requires Dag",
            }),
            ..UNLICENSED
        },
        K::RandomizedEffect => KindLicense {
            tabular: design(true, || {
                crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "PreparedStudy supports randomized ITT only on a graphless Bernoulli trial"
                )
            }),
            ..UNLICENSED
        },
        K::PolicyValue => KindLicense {
            tabular: design(false, || CausalError::Unsupported {
                message: "PolicyValue requires a graphless randomized trial",
            }),
            ..UNLICENSED
        },
        K::ContinuousDoseResponse => KindLicense {
            tabular: design(false, || CausalError::Unsupported {
                message: "continuous-dose response requires a graphless fixed baseline-group \
                          design",
            }),
            ..UNLICENSED
        },
        K::PanelDid => KindLicense {
            tabular: design(true, || CausalError::Unsupported {
                message: "PanelDid requires a graphless two-period panel design",
            }),
            ..UNLICENSED
        },
        K::SyntheticControl => KindLicense {
            tabular: design(true, || CausalError::Unsupported {
                message: "SyntheticControl requires a graphless balanced-panel design",
            }),
            ..UNLICENSED
        },
        K::LocalPolynomialRatio => KindLicense {
            tabular: design(true, || CausalError::Unsupported {
                message: "LocalPolynomialRatio requires a graphless fixed-cutoff design",
            }),
            ..UNLICENSED
        },
        K::Survival => KindLicense {
            tabular: design(false, || CausalError::Unsupported {
                message: "Survival requires a graphless randomized trial",
            }),
            ..UNLICENSED
        },
        K::LongitudinalRegime => KindLicense {
            tabular: design(false, || CausalError::Unsupported {
                message: "LongitudinalRegime requires a graphless sequential randomized trial",
            }),
            ..UNLICENSED
        },
        K::HistoryMatch => KindLicense {
            tabular: design(false, || CausalError::Unsupported {
                message: "HistoryMatch requires a graphless table of per-unit histories",
            }),
            ..UNLICENSED
        },
        K::SequentialContrast => KindLicense {
            tabular: design(false, || CausalError::Unsupported {
                message: "SequentialContrast requires a graphless two-decision panel with \
                          declared sequential exchangeability",
            }),
            ..UNLICENSED
        },
        // Off the support matrix and without a coverage license: the recorded sequential
        // randomization (or, for regime survival, declared sequential exchangeability)
        // identifies the target, so the study is a graphless design on a unit-period table.
        K::TemporalDesign => KindLicense {
            tabular: design(false, || CausalError::Unsupported {
                message: "TemporalDesign requires a graphless unit-period table with its recorded \
                          assignment design",
            }),
            ..UNLICENSED
        },
        // Off the support matrix and without a coverage license: a binary instrument that the
        // caller declares independent of the potential outcomes and excluded from the outcome
        // partially identifies the average effect, so the study is a graphless table of the
        // three coded columns and the answer is an identified set.
        K::IvBounds => KindLicense {
            tabular: design(false, || CausalError::Unsupported {
                message: "IvBounds requires a graphless table of a binary instrument, a binary \
                          treatment and a binary outcome",
            }),
            ..UNLICENSED
        },
        // Off the support matrix and without a coverage license: a noncausal association states
        // no intervention and no graph, so the study is a graphless table of the outcome and
        // its predictors, or one graphless series of them.
        K::Association => KindLicense {
            tabular: design(false, || CausalError::Unsupported {
                message: "Association requires a graphless table of its outcome and predictors",
            }),
            series: design(false, || CausalError::Unsupported {
                message: "Association requires a graphless series of its outcome and predictors",
            }),
            ..UNLICENSED
        },
    }
}

/// The rule `Study::prepare` enforces for `query` on `modality`, or `None` when
/// the pair is not licensed at all (refuse with [`unlicensed_combination`]).
pub(in crate::analysis) fn prepared_rule(
    query: &CausalQuery,
    modality: DataModality,
) -> Option<PreparedRule> {
    let license = kind_license(query.kind());
    match modality {
        DataModality::Tabular => match query {
            // A temporal response attached to tabular data has no static route.
            CausalQuery::Response(response) if response.is_temporal() => None,
            _ => license.tabular,
        },
        DataModality::TemporalOrEvent => match query {
            // A static response attached to series data has no temporal route.
            CausalQuery::Response(response) if !response.is_temporal() => None,
            _ => license.series,
        },
        DataModality::Panel => Some(license.panel.unwrap_or(PANEL_REFUSED)),
        DataModality::MultiEnv => license.multi_env,
    }
}

/// The graph classes `query` is licensed on over `modality`; empty when the
/// pair is not licensed.
#[cfg(test)]
pub(in crate::analysis) fn licensed_graph_classes(
    query: &CausalQuery,
    modality: DataModality,
) -> &'static [GraphClass] {
    prepared_rule(query, modality).map_or(&[], |rule| rule.classes)
}

/// The design facts of `query` when it is a design-based study: it has no
/// supplied graph, and its structure is the graphless randomized-trial source
/// built from the schema.
pub(crate) fn graphless_design(query: &CausalQuery) -> Option<GraphlessDesign> {
    kind_license(query.kind()).tabular.and_then(|rule| rule.graphless())
}

/// Whether `query` is a design-based study; see [`graphless_design`].
pub(crate) fn is_graphless_design(query: &CausalQuery) -> bool {
    graphless_design(query).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        AverageEffectQuery, RandomizedEffectQuery, TemporalEffectQuery, VariableId,
    };

    fn ate() -> CausalQuery {
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)).into()
    }

    fn randomized() -> CausalQuery {
        RandomizedEffectQuery::bernoulli_itt(
            VariableId::from_raw(1),
            vec![true, false],
            vec![0.5, 0.5],
            Vec::<std::sync::Arc<str>>::new(),
            Vec::<std::sync::Arc<str>>::new(),
            ("treated", "control"),
        )
        .into()
    }

    #[test]
    fn the_static_average_effect_runs_on_every_supplied_static_class_only() {
        assert_eq!(licensed_graph_classes(&ate(), DataModality::Tabular), STATIC_SUPPLIED);
        for temporal in TEMPORAL_ENVELOPES {
            assert!(!STATIC_SUPPLIED.contains(temporal));
        }
        assert!(licensed_graph_classes(&ate(), DataModality::TemporalOrEvent).is_empty());
        assert!(!is_graphless_design(&ate()));
    }

    #[test]
    fn a_design_is_graphless_and_licensed_on_the_randomized_trial_only() {
        let query = randomized();
        assert!(is_graphless_design(&query));
        assert_eq!(licensed_graph_classes(&query, DataModality::Tabular), RANDOMIZED);
        assert!(licensed_graph_classes(&query, DataModality::TemporalOrEvent).is_empty());
        let rule = prepared_rule(&query, DataModality::Tabular).unwrap();
        assert!(rule.admits(GraphClass::RandomizedTrial, StructureSource::RandomizedTrial));
        assert!(!rule.admits(GraphClass::RandomizedTrial, StructureSource::Explicit));
        assert!(!rule.admits(GraphClass::Dag, StructureSource::RandomizedTrial));
    }

    #[test]
    fn a_panel_refuses_every_query_kind_but_the_licensed_pairs_with_the_panel_wording() {
        let effect: CausalQuery =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .into();
        assert_eq!(licensed_graph_classes(&effect, DataModality::Panel), TEMPORAL_ENVELOPES);
        assert!(licensed_graph_classes(&ate(), DataModality::Panel).is_empty());
        let refused = prepared_rule(&ate(), DataModality::Panel).unwrap();
        assert_eq!(refused.refusal().to_string(), panel_refusal().to_string());
    }
}
