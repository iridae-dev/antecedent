//! Route admission, validation suites and strategy selection.

use super::{
    Arc, CausalError, CausalQuery, DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID,
    DEFAULT_CONDITIONAL_ESTIMATOR_ID, DEFAULT_CONDITIONAL_IDENTIFIER_ID,
    DEFAULT_DISTRIBUTION_ESTIMATOR_ID, DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID,
    DEFAULT_IDENTIFIER_ID, DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID,
    DEFAULT_PATH_ESTIMATOR_ID, DEFAULT_PATH_IDENTIFIER_ID, DEFAULT_RESPONSE_IDENTIFIER_ID,
    DataInput, EstimatorId, GraphClass, IdentifierId, RefuteSuite, ResponseQuery,
    reject_dag_only_on_pag, response_functional_is_derivative, wire_pair,
};

/// A route family that names its own validation suites.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ValidationFamily {
    PathSpecific,
    Distribution,
    TemporalMediation,
    MediationDecomposition,
}

impl ValidationFamily {
    /// The family whose suites validate a study of this query and graph class, if any.
    fn of(query: &CausalQuery, class: GraphClass) -> Option<Self> {
        match query {
            CausalQuery::PathSpecific(_) => Some(Self::PathSpecific),
            CausalQuery::Distribution(_) => Some(Self::Distribution),
            CausalQuery::Mediation(_) if class == GraphClass::TemporalDag => {
                Some(Self::TemporalMediation)
            }
            CausalQuery::Mediation(_)
                if matches!(
                    class,
                    GraphClass::Admg
                        | GraphClass::TemporalAdmg
                        | GraphClass::Cpdag
                        | GraphClass::Pag
                        | GraphClass::TemporalPag
                ) =>
            {
                Some(Self::MediationDecomposition)
            }
            _ => None,
        }
    }

    /// The id of the family's suite for `refute`; `None` when no validation runs.
    pub(super) fn suite_id(self, refute: RefuteSuite) -> Option<Arc<str>> {
        let (full, cheap) = match self {
            Self::PathSpecific => ("path.full", "path.cheap"),
            Self::Distribution => ("distribution.full", "distribution.cheap"),
            Self::TemporalMediation => ("temporal.mediation.full", "temporal.mediation.cheap"),
            Self::MediationDecomposition => {
                ("mediation.decomposition.full", "mediation.decomposition.cheap")
            }
        };
        match refute {
            RefuteSuite::None => None,
            RefuteSuite::Full => Some(Arc::from(full)),
            _ => Some(Arc::from(cheap)),
        }
    }
}

impl super::super::Study {
    /// The estimator the caller selected, or `None` for a builder placeholder.
    ///
    /// A caller selects an estimator through an estimator spec. An `estimator`
    /// without one is a placeholder the builder bound on its own
    /// (`inference()` binds `BayesianGcomp` on a static average effect,
    /// `query()` binds `TemporalLinearAdjustment` on a temporal one) and the
    /// compiled route keeps its own estimator instead. Every route that
    /// resolves or advertises an estimator decides "caller choice or
    /// placeholder" here.
    pub(crate) fn caller_selected_estimator(&self) -> Option<EstimatorId> {
        self.declared.estimator.filter(|_| self.declared.estimator_spec.is_some())
    }

    pub(in crate::analysis::execute) fn validation_suite_id(&self) -> Option<Arc<str>> {
        match ValidationFamily::of(&self.query, self.graph.class()) {
            Some(family) => family.suite_id(self.settings.refute),
            None => self.settings.refute.validation_suite_id().map(Arc::from),
        }
    }

    pub(in crate::analysis::execute) fn ensure_supported_combination(
        &self,
    ) -> Result<(), CausalError> {
        if let CausalQuery::Discover(query) = &self.query {
            // A structure search reads no supplied graph; its admission is the
            // method against the data modality.
            return self.ensure_discovery_admitted(query);
        }
        if let CausalQuery::Association(query) = &self.query {
            // An association reads no supplied graph; its admission is the declared
            // sampling design against the data modality.
            return self.ensure_association_admitted(query);
        }
        let class = self.graph.class();
        self.ensure_nested_counterfactual_supported(class)?;
        if self.declared.estimator == Some(EstimatorId::BayesianBasisGcomp)
            && (!matches!(self.data, DataInput::Tabular(_))
                || class != GraphClass::Dag
                || !matches!(
                    self.query,
                    CausalQuery::AverageEffect(_) | CausalQuery::ConditionalEffect(_)
                )
                || !self.settings.inference.is_bayesian())
        {
            return Err(CausalError::unsupported_inference(
                "bayesian.basis.gcomp requires Bayesian inference, tabular data, a supplied DAG, and an ATE or CATE query",
            ));
        }
        if let DataInput::Panel(panel) = &self.data {
            super::super::super::builder::refuse_unsupported_panel_route(
                &self.query,
                class,
                &self.settings.inference,
                panel,
                self.declared.split.as_ref(),
            )?;
        }
        self.ensure_data_query_graph_cell(class)?;
        self.ensure_temporal_effect_selection()
    }

    /// Nested-counterfactual admission: the temporal twin or the fixed three-node linear
    /// mediation DAG, and nothing else.
    pub(super) fn ensure_nested_counterfactual_supported(
        &self,
        class: GraphClass,
    ) -> Result<(), CausalError> {
        if let CausalQuery::NestedCounterfactual(q) = &self.query {
            q.validate()?;
            if self.graph_posterior.is_some() && matches!(self.data, DataInput::Tabular(_)) {
                // Each atom of a tabular graph posterior answers the natural direct effect on
                // its own structure; the placeholder graph names no edge to check.
            } else if matches!(
                self.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            ) && matches!(self.data, DataInput::Temporal(_) | DataInput::Event(_))
            {
                // The temporal twin is the linear direct slope on a TemporalDag.
            } else if matches!(
                (&self.data, class),
                (
                    DataInput::Tabular(_),
                    GraphClass::Admg
                        | GraphClass::TemporalAdmg
                        | GraphClass::Cpdag
                        | GraphClass::Pag
                )
            ) {
                // The natural direct effect on a mixed graph is identified by the
                // path-specific identifier over the ADMG, on a CPDAG in each DAG
                // completion, and on a PAG over the ADMGs each MAG completion stands for;
                // the identification decides whether the graph's structure allows it.
            } else {
                let Some(graph) = self.graph.as_dag() else {
                    return Err(crate::unsupported_reason!(
                        "cross_world_not_identified",
                        "natural direct effect requires the supported three-node DAG"
                    ));
                };
                let expected = [
                    (q.treatment.raw(), q.mediator.raw()),
                    (q.treatment.raw(), q.outcome.raw()),
                    (q.mediator.raw(), q.outcome.raw()),
                ];
                let observed = graph.directed_edge_pairs_sorted();
                let mut expected = expected.to_vec();
                expected.sort_unstable();
                if graph.node_count() != 3 || observed != expected {
                    return Err(crate::unsupported_reason!(
                        "cross_world_not_identified",
                        "query requires exactly X -> M, X -> Y, M -> Y with no other nodes or edges"
                    ));
                }
                if !matches!((&self.data, class), (DataInput::Tabular(_), GraphClass::Dag)) {
                    return Err(crate::unsupported_reason!(
                        "cross_world_not_identified",
                        "nested effect requires tabular data and the fixed DAG"
                    ));
                }
                if self
                    .declared
                    .estimator
                    .is_some_and(|id| id != EstimatorId::StaticMediationLinear)
                    || self
                        .declared
                        .identifier
                        .is_some_and(|id| id != IdentifierId::PathSpecificNatural)
                    || self.settings.bootstrap_replicates != 0
                {
                    return Err(crate::unsupported_reason!(
                        "cross_world_not_identified",
                        "only the point-only linear Gaussian mediation route is supported for this query"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Refusals for data / query / graph-class cells that no route covers, plus the
    /// PAG-only identifier check for tabular ATE.
    /// Response, distribution, and path-specific queries need their own data / graph-class
    /// pairings; the other cells are checked by [`Self::ensure_data_query_graph_cell`].
    pub(super) fn ensure_query_graph_class_cell(
        &self,
        class: GraphClass,
    ) -> Result<(), CausalError> {
        match (&self.data, &self.query, class) {
            (_, CausalQuery::Response(q), class)
                if !((!q.is_temporal()
                    && matches!(
                        (&self.data, class),
                        (
                            DataInput::Tabular(_),
                            GraphClass::Dag
                                | GraphClass::Cpdag
                                | GraphClass::Pag
                                | GraphClass::Admg
                                | GraphClass::TemporalAdmg
                        )
                    ))
                    || (q.is_temporal()
                        && matches!(
                            (&self.data, class),
                            (
                                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::Panel(_),
                                GraphClass::TemporalDag
                                    | GraphClass::TemporalCpdag
                                    | GraphClass::TemporalPag
                                    | GraphClass::TemporalAdmg
                            )
                        ))) =>
            {
                return Err(CausalError::Unsupported {
                    message: "static CausalQuery::Response requires tabular data and a Dag, \
                              Cpdag, Pag, or Admg (or a CoDetermined tier closure); temporal \
                              response requires series/event/panel data and a temporal graph",
                });
            }
            (_, CausalQuery::Distribution(_), class)
                if !(self.graph_posterior.is_some() && class == GraphClass::Pag
                    || matches!(
                        (&self.data, class),
                        (
                            DataInput::Tabular(_),
                            GraphClass::Dag
                                | GraphClass::Cpdag
                                | GraphClass::Pag
                                | GraphClass::Admg
                                | GraphClass::TemporalAdmg
                        ) | (
                            DataInput::Temporal(_) | DataInput::Event(_),
                            GraphClass::TemporalDag
                                | GraphClass::TemporalCpdag
                                | GraphClass::TemporalPag
                        )
                    )) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::Distribution requires tabular data and a static Dag, Cpdag, Pag or Admg",
                });
            }
            (_, CausalQuery::PathSpecific(_), class)
                if !matches!(
                    (&self.data, class),
                    (
                        DataInput::Tabular(_),
                        GraphClass::Dag
                            | GraphClass::Cpdag
                            | GraphClass::Pag
                            | GraphClass::Admg
                            | GraphClass::TemporalAdmg
                    ) | (
                        DataInput::Temporal(_) | DataInput::Event(_),
                        GraphClass::TemporalDag
                            | GraphClass::TemporalCpdag
                            | GraphClass::TemporalPag
                    )
                ) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::PathSpecific requires tabular data and a static Dag, Cpdag, Pag or Admg",
                });
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn ensure_data_query_graph_cell(
        &self,
        class: GraphClass,
    ) -> Result<(), CausalError> {
        self.ensure_query_graph_class_cell(class)?;
        match (&self.data, &self.query, class) {
            (DataInput::Tabular(_), CausalQuery::TemporalEffect(_), _) => {
                return Err(CausalError::Compile {
                    message: "temporal effect query requires temporal data".into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                CausalQuery::AverageEffect(_),
                _,
            ) => {
                return Err(CausalError::Compile {
                    message: "static ATE on temporal data is unsupported; use TemporalEffect"
                        .into(),
                });
            }
            (DataInput::Panel(_), CausalQuery::AverageEffect(_), _) => {
                return Err(CausalError::Compile {
                    message: "static ATE on panel data is unsupported; use TemporalEffect".into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                _,
                GraphClass::Pag | GraphClass::Cpdag | GraphClass::Admg,
            ) => {
                return Err(CausalError::Compile {
                    message: "static Pag/Cpdag/Admg requires tabular data and an average-effect \
                              or static response query"
                        .into(),
                });
            }
            (DataInput::MultiEnv(_), _, class) if class != GraphClass::TemporalDag => {
                return Err(CausalError::Compile {
                    message: "multi-environment data currently supports only a supplied \
                              TemporalDag"
                        .into(),
                });
            }
            (
                DataInput::Panel(_),
                CausalQuery::TemporalEffect(_) | CausalQuery::Response(_),
                GraphClass::TemporalCpdag | GraphClass::TemporalPag,
            ) => {}
            (DataInput::Panel(_), _, class) if class != GraphClass::TemporalDag => {
                return Err(CausalError::Compile {
                    message: "panel data supports only a supplied TemporalDag, TemporalCpdag, \
                              or TemporalPag with Pulse/Sustained or temporal response"
                        .into(),
                });
            }
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_), GraphClass::Pag) => {
                let (identifier, _) = self.resolve_pag_ids();
                reject_dag_only_on_pag(&self.graph, identifier)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// The temporal-effect path accepts only its linear / temporal identifier and
    /// estimator selections; any other explicit choice is refused here.
    pub(super) fn ensure_temporal_effect_selection(&self) -> Result<(), CausalError> {
        // The temporal path is linear/temporal-backdoor only; refuse an explicitly
        // selected non-temporal identifier/estimator rather than silently ignoring it.
        if matches!(&self.query, CausalQuery::TemporalEffect(_)) {
            let class_aware = self.graph.class().is_incomplete_temporal();
            // A temporal ADMG contrast is identified by the complete ID algorithm on the unrolled
            // window and evaluated by the functional plug-in; the builder's placeholder
            // estimator is not a selection.
            if self.graph.class() == GraphClass::TemporalAdmg {
                let identifier_ok =
                    self.declared.identifier.is_none_or(|id| id == IdentifierId::GeneralId);
                let estimator_ok = self
                    .caller_selected_estimator()
                    .is_none_or(|est| est == EstimatorId::FunctionalEffect);
                if !identifier_ok || !estimator_ok {
                    return Err(crate::compile_reason!(
                        "strategy_incompatible",
                        "a contrast on a TemporalAdmg is identified by general.id on its unrolled \
                         window and evaluated by functional.effect; got identifier {:?} and \
                         estimator {:?}",
                        self.declared.identifier,
                        self.caller_selected_estimator()
                    ));
                }
                return Ok(());
            }
            if let Some(id) = &self.declared.identifier {
                let ok = if class_aware {
                    *id == IdentifierId::GeneralizedAdjustment
                } else {
                    *id == IdentifierId::TemporalBackdoorUnfolded
                        || (*id == IdentifierId::TemporalIvUnfolded
                            && matches!(
                                self.declared.estimator,
                                Some(
                                    EstimatorId::TemporalLaggedInstrument
                                        | EstimatorId::BayesianTemporalLaggedInstrument
                                )
                            ))
                };
                if !ok {
                    return Err(CausalError::Compile {
                        message: format!(
                            "temporal path identifier {id:?} is not valid for this graph class"
                        ),
                    });
                }
            }
            if let Some(est) = &self.declared.estimator {
                let multi_step = matches!(
                    &self.query,
                    CausalQuery::TemporalEffect(q) if q.is_multi_step_sustained()
                );
                let bayesian = self.settings.inference.is_bayesian();
                let ok = *est == EstimatorId::TemporalLinearAdjustment
                    || matches!(
                        est,
                        EstimatorId::TemporalGlmLinkContrast
                            | EstimatorId::BayesianTemporalGlmLinkContrast
                            | EstimatorId::TemporalLaggedInstrument
                            | EstimatorId::BayesianTemporalLaggedInstrument
                    )
                    || (multi_step && *est == EstimatorId::TemporalSequentialGcomp)
                    || (bayesian && !multi_step && *est == EstimatorId::BayesianTemporalGcomp);
                if !ok {
                    return Err(CausalError::Compile {
                        message: format!(
                            "temporal path only supports estimator \"temporal.linear.adjustment\" \
                             (\"bayesian.temporal.gcomp\" under Bayesian inference, or \
                             \"temporal.sequential.gcomp\" for multi-step Sustained); got {est:?}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Resolve builder-selected identifier/estimator ids against modality defaults.
    pub(super) fn resolve_id_est_ids(
        &self,
        default_identifier: IdentifierId,
        default_estimator: EstimatorId,
    ) -> (IdentifierId, EstimatorId) {
        (
            self.declared.identifier.unwrap_or(default_identifier),
            self.declared.estimator.unwrap_or(default_estimator),
        )
    }

    /// Resolve builder-selected identifier/estimator ids, applying static-ATE defaults.
    pub(in crate::analysis::execute) fn resolve_static_ids(&self) -> (IdentifierId, EstimatorId) {
        let identifier = self.declared.identifier.unwrap_or(DEFAULT_IDENTIFIER_ID);
        let estimator = if let Some(native) = identifier.static_native_estimator() {
            self.declared
                .estimator
                .filter(|id| {
                    *id != EstimatorId::BayesianGcomp || self.declared.estimator_spec.is_some()
                })
                .unwrap_or(native)
        } else if self.settings.inference.is_bayesian() {
            self.declared.estimator.unwrap_or(EstimatorId::BayesianGcomp)
        } else {
            self.declared.estimator.unwrap_or(DEFAULT_ESTIMATOR_ID)
        };
        (identifier, estimator)
    }

    /// Resolve identifier/estimator for PAG ATE (generalized adjustment).
    pub(in crate::analysis::execute) fn resolve_pag_ids(&self) -> (IdentifierId, EstimatorId) {
        let estimator = if self.settings.inference.is_bayesian() {
            EstimatorId::BayesianGcomp
        } else {
            DEFAULT_PAG_ESTIMATOR_ID
        };
        self.resolve_id_est_ids(DEFAULT_PAG_IDENTIFIER_ID, estimator)
    }

    /// Incomplete TemporalCpdag/Pag pulse: envelope identifier + temporal linear estimator.
    pub(in crate::analysis::execute) fn resolve_temporal_class_ids(
        &self,
    ) -> (IdentifierId, EstimatorId) {
        self.resolve_id_est_ids(DEFAULT_PAG_IDENTIFIER_ID, EstimatorId::TemporalLinearAdjustment)
    }

    /// Resolve identifier/estimator for ADMG ATE (general ID + functional effect).
    ///
    /// `CoDetermined` tier-closure is generalized adjustment + AIPW, not `general.id`.
    /// A same-tier bidirected clique must not kick a supported AIPW study onto the
    /// The ADMG static route's identifier/estimator ids, applying defaults.
    pub(in crate::analysis::execute) fn resolve_admg_ids(&self) -> (IdentifierId, EstimatorId) {
        if self
            .declared
            .tiered
            .as_ref()
            .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::CoDetermined)
        {
            let default = if self.settings.inference.is_bayesian() {
                EstimatorId::BayesianGcomp
            } else {
                EstimatorId::Aipw
            };
            self.resolve_id_est_ids(IdentifierId::GeneralizedAdjustment, default)
        } else {
            let identifier = self.declared.identifier.unwrap_or(DEFAULT_ADMG_IDENTIFIER_ID);
            let estimator = self.caller_selected_estimator().unwrap_or(DEFAULT_ADMG_ESTIMATOR_ID);
            (identifier, estimator)
        }
    }

    /// Resolve identifier/estimator for `ConditionalEffect`.
    pub(in crate::analysis::execute) fn resolve_conditional_pair(&self) -> (Arc<str>, Arc<str>) {
        wire_pair(self.resolve_conditional_ids())
    }

    /// [`Self::resolve_conditional_pair`] as typed ids.
    pub(in crate::analysis::execute) fn resolve_conditional_ids(
        &self,
    ) -> (IdentifierId, EstimatorId) {
        let estimator = if self.settings.inference.is_bayesian() {
            EstimatorId::BayesianConditional
        } else {
            DEFAULT_CONDITIONAL_ESTIMATOR_ID
        };
        (
            self.declared.identifier.unwrap_or(DEFAULT_CONDITIONAL_IDENTIFIER_ID),
            self.caller_selected_estimator().unwrap_or(estimator),
        )
    }

    /// Resolve identifier/estimator for Distribution queries.
    pub(in crate::analysis::execute) fn resolve_distribution_ids(
        &self,
    ) -> (IdentifierId, EstimatorId) {
        let identifier = self.declared.identifier.unwrap_or(DEFAULT_DISTRIBUTION_IDENTIFIER_ID);
        let estimator = self
            .declared
            .estimator
            .filter(|id| {
                *id != EstimatorId::BayesianGcomp || self.declared.estimator_spec.is_some()
            })
            .unwrap_or(DEFAULT_DISTRIBUTION_ESTIMATOR_ID);
        (identifier, estimator)
    }

    /// Resolve the response estimator from the functional when the caller did not override it.
    pub(in crate::analysis::execute) fn resolve_response_ids(
        &self,
        query: &ResponseQuery,
    ) -> (IdentifierId, EstimatorId) {
        if self.settings.inference.is_bayesian()
            && !response_functional_is_derivative(&query.functional)
        {
            return (
                self.declared.identifier.unwrap_or(DEFAULT_RESPONSE_IDENTIFIER_ID),
                self.declared
                    .estimator_spec
                    .as_deref()
                    .map_or(EstimatorId::ResponseBayesian, crate::EstimatorSpec::id),
            );
        }
        self.resolve_id_est_ids(
            DEFAULT_RESPONSE_IDENTIFIER_ID,
            EstimatorId::default_for_response(&query.functional),
        )
    }

    /// [`Self::resolve_response_ids`] as wire names, as a plan binding carries them.
    pub(in crate::analysis::execute) fn resolve_response_pair(
        &self,
        query: &ResponseQuery,
    ) -> (Arc<str>, Arc<str>) {
        wire_pair(self.resolve_response_ids(query))
    }

    pub(in crate::analysis::execute) fn resolve_codetermined_joint_ids(
        &self,
    ) -> (IdentifierId, EstimatorId) {
        // The Bayesian joint level runs response.bayesian over the same closure set; a
        // builder placeholder estimator does not override it, a caller selection does.
        if self.settings.inference.is_bayesian() {
            return (
                self.declared.identifier.unwrap_or(IdentifierId::GeneralizedAdjustment),
                self.caller_selected_estimator().unwrap_or(EstimatorId::ResponseBayesian),
            );
        }
        self.resolve_id_est_ids(IdentifierId::GeneralizedAdjustment, EstimatorId::CellAipw)
    }

    /// Resolve identifier/estimator for Cpdag/Pag response (generalized adjustment).
    pub(in crate::analysis::execute) fn resolve_class_response_ids(
        &self,
        query: &ResponseQuery,
    ) -> (IdentifierId, EstimatorId) {
        if self.settings.inference.is_bayesian()
            && !response_functional_is_derivative(&query.functional)
        {
            return (
                self.declared.identifier.unwrap_or(DEFAULT_PAG_IDENTIFIER_ID),
                self.declared
                    .estimator_spec
                    .as_deref()
                    .map_or(EstimatorId::ResponseBayesian, crate::EstimatorSpec::id),
            );
        }
        self.resolve_id_est_ids(
            DEFAULT_PAG_IDENTIFIER_ID,
            EstimatorId::default_for_response(&query.functional),
        )
    }

    /// Resolve identifier/estimator for Cpdag/Pag `ConditionalEffect`.
    pub(in crate::analysis::execute) fn resolve_class_conditional_ids(
        &self,
    ) -> (IdentifierId, EstimatorId) {
        let estimator = if self.settings.inference.is_bayesian() {
            EstimatorId::BayesianConditional
        } else {
            EstimatorId::ConditionalLinearAdjustment
        };
        (
            self.declared.identifier.unwrap_or(DEFAULT_PAG_IDENTIFIER_ID),
            self.declared.estimator_spec.as_deref().map_or(estimator, crate::EstimatorSpec::id),
        )
    }

    /// Resolve identifier/estimator for `PathSpecific` queries.
    pub(in crate::analysis::execute) fn resolve_path_ids(&self) -> (IdentifierId, EstimatorId) {
        let identifier = self.declared.identifier.unwrap_or(DEFAULT_PATH_IDENTIFIER_ID);
        let estimator = self
            .declared
            .estimator
            .filter(|id| {
                *id != EstimatorId::BayesianGcomp || self.declared.estimator_spec.is_some()
            })
            .unwrap_or(DEFAULT_PATH_ESTIMATOR_ID);
        (identifier, estimator)
    }

    pub(in crate::analysis::execute) fn ensure_rd_config_present(
        &self,
        estimator: EstimatorId,
    ) -> Result<(), CausalError> {
        if matches!(estimator, EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear)
            && self.rd.is_none()
        {
            return Err(CausalError::Compile {
                message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
            });
        }
        Ok(())
    }
}
