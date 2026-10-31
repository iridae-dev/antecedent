//! Borrow retained canonical declarations; overlay only native projected coordinates.
use super::*;
use crate::expr_wire::Sequence;
use serde::ser::{SerializeSeq, SerializeStruct, SerializeStructVariant};

macro_rules! variant {
    ($s:ident,$name:literal,$($field:literal=>$value:expr),* $(,)?)=>{{
        let mut state=$s.serialize_struct_variant("CausalQueryWire",0,$name,0 $(+{let _=$field;1})*)?;
        $(state.serialize_field($field,&$value)?;)* state.end()
    }};
}

/// A canonical query already compiled for a retained producer, viewed at the
/// exact native proof query. Equal declarations borrow the original wire.
/// Coordinate projections use borrowed fields, not a copied query DTO.
pub struct BorrowedRetainedQueryWire<'a> {
    query: &'a CausalQuery,
    source: &'a CausalQuery,
    encoded: &'a CausalQueryWire,
}
impl<'a> BorrowedRetainedQueryWire<'a> {
    /// Bind an actual retained proof query to its original compiled declaration.
    #[must_use]
    pub const fn new(
        query: &'a CausalQuery,
        source: &'a CausalQuery,
        encoded: &'a CausalQueryWire,
    ) -> Self {
        Self { query, source, encoded }
    }
    pub(crate) fn population<'b>(&'b self, population: &'b TargetPopulation) -> Population<'b> {
        Population {
            population,
            source: self.source.target_population(),
            encoded: self.encoded.target_population(),
        }
    }
    pub(crate) fn intervention<'b>(
        &'b self,
        intervention: &'b Intervention,
    ) -> InterventionView<'b> {
        InterventionView { intervention, retained: self.matching_intervention(intervention) }
    }
    fn matching_intervention(&self, requested: &Intervention) -> Option<&InterventionWire> {
        match (self.source, self.encoded) {
            (
                CausalQuery::AverageEffect(q),
                CausalQueryWire::AverageEffect { control, active, .. },
            ) => {
                if requested == &q.control {
                    Some(control)
                } else if requested == &q.active {
                    Some(active)
                } else {
                    None
                }
            }
            (
                CausalQuery::TemporalEffect(q),
                CausalQueryWire::TemporalEffect { control, active, .. },
            ) => {
                if requested == &q.control {
                    Some(control)
                } else if requested == &q.active {
                    Some(active)
                } else {
                    None
                }
            }
            (CausalQuery::Mediation(q), CausalQueryWire::Mediation { control, active, .. }) => {
                if requested == &q.control {
                    Some(control)
                } else if requested == &q.active {
                    Some(active)
                } else {
                    None
                }
            }
            (CausalQuery::PathSpecific(q), CausalQueryWire::PathSpecific(wire)) => {
                if requested == &q.control {
                    Some(&wire.control)
                } else if requested == &q.active {
                    Some(&wire.active)
                } else {
                    None
                }
            }
            (CausalQuery::Distribution(q), CausalQueryWire::Distribution(wire)) => q
                .interventions
                .iter()
                .zip(&wire.interventions)
                .find(|(source, _)| *source == requested)
                .map(|(_, wire)| wire),
            (
                CausalQuery::Counterfactual(q),
                CausalQueryWire::Counterfactual { interventions, control, joint_control, .. },
            ) => {
                if requested == &q.control {
                    control.as_ref()
                } else {
                    q.interventions
                        .iter()
                        .zip(interventions)
                        .chain(q.joint_control.iter().zip(joint_control))
                        .find(|(source, _)| *source == requested)
                        .map(|(_, wire)| wire)
                }
            }
            (CausalQuery::Response(q), CausalQueryWire::Response(wire)) => {
                match (&q.functional, &wire.functional) {
                    (
                        antecedent_core::ResponseFunctional::InterventionResponse {
                            interventions,
                            ..
                        },
                        crate::response_wire::ResponseFunctionalWire::InterventionResponse {
                            interventions: encoded,
                            ..
                        },
                    ) => interventions
                        .iter()
                        .zip(encoded)
                        .find(|(source, _)| *source == requested)
                        .map(|(_, wire)| wire),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    fn average<S: serde::Serializer>(
        &self,
        q: &antecedent_core::AverageEffectQuery,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let include_functional = matches!(
            q.outcome_functional,
            OutcomeFunctional::Exceedance(_)
                | OutcomeFunctional::ExceedanceGrid(_)
                | OutcomeFunctional::Quantile(_)
        );
        let mut wire = s.serialize_struct_variant(
            "CausalQueryWire",
            0,
            "average_effect",
            6 + usize::from(include_functional),
        )?;
        wire.serialize_field("treatment", &q.treatment.raw())?;
        wire.serialize_field("outcome", &q.outcome.raw())?;
        wire.serialize_field("effect_modifiers", &RawVariables(&q.effect_modifiers))?;
        wire.serialize_field("control", &self.intervention(&q.control))?;
        wire.serialize_field("active", &self.intervention(&q.active))?;
        wire.serialize_field("target_population", &self.population(&q.target_population))?;
        if include_functional {
            wire.serialize_field("outcome_functional", &Functional(&q.outcome_functional))?;
        }
        wire.end()
    }
}
impl Serialize for BorrowedRetainedQueryWire<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.query == self.source {
            return self.encoded.serialize(s);
        }
        if let (
            CausalQuery::AverageEffect(query),
            CausalQuery::ConditionalEffect(source),
            CausalQueryWire::ConditionalEffect { inner },
        ) = (self.query, self.source, self.encoded)
        {
            if query == &source.inner {
                return inner.serialize(s);
            }
        }
        match self.query {
            CausalQuery::AverageEffect(q) => self.average(q, s),
            CausalQuery::TemporalEffect(q) => {
                let mut wire = s.serialize_struct_variant(
                    "CausalQueryWire",
                    1,
                    "temporal_effect",
                    8 + usize::from(!q.effect_modifiers.is_empty()),
                )?;
                wire.serialize_field("treatment", &q.treatment.raw())?;
                wire.serialize_field("outcome", &q.outcome.raw())?;
                wire.serialize_field("policy", &Temporal(&q.policy))?;
                wire.serialize_field("control", &self.intervention(&q.control))?;
                wire.serialize_field("active", &self.intervention(&q.active))?;
                wire.serialize_field("horizon_steps", &q.horizon_steps)?;
                wire.serialize_field("max_history_lag", &q.max_history_lag)?;
                wire.serialize_field("target_population", &self.population(&q.target_population))?;
                if !q.effect_modifiers.is_empty() {
                    wire.serialize_field("effect_modifiers", &RawVariables(&q.effect_modifiers))?;
                }
                wire.end()
            }
            CausalQuery::Mediation(q) => {
                let contrast: MediationContrastWire = q.contrast.into();
                variant!(s,"mediation","treatment"=>q.treatment.raw(),"outcome"=>q.outcome.raw(),
                    "mediators"=>RawVariables(&q.mediators),"contrast"=>contrast,"control"=>self.intervention(&q.control),
                    "active"=>self.intervention(&q.active),"target_population"=>self.population(&q.target_population),
                    "horizons"=>q.horizons.as_ref())
            }
            CausalQuery::NestedCounterfactual(q) => {
                variant!(s,"nested_counterfactual","treatment"=>q.treatment.raw(),
                "mediator"=>q.mediator.raw(),"outcome"=>q.outcome.raw(),"control_bits"=>q.control_value().to_bits(),
                "active_bits"=>q.active_value().to_bits())
            }
            CausalQuery::Counterfactual(q) => {
                let joint = !q.joint_control.is_empty();
                let mut wire = s.serialize_struct_variant(
                    "CausalQueryWire",
                    0,
                    "counterfactual",
                    4 + usize::from(joint),
                )?;
                wire.serialize_field("outcomes", &RawVariables(&q.outcomes))?;
                wire.serialize_field(
                    "interventions",
                    &Sequence(|| q.interventions.iter().map(|iv| self.intervention(iv))),
                )?;
                wire.serialize_field("control", &Some(self.intervention(&q.control)))?;
                if joint {
                    wire.serialize_field(
                        "joint_control",
                        &Sequence(|| q.joint_control.iter().map(|iv| self.intervention(iv))),
                    )?;
                }
                wire.serialize_field("allow_nested", &q.allow_nested)?;
                wire.end()
            }
            CausalQuery::ConditionalEffect(q) => {
                variant!(s,"conditional_effect","inner"=>AverageAsQuery{query:&q.inner,context:self})
            }
            CausalQuery::Distribution(q) => s.serialize_newtype_variant(
                "CausalQueryWire",
                0,
                "distribution",
                &Distribution { query: q, context: self },
            ),
            CausalQuery::PathSpecific(q) => s.serialize_newtype_variant(
                "CausalQueryWire",
                0,
                "path_specific",
                &PathSpecific { query: q, context: self },
            ),
            CausalQuery::Response(q) => s.serialize_newtype_variant(
                "CausalQueryWire",
                0,
                "response",
                &crate::response_wire::borrowed::BorrowedResponse { query: q, context: self },
            ),
            _ => Err(serde::ser::Error::custom(
                "retained proof query differs from its compiled producer and has no canonical coordinate projection",
            )),
        }
    }
}
struct AverageAsQuery<'a, 'b> {
    query: &'a antecedent_core::AverageEffectQuery,
    context: &'a BorrowedRetainedQueryWire<'b>,
}
impl Serialize for AverageAsQuery<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.context.average(self.query, s)
    }
}

pub(crate) struct RawVariables<'a>(pub(crate) &'a [VariableId]);
impl Serialize for RawVariables<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(self.0.iter().map(|v| v.raw()))
    }
}

pub(crate) struct Functional<'a>(pub(crate) &'a OutcomeFunctional);
impl Serialize for Functional<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            OutcomeFunctional::Exceedance(c) => s.serialize_newtype_variant(
                "OutcomeFunctionalWire",
                0,
                "exceedance",
                &super::canonical_zero(c.to_f64()),
            ),
            OutcomeFunctional::ExceedanceGrid(grid) => s.serialize_newtype_variant(
                "OutcomeFunctionalWire",
                0,
                "exceedance_grid",
                &Sequence(|| grid.iter().map(|v| super::canonical_zero(v.to_f64()))),
            ),
            OutcomeFunctional::Quantile(tau) => {
                s.serialize_newtype_variant("OutcomeFunctionalWire", 0, "quantile", &tau.to_f64())
            }
            _ => s.serialize_unit_variant("OutcomeFunctionalWire", 0, "mean"),
        }
    }
}

pub(crate) struct Population<'a> {
    population: &'a TargetPopulation,
    source: Option<&'a TargetPopulation>,
    encoded: Option<&'a TargetPopulationWire>,
}
impl Serialize for Population<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.source == Some(self.population) {
            if let Some(wire) = self.encoded {
                return wire.serialize(s);
            }
        }
        match self.population {
            TargetPopulation::AllObserved => {
                s.serialize_unit_variant("TargetPopulationWire", 0, "all_observed")
            }
            TargetPopulation::Treated => {
                s.serialize_unit_variant("TargetPopulationWire", 0, "treated")
            }
            TargetPopulation::Untreated => {
                s.serialize_unit_variant("TargetPopulationWire", 0, "untreated")
            }
            TargetPopulation::Environment(id) => {
                s.serialize_newtype_variant("TargetPopulationWire", 0, "environment", &id.raw())
            }
            TargetPopulation::Predicate(PredicateExpr::Rows(rows)) => s.serialize_newtype_variant(
                "TargetPopulationWire",
                0,
                "predicate_rows",
                &Rows(rows),
            ),
            TargetPopulation::RowWeights { weights, depends_on } => {
                let mut wire =
                    s.serialize_struct_variant("TargetPopulationWire", 0, "row_weights", 2)?;
                wire.serialize_field("weights", weights)?;
                wire.serialize_field("depends_on", &RawVariables(depends_on))?;
                wire.end()
            }
            TargetPopulation::LocalAtCutoff { running, cutoff } => {
                let mut wire =
                    s.serialize_struct_variant("TargetPopulationWire", 0, "local_at_cutoff", 2)?;
                wire.serialize_field("running", &running.raw())?;
                wire.serialize_field("cutoff", &cutoff.to_f64())?;
                wire.end()
            }
            _ => Err(serde::ser::Error::custom(
                "projected proof population differs from its compiled registry declaration",
            )),
        }
    }
}
struct Rows<'a>(&'a [usize]);
impl Serialize for Rows<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for &row in self.0 {
            seq.serialize_element(&u64::try_from(row).map_err(serde::ser::Error::custom)?)?;
        }
        seq.end()
    }
}

pub(crate) struct Temporal<'a>(pub(crate) &'a TemporalPolicy);
impl Serialize for Temporal<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            TemporalPolicy::Pulse { at } => {
                let mut wire = s.serialize_struct_variant("TemporalPolicyWire", 0, "pulse", 1)?;
                wire.serialize_field("at", at)?;
                wire.end()
            }
            TemporalPolicy::Sustained { from, until } => {
                let mut wire =
                    s.serialize_struct_variant("TemporalPolicyWire", 0, "sustained", 2)?;
                wire.serialize_field("from", from)?;
                wire.serialize_field("until", until)?;
                wire.end()
            }
            TemporalPolicy::Dynamic { rule, active_at } => {
                let mut wire = s.serialize_struct_variant("TemporalPolicyWire", 0, "dynamic", 2)?;
                wire.serialize_field("rule", &rule.raw())?;
                wire.serialize_field("active_at", active_at.as_ref())?;
                wire.end()
            }
            _ => Err(serde::ser::Error::custom("unsupported TemporalPolicy")),
        }
    }
}

pub(crate) struct InterventionView<'a> {
    intervention: &'a Intervention,
    retained: Option<&'a InterventionWire>,
}
impl Serialize for InterventionView<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if let Some(wire) = self.retained {
            return wire.serialize(s);
        }
        match self.intervention {
            Intervention::Set { variable, value } => {
                let mut wire = s.serialize_struct_variant("InterventionWire", 0, "set", 2)?;
                wire.serialize_field("variable", &variable.raw())?;
                wire.serialize_field("value", &Scalar(value))?;
                wire.end()
            }
            Intervention::Shift { variable, delta } => {
                let mut wire = s.serialize_struct_variant("InterventionWire", 0, "shift", 2)?;
                wire.serialize_field("variable", &variable.raw())?;
                wire.serialize_field("delta", &Scalar(delta))?;
                wire.end()
            }
            Intervention::Stochastic { variable, policy } => {
                let mut wire =
                    s.serialize_struct_variant("InterventionWire", 0, "stochastic", 2)?;
                wire.serialize_field("variable", &variable.raw())?;
                wire.serialize_field("policy", &Policy(policy))?;
                wire.end()
            }
            Intervention::Soft { variable, mechanism } => {
                let mut wire = s.serialize_struct_variant("InterventionWire", 0, "soft", 2)?;
                wire.serialize_field("variable", &variable.raw())?;
                wire.serialize_field("mechanism", &Mechanism(mechanism))?;
                wire.end()
            }
            _ => Err(serde::ser::Error::custom(
                "projected intervention differs from its compiled expression declaration",
            )),
        }
    }
}
struct Scalar<'a>(&'a Value);
impl Serialize for Scalar<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Value::Float64(v) => s.serialize_newtype_variant("ValueWire", 0, "float64", v),
            Value::Int64(v) => s.serialize_newtype_variant("ValueWire", 0, "int64", v),
            Value::Bool(v) => s.serialize_newtype_variant("ValueWire", 0, "bool", v),
            Value::Category(v) => s.serialize_newtype_variant("ValueWire", 0, "category", v),
            Value::Label(v) => s.serialize_newtype_variant("ValueWire", 0, "label", v.as_ref()),
        }
    }
}
struct Policy<'a>(&'a StochasticPolicy);
impl Serialize for Policy<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            StochasticPolicy::Bernoulli { p } => {
                let mut wire =
                    s.serialize_struct_variant("StochasticPolicyWire", 0, "bernoulli", 1)?;
                wire.serialize_field("p", p)?;
                wire.end()
            }
            StochasticPolicy::Gaussian { mean, variance } => {
                let mut wire =
                    s.serialize_struct_variant("StochasticPolicyWire", 0, "gaussian", 2)?;
                wire.serialize_field("mean", mean)?;
                wire.serialize_field("variance", variance)?;
                wire.end()
            }
            StochasticPolicy::Categorical { probs } => {
                let mut wire =
                    s.serialize_struct_variant("StochasticPolicyWire", 0, "categorical", 1)?;
                wire.serialize_field("probs", probs.as_ref())?;
                wire.end()
            }
            _ => Err(serde::ser::Error::custom("unsupported StochasticPolicy")),
        }
    }
}
struct Mechanism<'a>(&'a antecedent_core::MechanismOverride);
impl Serialize for Mechanism<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut wire = s.serialize_struct("MechanismOverrideWire", 2)?;
        wire.serialize_field("family_id", self.0.family_id.as_ref())?;
        wire.serialize_field("parameters", self.0.parameters.as_ref())?;
        wire.end()
    }
}

struct Distribution<'a, 'b> {
    query: &'a InterventionalDistributionQuery,
    context: &'a BorrowedRetainedQueryWire<'b>,
}
impl Serialize for Distribution<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let q = self.query;
        let mut wire = s.serialize_struct("InterventionalDistributionQueryWire", 4)?;
        wire.serialize_field("outcomes", &RawVariables(&q.outcomes))?;
        wire.serialize_field(
            "interventions",
            &Sequence(|| q.interventions.iter().map(|iv| self.context.intervention(iv))),
        )?;
        wire.serialize_field("conditioning", &RawVariables(&q.conditioning))?;
        wire.serialize_field("target_population", &self.context.population(&q.target_population))?;
        wire.end()
    }
}
struct PathSpecific<'a, 'b> {
    query: &'a PathSpecificEffectQuery,
    context: &'a BorrowedRetainedQueryWire<'b>,
}
impl Serialize for PathSpecific<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let q = self.query;
        let mut wire = s.serialize_struct("PathSpecificEffectQueryWire", 8)?;
        wire.serialize_field("treatment", &q.treatment.raw())?;
        wire.serialize_field("outcome", &q.outcome.raw())?;
        wire.serialize_field("path_nodes", &RawVariables(&q.path_nodes))?;
        wire.serialize_field("control", &self.context.intervention(&q.control))?;
        wire.serialize_field("active", &self.context.intervention(&q.active))?;
        wire.serialize_field("target_population", &self.context.population(&q.target_population))?;
        wire.serialize_field(
            "max_paths",
            &u64::try_from(q.max_paths).map_err(serde::ser::Error::custom)?,
        )?;
        wire.serialize_field(
            "max_len",
            &u64::try_from(q.max_len).map_err(serde::ser::Error::custom)?,
        )?;
        wire.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn compare(query: &CausalQuery, source: &CausalQuery) {
        let original = causal_query_to_wire(query).unwrap();
        let source_wire = causal_query_to_wire(source).unwrap();
        let borrowed = BorrowedRetainedQueryWire::new(query, source, &source_wire);
        assert_eq!(serde_json::to_vec(&borrowed).unwrap(), serde_json::to_vec(&original).unwrap());
        let mut old = Vec::new();
        let mut new = Vec::new();
        ciborium::into_writer(&original, &mut old).unwrap();
        ciborium::into_writer(&borrowed, &mut new).unwrap();
        assert_eq!(new, old);
    }
    #[test]
    fn retained_and_unfolded_queries_preserve_canonical_bytes_without_query_copies() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let source =
            CausalQuery::TemporalEffect(antecedent_core::TemporalEffectQuery::pulse(t, y, 1.0));
        compare(&source, &source);
        let mut projected = antecedent_core::AverageEffectQuery::binary_ate(
            VariableId::from_raw(2),
            VariableId::from_raw(3),
        );
        projected.outcome_functional = OutcomeFunctional::exceedance_grid(vec![0.25, 0.5, 1.0]);
        compare(&CausalQuery::AverageEffect(projected), &source);
        let source =
            antecedent_core::ResponseQuery::new(antecedent_core::ResponseFunctional::MeanCurve {
                outcome: y,
                treatment: antecedent_core::ContinuousDomain::new(
                    t,
                    antecedent_core::GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            });
        let mut projected = source.clone();
        projected.functional = antecedent_core::ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::set(t, Value::Float64(0.0))]),
        };
        compare(&CausalQuery::Response(projected), &CausalQuery::Response(source));
    }
    #[test]
    fn projected_default_omissions_and_all_coordinate_query_variants_match_canonical_codec() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let m = VariableId::from_raw(2);
        let source =
            CausalQuery::TemporalEffect(antecedent_core::TemporalEffectQuery::pulse(t, y, 1.0));
        let ate = AverageEffectQuery::binary_ate(t, y);
        compare(&CausalQuery::AverageEffect(ate.clone()), &source);
        compare(
            &CausalQuery::ConditionalEffect(
                ConditionalEffectQuery::try_new(ate.with_effect_modifiers([m])).unwrap(),
            ),
            &source,
        );
        compare(
            &CausalQuery::TemporalEffect(antecedent_core::TemporalEffectQuery::pulse(m, y, 2.0)),
            &source,
        );
        compare(
            &CausalQuery::TemporalEffect(
                antecedent_core::TemporalEffectQuery::pulse(m, y, 2.0).with_effect_modifiers([t]),
            ),
            &source,
        );
        compare(
            &CausalQuery::Mediation(MediationQuery::binary(
                t,
                y,
                [m],
                MediationContrast::NaturalIndirect,
            )),
            &source,
        );
        compare(
            &CausalQuery::NestedCounterfactual(
                antecedent_core::NestedCounterfactualQuery::with_levels(t, m, y, -0.5, 1.5)
                    .unwrap(),
            ),
            &source,
        );
        compare(
            &CausalQuery::Counterfactual(
                CounterfactualQuery::new(y, [Intervention::set(t, Value::f64(2.0))])
                    .with_nested(true),
            ),
            &source,
        );
        compare(
            &CausalQuery::Distribution(
                InterventionalDistributionQuery::new(y, [Intervention::set(t, Value::f64(2.0))])
                    .with_conditioning([m]),
            ),
            &source,
        );
        compare(
            &CausalQuery::PathSpecific(
                PathSpecificEffectQuery::binary(t, y)
                    .with_path_nodes([m])
                    .with_max_paths(8)
                    .with_max_len(4),
            ),
            &source,
        );
        for population in [
            TargetPopulation::AllObserved,
            TargetPopulation::Treated,
            TargetPopulation::Untreated,
            TargetPopulation::Environment(EnvironmentId::from_raw(3)),
            TargetPopulation::Predicate(PredicateExpr::rows([2usize, 0, 3])),
            TargetPopulation::local_at_cutoff(m, -1.25),
        ] {
            compare(
                &CausalQuery::AverageEffect(
                    AverageEffectQuery::binary_ate(t, y).with_target_population(population),
                ),
                &source,
            );
        }
    }
    #[test]
    fn projected_proofs_keep_registry_bound_population_digests_and_original_policy_declarations() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let mut registry = PopulationRegistry::new();
        registry.insert_predicate("cohort", vec![2usize, 0]);
        registry.insert_distribution_with_dependence(
            DistributionRef::from_raw(11),
            vec![0.25, 0.75],
            vec![y],
        );
        for population in [
            TargetPopulation::Predicate(PredicateExpr::named("cohort")),
            TargetPopulation::CustomDistribution(DistributionRef::from_raw(11)),
        ] {
            let source = CausalQuery::TemporalEffect(
                antecedent_core::TemporalEffectQuery::pulse(t, y, 1.0)
                    .with_target_population(population.clone()),
            );
            let projected = CausalQuery::AverageEffect(
                AverageEffectQuery::binary_ate(VariableId::from_raw(4), VariableId::from_raw(5))
                    .with_target_population(population),
            );
            let source_wire = causal_query_to_wire_with_registry(&source, Some(&registry)).unwrap();
            let original = causal_query_to_wire_with_registry(&projected, Some(&registry)).unwrap();
            let borrowed = BorrowedRetainedQueryWire::new(&projected, &source, &source_wire);
            assert_eq!(
                serde_json::to_vec(&borrowed).unwrap(),
                serde_json::to_vec(&original).unwrap()
            );
            let mut old = Vec::new();
            let mut new = Vec::new();
            ciborium::into_writer(&original, &mut old).unwrap();
            ciborium::into_writer(&borrowed, &mut new).unwrap();
            assert_eq!(new, old);
        }
        let sequence = Intervention::sequence(InterventionSequence::new(vec![
            SequencedIntervention::new(
                Intervention::set(t, Value::Label(Arc::from("original"))),
                TemporalPolicy::pulse(0),
            ),
            SequencedIntervention::new(
                Intervention::shift(t, Value::f64(0.5)),
                TemporalPolicy::sustained(1, 3),
            ),
        ]));
        let source = CausalQuery::Distribution(InterventionalDistributionQuery::new(y, [sequence]));
        let CausalQuery::Distribution(mut projected) = source.clone() else { unreachable!() };
        projected.conditioning = Arc::from([t]);
        compare(&CausalQuery::Distribution(projected), &source);
    }
}
