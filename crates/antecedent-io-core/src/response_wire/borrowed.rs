//! Borrowed canonical response declarations for retained proof projections.
use super::*;
use crate::expr_wire::Sequence;
use crate::query_wire::borrowed::{BorrowedRetainedQueryWire, Functional, RawVariables, Temporal};
use serde::ser::{SerializeStruct, SerializeStructVariant};

pub(crate) struct BorrowedResponse<'a, 'b> {
    pub(crate) query: &'a ResponseQuery,
    pub(crate) context: &'a BorrowedRetainedQueryWire<'b>,
}
impl Serialize for BorrowedResponse<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let q = self.query;
        let include_functional = matches!(
            q.outcome_functional,
            antecedent_core::OutcomeFunctional::Exceedance(_)
                | antecedent_core::OutcomeFunctional::ExceedanceGrid(_)
                | antecedent_core::OutcomeFunctional::Quantile(_)
        );
        let mut wire = s.serialize_struct(
            "ResponseQueryWire",
            4 + usize::from(q.temporal.is_some()) + usize::from(include_functional),
        )?;
        wire.serialize_field(
            "functional",
            &ResponseFunction { functional: &q.functional, context: self.context },
        )?;
        wire.serialize_field("target_population", &self.context.population(&q.target_population))?;
        // Observation conversion is a fixed-size enum of native u32 coordinates;
        // unlike the complete response DTO it allocates no owned collections.
        wire.serialize_field("observation", &observation_to_wire(&q.observation))?;
        wire.serialize_field(
            "observation_assumptions",
            &Sequence(|| q.observation_assumptions.iter().map(ObservationAssumptionView)),
        )?;
        if let Some(temporal) = &q.temporal {
            wire.serialize_field("temporal", &TemporalResponse(temporal))?;
        }
        if include_functional {
            wire.serialize_field("outcome_functional", &Functional(&q.outcome_functional))?;
        }
        wire.end()
    }
}
struct ObservationAssumptionView<'a>(&'a ObservationAssumption);
impl Serialize for ObservationAssumptionView<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            ObservationAssumption::IndependentGiven(variables) => s.serialize_newtype_variant(
                "ObservationAssumptionWire",
                0,
                "independent_given",
                &RawVariables(variables),
            ),
            ObservationAssumption::OutcomeIndependentGiven(variables) => s
                .serialize_newtype_variant(
                    "ObservationAssumptionWire",
                    0,
                    "outcome_independent_given",
                    &RawVariables(variables),
                ),
            ObservationAssumption::Structural(name) => s.serialize_newtype_variant(
                "ObservationAssumptionWire",
                0,
                "structural",
                name.as_ref(),
            ),
        }
    }
}
struct TemporalResponse<'a>(&'a antecedent_core::TemporalResponseSpec);
impl Serialize for TemporalResponse<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let t = self.0;
        let mut wire = s.serialize_struct(
            "TemporalResponseSpecWire",
            2 + usize::from(t.max_history_lag.is_some()) + usize::from(t.feedback.is_some()),
        )?;
        wire.serialize_field("horizons", t.horizons.as_ref())?;
        wire.serialize_field("policy", &Temporal(&t.policy))?;
        if let Some(lag) = t.max_history_lag {
            wire.serialize_field("max_history_lag", &lag)?;
        }
        if let Some(feedback) = &t.feedback {
            wire.serialize_field("feedback", &Feedback(feedback))?;
        }
        wire.end()
    }
}
struct Feedback<'a>(&'a antecedent_core::FeedbackDeclaration);
impl Serialize for Feedback<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let f = self.0;
        let name = f.initial_law.name();
        let mut wire =
            s.serialize_struct("FeedbackDeclarationWire", 6 + usize::from(name.is_some()))?;
        wire.serialize_field("delay_ticks", &f.delay_ticks)?;
        wire.serialize_field("level_bits", &f.level.to_bits())?;
        wire.serialize_field("autoregression_bits", &f.autoregression.to_bits())?;
        wire.serialize_field("input_gain_bits", &f.input_gain.to_bits())?;
        wire.serialize_field("innovation_variance_bits", &f.innovation_variance.to_bits())?;
        if let Some(name) = name {
            wire.serialize_field("initial_law", name)?;
        }
        wire.serialize_field("initial_state_bits", &f.initial_state.to_bits())?;
        wire.end()
    }
}
struct Grid<'a>(&'a GridSpec);
impl Serialize for Grid<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            GridSpec::Values(values) => {
                s.serialize_newtype_variant("GridSpecWire", 0, "values", values.as_ref())
            }
            GridSpec::Linspace { start, end, points } => {
                let mut wire = s.serialize_struct_variant("GridSpecWire", 0, "linspace", 3)?;
                wire.serialize_field("start", start)?;
                wire.serialize_field("end", end)?;
                wire.serialize_field(
                    "points",
                    &u64::try_from(*points).map_err(serde::ser::Error::custom)?,
                )?;
                wire.end()
            }
        }
    }
}
struct Domain<'a>(&'a ContinuousDomain);
impl Serialize for Domain<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut wire = s.serialize_struct("ContinuousDomainWire", 2)?;
        wire.serialize_field("variable", &self.0.variable.raw())?;
        wire.serialize_field("grid", &Grid(&self.0.grid))?;
        wire.end()
    }
}
struct Weighting<'a>(&'a DerivativeWeighting);
impl Serialize for Weighting<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            DerivativeWeighting::Observed => {
                s.serialize_unit_variant("DerivativeWeightingWire", 0, "observed")
            }
            DerivativeWeighting::Uniform { lower, upper } => {
                let mut wire =
                    s.serialize_struct_variant("DerivativeWeightingWire", 0, "uniform", 2)?;
                wire.serialize_field("lower", lower)?;
                wire.serialize_field("upper", upper)?;
                wire.end()
            }
            DerivativeWeighting::Custom(weights) => s.serialize_newtype_variant(
                "DerivativeWeightingWire",
                0,
                "custom",
                weights.as_ref(),
            ),
        }
    }
}
struct ResponseFunction<'a, 'b> {
    functional: &'a ResponseFunctional,
    context: &'a BorrowedRetainedQueryWire<'b>,
}
impl Serialize for ResponseFunction<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        macro_rules! variant{($name:literal,$($field:literal=>$value:expr),* $(,)?)=>{{let mut wire=s.serialize_struct_variant("ResponseFunctionalWire",0,$name,0 $(+{let _=$field;1})*)?;$(wire.serialize_field($field,&$value)?;)*wire.end()}};}
        match self.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                variant!("mean_curve","outcome"=>outcome.raw(),"treatment"=>Domain(treatment))
            }
            ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
                variant!("average_derivative","outcome"=>outcome.raw(),"treatment"=>treatment.raw(),"weighting"=>Weighting(weighting))
            }
            ResponseFunctional::PointDerivative { outcome, treatment, at, order, scale } => {
                variant!("point_derivative","outcome"=>outcome.raw(),"treatment"=>treatment.raw(),"at"=>at,"order"=>order,"scale"=>scale_to_wire(*scale))
            }
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                variant!("directional_derivative","outcomes"=>RawVariables(outcomes),"treatments"=>RawVariables(treatments),"at"=>at.as_ref(),"direction"=>direction.as_ref())
            }
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                variant!("jacobian","outcomes"=>RawVariables(outcomes),"treatments"=>RawVariables(treatments),"at"=>at.as_ref(),"scale"=>scale_to_wire(*scale))
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                variant!("intervention_response","outcome"=>outcome.raw(),"interventions"=>Sequence(||interventions.iter().map(|iv|self.context.intervention(iv))))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CausalQuery, FeedbackDeclaration, InitialLaw, Intervention, TargetPopulation,
        TemporalPolicy, TemporalResponseSpec, Value,
    };
    fn compare(query: ResponseQuery, source: &CausalQuery) {
        let query = CausalQuery::Response(query);
        let original = crate::query_wire::causal_query_to_wire(&query).unwrap();
        let retained = crate::query_wire::causal_query_to_wire(source).unwrap();
        let borrowed = BorrowedRetainedQueryWire::new(&query, source, &retained);
        assert_eq!(serde_json::to_vec(&borrowed).unwrap(), serde_json::to_vec(&original).unwrap());
        let mut old = Vec::new();
        let mut new = Vec::new();
        ciborium::into_writer(&original, &mut old).unwrap();
        ciborium::into_writer(&borrowed, &mut new).unwrap();
        assert_eq!(new, old);
    }
    #[test]
    fn projected_responses_preserve_observation_assumptions_temporal_feedback_and_functionals() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let z = VariableId::from_raw(2);
        let source = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(t, GridSpec::Values(Arc::from([0.0, 1.0]))),
        }));
        let functions = [
            ResponseFunctional::MeanCurve {
                outcome: y,
                treatment: ContinuousDomain::new(
                    t,
                    GridSpec::Linspace { start: 0.0, end: 2.0, points: 3 },
                ),
            },
            ResponseFunctional::AverageDerivative {
                outcome: y,
                treatment: t,
                weighting: DerivativeWeighting::Custom(Arc::from([0.25, 0.75])),
            },
            ResponseFunctional::AverageDerivative {
                outcome: y,
                treatment: t,
                weighting: DerivativeWeighting::Observed,
            },
            ResponseFunctional::AverageDerivative {
                outcome: y,
                treatment: t,
                weighting: DerivativeWeighting::Uniform { lower: 0.0, upper: 2.0 },
            },
            ResponseFunctional::PointDerivative {
                outcome: y,
                treatment: t,
                at: 0.5,
                order: 2,
                scale: DerivativeScale::LogLog,
            },
            ResponseFunctional::DirectionalDerivative {
                outcomes: Arc::from([y, z]),
                treatments: Arc::from([t]),
                at: Arc::from([0.5]),
                direction: Arc::from([-1.0]),
            },
            ResponseFunctional::Jacobian {
                outcomes: Arc::from([y, z]),
                treatments: Arc::from([t]),
                at: Arc::from([0.5]),
                scale: DerivativeScale::LogOutcome,
            },
            ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from([Intervention::set(t, Value::f64(0.25))]),
            },
        ];
        for (index, functional) in functions.into_iter().enumerate() {
            let mut query = ResponseQuery::new(functional);
            query.target_population = TargetPopulation::Treated;
            query.observation = ObservationSpec::Selected { latent: y, observed: y, indicator: z };
            query.observation_assumptions = Arc::from([
                ObservationAssumption::IndependentGiven(Arc::from([t, z])),
                ObservationAssumption::OutcomeIndependentGiven(Arc::from([z])),
                ObservationAssumption::Structural(Arc::from("original\\\"λ")),
            ]);
            query.outcome_functional = match index % 4 {
                0 => antecedent_core::OutcomeFunctional::Mean,
                1 => antecedent_core::OutcomeFunctional::exceedance(0.25),
                2 => antecedent_core::OutcomeFunctional::exceedance_grid(vec![0.25, 0.75]),
                _ => antecedent_core::OutcomeFunctional::quantile(0.75),
            };
            let mut temporal =
                TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(-1), Some(3)).unwrap();
            temporal.feedback = Some(FeedbackDeclaration {
                delay_ticks: 1,
                level: 1.0,
                autoregression: 0.5,
                input_gain: 2.0,
                innovation_variance: 0.25,
                initial_law: InitialLaw::contemporaneous_observed(),
                initial_state: 0.0,
            });
            query.temporal = Some(temporal);
            compare(query, &source);
        }
    }
}
