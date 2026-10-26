//! Data-free tiered identification, retaining the original lossless scientific certificate.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchema, CausalSchemaBuilder, Intervention,
    OutcomeFunctional, ResponseFunctional, ResponseQuery, TargetPopulation, Value,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

/// Parsed Python declarations. Scientific query encoding remains owned by antecedent-io.
enum TieredQuery {
    Average {
        treatment: String,
        outcome: String,
        control: Value,
        active: Value,
        population: TargetPopulation,
        functional: OutcomeFunctional,
    },
    Joint {
        outcome: String,
        interventions: Vec<(String, f64)>,
        population: TargetPopulation,
        functional: OutcomeFunctional,
        assumptions: Vec<crate::observation_api::ObservationArgs>,
    },
}

fn required<'py>(spec: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    spec.get_item(name)?.ok_or_else(|| crate::value_err(format!("tiered query requires {name}")))
}

fn contrast_value(spec: &Bound<'_, PyDict>, name: &str) -> PyResult<Value> {
    let value = required(spec, name)?;
    if let Ok(label) = value.extract::<String>() {
        Ok(Value::Label(Arc::from(label)))
    } else {
        Ok(Value::f64(value.extract()?))
    }
}

fn declared_assumptions(
    spec: &Bound<'_, PyDict>,
) -> PyResult<Vec<crate::observation_api::ObservationArgs>> {
    let Some(declarations) = spec.get_item("observation_assumptions")? else {
        return Ok(Vec::new());
    };
    declarations
        .extract::<Vec<Bound<'_, PyDict>>>()?
        .iter()
        .map(|declaration| {
            Ok(crate::observation_api::ObservationArgs {
                kind: "complete".into(),
                latent: String::new(),
                observed: None,
                censoring: None,
                event: None,
                lower: None,
                upper: None,
                indicator: None,
                assumption_kind: required(declaration, "assumption_kind")?.extract()?,
                assumption_variables: declaration
                    .get_item("assumption_variables")?
                    .map(|value| value.extract())
                    .transpose()?
                    .unwrap_or_default(),
                structural_model: declaration
                    .get_item("structural_model")?
                    .map(|value| value.extract())
                    .transpose()?,
            })
        })
        .collect()
}

impl TieredQuery {
    fn parse(spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        let population_value = spec.get_item("target_population")?;
        let population_spec = population_value
            .as_ref()
            .filter(|value| !value.is_none())
            .map(|value| value.cast::<PyDict>())
            .transpose()?;
        let population = crate::ate_api::parse_target_population(population_spec)?
            .unwrap_or(TargetPopulation::AllObserved);
        let functional_value = spec.get_item("outcome_functional")?;
        let functional_spec = functional_value
            .as_ref()
            .filter(|value| !value.is_none())
            .map(|value| value.cast::<PyDict>())
            .transpose()?;
        let functional = crate::ate_api::parse_outcome_functional(functional_spec)?
            .unwrap_or(OutcomeFunctional::Mean);
        let outcome = required(spec, "outcome")?.extract()?;
        match required(spec, "kind")?.extract::<String>()?.as_str() {
            "average" => Ok(Self::Average {
                treatment: required(spec, "treatment")?.extract()?,
                outcome,
                control: contrast_value(spec, "control_level")?,
                active: contrast_value(spec, "active_level")?,
                population,
                functional,
            }),
            "joint" => Ok(Self::Joint {
                outcome,
                interventions: required(spec, "interventions")?.extract()?,
                population,
                functional,
                assumptions: declared_assumptions(spec)?,
            }),
            _ => {
                Err(crate::value_err("tiered identify supports average or joint response queries"))
            }
        }
    }

    fn into_query(self, schema: &CausalSchema) -> PyResult<CausalQuery> {
        let variable = |name: &str| crate::schema_var_id(schema, name);
        match self {
            Self::Average { treatment, outcome, control, active, population, functional } => {
                let treatment = variable(&treatment)?;
                let mut query = AverageEffectQuery::binary_ate(treatment, variable(&outcome)?);
                query.control = Intervention::set(treatment, control);
                query.active = Intervention::set(treatment, active);
                query.target_population = population;
                query.outcome_functional = functional;
                query.validate().map_err(antecedent::CausalError::from).map_err(crate::py_err)?;
                Ok(CausalQuery::AverageEffect(query))
            }
            Self::Joint { outcome, interventions, population, functional, assumptions } => {
                let interventions = interventions
                    .into_iter()
                    .map(|(name, level)| {
                        variable(&name)
                            .map(|variable| Intervention::set(variable, Value::f64(level)))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let mut query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                    outcome: variable(&outcome)?,
                    interventions: interventions.into(),
                });
                query.target_population = population;
                query.outcome_functional = functional;
                query.observation_assumptions = assumptions
                    .iter()
                    .map(|args| crate::observation_api::build_assumption(schema, args))
                    .collect::<PyResult<Vec<_>>>()?
                    .into();
                query.validate().map_err(antecedent::CausalError::from).map_err(crate::py_err)?;
                Ok(CausalQuery::Response(query))
            }
        }
    }
}

/// Identify tier declarations without loading data, preparing an estimator, or executing it.
#[pyfunction(name = "_identify_tiered_structure")]
fn identify_tiered_structure(
    py: Python<'_>,
    names: Vec<String>,
    tiers: Vec<Vec<String>>,
    within_tier: String,
    query: &Bound<'_, PyDict>,
) -> PyResult<String> {
    let query = TieredQuery::parse(query)?;
    let within = crate::parse_within_tier(Some(&within_tier))?;
    crate::detach_catch(py, move || {
        let mut builder = CausalSchemaBuilder::new();
        for name in &names {
            builder = builder.continuous(name.as_str()).finish();
        }
        let schema = builder.build().map_err(crate::py_err)?;
        let named: Vec<Vec<&str>> =
            tiers.iter().map(|tier| tier.iter().map(String::as_str).collect()).collect();
        let background = antecedent_graph::TieredBackground::from_named(&schema, &named, within)
            .map_err(crate::py_err)?;
        let query = query.into_query(&schema)?;
        let result = match &query {
            CausalQuery::AverageEffect(query) => {
                antecedent_identify::identify_tiered(&background, query)
            }
            CausalQuery::Response(query) => {
                antecedent_identify::identify_tiered_joint(&background, &schema, query)
            }
            _ => return Err(crate::value_err("unsupported tiered query")),
        }
        .map_err(|error| {
            // Unknown within-tier order denotes two scenarios, not one joint
            // ADMG. Preserve that scientific license refusal as a typed public
            // boundary while leaving data/search failures on their own paths.
            if background.within_tier == antecedent_graph::WithinTier::Unknown
                && matches!(&query, CausalQuery::Response(_))
                && matches!(
                    &error,
                    antecedent_identify::IdentificationError::UnsupportedQuery { .. }
                )
            {
                crate::refusal(
                    antecedent_core::reason_code!("cell_not_licensed"),
                    error.to_string(),
                )
            } else {
                crate::py_err(antecedent::CausalError::from(error))
            }
        })?;
        crate::identification_details::tiered_to_json(
            &result,
            &query,
            &names,
            &background,
            &within_tier,
        )
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(identify_tiered_structure, module)?)
}
