//! Python bridge for composing independently fitted mechanisms.
//!
//! Python declares the mechanisms in semantic coordinates, their numbers and
//! the dependence between their estimates; the checks, the executable joint
//! model, the closed-form law, the structural sampling, the delta-method
//! variance, the artifact and every refusal are Rust's. A refusal comes back
//! as structured JSON (every defect found, in checker order) for the Python
//! layer to raise as a typed exception.
//!
//! A mechanism the analyst wrote down is labelled declared by the facade from
//! the source of its numbers; this bridge cannot label it fitted.

use antecedent::analysis::mechanism_composition::CompositionQueryWire;
use antecedent::{
    ConsumedComposition, DependenceInput, MechanismCompositionRefusal, MechanismCompositionRequest,
    MechanismInput, MechanismRefit, MechanismSource, PreparedMechanismComposition, StudyBuilder,
    consume_mechanism_composition_artifact,
};
use antecedent_contract::{
    AdmissionBasis, AdmissionCoordinate, ExternalRefusal, LatentCauseDeclaration,
    MechanismDeclaration, MechanismOriginKind, ScientificQuantity, TransportAdmission,
};
use antecedent_io::mechanism_composition_artifact::CompositionResultWire;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_model::compose::{
    ContrastArm, CrossCovarianceBlock, Domain, MeanContrast, MechanismFamily, MechanismUncertainty,
    ParameterId, ParameterTerm, TabularRow, UncertaintyMeaning,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Map, Value as Json, json};

use crate::measurement_model_api::{
    flag, list, member, object, parse, present, real, reals, text, texts,
};
use crate::transport_common::execution_context;

// ---------------------------------------------------------------------------
// Refusals as data: every defect, in checker order.
// ---------------------------------------------------------------------------

fn one_refusal(refusal: &ExternalRefusal) -> Json {
    let mut message = String::new();
    if let Some(expected) = &refusal.expected {
        message.push_str(&format!("expected `{expected}`"));
    }
    if let Some(supplied) = &refusal.supplied {
        if !message.is_empty() {
            message.push_str("; ");
        }
        message.push_str(&format!("supplied `{supplied}`"));
    }
    json!({
        "code": refusal.code,
        "stage": refusal.stage,
        "detail": refusal.detail,
        "message": message,
        "offending": refusal.offending,
        "expected": refusal.expected,
        "supplied": refusal.supplied,
        "remedy": refusal.remedy,
    })
}

fn refusal_json(refusal: &MechanismCompositionRefusal) -> String {
    let all: Vec<Json> = refusal.refusals.iter().map(one_refusal).collect();
    let mut first = all.first().cloned().unwrap_or_else(|| {
        json!({
            "code": refusal.code(),
            "stage": "compose",
            "detail": "mechanism_composition.request_unsupported",
            "message": "mechanism composition refused",
            "offending": Json::Null,
            "expected": Json::Null,
            "supplied": Json::Null,
            "remedy": Json::Null,
        })
    });
    first["refusals"] = Json::Array(all);
    first.to_string()
}

// ---------------------------------------------------------------------------
// Declarations.
// ---------------------------------------------------------------------------

fn parse_quantity(value: &Json) -> PyResult<ScientificQuantity> {
    let wire: ScientificQuantityWire = serde_json::from_value(value.clone())
        .map_err(|error| crate::value_err(format!("invalid quantity coordinate: {error}")))?;
    ScientificQuantity::try_from(wire).map_err(crate::value_err)
}

fn parse_quantities(map: &Map<String, Json>, key: &str) -> PyResult<Vec<ScientificQuantity>> {
    list(map, key)?.iter().map(parse_quantity).collect()
}

fn parse_term(value: &Json) -> PyResult<ParameterTerm> {
    let map = object(value, "parameter term")?;
    match text(map, "kind")?.as_str() {
        "intercept" => Ok(ParameterTerm::Intercept),
        "coefficient" => Ok(ParameterTerm::Coefficient(text(map, "parent")?)),
        other => Err(crate::value_err(format!("unknown parameter term `{other}`"))),
    }
}

fn parse_terms(map: &Map<String, Json>, key: &str) -> PyResult<Vec<ParameterTerm>> {
    list(map, key)?.iter().map(parse_term).collect()
}

fn parse_meaning(name: &str) -> PyResult<UncertaintyMeaning> {
    match name {
        "posterior" => Ok(UncertaintyMeaning::Posterior),
        "sampling_distribution" => Ok(UncertaintyMeaning::SamplingDistribution),
        other => Err(crate::value_err(format!("unknown uncertainty meaning `{other}`"))),
    }
}

fn parse_uncertainty(value: &Json) -> PyResult<MechanismUncertainty> {
    let map = object(value, "mechanism uncertainty")?;
    Ok(MechanismUncertainty {
        meaning: parse_meaning(&text(map, "meaning")?)?,
        parameters: parse_terms(map, "parameters")?,
        covariance: reals(map, "covariance")?,
    })
}

fn parse_optional_uncertainty(
    map: &Map<String, Json>,
    key: &str,
) -> PyResult<Option<MechanismUncertainty>> {
    match present(map, key) {
        None => Ok(None),
        Some(value) => parse_uncertainty(value).map(Some),
    }
}

fn parse_domain(value: &Json) -> PyResult<Domain> {
    let map = object(value, "domain")?;
    let bound = |key: &str, default: f64| -> PyResult<f64> {
        match present(map, key) {
            None => Ok(default),
            Some(value) => value
                .as_f64()
                .ok_or_else(|| crate::value_err(format!("domain bound `{key}` must be a number"))),
        }
    };
    match text(map, "kind")?.as_str() {
        "real" => Ok(Domain::Real),
        "interval" => Ok(Domain::Interval {
            lo: bound("lo", f64::NEG_INFINITY)?,
            hi: bound("hi", f64::INFINITY)?,
        }),
        "levels" => Ok(Domain::Levels(texts(map, "levels")?)),
        other => Err(crate::value_err(format!("unknown domain `{other}`"))),
    }
}

fn pair_list(map: &Map<String, Json>, key: &str) -> PyResult<Vec<(String, Json)>> {
    let mut out = Vec::new();
    for entry in list(map, key)? {
        let pair = entry.as_array().filter(|pair| pair.len() == 2).ok_or_else(|| {
            crate::value_err(format!("`{key}` holds an entry that is not a pair"))
        })?;
        let name = pair[0]
            .as_str()
            .ok_or_else(|| crate::value_err(format!("`{key}` holds a pair without a name")))?;
        out.push((name.to_owned(), pair[1].clone()));
    }
    Ok(out)
}

fn pair_numbers(map: &Map<String, Json>, key: &str) -> PyResult<Vec<(String, f64)>> {
    pair_list(map, key)?
        .into_iter()
        .map(|(name, value)| {
            value
                .as_f64()
                .map(|number| (name, number))
                .ok_or_else(|| crate::value_err(format!("`{key}` holds a non-number")))
        })
        .collect()
}

fn pair_texts(map: &Map<String, Json>, key: &str) -> PyResult<Vec<(String, String)>> {
    pair_list(map, key)?
        .into_iter()
        .map(|(name, value)| {
            value
                .as_str()
                .map(|level| (name, level.to_owned()))
                .ok_or_else(|| crate::value_err(format!("`{key}` holds a non-string")))
        })
        .collect()
}

fn parse_source(value: &Json) -> PyResult<MechanismSource> {
    let map = object(value, "mechanism source")?;
    match text(map, "kind")?.as_str() {
        "fitted_linear" => Ok(MechanismSource::FittedLinear {
            intercept: real(map, "intercept")?,
            sigma: real(map, "sigma")?,
            coefficients: pair_numbers(map, "coefficients")?,
        }),
        "declared_linear" => Ok(MechanismSource::DeclaredLinear {
            intercept: real(map, "intercept")?,
            sigma: real(map, "sigma")?,
            coefficients: pair_numbers(map, "coefficients")?,
        }),
        "declared_table" => {
            let mut rows = Vec::new();
            for entry in list(map, "rows")? {
                let row = object(entry, "table row")?;
                rows.push(TabularRow {
                    parent_levels: pair_texts(row, "parent_levels")?,
                    probabilities: pair_numbers(row, "probabilities")?,
                });
            }
            Ok(MechanismSource::DeclaredTable { levels: texts(map, "levels")?, rows })
        }
        other => Err(crate::value_err(format!("unknown mechanism source `{other}`"))),
    }
}

fn parse_declaration(value: &Json) -> PyResult<MechanismDeclaration> {
    let map = object(value, "mechanism declaration")?;
    let mut latent_causes = Vec::new();
    for entry in list(map, "latent_causes")? {
        let latent = object(entry, "latent cause")?;
        latent_causes.push(LatentCauseDeclaration {
            latent_id: text(latent, "latent_id")?,
            shared_with_variable_ids: texts(latent, "shared_with_variable_ids")?,
        });
    }
    Ok(MechanismDeclaration {
        output: parse_quantity(member(map, "output")?)?,
        parents: parse_quantities(map, "parents")?,
        conditioning_variable_ids: texts(map, "conditioning_variable_ids")?,
        // The facade overwrites the label from the source of the numbers.
        origin: MechanismOriginKind::Fitted,
        artifact_id: text(map, "artifact_id")?,
        evidence_id: text(map, "evidence_id")?,
        latent_causes,
    })
}

fn parse_mechanism(value: &Json) -> PyResult<MechanismInput> {
    let map = object(value, "mechanism")?;
    let mut parent_domains = Vec::new();
    for entry in list(map, "parent_domains")? {
        let pair = entry.as_array().filter(|pair| pair.len() == 2).ok_or_else(|| {
            crate::value_err("`parent_domains` holds an entry that is not a pair")
        })?;
        let name = pair[0]
            .as_str()
            .ok_or_else(|| crate::value_err("`parent_domains` holds a pair without a name"))?;
        parent_domains.push((name.to_owned(), parse_domain(&pair[1])?));
    }
    Ok(MechanismInput {
        declaration: parse_declaration(member(map, "declaration")?)?,
        source: parse_source(member(map, "source")?)?,
        output_support: parse_domain(member(map, "output_support")?)?,
        parent_domains,
        uncertainty: parse_optional_uncertainty(map, "uncertainty")?,
        latent_loadings: pair_numbers(map, "latent_loadings")?,
    })
}

fn parse_admission(value: &Json) -> PyResult<TransportAdmission> {
    let map = object(value, "admission")?;
    let coordinate = match text(map, "coordinate")?.as_str() {
        "population" => AdmissionCoordinate::Population,
        "regime" => AdmissionCoordinate::Regime,
        other => return Err(crate::value_err(format!("unknown admission coordinate `{other}`"))),
    };
    let basis = object(member(map, "basis")?, "admission basis")?;
    let basis = match text(basis, "kind")?.as_str() {
        "selection_diagram" => {
            AdmissionBasis::SelectionDiagram { targets: texts(basis, "targets")? }
        }
        "declared_invariance" => {
            AdmissionBasis::DeclaredInvariance { assumption_id: text(basis, "assumption_id")? }
        }
        other => return Err(crate::value_err(format!("unknown admission basis `{other}`"))),
    };
    Ok(TransportAdmission {
        variable_id: text(map, "variable_id")?,
        coordinate,
        source_id: text(map, "source_id")?,
        target_id: text(map, "target_id")?,
        basis,
    })
}

fn parse_parameter_id(value: &Json) -> PyResult<ParameterId> {
    let map = object(value, "parameter id")?;
    Ok(ParameterId { mechanism: text(map, "mechanism")?, term: parse_term(member(map, "term")?)? })
}

fn parse_dependence(value: &Json) -> PyResult<DependenceInput> {
    let map = object(value, "dependence")?;
    match text(map, "kind")?.as_str() {
        "unknown" => Ok(DependenceInput::Unknown),
        "independent_by_design" => {
            Ok(DependenceInput::IndependentByDesign { justification: text(map, "justification")? })
        }
        "aligned_joint_law" => Ok(DependenceInput::AlignedJointLaw {
            law_id: text(map, "law_id")?,
            meaning: parse_meaning(&text(map, "meaning")?)?,
            parameters: list(map, "parameters")?
                .iter()
                .map(parse_parameter_id)
                .collect::<PyResult<Vec<_>>>()?,
            covariance: reals(map, "covariance")?,
        }),
        "cross_covariance" => {
            let mut blocks = Vec::new();
            for entry in list(map, "blocks")? {
                let block = object(entry, "cross-covariance block")?;
                blocks.push(CrossCovarianceBlock {
                    first: text(block, "first")?,
                    second: text(block, "second")?,
                    first_parameters: parse_terms(block, "first_parameters")?,
                    second_parameters: parse_terms(block, "second_parameters")?,
                    values: reals(block, "values")?,
                });
            }
            Ok(DependenceInput::CrossCovariance { blocks })
        }
        other => Err(crate::value_err(format!("unknown dependence `{other}`"))),
    }
}

fn parse_request(value: &Json) -> PyResult<MechanismCompositionRequest> {
    let map = object(value, "composition")?;
    let mechanisms =
        list(map, "mechanisms")?.iter().map(parse_mechanism).collect::<PyResult<Vec<_>>>()?;
    let mut edges = Vec::new();
    for entry in list(map, "edges")? {
        let pair = entry
            .as_array()
            .filter(|pair| pair.len() == 2)
            .and_then(|pair| Some((pair[0].as_str()?, pair[1].as_str()?)))
            .ok_or_else(|| crate::value_err("`edges` holds an entry that is not a name pair"))?;
        edges.push((pair.0.to_owned(), pair.1.to_owned()));
    }
    let admissions =
        list(map, "admissions")?.iter().map(parse_admission).collect::<PyResult<Vec<_>>>()?;
    Ok(MechanismCompositionRequest {
        target_population_id: text(map, "target_population_id")?,
        target_regime_id: text(map, "target_regime_id")?,
        mechanisms,
        edges,
        admissions,
        dependence: parse_dependence(member(map, "dependence")?)?,
        admit_declared_mechanisms: flag(map, "admit_declared_mechanisms")?,
    })
}

fn parse_refit(value: &Json) -> PyResult<MechanismRefit> {
    let map = object(value, "refit")?;
    Ok(MechanismRefit {
        variable_id: text(map, "variable_id")?,
        source: parse_source(member(map, "source")?)?,
        artifact_id: text(map, "artifact_id")?,
        evidence_id: text(map, "evidence_id")?,
        uncertainty: parse_optional_uncertainty(map, "uncertainty")?,
    })
}

fn parse_interventions(value: &Json) -> PyResult<Vec<(String, f64)>> {
    let map = object(value, "interventions")?;
    pair_numbers(map, "interventions")
}

fn parse_finite_interventions(value: &Json) -> PyResult<Vec<(String, String)>> {
    let map = object(value, "interventions")?;
    pair_texts(map, "interventions")
}

fn parse_contrast(value: &Json) -> PyResult<MeanContrast> {
    let map = object(value, "contrast")?;
    let mut arms = Vec::new();
    for entry in list(map, "arms")? {
        let arm = object(entry, "contrast arm")?;
        arms.push(ContrastArm {
            weight: real(arm, "weight")?,
            interventions: pair_numbers(arm, "interventions")?,
        });
    }
    Ok(MeanContrast { outcome: text(map, "outcome")?, arms })
}

fn borrowed<T: AsRef<str>>(pairs: &[(T, f64)]) -> Vec<(&str, f64)> {
    pairs.iter().map(|(name, value)| (name.as_ref(), *value)).collect()
}

// ---------------------------------------------------------------------------
// Results as data.
// ---------------------------------------------------------------------------

const fn family_name(family: MechanismFamily) -> &'static str {
    match family {
        MechanismFamily::LinearGaussian => "linear_gaussian",
        MechanismFamily::Tabular => "tabular",
    }
}

fn term_json(term: &ParameterTerm) -> Json {
    match term {
        ParameterTerm::Intercept => json!({"kind": "intercept"}),
        ParameterTerm::Coefficient(parent) => json!({"kind": "coefficient", "parent": parent}),
    }
}

fn result_json(result: &CompositionResultWire) -> Json {
    match result {
        CompositionResultWire::GaussianLaw { mean_bits, variance_bits } => json!({
            "kind": "gaussian_law",
            "mean": f64::from_bits(*mean_bits),
            "variance": f64::from_bits(*variance_bits),
        }),
        CompositionResultWire::GaussianFunctional {
            point_bits,
            variance_bits,
            dependence,
            method,
        } => json!({
            "kind": "gaussian_functional",
            "point": f64::from_bits(*point_bits),
            "variance": f64::from_bits(*variance_bits),
            "dependence": dependence,
            "method": method,
        }),
        CompositionResultWire::FiniteDistribution { probabilities } => json!({
            "kind": "finite_distribution",
            "probabilities": probabilities.iter().map(|(level, bits)| {
                json!([level, f64::from_bits(*bits)])
            }).collect::<Vec<_>>(),
        }),
    }
}

fn consumed_json(consumed: &ConsumedComposition) -> Json {
    json!({
        "premises_digest": consumed.premises_digest,
        "family": family_name(consumed.family),
        "result": result_json(&consumed.result),
    })
}

// ---------------------------------------------------------------------------
// Handle.
// ---------------------------------------------------------------------------

/// A composition that compiled: one executable joint model and its plan.
/// Every method that changes it returns a new handle.
#[pyclass(name = "PreparedCompositionHandle", skip_from_py_object)]
pub(crate) struct PreparedCompositionHandle {
    inner: PreparedMechanismComposition,
}

type Refused<T> = (Option<T>, Option<String>);

/// Variable names and one column of draws per variable.
type SampledColumns = (Vec<String>, Vec<Vec<f64>>);

fn refused<T>(refusal: &MechanismCompositionRefusal) -> Refused<T> {
    (None, Some(refusal_json(refusal)))
}

#[pymethods]
impl PreparedCompositionHandle {
    /// Canonical identity of the semantically checked plan.
    fn plan_identity(&self) -> String {
        self.inner.plan_identity_hex()
    }

    /// The family the composition executes in.
    fn family(&self) -> &'static str {
        family_name(self.inner.family())
    }

    /// Variables of the joint model, in canonical order.
    fn variables(&self) -> Vec<String> {
        self.inner.variables().to_vec()
    }

    /// Variables whose mechanism was written down, not estimated.
    fn declared_mechanisms(&self) -> Vec<String> {
        self.inner.declared_mechanisms().into_iter().map(str::to_owned).collect()
    }

    /// Always the typed refusal: no measured coverage stands behind the
    /// parameter-covariance variance.
    fn interval(&self) -> String {
        match self.inner.interval() {
            Err(refusal) => refusal_json(&refusal),
            Ok(never) => match never {},
        }
    }

    /// The joint Gaussian law under `do(interventions)`.
    fn intervention_law(&self, interventions_json: &str) -> PyResult<Refused<String>> {
        let interventions = parse_interventions(&parse(interventions_json, "interventions")?)?;
        Ok(match self.inner.intervention_law(&borrowed(&interventions)) {
            Ok(law) => (
                Some(
                    json!({
                        "variables": law.variables(),
                        "mean": law.mean(),
                        "covariance": law.covariance_matrix(),
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// `sum_k weight_k * E[outcome | do(arm_k)]`.
    fn mean_contrast(&self, contrast_json: &str) -> PyResult<Refused<f64>> {
        let contrast = parse_contrast(&parse(contrast_json, "contrast")?)?;
        Ok(match self.inner.mean_contrast(&contrast) {
            Ok(value) => (Some(value), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// The first-order delta-method variance of a contrast, with its
    /// gradient and how the dependence was licensed.
    fn mean_contrast_uncertainty(&self, contrast_json: &str) -> PyResult<Refused<String>> {
        let contrast = parse_contrast(&parse(contrast_json, "contrast")?)?;
        Ok(match self.inner.mean_contrast_uncertainty(&contrast) {
            Ok(u) => (
                Some(
                    json!({
                        "point": u.point,
                        "variance": u.variance,
                        "standard_error": u.standard_error,
                        "meaning": u.meaning.map(|m| match m {
                            UncertaintyMeaning::Posterior => "posterior",
                            UncertaintyMeaning::SamplingDistribution => "sampling_distribution",
                        }),
                        "dependence": u.dependence,
                        "method": u.method,
                        "gradient": u.gradient.iter().map(|(id, g)| json!({
                            "mechanism": id.mechanism,
                            "term": term_json(&id.term),
                            "gradient": g,
                        })).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Draw `n_rows` units from the (possibly intervened) joint model.
    #[pyo3(signature = (interventions_json, n_rows, seed, *, cancel=None))]
    fn sample(
        &self,
        py: Python<'_>,
        interventions_json: &str,
        n_rows: usize,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<SampledColumns>> {
        let interventions = parse_interventions(&parse(interventions_json, "interventions")?)?;
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(seed, None, cancel);
            Ok(inner.sample(&borrowed(&interventions), n_rows, &ctx))
        })?;
        Ok(match outcome {
            Ok(table) => (Some((table.variables, table.columns)), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// The exact finite joint law under `do(interventions)`.
    #[pyo3(signature = (interventions_json, *, cancel=None))]
    fn finite_law(
        &self,
        py: Python<'_>,
        interventions_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let interventions =
            parse_finite_interventions(&parse(interventions_json, "interventions")?)?;
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            let named: Vec<(&str, &str)> =
                interventions.iter().map(|(v, l)| (v.as_str(), l.as_str())).collect();
            Ok(inner.finite_law(&named, &ctx))
        })?;
        Ok(match outcome {
            Ok(law) => (
                Some(
                    json!({
                        "variables": law.variables(),
                        "levels": law.levels(),
                        "probabilities": law.probabilities(),
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Replace the named mechanisms with their refits and recompile. A
    /// refusal leaves this handle as it was.
    #[pyo3(signature = (refits_json, dependence_json=None, *, cancel=None))]
    fn refresh_mechanisms(
        &self,
        py: Python<'_>,
        refits_json: &str,
        dependence_json: Option<&str>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<PreparedCompositionHandle>> {
        let refits = match parse(refits_json, "refits")? {
            Json::Array(items) => items.iter().map(parse_refit).collect::<PyResult<Vec<_>>>()?,
            _ => return Err(crate::value_err("refits must be a list")),
        };
        let dependence = match dependence_json {
            Some(text) => Some(parse_dependence(&parse(text, "dependence")?)?),
            None => None,
        };
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            Ok(inner.refresh_mechanisms(refits, dependence, &ctx))
        })?;
        Ok(match outcome {
            Ok(refreshed) => (Some(PreparedCompositionHandle { inner: refreshed }), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Seal the premises and one query into an artifact: the query is
    /// executed and its result stored as bit patterns.
    #[pyo3(signature = (query_json, *, cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        query_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<(Py<PyBytes>, String, String)>> {
        let query: CompositionQueryWire = serde_json::from_value(parse(query_json, "query")?)
            .map_err(|error| crate::value_err(format!("invalid composition query: {error}")))?;
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            Ok(inner.export(query, &ctx))
        })?;
        Ok(match outcome {
            Ok(done) => (
                Some((
                    PyBytes::new(py, &done.bytes).unbind(),
                    done.premises_digest,
                    result_json(&done.result).to_string(),
                )),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }
}

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------

/// Prepare a declared composition of independently fitted mechanisms.
#[pyfunction]
#[pyo3(signature = (composition_json, *, cancel=None))]
fn mechanism_composition_prepare(
    py: Python<'_>,
    composition_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<PreparedCompositionHandle>> {
    let request = parse_request(&parse(composition_json, "composition")?)?;
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(StudyBuilder::mechanism_composition(request, &ctx))
    })?;
    Ok(match outcome {
        Ok(inner) => (Some(PreparedCompositionHandle { inner }), None),
        Err(refusal) => refused(&refusal),
    })
}

/// Independently consume a composition artifact: recompute the premises
/// digest, recompile under the coherence checker and replay the query.
#[pyfunction]
#[pyo3(signature = (artifact, expected_premises_digest=None, *, cancel=None))]
fn mechanism_composition_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_premises_digest: Option<&str>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let bytes = artifact.to_vec();
    let expected = expected_premises_digest.map(str::to_owned);
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(consume_mechanism_composition_artifact(&bytes, expected.as_deref(), &ctx))
    })?;
    Ok(match outcome {
        Ok(consumed) => (Some(consumed_json(&consumed).to_string()), None),
        Err(refusal) => refused(&refusal),
    })
}

/// The numbers of one variable's fitted mechanism of a fitted graphical
/// causal model, as a composition source labelled fitted.
///
/// Only a plain linear-Gaussian slot composes; a basis expansion, discrete,
/// constant, vacant or pending slot is refused with the slot's own detail.
#[pyfunction]
fn mechanism_composition_fitted_source(
    model: PyRef<'_, crate::gcm_api::PyFittedGcm>,
    variable: &str,
) -> PyResult<Refused<String>> {
    let names = &model.names;
    let position = names
        .iter()
        .position(|name| name == variable)
        .ok_or_else(|| crate::value_err(format!("unknown variable `{variable}`")))?;
    let id = |index: usize| -> PyResult<antecedent_core::VariableId> {
        Ok(antecedent_core::VariableId::from_raw(
            u32::try_from(index).map_err(|_| crate::value_err("variable index does not fit"))?,
        ))
    };
    let compiled = &model.inner.model;
    let dense = compiled.dense_of(id(position)?).ok_or_else(|| {
        crate::value_err(format!("`{variable}` is not a node of the fitted model"))
    })?;
    let gather = compiled
        .gather_for(dense)
        .ok_or_else(|| crate::value_err(format!("`{variable}` has no parent gather plan")))?;
    let mut parents = Vec::with_capacity(gather.parents.len());
    for parent in gather.parents.iter() {
        let mut found = None;
        for (index, name) in names.iter().enumerate() {
            if compiled.dense_of(id(index)?) == Some(*parent) {
                found = Some(name.as_str());
                break;
            }
        }
        parents.push(found.ok_or_else(|| crate::value_err("a parent of the fit has no name"))?);
    }
    let slot = compiled.mechanisms.get(dense);
    Ok(match MechanismSource::from_fitted_slot(slot, &parents) {
        Ok(source) => (Some(source_json(&source).to_string()), None),
        Err(refusal) => refused(&refusal),
    })
}

fn source_json(source: &MechanismSource) -> Json {
    match source {
        MechanismSource::FittedLinear { intercept, sigma, coefficients }
        | MechanismSource::DeclaredLinear { intercept, sigma, coefficients } => json!({
            "kind": if matches!(source, MechanismSource::FittedLinear { .. }) {
                "fitted_linear"
            } else {
                "declared_linear"
            },
            "intercept": intercept,
            "sigma": sigma,
            "coefficients": coefficients.iter().map(|(p, c)| json!([p, c])).collect::<Vec<_>>(),
        }),
        MechanismSource::DeclaredTable { levels, rows } => json!({
            "kind": "declared_table",
            "levels": levels,
            "rows": rows.iter().map(|row| json!({
                "parent_levels": row.parent_levels.iter()
                    .map(|(p, l)| json!([p, l])).collect::<Vec<_>>(),
                "probabilities": row.probabilities.iter()
                    .map(|(l, p)| json!([l, p])).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        }),
        _ => json!({"kind": "unrecognized"}),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedCompositionHandle>()?;
    module.add_function(wrap_pyfunction!(mechanism_composition_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(mechanism_composition_consume, module)?)?;
    module.add_function(wrap_pyfunction!(mechanism_composition_fitted_source, module)?)?;
    Ok(())
}
