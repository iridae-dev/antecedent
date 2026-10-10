//! Binding of an external claim to the identified program and request it answers.
//!
//! A causal contract is only evidence about one concrete question: this graph,
//! this treatment and outcome, this target population, this regime and horizon,
//! this dose grid. A hash of the graph alone certifies none of those, and a
//! caller-supplied `quantities` override can silently describe a different
//! question. [`ProgramBinding`] names every one of them and derives one durable
//! identity from all of them; [`check_external_against_program`] refuses any
//! claim whose declared coordinates or identity differ from it, each with its own
//! typed detail under the `program_binding` namespace.
//!
//! Nothing here executes or estimates anything. It checks coordinates and
//! identities only, and it never converts, rescales or reinterprets a value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::FpCategory;

use crate::{
    BoundExternalClaim, CheckedCausalContract, ExternalRefusal, ExternalResult, QuantityRole,
    ScientificQuantity, bind_external_result,
};

const IDENTITY_KEY: &str = "antecedent.program_binding.v1";
const GRAPH_ONLY_KEY: &str = "antecedent.program_binding.graph_only.v1";

/// Render a dose as the shortest label that Python's `:g` format also produces
/// (six significant digits), so Rust and Python name the same regime.
#[must_use]
pub fn dose_label(value: f64) -> String {
    if !value.is_finite() {
        return format!("{value}");
    }
    if value.classify() == FpCategory::Zero {
        return "0".to_owned();
    }
    let scientific = format!("{value:.5e}");
    let Some((mantissa, exponent)) = scientific.split_once('e') else {
        return scientific;
    };
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if !(-4..6).contains(&exponent) {
        let mantissa = if mantissa.contains('.') {
            mantissa.trim_end_matches('0').trim_end_matches('.')
        } else {
            mantissa
        };
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.unsigned_abs());
    }
    let decimals = usize::try_from(5 - exponent).unwrap_or(0);
    let fixed = format!("{value:.decimals$}");
    if fixed.contains('.') {
        fixed.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        fixed
    }
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

/// The full identified-program and request identity an external claim must answer.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramBinding {
    /// Identity of the graph the contract was identified on.
    pub graph_id: String,
    /// Identity of the checked causal contract.
    pub contract_id: String,
    /// Stable identity of the one intervened treatment variable.
    pub treatment_id: String,
    /// Stable identity of the outcome variable.
    pub outcome_id: String,
    /// Stable identity of the target population.
    pub population_id: String,
    /// Kind of intervention that forms each regime, for example `do`.
    pub intervention_kind: String,
    /// Number of ordered causal steps to the outcome; zero is static.
    pub horizon: u32,
    /// Strictly increasing finite dose grid, in `dose_units`.
    pub dose_grid: Vec<f64>,
    /// Units of the treatment doses.
    pub dose_units: String,
    /// Units of the outcome.
    pub outcome_units: String,
    /// Estimand functional, for example `mean`.
    pub functional_id: String,
    /// Declared scale of the outcome, for example `identity`.
    pub transform_id: String,
}

impl ProgramBinding {
    /// Check that every identity is present and the dose grid is well formed.
    ///
    /// # Errors
    /// A blank identity, an empty or non-finite grid, or a grid that is not
    /// strictly increasing refuses with `program_binding.invalid_binding`.
    // The refusal is the cold path of a once-per-binding check; boxing would not pay.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self) -> Result<(), ExternalRefusal> {
        let blank_field = [
            ("graph_id", &self.graph_id),
            ("contract_id", &self.contract_id),
            ("treatment_id", &self.treatment_id),
            ("outcome_id", &self.outcome_id),
            ("population_id", &self.population_id),
            ("intervention_kind", &self.intervention_kind),
            ("dose_units", &self.dose_units),
            ("outcome_units", &self.outcome_units),
            ("functional_id", &self.functional_id),
            ("transform_id", &self.transform_id),
        ]
        .into_iter()
        .find(|(_, value)| blank(value));
        if let Some((name, _)) = blank_field {
            return Err(invalid(name));
        }
        let grid_ok = !self.dose_grid.is_empty()
            && self.dose_grid.iter().all(|dose| dose.is_finite())
            && self.dose_grid.windows(2).all(|pair| pair[0] < pair[1]);
        if grid_ok { Ok(()) } else { Err(invalid("dose_grid")) }
    }

    /// Regime identity for one dose, matching the coordinate Python derives.
    #[must_use]
    pub fn regime_for(&self, dose: f64) -> String {
        format!("{}({}={})", self.intervention_kind, self.treatment_id, dose_label(dose))
    }

    /// The scientific coordinates the program requests, one per dose.
    #[must_use]
    pub fn expected_quantities(&self) -> Vec<ScientificQuantity> {
        self.dose_grid
            .iter()
            .map(|dose| ScientificQuantity {
                variable_id: self.outcome_id.clone(),
                variable_name: self.outcome_id.clone(),
                role: QuantityRole::Outcome,
                units: self.outcome_units.clone(),
                population_id: self.population_id.clone(),
                regime_id: self.regime_for(*dose),
                horizon: self.horizon,
                functional_id: self.functional_id.clone(),
                conditioning: Vec::new(),
                transform_id: self.transform_id.clone(),
            })
            .collect()
    }

    /// Durable identity: BLAKE3 over every field of the program and request.
    ///
    /// Strings are length-prefixed and doses are hashed by their exact bits, so
    /// changing any one field, including the contract identity, changes it.
    #[must_use]
    pub fn identity(&self) -> String {
        let mut hasher = blake3::Hasher::new_derive_key(IDENTITY_KEY);
        for field in [
            &self.graph_id,
            &self.contract_id,
            &self.treatment_id,
            &self.outcome_id,
            &self.population_id,
            &self.intervention_kind,
            &self.dose_units,
            &self.outcome_units,
            &self.functional_id,
            &self.transform_id,
        ] {
            put_str(&mut hasher, field);
        }
        hasher.update(&self.horizon.to_le_bytes());
        hasher.update(&(self.dose_grid.len() as u64).to_le_bytes());
        for dose in &self.dose_grid {
            hasher.update(&dose.to_bits().to_le_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    /// The identity a graph-only derivation would give. It is insufficient: two
    /// different questions on one graph share it, so it is always refused.
    #[must_use]
    pub fn graph_only_identity(graph_id: &str) -> String {
        let mut hasher = blake3::Hasher::new_derive_key(GRAPH_ONLY_KEY);
        put_str(&mut hasher, graph_id);
        hasher.finalize().to_hex().to_string()
    }
}

/// What an external specification declares about the program it answers.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalProgramClaim {
    /// Contract identity the specification names.
    pub contract_id: String,
    /// Graph identity the specification names.
    pub graph_id: String,
    /// Durable program identity the specification declares.
    pub declared_identity: String,
    /// Treatment variable the specification intervenes on.
    pub treatment_id: String,
    /// Outcome variable the specification asks for.
    pub outcome_id: String,
    /// Target population the specification asks for.
    pub population_id: String,
    /// Dose grid the specification asks for.
    pub doses: Vec<f64>,
    /// Units of those doses.
    pub dose_units: String,
    /// The requested coordinates, including any caller override.
    pub quantities: Vec<ScientificQuantity>,
}

impl ExternalProgramClaim {
    /// The faithful claim for `binding`: its own identity and derived coordinates.
    #[must_use]
    pub fn declared_by(binding: &ProgramBinding) -> Self {
        Self {
            contract_id: binding.contract_id.clone(),
            graph_id: binding.graph_id.clone(),
            declared_identity: binding.identity(),
            treatment_id: binding.treatment_id.clone(),
            outcome_id: binding.outcome_id.clone(),
            population_id: binding.population_id.clone(),
            doses: binding.dose_grid.clone(),
            dose_units: binding.dose_units.clone(),
            quantities: binding.expected_quantities(),
        }
    }
}

/// A claim that passed every program check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramCheck {
    /// The verified durable program identity.
    pub identity: String,
    /// Number of coordinates verified against the program.
    pub coordinates: usize,
}

fn refuse(
    code: &'static str,
    detail: &str,
    offending: Option<String>,
    expected: String,
    supplied: String,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal {
        code,
        stage: "bind",
        detail: detail.to_owned(),
        offending,
        expected: Some(expected),
        supplied: Some(supplied),
        capability: None,
        remedy: Some(remedy),
    }
}

fn invalid(field: &str) -> ExternalRefusal {
    ExternalRefusal {
        code: crate::reason_code!("invalid_argument"),
        stage: "bind",
        detail: "program_binding.invalid_binding".to_owned(),
        offending: Some(field.to_owned()),
        expected: None,
        supplied: None,
        capability: None,
        remedy: Some("name every program identity and give a strictly increasing finite grid"),
    }
}

fn substitution(offending: &str, expected: &str, supplied: &str) -> ExternalRefusal {
    refuse(
        crate::reason_code!("external_binding_mismatch"),
        "program_binding.treatment_outcome_substitution",
        Some(offending.to_owned()),
        expected.to_owned(),
        supplied.to_owned(),
        "ask the provider for the identified treatment and outcome",
    )
}

fn population(offending: &str, expected: &str, supplied: &str) -> ExternalRefusal {
    refuse(
        crate::reason_code!("quantity_semantics_mismatch"),
        "program_binding.population_mismatch",
        Some(offending.to_owned()),
        expected.to_owned(),
        supplied.to_owned(),
        "request the result for the program's target population",
    )
}

fn grid_changed(offending: &str, expected: String, supplied: String) -> ExternalRefusal {
    refuse(
        crate::reason_code!("quantity_semantics_mismatch"),
        "program_binding.dose_grid_changed",
        Some(offending.to_owned()),
        expected,
        supplied,
        "request exactly the identified dose grid, in the identified units",
    )
}

fn identity_mismatch(offending: &str, expected: &str, supplied: &str) -> ExternalRefusal {
    refuse(
        crate::reason_code!("external_binding_mismatch"),
        "program_binding.contract_identity_mismatch",
        Some(offending.to_owned()),
        expected.to_owned(),
        supplied.to_owned(),
        "derive the program identity from the full contract and request",
    )
}

fn grid_text(doses: &[f64]) -> String {
    doses.iter().map(|dose| dose_label(*dose)).collect::<Vec<_>>().join(",")
}

fn same_grid(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Whether `declared` is an identity that depends on the graph alone.
fn graph_only(binding: &ProgramBinding, claim: &ExternalProgramClaim) -> bool {
    let declared = claim.declared_identity.as_str();
    let bare = claim.graph_id.strip_prefix("graph:").unwrap_or(&claim.graph_id);
    declared == claim.graph_id
        || declared == binding.graph_id
        || declared.starts_with("graph:")
        || declared == format!("contract:{bare}")
        || declared == ProgramBinding::graph_only_identity(&claim.graph_id)
        || declared == ProgramBinding::graph_only_identity(&binding.graph_id)
}

// The refusal is the cold path of a once-per-refusal check; boxing would not pay.
#[allow(clippy::result_large_err)]
fn check_identity(
    binding: &ProgramBinding,
    claim: &ExternalProgramClaim,
) -> Result<(), ExternalRefusal> {
    if graph_only(binding, claim) {
        return Err(refuse(
            crate::reason_code!("external_binding_mismatch"),
            "program_binding.graph_only_identity",
            Some("declared_identity".to_owned()),
            binding.identity(),
            claim.declared_identity.clone(),
            "a graph hash does not name the query, population or dose grid",
        ));
    }
    if claim.graph_id != binding.graph_id {
        return Err(identity_mismatch("graph_id", &binding.graph_id, &claim.graph_id));
    }
    if claim.contract_id != binding.contract_id {
        return Err(identity_mismatch("contract_id", &binding.contract_id, &claim.contract_id));
    }
    Ok(())
}

// The refusal is the cold path of a once-per-refusal check; boxing would not pay.
#[allow(clippy::result_large_err)]
fn check_variables(
    binding: &ProgramBinding,
    claim: &ExternalProgramClaim,
) -> Result<(), ExternalRefusal> {
    if claim.treatment_id != binding.treatment_id {
        return Err(substitution("treatment_id", &binding.treatment_id, &claim.treatment_id));
    }
    if claim.outcome_id != binding.outcome_id {
        return Err(substitution("outcome_id", &binding.outcome_id, &claim.outcome_id));
    }
    if claim.population_id != binding.population_id {
        return Err(population("population_id", &binding.population_id, &claim.population_id));
    }
    if !same_grid(&claim.doses, &binding.dose_grid) {
        return Err(grid_changed("doses", grid_text(&binding.dose_grid), grid_text(&claim.doses)));
    }
    if claim.dose_units != binding.dose_units {
        return Err(grid_changed(
            "dose_units",
            binding.dose_units.clone(),
            claim.dose_units.clone(),
        ));
    }
    Ok(())
}

fn regime_refusal(
    binding: &ProgramBinding,
    at: &str,
    expected: &ScientificQuantity,
    supplied: &ScientificQuantity,
) -> ExternalRefusal {
    let kind = format!("{}(", binding.intervention_kind);
    let own = format!("{}{}=", kind, binding.treatment_id);
    if supplied.regime_id.starts_with(&kind) && !supplied.regime_id.starts_with(&own) {
        substitution(at, &binding.treatment_id, &supplied.regime_id)
    } else {
        grid_changed(at, expected.regime_id.clone(), supplied.regime_id.clone())
    }
}

// The refusal is the cold path of a once-per-refusal check; boxing would not pay.
#[allow(clippy::result_large_err)]
fn check_quantity(
    binding: &ProgramBinding,
    index: usize,
    expected: &ScientificQuantity,
    supplied: &ScientificQuantity,
) -> Result<(), ExternalRefusal> {
    let at = format!("coordinate[{index}]");
    if supplied.variable_id != expected.variable_id || supplied.role != expected.role {
        return Err(substitution(&at, &expected.variable_id, &supplied.variable_id));
    }
    if supplied.population_id != expected.population_id {
        return Err(population(&at, &expected.population_id, &supplied.population_id));
    }
    if supplied.regime_id != expected.regime_id {
        return Err(regime_refusal(binding, &at, expected, supplied));
    }
    expected.require_same_coordinate(supplied).map_err(|mismatch| {
        refuse(
            crate::reason_code!("quantity_semantics_mismatch"),
            "program_binding.quantities_override_mismatch",
            Some(at.clone()),
            format!("{expected:?}"),
            format!("{mismatch:?}: {supplied:?}"),
            "drop the quantities override or make it equal the identified request",
        )
    })
}

// The refusal is the cold path of a once-per-refusal check; boxing would not pay.
#[allow(clippy::result_large_err)]
fn check_quantity_list(
    binding: &ProgramBinding,
    quantities: &[ScientificQuantity],
) -> Result<(), ExternalRefusal> {
    let expected = binding.expected_quantities();
    if quantities.len() != expected.len() {
        return Err(grid_changed(
            "quantities",
            expected.len().to_string(),
            quantities.len().to_string(),
        ));
    }
    for (index, (want, got)) in expected.iter().zip(quantities).enumerate() {
        check_quantity(binding, index, want, got)?;
    }
    Ok(())
}

/// Check an external specification's declared program against the identified one.
///
/// Order: the binding itself, a graph-only identity, graph and contract
/// identity, treatment and outcome, target population, dose grid and units, each
/// coordinate (an incompatible `quantities` override), and last the declared
/// durable identity against the one derived from the full program and request.
///
/// # Errors
/// `program_binding.invalid_binding`, `graph_only_identity`,
/// `contract_identity_mismatch`, `treatment_outcome_substitution`,
/// `population_mismatch`, `dose_grid_changed` or `quantities_override_mismatch`.
// The refusal is the cold path of a once-per-claim check; boxing would not pay.
#[allow(clippy::result_large_err)]
pub fn check_external_against_program(
    binding: &ProgramBinding,
    claim: &ExternalProgramClaim,
) -> Result<ProgramCheck, ExternalRefusal> {
    binding.validate()?;
    check_identity(binding, claim)?;
    check_variables(binding, claim)?;
    check_quantity_list(binding, &claim.quantities)?;
    let identity = binding.identity();
    if claim.declared_identity != identity {
        return Err(identity_mismatch("declared_identity", &identity, &claim.declared_identity));
    }
    Ok(ProgramCheck { identity, coordinates: binding.dose_grid.len() })
}

/// An external claim bound both to a checked contract and to the program it answers.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramBoundClaim {
    claim: BoundExternalClaim,
    program_identity: String,
}

impl ProgramBoundClaim {
    /// The bound external claim; it keeps its external trust and identity.
    #[must_use]
    pub fn claim(&self) -> &BoundExternalClaim {
        &self.claim
    }

    /// The verified durable identity of the full program and request.
    #[must_use]
    pub fn program_identity(&self) -> &str {
        &self.program_identity
    }
}

/// Bind an external result to the checked contract and to the actual program.
///
/// Runs [`check_external_against_program`], requires the contract itself to
/// carry the program's graph and exact coordinates, then binds the result with
/// [`bind_external_result`]. The bound claim stays external; nothing here
/// promotes it to native estimation.
///
/// # Errors
/// Any program refusal, a contract whose graph or coordinates differ from the
/// program, or the structured refusal of the underlying binding.
// The refusal is the cold path of a once-per-claim check; boxing would not pay.
#[allow(clippy::result_large_err)]
pub fn bind_external_result_to_program(
    binding: &ProgramBinding,
    claim: &ExternalProgramClaim,
    contract: &CheckedCausalContract,
    result: &ExternalResult,
) -> Result<ProgramBoundClaim, ExternalRefusal> {
    let checked = check_external_against_program(binding, claim)?;
    if contract.graph_id != binding.graph_id {
        return Err(identity_mismatch("contract.graph_id", &binding.graph_id, &contract.graph_id));
    }
    check_quantity_list(binding, &contract.estimand)?;
    let bound = bind_external_result(contract, result)
        .map_err(|error| error.to_refusal(contract, result))?;
    Ok(ProgramBoundClaim { claim: bound, program_identity: checked.identity })
}

#[cfg(test)]
mod tests {
    use super::dose_label;

    #[test]
    fn dose_labels_match_python_g_format() {
        assert_eq!(dose_label(1.0), "1");
        assert_eq!(dose_label(0.1), "0.1");
        assert_eq!(dose_label(2.5), "2.5");
        assert_eq!(dose_label(-0.75), "-0.75");
        assert_eq!(dose_label(100_000.0), "100000");
        assert_eq!(dose_label(1_000_000.0), "1e+06");
        assert_eq!(dose_label(0.0001), "0.0001");
        assert_eq!(dose_label(0.00001), "1e-05");
        assert_eq!(dose_label(0.0), "0");
    }
}
