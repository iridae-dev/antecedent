//! Binding a preference contract, or an ordinary decision contract, to the studies that
//! estimate its quantities.
//!
//! Every quantity an action reads must be a coordinate the study's query targets: its
//! outcome, a regime on its dose grid, the contract's population, the static horizon, the
//! mean on the identity scale, in the outcome's units (the schema's, when it declares one). A
//! refusal names the action and the attribute of a value, or the position (`input k`) of an
//! ordinary action's input.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use antecedent_contract::{ExternalRefusal, ProgramBinding, QuantityMismatch, ScientificQuantity};
use antecedent_core::reason_details::{
    decision_evaluation, decision_evidence, native_claims, preference_compile, program_binding,
};
use antecedent_core::{
    CanonicalWriter, CausalQuery, CausalSchema, ExecutionContext, ObservationSpec,
    ResponseFunctional, SemanticDigest, TargetPopulation,
};
use antecedent_data::TableView;
use antecedent_design::preference_contract::{ActionInput, PreferenceContract};
use antecedent_io::distribution_artifact::DistributionCalibration;

use super::evidence::{StudyUnits, analysed_rows, rows_digest, study_units};
use super::{
    BOUND_TAG, CompiledContract, DemandedQuantity, IDENTITY_TRANSFORM, MEAN_FUNCTIONAL,
    PreferenceRouteError, STAGE_BIND, UNDECLARED_UNITS, at, unsupported_query,
};
use crate::analysis::contract::CausalContract;
use crate::analysis::execute::Study;
use crate::analysis::native_claims::NativeResponseContext;

/// The mean response curve a study targets, read from its own query and schema.
#[derive(Clone, Debug)]
struct StudyQuery {
    outcome: String,
    treatment: String,
    outcome_units: Option<String>,
    dose_units: Option<String>,
    doses: Vec<f64>,
    schema: CausalSchema,
    causal_target: String,
    causal_identification: String,
    snapshot: String,
    rows: usize,
    origin: String,
    origin_rows: Option<Arc<[u64]>>,
    provenance_declared: bool,
    analysed: Option<Arc<[usize]>>,
    units: Option<StudyUnits>,
}

/// A study bound to the decision: its program binding and the coordinates it targets.
#[derive(Clone, Debug)]
pub(super) struct BoundStudy {
    pub(super) binding: ProgramBinding,
    pub(super) coordinates: Vec<ScientificQuantity>,
    pub(super) schema: CausalSchema,
    pub(super) causal_target: String,
    pub(super) causal_identification: String,
    pub(super) snapshot: String,
    /// Rows of the study's table: the observations its analysis reads.
    pub(super) rows: usize,
    /// Identity of the table the study's rows came from: the source table of a declared or
    /// remembered projection, else the study's own table. It binds the table's columns and
    /// rows, not its analysis mask, its weights, its schema roles or the analysis compiled
    /// over it, so studies of one table share it whatever outcome, mask or weighting they read.
    pub(super) origin: String,
    /// Index in the source table of each row, when the projection selected rows.
    pub(super) origin_rows: Option<Arc<[u64]>>,
    /// Whether the caller declared the table the study's rows came from. Without the
    /// declaration a study of a projection or row selection names its own table, so its
    /// overlap with a study of another table is unknown.
    pub(super) provenance_declared: bool,
    /// The rows of the study's table its analysis reads, when an analysis mask leaves some
    /// out; masked rows are not observations the study rests on.
    pub(super) analysed: Option<Arc<[usize]>>,
    /// The units the analysis reads, when the study declares a unit key.
    pub(super) units: Option<StudyUnits>,
}

impl BoundStudy {
    /// The response context the claim conversion reads: the compiler-derived binding, the
    /// data snapshot and the execution's seed.
    pub(super) fn context(&self, ctx: &ExecutionContext) -> NativeResponseContext {
        NativeResponseContext {
            program: self.binding.clone(),
            snapshot_id: self.snapshot.clone(),
            rng_id: format!("antecedent.execution.master_seed:{}", ctx.rng.master_seed()),
            calibration: DistributionCalibration::NotMeasured,
        }
    }
}

fn study_query(study: &Study, causal: &CausalContract) -> Result<StudyQuery, PreferenceRouteError> {
    let CausalQuery::Response(query) = &study.query else {
        return Err(unsupported_query("a query that is not a response"));
    };
    let ResponseFunctional::MeanCurve { outcome, treatment } = &query.functional else {
        return Err(unsupported_query("a response functional other than a mean curve"));
    };
    if query.temporal.is_some()
        || query.observation != ObservationSpec::Complete
        || query.target_population != TargetPopulation::AllObserved
        || !query.outcome_functional.is_mean()
    {
        return Err(unsupported_query(
            "a temporal, partially observed, retargeted or non-mean response curve",
        ));
    }
    let doses = treatment.grid.values().map_err(|_| unsupported_query("an invalid dose grid"))?;
    let crate::analysis::builder::DataInput::Tabular(data) = &study.data else {
        return Err(unsupported_query("a mean response curve of non-tabular data"));
    };
    let schema = data.schema().clone();
    let variable = |id| {
        schema.get(id).map(|v| (v.name.to_string(), v.unit.as_deref().map(str::to_owned))).map_err(
            |_| {
                PreferenceRouteError::from(
                    ExternalRefusal::new(
                        antecedent_core::reason_code!("unknown_variable"),
                        STAGE_BIND,
                        native_claims::UNKNOWN_VARIABLE,
                    )
                    .with_supplied(format!("{id:?}")),
                )
            },
        )
    };
    let (outcome_id, treatment_id) = (*outcome, treatment.variable);
    let (outcome, outcome_units) = variable(outcome_id)?;
    let (treatment, dose_units) = variable(treatment_id)?;
    let observations = SemanticDigest::from_bytes(rows_digest(data)?).to_hex();
    let declared = study.evidence_rows.as_deref();
    let provenance = declared.and_then(|rows| rows.provenance.as_ref());
    let resolved = declared.and_then(|rows| rows.units.as_ref());
    if let Some(key) = resolved.and_then(|units| units.variable)
        && (key == outcome_id || key == treatment_id)
    {
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            STAGE_BIND,
            preference_compile::UNIT_KEY_INVALID,
        )
        .with_offending(format!("{key:?}"))
        .with_expected("a unit identifier column that is neither the treatment nor the outcome")
        .with_remedy("declare the column that identifies the units the rows observe")
        .into());
    }
    Ok(StudyQuery {
        outcome,
        treatment,
        outcome_units,
        dose_units,
        doses,
        schema,
        causal_target: causal.identities.target.to_hex(),
        causal_identification: causal.identities.identification.to_hex(),
        snapshot: causal.identities.data_snapshot.to_hex(),
        origin: provenance.map_or(observations, |p| p.table().to_owned()),
        origin_rows: provenance.and_then(|p| p.rows().map(Arc::from)),
        provenance_declared: provenance.is_some(),
        analysed: analysed_rows(data),
        units: resolved.map(|units| study_units(units, data)),
        rows: data.row_count(),
    })
}

/// Every quantity input of `contract`: `(action, label, quantity)`, the label naming the
/// attribute a value reads, or the position (`input k`) of an ordinary action's input.
pub(super) fn quantity_inputs(
    contract: &CompiledContract,
) -> Vec<(&str, String, &ScientificQuantity)> {
    match contract {
        CompiledContract::Tradeoff(contract) => contract
            .actions
            .iter()
            .flat_map(|action| {
                action.inputs.iter().zip(&contract.model.attributes).filter_map(
                    move |(input, attribute)| match input {
                        ActionInput::Quantity(quantity) => {
                            Some((action.id.as_str(), attribute.id.clone(), quantity))
                        }
                        ActionInput::Fixed { .. } => None,
                    },
                )
            })
            .collect(),
        CompiledContract::Ordinary(contract) => {
            contract
                .actions
                .iter()
                .flat_map(|action| {
                    action.inputs.iter().enumerate().map(move |(k, quantity)| {
                        (action.id.as_str(), format!("input {k}"), quantity)
                    })
                })
                .collect()
        }
    }
}

/// The study whose outcome `quantity` names.
fn study_of(
    studies: &[StudyQuery],
    (action, attribute): (&str, &str),
    quantity: &ScientificQuantity,
) -> Result<usize, PreferenceRouteError> {
    studies.iter().position(|study| study.outcome == quantity.variable_id).ok_or_else(|| {
        ExternalRefusal::new(
            antecedent_core::reason_code!("quantity_semantics_mismatch"),
            STAGE_BIND,
            program_binding::TREATMENT_OUTCOME_SUBSTITUTION,
        )
        .with_offending(at(action, attribute))
        .with_expected(
            studies.iter().map(|study| study.outcome.as_str()).collect::<Vec<_>>().join(", "),
        )
        .with_supplied(quantity.variable_id.clone())
        .with_remedy("read an outcome the compiled studies estimate")
        .into()
    })
}

/// The refusal of a quantity no coordinate of a study matches, naming the first dimension
/// that differs from the closest coordinate (the one at the same regime, else the first).
pub(super) fn coordinate_refusal(
    coordinates: &[ScientificQuantity],
    (action, attribute): (&str, &str),
    quantity: &ScientificQuantity,
) -> PreferenceRouteError {
    let reference = coordinates
        .iter()
        .find(|coordinate| coordinate.regime_id == quantity.regime_id)
        .or_else(|| coordinates.first());
    let mismatch = reference.map_or(QuantityMismatch::InvalidCoordinate, |coordinate| {
        quantity.require_same_coordinate(coordinate).err().unwrap_or(QuantityMismatch::Regime)
    });
    let field_of =
        |field: fn(&ScientificQuantity) -> String| reference.map_or_else(String::new, field);
    let (detail, expected, supplied) = match mismatch {
        QuantityMismatch::Variable => (
            program_binding::TREATMENT_OUTCOME_SUBSTITUTION,
            field_of(|c| c.variable_id.clone()),
            quantity.variable_id.clone(),
        ),
        QuantityMismatch::Units => {
            (native_claims::UNITS_CONFLICT, field_of(|c| c.units.clone()), quantity.units.clone())
        }
        QuantityMismatch::Population => (
            program_binding::POPULATION_MISMATCH,
            field_of(|c| c.population_id.clone()),
            quantity.population_id.clone(),
        ),
        QuantityMismatch::Regime => (
            program_binding::DOSE_GRID_CHANGED,
            coordinates.iter().map(|c| c.regime_id.as_str()).collect::<Vec<_>>().join(", "),
            quantity.regime_id.clone(),
        ),
        QuantityMismatch::Horizon
        | QuantityMismatch::Functional
        | QuantityMismatch::Conditioning
        | QuantityMismatch::Transform
        | QuantityMismatch::InvalidCoordinate => (
            decision_evaluation::QUANTITY_NOT_FOUND,
            field_of(ScientificQuantity::canonical_identity),
            quantity.canonical_identity(),
        ),
    };
    ExternalRefusal::new(
        antecedent_core::reason_code!("quantity_semantics_mismatch"),
        STAGE_BIND,
        detail,
    )
    .with_offending(at(action, attribute))
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy("read a coordinate the study's query targets; nothing is reinterpreted")
    .into()
}

/// Bind every quantity input of `contract` to a coordinate of one study, resolving each
/// study's outcome units (the schema's, else the units every input reading it declares).
fn bind_studies(
    contract: &CompiledContract,
    studies: Vec<StudyQuery>,
) -> Result<(Vec<BoundStudy>, Vec<DemandedQuantity>), PreferenceRouteError> {
    let mut seen = HashSet::new();
    if let Some(repeated) = studies.iter().find(|study| !seen.insert(study.outcome.as_str())) {
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            STAGE_BIND,
            decision_evidence::INVALID_DECLARATION,
        )
        .with_offending(repeated.outcome.clone())
        .with_expected("one compiled study per outcome")
        .with_remedy("estimate each outcome once, or several outcomes of one design jointly")
        .into());
    }
    let mut units: Vec<Option<String>> =
        studies.iter().map(|study| study.outcome_units.clone()).collect();
    for (action, attribute, quantity) in quantity_inputs(contract) {
        let index = study_of(&studies, (action, attribute.as_str()), quantity)?;
        if units[index].is_none() {
            units[index] = Some(quantity.units.clone());
        }
    }
    let bound: Vec<BoundStudy> = studies
        .into_iter()
        .zip(units)
        .map(|(study, outcome_units)| {
            let binding = ProgramBinding {
                graph_id: study.causal_identification.clone(),
                contract_id: study.causal_target.clone(),
                treatment_id: study.treatment,
                outcome_id: study.outcome,
                population_id: contract.target_population().to_owned(),
                intervention_kind: "do".to_owned(),
                horizon: 0,
                dose_grid: study.doses,
                dose_units: study.dose_units.unwrap_or_else(|| UNDECLARED_UNITS.to_owned()),
                outcome_units: outcome_units.unwrap_or_else(|| UNDECLARED_UNITS.to_owned()),
                functional_id: MEAN_FUNCTIONAL.to_owned(),
                transform_id: IDENTITY_TRANSFORM.to_owned(),
            };
            BoundStudy {
                coordinates: binding.expected_quantities(),
                binding,
                schema: study.schema,
                causal_target: study.causal_target,
                causal_identification: study.causal_identification,
                snapshot: study.snapshot,
                rows: study.rows,
                origin: study.origin,
                origin_rows: study.origin_rows,
                provenance_declared: study.provenance_declared,
                analysed: study.analysed,
                units: study.units,
            }
        })
        .collect();
    let mut demanded: Vec<DemandedQuantity> = Vec::new();
    for (action, attribute, quantity) in quantity_inputs(contract) {
        let source = bound
            .iter()
            .position(|study| study.binding.outcome_id == quantity.variable_id)
            .ok_or_else(|| coordinate_refusal(&[], (action, attribute.as_str()), quantity))?;
        let coordinates = &bound[source].coordinates;
        let Some(coordinate) =
            coordinates.iter().find(|c| quantity.require_same_coordinate(c).is_ok())
        else {
            return Err(coordinate_refusal(coordinates, (action, attribute.as_str()), quantity));
        };
        let read = (action.to_owned(), attribute);
        match demanded.iter_mut().find(|d| d.quantity == *coordinate) {
            Some(entry) => entry.read_by.push(read),
            None => demanded.push(DemandedQuantity {
                quantity: coordinate.clone(),
                source,
                read_by: vec![read],
            }),
        }
    }
    Ok((bound, demanded))
}

pub(super) fn bound_identity(contract_identity: &str, causal: &[(String, String)]) -> String {
    let mut writer = CanonicalWriter::new(BOUND_TAG);
    writer.text(contract_identity).word(causal.len() as u64);
    for (target, identification) in causal {
        writer.text(target).text(identification);
    }
    writer.finish_hex()
}

/// A contract bound to the causal contracts of the studies that estimate its quantities.
#[derive(Clone, Debug)]
pub(super) struct BoundContract {
    pub(super) contract: CompiledContract,
    pub(super) contract_identity: String,
    pub(super) studies: Vec<BoundStudy>,
    pub(super) quantities: Vec<DemandedQuantity>,
    pub(super) identity: String,
}

impl BoundContract {
    pub(super) fn bind(
        contract: CompiledContract,
        studies: &[(&Study, &CausalContract)],
    ) -> Result<Self, PreferenceRouteError> {
        let contract_identity = contract.identity()?;
        let queries = studies
            .iter()
            .map(|(study, causal)| study_query(study, causal))
            .collect::<Result<Vec<_>, _>>()?;
        let (studies, quantities) = bind_studies(&contract, queries)?;
        let causal: Vec<(String, String)> = studies
            .iter()
            .map(|s| (s.causal_target.clone(), s.causal_identification.clone()))
            .collect();
        let identity = bound_identity(&contract_identity, &causal);
        Ok(Self { contract, contract_identity, studies, quantities, identity })
    }

    /// `(causal target, causal identification)` of every study, lowercase hex.
    pub(super) fn causal_bindings(&self) -> Vec<(String, String)> {
        self.studies
            .iter()
            .map(|s| (s.causal_target.clone(), s.causal_identification.clone()))
            .collect()
    }
}

/// A preference contract bound to the causal contracts of the studies that estimate its
/// quantities.
#[derive(Clone, Debug)]
pub struct BoundPreferenceContract {
    contract: PreferenceContract,
    studies: Vec<BoundStudy>,
    quantities: Vec<DemandedQuantity>,
    identity: String,
}

impl BoundPreferenceContract {
    pub(super) fn bind(
        contract: &PreferenceContract,
        studies: &[(&Study, &CausalContract)],
    ) -> Result<Self, PreferenceRouteError> {
        let bound = BoundContract::bind(CompiledContract::Tradeoff(contract.clone()), studies)?;
        Ok(Self {
            contract: contract.clone(),
            studies: bound.studies,
            quantities: bound.quantities,
            identity: bound.identity,
        })
    }

    /// The preference contract.
    #[must_use]
    pub const fn contract(&self) -> &PreferenceContract {
        &self.contract
    }

    /// The quantities the actions read, each with the study that targets it.
    #[must_use]
    pub fn quantities(&self) -> &[DemandedQuantity] {
        &self.quantities
    }

    /// The coordinates study `source` targets, one per dose of its grid.
    #[must_use]
    pub fn coordinates(&self, source: usize) -> Option<&[ScientificQuantity]> {
        self.studies.get(source).map(|study| &study.coordinates[..])
    }

    /// Canonical identity of the preference contract bound to every study's causal target
    /// and identification.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// `(causal target, causal identification)` of every study, lowercase hex.
    #[must_use]
    pub fn causal_bindings(&self) -> Vec<(String, String)> {
        self.studies
            .iter()
            .map(|s| (s.causal_target.clone(), s.causal_identification.clone()))
            .collect()
    }
}
