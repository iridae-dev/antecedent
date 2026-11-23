//! Cohort schema, transport support and retained-model prediction.

use std::collections::BTreeSet;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_learn::FittedEffect;

use super::{
    COVARIATE_SHIFT_ASSUMPTION, CohortBinding, CohortColumn, CohortRefusal,
    CovariateShiftDeclaration, NEGATIVE_VARIANCE_TOLERANCE, RetainedEffectModel,
    SUPPORT_BASIS_DECLARED, SUPPORT_BASIS_FITTED_BOX, compensated_sum, invalid, invalid_input,
    schema_mismatch, semantics_mismatch,
};
use crate::compact_export::{CompactExport, ExportQuery, ExportRefusal, TermSpec};

fn map_export_refusal(refusal: &ExportRefusal, row_id: &str, shifted: bool) -> CohortRefusal {
    match refusal.detail {
        antecedent_core::reason_details::compact_export::OUT_OF_SUPPORT
        | antecedent_core::reason_details::compact_export::MASKED_REGION => {
            outside_refusal(shifted, format!("row={row_id} feature_or_region={}", refusal.subject))
        }
        _ => schema_mismatch(
            antecedent_core::reason_details::cohort_decision::FEATURE_SCHEMA_MISMATCH,
            format!("{}: {}", refusal.detail, refusal.subject),
            "supply exactly the model's named features with their declared types",
        ),
    }
}

fn outside_refusal(shifted: bool, subject: String) -> CohortRefusal {
    if shifted {
        CohortRefusal::new(
            antecedent_core::reason_code!("transport_support_failure"),
            antecedent_core::reason_details::cohort_decision::SHIFT_SUPPORT_FAILURE,
            subject,
            "restrict the cohort to rows inside the fitted support, or supply a model fitted \
             on rows that cover the target population",
        )
    } else {
        invalid_input(
            antecedent_core::reason_details::cohort_decision::OUTSIDE_SUPPORT,
            subject,
            "restrict the cohort to rows inside the retained support contract (building a \
             restricted cohort is a new, separately identified cohort)",
        )
    }
}

pub(super) fn resolve_shift(
    model: &RetainedEffectModel,
    cohort: &CohortBinding,
    shift: Option<&CovariateShiftDeclaration>,
) -> Result<bool, CohortRefusal> {
    let same = cohort.population_id == model.population_id;
    match (same, shift) {
        (true, None) => Ok(false),
        (true, Some(_)) => Err(semantics_mismatch(
            antecedent_core::reason_details::cohort_decision::SHIFT_DECLARATION_UNNEEDED,
            cohort.population_id.clone(),
            "drop the shift declaration: the cohort is declared to be the model's population",
        )),
        (false, None) => Err(CohortRefusal::new(
            antecedent_core::reason_code!("transport_missing_evidence"),
            antecedent_core::reason_details::cohort_decision::SHIFT_DECLARATION_REQUIRED,
            format!("model={} cohort={}", model.population_id, cohort.population_id),
            "declare a covariate shift: source and target population and the effect \
             modifiers (the model's features) given which the conditional contrast is \
             invariant, with every cohort row inside the support contract",
        )),
        (false, Some(declared)) => {
            let mut features = model.feature_names();
            features.sort();
            let matches_declared = declared.source_population == model.population_id
                && declared.target_population == cohort.population_id
                && declared.invariant_modifiers == features
                && declared.assumption == COVARIATE_SHIFT_ASSUMPTION;
            if matches_declared {
                Ok(true)
            } else {
                Err(CohortRefusal::new(
                    antecedent_core::reason_code!("transport_not_certified"),
                    antecedent_core::reason_details::cohort_decision::SHIFT_DECLARATION_MISMATCH,
                    format!("model={} cohort={}", model.population_id, cohort.population_id),
                    "the declaration must name the model's population as source, the \
                     cohort's population as target and exactly the model's features as the \
                     invariant effect modifiers",
                ))
            }
        }
    }
}

pub(super) fn require_exact_schema(
    model: &RetainedEffectModel,
    cohort: &CohortBinding,
) -> Result<(), CohortRefusal> {
    let expected: BTreeSet<String> = model.feature_names().into_iter().collect();
    let supplied: BTreeSet<String> = cohort.features.iter().cloned().collect();
    if expected != supplied {
        let difference: Vec<String> = expected.symmetric_difference(&supplied).cloned().collect();
        return Err(schema_mismatch(
            antecedent_core::reason_details::cohort_decision::FEATURE_SCHEMA_MISMATCH,
            difference.join(","),
            "supply exactly the model's named features: none missing and none extra",
        ));
    }
    Ok(())
}

pub(super) struct Predictions {
    pub(super) row_contrasts: Vec<f64>,
    pub(super) basis: &'static str,
    pub(super) fitted_rows: Option<u64>,
    pub(super) standard_error: Option<f64>,
}

pub(super) fn portable_predictions(
    effect: &FittedEffect,
    names: &[String],
    cohort: &CohortBinding,
    shifted: bool,
    ctx: &ExecutionContext,
) -> Result<Predictions, CohortRefusal> {
    let Some(contract) = &effect.support else {
        return Err(invalid_input(
            antecedent_core::reason_details::cohort_decision::SUPPORT_CONTRACT_ABSENT,
            "fitted_effect.support",
            "refit with a producer that retains the fitted-row support contract; row-level \
             support cannot be checked without it",
        ));
    };
    let mut columns: Vec<&[f64]> = Vec::with_capacity(names.len());
    for name in names {
        let column = cohort.numeric(name).ok_or_else(|| {
            schema_mismatch(
                antecedent_core::reason_details::cohort_decision::FEATURE_TYPE_MISMATCH,
                name.clone(),
                "supply this feature as a finite numeric column",
            )
        })?;
        columns.push(column);
    }
    let report = effect
        .check_support(&columns, cohort.rows())
        .map_err(|e| {
            schema_mismatch(
                antecedent_core::reason_details::cohort_decision::FEATURE_SCHEMA_MISMATCH,
                e.to_string(),
                "supply the model's features",
            )
        })?
        .ok_or_else(|| {
            invalid_input(
                antecedent_core::reason_details::cohort_decision::SUPPORT_CONTRACT_ABSENT,
                "fitted_effect.support",
                "refit with a producer that retains the fitted-row support contract",
            )
        })?;
    if let (Some(row), Some(feature)) = (report.first_row, report.first_feature) {
        return Err(outside_refusal(
            shifted,
            format!(
                "row={} feature={} rows_outside={}",
                cohort.row_ids[row], names[feature], report.rows_outside
            ),
        ));
    }
    let ids: Vec<VariableId> = effect.features.iter().map(|v| VariableId::from_raw(*v)).collect();
    let row_contrasts = effect.predict(&ids, &columns, cohort.rows(), ctx).map_err(|e| {
        invalid_input(
            antecedent_core::reason_details::cohort_decision::PREDICTION_REFUSED,
            e.to_string(),
            "the retained model could not predict at these rows; check the cohort features",
        )
    })?;
    Ok(Predictions {
        row_contrasts,
        basis: SUPPORT_BASIS_FITTED_BOX,
        fitted_rows: Some(contract.rows),
        standard_error: None,
    })
}

fn basis_row(
    terms: &[TermSpec],
    cohort: &CohortBinding,
    row: usize,
) -> Result<Vec<f64>, CohortRefusal> {
    let mismatch = |quantity: &str| {
        schema_mismatch(
            antecedent_core::reason_details::cohort_decision::FEATURE_TYPE_MISMATCH,
            quantity.to_owned(),
            "supply each feature with the type its declared support has",
        )
    };
    terms
        .iter()
        .map(|term| match term {
            TermSpec::Intercept => Ok(1.0),
            TermSpec::Linear { quantity } => {
                cohort.numeric(quantity).map(|values| values[row]).ok_or_else(|| mismatch(quantity))
            }
            TermSpec::Power { quantity, degree } => cohort
                .numeric(quantity)
                .map(|values| values[row].powi(i32::try_from(*degree).unwrap_or(i32::MAX)))
                .ok_or_else(|| mismatch(quantity)),
            TermSpec::OneHot { quantity, level } => cohort
                .level(quantity, row)
                .map(|chosen| if chosen == level { 1.0 } else { 0.0 })
                .ok_or_else(|| mismatch(quantity)),
        })
        .collect()
}

pub(super) fn compact_predictions(
    export: &CompactExport,
    cohort: &CohortBinding,
    shifted: bool,
    ctx: &ExecutionContext,
) -> Result<Predictions, CohortRefusal> {
    let rows = cohort.rows();
    let body = export.body();
    let width = body.terms.len();
    let mut row_contrasts = Vec::with_capacity(rows);
    let mut weighted_basis = vec![Vec::with_capacity(rows); width];
    for row in 0..rows {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::CANCELLED,
                "cohort",
                "re-run without cancellation; no value was issued",
            ));
        }
        let mut query = ExportQuery::new();
        for (name, column) in cohort.features.iter().zip(&cohort.columns) {
            query = match column {
                CohortColumn::Numeric(values) => query.with_number(name, values[row]),
                CohortColumn::Categorical(values) => query.with_level(name, &values[row]),
            };
        }
        let point = export
            .evaluate(&query)
            .map_err(|refusal| map_export_refusal(&refusal, &cohort.row_ids[row], shifted))?;
        row_contrasts.push(point.point);
        let phi = basis_row(&body.terms, cohort, row)?;
        for (slot, value) in weighted_basis.iter_mut().zip(phi) {
            slot.push(value * cohort.weights[row]);
        }
    }
    let phibar: Vec<f64> =
        weighted_basis.into_iter().map(|terms| compensated_sum(terms.into_iter())).collect();
    let variance = compensated_sum(phibar.iter().enumerate().map(|(j, pj)| {
        let row = &body.covariance[j * width..(j + 1) * width];
        pj * compensated_sum(row.iter().zip(&phibar).map(|(v, pk)| v * pk))
    }));
    let scale: f64 =
        body.covariance.iter().step_by(width + 1).zip(&phibar).map(|(v, p)| v * p * p).sum();
    if variance < -NEGATIVE_VARIANCE_TOLERANCE * (1.0 + scale.abs()) || !variance.is_finite() {
        return Err(invalid_input(
            antecedent_core::reason_details::cohort_decision::NEGATIVE_VARIANCE,
            "phibar_v_phibar",
            "the stored covariance is not positive semidefinite on this cohort; refit",
        ));
    }
    Ok(Predictions {
        row_contrasts,
        basis: SUPPORT_BASIS_DECLARED,
        fitted_rows: None,
        standard_error: Some(variance.max(0.0).sqrt()),
    })
}
