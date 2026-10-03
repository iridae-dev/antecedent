//! Python execution boundary for the batch retarget (E3).
//!
//! The Rust [`BatchRetargetReport`] crosses as a dictionary (points, joint covariance,
//! contrasts, failed members, the closed simultaneous interval, and the tidy rows) and the
//! Python wrapper parses it into frozen dataclasses. The handle keeps the score tables of
//! its last `estimate`, so a retarget after an estimate reweights that estimate's rows.

use antecedent::{
    BatchRetargetError, BatchRetargetReport, BatchRetargetRequest, BatchScores, MemberFailure,
    RetargetClaim, RetargetContrast,
};
use antecedent_core::VariableId;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::ate_api::PyPreparedBatch;
use crate::transport_z_api::to_py_json;

/// `(name, query index, weights, depends_on names)` of one claim.
type ClaimSpec = (String, usize, Vec<f64>, Vec<String>);
/// `(name, [(claim name, coefficient)])` of one contrast.
type ContrastSpec = (String, Vec<(String, f64)>);

fn retarget_error(error: BatchRetargetError) -> PyErr {
    let fields = error.refusal_fields();
    crate::preflight_api::with_refusal_fields(
        crate::refusal(error.code, format!("{}: {}", error.detail, error.message)),
        Some(&fields),
    )
}

fn failure_json(failure: &MemberFailure) -> serde_json::Value {
    serde_json::json!({
        "reason_code": failure.reason_code,
        "detail": failure.detail,
        "message": failure.message,
        "support_refused": failure.support_refused,
        "refusal_fields": failure.fields.as_deref().map(crate::preflight_api::refusal_fields_value),
    })
}

fn report_json(report: &BatchRetargetReport) -> serde_json::Value {
    let claims: Vec<_> = report
        .claims
        .iter()
        .map(|claim| match &claim.outcome {
            Ok(point) => serde_json::json!({
                "name": claim.name,
                "query_index": claim.query_index,
                "estimand": claim.estimand,
                "status": if point.std_error.is_some() { "ok" } else { "point_only" },
                "value": point.value,
                "std_error": point.std_error,
                "uncertainty_kind": point.uncertainty_kind.as_str(),
                "n_eff": point.n_eff,
                "n_eff_by_arm": point.n_eff_by_arm,
                "propensity_range": point.propensity_range.map(|(lo, hi)| [lo, hi]),
                "nuisance_provenance": point.nuisance_provenance,
                "diagnostics": point.diagnostics,
                "failure": serde_json::Value::Null,
            }),
            Err(failure) => serde_json::json!({
                "name": claim.name,
                "query_index": claim.query_index,
                "estimand": claim.estimand,
                "status": "failed",
                "failure": failure_json(failure),
            }),
        })
        .collect();
    let contrasts: Vec<_> = report
        .contrasts
        .iter()
        .map(|contrast| match &contrast.outcome {
            Ok(point) => serde_json::json!({
                "name": contrast.name,
                "terms": contrast.terms,
                "status": if point.std_error.is_some() { "ok" } else { "point_only" },
                "value": point.value,
                "std_error": point.std_error,
                "uncertainty_kind": point.uncertainty_kind.as_str(),
                "failure": serde_json::Value::Null,
            }),
            Err(failure) => serde_json::json!({
                "name": contrast.name,
                "terms": contrast.terms,
                "status": "failed",
                "failure": failure_json(failure),
            }),
        })
        .collect();
    let covariance = report.covariance.as_ref().map(|c| {
        let dim = c.matrix.dim;
        let rows: Vec<Vec<f64>> =
            (0..dim).map(|i| (0..dim).map(|j| c.matrix.get(i, j)).collect()).collect();
        serde_json::json!({ "names": c.names, "matrix": rows })
    });
    let rows: Vec<_> = report
        .tidy_rows()
        .into_iter()
        .map(|row| {
            serde_json::json!({
                "family_id": row.family_id,
                "family_complete": row.family_complete,
                "family_size": row.family_size,
                "family_failed": row.family_failed,
                "kind": row.kind.as_str(),
                "name": row.name,
                "estimand": row.estimand,
                "status": row.status,
                "value": row.value,
                "std_error": row.std_error,
                "uncertainty_kind": row.uncertainty_kind,
                "simultaneous_interval": row.simultaneous_interval,
                "support_status": row.support_status,
                "n_eff": row.n_eff,
                "refusal_code": row.refusal_code,
                "refusal_detail": row.refusal_detail,
                "refusal_message": row.refusal_message,
                "diagnostics": row.diagnostics,
                "query_index": row.query_index,
                "scores_source": row.scores_source,
                "snapshot_id": row.snapshot_id,
                "estimator_fingerprint": row.estimator_fingerprint,
                "nuisance_provenance": row.nuisance_provenance,
            })
        })
        .collect();
    let complete = report.complete_family();
    serde_json::json!({
        "family_id": report.family_id,
        "snapshot_id": report.snapshot_id,
        "scores_source": report.scores_source.as_str(),
        "estimator_fingerprint": report.estimator_fingerprint,
        "claims": claims,
        "contrasts": contrasts,
        "covariance": covariance,
        "simultaneous_interval": {
            "status": "closed",
            "reason_code": report.simultaneous_interval.code,
            "detail": report.simultaneous_interval.detail,
            "message": report.simultaneous_interval.message,
        },
        "complete": complete.is_ok(),
        "complete_refusal": complete.as_ref().err().map(|e| serde_json::json!({
            "reason_code": e.code,
            "detail": e.detail,
            "message": e.message,
        })),
        "failed_members": report.failed_members(),
        "point_only_members": report.point_only_members(),
        "inference_claim": "point_only",
        "scope_note": report.scope_note,
        "rows": rows,
    })
}

impl PyPreparedBatch {
    /// The scores a retarget now reads: the last estimate's, else the prepare-time tables.
    fn current_scores(&self) -> PyResult<BatchScores> {
        let retained = self
            .last_scores
            .lock()
            .map_err(|_| PyValueError::new_err("batch scores lock poisoned"))?;
        Ok(retained.as_ref().map_or_else(|| self.inner.prepared_scores(), |s| s.as_ref().clone()))
    }

    fn variable_id(&self, name: &str) -> PyResult<VariableId> {
        let at = self
            .names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown depends_on {name}")))?;
        Ok(VariableId::from_raw(
            u32::try_from(at).map_err(|_| PyValueError::new_err("too many columns"))?,
        ))
    }
}

#[pymethods]
impl PyPreparedBatch {
    /// Original-row index every claim's weights must align with: the rows of the scores a
    /// retarget now reads (the last `estimate`'s, else the prepare-time rows). `None` when no
    /// plan has scores; refuses `row_weights_bound_to_snapshot` when plans disagree.
    fn retarget_rows(&self) -> PyResult<Option<Vec<u32>>> {
        let scores = self.current_scores()?;
        Ok(scores.common_rows().map_err(retarget_error)?.map(|rows| rows.to_vec()))
    }

    /// `prepared` or `estimated`: where the scores a retarget now reads came from.
    fn retarget_scores_source(&self) -> PyResult<String> {
        Ok(self.current_scores()?.source().as_str().to_string())
    }

    /// Identity of the row snapshot the scores a retarget now reads share.
    fn retarget_snapshot(&self) -> PyResult<Option<String>> {
        self.current_scores()?.snapshot_id().map_err(retarget_error)
    }

    /// Retarget a declared family of claims; the report as a dictionary.
    #[pyo3(signature = (claims, contrasts, *, expected_snapshot=None))]
    fn retarget_family(
        &self,
        py: Python<'_>,
        claims: Vec<ClaimSpec>,
        contrasts: Vec<ContrastSpec>,
        expected_snapshot: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let scores = self.current_scores()?;
        let claims = claims
            .into_iter()
            .map(|(name, query_index, weights, depends_on)| {
                Ok::<_, PyErr>(RetargetClaim {
                    name,
                    query_index,
                    weights,
                    depends_on: depends_on
                        .iter()
                        .map(|n| self.variable_id(n))
                        .collect::<PyResult<Vec<_>>>()?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let contrasts = contrasts
            .into_iter()
            .map(|(name, coefficients)| RetargetContrast { name, coefficients })
            .collect();
        let request = BatchRetargetRequest { claims, contrasts, expected_snapshot };
        let ctx = crate::py_execution_context(1, 1);
        let report = self.inner.retarget(&scores, &request, &ctx).map_err(retarget_error)?;
        to_py_json(py, &report_json(&report))
    }
}
