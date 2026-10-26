//! Native linear conditional-effect fit for a declared cohort mean.
//!
//! The outcome is regressed by ordinary least squares on a design of the declared
//! effect basis `phi(x) = (1, x_1, ..., x_k)`, its main effect and its interaction with
//! the treatment: columns `phi(x)` then `t * phi(x)`. The interaction coefficients are
//! the coefficients `b` of the linear conditional effect `tau(x) = b' phi(x)`; the
//! covariance of `b_hat` is the HC3 sandwich
//! `(X'X)^-1 [sum_i x_i x_i' e_i^2 / (1 - h_i)^2] (X'X)^-1`, valid under
//! heteroskedastic noise. The fit retains its row count `n` and design width `p`, so
//! the cohort-mean interval is `phibar' b_hat +/- t_{n-p} sqrt(phibar' V phibar)`.
//!
//! What licenses this fit. Treatment assignment that is randomized or ignorable given
//! exactly the declared features, an outcome mean linear in those features under each
//! treatment arm, and independent rows. A misspecified effect or arm mean, and a cohort
//! whose rows lie in high-leverage regions of the fitted design, are named boundaries of
//! the measured coverage, not claims.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    reason = "small dense linear algebra over bounded design widths"
)]

use std::sync::Arc;

use antecedent_core::CalibrationBasis;
use antecedent_core::ExecutionContext;
use antecedent_io_core::cohort_decision::{
    CohortBinding, CohortDecisionSource, CohortOptions, RetainedEffectModel,
};
use antecedent_io_core::compact_export::{
    CompactExport, ExportInput, ExportRefusal, ExportSpec, InputSupport, TermSpec,
};
use antecedent_io_core::quantity_wire::QuantityConditionWire;
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use antecedent_stats::special::student_t_ppf;

use crate::error::EstimationError;

/// Estimator identity of the coverage record.
pub const ESTIMATOR_ID: &str = "cohort.linear_effect.ols_hc3";
/// Interval method identity of the coverage record.
pub const INTERVAL_METHOD: &str = "student_t_hc3_sandwich";
/// Functional identity of the coverage record.
pub const FUNCTIONAL_ID: &str = "cohort_mean_conditional_effect";
/// Smallest leverage-complement `1 - h_i` the fit accepts before refusing.
const MIN_LEVERAGE_COMPLEMENT: f64 = 1e-8;

fn unsupported(message: &'static str) -> EstimationError {
    EstimationError::Unsupported { message }
}

/// A natively fitted linear conditional effect with its sandwich covariance.
#[derive(Clone, Debug, PartialEq)]
pub struct CohortLinearFit {
    /// Effect coefficients `b`, aligned with `[1, x_1, ..., x_k]`.
    pub coefficients: Vec<f64>,
    /// Row-major HC3 covariance of `b_hat`, `(k + 1)` squared.
    pub covariance: Vec<f64>,
    /// Fitted rows `n`.
    pub rows: usize,
    /// Design width `p = 2 (k + 1)`.
    pub width: usize,
    /// Largest leverage `h_i` of the design.
    pub max_leverage: f64,
    /// Fitted range of every feature, `(min, max)`.
    pub feature_ranges: Vec<(f64, f64)>,
}

/// Cholesky inverse of a symmetric positive definite row-major matrix.
fn spd_inverse(a: &[f64], p: usize) -> Option<Vec<f64>> {
    crate::linalg::SpdFactor::new(a, p).map(|factor| factor.inverse())
}

/// Fit the linear conditional effect by OLS with an HC3 sandwich covariance.
///
/// `features` holds one column per declared effect modifier (each as long as
/// `outcome`); `treatment` is the realized treatment (binary 0/1 or a finite dose).
///
/// # Errors
/// Mismatched lengths, a non-finite value, too few rows for the design width, a
/// singular design or a row with leverage 1 (HC3 is undefined there).
#[allow(
    clippy::too_many_lines,
    reason = "one fit: design matrix, sandwich covariance and identity"
)]
pub fn fit_cohort_linear_effect(
    outcome: &[f64],
    treatment: &[f64],
    features: &[&[f64]],
) -> Result<CohortLinearFit, EstimationError> {
    let n = outcome.len();
    let k = features.len();
    let basis = k + 1;
    let p = 2 * basis;
    if treatment.len() != n || features.iter().any(|column| column.len() != n) {
        return Err(unsupported("cohort linear effect: column lengths differ"));
    }
    if n <= p {
        return Err(unsupported(
            "cohort linear effect: the fit needs more rows than design columns",
        ));
    }
    if outcome.iter().chain(treatment).any(|v| !v.is_finite())
        || features.iter().any(|column| column.iter().any(|v| !v.is_finite()))
    {
        return Err(unsupported("cohort linear effect: non-finite input value"));
    }
    // Rows of the design, centred features for conditioning; the effect basis is
    // recovered by the exact affine map below.
    let centres: Vec<f64> = features.iter().map(|c| c.iter().sum::<f64>() / n as f64).collect();
    let row = |i: usize| -> Vec<f64> {
        let mut phi = Vec::with_capacity(basis);
        phi.push(1.0);
        for (column, centre) in features.iter().zip(&centres) {
            phi.push(column[i] - centre);
        }
        let mut x = Vec::with_capacity(p);
        x.extend_from_slice(&phi);
        x.extend(phi.iter().map(|v| v * treatment[i]));
        x
    };
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        let x = row(i);
        for a in 0..p {
            xty[a] += x[a] * outcome[i];
            for b in 0..p {
                xtx[a * p + b] += x[a] * x[b];
            }
        }
    }
    let inverse = spd_inverse(&xtx, p)
        .ok_or_else(|| unsupported("cohort linear effect: the design is singular"))?;
    let beta: Vec<f64> =
        (0..p).map(|a| (0..p).map(|b| inverse[a * p + b] * xty[b]).sum()).collect();
    let mut meat = vec![0.0; p * p];
    let mut max_leverage = 0.0_f64;
    for i in 0..n {
        let x = row(i);
        let fitted: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum();
        let residual = outcome[i] - fitted;
        let leverage: f64 =
            (0..p).map(|a| x[a] * (0..p).map(|b| inverse[a * p + b] * x[b]).sum::<f64>()).sum();
        let complement = 1.0 - leverage;
        if !(complement > MIN_LEVERAGE_COMPLEMENT) {
            return Err(unsupported(
                "cohort linear effect: a fitted row has leverage 1; the sandwich covariance is undefined",
            ));
        }
        max_leverage = max_leverage.max(leverage);
        let w = residual * residual / (complement * complement);
        for a in 0..p {
            for b in 0..p {
                meat[a * p + b] += x[a] * x[b] * w;
            }
        }
    }
    // V = inverse * meat * inverse.
    let mut left = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            left[a * p + b] = (0..p).map(|c| inverse[a * p + c] * meat[c * p + b]).sum();
        }
    }
    let mut full = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            full[a * p + b] = (0..p).map(|c| left[a * p + c] * inverse[c * p + b]).sum();
        }
    }
    // Interaction block on the centred basis, then the affine map back to the
    // uncentred basis: b0 = c0 - sum_j centre_j c_j, b_j = c_j.
    let mut map = vec![0.0; basis * basis];
    map[0] = 1.0;
    for j in 1..basis {
        map[j] = -centres[j - 1];
        map[j * basis + j] = 1.0;
    }
    let block = |a: usize, b: usize| full[(basis + a) * p + basis + b];
    let centred: Vec<f64> = beta[basis..].to_vec();
    let coefficients: Vec<f64> =
        (0..basis).map(|a| (0..basis).map(|b| map[a * basis + b] * centred[b]).sum()).collect();
    let mut covariance = vec![0.0; basis * basis];
    for a in 0..basis {
        for b in 0..basis {
            let mut sum = 0.0;
            for c in 0..basis {
                for d in 0..basis {
                    sum += map[a * basis + c] * block(c, d) * map[b * basis + d];
                }
            }
            covariance[a * basis + b] = sum;
        }
    }
    for a in 0..basis {
        for b in 0..a {
            let mean = 0.5 * (covariance[a * basis + b] + covariance[b * basis + a]);
            covariance[a * basis + b] = mean;
            covariance[b * basis + a] = mean;
        }
    }
    let feature_ranges = features
        .iter()
        .map(|column| {
            column
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
        })
        .collect();
    Ok(CohortLinearFit {
        coefficients,
        covariance,
        rows: n,
        width: p,
        max_leverage,
        feature_ranges,
    })
}

impl CohortLinearFit {
    /// Producer-computed digest binding the retained size, the coefficient vector and
    /// the covariance. Only this module computes it, from its own fit output.
    fn digest(&self) -> String {
        let mut canon =
            antecedent_core::CanonicalWriter::new("antecedent.cohort_linear_effect.fit.v1");
        canon.text(FIT_PROCEDURE_ID).word(self.rows as u64).word(self.width as u64);
        canon.reals(&self.coefficients).reals(&self.covariance);
        canon.finish_hex()
    }

    /// Build the compact export of this fit: the effect coefficients over
    /// `[1, x_1, ..., x_k]` with the HC3 covariance, marked in the response coordinate
    /// as natively fitted with its retained `n` and `p`.
    ///
    /// `response` must declare a conditional treatment contrast (functional
    /// `conditional_mean_difference`, regime `contrast:...`); its conditioning is
    /// replaced by the native-fit marker. `feature_quantities` are the declared
    /// modifiers in the order the fit received them; their supports are the fitted ranges.
    ///
    /// # Errors
    /// A feature count that differs from the fit, or an export the validator refuses.
    fn to_compact_export(
        &self,
        response: &ScientificQuantityWire,
        feature_quantities: &[ScientificQuantityWire],
    ) -> Result<CompactExport, ExportRefusal> {
        if feature_quantities.len() != self.feature_ranges.len() {
            return Err(ExportRefusal {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "compact_export.dimension_mismatch",
                subject: "feature_quantities".to_owned(),
            });
        }
        let mut response = response.clone();
        response.conditioning = native_fit_conditioning(
            NativeFitSize { rows: self.rows as u64, width: self.width as u64 },
            &self.digest(),
        );
        let mut terms = vec![TermSpec::Intercept];
        let mut inputs = Vec::new();
        for (quantity, (lo, hi)) in feature_quantities.iter().zip(&self.feature_ranges) {
            terms.push(TermSpec::Linear { quantity: quantity.variable_id.clone() });
            inputs.push(ExportInput {
                quantity: quantity.clone(),
                support: InputSupport::Range { lo: *lo, hi: *hi },
            });
        }
        CompactExport::build(ExportSpec {
            response,
            inputs,
            terms,
            coefficients: self.coefficients.clone(),
            covariance: self.covariance.clone(),
            mask: Vec::new(),
        })
    }
}

/// The calibration basis of a native cohort-mean interval at `rows` fitted rows.
///
/// One definition serves the measurement harness and the measured facade, so a record
/// can only describe the construction the facade reports.
#[must_use]
pub fn calibration_basis(rows: u64, level: f64) -> CalibrationBasis {
    basis_with_structure("fixed", rows, level)
}

/// The calibration basis of a named-boundary probe of the same construction. Its
/// structure differs from [`calibration_basis`], so a boundary design never shares a
/// calibration key with the gated design and no facade execution can resolve to it.
#[must_use]
pub fn boundary_calibration_basis(boundary: &str, rows: u64, level: f64) -> CalibrationBasis {
    basis_with_structure(&format!("boundary.{boundary}"), rows, level)
}

fn basis_with_structure(structure: &str, rows: u64, level: f64) -> CalibrationBasis {
    let a = |text: &str| Arc::<str>::from(text);
    CalibrationBasis::new(
        [
            a("CohortMeanConditionalEffect"),
            a("NotApplicable"),
            a(structure),
            a("declared_cohort_linear_effect"),
            a("Frequentist"),
            a(ESTIMATOR_ID),
            a(INTERVAL_METHOD),
            a("hc3_sandwich"),
            a("independent_rows"),
            a(""),
            a(FUNCTIONAL_ID),
        ],
        level,
        a("adjustment"),
        rows,
        None,
        None,
        0.0,
    )
}

// ---------------------------------------------------------------------------
// Native-fit marker and the cohort-mean interval
// ---------------------------------------------------------------------------

/// Conditioning variable carrying the producer's digest of the fit output.
pub const FIT_DIGEST_VARIABLE: &str = "fit.digest";
/// Conditioning variable naming the fitting procedure of a natively fitted export.
pub const FIT_PROCEDURE_VARIABLE: &str = "fit.procedure";
/// Conditioning variable carrying the retained row count.
pub const FIT_ROWS_VARIABLE: &str = "fit.n";
/// Conditioning variable carrying the retained design width.
pub const FIT_WIDTH_VARIABLE: &str = "fit.p";
/// The only fitting procedure whose covariance supports an interval.
pub const FIT_PROCEDURE_ID: &str = "ols_hc3_treatment_by_basis";
/// Refusal detail of a declared (caller-built) export asked for an interval.
pub const DETAIL_REQUIRES_NATIVE_FIT: &str = "cohort_decision.interval_requires_native_fit";
/// Refusal detail of a shifted or otherwise point-only source asked for an interval.
pub const DETAIL_NOT_LICENSED: &str = "cohort_decision.interval_not_licensed";
/// Refusal detail of an invalid nominal level.
pub const DETAIL_LEVEL_INVALID: &str = "cohort_decision.interval_level_invalid";
/// Scope of the issued interval.
pub const INTERVAL_SCOPE: &str = "interval for the declared-cohort mean of the true conditional effect under a correctly specified linear effect basis, ignorable or randomized treatment and independent rows, for a cohort of the fitted population; excludes support selection and a shifted target population";
/// Reference size of a replayed fit's rows. Not enforced: the normal-equation sizes follow the
/// rows and modifiers given.
pub const MAX_FIT_ROWS: usize = 1_000_000;
/// Reference size of a replayed fit's modifier count. Not enforced.
pub const MAX_FIT_FEATURES: usize = 32;

/// The retained size of a native fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeFitSize {
    /// Rows the model was fitted on.
    pub rows: u64,
    /// Columns of the fitted design (every coefficient of the outcome regression).
    pub width: u64,
}

fn native_fit_conditioning(size: NativeFitSize, digest: &str) -> Vec<QuantityConditionWire> {
    let item = |name: &str, value: String| QuantityConditionWire {
        variable_id: name.to_owned(),
        value_id: value,
    };
    vec![
        item(FIT_DIGEST_VARIABLE, digest.to_owned()),
        item(FIT_ROWS_VARIABLE, size.rows.to_string()),
        item(FIT_WIDTH_VARIABLE, size.width.to_string()),
        item(FIT_PROCEDURE_VARIABLE, FIT_PROCEDURE_ID.to_owned()),
    ]
}

/// The declared inputs a native fit was computed from: what a consumer replays.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeFitRows {
    /// Outcome, one per row.
    pub outcome: Vec<f64>,
    /// Treatment, one per row.
    pub treatment: Vec<f64>,
    /// One column of modifier values per declared modifier id.
    pub features: Vec<Vec<f64>>,
    /// Modifier variable ids, aligned with `features`.
    pub feature_ids: Vec<String>,
}

/// A natively fitted linear effect and its export. It can only be constructed by
/// running the fit (or replaying it from the rows), so holding one is the proof that the
/// coefficients, covariance, `n` and `p` were produced here and not declared.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeCohortFit {
    fit: CohortLinearFit,
    export: CompactExport,
    rows: NativeFitRows,
}

impl NativeCohortFit {
    /// Fit and export.
    ///
    /// `feature_quantities` are the declared modifiers in the column order of
    /// `features`; `response` must declare a conditional treatment contrast.
    ///
    /// # Errors
    /// A refused fit (see [`fit_cohort_linear_effect`]) or an export the validator refuses.
    pub fn fit(
        outcome: &[f64],
        treatment: &[f64],
        features: &[&[f64]],
        response: &ScientificQuantityWire,
        feature_quantities: &[ScientificQuantityWire],
    ) -> Result<Self, EstimationError> {
        let fit = fit_cohort_linear_effect(outcome, treatment, features)?;
        let export = fit
            .to_compact_export(response, feature_quantities)
            .map_err(|_| unsupported("cohort linear effect: the export was refused"))?;
        let rows = NativeFitRows {
            outcome: outcome.to_vec(),
            treatment: treatment.to_vec(),
            features: features.iter().map(|c| c.to_vec()).collect(),
            feature_ids: feature_quantities.iter().map(|q| q.variable_id.clone()).collect(),
        };
        Ok(Self { fit, export, rows })
    }

    /// Replay a retained fit: refit from `rows` under `response` and the modifier
    /// declarations of `like`, and require the result to equal `like` exactly.
    ///
    /// # Errors
    /// Rows that do not reproduce the export's coefficients, covariance, marker or
    /// identity (a declared or hand-edited export), or any refusal of the fit.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the rows are the retained fit and are consumed by their callers"
    )]
    pub fn replay(rows: NativeFitRows, like: &CompactExport) -> Result<Self, EstimationError> {
        let body = like.body();
        let mut quantities = Vec::with_capacity(rows.feature_ids.len());
        for id in &rows.feature_ids {
            let input = body
                .inputs
                .iter()
                .find(|input| &input.quantity.variable_id == id)
                .ok_or_else(|| unsupported("cohort linear effect: unknown modifier id"))?;
            quantities.push(input.quantity.clone());
        }
        if quantities.len() != body.inputs.len() {
            return Err(unsupported("cohort linear effect: modifier ids differ from the export"));
        }
        let columns: Vec<&[f64]> = rows.features.iter().map(Vec::as_slice).collect();
        let replayed =
            Self::fit(&rows.outcome, &rows.treatment, &columns, &body.response, &quantities)?;
        if replayed.export.identity() != like.identity() || replayed.export.body() != like.body() {
            return Err(unsupported(
                "cohort linear effect: the retained rows do not reproduce the export",
            ));
        }
        Ok(replayed)
    }

    /// The export of the fit.
    #[must_use]
    pub const fn export(&self) -> &CompactExport {
        &self.export
    }

    /// The fit.
    #[must_use]
    pub const fn linear_fit(&self) -> &CohortLinearFit {
        &self.fit
    }

    /// The rows the fit was computed from.
    #[must_use]
    pub const fn rows(&self) -> &NativeFitRows {
        &self.rows
    }

    /// The retained size.
    #[must_use]
    pub const fn size(&self) -> NativeFitSize {
        NativeFitSize { rows: self.fit.rows as u64, width: self.fit.width as u64 }
    }
}

/// The refusal of an interval for an export that is not a [`NativeCohortFit`]: its
/// coefficients and covariance were declared, not produced here.
#[must_use]
pub fn declared_export_refusal() -> CohortIntervalRefusal {
    refusal(
        DETAIL_REQUIRES_NATIVE_FIT,
        "compact_export",
        "the coefficients and covariance were declared by the caller; fit the effect \
         natively (outcome on treatment-by-basis with a sandwich covariance) to receive \
         an interval, or report the point value with its uncalibrated diagnostic",
    )
}

/// A typed refusal of a cohort-mean interval request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CohortIntervalRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Stable `cohort_decision.*` detail.
    pub detail: &'static str,
    /// What the refusal is about.
    pub subject: String,
    /// The declaration or evidence that would license the request.
    pub remedy: &'static str,
}

impl std::fmt::Display for CohortIntervalRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} ({})", self.code, self.detail, self.subject)
    }
}

impl std::error::Error for CohortIntervalRefusal {}

/// A two-sided Student-t interval for the weighted cohort-mean contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct CohortMeanInterval {
    /// The weighted cohort-mean contrast.
    pub point: f64,
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Nominal two-sided level.
    pub level: f64,
    /// `sqrt(phibar' V phibar)` with the retained HC3 covariance.
    pub standard_error: f64,
    /// Student-t critical value used.
    pub critical_value: f64,
    /// `n - p`.
    pub degrees_of_freedom: u64,
    /// Retained fit size.
    pub fit: NativeFitSize,
    /// Cohort row count the weighted mean was taken over.
    pub cohort_rows: u64,
}

fn refusal(detail: &'static str, subject: &str, remedy: &'static str) -> CohortIntervalRefusal {
    CohortIntervalRefusal {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        detail,
        subject: subject.to_owned(),
        remedy,
    }
}

/// The interval of `source`, which must be the unshifted decision source built from
/// exactly the export of `native`.
///
/// The standard error is the source's own recomputed `sqrt(phibar' V phibar)`; this
/// function adds only the retained `n - p` degrees of freedom and the critical value.
///
/// # Errors
/// A level outside `(0, 1)`; a source not built from `export`; a shifted source; an
/// export that was declared by the caller rather than fitted natively
/// ([`DETAIL_REQUIRES_NATIVE_FIT`]).
pub fn native_cohort_interval(
    native: &NativeCohortFit,
    source: &CohortDecisionSource,
    level: f64,
) -> Result<CohortMeanInterval, CohortIntervalRefusal> {
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(refusal(
            DETAIL_LEVEL_INVALID,
            "level",
            "request a two-sided level strictly between 0 and 1",
        ));
    }
    let export = native.export();
    let fit = native.size();
    if source.shift().is_some() {
        return Err(refusal(
            DETAIL_NOT_LICENSED,
            "declared_covariate_shift",
            "a shifted cohort is point-only; evaluate a cohort of the model's own \
             population to receive an interval",
        ));
    }
    let bound = RetainedEffectModel::from_compact_export(export)
        .is_ok_and(|model| model.identity() == source.model().identity());
    let Some(diagnostic) = source.uncalibrated_standard_error() else {
        return Err(refusal(
            DETAIL_NOT_LICENSED,
            "model",
            "the source retains no coefficient covariance",
        ));
    };
    if !bound || !source.model().is_linear_compact() {
        return Err(refusal(
            DETAIL_NOT_LICENSED,
            "source",
            "build the decision source from exactly the exported native fit",
        ));
    }
    let df = (fit.rows - fit.width) as f64;
    let critical_value = student_t_ppf(0.5 + level / 2.0, df);
    let half = critical_value * diagnostic.value;
    let point = source.mean_contrast();
    if !half.is_finite() {
        return Err(refusal(
            DETAIL_NOT_LICENSED,
            "interval",
            "the interval is not finite; refit with more rows",
        ));
    }
    Ok(CohortMeanInterval {
        point,
        lower: point - half,
        upper: point + half,
        level,
        standard_error: diagnostic.value,
        critical_value,
        degrees_of_freedom: fit.rows - fit.width,
        fit,
        cohort_rows: source.cohort().rows() as u64,
    })
}

/// Build the decision source of `export` over `cohort` and issue its interval.
///
/// # Errors
/// Any build refusal of the source, as [`CohortIntervalRefusal`] with the source's own
/// code and detail, and the refusals of [`native_cohort_interval`].
pub fn native_cohort_interval_for(
    native: &NativeCohortFit,
    cohort: &CohortBinding,
    outcome_units: &str,
    level: f64,
    ctx: &ExecutionContext,
) -> Result<(CohortDecisionSource, CohortMeanInterval), CohortIntervalRefusal> {
    let model = RetainedEffectModel::from_compact_export(native.export()).map_err(|r| {
        CohortIntervalRefusal {
            code: r.code,
            detail: r.detail,
            subject: r.subject.clone(),
            remedy: r.remedy,
        }
    })?;
    let source =
        CohortDecisionSource::build(&model, cohort, outcome_units, &CohortOptions::default(), ctx)
            .map_err(|r| CohortIntervalRefusal {
                code: r.code,
                detail: r.detail,
                subject: r.subject.clone(),
                remedy: r.remedy,
            })?;
    let interval = native_cohort_interval(native, &source, level)?;
    Ok((source, interval))
}
