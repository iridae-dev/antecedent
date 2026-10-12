//! Assumption records and tempering diagnostics.

use std::sync::Arc;

use crate::serial_dependence::{
    DEPENDENCE_ASSUMPTION_ID, tempering_capped_from_notes, tempering_inestimable_from_notes,
    tempering_kappa_from_notes,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, ParametricAssumption,
};

use super::block::ResponseBlockLength;

pub(super) fn with_pointwise_homoskedastic_ols_assumption(
    mut assumptions: AssumptionSet,
) -> AssumptionSet {
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("ols.linear_additive.gcomp"),
            description: Arc::from(
                "Temporal response levels use linear additive g-computation on each unfolded horizon. The numerical surface is model-dependent when the conditional outcome response is nonlinear or contains treatment-covariate interactions.",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("ols.homoskedastic.pointwise"),
            description: Arc::from(
                "With zero bootstrap replicates no band is published: the delta-method band \
                 from the homoskedastic OLS coefficient covariance would treat lag-aligned rows \
                 as independent draws, and neighbouring temporal rows share lagged treatment, \
                 confounder or residual terms. Request bootstrap replicates for the \
                 dependence-preserving joint circular-block pointwise and simultaneous bands.",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions
}

/// Construction text for the joint circular-block surface bootstrap.
pub(super) const BLOCK_BOOTSTRAP_CONSTRUCTION: &str = "max-studentized deviation of the joint circular-block \
     bootstrap replicates of the whole dose × horizon surface around the full-sample estimate; \
     each replicate resamples time-aligned blocks of lag-aligned rows (block length: support \
     diagnostic response.temporal.block_length), refits every horizon and recomputes the \
     covariate averages; replicate deviations carry the circular-Bartlett fixed-b factor for \
     b = block/rows, the HC1 factor sqrt(rows/(rows − p)) and each cell's kernel-bias factor \
     (response.temporal.kernel_bias_factor)";

/// What one surface-bootstrap replicate refits, for [`disclose_response_block_bootstrap`].
pub(super) const SURFACE_REFIT: &str = "one replicate refits every horizon on the same time-aligned \
     blocks and recomputes covariate averages";

/// Assumption id of the temporal response circular-block band.
pub(super) const BLOCK_BOOTSTRAP_ASSUMPTION_ID: &str = "temporal_response.block_bootstrap";

pub(super) fn block_bootstrap_assumption(
    block: ResponseBlockLength,
    factor: f64,
    kernel_factor: f64,
    refit: &str,
) -> AssumptionRecord {
    let ResponseBlockLength { length, rule, testing, rows } = block;
    AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from(BLOCK_BOOTSTRAP_ASSUMPTION_ID),
            description: Arc::from(format!(
                "Pointwise SEs are the SD of the level over a joint circular-block bootstrap of \
                 lag-aligned outcome-time tuples (n = {rows} rows); {refit}, so the band targets \
                 the population level under serially dependent rows whose dependence decays \
                 within a block. Block length {length} = max(rule {rule} = max(unfolded span, \
                 ceil(sqrt(n))), min(testing {testing} = ceil(b_PW·n^(1/6)), n/3)), with b_PW the \
                 largest Politis-White length over the level's estimating scores (every fitted \
                 regression's normal-equation scores and the centered covariate columns whose \
                 averages the level reads). Replicate deviations are scaled by {factor:.4}: the circular-Bartlett \
                 fixed-b critical-value ratio for b = block/rows (the variance is estimated from \
                 few blocks) times the HC1 factor sqrt(rows/(rows − p)); and by each cell's \
                 kernel-bias factor (largest {kernel_factor:.4}; support diagnostic \
                 response.temporal.kernel_bias_factor): 1/sqrt(f) with f the share of the \
                 long-run variance of the autoregression fitted to that cell's influence \
                 (Kendall-corrected AR(1), or the BIC-selected AR(q <= 4) when larger) that the \
                 block's Bartlett kernel keeps, the O(1/block) kernel truncation the fixed-b \
                 ratio does not repair. The factor corrects only dependence the autoregression \
                 fits; a persistent component that is a small share of the influence is not \
                 seen at short n. The pointwise band is \
                 mean ± 1.96·SE of the scaled replicates. The simultaneous band (support \
                 diagnostics response.simultaneous_band.*) is the max-studentized deviation over \
                 the whole grid from the same scaled replicates; the fixed-b ratio is derived for \
                 one pointwise 95% interval, so carrying it into the sup-t band is a heuristic \
                 extension, supported by the calibration rather than by theory."
            )),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    }
}

/// Support diagnostic carrying the per-horizon tempering factor `κ̂_h` of
/// `response.temporal.bayesian` (one value per requested horizon, in order).
pub const TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC: &str = "response.temporal_bayesian.tempering";

/// Warning code: at least one horizon's `κ̂` could not be estimated (`n` below
/// `max(8, p+2)`); that horizon's posterior is the untempered iid fit.
pub(crate) const TEMPORAL_BAYESIAN_TEMPERING_INESTIMABLE: &str =
    "response.temporal_bayesian.tempering_inestimable";

/// Warning code: at least one horizon's `κ̂` hit the `n/(p+2)` cap; the
/// correction there is incomplete and the posterior is still too narrow.
pub(crate) const TEMPORAL_BAYESIAN_TEMPERING_CAPPED: &str =
    "response.temporal_bayesian.tempering_capped";

pub(super) fn is_tempering_record(record: &AssumptionRecord) -> bool {
    matches!(
        &record.assumption,
        Assumption::ParametricRestriction(p) if p.id.as_ref() == DEPENDENCE_ASSUMPTION_ID
    )
}

/// One horizon's likelihood-tempering result, read off the posterior inference notes
/// (see [`crate::serial_dependence::TemperingFactor`]) so a capped or inestimable
/// factor cannot be dropped on the way to the response.
#[derive(Clone, Copy, Debug)]
pub(super) struct HorizonTempering {
    pub(super) kappa: f64,
    pub(super) capped: bool,
    pub(super) inestimable: bool,
}

impl HorizonTempering {
    pub(super) fn from_notes(notes: &[Arc<str>]) -> Self {
        Self {
            kappa: tempering_kappa_from_notes(notes).unwrap_or(1.0),
            capped: tempering_capped_from_notes(notes),
            inestimable: tempering_inestimable_from_notes(notes),
        }
    }

    /// `h=<horizon>: <kappa>` with a trailing ` capped` / ` inestimable` tag so the
    /// per-horizon summary cannot silently drop either disclosure.
    fn label(&self, horizon: u32) -> String {
        let flag = if self.inestimable {
            " inestimable"
        } else if self.capped {
            " capped"
        } else {
            ""
        };
        format!("h={horizon}: {:.4}{flag}", self.kappa)
    }
}

/// One serial-dependence record for the whole surface (each horizon's fit records its own
/// factor; only the first horizon's assumptions were ever forwarded).
pub(super) fn temporal_bayesian_tempering_assumption(
    horizons: &[u32],
    tempering: &[HorizonTempering],
) -> AssumptionRecord {
    let per_horizon =
        horizons.iter().zip(tempering).map(|(&h, t)| t.label(h)).collect::<Vec<_>>().join(", ");
    let any_capped = tempering.iter().any(|t| t.capped);
    let any_inestimable = tempering.iter().any(|t| t.inestimable);
    let caveat = match (any_inestimable, any_capped) {
        (true, true) => {
            "; a horizon marked inestimable keeps its untempered iid posterior (likely too \
             narrow), and a horizon marked capped has an incomplete correction (still too \
             narrow)"
        }
        (true, false) => {
            "; a horizon marked inestimable keeps its untempered iid posterior, which is \
             likely too narrow"
        }
        (false, true) => {
            "; a horizon marked capped has an incomplete correction, so its posterior is \
             still too narrow"
        }
        (false, false) => "",
    };
    AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from(DEPENDENCE_ASSUMPTION_ID),
            description: Arc::from(format!(
                "generalized (power) posterior with a serial-dependence correction at every \
                 horizon: each horizon's Gaussian likelihood on time-ordered lag-aligned rows is \
                 tempered by 1/kappa_h, kappa_h = the variance ratio of that horizon's driving \
                 grid-cell-level combination under a REML-fitted autoregressive residual (BIC \
                 order <= 4) given the design, times the residual-scale factor n/(n - tr(HR)) \
                 for the rows the projection removes, times exp(tau^2/2) for the delta-method \
                 spread of log kappa_h; bounded by three times the autoregressive-prewhitened \
                 (AR(1), plus a BIC-selected AR(q <= 4)) Newey-West long-run-variance ratio of \
                 the driving score, scaled by its squared fixed-b factor; floored at 1 and capped \
                 at n/(p+2) ({per_horizon}){caveat}. At h >= 2 the unfolded regression omits \
                 intermediate treatments and innovations, so its residuals are MA(h-1) whenever \
                 the outcome or treatment is persistent; the prior keeps full weight; \
                 heteroskedasticity and mean misspecification are not corrected"
            )),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("response.temporal.bayesian"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    }
}
