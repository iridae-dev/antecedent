//! Sharp bounds on the average effect of a binary treatment under a binary instrument, from data.
//!
//! The observed law of the instrument `Z`, the treatment `D` and the outcome `Y` is read from
//! three 0/1 columns, each instrument arm's conditional cell probabilities are formed (optionally
//! weighted by a sampling-weight column), and the Balke-Pearl bounds are solved on that law
//! ([`antecedent_identify::binary_iv_ate_bounds`]). The answer is an identified set, never a point:
//! the interval `[lower, upper]` is every average effect the observed law and the instrument
//! premises leave open.
//!
//! Sampling uncertainty is reported about the endpoints. Frequentist: the endpoints are refit on
//! row-bootstrap resamples (rows are independent subjects) and an Imbens-Manski interval, which
//! covers the true effect, is formed from the endpoint standard deviations and the standard
//! deviation of the width. Bayesian: the Bayesian bootstrap reweights the rows with exponential
//! weights and the same Imbens-Manski tail rule is read off the posterior draws of the endpoints.
//! A resample or a draw whose law violates the instrumental inequality has no bounds; it is
//! counted and left out, never replaced. No coverage record measures either interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, IvBoundsQuery, StreamDomain, Value, VariableId};
use antecedent_data::{
    ColumnView, DiscreteColumn, ResamplingPlan, TableView, TabularData,
    fill_resample_index_replicate,
};
use antecedent_identify::{BinaryIvLaw, binary_iv_ate_bounds};
use antecedent_kernels::{bayesian_bootstrap_weights, sample_std};

use crate::error::EstimationError;
use crate::identified_set::{
    IdentifiedSetInterval, imbens_manski_posterior_draws, imbens_manski_shared_replicates,
};
use crate::util::SamplingDeclaration;

/// Stream index of the row-bootstrap resamples.
const BOOTSTRAP_STREAM: u64 = 0x1B0D_5001;
/// Stream index of the Bayesian-bootstrap weight draws.
const POSTERIOR_STREAM: u64 = 0x1B0D_5002;
/// Fewest replicates or draws an interval is formed from.
const MIN_REPLICATES: usize = 20;

/// How the sampling interval for the endpoints was formed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IvBoundsIntervalMethod {
    /// Frequentist Imbens-Manski interval from row-bootstrap endpoint replicates.
    ImbensManskiRowBootstrap,
    /// Product-posterior quantiles of the endpoint draws at the Imbens-Manski tail probability.
    PosteriorEndpointQuantiles,
}

impl IvBoundsIntervalMethod {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ImbensManskiRowBootstrap => "imbens_manski_row_bootstrap",
            Self::PosteriorEndpointQuantiles => "posterior_endpoint_quantiles",
        }
    }
}

/// Interval that covers the true effect (frequentist) or holds it with the stated posterior
/// probability on each side (Bayesian), allowing for the identified set's width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IvBoundsInterval {
    /// Nominal level.
    pub level: f64,
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Sampling standard deviation of the lower bound.
    pub lower_se: f64,
    /// Sampling standard deviation of the upper bound.
    pub upper_se: f64,
    /// Imbens-Manski critical value.
    pub critical_value: f64,
    /// Whether the estimated width passed the moment-selection threshold.
    pub width_retained: bool,
    /// Resamples or draws behind the interval.
    pub replicates: usize,
    /// Resamples or draws left out because their law violates the instrumental inequality.
    pub infeasible: usize,
    /// Construction.
    pub method: IvBoundsIntervalMethod,
}

/// Posterior summary of the endpoints under the Bayesian bootstrap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IvBoundsPosterior {
    /// Draws requested.
    pub n_draws: usize,
    /// Draws whose law violates the instrumental inequality and carry no bounds.
    pub infeasible_draws: usize,
    /// Posterior mean of the lower bound.
    pub lower_mean: f64,
    /// Posterior mean of the upper bound.
    pub upper_mean: f64,
    /// Posterior standard deviation of the lower bound.
    pub lower_sd: f64,
    /// Posterior standard deviation of the upper bound.
    pub upper_sd: f64,
}

/// The identified set of average effects and the evidence behind it.
#[derive(Clone, Debug, PartialEq)]
pub struct IvBoundsEstimate {
    /// Lower bound of `E[Y(1)] - E[Y(0)]` on the observed law.
    pub lower: f64,
    /// Upper bound of `E[Y(1)] - E[Y(0)]` on the observed law.
    pub upper: f64,
    /// Observed (weighted) law `P(Y=y, D=d | Z=z)`; cell order `(0,0), (1,0), (0,1), (1,1)`.
    pub law: BinaryIvLaw,
    /// Weighted row mass in each instrument arm.
    pub arm_weights: [f64; 2],
    /// Complete rows behind the law.
    pub rows: usize,
    /// Rows left out for a missing cell or weight.
    pub rows_dropped: usize,
    /// Interval for the true effect, when replicates or draws were requested.
    pub interval: Option<IvBoundsInterval>,
    /// Posterior summary, under Bayesian inference.
    pub posterior: Option<IvBoundsPosterior>,
}

/// The three 0/1 columns and the weights of the complete rows.
#[derive(Clone, Debug)]
pub struct PreparedIvBounds {
    instrument: Vec<u8>,
    treatment: Vec<u8>,
    outcome: Vec<u8>,
    weight: Vec<f64>,
    rows_total: usize,
}

impl PreparedIvBounds {
    /// Complete rows behind the law.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.weight.len()
    }

    /// Rows left out for a missing cell or weight.
    #[must_use]
    pub fn rows_dropped(&self) -> usize {
        self.rows_total - self.weight.len()
    }

    /// Weighted cell masses by instrument arm over `multiplicity`-weighted rows.
    fn cells(&self, mut mass: impl FnMut(usize) -> f64) -> [[f64; 4]; 2] {
        let mut cells = [[0.0; 4]; 2];
        for row in 0..self.weight.len() {
            let weight = mass(row);
            if weight == 0.0 {
                continue;
            }
            let z = usize::from(self.instrument[row]);
            let cell = usize::from(self.treatment[row]) * 2 + usize::from(self.outcome[row]);
            cells[z][cell] += weight;
        }
        cells
    }
}

/// Balke-Pearl bounds from the observed law of a binary instrument, treatment and outcome.
#[derive(Clone, Debug)]
pub struct IvBoundsEstimator {
    /// Row-bootstrap replicates for the frequentist interval (0 = point bounds only).
    pub bootstrap_replicates: u32,
    /// Two-sided level of the interval.
    pub level: f64,
    /// The sampling design the caller declared; a draw the design does not allow is refused.
    pub sampling_design: SamplingDeclaration,
}

impl Default for IvBoundsEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl IvBoundsEstimator {
    /// Point bounds, a 95% interval level and no declared design.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bootstrap_replicates: 0,
            level: 0.95,
            sampling_design: SamplingDeclaration::Undeclared,
        }
    }

    /// Set the number of row-bootstrap replicates (0 = point bounds only).
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Declare the sampling design of the rows.
    #[must_use]
    pub const fn with_sampling_design(mut self, design: SamplingDeclaration) -> Self {
        self.sampling_design = design;
        self
    }

    /// Read the columns of `query` from `data`.
    ///
    /// # Errors
    ///
    /// A column that is missing or is not coded exactly 0 and 1, a weight that is negative or
    /// not finite, or no complete row.
    pub fn prepare(
        &self,
        data: &TabularData,
        query: &IvBoundsQuery,
    ) -> Result<PreparedIvBounds, EstimationError> {
        query.validate().map_err(|error| EstimationError::data_msg(error.to_string()))?;
        let instrument = binary_column(data, query.instrument, "instrument")?;
        let treatment = binary_column(data, query.treatment, "treatment")?;
        let outcome = binary_column(data, query.outcome, "outcome")?;
        let weights = match query.weight {
            Some(id) => Some(weight_column(data, id)?),
            None => None,
        };
        let rows_total = data.row_count();
        let mut prepared = PreparedIvBounds {
            instrument: Vec::new(),
            treatment: Vec::new(),
            outcome: Vec::new(),
            weight: Vec::new(),
            rows_total,
        };
        for row in 0..rows_total {
            let weight = match &weights {
                Some(column) => column[row],
                None => Some(1.0),
            };
            if let (Some(z), Some(d), Some(y), Some(w)) =
                (instrument[row], treatment[row], outcome[row], weight)
            {
                prepared.instrument.push(z);
                prepared.treatment.push(d);
                prepared.outcome.push(y);
                prepared.weight.push(w);
            }
        }
        if prepared.weight.is_empty() {
            return Err(EstimationError::data_msg(
                "no complete row holds the instrument, the treatment, the outcome and the weight",
            ));
        }
        Ok(prepared)
    }

    /// The bounds on the observed law, with a row-bootstrap Imbens-Manski interval when
    /// replicates were requested.
    ///
    /// # Errors
    ///
    /// An instrument arm with no mass, an observed law that violates the instrumental
    /// inequality, or a declared sampling design that refuses a row-level draw.
    pub fn estimate(
        &self,
        prepared: &PreparedIvBounds,
        ctx: &ExecutionContext,
    ) -> Result<IvBoundsEstimate, EstimationError> {
        let (law, arms) = observed_law(&prepared.cells(|row| prepared.weight[row]))?;
        let bounds = bounds_of(law)?;
        let interval = if self.bootstrap_replicates == 0 {
            None
        } else {
            self.refuse_row_draw()?;
            Some(self.bootstrap_interval(prepared, bounds, ctx)?)
        };
        Ok(estimate_of(prepared, law, arms, bounds, interval, None))
    }

    /// The same bounds with the Bayesian bootstrap of the endpoints.
    ///
    /// # Errors
    ///
    /// As [`Self::estimate`], and fewer than 20 draws, or too few draws with feasible bounds.
    pub fn estimate_bayesian(
        &self,
        prepared: &PreparedIvBounds,
        n_draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<IvBoundsEstimate, EstimationError> {
        let (law, arms) = observed_law(&prepared.cells(|row| prepared.weight[row]))?;
        let bounds = bounds_of(law)?;
        self.refuse_row_draw()?;
        if n_draws < MIN_REPLICATES {
            return Err(EstimationError::unsupported(
                "the Bayesian bootstrap of the IV bounds needs at least 20 draws to form an interval",
            ));
        }
        let _admitted = crate::util::admit_array::<f64>(ctx, n_draws, 2)?;
        let mut lower = Vec::with_capacity(n_draws);
        let mut upper = Vec::with_capacity(n_draws);
        let mut infeasible = 0_usize;
        let rows = prepared.weight.len();
        for draw in 0..n_draws {
            if ctx.cancellation.is_cancelled() {
                return Err(EstimationError::unsupported("IV bounds posterior cancelled"));
            }
            let mut rng = ctx.rng.stream_for(
                StreamDomain::Estimate,
                antecedent_core::derive_stream(POSTERIOR_STREAM, draw as u64),
            );
            let exponential = bayesian_bootstrap_weights(&mut rng, rows, false);
            let cells = prepared.cells(|row| prepared.weight[row] * exponential[row]);
            match observed_law(&cells).and_then(|(law, _)| bounds_of(law)) {
                Ok((low, high)) => {
                    lower.push(low);
                    upper.push(high);
                }
                Err(_) => infeasible += 1,
            }
        }
        if lower.len() < MIN_REPLICATES {
            return Err(EstimationError::unsupported(
                "too few Bayesian-bootstrap draws have an observed law that satisfies the \
                 instrumental inequality; the data are close to contradicting the instrument",
            ));
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        let posterior = IvBoundsPosterior {
            n_draws,
            infeasible_draws: infeasible,
            lower_mean: mean(&lower),
            upper_mean: mean(&upper),
            lower_sd: sample_std(&lower),
            upper_sd: sample_std(&upper),
        };
        let posterior_means = [posterior.lower_mean, posterior.upper_mean];
        let interval = imbens_manski_posterior_draws(
            &posterior_means,
            &[lower.as_slice(), upper.as_slice()],
            rows,
            self.level,
        )
        .map(|found| {
            interval_of(found, IvBoundsIntervalMethod::PosteriorEndpointQuantiles, infeasible)
        });
        Ok(estimate_of(prepared, law, arms, bounds, interval, Some(posterior)))
    }

    /// A declared design other than independent subjects refuses a row-level draw.
    fn refuse_row_draw(&self) -> Result<(), EstimationError> {
        use antecedent_core::DeclaredSamplingDesign as Design;
        match self.sampling_design {
            SamplingDeclaration::Undeclared
            | SamplingDeclaration::Declared(Design::IndependentSubjectTrajectories) => Ok(()),
            SamplingDeclaration::Declared(_) => Err(EstimationError::unsupported(
                "the IV bounds table carries one row per subject and no cluster or series labels, \
                 so a row-level draw is allowed only under the independent-subject design; next \
                 attempt: declare independent subjects, or omit the bootstrap for the point bounds",
            )),
        }
    }

    fn bootstrap_interval(
        &self,
        prepared: &PreparedIvBounds,
        bounds: (f64, f64),
        ctx: &ExecutionContext,
    ) -> Result<IvBoundsInterval, EstimationError> {
        let rows = prepared.weight.len();
        let replicates = self.bootstrap_replicates as usize;
        let _admitted = crate::util::admit_array::<f64>(ctx, replicates, 2)?;
        let mut draws: Vec<Vec<f64>> = Vec::with_capacity(replicates);
        let mut infeasible = 0_usize;
        let mut drawn = Vec::with_capacity(rows);
        for replicate in 0..replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(EstimationError::unsupported("IV bounds bootstrap cancelled"));
            }
            fill_resample_index_replicate(
                ResamplingPlan::IidBootstrap,
                rows,
                None,
                ctx,
                BOOTSTRAP_STREAM,
                replicate as u64,
                &mut drawn,
            )?;
            let mut multiplicity = vec![0.0_f64; rows];
            for id in &drawn {
                let row = usize::try_from(id.get()).map_err(|_| {
                    EstimationError::unsupported("IV bounds bootstrap row index exceeds usize")
                })?;
                multiplicity[row] += 1.0;
            }
            let cells = prepared.cells(|row| prepared.weight[row] * multiplicity[row]);
            match observed_law(&cells).and_then(|(law, _)| bounds_of(law)) {
                Ok((low, high)) => draws.push(vec![low, high]),
                Err(_) => infeasible += 1,
            }
        }
        if draws.len() < MIN_REPLICATES {
            return Err(EstimationError::unsupported(
                "too few bootstrap resamples have an observed law that satisfies the instrumental \
                 inequality; the data are close to contradicting the instrument",
            ));
        }
        imbens_manski_shared_replicates(&[bounds.0, bounds.1], &draws, 1.0, rows, self.level)
            .map(|found| {
                interval_of(found, IvBoundsIntervalMethod::ImbensManskiRowBootstrap, infeasible)
            })
            .ok_or_else(|| {
                EstimationError::stats_msg(
                    "the bootstrap endpoint replicates have no finite spread",
                )
            })
    }
}

fn interval_of(
    found: IdentifiedSetInterval,
    method: IvBoundsIntervalMethod,
    infeasible: usize,
) -> IvBoundsInterval {
    IvBoundsInterval {
        level: found.level,
        lower: found.lower,
        upper: found.upper,
        lower_se: found.lower_se,
        upper_se: found.upper_se,
        critical_value: found.critical_value,
        width_retained: found.width_retained,
        replicates: found.replicates,
        infeasible,
        method,
    }
}

fn estimate_of(
    prepared: &PreparedIvBounds,
    law: BinaryIvLaw,
    arm_weights: [f64; 2],
    bounds: (f64, f64),
    interval: Option<IvBoundsInterval>,
    posterior: Option<IvBoundsPosterior>,
) -> IvBoundsEstimate {
    IvBoundsEstimate {
        lower: bounds.0,
        upper: bounds.1,
        law,
        arm_weights,
        rows: prepared.rows(),
        rows_dropped: prepared.rows_dropped(),
        interval,
        posterior,
    }
}

/// Conditional cell probabilities by instrument arm and the mass of each arm.
fn observed_law(cells: &[[f64; 4]; 2]) -> Result<(BinaryIvLaw, [f64; 2]), EstimationError> {
    let mut law = [[0.0; 4]; 2];
    let mut arms = [0.0; 2];
    for z in 0..2 {
        let total: f64 = cells[z].iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(EstimationError::unsupported(
                "the instrument takes a single value among the complete rows, so the bounds have \
                 no second arm to compare; next attempt: a table in which the instrument varies",
            ));
        }
        arms[z] = total;
        for cell in 0..4 {
            law[z][cell] = cells[z][cell] / total;
        }
    }
    Ok((BinaryIvLaw { cells: law }, arms))
}

fn bounds_of(law: BinaryIvLaw) -> Result<(f64, f64), EstimationError> {
    binary_iv_ate_bounds(law).map(|set| (set.lower, set.upper)).map_err(|error| match error {
        // The identification crate states why the law has no bounds as a static message: a
        // malformed cell, or the Balke-Pearl instrumental inequality it violates.
        antecedent_identify::IdentificationError::UnsupportedQuery { message }
            if message.contains("instrumental inequality") =>
        {
            EstimationError::unsupported(
                "the observed law violates the Balke-Pearl instrumental inequality: no \
                 distribution over the response types reproduces both instrument arms, so the \
                 instrument cannot be both independent of the potential outcomes and excluded \
                 from the outcome. Next attempt: re-examine the instrument's independence and \
                 exclusion, or declare a different instrument",
            )
        }
        antecedent_identify::IdentificationError::UnsupportedQuery { message } => {
            EstimationError::unsupported(message)
        }
        other => EstimationError::data_msg(other.to_string()),
    })
}

/// A column coded exactly 0 and 1; `None` is a missing cell.
fn binary_column(
    data: &TabularData,
    id: VariableId,
    role: &str,
) -> Result<Vec<Option<u8>>, EstimationError> {
    let column = data.discrete_column(id).map_err(EstimationError::from)?;
    let mut bit_of_level = Vec::with_capacity(column.levels.len());
    for level in &column.levels {
        let bit = match level {
            Value::Float64(value) if (value + 0.0).to_bits() == 0.0_f64.to_bits() => Some(0_u8),
            Value::Float64(value) if value.to_bits() == 1.0_f64.to_bits() => Some(1),
            Value::Int64(0) | Value::Bool(false) => Some(0),
            Value::Int64(1) | Value::Bool(true) => Some(1),
            _ => None,
        };
        let Some(bit) = bit else {
            return Err(EstimationError::data_msg(format!(
                "the IV bounds {role} column must be coded exactly 0 and 1"
            )));
        };
        bit_of_level.push(bit);
    }
    Ok(column
        .codes
        .iter()
        .map(|&code| (code != DiscreteColumn::MISSING).then(|| bit_of_level[code as usize]))
        .collect())
}

/// A sampling-weight column of finite non-negative weights; `None` is a missing cell.
fn weight_column(data: &TabularData, id: VariableId) -> Result<Vec<Option<f64>>, EstimationError> {
    let ColumnView::Float64(column) = data.column(id).map_err(EstimationError::from)? else {
        return Err(EstimationError::data_msg("the IV bounds weight column must be float64"));
    };
    let mut weights = Vec::with_capacity(column.values.len());
    for row in 0..column.values.len() {
        if !column.validity.is_valid(row) {
            weights.push(None);
            continue;
        }
        let value = column.values.as_slice()[row];
        if !(value.is_finite() && value >= 0.0) {
            return Err(EstimationError::data_msg(
                "the IV bounds weights must be finite and non-negative",
            ));
        }
        weights.push(Some(value));
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use antecedent_core::ExecutionContext;

    use super::*;

    /// One row per `(z, d, y)` cell, repeated `counts[z][d][y]` times, as a float table.
    fn table(counts: [[[usize; 2]; 2]; 2]) -> TabularData {
        let (mut z, mut d, mut y) = (Vec::new(), Vec::new(), Vec::new());
        for (zi, by_d) in counts.iter().enumerate() {
            for (di, by_y) in by_d.iter().enumerate() {
                for (yi, &cell) in by_y.iter().enumerate() {
                    for _ in 0..cell {
                        z.push(zi as f64);
                        d.push(di as f64);
                        y.push(yi as f64);
                    }
                }
            }
        }
        TabularData::from_f64_columns([("z", &z[..]), ("d", &d[..]), ("y", &y[..])]).unwrap()
    }

    fn query() -> IvBoundsQuery {
        IvBoundsQuery::new(
            VariableId::from_raw(2),
            VariableId::from_raw(1),
            VariableId::from_raw(0),
        )
    }

    #[test]
    fn the_bounds_are_the_sharp_set_of_the_observed_law() {
        let data = table([[[60, 40], [0, 0]], [[0, 0], [30, 70]]]);
        let estimator = IvBoundsEstimator::new();
        let prepared = estimator.prepare(&data, &query()).unwrap();
        let ctx = ExecutionContext::for_tests(41);
        let estimate = estimator.estimate(&prepared, &ctx).unwrap();
        // Perfect compliance identifies the effect: E[Y|Z=1] - E[Y|Z=0] = 0.70 - 0.40.
        assert!((estimate.lower - 0.3).abs() < 1e-9, "{}", estimate.lower);
        assert!((estimate.upper - 0.3).abs() < 1e-9, "{}", estimate.upper);
        assert_eq!(estimate.rows, 200);
        assert_eq!(estimate.rows_dropped, 0);
        assert!(estimate.arm_weights.iter().all(|w| (w - 100.0).abs() < 1e-9));
        assert!(estimate.interval.is_none() && estimate.posterior.is_none());
    }

    #[test]
    fn a_weight_column_stands_for_repeated_rows_and_a_column_not_coded_zero_one_is_refused() {
        // The perfect-compliance design as four weighted rows.
        let compact = TabularData::from_f64_columns([
            ("z", &[0.0, 0.0, 1.0, 1.0][..]),
            ("d", &[0.0, 0.0, 1.0, 1.0][..]),
            ("y", &[0.0, 1.0, 0.0, 1.0][..]),
            ("w", &[60.0, 40.0, 30.0, 70.0][..]),
        ])
        .unwrap();
        let weighted = query().with_weight(VariableId::from_raw(3));
        let estimator = IvBoundsEstimator::new();
        let ctx = ExecutionContext::for_tests(45);
        let prepared = estimator.prepare(&compact, &weighted).unwrap();
        let estimate = estimator.estimate(&prepared, &ctx).unwrap();
        assert!((estimate.lower - 0.3).abs() < 1e-9 && (estimate.upper - 0.3).abs() < 1e-9);
        assert!(estimate.arm_weights.iter().all(|w| (w - 100.0).abs() < 1e-9));
        assert_eq!(estimate.rows, 4);
        let bad = TabularData::from_f64_columns([
            ("z", &[0.0, 1.0, 2.0, 1.0][..]),
            ("d", &[0.0, 1.0, 0.0, 1.0][..]),
            ("y", &[0.0, 1.0, 0.0, 1.0][..]),
        ])
        .unwrap();
        let error = estimator.prepare(&bad, &query()).unwrap_err();
        assert!(error.to_string().contains("coded exactly 0 and 1"), "{error}");
        let one_arm = TabularData::from_f64_columns([
            ("z", &[1.0, 1.0, 1.0][..]),
            ("d", &[0.0, 1.0, 0.0][..]),
            ("y", &[0.0, 1.0, 0.0][..]),
        ])
        .unwrap();
        let prepared = estimator.prepare(&one_arm, &query()).unwrap();
        let error = estimator.estimate(&prepared, &ctx).unwrap_err();
        assert!(error.to_string().contains("next attempt"), "{error}");
    }

    #[test]
    fn a_law_that_contradicts_the_instrument_is_refused_naming_the_inequality() {
        // The instrument moves the outcome though it leaves the treatment untouched.
        let data = table([[[90, 10], [0, 0]], [[10, 90], [0, 0]]]);
        let estimator = IvBoundsEstimator::new();
        let prepared = estimator.prepare(&data, &query()).unwrap();
        let error = estimator.estimate(&prepared, &ExecutionContext::for_tests(42)).unwrap_err();
        assert!(error.to_string().contains("instrumental inequality"), "{error}");
    }

    #[test]
    fn the_bootstrap_interval_contains_the_bounds_and_is_wider_than_them() {
        // Partial compliance: the bounds are an interval, and the interval around them is wider.
        let data = table([[[50, 20], [10, 20]], [[20, 10], [20, 50]]]);
        let estimator = IvBoundsEstimator::new().with_bootstrap_replicates(60);
        let prepared = estimator.prepare(&data, &query()).unwrap();
        let ctx = ExecutionContext::for_tests(43);
        let estimate = estimator.estimate(&prepared, &ctx).unwrap();
        assert!(estimate.upper > estimate.lower);
        let interval = estimate.interval.expect("an interval was requested");
        assert_eq!(interval.method, IvBoundsIntervalMethod::ImbensManskiRowBootstrap);
        assert!(interval.lower <= estimate.lower && interval.upper >= estimate.upper);
        assert!(interval.lower_se > 0.0 && interval.upper_se > 0.0);
        let again = estimator.estimate(&prepared, &ExecutionContext::for_tests(43)).unwrap();
        assert_eq!(again, estimate, "a fixed seed reproduces the interval");
        // A declared cluster design refuses a row-level draw.
        let clustered = IvBoundsEstimator::new()
            .with_bootstrap_replicates(30)
            .with_sampling_design(SamplingDeclaration::Declared(
                antecedent_core::DeclaredSamplingDesign::ClusteredPanel,
            ));
        let refusal = clustered.estimate(&prepared, &ctx).unwrap_err();
        assert!(refusal.to_string().contains("next attempt"), "{refusal}");
    }

    #[test]
    fn the_posterior_of_the_endpoints_brackets_the_observed_bounds() {
        let data = table([[[50, 20], [10, 20]], [[20, 10], [20, 50]]]);
        let estimator = IvBoundsEstimator::new();
        let prepared = estimator.prepare(&data, &query()).unwrap();
        let ctx = ExecutionContext::for_tests(44);
        let estimate = estimator.estimate_bayesian(&prepared, 200, &ctx).unwrap();
        let posterior = estimate.posterior.expect("a posterior summary");
        assert!(posterior.lower_mean <= posterior.upper_mean);
        assert!((posterior.lower_mean - estimate.lower).abs() < 0.1);
        assert!((posterior.upper_mean - estimate.upper).abs() < 0.1);
        assert!(posterior.lower_sd > 0.0 && posterior.upper_sd > 0.0);
        let interval = estimate.interval.expect("a credible interval");
        assert_eq!(interval.method, IvBoundsIntervalMethod::PosteriorEndpointQuantiles);
        assert!(interval.lower <= posterior.lower_mean && interval.upper >= posterior.upper_mean);
        let few = estimator.estimate_bayesian(&prepared, 5, &ctx).unwrap_err();
        assert!(few.to_string().contains("at least 20"), "{few}");
    }

    #[test]
    fn row_bootstrap_replicates_of_neighbouring_bases_are_distinct_streams() {
        // A base combined with the replicate id by exclusive-or maps every (base, replicate)
        // pair with equal XOR to one stream: 0x...5001 ^ 3 == 0x...5002 ^ 0.
        let ctx = ExecutionContext::for_tests(46);
        let rows = 64;
        let draw = |base: u64, replicate: u64| {
            let mut out = Vec::new();
            fill_resample_index_replicate(
                ResamplingPlan::IidBootstrap,
                rows,
                None,
                &ctx,
                base,
                replicate,
                &mut out,
            )
            .unwrap();
            out.iter().map(|id| id.get()).collect::<Vec<_>>()
        };
        let mut seen = std::collections::HashSet::new();
        for base in [BOOTSTRAP_STREAM, POSTERIOR_STREAM, BOOTSTRAP_STREAM + 0x10] {
            for replicate in 0..16_u64 {
                assert!(
                    seen.insert(draw(base, replicate)),
                    "base {base:#x} replicate {replicate} repeats another stream"
                );
            }
        }
    }

    #[test]
    fn posterior_draws_do_not_reuse_the_bootstrap_replicate_streams() {
        let ctx = ExecutionContext::for_tests(49);
        let posterior = |draw: u64| {
            ctx.rng
                .stream_for(
                    StreamDomain::Estimate,
                    antecedent_core::derive_stream(POSTERIOR_STREAM, draw),
                )
                .next_u64()
        };
        let mut seen = std::collections::HashSet::new();
        for draw in 0..64_u64 {
            assert!(seen.insert(posterior(draw)), "draw {draw} repeats another draw");
        }
        let xor =
            |base: u64, id: u64| ctx.rng.stream_for(StreamDomain::Estimate, base ^ id).next_u64();
        // The old derivation: these two draws were the same stream.
        assert_eq!(xor(BOOTSTRAP_STREAM, 3), xor(POSTERIOR_STREAM, 0));
        assert_ne!(posterior(3), posterior(0));
    }
}
