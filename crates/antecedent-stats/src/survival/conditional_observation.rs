//! Standardized event curves under baseline-dependent censoring and truncation.

use super::{
    RandomizedSurvivalSummary, SurvivalDifferenceBand, SurvivalEndpoint, SurvivalRows,
    bootstrap_support_count, for_each_resampled_subject, randomized_survival_summary,
    supremum_endpoints,
};

/// Retained result label for an owned reverse product-limit fit.
pub const STRATIFIED_CENSORING_PROVENANCE: &str =
    "same_subject_stratified_reverse_product_limit_refit";
/// Reported uncertainty for the standardized event curve.
pub const STRATIFIED_CENSORING_BAND: &str = "arm_stratified_full_refit_bootstrap_supremum_95";

/// Row-aligned baseline inputs for stratified reverse product-limit censoring.
pub struct ConditionalObservationRows<'a> {
    /// Event, assignment and optional entry columns, one independent row per subject.
    pub survival: SurvivalRows<'a>,
    /// A binary baseline stratum governing truncation and modification of the outcome law.
    pub entry_stratum: &'a [f64],
    /// A binary baseline censoring stratum within each arm and entry stratum.
    pub censoring_stratum: &'a [f64],
    /// Fixed pre-entry population proportions aligned with the entry stratum.
    pub population_weights: [f64; 2],
    /// Floor on the fitted censoring left limits through the horizon.
    pub censoring_floor: f64,
}

/// Standardized curves and a subject-bootstrap band that refits censoring in every draw.
pub struct ConditionalObservationEstimate {
    /// Standardized arm curves and restricted means on the original event-time grid.
    pub summary: RandomizedSurvivalSummary,
    /// Smallest fitted censoring left limit on the estimation grid.
    pub minimum_censoring_survival: f64,
    /// Whole-grid band when a bootstrap was requested and the support gate was satisfied.
    pub band: Option<SurvivalDifferenceBand>,
}

/// Fit arm/entry/censoring-stratum reverse product-limit curves, then standardize over entry.
///
/// Entry is independent of latent event time, cause, censoring time and censoring stratum
/// given arm and entry stratum. Censoring is independent of latent event time and cause
/// given arm and both baseline strata. Each pair of strata needs positive pre-entry
/// membership, time-zero entry and follow-up through the horizon. At ties events take
/// priority over censoring. The event-time risk weight is the inverse censoring left limit
/// from that same subject's cell. Mixing within-entry weighted event curves by the fixed
/// pre-entry population proportions identifies the marginal curve for that population.
///
/// # Errors
/// Returns a refusal for invalid inputs, inadequate entry or censoring overlap, an
/// unsupported bootstrap count, insufficient viable refits or cancellation.
pub fn standardized_conditional_survival(
    input: &ConditionalObservationRows<'_>,
    tau: f64,
    endpoint: SurvivalEndpoint,
    draws: u32,
    seed: u64,
    cancelled: impl Fn() -> bool,
) -> Result<ConditionalObservationEstimate, &'static str> {
    let rows = &input.survival;
    if rows.duration.iter().any(|&time| time <= 0.0) {
        return Err("conditional survival requires positive observed follow-up durations");
    }
    let original = randomized_survival_summary(rows, tau, endpoint)?;
    let n = rows.duration.len();
    if input.entry_stratum.len() != n || input.censoring_stratum.len() != n {
        return Err(
            "conditional survival identification requires both baseline strata for every subject",
        );
    }
    let [p0, p1] = input.population_weights;
    if !p0.is_finite() || !p1.is_finite() || p0 <= 0.0 || p1 <= 0.0 || (p0 + p1 - 1.0).abs() > 1e-12
    {
        return Err(
            "conditional survival identification requires two positive source-population weights summing to one",
        );
    }
    if !(input.censoring_floor > 0.0 && input.censoring_floor <= 1.0) {
        return Err("conditional survival requires a censoring probability floor in (0, 1]");
    }
    let binary = |value| match value {
        0.0 => Ok(0),
        1.0 => Ok(1),
        _ => Err("conditional survival baseline strata must be binary zero or one"),
    };
    let mut cells: [[[Vec<usize>; 2]; 2]; 2] = Default::default();
    let mut arms = [Vec::new(), Vec::new()];
    for (i, &arm) in rows.treated.iter().enumerate() {
        let arm = usize::from(arm);
        let entry = binary(input.entry_stratum[i])?;
        let censoring = binary(input.censoring_stratum[i])?;
        cells[arm][entry][censoring].push(i);
        arms[arm].push(i);
    }
    let mut grid = original.times.clone();
    grid.extend(
        rows.duration
            .iter()
            .zip(rows.event_code)
            .filter_map(|(&time, &code)| (code == 0 && time < tau).then_some(time)),
    );
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let design = Standardization { input, cells, grid, tau, endpoint };
    let mut counts = vec![1; n];
    let Fit { curves, minimum_g, minimum_horizon_risk } =
        design.fit(&counts, if draws > 0 { 25 } else { 1 }, &cancelled)?;
    let summary = design.summarize(original, &curves);
    let band = if draws > 0 {
        Some(design.band(
            &summary,
            &arms,
            &mut counts,
            draws,
            seed,
            minimum_horizon_risk,
            &cancelled,
        )?)
    } else {
        None
    };
    Ok(ConditionalObservationEstimate { summary, minimum_censoring_survival: minimum_g, band })
}

struct Standardization<'a> {
    input: &'a ConditionalObservationRows<'a>,
    cells: [[[Vec<usize>; 2]; 2]; 2],
    grid: Vec<f64>,
    tau: f64,
    endpoint: SurvivalEndpoint,
}

struct Fit {
    curves: [Vec<f64>; 2],
    minimum_g: f64,
    minimum_horizon_risk: usize,
}

#[derive(Default)]
struct RiskSet {
    risk: usize,
    events: usize,
    target_events: usize,
    censored: usize,
}

impl Standardization<'_> {
    fn entry(&self, i: usize) -> f64 {
        self.input.survival.delayed_entry.map_or(0.0, |entry| entry[i])
    }

    #[allow(clippy::float_cmp, reason = "risk sets match exactly recorded exit-time knots")]
    fn risk_set(&self, cell: &[usize], counts: &[usize], time: f64) -> RiskSet {
        let mut set = RiskSet::default();
        let rows = &self.input.survival;
        for &i in cell {
            let exit = rows.duration[i];
            if (self.entry(i) < time || (time == 0.0 && self.entry(i) == 0.0)) && exit >= time {
                let count = counts[i];
                set.risk += count;
                if exit == time {
                    let cause = rows.event_code[i];
                    if cause == 0 {
                        set.censored += count;
                    } else {
                        set.events += count;
                        if matches!(self.endpoint, SurvivalEndpoint::CumulativeIncidence { target_cause } if target_cause == cause)
                        {
                            set.target_events += count;
                        }
                    }
                }
            }
        }
        set
    }

    fn fit(
        &self,
        counts: &[usize],
        minimum_risk: usize,
        cancelled: &impl Fn() -> bool,
    ) -> Result<Fit, &'static str> {
        let mut curves = self.cells.each_ref().map(|_| vec![0.0; self.grid.len()]);
        let mut minimum_g = 1.0_f64;
        let mut minimum_horizon_risk = usize::MAX;
        for (arm, entries) in self.cells.iter().enumerate() {
            for (entry, strata) in entries.iter().enumerate() {
                let mut survival = 1.0;
                let mut incidence = 0.0;
                let mut g = [1.0; 2];
                for (position, &time) in self.grid.iter().enumerate() {
                    if cancelled() {
                        return Err("conditional survival fit was cancelled");
                    }
                    let sets = strata.each_ref().map(|cell| self.risk_set(cell, counts, time));
                    if sets.iter().any(|set| set.risk < minimum_risk) {
                        return Err(
                            "conditional survival has inadequate entry or follow-up overlap within an arm and baseline history",
                        );
                    }
                    if position + 1 == self.grid.len() {
                        minimum_horizon_risk =
                            minimum_horizon_risk.min(sets[0].risk.min(sets[1].risk));
                    }
                    let weighted = |count: fn(&RiskSet) -> usize| {
                        count(&sets[0]) as f64 / g[0] + count(&sets[1]) as f64 / g[1]
                    };
                    let risk = weighted(|set| set.risk);
                    let events = weighted(|set| set.events);
                    incidence += survival * weighted(|set| set.target_events) / risk;
                    survival *= 1.0 - (events / risk).min(1.0);
                    let value = if self.endpoint == SurvivalEndpoint::Survival {
                        survival
                    } else {
                        incidence
                    };
                    curves[arm][position] += self.input.population_weights[entry] * value;
                    for (probability, set) in g.iter_mut().zip(sets) {
                        minimum_g = minimum_g.min(*probability);
                        if set.censored > 0 {
                            *probability *=
                                1.0 - set.censored as f64 / (set.risk - set.events) as f64;
                        }
                    }
                    if time < self.tau && g.iter().any(|&value| value < self.input.censoring_floor)
                    {
                        return Err(
                            "conditional survival has inadequate fitted censoring positivity",
                        );
                    }
                }
            }
        }
        let initial = f64::from(self.endpoint == SurvivalEndpoint::Survival);
        let weight_sum = self.input.population_weights.iter().sum::<f64>();
        for curve in &mut curves {
            for value in curve.iter_mut() {
                *value = (*value / weight_sum).clamp(0.0, 1.0);
            }
            curve[0] = initial;
        }
        Ok(Fit { curves, minimum_g, minimum_horizon_risk })
    }

    fn summarize(
        &self,
        mut summary: RandomizedSurvivalSummary,
        curves: &[Vec<f64>; 2],
    ) -> RandomizedSurvivalSummary {
        let align = |values: &[f64]| -> Vec<f64> {
            summary
                .times
                .iter()
                .map(|time| values[self.grid.partition_point(|knot| knot <= time) - 1])
                .collect()
        };
        summary.control = align(&curves[0]);
        summary.treated = align(&curves[1]);
        if self.endpoint == SurvivalEndpoint::Survival {
            let integrate = |curve: &[f64]| {
                summary.times.windows(2).zip(curve).map(|(t, s)| (t[1] - t[0]) * s).sum()
            };
            summary.rmst_control = Some(integrate(&summary.control));
            summary.rmst_treated = Some(integrate(&summary.treated));
        }
        summary
    }

    #[allow(clippy::too_many_arguments, reason = "one retained subject resampling plan")]
    fn band(
        &self,
        summary: &RandomizedSurvivalSummary,
        arms: &[Vec<usize>; 2],
        counts: &mut [usize],
        draws: u32,
        seed: u64,
        minimum_horizon_risk: usize,
        cancelled: &impl Fn() -> bool,
    ) -> Result<SurvivalDifferenceBand, &'static str> {
        if draws < 399 {
            return Err("conditional survival band requires at least 399 full-refit replicates");
        }
        let difference =
            summary.treated.iter().zip(&summary.control).map(|(a, b)| a - b).collect::<Vec<_>>();
        let mut suprema = Vec::with_capacity(draws as usize);
        let mut rng = seed ^ 0x194A_72AF_4096_ED27;
        for _ in 0..draws {
            if cancelled() {
                return Err("conditional survival bootstrap was cancelled");
            }
            counts.fill(0);
            for_each_resampled_subject(arms, &mut rng, |i, _| counts[i] += 1);
            if let Ok(fit) = self.fit(counts, 1, cancelled) {
                let sup = summary
                    .times
                    .iter()
                    .zip(&difference)
                    .map(|(time, center)| {
                        let knot = self.grid.partition_point(|at| at <= time) - 1;
                        (fit.curves[1][knot] - fit.curves[0][knot] - center).abs()
                    })
                    .fold(0.0_f64, f64::max);
                suprema.push(sup);
            }
        }
        if cancelled() {
            return Err("conditional survival bootstrap was cancelled");
        }
        let replicates_ok = bootstrap_support_count(
            suprema.len(),
            draws,
            399,
            "too many conditional survival draws",
            "conditional survival band has insufficient viable subject refits",
        )?;
        let (lower, upper) = supremum_endpoints(&difference, &mut suprema);
        Ok(SurvivalDifferenceBand {
            times: summary.times.clone(),
            difference,
            lower,
            upper,
            replicates_ok,
            minimum_horizon_risk_set: self
                .input
                .survival
                .delayed_entry
                .map(|_| minimum_horizon_risk),
        })
    }
}
