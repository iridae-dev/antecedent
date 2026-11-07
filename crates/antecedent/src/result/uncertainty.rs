use super::EffectEstimate;

/// Nominal level of the interval a standard-error result reports.
///
/// An [`EffectEstimate`] carries a standard error, not endpoints; every
/// rendering of it (Python views, HTML, the analysis-result artifact's
/// `standard_error`) forms the two-sided `estimate ± 1.96·SE` interval (a Student-t
/// reference where the estimate declares `se_reference_df`, see
/// [`PublishedScalarUncertainty::interval`]).
/// Posterior summaries report the equal-tailed `q025` / `q975` interval at the
/// same level.
pub const REPORTED_SE_INTERVAL_LEVEL: f64 = 0.95;

/// Support diagnostic marking a function-valued response whose interval is a
/// normal interval from the plug-in bootstrap SE (the general-ID
/// `functional.effect` route resamples the whole front-door plug-in each
/// replicate). Its values are `[requested replicates, successful replicates]`.
/// A response carrying it reports [`IntervalMethod::BootstrapSe`](antecedent_core::IntervalMethod::BootstrapSe) rather than the
/// analytic-SE default; every other response band stays analytic.
pub(crate) const RESPONSE_BOOTSTRAP_SE: &str = "response.bootstrap_se";

/// Fewest successful replicates that earn a nominal 0.95 percentile band or a
/// normal interval from a bootstrap SE.
///
/// Under the (B+1) rule each tail holds `(B+1)·α/2` order statistics. At
/// `α = 0.05` that count exceeds one only for `B ≥ 40` (`B = 39` is exactly
/// one). Statistical transport and the tuple/block bootstrap floor both read
/// this constant.
pub const PERCENTILE_95_MIN_REPLICATES: u32 = 40;

/// Two-sided normal critical value of [`REPORTED_SE_INTERVAL_LEVEL`].
///
/// Every path that forms `estimate ± z·SE` at the published level reads this
/// one quantile rather than writing `normal_ppf(0.975)` again:
/// `0.5 + 0.95 / 2.0` is exactly `0.975` in binary64, so the value is
/// bit-identical to the literal it replaces.
#[must_use]
pub fn reported_se_interval_z() -> f64 {
    antecedent_stats::normal_ppf(0.5 + REPORTED_SE_INTERVAL_LEVEL / 2.0)
}

/// How one reported interval was formed: the construction the calibration
/// match key describes.
///
/// Derived from the result's own content by
/// [`StudyResult::primary_interval_binding`](super::StudyResult::primary_interval_binding) and
/// [`StudyResult::identified_set_interval_binding`](super::StudyResult::identified_set_interval_binding), so it always describes
/// what the execution reported, never what was requested.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct IntervalBinding {
    /// Interval method.
    pub method: antecedent_core::IntervalMethod,
    /// Analytic SE kind recorded by the estimator, when the method is analytic.
    pub se_kind: Option<antecedent_estimate::AnalyticSeKind>,
    /// Nominal coverage level of the reported interval.
    pub level: f64,
    /// Dependence rule of the construction: `iid`, `panel_cluster`, or
    /// `circular_block:<family>` ([`antecedent_estimate::CircularBlockFamily::tag`]).
    pub dependence: &'static str,
    /// Resampling replicates that succeeded (bootstrap, circular block, or
    /// simultaneous-band multipliers), when the interval is resampling-based.
    pub replicates_ok: Option<u32>,
    /// Resampling replicates that failed, when reported.
    pub replicates_failed: Option<u32>,
    /// Posterior draws behind a posterior interval.
    pub posterior_draws: Option<u32>,
}

impl IntervalBinding {
    pub(super) const fn new(
        method: antecedent_core::IntervalMethod,
        level: f64,
        dependence: &'static str,
    ) -> Self {
        Self {
            method,
            se_kind: None,
            level,
            dependence,
            replicates_ok: None,
            replicates_failed: None,
            posterior_draws: None,
        }
    }

    /// Binding of an execution that reported no interval.
    #[must_use]
    pub const fn none(dependence: &'static str) -> Self {
        Self::new(antecedent_core::IntervalMethod::None, f64::NAN, dependence)
    }
}

/// Dependence label of a circular-block family.
#[must_use]
pub const fn circular_block_dependence(
    family: antecedent_estimate::CircularBlockFamily,
) -> &'static str {
    match family {
        antecedent_estimate::CircularBlockFamily::SingleWindow => "circular_block:single_window",
        antecedent_estimate::CircularBlockFamily::Mediation => "circular_block:mediation",
        antecedent_estimate::CircularBlockFamily::Sequential => "circular_block:sequential",
        antecedent_estimate::CircularBlockFamily::Mixture => "circular_block:mixture",
        antecedent_estimate::CircularBlockFamily::RowBlock => "circular_block:row_block",
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Published scalar standard error and the label of its normal interval.
///
/// One owner of the facade's SE / `± z·SE` rule. Wire `standard_error`,
/// [`StudyResult::primary_interval_binding`](super::StudyResult::primary_interval_binding), the Uncertainty slot, and the
/// Python copies all read this; they do not prefer `se_bootstrap` on their own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedScalarUncertainty {
    /// The one scalar SE the facade publishes, when supported and positive-finite.
    pub standard_error: Option<f64>,
    /// Interval method for what was published: a normal interval from
    /// [`Self::standard_error`] ([`antecedent_core::IntervalMethod::AnalyticSe`] or
    /// [`antecedent_core::IntervalMethod::BootstrapSe`]), an Anderson–Rubin set
    /// ([`antecedent_core::IntervalMethod::AndersonRubin`], endpoints in [`Self::lower`] /
    /// [`Self::upper`], no standard error), or [`antecedent_core::IntervalMethod::None`].
    /// Circular-block labeling is applied by
    /// [`StudyResult::primary_interval_binding`](super::StudyResult::primary_interval_binding) when a block family is recorded.
    pub method: antecedent_core::IntervalMethod,
    /// Nominal level of the published interval, or `NaN` when nothing was published.
    pub level: f64,
    /// Why a scalar SE was withheld, when the estimate is an IV result that
    /// cannot publish a Wald or bootstrap SE and has no Anderson–Rubin set.
    pub withheld_reason: Option<&'static str>,
    /// Lower endpoint of a published Anderson–Rubin set. May be infinite.
    pub lower: Option<f64>,
    /// Upper endpoint of a published Anderson–Rubin set. May be infinite.
    pub upper: Option<f64>,
}

impl PublishedScalarUncertainty {
    /// Select the one published scalar SE for an [`EffectEstimate`].
    ///
    /// Rules:
    /// 1. Only a positive-finite SE is published (`Some` alone is not enough).
    /// 2. When first-stage diagnostics carry an Anderson–Rubin set, publish
    ///    those endpoints and no standard error. Do not fall through to
    ///    `se_bootstrap` or a Wald SE.
    /// 3. When first-stage diagnostics are present and no Anderson–Rubin set
    ///    was formed, do not fall through to `se_bootstrap`. A positive-finite
    ///    analytic SE is still published; otherwise withhold, keeping the
    ///    estimator's reason when it recorded one.
    /// 4. Otherwise prefer a positive-finite bootstrap SE over a positive-finite
    ///    analytic SE (AIPW and peers keep today's preference). IV and NN
    ///    matching do not emit a competing finite pair.
    #[must_use]
    pub fn select(estimate: &EffectEstimate) -> Self {
        use antecedent_core::IntervalMethod as M;

        if let Some(diagnostics) = &estimate.diagnostics.first_stage {
            if let Some((lower, upper, level)) = diagnostics.anderson_rubin {
                if !lower.is_nan()
                    && !upper.is_nan()
                    && level.is_finite()
                    && (0.0..1.0).contains(&level)
                {
                    return Self {
                        standard_error: None,
                        method: M::AndersonRubin,
                        level,
                        withheld_reason: None,
                        lower: Some(lower),
                        upper: Some(upper),
                    };
                }
            }
            if positive_finite(estimate.se_analytic()) {
                return Self {
                    standard_error: Some(estimate.se_analytic()),
                    method: M::AnalyticSe,
                    level: REPORTED_SE_INTERVAL_LEVEL,
                    withheld_reason: None,
                    lower: None,
                    upper: None,
                };
            }
            return Self::withhold_iv(diagnostics.uncertainty_withheld);
        }

        if estimate.se_bootstrap().is_some_and(positive_finite) {
            return Self {
                standard_error: estimate.se_bootstrap(),
                method: M::BootstrapSe,
                level: REPORTED_SE_INTERVAL_LEVEL,
                withheld_reason: None,
                lower: None,
                upper: None,
            };
        }
        if positive_finite(estimate.se_analytic()) {
            return Self {
                standard_error: Some(estimate.se_analytic()),
                method: M::AnalyticSe,
                level: REPORTED_SE_INTERVAL_LEVEL,
                withheld_reason: None,
                lower: None,
                upper: None,
            };
        }
        Self {
            standard_error: None,
            method: M::None,
            level: f64::NAN,
            withheld_reason: None,
            lower: None,
            upper: None,
        }
    }

    /// Two-sided critical value of the published interval for `estimate`: Student-t with
    /// `se_reference_df` degrees of freedom for an analytic SE that declares one (the
    /// cluster-DML few-cluster reference), the normal quantile otherwise.
    #[must_use]
    pub fn critical_value(&self, estimate: &EffectEstimate) -> f64 {
        match (self.method, estimate.uncertainty.analytic.reference_df) {
            (antecedent_core::IntervalMethod::AnalyticSe, Some(df)) if df > 0.0 => {
                antecedent_stats::student_t_ppf(0.5 + self.level / 2.0, df)
            }
            _ => reported_se_interval_z(),
        }
    }

    /// The published `estimate ± critical_value · SE` interval, when a scalar SE was
    /// published.
    #[must_use]
    pub fn interval(&self, estimate: &EffectEstimate) -> Option<(f64, f64)> {
        let se = self.standard_error?;
        let half = self.critical_value(estimate) * se;
        Some((estimate.ate - half, estimate.ate + half))
    }

    fn withhold_iv(reason: Option<&'static str>) -> Self {
        Self {
            standard_error: None,
            method: antecedent_core::IntervalMethod::None,
            level: f64::NAN,
            withheld_reason: Some(reason.unwrap_or(
                "Wald/bootstrap SE not supported (weak instrument or IV uncertainty withheld)",
            )),
            lower: None,
            upper: None,
        }
    }
}
