use super::{
    CausalResponse, Diagnostic, DiagnosticKind, IntervalBinding, PublishedScalarUncertainty,
    REPORTED_SE_INTERVAL_LEVEL, RESPONSE_BOOTSTRAP_SE, StudyResult, circular_block_dependence,
};

fn draws_u32(n: usize) -> Option<u32> {
    u32::try_from(n).ok()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the guard admits only finite values in [0, u32::MAX], so the cast neither wraps nor loses sign (fractions truncate by design)"
)]
fn count_u32(value: f64) -> Option<u32> {
    (value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX)).then_some(value as u32)
}

fn support_values<'a>(response: &'a CausalResponse, id: &str) -> Option<&'a [f64]> {
    response
        .support
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.id.as_ref() == id)
        .map(|diagnostic| diagnostic.values.as_ref())
}

/// A Frequentist temporal response band built from circular-block replicates
/// of the lag-aligned surface (support diagnostic `response.temporal.block_length`).
fn temporal_response_band(response: &CausalResponse, level: f64) -> Option<IntervalBinding> {
    support_values(response, antecedent_estimate::TEMPORAL_RESPONSE_BLOCK_LENGTH)?;
    let mut binding = IntervalBinding::new(
        antecedent_core::IntervalMethod::CircularBlockSe,
        level,
        "circular_block:response",
    );
    if let Some(counts) = support_values(response, "response.observation_block_bootstrap") {
        binding.replicates_ok = counts.get(1).and_then(|n| count_u32(*n));
    }
    Some(binding)
}

impl StudyResult {
    /// The diagnostics that describe whether the data support the estimand (overlap, positivity,
    /// coordinates outside the observed support): exactly those emitted as
    /// [`DiagnosticKind::Support`], whatever their code. The one owner of the Support slot's
    /// diagnostic selection.
    pub fn support_diagnostics(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter().filter(|d| d.kind == DiagnosticKind::Support)
    }

    /// Dependence rule the execution's data imposes before any circular-block
    /// family: panel executions cluster by unit, every other modality is `iid`
    /// unless a circular-block construction names its family.
    fn base_dependence(&self) -> &'static str {
        if self.logical_plan.data_classification == antecedent_contract::DataClassification::Panel {
            "panel_cluster"
        } else {
            "iid"
        }
    }

    /// Dependence of a frequentist standard-error interval that names no circular-block
    /// family.
    ///
    /// On time-ordered rows (a series or an event log) an SE that models no serial
    /// correlation is not the `iid` construction of exchangeable rows: its variance ignores
    /// the dependence the data have, and a calibration measured for `iid` rows says nothing
    /// about it. It is labelled `serial_unmodelled`; a Newey-West kernel SE models the serial
    /// correlation and is labelled `serial_hac`. Every other modality keeps
    /// [`Self::base_dependence`]. The posterior bindings keep the base label too: a temporal
    /// posterior's serial-dependence handling (long-run tempering) is part of its construction
    /// and named by its posterior key, not by this dimension.
    fn se_dependence(&self, se_kind: Option<antecedent_estimate::AnalyticSeKind>) -> &'static str {
        use antecedent_contract::DataClassification as C;
        match self.logical_plan.data_classification {
            C::Temporal | C::Event => match se_kind {
                Some(antecedent_estimate::AnalyticSeKind::NeweyWest { .. }) => "serial_hac",
                _ => "serial_unmodelled",
            },
            _ => self.base_dependence(),
        }
    }

    fn posterior_draw_count(&self) -> Option<u32> {
        self.posterior.as_ref().and_then(|posterior| draws_u32(posterior.draws.n_draws))
    }

    /// A frequentist response band formed as a normal interval from the plug-in
    /// bootstrap SE (support diagnostic [`RESPONSE_BOOTSTRAP_SE`]): the general-ID
    /// `functional.effect` route has no exposed influence function, so its
    /// interval is the 199-replicate front-door resample, not an analytic SE.
    fn bootstrap_response_band(
        &self,
        response: &CausalResponse,
        level: f64,
    ) -> Option<IntervalBinding> {
        let values = support_values(response, RESPONSE_BOOTSTRAP_SE)?;
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::BootstrapSe,
            level,
            self.se_dependence(None),
        );
        binding.replicates_ok = values.get(1).and_then(|n| count_u32(*n));
        Some(binding)
    }

    /// Construction of the primary interval this execution reported.
    ///
    /// `bayesian` is the program's inference family: a Bayesian response's
    /// band is a posterior band even when the draws live in a referenced
    /// artifact rather than on the result.
    ///
    /// One owner of the rule, read by the claim's calibration match key:
    ///
    /// 1. A function-valued response reports its `ResponseUncertainty`
    ///    (`None` → [`IntervalMethod::None`](antecedent_core::IntervalMethod::None);
    ///    simultaneous band; identified-envelope band; posterior; a temporal
    ///    circular-block band; otherwise the influence-function band).
    /// 2. An interventional distribution with a binary outcome reports its
    ///    bounded mean interval.
    /// 3. A posterior reports its `q025` / `q975` interval.
    /// 4. A standard error reports `estimate ± 1.96·SE` from
    ///    [`PublishedScalarUncertainty::select`]: the published bootstrap SE
    ///    when that is the selected SE (circular-block when a family is
    ///    recorded), else the published analytic SE with the estimator's
    ///    recorded kind. A bootstrap interval here is a normal interval from
    ///    the bootstrap SE, not a percentile interval (replicates are not
    ///    retained). A requested bootstrap that reported no positive-finite SE
    ///    is not a bootstrap interval; an IV result that withheld Wald/bootstrap
    ///    uncertainty is [`IntervalMethod::None`](antecedent_core::IntervalMethod::None).
    /// 5. When no finite scalar interval exists, a withheld unknown-orientation
    ///    effect reports its scenario max-t band as a simultaneous interval.
    /// 6. Otherwise a reported identified-set interval is the primary interval,
    ///    or nothing was reported.
    #[allow(
        clippy::too_many_lines,
        reason = "the primary-interval choice is one ordered precedence list (cluster and panel intervals, response bands, a withheld unknown-orientation band, the identified set); splitting it would hide the ranking"
    )]
    #[must_use]
    pub fn primary_interval_binding(&self, bayesian: bool) -> IntervalBinding {
        use antecedent_core::{IntervalMethod as M, ResponseUncertainty as U};
        let panel_interval = self.panel_did().is_some_and(|did| did.interval_95.is_some());
        let randomized_cluster_interval = self.randomized_effect().is_some_and(|fit| {
            fit.interval_95.is_some() && fit.assignment_design.as_ref() == "cluster"
        });
        let base = if panel_interval || randomized_cluster_interval {
            "cluster"
        } else if self.panel_did().is_some() {
            "iid"
        } else {
            self.base_dependence()
        };
        if let Some(response) = self.response() {
            // A Bayesian response publishes credible bands; the draws behind a
            // band are in the referenced posterior artifact, not on the result.
            let posterior = bayesian || self.posterior.is_some();
            let mut binding = match &response.uncertainty {
                U::None => return IntervalBinding::none(base),
                U::Posterior { .. } => {
                    IntervalBinding::new(M::PosteriorQuantile, REPORTED_SE_INTERVAL_LEVEL, base)
                }
                U::Scalar { level, .. } | U::PointwiseBand { level, .. } if posterior => {
                    IntervalBinding::new(M::PosteriorQuantile, *level, base)
                }
                U::Scalar { level, .. } | U::PointwiseBand { level, .. } => self
                    .bootstrap_response_band(response, *level)
                    .or_else(|| temporal_response_band(response, *level))
                    .unwrap_or_else(|| {
                        IntervalBinding::new(M::AnalyticSe, *level, self.se_dependence(None))
                    }),
                U::SimultaneousBand { level, replicates, .. } => {
                    let mut binding = IntervalBinding::new(M::SimultaneousBand, *level, base);
                    binding.replicates_ok = Some(*replicates);
                    binding
                }
                U::IdentifiedEnvelopeBand { level, .. } => {
                    IntervalBinding::new(M::IdentifiedSet, *level, base)
                }
            };
            if binding.method == M::PosteriorQuantile {
                binding.posterior_draws = self.posterior_draw_count();
            }
            return binding;
        }
        if let Some(distribution) = self.distribution() {
            if self.posterior.is_none() {
                match distribution.mean_interval {
                    Some(antecedent_estimate::ProbabilityInterval::Bounded { level, .. }) => {
                        let mut binding = IntervalBinding::new(M::BootstrapSe, level, base);
                        binding.replicates_ok = distribution.bootstrap_replicates_ok;
                        binding.replicates_failed = distribution.bootstrap_replicates_failed;
                        return binding;
                    }
                    Some(antecedent_estimate::ProbabilityInterval::Unavailable(_)) => {
                        return IntervalBinding::none(base);
                    }
                    None => {}
                }
            }
        }
        if self.posterior.is_some() {
            let mut binding =
                IntervalBinding::new(M::PosteriorQuantile, REPORTED_SE_INTERVAL_LEVEL, base);
            binding.posterior_draws = self.posterior_draw_count();
            return binding;
        }
        let Some(estimate) = self.estimate.as_effect() else {
            return IntervalBinding::none(base);
        };
        if estimate.ate.is_finite() {
            let published = PublishedScalarUncertainty::select(estimate);
            match published.method {
                M::BootstrapSe => {
                    let (method, dependence) = match estimate.uncertainty.block.family {
                        Some(family) => (M::CircularBlockSe, circular_block_dependence(family)),
                        None => (M::BootstrapSe, self.se_dependence(None)),
                    };
                    let mut binding = IntervalBinding::new(method, published.level, dependence);
                    binding.replicates_ok = estimate.uncertainty.bootstrap.replicates_ok;
                    binding.replicates_failed = estimate.uncertainty.bootstrap.replicates_failed;
                    return binding;
                }
                M::AnalyticSe => {
                    let mut binding = IntervalBinding::new(
                        M::AnalyticSe,
                        published.level,
                        if panel_interval || randomized_cluster_interval {
                            "cluster"
                        } else {
                            self.se_dependence(estimate.uncertainty.analytic.kind)
                        },
                    );
                    binding.se_kind = estimate.uncertainty.analytic.kind;
                    return binding;
                }
                M::AndersonRubin => {
                    return IntervalBinding::new(M::AndersonRubin, published.level, base);
                }
                M::None if published.withheld_reason.is_some() => {
                    return IntervalBinding::none(base);
                }
                _ => {}
            }
        }
        let Some(effect_estimate) = self.estimate.as_effect() else {
            return IntervalBinding::none(base);
        };
        if !effect_estimate.ate.is_finite() {
            if let Some(bands) = effect_estimate.scenarios.intervals.as_deref() {
                if !bands.is_empty() {
                    return IntervalBinding::new(
                        M::SimultaneousBand,
                        REPORTED_SE_INTERVAL_LEVEL,
                        base,
                    );
                }
            }
        }
        self.identified_set_interval_binding().unwrap_or_else(|| IntervalBinding::none(base))
    }

    /// Construction of a reported identified-set interval
    /// (`structural_response.identified_set_interval`), when one was reported.
    ///
    /// A class-aware scalar can report both a frozen-weight point with its
    /// shared-block SE and an Imbens–Manski interval for the identified set;
    /// each states its own calibration.
    #[must_use]
    pub fn identified_set_interval_binding(&self) -> Option<IntervalBinding> {
        use antecedent_estimate::IdentifiedSetIntervalMethod as Method;
        let structural = self.structural_response.as_ref()?;
        let interval = structural.identified_set_interval.as_ref()?;
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::IdentifiedSet,
            interval.level,
            self.base_dependence(),
        );
        match interval.method {
            Method::ImbensManskiSharedBlock => {
                binding.dependence =
                    circular_block_dependence(antecedent_estimate::CircularBlockFamily::Mixture);
                binding.replicates_ok = self
                    .estimate
                    .as_effect()
                    .and_then(|effect| effect.uncertainty.bootstrap.replicates_ok);
                binding.replicates_failed = self
                    .estimate
                    .as_effect()
                    .and_then(|effect| effect.uncertainty.bootstrap.replicates_failed);
            }
            Method::ProductPosteriorEnvelopeQuantile => {
                binding.posterior_draws = structural
                    .atoms
                    .iter()
                    .filter_map(|atom| atom.posterior.as_ref())
                    .filter_map(|posterior| draws_u32(posterior.draws.n_draws))
                    .min()
                    .or_else(|| self.posterior_draw_count());
            }
            _ => {}
        }
        Some(binding)
    }

    /// Construction of a counterfactual's published per-unit intervals
    /// (`counterfactual.unit_effect_intervals`), when the execution formed them.
    #[must_use]
    pub fn unit_effect_interval_binding(&self) -> Option<IntervalBinding> {
        let intervals = match self.counterfactual() {
            Some(single) => single.unit_effect_intervals.as_ref()?,
            None => self.counterfactual_contrasts()?.primary().unit_effect_intervals.as_ref()?,
        };
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::UnitPosteriorQuantile,
            intervals.level,
            self.base_dependence(),
        );
        binding.posterior_draws = self.posterior_draw_count();
        Some(binding)
    }

    /// Construction of a temporal response's published simultaneous band
    /// (support diagnostic `response.simultaneous_band.critical`), when one
    /// accompanies a pointwise band.
    #[must_use]
    pub fn simultaneous_band_binding(&self, bayesian: bool) -> Option<IntervalBinding> {
        let response = self.response()?;
        if matches!(
            response.uncertainty,
            antecedent_core::ResponseUncertainty::SimultaneousBand { .. }
        ) {
            return None;
        }
        let critical = support_values(response, antecedent_estimate::SIMULTANEOUS_BAND_CRITICAL)?;
        let level = *critical.first()?;
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::SimultaneousBand,
            level,
            self.primary_interval_binding(bayesian).dependence,
        );
        binding.replicates_ok = critical.get(2).and_then(|n| count_u32(*n));
        if bayesian || self.posterior.is_some() {
            binding.posterior_draws = binding.replicates_ok.take();
        }
        Some(binding)
    }

    /// Every interval this execution reported: the primary interval first,
    /// then an identified-set interval and a simultaneous band when they are
    /// reported beside it. Each states its own calibration on the claim.
    #[must_use]
    pub fn reported_interval_bindings(&self, bayesian: bool) -> Vec<IntervalBinding> {
        let primary = self.primary_interval_binding(bayesian);
        let mut out = vec![primary];
        for extra in [
            self.identified_set_interval_binding(),
            self.simultaneous_band_binding(bayesian),
            self.unit_effect_interval_binding(),
        ]
        .into_iter()
        .flatten()
        {
            if out.iter().all(|seen| seen.method != extra.method) {
                out.push(extra);
            }
        }
        out
    }

    /// Re-derive [`Self::interval`] from the reported content.
    pub(crate) fn rebind_interval(&mut self, bayesian: bool) {
        self.extras.interval = Some(self.primary_interval_binding(bayesian));
    }
}
