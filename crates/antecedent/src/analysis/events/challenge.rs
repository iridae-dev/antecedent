//! Compare a constant-baseline Hawkes fit with a declared immigrant alternative.
//!
//! Both fits are conditional intensities. The diagnostic agrees when the
//! horizon-specific direct masses differ by at most a caller-supplied tolerance
//! and neither fit is weakly identified or pinned to its stability cap.
//! Agreement is not identification. A shared latent trend can survive both
//! models, and the tolerance is not a derived uncertainty test. A causal
//! reading still needs an identification strategy, or explicit structural
//! assumptions, and the evidence for them. Release-time offspring are a
//! conditional scenario under an agreed diagnostic, not an identified market
//! response.

use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

use super::hawkes::{self, BaselineFeature, HawkesFit, HawkesFitSpec, HawkesParameters};
use super::marked::{self, MarkedFit, MarkedFitSpec, MarkedHistory};
use super::{
    AdmissionLease, EventError, EventHistory, PreparedKernel, admit, digest, history_bytes,
};

/// Outcome of the declared immigrant diagnostic.
///
/// [`DiagnosticStatus::Agreed`] means the horizon masses moved by at most the
/// caller's tolerance and neither fit was weakly identified or on its stability
/// cap. That checks two conditional intensities. It does not identify a causal
/// effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticStatus {
    /// The caller's tolerance and the interior-fit checks both hold.
    Agreed,
    /// The diagnostic does not support a conditional scenario.
    Withheld,
}

/// Result of challenging one excitation edge.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExcitationReading {
    /// [`DiagnosticStatus::Agreed`] only under the declared tolerance and interior fits.
    pub status: DiagnosticStatus,
    /// Why the diagnostic withheld a conditional scenario.
    pub reason: Option<&'static str>,
    /// Queried source index.
    pub source: usize,
    /// Queried target index.
    pub target: usize,
    /// Horizon the direct masses were compared on.
    pub horizon: f64,
    /// Direct mass of the constant-baseline fit on `[0, horizon)`.
    pub conditional_direct: f64,
    /// Direct mass of the immigrant-alternative fit on `[0, horizon)`.
    pub alternative_direct: f64,
    /// Lifetime direct mass of the constant-baseline fit.
    pub lifetime_direct: f64,
    /// Lifetime descendants of the constant-baseline fit, excluding the initiator.
    pub lifetime_descendants: f64,
    /// Horizon descendants of the constant-baseline fit, from the filter ODE.
    pub horizon_descendants: f64,
}

impl ExcitationReading {
    /// Horizon mass of the constant-baseline fit when the diagnostic agrees.
    ///
    /// The number is that conditional-intensity functional. Agreement does not
    /// turn it into an identified causal effect.
    ///
    /// # Errors
    /// The diagnostic was withheld.
    pub fn agreed_direct(&self) -> Result<f64, EventError> {
        match self.status {
            DiagnosticStatus::Agreed => Ok(self.conditional_direct),
            DiagnosticStatus::Withheld => Err(EventError::Fit(
                "immigrant diagnostic did not agree; the mass stays a conditional intensity",
            )),
        }
    }
}

/// In-window offspring of one added event at two times.
///
/// Computed from the constant-baseline kernel after the immigrant diagnostic
/// agrees. This is a conditional scenario under that kernel and those
/// diagnostics, not an identified market response.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReleaseChoice {
    /// Earlier release time.
    pub earlier: f64,
    /// Later release time.
    pub later: f64,
    /// End of the window the publisher scores.
    pub window_end: f64,
    /// Direct target children inside the window from the earlier release.
    pub earlier_direct: f64,
    /// Direct target children inside the window from the later release.
    pub later_direct: f64,
    /// Target descendants inside the window from the earlier release.
    pub earlier_descendants: f64,
    /// Target descendants inside the window from the later release.
    pub later_descendants: f64,
    /// `Some(0)` when the earlier release puts more target descendants in the
    /// window, `Some(1)` when the later one does, `None` when they tie.
    pub greater_descendant_release: Option<u8>,
}

pub(super) struct FitFlags {
    pub weak: bool,
    pub capped: bool,
}

pub(super) struct EdgeFacts {
    pub source: usize,
    pub target: usize,
    pub horizon: f64,
    pub tolerance: f64,
    pub conditional_direct: f64,
    pub alternative_direct: f64,
    pub constant: FitFlags,
    pub alternative: FitFlags,
    pub lifetime_direct: f64,
    pub lifetime_descendants: f64,
    pub horizon_descendants: f64,
}

pub(super) fn judge(facts: &EdgeFacts) -> Result<ExcitationReading, EventError> {
    if !facts.horizon.is_finite()
        || facts.horizon <= 0.0
        || !facts.tolerance.is_finite()
        || facts.tolerance < 0.0
        || !facts.conditional_direct.is_finite()
        || !facts.alternative_direct.is_finite()
    {
        return Err(EventError::invalid(
            "excitation challenge needs a positive horizon and a nonnegative tolerance",
        ));
    }
    let reason = if facts.constant.weak {
        Some("constant-baseline fit is weakly identified")
    } else if facts.alternative.weak {
        Some("immigrant-alternative fit is weakly identified")
    } else if facts.constant.capped {
        Some("constant-baseline fit is on its stability cap")
    } else if facts.alternative.capped {
        Some("immigrant-alternative fit is on its stability cap")
    } else if (facts.conditional_direct - facts.alternative_direct).abs() > facts.tolerance {
        Some("immigrant alternative moved the horizon mass")
    } else {
        None
    };
    Ok(ExcitationReading {
        status: if reason.is_none() {
            DiagnosticStatus::Agreed
        } else {
            DiagnosticStatus::Withheld
        },
        reason,
        source: facts.source,
        target: facts.target,
        horizon: facts.horizon,
        conditional_direct: facts.conditional_direct,
        alternative_direct: facts.alternative_direct,
        lifetime_direct: facts.lifetime_direct,
        lifetime_descendants: facts.lifetime_descendants,
        horizon_descendants: facts.horizon_descendants,
    })
}

/// Catalog challenge: constant baseline against a declared immigrant feature.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcitationChallenge {
    /// Complete history the two fits share.
    pub history: EventHistory,
    /// Fixed-decay scope. Both fits use it.
    pub fit: HawkesFitSpec,
    /// Varying nonnegative immigrant regressor.
    pub alternative: BaselineFeature,
    /// Source type.
    pub source: usize,
    /// Target type.
    pub target: usize,
    /// Horizon of the direct mass being challenged.
    pub horizon: f64,
    /// Largest absolute movement of that mass still read as the same effect.
    pub tolerance: f64,
}

/// Compiled catalog challenge. Execution fits both models.
pub struct PreparedExcitationChallenge {
    request: ExcitationChallenge,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

impl crate::Study {
    /// Compile a catalog excitation challenge before either fit runs.
    ///
    /// # Errors
    /// Invalid history, fit scope, feature or resource refusal.
    pub fn prepare_excitation_challenge(
        request: ExcitationChallenge,
        ctx: &ExecutionContext,
    ) -> Result<PreparedExcitationChallenge, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        request.history.validate(true)?;
        request.fit.validate(request.history.catalog.len())?;
        request.alternative.validate(request.history.window)?;
        if request.source >= request.history.catalog.len()
            || request.target >= request.history.catalog.len()
        {
            return Err(EventError::invalid("excitation challenge type is outside the catalog"));
        }
        let lease = admit(ctx, 0, history_bytes(&request.history))?;
        let source = digest("antecedent.events.excitation_challenge.v1", &request)?;
        let plan = PreparedKernel::new(source, "excitation_challenge", super::CAPABILITY)?;
        Ok(PreparedExcitationChallenge { request, plan, _lease: lease })
    }
}

impl PreparedExcitationChallenge {
    /// Inspect the compiled challenge.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Fit both immigrant models and judge the queried edge.
    ///
    /// # Errors
    /// A fit fails, or the fitted kernel is not stable.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<ExcitationReading, EventError> {
        let scoped = super::scoped_execution(ctx, &self.plan)?;
        let mut work = hawkes::Work::new(&scoped);
        let constant = hawkes::fit(&self.request.history, &self.request.fit, &mut work)?;
        let alternative = hawkes::fit_alternative(
            &self.request.history,
            &self.request.fit,
            &self.request.alternative,
            &mut work,
        )?;
        catalog_reading(
            &constant,
            &alternative,
            self.request.source,
            self.request.target,
            self.request.horizon,
            self.request.tolerance,
        )
    }

    /// Fit the constant-baseline kernel the diagnostic compares.
    ///
    /// # Errors
    /// The optimizer fails, or the fitted kernel is not stable.
    pub fn kernel(&self, ctx: &ExecutionContext) -> Result<super::ExcitationSubject, EventError> {
        let scoped = super::scoped_execution(ctx, &self.plan)?;
        let mut work = hawkes::Work::new(&scoped);
        let constant = hawkes::fit(&self.request.history, &self.request.fit, &mut work)?;
        Ok(super::ExcitationSubject::Catalog {
            parameters: constant.parameters,
            history: self.request.history.clone(),
            column_cap: self.request.fit.column_cap,
        })
    }
}

fn catalog_reading(
    constant: &HawkesFit,
    alternative: &HawkesFit,
    source: usize,
    target: usize,
    horizon: f64,
    tolerance: f64,
) -> Result<ExcitationReading, EventError> {
    let offspring = constant.parameters.excitation_offspring(source, horizon)?;
    let other = alternative.parameters.excitation_offspring(source, horizon)?;
    judge(&EdgeFacts {
        source,
        target,
        horizon,
        tolerance,
        conditional_direct: offspring.direct[target],
        alternative_direct: other.direct[target],
        constant: FitFlags {
            weak: constant.weak_identification,
            capped: constant.active_columns.contains(&source),
        },
        alternative: FitFlags {
            weak: alternative.weak_identification,
            capped: alternative.active_columns.contains(&source),
        },
        lifetime_direct: offspring.lifetime_direct[target],
        lifetime_descendants: offspring.lifetime_descendants[target],
        horizon_descendants: offspring.horizon_descendants[target],
    })
}

/// Compare two release times under a kernel whose diagnostic agreed.
///
/// A time `t` is scored on the remaining horizon `window_end - t`, as offspring
/// of one added event in the constant-baseline kernel. The comparison is
/// available only when [`ExcitationReading::agreed_direct`] succeeds. It does
/// not identify the effect of releasing. A utility over path outcomes is a
/// decision contract on the event worlds.
///
/// # Errors
/// The diagnostic was withheld, the times are unordered, or the parameters are
/// not the kernel the reading was computed from.
pub fn compare_release_times(
    parameters: &HawkesParameters,
    reading: &ExcitationReading,
    earlier: f64,
    later: f64,
    window_end: f64,
) -> Result<ReleaseChoice, EventError> {
    reading.agreed_direct()?;
    if !(earlier < later && later <= window_end && window_end.is_finite()) {
        return Err(EventError::invalid(
            "release times must be ordered and fall at or before the window end",
        ));
    }
    let matched = parameters.excitation_offspring(reading.source, reading.horizon)?;
    if (matched.direct[reading.target] - reading.conditional_direct).abs() > 1e-8
        || matched.source != reading.source
    {
        return Err(EventError::invalid(
            "release comparison parameters do not match the retained reading",
        ));
    }
    let earlier_offspring = offspring_or_zero(parameters, reading, window_end - earlier)?;
    let later_offspring = offspring_or_zero(parameters, reading, window_end - later)?;
    Ok(choice(
        earlier,
        later,
        window_end,
        earlier_offspring.0,
        later_offspring.0,
        earlier_offspring.1,
        later_offspring.1,
    ))
}

fn offspring_or_zero(
    parameters: &HawkesParameters,
    reading: &ExcitationReading,
    remaining: f64,
) -> Result<(f64, f64), EventError> {
    if remaining <= 0.0 {
        return Ok((0.0, 0.0));
    }
    let offspring = parameters.excitation_offspring(reading.source, remaining)?;
    Ok((offspring.direct[reading.target], offspring.horizon_descendants[reading.target]))
}

/// Marked-kernel challenge against the same kind of immigrant feature.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkedExcitationChallenge {
    /// Marked history.
    pub history: MarkedHistory,
    /// Fixed-decay marked scope.
    pub fit: MarkedFitSpec,
    /// Varying nonnegative immigrant regressor.
    pub alternative: BaselineFeature,
    /// Source prototype.
    pub source: usize,
    /// Target prototype.
    pub target: usize,
    /// Horizon of the direct mass being challenged.
    pub horizon: f64,
    /// Largest absolute movement still read as the same effect.
    pub tolerance: f64,
}

/// Compiled marked challenge.
pub struct PreparedMarkedChallenge {
    request: MarkedExcitationChallenge,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

impl crate::Study {
    /// Compile a marked excitation challenge before either fit runs.
    ///
    /// # Errors
    /// Invalid history, fit scope, feature or resource refusal.
    pub fn prepare_marked_challenge(
        request: MarkedExcitationChallenge,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMarkedChallenge, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        request.history.validate()?;
        request.fit.validate(&request.history)?;
        request.alternative.validate(request.history.window)?;
        if request.source >= request.history.prototypes.len()
            || request.target >= request.history.prototypes.len()
        {
            return Err(EventError::invalid("marked challenge prototype is outside the history"));
        }
        let lease = admit(ctx, 0, 8_192 + request.history.events.len() as u64 * 64)?;
        let source = digest("antecedent.events.marked_challenge.v1", &request)?;
        let plan = PreparedKernel::new(source, "marked_excitation_challenge", super::CAPABILITY)?;
        Ok(PreparedMarkedChallenge { request, plan, _lease: lease })
    }
}

impl PreparedMarkedChallenge {
    /// Inspect the compiled marked challenge.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Fit both marked models and judge the queried prototypes.
    ///
    /// # Errors
    /// A fit fails, or the fitted kernel is not stable.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<ExcitationReading, EventError> {
        let scoped = super::scoped_execution(ctx, &self.plan)?;
        let mut work = hawkes::Work::new(&scoped);
        let constant = marked::fit(&self.request.history, &self.request.fit, &mut work)?;
        let alternative = marked::fit_alternative(
            &self.request.history,
            &self.request.fit,
            &self.request.alternative,
            &mut work,
        )?;
        marked_reading(
            &constant,
            &alternative,
            self.request.source,
            self.request.target,
            self.request.horizon,
            self.request.tolerance,
        )
    }

    /// Fit the constant-baseline marked kernel the diagnostic compares.
    ///
    /// # Errors
    /// The optimizer fails, or the fitted kernel is not stable.
    pub fn kernel(&self, ctx: &ExecutionContext) -> Result<super::ExcitationSubject, EventError> {
        let scoped = super::scoped_execution(ctx, &self.plan)?;
        let mut work = hawkes::Work::new(&scoped);
        let constant = marked::fit(&self.request.history, &self.request.fit, &mut work)?;
        Ok(super::ExcitationSubject::Marked {
            parameters: constant.parameters,
            prototypes: constant.prototypes,
            history: self.request.history.clone(),
            reproduction_cap: self.request.fit.reproduction_cap,
        })
    }
}

fn marked_reading(
    constant: &MarkedFit,
    alternative: &MarkedFit,
    source: usize,
    target: usize,
    horizon: f64,
    tolerance: f64,
) -> Result<ExcitationReading, EventError> {
    let offspring =
        constant.parameters.excitation_offspring(&constant.prototypes, source, horizon)?;
    let other =
        alternative.parameters.excitation_offspring(&alternative.prototypes, source, horizon)?;
    judge(&EdgeFacts {
        source,
        target,
        horizon,
        tolerance,
        conditional_direct: offspring.direct[target],
        alternative_direct: other.direct[target],
        constant: FitFlags { weak: constant.weak_identification, capped: constant.cap_active },
        alternative: FitFlags {
            weak: alternative.weak_identification,
            capped: alternative.cap_active,
        },
        lifetime_direct: offspring.lifetime_direct[target],
        lifetime_descendants: offspring.lifetime_descendants[target],
        horizon_descendants: offspring.horizon_descendants[target],
    })
}

/// Compare two release times for a marked kernel whose diagnostic agreed.
///
/// The comparison uses the same conditional scenario as
/// [`compare_release_times`]: offspring of one added event, not an identified
/// market response.
///
/// # Errors
/// The diagnostic was withheld, the times are unordered, or the kernel does not
/// match the reading.
pub fn compare_marked_release_times(
    parameters: &super::marked::MarkedParameters,
    prototypes: &[super::marked::MarkPrototype],
    reading: &ExcitationReading,
    earlier: f64,
    later: f64,
    window_end: f64,
) -> Result<ReleaseChoice, EventError> {
    reading.agreed_direct()?;
    if !(earlier < later && later <= window_end && window_end.is_finite()) {
        return Err(EventError::invalid(
            "release times must be ordered and fall at or before the window end",
        ));
    }
    let matched = parameters.excitation_offspring(prototypes, reading.source, reading.horizon)?;
    if (matched.direct[reading.target] - reading.conditional_direct).abs() > 1e-8 {
        return Err(EventError::invalid(
            "release comparison parameters do not match the retained reading",
        ));
    }
    let at = |remaining: f64| -> Result<(f64, f64), EventError> {
        if remaining <= 0.0 {
            return Ok((0.0, 0.0));
        }
        let offspring = parameters.excitation_offspring(prototypes, reading.source, remaining)?;
        Ok((offspring.direct[reading.target], offspring.horizon_descendants[reading.target]))
    };
    let earlier_offspring = at(window_end - earlier)?;
    let later_offspring = at(window_end - later)?;
    Ok(choice(
        earlier,
        later,
        window_end,
        earlier_offspring.0,
        later_offspring.0,
        earlier_offspring.1,
        later_offspring.1,
    ))
}

fn choice(
    earlier: f64,
    later: f64,
    window_end: f64,
    earlier_direct: f64,
    later_direct: f64,
    earlier_descendants: f64,
    later_descendants: f64,
) -> ReleaseChoice {
    let scale = 1.0 + earlier_descendants.abs().max(later_descendants.abs());
    let greater_descendant_release =
        if (earlier_descendants - later_descendants).abs() <= 1e-12 * scale {
            None
        } else if earlier_descendants > later_descendants {
            Some(0)
        } else {
            Some(1)
        };
    ReleaseChoice {
        earlier,
        later,
        window_end,
        earlier_direct,
        later_direct,
        earlier_descendants,
        later_descendants,
        greater_descendant_release,
    }
}

#[cfg(test)]
mod reading_tests {
    use super::{DiagnosticStatus, catalog_reading, compare_release_times};
    use crate::analysis::events::hawkes::{HawkesFit, HawkesParameters};

    fn shell(alpha: f64, weak: bool, active: bool) -> HawkesFit {
        HawkesFit {
            parameters: HawkesParameters {
                baseline: vec![0.5],
                branching: vec![alpha],
                decay: vec![1.0],
            },
            log_likelihood: 0.0,
            iterations: 1,
            projected_residual: 0.0,
            active_columns: if active { vec![0] } else { vec![] },
            zero_edges: vec![],
            weak_identification: weak,
        }
    }

    #[test]
    fn agreement_returns_the_horizon_mass_and_a_shift_withholds_it() {
        let constant = shell(0.4, false, false);
        let reading =
            catalog_reading(&constant, &shell(0.1, false, false), 0, 0, 1.0, 0.05).unwrap();
        assert_eq!(reading.status, DiagnosticStatus::Withheld);
        assert_eq!(reading.reason, Some("immigrant alternative moved the horizon mass"));
        assert!(reading.agreed_direct().is_err());
        let agreed = catalog_reading(&constant, &shell(0.4, false, false), 0, 0, 1.0, 0.0).unwrap();
        let direct = 0.4 * (1.0 - (-1.0_f64).exp());
        assert!((agreed.agreed_direct().unwrap() - direct).abs() < 1e-12);
        assert_eq!(
            catalog_reading(&shell(0.4, true, false), &constant, 0, 0, 1.0, 1.0).unwrap().reason,
            Some("constant-baseline fit is weakly identified")
        );
        assert_eq!(
            catalog_reading(&constant, &shell(0.4, false, true), 0, 0, 1.0, 1.0).unwrap().reason,
            Some("immigrant-alternative fit is on its stability cap")
        );
        let choice = compare_release_times(&constant.parameters, &agreed, 0.0, 7.0, 10.0).unwrap();
        let early = 0.4 * (1.0 - (-10.0_f64).exp());
        let late = 0.4 * (1.0 - (-3.0_f64).exp());
        assert!((choice.earlier_direct - early).abs() < 1e-12);
        assert!((choice.later_direct - late).abs() < 1e-12);
        assert_eq!(choice.greater_descendant_release, Some(0));
        assert!(compare_release_times(&constant.parameters, &reading, 0.0, 1.0, 2.0).is_err());
    }
}
