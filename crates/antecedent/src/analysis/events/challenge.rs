//! Decide whether a Hawkes excitation mass can be read as a structural effect.
//!
//! The fit reports a conditional intensity. A second fit adds a declared
//! immigrant feature. The structural reading is retained only when the
//! horizon-specific direct mass agrees within a declared tolerance and neither
//! fit is weakly identified or pinned to its stability cap. Otherwise the mass
//! stays a conditional-intensity functional.

use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

use super::hawkes::{self, BaselineFeature, HawkesFit, HawkesFitSpec, HawkesParameters};
use super::marked::{self, MarkedFit, MarkedFitSpec, MarkedHistory};
use super::{
    AdmissionLease, EventError, EventHistory, PreparedKernel, admit, digest, history_bytes,
};

/// Whether the horizon-specific direct mass survived the immigrant challenge.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralStatus {
    /// The two fits agree on an interior mass. It may be read as a direct effect.
    Retained,
    /// The mass is a conditional-intensity functional.
    Refused,
}

/// Result of challenging one excitation edge.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExcitationReading {
    /// Retained only under the declared tolerance and interior fits.
    pub status: StructuralStatus,
    /// Why a structural reading was refused.
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
    /// Direct effect on the challenged horizon.
    ///
    /// # Errors
    /// The immigrant challenge refused a structural reading.
    pub fn structural_direct(&self) -> Result<f64, EventError> {
        match self.status {
            StructuralStatus::Retained => Ok(self.conditional_direct),
            StructuralStatus::Refused => Err(EventError::Fit(
                "excitation mass is a conditional intensity, not a structural effect",
            )),
        }
    }
}

/// In-window offspring of releasing at two times, under a retained reading.
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

pub(super) struct EdgeFacts {
    pub source: usize,
    pub target: usize,
    pub horizon: f64,
    pub tolerance: f64,
    pub conditional_direct: f64,
    pub alternative_direct: f64,
    pub weak_constant: bool,
    pub weak_alternative: bool,
    pub capped_constant: bool,
    pub capped_alternative: bool,
    pub lifetime_direct: f64,
    pub lifetime_descendants: f64,
    pub horizon_descendants: f64,
}

pub(super) fn judge(facts: EdgeFacts) -> Result<ExcitationReading, EventError> {
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
    let reason = if facts.weak_constant {
        Some("constant-baseline fit is weakly identified")
    } else if facts.weak_alternative {
        Some("immigrant-alternative fit is weakly identified")
    } else if facts.capped_constant {
        Some("constant-baseline fit is on its stability cap")
    } else if facts.capped_alternative {
        Some("immigrant-alternative fit is on its stability cap")
    } else if (facts.conditional_direct - facts.alternative_direct).abs() > facts.tolerance {
        Some("immigrant alternative moved the horizon mass")
    } else {
        None
    };
    Ok(ExcitationReading {
        status: if reason.is_none() {
            StructuralStatus::Retained
        } else {
            StructuralStatus::Refused
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
        let scoped = super::scoped_execution(ctx, self.plan.identity(), super::CAPABILITY)?;
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
    judge(EdgeFacts {
        source,
        target,
        horizon,
        tolerance,
        conditional_direct: offspring.direct[target],
        alternative_direct: other.direct[target],
        weak_constant: constant.weak_identification,
        weak_alternative: alternative.weak_identification,
        capped_constant: constant.active_columns.contains(&source),
        capped_alternative: alternative.active_columns.contains(&source),
        lifetime_direct: offspring.lifetime_direct[target],
        lifetime_descendants: offspring.lifetime_descendants[target],
        horizon_descendants: offspring.horizon_descendants[target],
    })
}

/// Compare two release times inside a fixed window, under a retained catalog reading.
///
/// A release at time `t` contributes over the remaining horizon `window_end - t`.
///
/// # Errors
/// The reading was refused, the times are unordered, or the parameters are not
/// the kernel the reading was computed from.
pub fn compare_release_times(
    parameters: &HawkesParameters,
    reading: &ExcitationReading,
    earlier: f64,
    later: f64,
    window_end: f64,
) -> Result<ReleaseChoice, EventError> {
    reading.structural_direct()?;
    if !(earlier < later && later <= window_end) || !window_end.is_finite() {
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
        let scoped = super::scoped_execution(ctx, self.plan.identity(), super::CAPABILITY)?;
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
    judge(EdgeFacts {
        source,
        target,
        horizon,
        tolerance,
        conditional_direct: offspring.direct[target],
        alternative_direct: other.direct[target],
        weak_constant: constant.weak_identification,
        weak_alternative: alternative.weak_identification,
        capped_constant: constant.cap_active,
        capped_alternative: alternative.cap_active,
        lifetime_direct: offspring.lifetime_direct[target],
        lifetime_descendants: offspring.lifetime_descendants[target],
        horizon_descendants: offspring.horizon_descendants[target],
    })
}

/// Compare two release times for a marked kernel under a retained reading.
///
/// # Errors
/// The reading was refused, the times are unordered, or the kernel does not
/// match the reading.
pub fn compare_marked_release_times(
    parameters: &super::marked::MarkedParameters,
    prototypes: &[super::marked::MarkPrototype],
    reading: &ExcitationReading,
    earlier: f64,
    later: f64,
    window_end: f64,
) -> Result<ReleaseChoice, EventError> {
    reading.structural_direct()?;
    if !(earlier < later && later <= window_end) || !window_end.is_finite() {
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

#[cfg(test)]
mod reading_tests {
    use super::{StructuralStatus, catalog_reading, compare_release_times};
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
    fn agreement_retains_the_horizon_mass_and_a_shift_refuses_it() {
        let constant = shell(0.4, false, false);
        let reading =
            catalog_reading(&constant, &shell(0.1, false, false), 0, 0, 1.0, 0.05).unwrap();
        assert_eq!(reading.status, StructuralStatus::Refused);
        assert_eq!(reading.reason, Some("immigrant alternative moved the horizon mass"));
        assert!(reading.structural_direct().is_err());
        let retained =
            catalog_reading(&constant, &shell(0.4, false, false), 0, 0, 1.0, 0.0).unwrap();
        let direct = 0.4 * (1.0 - (-1.0_f64).exp());
        assert!((retained.structural_direct().unwrap() - direct).abs() < 1e-12);
        assert_eq!(
            catalog_reading(&shell(0.4, true, false), &constant, 0, 0, 1.0, 1.0).unwrap().reason,
            Some("constant-baseline fit is weakly identified")
        );
        assert_eq!(
            catalog_reading(&constant, &shell(0.4, false, true), 0, 0, 1.0, 1.0).unwrap().reason,
            Some("immigrant-alternative fit is on its stability cap")
        );
        let choice =
            compare_release_times(&constant.parameters, &retained, 0.0, 7.0, 10.0).unwrap();
        let early = 0.4 * (1.0 - (-10.0_f64).exp());
        let late = 0.4 * (1.0 - (-3.0_f64).exp());
        assert!((choice.earlier_direct - early).abs() < 1e-12);
        assert!((choice.later_direct - late).abs() < 1e-12);
        assert_eq!(choice.greater_descendant_release, Some(0));
        assert!(compare_release_times(&constant.parameters, &reading, 0.0, 1.0, 2.0).is_err());
    }
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
