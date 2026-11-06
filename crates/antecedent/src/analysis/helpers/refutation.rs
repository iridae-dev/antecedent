//! Shared refutation runs and validator diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AverageEffectQuery, CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimationWorkspace, ExecutionContext, IdentifiedEstimand, RefutationProblem,
    RefutationReport, RefuteSuite, TabularData, ValidationSuite,
};

/// Diagnostic surfacing one validator skipped as [`antecedent_validate::ValidationOutcome::NotApplicable`]
/// for this run.
///
/// This is a **per-run, data-dependent** skip — e.g. `OverlapRefuter` on a temporal
/// design, `EValue` on a non-binary treatment, an MCMC diagnostic on a Laplace posterior
/// — not the support matrix's permanent typed impossibility
/// (`SupportRefusal::NotApplicable` / wire `not_applicable`). The same validator can run
/// cleanly on a different call against the same supported cell; a matrix `not_applicable`
/// cell never reaches validation at all. See `docs/capabilities.md`'s "'Not applicable'
/// means three different things".
pub(crate) fn validator_not_applicable_diagnostic(
    validator: antecedent_validate::ValidatorId,
    reason: &str,
) -> Diagnostic {
    let mut d = Diagnostic::new(
        "refute.validator.not_applicable",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "validator '{validator}' was requested but is not applicable to this run's data/estimand \
             (per-run skip, not a permanent support-matrix refusal): {reason}"
        ),
    );
    d.fields = Arc::from([(Arc::from("validator"), Arc::from(validator.as_str()))]);
    d
}

/// Build one [`validator_not_applicable_diagnostic`] per skipped outcome.
pub(crate) fn validator_not_applicable_diagnostics(
    outcomes: &[antecedent_validate::ValidationOutcome],
) -> Vec<Diagnostic> {
    let mut diagnostics: Vec<_> = ValidationSuite::not_applicable_only(outcomes)
        .into_iter()
        .map(|(validator, reason)| validator_not_applicable_diagnostic(validator, &reason))
        .collect();
    for outcome in outcomes {
        if let antecedent_validate::ValidationOutcome::Failed { validator, reason } = outcome {
            let mut d = Diagnostic::new(
                "refute.validator.failed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!("validator {validator} could not compute a verdict: {reason}"),
            );
            d.fields = Arc::from([
                (Arc::from("validator"), validator.clone()),
                (Arc::from("reason"), reason.clone()),
            ]);
            diagnostics.push(d);
        }
    }
    diagnostics
}

/// One effect estimate under refutation and the suite that checks it.
///
/// The three entry points differ only in what they return: raw outcomes, reports plus
/// per-run skip diagnostics, or the plugin-level subset for an intervention *level*.
pub(crate) struct RefutationRun<'a> {
    pub(crate) data: &'a TabularData,
    pub(crate) estimand: &'a IdentifiedEstimand,
    pub(crate) query: &'a AverageEffectQuery,
    pub(crate) estimate: &'a EffectEstimate,
    pub(crate) suite: RefuteSuite,
    /// Estimator id the refuters report against.
    pub(crate) estimator: &'a str,
    pub(crate) custom: &'a [Arc<dyn antecedent_validate::CustomEffectValidator>],
    /// Present when the refuters must refit through a temporal indexer.
    pub(crate) temporal: Option<antecedent_validate::TemporalRefitContext<'a>>,
}

impl RefutationRun<'_> {
    /// Run the requested refuter suite, returning raw outcomes (reports and
    /// per-run `NotApplicable` skips) without collapsing them.
    pub(crate) fn outcomes(
        self,
        workspace: &mut EstimationWorkspace,
        propensity: Option<&mut antecedent_stats::PropensityWorkspace>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<antecedent_validate::ValidationOutcome>, CausalError> {
        let problem = RefutationProblem::new(
            self.data,
            self.estimand,
            self.query,
            self.estimate,
            Some(self.estimator),
            self.temporal,
        );
        let mut validation = match self.suite {
            RefuteSuite::None => {
                if self.custom.is_empty() {
                    return Ok(Vec::new());
                }
                ValidationSuite::new()
            }
            RefuteSuite::Cheap => ValidationSuite::overlap_and_evalue(),
            RefuteSuite::PlaceboAndRcc => ValidationSuite::placebo_and_rcc(),
            RefuteSuite::Full => ValidationSuite::full_effect(),
        };
        for v in self.custom {
            validation = validation.with_custom(Arc::clone(v));
        }
        match propensity {
            Some(pws) => validation
                .run_with_propensity(&problem, workspace, pws, ctx)
                .map_err(CausalError::from),
            None => validation.run(&problem, workspace, ctx).map_err(CausalError::from),
        }
    }

    /// Run the requested refuter suite, returning both the produced reports and one
    /// diagnostic per validator that was requested but skipped as `NotApplicable` for this
    /// run (see [`validator_not_applicable_diagnostic`]).
    pub(crate) fn reports(
        self,
        workspace: &mut EstimationWorkspace,
        propensity: Option<&mut antecedent_stats::PropensityWorkspace>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<RefutationReport>, Vec<Diagnostic>), CausalError> {
        let outcomes = self.outcomes(workspace, propensity, ctx)?;
        let diagnostics = validator_not_applicable_diagnostics(&outcomes);
        Ok((ValidationSuite::reports_only(&outcomes), diagnostics))
    }

    /// Cheap/full on a plugin intervention *level*. Contrast-shaped refuters are
    /// not supported. Cheap is overlap only; full is overlap plus sampling-stability.
    pub(crate) fn plugin_level(
        self,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<RefutationReport>, Vec<Diagnostic>), CausalError> {
        let problem = RefutationProblem::new(
            self.data,
            self.estimand,
            self.query,
            self.estimate,
            Some(self.estimator),
            None,
        );
        let mut validation = match self.suite {
            RefuteSuite::None => {
                if self.custom.is_empty() {
                    return Ok((Vec::new(), Vec::new()));
                }
                ValidationSuite::new()
            }
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc => ValidationSuite::overlap_only(),
            RefuteSuite::Full => ValidationSuite::plugin_level_full(),
        };
        for v in self.custom {
            validation = validation.with_custom(Arc::clone(v));
        }
        let outcomes = validation.run(&problem, workspace, ctx).map_err(CausalError::from)?;
        let diagnostics = validator_not_applicable_diagnostics(&outcomes);
        Ok((ValidationSuite::reports_only(&outcomes), diagnostics))
    }
}
