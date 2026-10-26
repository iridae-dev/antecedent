// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of a class-aware (Cpdag / Pag, explicit or accepted) derivative,
//! elasticity or Jacobian.
//!
//! Each evaluated completion is an atom of the identified set and carries its own
//! adjustment (or front-door) estimand. The derivative refuter state runs on every atom
//! against that atom's own estimand and its own published derivative:
//!
//! * an adjustment-identified completion runs the static derivative state
//!   (`derivative.support`, `derivative.finite_difference`, under `full`
//!   `derivative.bootstrap_stability`, and for Bayesian studies the predictive checks
//!   and prior sensitivity of the outcome regression);
//! * a front-door-identified completion runs the front-door derivative state;
//! * a completion identified by the general ID algorithm has no plug-in to refit, so it
//!   is reported as unchecked mass.
//!
//! Reports are mixed into one verdict by completion weight only when every completion
//! publishes the same derivative; otherwise each completion's reports are kept on their
//! own and a disagreement diagnostic states how far the completions spread. The state
//! executes and is unlicensed: no coverage or evidence record names it.

use super::atom_mix::{ScalarRefuterMix, finish_atom_refutations};
use super::derivative_refutation::{
    DerivativeRefutationInput, derivative_scalar, refute_derivative,
};
use super::response_path::FrontDoorClaim;
use super::{
    CausalError, CausalResponse, ContinuousResponseEstimator, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EstimandMethod, ExecutionContext, InferenceMode, ResponseQuery,
    TabularData,
};

/// Relative tolerance under which completion derivatives count as one value.
const SAME_VALUE_TOLERANCE: f64 = 1e-9;

/// Reports, predictive checks and notes of the per-completion derivative state.
pub(super) type ClassDerivativeRefutation = (
    Vec<antecedent_validate::RefutationReport>,
    Vec<antecedent_validate::PredictiveCheckReport>,
    Vec<Diagnostic>,
);

impl super::Study {
    /// Run the per-completion derivative refuter state of a class-aware derivative.
    ///
    /// `weighted` is the evaluated completions as `(case index, weight, response)`.
    ///
    /// # Errors
    ///
    /// Cancellation, admission, or a failure of the Bayesian posterior checks.
    pub(super) fn refute_class_derivative_atoms<G>(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        envelope: &antecedent_identify::IdentificationEnvelope<G>,
        weighted: &[(u64, f64, CausalResponse)],
        ctx: &ExecutionContext,
    ) -> Result<ClassDerivativeRefutation, CausalError> {
        let mut per_atom = Vec::new();
        let mut checks = Vec::new();
        let mut diagnostics = Vec::new();
        let mut values = Vec::new();
        let mut unchecked = 0_usize;
        for (key, weight, response) in weighted {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_VALIDATE,
                });
            }
            let Some(case) = usize::try_from(*key).ok().and_then(|index| envelope.cases.get(index))
            else {
                continue;
            };
            let Some(estimand) = case.result.estimands.first() else { continue };
            let (value, _) = derivative_scalar(response);
            if !value.is_finite() || *weight <= 0.0 {
                continue;
            }
            values.push(value);
            let reports = match estimand.method_kind().ok() {
                Some(EstimandMethod::FrontDoor) => {
                    let claim = FrontDoorClaim {
                        estimand,
                        status: case.result.status,
                        assumptions: case.result.required_assumptions.clone(),
                    };
                    let (reports, notes) = self.refute_front_door_derivative_response(
                        (data, query),
                        &claim,
                        response,
                        1,
                        ctx,
                    )?;
                    diagnostics.extend(notes);
                    reports
                }
                Some(EstimandMethod::GeneralId) => {
                    unchecked += 1;
                    Vec::new()
                }
                _ => {
                    let mut fitter = ContinuousResponseEstimator::new(std::sync::Arc::clone(
                        &estimand.adjustment_set,
                    ));
                    if let Some(options) = &self.response_options {
                        fitter.options = options.clone();
                    }
                    let bayesian = match &self.inference {
                        InferenceMode::Bayesian(config) => Some(
                            super::bayesian_level_refutation::bayesian_response_estimator_in_force(
                                &fitter, data, query, config, ctx,
                            )?,
                        ),
                        InferenceMode::Frequentist => None,
                    };
                    let (reports, atom_checks, notes) = refute_derivative(
                        &DerivativeRefutationInput {
                            data,
                            query,
                            estimand,
                            fitter: &fitter,
                            inference: &self.inference,
                            bayesian: bayesian.as_ref(),
                            status: case.result.status,
                            assumptions: &case.result.required_assumptions,
                            response,
                            suite: self.refute,
                            latency: self.latency_mode,
                            replicates: self.bootstrap_replicates,
                        },
                        ctx,
                    )?;
                    checks.extend(atom_checks);
                    diagnostics.extend(notes);
                    reports
                }
            };
            per_atom.push((*weight, reports));
        }
        let same_value = class_derivative_diagnostics(&values, unchecked, &mut diagnostics);
        let (reports, diagnostics) = finish_atom_refutations(
            per_atom,
            diagnostics,
            same_value,
            ScalarRefuterMix::Effect,
            None,
            false,
        );
        Ok((reports, checks, diagnostics))
    }
}

fn class_derivative_diagnostics(
    values: &[f64],
    unchecked: usize,
    diagnostics: &mut Vec<Diagnostic>,
) -> bool {
    let lower = values.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let spread = if values.is_empty() { f64::NAN } else { upper - lower };
    let same_value = spread.is_finite()
        && spread <= SAME_VALUE_TOLERANCE * lower.abs().max(upper.abs()).max(1.0);
    diagnostics.push(
        Diagnostic::new(
            "refute.response.class_envelope_disagreement",
            DiagnosticKind::Scientific,
            if same_value { DiagnosticSeverity::Info } else { DiagnosticSeverity::Warning },
            format!(
                "{} evaluated completion(s); completion derivatives span [{lower}, {upper}] \
                 (spread {spread}). {}",
                values.len(),
                if same_value {
                    "Every completion publishes the same derivative, so the per-completion \
                     refuter reports are mixed by completion weight."
                } else {
                    "The completions disagree, so each completion's refuter reports are \
                     kept on their own; no mixed verdict is formed over an identified set."
                }
            ),
        )
        .with_fields([
            ("completions", values.len().to_string()),
            ("lower", lower.to_string()),
            ("upper", upper.to_string()),
        ]),
    );
    if unchecked > 0 {
        diagnostics.push(Diagnostic::new(
            "refute.derivative.general_id_unchecked",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "{unchecked} completion(s) are identified by the general ID algorithm \
                 through a functional that is neither an adjustment nor a front door, so \
                 no derivative plug-in exists to refit; their mass is not covered by the \
                 derivative refuter state"
            ),
        ));
    }
    diagnostics.push(Diagnostic::new(
        "refute.response.class_derivative_state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "class-aware derivative refuter state: the static derivative state (support, \
         finite difference of the identified levels, and under full resampling stability) \
         on each completion's own adjustment estimand, the front-door state on a \
         front-door completion. It executes unlicensed: no coverage or evidence record \
         names it, and the verdicts are checks, not a calibration claim.",
    ));
    same_value
}
