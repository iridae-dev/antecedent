//! Caller-supplied outcome regressions on the trial-to-target transport route.
//!
//! A study that declares [`crate::StudyBuilder::transport_outcome_regressions`] reads the two
//! regression columns and hands them to the same kernel the direct facade entry calls
//! (`antecedent_estimate::trial_to_target_effect`, or
//! `antecedent_estimate::trial_to_weighted_target_effect` for a weighted target), so the
//! augmented contrast the route carries is the one the direct call returns, bit for bit. The
//! IPW contrast in the same kernel result is computed by the same loop with or without
//! regressions, so the published estimate does not move when regressions are declared.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Diagnostic, DiagnosticKind, DiagnosticSeverity};
use antecedent_data::TableView;
use antecedent_estimate::{TransportEffectEstimate, trial_to_target_effect};
use antecedent_identify::TransportIdentification;

use super::TabularData;
use super::transport_interference_path::{TrialColumns, transport_point};
use crate::error::CausalError;

/// Code of the diagnostic that states the augmented contrast a study carries.
pub(super) const SUPPLIED_AUGMENTATION_DIAGNOSTIC: &str = "estimate.transport.trial_aipw_supplied";

/// The two supplied regression columns, read from the executed table.
pub(super) struct SuppliedRegressions {
    control: Vec<f64>,
    treated: Vec<f64>,
}

impl SuppliedRegressions {
    /// `(mu0, mu1)` as the kernel reads them.
    pub(super) fn pair(&self) -> (&[f64], &[f64]) {
        (&self.control, &self.treated)
    }
}

impl super::Study {
    /// The declared regression columns read from `data`; `None` when none were declared.
    ///
    /// # Errors
    ///
    /// A Bayesian study or rows that carry observation weights (`option_not_applicable`), or
    /// a column the table does not hold.
    pub(super) fn supplied_outcome_regressions(
        &self,
        data: &TabularData,
        weighted_rows: bool,
    ) -> Result<Option<SuppliedRegressions>, CausalError> {
        let Some(declared) = self.declared.transport_outcome_regressions.as_ref() else {
            return Ok(None);
        };
        if self.settings.inference.is_bayesian() {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "supplied outcome regressions are read by the frequentist trial-to-target \
                 route; the Bayesian route draws an empirical-support row law and evaluates no \
                 augmented contrast. Next attempt: drop the regressions or run the study \
                 frequentist"
            ));
        }
        if weighted_rows {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "supplied outcome regressions enter the unweighted augmented contrast; the \
                 table's observation weights would be ignored by it rather than refused. Next \
                 attempt: drop the observation weights or the regressions"
            ));
        }
        let control = data.float64_values(declared.control).map_err(CausalError::rejected)?;
        let treated = data.float64_values(declared.treated).map_err(CausalError::rejected)?;
        Ok(Some(SuppliedRegressions { control, treated }))
    }

    /// Refuse supplied regressions on a route that has no single selection diagram.
    ///
    /// # Errors
    ///
    /// `option_not_applicable` when regressions were declared.
    pub(super) fn refuse_supplied_regressions_without_one_diagram(
        &self,
    ) -> Result<(), CausalError> {
        if self.declared.transport_outcome_regressions.is_some() {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "supplied outcome regressions apply to a trial-to-target transport on one \
                 selection diagram under frequentist inference; a graph class or a graph \
                 posterior does not carry them"
            ));
        }
        Ok(())
    }
}

/// The trial-to-target point of `columns`: with supplied regressions, the kernel's IPW and
/// augmented contrast on the same rows; without, [`transport_point`].
///
/// # Errors
///
/// An uncertified formula, or invalid rows, probabilities or regressions.
pub(super) fn supplied_transport_point(
    transport: &TransportIdentification,
    columns: &TrialColumns<'_>,
    regressions: Option<&SuppliedRegressions>,
) -> Result<TransportEffectEstimate, CausalError> {
    let Some(regressions) = regressions else {
        return transport_point(transport, columns);
    };
    let TrialColumns { outcomes, treatment, trial, selection, propensity, weights: _ } = *columns;
    trial_to_target_effect(
        transport,
        outcomes,
        treatment,
        trial,
        selection,
        propensity,
        Some(regressions.pair()),
    )
    .map_err(CausalError::from)
}

/// Field of [`SUPPLIED_AUGMENTATION_DIAGNOSTIC`] holding the augmented contrast's IEEE bits
/// (16 lowercase hex digits).
pub(super) const AIPW_BITS_FIELD: &str = "aipw_bits";
/// Field holding the bits of its standard error, when one is formed.
pub(super) const AIPW_SE_BITS_FIELD: &str = "aipw_se_bits";

/// The route's diagnostics, with the statement of the augmented contrast when the point
/// carries one. The augmented contrast and its standard error are also stated as their IEEE
/// bits in structured fields, so a reader recovers exactly the numbers the kernel returned.
pub(super) fn with_supplied_augmentation(
    diagnostic: Diagnostic,
    point: &TransportEffectEstimate,
    aipw_se: Option<f64>,
) -> Vec<Diagnostic> {
    let Some(aipw) = point.aipw else {
        return vec![diagnostic];
    };
    let bits = |value: f64| format!("{:016x}", value.to_bits());
    let mut fields = vec![(AIPW_BITS_FIELD, bits(aipw))];
    if let Some(se) = aipw_se {
        fields.push((AIPW_SE_BITS_FIELD, bits(se)));
    }
    let message = match aipw_se {
        Some(se) => format!(
            "augmented trial-to-target contrast on the supplied outcome regressions: aipw={aipw} \
             aipw_se={se}; the standard error is the delta-method error of the weighted ratio \
             conditional on the supplied probabilities, regressions and target weights. The \
             published estimate and interval are the IPW ones"
        ),
        None => format!(
            "augmented trial-to-target contrast on the supplied outcome regressions: aipw={aipw}; \
             no standard error is formed for it. The published estimate and interval are the \
             IPW ones"
        ),
    };
    vec![
        diagnostic,
        Diagnostic::new(
            SUPPLIED_AUGMENTATION_DIAGNOSTIC,
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            message,
        )
        .with_fields(fields),
    ]
}
