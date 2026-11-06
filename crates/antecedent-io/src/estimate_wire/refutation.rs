//! Conversion of refutation reports and descriptive scopes.

use std::sync::Arc;

use antecedent_io_core::analysis_wire::{
    OriginalRefutationScopeWire, RefutationReportWire, RefutationScopeWire, RefutationTargetWire,
    WholeTargetDiagnosticWire,
};
use antecedent_validate::target_native::{
    RefutationTarget, TargetFamily, TargetQuadrant, TargetRefutationReport,
};
use antecedent_validate::{RefutationReport, RefutationScope};

use crate::error::IoError;

fn target_to_wire(target: &RefutationTarget) -> RefutationTargetWire {
    RefutationTargetWire {
        version: 1,
        family: target.family().as_str().to_owned(),
        temporal: target.quadrant().temporal,
        bayesian: target.quadrant().bayesian,
        labels: target.labels().iter().map(ToString::to_string).collect(),
        source: *target.source(),
        identity: *target.identity(),
    }
}

fn target_from_wire(wire: &RefutationTargetWire) -> Result<RefutationTarget, IoError> {
    if wire.version != 1 {
        return Err(IoError::Convert("unsupported refutation scope version".into()));
    }
    let family = TargetFamily::parse(&wire.family)
        .ok_or_else(|| IoError::Convert("unknown refutation target family".into()))?;
    let target = RefutationTarget::new(
        family,
        TargetQuadrant { temporal: wire.temporal, bayesian: wire.bayesian },
        wire.labels.iter().cloned(),
        wire.source,
    )
    .map_err(|error| IoError::Convert(error.to_string()))?;
    if target.identity() != &wire.identity {
        return Err(IoError::Convert("refutation target identity mismatch".into()));
    }
    Ok(target)
}

fn scope_to_wire(scope: &RefutationScope) -> RefutationScopeWire {
    match scope {
        RefutationScope::WholeTarget { target } => {
            RefutationScopeWire::WholeTarget { target: target_to_wire(target) }
        }
        RefutationScope::TemporalClassCompletion {
            outer_atom,
            horizon_index,
            horizon,
            completion_digest,
            target,
        } => RefutationScopeWire::TemporalClassCompletion {
            outer_atom: *outer_atom,
            horizon_index: *horizon_index as u64,
            horizon: *horizon,
            completion_digest: *completion_digest,
            target: target_to_wire(target),
        },
        RefutationScope::StaticClassCompletion { outer_atom, completion_digest, target } => {
            RefutationScopeWire::StaticClassCompletion {
                outer_atom: *outer_atom,
                completion_digest: *completion_digest,
                target: target_to_wire(target),
            }
        }
    }
}

fn scope_from_wire(scope: &RefutationScopeWire) -> Result<RefutationScope, IoError> {
    antecedent_io_core::analysis_wire::validate_refutation_scope(scope)?;
    match scope {
        RefutationScopeWire::TemporalClassCompletion { target, .. } if !target.temporal => {
            return Err(IoError::Convert(
                "temporal completion scope requires a temporal target".into(),
            ));
        }
        RefutationScopeWire::StaticClassCompletion { target, .. } if target.temporal => {
            return Err(IoError::Convert(
                "static completion scope requires a static target".into(),
            ));
        }
        _ => {}
    }
    Ok(match scope {
        RefutationScopeWire::RouteSummary { .. }
        | RefutationScopeWire::ScopedReport { .. }
        | RefutationScopeWire::WholeTargetDiagnostic { .. } => {
            return Err(IoError::Convert(
                "descriptive route summary has no native refutation scope".into(),
            ));
        }
        RefutationScopeWire::WholeTarget { target } => {
            RefutationScope::WholeTarget { target: target_from_wire(target)? }
        }
        RefutationScopeWire::TemporalClassCompletion {
            outer_atom,
            horizon_index,
            horizon,
            completion_digest,
            target,
        } => RefutationScope::TemporalClassCompletion {
            outer_atom: *outer_atom,
            horizon_index: usize::try_from(*horizon_index).map_err(|_| IoError::TooLarge)?,
            horizon: *horizon,
            completion_digest: *completion_digest,
            target: target_from_wire(target)?,
        },
        RefutationScopeWire::StaticClassCompletion { outer_atom, completion_digest, target } => {
            RefutationScope::StaticClassCompletion {
                outer_atom: *outer_atom,
                completion_digest: *completion_digest,
                target: target_from_wire(target)?,
            }
        }
    })
}

/// Wrap an original scope with the actual producing-program identity.
///
/// # Errors
/// Refuses descriptive wrappers, summaries, and malformed original scopes.
pub fn descriptive_scope(
    scope: RefutationScopeWire,
    produced_by: [u8; 32],
) -> Result<RefutationScopeWire, IoError> {
    antecedent_io_core::analysis_wire::validate_refutation_scope(&scope)?;
    let scope = match scope {
        RefutationScopeWire::WholeTarget { target } => {
            OriginalRefutationScopeWire::WholeTarget { target }
        }
        RefutationScopeWire::TemporalClassCompletion {
            outer_atom,
            horizon_index,
            horizon,
            completion_digest,
            target,
        } => OriginalRefutationScopeWire::TemporalClassCompletion {
            outer_atom,
            horizon_index,
            horizon,
            completion_digest,
            target,
        },
        RefutationScopeWire::StaticClassCompletion { outer_atom, completion_digest, target } => {
            OriginalRefutationScopeWire::StaticClassCompletion {
                outer_atom,
                completion_digest,
                target,
            }
        }
        _ => return Err(IoError::Convert("scope is already descriptive".into())),
    };
    Ok(RefutationScopeWire::ScopedReport { scope: Box::new(scope), produced_by })
}

/// Encode an existing full-target report as descriptive metadata.
///
/// This never creates execution authority or a replay verdict.
///
/// # Errors
/// The report's source is not the producing program, its shape is invalid,
/// or its degrees of freedom cannot be represented.
pub fn whole_target_diagnostic_scope(
    report: &TargetRefutationReport,
    produced_by: [u8; 32],
) -> Result<RefutationScopeWire, IoError> {
    if report.target.source() != &produced_by {
        return Err(IoError::Convert("target report source differs from producing program".into()));
    }
    let scope = RefutationScopeWire::WholeTargetDiagnostic {
        produced_by,
        report: Box::new(WholeTargetDiagnosticWire {
            target: target_to_wire(&report.target),
            refuter: report.refuter().to_owned(),
            original: report.original.iter().map(|v| v.to_bits()).collect(),
            refuted: report.refuted.iter().map(|v| v.to_bits()).collect(),
            statistic_kind: report.statistic.kind.as_str().to_owned(),
            statistic_value: report.statistic.value.to_bits(),
            statistic_df: u64::try_from(report.statistic.df).map_err(|_| IoError::TooLarge)?,
            statistic_p_value: report.statistic.p_value.map(f64::to_bits),
            standardised_bias: report.standardised_bias.iter().map(|v| v.to_bits()).collect(),
            standardised_bias_norm: report.standardised_bias_norm.map(f64::to_bits),
            passed: report.passed,
            informative: report.informative,
            replicates: report.replicates,
            failure: report.failure.as_deref().map(str::to_owned),
        }),
    };
    antecedent_io_core::analysis_wire::validate_refutation_scope(&scope)?;
    Ok(scope)
}

/// Encode a report's descriptive scope; this does not serialize runtime authority.
#[must_use]
pub fn refutation_to_wire(r: &RefutationReport) -> RefutationReportWire {
    RefutationReportWire {
        refuter: r.refuter.to_string(),
        original_ate: r.original_ate,
        refuted_ate: r.refuted_ate,
        comparison: r.comparison,
        informative: r.informative,
        passed: r.passed,
        failure_condition: r.failure_condition.as_ref().map(ToString::to_string),
        replicates: r.replicates,
        scope: r.scope.as_ref().map(scope_to_wire),
    }
}

/// Decode descriptive metadata without recreating runtime authority.
///
/// # Errors
/// Unknown or inconsistent target identity, or a horizon index outside the native range.
pub fn refutation_from_wire(w: &RefutationReportWire) -> Result<RefutationReport, IoError> {
    let mut report = RefutationReport::new(
        Arc::from(w.refuter.as_str()),
        w.original_ate,
        w.refuted_ate,
        w.comparison,
        w.informative,
        w.passed,
        w.failure_condition.as_ref().map(|s| Arc::<str>::from(s.as_str())),
        w.replicates,
    );
    report.scope = w.scope.as_ref().map(scope_from_wire).transpose()?;
    Ok(report)
}
