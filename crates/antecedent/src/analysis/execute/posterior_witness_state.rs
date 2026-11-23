// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-atom Pulse-witness refuter states of a temporal derivative over a graph posterior.
//!
//! A posterior atom is a stationary template (or a class of them); its derivative is refuted by
//! the same Pulse-witness state the explicit cell of the atom's kind runs, against that atom's
//! own response and its own identified completions. The reports of one refuter are then mixed by
//! the posterior weight of the atoms that produced them, a mixed verdict passes only if every
//! contributing atom passes, and a refuter that ran on only part of the contributing mass says
//! so (`refute.envelope.partial_coverage`).

use super::temporal_response_refutation::{PulseWitnessSite, refute_temporal_derivative};
use super::{
    CausalError, CausalResponse, Diagnostic, DiagnosticKind, DiagnosticSeverity, EstimatorId,
    ExecutionContext, PredictiveCheckReport, ResponseQuery, TemporalEffectQuery, TimeSeriesData,
    identification_status_ok_for_case, mix_atom_refutation_reports, select_estimand,
};

/// Accumulated per-atom refuter states of one posterior.
#[derive(Default)]
pub(super) struct AtomWitnessStates {
    per_atom: Vec<(f64, Vec<antecedent_validate::RefutationReport>)>,
    checks: Vec<PredictiveCheckReport>,
    notes: Vec<Diagnostic>,
    listed: Vec<String>,
}

impl AtomWitnessStates {
    /// Run the derivative state of one atom (its own response and witness sites) and keep its
    /// reports under the atom's posterior weight.
    ///
    /// # Errors
    ///
    /// As [`refute_temporal_derivative`].
    pub(super) fn push_derivative(
        &mut self,
        study: &super::Study,
        data: &TimeSeriesData,
        atom: (u64, f64),
        (query, response): (&ResponseQuery, &CausalResponse),
        sites: &[PulseWitnessSite],
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let (reports, checks, notes) =
            refute_temporal_derivative(study, data, query, response, sites, ctx)?;
        self.listed.push(format!("{:x}:weight={}:reports={}", atom.0, atom.1, reports.len()));
        self.per_atom.push((atom.1, reports));
        self.checks.extend(checks);
        self.notes.extend(notes);
        Ok(())
    }

    /// The mixed reports, the atoms' predictive checks and the disclosures.
    #[must_use]
    pub(super) fn finish(
        self,
    ) -> (Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>, Vec<Diagnostic>)
    {
        let (reports, coverage) = mix_atom_refutation_reports(&self.per_atom);
        let mut notes = self.notes;
        notes.push(Diagnostic::new(
            "refute.graph_posterior.per_atom",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "cheap/full validation ran the temporal derivative refuter state on each \
                 contributing atom against that atom's own response; reports of one refuter are \
                 mixed by posterior weight and pass only if every contributing atom passes \
                 (atoms: {})",
                self.listed.join(",")
            ),
        ));
        notes.extend(coverage);
        (reports, self.checks, notes)
    }
}

/// The Pulse witnesses of one class envelope at `horizon`: one per identified completion, with
/// its normalized completion weight.
///
/// # Errors
///
/// A completion whose estimand cannot be selected.
pub(super) fn class_completion_sites(
    bundle: &antecedent_identify::TemporalClassEnvelope,
    probe: &TemporalEffectQuery,
    horizon: u32,
) -> Result<Vec<PulseWitnessSite>, CausalError> {
    let mut sites = Vec::new();
    for (case_index, (case, indexer)) in
        bundle.envelope.cases.iter().zip(bundle.indexers.iter()).enumerate()
    {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        sites.push(PulseWitnessSite {
            key: super::temporal_path::class_completion_key(0, case_index),
            horizon,
            weight: case.weight.0,
            witness: probe.clone(),
            estimand,
            indexer: indexer.clone(),
            status: case.result.status,
            assumptions: case.result.required_assumptions.clone(),
        });
    }
    Ok(sites)
}
