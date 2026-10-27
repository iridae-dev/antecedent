// SPDX-License-Identifier: MIT OR Apache-2.0
//! Natural mediation on one series under a temporal PAG.
//!
//! The template is unfolded over the window a pulse and its outcome horizon need, each
//! stationary MAG completion is marginalized onto that window (a cause that precedes the
//! window stays a bidirected edge between its in-window effects) and read through the ADMGs
//! that share its MAG, as a static PAG is: the total, natural direct and natural indirect
//! effects are identified only when every candidate of every completion identifies them (see
//! [`antecedent_identify::temporal_pag_mediation_window`]). The decomposition is estimated
//! on the lag-aligned window table, built straight from the schema's layout (there is no
//! unrolled ADMG to read it from), with circular blocks of consecutive windows (never shorter
//! than the window) or one Dirichlet weight per block.
//!
//! When every component is identified by one functional the components are evaluated at the
//! same resampled tables and the result discloses their joint uncertainty. When a component's
//! completions identify different functionals its value is the identified set over them, read
//! through the class estimator of the static PAG route on the same window table, and the
//! components are joined by their Bayesian-bootstrap draws only.

use antecedent_core::{MediationContrast, MediationQuery, VariableId};
use antecedent_data::TableView;
use antecedent_estimate::TemporalMediationEstimate;
use antecedent_graph::TemporalPag;
use antecedent_identify::{
    GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier, TemporalClassEnvelope,
    TemporalPagMediationWindow, temporal_pag_mediation_window,
};

use super::class_path_specific::ClassPathRun;
use super::mediation_mixed::{ComponentRun, component_slot, joint_diagnostic};
use super::mediation_refutation::{MemberProofs, envelope_cases};
use super::pag_path::{ClassTag, pag_envelope_diagnostic};
use super::temporal_admg_mediation::window_effect_query;
use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, EstimatorId, ExecutionContext,
    GraphRef, IdentificationStatus, Instant, PathSpecificMixedSource, PhysicalExecutionPlan,
    StudyResult, TimeSeriesData, identification_status_ok_for_case, report_identify_compute,
    select_estimand,
};
use crate::analysis::temporal_admg_contrast::window_table_over;
use crate::planner::RouteDefaults;

/// Stable code of the disclosure every temporal PAG mediation window carries.
const WINDOW_DISCLOSURE: &str = "estimate.temporal_pag.window";

/// The component contrasts, in publication order, with the names their draws carry.
const CONTRASTS: [(MediationContrast, &str); 3] = [
    (MediationContrast::Total, "total"),
    (MediationContrast::NaturalDirect, "direct"),
    (MediationContrast::NaturalIndirect, "indirect"),
];

/// The identified window, component proofs and uncertainty declaration used by one execution.
struct PagMediationRun<'a> {
    table: &'a super::TabularData,
    windows: &'a [TemporalPagMediationWindow],
    envelopes: &'a [Option<TemporalClassEnvelope>],
    requested: usize,
    requested_envelope: &'a TemporalClassEnvelope,
    query: &'a MediationQuery,
    strategy: (super::IdentifierId, EstimatorId),
    resampling:
        (antecedent_estimate::RowResampling, antecedent_estimate::util::SamplingDeclaration),
    started: Instant,
    disclosure: Diagnostic,
}

/// Whether every completion of the envelope is identified (no unidentified mass).
fn fully_identified(envelope: &TemporalClassEnvelope) -> bool {
    let envelope = &envelope.envelope;
    envelope.unidentified_weight.0 <= 0.0
        && envelope.identified_weight.0 > 0.0
        && !matches!(envelope.status, IdentificationStatus::NotIdentified)
}

/// The disclosure of how the window was read: the premises a window contrast adds to the ones
/// the identification carries, stated where a reader of the result will find them.
fn window_disclosure(
    window: &TemporalPagMediationWindow,
    rows: usize,
    block_length: Option<usize>,
    cases: usize,
) -> Diagnostic {
    let periods = window.indexer.history() + window.indexer.horizon();
    let block = block_length.map_or_else(|| "none".to_owned(), |length| length.to_string());
    Diagnostic::new(
        WINDOW_DISCLOSURE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "the mediation was identified on the {periods}-period unfolding of the temporal PAG \
             (each stationary MAG completion marginalized onto the window by the inducing-path \
             rule, so a cause that precedes the window stays a bidirected edge between its \
             in-window effects, and read through the ADMGs that share its MAG) and evaluated on \
             {rows} overlapping windows of the series by a plug-in of the identified functional. \
             Every window is read as a draw from the same law, so the series is assumed \
             stationary over the template, and history deeper than one further template lag is \
             the initial state. Windows overlap, so uncertainty resamples circular blocks of \
             {block} consecutive windows (never shorter than the window), the declared \
             dependent-series design; the result is unlicensed and no coverage record names it"
        ),
    )
    .with_fields([
        ("periods", periods.to_string()),
        ("history", window.indexer.history().to_string()),
        ("windows", rows.to_string()),
        ("block_length", block),
        ("cases", cases.to_string()),
    ])
}

impl super::Study {
    /// Identify one component on the window of `template`.
    fn identify_temporal_pag_window(
        &self,
        template: &TemporalPag,
        window: &TemporalPagMediationWindow,
        ctx: &ExecutionContext,
    ) -> Result<TemporalClassEnvelope, CausalError> {
        let mut config =
            GeneralizedAdjustmentConfig { context: Some(ctx.clone()), ..Default::default() };
        if let Some(max) = self.declared.max_completions {
            config.max_completions = max;
        }
        GeneralizedAdjustmentIdentifier { config }
            .identify_temporal_pag_path_specific_envelope(template, window)
            .map_err(CausalError::from)
    }

    /// Identify the total, direct and indirect components on their windows. The requested
    /// component must be identified in every candidate of every completion; another may be
    /// left unidentified.
    ///
    /// # Errors
    ///
    /// The requested contrast is not identified, or an identifier refusal other than
    /// non-identification of an unrequested component.
    pub(crate) fn identify_pag_window_components(
        &self,
        template: &TemporalPag,
        windows: &[TemporalPagMediationWindow],
        requested: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Option<TemporalClassEnvelope>>, CausalError> {
        let mut envelopes: Vec<Option<TemporalClassEnvelope>> = Vec::new();
        for (slot, window) in windows.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_IDENTIFY,
                });
            }
            let envelope = match self.identify_temporal_pag_window(template, window, ctx) {
                Ok(envelope) if fully_identified(&envelope) => Some(envelope),
                Ok(envelope) if slot == requested => {
                    return Err(CausalError::not_identified(
                        envelope.envelope.status,
                        envelope.envelope.truncated_completions > 0,
                        "temporal PAG mediation not identified: an ADMG sharing the \
                         m-separation model of a MAG completion of the window leaves the \
                         requested contrast unidentified, so no functional is correct for every \
                         model the TemporalPag represents",
                    ));
                }
                Ok(_) => None,
                Err(error)
                    if slot != requested
                        && error.reason_code() == Some("effect_not_identified") =>
                {
                    None
                }
                Err(error) => return Err(error),
            };
            envelopes.push(envelope);
        }
        Ok(envelopes)
    }

    /// The window certificates compile retains with the program, when the window
    /// identifies; `None` leaves identification (and its refusal) to execution.
    pub(crate) fn retain_pag_window_identification(
        &self,
        variable_count: u32,
        ctx: &ExecutionContext,
    ) -> Result<Option<crate::planner::lowered::RetainedExecutionProof>, CausalError> {
        let inputs = (|| {
            let GraphRef::TemporalPag(template) = self.graph.graph_ref() else {
                return None;
            };
            let antecedent_core::CausalQuery::Mediation(query) = &self.query else {
                return None;
            };
            let effect = window_effect_query(query).ok()?;
            let base =
                temporal_pag_mediation_window(template, &effect, query, variable_count).ok()?;
            let windows: Vec<TemporalPagMediationWindow> = CONTRASTS
                .iter()
                .map(|(contrast, _)| {
                    let mut window = base.clone();
                    window.query.contrast = *contrast;
                    window
                })
                .collect();
            Some((template, windows, component_slot(query.contrast)))
        })();
        let Some((template, windows, requested)) = inputs else {
            return Ok(None);
        };
        match self.identify_pag_window_components(template, &windows, requested, ctx) {
            Ok(envelopes) => {
                Ok(Some(crate::planner::lowered::RetainedExecutionProof::Pag { envelopes }))
            }
            Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. })) => {
                Err(error)
            }
            Err(_) => Ok(None),
        }
    }

    /// Identify and estimate the mediation decomposition on the unfolded window of a
    /// `TemporalPag`.
    ///
    /// # Errors
    ///
    /// The requested contrast is not identified in every candidate of every completion (the
    /// refusal names the status and whether the search was capped), a custom validator was
    /// attached, the series holds a gap inside a window, or more than one outcome horizon was
    /// requested.
    pub(super) fn execute_temporal_pag_mediation(
        &self,
        series: &TimeSeriesData,
        query: &MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if !self.declared.custom_validators.is_empty() {
            return Err(crate::unsupported_reason!(
                "validators_not_applicable",
                "a custom validator refutes one scalar average effect and none applies to a \
                 mediation decomposition on the window of a temporal PAG"
            ));
        }
        let started = Instant::now();
        let GraphRef::TemporalPag(template) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "temporal PAG mediation requires a supplied TemporalPag".into(),
            });
        };
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PATH)?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "temporal PAG mediation requires estimator functional.effect; got \
                     {estimator_id}"
                ),
            });
        }
        let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
            CausalError::Compile { message: "series schema exceeds the variable id range".into() }
        })?;
        let effect = window_effect_query(query)?;
        let base = temporal_pag_mediation_window(template, &effect, query, variable_count)
            .map_err(CausalError::from)?;

        let requested = component_slot(query.contrast);
        let windows: Vec<TemporalPagMediationWindow> = CONTRASTS
            .iter()
            .map(|(contrast, _)| {
                let mut window = base.clone();
                window.query.contrast = *contrast;
                window
            })
            .collect();
        // The window is identified once, at compile, and the program retains the
        // certificates; a plan that retained none identifies here.
        let envelopes =
            if let Some(crate::planner::lowered::RetainedExecutionProof::Pag { envelopes }) =
                crate::analysis::lowering::plan_retained_execution_proof(self, physical)
            {
                envelopes.clone()
            } else {
                report_identify_compute(ctx);
                self.identify_pag_window_components(template, &windows, requested, ctx)?
            };
        let requested_envelope = envelopes[requested].clone().ok_or_else(|| {
            CausalError::Compile { message: "requested contrast has no proof".into() }
        })?;

        let periods = base.indexer.history() + base.indexer.horizon();
        let variables: Vec<VariableId> = (0..variable_count).map(VariableId::from_raw).collect();
        let mut used: Vec<VariableId> = template
            .nodes()
            .iter()
            .filter_map(|node| match node {
                antecedent_core::NodeRef::Lagged { variable, .. } => Some(*variable),
                _ => None,
            })
            .chain([query.treatment, query.outcome])
            .chain(query.mediators.iter().copied())
            .collect();
        used.sort_unstable();
        used.dedup();
        let table = window_table_over(series, &variables, periods, &used, &ctx.kernel_policy)?;
        let (resampling, design) = self.window_row_resampling(periods)?;
        let disclosure = window_disclosure(
            &base,
            table.row_count(),
            resampling.block_length(table.row_count()),
            requested_envelope.envelope.cases.len(),
        );
        // Every present component identified by one functional: evaluate them at the same
        // resampled tables.
        let run = PagMediationRun {
            table: &table,
            windows: &windows,
            envelopes: &envelopes,
            requested,
            requested_envelope: &requested_envelope,
            query,
            strategy: (identifier_id, estimator_id),
            resampling: (resampling, design),
            started,
            disclosure,
        };
        if envelopes.iter().flatten().all(|found| found.envelope.invariant.is_some()) {
            return self.estimate_pag_invariant_components(run, physical, ctx);
        }
        self.estimate_pag_class_components(run, physical, ctx)
    }
    fn temporal_pag_certificate(
        &self,
        envelope: &TemporalClassEnvelope,
        identifier: super::IdentifierId,
    ) -> crate::Identification {
        crate::Identification::TemporalEnvelope {
            envelope: envelope.clone(),
            strategy: identifier,
            structure_version: self.graph.version(),
        }
    }

    /// Common functional components share the same resampled window tables.
    fn estimate_pag_invariant_components(
        &self,
        run: PagMediationRun<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let PagMediationRun {
            table,
            windows,
            envelopes,
            requested,
            requested_envelope,
            query,
            strategy: (identifier_id, estimator_id),
            resampling: (resampling, design),
            started,
            disclosure,
        } = run;
        let mut identified = Vec::new();
        for entry in envelopes {
            identified.push(match entry {
                Some(found) => {
                    let case = found
                        .envelope
                        .cases
                        .iter()
                        .find(|case| {
                            identification_status_ok_for_case(case.result.status)
                                && !case.result.estimands.is_empty()
                        })
                        .ok_or_else(|| CausalError::Compile {
                            message: "an identified temporal PAG envelope has no case that \
                                          carries an estimand"
                                .into(),
                        })?;
                    let mut result = case.result.clone();
                    result.status = found.envelope.status;
                    let estimand = select_estimand(&result, estimator_id)?;
                    Some((result, estimand))
                }
                None => None,
            });
        }
        let mut wq_roles =
            vec![windows[requested].query.treatment, windows[requested].query.outcome];
        wq_roles.extend(windows[requested].query.mediators.iter().copied());
        let result = self.estimate_components(
            table,
            ComponentRun {
                identified,
                requested,
                roles: wq_roles,
                reported: (query.treatment, query.outcome),
                resampling: (resampling, design),
                structure: "unfolded temporal PAG window",
                strategy: (identifier_id, estimator_id),
                started,
                extra_diagnostics: vec![
                    disclosure,
                    pag_envelope_diagnostic(&requested_envelope.envelope),
                ],
            },
            physical,
            ctx,
        )?;
        Ok(self.attach_certificate(
            result,
            self.temporal_pag_certificate(requested_envelope, identifier_id),
        ))
    }

    /// Disagreeing functionals retain class sets and their component-native refutations.
    fn estimate_pag_class_components(
        &self,
        run: PagMediationRun<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let PagMediationRun {
            table,
            windows,
            envelopes,
            requested,
            requested_envelope: _,
            query: _,
            strategy: (identifier_id, estimator_id),
            resampling: (resampling, design),
            started,
            disclosure,
        } = run;
        // Completions that identify different functionals: each component is the class result
        // of the static PAG route on the window table.
        let mut results: Vec<Option<StudyResult>> = Vec::new();
        for (slot, entry) in envelopes.iter().enumerate() {
            results.push(match entry {
                Some(found) => Some(self.finish_class_path_specific(
                    table,
                    PathSpecificMixedSource::Mediation(&windows[slot].query),
                    ClassPathRun {
                        envelope: &found.envelope,
                        class: ClassTag::TemporalPag,
                        envelope_diagnostic: pag_envelope_diagnostic(&found.envelope),
                        certificate: self.temporal_pag_certificate(found, identifier_id),
                        identifier_id,
                        estimator_id,
                        started,
                        resampling: Some((resampling, design)),
                    },
                    physical,
                    ctx,
                )?),
                None => None,
            });
        }
        let mut values: [Option<f64>; 3] = [None; 3];
        let mut columns: Vec<(&str, Vec<f64>)> = Vec::new();
        for (slot, entry) in results.iter().enumerate() {
            if let Some(found) = entry {
                values[slot] = Some(found.effect());
                if let Some(posterior) = found.posterior.as_ref() {
                    columns.push((CONTRASTS[slot].1, posterior.draws.values.to_vec()));
                }
            }
        }
        let mut result = results[requested].take().ok_or_else(|| CausalError::Compile {
            message: "requested contrast has no estimate".into(),
        })?;
        result.diagnostics.push(disclosure);
        let paired: Vec<(&str, &[f64])> =
            columns.iter().map(|(name, draws)| (*name, draws.as_slice())).collect();
        if let Some((_, first)) = paired.first() {
            if paired.len() > 1 && paired.iter().all(|(_, draws)| draws.len() == first.len()) {
                result.diagnostics.push(joint_diagnostic(
                    &paired,
                    first.len(),
                    "Bayesian-bootstrap",
                    "unfolded temporal PAG window",
                ));
            }
        }
        if let Some(effect) = result.estimate.as_effect() {
            result.payload = crate::result::ResultPayload::Mediation {
                decomposition: Some(TemporalMediationEstimate {
                    effect: effect.clone(),
                    total: values[0],
                    direct: values[1],
                    mediated: values[2],
                }),
                grid: None,
            };
        }
        // The class members that identify different functionals are refuted one by one, each
        // on its own proofs of the three components.
        let per_slot: Vec<Option<Vec<Option<antecedent_identify::IdentificationResult>>>> =
            envelopes
                .iter()
                .map(|entry| entry.as_ref().map(|found| envelope_cases(&found.envelope)))
                .collect();
        let mut member_roles =
            vec![windows[requested].query.treatment, windows[requested].query.outcome];
        member_roles.extend(windows[requested].query.mediators.iter().copied());
        result.refutations = self.refute_mediation_members(
            table,
            MemberProofs { per_slot: &per_slot, requested, estimator_id },
            &member_roles,
            (resampling, design),
            ctx,
        )?;
        Ok(result)
    }
}
