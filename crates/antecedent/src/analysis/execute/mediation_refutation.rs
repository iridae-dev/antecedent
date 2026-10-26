// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of a mediation decomposition over a mixed graph, a graph class or the window of
//! a temporal graph.
//!
//! The total, natural direct and natural indirect effects are three functionals of one table
//! (see [`antecedent_validate::functional_mediation`]), so the ATE-shaped refuters and the
//! linear mediation refuters do not apply. The suite refits every identified component
//! together on perturbed tables: the sensitivity of the indirect effect to a violation of
//! sequential ignorability, a mediator placebo, and the stability of the three components
//! jointly under subsets and, under full validation, a joint bootstrap.
//!
//! Over a CPDAG or a PAG the class does not determine one functional, so the suite runs on
//! every distinct identified member (distinct by the value of the requested contrast, capped
//! at [`MAX_CLASS_MEMBERS`]), each member read through its own proofs of the three components.
//!
//! The state executes under both inference modes and is unlicensed: its verdicts are checks,
//! not a calibration claim, and no coverage record names it.

use std::sync::Arc;

use antecedent_core::{Intervention, MediationContrast, MediationQuery, VariableId};
use antecedent_estimate::util::SamplingDeclaration;
use antecedent_estimate::{FunctionalDistributionWorkspace, FunctionalEffect, RowResampling};
use antecedent_identify::IdentificationResult;
use antecedent_validate::RefutationReport;
use antecedent_validate::functional_mediation::{
    MediationComponent, MediationSuite, refute_mediation_components,
};

use super::mediation_mixed::{ClassStructure, component_slot};
use super::{
    CausalError, CausalQuery, EstimatorId, ExecutionContext, IdentificationEnvelope,
    IdentifiedEstimand, PhysicalExecutionPlan, RefuteSuite, TabularData,
    identification_status_ok_for_case, select_estimand,
};
use crate::planner::RouteDefaults;

/// Most distinct class members the suite refits one by one.
const MAX_CLASS_MEMBERS: usize = 8;

/// The three contrasts, in the order the suite reads them.
const CONTRASTS: [MediationContrast; 3] = [
    MediationContrast::Total,
    MediationContrast::NaturalDirect,
    MediationContrast::NaturalIndirect,
];

/// One identified component: its proof, its functional and the value the plug-in gives.
pub(super) type IdentifiedComponent = (IdentificationResult, IdentifiedEstimand, f64);

/// Active level minus control level of the treatment (one when either is not a number).
pub(super) fn contrast_width(query: &CausalQuery) -> f64 {
    let CausalQuery::Mediation(mediation) = query else { return 1.0 };
    match (&mediation.control, &mediation.active) {
        (Intervention::Set { value: control, .. }, Intervention::Set { value: active, .. }) => {
            match (control.as_f64(), active.as_f64()) {
                (Some(control), Some(active)) if (active - control).abs() > 0.0 => active - control,
                _ => 1.0,
            }
        }
        _ => 1.0,
    }
}

impl super::Study {
    /// The mediation suite on the components of one decomposition; empty for no validation.
    ///
    /// `roles` is the treatment, the outcome and then every mediator column the plug-in reads.
    ///
    /// # Errors
    ///
    /// A cancelled run, a table the perturbations cannot rebuild, or a declared sampling design
    /// the study's resampling refuses.
    pub(super) fn refute_mediation_decomposition(
        &self,
        data: &TabularData,
        components: &[Option<IdentifiedComponent>; 3],
        roles: &[VariableId],
        resampling: (RowResampling, SamplingDeclaration),
        ctx: &ExecutionContext,
    ) -> Result<Vec<RefutationReport>, CausalError> {
        if self.refute == RefuteSuite::None {
            return Ok(Vec::new());
        }
        let requested = match &self.query {
            CausalQuery::Mediation(query) => component_slot(query.contrast),
            _ => 0,
        };
        let [treatment, outcome, mediators @ ..] = roles else {
            return Err(CausalError::Compile {
                message: "mediation validation needs the treatment, outcome and mediator roles"
                    .into(),
            });
        };
        let window_span = match resampling.0 {
            RowResampling::WindowBlocks { span } => Some(span),
            _ => None,
        };
        let component = |slot: usize| {
            components[slot].as_ref().map(|(identification, estimand, published)| {
                MediationComponent { identification, estimand, published: *published }
            })
        };
        let suite = MediationSuite {
            data,
            components: [component(0), component(1), component(2)],
            requested,
            roles,
            treatment: *treatment,
            outcome: *outcome,
            mediators,
            contrast_width: contrast_width(&self.query),
            window_span,
            resampling: resampling.0,
            design: resampling.1,
            bootstrap_replicates: self.bootstrap_replicates,
        };
        refute_mediation_components(&suite, self.refute == RefuteSuite::Full, ctx)
            .map_err(CausalError::from)
    }

    /// The mediation suite over the members of a CPDAG or a PAG: each distinct identified
    /// member is refuted on its own proofs of the three components.
    ///
    /// # Errors
    ///
    /// The identifier or the plug-in failing on a member for a reason other than the contrast
    /// being unidentified or an empirical cell being empty, a cancelled run, or a declared
    /// sampling design the study's resampling refuses.
    pub(super) fn refute_mediation_class(
        &self,
        data: &TabularData,
        class: ClassStructure<'_>,
        query: &MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Vec<RefutationReport>, CausalError> {
        if self.refute == RefuteSuite::None {
            return Ok(Vec::new());
        }
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PATH)?;
        let requested = component_slot(query.contrast);
        let mut roles = vec![query.treatment, query.outcome];
        roles.extend(query.mediators.iter().copied());

        // The proof of each contrast in every case of the class, aligned by case index.
        let mut per_slot: Vec<Option<Vec<Option<IdentificationResult>>>> = Vec::new();
        for (slot, contrast) in CONTRASTS.iter().enumerate() {
            let mut component = query.clone();
            component.contrast = *contrast;
            let target = CausalQuery::Mediation(component);
            let cases = match class {
                ClassStructure::Cpdag(cpdag) => {
                    crate::strategy_table::identify_cpdag_path_specific_in_context(
                        identifier_id,
                        cpdag,
                        &target,
                        Some(ctx),
                    )
                    .map(|envelope| envelope_cases(&envelope))
                }
                ClassStructure::Pag(pag) => {
                    crate::strategy_table::identify_pag_path_specific_in_context(
                        identifier_id,
                        pag,
                        &target,
                        Some(ctx),
                    )
                    .map(|envelope| envelope_cases(&envelope))
                }
            };
            match cases {
                Ok(cases) => per_slot.push(Some(cases)),
                Err(error)
                    if slot != requested
                        && error.reason_code() == Some("effect_not_identified") =>
                {
                    per_slot.push(None);
                }
                Err(error) => return Err(error),
            }
        }
        let resampling = self.functional_row_resampling()?;
        self.refute_mediation_members(
            data,
            MemberProofs { per_slot: &per_slot, requested, estimator_id },
            &roles,
            resampling,
            ctx,
        )
    }

    /// The mediation suite on every distinct identified member of a class. `proofs` holds, per
    /// contrast, the proof of each case of the class aligned by case index (`None` where the
    /// contrast is not identified).
    ///
    /// # Errors
    ///
    /// The plug-in failing on a member for a reason other than an empty empirical cell, a
    /// cancelled run, or a declared sampling design the study's resampling refuses.
    pub(super) fn refute_mediation_members(
        &self,
        data: &TabularData,
        proofs: MemberProofs<'_>,
        roles: &[VariableId],
        resampling: (RowResampling, SamplingDeclaration),
        ctx: &ExecutionContext,
    ) -> Result<Vec<RefutationReport>, CausalError> {
        let MemberProofs { per_slot, requested, estimator_id } = proofs;
        let Some(requested_cases) = per_slot[requested].as_ref() else { return Ok(Vec::new()) };

        // One member per distinct value of the requested contrast.
        let mut members: Vec<(usize, [Option<IdentifiedComponent>; 3])> = Vec::new();
        let mut seen: Vec<f64> = Vec::new();
        for index in 0..requested_cases.len() {
            if members.len() == MAX_CLASS_MEMBERS {
                break;
            }
            // The requested contrast decides whether the case is a new member; the other
            // components are only fitted for a case that is.
            let Some(requested_result) = per_slot[requested]
                .as_ref()
                .and_then(|cases| cases.get(index))
                .and_then(Option::as_ref)
            else {
                continue;
            };
            let requested_estimand = select_estimand(requested_result, estimator_id)?;
            let Some(value) =
                plug_in_value(data, &requested_estimand, requested_result, roles, ctx)
            else {
                continue;
            };
            if seen.iter().any(|earlier| (earlier - value).abs() <= 1e-9 * earlier.abs().max(1.0)) {
                continue;
            }
            seen.push(value);
            let mut owned: [Option<IdentifiedComponent>; 3] = [None, None, None];
            for (slot, cases) in per_slot.iter().enumerate() {
                if slot == requested {
                    owned[slot] =
                        Some((requested_result.clone(), requested_estimand.clone(), value));
                    continue;
                }
                let Some(result) =
                    cases.as_ref().and_then(|cases| cases.get(index)).and_then(Option::as_ref)
                else {
                    continue;
                };
                let estimand = select_estimand(result, estimator_id)?;
                if let Some(other) = plug_in_value(data, &estimand, result, roles, ctx) {
                    owned[slot] = Some((result.clone(), estimand, other));
                }
            }
            members.push((index, owned));
        }

        let several = members.len() > 1;
        let mut reports = Vec::new();
        for (index, owned) in &members {
            let mut member =
                self.refute_mediation_decomposition(data, owned, roles, resampling, ctx)?;
            if several {
                for report in &mut member {
                    report.refuter = Arc::from(format!("{}.case{index}", report.refuter));
                }
            }
            reports.extend(member);
        }
        Ok(reports)
    }
}

/// The proofs of the cases of a class and the contrast the study asked for.
#[derive(Clone, Copy)]
pub(super) struct MemberProofs<'a> {
    /// Per contrast (total, direct, indirect), the proof of each case; `None` where the
    /// contrast is not identified in the class.
    pub(super) per_slot: &'a [Option<Vec<Option<IdentificationResult>>>],
    /// Slot of the requested contrast.
    pub(super) requested: usize,
    /// The estimator the functionals are read for.
    pub(super) estimator_id: EstimatorId,
}

/// The proof of every case that identifies and carries an estimand.
pub(super) fn envelope_cases<G>(
    envelope: &IdentificationEnvelope<G>,
) -> Vec<Option<IdentificationResult>> {
    envelope
        .cases
        .iter()
        .map(|case| {
            (identification_status_ok_for_case(case.result.status)
                && !case.result.estimands.is_empty())
            .then(|| case.result.clone())
        })
        .collect()
}

/// The plug-in value of one functional on `data`; `None` where a required empirical cell is
/// empty.
pub(super) fn plug_in_value(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    identification: &IdentificationResult,
    roles: &[VariableId],
    ctx: &ExecutionContext,
) -> Option<f64> {
    let estimator = FunctionalEffect::new();
    let prepared = estimator
        .prepare(
            data,
            estimand,
            &identification.arena,
            identification.required_assumptions.clone(),
            roles,
        )
        .ok()?;
    let estimate =
        estimator.estimate(&prepared, &mut FunctionalDistributionWorkspace::default(), ctx).ok()?;
    estimate.ate.is_finite().then_some(estimate.ate)
}
