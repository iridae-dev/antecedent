// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mediation decomposition over a mixed graph or a graph class.
//!
//! The total, natural direct and natural indirect effects are three functionals of the
//! observed law, each identified by the path-specific identifier over the structure (the
//! surgical total effect, the edge g-formula or Shpitser's district formula, whichever the
//! latent confounding calls for). They are estimated by the finite-discrete plug-in on the
//! same rows, so a frequentist bootstrap replicate (or a Bayesian-bootstrap draw) evaluates
//! all three at the same resampled table: their uncertainty is joint, and the covariance
//! between the components is published beside the requested contrast.
//!
//! The pure natural indirect effect is identified only when no recanting district stands in
//! the way; a component the identifier refuses is reported absent, not imputed as
//! `total - direct`. The requested contrast must be identified.
//!
//! Over a CPDAG or a PAG each component is read through the class envelope of the
//! path-specific executors: the requested contrast carries the class result, and each other
//! component is the class result of its own contrast, so a component that is only partially
//! identified keeps its own identified set.

use antecedent_core::{MediationContrast, MediationQuery, VariableId};
use antecedent_estimate::{FunctionalEffect, RowResampling, TemporalMediationEstimate};
use antecedent_graph::Admg;

use super::{
    CausalError, CausalPosterior, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, ExecutionContext, IdentificationResult, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, Instant,
    PathSpecificMixedSource, PhysicalExecutionPlan, Strategy, StudyResult, TabularData, elapsed_ns,
    identify_admg_query, report_identify_compute, run_functional, select_estimand,
};
use crate::planner::RouteDefaults;

/// Aligned preparation and fitted uncertainty for every mediation component slot.
struct ComponentFits {
    prepared: Vec<Option<antecedent_estimate::PreparedFunctionalEffect>>,
    estimates: Vec<Option<(EffectEstimate, Option<CausalPosterior>)>>,
}

/// Stable code of the joint-uncertainty disclosure.
const JOINT_CODE: &str = "estimate.mediation.joint";

/// The component contrasts, in publication order.
const COMPONENTS: [(MediationContrast, &str); 3] = [
    (MediationContrast::Total, "total"),
    (MediationContrast::NaturalDirect, "direct"),
    (MediationContrast::NaturalIndirect, "indirect"),
];

/// The slot a requested contrast reads its value from.
pub(super) fn component_slot(contrast: MediationContrast) -> usize {
    match contrast {
        MediationContrast::Total => 0,
        MediationContrast::Direct | MediationContrast::NaturalDirect => 1,
        MediationContrast::Mediated | MediationContrast::NaturalIndirect => 2,
    }
}

/// Mean, standard deviation and pairwise correlations of paired draws, as disclosure fields.
fn joint_fields(columns: &[(&str, &[f64])]) -> Vec<(&'static str, String)> {
    fn moments(values: &[f64]) -> (f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = if values.len() > 1 {
            values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)
        } else {
            f64::NAN
        };
        (mean, variance.sqrt())
    }
    let mut fields: Vec<(&'static str, String)> = Vec::new();
    let names: [(&str, &'static str, &'static str); 3] = [
        ("total", "total_mean", "total_sd"),
        ("direct", "direct_mean", "direct_sd"),
        ("indirect", "indirect_mean", "indirect_sd"),
    ];
    for (name, mean_key, sd_key) in names {
        if let Some((_, values)) = columns.iter().find(|(column, _)| *column == name) {
            let (mean, sd) = moments(values);
            fields.push((mean_key, format!("{mean}")));
            fields.push((sd_key, format!("{sd}")));
        }
    }
    let pairs: [(&str, &str, &'static str); 3] = [
        ("total", "direct", "corr_total_direct"),
        ("total", "indirect", "corr_total_indirect"),
        ("direct", "indirect", "corr_direct_indirect"),
    ];
    for (a, b, key) in pairs {
        let left = columns.iter().find(|(column, _)| *column == a);
        let right = columns.iter().find(|(column, _)| *column == b);
        if let (Some((_, x)), Some((_, y))) = (left, right) {
            if x.len() == y.len() && x.len() > 1 {
                let (mx, sx) = moments(x);
                let (my, sy) = moments(y);
                let n = x.len() as f64;
                let covariance =
                    x.iter().zip(*y).map(|(u, v)| (u - mx) * (v - my)).sum::<f64>() / (n - 1.0);
                fields.push((key, format!("{}", covariance / (sx * sy))));
            }
        }
    }
    fields
}

pub(super) fn joint_diagnostic(
    columns: &[(&str, &[f64])],
    draws: usize,
    kind: &str,
    graph: &str,
) -> Diagnostic {
    let mut fields = joint_fields(columns);
    fields.push(("draws", draws.to_string()));
    fields.push(("draw_kind", kind.to_owned()));
    Diagnostic::new(
        JOINT_CODE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "the total, natural direct and natural indirect effects over the {graph} were \
             evaluated on the same {draws} {kind} draws, so their means, standard deviations \
             and correlations are joint; a component the identifier refuses is absent"
        ),
    )
    .with_fields(fields)
}

/// Identify the three components of `query` over `admg`; a component other than the requested
/// one that the identifier refuses as not identified is `None`.
///
/// # Errors
///
/// The requested contrast is not identified, or the proof has no estimand the plug-in reads.
pub(super) fn identify_components(
    admg: &Admg,
    query: &MediationQuery,
    identifier_id: IdentifierId,
    estimator_id: EstimatorId,
) -> Result<Vec<Option<(IdentificationResult, IdentifiedEstimand)>>, CausalError> {
    let requested = component_slot(query.contrast);
    let mut identified = Vec::new();
    for (slot, (contrast, _)) in COMPONENTS.iter().enumerate() {
        let mut component = query.clone();
        component.contrast = *contrast;
        match identify_admg_query(identifier_id, admg, &CausalQuery::Mediation(component)) {
            Ok(identification) => {
                let estimand = select_estimand(&identification, estimator_id)?;
                identified.push(Some((identification, estimand)));
            }
            Err(error)
                if slot != requested && error.reason_code() == Some("effect_not_identified") =>
            {
                identified.push(None);
            }
            Err(error) => return Err(error),
        }
    }
    Ok(identified)
}

/// What the estimation of the identified components reads and reports.
pub(super) struct ComponentRun<'a> {
    /// One proof per component (total, direct, indirect); `None` where not identified.
    pub(super) identified: Vec<Option<(IdentificationResult, IdentifiedEstimand)>>,
    /// Slot of the requested contrast.
    pub(super) requested: usize,
    /// Columns the functionals read.
    pub(super) roles: Vec<VariableId>,
    /// Treatment and outcome the result reports.
    pub(super) reported: (VariableId, VariableId),
    /// How the rows are resampled.
    pub(super) resampling: (RowResampling, antecedent_estimate::util::SamplingDeclaration),
    /// Name of the structure the disclosure speaks of.
    pub(super) structure: &'a str,
    /// Strategy the result records.
    pub(super) strategy: (IdentifierId, EstimatorId),
    /// When the execution began.
    pub(super) started: Instant,
    /// Disclosures the caller adds.
    pub(super) extra_diagnostics: Vec<Diagnostic>,
}

impl super::Study {
    /// Identify the mediation decomposition over an ADMG and estimate its components jointly.
    ///
    /// # Errors
    ///
    /// A requested contrast the identifier refuses (the refusal names the recanting
    /// district), a plug-in that cannot evaluate a required empirical cell, or a validation
    /// suite that cannot be refit.
    pub(super) fn execute_mediation_mixed(
        &self,
        data: &TabularData,
        admg: &Admg,
        query: &MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        query.validate()?;
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PATH)?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "mediation over a mixed graph requires estimator functional.effect; got \
                     {estimator_id}"
                ),
            });
        }
        let mut roles = vec![query.treatment, query.outcome];
        roles.extend(query.mediators.iter().copied());
        report_identify_compute(ctx);
        let identified = identify_components(admg, query, identifier_id, estimator_id)?;
        self.estimate_components(
            data,
            ComponentRun {
                identified,
                requested: component_slot(query.contrast),
                roles,
                reported: (query.treatment, query.outcome),
                resampling: self.functional_row_resampling()?,
                structure: "structure",
                strategy: (identifier_id, estimator_id),
                started,
                extra_diagnostics: Vec::new(),
            },
            physical,
            ctx,
        )
    }

    /// Estimate the identified components on the same rows and publish the decomposition with
    /// the joint disclosure.
    ///
    /// # Errors
    ///
    /// A plug-in that cannot evaluate a required empirical cell.
    pub(super) fn estimate_components(
        &self,
        data: &TabularData,
        run: ComponentRun<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let ComponentRun {
            mut identified,
            requested,
            roles,
            reported: (treatment, outcome),
            resampling: (row_resampling, sampling_design),
            structure,
            strategy: (identifier_id, estimator_id),
            started,
            extra_diagnostics,
        } = run;
        let fitter = FunctionalEffect {
            bootstrap_replicates: self.settings.bootstrap_replicates,
            ..FunctionalEffect::new()
        }
        .with_row_resampling(row_resampling, sampling_design);
        let ComponentFits { prepared, mut estimates } =
            self.fit_mediation_components(data, &fitter, &identified, &roles, ctx)?;

        // Paired draws: the bootstrap replicates (frequentist) or the Dirichlet draws
        // (Bayesian) of every identified component, evaluated at the same resampled tables.
        let mut diagnostics = extra_diagnostics;
        diagnostics.extend(
            self.component_joint_diagnostics(&fitter, &prepared, &estimates, structure, ctx)?,
        );

        // The mediation-native suite refits every identified component together; it reads the
        // proofs before the requested one is taken out below.
        let refutations = if self.settings.refute == super::RefuteSuite::None {
            Vec::new()
        } else {
            let components: [Option<super::mediation_refutation::IdentifiedComponent>; 3] =
                std::array::from_fn(|slot| {
                    identified[slot].as_ref().zip(estimates[slot].as_ref()).map(
                        |((identification, estimand), (estimate, _))| {
                            (identification.clone(), estimand.clone(), estimate.ate)
                        },
                    )
                });
            self.refute_mediation_decomposition(
                data,
                &components,
                &roles,
                (row_resampling, sampling_design),
                ctx,
            )?
        };

        let value = |slot: usize| estimates[slot].as_ref().map(|(estimate, _)| estimate.ate);
        let (total, direct, mediated) = (value(0), value(1), value(2));
        let (estimate, posterior) = estimates[requested].take().ok_or_else(|| {
            CausalError::Compile { message: "requested contrast has no estimate".into() }
        })?;
        let (mut identification, estimand) = identified[requested].take().ok_or_else(|| {
            CausalError::Compile { message: "requested contrast has no proof".into() }
        })?;
        identification.query = self.query.clone();
        let decomposition =
            TemporalMediationEstimate { effect: estimate.clone(), total, direct, mediated };
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(false)
            .mediation(Some(decomposition))
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        ))
    }

    fn fit_mediation_components(
        &self,
        data: &TabularData,
        fitter: &FunctionalEffect,
        identified: &[Option<(IdentificationResult, IdentifiedEstimand)>],
        roles: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<ComponentFits, CausalError> {
        let mut prepared = Vec::new();
        for entry in identified {
            prepared.push(match entry {
                Some((identification, estimand)) => Some(
                    fitter
                        .prepare(
                            data,
                            estimand,
                            &identification.arena,
                            identification.required_assumptions.clone(),
                            roles,
                        )
                        .map_err(CausalError::from)?,
                ),
                None => None,
            });
        }
        let refs: Vec<&antecedent_estimate::PreparedFunctionalEffect> =
            prepared.iter().flatten().collect();
        fitter.assert_same_rows(&refs).map_err(CausalError::from)?;

        let mut estimates: Vec<Option<(EffectEstimate, Option<CausalPosterior>)>> = Vec::new();
        for (entry, prepared) in identified.iter().zip(&prepared) {
            estimates.push(match (entry, prepared) {
                (Some((identification, _)), Some(prepared)) => Some(run_functional(
                    fitter,
                    prepared,
                    &self.settings.inference,
                    identification.status,
                    ctx,
                )?),
                _ => None,
            });
        }

        Ok(ComponentFits { prepared, estimates })
    }

    /// Publish the paired component law on the same draws or resampled tables.
    fn component_joint_diagnostics(
        &self,
        fitter: &FunctionalEffect,
        prepared: &[Option<antecedent_estimate::PreparedFunctionalEffect>],
        estimates: &[Option<(EffectEstimate, Option<CausalPosterior>)>],
        structure: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Diagnostic>, CausalError> {
        let mut diagnostics = Vec::new();
        let present: Vec<usize> =
            (0..COMPONENTS.len()).filter(|&i| estimates[i].is_some()).collect();
        if self.settings.inference.is_bayesian() {
            let columns: Vec<(&str, &[f64])> = present
                .iter()
                .filter_map(|&i| {
                    estimates[i]
                        .as_ref()
                        .and_then(|(_, posterior)| posterior.as_ref())
                        .map(|p| (COMPONENTS[i].1, p.draws.values.as_ref()))
                })
                .collect();
            if let Some((_, first)) = columns.first() {
                if columns.iter().all(|(_, values)| values.len() == first.len()) {
                    diagnostics.push(joint_diagnostic(
                        &columns,
                        first.len(),
                        "Bayesian-bootstrap",
                        structure,
                    ));
                }
            }
        } else if self.settings.bootstrap_replicates > 0 {
            let refs: Vec<&antecedent_estimate::PreparedFunctionalEffect> =
                present.iter().filter_map(|&i| prepared[i].as_ref()).collect();
            let rows = fitter.shared_replicate_draws(&refs, ctx).map_err(CausalError::from)?;
            if rows.len() > 1 {
                let columns: Vec<Vec<f64>> =
                    (0..present.len()).map(|c| rows.iter().map(|row| row[c]).collect()).collect();
                let named: Vec<(&str, &[f64])> = present
                    .iter()
                    .zip(&columns)
                    .map(|(&i, values)| (COMPONENTS[i].1, values.as_slice()))
                    .collect();
                diagnostics.push(joint_diagnostic(
                    &named,
                    rows.len(),
                    "circular-block or row bootstrap",
                    structure,
                ));
            }
        }

        Ok(diagnostics)
    }

    /// The mediation decomposition over a CPDAG or a PAG: each contrast is the class result of
    /// the path-specific executor, and the requested contrast carries the result.
    ///
    /// # Errors
    ///
    /// The requested contrast is not identified in the class, or the validation suite cannot
    /// be refit.
    pub(super) fn execute_mediation_class(
        &self,
        data: &TabularData,
        class: ClassStructure<'_>,
        query: &MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        query.validate()?;
        // The components run through the path-specific executors, which carry no validation
        // suite; the mediation suite below refits all three together on each class member.
        let plain = (self.settings.refute != super::RefuteSuite::None).then(|| {
            let mut plain = self.clone();
            plain.settings.refute = super::RefuteSuite::None;
            plain
        });
        let runner = plain.as_ref().unwrap_or(self);
        let run = |contrast_query: &MediationQuery| -> Result<StudyResult, CausalError> {
            let source = PathSpecificMixedSource::Mediation(contrast_query);
            match class {
                ClassStructure::Cpdag(cpdag) => {
                    runner.execute_class_path_specific(data, cpdag, source, physical, ctx)
                }
                ClassStructure::Pag(pag) => {
                    runner.execute_pag_path_specific(data, pag, source, physical, ctx)
                }
            }
        };
        let requested = component_slot(query.contrast);
        let mut result = run(query)?;
        let mut values: [Option<f64>; 3] = [None; 3];
        let mut columns: Vec<(&str, Vec<f64>)> = Vec::new();
        values[requested] = Some(result.effect());
        if let Some(posterior) = result.posterior.as_ref() {
            columns.push((COMPONENTS[requested].1, posterior.draws.values.to_vec()));
        }
        for (slot, (contrast, name)) in COMPONENTS.iter().enumerate() {
            if slot == requested {
                continue;
            }
            let mut component = query.clone();
            component.contrast = *contrast;
            match run(&component) {
                Ok(other) => {
                    values[slot] = Some(other.effect());
                    if let Some(posterior) = other.posterior.as_ref() {
                        columns.push((*name, posterior.draws.values.to_vec()));
                    }
                }
                Err(error) if error.reason_code() == Some("effect_not_identified") => {}
                Err(error) => return Err(error),
            }
        }
        let paired: Vec<(&str, &[f64])> =
            columns.iter().map(|(name, draws)| (*name, draws.as_slice())).collect();
        if let Some((_, first)) = paired.first() {
            if paired.len() > 1 && paired.iter().all(|(_, draws)| draws.len() == first.len()) {
                result.diagnostics.push(joint_diagnostic(
                    &paired,
                    first.len(),
                    "Bayesian-bootstrap",
                    "class",
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
        result.refutations = self.refute_mediation_class(data, class, query, physical, ctx)?;
        Ok(result)
    }
}

/// The class structure a mediation decomposition is read over.
#[derive(Clone, Copy)]
pub(super) enum ClassStructure<'a> {
    Cpdag(&'a antecedent_graph::Cpdag),
    Pag(&'a antecedent_graph::Pag),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_draws_report_means_spreads_and_correlations() {
        // total = direct + indirect draw by draw, so corr(direct, indirect) is the one of the
        // two series themselves and total moves with both.
        let direct = [1.0, 2.0, 3.0, 4.0];
        let indirect = [2.0, 1.0, 4.0, 3.0];
        let total: Vec<f64> = direct.iter().zip(&indirect).map(|(a, b)| a + b).collect();
        let fields = joint_fields(&[
            ("total", total.as_slice()),
            ("direct", direct.as_slice()),
            ("indirect", indirect.as_slice()),
        ]);
        let get = |key: &str| -> f64 {
            fields.iter().find(|(k, _)| *k == key).unwrap().1.parse().unwrap()
        };
        assert!((get("direct_mean") - 2.5).abs() < 1e-12);
        assert!((get("indirect_mean") - 2.5).abs() < 1e-12);
        assert!((get("total_mean") - 5.0).abs() < 1e-12);
        // Sample variance of 1..4 is 5/3.
        assert!((get("direct_sd") - (5.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        // cov(direct, indirect) = ((-1.5)(-0.5) + (-0.5)(-1.5) + (0.5)(1.5) + (1.5)(0.5)) / 3
        // = 3 / 3 = 1, so the correlation is 1 / (5/3) = 0.6.
        assert!((get("corr_direct_indirect") - 0.6).abs() < 1e-12);
        // Var(total) = Var(d) + Var(i) + 2 cov = 10/3 + 2 = 16/3.
        assert!((get("total_sd").powi(2) - 16.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_requested_contrast_reads_its_own_component_slot() {
        assert_eq!(component_slot(MediationContrast::Total), 0);
        assert_eq!(component_slot(MediationContrast::Direct), 1);
        assert_eq!(component_slot(MediationContrast::NaturalDirect), 1);
        assert_eq!(component_slot(MediationContrast::Mediated), 2);
        assert_eq!(component_slot(MediationContrast::NaturalIndirect), 2);
    }
}
