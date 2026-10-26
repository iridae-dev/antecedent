// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph-posterior contrasts and mediation over lagged temporal-ADMG atoms.
//!
//! An atom of the posterior is a stationary template of a latent-confounded process
//! ([`antecedent_discovery::TemporalAdmgAtom`]): a contemporaneous ADMG plus lagged directed and
//! lagged latent-confounder edges. Each atom is identified on the unrolled window of its own
//! template, as a supplied `TemporalAdmg` is: the complete ID algorithm decides a pulse,
//! sustained or conditional contrast, the path-specific identifier decides the total, natural
//! direct and natural indirect effects of a mediation, and a cause that precedes the window is a
//! latent common cause of its in-window effects. Atoms differ in the lags they carry, so they
//! differ in the length of their window; every distinct length reads its own lag-aligned window
//! table, resampled in circular blocks of consecutive windows (frequentist) or reweighted by one
//! Dirichlet weight per block (Bayesian).
//!
//! The atoms are mixed by posterior weight (see `atom_mix`); an atom whose window leaves the
//! target unidentified keeps its mass as unidentified mass, and the mixture is then
//! graph-dependent rather than a point. The frequentist standard error of a mixture of several
//! atoms is the spread of the draw-wise mixture of their bootstrap replicates, all evaluated at
//! the same resampled window table. A Bayesian mixture withholds its aggregate posterior (the
//! atoms' posteriors are conditional on their graphs), as the static ADMG route does.

use antecedent_data::TableView;
use std::collections::BTreeMap;

use antecedent_core::{MediationQuery, TemporalEffectQuery, VariableId};
use antecedent_discovery::GraphPosterior;
use antecedent_estimate::{FunctionalEffect, PreparedFunctionalEffect, TemporalMediationEstimate};
use antecedent_identify::temporal_admg::{TemporalAdmgWindowPlan, plan_temporal_admg_window};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use super::atom_mix::{AtomBoundRule, AtomEvalView, AtomMixInput, MixedAtomEvals, mix_atom_evals};
use super::mediation_mixed::{component_slot, identify_components};
use super::mediation_refutation::plug_in_value;
use super::temporal_admg_mediation::window_effect_query;
use super::{
    Arc, BootstrapAccounting, CausalError, CausalPosterior, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, IdentifierId, Instant, PhysicalExecutionPlan, PosteriorMassSummary,
    RefuteSuite, Strategy, StudyResult, TabularData, TimeSeriesData, elapsed_ns, mass_fields,
    overlap_diagnostic, provenance_ids, resolve_posterior_status, run_functional,
};
use crate::analysis::identification_cache::{
    CachedGraphPosteriorAtomIdentification, CachedGraphPosteriorIdentification,
    build_temporal_admg_posterior_identification_cache,
};
use crate::analysis::proof_support::graph_posterior_withheld_diagnostic;
use crate::analysis::temporal_admg_contrast::{used_variables, window_table_over};

/// Stable code of the disclosure every lagged temporal-ADMG posterior result carries.
const WINDOW_DISCLOSURE: &str = "estimate.graph_posterior.temporal_admg_window";

/// The component contrasts of a mediation, in publication order.
const COMPONENTS: [antecedent_core::MediationContrast; 3] = [
    antecedent_core::MediationContrast::Total,
    antecedent_core::MediationContrast::NaturalDirect,
    antecedent_core::MediationContrast::NaturalIndirect,
];

/// Where one atom's target sits in the window of its own template.
#[derive(Clone, Debug)]
struct AtomWindow {
    /// Periods of the window, which names the window table the atom reads.
    periods: u32,
    /// Window columns the functional reads.
    roles: Vec<VariableId>,
    /// Variables of the series the window table must hold values for.
    used: Vec<VariableId>,
    /// Window columns of the treated periods (empty for a mediation).
    treated: Vec<VariableId>,
}

/// One atom's evaluation: the plug-in of its identified functional on its window table.
struct WindowAtomEval {
    key: u64,
    status: IdentificationStatus,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
}

impl AtomEvalView for WindowAtomEval {
    fn key(&self) -> u64 {
        self.key
    }

    fn status(&self) -> IdentificationStatus {
        self.status
    }

    fn estimand(&self) -> Option<&IdentifiedEstimand> {
        Some(&self.estimand)
    }

    fn scalar(&self) -> Option<f64> {
        self.estimate.ate.is_finite().then_some(self.estimate.ate)
    }

    fn posterior(&self) -> Option<&CausalPosterior> {
        self.posterior.as_ref()
    }

    fn assumptions(&self) -> Option<&antecedent_core::AssumptionSet> {
        Some(&self.estimate.provenance.assumptions)
    }

    fn identified_bounds(&self) -> Option<(f64, f64)> {
        None
    }

    fn partial(&self) -> bool {
        matches!(self.status, IdentificationStatus::PartiallyIdentified)
    }

    fn analytic_se(&self) -> Option<f64> {
        None
    }

    fn influence(&self) -> Option<&[f64]> {
        None
    }

    fn reported_uncertainty(&self) -> Option<antecedent_estimate::EstimateUncertainty> {
        Some(self.estimate.uncertainty)
    }
}

/// One fitted atom kept for the draw-wise mixture standard error.
struct AtomFit {
    weight: f64,
    fitter: FunctionalEffect,
    prepared: PreparedFunctionalEffect,
    periods: u32,
}

/// The window tables of one execution: one lag-aligned table per window length.
struct WindowTables {
    tables: BTreeMap<u32, TabularData>,
}

impl WindowTables {
    /// Build the table of every window length the atoms read, over the union of the variables
    /// the atoms of that length name.
    fn build(
        series: &TimeSeriesData,
        variables: &[VariableId],
        windows: &[AtomWindow],
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let mut used: BTreeMap<u32, Vec<VariableId>> = BTreeMap::new();
        for window in windows {
            used.entry(window.periods).or_default().extend(window.used.iter().copied());
        }
        let mut tables = BTreeMap::new();
        for (periods, mut names) in used {
            names.sort_unstable();
            names.dedup();
            tables.insert(
                periods,
                window_table_over(series, variables, periods, &names, &ctx.kernel_policy)?,
            );
        }
        Ok(Self { tables })
    }

    fn get(&self, periods: u32) -> Result<&TabularData, CausalError> {
        self.tables.get(&periods).ok_or_else(|| CausalError::Compile {
            message: format!("no window table of {periods} periods was built"),
        })
    }
}

/// What mixing the atoms of one slot reads.
struct WindowMix<'a> {
    tables: &'a WindowTables,
    /// One entry per proof, in order.
    windows: &'a [AtomWindow],
    graphs: &'a WeightedGraphSamples,
    proofs: &'a [CachedGraphPosteriorAtomIdentification],
    unidentified_mass: f64,
}

/// Publication inputs shared by posterior window contrasts and native mediation.
struct WindowPosteriorFinish<'a> {
    physical: &'a PhysicalExecutionPlan,
    reported: (VariableId, VariableId),
    strategy: Strategy,
    identify_cached: bool,
    refutations: Vec<antecedent_validate::RefutationReport>,
    started: Instant,
    diagnostics: Vec<Diagnostic>,
    decomposition: Option<(Option<f64>, Option<f64>, Option<f64>)>,
}

/// The standard error of the draw-wise mixture, and the replicates it rests on.
struct MixtureSe {
    se: f64,
    replicates: u32,
}

/// The series variables as window columns, and the count the posterior must cover.
fn window_variables(
    series: &TimeSeriesData,
    gp: &GraphPosterior,
) -> Result<(Vec<VariableId>, u32), CausalError> {
    let count = series.schema().len();
    if count != gp.n_vars {
        return Err(CausalError::Compile {
            message: format!(
                "the lagged temporal-ADMG posterior covers {} variables and the series has \
                 {count}; the masks name the series' variables by position",
                gp.n_vars
            ),
        });
    }
    let count = u32::try_from(count).map_err(|_| CausalError::Compile {
        message: "series schema exceeds the variable id range".into(),
    })?;
    Ok(((0..count).map(VariableId::from_raw).collect(), count))
}

/// Posterior position named by a position-derived graph key.
fn position_of(key: u64) -> Result<usize, CausalError> {
    usize::try_from(key).map_err(|_| CausalError::Compile {
        message: "a lagged temporal-ADMG posterior atom key exceeds the atom range".into(),
    })
}

/// Position-derived graph key: lag-distinct atoms share a contemporaneous mask, so the
/// posterior's own keys do not tell them apart.
fn position_key(index: usize) -> Result<u64, CausalError> {
    u64::try_from(index).map_err(|_| CausalError::Compile {
        message: "a lagged temporal-ADMG posterior has too many atoms for position keys".into(),
    })
}

/// The window column of `variable` in `period` (a window numbered over the schema's variables).
fn coordinate(variable: VariableId, period: u32, variable_count: u32) -> VariableId {
    VariableId::from_raw(period * variable_count + variable.raw())
}

/// Window columns a pulse, sustained or conditional contrast reads: the treatment at each
/// treated period, the outcome, and each modifier at the first treated period.
fn contrast_roles(
    plan: &TemporalAdmgWindowPlan,
    query: &TemporalEffectQuery,
    variable_count: u32,
) -> Vec<VariableId> {
    let first = plan.treated.iter().copied().min().unwrap_or(plan.origin);
    plan.treated
        .iter()
        .map(|&period| coordinate(query.treatment, period, variable_count))
        .chain([coordinate(query.outcome, plan.outcome_period, variable_count)])
        .chain(
            query
                .effect_modifiers
                .iter()
                .map(|&modifier| coordinate(modifier, first, variable_count)),
        )
        .collect()
}

/// The disclosure of how the atoms were read.
fn window_disclosure(
    atoms: usize,
    windows: &BTreeMap<u32, usize>,
    unidentified_mass: f64,
) -> Diagnostic {
    let lengths = windows
        .iter()
        .map(|(periods, count)| format!("{periods} periods x {count} atom(s)"))
        .collect::<Vec<_>>()
        .join(", ");
    Diagnostic::new(
        WINDOW_DISCLOSURE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "each of the {atoms} identified lagged temporal-ADMG atom(s) was identified on the \
             unrolled window of its own template (a cause that precedes the window is a latent \
             common cause of its in-window effects) and evaluated by a plug-in of its identified \
             functional on the lag-aligned window table of its window length ({lengths}); \
             windows overlap, so uncertainty resamples circular blocks of consecutive windows \
             (never shorter than the window); the series is assumed stationary over every \
             template; outer weights are posterior probabilities; unidentified_mass=\
             {unidentified_mass}"
        ),
    )
    .with_fields([
        ("atoms", atoms.to_string()),
        ("window_lengths", windows.len().to_string()),
        ("unidentified_mass", unidentified_mass.to_string()),
    ])
}

impl super::Study {
    /// Refusals every window route over lagged temporal-ADMG atoms shares.
    fn ensure_admg_posterior_window_licensed(&self) -> Result<(), CausalError> {
        if !self.custom_validators.is_empty() {
            return Err(crate::unsupported_reason!(
                "validators_not_applicable",
                "a custom validator refutes one scalar average effect and none applies to a \
                 mixture of window contrasts over temporal-ADMG atoms"
            ));
        }
        Ok(())
    }

    /// The window-native suite on every atom that carries posterior weight, each refit on the
    /// window table of its own template, mixed by posterior weight (the mixed check passes
    /// only when every contributing atom passed); empty at validation none.
    fn refute_window_atoms(
        &self,
        input: &WindowMix<'_>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<antecedent_validate::RefutationReport>, CausalError> {
        if self.refute == RefuteSuite::None {
            return Ok(Vec::new());
        }
        let key_weights = IdentifiedKeyWeights::new(input.graphs);
        let mut per_atom = Vec::new();
        for (proof, window) in input.proofs.iter().zip(input.windows) {
            let weight = key_weights.get(proof.key);
            if weight <= 0.0 {
                continue;
            }
            let table = input.tables.get(window.periods)?;
            let Some(original) =
                plug_in_value(table, &proof.estimand, &proof.identification, &window.roles, ctx)
            else {
                continue;
            };
            let reports = antecedent_validate::functional::refute_window_contrast(
                table,
                &proof.identification,
                &proof.estimand,
                &window.roles,
                &window.treated,
                window.periods,
                original,
                self.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?;
            per_atom.push((weight, reports));
        }
        Ok(antecedent_validate::QueryRefutationPlan::mix_weighted(per_atom))
    }

    /// Estimate every identified atom of one slot on its window table and mix them by posterior
    /// weight.
    fn mix_window_atoms(
        &self,
        input: &WindowMix<'_>,
        ctx: &ExecutionContext,
    ) -> Result<(MixedAtomEvals, Option<MixtureSe>), CausalError> {
        let key_weights = IdentifiedKeyWeights::new(input.graphs);
        let mut evals = Vec::new();
        let mut fits = Vec::new();
        for (proof, window) in input.proofs.iter().zip(input.windows) {
            let weight = key_weights.get(proof.key);
            if weight <= 0.0 {
                continue;
            }
            let table = input.tables.get(window.periods)?;
            let (resampling, design) = self.window_row_resampling(window.periods)?;
            let fitter = FunctionalEffect {
                bootstrap_replicates: self.bootstrap_replicates,
                ..FunctionalEffect::new()
            }
            .with_row_resampling(resampling, design);
            let prepared = fitter
                .prepare(
                    table,
                    &proof.estimand,
                    &proof.identification.arena,
                    proof.identification.required_assumptions.clone(),
                    &window.roles,
                )
                .map_err(CausalError::from)?;
            let (estimate, posterior) = run_functional(
                &fitter,
                &prepared,
                &self.inference,
                proof.identification.status,
                ctx,
            )?;
            if estimate.ate.is_finite() {
                fits.push(AtomFit { weight, fitter, prepared, periods: window.periods });
            }
            evals.push(WindowAtomEval {
                key: proof.key,
                status: proof.identification.status,
                estimand: proof.estimand.clone(),
                estimate,
                posterior,
            });
        }
        let mixed = mix_atom_evals(
            &evals,
            AtomMixInput {
                graphs: input.graphs,
                proofs: input.proofs,
                unidentified_mass: input.unidentified_mass,
                full_mass_scope: true,
                truncated_atoms: 0,
                bounds: AtomBoundRule::Point,
                estimand_fallback: true,
            },
            |n, mixable| {
                format!(
                    "temporal-ADMG graph-posterior mix: no evaluable atom values (evals={n}, \
                     mixable={mixable})"
                )
            },
            || "temporal-ADMG graph-posterior envelope: missing estimand".to_string(),
        )?;
        let se = if mixed.mixable_scalar && fits.len() > 1 {
            self.window_mixture_se(&fits, ctx)
        } else {
            None
        };
        Ok((mixed, se))
    }

    /// Spread of the draw-wise mixture of the atoms' bootstrap replicates, all evaluated at the
    /// same resampled window table; `None` when the atoms read tables of different lengths (no
    /// common resampled table), no replicates were requested, or the inference is Bayesian.
    fn window_mixture_se(&self, fits: &[AtomFit], ctx: &ExecutionContext) -> Option<MixtureSe> {
        if self.inference.is_bayesian() || self.bootstrap_replicates == 0 {
            return None;
        }
        let periods = fits.first()?.periods;
        if fits.iter().any(|fit| fit.periods != periods) {
            return None;
        }
        let refs: Vec<&PreparedFunctionalEffect> = fits.iter().map(|fit| &fit.prepared).collect();
        let rows = fits.first()?.fitter.shared_replicate_draws(&refs, ctx).ok()?;
        if rows.len() < 2 {
            return None;
        }
        let total: f64 = fits.iter().map(|fit| fit.weight).sum();
        let mixed: Vec<f64> = rows
            .iter()
            .map(|row| {
                fits.iter().zip(row).map(|(fit, value)| fit.weight * value).sum::<f64>() / total
            })
            .collect();
        let n = mixed.len() as f64;
        let mean = mixed.iter().sum::<f64>() / n;
        let variance = mixed.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
        variance.is_finite().then(|| MixtureSe {
            se: variance.sqrt(),
            replicates: u32::try_from(rows.len()).unwrap_or(u32::MAX),
        })
    }

    /// Adopt the mixture standard error on a mixed estimate of several atoms.
    fn adopt_mixture_se(mixed: &mut MixedAtomEvals, se: Option<&MixtureSe>) {
        if let Some(MixtureSe { se, replicates }) = se {
            mixed.estimate.uncertainty.bootstrap = BootstrapAccounting {
                se: Some(*se),
                replicates_ok: Some(*replicates),
                replicates_failed: Some(0),
                cancelled: false,
                early_stopped: false,
            };
        }
    }

    /// Refit the three native components per contributing window, then mix reports by atom mass.
    fn refute_mediation_window_atoms(
        &self,
        found: &[Option<AtomComponents>],
        weights: &[f64],
        tables: &WindowTables,
        requested: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<antecedent_validate::RefutationReport>, CausalError> {
        let mut refutations = Vec::new();
        if self.refute != RefuteSuite::None {
            let mut per_atom = Vec::new();
            for (position, atom) in found.iter().enumerate() {
                let Some(atom) = atom else { continue };
                let weight = weights[position];
                if weight <= 0.0 || atom.components.get(requested).is_none_or(Option::is_none) {
                    continue;
                }
                let table = tables.get(atom.window.periods)?;
                let components: [Option<super::mediation_refutation::IdentifiedComponent>; 3] =
                    std::array::from_fn(|slot| {
                        let (identification, estimand) =
                            atom.components.get(slot).and_then(Option::as_ref)?;
                        let value = plug_in_value(
                            table,
                            estimand,
                            identification,
                            &atom.window.roles,
                            ctx,
                        )?;
                        Some((identification.clone(), estimand.clone(), value))
                    });
                let resampling = self.window_row_resampling(atom.window.periods)?;
                per_atom.push((
                    weight,
                    self.refute_mediation_decomposition(
                        table,
                        &components,
                        &atom.window.roles,
                        resampling,
                        ctx,
                    )?,
                ));
            }
            refutations = antecedent_validate::QueryRefutationPlan::mix_weighted(per_atom);
        }
        Ok(refutations)
    }

    /// Disclose the windows, posterior availability, aggregation policy and overlap of the mixture.
    fn window_mixture_diagnostics(
        &self,
        mixed: &MixedAtomEvals,
        windows: &[AtomWindow],
        atoms: usize,
        unidentified_mass: f64,
    ) -> Vec<Diagnostic> {
        let mut lengths = BTreeMap::new();
        for window in windows {
            *lengths.entry(window.periods).or_insert(0usize) += 1;
        }
        let mut diagnostics = mixed.identification.diagnostics.clone();
        if self.inference.is_bayesian() && mixed.posterior.is_none() {
            diagnostics.push(graph_posterior_withheld_diagnostic());
        }
        diagnostics.push(window_disclosure(atoms, &lengths, unidentified_mass));
        diagnostics.push(structural_aggregation_diagnostic(mixed));
        diagnostics.push(overlap_diagnostic(mixed.estimate.provenance.overlap));
        diagnostics
    }

    /// Bind native family outputs to their retained physical plan and provenance.
    fn finish_window_posterior(
        &self,
        mixed: MixedAtomEvals,
        finish: WindowPosteriorFinish<'_>,
    ) -> StudyResult {
        let WindowPosteriorFinish {
            physical,
            reported,
            strategy,
            identify_cached,
            refutations,
            started,
            diagnostics,
            decomposition,
        } = finish;
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let mediation = decomposition.map(|(total, direct, mediated)| TemporalMediationEstimate {
            effect: mixed.estimate.clone(),
            total,
            direct,
            mediated,
        });
        let MixedAtomEvals {
            policy, identification, estimand, estimate, posterior, mixture, ..
        } = mixed;
        self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                strategy,
                reported.0,
                reported.1,
            )
            .identify_cached(identify_cached)
            .mediation(mediation)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal_admg_graph_posterior",
                    policy.as_str(),
                )),
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                diagnostics: Some(diagnostics),
                structural_response: Some(mixture),
                ..Default::default()
            }),
        )
    }

    /// Mix the pulse, sustained or conditional contrast of `query` over the atoms of a lagged
    /// temporal-ADMG posterior.
    ///
    /// # Errors
    ///
    /// A validation suite or custom validator, a posterior that does not cover the series'
    /// variables, an atom that does not unroll, a series that holds a gap inside a window, or no
    /// identified atom.
    pub(super) fn execute_temporal_admg_graph_posterior(
        &self,
        series: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        self.ensure_admg_posterior_window_licensed()?;
        let (variables, variable_count) = window_variables(series, gp)?;
        let max_lag = gp.max_lag.unwrap_or(0);
        let (identified, identify_cached): (Arc<CachedGraphPosteriorIdentification>, bool) =
            if let Some(cache) = self.graph_posterior_identification_cache.as_ref() {
                (Arc::clone(cache), true)
            } else if let Some(cache) =
                crate::analysis::lowering::plan_retained_graph_posterior(self, physical)
            {
                (cache, false)
            } else {
                (
                    Arc::new(build_temporal_admg_posterior_identification_cache(
                        gp, &variables, query, ctx,
                    )?),
                    false,
                )
            };
        if identified.atoms.is_empty() {
            return Err(CausalError::not_identified(
                IdentificationStatus::NotIdentified,
                false,
                "no atom of the lagged temporal-ADMG posterior identifies the contrast on the \
                 window of its template",
            ));
        }
        let mut windows = Vec::with_capacity(identified.atoms.len());
        for atom in identified.atoms.iter() {
            let template = gp
                .temporal_admg_atom(position_of(atom.key)?)
                .ok_or_else(|| CausalError::Compile {
                    message: "a retained atom has no lagged template in the posterior".into(),
                })?
                .to_template(gp.n_vars, max_lag, &variables)?;
            let plan = plan_temporal_admg_window(&template, query)?;
            windows.push(AtomWindow {
                periods: plan.periods,
                roles: contrast_roles(&plan, query, variable_count),
                used: used_variables(&template, query),
                treated: plan
                    .treated
                    .iter()
                    .map(|&period| coordinate(query.treatment, period, variable_count))
                    .collect(),
            });
        }
        let tables = WindowTables::build(series, &variables, &windows, ctx)?;
        let unidentified_mass = identified.graphs.unidentified_mass();
        let (mut mixed, se) = self.mix_window_atoms(
            &WindowMix {
                tables: &tables,
                windows: &windows,
                graphs: &identified.graphs,
                proofs: &identified.atoms,
                unidentified_mass,
            },
            ctx,
        )?;
        Self::adopt_mixture_se(&mut mixed, se.as_ref());
        let refutations = self.refute_window_atoms(
            &WindowMix {
                tables: &tables,
                windows: &windows,
                graphs: &identified.graphs,
                proofs: &identified.atoms,
                unidentified_mass,
            },
            ctx,
        )?;
        mixed.identification.status = resolve_posterior_status(
            mixed.identification.status,
            mixed.policy,
            PosteriorMassSummary { unidentified: unidentified_mass, ..Default::default() },
        );
        let diagnostics = self.window_mixture_diagnostics(
            &mixed,
            &windows,
            identified.atoms.len(),
            unidentified_mass,
        );
        Ok(self.finish_window_posterior(
            mixed,
            WindowPosteriorFinish {
                physical,
                reported: (query.treatment, query.outcome),
                strategy: Strategy::new(IdentifierId::GeneralId, EstimatorId::FunctionalEffect),
                identify_cached,
                refutations,
                started,
                diagnostics,
                decomposition: None,
            },
        ))
    }

    /// Mix the total, natural direct and natural indirect effects of `query` over the atoms of a
    /// lagged temporal-ADMG posterior.
    ///
    /// Each atom decomposes on the unrolled window of its own template. The requested contrast
    /// must be identified in at least one atom (an atom that leaves it unidentified keeps its
    /// mass as unidentified mass); a component other than the requested one is mixed over the
    /// atoms that identify it and reported absent when none does.
    ///
    /// # Errors
    ///
    /// As [`Self::execute_temporal_admg_graph_posterior`], more than one outcome horizon, or no
    /// atom that identifies the requested contrast.
    pub(super) fn execute_temporal_admg_graph_posterior_mediation(
        &self,
        series: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        self.ensure_admg_posterior_window_licensed()?;
        let effect = window_effect_query(query)?;
        let (variables, variable_count) = window_variables(series, gp)?;
        let requested = component_slot(query.contrast);
        crate::analysis::execute::report_identify_compute(ctx);
        let found = ctx.map_indexed(gp.n_graphs, |i, inner| {
            if inner.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_IDENTIFY,
                });
            }
            Ok::<Option<AtomComponents>, CausalError>(
                atom_components(gp, i, &variables, variable_count, query, &effect).ok().flatten(),
            )
        })?;
        let (slots, keys) = mediation_component_slots(gp, &found)?;
        if slots[requested].proofs.is_empty() {
            return Err(CausalError::not_identified(
                IdentificationStatus::NotIdentified,
                false,
                "no atom of the lagged temporal-ADMG posterior identifies the requested \
                 mediation contrast on the window of its template",
            ));
        }
        let all_windows: Vec<AtomWindow> =
            slots.iter().flat_map(|layer| layer.windows.iter().cloned()).collect();
        let tables = WindowTables::build(series, &variables, &all_windows, ctx)?;
        let weights: Vec<f64> = gp.weights.to_vec();
        let mut results: Vec<Option<(MixedAtomEvals, f64)>> = Vec::new();
        for (slot, layer) in slots.iter().enumerate() {
            if layer.proofs.is_empty() {
                results.push(None);
                continue;
            }
            let graphs =
                WeightedGraphSamples::new(weights.clone(), layer.flags.clone(), keys.clone())?;
            let unidentified_mass = graphs.unidentified_mass();
            let (mut mixed, se) = self.mix_window_atoms(
                &WindowMix {
                    tables: &tables,
                    windows: &layer.windows,
                    graphs: &graphs,
                    proofs: &layer.proofs,
                    unidentified_mass,
                },
                ctx,
            )?;
            if slot == requested {
                Self::adopt_mixture_se(&mut mixed, se.as_ref());
            }
            mixed.identification.status = resolve_posterior_status(
                mixed.identification.status,
                mixed.policy,
                PosteriorMassSummary { unidentified: unidentified_mass, ..Default::default() },
            );
            results.push(Some((mixed, unidentified_mass)));
        }
        let refutations =
            self.refute_mediation_window_atoms(&found, &weights, &tables, requested, ctx)?;
        let value = |slot: usize| {
            results[slot]
                .as_ref()
                .filter(|(mixed, _)| mixed.mixable_scalar)
                .map(|(mixed, _)| mixed.estimate.ate)
        };
        let (total, direct, mediated) = (value(0), value(1), value(2));
        let (mixed, unidentified_mass) = results[requested].take().ok_or_else(|| {
            CausalError::Compile { message: "requested contrast has no mixture".into() }
        })?;
        let diagnostics = self.window_mixture_diagnostics(
            &mixed,
            &slots[requested].windows,
            slots[requested].proofs.len(),
            unidentified_mass,
        );
        Ok(self.finish_window_posterior(
            mixed,
            WindowPosteriorFinish {
                physical,
                reported: (query.treatment, query.outcome),
                strategy: Strategy::new(
                    IdentifierId::PathSpecificNatural,
                    EstimatorId::FunctionalEffect,
                ),
                identify_cached: false,
                refutations,
                started,
                diagnostics,
                decomposition: Some((total, direct, mediated)),
            },
        ))
    }
}

/// The atoms of one component of a mediation: flags over every posterior position, and the
/// proofs and windows of the positions that identify it.
#[derive(Default)]
struct SlotAtoms {
    flags: Vec<GraphIdentFlag>,
    proofs: Vec<CachedGraphPosteriorAtomIdentification>,
    windows: Vec<AtomWindow>,
}

/// Preserve one aligned flag per posterior sample and component, with proofs only where identified.
fn mediation_component_slots(
    gp: &GraphPosterior,
    found: &[Option<AtomComponents>],
) -> Result<(Vec<SlotAtoms>, Vec<u64>), CausalError> {
    // One identification layer per component: the atoms that identify it.
    let mut slots: Vec<SlotAtoms> = (0..COMPONENTS.len()).map(|_| SlotAtoms::default()).collect();
    let mut keys = Vec::with_capacity(gp.n_graphs);
    for (i, atom) in found.iter().enumerate() {
        keys.push(position_key(i)?);
        for (slot, layer) in slots.iter_mut().enumerate() {
            let identified = atom
                .as_ref()
                .and_then(|atom| atom.components[slot].as_ref().map(|component| (atom, component)));
            if let Some((atom, (identification, estimand))) = identified {
                layer.flags.push(GraphIdentFlag::Identified);
                layer.proofs.push(CachedGraphPosteriorAtomIdentification {
                    key: keys[i],
                    estimand: estimand.clone(),
                    identification: identification.clone(),
                    transport: None,
                });
                layer.windows.push(atom.window.clone());
            } else {
                layer.flags.push(GraphIdentFlag::Unidentified);
            }
        }
    }
    Ok((slots, keys))
}

/// One atom's three components on its window.
struct AtomComponents {
    window: AtomWindow,
    components: Vec<Option<(IdentificationResult, IdentifiedEstimand)>>,
}

/// Decompose the mediation on the unrolled window of atom `index`; `None` when the atom's
/// template does not unroll or does not identify the requested contrast.
fn atom_components(
    gp: &GraphPosterior,
    index: usize,
    variables: &[VariableId],
    variable_count: u32,
    query: &MediationQuery,
    effect: &TemporalEffectQuery,
) -> Result<Option<AtomComponents>, CausalError> {
    let Some(atom) = gp.temporal_admg_atom(index) else { return Ok(None) };
    let Ok(template) = atom.to_template(gp.n_vars, gp.max_lag.unwrap_or(0), variables) else {
        return Ok(None);
    };
    let plan = plan_temporal_admg_window(&template, effect)?;
    let window = template.unroll_window_over(plan.periods, variable_count)?;
    let first = plan.treated.first().copied().ok_or_else(|| CausalError::Compile {
        message: "the pulse has no treated period".into(),
    })?;
    let (Some(treatment), Some(outcome)) = (
        window.coordinate(query.treatment, first),
        window.coordinate(query.outcome, plan.outcome_period),
    ) else {
        return Ok(None);
    };
    let mut mediators = Vec::new();
    for &variable in query.mediators.iter() {
        for period in first..=plan.outcome_period {
            if let Some(node) = window.coordinate(variable, period) {
                if node != treatment && node != outcome {
                    mediators.push(node);
                }
            }
        }
    }
    if mediators.is_empty() {
        return Ok(None);
    }
    let level = |intervention: &antecedent_core::Intervention| match intervention {
        antecedent_core::Intervention::Set { value, .. } => {
            Some(antecedent_core::Intervention::set(treatment, value.clone()))
        }
        _ => None,
    };
    let (Some(control), Some(active)) = (level(&query.control), level(&query.active)) else {
        return Ok(None);
    };
    let mut window_query =
        MediationQuery::binary(treatment, outcome, mediators.clone(), query.contrast);
    window_query.control = control;
    window_query.active = active;
    window_query.target_population = query.target_population.clone();
    let Ok(components) = identify_components(
        window.admg(),
        &window_query,
        IdentifierId::PathSpecificNatural,
        EstimatorId::FunctionalEffect,
    ) else {
        return Ok(None);
    };
    let mut roles = vec![treatment, outcome];
    roles.extend(mediators);
    let mut used = used_variables(&template, effect);
    used.extend(query.mediators.iter().copied());
    used.sort_unstable();
    used.dedup();
    Ok(Some(AtomComponents {
        window: AtomWindow { periods: plan.periods, roles, used, treated: Vec::new() },
        components,
    }))
}

/// The aggregation-policy and mass disclosure of a mixture.
fn structural_aggregation_diagnostic(mixed: &MixedAtomEvals) -> Diagnostic {
    Diagnostic::new(
        "estimate.graph_posterior.structural_aggregation",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "policy={}; weight_basis=posterior_probability; completion enumeration is not \
             posterior probability; identified_mass={}; unidentified_mass={}",
            mixed.policy.as_str(),
            mixed.mixture.identified_mass,
            mixed.mixture.unidentified_mass
        ),
    )
    .with_fields(mass_fields(Some(mixed.mixture.identified_mass), mixed.mixture.unidentified_mass))
}
