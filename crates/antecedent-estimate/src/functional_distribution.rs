//! Nonparametric estimation of identified interventional distributions via
//! discrete empirical CPT plug-in into compiled ID/IDC functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::manual_flatten,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::zero_sized_map_values
)]

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use antecedent_core::{
    AssumptionSet, CausalRng, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext,
    IdentificationStatus, Intervention, InterventionalDistributionQuery, SupportDiagnostic,
    SupportRegion, SupportReport, SupportStatus, TargetPopulation, Value, VariableId,
};
use antecedent_data::{ColumnView, TableView, TabularData};
use antecedent_expr::{
    Assignment, CausalExprArena, CompiledEvaluator, DistributionProvider, DomainRef,
    EmpiricalTableProvider, EstimandMethod, EvalContext, EvalError, ExprId, ExprNode, FactorSpec,
    IdentifiedEstimand, InterventionAssignment,
};
use antecedent_prob::{
    InferenceDiagnostics, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema,
};

use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::prepare::require_method;
use crate::util::bootstrap_se;

/// Hard cap on discrete levels per variable (fail-closed beyond this).
const MAX_DISCRETE_LEVELS: usize = 64;

/// One outcome-level probability mass under an interventional (and optional
/// observational) conditioning assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct DistributionAtom {
    /// Outcome variable assignments (aligned to query outcomes order).
    pub outcomes: Arc<[(VariableId, Value)]>,
    /// Conditioning assignments (empty when unconditional).
    pub conditioning: Arc<[(VariableId, Value)]>,
    /// Estimated probability mass.
    pub probability: f64,
}

/// Estimated interventional distribution P(Y | do(X)[, Z]).
#[derive(Clone, Debug)]
pub struct InterventionalDistributionEstimate {
    /// Probability atoms over the outcome support. Bayesian estimates publish
    /// posterior mean probabilities, aligned with the posterior atom columns.
    pub atoms: Arc<[DistributionAtom]>,
    /// Interventional mean for one numeric outcome and one conditioning assignment;
    /// otherwise NaN.
    pub mean: f64,
    /// Posterior SD of the mean in Bayesian mode. Analytic SE is not defined for
    /// the frequentist discrete plug-in (multinomial delta-method out of scope).
    pub se_analytic: f64,
    /// Bootstrap SE of the interventional mean when requested. For a binary
    /// outcome the mean is a probability: use [`Self::mean_interval`], not
    /// `mean ± z·se`, which can leave `[0, 1]` near the boundary.
    pub se_bootstrap: Option<f64>,
    /// Successful bootstrap replicates contributing to [`Self::se_bootstrap`].
    pub bootstrap_replicates_ok: Option<u32>,
    /// Soft-failed bootstrap replicates.
    pub bootstrap_replicates_failed: Option<u32>,
    /// Bootstrap loop observed cooperative cancellation.
    pub bootstrap_cancelled: bool,
    /// Adaptive bootstrap early-stop.
    pub bootstrap_early_stopped: bool,
    /// Assumptions carried from identification.
    pub assumptions: AssumptionSet,
    /// Overlap policy recorded on the artifact.
    pub overlap: OverlapPolicy,
    /// Estimated retained-memory cost of fitted scratch (bytes), when known.
    pub retained_memory_bytes: Option<u64>,
    /// Frequentist per-atom bootstrap uncertainty, aligned with [`Self::atoms`].
    /// Empty when no bootstrap ran (zero replicates requested, or Bayesian mode,
    /// whose atom uncertainty lives in the posterior draws).
    pub atom_uncertainty: Arc<[AtomUncertainty]>,
    /// Bounded interval for [`Self::mean`] when the single outcome takes values
    /// in `{0, 1}`, so the mean is the probability `P(Y = 1 | do(x)[, z])` and the
    /// interval is that atom's [`AtomUncertainty::interval`]. `None` otherwise.
    pub mean_interval: Option<ProbabilityInterval>,
}

/// Why a bounded probability interval could not be formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbabilityIntervalUnavailable {
    /// The plug-in probability is exactly 0 or 1 (or not a probability): its
    /// logit is infinite and every bootstrap replicate sits on the same
    /// boundary, so no sampling spread is observed to build an interval from.
    BoundaryEstimate,
    /// Fewer than two usable bootstrap replicates, or more than half failed.
    BootstrapUnavailable,
    /// Every usable replicate reproduced the point value (zero spread).
    ZeroSpread,
}

impl ProbabilityIntervalUnavailable {
    /// Stable machine-readable reason.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BoundaryEstimate => "boundary_estimate",
            Self::BootstrapUnavailable => "bootstrap_unavailable",
            Self::ZeroSpread => "zero_spread",
        }
    }
}

/// Two-sided interval for one interventional probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProbabilityInterval {
    /// Interval inside `[0, 1]` at nominal two-sided `level`.
    Bounded {
        /// Nominal two-sided coverage.
        level: f64,
        /// Lower endpoint.
        lower: f64,
        /// Upper endpoint.
        upper: f64,
    },
    /// No interval could be formed honestly.
    Unavailable(ProbabilityIntervalUnavailable),
}

impl ProbabilityInterval {
    /// `(lower, upper)` when the interval exists.
    #[must_use]
    pub const fn bounds(&self) -> Option<(f64, f64)> {
        match *self {
            Self::Bounded { lower, upper, .. } => Some((lower, upper)),
            Self::Unavailable(_) => None,
        }
    }
}

/// Frequentist bootstrap uncertainty of one [`DistributionAtom`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomUncertainty {
    /// Bootstrap SE of the atom probability (probability scale).
    pub se_bootstrap: Option<f64>,
    /// Replicates that produced a value for this atom.
    pub replicates_ok: u32,
    /// Bounded interval at the estimator's confidence level.
    pub interval: ProbabilityInterval,
}

/// Logit-scale delta-method interval for a probability from its bootstrap SE:
/// `expit(logit(p̂) ± z·se / (p̂(1 − p̂)))`, `z = Φ⁻¹(1/2 + level/2)`.
///
/// Why this construction: a probability's sampling law is skewed toward the
/// interior near 0 and 1, so the symmetric `p̂ ± z·se` both leaves `[0, 1]` and
/// under-covers on the short side. On the logit scale the law is close to
/// symmetric, and `se / (p̂(1 − p̂))` is the delta-method SE of `logit(p̂)`
/// (the same transformation that makes the logit Wald interval for a binomial
/// proportion behave well at small counts). The back-transformed endpoints lie
/// strictly inside `(0, 1)` and the interval widens on the interior side. It
/// reuses the bootstrap SE that is already published, so the scalar SE and the
/// interval describe the same replicates; a percentile interval would need the
/// replicate probabilities to be retained and inherits the Wald-like
/// short-side under-coverage of a discrete proportion.
///
/// At `p̂ ∈ {0, 1}` the logit is infinite and the bootstrap spread is
/// degenerate, so the interval is [`ProbabilityIntervalUnavailable::BoundaryEstimate`]
/// rather than a zero-width or clamped fake.
#[must_use]
pub fn logit_probability_interval(p: f64, se: Option<f64>, level: f64) -> ProbabilityInterval {
    if !(p > 0.0 && p < 1.0) {
        return ProbabilityInterval::Unavailable(ProbabilityIntervalUnavailable::BoundaryEstimate);
    }
    let Some(se) = se.filter(|s| s.is_finite() && *s >= 0.0) else {
        return ProbabilityInterval::Unavailable(
            ProbabilityIntervalUnavailable::BootstrapUnavailable,
        );
    };
    if se == 0.0 {
        return ProbabilityInterval::Unavailable(ProbabilityIntervalUnavailable::ZeroSpread);
    }
    let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
    if !z.is_finite() || z <= 0.0 {
        return ProbabilityInterval::Unavailable(
            ProbabilityIntervalUnavailable::BootstrapUnavailable,
        );
    }
    let centre = (p / (1.0 - p)).ln();
    let half = z * se / (p * (1.0 - p));
    ProbabilityInterval::Bounded { level, lower: expit(centre - half), upper: expit(centre + half) }
}

/// Numerically stable logistic function; always in `[0, 1]`.
fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Reusable scratch for [`FunctionalDistribution`] estimation.
#[derive(Clone, Debug, Default)]
pub struct FunctionalDistributionWorkspace {
    /// Scratch assignment reused across outcome atoms.
    pub assignment: Assignment,
}

impl FunctionalDistributionWorkspace {
    /// Clear reusable buffers.
    pub fn clear(&mut self) {
        self.assignment = Assignment::new();
    }
}

/// Prepared discrete functional-distribution problem.
#[derive(Clone, Debug)]
pub struct PreparedFunctionalDistribution {
    /// Identified estimand (`GeneralId` / IDC).
    pub estimand: IdentifiedEstimand,
    /// Expression arena owning the functional. Shared (never mutated after
    /// `prepare`), so per-replicate bootstrap clones of the prepared object
    /// stay cheap instead of deep-copying the arena.
    pub arena: Arc<CausalExprArena>,
    /// Compiled evaluator for the functional root.
    pub compiled: CompiledEvaluator,
    /// Empirical CPT provider built from data.
    pub provider: EmpiricalTableProvider,
    /// Outcome variables (query order).
    pub outcomes: Arc<[VariableId]>,
    /// Hard intervention bindings from the query.
    pub interventions: Arc<[InterventionAssignment]>,
    /// Observational conditioning bindings (IDC); empty when unconditional.
    pub conditioning: Arc<[InterventionAssignment]>,
    /// Assumptions from identification.
    pub assumptions: AssumptionSet,
    /// Row-aligned discrete columns for bootstrap CPT refits.
    bootstrap_columns: HashMap<VariableId, Vec<Option<Value>>>,
    /// Factor specs used to rebuild the empirical provider.
    bootstrap_factors: Vec<(Arc<[VariableId]>, Arc<[VariableId]>)>,
    /// Interventional signatures used to rebuild the empirical provider.
    bootstrap_signatures:
        Vec<(Arc<[VariableId]>, Arc<[VariableId]>, Arc<[InterventionAssignment]>, DomainRef)>,
}

/// Plug-in estimator for identified interventional distributions (discrete).
#[derive(Clone, Debug)]
pub struct FunctionalDistribution {
    /// Overlap policy (positivity is implicit in CPT support; override allowed).
    pub overlap: OverlapPolicy,
    /// Bootstrap replicates for the interventional mean SE and the per-atom
    /// probability intervals (0 = skip).
    pub bootstrap_replicates: u32,
    /// Two-sided level of the published atom-probability intervals.
    pub confidence_level: f64,
}

impl Default for FunctionalDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionalDistribution {
    /// Create with default overlap override (CPT positivity is data-driven).
    #[must_use]
    pub fn new() -> Self {
        Self {
            overlap: OverlapPolicy::ExplicitOverride,
            bootstrap_replicates: 0,
            confidence_level: 0.95,
        }
    }

    /// Set the two-sided level of the published atom-probability intervals.
    #[must_use]
    pub const fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Set the overlap policy recorded on the estimate artifact.
    ///
    /// Positivity is implicit in the empirical CPT's support, so override is generally safe.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the number of bootstrap replicates used for the interventional mean's standard
    /// error (0 = skip).
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Prepare from an identified `GeneralId` functional and tabular data.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, continuous/high-cardinality columns, empty support,
    /// unsupported interventions, or expression compile failure.
    pub fn prepare(
        &self,
        data: &TabularData,
        query: &InterventionalDistributionQuery,
        estimand: &IdentifiedEstimand,
        arena: &CausalExprArena,
        assumptions: AssumptionSet,
    ) -> Result<PreparedFunctionalDistribution, EstimationError> {
        query.validate()?;
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::TargetPopulation);
        }
        require_method(
            estimand,
            &[EstimandMethod::GeneralId],
            "functional.distribution requires a general.id estimand",
        )?;

        let interventions = set_assignments(&query.interventions)?;
        let conditioning: Vec<InterventionAssignment> = query
            .conditioning
            .iter()
            .map(|&variable| {
                // Conditioning values are supplied at evaluate time per atom when
                // estimating the full conditional table; prepare stores variable ids
                // as placeholders (NaN) only when empty bindings are needed for CPT
                // domain collection. Concrete Z values come from `conditioning_values`
                // on estimate, or from evaluating the conditional density functional
                // which already conditions structurally via IDC.
                InterventionAssignment { variable, value: Value::f64(f64::NAN) }
            })
            .collect();

        let factor_specs = collect_observational_factors(arena, estimand.functional);
        let signatures = collect_factor_signatures(arena, estimand.functional);
        let mut vars_needed = HashSet::new();
        for (vars, cond) in &factor_specs {
            vars_needed.extend(vars.iter().copied());
            vars_needed.extend(cond.iter().copied());
        }
        for &y in query.outcomes.iter() {
            vars_needed.insert(y);
        }
        for a in &interventions {
            vars_needed.insert(a.variable);
        }
        for &z in query.conditioning.iter() {
            vars_needed.insert(z);
        }

        let (provider, columns) =
            build_empirical_provider(data, &vars_needed, &factor_specs, &signatures)?;
        let compiled = arena.compile(estimand.functional).map_err(eval_err)?;

        Ok(PreparedFunctionalDistribution {
            estimand: estimand.clone(),
            arena: Arc::new(arena.clone()),
            compiled,
            provider,
            outcomes: Arc::clone(&query.outcomes),
            interventions: Arc::from(interventions),
            conditioning: Arc::from(conditioning),
            assumptions,
            bootstrap_columns: columns,
            bootstrap_factors: factor_specs,
            bootstrap_signatures: signatures,
        })
    }

    /// Estimate the interventional distribution over the outcome support.
    ///
    /// For unconditional queries, returns P(Y=y | do(X)) for each outcome atom.
    /// For IDC queries with nonempty conditioning:
    /// - if `conditioning_values` is nonempty, binds that single Z point;
    /// - if empty, enumerates the empirical support of Z and returns atoms for each (y, z).
    ///
    /// # Errors
    ///
    /// Partial conditioning bindings, empty support, or evaluation failure.
    pub fn estimate(
        &self,
        prepared: &PreparedFunctionalDistribution,
        conditioning_values: &[(VariableId, Value)],
        workspace: &mut FunctionalDistributionWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<InterventionalDistributionEstimate, EstimationError> {
        if !(self.confidence_level > 0.0 && self.confidence_level < 1.0) {
            return Err(EstimationError::unsupported(
                "functional.distribution confidence_level must lie in (0, 1)",
            ));
        }
        let mut out = self.estimate_point(prepared, conditioning_values, workspace)?;
        if self.bootstrap_replicates == 0 || out.atoms.is_empty() {
            return Ok(out);
        }
        let has_mean = out.mean.is_finite();
        let index: HashMap<AtomKey, usize> = out
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| ((Arc::clone(&a.outcomes), Arc::clone(&a.conditioning)), i))
            .collect();
        let atom_replicates = Mutex::new(vec![Vec::new(); out.atoms.len()]);
        let n_atoms = out.atoms.len();
        let n = prepared.bootstrap_columns.values().next().map_or(0, Vec::len);
        let boot = bootstrap_se(self.bootstrap_replicates, ctx, 0xF01D_u64, n, |idx| {
            let columns = gather_columns(&prepared.bootstrap_columns, idx);
            let provider = provider_from_columns(
                &columns,
                idx.len(),
                &prepared.bootstrap_factors,
                &prepared.bootstrap_signatures,
                None,
            )?;
            let mut prep = prepared.clone();
            prep.provider = provider;
            let mut ws = FunctionalDistributionWorkspace::default();
            let Ok(est) = self.estimate_point(&prep, conditioning_values, &mut ws) else {
                return Ok(None);
            };
            let mut aligned = vec![f64::NAN; n_atoms];
            align_replicate_atoms(&index, &out.atoms, &est.atoms, &mut aligned);
            // The scalar SE tracks the interventional mean exactly as before;
            // tables without a mean let the first defined atom drive the
            // adaptive early-stop, and publish no scalar SE.
            let tracked = if has_mean {
                est.mean
            } else {
                aligned.iter().copied().find(|p| p.is_finite()).unwrap_or(f64::NAN)
            };
            if !tracked.is_finite() {
                return Ok(None);
            }
            let mut reps = atom_replicates.lock().expect("atom replicate lock");
            for (values, &p) in reps.iter_mut().zip(aligned.iter()) {
                if p.is_finite() {
                    values.push(p);
                }
            }
            Ok(Some(tracked))
        })?;
        let atom_replicates = atom_replicates.into_inner().expect("atom replicate lock");
        let attempted = boot.replicates_ok.saturating_add(boot.replicates_failed);
        let uncertainty: Vec<AtomUncertainty> = out
            .atoms
            .iter()
            .zip(&atom_replicates)
            .map(|(atom, values)| {
                let se = crate::util::finalize_bootstrap_se_ex(
                    values,
                    attempted,
                    boot.cancelled,
                    boot.early_stopped,
                )
                .se;
                AtomUncertainty {
                    se_bootstrap: se,
                    replicates_ok: u32::try_from(values.len()).unwrap_or(u32::MAX),
                    interval: logit_probability_interval(
                        atom.probability,
                        se,
                        self.confidence_level,
                    ),
                }
            })
            .collect();
        out.mean_interval = binary_mean_interval(&out.atoms, &uncertainty, has_mean);
        out.atom_uncertainty = Arc::from(uncertainty);
        if has_mean {
            out.se_bootstrap = boot.se;
            out.bootstrap_replicates_ok = Some(boot.replicates_ok);
            out.bootstrap_replicates_failed = Some(boot.replicates_failed);
            out.bootstrap_cancelled = boot.cancelled;
            out.bootstrap_early_stopped = boot.early_stopped;
        }
        Ok(out)
    }

    fn estimate_point(
        &self,
        prepared: &PreparedFunctionalDistribution,
        conditioning_values: &[(VariableId, Value)],
        workspace: &mut FunctionalDistributionWorkspace,
    ) -> Result<InterventionalDistributionEstimate, EstimationError> {
        workspace.clear();

        let needed_z: Vec<VariableId> = prepared.conditioning.iter().map(|a| a.variable).collect();
        let z_points: Vec<Vec<(VariableId, Value)>> = if needed_z.is_empty() {
            if !conditioning_values.is_empty() {
                return Err(EstimationError::unsupported(
                    "conditioning_values supplied for an unconditional distribution query",
                ));
            }
            vec![Vec::new()]
        } else if conditioning_values.is_empty() {
            let support =
                prepared.provider.support(&needed_z, &EvalContext::default()).map_err(eval_err)?;
            support
                .iter()
                .map(|row| needed_z.iter().copied().zip(row.iter().cloned()).collect::<Vec<_>>())
                .collect()
        } else {
            let provided: HashSet<VariableId> =
                conditioning_values.iter().map(|(v, _)| *v).collect();
            let needed: HashSet<VariableId> = needed_z.iter().copied().collect();
            if provided != needed {
                return Err(EstimationError::unsupported(
                    "conditioning_values must bind exactly the query conditioning set",
                ));
            }
            vec![conditioning_values.to_vec()]
        };

        let y_support = prepared
            .provider
            .support(prepared.outcomes.as_ref(), &EvalContext::default())
            .map_err(eval_err)?;
        if y_support.is_empty() {
            return Err(EstimationError::data_msg("empty outcome support"));
        }

        let mut atoms = Vec::with_capacity(y_support.len().saturating_mul(z_points.len().max(1)));
        let mut mean_acc = 0.0;
        let mut mean_ok = prepared.outcomes.len() == 1 && z_points.len() == 1;

        for z_bind in &z_points {
            for row in y_support.iter() {
                workspace.assignment = Assignment::new();
                for a in prepared.interventions.iter() {
                    workspace.assignment.set(a.variable, a.value.clone());
                }
                for (v, val) in z_bind {
                    workspace.assignment.set(*v, val.clone());
                }
                let mut outcome_pairs = Vec::with_capacity(prepared.outcomes.len());
                for (i, &y) in prepared.outcomes.iter().enumerate() {
                    let val = row.get(i).cloned().ok_or_else(|| {
                        EstimationError::data_msg("outcome support row shorter than outcomes")
                    })?;
                    workspace.assignment.set(y, val.clone());
                    outcome_pairs.push((y, val));
                }

                let p = prepared
                    .compiled
                    .evaluate_with(
                        &prepared.arena,
                        &prepared.provider,
                        &EvalContext::default(),
                        &workspace.assignment,
                    )
                    .map_err(eval_err)?;

                if mean_ok {
                    if let Some((_, val)) = outcome_pairs.first() {
                        if let Some(y) = val.as_f64() {
                            mean_acc += p * y;
                        } else {
                            mean_ok = false;
                        }
                    }
                }

                atoms.push(DistributionAtom {
                    outcomes: Arc::from(outcome_pairs),
                    conditioning: Arc::from(z_bind.clone()),
                    probability: p,
                });
            }
        }

        Ok(InterventionalDistributionEstimate {
            atoms: Arc::from(atoms),
            mean: if mean_ok { mean_acc } else { f64::NAN },
            se_analytic: f64::NAN,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions: prepared.assumptions.clone(),
            overlap: self.overlap,
            retained_memory_bytes: None,
            atom_uncertainty: Arc::from([]),
            mean_interval: None,
        })
    }

    /// Bayesian-bootstrap posterior over the identified interventional
    /// distribution: all CPT factors share one Rubin Dirichlet(1, ..., 1)
    /// row-law draw. Published atoms are posterior mean probabilities, with
    /// joint draws stored as `probability_atom_{i}` scalars in atom order.
    /// A scalar mean effect is included only for a single numeric outcome at
    /// one conditioning assignment; joint and conditional tables need no scalar.
    ///
    /// # Errors
    ///
    /// Evaluation failure or a non-finite posterior draw.
    pub fn estimate_bayesian(
        &self,
        prepared: &PreparedFunctionalDistribution,
        conditioning_values: &[(VariableId, Value)],
        n_draws: usize,
        identification: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<(InterventionalDistributionEstimate, crate::CausalPosterior), EstimationError> {
        let mut ws = FunctionalDistributionWorkspace::default();
        let point = self.estimate_point(prepared, conditioning_values, &mut ws)?;
        let n = prepared.bootstrap_columns.values().next().map_or(0, Vec::len);
        if n == 0 {
            return Err(EstimationError::data_msg(
                "functional Bayesian requires complete discrete rows",
            ));
        }
        let draws = crate::require_bayesian_n_draws(n_draws)?;
        let has_mean = point.mean.is_finite();
        let atom_offset = usize::from(has_mean);
        let mut quantities = Vec::with_capacity(atom_offset + point.atoms.len());
        if has_mean {
            quantities.push(PosteriorQuantityKind::Effect { name: Arc::from("functional") });
        }
        quantities.extend((0..point.atoms.len()).map(|i| PosteriorQuantityKind::Scalar {
            name: Arc::from(format!("probability_atom_{i}")),
        }));
        let mut values = vec![0.0; draws * quantities.len()];
        let mut rng = ctx.rng.stream(0xF01E_u64);
        let mut draw_prepared = prepared.clone();
        let mut draw_ws = FunctionalDistributionWorkspace::default();
        for draw in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(EstimationError::unsupported("functional Bayesian cancelled"));
            }
            let provider = provider_from_bayesian_bootstrap(
                &prepared.bootstrap_columns,
                n,
                &prepared.bootstrap_factors,
                &prepared.bootstrap_signatures,
                &mut rng,
            )?;
            draw_prepared.provider = provider;
            let est = self.estimate_point(&draw_prepared, conditioning_values, &mut draw_ws)?;
            if has_mean {
                if !est.mean.is_finite() {
                    return Err(EstimationError::stats_msg(
                        "functional Bayesian mean was non-finite",
                    ));
                }
                values[draw] = est.mean;
            }
            if est.atoms.len() != point.atoms.len() {
                return Err(EstimationError::stats_msg("functional Bayesian atom support changed"));
            }
            for (i, (atom, expected)) in est.atoms.iter().zip(point.atoms.iter()).enumerate() {
                if atom.outcomes != expected.outcomes
                    || atom.conditioning != expected.conditioning
                    || !atom.probability.is_finite()
                {
                    return Err(EstimationError::stats_msg("functional Bayesian atom was invalid"));
                }
                values[(atom_offset + i) * draws + draw] = atom.probability;
            }
        }
        let draws = PosteriorDraws::from_column_major(
            PosteriorSchema { quantities: Arc::from(quantities) },
            draws,
            Arc::<[f64]>::from(values),
        )
        .map_err(|e| EstimationError::stats_msg(e.to_string()))?;
        let posterior =
            functional_posterior_from_draws(draws, prepared.assumptions.clone(), identification);
        let mut out = point;
        out.assumptions = posterior.assumptions.clone();
        // Expectation is linear in the final probability table: averaging its
        // cells preserves normalization and agrees with the scalar mean draws.
        for (i, atom) in Arc::make_mut(&mut out.atoms).iter_mut().enumerate() {
            atom.probability = posterior.summaries.mean[atom_offset + i];
        }
        if has_mean {
            out.mean = posterior.summaries.mean[0];
            out.se_analytic = posterior.summaries.sd[0];
        }
        Ok((out, posterior))
    }
}

/// Prepared scalar functional (ATE / path-specific NE contrast).
#[derive(Clone, Debug)]
pub struct PreparedFunctionalEffect {
    /// Identified estimand.
    pub estimand: IdentifiedEstimand,
    /// Arena owning the functional. Shared (never mutated after `prepare`);
    /// see [`PreparedFunctionalDistribution::arena`].
    pub arena: Arc<CausalExprArena>,
    /// Compiled evaluator.
    pub compiled: CompiledEvaluator,
    /// Empirical CPT provider.
    pub provider: EmpiricalTableProvider,
    /// Assumptions from identification.
    pub assumptions: AssumptionSet,
    bootstrap_columns: HashMap<VariableId, Vec<Option<Value>>>,
    bootstrap_factors: Vec<(Arc<[VariableId]>, Arc<[VariableId]>)>,
    bootstrap_signatures:
        Vec<(Arc<[VariableId]>, Arc<[VariableId]>, Arc<[InterventionAssignment]>, DomainRef)>,
}

/// Discrete plug-in estimator for identified scalar functionals (contrasts).
#[derive(Clone, Debug)]
pub struct FunctionalEffect {
    /// Overlap policy.
    pub overlap: OverlapPolicy,
    /// Bootstrap replicates for the scalar SE (0 = skip).
    pub bootstrap_replicates: u32,
}

impl Default for FunctionalEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionalEffect {
    /// Create with explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self { overlap: OverlapPolicy::ExplicitOverride, bootstrap_replicates: 0 }
    }

    /// Set the overlap policy recorded on the estimate artifact.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the number of bootstrap replicates used for the scalar functional's standard
    /// error (0 = skip).
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Prepare CPT plug-in for a path-specific / general-ID contrast functional.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, continuous columns, or compile failure.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        arena: &CausalExprArena,
        assumptions: AssumptionSet,
        extra_vars: &[VariableId],
    ) -> Result<PreparedFunctionalEffect, EstimationError> {
        require_method(
            estimand,
            &[EstimandMethod::PathSpecificNatural, EstimandMethod::GeneralId],
            "functional.effect requires path_specific.natural or general.id",
        )?;
        let factor_specs = collect_observational_factors(arena, estimand.functional);
        let signatures = collect_factor_signatures(arena, estimand.functional);
        let mut vars_needed = HashSet::new();
        for (vars, cond) in &factor_specs {
            vars_needed.extend(vars.iter().copied());
            vars_needed.extend(cond.iter().copied());
        }
        vars_needed.extend(extra_vars.iter().copied());
        let (provider, columns) =
            build_empirical_provider(data, &vars_needed, &factor_specs, &signatures)?;
        let compiled = arena.compile(estimand.functional).map_err(eval_err)?;
        Ok(PreparedFunctionalEffect {
            estimand: estimand.clone(),
            arena: Arc::new(arena.clone()),
            compiled,
            provider,
            assumptions,
            bootstrap_columns: columns,
            bootstrap_factors: factor_specs,
            bootstrap_signatures: signatures,
        })
    }

    /// Evaluate the scalar functional.
    ///
    /// # Errors
    ///
    /// Evaluation / missing CPT entries.
    pub fn estimate(
        &self,
        prepared: &PreparedFunctionalEffect,
        _workspace: &mut FunctionalDistributionWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<crate::adjustment::EffectEstimate, EstimationError> {
        let ate = prepared
            .compiled
            .evaluate(&prepared.arena, &prepared.provider, &EvalContext::default())
            .map_err(eval_err)?;
        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            let n = prepared.bootstrap_columns.values().next().map_or(0, Vec::len);
            Some(bootstrap_se(self.bootstrap_replicates, ctx, 0xF02D_u64, n, |idx| {
                let columns = gather_columns(&prepared.bootstrap_columns, idx);
                let provider = provider_from_columns(
                    &columns,
                    idx.len(),
                    &prepared.bootstrap_factors,
                    &prepared.bootstrap_signatures,
                    None,
                )?;
                match prepared.compiled.evaluate(
                    &prepared.arena,
                    &provider,
                    &EvalContext::default(),
                ) {
                    Ok(v) if v.is_finite() => Ok(Some(v)),
                    _ => Ok(None),
                }
            })?)
        };
        Ok(crate::adjustment::EffectEstimate::new(
            ate,
            // Multinomial delta-method analytic SE is out of scope for the discrete plug-in.
            f64::NAN,
            prepared.assumptions.clone(),
            self.overlap,
        )
        .with_bootstrap(boot))
    }

    /// Bayesian bootstrap of the identified functional. All CPT factors are
    /// derived from a shared Rubin (1981) Dirichlet(1, ..., 1) draw over
    /// empirical rows.
    /// Unobserved joint cells retain zero mass; this is an empirical-support
    /// posterior, not a positive pseudocount prior over unobserved categories.
    ///
    /// # Errors
    ///
    /// Evaluation / missing CPT entries, or a non-finite posterior.
    pub fn estimate_bayesian(
        &self,
        prepared: &PreparedFunctionalEffect,
        n_draws: usize,
        identification: antecedent_core::IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<crate::CausalPosterior, EstimationError> {
        let n = prepared.bootstrap_columns.values().next().map_or(0, Vec::len);
        if n == 0 {
            return Err(EstimationError::data_msg(
                "functional Bayesian requires complete discrete rows",
            ));
        }
        let draws = crate::require_bayesian_n_draws(n_draws)?;
        let mut values = Vec::with_capacity(draws);
        let mut rng = ctx.rng.stream(0xF02E_u64);
        for _ in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(EstimationError::unsupported("functional Bayesian cancelled"));
            }
            let provider = provider_from_bayesian_bootstrap(
                &prepared.bootstrap_columns,
                n,
                &prepared.bootstrap_factors,
                &prepared.bootstrap_signatures,
                &mut rng,
            )?;
            let value = prepared
                .compiled
                .evaluate(&prepared.arena, &provider, &EvalContext::default())
                .map_err(eval_err)?;
            if !value.is_finite() {
                return Err(EstimationError::stats_msg("functional Bayesian draw was non-finite"));
            }
            values.push(value);
        }
        functional_posterior(values, prepared.assumptions.clone(), identification)
    }
}

/// Whether `err` is the evaluator refusing a required empirical CPT cell.
#[must_use]
pub const fn functional_cell_unevaluable(err: &EvalError) -> bool {
    matches!(
        err,
        EvalError::MissingTableEntry | EvalError::EmptySupport(_) | EvalError::DivisionByZero
    )
}

/// Map a functional-evaluator cell decision onto the shared [`SupportReport`].
///
/// `None` means every required empirical CPT / conditioning cell was evaluable,
/// so [`SupportStatus::Supported`] is earned. A missing, empty, or undefined
/// (ratio-zero) required cell is [`SupportStatus::OutsideEmpiricalSupport`] —
/// the same report type other estimators use, not a new positivity theory.
/// Other eval failures stay with the caller.
///
/// # Errors
///
/// Eval failures that are not cell-support refusals.
pub fn support_from_functional_eval(err: Option<&EvalError>) -> Result<SupportReport, EvalError> {
    match err {
        None => Ok(SupportReport {
            status: SupportStatus::Supported,
            query_region: SupportRegion { minima: Arc::from([]), maxima: Arc::from([]) },
            diagnostics: Vec::new(),
            warnings: Vec::new(),
            point_status: None,
        }),
        Some(e) if functional_cell_unevaluable(e) => Ok(SupportReport {
            status: SupportStatus::OutsideEmpiricalSupport,
            query_region: SupportRegion { minima: Arc::from([]), maxima: Arc::from([]) },
            diagnostics: vec![SupportDiagnostic {
                id: Arc::from("functional.required_cell"),
                values: Arc::from([]),
                detail: Arc::from(e.to_string()),
            }],
            warnings: vec![Diagnostic::new(
                "functional.required_cell",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                e.to_string(),
            )],
            point_status: None,
        }),
        Some(e) => Err(e.clone()),
    }
}

fn set_assignments(
    interventions: &[Intervention],
) -> Result<Vec<InterventionAssignment>, EstimationError> {
    let mut out = Vec::with_capacity(interventions.len());
    for iv in interventions {
        match iv {
            Intervention::Set { variable, value } => {
                if value.as_f64().is_some_and(f64::is_nan) {
                    return Err(EstimationError::unsupported(
                        "functional.distribution requires concrete Set intervention values",
                    ));
                }
                out.push(InterventionAssignment { variable: *variable, value: value.clone() });
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "functional.distribution supports hard Set interventions only",
                ));
            }
        }
    }
    Ok(out)
}

fn collect_observational_factors(
    arena: &CausalExprArena,
    root: ExprId,
) -> Vec<(Arc<[VariableId]>, Arc<[VariableId]>)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match arena.node(id) {
            ExprNode::Distribution { variables, conditioned_on, .. } => {
                let vars: Arc<[VariableId]> = Arc::from(arena.var_set(*variables).to_vec());
                let cond: Arc<[VariableId]> = Arc::from(arena.var_set(*conditioned_on).to_vec());
                let key = (vars.clone(), cond.clone());
                if seen.insert((vars.as_ref().to_vec(), cond.as_ref().to_vec())) {
                    out.push(key);
                }
            }
            ExprNode::Product(list) => {
                for &c in arena.list(*list) {
                    stack.push(c);
                }
            }
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                stack.push(*expr);
            }
            ExprNode::Ratio { numerator, denominator } => {
                stack.push(*numerator);
                stack.push(*denominator);
            }
            ExprNode::Expectation { distribution, .. } => stack.push(*distribution),
            ExprNode::Contrast { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            }
        }
    }
    out
}

/// Collect (variables, `conditioned_on`, intervention set, domain) for CPT duplication.
fn collect_factor_signatures(
    arena: &CausalExprArena,
    root: ExprId,
) -> Vec<(Arc<[VariableId]>, Arc<[VariableId]>, Arc<[InterventionAssignment]>, DomainRef)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match arena.node(id) {
            ExprNode::Distribution { variables, conditioned_on, intervention, domain } => {
                let vars: Arc<[VariableId]> = Arc::from(arena.var_set(*variables).to_vec());
                let cond: Arc<[VariableId]> = Arc::from(arena.var_set(*conditioned_on).to_vec());
                let interv: Arc<[InterventionAssignment]> =
                    Arc::from(arena.intervention_assignments(*intervention).to_vec());
                let key = (
                    vars.as_ref().to_vec(),
                    cond.as_ref().to_vec(),
                    interv.iter().map(|a| (a.variable.raw(), a.value.clone())).collect::<Vec<_>>(),
                    *domain,
                );
                if seen.insert(key) {
                    out.push((vars, cond, interv, *domain));
                }
            }
            ExprNode::Product(list) => {
                for &c in arena.list(*list) {
                    stack.push(c);
                }
            }
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                stack.push(*expr);
            }
            ExprNode::Ratio { numerator, denominator } => {
                stack.push(*numerator);
                stack.push(*denominator);
            }
            ExprNode::Expectation { distribution, .. } => stack.push(*distribution),
            ExprNode::Contrast { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            }
        }
    }
    out
}

fn build_empirical_provider(
    data: &TabularData,
    vars_needed: &HashSet<VariableId>,
    factors: &[(Arc<[VariableId]>, Arc<[VariableId]>)],
    signatures: &[(
        Arc<[VariableId]>,
        Arc<[VariableId]>,
        Arc<[InterventionAssignment]>,
        DomainRef,
    )],
) -> Result<(EmpiricalTableProvider, HashMap<VariableId, Vec<Option<Value>>>), EstimationError> {
    let mut columns: HashMap<VariableId, Vec<Option<Value>>> = HashMap::new();
    let n = data.row_count();

    for &id in vars_needed {
        let (col, _domain) = discrete_column(data, id)?;
        if col.len() != n {
            return Err(EstimationError::data_msg("column length mismatch"));
        }
        columns.insert(id, col);
    }

    // A functional's factors must describe the same complete-case law. Using
    // a different retained sample for each factor can violate the chain rule.
    let complete: Vec<_> =
        (0..n).filter(|&row| columns.values().all(|col| col[row].is_some())).collect();
    if complete.is_empty() {
        return Err(EstimationError::data_msg(
            "no jointly complete rows for functional evaluation",
        ));
    }
    let columns = gather_columns(&columns, &complete);
    let provider = provider_from_columns(&columns, complete.len(), factors, signatures, None)?;
    Ok((provider, columns))
}

/// `(outcomes, conditioning)` identity of one distribution atom.
type AtomKey = (Arc<[(VariableId, Value)]>, Arc<[(VariableId, Value)]>);

/// Map one bootstrap replicate's atoms onto the point-estimate atom order.
///
/// A resample can lose support. An outcome level missing under a conditioning
/// assignment the replicate still has is an empirical probability of 0; an atom
/// whose conditioning assignment vanished is undefined in that replicate (NaN).
fn align_replicate_atoms(
    index: &HashMap<AtomKey, usize>,
    point: &[DistributionAtom],
    replicate: &[DistributionAtom],
    aligned: &mut [f64],
) {
    aligned.fill(f64::NAN);
    let mut present: HashSet<&[(VariableId, Value)]> = HashSet::new();
    for atom in replicate {
        present.insert(atom.conditioning.as_ref());
        let key = (Arc::clone(&atom.outcomes), Arc::clone(&atom.conditioning));
        if let Some(&i) = index.get(&key) {
            aligned[i] = atom.probability;
        }
    }
    for (slot, atom) in aligned.iter_mut().zip(point) {
        if slot.is_nan() && present.contains(atom.conditioning.as_ref()) {
            *slot = 0.0;
        }
    }
}

/// The mean's interval when the single outcome is binary `{0, 1}` (then the
/// mean is `P(Y = 1)`): the `Y = 1` atom's interval, or a boundary refusal when
/// the level `1` never occurs (`p̂ = 0`).
#[allow(clippy::float_cmp, reason = "outcome levels are exact data values, not arithmetic")]
fn binary_mean_interval(
    atoms: &[DistributionAtom],
    uncertainty: &[AtomUncertainty],
    has_mean: bool,
) -> Option<ProbabilityInterval> {
    if !has_mean || atoms.len() != uncertainty.len() {
        return None;
    }
    let level = |a: &DistributionAtom| match a.outcomes.as_ref() {
        [(_, v)] => v.as_f64(),
        _ => None,
    };
    if !atoms.iter().all(|a| matches!(level(a), Some(x) if x == 0.0 || x == 1.0)) {
        return None;
    }
    Some(atoms.iter().zip(uncertainty).find(|(a, _)| level(a) == Some(1.0)).map_or(
        ProbabilityInterval::Unavailable(ProbabilityIntervalUnavailable::BoundaryEstimate),
        |(_, u)| u.interval,
    ))
}

fn gather_columns(
    columns: &HashMap<VariableId, Vec<Option<Value>>>,
    idx: &[usize],
) -> HashMap<VariableId, Vec<Option<Value>>> {
    columns
        .iter()
        .map(|(&id, col)| {
            let gathered: Vec<Option<Value>> =
                idx.iter().map(|&i| col.get(i).cloned().flatten()).collect();
            (id, gathered)
        })
        .collect()
}

fn unidentified_mass_from_status(status: IdentificationStatus) -> f64 {
    match status {
        IdentificationStatus::NotIdentified => 1.0,
        IdentificationStatus::NonparametricallyIdentified
        | IdentificationStatus::IdentifiedUnderParametricRestrictions
        | IdentificationStatus::IdentifiedUnderPriorRestrictions
        | IdentificationStatus::PartiallyIdentified
        | IdentificationStatus::GraphDependent => 0.0,
    }
}

fn functional_posterior(
    values: Vec<f64>,
    assumptions: AssumptionSet,
    identification: IdentificationStatus,
) -> Result<crate::CausalPosterior, EstimationError> {
    let n = values.len();
    let schema = PosteriorSchema {
        quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("functional") }]),
    };
    let draws = PosteriorDraws::from_column_major(schema, n, Arc::<[f64]>::from(values))
        .map_err(|e| EstimationError::stats_msg(e.to_string()))?;
    Ok(functional_posterior_from_draws(draws, assumptions, identification))
}

fn functional_posterior_from_draws(
    draws: PosteriorDraws,
    mut assumptions: AssumptionSet,
    identification: IdentificationStatus,
) -> crate::CausalPosterior {
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from("functional.empirical_support_prior"),
            description: Arc::from("Rubin Bayesian bootstrap of one observational row law; all factors share Dirichlet(1, ..., 1) row masses. Unobserved joint cells have zero posterior mass; coefficient prior_scale and backend selection do not define this nonparametric posterior."),
        }),
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("functional.dirichlet") },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let summaries = draws.summarize();
    crate::CausalPosterior {
        subsampled_out_mass: 0.0,
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: InferenceDiagnostics::analytic("functional.dirichlet"),
        assumptions,
        unidentified_mass: unidentified_mass_from_status(identification),
        early_stopped: false,
        treatment_contrast: None,
    }
}

fn provider_from_columns(
    columns: &HashMap<VariableId, Vec<Option<Value>>>,
    n: usize,
    factors: &[(Arc<[VariableId]>, Arc<[VariableId]>)],
    signatures: &[(
        Arc<[VariableId]>,
        Arc<[VariableId]>,
        Arc<[InterventionAssignment]>,
        DomainRef,
    )],
    weights: Option<&[f64]>,
) -> Result<EmpiricalTableProvider, EstimationError> {
    let mut domains: HashMap<VariableId, Vec<Value>> = HashMap::new();
    for (&id, col) in columns {
        let mut seen = HashSet::new();
        let mut domain = Vec::new();
        for cell in col {
            if let Some(val) = cell {
                if seen.insert(val.clone()) {
                    domain.push(val.clone());
                }
            }
        }
        domains.insert(id, domain);
    }

    let mut provider = EmpiricalTableProvider::new();
    for (id, domain) in &domains {
        provider.set_domain(*id, domain.iter().cloned());
    }

    // Vacuous empty factor used by some ID edge cases.
    let empty_spec = FactorSpec {
        variables: &[],
        conditioned_on: &[],
        intervention: &[],
        domain: DomainRef::Observational,
    };
    provider.insert_probability(&empty_spec, &Assignment::from_pairs([]), 1.0).map_err(eval_err)?;

    for (vars, cond) in factors {
        if vars.is_empty() && cond.is_empty() {
            continue;
        }
        // Observational CPT.
        insert_cpt(&mut provider, columns, n, vars, cond, &[], DomainRef::Observational, weights)?;
        // Duplicate under every interventional signature with the same (vars, cond).
        for (s_vars, s_cond, interv, domain) in signatures {
            if s_vars.as_ref() != vars.as_ref() || s_cond.as_ref() != cond.as_ref() {
                continue;
            }
            if *domain == DomainRef::Observational && interv.is_empty() {
                continue;
            }
            // Intervened coordinates in `vars` are Dirac under do(.); other factors
            // reuse the observational CPT under the interventional FactorKey.
            let intervened_in_vars: Vec<_> =
                interv.iter().filter(|a| vars.iter().any(|&v| v == a.variable)).cloned().collect();
            if intervened_in_vars.is_empty() {
                insert_cpt(
                    &mut provider,
                    columns,
                    n,
                    vars,
                    cond,
                    interv.as_ref(),
                    *domain,
                    weights,
                )?;
            } else {
                insert_dirac_intervened(
                    &mut provider,
                    &domains,
                    vars,
                    cond,
                    interv.as_ref(),
                    *domain,
                    &intervened_in_vars,
                )?;
            }
        }
    }

    Ok(provider)
}

// One random observational law per draw. All marginal/conditional factors and
// interventional aliases are derived from the same row masses, preserving
// probability identities even when the compiled expression reuses a kernel.
fn provider_from_bayesian_bootstrap(
    columns: &HashMap<VariableId, Vec<Option<Value>>>,
    n: usize,
    factors: &[(Arc<[VariableId]>, Arc<[VariableId]>)],
    signatures: &[(
        Arc<[VariableId]>,
        Arc<[VariableId]>,
        Arc<[InterventionAssignment]>,
        DomainRef,
    )],
    rng: &mut CausalRng,
) -> Result<EmpiricalTableProvider, EstimationError> {
    let weights: Vec<f64> = (0..n).map(|_| -rng.next_f64().max(f64::MIN_POSITIVE).ln()).collect();
    provider_from_columns(columns, n, factors, signatures, Some(&weights))
}

fn insert_dirac_intervened(
    provider: &mut EmpiricalTableProvider,
    domains: &HashMap<VariableId, Vec<Value>>,
    vars: &[VariableId],
    cond: &[VariableId],
    intervention: &[InterventionAssignment],
    domain: DomainRef,
    intervened_in_vars: &[InterventionAssignment],
) -> Result<(), EstimationError> {
    // Free vars = vars not fixed by intervention.
    let free: Vec<VariableId> = vars
        .iter()
        .copied()
        .filter(|v| !intervened_in_vars.iter().any(|a| a.variable == *v))
        .collect();
    let free_rows = cartesian_domain(domains, &free)?;
    let cond_rows = cartesian_domain(domains, cond)?;
    for free_vals in &free_rows {
        for cond_vals in &cond_rows {
            let mut assign = Assignment::new();
            for a in intervened_in_vars {
                assign.set(a.variable, a.value.clone());
            }
            for (v, val) in free.iter().copied().zip(free_vals.iter().cloned()) {
                assign.set(v, val);
            }
            for (v, val) in cond.iter().copied().zip(cond_vals.iter().cloned()) {
                assign.set(v, val);
            }
            // Probability 1: intervened vars are fixed; free vars still need a
            // density — if there are free vars, fall back is wrong. For pure
            // Dirac on all vars, mass is 1 only for the intervened assignment.
            let p = if free.is_empty() {
                1.0
            } else {
                // Should not happen for ID treatment factors; refuse.
                return Err(EstimationError::unsupported(
                    "intervened factor with free variables is unsupported in functional.effect",
                ));
            };
            let spec = FactorSpec { variables: vars, conditioned_on: cond, intervention, domain };
            provider.insert_probability(&spec, &assign, p).map_err(eval_err)?;
        }
    }
    Ok(())
}

fn cartesian_domain(
    domains: &HashMap<VariableId, Vec<Value>>,
    vars: &[VariableId],
) -> Result<Vec<Vec<Value>>, EstimationError> {
    if vars.is_empty() {
        return Ok(vec![Vec::new()]);
    }
    let mut rows: Vec<Vec<Value>> = vec![Vec::new()];
    for &v in vars {
        let domain = domains.get(&v).ok_or_else(|| EstimationError::data_msg("missing domain"))?;
        let mut next = Vec::with_capacity(rows.len() * domain.len());
        for prefix in &rows {
            for val in domain {
                let mut row = prefix.clone();
                row.push(val.clone());
                next.push(row);
            }
        }
        rows = next;
    }
    Ok(rows)
}

fn insert_cpt(
    provider: &mut EmpiricalTableProvider,
    columns: &HashMap<VariableId, Vec<Option<Value>>>,
    n: usize,
    vars: &[VariableId],
    cond: &[VariableId],
    intervention: &[InterventionAssignment],
    domain: DomainRef,
    weights: Option<&[f64]>,
) -> Result<(), EstimationError> {
    // Count (vars, cond) joint and cond marginal among complete cases.
    let mut joint: HashMap<Vec<Value>, f64> = HashMap::new();
    let mut marg: HashMap<Vec<Value>, f64> = HashMap::new();

    for row in 0..n {
        let weight = weights.map_or(1.0, |w| w.get(row).copied().unwrap_or(0.0));
        if weight <= 0.0 {
            continue;
        }
        let mut ok = true;
        let mut cond_vals = Vec::with_capacity(cond.len());
        for &v in cond {
            if let Some(val) = columns.get(&v).and_then(|c| c.get(row)).and_then(|o| o.as_ref()) {
                cond_vals.push(val.clone());
            } else {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let mut var_vals = Vec::with_capacity(vars.len());
        for &v in vars {
            if let Some(val) = columns.get(&v).and_then(|c| c.get(row)).and_then(|o| o.as_ref()) {
                var_vals.push(val.clone());
            } else {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        *marg.entry(cond_vals.clone()).or_insert(0.0) += weight;
        let mut key = var_vals;
        key.extend(cond_vals);
        *joint.entry(key).or_insert(0.0) += weight;
    }

    if cond.is_empty() {
        let total: f64 = joint.values().sum();
        if total <= 0.0 {
            return Err(EstimationError::data_msg("no complete cases for CPT"));
        }
        let var_rows = cartesian_domain(&domains_for_insert(columns, vars)?, vars)?;
        for var_vals in &var_rows {
            let key = var_vals.clone();
            let count = joint.get(&key).copied().unwrap_or(0.0);
            let assign = Assignment::from_pairs(vars.iter().copied().zip(var_vals.iter().cloned()));
            let spec = FactorSpec { variables: vars, conditioned_on: cond, intervention, domain };
            provider.insert_probability(&spec, &assign, count / total).map_err(eval_err)?;
        }
    } else {
        let var_rows = cartesian_domain(&domains_for_insert(columns, vars)?, vars)?;
        let cond_rows = cartesian_domain(&domains_for_insert(columns, cond)?, cond)?;
        for cond_vals in &cond_rows {
            let cond_count = marg.get(cond_vals).copied().unwrap_or(0.0);
            // An unobserved conditioning cell is not P=0. Leave it absent so
            // `probability()` returns `MissingTableEntry` — the evaluator's
            // existing "can I evaluate this cell?" contract.
            if cond_count <= 0.0 {
                continue;
            }
            for var_vals in &var_rows {
                let mut key = var_vals.clone();
                key.extend(cond_vals.iter().cloned());
                let count = joint.get(&key).copied().unwrap_or(0.0);
                let p = count / cond_count;
                let assign = Assignment::from_pairs(
                    vars.iter()
                        .copied()
                        .zip(var_vals.iter().cloned())
                        .chain(cond.iter().copied().zip(cond_vals.iter().cloned())),
                );
                let spec =
                    FactorSpec { variables: vars, conditioned_on: cond, intervention, domain };
                provider.insert_probability(&spec, &assign, p).map_err(eval_err)?;
            }
        }
    }
    Ok(())
}

fn domains_for_insert(
    columns: &HashMap<VariableId, Vec<Option<Value>>>,
    vars: &[VariableId],
) -> Result<HashMap<VariableId, Vec<Value>>, EstimationError> {
    let mut domains = HashMap::new();
    for &v in vars {
        let col = columns.get(&v).ok_or_else(|| EstimationError::data_msg("missing column"))?;
        let mut seen = HashSet::new();
        let mut domain = Vec::new();
        for cell in col {
            if let Some(val) = cell {
                if seen.insert(val.clone()) {
                    domain.push(val.clone());
                }
            }
        }
        if domain.is_empty() {
            return Err(EstimationError::data_msg("empty domain in CPT insert"));
        }
        domain.sort_by(|a, b| match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
            _ => std::cmp::Ordering::Equal,
        });
        domains.insert(v, domain);
    }
    Ok(domains)
}

fn discrete_column(
    data: &TabularData,
    id: VariableId,
) -> Result<(Vec<Option<Value>>, Vec<Value>), EstimationError> {
    let view = data.column(id).map_err(EstimationError::from)?;
    let n = view.len();
    let validity = view.validity();
    let mut values = Vec::with_capacity(n);
    let mut domain_set: HashMap<Value, ()> = HashMap::new();
    let mut domain = Vec::new();

    match view {
        ColumnView::Float64(c) => {
            for i in 0..n {
                if !validity.is_valid(i)
                    || data.storage().analysis_mask().is_some_and(|mask| !mask.is_valid(i))
                {
                    values.push(None);
                    continue;
                }
                let v = Value::f64(c.values[i]);
                if domain_set.insert(v.clone(), ()).is_none() {
                    domain.push(v.clone());
                }
                values.push(Some(v));
            }
        }
        ColumnView::Int64(c) => {
            for i in 0..n {
                if !validity.is_valid(i)
                    || data.storage().analysis_mask().is_some_and(|mask| !mask.is_valid(i))
                {
                    values.push(None);
                    continue;
                }
                let v = Value::Int64(c.values[i]);
                if domain_set.insert(v.clone(), ()).is_none() {
                    domain.push(v.clone());
                }
                values.push(Some(v));
            }
        }
        ColumnView::Categorical(c) => {
            for i in 0..n {
                if !validity.is_valid(i)
                    || data.storage().analysis_mask().is_some_and(|mask| !mask.is_valid(i))
                {
                    values.push(None);
                    continue;
                }
                let v = Value::Category(c.codes[i].raw());
                if domain_set.insert(v.clone(), ()).is_none() {
                    domain.push(v.clone());
                }
                values.push(Some(v));
            }
        }
        _ => {
            return Err(EstimationError::unsupported(
                "functional.distribution supports float64 / int64 / categorical columns only",
            ));
        }
    }

    if domain.is_empty() {
        return Err(EstimationError::data_msg("empty discrete domain"));
    }
    if domain.len() > MAX_DISCRETE_LEVELS {
        return Err(EstimationError::unsupported(
            "variable exceeds discrete level cap for functional.distribution",
        ));
    }
    // Stable order by Display/hash — sort by f64/i64 when possible.
    domain.sort_by(|a, b| match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        _ => std::cmp::Ordering::Equal,
    });
    Ok((values, domain))
}

fn eval_err(e: EvalError) -> EstimationError {
    EstimationError::data_msg(e.to_string())
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_identify::{IdIdentifier, IdentificationStatus, IdentificationWorkspace};

    use super::*;

    fn f(x: f64) -> Value {
        Value::f64(x)
    }

    fn binary_confounding_table() -> TabularData {
        // Z, T, Y with known interventional mean E[Y|do(T=1)] = 0.7
        // Rows generated from: P(Z)=0.5, P(T|Z)=..., P(Y|T,Z) matching id_scm tables.
        // Simplified: enumerate all (Z,T,Y) with multiplicity proportional to joint.
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "y", "z"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();

        // Joint from: P(Z=0)=P(Z=1)=0.5
        // P(T=1|Z=0)=0.4, P(T=1|Z=1)=0.7 (arbitrary; only Y|T,Z and P(Z) matter for do)
        // E[Y|T,Z] as in id_scm: (1,0)->0.8, (1,1)->0.6, (0,0)->0.3, (0,1)->0.2
        // Use 200 rows.
        let mut t_vals = Vec::new();
        let mut y_vals = Vec::new();
        let mut z_vals = Vec::new();
        let combos = [
            // (z, t, y, count) — counts encode joint
            (0.0, 0.0, 0.0, 21), // P(Y=0|T=0,Z=0)=0.7 → among T=0,Z=0
            (0.0, 0.0, 1.0, 9),  // 0.3
            (0.0, 1.0, 0.0, 4),  // P(Y=0|T=1,Z=0)=0.2
            (0.0, 1.0, 1.0, 16), // 0.8
            (1.0, 0.0, 0.0, 12), // P(Y=0|T=0,Z=1)=0.8
            (1.0, 0.0, 1.0, 3),  // 0.2
            (1.0, 1.0, 0.0, 14), // P(Y=0|T=1,Z=1)=0.4
            (1.0, 1.0, 1.0, 21), // 0.6
        ];
        // Normalize Z marginal toward 0.5 by the counts above:
        // Z=0: 21+9+4+16=50, Z=1: 12+3+14+21=50. Good.
        for (z, t, y, count) in combos {
            for _ in 0..count {
                z_vals.push(z);
                t_vals.push(t);
                y_vals.push(y);
            }
        }
        let n = t_vals.len();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t_vals),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y_vals),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z_vals),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TabularData::new(storage)
    }

    #[test]
    fn plug_in_matches_known_interventional_mean() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/functional_plugin/expected.json"
        ))
        .unwrap();
        let mut dag = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        dag.insert_directed(z, t).unwrap();
        dag.insert_directed(z, y).unwrap();
        dag.insert_directed(t, y).unwrap();

        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let query = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), f(1.0))],
        );
        let cq = antecedent_core::CausalQuery::Distribution(query.clone());
        let mut ws = IdentificationWorkspace::default();
        let id_res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(id_res.status, IdentificationStatus::NonparametricallyIdentified);

        let data = binary_confounding_table();
        let est = FunctionalDistribution::new();
        let prepared = est
            .prepare(
                &data,
                &query,
                &id_res.estimands[0],
                &id_res.arena,
                id_res.required_assumptions.clone(),
            )
            .unwrap();
        let mut ews = FunctionalDistributionWorkspace::default();
        let out = est.estimate(&prepared, &[], &mut ews, &ExecutionContext::for_tests(0)).unwrap();
        let tolerance = fixture["acceptance"]["atol"].as_f64().unwrap();
        let expected_mean = fixture["case"]["interventional_mean"].as_f64().unwrap();
        assert!(
            (out.mean - expected_mean).abs() <= tolerance,
            "mean={} atoms={:?}",
            out.mean,
            out.atoms
        );
        let mass: f64 = out.atoms.iter().map(|a| a.probability).sum();
        let expected_mass = fixture["case"]["atom_probability_sum"].as_f64().unwrap();
        assert!((mass - expected_mass).abs() <= tolerance, "mass={mass}");
    }

    #[test]
    fn plug_in_bootstrap_se_is_finite() {
        let mut dag = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        dag.insert_directed(z, t).unwrap();
        dag.insert_directed(z, y).unwrap();
        dag.insert_directed(t, y).unwrap();

        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let query = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), f(1.0))],
        );
        let cq = antecedent_core::CausalQuery::Distribution(query.clone());
        let mut ws = IdentificationWorkspace::default();
        let id_res = id.identify(&prep, &cq, &mut ws).unwrap();

        let data = binary_confounding_table();
        let est =
            FunctionalDistribution { bootstrap_replicates: 40, ..FunctionalDistribution::new() };
        let prepared = est
            .prepare(
                &data,
                &query,
                &id_res.estimands[0],
                &id_res.arena,
                id_res.required_assumptions.clone(),
            )
            .unwrap();
        let mut ews = FunctionalDistributionWorkspace::default();
        let out = est.estimate(&prepared, &[], &mut ews, &ExecutionContext::for_tests(7)).unwrap();
        let se = out.se_bootstrap.expect("bootstrap SE");
        assert!(se.is_finite() && se > 0.0, "se={se}");
    }

    #[test]
    fn logit_interval_refuses_boundary_probabilities() {
        let boundary =
            ProbabilityInterval::Unavailable(ProbabilityIntervalUnavailable::BoundaryEstimate);
        for p in [0.0, 1.0, -0.1, 1.1, f64::NAN] {
            assert_eq!(logit_probability_interval(p, Some(0.02), 0.95), boundary, "p={p}");
        }
        assert_eq!(
            logit_probability_interval(0.3, None, 0.95),
            ProbabilityInterval::Unavailable(ProbabilityIntervalUnavailable::BootstrapUnavailable)
        );
        assert_eq!(
            logit_probability_interval(0.3, Some(0.0), 0.95),
            ProbabilityInterval::Unavailable(ProbabilityIntervalUnavailable::ZeroSpread)
        );
    }

    #[test]
    fn logit_interval_stays_inside_unit_interval_and_matches_delta_method() {
        // Near the boundary a symmetric p ± z·se leaves [0, 1]; the logit
        // interval does not, and it widens toward the interior.
        let (p, se) = (0.01, 0.02);
        let (lo, hi) = logit_probability_interval(p, Some(se), 0.95).bounds().unwrap();
        assert!(0.0 < lo && lo < p && p < hi && hi < 1.0, "[{lo}, {hi}]");
        assert!(hi - p > p - lo, "skewed toward the interior");
        let (lo1, hi1) = logit_probability_interval(1.0 - p, Some(se), 0.95).bounds().unwrap();
        assert!((lo1 - (1.0 - hi)).abs() < 1e-12 && (hi1 - (1.0 - lo)).abs() < 1e-12);
        // Delta-method width on the logit scale.
        let z = antecedent_stats::normal_ppf(0.975);
        let logit = |x: f64| (x / (1.0 - x)).ln();
        let half = z * 0.05 / (0.4 * 0.6);
        let (lo, hi) = logit_probability_interval(0.4, Some(0.05), 0.95).bounds().unwrap();
        assert!((logit(lo) - (logit(0.4) - half)).abs() < 1e-10);
        assert!((logit(hi) - (logit(0.4) + half)).abs() < 1e-10);
        // Huge SEs saturate inside [0, 1] instead of escaping it.
        let (lo, hi) = logit_probability_interval(0.5, Some(1e6), 0.95).bounds().unwrap();
        assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "boundary plug-in probabilities are exactly 0 or 1")]
    fn plug_in_publishes_bounded_atom_intervals_and_refuses_boundary_atoms() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let query = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), f(1.0))],
        );
        let cq = antecedent_core::CausalQuery::Distribution(query.clone());
        let mut ws = IdentificationWorkspace::default();
        let id_res = id.identify(&prep, &cq, &mut ws).unwrap();
        let est =
            FunctionalDistribution { bootstrap_replicates: 40, ..FunctionalDistribution::new() };
        let run = |data: &TabularData| {
            let prepared = est
                .prepare(
                    data,
                    &query,
                    &id_res.estimands[0],
                    &id_res.arena,
                    id_res.required_assumptions.clone(),
                )
                .unwrap();
            let mut ews = FunctionalDistributionWorkspace::default();
            est.estimate(&prepared, &[], &mut ews, &ExecutionContext::for_tests(7)).unwrap()
        };

        let out = run(&binary_confounding_table());
        assert_eq!(out.atom_uncertainty.len(), out.atoms.len());
        for (atom, u) in out.atoms.iter().zip(out.atom_uncertainty.iter()) {
            let (lo, hi) = u.interval.bounds().expect("interior atom");
            assert!(0.0 < lo && lo < atom.probability && atom.probability < hi && hi < 1.0);
            assert_eq!(u.replicates_ok, out.bootstrap_replicates_ok.unwrap());
        }
        // Binary {0, 1} outcome: the Y = 1 atom SE is the mean's SE and its
        // interval is the mean's interval.
        let one = out.atoms.iter().position(|a| a.outcomes[0].1.as_f64() == Some(1.0)).unwrap();
        let mean_se = out.se_bootstrap.unwrap();
        assert!((out.atom_uncertainty[one].se_bootstrap.unwrap() - mean_se).abs() < 1e-12);
        assert_eq!(out.mean_interval, Some(out.atom_uncertainty[one].interval));

        // Force P(y = 1 | do(t = 1)) = 0: every treated row has y = 0.
        let mut boundary = binary_confounding_table();
        let t: Vec<f64> = match boundary.column(VariableId::from_raw(0)).unwrap() {
            ColumnView::Float64(c) => c.values.to_vec(),
            _ => unreachable!(),
        };
        let y: Vec<f64> = match boundary.column(VariableId::from_raw(1)).unwrap() {
            ColumnView::Float64(c) => {
                c.values.iter().zip(&t).map(|(&y, &t)| if t > 0.5 { 0.0 } else { y }).collect()
            }
            _ => unreachable!(),
        };
        boundary = rebuild_with_y(&boundary, y);
        let out = run(&boundary);
        let refused =
            ProbabilityInterval::Unavailable(ProbabilityIntervalUnavailable::BoundaryEstimate);
        assert_eq!(out.atoms.len(), 2);
        for (atom, u) in out.atoms.iter().zip(out.atom_uncertainty.iter()) {
            assert!(atom.probability == 0.0 || atom.probability == 1.0);
            assert_eq!(u.interval, refused, "p̂ ∈ {{0, 1}} has no interval");
        }
        assert_eq!(out.mean_interval, Some(refused));
        // The scalar SE is unchanged in meaning: the bootstrap spread is zero.
        assert_eq!(out.se_bootstrap, Some(0.0));
    }

    /// Copy `t, z` from `data` and replace `y`.
    fn rebuild_with_y(data: &TabularData, y: Vec<f64>) -> TabularData {
        let col = |raw: u32| match data.column(VariableId::from_raw(raw)).unwrap() {
            ColumnView::Float64(c) => c.values.to_vec(),
            _ => unreachable!(),
        };
        let n = y.len();
        let owned = |raw: u32, values: Vec<f64>| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(raw),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        };
        let cols = vec![owned(0, col(0)), owned(1, y), owned(2, col(2))];
        let storage =
            OwnedColumnarStorage::try_new(data.schema().clone(), cols, None, None).unwrap();
        TabularData::new(storage)
    }

    #[test]
    fn functional_effect_no_bootstrap_leaves_replicate_counts_none() {
        // Pins today's behavior ahead of converting `boot` (in `FunctionalEffect::estimate`)
        // from a bare `BootstrapSeResult` to `Option<BootstrapSeResult>`: with
        // `bootstrap_replicates == 0` (the default), the returned estimate's
        // bootstrap_replicates_ok/_failed must stay `None`, not `Some(0)`.
        //
        // Builds the functional directly via `CausalExprArena::backdoor_ate` (skipping full
        // DAG identification) so the estimand's `"general.id"` method tag and functional are
        // guaranteed consistent for `FunctionalEffect::prepare`.
        let mut arena = CausalExprArena::new();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let z = VariableId::from_raw(2);
        let functional =
            arena.backdoor_ate(treatment, outcome, &[z], Value::f64(1.0), Value::f64(0.0));
        let estimand = IdentifiedEstimand::backdoor("general.id", Arc::from([z]), functional);

        let data = binary_confounding_table();
        let est = FunctionalEffect::new(); // bootstrap_replicates: 0 by default
        let extra = [treatment, outcome];
        let prepared =
            est.prepare(&data, &estimand, &arena, AssumptionSet::default(), &extra).unwrap();
        let mut ews = FunctionalDistributionWorkspace::default();
        let out = est.estimate(&prepared, &mut ews, &ExecutionContext::for_tests(0)).unwrap();
        assert!(out.bootstrap_replicates_ok.is_none(), "ok={:?}", out.bootstrap_replicates_ok);
        assert!(
            out.bootstrap_replicates_failed.is_none(),
            "failed={:?}",
            out.bootstrap_replicates_failed
        );
        assert!(out.se_bootstrap.is_none());
    }

    #[test]
    fn empty_required_conditioning_cell_is_missing_table_entry() {
        // Front-door needs P(M | T=1). Data never observes T=1, so that cell
        // must stay absent and evaluate as MissingTableEntry — not P=0.
        let mut arena = CausalExprArena::new();
        let t = VariableId::from_raw(0);
        let m = VariableId::from_raw(1);
        let y = VariableId::from_raw(2);
        let functional = arena.frontdoor_ate(t, y, &[m], Value::f64(1.0), Value::f64(0.0));
        let estimand = IdentifiedEstimand::frontdoor("general.id", Arc::from([m]), functional);

        let t_col = vec![0.0; 20];
        let m_col: Vec<f64> = (0..20).map(|i| f64::from(i % 2)).collect();
        let y_col: Vec<f64> = (0..20).map(|i| f64::from((i / 2) % 2)).collect();
        let data = TabularData::from_f64_columns([
            ("t", t_col.as_slice()),
            ("m", m_col.as_slice()),
            ("y", y_col.as_slice()),
        ])
        .unwrap();

        let est = FunctionalEffect::new();
        let prepared =
            est.prepare(&data, &estimand, &arena, AssumptionSet::default(), &[t, y]).unwrap();
        let mut ews = FunctionalDistributionWorkspace::default();
        let err = est.estimate(&prepared, &mut ews, &ExecutionContext::for_tests(0)).unwrap_err();
        assert!(err.to_string().contains("missing probability table entry"), "{err}");
        let eval_err = prepared
            .compiled
            .evaluate(&prepared.arena, &prepared.provider, &EvalContext::default())
            .unwrap_err();
        assert!(functional_cell_unevaluable(&eval_err));
        let support = support_from_functional_eval(Some(&eval_err)).unwrap();
        assert_eq!(support.status, SupportStatus::OutsideEmpiricalSupport);
        assert_eq!(support_from_functional_eval(None).unwrap().status, SupportStatus::Supported);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Exact aliases and structural zeros are the contract.
    fn joint_dirichlet_preserves_probability_identities_and_beta_moments() {
        use antecedent_expr::DistributionProvider;
        let x = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let columns = HashMap::from([
            (x, vec![Some(f(0.0)), Some(f(0.0)), Some(f(1.0)), Some(f(1.0))]),
            (y, vec![Some(f(0.0)), Some(f(0.0)), Some(f(0.0)), Some(f(1.0))]),
        ]);
        let factors: Vec<(Arc<[VariableId]>, Arc<[VariableId]>)> = vec![
            (Arc::from([x]), Arc::from([])),
            (Arc::from([y]), Arc::from([x])),
            (Arc::from([x, y]), Arc::from([])),
            (Arc::from([y]), Arc::from([])),
        ];
        let interventions: Arc<[InterventionAssignment]> =
            Arc::from([InterventionAssignment { variable: x, value: f(1.0) }]);
        let signatures =
            vec![(Arc::from([y]), Arc::from([x]), interventions.clone(), DomainRef::Observational)];
        let mut rng = ExecutionContext::for_tests(42).rng.stream(1);
        let mut draws = Vec::new();
        for _ in 0..4000 {
            let p = provider_from_bayesian_bootstrap(&columns, 4, &factors, &signatures, &mut rng)
                .unwrap();
            let assignment = Assignment::from_pairs([(x, f(1.0)), (y, f(1.0))]);
            let prob = |vars: &[VariableId],
                        cond: &[VariableId],
                        intervention: &[InterventionAssignment]| {
                p.probability(
                    &FactorSpec {
                        variables: vars,
                        conditioned_on: cond,
                        intervention,
                        domain: DomainRef::Observational,
                    },
                    &assignment,
                    &EvalContext::default(),
                )
                .unwrap()
            };
            assert!(
                (prob(&[x, y], &[], &[]) - prob(&[x], &[], &[]) * prob(&[y], &[x], &[])).abs()
                    < 1e-12
            );
            assert_eq!(prob(&[y], &[x], &[]), prob(&[y], &[x], &interventions));
            draws.push(prob(&[y], &[], &[]));
            let zero = Assignment::from_pairs([(x, f(0.0)), (y, f(1.0))]);
            assert_eq!(
                p.probability(
                    &FactorSpec {
                        variables: &[x, y],
                        conditioned_on: &[],
                        intervention: &[],
                        domain: DomainRef::Observational
                    },
                    &zero,
                    &EvalContext::default()
                )
                .unwrap(),
                0.0
            );
        }
        // One success among four rows: Beta(1,3), var = 3/(16*5).
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / draws.len() as f64;
        assert!((mean - 0.25).abs() < 0.012);
        assert!((variance - 0.0375).abs() < 0.004);
    }
    #[test]
    fn functional_factors_share_complete_cases() {
        let x = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let schema = TabularData::from_f64_columns([
            ("x", &[0.0, 1.0, 1.0][..]),
            ("y", &[0.0, 1.0, 0.0][..]),
        ])
        .unwrap()
        .schema()
        .clone();
        let columns = vec![
            OwnedColumn::Float64(
                Float64Column::new(x, Arc::from([0.0, 1.0, 1.0]), ValidityBitmap::all_valid(3))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    y,
                    Arc::from([0.0, 1.0, 0.0]),
                    ValidityBitmap::from_bytes(vec![3_u8], 3).unwrap(),
                )
                .unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap());
        let factors = vec![(Arc::from([x]), Arc::from([])), (Arc::from([y]), Arc::from([x]))];
        let (_, retained) =
            build_empirical_provider(&data, &HashSet::from([x, y]), &factors, &[]).unwrap();
        assert_eq!(retained[&x], vec![Some(f(0.0)), Some(f(1.0))]);
        assert_eq!(retained[&y], vec![Some(f(0.0)), Some(f(1.0))]);
    }

    /// Frozen two-path table (`t, m, y, c`) from
    /// `conformance/estimate/path_specific_edge_gformula`.
    fn edge_gformula_table() -> (TabularData, serde_json::Value) {
        let pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/path_specific_edge_gformula/expected.json"
        ))
        .unwrap();
        let names = ["t", "m", "y", "c"];
        let mut cols: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
        for cell in pin["contingency_table"].as_array().unwrap() {
            let count = cell["count"].as_u64().unwrap() as usize;
            for (i, name) in names.iter().enumerate() {
                cols[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
            }
        }
        let pairs: Vec<(&str, &[f64])> =
            names.iter().zip(cols.iter()).map(|(n, c)| (*n, c.as_slice())).collect();
        (TabularData::from_f64_columns(pairs).unwrap(), pin)
    }

    fn edge_gformula_identification() -> antecedent_identify::IdentificationResult {
        // c -> t, c -> m, c -> y, t -> m, t -> y, m -> y; selected path through m.
        let mut dag = Dag::with_variables(4);
        for (a, b) in [(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)] {
            dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let id = antecedent_identify::PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let query = antecedent_core::PathSpecificEffectQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        )
        .with_path_nodes([VariableId::from_raw(1)]);
        let res = id
            .identify(
                &prep,
                &antecedent_core::CausalQuery::PathSpecific(query),
                &mut IdentificationWorkspace::default(),
            )
            .unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        res
    }

    /// The path-specific functional with a complementary direct path is the edge
    /// g-formula (0.12 on the frozen law), not the total effect (0.356).
    #[test]
    fn path_specific_edge_g_formula_plug_in_matches_reference() {
        let (data, pin) = edge_gformula_table();
        let res = edge_gformula_identification();
        let est = FunctionalEffect::new().with_bootstrap_replicates(20);
        let prepared = est
            .prepare(
                &data,
                &res.estimands[0],
                &res.arena,
                res.required_assumptions.clone(),
                &[VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)],
            )
            .unwrap();
        let out = est
            .estimate(
                &prepared,
                &mut FunctionalDistributionWorkspace::default(),
                &ExecutionContext::for_tests(3),
            )
            .unwrap();
        let truth = pin["truth"]["path_specific_effect"].as_f64().unwrap();
        assert!((out.ate - truth).abs() < 1e-9, "ate={} truth={truth}", out.ate);
        assert!(out.se_bootstrap.is_some_and(|s| s.is_finite() && s > 0.0));
    }

    /// Each Bayesian draw evaluates the edge g-formula on one reweighted row law:
    /// recompute it by hand from the same weights.
    #[test]
    #[allow(clippy::float_cmp, clippy::many_single_char_names)] // exact binary levels
    fn path_specific_edge_g_formula_uses_one_row_law_per_draw() {
        let (data, _) = edge_gformula_table();
        let res = edge_gformula_identification();
        let est = FunctionalEffect::new();
        let prepared = est
            .prepare(
                &data,
                &res.estimands[0],
                &res.arena,
                res.required_assumptions.clone(),
                &[VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)],
            )
            .unwrap();
        let n = data.row_count();
        let weights: Vec<f64> = (0..n).map(|i| 0.25 + ((i * 7919) % 13) as f64).collect();
        let provider = provider_from_columns(
            &prepared.bootstrap_columns,
            n,
            &prepared.bootstrap_factors,
            &prepared.bootstrap_signatures,
            Some(&weights),
        )
        .unwrap();
        let value = prepared
            .compiled
            .evaluate(&prepared.arena, &provider, &EvalContext::default())
            .unwrap();

        let col = |j: u32| -> Vec<f64> {
            prepared.bootstrap_columns[&VariableId::from_raw(j)]
                .iter()
                .map(|v| v.as_ref().unwrap().as_f64().unwrap())
                .collect()
        };
        let (t, m, y, c) = (col(0), col(1), col(2), col(3));
        let wsum = |pred: &dyn Fn(usize) -> bool, val: &dyn Fn(usize) -> f64| -> f64 {
            (0..n).filter(|&i| pred(i)).map(|i| weights[i] * val(i)).sum()
        };
        let g = |t_m: f64, t_y: f64| -> f64 {
            let mut total = 0.0;
            for cv in [0.0, 1.0] {
                let pc = wsum(&|i| c[i] == cv, &|_| 1.0) / wsum(&|_| true, &|_| 1.0);
                for mv in [0.0, 1.0] {
                    let pm = wsum(&|i| t[i] == t_m && c[i] == cv && m[i] == mv, &|_| 1.0)
                        / wsum(&|i| t[i] == t_m && c[i] == cv, &|_| 1.0);
                    let ey = wsum(&|i| t[i] == t_y && m[i] == mv && c[i] == cv, &|i| y[i])
                        / wsum(&|i| t[i] == t_y && m[i] == mv && c[i] == cv, &|_| 1.0);
                    total += pc * pm * ey;
                }
            }
            total
        };
        let by_hand = g(1.0, 0.0) - g(0.0, 0.0);
        assert!((value - by_hand).abs() < 1e-12, "draw {value} vs by-hand {by_hand}");
    }
}
