use super::{
    Arc, Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, CausalRng,
    CompiledDesign, DependenceScope, EffectEstimate, EstimationError, ExecutionContext,
    HistoryScan, IdentificationStatus, Lag, LaggedColumn, LeastSquaresWorkspace, OverlapPolicy,
    ParametricAssumption, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema,
    PreparedBayesianProblem, ROOT_MEAN_STREAM, SequentialBayesianMechanism, SequentialEval,
    SequentialProblem, SequentialSetup, SerialDependence, VariableId, extend_mechanism_design,
};

/// The inputs of the composed Bayesian posterior of one sequential estimate: every stationary
/// mechanism gets its own serial-dependence-tempered posterior, and the target is evaluated
/// on the joint draws.
pub(super) struct BayesianComposition<'a> {
    pub(super) setup: &'a SequentialSetup,
    pub(super) problem: &'a SequentialProblem<'a>,
    pub(super) eval: SequentialEval,
    pub(super) estimator: &'a BayesianGComputationAte,
    pub(super) status: IdentificationStatus,
    pub(super) ctx: &'a ExecutionContext,
    pub(super) gaps: bool,
}

/// The fitted stationary mechanisms: the posterior of each distinct mechanism, the node that
/// owns each node's mechanism, and the fewest draws any of them retained.
struct FittedMechanisms {
    posts: Vec<Option<CausalPosterior>>,
    owner: Vec<usize>,
    count: usize,
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "dense node ids and row indices are u32 by construction, so every index below the graph's node count fits"
)]
impl BayesianComposition<'_> {
    /// Fit the mechanisms, compose the target on their joint draws and publish the estimate.
    pub(super) fn estimate(
        &self,
        assumptions: AssumptionSet,
        validation: Option<&mut Vec<SequentialBayesianMechanism>>,
        ls_ws: &mut LeastSquaresWorkspace,
    ) -> Result<(EffectEstimate, CausalPosterior), EstimationError> {
        let setup = self.setup;
        let ols_base = setup.fit_ols(None, ls_ws, &mut Vec::new(), &mut Vec::new()).ok();
        let contrast_gradient = self.contrast_gradient(ols_base.as_deref())?;
        // A level also reads the sample means of its exogenous roots, held fixed in the
        // g-formula. Their sampling error is a first-order (delta-method) shift of the
        // level: the score of the root means combined by their gradient, whose long-run
        // variance sets the shift's standard deviation.
        // A contrast with effect modifiers reads the sample means of those modifiers the
        // same way, so their score enters by the same shift. At most one of the two
        // scores exists: a contrast cancels every root mean, and a level reads no
        // modifier mean.
        let root_mean_se = ols_base
            .as_ref()
            .and_then(|base| {
                setup.root_mean_scores(base).or_else(|| setup.modifier_mean_scores(base))
            })
            .map_or(0.0, |scores| crate::serial_dependence::mean_standard_error(&scores));
        let fitted = self.fit_mechanisms(contrast_gradient.as_ref(), validation)?;
        let mut posterior = self.compose_posterior(&fitted, root_mean_se)?;
        for post in fitted.posts.iter().flatten().skip(1) {
            posterior.assumptions.entries.extend(post.assumptions.entries.iter().cloned());
            for note in &post.diagnostics.notes {
                if !posterior.diagnostics.notes.contains(note) {
                    posterior.diagnostics.notes.push(Arc::clone(note));
                }
            }
        }
        if matches!(self.eval, SequentialEval::Level) {
            posterior.assumptions.entries.push(level_posterior_assumption());
        } else if root_mean_se > 0.0 {
            posterior.assumptions.entries.push(modifier_mean_posterior_assumption());
        }
        posterior.assumptions.entries.extend(assumptions.entries);
        let effect = EffectEstimate::new(
            posterior.summaries.mean[0],
            posterior.summaries.sd[0],
            posterior.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        Ok((effect, posterior))
    }

    /// Both the Sustained contrast and a Sequence level temper each stationary mechanism
    /// by the long-run-variance ratio of the target's own linear combination of that
    /// mechanism's coefficients: the gradient of the composed value, summed over the
    /// mechanism's unfolded time copies (coefficient positions are stable across
    /// copies). The value is linear in each coefficient, so central differences at the
    /// OLS fit are exact up to rounding. `None` without an OLS fit.
    fn contrast_gradient(
        &self,
        ols_base: Option<&[Vec<f64>]>,
    ) -> Result<Option<std::collections::HashMap<VariableId, Vec<f64>>>, EstimationError> {
        let Some(base) = ols_base else {
            return Ok(None);
        };
        let per_node = self.setup.gradient(base);
        let mut gradients: std::collections::HashMap<VariableId, Vec<f64>> =
            std::collections::HashMap::new();
        for &i in &self.setup.order {
            if base[i].is_empty() {
                continue;
            }
            let variable = self.problem.indexer.key_of(i as u32)?.variable;
            let entry = gradients.entry(variable).or_insert_with(|| vec![0.0; base[i].len()]);
            for k in 0..base[i].len().min(entry.len()) {
                entry[k] += per_node[i][k];
            }
        }
        Ok(Some(gradients))
    }

    /// The lagged parent columns of unfolded node `i`, with the lag of each parent behind it.
    fn parent_columns(&self, i: usize) -> Result<Vec<LaggedColumn>, EstimationError> {
        let indexer = self.problem.indexer;
        let child = indexer.key_of(i as u32)?;
        let mut parent_columns = Vec::with_capacity(self.setup.parents[i].len());
        for &p in &self.setup.parents[i] {
            let parent = indexer.key_of(p as u32)?;
            let lag = child
                .offset
                .checked_sub(parent.offset)
                .ok_or_else(|| EstimationError::unsupported("unfolded parent lag overflowed"))
                .and_then(|delta| {
                    u32::try_from(delta).map_err(|_| {
                        EstimationError::unsupported("causal lag is not a nonnegative offset")
                    })
                })?;
            parent_columns
                .push(LaggedColumn { variable: parent.variable, lag: Lag::from_raw(lag) });
        }
        Ok(parent_columns)
    }

    /// Fit each stationary mechanism once on its unique observed time-series rows. Overlapping
    /// unfolded windows must not multiply the likelihood or create independent copies of beta.
    #[allow(
        clippy::too_many_lines,
        reason = "the bounded estimation procedure keeps its related calculations together"
    )]
    fn fit_mechanisms(
        &self,
        contrast_gradient: Option<&std::collections::HashMap<VariableId, Vec<f64>>>,
        mut validation: Option<&mut Vec<SequentialBayesianMechanism>>,
    ) -> Result<FittedMechanisms, EstimationError> {
        let setup = self.setup;
        let designs = &setup.designs;
        let parents = &setup.parents;
        let data = self.problem.data;
        let mut posts = vec![None; setup.node_count];
        let mut owner = vec![0; setup.node_count];
        let mut mechanisms: Vec<(VariableId, Vec<LaggedColumn>, usize)> = Vec::new();
        let mut count = usize::MAX;
        for &i in &setup.order {
            if designs[i].is_none() {
                continue;
            }
            let child = self.problem.indexer.key_of(i as u32)?;
            let parent_columns = self.parent_columns(i)?;
            if mechanisms.iter().any(|(variable, columns, _)| {
                *variable == child.variable && columns != &parent_columns
            }) {
                return Err(EstimationError::unsupported(
                    "stationary mechanism has inconsistent unfolded parents",
                ));
            }
            if let Some((_, _, owning)) = mechanisms.iter().find(|(variable, columns, owning)| {
                *variable == child.variable
                    && columns == &parent_columns
                    && designs[*owning].as_ref().map(|design| design.ncols)
                        == designs[i].as_ref().map(|design| design.ncols)
            }) {
                owner[i] = *owning;
                continue;
            }
            let mut columns = parent_columns.clone();
            columns.push(LaggedColumn { variable: child.variable, lag: Lag::CONTEMPORANEOUS });
            let use_modifiers = setup.interaction_at[i].iter().any(|slots| !slots.is_empty());
            if use_modifiers {
                for &modifier in &setup.modifiers {
                    columns.push(LaggedColumn { variable: modifier, lag: Lag::CONTEMPORANEOUS });
                }
            }
            let max_lag = columns.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
            let mut scan = HistoryScan::plan(data, max_lag, Arc::from(columns))?;
            let (sample, complete) = scan.windows(data, &self.ctx.kernel_policy, self.gaps)?;
            let segments =
                complete.map(|complete| crate::serial_dependence::window_segments(&complete));
            let covs: Vec<_> = parent_columns
                .iter()
                .enumerate()
                .skip(1)
                .map(|(j, c)| (c.variable, sample.column(j)))
                .collect();
            let design = CompiledDesign::linear_adjustment(
                sample.column(0),
                &covs,
                sample.column(parent_columns.len()),
                &[],
            )?;
            let design = if use_modifiers {
                let parent_values: Vec<&[f64]> =
                    (0..parent_columns.len()).map(|j| sample.column(j)).collect();
                let modifier_values: Vec<&[f64]> = setup
                    .modifiers
                    .iter()
                    .enumerate()
                    .map(|(m, _)| sample.column(parent_columns.len() + 1 + m))
                    .collect();
                let intervened: Vec<bool> =
                    parents[i].iter().map(|&parent| setup.intervention[parent]).collect();
                let modifier_is_parent: Vec<bool> = setup
                    .modifiers
                    .iter()
                    .map(|modifier| {
                        parent_columns.iter().any(|column| column.variable == *modifier)
                    })
                    .collect();
                let (design, slots) = extend_mechanism_design(
                    design,
                    &parent_values,
                    &intervened,
                    &modifier_values,
                    &setup.modifiers,
                    &modifier_is_parent,
                )?;
                if slots != setup.interaction_at[i] {
                    return Err(EstimationError::unsupported(
                        "sustained modifier columns do not match the unfolded design",
                    ));
                }
                design
            } else {
                design
            };
            let width = design.ncols;
            let prep = PreparedBayesianProblem {
                design,
                method: self.problem.estimand.method.clone(),
                adjustment_set: Arc::from([]),
                active: 1.0,
                control: 0.0,
                overlap: OverlapPolicy::ExplicitOverride,
                coef_names: None,
                unit_ids: None,
                // Rows are time-ordered, so the stated model is the serial-dependence-
                // corrected generalized posterior for the Sustained contrast and the
                // Sequence level alike, each tempered along its own gradient.
                serial_dependence: mechanism_serial_dependence(
                    contrast_gradient,
                    child.variable,
                    width,
                    self.eval,
                    segments,
                ),
                preparation: None,
            };
            let mut est = self.estimator.clone();
            est.seed = est.seed.wrapping_add(u64::from(child.variable.raw()));
            let post =
                est.fit(&prep, self.status, &mut BayesianGCompWorkspace::default(), self.ctx)?;
            if let Some(contexts) = validation.as_deref_mut() {
                contexts.push(SequentialBayesianMechanism {
                    variable: child.variable,
                    prepared: prep.clone(),
                    posterior: post.clone(),
                });
            }
            count = count.min(post.draws.n_draws);
            posts[i] = Some(post);
            owner[i] = i;
            mechanisms.push((child.variable, parent_columns, i));
        }
        Ok(FittedMechanisms { posts, owner, count })
    }

    /// The joint posterior of the target: each retained draw evaluates the target at that draw
    /// of every mechanism's coefficients, and a level adds the root-mean shift, a
    /// contrast with effect modifiers the modifier-mean shift.
    fn compose_posterior(
        &self,
        fitted: &FittedMechanisms,
        root_mean_se: f64,
    ) -> Result<CausalPosterior, EstimationError> {
        let setup = self.setup;
        let FittedMechanisms { posts, owner, count } = fitted;
        let count = *count;
        let mut posterior = posts.iter().flatten().next().cloned().ok_or_else(|| {
            EstimationError::unsupported("sustained contrast has no fitted outcome mechanism")
        })?;
        let mut coefficient_columns = vec![Vec::new(); setup.node_count];
        let mut coefficients = vec![Vec::new(); setup.node_count];
        for &i in &setup.order {
            let Some(design) = &setup.designs[i] else {
                continue;
            };
            let post = posts[owner[i]].as_ref().ok_or_else(|| {
                EstimationError::unsupported("stationary mechanism was not fitted")
            })?;
            coefficient_columns[i] = (0..design.ncols)
                .map(|index| {
                    post.draws
                        .schema
                        .quantities
                        .iter()
                        .position(|q| {
                            matches!(
                                q,
                                PosteriorQuantityKind::Coefficient { index: j, .. }
                                    if *j == index
                            )
                        })
                        .ok_or_else(|| EstimationError::stats_msg("missing sequential coefficient"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            coefficients[i].resize(design.ncols, 0.0);
        }
        // The shift is zero-mean by construction, so its draws are centred: the posterior
        // mean stays the mean of the propagated levels exactly, free of the Monte Carlo
        // error of the shift, and only the spread carries the sampling error of the means.
        let shifts = if root_mean_se > 0.0 {
            let mut root_rng = CausalRng::from_seed(self.estimator.seed ^ ROOT_MEAN_STREAM);
            let mut shifts: Vec<f64> =
                (0..count).map(|_| antecedent_kernels::standard_normal(&mut root_rng)).collect();
            #[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^53")]
            let mean = shifts.iter().sum::<f64>() / count.max(1) as f64;
            for shift in &mut shifts {
                *shift = root_mean_se * (*shift - mean);
            }
            shifts
        } else {
            Vec::new()
        };
        let mut values = Vec::with_capacity(count);
        for draw in 0..count {
            for &i in &setup.order {
                if setup.designs[i].is_some() {
                    let post = posts[owner[i]].as_ref().ok_or_else(|| {
                        EstimationError::unsupported("stationary mechanism was not fitted")
                    })?;
                    for (value, &column) in coefficients[i].iter_mut().zip(&coefficient_columns[i])
                    {
                        *value = post.draws.column(column)?[draw];
                    }
                }
            }
            let level = setup.propagate(&coefficients, &setup.factual, None);
            values.push(level + shifts.get(draw).copied().unwrap_or(0.0));
        }
        let draws = PosteriorDraws::from_column_major(
            PosteriorSchema {
                quantities: Arc::from([PosteriorQuantityKind::Effect {
                    name: Arc::from("sustained_window"),
                }]),
            },
            count,
            values,
        )?;
        posterior.summaries = draws.summarize();
        posterior.draws = draws;
        Ok(posterior)
    }
}

/// The serial-dependence correction of one stationary mechanism's likelihood.
///
/// Without an OLS gradient (rank-deficient unfolded design) the sum of slopes stands in for
/// the direction; a level also reads every intercept.
fn mechanism_serial_dependence(
    contrast_gradient: Option<&std::collections::HashMap<VariableId, Vec<f64>>>,
    variable: VariableId,
    width: usize,
    eval: SequentialEval,
    segments: Option<Arc<[u32]>>,
) -> SerialDependence {
    let direction = contrast_gradient
        .and_then(|g| g.get(&variable).cloned())
        .filter(|direction| direction.len() == width)
        .unwrap_or_else(|| {
            let intercept = match eval {
                SequentialEval::Contrast { .. } => 0.0,
                SequentialEval::Level => 1.0,
            };
            (0..width).map(|k| if k == 0 { intercept } else { 1.0 }).collect()
        });
    let scope = DependenceScope::Direction(Arc::from(direction));
    match segments {
        Some(segments) => SerialDependence::SegmentedTempering { scope, segments },
        None => SerialDependence::LongRunTempering(scope),
    }
}

/// The assumption a Sequence level posterior adds over the contrast's.
fn level_posterior_assumption() -> AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: "temporal.sequential.level_posterior".into(),
            description: "each stationary mechanism's likelihood is tempered by the \
                long-run-variance ratio of the level's own gradient (rows are \
                time-ordered), and the sample means of exogenous root nodes the level \
                reads enter as a first-order (delta-method) normal shift whose \
                variance is the long-run variance of their gradient-weighted scores"
                .into(),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: "temporal.sequential.linear_sem".into(),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    }
}

/// The assumption a Sustained contrast with effect modifiers adds over the plain contrast's.
fn modifier_mean_posterior_assumption() -> AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: "temporal.sequential.modifier_mean_posterior".into(),
            description: "the sample means of the effect modifiers the contrast is \
                evaluated at enter as a first-order (delta-method) normal shift whose \
                variance is the long-run variance of their gradient-weighted scores"
                .into(),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: "temporal.sequential.linear_sem".into(),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    }
}
