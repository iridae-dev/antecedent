//! Family-specific ODE, OU and mechanism adapters for dynamic inference.

use super::{
    AdmissionLease, BTreeMap, CausalRng, Clamp, ClampLaw, CyclicBootstrapDesign,
    DynamicInferenceError, DynamicQuantity, EquilibriumError, ExecutionContext, Fallible, Family,
    ODE_STREAM, OU_MECHANISM_STREAM, OU_STREAM, OdeFit, OdeInferenceProblem, OdeObservation,
    OdeParameter, OdeSystem, OrnsteinUhlenbeck, OuEquationReplacement, OuFit, OuInferenceProblem,
    OuMechanismFit, OuMechanismInferenceProblem, OuMechanismRegime, OuMechanisms, OuParameter,
    RegimeRow, Summary, TimedClamp, admit, cholesky, fit_ode_likelihood, fit_ou_likelihood,
    fit_ou_mechanisms, invalid, mismatch, model_invalid, standard_normal,
};

fn family_mismatch() -> EquilibriumError {
    model_invalid("quantity belongs to another dynamic family")
}

fn clamps_valid(clamps: &[Clamp], n: usize) -> bool {
    clamps.iter().enumerate().all(|(k, c)| {
        c.variable < n
            && c.value.is_finite()
            && clamps[..k].iter().all(|other| other.variable != c.variable)
    })
}

// ---------------------------------------------------------------- ODE family

fn ode_values(model: &OdeSystem, parameters: &[OdeParameter]) -> Fallible<Vec<f64>> {
    if parameters.is_empty() {
        return Err(model_invalid("ODE parameter count"));
    }
    parameters
        .iter()
        .enumerate()
        .map(|(k, parameter)| {
            if parameters[..k].contains(parameter) {
                return Err(model_invalid("duplicate ODE parameter"));
            }
            let value = match *parameter {
                OdeParameter::ExpressionParameter { equation, parameter } => {
                    model.equations.get(equation).and_then(|e| e.parameters.get(parameter)).copied()
                }
                OdeParameter::Intercept { equation } => {
                    model.equations.get(equation).map(|e| e.intercept)
                }
                OdeParameter::Weight { equation, parent } => model
                    .equations
                    .get(equation)
                    .filter(|e| e.expression.is_none())
                    .and_then(|e| e.weights.get(parent))
                    .copied(),
                OdeParameter::Decay { equation } => model.decay.get(equation).copied(),
            };
            value.ok_or_else(|| model_invalid("ODE parameter coordinate"))
        })
        .collect()
}

fn schedule_key(schedule: &[TimedClamp], design: CyclicBootstrapDesign) -> Vec<u64> {
    match design {
        CyclicBootstrapDesign::FixedAssignments => {
            let mut key = Vec::new();
            for action in schedule {
                let mut clamps: Vec<_> =
                    action.clamps.iter().map(|c| (c.variable as u64, c.value.to_bits())).collect();
                clamps.sort_unstable();
                key.push(action.time.to_bits());
                key.push(clamps.len() as u64);
                key.extend(clamps.into_iter().flat_map(|(variable, value)| [variable, value]));
            }
            key
        }
        CyclicBootstrapDesign::RandomizedWithinTargets => {
            let mut targets: Vec<u64> = schedule
                .iter()
                .flat_map(|action| action.clamps.iter().map(|c| c.variable as u64))
                .collect();
            targets.sort_unstable();
            targets.dedup();
            targets
        }
    }
}

pub(super) struct OdeFamily<'a> {
    problem: &'a OdeInferenceProblem,
    estimate: Vec<f64>,
    converged: bool,
    reported: f64,
}

impl<'a> OdeFamily<'a> {
    pub(super) fn new(
        problem: &'a OdeInferenceProblem,
        fit: &OdeFit,
    ) -> Result<Self, DynamicInferenceError> {
        problem.template.validate()?;
        if problem.data.is_empty() {
            return Err(invalid("ODE problem has no trajectories"));
        }
        ode_values(&problem.template, &problem.parameters)?;
        let estimate = ode_values(&fit.model, &problem.parameters)?;
        if crate::cyclic_dynamics::bind(&problem.template, &problem.parameters, &estimate)
            != fit.model
        {
            return Err(mismatch("fitted ODE differs from the template outside its parameters"));
        }
        Ok(Self { problem, estimate, converged: fit.converged, reported: fit.log_likelihood })
    }

    fn model(&self, theta: &[f64]) -> OdeSystem {
        crate::cyclic_dynamics::bind(&self.problem.template, &self.problem.parameters, theta)
    }

    fn path(
        &self,
        model: &OdeSystem,
        initial: &[f64],
        times: &[f64],
        schedule: &[TimedClamp],
        ctx: &ExecutionContext,
    ) -> Fallible<Vec<Vec<f64>>> {
        // The likelihood integrates the sensitivity system; reuse that exact path.
        Ok(model
            .sensitivities(
                initial,
                times,
                schedule,
                &self.problem.parameters,
                self.problem.integration,
                ctx,
            )?
            .trajectory
            .states)
    }
}

impl Family for OdeFamily<'_> {
    type Data = Vec<OdeObservation>;

    fn stream(&self) -> u64 {
        ODE_STREAM
    }
    fn estimate(&self) -> &[f64] {
        &self.estimate
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn reported_log_likelihood(&self) -> f64 {
        self.reported
    }
    fn log_likelihood(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<f64> {
        let p = self.problem;
        crate::cyclic_dynamics::likelihood_score(
            &self.model(theta),
            &p.parameters,
            &p.data,
            p.integration,
            ctx,
        )
        .map(|(likelihood, _)| likelihood)
    }
    fn has_score(&self) -> bool {
        true
    }
    fn score(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<Vec<f64>> {
        let p = self.problem;
        crate::cyclic_dynamics::likelihood_score(
            &self.model(theta),
            &p.parameters,
            &p.data,
            p.integration,
            ctx,
        )
        .map(|(_, score)| score)
    }
    fn units(&self) -> usize {
        self.problem.data.len()
    }
    fn strata(&self, design: CyclicBootstrapDesign) -> Vec<Vec<usize>> {
        let mut groups: BTreeMap<Vec<u64>, Vec<usize>> = BTreeMap::new();
        for (index, observed) in self.problem.data.iter().enumerate() {
            groups.entry(schedule_key(&observed.schedule, design)).or_default().push(index);
        }
        groups.into_values().collect()
    }
    fn gather(&self, picks: &[usize]) -> Self::Data {
        picks.iter().map(|&index| self.problem.data[index].clone()).collect()
    }
    fn simulate(&self, rng: &mut CausalRng, ctx: &ExecutionContext) -> Fallible<Self::Data> {
        let model = self.model(&self.estimate);
        self.problem
            .data
            .iter()
            .map(|observed| {
                let states =
                    self.path(&model, &observed.initial, &observed.times, &observed.schedule, ctx)?;
                let mut values = Vec::with_capacity(states.len());
                // The first state is exact by design and is not a measurement.
                values.push(observed.initial.clone());
                for state in states.iter().skip(1) {
                    values.push(
                        state
                            .iter()
                            .zip(&observed.variances)
                            .map(|(x, variance)| x + variance.sqrt() * standard_normal(rng))
                            .collect(),
                    );
                }
                Ok(OdeObservation {
                    initial: observed.initial.clone(),
                    times: observed.times.clone(),
                    values,
                    variances: observed.variances.clone(),
                    schedule: observed.schedule.clone(),
                })
            })
            .collect()
    }
    fn refit(&self, data: &Self::Data, ctx: &ExecutionContext) -> Fallible<(Vec<f64>, bool)> {
        let p = self.problem;
        let fit =
            fit_ode_likelihood(&p.template, &p.parameters, data, p.integration, p.policy, ctx)?;
        Ok((ode_values(&fit.model, &p.parameters)?, fit.converged))
    }
    fn check_quantity(&self, quantity: &DynamicQuantity) -> bool {
        let n = self.problem.template.equations.len();
        match quantity {
            DynamicQuantity::OdeState { initial, start, time, schedule, variable } => {
                initial.len() == n
                    && initial.iter().all(|v| v.is_finite())
                    && start.is_finite()
                    && time.is_finite()
                    && time > start
                    && *variable < n
                    && schedule
                        .iter()
                        .all(|action| action.time.is_finite() && clamps_valid(&action.clamps, n))
            }
            _ => false,
        }
    }
    fn quantity(
        &self,
        theta: &[f64],
        quantity: &DynamicQuantity,
        ctx: &ExecutionContext,
    ) -> Fallible<f64> {
        let DynamicQuantity::OdeState { initial, start, time, schedule, variable } = quantity
        else {
            return Err(family_mismatch());
        };
        let states = self.path(&self.model(theta), initial, &[*start, *time], schedule, ctx)?;
        states
            .get(1)
            .and_then(|state| state.get(*variable))
            .copied()
            .ok_or_else(|| model_invalid("ODE state quantity"))
    }
    fn data_bytes(&self) -> usize {
        self.problem
            .data
            .iter()
            .map(|o| {
                let cells = o.initial.len()
                    + o.times.len()
                    + o.variances.len()
                    + o.values.iter().map(Vec::len).sum::<usize>()
                    + o.schedule.iter().map(|a| 1 + 2 * a.clamps.len()).sum::<usize>();
                cells * 8 + 160
            })
            .sum()
    }
}

// ----------------------------------------------------------------- OU family

fn ou_values(model: &OrnsteinUhlenbeck, parameters: &[OuParameter]) -> Fallible<Vec<f64>> {
    let n = model.center.len();
    if parameters.is_empty() || model.drift.len() != n * n {
        return Err(model_invalid("OU parameter count or shape"));
    }
    parameters
        .iter()
        .enumerate()
        .map(|(k, parameter)| {
            if parameters[..k].contains(parameter) {
                return Err(model_invalid("duplicate OU parameter"));
            }
            match *parameter {
                OuParameter::Drift { row, column } if row < n && column < n => {
                    Ok(model.drift[row * n + column])
                }
                OuParameter::Center { variable } if variable < n => Ok(model.center[variable]),
                _ => Err(model_invalid("OU parameter coordinate")),
            }
        })
        .collect()
}

fn bind_ou(
    template: &OrnsteinUhlenbeck,
    parameters: &[OuParameter],
    values: &[f64],
) -> OrnsteinUhlenbeck {
    let n = template.center.len();
    let mut model = template.clone();
    for (&parameter, &value) in parameters.iter().zip(values) {
        match parameter {
            OuParameter::Drift { row, column } => model.drift[row * n + column] = value,
            OuParameter::Center { variable } => model.center[variable] = value,
        }
    }
    model
}

pub(super) struct OuFamily<'a> {
    problem: &'a OuInferenceProblem,
    estimate: Vec<f64>,
    converged: bool,
    reported: f64,
}

impl<'a> OuFamily<'a> {
    pub(super) fn new(
        problem: &'a OuInferenceProblem,
        fit: &OuFit,
    ) -> Result<Self, DynamicInferenceError> {
        if problem.rows.is_empty() {
            return Err(invalid("OU problem has no stationary replicates"));
        }
        ou_values(&problem.template, &problem.parameters)?;
        let estimate = ou_values(&fit.model, &problem.parameters)?;
        if bind_ou(&problem.template, &problem.parameters, &estimate) != fit.model {
            return Err(mismatch("fitted OU differs from the template outside its parameters"));
        }
        Ok(Self { problem, estimate, converged: fit.converged, reported: fit.log_likelihood })
    }

    fn model(&self, theta: &[f64]) -> OrnsteinUhlenbeck {
        bind_ou(&self.problem.template, &self.problem.parameters, theta)
    }
}

impl Family for OuFamily<'_> {
    type Data = Vec<RegimeRow>;

    fn stream(&self) -> u64 {
        OU_STREAM
    }
    fn estimate(&self) -> &[f64] {
        &self.estimate
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn reported_log_likelihood(&self) -> f64 {
        self.reported
    }
    fn log_likelihood(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<f64> {
        self.model(theta).stationary_log_likelihood(&self.problem.rows, ctx)
    }
    fn has_score(&self) -> bool {
        false
    }
    fn score(&self, _theta: &[f64], _ctx: &ExecutionContext) -> Fallible<Vec<f64>> {
        Err(model_invalid("stationary OU fits carry no analytic score"))
    }
    fn units(&self) -> usize {
        self.problem.rows.len()
    }
    fn strata(&self, design: CyclicBootstrapDesign) -> Vec<Vec<usize>> {
        crate::cyclic_likelihood::bootstrap_groups(&self.problem.rows, design)
            .into_values()
            .collect()
    }
    fn gather(&self, picks: &[usize]) -> Self::Data {
        picks.iter().map(|&index| self.problem.rows[index].clone()).collect()
    }
    fn simulate(&self, rng: &mut CausalRng, ctx: &ExecutionContext) -> Fallible<Self::Data> {
        let model = self.model(&self.estimate);
        let n = model.center.len();
        // Consecutive rows with one clamp assignment share a law and its factor.
        let mut previous: Option<ClampLaw<'_>> = None;
        let mut rows = Vec::with_capacity(self.problem.rows.len());
        for row in &self.problem.rows {
            if previous.as_ref().is_none_or(|(clamps, ..)| *clamps != row.clamps.as_slice()) {
                let law = model.stationary_intervention(&row.clamps, ctx)?;
                let full = &law.covariance;
                let block: Vec<f64> = law
                    .free
                    .iter()
                    .flat_map(|&i| law.free.iter().map(move |&j| full[i * n + j]))
                    .collect();
                let lower = cholesky(&block, law.free.len()).ok_or(EquilibriumError::Singular)?;
                previous = Some((row.clamps.as_slice(), law.mean, law.free, lower));
            }
            let Some((_, mean, free, lower)) = previous.as_ref() else {
                return Err(model_invalid("OU stationary law"));
            };
            let k = free.len();
            let z: Vec<f64> = (0..k).map(|_| standard_normal(rng)).collect();
            let mut values = mean.clone();
            for (a, &i) in free.iter().enumerate() {
                values[i] += (0..=a).map(|b| lower[a * k + b] * z[b]).sum::<f64>();
            }
            rows.push(RegimeRow { values, clamps: row.clamps.clone() });
        }
        Ok(rows)
    }
    fn refit(&self, data: &Self::Data, ctx: &ExecutionContext) -> Fallible<(Vec<f64>, bool)> {
        let p = self.problem;
        let fit = fit_ou_likelihood(&p.template, &p.parameters, data, p.options, ctx)?;
        Ok((ou_values(&fit.model, &p.parameters)?, fit.converged))
    }
    fn check_quantity(&self, quantity: &DynamicQuantity) -> bool {
        let n = self.problem.template.center.len();
        match quantity {
            DynamicQuantity::OuStationaryMean { clamps, variable } => {
                *variable < n && clamps_valid(clamps, n)
            }
            DynamicQuantity::OuStationaryCovariance { clamps, variable, other } => {
                *variable < n && *other < n && clamps_valid(clamps, n)
            }
            _ => false,
        }
    }
    fn quantity(
        &self,
        theta: &[f64],
        quantity: &DynamicQuantity,
        ctx: &ExecutionContext,
    ) -> Fallible<f64> {
        let model = self.model(theta);
        let n = model.center.len();
        let value = match quantity {
            DynamicQuantity::OuStationaryMean { clamps, variable } => {
                model.stationary_intervention(clamps, ctx)?.mean.get(*variable).copied()
            }
            DynamicQuantity::OuStationaryCovariance { clamps, variable, other } => model
                .stationary_intervention(clamps, ctx)?
                .covariance
                .get(variable * n + other)
                .copied(),
            _ => return Err(family_mismatch()),
        };
        value.ok_or_else(|| model_invalid("OU quantity coordinate"))
    }
    fn data_bytes(&self) -> usize {
        self.problem.rows.iter().map(|row| (row.values.len() + 2 * row.clamps.len()) * 8 + 64).sum()
    }
}

// -------------------------------------------------- shared-mechanism OU family

fn replacements_valid(template: &OuMechanisms, replacements: &[OuEquationReplacement]) -> bool {
    let n = template.equations.len();
    replacements.iter().enumerate().all(|(k, r)| {
        template.equations.contains(&r.equation)
            && replacements[..k].iter().all(|other| other.equation != r.equation)
            && r.drift.len() == n
            && r.diffusion_factor.len() == n
            && r.forcing.is_finite()
            && r.drift.iter().chain(&r.diffusion_factor).all(|v| v.is_finite())
    })
}

pub(super) struct OuMechanismFamily<'a> {
    problem: &'a OuMechanismInferenceProblem,
    estimate: Vec<f64>,
    converged: bool,
    reported: f64,
    summaries: Vec<Summary>,
    /// Unit offsets of each regime; the last entry is the unit count.
    offsets: Vec<usize>,
    _summary_lease: AdmissionLease,
}

impl<'a> OuMechanismFamily<'a> {
    pub(super) fn new(
        problem: &'a OuMechanismInferenceProblem,
        fit: &OuMechanismFit,
        ctx: &ExecutionContext,
    ) -> Result<Self, DynamicInferenceError> {
        let n = problem.template.validate()?;
        crate::ou_mechanisms::parameters(&problem.template, &problem.parameters)?;
        let estimate = crate::ou_mechanisms::parameters(&fit.model, &problem.parameters)?;
        if crate::ou_mechanisms::bind(&problem.template, &problem.parameters, &estimate)
            != fit.model
        {
            return Err(mismatch("fitted mechanisms differ from the template outside parameters"));
        }
        let regimes = problem.regimes.len();
        let bytes = regimes
            .checked_mul((n * n + n) * 8 + n * 16 + 96)
            .and_then(|b| b.checked_add((regimes + 1) * 8));
        let summary_lease = admit(ctx, bytes)?;
        let summaries = crate::ou_mechanisms::summaries(&problem.regimes, n, ctx)?;
        let mut offsets = Vec::with_capacity(regimes + 1);
        offsets.push(0);
        for regime in &problem.regimes {
            offsets.push(offsets[offsets.len() - 1] + regime.values.len());
        }
        Ok(Self {
            problem,
            estimate,
            converged: fit.converged,
            reported: fit.log_likelihood,
            summaries,
            offsets,
            _summary_lease: summary_lease,
        })
    }

    fn model(&self, theta: &[f64]) -> OuMechanisms {
        crate::ou_mechanisms::bind(&self.problem.template, &self.problem.parameters, theta)
    }

    fn scores(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<(f64, Vec<f64>)> {
        crate::ou_mechanisms::scores(
            &self.model(theta),
            &self.summaries,
            &self.problem.parameters,
            ctx,
        )
    }
}

impl Family for OuMechanismFamily<'_> {
    type Data = Vec<OuMechanismRegime>;

    fn stream(&self) -> u64 {
        OU_MECHANISM_STREAM
    }
    fn estimate(&self) -> &[f64] {
        &self.estimate
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn reported_log_likelihood(&self) -> f64 {
        self.reported
    }
    fn log_likelihood(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<f64> {
        self.scores(theta, ctx).map(|(likelihood, _)| likelihood)
    }
    fn has_score(&self) -> bool {
        true
    }
    fn score(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<Vec<f64>> {
        self.scores(theta, ctx).map(|(_, score)| score)
    }
    fn units(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }
    fn strata(&self, design: CyclicBootstrapDesign) -> Vec<Vec<usize>> {
        let units = |r: usize| (self.offsets[r]..self.offsets[r + 1]).collect::<Vec<_>>();
        match design {
            CyclicBootstrapDesign::FixedAssignments => {
                (0..self.problem.regimes.len()).map(units).collect()
            }
            CyclicBootstrapDesign::RandomizedWithinTargets => {
                let mut groups: BTreeMap<Vec<&str>, Vec<usize>> = BTreeMap::new();
                for (r, regime) in self.problem.regimes.iter().enumerate() {
                    let mut key: Vec<&str> =
                        regime.replacements.iter().map(|x| x.equation.as_str()).collect();
                    key.sort_unstable();
                    groups.entry(key).or_default().extend(units(r));
                }
                groups.into_values().collect()
            }
        }
    }
    fn gather(&self, picks: &[usize]) -> Self::Data {
        let regimes = &self.problem.regimes;
        let mut values: Vec<Vec<Vec<f64>>> = vec![Vec::new(); regimes.len()];
        for &unit in picks {
            // A drawn unit keeps its original regime, including under pooled strata.
            let r = self.offsets.partition_point(|&offset| offset <= unit) - 1;
            values[r].push(regimes[r].values[unit - self.offsets[r]].clone());
        }
        regimes
            .iter()
            .zip(values)
            .filter(|(_, values)| !values.is_empty())
            .map(|(regime, values)| OuMechanismRegime {
                id: regime.id.clone(),
                replacements: regime.replacements.clone(),
                values,
            })
            .collect()
    }
    fn simulate(&self, rng: &mut CausalRng, ctx: &ExecutionContext) -> Fallible<Self::Data> {
        let model = self.model(&self.estimate);
        let n = self.problem.template.equations.len();
        self.problem
            .regimes
            .iter()
            .map(|regime| {
                let law = model.stationary(&regime.replacements, ctx)?;
                let lower = cholesky(&law.covariance, n).ok_or(EquilibriumError::Singular)?;
                let values: Vec<Vec<f64>> = (0..regime.values.len())
                    .map(|_| {
                        let z: Vec<f64> = (0..n).map(|_| standard_normal(rng)).collect();
                        (0..n)
                            .map(|i| {
                                law.mean[i] + (0..=i).map(|j| lower[i * n + j] * z[j]).sum::<f64>()
                            })
                            .collect::<Vec<f64>>()
                    })
                    .collect();
                Ok(OuMechanismRegime {
                    id: regime.id.clone(),
                    replacements: regime.replacements.clone(),
                    values,
                })
            })
            .collect()
    }
    fn refit(&self, data: &Self::Data, ctx: &ExecutionContext) -> Fallible<(Vec<f64>, bool)> {
        let p = self.problem;
        let fit = fit_ou_mechanisms(&p.template, &p.parameters, data, p.policy, ctx)?;
        Ok((crate::ou_mechanisms::parameters(&fit.model, &p.parameters)?, fit.converged))
    }
    fn check_quantity(&self, quantity: &DynamicQuantity) -> bool {
        let template = &self.problem.template;
        let n = template.equations.len();
        match quantity {
            DynamicQuantity::OuMechanismMean { replacements, variable } => {
                *variable < n && replacements_valid(template, replacements)
            }
            DynamicQuantity::OuMechanismCovariance { replacements, variable, other } => {
                *variable < n && *other < n && replacements_valid(template, replacements)
            }
            _ => false,
        }
    }
    fn quantity(
        &self,
        theta: &[f64],
        quantity: &DynamicQuantity,
        ctx: &ExecutionContext,
    ) -> Fallible<f64> {
        let model = self.model(theta);
        let n = self.problem.template.equations.len();
        let value = match quantity {
            DynamicQuantity::OuMechanismMean { replacements, variable } => {
                model.stationary(replacements, ctx)?.mean.get(*variable).copied()
            }
            DynamicQuantity::OuMechanismCovariance { replacements, variable, other } => {
                model.stationary(replacements, ctx)?.covariance.get(variable * n + other).copied()
            }
            _ => return Err(family_mismatch()),
        };
        value.ok_or_else(|| model_invalid("OU mechanism quantity coordinate"))
    }
    fn data_bytes(&self) -> usize {
        let n = self.problem.template.equations.len();
        self.problem
            .regimes
            .iter()
            .map(|r| r.values.len() * (n * 8 + 24) + r.replacements.len() * (2 * n + 1) * 8 + 128)
            .sum()
    }
}

// ------------------------------------------------------------- shared helpers
