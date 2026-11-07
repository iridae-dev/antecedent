//! Fit and refit typed mechanisms to observed columns.

use super::{
    BTreeMap, BTreeSet, FittedMechanism, MAX_EXPRESSION_PARAMETERS, MAX_POLYNOMIAL_DEGREE,
    MAX_TYPED_PARENTS, MeanSpec, MechanismContext, MechanismProvenance, MechanismSpec,
    MechanismTape, ResponseFamily, TypedFitError, invert_square, logistic, polynomial_width,
    scale_dim, softplus,
};

#[derive(Clone, Copy)]
enum Link {
    Gaussian,
    Bernoulli,
    Poisson,
    Gamma,
}

impl Link {
    fn mean(self, eta: f64) -> f64 {
        match self {
            Self::Gaussian => eta,
            Self::Bernoulli => logistic(eta),
            Self::Poisson | Self::Gamma => eta.exp(),
        }
    }

    /// Unit-dispersion log-likelihood up to a constant.
    fn objective(self, y: f64, eta: f64) -> f64 {
        match self {
            Self::Gaussian => -0.5 * (y - eta) * (y - eta),
            Self::Bernoulli => y * eta - softplus(eta),
            Self::Poisson => y * eta - eta.exp(),
            Self::Gamma => -y * (-eta).exp() - eta,
        }
    }

    /// `(score weight, Fisher weight)` per observation.
    fn score_and_weight(self, y: f64, eta: f64) -> (f64, f64) {
        let mu = self.mean(eta);
        match self {
            Self::Gaussian => (y - mu, 1.0),
            Self::Bernoulli => (y - mu, mu * (1.0 - mu)),
            Self::Poisson => (y - mu, mu),
            Self::Gamma => ((y - mu) / mu, 1.0),
        }
    }
}

struct MeanProblem<'a> {
    base: &'a FittedMechanism,
    rows: &'a [Vec<f64>],
    y: &'a [f64],
    link: Link,
    features: Option<Vec<f64>>,
}

impl<'a> MeanProblem<'a> {
    fn new(base: &'a FittedMechanism, rows: &'a [Vec<f64>], y: &'a [f64], link: Link) -> Self {
        let features = match base.spec.mean {
            MeanSpec::Polynomial { .. } => {
                let mut all = Vec::with_capacity(rows.len() * base.mean_dim);
                let mut f = Vec::new();
                for row in rows {
                    let z = base.standardize(row);
                    base.polynomial_features(&z, &mut f);
                    all.extend_from_slice(&f);
                }
                Some(all)
            }
            MeanSpec::Expression { .. } => None,
        };
        Self { base, rows, y, link, features }
    }

    fn eta(&self, theta: &[f64]) -> Vec<f64> {
        let k = self.base.mean_dim;
        match &self.features {
            Some(x) => (0..self.rows.len())
                .map(|i| x[i * k..(i + 1) * k].iter().zip(theta).map(|(a, b)| a * b).sum())
                .collect(),
            None => self.rows.iter().map(|r| self.base.eta_with(theta, r)).collect(),
        }
    }

    fn jacobian(&self, theta: &[f64]) -> Vec<f64> {
        let k = self.base.mean_dim;
        if let Some(x) = &self.features {
            return x.clone();
        }
        let n = self.rows.len();
        let mut jac = vec![0.0; n * k];
        let mut shifted = theta.to_vec();
        for j in 0..k {
            let h = 1e-6 * (1.0 + theta[j].abs());
            shifted[j] = theta[j] + h;
            let up = self.eta(&shifted);
            shifted[j] = theta[j] - h;
            let down = self.eta(&shifted);
            shifted[j] = theta[j];
            for i in 0..n {
                jac[i * k + j] = (up[i] - down[i]) / (2.0 * h);
            }
        }
        jac
    }

    fn objective(&self, eta: &[f64], obs_weights: Option<&[f64]>) -> f64 {
        let mut total = 0.0;
        for (i, e) in eta.iter().enumerate() {
            let w = obs_weights.map_or(1.0, |w| w[i]);
            total += w * self.link.objective(self.y[i], *e);
        }
        total
    }
}

pub(super) fn matvec(a: &[f64], k: usize, v: &[f64]) -> Vec<f64> {
    (0..k).map(|i| (0..k).map(|j| a[i * k + j] * v[j]).sum()).collect()
}

/// Fisher scoring with Levenberg damping for the mean parameters. Returns the
/// parameters and the Fisher information at them.
fn fit_mean(
    problem: &MeanProblem<'_>,
    start: &[f64],
    obs_weights: Option<&[f64]>,
    variable: &str,
    cancelled: &dyn Fn() -> bool,
) -> Result<(Vec<f64>, Vec<f64>), TypedFitError> {
    let k = problem.base.mean_dim;
    let n = problem.rows.len();
    let max_iter = if problem.features.is_some() { 200 } else { 500 };
    let mut theta = start.to_vec();
    let mut eta = problem.eta(&theta);
    if eta.iter().any(|e| !e.is_finite()) {
        return Err(TypedFitError::NotConverged(variable.to_owned()));
    }
    let mut objective = problem.objective(&eta, obs_weights);
    let mut lambda = 0.0_f64;
    let mut converged = false;
    for _ in 0..max_iter {
        if cancelled() {
            return Err(TypedFitError::Cancelled);
        }
        let jac = problem.jacobian(&theta);
        let mut score = vec![0.0; k];
        let mut info = vec![0.0; k * k];
        for i in 0..n {
            let ow = obs_weights.map_or(1.0, |w| w[i]);
            let (s, w) = problem.link.score_and_weight(problem.y[i], eta[i]);
            let row = &jac[i * k..(i + 1) * k];
            for a in 0..k {
                score[a] += ow * s * row[a];
                let wa = ow * w * row[a];
                for b in 0..=a {
                    info[a * k + b] += wa * row[b];
                }
            }
        }
        for a in 0..k {
            for b in 0..a {
                info[b * k + a] = info[a * k + b];
            }
        }
        let mut accepted = false;
        for _ in 0..40 {
            let mut damped = info.clone();
            for a in 0..k {
                damped[a * k + a] += lambda * damped[a * k + a].abs().max(1e-12);
            }
            let Some(inverse) = invert_square(&damped, k) else {
                lambda = (lambda * 10.0).max(1e-6);
                continue;
            };
            let delta = matvec(&inverse, k, &score);
            let candidate: Vec<f64> = theta.iter().zip(&delta).map(|(t, d)| t + d).collect();
            let candidate_eta = problem.eta(&candidate);
            let candidate_objective = problem.objective(&candidate_eta, obs_weights);
            if candidate_objective.is_finite()
                && candidate_objective >= objective - 1e-12 * objective.abs().max(1.0)
            {
                let step = delta
                    .iter()
                    .zip(&theta)
                    .map(|(d, t)| d.abs() / (1.0 + t.abs()))
                    .fold(0.0_f64, f64::max);
                theta = candidate;
                eta = candidate_eta;
                objective = candidate_objective;
                lambda = if lambda < 1e-12 { 0.0 } else { lambda / 10.0 };
                accepted = true;
                converged = step < 1e-10;
                break;
            }
            lambda = (lambda * 10.0).max(1e-6);
        }
        if !accepted {
            // Exhausted damping attempts do not establish stationarity or convergence.
            break;
        }
        if converged {
            break;
        }
    }
    if !converged {
        return Err(TypedFitError::NotConverged(variable.to_owned()));
    }
    // A linear predictor this large is a divergent (separated) fit, not an estimate.
    let bound = if matches!(problem.link, Link::Bernoulli) { 25.0 } else { 40.0 };
    if eta.iter().any(|e| !e.is_finite() || e.abs() > bound)
        && !matches!(problem.link, Link::Gaussian)
    {
        return Err(TypedFitError::NotConverged(variable.to_owned()));
    }
    // Information at the final parameters.
    let jac = problem.jacobian(&theta);
    let mut info = vec![0.0; k * k];
    for i in 0..n {
        let ow = obs_weights.map_or(1.0, |w| w[i]);
        let (_, w) = problem.link.score_and_weight(problem.y[i], eta[i]);
        let row = &jac[i * k..(i + 1) * k];
        for a in 0..k {
            for b in 0..k {
                info[a * k + b] += ow * w * row[a] * row[b];
            }
        }
    }
    Ok((theta, info))
}

fn validate_spec(spec: &MechanismSpec) -> Result<Option<MechanismTape>, TypedFitError> {
    if spec.variable.trim().is_empty() {
        return Err(TypedFitError::Spec("blank variable".into()));
    }
    let distinct: BTreeSet<&String> = spec.parents.iter().collect();
    if distinct.len() != spec.parents.len()
        || spec.parents.iter().any(|p| p.trim().is_empty() || *p == spec.variable)
    {
        return Err(TypedFitError::Spec(
            "parents must be distinct, non-blank and not the variable itself".into(),
        ));
    }
    if spec.parents.len() > MAX_TYPED_PARENTS {
        return Err(TypedFitError::Spec(format!("more than {MAX_TYPED_PARENTS} parents")));
    }
    match &spec.mean {
        MeanSpec::Polynomial { degree, .. } => {
            if *degree == 0 || *degree > MAX_POLYNOMIAL_DEGREE {
                return Err(TypedFitError::Spec(format!(
                    "polynomial degree outside 1..={MAX_POLYNOMIAL_DEGREE}"
                )));
            }
            Ok(None)
        }
        MeanSpec::Expression { expression, start } => {
            if spec.parents.is_empty() {
                return Err(TypedFitError::Spec(
                    "an expression mean needs at least one parent".into(),
                ));
            }
            if start.is_empty()
                || start.len() > MAX_EXPRESSION_PARAMETERS
                || start.iter().any(|v| !v.is_finite())
            {
                return Err(TypedFitError::Spec(
                    "expression start must be finite with 1..=64 parameters".into(),
                ));
            }
            expression
                .compile(spec.parents.len(), start.len())
                .map(Some)
                .map_err(|e| TypedFitError::Spec(e.to_string()))
        }
    }
}

fn column<'a>(
    columns: &'a BTreeMap<String, Vec<f64>>,
    name: &str,
) -> Result<&'a Vec<f64>, TypedFitError> {
    columns.get(name).ok_or_else(|| TypedFitError::MissingColumn(name.to_owned()))
}

#[allow(
    clippy::float_cmp,
    reason = "exact equality is required for discrete support or floating point progress"
)]
fn check_support(spec: &MechanismSpec, y: &[f64]) -> Result<(), TypedFitError> {
    let bad = |reason: &str| TypedFitError::Support {
        variable: spec.variable.clone(),
        family: spec.family.as_str(),
        reason: reason.to_owned(),
    };
    match spec.family {
        ResponseFamily::Gaussian | ResponseFamily::HeteroscedasticGaussian => Ok(()),
        ResponseFamily::LogNormal | ResponseFamily::Gamma => {
            if y.iter().any(|v| *v <= 0.0) {
                Err(bad("values must be strictly positive"))
            } else {
                Ok(())
            }
        }
        ResponseFamily::Bernoulli => {
            if y.iter().any(|v| *v != 0.0 && *v != 1.0) {
                Err(bad("values must be 0 or 1"))
            } else {
                Ok(())
            }
        }
        ResponseFamily::Poisson => {
            if y.iter().any(|v| *v < 0.0 || v.fract() != 0.0) {
                Err(bad("values must be non-negative integers"))
            } else {
                Ok(())
            }
        }
    }
}

/// Fit one mechanism from one population's data.
///
/// `design_from` fixes the parent standardization (so the parameters are
/// comparable to that fit's); the specification must then equal that fit's
/// except for nothing — pass it exactly when the fit is a target refit of that
/// source mechanism.
///
/// # Errors
/// [`TypedFitError`]: a malformed specification or data, a response outside
/// the family's support, a singular design, or an iteration that did not
/// converge.
pub fn fit_mechanism(
    spec: &MechanismSpec,
    columns: &BTreeMap<String, Vec<f64>>,
    context: MechanismContext,
    provenance: MechanismProvenance,
    design_from: Option<&FittedMechanism>,
) -> Result<FittedMechanism, TypedFitError> {
    fit_mechanism_polled(spec, columns, context, provenance, design_from, &|| false)
}

/// [`fit_mechanism`] that polls `cancelled` once per scoring iteration and stops with
/// [`TypedFitError::Cancelled`].
///
/// # Errors
/// As [`fit_mechanism`], and [`TypedFitError::Cancelled`].
#[allow(
    clippy::too_many_lines,
    reason = "the scientific calculation and its checks form one bounded procedure"
)]
pub fn fit_mechanism_polled(
    spec: &MechanismSpec,
    columns: &BTreeMap<String, Vec<f64>>,
    context: MechanismContext,
    provenance: MechanismProvenance,
    design_from: Option<&FittedMechanism>,
    cancelled: &dyn Fn() -> bool,
) -> Result<FittedMechanism, TypedFitError> {
    let tape = validate_spec(spec)?;
    let y_raw = column(columns, &spec.variable)?;
    let n = y_raw.len();
    let parent_columns: Vec<&Vec<f64>> =
        spec.parents.iter().map(|p| column(columns, p)).collect::<Result<_, _>>()?;
    if parent_columns.iter().any(|c| c.len() != n) {
        return Err(TypedFitError::Data("columns differ in length".into()));
    }
    if y_raw.iter().chain(parent_columns.iter().flat_map(|c| c.iter())).any(|v| !v.is_finite()) {
        return Err(TypedFitError::Data("non-finite value".into()));
    }
    check_support(spec, y_raw)?;
    let p = spec.parents.len();
    let mean_dim = match &spec.mean {
        MeanSpec::Polynomial { degree, interactions } => {
            polynomial_width(p, *degree, *interactions)
        }
        MeanSpec::Expression { start, .. } => start.len(),
    };
    let total_dim = mean_dim + scale_dim(spec.family, p);
    if n < total_dim + 2 {
        return Err(TypedFitError::Data(format!(
            "{n} rows cannot identify {total_dim} parameters"
        )));
    }
    let scaling: Vec<(f64, f64)> = if let Some(source) = design_from {
        if source.spec.parents != spec.parents {
            return Err(TypedFitError::Spec(
                "the design source has other parents than the refit".into(),
            ));
        }
        source.scaling.clone()
    } else {
        let mut out = Vec::with_capacity(p);
        for (name, col) in spec.parents.iter().zip(&parent_columns) {
            let mean = col.iter().sum::<f64>() / n as f64;
            let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
            if var <= 0.0 || !var.is_finite() {
                return Err(TypedFitError::Singular(format!("{}:{name}", spec.variable)));
            }
            out.push((mean, var.sqrt()));
        }
        out
    };
    let rows: Vec<Vec<f64>> =
        (0..n).map(|i| parent_columns.iter().map(|c| c[i]).collect()).collect();

    let mut base = FittedMechanism {
        spec: spec.clone(),
        scaling,
        theta: vec![0.0; total_dim],
        mean_dim,
        covariance: vec![0.0; total_dim * total_dim],
        n_obs: n,
        context,
        provenance,
        tape,
    };
    if let MeanSpec::Expression { start, .. } = &spec.mean {
        base.theta[..mean_dim].copy_from_slice(start);
    }

    let nf = n as f64;
    let mut start = base.theta[..mean_dim].to_vec();
    let variable = spec.variable.as_str();
    match spec.family {
        ResponseFamily::Gaussian | ResponseFamily::LogNormal => {
            let y: Vec<f64> = if spec.family == ResponseFamily::LogNormal {
                y_raw.iter().map(|v| v.ln()).collect()
            } else {
                y_raw.clone()
            };
            let problem = MeanProblem::new(&base, &rows, &y, Link::Gaussian);
            let (theta, info) = fit_mean(&problem, &start, None, variable, cancelled)?;
            let eta = problem.eta(&theta);
            let rss: f64 = y.iter().zip(&eta).map(|(a, b)| (a - b) * (a - b)).sum();
            let dof = nf - mean_dim as f64;
            let sigma2 = rss / dof;
            if !(sigma2.is_finite() && sigma2 > 0.0) {
                return Err(TypedFitError::Singular(variable.to_owned()));
            }
            let inverse = invert_square(&info, mean_dim)
                .ok_or_else(|| TypedFitError::Singular(variable.to_owned()))?;
            base.theta[..mean_dim].copy_from_slice(&theta);
            base.theta[mean_dim] = 0.5 * sigma2.ln();
            for a in 0..mean_dim {
                for b in 0..mean_dim {
                    base.covariance[a * total_dim + b] = sigma2 * inverse[a * mean_dim + b];
                }
            }
            base.covariance[mean_dim * total_dim + mean_dim] = 1.0 / (2.0 * dof);
        }
        ResponseFamily::HeteroscedasticGaussian => {
            fit_heteroscedastic(&mut base, &rows, y_raw, &start, cancelled)?;
        }
        ResponseFamily::Bernoulli | ResponseFamily::Poisson | ResponseFamily::Gamma => {
            let link = match spec.family {
                ResponseFamily::Bernoulli => Link::Bernoulli,
                ResponseFamily::Poisson => Link::Poisson,
                _ => Link::Gamma,
            };
            if matches!(spec.mean, MeanSpec::Polynomial { .. })
                && matches!(link, Link::Poisson | Link::Gamma)
            {
                start[0] = (y_raw.iter().sum::<f64>() / nf).max(1e-12).ln();
            }
            let problem = MeanProblem::new(&base, &rows, y_raw, link);
            let (theta, info) = fit_mean(&problem, &start, None, variable, cancelled)?;
            let inverse = invert_square(&info, mean_dim)
                .ok_or_else(|| TypedFitError::Singular(variable.to_owned()))?;
            let final_eta = problem.eta(&theta);
            base.theta[..mean_dim].copy_from_slice(&theta);
            if matches!(link, Link::Gamma) {
                let eta = final_eta;
                let r2: Vec<f64> = y_raw
                    .iter()
                    .zip(&eta)
                    .map(|(y, e)| {
                        let mu = e.exp();
                        ((y - mu) / mu) * ((y - mu) / mu)
                    })
                    .collect();
                let dof = nf - mean_dim as f64;
                let phi = r2.iter().sum::<f64>() / dof;
                if !(phi.is_finite() && phi > 0.0) {
                    return Err(TypedFitError::Singular(variable.to_owned()));
                }
                let mean_r2 = r2.iter().sum::<f64>() / nf;
                let var_r2 =
                    r2.iter().map(|v| (v - mean_r2) * (v - mean_r2)).sum::<f64>() / (nf - 1.0);
                base.theta[mean_dim] = phi.ln();
                for a in 0..mean_dim {
                    for b in 0..mean_dim {
                        base.covariance[a * total_dim + b] = phi * inverse[a * mean_dim + b];
                    }
                }
                base.covariance[mean_dim * total_dim + mean_dim] =
                    var_r2 / (nf * mean_r2 * mean_r2);
            } else {
                base.covariance.copy_from_slice(&inverse);
            }
        }
    }
    if base.theta.iter().any(|v| !v.is_finite()) || base.covariance.iter().any(|v| !v.is_finite()) {
        return Err(TypedFitError::NotConverged(variable.to_owned()));
    }
    Ok(base)
}

fn fit_heteroscedastic(
    base: &mut FittedMechanism,
    rows: &[Vec<f64>],
    y: &[f64],
    start: &[f64],
    cancelled: &dyn Fn() -> bool,
) -> Result<(), TypedFitError> {
    let variable = base.spec.variable.clone();
    let n = rows.len();
    let k = base.mean_dim;
    let p = base.spec.parents.len();
    let q = p + 1;
    let total = k + q;
    // Design of the log-variance: intercept and standardized parents.
    let mut z_all = Vec::with_capacity(n * q);
    for row in rows {
        z_all.push(1.0);
        z_all.extend(base.standardize(row));
    }
    let mut ztz = vec![0.0; q * q];
    for i in 0..n {
        for a in 0..q {
            for b in 0..q {
                ztz[a * q + b] += z_all[i * q + a] * z_all[i * q + b];
            }
        }
    }
    let ztz_inverse =
        invert_square(&ztz, q).ok_or_else(|| TypedFitError::Singular(variable.clone()))?;

    let problem_frozen = base.clone();
    let problem = MeanProblem::new(&problem_frozen, rows, y, Link::Gaussian);
    let (mut theta, mut info) = fit_mean(&problem, start, None, &variable, cancelled)?;
    let mut eta = problem.eta(&theta);
    let mean_r2 = y.iter().zip(&eta).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / n as f64;
    let mut gamma = vec![0.0; q];
    gamma[0] = mean_r2.max(1e-300).ln();
    let mut converged = false;
    for _ in 0..500 {
        let variances: Vec<f64> = (0..n)
            .map(|i| (0..q).map(|a| gamma[a] * z_all[i * q + a]).sum::<f64>().exp())
            .collect();
        if variances.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(TypedFitError::NotConverged(variable));
        }
        let weights: Vec<f64> = variances.iter().map(|v| 1.0 / v).collect();
        let (next_theta, next_info) =
            fit_mean(&problem, &theta, Some(&weights), &variable, cancelled)?;
        let theta_step = next_theta
            .iter()
            .zip(&theta)
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0_f64, f64::max);
        theta = next_theta;
        info = next_info;
        eta = problem.eta(&theta);
        let u: Vec<f64> =
            (0..n).map(|i| (y[i] - eta[i]) * (y[i] - eta[i]) / variances[i] - 1.0).collect();
        let mut zu = vec![0.0; q];
        for i in 0..n {
            for a in 0..q {
                zu[a] += z_all[i * q + a] * u[i];
            }
        }
        let mut delta = matvec(&ztz_inverse, q, &zu);
        let largest = delta.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
        if largest > 2.0 {
            for d in &mut delta {
                *d *= 2.0 / largest;
            }
        }
        for (g, d) in gamma.iter_mut().zip(&delta) {
            *g += d;
        }
        if largest < 1e-9 && theta_step < 1e-9 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(TypedFitError::NotConverged(variable));
    }
    let inverse =
        invert_square(&info, k).ok_or_else(|| TypedFitError::Singular(variable.clone()))?;
    base.theta[..k].copy_from_slice(&theta);
    base.theta[k..].copy_from_slice(&gamma);
    for a in 0..k {
        for b in 0..k {
            base.covariance[a * total + b] = inverse[a * k + b];
        }
    }
    for a in 0..q {
        for b in 0..q {
            base.covariance[(k + a) * total + (k + b)] = 2.0 * ztz_inverse[a * q + b];
        }
    }
    Ok(())
}

/// Refit a source mechanism's specification on target data, on the source's
/// design, so the two fits' parameters are comparable.
///
/// # Errors
/// As [`fit_mechanism`].
pub fn refit_in_target(
    source: &FittedMechanism,
    columns: &BTreeMap<String, Vec<f64>>,
    context: MechanismContext,
    provenance: MechanismProvenance,
) -> Result<FittedMechanism, TypedFitError> {
    refit_in_target_polled(source, columns, context, provenance, &|| false)
}

/// [`refit_in_target`] polling `cancelled` as [`fit_mechanism_polled`] does.
///
/// # Errors
/// As [`refit_in_target`], and [`TypedFitError::Cancelled`].
pub fn refit_in_target_polled(
    source: &FittedMechanism,
    columns: &BTreeMap<String, Vec<f64>>,
    context: MechanismContext,
    provenance: MechanismProvenance,
    cancelled: &dyn Fn() -> bool,
) -> Result<FittedMechanism, TypedFitError> {
    fit_mechanism_polled(&source.spec, columns, context, provenance, Some(source), cancelled)
}
