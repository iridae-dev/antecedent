// SPDX-License-Identifier: MIT OR Apache-2.0
//! Continuous identified-functional consumption under an explicit Gaussian
//! observational law. Canonical quadratic factors preserve the existing IR;
//! continuous marginalization is analytic, without finite integration cutoffs.
//! A fitted observational density is a parametric plug-in, not latent recovery
//! or a statistical uncertainty certificate.
use crate::cyclic_functional::{CyclicFunctionalValue, checked_functional_source};
use crate::cyclic_graph::CyclicGraphRequest;
use crate::cyclic_system_artifact::CyclicArtifactError;
use crate::{FunctionalProgramWire, functional_program_from_wire};
use antecedent_core::ExecutionContext;
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode, ProgramLimits};
use serde::{Deserialize, Serialize};
/// Serializable observational law or IID Gaussian density estimation recipe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GaussianCyclicLaw {
    /// A specified observational Gaussian, retaining no estimation uncertainty claim.
    Specified {
        /// Dense observational mean.
        mean: Vec<f64>,
        /// Row-major strictly positive-definite observational covariance.
        covariance: Vec<f64>,
    },
    /// Unregularized maximum-likelihood observational Gaussian from independent rows.
    Fit {
        /// IID complete observational rows; no intervention/time-series substitution.
        rows: Vec<Vec<f64>>,
        /// The sampling contract is an explicit scientific premise.
        independent_observational_replicates: bool,
    },
}
/// Source-bound continuous plug-in evaluation and its observational fit assumptions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianCyclicFunctionalReport {
    /// Independently retained graphical identification source identity.
    pub source_identity: String,
    /// Retained theorem assumptions plus Gaussian provider scope.
    pub derivation: Vec<String>,
    /// Specified or maximum-likelihood observational mean.
    pub mean: Vec<f64>,
    /// Specified or maximum-likelihood observational covariance.
    pub covariance: Vec<f64>,
    /// None for a specified law; otherwise the number of independent fit rows.
    pub fitted_rows: Option<usize>,
    /// Numerical infinity-norm covariance condition estimate, not a proof.
    pub covariance_condition_estimate: f64,
    /// Original assignments and explicit evaluation failures.
    pub values: Vec<CyclicFunctionalValue>,
    /// Actual canonical IR nodes evaluated, with shared subexpressions cached.
    pub operations: usize,
}
fn invalid(s: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(s.to_string())
}
fn inverse(matrix: &[f64], n: usize) -> Result<(Vec<f64>, f64), CyclicArtifactError> {
    if n == 0 {
        return Ok((vec![], 0.));
    }
    let mut lower = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            if !matrix[i * n + j].is_finite()
                || (matrix[i * n + j] - matrix[j * n + i]).abs()
                    > 1e-10 * (1. + matrix[i * n + j].abs().max(matrix[j * n + i].abs()))
            {
                return Err(invalid("Gaussian covariance/precision symmetry"));
            }
            let value = (matrix[i * n + j] + matrix[j * n + i]) * 0.5
                - (0..j).map(|k| lower[i * n + k] * lower[j * n + k]).sum::<f64>();
            if i == j {
                if !value.is_finite() || value <= 0. {
                    return Err(invalid(
                        "Gaussian covariance or integrated precision is not numerically positive definite",
                    ));
                }
                lower[i * n + j] = value.sqrt();
            } else {
                lower[i * n + j] = value / lower[j * n + j];
            }
        }
    }
    let logdet = 2. * (0..n).map(|i| lower[i * n + i].ln()).sum::<f64>();
    let mut result = vec![0.; n * n];
    for column in 0..n {
        let mut x = vec![0.; n];
        x[column] = 1.;
        for i in 0..n {
            x[i] = (x[i] - (0..i).map(|j| lower[i * n + j] * x[j]).sum::<f64>()) / lower[i * n + i];
        }
        for i in (0..n).rev() {
            x[i] = (x[i] - (i + 1..n).map(|j| lower[j * n + i] * x[j]).sum::<f64>())
                / lower[i * n + i];
        }
        for i in 0..n {
            result[i * n + column] = x[i];
        }
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite Gaussian inverse"));
    }
    Ok((result, logdet))
}
#[derive(Clone)]
struct Quadratic {
    precision: Vec<f64>,
    linear: Vec<f64>,
    constant: f64,
    active: Vec<bool>,
}
impl Quadratic {
    fn one(n: usize) -> Self {
        Self {
            precision: vec![0.; n * n],
            linear: vec![0.; n],
            constant: 0.,
            active: vec![false; n],
        }
    }
    fn combine(&mut self, other: &Self, sign: f64) {
        for (a, b) in self.precision.iter_mut().zip(&other.precision) {
            *a += sign * b;
        }
        for (a, b) in self.linear.iter_mut().zip(&other.linear) {
            *a += sign * b;
        }
        self.constant += sign * other.constant;
        for (a, b) in self.active.iter_mut().zip(&other.active) {
            *a |= *b;
        }
    }
    fn marginal_joint(
        mean: &[f64],
        cov: &[f64],
        variables: &[usize],
    ) -> Result<Self, CyclicArtifactError> {
        let n = mean.len();
        let k = variables.len();
        let mut result = Self::one(n);
        let covariance: Vec<_> = variables
            .iter()
            .flat_map(|&i| variables.iter().map(move |&j| cov[i * n + j]))
            .collect();
        let (p, logdet) = inverse(&covariance, k)?;
        result.constant = -0.5 * (k as f64 * (2. * std::f64::consts::PI).ln() + logdet);
        for (a, &i) in variables.iter().enumerate() {
            result.active[i] = true;
            for (b, &j) in variables.iter().enumerate() {
                result.precision[i * n + j] = p[a * k + b];
                result.linear[i] += p[a * k + b] * mean[j];
            }
            result.constant -= 0.5 * mean[i] * result.linear[i];
        }
        Ok(result)
    }
    fn integrate(mut self, variables: &[usize]) -> Result<Self, CyclicArtifactError> {
        let n = self.linear.len();
        let k = variables.len();
        let mut matrix = Vec::with_capacity(k * k);
        for &i in variables {
            for &j in variables {
                matrix.push(self.precision[i * n + j]);
            }
        }
        let (inv, logdet) = inverse(&matrix, k)?;
        let free: Vec<_> = (0..n).filter(|i| !variables.contains(i)).collect();
        let solved: Vec<_> = (0..k)
            .map(|a| (0..k).map(|b| inv[a * k + b] * self.linear[variables[b]]).sum::<f64>())
            .collect();
        self.constant += 0.5 * (0..k).map(|a| self.linear[variables[a]] * solved[a]).sum::<f64>()
            + 0.5 * k as f64 * (2. * std::f64::consts::PI).ln()
            - 0.5 * logdet;
        for &i in &free {
            self.linear[i] -=
                (0..k).map(|a| self.precision[i * n + variables[a]] * solved[a]).sum::<f64>();
        }
        let old = self.precision.clone();
        for &i in &free {
            for &j in &free {
                self.precision[i * n + j] -= (0..k)
                    .map(|a| {
                        (0..k)
                            .map(|b| {
                                old[i * n + variables[a]]
                                    * inv[a * k + b]
                                    * old[variables[b] * n + j]
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>();
            }
        }
        for &i in variables {
            self.active[i] = false;
            self.linear[i] = 0.;
            for j in 0..n {
                self.precision[i * n + j] = 0.;
                self.precision[j * n + i] = 0.;
            }
        }
        Ok(self)
    }
}
struct Compiler<'a> {
    arena: &'a CausalExprArena,
    mean: &'a [f64],
    covariance: &'a [f64],
    cache: Vec<Option<Quadratic>>,
    operations: usize,
    ctx: &'a ExecutionContext,
}
impl Compiler<'_> {
    fn lower(&mut self, id: ExprId, depth: usize) -> Result<Quadratic, CyclicArtifactError> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(invalid("Gaussian functional cancelled"));
        }
        if depth > 64 {
            return Err(invalid("Gaussian functional depth limit"));
        }
        if let Some(q) = &self.cache[id.raw() as usize] {
            return Ok(q.clone());
        }
        self.operations += 1;
        let q = match self.arena.node(id).clone() {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            } => {
                if domain != DomainRef::Observational
                    || !self.arena.intervention_assignments(intervention).is_empty()
                    || !self.arena.population(population).is_empty()
                    || regime.is_some()
                {
                    return Err(invalid(
                        "Gaussian law only supplies observational factors in one study",
                    ));
                }
                let condition: Vec<_> =
                    self.arena.var_set(conditioned_on).iter().map(|v| v.raw() as usize).collect();
                let mut joint: Vec<_> =
                    self.arena.var_set(variables).iter().map(|v| v.raw() as usize).collect();
                joint.extend(&condition);
                joint.sort_unstable();
                joint.dedup();
                let mut q = Quadratic::marginal_joint(self.mean, self.covariance, &joint)?;
                q.combine(&Quadratic::marginal_joint(self.mean, self.covariance, &condition)?, -1.);
                q
            }
            ExprNode::Product(list) => {
                let children = self.arena.list(list).to_vec();
                let mut q = Quadratic::one(self.mean.len());
                for child in children {
                    q.combine(&self.lower(child, depth + 1)?, 1.);
                }
                q
            }
            ExprNode::Ratio { numerator, denominator } => {
                let mut q = self.lower(numerator, depth + 1)?;
                q.combine(&self.lower(denominator, depth + 1)?, -1.);
                q
            }
            ExprNode::IntegralOut { variables, expr } => {
                let variables: Vec<_> =
                    self.arena.var_set(variables).iter().map(|v| v.raw() as usize).collect();
                self.lower(expr, depth + 1)?.integrate(&variables)?
            }
            ExprNode::Kernel { body, population, regime, .. }
                if self.arena.population(population).is_empty() && regime.is_none() =>
            {
                self.lower(body, depth + 1)?
            }
            _ => {
                return Err(invalid(
                    "Gaussian density consumer supports observational density products, ratios and analytic continuous marginalization",
                ));
            }
        };
        if q.precision
            .iter()
            .chain(&q.linear)
            .chain(std::iter::once(&q.constant))
            .any(|v| !v.is_finite())
        {
            return Err(invalid("nonfinite Gaussian canonical factor"));
        }
        self.cache[id.raw() as usize] = Some(q.clone());
        Ok(q)
    }
}
/// Recheck the theorem-derived IR, estimate an optional IID Gaussian density,
/// analytically marginalize its canonical factors, and evaluate original assignments.
/// # Errors
/// Source/IR mismatch, wrong semantics, nonintegrability, malformed law, cancellation or resources.
#[allow(
    clippy::too_many_lines,
    reason = "source checking, explicit density estimation and bounded canonical lowering are one retained scientific consumer"
)]
pub fn evaluate_gaussian_cyclic_functional(
    source: &CyclicGraphRequest,
    source_identity: &str,
    program: &FunctionalProgramWire,
    law: &GaussianCyclicLaw,
    assignments: &[Vec<(u32, f64)>],
    ctx: &ExecutionContext,
) -> Result<GaussianCyclicFunctionalReport, CyclicArtifactError> {
    let (n, continuous, mut derivation) =
        checked_functional_source(source, source_identity, program, ctx)?;
    if (!continuous && !matches!(source, CyclicGraphRequest::Calculus { .. }))
        || n == 0
        || n > 16
        || assignments.is_empty()
        || assignments.len() > 1024
        || assignments.iter().any(|a| a.len() > n)
    {
        return Err(invalid(
            "Gaussian functional requires continuous semantics, at most 16 variables and 1024 assignments",
        ));
    }
    let checked =
        functional_program_from_wire(program, ProgramLimits::default()).map_err(invalid)?;
    if checked.arena().len() > 4096 {
        return Err(invalid("Gaussian canonical node limit"));
    }
    let _lease = ctx
        .acquire_admission(1, (checked.arena().len() * ((n * n + n) * 32 + 256) + 65536) as u64)
        .map_err(invalid)?;
    let (mean, covariance, fitted_rows) = match law {
        GaussianCyclicLaw::Specified { mean, covariance } => {
            if mean.len() != n || covariance.len() != n * n {
                return Err(invalid("Gaussian observational law shape"));
            }
            (mean.clone(), covariance.clone(), None)
        }
        GaussianCyclicLaw::Fit { rows, independent_observational_replicates } => {
            if !independent_observational_replicates || rows.len() <= n || rows.len() > 100_000 {
                return Err(invalid(
                    "Gaussian fit requires n+1..100000 IID observational replicates",
                ));
            }
            let mut mean = vec![0.; n];
            let mut scatter = vec![0.; n * n];
            for (k, row) in rows.iter().enumerate() {
                if ctx.cancellation.is_cancelled() {
                    return Err(invalid("Gaussian fit cancelled"));
                }
                if row.len() != n || row.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("Gaussian fit row shape/nonfinite"));
                }
                let delta: Vec<_> = row.iter().zip(&mean).map(|(x, m)| x - m).collect();
                for i in 0..n {
                    mean[i] += delta[i] / (k + 1) as f64;
                }
                for i in 0..n {
                    for j in 0..n {
                        scatter[i * n + j] += delta[i] * (row[j] - mean[j]);
                    }
                }
            }
            for value in &mut scatter {
                *value /= rows.len() as f64;
            }
            (mean, scatter, Some(rows.len()))
        }
    };
    if mean.iter().chain(&covariance).any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite Gaussian law"));
    }
    let (inv, _) = inverse(&covariance, n)?;
    let norm = |matrix: &[f64]| {
        matrix.chunks(n).map(|r| r.iter().map(|v| v.abs()).sum::<f64>()).fold(0., f64::max)
    };
    let condition = norm(&covariance) * norm(&inv);
    if !condition.is_finite() {
        return Err(invalid("nonfinite covariance condition estimate"));
    }
    let mut compiler = Compiler {
        arena: checked.arena(),
        mean: &mean,
        covariance: &covariance,
        cache: vec![None; checked.arena().len()],
        operations: 0,
        ctx,
    };
    let q = compiler.lower(checked.mapping().executable, 0)?;
    let operations = compiler.operations;
    let mut values = Vec::new();
    for pairs in assignments {
        let mut state = vec![0.; n];
        let mut seen = vec![false; n];
        let mut failure = None;
        for &(v, x) in pairs {
            let i = v as usize;
            if i >= n || !x.is_finite() || seen[i] {
                failure = Some("invalid or duplicate Gaussian assignment".into());
                break;
            }
            state[i] = x;
            seen[i] = true;
        }
        if failure.is_none() && q.active.iter().enumerate().any(|(i, a)| *a && !seen[i]) {
            failure = Some("missing continuous outcome/conditioning assignment".into());
        }
        let mut value = None;
        if failure.is_none() {
            let log = q.constant + (0..n).map(|i| q.linear[i] * state[i]).sum::<f64>()
                - 0.5
                    * (0..n)
                        .map(|i| {
                            (0..n)
                                .map(|j| state[i] * q.precision[i * n + j] * state[j])
                                .sum::<f64>()
                        })
                        .sum::<f64>();
            let density = log.exp();
            if density.is_finite() {
                value = Some(density);
            } else {
                failure = Some("nonfinite Gaussian functional density".into());
            }
        }
        values.push(CyclicFunctionalValue { assignment: pairs.clone(), value, failure });
    }
    derivation.push("Observational Gaussian density provider; optional unregularized IID maximum-likelihood plug-in. Continuous integrals use canonical Gaussian elimination on all real coordinates; no truncation, latent recovery, statistical uncertainty or calibration claim.".into());
    Ok(GaussianCyclicFunctionalReport {
        source_identity: source_identity.into(),
        derivation,
        mean,
        covariance,
        fitted_rows,
        covariance_condition_estimate: condition,
        values,
        operations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_family_artifact::{CyclicFamilyAnswer, CyclicFamilyRequest, consume_cyclic_family},
        cyclic_graph::{CyclicGraphReport, CyclicGraphSpec},
    };
    #[test]
    fn continuous_frontdoor_matches_independent_latent_gaussian_truth_and_replays_fitting() {
        let ctx = ExecutionContext::for_tests(4);
        let source = CyclicGraphRequest::Identify {
            graph: CyclicGraphSpec {
                nodes: 3,
                directed: vec![(0, 1), (1, 2)],
                bidirected: vec![(0, 2)],
            },
            outcomes: vec![2],
            treatments: vec![0],
            continuous: true,
            max_steps: 1000,
            io_scm: true,
            scc_product_density: true,
        };
        let source_report =
            CyclicFamilyRequest::Graph { request: source.clone() }.execute(&ctx).unwrap();
        let CyclicFamilyAnswer::Graph(CyclicGraphReport::Identify {
            program: Some(program), ..
        }) = source_report.answer
        else {
            panic!("frontdoor identification")
        };
        let covariance = vec![2., 1., 1.5, 1., 1.5, 1.55, 1.5, 1.55, 3.01];
        let request = CyclicFamilyRequest::GaussianFunctional {
            source: source.clone(),
            source_identity: source_report.identity.clone(),
            program: program.clone(),
            law: GaussianCyclicLaw::Specified { mean: vec![0.; 3], covariance: covariance.clone() },
            assignments: vec![vec![(0, 1.), (2, 0.)], vec![(0, 1.)]],
        };
        let report = request.execute(&ctx).unwrap();
        let CyclicFamilyAnswer::GaussianFunctional(answer) = report.answer else { panic!() };
        let truth = (-0.5_f64 * 0.4 * 0.4 / 2.13).exp() / (2. * std::f64::consts::PI * 2.13).sqrt();
        assert!((answer.values[0].value.unwrap() - truth).abs() < 1e-12, "{answer:?} {truth}");
        assert!(answer.values[1].value.is_none());
        // Independent symmetric moment fixture: X=U+E, M=.5X+V,
        // Y=.8M+.7U+W. The observational MLE moments are exactly known.
        let mut rows = Vec::new();
        for mask in 0..16 {
            let e: Vec<_> = (0..4).map(|i| if mask & (1 << i) == 0 { -1. } else { 1. }).collect();
            let x = e[0] + e[1];
            let m = 0.5 * x + e[2];
            let y = 0.8 * m + 0.7 * e[0] + e[3];
            rows.push(vec![x, m, y]);
        }
        let fitted = CyclicFamilyRequest::GaussianFunctional {
            source,
            source_identity: source_report.identity,
            program,
            law: GaussianCyclicLaw::Fit { rows, independent_observational_replicates: true },
            assignments: vec![vec![(0, 1.), (2, 0.)]],
        };
        let (report, bytes) = fitted.export(&ctx).unwrap();
        let CyclicFamilyAnswer::GaussianFunctional(answer) = &report.answer else { panic!() };
        assert_eq!(answer.fitted_rows, Some(16));
        for (a, b) in answer.covariance.iter().zip(covariance) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((answer.values[0].value.unwrap() - truth).abs() < 1e-12);
        assert_eq!(consume_cyclic_family(&bytes, &report.identity, &ctx).unwrap(), report);
        let mut altered = fitted.clone();
        if let CyclicFamilyRequest::GaussianFunctional { source_identity, .. } = &mut altered {
            *source_identity = "0".repeat(64);
        }
        assert!(altered.execute(&ctx).is_err());
    }
}
