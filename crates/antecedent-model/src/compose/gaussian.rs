//! Executable joint law of composed linear-Gaussian mechanisms.
//!
//! Every mechanism is `X_v = a_v + sum_p b_vp X_p + sum_l lambda_vl U_l + s_v e_v`
//! with independent standard-normal `e_v` and declared shared latent `U_l`.
//! Under the declared acyclic graph the joint law of all variables is Gaussian
//! and is computed in closed form by path algebra (Bollen 1989, ch. 4): each
//! variable carries its coefficient vector over the independent noise sources
//! (a forward substitution along a topological order), so means and
//! covariances are exact up to floating-point rounding. A hard intervention
//! `do(X_v = x)` removes the mechanism of `X_v` (Pearl 2009, ch. 3); the same
//! recursion then gives the interventional law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;
use antecedent_kernels::standard_normal;

use super::coherence::{CoherenceFinding, CoherenceKind, check_composition};
use super::descriptor::{
    ComposedModelSpec, Domain, MechanismFamily, MechanismForm, ParameterId, ParameterTerm,
    UncertaintyMeaning,
};
use super::uncertainty::assemble_parameter_covariance;
use super::{
    COMPOSITION_STREAM_DOMAIN, CompositionError, MAX_SAMPLE_CELLS, incoherent, noise_stream_index,
};

/// Joint Gaussian law of all variables under one regime.
#[derive(Clone, Debug, PartialEq)]
pub struct JointGaussian {
    variables: Vec<String>,
    mean: Vec<f64>,
    covariance: Vec<f64>,
}

impl JointGaussian {
    /// Variables, in canonical (sorted) order.
    #[must_use]
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Mean vector aligned with [`Self::variables`].
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Row-major covariance aligned with [`Self::variables`].
    #[must_use]
    pub fn covariance_matrix(&self) -> &[f64] {
        &self.covariance
    }

    fn position(&self, variable: &str) -> Option<usize> {
        self.variables.iter().position(|v| v == variable)
    }

    /// Mean of one variable.
    #[must_use]
    pub fn mean_of(&self, variable: &str) -> Option<f64> {
        self.position(variable).map(|i| self.mean[i])
    }

    /// Variance of one variable.
    #[must_use]
    pub fn variance_of(&self, variable: &str) -> Option<f64> {
        let i = self.position(variable)?;
        Some(self.covariance[i * self.variables.len() + i])
    }

    /// Covariance of two variables.
    #[must_use]
    pub fn covariance_of(&self, a: &str, b: &str) -> Option<f64> {
        let i = self.position(a)?;
        let j = self.position(b)?;
        Some(self.covariance[i * self.variables.len() + j])
    }
}

/// Columns of one structural sample, aligned with canonical variable order.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleTable {
    /// Variables, in canonical order.
    pub variables: Vec<String>,
    /// One column per variable.
    pub columns: Vec<Vec<f64>>,
}

/// One interventional mean in a linear contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastArm {
    /// Weight of the arm in the contrast.
    pub weight: f64,
    /// Hard interventions `(variable, value)` of the arm.
    pub interventions: Vec<(String, f64)>,
}

/// A linear contrast of interventional means of one outcome:
/// `sum_k weight_k * E[outcome | do(arm_k)]`.
#[derive(Clone, Debug, PartialEq)]
pub struct MeanContrast {
    /// Outcome variable.
    pub outcome: String,
    /// Arms of the contrast.
    pub arms: Vec<ContrastArm>,
}

/// Parameter-uncertainty propagation result for a [`MeanContrast`].
///
/// This is the variance of the functional induced by the declared parameter
/// covariance alone, by the first-order delta method. It is not a sampling
/// interval and carries no coverage claim.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionalUncertainty {
    /// The functional at the fitted parameters.
    pub point: f64,
    /// Gradient of the functional with respect to every parameter.
    pub gradient: Vec<(ParameterId, f64)>,
    /// First-order variance `g' Sigma g`.
    pub variance: f64,
    /// Square root of the variance.
    pub standard_error: f64,
    /// What the declared parameter law means, when any mechanism reports one.
    pub meaning: Option<UncertaintyMeaning>,
    /// How the dependence between mechanisms was licensed.
    pub dependence: &'static str,
    /// Method label.
    pub method: &'static str,
}

#[derive(Clone, Debug)]
struct Node {
    intercept: f64,
    sigma: f64,
    parents: Vec<(usize, f64)>,
    loadings: Vec<(usize, f64)>,
}

/// An executable joint linear-Gaussian model compiled from coherent mechanisms.
#[derive(Clone, Debug)]
pub struct ComposedLinearGaussian {
    spec: ComposedModelSpec,
    variables: Vec<String>,
    latents: Vec<String>,
    order: Vec<usize>,
    nodes: Vec<Node>,
    input_domains: Vec<Vec<Domain>>,
}

impl ComposedLinearGaussian {
    /// Compile declared mechanisms into one executable joint model.
    ///
    /// # Errors
    /// [`CompositionError::Incoherent`] with every finding when the
    /// declarations do not establish one acyclic joint law;
    /// [`CompositionError::Unsupported`] when the mechanisms are not
    /// linear-Gaussian.
    pub fn compile(spec: &ComposedModelSpec) -> Result<Self, CompositionError> {
        let findings = check_composition(spec);
        if !findings.is_empty() {
            return Err(CompositionError::Incoherent { findings });
        }
        let spec = spec.canonical();
        if spec.mechanisms.iter().any(|d| d.form.family() != MechanismFamily::LinearGaussian) {
            return Err(CompositionError::Unsupported {
                message: "the linear-Gaussian executor needs linear-Gaussian mechanisms".into(),
            });
        }
        let variables: Vec<String> = spec.mechanisms.iter().map(|d| d.variable.clone()).collect();
        let index: BTreeMap<&str, usize> =
            variables.iter().enumerate().map(|(i, v)| (v.as_str(), i)).collect();
        let mut latent_names: Vec<String> = spec
            .mechanisms
            .iter()
            .flat_map(|d| d.latents.iter().map(|l| l.latent.clone()))
            .collect();
        latent_names.sort();
        latent_names.dedup();
        let latent_index: BTreeMap<&str, usize> =
            latent_names.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();

        let mut nodes = Vec::with_capacity(variables.len());
        let mut input_domains: Vec<Vec<Domain>> = vec![Vec::new(); variables.len()];
        for d in &spec.mechanisms {
            let MechanismForm::LinearGaussian { intercept, sigma, coefficients } = &d.form else {
                return Err(CompositionError::Unsupported {
                    message: "non linear-Gaussian form".into(),
                });
            };
            let mut parents = Vec::with_capacity(coefficients.len());
            for (name, coef) in coefficients {
                let Some(&pi) = index.get(name.as_str()) else {
                    return Err(CompositionError::Numerical {
                        message: "parent index missing after coherence check".into(),
                    });
                };
                parents.push((pi, *coef));
            }
            for binding in &d.parents {
                if let Some(&pi) = index.get(binding.variable.as_str()) {
                    input_domains[pi].push(binding.domain.clone());
                }
            }
            let mut loadings = Vec::with_capacity(d.latents.len());
            for l in &d.latents {
                let (Some(&li), Some(loading)) = (latent_index.get(l.latent.as_str()), l.loading)
                else {
                    return Err(CompositionError::Numerical {
                        message: "latent loading missing after coherence check".into(),
                    });
                };
                loadings.push((li, loading));
            }
            nodes.push(Node { intercept: *intercept, sigma: *sigma, parents, loadings });
        }

        // Deterministic topological order: smallest canonical index first.
        let n = nodes.len();
        let mut done = vec![false; n];
        let mut order = Vec::with_capacity(n);
        while order.len() < n {
            let next = (0..n)
                .find(|&v| !done[v] && nodes[v].parents.iter().all(|&(p, _)| done[p]));
            let Some(v) = next else {
                return Err(CompositionError::Numerical {
                    message: "no topological order after coherence check".into(),
                });
            };
            done[v] = true;
            order.push(v);
        }
        Ok(Self { spec, variables, latents: latent_names, order, nodes, input_domains })
    }

    /// Variables in canonical order.
    #[must_use]
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// The canonical specification this model was compiled from.
    #[must_use]
    pub fn spec(&self) -> &ComposedModelSpec {
        &self.spec
    }

    /// Topological order of canonical indices.
    #[must_use]
    pub fn topological_order(&self) -> &[usize] {
        &self.order
    }

    fn resolve(
        &self,
        interventions: &[(&str, f64)],
    ) -> Result<Vec<Option<f64>>, CompositionError> {
        let mut fixed: Vec<Option<f64>> = vec![None; self.variables.len()];
        for &(name, value) in interventions {
            let Some(i) = self.variables.iter().position(|v| v == name) else {
                return Err(CompositionError::Unsupported {
                    message: format!("intervention names unknown variable `{name}`"),
                });
            };
            if !value.is_finite() {
                return Err(CompositionError::Unsupported {
                    message: format!("intervention value for `{name}` is not finite"),
                });
            }
            if fixed[i].is_some() {
                return Err(CompositionError::Unsupported {
                    message: format!("variable `{name}` is intervened on twice"),
                });
            }
            let own = &self.spec.mechanisms[i].output_support;
            let outside_consumer =
                self.input_domains[i].iter().any(|domain| !domain.contains_value(value));
            if outside_consumer || !own.contains_value(value) {
                return Err(incoherent(
                    CoherenceFinding::new(
                        CoherenceKind::Support,
                        "mechanism_composition.support_intervention_outside_domain",
                        name,
                        "intervene only within the domain every consumer of this variable was fitted on",
                    )
                    .comparing("a value inside the fitted domains", value.to_string()),
                ));
            }
            fixed[i] = Some(value);
        }
        Ok(fixed)
    }

    fn law_with(&self, fixed: &[Option<f64>]) -> JointGaussian {
        let n = self.variables.len();
        let dim = n + self.latents.len();
        let mut mean = vec![0.0_f64; n];
        let mut coef = vec![0.0_f64; n * dim];
        for &v in &self.order {
            if let Some(x) = fixed[v] {
                mean[v] = x;
                continue;
            }
            let node = &self.nodes[v];
            let mut m = node.intercept;
            for &(p, b) in &node.parents {
                m += b * mean[p];
                for s in 0..dim {
                    let add = b * coef[p * dim + s];
                    coef[v * dim + s] += add;
                }
            }
            coef[v * dim + v] += node.sigma;
            for &(l, loading) in &node.loadings {
                coef[v * dim + n + l] += loading;
            }
            mean[v] = m;
        }
        let mut covariance = vec![0.0_f64; n * n];
        for v in 0..n {
            for w in 0..=v {
                let mut s = 0.0;
                for k in 0..dim {
                    s += coef[v * dim + k] * coef[w * dim + k];
                }
                covariance[v * n + w] = s;
                covariance[w * n + v] = s;
            }
        }
        JointGaussian { variables: self.variables.clone(), mean, covariance }
    }

    /// The joint law of all variables under `do(interventions)`; an empty list
    /// is the observational law of the composition.
    ///
    /// # Errors
    /// An unknown or repeated variable, a non-finite value, or a value outside
    /// the domain a consumer was fitted on.
    pub fn law(&self, interventions: &[(&str, f64)]) -> Result<JointGaussian, CompositionError> {
        let fixed = self.resolve(interventions)?;
        Ok(self.law_with(&fixed))
    }

    /// `E[outcome | do(arm)]` for every arm, weighted and summed.
    ///
    /// # Errors
    /// As [`Self::law`], or an unknown outcome.
    pub fn mean_contrast(&self, contrast: &MeanContrast) -> Result<f64, CompositionError> {
        let mut total = 0.0;
        for arm in &contrast.arms {
            let pairs: Vec<(&str, f64)> =
                arm.interventions.iter().map(|(v, x)| (v.as_str(), *x)).collect();
            let law = self.law(&pairs)?;
            let Some(mean) = law.mean_of(&contrast.outcome) else {
                return Err(CompositionError::Unsupported {
                    message: format!("unknown outcome `{}`", contrast.outcome),
                });
            };
            total += arm.weight * mean;
        }
        Ok(total)
    }

    /// Propagate the declared parameter uncertainty to a linear contrast of
    /// interventional means by the first-order delta method.
    ///
    /// The functional is multilinear in the parameters, so the first-order
    /// variance omits terms of order `Sigma^2`; it is exact for a contrast
    /// that is linear in the parameters and otherwise accurate when the
    /// parameter covariance is small relative to the parameters. Only
    /// uncertainty in the intercepts and coefficients is propagated; the
    /// residual scales are treated as fixed, which is why only functionals of
    /// conditional means are offered.
    ///
    /// # Errors
    /// [`CompositionError::Incoherent`] with the blocking findings when a
    /// marginal law is missing, dependence is unknown (joint-law refusal), a
    /// cross-covariance pair is missing, or the assembled law is not positive
    /// semidefinite; and the errors of [`Self::law`].
    pub fn mean_contrast_uncertainty(
        &self,
        contrast: &MeanContrast,
    ) -> Result<FunctionalUncertainty, CompositionError> {
        let assembled = assemble_parameter_covariance(&self.spec)
            .map_err(|findings| CompositionError::Incoherent { findings })?;
        let n = self.variables.len();
        let Some(outcome) = self.variables.iter().position(|v| *v == contrast.outcome) else {
            return Err(CompositionError::Unsupported {
                message: format!("unknown outcome `{}`", contrast.outcome),
            });
        };
        let mut point = 0.0;
        let mut gradient: BTreeMap<ParameterId, f64> = BTreeMap::new();
        for arm in &contrast.arms {
            let pairs: Vec<(&str, f64)> =
                arm.interventions.iter().map(|(v, x)| (v.as_str(), *x)).collect();
            let fixed = self.resolve(&pairs)?;
            let law = self.law_with(&fixed);
            point += arm.weight * law.mean[outcome];
            // sens[v * n + j] = d mean_v / d intercept_j under the surgery.
            let mut sens = vec![0.0_f64; n * n];
            for &v in &self.order {
                if fixed[v].is_some() {
                    continue;
                }
                sens[v * n + v] += 1.0;
                for &(p, b) in &self.nodes[v].parents {
                    for j in 0..n {
                        let add = b * sens[p * n + j];
                        sens[v * n + j] += add;
                    }
                }
            }
            for (j, mechanism) in self.spec.mechanisms.iter().enumerate() {
                let s = sens[outcome * n + j];
                *gradient
                    .entry(ParameterId {
                        mechanism: mechanism.variable.clone(),
                        term: ParameterTerm::Intercept,
                    })
                    .or_insert(0.0) += arm.weight * s;
                for &(p, _) in &self.nodes[j].parents {
                    *gradient
                        .entry(ParameterId {
                            mechanism: mechanism.variable.clone(),
                            term: ParameterTerm::Coefficient(self.variables[p].clone()),
                        })
                        .or_insert(0.0) += arm.weight * s * law.mean[p];
                }
            }
        }
        let k = assembled.parameters.len();
        let g: Vec<f64> = assembled
            .parameters
            .iter()
            .map(|id| gradient.get(id).copied().unwrap_or(0.0))
            .collect();
        let mut variance = 0.0;
        for i in 0..k {
            for j in 0..k {
                variance += g[i] * assembled.covariance[i * k + j] * g[j];
            }
        }
        if variance < 0.0 {
            // A PSD matrix cannot give a negative quadratic form; this is
            // rounding at the level of the tolerance.
            variance = 0.0;
        }
        let meaning = self.spec.mechanisms.iter().find_map(|d| d.uncertainty.as_ref().map(|u| u.meaning));
        let gradient: Vec<(ParameterId, f64)> =
            assembled.parameters.iter().cloned().zip(g.iter().copied()).collect();
        Ok(FunctionalUncertainty {
            point,
            gradient,
            variance,
            standard_error: variance.sqrt(),
            meaning,
            dependence: assembled.dependence_label,
            method: "first_order_delta_method",
        })
    }

    /// Draw `n_rows` units from the (possibly intervened) joint model.
    ///
    /// Noise sources are keyed by name, so the draws of one variable do not
    /// depend on the other variables of the composition or on their order.
    ///
    /// # Errors
    /// As [`Self::law`]; a request above [`MAX_SAMPLE_CELLS`];
    /// [`CompositionError::Cancelled`] when cancellation is observed.
    pub fn sample(
        &self,
        interventions: &[(&str, f64)],
        n_rows: usize,
        ctx: &ExecutionContext,
    ) -> Result<SampleTable, CompositionError> {
        let fixed = self.resolve(interventions)?;
        let n = self.variables.len();
        let sources = n + self.latents.len();
        if n_rows == 0 || n_rows.saturating_mul(sources.max(1)) > MAX_SAMPLE_CELLS {
            return Err(CompositionError::Unsupported {
                message: format!("sample size {n_rows} outside 1..={}", MAX_SAMPLE_CELLS / sources.max(1)),
            });
        }
        let draw = |role: u64, name: &str| -> Result<Vec<f64>, CompositionError> {
            if ctx.cancellation.is_cancelled() {
                return Err(CompositionError::Cancelled);
            }
            let mut rng = ctx
                .rng
                .stream_for(COMPOSITION_STREAM_DOMAIN, noise_stream_index(role, name));
            let mut column = Vec::with_capacity(n_rows);
            for r in 0..n_rows {
                if r % 4096 == 4095 && ctx.cancellation.is_cancelled() {
                    return Err(CompositionError::Cancelled);
                }
                column.push(standard_normal(&mut rng));
            }
            Ok(column)
        };
        let mut latent_draws = Vec::with_capacity(self.latents.len());
        for name in &self.latents {
            latent_draws.push(draw(1, name)?);
        }
        let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n];
        for &v in &self.order {
            if let Some(x) = fixed[v] {
                columns[v] = vec![x; n_rows];
                continue;
            }
            let node = &self.nodes[v];
            let noise = draw(0, &self.variables[v])?;
            let mut column = Vec::with_capacity(n_rows);
            for r in 0..n_rows {
                let mut value = node.intercept + node.sigma * noise[r];
                for &(p, b) in &node.parents {
                    value += b * columns[p][r];
                }
                for &(l, loading) in &node.loadings {
                    value += loading * latent_draws[l][r];
                }
                column.push(value);
            }
            columns[v] = column;
        }
        Ok(SampleTable { variables: self.variables.clone(), columns })
    }
}
