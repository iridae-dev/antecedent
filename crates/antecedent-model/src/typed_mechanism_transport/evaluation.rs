use std::collections::{BTreeMap, BTreeSet};

use crate::compose::{MeanContrast, noise_stream_index};
use antecedent_core::{CausalRng, ExecutionContext, mix64};

use super::{
    CoherenceKind, CompositionError, ContrastEstimate, DETAIL_DEPENDENCE_INCOMPLETE,
    DETAIL_DEPENDENCE_INVALID, DETAIL_DEPENDENCE_UNKNOWN, DETAIL_UNKNOWN_VARIABLE, FittedMechanism,
    MAX_TYPED_DRAWS, MechanismContext, MechanismRole, ParameterDependence,
    SelectionDiagramDeclaration, TYPED_TRANSPORT_STREAM_DOMAIN, TransportedTypedModel,
    TypedDisposition, TypedFunctionalUncertainty, UNCERTAINTY_SCOPE, matvec, single,
    valid_covariance,
};

struct ContrastRun {
    value: f64,
    monte_carlo_se: f64,
    gradient: Vec<f64>,
    involved: BTreeSet<usize>,
}

impl TransportedTypedModel {
    /// Every component, sorted by variable.
    #[must_use]
    pub fn dispositions(&self) -> &[TypedDisposition] {
        &self.dispositions
    }

    /// The mechanisms in topological order.
    #[must_use]
    pub fn mechanisms(&self) -> &[FittedMechanism] {
        &self.mechanisms
    }

    /// The target population and regime.
    #[must_use]
    pub fn target(&self) -> &MechanismContext {
        &self.target
    }

    /// The selection diagram the model was admitted under.
    #[must_use]
    pub fn diagram(&self) -> &SelectionDiagramDeclaration {
        &self.diagram
    }

    /// Variables whose mechanism was carried from the source, sorted.
    #[must_use]
    pub fn transported_variables(&self) -> Vec<&str> {
        self.dispositions
            .iter()
            .filter(|d| d.role == MechanismRole::Transported)
            .map(|d| d.variable.as_str())
            .collect()
    }

    /// Identity of the transported composition. Equal requests give equal
    /// identities; a changed number at an unchanged shape changes it.
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    /// Lowercase hexadecimal [`Self::identity`].
    #[must_use]
    pub fn identity_hex(&self) -> String {
        antecedent_core::identity::to_hex(&self.identity)
    }

    fn offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.mechanisms.len());
        let mut acc = 0;
        for m in &self.mechanisms {
            out.push(acc);
            acc += m.theta.len();
        }
        out
    }

    fn total_dimension(&self) -> usize {
        self.mechanisms.iter().map(|m| m.theta.len()).sum()
    }

    /// `sum_k w_k E[outcome | do(arm_k)]` by Monte Carlo over `draws` paired units per arm.
    ///
    /// # Errors
    /// [`CompositionError::Incoherent`] for an unknown variable or a malformed
    /// contrast; [`CompositionError::Unsupported`] for a draw count outside
    /// `1..=`[`MAX_TYPED_DRAWS`]; [`CompositionError::Numerical`] for a
    /// non-finite draw; [`CompositionError::Cancelled`].
    pub fn mean_contrast(
        &self,
        contrast: &MeanContrast,
        draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<ContrastEstimate, CompositionError> {
        let run = self.run_contrast(contrast, draws, ctx, false)?;
        Ok(ContrastEstimate { value: run.value, monte_carlo_se: run.monte_carlo_se, draws })
    }

    /// First-order parameter uncertainty of a contrast, scoped honestly.
    ///
    /// # Errors
    /// As [`Self::mean_contrast`]; additionally a refusal when more than one
    /// mechanism's parameters are involved and the declared dependence is
    /// unknown or omits a pair of them.
    pub fn mean_contrast_uncertainty(
        &self,
        contrast: &MeanContrast,
        draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<TypedFunctionalUncertainty, CompositionError> {
        let run = self.run_contrast(contrast, draws, ctx, true)?;
        let (sigma, dependence) = self.parameter_covariance(&run.involved)?;
        let d = run.gradient.len();
        let solved = matvec(&sigma, d, &run.gradient);
        let terms: Vec<f64> = run.gradient.iter().zip(&solved).map(|(a, b)| a * b).collect();
        let variance: f64 = terms.iter().sum();
        let magnitude: f64 = terms.iter().map(|x| x.abs()).sum();
        let tolerance = 64.0 * f64::EPSILON * d as f64 * magnitude;
        if !variance.is_finite() || !magnitude.is_finite() || variance < -tolerance {
            return Err(single(
                CoherenceKind::PosteriorDependence,
                DETAIL_DEPENDENCE_INVALID,
                "parameter_dependence",
                "use a finite positive semidefinite parameter covariance",
            ));
        }
        let variance = variance.max(0.0);
        let offsets = self.offsets();
        let mut gradient = Vec::new();
        for &idx in &run.involved {
            let m = &self.mechanisms[idx];
            for (j, label) in m.parameter_labels().into_iter().enumerate() {
                gradient.push((m.variable().to_owned(), label, run.gradient[offsets[idx] + j]));
            }
        }
        Ok(TypedFunctionalUncertainty {
            estimate: ContrastEstimate {
                value: run.value,
                monte_carlo_se: run.monte_carlo_se,
                draws,
            },
            gradient,
            variance,
            standard_error: variance.sqrt(),
            dependence,
            scope: UNCERTAINTY_SCOPE,
        })
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the scientific calculation and its checks form one bounded procedure"
    )]
    fn parameter_covariance(
        &self,
        involved: &BTreeSet<usize>,
    ) -> Result<(Vec<f64>, &'static str), CompositionError> {
        let d = self.total_dimension();
        let offsets = self.offsets();
        let mut sigma = vec![0.0; d * d];
        for (idx, m) in self.mechanisms.iter().enumerate() {
            let k = m.theta.len();
            for a in 0..k {
                for b in 0..k {
                    sigma[(offsets[idx] + a) * d + offsets[idx] + b] = m.covariance[a * k + b];
                }
            }
        }
        let index_of = |v: &str| self.mechanisms.iter().position(|m| m.variable() == v);
        let label = match &self.dependence {
            ParameterDependence::Unknown => {
                if involved.len() > 1 {
                    return Err(single(
                        CoherenceKind::PosteriorDependence,
                        DETAIL_DEPENDENCE_UNKNOWN,
                        "parameter_dependence",
                        "declare independence by design, a joint covariance, or cross-covariance blocks",
                    ));
                }
                "unknown_single_mechanism"
            }
            ParameterDependence::IndependentByDesign { .. } => "independent_by_design",
            ParameterDependence::Joint { variables, covariance } => {
                let listed: Vec<usize> = variables.iter().filter_map(|v| index_of(v)).collect();
                if involved.iter().any(|i| !listed.contains(i)) {
                    return Err(single(
                        CoherenceKind::PosteriorDependence,
                        DETAIL_DEPENDENCE_INCOMPLETE,
                        "joint",
                        "the joint covariance lists every mechanism whose parameters the contrast involves",
                    ));
                }
                let joint_dim: usize = listed.iter().map(|&i| self.mechanisms[i].theta.len()).sum();
                let mut joint_offsets = Vec::with_capacity(listed.len());
                let mut acc = 0;
                for &i in &listed {
                    joint_offsets.push(acc);
                    acc += self.mechanisms[i].theta.len();
                }
                for (pa, &ia) in listed.iter().enumerate() {
                    for (pb, &ib) in listed.iter().enumerate() {
                        for a in 0..self.mechanisms[ia].theta.len() {
                            for b in 0..self.mechanisms[ib].theta.len() {
                                sigma[(offsets[ia] + a) * d + offsets[ib] + b] = covariance
                                    [(joint_offsets[pa] + a) * joint_dim + joint_offsets[pb] + b];
                            }
                        }
                    }
                }
                "joint"
            }
            ParameterDependence::CrossCovariance { blocks } => {
                let involved_list: Vec<usize> = involved.iter().copied().collect();
                for (pos, &ia) in involved_list.iter().enumerate() {
                    for &ib in &involved_list[pos + 1..] {
                        let (va, vb) =
                            (self.mechanisms[ia].variable(), self.mechanisms[ib].variable());
                        let (ka, kb) =
                            (self.mechanisms[ia].theta.len(), self.mechanisms[ib].theta.len());
                        let direct = blocks.iter().find(|b| b.first == va && b.second == vb);
                        let reverse = blocks.iter().find(|b| b.first == vb && b.second == va);
                        match (direct, reverse) {
                            (Some(b), _) => {
                                for a in 0..ka {
                                    for c in 0..kb {
                                        let v = b.values[a * kb + c];
                                        sigma[(offsets[ia] + a) * d + offsets[ib] + c] = v;
                                        sigma[(offsets[ib] + c) * d + offsets[ia] + a] = v;
                                    }
                                }
                            }
                            (None, Some(b)) => {
                                for a in 0..kb {
                                    for c in 0..ka {
                                        let v = b.values[a * ka + c];
                                        sigma[(offsets[ib] + a) * d + offsets[ia] + c] = v;
                                        sigma[(offsets[ia] + c) * d + offsets[ib] + a] = v;
                                    }
                                }
                            }
                            (None, None) => {
                                return Err(single(
                                    CoherenceKind::PosteriorDependence,
                                    DETAIL_DEPENDENCE_INCOMPLETE,
                                    &format!("{va},{vb}"),
                                    "declare a cross-covariance block for every pair of involved mechanisms",
                                ));
                            }
                        }
                    }
                }
                "cross_covariance"
            }
        };
        let indices: Vec<usize> = involved
            .iter()
            .flat_map(|&i| offsets[i]..offsets[i] + self.mechanisms[i].theta.len())
            .collect();
        if !indices.is_empty() {
            let principal: Vec<f64> = indices
                .iter()
                .flat_map(|&i| indices.iter().map(move |&j| (i, j)))
                .map(|(i, j)| sigma[i * d + j])
                .collect();
            if !valid_covariance(&principal, indices.len()) {
                return Err(single(
                    CoherenceKind::PosteriorDependence,
                    DETAIL_DEPENDENCE_INVALID,
                    "parameter_dependence",
                    "use a symmetric positive semidefinite parameter covariance",
                ));
            }
        }
        Ok((sigma, label))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the scientific calculation and its checks form one bounded procedure"
    )]
    fn run_contrast(
        &self,
        contrast: &MeanContrast,
        draws: usize,
        ctx: &ExecutionContext,
        with_gradient: bool,
    ) -> Result<ContrastRun, CompositionError> {
        if draws == 0 || draws > MAX_TYPED_DRAWS {
            return Err(CompositionError::Unsupported {
                message: format!("draw count {draws} outside 1..={MAX_TYPED_DRAWS}"),
            });
        }
        if contrast.arms.is_empty() || contrast.arms.iter().any(|a| !a.weight.is_finite()) {
            return Err(single(
                CoherenceKind::Shape,
                DETAIL_UNKNOWN_VARIABLE,
                "contrast",
                "a contrast has at least one arm and finite weights",
            ));
        }
        let index_of: BTreeMap<&str, usize> =
            self.mechanisms.iter().enumerate().map(|(i, m)| (m.variable(), i)).collect();
        let outcome = *index_of.get(contrast.outcome.as_str()).ok_or_else(|| {
            single(
                CoherenceKind::VariableIdentity,
                DETAIL_UNKNOWN_VARIABLE,
                &contrast.outcome,
                "the outcome is a variable of the transported model",
            )
        })?;
        let offsets = self.offsets();
        let total = self.total_dimension();
        let parent_indices: Vec<Vec<usize>> = self
            .mechanisms
            .iter()
            .map(|m| {
                m.spec.parents.iter().filter_map(|p| index_of.get(p.as_str()).copied()).collect()
            })
            .collect();

        let mut arm_values: Vec<Vec<f64>> = Vec::with_capacity(contrast.arms.len());
        let mut gradient = vec![0.0; total];
        let mut involved: BTreeSet<usize> = BTreeSet::new();

        for arm in &contrast.arms {
            let mut fixed: BTreeMap<usize, f64> = BTreeMap::new();
            for (name, value) in &arm.interventions {
                let idx = *index_of.get(name.as_str()).ok_or_else(|| {
                    single(
                        CoherenceKind::VariableIdentity,
                        DETAIL_UNKNOWN_VARIABLE,
                        name,
                        "an intervention names a variable of the transported model",
                    )
                })?;
                if !value.is_finite() || fixed.insert(idx, *value).is_some() {
                    return Err(single(
                        CoherenceKind::Shape,
                        DETAIL_UNKNOWN_VARIABLE,
                        name,
                        "an intervention value is finite and each variable is set once per arm",
                    ));
                }
            }
            // Relevant nodes: ancestors of the outcome, not looking past an intervened node.
            let mut relevant: BTreeSet<usize> = BTreeSet::new();
            let mut stack = vec![outcome];
            while let Some(node) = stack.pop() {
                if !relevant.insert(node) {
                    continue;
                }
                if !fixed.contains_key(&node) {
                    stack.extend(parent_indices[node].iter().copied());
                }
            }
            let order: Vec<usize> =
                (0..self.mechanisms.len()).filter(|i| relevant.contains(i)).collect();
            if !fixed.contains_key(&outcome) {
                for &i in &order {
                    if !fixed.contains_key(&i) {
                        involved.insert(i);
                    }
                }
            }
            let seeds: Vec<u64> = (0..self.mechanisms.len())
                .map(|i| {
                    ctx.rng
                        .stream_for(
                            TYPED_TRANSPORT_STREAM_DOMAIN,
                            noise_stream_index(0, self.mechanisms[i].variable()),
                        )
                        .next_u64()
                })
                .collect();

            let mut values = vec![0.0; self.mechanisms.len()];
            let mut f_units = Vec::with_capacity(draws);
            let mut sum_fs = vec![0.0; total];
            let mut sum_s = vec![0.0; total];
            let mut sum_pathwise = vec![0.0; total];
            let mut unit_scores = vec![0.0; total];
            for unit in 0..draws {
                if unit % 4096 == 4095 && ctx.cancellation.is_cancelled() {
                    return Err(CompositionError::Cancelled);
                }
                if with_gradient {
                    unit_scores.iter_mut().for_each(|s| *s = 0.0);
                }
                let mut f = f64::NAN;
                for &node in &order {
                    if let Some(v) = fixed.get(&node) {
                        values[node] = *v;
                        if node == outcome {
                            f = *v;
                        }
                        continue;
                    }
                    let m = &self.mechanisms[node];
                    let raw: Vec<f64> = parent_indices[node].iter().map(|&p| values[p]).collect();
                    if node == outcome {
                        f = m.mean_with(&m.theta, &raw);
                        if with_gradient {
                            let mut shifted = m.theta.clone();
                            for j in 0..m.theta.len() {
                                let h = 1e-5 * (1.0 + m.theta[j].abs());
                                shifted[j] = m.theta[j] + h;
                                let up = m.mean_with(&shifted, &raw);
                                shifted[j] = m.theta[j] - h;
                                let down = m.mean_with(&shifted, &raw);
                                shifted[j] = m.theta[j];
                                sum_pathwise[offsets[node] + j] += (up - down) / (2.0 * h);
                            }
                        }
                        continue;
                    }
                    let mut rng = CausalRng::from_seed(mix64(
                        seeds[node].wrapping_add(mix64((unit as u64).wrapping_add(1))),
                    ));
                    let x = m.sample_with(&m.theta, &raw, &mut rng);
                    if !x.is_finite() {
                        return Err(CompositionError::Numerical {
                            message: format!("non-finite draw of `{}`", m.variable()),
                        });
                    }
                    values[node] = x;
                    if with_gradient {
                        let mut shifted = m.theta.clone();
                        for j in 0..m.theta.len() {
                            let h = 1e-5 * (1.0 + m.theta[j].abs());
                            shifted[j] = m.theta[j] + h;
                            let up = m.log_density_with(&shifted, &raw, x);
                            shifted[j] = m.theta[j] - h;
                            let down = m.log_density_with(&shifted, &raw, x);
                            shifted[j] = m.theta[j];
                            unit_scores[offsets[node] + j] = (up - down) / (2.0 * h);
                        }
                    }
                }
                if !f.is_finite() {
                    return Err(CompositionError::Numerical {
                        message: format!("non-finite outcome for `{}`", contrast.outcome),
                    });
                }
                if with_gradient {
                    for j in 0..total {
                        sum_fs[j] += f * unit_scores[j];
                        sum_s[j] += unit_scores[j];
                    }
                }
                f_units.push(f);
            }
            if with_gradient {
                let n = draws as f64;
                let mean_f = f_units.iter().sum::<f64>() / n;
                for j in 0..total {
                    gradient[j] +=
                        arm.weight * (sum_fs[j] / n - mean_f * sum_s[j] / n + sum_pathwise[j] / n);
                }
            }
            arm_values.push(f_units);
        }

        let n = draws as f64;
        let combos: Vec<f64> = (0..draws)
            .map(|i| contrast.arms.iter().zip(&arm_values).map(|(a, v)| a.weight * v[i]).sum())
            .collect();
        let value = combos.iter().sum::<f64>() / n;
        let monte_carlo_se = if draws > 1 {
            (combos.iter().map(|c| (c - value) * (c - value)).sum::<f64>() / (n - 1.0) / n).sqrt()
        } else {
            f64::NAN
        };
        Ok(ContrastRun { value, monte_carlo_se, gradient, involved })
    }
}
