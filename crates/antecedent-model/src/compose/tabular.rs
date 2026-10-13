//! Executable joint law of composed finite-discrete conditional tables.
//!
//! The joint law of an acyclic composition of tables is the Markov
//! factorization `P(v) = prod_i P(v_i | pa_i)`; `do(X = x)` is the truncated
//! factorization that drops the factor of every intervened variable
//! (Pearl 2009, ch. 3). The joint is enumerated exactly, so every probability
//! is a finite sum of products of the declared table entries. The enumeration
//! is bounded by [`MAX_FINITE_STATES`]; parameter uncertainty is not propagated
//! for this family.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;

use super::coherence::check_composition;
use super::descriptor::{ComposedModelSpec, MAX_FINITE_STATES, MechanismFamily, MechanismForm};
use super::{COMPOSITION_STREAM_DOMAIN, CompositionError, noise_stream_index};

/// The exact joint law of a finite composition under one regime.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteLaw {
    variables: Vec<String>,
    levels: Vec<Vec<String>>,
    probabilities: Vec<f64>,
}

impl FiniteLaw {
    /// Variables in canonical order.
    #[must_use]
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Levels of each variable, aligned with [`Self::variables`], sorted.
    #[must_use]
    pub fn levels(&self) -> &[Vec<String>] {
        &self.levels
    }

    /// Probability of every joint state. The last variable varies fastest.
    #[must_use]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Probability of the event that every named variable takes its named level.
    ///
    /// # Errors
    /// An unknown variable or level.
    pub fn probability(&self, event: &[(&str, &str)]) -> Result<f64, CompositionError> {
        let mut required: Vec<(usize, usize)> = Vec::with_capacity(event.len());
        for &(var, level) in event {
            let Some(i) = self.variables.iter().position(|v| v == var) else {
                return Err(CompositionError::Unsupported {
                    message: format!("event names unknown variable `{var}`"),
                });
            };
            let Some(l) = self.levels[i].iter().position(|x| x == level) else {
                return Err(CompositionError::Unsupported {
                    message: format!("event names unknown level `{level}` of `{var}`"),
                });
            };
            required.push((i, l));
        }
        let n = self.variables.len();
        let mut strides = vec![1_usize; n];
        for i in (0..n.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.levels[i + 1].len();
        }
        let mut total = 0.0;
        for (state, p) in self.probabilities.iter().enumerate() {
            let hit =
                required.iter().all(|&(i, l)| (state / strides[i]) % self.levels[i].len() == l);
            if hit {
                total += p;
            }
        }
        Ok(total)
    }

    /// Marginal `(level, probability)` of one variable.
    ///
    /// # Errors
    /// An unknown variable.
    pub fn marginal(&self, variable: &str) -> Result<Vec<(String, f64)>, CompositionError> {
        let Some(i) = self.variables.iter().position(|v| v == variable) else {
            return Err(CompositionError::Unsupported {
                message: format!("unknown variable `{variable}`"),
            });
        };
        let mut out = Vec::with_capacity(self.levels[i].len());
        for level in &self.levels[i] {
            out.push((level.clone(), self.probability(&[(variable, level.as_str())])?));
        }
        Ok(out)
    }
}

/// An executable finite joint model compiled from coherent tables.
#[derive(Clone, Debug)]
pub struct ComposedFinite {
    spec: ComposedModelSpec,
    variables: Vec<String>,
    levels: Vec<Vec<String>>,
    order: Vec<usize>,
    parents: Vec<Vec<usize>>,
    // tables[v][config * n_levels(v) + level]; config is mixed radix over the
    // producers' levels of the parents in canonical parent order.
    tables: Vec<Vec<f64>>,
    n_states: usize,
}

impl ComposedFinite {
    /// Compile declared tables into one executable joint model.
    ///
    /// # Errors
    /// [`CompositionError::Incoherent`] with every finding; an unsupported
    /// family or a joint space above [`MAX_FINITE_STATES`].
    pub fn compile(spec: &ComposedModelSpec) -> Result<Self, CompositionError> {
        let findings = check_composition(spec);
        if !findings.is_empty() {
            return Err(CompositionError::Incoherent { findings });
        }
        let spec = spec.canonical();
        if spec.mechanisms.iter().any(|d| d.form.family() != MechanismFamily::Tabular) {
            return Err(CompositionError::Unsupported {
                message: "the finite executor needs finite-table mechanisms".into(),
            });
        }
        let variables: Vec<String> = spec.mechanisms.iter().map(|d| d.variable.clone()).collect();
        let index: BTreeMap<&str, usize> =
            variables.iter().enumerate().map(|(i, v)| (v.as_str(), i)).collect();
        let mut levels: Vec<Vec<String>> = Vec::with_capacity(variables.len());
        for d in &spec.mechanisms {
            let MechanismForm::Tabular { levels: l, .. } = &d.form else {
                return Err(CompositionError::Unsupported { message: "non-tabular form".into() });
            };
            levels.push(l.clone());
        }
        let mut n_states = 1_usize;
        for l in &levels {
            n_states = n_states.saturating_mul(l.len());
        }
        if n_states > MAX_FINITE_STATES {
            return Err(CompositionError::Unsupported {
                message: format!(
                    "joint space of {n_states} states exceeds the enumeration bound {MAX_FINITE_STATES}"
                ),
            });
        }
        let mut parents: Vec<Vec<usize>> = Vec::with_capacity(variables.len());
        let mut tables: Vec<Vec<f64>> = Vec::with_capacity(variables.len());
        for d in &spec.mechanisms {
            let MechanismForm::Tabular { levels: own_levels, rows } = &d.form else {
                return Err(CompositionError::Unsupported { message: "non-tabular form".into() });
            };
            let mut pa = Vec::with_capacity(d.parents.len());
            for p in &d.parents {
                let Some(&pi) = index.get(p.variable.as_str()) else {
                    return Err(CompositionError::Numerical {
                        message: "parent index missing after coherence check".into(),
                    });
                };
                pa.push(pi);
            }
            let mut n_configs = 1_usize;
            for &pi in &pa {
                n_configs *= levels[pi].len();
            }
            let width = own_levels.len();
            let mut table = vec![f64::NAN; n_configs * width];
            for row in rows {
                let mut config = 0_usize;
                let mut producible = true;
                for (k, &pi) in pa.iter().enumerate() {
                    let level_name = &row.parent_levels[k].1;
                    if let Some(li) = levels[pi].iter().position(|l| l == level_name) {
                        config = config * levels[pi].len() + li;
                    } else {
                        producible = false;
                        break;
                    }
                }
                if !producible {
                    continue;
                }
                for (li, (_, p)) in row.probabilities.iter().enumerate() {
                    table[config * width + li] = *p;
                }
            }
            if table.iter().any(|p| p.is_nan()) {
                return Err(CompositionError::Numerical {
                    message: format!(
                        "table of `{}` leaves a producible configuration undefined",
                        d.variable
                    ),
                });
            }
            parents.push(pa);
            tables.push(table);
        }
        let n = variables.len();
        let mut done = vec![false; n];
        let mut order = Vec::with_capacity(n);
        while order.len() < n {
            let next = (0..n).find(|&v| !done[v] && parents[v].iter().all(|&p| done[p]));
            let Some(v) = next else {
                return Err(CompositionError::Numerical {
                    message: "no topological order after coherence check".into(),
                });
            };
            done[v] = true;
            order.push(v);
        }
        Ok(Self { spec, variables, levels, order, parents, tables, n_states })
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

    fn resolve(
        &self,
        interventions: &[(&str, &str)],
    ) -> Result<Vec<Option<usize>>, CompositionError> {
        let mut fixed: Vec<Option<usize>> = vec![None; self.variables.len()];
        for &(name, level) in interventions {
            let Some(i) = self.variables.iter().position(|v| v == name) else {
                return Err(CompositionError::Unsupported {
                    message: format!("intervention names unknown variable `{name}`"),
                });
            };
            let Some(l) = self.levels[i].iter().position(|x| x == level) else {
                return Err(CompositionError::Unsupported {
                    message: format!("intervention level `{level}` is not a level of `{name}`"),
                });
            };
            if fixed[i].is_some() {
                return Err(CompositionError::Unsupported {
                    message: format!("variable `{name}` is intervened on twice"),
                });
            }
            fixed[i] = Some(l);
        }
        Ok(fixed)
    }

    fn config(&self, v: usize, state: &[usize]) -> usize {
        let mut config = 0_usize;
        for &p in &self.parents[v] {
            config = config * self.levels[p].len() + state[p];
        }
        config
    }

    /// The exact joint law under `do(interventions)` by truncated
    /// factorization; an empty list is the observational law.
    ///
    /// # Errors
    /// An unknown variable or level; [`CompositionError::Cancelled`].
    pub fn law(
        &self,
        interventions: &[(&str, &str)],
        ctx: &ExecutionContext,
    ) -> Result<FiniteLaw, CompositionError> {
        let fixed = self.resolve(interventions)?;
        let n = self.variables.len();
        let mut state = vec![0_usize; n];
        let mut probabilities = Vec::with_capacity(self.n_states);
        for s in 0..self.n_states {
            if s % 4096 == 4095 && ctx.cancellation.is_cancelled() {
                return Err(CompositionError::Cancelled);
            }
            let mut rest = s;
            for i in (0..n).rev() {
                let width = self.levels[i].len();
                state[i] = rest % width;
                rest /= width;
            }
            let mut p = 1.0;
            for v in 0..n {
                let pinned = fixed[v];
                if let Some(l) = pinned {
                    if state[v] != l {
                        p = 0.0;
                        break;
                    }
                } else {
                    let width = self.levels[v].len();
                    p *= self.tables[v][self.config(v, &state) * width + state[v]];
                }
            }
            probabilities.push(p);
        }
        Ok(FiniteLaw {
            variables: self.variables.clone(),
            levels: self.levels.clone(),
            probabilities,
        })
    }

    /// Draw `n_rows` units by ancestral sampling; each row is the level index
    /// of every variable in canonical order.
    ///
    /// # Errors
    /// An unknown variable or level; [`CompositionError::Cancelled`].
    pub fn sample(
        &self,
        interventions: &[(&str, &str)],
        n_rows: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Vec<usize>>, CompositionError> {
        let fixed = self.resolve(interventions)?;
        let n = self.variables.len();
        if n_rows == 0 || n_rows > (1 << 22) {
            return Err(CompositionError::Unsupported {
                message: format!("sample size {n_rows} outside 1..={}", 1_u32 << 22),
            });
        }
        // One uniform stream per variable, keyed by its name.
        let mut uniforms: Vec<Vec<f64>> = vec![Vec::new(); n];
        for v in 0..n {
            if fixed[v].is_some() {
                continue;
            }
            if ctx.cancellation.is_cancelled() {
                return Err(CompositionError::Cancelled);
            }
            let mut rng = ctx
                .rng
                .stream_for(COMPOSITION_STREAM_DOMAIN, noise_stream_index(2, &self.variables[v]));
            uniforms[v] = (0..n_rows).map(|_| rng.next_f64()).collect();
        }
        let mut rows = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            if r % 4096 == 4095 && ctx.cancellation.is_cancelled() {
                return Err(CompositionError::Cancelled);
            }
            let mut state = vec![0_usize; n];
            for &v in &self.order {
                if let Some(l) = fixed[v] {
                    state[v] = l;
                    continue;
                }
                let width = self.levels[v].len();
                let base = self.config(v, &state) * width;
                let u = uniforms[v][r];
                let mut cumulative = 0.0;
                let mut chosen = width - 1;
                for l in 0..width {
                    cumulative += self.tables[v][base + l];
                    if u < cumulative {
                        chosen = l;
                        break;
                    }
                }
                state[v] = chosen;
            }
            rows.push(state);
        }
        Ok(rows)
    }
}
