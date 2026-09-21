//! Mechanism store wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_model::{BasisTerm, CompiledMechanismStore, MechanismSlot, ParentBasis};
use serde::{Deserialize, Serialize};

use crate::error::IoError;

/// Model kind tag for bundles.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKindWire {
    /// Probabilistic causal model.
    Pcm,
    /// Structural causal model.
    Scm,
    /// Invertible SCM.
    InvertibleScm,
}

/// One mechanism slot on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MechanismSlotWire {
    /// Vacant.
    Vacant,
    /// Pending fit.
    Pending {
        /// Family id.
        family_id: String,
    },
    /// Linear Gaussian.
    LinearGaussian {
        /// Intercept.
        intercept: f64,
        /// Parent coeffs.
        coeffs: Vec<f64>,
        /// Residual σ.
        sigma: f64,
    },
    /// Discrete.
    Discrete {
        /// Support.
        support: Vec<f64>,
        /// Unconditional probs.
        probs: Vec<f64>,
        /// Optional logit coeffs.
        logit_coeffs: Option<Vec<f64>>,
    },
    /// Constant.
    Constant {
        /// Value.
        value: f64,
    },
    /// Hierarchical linear Gaussian.
    HierarchicalLinear {
        /// Intercept.
        intercept: f64,
        /// Coeffs.
        coeffs: Vec<f64>,
        /// Sigma.
        sigma: f64,
        /// Shrinkage.
        shrinkage: f64,
    },
    /// BVAR-style linear.
    Bvar {
        /// Intercept.
        intercept: f64,
        /// Coeffs.
        coeffs: Vec<f64>,
        /// Sigma.
        sigma: f64,
    },
    /// LGSSM.
    LinearGaussianStateSpace {
        /// AR.
        a: f64,
        /// Process std.
        process_std: f64,
        /// Obs std.
        obs_std: f64,
        /// Initial mean.
        initial_mean: f64,
    },
    /// LGSSM residual process with a conditional regression mean.
    ConditionalLinearGaussianStateSpace {
        /// Regression intercept.
        intercept: f64,
        /// Parent coefficients in graph-parent order.
        coeffs: Vec<f64>,
        /// Residual autoregressive coefficient.
        a: f64,
        /// Process standard deviation.
        process_std: f64,
        /// Observation standard deviation.
        obs_std: f64,
        /// Initial residual-state mean.
        initial_mean: f64,
    },
    /// GP dual form.
    GaussianProcess {
        /// Length scale.
        length_scale: f64,
        /// Variance.
        variance: f64,
        /// Noise std.
        noise_std: f64,
        /// Prior mean the surface reverts to away from the training rows. Absent on
        /// bundles written before it existed, where the surface reverted to zero.
        #[serde(default)]
        mean: f64,
        /// Training X row-major.
        x_train: Vec<f64>,
        /// `n_train`.
        n_train: usize,
        /// `n_parents`.
        n_parents: usize,
        /// Dual coefficients.
        alpha: Vec<f64>,
    },
    /// Linear Gaussian in a deterministic parent-basis expansion.
    LinearBasis {
        /// Intercept.
        intercept: f64,
        /// Expansion of the parent row.
        basis: ParentBasisWire,
        /// One coefficient per expanded column.
        coeffs: Vec<f64>,
        /// Residual σ.
        sigma: f64,
    },
    /// Discrete categorical with logits linear in a parent-basis expansion.
    DiscreteBasis {
        /// Support.
        support: Vec<f64>,
        /// Marginal probs.
        probs: Vec<f64>,
        /// Expansion of the parent row.
        basis: ParentBasisWire,
        /// Row-major `k × (1 + n_terms)` logit coefficients.
        logit_coeffs: Vec<f64>,
    },
}

/// A [`ParentBasis`] on the wire. Every constant the expansion needs is here, so
/// a decoded mechanism evaluates at unseen covariate cells without the training
/// table.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ParentBasisWire {
    /// Parent arity.
    pub n_parents: usize,
    /// Per-parent centers.
    pub centers: Vec<f64>,
    /// Per-parent scales.
    pub scales: Vec<f64>,
    /// Per-parent interior knots (standardized scale).
    pub knots: Vec<Vec<f64>>,
    /// Expanded columns in order.
    pub terms: Vec<BasisTermWire>,
}

/// One [`BasisTerm`] on the wire.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BasisTermWire {
    /// `z_parent^degree`.
    Power {
        /// Parent index.
        parent: u32,
        /// Exponent.
        degree: u8,
    },
    /// Truncated cubic at a stored knot.
    Truncated {
        /// Parent index.
        parent: u32,
        /// Knot index.
        knot: u32,
    },
    /// Product of two earlier columns.
    Product {
        /// Left column index.
        left: u32,
        /// Right column index.
        right: u32,
    },
}

fn basis_to_wire(basis: &ParentBasis) -> ParentBasisWire {
    ParentBasisWire {
        n_parents: basis.n_parents(),
        centers: basis.centers().to_vec(),
        scales: basis.scales().to_vec(),
        knots: basis.knots().iter().map(|k| k.to_vec()).collect(),
        terms: basis
            .terms()
            .iter()
            .map(|t| match *t {
                BasisTerm::Power { parent, degree } => BasisTermWire::Power { parent, degree },
                BasisTerm::Truncated { parent, knot } => BasisTermWire::Truncated { parent, knot },
                BasisTerm::Product { left, right } => BasisTermWire::Product { left, right },
            })
            .collect(),
    }
}

fn basis_from_wire(w: &ParentBasisWire) -> Result<ParentBasis, IoError> {
    let terms: Vec<BasisTerm> = w
        .terms
        .iter()
        .map(|t| match *t {
            BasisTermWire::Power { parent, degree } => BasisTerm::Power { parent, degree },
            BasisTermWire::Truncated { parent, knot } => BasisTerm::Truncated { parent, knot },
            BasisTermWire::Product { left, right } => BasisTerm::Product { left, right },
        })
        .collect();
    ParentBasis::new(
        w.n_parents,
        Arc::from(w.centers.as_slice()),
        Arc::from(w.scales.as_slice()),
        w.knots.iter().map(|k| Arc::from(k.as_slice())).collect::<Vec<Arc<[f64]>>>().into(),
        Arc::from(terms),
    )
    .map_err(|e| IoError::Convert(format!("invalid parent basis: {e}")))
}

/// Mechanism store wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MechanismStoreWire {
    /// Slots in dense node order.
    pub slots: Vec<MechanismSlotWire>,
}

/// Encode mechanism store.
///
/// # Errors
///
/// [`MechanismSlot::Dynamic`] cannot be serialized .
pub fn mechanisms_to_wire(store: &CompiledMechanismStore) -> Result<MechanismStoreWire, IoError> {
    let mut slots = Vec::with_capacity(store.slots.len());
    for s in store.slots.iter() {
        slots.push(slot_to_wire(s)?);
    }
    Ok(MechanismStoreWire { slots })
}

/// Decode mechanism store.
///
/// # Errors
///
/// A parent basis whose stored constants violate its invariants.
pub fn mechanisms_from_wire(w: &MechanismStoreWire) -> Result<CompiledMechanismStore, IoError> {
    Ok(CompiledMechanismStore {
        slots: w.slots.iter().map(slot_from_wire).collect::<Result<Vec<_>, _>>()?.into(),
    })
}

fn slot_to_wire(s: &MechanismSlot) -> Result<MechanismSlotWire, IoError> {
    match s {
        MechanismSlot::Vacant => Ok(MechanismSlotWire::Vacant),
        MechanismSlot::Pending { family_id } => {
            Ok(MechanismSlotWire::Pending { family_id: family_id.to_string() })
        }
        MechanismSlot::LinearGaussian { intercept, coeffs, sigma } => {
            Ok(MechanismSlotWire::LinearGaussian {
                intercept: *intercept,
                coeffs: coeffs.to_vec(),
                sigma: *sigma,
            })
        }
        MechanismSlot::Discrete { support, probs, logit_coeffs } => {
            Ok(MechanismSlotWire::Discrete {
                support: support.to_vec(),
                probs: probs.to_vec(),
                logit_coeffs: logit_coeffs.as_ref().map(|c| c.to_vec()),
            })
        }
        MechanismSlot::Constant { value } => Ok(MechanismSlotWire::Constant { value: *value }),
        MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, shrinkage } => {
            Ok(MechanismSlotWire::HierarchicalLinear {
                intercept: *intercept,
                coeffs: coeffs.to_vec(),
                sigma: *sigma,
                shrinkage: *shrinkage,
            })
        }
        MechanismSlot::Bvar { intercept, coeffs, sigma } => Ok(MechanismSlotWire::Bvar {
            intercept: *intercept,
            coeffs: coeffs.to_vec(),
            sigma: *sigma,
        }),
        MechanismSlot::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean } => {
            Ok(MechanismSlotWire::LinearGaussianStateSpace {
                a: *a,
                process_std: *process_std,
                obs_std: *obs_std,
                initial_mean: *initial_mean,
            })
        }
        MechanismSlot::ConditionalLinearGaussianStateSpace {
            intercept,
            coeffs,
            a,
            process_std,
            obs_std,
            initial_mean,
        } => Ok(MechanismSlotWire::ConditionalLinearGaussianStateSpace {
            intercept: *intercept,
            coeffs: coeffs.to_vec(),
            a: *a,
            process_std: *process_std,
            obs_std: *obs_std,
            initial_mean: *initial_mean,
        }),
        MechanismSlot::GaussianProcess {
            length_scale,
            variance,
            noise_std,
            mean,
            x_train,
            n_train,
            n_parents,
            alpha,
        } => Ok(MechanismSlotWire::GaussianProcess {
            length_scale: *length_scale,
            variance: *variance,
            noise_std: *noise_std,
            mean: *mean,
            x_train: x_train.to_vec(),
            n_train: *n_train,
            n_parents: *n_parents,
            alpha: alpha.to_vec(),
        }),
        MechanismSlot::LinearBasis { intercept, basis, coeffs, sigma } => {
            Ok(MechanismSlotWire::LinearBasis {
                intercept: *intercept,
                basis: basis_to_wire(basis),
                coeffs: coeffs.to_vec(),
                sigma: *sigma,
            })
        }
        MechanismSlot::DiscreteBasis { support, probs, basis, logit_coeffs } => {
            Ok(MechanismSlotWire::DiscreteBasis {
                support: support.to_vec(),
                probs: probs.to_vec(),
                basis: basis_to_wire(basis),
                logit_coeffs: logit_coeffs.to_vec(),
            })
        }
        MechanismSlot::Dynamic { id, .. } => Err(IoError::Convert(format!(
            "cannot serialize Dynamic mechanism slot `{id}` (Python/user callbacks are not artifact-safe)"
        ))),
    }
}

fn slot_from_wire(s: &MechanismSlotWire) -> Result<MechanismSlot, IoError> {
    Ok(match s {
        MechanismSlotWire::Vacant => MechanismSlot::Vacant,
        MechanismSlotWire::Pending { family_id } => {
            MechanismSlot::Pending { family_id: Arc::from(family_id.as_str()) }
        }
        MechanismSlotWire::LinearGaussian { intercept, coeffs, sigma } => {
            MechanismSlot::LinearGaussian {
                intercept: *intercept,
                coeffs: Arc::from(coeffs.as_slice()),
                sigma: *sigma,
            }
        }
        MechanismSlotWire::Discrete { support, probs, logit_coeffs } => MechanismSlot::Discrete {
            support: Arc::from(support.as_slice()),
            probs: Arc::from(probs.as_slice()),
            logit_coeffs: logit_coeffs.as_ref().map(|c| Arc::from(c.as_slice())),
        },
        MechanismSlotWire::Constant { value } => MechanismSlot::Constant { value: *value },
        MechanismSlotWire::HierarchicalLinear { intercept, coeffs, sigma, shrinkage } => {
            MechanismSlot::HierarchicalLinear {
                intercept: *intercept,
                coeffs: Arc::from(coeffs.as_slice()),
                sigma: *sigma,
                shrinkage: *shrinkage,
            }
        }
        MechanismSlotWire::Bvar { intercept, coeffs, sigma } => MechanismSlot::Bvar {
            intercept: *intercept,
            coeffs: Arc::from(coeffs.as_slice()),
            sigma: *sigma,
        },
        MechanismSlotWire::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean } => {
            MechanismSlot::LinearGaussianStateSpace {
                a: *a,
                process_std: *process_std,
                obs_std: *obs_std,
                initial_mean: *initial_mean,
            }
        }
        MechanismSlotWire::ConditionalLinearGaussianStateSpace {
            intercept,
            coeffs,
            a,
            process_std,
            obs_std,
            initial_mean,
        } => MechanismSlot::ConditionalLinearGaussianStateSpace {
            intercept: *intercept,
            coeffs: Arc::from(coeffs.as_slice()),
            a: *a,
            process_std: *process_std,
            obs_std: *obs_std,
            initial_mean: *initial_mean,
        },
        MechanismSlotWire::GaussianProcess {
            length_scale,
            variance,
            noise_std,
            mean,
            x_train,
            n_train,
            n_parents,
            alpha,
        } => MechanismSlot::GaussianProcess {
            length_scale: *length_scale,
            variance: *variance,
            noise_std: *noise_std,
            mean: *mean,
            x_train: Arc::from(x_train.as_slice()),
            n_train: *n_train,
            n_parents: *n_parents,
            alpha: Arc::from(alpha.as_slice()),
        },
        MechanismSlotWire::LinearBasis { intercept, basis, coeffs, sigma } => {
            MechanismSlot::LinearBasis {
                intercept: *intercept,
                basis: basis_from_wire(basis)?,
                coeffs: Arc::from(coeffs.as_slice()),
                sigma: *sigma,
            }
        }
        MechanismSlotWire::DiscreteBasis { support, probs, basis, logit_coeffs } => {
            MechanismSlot::DiscreteBasis {
                support: Arc::from(support.as_slice()),
                probs: Arc::from(probs.as_slice()),
                basis: basis_from_wire(basis)?,
                logit_coeffs: Arc::from(logit_coeffs.as_slice()),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_model::DynamicMechanism;
    use std::sync::Arc;

    struct StubMech;
    impl DynamicMechanism for StubMech {
        fn sample_noise_column(
            &self,
            n_rows: usize,
            _rng: &mut antecedent_core::CausalRng,
            output: &mut [f64],
        ) -> Result<(), antecedent_model::ModelError> {
            output[..n_rows].fill(0.0);
            Ok(())
        }
        fn evaluate_column(
            &self,
            parents: antecedent_model::ParentBatch<'_>,
            _noise: &[f64],
            output: &mut [f64],
            _ws: &mut antecedent_model::MechanismWorkspace,
        ) -> Result<(), antecedent_model::ModelError> {
            output[..parents.n_rows].fill(0.0);
            Ok(())
        }
    }

    #[test]
    fn conditional_lgssm_roundtrip_retains_regression_and_state_parameters() {
        let store = CompiledMechanismStore {
            slots: Arc::from([MechanismSlot::ConditionalLinearGaussianStateSpace {
                intercept: 3.0,
                coeffs: Arc::from([1.5, -2.0]),
                a: 0.7,
                process_std: 0.2,
                obs_std: 0.1,
                initial_mean: -0.5,
            }]),
        };
        let wire = mechanisms_to_wire(&store).unwrap();
        let bytes = crate::to_cbor(&wire).unwrap();
        let decoded = crate::from_cbor(&bytes).unwrap();
        let restored = mechanisms_from_wire(&decoded).unwrap();
        assert_eq!(mechanisms_to_wire(&restored).unwrap(), wire);
        let MechanismSlot::ConditionalLinearGaussianStateSpace { intercept, coeffs, .. } =
            &restored.slots[0]
        else {
            panic!("conditional state-space family was lost");
        };
        // Regression mean for parents (4, 1) must remain 3 + 1.5*4 - 2 = 7.
        assert!((intercept + coeffs[0] * 4.0 + coeffs[1] - 7.0).abs() < 1e-12);
    }

    /// A basis mechanism evaluates at an unseen covariate cell from the decoded
    /// slot alone: every standardization and knot constant must survive the wire.
    #[test]
    fn basis_slots_roundtrip_and_evaluate_identically_at_unseen_cells() {
        use antecedent_model::{MechanismWorkspace, ParentBatch, evaluate_column};
        let knots: Arc<[Arc<[f64]>]> =
            Arc::from(vec![Arc::from([]) as Arc<[f64]>, Arc::from([-0.4, 0.1, 0.6])]);
        let basis = ParentBasis::spline_interactions(
            2,
            Arc::from([0.5, 2020.0]),
            Arc::from([0.5, 3.2]),
            knots,
        )
        .unwrap();
        let t = basis.n_terms();
        let coeffs: Vec<f64> =
            (1..=t).map(|i| 0.1 * f64::from(u32::try_from(i).unwrap())).collect();
        let logits: Vec<f64> = (0..2 * (t + 1)).map(|i| if i <= t { 0.0 } else { 0.05 }).collect();
        let store = CompiledMechanismStore {
            slots: Arc::from([
                MechanismSlot::LinearBasis {
                    intercept: 0.3,
                    basis: basis.clone(),
                    coeffs: Arc::from(coeffs),
                    sigma: 0.9,
                },
                MechanismSlot::DiscreteBasis {
                    support: Arc::from([0.0, 1.0]),
                    probs: Arc::from([0.4, 0.6]),
                    basis,
                    logit_coeffs: Arc::from(logits),
                },
            ]),
        };
        let wire = mechanisms_to_wire(&store).unwrap();
        let bytes = crate::to_cbor(&wire).unwrap();
        let decoded: MechanismStoreWire = crate::from_cbor(&bytes).unwrap();
        let restored = mechanisms_from_wire(&decoded).unwrap();
        assert_eq!(mechanisms_to_wire(&restored).unwrap(), wire);
        // A covariate cell far outside anything the constants were built from.
        let parents = [1.0, 2031.0];
        let batch = ParentBatch { n_rows: 1, n_parents: 2, values: &parents };
        for i in 0..2 {
            let mut before = [0.0];
            let mut after = [0.0];
            let mut ws = MechanismWorkspace::default();
            evaluate_column(&store.slots[i], batch, &[0.25], &mut before, &mut ws).unwrap();
            evaluate_column(&restored.slots[i], batch, &[0.25], &mut after, &mut ws).unwrap();
            assert_eq!(before[0].to_bits(), after[0].to_bits(), "slot {i}");
        }
    }

    #[test]
    fn corrupted_basis_is_refused_on_decode() {
        let basis =
            ParentBasis::interactions(2, Arc::from([0.0, 0.0]), Arc::from([1.0, 1.0])).unwrap();
        let store = CompiledMechanismStore {
            slots: Arc::from([MechanismSlot::LinearBasis {
                intercept: 0.0,
                basis,
                coeffs: Arc::from([1.0, 1.0, 1.0]),
                sigma: 1.0,
            }]),
        };
        let mut wire = mechanisms_to_wire(&store).unwrap();
        let MechanismSlotWire::LinearBasis { basis, .. } = &mut wire.slots[0] else {
            panic!("basis slot expected");
        };
        basis.scales[1] = 0.0;
        assert!(matches!(mechanisms_from_wire(&wire), Err(IoError::Convert(_))));
    }

    #[test]
    fn dynamic_slot_refuses_serialization() {
        let store = CompiledMechanismStore {
            slots: Arc::from([MechanismSlot::Dynamic {
                id: Arc::from("y"),
                mechanism: Arc::new(StubMech),
            }]),
        };
        let err = mechanisms_to_wire(&store).unwrap_err();
        assert!(matches!(err, IoError::Convert(_)));
    }
}
