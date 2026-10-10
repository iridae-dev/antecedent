//! Binding and independent sampling shared only by closed-candidate measurement suites.
#![allow(dead_code)]
use super::calibration::{Construction, CoverageTally, ScopeFacts};
use antecedent_core::CalibrationBasis;

/// Read the candidate engine's own construction; no hand-declared test key.
pub fn bind(tally: &mut CoverageTally, basis: &CalibrationBasis) {
    tally.bind(
        &Construction {
            query: basis.query.to_string(),
            graph_class: basis.graph_class.to_string(),
            structure: basis.structure.to_string(),
            modality: basis.modality.to_string(),
            inference: basis.inference.to_string(),
            estimator: basis.estimator.to_string(),
            interval_method: basis.interval_method.to_string(),
            se_kind: basis.se_kind.to_string(),
            dependence: basis.dependence.to_string(),
            posterior: basis.posterior.to_string(),
            functional: basis.functional.to_string(),
            identification: basis.identification.to_string(),
            reported_level: basis.level,
        },
        ScopeFacts {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
    );
}

/// Independent generator: production samplers and multinomial probabilities are not reused.
pub struct Generator(u64);
impl Generator {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[allow(clippy::cast_precision_loss, reason = "upper 53 bits are exact in f64")]
    pub fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
    pub fn normal(&mut self) -> f64 {
        let u = self.uniform();
        (-2. * u.ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
    pub fn binary(&mut self, probability: f64) -> usize {
        usize::from(self.uniform() < probability)
    }
}
