//! Versioned family-wise Monte Carlo coverage decisions.
//!
//! A source-fixed family allocates one Bonferroni budget per level/label/grid
//! cell, then spends that budget over three predetermined attempted-replicate
//! looks. Exact binomial tails test compatibility with nominal coverage;
//! non-rejection does not positively certify equality or a practical margin.
//! Reporting floors, skip caps, width checks and named boundaries are separate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

pub const ENV: &str = "ANTECEDENT_CALIBRATION_DECISION_CONFIG";
pub const ID: &str = "suite_coverage_fwer.v1";
pub const VALIDATION_DOMAIN: &str = "release_2_4_1_validation_v1";
/// SHA256 of the UTF-8 validation domain, first eight bytes interpreted little-endian.
pub const VALIDATION_SALT: u64 = 0xE1FC_1261_2A1E_C540;
pub const HC3_VALIDATION_DOMAIN: &str = "release_2_4_1_weighted_hc3_validation_v1";
/// SHA256 of the HC3 validation domain, first eight bytes interpreted little-endian.
pub const HC3_VALIDATION_SALT: u64 = 0xDD5E_E217_1F1D_846C;

pub const PAG_VALIDATION_DOMAIN: &str = "release_2_4_1_pag_trial_shift_validation_v1";
/// SHA256 of the PAG validation domain, first eight bytes interpreted little-endian.
pub const PAG_VALIDATION_SALT: u64 = 0xDE3D61A040927214;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Look {
    pub planned_replicates: u32,
    pub weight: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub id: String,
    pub family_digest: String,
    pub validation_domain: String,
    pub family_alpha: f64,
    pub family_groups: u32,
    pub family_grid_points: u32,
    pub family_cells: u32,
    pub cell_cap_per_group: u32,
    pub looks: Vec<Look>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Decision {
    #[serde(flatten)]
    pub config: Config,
    pub planned_replicates: u32,
    pub look_weight: f64,
    /// Existing coverage denominator, including every skip counted as a miss.
    pub scored: u32,
    pub covered: u32,
    pub lower_only: bool,
    pub p_value: f64,
    pub compatible: bool,
}

impl Config {
    /// A registered domain fixes its random stream before any validation outcomes.
    pub fn validation_salt(&self) -> Option<u64> {
        match self.validation_domain.as_str() {
            VALIDATION_DOMAIN => Some(VALIDATION_SALT),
            HC3_VALIDATION_DOMAIN => Some(HC3_VALIDATION_SALT),
            PAG_VALIDATION_DOMAIN => Some(PAG_VALIDATION_SALT),
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.id != ID
            || self.validation_salt().is_none()
            || self.family_alpha.to_bits() != 0.05_f64.to_bits()
        {
            return Err("unknown coverage decision protocol or family risk");
        }
        if self.family_digest.len() != 64
            || !self.family_digest.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("family digest must identify the frozen source family");
        }
        if self.family_groups == 0
            || self.family_grid_points != 3
            || self.cell_cap_per_group != 256
            || self.family_groups.checked_mul(768) != Some(self.family_cells)
        {
            return Err("invalid complete-family cell budget");
        }
        let expected = [(400, 0.5_f64), (2000, 0.25), (20000, 0.25)];
        if self.looks.len() != expected.len()
            || self.looks.iter().zip(expected).any(|(look, (n, w))| {
                look.planned_replicates != n || look.weight.to_bits() != w.to_bits()
            })
        {
            return Err("unknown attempted-replicate spending sequence");
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        planned_replicates: u32,
        scored: u32,
        covered: u32,
        nominal: f64,
        lower_only: bool,
    ) -> Result<Decision, &'static str> {
        self.validate()?;
        let look = self
            .looks
            .iter()
            .find(|look| look.planned_replicates == planned_replicates)
            .ok_or("attempt budget is not a registered Monte Carlo look")?;
        if scored > planned_replicates {
            return Err("coverage denominator exceeds the attempted-replicate budget");
        }
        let p_value = binomial_p_value(scored, covered, nominal, lower_only)?;
        let alpha = self.family_alpha * look.weight / f64::from(self.family_cells);
        Ok(Decision {
            config: self.clone(),
            planned_replicates,
            look_weight: look.weight,
            scored,
            covered,
            lower_only,
            p_value,
            compatible: p_value > alpha,
        })
    }
}

/// Binomial lower-tail log probability when `k < n*p`.
/// Starting at the largest mass in this tail, sum the decreasing recurrence.
fn log_lower_direct(n: u32, k: u32, p: f64) -> f64 {
    let nf = f64::from(n);
    let kf = f64::from(k);
    let log_mass = antecedent_stats::special::ln_gamma(nf + 1.0)
        - antecedent_stats::special::ln_gamma(kf + 1.0)
        - antecedent_stats::special::ln_gamma(nf - kf + 1.0)
        + kf * p.ln()
        + (nf - kf) * (-p).ln_1p();
    let (mut term, mut sum) = (1.0, 1.0);
    for j in (1..=k).rev() {
        term *= f64::from(j) / f64::from(n - j + 1) * (1.0 - p) / p;
        sum += term;
        if term <= sum * f64::EPSILON {
            break;
        }
    }
    log_mass + sum.ln()
}

#[allow(clippy::float_cmp, reason = "exact Bernoulli point-mass endpoints")]
fn binomial_lower(n: u32, k: u32, p: f64) -> f64 {
    if k == n || p == 0.0 {
        return 1.0;
    }
    if p == 1.0 {
        return 0.0;
    }
    if f64::from(k) < f64::from(n) * p {
        log_lower_direct(n, k, p).exp().min(1.0)
    } else {
        // P(X<=k) = 1-P(n-X<=n-k-1); avoid cancellation for small tails.
        -log_lower_direct(n, n - k - 1, 1.0 - p).exp_m1()
    }
}

/// Equal-tail two-sided (or conservative lower-tail) exact binomial p-value.
/// Discreteness makes these tests conservative; no normal approximation is used.
pub fn binomial_p_value(
    n: u32,
    covered: u32,
    nominal: f64,
    lower_only: bool,
) -> Result<f64, &'static str> {
    if n == 0 || covered > n || !nominal.is_finite() || !(0.0..=1.0).contains(&nominal) {
        return Err("invalid binomial coverage counts or nominal probability");
    }
    let lower = binomial_lower(n, covered, nominal);
    if lower_only {
        return Ok(lower);
    }
    let upper = if covered == 0 { 1.0 } else { binomial_lower(n, n - covered, 1.0 - nominal) };
    Ok((2.0 * lower.min(upper)).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            id: ID.into(),
            family_digest: "a".repeat(64),
            validation_domain: VALIDATION_DOMAIN.into(),
            family_alpha: 0.05,
            family_groups: 700,
            family_grid_points: 3,
            family_cells: 700 * 768,
            cell_cap_per_group: 256,
            looks: vec![
                Look { planned_replicates: 400, weight: 0.5 },
                Look { planned_replicates: 2000, weight: 0.25 },
                Look { planned_replicates: 20000, weight: 0.25 },
            ],
        }
    }

    #[test]
    fn registered_domains_keep_distinct_fixed_streams_and_unknown_domains_fail_closed() {
        let mut c = config();
        assert_eq!(c.validation_salt(), Some(0xE1FC_1261_2A1E_C540));
        assert!(c.validate().is_ok());
        c.validation_domain = HC3_VALIDATION_DOMAIN.into();
        assert_eq!(c.validation_salt(), Some(0xDD5E_E217_1F1D_846C));
        assert!(c.validate().is_ok());
        assert_ne!(VALIDATION_SALT, HC3_VALIDATION_SALT);
        c.validation_domain = PAG_VALIDATION_DOMAIN.into();
        assert_eq!(c.validation_salt(), Some(0xDE3D61A040927214));
        assert!(c.validate().is_ok());
        assert_ne!(PAG_VALIDATION_SALT, VALIDATION_SALT);
        assert_ne!(PAG_VALIDATION_SALT, HC3_VALIDATION_SALT);
        for unknown in ["", "release_2_4_1_weighted_hc3_validation_v2", "arbitrary_retry"] {
            c.validation_domain = unknown.into();
            assert_eq!(c.validation_salt(), None);
            assert!(c.validate().is_err());
        }
    }

    #[test]
    fn binomial_tails_match_independent_integer_enumeration() {
        for n in 1..=20_u32 {
            let mut masses = Vec::new();
            for k in 0..=n {
                let choose = (0..k).fold(1_u128, |v, j| v * u128::from(n - j) / u128::from(j + 1));
                // Exact integer numerator for Binomial(n,19/20).
                masses.push(choose * 19_u128.pow(k));
            }
            for k in 0..=n {
                #[allow(clippy::cast_precision_loss, reason = "bounded integer probability oracle")]
                let lower =
                    masses[..=k as usize].iter().sum::<u128>() as f64 / 20_u128.pow(n) as f64;
                #[allow(clippy::cast_precision_loss, reason = "bounded integer probability oracle")]
                let upper =
                    masses[k as usize..].iter().sum::<u128>() as f64 / 20_u128.pow(n) as f64;
                let got = binomial_p_value(n, k, 0.95, true).unwrap();
                assert!((got - lower).abs() < 1e-12, "n={n} k={k} {got} {lower}");
                let got = binomial_p_value(n, k, 0.95, false).unwrap();
                assert!((got - (2.0 * lower.min(upper)).min(1.0)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn tiny_prefix_misses_are_distinct_from_clear_miscalibration() {
        let c = config();
        assert!(c.evaluate(20000, 20000, 18926, 0.95, false).unwrap().compatible);
        assert!(!c.evaluate(20000, 20000, 17200, 0.95, false).unwrap().compatible);
        assert!(!c.evaluate(20000, 20000, 19900, 0.95, false).unwrap().compatible);
        assert!(c.evaluate(20000, 20000, 19900, 0.95, true).unwrap().compatible);
    }

    #[test]
    fn reported_denominator_does_not_choose_the_spending_look() {
        let d = config().evaluate(2000, 1953, 1829, 0.95, false).unwrap();
        assert_eq!(d.planned_replicates, 2000);
        assert_eq!(d.scored, 1953);
        assert_eq!(d.look_weight.to_bits(), 0.25_f64.to_bits());
        assert!(d.compatible);
        assert!(config().evaluate(1953, 1953, 1829, 0.95, false).is_err());
    }

    #[test]
    fn one_cell_cannot_test_conflicting_prefixes_at_the_same_look() {
        let mut registry = CellRegistry::default();
        let c = config();
        let d = c.evaluate(2000, 2000, 1900, 0.95, false).unwrap();
        assert!(registry.register("fixed cell", &d, 0.95).is_ok());
        assert!(registry.register("fixed cell", &d, 0.95).is_ok());
        let changed = c.evaluate(2000, 1999, 1900, 0.95, false).unwrap();
        assert!(registry.register("fixed cell", &changed, 0.95).is_err());
        assert!(registry.register("fixed cell", &d, 0.9).is_err());
        let later = c.evaluate(20000, 20000, 19000, 0.95, false).unwrap();
        assert!(registry.register("fixed cell", &later, 0.95).is_ok());
        for index in 1..256 {
            assert!(registry.register(&format!("fixed cell {index}"), &d, 0.95).is_ok());
        }
        assert!(registry.register("extra cell", &d, 0.95).is_err());
    }

    #[test]
    fn malformed_families_and_unregistered_looks_fail_closed() {
        let c = config();
        assert!(c.evaluate(3000, 2000, 1900, 0.95, false).is_err());
        assert!(c.evaluate(2000, 2001, 1900, 0.95, false).is_err());
        for field in 0..5 {
            let mut bad = c.clone();
            match field {
                0 => bad.family_cells -= 1,
                1 => bad.cell_cap_per_group = 257,
                2 => bad.looks[0].weight = 1.0,
                3 => bad.family_digest = "x".repeat(64),
                _ => bad.family_alpha = 0.1,
            }
            assert!(bad.validate().is_err());
        }
        assert!(binomial_p_value(0, 0, 0.95, false).is_err());
        assert!(binomial_p_value(20, 21, 0.95, false).is_err());
    }
}

/// The runner supplies an immutable complete-family declaration. No declaration
/// means the historical per-cell decision protocol, including during old replay.
pub fn from_environment() -> Option<&'static Config> {
    static CONFIG: std::sync::OnceLock<Option<Config>> = std::sync::OnceLock::new();
    CONFIG
        .get_or_init(|| match std::env::var(ENV) {
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => panic!("{ENV}: {error}"),
            Ok(raw) => {
                let config: Config =
                    serde_json::from_str(&raw).unwrap_or_else(|error| panic!("{ENV}: {error}"));
                config.validate().unwrap_or_else(|error| panic!("{ENV}: {error}"));
                Some(config)
            }
        })
        .as_ref()
}

/// Source-audited cells and their registered looks. A repeated evaluation at
/// the same look must use identical counts and semantics; otherwise it would
/// spend risk on unregistered intermediate prefixes.
#[derive(Default)]
struct CellRegistry {
    cells: std::collections::BTreeMap<String, CellLooks>,
}

type CellLooks = std::collections::BTreeMap<u32, (u32, u32, u64, bool)>;

impl CellRegistry {
    fn register(
        &mut self,
        key: &str,
        decision: &Decision,
        nominal: f64,
    ) -> Result<(), &'static str> {
        let observation =
            (decision.scored, decision.covered, nominal.to_bits(), decision.lower_only);
        let looks = self.cells.entry(key.to_owned()).or_default();
        if let Some(previous) = looks.get(&decision.planned_replicates) {
            if *previous != observation {
                return Err("conflicting coverage prefixes or semantics at one registered look");
            }
        } else {
            looks.insert(decision.planned_replicates, observation);
        }
        if self.cells.len()
            > usize::try_from(decision.config.cell_cap_per_group).expect("bounded cap")
        {
            return Err("coverage group exceeded its predeclared hypothesis budget");
        }
        Ok(())
    }
}

/// The runtime cap cannot establish that hypothesis identities were fixed
/// before outcomes. The separately frozen family source audit must do that.
pub fn register_cell(key: &str, decision: &Decision, nominal: f64) {
    static CELLS: std::sync::OnceLock<std::sync::Mutex<CellRegistry>> = std::sync::OnceLock::new();
    CELLS
        .get_or_init(Default::default)
        .lock()
        .expect("coverage cell registry")
        .register(key, decision, nominal)
        .unwrap_or_else(|error| panic!("{error}"));
}
