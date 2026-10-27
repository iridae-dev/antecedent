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
pub const PAG_VALIDATION_SALT: u64 = 0xDE3D_61A0_4092_7214;

pub const PAG_CONFIDENCE_SET_VALIDATION_DOMAIN: &str =
    "release_2_4_1_pag_confidence_set_validation_v1";
/// SHA256 of the confidence-set validation domain, first eight bytes little-endian.
pub const PAG_CONFIDENCE_SET_VALIDATION_SALT: u64 = 0x6921_1267_88F4_5DC5;

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
            PAG_CONFIDENCE_SET_VALIDATION_DOMAIN => Some(PAG_CONFIDENCE_SET_VALIDATION_SALT),
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
pub(super) struct CellRegistry {
    cells: std::collections::BTreeMap<String, CellLooks>,
}

type CellLooks = std::collections::BTreeMap<u32, (u32, u32, u64, bool)>;

impl CellRegistry {
    pub(super) fn register(
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
