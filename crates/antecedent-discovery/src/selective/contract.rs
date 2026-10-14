//! Frozen discovery contracts: one for static (tabular) discovery and one for temporal
//! (lagged) discovery.
//!
//! A contract is everything about a run except the data: the variables and their search
//! order, the conditional-independence test and its settings, the mask type, the threshold
//! and multiplicity control, the search and orientation settings, the background knowledge,
//! the lag bounds and the random seed. Its digests are the declared inputs of the stages in
//! [`super::stages`]. The static and temporal contracts are separate types because their
//! inputs differ (a temporal run depends on lag bounds, the lagged frame depth and the
//! sampling index; a static run on neither) and a value of one cannot be passed for the
//! other.
//!
//! # What a contract does not state
//!
//! No contract field claims recovery of the true graph, a false-discovery or type-I error
//! rate for the output graph, or an interval for a downstream estimate. Those claims need a
//! calibration of the complete discovery-plus-analysis method that is not part of this
//! module; see [`crate::selective`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{CanonicalWriter, KernelPolicy, VariableId};
use antecedent_stats::{
    ConditionalIndependenceTest, FdrAdjustment, MultipleTestingMethod, SignificanceMethod,
};

use crate::constraints::{CiMaskType, DiscoveryConstraints};
use crate::discriminating_paths::DiscriminatingPathBudget;
use crate::fci::Fci;
use crate::pc::Pc;
use crate::pcmci::Pcmci;
use crate::pcmci_plus::PcmciPlus;
use crate::result::LaggedLink;

use super::identity::{DataIdentity, Digest, SELECTIVE_IDENTITY_VERSION};
use super::refusal::SelectiveDiscoveryError;
use super::stages::StageInputs;

fn invalid(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        remedy,
    )
}

fn not_supported(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("route_not_supported"),
        detail,
        remedy,
    )
}

const COMPONENT_TAG: &str = "antecedent.selective_discovery.component.v1";
const CONTRACT_TAG: &str = "antecedent.selective_discovery.contract.v1";

/// A conditional-independence test as a contract declares it.
#[derive(Clone)]
pub enum CiTestSpec {
    /// A test resolved by its registered name.
    Named(String),
    /// A caller-supplied test with the identity the caller vouches for.
    ///
    /// The identity must name the test and every setting that changes a result (a
    /// permutation count, a salt, a bandwidth, supplied weights). Two different tests must
    /// not share one: a retained statistic is served on identity equality.
    Declared {
        /// Identity of the test and its settings.
        identity: String,
        /// The test.
        test: Arc<dyn ConditionalIndependenceTest + Send + Sync>,
    },
}

impl std::fmt::Debug for CiTestSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Named(name) => f.debug_tuple("Named").field(name).finish(),
            Self::Declared { identity, .. } => {
                f.debug_struct("Declared").field("identity", identity).finish_non_exhaustive()
            }
        }
    }
}

impl CiTestSpec {
    /// A named test.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self::Named(name.into())
    }

    /// A declared test.
    #[must_use]
    pub fn declared(
        identity: impl Into<String>,
        test: Arc<dyn ConditionalIndependenceTest + Send + Sync>,
    ) -> Self {
        Self::Declared { identity: identity.into(), test }
    }

    /// The test and its identity text.
    ///
    /// # Errors
    ///
    /// An unknown name or a blank declared identity.
    pub fn resolve(
        &self,
    ) -> Result<(Arc<dyn ConditionalIndependenceTest + Send + Sync>, String), SelectiveDiscoveryError>
    {
        match self {
            Self::Named(name) => {
                let key = name.trim().to_ascii_lowercase();
                let test = antecedent_stats::ci_from_name(&key)
                    .map_err(crate::error::DiscoveryError::from)?;
                Ok((test, format!("named:{key}")))
            }
            Self::Declared { identity, test } => {
                if identity.trim().is_empty() {
                    return Err(invalid(
                        "selective_discovery.test_identity_missing",
                        "declare an identity naming the test and every setting that changes \
                         its result",
                    ));
                }
                Ok((Arc::clone(test), format!("declared:{}", identity.trim())))
            }
        }
    }
}

/// Algorithm of a static contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StaticAlgorithm {
    /// PC → CPDAG.
    Pc,
    /// FCI → PAG (latent confounding allowed).
    Fci,
}

impl StaticAlgorithm {
    /// Stable route label.
    #[must_use]
    pub const fn route(self) -> &'static str {
        match self {
            Self::Pc => "static_pc",
            Self::Fci => "static_fci",
        }
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Pc => 0,
            Self::Fci => 1,
        }
    }
}

/// Algorithm of a temporal contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TemporalAlgorithm {
    /// PCMCI: lagged links only → temporal DAG.
    Pcmci,
    /// PCMCI+: contemporaneous and lagged links → temporal CPDAG.
    PcmciPlus,
}

impl TemporalAlgorithm {
    /// Stable route label.
    #[must_use]
    pub const fn route(self) -> &'static str {
        match self {
            Self::Pcmci => "temporal_pcmci",
            Self::PcmciPlus => "temporal_pcmci_plus",
        }
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Pcmci => 10,
            Self::PcmciPlus => 11,
        }
    }
}

fn component(name: &str, fill: impl FnOnce(&mut CanonicalWriter)) -> Digest {
    let mut writer = CanonicalWriter::new(COMPONENT_TAG);
    writer.word(SELECTIVE_IDENTITY_VERSION).text(name);
    fill(&mut writer);
    writer.finish()
}

fn write_link(writer: &mut CanonicalWriter, link: &LaggedLink) {
    writer
        .word(u64::from(link.source.raw()))
        .word(u64::from(link.source_lag.raw()))
        .word(u64::from(link.target.raw()))
        .word(u64::from(link.target_lag.raw()));
}

const fn mask_tag(mask: CiMaskType) -> u64 {
    match mask {
        CiMaskType::Y => 0,
        CiMaskType::X => 1,
        CiMaskType::Z => 2,
        CiMaskType::Yx => 3,
        CiMaskType::Yz => 4,
        CiMaskType::Xz => 5,
        CiMaskType::Yxz => 6,
    }
}

const fn method_tag(method: MultipleTestingMethod) -> u64 {
    match method {
        MultipleTestingMethod::BenjaminiHochberg => 0,
        MultipleTestingMethod::BenjaminiYekutieli => 1,
        MultipleTestingMethod::Bonferroni => 2,
        MultipleTestingMethod::Holm => 3,
    }
}

fn write_significance(writer: &mut CanonicalWriter, significance: SignificanceMethod) {
    match significance {
        SignificanceMethod::Analytic => {
            writer.word(0);
        }
        SignificanceMethod::BlockShuffle { replicates, block_size } => {
            writer.word(1).word(u64::from(replicates)).word(block_size as u64);
        }
    }
}

fn sorted_links(links: &[LaggedLink]) -> Vec<LaggedLink> {
    let mut out = links.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

/// Settings shared by the two contract types, as the digests the stages declare.
struct Settings<'a> {
    route: &'static str,
    algorithm_tag: u64,
    variables: &'a [VariableId],
    constraints: &'a DiscoveryConstraints,
    ci_identity: &'a str,
    fdr: Option<FdrAdjustment>,
    population: &'a str,
    rng_seed: u64,
    frame_depth: u32,
    /// Extra words of the search limits (Possible-D-Sep nodes, lag bounds).
    search_limits: Vec<u64>,
    /// Words of the orientation settings.
    orientation: Vec<u64>,
}

#[allow(
    clippy::too_many_lines,
    reason = "one declaration of every stage input; each component is a short labelled block"
)]
fn stage_inputs(
    settings: &Settings<'_>,
    data: &DataIdentity,
    kernel: KernelPolicy,
) -> Result<StageInputs, SelectiveDiscoveryError> {
    let c = settings.constraints;
    let mut components: BTreeMap<&'static str, Digest> = BTreeMap::new();
    components.insert("data", data.digest);
    components.insert(
        "variables",
        component("variables", |w| {
            w.word(settings.variables.len() as u64);
            for variable in settings.variables {
                w.word(u64::from(variable.raw()));
            }
        }),
    );
    components.insert(
        "population",
        component("population", |w| {
            w.text(settings.population);
        }),
    );
    components.insert(
        "ci_test",
        component("ci_test", |w| {
            w.text(settings.ci_identity);
        }),
    );
    components.insert(
        "significance",
        component("significance", |w| {
            write_significance(w, c.significance);
        }),
    );
    components.insert(
        "rng",
        component("rng", |w| {
            w.word(settings.rng_seed);
        }),
    );
    components.insert(
        "kernel_policy",
        component("kernel_policy", |w| {
            w.word(u64::from(kernel.allow_portable_optimized))
                .word(u64::from(kernel.allow_arch_simd))
                .word(u64::from(kernel.force_scalar));
        }),
    );
    components.insert(
        "mask",
        component("mask", |w| {
            w.word(mask_tag(c.mask_type));
        }),
    );
    components.insert(
        "frame_depth",
        component("frame_depth", |w| {
            w.word(u64::from(settings.frame_depth));
        }),
    );
    components.insert(
        "algorithm",
        component("algorithm", |w| {
            w.word(settings.algorithm_tag);
        }),
    );
    components.insert(
        "thresholds",
        component("thresholds", |w| {
            w.real(c.alpha);
        }),
    );
    components.insert(
        "multiplicity",
        component("multiplicity", |w| match settings.fdr {
            None => {
                w.word(0);
            }
            Some(fdr) => {
                w.word(1).word(method_tag(fdr.method)).word(u64::from(fdr.exclude_contemporaneous));
            }
        }),
    );
    components.insert(
        "background_knowledge",
        component("background_knowledge", |w| {
            let required = sorted_links(&c.required);
            w.word(required.len() as u64);
            for link in &required {
                write_link(w, link);
            }
            let forbidden = sorted_links(&c.forbidden);
            w.word(forbidden.len() as u64);
            for link in &forbidden {
                write_link(w, link);
            }
            w.word(c.tiers.len() as u64);
            for tier in c.tiers.iter() {
                let members: BTreeSet<u32> = tier.iter().map(|v| v.raw()).collect();
                w.word(members.len() as u64);
                for member in members {
                    w.word(u64::from(member));
                }
            }
            let groups = c.vector_groups.groups();
            w.word(groups.len() as u64);
            for group in groups {
                w.word(group.len() as u64);
                for v in group.iter() {
                    w.word(u64::from(v.raw()));
                }
            }
        }),
    );
    components.insert(
        "search_limits",
        component("search_limits", |w| {
            w.word(c.max_cond_size as u64).opt_word(c.max_parents.map(|p| p as u64));
            w.word(u64::from(c.temporal.min_lag.raw())).word(u64::from(c.temporal.max_lag.raw()));
            w.word(settings.search_limits.len() as u64);
            for word in &settings.search_limits {
                w.word(*word);
            }
        }),
    );
    components.insert(
        "orientation_settings",
        component("orientation_settings", |w| {
            w.word(settings.orientation.len() as u64);
            for word in &settings.orientation {
                w.word(*word);
            }
        }),
    );
    StageInputs::new(settings.route, components).ok_or_else(|| {
        invalid(
            "selective_discovery.invalid_contract",
            "the contract did not declare every stage input",
        )
    })
}

fn validate_common(
    variables: &[VariableId],
    constraints: &DiscoveryConstraints,
    population: &str,
    additional: &[String],
) -> Result<(), SelectiveDiscoveryError> {
    if !additional.is_empty() {
        return Err(not_supported(
            "selective_discovery.multi_population",
            "discovery from several source datasets is not part of selective discovery: a \
             pooled or per-population search has its own identification and invariance \
             assumptions. Run discovery on one declared population, or declare the \
             cross-population assumptions under the transport contract",
        ));
    }
    let md = &constraints.multi_dataset;
    if !md.context_variables.is_empty()
        || !md.space_dummy_variables.is_empty()
        || !md.time_dummy_variables.is_empty()
        || md.include_time_dummy
    {
        return Err(not_supported(
            "selective_discovery.multi_population",
            "context variables and dataset dummies describe a multi-dataset search, which \
             this adapter does not retain statistics for; remove them or use the pooled \
             algorithm directly",
        ));
    }
    if population.trim().is_empty() {
        return Err(invalid(
            "selective_discovery.population_missing",
            "declare the one population the data were drawn from",
        ));
    }
    if variables.is_empty() {
        return Err(invalid(
            "selective_discovery.invalid_contract",
            "declare at least one searched variable",
        ));
    }
    let distinct: BTreeSet<u32> = variables.iter().map(|v| v.raw()).collect();
    if distinct.len() != variables.len() {
        return Err(invalid(
            "selective_discovery.invalid_contract",
            "the searched variables repeat",
        ));
    }
    if !(constraints.alpha > 0.0 && constraints.alpha < 1.0) {
        return Err(invalid(
            "selective_discovery.invalid_contract",
            "the significance threshold must lie strictly between 0 and 1",
        ));
    }
    constraints.validate()?;
    Ok(())
}

/// Identity of a contract from its route and declared inputs, in canonical name order.
///
/// The data and the kernel policy are inputs of a run, not of the contract, and are left out.
/// `components` must be in ascending name order (as [`StageInputs::components`] yields them).
pub(crate) fn contract_identity_of<'a, 'b>(
    route: &str,
    components: impl IntoIterator<Item = (&'a str, &'b Digest)>,
) -> Digest {
    let mut writer = CanonicalWriter::new(CONTRACT_TAG);
    writer.word(SELECTIVE_IDENTITY_VERSION).text(route);
    for (name, digest) in components {
        if name != "data" && name != "kernel_policy" {
            writer.text(name).bytes(digest);
        }
    }
    writer.finish()
}

fn contract_digest(inputs: &StageInputs) -> Digest {
    contract_identity_of(inputs.route(), inputs.components())
}

/// Frozen contract of a static discovery run over a table.
#[derive(Clone, Debug)]
pub struct StaticDiscoveryContract {
    /// Algorithm.
    pub algorithm: StaticAlgorithm,
    /// Searched variables in search order. The order is part of the contract: PC is
    /// order-dependent and test coordinates are column positions.
    pub variables: Vec<VariableId>,
    /// Background knowledge, conditioning-set limit, threshold, significance method and
    /// mask type.
    pub constraints: DiscoveryConstraints,
    /// The conditional-independence test.
    pub ci_test: CiTestSpec,
    /// Multiple-testing adjustment (`None` = off). FCI refuses a configured adjustment.
    pub fdr: Option<FdrAdjustment>,
    /// Possible-D-Sep expansion budget (FCI).
    pub possible_d_sep_max_nodes: usize,
    /// Discriminating-path search budget (FCI orientation).
    pub discriminating_path_budget: DiscriminatingPathBudget,
    /// The one population the rows were drawn from.
    pub population: String,
    /// Further populations. Must be empty: multi-population discovery is closed.
    pub additional_populations: Vec<String>,
    /// Master seed of the execution context the run uses.
    pub rng_seed: u64,
}

impl StaticDiscoveryContract {
    /// A contract with the algorithm's own defaults.
    #[must_use]
    pub fn new(
        algorithm: StaticAlgorithm,
        variables: Vec<VariableId>,
        ci_test: CiTestSpec,
        rng_seed: u64,
    ) -> Self {
        let (constraints, fdr, pds, budget) = match algorithm {
            StaticAlgorithm::Pc => {
                let pc = Pc::new();
                (
                    pc.constraints,
                    pc.fdr,
                    Fci::new().pds_max_nodes,
                    DiscriminatingPathBudget::default(),
                )
            }
            StaticAlgorithm::Fci => {
                let fci = Fci::new();
                (fci.constraints, fci.fdr, fci.pds_max_nodes, fci.discriminating_path_budget)
            }
        };
        Self {
            algorithm,
            variables,
            constraints,
            ci_test,
            fdr,
            possible_d_sep_max_nodes: pds,
            discriminating_path_budget: budget,
            population: "population".to_owned(),
            additional_populations: Vec::new(),
            rng_seed,
        }
    }

    /// Check the declaration without data.
    ///
    /// # Errors
    ///
    /// A closed route (several populations, multi-dataset constraints), a blank population,
    /// repeated or no variables, a threshold outside `(0, 1)` or inconsistent constraints.
    pub fn validate(&self) -> Result<(), SelectiveDiscoveryError> {
        validate_common(
            &self.variables,
            &self.constraints,
            &self.population,
            &self.additional_populations,
        )
    }

    /// The declared stage inputs for `data` under `kernel`.
    ///
    /// # Errors
    ///
    /// A declaration that [`Self::validate`] refuses, or an unresolvable test.
    pub fn stage_inputs(
        &self,
        data: &DataIdentity,
        kernel: KernelPolicy,
    ) -> Result<StageInputs, SelectiveDiscoveryError> {
        self.validate()?;
        let (_, ci_identity) = self.ci_test.resolve()?;
        let budget = self.discriminating_path_budget;
        let settings = Settings {
            route: self.algorithm.route(),
            algorithm_tag: self.algorithm.tag(),
            variables: &self.variables,
            constraints: &self.constraints,
            ci_identity: &ci_identity,
            fdr: self.fdr,
            population: &self.population,
            rng_seed: self.rng_seed,
            frame_depth: 0,
            search_limits: vec![self.possible_d_sep_max_nodes as u64],
            orientation: vec![budget.max_paths_per_edge as u64, budget.max_len as u64],
        };
        stage_inputs(&settings, data, kernel)
    }

    /// Identity of the contract itself: everything but the data and the kernel policy.
    ///
    /// # Errors
    ///
    /// As [`Self::stage_inputs`].
    pub fn identity(&self) -> Result<Digest, SelectiveDiscoveryError> {
        let placeholder = DataIdentity { rows: 0, columns: Vec::new(), digest: [0; 32] };
        let inputs = self.stage_inputs(&placeholder, KernelPolicy::scalar_only())?;
        Ok(contract_digest(&inputs))
    }
}

/// Frozen contract of a temporal discovery run over a time series.
#[derive(Clone, Debug)]
pub struct TemporalDiscoveryContract {
    /// Algorithm.
    pub algorithm: TemporalAlgorithm,
    /// Searched variables in search order.
    pub variables: Vec<VariableId>,
    /// Background knowledge, lag bounds, conditioning-set limit, threshold, significance
    /// method and mask type.
    pub constraints: DiscoveryConstraints,
    /// The conditional-independence test.
    pub ci_test: CiTestSpec,
    /// Multiple-testing adjustment over the MCI family (`None` = off).
    pub fdr: Option<FdrAdjustment>,
    /// The one population the series was drawn from.
    pub population: String,
    /// Further populations. Must be empty: multi-population discovery is closed.
    pub additional_populations: Vec<String>,
    /// Master seed of the execution context the run uses.
    pub rng_seed: u64,
}

impl TemporalDiscoveryContract {
    /// A contract with the algorithm's own defaults.
    #[must_use]
    pub fn new(
        algorithm: TemporalAlgorithm,
        variables: Vec<VariableId>,
        ci_test: CiTestSpec,
        rng_seed: u64,
    ) -> Self {
        let (constraints, fdr) = match algorithm {
            TemporalAlgorithm::Pcmci => {
                let alg = Pcmci::new();
                (alg.engine().constraints.clone(), alg.fdr)
            }
            TemporalAlgorithm::PcmciPlus => {
                let alg = PcmciPlus::new();
                (alg.engine().constraints.clone(), alg.fdr)
            }
        };
        Self {
            algorithm,
            variables,
            constraints,
            ci_test,
            fdr,
            population: "population".to_owned(),
            additional_populations: Vec::new(),
            rng_seed,
        }
    }

    /// Check the declaration without data.
    ///
    /// # Errors
    ///
    /// As [`StaticDiscoveryContract::validate`], plus a lag window that is empty, or a PCMCI
    /// request for contemporaneous links.
    pub fn validate(&self) -> Result<(), SelectiveDiscoveryError> {
        validate_common(
            &self.variables,
            &self.constraints,
            &self.population,
            &self.additional_populations,
        )?;
        let t = &self.constraints.temporal;
        if t.max_lag.raw() == 0 || t.min_lag.raw() > t.max_lag.raw() {
            return Err(invalid(
                "selective_discovery.invalid_contract",
                "the lag window must satisfy 0 <= min_lag <= max_lag with max_lag >= 1",
            ));
        }
        if self.algorithm == TemporalAlgorithm::Pcmci && t.min_lag.raw() == 0 {
            return Err(not_supported(
                "selective_discovery.contemporaneous_links",
                "PCMCI searches lagged links only; request contemporaneous links with PCMCI+, \
                 which orients them up to an equivalence class",
            ));
        }
        Ok(())
    }

    /// The declared stage inputs for `data` under `kernel`.
    ///
    /// # Errors
    ///
    /// A declaration that [`Self::validate`] refuses, or an unresolvable test.
    pub fn stage_inputs(
        &self,
        data: &DataIdentity,
        kernel: KernelPolicy,
    ) -> Result<StageInputs, SelectiveDiscoveryError> {
        self.validate()?;
        let (_, ci_identity) = self.ci_test.resolve()?;
        let settings = Settings {
            route: self.algorithm.route(),
            algorithm_tag: self.algorithm.tag(),
            variables: &self.variables,
            constraints: &self.constraints,
            ci_identity: &ci_identity,
            fdr: self.fdr,
            population: &self.population,
            rng_seed: self.rng_seed,
            frame_depth: 2 * self.constraints.temporal.max_lag.raw(),
            search_limits: Vec::new(),
            orientation: Vec::new(),
        };
        stage_inputs(&settings, data, kernel)
    }

    /// Identity of the contract itself: everything but the data and the kernel policy.
    ///
    /// # Errors
    ///
    /// As [`Self::stage_inputs`].
    pub fn identity(&self) -> Result<Digest, SelectiveDiscoveryError> {
        let placeholder = DataIdentity { rows: 0, columns: Vec::new(), digest: [0; 32] };
        let inputs = self.stage_inputs(&placeholder, KernelPolicy::scalar_only())?;
        Ok(contract_digest(&inputs))
    }
}

pub(crate) fn inputs_contract_identity(inputs: &StageInputs) -> Digest {
    contract_digest(inputs)
}

#[cfg(test)]
mod tests {
    use antecedent_core::Lag;

    use super::*;

    fn variables(n: u32) -> Vec<VariableId> {
        (0..n).map(VariableId::from_raw).collect()
    }

    fn data(tag: u8) -> DataIdentity {
        DataIdentity { rows: 10, columns: Vec::new(), digest: [tag; 32] }
    }

    #[test]
    fn contract_identity_excludes_data_and_sees_every_setting() {
        let base = StaticDiscoveryContract::new(
            StaticAlgorithm::Pc,
            variables(3),
            CiTestSpec::named("parcorr"),
            7,
        );
        let id = base.identity().unwrap();
        let mut permuted = base.clone();
        permuted.variables =
            vec![VariableId::from_raw(2), VariableId::from_raw(0), VariableId::from_raw(1)];
        assert_ne!(id, permuted.identity().unwrap());
        let mut alpha = base.clone();
        alpha.constraints.alpha = 0.01;
        assert_ne!(id, alpha.identity().unwrap());
        let mut seed = base.clone();
        seed.rng_seed = 8;
        assert_ne!(id, seed.identity().unwrap());
        let mut test = base.clone();
        test.ci_test = CiTestSpec::named("gsquared");
        assert_ne!(id, test.identity().unwrap());
        // The data digest is a stage input, not part of the contract identity.
        let kernel = KernelPolicy::scalar_only();
        let a = base.stage_inputs(&data(1), kernel).unwrap();
        let b = base.stage_inputs(&data(2), kernel).unwrap();
        assert_ne!(a, b);
        assert_eq!(id, base.identity().unwrap());
    }

    #[test]
    fn required_and_forbidden_order_does_not_change_the_identity() {
        let link = |s: u32, t: u32| LaggedLink {
            source: VariableId::from_raw(s),
            source_lag: Lag::CONTEMPORANEOUS,
            target: VariableId::from_raw(t),
            target_lag: Lag::CONTEMPORANEOUS,
        };
        let mut a = StaticDiscoveryContract::new(
            StaticAlgorithm::Pc,
            variables(3),
            CiTestSpec::named("parcorr"),
            1,
        );
        let mut b = a.clone();
        a.constraints.forbidden = Arc::from([link(0, 1), link(1, 2)]);
        b.constraints.forbidden = Arc::from([link(1, 2), link(0, 1)]);
        assert_eq!(a.identity().unwrap(), b.identity().unwrap());
        b.constraints.forbidden = Arc::from([link(0, 1)]);
        assert_ne!(a.identity().unwrap(), b.identity().unwrap());
    }

    #[test]
    fn multi_population_and_context_requests_are_refused_with_a_remedy() {
        let mut contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Pc,
            variables(2),
            CiTestSpec::named("parcorr"),
            1,
        );
        contract.additional_populations = vec!["other".to_owned()];
        let err = contract.validate().unwrap_err();
        assert_eq!(err.detail(), Some("selective_discovery.multi_population"));
        assert_eq!(err.code(), Some("route_not_supported"));
        contract.additional_populations.clear();
        contract.constraints.multi_dataset.context_variables = Arc::from([VariableId::from_raw(1)]);
        assert_eq!(
            contract.validate().unwrap_err().detail(),
            Some("selective_discovery.multi_population")
        );
    }

    #[test]
    fn temporal_contract_refuses_pcmci_with_contemporaneous_links() {
        let mut contract = TemporalDiscoveryContract::new(
            TemporalAlgorithm::Pcmci,
            variables(2),
            CiTestSpec::named("parcorr"),
            1,
        );
        assert!(contract.validate().is_ok());
        contract.constraints.temporal.min_lag = Lag::CONTEMPORANEOUS;
        assert_eq!(
            contract.validate().unwrap_err().detail(),
            Some("selective_discovery.contemporaneous_links")
        );
        let plus = TemporalDiscoveryContract {
            algorithm: TemporalAlgorithm::PcmciPlus,
            ..contract.clone()
        };
        assert!(plus.validate().is_ok());
    }

    #[test]
    fn frame_depth_follows_the_lag_bound_for_temporal_runs_only() {
        let mut temporal = TemporalDiscoveryContract::new(
            TemporalAlgorithm::Pcmci,
            variables(2),
            CiTestSpec::named("parcorr"),
            1,
        );
        let kernel = KernelPolicy::scalar_only();
        let one = temporal.stage_inputs(&data(1), kernel).unwrap();
        temporal.constraints.temporal.max_lag = Lag::from_raw(2);
        let two = temporal.stage_inputs(&data(1), kernel).unwrap();
        assert_ne!(one.component("frame_depth"), two.component("frame_depth"));
        let static_contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Pc,
            variables(2),
            CiTestSpec::named("parcorr"),
            1,
        );
        let a = static_contract.stage_inputs(&data(1), kernel).unwrap();
        assert_eq!(
            a.component("frame_depth"),
            Some(&component("frame_depth", |w| {
                w.word(0);
            }))
        );
    }
}
