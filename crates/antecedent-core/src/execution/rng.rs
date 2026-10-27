//! Execution context: deterministic RNG streams.
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// RNG stream domain: a tag that keeps indexed stream families apart.
///
/// Indexed families must use [`RngFactory::stream_for`] with a dedicated domain
/// rather than `CONST.wrapping_add(index)` into [`RngFactory::stream`]: additive
/// family offsets collide across domains and, under an additive mixer, across
/// adjacent master seeds.
///
/// The domains the workspace's own crates draw from are the associated
/// constants below, tags `1..=19`. The set is open: a crate outside this
/// workspace names its own domain with [`Self::external`], which accepts only
/// tags from [`Self::EXTERNAL_BASE`] up so it cannot alias a built-in one. The
/// tag is the only input a domain contributes to a derived stream, so the
/// built-in constants derive exactly the streams the closed enum did.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct StreamDomain(u64);

// The built-in domains keep their former variant spellings so no call site changes.
#[allow(
    non_upper_case_globals,
    reason = "the built-in domains keep their former variant spellings so no call site changes"
)]
impl StreamDomain {
    /// First tag available to [`Self::external`]; lower tags are reserved for
    /// the constants below.
    pub const EXTERNAL_BASE: u64 = 1 << 16;

    /// IID / block / cluster / Bayesian bootstrap replicates.
    pub const Resample: Self = Self(1);
    /// Lag-aligned temporal block-bootstrap families.
    pub const TemporalBlock: Self = Self(2);
    /// Bayesian draw / posterior simulation streams.
    pub const Bayesian: Self = Self(3);
    /// Structure-MCMC chains.
    pub const McmcStructure: Self = Self(4);
    /// Order-MCMC chains.
    pub const McmcOrder: Self = Self(5);
    /// DBN posterior MCMC chains.
    pub const McmcDbn: Self = Self(6);
    /// Graph-completion / identified-set posterior streams.
    pub const Completion: Self = Self(7);
    /// Counterfactual abduction / action / prediction noise.
    pub const Counterfactual: Self = Self(8);
    /// Attribution Monte Carlo arms.
    pub const Attribution: Self = Self(9);
    /// Design ranking and allocation draws.
    pub const Design: Self = Self(10);
    /// Statistical transport / retargeting draws.
    pub const Transport: Self = Self(11);
    /// Conditional-independence / CI null Monte Carlo.
    pub const StatsCi: Self = Self(12);
    /// Learner-internal randomness (forest, GBT, neural).
    pub const Learner: Self = Self(13);
    /// Estimator-local streams (AIPW, IV, RD, …).
    pub const Estimate: Self = Self(14);
    /// Temporal / DBN mediation block streams.
    pub const Mediation: Self = Self(15);
    /// Panel / interference path streams.
    pub const Panel: Self = Self(16);
    /// Facade execute-path streams.
    pub const Execute: Self = Self(17);
    /// Unit / integration test fixtures.
    pub const Test: Self = Self(18);
    /// Example programs.
    pub const Example: Self = Self(19);

    /// A domain outside the built-in set.
    ///
    /// # Panics
    /// If `tag` is below [`Self::EXTERNAL_BASE`]; in a `const` item this is a
    /// compile-time error.
    #[must_use]
    pub const fn external(tag: u64) -> Self {
        assert!(
            tag >= Self::EXTERNAL_BASE,
            "external stream domain tag collides with a built-in domain"
        );
        Self(tag)
    }

    /// The tag mixed into every stream this domain derives.
    #[must_use]
    pub const fn tag(self) -> u64 {
        self.0
    }
}

impl std::fmt::Debug for StreamDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self.0 {
            1 => "Resample",
            2 => "TemporalBlock",
            3 => "Bayesian",
            4 => "McmcStructure",
            5 => "McmcOrder",
            6 => "McmcDbn",
            7 => "Completion",
            8 => "Counterfactual",
            9 => "Attribution",
            10 => "Design",
            11 => "Transport",
            12 => "StatsCi",
            13 => "Learner",
            14 => "Estimate",
            15 => "Mediation",
            16 => "Panel",
            17 => "Execute",
            18 => "Test",
            19 => "Example",
            tag => return write!(f, "StreamDomain::external({tag})"),
        };
        f.write_str(name)
    }
}

/// Factory for deterministic RNG streams.
///
/// Streams are derived from a master seed and a stream id via a non-additive
/// mixer so algorithms can request reproducible substreams without a global RNG.
/// Prefer [`Self::stream_for`] so indexed families do not share additive ids.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RngFactory {
    pub(super) master_seed: u64,
}

impl RngFactory {
    /// Create a factory from a master seed.
    #[must_use]
    pub const fn from_seed(master_seed: u64) -> Self {
        Self { master_seed }
    }

    /// Master seed.
    #[must_use]
    pub const fn master_seed(&self) -> u64 {
        self.master_seed
    }

    /// Derive a stream for a raw `stream_id`.
    ///
    /// Prefer [`Self::stream_for`] for new call sites. This entry point remains
    /// for low-level ids that are already domain-unique.
    #[must_use]
    pub fn stream(&self, stream_id: u64) -> CausalRng {
        CausalRng::from_seed(derive_stream(self.master_seed, stream_id))
    }

    /// Derive a stream for `(domain, index)`.
    ///
    /// Domain and index are packed without an additive family offset, then passed
    /// through the single [`derive_stream`] mixer with the master seed.
    #[must_use]
    pub fn stream_for(&self, domain: StreamDomain, index: u64) -> CausalRng {
        // Odd multiplier so domain tags stay distinct under xor with index.
        let stream_id = domain.tag().wrapping_mul(0xD1B5_4A32_D192_ED03) ^ index;
        self.stream(stream_id)
    }
}

/// The `SplitMix64` increment (the golden-ratio gamma).
pub const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The `SplitMix64` output finalizer: a bijective avalanche mix of one word.
#[must_use]
pub const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Advance a `SplitMix64` state by [`GOLDEN_GAMMA`] and return its next output.
///
/// [`CausalRng::next_u64`] is exactly this step on its own state.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN_GAMMA);
    mix64(*state)
}

/// Deterministic SplitMix64-based RNG for library algorithms.
#[derive(Clone, Debug)]
pub struct CausalRng {
    pub(super) state: u64,
}

impl CausalRng {
    /// Create from a 64-bit seed.
    #[must_use]
    pub const fn from_seed(seed: u64) -> Self {
        // The seed is offset by the golden-ratio increment (a bijection on `u64`),
        // so distinct seeds give distinct initial states. SplitMix advances its
        // state by that increment before each output mix, so the all-zero state is
        // not a fixed point of the generator either way; the offset only decorrelates
        // the initial state from the raw seed.
        Self { state: seed ^ GOLDEN_GAMMA }
    }

    /// Restore from a previously exported [`Self::state`].
    #[must_use]
    pub const fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// Opaque stream state for checkpoint / CRN continuation.
    #[must_use]
    pub const fn state(&self) -> u64 {
        self.state
    }

    /// Next `u64` from the stream.
    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        splitmix64(&mut self.state)
    }

    /// Next `f64` in `[0, 1)`.
    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa extraction; precision loss vs full u64 is intentional.
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Non-additive mix of `(master, stream_id)`.
///
/// The previous `master.wrapping_add(stream_id)` form made
/// `(master=s, stream=k)` identical to `(s−δ, k+δ)`, so adjacent user seeds
/// shared almost every bootstrap replicate stream.
///
/// This is the seed [`RngFactory::stream`] hands to [`CausalRng::from_seed`]; it
/// is a different mixer from the [`mix64`] finalizer (it keys two words), so the
/// two are not interchangeable.
#[must_use]
pub const fn derive_stream(master: u64, stream_id: u64) -> u64 {
    let mut z = master.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z ^= stream_id.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 32)).wrapping_mul(0x1656_67B1_9E37_79F9);
    z = (z ^ (z >> 32)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z ^ (z >> 32)
}
